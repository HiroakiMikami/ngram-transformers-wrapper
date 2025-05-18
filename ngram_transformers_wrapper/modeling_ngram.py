import os
import re
import tempfile
import warnings
from typing import Any, Dict, List, Optional

import kenlm
import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import load_state_dict


class NgramConfig(PretrainedConfig):  # type: ignore
    model_type: str = "ngram"

    def __init__(
        self,
        order: int = 2,
        vocab_size: int = 32000,
        discount_fallback: list[float] | None = None,
        use_cache: bool = True,
        always_compute_actual_logits: bool = False,
        **kwargs: Any,
    ) -> None:
        self.order = order
        self.vocab_size = vocab_size
        self.discount_fallback = discount_fallback if discount_fallback is not None else [0.5, 1.0, 1.5]
        self.always_compute_actual_logits = always_compute_actual_logits

        super().__init__(use_cache=use_cache, **kwargs)


class NgramForCausalLM(PreTrainedModel):  # type: ignore
    config_class = NgramConfig

    def __init__(self, config: NgramConfig) -> None:
        super().__init__(config)
        self.config = config
        self._model: kenlm.Model | None = None
        self.register_buffer("model_bin", torch.zeros(0, dtype=torch.uint8))

    def set_model_data(self, data: torch.Tensor) -> None:
        self._model = None
        assert data.dtype == torch.uint8
        self.register_buffer("model_bin", data)

    def forward(  # type: ignore
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[kenlm.State]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        self._load_model()

        assert input_ids is not None
        if attention_mask is not None:
            L = input_ids.shape[1]
            attention_mask = attention_mask[:, -L:]
            assert input_ids.shape == attention_mask.shape, (input_ids.shape, attention_mask.shape)
        assert position_ids is None
        assert inputs_embeds is None
        if use_cache is None:
            use_cache = False
        if output_attentions is None:
            output_attentions = False
        assert not output_attentions
        if output_hidden_states is None:
            output_hidden_states = False
        assert not output_hidden_states
        assert cache_position is None

        if past_key_values is None:
            past_key_values = [self._initialize_state() for _ in range(len(input_ids))]

        logits_list = []
        for i, input_id_i in enumerate(input_ids):
            logits_i = []
            for j, word_j in enumerate([w.item() for w in input_id_i] + [None]):
                out_state: kenlm.State | None
                if word_j is not None and attention_mask is not None and int(attention_mask[i, j]) == 0:
                    # padded
                    logit = torch.zeros(self.config.vocab_size)
                    out_state = None
                else:
                    logit, out_state = self._compute_logits(past_key_values[i], word_j)
                logits_i.append(logit)
                if out_state is not None:
                    past_key_values[i] = out_state
            logits_i = logits_i[1:]
            logits_list.append(torch.stack(logits_i, dim=0))
        logits = torch.stack(logits_list, dim=0)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = logits[:, slice_indices, : self.config.vocab_size]

        loss = None
        if labels is not None:
            if len(kwargs) > 0 and set(kwargs.keys()) != set(["ignore_index"]):
                warnings.warn(
                    f"The following kwargs may not be supported: {', '.join(kwargs.keys())}. ",
                    stacklevel=2,
                )
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[list[kenlm.State]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        assert inputs_embeds is None

        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
        return model_inputs

    def _compute_logits(self, state: kenlm.State, token: int | None) -> tuple[torch.Tensor, kenlm.State | None]:
        self._load_model()
        assert self._model is not None

        require_actual_logits = token is None
        if self.config.always_compute_actual_logits:
            require_actual_logits = True

        out_state: kenlm.State | None = None
        if require_actual_logits:
            logprobs = torch.empty(self.config.vocab_size)
            probs_l: list[float | None] = [None for _ in range(self.config.vocab_size)]
            for i in range(self.config.vocab_size):
                state_i = kenlm.State()
                score = self._model.BaseFullScore(state, str(i), state_i)
                l10prob = score.log_prob
                oov = score.oov
                if not oov:
                    probs_l[i] = 10.0**l10prob
                if token == i:
                    out_state = state_i
            n_oov = sum([prob is None for prob in probs_l])
            p_total = sum(prob or 0.0 for prob in probs_l)
            remain = (1.0 - p_total) / n_oov
            for i in range(self.config.vocab_size):
                if probs_l[i] is None:
                    probs_l[i] = remain
            probs = torch.tensor(probs_l)
        else:
            assert token is not None
            out_state = kenlm.State()
            l10p = float(self._model.BaseScore(state, str(token), out_state))
            p = 10**l10p
            remain = (1.0 - p) / (self.config.vocab_size - 1)
            probs = torch.full(size=(self.config.vocab_size,), fill_value=remain)
            probs[token] = p
        logprobs = torch.log(probs)
        logit = logprobs - torch.max(logprobs)
        return logit, out_state

    def _initialize_state(self) -> kenlm.State:
        self._load_model()
        state = kenlm.State()
        assert self._model is not None
        self._model.BeginSentenceWrite(state)
        return state

    def _load_model(self) -> None:
        if self._model is not None:
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            bin_file = os.path.join(tmpdir, "model.bin")
            with open(bin_file, "wb") as f:
                f.write(self.model_bin.numpy().tobytes())
            self._model = kenlm.Model(bin_file)

    @classmethod
    def _load_pretrained_model(
        cls,
        model: "NgramForCausalLM",
        state_dict: Optional[Dict],
        checkpoint_files: Optional[List[str]],
        pretrained_model_name_or_path: Optional[str],
        ignore_mismatched_sizes: bool = False,
        sharded_metadata: Optional[Dict] = None,
        device_map: Optional[Dict] = None,
        disk_offload_folder: Optional[str] = None,
        offload_state_dict: Optional[bool] = None,
        dtype: Optional[torch.dtype] = None,
        hf_quantizer: Optional[Any] = None,
        keep_in_fp32_regex: Optional[re.Pattern] = None,
        device_mesh: Optional["torch.distributed.device_mesh.DeviceMesh"] = None,
        key_mapping: Optional[Dict[str, str]] = None,
        weights_only: bool = True,
    ) -> tuple["NgramForCausalLM", Any, Any, Any, Any, Any]:
        state_dict = state_dict.copy() if state_dict is not None else {}
        if checkpoint_files is not None:
            for shard_file in checkpoint_files:
                if shard_file == "":
                    continue
                state_dict_i = load_state_dict(
                    shard_file, is_quantized=False, map_location="cpu", weights_only=weights_only
                )
                state_dict.update(state_dict_i)
        assert set(state_dict.keys()) == set(["model_bin"])
        model_bin = state_dict["model_bin"]
        model.set_model_data(model_bin)

        return model, None, None, None, None, None

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")
