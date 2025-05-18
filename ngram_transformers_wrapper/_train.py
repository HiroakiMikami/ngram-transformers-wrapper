import logging
import tempfile
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoTokenizer

from ngram_transformers_wrapper._kenlm_command import build_binary, lmplz
from ngram_transformers_wrapper._registry import register_models
from ngram_transformers_wrapper.modeling_ngram import NgramConfig, NgramForCausalLM

_logger = logging.getLogger(__name__)


def train(
    train_texts: Iterable[str],
    val_texts: Iterable[str],
    tokenizer: AutoTokenizer,
    config: NgramConfig,
) -> NgramForCausalLM:
    register_models()

    with tempfile.TemporaryDirectory() as tmpdir:
        arpa_file = Path(tmpdir) / "model.arpa"
        bin_file = Path(tmpdir) / "model.bin"

        def _train_inputs() -> Iterable[str]:
            for text in train_texts:
                input_ids = tokenizer.encode(text)
                encoded = " ".join([str(int(id)) for id in input_ids])
                yield encoded

        lmplz(
            _train_inputs(),
            arpa_file,
            order=config.order,
            discount_fallback=config.discount_fallback,
            vocab_estimate=config.vocab_size,
        )
        build_binary(arpa_file, bin_file, type="trie")

        assert config.vocab_size == tokenizer.vocab_size
        model = NgramForCausalLM(config)
        with open(bin_file, "rb") as f:
            t = torch.frombuffer(f.read(), dtype=torch.uint8)
            model.set_model_data(t)

    # TODO calculate val score
    losses = []
    for val_text in val_texts:
        input_ids = tokenizer(val_text, return_tensors="pt").input_ids
        losses.append(model(input_ids, labels=input_ids).loss)
    if len(losses) != 0:
        loss = sum(losses) / len(losses)
        _logger.info(f"Val loss: {loss:.3f}")

    return model
