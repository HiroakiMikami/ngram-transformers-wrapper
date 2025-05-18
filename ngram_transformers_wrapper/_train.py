import logging
import tempfile
from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from ngram_transformers_wrapper._kenlm_command import build_binary, lmplz
from ngram_transformers_wrapper._registry import register_models
from ngram_transformers_wrapper.modeling_ngram import NgramConfig, NgramForCausalLM

_logger = logging.getLogger(__name__)


def train(
    train_texts: Iterable[str],
    val_texts: list[str],
    tokenizer: AutoTokenizer,
    config: NgramConfig,
) -> NgramForCausalLM:
    register_models()

    with tempfile.TemporaryDirectory() as tmpdir:
        input_text = Path(tmpdir) / "input.txt"
        arpa_file = Path(tmpdir) / "model.arpa"
        bin_file = Path(tmpdir) / "model.bin"

        _logger.info("Prepare training texts")
        with open(input_text, "w") as f:
            for text in tqdm(train_texts):
                input_ids = tokenizer.encode(text)
                encoded = " ".join([str(int(id)) for id in input_ids])
                f.write(encoded + "\n")

        _logger.info("Train NGram LM")
        lmplz(
            input_text,
            arpa_file,
            order=config.order,
            discount_fallback=config.discount_fallback,
            vocab_estimate=config.vocab_size,
        )
        build_binary(arpa_file, bin_file, type="trie")

        model = NgramForCausalLM(config)
        with open(bin_file, "rb") as f:
            t = torch.frombuffer(f.read(), dtype=torch.uint8)
            model.set_model_data(t)

    # TODO calculate val score

    return model
