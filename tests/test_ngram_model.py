import tempfile
from pathlib import Path
from typing import Iterable

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ngram_transformers_wrapper._registry import register_models
from ngram_transformers_wrapper._train import train
from ngram_transformers_wrapper.modeling_ngram import NgramConfig, NgramForCausalLM


def test_save_pretrained_and_from_pretrained() -> None:
    register_models()

    with tempfile.TemporaryDirectory() as tmpdir:
        model = NgramForCausalLM(NgramConfig(order=2, discount_fallback=[0.5, 1.0, 1.5]))
        model.set_model_data(torch.randint(0, 255, size=(1024,)).to(torch.uint8))
        model.save_pretrained(Path(tmpdir) / "model")

        model2 = AutoModelForCausalLM.from_pretrained(Path(tmpdir) / "model")
        torch.testing.assert_close(model.model_bin, model2.model_bin)


@pytest.fixture(scope="session")
def dummy_model() -> Iterable[tuple[AutoTokenizer, NgramForCausalLM]]:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = train(
        ["Hello World !!", "A B C D E F"],
        ["Hello World", "hello world"],
        tokenizer,
        NgramConfig(vocab_size=tokenizer.vocab_size),
    )
    yield tokenizer, model


def test_forward(dummy_model: tuple[AutoTokenizer, NgramForCausalLM]) -> None:
    tokenizer, model = dummy_model
    expected_tokens = tokenizer.encode("Hello World !!")
    input_ids = tokenizer(["Hello World"], return_tensors="pt").input_ids
    logits = model(input_ids=input_ids).logits
    prob = torch.softmax(logits, dim=-1)
    for i, token in enumerate(expected_tokens[1:]):
        assert prob[0, i, int(token)] > 0.5, (i, int(token), tokenizer.decode(int(token)))


def test_bached_forward(dummy_model: tuple[AutoTokenizer, NgramForCausalLM]) -> None:
    tokenizer, model = dummy_model
    texts = ["Hello World !!", "A B C D E"]
    expected_l = []
    for text in texts:
        input_ids = tokenizer([text], return_tensors="pt").input_ids
        expected_l.append(model(input_ids).logits[0])
    actual = model(**tokenizer(texts, return_tensors="pt", padding=True)).logits
    torch.testing.assert_close(actual[0, 2:], expected_l[0])
    torch.testing.assert_close(actual[1, 0:], expected_l[1])


def test_use_cache(dummy_model: tuple[AutoTokenizer, NgramForCausalLM]) -> None:
    tokenizer, model = dummy_model
    expected = model(**tokenizer(["A B C D E"], return_tensors="pt")).logits

    interm = model(**tokenizer(["A B C"], return_tensors="pt"), use_cache=True)
    actual_0 = interm.logits
    actual_1 = model(
        **tokenizer([" D E"], return_tensors="pt", add_special_tokens=False),
        use_cache=True,
        past_key_values=interm.past_key_values,
    ).logits
    actual = torch.cat([actual_0, actual_1], dim=1)
    expected_m = torch.max(expected, dim=-1)
    actual_m = torch.max(actual, dim=-1)
    for i in range(5):
        if i != 2:
            torch.testing.assert_close(actual[:, i], expected[:, i])
    torch.testing.assert_close(actual_m.values, expected_m.values)
    torch.testing.assert_close(actual_m.indices, expected_m.indices)


def test_labels(dummy_model: tuple[AutoTokenizer, NgramForCausalLM]) -> None:
    tokenizer, model = dummy_model
    torch.manual_seed(42)
    input_ids = tokenizer(["A B C D E F"], return_tensors="pt").input_ids
    model_inputs = input_ids[:, :-1]
    labels = input_ids[:, 1:]

    model.train()
    logits = model(input_ids=model_inputs).logits
    expected = torch.nn.functional.cross_entropy(logits[0], labels[0])
    actual = model(input_ids=input_ids, labels=input_ids).loss
    torch.testing.assert_close(actual, expected)


def test_logits_to_keep(dummy_model: tuple[AutoTokenizer, NgramForCausalLM]) -> None:
    tokenizer, model = dummy_model
    torch.manual_seed(42)
    input_ids = tokenizer(["A B C D E"], return_tensors="pt").input_ids

    model.train()
    expected = model(input_ids=input_ids)
    outputs = model(input_ids=input_ids, logits_to_keep=torch.tensor([-1]))
    torch.testing.assert_close(expected.logits[:, torch.tensor([-1]), :], outputs.logits)


def test_generate(dummy_model: tuple[AutoTokenizer, NgramForCausalLM]) -> None:
    tokenizer, model = dummy_model
    input_ids = tokenizer(["A B"], return_tensors="pt").input_ids

    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=2,
        num_beams=1,
        do_sample=False,
        temperature=1.0,
        min_length=1,
    )[0]
    assert tokenizer.decode(out) == "A B C D"


def test_batch_generate(dummy_model: tuple[AutoTokenizer, NgramForCausalLM]) -> None:
    tokenizer, model = dummy_model
    text0 = "A B C D E"
    text1 = "A B"
    batch = tokenizer([text0, text1], return_tensors="pt", padding=True)

    expected0 = model.generate(
        **{k: v[:1] for k, v in batch.items()}, max_new_tokens=2, num_beams=1, do_sample=False, temperature=0.0
    )
    expected1 = model.generate(
        **{k: v[1:, -2:] for k, v in batch.items()}, max_new_tokens=2, num_beams=1, do_sample=False, temperature=0.0
    )
    actual = model.generate(**batch, max_new_tokens=2, num_beams=1, do_sample=False, temperature=0.0)
    torch.testing.assert_close(expected0[0], actual[0])
    torch.testing.assert_close(expected1[0], actual[1][-4:])
