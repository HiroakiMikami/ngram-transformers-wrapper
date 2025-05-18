import time
import os
import torch
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from typing import Iterable

import pytest
import datasets
from transformers import AutoTokenizer

from ngram_transformers_wrapper._train import train
from ngram_transformers_wrapper.modeling_ngram import NgramConfig


@pytest.fixture(scope="session")
def dummy_dataset() -> Iterable[datasets.Dataset]:
    ds = datasets.load_dataset("rajpurkar/squad", split="validation")

    def _f(item: dict) -> dict:
        return {
            "text": item["context"] + "\n\n" + "Question: " + item["question"] + "\n\n" + "Answer: " + item["answers"]["text"][0]
        }

    ds = ds.map(_f)
    yield ds


def _generate_dummy_train_data(ds: datasets.Dataset, size: int) -> Iterable[str]:
    n_size = 0
    i = 0
    while n_size < size:
        t = ds[i]["text"]
        yield t
        n_size += len(t.encode())
        i = (i + 1) % len(ds)


@contextmanager
def _measure(tag: str) -> Iterable[None]:
    begin = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"#benchmark: {tag}\t: {end - begin} sec")



@pytest.mark.benchmark
@pytest.mark.parametrize("order", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("size_", [
        ("1KiB", 1024),
        ("1MiB", 1024 * 1024),
        ("10MiB", 10 * 1024 * 1024),
    ])
def test_train_benchmark(dummy_dataset: datasets.Dataset, order: int, size_: tuple[str, int]) -> None:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    size_key, size = size_
    with _measure(f"train-{size_key}-order-{order}"):
        with redirect_stdout(os.devnull), redirect_stderr(os.devnull):
            train(
                _generate_dummy_train_data(dummy_dataset, size),
                [],
                tokenizer,
                NgramConfig(order=order, vocab_size=tokenizer.vocab_size),
            )


@pytest.mark.benchmark
@pytest.mark.parametrize("order", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("length", [1024, 2048, 4096])
def test_eval_benchmark(dummy_dataset: datasets.Dataset, order: int, batch: int, length: int) -> None:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = train(
        _generate_dummy_train_data(dummy_dataset, 1024),
        [],
        tokenizer,
        NgramConfig(order=order, vocab_size=tokenizer.vocab_size),
    )
    dummy_data = torch.randint(0, 10, size=(batch, length))
    with _measure(f"eval-order-{order}-B{batch}-L{length}"):
        model(input_ids=dummy_data)
