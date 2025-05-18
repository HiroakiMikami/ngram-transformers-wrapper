import os


def try_get(key: str) -> str | None:
    return os.environ.get(f"NGRAM_TRANSFORMERS_WRAPPER_{key}")


def get(key: str, defalut_value: str) -> str:
    v = try_get(key)
    return v if v is not None else defalut_value
