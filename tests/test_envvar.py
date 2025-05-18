import pytest

from ngram_transformers_wrapper._envvar import get, try_get


def test_get(monkeypatch: pytest.MonkeyPatch) -> None:
    assert try_get("FOO") is None
    assert get("FOO", "default") == "default"
    monkeypatch.setenv("NGRAM_TRANSFORMERS_WRAPPER_FOO", "1")
    assert try_get("FOO") == "1"
    assert get("FOO", "default") == "1"
