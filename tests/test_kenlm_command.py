import os
import tempfile
from pathlib import Path

import kenlm

from ngram_transformers_wrapper._kenlm_command import build_binary, lmplz


def test_build_ngram_lm() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        # prepare input
        arpa_file = os.path.join(tmpdir, "test.arpa")
        bin_file = os.path.join(tmpdir, "test.bin")
        lmplz(["Hello World !!", "A B C D", ""], Path(arpa_file), order=2, discount_fallback=[0.5, 1.0, 1.5])
        build_binary(Path(arpa_file), Path(bin_file))

        model = kenlm.Model(bin_file)
        p0 = model.perplexity("Hello World")
        p1 = model.perplexity("World Hello")
        assert p0 < p1
