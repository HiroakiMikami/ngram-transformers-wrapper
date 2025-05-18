import logging
import multiprocessing
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Iterable

from ngram_transformers_wrapper import _envvar

_KENLM_REPOSITORY_URL = "https://github.com/kpu/kenlm.git"

_logger = logging.getLogger(__name__)

_initialized = False


def _initialize() -> None:
    global _initialized

    if _initialized:
        return

    git_command = _envvar.get("GIT_COMMAND", "git")
    cmake_command = _envvar.get("CMAKE_COMMAND", "cmake")
    make_command = _envvar.get("CMAKE_COMMAND", "make")
    cache_dir = _envvar.get("CACHE_DIR", str(Path.home() / ".cache" / "ngram_transformer_wrapper"))
    num_job = _envvar.get("MAX_BUILD_JOB", str(multiprocessing.cpu_count()))

    suffix = uuid.uuid4().hex

    dst_dir = Path(cache_dir) / "kenlm"
    dst_path = dst_dir / "bin" / "query"
    dst_dir_tmp = Path(cache_dir) / f"kenlm_{suffix}"
    if not dst_path.exists():
        _logger.info("Download and build kenlm from GitHub")
        subprocess.run(
            [git_command, "clone", _KENLM_REPOSITORY_URL, "--depth", "1", dst_dir_tmp],
            check=True,
        )
        subprocess.run([cmake_command, "."], cwd=dst_dir_tmp, check=True)
        subprocess.run([make_command, "-j", str(num_job)], cwd=dst_dir_tmp, check=True)
        os.rename(dst_dir_tmp, dst_dir)
    _initialized = True


def lmplz(
    inputs: Iterable[str],
    output_path: Path,
    order: int,
    skip_symbols: bool = False,
    interpolate_unigram: int = 1,
    minimum_block: str = "8K",
    sort_block: str = "64M",
    block_count: int = 2,
    vocab_estimate: int = 1000000,
    vocab_pad: int = 0,
    renumber: bool = False,
    collapse_values: bool = False,
    prune: int = 0,
    limit_vocab_file: str | None = None,
    discount_fallback: list[float] | None = None,
) -> None:
    _initialize()

    cache_dir = _envvar.get("CACHE_DIR", str(Path.home() / ".cache" / "ngram_transformer_wrapper"))
    args: list[str] = [
        "--order",
        str(order),
        "--interpolate_unigram",
        str(interpolate_unigram),
        "--minimum_block",
        minimum_block,
        "--sort_block",
        sort_block,
        "--block_count",
        str(block_count),
        "--vocab_estimate",
        str(vocab_estimate),
        "--vocab_pad",
        str(vocab_pad),
        "--prune",
        str(prune),
        "--arpa",
        str(output_path),
    ]
    if skip_symbols:
        args.append("--skip_symbols")
    if renumber:
        args.append("--renumber")
    if collapse_values:
        args.append("--collapse_values")
    if limit_vocab_file is not None:
        args.extend(["--limit_vocab_file", limit_vocab_file])
    if discount_fallback is not None:
        args.extend(["--discount_fallback"] + [str(x) for x in discount_fallback])
    with tempfile.TemporaryDirectory() as tmpdir:
        p = subprocess.Popen(
            [str(Path(cache_dir) / "kenlm" / "bin" / "lmplz")] + args + ["--temp_prefix", tmpdir],
            stdin=subprocess.PIPE,
        )
        for text in inputs:
            p.stdin.write((text + "\n").encode())
        p.stdin.close()
        p.wait()
        assert p.returncode == 0


def build_binary(
    input_path: Path,
    output_path: Path,
    unk_log10_probability: float = -100,
    verbose: bool = False,
    type: str = "probing",
) -> None:
    _initialize()

    cache_dir = _envvar.get("CACHE_DIR", str(Path.home() / ".cache" / "ngram_transformer_wrapper"))
    args: list[str] = ["-u", str(unk_log10_probability)]
    if verbose:
        args.append("-v")
    subprocess.run(
        [str(Path(cache_dir) / "kenlm" / "bin" / "build_binary")] + args + [type, str(input_path), str(output_path)],
        check=True,
    )
