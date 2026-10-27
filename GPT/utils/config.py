"""
Typed config for the GPT pipeline.

Mirrors the structure of `configs/*.yaml`. Loading via `Config.from_yaml(path)`
catches three classes of mistakes at load time:
  1. Missing or unexpected keys            ->  pydantic ValidationError
  2. Wrong type (e.g. vocab_size: "8192")   ->  pydantic ValidationError
  3. Out-of-range values                   ->  @model_validator asserts below
"""
import os
from pathlib import Path

import yaml
from pydantic import model_validator

from BERT.utils.config import PathsConfig      # tokenizer_dir, checkpoint_dir, log_dir — imported unchanged
from transformer.utils.config import _Strict   # extra="forbid", strict=True — imported unchanged


class DataConfig(_Strict):
    source_path: str     # BERT's corpus: one article per line, blank line between
    corpus_path: str     # ours: the same articles, each ending with <|endoftext|>
    vocab_size: int      # 256 bytes + merges + 1 <|endoftext|>

    @model_validator(mode="after")
    def _check(self):
        # train.bin / val.bin store ids as uint16, so every id must be < 2**16.
        # Checked here so a bad value fails now, not after hours of BPE training.
        assert 257 <= self.vocab_size <= 2**16, \
            f"vocab_size must be in [257, 65536], got {self.vocab_size}"
        return self


class Config(_Strict):
    data: DataConfig
    paths: PathsConfig

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        config = cls(**raw)

        # Resolve relative paths against the repo root, not CWD — so
        # `python -m GPT.scripts.prepare_corpus` works from anywhere.
        # parents[2]: utils/ -> GPT/ -> repo root.
        repo_root = Path(__file__).resolve().parents[2]
        for section, fields in ((config.data, ("source_path", "corpus_path")),
                                (config.paths, ("tokenizer_dir", "checkpoint_dir", "log_dir"))):
            for field in fields:
                val = getattr(section, field)
                if not os.path.isabs(val):
                    setattr(section, field, str(repo_root / val))
        return config
