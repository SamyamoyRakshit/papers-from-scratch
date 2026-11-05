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
from typing import Literal

import yaml
from pydantic import model_validator

from BERT.utils.config import PathsConfig      # tokenizer_dir, checkpoint_dir, log_dir — imported unchanged
from transformer.utils.config import _Strict   # extra="forbid", strict=True — imported unchanged


class DataConfig(_Strict):
    source_path: str     # BERT's corpus: one article per line, blank line between
    corpus_path: str     # ours: the same articles, each ending with <|endoftext|>
    vocab_size: int      # 256 bytes + merges + 1 <|endoftext|>
    train_path: str      # prepare_data.py writes the train ids here, raw uint16
    val_path: str        # ... and the val ids here
    val_split: float     # fraction of ARTICLES held out for val

    @model_validator(mode="after")
    def _check(self):
        # train.bin / val.bin store ids as uint16, so every id must be < 2**16.
        # Checked here so a bad value fails now, not after hours of BPE training.
        assert 257 <= self.vocab_size <= 2**16, \
            f"vocab_size must be in [257, 65536], got {self.vocab_size}"
        # BERT allows 0; here 0 would write an empty val.bin that evaluate.py can't use.
        assert 0.0 < self.val_split < 1.0, f"val_split must be in (0, 1), got {self.val_split}"
        return self


class ModelConfig(_Strict):
    num_layers: int      # L — decoder blocks
    d_model: int         # H — width of the residual stream
    num_heads: int       # A — attention heads
    d_ff: int            # MLP inner width (paper: 4·H)
    block_size: int      # T — context length AND the size of the learned position table (wpe).
                         #   One field, not two: BERT had max_position_embeddings=512 but trained
                         #   at 128 → 384 × 256 = 98,304 dead parameters. Here they cannot differ.
    dropout: float
    attention_impl: Literal["sdpa", "reference"]   # sdpa: F.scaled_dot_product_attention
                                                   # reference: transformer/'s MHA, to check sdpa against
    init_std: float      # σ of N(0, σ) for every Linear / Embedding weight (GPT-1 §4.1: 0.02);
                         #   gpt.py gives the two c_proj per block σ/√(2·num_layers) (GPT-2 §2.3)

    # mode="after" -> runs once all fields are set & type-checked, so we can compare them.
    @model_validator(mode="after")
    def _check(self):
        # Heads split d_model; non-divisible -> cryptic shape error deep in attention.
        assert self.d_model % self.num_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        assert 0.0 <= self.dropout < 1.0, f"dropout must be in [0, 1), got {self.dropout}"
        # nn.init.normal_ takes std 0 without an error and zeroes every weight.
        assert self.init_std > 0.0, f"init_std must be > 0, got {self.init_std}"
        return self


class Config(_Strict):
    data: DataConfig
    model: ModelConfig
    paths: PathsConfig
    seed: int            # shuffles the articles before the train/val split

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        config = cls(**raw)

        # Resolve relative paths against the repo root, not CWD — so
        # `python -m GPT.scripts.prepare_corpus` works from anywhere.
        # parents[2]: utils/ -> GPT/ -> repo root.
        repo_root = Path(__file__).resolve().parents[2]
        for section, fields in ((config.data, ("source_path", "corpus_path", "train_path", "val_path")),
                                (config.paths, ("tokenizer_dir", "checkpoint_dir", "log_dir"))):
            for field in fields:
                val = getattr(section, field)
                if not os.path.isabs(val):
                    setattr(section, field, str(repo_root / val))
        return config
