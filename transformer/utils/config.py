"""
Typed config for the Transformer training pipeline.

Mirrors the structure of `configs/*.yaml`. Loading via `Config.from_yaml(path)`
catches three classes of mistakes at load time:
  1. Missing or unexpected keys      ->  pydantic ValidationError
  2. Wrong type (e.g. d_model: "256") -> pydantic ValidationError
  3. Cross-field invariants          ->  @model_validator asserts below
"""
import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, model_validator


class _Strict(BaseModel):
    # Shared base so every nested config below inherits the same strictness — DRY.
    # extra="forbid":  typo like `d_modle: 256` raises instead of being silently ignored.
    # strict=True:     no type coercion — `d_model: "256"` (string) raises instead of becoming 256.
    model_config = ConfigDict(extra="forbid", strict=True)


class ModelConfig(_Strict):
    d_model: int
    num_heads: int
    d_ff: int
    num_layers: int
    dropout: float
    max_len: int

    # mode="after" -> runs once all fields are set & type-checked, so we can compare them.
    # (mode="before" would run on the raw dict, before any field validation.)
    @model_validator(mode="after")
    def _check(self):
        # MHA splits d_model across heads; non-divisible -> cryptic shape error deep in attention
        assert self.d_model % self.num_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        assert 0.0 <= self.dropout < 1.0, f"dropout must be in [0, 1), got {self.dropout}"
        return self


class TrainingConfig(_Strict):
    max_tokens_per_batch: int
    max_seq_len: int
    num_epochs: int
    label_smoothing: float
    warmup_steps: int
    clip_grad_norm: float
    val_split: float


class OptimizerConfig(_Strict):
    betas: list     # YAML parses [0.9, 0.98] as list; build_optimizer casts to tuple
    eps: float


class DataConfig(_Strict):
    src_lang: str
    tgt_lang: str
    dataset: str
    max_rows: Optional[int]
    vocab_size: int
    num_workers: int
    filter_max_ratio: float
    filter_min_words: int


class TokensConfig(_Strict):
    pad_idx: int
    sos_idx: int
    eos_idx: int
    unk_idx: int


class PathsConfig(_Strict):
    checkpoint_dir: str
    log_dir: str
    tokenizer_path: str


class Config(_Strict):
    model: ModelConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    data: DataConfig
    tokens: TokensConfig
    paths: PathsConfig
    seed: int
    device: str

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        config = cls(**raw)

        # Resolve relative paths against the repo root, not CWD — so
        # `python -m transformer.scripts.train` works from anywhere.
        # parents[2]: utils/ -> transformer/ -> repo root.
        repo_root = Path(__file__).resolve().parents[2]
        for field in ("checkpoint_dir", "log_dir", "tokenizer_path"):
            val = getattr(config.paths, field)
            if not os.path.isabs(val):
                setattr(config.paths, field, str(repo_root / val))
        return config
