"""
Typed config for the BERT pre-training pipeline.

Mirrors the structure of `configs/*.yaml`. Loading via `Config.from_yaml(path)`
catches three classes of mistakes at load time:
  1. Missing or unexpected keys        ->  pydantic ValidationError
  2. Wrong type (e.g. d_model: "256")  ->  pydantic ValidationError
  3. Cross-field invariants            ->  @model_validator asserts below
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
    d_model: int                    # H — hidden size
    num_heads: int                  # A — attention heads
    d_ff: int                       # intermediate size (paper: 4·H)
    num_layers: int                 # L — encoder layers
    dropout: float
    max_position_embeddings: int    # size of the LEARNED position table (paper: 512)
    num_segments: int               # segment embeddings A/B for NSP (always 2)

    # mode="after" -> runs once all fields are set & type-checked, so we can compare them.
    @model_validator(mode="after")
    def _check(self):
        # MHA splits d_model across heads; non-divisible -> cryptic shape error deep in attention.
        assert self.d_model % self.num_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        assert 0.0 <= self.dropout < 1.0, f"dropout must be in [0, 1), got {self.dropout}"
        return self


class TrainingConfig(_Strict):
    max_seq_len: int
    batch_size: int
    num_epochs: int
    mlm_probability: float
    warmup_steps: int
    clip_grad_norm: float
    val_split: float

    @model_validator(mode="after")
    def _check(self):
        # 15% in the paper; a value outside (0, 1) silently corrupts every masked example.
        assert 0.0 < self.mlm_probability < 1.0, \
            f"mlm_probability must be in (0, 1), got {self.mlm_probability}"
        assert 0.0 <= self.val_split < 1.0, f"val_split must be in [0, 1), got {self.val_split}"
        return self


class OptimizerConfig(_Strict):
    lr: float
    betas: list          # YAML parses [0.9, 0.999] as list; build_optimizer casts to tuple
    eps: float
    weight_decay: float


class DataConfig(_Strict):
    dataset: str
    wiki_dump: str
    corpus_path: str
    max_articles: Optional[int]    # null = use the full dump
    min_chars: int
    vocab_size: int
    num_workers: int


class TokensConfig(_Strict):
    pad_idx: int
    unk_idx: int
    cls_idx: int
    sep_idx: int
    mask_idx: int

    @model_validator(mode="after")
    def _check(self):
        # These mirror SPECIAL_TOKENS order in data_utils.py ([PAD]=0 … [MASK]=4).
        # If two collide, embeddings/loss silently address the wrong row — catch it here.
        ids = [self.pad_idx, self.unk_idx, self.cls_idx, self.sep_idx, self.mask_idx]
        assert len(set(ids)) == len(ids), f"special-token ids must be distinct, got {ids}"
        return self


class PathsConfig(_Strict):
    tokenizer_dir: str
    checkpoint_dir: str
    log_dir: str


class Config(_Strict):
    model: ModelConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    data: DataConfig
    tokens: TokensConfig
    paths: PathsConfig
    seed: int
    device: str

    @model_validator(mode="after")
    def _check(self):
        # Cross-block: every position index 0..max_seq_len-1 must exist in the learned table.
        assert self.training.max_seq_len <= self.model.max_position_embeddings, \
            (f"max_seq_len ({self.training.max_seq_len}) exceeds max_position_embeddings "
             f"({self.model.max_position_embeddings})")
        return self

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        config = cls(**raw)

        # Resolve relative paths against the repo root, not CWD — so
        # `python -m BERT.scripts.prepare_corpus` works from anywhere.
        # parents[2]: utils/ -> BERT/ -> repo root.
        repo_root = Path(__file__).resolve().parents[2]
        for field in ("tokenizer_dir", "checkpoint_dir", "log_dir"):
            val = getattr(config.paths, field)
            if not os.path.isabs(val):
                setattr(config.paths, field, str(repo_root / val))
        # corpus_path lives under data: but is a real file path (prepare_corpus writes it,
        # build_documents reads it) — resolve it the same way.
        if not os.path.isabs(config.data.corpus_path):
            config.data.corpus_path = str(repo_root / config.data.corpus_path)
        return config