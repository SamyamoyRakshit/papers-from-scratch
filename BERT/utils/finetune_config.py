"""
Typed config for BERT fine-tuning (single-sentence classification).

Separate from the pre-training Config — different shape (points at a pre-trained
checkpoint + a downstream dataset instead of a raw corpus). Reuses config.py's
_Strict base so the same extra="forbid"/strict rules apply.
"""
import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import model_validator

from .config import _Strict


class PretrainedConfig(_Strict):
    checkpoint: str      # best.pt from a pre-training run
    config: str          # that run's config.yaml snapshot — rebuilds the encoder to match


class FinetuneDataConfig(_Strict):
    dataset_id: str
    subset: Optional[str]          # HF config name (e.g. "sna.bn"); null if the dataset has none
    text_field: str
    label_field: str
    num_labels: Optional[int]      # null → inferred from the train split


class FinetuneTrainingConfig(_Strict):
    max_seq_len: int
    batch_size: int
    num_epochs: int
    warmup_ratio: float
    clip_grad_norm: float
    # Default False so old run snapshots (config.yaml without this key) still parse
    # when evaluate.py / inference.py reload them.
    class_weighting: bool = False

    @model_validator(mode="after")
    def _check(self):
        assert 0.0 <= self.warmup_ratio < 1.0, \
            f"warmup_ratio must be in [0, 1), got {self.warmup_ratio}"
        return self


class FinetuneOptimizerConfig(_Strict):
    lr: float
    betas: list          # cast to tuple in build_optimizer
    eps: float
    weight_decay: float


class FinetunePathsConfig(_Strict):
    tokenizer_dir: str
    checkpoint_dir: str
    log_dir: str


class FinetuneConfig(_Strict):
    pretrained: PretrainedConfig
    data: FinetuneDataConfig
    training: FinetuneTrainingConfig
    optimizer: FinetuneOptimizerConfig
    paths: FinetunePathsConfig
    seed: int
    device: str

    @classmethod
    def from_yaml(cls, path: str) -> "FinetuneConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        config = cls(**raw)

        # Resolve every filesystem path against the repo root, not CWD — so
        # `python -m BERT.scripts.finetune` works from anywhere. parents[2]: utils/ -> BERT/ -> root.
        repo_root = Path(__file__).resolve().parents[2]
        for obj, field in [
            (config.pretrained, "checkpoint"),
            (config.pretrained, "config"),
            (config.paths, "tokenizer_dir"),
            (config.paths, "checkpoint_dir"),
            (config.paths, "log_dir"),
        ]:
            val = getattr(obj, field)
            if not os.path.isabs(val):
                setattr(obj, field, str(repo_root / val))
        return config