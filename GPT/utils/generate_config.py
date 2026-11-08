"""
Typed config for GPT sampling (configs/generate.yaml → scripts/generate.py).

Separate from the pre-training Config, two reasons:
  1. generate.py rebuilds the model from the config.yaml snapshot frozen beside
     best.pt. A new required section in Config would make every finished run's
     snapshot fail validation ("Field required") — the model could not be loaded.
  2. These knobs only change how tokens are drawn; the trained model is the same.
     Same split as BERT's finetune_config.py.
"""
import yaml
from pydantic import model_validator

from transformer.utils.config import _Strict   # extra="forbid", strict=True — imported unchanged


class GenerateConfig(_Strict):
    max_new_tokens: int      # tokens to add at most
    temperature: float       # logits / temperature; 0 = greedy (argmax)
    top_k: int | None        # keep the k most likely ids; None = off
    top_p: float | None      # nucleus: smallest set with probability ≥ p; None = off
    num_samples: int         # independent continuations of the same prompt
    seed: int | None         # an int → the same samples every run; None → different each run

    @model_validator(mode="after")
    def _check(self):
        # temperature < 0 flips the ranking: the LEAST likely token becomes the most likely.
        assert self.temperature >= 0.0, f"temperature must be ≥ 0 (0 = greedy), got {self.temperature}"
        # top_k 0 would keep nothing; "off" is null.
        assert self.top_k is None or self.top_k >= 1, f"top_k must be ≥ 1 or null, got {self.top_k}"
        assert self.top_p is None or 0.0 < self.top_p <= 1.0, f"top_p must be in (0, 1] or null, got {self.top_p}"
        return self

    @classmethod
    def from_yaml(cls, path: str) -> "GenerateConfig":
        with open(path, "r") as f:
            return cls(**yaml.safe_load(f))
