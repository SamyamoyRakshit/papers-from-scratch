"""
Generic run/experiment helpers shared across replications.

Nothing model- or paper-specific here — device resolution, file hashing for
checkpoint provenance, git-hash capture, logging setup. BERT imports these; the
transformer keeps its own copies (already shipped — left untouched on purpose).
"""
import hashlib
import json
import logging
import os
import subprocess

import torch

logger = logging.getLogger(__name__)

def get_device(device_config: str) -> torch.device:
    """Resolve the config device string ('auto'|'mps'|'cuda'|'cpu') to a torch.device."""
    if device_config == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_config)


def get_git_hash() -> str:
    """
    Current git commit hash for run provenance, or 'unknown' if not in a repo.

    Appends '-dirty' if the working tree has uncommitted changes — otherwise a
    clean-looking hash would lie about which code produced the run.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return f"{commit}-dirty" if dirty else commit
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def sha256_file(path: str) -> str:
    """SHA-256 of a file's bytes — pins a tokenizer / corpus to a checkpoint."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def setup_logging(log_dir: str) -> None:
    """Log to console + {log_dir}/train.log so every run keeps its own log file."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "train.log")),
            logging.StreamHandler(),
        ],
    )


def update_leaderboard(parent_dir: str, run_name: str, score: float,
                       higher_is_better: bool = False) -> None:
    """
    Record this run's best score in {parent_dir}/leaderboard.json and repoint the
    {parent_dir}/best.pt symlink at the global best across all runs.

    parent_dir holds run_<timestamp>/ subdirs; run_name is the basename of the
    current run dir. Symlink target is relative ("run_X/best.pt") so the parent
    dir stays portable if moved.

    higher_is_better picks the sort direction — the only thing that differs
    between metrics: pre-training ranks val_loss ascending (default, lower wins),
    fine-tuning ranks val_acc descending (higher wins).
    """
    leaderboard_path = os.path.join(parent_dir, "leaderboard.json")
    board: dict[str, float] = {}
    if os.path.exists(leaderboard_path):
        with open(leaderboard_path) as f:
            board = json.load(f)

    board[run_name] = score
    # Sort so the file reads top-down as a ranking (best run first).
    board = dict(sorted(board.items(), key=lambda kv: kv[1], reverse=higher_is_better))
    with open(leaderboard_path, "w") as f:
        json.dump(board, f, indent=2)

    best_run = next(iter(board))
    symlink = os.path.join(parent_dir, "best.pt")
    target = os.path.join(best_run, "best.pt")  # relative -> portable across moves
    if os.path.islink(symlink) or os.path.exists(symlink):
        os.unlink(symlink)
    os.symlink(target, symlink)


def warn_if_config_diverges(snapshot: dict, current: dict, safe_keys: set[str]) -> None:
    """
    Warn for config fields that changed between a checkpoint's snapshot and the
    current run. Pure dict-diff — the caller loads both configs (each replication
    has its own Config class) and supplies its own safe_keys.
    """
    risky: list[str] = []

    def walk(a, b, prefix: str = "") -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            # sorted() so warnings appear in stable, alphabetical order across runs.
            for k in sorted(set(a) | set(b)):
                walk(a.get(k), b.get(k), f"{prefix}.{k}" if prefix else k)
        elif a != b and prefix not in safe_keys:
            risky.append(f"  {prefix}: {a!r} -> {b!r}")

    walk(snapshot, current)
    if risky:
        logger.warning("resumed config differs from checkpoint snapshot:")
        for r in risky:
            logger.warning(r)
        logger.warning("  (continuing — trajectory may differ from the original run)")
