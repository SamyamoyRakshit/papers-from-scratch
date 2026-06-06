"""
Centralized logging setup for the transformer pipeline.

Call setup_logging(log_dir) once at the start of a run; modules then use
`logger = logging.getLogger(__name__)` to log without re-configuring.
"""
import logging
import os
import sys


def setup_logging(log_dir: str, level: str = "INFO") -> None:
    """
    Configure the root logger with a console handler (stdout) and a file
    handler writing to {log_dir}/train.log.

    Console writes to stdout, not stderr — tqdm uses stderr for its progress
    bar, so keeping them on separate streams avoids interleaving.

    Idempotent: clears prior handlers so re-runs (notebooks, tests) don't
    accumulate duplicate handlers and double-log every line.

    Args:
        log_dir: Directory for train.log; created if missing.
        level: Root log level (e.g. "INFO", "DEBUG").
    """
    os.makedirs(log_dir, exist_ok=True)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    for h in list(root.handlers):
        root.removeHandler(h)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = logging.FileHandler(os.path.join(log_dir, "train.log"))
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # Quiet down noisy third-party loggers — httpx logs every HTTP request from
    # huggingface_hub/datasets at INFO, which floods our pipeline logs.
    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
