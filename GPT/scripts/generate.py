"""
Prompt → continuation with a pre-trained GPT (GPT-2 §3: the LM is used as is, no fine-tuning).

Two configs:
  - the model: the config.yaml snapshot beside --checkpoint (same convention as
    BERT's evaluate/inference), so it is rebuilt exactly as it was trained.
  - the sampling: --config, default configs/generate.yaml (temperature, top-k, ...).

An empty --prompt starts from <|endoftext|>, as GPT-2's unconditional samples do:
in train.bin the token after it is always the first token of an article.

Usage (from repo root):
    python -m GPT.scripts.generate --prompt "কলকাতা"
    python -m GPT.scripts.generate --prompt ""
    python -m GPT.scripts.generate --prompt "রবীন্দ্রনাথ ঠাকুর" --config GPT/configs/generate.yaml
"""
import argparse
import logging
import os
from pathlib import Path

import torch

from BERT.utils.train_utils import set_seed
from common.run_utils import get_device, sha256_file
from ..bpe.bpe import BPETokenizer
from ..utils.config import Config
from ..utils.generate_config import GenerateConfig
from ..utils.sampling import generate
from ._common import build_model, load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_model(checkpoint_arg):
    """
    Resolve a best.pt into a ready-to-sample model. Shared with app.py — call once,
    then generate() per prompt.

    Returns:
        model:     GPT on `device`, weights loaded, eval mode.
        tokenizer: the BPE the model was trained with.
        config:    the run's Config (callers need config.model.block_size).
        device:    the resolved torch.device.
    """
    # realpath: the leaderboard best.pt is a symlink; config.yaml sits beside its target.
    ckpt_path = os.path.realpath(checkpoint_arg)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    config_path = os.path.join(os.path.dirname(ckpt_path), "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config snapshot not found beside checkpoint: {config_path}")

    config = Config.from_yaml(config_path)
    device = get_device(config.device)
    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info(f"Using device: {device}")

    # A retrained tokenizer at the same path would decode the ids into the wrong text.
    tokenizer = BPETokenizer.load(config.paths.tokenizer_dir)
    checkpoint = load_checkpoint(ckpt_path, device)
    tokenizer_sha256 = sha256_file(os.path.join(config.paths.tokenizer_dir, "merges.txt"))
    if checkpoint["tokenizer_sha256"] != tokenizer_sha256:
        raise RuntimeError(
            f"Tokenizer mismatch: checkpoint expects {checkpoint['tokenizer_sha256'][:12]}..., "
            f"current is {tokenizer_sha256[:12]}..."
        )

    model = build_model(config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(f"Step {checkpoint['step']} | val loss {checkpoint['val_loss']:.4f}")
    return model, tokenizer, config, device


def main():
    parser = argparse.ArgumentParser(description="Generate text with a pre-trained GPT")
    # Defaults resolved relative to this file, not CWD — so `python -m ...` works from anywhere.
    default_config = Path(__file__).parent.parent / "configs" / "generate.yaml"
    default_ckpt = Path(__file__).parent.parent / "checkpoints" / "base" / "best.pt"
    parser.add_argument("--config", type=str, default=str(default_config),
                        help="sampling config YAML (temperature, top_k, top_p, ...)")
    parser.add_argument("--checkpoint", type=str, default=str(default_ckpt),
                        help="best.pt from a pretrain run (or the leaderboard best.pt symlink)")
    parser.add_argument("--prompt", type=str, default="",
                        help='text to continue; "" = unconditional (starts from <|endoftext|>)')
    args = parser.parse_args()

    gen_config = GenerateConfig.from_yaml(args.config)
    model, tokenizer, config, device = load_model(args.checkpoint)
    logger.info(f"Sampling: {gen_config.model_dump()}")
    if gen_config.seed is not None:
        set_seed(gen_config.seed)

    prompt_ids = tokenizer.encode(args.prompt) if args.prompt else [tokenizer.eot_id]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)   # (1, T)
    logger.info(f"Prompt: {len(prompt_ids)} tokens")

    for i in range(gen_config.num_samples):
        out = generate(
            model, idx,
            max_new_tokens=gen_config.max_new_tokens,
            block_size=config.model.block_size,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            eot_id=tokenizer.eot_id,
        )
        # Drop the <|endoftext|> start of an unconditional sample; decode prompt + continuation.
        ids = [t for t in out[0].tolist() if t != tokenizer.eot_id]
        print(f"\n----- sample {i + 1} ({out.size(1) - len(prompt_ids)} new tokens) -----")
        print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()
