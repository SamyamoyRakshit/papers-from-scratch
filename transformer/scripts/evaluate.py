"""
BLEU evaluation for a trained Transformer checkpoint.

Computes corpus-level BLEU on the validation split via beam-search translation.
Uses sacrebleu (the de-facto standard, comparable across papers).

Usage:
    python -m transformer.scripts.evaluate \
        --config transformer/configs/base.yaml \
        --checkpoint transformer/checkpoints/base/run_<ts>/best.pt \
        --max_samples 500

`--max_samples` caps how many val pairs to translate. Full val (50K pairs) takes
hours via beam search; 500-1000 is enough for a stable BLEU estimate.
"""
import argparse

import sacrebleu
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

from ..utils.config import Config
from ..utils.data_utils import load_tokenizer
from ._common import build_model, load_checkpoint
from .inference import translate
from .train import get_device


def main() -> None:
    # Load .env (HF_TOKEN for gated datasets) — mirrors train.py so eval resolves
    # the same dataset access. Samanantar is public, but kept for parity/gated cases.
    load_dotenv()

    parser = argparse.ArgumentParser(description="BLEU evaluation on val split")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--max_samples", type=int, default=500,
                        help="Cap number of val pairs translated (default: 500)")
    parser.add_argument("--show_samples", type=int, default=5,
                        help="Print N sample translations after scoring (default: 5)")
    args = parser.parse_args()

    # --- Load config + device + tokenizer ---
    config = Config.from_yaml(args.config)
    device = get_device(config.device)
    sp = load_tokenizer(config.paths.tokenizer_path)

    # --- Build model + restore weights ---
    model = build_model(config=config, vocab_size=sp.vocab_size(), device=device)
    ckpt = load_checkpoint(path=args.checkpoint, device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(
        f"Loaded {args.checkpoint} (epoch {ckpt['epoch']}, "
        f"val_loss {ckpt['val_loss']:.4f}, "
        f"git_hash {ckpt.get('git_hash', 'unknown')[:7]})"
    )

    # --- Load val split (mirror train.py's slicing exactly) ---
    # Same seed + same max_rows + same val_split = same val pairs as training saw.
    raw_dataset = load_dataset(
        path=config.data.dataset,
        name=config.data.tgt_lang,
        split="train",
    )
    if config.data.max_rows is not None:
        raw_dataset = raw_dataset.shuffle(seed=config.seed).select(
            range(min(config.data.max_rows, len(raw_dataset)))
        )
    split = raw_dataset.train_test_split(test_size=config.training.val_split, seed=config.seed)
    val_raw = split["test"]

    # Cap to max_samples — full val set is too slow under beam search
    n = min(args.max_samples, len(val_raw))
    val_subset = val_raw.select(range(n))
    print(f"Evaluating on {n} val pairs (full val: {len(val_raw)})")

    # --- Translate each src, collect (hyp, ref) ---
    hypotheses = []
    references = []
    for example in tqdm(val_subset, desc="Translating"):
        src_text = example["src"].strip()
        ref_text = example["tgt"].strip()
        hyp_text = translate(
            model=model,
            sp=sp,
            text=src_text,
            sos_idx=config.tokens.sos_idx,
            eos_idx=config.tokens.eos_idx,
            pad_idx=config.tokens.pad_idx,
            max_seq_len=config.training.max_seq_len,
            device=device,
            beam_size=config.inference.beam_size,
            alpha=config.inference.length_penalty,
        )
        hypotheses.append(hyp_text)
        references.append(ref_text)

    # --- Perplexity (intrinsic) ---
    # PPL = exp(cross_entropy_loss). The checkpoint's val_loss is the mean per-token
    # CE over the FULL val set, so this PPL is more stable than the BLEU estimate
    # (which only covers --max_samples). Note: label smoothing inflates this — the
    # paper accepts that tradeoff ("hurts perplexity ... improves BLEU").
    ppl = float(torch.exp(torch.tensor(ckpt["val_loss"])))
    print()
    print(f"=== Perplexity (val): {ppl:.2f} ===")
    print(f"  (from checkpoint val_loss={ckpt['val_loss']:.4f}, full val set)")

    # --- Corpus BLEU ---
    # sacrebleu expects list-of-list for refs (multiple refs per hyp supported).
    # We have only one reference per hyp, so wrap in a single-element outer list.
    bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize="intl")
    print()
    print(f"=== BLEU: {bleu.score:.2f} ===")
    print(f"  ({bleu.bp:.3f} brevity penalty, "
          f"ratio {bleu.sys_len}/{bleu.ref_len} = {bleu.sys_len/bleu.ref_len:.3f})")

    # --- Sample translations ---
    if args.show_samples > 0:
        print()
        print(f"--- Sample translations (first {args.show_samples}) ---")
        for i in range(min(args.show_samples, len(hypotheses))):
            print(f"[{i+1}]")
            print(f"  src: {val_subset[i]['src'].strip()}")
            print(f"  ref: {references[i]}")
            print(f"  hyp: {hypotheses[i]}")


if __name__ == "__main__":
    main()