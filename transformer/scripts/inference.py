"""
Beam-search translation inference for a trained Transformer checkpoint.

Defaults: beam_size=2, α=0.6. The paper uses beam_size=4 with α=0.6
(Section 6.1) — we drop to k=2 for cheaper inference on small models.
α=0.6 is from Wu et al. 2016 (arXiv:1609.08144), tuned on WMT En→Fr/En→De.

Usage:
    python -m transformer.scripts.inference \
        --config transformer/configs/base.yaml \
        --checkpoint transformer/checkpoints/base/run_<ts>/best.pt \
        --text "How are you?"

If --text is omitted, drops into a REPL — type a sentence, hit enter, get a
translation. Ctrl-C to exit.
"""
import argparse

import torch

from ..utils.config import Config
from ..utils.data_utils import ( 
    load_tokenizer,
    encode as tokenize,
    decode as detokenize
)
from ..utils.mask_utils import ( 
    create_src_mask,
    create_memory_mask,
    create_tgt_mask
)
from ._common import build_model, load_checkpoint
from .train import get_device

@torch.no_grad()
def translate(
    model: torch.nn.Module,
    sp,
    text: str,
    sos_idx: int,
    eos_idx: int,
    pad_idx: int,
    max_seq_len: int,
    device: torch.device,
    beam_size: int = 2,
    alpha: float = 0.6
) -> str:
    """
    Translate `text` via beam search.

    Algorithm:
        1. Encode the source once (`run_encoder_stack`), reuse `memory` for all steps.
        2. Start with one live beam: [<sos>], score = 0.
        3. At each step:
            - Expand every live beam by its top `beam_size` next-token log-probs.
            - From the resulting `beam_size * beam_size` candidates, keep the top
              `beam_size` by cumulative log-prob.
            - Move any beam ending in <eos> to the "finished" pool.
        4. Stop when all beams finish or `max_seq_len` is hit.
        5. Rank finished beams by length-penalized score and return the best.

    Length penalty:
        Score is `sum(log_probs) / length^alpha`. Without it, summing log-probs
        biases toward shorter sequences (each step adds a negative number).
        alpha=0.6 from Wu et al. 2016 (WMT-tuned).

    Args:
        model: Trained Transformer in eval mode.
        sp: SentencePiece processor (shared src/tgt vocab).
        text: Source-language string.
        sos_idx, eos_idx, pad_idx: Special token ids from config.
        max_seq_len: Hard cap on generated length, mirroring training-time
            truncation (`training.max_seq_len`, 128). The model only ever saw
            sequences up to this length during training, so longer outputs are
            out-of-distribution.
        device: Where to run inference.
        beam_size: Beams kept alive per step. Default 2; paper uses 4.
        alpha: Length-penalty exponent. 0.6 from Wu et al. 2016 (WMT-tuned).

    Returns:
        Decoded target string with <sos>/<eos> stripped.
    """
    model.eval()

    # --- Encode source once ---
    # The encoder output (`memory`) is constant across decoding steps, so we
    # run it before the loop and reuse it.
    src_ids = tokenize(sp, text, max_seq_len)
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = create_src_mask(src, pad_idx).to(device)
    memory_mask = create_memory_mask(src, pad_idx).to(device)
    memory = model.run_encoder_stack(src, src_mask)

    # --- Beam search ---
    # Each beam is (cumulative_log_prob, token_ids). Start with a single beam
    # of just <sos>; cumulative log-prob starts at 0.0 (= log(1.0)).
    beams = [(0.0, [sos_idx])]
    completed = []

    for _ in range(max_seq_len):
        # Expand every live beam by its top-k next tokens.
        all_candidates = []
        for score, tokens in beams:
            tgt = torch.tensor([tokens], dtype=torch.long, device=device)
            tgt_mask = create_tgt_mask(tgt, pad_idx)
            logits = model.run_decoder_stack(tgt, memory, tgt_mask, memory_mask)
            # log_softmax over vocab at the LAST position only — that's the
            # next-token distribution. Log keeps argmax intact (monotonic) and
            # lets us sum across steps instead of multiplying tiny floats.
            log_probs = torch.log_softmax(logits[0, -1], dim=-1)
            topk_log_probs, topk_ids = log_probs.topk(beam_size)    # topk returns 2 things - values and indicies; see `torch.topk`'s docstring for more
            for log_prob, next_id in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                all_candidates.append((score + log_prob, tokens + [next_id]))

        # Pick the top-k from ALL candidates pooled together, not one per parent.
        # This lets a strong parent win multiple slots — e.g. if "tumi kemon" and
        # "tumi bhalo" both score better than anything from "tomar", both can survive.
        all_candidates.sort(key=lambda x: x[0], reverse=True)

        beams = []
        for score, tokens in all_candidates:
            if tokens[-1] == eos_idx:
                # Length penalty applied only at completion — comparing
                # finished beams of different lengths fairly. -1 to exclude <sos>.
                length = len(tokens) - 1
                completed.append((score / (length ** alpha), tokens))
            else:
                beams.append((score, tokens))
            if len(beams) == beam_size:
                break
        
        # All survivors ended in <eos> — no live beams left to extend.
        if not beams:
            break

    # Hit max_seq_len with no beam emitting <eos>: score the best live beam
    # with the length penalty so it's comparable to completed ones.
    if not completed:
        score, tokens = beams[0]
        completed.append((score / ((len(tokens) - 1) ** alpha), tokens))

    completed.sort(key=lambda x: x[0], reverse=True)
    out_ids = completed[0][1][1:]                     # drop leading <sos>
    if out_ids and out_ids[-1] == eos_idx:
        out_ids = out_ids[:-1]                        # drop trailing <eos>
    return detokenize(sp, out_ids)
    
def main() -> None:
    # --- CLI args ---
    parser = argparse.ArgumentParser(description="Translate via a trained Transformer")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--text", type=str, default=None,
                        help="Source text to translate. If omitted, drops into REPL.")
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

    def _translate_one(text:str) -> None:
        out = translate(
            model=model,
            sp=sp,
            text=text,
            sos_idx=config.tokens.sos_idx,
            eos_idx=config.tokens.eos_idx,
            pad_idx=config.tokens.pad_idx,
            max_seq_len=config.training.max_seq_len,
            device=device,
            beam_size=config.inference.beam_size,
            alpha=config.inference.length_penalty
        )
        print(f"  {config.data.src_lang}: {text}")
        print(f"  {config.data.tgt_lang}: {out}")

    if args.text is not None:
        _translate_one(args.text)
        return
    
    # --- REPL mode ---
    print("REPL mode — type a sentence, Ctrl-C to exit.")
    try:
        while True:
            text = input(f"{config.data.src_lang}> ").strip()
            if not text:
                continue
            _translate_one(text)
    except(KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()