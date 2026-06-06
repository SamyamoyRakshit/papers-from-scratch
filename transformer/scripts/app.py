"""
Local Gradio demo for English → Bengali translation.

Run from the repo root:
    uv add gradio
    uv run python -m transformer.scripts.app
"""
import gradio as gr

from ..utils.config import Config
from ..utils.data_utils import load_tokenizer
from ._common import build_model, load_checkpoint
from .train import get_device
from .inference import translate

CONFIG_PATH = "transformer/configs/base.yaml"
CHECKPOINT_PATH = "transformer/checkpoints/base/run_2026-05-31_18-00-50/best.pt"

# --- Load once at startup ---
config = Config.from_yaml(CONFIG_PATH)
device = get_device(config.device)
sp = load_tokenizer(config.paths.tokenizer_path)

model = build_model(config=config, vocab_size=sp.vocab_size(), device=device)
ckpt = load_checkpoint(path=CHECKPOINT_PATH, device=device)
model.load_state_dict(ckpt["model_state_dict"])
print(f"Loaded {CHECKPOINT_PATH} (epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f})")


def run_translate(text: str, beam_size: int, alpha: float) -> str:
    if not text.strip():
        return ""
    return translate(
        model=model,
        sp=sp,
        text=text,
        sos_idx=config.tokens.sos_idx,
        eos_idx=config.tokens.eos_idx,
        pad_idx=config.tokens.pad_idx,
        max_seq_len=config.training.max_seq_len,
        device=device,
        beam_size=int(beam_size),
        alpha=alpha,
    )


demo = gr.Interface(
    fn=run_translate,
    inputs=[
        gr.Textbox(label="English", placeholder="Type an English sentence…", lines=2),
        gr.Slider(1, 8, value=4, step=1, label="Beam size"),
        gr.Slider(0.0, 1.5, value=1.0, step=0.1, label="Length penalty (α)"),
    ],
    outputs=gr.Textbox(label="Bengali"),
    title="English → Bengali Transformer (from scratch, ~11M params, M1)",
    description="A Transformer rebuilt from 'Attention Is All You Need', trained on a 16GB MacBook. "
                "Best BLEU at beam=4, α=1.0.",
    examples=[
        ["What are you saying?", 4, 1.0],
        ["I do not know.", 4, 1.0],
        ["This is a big deal.", 4, 1.0],
    ],
)

if __name__ == "__main__":
    demo.launch()  # local only: http://127.0.0.1:7860
