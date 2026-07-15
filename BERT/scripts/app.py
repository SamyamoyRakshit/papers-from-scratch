"""
Local Gradio demo for Bengali news-topic classification.

Run from the repo root:
    python -m BERT.scripts.app
"""
import gradio as gr

from .inference import encode_text, load_finetuned_classifier, predict

CHECKPOINT_PATH = "BERT/checkpoints/finetune/sna_bn/best.pt"   # leaderboard symlink → winning run

# --- Load once at startup (inference.py pays this per sentence; the app pays it once) ---
model, tokenizer, label_names, config, device = load_finetuned_classifier(CHECKPOINT_PATH)


def classify(text: str) -> dict:
    if not text.strip():
        return {}
    input_ids, token_type_ids = encode_text(text, tokenizer, config.training.max_seq_len)
    probs = predict(model, input_ids, token_type_ids, device)
    return {name: float(p) for name, p in zip(label_names, probs)}   # gr.Label draws these as bars


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(label="Bengali news text", placeholder="এখানে বাংলা ভাষায় খবর লিখুন…", lines=3),
    outputs=gr.Label(label="Predicted topic"),
    title="Bengali News Topic Classifier (from-scratch BERT, 7.5M params, mps)",
    description="BERT rebuilt from Devlin et al. 2019, pretrained on Bengali Wikipedia, "
                "fine-tuned on IndicGLUE sna.bn (Soham articles). Held-out test accuracy: 86.5%.",
    examples=[
        "কলকাতায় আজ বৃষ্টি হবে",
        "মোহনবাগান আজ ডার্বি জিতেছে",
    ],
)

if __name__ == "__main__":
    demo.launch()  # local only: http://127.0.0.1:7860
