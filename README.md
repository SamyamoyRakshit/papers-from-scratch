# papers-from-scratch

Paper/architecture replications of SOTA AI/ML models from scratch using PyTorch on Bengali
data — public text corpora for BERT and the transformer, a custom hand-collected image set
for ViT — everything trained on a single Apple M1 with 16 GB of RAM.

| Replication | Paper | Task / data | Headline |
|---|---|---|---|
| [**BERT**](BERT/) | Devlin et al., 2019 | Pre-trained on Bengali Wikipedia, fine-tuned on IndicGLUE `sna.bn` news topics | **86.5% test accuracy** — above published mBERT (80.23) and IndicBERT (78.45), ~1 pt under XLM-R, at ~7.5M params |
| [**transformer**](transformer/) | Vaswani et al., 2017 | English → Bengali translation (AI4Bharat Samanantar) | complete from-scratch seq2seq — attention, Noam schedule, beam search — with the small-scale BLEU reported honestly |
| [**ViT**](ViT/) | Dosovitskiy et al., 2021 | Bengali temple image classification (hand-collected dataset) | "An Image is Worth 16x16 Words", from scratch |

No `nn.Transformer`, no HuggingFace models — every module is hand-written (and BERT reuses the
transformer replication's attention/LayerNorm, the way the paper itself does). Each folder has
its own README with results, setup, and usage; BERT and the transformer also carry a `docs/`
set (getting started, training, architecture), while ViT lives in a single showcase notebook.

## Weights

Trained weights are **not published yet**. To keep the repo lean (no large binaries in git
history), all checkpoints are gitignored. They will be **released on Hugging Face** later —
links will be added here once published.

## Author

[**Samyamoy Rakshit**](https://github.com/SamyamoyRakshit)

## Citation

The replicated papers are cited in each folder's README ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) ·
[Devlin et al., 2019](https://arxiv.org/abs/1810.04805) · [Dosovitskiy et al., 2021](https://arxiv.org/abs/2010.11929)).
If this repository's code or write-ups are useful to you, cite it as:

```bibtex
@misc{rakshit2026replicating,
  author       = {Rakshit, Samyamoy},
  title        = {papers-from-scratch: SOTA architectures rebuilt from scratch
                  in PyTorch on Bengali datasets},
  year         = {2026},
  howpublished = {\url{https://github.com/SamyamoyRakshit/papers-from-scratch}}
}
```
