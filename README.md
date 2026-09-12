# African Folktales SLM

A small, from-scratch Transformer language model (TensorFlow/Keras) fine-tuned to generate short African folktales from thematic prompts — built for the [African Folktales SLM Challenge](https://www.kaggle.com/competitions/african-folktales-slm-challenge).

## What this does

Given a prompt like *"Tell a moral market tale about returning a lost cowrie shell"* plus a `theme` and `culture_region`, the model generates a short (~10–65 word) folktale in a comparable narrative voice, scored against a hidden reference story by mean character-level Levenshtein distance (lower is better).

- **Model**: a small causal Transformer built from scratch (~0.9M params) — no external pretrained weights.
- **Training**: two stages —
  1. *Domain pretraining* on the 24-document folktale corpus (plain next-token language modeling; not the fine-tuning step).
  2. **One fine-tuning technique**: full-parameter supervised fine-tuning on the 38 `(prompt → reference_story)` pairs, with the loss masked to only the story span. No LoRA/adapters/RAG stacked on top.
- **Generation**: temperature/top-k sampling with a length target calibrated per-prompt from the real training distribution (the corpus is bimodal, ~9–64 words, not the ~60–120 words suggested in the brief).
- **Result**: ~196 mean Levenshtein distance on a held-out validation split, vs. the ≈210.3 TF-IDF retrieval baseline.

## Responsible-AI features

This project also implements the safeguards described in its own `Data Card` and `Impact Statement`:

- **License filtering** — only documents with an open/redistributable license are used for training; anything else is dropped and logged.
- **Balanced regional representation** — under-represented `culture_region` groups (e.g. `diaspora`) are oversampled during both pretraining and fine-tuning so dominant regions don't drown them out.
- **Transparency report** — every generated story is accompanied by a `provenance_report.csv` entry: its closest matching source document and license, a heuristic character-archetype tag, and a same-region vs. other-region vocabulary check that flags possible "flattening" into a generic pan-African voice.
- **Disclaimer by design** — generated stories are explicitly labeled as model output, not a substitute for community storykeepers.

None of this touches the Kaggle submission format itself — `submission.csv` stays exactly `PromptId,Story`.

## Project structure

```
├── train_slm.py              # main pipeline: data → pretrain → fine-tune → generate → evaluate
├── train_slm.ipynb           # same pipeline as a Jupyter notebook, cell-by-cell
├── documents.csv             # 24-document folktale corpus (provided by the competition)
├── train_prompts.csv         # 38 (prompt, theme, region, reference_story) pairs
├── test_prompts.csv          # 10 held-out prompts to generate stories for
├── submission.csv            # generated output: PromptId, Story
├── provenance_report.csv     # transparency/attribution metadata (not part of scoring)
└── README.md
```

## Usage

```bash
pip install tensorflow-cpu   # or tensorflow, if you have a GPU

python train_slm.py \
  --documents documents.csv \
  --train_prompts train_prompts.csv \
  --test_prompts test_prompts.csv \
  --output submission.csv \
  --provenance_output provenance_report.csv
```

Useful flags:

| Flag | Default | Purpose |
|---|---|---|
| `--pretrain_epochs` / `--finetune_epochs` | 25 | training length for each stage |
| `--temperature` / `--top_k` | 0.8 / 30 | sampling randomness during generation |
| `--verbose_validation` | off | print each held-out prediction next to its reference |
| `--d_model`, `--num_layers`, `--num_heads`, `--ff_dim` | 160 / 3 / 4 / 384 | model size |

Full hyperparameter list: `python train_slm.py --help`.

## Notes

- Trains in well under a minute on CPU — the corpus is intentionally small (synthetic v1).
- The console output's "Governance checks" section reports any license exclusions and region/theme imbalance in your data — check it first if you swap in a different dataset.
- `train_slm.ipynb` mirrors the script exactly, split into labeled cells, for interactive experimentation.

## License

Corpus: CC0-1.0 (synthetic v1). Code: add your preferred license here.
