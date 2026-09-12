# African Folktales SLM Challenge — TensorFlow pipeline

`train_slm.py` builds a small **from-scratch causal Transformer** in
TensorFlow/Keras, trains it, and generates 60–120 word folktales for the
Kaggle competition — while also following the project's own governance
documents (`PROBLEM_STATEMENT.txt`, the Data Card, and the Impact
Statement).

## Modeling pipeline (unchanged core logic)

1. **Domain pretraining** (not the fine-tuning step): plain next-token LM
   training on `documents.csv`, so the model learns folktale vocabulary and
   rhythm before it ever sees a prompt.
2. **One fine-tuning technique**: full-parameter supervised fine-tuning on
   the `(prompt → reference_story)` pairs in `train_prompts.csv`, loss-masked
   so only the story span (not the prompt/theme/region prefix) drives
   gradients. No LoRA/adapters/prefix-tuning/RAG stacked on top.
3. **Generation**: word-count-aware, temperature/top-k sampling targeting
   60–120 words, since the Levenshtein metric penalizes longer strings.
4. **Local evaluation**: the pipeline implements the competition's own mean
   character-level Levenshtein metric and reports it on an 80/20 split of
   the 38 training pairs before it ever touches the test set.

## What's new: following the Data Card / Impact Statement / Problem Statement

| Document says... | What the script does about it |
|---|---|
| Data Card — "respect for IP", consent-first pipeline, avoid extractive use of oral histories | `filter_licensed_documents()` drops any document row whose `license` field isn't an open/redistributable license (`CC0-1.0`, `CC-BY-4.0`, `CC-BY-SA-4.0`, public domain) before it ever reaches training, and logs what was excluded and why. |
| Data Card — "purposefully over-sample smaller ethnolinguistic groups to prevent dominant languages ... from overshadowing minority voices" | `oversample_to_balance()` duplicates pretraining windows and fine-tuning pairs from under-represented `culture_region` groups so every region gets equal weight in training, instead of the region with the most documents dominating. |
| Data Card — "Who might be left out of your dataset?" / transparency | `representation_summary()` prints document/pair counts by region and theme up front and explicitly flags anything under-represented *before* the balancing kicks in, so you can see the raw skew in your real data. |
| Impact Statement — risk of "flattening distinct traditions into a generic pan-African trickster tale voice" | Each generated story gets an `own_region_vocab_overlap` vs `other_region_vocab_overlap` check in `provenance_report.csv`. If a story's vocabulary leans more on *other* regions' documents than its own target region's, it's flagged `possible_flattening_flag = True` and summarized as a warning at the end of the run. |
| Impact Statement — "misattribution ... is a form of erasure" / Data Card — "rigorously tagging data origins so end-users know exactly which community a story belongs to" | `provenance_report.csv` records, per generated story: the theme/region it was conditioned on, its closest-matching source document (by Jaccard word overlap) and that document's license, and a heuristic character-archetype tag (e.g. "Trickster Hare"). |
| Impact Statement — the tool risks "quietly substituting for the relationship it's supposed to support" with real storykeepers | Every provenance row carries a fixed `disclaimer` string stating the story is a model generation, not a verbatim community source, and not a substitute for hearing the tradition from its storykeepers. |

**Important scoping note:** the Kaggle `submission.csv` format itself
(`PromptId,Story`) is left exactly as specified — none of this governance
metadata is added to the graded submission file. It all lives in the
separate `provenance_report.csv`, which is the right place for it: useful
for you/reviewers, invisible to the scorer.

## Run it

```bash
pip install tensorflow-cpu   # or tensorflow, if you have a GPU box

python train_slm.py \
  --documents documents.csv \
  --train_prompts train_prompts.csv \
  --test_prompts test_prompts.csv \
  --output submission.csv \
  --provenance_output provenance_report.csv
```

Defaults are tuned for a corpus this size (24 documents, 38 pairs); training
takes well under a minute on CPU. All hyperparameters are CLI flags — see
`build_argparser()`.

## What I tested this on

I still don't have your actual CSVs, so I re-ran the updated pipeline
end-to-end on the same synthetic mock dataset as before (24 docs, 38 train
pairs, 10 test prompts) — the license filter, representation summary,
balanced oversampling, fine-tuning, validation, and provenance report all
run cleanly and produce sensible output (e.g. it correctly flagged 2/10
generated stories as possibly "flattening" into another region's vocabulary
in this run). Swap in your real `documents.csv` / `train_prompts.csv` /
`test_prompts.csv` (including real `license` values) and it should behave
the same way — check the console output's governance section first thing,
since that's where you'll see if your real license/region distribution
triggers any filtering or oversampling you didn't expect.
