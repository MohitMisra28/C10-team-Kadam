
import argparse
import csv
import random
import re
from collections import Counter, defaultdict

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

ALLOWED_LICENSES = {
    "cc0-1.0", "cc0", "cc-by-4.0", "cc-by", "cc-by-sa-4.0", "cc-by-sa",
    "public domain", "pd",
}

DISCLAIMER = (
    "Model-generated story conditioned on the given theme/region. "
    "This is a synthetic retelling, not a verbatim community source, and is "
    "not a substitute for hearing this tradition from its storykeepers."
)


ARCHETYPE_KEYWORDS = {
    "spider": "Trickster (spider-type, e.g. Anansi)",
    "hare": "Trickster Hare",
    "tortoise": "Trickster/Wisdom Tortoise",
    "hyena": "Foolish Antagonist",
    "elephant": "Powerful/Wise Figure",
    "weaverbird": "Community/Craft Figure",
    "cricket": "Small but Clever Figure",
    "monkey": "Trickster Monkey",
}


def normalize_license(lic: str) -> str:
    return re.sub(r"\s+", " ", (lic or "").strip().lower())


def filter_licensed_documents(documents):
    kept, dropped = [], []
    for d in documents:
        if normalize_license(d.get("license", "")) in ALLOWED_LICENSES:
            kept.append(d)
        else:
            dropped.append(d)
    if dropped:
        print(f"[license filter] excluding {len(dropped)} document(s) with "
              f"non-open licenses: "
              f"{sorted(set(d.get('license', '?') for d in dropped))}")
    print(f"[license filter] {len(kept)}/{len(documents)} documents retained for training")
    return kept


def representation_summary(rows, key, label):
    counts = Counter(r.get(key, "unknown") for r in rows)
    if not counts:
        return counts
    max_count = max(counts.values())
    print(f"[representation] {label} counts: {dict(counts)}")
    for k, c in counts.items():
        if c < 0.5 * max_count:
            print(f"  [under-represented] {label}='{k}' has {c} example(s) "
                  f"vs max {max_count} — will be oversampled during training")
    return counts


def detect_archetypes(text):
    text_l = text.lower()
    hits = [label for kw, label in ARCHETYPE_KEYWORDS.items() if kw in text_l]
    return hits


def jaccard(a_tokens, b_tokens):
    a, b = set(a_tokens), set(b_tokens)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# 1. Tokenizer (simple, dependency-free, full control over special tokens)
# ---------------------------------------------------------------------------

PAD, UNK, BOS, SEP, EOS = "<pad>", "<unk>", "<bos>", "<sep>", "<eos>"
SPECIALS = [PAD, UNK, BOS, SEP, EOS]

_word_re = re.compile(r"[a-z0-9']+")


def tokenize(text: str):
    return _word_re.findall(text.lower())


class Vocab:
    def __init__(self, texts, max_size=6000, min_freq=1):
        counter = Counter()
        for t in texts:
            counter.update(tokenize(t))
        most_common = [w for w, c in counter.most_common(max_size) if c >= min_freq]
        self.itos = SPECIALS + [w for w in most_common if w not in SPECIALS]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD]
        self.unk_id = self.stoi[UNK]
        self.bos_id = self.stoi[BOS]
        self.sep_id = self.stoi[SEP]
        self.eos_id = self.stoi[EOS]

    def __len__(self):
        return len(self.itos)

    def encode(self, text):
        return [self.stoi.get(w, self.unk_id) for w in tokenize(text)]

    def decode(self, ids):
        words = []
        for i in ids:
            if i in (self.pad_id, self.bos_id, self.sep_id):
                continue
            if i == self.eos_id:
                break
            words.append(self.itos[i] if i < len(self.itos) else UNK)
        return " ".join(words)


class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(d_model),
        ])
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn_out = self.att(x, x, use_causal_mask=True)
        x = self.ln1(x + self.drop1(attn_out, training=training))
        ffn_out = self.ffn(x)
        x = self.ln2(x + self.drop2(ffn_out, training=training))
        return x


def build_model(vocab_size, max_len, d_model=160, num_heads=4, ff_dim=384, num_layers=3, dropout=0.1):
    inputs = layers.Input(shape=(max_len,), dtype="int32")
    tok_emb = layers.Embedding(vocab_size, d_model, name="tok_emb")(inputs)
    positions = tf.range(start=0, limit=max_len, delta=1)
    pos_emb = layers.Embedding(max_len, d_model, name="pos_emb")(positions)
    x = tok_emb + pos_emb
    x = layers.Dropout(dropout)(x)
    for i in range(num_layers):
        x = TransformerBlock(d_model, num_heads, ff_dim, dropout, name=f"block_{i}")(x)
    logits = layers.Dense(vocab_size, name="lm_head")(x)
    return tf.keras.Model(inputs, logits, name="folktale_slm")



def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def oversample_to_balance(rows, key):

    groups = defaultdict(list)
    for r in rows:
        groups[r.get(key, "unknown")].append(r)
    max_count = max(len(v) for v in groups.values())
    balanced = []
    for g, items in groups.items():
        reps = max_count // len(items)
        remainder = max_count % len(items)
        balanced.extend(items * reps)
        balanced.extend(random.sample(items, remainder) if remainder else [])
    random.shuffle(balanced)
    return balanced


def make_pretrain_windows(vocab, documents, max_len):

    windows = []  
    step = max_len // 2
    for d in documents:
        ids = [vocab.bos_id] + vocab.encode(d["text"]) + [vocab.eos_id]
        if len(ids) < 8:
            continue
        for start in range(0, max(1, len(ids) - 1), step):
            window = ids[start:start + max_len + 1]
            if len(window) < 8:
                continue
            window = window + [vocab.pad_id] * (max_len + 1 - len(window))
            windows.append({"tokens": window, "culture_region": d.get("culture_region", "unknown")})
    balanced = oversample_to_balance(windows, "culture_region")
    seqs = np.array([w["tokens"] for w in balanced], dtype="int32")
    return seqs


def make_finetune_example(vocab, theme, region, prompt, story, max_len):
    prefix = vocab.encode(f"{theme} {region} {prompt}")
    story_ids = vocab.encode(story)
    seq = [vocab.bos_id] + prefix + [vocab.sep_id] + story_ids + [vocab.eos_id]
    seq = seq[:max_len + 1]
    loss_mask_start = 1 + len(prefix) + 1  
    pad_len = max_len + 1 - len(seq)
    seq = seq + [vocab.pad_id] * pad_len
    x = seq[:-1]
    y = seq[1:]
    weights = np.zeros(max_len, dtype="float32")
    for i in range(max_len):
        target_pos = i + 1
        if loss_mask_start <= target_pos and y[i] != vocab.pad_id:
            weights[i] = 1.0
    return np.array(x, dtype="int32"), np.array(y, dtype="int32"), weights


def build_finetune_dataset(vocab, rows, max_len, balance_by="culture_region"):
    rows = oversample_to_balance(rows, balance_by) if balance_by else rows
    xs, ys, ws = [], [], []
    for r in rows:
        x, y, w = make_finetune_example(
            vocab, r["theme"], r["culture_region"], r["prompt"], r["reference_story"], max_len
        )
        xs.append(x); ys.append(y); ws.append(w)
    return np.stack(xs), np.stack(ys), np.stack(ws)




def generate_story(model, vocab, theme, region, prompt, max_len,
                    min_words=60, max_words=115, temperature=0.8, top_k=30):
    prefix = [vocab.bos_id] + vocab.encode(f"{theme} {region} {prompt}") + [vocab.sep_id]
    generated = list(prefix)
    story_words = 0

    for _ in range(max_len - len(prefix)):
        window = generated[-max_len:]
        pad_needed = max_len - len(window)
        model_input = np.array([[vocab.pad_id] * pad_needed + window], dtype="int32")
        logits = model(model_input, training=False).numpy()[0, -1]

        logits[vocab.pad_id] = -1e9
        logits[vocab.bos_id] = -1e9
        logits[vocab.sep_id] = -1e9
        if story_words < min_words:
            logits[vocab.eos_id] = -1e9

        logits = logits / max(temperature, 1e-5)
        top_ids = np.argpartition(logits, -top_k)[-top_k:]
        top_logits = logits[top_ids]
        probs = tf.nn.softmax(top_logits).numpy()
        next_id = int(np.random.choice(top_ids, p=probs))

        generated.append(next_id)
        if next_id == vocab.eos_id:
            break
        if next_id not in (vocab.pad_id, vocab.bos_id, vocab.sep_id):
            story_words += 1
        if story_words >= max_words:
            generated.append(vocab.eos_id)
            break

    story_ids = generated[len(prefix):]
    return vocab.decode(story_ids)




def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[lb]




def build_provenance_row(prompt_row, story, documents):
    theme = prompt_row.get("theme", "")
    region = prompt_row.get("culture_region", "")
    story_tokens = tokenize(story)

    same_region_docs = [d for d in documents if d.get("culture_region") == region] or documents
    other_region_docs = [d for d in documents if d.get("culture_region") != region]

    best_doc, best_sim = None, 0.0
    for d in same_region_docs:
        sim = jaccard(story_tokens, tokenize(d["text"]))
        if sim > best_sim:
            best_doc, best_sim = d, sim

    own_region_vocab = set(t for d in same_region_docs for t in tokenize(d["text"]))
    other_region_vocab = set(t for d in other_region_docs for t in tokenize(d["text"]))
    own_overlap = jaccard(story_tokens, list(own_region_vocab))
    other_overlap = jaccard(story_tokens, list(other_region_vocab))
    possible_flattening = other_overlap > own_overlap and other_region_docs

    return {
        "PromptId": prompt_row["PromptId"],
        "theme": theme,
        "culture_region": region,
        "closest_document_id": best_doc["document_id"] if best_doc else "",
        "closest_document_similarity": round(best_sim, 3),
        "closest_document_license": best_doc.get("license", "") if best_doc else "",
        "archetypes_detected": "; ".join(detect_archetypes(story)) or "none",
        "own_region_vocab_overlap": round(own_overlap, 3),
        "other_region_vocab_overlap": round(other_overlap, 3),
        "possible_flattening_flag": bool(possible_flattening),
        "disclaimer": DISCLAIMER,
    }




def main(args):
    documents_raw = read_csv(args.documents)
    train_rows = read_csv(args.train_prompts)
    test_rows = read_csv(args.test_prompts)

    print("=== Governance checks (Data Card / Impact Statement) ===")
    documents = filter_licensed_documents(documents_raw)
    representation_summary(documents, "culture_region", "documents by region")
    representation_summary(documents, "theme", "documents by theme")
    representation_summary(train_rows, "culture_region", "train pairs by region")
    print("=========================================================")

    rng = random.Random(SEED)
    shuffled = train_rows[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(0.2 * len(shuffled)))
    val_rows, fit_rows = shuffled[:n_val], shuffled[n_val:]

    corpus_texts = (
        [d["text"] for d in documents]
        + [r["prompt"] for r in train_rows]
        + [r["reference_story"] for r in train_rows]
        + [r["prompt"] for r in test_rows]
    )
    vocab = Vocab(corpus_texts, max_size=args.vocab_size)
    print(f"Vocab size: {len(vocab)}")

    model = build_model(len(vocab), args.max_len, d_model=args.d_model,
                         num_heads=args.num_heads, ff_dim=args.ff_dim,
                         num_layers=args.num_layers, dropout=args.dropout)
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.pretrain_lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

    
    pre_seqs = make_pretrain_windows(vocab, documents, args.max_len)
    pre_x, pre_y = pre_seqs[:, :-1], pre_seqs[:, 1:]
    print(f"Pretraining sequences (region-balanced): {pre_x.shape}")

    def lm_loss(y_true, y_pred, mask_pad=True):
        loss = loss_fn(y_true, y_pred)
        if mask_pad:
            mask = tf.cast(tf.not_equal(y_true, vocab.pad_id), loss.dtype)
            loss = loss * mask
            return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(mask), 1.0)
        return tf.reduce_mean(loss)

    model.compile(optimizer=optimizer, loss=lm_loss)
    if len(pre_x) > 0:
        model.fit(pre_x, pre_y, batch_size=args.batch_size, epochs=args.pretrain_epochs, verbose=2)


    fit_x, fit_y, fit_w = build_finetune_dataset(vocab, fit_rows, args.max_len)
    print(f"Fine-tuning examples (region-balanced): {fit_x.shape}")

    ft_optimizer = tf.keras.optimizers.Adam(learning_rate=args.finetune_lr)

    @tf.function
    def train_step(x, y, w):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            per_tok_loss = loss_fn(y, logits)
            loss = tf.reduce_sum(per_tok_loss * w) / tf.maximum(tf.reduce_sum(w), 1.0)
        grads = tape.gradient(loss, model.trainable_variables)
        ft_optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    n = len(fit_x)
    for epoch in range(args.finetune_epochs):
        idx = np.random.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, args.batch_size):
            batch_idx = idx[start:start + args.batch_size]
            loss = train_step(
                tf.constant(fit_x[batch_idx]),
                tf.constant(fit_y[batch_idx]),
                tf.constant(fit_w[batch_idx]),
            )
            epoch_loss += float(loss) * len(batch_idx)
        print(f"[fine-tune] epoch {epoch + 1}/{args.finetune_epochs} loss={epoch_loss / n:.4f}")

    
    if val_rows:
        distances = []
        for r in val_rows:
            pred = generate_story(model, vocab, r["theme"], r["culture_region"], r["prompt"], args.max_len)
            distances.append(levenshtein(pred, r["reference_story"]))
        print(f"Validation mean Levenshtein distance: {np.mean(distances):.1f} "
              f"(retrieval baseline ~= 210.3)")

   
    all_x, all_y, all_w = build_finetune_dataset(vocab, train_rows, args.max_len)
    n_all = len(all_x)
    for epoch in range(args.finetune_epochs):
        idx = np.random.permutation(n_all)
        for start in range(0, n_all, args.batch_size):
            batch_idx = idx[start:start + args.batch_size]
            train_step(
                tf.constant(all_x[batch_idx]),
                tf.constant(all_y[batch_idx]),
                tf.constant(all_w[batch_idx]),
            )

   
    provenance_rows = []
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["PromptId", "Story"])
        for r in test_rows:
            story = generate_story(model, vocab, r["theme"], r["culture_region"], r["prompt"], args.max_len)
            writer.writerow([r["PromptId"], story])
            provenance_rows.append(build_provenance_row(r, story, documents))
    print(f"Wrote {args.output}")

    flattening_flags = sum(1 for p in provenance_rows if p["possible_flattening_flag"])
    if flattening_flags:
        print(f"[provenance] WARNING: {flattening_flags}/{len(provenance_rows)} generated "
              f"stories overlap more with OTHER regions' vocabulary than their own — "
              f"possible flattening into a generic voice. See provenance_report.csv.")

    with open(args.provenance_output, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(provenance_rows[0].keys()) if provenance_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(provenance_rows)
    print(f"Wrote {args.provenance_output}")


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--documents", default="documents.csv")
    p.add_argument("--train_prompts", default="train_prompts.csv")
    p.add_argument("--test_prompts", default="test_prompts.csv")
    p.add_argument("--output", default="submission.csv")
    p.add_argument("--provenance_output", default="provenance_report.csv")
    p.add_argument("--vocab_size", type=int, default=4000)
    p.add_argument("--max_len", type=int, default=160)
    p.add_argument("--d_model", type=int, default=160)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--ff_dim", type=int, default=384)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--pretrain_lr", type=float, default=3e-4)
    p.add_argument("--pretrain_epochs", type=int, default=25)
    p.add_argument("--finetune_lr", type=float, default=2e-4)
    p.add_argument("--finetune_epochs", type=int, default=25)
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
