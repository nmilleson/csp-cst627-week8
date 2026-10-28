"""Post-hoc audit of every saved prediction file.

1. Train/test overlap: exact (after normalisation) and near-duplicate (TF-IDF cosine) test items.
   Accuracy is re-reported on the clean subset so overlap cannot inflate the headline.
2. Slices: accuracy by query length and per intent class, averaged over seeds.

This cannot detect pretraining contamination (Qwen2.5's corpus is not published); the README
discusses that separately.
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from data import ROOT, load_config, load_splits

LENGTH_BUCKETS = [(1, 5, "1-5"), (6, 10, "6-10"), (11, 20, "11-20"), (21, 10**6, "21+")]


def normalise(text):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def length_bucket(text):
    n = len(text.split())
    return next(name for lo, hi, name in LENGTH_BUCKETS if lo <= n <= hi)


def overlap(splits, threshold):
    train_text, test_text = splits["train"]["text"], splits["test"]["text"]
    train_norm = {normalise(t) for t in train_text}
    exact = [normalise(t) in train_norm for t in test_text]

    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(train_text)
    sims = (vec.transform(test_text) @ vec.transform(train_text).T).toarray()
    nearest = sims.argmax(axis=1)
    max_cos = sims[np.arange(len(test_text)), nearest]
    rows = [{"idx": i, "text": t, "exact_dup": exact[i], "max_cosine": round(float(max_cos[i]), 4),
             "near_dup": bool(exact[i] or max_cos[i] >= threshold), "nearest_train_text": train_text[nearest[i]]}
            for i, t in enumerate(test_text)]
    return rows


def load_predictions(pred_dir):
    """{model_key: [(seed, preds_by_idx), ...]} where model_key is e.g. 'fp16_r8' or 'tfidf_logreg'."""
    models = defaultdict(list)
    for path in sorted(pred_dir.glob("*.csv")):
        m = re.fullmatch(r"(fp16|nf4)_r(\d+)_s(\d+)", path.stem)
        key, seed = (f"{m[1]}_r{m[2]}", int(m[3])) if m else (path.stem, 0)
        with open(path, newline="", encoding="utf-8") as f:
            preds = {int(r["idx"]): int(r["pred"]) for r in csv.DictReader(f)}
        models[key].append((seed, preds))
    return models


def accuracy(preds, labels, idx):
    idx = [i for i in idx if i in preds]  # smoke runs only predict a prefix of the test set
    return float(np.mean([preds[i] == labels[i] for i in idx])) if idx else float("nan")


def contrasts(keys):
    """The comparisons the README makes claims about, as (a, b) pairs meaning 'a minus b'."""
    pairs = []
    for prec in ("fp16", "nf4"):
        pairs += [(f"{prec}_r{hi}", f"{prec}_r4") for hi in (8, 16, 32)]
    for rank in (4, 8, 16, 32):
        pairs.append((f"nf4_r{rank}", f"fp16_r{rank}"))
    pairs += [(k, "tfidf_logreg") for k in ("fp16_r4", "nf4_r4")]
    return [(a, b) for a, b in pairs if a in keys and b in keys]


def paired_bootstrap(models, labels, n_boot=10_000, seed=0):
    """Paired difference in accuracy between configurations on the *same* test items.

    Each item's correctness is first averaged over seeds, then items are resampled with
    replacement. The CI therefore reflects test-set sampling only; seed spread is reported
    separately as the SD in summary.md.
    """
    common = sorted(set.intersection(*(set(p) for runs in models.values() for _, p in runs)))
    y = np.array([labels[i] for i in common])
    correct = {k: np.mean([np.array([p[i] for i in common]) == y for _, p in runs], axis=0)
               for k, runs in models.items()}
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, len(common), size=(n_boot, len(common)), dtype=np.int32)
    rows = []
    for a, b in contrasts(correct):
        d = correct[a] - correct[b]
        boots = np.concatenate([d[chunk].mean(axis=1) for chunk in np.array_split(boot_idx, 10)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"a": a, "b": b, "diff_pp": round(100 * d.mean(), 3), "ci95_low_pp": round(100 * lo, 3),
                     "ci95_high_pp": round(100 * hi, 3), "excludes_zero": bool(lo > 0 or hi < 0)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    out = results_dir / "audit"
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    splits = load_splits(cfg)
    labels, names, test_text = splits["test"]["label"], splits["labels"], splits["test"]["text"]

    overlap_rows = overlap(splits, cfg["near_duplicate_cosine"])
    with open(out / "test_overlap.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(overlap_rows[0]))
        writer.writeheader()
        writer.writerows(overlap_rows)
    dup_idx = [r["idx"] for r in overlap_rows if r["near_dup"]]
    clean_idx = [r["idx"] for r in overlap_rows if not r["near_dup"]]

    bucket_of = [length_bucket(t) for t in test_text]
    models = load_predictions(results_dir / "predictions")
    slice_rows, class_rows, overlap_summary = [], [], {}
    for key, runs in models.items():
        all_acc = [accuracy(p, labels, range(len(labels))) for _, p in runs]
        clean_acc = [accuracy(p, labels, clean_idx) for _, p in runs]
        dup_acc = [accuracy(p, labels, dup_idx) for _, p in runs]
        overlap_summary[key] = {"n_seeds": len(runs), "acc_all": float(np.mean(all_acc)),
                                "acc_clean": float(np.mean(clean_acc)), "acc_overlap": float(np.mean(dup_acc))}
        for _, _, bucket in LENGTH_BUCKETS:
            idx = [i for i, b in enumerate(bucket_of) if b == bucket]
            accs = [accuracy(p, labels, idx) for _, p in runs]
            slice_rows.append({"model": key, "slice": "words " + bucket, "n": len(idx),
                               "acc_mean": round(float(np.mean(accs)), 4), "acc_std": round(float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0, 4)})
        for c, name in enumerate(names):
            idx = [i for i, y in enumerate(labels) if y == c]
            class_rows.append({"model": key, "intent": name, "n": len(idx),
                               "acc_mean": round(float(np.mean([accuracy(p, labels, idx) for _, p in runs])), 4)})

    paired_rows = paired_bootstrap(models, labels)
    for path, rows in ((out / "slices.csv", slice_rows), (out / "per_class.csv", class_rows),
                       (out / "paired_differences.csv", paired_rows)):
        if not rows:
            continue
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    lora_keys = [k for k in overlap_summary if k != "tfidf_logreg"]
    best = max(lora_keys, key=lambda k: overlap_summary[k]["acc_all"]) if lora_keys else None
    worst = sorted((r for r in class_rows if r["model"] == best), key=lambda r: r["acc_mean"])[:5]
    summary = {
        "n_test": len(labels),
        "exact_duplicates_in_train": sum(r["exact_dup"] for r in overlap_rows),
        "near_duplicates_in_train": len(dup_idx),
        "near_duplicate_cosine_threshold": cfg["near_duplicate_cosine"],
        "accuracy_by_overlap": overlap_summary,
        "best_lora_config": best,
        "best_config_worst_intents": worst,
        "paired_differences": paired_rows,
    }
    with open(out / "summary.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
