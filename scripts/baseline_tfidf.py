"""CPU reference point: TF-IDF + logistic regression on the same split.

If LoRA does not clearly beat this, the GPU spend is not justified for this task. The
regularisation strength C is chosen on the validation split only; test is touched once.
"""
import argparse
import csv
import json
import time
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from data import ROOT, load_config, load_splits

C_GRID = [1.0, 10.0, 100.0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    (results_dir / "predictions").mkdir(parents=True, exist_ok=True)

    splits = load_splits(load_config())
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True)
    x_train = vec.fit_transform(splits["train"]["text"])
    x_val = vec.transform(splits["val"]["text"])
    x_test = vec.transform(splits["test"]["text"])

    val_acc = {}
    for c in C_GRID:
        clf = LogisticRegression(C=c, max_iter=3000).fit(x_train, splits["train"]["label"])
        val_acc[c] = accuracy_score(splits["val"]["label"], clf.predict(x_val))
    best_c = max(val_acc, key=val_acc.get)

    start = time.perf_counter()
    clf = LogisticRegression(C=best_c, max_iter=3000).fit(x_train, splits["train"]["label"])
    fit_seconds = time.perf_counter() - start
    test_y = splits["test"]["label"]
    test_pred = clf.predict(x_test).tolist()

    result = {
        "model": "tfidf_logreg",
        "ngram_range": [1, 2],
        "C": best_c,
        "val_acc_by_C": {str(k): v for k, v in val_acc.items()},
        "test_acc": accuracy_score(test_y, test_pred),
        "test_macro_f1": f1_score(test_y, test_pred, average="macro"),
        "fit_seconds_cpu": round(fit_seconds, 2),
        "n_train": x_train.shape[0],
        "n_test": len(test_y),
    }
    with open(results_dir / "baseline.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2)
    with open(results_dir / "predictions" / "tfidf_logreg.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "label", "pred"])
        writer.writerows(zip(range(len(test_y)), test_y, test_pred))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
