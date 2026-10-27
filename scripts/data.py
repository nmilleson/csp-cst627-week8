"""Banking77 download, integrity check, and the fixed train/val/test split.

Every script imports its splits from here so the split is defined in exactly one place.
"""
import csv
import hashlib
import json
import urllib.request
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "ablation.json"
RAW_DIR = ROOT / "data" / "raw"


def load_config(path=CONFIG_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(cfg):
    """Fetch the pinned CSVs once and refuse to continue if the bytes differ from the recorded hash."""
    ds = cfg["dataset"]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, meta in ds["files"].items():
        dest = RAW_DIR / meta["path"]
        if not dest.exists():
            urllib.request.urlretrieve(ds["base_url"] + meta["path"], dest)
        digest = _sha256(dest)
        if digest != meta["sha256"]:
            raise RuntimeError(f"{dest} sha256 {digest} does not match pinned {meta['sha256']}")
        paths[split] = dest
    return paths


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r["text"] for r in rows], [r["category"] for r in rows]


def load_splits(cfg):
    """Return {"labels", "train", "val", "test"}; each split is {"text": [...], "label": [...]}.

    Validation is a stratified slice of the official training file, fixed by split_seed so it
    is identical for every run regardless of the training seed. The official test file is
    only used for final reporting.
    """
    paths = download(cfg)
    train_text, train_cat = _read(paths["train"])
    test_text, test_cat = _read(paths["test"])
    labels = sorted(set(train_cat))
    to_id = {c: i for i, c in enumerate(labels)}

    tr_text, va_text, tr_cat, va_cat = train_test_split(
        train_text, train_cat,
        test_size=cfg["val_fraction"], random_state=cfg["split_seed"], stratify=train_cat,
    )
    return {
        "labels": labels,
        "train": {"text": tr_text, "label": [to_id[c] for c in tr_cat]},
        "val": {"text": va_text, "label": [to_id[c] for c in va_cat]},
        "test": {"text": test_text, "label": [to_id[c] for c in test_cat]},
    }
