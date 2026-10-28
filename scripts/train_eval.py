"""Train and evaluate one LoRA configuration on Banking77, then append one row to runs.csv.

One configuration = (precision, rank, seed). run_ablation.py calls this once per configuration in
a fresh process so peak-memory readings are not polluted by the previous run.
"""
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          BitsAndBytesConfig, get_linear_schedule_with_warmup)

from data import ROOT, load_config, load_splits
from env_info import collect as collect_env

MB = 2**20


def run_id(precision, rank, seed):
    return f"{precision}_r{rank}_s{seed}"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode(tokenizer, texts, max_length):
    """Tokenize without padding; return token-id lists and how many inputs had to be truncated."""
    ids = tokenizer(texts, add_special_tokens=True)["input_ids"]
    n_truncated = sum(len(x) > max_length for x in ids)
    return [x[:max_length] for x in ids], n_truncated


def collate(ids, idx, pad_id):
    """Right-pad one batch on the CPU. Returns (input_ids, attention_mask, n_real_tokens)."""
    seqs = [ids[i] for i in idx]
    width = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for row, s in enumerate(seqs):
        input_ids[row, :len(s)] = torch.tensor(s)
        mask[row, :len(s)] = 1
    return input_ids.cuda(non_blocking=True), mask.cuda(non_blocking=True), sum(len(s) for s in seqs)


def load_model(cfg, precision, n_labels, pad_id):
    kwargs = {"num_labels": n_labels, "revision": cfg["model_revision"], "dtype": torch.float16}
    if precision == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["score"],  # keep the new classification head out of 4-bit
        )
        kwargs["device_map"] = {"": 0}
    elif precision != "fp16":
        raise ValueError(f"unknown precision {precision!r}")
    model = AutoModelForSequenceClassification.from_pretrained(cfg["model_id"], **kwargs)
    if precision == "fp16":
        model.cuda()
    model.config.pad_token_id = pad_id
    # Qwen's final hidden states are large; a randomly initialised head starts at loss ~16 instead
    # of ln(77) ~ 4.3. Zero-init gives every run the same neutral starting point.
    torch.nn.init.zeros_(model.score.weight)
    return model


def add_lora(model, cfg, rank):
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # also marks the "score" head as fully trainable
        r=rank,
        lora_alpha=cfg["lora_alpha_multiplier"] * rank,  # alpha/r held constant across ranks
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
    )
    model = get_peft_model(model, lora)
    # Trainable weights live in fp32 for both precisions so the GradScaler can unscale them and
    # the only difference between arms is how the frozen base is stored.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return model


@torch.no_grad()
def predict(model, ids, labels, batch_size, pad_id):
    """Batched forward pass over a split. Returns (preds, mean loss, wall seconds)."""
    model.eval()
    logits_all, loss_sum = [], torch.zeros((), device="cuda")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for lo in range(0, len(ids), batch_size):
        idx = range(lo, min(lo + batch_size, len(ids)))
        x, m, _ = collate(ids, idx, pad_id)
        y = torch.tensor([labels[i] for i in idx], device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(input_ids=x, attention_mask=m).logits.float()
        loss_sum += F.cross_entropy(logits, y, reduction="sum")
        logits_all.append(logits.argmax(-1))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return torch.cat(logits_all).tolist(), loss_sum.item() / len(ids), elapsed


@torch.no_grad()
def batch1_latency(model, ids, pad_id, n_queries, warmup):
    """Per-query latency at batch size 1, the setting a live support-ticket router would see."""
    model.eval()
    times = []
    for k in range(warmup + n_queries):
        x, m, _ = collate(ids, [k % len(ids)], pad_id)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            model(input_ids=x, attention_mask=m)
        torch.cuda.synchronize()
        if k >= warmup:
            times.append(time.perf_counter() - start)
    return 1000 * np.percentile(times, 50), 1000 * np.percentile(times, 95)


def train(model, cfg, train_ids, train_y, val_ids, val_y, pad_id, seed, epochs):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    bs = cfg["batch_size"]
    steps_per_epoch = math.ceil(len(train_ids) / bs)
    total_steps = steps_per_epoch * epochs
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda")
    order_gen = torch.Generator().manual_seed(seed)

    curve = {"step_loss": [], "epoch": [], "train_loss": [], "val_loss": [], "val_acc": []}
    train_seconds, train_tokens, diverged = 0.0, 0, False
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_ids), generator=order_gen).tolist()
        losses = []
        torch.cuda.synchronize()
        start = time.perf_counter()
        for step in range(steps_per_epoch):
            idx = order[step * bs:(step + 1) * bs]
            x, m, n_tok = collate(train_ids, idx, pad_id)
            y = torch.tensor([train_y[i] for i in idx], device="cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(input_ids=x, attention_mask=m).logits
            loss = F.cross_entropy(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.detach())
            train_tokens += n_tok
        torch.cuda.synchronize()
        train_seconds += time.perf_counter() - start

        step_losses = torch.stack(losses).tolist()
        k = cfg["log_every"]
        curve["step_loss"] += [float(np.mean(step_losses[i:i + k])) for i in range(0, len(step_losses), k)]
        epoch_loss = float(np.mean(step_losses))
        val_pred, val_loss, _ = predict(model, val_ids, val_y, cfg["eval_batch_size"], pad_id)
        val_acc = accuracy_score(val_y, val_pred)
        curve["epoch"].append(epoch)
        curve["train_loss"].append(epoch_loss)
        curve["val_loss"].append(val_loss)
        curve["val_acc"].append(val_acc)
        print(f"  epoch {epoch}: train_loss={epoch_loss:.4f} val_loss={val_loss:.4f} "
              f"val_acc={val_acc:.4f} ({train_seconds:.0f}s)", flush=True)
        if not (math.isfinite(epoch_loss) and math.isfinite(val_loss)):
            diverged = True
            break
    return curve, train_seconds, train_tokens, diverged


def append_row(path, row):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=["fp16", "nf4"], required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--results-dir", default=None,
                        help="default: results/ (or results_smoke/ with --smoke)")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny subset, 1 epoch: checks the pipeline end to end in ~1 minute")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required (the reference runs used a Colab T4).")

    cfg = load_config()
    results_dir = Path(args.results_dir or ROOT / ("results_smoke" if args.smoke else "results"))
    for sub in ("predictions", "curves"):
        (results_dir / sub).mkdir(parents=True, exist_ok=True)
    rid = run_id(args.precision, args.rank, args.seed)
    print(f"[{rid}] starting", flush=True)

    splits = load_splits(cfg)
    epochs = cfg["epochs"]
    if args.smoke:
        s = cfg["smoke"]
        for name, n in (("train", s["n_train"]), ("val", s["n_val"]), ("test", s["n_test"])):
            splits[name] = {key: vals[:n] for key, vals in splits[name].items()}
        epochs = s["epochs"]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["model_revision"])
    pad_id = tokenizer.pad_token_id
    encoded, truncated = {}, 0
    for name in ("train", "val", "test"):
        encoded[name], n_trunc = encode(tokenizer, splits[name]["text"], cfg["max_length"])
        truncated += n_trunc

    set_seed(args.seed)  # seeds the classification-head init, LoRA init, dropout, and data order
    torch.cuda.reset_peak_memory_stats()
    model = add_lora(load_model(cfg, args.precision, len(splits["labels"]), pad_id), cfg, args.rank)
    trainable, total = model.get_nb_trainable_parameters()
    mem_after_load = torch.cuda.memory_allocated() / MB

    torch.cuda.reset_peak_memory_stats()
    curve, train_seconds, train_tokens, diverged = train(
        model, cfg, encoded["train"], splits["train"]["label"],
        encoded["val"], splits["val"]["label"], pad_id, args.seed, epochs)
    peak_train_alloc = torch.cuda.max_memory_allocated() / MB
    peak_train_reserved = torch.cuda.max_memory_reserved() / MB

    torch.cuda.reset_peak_memory_stats()
    test_y = splits["test"]["label"]
    test_pred, test_loss, test_seconds = predict(model, encoded["test"], test_y, cfg["eval_batch_size"], pad_id)
    peak_infer_alloc = torch.cuda.max_memory_allocated() / MB
    lat = cfg["latency"]
    p50, p95 = batch1_latency(model, encoded["test"], pad_id, lat["n_queries"], lat["warmup"])

    with open(results_dir / "predictions" / f"{rid}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "label", "pred"])
        writer.writerows(zip(range(len(test_y)), test_y, test_pred))
    with open(results_dir / "curves" / f"{rid}.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(curve, f)

    env = collect_env()
    row = {
        "run_id": rid,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "precision": args.precision,
        "rank": args.rank,
        "lora_alpha": cfg["lora_alpha_multiplier"] * args.rank,
        "seed": args.seed,
        "model_id": cfg["model_id"],
        "model_revision": cfg["model_revision"],
        "epochs": len(curve["epoch"]),
        "batch_size": cfg["batch_size"],
        "max_length": cfg["max_length"],
        "lr": cfg["lr"],
        "n_train": len(encoded["train"]),
        "n_val": len(encoded["val"]),
        "n_test": len(encoded["test"]),
        "n_truncated": truncated,
        "trainable_params": trainable,
        "total_params": total,
        "embedding_params": model.get_input_embeddings().weight.numel(),
        "diverged": diverged,
        "test_acc": accuracy_score(test_y, test_pred),
        "test_macro_f1": f1_score(test_y, test_pred, average="macro"),
        "test_loss": test_loss,
        "final_train_loss": curve["train_loss"][-1],
        "final_val_loss": curve["val_loss"][-1],
        "final_val_acc": curve["val_acc"][-1],
        "train_seconds": round(train_seconds, 2),
        "train_tokens": train_tokens,
        "mem_after_load_mb": round(mem_after_load, 1),
        "peak_train_alloc_mb": round(peak_train_alloc, 1),
        "peak_train_reserved_mb": round(peak_train_reserved, 1),
        "peak_infer_alloc_mb": round(peak_infer_alloc, 1),
        "test_throughput_ex_per_s": round(len(test_y) / test_seconds, 1),
        "latency_b1_p50_ms": round(p50, 2),
        "latency_b1_p95_ms": round(p95, 2),
        **{k: env[k] for k in ("gpu_name", "gpu_mem_total_mb", "gpu_capability", "driver",
                                "cuda_runtime", "torch", "transformers", "peft", "bitsandbytes",
                                "python", "colab", "git_commit", "git_dirty")},
    }
    append_row(results_dir / "runs.csv", row)
    print(f"[{rid}] test_acc={row['test_acc']:.4f} peak_train={peak_train_alloc:.0f}MB "
          f"train={train_seconds:.0f}s p50={p50:.1f}ms", flush=True)


if __name__ == "__main__":
    main()
