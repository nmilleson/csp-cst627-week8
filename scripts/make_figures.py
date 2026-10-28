"""Render every figure and the summary table from the committed raw results.

Reads only results/runs.csv, results/baseline.json and results/audit/summary.json, so it runs on
a CPU in seconds and needs no GPU or model download.
"""
import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data import ROOT  # noqa: E402

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"fp16": ("#2a78d6", "fp16 base + LoRA"), "nf4": ("#eb6834", "NF4 base + LoRA (QLoRA)")}
BASELINE_COLOR = "#898781"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 7,
})


def summarise(runs):
    runs = runs.assign(
        train_minutes=runs["train_seconds"] / 60,
        peak_train_gb=runs["peak_train_alloc_mb"] / 1024,
        peak_infer_gb=runs["peak_infer_alloc_mb"] / 1024,
        gen_gap=runs["final_val_loss"] - runs["final_train_loss"],
        # LoRA training cost ~4 FLOPs per non-embedding parameter per token: 2 forward + 2 for
        # activation gradients. Frozen weights get no weight-gradient pass.
        train_tflop=4 * (runs["total_params"] - runs["embedding_params"]) * runs["train_tokens"] / 1e12,
    )
    agg = runs.groupby(["precision", "rank"]).agg(
        n_seeds=("seed", "nunique"),
        acc_mean=("test_acc", "mean"), acc_sd=("test_acc", "std"),
        f1_mean=("test_macro_f1", "mean"),
        peak_train_gb=("peak_train_gb", "mean"), peak_infer_gb=("peak_infer_gb", "mean"),
        train_min_mean=("train_minutes", "mean"), train_min_sd=("train_minutes", "std"),
        p50_mean=("latency_b1_p50_ms", "mean"), p50_sd=("latency_b1_p50_ms", "std"),
        p95_mean=("latency_b1_p95_ms", "mean"),
        throughput=("test_throughput_ex_per_s", "mean"),
        gap_mean=("gen_gap", "mean"), gap_sd=("gen_gap", "std"),
        trainable_m=("trainable_params", lambda s: s.iloc[0] / 1e6),
        train_tflop=("train_tflop", "mean"),
        n_test=("n_test", "first"),
        diverged=("diverged", lambda s: int(s.astype(str).str.lower().eq("true").sum())),
    ).reset_index()
    agg["test_se"] = np.sqrt(agg["acc_mean"] * (1 - agg["acc_mean"]) / agg["n_test"])
    return agg.fillna({"acc_sd": 0, "train_min_sd": 0, "p50_sd": 0, "gap_sd": 0})


def hardware_note(runs):
    gpus = ", ".join(sorted(runs["gpu_name"].unique()))
    torch_v = ", ".join(sorted(runs["torch"].astype(str).unique()))
    seeds = runs.groupby(["precision", "rank"])["seed"].nunique()
    n = f"n={seeds.min()}" if seeds.min() == seeds.max() else f"n={seeds.min()}-{seeds.max()}"
    return f"{gpus} | torch {torch_v} | Qwen2.5-0.5B on Banking77 | error bars: ±1 SD across seeds ({n})"


def finish(fig, ax, title, note, path):
    """Title, legend in a row under it (never over the data), wrapped hardware footnote, save."""
    ax.set_title(title, pad=30)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.2, handletextpad=0.4)
    fig.text(0.01, 0.01, textwrap.fill(note, 115), fontsize=7.5, color=INK_2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def rank_axis(ax, ranks):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks, [str(r) for r in ranks])
    ax.minorticks_off()
    ax.set_xlabel("LoRA rank r (alpha = 2r)")


def fig_accuracy(agg, runs, baseline, note, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ranks = sorted(agg["rank"].unique())
    for i, (prec, (color, label)) in enumerate(SERIES.items()):
        a = agg[agg["precision"] == prec]
        if a.empty:
            continue
        shift = 0.96 if i == 0 else 1.04  # nudge the two series apart so error bars don't overlap
        x = a["rank"] * shift
        ax.errorbar(x, 100 * a["acc_mean"], yerr=100 * a["acc_sd"], color=color, marker="o",
                    capsize=3, label=label, zorder=3)
        r = runs[runs["precision"] == prec]
        ax.scatter(r["rank"] * shift, 100 * r["test_acc"], s=14, color=color, alpha=0.35,
                   linewidths=0, zorder=2)
    if baseline:
        ax.axhline(100 * baseline["test_acc"], color=BASELINE_COLOR, linestyle="--", linewidth=1.5)
        ax.text(ranks[0] * 0.92, 100 * baseline["test_acc"] + 0.15, "TF-IDF + logistic regression (CPU)",
                color=INK_2, fontsize=8.5, va="bottom")
    rank_axis(ax, ranks)
    ax.set_ylabel("Test accuracy (%)")
    finish(fig, ax, "Test accuracy vs LoRA rank", note + " | faint dots = individual seeds", path)


def fig_memory(agg, note, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for prec, (color, label) in SERIES.items():
        a = agg[agg["precision"] == prec].sort_values("rank")
        if a.empty:
            continue
        ax.errorbar(a["peak_train_gb"], 100 * a["acc_mean"], yerr=100 * a["acc_sd"], color=color,
                    marker="o", capsize=3, label=label)
        for _, row in a.iterrows():
            ax.annotate(f"r{row['rank']}", (row["peak_train_gb"], 100 * row["acc_mean"]),
                        textcoords="offset points", xytext=(6, -10), fontsize=8, color=INK_2)
    ax.set_xlabel("Peak training memory, torch allocated (GB)")
    ax.set_ylabel("Test accuracy (%)")
    finish(fig, ax, "Accuracy vs training memory", note, path)


def fig_by_rank(agg, mean_col, sd_col, ylabel, title, note, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for i, (prec, (color, label)) in enumerate(SERIES.items()):
        a = agg[agg["precision"] == prec]
        if a.empty:
            continue
        x = a["rank"] * (0.96 if i == 0 else 1.04)
        ax.errorbar(x, a[mean_col], yerr=a[sd_col], color=color, marker="o", capsize=3, label=label)
    rank_axis(ax, sorted(agg["rank"].unique()))
    ax.set_ylabel(ylabel)
    finish(fig, ax, title, note, path)


def markdown_table(agg, baseline, audit, note):
    def pm(m, s, scale=1, digits=1):
        return f"{m * scale:.{digits}f} ± {s * scale:.{digits}f}"

    lines = [
        "| Base precision | Rank | Seeds | Test acc (%) | Macro-F1 (%) | Trainable params (M) | Peak train mem (GB) "
        "| Train time (min) | Batch-1 latency p50 / p95 (ms) | Throughput, batch 64 (ex/s) | Est. train compute (TFLOP) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in agg.sort_values(["precision", "rank"]).iterrows():
        lines.append(
            f"| {r['precision']} | {r['rank']} | {r['n_seeds']} | {pm(r['acc_mean'], r['acc_sd'], 100, 2)} "
            f"| {100 * r['f1_mean']:.2f} | {r['trainable_m']:.2f} | {r['peak_train_gb']:.2f} "
            f"| {pm(r['train_min_mean'], r['train_min_sd'])} | {r['p50_mean']:.1f} / {r['p95_mean']:.1f} "
            f"| {r['throughput']:.0f} | {r['train_tflop']:.0f} |")
    if baseline:
        lines.append(f"| TF-IDF + LogReg (CPU) | – | 1 | {100 * baseline['test_acc']:.2f} "
                     f"| {100 * baseline['test_macro_f1']:.2f} | – | – | {baseline['fit_seconds_cpu'] / 60:.1f} (CPU) | – | – | – |")
    se = 100 * agg["test_se"].max() if not agg.empty else float("nan")
    lines += ["", f"Values are mean ± 1 sample SD across seeds. Test-set sampling error of a single "
                  f"accuracy is about ±{se:.2f} pp (binomial SE, n = {int(agg['n_test'].iloc[0])}). "
                  "Configurations share the same test items, so differences between them are judged with "
                  "the paired bootstrap in results/audit/paired_differences.csv.",
              f"Hardware and software: {note.split(' | error')[0]}."]
    if audit:
        lines.append(f"Train/test overlap: {audit['exact_duplicates_in_train']} exact and "
                     f"{audit['near_duplicates_in_train']} near-duplicate test items "
                     f"(TF-IDF cosine >= {audit['near_duplicate_cosine_threshold']}); see results/audit/.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--figures-dir", default=str(ROOT / "figures"))
    args = parser.parse_args()
    results_dir, figures_dir = Path(args.results_dir), Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    runs_csv = results_dir / "runs.csv"
    if not runs_csv.exists():
        raise SystemExit(f"{runs_csv} not found - run scripts/run_ablation.py first.")
    runs = pd.read_csv(runs_csv)
    baseline_path, audit_path = results_dir / "baseline.json", results_dir / "audit" / "summary.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else None

    agg = summarise(runs)
    note = hardware_note(runs)
    # LF everywhere so a regeneration on Windows is byte-identical to the committed files
    agg.round(5).to_csv(results_dir / "summary.csv", index=False, lineterminator="\n")
    (results_dir / "summary.md").write_text(markdown_table(agg, baseline, audit, note),
                                            encoding="utf-8", newline="\n")

    fig_accuracy(agg, runs, baseline, note, figures_dir / "fig1_accuracy_vs_rank.png")
    fig_memory(agg, note, figures_dir / "fig2_accuracy_vs_memory.png")
    fig_by_rank(agg, "p50_mean", "p50_sd", "Median latency, batch size 1 (ms)",
                "Single-query inference latency (adapters unmerged)", note, figures_dir / "fig3_latency.png")
    fig_by_rank(agg, "gap_mean", "gap_sd", "Final val loss − train loss",
                "Generalisation gap after training", note, figures_dir / "fig4_generalisation_gap.png")
    print((results_dir / "summary.md").read_text(encoding="utf-8"))
    print(f"Figures written to {figures_dir}")


if __name__ == "__main__":
    main()
