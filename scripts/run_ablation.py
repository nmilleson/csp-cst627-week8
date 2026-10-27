"""Run the full precision x rank x seed grid, one fresh process per configuration.

Resumable: configurations already present in runs.csv are skipped, so a dropped Colab session
can be picked up by re-running the same command. The loop is seed-major, so if the budget runs
out part-way you still have a complete precision x rank picture at seed 0.
"""
import argparse
import csv
import itertools
import subprocess
import sys
from pathlib import Path

from data import ROOT, load_config
from train_eval import run_id


def completed(runs_csv):
    if not runs_csv.exists():
        return set()
    with open(runs_csv, newline="", encoding="utf-8") as f:
        return {row["run_id"] for row in csv.DictReader(f)}


def main():
    cfg = load_config()
    grid = cfg["grid"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=None,
                        help="default: results/ (or results_smoke/ with --smoke)")
    parser.add_argument("--seed", type=int, nargs="+", default=grid["seed"])
    parser.add_argument("--precision", nargs="+", default=grid["precision"])
    parser.add_argument("--rank", type=int, nargs="+", default=grid["rank"])
    parser.add_argument("--smoke", action="store_true", help="pass --smoke to every run")
    args = parser.parse_args()

    results_dir = Path(args.results_dir or ROOT / ("results_smoke" if args.smoke else "results"))
    done = completed(results_dir / "runs.csv")
    todo = [c for c in itertools.product(args.seed, args.precision, args.rank) if run_id(c[1], c[2], c[0]) not in done]
    print(f"{len(todo)} configurations to run, {len(done)} already in {results_dir / 'runs.csv'}")

    failed = []
    for i, (seed, precision, rank) in enumerate(todo, 1):
        print(f"\n=== {i}/{len(todo)}: {run_id(precision, rank, seed)} ===", flush=True)
        cmd = [sys.executable, str(ROOT / "scripts" / "train_eval.py"),
               "--precision", precision, "--rank", str(rank), "--seed", str(seed),
               "--results-dir", str(results_dir)] + (["--smoke"] if args.smoke else [])
        if subprocess.run(cmd).returncode != 0:
            failed.append(run_id(precision, rank, seed))

    if failed:
        sys.exit(f"Failed configurations (re-run to retry): {', '.join(failed)}")
    print("\nGrid complete.")


if __name__ == "__main__":
    main()
