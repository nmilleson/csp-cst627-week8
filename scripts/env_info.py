"""Record the hardware and software environment.

Imported by train_eval.py so every results row carries its own environment, and runnable on its
own to write results/environment.json and results/pip_freeze.txt.
"""
import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys

from data import ROOT

PACKAGES = ["torch", "transformers", "peft", "bitsandbytes", "accelerate",
            "scikit-learn", "numpy", "pandas", "matplotlib"]


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=ROOT).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def collect():
    import torch

    driver = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).splitlines()
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "colab": "COLAB_RELEASE_TAG" in os.environ,
        "git_commit": _run(["git", "rev-parse", "--short", "HEAD"]),
        "git_dirty": bool(_run(["git", "status", "--porcelain", "--", "scripts", "configs"])),
        "cuda_runtime": torch.version.cuda or "",
        "cudnn": str(torch.backends.cudnn.version() or ""),
        "gpu_name": "",
        "gpu_mem_total_mb": 0,
        "gpu_capability": "",
        "driver": driver[0] if driver else "",
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_mem_total_mb"] = round(props.total_memory / 2**20)
        info["gpu_capability"] = f"{props.major}.{props.minor}"
    for pkg in PACKAGES:
        try:
            info[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            info[pkg] = ""
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    args = parser.parse_args()

    out = os.path.abspath(args.results_dir)
    os.makedirs(out, exist_ok=True)
    info = collect()
    with open(os.path.join(out, "environment.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(info, f, indent=2)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    with open(os.path.join(out, "pip_freeze.txt"), "w", encoding="utf-8") as f:
        f.write(freeze)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
