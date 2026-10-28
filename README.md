# How much LoRA do you need on one T4? A seed-controlled rank × precision ablation

**Question.** For routing customer-support messages into 77 banking intents, which LoRA rank (4, 8, 16, 32) and which base-model precision (fp16 vs 4-bit NF4, i.e. QLoRA) gives the best held-out accuracy for Qwen2.5-0.5B, and what does each choice cost in GPU memory, training time and inference latency?
**Budget.** One NVIDIA T4 (16 GB, free-tier Google Colab), about 3 GPU-hours total: 24 runs (2 precisions × 4 ranks × 3 seeds) at fixed hyperparameters.
**Decision it supports.** Which adapter configuration a small team without a GPU cluster should pick for an intent classifier, and whether it beats a CPU-only TF-IDF baseline by enough to justify a GPU at all.

<!-- FILL after the Colab run: one or two sentences, bounded. Template:
**Headline (bounded).** On this task and budget, rank ___ with an ___ base reached ___ ± ___ % test accuracy (mean ± SD, 3 seeds), ___ pp above the CPU baseline. Ranks above ___ did not improve accuracy beyond seed-to-seed spread; NF4 cut peak training memory by ___ GB (___ %) at a cost of ___ pp accuracy and ___ ms single-query latency. These conclusions are specific to a 0.5B model, 9k training examples and untuned per-rank learning rates (see Limitations).
-->

![Test accuracy vs LoRA rank](figures/fig1_accuracy_vs_rank.png)

---

## Contents
1. [Reproduce it](#1-reproduce-it)
2. [Repository layout](#2-repository-layout)
3. [Evaluation protocol](#3-evaluation-protocol)
4. [Results](#4-results)
5. [What this does and does not show](#5-what-this-does-and-does-not-show)
6. [Limitations and threats to validity](#6-limitations-and-threats-to-validity)
7. [Model and data documentation (audit view)](#7-model-and-data-documentation-audit-view)
8. [EU AI Act Annex IV mapping](#8-eu-ai-act-annex-iv-mapping)
9. [Why this matters here](#9-why-this-matters-here)
10. [Skills demonstrated](#10-skills-demonstrated)
11. [What more compute would buy](#11-what-more-compute-would-buy)

---

## 1. Reproduce it

**Figures only (any machine, no GPU, ~1 minute).** Every figure and the results table are rendered from the committed raw measurements in `results/`:

```bash
git clone <this-repo-url> && cd <repo>
pip install -r requirements.txt   # Python 3.11-3.13
pip uninstall -y torchao          # only needed if torchao < 0.16 is present (Colab ships 0.10)
python scripts/make_figures.py    # writes figures/*.png and results/summary.md
```

**Full re-run (one GPU, ~3 hours on a T4).** Open `run_on_colab.ipynb` in Colab with a T4 runtime and run the cells top to bottom. The notebook only calls the scripts below, so the same commands work on any CUDA machine:

```bash
python scripts/env_info.py      --results-dir results_rerun   # hardware + pip freeze
python scripts/baseline_tfidf.py --results-dir results_rerun   # CPU baseline, ~1 min
python scripts/run_ablation.py   --results-dir results_rerun   # 24 runs, resumable
python scripts/audit.py          --results-dir results_rerun   # overlap + slice audit
python scripts/make_figures.py   --results-dir results_rerun --figures-dir results_rerun/figures
```

`run_ablation.py --smoke --seed 0 --rank 8` checks the whole pipeline on a 256-example subset in about two minutes. The dataset is downloaded from a pinned commit and checked against a SHA-256 hash; the model is loaded at a pinned Hugging Face revision. Nothing depends on files outside the repository.

## 2. Repository layout

| Path | Purpose |
|---|---|
| `configs/ablation.json` | Single source of truth: model/data pins, hyperparameters, grid |
| `scripts/data.py` | Download, hash check, fixed stratified train/val split |
| `scripts/train_eval.py` | One configuration: train, evaluate, time, measure memory, append one row |
| `scripts/run_ablation.py` | The grid, one fresh process per run (clean peak-memory readings), resumable |
| `scripts/baseline_tfidf.py` | TF-IDF + logistic regression reference on the same split |
| `scripts/audit.py` | Train/test overlap, accuracy on clean vs overlapping items, length and per-intent slices |
| `scripts/make_figures.py` | All figures and `results/summary.md` from the raw CSV |
| `scripts/env_info.py` | Hardware/software capture |
| `results/runs.csv` | **Raw measurements**, one row per run, including hardware and library versions |
| `results/predictions/`, `results/curves/` | Per-run test predictions and training curves |
| `results/audit/` | Overlap table, slice and per-intent accuracy |
| `figures/` | Rendered from `results/` only |
| `run_on_colab.ipynb` | Thin notebook that calls the scripts |

## 3. Evaluation protocol

| Item | Setting |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B` at revision `060db6499f32faf8b98477b0a26969ef7d8b9987` (Apache-2.0), with a 77-way linear classification head on the last non-padding token |
| Data | Banking77 ([Casanueva et al., 2020](https://arxiv.org/abs/2003.04807)), CC-BY-4.0, from commit `57ec275` of `PolyAI-LDN/task-specific-datasets`; SHA-256 recorded in `configs/ablation.json` |
| Split | Official train file (10,003) split once, stratified, `split_seed=1234`: 9,002 train / 1,001 validation. Official test file (3,080) used only for final reporting. Same split for every run. |
| Grid | Base precision {fp16, NF4 with double quantisation} × LoRA rank {4, 8, 16, 32} × seed {0, 1, 2} = 24 runs |
| LoRA | All attention and MLP projections (`q,k,v,o,gate,up,down`), alpha = 2r (constant scaling across ranks), dropout 0.05; classification head fully trained and zero-initialised. Adapter and head weights kept in fp32 in both arms, so the arms differ only in how the frozen base is stored. |
| Optimisation | AdamW, lr 2e-4, no weight decay, 6 % linear warm-up then linear decay, 3 epochs, batch 32, grad-norm clip 1.0, fp16 autocast with dynamic loss scaling (T4 has no bf16). Max length 128 tokens (no input was truncated). |
| What the seed controls | LoRA A-matrix initialisation, dropout masks, data order. **Not** the data split. |
| Accuracy | Test accuracy and macro-F1 on all 3,080 test items, after the final epoch (no checkpoint selection on test) |
| Memory | `torch.cuda.max_memory_allocated` during training, reset after model load; `max_memory_reserved` also logged. Excludes the CUDA context (~0.3-0.5 GB). |
| Time | Wall-clock of training steps only (validation passes excluded), `torch.cuda.synchronize()` at boundaries |
| Latency | Batch size 1, 20 warm-up then 200 timed test queries, p50 and p95; adapters **unmerged**. Throughput: full test set at batch 64. |
| Compute estimate | ≈ 4 × (non-embedding params) × (training tokens): 2 FLOPs/param/token forward plus 2 for activation gradients; frozen weights get no weight-gradient pass |
| Uncertainty | Mean ± 1 sample SD over 3 seeds; binomial test-set SE reported alongside |

## 4. Results

<!-- FILL: paste the contents of results/summary.md here after the Colab run. -->

*Table to be generated by `scripts/make_figures.py` → `results/summary.md`.*

| Figure | What to read from it |
|---|---|
| ![](figures/fig1_accuracy_vs_rank.png) | Does accuracy rise with rank beyond the seed spread? Where is the CPU baseline? |
| ![](figures/fig2_accuracy_vs_memory.png) | The memory-accuracy trade-off between fp16 and NF4 bases |
| ![](figures/fig3_latency.png) | Single-query latency cost of a 4-bit base |
| ![](figures/fig4_generalisation_gap.png) | Does higher rank overfit more (val - train loss)? |

**Hardware footnote.** <!-- FILL from results/environment.json: GPU, driver, CUDA runtime, torch, Colab or not. --> All rows of `results/runs.csv` carry their own GPU name, driver, CUDA and library versions.

## 5. What this does and does not show

<!-- FILL after results. Keep each claim tied to the budget. Templates: -->
- **Shown, within this budget:** <!-- e.g. "At 0.5B parameters and 9k examples, rank 4 and rank 32 are indistinguishable: the difference (__ pp) is smaller than the seed SD (__ pp) and the test-set SE (__ pp)." -->
- **Shown:** <!-- e.g. "An NF4 base saves __ GB of peak training memory, which is only __ % because activations, not weights, dominate memory at this model size." -->
- **Not shown:** that the best rank here is best for larger models, other tasks, or tuned learning rates.
- **Not shown:** production latency. Batch-1 latency on a 0.5B model is dominated by kernel-launch overhead and the host CPU, and adapters were not merged.
- **Not shown:** that the model is free of pretraining contamination (see §7).

## 6. Limitations and threats to validity

Each item says how a reader should discount the result.

1. **Three seeds, one split.** Seeds vary initialisation, dropout and data order but not the train/val split or the test set. Spread across seeds therefore understates total variance. *Discount:* treat any difference smaller than about 2 SD or 2 × test SE as no difference.
2. **Hyperparameters fixed, not tuned per arm.** Learning rate, epochs and alpha/r were chosen once and held fixed. Higher ranks or the NF4 base might prefer a different learning rate, so a gap between arms may be a tuning artefact. *Discount:* read rank comparisons as "at lr 2e-4, 3 epochs", not in general.
3. **One model size, one task, one language.** 0.5B parameters, English, short single-sentence queries, balanced intents. Rank and quantisation effects are known to change with scale. *Discount:* do not extrapolate to 7B+ models or to long-document tasks.
4. **Train/test overlap in the public benchmark.** 24 test items are exact duplicates of training items after normalisation, and 124 (4.0 %) are near-duplicates (TF-IDF cosine ≥ 0.9). `results/audit/summary.json` reports accuracy on the clean subset separately. *Discount:* use the clean-subset number if it differs.
5. **Pretraining contamination cannot be ruled out.** Banking77 has been public since 2020, and Qwen2.5's training corpus is not disclosed. *Discount:* absolute accuracy may be optimistic. The comparison *between* arms is less affected because all arms share the same base model.
6. **Shared, uncontrolled hardware.** Colab T4s are shared VMs with variable clocks and host CPUs, and runs may land on different machines across sessions. *Discount:* latency and time numbers are indicative (±10-20 %). Check `gpu_name`/`driver` per row in `runs.csv`.
7. **No bf16 on T4.** Training used fp16 with loss scaling. Results on bf16-capable GPUs (A100, L4, H100) may differ slightly in stability and speed.
8. **Memory excludes the CUDA context and allocator fragmentation.** The `nvidia-smi` footprint will be higher. `peak_train_reserved_mb` gives a closer upper bound.

## 7. Model and data documentation (audit view)

**Intended use.** A reference measurement for choosing a parameter-efficient fine-tuning configuration for short-text intent classification on a single small GPU. The trained adapters are research artefacts used to produce these measurements.

**Out-of-scope uses.**
- Routing real customer messages without a human fallback.
- Any decision about a customer's credit, account access or eligibility.
- Non-English text, or messages longer than about 80 words (outside the training distribution).
- Treating the reported accuracy as a guarantee on a bank's own traffic.

**Data provenance and licensing.**
- **Data:** Banking77, 13,083 English customer-service queries labelled with 77 intents by the authors (PolyAI). Licensed CC-BY-4.0, attribution: Casanueva et al., 2020. It contains no personal data by design, but the queries are synthetic-style and cleaner than real chat logs.
- **Model:** Qwen2.5-0.5B, Apache-2.0, © Alibaba Cloud.
- **Pinning:** both are pinned by commit hash, and the data is verified by SHA-256 on every load.

**Known failure modes (from the audit).** <!-- FILL from results/audit/summary.json: the 5 weakest intents for the chosen config, and accuracy by length slice. Expected pattern: confusable pairs such as pending vs declined transfers. -->

**Fairness note.** Banking77 has no demographic attributes, so group-fairness metrics cannot be computed. The audit reports the available proxies instead: per-intent accuracy (does any customer need get systematically worse service?) and query-length slices (do terse messages fail more?). A deployment would need dialect, language and accessibility testing on real traffic.

**Risk framing (NIST AI RMF).**
- **Map:** intent routing is low-to-moderate impact, because a misrouted message delays service but makes no decision about the customer.
- **Measure:** this repository.
- **Manage:** the deployment controls named in §8.

## 8. EU AI Act Annex IV mapping

Intent routing for customer service is not an Annex III high-risk use; credit scoring would be. The mapping below shows where each Annex IV technical-documentation element is covered, as practice for systems where it is mandatory.

| Annex IV element | Where it is covered |
|---|---|
| 1. General description: intended purpose, versions, hardware | §7 intended use, §3 model/data pins, `results/environment.json`, hardware columns in `runs.csv` |
| 2. Development process: design choices, data provenance, training methodology, validation and testing procedures, metrics, test logs | §3 protocol, `configs/ablation.json`, `scripts/`, `results/runs.csv`, `results/curves/`, `results/predictions/` |
| 3. Capabilities and limitations, accuracy levels, foreseeable unintended outcomes | §4 results, §5, §6, §7 failure modes |
| 4. Appropriateness of performance metrics | §3 (accuracy + macro-F1 on a balanced 77-class set; latency at batch 1 for interactive use) |
| 5. Risk management | §6, §7 risk framing |
| 6. Changes over the lifecycle | Git history; every row records `git_commit` and `git_dirty` |
| 7. Standards applied | Reporting follows NIST AI RMF 1.0 *Measure* practices; documentation structure follows ISO/IEC 42001 Annex A (A.6 lifecycle, A.7 data) |
| 8. EU declaration of conformity | Not applicable (research artefact, not placed on the market) |
| 9. Post-market monitoring | Not deployed. A deployment would monitor per-intent accuracy on sampled, human-labelled traffic and the share of low-confidence predictions. |

## 9. Why this matters here

<!-- FILL: name a specific company and role. Draft for a typical target: -->
**Target role:** <!-- e.g. "Machine Learning Engineer, digital banking team at [named regional bank / credit-union fintech], ~30-person data group, no in-house GPU cluster" or "Model Validation Analyst at [named regulated lender]" -->

<!-- Draft paragraph, edit to the employer's own vocabulary:
Teams like [company]'s route thousands of customer messages a day and are under model-risk-management expectations (e.g. SR 11-7 in the US) to show that a model's performance claims are reproducible, bounded and independently checkable. They usually do not have a GPU cluster, so the useful question is not "what is the best model" but "what is the cheapest configuration that is reliably better than what we run today, and how sure are we". This project answers that for one realistic task. It reports spread across seeds instead of a best run, compares against a CPU baseline a validator would ask about, separates train/test overlap from the headline number, and is packaged so a second-line reviewer can regenerate every figure from raw logs.
-->

## 10. Skills demonstrated

| Skill (course outcome) | Evidence in this repository |
|---|---|
| Parameter-efficient adaptation within a free-tier GPU (CLO2) | `scripts/train_eval.py`: LoRA and QLoRA on a 0.5B model inside a 16 GB T4 |
| Comparing accuracy, latency and memory alongside compute estimates (CLO4) | `results/runs.csv` columns; Figures 1-3; TFLOP estimate in `results/summary.md` |
| Memory budgeting and profiling | Peak allocated vs reserved memory per run; fresh process per run so peaks are not contaminated; §6.8 |
| Honest benchmarking | 3 seeds with SD, binomial SE, CPU baseline, no test-set checkpoint selection, synchronised timers, warm-up before latency |
| Interpreting generalisation evidence (CLO1) | Figure 4 and `results/curves/`: train vs validation loss by rank |
| Auditing for contamination and residual risk (CLO5) | `scripts/audit.py`: exact and near-duplicate overlap, clean-subset accuracy; §6.4-6.5; NIST AI RMF framing |
| Controlled ablation as a reproducible package (CLO6) | Pinned data hash and model revision, `requirements.txt`, resumable grid, figures rendered only from committed CSV; §8 Annex IV mapping |

## 11. What more compute would buy

In order of how much it would change the conclusions:

1. **A per-arm learning-rate sweep** (3-5 values). This removes threat §6.2 and is about 5× the current budget.
2. **More seeds plus resampled train/val splits** (10 seeds × 3 splits), to resolve differences smaller than about 0.5 pp.
3. **The same grid at 1.5B and 7B.** 7B needs NF4 even on a 24 GB card, which is exactly where QLoRA's memory saving starts to matter.
4. **A real, noisier dataset** (actual chat logs, multilingual) with held-out time periods, to test robustness rather than benchmark accuracy.
5. **Merged-adapter and batched-serving latency** on the target inference hardware.

---

**Citation for the data:** Casanueva, I., Temčinas, T., Gerz, D., Henderson, M., Vulić, I. (2020). *Efficient Intent Detection with Dual Sentence Encoders.* NLP4ConvAI workshop. arXiv:2003.04807.
