| Base precision | Rank | Seeds | Test acc (%) | Macro-F1 (%) | Trainable params (M) | Peak train mem (GB) | Train time (min) | Batch-1 latency p50 / p95 (ms) | Throughput, batch 64 (ex/s) | Est. train compute (TFLOP) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 4 | 3 | 93.93 ± 0.15 | 93.93 | 2.27 | 6.04 | 4.2 ± 0.1 | 77.8 / 92.7 | 305 | 535 |
| fp16 | 8 | 3 | 94.05 ± 0.15 | 94.04 | 4.47 | 6.08 | 4.2 ± 0.0 | 79.0 / 93.1 | 308 | 538 |
| fp16 | 16 | 3 | 94.29 ± 0.09 | 94.28 | 8.87 | 6.16 | 4.2 ± 0.0 | 77.7 / 92.1 | 306 | 545 |
| fp16 | 32 | 3 | 94.26 ± 0.18 | 94.26 | 17.67 | 6.33 | 4.3 ± 0.0 | 77.8 / 92.4 | 304 | 558 |
| nf4 | 4 | 3 | 93.96 ± 0.06 | 93.96 | 2.27 | 5.54 | 4.3 ± 0.0 | 93.7 / 107.1 | 320 | 535 |
| nf4 | 8 | 3 | 94.15 ± 0.05 | 94.15 | 4.47 | 5.58 | 4.2 ± 0.0 | 94.7 / 109.6 | 323 | 538 |
| nf4 | 16 | 3 | 94.17 ± 0.21 | 94.17 | 8.87 | 5.66 | 4.3 ± 0.1 | 97.0 / 110.5 | 321 | 545 |
| nf4 | 32 | 3 | 94.46 ± 0.05 | 94.45 | 17.67 | 5.82 | 4.3 ± 0.0 | 95.3 / 108.7 | 318 | 558 |
| TF-IDF + LogReg (CPU) | – | 1 | 89.03 | 89.04 | – | – | 0.4 (CPU) | – | – | – |

Values are mean ± 1 sample SD across seeds. Test-set sampling error of a single accuracy is about ±0.43 pp (binomial SE, n = 3080). Configurations share the same test items, so differences between them are judged with the paired bootstrap in results/audit/paired_differences.csv.
Hardware and software: Tesla T4 | torch 2.11.0+cu130 | Qwen2.5-0.5B on Banking77.
Train/test overlap: 24 exact and 124 near-duplicate test items (TF-IDF cosine >= 0.9); see results/audit/.
