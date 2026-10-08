# Log Anomaly Predictor

An unsupervised network-flow detector built around a Transformer next-event model and
benign-trained behavioral baselines. Offline evaluation and the Flask API consume the
same versioned detector bundle: vocabulary, destination buckets, byte bins, model,
entity/global NLL statistics, behavioral baselines, score definition, threshold, and
suppression policy are all hashed and loaded together.

## Benchmark protocol

- **CICIDS2017 development:** train and early-stop on benign July 3 traffic; calibrate
  on benign July 4 traffic; use July 4 attack labels only to select among the fixed
  candidate set. After selection, freeze preprocessing, score, threshold, and
  suppression and evaluate July 5–7 once.
- **UNSW-NB15 retrained:** use the full time-bearing release and a global chronological
  60/20/20 split, with benign-only fitting and benign calibration.
- **UNSW-NB15 zero-shot:** apply the frozen CICIDS bundle and threshold without any
  UNSW tuning. This is a stress test, not a selection input.
- Candidate selection maximizes macro attack-family recall at validation FPR ≤ 1%,
  breaking ties by precision and then p95 latency. Per-candidate latency is not
  measured, so that last tie-break is currently inactive (recorded as `null`); the
  shipped model's batch-1 latency is measured separately and reported below.
  Seeds are fixed at 42, 43, and 44.

Both datasets are testbed traffic and should not be read as production performance.
Large source datasets are ignored by Git; each run manifest records source SHA-256,
row and label counts, exact temporal boundaries, configuration, seed, and environment.

## Results

<!-- RESULTS:START -->
All headline values use thresholds frozen on held-out benign calibration traffic.

| Dataset / experiment | Seeds | Recall | Macro family recall | Precision | F1 | Test FPR | Benign alerts / 10k | Batch-1 p95 latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CICIDS2017 (cicids_days split) | 42, 43, 44 | 98.42 ± 1.09% | 67.54 ± 0.78% | 92.12 ± 0.52% | 95.16 ± 0.58% | 3.86 ± 0.28% | 386.18 ± 28.36 | 32.38 ± 2.99 |

**Per-attack-family recall: CICIDS2017 (cicids_days split)** (frozen threshold; mean ± std over runs)

| Attack family | Test samples | Recall |
|---|---:|---:|
| DoS Hulk | 230,123 | 99.80 ± 0.08% |
| PortScan | 158,804 | 99.75 ± 0.00% |
| DDoS | 128,025 | 96.21 ± 4.47% |
| DoS GoldenEye | 10,293 | 98.24 ± 0.32% |
| DoS slowloris | 5,796 | 93.52 ± 1.71% |
| DoS Slowhttptest | 5,499 | 96.91 ± 0.21% |
| Bot | 1,952 | 0.44 ± 0.02% |
| Web Attack – Brute Force | 1,507 | 92.02 ± 0.36% |
| Web Attack – XSS | 652 | 91.10 ± 0.25% |
| Infiltration | 23 | 0.00 ± 0.00% |
| Web Attack – Sql Injection | 21 | 0.00 ± 0.00% |
| Heartbleed | 11 | 42.42 ± 4.29% |

**Candidate selection** (CICIDS2017 July 4 development labels only; each threshold at 1% development FPR)

| Candidate | Macro family recall | Precision | Dev FPR |
|---|---:|---:|---:|
| transformer_nll | 0.00% | 0.00% | 1.00% |
| equal_weight_combined (selected) | 84.86% | 76.27% | 1.00% |
| multi_timescale | 0.00% | 0.00% | 1.00% |
| isolation_forest | 0.13% | 0.46% | 1.00% |

**Ablation** (seed 42 bundle; each score's threshold calibrated on July 4 benign traffic, applied unchanged to July 5–7)

| Score | Recall | Macro family recall | Precision | Test FPR |
|---|---:|---:|---:|---:|
| combo_score | 96.88% | 66.45% | 92.09% | 3.70% |
| combo_without_nll | 99.24% | 68.76% | 94.57% | 2.53% |
| transformer_nll_z | 0.36% | 0.21% | 16.44% | 0.80% |

Diagnostic ROC-derived TPR values, when present in artifacts, are not frozen-threshold results.
<!-- RESULTS:END -->

The previous label-balanced split results were removed because labels influenced row
assignment. `scripts/render_results.py` is the only supported way to populate this
section, and it reads only `outputs/benchmark_summary.json`. Results should be
published even when they are lower than earlier experiments.

## Findings & limitations

- **Headline recall is dominated by volumetric attacks.** DoS, DDoS, and PortScan make up
  most attack rows on July 5–7 and are caught almost entirely. Low-and-slow families
  (Bot, Infiltration, SQL injection) are close to undetected. The macro family recall column
  weights every family equally and is the fairer single number. See the per-family table.
- **The Transformer is not doing the detecting.** Its per-entity NLL z-score alone reaches
  under 0.4% test recall at its frozen 1%-FPR threshold (ROC-AUC ≈ 0.70), and 0% macro family
  recall on the development day. Detection comes from `combo_score`, a mean of robust z-scores
  over seven behavioral window features plus NLL as an eighth. The ablation table shows what
  removing the NLL feature does.
- **The NLL feature hurts:** removing it raises macro family recall from 66.45% to 68.76%.
- **The 1% FPR target does not hold on later days.** Thresholds are calibrated on July 4
  benign traffic; on July 5–7 benign traffic the observed FPR is about 3.9%. Benign traffic
  drifts day to day, and a single calibration day is not enough to bound it.
- **Selection order.** The development-only candidate selection was run after the 3-seed test
  results were first published. It independently chose the same score (`equal_weight_combined`
  is the bundle's `combo_score`) at the same threshold (4.29799 for seed 42), so the published
  test numbers are the selected candidate's. The pre-registered order (select, then test) was
  not followed, though, and the selection was run on seed 42 only.
- **Alert suppression is off.** A 300-second per-entity cooldown removes 99.97% of alerts but
  also about 97% of recall (most attacks arrive as dense bursts), so it is disabled by default.
- **Testbed data.** CICIDS2017 is generated lab traffic. These numbers are not a claim about
  production performance.

## Reproduce

Install dependencies and run the deterministic fixture suite:

```bash
python -m pip install -r requirements.txt
python -m pytest tests -q
```

Run the CICIDS protocol for each reference seed (large dataset and GPU required):

```bash
python new.py --train_csv data/cicids_clean.csv --train_format clean \
  --split_mode cicids_days --train_only_label BENIGN --seed 42 \
  --seq_len 16 --step 1 --target_fpr 0.01 \
  --out_dir outputs/runs/cicids2017/seed-42 \
  --bundle_dir models/detector_bundle
```

For retrained UNSW-NB15, use the full release containing `Stime`, `srcip`, and `dstip`:

```bash
python new.py --train_csv data/unsw_nb15_full.csv --train_format unsw \
  --split_mode time --train_frac 0.60 --val_frac 0.20 \
  --train_only_label BENIGN --seed 42 --target_fpr 0.01 \
  --out_dir outputs/runs/unsw_nb15_retrained/seed-42
```

Repeat both retrained commands with seeds 43 and 44. Keep the CICIDS bundle unchanged
for the zero-shot UNSW run:

```bash
python scripts/score_zero_shot_unsw.py --csv data/unsw_nb15_full.csv \
  --bundle models/detector_bundle \
  --out outputs/runs/unsw_nb15_zero_shot/cicids-frozen
```

Select the winning score variant on the CICIDS development split (July 4) using a
frozen bundle — this is the candidate-screening step referenced above, using
development labels only:

```bash
python scripts/select_candidate.py --csv data/cicids_clean.csv \
  --bundle models/detector_bundle \
  --out outputs/runs/candidate_selection/seed-42
```

Measure what the Transformer contributes by re-scoring development and test with the frozen
bundle, with and without its NLL feature:

```bash
python scripts/ablate_nll.py --csv data/cicids_clean.csv \
  --bundle models/detector_bundle --out outputs/runs/ablation/seed-42
```

Consolidate reviewed artifacts into the README data source:

```bash
python scripts/consolidate_benchmarks.py \
  --run "CICIDS2017 (cicids_days split)=outputs/runs/cicids2017/seed-42/metrics_summary.json" \
  --run "CICIDS2017 (cicids_days split)=outputs/runs/cicids2017/seed-43/metrics_summary.json" \
  --run "CICIDS2017 (cicids_days split)=outputs/runs/cicids2017/seed-44/metrics_summary.json" \
  --selection outputs/runs/candidate_selection/seed-42-rerun/candidate_selection.json \
  --ablation outputs/runs/ablation/seed-42/ablation.json
```

Generate the README table only after consolidating complete
run outputs:

```bash
python scripts/render_results.py --summary outputs/benchmark_summary.json
```

## Repository layout

| Path | Role |
|---|---|
| `datasets.py` | UNSW adapter, CICIDS day contract, global label-blind splits, hashes |
| `benchmark.py` | fixed candidate screening, multi-timescale features, seed aggregation, manifests |
| `benchmark_metrics.py` | frozen operating point, per-family, suppression, and latency metrics |
| `detector_bundle.py` | hashed bundle serialization and shared stateful scoring runtime |
| `new.py` | model training, calibration, evaluation, and bundle export |
| `backend.py` | compatible Flask routes backed by the shared detector runtime |
| `scripts/select_candidate.py` | runs `benchmark.py`'s candidate screening against a real development split |
| `scripts/ablate_nll.py` | frozen-bundle ablation: combo score with vs. without the Transformer NLL feature |
| `scripts/consolidate_benchmarks.py` | builds `outputs/benchmark_summary.json` from run artifacts |
| `scripts/render_results.py` | the only writer of the README results block |
| `tests/` | deterministic split, adapter, metric, scoring, and parity fixtures |

The API continues to support the existing routes. Anomaly responses additionally expose
`score`, `threshold`, `top_contributing_feature`, `model_version`, `warm_up_state`, and
`batch_one_latency_ms`. A present but corrupt or dimensionally incompatible bundle
causes startup to fail clearly instead of silently degrading most events to `UNK`.
