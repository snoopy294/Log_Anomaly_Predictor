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
  breaking ties by precision and then p95 latency. Seeds are fixed at 42, 43, and 44.

Both datasets are testbed traffic and should not be read as production performance.
Large source datasets are ignored by Git; each run manifest records source SHA-256,
row and label counts, exact temporal boundaries, configuration, seed, and environment.

## Results

<!-- RESULTS:START -->
_No leak-safe benchmark has been published yet._
<!-- RESULTS:END -->

The previous label-balanced split results were removed because labels influenced row
assignment. `scripts/render_results.py` is the only supported way to populate this
section, and it reads only `outputs/benchmark_summary.json`. Results should be
published even when they are lower than earlier experiments.

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
| `tests/` | deterministic split, adapter, metric, scoring, and parity fixtures |

The API continues to support the existing routes. Anomaly responses additionally expose
`score`, `threshold`, `top_contributing_feature`, `model_version`, `warm_up_state`, and
`batch_one_latency_ms`. A present but corrupt or dimensionally incompatible bundle
causes startup to fail clearly instead of silently degrading most events to `UNK`.
