# Log_Anomaly_Predictor

Unsupervised intrusion detection over network-flow logs. A Transformer next-event
language model learns what each host's normal event sequences look like, trained on
benign traffic only. Each sliding window of events is then scored against that host's
own baseline, which is also fit on benign training windows. The score combines two things:

- how *surprising* the next event is to the model (NLL)
- how *unusual* the window's shape is: token/destination diversity, repetition, byte statistics

Surprise alone misses repetitive attacks such as port scans, slow DoS and web brute
force, because a flood of identical events is highly *predictable*. The window-shape
statistics catch exactly those.

```
flow logs ─► tokenize (event | dst | bytes bucket) ─► per-entity sliding windows
                                                          │
                     ┌────────────────────────────────────┴───────────────┐
                     ▼                                                    ▼
        Transformer next-event LM ─► NLL                       window statistics
                     └──────────────► robust z vs. per-entity TRAIN baseline ◄┘
                                               │
                              combo_score = mean |z| ─► threshold @ 1% FPR on benign val
                                               │
                                     alerts CSV ─► Flask API ─► dashboard
```

## Results — CICIDS2017

<!-- RESULTS:START -->
Evaluated on the held-out **internal_test** split: 363,542 windows, 83,475 of them attacks. Operating threshold: quantile of 278,371 benign val windows @ FPR=0.01; test labels are never used to pick it.

| Metric | combo_score (alerting) | entity_nll_z (LM surprise only) |
|---|---:|---:|
| ROC-AUC | 0.9963 | 0.4950 |
| PR-AUC | 0.9902 | 0.4142 |
| Detection rate @ 1% FPR | 99.2% | 0.0% |
| Detection rate @ 0.1% FPR | 70.9% | 0.0% |
| Precision @ operating threshold | 0.983 | 0.065 |
| Recall @ operating threshold | 0.986 | 0.003 |
| F1 @ operating threshold | 0.985 | 0.006 |
| Operating threshold | 3.366 | 3.533 |

Per-attack ROC-AUC (each attack type vs. all benign test windows):

| Attack | Test windows | combo_score | entity_nll_z |
|---|---:|---:|---:|
| DoS Hulk | 34,519 | 0.999 | 0.438 |
| PortScan | 23,822 | 0.998 | 0.154 |
| DDoS | 19,204 | 0.998 | 0.977 |
| DoS GoldenEye | 1,545 | 0.999 | 0.838 |
| FTP-Patator | 1,191 | 0.997 | 0.805 |
| DoS slowloris | 870 | 0.997 | 0.466 |
| SSH-Patator | 870 | 0.996 | 0.617 |
| DoS Slowhttptest | 826 | 0.999 | 0.456 |
| Bot | 298 | 0.349 | 0.510 |
| Web Attack – Brute Force | 227 | 0.999 | 0.140 |
| Web Attack – XSS | 99 | 1.000 | 0.106 |
| Web Attack – Sql Injection | 4 | 0.998 | 0.627 |

_Auxiliary (not a detection metric): next-event LM accuracy on internal_test = 0.390, top-5 = 0.762._
<!-- RESULTS:END -->

The numbers above are generated, never hand-edited. `scripts/render_results.py` writes
them from `outputs/metrics_summary.json`.

### Limitations

- **Bot is not detected.** Its traffic blends into the host's normal pattern instead of
  arriving as a burst, so neither score separates it from benign.
- **Some attack classes have very few test windows** (SQL injection especially), so their
  per-class AUCs are noisy.
- **The window features were designed after looking at held-out results.** An earlier
  analysis showed where NLL alone failed. The operating threshold never sees test
  labels, but a second labeled dataset would give a cleaner estimate of generalization.
- **CICIDS attacks are concentrated bursts against a few hosts.** That favors
  window-shape statistics, and results on low-and-slow real-world traffic may be lower.

## Repository layout

| Path | Role |
|---|---|
| `new.py` | Main pipeline: tokenize, split, train, score, calibrate, write alerts and metrics |
| `backend.py` | Flask API that serves the dashboard (`frontend.html`) |
| `model.py` | Explainability helpers used by `/api/explain` |
| `cicids_into_clean.R`, `cicids_to_clean.py` | CICIDS2017 → clean event schema converters (R, plus a Python fallback) |
| `scripts/` | Utilities: `render_results.py` (README tables), `generate_data.py` (synthetic sample) |
| `tests/` | pytest suite |
| `outputs/` | Committed metrics and plots from the reference run |
| `experiments/` | Side experiments, not part of the pipeline |

Common tasks: `make train` (small synthetic sample), `make train-cicids`, `make results`,
`make test`, `make serve`.

## Reproduce

Training runs on a GPU. Data splits are chronological within each (entity, label)
group. The model is trained on BENIGN windows only.

```bash
python new.py --train_csv data/cicids_clean.csv --train_format clean \
  --seq_len 16 --step 1 --min_events_per_entity 10 \
  --split_mode time --train_frac 0.7 --val_frac 0.15 \
  --dst_top_n 200 --bytes_num_buckets 8 --epochs 20 \
  --train_only_label BENIGN --alert_score combo_score --target_fpr 0.01 --plot
python scripts/render_results.py
```
