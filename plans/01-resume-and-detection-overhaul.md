# Plan: Make Log Anomaly Predictor Resume-Ready (Honest Detection Story)

**Target framing:** Security / ML (Detection) role.
**Guiding principle from the prior "resume-honesty audit":** every number on the
resume and dashboard must be backed by a real artifact produced by a real run.
No aspirational claims (the unsupported **"85.1% on external dataset"** must go).

---

## Phase 0 — Discovery findings (DONE — read before executing any phase)

These were verified against the live code on 2026-06-29. Treat as ground truth;
re-grep if a line number has drifted.

### Current architecture (what actually exists)
- `new.py` (996 lines) — **the real pipeline**. Tokenization (event_type + dst
  bucket + bytes bucket), leak-safe vocab fit on train only, time/entity splits,
  builds a Transformer next-event model, trains, evaluates, writes
  `outputs/metrics_summary.json`, scores anomalies, writes alerts CSV + plots.
  Entry point: `main()` at `new.py:709`. CLI via `argparse` (`parse_args()` `new.py:62`).
- `model.py` (611 lines) — an "improved" Transformer (`build_improved_transformer_model`,
  attention rollout, `get_important_events`). Imported by the API for the
  `/api/explain` endpoint.
- `backend.py` (894 lines) — Flask API. Imports the pipeline from `new.py` and the
  improved model from `model.py` (`backend.py:26-37`). Routes listed below.
- `frontend.html` (1025 lines) — single-page dashboard (vendored `recharts.js`, 528 KB).
- `compare.py`, `malware_tabnet_keras.py` — side experiments (model comparison /
  TabNet). **Not part of the main story.**
- `cicids_into_clean.R` — R adapter: converts CICIDS/ISCX flow CSV →
  clean schema `timestamp, entity_id, event_type, dst_id, bytes, Label`.

### The core honesty problem (verified)
1. **No labeled attacks in the data.** `data/train_data.csv` is 1000 rows, **all
   `Label=Normal`**. With a single class, ROC-AUC / PR-AUC / precision / recall are
   undefined. The detection story currently has *no data behind it*.
2. **`acc=0.0` is not a bug — it's the wrong metric.** `eval_split` (`new.py:878-895`)
   reports `loss/acc/top5_acc/perplexity` of the **next-event language model** over a
   1122-token vocab. That LM accuracy is *not* detection quality, but the dashboard
   surfaces it as `modelAccuracy` (`backend.py:314-356`, `TRAINING_STATE["accuracy"]`).
3. **Detection scoring already exists but is never measured.** `score_anomalies_df`
   builds `scores_all` with the exact columns needed for detection metrics:
   `Label`, `anomaly_score_nll`, `entity_nll_z` (`new.py:667-676`). Nothing computes
   ROC-AUC/PR-AUC from them.
4. **`metrics_summary.json`** (current) shows the toy run: `train.n_events=1000`,
   `n_entities=9`, `val.acc=0.0`, `internal_test.top5_acc≈0.0147`, `perplexity≈1000`.

### Allowed APIs / dependencies (already available — do NOT invent)
- `scikit-learn>=1.3.0` is in `requirements.txt:19`. Use **only** documented sklearn
  functions: `sklearn.metrics.roc_auc_score`, `average_precision_score`,
  `precision_recall_fscore_support`, `roc_curve`, `precision_recall_curve`,
  `confusion_matrix`. (Verify signatures at https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics .)
- `numpy`, `pandas`, `tensorflow/keras` already used throughout `new.py`.
- Helper already present: `compute_nll_only(model, X, y)` (used at `new.py:885,919,931`).

### Flask routes (for Phase 5)
`/api/health` (285), `/api/stats` (295), `/api/events/ingest` (371), `/api/predict`
(397), `/api/alerts` (422), `/api/timeline` (461), `/api/train` (529, runs `new.py`
via subprocess), `/api/explain` (688), `/api/train/status` (732), `/api/upload_csv`
(745), `/api/model/info` (801), `/api/entity/<id>` (820), `/` (865).

### Anti-patterns to guard against (whole plan)
- ❌ Reporting next-event LM accuracy as "model accuracy" / detection quality.
- ❌ Computing ROC-AUC on single-class data (will throw / be meaningless).
- ❌ Inventing sklearn params or metric names not in the docs above.
- ❌ Re-introducing the unsupported "85.1%" string anywhere.
- ❌ Committing `venv/`, `__pycache__/`, large model `.keras` blobs, or `recharts.js`.

---

## Phase 1 — Truth in metrics: wire real detection evaluation

**Goal:** produce honest, defensible detection metrics from the scores that already
exist, and stop mislabeling LM accuracy as model accuracy.

### What to implement
1. Add a function `detection_metrics(scores_df, score_col, positive_labels)` in
   `new.py` (place it next to `score_anomalies_df`, before `main()` at `new.py:709`).
   - Map `scores_df["Label"]` → binary `y_true` (1 if label in `positive_labels`
     i.e. anything not in {"Normal","BENIGN","0",""}, else 0).
   - Guard: if `y_true` has fewer than 2 classes, return
     `{"note": "single-class data — detection metrics undefined", ...}` instead of
     calling sklearn. (This is the current synthetic-data case — fail loud, not fake.)
   - When two classes exist, compute with sklearn:
     `roc_auc = roc_auc_score(y_true, scores)`,
     `pr_auc = average_precision_score(y_true, scores)`,
     and at the operating threshold (`entity_nll_z >= alert_z_thresh`):
     `precision/recall/f1` via `precision_recall_fscore_support(..., average="binary")`,
     plus `detection@FPR=1%` and `@0.1%` by interpolating `roc_curve`.
   - Compute for both `score_col="anomaly_score_nll"` and `score_col="entity_nll_z"`;
     report whichever the README will cite (recommend `entity_nll_z`, the alerting score).
2. Call it in `main()` right after `score_anomalies_df` returns `scores_all`
   (after `new.py:962`), and add the result under a new
   `metrics_summary["detection"]` key before the JSON is written (`new.py:905-917`).
3. **Relabel the LM metrics.** In `metrics_summary` rename the intent: keep
   `loss/perplexity/top5_acc` but document them as *language-model* metrics, and add
   `"note": "acc/top5_acc are next-event LM metrics, NOT detection accuracy"`.
4. In `backend.py`, change the dashboard's headline number: `modelAccuracy`
   (`backend.py:314-356`) should read detection ROC-AUC from `metrics_summary["detection"]`
   when available, falling back to "N/A (untrained / single-class data)" — never 0.0
   presented as accuracy. Update `TRAINING_STATE` plumbing (`backend.py:54-55,660-666`)
   to store `roc_auc`/`pr_auc` instead of `accuracy`/`top5_accuracy` as the headline.

### Documentation references
- Scores source: `new.py:667-676` (`scores_all` columns).
- Eval/summary site: `new.py:878-917`.
- Alerts call returning `scores_all`: `new.py:951-962`.
- Dashboard accuracy plumbing: `backend.py:314-356`, `backend.py:54-55,660-666`.

### Verification checklist
- [ ] `grep -n "roc_auc_score\|average_precision_score" new.py` returns the new code.
- [ ] Run `python new.py --train_csv data/train_data.csv` → `metrics_summary.json`
      now contains a `"detection"` block that explicitly says *single-class* (proving
      the guard works and we did not fabricate a number).
- [ ] `grep -rn "85.1" .` (excluding `venv/`) returns nothing.
- [ ] Dashboard no longer shows `0.0%` labeled as accuracy.

### Anti-pattern guards
- Do not call `roc_auc_score` without the 2-class guard.
- Do not delete the LM metrics — relabel them; perplexity is a legitimate LM metric.

---

## Phase 2 — Get real labeled data (so detection metrics are real)

**Goal:** run the pipeline on a public, labeled intrusion dataset so the `detection`
block in Phase 1 contains genuine ROC-AUC/PR-AUC numbers.

### What to implement
1. Acquire **CICIDS2017** (or a single labeled day, e.g. the
   `MachineLearningCVE` CSVs) — small enough to run locally. Document the exact source
   URL and SHA in `data/README.md`. Do **not** commit the raw CSVs (add to `.gitignore`).
2. Convert with the existing R adapter (do not rewrite it): the pipeline already
   supports `--use_r_adapter` + `--train_format cicids` (`new.py:724-729`,
   `run_r_cicids_adapter` at `new.py:49`). Verify `Rscript` is installed; if R is
   unavailable on the user's machine, add a thin Python fallback adapter that emits the
   identical clean schema (`timestamp, entity_id, event_type, dst_id, bytes, Label`) —
   match `cicids_into_clean.R` field semantics exactly.
3. Run end-to-end and capture the artifacts:
   `python new.py --train_csv <clean.csv> --use_r_adapter --train_format cicids --plot`.
4. Record the resulting `metrics_summary.json["detection"]` numbers — these become the
   ONLY numbers cited on the resume.

### Documentation references
- R adapter invocation: `new.py:49-60`, `new.py:724-729`.
- Clean schema contract: `new.py:17`, `new.py:241`.

### Verification checklist
- [ ] `metrics_summary.json["detection"]` has real `roc_auc` in (0.5, 1.0] with
      `y_true` containing both classes.
- [ ] `train.n_events` ≫ 1000 and labels include attack classes
      (`awk -F, '{print $NF}' clean.csv | sort | uniq -c` shows non-Normal rows).
- [ ] Raw + adapted data are git-ignored; `data/README.md` documents provenance.

### Anti-pattern guards
- Do not commit multi-MB datasets. Do not hand-pick a threshold to inflate metrics —
  report ROC-AUC/PR-AUC (threshold-free) as the headline.

---

## Phase 3 — Repo hygiene & structure

**Goal:** a clean, legible repo a reviewer can navigate in 60 seconds.

### What to implement
1. Stop tracking junk: confirm `venv/` is untracked (it was un-tracked in commit
   `0728155e`), and `git rm -r --cached __pycache__ models/*.keras recharts.js` then
   add them to `.gitignore` (the model `.keras` files are 4.8 MB each; `recharts.js`
   is 528 KB — load Recharts from a CDN in `frontend.html` instead).
2. Add a top-level `LICENSE` (MIT) and `.gitignore` entries for `outputs/`,
   `data/*.csv` (except a tiny sample), `*.keras`.
3. Clarify entry points without a risky rewrite: add a short module docstring header to
   `new.py`/`model.py`/`backend.py` stating role, and a `Makefile` or `run` recipe with
   three commands: `train`, `serve`, `demo`. (`run.sh`/`run.bat` already exist — align them.)
4. Move side experiments (`compare.py`, `malware_tabnet_keras.py`) into
   `experiments/` so the root shows only the main pipeline.

### Verification checklist
- [ ] `git ls-files | grep -E "venv/|__pycache__|\.keras$|recharts.js"` is empty.
- [ ] Root dir lists ≤ ~8 source files; experiments are in `experiments/`.
- [ ] `LICENSE` present; dashboard still renders with CDN Recharts.

### Anti-pattern guards
- Do not delete the trained model from disk (only untrack it). Keep a download/regen path.

---

## Phase 4 — README & resume presentation

**Goal:** the README *is* the resume artifact. One-line README → full story.

### What to implement (README.md sections, in order)
1. **One-paragraph pitch:** "Transformer next-event model that flags anomalous
   security-log sequences by per-entity surprise (NLL z-score)."
2. **Architecture diagram** (ASCII or a committed PNG): logs → tokenize
   (event+dst+bytes) → Transformer next-event LM → per-entity NLL z-score → alerts →
   Flask API → dashboard.
3. **Results table** populated ONLY from Phase 2's real run: ROC-AUC, PR-AUC,
   precision/recall/F1 @ operating point, detection@1%FPR — with the dataset named
   (CICIDS2017) and a note that LM perplexity is an auxiliary metric, not the headline.
4. **How it works** (why next-event modeling detects intrusions; leak-safe vocab fit).
5. **Reproduce:** exact commands from Phases 1–2.
6. **Screenshots** of the dashboard showing the real detection metric.
7. **Honest limitations** section (synthetic-data caveat, single-host scope).

### Verification checklist
- [ ] Every number in the README appears in `outputs/metrics_summary.json`.
- [ ] A new reader can reproduce results by copy-pasting the README commands.
- [ ] No "85.1%" or other un-backed figure anywhere.

### Anti-pattern guards
- Do not cite metrics that aren't in a committed artifact.

---

## Phase 5 — Dashboard honesty (optional but high-impact for demo)

**Goal:** the live demo shows the same honest detection metrics as the README.

### What to implement
1. `/api/model/info` (`backend.py:801`) and `/api/stats` (`backend.py:295`) return the
   `detection` block from `metrics_summary.json`; frontend headline tile shows
   **ROC-AUC**, not next-event accuracy.
2. Add a small "Evaluation" panel: ROC curve + PR curve (data from the new metrics).
3. Keep `/api/train` (`backend.py:529`) but have its status report ROC-AUC on completion
   (`backend.py:660-666`).

### Verification checklist
- [ ] Dashboard tile labeled "Detection ROC-AUC" matches `metrics_summary.json`.
- [ ] No UI element labels next-event accuracy as model accuracy.

---

## Phase 6 — Final verification (run before declaring done)

1. `grep -rn "85.1\|modelAccuracy.*0.0" .` (excl. `venv/`) → empty.
2. Fresh clone test: `git clone` → follow README → `python new.py ...` reproduces the
   `detection` metrics within run-to-run noise.
3. `git ls-files` contains no venv/caches/large blobs (Phase 3 check).
4. README results table === `metrics_summary.json["detection"]` (manual diff).
5. Dashboard launches (`python backend.py`) and headline shows ROC-AUC.

---

## Suggested execution order & sequencing notes
- **Phase 1 and Phase 2 are coupled**: Phase 1 builds the measurement; Phase 2 supplies
  data that makes it non-trivial. Do Phase 1 first (it's safe on current data and proves
  the single-class guard), then Phase 2.
- Phases 3–5 are independent and can be parallelized.
- Each phase is self-contained with its own file:line references so it can run in a
  fresh session (e.g. via `/do`).
