# CICIDS Destination-Centric Entity Redesign

## Problem

Phase 2 CICIDS scoring shows DoS Hulk, PortScan, and DDoS detection near
random (AUC ≈ 0.51) despite a correct NLL scoring pipeline (fixed in commit
`6e5e1c3e`).

Root cause, confirmed against `data/cicids_clean.csv`:

- `entity_id` is currently `Source IP` (`cicids_to_clean.py`).
- Of 17,002 entities, only 2 ever carry attack-majority traffic. Essentially
  all DoS Hulk (230,123 rows), PortScan (158,804 rows), DDoS (128,022 of
  128,025 rows), GoldenEye, and slowloris traffic originates from a single
  source IP, `172.16.0.1` — the CICIDS attacker VM.
- That entity's own history is 0.6% BENIGN (3,599 of 558,159 rows). The
  pipeline's per-entity temporal split (`split_time_within_groups`, first
  70%/15%/15% of each entity's own timeline) puts a majority-attack mix into
  that entity's *training* data (e.g. train slice: 59% DoS Hulk, 31%
  PortScan). The model learns this entity's "normal" baseline directly from
  attack traffic, so later attack traffic from the same entity is not
  surprising at scoring time — hence AUC ≈ 0.5.
- 16,992 of 17,002 entities are pure benign always. A source-IP entity
  either never attacks or (for the one attacker VM) never behaves normally —
  there is no meaningful "normal baseline vs. deviation" signal to learn per
  source entity.

This is an architectural problem in entity choice and split/training data
composition, not a scoring-math bug.

## Design

### 1. Entity redefinition

`cicids_to_clean.py`: `entity_id` becomes `Destination IP` (was `Source
IP`). The per-event feature column that held destination IP (`dst_id`)
becomes `src_id`, holding Source IP instead. No other schema change.

Rationale: DoS/DDoS is a victim-side phenomenon. A destination accumulates
traffic from many sources over the full capture window and has a genuine,
data-rich BENIGN baseline even when it is later attacked (e.g. the primary
DoS Hulk victim, `192.168.10.50`, has 582,750 rows with a real if
attack-heavy benign baseline throughout — see Verification below). This
also naturally represents distributed attacks (many sources → one victim)
as a single entity's anomalous timeline, which source-based entities
cannot.

### 2. Token/vocab features

`fit_dst_vocab`/`bucket_dst` are renamed to `fit_src_vocab`/`bucket_src` and
operate on source-IP frequency (top-N + `OTHER`) instead of destination —
same bucketing logic, now describing "who is talking to me" instead of "who
am I talking to." Token strings become
`event_type|SRC=...|BYTES_Qn`. `build_vocab_from_buckets` vocabulary naming
updates accordingly. Bytes quantile bucketing (`fit_bytes_bins`/
`bucket_bytes`) is unchanged.

### 3. Split and training-data filtering

`split_time_within_groups` is unchanged — it still produces per-entity,
time-ordered train/val/test partitions (default 70/15/15). Val and test
remain a full, honest mix of benign and attack rows, identical in
composition to today, so ROC-AUC/PR-AUC/per-attack-type metrics stay
comparable to the currently-committed (broken) baseline.

One filtering step is added, applied only to the train partition, only in
the CICIDS training path (`new.py`, around where `df_tr` is produced): after
splitting, `df_tr = df_tr[df_tr["Label"] == "BENIGN"]` before vocab fitting,
sequence building, or model training.

Effects:
- The transformer next-event model trains only on genuine benign sequences
  per entity — it never learns attack token patterns as "predictable."
- `fit_entity_nll_stats_train_only` is unchanged; it already just consumes
  whatever NLL array it's given, which is now benign-only NLL by
  construction.
- `fit_src_vocab`/`fit_bytes_bins` also fit on this benign-only `df_tr`, so
  the vocabulary reflects normal traffic composition, not attack-dominated
  composition — consistent with defining "normal" from normal data only.

This filtering is CICIDS-specific (applied only in the labeled-eval path).
The generic pipeline (used for synthetic/unlabeled data) is untouched,
since it has no label column to filter on.

### 4. Edge cases

- Entity ordering: `filter_entities(df, min_events_per_entity)` runs on the
  full labeled data *before* splitting, so sparse-entity filtering isn't
  skewed by the benign-only filter. Order: load → `filter_entities` → split
  → benign-filter train partition only.
- Entities that are 100% benign (the large majority) are unaffected by the
  new filter — nothing is removed for them.
- Entities whose train-window benign rows fall below `seq_len + 1` are
  already handled: `make_sequences_from_events` skips any entity with
  `len(ev) < seq_len + 1`. No new logic needed; this now also correctly
  skips entities that lack enough *benign* history.

### 5. Verification plan

Retrain on CICIDS with the new entity/split/filter, then compare against
the metrics committed at `6e5e1c3e` on the same axes:
- Overall ROC-AUC / PR-AUC.
- Per-attack-type AUC for DoS Hulk, PortScan, DDoS specifically (currently
  ≈0.51 each) — success is these moving meaningfully above 0.5.
- BENIGN mean_z stays low (no regression of the std=0-floor fix from the
  prior session, commit `6e5e1c3e`).

## Out of scope

- Any change to the transformer architecture, NLL scoring machinery, or
  `fit_entity_nll_stats_train_only`'s fallback-to-global-stats logic.
- The generic (non-CICIDS) pipeline used for synthetic data.
- A formal, reusable "exclude labeled-anomalous rows from baseline fitting"
  mechanism for future labeled datasets — deferred; this fix is
  CICIDS-specific for now.
