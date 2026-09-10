# CICIDS Destination-Centric Entity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make CICIDS DoS Hulk/PortScan/DDoS detection actually work by (1) keying entities on destination IP instead of source IP, and (2) training the transformer and per-entity baselines only on BENIGN-labeled rows.

**Architecture:** `cicids_to_clean.py` swaps which IP column feeds `entity_id` vs. `dst_id` (no renaming — `dst_id` stays a generic-pipeline column name). `new.py` gets one new opt-in CLI flag, `--train_only_label`, that restricts the post-split TRAIN partition to rows matching a given `Label` value before any vocab fitting, sequence building, or model training happens. Val/test partitions and every downstream metric/scoring path are untouched.

**Tech Stack:** Python, pandas, pytest (new — not currently wired up in this repo).

**Spec:** `docs/superpowers/specs/2026-09-10-cicids-destination-centric-entity-design.md`

## Global Constraints

- No changes to `new.py`'s generic column names or shared functions (`dst_id`, `fit_dst_vocab`, `bucket_dst`, `build_vocab_from_buckets`) — these are used by `backend.py`, `frontend.html`, `model.py`, `generate_data.py`, and the R scripts and must keep working unmodified for non-CICIDS data.
- The new training-data filter must be opt-in via CLI flag, defaulting to off, so the synthetic-data pipeline's behavior is byte-for-byte unchanged when the flag isn't passed.
- `split_time_within_groups` and all val/test partition logic are unchanged.
- `fit_entity_nll_stats_train_only` is unchanged.

---

### Task 1: pytest scaffolding + `cicids_to_clean.py` entity/feature swap

**Files:**
- Modify: `requirements.txt` (uncomment `pytest>=7.4.0`)
- Create: `tests/__init__.py` (empty)
- Create: `tests/test_cicids_to_clean.py`
- Modify: `cicids_to_clean.py:20` (docstring), `cicids_to_clean.py:141-148` (`convert()`)

**Interfaces:**
- Consumes: `cicids_to_clean.convert(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame` (existing signature, unchanged).
- Produces: `convert()`'s output DataFrame now has `entity_id` = Destination IP values and `dst_id` = Source IP values (schema/column names unchanged; content swapped). Task 3 depends on this.

- [ ] **Step 1: Uncomment pytest in requirements.txt**

Open `requirements.txt` and change:
```
# pytest>=7.4.0
```
to:
```
pytest>=7.4.0
```

- [ ] **Step 2: Install pytest**

Run: `pip install pytest`

- [ ] **Step 3: Write the failing test**

Create `tests/__init__.py` (empty file).

Create `tests/test_cicids_to_clean.py`:

```python
import pandas as pd
from cicids_to_clean import convert


def _raw_cicids_df():
    return pd.DataFrame({
        "Timestamp": ["03/07/2017 08:55:58", "03/07/2017 08:56:01"],
        "Source IP": ["172.16.0.1", "192.168.10.5"],
        "Destination IP": ["192.168.10.50", "192.168.10.50"],
        "Destination Port": [80, 443],
        "Protocol": [6, 6],
        "Label": ["DoS Hulk", "BENIGN"],
        "Total Length of Fwd Packets": [100, 200],
        "Total Length of Bwd Packets": [50, 60],
    })


def test_entity_id_is_destination_ip():
    out = convert(_raw_cicids_df())
    assert out["entity_id"].tolist() == ["192.168.10.50", "192.168.10.50"]


def test_dst_id_column_holds_source_ip():
    out = convert(_raw_cicids_df())
    assert out["dst_id"].tolist() == ["172.16.0.1", "192.168.10.5"]


def test_label_and_bytes_unaffected_by_swap():
    out = convert(_raw_cicids_df())
    assert out["Label"].tolist() == ["DoS Hulk", "BENIGN"]
    assert out["bytes"].tolist() == [150.0, 260.0]
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest tests/test_cicids_to_clean.py -v`
Expected: `test_entity_id_is_destination_ip` and `test_dst_id_column_holds_source_ip` FAIL (current code has `entity_id` = Source IP, `dst_id` = Destination IP — assertions are reversed relative to current behavior). `test_label_and_bytes_unaffected_by_swap` PASSes already (nothing to change there).

- [ ] **Step 5: Update the docstring**

In `cicids_to_clean.py`, replace line 20:
```python
- entity_id = Source IP, dst_id = Destination IP, Label = Label column
  (empty string if missing) (R lines 94-101).
```
with:
```python
- entity_id = Destination IP, dst_id = Source IP, Label = Label column
  (empty string if missing). Destination is the entity (not R lines
  94-101's original Source-IP mapping) so that per-entity anomaly
  baselines are built from a host's own traffic history, which has a
  genuine benign period, rather than from a single attacker machine's
  history, which does not. See
  docs/superpowers/specs/2026-09-10-cicids-destination-centric-entity-design.md.
```

- [ ] **Step 6: Swap the column assignment in `convert()`**

In `cicids_to_clean.py`, replace lines 141-148:
```python
    out = pd.DataFrame({
        "timestamp": parse_timestamp_utc(df[ts_col]),
        "entity_id": df[src_col].astype(str).str.strip(),
        "event_type": event_type_from_proto_dport(df[proto_col], df[dport_col]),
        "dst_id": df[dst_col].astype(str).str.strip(),
        "bytes": compute_bytes(df),
        "Label": df[label_col].astype(str).str.strip() if label_col else "",
    })
```
with:
```python
    out = pd.DataFrame({
        "timestamp": parse_timestamp_utc(df[ts_col]),
        "entity_id": df[dst_col].astype(str).str.strip(),
        "event_type": event_type_from_proto_dport(df[proto_col], df[dport_col]),
        "dst_id": df[src_col].astype(str).str.strip(),
        "bytes": compute_bytes(df),
        "Label": df[label_col].astype(str).str.strip() if label_col else "",
    })
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_cicids_to_clean.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 8: Commit**

```bash
git add requirements.txt tests/__init__.py tests/test_cicids_to_clean.py cicids_to_clean.py
git commit -m "$(cat <<'EOF'
Key CICIDS entities on destination IP instead of source IP

Nearly all attack traffic (DoS Hulk, PortScan, DDoS, GoldenEye,
slowloris) originates from one source IP with almost no genuine
benign history, so the per-entity baseline was being trained on
attack traffic itself. Destination hosts have real benign traffic
throughout, giving the anomaly model something to actually deviate
from.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016x7aQC5ZstGst1MNcWCQ25
EOF
)"
```

---

### Task 2: `new.py` — restrict TRAIN partition to a given Label via `--train_only_label`

**Files:**
- Modify: `new.py:90` (add CLI arg near existing `--dst_top_n`)
- Modify: `new.py` (new function, placed near `split_time_within_groups`/`split_entity_holdout`, i.e. after line 397)
- Modify: `new.py:831-841` (wire the filter into the split step of `main()`)
- Create: `tests/test_train_filter.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `filter_train_by_label(df_tr: pd.DataFrame, label: str) -> pd.DataFrame`. Task 3 relies on invoking `new.py` with `--train_only_label BENIGN` and on this function being called on `df_tr` (not `df_va`/`df_te_internal`) before vocab fitting.

- [ ] **Step 1: Write the failing test**

Create `tests/test_train_filter.py`:

```python
import pandas as pd
from new import filter_train_by_label


def _df():
    return pd.DataFrame({
        "entity_id": ["a", "a", "a", "b"],
        "Label": ["BENIGN", "DoS Hulk", "BENIGN", "PortScan"],
        "timestamp": [1, 2, 3, 4],
    })


def test_empty_label_returns_df_unchanged():
    df = _df()
    out = filter_train_by_label(df, "")
    pd.testing.assert_frame_equal(out, df)


def test_nonempty_label_keeps_only_matching_rows():
    out = filter_train_by_label(_df(), "BENIGN")
    assert out["Label"].tolist() == ["BENIGN", "BENIGN"]
    assert out["entity_id"].tolist() == ["a", "a"]


def test_filter_returns_copy_not_view():
    df = _df()
    out = filter_train_by_label(df, "BENIGN")
    out["Label"] = "MUTATED"
    assert df["Label"].tolist() == ["BENIGN", "DoS Hulk", "BENIGN", "PortScan"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_filter.py -v`
Expected: FAIL with `ImportError: cannot import name 'filter_train_by_label' from 'new'`.

- [ ] **Step 3: Implement `filter_train_by_label`**

In `new.py`, immediately after `split_time_within_groups` (after line 397, before the `# -----------------------------` / `# Tokenization` section comment at line 399), add:

```python
def filter_train_by_label(df_tr: pd.DataFrame, label: str) -> pd.DataFrame:
    """Restrict a TRAIN partition to rows matching `label` (e.g. "BENIGN").

    Applied only to the TRAIN split, never to val/test, so the model and
    its per-entity NLL baselines are fit exclusively on known-normal
    behavior while evaluation stays an honest mix of normal + anomalous.
    No-op (returns df_tr unchanged) when label is empty/falsy, so callers
    that never pass --train_only_label see no behavior change.
    """
    if not label:
        return df_tr
    return df_tr[df_tr["Label"].astype(str) == label].copy()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_filter.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Add the `--train_only_label` CLI argument**

In `new.py`, in `parse_args()`, immediately after line 91 (`p.add_argument("--bytes_num_buckets", type=int, default=8)`), add:

```python
    p.add_argument("--train_only_label", type=str, default="",
                   help="If set, restrict the TRAIN partition (post-split, pre-vocab-fit) to "
                        "rows where Label equals this value, e.g. 'BENIGN' for CICIDS. "
                        "Val/test partitions are never filtered. Default '' = no filtering "
                        "(synthetic/unlabeled data behavior is unchanged).")
```

- [ ] **Step 6: Wire the filter into `main()`**

In `new.py`, replace lines 831-841:
```python
    # 2) Split TRAIN events BEFORE fitting bucketing/vocab
    if args.split_mode == "time":
        df_tr, df_va, df_te_internal = split_time_within_groups(df, args.train_frac, args.val_frac)
    elif args.split_mode == "entity":
        df_tr, df_va, df_te_internal = split_entity_holdout(df, args.train_frac, args.val_frac, args.seed)
    else:  # time_entity
        df_tr0, df_va0, df_te0 = split_entity_holdout(df, args.train_frac, args.val_frac, args.seed)
        df_tr, _, _ = split_time_within_groups(df_tr0, 0.85, 0.0)
        df_va, _, _ = split_time_within_groups(df_va0, 0.50, 0.0)
        df_te_internal, _, _ = split_time_within_groups(df_te0, 0.50, 0.0)
```
with:
```python
    # 2) Split TRAIN events BEFORE fitting bucketing/vocab
    if args.split_mode == "time":
        df_tr, df_va, df_te_internal = split_time_within_groups(df, args.train_frac, args.val_frac)
    elif args.split_mode == "entity":
        df_tr, df_va, df_te_internal = split_entity_holdout(df, args.train_frac, args.val_frac, args.seed)
    else:  # time_entity
        df_tr0, df_va0, df_te0 = split_entity_holdout(df, args.train_frac, args.val_frac, args.seed)
        df_tr, _, _ = split_time_within_groups(df_tr0, 0.85, 0.0)
        df_va, _, _ = split_time_within_groups(df_va0, 0.50, 0.0)
        df_te_internal, _, _ = split_time_within_groups(df_te0, 0.50, 0.0)

    if args.train_only_label:
        n_before = len(df_tr)
        df_tr = filter_train_by_label(df_tr, args.train_only_label)
        print(f"[train-filter] Label=={args.train_only_label!r}: TRAIN {n_before} -> {len(df_tr)} rows")
        if len(df_tr) == 0:
            raise RuntimeError(
                f"No TRAIN rows remain after --train_only_label={args.train_only_label!r}. "
                "Check that this Label value exists in the TRAIN time window."
            )
```

- [ ] **Step 7: Save metadata field for reproducibility**

In `new.py`, in the `meta` dict (around line 924, right after `"dst_top_n": int(args.dst_top_n),`), add:
```python
        "train_only_label": args.train_only_label,
```

- [ ] **Step 8: Run the full test suite**

Run: `pytest tests/ -v`
Expected: all tests PASS (Task 1's 3 tests + Task 2's 3 tests).

- [ ] **Step 9: Commit**

```bash
git add new.py tests/test_train_filter.py
git commit -m "$(cat <<'EOF'
Add --train_only_label to restrict TRAIN partition by Label

Opt-in (default off, so synthetic-data runs are unaffected). Lets a
labeled eval dataset like CICIDS train the model and per-entity NLL
baselines exclusively on known-benign sequences, instead of learning
attack traffic as part of an entity's "normal" behavior.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016x7aQC5ZstGst1MNcWCQ25
EOF
)"
```

---

### Task 3: Regenerate CICIDS data, retrain, and verify detection quality improved

**Files:**
- Regenerate: `data/cicids_clean.csv` (via `cicids_to_clean.py`)
- Regenerate: `models/log_transformer.keras`, `models/log_transformer_last.keras`, `models/log_transformer_meta.json`
- Regenerate: `outputs/metrics_summary.json`, `outputs/log_alerts_train_val_internal_test.csv`, training curve/history files
- Read (for baseline comparison, do not modify): the `outputs/metrics_summary.json` and alert CSVs already committed at `6e5e1c3e`

**Interfaces:**
- Consumes: Task 1's updated `cicids_to_clean.py` (destination-centric `entity_id`); Task 2's `--train_only_label` flag.
- Produces: final artifacts for this fix — no further tasks depend on this one.

- [ ] **Step 1: Save the current (pre-fix) metrics for comparison**

```bash
cp outputs/metrics_summary.json /tmp/metrics_summary_before.json
```
(Windows/PowerShell equivalent: `Copy-Item outputs/metrics_summary.json $env:TEMP/metrics_summary_before.json`)

- [ ] **Step 2: Regenerate `data/cicids_clean.csv` with the destination-centric entity mapping**

Run:
```bash
python cicids_to_clean.py --in_path data/raw_cicids/CICIDS_Flow.parquet --out_csv data/cicids_clean.csv --verbose
```
Expected stderr output includes `Unique entities:` now showing the destination-IP cardinality (~19,112 per the exploration done during design, not 17,002).

- [ ] **Step 3: Sanity-check the regenerated CSV before spending time on a full retrain**

Run:
```bash
python3 - <<'EOF'
import pandas as pd
df = pd.read_csv("data/cicids_clean.csv")
hulk_victim = df[df["Label"].str.contains("Hulk", case=False, na=False)]["entity_id"].value_counts()
print("Top DoS Hulk victim entities:\n", hulk_victim.head(3))
top = hulk_victim.index[0]
sub = df[df["entity_id"] == top]
print("Victim entity BENIGN fraction:", (sub["Label"] == "BENIGN").mean())
EOF
```
Expected: top entity is `192.168.10.50` (or whichever host CICIDS_Flow.parquet records as the DoS Hulk target) with a BENIGN fraction well above the ~0.6% seen for the old source-IP-keyed attacker entity — confirms the swap took effect before committing to a full retrain.

- [ ] **Step 4: Retrain with the benign-only training filter**

Run (same flags as the model committed at `6e5e1c3e`, per `models/log_transformer_meta.json`, plus the new flag):
```bash
python new.py --train_csv data/cicids_clean.csv --train_format clean \
  --seq_len 16 --step 1 --min_events_per_entity 10 \
  --split_mode time --train_frac 0.7 --val_frac 0.15 \
  --dst_top_n 200 --bytes_num_buckets 8 --epochs 20 \
  --train_only_label BENIGN \
  --best_model_path models/log_transformer.keras --out_dir outputs --plot
```
This is a long-running step (prior CICIDS training runs took on the order of hours per the project's session history) — expect to wait for completion rather than treating a lack of immediate output as failure.

- [ ] **Step 5: Verify the run completed and produced fresh artifacts**

Run:
```bash
ls -la outputs/metrics_summary.json models/log_transformer_meta.json
```
Expected: both files have a modification time after Step 4 started, and `models/log_transformer_meta.json`'s `"train_only_label"` field reads `"BENIGN"`.

- [ ] **Step 6: Compare detection quality before/after**

Run:
```bash
python3 - <<'EOF'
import json
before = json.load(open("/tmp/metrics_summary_before.json"))
after = json.load(open("outputs/metrics_summary.json"))
print("BEFORE detection:", json.dumps(before.get("detection", {}), indent=2))
print("AFTER detection:", json.dumps(after.get("detection", {}), indent=2))
EOF
```
(Adjust the `/tmp` path on Windows to match Step 1's copy location.)

Also inspect per-attack-type AUC the same way the prior session's diagnostic did — recompute ROC-AUC for DoS Hulk, PortScan, and DDoS specifically from `outputs/log_alerts_train_val_internal_test.csv` and/or the full scores if a per-attack-type breakdown script exists from the prior debugging session (check `outputs/` and `experiments/` for a script producing the "per-attack-type NLL breakdown" mentioned in the project's session history before writing a new one).

Expected: DoS Hulk, PortScan, and DDoS AUC move meaningfully above the prior ~0.51, and BENIGN mean_z stays low (not regressing the std=0-floor fix from commit `6e5e1c3e`).

- [ ] **Step 7: If results confirm the fix, commit the regenerated data and artifacts**

```bash
git add data/cicids_clean.csv models/log_transformer.keras models/log_transformer_last.keras \
  models/log_transformer_meta.json outputs/metrics_summary.json \
  outputs/log_alerts_train_val_internal_test.csv outputs/training_history.csv
git status
```
Review `git status` output for any other changed files under `outputs/` (e.g. new/updated timeline PNGs, `training_loss.png`, `training_accuracy.png`) and add those too before committing.

```bash
git commit -m "$(cat <<'EOF'
Retrain CICIDS model with destination-centric entities and
benign-only training data

Verifies the fix from the prior two commits: DoS Hulk/PortScan/DDoS
AUC moves meaningfully above the previous ~0.51 (near-random) baseline
now that per-entity NLL baselines are learned from genuine benign
traffic instead of a single attacker IP's own attack history.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016x7aQC5ZstGst1MNcWCQ25
EOF
)"
```

- [ ] **Step 8: If results do NOT show improvement**

Do not commit. This would mean the entity/split leakage was not the (sole) root cause — stop and re-open investigation with the systematic-debugging skill rather than attempting further fixes on top of this one, per the project's debugging process. Report the before/after numbers to the user.
