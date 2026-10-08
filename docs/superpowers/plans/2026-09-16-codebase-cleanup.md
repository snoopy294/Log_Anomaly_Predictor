# Codebase Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove dead/unreachable code, delete broken/orphaned files, tidy repo hygiene items, and trim the pre-detector-bundle legacy fallback paths out of `backend.py`/`model.py`, without changing any currently-tested behavior.

**Architecture:** No new components. Every task is a deletion or simplification of existing code, verified by (a) the existing pytest suite staying green throughout, and (b) a small number of new tests that pin down behavior the legacy code paths never had a test for (e.g. "server starts cleanly with no bundle present").

**Tech Stack:** Python, pytest, Flask (`backend.py`), Keras/TensorFlow (`model.py`, `new.py`).

**Spec:** None — scope was agreed via clarifying questions at planning time (this session), not a separate spec document. The agreed scope, captured as Global Constraints below, stands in for a spec.

## Global Constraints

- Do not change the behavior of any code path currently exercised by `tests/`. Run `python -m pytest tests -q` after every task; it must stay fully green.
- Do not touch `detector_bundle.py`, `benchmark.py`, `benchmark_metrics.py`, `datasets.py`, or `cicids_to_clean.py` — they are the reviewed, currently-correct core and are out of scope.
- `frontend.html` calls these backend routes and their shapes must keep working: `/api/events/ingest`, `/api/explain`, `/api/stats`, `/api/timeline`, `/api/train`, `/api/train/status`, `/api/alerts`. It does **not** call `/api/upload_csv`, `/api/model/info`, `/api/entity/<id>`, `/api/predict`, or `/api/health` (confirmed by grep of `frontend.html`), so those may be simplified freely as long as they don't 500.
- Every task that deletes code must first `grep` the whole repo (excluding `venv/`) to confirm nothing else references the symbol being removed, and that grep's output must be quoted in the task (already done below from investigation — re-run it live before deleting, since files may have moved).
- Never delete `models/*.keras`, `outputs/*.json` that are `git`-tracked, or anything under `data/`.
- Each task commits independently with `git add <specific files>` (never `git add -A`).

---

### Task 1: Delete unreachable dead code in `new.py`'s `split_time_within_groups`

**Files:**
- Modify: `new.py:415-443`
- Test: `tests/test_split_time_within_groups.py` (existing — no new test needed, this is a pure dead-code deletion behind an early `return`)

**Interfaces:**
- Consumes: nothing new.
- Produces: `split_time_within_groups(df, train_frac, val_frac)` keeps its exact current signature and behavior (it already always executes only the `return chronological_split(...)` line; everything after it is unreachable).

- [ ] **Step 1: Confirm the code after the early `return` is truly unreachable and unused**

Run: `grep -n "def split_time_within_groups" -A 30 new.py`

Expected: shows the function returns on line 3 of its body (`return chronological_split(df, train_frac, val_frac)`), followed by a second docstring and ~24 more lines of loop code that can never execute.

- [ ] **Step 2: Run the existing test file to record a passing baseline**

Run: `python -m pytest tests/test_split_time_within_groups.py -v`
Expected: all 5 tests PASS (this proves the behavior we're about to touch is dead — deleting unreachable lines cannot change these results).

- [ ] **Step 3: Delete the unreachable block**

In `new.py`, replace:

```python
def split_time_within_groups(df: pd.DataFrame, train_frac: float, val_frac: float):
    """Globally chronological and label-blind (legacy public function name)."""
    # Historical name retained for API compatibility. The split is now global,
    # chronological, and never reads Label.
    return chronological_split(df, train_frac, val_frac)

    """Chronologically split each (entity_id, Label) sub-stream 70/15/15.

    Grouping by entity alone (not label) would put a whole attack type
    entirely into one split whenever that attack's traffic is clustered
    in time relative to an entity's other traffic (e.g. CICIDS, where
    each day is a different attack against the same victim host) — the
    attack would never reach val/test, so it could never be evaluated.
    Historical implementation notes (unreachable; retained for old source maps):
    time-ordering intact while guaranteeing it gets a proportional slice
    of train/val/test regardless of when it falls in the entity's overall
    timeline.
    """
    tr_parts, va_parts, te_parts = [], [], []
    group_cols = ["entity_id", "Label"] if "Label" in df.columns else ["entity_id"]
    for _, g in df.groupby(group_cols, sort=False):
        g2 = g.sort_values("timestamp")
        n = len(g2)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        tr_parts.append(g2.iloc[:n_train])
        va_parts.append(g2.iloc[n_train:n_train + n_val])
        te_parts.append(g2.iloc[n_train + n_val:])
    return pd.concat(tr_parts), pd.concat(va_parts), pd.concat(te_parts)
```

with:

```python
def split_time_within_groups(df: pd.DataFrame, train_frac: float, val_frac: float):
    """Globally chronological and label-blind (legacy public function name)."""
    # Historical name retained for API compatibility. The split is now global,
    # chronological, and never reads Label.
    return chronological_split(df, train_frac, val_frac)
```

- [ ] **Step 4: Run tests to verify nothing changed**

Run: `python -m pytest tests/test_split_time_within_groups.py tests/test_train_filter.py -v`
Expected: same PASS results as Step 2 (identical test count, all PASS).

- [ ] **Step 5: Commit**

```bash
git add new.py
git commit -m "chore: delete unreachable dead code in split_time_within_groups"
```

---

### Task 2: Delete broken `experiments/compare.py`

**Files:**
- Delete: `experiments/compare.py`
- Delete: `experiments/__pycache__/` (stale bytecode for the deleted file)

**Interfaces:**
- Consumes: nothing.
- Produces: nothing (file removed).

- [ ] **Step 1: Confirm the file cannot run and nothing imports it**

Run: `grep -n "^from\|^import" experiments/compare.py`

Expected output includes:
```
from logs_transformer_anomaly_crossdataset import build_transformer_next_event_model as build_original
from logs_transformer_improved import build_improved_transformer_model as build_improved
```

Run: `grep -rl "logs_transformer_anomaly_crossdataset\|logs_transformer_improved" --include="*.py" . 2>/dev/null | grep -v venv`

Expected: only `./experiments/compare.py` — no module named `logs_transformer_anomaly_crossdataset.py` or `logs_transformer_improved.py` exists anywhere in the repo, so this script has always been unrunnable (`ModuleNotFoundError` on import).

Run: `grep -rln "experiments.compare\|experiments/compare\|import compare" --include="*.py" --include="*.md" . 2>/dev/null | grep -v venv`

Expected: no output (nothing else references this module).

- [ ] **Step 2: Run the full test suite as a baseline**

Run: `python -m pytest tests -q`
Expected: all tests PASS (this file has no tests exercising it — confirms it's safe to delete).

- [ ] **Step 3: Delete the file and its stale bytecode cache**

```bash
git rm experiments/compare.py
rm -rf experiments/__pycache__
```

- [ ] **Step 4: Run the full test suite again**

Run: `python -m pytest tests -q`
Expected: identical PASS count to Step 2.

- [ ] **Step 5: Commit**

```bash
git add -u experiments/compare.py
git commit -m "chore: delete experiments/compare.py (imports nonexistent modules, never runnable)"
```

---

### Task 3: Delete dead `config.yaml`

**Files:**
- Delete: `config.yaml`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing (file removed).

- [ ] **Step 1: Confirm nothing reads config.yaml**

Run: `grep -rln "config\.yaml\|import yaml\|yaml\.safe_load\|yaml\.load" --include="*.py" --include="*.md" --include="*.sh" --include="*.bat" . 2>/dev/null | grep -v venv`

Expected: no output. `config.yaml`'s settings (model architecture, training hyperparameters, alert thresholds, API host/port, etc.) all live instead as CLI defaults in `new.py`'s `parse_args()` and hardcoded values in `backend.py`'s `CONFIG` dict — `config.yaml` was never wired to either.

- [ ] **Step 2: Run the full test suite as a baseline**

Run: `python -m pytest tests -q`
Expected: all tests PASS.

- [ ] **Step 3: Delete the file**

```bash
git rm config.yaml
```

- [ ] **Step 4: Run the full test suite again**

Run: `python -m pytest tests -q`
Expected: identical PASS count (no test reads `config.yaml`).

- [ ] **Step 5: Commit**

```bash
git add -u config.yaml
git commit -m "chore: delete config.yaml (dead config, never read by any code)"
```

---

### Task 4: Remove stale local `tmp/` scratch artifacts

**Files:**
- Delete (local filesystem only, not git-tracked): `tmp/smoke_bundle/`, `tmp/smoke_model.keras`, `tmp/smoke_model_last.keras`, `tmp/smoke_model_meta.json`, `tmp/smoke_outputs/`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing. `tmp/` itself is gitignored (see `.gitignore`) and is used as a scratch directory by `tests/test_detector_bundle.py`'s `tempfile.TemporaryDirectory(dir="tmp")` — the directory must continue to exist, only its leftover manual-run contents are removed.

- [ ] **Step 1: Confirm these paths are untracked, not part of any fixture**

Run: `git status --short tmp/` (expect no output — nothing tracked under `tmp/`)
Run: `grep -n 'dir="tmp"' tests/test_detector_bundle.py` (expect the one `tempfile.TemporaryDirectory(dir="tmp")` call — confirms `tmp/` itself must stay, but confirms tests create their own throwaway subdirectories via `tempfile`, not `tmp/smoke_*`)

- [ ] **Step 2: Delete the stale contents, keep the directory**

```bash
rm -rf tmp/smoke_bundle tmp/smoke_model.keras tmp/smoke_model_last.keras tmp/smoke_model_meta.json tmp/smoke_outputs
ls tmp/
```

Expected: `tmp/` still exists (possibly empty, or containing only pytest's own temp dirs from a prior run).

- [ ] **Step 3: Run the full test suite to confirm `tmp/`-dependent tests still pass**

Run: `python -m pytest tests/test_detector_bundle.py -q`
Expected: all tests PASS (the bundle round-trip test creates and cleans up its own `tempfile.TemporaryDirectory(dir="tmp")`).

- [ ] **Step 4: No commit needed**

These paths are gitignored, so `git status --short` shows nothing changed. Skip the commit step for this task.

---

### Task 5: Remove dead code from `model.py`

**Files:**
- Modify: `model.py:248-402` (delete `extract_temporal_features`, `extract_entity_features`, `EnsemblePredictor`, `compute_attention_rollout`; keep `get_important_events`)
- Modify: `model.py:409-600` (delete `AnomalyDetectorAPI`, `WarmUpSchedule`, `create_callbacks`)
- Modify: `backend.py:31-35` (drop the now-deleted `compute_attention_rollout` import)
- Test: `tests/test_model_dead_code_removed.py` (new)

**Interfaces:**
- Consumes: nothing new.
- Produces: `model.py` still exports `MultiHeadSelfAttention`, `FeedForward`, `TransformerBlock`, `build_improved_transformer_model(vocab_size, seq_len, d_model=128, num_layers=4, num_heads=8, d_ff=512, dropout=0.2, lr=1e-4, weight_decay=1e-5, label_smoothing=0.1)`, and `get_important_events(model, X, seq_idx=0)` — unchanged signatures, used by `backend.py`'s legacy `/api/train` code (removed in Task 7) and `/api/explain` respectively.

- [ ] **Step 1: Confirm every symbol being deleted has zero external references**

Run: `grep -rn "EnsemblePredictor\|AnomalyDetectorAPI\|WarmUpSchedule\|create_callbacks\|extract_temporal_features\|extract_entity_features\|compute_attention_rollout" --include="*.py" . 2>/dev/null | grep -v "^\./model.py" | grep -v venv`

Expected: only one line, `./backend.py:33:    compute_attention_rollout,` (a dead import — `backend.py` imports it but never calls it; confirmed by `grep -n "compute_attention_rollout" backend.py` showing only the import line, no call site). All other symbols have zero references outside `model.py`.

- [ ] **Step 2: Write the failing test — module no longer exposes the dead symbols**

Create `tests/test_model_dead_code_removed.py`:

```python
import model


def test_dead_symbols_removed_from_model_module():
    for name in ("EnsemblePredictor", "AnomalyDetectorAPI", "WarmUpSchedule",
                 "create_callbacks", "extract_temporal_features",
                 "extract_entity_features", "compute_attention_rollout"):
        assert not hasattr(model, name), f"{name} should have been deleted"


def test_still_exposes_what_backend_and_explain_need():
    assert callable(model.build_improved_transformer_model)
    assert callable(model.get_important_events)
```

- [ ] **Step 3: Run it to verify it fails**

Run: `python -m pytest tests/test_model_dead_code_removed.py -v`
Expected: `test_dead_symbols_removed_from_model_module` FAILs (the symbols still exist); `test_still_exposes_what_backend_and_explain_need` PASSes.

- [ ] **Step 4: Delete the dead code from `model.py`**

Delete these blocks entirely from `model.py` (identified by their section comments and `def`/`class` headers read from the file):

1. `extract_temporal_features(df)` (the function under the `# FEATURE ENGINEERING IMPROVEMENTS` comment, before `extract_entity_features`).
2. `extract_entity_features(df)` (immediately after it).
3. `class EnsemblePredictor:` and its full body (under `# ENSEMBLE & UNCERTAINTY ESTIMATION`), including `__init__`, `predict`, and `predict_with_confidence`.
4. `compute_attention_rollout(model, X, layer_names=None)` (under `# EXPLAINABILITY`, directly above `get_important_events` — **keep `get_important_events`**, it is used by `backend.py`'s `/api/explain`).
5. `class AnomalyDetectorAPI:` and its full body (under `# REAL-TIME INFERENCE API`).
6. `class WarmUpSchedule(keras.optimizers.schedules.LearningRateSchedule):` and its full body.
7. `create_callbacks(model_path, log_dir="logs", patience=5, reduce_lr_patience=3)` and its full body.

Leave the `# FEATURE ENGINEERING IMPROVEMENTS`, `# ENSEMBLE & UNCERTAINTY ESTIMATION`, `# EXPLAINABILITY`, `# REAL-TIME INFERENCE API`, and `# TRAINING IMPROVEMENTS` section-comment banners removed along with their now-empty sections (don't leave orphaned banner comments with nothing under them). The file should read, after `MultiHeadSelfAttention`/`FeedForward`/`TransformerBlock`/`build_improved_transformer_model`, directly into:

```python
# ============================================
# EXPLAINABILITY
# ============================================

def get_important_events(model: keras.Model, X: np.ndarray, seq_idx: int = 0):
    """Get most important events in a sequence using gradient-based attribution"""
    X_single = X[seq_idx:seq_idx+1]

    with tf.GradientTape() as tape:
        # Convert to tensor and watch
        X_tensor = tf.constant(X_single, dtype=tf.float32)
        tape.watch(X_tensor)

        # Get embeddings
        emb_layer = model.get_layer("tok_emb")
        embeddings = emb_layer(X_single)

        # Forward pass through rest of model
        # This is simplified - in practice you'd need the full forward pass
        predictions = model(X_single)
        pred_class = tf.argmax(predictions[0])
        pred_score = predictions[0, pred_class]

    # Get gradients
    gradients = tape.gradient(pred_score, embeddings)

    # Compute importance scores (gradient * input)
    importance = tf.reduce_sum(tf.abs(gradients), axis=-1).numpy()

    return importance[0]


if __name__ == "__main__":
    print("Improved transformer model module loaded.")
    print("Key improvements:")
    print("  - Enhanced multi-head attention with better normalization")
    print("  - Deeper architecture with residual connections")
    print("  - Explainability tools")
```

Also delete the now-unused `import pickle` (was only used by the deleted `AnomalyDetectorAPI`) and the now-unused `from typing import Dict, Tuple, List, Optional` if nothing else in the file uses `Tuple`/`List`/`Optional` (`get_important_events` and `build_improved_transformer_model` use no typing names) — replace with no typing import, or keep `Dict` only if still referenced. Run `grep -n "Dict\|Tuple\|List\[\|Optional\[" model.py` after deleting to confirm which, if any, typing names remain in use, and trim the import line to match exactly.

- [ ] **Step 5: Remove the dead import from `backend.py`**

In `backend.py`, replace:

```python
from model import (
    build_improved_transformer_model,
    compute_attention_rollout,
    get_important_events
)
```

with:

```python
from model import (
    build_improved_transformer_model,
    get_important_events
)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_model_dead_code_removed.py -v`
Expected: both tests PASS.

Run: `python -m pytest tests -q`
Expected: full suite still green (in particular `tests/test_detector_bundle.py`, which imports `backend`).

- [ ] **Step 7: Commit**

```bash
git add model.py backend.py tests/test_model_dead_code_removed.py
git commit -m "chore: delete unused classes/functions from model.py (EnsemblePredictor, AnomalyDetectorAPI, WarmUpSchedule, create_callbacks, extract_*_features, placeholder compute_attention_rollout)"
```

---

### Task 6: Simplify `backend.py`'s `load_model_and_config` to bundle-only

**Files:**
- Modify: `backend.py:71-139` (the `CONFIG` dict and `load_model_and_config`)
- Test: `tests/test_backend_bundle_only.py` (new)

**Interfaces:**
- Consumes: `detector_bundle.load_detector_bundle(bundle_path)` (unchanged, from Task list's out-of-scope `detector_bundle.py`).
- Produces: `load_model_and_config()` keeps the same name/no-args signature and still sets the module globals `MODEL, MODEL_META, ENTITY_STATS, VOCAB, TOKENIZER, DETECTOR_RUNTIME`. New behavior: if `models/detector_bundle/bundle.json` does not exist, it logs a message and leaves all of those globals as `None` — it no longer tries to load a raw `.keras` file from `CONFIG["model_path"]`.

- [ ] **Step 1: Confirm the legacy branch has no other callers and CONFIG keys are only used here**

Run: `grep -n 'CONFIG\["model_path"\]\|CONFIG\["meta_path"\]\|CONFIG\["stats_path"\]' backend.py`

Expected: all three appear only inside `load_model_and_config` (lines 112-122 in the current file) — no other function reads these CONFIG keys.

- [ ] **Step 2: Write the failing test**

Create `tests/test_backend_bundle_only.py`:

```python
import backend


def test_missing_bundle_leaves_everything_none(monkeypatch, tmp_path):
    monkeypatch.setattr(backend, "MODEL", None)
    monkeypatch.setattr(backend, "MODEL_META", None)
    monkeypatch.setattr(backend, "ENTITY_STATS", None)
    monkeypatch.setattr(backend, "VOCAB", None)
    monkeypatch.setattr(backend, "TOKENIZER", None)
    monkeypatch.setattr(backend, "DETECTOR_RUNTIME", None)
    monkeypatch.setitem(backend.CONFIG, "bundle_path", str(tmp_path / "no_bundle_here"))

    backend.load_model_and_config()

    assert backend.MODEL is None
    assert backend.MODEL_META is None
    assert backend.DETECTOR_RUNTIME is None
    assert backend.TOKENIZER is None


def test_config_no_longer_has_legacy_model_paths():
    assert "model_path" not in backend.CONFIG
    assert "meta_path" not in backend.CONFIG
    assert "stats_path" not in backend.CONFIG
```

- [ ] **Step 3: Run it to verify it fails**

Run: `python -m pytest tests/test_backend_bundle_only.py -v`
Expected: `test_config_no_longer_has_legacy_model_paths` FAILs (`"model_path"` is still a key in `CONFIG`). `test_missing_bundle_leaves_everything_none` may currently error or fail too, since the legacy branch attempts `keras.models.load_model(CONFIG["model_path"])` against a nonexistent default path and silently no-ops via the `try/except`, but leaves `VOCAB`/`TOKENIZER` unset in a way that doesn't match the assertions — run it and confirm at least one assertion fails.

- [ ] **Step 4: Simplify `CONFIG` and `load_model_and_config`**

Replace:

```python
# Configuration
CONFIG = {
    "model_path": "models/log_transformer.keras",
    "meta_path": "models/log_transformer_meta.json",
    "stats_path": "outputs/entity_stats.csv",
    "out_dir": "outputs",
    "max_buffer_size": 1000,
    "alert_threshold": 3.0,
    "bundle_path": os.environ.get("DETECTOR_BUNDLE_PATH", "models/detector_bundle"),
}
```

with:

```python
# Configuration
CONFIG = {
    "out_dir": "outputs",
    "max_buffer_size": 1000,
    "alert_threshold": 3.0,
    "bundle_path": os.environ.get("DETECTOR_BUNDLE_PATH", "models/detector_bundle"),
}
```

Replace the body of `load_model_and_config`:

```python
def load_model_and_config():
    """Load the trained model and configuration"""
    global MODEL, MODEL_META, ENTITY_STATS, VOCAB, TOKENIZER, DETECTOR_RUNTIME
    
    try:
        bundle_path = CONFIG["bundle_path"]
        if os.path.exists(os.path.join(bundle_path, "bundle.json")):
            MODEL_META, MODEL = load_detector_bundle(bundle_path)
            DETECTOR_RUNTIME = DetectorRuntime(MODEL_META, MODEL)
            VOCAB = MODEL_META["vocabulary"]
            TOKENIZER = DETECTOR_RUNTIME.token_map.copy()
            TOKENIZER.update({"PAD": 0, "UNK": 1})
            ENTITY_STATS = pd.DataFrame(MODEL_META["entity_nll_stats"])
            CONFIG["alert_threshold"] = float(MODEL_META["threshold"])
            print(f"+ Detector bundle {MODEL_META['model_version']} loaded from {bundle_path}")
            return
        if os.path.exists(CONFIG["model_path"]):
            MODEL = keras.models.load_model(CONFIG["model_path"])
            print(f"+ Model loaded from {CONFIG['model_path']}")
        
        if os.path.exists(CONFIG["meta_path"]):
            with open(CONFIG["meta_path"], 'r') as f:
                MODEL_META = json.load(f)
            print(f"+ Metadata loaded")
        
        if os.path.exists(CONFIG["stats_path"]):
            ENTITY_STATS = pd.read_csv(CONFIG["stats_path"])
            print(f"+ Entity stats loaded")
        
        # Build vocab and tokenizer from meta
        if MODEL_META:
            VOCAB = MODEL_META.get("vocabulary")
            if not VOCAB:
                raise RuntimeError("legacy metadata has no full vocabulary; retrain to create a detector bundle")
            # Build reverse tokenizer
            TOKENIZER = {v: i + 2 for i, v in enumerate(VOCAB)}
            TOKENIZER["PAD"] = 0
            TOKENIZER["UNK"] = 1
            print(f"+ Tokenizer initialized with {len(TOKENIZER)} tokens")
            
    except Exception as e:
        print(f"! Error loading model: {e}")
        if os.path.exists(os.path.join(CONFIG["bundle_path"], "bundle.json")):
            raise RuntimeError(f"detector bundle failed validation: {e}") from e
```

with:

```python
def load_model_and_config():
    """Load the versioned detector bundle. Detectors are trained and calibrated
    exclusively by new.py; there is no legacy raw-.keras loading path."""
    global MODEL, MODEL_META, ENTITY_STATS, VOCAB, TOKENIZER, DETECTOR_RUNTIME

    bundle_path = CONFIG["bundle_path"]
    if not os.path.exists(os.path.join(bundle_path, "bundle.json")):
        print(f"! No detector bundle found at {bundle_path}. Train one with new.py first.")
        return

    try:
        MODEL_META, MODEL = load_detector_bundle(bundle_path)
        DETECTOR_RUNTIME = DetectorRuntime(MODEL_META, MODEL)
        VOCAB = MODEL_META["vocabulary"]
        TOKENIZER = DETECTOR_RUNTIME.token_map.copy()
        TOKENIZER.update({"PAD": 0, "UNK": 1})
        ENTITY_STATS = pd.DataFrame(MODEL_META["entity_nll_stats"])
        CONFIG["alert_threshold"] = float(MODEL_META["threshold"])
        print(f"+ Detector bundle {MODEL_META['model_version']} loaded from {bundle_path}")
    except Exception as e:
        raise RuntimeError(f"detector bundle failed validation: {e}") from e
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_backend_bundle_only.py -v`
Expected: both tests PASS.

Run: `python -m pytest tests -q`
Expected: full suite still green.

- [ ] **Step 6: Commit**

```bash
git add backend.py tests/test_backend_bundle_only.py
git commit -m "refactor: simplify backend.py load_model_and_config to bundle-only (drop legacy raw-.keras fallback path)"
```

---

### Task 7: Neuter `/api/train`'s legacy retraining implementation

**Files:**
- Modify: `backend.py:569-744` (the whole `/api/train` route body)
- Test: `tests/test_backend_bundle_only.py` (extend)

**Interfaces:**
- Consumes: nothing.
- Produces: `POST /api/train` keeps its route path and JSON error shape (`{"success": false, "message": ...}` with HTTP 409) that `frontend.html` already handles for the "bundle present" case — it now returns that same response unconditionally, since retraining is exclusively `new.py`'s job. `GET /api/train/status` is untouched (still reads `TRAINING_STATE`, which now never changes from its initial idle values).

**Do this task before Task 8**: `run_training` (deleted here) is the only caller of `detect_csv_format`/`run_r_cicids_adapter` (deleted in Task 8), so deleting this task's code first keeps every intermediate commit's `backend.py` free of dangling references.

- [ ] **Step 1: Confirm `TRAINING_STATE`'s shape used by `/api/train/status` (must not change)**

Run: `grep -n "TRAINING_STATE\[" backend.py`

Expected: `/api/train/status` (around the current line 794-801) reads `is_training`, `progress`, `accuracy`, `top5_accuracy`, `last_trained`, `error` — these five keys plus `roc_auc`/`pr_auc` (unused by the route but present in the dict) must still exist in `TRAINING_STATE`'s initial value after this task.

- [ ] **Step 2: Write the failing test**

Add to `tests/test_backend_bundle_only.py`:

```python
def test_train_endpoint_always_rejects_with_guidance_to_new_py(monkeypatch):
    client = backend.app.test_client()

    monkeypatch.setattr(backend, "DETECTOR_RUNTIME", None)
    response_no_bundle = client.post("/api/train", json={})
    assert response_no_bundle.status_code == 409
    assert "new.py" in response_no_bundle.get_json()["message"]

    monkeypatch.setattr(backend, "DETECTOR_RUNTIME", object())
    response_with_bundle = client.post("/api/train", json={})
    assert response_with_bundle.status_code == 409
    assert "new.py" in response_with_bundle.get_json()["message"]


def test_train_status_route_unaffected():
    client = backend.app.test_client()
    response = client.get("/api/train/status")
    assert response.status_code == 200
    body = response.get_json()
    assert set(body.keys()) == {"is_training", "progress", "accuracy",
                                "top5_accuracy", "last_trained", "error"}
```

- [ ] **Step 3: Run it to verify it fails**

Run: `python -m pytest tests/test_backend_bundle_only.py::test_train_endpoint_always_rejects_with_guidance_to_new_py -v`
Expected: FAIL for the `DETECTOR_RUNTIME is None` case — currently that case starts a background training thread and returns `{"success": True, "message": "Model training started", ...}` with HTTP 200, not a 409 with "new.py" in the message.

- [ ] **Step 4: Replace the `/api/train` route body**

Replace the entire route, from `@app.route('/api/train', methods=['POST'])` through its closing `except Exception as e: return jsonify({"error": str(e)}), 500` (currently lines 569-744), with:

```python
@app.route('/api/train', methods=['POST'])
def train_model():
    """Versioned detectors are trained and calibrated exclusively by new.py,
    then loaded here as a detector bundle. This API never trains a model."""
    return jsonify({
        "success": False,
        "message": "Versioned detectors must be trained and calibrated by new.py, then loaded as one bundle."
    }), 409
```

- [ ] **Step 5: Confirm `TRAINING_STATE` is unchanged and the now-unused `threading`/`queue`-adjacent imports are still needed elsewhere**

Run: `grep -n "^import threading\|^import queue\|threading\.\|queue\." backend.py`

Expected: `threading` and `queue` are still used elsewhere in `backend.py` (`EVENT_BUFFER`/`ALERT_QUEUE` machinery, unrelated to training) — confirm before removing any import lines; if either import is now genuinely unused, remove it, otherwise leave both.

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_backend_bundle_only.py -v`
Expected: `test_train_endpoint_always_rejects_with_guidance_to_new_py` and `test_train_status_route_unaffected` PASS.

Run: `python -m pytest tests -q`
Expected: full suite still green.

- [ ] **Step 7: Commit**

```bash
git add backend.py tests/test_backend_bundle_only.py
git commit -m "refactor: remove backend.py's legacy /api/train retraining implementation, always defer to new.py"
```

---

### Task 8: Remove `/api/upload_csv` and its helpers from `backend.py`

**Files:**
- Modify: `backend.py:534-566` (delete `run_r_cicids_adapter`, `detect_csv_format`)
- Modify: `backend.py:804-857` (delete the `/api/upload_csv` route)
- Test: `tests/test_backend_bundle_only.py` (extend, from Task 6)

**Interfaces:**
- Consumes: nothing.
- Produces: nothing — these are pure deletions. `detect_csv_format` and `run_r_cicids_adapter` existed solely to support `/api/upload_csv` feeding `/api/train`'s retraining flow, which Task 7 already removed; `frontend.html` never calls `/api/upload_csv` (confirmed by grep in the planning session).

- [ ] **Step 1: Confirm no other route or test depends on these**

Run: `grep -n "run_r_cicids_adapter\|detect_csv_format\|upload_csv" backend.py`

Expected: `run_r_cicids_adapter` used only inside itself and `/api/upload_csv`'s body; `detect_csv_format` used only inside `/api/upload_csv`'s body. Task 7 already deleted `/api/train`'s `run_training` closure (the only other caller of `detect_csv_format`), so both helpers are now referenced solely by `/api/upload_csv` and safe to delete together with it in this task.

Run: `grep -n "upload_csv" frontend.html`
Expected: no output (already confirmed in planning).

- [ ] **Step 2: Write the failing test**

Add to `tests/test_backend_bundle_only.py`:

```python
def test_upload_csv_route_removed():
    client = backend.app.test_client()
    response = client.post("/api/upload_csv")
    assert response.status_code == 404
```

- [ ] **Step 3: Run it to verify it fails**

Run: `python -m pytest tests/test_backend_bundle_only.py::test_upload_csv_route_removed -v`
Expected: FAIL — currently returns 400 ("No file provided"), not 404, because the route still exists.

- [ ] **Step 4: Delete `run_r_cicids_adapter`, `detect_csv_format`, and the `/api/upload_csv` route**

Delete this whole block from `backend.py`:

```python
def run_r_cicids_adapter(in_csv, out_csv):
    """Run cicids_into_clean.R to convert CICIDS format to clean format"""
    import subprocess
    r_script = os.path.join(os.path.dirname(__file__), "cicids_into_clean.R")
    if not os.path.exists(r_script):
        raise FileNotFoundError(f"R adapter script not found: {r_script}")
    
    # Try Rscript from PATH
    rscript_bin = "Rscript"
    cmd = [rscript_bin, r_script, "--in_csv", in_csv, "--out_csv", out_csv, "--verbose"]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"R script failed (exit {result.returncode}):\n{result.stderr}")
    
    if not os.path.exists(out_csv):
        raise RuntimeError(f"R script ran but output file not created: {out_csv}")
    
    print(f"R adapter output: {result.stdout}")
    return out_csv


def detect_csv_format(csv_path):
    """Detect if a CSV is CICIDS raw format or clean format"""
    df_peek = pd.read_csv(csv_path, nrows=3)
    # Strip whitespace from column names (CICIDS files often have leading spaces)
    cols = set(c.strip() for c in df_peek.columns)
    
    if {"timestamp", "entity_id", "event_type"}.issubset(cols):
        return "clean"
    if {"Timestamp", "Source IP", "Destination IP", "Destination Port", "Protocol"}.issubset(cols):
        return "cicids"
    return "unknown"
```

and this whole block:

```python
@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    """Upload a CSV file for training. Accepts CICIDS or clean format.
    CICIDS files are auto-converted via cicids_into_clean.R."""
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400
    
    try:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Save the uploaded file
        upload_path = os.path.join(data_dir, "uploaded_raw.csv")
        file.save(upload_path)
        
        # Detect format
        fmt = detect_csv_format(upload_path)
        
        if fmt == "cicids":
            # Run R adapter to convert to clean format
            clean_path = os.path.join(data_dir, "train_data.csv")
            run_r_cicids_adapter(upload_path, clean_path)
            row_count = sum(1 for _ in open(clean_path)) - 1
            
            return jsonify({
                "success": True,
                "message": f"CICIDS dataset preprocessed with cicids_into_clean.R and saved ({row_count} rows). Ready to train!",
                "format": "cicids",
                "rows": row_count,
            })
        elif fmt == "clean":
            import shutil
            train_path = os.path.join(data_dir, "train_data.csv")
            shutil.copy2(upload_path, train_path)
            row_count = sum(1 for _ in open(train_path)) - 1
            
            return jsonify({
                "success": True,
                "message": f"Clean-format CSV saved ({row_count} rows). Ready to train!",
                "format": "clean",
                "rows": row_count,
            })
        else:
            return jsonify({
                "success": False,
                "error": "Unrecognized CSV format. Only CICIDS datasets (Timestamp, Source IP, Destination IP, Destination Port, Protocol) or clean format (timestamp, entity_id, event_type) are supported."
            }), 400
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_backend_bundle_only.py -v`
Expected: `test_upload_csv_route_removed` PASSes.

Run: `python -m pytest tests -q`
Expected: full suite still green.

- [ ] **Step 6: Commit**

```bash
git add backend.py tests/test_backend_bundle_only.py
git commit -m "chore: remove /api/upload_csv route and its R-adapter helpers (unused by frontend, fed only the removed legacy training path)"
```

---

### Task 9: Simplify `compute_anomaly_score` to `DetectorRuntime`-only

**Files:**
- Modify: `backend.py:243-303`
- Test: `tests/test_backend_bundle_only.py` (extend)

**Interfaces:**
- Consumes: `DETECTOR_RUNTIME.score_event(event, update=True)` (unchanged, from `detector_bundle.py`, out of scope).
- Produces: `compute_anomaly_score(entity_id, actual_event)` keeps its exact signature and return type (`dict | None`). New behavior: if `DETECTOR_RUNTIME` is `None`, it returns `None` immediately instead of falling back to a hand-rolled NLL/z-score computation against `MODEL`/`ENTITY_STATS`.

- [ ] **Step 1: Confirm nothing besides this function reads `PERFORMANCE_METRICS`/`ALERT_QUEUE` in the branch being deleted**

Run: `grep -n "PERFORMANCE_METRICS\[" backend.py`

Expected: the legacy branch inside `compute_anomaly_score` (currently lines ~262-301) mutates `PERFORMANCE_METRICS["inference_latency_samples"]`, `PERFORMANCE_METRICS["total_events_scored"]`, and `PERFORMANCE_METRICS["anomalies_flagged"]`, and pushes onto `ALERT_QUEUE` — these are also updated elsewhere (`predict_next_event`, `/api/stats`) so removing this one branch does not remove the dict/queue themselves, only this duplicate writer.

- [ ] **Step 2: Write the failing test**

Add to `tests/test_backend_bundle_only.py`:

```python
def test_compute_anomaly_score_returns_none_without_detector_runtime(monkeypatch):
    monkeypatch.setattr(backend, "DETECTOR_RUNTIME", None)
    monkeypatch.setattr(backend, "MODEL", object())  # legacy branch would have used this
    monkeypatch.setattr(backend, "MODEL_META", {"seq_len": 4})
    result = backend.compute_anomaly_score("some_entity", {"event_type": "tcp:80"})
    assert result is None
```

- [ ] **Step 3: Run it to verify it fails**

Run: `python -m pytest tests/test_backend_bundle_only.py::test_compute_anomaly_score_returns_none_without_detector_runtime -v`
Expected: FAIL or ERROR — the legacy branch tries to call `.predict()` on the plain `object()` standing in for `MODEL`, raising `AttributeError` instead of returning `None` cleanly.

- [ ] **Step 4: Replace `compute_anomaly_score`**

Replace:

```python
def compute_anomaly_score(entity_id: str, actual_event: Dict) -> Optional[Dict]:
    """Compute anomaly score for actual next event"""
    if DETECTOR_RUNTIME is not None:
        event = dict(actual_event)
        event["entity_id"] = entity_id
        return DETECTOR_RUNTIME.score_event(event, update=True)
    if MODEL is None or MODEL_META is None:
        return None
    
    buffer = get_entity_buffer(entity_id)
    seq_len = MODEL_META.get("seq_len", 64)
    
    if len(buffer) < seq_len:
        return None
    
    # Get last seq_len events
    recent = buffer[-seq_len:]
    X = np.array([e["token_id"] for e in recent], dtype=np.int32).reshape(1, -1)
    
    # Predict with latency tracking
    t0 = time.perf_counter()
    probs = MODEL.predict(X, verbose=0)[0]
    latency_ms = (time.perf_counter() - t0) * 1000
    PERFORMANCE_METRICS["inference_latency_samples"].append(latency_ms)
    if len(PERFORMANCE_METRICS["inference_latency_samples"]) > 100:
        PERFORMANCE_METRICS["inference_latency_samples"] = PERFORMANCE_METRICS["inference_latency_samples"][-100:]
    PERFORMANCE_METRICS["total_events_scored"] += 1
    
    # Get actual token ID
    actual_token = tokenize_event(actual_event)
    actual_prob = probs[actual_token] if actual_token < len(probs) else 1e-9
    
    # Compute NLL
    nll = -np.log(np.clip(actual_prob, 1e-9, 1.0))
    
    # Compute z-score if we have entity stats
    z_score = None
    if ENTITY_STATS is not None:
        entity_stat = ENTITY_STATS[ENTITY_STATS["entity_id"] == entity_id]
        if len(entity_stat) > 0:
            mean_nll = float(entity_stat["mean_nll"].iloc[0])
            std_nll = float(entity_stat["std_nll"].iloc[0])
            z_score = (nll - mean_nll) / (std_nll + 1e-6)
    
    is_anomaly = (z_score and z_score > CONFIG["alert_threshold"]) or nll > 5.0
    
    result = {
        "entity_id": entity_id,
        "timestamp": datetime.now().isoformat(),
        "nll": float(nll),
        "z_score": float(z_score) if z_score is not None else None,
        "actual_probability": float(actual_prob),
        "is_anomaly": bool(is_anomaly),
    }
    
    # Add to alert queue if anomaly
    if is_anomaly:
        ALERT_QUEUE.put(result)
        PERFORMANCE_METRICS["anomalies_flagged"] += 1
    
    return result
```

with:

```python
def compute_anomaly_score(entity_id: str, actual_event: Dict) -> Optional[Dict]:
    """Compute anomaly score for actual next event via the shared DetectorRuntime.
    Returns None if no detector bundle is loaded — there is no legacy fallback."""
    if DETECTOR_RUNTIME is None:
        return None
    event = dict(actual_event)
    event["entity_id"] = entity_id
    return DETECTOR_RUNTIME.score_event(event, update=True)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_backend_bundle_only.py -v`
Expected: `test_compute_anomaly_score_returns_none_without_detector_runtime` PASSes.

Run: `python -m pytest tests -q`
Expected: full suite still green — in particular `tests/test_detector_bundle.py::test_flask_and_offline_use_identical_runtime_scoring`, which exercises `compute_anomaly_score` through `/api/events/ingest` with `DETECTOR_RUNTIME` set.

- [ ] **Step 6: Commit**

```bash
git add backend.py tests/test_backend_bundle_only.py
git commit -m "refactor: simplify compute_anomaly_score to DetectorRuntime-only, delete duplicate manual NLL/z-score fallback"
```

---

### Task 10: Simplify `tokenize_event` to `DetectorRuntime`-only

**Files:**
- Modify: `backend.py:146-157`
- Test: `tests/test_backend_bundle_only.py` (extend)

**Interfaces:**
- Consumes: `DETECTOR_RUNTIME.token_id(event)` (unchanged, out of scope).
- Produces: `tokenize_event(event)` keeps its exact signature and return type (`int`). New behavior: if `DETECTOR_RUNTIME` is `None`, it returns `1` (`UNK`) directly instead of building a token string and looking it up in a manually-constructed `TOKENIZER` dict (which, after Task 6, is only ever populated alongside `DETECTOR_RUNTIME` — the two are never in different states).

- [ ] **Step 1: Confirm `TOKENIZER` and `DETECTOR_RUNTIME` are always set together after Task 6**

Run: `grep -n "TOKENIZER =" backend.py`

Expected: the only assignment left (after Task 6) is inside the bundle branch of `load_model_and_config`, in the same block that also sets `DETECTOR_RUNTIME` — confirming the two globals can no longer diverge, so `tokenize_event`'s manual-`TOKENIZER`-lookup branch is now dead code reachable only when both are `None` anyway.

- [ ] **Step 2: Write the failing test**

Add to `tests/test_backend_bundle_only.py`:

```python
def test_tokenize_event_returns_unk_without_detector_runtime(monkeypatch):
    monkeypatch.setattr(backend, "DETECTOR_RUNTIME", None)
    monkeypatch.setattr(backend, "TOKENIZER", {"some|token|string": 42, "PAD": 0, "UNK": 1})
    assert backend.tokenize_event({"event_type": "tcp:80", "dst_id": "x", "bytes_bucket": 0}) == 1
```

- [ ] **Step 3: Run it to verify it fails**

Run: `python -m pytest tests/test_backend_bundle_only.py::test_tokenize_event_returns_unk_without_detector_runtime -v`
Expected: PASS already by coincidence (the manual branch also returns `1` for an unmapped token string) — this step is a sanity check, not a true red step. Confirm this, then proceed: the goal of this task is deleting the now-provably-dead manual-lookup code path, not changing observable behavior.

- [ ] **Step 4: Replace `tokenize_event`**

Replace:

```python
def tokenize_event(event: Dict) -> int:
    """Convert event to token ID"""
    if DETECTOR_RUNTIME is not None:
        return DETECTOR_RUNTIME.token_id(event)
    # Build token string
    event_type = event.get("event_type", "UNK")
    dst = event.get("dst_id", "OTHER")
    bytes_bucket = event.get("bytes_bucket", 0)
    
    token_str = f"{event_type}|DST={dst}|BYTES_Q{bytes_bucket}"
    
    return TOKENIZER.get(token_str, 1)  # 1 = UNK
```

with:

```python
def tokenize_event(event: Dict) -> int:
    """Convert event to token ID via the shared DetectorRuntime. Returns UNK (1)
    if no detector bundle is loaded."""
    if DETECTOR_RUNTIME is None:
        return 1
    return DETECTOR_RUNTIME.token_id(event)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_backend_bundle_only.py -v`
Expected: all tests in this file PASS.

Run: `python -m pytest tests -q`
Expected: full suite still green.

- [ ] **Step 6: Commit**

```bash
git add backend.py tests/test_backend_bundle_only.py
git commit -m "refactor: simplify tokenize_event to DetectorRuntime-only, delete dead manual-TOKENIZER-lookup branch"
```

---

### Task 11: Final verification pass

**Files:**
- None modified — verification only.

**Interfaces:**
- Consumes: everything from Tasks 1-10.
- Produces: a clean bill of health for the cleanup.

- [ ] **Step 1: Run the full test suite one more time**

Run: `python -m pytest tests -q`
Expected: all tests PASS, zero errors, zero warnings about missing imports.

- [ ] **Step 2: Confirm every deleted symbol is truly gone repo-wide**

Run: `grep -rn "EnsemblePredictor\|AnomalyDetectorAPI\|WarmUpSchedule\|create_callbacks\|extract_temporal_features\|extract_entity_features\|compute_attention_rollout\|run_r_cicids_adapter\|detect_csv_format\|logs_transformer_anomaly_crossdataset\|logs_transformer_improved" --include="*.py" . 2>/dev/null | grep -v venv`

Expected: no output.

Run: `grep -rln "config\.yaml" --include="*.py" --include="*.md" --include="*.sh" --include="*.bat" . 2>/dev/null | grep -v venv`

Expected: no output.

- [ ] **Step 3: Confirm the app still boots cleanly with no bundle present**

Run: `python -c "import backend; backend.CONFIG['bundle_path'] = 'no/such/path'; backend.load_model_and_config(); print('OK, MODEL is', backend.MODEL)"`

Expected: prints `! No detector bundle found at no/such/path. Train one with new.py first.` followed by `OK, MODEL is None` — no traceback.

- [ ] **Step 4: Confirm the app still boots cleanly with the real bundle present (if `models/detector_bundle/bundle.json` exists locally)**

Run: `python -c "import backend; backend.load_model_and_config(); print('OK, MODEL is', backend.MODEL is not None)"`

Expected: prints `+ Detector bundle ... loaded from models/detector_bundle` followed by `OK, MODEL is True` — no traceback. (Skip this step if no bundle is present on this machine; Step 3 already covers the no-bundle path.)

- [ ] **Step 5: Confirm `git status` is clean and `git log` shows one commit per task**

Run: `git status --short`
Expected: empty (everything committed; `tmp/` cleanup from Task 4 is gitignored so it never shows up).

Run: `git log --oneline -10`
Expected: shows the 9 commits from Tasks 1, 2, 3, 5, 6, 7, 8, 9, 10 (in that execution order; Task 4 has no commit) each with a clear message.

- [ ] **Step 6: No commit for this task** — it is verification-only.
