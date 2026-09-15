"""Leak-safe candidate screening and immutable benchmark artifact helpers."""

from __future__ import annotations

import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from benchmark_metrics import frozen_operating_point, is_benign, per_attack_metrics
from datasets import sha256_file, split_manifest
from new import calibrate_threshold


CANDIDATES = ("transformer_nll", "equal_weight_combined", "multi_timescale", "isolation_forest")
REFERENCE_SEEDS = (42, 43, 44)


def multi_timescale_features(events: pd.DataFrame, windows=(8, 32, 128)) -> pd.DataFrame:
    """Causal behavior features at fixed short/medium/long event horizons."""
    ordered = events.sort_values(["entity_id", "timestamp"], kind="stable").copy()
    result = pd.DataFrame(index=ordered.index)
    for entity, group in ordered.groupby("entity_id", sort=False):
        ts = pd.to_datetime(group["timestamp"], utc=True).astype("int64").to_numpy() / 1e9
        token = group["event_type"].astype(str).to_numpy()
        peer = group["dst_id"].astype(str).to_numpy()
        byte = np.log1p(np.clip(group["bytes"].to_numpy(float), 0, None))
        port = group["event_type"].astype(str).str.rsplit(":", n=1).str[-1].to_numpy()
        for width in windows:
            rows = []
            for i in range(len(group)):
                start = max(0, i - width + 1)
                n = i - start + 1
                duration = max(ts[i] - ts[start], 1.0)
                _, counts = np.unique(token[start:i + 1], return_counts=True)
                history = set(peer[max(0, start - width):start])
                current_peers = set(peer[start:i + 1])
                rows.append((n / duration, counts.max() / n,
                             len(current_peers - history) / max(1, len(current_peers)),
                             len(set(port[start:i + 1])), float(byte[start:i + 1].mean()),
                             float(byte[start:i + 1].std()), len(current_peers)))
            names = ("rate", "repetition", "peer_novelty", "port_diversity",
                     "bytes_mean", "bytes_std", "fanout")
            for pos, name in enumerate(names):
                result.loc[group.index, f"w{width}_{name}"] = [row[pos] for row in rows]
    return result.sort_index().reset_index(drop=True)


def fit_isolation_forest(train_features: pd.DataFrame, seed: int) -> IsolationForest:
    model = IsolationForest(n_estimators=300, contamination="auto", random_state=seed, n_jobs=-1)
    model.fit(train_features.to_numpy(float))
    return model


def isolation_scores(model: IsolationForest, features: pd.DataFrame) -> np.ndarray:
    return -model.score_samples(features.to_numpy(float))


def screen_candidates(validation: pd.DataFrame, score_columns: list[str], target_fpr=.01,
                      latency_p95_ms: dict[str, float] | None = None) -> dict:
    """Select solely on development labels; thresholds use benign rows only."""
    latency_p95_ms = latency_p95_ms or {}
    benign = is_benign(validation["Label"])
    rows = []
    for name in score_columns:
        scores = validation[name].to_numpy(float)
        threshold = calibrate_threshold(scores[benign], target_fpr)
        overall = frozen_operating_point(validation["Label"], scores, threshold)
        attacks = per_attack_metrics(validation["Label"], scores, threshold)
        macro = float(np.mean([v["recall"] for v in attacks.values()])) if attacks else 0.0
        rows.append({"variant": name, "threshold": threshold, "macro_attack_family_recall": macro,
                     "precision": overall["precision"], "fpr": overall["observed_test_fpr"],
                     "p95_latency_ms": float(latency_p95_ms.get(name, np.inf))})
    eligible = [r for r in rows if r["fpr"] <= target_fpr + 1e-12]
    if not eligible:
        raise RuntimeError("no candidate met the validation FPR constraint")
    winner = sorted(eligible, key=lambda r: (-r["macro_attack_family_recall"],
                                              -r["precision"], r["p95_latency_ms"], r["variant"]))[0]
    return {"selection_population": "development only", "target_fpr": target_fpr,
            "candidates": rows, "winner": winner}


def improvement_gate(current: dict, candidate: dict) -> bool:
    return bool(candidate["macro_attack_family_recall"] - current["macro_attack_family_recall"] >= .02
                and candidate["fpr"] - current["fpr"] <= .01
                and candidate["p95_latency_ms"] <= current["p95_latency_ms"] * 1.10)


def aggregate_seed_metrics(runs: list[dict]) -> dict:
    keys = ("recall", "precision", "f1", "observed_test_fpr", "benign_alerts_per_10000_events")
    aggregate = {}
    for key in keys:
        values = np.asarray([run["frozen_operating_point"][key] for run in runs], float)
        aggregate[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
    return {"seeds": [run["seed"] for run in runs], "aggregate": aggregate, "runs": runs}


def write_run_manifest(run_dir: str | Path, *, dataset_name: str, source_files: list[str | Path],
                       splits: dict[str, pd.DataFrame], config: dict, seed: int) -> Path:
    target = Path(run_dir)
    target.mkdir(parents=True, exist_ok=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "dataset": dataset_name,
        "sources": [{"path": str(path), "sha256": sha256_file(path),
                     "bytes": Path(path).stat().st_size} for path in source_files],
        "splits": split_manifest(splits), "config": config, "seed": int(seed),
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__,
                        "tensorflow": _tensorflow_version(), "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES")},
    }
    path = target / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _tensorflow_version():
    try:
        import tensorflow as tf
        return tf.__version__
    except Exception:
        return None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass
