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


def _windowed(cumulative: np.ndarray, width: int) -> np.ndarray:
    """Trailing window sum ending at each row, from a running cumulative sum.

    Row i covers indices [max(0, i - width + 1), i]; equivalent to
    cumulative[i] - cumulative[i - width] (treating negative index as 0).
    """
    shifted = np.zeros_like(cumulative)
    if width < len(cumulative):
        shifted[width:] = cumulative[:-width]
    return cumulative - shifted


def _peer_window_stats(peer_codes: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Fanout and peer-novelty via a bounded double sliding window.

    Cost is O(n * width) but with O(1) integer dict operations, bounded by the
    window size (<=128) rather than the entity's total distinct-peer count —
    unlike a one-hot/cumsum approach, which blows up for high-fanout entities
    (e.g. a DDoS/PortScan target hit by thousands of distinct source IPs).
    """
    n = len(peer_codes)
    window_current: dict[int, int] = {}
    window_history: dict[int, int] = {}
    queue_current: list[int] = []
    queue_history: list[int] = []
    fanout = np.empty(n, dtype=np.int64)
    novelty = np.empty(n, dtype=np.float64)
    for i in range(n):
        code = int(peer_codes[i])
        if len(queue_current) == width:
            moved = queue_current.pop(0)
            window_current[moved] -= 1
            if window_current[moved] == 0:
                del window_current[moved]
            if len(queue_history) == width:
                dropped = queue_history.pop(0)
                window_history[dropped] -= 1
                if window_history[dropped] == 0:
                    del window_history[dropped]
            queue_history.append(moved)
            window_history[moved] = window_history.get(moved, 0) + 1
        queue_current.append(code)
        window_current[code] = window_current.get(code, 0) + 1

        distinct = len(window_current)
        novel = sum(1 for key in window_current if key not in window_history)
        fanout[i] = distinct
        novelty[i] = novel / max(1, distinct)
    return fanout, novelty


def multi_timescale_features(events: pd.DataFrame, windows=(8, 32, 128)) -> pd.DataFrame:
    """Causal behavior features at fixed short/medium/long event horizons."""
    ordered = events.sort_values(["entity_id", "timestamp"], kind="stable").copy()
    result = pd.DataFrame(index=ordered.index)
    for entity, group in ordered.groupby("entity_id", sort=False):
        n = len(group)
        pos = np.arange(n)
        ts = pd.to_datetime(group["timestamp"], utc=True).astype("int64").to_numpy() / 1e9
        token = group["event_type"].astype(str).to_numpy()
        peer = group["dst_id"].astype(str).to_numpy()
        byte = np.log1p(np.clip(group["bytes"].to_numpy(float), 0, None))
        port = group["event_type"].astype(str).str.rsplit(":", n=1).str[-1].to_numpy()

        token_codes, _ = pd.factorize(token)
        port_codes, _ = pd.factorize(port)
        peer_codes, _ = pd.factorize(peer)

        token_onehot = np.zeros((n, token_codes.max() + 1), dtype=np.int32)
        token_onehot[pos, token_codes] = 1
        port_onehot = np.zeros((n, port_codes.max() + 1), dtype=np.int32)
        port_onehot[pos, port_codes] = 1

        token_cum = np.cumsum(token_onehot, axis=0)
        port_cum = np.cumsum(port_onehot, axis=0)
        byte_cum = np.cumsum(byte)
        byte_sq_cum = np.cumsum(byte * byte)

        for width in windows:
            start = np.maximum(0, pos - width + 1)
            win_len = pos - start + 1

            duration = np.maximum(ts[pos] - ts[start], 1.0)
            rate = win_len / duration

            token_win = _windowed(token_cum, width)
            repetition = token_win.max(axis=1) / win_len

            port_win = _windowed(port_cum, width)
            port_diversity = (port_win > 0).sum(axis=1)

            fanout, peer_novelty = _peer_window_stats(peer_codes, width)

            bytes_sum = _windowed(byte_cum, width)
            bytes_sq_sum = _windowed(byte_sq_cum, width)
            bytes_mean = bytes_sum / win_len
            bytes_var = np.maximum(bytes_sq_sum / win_len - bytes_mean ** 2, 0.0)
            bytes_std = np.sqrt(bytes_var)

            values = {"rate": rate, "repetition": repetition, "peer_novelty": peer_novelty,
                     "port_diversity": port_diversity, "bytes_mean": bytes_mean,
                     "bytes_std": bytes_std, "fanout": fanout}
            for name, array in values.items():
                result.loc[group.index, f"w{width}_{name}"] = array
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
