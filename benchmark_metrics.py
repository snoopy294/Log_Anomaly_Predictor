"""Metrics at frozen operating points; test labels never set thresholds here."""

from __future__ import annotations

import platform
import time
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from datasets import BENIGN_LABELS


def is_benign(labels) -> np.ndarray:
    return pd.Series(labels).fillna("").astype(str).str.strip().str.lower().isin(BENIGN_LABELS).to_numpy()


def frozen_operating_point(labels, scores, threshold: float, calibration: dict | None = None) -> dict:
    benign = is_benign(labels)
    truth = ~benign
    pred = np.asarray(scores, dtype=float) >= float(threshold)
    tp, fp = int((pred & truth).sum()), int((pred & benign).sum())
    tn, fn = int((~pred & benign).sum()), int((~pred & truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": float(threshold),
        "calibration_population": calibration or {},
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "recall": recall,
        "tpr": recall,
        "precision": precision,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "observed_test_fpr": fp / (fp + tn) if fp + tn else 0.0,
        "benign_alerts_per_10000_events": 10000 * fp / (fp + tn) if fp + tn else 0.0,
    }


def diagnostic_metrics(labels, scores) -> dict:
    truth = ~is_benign(labels)
    values = np.asarray(scores, dtype=float)
    if len(np.unique(truth)) < 2:
        return {"note": "single-class data; ranking metrics undefined"}
    fpr, tpr, _ = roc_curve(truth, values)
    idx = max(0, np.searchsorted(fpr, 0.01, side="right") - 1)
    return {
        "roc_auc": float(roc_auc_score(truth, values)),
        "pr_auc": float(average_precision_score(truth, values)),
        "test_roc_tpr_at_fpr_1pct": float(tpr[min(idx, len(tpr) - 1)]),
    }


def per_attack_metrics(labels, scores, threshold: float) -> dict:
    labels = pd.Series(labels).fillna("").astype(str)
    values = np.asarray(scores, dtype=float)
    benign = is_benign(labels)
    output = {}
    for family in sorted(labels.loc[~benign].unique()):
        attack = labels.to_numpy() == family
        population = benign | attack
        truth = attack[population]
        family_scores = values[population]
        pred = family_scores >= threshold
        false_negatives = int((truth & ~pred).sum())
        ranking_defined = len(np.unique(truth)) == 2
        output[family] = {
            "sample_count": int(truth.sum()),
            "false_negative_count": false_negatives,
            "recall": float(1 - false_negatives / truth.sum()),
            "roc_auc": float(roc_auc_score(truth, family_scores)) if ranking_defined else None,
            "pr_auc": float(average_precision_score(truth, family_scores)) if ranking_defined else None,
        }
    return output


def suppression_metrics(labels, raw_alert, suppressed_alert) -> dict:
    labels = pd.Series(labels)
    truth = ~is_benign(labels)
    raw, kept = np.asarray(raw_alert, bool), np.asarray(suppressed_alert, bool)
    raw_recall = float((raw & truth).sum() / truth.sum()) if truth.sum() else 0.0
    kept_recall = float((kept & truth).sum() / truth.sum()) if truth.sum() else 0.0
    return {
        "raw_alerts": int(raw.sum()),
        "post_suppression_alerts": int(kept.sum()),
        "alert_reduction": float(1 - kept.sum() / raw.sum()) if raw.sum() else 0.0,
        "retained_recall": kept_recall,
        "recall_loss": raw_recall - kept_recall,
        "enabled_by_default": bool(((raw & is_benign(labels)).sum() == 0 or
                                    (kept & is_benign(labels)).sum() <= .8 * (raw & is_benign(labels)).sum())
                                   and raw_recall - kept_recall <= .01),
    }


def suppression_masks(frame: pd.DataFrame, score_col: str, threshold: float,
                      allowlist: set[str] | None = None, allowlist_margin: float = 0.0,
                      dedup_seconds: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Apply allowlist and per-entity cooldown before any export/top-k cap."""
    raw = frame[score_col].to_numpy(float) >= threshold
    kept = raw.copy()
    if allowlist:
        dst = frame["dst_parsed"].astype(str).isin(set(map(str, allowlist))).to_numpy()
        kept &= ~dst | (frame[score_col].to_numpy(float) >= threshold + allowlist_margin)
    if dedup_seconds > 0:
        last: dict[str, pd.Timestamp] = {}
        timestamps = pd.to_datetime(frame["timestamp"], utc=True)
        for pos in np.flatnonzero(kept):
            entity = str(frame.iloc[pos]["entity_id"])
            now = timestamps.iloc[pos]
            if entity in last and (now - last[entity]).total_seconds() < dedup_seconds:
                kept[pos] = False
            else:
                last[entity] = now
    return raw, kept


def latency_benchmark(score_one: Callable[[], object], score_batch: Callable[[int], object],
                      iterations: int = 10_000, warmup: int = 100, batch_size: int = 512) -> dict:
    for _ in range(warmup):
        score_one()
    samples = np.empty(iterations)
    for i in range(iterations):
        start = time.perf_counter()
        score_one()
        samples[i] = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    score_batch(iterations)
    elapsed = time.perf_counter() - start
    return {
        "batch_one_ms": {"p50": float(np.percentile(samples, 50)),
                         "p95": float(np.percentile(samples, 95)),
                         "p99": float(np.percentile(samples, 99))},
        "offline_throughput_windows_per_second": iterations / elapsed,
        "iterations": iterations,
        "warmup_iterations": warmup,
        "offline_batch_size": batch_size,
        "hardware": {"platform": platform.platform(), "processor": platform.processor()},
    }
