"""Versioned detector bundle and shared offline/online scoring runtime."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA_VERSION = "1.0"
BUNDLE_FILE = "bundle.json"
MODEL_FILE = "model.keras"


def _canonical_hash(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def serialize_window_baselines(baselines: dict) -> dict:
    return {
        "features": list(baselines["cols"]),
        "global_median": {k: float(v) for k, v in baselines["g_med"].items()},
        "global_scale": {k: float(v) for k, v in baselines["g_scale"].items()},
        "entity_median": {str(i): {k: float(v) for k, v in row.items()}
                          for i, row in baselines["e_med"].to_dict("index").items()},
        "entity_scale": {str(i): {k: float(v) for k, v in row.items()}
                         for i, row in baselines["e_scale"].to_dict("index").items()},
    }


def save_detector_bundle(path: str | Path, model_path: str | Path, *, vocab: list[str],
                         destinations: set[str], bytes_edges, entity_stats: pd.DataFrame,
                         global_nll_mean: float, global_nll_std: float, window_baselines: dict,
                         seq_len: int, score_definition: dict, threshold: float,
                         allowlist: set[str] | None = None, model_version: str = "1",
                         dataset: dict | None = None) -> Path:
    """Persist every fitted component needed to reproduce a score."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    model_target = target / MODEL_FILE
    source = Path(model_path)
    if source.resolve() != model_target.resolve():
        shutil.copy2(source, model_target)
    stats = entity_stats.copy()
    stats["entity_id"] = stats["entity_id"].astype(str)
    data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_version": str(model_version),
        "model_file": MODEL_FILE,
        "seq_len": int(seq_len),
        "vocabulary": list(vocab),
        "token_ids": {"PAD": 0, "UNK": 1, "KNOWN_START": 2},
        "destinations": sorted(map(str, destinations)),
        "bytes_edges_log1p": [float(x) for x in bytes_edges],
        "entity_nll_stats": stats.to_dict("records"),
        "global_nll": {"mean": float(global_nll_mean), "std": float(global_nll_std)},
        "window_baselines": serialize_window_baselines(window_baselines),
        "score_definition": score_definition,
        "threshold": float(threshold),
        "allowlist": sorted(map(str, allowlist or set())),
        "dataset": dataset or {},
        "hashes": {"model_sha256": _file_hash(model_target)},
    }
    data["hashes"]["payload_sha256"] = _canonical_hash({k: v for k, v in data.items() if k != "hashes"})
    (target / BUNDLE_FILE).write_text(json.dumps(data, indent=2), encoding="utf-8")
    return target


def load_detector_bundle(path: str | Path, *, load_model: bool = True) -> tuple[dict, Any]:
    root = Path(path)
    data = json.loads((root / BUNDLE_FILE).read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"unsupported detector bundle schema {data.get('schema_version')!r}")
    expected_payload = _canonical_hash({k: v for k, v in data.items() if k != "hashes"})
    if data.get("hashes", {}).get("payload_sha256") != expected_payload:
        raise RuntimeError("detector bundle payload hash mismatch")
    model_path = root / data["model_file"]
    if _file_hash(model_path) != data["hashes"].get("model_sha256"):
        raise RuntimeError("detector bundle model hash mismatch")
    model = None
    if load_model:
        from tensorflow import keras
        # Importing registers the custom layer for fresh processes that load a
        # bundle without importing the training entry point first.
        from new import TakeLastToken
        model = keras.models.load_model(model_path, custom_objects={"TakeLastToken": TakeLastToken})
        expected = len(data["vocabulary"]) + 2
        if int(model.output_shape[-1]) != expected:
            raise RuntimeError(f"bundle/model vocabulary mismatch: {expected} != {model.output_shape[-1]}")
        if int(model.input_shape[-1]) != int(data["seq_len"]):
            raise RuntimeError("bundle/model sequence length mismatch")
    return data, model


class DetectorRuntime:
    """Canonical stateful scorer used by both offline fixtures and Flask."""

    def __init__(self, bundle: dict, model: Any):
        self.bundle, self.model = bundle, model
        self.vocab = list(bundle["vocabulary"])
        self.token_map = {token: i + 2 for i, token in enumerate(self.vocab)}
        self.destinations = set(bundle["destinations"])
        self.edges = np.asarray(bundle["bytes_edges_log1p"], dtype=float)
        self.buffers: dict[str, list[dict]] = {}
        self.warmed_up = False
        self.last_latency_ms = None
        self.last_alert_time: dict[str, pd.Timestamp] = {}
        self.entity_nll = {str(row["entity_id"]): (float(row["mean_nll"]), float(row["std_nll"]))
                           for row in bundle["entity_nll_stats"]}

    def token_string(self, event: dict) -> str:
        from new import bucket_bytes, bucket_dst
        return f"{str(event.get('event_type', 'UNK'))}|{bucket_dst(event.get('dst_id', ''), self.destinations)}|{bucket_bytes(event.get('bytes', 0), self.edges)}"

    def token_id(self, event: dict) -> int:
        return self.token_map.get(self.token_string(event), 1)

    def _window_features(self, context: list[dict], target: dict, nll: float) -> dict[str, float]:
        events = context + [target]
        tokens = np.asarray([self.token_id(e) for e in events])
        destinations = np.asarray([str(e.get("dst_id", "")) for e in events])
        byte_values = np.log1p(np.clip([float(e.get("bytes", 0) or 0) for e in events], 0, None))
        _, tc = np.unique(tokens, return_counts=True)
        _, dc = np.unique(destinations, return_counts=True)
        return {"tok_uniq": float(len(tc)), "tok_top_share": float(tc.max() / len(tokens)),
                "dst_uniq": float(len(dc)), "dst_top_share": float(dc.max() / len(tokens)),
                "bytes_mean": float(byte_values.mean()), "bytes_std": float(byte_values.std()),
                "repeat_last": float(tokens[-1] == tokens[-2]), "nll": float(nll)}

    def _score_features(self, entity: str, features: dict[str, float]) -> tuple[float, str]:
        baseline = self.bundle["window_baselines"]
        med = baseline["entity_median"].get(entity, baseline["global_median"])
        scale = baseline["entity_scale"].get(entity, baseline["global_scale"])
        contributions = {name: abs(np.clip((features[name] - med[name]) / max(scale[name], 1e-6), -50, 50))
                         for name in baseline["features"]}
        return float(np.mean(list(contributions.values()))), max(contributions, key=contributions.get)

    def score_event(self, event: dict, *, update: bool = True) -> dict | None:
        entity = str(event.get("entity_id", ""))
        if not entity:
            raise ValueError("entity_id is required")
        seq_len = int(self.bundle["seq_len"])
        context = self.buffers.setdefault(entity, [])[-seq_len:]
        result = None
        if len(context) == seq_len:
            x = np.asarray([[self.token_id(e) for e in context]], dtype=np.int32)
            target_id = self.token_id(event)
            start = time.perf_counter()
            probs = np.asarray(self.model(x, training=False))[0]
            self.last_latency_ms = (time.perf_counter() - start) * 1000
            self.warmed_up = True
            nll = float(-np.log(np.clip(probs[target_id] if target_id < len(probs) else 1e-9, 1e-9, 1)))
            mean, std = self.entity_nll.get(entity, (self.bundle["global_nll"]["mean"], self.bundle["global_nll"]["std"]))
            nll_z = float((nll - mean) / max(std, 1e-6))
            features = self._window_features(context, event, nll)
            variant = self.bundle["score_definition"].get("variant", "combo_score")
            score, top_feature = (self._score_features(entity, features) if variant != "entity_nll_z"
                                  else (nll_z, "nll"))
            threshold = float(self.bundle["threshold"])
            suppressed = str(event.get("dst_id", "")) in set(self.bundle.get("allowlist", [])) and score < threshold + float(self.bundle["score_definition"].get("allowlist_margin", 0))
            dedup_seconds = int(self.bundle["score_definition"].get("dedup_seconds", 0))
            timestamp = pd.to_datetime(event.get("timestamp"), utc=True, errors="coerce")
            duplicate = (dedup_seconds > 0 and entity in self.last_alert_time and pd.notna(timestamp)
                         and (timestamp - self.last_alert_time[entity]).total_seconds() < dedup_seconds)
            is_anomaly = bool(score >= threshold and not suppressed and not duplicate)
            if is_anomaly and pd.notna(timestamp):
                self.last_alert_time[entity] = timestamp
            result = {"entity_id": entity, "timestamp": event.get("timestamp"), "score": score,
                      "threshold": threshold, "nll": nll, "nll_z": nll_z,
                      "top_contributing_feature": top_feature,
                      "is_anomaly": is_anomaly,
                      "suppressed": bool(suppressed or duplicate), "model_version": self.bundle["model_version"],
                      "warm_up_state": "warm", "batch_one_latency_ms": self.last_latency_ms}
        if update:
            self.buffers[entity].append(dict(event))
            self.buffers[entity] = self.buffers[entity][-(seq_len + 1):]
        return result


def score_events(events: pd.DataFrame, runtime: DetectorRuntime) -> pd.DataFrame:
    """Score a frame causally with the exact serving runtime (zero-shot safe)."""
    rows = []
    ordered = events.sort_values("timestamp", kind="stable")
    for event in ordered.to_dict("records"):
        result = runtime.score_event(event)
        if result is not None:
            result["Label"] = str(event.get("Label", ""))
            result["dst_id"] = str(event.get("dst_id", ""))
            rows.append(result)
    return pd.DataFrame(rows)
