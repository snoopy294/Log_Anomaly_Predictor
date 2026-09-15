import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from detector_bundle import DetectorRuntime, load_detector_bundle, save_detector_bundle


class FakeModel:
    def __call__(self, x, training=False):
        return np.array([[0.0, 0.1, 0.9]], dtype=float)


def _bundle():
    features = ["tok_uniq", "tok_top_share", "dst_uniq", "dst_top_share",
                "bytes_mean", "bytes_std", "repeat_last", "nll"]
    med = {k: 0.0 for k in features}
    scale = {k: 1.0 for k in features}
    return {"vocabulary": ["tcp:80|DST=OTHER|BYTES_Q0"], "destinations": [],
            "bytes_edges_log1p": [0, 10], "seq_len": 2, "entity_nll_stats": [],
            "global_nll": {"mean": 1.0, "std": 2.0},
            "window_baselines": {"features": features, "global_median": med,
                                 "global_scale": scale, "entity_median": {}, "entity_scale": {}},
            "score_definition": {"variant": "entity_nll_z"}, "threshold": 1.0,
            "allowlist": [], "model_version": "test"}


def test_unknown_entity_uses_global_nll_and_unknown_token_policy():
    runtime = DetectorRuntime(_bundle(), FakeModel())
    events = [{"entity_id": "new", "event_type": "tcp:80", "dst_id": "x", "bytes": 1}] * 2
    assert runtime.score_event(events[0]) is None
    assert runtime.score_event(events[1]) is None
    result = runtime.score_event({"entity_id": "new", "event_type": "unknown", "dst_id": "x", "bytes": 1})
    assert runtime.token_id({"event_type": "unknown", "dst_id": "x", "bytes": 1}) == 1
    assert np.isclose(result["nll_z"], (-np.log(.1) - 1) / 2)
    assert result["model_version"] == "test"


def test_score_events_passes_through_event_row_id_for_joins():
    from detector_bundle import score_events
    events = pd.DataFrame([
        {"entity_id": "e", "event_type": "tcp:80", "dst_id": "x", "bytes": 1,
         "timestamp": pd.Timestamp("2020-01-01", tz="UTC"), "Label": "BENIGN", "event_row_id": 10},
        {"entity_id": "e", "event_type": "tcp:80", "dst_id": "x", "bytes": 1,
         "timestamp": pd.Timestamp("2020-01-01", tz="UTC"), "Label": "BENIGN", "event_row_id": 11},
        {"entity_id": "e", "event_type": "tcp:80", "dst_id": "x", "bytes": 1,
         "timestamp": pd.Timestamp("2020-01-01", tz="UTC"), "Label": "BENIGN", "event_row_id": 12},
    ])
    scored = score_events(events, DetectorRuntime(_bundle(), FakeModel()))
    assert list(scored["event_row_id"]) == [12]


def test_flask_and_offline_use_identical_runtime_scoring(monkeypatch):
    import backend
    events = [{"entity_id": "e", "event_type": "tcp:80", "dst_id": "x", "bytes": 1}] * 3
    offline = DetectorRuntime(_bundle(), FakeModel())
    expected = [offline.score_event(event) for event in events][-1]

    served = DetectorRuntime(_bundle(), FakeModel())
    monkeypatch.setattr(backend, "DETECTOR_RUNTIME", served)
    monkeypatch.setattr(backend, "TOKENIZER", {"PAD": 0, "UNK": 1})
    backend.EVENT_BUFFER.clear()
    responses = [backend.app.test_client().post("/api/events/ingest", json=event).get_json()
                 for event in events]
    actual = responses[-1]["anomaly_score"]
    assert actual["score"] == expected["score"]
    assert actual["threshold"] == expected["threshold"]
    assert actual["top_contributing_feature"] == expected["top_contributing_feature"]


def test_bundle_round_trip_and_payload_hash_validation():
    with tempfile.TemporaryDirectory(dir="tmp") as directory:
        tmp_path = Path(directory)
        model = tmp_path / "source.keras"
        model.write_bytes(b"model fixture")
        features = _bundle()["window_baselines"]
        baselines = {"cols": features["features"],
                     "g_med": pd.Series(features["global_median"]),
                     "g_scale": pd.Series(features["global_scale"]),
                     "e_med": pd.DataFrame(columns=features["features"]),
                     "e_scale": pd.DataFrame(columns=features["features"])}
        save_detector_bundle(tmp_path / "bundle", model, vocab=["x"], destinations=set(),
                             bytes_edges=[0, 1], entity_stats=pd.DataFrame(
                                 columns=["entity_id", "mean_nll", "std_nll"]),
                             global_nll_mean=0, global_nll_std=1, window_baselines=baselines,
                             seq_len=2, score_definition={"variant": "combo_score"}, threshold=1)
        data, loaded_model = load_detector_bundle(tmp_path / "bundle", load_model=False)
        assert data["vocabulary"] == ["x"] and loaded_model is None
        manifest = tmp_path / "bundle" / "bundle.json"
        manifest.write_text(manifest.read_text().replace('"threshold": 1.0', '"threshold": 2.0'))
        with pytest.raises(RuntimeError, match="payload hash mismatch"):
            load_detector_bundle(tmp_path / "bundle", load_model=False)
