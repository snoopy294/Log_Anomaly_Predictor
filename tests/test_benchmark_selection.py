import numpy as np
import pandas as pd
import pytest

from benchmark import improvement_gate, multi_timescale_features, screen_candidates


def _multi_timescale_features_reference(events: pd.DataFrame, windows) -> pd.DataFrame:
    """Brute-force reference matching the pre-vectorization semantics (small inputs only)."""
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


def test_multiscale_features_matches_reference_on_random_multientity_data():
    rng = np.random.default_rng(7)
    n = 300
    frame = pd.DataFrame({
        "timestamp": pd.Timestamp("2020-01-01", tz="UTC") + pd.to_timedelta(
            np.cumsum(rng.integers(1, 5, size=n)), unit="s"),
        "entity_id": rng.choice(["e1", "e2", "e3"], size=n),
        "event_type": rng.choice(["tcp:80", "tcp:443", "udp:53"], size=n),
        "dst_id": rng.choice(["a", "b", "c", "d"], size=n),
        "bytes": rng.integers(0, 5000, size=n).astype(float),
    })
    windows = (3, 8)
    expected = _multi_timescale_features_reference(frame, windows)
    actual = multi_timescale_features(frame, windows=windows)
    pd.testing.assert_frame_equal(expected, actual, check_dtype=False, atol=1e-8)


def test_candidate_selection_uses_macro_recall_then_precision_then_latency():
    frame = pd.DataFrame({"Label": ["BENIGN"] * 100 + ["A"] * 10 + ["B"] * 10,
                          "one": list(range(100)) + [200] * 10 + [0] * 10,
                          "both": list(range(100)) + [200] * 20})
    out = screen_candidates(frame, ["one", "both"], target_fpr=.01,
                            latency_p95_ms={"one": 1, "both": 2})
    assert out["winner"]["variant"] == "both"


def test_candidate_selection_tolerates_unavoidable_one_sample_fpr_overshoot():
    # 37 benign rows: target_fpr(.1) * 37 = 3.7 is not a whole number, so
    # calibrate_threshold's quantile(method="higher") + the ">=" comparison in
    # frozen_operating_point necessarily overshoots to 4/37 (~0.108) rather than
    # landing exactly on 0.1 -- this must not make every candidate ineligible.
    frame = pd.DataFrame({"Label": ["BENIGN"] * 37 + ["A"] * 5,
                          "score": list(range(37)) + [200] * 5})
    out = screen_candidates(frame, ["score"], target_fpr=0.1)
    assert out["winner"]["variant"] == "score"
    assert out["winner"]["fpr"] == pytest.approx(4 / 37)


def test_improvement_gate_enforces_all_three_limits():
    current = {"macro_attack_family_recall": .5, "fpr": .01, "p95_latency_ms": 10}
    assert improvement_gate(current, {"macro_attack_family_recall": .52, "fpr": .02,
                                      "p95_latency_ms": 11})
    assert not improvement_gate(current, {"macro_attack_family_recall": .519, "fpr": .01,
                                          "p95_latency_ms": 10})


def test_multiscale_features_are_causal_and_complete():
    frame = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=4, freq="s", tz="UTC"),
                          "entity_id": ["e"] * 4, "event_type": ["tcp:80"] * 4,
                          "dst_id": ["a", "a", "b", "c"], "bytes": [1, 2, 3, 4]})
    features = multi_timescale_features(frame, windows=(2,))
    assert {"w2_rate", "w2_repetition", "w2_peer_novelty", "w2_port_diversity",
            "w2_bytes_mean", "w2_bytes_std", "w2_fanout"} == set(features.columns)
    assert features.loc[0, "w2_fanout"] == 1
    assert features.loc[3, "w2_fanout"] == 2
