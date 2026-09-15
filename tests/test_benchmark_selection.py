import pandas as pd

from benchmark import improvement_gate, multi_timescale_features, screen_candidates


def test_candidate_selection_uses_macro_recall_then_precision_then_latency():
    frame = pd.DataFrame({"Label": ["BENIGN"] * 100 + ["A"] * 10 + ["B"] * 10,
                          "one": list(range(100)) + [200] * 10 + [0] * 10,
                          "both": list(range(100)) + [200] * 20})
    out = screen_candidates(frame, ["one", "both"], target_fpr=.01,
                            latency_p95_ms={"one": 1, "both": 2})
    assert out["winner"]["variant"] == "both"


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
