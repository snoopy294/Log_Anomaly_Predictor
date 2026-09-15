import numpy as np
import pandas as pd

from scripts.select_candidate import run_candidate_selection


class FakeModel:
    def __call__(self, x, training=False):
        n = x.shape[0]
        return np.tile(np.array([0.0, 0.1, 0.9]), (n, 1))


def _bundle():
    features = ["tok_uniq", "tok_top_share", "dst_uniq", "dst_top_share",
                "bytes_mean", "bytes_std", "repeat_last"]
    med = {k: 0.0 for k in features}
    scale = {k: 1.0 for k in features}
    return {"vocabulary": ["tcp:80|DST=OTHER|BYTES_Q0"], "destinations": [],
            "bytes_edges_log1p": [0, 10], "seq_len": 2, "entity_nll_stats": [],
            "global_nll": {"mean": 1.0, "std": 2.0},
            "window_baselines": {"features": features, "global_median": med,
                                 "global_scale": scale, "entity_median": {}, "entity_scale": {}},
            "score_definition": {"variant": "combo_score"}, "threshold": 1.0,
            "allowlist": [], "model_version": "test"}


def _events(n_benign_entities=6, n_attack=8):
    rows = []
    ts = pd.Timestamp("2020-01-01", tz="UTC")
    row_id = 0
    for entity in range(n_benign_entities):
        for i in range(20):
            rows.append({"entity_id": f"benign_{entity}", "dst_id": "peer", "event_type": "tcp:80",
                        "bytes": 100 + i, "timestamp": ts + pd.Timedelta(seconds=i),
                        "Label": "BENIGN", "event_row_id": row_id})
            row_id += 1
    for i in range(n_attack):
        rows.append({"entity_id": "attacker", "dst_id": f"target_{i}", "event_type": "tcp:4444",
                    "bytes": 50000 + i * 1000, "timestamp": ts + pd.Timedelta(seconds=i),
                    "Label": "PortScan", "event_row_id": row_id})
        row_id += 1
    return pd.DataFrame(rows)


def test_run_candidate_selection_scores_all_four_candidates_on_shared_rows():
    development = _events()
    result = run_candidate_selection(development, development, _bundle(), FakeModel(),
                                     target_fpr=0.5, seed=42)
    assert set(result["winner"].keys()) >= {"variant", "threshold", "macro_attack_family_recall"}
    variants = {row["variant"] for row in result["candidates"]}
    assert variants == {"transformer_nll", "equal_weight_combined", "multi_timescale", "isolation_forest"}
    assert result["selection_population"] == "development only"
