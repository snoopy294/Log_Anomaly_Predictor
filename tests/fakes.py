"""Shared lightweight stand-ins for the Keras model and a detector bundle."""

import numpy as np
import pandas as pd

COMBO_FEATURES = ["tok_uniq", "tok_top_share", "dst_uniq", "dst_top_share",
                  "bytes_mean", "bytes_std", "repeat_last", "nll"]


class FakeModel:
    """Fixed next-token distribution over 3 ids (PAD, UNK, one known token)."""

    def __call__(self, x, training=False):
        return np.tile(np.array([0.2, 0.3, 0.5]), (np.asarray(x).shape[0], 1))


def fake_bundle(vocabulary=("TCP_WELL_KNOWN|DST=OTHER|BYTES_Q0",), seq_len=2):
    med = {k: 0.0 for k in COMBO_FEATURES}
    scale = {k: 1.0 for k in COMBO_FEATURES}
    return {"vocabulary": list(vocabulary), "destinations": [], "bytes_edges_log1p": [0, 10],
            "seq_len": seq_len, "entity_nll_stats": [], "global_nll": {"mean": 1.0, "std": 2.0},
            "window_baselines": {"features": list(COMBO_FEATURES), "global_median": med,
                                 "global_scale": scale, "entity_median": {}, "entity_scale": {}},
            "score_definition": {"variant": "combo_score"}, "threshold": 1.0,
            "allowlist": [], "model_version": "test", "dataset": {"seed": 42},
            "hashes": {"payload_sha256": "test"}}


def entity_events(entity, day, n, *, label="BENIGN", varied_peers=False, bytes_base=100.0):
    start = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=9)
    return pd.DataFrame({
        "timestamp": [start + pd.Timedelta(seconds=i) for i in range(n)],
        "entity_id": entity,
        "dst_id": [f"peer_{i}" if varied_peers else "peer" for i in range(n)],
        "event_type": "TCP_WELL_KNOWN",
        "bytes": [bytes_base + i for i in range(n)],
        "Label": label,
    })
