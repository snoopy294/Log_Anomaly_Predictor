import json

import numpy as np
import pandas as pd

from scripts.explain_false_positives import explain


def _rows(day, entity, n, score, label="BENIGN", feature="dst_uniq"):
    return pd.DataFrame({
        "timestamp": pd.Timestamp(day, tz="UTC") + pd.to_timedelta(np.arange(n), unit="min"),
        "entity_id": entity, "dst_id": "scanner" if entity == "new_host" else "peer",
        "score": score, "Label": label, "top_contributing_feature": feature,
    })


def _stream():
    return pd.concat([
        _rows("2017-07-04", "a", 100, np.linspace(0, 1, 100)),
        _rows("2017-07-05", "a", 100, 0.5),
        _rows("2017-07-05", "new_host", 20, 5.0, feature="bytes_mean"),
        _rows("2017-07-05", "attacker", 10, 9.0, label="PortScan"),
    ], ignore_index=True)


def test_false_positives_are_attributed_to_cold_start_entity():
    result = explain(_stream(), {"a"}, development_day="2017-07-04", target_fpr=0.05)
    day = result["days"]["2017-07-05"]
    assert day["benign_rows"] == 120 and day["false_positives"] == 20
    assert day["top_entities"][0] == {"entity_id": "new_host", "benign_rows": 20, "false_positives": 20,
                                      "entity_fpr": 1.0, "has_baseline": False}
    assert day["cold_start"]["fp_share"] == 1.0
    assert day["cold_start"]["fpr_entities_with_baseline"] == 0.0
    assert day["fp_share_by_top_feature"] == {"bytes_mean": 1.0}
    assert day["fpr_without_top_5_entities"] == 0.0
    assert day["top_peers"] == [{"peer": "scanner", "false_positives": 20, "fp_share": 1.0}]
    assert day["fpr_without_top_peer"] == 0.0
    json.dumps(result, allow_nan=False)


def test_threshold_comes_from_development_benign_rows_only():
    stream = _stream()
    a = explain(stream, {"a"}, development_day="2017-07-04", target_fpr=0.05)
    b = explain(stream.assign(Label="BENIGN").where(stream["timestamp"].dt.day == 5, stream),
                {"a"}, development_day="2017-07-04", target_fpr=0.05)
    assert a["threshold"] == b["threshold"]
