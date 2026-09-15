import numpy as np
import pandas as pd

from benchmark_metrics import frozen_operating_point, per_attack_metrics, suppression_masks, suppression_metrics


def test_frozen_confusion_and_benign_alert_rate_with_ties():
    labels = ["BENIGN", "BENIGN", "A", "A"]
    out = frozen_operating_point(labels, [0.5, 1.0, 1.0, 0.0], 1.0, {"rows": 10})
    assert out["confusion"] == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    assert out["benign_alerts_per_10000_events"] == 5000
    assert out["calibration_population"] == {"rows": 10}


def test_per_attack_metrics_has_recall_auc_counts_and_false_negatives():
    out = per_attack_metrics(["BENIGN", "BENIGN", "A", "A"], [0, .1, .9, .2], .5)
    assert out["A"]["sample_count"] == 2
    assert out["A"]["false_negative_count"] == 1
    assert out["A"]["recall"] == .5
    assert out["A"]["roc_auc"] == 1.0


def test_suppression_is_measured_before_export_cap():
    frame = pd.DataFrame({"timestamp": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 00:01", "2020-01-01 00:10"], utc=True),
                          "entity_id": ["e", "e", "e"], "dst_parsed": ["ok", "x", "x"],
                          "score": [2.1, 3.0, 3.0], "Label": ["BENIGN", "A", "A"]})
    raw, kept = suppression_masks(frame, "score", 2, {"ok"}, .5, 300)
    assert raw.tolist() == [True, True, True]
    assert kept.tolist() == [False, True, True]
    out = suppression_metrics(frame["Label"], raw, kept)
    assert out["raw_alerts"] == 3 and out["post_suppression_alerts"] == 2
