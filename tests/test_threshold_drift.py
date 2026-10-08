import json

import numpy as np
import pandas as pd

from scripts.threshold_drift import POLICIES, evaluate_days


def _scored(day, benign_shift, attack_score=None):
    rng = np.random.default_rng(0)
    n = 200
    frame = pd.DataFrame({
        "timestamp": pd.Timestamp(day, tz="UTC") + pd.to_timedelta(np.arange(n), unit="s"),
        "score": rng.normal(benign_shift, 1.0, n),
        "Label": "BENIGN",
    })
    if attack_score is not None:
        frame.loc[:19, "Label"] = "PortScan"
        frame.loc[:19, "score"] = attack_score
    frame["score_without_nll"] = frame["score"]
    return frame


def _stream():
    return pd.concat([_scored("2017-07-04", 0.0), _scored("2017-07-05", 2.0, attack_score=9.0),
                      _scored("2017-07-06", 2.0, attack_score=9.0)], ignore_index=True)


def test_reports_every_policy_for_each_test_day():
    result = evaluate_days(_stream(), development_day="2017-07-04", target_fpr=0.05)
    assert result["test_days"] == ["2017-07-05", "2017-07-06"]
    for policies in result["scores"].values():
        assert set(policies) == set(POLICIES)
        for entry in policies.values():
            assert set(entry["days"]) == {"2017-07-05", "2017-07-06"}
    json.dumps(result, allow_nan=False)


def test_previous_day_recalibration_tracks_benign_drift():
    result = evaluate_days(_stream(), development_day="2017-07-04", target_fpr=0.05)["scores"]["combo_score"]
    # Benign scores shift up by 2 sigma after July 4, so the frozen threshold overshoots the FPR target.
    assert result["frozen"]["days"]["2017-07-06"]["observed_test_fpr"] > 0.3
    assert result["prev_day_benign"]["days"]["2017-07-06"]["observed_test_fpr"] < 0.15


def test_unlabeled_policy_uses_no_labels_and_only_past_days():
    stream = _stream()
    a = evaluate_days(stream, development_day="2017-07-04", target_fpr=0.05)
    relabeled = stream.assign(Label=np.where(stream["timestamp"].dt.day == 5, "BENIGN", stream["Label"]))
    b = evaluate_days(relabeled, development_day="2017-07-04", target_fpr=0.05)
    day6 = lambda r: r["scores"]["combo_score"]["prev_day_unlabeled"]["days"]["2017-07-06"]["threshold"]
    assert day6(a) == day6(b)
    future = stream.assign(score=np.where(stream["timestamp"].dt.day == 6, 100.0, stream["score"]))
    c = evaluate_days(future, development_day="2017-07-04", target_fpr=0.05)
    assert day6(a) == day6(c)


def test_drift_flows_through_consolidation_and_render(tmp_path):
    from scripts.consolidate_benchmarks import summarize_drift
    from scripts.render_results import render

    raw = evaluate_days(_stream(), development_day="2017-07-04", target_fpr=0.05)
    raw["bundle_seed"] = 42
    path = tmp_path / "threshold_drift.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    drift = summarize_drift(str(path))
    assert drift["policies"]["prev_day_unlabeled"]["labels_used_for_threshold"] is False
    assert set(drift["policies"]["frozen"]["daily_fpr"]) == {"2017-07-05", "2017-07-06"}
    text = render({"datasets": {"x": {"seeds": [42], "aggregate": {}}}, "threshold_drift": drift})
    assert "**Threshold drift**" in text and "| prev_day_unlabeled | no |" in text
    assert "07-05 FPR" in text
