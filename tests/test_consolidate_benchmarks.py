import json

import pytest

from scripts.consolidate_benchmarks import build_summary

POINT = {"recall": .9, "precision": .8, "f1": .85, "observed_test_fpr": .03,
         "benign_alerts_per_10000_events": 300.0}


def _run(tmp_path, seed, families, p95=30.0):
    raw = {"run_manifest": {"seed": seed}, "detection": {"frozen_operating_point": POINT},
           "detection_by_label": {f: {"sample_count": n, "recall": r} for f, (n, r) in families.items()},
           "inference_benchmark": {"batch_one_ms": {"p95": p95}}}
    path = tmp_path / f"seed-{seed}.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return str(path)


def _three_seeds(tmp_path):
    fam = {"Bot": (10, .0), "DDoS": (100, 1.0)}
    return [("CICIDS", _run(tmp_path, s, fam, p95)) for s, p95 in ((42, 30.0), (43, 32.0), (44, 34.0))]


def test_macro_recall_per_family_and_latency_are_aggregated(tmp_path):
    summary = build_summary(_three_seeds(tmp_path))
    data = summary["datasets"]["CICIDS"]
    assert data["aggregate"]["macro_attack_family_recall"]["mean"] == pytest.approx(.5)
    assert data["aggregate"]["batch_one_p95_ms"]["mean"] == pytest.approx(32.0)
    assert data["per_family"]["Bot"] == {"mean": 0.0, "std": 0.0, "n_runs": 3, "sample_count": 10}
    json.dumps(summary, allow_nan=False)


def test_family_missing_from_one_run_is_averaged_over_present_runs(tmp_path):
    runs = [("CICIDS", _run(tmp_path, 42, {"Bot": (10, .2), "DDoS": (100, 1.0)})),
            ("CICIDS", _run(tmp_path, 43, {"Bot": (10, .4), "DDoS": (100, 1.0)})),
            ("CICIDS", _run(tmp_path, 44, {"DDoS": (100, 1.0)}))]
    bot = build_summary(runs)["datasets"]["CICIDS"]["per_family"]["Bot"]
    assert bot["n_runs"] == 2
    assert bot["mean"] == pytest.approx(.3)


def test_reference_runs_must_cover_all_three_seeds(tmp_path):
    with pytest.raises(ValueError, match="exactly seeds"):
        build_summary(_three_seeds(tmp_path)[:2])


def test_zero_shot_artifact_shape_is_accepted(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"frozen_operating_point": POINT,
                                "per_attack": {"Exploits": {"sample_count": 5, "recall": .4}}}),
                    encoding="utf-8")
    data = build_summary([("UNSW zero-shot", str(path))])["datasets"]["UNSW zero-shot"]
    assert data["seeds"] == ["frozen"]
    assert "batch_one_p95_ms" not in data["aggregate"]
    assert data["aggregate"]["macro_attack_family_recall"]["mean"] == pytest.approx(.4)


def test_selection_with_infinite_latency_consolidates_to_strict_json(tmp_path):
    row = {"variant": "equal_weight_combined", "threshold": 4.3, "macro_attack_family_recall": .85,
           "precision": .76, "fpr": .01, "p95_latency_ms": float("inf")}
    selection = tmp_path / "candidate_selection.json"
    selection.write_text(json.dumps({"candidates": [row], "winner": row}), encoding="utf-8")
    summary = build_summary(_three_seeds(tmp_path), selection_path=str(selection))
    assert summary["selection"]["winner"] == "equal_weight_combined"
    assert summary["selection"]["winner_threshold"] == 4.3
    json.dumps(summary, allow_nan=False)


def test_ablation_is_flattened(tmp_path):
    variant = {"score_column": "score", "frozen_operating_point": POINT,
               "macro_attack_family_recall": .6, "per_attack": {}}
    ablation = tmp_path / "ablation.json"
    ablation.write_text(json.dumps({"bundle_seed": 42, "variants": {"combo_score": variant}}),
                        encoding="utf-8")
    summary = build_summary(_three_seeds(tmp_path), ablation_path=str(ablation))
    assert summary["ablation"]["bundle_seed"] == 42
    assert summary["ablation"]["variants"]["combo_score"] == {
        "recall": .9, "precision": .8, "f1": .85, "observed_test_fpr": .03,
        "macro_attack_family_recall": .6}
