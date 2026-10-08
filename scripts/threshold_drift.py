"""Does the 1% FPR target survive day-to-day drift? Per-day threshold study.

Scores CICIDS2017 development (July 4) and test (July 5-7) as one causal stream
with the frozen bundle, then evaluates each test day under three threshold
policies. Every policy only looks at days strictly before the one it scores:

- frozen:             calibrated once on July 4 benign rows (the published setup)
- prev_day_benign:    recalibrated nightly on the previous day's benign rows
                      (assumes yesterday's alerts were triaged, so its labels are known)
- prev_day_unlabeled: recalibrated nightly on all of the previous day's rows,
                      no labels at all (attacks in the window push the threshold up)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import write_run_manifest
from benchmark_metrics import frozen_operating_point, is_benign, per_attack_metrics
from datasets import cicids_day_split
from detector_bundle import DetectorRuntime, load_detector_bundle, score_events_batched
from new import calibrate_threshold, load_and_adapt_events

SCORES = {"combo_score": "score", "combo_without_nll": "score_without_nll"}
POLICIES = ("frozen", "prev_day_benign", "prev_day_unlabeled")


def _day(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["timestamp"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")


def _thresholds(prev: pd.DataFrame, dev: pd.DataFrame, column: str, target_fpr: float) -> dict:
    return {
        "frozen": calibrate_threshold(dev[column].to_numpy(float)[is_benign(dev["Label"])], target_fpr),
        "prev_day_benign": calibrate_threshold(prev[column].to_numpy(float)[is_benign(prev["Label"])],
                                               target_fpr),
        "prev_day_unlabeled": calibrate_threshold(prev[column].to_numpy(float), target_fpr),
    }


def evaluate_days(scored: pd.DataFrame, *, development_day: str, target_fpr: float) -> dict:
    """Per-day operating points for each score and threshold policy.

    `scored` holds the development day followed by the test days; each test day
    is thresholded with information from earlier days only."""
    day = _day(scored)
    days = sorted(day.unique())
    if days[0] != development_day:
        raise ValueError(f"stream must start on the development day {development_day}")
    dev = scored.loc[day == development_day]

    result = {"target_fpr": float(target_fpr), "development_day": development_day,
              "test_days": days[1:], "scores": {}}
    for name, column in SCORES.items():
        per_policy = {policy: {"days": {}} for policy in POLICIES}
        for prev_day, today in zip(days, days[1:]):
            prev, cur = scored.loc[day == prev_day], scored.loc[day == today]
            for policy, threshold in _thresholds(prev, dev, column, target_fpr).items():
                families = per_attack_metrics(cur["Label"], cur[column], threshold)
                point = frozen_operating_point(cur["Label"], cur[column], threshold, {
                    "policy": policy,
                    "calibration_day": development_day if policy == "frozen" else prev_day,
                    "labels_used_for_threshold": policy != "prev_day_unlabeled",
                })
                point["macro_attack_family_recall"] = (
                    float(np.mean([f["recall"] for f in families.values()])) if families else None)
                point["attack_families"] = sorted(families)
                per_policy[policy]["days"][today] = point
        for policy, entry in per_policy.items():
            fprs = [d["observed_test_fpr"] for d in entry["days"].values()]
            entry["max_daily_fpr"] = float(max(fprs))
            entry["mean_daily_fpr"] = float(np.mean(fprs))
            entry["days_within_2x_target"] = int(sum(f <= 2 * target_fpr for f in fprs))
        result["scores"][name] = per_policy
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="clean-format CICIDS csv (data/cicids_clean.csv)")
    parser.add_argument("--bundle", default="models/detector_bundle")
    parser.add_argument("--target_fpr", type=float, default=0.01)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    events = load_and_adapt_events(args.csv, fmt="clean")
    _train, development, test = cicids_day_split(events)
    bundle, model = load_detector_bundle(args.bundle)

    stream = pd.concat([development, test], ignore_index=True)
    stream["event_row_id"] = np.arange(len(stream))
    runtime = DetectorRuntime(bundle, model, ablations={"without_nll": ("nll",)})
    scored = score_events_batched(stream, runtime).sort_values("event_row_id", kind="stable")
    result = evaluate_days(scored, development_day="2017-07-04", target_fpr=args.target_fpr)
    result["bundle_seed"] = bundle.get("dataset", {}).get("seed")
    result["bundle_hashes"] = bundle["hashes"]

    out = Path(args.out)
    write_run_manifest(out, dataset_name="CICIDS2017 per-day threshold drift (frozen bundle)",
                       source_files=[args.csv],
                       splits={"development": development, "test": test},
                       config={"target_fpr": args.target_fpr, "bundle": args.bundle,
                               "policies": list(POLICIES)},
                       seed=int(result["bundle_seed"] or 0))
    (out / "threshold_drift.json").write_text(json.dumps(result, indent=2, allow_nan=False),
                                              encoding="utf-8")


if __name__ == "__main__":
    main()
