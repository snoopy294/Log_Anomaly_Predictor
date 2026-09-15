"""Leak-safe candidate score selection on a real dataset's development split.

Wires benchmark.py's candidate screening (previously unit-tested only) into
a runnable pipeline: scores a development split with the frozen detector
bundle (transformer_nll, equal_weight_combined) and with two additional
candidates computed from causal multi-timescale features (multi_timescale,
isolation_forest), then picks a winner using development labels only.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import (CANDIDATES, fit_isolation_forest, isolation_scores,
                       multi_timescale_features, screen_candidates, write_run_manifest)
from benchmark_metrics import is_benign
from datasets import cicids_day_split
from detector_bundle import DetectorRuntime, load_detector_bundle, score_events
from new import combo_score, fit_window_baselines, load_and_adapt_events, robust_window_z


def run_candidate_selection(train_benign: pd.DataFrame, development: pd.DataFrame,
                            bundle: dict, model, *, target_fpr: float, seed: int) -> dict:
    """Score `development` with all CANDIDATES on the same row set and select a winner.

    `train_benign` fits the multi-timescale baselines/isolation forest;
    the frozen `bundle`/`model` already encode transformer_nll and
    equal_weight_combined (combo_score) fit on the original training split.
    """
    scored = score_events(development, DetectorRuntime(bundle, model))
    if "event_row_id" not in scored.columns:
        raise ValueError("development events must carry a unique event_row_id column")

    train_sorted = train_benign.sort_index()
    mt_train = multi_timescale_features(train_benign)
    win_baselines = fit_window_baselines(mt_train, train_sorted["entity_id"].to_numpy())
    forest = fit_isolation_forest(mt_train, seed=seed)

    dev_sorted = development.sort_index()
    mt_dev = multi_timescale_features(development)
    z_dev = robust_window_z(mt_dev, dev_sorted["entity_id"].to_numpy(), win_baselines)
    candidate_scores = pd.DataFrame({
        "event_row_id": dev_sorted["event_row_id"].to_numpy(),
        "multi_timescale": combo_score(z_dev),
        "isolation_forest": isolation_scores(forest, mt_dev),
    })

    validation = scored.merge(candidate_scores, on="event_row_id", how="inner")
    validation["transformer_nll"] = validation["nll_z"]
    validation["equal_weight_combined"] = validation["score"]

    return screen_candidates(validation, list(CANDIDATES), target_fpr=target_fpr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="clean-format events csv (e.g. data/cicids_clean.csv)")
    parser.add_argument("--bundle", default="models/detector_bundle")
    parser.add_argument("--target_fpr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)

    events = load_and_adapt_events(args.csv, fmt="clean")
    train, development, _test = cicids_day_split(events)
    train_benign = train.loc[is_benign(train["Label"])]

    bundle, model = load_detector_bundle(args.bundle)
    result = run_candidate_selection(train_benign, development, bundle, model,
                                     target_fpr=args.target_fpr, seed=args.seed)

    write_run_manifest(out, dataset_name="CICIDS2017 development candidate selection",
                       source_files=[args.csv],
                       splits={"train_benign": train_benign, "development": development},
                       config={"target_fpr": args.target_fpr, "bundle": args.bundle}, seed=args.seed)
    (out / "candidate_selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
