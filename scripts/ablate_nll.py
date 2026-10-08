"""What does the Transformer contribute? NLL-feature ablation of the frozen detector.

Scores CICIDS2017 development (July 4) and frozen test (July 5-7) as one causal
stream with the shipped bundle, in a single pass that emits three scores per
event: the shipped combo_score, combo_score with the Transformer NLL feature
removed, and the Transformer's entity NLL z-score alone. Each variant's
threshold is calibrated on development benign rows only, then applied
unchanged to the test days.
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

VARIANTS = {"combo_score": "score",
            "combo_without_nll": "score_without_nll",
            "transformer_nll_z": "nll_z"}


def run_ablation(development: pd.DataFrame, test: pd.DataFrame, bundle: dict, model, *,
                 target_fpr: float) -> dict:
    stream = pd.concat([development, test], ignore_index=True)
    stream["event_row_id"] = np.arange(len(stream))
    runtime = DetectorRuntime(bundle, model, ablations={"without_nll": ("nll",)})
    scored = score_events_batched(stream, runtime)
    on_test = scored["event_row_id"].to_numpy() >= len(development)
    dev, tst = scored.loc[~on_test], scored.loc[on_test]
    dev_benign = is_benign(dev["Label"])

    result = {"target_fpr": float(target_fpr),
              "bundle_seed": bundle.get("dataset", {}).get("seed"),
              "development_rows": int(len(dev)), "test_rows": int(len(tst)), "variants": {}}
    for variant, column in VARIANTS.items():
        threshold = calibrate_threshold(dev[column].to_numpy(float)[dev_benign], target_fpr)
        calibration = {"split": "development", "benign_rows": int(dev_benign.sum()),
                       "target_fpr": float(target_fpr), "labels_used_for_threshold": False}
        families = per_attack_metrics(tst["Label"], tst[column], threshold)
        result["variants"][variant] = {
            "score_column": column,
            "frozen_operating_point": frozen_operating_point(tst["Label"], tst[column],
                                                             threshold, calibration),
            "macro_attack_family_recall": (float(np.mean([f["recall"] for f in families.values()]))
                                           if families else 0.0),
            "per_attack": families,
        }
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
    result = run_ablation(development, test, bundle, model, target_fpr=args.target_fpr)
    result["bundle_hashes"] = bundle["hashes"]

    out = Path(args.out)
    write_run_manifest(out, dataset_name="CICIDS2017 NLL ablation (frozen bundle)",
                       source_files=[args.csv],
                       splits={"development": development, "frozen_test": test},
                       config={"target_fpr": args.target_fpr, "bundle": args.bundle},
                       seed=int(result["bundle_seed"] or 0))
    (out / "ablation.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
