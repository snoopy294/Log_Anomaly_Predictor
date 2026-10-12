"""Where do the frozen threshold's false positives come from, day by day?

Scores CICIDS2017 development (July 4) and test (July 5-7) with the frozen
bundle, applies the July 4 benign-calibrated threshold, and breaks each test
day's benign alerts down by entity, by whether the entity has its own window
baseline in the bundle (cold-start entities fall back to the global baseline),
by the feature that dominated the score, and by hour.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import write_run_manifest
from benchmark_metrics import is_benign
from datasets import cicids_day_split
from detector_bundle import DetectorRuntime, load_detector_bundle, score_events_batched
from new import calibrate_threshold, load_and_adapt_events


def _rate(alerts: int, rows: int) -> float:
    return alerts / rows if rows else 0.0


def explain(scored: pd.DataFrame, known_entities: set[str], *, development_day: str,
            target_fpr: float, top_n: int = 10) -> dict:
    day = pd.to_datetime(scored["timestamp"], errors="coerce", utc=True)
    scored = scored.assign(day=day.dt.strftime("%Y-%m-%d"), hour=day.dt.hour,
                           benign=is_benign(scored["Label"]),
                           known=scored["entity_id"].astype(str).isin(known_entities))
    dev = scored.loc[scored["day"] == development_day]
    threshold = calibrate_threshold(dev.loc[dev["benign"], "score"].to_numpy(float), target_fpr)
    scored["alert"] = scored["score"].to_numpy(float) >= threshold

    result = {"target_fpr": float(target_fpr), "threshold": float(threshold),
              "development_day": development_day, "days": {}}
    for name, frame in scored.loc[scored["day"] != development_day].groupby("day"):
        benign = frame.loc[frame["benign"]]
        fp = benign.loc[benign["alert"]]
        per_entity = (benign.groupby("entity_id")
                      .agg(benign_rows=("alert", "size"), false_positives=("alert", "sum"),
                           has_baseline=("known", "first"))
                      .sort_values("false_positives", ascending=False))
        top = per_entity.head(top_n)
        top_peer = fp["dst_id"].astype(str).mode().iloc[0] if len(fp) else None
        cumulative = per_entity["false_positives"].cumsum() / max(len(fp), 1)
        result["days"][name] = {
            "benign_rows": int(len(benign)), "false_positives": int(len(fp)),
            "fpr": _rate(len(fp), len(benign)),
            "benign_entities": int(len(per_entity)),
            "entities_with_false_positives": int((per_entity["false_positives"] > 0).sum()),
            "fp_share_top_1_entity": float(cumulative.iloc[0]) if len(fp) else 0.0,
            "fp_share_top_5_entities": float(cumulative.iloc[min(4, len(cumulative) - 1)]) if len(fp) else 0.0,
            "fpr_without_top_5_entities": _rate(
                len(fp) - int(per_entity["false_positives"].head(5).sum()),
                len(benign) - int(per_entity["benign_rows"].head(5).sum())),
            "cold_start": {
                "benign_row_share": float((~benign["known"]).mean()) if len(benign) else 0.0,
                "fp_share": float((~fp["known"]).mean()) if len(fp) else 0.0,
                "fpr_entities_with_baseline": _rate(int(benign.loc[benign["known"], "alert"].sum()),
                                                    int(benign["known"].sum())),
                "fpr_cold_start_entities": _rate(int(benign.loc[~benign["known"], "alert"].sum()),
                                                 int((~benign["known"]).sum())),
            },
            # dst_id holds the flow's source IP (entities are destinations).
            "top_peers": [{"peer": str(peer), "false_positives": int(n), "fp_share": float(n / len(fp))}
                          for peer, n in fp["dst_id"].astype(str).value_counts().head(top_n).items()],
            "fpr_without_top_peer": _rate(
                len(fp) - int((fp["dst_id"].astype(str) == top_peer).sum()),
                len(benign) - int((benign["dst_id"].astype(str) == top_peer).sum())),
            "fp_share_by_top_feature": {k: float(v) for k, v in
                                        fp["top_contributing_feature"].value_counts(normalize=True).items()},
            "fpr_by_hour": {int(h): _rate(int(g["alert"].sum()), len(g))
                            for h, g in benign.groupby("hour")},
            "top_entities": [{"entity_id": str(entity), "benign_rows": int(row.benign_rows),
                              "false_positives": int(row.false_positives),
                              "entity_fpr": _rate(int(row.false_positives), int(row.benign_rows)),
                              "has_baseline": bool(row.has_baseline)}
                             for entity, row in top.iterrows()],
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
    stream = pd.concat([development, test], ignore_index=True)
    stream["event_row_id"] = np.arange(len(stream))
    scored = score_events_batched(stream, DetectorRuntime(bundle, model))
    known = set(map(str, bundle["window_baselines"]["entity_median"]))
    result = explain(scored, known, development_day="2017-07-04", target_fpr=args.target_fpr)
    result["bundle_seed"] = bundle.get("dataset", {}).get("seed")

    out = Path(args.out)
    write_run_manifest(out, dataset_name="CICIDS2017 false-positive breakdown (frozen bundle)",
                       source_files=[args.csv], splits={"development": development, "test": test},
                       config={"target_fpr": args.target_fpr, "bundle": args.bundle},
                       seed=int(result["bundle_seed"] or 0))
    (out / "false_positives.json").write_text(json.dumps(result, indent=2, allow_nan=False),
                                              encoding="utf-8")


if __name__ == "__main__":
    main()
