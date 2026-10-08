"""Build the sole README data source from reviewed run metric artifacts."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

KEYS = ("recall", "precision", "f1", "observed_test_fpr", "benign_alerts_per_10000_events")
REFERENCE_SEEDS = {42, 43, 44}


def _mean_std(values) -> dict:
    array = np.asarray(values, float)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0))}


def _read(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_run(filename: str) -> dict:
    raw = _read(filename)
    point = raw.get("frozen_operating_point") or raw.get("detection", {}).get("frozen_operating_point")
    if not point:
        raise ValueError(f"{filename} has no frozen_operating_point")
    by_label = raw.get("detection_by_label") or raw.get("per_attack") or {}
    recalls = [float(v["recall"]) for v in by_label.values()]
    p95 = raw.get("inference_benchmark", {}).get("batch_one_ms", {}).get("p95")
    return {"seed": raw.get("run_manifest", {}).get("seed", raw.get("seed", "frozen")),
            "frozen_operating_point": point,
            "macro_attack_family_recall": float(np.mean(recalls)) if recalls else None,
            "per_family": {family: {"sample_count": int(v["sample_count"]), "recall": float(v["recall"])}
                           for family, v in by_label.items()},
            "batch_one_p95_ms": None if p95 is None else float(p95),
            "artifact": str(filename)}


def aggregate_runs(name: str, runs: list[dict]) -> dict:
    numeric_seeds = {run["seed"] for run in runs if isinstance(run["seed"], int)}
    if numeric_seeds and numeric_seeds != REFERENCE_SEEDS:
        raise ValueError(f"{name} must contain exactly seeds 42, 43, and 44; got {sorted(numeric_seeds)}")
    aggregate = {key: _mean_std([run["frozen_operating_point"][key] for run in runs]) for key in KEYS}
    for key in ("macro_attack_family_recall", "batch_one_p95_ms"):
        values = [run[key] for run in runs if run[key] is not None]
        if values:
            aggregate[key] = _mean_std(values)
    families: dict[str, dict] = {}
    for run in runs:
        for family, stats in run["per_family"].items():
            entry = families.setdefault(family, {"sample_count": stats["sample_count"], "recalls": []})
            entry["recalls"].append(stats["recall"])
    per_family = {family: {**_mean_std(entry["recalls"]), "n_runs": len(entry["recalls"]),
                           "sample_count": entry["sample_count"]}
                  for family, entry in families.items()}
    return {"seeds": [run["seed"] for run in runs], "aggregate": aggregate,
            "per_family": per_family, "runs": runs}


def summarize_selection(path: str) -> dict:
    raw = _read(path)
    return {"artifact": str(path), "winner": raw["winner"]["variant"],
            "winner_threshold": float(raw["winner"]["threshold"]),
            "candidates": [{"variant": c["variant"],
                            "macro_attack_family_recall": float(c["macro_attack_family_recall"]),
                            "precision": float(c["precision"]), "fpr": float(c["fpr"])}
                           for c in raw["candidates"]]}


def summarize_ablation(path: str) -> dict:
    raw = _read(path)
    variants = {}
    for name, variant in raw["variants"].items():
        point = variant["frozen_operating_point"]
        variants[name] = {**{key: float(point[key]) for key in ("recall", "precision", "f1",
                                                                "observed_test_fpr")},
                          "macro_attack_family_recall": float(variant["macro_attack_family_recall"])}
    return {"artifact": str(path), "bundle_seed": raw.get("bundle_seed"), "variants": variants}


def summarize_drift(path: str, score: str = "combo_score") -> dict:
    raw = _read(path)
    policies = {}
    for policy, entry in raw["scores"][score].items():
        days = entry["days"]
        macro = [d["macro_attack_family_recall"] for d in days.values()
                 if d["macro_attack_family_recall"] is not None]
        policies[policy] = {
            "labels_used_for_threshold": next(iter(days.values()))["calibration_population"]
                                         ["labels_used_for_threshold"],
            "daily_fpr": {day: float(d["observed_test_fpr"]) for day, d in days.items()},
            "max_daily_fpr": float(entry["max_daily_fpr"]),
            "mean_daily_recall": float(np.mean([d["recall"] for d in days.values()])),
            "mean_daily_macro_attack_family_recall": float(np.mean(macro)) if macro else None,
        }
    return {"artifact": str(path), "score": score, "bundle_seed": raw.get("bundle_seed"),
            "target_fpr": float(raw["target_fpr"]), "test_days": raw["test_days"], "policies": policies}


def build_summary(runs: list[tuple[str, str]], selection_path: str | None = None,
                  ablation_path: str | None = None, drift_path: str | None = None) -> dict:
    grouped = defaultdict(list)
    for name, filename in runs:
        grouped[name].append(load_run(filename))
    summary = {"schema_version": "1.1",
               "datasets": {name: aggregate_runs(name, group) for name, group in grouped.items()}}
    if selection_path:
        summary["selection"] = summarize_selection(selection_path)
    if ablation_path:
        summary["ablation"] = summarize_ablation(ablation_path)
    if drift_path:
        summary["threshold_drift"] = summarize_drift(drift_path)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True,
                        help="NAME=metrics.json (repeat for each seed/experiment)")
    parser.add_argument("--selection", help="candidate_selection.json from scripts/select_candidate.py")
    parser.add_argument("--ablation", help="ablation.json from scripts/ablate_nll.py")
    parser.add_argument("--drift", help="threshold_drift.json from scripts/threshold_drift.py")
    parser.add_argument("--out", default="outputs/benchmark_summary.json")
    args = parser.parse_args()
    runs = []
    for spec in args.run:
        name, separator, filename = spec.partition("=")
        if not separator:
            parser.error("--run must be NAME=metrics.json")
        runs.append((name, filename))
    summary = build_summary(runs, args.selection, args.ablation, args.drift)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
