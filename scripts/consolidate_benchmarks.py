"""Build the sole README data source from reviewed run metric artifacts."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True,
                        help="NAME=metrics.json (repeat for each seed/experiment)")
    parser.add_argument("--out", default="outputs/benchmark_summary.json")
    args = parser.parse_args()
    grouped = defaultdict(list)
    for spec in args.run:
        name, separator, filename = spec.partition("=")
        if not separator:
            parser.error("--run must be NAME=metrics.json")
        raw = json.loads(Path(filename).read_text(encoding="utf-8"))
        point = raw.get("frozen_operating_point") or raw.get("detection", {}).get("frozen_operating_point")
        if not point:
            raise ValueError(f"{filename} has no frozen_operating_point")
        seed = raw.get("run_manifest", {}).get("seed", raw.get("seed", "frozen"))
        grouped[name].append({"seed": seed, "frozen_operating_point": point,
                              "artifact": filename})
    datasets = {}
    keys = ("recall", "precision", "f1", "observed_test_fpr", "benign_alerts_per_10000_events")
    for name, runs in grouped.items():
        numeric_seeds = {run["seed"] for run in runs if isinstance(run["seed"], int)}
        if numeric_seeds and numeric_seeds != {42, 43, 44}:
            raise ValueError(f"{name} must contain exactly seeds 42, 43, and 44; got {sorted(numeric_seeds)}")
        aggregate = {}
        for key in keys:
            values = np.asarray([run["frozen_operating_point"][key] for run in runs], float)
            aggregate[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
        datasets[name] = {"seeds": [run["seed"] for run in runs],
                          "aggregate": aggregate, "runs": runs}
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"schema_version": "1.0", "datasets": datasets}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
