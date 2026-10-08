"""Apply a frozen CICIDS detector bundle to UNSW-NB15 without tuning."""

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_metrics import diagnostic_metrics, frozen_operating_point, per_attack_metrics
from datasets import adapt_unsw_nb15, sha256_file
from detector_bundle import DetectorRuntime, load_detector_bundle, score_events_batched

UNK_TOKEN_ID = 1


def run_zero_shot(events: pd.DataFrame, bundle: dict, model) -> tuple[pd.DataFrame, dict]:
    runtime = DetectorRuntime(bundle, model)
    token_ids = np.fromiter((runtime.token_id(e) for e in events.to_dict("records")),
                            dtype=np.int64, count=len(events))
    scored = score_events_batched(events, runtime)
    threshold = float(bundle["threshold"])
    per_attack = per_attack_metrics(scored["Label"], scored["score"], threshold)
    metrics = {
        "experiment": "UNSW-NB15 zero-shot from frozen CICIDS bundle",
        "tuning_on_unsw": False,
        "event_type_scheme": "cicids",
        "token_unk_rate": float((token_ids == UNK_TOKEN_ID).mean()) if len(token_ids) else 0.0,
        "frozen_operating_point": frozen_operating_point(
            scored["Label"], scored["score"], threshold,
            {"dataset": "CICIDS2017", "bundle_model_version": bundle["model_version"]}),
        "diagnostic": diagnostic_metrics(scored["Label"], scored["score"]),
        "per_attack": per_attack,
        "macro_attack_family_recall": (float(np.mean([v["recall"] for v in per_attack.values()]))
                                       if per_attack else 0.0),
    }
    return scored, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--bundle", default="models/detector_bundle")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)

    events = adapt_unsw_nb15(pd.read_csv(args.csv, low_memory=False), event_type_scheme="cicids")
    bundle, model = load_detector_bundle(args.bundle)
    scored, metrics = run_zero_shot(events, bundle, model)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "source": {"path": args.csv, "sha256": sha256_file(args.csv), "rows": len(events)},
                "bundle_hashes": bundle["hashes"], "model_version": bundle["model_version"],
                "environment": {"python": sys.version, "platform": platform.platform()}}
    scored.to_csv(out / "scores.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
