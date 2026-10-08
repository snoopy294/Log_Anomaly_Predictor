"""Render README benchmark tables only from a consolidated benchmark summary."""

import argparse
import json
import re
from pathlib import Path

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"


def _mean_std(value: dict, percent=False) -> str:
    if not isinstance(value, dict) or "mean" not in value:
        return "n/a"
    scale = 100 if percent else 1
    suffix = "%" if percent else ""
    return f"{value['mean'] * scale:.2f} ± {value.get('std', 0) * scale:.2f}{suffix}"


def _pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def _family_lines(name: str, result: dict) -> list[str]:
    families = result.get("per_family", {})
    if not families:
        return []
    lines = ["", f"**Per-attack-family recall: {name}** (frozen threshold; mean ± std over runs)", "",
             "| Attack family | Test samples | Recall |", "|---|---:|---:|"]
    for family, stats in sorted(families.items(), key=lambda item: -item[1]["sample_count"]):
        lines.append(f"| {family} | {stats['sample_count']:,} | {_mean_std(stats, True)} |")
    return lines


def _selection_lines(selection: dict) -> list[str]:
    lines = ["", "**Candidate selection** (CICIDS2017 July 4 development labels only; "
                 "each threshold at 1% development FPR)", "",
             "| Candidate | Macro family recall | Precision | Dev FPR |", "|---|---:|---:|---:|"]
    for row in selection["candidates"]:
        mark = " (selected)" if row["variant"] == selection["winner"] else ""
        lines.append(f"| {row['variant']}{mark} | {_pct(row['macro_attack_family_recall'])} | "
                     f"{_pct(row['precision'])} | {_pct(row['fpr'])} |")
    return lines


def _ablation_lines(ablation: dict) -> list[str]:
    lines = ["", f"**Ablation** (seed {ablation['bundle_seed']} bundle; each score's threshold "
                 "calibrated on July 4 benign traffic, applied unchanged to July 5–7)", "",
             "| Score | Recall | Macro family recall | Precision | Test FPR |", "|---|---:|---:|---:|---:|"]
    for name, row in ablation["variants"].items():
        lines.append(f"| {name} | {_pct(row['recall'])} | {_pct(row['macro_attack_family_recall'])} | "
                     f"{_pct(row['precision'])} | {_pct(row['observed_test_fpr'])} |")
    return lines


def _drift_lines(drift: dict) -> list[str]:
    days = drift["test_days"]
    lines = ["", f"**Threshold drift** (seed {drift['bundle_seed']} bundle, `{drift['score']}`; target FPR "
                 f"{_pct(drift['target_fpr'])}; each day thresholded using earlier days only)", "",
             "| Threshold policy | Labels used | " + " | ".join(f"{d[5:]} FPR" for d in days)
             + " | Max daily FPR | Mean daily recall | Mean daily macro family recall |",
             "|---|---|" + "---:|" * (len(days) + 3)]
    for policy, row in drift["policies"].items():
        lines.append(f"| {policy} | {'yes' if row['labels_used_for_threshold'] else 'no'} | "
                     + " | ".join(_pct(row["daily_fpr"].get(d)) for d in days)
                     + f" | {_pct(row['max_daily_fpr'])} | {_pct(row['mean_daily_recall'])} | "
                     f"{_pct(row['mean_daily_macro_attack_family_recall'])} |")
    return lines


def render(summary: dict) -> str:
    datasets = summary.get("datasets", {})
    if not datasets:
        return "_No leak-safe benchmark has been published yet._"
    lines = ["All headline values use thresholds frozen on held-out benign calibration traffic.", "",
             "| Dataset / experiment | Seeds | Recall | Macro family recall | Precision | F1 | Test FPR "
             "| Benign alerts / 10k | Batch-1 p95 latency (ms) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, result in datasets.items():
        aggregate = result.get("aggregate", {})
        lines.append("| " + " | ".join([
            name, ", ".join(map(str, result.get("seeds", []))) or "n/a",
            _mean_std(aggregate.get("recall"), True),
            _mean_std(aggregate.get("macro_attack_family_recall"), True),
            _mean_std(aggregate.get("precision"), True),
            _mean_std(aggregate.get("f1"), True), _mean_std(aggregate.get("observed_test_fpr"), True),
            _mean_std(aggregate.get("benign_alerts_per_10000_events")),
            _mean_std(aggregate.get("batch_one_p95_ms")),
        ]) + " |")
    for name, result in datasets.items():
        lines += _family_lines(name, result)
    if summary.get("selection"):
        lines += _selection_lines(summary["selection"])
    if summary.get("ablation"):
        lines += _ablation_lines(summary["ablation"])
    if summary.get("threshold_drift"):
        lines += _drift_lines(summary["threshold_drift"])
    lines += ["", "Diagnostic ROC-derived TPR values, when present in artifacts, are not frozen-threshold results."]
    return "\n".join(lines)


def update_readme(readme: str, body: str) -> str:
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
    if not pattern.search(readme):
        raise ValueError(f"README is missing the {START} / {END} markers")
    return pattern.sub(lambda _: f"{START}\n{body}\n{END}", readme)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="outputs/benchmark_summary.json")
    parser.add_argument("--readme", default="README.md")
    args = parser.parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    readme = Path(args.readme)
    readme.write_text(update_readme(readme.read_text(encoding="utf-8"), render(summary)), encoding="utf-8")
    print(f"Updated {args.readme} from {args.summary}")


if __name__ == "__main__":
    main()
