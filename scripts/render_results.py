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


def render(summary: dict) -> str:
    datasets = summary.get("datasets", {})
    if not datasets:
        return "_No leak-safe benchmark has been published yet._"
    lines = ["All headline values use thresholds frozen on held-out benign calibration traffic.", "",
             "| Dataset / experiment | Seeds | Recall | Precision | F1 | Test FPR | Benign alerts / 10k |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, result in datasets.items():
        aggregate = result.get("aggregate", {})
        lines.append("| " + " | ".join([
            name, ", ".join(map(str, result.get("seeds", []))) or "n/a",
            _mean_std(aggregate.get("recall"), True), _mean_std(aggregate.get("precision"), True),
            _mean_std(aggregate.get("f1"), True), _mean_std(aggregate.get("observed_test_fpr"), True),
            _mean_std(aggregate.get("benign_alerts_per_10000_events")),
        ]) + " |")
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
