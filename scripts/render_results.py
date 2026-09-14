"""Render the README results tables from outputs/metrics_summary.json.

Every number in the README's results section comes from this script, so the
README can never drift from the committed metrics artifact. Re-run after any
training run:

    python scripts/render_results.py
"""
import argparse
import json
import re
from pathlib import Path

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"

SCORES = [("combo_score", "combo_score (alerting)"), ("entity_nll_z", "entity_nll_z (LM surprise only)")]
ROWS = [
    ("ROC-AUC", "roc_auc", "{:.4f}"),
    ("PR-AUC", "pr_auc", "{:.4f}"),
    ("Detection rate @ 1% FPR", "detection_at_fpr_1pct", "{:.1%}"),
    ("Detection rate @ 0.1% FPR", "detection_at_fpr_0_1pct", "{:.1%}"),
    ("Precision @ operating threshold", "precision_at_thresh", "{:.3f}"),
    ("Recall @ operating threshold", "recall_at_thresh", "{:.3f}"),
    ("F1 @ operating threshold", "f1_at_thresh", "{:.3f}"),
    ("Operating threshold", "alert_z_thresh", "{:.3f}"),
]


def _fmt(d: dict, key: str, fmt: str) -> str:
    v = d.get(key)
    return fmt.format(v) if isinstance(v, (int, float)) else "n/a"


def render(ms: dict) -> str:
    det = ms["detection"]
    if "roc_auc" not in det:
        return f"_Detection metrics unavailable: {det.get('note', 'no detection block')}_"

    comp = ms.get("detection_comparison", {})
    split = det.get("eval_split", "test")
    # Backslashes aren't allowed inside f-string expressions before Python 3.12
    source = re.sub(r"\d{4,}", lambda m: f"{int(m.group()):,}", det.get("threshold_source", "n/a"))
    lines = [
        f"Evaluated on the held-out **{split}** split: {det['n_total']:,} windows, "
        f"{det['n_positive']:,} of them attacks. Operating threshold: {source}; "
        "test labels are never used to pick it.",
        "",
        "| Metric | " + " | ".join(name for _, name in SCORES) + " |",
        "|---|" + "---:|" * len(SCORES),
    ]
    for label, key, fmt in ROWS:
        lines.append(f"| {label} | " + " | ".join(
            _fmt(comp.get(col, {}).get("overall", {}), key, fmt) for col, _ in SCORES) + " |")

    by_label = {col: comp.get(col, {}).get("by_label", {}) for col, _ in SCORES}
    labels = sorted(by_label["combo_score"], key=lambda l: -by_label["combo_score"][l]["n_positive"])
    if labels:
        lines += ["", "Per-attack ROC-AUC (each attack type vs. all benign test windows):", "",
                  "| Attack | Test windows | " + " | ".join(col for col, _ in SCORES) + " |",
                  "|---|---:|" + "---:|" * len(SCORES)]
        for l in labels:
            n = by_label["combo_score"][l]["n_positive"]
            lines.append(f"| {l} | {n:,} | " + " | ".join(
                _fmt(by_label[col].get(l, {}), "roc_auc", "{:.3f}") for col, _ in SCORES) + " |")

    lm = ms.get(split) or {}
    if "acc" in lm:
        lines += ["", f"_Auxiliary (not a detection metric): next-event LM accuracy on {split} = "
                      f"{lm['acc']:.3f}, top-5 = {lm.get('top5_acc', float('nan')):.3f}._"]
    return "\n".join(lines)


def update_readme(readme: str, body: str) -> str:
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
    if not pattern.search(readme):
        raise ValueError(f"README is missing the {START} / {END} markers")
    return pattern.sub(lambda _: f"{START}\n{body}\n{END}", readme)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", default="outputs/metrics_summary.json")
    p.add_argument("--readme", default="README.md")
    args = p.parse_args()
    ms = json.loads(Path(args.metrics).read_text())
    readme = Path(args.readme)
    readme.write_text(update_readme(readme.read_text(encoding="utf-8"), render(ms)), encoding="utf-8")
    print(f"Updated {args.readme} from {args.metrics}")


if __name__ == "__main__":
    main()
