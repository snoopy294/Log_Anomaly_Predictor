import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from render_results import render, update_readme, START, END  # noqa: E402


def _overall(auc):
    return {"roc_auc": auc, "pr_auc": 0.5, "detection_at_fpr_1pct": 0.9, "detection_at_fpr_0_1pct": 0.4,
            "precision_at_thresh": 0.8, "recall_at_thresh": 0.7, "f1_at_thresh": 0.75,
            "alert_z_thresh": 2.25, "n_total": 1000, "n_positive": 100}


def _ms():
    det = dict(_overall(0.99), eval_split="internal_test", threshold_source="val benign @ FPR=0.01")
    return {
        "detection": det,
        "detection_comparison": {
            "combo_score": {"overall": _overall(0.99),
                            "by_label": {"PortScan": {"roc_auc": 0.998, "n_positive": 90},
                                         "Bot": {"roc_auc": 0.364, "n_positive": 10}}},
            "entity_nll_z": {"overall": _overall(0.55),
                             "by_label": {"PortScan": {"roc_auc": 0.136, "n_positive": 90}}},
        },
        "internal_test": {"acc": 0.386, "top5_acc": 0.75},
    }


def test_render_uses_metrics_values():
    out = render(_ms())
    assert "| ROC-AUC | 0.9900 | 0.5500 |" in out
    assert "| Detection rate @ 1% FPR | 90.0% | 90.0% |" in out
    assert "internal_test" in out and "1,000 windows" in out


def test_render_per_label_sorted_by_count_and_missing_is_na():
    out = render(_ms())
    assert out.index("| PortScan |") < out.index("| Bot |")
    assert "| Bot | 10 | 0.364 | n/a |" in out


def test_render_single_class_note():
    assert "unavailable" in render({"detection": {"note": "single-class data"}})


def test_update_readme_replaces_only_between_markers():
    readme = f"# T\n{START}\nold\n{END}\ntail\n"
    assert update_readme(readme, "new") == f"# T\n{START}\nnew\n{END}\ntail\n"
    with pytest.raises(ValueError):
        update_readme("# no markers", "x")
