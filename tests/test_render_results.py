from scripts.render_results import render, update_readme


def test_render_contains_only_consolidated_metrics():
    metric = {"mean": .8, "std": .01}
    summary = {"datasets": {"CICIDS2017 frozen test": {"seeds": [42, 43, 44], "aggregate": {
        "recall": metric, "precision": metric, "f1": metric,
        "observed_test_fpr": {"mean": .01, "std": 0},
        "benign_alerts_per_10000_events": {"mean": 100, "std": 0}}}}}
    text = render(summary)
    assert "CICIDS2017 frozen test" in text
    assert "42, 43, 44" in text
    assert "80.00 ± 1.00%" in text
    assert "thresholds frozen" in text


def test_no_results_is_explicit():
    assert "No leak-safe benchmark" in render({"datasets": {}})


def test_update_readme_replaces_only_marked_section():
    original = "before\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\nafter"
    assert update_readme(original, "new") == "before\n<!-- RESULTS:START -->\nnew\n<!-- RESULTS:END -->\nafter"


def test_render_includes_families_selection_and_ablation():
    metric = {"mean": .8, "std": .01}
    summary = {
        "datasets": {"D": {"seeds": [42], "aggregate": {
            "recall": metric, "precision": metric, "f1": metric,
            "observed_test_fpr": metric, "benign_alerts_per_10000_events": {"mean": 100, "std": 0},
            "macro_attack_family_recall": {"mean": .5, "std": 0},
            "batch_one_p95_ms": {"mean": 33.7, "std": 1.0}},
            "per_family": {"Bot": {"mean": .005, "std": 0, "n_runs": 1, "sample_count": 1952},
                           "DDoS": {"mean": .9, "std": 0, "n_runs": 1, "sample_count": 128025}}}},
        "selection": {"winner": "equal_weight_combined", "winner_threshold": 4.3, "candidates": [
            {"variant": "transformer_nll", "macro_attack_family_recall": 0.0, "precision": 0.0, "fpr": .01},
            {"variant": "equal_weight_combined", "macro_attack_family_recall": .8486,
             "precision": .7627, "fpr": .01}]},
        "ablation": {"bundle_seed": 42, "variants": {"combo_score": {
            "recall": .97, "precision": .92, "f1": .94, "observed_test_fpr": .038,
            "macro_attack_family_recall": .6}}},
    }
    text = render(summary)
    assert "50.00 ± 0.00%" in text
    assert "33.70 ± 1.00" in text
    assert "| Bot | 1,952 | 0.50 ± 0.00% |" in text
    assert text.index("| DDoS |") < text.index("| Bot |")
    assert "| equal_weight_combined (selected) | 84.86% | 76.27% | 1.00% |" in text
    assert "seed 42 bundle" in text
    assert "| combo_score | 97.00% | 60.00% | 92.00% | 3.80% |" in text
