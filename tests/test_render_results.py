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
