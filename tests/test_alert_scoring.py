import numpy as np
import pandas as pd

from new import calibrate_threshold, select_alerts, entity_nll_z_from_stats


def test_calibrate_threshold_hits_target_fpr():
    rng = np.random.default_rng(0)
    benign = rng.normal(size=100_000)
    thr = calibrate_threshold(benign, target_fpr=0.01)
    assert np.isclose((benign >= thr).mean(), 0.01, atol=0.001)


def test_calibrate_threshold_ignores_nan_and_empty():
    assert calibrate_threshold(np.array([np.nan, 1.0, 2.0, 3.0]), target_fpr=0.0) == 3.0
    assert np.isnan(calibrate_threshold(np.array([]), target_fpr=0.01))


def _scores():
    return pd.DataFrame({
        "entity_id": ["a", "a", "a", "b", "b"],
        "s": [5.0, 4.0, 1.0, 3.0, 2.5],
        "dst_parsed": ["x", "allow", "x", "x", "allow"],
    })


def test_select_alerts_threshold_sort_and_per_entity_cap():
    al = select_alerts(_scores(), "s", thresh=2.0, top_k=10, max_per_entity=1)
    assert al["s"].tolist() == [5.0, 3.0]


def test_select_alerts_allowlist_requires_margin():
    al = select_alerts(_scores(), "s", thresh=2.0, top_k=10, max_per_entity=10,
                       allowlist_dst={"allow"}, allowlist_margin=1.5)
    # 4.0 on an allowlisted dst clears 2.0 + 1.5; 2.5 does not
    assert al["s"].tolist() == [5.0, 4.0, 3.0]


def test_entity_nll_z_from_stats_falls_back_to_global():
    stats = pd.DataFrame({"entity_id": ["a"], "mean_nll": [1.0], "std_nll": [0.5]})
    z = entity_nll_z_from_stats(np.array([2.0, 3.0]), np.array(["a", "unseen"], dtype=object),
                                stats, global_mean=1.0, global_std=2.0)
    np.testing.assert_allclose(z, [2.0, 1.0])
