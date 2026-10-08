import pytest

from detector_bundle import DetectorRuntime
from tests.fakes import FakeModel, entity_events, fake_bundle


def _last_result(runtime):
    results = [runtime.score_event(e) for e in
               entity_events("h", "2017-07-04", 3).to_dict("records")]
    assert results[-1] is not None
    return results[-1]


def test_ablation_score_drops_named_feature_from_mean():
    result = _last_result(DetectorRuntime(fake_bundle(), FakeModel(),
                                          ablations={"without_nll": ("nll",)}))
    # fake_bundle has median 0 / scale 1, so each contribution is |feature|.
    assert result["score_without_nll"] * 7 == pytest.approx(result["score"] * 8 - abs(result["nll"]))


def test_ablation_leaves_shipped_score_unchanged():
    plain = _last_result(DetectorRuntime(fake_bundle(), FakeModel()))
    ablated = _last_result(DetectorRuntime(fake_bundle(), FakeModel(),
                                           ablations={"without_nll": ("nll",)}))
    assert ablated["score"] == plain["score"]
    assert ablated["is_anomaly"] == plain["is_anomaly"]


def test_no_ablations_adds_no_extra_score_keys():
    result = _last_result(DetectorRuntime(fake_bundle(), FakeModel()))
    assert [k for k in result if k.startswith("score_")] == []


def test_unknown_ablation_feature_is_rejected():
    with pytest.raises(ValueError, match="unknown features"):
        DetectorRuntime(fake_bundle(), FakeModel(), ablations={"x": ("bogus",)})


def test_ablation_dropping_every_feature_is_rejected():
    bundle = fake_bundle()
    with pytest.raises(ValueError, match="every feature"):
        DetectorRuntime(bundle, FakeModel(),
                        ablations={"x": tuple(bundle["window_baselines"]["features"])})
