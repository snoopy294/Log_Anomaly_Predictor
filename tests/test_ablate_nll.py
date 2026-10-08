import json

import pandas as pd

from scripts.ablate_nll import run_ablation
from tests.fakes import FakeModel, entity_events, fake_bundle


def _day(day):
    benign = [entity_events(f"h{k}", day, 20) for k in range(6)]
    attack = entity_events("attacker", day, 8, label="PortScan", varied_peers=True, bytes_base=50000.0)
    return pd.concat(benign + [attack], ignore_index=True)


def test_run_ablation_reports_three_variants_on_test_rows_only():
    result = run_ablation(_day("2017-07-04"), _day("2017-07-05"), fake_bundle(), FakeModel(),
                          target_fpr=0.5)
    assert set(result["variants"]) == {"combo_score", "combo_without_nll", "transformer_nll_z"}
    assert result["development_rows"] > 0 and result["test_rows"] > 0
    assert result["bundle_seed"] == 42
    for variant in result["variants"].values():
        point = variant["frozen_operating_point"]
        assert 0.0 <= point["observed_test_fpr"] <= 1.0
        assert point["calibration_population"]["labels_used_for_threshold"] is False
    json.dumps(result, allow_nan=False)


def test_ablation_thresholds_ignore_test_labels():
    dev, test = _day("2017-07-04"), _day("2017-07-05")
    a = run_ablation(dev, test, fake_bundle(), FakeModel(), target_fpr=0.5)
    b = run_ablation(dev, test.assign(Label="BENIGN"), fake_bundle(), FakeModel(), target_fpr=0.5)
    for name in a["variants"]:
        assert (a["variants"][name]["frozen_operating_point"]["threshold"]
                == b["variants"][name]["frozen_operating_point"]["threshold"])
