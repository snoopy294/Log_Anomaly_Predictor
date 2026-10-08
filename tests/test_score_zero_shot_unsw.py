import json

import pandas as pd

from scripts.score_zero_shot_unsw import run_zero_shot
from tests.fakes import FakeModel, entity_events, fake_bundle


def test_zero_shot_reports_unk_rate_and_frozen_point():
    events = pd.concat([entity_events("h1", "2015-01-22", 10),
                        entity_events("h2", "2015-01-22", 10, label="Exploits")], ignore_index=True)
    scored, metrics = run_zero_shot(events, fake_bundle(vocabulary=()), FakeModel())
    assert metrics["token_unk_rate"] == 1.0
    assert metrics["tuning_on_unsw"] is False
    assert metrics["event_type_scheme"] == "cicids"
    assert metrics["frozen_operating_point"]["threshold"] == 1.0
    assert len(scored) > 0
    json.dumps(metrics, allow_nan=False)
