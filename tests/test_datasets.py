import pandas as pd
import pytest

from datasets import adapt_unsw_nb15, cicids_day_split


def test_unsw_mapping_and_benign_normalization():
    raw = pd.DataFrame({"Stime": [1_500_000_000, 1_500_000_001], "dstip": ["d1", "d2"],
                        "srcip": ["s1", "s2"], "proto": ["tcp", "udp"],
                        "dsport": [80, 53], "sbytes": [10, 20], "dbytes": [2, 3],
                        "attack_cat": ["Normal", "Exploits"], "label": [0, 1]})
    out = adapt_unsw_nb15(raw)
    assert out["entity_id"].tolist() == ["d1", "d2"]
    assert out["dst_id"].tolist() == ["s1", "s2"]
    assert out["event_type"].tolist() == ["tcp:80", "udp:53"]
    assert out["bytes"].tolist() == [12.0, 23.0]
    assert out["Label"].tolist() == ["BENIGN", "Exploits"]
    assert str(out["timestamp"].dt.tz) == "UTC"


def test_unsw_rejects_non_time_bearing_release():
    with pytest.raises(ValueError, match="full-release columns"):
        adapt_unsw_nb15(pd.DataFrame({"proto": ["tcp"]}))


def test_unsw_malformed_policy():
    raw = pd.DataFrame({"Stime": ["bad"], "dstip": ["d"], "srcip": ["s"],
                        "proto": ["tcp"], "dsport": [80], "sbytes": [1], "dbytes": [2]})
    assert adapt_unsw_nb15(raw).empty
    with pytest.raises(ValueError, match="malformed"):
        adapt_unsw_nb15(raw, drop_malformed=False)


def test_cicids_day_boundaries():
    df = pd.DataFrame({"timestamp": pd.to_datetime([
        "2017-07-03T23:59:59Z", "2017-07-04T00:00:00Z", "2017-07-05T00:00:00Z",
        "2017-07-07T23:59:59Z"]), "Label": ["BENIGN"] * 4})
    train, dev, test = cicids_day_split(df)
    assert [len(train), len(dev), len(test)] == [1, 1, 2]
