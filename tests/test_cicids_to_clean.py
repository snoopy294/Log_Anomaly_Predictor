import pandas as pd
from cicids_to_clean import convert


def _raw_cicids_df():
    return pd.DataFrame({
        "Timestamp": ["03/07/2017 08:55:58", "03/07/2017 08:56:01"],
        "Source IP": ["172.16.0.1", "192.168.10.5"],
        "Destination IP": ["192.168.10.50", "192.168.10.50"],
        "Destination Port": [80, 443],
        "Protocol": [6, 6],
        "Label": ["DoS Hulk", "BENIGN"],
        "Total Length of Fwd Packets": [100, 200],
        "Total Length of Bwd Packets": [50, 60],
    })


def test_entity_id_is_destination_ip():
    out = convert(_raw_cicids_df())
    assert out["entity_id"].tolist() == ["192.168.10.50", "192.168.10.50"]


def test_dst_id_column_holds_source_ip():
    out = convert(_raw_cicids_df())
    assert out["dst_id"].tolist() == ["172.16.0.1", "192.168.10.5"]


def test_label_and_bytes_unaffected_by_swap():
    out = convert(_raw_cicids_df())
    assert out["Label"].tolist() == ["DoS Hulk", "BENIGN"]
    assert out["bytes"].tolist() == [150.0, 260.0]


def test_parse_timestamp_utc_maps_twelve_hour_afternoon_times_after_morning():
    from cicids_to_clean import parse_timestamp_utc

    raw = pd.Series(["06/07/2017 08:55:58", "6/7/2017 12:30", "6/7/2017 1:05", "06/07/2017 03:45:00"])
    parsed = parse_timestamp_utc(raw)
    assert list(parsed.dt.hour) == [8, 12, 13, 15]
    assert parsed.is_monotonic_increasing
    assert (parsed.dt.strftime("%Y-%m-%d") == "2017-07-06").all()
