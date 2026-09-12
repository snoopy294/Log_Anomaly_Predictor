import pandas as pd
from new import filter_train_by_label


def _df():
    return pd.DataFrame({
        "entity_id": ["a", "a", "a", "b"],
        "Label": ["BENIGN", "DoS Hulk", "BENIGN", "PortScan"],
        "timestamp": [1, 2, 3, 4],
    })


def test_empty_label_returns_df_unchanged():
    df = _df()
    out = filter_train_by_label(df, "")
    pd.testing.assert_frame_equal(out, df)


def test_nonempty_label_keeps_only_matching_rows():
    out = filter_train_by_label(_df(), "BENIGN")
    assert out["Label"].tolist() == ["BENIGN", "BENIGN"]
    assert out["entity_id"].tolist() == ["a", "a"]


def test_filter_returns_copy_not_view():
    df = _df()
    out = filter_train_by_label(df, "BENIGN")
    out["Label"] = "MUTATED"
    assert df["Label"].tolist() == ["BENIGN", "DoS Hulk", "BENIGN", "PortScan"]
