import pandas as pd

from new import detection_metrics_by_label


def test_multiple_attack_labels_scored_against_all_benign_rows():
    df = pd.DataFrame({
        "Label": ["BENIGN"] * 4 + ["DoS Hulk"] * 3 + ["PortScan"] * 3,
        "score": [0.1, 0.2, 0.1, 0.2] + [0.9, 0.8, 0.95] + [0.05, 0.06, 0.04],
    })
    out = detection_metrics_by_label(df, score_col="score")

    assert set(out.keys()) == {"DoS Hulk", "PortScan"}

    # DoS Hulk scores are all higher than benign -> perfect separation, AUC 1.0
    assert out["DoS Hulk"]["roc_auc"] == 1.0
    assert out["DoS Hulk"]["n_positive"] == 3
    assert out["DoS Hulk"]["n_benign_compared"] == 4

    # PortScan scores are all lower than benign -> AUC 0.0 (each attack label is
    # compared independently against the same full benign set)
    assert out["PortScan"]["roc_auc"] == 0.0
    assert out["PortScan"]["n_positive"] == 3
    assert out["PortScan"]["n_benign_compared"] == 4


def test_label_with_zero_rows_is_skipped_not_errored():
    # "DDoS" appears as a categorical value with zero actual rows in the df
    # (simulated via a categorical dtype so .unique() would still surface it
    # if the function iterated over dtype categories instead of present rows).
    labels = pd.Categorical(
        ["BENIGN", "BENIGN", "DoS Hulk"],
        categories=["BENIGN", "DoS Hulk", "DDoS"],
    )
    df = pd.DataFrame({
        "Label": labels,
        "score": [0.1, 0.2, 0.9],
    })
    out = detection_metrics_by_label(df, score_col="score")

    assert "DDoS" not in out
    assert "DoS Hulk" in out
    assert out["DoS Hulk"]["n_positive"] == 1
    assert out["DoS Hulk"]["n_benign_compared"] == 2


def test_all_benign_returns_empty_dict():
    df = pd.DataFrame({
        "Label": ["BENIGN", "BENIGN", "BENIGN"],
        "score": [0.1, 0.2, 0.3],
    })
    out = detection_metrics_by_label(df, score_col="score")
    assert out == {}
