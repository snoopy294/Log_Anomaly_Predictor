import pandas as pd
from new import split_time_within_groups


def _df():
    # Entity "victim" has two labels clustered in disjoint time blocks:
    # Hulk entirely early (ts 0-9), BENIGN entirely late (ts 10-19).
    # A naive entity-only chronological split would put all of Hulk in
    # train and all of BENIGN in val/test.
    rows = []
    for ts in range(10):
        rows.append({"entity_id": "victim", "Label": "Hulk", "timestamp": ts})
    for ts in range(10, 20):
        rows.append({"entity_id": "victim", "Label": "BENIGN", "timestamp": ts})
    return pd.DataFrame(rows)


def test_each_label_gets_proportional_train_val_test_slice():
    df_tr, df_va, df_te = split_time_within_groups(_df(), train_frac=0.7, val_frac=0.15)

    assert set(df_tr["Label"].unique()) == {"Hulk", "BENIGN"}
    assert set(df_va["Label"].unique()) == {"Hulk", "BENIGN"}
    assert set(df_te["Label"].unique()) == {"Hulk", "BENIGN"}


def test_within_label_chronological_order_preserved():
    df_tr, df_va, df_te = split_time_within_groups(_df(), train_frac=0.7, val_frac=0.15)

    hulk_tr = df_tr[df_tr["Label"] == "Hulk"]["timestamp"].tolist()
    hulk_va = df_va[df_va["Label"] == "Hulk"]["timestamp"].tolist()
    hulk_te = df_te[df_te["Label"] == "Hulk"]["timestamp"].tolist()
    assert hulk_tr == sorted(hulk_tr)
    assert max(hulk_tr) <= min(hulk_va + hulk_te, default=max(hulk_tr))
    assert (not hulk_va or not hulk_te) or max(hulk_va) <= min(hulk_te)


def test_no_label_column_falls_back_to_entity_only_split():
    df = pd.DataFrame({"entity_id": ["a"] * 10, "timestamp": range(10)})
    df_tr, df_va, df_te = split_time_within_groups(df, train_frac=0.7, val_frac=0.15)
    assert len(df_tr) == 7
    assert len(df_va) == 1
    assert len(df_te) == 2


def test_row_counts_conserved():
    df = _df()
    df_tr, df_va, df_te = split_time_within_groups(df, train_frac=0.7, val_frac=0.15)
    assert len(df_tr) + len(df_va) + len(df_te) == len(df)
