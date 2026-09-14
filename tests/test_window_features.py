import numpy as np
import pandas as pd

from new import (make_sequences_from_events, make_window_features,
                 fit_window_baselines, robust_window_z, WINDOW_FEATURES)

SEQ = 3


def _events():
    # entity "a": 4 identical events then 3 distinct ones; "b": too short for a window
    rows = []
    for i in range(4):
        rows.append(("a", i, "T0", "x", 10.0))
    for i, (tok, dst) in enumerate([("T1", "y"), ("T2", "z"), ("T3", "w")]):
        rows.append(("a", 4 + i, tok, dst, 100.0))
    rows += [("b", 0, "T0", "x", 1.0), ("b", 1, "T1", "x", 1.0)]
    df = pd.DataFrame(rows, columns=["entity_id", "timestamp", "token_str", "dst_id", "bytes"])
    df["Label"] = "BENIGN"
    df["event_row_id"] = np.arange(len(df))
    return df


TOK_MAP = {"T0": 2, "T1": 3, "T2": 4, "T3": 5}


def test_features_align_with_sequences():
    df = _events()
    for step in (1, 2):
        X, y, *_ = make_sequences_from_events(df, TOK_MAP, SEQ, step)
        feats = make_window_features(df, TOK_MAP, SEQ, step)
        assert list(feats.columns) == WINDOW_FEATURES
        assert len(feats) == len(X)
        np.testing.assert_array_equal(feats["repeat_last"].to_numpy(), (X[:, -1] == y).astype(np.float32))


def test_repetitive_vs_diverse_windows():
    feats = make_window_features(_events(), TOK_MAP, SEQ, 1)
    first, last = feats.iloc[0], feats.iloc[-1]
    # first window = 4 identical events
    assert first["tok_uniq"] == 1 and first["tok_top_share"] == 1
    assert first["dst_uniq"] == 1 and first["bytes_std"] == 0
    # last window = T0,T1,T2,T3 all distinct
    assert np.isclose(last["tok_uniq"], 4) and np.isclose(last["tok_top_share"], 0.25)
    assert np.isclose(last["dst_uniq"], 4)


def test_robust_z_entity_baseline_fallback_and_floor():
    train = pd.DataFrame({"f": [1.0] * 40 + [0.0, 10.0]})
    ents = np.array(["known"] * 40 + ["rare", "rare"], dtype=object)
    base = fit_window_baselines(train, ents, min_n=30)

    z = robust_window_z(pd.DataFrame({"f": [1.0, 2.0, 2.0]}),
                        np.array(["known", "known", "unseen"], dtype=object), base)
    assert list(z.columns) == ["z_f"]
    floor = 0.25 * train["f"].std(ddof=0)
    # known entity: median 1, MAD 0 -> scale floored (finite, non-zero)
    assert z["z_f"].iloc[0] == 0.0
    assert np.isclose(z["z_f"].iloc[1], 1.0 / floor)
    # "rare" (< min_n) is not a known entity; "unseen" falls back to the global
    # median (1.0) and global scale (MAD 0 -> same floor)
    assert "rare" not in base["e_med"].index
    assert np.isclose(z["z_f"].iloc[2], 1.0 / floor)


def test_robust_z_is_clipped():
    train = pd.DataFrame({"f": np.linspace(0, 1, 50)})
    base = fit_window_baselines(train, np.array(["e"] * 50, dtype=object), min_n=30)
    z = robust_window_z(pd.DataFrame({"f": [1e9, -1e9]}), np.array(["e", "e"], dtype=object), base, clip=50.0)
    assert z["z_f"].tolist() == [50.0, -50.0]
