"""Dataset adapters and label-blind temporal split contracts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


BENIGN_LABELS = {"", "0", "benign", "normal", "none", "nan"}
UNSW_REQUIRED = {"Stime", "dstip", "srcip", "proto", "dsport", "sbytes", "dbytes"}


def normalize_label(value: object, numeric_label: object | None = None) -> str:
    """Return a stable BENIGN label while retaining named attack families."""
    numeric = pd.to_numeric(pd.Series([numeric_label]), errors="coerce").iloc[0]
    if pd.notna(numeric) and int(numeric) == 0:
        return "BENIGN"
    text = "" if value is None else str(value).strip()
    if text.lower() in BENIGN_LABELS:
        return "BENIGN"
    return text or ("ATTACK" if pd.notna(numeric) and int(numeric) else "BENIGN")


def adapt_unsw_nb15(df: pd.DataFrame, *, drop_malformed: bool = True) -> pd.DataFrame:
    """Adapt the time-bearing UNSW-NB15 release to the detector event schema.

    The commonly distributed pre-split training/testing files omit source and
    destination IPs. They are deliberately rejected: zero-shot/entity behavior
    needs the full release containing ``Stime``, ``srcip`` and ``dstip``.
    """
    source = df.copy()
    source.columns = [str(c).strip() for c in source.columns]
    missing = sorted(UNSW_REQUIRED - set(source.columns))
    if missing:
        raise ValueError(f"UNSW-NB15 missing required full-release columns: {missing}")

    timestamp = pd.to_datetime(pd.to_numeric(source["Stime"], errors="coerce"), unit="s", utc=True)
    sbytes = pd.to_numeric(source["sbytes"], errors="coerce")
    dbytes = pd.to_numeric(source["dbytes"], errors="coerce")
    proto = source["proto"].astype("string").str.strip().str.lower()
    dsport = source["dsport"].astype("string").str.strip()
    attack = source["attack_cat"] if "attack_cat" in source else pd.Series("", index=source.index)
    numeric = source["label"] if "label" in source else pd.Series(np.nan, index=source.index)

    out = pd.DataFrame({
        "timestamp": timestamp,
        "entity_id": source["dstip"].astype("string").str.strip(),
        "dst_id": source["srcip"].astype("string").str.strip(),
        "event_type": proto + ":" + dsport,
        "bytes": sbytes + dbytes,
        "Label": [normalize_label(a, n) for a, n in zip(attack, numeric)],
    })
    malformed = (out["timestamp"].isna() | out["entity_id"].isna() |
                 out["dst_id"].isna() | proto.isna() | dsport.isna() |
                 out["bytes"].isna())
    if malformed.any() and not drop_malformed:
        raise ValueError(f"UNSW-NB15 contains {int(malformed.sum())} malformed required rows")
    out = out.loc[~malformed].copy()
    out["bytes"] = out["bytes"].clip(lower=0).astype(float)
    return out.sort_values("timestamp", kind="stable").reset_index(drop=True)


def chronological_split(df: pd.DataFrame, train_frac: float, val_frac: float):
    """Globally chronological, label-blind 3-way split."""
    if not 0 < train_frac < 1 or not 0 <= val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("fractions must satisfy 0 < train_frac and train_frac + val_frac < 1")
    ordered = df.sort_values("timestamp", kind="stable").copy()
    n = len(ordered)
    a, b = int(n * train_frac), int(n * (train_frac + val_frac))
    return ordered.iloc[:a].copy(), ordered.iloc[a:b].copy(), ordered.iloc[b:].copy()


def cicids_day_split(df: pd.DataFrame):
    """CICIDS contract: Jul 3 train, Jul 4 development/calibration, Jul 5-7 test."""
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    day = ts.dt.strftime("%Y-%m-%d")
    train = df.loc[day == "2017-07-03"].sort_values("timestamp", kind="stable").copy()
    development = df.loc[day == "2017-07-04"].sort_values("timestamp", kind="stable").copy()
    test = df.loc[day.isin(["2017-07-05", "2017-07-06", "2017-07-07"])] \
        .sort_values("timestamp", kind="stable").copy()
    if min(map(len, (train, development, test))) == 0:
        raise ValueError("CICIDS split requires events on July 3, July 4, and July 5-7 2017")
    return train, development, test


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_manifest(parts: dict[str, pd.DataFrame]) -> dict:
    result = {}
    for name, frame in parts.items():
        labels = frame.get("Label", pd.Series(dtype=str)).astype(str).value_counts().sort_index()
        ts = pd.to_datetime(frame["timestamp"], utc=True) if len(frame) else pd.Series(dtype="datetime64[ns, UTC]")
        result[name] = {
            "rows": int(len(frame)),
            "label_counts": {str(k): int(v) for k, v in labels.items()},
            "start": ts.min().isoformat() if len(ts) else None,
            "end": ts.max().isoformat() if len(ts) else None,
        }
    return result
