"""Assemble the time-bearing UNSW-NB15 release into one headed CSV.

The official full release ships four headerless CSVs (UNSW-NB15_1..4.csv) plus
NUSW-NB15_features.csv, whose "Name" column lists the 49 column names in order.
Download all five from the UNSW-NB15 project page into one directory first.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import UNSW_REQUIRED

PARTS = [f"UNSW-NB15_{i}.csv" for i in range(1, 5)]
FEATURES_FILE = "NUSW-NB15_features.csv"
EXPECTED_COLUMNS = 49


def read_feature_names(features_csv: Path) -> list[str]:
    features = pd.read_csv(features_csv, encoding="latin-1")
    features.columns = [str(c).strip() for c in features.columns]
    return features["Name"].astype(str).str.strip().tolist()


def assemble(raw_dir: Path, out_csv: Path) -> int:
    raw_dir, out_csv = Path(raw_dir), Path(out_csv)
    names = read_feature_names(raw_dir / FEATURES_FILE)
    if len(names) != EXPECTED_COLUMNS:
        raise ValueError(f"expected {EXPECTED_COLUMNS} feature names, found {len(names)}")
    missing = sorted(UNSW_REQUIRED - set(names))
    if missing:
        raise ValueError(f"feature list is missing required columns: {missing}")
    frames = [pd.read_csv(raw_dir / part, header=None, names=names, low_memory=False,
                          encoding="latin-1") for part in PARTS]
    full = pd.concat(frames, ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_csv, index=False)
    return len(full)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw_unsw")
    parser.add_argument("--out", default="data/unsw_nb15_full.csv")
    args = parser.parse_args()
    rows = assemble(Path(args.raw_dir), Path(args.out))
    print(f"Wrote {rows} rows to {args.out}")


if __name__ == "__main__":
    main()
