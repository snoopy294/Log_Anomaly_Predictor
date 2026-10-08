import pandas as pd
import pytest

from datasets import UNSW_REQUIRED
from scripts.assemble_unsw_full import assemble


def _names():
    required = sorted(UNSW_REQUIRED) + ["attack_cat", "Label"]
    return required + [f"filler_{i}" for i in range(49 - len(required))]


def _write_raw(raw_dir, names):
    pd.DataFrame({"No.": range(1, len(names) + 1), "Name": [f" {n} " for n in names],
                  "Type ": "x", "Description": "x"}).to_csv(raw_dir / "NUSW-NB15_features.csv", index=False)
    for i in range(1, 5):
        pd.DataFrame([list(range(len(names)))]).to_csv(raw_dir / f"UNSW-NB15_{i}.csv",
                                                       header=False, index=False)


def test_assemble_concatenates_four_headerless_parts_with_feature_names(tmp_path):
    _write_raw(tmp_path, _names())
    out = tmp_path / "unsw_nb15_full.csv"
    assert assemble(tmp_path, out) == 4
    assembled = pd.read_csv(out)
    assert list(assembled.columns) == _names()


def test_assemble_rejects_wrong_feature_count(tmp_path):
    _write_raw(tmp_path, _names()[:-1])
    with pytest.raises(ValueError, match="49"):
        assemble(tmp_path, tmp_path / "out.csv")
