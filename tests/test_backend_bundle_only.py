import backend


def test_missing_bundle_leaves_everything_none(monkeypatch, tmp_path):
    monkeypatch.setattr(backend, "MODEL", None)
    monkeypatch.setattr(backend, "MODEL_META", None)
    monkeypatch.setattr(backend, "ENTITY_STATS", None)
    monkeypatch.setattr(backend, "VOCAB", None)
    monkeypatch.setattr(backend, "TOKENIZER", None)
    monkeypatch.setattr(backend, "DETECTOR_RUNTIME", None)
    monkeypatch.setitem(backend.CONFIG, "bundle_path", str(tmp_path / "no_bundle_here"))

    backend.load_model_and_config()

    assert backend.MODEL is None
    assert backend.MODEL_META is None
    assert backend.DETECTOR_RUNTIME is None
    assert backend.TOKENIZER is None


def test_config_no_longer_has_legacy_model_paths():
    assert "model_path" not in backend.CONFIG
    assert "meta_path" not in backend.CONFIG
    assert "stats_path" not in backend.CONFIG
