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


def test_train_endpoint_always_rejects_with_guidance_to_new_py(monkeypatch):
    client = backend.app.test_client()

    monkeypatch.setattr(backend, "DETECTOR_RUNTIME", None)
    response_no_bundle = client.post("/api/train", json={})
    assert response_no_bundle.status_code == 409
    assert "new.py" in response_no_bundle.get_json()["message"]

    monkeypatch.setattr(backend, "DETECTOR_RUNTIME", object())
    response_with_bundle = client.post("/api/train", json={})
    assert response_with_bundle.status_code == 409
    assert "new.py" in response_with_bundle.get_json()["message"]


def test_train_status_route_unaffected():
    client = backend.app.test_client()
    response = client.get("/api/train/status")
    assert response.status_code == 200
    body = response.get_json()
    assert set(body.keys()) == {"is_training", "progress", "accuracy",
                                "top5_accuracy", "last_trained", "error"}


def test_upload_csv_route_removed():
    client = backend.app.test_client()
    response = client.post("/api/upload_csv")
    assert response.status_code == 404


def test_compute_anomaly_score_returns_none_without_detector_runtime(monkeypatch):
    monkeypatch.setattr(backend, "DETECTOR_RUNTIME", None)
    monkeypatch.setattr(backend, "MODEL", object())  # legacy branch would have used this
    monkeypatch.setattr(backend, "MODEL_META", {"seq_len": 4})
    # A full buffer so a legacy fallback would get past its length check.
    monkeypatch.setitem(backend.EVENT_BUFFER, "some_entity", [{"token_id": 2}] * 4)
    result = backend.compute_anomaly_score("some_entity", {"event_type": "tcp:80"})
    assert result is None
