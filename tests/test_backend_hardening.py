import numpy as np
import pytest

import backend
import model as model_module
from new import build_transformer_next_event_model

SEQ_LEN = 8
VOCAB = ["TOK_A", "TOK_B", "TOK_C", "TOK_D"]


# ---------------- server settings / CORS ----------------

def test_debug_off_and_loopback_by_default():
    settings = backend.server_settings_from_env({})
    assert settings == {"host": "127.0.0.1", "port": 5000, "debug": False}


@pytest.mark.parametrize("value", ["0", "false", "no", "", "maybe"])
def test_debug_stays_off_unless_explicitly_enabled(value):
    assert backend.server_settings_from_env({"FLASK_DEBUG": value})["debug"] is False


def test_server_settings_read_from_env():
    settings = backend.server_settings_from_env(
        {"HOST": "0.0.0.0", "PORT": "8080", "FLASK_DEBUG": "1"})
    assert settings == {"host": "0.0.0.0", "port": 8080, "debug": True}


def test_flask_app_debug_off_at_import():
    assert backend.app.debug is False


def test_cors_defaults_to_localhost_only():
    origins = backend.cors_origins_from_env({})
    assert origins and all(("localhost" in o or "127.0.0.1" in o) for o in origins)


def test_cors_origins_parsed_from_env():
    assert backend.cors_origins_from_env({"CORS_ORIGINS": " https://a.example , https://b.example ,"}) \
        == ["https://a.example", "https://b.example"]


def test_cors_rejects_unlisted_origin_and_allows_localhost():
    client = backend.app.test_client()
    evil = client.get("/api/train/status", headers={"Origin": "https://evil.example"})
    assert "Access-Control-Allow-Origin" not in evil.headers
    local = client.get("/api/train/status", headers={"Origin": "http://localhost:5000"})
    assert local.headers.get("Access-Control-Allow-Origin") == "http://localhost:5000"


# ---------------- /api/explain attribution ----------------

@pytest.fixture(scope="module")
def tiny_model():
    return build_transformer_next_event_model(
        vocab_size=len(VOCAB) + 2, seq_len=SEQ_LEN, d_model=8, num_layers=1,
        num_heads=2, d_ff=16, dropout=0.0)


def test_get_important_events_returns_real_gradient(tiny_model):
    X = np.array([[2, 3, 4, 5, 2, 3, 4, 5]], dtype=np.int32)
    importance = model_module.get_important_events(tiny_model, X)
    assert importance is not None
    assert importance.shape == (SEQ_LEN,)
    assert np.all(np.isfinite(importance))
    assert np.any(importance > 0)


def test_explain_clone_does_not_change_live_model_predictions(tiny_model):
    X = np.array([[2, 3, 4, 5, 2, 3, 4, 5]], dtype=np.int32)
    before = np.asarray(tiny_model(X, training=False))
    model_module.get_important_events(tiny_model, X)
    after = np.asarray(tiny_model(X, training=False))
    np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-7)
    assert type(tiny_model.get_layer("tok_emb")).__name__ == "Embedding"


def test_explain_route_returns_non_null_attribution(monkeypatch, tiny_model):
    monkeypatch.setattr(backend, "MODEL", tiny_model)
    monkeypatch.setattr(backend, "MODEL_META", {"seq_len": SEQ_LEN})
    monkeypatch.setattr(backend, "VOCAB", VOCAB)
    events = [{"token_id": 2 + (i % len(VOCAB)), "timestamp": f"2017-07-03T09:00:{i:02d}"}
              for i in range(SEQ_LEN)]
    monkeypatch.setitem(backend.EVENT_BUFFER, "10.0.0.1", events)

    response = backend.app.test_client().post("/api/explain", json={"entity_id": "10.0.0.1"})

    assert response.status_code == 200, response.get_json()
    tokens = response.get_json()["tokens"]
    assert len(tokens) == SEQ_LEN
    importances = [t["importance"] for t in tokens]
    assert all(isinstance(v, float) and np.isfinite(v) for v in importances)
    assert any(v > 0 for v in importances)


def test_explain_route_503_without_model(monkeypatch):
    monkeypatch.setattr(backend, "MODEL", None)
    monkeypatch.setattr(backend, "MODEL_META", None)
    response = backend.app.test_client().post("/api/explain", json={"entity_id": "x"})
    assert response.status_code == 503
