import model


def test_dead_symbols_removed_from_model_module():
    for name in ("EnsemblePredictor", "AnomalyDetectorAPI", "WarmUpSchedule",
                 "create_callbacks", "extract_temporal_features",
                 "extract_entity_features", "compute_attention_rollout"):
        assert not hasattr(model, name), f"{name} should have been deleted"


def test_still_exposes_what_backend_and_explain_need():
    assert callable(model.build_improved_transformer_model)
    assert callable(model.get_important_events)
