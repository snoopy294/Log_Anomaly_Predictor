import numpy as np

from new import build_transformer_next_event_model, compute_nll_only, compute_topk_for_alerts


def test_on_device_topk_matches_numpy_reference():
    rng = np.random.default_rng(0)
    vocab, seq, n = 50, 4, 37  # n not a multiple of batch_size -> partial last batch
    model = build_transformer_next_event_model(vocab_size=vocab, seq_len=seq)
    X = rng.integers(2, vocab, size=(n, seq)).astype(np.int32)
    y = rng.integers(2, vocab, size=(n,)).astype(np.int32)

    probs = model.predict(X, verbose=0)
    ref_nll = -np.log(np.clip(probs[np.arange(n), y], 1e-9, 1.0))
    ref_ids = np.argsort(-probs, axis=1)[:, :5]

    nll, ids, ps = compute_topk_for_alerts(model, X, y, k=5, batch_size=16)
    np.testing.assert_allclose(nll, ref_nll, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(ps, np.take_along_axis(probs, ref_ids, axis=1), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(np.take_along_axis(probs, ids, axis=1), ps, rtol=1e-5)
    np.testing.assert_allclose(compute_nll_only(model, X, y, batch_size=16), ref_nll, rtol=1e-4, atol=1e-5)
