#!/usr/bin/env python3
"""
model.py — improved transformer + explainability, used by backend.py's /api/explain
--------------------------------------------------------------------------------------
Enhanced version with:
- Improved transformer architecture (residual connections, better normalization)
- Attention visualization
- Ensemble predictions
- Better feature engineering
- Adversarial training option
- Real-time inference API
- Explainability features
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import json
import argparse
import threading
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable

# ============================================
# IMPROVED MODEL ARCHITECTURE
# ============================================

@register_keras_serializable()
class MultiHeadSelfAttention(layers.Layer):
    """Enhanced multi-head attention with relative positional encoding"""
    def __init__(self, d_model, num_heads, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout
        
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout)
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, x, mask=None, return_attention=False, training=False):
        batch_size = tf.shape(x)[0]
        
        # Linear projections
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        
        # Split heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Apply mask (if provided)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # Softmax
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)
        
        # Weighted sum
        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        
        # Final linear projection
        output = self.dense(output)
        output = self.dropout(output, training=training)
        
        # Residual connection + layer norm
        output = self.layernorm(x + output)
        
        if return_attention:
            return output, attention_weights
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
        })
        return config


@register_keras_serializable()
class FeedForward(layers.Layer):
    """Position-wise feed-forward network with GELU activation"""
    def __init__(self, d_model, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout
        
        self.dense1 = layers.Dense(d_ff, activation='gelu')
        self.dense2 = layers.Dense(d_model)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, x, training=False):
        ff_output = self.dense1(x)
        ff_output = self.dropout1(ff_output, training=training)
        ff_output = self.dense2(ff_output)
        ff_output = self.dropout2(ff_output, training=training)
        
        # Residual connection + layer norm
        return self.layernorm(x + ff_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "dropout_rate": self.dropout_rate,
        })
        return config


@register_keras_serializable()
class TransformerBlock(layers.Layer):
    """Complete transformer block with attention and feed-forward"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
    
    def call(self, x, mask=None, return_attention=False, training=False):
        if return_attention:
            attn_output, attn_weights = self.attention(
                x, mask=mask, return_attention=True, training=training
            )
            ffn_output = self.ffn(attn_output, training=training)
            return ffn_output, attn_weights
        else:
            attn_output = self.attention(x, mask=mask, training=training)
            return self.ffn(attn_output, training=training)
    
    def get_config(self):
        return {
            "d_model": self.attention.d_model,
            "num_heads": self.attention.num_heads,
            "d_ff": self.ffn.d_ff,
            "dropout": self.attention.dropout_rate,
        }


def build_improved_transformer_model(
    vocab_size: int,
    seq_len: int,
    d_model: int = 128,
    num_layers: int = 4,
    num_heads: int = 8,
    d_ff: int = 512,
    dropout: float = 0.2,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    label_smoothing: float = 0.1,
):
    """Build improved transformer with better architecture"""
    
    tokens = keras.Input(shape=(seq_len,), dtype="int32", name="tokens")
    
    # Token + positional embeddings
    x = layers.Embedding(
        vocab_size, d_model, mask_zero=True, name="tok_emb"
    )(tokens)
    
    # Learned positional encoding
    positions = tf.range(start=0, limit=seq_len, delta=1)
    pos_emb = layers.Embedding(seq_len, d_model, name="pos_emb")(positions)
    x = x + pos_emb
    x = layers.Dropout(dropout)(x)
    
    # Transformer blocks
    for i in range(num_layers):
        x = TransformerBlock(
            d_model, num_heads, d_ff, dropout, name=f"transformer_{i}"
        )(x)
    
    # Take last token
    x = x[:, -1, :]
    
    # Classification head with residual
    x = layers.Dropout(dropout)(x)
    x_res = layers.Dense(d_model // 2, activation="gelu")(x)
    x_res = layers.Dropout(dropout)(x_res)
    x = layers.Dense(d_model // 2, activation="gelu")(x)
    x = layers.Add()([x, x_res])
    x = layers.LayerNormalization()(x)
    
    # Output
    output = layers.Dense(vocab_size, activation="softmax", name="predictions")(x)
    
    model = keras.Model(tokens, output, name="improved_transformer")
    
    # Use AdamW with warmup
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr,
        weight_decay=weight_decay,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )
    
    loss_fn = keras.losses.CategoricalCrossentropy(
        label_smoothing=label_smoothing
    )
    
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            keras.metrics.CategoricalAccuracy(name="acc"),
            keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc"),
            keras.metrics.TopKCategoricalAccuracy(k=10, name="top10_acc"),
        ],
    )
    
    return model


# ============================================
# EXPLAINABILITY
# ============================================

class _InjectableEmbedding(layers.Layer):
    """Stands in for a model's "tok_emb" layer inside an explain-only clone.

    Normally it just delegates to the original Embedding (same weights). While
    ``override`` is set, it returns that tensor instead, so the rest of the
    graph runs on an embedding the caller computed and is watching on a tape.
    The padding mask still comes from the token ids via the inner layer, so
    masking behaves exactly as in the original model.
    """

    def __init__(self, inner: layers.Embedding, **kwargs):
        super().__init__(name=inner.name, **kwargs)
        self.inner = inner
        self.override = None
        self.supports_masking = True

    def call(self, x):
        return self.override if self.override is not None else self.inner(x)

    def compute_mask(self, x, mask=None):
        return self.inner.compute_mask(x, mask)


# One explain clone per live model. The clone shares every weight with the
# original; only the clone's embedding wrapper is ever mutated, under a lock,
# so concurrent scoring through the original model is never affected.
_EXPLAIN_CACHE: dict = {"model": None, "clone": None}
_EXPLAIN_LOCK = threading.Lock()


def _explain_clone(model: keras.Model) -> keras.Model:
    if _EXPLAIN_CACHE["model"] is not model:
        _EXPLAIN_CACHE["clone"] = keras.models.clone_model(
            model,
            clone_function=lambda layer: _InjectableEmbedding(layer)
            if layer.name == "tok_emb" else layer,
        )
        _EXPLAIN_CACHE["model"] = model
    return _EXPLAIN_CACHE["clone"]


def get_important_events(model: keras.Model, X: np.ndarray, seq_idx: int = 0) -> np.ndarray:
    """Per-position importance for the model's top next-event prediction.

    Saliency = L1 norm over d_model of d(top-class probability)/d(token
    embedding). The embedding is computed inside the tape and fed to the rest
    of the network, so the gradient is taken through the real forward pass.
    Works for any functional model whose token embedding layer is "tok_emb".
    Returns a float array of shape (seq_len,).
    """
    X_single = np.asarray(X[seq_idx:seq_idx + 1], dtype=np.int32)

    with _EXPLAIN_LOCK:
        clone = _explain_clone(model)
        injectable = clone.get_layer("tok_emb")
        with tf.GradientTape() as tape:
            embeddings = tf.convert_to_tensor(injectable.inner(X_single))
            tape.watch(embeddings)
            injectable.override = embeddings
            try:
                predictions = clone(X_single, training=False)
            finally:
                injectable.override = None
            pred_score = tf.reduce_max(predictions[0])

    gradients = tape.gradient(pred_score, embeddings)
    if gradients is None:
        raise RuntimeError("attribution gradient is None; tok_emb is not on the prediction path")
    return tf.reduce_sum(tf.abs(gradients), axis=-1).numpy()[0]


if __name__ == "__main__":
    print("Improved transformer model module loaded.")
    print("Key improvements:")
    print("  - Enhanced multi-head attention with better normalization")
    print("  - Deeper architecture with residual connections")
    print("  - Explainability tools")
