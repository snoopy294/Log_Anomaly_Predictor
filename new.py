#!/usr/bin/env python3
"""
new.py — main pipeline entry point
-----------------------------------
Transformer next-event model for security logs with:
- richer tokenization (event_type + dst_bucket + bytes_bucket)
- leak-safe bucketing/vocab fit on TRAIN only
- split modes within TRAIN dataset (time/entity/time_entity)
- evaluation + metrics_summary.json (baseline, perplexity, etc.)
- timeline plots per entity
- allowlist suppression (optional)
- alert packet export (seq_len context + target) + manifest.csv

Cross-dataset:
- --train_csv and optional --test_csv
- optional R adapter to convert CICIDS/ISCX flow CSVs into model schema:
    timestamp, entity_id, event_type, dst_id, bytes, Label
"""


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable
import subprocess
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_fscore_support, roc_curve)

# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)

def _strip_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _to_utc_datetime(series: pd.Series) -> pd.Series:
    # robust parse -> UTC
    return pd.to_datetime(series, errors="coerce", utc=True)

def run_r_cicids_adapter(in_csv: str, out_csv: str, rscript_bin: str, r_script: str):
    cmd = [rscript_bin, r_script, "--in_csv", in_csv, "--out_csv", out_csv]
    print("[R-ADAPTER]", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("[R-ADAPTER][STDOUT]\n", res.stdout)
        print("[R-ADAPTER][STDERR]\n", res.stderr)
        raise RuntimeError(f"R adapter failed with exit code {res.returncode}")
    if res.stdout.strip():
        print("[R-ADAPTER][STDOUT]\n", res.stdout)
    if res.stderr.strip():
        print("[R-ADAPTER][STDERR]\n", res.stderr)

def parse_args():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--train_csv", type=str, required=True, help="Training dataset CSV (clean schema).")
    p.add_argument("--test_csv", type=str, default="", help="Optional external test CSV.")

    # Keep these flags (base logic), but now CICIDS requires R adapter
    p.add_argument("--train_format", type=str, default="auto", choices=["auto", "clean", "cicids"],
                   help="How to interpret train_csv. If cicids, use --use_r_adapter.")
    p.add_argument("--test_format", type=str, default="auto", choices=["auto", "clean", "cicids"],
                   help="How to interpret test_csv. If cicids, use --use_r_adapter.")

    # sequence params
    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--step", type=int, default=1, help="sequence step size (1=sliding, 64=chunked)")
    p.add_argument("--min_events_per_entity", type=int, default=10)

    # splitting (applies to TRAIN dataset only; external test_csv is not split)
    p.add_argument("--split_mode", type=str, default="time",
                   choices=["time", "entity", "time_entity"])
    p.add_argument("--train_frac", type=float, default=0.7)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)

    # tokenization controls
    p.add_argument("--dst_top_n", type=int, default=200)
    p.add_argument("--bytes_num_buckets", type=int, default=8)

    # training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)

    # alerting
    p.add_argument("--top_k_alerts", type=int, default=300)
    p.add_argument("--alert_z_thresh", type=float, default=3.0)
    p.add_argument("--max_alerts_per_entity", type=int, default=10)

    # allowlist suppression
    p.add_argument("--allowlist_top_dst", type=int, default=0)
    p.add_argument("--allowlist_margin", type=float, default=1.5)

    # plotting
    p.add_argument("--plot_max_entities", type=int, default=50)
    p.add_argument("--plot", action="store_true")

    # io
    p.add_argument("--history_csv", type=str, default="outputs/training_history.csv")
    p.add_argument("--best_model_path", type=str, default="models/log_transformer.keras")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--append_history", action="store_true")

    # alert packet export
    p.add_argument("--export_alert_packets", action="store_true")
    p.add_argument("--alert_packets_dir", type=str, default="outputs/alert_packets")

    # output
    p.add_argument("--out_dir", type=str, default="outputs")

    # R adapter
    p.add_argument("--use_r_adapter", action="store_true",
                   help="If set, run an R script to convert CICIDS->model schema before Python parsing.")
    p.add_argument("--rscript_bin", type=str, default="Rscript")
    p.add_argument("--r_adapter_script", type=str, default="cicids_to_model_events.R")

    return p.parse_args()

# -----------------------------
# History + plots
# -----------------------------
def save_history_csv(history: keras.callbacks.History, path: str, append: bool = False):
    ensure_dir(os.path.dirname(path) or ".")
    new_df = pd.DataFrame(history.history).copy()
    new_df["epoch"] = np.arange(len(new_df), dtype=int)

    if append and os.path.exists(path):
        old_df = pd.read_csv(path)
        start = int(old_df["epoch"].max()) + 1 if ("epoch" in old_df.columns and len(old_df) > 0) else len(old_df)
        new_df["epoch"] = new_df["epoch"] + start
        out_df = pd.concat([old_df, new_df], ignore_index=True)
        out_df.to_csv(path, index=False)
        return out_df

    new_df.to_csv(path, index=False)
    return new_df

def plot_training_curves_from_df(hist_df: pd.DataFrame, out_dir: str):
    ensure_dir(out_dir)
    import matplotlib.pyplot as plt

    x = hist_df["epoch"] if "epoch" in hist_df.columns else np.arange(len(hist_df))

    plt.figure()
    if "loss" in hist_df.columns:
        plt.plot(x, hist_df["loss"], label="train_loss")
    if "val_loss" in hist_df.columns:
        plt.plot(x, hist_df["val_loss"], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training vs Validation Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_loss.png"), dpi=150)
    plt.close()

    acc_key = "acc" if "acc" in hist_df.columns else ("sparse_categorical_accuracy" if "sparse_categorical_accuracy" in hist_df.columns else None)
    val_acc_key = "val_acc" if "val_acc" in hist_df.columns else ("val_sparse_categorical_accuracy" if "val_sparse_categorical_accuracy" in hist_df.columns else None)

    plt.figure()
    if acc_key:
        plt.plot(x, hist_df[acc_key], label="train_acc")
    if val_acc_key:
        plt.plot(x, hist_df[val_acc_key], label="val_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Training vs Validation Accuracy")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_accuracy.png"), dpi=150)
    plt.close()

def plot_nll_hist(nll: np.ndarray, out_path: str):
    ensure_dir(os.path.dirname(out_path) or ".")
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(nll, bins=60)
    plt.xlabel("Negative Log-Likelihood (NLL)")
    plt.ylabel("Count")
    plt.title("NLL Distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_entity_timelines(all_scores_df: pd.DataFrame, alerts_df: pd.DataFrame, out_dir: str, max_entities: int = 50):
    ensure_dir(out_dir)
    import matplotlib.pyplot as plt

    df = all_scores_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df["_key"] = (
        df["entity_id"].astype(str) + "|" +
        df["timestamp"].astype("int64").astype(str) + "|" +
        df["true_next_event"].astype(str)
    )
    al = alerts_df.copy()
    al["timestamp"] = pd.to_datetime(al["timestamp"], utc=True)
    al["_key"] = (
        al["entity_id"].astype(str) + "|" +
        al["timestamp"].astype("int64").astype(str) + "|" +
        al["true_next_event"].astype(str)
    )
    alert_keys = set(al["_key"].tolist())
    df["is_alert"] = df["_key"].isin(alert_keys)

    ents_with_alerts = df.loc[df["is_alert"], "entity_id"].astype(str).unique().tolist()
    ents_all = df["entity_id"].astype(str).unique().tolist()
    ents = (ents_with_alerts + [e for e in ents_all if e not in set(ents_with_alerts)])[:max_entities]

    for ent in ents:
        g = df[df["entity_id"].astype(str) == ent].sort_values("timestamp")
        if len(g) < 2:
            continue

        plt.figure()
        plt.plot(g["timestamp"], g["entity_nll_z"], label="entity_nll_z")
        ga = g[g["is_alert"]]
        if len(ga) > 0:
            plt.scatter(ga["timestamp"], ga["entity_nll_z"], label="alert_point", zorder=3)

        plt.title(f"Entity timeline: {ent}")
        plt.xlabel("Time")
        plt.ylabel("Entity NLL z-score")
        plt.legend()
        plt.tight_layout()
        safe_ent = str(ent).replace("/", "_").replace("\\", "_").replace(" ", "_")
        plt.savefig(os.path.join(out_dir, f"{safe_ent}.png"), dpi=150)
        plt.close()

# -----------------------------
# Alert packet export
# -----------------------------
def export_alert_packets(alerts_df: pd.DataFrame, df_all_events: pd.DataFrame, seq_len: int, out_dir: str, raw_cols=None):
    ensure_dir(out_dir)
    if raw_cols is None:
        raw_cols = ["timestamp", "entity_id", "event_type", "dst_id", "bytes", "Label", "token_str", "event_row_id"]

    df_all = df_all_events.sort_values(["entity_id", "timestamp"]).copy()
    df_all["_pos_in_entity"] = df_all.groupby("entity_id").cumcount()

    id_to_pos = dict(
        zip(
            df_all["event_row_id"].astype(np.int64),
            zip(df_all["entity_id"].astype(str), df_all["_pos_in_entity"].astype(int))
        )
    )

    manifest_rows = []
    al = alerts_df.copy()
    al["timestamp"] = pd.to_datetime(al["timestamp"], utc=True)

    for i, row in al.reset_index(drop=True).iterrows():
        ent = str(row["entity_id"])
        end_rid = int(row["end_row_id"])
        ts = pd.to_datetime(row["timestamp"], utc=True)

        if end_rid not in id_to_pos:
            continue

        ent2, pos = id_to_pos[end_rid]
        if ent2 != ent:
            ent = ent2

        g = df_all[df_all["entity_id"].astype(str) == ent].sort_values("timestamp")
        start = max(0, pos - seq_len)
        end = pos
        packet = g.iloc[start:end + 1].copy()

        packet["is_target"] = False
        packet.iloc[-1, packet.columns.get_loc("is_target")] = True

        safe_ent = ent.replace("/", "_").replace("\\", "_").replace(" ", "_")
        fname = f"alert_{i:04d}_{safe_ent}_{ts.strftime('%Y%m%dT%H%M%SZ')}.csv"
        fpath = os.path.join(out_dir, fname)

        keep = [c for c in raw_cols if c in packet.columns] + ["is_target"]
        packet[keep].to_csv(fpath, index=False)

        manifest_rows.append({
            "alert_index": i,
            "entity_id": ent,
            "timestamp": ts.isoformat(),
            "end_row_id": end_rid,
            "packet_path": fpath,
            "entity_nll_z": float(row.get("entity_nll_z", np.nan)),
            "anomaly_score_nll": float(row.get("anomaly_score_nll", np.nan)),
            "true_next_event": row.get("true_next_event", ""),
            "reason": row.get("reason", ""),
        })

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(os.path.join(out_dir, "manifest.csv"), index=False)
    return manifest

# -----------------------------
# Dataset adapters
# -----------------------------
def adapt_clean_csv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = {"timestamp", "entity_id", "event_type"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"clean format missing required columns: {missing}")

    if "dst_id" not in df.columns:
        df["dst_id"] = ""
    if "bytes" not in df.columns:
        df["bytes"] = 0.0
    if "Label" not in df.columns:
        df["Label"] = ""

    df["timestamp"] = _to_utc_datetime(df["timestamp"])
    df = df.dropna(subset=["timestamp", "entity_id", "event_type"]).copy()

    df["bytes"] = pd.to_numeric(df["bytes"], errors="coerce").fillna(0.0)
    df["entity_id"] = df["entity_id"].astype(str)
    df["event_type"] = df["event_type"].astype(str)
    df["dst_id"] = df["dst_id"].astype(str)
    df["Label"] = df["Label"].astype(str)

    return df

def detect_format_from_df(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    if {"timestamp", "entity_id", "event_type"}.issubset(cols):
        return "clean"
    if {"Timestamp", "Source IP", "Destination IP", "Destination Port", "Protocol"}.issubset(cols):
        return "cicids"
    return "unknown"

def load_and_adapt_events(csv_path: str, fmt: str = "auto") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = _strip_colnames(df)

    if fmt == "auto":
        fmt = detect_format_from_df(df)

    if fmt == "clean":
        df2 = adapt_clean_csv(df)
    elif fmt == "cicids":
        # Python CICIDS adapter removed in this version; must use R converter first.
        raise ValueError(
            "Detected CICIDS/ISCX format but Python CICIDS adapter is disabled in this file.\n"
            "Use: --use_r_adapter --train_format cicids (and/or --test_format cicids)\n"
            "so the R script converts it to clean schema first."
        )
    else:
        raise ValueError("Could not detect/parse dataset. Use --train_format clean or --use_r_adapter for cicids.")

    df2 = df2.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
    df2["event_row_id"] = np.arange(len(df2), dtype=np.int64)
    return df2

def filter_entities(df: pd.DataFrame, min_events_per_entity: int) -> pd.DataFrame:
    counts = df["entity_id"].value_counts()
    keep = set(counts[counts >= min_events_per_entity].index)
    return df[df["entity_id"].isin(keep)].copy()

# -----------------------------
# Splitting (TRAIN dataset only)
# -----------------------------
def split_entity_holdout(df: pd.DataFrame, train_frac: float, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    ents = df["entity_id"].unique()
    rng.shuffle(ents)

    n = len(ents)
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))

    train_ents = set(ents[:n_train])
    val_ents = set(ents[n_train:n_train + n_val])
    test_ents = set(ents[n_train + n_val:])

    df_tr = df[df["entity_id"].isin(train_ents)].copy()
    df_va = df[df["entity_id"].isin(val_ents)].copy()
    df_te = df[df["entity_id"].isin(test_ents)].copy()
    return df_tr, df_va, df_te

def split_time_within_groups(df: pd.DataFrame, train_frac: float, val_frac: float):
    tr_parts, va_parts, te_parts = [], [], []
    for _, g in df.groupby("entity_id", sort=False):
        g2 = g.sort_values("timestamp")
        n = len(g2)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        tr_parts.append(g2.iloc[:n_train])
        va_parts.append(g2.iloc[n_train:n_train + n_val])
        te_parts.append(g2.iloc[n_train + n_val:])
    return pd.concat(tr_parts), pd.concat(va_parts), pd.concat(te_parts)

# -----------------------------
# Tokenization (fit on TRAIN only)
# -----------------------------
def fit_dst_vocab(df_train: pd.DataFrame, top_n: int):
    vc = df_train["dst_id"].astype(str).fillna("").value_counts()
    return set(vc.head(top_n).index.tolist())

def bucket_dst(dst: str, keep_set: set):
    s = "" if dst is None else str(dst)
    return f"DST={s}" if s in keep_set else "DST=OTHER"

def fit_bytes_bins(df_train: pd.DataFrame, num_buckets: int):
    x = df_train["bytes"].to_numpy(np.float64)
    x = np.log1p(np.clip(x, 0.0, None))
    qs = np.linspace(0, 1, num_buckets + 1)
    edges = np.quantile(x, qs)
    edges = np.unique(edges)
    if len(edges) <= 2:
        return np.array([x.min(), x.max() + 1e-6], dtype=np.float64)
    return edges.astype(np.float64)

def bucket_bytes(b: float, edges: np.ndarray):
    x = np.log1p(max(0.0, float(b)))
    idx = int(np.searchsorted(edges, x, side="right") - 1)
    idx = max(0, min(idx, len(edges) - 2))
    return f"BYTES_Q{idx}"

def make_token_strings(df: pd.DataFrame, dst_keep: set, bytes_edges: np.ndarray):
    dst_bucketed = df["dst_id"].apply(lambda v: bucket_dst(v, dst_keep))
    bytes_bucketed = df["bytes"].apply(lambda v: bucket_bytes(v, bytes_edges))
    return df["event_type"].astype(str) + "|" + dst_bucketed + "|" + bytes_bucketed

def build_vocab_from_buckets(dst_keep: set, bytes_edges: np.ndarray):
    n_b = max(1, len(bytes_edges) - 1)
    bytes_vocab = [f"BYTES_Q{i}" for i in range(n_b)]

    dst_vocab = [f"DST={d}" for d in sorted(map(str, dst_keep))]
    dst_vocab.append("DST=OTHER")

    event_types = [
        "TCP_REGISTERED", "TCP_EPHEMERAL", "UDP_REGISTERED",
        "TCP_WELL_KNOWN", "P0_WELL_KNOWN", "UDP_WELL_KNOWN",
        "UDP_EPHEMERAL"
    ]

    vocab = [f"{et}|{dst}|{bq}" for et in event_types for dst in dst_vocab for bq in bytes_vocab]
    tok = {v: i + 2 for i, v in enumerate(vocab)}  # 0 PAD, 1 UNK
    itok = {i: v for v, i in tok.items()}
    return vocab, tok, itok

# -----------------------------
# Sequence building
# -----------------------------
def make_sequences_from_events(df: pd.DataFrame, tok_map: dict, seq_len: int, step: int):
    df = df.sort_values(["entity_id", "timestamp"]).copy()
    df["tok_id"] = df["token_str"].map(tok_map).fillna(1).astype(np.int32)

    X_list, y_list, t_list, e_list, label_list, rid_list = [], [], [], [], [], []

    for ent, g in df.groupby("entity_id", sort=False):
        ev = g["tok_id"].to_numpy(np.int32)
        ts = g["timestamp"].astype("int64").to_numpy(np.int64)
        lbl = g["Label"].astype(str).to_numpy(object)
        rid = g["event_row_id"].to_numpy(np.int64)

        if len(ev) < seq_len + 1:
            continue

        for i in range(0, len(ev) - (seq_len + 1) + 1, step):
            X_list.append(ev[i:i + seq_len])
            y_list.append(ev[i + seq_len])
            t_list.append(ts[i + seq_len])
            e_list.append(ent)
            label_list.append(lbl[i + seq_len] if len(lbl) > 0 else "")
            rid_list.append(rid[i + seq_len])

    if not X_list:
        return (
            np.zeros((0, seq_len), np.int32),
            np.zeros((0,), np.int32),
            np.zeros((0,), np.int64),
            np.zeros((0,), object),
            np.zeros((0,), object),
            np.zeros((0,), np.int64),
        )

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int32)
    t = np.array(t_list, dtype=np.int64)
    e = np.array(e_list, dtype=object)
    lab = np.array(label_list, dtype=object)
    rid_end = np.array(rid_list, dtype=np.int64)
    return X, y, t, e, lab, rid_end

# -----------------------------
# Model
# -----------------------------
@register_keras_serializable()
class TakeLastToken(layers.Layer):
    def call(self, x):
        return x[:, -1, :]

def transformer_block(x, d_model, num_heads, d_ff, dropout):
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)
    ff = layers.Dense(d_ff, activation="relu")(x)
    ff = layers.Dropout(dropout)(ff)
    ff = layers.Dense(d_model)(ff)
    x = layers.Add()([x, ff])
    x = layers.LayerNormalization()(x)
    return x

def build_transformer_next_event_model(
    vocab_size: int,
    seq_len: int,
    d_model: int = 64,
    num_layers: int = 2,
    num_heads: int = 4,
    d_ff: int = 128,
    dropout: float = 0.3,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
):
    tokens = keras.Input(shape=(seq_len,), dtype="int32", name="tokens")
    x = layers.Embedding(vocab_size, d_model, mask_zero=True, name="tok_emb")(tokens)

    pos = tf.range(start=0, limit=seq_len, delta=1)
    pos_emb = layers.Embedding(seq_len, d_model, name="pos_emb")(pos)
    x = x + pos_emb

    for _ in range(num_layers):
        x = transformer_block(x, d_model, num_heads, d_ff, dropout)

    x = TakeLastToken(name="last_token")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(vocab_size, activation="softmax", name="next_event_softmax")(x)

    model = keras.Model(tokens, out)
    optimizer = keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            keras.metrics.CategoricalAccuracy(name="acc"),
            keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc"),
        ],
    )
    return model

# -----------------------------
# Scoring + metrics
# -----------------------------
def _make_predict_fn(model: keras.Model):
    # tf.function traces once per input shape (batch_size, then again for the
    # final partial batch) and reuses the compiled graph after that — unlike
    # calling model.predict() per batch, whose Python-level setup cost is
    # paid on every single call.
    @tf.function(reduce_retracing=True)
    def _predict(xb):
        return model(xb, training=False)
    return _predict

def compute_nll_only(model: keras.Model, X: np.ndarray, y: np.ndarray, batch_size=512):
    n = len(y)
    nll = np.empty((n,), dtype=np.float32)
    predict_fn = _make_predict_fn(model)
    # Chunk to bound memory — holding all n x vocab_size predictions in one
    # array at once OOMs on large datasets.
    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size]
        yb = y[i:i + batch_size]
        pb = predict_fn(xb).numpy()
        p_true = pb[np.arange(len(yb)), yb]
        nll[i:i + len(yb)] = -np.log(np.clip(p_true, 1e-9, 1.0))
    return nll

def compute_topk_for_alerts(model: keras.Model, X: np.ndarray, y: np.ndarray, k: int = 5, batch_size=512):
    n = len(y)
    nll = np.empty((n,), dtype=np.float32)
    top_ids = np.empty((n, k), dtype=np.int32)
    top_ps = np.empty((n, k), dtype=np.float32)
    predict_fn = _make_predict_fn(model)

    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size]
        yb = y[i:i + batch_size]
        pb = predict_fn(xb).numpy()

        p_true = pb[np.arange(len(yb)), yb]
        nll[i:i + len(yb)] = -np.log(np.clip(p_true, 1e-9, 1.0))

        ids = np.argsort(-pb, axis=1)[:, :k]
        ps = np.take_along_axis(pb, ids, axis=1)

        top_ids[i:i + len(yb)] = ids
        top_ps[i:i + len(yb)] = ps

    return nll, top_ids, top_ps


def fit_entity_nll_stats_train_only(entities_train: np.ndarray, nll_train: np.ndarray, min_n: int = 5):
    df = pd.DataFrame({"entity_id": entities_train.astype(str), "nll": nll_train.astype(float)})
    stats = df.groupby("entity_id")["nll"].agg(["mean", "std", "count"]).reset_index()
    stats.columns = ["entity_id", "mean_nll", "std_nll", "n_nll"]

    global_mean = float(df["nll"].mean())
    global_std = float(df["nll"].std(ddof=0))
    if not np.isfinite(global_std) or global_std <= 0:
        global_std = 1e-6

    # An entity with too few train sequences (or near-constant NLL) yields a
    # near-zero/NaN std. Flooring that to a tiny epsilon (as before) turns any
    # later deviation, however ordinary, into a z-score in the millions —
    # silently drowning out real anomalies for every other entity. Fall back
    # to the global std/mean instead: not entity-specific, but not degenerate.
    unreliable = (~np.isfinite(stats["std_nll"])) | (stats["std_nll"] <= 1e-3) | (stats["n_nll"] < min_n)
    stats.loc[unreliable, "mean_nll"] = global_mean
    stats.loc[unreliable, "std_nll"] = global_std
    stats = stats.drop(columns=["n_nll"])
    return stats, global_mean, global_std

def baseline_last_token_accuracy(X: np.ndarray, y: np.ndarray):
    if len(X) == 0:
        return float("nan")
    return float((X[:, -1] == y).mean())

def parse_dst_from_token(token_str: str) -> str:
    parts = str(token_str).split("|")
    for p in parts:
        if p.startswith("DST="):
            return p.replace("DST=", "")
    return ""

def score_anomalies_df(nll: np.ndarray, top_ids: np.ndarray, top_ps: np.ndarray, y: np.ndarray,
                       vocab: list, entities: np.ndarray, t: np.ndarray,
                       labels: np.ndarray, end_row_id: np.ndarray,
                       top_k: int,
                       z_thresh: float = 3.0, max_alerts_per_entity: int = 10,
                       entity_stats: pd.DataFrame | None = None,
                       global_mean: float | None = None,
                       global_std: float | None = None,
                       allowlist_dst: set[str] | None = None,
                       allowlist_margin: float = 1.5):
    df_tmp = pd.DataFrame({
        "entity_id": entities.astype(str),
        "nll": nll.astype(float)
    })

    if entity_stats is not None:
        stats = entity_stats.copy()
        stats["entity_id"] = stats["entity_id"].astype(str)
        df_tmp = df_tmp.merge(stats, on="entity_id", how="left")

        if global_mean is None:
            global_mean = float(np.mean(nll))
        if global_std is None or (not np.isfinite(global_std)) or global_std <= 0:
            global_std = float(np.std(nll)) if float(np.std(nll)) > 0 else 1e-6

        df_tmp["mean_nll"] = df_tmp["mean_nll"].fillna(global_mean)
        df_tmp["std_nll"] = df_tmp["std_nll"].fillna(global_std).replace(0, 1e-6)
    else:
        stats = df_tmp.groupby("entity_id")["nll"].agg(["mean", "std"]).reset_index()
        stats.columns = ["entity_id", "mean_nll", "std_nll"]
        df_tmp = df_tmp.merge(stats, on="entity_id", how="left")
        df_tmp["std_nll"] = df_tmp["std_nll"].fillna(1e-6).replace(0, 1e-6)

    entity_nll_z = (df_tmp["nll"] - df_tmp["mean_nll"]) / df_tmp["std_nll"]

    reasons = []
    for z in entity_nll_z:
        if z > 6:
            reasons.append("extreme deviation for entity")
        elif z > 4:
            reasons.append("unexpected event sequence")
        elif z > 3:
            reasons.append("rare but plausible behavior")
        else:
            reasons.append("normal variation")

    def tok_to_name(token_id: int) -> str:
        if token_id == 0:
            return "PAD"
        if token_id == 1:
            return "UNK"
        idx = token_id - 2
        return vocab[idx] if 0 <= idx < len(vocab) else "UNK"

    scores_all = pd.DataFrame({
        "entity_id": entities.astype(str),
        "timestamp": pd.to_datetime(t, utc=True),
        "end_row_id": end_row_id.astype(np.int64),
        "Label": labels.astype(str),
        "true_next_event": [tok_to_name(i) for i in y],
        "anomaly_score_nll": nll,
        "entity_nll_z": entity_nll_z.to_numpy(),
        "reason": reasons,
    })

    k = int(top_ids.shape[1])

    scores_all[f"top{k}_predicted_events"] = [
        ", ".join(tok_to_name(j) for j in top_ids[i]) for i in range(len(y))
    ]   

    scores_all[f"top{k}_probs"] = [
        ", ".join(f"{p:.3f}" for p in top_ps[i]) for i in range(len(y))
    ]

    scores_all["dst_parsed"] = scores_all["true_next_event"].apply(parse_dst_from_token)

    scores_sorted = scores_all.sort_values("entity_nll_z", ascending=False)
    cand = scores_sorted[scores_sorted["entity_nll_z"] >= float(z_thresh)].copy()

    if allowlist_dst:
        allowlist_dst = set(map(str, allowlist_dst))
        is_allow = cand["dst_parsed"].isin(allowlist_dst)
        cand = cand[~is_allow | (cand["entity_nll_z"] >= float(z_thresh + allowlist_margin))].copy()

    if "entity_id" in cand.columns and top_k > 0:
        cand = (cand.sort_values("entity_nll_z", ascending=False)
                    .groupby("entity_id", sort=False, as_index=False)
                    .head(int(max_alerts_per_entity)))

    alerts_df = cand.head(top_k).copy()
    return scores_all, alerts_df

def detection_metrics(scores_df, score_col, positive_labels=None, alert_z_thresh=3.0):
    """Compute detection metrics from scored DataFrame.
    Returns a dict safe to embed in metrics_summary.json.
    Positive class = any Label not in the Normal/benign set.
    """
    if positive_labels is None:
        positive_labels = {"Normal", "BENIGN", "0", ""}

    y_true = (~scores_df["Label"].isin(positive_labels)).astype(int).values
    scores = scores_df[score_col].values

    if len(np.unique(y_true)) < 2:
        return {
            "note": "single-class data — detection metrics undefined (no attack labels in dataset)",
            "score_col": score_col,
            "n_positive": int(y_true.sum()),
            "n_total": int(len(y_true)),
        }

    roc_auc = float(roc_auc_score(y_true, scores))
    pr_auc = float(average_precision_score(y_true, scores))

    # Metrics at operating threshold (entity_nll_z >= alert_z_thresh)
    y_pred = (scores >= alert_z_thresh).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    # Detection rate at low FPR targets
    fpr_arr, tpr_arr, _ = roc_curve(y_true, scores)

    def tpr_at_fpr(target_fpr):
        idx = np.searchsorted(fpr_arr, target_fpr, side="right") - 1
        idx = max(0, min(idx, len(tpr_arr) - 1))
        return float(tpr_arr[idx])

    return {
        "score_col": score_col,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision_at_thresh": float(prec),
        "recall_at_thresh": float(rec),
        "f1_at_thresh": float(f1),
        "alert_z_thresh": alert_z_thresh,
        "detection_at_fpr_1pct": tpr_at_fpr(0.01),
        "detection_at_fpr_0_1pct": tpr_at_fpr(0.001),
        "n_positive": int(y_true.sum()),
        "n_total": int(len(y_true)),
    }

# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    ensure_dir("models")
    ensure_dir(args.out_dir)
    out_dir = args.out_dir

    tf.config.optimizer.set_jit(False)

    # 1) Load + adapt TRAIN
    train_csv_to_use = args.train_csv
    train_fmt_to_use = args.train_format

    if train_fmt_to_use == "cicids" and not args.use_r_adapter:
        raise RuntimeError("train_format=cicids requires --use_r_adapter (Python CICIDS parser is disabled).")

    if args.use_r_adapter and train_fmt_to_use in ("cicids", "auto"):
        tmp_out = os.path.join(args.out_dir, "train_r_adapted.csv")
        run_r_cicids_adapter(train_csv_to_use, tmp_out, args.rscript_bin, args.r_adapter_script)
        train_csv_to_use = tmp_out
        train_fmt_to_use = "clean"

    df = load_and_adapt_events(train_csv_to_use, fmt=train_fmt_to_use)

    df = filter_entities(df, args.min_events_per_entity)
    if len(df) == 0:
        raise RuntimeError("No TRAIN data after filtering entities. Lower --min_events_per_entity.")

    # 1b) Optional external test dataset
    df_external_test = None
    if args.test_csv:
        test_csv_to_use = args.test_csv
        test_fmt_to_use = args.test_format

        if test_fmt_to_use == "cicids" and not args.use_r_adapter:
            raise RuntimeError("test_format=cicids requires --use_r_adapter (Python CICIDS parser is disabled).")

        if args.use_r_adapter and test_fmt_to_use in ("cicids", "auto"):
            tmp_out = os.path.join(args.out_dir, "test_r_adapted.csv")
            run_r_cicids_adapter(test_csv_to_use, tmp_out, args.rscript_bin, args.r_adapter_script)
            test_csv_to_use = tmp_out
            test_fmt_to_use = "clean"

        df_external_test = load_and_adapt_events(test_csv_to_use, fmt=test_fmt_to_use)
        df_external_test = filter_entities(df_external_test, args.seq_len + 1)


    # 2) Split TRAIN events BEFORE fitting bucketing/vocab
    if args.split_mode == "time":
        df_tr, df_va, df_te_internal = split_time_within_groups(df, args.train_frac, args.val_frac)
    elif args.split_mode == "entity":
        df_tr, df_va, df_te_internal = split_entity_holdout(df, args.train_frac, args.val_frac, args.seed)
    else:  # time_entity
        df_tr0, df_va0, df_te0 = split_entity_holdout(df, args.train_frac, args.val_frac, args.seed)
        df_tr, _, _ = split_time_within_groups(df_tr0, 0.85, 0.0)
        df_va, _, _ = split_time_within_groups(df_va0, 0.50, 0.0)
        df_te_internal, _, _ = split_time_within_groups(df_te0, 0.50, 0.0)

    # 3) Fit tokenization pieces on TRAIN ONLY
    dst_keep = fit_dst_vocab(df_tr, top_n=args.dst_top_n)
    bytes_edges = fit_bytes_bins(df_tr, num_buckets=args.bytes_num_buckets)

    # 4) Token strings
    for d in (df_tr, df_va, df_te_internal):
        d["token_str"] = make_token_strings(d, dst_keep, bytes_edges)
    if df_external_test is not None:
        df_external_test["token_str"] = make_token_strings(df_external_test, dst_keep, bytes_edges)

    df_tr = df_tr.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
    df_va = df_va.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
    df_te_internal = df_te_internal.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)

    # Allowlist from TRAIN only
    allowlist_dst = set()
    if args.allowlist_top_dst and args.allowlist_top_dst > 0:
        vc = df_tr["dst_id"].astype(str).fillna("").value_counts()
        allowlist_dst = set(vc.head(int(args.allowlist_top_dst)).index.tolist())

    # 5) Vocab (fixed grid)
    vocab, tok_map, _ = build_vocab_from_buckets(dst_keep, bytes_edges)
    vocab_size = len(vocab) + 2  # PAD=0, UNK=1

    # Resume safety check
    def model_vocab_size(m): return int(m.output_shape[-1])

    if args.resume and os.path.exists(args.best_model_path):
        print(f"Resuming from existing model: {args.best_model_path}")
        loaded = keras.models.load_model(args.best_model_path)
        if model_vocab_size(loaded) != vocab_size:
            print(f"[WARN] Saved model vocab_size={model_vocab_size(loaded)} but current run vocab_size={vocab_size}.")
            print("[WARN] Not resuming. Training new model from scratch.")
            model = build_transformer_next_event_model(vocab_size=vocab_size, seq_len=args.seq_len)
        else:
            model = loaded
    else:
        print("Training a new model from scratch")
        model = build_transformer_next_event_model(vocab_size=vocab_size, seq_len=args.seq_len)

    # 6) Sequences
    Xtr, ytr, ttr, etr, ltr, rid_tr = make_sequences_from_events(df_tr, tok_map, args.seq_len, args.step)
    Xva, yva, tva, eva, lva, rid_va = make_sequences_from_events(df_va, tok_map, args.seq_len, args.step)
    Xte_i, yte_i, tte_i, ete_i, lte_i, rid_te_i = make_sequences_from_events(df_te_internal, tok_map, args.seq_len, args.step)

    if len(Xtr) == 0 or len(Xva) == 0:
        raise RuntimeError("Not enough sequences after TRAIN split. Try lowering --seq_len, --min_events_per_entity, or using --step 1.")

    Xte_ext = yte_ext = tte_ext = ete_ext = lte_ext = rid_te_ext = None
    if df_external_test is not None:
        Xte_ext, yte_ext, tte_ext, ete_ext, lte_ext, rid_te_ext = make_sequences_from_events(df_external_test, tok_map, args.seq_len, args.step)

    # 7) Datasets
    def to_onehot(x, y): return x, tf.one_hot(y, depth=vocab_size)

    ds_tr = (tf.data.Dataset.from_tensor_slices((Xtr, ytr))
             .shuffle(min(50000, len(Xtr)))
             .batch(args.batch_size)
             .map(to_onehot, num_parallel_calls=tf.data.AUTOTUNE)
             .prefetch(tf.data.AUTOTUNE))

    ds_va = (tf.data.Dataset.from_tensor_slices((Xva, yva))
             .batch(args.batch_size)
             .map(to_onehot, num_parallel_calls=tf.data.AUTOTUNE)
             .prefetch(tf.data.AUTOTUNE))

    print("Model ready. Starting fit()...")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(args.best_model_path, monitor="val_loss", save_best_only=True),
    ]
    history = model.fit(ds_tr, validation_data=ds_va, epochs=args.epochs, callbacks=callbacks, verbose=1)
    model.save(args.best_model_path.replace(".keras", "_last.keras"))

    # 8) Save metadata
    meta = {
        "seq_len": int(args.seq_len),
        "step": int(args.step),
        "min_events_per_entity": int(args.min_events_per_entity),
        "split_mode": args.split_mode,
        "train_frac": float(args.train_frac),
        "val_frac": float(args.val_frac),
        "dst_top_n": int(args.dst_top_n),
        "bytes_num_buckets": int(args.bytes_num_buckets),
        "bytes_edges_log1p": bytes_edges.tolist(),
        "vocab_size": int(vocab_size),
        "vocab_preview": vocab[:50],
        "token_ids": {"PAD": 0, "UNK": 1, "KNOWN_START": 2},
        "allowlist_top_dst": int(args.allowlist_top_dst),
        "allowlist_margin": float(args.allowlist_margin),
        "train_csv": args.train_csv,
        "test_csv": args.test_csv,
        "train_format": args.train_format,
        "test_format": args.test_format,
    }
    meta_path = args.best_model_path.replace(".keras", "_meta.json")
    ensure_dir(os.path.dirname(meta_path) or ".")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # 9) Save training history + plots
    hist_path = args.history_csv
    hist_df = save_history_csv(history, hist_path, append=args.append_history)
    if args.plot:
        plot_training_curves_from_df(hist_df, out_dir=out_dir)

    # 10) Evaluate
    best_model = keras.models.load_model(args.best_model_path)

    def eval_split(name: str, X, y):
        if X is None or len(X) == 0:
            return {"loss": float("nan"), "acc": float("nan"), "top5_acc": float("nan"),
                    "baseline_last_token_acc": float("nan"), "perplexity": float("nan"),
                    "mean_nll": float("nan"), "p95_nll": float("nan"), "n_sequences": 0}
        y_oh = tf.one_hot(y, depth=vocab_size)
        loss, acc, top5 = best_model.evaluate(X, y_oh, batch_size=512, verbose=0)
        nll = compute_nll_only(best_model, X, y)
        return {
            "loss": float(loss),
            "acc": float(acc),
            "top5_acc": float(top5),
            "baseline_last_token_acc": baseline_last_token_accuracy(X, y),
            "perplexity": float(np.exp(np.mean(nll))),
            "mean_nll": float(np.mean(nll)),
            "p95_nll": float(np.quantile(nll, 0.95)),
            "n_sequences": int(len(X)),
        }

    val_metrics = eval_split("val", Xva, yva)
    int_test_metrics = eval_split("internal_test", Xte_i, yte_i)
    ext_test_metrics = eval_split("external_test", Xte_ext, yte_ext) if Xte_ext is not None else None

    unk_rate_ext = None
    if Xte_ext is not None and len(Xte_ext) > 0:
        unk_rate_ext = float((Xte_ext == 1).mean())

    metrics_summary = {
        "train": {
            "n_events": int(len(df)),
            "n_entities": int(df["entity_id"].nunique()),
            "n_sequences": int(len(Xtr)),
        },
        "val": val_metrics,
        "internal_test": int_test_metrics,
        "external_test": ext_test_metrics,
        "external_test_unk_rate": unk_rate_ext,
    }
    metrics_summary["lm_metrics_note"] = "acc/top5_acc/perplexity are next-event language model metrics, NOT detection accuracy"

    nll_val = compute_nll_only(best_model, Xva, yva)
    plot_nll_hist(nll_val, out_path=os.path.join(out_dir, "nll_hist_val.png"))

    if Xte_ext is not None and len(Xte_ext) > 0:
        nll_ext = compute_nll_only(best_model, Xte_ext, yte_ext)
        plot_nll_hist(nll_ext, out_path=os.path.join(out_dir, "nll_hist_external_test.png"))
    elif len(Xte_i) > 0:
        nll_int = compute_nll_only(best_model, Xte_i, yte_i)
        plot_nll_hist(nll_int, out_path=os.path.join(out_dir, "nll_hist_internal_test.png"))


    # 11) Alerts (same logic as your version)
    nll_tr_only = compute_nll_only(best_model, Xtr, ytr)
    ent_stats_df, global_mean, global_std = fit_entity_nll_stats_train_only(etr, nll_tr_only)


    if Xte_ext is not None and len(Xte_ext) > 0:
        X_alert, y_alert, t_alert, e_alert, l_alert, rid_alert = Xte_ext, yte_ext, tte_ext, ete_ext, lte_ext, rid_te_ext
        df_events_for_packets = df_external_test
        alert_tag = "external_test"
    else:
        X_alert = np.concatenate([Xtr, Xva, Xte_i], axis=0)
        y_alert = np.concatenate([ytr, yva, yte_i], axis=0)
        t_alert = np.concatenate([ttr, tva, tte_i], axis=0)
        e_alert = np.concatenate([etr, eva, ete_i], axis=0)
        l_alert = np.concatenate([ltr, lva, lte_i], axis=0)
        rid_alert = np.concatenate([rid_tr, rid_va, rid_te_i], axis=0)
        df_events_for_packets = pd.concat([df_tr, df_va, df_te_internal], ignore_index=True)
        alert_tag = "train_val_internal_test"

    nll_alert, top_ids_alert, top_ps_alert = compute_topk_for_alerts(best_model, X_alert, y_alert, k=5)

    scores_all, alerts = score_anomalies_df(
        nll_alert, top_ids_alert, top_ps_alert, y_alert,
        vocab, e_alert, t_alert, l_alert, rid_alert,
        top_k=args.top_k_alerts,
        z_thresh=args.alert_z_thresh,
        max_alerts_per_entity=args.max_alerts_per_entity,
        entity_stats=ent_stats_df,
        global_mean=global_mean,
        global_std=global_std,
        allowlist_dst=allowlist_dst if len(allowlist_dst) else None,
        allowlist_margin=args.allowlist_margin,
    )

    det_metrics = detection_metrics(scores_all, score_col="entity_nll_z", alert_z_thresh=args.alert_z_thresh)
    metrics_summary["detection"] = det_metrics
    with open(os.path.join(out_dir, "metrics_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)

    alerts_path = os.path.join(out_dir, f"log_alerts_{alert_tag}.csv")
    alerts.to_csv(alerts_path, index=False)

    timelines_dir = os.path.join(out_dir, f"entity_timelines_{alert_tag}")
    plot_entity_timelines(scores_all, alerts, out_dir=timelines_dir, max_entities=args.plot_max_entities)

    if args.export_alert_packets:
        packets_dir = args.alert_packets_dir
        if not os.path.isabs(packets_dir):
            packets_dir = os.path.join(out_dir, packets_dir)
        export_alert_packets(alerts_df=alerts, df_all_events=df_events_for_packets, seq_len=args.seq_len, out_dir=packets_dir)

    print("\nSaved:")
    print(f"  {args.best_model_path}  (best by val_loss)")
    print(f"  {meta_path}")
    print(f"  {hist_path}")
    print(f"  {os.path.join(out_dir, 'metrics_summary.json')}")
    print(f"  {alerts_path}")
    print(f"  {timelines_dir}/<entity>.png")
    if args.export_alert_packets:
        print(f"  {packets_dir}/manifest.csv")
        print(f"  {packets_dir}/alert_*.csv")
    if args.plot:
        print(f"  {os.path.join(out_dir, 'training_loss.png')}")
        print(f"  {os.path.join(out_dir, 'training_accuracy.png')}")
    print(f"  {os.path.join(out_dir, 'nll_hist_val.png')}")
    if Xte_ext is not None and len(Xte_ext) > 0:
        print(f"  {os.path.join(out_dir, 'nll_hist_external_test.png')}")
    elif len(Xte_i) > 0:
        print(f"  {os.path.join(out_dir, 'nll_hist_internal_test.png')}")

if __name__ == "__main__":
    main()
