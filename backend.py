#!/usr/bin/env python3
"""
flask_api.py
-----------
Flask REST API for the anomaly detection system.
Connects the React dashboard with the TensorFlow model.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
from datetime import datetime, timedelta
import threading
import queue
from typing import Dict, List, Optional
import pickle
import sys

# Import training pipeline from new.py and improved model from model.py
from new import (
    load_and_adapt_events, filter_entities,
    split_time_within_groups, fit_dst_vocab, fit_bytes_bins,
    make_token_strings, build_vocab_from_buckets,
    make_sequences_from_events, build_transformer_next_event_model,
)
from model import (
    build_improved_transformer_model,
    compute_attention_rollout,
    get_important_events
)

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global model and configuration
MODEL = None
MODEL_META = None
VOCAB = None
TOKENIZER = None
ENTITY_STATS = None
EVENT_BUFFER = {}  # Store recent events per entity
ALERT_QUEUE = queue.Queue()

# Training state — tracks real accuracy from training
TRAINING_STATE = {
    "is_training": False,
    "progress": "",
    "accuracy": 0.0,
    "top5_accuracy": 0.0,
    "last_trained": None,
    "error": None,
}

# Performance tracking
PREVIOUS_STATS = {"totalEvents": 0, "anomaliesDetected": 0, "activeEntities": 0}
PERFORMANCE_METRICS = {
    "inference_latency_samples": [],  # last N latency measurements in ms
    "total_events_scored": 0,
    "anomalies_flagged": 0,
}

# Configuration
CONFIG = {
    "model_path": "models/log_transformer.keras",
    "meta_path": "models/log_transformer_meta.json",
    "stats_path": "outputs/entity_stats.csv",
    "max_buffer_size": 1000,
    "alert_threshold": 3.0,
}


# ============================================
# CUSTOM LAYERS
# ============================================

@keras.utils.register_keras_serializable()
class TakeLastToken(keras.layers.Layer):
    def call(self, x):
        return x[:, -1, :]

# ============================================
# INITIALIZATION
# ============================================

def load_model_and_config():
    """Load the trained model and configuration"""
    global MODEL, MODEL_META, ENTITY_STATS, VOCAB, TOKENIZER
    
    try:
        if os.path.exists(CONFIG["model_path"]):
            MODEL = keras.models.load_model(CONFIG["model_path"])
            print(f"+ Model loaded from {CONFIG['model_path']}")
        
        if os.path.exists(CONFIG["meta_path"]):
            with open(CONFIG["meta_path"], 'r') as f:
                MODEL_META = json.load(f)
            print(f"+ Metadata loaded")
        
        if os.path.exists(CONFIG["stats_path"]):
            ENTITY_STATS = pd.read_csv(CONFIG["stats_path"])
            print(f"+ Entity stats loaded")
        
        # Build vocab and tokenizer from meta
        if MODEL_META:
            vocab_preview = MODEL_META.get("vocab_preview", [])
            VOCAB = vocab_preview
            # Build reverse tokenizer
            TOKENIZER = {v: i + 2 for i, v in enumerate(VOCAB)}
            TOKENIZER["PAD"] = 0
            TOKENIZER["UNK"] = 1
            print(f"+ Tokenizer initialized with {len(TOKENIZER)} tokens")
            
    except Exception as e:
        print(f"! Error loading model: {e}")


# ============================================
# UTILITY FUNCTIONS
# ============================================

def tokenize_event(event: Dict) -> int:
    """Convert event to token ID"""
    # Build token string
    event_type = event.get("event_type", "UNK")
    dst = event.get("dst_id", "OTHER")
    bytes_bucket = event.get("bytes_bucket", 0)
    
    token_str = f"{event_type}|DST={dst}|BYTES_Q{bytes_bucket}"
    
    return TOKENIZER.get(token_str, 1)  # 1 = UNK


def get_entity_buffer(entity_id: str) -> List[Dict]:
    """Get event buffer for entity"""
    if entity_id not in EVENT_BUFFER:
        EVENT_BUFFER[entity_id] = []
    return EVENT_BUFFER[entity_id]


def add_event_to_buffer(entity_id: str, event: Dict):
    """Add event to entity buffer"""
    # Ensure initialized
    if TOKENIZER is None:
        load_model_and_config()
        
    if TOKENIZER is None:
        # Fallback if still failed
        print("Warning: Tokenizer not initialized, skipping event ingest")
        return

    buffer = get_entity_buffer(entity_id)
    buffer.append({
        "timestamp": event.get("timestamp", datetime.now().isoformat()),
        "token_id": tokenize_event(event),
        "raw_event": event
    })
    
    # Keep buffer size manageable
    if len(buffer) > CONFIG["max_buffer_size"]:
        EVENT_BUFFER[entity_id] = buffer[-CONFIG["max_buffer_size"]:]


def predict_next_event(entity_id: str, return_top_k: int = 5) -> Optional[Dict]:
    """Predict next event for entity"""
    if MODEL is None or MODEL_META is None:
        return None
    
    buffer = get_entity_buffer(entity_id)
    seq_len = MODEL_META.get("seq_len", 64)
    
    if len(buffer) < seq_len:
        return None
    
    # Get last seq_len events
    recent = buffer[-seq_len:]
    X = np.array([e["token_id"] for e in recent], dtype=np.int32).reshape(1, -1)
    
    # Predict with latency tracking
    t0 = time.perf_counter()
    probs = MODEL.predict(X, verbose=0)[0]
    latency_ms = (time.perf_counter() - t0) * 1000
    PERFORMANCE_METRICS["inference_latency_samples"].append(latency_ms)
    # Keep only last 100 samples
    if len(PERFORMANCE_METRICS["inference_latency_samples"]) > 100:
        PERFORMANCE_METRICS["inference_latency_samples"] = PERFORMANCE_METRICS["inference_latency_samples"][-100:]
    
    # Get top-k
    top_indices = np.argsort(probs)[-return_top_k:][::-1]
    top_probs = probs[top_indices]
    
    # Map indices to event names
    top_events = []
    for idx, prob in zip(top_indices, top_probs):
        if idx == 0:
            event_name = "PAD"
        elif idx == 1:
            event_name = "UNK"
        elif idx - 2 < len(VOCAB):
            event_name = VOCAB[idx - 2]
        else:
            event_name = "UNK"
        
        top_events.append({
            "event": event_name,
            "probability": float(prob)
        })
    
    return {
        "entity_id": entity_id,
        "timestamp": recent[-1]["timestamp"],
        "top_predictions": top_events,
        "entropy": float(-(probs * np.log(probs + 1e-10)).sum())
    }


def compute_anomaly_score(entity_id: str, actual_event: Dict) -> Optional[Dict]:
    """Compute anomaly score for actual next event"""
    if MODEL is None or MODEL_META is None:
        return None
    
    buffer = get_entity_buffer(entity_id)
    seq_len = MODEL_META.get("seq_len", 64)
    
    if len(buffer) < seq_len:
        return None
    
    # Get last seq_len events
    recent = buffer[-seq_len:]
    X = np.array([e["token_id"] for e in recent], dtype=np.int32).reshape(1, -1)
    
    # Predict with latency tracking
    t0 = time.perf_counter()
    probs = MODEL.predict(X, verbose=0)[0]
    latency_ms = (time.perf_counter() - t0) * 1000
    PERFORMANCE_METRICS["inference_latency_samples"].append(latency_ms)
    if len(PERFORMANCE_METRICS["inference_latency_samples"]) > 100:
        PERFORMANCE_METRICS["inference_latency_samples"] = PERFORMANCE_METRICS["inference_latency_samples"][-100:]
    PERFORMANCE_METRICS["total_events_scored"] += 1
    
    # Get actual token ID
    actual_token = tokenize_event(actual_event)
    actual_prob = probs[actual_token] if actual_token < len(probs) else 1e-9
    
    # Compute NLL
    nll = -np.log(np.clip(actual_prob, 1e-9, 1.0))
    
    # Compute z-score if we have entity stats
    z_score = None
    if ENTITY_STATS is not None:
        entity_stat = ENTITY_STATS[ENTITY_STATS["entity_id"] == entity_id]
        if len(entity_stat) > 0:
            mean_nll = float(entity_stat["mean_nll"].iloc[0])
            std_nll = float(entity_stat["std_nll"].iloc[0])
            z_score = (nll - mean_nll) / (std_nll + 1e-6)
    
    is_anomaly = (z_score and z_score > CONFIG["alert_threshold"]) or nll > 5.0
    
    result = {
        "entity_id": entity_id,
        "timestamp": datetime.now().isoformat(),
        "nll": float(nll),
        "z_score": float(z_score) if z_score is not None else None,
        "actual_probability": float(actual_prob),
        "is_anomaly": bool(is_anomaly),
    }
    
    # Add to alert queue if anomaly
    if is_anomaly:
        ALERT_QUEUE.put(result)
        PERFORMANCE_METRICS["anomalies_flagged"] += 1
    
    return result


# ============================================
# API ENDPOINTS
# ============================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    global PREVIOUS_STATS
    
    total_events = sum(len(buffer) for buffer in EVENT_BUFFER.values())
    active_entities = len(EVENT_BUFFER)
    
    # Get recent alerts
    alerts = []
    while not ALERT_QUEUE.empty():
        try:
            alert = ALERT_QUEUE.get_nowait()
            alerts.append(alert)
        except queue.Empty:
            break
    
    anomalies_detected = len(alerts)
    
    # Use real accuracy from training (0.0 if never trained)
    model_accuracy = TRAINING_STATE["accuracy"]
    
    # Compute trends (% change from previous snapshot)
    def calc_trend(current, previous):
        if previous == 0:
            return 0.0 if current == 0 else 100.0
        return round(((current - previous) / previous) * 100, 1)
    
    events_trend = calc_trend(total_events, PREVIOUS_STATS["totalEvents"])
    anomalies_trend = calc_trend(anomalies_detected, PREVIOUS_STATS["anomaliesDetected"])
    entities_trend = calc_trend(active_entities, PREVIOUS_STATS["activeEntities"])
    
    # Update previous stats for next comparison
    PREVIOUS_STATS = {
        "totalEvents": total_events,
        "anomaliesDetected": anomalies_detected,
        "activeEntities": active_entities,
    }
    
    # Compute average inference latency
    latency_samples = PERFORMANCE_METRICS["inference_latency_samples"]
    avg_latency_ms = round(sum(latency_samples) / len(latency_samples), 1) if latency_samples else 0.0
    
    # Compute anomaly rate (proxy for false positive rate without ground truth)
    total_scored = PERFORMANCE_METRICS["total_events_scored"]
    anomaly_rate = round((PERFORMANCE_METRICS["anomalies_flagged"] / total_scored) * 100, 2) if total_scored > 0 else 0.0
    
    # Get model engine name from metadata
    engine_name = "No Model Loaded"
    if MODEL is not None and MODEL_META is not None:
        engine_name = MODEL_META.get("architecture", MODEL.name or "transformer").upper()
    elif MODEL is not None:
        engine_name = (MODEL.name or "transformer").upper()
    
    # Sequence length from metadata
    seq_len = MODEL_META.get("seq_len", 0) if MODEL_META else 0
    
    return jsonify({
        "totalEvents": total_events,
        "anomaliesDetected": anomalies_detected,
        "activeEntities": active_entities,
        "modelAccuracy": model_accuracy,
        "eventsTrend": events_trend,
        "anomaliesTrend": anomalies_trend,
        "entitiesTrend": entities_trend,
        "inferenceLatencyMs": avg_latency_ms,
        "anomalyRate": anomaly_rate,
        "engineName": engine_name,
        "seqLen": seq_len,
        "alertThreshold": CONFIG["alert_threshold"],
        "maxBufferSize": CONFIG["max_buffer_size"],
        "modelLoaded": MODEL is not None,
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/events/ingest', methods=['POST'])
def ingest_event():
    """Ingest a new event"""
    try:
        event = request.json
        entity_id = event.get("entity_id")
        
        if not entity_id:
            return jsonify({"error": "entity_id is required"}), 400
        
        # Add to buffer
        add_event_to_buffer(entity_id, event)
        
        # Compute anomaly score
        score = compute_anomaly_score(entity_id, event)
        
        return jsonify({
            "success": True,
            "anomaly_score": score,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict next event for entity"""
    try:
        data = request.json
        entity_id = data.get("entity_id")
        top_k = data.get("top_k", 5)
        
        if not entity_id:
            return jsonify({"error": "entity_id is required"}), 400
        
        prediction = predict_next_event(entity_id, return_top_k=top_k)
        
        if prediction is None:
            return jsonify({
                "error": "Insufficient data for prediction",
                "message": f"Need at least {MODEL_META.get('seq_len', 64)} events"
            }), 400
        
        return jsonify(prediction)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get recent alerts"""
    limit = request.args.get('limit', 50, type=int)
    
    # Collect alerts from queue
    alerts = []
    temp_alerts = []
    
    while not ALERT_QUEUE.empty() and len(alerts) < limit:
        try:
            alert = ALERT_QUEUE.get_nowait()
            alerts.append(alert)
            temp_alerts.append(alert)
        except queue.Empty:
            break
    
    # Put alerts back in queue
    for alert in temp_alerts:
        ALERT_QUEUE.put(alert)
    
    # Format alerts
    formatted_alerts = []
    for i, alert in enumerate(alerts):
        severity = "high" if alert.get("z_score", 0) > 5 else "medium" if alert.get("z_score", 0) > 3.5 else "low"
        
        formatted_alerts.append({
            "id": i + 1,
            "timestamp": alert.get("timestamp"),
            "entity": alert.get("entity_id"),
            "eventType": "Unknown",  # Would need to decode from token
            "score": alert.get("z_score", alert.get("nll", 0)),
            "severity": severity,
            "reason": f"Anomaly score: {alert.get('z_score', 'N/A')}"
        })
    
    return jsonify(formatted_alerts)


@app.route('/api/timeline', methods=['GET'])
def get_timeline():
    """Get timeline data for charts"""
    hours = request.args.get('hours', 24, type=int)
    
    # Generate timeline data (mock for now - would aggregate from actual events)
    now = datetime.now()
    timeline = []
    
    for i in range(hours):
        time_point = now - timedelta(hours=hours - i - 1)
        
        # Count events in this hour (simplified)
        events_count = 0
        anomalies_count = 0
        nll_values = []
        
        for entity_id, buffer in EVENT_BUFFER.items():
            for event in buffer:
                event_time = datetime.fromisoformat(event["timestamp"].replace('Z', '+00:00'))
                if event_time.hour == time_point.hour:
                    events_count += 1
        
        timeline.append({
            "time": time_point.strftime("%H:00"),
            "events": events_count,
            "anomalies": anomalies_count,
            "nll": 0.0
        })
    
    return jsonify(timeline)


def run_r_cicids_adapter(in_csv, out_csv):
    """Run cicids_into_clean.R to convert CICIDS format to clean format"""
    import subprocess
    r_script = os.path.join(os.path.dirname(__file__), "cicids_into_clean.R")
    if not os.path.exists(r_script):
        raise FileNotFoundError(f"R adapter script not found: {r_script}")
    
    # Try Rscript from PATH
    rscript_bin = "Rscript"
    cmd = [rscript_bin, r_script, "--in_csv", in_csv, "--out_csv", out_csv, "--verbose"]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"R script failed (exit {result.returncode}):\n{result.stderr}")
    
    if not os.path.exists(out_csv):
        raise RuntimeError(f"R script ran but output file not created: {out_csv}")
    
    print(f"R adapter output: {result.stdout}")
    return out_csv


def detect_csv_format(csv_path):
    """Detect if a CSV is CICIDS raw format or clean format"""
    df_peek = pd.read_csv(csv_path, nrows=3)
    # Strip whitespace from column names (CICIDS files often have leading spaces)
    cols = set(c.strip() for c in df_peek.columns)
    
    if {"timestamp", "entity_id", "event_type"}.issubset(cols):
        return "clean"
    if {"Timestamp", "Source IP", "Destination IP", "Destination Port", "Protocol"}.issubset(cols):
        return "cicids"
    return "unknown"


@app.route('/api/train', methods=['POST'])
def train_model():
    """Trigger model training in background thread"""
    global TRAINING_STATE
    
    if TRAINING_STATE["is_training"]:
        return jsonify({"success": False, "message": "Training already in progress"}), 409
    
    try:
        config = request.json or {}
        epochs = config.get("epochs", 2)
        batch_size = config.get("batchSize", 32)
        learning_rate = config.get("learningRate", 0.0001)
        
        def run_training():
            global MODEL, MODEL_META, TRAINING_STATE
            try:
                TRAINING_STATE["is_training"] = True
                TRAINING_STATE["error"] = None
                TRAINING_STATE["progress"] = "Loading data..."
                
                train_csv = os.path.join(os.path.dirname(__file__), "data", "train_data.csv")
                if not os.path.exists(train_csv):
                    raise FileNotFoundError(f"Training data not found: {train_csv}")
                
                # Auto-detect format: if CICIDS, run R adapter first
                fmt = detect_csv_format(train_csv)
                
                if fmt == "cicids":
                    TRAINING_STATE["progress"] = "Running cicids_into_clean.R to preprocess CICIDS data..."
                    clean_csv = os.path.join(os.path.dirname(__file__), "data", "train_data_clean.csv")
                    run_r_cicids_adapter(train_csv, clean_csv)
                    train_csv = clean_csv
                    fmt = "clean"
                elif fmt == "unknown":
                    raise ValueError(
                        "Unrecognized CSV format. Expected either:\n"
                        "  • CICIDS format (Timestamp, Source IP, Destination IP, Destination Port, Protocol)\n"
                        "  • Clean format (timestamp, entity_id, event_type, dst_id, bytes, Label)\n"
                        "Only CICIDS datasets or datasets in that format are supported."
                    )
                
                TRAINING_STATE["progress"] = "Loading clean data..."
                
                # Load and adapt events
                df = load_and_adapt_events(train_csv, fmt="clean")
                df = filter_entities(df, min_events_per_entity=10)
                if len(df) == 0:
                    raise RuntimeError("No data after filtering entities.")
                
                TRAINING_STATE["progress"] = "Splitting data..."
                
                # Split train/val (70/15/15)
                df_tr, df_va, df_te = split_time_within_groups(df, 0.7, 0.15)
                
                # Fit tokenization on train only
                dst_keep = fit_dst_vocab(df_tr, top_n=200)
                bytes_edges = fit_bytes_bins(df_tr, num_buckets=8)
                
                # Token strings
                for d in (df_tr, df_va, df_te):
                    d["token_str"] = make_token_strings(d, dst_keep, bytes_edges)
                
                df_tr = df_tr.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
                df_va = df_va.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
                
                # Build vocab
                vocab, tok_map, _ = build_vocab_from_buckets(dst_keep, bytes_edges)
                vocab_size = len(vocab) + 2  # PAD=0, UNK=1
                seq_len = 10
                
                TRAINING_STATE["progress"] = "Building sequences..."
                
                # Build sequences
                Xtr, ytr, *_ = make_sequences_from_events(df_tr, tok_map, seq_len, step=1)
                Xva, yva, *_ = make_sequences_from_events(df_va, tok_map, seq_len, step=1)
                
                if len(Xtr) == 0 or len(Xva) == 0:
                    raise RuntimeError("Not enough sequences for training. Need more data.")
                
                TRAINING_STATE["progress"] = "Building model..."
                
                # Build and train model using improved architecture
                model = build_improved_transformer_model(
                    vocab_size=vocab_size,
                    seq_len=seq_len,
                    d_model=128,
                    num_layers=4,
                    num_heads=8,
                    lr=learning_rate,
                )
                
                # Prepare datasets
                def to_onehot(x, y):
                    return x, tf.one_hot(y, depth=vocab_size)
                
                ds_tr = (tf.data.Dataset.from_tensor_slices((Xtr, ytr))
                         .shuffle(min(50000, len(Xtr)))
                         .batch(batch_size)
                         .map(to_onehot, num_parallel_calls=tf.data.AUTOTUNE)
                         .prefetch(tf.data.AUTOTUNE))
                
                ds_va = (tf.data.Dataset.from_tensor_slices((Xva, yva))
                         .batch(batch_size)
                         .map(to_onehot, num_parallel_calls=tf.data.AUTOTUNE)
                         .prefetch(tf.data.AUTOTUNE))
                
                # Custom callback to update progress
                class ProgressCallback(keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        acc = logs.get("val_acc", 0) * 100
                        TRAINING_STATE["progress"] = f"Epoch {epoch + 1}/{epochs} — val_acc: {acc:.1f}%"
                
                TRAINING_STATE["progress"] = f"Training epoch 1/{epochs}..."
                
                model_path = os.path.join(os.path.dirname(__file__), "models", "log_transformer.keras")
                callbacks = [
                    keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
                    ProgressCallback(),
                ]
                
                history = model.fit(ds_tr, validation_data=ds_va, epochs=epochs, callbacks=callbacks, verbose=1)
                
                # Get final validation metrics
                val_acc = history.history.get("val_acc", [0])[-1] * 100
                val_top5 = history.history.get("val_top5_acc", [0])[-1] * 100
                
                # Reload best model
                MODEL = keras.models.load_model(model_path)
                
                # Update training state with real accuracy
                TRAINING_STATE["accuracy"] = round(val_acc, 1)
                TRAINING_STATE["top5_accuracy"] = round(val_top5, 1)
                TRAINING_STATE["last_trained"] = datetime.now().isoformat()
                TRAINING_STATE["progress"] = f"Complete — accuracy: {val_acc:.1f}%"
                TRAINING_STATE["is_training"] = False
                
                print(f"Training complete. Val accuracy: {val_acc:.1f}%, Top-5: {val_top5:.1f}%")
                
            except Exception as e:
                TRAINING_STATE["is_training"] = False
                TRAINING_STATE["error"] = str(e)
                TRAINING_STATE["progress"] = f"Error: {str(e)}"
                print(f"Training error: {e}", file=sys.stderr)
        
        thread = threading.Thread(target=run_training, daemon=True)
        thread.start()
        
        return jsonify({
            "success": True,
            "message": "Model training started",
            "config": {"epochs": epochs, "batchSize": batch_size, "learningRate": learning_rate},
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/explain', methods=['POST'])
def explain_anomaly():
    """Explain why a sequence is considered anomalous"""
    try:
        data = request.json
        entity_id = data.get("entity_id")
        
        if not entity_id:
            return jsonify({"error": "entity_id is required"}), 400
            
        buffer = get_entity_buffer(entity_id)
        seq_len = MODEL_META.get("seq_len", 64)
        
        if len(buffer) < seq_len:
            return jsonify({"error": "Insufficient data for explanation"}), 400
            
        # Get last sequence
        recent = buffer[-seq_len:]
        X = np.array([e["token_id"] for e in recent], dtype=np.int32).reshape(1, -1)
        
        # Get attention weights or important events
        importance = get_important_events(MODEL, X, seq_idx=0)
        
        # Prepare tokens for frontend
        tokens = []
        for i, (e, imp) in enumerate(zip(recent, importance)):
            tokens.append({
                "index": i,
                "token_id": int(e["token_id"]),
                "token_str": VOCAB[e["token_id"] - 2] if 2 <= e["token_id"] < len(VOCAB) + 2 else "UNK",
                "importance": float(imp),
                "timestamp": e["timestamp"]
            })
            
        return jsonify({
            "entity_id": entity_id,
            "tokens": tokens,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/train/status', methods=['GET'])
def train_status():
    """Get current training status"""
    return jsonify({
        "is_training": TRAINING_STATE["is_training"],
        "progress": TRAINING_STATE["progress"],
        "accuracy": TRAINING_STATE["accuracy"],
        "top5_accuracy": TRAINING_STATE["top5_accuracy"],
        "last_trained": TRAINING_STATE["last_trained"],
        "error": TRAINING_STATE["error"],
    })


@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    """Upload a CSV file for training. Accepts CICIDS or clean format.
    CICIDS files are auto-converted via cicids_into_clean.R."""
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400
    
    try:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Save the uploaded file
        upload_path = os.path.join(data_dir, "uploaded_raw.csv")
        file.save(upload_path)
        
        # Detect format
        fmt = detect_csv_format(upload_path)
        
        if fmt == "cicids":
            # Run R adapter to convert to clean format
            clean_path = os.path.join(data_dir, "train_data.csv")
            run_r_cicids_adapter(upload_path, clean_path)
            row_count = sum(1 for _ in open(clean_path)) - 1
            
            return jsonify({
                "success": True,
                "message": f"CICIDS dataset preprocessed with cicids_into_clean.R and saved ({row_count} rows). Ready to train!",
                "format": "cicids",
                "rows": row_count,
            })
        elif fmt == "clean":
            import shutil
            train_path = os.path.join(data_dir, "train_data.csv")
            shutil.copy2(upload_path, train_path)
            row_count = sum(1 for _ in open(train_path)) - 1
            
            return jsonify({
                "success": True,
                "message": f"Clean-format CSV saved ({row_count} rows). Ready to train!",
                "format": "clean",
                "rows": row_count,
            })
        else:
            return jsonify({
                "success": False,
                "error": "Unrecognized CSV format. Only CICIDS datasets (Timestamp, Source IP, Destination IP, Destination Port, Protocol) or clean format (timestamp, entity_id, event_type) are supported."
            }), 400
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/model/info', methods=['GET'])
def get_model_info():
    """Get model information"""
    if MODEL_META is None:
        return jsonify({"error": "Model not loaded"}), 404
    
    return jsonify({
        "model_loaded": MODEL is not None,
        "vocab_size": MODEL_META.get("vocab_size"),
        "seq_len": MODEL_META.get("seq_len"),
        "split_mode": MODEL_META.get("split_mode"),
        "train_csv": MODEL_META.get("train_csv"),
        "metrics": {
            "train_frac": MODEL_META.get("train_frac"),
            "val_frac": MODEL_META.get("val_frac"),
        }
    })


@app.route('/api/entity/<entity_id>', methods=['GET'])
def get_entity_info(entity_id: str):
    """Get information about specific entity"""
    buffer = get_entity_buffer(entity_id)
    
    if not buffer:
        return jsonify({"error": "Entity not found"}), 404
    
    # Get entity stats
    entity_stat = None
    if ENTITY_STATS is not None:
        entity_stat_df = ENTITY_STATS[ENTITY_STATS["entity_id"] == entity_id]
        if len(entity_stat_df) > 0:
            entity_stat = {
                "mean_nll": float(entity_stat_df["mean_nll"].iloc[0]),
                "std_nll": float(entity_stat_df["std_nll"].iloc[0])
            }
    
    return jsonify({
        "entity_id": entity_id,
        "total_events": len(buffer),
        "first_seen": buffer[0]["timestamp"] if buffer else None,
        "last_seen": buffer[-1]["timestamp"] if buffer else None,
        "statistics": entity_stat
    })




# ============================================
# SERVE DASHBOARD
# ============================================

@app.route('/assets/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(base_dir, filename)
    if not os.path.isfile(filepath):
        return 'File not found: ' + filepath, 404
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    mime = 'application/javascript' if filename.endswith('.js') else 'text/plain'
    return Response(content, mimetype=mime)

@app.route('/')
def serve_dashboard():
    """Serve the React dashboard"""
    return send_file('frontend.html')


# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    print("=" * 60)
    print("Starting Anomaly Detection API Server")
    print("=" * 60)
    
    # Load model
    load_model_and_config()
    
    print("\n+ Server ready!")
    print(f"+ Dashboard: http://localhost:5000")
    print(f"+ API: http://localhost:5000/api/health")
    print("\n" + "=" * 60)
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
