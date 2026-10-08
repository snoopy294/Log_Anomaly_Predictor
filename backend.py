#!/usr/bin/env python3
"""
backend.py — Flask REST API
----------------------------
Connects frontend.html (the dashboard) to the new.py pipeline and model.py.
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
    get_important_events
)
from detector_bundle import DetectorRuntime, load_detector_bundle

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global model and configuration
MODEL = None
MODEL_META = None
VOCAB = None
TOKENIZER = None
ENTITY_STATS = None
DETECTOR_RUNTIME = None
EVENT_BUFFER = {}  # Store recent events per entity
ALERT_QUEUE = queue.Queue()

# Training state — tracks real accuracy from training
TRAINING_STATE = {
    "is_training": False,
    "progress": "",
    "roc_auc": None,
    "pr_auc": None,
    "accuracy": 0.0,      # keep for backward compat but don't show as headline
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
    "out_dir": "outputs",
    "max_buffer_size": 1000,
    "alert_threshold": 3.0,
    "bundle_path": os.environ.get("DETECTOR_BUNDLE_PATH", "models/detector_bundle"),
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
    """Load the versioned detector bundle. Detectors are trained and calibrated
    exclusively by new.py; there is no legacy raw-.keras loading path."""
    global MODEL, MODEL_META, ENTITY_STATS, VOCAB, TOKENIZER, DETECTOR_RUNTIME

    bundle_path = CONFIG["bundle_path"]
    if not os.path.exists(os.path.join(bundle_path, "bundle.json")):
        print(f"! No detector bundle found at {bundle_path}. Train one with new.py first.")
        return

    try:
        MODEL_META, MODEL = load_detector_bundle(bundle_path)
        DETECTOR_RUNTIME = DetectorRuntime(MODEL_META, MODEL)
        VOCAB = MODEL_META["vocabulary"]
        TOKENIZER = DETECTOR_RUNTIME.token_map.copy()
        TOKENIZER.update({"PAD": 0, "UNK": 1})
        ENTITY_STATS = pd.DataFrame(MODEL_META["entity_nll_stats"])
        CONFIG["alert_threshold"] = float(MODEL_META["threshold"])
        print(f"+ Detector bundle {MODEL_META['model_version']} loaded from {bundle_path}")
    except Exception as e:
        raise RuntimeError(f"detector bundle failed validation: {e}") from e


# ============================================
# UTILITY FUNCTIONS
# ============================================

def tokenize_event(event: Dict) -> int:
    """Convert event to token ID"""
    if DETECTOR_RUNTIME is not None:
        return DETECTOR_RUNTIME.token_id(event)
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
    """Compute anomaly score for actual next event via the shared DetectorRuntime.
    Returns None if no detector bundle is loaded — there is no legacy fallback."""
    if DETECTOR_RUNTIME is None:
        return None
    event = dict(actual_event)
    event["entity_id"] = entity_id
    return DETECTOR_RUNTIME.score_event(event, update=True)


# ============================================
# API ENDPOINTS
# ============================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "model_version": MODEL_META.get("model_version") if MODEL_META else None,
        "warm_up_state": ("warm" if DETECTOR_RUNTIME and DETECTOR_RUNTIME.warmed_up else "cold"),
        "batch_one_latency_ms": DETECTOR_RUNTIME.last_latency_ms if DETECTOR_RUNTIME else None,
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
    
    # Read detection ROC-AUC from metrics summary (preferred) or fall back
    model_accuracy = None
    metrics_path = os.path.join(CONFIG.get("out_dir", "outputs"), "metrics_summary.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as _f:
                _ms = json.load(_f)
            _det = _ms.get("detection", {})
            model_accuracy = _det.get("roc_auc")
        except Exception:
            pass
    if model_accuracy is None:
        model_accuracy = TRAINING_STATE.get("roc_auc")
    
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
        "modelVersion": MODEL_META.get("model_version") if MODEL_META else None,
        "warmUpState": "warm" if DETECTOR_RUNTIME and DETECTOR_RUNTIME.warmed_up else "cold",
        "batchOneLatencyMs": DETECTOR_RUNTIME.last_latency_ms if DETECTOR_RUNTIME else None,
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
        
        # The shared runtime scores against prior context, then appends the event.
        score = compute_anomaly_score(entity_id, event)
        add_event_to_buffer(entity_id, event)
        
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


@app.route('/api/train', methods=['POST'])
def train_model():
    """Versioned detectors are trained and calibrated exclusively by new.py,
    then loaded here as a detector bundle. This API never trains a model."""
    return jsonify({
        "success": False,
        "message": "Versioned detectors must be trained and calibrated by new.py, then loaded as one bundle."
    }), 409


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
