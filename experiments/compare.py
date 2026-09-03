#!/usr/bin/env python3
"""
model_comparison.py
------------------
Compare original vs improved transformer models
"""

import os
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns

# Import both model builders
from logs_transformer_anomaly_crossdataset import build_transformer_next_event_model as build_original
from logs_transformer_improved import build_improved_transformer_model as build_improved


def generate_dummy_data(n_samples=10000, vocab_size=1000, seq_len=64):
    """Generate dummy data for testing"""
    X = np.random.randint(0, vocab_size, size=(n_samples, seq_len), dtype=np.int32)
    y = np.random.randint(0, vocab_size, size=(n_samples,), dtype=np.int32)
    return X, y


def evaluate_model(model, X, y, name="Model"):
    """Evaluate model performance"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print('='*60)
    
    # Convert y to one-hot for evaluation
    y_oh = tf.one_hot(y, depth=model.output_shape[-1])
    
    # Timing
    start = time.time()
    loss, acc, top5 = model.evaluate(X, y_oh, verbose=0)
    eval_time = time.time() - start
    
    # Inference speed
    start = time.time()
    _ = model.predict(X[:1000], verbose=0)
    inference_time = (time.time() - start) / 1000  # per sample
    
    # Memory usage (approximate)
    params = model.count_params()
    memory_mb = (params * 4) / (1024 * 1024)  # 4 bytes per param
    
    results = {
        'name': name,
        'loss': loss,
        'accuracy': acc * 100,
        'top5_accuracy': top5 * 100,
        'eval_time': eval_time,
        'inference_time_ms': inference_time * 1000,
        'parameters': params,
        'memory_mb': memory_mb
    }
    
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Top-5 Accuracy: {top5*100:.2f}%")
    print(f"Evaluation Time: {eval_time:.2f}s")
    print(f"Inference Time: {inference_time*1000:.3f}ms per sample")
    print(f"Parameters: {params:,}")
    print(f"Memory Usage: {memory_mb:.2f} MB")
    
    return results


def compare_architectures(vocab_size=1000, seq_len=64, n_samples=10000):
    """Compare original and improved architectures"""
    print("="*60)
    print("🔬 MODEL COMPARISON")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Vocab Size: {vocab_size}")
    print(f"  Sequence Length: {seq_len}")
    print(f"  Test Samples: {n_samples}")
    
    # Generate test data
    print("\n📊 Generating test data...")
    X_test, y_test = generate_dummy_data(n_samples, vocab_size, seq_len)
    print(f"  X shape: {X_test.shape}")
    print(f"  y shape: {y_test.shape}")
    
    # Build models
    print("\n🏗️  Building models...")
    
    print("\n  Building Original Model...")
    model_original = build_original(
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=128,
        dropout=0.3
    )
    
    print("  Building Improved Model...")
    model_improved = build_improved(
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=128,
        num_layers=4,
        num_heads=8,
        d_ff=512,
        dropout=0.2
    )
    
    # Evaluate both models
    results = []
    
    results.append(evaluate_model(model_original, X_test, y_test, "Original Model"))
    results.append(evaluate_model(model_improved, X_test, y_test, "Improved Model"))
    
    # Create comparison DataFrame
    df_results = pd.DataFrame(results)
    
    # Calculate improvements
    print("\n" + "="*60)
    print("📈 IMPROVEMENTS")
    print("="*60)
    
    orig = df_results[df_results['name'] == 'Original Model'].iloc[0]
    impr = df_results[df_results['name'] == 'Improved Model'].iloc[0]
    
    metrics = {
        'Accuracy': (impr['accuracy'] - orig['accuracy'], '%'),
        'Top-5 Accuracy': (impr['top5_accuracy'] - orig['top5_accuracy'], '%'),
        'Inference Time': (impr['inference_time_ms'] - orig['inference_time_ms'], 'ms'),
        'Memory Usage': (impr['memory_mb'] - orig['memory_mb'], 'MB'),
        'Parameters': (impr['parameters'] - orig['parameters'], ''),
    }
    
    for metric, (diff, unit) in metrics.items():
        sign = "+" if diff > 0 else ""
        print(f"{metric:20s}: {sign}{diff:.2f} {unit}")
    
    # Save results
    output_dir = "outputs/comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    df_results.to_csv(f"{output_dir}/model_comparison.csv", index=False)
    print(f"\n💾 Results saved to: {output_dir}/model_comparison.csv")
    
    # Create visualizations
    create_comparison_plots(df_results, output_dir)
    
    return df_results


def create_comparison_plots(df, output_dir):
    """Create comparison visualizations"""
    print("\n📊 Creating comparison plots...")
    
    # Set style
    plt.style.use('dark_background')
    sns.set_palette("husl")
    
    # 1. Accuracy Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Accuracy
    axes[0].bar(df['name'], df['accuracy'], color=['#00f5ff', '#ff006e'])
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Model Accuracy')
    axes[0].set_ylim([0, 100])
    
    # Top-5 Accuracy
    axes[1].bar(df['name'], df['top5_accuracy'], color=['#00f5ff', '#ff006e'])
    axes[1].set_ylabel('Top-5 Accuracy (%)')
    axes[1].set_title('Top-5 Accuracy')
    axes[1].set_ylim([0, 100])
    
    # Inference Time
    axes[2].bar(df['name'], df['inference_time_ms'], color=['#00f5ff', '#ff006e'])
    axes[2].set_ylabel('Inference Time (ms)')
    axes[2].set_title('Inference Speed')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_comparison.png", dpi=150, facecolor='#0a0e27')
    plt.close()
    
    # 2. Resource Usage
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Parameters
    axes[0].bar(df['name'], df['parameters']/1e6, color=['#00f5ff', '#ff006e'])
    axes[0].set_ylabel('Parameters (Millions)')
    axes[0].set_title('Model Size')
    
    # Memory
    axes[1].bar(df['name'], df['memory_mb'], color=['#00f5ff', '#ff006e'])
    axes[1].set_ylabel('Memory (MB)')
    axes[1].set_title('Memory Usage')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/resource_comparison.png", dpi=150, facecolor='#0a0e27')
    plt.close()
    
    # 3. Radar Chart
    categories = ['Accuracy', 'Top-5 Acc', 'Speed', 'Efficiency']
    
    # Normalize values for radar chart
    orig = df[df['name'] == 'Original Model'].iloc[0]
    impr = df[df['name'] == 'Improved Model'].iloc[0]
    
    # Higher is better for accuracy, lower is better for time/memory
    orig_values = [
        orig['accuracy'] / 100,
        orig['top5_accuracy'] / 100,
        1 - (orig['inference_time_ms'] / (orig['inference_time_ms'] + impr['inference_time_ms'])),
        1 - (orig['memory_mb'] / (orig['memory_mb'] + impr['memory_mb']))
    ]
    
    impr_values = [
        impr['accuracy'] / 100,
        impr['top5_accuracy'] / 100,
        1 - (impr['inference_time_ms'] / (orig['inference_time_ms'] + impr['inference_time_ms'])),
        1 - (impr['memory_mb'] / (orig['memory_mb'] + impr['memory_mb']))
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    orig_values += orig_values[:1]
    impr_values += impr_values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, orig_values, 'o-', linewidth=2, label='Original', color='#00f5ff')
    ax.fill(angles, orig_values, alpha=0.25, color='#00f5ff')
    ax.plot(angles, impr_values, 'o-', linewidth=2, label='Improved', color='#ff006e')
    ax.fill(angles, impr_values, alpha=0.25, color='#ff006e')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title('Model Performance Comparison', size=16, pad=20)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/radar_comparison.png", dpi=150, facecolor='#0a0e27')
    plt.close()
    
    print(f"  ✓ Plots saved to: {output_dir}/")


def main():
    """Main comparison function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare model architectures")
    parser.add_argument("--vocab_size", type=int, default=1000)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--n_samples", type=int, default=10000)
    
    args = parser.parse_args()
    
    # Run comparison
    results = compare_architectures(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        n_samples=args.n_samples
    )
    
    print("\n" + "="*60)
    print("✨ Comparison Complete!")
    print("="*60)
    print("\nCheck the outputs/comparison/ directory for:")
    print("  - model_comparison.csv")
    print("  - accuracy_comparison.png")
    print("  - resource_comparison.png")
    print("  - radar_comparison.png")
    print("")


if __name__ == "__main__":
    main()
