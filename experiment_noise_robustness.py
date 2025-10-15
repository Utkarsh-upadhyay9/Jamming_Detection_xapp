#!/usr/bin/env python3
"""
Noise Robustness Analysis: Fixed Jamming Power, Varying Noise Levels
Analyzes whether detection algorithm performance degrades with increased noise.
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set MATLAB-style plotting
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '-'

# MATLAB default colors
matlab_colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']

np.random.seed(42)

class SimpleEnsemble:
    """Simple ensemble of CatBoost + Isolation Forest"""
    def __init__(self, catboost_weight=0.75):
        self.cb_weight = catboost_weight
        self.if_weight = 1 - catboost_weight
        self.catboost = None
        self.iforest = None
        
    def fit(self, X, y):
        # Train CatBoost
        self.catboost = CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            random_seed=42,
            verbose=False
        )
        self.catboost.fit(X, y)
        
        # Train Isolation Forest (unsupervised)
        self.iforest = IsolationForest(
            contamination=0.25,
            random_state=42,
            n_estimators=100
        )
        self.iforest.fit(X)
        
    def predict(self, X):
        cb_pred = self.catboost.predict(X).ravel().astype(int)
        if_pred = self.iforest.predict(X)
        if_pred = np.where(if_pred == 1, 0, 1)  # Convert: 1 (normal) -> 0, -1 (anomaly) -> 1
        
        # Weighted voting (simplified)
        ensemble_pred = np.where(
            cb_pred == 0,
            0,  # If CB says normal, trust it more
            np.where(if_pred == 1, cb_pred, 0)  # Otherwise use IF as gate
        )
        return ensemble_pred


def generate_features_with_noise(n_samples, class_label, noise_power_dbm, jamming_power_fixed=-30):
    """
    Generate synthetic features with fixed jamming power but varying noise.
    
    Args:
        n_samples: Number of samples
        class_label: 0=normal, 1=constant, 2=random, 3=reactive
        noise_power_dbm: Noise floor in dBm (-110=ideal, -100=high noise)
        jamming_power_fixed: Fixed jamming power in dBm (default -30)
    """
    # Convert noise power from dBm to linear scale (normalized)
    noise_linear = 10 ** (noise_power_dbm / 10) / 1e12  # Normalize to reasonable range
    
    # Base signal quality (pre-jamming)
    base_rsrp = np.random.normal(-70, 10, n_samples)
    base_sinr = np.random.normal(15, 5, n_samples)
    
    # Jamming impact (class-dependent, fixed power)
    if class_label == 0:  # Normal - no jamming
        jamming_factor = 0.0
        throughput_mean = 70
        packet_rate_mean = 7000
    elif class_label == 1:  # Constant jamming
        jamming_factor = 0.7  # 70% degradation
        throughput_mean = 20
        packet_rate_mean = 2000
    elif class_label == 2:  # Random jamming
        jamming_factor = np.random.uniform(0.3, 0.9, n_samples)
        throughput_mean = 35
        packet_rate_mean = 3500
    else:  # Reactive jamming
        jamming_factor = np.random.uniform(0.5, 0.95, n_samples)
        throughput_mean = 25
        packet_rate_mean = 2500
    
    # Add noise to all measurements
    rsrp = base_rsrp - jamming_factor * 20 + np.random.normal(0, noise_linear * 100, n_samples)
    sinr = base_sinr - jamming_factor * 25 + np.random.normal(0, noise_linear * 150, n_samples)
    
    # Throughput degradation (noise causes additional fluctuations)
    throughput = np.maximum(0, np.random.normal(
        throughput_mean * (1 - jamming_factor * 0.5),
        15 + noise_linear * 200,
        n_samples
    ))
    
    # Packet rate (correlated with throughput)
    packet_rate = np.maximum(0, np.random.normal(
        packet_rate_mean * (1 - jamming_factor * 0.6),
        1000 + noise_linear * 1500,
        n_samples
    ))
    
    # Buffer occupancy increases with congestion and noise
    buffer_occupancy = np.clip(
        jamming_factor * 0.7 + np.random.normal(0, 0.1 + noise_linear * 2, n_samples),
        0, 1
    )
    
    # Spectral features (noise affects entropy/flatness)
    spectral_entropy = np.random.uniform(
        0.3 + jamming_factor * 0.4 - noise_linear * 0.5,
        0.8 + jamming_factor * 0.2 - noise_linear * 0.3,
        n_samples
    )
    spectral_flatness = np.random.uniform(
        0.2 - jamming_factor * 0.15 + noise_linear * 0.3,
        0.6 - jamming_factor * 0.3 + noise_linear * 0.2,
        n_samples
    )
    
    # BLER increases with both jamming and noise
    bler = np.clip(
        jamming_factor * 0.6 + noise_linear * 50 + np.random.normal(0, 0.1, n_samples),
        0, 1
    )
    
    # Latency increases with congestion and noise
    latency = np.maximum(10, np.random.normal(
        50 + jamming_factor * 200 + noise_linear * 3000,
        50,
        n_samples
    ))
    
    features = pd.DataFrame({
        'rsrp': rsrp,
        'sinr': sinr,
        'throughput': throughput,
        'packet_rate': packet_rate,
        'buffer_occupancy': buffer_occupancy,
        'spectral_entropy': spectral_entropy,
        'spectral_flatness': spectral_flatness,
        'bler': bler,
        'latency': latency,
        'label': class_label
    })
    
    return features


def run_noise_experiment(noise_levels_dbm):
    """Run experiment across multiple noise levels"""
    results = {}
    
    for noise_dbm in noise_levels_dbm:
        print(f"\n{'='*70}")
        print(f"Testing Noise Level: {noise_dbm} dBm")
        print(f"{'='*70}")
        
        # Generate training data (fixed noise at -110 dBm - ideal conditions)
        train_data = []
        for class_label in range(4):
            df = generate_features_with_noise(350, class_label, noise_power_dbm=-110)
            train_data.append(df)
        train_df = pd.concat(train_data, ignore_index=True)
        
        X_train = train_df.drop('label', axis=1).values
        y_train = train_df['label'].values
        
        # Generate test data with varying noise
        test_data = []
        for class_label in range(4):
            df = generate_features_with_noise(150, class_label, noise_power_dbm=noise_dbm)
            test_data.append(df)
        test_df = pd.concat(test_data, ignore_index=True)
        
        X_test = test_df.drop('label', axis=1).values
        y_test = test_df['label'].values
        
        # Train and evaluate ensemble
        ensemble = SimpleEnsemble(catboost_weight=0.75)
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)
        
        # Compute metrics
        f1 = f1_score(y_test, y_pred, average='weighted')
        accuracy = accuracy_score(y_test, y_pred)
        
        # Per-class F1 scores
        f1_per_class = f1_score(y_test, y_pred, average=None)
        
        # Classification report
        class_names = ['Normal', 'Constant', 'Random', 'Reactive']
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        results[noise_dbm] = {
            'f1_weighted': f1,
            'accuracy': accuracy,
            'f1_per_class': f1_per_class.tolist(),
            'classification_report': report,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        print(f"Weighted F1: {f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Per-class F1: Normal={f1_per_class[0]:.4f}, Constant={f1_per_class[1]:.4f}, "
              f"Random={f1_per_class[2]:.4f}, Reactive={f1_per_class[3]:.4f}")
    
    return results


def generate_visualizations(results, noise_levels):
    """Generate publication-quality figures"""
    
    # Extract metrics
    f1_scores = [results[n]['f1_weighted'] for n in noise_levels]
    accuracies = [results[n]['accuracy'] for n in noise_levels]
    
    f1_per_class = np.array([results[n]['f1_per_class'] for n in noise_levels])
    
    # Create output directory
    import os
    os.makedirs('figs/noise_robustness/', exist_ok=True)
    
    # Figure 1: Overall Performance vs Noise
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Panel A: F1-Score vs Noise
    ax = axes[0]
    ax.plot(noise_levels, f1_scores, 'o-', color=matlab_colors[0], linewidth=2, markersize=8, label='Weighted F1')
    ax.axhline(y=0.90, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='90% Threshold')
    ax.set_xlabel('Noise Floor (dBm)', fontsize=11, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
    ax.set_title('(a) Overall F1-Score vs Noise Level', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_ylim([0.5, 1.0])
    
    # Panel B: Accuracy vs Noise
    ax = axes[1]
    ax.plot(noise_levels, accuracies, 's-', color=matlab_colors[1], linewidth=2, markersize=8, label='Accuracy')
    ax.axhline(y=0.90, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='90% Threshold')
    ax.set_xlabel('Noise Floor (dBm)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('(b) Classification Accuracy vs Noise Level', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_ylim([0.5, 1.0])
    
    plt.tight_layout()
    plt.savefig('figs/noise_robustness/noise_overall_performance.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: noise_overall_performance.png")
    plt.close()
    
    # Figure 2: Per-Class Performance
    fig, ax = plt.subplots(figsize=(10, 5))
    
    class_names = ['Normal', 'Constant', 'Random', 'Reactive']
    for i, class_name in enumerate(class_names):
        ax.plot(noise_levels, f1_per_class[:, i], 'o-', color=matlab_colors[i], 
                linewidth=2, markersize=8, label=class_name)
    
    ax.axhline(y=0.90, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='90% Target')
    ax.set_xlabel('Noise Floor (dBm)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Per-Class F1-Score', fontsize=11, fontweight='bold')
    ax.set_title('Per-Class Detection Performance vs Noise Level', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='lower left')
    ax.set_ylim([0.4, 1.0])
    
    plt.tight_layout()
    plt.savefig('figs/noise_robustness/noise_per_class_performance.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: noise_per_class_performance.png")
    plt.close()


if __name__ == '__main__':
    print("="*70)
    print("NOISE ROBUSTNESS ANALYSIS")
    print("Fixed Jamming Power | Varying Noise Levels")
    print("="*70)
    
    # Define noise levels (dBm) - from ideal to high noise
    noise_levels = [-110, -108, -105, -103, -100, -98, -95]
    
    # Run experiment
    results = run_noise_experiment(noise_levels)
    
    # Generate visualizations
    generate_visualizations(results, noise_levels)
    
    # Save results
    output_file = 'results/noise_robustness_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'noise_levels_dbm': noise_levels,
            'results': results,
            'summary': {
                'best_f1': max([results[n]['f1_weighted'] for n in noise_levels]),
                'worst_f1': min([results[n]['f1_weighted'] for n in noise_levels]),
                'f1_degradation': max([results[n]['f1_weighted'] for n in noise_levels]) - 
                                 min([results[n]['f1_weighted'] for n in noise_levels])
            }
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    f1_values = [results[n]['f1_weighted'] for n in noise_levels]
    print(f"Best F1 (@ {noise_levels[f1_values.index(max(f1_values))]} dBm): {max(f1_values):.4f}")
    print(f"Worst F1 (@ {noise_levels[f1_values.index(min(f1_values))]} dBm): {min(f1_values):.4f}")
    print(f"F1 Degradation: {max(f1_values) - min(f1_values):.4f} ({(max(f1_values) - min(f1_values))*100:.2f}%)")
    
    if max(f1_values) - min(f1_values) < 0.05:
        print("\n✅ ROBUST: Noise has minimal impact (<5% degradation)")
    else:
        print("\n⚠️  SENSITIVE: Noise causes significant degradation (≥5%)")
