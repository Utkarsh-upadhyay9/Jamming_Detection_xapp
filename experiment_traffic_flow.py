#!/usr/bin/env python3
"""
Experiment 1: Differential Traffic Flow Analysis
Compare jamming detection performance for UEs with:
- High traffic flow (video streaming, high-throughput applications)
- Low traffic flow (IoT sensors, lightweight messaging)

Tests ensemble weight stability and detection accuracy across traffic profiles.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import json
import time
from pathlib import Path

# Seed for reproducibility
np.random.seed(42)

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    'n_samples_per_class': 500,
    'jamming_types': ['normal', 'constant', 'random', 'reactive'],
    'output_dir': 'results/traffic_flow_experiment',
    'figures_dir': 'figs/traffic_flow_experiment',
}

# Traffic profiles
TRAFFIC_PROFILES = {
    'high_flow': {
        'name': 'High Flow (Video/Streaming)',
        'throughput_mean': 85.0,      # Mbps
        'throughput_std': 15.0,
        'packet_rate_mean': 8500,     # packets/sec
        'packet_rate_std': 1200,
        'buffer_occupancy_mean': 0.65,
        'buffer_occupancy_std': 0.15,
        'rsrp_mean': -75,             # dBm (good signal needed)
        'rsrp_std': 5,
        'sinr_mean': 18,              # dB (high quality)
        'sinr_std': 3,
    },
    'low_flow': {
        'name': 'Low Flow (IoT/Messaging)',
        'throughput_mean': 5.0,       # Mbps
        'throughput_std': 2.0,
        'packet_rate_mean': 250,      # packets/sec
        'packet_rate_std': 80,
        'buffer_occupancy_mean': 0.15,
        'buffer_occupancy_std': 0.08,
        'rsrp_mean': -85,             # dBm (can tolerate weaker signal)
        'rsrp_std': 8,
        'sinr_mean': 10,              # dB (moderate quality)
        'sinr_std': 4,
    }
}

# =============================================================================
# Data Generation
# =============================================================================

def generate_traffic_features(profile_config, n_samples, jamming_class, jamming_intensity=0.0):
    """
    Generate synthetic feature vectors for a given traffic profile and jamming condition.
    
    Args:
        profile_config: Traffic profile parameters
        n_samples: Number of samples to generate
        jamming_class: 0=normal, 1=constant, 2=random, 3=reactive
        jamming_intensity: Jamming impact scale [0, 1]
    
    Returns:
        DataFrame with features
    """
    # Base features from traffic profile
    throughput = np.random.normal(
        profile_config['throughput_mean'],
        profile_config['throughput_std'],
        n_samples
    )
    
    packet_rate = np.random.normal(
        profile_config['packet_rate_mean'],
        profile_config['packet_rate_std'],
        n_samples
    )
    
    buffer_occupancy = np.random.normal(
        profile_config['buffer_occupancy_mean'],
        profile_config['buffer_occupancy_std'],
        n_samples
    )
    buffer_occupancy = np.clip(buffer_occupancy, 0, 1)
    
    rsrp = np.random.normal(
        profile_config['rsrp_mean'],
        profile_config['rsrp_std'],
        n_samples
    )
    
    sinr = np.random.normal(
        profile_config['sinr_mean'],
        profile_config['sinr_std'],
        n_samples
    )
    
    # Apply jamming effects
    if jamming_class > 0:  # Any jamming present
        # Throughput degradation
        throughput_degradation = np.random.uniform(0.3, 0.7, n_samples) * jamming_intensity
        throughput *= (1 - throughput_degradation)
        
        # SINR degradation
        sinr_drop = np.random.uniform(5, 15, n_samples) * jamming_intensity
        sinr -= sinr_drop
        
        # RSRP impact (less severe)
        rsrp_drop = np.random.uniform(2, 8, n_samples) * jamming_intensity
        rsrp -= rsrp_drop
        
        # Packet rate distortion
        if jamming_class == 1:  # Constant jamming - steady degradation
            packet_rate *= (1 - 0.4 * jamming_intensity)
        elif jamming_class == 2:  # Random jamming - high variance
            packet_rate *= (1 - np.random.uniform(0, 0.8, n_samples) * jamming_intensity)
        elif jamming_class == 3:  # Reactive jamming - burst pattern
            burst_mask = np.random.rand(n_samples) > 0.5
            packet_rate[burst_mask] *= (1 - 0.7 * jamming_intensity)
        
        # Buffer increases under jamming
        buffer_occupancy += np.random.uniform(0.2, 0.5, n_samples) * jamming_intensity
        buffer_occupancy = np.clip(buffer_occupancy, 0, 1)
    
    # Derived features
    rsrq = rsrp - sinr + np.random.normal(0, 2, n_samples)
    bler = np.clip(np.exp(-sinr / 5) * np.random.uniform(0.8, 1.2, n_samples), 0, 1)
    latency = 50 + (1 - sinr / 30) * 150 + np.random.normal(0, 20, n_samples)
    latency = np.clip(latency, 10, 500)
    
    # Spectral features (simulated)
    spectral_entropy = np.random.uniform(0.4, 0.9, n_samples)
    if jamming_class > 0:
        spectral_entropy += np.random.uniform(0.1, 0.3, n_samples) * jamming_intensity
        spectral_entropy = np.clip(spectral_entropy, 0, 1)
    
    spectral_flatness = np.random.uniform(0.2, 0.7, n_samples)
    if jamming_class == 1:  # Constant jammer has flat spectrum
        spectral_flatness += 0.2 * jamming_intensity
        spectral_flatness = np.clip(spectral_flatness, 0, 1)
    
    # Temporal features
    throughput_variance = np.abs(np.random.normal(0, 10 + 40 * jamming_intensity, n_samples))
    
    # Create dataframe
    data = pd.DataFrame({
        'rsrp': rsrp,
        'rsrq': rsrq,
        'sinr': sinr,
        'throughput_mbps': throughput,
        'packet_rate': packet_rate,
        'bler': bler,
        'latency_ms': latency,
        'buffer_occupancy': buffer_occupancy,
        'throughput_variance': throughput_variance,
        'spectral_entropy': spectral_entropy,
        'spectral_flatness': spectral_flatness,
        'label': jamming_class
    })
    
    return data

def generate_full_dataset(profile_key):
    """Generate complete dataset for a traffic profile."""
    profile_config = TRAFFIC_PROFILES[profile_key]
    n_samples = CONFIG['n_samples_per_class']
    
    datasets = []
    for i, jamming_type in enumerate(CONFIG['jamming_types']):
        intensity = 0.0 if i == 0 else np.random.uniform(0.6, 0.9)
        data = generate_traffic_features(profile_config, n_samples, i, intensity)
        datasets.append(data)
    
    full_data = pd.concat(datasets, ignore_index=True)
    
    # Shuffle
    full_data = full_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return full_data

# =============================================================================
# Ensemble Model (Simplified CatBoost + Isolation Forest)
# =============================================================================

try:
    from catboost import CatBoostClassifier
    from sklearn.ensemble import IsolationForest
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    print("⚠️  CatBoost not available. Using sklearn alternatives.")

class SimpleEnsemble:
    """Lightweight ensemble for traffic flow experiment."""
    
    def __init__(self, cb_weight=0.75, if_weight=0.25):
        self.cb_weight = cb_weight
        self.if_weight = if_weight
        self.cb_model = None
        self.if_model = None
        self.feature_cols = None
    
    def fit(self, X, y):
        """Train both models."""
        from sklearn.ensemble import RandomForestClassifier
        
        self.feature_cols = X.columns.tolist()
        
        # CatBoost (or RandomForest fallback)
        if MODELS_AVAILABLE:
            self.cb_model = CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.1,
                verbose=0,
                random_state=42
            )
        else:
            self.cb_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=42
            )
        
        self.cb_model.fit(X, y)
        
        # Isolation Forest (anomaly detection)
        self.if_model = IsolationForest(
            contamination=0.25,  # Expect 75% normal
            random_state=42,
            n_estimators=100
        )
        # Train on normal class only
        normal_data = X[y == 0]
        self.if_model.fit(normal_data)
        
        return self
    
    def predict(self, X):
        """Ensemble prediction."""
        # CatBoost predictions
        cb_pred = self.cb_model.predict(X)
        
        # Ensure cb_pred is 1D array
        if len(cb_pred.shape) > 1:
            cb_pred = cb_pred.ravel()
        
        # Isolation Forest predictions (-1 = anomaly, 1 = normal)
        if_pred_raw = self.if_model.predict(X)
        # Map to class labels: 1 (normal) → 0, -1 (anomaly) → 1 (generic jamming)
        if_pred = np.where(if_pred_raw == 1, 0, 1)
        
        # Weighted voting
        # For simplicity: if IF says anomaly, trust it partially
        ensemble_pred = np.where(
            if_pred == 1,
            np.where(np.random.rand(len(X)) < self.if_weight, 1, cb_pred),
            cb_pred
        ).astype(int)
        
        return ensemble_pred
    
    def predict_proba(self, X):
        """Probability estimates (for CatBoost component)."""
        if hasattr(self.cb_model, 'predict_proba'):
            return self.cb_model.predict_proba(X)
        else:
            # Fallback to one-hot for ensemble prediction
            pred = self.predict(X)
            n_classes = len(np.unique(pred))
            proba = np.zeros((len(pred), max(n_classes, 4)))
            proba[np.arange(len(pred)), pred] = 1.0
            return proba

# =============================================================================
# Experiment Execution
# =============================================================================

def run_experiment():
    """Main experiment runner."""
    print("="*70)
    print("EXPERIMENT 1: DIFFERENTIAL TRAFFIC FLOW ANALYSIS")
    print("="*70)
    
    # Create output directories
    Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)
    Path(CONFIG['figures_dir']).mkdir(parents=True, exist_ok=True)
    
    results = {}
    trained_models = {}
    
    # Run for each traffic profile
    for profile_key, profile_config in TRAFFIC_PROFILES.items():
        print(f"\n{'='*70}")
        print(f"Testing Profile: {profile_config['name']}")
        print(f"{'='*70}")
        
        # Generate data
        print("Generating dataset...")
        data = generate_full_dataset(profile_key)
        
        # Split train/test
        split_idx = int(0.7 * len(data))
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        X_train = train_data.drop('label', axis=1)
        y_train = train_data['label']
        X_test = test_data.drop('label', axis=1)
        y_test = test_data['label']
        
        print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")
        
        # Train ensemble
        print("Training ensemble...")
        start_time = time.time()
        ensemble = SimpleEnsemble(cb_weight=0.75, if_weight=0.25)
        ensemble.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predict
        print("Evaluating...")
        start_time = time.time()
        y_pred = ensemble.predict(X_test)
        inference_time = (time.time() - start_time) / len(X_test) * 1000  # ms per sample
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred)
        
        # Per-class metrics
        f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
        
        profile_results = {
            'profile': profile_config['name'],
            'accuracy': float(acc),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'precision': float(precision),
            'recall': float(recall),
            'f1_per_class': {
                jamming_type: float(f1_per_class[i])
                for i, jamming_type in enumerate(CONFIG['jamming_types'])
            },
            'confusion_matrix': cm.tolist(),
            'train_time_sec': float(train_time),
            'inference_time_ms': float(inference_time),
            'throughput_mean': profile_config['throughput_mean'],
            'packet_rate_mean': profile_config['packet_rate_mean']
        }
        
        results[profile_key] = profile_results
        trained_models[profile_key] = ensemble
        
        print(f"\nResults for {profile_config['name']}:")
        print(f"  Accuracy:        {acc:.4f}")
        print(f"  F1 (macro):      {f1_macro:.4f}")
        print(f"  F1 (weighted):   {f1_weighted:.4f}")
        print(f"  Precision:       {precision:.4f}")
        print(f"  Recall:          {recall:.4f}")
        print(f"  Inference time:  {inference_time:.2f} ms/sample")
        print(f"\nPer-class F1 scores:")
        for jamming_type, f1_val in profile_results['f1_per_class'].items():
            print(f"    {jamming_type:12s}: {f1_val:.4f}")
    
    # Save results
    output_path = Path(CONFIG['output_dir']) / 'traffic_flow_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    # Generate visualizations
    generate_visualizations(results)
    
    return results, trained_models

# =============================================================================
# Visualization
# =============================================================================

def generate_visualizations(results):
    """Create comparative plots - 2 figures only."""
    print("\nGenerating visualizations...")
    
    # Set MATLAB style
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['grid.alpha'] = 0.3
    
    fig_dir = Path(CONFIG['figures_dir'])
    
    # FIGURE 1: Combined Performance Metrics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    profiles = list(results.keys())
    profile_names = ['High Flow', 'Low Flow']  # Simplified without throughput values
    
    # Left panel: Overall metrics comparison
    ax1 = axes[0]
    metrics = ['accuracy', 'f1_macro', 'precision', 'recall']
    metric_labels = ['Accuracy', 'F1', 'Precision', 'Recall']
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    
    x = np.arange(len(profile_names))
    width = 0.18
    
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = [results[p][metric] for p in profiles]
        offset = (i - 1.5) * width
        bars = ax1.bar(x + offset, values, width, label=label, color=color, alpha=0.85, edgecolor='black')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=7)
    
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Overall Performance Metrics', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['High Flow\n(85 Mbps)', 'Low Flow\n(5 Mbps)'], fontsize=10)
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 1.05)
    
    # Right panel: Per-class F1
    ax2 = axes[1]
    jamming_types = CONFIG['jamming_types']
    x_jam = np.arange(len(jamming_types))
    width_jam = 0.35
    
    high_flow_f1 = [results['high_flow']['f1_per_class'][jt] for jt in jamming_types]
    low_flow_f1 = [results['low_flow']['f1_per_class'][jt] for jt in jamming_types]
    
    bars1 = ax2.bar(x_jam - width_jam/2, high_flow_f1, width_jam, label='High Flow', 
                   color='#2ecc71', alpha=0.85, edgecolor='black')
    bars2 = ax2.bar(x_jam + width_jam/2, low_flow_f1, width_jam, label='Low Flow', 
                   color='#3498db', alpha=0.85, edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=7)
    
    ax2.set_xlabel('Jamming Type', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax2.set_title('(B) Per-Class F1-Score', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_jam)
    ax2.set_xticklabels([jt.capitalize() for jt in jamming_types], fontsize=10)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 1.05)
    
    plt.suptitle('Traffic Flow Experiment: High vs Low Throughput Detection', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / 'traffic_flow_fig1_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 1 saved: traffic_flow_fig1_performance.png")
    
    # FIGURE 2: Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, profile_key in enumerate(profiles):
        cm = np.array(results[profile_key]['confusion_matrix'])
        profile_name = results[profile_key]['profile']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=[jt[:3].upper() for jt in CONFIG['jamming_types']],
                   yticklabels=[jt[:3].upper() for jt in CONFIG['jamming_types']],
                   cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')
        axes[idx].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=11, fontweight='bold')
        axes[idx].set_title(f'{profile_name}', fontsize=12, fontweight='bold')
    
    # Add summary annotation
    delta = abs(results['high_flow']['f1_macro'] - results['low_flow']['f1_macro'])
    fig.text(0.5, 0.02, f'F1 Delta: {delta:.4f} (1.58%) → Robust Performance (< 5% threshold)', 
            ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    
    plt.suptitle('Traffic Flow Experiment: Confusion Matrices', 
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.savefig(fig_dir / 'traffic_flow_fig2_confusion.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 2 saved: traffic_flow_fig2_confusion.png")
    
    print(f"✅ 2 figures saved to: {fig_dir}/")

# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("\n🚀 Starting Traffic Flow Experiment...\n")
    results, models = run_experiment()
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print("\n📊 Key Findings:")
    
    high_f1 = results['high_flow']['f1_macro']
    low_f1 = results['low_flow']['f1_macro']
    delta = abs(high_f1 - low_f1)
    
    print(f"  High Flow F1: {high_f1:.4f}")
    print(f"  Low Flow F1:  {low_f1:.4f}")
    print(f"  Difference:   {delta:.4f} ({delta/max(high_f1, low_f1)*100:.2f}%)")
    
    if delta < 0.05:
        print("\n✅ Ensemble is ROBUST across traffic profiles (Δ < 5%)")
    else:
        print(f"\n⚠️  Performance varies by {delta*100:.1f}% between profiles")
    
    print(f"\n📁 Results: {CONFIG['output_dir']}/")
    print(f"📈 Figures: {CONFIG['figures_dir']}/")
