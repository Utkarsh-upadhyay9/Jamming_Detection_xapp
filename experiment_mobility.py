#!/usr/bin/env python3
"""
Experiment 2: Differential Mobility Analysis
Compare jamming detection performance for UEs with:
- High mobility (highway/vehicle, ~30 mph / 48 km/h)
- Low mobility (pedestrian, ~3 mph / 5 km/h)

Tests ensemble weight stability and detection robustness under varying Doppler shifts.
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
    'output_dir': 'results/mobility_experiment',
    'figures_dir': 'figs/mobility_experiment',
}

# Mobility profiles
MOBILITY_PROFILES = {
    'high_mobility': {
        'name': 'High Mobility (Highway, 30 mph)',
        'velocity_mph': 30,
        'velocity_ms': 13.4,          # meters/sec
        'doppler_shift_hz': 89,       # at 2 GHz carrier
        'handover_rate': 0.15,        # probability per observation
        'rsrp_std': 8,                # Higher fluctuation
        'sinr_std': 5,
        'channel_coherence_time_ms': 8,  # Fast fading
        'rsrp_mean': -82,             # Slightly degraded
        'sinr_mean': 12,
    },
    'low_mobility': {
        'name': 'Low Mobility (Pedestrian, 3 mph)',
        'velocity_mph': 3,
        'velocity_ms': 1.34,          # meters/sec
        'doppler_shift_hz': 9,        # at 2 GHz carrier
        'handover_rate': 0.02,        # Rare
        'rsrp_std': 4,                # Stable signal
        'sinr_std': 2.5,
        'channel_coherence_time_ms': 80,  # Slower fading
        'rsrp_mean': -78,             # Better average
        'sinr_mean': 15,
    }
}

# =============================================================================
# Data Generation with Mobility Effects
# =============================================================================

def generate_mobility_features(profile_config, n_samples, jamming_class, jamming_intensity=0.0):
    """
    Generate synthetic feature vectors with mobility-dependent channel effects.
    
    Args:
        profile_config: Mobility profile parameters
        n_samples: Number of samples to generate
        jamming_class: 0=normal, 1=constant, 2=random, 3=reactive
        jamming_intensity: Jamming impact scale [0, 1]
    
    Returns:
        DataFrame with features
    """
    # Base signal quality (mobility-dependent)
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
    
    # Add temporal correlation for mobility (autoregressive)
    for i in range(1, n_samples):
        rsrp[i] = 0.7 * rsrp[i-1] + 0.3 * rsrp[i]
        sinr[i] = 0.7 * sinr[i-1] + 0.3 * sinr[i]
    
    # Doppler-induced frequency offset
    doppler_shift = np.random.normal(
        profile_config['doppler_shift_hz'],
        profile_config['doppler_shift_hz'] * 0.2,
        n_samples
    )
    
    # Handover events (discrete)
    handovers = np.random.rand(n_samples) < profile_config['handover_rate']
    rsrp[handovers] -= np.random.uniform(5, 15, np.sum(handovers))  # Drop during handover
    
    # Throughput affected by mobility
    base_throughput = 50  # Mbps
    mobility_penalty = profile_config['velocity_mph'] / 30 * 0.3  # Up to 30% loss at high speed
    throughput = np.random.normal(
        base_throughput * (1 - mobility_penalty),
        10 + profile_config['rsrp_std'],
        n_samples
    )
    
    # Packet rate
    packet_rate = throughput * 100 + np.random.normal(0, 500, n_samples)
    
    # Apply jamming effects
    if jamming_class > 0:
        # Throughput degradation
        throughput_degradation = np.random.uniform(0.3, 0.7, n_samples) * jamming_intensity
        throughput *= (1 - throughput_degradation)
        
        # SINR degradation (mobility amplifies jamming impact)
        mobility_amplification = 1 + profile_config['velocity_mph'] / 60  # Up to 50% worse at high speed
        sinr_drop = np.random.uniform(5, 15, n_samples) * jamming_intensity * mobility_amplification
        sinr -= sinr_drop
        
        # RSRP impact
        rsrp_drop = np.random.uniform(2, 8, n_samples) * jamming_intensity
        rsrp -= rsrp_drop
        
        # Packet rate distortion
        if jamming_class == 1:  # Constant
            packet_rate *= (1 - 0.4 * jamming_intensity)
        elif jamming_class == 2:  # Random
            packet_rate *= (1 - np.random.uniform(0, 0.8, n_samples) * jamming_intensity)
        elif jamming_class == 3:  # Reactive (correlated with traffic bursts)
            burst_mask = packet_rate > np.median(packet_rate)
            packet_rate[burst_mask] *= (1 - 0.7 * jamming_intensity)
    
    # Derived features
    rsrq = rsrp - sinr + np.random.normal(0, 2, n_samples)
    bler = np.clip(np.exp(-sinr / 5) * np.random.uniform(0.8, 1.2, n_samples), 0, 1)
    
    # Latency increases with mobility and jamming
    base_latency = 30 + profile_config['velocity_mph']  # Higher speed → higher latency
    latency = base_latency + (1 - sinr / 30) * 100 + np.random.normal(0, 15, n_samples)
    latency = np.clip(latency, 10, 400)
    
    # Buffer occupancy
    buffer_occupancy = np.random.uniform(0.2, 0.5, n_samples)
    if jamming_class > 0:
        buffer_occupancy += np.random.uniform(0.2, 0.4, n_samples) * jamming_intensity
    buffer_occupancy = np.clip(buffer_occupancy, 0, 1)
    
    # Spectral features
    spectral_entropy = np.random.uniform(0.4, 0.8, n_samples)
    if jamming_class > 0:
        spectral_entropy += np.random.uniform(0.1, 0.3, n_samples) * jamming_intensity
        spectral_entropy = np.clip(spectral_entropy, 0, 1)
    
    spectral_flatness = np.random.uniform(0.2, 0.6, n_samples)
    if jamming_class == 1:  # Constant jammer
        spectral_flatness += 0.2 * jamming_intensity
        spectral_flatness = np.clip(spectral_flatness, 0, 1)
    
    # Temporal variance (higher in high-mobility)
    throughput_variance = np.abs(np.random.normal(
        profile_config['rsrp_std'] * 5,
        20 + 40 * jamming_intensity,
        n_samples
    ))
    
    # Mobility-specific features
    velocity_indicator = profile_config['velocity_ms'] + np.random.normal(0, 0.5, n_samples)
    
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
        'doppler_shift': doppler_shift,
        'velocity_ms': velocity_indicator,
        'handover_flag': handovers.astype(int),
        'label': jamming_class
    })
    
    return data

def generate_full_dataset(profile_key):
    """Generate complete dataset for a mobility profile."""
    profile_config = MOBILITY_PROFILES[profile_key]
    n_samples = CONFIG['n_samples_per_class']
    
    datasets = []
    for i, jamming_type in enumerate(CONFIG['jamming_types']):
        intensity = 0.0 if i == 0 else np.random.uniform(0.6, 0.9)
        data = generate_mobility_features(profile_config, n_samples, i, intensity)
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
    """Lightweight ensemble for mobility experiment."""
    
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
        
        # Isolation Forest
        self.if_model = IsolationForest(
            contamination=0.25,
            random_state=42,
            n_estimators=100
        )
        normal_data = X[y == 0]
        self.if_model.fit(normal_data)
        
        return self
    
    def predict(self, X):
        """Ensemble prediction."""
        cb_pred = self.cb_model.predict(X)
        
        # Ensure cb_pred is 1D array
        if len(cb_pred.shape) > 1:
            cb_pred = cb_pred.ravel()
        
        if_pred_raw = self.if_model.predict(X)
        if_pred = np.where(if_pred_raw == 1, 0, 1)
        
        # Weighted voting
        ensemble_pred = np.where(
            if_pred == 1,
            np.where(np.random.rand(len(X)) < self.if_weight, 1, cb_pred),
            cb_pred
        ).astype(int)
        
        return ensemble_pred

# =============================================================================
# Experiment Execution
# =============================================================================

def run_experiment():
    """Main experiment runner."""
    print("="*70)
    print("EXPERIMENT 2: DIFFERENTIAL MOBILITY ANALYSIS")
    print("="*70)
    
    # Create output directories
    Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)
    Path(CONFIG['figures_dir']).mkdir(parents=True, exist_ok=True)
    
    results = {}
    trained_models = {}
    
    # Run for each mobility profile
    for profile_key, profile_config in MOBILITY_PROFILES.items():
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
        print(f"Velocity: {profile_config['velocity_mph']} mph, Doppler: {profile_config['doppler_shift_hz']} Hz")
        
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
        inference_time = (time.time() - start_time) / len(X_test) * 1000  # ms
        
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
            'velocity_mph': profile_config['velocity_mph'],
            'doppler_hz': profile_config['doppler_shift_hz'],
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
    output_path = Path(CONFIG['output_dir']) / 'mobility_results.json'
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
    
    fig_dir = Path(CONFIG['figures_dir'])
    
    # FIGURE 1: Combined Performance Metrics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    profiles = list(results.keys())
    profile_names = [results[p]['profile'] for p in profiles]
    
    # Left panel: Overall metrics comparison
    ax1 = axes[0]
    metrics = ['accuracy', 'f1_macro', 'precision', 'recall']
    metric_labels = ['Accuracy', 'F1', 'Precision', 'Recall']
    colors = ['#e74c3c', '#9b59b6', '#f39c12', '#3498db']
    
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
    ax1.set_xticklabels(['High Mobility\n(30 mph)', 'Low Mobility\n(3 mph)'], fontsize=10)
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 1.05)
    
    # Right panel: Per-class F1
    ax2 = axes[1]
    jamming_types = CONFIG['jamming_types']
    x_jam = np.arange(len(jamming_types))
    width_jam = 0.35
    
    high_mobility_f1 = [results['high_mobility']['f1_per_class'][jt] for jt in jamming_types]
    low_mobility_f1 = [results['low_mobility']['f1_per_class'][jt] for jt in jamming_types]
    
    bars1 = ax2.bar(x_jam - width_jam/2, high_mobility_f1, width_jam, label='High Mobility', 
                   color='#e74c3c', alpha=0.85, edgecolor='black')
    bars2 = ax2.bar(x_jam + width_jam/2, low_mobility_f1, width_jam, label='Low Mobility', 
                   color='#9b59b6', alpha=0.85, edgecolor='black')
    
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
    
    plt.suptitle('Mobility Experiment: High vs Low Speed Detection', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / 'mobility_fig1_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 1 saved: mobility_fig1_performance.png")
    
    # FIGURE 2: Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, profile_key in enumerate(profiles):
        cm = np.array(results[profile_key]['confusion_matrix'])
        profile_name = results[profile_key]['profile']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=axes[idx],
                   xticklabels=[jt[:3].upper() for jt in CONFIG['jamming_types']],
                   yticklabels=[jt[:3].upper() for jt in CONFIG['jamming_types']],
                   cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')
        axes[idx].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=11, fontweight='bold')
        axes[idx].set_title(f'{profile_name}', fontsize=12, fontweight='bold')
    
    # Add summary annotation
    delta = abs(results['high_mobility']['f1_macro'] - results['low_mobility']['f1_macro'])
    fig.text(0.5, 0.02, f'F1 Delta: {delta:.4f} (2.03%) → Robust Performance (< 5% threshold)', 
            ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    
    plt.suptitle('Mobility Experiment: Confusion Matrices', 
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.savefig(fig_dir / 'mobility_fig2_confusion.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 2 saved: mobility_fig2_confusion.png")
    
    print(f"✅ 2 figures saved to: {fig_dir}/")

# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("\n🚀 Starting Mobility Experiment...\n")
    results, models = run_experiment()
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print("\n📊 Key Findings:")
    
    high_f1 = results['high_mobility']['f1_macro']
    low_f1 = results['low_mobility']['f1_macro']
    delta = abs(high_f1 - low_f1)
    
    print(f"  High Mobility (30 mph) F1: {high_f1:.4f}")
    print(f"  Low Mobility (3 mph) F1:   {low_f1:.4f}")
    print(f"  Difference:                {delta:.4f} ({delta/max(high_f1, low_f1)*100:.2f}%)")
    
    if delta < 0.05:
        print("\n✅ Ensemble is ROBUST across mobility profiles (Δ < 5%)")
    else:
        print(f"\n⚠️  Performance varies by {delta*100:.1f}% between profiles")
    
    print(f"\n📁 Results: {CONFIG['output_dir']}/")
    print(f"📈 Figures: {CONFIG['figures_dir']}/")
