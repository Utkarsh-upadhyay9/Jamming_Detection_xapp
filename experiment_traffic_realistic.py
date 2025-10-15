#!/usr/bin/env python3
"""
REALISTIC Traffic Flow Experiment - Challenging Environment
Uses realistic noise, fading, and interference levels from the paper
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from catboost import CatBoostClassifier
from sklearn.ensemble import IsolationForest, ExtraTreesClassifier, GradientBoostingClassifier
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from improved_feature_engineering import extract_advanced_features

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'serif'
matlab_colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30']

np.random.seed(42)


class RealisticEnsemble:
    """Enhanced ensemble for realistic scenarios"""
    
    def __init__(self):
        self.models = {}
        self.weights = {
            'catboost': 0.75,
            'isolation_forest': 0.25
        }
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def _initialize_models(self):
        """Initialize models with paper-specified hyperparameters"""
        models = {}
        
        # CatBoost from paper (T=3000, depth=10, lr=0.03, L2=3.0)
        models['catboost'] = CatBoostClassifier(
            iterations=3000,
            learning_rate=0.03,
            depth=10,
            l2_leaf_reg=3.0,
            border_count=254,
            random_strength=1,
            verbose=False,
            random_seed=42,
            thread_count=-1,
            auto_class_weights='Balanced'
        )
        
        # Isolation Forest from paper (t=300, contamination=0.25)
        models['isolation_forest'] = IsolationForest(
            n_estimators=300,
            max_samples=0.8,
            contamination=0.25,
            max_features=0.9,
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        
        return models
    
    def fit(self, X, y):
        """Train ensemble"""
        print("\nTraining ensemble...")
        X_scaled = self.scaler.fit_transform(X)
        self.models = self._initialize_models()
        
        for name, model in self.models.items():
            print(f"  Training {name}...")
            if name == 'isolation_forest':
                model.fit(X_scaled)
            else:
                model.fit(X_scaled, y)
        
        self.is_trained = True
        print("✓ Training complete")
    
    def predict(self, X):
        """Ensemble prediction with 75/25 weighting"""
        if not self.is_trained:
            raise ValueError("Models must be trained first")
        
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        cb_pred = self.models['catboost'].predict(X_scaled)
        if_pred = self.models['isolation_forest'].predict(X_scaled)
        
        # Convert IF anomaly to jamming indicator
        if_pred = np.where(if_pred == 1, 0, 1)  # 1=normal, -1=anomaly
        
        # Weighted voting
        ensemble_pred = np.zeros(len(X), dtype=int)
        for i in range(len(X)):
            votes = {}
            
            # CatBoost vote (75% weight)
            cb_vote = int(cb_pred[i])
            votes[cb_vote] = votes.get(cb_vote, 0) + self.weights['catboost']
            
            # IF binary vote (25% weight) - split among non-normal classes if anomaly
            if if_pred[i] == 0:
                votes[0] = votes.get(0, 0) + self.weights['isolation_forest']
            else:
                # IF says anomaly, support CatBoost's specific jamming type
                votes[cb_vote] = votes.get(cb_vote, 0) + self.weights['isolation_forest']
            
            ensemble_pred[i] = max(votes, key=votes.get)
        
        return ensemble_pred


def generate_realistic_traffic_features(n_samples, class_label):
    """
    Generate REALISTIC features using challenging environment from paper:
    - Noise floor: σ²_noise = 4.5
    - Rayleigh fading: σ = 0.75  
    - Interference: I₀ = -95 dBm
    - SNR factor: β = 0.65
    - Environmental degradation: δ = 0.4
    - Multipath: τ_delay ~ Exponential(λ=0.08)
    """
    
    # Realistic environment parameters (from paper)
    noise_variance = 4.5
    fading_param = 0.75
    interference_dbm = -95
    snr_factor = 0.65
    env_degradation = 0.4
    delay_lambda = 0.08
    
    # Class-specific jamming characteristics with REALISTIC overlap
    if class_label == 0:  # Normal
        jamming_factor = np.random.uniform(0.0, 0.15, n_samples)  # Some natural variation
        base_throughput = 70
        base_packet_rate = 7000
    elif class_label == 1:  # Constant jamming
        jamming_factor = np.random.uniform(0.5, 0.85, n_samples)  # Not always 70%
        base_throughput = 35
        base_packet_rate = 3500
    elif class_label == 2:  # Random jamming
        jamming_factor = np.random.uniform(0.3, 0.9, n_samples)  # High variance
        base_throughput = 40
        base_packet_rate = 4000
    else:  # Reactive jamming  
        jamming_factor = np.random.uniform(0.4, 0.95, n_samples)  # Adaptive
        base_throughput = 30
        base_packet_rate = 3000
    
    # Generate time-series with realistic impairments
    raw_signals = {}
    
    # RSRP with Rayleigh fading and multipath
    base_rsrp = -75  # More realistic starting point
    rayleigh_fading = np.random.rayleigh(fading_param, n_samples) * 10
    multipath_delay = np.random.exponential(1/delay_lambda, n_samples)
    rsrp_series = base_rsrp - jamming_factor * 15  # Reduced separation
    rsrp_series -= rayleigh_fading
    rsrp_series -= multipath_delay * 2
    rsrp_series += np.random.normal(0, np.sqrt(noise_variance), n_samples)
    rsrp_series += env_degradation * np.random.randn(n_samples) * 5
    raw_signals['rsrp'] = rsrp_series
    
    # SINR with realistic degradation
    base_sinr = 12  # Lower baseline
    sinr_series = base_sinr * snr_factor - jamming_factor * 18  # Reduced separation
    sinr_series -= rayleigh_fading * 0.8
    sinr_series += np.random.normal(0, np.sqrt(noise_variance) * 1.5, n_samples)
    sinr_series += env_degradation * np.random.randn(n_samples) * 4
    raw_signals['sinr'] = sinr_series
    
    # Throughput with high variance and overlap
    throughput_mean = base_throughput * (1 - jamming_factor * 0.6)
    throughput_std = 20 + env_degradation * 15  # High variance
    throughput_series = np.maximum(0, np.random.normal(throughput_mean, throughput_std, n_samples))
    # Add bursty traffic patterns
    throughput_series += np.random.exponential(5, n_samples) * np.random.choice([-1, 1], n_samples)
    raw_signals['throughput'] = throughput_series
    
    # Packet rate with realistic correlation
    packet_rate_mean = base_packet_rate * (1 - jamming_factor * 0.7)
    packet_rate_std = 1500 + env_degradation * 1000  # High variance
    packet_rate_series = np.maximum(0, np.random.normal(packet_rate_mean, packet_rate_std, n_samples))
    # Add correlation with throughput (imperfect)
    packet_rate_series += 0.3 * (throughput_series - np.mean(throughput_series))
    raw_signals['packet_rate'] = packet_rate_series
    
    # Extract advanced features
    features_df = extract_advanced_features(raw_signals)
    features_df['label'] = class_label
    
    return features_df


def run_realistic_experiment():
    """Run realistic traffic flow experiment"""
    
    print("="*70)
    print("REALISTIC TRAFFIC FLOW EXPERIMENT")
    print("Using CHALLENGING environment from paper")
    print("="*70)
    
    # Generate realistic dataset
    print("\nGenerating realistic dataset...")
    train_data = []
    samples_per_class = [225, 100, 100, 75]  # Imbalanced like paper (45%, 20%, 20%, 15%)
    
    for class_label in range(4):
        print(f"  Class {class_label}: generating {samples_per_class[class_label]} samples...", end='')
        for _ in range(samples_per_class[class_label]):
            df = generate_realistic_traffic_features(100, class_label)
            train_data.append(df)
        print(" ✓")
    
    train_df = pd.concat(train_data, ignore_index=True)
    print(f"Total samples: {len(train_df)}")
    print(f"Total features: {len(train_df.columns) - 1}")
    
    X = train_df.drop('label', axis=1).values
    y = train_df['label'].values
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    # Train ensemble
    ensemble = RealisticEnsemble()
    ensemble.fit(X_train, y_train)
    
    # Evaluate
    print("\n" + "="*70)
    print("EVALUATION ON TEST SET")
    print("="*70)
    y_pred = ensemble.predict(X_test)
    
    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    f1_per_class = f1_score(y_test, y_pred, average=None)
    
    print(f"\n✅ TEST SET RESULTS:")
    print(f"  Weighted F1: {f1:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    class_names = ['Normal', 'Constant', 'Random', 'Reactive']
    print(f"\nPer-Class F1:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {f1_per_class[i]:.4f}")
    
    # Cross-validation
    print("\n" + "="*70)
    print("5-FOLD CROSS-VALIDATION")
    print("="*70)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
        
        fold_ensemble = RealisticEnsemble()
        fold_ensemble.fit(X_fold_train, y_fold_train)
        y_fold_pred = fold_ensemble.predict(X_fold_val)
        fold_f1 = f1_score(y_fold_val, y_fold_pred, average='weighted')
        cv_scores.append(fold_f1)
        print(f"  Fold {fold+1}: F1 = {fold_f1:.4f}")
    
    print(f"\nCV Mean F1: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # Save results
    results = {
        'test_f1': float(f1),
        'test_accuracy': float(accuracy),
        'f1_per_class': {class_names[i]: float(f1_per_class[i]) for i in range(4)},
        'cv_mean': float(np.mean(cv_scores)),
        'cv_std': float(np.std(cv_scores)),
        'cv_scores': [float(x) for x in cv_scores]
    }
    
    with open('results/traffic_realistic_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate figure
    generate_figure(y_test, y_pred, f1, accuracy, class_names, f1_per_class)
    
    return results


def generate_figure(y_test, y_pred, f1, accuracy, class_names, f1_per_class):
    """Generate publication-quality figure"""
    
    import os
    os.makedirs('figs/realistic_experiments/', exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Confusion matrix
    ax = axes[0]
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
    ax.set_title(f'(a) Confusion Matrix\nF1={f1:.4f}, Acc={accuracy:.4f}',
                 fontsize=12, fontweight='bold')
    
    # Per-class F1
    ax = axes[1]
    bars = ax.bar(class_names, f1_per_class, color=matlab_colors[:4], alpha=0.8, edgecolor='black')
    ax.axhline(y=0.90, color='red', linestyle='--', linewidth=1.5, label='90% Target', alpha=0.7)
    ax.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
    ax.set_xlabel('Jamming Type', fontsize=11, fontweight='bold')
    ax.set_title('(b) Per-Class F1-Score Performance', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, value in zip(bars, f1_per_class):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figs/realistic_experiments/traffic_realistic_performance.png', 
                dpi=300, bbox_inches='tight')
    print(f"\n✅ Figure saved: figs/realistic_experiments/traffic_realistic_performance.png")
    plt.close()


if __name__ == '__main__':
    results = run_realistic_experiment()
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Test F1-Score: {results['test_f1']:.4f}")
    print(f"CV F1-Score: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
    
    if results['test_f1'] >= 0.90:
        print("✅ TARGET ACHIEVED: F1 > 0.90!")
    else:
        gap = 0.90 - results['test_f1']
        print(f"Gap to 90%: {gap:.4f} ({gap*100:.1f} percentage points)")
    
    print(f"\nPer-class results:")
    for cls, f1_val in results['f1_per_class'].items():
        status = "✅" if f1_val >= 0.90 else "⚠️"
        print(f"  {status} {cls}: {f1_val:.4f}")
