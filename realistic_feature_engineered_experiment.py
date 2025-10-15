#!/usr/bin/env python3
"""
Realistic experiment using time-series feature engineering (no easy separation).
- Challenging environment per paper
- Simulates multipath (exp delay spread), Rayleigh fading, strong interference
- Extracts advanced features via improved_feature_engineering.extract_advanced_features
- Trains robust ensemble and evaluates with CV
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns

from improved_feature_engineering import extract_advanced_features

np.random.seed(42)

# Realistic environment parameters
NOISE_FLOOR = 4.5          # σ²_noise surrogate (std magnitude)
RAYLEIGH_SIGMA = 0.75      # Rayleigh fading σ
I0_DBM = -95               # Interference floor
BETA = 0.65                # SNR factor
DELTA = 0.4                # Environmental degradation factor
LAMBDA_DELAY = 0.08        # Exponential multipath parameter

T = 64  # time samples per signal
N_PER_CLASS = 200
CLASSES = {0: 'Normal', 1: 'Constant', 2: 'Random', 3: 'Reactive'}


def exp_kernel(T: int, lam: float) -> np.ndarray:
    k = np.exp(-lam * np.arange(T))
    k /= k.sum() + 1e-12
    return k


def correlated_noise(std: float, T: int) -> np.ndarray:
    # Base white noise then smooth via exponential moving average to add correlation
    w = np.random.normal(0, std, T)
    alpha = 0.2
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = alpha * w[t] + (1 - alpha) * y[t-1]
    return y


def simulate_signals(cls: int) -> Dict[str, np.ndarray]:
    # jamming factor and pattern per class
    if cls == 0:   # Normal
        jf = np.random.uniform(0.0, 0.08)
        var_scale = 0.3
        burst_prob = 0.02
    elif cls == 1: # Constant
        jf = np.random.uniform(0.65, 0.85)
        var_scale = 0.2
        burst_prob = 0.03
    elif cls == 2: # Random
        jf = np.random.uniform(0.35, 0.95)
        var_scale = 1.0
        burst_prob = 0.15
    else:          # Reactive
        jf = np.random.uniform(0.45, 0.9)
        var_scale = 0.7
        burst_prob = 0.12

    # Base time axes
    t = np.arange(T)

    # Multipath as convolution with exponential kernel
    k = exp_kernel(T, LAMBDA_DELAY)

    # Rayleigh fading over time (slowly varying)
    fading = np.random.rayleigh(RAYLEIGH_SIGMA, T)
    fading = np.convolve(fading, k, mode='same')

    # Interference baseline (convert dBm surrogate to normalized offset)
    i_base = (I0_DBM + 120) / 40.0  # map to ~[0,1]

    # Generate core signals with AWGN, fading, interference and class-specific variance
    rsrp_base = -70 - jf * 25 + (1 - fading) * 8
    rsrp = rsrp_base + correlated_noise(NOISE_FLOOR * 0.9, T) + np.random.normal(0, 6 * DELTA, T)

    sinr_base = 16 * BETA - jf * 30
    sinr = sinr_base + correlated_noise(NOISE_FLOOR * 0.7, T) - i_base * 2 + np.random.normal(0, 8 * DELTA, T)

    th_base = 75 * (1 - jf * 0.75) * BETA
    throughput = th_base + correlated_noise(10 * (1 + DELTA) * var_scale, T)
    throughput = np.clip(throughput, 0, None)

    pr_base = 7200 * (1 - jf * 0.8) * BETA
    packet_rate = pr_base + correlated_noise(1200 * (1 + DELTA) * var_scale, T)
    packet_rate = np.clip(packet_rate, 0, None)

    # Buffer occupancy and BLER
    buffer = np.clip(jf * 0.8 + np.random.normal(0, 0.08 * (1 + DELTA), T), 0, 1)
    bler = np.clip(jf * 0.65 + np.abs(correlated_noise(0.12, T)), 0, 1)

    # Spectral features proxies (time-varying)
    spectral_entropy = np.clip(0.35 + jf * 0.45 + np.random.normal(0, 0.08 * var_scale, T), 0, 1)
    spectral_flatness = np.clip(0.55 - jf * 0.35 + np.random.normal(0, 0.07 * var_scale, T), 0, 1)

    # Latency with bursts
    latency = 25 + jf * 280 + np.random.normal(0, 15, T)
    bursts = (np.random.rand(T) < burst_prob).astype(float)
    latency += bursts * np.random.uniform(60, 180, T)

    # Apply multipath smoothing to key series
    rsrp = np.convolve(rsrp, k, mode='same')
    sinr = np.convolve(sinr, k, mode='same')
    throughput = np.convolve(throughput, k, mode='same')
    packet_rate = np.convolve(packet_rate, k, mode='same')

    return {
        'rsrp': rsrp,
        'sinr': sinr,
        'throughput': throughput,
        'packet_rate': packet_rate,
        'buffer': buffer,
        'spectral_entropy': spectral_entropy,
        'spectral_flatness': spectral_flatness,
        'bler': bler,
        'latency': latency
    }


def build_dataset() -> pd.DataFrame:
    rows = []
    for cls in CLASSES.keys():
        for _ in range(N_PER_CLASS):
            raw = simulate_signals(cls)
            # Map keys to names expected by feature extractor
            raw_signals = {
                'rsrp': raw['rsrp'],
                'sinr': raw['sinr'],
                'throughput': raw['throughput'],
                'packet_rate': raw['packet_rate'],
            }
            feats = extract_advanced_features(raw_signals, fast=True)

            # Add a few domain extras
            feats['buffer_mean'] = np.mean(raw['buffer'])
            feats['bler_mean'] = np.mean(raw['bler'])
            feats['latency_mean'] = np.mean(raw['latency'])
            feats['entropy_mean'] = np.mean(raw['spectral_entropy'])
            feats['flatness_mean'] = np.mean(raw['spectral_flatness'])
            feats['label'] = cls
            rows.append(feats)
    df = pd.concat(rows, ignore_index=True)
    return df


def train_and_eval(df: pd.DataFrame) -> dict:
    X = df.drop('label', axis=1).values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    scaler = RobustScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Main model
    cb = CatBoostClassifier(
        iterations=600,
        learning_rate=0.04,
        depth=10,
        l2_leaf_reg=6,
        border_count=128,
        random_seed=42,
        verbose=False,
        auto_class_weights='Balanced'
    )
    cb.fit(X_train_sc, y_train, eval_set=(X_test_sc, y_test), early_stopping_rounds=150, verbose=False)
    cb_pred = cb.predict(X_test_sc)

    # Diversity model
    et = ExtraTreesClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    et.fit(X_train_sc, y_train)
    et_pred = et.predict(X_test_sc)

    # Weighted ensemble
    weights = {'cb': 0.7, 'et': 0.3}
    ens = []
    for i in range(len(y_test)):
        votes = {}
        cb_i = int(cb_pred[i]) if hasattr(cb_pred[i], 'item') is False else int(cb_pred[i].item())
        et_i = int(et_pred[i]) if hasattr(et_pred[i], 'item') is False else int(et_pred[i].item())
        votes[cb_i] = votes.get(cb_i, 0) + weights['cb']
        votes[et_i] = votes.get(et_i, 0) + weights['et']
        ens.append(max(votes, key=votes.get))

    f1 = f1_score(y_test, ens, average='weighted')
    acc = accuracy_score(y_test, ens)
    f1_per_class = f1_score(y_test, ens, average=None)

    # CV for stability
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_f1 = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]
        scf = RobustScaler()
        Xtr = scf.fit_transform(Xtr)
        Xva = scf.transform(Xva)
        cbf = CatBoostClassifier(iterations=400, learning_rate=0.05, depth=10, random_seed=42, verbose=False, auto_class_weights='Balanced')
        cbf.fit(Xtr, ytr, verbose=False)
        yhat = cbf.predict(Xva)
        cv_f1.append(f1_score(yva, yhat, average='weighted'))

    results = {
        'weighted_f1': float(f1),
        'accuracy': float(acc),
        'f1_per_class': [float(x) for x in f1_per_class],
        'cv_mean': float(np.mean(cv_f1)),
        'cv_std': float(np.std(cv_f1)),
        'num_features': int(X.shape[1]),
        'samples': int(len(X)),
        'environment': 'realistic_challenging_sequences',
        'noise_floor': NOISE_FLOOR,
        'interference_dbm': I0_DBM
    }

    os.makedirs('results', exist_ok=True)
    with open('results/realistic_feature_engineered_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Confusion matrix
    plt.figure(figsize=(7,5))
    cm = confusion_matrix(y_test, ens)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(CLASSES.values()), yticklabels=list(CLASSES.values()))
    plt.title(f'Realistic Feature-Engineered (F1={f1:.3f})')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('results/realistic_feature_engineered_cm.png', dpi=150)

    return results


if __name__ == '__main__':
    print('Building feature-engineered realistic dataset...')
    df = build_dataset()
    print(f'Dataset shape: {df.shape}')
    res = train_and_eval(df)
    print(f"\nWeighted F1: {res['weighted_f1']:.4f} | Acc: {res['accuracy']:.4f} | CV: {res['cv_mean']:.4f} ± {res['cv_std']:.4f}")
    print('Saved: results/realistic_feature_engineered_results.json and CM plot')
