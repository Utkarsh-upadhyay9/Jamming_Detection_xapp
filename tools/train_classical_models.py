#!/usr/bin/env python3
"""Train classical models (RF, SVM, IsolationForest) on a labeled CSV and save models/metrics/plots.

Usage:
    python3 tools/train_classical_models.py --csv synthetic_dataset.csv --out models/classical_models
"""
import sys
import os
# Ensure repository root is on sys.path so local package imports (models, src) work
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import argparse
import os
import json
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

from models.rf_model import RandomForestJammingDetector
from models.svm_model import SVMJammingDetector
from models.isolation_forest_model import IsolationForestJammingDetector
from src.data_processor import JammingDataProcessor
from config.model_config import TRAINING_CONFIG


def load_tabular_dataset(csv_path: str) -> Tuple[np.ndarray, np.ndarray, list]:
    df = pd.read_csv(csv_path)
    if 'label' not in df.columns:
        raise ValueError("CSV must contain a 'label' column with class names")

    feature_columns = [c for c in df.columns if c != 'label']
    X = df[feature_columns].values
    y = df['label'].values
    return X, y, feature_columns


def parse_raw_metrics_file(path: str, n_features: int = 15, max_lines: int | None = None) -> np.ndarray:
    """Parse a raw metrics text file into numeric feature rows.

    Heuristic parser: each non-empty line is scanned for numeric tokens. The first
    `n_features` numeric tokens become a row; if fewer tokens are found the row is
    padded with zeros. This is a best-effort parser intended to convert the raw
    E2-style logs shipped in `Ensemble_ML_Jamming_detection_dataset` into a tabular
    form for classical model training.
    """
    import re

    float_re = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")
    rows = []
    with open(path, 'r') as fh:
        for i, line in enumerate(fh):
            if max_lines is not None and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            tokens = float_re.findall(line)
            if not tokens:
                continue
            nums = [float(t) for t in tokens]
            if len(nums) >= n_features:
                row = nums[:n_features]
            else:
                row = nums + [0.0] * (n_features - len(nums))
            rows.append(row)

    if not rows:
        raise ValueError(f"No numeric rows parsed from {path}")

    return np.array(rows)


def prepare_and_split(X, y, test_size: float, random_state: int):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_and_evaluate(csv_path: str, out_dir: str, test_size: float, random_state: int):
    os.makedirs(out_dir, exist_ok=True)
    figures_dir = os.path.join(out_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # Allow either a single labeled CSV or two raw metric files (normal + power jammer)
    if csv_path and os.path.exists(csv_path):
        print(f"Loading dataset from {csv_path}")
        X, y, feature_names = load_tabular_dataset(csv_path)
        print(f"Dataset shape: X={X.shape}, labels={np.unique(y)}")
    else:
        raise ValueError("No CSV dataset provided; use --csv <path> or supply --normal-file and --power-file via CLI")

    X_train, X_test, y_train, y_test, scaler = prepare_and_split(X, y, test_size, random_state)

    # Initialize models
    rf = RandomForestJammingDetector()
    svm = SVMJammingDetector()
    if_model = IsolationForestJammingDetector()

    results = {}

    # Random Forest
    print("Training Random Forest...")
    rf_metrics = rf.train(X_train, y_train, X_test, y_test, feature_names=feature_names)
    rf_preds = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test) if hasattr(rf, 'predict_proba') else None

    results['rf'] = {
        'training_metrics': rf_metrics,
        'test_accuracy': accuracy_score(y_test, rf_preds),
        'test_f1_weighted': f1_score(y_test, rf_preds, average='weighted'),
        'classification_report': classification_report(y_test, rf_preds, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, rf_preds).tolist()
    }

    rf_save_path = os.path.join(out_dir, 'rf_model.joblib')
    rf.save_model(rf_save_path)
    print(f"Saved RF model -> {rf_save_path}")

    # SVM
    print("Training SVM (with internal scaling)...")
    svm_metrics = svm.train(X_train, y_train, X_test, y_test, feature_names=feature_names)
    svm_preds = svm.predict(X_test)

    results['svm'] = {
        'training_metrics': svm_metrics,
        'test_accuracy': accuracy_score(y_test, svm_preds),
        'test_f1_weighted': f1_score(y_test, svm_preds, average='weighted'),
        'classification_report': classification_report(y_test, svm_preds, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, svm_preds).tolist()
    }

    svm_save_path = os.path.join(out_dir, 'svm_model.joblib')
    svm.save_model(svm_save_path)
    print(f"Saved SVM model -> {svm_save_path}")

    # Isolation Forest
    print("Training Isolation Forest (unsupervised, using training features only)...")
    # Train IF on normal samples only if labels available (convert labels to binary logic)
    try:
        if hasattr(if_model, 'train'):
            if_metrics = if_model.train(X_train, y_train, X_test, y_test, feature_names=feature_names)
        else:
            if_model.model.fit(X_train)
            if_metrics = {}
    except Exception as e:
        print(f"Isolation Forest training (supervised conversion) failed: {e}; falling back to unsupervised fit on X_train")
        if_model.model.fit(X_train)
        if_model.is_trained = True
        if_metrics = {}

    # For reporting, map IF predictions back to labels consistent with y_test for scoring
    if_preds_raw = if_model.predict(X_test)
    # IsolationForest predicts 1 for normal, -1 for anomaly; map to class labels
    # Determine mapping: find majority label in training set for normal (1)
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    # Heuristic: assume label not 'normal' is jamming; map -1 -> jamming label if present else first non-normal
    jamming_label_candidates = [lab for lab in np.unique(y) if lab != 'normal']
    if len(jamming_label_candidates) > 0:
        jamming_label = jamming_label_candidates[0]
    else:
        jamming_label = 'jamming'

    mapped_if_preds = np.array([ 'normal' if p == 1 else jamming_label for p in if_preds_raw ])

    results['if'] = {
        'training_metrics': if_metrics,
        'test_accuracy': accuracy_score(np.where(y_test=='normal', 'normal', jamming_label), mapped_if_preds),
        'test_f1_weighted': f1_score(np.where(y_test=='normal', 'normal', jamming_label), mapped_if_preds, average='weighted'),
        'confusion_matrix': confusion_matrix(np.where(y_test=='normal', 'normal', jamming_label), mapped_if_preds).tolist()
    }

    if_save_path = os.path.join(out_dir, 'isolation_forest_model.joblib')
    if_model.save_model(if_save_path)
    print(f"Saved Isolation Forest model -> {if_save_path}")

    # Save combined results
    results_path = os.path.join(out_dir, 'classical_models_results.json')
    def _to_builtin(obj):
        """Recursively convert numpy types to Python built-ins for JSON serialization."""
        import numpy as _np

        if isinstance(obj, dict):
            return { _to_builtin(k): _to_builtin(v) for k, v in obj.items() }
        if isinstance(obj, list):
            return [ _to_builtin(v) for v in obj ]
        if isinstance(obj, tuple):
            return tuple(_to_builtin(v) for v in obj)
        if isinstance(obj, _np.ndarray):
            return _to_builtin(obj.tolist())
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if obj is None:
            return None
        return obj

    with open(results_path, 'w') as fh:
        json.dump(_to_builtin(results), fh, indent=2)

    print(f"Saved results JSON -> {results_path}")

    # Plot simple bar chart of F1 scores
    labels = ['RandomForest', 'SVM', 'IsolationForest']
    f1s = [results['rf']['test_f1_weighted'], results['svm']['test_f1_weighted'], results['if']['test_f1_weighted']]

    plt.figure(figsize=(6,4))
    bars = plt.bar(labels, f1s, color=['tab:blue','tab:orange','tab:green'])
    plt.ylim(0,1)
    plt.ylabel('F1 (weighted)')
    plt.title('Classical model test F1')
    for bar, val in zip(bars, f1s):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.3f}", ha='center')

    fig_path = os.path.join(figures_dir, 'classical_models_f1.png')
    plt.tight_layout()
    plt.savefig(fig_path)
    print(f"Saved figure -> {fig_path}")

    print("Summary:")
    for k, v in results.items():
        print(f"  {k}: accuracy={v['test_accuracy']:.3f}, f1={v['test_f1_weighted']:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=False, help='Path to labeled CSV with a "label" column')
    parser.add_argument('--normal-file', help='Path to raw normal metrics text file (best-effort parse)')
    parser.add_argument('--power-file', help='Path to raw power-jammer metrics text file (best-effort parse)')
    parser.add_argument('--out', default='models/classical_models', help='Output directory for models and results')
    parser.add_argument('--test-size', type=float, default=TRAINING_CONFIG.get('test_size', 0.2))
    parser.add_argument('--random-state', type=int, default=TRAINING_CONFIG.get('random_state', 42))
    args = parser.parse_args()

    # If raw normal/power files provided, build a combined CSV-like dataset in memory
    if args.normal_file and args.power_file:
        print(f"Parsing raw normal file: {args.normal_file}")
        normal_rows = parse_raw_metrics_file(args.normal_file)
        print(f"Parsed {len(normal_rows)} normal rows")

        print(f"Parsing raw power file: {args.power_file}")
        power_rows = parse_raw_metrics_file(args.power_file)
        print(f"Parsed {len(power_rows)} power rows")

        X = np.vstack([normal_rows, power_rows])
        y = np.array(['normal'] * len(normal_rows) + ['power_jamming'] * len(power_rows))
        # write a temporary CSV so existing pipeline can reuse feature names
        temp_csv = os.path.join('tools', 'temp_combined_dataset.csv')
        df = pd.DataFrame(X, columns=[f'f{i+1}' for i in range(X.shape[1])])
        df['label'] = y
        df.to_csv(temp_csv, index=False)

        train_and_evaluate(temp_csv, args.out, args.test_size, args.random_state)
    else:
        if not args.csv:
            raise ValueError('Please provide --csv or both --normal-file and --power-file')
        train_and_evaluate(args.csv, args.out, args.test_size, args.random_state)
