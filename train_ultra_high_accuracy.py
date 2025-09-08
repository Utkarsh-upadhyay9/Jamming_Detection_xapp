#!/usr/bin/env python3
"""
Ultra-High Accuracy Jamming Detection Training
==============================================

Target Accuracies:
- Power jamming detection: >99.75%
- Sweep jamming detection: ≥98%
- Reactive jamming detection: ≥95%

This script uses enhanced feature engineering and advanced ensemble methods.
"""

import numpy as np
import pandas as pd
import time
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from train_catboost_ensemble import HighAccuracyCatBoostEnsemble
from ultra_high_accuracy_features import generate_ultra_high_accuracy_dataset
import json

class UltraHighAccuracyEnsemble(HighAccuracyCatBoostEnsemble):
    """Enhanced ensemble specifically for ultra-high accuracy targets"""
    
    def __init__(self, target_accuracies: dict = None):
        if target_accuracies is None:
            target_accuracies = {
                'power_jamming': 0.9975,
                'sweep_jamming': 0.98,
                'reactive_jamming': 0.95,
                'normal': 0.99
            }
        
        super().__init__(target_accuracy=0.9975)
        self.target_accuracies = target_accuracies
        
        # Enhanced ensemble weights for better performance
        self.weights = {
            'catboost': 0.6,     # Increased weight for best performer
            'lightgbm': 0.25,    # Reduced weight
            'extratrees': 0.15   # Base weight for diversity
        }
        
        print(f"Ultra-High Accuracy Ensemble")
        print(f"Target Accuracies: {target_accuracies}")
        print(f"Enhanced Weights: {self.weights}")
    
    def _initialize_models(self):
        """Initialize models with enhanced hyperparameters for ultra-high accuracy"""
        
        models = {}
        
        # Enhanced CatBoost for maximum accuracy
        try:
            from catboost import CatBoostClassifier
            models['catboost'] = CatBoostClassifier(
                iterations=2000,             # More iterations
                learning_rate=0.08,          # Slightly lower learning rate
                depth=8,                     # Deeper trees
                l2_leaf_reg=3,               # L2 regularization
                border_count=254,            # More borders for better splits
                random_seed=42,
                verbose=False,
                thread_count=-1,
                auto_class_weights='Balanced'  # Enhanced class weighting
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            models['catboost'] = GradientBoostingClassifier(
                n_estimators=2000, learning_rate=0.05, max_depth=10, random_state=42
            )
        
        # Enhanced LightGBM
        try:
            from lightgbm import LGBMClassifier
            models['lightgbm'] = LGBMClassifier(
                n_estimators=2000,           # More estimators
                learning_rate=0.04,          # Lower learning rate
                max_depth=12,                # Deeper trees
                num_leaves=256,              # More leaves
                subsample=0.85,              # Higher subsample
                colsample_bytree=0.85,       # Higher column sample
                reg_alpha=0.2,               # Higher L1 regularization
                reg_lambda=0.2,              # Higher L2 regularization
                min_child_samples=15,        # Lower minimum samples
                random_state=42,
                n_jobs=-1,
                class_weight='balanced',
                objective='multiclass',
                metric='multi_logloss',
                verbosity=-1,
                min_split_gain=0.1,          # Minimum split gain
                min_child_weight=0.001       # Minimum child weight
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            models['lightgbm'] = GradientBoostingClassifier(
                n_estimators=1500, learning_rate=0.04, max_depth=12, random_state=42
            )
        
        # Enhanced Extra Trees
        from sklearn.ensemble import ExtraTreesClassifier
        models['extratrees'] = ExtraTreesClassifier(
            n_estimators=1500,           # More trees
            max_depth=20,                # Much deeper trees
            min_samples_split=2,         # Minimum split samples
            min_samples_leaf=1,          # Minimum leaf samples
            max_features='sqrt',         # Feature randomization
            bootstrap=True,              # Bootstrap sampling
            random_state=42,
            n_jobs=-1,
            class_weight='balanced_subsample',  # Enhanced class weighting
            criterion='gini',            # Gini impurity
            min_impurity_decrease=0.0001 # Minimum impurity decrease
        )
        
        return models
    
    def _compute_advanced_class_weights(self, y):
        """Compute ultra-enhanced class weights for target accuracies"""
        
        unique_classes = np.unique(y)
        n_samples = len(y)
        n_classes = len(unique_classes)
        
        class_counts = {cls: np.sum(y == cls) for cls in unique_classes}
        
        weights = {}
        for cls in unique_classes:
            # Base inverse frequency weight
            base_weight = n_samples / (n_classes * class_counts[cls])
            
            # Target-based boost factors
            if cls == 'power_jamming':
                boost_factor = 2.5  # Massive boost for 99.75% target
            elif cls == 'sweep_jamming':
                boost_factor = 2.0  # High boost for 98% target
            elif cls == 'reactive_jamming':
                boost_factor = 1.5  # Moderate boost for 95% target
            elif cls == 'normal':
                boost_factor = 0.7  # Slight reduction to prevent over-fitting
            else:
                boost_factor = 1.0
            
            weights[cls] = base_weight * boost_factor
        
        print(f"Ultra-enhanced class weights: {weights}")
        return weights
    
    def evaluate_target_accuracies(self, X_test, y_test):
        """Evaluate if target accuracies are met for each jamming type"""
        
        predictions = self.predict(X_test)
        
        results = {}
        targets_met = []
        
        for jamming_type in np.unique(y_test):
            type_mask = y_test == jamming_type
            type_predictions = predictions[type_mask]
            type_labels = y_test[type_mask]
            
            accuracy = accuracy_score(type_labels, type_predictions)
            target = self.target_accuracies.get(jamming_type, 0.9)
            
            results[jamming_type] = {
                'accuracy': accuracy,
                'target': target,
                'met': accuracy >= target,
                'samples': len(type_labels),
                'correct': np.sum(type_predictions == type_labels)
            }
            
            if accuracy >= target:
                targets_met.append(jamming_type)
        
        return results, targets_met

def train_ultra_high_accuracy_model():
    """Train model to achieve ultra-high accuracy targets"""
    
    print("🚀 Ultra-High Accuracy Jamming Detection Training")
    print("=" * 60)
    print("Targets: Power>99.75%, Sweep≥98%, Reactive≥95%")
    print()
    
    # Generate ultra-high accuracy dataset
    print("📊 Generating ultra-high accuracy dataset...")
    X, y, feature_names = generate_ultra_high_accuracy_dataset(n_samples=40000)
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print(f"Distribution: {class_dist}")
    print()
    
    # Split data with larger test set for reliable accuracy measurement
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    
    print(f"Training: {len(X_train)} samples")
    print(f"Testing: {len(X_test)} samples")
    print()
    
    # Initialize ultra-high accuracy ensemble
    ensemble = UltraHighAccuracyEnsemble()
    
    # Train with enhanced parameters
    print("🔄 Training ultra-high accuracy ensemble...")
    start_time = time.time()
    
    training_results = ensemble.train(X_train, y_train, validation_split=0.15)
    
    training_time = time.time() - start_time
    print(f"⏱️  Training completed in {training_time:.2f} seconds")
    print()
    
    # Test on holdout set
    print("🧪 Testing on holdout test set...")
    test_start = time.time()
    
    predictions = ensemble.predict(X_test)
    probabilities = ensemble.predict_proba(X_test)
    
    test_time = time.time() - test_start
    avg_prediction_time = (test_time / len(X_test)) * 1000
    
    print(f"⏱️  Testing completed in {test_time:.2f} seconds")
    print(f"⚡ Average prediction time: {avg_prediction_time:.3f}ms")
    print()
    
    # Overall performance
    overall_accuracy = accuracy_score(y_test, predictions)
    print(f"🎯 OVERALL ACCURACY: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print()
    
    # Target accuracy evaluation
    target_results, targets_met = ensemble.evaluate_target_accuracies(X_test, y_test)
    
    print("🎯 TARGET ACCURACY RESULTS")
    print("=" * 40)
    
    all_targets_met = True
    for jamming_type, metrics in target_results.items():
        status = "✅" if metrics['met'] else "❌"
        print(f"{status} {jamming_type.upper()}: {metrics['accuracy']*100:.2f}% "
              f"(Target: {metrics['target']*100:.1f}%, "
              f"Samples: {metrics['correct']}/{metrics['samples']})")
        
        if not metrics['met']:
            all_targets_met = False
    
    print()
    if all_targets_met:
        print("🎉 ALL TARGET ACCURACIES ACHIEVED!")
    else:
        print("⚠️  Some targets not met - may need further tuning")
    
    print()
    
    # Detailed classification report
    print("📊 DETAILED CLASSIFICATION REPORT")
    print("=" * 45)
    print(classification_report(y_test, predictions, digits=4))
    
    # Confusion matrix
    print("\n📈 CONFUSION MATRIX")
    print("=" * 25)
    cm = confusion_matrix(y_test, predictions)
    labels = sorted(np.unique(y_test))
    
    print("Predicted:", end="")
    for label in labels:
        print(f"{label[:8]:>10}", end="")
    print()
    
    for i, true_label in enumerate(labels):
        print(f"{true_label[:8]:>10}", end="")
        for j in range(len(labels)):
            print(f"{cm[i,j]:>10}", end="")
        print()
    
    # Binary classification
    binary_labels = ['normal' if label == 'normal' else 'jamming' for label in y_test]
    binary_predictions = ['normal' if pred == 'normal' else 'jamming' for pred in predictions]
    binary_accuracy = accuracy_score(binary_labels, binary_predictions)
    
    print(f"\n🔍 BINARY CLASSIFICATION")
    print("=" * 30)
    print(f"Jamming vs Normal: {binary_accuracy:.4f} ({binary_accuracy*100:.2f}%)")
    
    # Save ultra-high accuracy model
    model_path = "saved_models/ultra_high_accuracy_ensemble.joblib"
    os.makedirs("saved_models", exist_ok=True)
    ensemble.save_model(model_path)
    print(f"\n💾 Ultra-high accuracy model saved: {model_path}")
    
    # Save detailed results
    results = {
        'dataset_info': {
            'total_samples': len(X),
            'enhanced_features': X.shape[1],
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'class_distribution': class_dist
        },
        'performance_metrics': {
            'overall_accuracy': float(overall_accuracy),
            'binary_accuracy': float(binary_accuracy),
            'training_time_seconds': float(training_time),
            'avg_prediction_time_ms': float(avg_prediction_time)
        },
        'target_results': {
            jamming_type: {
                'accuracy': float(metrics['accuracy']),
                'target': float(metrics['target']),
                'met': bool(metrics['met']),
                'correct': int(metrics['correct']),
                'total': int(metrics['samples'])
            }
            for jamming_type, metrics in target_results.items()
        },
        'targets_achieved': targets_met,
        'all_targets_met': all_targets_met,
        'confusion_matrix': cm.tolist(),
        'class_labels': labels,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    results_file = "ultra_high_accuracy_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved: {results_file}")
    
    return results

def quick_validation_test():
    """Quick validation with smaller dataset to verify improvements"""
    
    print("\n" + "="*50)
    print("🔬 QUICK VALIDATION TEST")
    print("="*50)
    
    # Generate smaller test dataset
    X, y, _ = generate_ultra_high_accuracy_dataset(n_samples=5000)
    
    # Load the ultra-high accuracy model
    model_path = "saved_models/ultra_high_accuracy_ensemble.joblib"
    if not os.path.exists(model_path):
        print("❌ Ultra-high accuracy model not found. Train first.")
        return
    
    ensemble = UltraHighAccuracyEnsemble()
    ensemble.load_model(model_path)
    
    # Quick test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=123
    )
    
    predictions = ensemble.predict(X_test)
    target_results, targets_met = ensemble.evaluate_target_accuracies(X_test, y_test)
    
    print("Quick Validation Results:")
    for jamming_type, metrics in target_results.items():
        status = "✅" if metrics['met'] else "❌"
        print(f"{status} {jamming_type}: {metrics['accuracy']*100:.2f}% "
              f"(Target: {metrics['target']*100:.1f}%)")
    
    print(f"\nTargets met: {len(targets_met)}/4")

if __name__ == "__main__":
    # Train ultra-high accuracy model
    results = train_ultra_high_accuracy_model()
    
    # Quick validation
    if results and results.get('all_targets_met', False):
        quick_validation_test()
    
    print(f"\n" + "="*60)
    print("🎯 ULTRA-HIGH ACCURACY TRAINING COMPLETE")
    print("="*60)
    
    if results:
        print(f"Overall Accuracy: {results['performance_metrics']['overall_accuracy']*100:.2f}%")
        print(f"Training Time: {results['performance_metrics']['training_time_seconds']:.1f}s")
        print(f"Prediction Speed: {results['performance_metrics']['avg_prediction_time_ms']:.3f}ms")
        
        print(f"\nTarget Achievement:")
        for jamming_type, metrics in results['target_results'].items():
            status = "✅" if metrics['met'] else "❌"
            print(f"  {status} {jamming_type}: {metrics['accuracy']*100:.2f}%")
        
        if results['all_targets_met']:
            print(f"\n🎉 ALL ULTRA-HIGH ACCURACY TARGETS ACHIEVED!")
        else:
            print(f"\n⚠️  Continue tuning for remaining targets")
    
    print(f"\nModel saved: saved_models/ultra_high_accuracy_ensemble.joblib")
