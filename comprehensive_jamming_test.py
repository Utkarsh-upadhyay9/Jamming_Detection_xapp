#!/usr/bin/env python3
"""
Comprehensive Real-World Jamming Detection Test
===============================================

Tests all jamming types with real performance measurements:
- Normal traffic detection
- Power jamming detection  
- Sweep jamming detection
- Reactive jamming detection

No fabricated results - only actual model performance.
"""

import numpy as np
import pandas as pd
from high_accuracy_jamming_detection import HighAccuracyJammingDetector
import time
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import json

def load_real_dataset():
    """Load the actual realistic dataset"""
    
    dataset_dir = "Ensemble_ML_Jamming_detection_dataset/realistic_dataset"
    normal_path = f"{dataset_dir}/normal_traffic.csv"
    jamming_path = f"{dataset_dir}/jamming_attacks.csv"
    
    if not os.path.exists(normal_path) or not os.path.exists(jamming_path):
        raise FileNotFoundError("Dataset not found. Please run generate_realistic_usrp_dataset.py first")
    
    print("📊 Loading realistic USRP dataset...")
    normal_df = pd.read_csv(normal_path)
    jamming_df = pd.read_csv(jamming_path)
    
    # Combine datasets
    df = pd.concat([normal_df, jamming_df], ignore_index=True)
    
    # Prepare features and labels
    feature_cols = [col for col in df.columns if col not in 
                   ['scenario', 'binary_label', 'timestamp', 'attack_type']]
    
    X = df[feature_cols].values
    y = df['scenario'].values
    
    print(f"✅ Dataset loaded: {len(X)} samples, {len(feature_cols)} features")
    
    # Show class distribution
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print(f"📈 Class distribution: {class_dist}")
    
    return X, y, feature_cols

def test_all_jamming_types():
    """Test detection performance on all jamming types"""
    
    print("🎯 Comprehensive Jamming Detection Test")
    print("=" * 50)
    print("Testing ALL jamming types with REAL dataset")
    print()
    
    # Load real dataset
    X, y, feature_names = load_real_dataset()
    
    # Check if high-accuracy model exists
    model_path = "saved_models/high_accuracy_focused.joblib"
    if not os.path.exists(model_path):
        print(f"❌ High-accuracy model not found at: {model_path}")
        print("Please run: python3 quick_high_accuracy_test.py first")
        return None
    
    # Initialize detector with real model
    print("🔄 Loading high-accuracy CatBoost ensemble...")
    detector = HighAccuracyJammingDetector(model_path)
    print()
    
    # Split data for testing (use 30% for testing to get substantial results)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    print(f"🧪 Test set: {len(X_test)} samples")
    
    # Get class distribution in test set
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    test_dist = dict(zip(unique_test, counts_test))
    print(f"📊 Test set distribution: {test_dist}")
    print()
    
    # Make predictions on real test data
    print("🔍 Running detection on real test samples...")
    start_time = time.time()
    
    predictions = []
    detection_times = []
    
    # Process in batches for progress tracking
    batch_size = 100
    n_batches = len(X_test) // batch_size + (1 if len(X_test) % batch_size else 0)
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(X_test))
        
        if batch_idx % 10 == 0:
            print(f"  Progress: {end_idx}/{len(X_test)} samples")
        
        for i in range(start_idx, end_idx):
            batch_start = time.time()
            result = detector.detect_jamming(X_test[i])
            batch_time = (time.time() - batch_start) * 1000  # ms
            
            predictions.append(result['prediction'])
            detection_times.append(batch_time)
    
    total_time = time.time() - start_time
    avg_detection_time = np.mean(detection_times)
    
    print(f"✅ Detection completed in {total_time:.2f}s")
    print(f"⚡ Average detection time: {avg_detection_time:.2f}ms")
    print()
    
    # Calculate overall metrics
    overall_accuracy = accuracy_score(y_test, predictions)
    
    print("🎯 REAL PERFORMANCE RESULTS")
    print("=" * 40)
    print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print()
    
    # Per-class accuracy analysis
    results = {}
    
    for jamming_type in np.unique(y_test):
        # Get samples for this specific type
        type_mask = y_test == jamming_type
        type_predictions = np.array(predictions)[type_mask]
        type_labels = y_test[type_mask]
        
        # Calculate accuracy for this type
        type_accuracy = accuracy_score(type_labels, type_predictions)
        
        # Count correct predictions
        correct_count = np.sum(type_predictions == type_labels)
        total_count = len(type_labels)
        
        results[jamming_type] = {
            'accuracy': type_accuracy,
            'correct': correct_count,
            'total': total_count,
            'percentage': type_accuracy * 100
        }
        
        print(f"{jamming_type.upper()} Detection:")
        print(f"  Samples: {total_count}")
        print(f"  Correct: {correct_count}")
        print(f"  Accuracy: {type_accuracy:.4f} ({type_accuracy*100:.2f}%)")
        print()
    
    # Detailed classification report
    print("📊 DETAILED CLASSIFICATION REPORT")
    print("=" * 45)
    print(classification_report(y_test, predictions, digits=4))
    
    # Confusion matrix
    print("\n📈 CONFUSION MATRIX")
    print("=" * 25)
    cm = confusion_matrix(y_test, predictions)
    
    # Get unique labels in order
    labels = sorted(np.unique(y_test))
    
    # Print header
    print("Predicted:", end="")
    for label in labels:
        print(f"{label[:8]:>10}", end="")
    print()
    
    # Print matrix
    for i, true_label in enumerate(labels):
        print(f"{true_label[:8]:>10}", end="")
        for j in range(len(labels)):
            print(f"{cm[i,j]:>10}", end="")
        print()
    
    # Binary classification (jamming vs normal)
    binary_labels = ['normal' if label == 'normal' else 'jamming' for label in y_test]
    binary_predictions = ['normal' if pred == 'normal' else 'jamming' for pred in predictions]
    binary_accuracy = accuracy_score(binary_labels, binary_predictions)
    
    print(f"\n🔍 BINARY CLASSIFICATION (Jamming vs Normal)")
    print("=" * 50)
    print(f"Binary Accuracy: {binary_accuracy:.4f} ({binary_accuracy*100:.2f}%)")
    
    # Performance analysis
    print(f"\n⚡ PERFORMANCE METRICS")
    print("=" * 30)
    print(f"Average Detection Time: {avg_detection_time:.2f}ms")
    print(f"Total Test Samples: {len(X_test)}")
    print(f"Processing Rate: {len(X_test)/total_time:.1f} samples/second")
    
    # Check if any type meets >99% accuracy
    high_accuracy_types = []
    for jamming_type, metrics in results.items():
        if metrics['accuracy'] >= 0.99:
            high_accuracy_types.append(jamming_type)
    
    if high_accuracy_types:
        print(f"\n✅ HIGH ACCURACY ACHIEVED (>99%):")
        for jamming_type in high_accuracy_types:
            acc = results[jamming_type]['accuracy']
            print(f"  {jamming_type}: {acc*100:.2f}%")
    
    # Save detailed results
    detailed_results = {
        'overall_accuracy': float(overall_accuracy),
        'binary_accuracy': float(binary_accuracy),
        'avg_detection_time_ms': float(avg_detection_time),
        'total_samples': int(len(X_test)),
        'processing_rate_per_second': float(len(X_test)/total_time),
        'per_type_results': {
            jamming_type: {
                'accuracy': float(metrics['accuracy']),
                'correct': int(metrics['correct']),
                'total': int(metrics['total']),
                'percentage': float(metrics['percentage'])
            }
            for jamming_type, metrics in results.items()
        },
        'confusion_matrix': cm.tolist(),
        'class_labels': labels,
        'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    results_file = "comprehensive_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    return detailed_results

def run_focused_type_test(jamming_type: str, n_samples: int = 1000):
    """Run focused test on specific jamming type with fresh samples"""
    
    print(f"\n🎯 Focused {jamming_type.upper()} Test ({n_samples} samples)")
    print("=" * 50)
    
    # Load detector
    model_path = "saved_models/high_accuracy_focused.joblib"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
    
    detector = HighAccuracyJammingDetector(model_path)
    
    # Load real dataset and filter for specific type
    X, y, _ = load_real_dataset()
    
    # Get samples of specific type
    type_mask = y == jamming_type
    if not np.any(type_mask):
        print(f"❌ No samples found for type: {jamming_type}")
        return None
    
    X_type = X[type_mask]
    y_type = y[type_mask]
    
    # Sample if we have more than requested
    if len(X_type) > n_samples:
        indices = np.random.choice(len(X_type), n_samples, replace=False)
        X_type = X_type[indices]
        y_type = y_type[indices]
    
    print(f"Testing {len(X_type)} real {jamming_type} samples...")
    
    # Make predictions
    correct = 0
    times = []
    
    for i, features in enumerate(X_type):
        start_time = time.time()
        result = detector.detect_jamming(features)
        detection_time = (time.time() - start_time) * 1000
        
        times.append(detection_time)
        if result['prediction'] == jamming_type:
            correct += 1
        
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(X_type)}")
    
    accuracy = correct / len(X_type)
    avg_time = np.mean(times)
    
    print(f"\nResults for {jamming_type}:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Correct: {correct}/{len(X_type)}")
    print(f"  Avg Time: {avg_time:.2f}ms")
    
    return {
        'type': jamming_type,
        'accuracy': accuracy,
        'correct': correct,
        'total': len(X_type),
        'avg_time_ms': avg_time
    }

if __name__ == "__main__":
    # Run comprehensive test on all types
    comprehensive_results = test_all_jamming_types()
    
    if comprehensive_results:
        print(f"\n" + "="*60)
        print("🎯 FINAL COMPREHENSIVE SUMMARY")
        print("="*60)
        
        print(f"Overall System Accuracy: {comprehensive_results['overall_accuracy']*100:.2f}%")
        print(f"Binary Detection Accuracy: {comprehensive_results['binary_accuracy']*100:.2f}%")
        print(f"Average Detection Speed: {comprehensive_results['avg_detection_time_ms']:.2f}ms")
        
        print(f"\nPer-Type Real Performance:")
        for jamming_type, metrics in comprehensive_results['per_type_results'].items():
            print(f"  {jamming_type:15}: {metrics['percentage']:6.2f}% ({metrics['correct']}/{metrics['total']})")
        
        # Check if power jamming meets target
        if 'power_jamming' in comprehensive_results['per_type_results']:
            power_acc = comprehensive_results['per_type_results']['power_jamming']['accuracy']
            if power_acc >= 0.9975:
                print(f"\n✅ Power jamming target >99.75% ACHIEVED: {power_acc*100:.2f}%")
            else:
                print(f"\n⚠️  Power jamming target not met: {power_acc*100:.2f}% < 99.75%")
        
        print(f"\nNo fabricated results - all measurements from real model predictions.")
