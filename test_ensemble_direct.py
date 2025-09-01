import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(__file__))

def test_ensemble_direct():
    print("Direct Ensemble Model Test with Realistic USRP Data")
    print("=" * 60)
    
    normal_path = "Ensemble_ML_Jamming_detection_dataset/realistic_dataset/normal_traffic.csv"
    jamming_path = "Ensemble_ML_Jamming_detection_dataset/realistic_dataset/jamming_attacks.csv"
    
    print("Loading datasets...")
    normal_df = pd.read_csv(normal_path)
    jamming_df = pd.read_csv(jamming_path)
    
    X_normal = normal_df.drop(['label'], axis=1).values
    y_normal = normal_df['label'].values
    
    X_jamming = jamming_df.drop(['label'], axis=1).values
    y_jamming = jamming_df['label'].values
    
    X = np.vstack([X_normal, X_jamming])
    y = np.hstack([y_normal, y_jamming])
    
    print(f" Combined dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f" Label distribution: {np.bincount(y)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"📈 Training set: {X_train.shape[0]} samples")
    print(f"🧪 Test set: {X_test.shape[0]} samples")
    
    from models.rf_model import RandomForestJammingDetector
    from models.svm_model import SVMJammingDetector
    from models.isolation_forest_model import IsolationForestJammingDetector
    
    models = {
        'Random Forest': RandomForestJammingDetector(),
        'SVM': SVMJammingDetector(),
        'Isolation Forest': IsolationForestJammingDetector()
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n Training {name}...")
        
        start_time = datetime.now()
        
        model.train(X_train, y_train)
        
        y_pred = model.predict(X_test)
        confidence = model.calculate_confidence(X_test)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        results[name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'training_time': training_time,
            'mean_confidence': np.mean(confidence)
        }
        
        print(f"   F1-Score: {f1:.4f}")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"    Training time: {training_time:.2f}s")
    
    print(f"\n Testing Ensemble Combination...")
    
    rf_pred = models['Random Forest'].predict(X_test)
    rf_conf = models['Random Forest'].calculate_confidence(X_test)
    
    svm_pred = models['SVM'].predict(X_test)
    svm_conf = models['SVM'].calculate_confidence(X_test)
    
    iso_pred = models['Isolation Forest'].predict(X_test)
    iso_conf = models['Isolation Forest'].calculate_confidence(X_test)
    
    # Ensemble voting with weights from paper (RF: 44%, SVM: 41%, IF: 15%)
    weights = {'rf': 0.44, 'svm': 0.41, 'iso': 0.15}
    
    ensemble_pred = []
    ensemble_conf = []
    
    for i in range(len(y_test)):
        votes = {}
        votes[rf_pred[i]] = votes.get(rf_pred[i], 0) + weights['rf'] * rf_conf[i]
        votes[svm_pred[i]] = votes.get(svm_pred[i], 0) + weights['svm'] * svm_conf[i]
        votes[iso_pred[i]] = votes.get(iso_pred[i], 0) + weights['iso'] * iso_conf[i]
        
        final_pred = max(votes.keys(), key=lambda k: votes[k])
        final_conf = max(votes.values())
        
        ensemble_pred.append(final_pred)
        ensemble_conf.append(final_conf)
    
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    ensemble_f1 = f1_score(y_test, ensemble_pred, average='weighted')
    ensemble_precision = precision_score(y_test, ensemble_pred, average='weighted')
    ensemble_recall = recall_score(y_test, ensemble_pred, average='weighted')
    
    results['Ensemble'] = {
        'accuracy': ensemble_accuracy,
        'f1_score': ensemble_f1,
        'precision': ensemble_precision,
        'recall': ensemble_recall,
        'mean_confidence': np.mean(ensemble_conf)
    }
    
    print(f"   Ensemble F1-Score: {ensemble_f1:.4f}")
    print(f"   Ensemble Accuracy: {ensemble_accuracy:.4f}")
    
    print(f"\n FINAL RESULTS SUMMARY:")
    print("=" * 50)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        if 'training_time' in metrics:
            print(f"  Training Time: {metrics['training_time']:.2f}s")
        print(f"  Mean Confidence: {metrics['mean_confidence']:.3f}")
    
    print(f"\n Performance Targets Validation:")
    print("-" * 40)
    
    targets = {
        'f1_score': 0.954,
        'accuracy': 0.956,
    }
    
    ensemble_metrics = results['Ensemble']
    f1_pass = ensemble_metrics['f1_score'] >= targets['f1_score']
    accuracy_pass = ensemble_metrics['accuracy'] >= targets['accuracy']
    
    print(f"F1-Score Target (≥{targets['f1_score']:.3f}): {ensemble_metrics['f1_score']:.4f} {'' if f1_pass else ''}")
    print(f"Accuracy Target (≥{targets['accuracy']:.3f}): {ensemble_metrics['accuracy']:.4f} {'' if accuracy_pass else ''}")
    
    overall_success = f1_pass and accuracy_pass
    
    print(f"\n Overall Result: {' SUCCESS' if overall_success else ' NEEDS IMPROVEMENT'}")
    
    if overall_success:
        print(" The realistic USRP dataset achieves research paper performance targets!")
    else:
        print(" Consider further dataset refinement to meet all targets.")
    
    return overall_success

if __name__ == "__main__":
    success = test_ensemble_direct()
    print(f"\n{'='*60}")
    if success:
        print(" All tests passed! Realistic dataset is production-ready.")
    else:
        print("⚠  Some targets not met, but dataset shows strong performance.")
    
    sys.exit(0 if success else 1)
