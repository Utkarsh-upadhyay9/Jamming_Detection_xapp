import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(__file__))

def simple_test():
    print("🧪 Simple Ensemble Model Test with Realistic Data")
    print("=" * 55)
    
    normal_path = "Ensemble_ML_Jamming_detection_dataset/realistic_dataset/normal_traffic.csv"
    jamming_path = "Ensemble_ML_Jamming_detection_dataset/realistic_dataset/jamming_attacks.csv"
    
    if not os.path.exists(normal_path) or not os.path.exists(jamming_path):
        print("❌ Converted datasets not found. Please run convert_dataset_format.py first.")
        return False
    
    print("📊 Loading datasets...")
    normal_df = pd.read_csv(normal_path)
    jamming_df = pd.read_csv(jamming_path)
    
    print(f"✅ Loaded {len(normal_df)} normal samples and {len(jamming_df)} jamming samples")
    
    print("\n📈 Dataset Overview:")
    print(f"Normal traffic labels: {normal_df['label'].unique()}")
    print(f"Jamming attack labels: {jamming_df['label'].unique()}")
    
    print(f"\nColumns in normal data: {len(normal_df.columns)}")
    print(f"Columns in jamming data: {len(jamming_df.columns)}")
    print(f"Sample columns: {list(normal_df.columns[:5])}")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, f1_score, accuracy_score
    
    print("\n🚀 Testing with Random Forest...")
    
    X_normal = normal_df.drop(['label'], axis=1).values
    y_normal = normal_df['label'].values
    
    X_jamming = jamming_df.drop(['label'], axis=1).values
    y_jamming = jamming_df['label'].values
    
    X = np.vstack([X_normal, X_jamming])
    y = np.hstack([y_normal, y_jamming])
    
    print(f"Combined dataset shape: {X.shape}")
    print(f"Labels distribution: {np.bincount(y)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n📊 Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    print(f"\n📋 Classification Report:")
    target_names = ['Normal', 'Power Jamming', 'Sweep Jamming', 'Intelligent Jamming']
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    if accuracy > 0.8:  # Basic sanity check
        print("\n🎯 Testing Ensemble Model...")
        try:
            from src.ensemble_model import EnsembleJammingDetector
            
            detector = EnsembleJammingDetector()
            
            metrics = detector.train_ensemble(normal_path, jamming_path)
            
            print("✅ Ensemble training successful!")
            print(f"Ensemble F1-Score: {metrics['ensemble']['f1_score']:.4f}")
            print(f"Ensemble Accuracy: {metrics['ensemble']['accuracy']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Ensemble training failed: {e}")
            print("But sklearn RandomForest worked, so the data format is correct.")
            return False
    
    return accuracy > 0.8

if __name__ == "__main__":
    success = simple_test()
    print(f"\n{'='*55}")
    if success:
        print("✅ Test completed successfully!")
    else:
        print("❌ Test failed. Check the data format.")
    
    sys.exit(0 if success else 1)
