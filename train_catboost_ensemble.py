#!/usr/bin/env python3
"""
High-Accuracy CatBoost Ensemble for USRP Jamming Detection
Implements state-of-the-art gradient boosting ensemble with CatBoost
Target: >99.75% detection accuracy for power jamming

Algorithms:
- CatBoost (55% weight) - Superior gradient boosting
- LightGBM (30% weight) - Fast gradient boosting  
- Extra Trees (15% weight) - Randomized decision trees

This is an advanced version that replaces Random Forest/SVM with
industry-leading gradient boosting algorithms for maximum accuracy.
"""

import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Advanced ML imports
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not installed. Install with: pip install catboost")

try:
    from lightgbm import LGBMClassifier
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not installed. Install with: pip install lightgbm")

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, matthews_corrcoef)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns


class HighAccuracyCatBoostEnsemble:
    """
    Advanced ensemble using CatBoost, LightGBM, and Extra Trees
    Designed for >99.75% power jamming detection accuracy
    """
    
    def __init__(self, target_accuracy: float = 0.9975):
        self.target_accuracy = target_accuracy
        self.models = {}
        self.weights = {
            'catboost': 0.55,    # Primary model - best performance
            'lightgbm': 0.30,    # Fast and accurate
            'extratrees': 0.15   # Diversity and robustness
        }
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_names = None
        
        print(f"High-Accuracy CatBoost Ensemble")
        print(f"Target Accuracy: {target_accuracy*100:.2f}%")
        print(f"Ensemble Weights: {self.weights}")
        
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize optimized models for maximum accuracy"""
        
        models = {}
        
        # CatBoost - State-of-the-art gradient boosting
        if CATBOOST_AVAILABLE:
            models['catboost'] = CatBoostClassifier(
                iterations=1000,              # Reduced for stability
                learning_rate=0.1,           # Standard learning rate
                depth=6,                     # Moderate depth
                random_seed=42,
                verbose=False,               # Silent training
                thread_count=-1             # Use all cores
            )
        else:
            print("Warning: CatBoost not available, using fallback model")
            from sklearn.ensemble import GradientBoostingClassifier
            models['catboost'] = GradientBoostingClassifier(
                n_estimators=1000, learning_rate=0.05, max_depth=8, random_state=42
            )
        
        # LightGBM - Fast and accurate gradient boosting
        if LIGHTGBM_AVAILABLE:
            models['lightgbm'] = LGBMClassifier(
                n_estimators=1500,           # High estimator count
                learning_rate=0.05,          # Conservative learning rate
                max_depth=10,                # Deep trees
                num_leaves=128,              # Many leaves for expressiveness
                subsample=0.8,               # Subsample for regularization
                colsample_bytree=0.8,        # Feature subsampling
                reg_alpha=0.1,               # L1 regularization
                reg_lambda=0.1,              # L2 regularization
                min_child_samples=20,        # Minimum samples per leaf
                random_state=42,
                n_jobs=-1,                   # Use all cores
                class_weight='balanced',     # Handle class imbalance
                objective='multiclass',
                metric='multi_logloss',
                verbosity=-1                 # Silent
            )
        else:
            print("Warning: LightGBM not available, using fallback model")
            from sklearn.ensemble import GradientBoostingClassifier
            models['lightgbm'] = GradientBoostingClassifier(
                n_estimators=800, learning_rate=0.05, max_depth=10, random_state=42
            )
        
        # Extra Trees - Extremely randomized trees for diversity
        models['extratrees'] = ExtraTreesClassifier(
            n_estimators=1000,           # Many trees
            max_depth=15,                # Deep trees
            min_samples_split=2,         # Minimum split samples
            min_samples_leaf=1,          # Minimum leaf samples
            max_features='sqrt',         # Feature randomization
            bootstrap=True,              # Bootstrap sampling
            random_state=42,
            n_jobs=-1,                   # Use all cores
            class_weight='balanced'      # Handle class imbalance
        )
        
        return models
    
    def _compute_advanced_class_weights(self, y: np.ndarray) -> Dict[str, float]:
        """Compute advanced class weights for imbalanced data"""
        
        unique_classes = np.unique(y)
        n_samples = len(y)
        n_classes = len(unique_classes)
        
        # Compute class frequencies
        class_counts = {cls: np.sum(y == cls) for cls in unique_classes}
        
        # Advanced weighting strategy
        weights = {}
        for cls in unique_classes:
            # Inverse frequency with smoothing
            base_weight = n_samples / (n_classes * class_counts[cls])
            
            # Boost power jamming detection (critical scenario)
            if cls == 'power_jamming':
                boost_factor = 1.5  # 50% boost for power jamming
            elif cls == 'normal':
                boost_factor = 0.8  # Slight reduction for normal
            else:
                boost_factor = 1.0  # Standard weight for other attacks
            
            weights[cls] = base_weight * boost_factor
        
        print(f"Advanced class weights: {weights}")
        return weights
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2) -> Dict[str, float]:
        """Train the high-accuracy ensemble"""
        
        print(f"\nTraining High-Accuracy CatBoost Ensemble")
        print("=" * 60)
        
        # Store feature names
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
            X = X.values
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=validation_split, 
            stratify=y_encoded, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Compute class weights
        class_weights = self._compute_advanced_class_weights(y)
        
        # Initialize models
        self.models = self._initialize_models()
        
        # Train each model
        training_scores = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name.upper()}...")
            
            # Special handling for different model types
            if name == 'catboost' and CATBOOST_AVAILABLE:
                # CatBoost - simple training
                model.fit(X_train_scaled, y_train)
            elif name == 'lightgbm' and LIGHTGBM_AVAILABLE:
                # LightGBM with validation set
                try:
                    model.fit(
                        X_train_scaled, y_train,
                        eval_set=[(X_val_scaled, y_val)],
                        callbacks=[lgb.early_stopping(50, verbose=False)]
                    )
                except:
                    # Fallback if callbacks don't work
                    model.fit(X_train_scaled, y_train)
            else:
                # Standard sklearn interface
                model.fit(X_train_scaled, y_train)
            
            # Evaluate on validation set
            y_pred = model.predict(X_val_scaled)
            accuracy = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='weighted')
            
            training_scores[name] = {
                'accuracy': accuracy,
                'f1_score': f1
            }
            
            print(f"  {name.upper()} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        # Set trained flag
        self.is_trained = True
        
        # Evaluate ensemble on validation set
        ensemble_pred = self._predict_ensemble(X_val_scaled)
        ensemble_accuracy = accuracy_score(y_val, ensemble_pred)
        ensemble_f1 = f1_score(y_val, ensemble_pred, average='weighted')
        
        print(f"\nEnsemble Performance:")
        print(f"  Accuracy: {ensemble_accuracy:.4f}")
        print(f"  F1-Score: {ensemble_f1:.4f}")
        
        # Check if target accuracy is met
        if ensemble_accuracy >= self.target_accuracy:
            print(f"✅ Target accuracy {self.target_accuracy:.4f} ACHIEVED!")
        else:
            print(f"⚠️  Target accuracy {self.target_accuracy:.4f} not yet reached")
        
        # Remove the redundant is_trained assignment
        
        return {
            'ensemble_accuracy': ensemble_accuracy,
            'ensemble_f1': ensemble_f1,
            'individual_scores': training_scores
        }
    
    def _predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions using weighted voting"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Get predictions from each model
        predictions = {}
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions[name] = pred
        
        # Weighted ensemble prediction
        ensemble_pred = np.zeros(len(X), dtype=int)
        
        for i in range(len(X)):
            weighted_votes = {}
            
            for name, pred in predictions.items():
                vote = pred[i]
                weight = self.weights[name]
                
                # Convert numpy arrays to scalars
                if hasattr(vote, 'item'):
                    vote = vote.item()
                
                if vote not in weighted_votes:
                    weighted_votes[vote] = 0
                weighted_votes[vote] += weight
            
            # Select class with highest weighted vote
            ensemble_pred[i] = max(weighted_votes, key=weighted_votes.get)
        
        return ensemble_pred
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data"""
        
        if hasattr(X, 'values'):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        predictions_encoded = self._predict_ensemble(X_scaled)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if hasattr(X, 'values'):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities from each model
        ensemble_proba = np.zeros((len(X), len(self.label_encoder.classes_)))
        
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_scaled)
                weight = self.weights[name]
                ensemble_proba += weight * proba
        
        return ensemble_proba
    
    def evaluate_comprehensive(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Comprehensive evaluation of the ensemble"""
        
        print(f"\nComprehensive Ensemble Evaluation")
        print("=" * 50)
        
        # Make predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Encode true labels for metric calculation
        y_test_encoded = self.label_encoder.transform(y_test)
        y_pred_encoded = self.label_encoder.transform(y_pred)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test_encoded, y_pred_encoded),
            'precision_macro': precision_score(y_test_encoded, y_pred_encoded, average='macro'),
            'recall_macro': recall_score(y_test_encoded, y_pred_encoded, average='macro'),
            'f1_macro': f1_score(y_test_encoded, y_pred_encoded, average='macro'),
            'f1_weighted': f1_score(y_test_encoded, y_pred_encoded, average='weighted'),
            'matthews_cc': matthews_corrcoef(y_test_encoded, y_pred_encoded)
        }
        
        # Class-specific metrics
        class_report = classification_report(
            y_test, y_pred, 
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        # Special focus on power jamming detection
        power_jamming_metrics = {}
        if 'power_jamming' in self.label_encoder.classes_:
            power_idx = list(self.label_encoder.classes_).index('power_jamming')
            power_jamming_metrics = {
                'power_jamming_precision': class_report['power_jamming']['precision'],
                'power_jamming_recall': class_report['power_jamming']['recall'],
                'power_jamming_f1': class_report['power_jamming']['f1-score']
            }
        
        # Print results
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
        print(f"Matthews Correlation: {metrics['matthews_cc']:.4f}")
        
        if power_jamming_metrics:
            print(f"\nPower Jamming Detection:")
            print(f"  Precision: {power_jamming_metrics['power_jamming_precision']:.4f}")
            print(f"  Recall: {power_jamming_metrics['power_jamming_recall']:.4f}")
            print(f"  F1-Score: {power_jamming_metrics['power_jamming_f1']:.4f}")
            
            if power_jamming_metrics['power_jamming_f1'] >= self.target_accuracy:
                print(f"✅ Power jamming detection target ACHIEVED!")
            else:
                print(f"⚠️  Power jamming detection target not reached")
        
        # Confusion matrix
        cm = confusion_matrix(y_test_encoded, y_pred_encoded)
        
        return {
            'overall_metrics': metrics,
            'class_report': class_report,
            'power_jamming_metrics': power_jamming_metrics,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_proba
        }
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from all models"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        importance_dict = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[name] = model.feature_importances_
            elif hasattr(model, 'get_feature_importance'):  # CatBoost
                importance_dict[name] = model.get_feature_importance()
        
        return importance_dict
    
    def save_model(self, filepath: str):
        """Save the trained ensemble model"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'models': self.models,
            'weights': self.weights,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'target_accuracy': self.target_accuracy,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"High-accuracy ensemble saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained ensemble model"""
        
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.weights = model_data['weights']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.target_accuracy = model_data['target_accuracy']
        self.is_trained = model_data['is_trained']
        
        print(f"High-accuracy ensemble loaded from: {filepath}")


def train_high_accuracy_model(normal_path: str, jamming_path: str, 
                            model_save_path: str = "saved_models/catboost_ensemble.joblib") -> Dict[str, Any]:
    """Train the high-accuracy CatBoost ensemble"""
    
    print(f"Training High-Accuracy CatBoost Ensemble")
    print("=" * 60)
    
    # Load data
    print("Loading realistic USRP dataset...")
    normal_df = pd.read_csv(normal_path)
    jamming_df = pd.read_csv(jamming_path)
    
    # Combine datasets
    df = pd.concat([normal_df, jamming_df], ignore_index=True)
    
    # Prepare features and labels
    feature_cols = [col for col in df.columns if col not in 
                   ['scenario', 'binary_label', 'timestamp', 'attack_type']]
    
    X = df[feature_cols]
    y = df['scenario']
    
    print(f"Dataset loaded: {len(X)} samples, {len(feature_cols)} features")
    print(f"Classes: {y.value_counts().to_dict()}")
    
    # Initialize and train ensemble
    ensemble = HighAccuracyCatBoostEnsemble(target_accuracy=0.9975)
    training_results = ensemble.train(X, y, validation_split=0.2)
    
    # Create test set for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Comprehensive evaluation
    eval_results = ensemble.evaluate_comprehensive(X_test, y_test)
    
    # Save model
    import os
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    ensemble.save_model(model_save_path)
    
    # Feature importance analysis
    importance = ensemble.get_feature_importance()
    
    return {
        'ensemble': ensemble,
        'training_results': training_results,
        'evaluation_results': eval_results,
        'feature_importance': importance,
        'model_path': model_save_path
    }


def main():
    """Main execution function"""
    
    # Check if dataset exists
    dataset_dir = "Ensemble_ML_Jamming_detection_dataset/realistic_dataset"
    normal_path = f"{dataset_dir}/normal_traffic.csv"
    jamming_path = f"{dataset_dir}/jamming_attacks.csv"
    
    if not os.path.exists(normal_path) or not os.path.exists(jamming_path):
        print("Dataset not found. Please run generate_realistic_usrp_dataset.py first")
        return
    
    # Train high-accuracy model
    results = train_high_accuracy_model(normal_path, jamming_path)
    
    print(f"\nHigh-Accuracy CatBoost Ensemble Training Complete!")
    print(f"Target: >99.75% power jamming detection accuracy")
    print(f"Model saved: {results['model_path']}")


if __name__ == "__main__":
    main()
