import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, Tuple, Any, Optional
import joblib
import os

from config.model_config import RF_CONFIG, TRAINING_CONFIG

class RandomForestJammingDetector:
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or RF_CONFIG
        self.model = RandomForestClassifier(**self.config)
        self.is_trained = False
        self.feature_names = None
        self.class_names = None
        self.training_metrics = {}
        self.prediction_history = []
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              feature_names: Optional[list] = None) -> Dict[str, float]:
        self.feature_names = feature_names
        self.class_names = list(np.unique(y_train))
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        train_pred = self.model.predict(X_train)
        self.training_metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'train_f1_score': f1_score(y_train, train_pred, average='weighted')
        }
        
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            self.training_metrics.update({
                'val_accuracy': accuracy_score(y_val, val_pred),
                'val_f1_score': f1_score(y_val, val_pred, average='weighted')
            })
        
        return self.training_metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X)
        self.prediction_history.extend(predictions.tolist())
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        importance = self.model.feature_importances_
        
        if self.feature_names:
            return dict(zip(self.feature_names, importance))
        else:
            return {f'feature_{i}': imp for i, imp in enumerate(importance)}
    
    def calculate_confidence(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        
        # Calculate entropy-based confidence
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10), axis=1)
        max_entropy = np.log2(probabilities.shape[1])  # Maximum possible entropy
        
        # Normalize to [0, 1] where 1 is high confidence
        confidence = 1 - (entropy / max_entropy)
        
        return confidence
    
    def get_tree_predictions(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        tree_predictions = np.array([
            tree.predict(X) for tree in self.model.estimators_
        ]).T
        
        return tree_predictions
    
    def calculate_uncertainty(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        probabilities = self.predict_proba(X)
        tree_predictions = self.get_tree_predictions(X)
        
        # Entropy-based uncertainty
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10), axis=1)
        
        tree_variance = np.var(tree_predictions, axis=1)
        
        sorted_probs = np.sort(probabilities, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]
        
        return {
            'entropy': entropy,
            'tree_variance': tree_variance,
            'margin': margin,
            'confidence': 1 - (entropy / np.log2(probabilities.shape[1]))
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        if not self.is_trained:
            return {'is_trained': False}
        
        info = {
            'is_trained': True,
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'n_features': self.model.n_features_in_,
            'n_classes': self.model.n_classes_,
            'class_names': self.class_names,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'prediction_count': len(self.prediction_history)
        }
        
        if hasattr(self.model, 'oob_score_'):
            info['oob_score'] = self.model.oob_score_
        
        return info
    
    def save_model(self, filepath: str):
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'config': self.config,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'training_metrics': self.training_metrics,
            'is_trained': self.is_trained
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.config = model_data['config']
        self.feature_names = model_data['feature_names']
        self.class_names = model_data['class_names']
        self.training_metrics = model_data['training_metrics']
        self.is_trained = model_data['is_trained']
    
    def reset_prediction_history(self):
        self.prediction_history = []
    
    def get_decision_path(self, X: np.ndarray, tree_index: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_trained:
            raise ValueError("Model must be trained to get decision paths")
        
        if tree_index >= len(self.model.estimators_):
            raise ValueError(f"Tree index {tree_index} out of range")
        
        tree = self.model.estimators_[tree_index]
        return tree.decision_path(X)
    
    def explain_prediction(self, X: np.ndarray, sample_index: int = 0) -> Dict[str, Any]:
        if not self.is_trained:
            raise ValueError("Model must be trained to explain predictions")
        
        sample = X[sample_index:sample_index+1]
        
        prediction = self.predict(sample)[0]
        probabilities = self.predict_proba(sample)[0]
        confidence = self.calculate_confidence(sample)[0]
        
        feature_importance = self.get_feature_importance()
        
        explanation = {
            'prediction': prediction,
            'probabilities': dict(zip(self.class_names, probabilities)),
            'confidence': confidence,
            'feature_importance': feature_importance,
            'sample_features': {}
        }
        
        if self.feature_names:
            explanation['sample_features'] = dict(zip(self.feature_names, sample[0]))
        
        return explanation
