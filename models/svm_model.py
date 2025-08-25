import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Any, Optional
import joblib
import os

from config.model_config import SVM_CONFIG, TRAINING_CONFIG

class SVMJammingDetector:
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or SVM_CONFIG
        self.model = SVC(**self.config)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None
        self.class_names = None
        self.training_metrics = {}
        self.prediction_history = []
        self.support_vectors_info = {}
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              feature_names: Optional[list] = None) -> Dict[str, float]:
        self.feature_names = feature_names
        self.class_names = list(np.unique(y_train))
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        self._extract_support_vector_info()
        
        train_pred = self.model.predict(X_train_scaled)
        self.training_metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'train_f1_score': f1_score(y_train, train_pred, average='weighted'),
            'n_support_vectors': self.model.n_support_.sum(),
            'support_vector_ratio': self.model.n_support_.sum() / len(X_train)
        }
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_pred = self.model.predict(X_val_scaled)
            self.training_metrics.update({
                'val_accuracy': accuracy_score(y_val, val_pred),
                'val_f1_score': f1_score(y_val, val_pred, average='weighted')
            })
        
        return self.training_metrics
    
    def _extract_support_vector_info(self):
        if not self.is_trained:
            return
        
        self.support_vectors_info = {
            'n_support_total': self.model.n_support_.sum(),
            'n_support_per_class': dict(zip(self.class_names, self.model.n_support_)),
            'support_vector_indices': self.model.support_,
            'dual_coefficients_shape': self.model.dual_coef_.shape,
            'intercept': self.model.intercept_
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        self.prediction_history.extend(predictions.tolist())
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if not self.config.get('probability', False):
            raise ValueError("SVM must be trained with probability=True for probability predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before computing decision function")
        
        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)
    
    def calculate_confidence(self, X: np.ndarray) -> np.ndarray:
        decision_values = self.decision_function(X)
        
        if len(self.class_names) == 2:
            confidence = np.abs(decision_values)
            confidence = 1 / (1 + np.exp(-confidence)) # sigmoid normalization
        else:
            if decision_values.ndim == 1:
                decision_values = decision_values.reshape(-1, 1)
            
            sorted_decisions = np.sort(decision_values, axis=1)
            confidence = sorted_decisions[:, -1] - sorted_decisions[:, -2]
            confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min() + 1e-10)
        
        return confidence
    
    def get_support_vector_info(self) -> Dict[str, Any]:
        return self.support_vectors_info.copy()
    
    def calculate_margin(self, X: np.ndarray) -> np.ndarray:
        decision_values = self.decision_function(X)
        
        if len(self.class_names) == 2:
            return np.abs(decision_values)
        else:
            if decision_values.ndim == 1:
                decision_values = decision_values.reshape(-1, 1)
            return np.min(np.abs(decision_values), axis=1)
    
    def get_kernel_matrix(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained to compute kernel matrix")
        
        X_scaled = self.scaler.transform(X)
        
        if Y is None:
            Y_scaled = self.model.support_vectors_
        else:
            Y_scaled = self.scaler.transform(Y)
        
        gamma = self.config.get('gamma', 'scale')
        if gamma == 'scale':
            gamma = 1.0 / (X.shape[1] * X.var())
        elif gamma == 'auto':
            gamma = 1.0 / X.shape[1]
        
        # RBF kernel: K(x, y) = exp(-gamma * ||x - y||^2)
        X_norm = np.sum(X_scaled**2, axis=1, keepdims=True)
        Y_norm = np.sum(Y_scaled**2, axis=1, keepdims=True).T
        distances = X_norm + Y_norm - 2 * np.dot(X_scaled, Y_scaled.T)
        
        return np.exp(-gamma * distances)
    
    def explain_prediction(self, X: np.ndarray, sample_index: int = 0) -> Dict[str, Any]:
        if not self.is_trained:
            raise ValueError("Model must be trained to explain predictions")
        
        sample = X[sample_index:sample_index+1]
        
        prediction = self.predict(sample)[0]
        decision_value = self.decision_function(sample)[0]
        confidence = self.calculate_confidence(sample)[0]
        margin = self.calculate_margin(sample)[0]
        
        explanation = {
            'prediction': prediction,
            'decision_value': decision_value,
            'confidence': confidence,
            'margin': margin,
            'support_vector_contributions': self._get_sv_contributions(sample),
            'sample_features': {}
        }
        
        if self.config.get('probability', False):
            probabilities = self.predict_proba(sample)[0]
            explanation['probabilities'] = dict(zip(self.class_names, probabilities))
        
        if self.feature_names:
            explanation['sample_features'] = dict(zip(self.feature_names, sample[0]))
        
        return explanation
    
    def _get_sv_contributions(self, X: np.ndarray) -> Dict[str, float]:
        if not self.is_trained:
            return {}
        
        kernel_values = self.get_kernel_matrix(X)
        
        contributions = {
            'total_support_vectors': len(self.model.support_vectors_),
            'kernel_sum': np.sum(kernel_values),
            'max_kernel_value': np.max(kernel_values),
            'min_kernel_value': np.min(kernel_values),
            'mean_kernel_value': np.mean(kernel_values)
        }
        
        return contributions
    
    def get_model_info(self) -> Dict[str, Any]:
        if not self.is_trained:
            return {'is_trained': False}
        
        info = {
            'is_trained': True,
            'kernel': self.config.get('kernel', 'rbf'),
            'gamma': self.config.get('gamma', 'scale'),
            'C': self.config.get('C', 1.0),
            'n_features': self.model.n_features_in_,
            'n_classes': len(self.class_names),
            'class_names': self.class_names,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'prediction_count': len(self.prediction_history),
            'support_vectors_info': self.support_vectors_info
        }
        
        return info
    
    def save_model(self, filepath: str):
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'training_metrics': self.training_metrics,
            'support_vectors_info': self.support_vectors_info,
            'is_trained': self.is_trained
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.config = model_data['config']
        self.feature_names = model_data['feature_names']
        self.class_names = model_data['class_names']
        self.training_metrics = model_data['training_metrics']
        self.support_vectors_info = model_data.get('support_vectors_info', {})
        self.is_trained = model_data['is_trained']
    
    def reset_prediction_history(self):
        self.prediction_history = []
