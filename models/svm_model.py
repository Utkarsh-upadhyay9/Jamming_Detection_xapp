"""
Support Vector Machine model implementation for jamming detection.
Based on the research paper specifications with RBF kernel and Platt scaling.
"""

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
    """Support Vector Machine classifier for jamming detection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SVM detector.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or SVM_CONFIG
        self.model = SVC(**self.config)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None
        self.class_names = None
        
        # Performance tracking
        self.training_metrics = {}
        self.prediction_history = []
        
        # Support vector information
        self.support_vectors_info = {}
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              feature_names: Optional[list] = None) -> Dict[str, float]:
        """
        Train the SVM model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            feature_names: Names of features
            
        Returns:
            Training metrics
        """
        self.feature_names = feature_names
        self.class_names = list(np.unique(y_train))
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Store support vector information
        self._extract_support_vector_info()
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train_scaled)
        self.training_metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'train_f1_score': f1_score(y_train, train_pred, average='weighted'),
            'n_support_vectors': self.model.n_support_.sum(),
            'support_vector_ratio': self.model.n_support_.sum() / len(X_train)
        }
        
        # Validation metrics if provided
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_pred = self.model.predict(X_val_scaled)
            self.training_metrics.update({
                'val_accuracy': accuracy_score(y_val, val_pred),
                'val_f1_score': f1_score(y_val, val_pred, average='weighted')
            })
        
        return self.training_metrics
    
    def _extract_support_vector_info(self):
        """Extract and store support vector information."""
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
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        self.prediction_history.extend(predictions.tolist())
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities using Platt scaling.
        
        Args:
            X: Input features
            
        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if not self.config.get('probability', False):
            raise ValueError("SVM must be trained with probability=True for probability predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Get decision function values (distance to hyperplane).
        
        Args:
            X: Input features
            
        Returns:
            Decision function values
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before computing decision function")
        
        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)
    
    def calculate_confidence(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate prediction confidence using decision function distance.
        
        Args:
            X: Input features
            
        Returns:
            Confidence scores
        """
        decision_values = self.decision_function(X)
        
        # For binary classification, use absolute distance to hyperplane
        if len(self.class_names) == 2:
            confidence = np.abs(decision_values)
            # Normalize using sigmoid-like function
            confidence = 1 / (1 + np.exp(-confidence))
        else:
            # For multi-class, use the difference between top two decision values
            if decision_values.ndim == 1:
                decision_values = decision_values.reshape(-1, 1)
            
            sorted_decisions = np.sort(decision_values, axis=1)
            confidence = sorted_decisions[:, -1] - sorted_decisions[:, -2]
            # Normalize to [0, 1]
            confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min() + 1e-10)
        
        return confidence
    
    def get_support_vector_info(self) -> Dict[str, Any]:
        """
        Get detailed support vector information.
        
        Returns:
            Dictionary with support vector details
        """
        return self.support_vectors_info.copy()
    
    def calculate_margin(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate margin (distance to decision boundary) for samples.
        
        Args:
            X: Input features
            
        Returns:
            Margin values
        """
        decision_values = self.decision_function(X)
        
        # For binary classification
        if len(self.class_names) == 2:
            return np.abs(decision_values)
        else:
            # For multi-class, return the minimum margin among all binary classifiers
            if decision_values.ndim == 1:
                decision_values = decision_values.reshape(-1, 1)
            return np.min(np.abs(decision_values), axis=1)
    
    def get_kernel_matrix(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute kernel matrix between X and Y (or X and support vectors if Y is None).
        
        Args:
            X: First set of samples
            Y: Second set of samples (optional)
            
        Returns:
            Kernel matrix
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to compute kernel matrix")
        
        X_scaled = self.scaler.transform(X)
        
        if Y is None:
            # Compute kernel with support vectors
            Y_scaled = self.model.support_vectors_
        else:
            Y_scaled = self.scaler.transform(Y)
        
        # Compute RBF kernel
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
        """
        Explain a single prediction using support vector contributions.
        
        Args:
            X: Input features
            sample_index: Index of the sample to explain
            
        Returns:
            Dictionary with explanation details
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to explain predictions")
        
        sample = X[sample_index:sample_index+1]
        
        # Get prediction and decision function
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
        """
        Calculate support vector contributions to the decision function.
        
        Args:
            X: Input sample
            
        Returns:
            Dictionary with support vector contribution analysis
        """
        if not self.is_trained:
            return {}
        
        # Get kernel values between sample and support vectors
        kernel_values = self.get_kernel_matrix(X)
        
        # Calculate contributions (simplified analysis)
        contributions = {
            'total_support_vectors': len(self.model.support_vectors_),
            'kernel_sum': np.sum(kernel_values),
            'max_kernel_value': np.max(kernel_values),
            'min_kernel_value': np.min(kernel_values),
            'mean_kernel_value': np.mean(kernel_values)
        }
        
        return contributions
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and statistics.
        
        Returns:
            Dictionary with model information
        """
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
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
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
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
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
        """Reset prediction history."""
        self.prediction_history = []
