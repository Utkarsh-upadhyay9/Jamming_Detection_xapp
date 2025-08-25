import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score, classification_report
from typing import Dict, Tuple, Any, Optional
import joblib
import os

from config.model_config import IF_CONFIG, TRAINING_CONFIG

class IsolationForestJammingDetector:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or IF_CONFIG
        self.model = IsolationForest(**self.config)
        self.is_trained = False
        self.feature_names = None
        self.class_names = ['normal', 'jamming']  # Binary classification for anomaly detection
        self.anomaly_threshold = 0.65  # From paper: τ = 0.65
        
        self.training_metrics = {}
        self.prediction_history = []
        self.score_history = []
    
    def train(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              feature_names: Optional[list] = None) -> Dict[str, float]:
        self.feature_names = feature_names
        
        self.model.fit(X_train)
        self.is_trained = True
        
        self.training_metrics = {}
        
        if y_train is not None:
            y_train_binary = self._convert_labels_to_binary(y_train)
            train_pred_scores = self.model.decision_function(X_train)
            train_pred = self._scores_to_predictions(train_pred_scores)
            
            self.training_metrics.update({
                'train_accuracy': accuracy_score(y_train_binary, train_pred),
                'train_f1_score': f1_score(y_train_binary, train_pred, average='binary', pos_label=-1),
                'anomaly_ratio': np.mean(train_pred == -1),
                'mean_anomaly_score': np.mean(train_pred_scores[train_pred == -1]) if np.any(train_pred == -1) else 0
            })
        
        if X_val is not None and y_val is not None:
            y_val_binary = self._convert_labels_to_binary(y_val)
            val_pred_scores = self.model.decision_function(X_val)
            val_pred = self._scores_to_predictions(val_pred_scores)
            
            self.training_metrics.update({
                'val_accuracy': accuracy_score(y_val_binary, val_pred),
                'val_f1_score': f1_score(y_val_binary, val_pred, average='binary', pos_label=-1)
            })
        
        return self.training_metrics
    
    def _convert_labels_to_binary(self, y: np.ndarray) -> np.ndarray:
        if isinstance(y[0], str):
            return np.where(y == 'normal', 1, -1)
        else:
            return np.where(y == 0, 1, -1)  # Assuming 0 is normal class
    
    def _scores_to_predictions(self, scores: np.ndarray) -> np.ndarray:
        return np.where(scores > self.anomaly_threshold, 1, -1)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X)
        self.prediction_history.extend(predictions.tolist())
        
        return predictions
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before computing decision function")
        
        scores = self.model.decision_function(X)
        self.score_history.extend(scores.tolist())
        
        return scores
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        scores = self.decision_function(X)
        
        # Convert scores to probabilities using sigmoid transformation
        normal_prob = 1 / (1 + np.exp(-scores))
        jamming_prob = 1 - normal_prob
        
        return np.column_stack([normal_prob, jamming_prob])
    
    def calculate_confidence(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        
        confidence = np.abs(scores - self.anomaly_threshold)
        
        # Normalize to [0, 1]
        max_distance = max(np.abs(scores.max() - self.anomaly_threshold), 
                          np.abs(scores.min() - self.anomaly_threshold))
        
        if max_distance > 0:
            confidence = confidence / max_distance
        
        return confidence
    
    def get_path_lengths(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained to get path lengths")
        
        scores = self.decision_function(X)
        
        n_samples = len(X)
        c_n = self._calculate_c_n(n_samples)
        
        normalized_scores = (scores + 1) / 2  # Normalize to [0, 1]
        path_lengths = -c_n * np.log2(normalized_scores + 1e-10)
        
        return path_lengths
    
    def _calculate_c_n(self, n: int) -> float:
        if n <= 1:
            return 0
        
        euler_gamma = 0.5772156649015329  # Euler-Mascheroni constant
        return 2 * np.log(n - 1) + euler_gamma - 2 * (n - 1) / n
    
    def analyze_anomalies(self, X: np.ndarray, feature_names: Optional[list] = None) -> Dict[str, Any]:
        if not self.is_trained:
            raise ValueError("Model must be trained to analyze anomalies")
        
        scores = self.decision_function(X)
        predictions = self.predict(X)
        path_lengths = self.get_path_lengths(X)
        
        anomaly_indices = np.where(predictions == -1)[0]
        normal_indices = np.where(predictions == 1)[0]
        
        analysis = {
            'total_samples': len(X),
            'anomaly_count': len(anomaly_indices),
            'normal_count': len(normal_indices),
            'anomaly_ratio': len(anomaly_indices) / len(X),
            'score_statistics': {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores)
            },
            'path_length_statistics': {
                'mean_path_length': np.mean(path_lengths),
                'std_path_length': np.std(path_lengths),
                'min_path_length': np.min(path_lengths),
                'max_path_length': np.max(path_lengths)
            }
        }
        
        if len(anomaly_indices) > 0:
            analysis['anomaly_score_stats'] = {
                'mean_anomaly_score': np.mean(scores[anomaly_indices]),
                'std_anomaly_score': np.std(scores[anomaly_indices]),
                'most_anomalous_index': anomaly_indices[np.argmin(scores[anomaly_indices])]
            }
        
        if len(normal_indices) > 0:
            analysis['normal_score_stats'] = {
                'mean_normal_score': np.mean(scores[normal_indices]),
                'std_normal_score': np.std(scores[normal_indices])
            }
        
        return analysis
    
    def explain_prediction(self, X: np.ndarray, sample_index: int = 0) -> Dict[str, Any]:
        if not self.is_trained:
            raise ValueError("Model must be trained to explain predictions")
        
        sample = X[sample_index:sample_index+1]
        
        prediction = self.predict(sample)[0]
        score = self.decision_function(sample)[0]
        confidence = self.calculate_confidence(sample)[0]
        path_length = self.get_path_lengths(sample)[0]
        probabilities = self.predict_proba(sample)[0]
        
        explanation = {
            'prediction': 'normal' if prediction == 1 else 'jamming',
            'anomaly_score': score,
            'confidence': confidence,
            'path_length': path_length,
            'probabilities': {
                'normal': probabilities[0],
                'jamming': probabilities[1]
            },
            'threshold': self.anomaly_threshold,
            'is_anomaly': prediction == -1,
            'sample_features': {}
        }
        
        if self.feature_names:
            explanation['sample_features'] = dict(zip(self.feature_names, sample[0]))
        
        return explanation
    
    def get_model_info(self) -> Dict[str, Any]:
        if not self.is_trained:
            return {'is_trained': False}
        
        info = {
            'is_trained': True,
            'n_estimators': self.model.n_estimators,
            'contamination': self.config.get('contamination', 'auto'),
            'max_samples': self.model.max_samples_,
            'max_features': self.model.max_features_,
            'anomaly_threshold': self.anomaly_threshold,
            'n_features': getattr(self.model, 'n_features_in_', None),
            'class_names': self.class_names,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'prediction_count': len(self.prediction_history),
            'score_count': len(self.score_history)
        }
        
        return info
    
    def save_model(self, filepath: str):
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'config': self.config,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'anomaly_threshold': self.anomaly_threshold,
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
        self.anomaly_threshold = model_data['anomaly_threshold']
        self.training_metrics = model_data['training_metrics']
        self.is_trained = model_data['is_trained']
    
    def reset_prediction_history(self):
        self.prediction_history = []
        self.score_history = []
    
    def set_threshold(self, new_threshold: float):
        self.anomaly_threshold = new_threshold
    
    def optimize_threshold(self, X_val: np.ndarray, y_val: np.ndarray, 
                          metric: str = 'f1') -> float:
        if not self.is_trained:
            raise ValueError("Model must be trained to optimize threshold")
        
        scores = self.decision_function(X_val)
        y_val_binary = self._convert_labels_to_binary(y_val)
        
        thresholds = np.percentile(scores, range(1, 100))
        best_score = -1
        best_threshold = self.anomaly_threshold
        
        for threshold in thresholds:
            predictions = np.where(scores > threshold, 1, -1)
            
            if metric == 'f1':
                score = f1_score(y_val_binary, predictions, average='binary', pos_label=-1)
            elif metric == 'accuracy':
                score = accuracy_score(y_val_binary, predictions)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        self.anomaly_threshold = best_threshold
        return best_threshold
