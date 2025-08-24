"""
Ensemble machine learning model for jamming detection.
Combines Random Forest, SVM, and Isolation Forest with optimized weights.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import time
import joblib
import os
from itertools import product

from models.rf_model import RandomForestJammingDetector
from models.svm_model import SVMJammingDetector  
from models.isolation_forest_model import IsolationForestJammingDetector
from src.data_processor import JammingDataProcessor
from utils.metrics import PerformanceMetrics, LatencyTracker
from utils.logger import JammingDetectionLogger
from config.model_config import ENSEMBLE_WEIGHTS, THRESHOLDS, CONFIDENCE_CONFIG

class EnsembleJammingDetector:
    """
    Ensemble machine learning detector combining RF, SVM, and IF.
    Based on the research paper specifications with optimal weights.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 threshold: float = None):
        """
        Initialize ensemble detector.
        
        Args:
            weights: Optional custom weights for ensemble models
            threshold: Optional custom detection threshold
        """
        # Model weights (optimized from paper: 44% RF, 41% SVM, 15% IF)
        self.weights = weights or ENSEMBLE_WEIGHTS
        self.threshold = threshold or THRESHOLDS['binary_detection']
        
        # Initialize individual models
        self.rf_model = RandomForestJammingDetector()
        self.svm_model = SVMJammingDetector()
        self.if_model = IsolationForestJammingDetector()
        
        # Data processor
        self.data_processor = JammingDataProcessor()
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.latency_tracker = LatencyTracker()
        self.logger = JammingDetectionLogger()
        
        # Training state
        self.is_trained = False
        self.feature_names = None
        self.class_names = None
        
        # Performance history
        self.training_history = {}
        self.prediction_history = []
    
    def load_and_prepare_data(self, normal_traffic_path: str, 
                            jamming_attacks_path: str) -> Dict[str, Any]:
        """
        Load and prepare the dataset for training.
        
        Args:
            normal_traffic_path: Path to normal traffic CSV
            jamming_attacks_path: Path to jamming attacks CSV
            
        Returns:
            Prepared dataset dictionary
        """
        self.logger.log_system_event("data_loading", "Loading dataset")
        
        # Load data
        normal_data, jamming_data = self.data_processor.load_dataset(
            normal_traffic_path, jamming_attacks_path
        )
        
        # Prepare features and labels
        features, labels = self.data_processor.prepare_dataset(normal_data, jamming_data)
        
        # Preprocess for training
        dataset = self.data_processor.preprocess_for_training(features, labels)
        
        self.feature_names = dataset['feature_names']
        self.class_names = list(np.unique(labels))
        
        self.logger.log_system_event(
            "data_prepared", 
            "Dataset prepared successfully",
            additional_data={
                'n_samples': len(features),
                'n_features': len(self.feature_names),
                'n_classes': len(self.class_names)
            }
        )
        
        return dataset
    
    def train_ensemble(self, dataset: Optional[Dict[str, Any]] = None,
                      normal_traffic_path: str = None, 
                      jamming_attacks_path: str = None) -> Dict[str, float]:
        """
        Train the ensemble model.
        
        Args:
            dataset: Optional pre-prepared dataset
            normal_traffic_path: Path to normal traffic data
            jamming_attacks_path: Path to jamming attacks data
            
        Returns:
            Training metrics
        """
        start_time = time.time()
        
        # Load data if not provided
        if dataset is None:
            if normal_traffic_path is None or jamming_attacks_path is None:
                raise ValueError("Either dataset or data paths must be provided")
            dataset = self.load_and_prepare_data(normal_traffic_path, jamming_attacks_path)
        
        X_train, X_test = dataset['X_train'], dataset['X_test']
        y_train, y_test = dataset['y_train'], dataset['y_test']
        
        self.logger.log_system_event("training_started", "Starting ensemble training")
        
        # Create validation split from training data
        from sklearn.model_selection import train_test_split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
        )
        
        # Train individual models
        training_metrics = {}
        
        # Train Random Forest
        self.logger.log_system_event("rf_training", "Training Random Forest")
        rf_metrics = self.rf_model.train(
            X_train_split, y_train_split, X_val, y_val, self.feature_names
        )
        training_metrics['rf'] = rf_metrics
        self.logger.log_model_training("RandomForest", rf_metrics)
        
        # Train SVM
        self.logger.log_system_event("svm_training", "Training SVM")
        svm_metrics = self.svm_model.train(
            X_train_split, y_train_split, X_val, y_val, self.feature_names
        )
        training_metrics['svm'] = svm_metrics
        self.logger.log_model_training("SVM", svm_metrics)
        
        # Train Isolation Forest (unsupervised)
        self.logger.log_system_event("if_training", "Training Isolation Forest")
        if_metrics = self.if_model.train(
            X_train_split, y_train_split, X_val, y_val, self.feature_names
        )
        training_metrics['if'] = if_metrics
        self.logger.log_model_training("IsolationForest", if_metrics)
        
        # Evaluate ensemble on test set
        ensemble_metrics = self.evaluate_model(X_test, y_test)
        training_metrics['ensemble'] = ensemble_metrics
        
        # Calculate training time
        training_time = time.time() - start_time
        training_metrics['training_time_seconds'] = training_time
        
        self.is_trained = True
        self.training_history = training_metrics
        
        self.logger.log_system_event(
            "training_completed", 
            "Ensemble training completed",
            additional_data={'training_time': training_time, 'metrics': ensemble_metrics}
        )
        
        return training_metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        # Get predictions from individual models
        probabilities = self.predict_proba(X)
        
        # Apply threshold for binary classification
        binary_predictions = (probabilities[:, 1] > self.threshold).astype(int)
        
        # Convert to class labels
        predictions = np.array(['normal' if pred == 0 else 'jamming' for pred in binary_predictions])
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get ensemble prediction probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Prediction probabilities [normal_prob, jamming_prob]
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        start_time = time.perf_counter()
        
        # Get probabilities from individual models
        rf_proba = self.rf_model.predict_proba(X)
        svm_proba = self.svm_model.predict_proba(X)
        if_proba = self.if_model.predict_proba(X)
        
        # Handle different probability array shapes
        rf_jamming_prob = self._extract_jamming_probability(rf_proba, 'rf')
        svm_jamming_prob = self._extract_jamming_probability(svm_proba, 'svm')  
        if_jamming_prob = self._extract_jamming_probability(if_proba, 'if')
        
        # Weighted ensemble combination
        ensemble_jamming_prob = (
            self.weights['rf'] * rf_jamming_prob +
            self.weights['svm'] * svm_jamming_prob +
            self.weights['if'] * if_jamming_prob
        )
        
        # Create probability matrix [normal_prob, jamming_prob]
        ensemble_proba = np.column_stack([
            1 - ensemble_jamming_prob,  # Normal probability
            ensemble_jamming_prob       # Jamming probability
        ])
        
        # Track latency
        latency = (time.perf_counter() - start_time) * 1000  # ms
        self.latency_tracker.add_measurement(latency)
        
        return ensemble_proba
    
    def _extract_jamming_probability(self, proba: np.ndarray, model_type: str) -> np.ndarray:
        """
        Extract jamming probability from model predictions.
        
        Args:
            proba: Probability array from model
            model_type: Type of model ('rf', 'svm', 'if')
            
        Returns:
            Jamming probability array
        """
        if model_type == 'if':
            # IF returns [normal_prob, jamming_prob]
            return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        else:
            # RF and SVM: find jamming-related classes
            if proba.shape[1] == 2:
                # Binary classification
                return proba[:, 1]
            else:
                # Multi-class: sum probabilities of all jamming types
                # Assume class 0 is normal, others are jamming
                return np.sum(proba[:, 1:], axis=1)
    
    def detect_jamming_type(self, X: np.ndarray) -> List[str]:
        """
        Perform multi-class jamming classification.
        
        Args:
            X: Input features
            
        Returns:
            List of detected jamming types
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before classification")
        
        # First, detect if jamming is present
        ensemble_proba = self.predict_proba(X)
        jamming_detected = ensemble_proba[:, 1] > self.threshold
        
        results = []
        
        for i, is_jamming in enumerate(jamming_detected):
            if not is_jamming:
                results.append('normal')
                continue
            
            # Multi-class classification for jamming type
            sample = X[i:i+1]
            jamming_type = self._classify_jamming_type(sample, ensemble_proba[i, 1])
            results.append(jamming_type)
        
        return results
    
    def _classify_jamming_type(self, sample: np.ndarray, ensemble_confidence: float) -> str:
        """
        Classify specific jamming type using hierarchical rules from paper.
        
        Args:
            sample: Single sample features
            ensemble_confidence: Ensemble jamming confidence
            
        Returns:
            Specific jamming type
        """
        # Extract key features for jamming type classification
        sinr_mean = sample[0, 0]  # SINR mean
        sinr_std = sample[0, 1]   # SINR std
        rsrp_std = sample[0, 3]   # RSRP std
        retx_rate = sample[0, 7]  # Retransmission rate
        
        # Reference values (would be calculated from normal traffic)
        sinr_ref = 15.0  # Reference SINR
        rsrp_std_ref = 3.0  # Reference RSRP std
        rssi_std_ref = 3.0  # Reference RSSI std
        
        # Power Jamming Detection (highest priority)
        rssi_condition = rsrp_std > 2 * rssi_std_ref  # σ_RSSI > 2σ_ref
        sinr_condition = sinr_mean < sinr_ref - 1.5 * rssi_std_ref  # μ_SINR < μ_ref - 1.5σ_ref
        
        if rssi_condition and sinr_condition and ensemble_confidence > 0.8:
            return 'power_jamming'
        
        # Sweep Jamming Detection
        # Simplified spectral analysis - in practice would use FFT
        psd_variance_condition = rsrp_std > 1.5 * rsrp_std_ref
        periodicity_score = self._estimate_periodicity(sample)
        
        if periodicity_score > 0.7 and psd_variance_condition and ensemble_confidence > 0.7:
            return 'sweep_jamming'
        
        # Intelligent Jamming Detection
        correlation_score = self._estimate_correlation(sample)
        adaptivity_score = self._estimate_adaptivity(sample)
        
        if correlation_score > 0.6 and adaptivity_score > 0.8 and ensemble_confidence > 0.6:
            return 'intelligent_jamming'
        
        # Default to power jamming if confidence is high
        if ensemble_confidence > 0.8:
            return 'power_jamming'
        elif ensemble_confidence > 0.7:
            return 'sweep_jamming'
        else:
            return 'intelligent_jamming'
    
    def _estimate_periodicity(self, sample: np.ndarray) -> float:
        """Estimate periodicity score for sweep jamming detection."""
        # Simplified periodicity estimation
        # In practice, would analyze frequency domain characteristics
        retx_rate = sample[0, 7]
        packet_loss = sample[0, 8]
        
        # High variability suggests sweep pattern
        variability = abs(retx_rate - 0.02) + abs(packet_loss - 0.001)
        return min(variability * 10, 1.0)  # Normalize to [0, 1]
    
    def _estimate_correlation(self, sample: np.ndarray) -> float:
        """Estimate temporal correlation for intelligent jamming."""
        # Simplified correlation estimation
        ul_throughput = sample[0, 5]
        dl_throughput = sample[0, 6]
        
        # Adaptive jammers show correlation with traffic patterns
        throughput_ratio = min(ul_throughput, dl_throughput) / (max(ul_throughput, dl_throughput) + 1e-10)
        return throughput_ratio
    
    def _estimate_adaptivity(self, sample: np.ndarray) -> float:
        """Estimate behavior adaptivity for intelligent jamming."""
        # Simplified adaptivity estimation
        prb_ul = sample[0, 11]
        prb_dl = sample[0, 12]
        grant_count = sample[0, 13]
        
        # Intelligent jammers adapt to resource utilization
        resource_awareness = (prb_ul + prb_dl) / 2 * (grant_count / 50.0)
        return min(resource_awareness, 1.0)
    
    def calculate_ensemble_confidence(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate ensemble prediction confidence.
        
        Args:
            X: Input features
            
        Returns:
            Confidence scores
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before calculating confidence")
        
        # Get probabilities from individual models
        rf_proba = self.rf_model.predict_proba(X)
        svm_proba = self.svm_model.predict_proba(X)
        if_proba = self.if_model.predict_proba(X)
        
        # Extract jamming probabilities
        rf_jamming_prob = self._extract_jamming_probability(rf_proba, 'rf')
        svm_jamming_prob = self._extract_jamming_probability(svm_proba, 'svm')
        if_jamming_prob = self._extract_jamming_probability(if_proba, 'if')
        
        # Calculate mean and variance of predictions
        predictions = np.column_stack([rf_jamming_prob, svm_jamming_prob, if_jamming_prob])
        
        pred_mean = np.mean(predictions, axis=1)
        pred_variance = np.var(predictions, axis=1)
        
        # Confidence based on agreement between models (low variance = high confidence)
        max_variance = CONFIDENCE_CONFIG['variance_threshold']
        confidence = 1 - (pred_variance / max_variance)
        confidence = np.clip(confidence, 0, 1)
        
        return confidence
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate ensemble model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Performance metrics
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before evaluation")
        
        # Make predictions
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)
        
        # Convert multi-class labels to binary for evaluation
        y_binary = np.array(['normal' if label == 'normal' else 'jamming' for label in y_test])
        
        # Calculate metrics
        basic_metrics = self.metrics.calculate_basic_metrics(y_binary, predictions)
        
        # Calculate latency statistics
        latency_stats = self.latency_tracker.get_statistics()
        
        # Combine metrics
        performance_metrics = {
            **basic_metrics,
            'mean_latency_ms': latency_stats.get('mean_latency_ms', 0),
            'p95_latency_ms': latency_stats.get('p95_latency_ms', 0),
            'target_compliance_rate': latency_stats.get('target_compliance_rate', 1.0)
        }
        
        # Log performance
        self.logger.log_performance_metrics(performance_metrics)
        
        return performance_metrics
    
    def optimize_weights(self, X_val: np.ndarray, y_val: np.ndarray, 
                        search_space: int = 21) -> Dict[str, float]:
        """
        Optimize ensemble weights using grid search.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            search_space: Number of weight values to test (21 gives 0.0, 0.05, ..., 1.0)
            
        Returns:
            Optimal weights
        """
        if not self.is_trained:
            raise ValueError("Individual models must be trained before weight optimization")
        
        self.logger.log_system_event("weight_optimization", "Starting weight optimization")
        
        # Generate weight combinations
        weight_values = np.linspace(0, 1, search_space)
        
        best_f1 = 0
        best_weights = self.weights.copy()
        
        # Grid search over weight combinations
        total_combinations = 0
        for rf_weight in weight_values:
            for svm_weight in weight_values:
                for if_weight in weight_values:
                    # Ensure weights sum to 1
                    total_weight = rf_weight + svm_weight + if_weight
                    if total_weight == 0:
                        continue
                    
                    # Normalize weights
                    normalized_weights = {
                        'rf': rf_weight / total_weight,
                        'svm': svm_weight / total_weight,
                        'if': if_weight / total_weight
                    }
                    
                    # Test this weight combination
                    old_weights = self.weights.copy()
                    self.weights = normalized_weights
                    
                    try:
                        # Make predictions with new weights
                        y_pred = self.predict(X_val)
                        y_binary = np.array(['normal' if label == 'normal' else 'jamming' for label in y_val])
                        
                        # Calculate F1 score
                        from sklearn.metrics import f1_score
                        f1 = f1_score(y_binary, y_pred, average='weighted')
                        
                        # Update best weights if this is better
                        if f1 > best_f1:
                            best_f1 = f1
                            best_weights = normalized_weights.copy()
                    
                    except:
                        pass  # Skip invalid combinations
                    
                    # Restore original weights
                    self.weights = old_weights
                    total_combinations += 1
        
        # Set optimal weights
        self.weights = best_weights
        
        self.logger.log_ensemble_optimization(best_weights, best_f1)
        self.logger.log_system_event(
            "weight_optimization_completed",
            "Weight optimization completed",
            additional_data={
                'optimal_weights': best_weights,
                'best_f1_score': best_f1,
                'combinations_tested': total_combinations
            }
        )
        
        return best_weights
    
    def save_ensemble(self, model_dir: str):
        """
        Save the entire ensemble model.
        
        Args:
            model_dir: Directory to save models
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained ensemble")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save individual models
        self.rf_model.save_model(os.path.join(model_dir, 'rf_model.joblib'))
        self.svm_model.save_model(os.path.join(model_dir, 'svm_model.joblib'))
        self.if_model.save_model(os.path.join(model_dir, 'if_model.joblib'))
        
        # Save ensemble configuration
        ensemble_config = {
            'weights': self.weights,
            'threshold': self.threshold,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'is_trained': self.is_trained,
            'training_history': self.training_history
        }
        
        joblib.dump(ensemble_config, os.path.join(model_dir, 'ensemble_config.joblib'))
        
        # Save data processor
        joblib.dump(self.data_processor, os.path.join(model_dir, 'data_processor.joblib'))
        
        self.logger.log_system_event("model_saved", f"Ensemble saved to {model_dir}")
    
    def load_ensemble(self, model_dir: str):
        """
        Load the entire ensemble model.
        
        Args:
            model_dir: Directory containing saved models
        """
        # Load individual models
        self.rf_model.load_model(os.path.join(model_dir, 'rf_model.joblib'))
        self.svm_model.load_model(os.path.join(model_dir, 'svm_model.joblib'))
        self.if_model.load_model(os.path.join(model_dir, 'if_model.joblib'))
        
        # Load ensemble configuration
        ensemble_config = joblib.load(os.path.join(model_dir, 'ensemble_config.joblib'))
        
        self.weights = ensemble_config['weights']
        self.threshold = ensemble_config['threshold']
        self.feature_names = ensemble_config['feature_names']
        self.class_names = ensemble_config['class_names']
        self.is_trained = ensemble_config['is_trained']
        self.training_history = ensemble_config.get('training_history', {})
        
        # Load data processor
        self.data_processor = joblib.load(os.path.join(model_dir, 'data_processor.joblib'))
        
        self.logger.log_system_event("model_loaded", f"Ensemble loaded from {model_dir}")
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """
        Get comprehensive ensemble information.
        
        Returns:
            Dictionary with ensemble details
        """
        info = {
            'is_trained': self.is_trained,
            'weights': self.weights,
            'threshold': self.threshold,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'training_history': self.training_history,
            'latency_stats': self.latency_tracker.get_statistics(),
            'individual_models': {
                'rf': self.rf_model.get_model_info(),
                'svm': self.svm_model.get_model_info(),
                'if': self.if_model.get_model_info()
            }
        }
        
        return info
