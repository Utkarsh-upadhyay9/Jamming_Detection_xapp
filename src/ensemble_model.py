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
from models.drl_jamming_detector import DDPGJammingDetector
from src.data_processor import JammingDataProcessor
from utils.metrics import PerformanceMetrics, LatencyTracker
from utils.logger import JammingDetectionLogger
from config.model_config import ENSEMBLE_WEIGHTS, THRESHOLDS, CONFIDENCE_CONFIG

class EnsembleJammingDetector:
    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 threshold: float = None, use_drl: bool = False, 
                 drl_actor_type: str = 'hybrid'):
        self.weights = weights or ENSEMBLE_WEIGHTS
        self.threshold = threshold or THRESHOLDS['binary_detection']
        self.use_drl = use_drl
        
        self.rf_model = RandomForestJammingDetector()
        self.svm_model = SVMJammingDetector()
        self.if_model = IsolationForestJammingDetector()
        
        if self.use_drl:
            self.drl_model = DDPGJammingDetector(
                state_dim=10,
                action_dim=5,
                actor_type=drl_actor_type
            )
            self.weights = weights or {
                'rf': 0.35, 'svm': 0.30, 'if': 0.15, 'drl': 0.20
            }
        
        self.data_processor = JammingDataProcessor()
        
        self.metrics = PerformanceMetrics()
        self.latency_tracker = LatencyTracker()
        self.logger = JammingDetectionLogger()
        
        self.is_trained = False
        self.feature_names = None
        self.class_names = None
        
        self.training_history = {}
        self.prediction_history = []
    
    def load_and_prepare_data(self, normal_traffic_path: str, 
                            jamming_attacks_path: str) -> Dict[str, Any]:
        self.logger.log_system_event("data_loading", "Loading dataset")
        
        normal_data, jamming_data = self.data_processor.load_dataset(
            normal_traffic_path, jamming_attacks_path
        )
        
        features, labels = self.data_processor.prepare_dataset(normal_data, jamming_data)
        
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
        start_time = time.time()
        
        if dataset is None:
            if normal_traffic_path is None or jamming_attacks_path is None:
                raise ValueError("Either dataset or data paths must be provided")
            dataset = self.load_and_prepare_data(normal_traffic_path, jamming_attacks_path)
        
        X_train, X_test = dataset['X_train'], dataset['X_test']
        y_train, y_test = dataset['y_train'], dataset['y_test']
        
        self.logger.log_system_event("training_started", "Starting ensemble training")
        
        from sklearn.model_selection import train_test_split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
        )
        
        training_metrics = {}
        
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
        
        self.logger.log_system_event("if_training", "Training Isolation Forest")
        if_metrics = self.if_model.train(
            X_train_split, y_train_split, X_val, y_val, self.feature_names
        )
        training_metrics['if'] = if_metrics
        self.logger.log_model_training("IsolationForest", if_metrics)
        
        if self.use_drl:
            self.logger.log_system_event("drl_training", "Training DRL Agent")
            drl_metrics = self._train_drl_component(X_train_split, y_train_split, X_val, y_val)
            training_metrics['drl'] = drl_metrics
            self.logger.log_model_training("DRL", drl_metrics)
        
        ensemble_metrics = self.evaluate_model(X_test, y_test)
        training_metrics['ensemble'] = ensemble_metrics
        
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
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        probabilities = self.predict_proba(X)
        
        binary_predictions = (probabilities[:, 1] > self.threshold).astype(int)
        
        predictions = np.array(['normal' if pred == 0 else 'jamming' for pred in binary_predictions])
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        start_time = time.perf_counter()
        
        rf_proba = self.rf_model.predict_proba(X)
        svm_proba = self.svm_model.predict_proba(X)
        if_proba = self.if_model.predict_proba(X)
        
        rf_jamming_prob = self._extract_jamming_probability(rf_proba, 'rf')
        svm_jamming_prob = self._extract_jamming_probability(svm_proba, 'svm')  
        if_jamming_prob = self._extract_jamming_probability(if_proba, 'if')
        
        if self.use_drl:
            drl_jamming_prob = self._get_drl_probability(X)
            
            ensemble_jamming_prob = (
                self.weights['rf'] * rf_jamming_prob +
                self.weights['svm'] * svm_jamming_prob +
                self.weights['if'] * if_jamming_prob +
                self.weights['drl'] * drl_jamming_prob
            )
        else:
            ensemble_jamming_prob = (
                self.weights['rf'] * rf_jamming_prob +
                self.weights['svm'] * svm_jamming_prob +
                self.weights['if'] * if_jamming_prob
            )
        
        ensemble_proba = np.column_stack([
            1 - ensemble_jamming_prob,  # Normal probability
            ensemble_jamming_prob       # Jamming probability
        ])
        
        latency = (time.perf_counter() - start_time) * 1000  # ms
        self.latency_tracker.add_measurement(latency)
        
        return ensemble_proba
    
    def _get_drl_probability(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self.drl_model, 'is_trained') or not self.drl_model:
            return np.ones(len(X)) * 0.5
        
        drl_probabilities = []
        
        for sample in X:
            if len(sample) < 10:
                state = np.zeros(10)
                state[:len(sample)] = sample
            else:
                state = sample[:10]
            
            action = self.drl_model.select_action(state, add_noise=False)
            
            detection_threshold = action[0] if len(action) > 0 else 0.0
            confidence_score = action[1] if len(action) > 1 else 0.0
            
            jamming_probability = (1 + confidence_score) / 2
            jamming_probability = np.clip(jamming_probability, 0.0, 1.0)
            
            drl_probabilities.append(jamming_probability)
        
        return np.array(drl_probabilities)
    
    def _extract_jamming_probability(self, proba: np.ndarray, model_type: str) -> np.ndarray:
        if model_type == 'if':
            return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        else:
            # RF and SVM: find jamming-related classes
            if proba.shape[1] == 2:
                return proba[:, 1]
            else:
                return np.sum(proba[:, 1:], axis=1)
    
    def detect_jamming_type(self, X: np.ndarray) -> List[str]:
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before classification")
        
        ensemble_proba = self.predict_proba(X)
        jamming_detected = ensemble_proba[:, 1] > self.threshold
        
        results = []
        
        for i, is_jamming in enumerate(jamming_detected):
            if not is_jamming:
                results.append('normal')
                continue
            
            sample = X[i:i+1]
            jamming_type = self._classify_jamming_type(sample, ensemble_proba[i, 1])
            results.append(jamming_type)
        
        return results
    
    def _classify_jamming_type(self, sample: np.ndarray, ensemble_confidence: float) -> str:
        sinr_mean = sample[0, 0]  # SINR mean
        sinr_std = sample[0, 1]   # SINR std
        rsrp_std = sample[0, 3]   # RSRP std
        retx_rate = sample[0, 7]  # Retransmission rate
        
        sinr_ref = 15.0  # Reference SINR
        rsrp_std_ref = 3.0  # Reference RSRP std
        rssi_std_ref = 3.0  # Reference RSSI std
        
        rssi_condition = rsrp_std > 2 * rssi_std_ref  # σ_RSSI > 2σ_ref
        sinr_condition = sinr_mean < sinr_ref - 1.5 * rssi_std_ref  # μ_SINR < μ_ref - 1.5σ_ref
        
        if rssi_condition and sinr_condition and ensemble_confidence > 0.8:
            return 'power_jamming'
        
        psd_variance_condition = rsrp_std > 1.5 * rsrp_std_ref
        periodicity_score = self._estimate_periodicity(sample)
        
        if periodicity_score > 0.7 and psd_variance_condition and ensemble_confidence > 0.7:
            return 'sweep_jamming'
        
        correlation_score = self._estimate_correlation(sample)
        adaptivity_score = self._estimate_adaptivity(sample)
        
        if correlation_score > 0.6 and adaptivity_score > 0.8 and ensemble_confidence > 0.6:
            return 'intelligent_jamming'
        
        if ensemble_confidence > 0.8:
            return 'power_jamming'
        elif ensemble_confidence > 0.7:
            return 'sweep_jamming'
        else:
            return 'intelligent_jamming'
    
    def _estimate_periodicity(self, sample: np.ndarray) -> float:
        retx_rate = sample[0, 7]
        packet_loss = sample[0, 8]
        
        variability = abs(retx_rate - 0.02) + abs(packet_loss - 0.001)
        return min(variability * 10, 1.0)  # Normalize to [0, 1]
    
    def _estimate_correlation(self, sample: np.ndarray) -> float:
        ul_throughput = sample[0, 5]
        dl_throughput = sample[0, 6]
        
        throughput_ratio = min(ul_throughput, dl_throughput) / (max(ul_throughput, dl_throughput) + 1e-10)
        return throughput_ratio
    
    def _estimate_adaptivity(self, sample: np.ndarray) -> float:
        prb_ul = sample[0, 11]
        prb_dl = sample[0, 12]
        grant_count = sample[0, 13]
        
        resource_awareness = (prb_ul + prb_dl) / 2 * (grant_count / 50.0)
        return min(resource_awareness, 1.0)
    
    def calculate_ensemble_confidence(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before calculating confidence")
        
        rf_proba = self.rf_model.predict_proba(X)
        svm_proba = self.svm_model.predict_proba(X)
        if_proba = self.if_model.predict_proba(X)
        
        rf_jamming_prob = self._extract_jamming_probability(rf_proba, 'rf')
        svm_jamming_prob = self._extract_jamming_probability(svm_proba, 'svm')
        if_jamming_prob = self._extract_jamming_probability(if_proba, 'if')
        
        predictions = np.column_stack([rf_jamming_prob, svm_jamming_prob, if_jamming_prob])
        
        pred_mean = np.mean(predictions, axis=1)
        pred_variance = np.var(predictions, axis=1)
        
        max_variance = CONFIDENCE_CONFIG['variance_threshold']
        confidence = 1 - (pred_variance / max_variance)
        confidence = np.clip(confidence, 0, 1)
        
        return confidence
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before evaluation")
        
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)
        
        y_binary = np.array(['normal' if label == 'normal' else 'jamming' for label in y_test])
        
        basic_metrics = self.metrics.calculate_basic_metrics(y_binary, predictions)
        
        latency_stats = self.latency_tracker.get_statistics()
        
        performance_metrics = {
            **basic_metrics,
            'mean_latency_ms': latency_stats.get('mean_latency_ms', 0),
            'p95_latency_ms': latency_stats.get('p95_latency_ms', 0),
            'target_compliance_rate': latency_stats.get('target_compliance_rate', 1.0)
        }
        
        self.logger.log_performance_metrics(performance_metrics)
        
        return performance_metrics
    
    def optimize_weights(self, X_val: np.ndarray, y_val: np.ndarray, 
                        search_space: int = 21) -> Dict[str, float]:
        if not self.is_trained:
            raise ValueError("Individual models must be trained before weight optimization")
        
        self.logger.log_system_event("weight_optimization", "Starting weight optimization")
        
        weight_values = np.linspace(0, 1, search_space)
        
        best_f1 = 0
        best_weights = self.weights.copy()
        
        total_combinations = 0
        for rf_weight in weight_values:
            for svm_weight in weight_values:
                for if_weight in weight_values:
                    total_weight = rf_weight + svm_weight + if_weight
                    if total_weight == 0:
                        continue
                    
                    # Normalize weights
                    normalized_weights = {
                        'rf': rf_weight / total_weight,
                        'svm': svm_weight / total_weight,
                        'if': if_weight / total_weight
                    }
                    
                    old_weights = self.weights.copy()
                    self.weights = normalized_weights
                    
                    try:
                        y_pred = self.predict(X_val)
                        y_binary = np.array(['normal' if label == 'normal' else 'jamming' for label in y_val])
                        
                        from sklearn.metrics import f1_score
                        f1 = f1_score(y_binary, y_pred, average='weighted')
                        
                        if f1 > best_f1:
                            best_f1 = f1
                            best_weights = normalized_weights.copy()
                    
                    except:
                        pass  # Skip invalid combinations
                    
                    self.weights = old_weights
                    total_combinations += 1
        
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
        if not self.is_trained:
            raise ValueError("Cannot save untrained ensemble")
        
        os.makedirs(model_dir, exist_ok=True)
        
        self.rf_model.save_model(os.path.join(model_dir, 'rf_model.joblib'))
        self.svm_model.save_model(os.path.join(model_dir, 'svm_model.joblib'))
        self.if_model.save_model(os.path.join(model_dir, 'if_model.joblib'))
        
        ensemble_config = {
            'weights': self.weights,
            'threshold': self.threshold,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'is_trained': self.is_trained,
            'training_history': self.training_history
        }
        
        joblib.dump(ensemble_config, os.path.join(model_dir, 'ensemble_config.joblib'))
        
        joblib.dump(self.data_processor, os.path.join(model_dir, 'data_processor.joblib'))
        
        self.logger.log_system_event("model_saved", f"Ensemble saved to {model_dir}")
    
    def load_ensemble(self, model_dir: str):
        self.rf_model.load_model(os.path.join(model_dir, 'rf_model.joblib'))
        self.svm_model.load_model(os.path.join(model_dir, 'svm_model.joblib'))
        self.if_model.load_model(os.path.join(model_dir, 'if_model.joblib'))
        
        ensemble_config = joblib.load(os.path.join(model_dir, 'ensemble_config.joblib'))
        
        self.weights = ensemble_config['weights']
        self.threshold = ensemble_config['threshold']
        self.feature_names = ensemble_config['feature_names']
        self.class_names = ensemble_config['class_names']
        self.is_trained = ensemble_config['is_trained']
        self.training_history = ensemble_config.get('training_history', {})
        
        self.data_processor = joblib.load(os.path.join(model_dir, 'data_processor.joblib'))
        
        self.logger.log_system_event("model_loaded", f"Ensemble loaded from {model_dir}")
    
    def _train_drl_component(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        from envs.jamming_environment import JammingDetectionEnvironment
        from training.drl_trainer import DRLTrainer
        
        env_config = {
            'state_dim': 10,
            'action_dim': 5,
            'max_episode_steps': 100,
            'jamming_probability': 0.3
        }
        
        trainer = DRLTrainer()
        
        try:
            checkpoint_path = 'models/checkpoints/best_hybrid_agent.pth'
            if os.path.exists(checkpoint_path):
                self.drl_model = trainer.load_agent('best_hybrid_agent.pth', 'hybrid')
                self.logger.log_system_event("drl_loaded", "DRL model loaded from checkpoint")
            else:
                self.drl_model = trainer.train_single_agent('hybrid', num_episodes=100)
                trainer.save_agent(self.drl_model, 'ensemble_drl_agent.pth')
                self.logger.log_system_event("drl_trained", "DRL model trained from scratch")
            
            env = JammingDetectionEnvironment(env_config)
            eval_results = self.drl_model.evaluate(env, num_episodes=10)
            
            return {
                'mean_reward': eval_results['mean_reward'],
                'std_reward': eval_results['std_reward'],
                'training_episodes': 100
            }
            
        except Exception as e:
            self.logger.log_system_event("drl_error", f"DRL training failed: {str(e)}")
            return {
                'mean_reward': 0.0,
                'std_reward': 0.0,
                'training_episodes': 0,
                'error': str(e)
            }
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        info = {
            'is_trained': self.is_trained,
            'weights': self.weights,
            'threshold': self.threshold,
            'use_drl': self.use_drl,
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
        
        if self.use_drl and hasattr(self, 'drl_model'):
            info['individual_models']['drl'] = self.drl_model.get_training_info()
        
        return info
