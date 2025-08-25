import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve
)
import time
from typing import Dict, List, Tuple, Any

class PerformanceMetrics:
    def __init__(self):
        self.metrics_history = []
        self.timing_data = []
    
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
    
    def calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return confusion_matrix(y_true, y_pred)
    
    def calculate_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  class_names: List[str] = None) -> Dict[str, Dict[str, float]]:
        if class_names is None:
            class_names = [f'Class_{i}' for i in range(len(np.unique(y_true)))]
        
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        per_class_metrics = {}
        for i, class_name in enumerate(class_names):
            per_class_metrics[class_name] = {
                'precision': precision[i] if i < len(precision) else 0.0,
                'recall': recall[i] if i < len(recall) else 0.0,
                'f1_score': f1[i] if i < len(f1) else 0.0
            }
        
        return per_class_metrics
    
    def calculate_roc_metrics(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        try:
            auc_score = roc_auc_score(y_true, y_prob)
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            
            return {
                'auc_score': auc_score,
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds
            }
        except Exception as e:
            print(f"Error calculating ROC metrics: {e}")
            return {'auc_score': 0.0, 'fpr': None, 'tpr': None, 'thresholds': None}
    
    def calculate_pr_metrics(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        try:
            precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
            
            return {
                'precision': precision,
                'recall': recall,
                'thresholds': thresholds
            }
        except Exception as e:
            print(f"Error calculating PR metrics: {e}")
            return {'precision': None, 'recall': None, 'thresholds': None}
    
    def measure_latency(self, func, *args, **kwargs) -> Tuple[Any, float]:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        self.timing_data.append(latency_ms)
        
        return result, latency_ms
    
    def calculate_ensemble_improvement(self, ensemble_metrics: Dict[str, float], 
                                     individual_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        improvements = {}
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            ensemble_value = ensemble_metrics.get(metric, 0)
            
            for model_name, model_metrics in individual_metrics.items():
                individual_value = model_metrics.get(metric, 0)
                
                if individual_value > 0:
                    improvement = ((ensemble_value - individual_value) / individual_value) * 100
                    improvements[f'{model_name}_{metric}_improvement'] = improvement
        
        return improvements
    
    def calculate_confidence_metrics(self, predictions: np.ndarray) -> Dict[str, float]:
        return {
            'mean_confidence': np.mean(predictions),
            'std_confidence': np.std(predictions),
            'min_confidence': np.min(predictions),
            'max_confidence': np.max(predictions),
            'confidence_entropy': -np.mean(predictions * np.log2(predictions + 1e-10))
        }
    
    def generate_performance_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  y_prob: np.ndarray = None, 
                                  class_names: List[str] = None) -> Dict[str, Any]:
        report = {}
        
        report['basic_metrics'] = self.calculate_basic_metrics(y_true, y_pred)
        
        report['confusion_matrix'] = self.calculate_confusion_matrix(y_true, y_pred)
        
        report['per_class_metrics'] = self.calculate_per_class_metrics(y_true, y_pred, class_names)
        
        if y_prob is not None and len(np.unique(y_true)) == 2:
            report['roc_metrics'] = self.calculate_roc_metrics(y_true, y_prob)
            report['pr_metrics'] = self.calculate_pr_metrics(y_true, y_prob)
        
        if y_prob is not None:
            report['confidence_metrics'] = self.calculate_confidence_metrics(y_prob)
        
        if self.timing_data:
            report['timing_metrics'] = {
                'mean_latency_ms': np.mean(self.timing_data),
                'std_latency_ms': np.std(self.timing_data),
                'min_latency_ms': np.min(self.timing_data),
                'max_latency_ms': np.max(self.timing_data),
                'p95_latency_ms': np.percentile(self.timing_data, 95),
                'p99_latency_ms': np.percentile(self.timing_data, 99)
            }
        
        return report
    
    def save_metrics_history(self, filepath: str):
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(filepath, index=False)
    
    def load_metrics_history(self, filepath: str):
        df = pd.read_csv(filepath)
        self.metrics_history = df.to_dict('records')

class LatencyTracker:
    def __init__(self, target_latency_ms: float = 100):
        self.target_latency_ms = target_latency_ms
        self.measurements = []
    
    def add_measurement(self, latency_ms: float):
        self.measurements.append(latency_ms)
    
    def get_statistics(self) -> Dict[str, float]:
        if not self.measurements:
            return {}
        
        measurements = np.array(self.measurements)
        
        return {
            'mean_latency_ms': np.mean(measurements),
            'median_latency_ms': np.median(measurements),
            'std_latency_ms': np.std(measurements),
            'min_latency_ms': np.min(measurements),
            'max_latency_ms': np.max(measurements),
            'p95_latency_ms': np.percentile(measurements, 95),
            'p99_latency_ms': np.percentile(measurements, 99),
            'target_compliance_rate': np.mean(measurements <= self.target_latency_ms),
            'violations_count': np.sum(measurements > self.target_latency_ms)
        }
    
    def is_compliant(self) -> bool:
        if not self.measurements:
            return True
        
        return np.mean(self.measurements) <= self.target_latency_ms
