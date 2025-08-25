import numpy as np
import pandas as pd
import pytest
import time
from typing import Dict, List, Any
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ensemble_model import EnsembleJammingDetector
from src.data_processor import JammingDataProcessor
from src.jamming_detector import JammingDetectionXApp
from utils.metrics import PerformanceMetrics, LatencyTracker
from utils.visualization import JammingDetectionVisualizer
from config.model_config import PERFORMANCE_REQUIREMENTS

class PerformanceEvaluator:
    def __init__(self):
        self.ensemble_detector = EnsembleJammingDetector()
        self.data_processor = JammingDataProcessor()
        self.metrics = PerformanceMetrics()
        self.visualizer = JammingDetectionVisualizer()
        
        self.target_f1_score = 0.954  # Target F1-score from paper
        self.target_accuracy = 0.956  # Target accuracy
        self.target_latency_ms = 100   # Maximum latency
        self.target_improvement_rf = 14.7  # % improvement over RF
        self.target_improvement_svm = 16.5  # % improvement over SVM
        
        self.results = {}
    
    def load_test_data(self, normal_path: str = None, jamming_path: str = None) -> Dict[str, Any]:
        if normal_path is None:
            normal_path = "Ensemble_ML_Jamming_detection_dataset/dataset/normal_traffic.csv"
        if jamming_path is None:
            jamming_path = "Ensemble_ML_Jamming_detection_dataset/dataset/jamming_attacks.csv"
        
        dataset = self.ensemble_detector.load_and_prepare_data(normal_path, jamming_path)
        
        print(f"Loaded test dataset:")
        print(f"  Training samples: {len(dataset['X_train'])}")
        print(f"  Test samples: {len(dataset['X_test'])}")
        print(f"  Features: {len(dataset['feature_names'])}")
        
        return dataset
    
    def test_ensemble_training(self, dataset: Dict[str, Any]) -> Dict[str, float]:
        print("\n=== Testing Ensemble Training ===")
        
        start_time = time.time()
        
        training_metrics = self.ensemble_detector.train_ensemble(dataset)
        
        training_time = time.time() - start_time
        
        print(f"Training completed in {training_time:.2f} seconds")
        print("Training metrics:")
        for model, metrics in training_metrics.items():
            if isinstance(metrics, dict):
                print(f"  {model.upper()}:")
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"    {metric}: {value:.4f}")
        
        self.results['training_metrics'] = training_metrics
        self.results['training_time'] = training_time
        
        return training_metrics
    
    def test_paper_performance_requirements(self, dataset: Dict[str, Any]) -> Dict[str, bool]:
        print("\n=== Testing Paper Performance Requirements ===")
        
        X_test, y_test = dataset['X_test'], dataset['y_test']
        
        ensemble_metrics = self.ensemble_detector.evaluate_model(X_test, y_test)
        
        individual_metrics = {}
        
        rf_pred = self.ensemble_detector.rf_model.predict(X_test)
        y_binary = np.array(['normal' if label == 'normal' else 'jamming' for label in y_test])
        rf_metrics = self.metrics.calculate_basic_metrics(y_binary, rf_pred)
        individual_metrics['rf'] = rf_metrics
        
        # SVM only
        svm_pred = self.ensemble_detector.svm_model.predict(X_test)
        svm_metrics = self.metrics.calculate_basic_metrics(y_binary, svm_pred)
        individual_metrics['svm'] = svm_metrics
        
        improvements = self.metrics.calculate_ensemble_improvement(
            ensemble_metrics, individual_metrics
        )
        
        requirements_met = {}
        
        requirements_met['f1_score_target'] = ensemble_metrics['f1_score'] >= self.target_f1_score * 0.95  # 5% tolerance
        print(f"F1-Score: {ensemble_metrics['f1_score']:.4f} (Target: {self.target_f1_score:.4f}) - {'✓' if requirements_met['f1_score_target'] else '✗'}")
        
        requirements_met['accuracy_target'] = ensemble_metrics['accuracy'] >= self.target_accuracy * 0.95
        print(f"Accuracy: {ensemble_metrics['accuracy']:.4f} (Target: {self.target_accuracy:.4f}) - {'✓' if requirements_met['accuracy_target'] else '✗'}")
        
        requirements_met['latency_target'] = ensemble_metrics.get('mean_latency_ms', 0) <= self.target_latency_ms
        print(f"Latency: {ensemble_metrics.get('mean_latency_ms', 0):.2f}ms (Target: <{self.target_latency_ms}ms) - {'✓' if requirements_met['latency_target'] else '✗'}")
        
        rf_improvement = improvements.get('rf_f1_score_improvement', 0)
        requirements_met['rf_improvement'] = rf_improvement >= self.target_improvement_rf * 0.8  # 20% tolerance
        print(f"RF Improvement: {rf_improvement:.1f}% (Target: >{self.target_improvement_rf:.1f}%) - {'✓' if requirements_met['rf_improvement'] else '✗'}")
        
        # Improvement over SVM
        svm_improvement = improvements.get('svm_f1_score_improvement', 0)
        requirements_met['svm_improvement'] = svm_improvement >= self.target_improvement_svm * 0.8
        print(f"SVM Improvement: {svm_improvement:.1f}% (Target: >{self.target_improvement_svm:.1f}%) - {'✓' if requirements_met['svm_improvement'] else '✗'}")
        
        requirements_met['overall'] = all(requirements_met.values())
        print(f"\nOverall Requirements Met: {'✓' if requirements_met['overall'] else '✗'}")
        
        self.results['ensemble_metrics'] = ensemble_metrics
        self.results['individual_metrics'] = individual_metrics
        self.results['improvements'] = improvements
        self.results['requirements_met'] = requirements_met
        
        return requirements_met
    
    def test_multi_class_classification(self, dataset: Dict[str, Any]) -> Dict[str, float]:
        print("\n=== Testing Multi-Class Classification ===")
        
        X_test, y_test = dataset['X_test'], dataset['y_test']
        
        jamming_types = self.ensemble_detector.detect_jamming_type(X_test)
        
        class_names = ['normal', 'power_jamming', 'sweep_jamming', 'intelligent_jamming']
        per_class_metrics = self.metrics.calculate_per_class_metrics(
            y_test, jamming_types, class_names
        )
        
        print("Per-class performance:")
        for class_name, metrics in per_class_metrics.items():
            print(f"  {class_name}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1-Score: {metrics['f1_score']:.4f}")
        
        confusion_mat = self.metrics.calculate_confusion_matrix(y_test, jamming_types)
        
        self.results['per_class_metrics'] = per_class_metrics
        self.results['confusion_matrix'] = confusion_mat
        self.results['y_true'] = y_test
        self.results['y_pred'] = jamming_types
        
        return per_class_metrics
    
    def test_different_environments(self, dataset: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        print("\n=== Testing Different Network Environments ===")
        
        X_test, y_test = dataset['X_test'], dataset['y_test']
        environments = ['ideal', 'moderate', 'realistic']
        environment_results = {}
        
        for env in environments:
            print(f"\nTesting {env} environment:")
            
            X_env = self.data_processor.simulate_network_environment(X_test, env)
            
            ensemble_pred = self.ensemble_detector.predict(X_env)
            y_binary = np.array(['normal' if label == 'normal' else 'jamming' for label in y_test])
            ensemble_metrics = self.metrics.calculate_basic_metrics(y_binary, ensemble_pred)
            
            rf_pred = self.ensemble_detector.rf_model.predict(X_env)
            svm_pred = self.ensemble_detector.svm_model.predict(X_env)
            if_pred = self.ensemble_detector.if_model.predict(X_env)
            if_pred = np.array(['normal' if pred == 1 else 'jamming' for pred in if_pred])
            
            rf_metrics = self.metrics.calculate_basic_metrics(y_binary, rf_pred)
            svm_metrics = self.metrics.calculate_basic_metrics(y_binary, svm_pred)
            if_metrics = self.metrics.calculate_basic_metrics(y_binary, if_pred)
            
            environment_results[env] = {
                'ensemble': ensemble_metrics,
                'rf': rf_metrics,
                'svm': svm_metrics,
                'if': if_metrics
            }
            
            print(f"  Ensemble F1-Score: {ensemble_metrics['f1_score']:.4f}")
            print(f"  RF F1-Score: {rf_metrics['f1_score']:.4f}")
            print(f"  SVM F1-Score: {svm_metrics['f1_score']:.4f}")
            print(f"  IF F1-Score: {if_metrics['f1_score']:.4f}")
        
        self.results['environment_results'] = environment_results
        
        return environment_results
    
    def test_weight_optimization(self, dataset: Dict[str, Any]) -> Dict[str, float]:
        print("\n=== Testing Weight Optimization ===")
        
        X_val, y_val = dataset['X_test'][:500], dataset['y_test'][:500]  # Use subset for validation
        
        current_performance = self.ensemble_detector.evaluate_model(X_val, y_val)
        print(f"Current weights performance: {current_performance['f1_score']:.4f}")
        
        optimal_weights = self.ensemble_detector.optimize_weights(X_val, y_val, search_space=11)
        
        optimized_performance = self.ensemble_detector.evaluate_model(X_val, y_val)
        print(f"Optimized weights performance: {optimized_performance['f1_score']:.4f}")
        
        print(f"Optimal weights: {optimal_weights}")
        
        self.results['optimal_weights'] = optimal_weights
        self.results['weight_optimization'] = {
            'before': current_performance,
            'after': optimized_performance
        }
        
        return optimal_weights
    
    def test_latency_performance(self, dataset: Dict[str, Any], num_samples: int = 1000) -> Dict[str, float]:
        print("\n=== Testing Latency Performance ===")
        
        X_test = dataset['X_test'][:num_samples]
        latency_tracker = LatencyTracker(target_latency_ms=100)
        
        latencies = []
        
        for i in range(len(X_test)):
            sample = X_test[i:i+1]
            
            start_time = time.perf_counter()
            prediction = self.ensemble_detector.predict(sample)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            latency_tracker.add_measurement(latency_ms)
        
        latency_stats = latency_tracker.get_statistics()
        
        print(f"Latency Statistics:")
        print(f"  Mean: {latency_stats['mean_latency_ms']:.2f}ms")
        print(f"  P95: {latency_stats['p95_latency_ms']:.2f}ms")
        print(f"  P99: {latency_stats['p99_latency_ms']:.2f}ms")
        print(f"  Compliance Rate: {latency_stats['target_compliance_rate']:.2%}")
        print(f"  Violations: {latency_stats['violations_count']}")
        
        self.results['latency_stats'] = latency_stats
        self.results['latencies'] = latencies
        
        return latency_stats
    
    def test_real_time_xapp(self, duration_seconds: int = 30) -> Dict[str, Any]:
        print(f"\n=== Testing Real-time xApp Performance ({duration_seconds}s) ===")
        
        xapp = JammingDetectionXApp()
        
        xapp.ensemble_detector = self.ensemble_detector
        xapp.is_trained = True
        
        xapp.start_monitoring()
        
        attack_schedule = [
            (5, 'power_jamming', 3),
            (12, 'sweep_jamming', 4),
            (20, 'intelligent_jamming', 5)
        ]
        
        for start_time, jamming_type, duration in attack_schedule:
            def schedule_attack():
                time.sleep(start_time)
                xapp.simulate_jamming_attack(jamming_type, duration)
            
            import threading
            threading.Thread(target=schedule_attack, daemon=True).start()
        
        time.sleep(duration_seconds)
        
        xapp.stop_monitoring()
        
        performance_summary = xapp.get_performance_summary()
        
        print(f"Real-time Performance Summary:")
        print(f"  Total Detections: {performance_summary['total_detections']}")
        print(f"  Jamming Detections: {performance_summary['jamming_detections']}")
        print(f"  Detection Rate: {performance_summary['jamming_detection_rate']:.2%}")
        print(f"  Latency Compliance: {performance_summary['latency_compliance']:.2%}")
        print(f"  Mean Latency: {performance_summary['mean_latency_ms']:.2f}ms")
        
        self.results['real_time_performance'] = performance_summary
        
        return performance_summary
    
    def generate_performance_report(self, save_path: str = "performance_report.html") -> str:
        print(f"\n=== Generating Performance Report ===")
        
        plot_paths = self.visualizer.generate_comprehensive_report(self.results)
        
        html_content = self._create_html_report(plot_paths)
        
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        print(f"Performance report saved to: {save_path}")
        
        return save_path
    
    def _create_html_report(self, plot_paths: List[str]) -> str:
        requirements = self.results.get('requirements_met', {})
        ensemble_metrics = self.results.get('ensemble_metrics', {})
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Jamming Detection Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
                .metric {{ margin: 10px 0; }}
                .success {{ color: green; }}
                .warning {{ color: orange; }}
                .error {{ color: red; }}
                .plot {{ margin: 20px 0; text-align: center; }}
                .plot img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Ensemble Machine Learning Jamming Detection Performance Report</h1>
                <p>Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <h2>Performance Summary</h2>
            <div class="metric">F1-Score: <strong>{ensemble_metrics.get('f1_score', 0):.4f}</strong></div>
            <div class="metric">Accuracy: <strong>{ensemble_metrics.get('accuracy', 0):.4f}</strong></div>
            <div class="metric">Latency: <strong>{ensemble_metrics.get('mean_latency_ms', 0):.2f}ms</strong></div>
            
            <h2>Requirements Validation</h2>
            <div class="metric">F1-Score Target: <span class="{'success' if requirements.get('f1_score_target') else 'error'}">{'✓ Met' if requirements.get('f1_score_target') else '✗ Not Met'}</span></div>
            <div class="metric">Latency Target: <span class="{'success' if requirements.get('latency_target') else 'error'}">{'✓ Met' if requirements.get('latency_target') else '✗ Not Met'}</span></div>
            <div class="metric">RF Improvement: <span class="{'success' if requirements.get('rf_improvement') else 'error'}">{'✓ Met' if requirements.get('rf_improvement') else '✗ Not Met'}</span></div>
            <div class="metric">SVM Improvement: <span class="{'success' if requirements.get('svm_improvement') else 'error'}">{'✓ Met' if requirements.get('svm_improvement') else '✗ Not Met'}</span></div>
            
            <h2>Visualizations</h2>
        </body>
        </html>
        """
        
        return html
    
    def run_comprehensive_evaluation(self, normal_path: str = None, jamming_path: str = None) -> Dict[str, Any]:
        print("Starting Comprehensive Performance Evaluation")
        print("=" * 50)
        
        try:
            dataset = self.load_test_data(normal_path, jamming_path)
            
            training_metrics = self.test_ensemble_training(dataset)
            
            requirements_met = self.test_paper_performance_requirements(dataset)
            
            class_metrics = self.test_multi_class_classification(dataset)
            
            env_results = self.test_different_environments(dataset)
            
            optimal_weights = self.test_weight_optimization(dataset)
            
            latency_stats = self.test_latency_performance(dataset)
            
            real_time_performance = self.test_real_time_xapp(duration_seconds=20)
            
            report_path = self.generate_performance_report()
            
            print("\n" + "=" * 50)
            print("EVALUATION SUMMARY")
            print("=" * 50)
            print(f"Overall Requirements Met: {'✓ PASS' if requirements_met.get('overall', False) else '✗ FAIL'}")
            print(f"F1-Score: {self.results['ensemble_metrics']['f1_score']:.4f}")
            print(f"Latency: {latency_stats['mean_latency_ms']:.2f}ms")
            print(f"Report: {report_path}")
            
            return self.results
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

if __name__ == "__main__":
    evaluator = PerformanceEvaluator()
    results = evaluator.run_comprehensive_evaluation()
