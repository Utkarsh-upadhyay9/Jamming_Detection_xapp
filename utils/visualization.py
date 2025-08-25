import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
import os

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class JammingDetectionVisualizer:
    def __init__(self, save_dir: str = "plots"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.colors = {
            'normal': '#2E8B57',          # Sea Green
            'power_jamming': '#DC143C',    # Crimson
            'sweep_jamming': '#FF8C00',    # Dark Orange
            'intelligent_jamming': '#4169E1',  # Royal Blue
            'ensemble': '#800080',         # Purple
            'rf': '#228B22',              # Forest Green
            'svm': '#B22222',             # Fire Brick
            'if': '#4682B4'               # Steel Blue
        }
    
    def plot_f1_score_comparison(self, results: Dict[str, Dict[str, float]], 
                                environments: List[str] = None,
                                save_path: Optional[str] = None) -> str:
        if environments is None:
            environments = ['Ideal', 'Moderate', 'Realistic']
        
        models = ['RF', 'SVM', 'IF', 'Ensemble']
        x = np.arange(len(environments))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, model in enumerate(models):
            model_key = model.lower()
            if model == 'Ensemble':
                model_key = 'ensemble'
            
            f1_scores = [results[env][model_key]['f1_score'] for env in environments]
            
            bars = ax.bar(x + i * width, f1_scores, width, 
                         label=model, color=self.colors.get(model_key, f'C{i}'),
                         alpha=0.8, edgecolor='black', linewidth=0.5)
            
            for bar, score in zip(bars, f1_scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Network Environment', fontsize=12, fontweight='bold')
        ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
        ax.set_title('F1-Score Comparison Across Models and Environments', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(environments)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.05)
        
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'f1_score_comparison.pdf')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: List[str] = None,
                             save_path: Optional[str] = None) -> str:
        if class_names is None:
            class_names = ['Normal', 'Power', 'Sweep', 'Intelligent']
        
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Frequency', fontsize=12)
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text = ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2f})',
                             ha="center", va="center", color="black", fontsize=10)
        
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual Class', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix - Ensemble Model', fontsize=14, fontweight='bold')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'confusion_matrix.pdf')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_improvement_analysis(self, ensemble_metrics: Dict[str, float],
                                individual_metrics: Dict[str, Dict[str, float]],
                                save_path: Optional[str] = None) -> str:
        improvements = {}
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for model_name, model_metrics in individual_metrics.items():
            improvements[model_name] = {}
            for metric in metrics:
                if metric in ensemble_metrics and metric in model_metrics:
                    ensemble_val = ensemble_metrics[metric]
                    individual_val = model_metrics[metric]
                    improvement = ((ensemble_val - individual_val) / individual_val) * 100
                    improvements[model_name][metric] = improvement
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            models = list(improvements.keys())
            improvement_values = [improvements[model].get(metric, 0) for model in models]
            
            colors = [self.colors.get(model.lower(), f'C{i}') for i, model in enumerate(models)]
            bars = ax.bar(models, improvement_values, color=colors, alpha=0.8,
                         edgecolor='black', linewidth=0.5)
            
            for bar, value in zip(bars, improvement_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., 
                       height + (0.5 if height >= 0 else -1),
                       f'{value:.1f}%', ha='center', 
                       va='bottom' if height >= 0 else 'top', fontsize=10)
            
            ax.set_title(f'{metric.replace("_", " ").title()} Improvement', 
                        fontsize=12, fontweight='bold')
            ax.set_ylabel('Improvement (%)', fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.suptitle('Ensemble Improvement Over Individual Models', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'improvement_analysis.pdf')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_latency_f1_tradeoff(self, performance_data: List[Dict[str, Any]],
                               save_path: Optional[str] = None) -> str:
        f1_scores = [data['f1_score'] for data in performance_data]
        latencies = [data['latency_ms'] for data in performance_data]
        configs = [data['config_name'] for data in performance_data]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scatter = ax.scatter(latencies, f1_scores, c=range(len(f1_scores)), 
                           cmap='viridis', s=100, alpha=0.7, edgecolors='black')
        
        for i, config in enumerate(configs):
            ax.annotate(config, (latencies[i], f1_scores[i]), 
                       xytext=(5, 5), textcoords='offset points', 
                       fontsize=9, alpha=0.8)
        
        optimal_idx = np.argmax(f1_scores)
        ax.scatter(latencies[optimal_idx], f1_scores[optimal_idx], 
                  c='red', s=200, marker='*', edgecolors='black', linewidth=2,
                  label='Optimal Configuration')
        
        target_latency = 100  # ms
        ax.axvline(x=target_latency, color='red', linestyle='--', alpha=0.7,
                  label=f'Target Latency ({target_latency}ms)')
        
        ax.set_xlabel('Detection Latency (ms)', fontsize=12, fontweight='bold')
        ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
        ax.set_title('F1-Score vs Detection Latency Trade-off', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('Configuration Index', fontsize=11)
        
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'latency_f1_tradeoff.pdf')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_feature_importance(self, feature_importance: Dict[str, float],
                              save_path: Optional[str] = None) -> str:
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        features, importances = zip(*sorted_features)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, importances, color='skyblue', alpha=0.8,
                      edgecolor='black', linewidth=0.5)
        
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            width = bar.get_width()
            ax.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                   f'{importance:.3f}', ha='left', va='center', fontsize=10)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f.replace('_', ' ').title() for f in features])
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_title('Feature Importance Analysis', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'feature_importance.pdf')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_real_time_detection(self, detection_history: List[Dict[str, Any]],
                               window_size: int = 100,
                               save_path: Optional[str] = None) -> str:
        recent_data = detection_history[-window_size:] if len(detection_history) > window_size else detection_history
        
        timestamps = [data['timestamp'] for data in recent_data]
        is_jamming = [data['is_jamming'] for data in recent_data]
        jamming_types = [data['jamming_type'] for data in recent_data]
        confidences = [data['confidence'] for data in recent_data]
        latencies = [data['latency_ms'] for data in recent_data]
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        ax1 = axes[0]
        colors = ['red' if jam else 'green' for jam in is_jamming]
        ax1.scatter(range(len(timestamps)), is_jamming, c=colors, alpha=0.7, s=50)
        ax1.set_ylabel('Jamming Detected', fontsize=11, fontweight='bold')
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(['Normal', 'Jamming'])
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Real-time Jamming Detection Results', fontsize=14, fontweight='bold')
        
        ax2 = axes[1]
        ax2.plot(range(len(timestamps)), confidences, 'b-', linewidth=2, alpha=0.7)
        ax2.fill_between(range(len(timestamps)), confidences, alpha=0.3)
        ax2.set_ylabel('Confidence', fontsize=11, fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[2]
        ax3.plot(range(len(timestamps)), latencies, 'g-', linewidth=2, alpha=0.7)
        ax3.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Target (100ms)')
        ax3.set_ylabel('Latency (ms)', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Time Steps', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'real_time_detection.pdf')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_interactive_dashboard(self, performance_data: Dict[str, Any],
                                   save_path: Optional[str] = None) -> str:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('F1-Score by Environment', 'Latency Distribution',
                           'Detection Confidence', 'Model Comparison'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        environments = ['Ideal', 'Moderate', 'Realistic']
        f1_scores = performance_data.get('f1_by_environment', [0.95, 0.92, 0.88])
        
        fig.add_trace(
            go.Bar(x=environments, y=f1_scores, name='F1-Score',
                  marker_color='lightblue'),
            row=1, col=1
        )
        
        latencies = performance_data.get('latencies', np.random.normal(35, 10, 100))
        fig.add_trace(
            go.Histogram(x=latencies, name='Latency', nbinsx=20,
                        marker_color='lightgreen'),
            row=1, col=2
        )
        
        time_steps = list(range(50))
        confidences = performance_data.get('confidences', np.random.random(50))
        
        fig.add_trace(
            go.Scatter(x=time_steps, y=confidences, mode='lines+markers',
                      name='Confidence', line=dict(color='orange')),
            row=2, col=1
        )
        
        models = ['RF', 'SVM', 'IF', 'Ensemble']
        accuracies = performance_data.get('model_accuracies', [0.85, 0.82, 0.78, 0.94])
        
        fig.add_trace(
            go.Bar(x=models, y=accuracies, name='Accuracy',
                  marker_color=['red', 'blue', 'green', 'purple']),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Jamming Detection Performance Dashboard",
            title_x=0.5,
            showlegend=True
        )
        
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'interactive_dashboard.html')
        
        fig.write_html(save_path)
        
        return save_path
    
    def generate_comprehensive_report(self, results: Dict[str, Any],
                                    save_dir: Optional[str] = None) -> List[str]:
        if save_dir:
            self.save_dir = save_dir
            os.makedirs(save_dir, exist_ok=True)
        
        generated_plots = []
        
        if 'environment_results' in results:
            path = self.plot_f1_score_comparison(results['environment_results'])
            generated_plots.append(path)
        
        if 'y_true' in results and 'y_pred' in results:
            path = self.plot_confusion_matrix(results['y_true'], results['y_pred'])
            generated_plots.append(path)
        
        if 'ensemble_metrics' in results and 'individual_metrics' in results:
            path = self.plot_improvement_analysis(
                results['ensemble_metrics'], results['individual_metrics']
            )
            generated_plots.append(path)
        
        if 'performance_configurations' in results:
            path = self.plot_latency_f1_tradeoff(results['performance_configurations'])
            generated_plots.append(path)
        
        if 'feature_importance' in results:
            path = self.plot_feature_importance(results['feature_importance'])
            generated_plots.append(path)
        
        if 'detection_history' in results:
            path = self.plot_real_time_detection(results['detection_history'])
            generated_plots.append(path)
        
        if 'dashboard_data' in results:
            path = self.create_interactive_dashboard(results['dashboard_data'])
            generated_plots.append(path)
        
        print(f"Generated {len(generated_plots)} visualization plots in {self.save_dir}")
        
        return generated_plots
