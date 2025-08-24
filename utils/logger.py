"""
Logging utilities for the jamming detection xApp.
"""

import logging
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional

class JammingDetectionLogger:
    """Logger for jamming detection events and system performance."""
    
    def __init__(self, log_dir: str = "logs", log_level: int = logging.INFO):
        self.log_dir = log_dir
        self.log_level = log_level
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup loggers
        self.detection_logger = self._setup_logger("detection", "jamming_detection.log")
        self.performance_logger = self._setup_logger("performance", "performance.log")
        self.system_logger = self._setup_logger("system", "system.log")
        
        # Event counters
        self.event_counts = {
            'normal_traffic': 0,
            'power_jamming': 0,
            'sweep_jamming': 0,
            'intelligent_jamming': 0,
            'false_positives': 0,
            'false_negatives': 0
        }
        
        # Performance tracking
        self.performance_metrics = {
            'total_detections': 0,
            'detection_latencies': [],
            'accuracy_scores': [],
            'f1_scores': []
        }
    
    def _setup_logger(self, name: str, filename: str) -> logging.Logger:
        """Setup individual logger with file and console handlers."""
        logger = logging.getLogger(name)
        logger.setLevel(self.log_level)
        
        # Prevent duplicate handlers
        if logger.handlers:
            return logger
        
        # File handler
        file_handler = logging.FileHandler(
            os.path.join(self.log_dir, filename), 
            mode='a'
        )
        file_handler.setLevel(self.log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_detection_event(self, detection_result: Dict[str, Any]):
        """Log jamming detection event."""
        timestamp = datetime.now().isoformat()
        
        # Update event counters
        detected_class = detection_result.get('predicted_class', 'unknown')
        if detected_class in self.event_counts:
            self.event_counts[detected_class] += 1
        
        # Log the event
        log_entry = {
            'timestamp': timestamp,
            'event_type': 'detection',
            'predicted_class': detected_class,
            'confidence': detection_result.get('confidence', 0.0),
            'latency_ms': detection_result.get('latency_ms', 0.0),
            'features': detection_result.get('features', {}),
            'ensemble_scores': detection_result.get('ensemble_scores', {})
        }
        
        self.detection_logger.info(json.dumps(log_entry))
        
        # Update performance tracking
        self.performance_metrics['total_detections'] += 1
        if 'latency_ms' in detection_result:
            self.performance_metrics['detection_latencies'].append(detection_result['latency_ms'])
    
    def log_performance_metrics(self, metrics: Dict[str, float]):
        """Log performance metrics."""
        timestamp = datetime.now().isoformat()
        
        # Update performance tracking
        if 'accuracy' in metrics:
            self.performance_metrics['accuracy_scores'].append(metrics['accuracy'])
        if 'f1_score' in metrics:
            self.performance_metrics['f1_scores'].append(metrics['f1_score'])
        
        log_entry = {
            'timestamp': timestamp,
            'event_type': 'performance_metrics',
            'metrics': metrics
        }
        
        self.performance_logger.info(json.dumps(log_entry))
    
    def log_system_event(self, event_type: str, message: str, level: str = "INFO", 
                        additional_data: Optional[Dict[str, Any]] = None):
        """Log system events."""
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'event_type': event_type,
            'message': message
        }
        
        if additional_data:
            log_entry.update(additional_data)
        
        log_message = json.dumps(log_entry)
        
        if level.upper() == "DEBUG":
            self.system_logger.debug(log_message)
        elif level.upper() == "INFO":
            self.system_logger.info(log_message)
        elif level.upper() == "WARNING":
            self.system_logger.warning(log_message)
        elif level.upper() == "ERROR":
            self.system_logger.error(log_message)
        elif level.upper() == "CRITICAL":
            self.system_logger.critical(log_message)
    
    def log_model_training(self, model_name: str, training_metrics: Dict[str, float]):
        """Log model training results."""
        self.log_system_event(
            event_type="model_training",
            message=f"Model {model_name} training completed",
            additional_data={
                'model': model_name,
                'training_metrics': training_metrics
            }
        )
    
    def log_ensemble_optimization(self, weights: Dict[str, float], f1_score: float):
        """Log ensemble weight optimization results."""
        self.log_system_event(
            event_type="ensemble_optimization",
            message="Ensemble weights optimized",
            additional_data={
                'optimized_weights': weights,
                'achieved_f1_score': f1_score
            }
        )
    
    def log_error(self, error_type: str, error_message: str, 
                 exception: Optional[Exception] = None):
        """Log error events."""
        error_data = {
            'error_type': error_type,
            'error_message': error_message
        }
        
        if exception:
            error_data['exception_type'] = type(exception).__name__
            error_data['exception_details'] = str(exception)
        
        self.log_system_event(
            event_type="error",
            message=f"Error occurred: {error_type}",
            level="ERROR",
            additional_data=error_data
        )
    
    def log_e2_interface_event(self, event_type: str, data: Dict[str, Any]):
        """Log E2 interface events."""
        self.log_system_event(
            event_type=f"e2_interface_{event_type}",
            message=f"E2 interface event: {event_type}",
            additional_data=data
        )
    
    def log_ric_communication(self, message_type: str, payload: Dict[str, Any]):
        """Log RIC communication events."""
        self.log_system_event(
            event_type="ric_communication",
            message=f"RIC communication: {message_type}",
            additional_data={
                'message_type': message_type,
                'payload': payload
            }
        )
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """Get summary of detection events."""
        total_events = sum(self.event_counts.values())
        
        summary = {
            'total_detections': total_events,
            'event_counts': self.event_counts.copy(),
            'event_percentages': {}
        }
        
        if total_events > 0:
            for event_type, count in self.event_counts.items():
                summary['event_percentages'][event_type] = (count / total_events) * 100
        
        return summary
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics."""
        latencies = self.performance_metrics['detection_latencies']
        accuracies = self.performance_metrics['accuracy_scores']
        f1_scores = self.performance_metrics['f1_scores']
        
        summary = {
            'total_detections': self.performance_metrics['total_detections']
        }
        
        if latencies:
            summary['latency_stats'] = {
                'mean_ms': sum(latencies) / len(latencies),
                'min_ms': min(latencies),
                'max_ms': max(latencies),
                'count': len(latencies)
            }
        
        if accuracies:
            summary['accuracy_stats'] = {
                'mean': sum(accuracies) / len(accuracies),
                'min': min(accuracies),
                'max': max(accuracies),
                'count': len(accuracies)
            }
        
        if f1_scores:
            summary['f1_score_stats'] = {
                'mean': sum(f1_scores) / len(f1_scores),
                'min': min(f1_scores),
                'max': max(f1_scores),
                'count': len(f1_scores)
            }
        
        return summary
    
    def export_logs(self, export_format: str = "json") -> str:
        """Export logs in specified format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if export_format.lower() == "json":
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'detection_summary': self.get_detection_summary(),
                'performance_summary': self.get_performance_summary(),
                'event_counts': self.event_counts,
                'performance_metrics': {
                    'total_detections': self.performance_metrics['total_detections'],
                    'latency_count': len(self.performance_metrics['detection_latencies']),
                    'accuracy_count': len(self.performance_metrics['accuracy_scores']),
                    'f1_score_count': len(self.performance_metrics['f1_scores'])
                }
            }
            
            export_filename = f"jamming_detection_export_{timestamp}.json"
            export_path = os.path.join(self.log_dir, export_filename)
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return export_path
        
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
    
    def reset_counters(self):
        """Reset all counters and metrics."""
        self.event_counts = {key: 0 for key in self.event_counts}
        self.performance_metrics = {
            'total_detections': 0,
            'detection_latencies': [],
            'accuracy_scores': [],
            'f1_scores': []
        }
        
        self.log_system_event(
            event_type="system_reset",
            message="Counters and metrics reset"
        )
