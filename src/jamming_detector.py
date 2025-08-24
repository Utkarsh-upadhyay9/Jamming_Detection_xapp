"""
Jamming Detection xApp for O-RAN integration.
Implements real-time jamming detection using ensemble ML models.
"""

import numpy as np
import pandas as pd
import time
import threading
import queue
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import json

from src.ensemble_model import EnsembleJammingDetector
from src.data_processor import JammingDataProcessor
from utils.logger import JammingDetectionLogger
from utils.metrics import LatencyTracker
from config.model_config import PERFORMANCE_REQUIREMENTS
from config.network_config import E2_INTERFACE_CONFIG

@dataclass
class E2Metrics:
    """Data structure for E2 interface metrics."""
    timestamp: datetime
    sinr_mean: float
    sinr_std: float
    rsrp_mean: float
    rsrp_std: float
    rsrq_mean: float
    ul_throughput: float
    dl_throughput: float
    retx_rate: float
    packet_loss_rate: float
    inter_arrival_variance: float
    buffer_occupancy: float
    prb_utilization_ul: float
    prb_utilization_dl: float
    grant_count: float
    cqi_variance: float

@dataclass
class DetectionResult:
    """Data structure for detection results."""
    timestamp: datetime
    is_jamming: bool
    jamming_type: str
    confidence: float
    latency_ms: float
    features: np.ndarray
    ensemble_scores: Dict[str, float]

class E2InterfaceSimulator:
    """Simulates E2 interface for collecting MAC layer metrics."""
    
    def __init__(self, data_queue: queue.Queue, config: Dict[str, Any] = None):
        """
        Initialize E2 interface simulator.
        
        Args:
            data_queue: Queue for passing metrics to xApp
            config: E2 interface configuration
        """
        self.data_queue = data_queue
        self.config = config or E2_INTERFACE_CONFIG
        self.is_running = False
        self.thread = None
        
        # Simulation parameters
        self.base_metrics = self._get_default_metrics()
        self.jamming_simulation = False
        self.current_jamming_type = None
        
    def _get_default_metrics(self) -> Dict[str, float]:
        """Get default normal network metrics."""
        return {
            'sinr_mean': 15.0,
            'sinr_std': 2.0,
            'rsrp_mean': -90.0,
            'rsrp_std': 5.0,
            'rsrq_mean': -10.0,
            'ul_throughput': 5000000.0,  # 5 Mbps
            'dl_throughput': 10000000.0,  # 10 Mbps
            'retx_rate': 0.02,
            'packet_loss_rate': 0.001,
            'inter_arrival_variance': 0.05,
            'buffer_occupancy': 0.3,
            'prb_utilization_ul': 0.4,
            'prb_utilization_dl': 0.5,
            'grant_count': 45.0,
            'cqi_variance': 0.8
        }
    
    def start_monitoring(self):
        """Start E2 metrics collection."""
        if self.is_running:
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._monitoring_loop)
        self.thread.daemon = True
        self.thread.start()
        print("E2 interface monitoring started")
    
    def stop_monitoring(self):
        """Stop E2 metrics collection."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        print("E2 interface monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        interval = self.config['reporting_interval_ms'] / 1000.0
        
        while self.is_running:
            try:
                # Generate current metrics
                metrics = self._generate_current_metrics()
                
                # Create E2Metrics object
                e2_metrics = E2Metrics(
                    timestamp=datetime.now(),
                    **metrics
                )
                
                # Send to xApp
                try:
                    self.data_queue.put(e2_metrics, timeout=0.1)
                except queue.Full:
                    print("Warning: E2 metrics queue is full")
                
                time.sleep(interval)
                
            except Exception as e:
                print(f"Error in E2 monitoring loop: {e}")
                time.sleep(interval)
    
    def _generate_current_metrics(self) -> Dict[str, float]:
        """Generate current network metrics based on simulation state."""
        metrics = self.base_metrics.copy()
        
        # Add normal variations
        for key in metrics:
            if key in ['ul_throughput', 'dl_throughput']:
                metrics[key] += np.random.normal(0, metrics[key] * 0.1)
            else:
                metrics[key] += np.random.normal(0, abs(metrics[key]) * 0.05)
        
        # Apply jamming effects if simulation is active
        if self.jamming_simulation:
            metrics = self._apply_jamming_effects(metrics)
        
        return metrics
    
    def _apply_jamming_effects(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Apply jamming effects to metrics."""
        if self.current_jamming_type == 'power_jamming':
            # Power jamming: severe SINR degradation, high retx/loss rates
            metrics['sinr_mean'] *= 0.1  # Severe degradation
            metrics['sinr_std'] *= 5.0   # High variability
            metrics['rsrp_std'] *= 3.0   # High RSRP variability
            metrics['retx_rate'] = min(0.8, metrics['retx_rate'] * 10)
            metrics['packet_loss_rate'] = min(0.5, metrics['packet_loss_rate'] * 50)
            metrics['ul_throughput'] *= 0.1
            metrics['dl_throughput'] *= 0.1
            
        elif self.current_jamming_type == 'sweep_jamming':
            # Sweep jamming: periodic interference patterns
            phase = (time.time() * 10) % (2 * np.pi)  # 10 Hz sweep
            interference_factor = 0.5 + 0.5 * np.sin(phase)
            
            metrics['sinr_mean'] *= (0.5 + 0.5 * interference_factor)
            metrics['rsrp_std'] *= (1 + 2 * interference_factor)
            metrics['retx_rate'] = min(0.3, metrics['retx_rate'] * (1 + 5 * interference_factor))
            metrics['cqi_variance'] *= (1 + 3 * interference_factor)
            
        elif self.current_jamming_type == 'intelligent_jamming':
            # Intelligent jamming: adaptive, traffic-aware interference
            traffic_factor = (metrics['ul_throughput'] + metrics['dl_throughput']) / 15000000.0
            adaptation = min(1.0, traffic_factor)
            
            metrics['sinr_mean'] *= (1 - 0.3 * adaptation)
            metrics['retx_rate'] = min(0.15, metrics['retx_rate'] * (1 + 3 * adaptation))
            metrics['packet_loss_rate'] = min(0.05, metrics['packet_loss_rate'] * (1 + 10 * adaptation))
            metrics['inter_arrival_variance'] *= (1 + 2 * adaptation)
        
        return metrics
    
    def simulate_jamming(self, jamming_type: str, duration_seconds: float = 10.0):
        """
        Simulate jamming attack.
        
        Args:
            jamming_type: Type of jamming ('power_jamming', 'sweep_jamming', 'intelligent_jamming')
            duration_seconds: Duration of attack
        """
        self.current_jamming_type = jamming_type
        self.jamming_simulation = True
        
        print(f"Simulating {jamming_type} for {duration_seconds} seconds")
        
        # Schedule end of jamming
        def end_jamming():
            time.sleep(duration_seconds)
            self.jamming_simulation = False
            self.current_jamming_type = None
            print(f"Jamming simulation ended")
        
        threading.Thread(target=end_jamming, daemon=True).start()

class JammingDetectionXApp:
    """
    Main jamming detection xApp for O-RAN integration.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize jamming detection xApp.
        
        Args:
            model_path: Path to pre-trained model (optional)
        """
        # Core components
        self.ensemble_detector = EnsembleJammingDetector()
        self.data_processor = JammingDataProcessor()
        self.logger = JammingDetectionLogger()
        self.latency_tracker = LatencyTracker(
            target_latency_ms=PERFORMANCE_REQUIREMENTS['max_latency_ms']
        )
        
        # Communication queues
        self.e2_metrics_queue = queue.Queue(maxsize=100)
        self.detection_results_queue = queue.Queue(maxsize=50)
        
        # E2 interface simulator
        self.e2_interface = E2InterfaceSimulator(self.e2_metrics_queue)
        
        # Processing threads
        self.detection_thread = None
        self.ric_communication_thread = None
        
        # State management
        self.is_running = False
        self.is_trained = False
        
        # Performance monitoring
        self.detection_count = 0
        self.jamming_detections = 0
        self.false_alarms = 0
        
        # Load pre-trained model if provided
        if model_path:
            self.load_model(model_path)
    
    def train_model(self, normal_traffic_path: str, jamming_attacks_path: str) -> Dict[str, float]:
        """
        Train the ensemble model.
        
        Args:
            normal_traffic_path: Path to normal traffic data
            jamming_attacks_path: Path to jamming attacks data
            
        Returns:
            Training metrics
        """
        self.logger.log_system_event("xapp_training", "Starting xApp model training")
        
        try:
            training_metrics = self.ensemble_detector.train_ensemble(
                normal_traffic_path=normal_traffic_path,
                jamming_attacks_path=jamming_attacks_path
            )
            
            self.is_trained = True
            
            self.logger.log_system_event(
                "xapp_training_completed",
                "xApp model training completed successfully",
                additional_data=training_metrics
            )
            
            return training_metrics
            
        except Exception as e:
            self.logger.log_error("training_error", str(e), e)
            raise
    
    def start_monitoring(self):
        """Start real-time jamming detection monitoring."""
        if not self.is_trained:
            raise ValueError("Model must be trained before starting monitoring")
        
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start E2 interface monitoring
        self.e2_interface.start_monitoring()
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        # Start RIC communication thread
        self.ric_communication_thread = threading.Thread(target=self._ric_communication_loop)
        self.ric_communication_thread.daemon = True
        self.ric_communication_thread.start()
        
        self.logger.log_system_event("xapp_monitoring_started", "xApp monitoring started")
        print("Jamming detection xApp monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.is_running = False
        
        # Stop E2 interface
        self.e2_interface.stop_monitoring()
        
        # Wait for threads to finish
        if self.detection_thread:
            self.detection_thread.join(timeout=2.0)
        if self.ric_communication_thread:
            self.ric_communication_thread.join(timeout=2.0)
        
        self.logger.log_system_event("xapp_monitoring_stopped", "xApp monitoring stopped")
        print("Jamming detection xApp monitoring stopped")
    
    def _detection_loop(self):
        """Main detection processing loop."""
        while self.is_running:
            try:
                # Get E2 metrics
                e2_metrics = self.e2_metrics_queue.get(timeout=1.0)
                
                # Process detection
                result = self._process_detection(e2_metrics)
                
                # Send result to RIC communication queue
                try:
                    self.detection_results_queue.put(result, timeout=0.1)
                except queue.Full:
                    print("Warning: Detection results queue is full")
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.log_error("detection_loop_error", str(e), e)
                time.sleep(0.1)
    
    def _process_detection(self, e2_metrics: E2Metrics) -> DetectionResult:
        """
        Process jamming detection for given E2 metrics.
        
        Args:
            e2_metrics: E2 interface metrics
            
        Returns:
            Detection result
        """
        start_time = time.perf_counter()
        
        # Convert E2 metrics to feature vector
        features = self._e2_metrics_to_features(e2_metrics)
        
        # Make ensemble prediction
        probabilities = self.ensemble_detector.predict_proba(features.reshape(1, -1))
        jamming_probability = probabilities[0, 1]
        
        # Determine if jamming is detected
        is_jamming = jamming_probability > self.ensemble_detector.threshold
        
        # Multi-class classification if jamming is detected
        if is_jamming:
            jamming_types = self.ensemble_detector.detect_jamming_type(features.reshape(1, -1))
            jamming_type = jamming_types[0]
        else:
            jamming_type = 'normal'
        
        # Calculate confidence
        confidence = self.ensemble_detector.calculate_ensemble_confidence(features.reshape(1, -1))[0]
        
        # Calculate processing latency
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.latency_tracker.add_measurement(latency_ms)
        
        # Update counters
        self.detection_count += 1
        if is_jamming:
            self.jamming_detections += 1
        
        # Create result
        result = DetectionResult(
            timestamp=e2_metrics.timestamp,
            is_jamming=is_jamming,
            jamming_type=jamming_type,
            confidence=confidence,
            latency_ms=latency_ms,
            features=features,
            ensemble_scores={
                'jamming_probability': jamming_probability,
                'normal_probability': probabilities[0, 0]
            }
        )
        
        # Log detection event
        self.logger.log_detection_event({
            'predicted_class': jamming_type,
            'confidence': confidence,
            'latency_ms': latency_ms,
            'features': features.tolist(),
            'ensemble_scores': result.ensemble_scores
        })
        
        return result
    
    def _e2_metrics_to_features(self, e2_metrics: E2Metrics) -> np.ndarray:
        """
        Convert E2Metrics to feature vector.
        
        Args:
            e2_metrics: E2 interface metrics
            
        Returns:
            Feature vector (15 features)
        """
        features = np.array([
            e2_metrics.sinr_mean,
            e2_metrics.sinr_std,
            e2_metrics.rsrp_mean,
            e2_metrics.rsrp_std,
            e2_metrics.rsrq_mean,
            e2_metrics.ul_throughput,
            e2_metrics.dl_throughput,
            e2_metrics.retx_rate,
            e2_metrics.packet_loss_rate,
            e2_metrics.inter_arrival_variance,
            e2_metrics.buffer_occupancy,
            e2_metrics.prb_utilization_ul,
            e2_metrics.prb_utilization_dl,
            e2_metrics.grant_count,
            e2_metrics.cqi_variance
        ])
        
        return features
    
    def _ric_communication_loop(self):
        """RIC communication loop for sending alerts and receiving commands."""
        while self.is_running:
            try:
                # Get detection result
                result = self.detection_results_queue.get(timeout=1.0)
                
                # Send to RIC if jamming is detected
                if result.is_jamming:
                    self._send_ric_alert(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.log_error("ric_communication_error", str(e), e)
                time.sleep(0.1)
    
    def _send_ric_alert(self, result: DetectionResult):
        """
        Send jamming alert to RIC.
        
        Args:
            result: Detection result
        """
        alert_payload = {
            'timestamp': result.timestamp.isoformat(),
            'alert_type': 'jamming_detection',
            'jamming_type': result.jamming_type,
            'confidence': result.confidence,
            'latency_ms': result.latency_ms,
            'recommended_actions': self._get_recommended_actions(result.jamming_type)
        }
        
        # Log RIC communication
        self.logger.log_ric_communication('jamming_alert', alert_payload)
        
        # In a real implementation, this would send via E2 interface to RIC
        print(f"RIC Alert: {result.jamming_type} detected with confidence {result.confidence:.3f}")
    
    def _get_recommended_actions(self, jamming_type: str) -> List[str]:
        """
        Get recommended actions for detected jamming type.
        
        Args:
            jamming_type: Type of detected jamming
            
        Returns:
            List of recommended actions
        """
        actions = {
            'power_jamming': [
                'Increase transmission power',
                'Switch to different frequency band',
                'Activate interference cancellation',
                'Implement frequency hopping'
            ],
            'sweep_jamming': [
                'Implement adaptive frequency allocation',
                'Activate anti-jamming protocols',
                'Switch to frequency hopping mode',
                'Increase error correction coding'
            ],
            'intelligent_jamming': [
                'Randomize transmission patterns',
                'Implement traffic obfuscation',
                'Switch to low-latency mode',
                'Activate enhanced security protocols'
            ]
        }
        
        return actions.get(jamming_type, ['Monitor and assess situation'])
    
    def simulate_jamming_attack(self, jamming_type: str, duration: float = 10.0):
        """
        Simulate a jamming attack for testing.
        
        Args:
            jamming_type: Type of jamming attack
            duration: Duration in seconds
        """
        if not self.is_running:
            print("xApp must be running to simulate attacks")
            return
        
        self.e2_interface.simulate_jamming(jamming_type, duration)
        
        self.logger.log_system_event(
            "jamming_simulation",
            f"Simulating {jamming_type} attack",
            additional_data={'duration': duration, 'type': jamming_type}
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get current performance summary.
        
        Returns:
            Performance statistics
        """
        latency_stats = self.latency_tracker.get_statistics()
        detection_summary = self.logger.get_detection_summary()
        
        summary = {
            'total_detections': self.detection_count,
            'jamming_detections': self.jamming_detections,
            'jamming_detection_rate': self.jamming_detections / max(self.detection_count, 1),
            'latency_compliance': latency_stats.get('target_compliance_rate', 1.0),
            'mean_latency_ms': latency_stats.get('mean_latency_ms', 0),
            'p95_latency_ms': latency_stats.get('p95_latency_ms', 0),
            'detection_summary': detection_summary,
            'is_running': self.is_running,
            'is_trained': self.is_trained
        }
        
        return summary
    
    def save_model(self, model_path: str):
        """
        Save the trained model.
        
        Args:
            model_path: Path to save model
        """
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        self.ensemble_detector.save_ensemble(model_path)
        self.logger.log_system_event("model_saved", f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """
        Load a pre-trained model.
        
        Args:
            model_path: Path to load model from
        """
        self.ensemble_detector.load_ensemble(model_path)
        self.is_trained = True
        self.logger.log_system_event("model_loaded", f"Model loaded from {model_path}")
    
    def reset_performance_counters(self):
        """Reset all performance counters."""
        self.detection_count = 0
        self.jamming_detections = 0
        self.false_alarms = 0
        self.latency_tracker = LatencyTracker(
            target_latency_ms=PERFORMANCE_REQUIREMENTS['max_latency_ms']
        )
        self.logger.reset_counters()
        
        self.logger.log_system_event("counters_reset", "Performance counters reset")
