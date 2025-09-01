#!/usr/bin/env python3
"""
High-Accuracy USRP Jamming Detection Application
Advanced version using CatBoost ensemble for >99.75% accuracy

This is the new high-performance version that uses:
- CatBoost (55% weight) - State-of-the-art gradient boosting
- LightGBM (30% weight) - Fast gradient boosting  
- Extra Trees (15% weight) - Randomized decision trees

Designed specifically for realistic USRP hardware scenarios
with industry-standard feature engineering.
"""

import argparse
import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import numpy as np
    import pandas as pd
    from train_catboost_ensemble import HighAccuracyCatBoostEnsemble, train_high_accuracy_model
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False


class HighAccuracyJammingDetector:
    """
    High-accuracy jamming detection application
    Uses CatBoost ensemble for superior performance
    """
    
    def __init__(self, model_path: str = "saved_models/high_accuracy_focused.joblib"):
        """Initialize the High-Accuracy Jamming Detector
        
        Args:
            model_path: Path to the saved high-accuracy CatBoost ensemble model
        """
        self.model_path = model_path
        self.ensemble = None
        self.performance_log = []  # Track performance metrics
        
        # Load model if it exists
        if os.path.exists(model_path):
            self.load_model()
    
    def load_model(self):
        """Load the trained CatBoost ensemble"""
        if not self.model_path:
            raise ValueError("Model path not specified")
        
        print(f"Loading high-accuracy CatBoost ensemble from: {self.model_path}")
        self.ensemble = HighAccuracyCatBoostEnsemble()
        self.ensemble.load_model(self.model_path)
        print("✅ High-accuracy model loaded successfully")
    
    def train_new_model(self, normal_path: str, jamming_path: str, save_path: str):
        """Train a new high-accuracy model"""
        
        print("Training new high-accuracy CatBoost ensemble...")
        results = train_high_accuracy_model(normal_path, jamming_path, save_path)
        
        self.ensemble = results['ensemble']
        self.model_path = save_path
        
        print(f"✅ New high-accuracy model trained and saved to: {save_path}")
        return results
    
    def detect_jamming(self, features: np.ndarray) -> Dict[str, Any]:
        """Detect jamming with high accuracy"""
        
        if self.ensemble is None:
            raise ValueError("Model not loaded. Please load or train a model first.")
        
        start_time = time.time()
        
        # Make prediction
        prediction = self.ensemble.predict(features.reshape(1, -1))[0]
        probabilities = self.ensemble.predict_proba(features.reshape(1, -1))[0]
        
        detection_time = (time.time() - start_time) * 1000  # ms
        
        # Create result
        result = {
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(self.ensemble.label_encoder.classes_, probabilities)
            },
            'detection_time_ms': detection_time,
            'is_jamming': prediction != 'normal',
            'confidence': float(max(probabilities))
        }
        
        # Log performance
        self.performance_log.append({
            'timestamp': result['timestamp'],
            'detection_time_ms': detection_time,
            'prediction': prediction,
            'confidence': result['confidence']
        })
        
        return result
    
    def run_real_time_detection(self, duration_seconds: int = 60):
        """Run real-time jamming detection simulation"""
        
        if self.ensemble is None:
            raise ValueError("Model not loaded. Please load or train a model first.")
        
        print(f"🚀 Starting high-accuracy real-time jamming detection")
        print(f"Duration: {duration_seconds} seconds")
        print("=" * 60)
        
        start_time = time.time()
        detection_count = 0
        jamming_detections = 0
        
        while (time.time() - start_time) < duration_seconds:
            # Simulate realistic USRP features
            features = self._generate_realistic_features()
            
            # Detect jamming
            result = self.detect_jamming(features)
            detection_count += 1
            
            if result['is_jamming']:
                jamming_detections += 1
                print(f"🚨 JAMMING DETECTED: {result['prediction']} "
                      f"(Confidence: {result['confidence']:.3f}, "
                      f"Time: {result['detection_time_ms']:.1f}ms)")
            else:
                if detection_count % 10 == 0:  # Print every 10th normal detection
                    print(f"✅ Normal operation "
                          f"(Confidence: {result['confidence']:.3f}, "
                          f"Time: {result['detection_time_ms']:.1f}ms)")
            
            # Sleep to simulate real-time operation
            time.sleep(0.1)  # 100ms intervals
        
        # Performance summary
        elapsed_time = time.time() - start_time
        avg_detection_time = np.mean([log['detection_time_ms'] for log in self.performance_log[-detection_count:]])
        
        print(f"\n📊 Real-time Detection Summary")
        print("=" * 40)
        print(f"Duration: {elapsed_time:.1f} seconds")
        print(f"Total detections: {detection_count}")
        print(f"Jamming detections: {jamming_detections}")
        print(f"Normal detections: {detection_count - jamming_detections}")
        print(f"Average detection time: {avg_detection_time:.1f}ms")
        print(f"Detection rate: {detection_count/elapsed_time:.1f} detections/second")
    
    def _generate_realistic_features(self) -> np.ndarray:
        """Generate realistic USRP features for simulation"""
        
        # Simulate different scenarios randomly
        scenario = np.random.choice(['normal', 'power_jamming', 'sweep_jamming', 'reactive_jamming'], 
                                  p=[0.7, 0.1, 0.1, 0.1])  # 70% normal, 30% jamming
        
        if scenario == 'normal':
            # Normal operation features
            features = np.array([
                np.random.normal(-30, 5),     # rsrp_dbm
                np.random.normal(-10, 3),     # rsrq_db
                np.random.normal(20, 5),      # sinr_db
                np.random.normal(-25, 4),     # rssi_dbm
                np.random.rayleigh(1.0),      # channel_state_info
                np.random.uniform(0, 50),     # doppler_spread_hz
                np.random.exponential(100),   # delay_spread_ns
                np.random.uniform(1e6, 10e6), # coherence_bandwidth_hz
                np.random.normal(-95, 3),     # interference_power_dbm
                np.random.uniform(0.01, 0.1), # adjacent_channel_power
                np.random.uniform(-80, -40),  # spurious_emissions
                np.random.uniform(-60, -20),  # intermodulation_distortion
                np.random.normal(-20, 3),     # power_spectral_density
                np.random.uniform(0.7, 0.95), # autocorrelation_peak
                np.random.uniform(0.8, 0.95), # cross_correlation
                np.random.uniform(4, 6),      # signal_entropy
                np.random.uniform(0.01, 0.1), # iq_imbalance_magnitude
                np.random.normal(0, 0.01),    # dc_offset_i
                np.random.normal(0, 0.01),    # dc_offset_q
                np.random.normal(-85, 5),     # phase_noise_integrated
                np.random.uniform(2.4e9, 5.8e9), # spectral_centroid
                np.random.uniform(2.5e9, 6.0e9), # spectral_rolloff
                np.random.exponential(0.3),   # spectral_flux
                np.random.uniform(0.2, 0.4),  # zero_crossing_rate
                np.random.uniform(0.6, 0.9),  # signal_complexity
                np.random.uniform(0.4, 0.6),  # hurst_exponent
                np.random.uniform(1.3, 1.7)   # fractal_dimension
            ])
        
        elif scenario == 'power_jamming':
            # Power jamming features (degraded performance)
            features = np.array([
                np.random.normal(-20, 8),     # rsrp_dbm (higher, more variable)
                np.random.normal(-15, 5),     # rsrq_db (degraded)
                np.random.normal(5, 8),       # sinr_db (low SINR)
                np.random.normal(-15, 6),     # rssi_dbm (elevated)
                np.random.rayleigh(0.5),      # channel_state_info (degraded)
                np.random.uniform(10, 200),   # doppler_spread_hz (high)
                np.random.exponential(300),   # delay_spread_ns (high)
                np.random.uniform(0.5e6, 5e6), # coherence_bandwidth_hz (low)
                np.random.normal(-75, 5),     # interference_power_dbm (high)
                np.random.uniform(0.3, 0.8),  # adjacent_channel_power (high)
                np.random.uniform(-60, -30),  # spurious_emissions (high)
                np.random.uniform(-40, -10),  # intermodulation_distortion (high)
                np.random.normal(-10, 5),     # power_spectral_density (elevated)
                np.random.uniform(0.2, 0.6),  # autocorrelation_peak (low)
                np.random.uniform(0.3, 0.6),  # cross_correlation (low)
                np.random.uniform(2, 4),      # signal_entropy (low)
                np.random.uniform(0.1, 0.5),  # iq_imbalance_magnitude (high)
                np.random.normal(0, 0.05),    # dc_offset_i (high)
                np.random.normal(0, 0.05),    # dc_offset_q (high)
                np.random.normal(-75, 8),     # phase_noise_integrated (degraded)
                np.random.uniform(2.3e9, 6.0e9), # spectral_centroid
                np.random.uniform(2.4e9, 6.2e9), # spectral_rolloff
                np.random.exponential(1.0),   # spectral_flux (high)
                np.random.uniform(0.4, 0.7),  # zero_crossing_rate (high)
                np.random.uniform(0.2, 0.5),  # signal_complexity (low)
                np.random.uniform(0.1, 0.4),  # hurst_exponent (low)
                np.random.uniform(1.1, 1.5)   # fractal_dimension (low)
            ])
        
        else:  # sweep_jamming or reactive_jamming
            # Frequency-selective or adaptive jamming
            features = np.array([
                np.random.normal(-25, 10),    # rsrp_dbm (highly variable)
                np.random.normal(-12, 6),     # rsrq_db (variable)
                np.random.normal(10, 10),     # sinr_db (variable)
                np.random.normal(-20, 8),     # rssi_dbm (variable)
                np.random.rayleigh(0.7),      # channel_state_info
                np.random.uniform(20, 150),   # doppler_spread_hz
                np.random.exponential(200),   # delay_spread_ns
                np.random.uniform(1e6, 8e6),  # coherence_bandwidth_hz
                np.random.normal(-85, 8),     # interference_power_dbm
                np.random.uniform(0.1, 0.5),  # adjacent_channel_power
                np.random.uniform(-70, -35),  # spurious_emissions
                np.random.uniform(-50, -15),  # intermodulation_distortion
                np.random.normal(-15, 4),     # power_spectral_density
                np.random.uniform(0.3, 0.7),  # autocorrelation_peak
                np.random.uniform(0.4, 0.7),  # cross_correlation
                np.random.uniform(2.5, 4.5),  # signal_entropy
                np.random.uniform(0.05, 0.3), # iq_imbalance_magnitude
                np.random.normal(0, 0.03),    # dc_offset_i
                np.random.normal(0, 0.03),    # dc_offset_q
                np.random.normal(-80, 6),     # phase_noise_integrated
                np.random.uniform(2.35e9, 5.9e9), # spectral_centroid
                np.random.uniform(2.45e9, 6.1e9), # spectral_rolloff
                np.random.exponential(0.7),   # spectral_flux
                np.random.uniform(0.3, 0.6),  # zero_crossing_rate
                np.random.uniform(0.3, 0.7),  # signal_complexity
                np.random.uniform(0.2, 0.8),  # hurst_exponent
                np.random.uniform(1.15, 1.6)  # fractal_dimension
            ])
        
        return features
    
    def run_interactive_demo(self):
        """Run interactive demonstration"""
        
        if self.ensemble is None:
            raise ValueError("Model not loaded. Please load or train a model first.")
        
        print(f"🎮 Interactive High-Accuracy Jamming Detection Demo")
        print("=" * 60)
        print("Commands:")
        print("  'normal' - Test normal operation scenario")
        print("  'power' - Test power jamming scenario")
        print("  'sweep' - Test sweep jamming scenario")
        print("  'reactive' - Test reactive jamming scenario")
        print("  'random' - Test random scenario")
        print("  'auto' - Run automatic detection sequence")
        print("  'quit' - Exit demo")
        print()
        
        while True:
            try:
                command = input("Enter command: ").strip().lower()
                
                if command == 'quit':
                    break
                
                elif command in ['normal', 'power', 'sweep', 'reactive']:
                    # Generate specific scenario features
                    scenario_map = {
                        'normal': 'normal',
                        'power': 'power_jamming',
                        'sweep': 'sweep_jamming',
                        'reactive': 'reactive_jamming'
                    }
                    
                    # Temporarily set scenario for feature generation
                    original_choice = np.random.choice
                    np.random.choice = lambda *args, **kwargs: scenario_map[command]
                    features = self._generate_realistic_features()
                    np.random.choice = original_choice
                    
                    result = self.detect_jamming(features)
                    
                    print(f"\n🔍 Detection Result:")
                    print(f"  Predicted: {result['prediction']}")
                    print(f"  Confidence: {result['confidence']:.3f}")
                    print(f"  Detection Time: {result['detection_time_ms']:.1f}ms")
                    print(f"  Is Jamming: {result['is_jamming']}")
                    
                    if result['is_jamming']:
                        print(f"  🚨 JAMMING DETECTED!")
                    else:
                        print(f"  ✅ Normal Operation")
                    print()
                
                elif command == 'random':
                    features = self._generate_realistic_features()
                    result = self.detect_jamming(features)
                    
                    print(f"\n🎲 Random Detection Result:")
                    print(f"  Predicted: {result['prediction']}")
                    print(f"  Confidence: {result['confidence']:.3f}")
                    print(f"  Detection Time: {result['detection_time_ms']:.1f}ms")
                    print(f"  Is Jamming: {result['is_jamming']}")
                    print()
                
                elif command == 'auto':
                    print("\n🤖 Running automatic detection sequence...")
                    scenarios = ['normal', 'power', 'sweep', 'reactive']
                    
                    for scenario in scenarios:
                        # Generate scenario features
                        scenario_map = {
                            'normal': 'normal',
                            'power': 'power_jamming',
                            'sweep': 'sweep_jamming',
                            'reactive': 'reactive_jamming'
                        }
                        
                        original_choice = np.random.choice
                        np.random.choice = lambda *args, **kwargs: scenario_map[scenario]
                        features = self._generate_realistic_features()
                        np.random.choice = original_choice
                        
                        result = self.detect_jamming(features)
                        
                        status = "🚨 JAMMING" if result['is_jamming'] else "✅ NORMAL"
                        print(f"  {scenario.upper()}: {result['prediction']} "
                              f"(Conf: {result['confidence']:.3f}) {status}")
                    print()
                
                else:
                    print("Unknown command. Type 'quit' to exit.")
            
            except KeyboardInterrupt:
                print("\nDemo interrupted.")
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("Demo ended.")


def main():
    """Main application entry point"""
    
    parser = argparse.ArgumentParser(
        description="High-Accuracy USRP Jamming Detection Application (CatBoost Ensemble)"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train new high-accuracy model')
    train_parser.add_argument('--normal', required=True, help='Path to normal traffic CSV')
    train_parser.add_argument('--jamming', required=True, help='Path to jamming attacks CSV')
    train_parser.add_argument('--output', required=True, help='Output path for trained model')
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Run real-time detection')
    detect_parser.add_argument('--model', required=True, help='Path to trained model')
    detect_parser.add_argument('--duration', type=int, default=60, help='Duration in seconds')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run interactive demo')
    demo_parser.add_argument('--model', required=True, help='Path to trained model')
    
    # Generate dataset command
    generate_parser = subparsers.add_parser('generate', help='Generate realistic USRP dataset')
    
    args = parser.parse_args()
    
    if not DEPENDENCIES_AVAILABLE:
        print("❌ Required dependencies not available. Please install:")
        print("pip install numpy pandas scikit-learn catboost lightgbm matplotlib seaborn joblib")
        return
    
    try:
        if args.command == 'train':
            # Train new model
            detector = HighAccuracyJammingDetector()
            results = detector.train_new_model(args.normal, args.jamming, args.output)
            
            print(f"\n✅ Training completed successfully!")
            print(f"Model saved to: {args.output}")
            
            # Print training summary
            eval_results = results['evaluation_results']
            print(f"\nModel Performance:")
            print(f"  Overall Accuracy: {eval_results['overall_metrics']['accuracy']:.4f}")
            print(f"  F1-Score: {eval_results['overall_metrics']['f1_weighted']:.4f}")
            
            if 'power_jamming_metrics' in eval_results and eval_results['power_jamming_metrics']:
                power_metrics = eval_results['power_jamming_metrics']
                print(f"  Power Jamming F1: {power_metrics['power_jamming_f1']:.4f}")
        
        elif args.command == 'detect':
            # Run real-time detection
            detector = HighAccuracyJammingDetector(args.model)
            detector.run_real_time_detection(args.duration)
        
        elif args.command == 'demo':
            # Run interactive demo
            detector = HighAccuracyJammingDetector(args.model)
            detector.run_interactive_demo()
        
        elif args.command == 'generate':
            # Generate realistic dataset
            print("Generating realistic USRP dataset...")
            from generate_realistic_usrp_dataset import main as generate_main
            generate_main()
        
        else:
            parser.print_help()
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
