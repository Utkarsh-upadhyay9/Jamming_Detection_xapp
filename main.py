import argparse
import os
import sys
import time
from typing import Optional

sys.path.append(os.path.dirname(__file__))

from src.jamming_detector import JammingDetectionXApp
from src.ensemble_model import EnsembleJammingDetector
from tests.test_performance import PerformanceEvaluator
from utils.visualization import JammingDetectionVisualizer

def train_model(normal_path: str, jamming_path: str, model_save_path: str) -> None:
    print("Training Ensemble Jamming Detection Model")
    print("=" * 50)
    
    detector = EnsembleJammingDetector()
    
    training_metrics = detector.train_ensemble(
        normal_traffic_path=normal_path,
        jamming_attacks_path=jamming_path
    )
    
    detector.save_ensemble(model_save_path)
    
    print(f"\nTraining completed!")
    print(f"Model saved to: {model_save_path}")
    print(f"Training F1-Score: {training_metrics['ensemble']['f1_score']:.4f}")
    print(f"Training Accuracy: {training_metrics['ensemble']['accuracy']:.4f}")

def run_real_time_detection(model_path: str, duration: int = 60) -> None:
    print(f"Starting Real-time Jamming Detection ({duration}s)")
    print("=" * 50)
    
    xapp = JammingDetectionXApp(model_path)
    
    try:
        xapp.start_monitoring()
        print("✓ xApp monitoring started")
        print("✓ E2 interface simulator active")
        print("✓ RIC communication established")
        
        print("\nScheduled jamming attacks:")
        attack_schedule = [
            (10, 'power_jamming', 5, "High-power broadband interference"),
            (25, 'sweep_jamming', 6, "Frequency-sweeping interference"),
            (45, 'intelligent_jamming', 8, "Adaptive traffic-aware interference")
        ]
        
        for start_time, attack_type, attack_duration, description in attack_schedule:
            print(f"  {start_time}s: {attack_type} ({attack_duration}s) - {description}")
        
        print(f"\nMonitoring for {duration} seconds...")
        print("Press Ctrl+C to stop early\n")
        
        import threading
        for start_time, attack_type, attack_duration, _ in attack_schedule:
            def schedule_attack(delay, jamming_type, jam_duration):
                time.sleep(delay)
                print(f"\n🚨 ATTACK: {jamming_type} started ({jam_duration}s)")
                xapp.simulate_jamming_attack(jamming_type, jam_duration)
            
            threading.Thread(
                target=schedule_attack, 
                args=(start_time, attack_type, attack_duration),
                daemon=True
            ).start()
        
        start_time = time.time()
        last_summary_time = 0
        
        while time.time() - start_time < duration:
            current_time = time.time() - start_time
            
            if current_time - last_summary_time >= 10:
                summary = xapp.get_performance_summary()
                print(f"\n--- Status Update ({current_time:.0f}s) ---")
                print(f"Total Detections: {summary['total_detections']}")
                print(f"Jamming Detected: {summary['jamming_detections']}")
                print(f"Detection Rate: {summary['jamming_detection_rate']:.1%}")
                print(f"Avg Latency: {summary['mean_latency_ms']:.1f}ms")
                print(f"Latency Compliance: {summary['latency_compliance']:.1%}")
                
                last_summary_time = current_time
            
            time.sleep(1)
        
        final_summary = xapp.get_performance_summary()
        print(f"\n" + "=" * 50)
        print("FINAL PERFORMANCE SUMMARY")
        print("=" * 50)
        print(f"Total Detections: {final_summary['total_detections']}")
        print(f"Jamming Detections: {final_summary['jamming_detections']}")
        print(f"Overall Detection Rate: {final_summary['jamming_detection_rate']:.1%}")
        print(f"Average Latency: {final_summary['mean_latency_ms']:.1f}ms")
        print(f"P95 Latency: {final_summary['p95_latency_ms']:.1f}ms")
        print(f"Latency Compliance: {final_summary['latency_compliance']:.1%}")
        
    except KeyboardInterrupt:
        print("\nStopping monitoring...")
    finally:
        xapp.stop_monitoring()
        print("✓ xApp stopped")

def run_performance_evaluation(normal_path: str, jamming_path: str) -> None:
    print("Running Comprehensive Performance Evaluation")
    print("=" * 50)
    
    evaluator = PerformanceEvaluator()
    results = evaluator.run_comprehensive_evaluation(normal_path, jamming_path)
    
    if results:
        print("\n✓ Evaluation completed successfully!")
        print(f"✓ Performance report generated")
        print(f"✓ All visualization plots created")
    else:
        print("\n✗ Evaluation failed!")

def demonstrate_xapp_capabilities(model_path: str = None) -> None:
    print("Jamming Detection xApp Demonstration")
    print("=" * 50)
    
    if model_path and os.path.exists(model_path):
        xapp = JammingDetectionXApp(model_path)
        print("✓ Loaded pre-trained model")
    else:
        print("Training model for demonstration...")
        xapp = JammingDetectionXApp()
        
        normal_path = "Ensemble_ML_Jamming_detection_dataset/dataset/normal_traffic.csv"
        jamming_path = "Ensemble_ML_Jamming_detection_dataset/dataset/jamming_attacks.csv"
        
        if os.path.exists(normal_path) and os.path.exists(jamming_path):
            xapp.train_model(normal_path, jamming_path)
            print("✓ Model trained successfully")
        else:
            print("✗ Dataset not found. Please ensure dataset is available.")
            return
    
    xapp.start_monitoring()
    print("✓ xApp monitoring started")
    
    try:
        print("\nAvailable commands:")
        print("  1: Simulate Power Jamming (5s)")
        print("  2: Simulate Sweep Jamming (5s)")
        print("  3: Simulate Intelligent Jamming (5s)")
        print("  s: Show performance summary")
        print("  q: Quit demonstration")
        
        while True:
            command = input("\nEnter command: ").strip().lower()
            
            if command == '1':
                print("🚨 Simulating Power Jamming attack...")
                xapp.simulate_jamming_attack('power_jamming', 5.0)
                
            elif command == '2':
                print("🚨 Simulating Sweep Jamming attack...")
                xapp.simulate_jamming_attack('sweep_jamming', 5.0)
                
            elif command == '3':
                print("🚨 Simulating Intelligent Jamming attack...")
                xapp.simulate_jamming_attack('intelligent_jamming', 5.0)
                
            elif command == 's':
                summary = xapp.get_performance_summary()
                print("\n--- Performance Summary ---")
                print(f"Total Detections: {summary['total_detections']}")
                print(f"Jamming Detections: {summary['jamming_detections']}")
                print(f"Detection Rate: {summary['jamming_detection_rate']:.1%}")
                print(f"Average Latency: {summary['mean_latency_ms']:.1f}ms")
                print(f"Latency Compliance: {summary['latency_compliance']:.1%}")
                
            elif command == 'q':
                break
                
            else:
                print("Invalid command. Try again.")
    
    except KeyboardInterrupt:
        print("\nStopping demonstration...")
    finally:
        xapp.stop_monitoring()
        print("✓ Demonstration ended")

def main():
    parser = argparse.ArgumentParser(
        description="Ensemble ML Jamming Detection xApp for O-RAN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train --normal data/normal.csv --jamming data/jamming.csv --output models/ensemble

  python main.py detect --model models/ensemble --duration 60

  python main.py evaluate --normal data/normal.csv --jamming data/jamming.csv

  python main.py demo --model models/ensemble
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    train_parser = subparsers.add_parser('train', help='Train ensemble model')
    train_parser.add_argument('--normal', required=True, help='Path to normal traffic CSV')
    train_parser.add_argument('--jamming', required=True, help='Path to jamming attacks CSV')
    train_parser.add_argument('--output', required=True, help='Output directory for trained model')
    
    detect_parser = subparsers.add_parser('detect', help='Run real-time detection')
    detect_parser.add_argument('--model', required=True, help='Path to trained model directory')
    detect_parser.add_argument('--duration', type=int, default=60, help='Duration in seconds (default: 60)')
    
    evaluate_parser = subparsers.add_parser('evaluate', help='Run performance evaluation')
    evaluate_parser.add_argument('--normal', required=True, help='Path to normal traffic CSV')
    evaluate_parser.add_argument('--jamming', required=True, help='Path to jamming attacks CSV')
    
    demo_parser = subparsers.add_parser('demo', help='Interactive demonstration')
    demo_parser.add_argument('--model', help='Path to trained model directory (optional)')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args.normal, args.jamming, args.output)
        
    elif args.command == 'detect':
        if not os.path.exists(args.model):
            print(f"Error: Model directory '{args.model}' not found")
            sys.exit(1)
        run_real_time_detection(args.model, args.duration)
        
    elif args.command == 'evaluate':
        if not os.path.exists(args.normal) or not os.path.exists(args.jamming):
            print("Error: Dataset files not found")
            sys.exit(1)
        run_performance_evaluation(args.normal, args.jamming)
        
    elif args.command == 'demo':
        demonstrate_xapp_capabilities(args.model)
        
    else:
        parser.print_help()
        print("\n" + "=" * 50)
        print("QUICK START DEMONSTRATION")
        print("=" * 50)
        
        normal_path = "Ensemble_ML_Jamming_detection_dataset/dataset/normal_traffic.csv"
        jamming_path = "Ensemble_ML_Jamming_detection_dataset/dataset/jamming_attacks.csv"
        
        if os.path.exists(normal_path) and os.path.exists(jamming_path):
            print("Dataset found! Running quick demonstration...")
            demonstrate_xapp_capabilities()
        else:
            print("Dataset not found. Please clone the dataset repository:")
            print("git clone https://github.com/Utkarsh-upadhyay9/Ensemble_ML_Jamming_detection_dataset.git")

if __name__ == "__main__":
    main()
