#!/usr/bin/env python3

import numpy as np
import torch
import os
import sys
import argparse
import logging
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.drl_trainer import DRLTrainer
from models.drl_jamming_detector import DDPGJammingDetector
from envs.jamming_environment import JammingDetectionEnvironment
from config.model_config import DRL_CONFIG
from utils.logger import JammingDetectionLogger

# Import USRP calibration if available
try:
    from utils.performance_calibration import USRP_CALIBRATOR, PaperBaseline
    USRP_CALIBRATION_AVAILABLE = True
except ImportError:
    USRP_CALIBRATION_AVAILABLE = False
    print("⚠️  USRP Performance Calibration module not available")

def main():
    parser = argparse.ArgumentParser(description='Deep Reinforcement Learning Jamming Detection System')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'compare', 'multi_agent', 'benchmark', 'usrp_test'], 
                       default='train', help='Mode to run the system')
    parser.add_argument('--actor_type', choices=['mlp', 'llm', 'hybrid'], 
                       default='hybrid', help='Type of actor network')
    parser.add_argument('--episodes', type=int, default=500, help='Number of training episodes')
    parser.add_argument('--eval_episodes', type=int, default=50, help='Number of evaluation episodes')
    parser.add_argument('--eval_freq', type=int, default=None, help='Override evaluation frequency (episodes)')
    parser.add_argument('--save_freq', type=int, default=None, help='Override save frequency (episodes)')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size for training')
    parser.add_argument('--max_steps', type=int, default=None, help='Override max steps per episode')
    parser.add_argument('--num_agents', type=int, default=2, help='Number of agents for multi-agent training')
    parser.add_argument('--save_plots', action='store_true', help='Save training plots')
    parser.add_argument('--load_model', type=str, help='Path to load pre-trained model')
    parser.add_argument('--environment', choices=['ideal', 'moderate', 'realistic'], 
                       default='realistic', help='USRP test environment type')
    parser.add_argument('--usrp_calibration', action='store_true', default=True, 
                       help='Enable USRP performance calibration')
    parser.add_argument('--dataset_csv', type=str, help='Path to offline USRP / captured dataset CSV for dataset-driven environment')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose console logging (INFO level)')
    
    args = parser.parse_args()
    
    logger = JammingDetectionLogger()
    if '--verbose' in sys.argv or getattr(args, 'verbose', False):
        logger.set_console_level(logging.INFO)
    logger.log_system_event("system_start", "Starting Deep Reinforcement Learning Jamming Detection System")
    logger.log_system_event("mode_set", f"Mode: {args.mode}")
    logger.log_system_event("device_info", f"PyTorch device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Display USRP calibration status
    if USRP_CALIBRATION_AVAILABLE and args.usrp_calibration:
        logger.log_system_event("usrp_calibration", "✅ USRP Performance Calibration: ENABLED")
        if args.mode in ['train', 'compare', 'usrp_test']:
            print(USRP_CALIBRATOR.get_calibration_report(args.environment))
    else:
        logger.log_system_event("usrp_calibration", "⚠️  USRP Performance Calibration: DISABLED")
    
    trainer = DRLTrainer()
    if args.dataset_csv:
        try:
            trainer.use_dataset(args.dataset_csv)
            logger.log_system_event("dataset_mode", f"Using dataset environment: {args.dataset_csv}")
        except Exception as e:
            logger.log_system_event("dataset_error", f"Failed to init dataset env: {e}", level="ERROR")
    
    # Configure environment for USRP calibration
    if args.usrp_calibration and USRP_CALIBRATION_AVAILABLE:
        trainer.env_config.update({
            'environment': args.environment,
            'use_usrp_calibration': True
        })
        trainer.config.update({
            'use_usrp_calibration': True,
            'environment_type': args.environment
        })
    
    if args.mode == 'train':
        logger.log_system_event("training_start", f"Training {args.actor_type.upper()} agent for {args.episodes} episodes")
        # Apply overrides
        if args.eval_freq:
            trainer.config['evaluation_frequency'] = args.eval_freq
        if args.save_freq:
            trainer.config['save_frequency'] = args.save_freq
        if args.batch_size:
            trainer.config['batch_size'] = args.batch_size
        if args.max_steps:
            trainer.config['max_steps_per_episode'] = args.max_steps

        agent = trainer.train_single_agent(args.actor_type, args.episodes)
        
        trainer.save_agent(agent, f'final_{args.actor_type}_agent.pth')
        
        if args.save_plots:
            trainer.plot_training_results(f'{args.actor_type}_training_results.png')
        
        trainer.save_training_history(f'{args.actor_type}_training_history.json')
        
        logger.log_system_event("training_complete", "Training completed successfully")
    
    elif args.mode == 'evaluate':
        if args.load_model:
            agent = trainer.load_agent(args.load_model, args.actor_type)
        else:
            logger.log_system_event("info", "No model specified. Training new agent...")
            agent = trainer.train_single_agent(args.actor_type, args.episodes)
        
        env = JammingDetectionEnvironment(trainer.env_config)
        results = agent.evaluate(env, args.eval_episodes)
        
        logger.log_system_event("info", "Evaluation Results:")
        logger.log_system_event("info", f"Mean Reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
        logger.log_system_event("info", f"Min Reward: {results['min_reward']:.3f}")
        logger.log_system_event("info", f"Max Reward: {results['max_reward']:.3f}")
        
        demo_episodes = 3
        logger.log_system_event("info", f"\nRunning {demo_episodes} demonstration episodes:")
        
        for episode in range(demo_episodes):
            state = env.reset()
            total_reward = 0.0
            step = 0
            done = False
            
            logger.log_system_event("info", f"\n--- Episode {episode + 1} ---")
            
            while not done and step < 20:
                action = agent.select_action(state, add_noise=False)
                next_state, reward, done, info = env.step(action)
                
                logger.log_system_event("info", f"Step {step + 1}:")
                logger.log_system_event("info", f"  Action: {action}")
                logger.log_system_event("info", f"  Reward: {reward:.3f}")
                logger.log_system_event("info", f"  Jamming Active: {info['jamming_active']}")
                logger.log_system_event("info", f"  Detection Confidence: {info['detection_confidence']:.3f}")
                logger.log_system_event("info", f"  Communication Rate: {info['communication_rate']:.3f}")
                logger.log_system_event("info", f"  Sensing Rate: {info['sensing_rate']:.3f}")
                
                total_reward += reward
                state = next_state
                step += 1
            
            logger.log_system_event("info", f"Episode Total Reward: {total_reward:.3f}")
    
    elif args.mode == 'compare':
        logger.log_system_event("info", "Training and comparing all agent types")
        
        agents = trainer.train_all_agents(args.episodes)
        results = trainer.compare_agents(agents, args.eval_episodes)
        
        logger.log_system_event("info", "\nComparison Results:")
        for actor_type, result in results.items():
            logger.log_system_event("info", f"{actor_type.upper()}: {result['mean_reward']:.3f} ± {result['std_reward']:.3f}")
        
        best_agent = max(results.items(), key=lambda x: x[1]['mean_reward'])
        logger.log_system_event("info", f"\nBest performing agent: {best_agent[0].upper()} with reward {best_agent[1]['mean_reward']:.3f}")
        
        if args.save_plots:
            trainer.plot_training_results('agent_comparison_results.png')
        
        trainer.save_training_history('comparison_training_history.json')
    
    elif args.mode == 'multi_agent':
        logger.log_system_event("info", f"Training multi-agent system with {args.num_agents} agents")
        
        agents = trainer.train_multi_agent_system(args.num_agents, args.episodes)
        
        logger.log_system_event("info", "Multi-agent training completed")
        logger.log_system_event("info", f"Trained {len(agents)} agents successfully")
        
        for i, agent in enumerate(agents):
            info = agent.get_training_info()
            logger.log_system_event("info", f"Agent {i}: Episodes trained: {info['episodes_trained']}")
    
    elif args.mode == 'benchmark':
        logger.log_system_event("info", "Running comprehensive benchmark")
        
        results = trainer.benchmark_performance(args.eval_episodes)
        
        logger.log_system_event("info", "\nBenchmark Results Summary:")
        logger.log_system_event("info", "-" * 80)
        
        for actor_type, metrics in results.items():
            logger.log_system_event("info", f"{actor_type.upper()} Agent:")
            logger.log_system_event("info", f"  Reward: {metrics['mean_reward']:.3f} ± {metrics['std_reward']:.3f}")
            logger.log_system_event("info", f"  Detection Accuracy: {metrics['mean_accuracy']:.3f} ± {metrics['std_accuracy']:.3f}")
            logger.log_system_event("info", f"  False Alarm Rate: {metrics['mean_false_alarm_rate']:.3f} ± {metrics['std_false_alarm_rate']:.3f}")
            logger.log_system_event("info", "")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        benchmark_file = f"benchmark_results_{timestamp}.json"
        
        import json
        with open(benchmark_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.log_system_event("info", f"Benchmark results saved to {benchmark_file}")
        
    elif args.mode == 'usrp_test':
        logger.log_system_event("info", f"Running USRP calibrated performance test in {args.environment} environment")
        
        if not USRP_CALIBRATION_AVAILABLE:
            logger.log_system_event("error", "USRP calibration not available. Cannot run USRP test mode.")
            sys.exit(1)
        
        # Test all actor types with USRP calibration
        actor_types = ['mlp', 'llm', 'hybrid']
        usrp_results = {}
        
        for actor_type in actor_types:
            logger.log_system_event("info", f"\n🔬 Testing {actor_type.upper()} with USRP calibration...")
            
            # Train agent with USRP calibration
            agent = trainer.train_single_agent(actor_type, args.episodes)
            
            # Evaluate with USRP environment
            env = JammingDetectionEnvironment(trainer.env_config)
            eval_results = agent.evaluate(env, args.eval_episodes)
            
            # Get USRP calibration report
            calibration_report = agent.get_usrp_calibration_report()
            training_info = agent.get_training_info()
            
            usrp_results[actor_type] = {
                'evaluation': eval_results,
                'training_info': training_info,
                'calibration_report': calibration_report
            }
            
            logger.log_system_event("info", f"  Mean Reward: {eval_results['mean_reward']:.3f}")
            logger.log_system_event("info", f"  USRP Ready: {training_info.get('usrp_ready', False)}")
            
            # Save USRP calibrated model
            model_path = f"models/saved/drl_{actor_type}_usrp_{args.environment}.pth"
            agent.save_model(model_path)
            logger.log_system_event("info", f"  Model saved: {model_path}")
        
        # Performance validation against paper baseline
        logger.log_system_event("info", "\n🎯 USRP Performance Validation")
        logger.log_system_event("info", "-" * 50)
        
        if USRP_CALIBRATION_AVAILABLE:
            paper_baseline = PaperBaseline()
            targets = USRP_CALIBRATOR.get_performance_targets(args.environment)
            
            best_actor = max(usrp_results.keys(), 
                           key=lambda k: usrp_results[k]['training_info'].get('estimated_f1_score', 0))
            best_results = usrp_results[best_actor]
            
            estimated_f1 = best_results['training_info'].get('estimated_f1_score', 0)
            improvement = estimated_f1 - paper_baseline.f1_score if estimated_f1 > 0 else 0
            
            logger.log_system_event("info", f"Best Actor: {best_actor.upper()}")
            logger.log_system_event("info", f"Paper Baseline F1: {paper_baseline.f1_score:.1%}")
            logger.log_system_event("info", f"Estimated F1: {estimated_f1:.1%}")
            logger.log_system_event("info", f"Improvement: +{improvement:.1%}")
            logger.log_system_event("info", f"Target Range: 1-7%")
            
            success = (improvement >= 0.01 and improvement <= 0.07 and 
                      estimated_f1 >= targets['f1_score'] * 0.95)
            
            if success:
                logger.log_system_event("info", "✅ SUCCESS: USRP calibration targets achieved!")
                logger.log_system_event("info", "🚀 Ready for real USRP deployment!")
            else:
                logger.log_system_event("info", "⚠️  PARTIAL: Continue training for optimal USRP performance")
        
        # Save USRP test results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        usrp_results_file = f"usrp_test_results_{args.environment}_{timestamp}.json"
        
        import json
        with open(usrp_results_file, 'w') as f:
            # Convert non-serializable objects to strings
            serializable_results = {}
            for actor_type, results in usrp_results.items():
                serializable_results[actor_type] = {
                    'evaluation': results['evaluation'],
                    'training_info': results['training_info'],
                    'calibration_summary': f"USRP calibration completed for {actor_type} in {args.environment} environment"
                }
            json.dump(serializable_results, f, indent=2)
        
        logger.log_system_event("info", f"USRP test results saved to {usrp_results_file}")
    
    logger.log_system_event("info", "DRL Jamming Detection System execution completed")

def test_environment():
    """Test the jamming detection environment"""
    print("Testing Jamming Detection Environment...")
    
    env_config = {
        'state_dim': 10,
        'action_dim': 5,
        'max_episode_steps': 100,
        'jamming_probability': 0.4
    }
    
    env = JammingDetectionEnvironment(env_config)
    
    print("Environment created successfully")
    print(f"State space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"Initial state: {state}")
    
    for step in range(10):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        
        print(f"\nStep {step + 1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.3f}")
        print(f"  Done: {done}")
        print(f"  Jamming: {info['jamming_active']}")
        print(f"  Detection confidence: {info['detection_confidence']:.3f}")
        
        if done:
            break
        
        state = next_state
    
    print("\nEnvironment test completed successfully!")

def test_drl_agent():
    """Test the DRL agent creation and basic functionality"""
    print("Testing DRL Agent...")
    
    state_dim = 10
    action_dim = 5
    
    for actor_type in ['mlp', 'llm', 'hybrid']:
        print(f"\nTesting {actor_type.upper()} actor...")
        
        try:
            agent = DDPGJammingDetector(
                state_dim=state_dim,
                action_dim=action_dim,
                actor_type=actor_type
            )
            
            test_state = np.random.randn(state_dim)
            action = agent.select_action(test_state, add_noise=False)
            
            print(f"  Agent created successfully")
            print(f"  Test action shape: {action.shape}")
            print(f"  Test action: {action}")
            
            agent_info = agent.get_training_info()
            print(f"  Agent info: {agent_info}")
            
        except Exception as e:
            print(f"  Error testing {actor_type} agent: {e}")
    
    print("\nDRL agent test completed!")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("No arguments provided. Running tests...")
        test_environment()
        test_drl_agent()
        print("\nTo train agents, use: python drl_jamming_detection.py --mode train")
        print("For help: python drl_jamming_detection.py --help")
    else:
        main()
