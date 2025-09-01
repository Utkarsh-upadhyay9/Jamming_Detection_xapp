#!/usr/bin/env python3

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ensemble_model import EnsembleJammingDetector
from drl_jamming_detection import test_environment, test_drl_agent
from training.drl_trainer import DRLTrainer
from utils.logger import Logger

def test_drl_integration():
    """Test the integration of DRL with the ensemble jamming detector"""
    
    logger = Logger()
    logger.info("Testing DRL Integration with Ensemble Jamming Detector")
    
    logger.info("Creating ensemble with DRL enabled...")
    ensemble = EnsembleJammingDetector(
        use_drl=True,
        drl_actor_type='hybrid'
    )
    
    logger.info("Testing ensemble with synthetic data...")
    
    num_samples = 1000
    num_features = 20
    
    X = np.random.randn(num_samples, num_features)
    y = np.random.choice(['normal', 'jamming'], size=num_samples, p=[0.7, 0.3])
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    dataset = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    try:
        logger.info("Training ensemble with DRL component...")
        training_metrics = ensemble.train_ensemble(dataset=dataset)
        
        logger.info("Training completed successfully!")
        logger.info(f"Training metrics: {training_metrics}")
        
        logger.info("Testing predictions...")
        test_sample = X_test[:10]
        predictions = ensemble.predict(test_sample)
        probabilities = ensemble.predict_proba(test_sample)
        
        logger.info(f"Sample predictions: {predictions}")
        logger.info(f"Sample probabilities shape: {probabilities.shape}")
        
        ensemble_info = ensemble.get_ensemble_info()
        logger.info(f"Ensemble info: {ensemble_info}")
        
        if 'drl' in ensemble_info['individual_models']:
            logger.info("DRL component successfully integrated!")
            drl_info = ensemble_info['individual_models']['drl']
            logger.info(f"DRL training info: {drl_info}")
        else:
            logger.warning("DRL component not found in ensemble")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during DRL integration test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_standalone_drl():
    """Test standalone DRL jamming detection"""
    
    logger = Logger()
    logger.info("Testing Standalone DRL Jamming Detection")
    
    try:
        trainer = DRLTrainer()
        
        logger.info("Training MLP agent for 50 episodes...")
        mlp_agent = trainer.train_single_agent('mlp', num_episodes=50)
        
        logger.info("Training LLM agent for 50 episodes...")
        llm_agent = trainer.train_single_agent('llm', num_episodes=50)
        
        logger.info("Training Hybrid agent for 50 episodes...")
        hybrid_agent = trainer.train_single_agent('hybrid', num_episodes=50)
        
        agents = {
            'mlp': mlp_agent,
            'llm': llm_agent,
            'hybrid': hybrid_agent
        }
        
        logger.info("Comparing agent performance...")
        comparison_results = trainer.compare_agents(agents, num_eval_episodes=20)
        
        logger.info("Agent comparison results:")
        for agent_type, results in comparison_results.items():
            logger.info(
                f"{agent_type.upper()}: "
                f"Mean Reward = {results['mean_reward']:.3f} ± {results['std_reward']:.3f}"
            )
        
        best_agent = max(comparison_results.items(), key=lambda x: x[1]['mean_reward'])
        logger.info(f"Best performing agent: {best_agent[0].upper()}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during standalone DRL test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_multi_agent():
    """Demonstrate multi-agent DRL jamming detection"""
    
    logger = Logger()
    logger.info("Demonstrating Multi-Agent DRL Jamming Detection")
    
    try:
        trainer = DRLTrainer()
        
        logger.info("Training multi-agent system with 3 agents...")
        agents = trainer.train_multi_agent_system(num_agents=3, num_episodes=100)
        
        logger.info(f"Successfully trained {len(agents)} agents")
        
        for i, agent in enumerate(agents):
            info = agent.get_training_info()
            logger.info(f"Agent {i}: Episodes = {info['episodes_trained']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during multi-agent test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_comprehensive_demo():
    """Run comprehensive demonstration of DRL jamming detection"""
    
    logger = Logger()
    logger.info("=" * 80)
    logger.info("DEEP REINFORCEMENT LEARNING JAMMING DETECTION SYSTEM")
    logger.info("Based on: Secure and Privacy-Preserving ISAC in RIS-Aided IAB Networks")
    logger.info("=" * 80)
    
    results = {
        'environment_test': False,
        'drl_agent_test': False,
        'standalone_drl': False,
        'drl_integration': False,
        'multi_agent': False
    }
    
    logger.info("\n1. Testing Environment...")
    try:
        test_environment()
        results['environment_test'] = True
        logger.info("✓ Environment test passed")
    except Exception as e:
        logger.error(f"✗ Environment test failed: {e}")
    
    logger.info("\n2. Testing DRL Agents...")
    try:
        test_drl_agent()
        results['drl_agent_test'] = True
        logger.info("✓ DRL agent test passed")
    except Exception as e:
        logger.error(f"✗ DRL agent test failed: {e}")
    
    logger.info("\n3. Testing Standalone DRL...")
    results['standalone_drl'] = test_standalone_drl()
    
    logger.info("\n4. Testing DRL Integration with Ensemble...")
    results['drl_integration'] = test_drl_integration()
    
    logger.info("\n5. Testing Multi-Agent System...")
    results['multi_agent'] = demonstrate_multi_agent()
    
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    logger.info(f"\nOverall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        logger.info(" All tests passed! DRL Jamming Detection System is ready!")
    else:
        logger.info("⚠  Some tests failed. Check the logs above for details.")
    
    logger.info("\nNext steps:")
    logger.info("- Run: python drl_jamming_detection.py --mode train --actor_type hybrid")
    logger.info("- Run: python drl_jamming_detection.py --mode compare")
    logger.info("- Run: python drl_jamming_detection.py --mode benchmark")
    
    return results

if __name__ == "__main__":
    run_comprehensive_demo()
