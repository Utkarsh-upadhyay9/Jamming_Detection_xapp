import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from typing import Dict, List, Tuple, Any, Optional
import json
from datetime import datetime

from models.drl_jamming_detector import DDPGJammingDetector
from envs.jamming_environment import JammingDetectionEnvironment, MultiAgentJammingEnvironment
try:
    from envs.jamming_dataset_environment import JammingDatasetEnvironment
except ImportError:
    JammingDatasetEnvironment = None
from config.model_config import DRL_CONFIG
from utils.logger import JammingDetectionLogger
from utils.metrics import PerformanceMetrics, LatencyTracker

class DRLTrainer:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or DRL_CONFIG
        self.logger = JammingDetectionLogger()
        
        self.env_config = {
            'state_dim': 10,
            'action_dim': 5,
            'max_episode_steps': self.config['max_steps_per_episode'],
            'jamming_probability': 0.3,
            'noise_std': 0.1,
            'communication_channels': 4,
            'sensing_targets': 2
        }
        
        self.training_history = {
            'mlp': {'rewards': [], 'losses': [], 'eval_rewards': [], 'eval_f1s': [], 'eval_episode_indices': []},
            'llm': {'rewards': [], 'losses': [], 'eval_rewards': [], 'eval_f1s': [], 'eval_episode_indices': []},
            'hybrid': {'rewards': [], 'losses': [], 'eval_rewards': [], 'eval_f1s': [], 'eval_episode_indices': []}
        }
        # Dataset path (optional offline environment)
        self.dataset_path: Optional[str] = None

    def use_dataset(self, csv_path: str):
        if JammingDatasetEnvironment is None:
            raise RuntimeError("Dataset environment module not available")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(csv_path)
        self.dataset_path = csv_path
        # Update env config state_dim after probing dataset
        env = JammingDatasetEnvironment(csv_path, scale_features=self.config.get('scale_dataset_features', True))
        self.env_config['state_dim'] = env.state_dim
        self.logger.log_system_event("dataset_env_config", f"Using dataset environment with state_dim={env.state_dim}")
        
    def train_single_agent(self, actor_type: str, num_episodes: int = None) -> DDPGJammingDetector:
        num_episodes = num_episodes or self.config['max_episodes']
        
        if self.dataset_path:
            env = JammingDatasetEnvironment(self.dataset_path, scale_features=self.config.get('scale_dataset_features', True))
        else:
            env = JammingDetectionEnvironment(self.env_config)
        agent = DDPGJammingDetector(
            state_dim=self.env_config['state_dim'],
            action_dim=self.env_config['action_dim'],
            actor_type=actor_type,
            config=self.config
        )
        
        self.logger.log_system_event("agent_training_start", f"Starting training for {actor_type.upper()} agent")
        
        episode_rewards = []
        best_reward = float('-inf')
        
        # Determine effective max steps (avoid exceeding dataset length)
        if self.dataset_path and hasattr(env, 'episode_length'):
            effective_max_steps = min(self.config['max_steps_per_episode'], getattr(env, 'episode_length'))
        else:
            effective_max_steps = self.config['max_steps_per_episode']

        # Adaptive evaluation frequency for small runs
        eval_freq = self.config['evaluation_frequency']
        if num_episodes < eval_freq:
            eval_freq = max(1, num_episodes // 5) or 1

        # Optional progress bar
        use_pbar = self.config.get('use_progress_bar', False)
        iterator = range(num_episodes)
        pbar = None
        try:
            if use_pbar:
                from tqdm import tqdm  # type: ignore
                pbar = tqdm(iterator, desc=f"Training {actor_type}")
                iterator = pbar
        except Exception:
            use_pbar = False

        best_eval = float('-inf')
        patience_left = self.config.get('early_stop_patience')
        delta = self.config.get('early_stop_delta', 0.0)

        for episode in iterator:
            total_reward = agent.train_episode(env, effective_max_steps)
            episode_rewards.append(total_reward)
            
            self.training_history[actor_type]['rewards'].append(total_reward)
            
            if hasattr(agent, 'training_metrics') and agent.training_metrics:
                self.training_history[actor_type]['losses'].append(
                    agent.training_metrics.get('critic_loss', 0)
                )
            
            improved = False
            if episode % eval_freq == 0:
                eval_results = agent.evaluate(env, num_episodes=10)
                mean_eval_reward = eval_results.get('mean_reward', 0.0)
                self.training_history[actor_type]['eval_rewards'].append(mean_eval_reward)

                # record eval episode index
                self.training_history[actor_type]['eval_episode_indices'].append(episode)

                # if evaluate returned f1/precision/recall, save f1 as well
                f1 = eval_results.get('f1', None)
                if f1 is not None:
                    self.training_history[actor_type]['eval_f1s'].append(f1)

                # Logging: prefer f1 when available
                if f1 is not None:
                    self.logger.log_system_event("info", 
                        f"Episode {episode}: Train Reward: {total_reward:.3f}, "
                        f"Eval F1: {f1:.4f}, Eval Reward: {mean_eval_reward:.3f} ± {eval_results.get('std_reward', 0.0):.3f}"
                    )
                else:
                    self.logger.log_system_event("info", 
                        f"Episode {episode}: Train Reward: {total_reward:.3f}, "
                        f"Eval Reward: {mean_eval_reward:.3f} ± {eval_results.get('std_reward', 0.0):.3f}"
                    )

                if mean_eval_reward > best_reward:
                    best_reward = mean_eval_reward
                    improved = True
                    self.save_agent(agent, f'best_{actor_type}_agent.pth')
            
            if episode % self.config['save_frequency'] == 0 and episode > 0:
                self.save_agent(agent, f'{actor_type}_agent_episode_{episode}.pth')

            # Lightweight per-episode progress log
            self.logger.log_system_event(
                "episode_progress",
                f"ep={episode+1}/{num_episodes} reward={total_reward:.2f} eval_best={best_reward:.2f} buffer={len(agent.replay_buffer)} actor_loss={agent.training_metrics.get('actor_loss', 0):.4f} critic_loss={agent.training_metrics.get('critic_loss', 0):.4f}"
            )
            if pbar:
                pbar.set_postfix({
                    'R': f"{total_reward:.1f}",
                    'BestEval': f"{best_reward:.1f}",
                    'Buf': len(agent.replay_buffer)
                })

            # Early stopping logic
            if patience_left and episode % eval_freq == 0:
                if improved and (best_reward - best_eval) > delta:
                    best_eval = best_reward
                    patience_left = self.config.get('early_stop_patience')
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        self.logger.log_system_event("early_stop", f"Stopping early at episode {episode+1}")
                        break

        if pbar:
            pbar.close()
        
        self.logger.log_system_event("info", f"Training completed for {actor_type.upper()} agent. Best eval reward: {best_reward:.3f}")
        return agent
    
    def train_all_agents(self, num_episodes: int = None) -> Dict[str, DDPGJammingDetector]:
        agents = {}
        
        for actor_type in ['mlp', 'llm', 'hybrid']:
            agents[actor_type] = self.train_single_agent(actor_type, num_episodes)
        
        return agents
    
    def compare_agents(self, agents: Dict[str, DDPGJammingDetector], num_eval_episodes: int = 50):
        env = JammingDetectionEnvironment(self.env_config)
        
        results = {}
        
        for actor_type, agent in agents.items():
            eval_results = agent.evaluate(env, num_episodes=num_eval_episodes)
            results[actor_type] = eval_results
            
            self.logger.log_system_event("info", 
                f"{actor_type.upper()} Agent - "
                f"Mean Reward: {eval_results['mean_reward']:.3f} ± {eval_results['std_reward']:.3f}, "
                f"Range: [{eval_results['min_reward']:.3f}, {eval_results['max_reward']:.3f}]"
            )
        
        return results
    
    def train_multi_agent_system(self, num_agents: int = 2, num_episodes: int = None) -> List[DDPGJammingDetector]:
        num_episodes = num_episodes or self.config['max_episodes']
        
        env = MultiAgentJammingEnvironment(num_agents=num_agents, config=self.env_config)
        
        agents = []
        for i in range(num_agents):
            agent = DDPGJammingDetector(
                state_dim=self.env_config['state_dim'],
                action_dim=self.env_config['action_dim'],
                actor_type='hybrid',
                config=self.config
            )
            agents.append(agent)
        
        self.logger.log_system_event("info", f"Starting multi-agent training with {num_agents} agents")
        
        episode_rewards = []
        
        for episode in range(num_episodes):
            observations = env.reset()
            
            for agent in agents:
                agent.noise.reset()
            
            total_rewards = [0.0] * num_agents
            
            for step in range(self.config['max_steps_per_episode']):
                actions = {}
                
                for i, agent in enumerate(agents):
                    state = observations[i]
                    action = agent.select_action(state, add_noise=True)
                    actions[i] = action
                
                next_observations, rewards, dones, infos = env.step(actions)
                
                for i, agent in enumerate(agents):
                    agent.store_transition(
                        observations[i], actions[i], rewards[i], 
                        next_observations[i], dones[i]
                    )
                    
                    if len(agent.replay_buffer) >= self.config['batch_size']:
                        agent.update_networks()
                    
                    total_rewards[i] += rewards[i]
                
                observations = next_observations
                
                if all(dones.values()):
                    break
            
            avg_reward = np.mean(total_rewards)
            episode_rewards.append(avg_reward)
            
            if episode % self.config['evaluation_frequency'] == 0:
                self.logger.log_system_event("info", 
                    f"Multi-Agent Episode {episode}: "
                    f"Average Reward: {avg_reward:.3f}, "
                    f"Individual Rewards: {[f'{r:.3f}' for r in total_rewards]}"
                )
        
        self.logger.log_system_event("info", "Multi-agent training completed")
        
        for i, agent in enumerate(agents):
            self.save_agent(agent, f'multi_agent_{i}_final.pth')
        
        return agents
    
    def save_agent(self, agent: DDPGJammingDetector, filename: str):
        save_dir = 'models/checkpoints'
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename)
        agent.save_model(filepath)
        self.logger.log_system_event("info", f"Agent saved to {filepath}")
    
    def load_agent(self, filename: str, actor_type: str) -> DDPGJammingDetector:
        agent = DDPGJammingDetector(
            state_dim=self.env_config['state_dim'],
            action_dim=self.env_config['action_dim'],
            actor_type=actor_type,
            config=self.config
        )
        
        save_dir = 'models/checkpoints'
        filepath = os.path.join(save_dir, filename)
        agent.load_model(filepath)
        
        self.logger.log_system_event("info", f"Agent loaded from {filepath}")
        return agent
    
    def plot_training_results(self, save_path: str = 'training_results.png'):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        colors = {'mlp': 'blue', 'llm': 'red', 'hybrid': 'green'}
        
        for actor_type in ['mlp', 'llm', 'hybrid']:
            if self.training_history[actor_type]['rewards']:
                episodes = range(len(self.training_history[actor_type]['rewards']))
                
                axes[0, 0].plot(episodes, self.training_history[actor_type]['rewards'], 
                               color=colors[actor_type], label=f'{actor_type.upper()}', alpha=0.7)
                
                if self.training_history[actor_type]['eval_rewards']:
                    eval_episodes = range(0, len(self.training_history[actor_type]['rewards']), 
                                        self.config['evaluation_frequency'])
                    eval_episodes = eval_episodes[:len(self.training_history[actor_type]['eval_rewards'])]
                    
                    axes[0, 1].plot(eval_episodes, self.training_history[actor_type]['eval_rewards'],
                                   color=colors[actor_type], label=f'{actor_type.upper()}', marker='o')
                
                if self.training_history[actor_type]['losses']:
                    axes[1, 0].plot(range(len(self.training_history[actor_type]['losses'])), 
                                   self.training_history[actor_type]['losses'],
                                   color=colors[actor_type], label=f'{actor_type.upper()}', alpha=0.7)
        
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].set_title('Evaluation Rewards')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Mean Reward')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 0].set_title('Training Losses')
        axes[1, 0].set_xlabel('Update Step')
        axes[1, 0].set_ylabel('Critic Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        reward_comparison = []
        labels = []
        for actor_type in ['mlp', 'llm', 'hybrid']:
            if self.training_history[actor_type]['eval_rewards']:
                reward_comparison.append(self.training_history[actor_type]['eval_rewards'][-1])
                labels.append(actor_type.upper())
        
        if reward_comparison:
            axes[1, 1].bar(labels, reward_comparison, color=[colors[label.lower()] for label in labels])
            axes[1, 1].set_title('Final Evaluation Performance')
            axes[1, 1].set_ylabel('Mean Reward')
            axes[1, 1].grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.log_system_event("info", f"Training results plot saved to {save_path}")
    
    def save_training_history(self, filename: str = 'training_history.json'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        
        with open(filename, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        self.logger.log_system_event("info", f"Training history saved to {filename}")
    
    def benchmark_performance(self, num_test_episodes: int = 100) -> Dict[str, Any]:
        env = JammingDetectionEnvironment(self.env_config)
        
        results = {}
        
        checkpoint_dir = 'models/checkpoints'
        if not os.path.exists(checkpoint_dir):
            self.logger.warning("No checkpoints found. Training agents first...")
            agents = self.train_all_agents(num_episodes=200)
        else:
            agents = {}
            for actor_type in ['mlp', 'llm', 'hybrid']:
                try:
                    agents[actor_type] = self.load_agent(f'best_{actor_type}_agent.pth', actor_type)
                except:
                    self.logger.warning(f"Could not load {actor_type} agent. Training new one...")
                    agents[actor_type] = self.train_single_agent(actor_type, num_episodes=100)
        
        for actor_type, agent in agents.items():
            test_rewards = []
            detection_accuracies = []
            false_alarm_rates = []
            
            for episode in range(num_test_episodes):
                state = env.reset()
                total_reward = 0.0
                correct_detections = 0
                false_alarms = 0
                total_steps = 0
                
                done = False
                while not done:
                    action = agent.select_action(state, add_noise=False)
                    next_state, reward, done, info = env.step(action)
                    
                    total_reward += reward
                    
                    detection_threshold = action[0]
                    detection_confidence = info['detection_confidence']
                    predicted_jamming = detection_confidence > detection_threshold
                    actual_jamming = info['jamming_active']
                    
                    if predicted_jamming == actual_jamming:
                        correct_detections += 1
                    elif predicted_jamming and not actual_jamming:
                        false_alarms += 1
                    
                    total_steps += 1
                    state = next_state
                
                test_rewards.append(total_reward)
                detection_accuracies.append(correct_detections / total_steps if total_steps > 0 else 0)
                false_alarm_rates.append(false_alarms / total_steps if total_steps > 0 else 0)
            
            results[actor_type] = {
                'mean_reward': np.mean(test_rewards),
                'std_reward': np.std(test_rewards),
                'mean_accuracy': np.mean(detection_accuracies),
                'std_accuracy': np.std(detection_accuracies),
                'mean_false_alarm_rate': np.mean(false_alarm_rates),
                'std_false_alarm_rate': np.std(false_alarm_rates)
            }
            
            self.logger.log_system_event("info", 
                f"{actor_type.upper()} Performance: "
                f"Reward: {results[actor_type]['mean_reward']:.3f}±{results[actor_type]['std_reward']:.3f}, "
                f"Accuracy: {results[actor_type]['mean_accuracy']:.3f}±{results[actor_type]['std_accuracy']:.3f}, "
                f"False Alarm Rate: {results[actor_type]['mean_false_alarm_rate']:.3f}±{results[actor_type]['std_false_alarm_rate']:.3f}"
            )
        
        return results
