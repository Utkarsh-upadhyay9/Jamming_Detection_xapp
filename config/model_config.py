ENSEMBLE_WEIGHTS = {
    'rf': 0.44,      # Random Forest: 44%
    'svm': 0.41,     # Support Vector Machine: 41%
    'if': 0.15       # Isolation Forest: 15%
}

RF_CONFIG = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42,
    'n_jobs': -1
}

# SVM Configuration  
from typing import Dict, Any

SVM_CONFIG: Dict[str, Any] = {
    'C': 1.0,
    'kernel': 'rbf',
    'gamma': 'scale',
    'probability': True,
    'random_state': 42
}

RF_CONFIG: Dict[str, Any] = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42,
    'n_jobs': -1
}

ISOLATION_FOREST_CONFIG: Dict[str, Any] = {
    'n_estimators': 100,
    'contamination': 0.1,
    'random_state': 42,
    'n_jobs': -1
}

ENSEMBLE_CONFIG: Dict[str, Any] = {
    'voting': 'soft',
    'weights': [0.4, 0.3, 0.3]
}

TRAINING_CONFIG: Dict[str, Any] = {
    'test_size': 0.2,
    'validation_size': 0.2,
    'random_state': 42,
    'cross_validation_folds': 5
}

DRL_CONFIG: Dict[str, Any] = {
    'actor_lr': 0.0001,
    'critic_lr': 0.001,
    'gamma': 0.99,
    'tau': 0.005,
    'batch_size': 128,
    'replay_buffer_size': 300000,
    'noise_std': 0.15,
    'noise_decay': 0.995,
    'hidden_dims': [256, 256],
    'max_episodes': 1000,
    'max_steps_per_episode': 200,
    'update_frequency': 1,
    'target_update_frequency': 1,
    'warmup_steps': 2000,
    'evaluation_frequency': 10,
    'save_frequency': 50,
    # New training stabilization features
    'reward_clip': 2.0,              # Clip per-step reward to [-reward_clip, reward_clip] (None to disable)
    'early_stop_patience': None,      # Number of eval checks without improvement before early stop (None to disable)
    'early_stop_delta': 1e-4,         # Minimum improvement in eval reward to reset patience
    'use_progress_bar': True,         # Enable tqdm progress bar (enabled by default per user request)
    'scale_dataset_features': True,   # Apply feature standardization in dataset env
    
    # USRP-specific calibration settings
    'use_usrp_calibration': True,
    'environment_type': 'realistic',  # 'ideal', 'moderate', 'realistic'
    'target_improvement_percent': 3.8,  # 3.8% improvement over paper baseline
    'usrp_noise_modeling': True,
    'channel_impairments': True,
    'performance_tracking': True
}

IF_CONFIG = {
    'n_estimators': 100,
    'contamination': 0.3,  # 30% jamming in dataset
    'random_state': 42,
    'n_jobs': -1
}

THRESHOLDS = {
    'binary_detection': 0.52,     # Binary jamming detection threshold
    'power_jamming': 0.8,         # Power jamming confidence threshold
    'sweep_jamming': 0.7,         # Sweep jamming confidence threshold
    'intelligent_jamming': 0.6    # Intelligent jamming confidence threshold
}

FEATURE_CONFIG = {
    'window_size': 10,            # Sampling window size
    'normalization': 'zscore',    # Z-score normalization
    'feature_count': 15           # Total engineered features
}

PERFORMANCE_REQUIREMENTS = {
    'max_latency_ms': 100,        # Maximum detection latency (ms)
    'min_accuracy': 0.90,         # Minimum required accuracy
    'min_f1_score': 0.85,         # Minimum F1-score
    'memory_limit_mb': 64         # Memory usage limit (MB)
}

TRAINING_CONFIG = {
    'test_size': 0.2,             # Train/test split ratio
    'validation_size': 0.1,       # Validation set size
    'cross_validation_folds': 5,  # K-fold CV
    'random_state': 42
}

CONFIDENCE_CONFIG = {
    'entropy_weight': 0.3,        # Entropy weighting factor
    'variance_threshold': 0.25,   # Maximum variance for confidence
    'uncertainty_penalty': 0.1    # Penalty for high uncertainty
}
