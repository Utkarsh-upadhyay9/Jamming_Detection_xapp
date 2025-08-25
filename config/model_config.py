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
SVM_CONFIG = {
    'kernel': 'rbf',
    'gamma': 0.001,
    'C': 1.0,
    'probability': True,
    'random_state': 42
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
