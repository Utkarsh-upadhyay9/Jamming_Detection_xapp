"""
Complete project configuration file.
Contains all system-wide settings and constants.
"""

import os
from typing import Dict, Any

# =====================================================
# PROJECT METADATA
# =====================================================

PROJECT_NAME = "Ensemble ML Jamming Detection xApp"
VERSION = "1.0.0"
AUTHOR = "Research Implementation"
DESCRIPTION = "O-RAN xApp for jamming detection using ensemble machine learning"

# =====================================================
# RESEARCH PAPER SPECIFICATIONS
# =====================================================

# Performance targets from research paper
PERFORMANCE_TARGETS = {
    'f1_score': 0.954,
    'accuracy_improvement_over_rf': 0.172,  # 17.2%
    'accuracy_improvement_over_svm': 0.165,  # 16.5%
    'detection_latency_ms': 100,
    'latency_compliance_rate': 0.95  # 95% of detections under 100ms
}

# Ensemble weights from empirical optimization
ENSEMBLE_WEIGHTS = {
    'rf': 0.44,      # Random Forest - 44%
    'svm': 0.41,     # Support Vector Machine - 41%
    'if': 0.15       # Isolation Forest - 15%
}

# =====================================================
# MODEL CONFIGURATIONS
# =====================================================

# Random Forest configuration
RF_CONFIG = {
    'n_estimators': 100,
    'max_depth': 20,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'bootstrap': True,
    'random_state': 42
}

# SVM configuration
SVM_CONFIG = {
    'kernel': 'rbf',
    'C': 1.0,
    'gamma': 'scale',
    'probability': True,
    'random_state': 42
}

# Isolation Forest configuration
IF_CONFIG = {
    'n_estimators': 100,
    'max_samples': 'auto',
    'contamination': 0.3,  # 30% jamming attacks in dataset
    'max_features': 1.0,
    'bootstrap': False,
    'random_state': 42
}

# =====================================================
# FEATURE ENGINEERING
# =====================================================

# 15 engineered features as specified in paper
FEATURE_NAMES = [
    # Signal Quality Features (5)
    'rsrp_dbm',          # Reference Signal Received Power
    'rsrq_db',           # Reference Signal Received Quality  
    'sinr_db',           # Signal-to-Interference-plus-Noise Ratio
    'cqi',               # Channel Quality Indicator (1-15)
    'pmi',               # Precoding Matrix Indicator
    
    # Traffic Pattern Features (4)
    'prb_utilization',   # Physical Resource Block usage (%)
    'throughput_mbps',   # Current data throughput
    'packet_loss_rate',  # Lost packet percentage
    'latency_ms',        # Round-trip latency
    
    # Temporal Features (3)
    'rsrp_variance',     # Signal stability over time window
    'throughput_variance', # Traffic stability measure
    'cqi_trend',         # Quality trend indicator (-1, 0, 1)
    
    # Resource Utilization Features (3)
    'active_ues',        # Number of connected devices
    'scheduling_requests', # Resource request rate
    'buffer_status'      # Average buffer occupancy (%)
]

# Feature value ranges for normalization
FEATURE_RANGES = {
    'rsrp_dbm': (-140, -60),
    'rsrq_db': (-20, 3),
    'sinr_db': (-10, 40),
    'cqi': (1, 15),
    'pmi': (0, 15),
    'prb_utilization': (0, 100),
    'throughput_mbps': (0, 1000),
    'packet_loss_rate': (0, 100),
    'latency_ms': (1, 500),
    'rsrp_variance': (0, 100),
    'throughput_variance': (0, 10000),
    'cqi_trend': (-1, 1),
    'active_ues': (1, 200),
    'scheduling_requests': (0, 1000),
    'buffer_status': (0, 100)
}

# =====================================================
# DETECTION THRESHOLDS
# =====================================================

DETECTION_THRESHOLDS = {
    'ensemble_confidence': 0.7,     # Minimum confidence for detection
    'individual_confidence': 0.6,   # Threshold for individual models
    'anomaly_threshold': -0.1,      # Isolation Forest threshold
    'consensus_threshold': 2        # Minimum models agreeing
}

# =====================================================
# NETWORK ENVIRONMENTS
# =====================================================

NETWORK_ENVIRONMENTS = {
    'ideal': {
        'noise_floor': -110,         # dBm
        'interference_level': 0.1,   # Low interference
        'channel_conditions': 'excellent',
        'mobility': 'static'
    },
    'moderate': {
        'noise_floor': -105,         # dBm
        'interference_level': 0.3,   # Moderate interference
        'channel_conditions': 'good',
        'mobility': 'low'
    },
    'realistic': {
        'noise_floor': -100,         # dBm  
        'interference_level': 0.5,   # High interference
        'channel_conditions': 'varying',
        'mobility': 'high'
    }
}

# =====================================================
# JAMMING ATTACK TYPES
# =====================================================

JAMMING_TYPES = {
    'normal': {
        'id': 0,
        'name': 'Normal Traffic',
        'description': 'Legitimate network traffic without interference'
    },
    'power_jamming': {
        'id': 1,
        'name': 'Power Jamming',
        'description': 'High-power broadband interference',
        'characteristics': {
            'power_increase': (20, 40),  # dB above normal
            'frequency_range': 'broadband',
            'duration': (5, 60),         # seconds
            'detection_difficulty': 'easy'
        }
    },
    'sweep_jamming': {
        'id': 2,
        'name': 'Sweep Jamming', 
        'description': 'Frequency-sweeping interference pattern',
        'characteristics': {
            'power_increase': (10, 25),  # dB above normal
            'frequency_range': 'swept',
            'duration': (3, 30),         # seconds
            'detection_difficulty': 'medium'
        }
    },
    'intelligent_jamming': {
        'id': 3,
        'name': 'Intelligent Jamming',
        'description': 'Adaptive traffic-aware interference',
        'characteristics': {
            'power_increase': (5, 15),   # dB above normal
            'frequency_range': 'selective',
            'duration': (10, 120),       # seconds
            'detection_difficulty': 'hard'
        }
    }
}

# =====================================================
# O-RAN INTEGRATION
# =====================================================

ORAN_CONFIG = {
    'xapp_name': 'jamming-detector',
    'xapp_version': '1.0.0',
    'ric_platform': 'flexric',
    'e2_interface_version': 'v2.0',
    'subscription_intervals': {
        'mac_metrics': 100,          # ms
        'rlc_metrics': 500,          # ms
        'pdcp_metrics': 1000         # ms
    }
}

# E2 Interface configuration
E2_INTERFACE_CONFIG = {
    'node_id': 'gnb_001',
    'plmn_id': '001001',
    'cell_ids': [1, 2, 3],
    'reporting_period_ms': 100,
    'subscription_timeout_s': 60
}

# =====================================================
# SYSTEM PERFORMANCE
# =====================================================

PERFORMANCE_CONFIG = {
    'max_detection_latency_ms': 100,
    'max_memory_usage_mb': 512,
    'max_cpu_usage_percent': 25,
    'monitoring_interval_ms': 100,
    'metrics_buffer_size': 1000,
    'log_retention_days': 30
}

# =====================================================
# LOGGING CONFIGURATION
# =====================================================

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_path': 'logs/jamming_detection.log',
    'max_file_size_mb': 100,
    'backup_count': 5,
    'console_output': True
}

# =====================================================
# VISUALIZATION SETTINGS
# =====================================================

VISUALIZATION_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'color_scheme': {
        'normal': '#2E8B57',        # Sea Green
        'power_jamming': '#DC143C',  # Crimson
        'sweep_jamming': '#FF8C00',  # Dark Orange
        'intelligent_jamming': '#9400D3'  # Violet
    },
    'save_formats': ['png', 'pdf', 'svg'],
    'output_directory': 'plots'
}

# =====================================================
# DATA CONFIGURATION
# =====================================================

DATA_CONFIG = {
    'dataset_size': 25000,
    'train_split': 0.7,
    'validation_split': 0.15,
    'test_split': 0.15,
    'class_distribution': {
        'normal': 0.70,
        'power_jamming': 0.10,
        'sweep_jamming': 0.10,
        'intelligent_jamming': 0.10
    },
    'sampling_frequency_hz': 1000,
    'time_window_ms': 100
}

# =====================================================
# TESTING CONFIGURATION  
# =====================================================

TESTING_CONFIG = {
    'performance_test_iterations': 1000,
    'latency_test_samples': 5000,
    'stress_test_duration_s': 300,
    'accuracy_tolerance': 0.02,      # 2% tolerance
    'latency_tolerance_ms': 10,      # 10ms tolerance
    'memory_limit_mb': 1024
}

# =====================================================
# UTILITY FUNCTIONS
# =====================================================

def get_config_value(key_path: str, default: Any = None) -> Any:
    """
    Get configuration value using dot notation.
    
    Args:
        key_path: Dot-separated key path (e.g., 'RF_CONFIG.n_estimators')
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    try:
        keys = key_path.split('.')
        value = globals()[keys[0]]
        
        for key in keys[1:]:
            value = value[key]
            
        return value
    except (KeyError, TypeError):
        return default

def validate_config() -> bool:
    """
    Validate configuration consistency.
    
    Returns:
        True if configuration is valid
    """
    # Check ensemble weights sum to 1.0
    weight_sum = sum(ENSEMBLE_WEIGHTS.values())
    if abs(weight_sum - 1.0) > 0.001:
        print(f"Warning: Ensemble weights sum to {weight_sum}, not 1.0")
        return False
    
    # Check class distribution sums to 1.0
    class_sum = sum(DATA_CONFIG['class_distribution'].values())
    if abs(class_sum - 1.0) > 0.001:
        print(f"Warning: Class distribution sums to {class_sum}, not 1.0")
        return False
    
    # Check feature count
    if len(FEATURE_NAMES) != 15:
        print(f"Warning: Expected 15 features, got {len(FEATURE_NAMES)}")
        return False
    
    return True

# =====================================================
# EXPORT CONFIGURATION
# =====================================================

def get_all_config() -> Dict[str, Any]:
    """Get all configuration as dictionary."""
    return {
        'PROJECT_NAME': PROJECT_NAME,
        'VERSION': VERSION,
        'PERFORMANCE_TARGETS': PERFORMANCE_TARGETS,
        'ENSEMBLE_WEIGHTS': ENSEMBLE_WEIGHTS,
        'RF_CONFIG': RF_CONFIG,
        'SVM_CONFIG': SVM_CONFIG,
        'IF_CONFIG': IF_CONFIG,
        'FEATURE_NAMES': FEATURE_NAMES,
        'FEATURE_RANGES': FEATURE_RANGES,
        'DETECTION_THRESHOLDS': DETECTION_THRESHOLDS,
        'NETWORK_ENVIRONMENTS': NETWORK_ENVIRONMENTS,
        'JAMMING_TYPES': JAMMING_TYPES,
        'ORAN_CONFIG': ORAN_CONFIG,
        'E2_INTERFACE_CONFIG': E2_INTERFACE_CONFIG,
        'PERFORMANCE_CONFIG': PERFORMANCE_CONFIG,
        'LOGGING_CONFIG': LOGGING_CONFIG,
        'VISUALIZATION_CONFIG': VISUALIZATION_CONFIG,
        'DATA_CONFIG': DATA_CONFIG,
        'TESTING_CONFIG': TESTING_CONFIG
    }

# Validate configuration on import
if __name__ == "__main__":
    if validate_config():
        print("✓ Configuration validation passed")
    else:
        print("✗ Configuration validation failed")
