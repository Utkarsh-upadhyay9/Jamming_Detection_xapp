import numpy as np

NETWORK_ENVIRONMENTS = {
    'ideal': {
        'description': 'Baseline scenario with minimal environmental interference',
        'noise_floor': 1.0,                    # σ²_noise = 1.0
        'rayleigh_sigma': 1.0,                 # Rayleigh fading parameter
        'interference_level_dbm': -110,        # I₀ = -110 dBm
        'snr_enhancement_factor': 1.2,         # β = 1.2
        'environmental_degradation': 0.0,      # δ = 0.0 (no degradation)
        'multipath_enabled': False
    },
    
    'moderate': {
        'description': 'Scenario with controlled interference',
        'noise_floor': 4.0,                    # σ²_noise = 4.0
        'rayleigh_sigma': 0.8,                 # Rayleigh fading parameter
        'interference_level_dbm': -110,        # I₀ = -110 dBm
        'snr_enhancement_factor': 1.0,         # β = 1.0
        'environmental_degradation': 0.15,     # δ = 0.15
        'multipath_enabled': False
    },
    
    'realistic': {
        'description': 'Highly challenging scenario reflecting practical deployment',
        'noise_floor': 4.0,                    # σ²_noise = 4.0
        'rayleigh_sigma': 0.8,                 # Rayleigh fading parameter
        'interference_level_dbm': -100,        # I₀ = -100 dBm
        'snr_enhancement_factor': 0.7,         # β = 0.7
        'environmental_degradation': 0.35,     # δ = 0.35
        'multipath_enabled': True,
        'delay_spread_lambda': 0.1             # Exponential delay spread λ = 0.1
    }
}

JAMMING_SCENARIOS = {
    'power_jamming': {
        'description': 'High-power broadband noise overwhelming legitimate signals',
        'power_level_dbm': 10,                 # High power level
        'bandwidth_ratio': 1.0,                # Full spectrum coverage
        'temporal_pattern': 'continuous',
        'spectral_pattern': 'broadband',
        'detection_difficulty': 'easy'
    },
    
    'sweep_jamming': {
        'description': 'Cyclically scanning across frequency bands',
        'power_level_dbm': 5,                  # Moderate power level
        'bandwidth_ratio': 0.3,                # Partial spectrum coverage
        'temporal_pattern': 'periodic',
        'spectral_pattern': 'frequency_hopping',
        'sweep_period_ms': 100,                # Sweep period
        'detection_difficulty': 'moderate'
    },
    
    'intelligent_jamming': {
        'description': 'Adaptive interference targeting specific slots',
        'power_level_dbm': 0,                  # Variable power level
        'bandwidth_ratio': 0.1,                # Targeted spectrum usage
        'temporal_pattern': 'adaptive',
        'spectral_pattern': 'selective',
        'adaptation_rate': 0.8,                # High adaptation capability
        'detection_difficulty': 'hard'
    }
}

SIGNAL_QUALITY_REFERENCE = {
    'sinr_mean_ref': 15.0,                     # Reference SINR (dB)
    'sinr_std_ref': 2.0,                       # Reference SINR std dev
    'rsrp_mean_ref': -90.0,                    # Reference RSRP (dBm)
    'rsrp_std_ref': 5.0,                       # Reference RSRP std dev
    'rsrq_mean_ref': -10.0,                    # Reference RSRQ (dB)
    'rssi_std_ref': 3.0,                       # Reference RSSI std dev
    'psd_variance_ref': 1.0                    # Reference PSD variance
}

NETWORK_TOPOLOGY = {
    'cell_radius_m': 1000,                     # Cell coverage radius
    'base_station_height_m': 30,               # BS antenna height
    'user_equipment_height_m': 1.5,            # UE antenna height
    'carrier_frequency_ghz': 3.5,              # 5G carrier frequency
    'channel_bandwidth_mhz': 20,               # Channel bandwidth
    'num_prbs': 100,                           # Physical Resource Blocks
    'subcarrier_spacing_khz': 15               # Subcarrier spacing
}

E2_INTERFACE_CONFIG = {
    'reporting_interval_ms': 1000,             # 1 second reporting interval
    'max_metrics_per_report': 15,              # Maximum metrics per report
    'connection_timeout_s': 30,                # Connection timeout
    'retry_attempts': 3,                       # Connection retry attempts
    'buffer_size': 1024                        # Data buffer size
}

QOS_REQUIREMENTS = {
    'urllc': {
        'latency_ms': 1,                       # Ultra-low latency
        'reliability': 0.99999,                # 99.999% reliability
        'throughput_mbps': 1                   # Minimum throughput
    },
    'embb': {
        'latency_ms': 10,                      # Enhanced mobile broadband
        'reliability': 0.999,                  # 99.9% reliability
        'throughput_mbps': 100                 # High throughput
    },
    'mmtc': {
        'latency_ms': 1000,                    # Massive machine-type communications
        'reliability': 0.99,                   # 99% reliability
        'throughput_kbps': 100                 # Low throughput
    }
}

def generate_rayleigh_fading(sigma, size=1):
    return np.random.rayleigh(sigma, size)

def generate_exponential_delay_spread(lambda_param, size=1):
    return np.random.exponential(1/lambda_param, size)

def apply_environmental_degradation(signal, degradation_factor):
    noise = np.random.normal(0, degradation_factor, len(signal))
    return signal + noise

def calculate_path_loss(distance_m, frequency_ghz):
    path_loss_db = 20 * np.log10(distance_m) + 20 * np.log10(frequency_ghz) + 32.45
    return path_loss_db
