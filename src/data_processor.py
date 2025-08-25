import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Dict, List, Tuple, Any, Optional
import warnings

warnings.filterwarnings('ignore')

class JammingDataProcessor:
    def __init__(self, normalization_method: str = 'zscore'):
        self.normalization_method = normalization_method
        self.scaler = self._get_scaler()
        self.feature_names = self._get_feature_names()
        self.is_fitted = False
        
        self.feature_stats = {}
        
    def _get_scaler(self):
        if self.normalization_method == 'zscore':
            return StandardScaler()
        elif self.normalization_method == 'minmax':
            return MinMaxScaler()
        elif self.normalization_method == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unsupported normalization method: {self.normalization_method}")
    
    def _get_feature_names(self) -> List[str]:
        return [
            'sinr_mean', 'sinr_std', 'rsrp_mean', 'rsrp_std', 'rsrq_mean',
            
            'ul_throughput', 'dl_throughput', 'retx_rate', 'packet_loss_rate',
            
            'inter_arrival_variance', 'buffer_occupancy',
            
            'prb_utilization_ul', 'prb_utilization_dl', 'grant_count', 'cqi_variance'
        ]
    
    def load_dataset(self, normal_traffic_path: str, jamming_attacks_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        normal_data = pd.read_csv(normal_traffic_path)
        jamming_data = pd.read_csv(jamming_attacks_path)
        
        print(f"Loaded {len(normal_data)} normal traffic samples")
        print(f"Loaded {len(jamming_data)} jamming attack samples")
        
        return normal_data, jamming_data
    
    def prepare_dataset(self, normal_data: pd.DataFrame, jamming_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        normal_data = normal_data.copy()
        jamming_data = jamming_data.copy()
        
        normal_data['label'] = 'normal'
        
        jamming_type_mapping = {
            'power': 'power_jamming',
            'sweep': 'sweep_jamming', 
            'intelligent': 'intelligent_jamming'
        }
        
        jamming_data['label'] = jamming_data['jamming_type'].map(jamming_type_mapping)
        
        combined_data = pd.concat([normal_data, jamming_data], ignore_index=True)
        
        feature_columns = [col for col in combined_data.columns if col not in ['label', 'jamming_type', 'environment']]
        features = combined_data[feature_columns].values
        labels = combined_data['label'].values
        
        print(f"Prepared dataset with {len(features)} samples and {features.shape[1]} features")
        print(f"Label distribution:")
        for label in np.unique(labels):
            count = np.sum(labels == label)
            print(f"  {label}: {count} ({count/len(labels)*100:.1f}%)")
        
        return features, labels
    
    def engineer_features(self, raw_data: Dict[str, np.ndarray]) -> np.ndarray:
        features = np.zeros(15)
        
        features[0] = self._calculate_sinr_mean(raw_data)
        features[1] = self._calculate_sinr_std(raw_data)
        features[2] = self._calculate_rsrp_mean(raw_data)
        features[3] = self._calculate_rsrp_std(raw_data)
        features[4] = self._calculate_rsrq_mean(raw_data)
        
        features[5] = self._calculate_ul_throughput(raw_data)
        features[6] = self._calculate_dl_throughput(raw_data)
        features[7] = self._calculate_retx_rate(raw_data)
        features[8] = self._calculate_packet_loss_rate(raw_data)
        
        features[9] = self._calculate_inter_arrival_variance(raw_data)
        features[10] = self._calculate_buffer_occupancy(raw_data)
        
        features[11] = self._calculate_prb_utilization_ul(raw_data)
        features[12] = self._calculate_prb_utilization_dl(raw_data)
        features[13] = self._calculate_grant_count(raw_data)
        features[14] = self._calculate_cqi_variance(raw_data)
        
        return features
    
    def _calculate_sinr_mean(self, data: Dict[str, np.ndarray]) -> float:
        sinr_values = data.get('sinr', np.array([0]))
        return np.mean(sinr_values)
    
    def _calculate_sinr_std(self, data: Dict[str, np.ndarray]) -> float:
        sinr_values = data.get('sinr', np.array([0]))
        return np.std(sinr_values)
    
    def _calculate_rsrp_mean(self, data: Dict[str, np.ndarray]) -> float:
        rsrp_values = data.get('rsrp', np.array([-100]))
        return np.mean(rsrp_values)
    
    def _calculate_rsrp_std(self, data: Dict[str, np.ndarray]) -> float:
        rsrp_values = data.get('rsrp', np.array([-100]))
        return np.std(rsrp_values)
    
    def _calculate_rsrq_mean(self, data: Dict[str, np.ndarray]) -> float:
        rsrq_values = data.get('rsrq', np.array([-10]))
        return np.mean(rsrq_values)
    
    def _calculate_ul_throughput(self, data: Dict[str, np.ndarray]) -> float:
        ul_bytes = data.get('ul_bytes', np.array([0]))
        time_interval = data.get('time_interval', 1.0)  # seconds
        return np.sum(ul_bytes) * 8 / time_interval  # bits per second
    
    def _calculate_dl_throughput(self, data: Dict[str, np.ndarray]) -> float:
        dl_bytes = data.get('dl_bytes', np.array([0]))
        time_interval = data.get('time_interval', 1.0)  # seconds
        return np.sum(dl_bytes) * 8 / time_interval  # bits per second
    
    def _calculate_retx_rate(self, data: Dict[str, np.ndarray]) -> float:
        retx_count = data.get('retx_count', np.array([0]))
        total_tx = data.get('total_tx', np.array([1]))
        total_tx = np.maximum(total_tx, 1)  # Avoid division by zero
        return np.sum(retx_count) / np.sum(total_tx)
    
    def _calculate_packet_loss_rate(self, data: Dict[str, np.ndarray]) -> float:
        lost_packets = data.get('lost_packets', np.array([0]))
        total_packets = data.get('total_packets', np.array([1]))
        total_packets = np.maximum(total_packets, 1)  # Avoid division by zero
        return np.sum(lost_packets) / np.sum(total_packets)
    
    def _calculate_inter_arrival_variance(self, data: Dict[str, np.ndarray]) -> float:
        arrival_times = data.get('arrival_times', np.array([0]))
        if len(arrival_times) < 2:
            return 0.0
        inter_arrival_times = np.diff(arrival_times)
        return np.var(inter_arrival_times)
    
    def _calculate_buffer_occupancy(self, data: Dict[str, np.ndarray]) -> float:
        buffer_size = data.get('buffer_size', np.array([1]))
        buffer_capacity = data.get('buffer_capacity', 1000)
        return np.mean(buffer_size) / buffer_capacity
    
    def _calculate_prb_utilization_ul(self, data: Dict[str, np.ndarray]) -> float:
        used_prbs_ul = data.get('used_prbs_ul', np.array([0]))
        total_prbs = data.get('total_prbs', 100)
        return np.mean(used_prbs_ul) / total_prbs
    
    def _calculate_prb_utilization_dl(self, data: Dict[str, np.ndarray]) -> float:
        used_prbs_dl = data.get('used_prbs_dl', np.array([0]))
        total_prbs = data.get('total_prbs', 100)
        return np.mean(used_prbs_dl) / total_prbs
    
    def _calculate_grant_count(self, data: Dict[str, np.ndarray]) -> float:
        grants = data.get('scheduling_grants', np.array([0]))
        return np.sum(grants)
    
    def _calculate_cqi_variance(self, data: Dict[str, np.ndarray]) -> float:
        cqi_values = data.get('cqi', np.array([7]))  # Default CQI value
        return np.var(cqi_values)
    
    def detect_periodic_patterns(self, frequency_sequence: np.ndarray, 
                                threshold: float = 0.7) -> float:
        if len(frequency_sequence) < 4:
            return 0.0
        
        fft_result = np.abs(fft(frequency_sequence))
        
        dominant_freq_power = np.max(fft_result[1:len(fft_result)//2])  # Exclude DC component
        total_power = np.sum(fft_result[1:len(fft_result)//2])
        
        if total_power == 0:
            return 0.0
        
        periodicity_score = dominant_freq_power / total_power
        return periodicity_score
    
    def calculate_temporal_correlation(self, features: np.ndarray, 
                                     window_size: int = 10) -> float:
        if len(features) < window_size:
            return 0.0
        
        correlations = []
        for lag in range(1, min(window_size, len(features))):
            if len(features) > lag:
                corr = np.corrcoef(features[:-lag], features[lag:])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0
    
    def calculate_adaptivity_score(self, features: np.ndarray,
                                 traffic_patterns: np.ndarray) -> float:
        if len(features) < 2 or len(traffic_patterns) < 2:
            return 0.0
        
        try:
            correlation = np.corrcoef(features, traffic_patterns)[0, 1]
            return abs(correlation) if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def normalize_features(self, features: np.ndarray, 
                          fit: bool = False) -> np.ndarray:
        if fit:
            normalized = self.scaler.fit_transform(features)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler must be fitted before transforming")
            normalized = self.scaler.transform(features)
        
        return normalized
    
    def preprocess_for_training(self, features: np.ndarray, labels: np.ndarray,
                               test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, 
            random_state=random_state, stratify=labels
        )
        
        # Normalize features
        X_train_norm = self.normalize_features(X_train, fit=True)
        X_test_norm = self.normalize_features(X_test, fit=False)
        
        self._calculate_feature_statistics(X_train_norm, y_train)
        
        return {
            'X_train': X_train_norm,
            'X_test': X_test_norm,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.feature_names,
            'scaler': self.scaler
        }
    
    def _calculate_feature_statistics(self, features: np.ndarray, labels: np.ndarray):
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            mask = labels == label
            label_features = features[mask]
            
            self.feature_stats[label] = {
                'mean': np.mean(label_features, axis=0),
                'std': np.std(label_features, axis=0),
                'min': np.min(label_features, axis=0),
                'max': np.max(label_features, axis=0)
            }
    
    def get_feature_statistics(self) -> Dict[str, Dict[str, np.ndarray]]:
        return self.feature_stats.copy()
    
    def simulate_network_environment(self, base_features: np.ndarray, 
                                   environment: str = 'realistic') -> np.ndarray:
        features = base_features.copy()
        
        if environment == 'ideal':
            noise_scale = 0.01
        elif environment == 'moderate':
            noise_scale = 0.05
        elif environment == 'realistic':
            noise_scale = 0.1
        else:
            raise ValueError(f"Unknown environment: {environment}")
        
        noise = np.random.normal(0, noise_scale, features.shape)
        features += noise
        
        if environment == 'realistic':
            fading = np.random.rayleigh(0.8, features.shape[0])
            features[:, 0] *= fading  # SINR mean
            features[:, 2] *= fading  # RSRP mean
            
            features[:, 7] += np.random.exponential(0.02, features.shape[0])  # Retx rate
            features[:, 8] += np.random.exponential(0.01, features.shape[0])  # Packet loss
        
        return features
    
    def create_feature_windows(self, features: np.ndarray, 
                              window_size: int = 10, 
                              stride: int = 1) -> np.ndarray:
        n_samples, n_features = features.shape
        n_windows = (n_samples - window_size) // stride + 1
        
        windowed_features = np.zeros((n_windows, window_size, n_features))
        
        for i in range(n_windows):
            start_idx = i * stride
            end_idx = start_idx + window_size
            windowed_features[i] = features[start_idx:end_idx]
        
        return windowed_features
    
    def get_feature_importance_analysis(self, features: np.ndarray, 
                                      labels: np.ndarray) -> Dict[str, float]:
        importance_scores = {}
        
        for i, feature_name in enumerate(self.feature_names):
            feature_values = features[:, i]
            
            variance_score = np.var(feature_values)
            
            unique_labels = np.unique(labels)
            groups = [feature_values[labels == label] for label in unique_labels]
            
            try:
                f_stat, _ = stats.f_oneway(*groups)
                separability_score = f_stat if not np.isnan(f_stat) else 0
            except:
                separability_score = 0
            
            importance_scores[feature_name] = {
                'variance': variance_score,
                'separability': separability_score,
                'combined_score': variance_score * separability_score
            }
        
        return importance_scores
