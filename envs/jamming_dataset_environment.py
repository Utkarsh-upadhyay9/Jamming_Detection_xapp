import os
import numpy as np
import pandas as pd
from gymnasium import spaces
import gymnasium as gym
from typing import Dict, List, Optional, Any
from sklearn.preprocessing import StandardScaler


class JammingDatasetEnvironment(gym.Env):
    """Offline dataset-driven environment for DRL fine-tuning with captured USRP data."""

    metadata = {"render.modes": []}

    def __init__(self, csv_path: str, shuffle: bool = True, episode_length: Optional[int] = None, scale_features: bool = True):
        super().__init__()
        # Load dataset
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(csv_path)
        self.data = pd.read_csv(csv_path)

        # Derive jamming label column
        if 'jamming_active' in self.data.columns:
            self.data['__jammer__'] = self.data['jamming_active'].astype(int)
        elif 'label' in self.data.columns:
            mapping = {
                'normal': 0,
                'power_jamming': 1,
                'sweep_jamming': 1,
                'intelligent_jamming': 1,
                0: 0, 1: 1
            }
            self.data['__jammer__'] = self.data['label'].map(mapping)
        else:
            raise ValueError("Need 'jamming_active' or 'label' column in dataset")

        # Select numeric feature columns
        exclude = {'timestamp', 'label', 'jamming_active', '__jammer__'}
        self.feature_cols = [c for c in self.data.columns if c not in exclude and np.issubdtype(self.data[c].dtype, np.number)]
        if not self.feature_cols:
            raise ValueError("No numeric feature columns found for state")

        # Define spaces
        self.state_dim = len(self.feature_cols)
        self.action_dim = 5
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

        # Episode / pointer bookkeeping
        self.shuffle = shuffle
        self.indices = np.arange(len(self.data))
        self.ptr = 0
        self.episode_length = episode_length or len(self.data)
        self.current_step = 0
        self.detection_history: List[bool] = []

        # Optional feature scaling
        self.scale_features = scale_features
        if self.scale_features:
            self.scaler = StandardScaler()
            self.scaler.fit(self.data[self.feature_cols].values)
        else:
            self.scaler = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):  # type: ignore[override]
        super().reset(seed=seed)
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.ptr = 0
        self.current_step = 0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        row = self.data.iloc[self.indices[self.ptr]]
        feats = row[self.feature_cols].to_numpy(dtype=np.float32)
        if self.scaler is not None:
            feats = self.scaler.transform([feats])[0]
        return feats.astype(np.float32)

    def step(self, action: np.ndarray):  # type: ignore[override]
        action = np.asarray(action, dtype=np.float32)
        detection_threshold = (action[0] + 1.0) / 2.0
        row = self.data.iloc[self.indices[self.ptr]]
        jam_active = bool(row['__jammer__'])
        feat_vals = row[self.feature_cols].to_numpy(dtype=np.float32)
        if self.scaler is not None:
            feat_vals = self.scaler.transform([feat_vals])[0]
        norm_conf = 1 / (1 + np.exp(-0.01 * np.clip(feat_vals, -500, 500)))
        detection_confidence = float(np.mean(norm_conf))

        if jam_active:
            detection_success = detection_confidence > detection_threshold
            detection_reward = 1.0 if detection_success else -0.5
        else:
            detection_success = detection_confidence <= detection_threshold
            detection_reward = 0.5 if detection_success else -0.2

        stability_bonus = 0.1 if abs(detection_threshold - 0.5) < 0.2 else 0.0
        power_penalty = -0.05 * float(np.sum(action[1:] ** 2))
        reward = detection_reward + stability_bonus + power_penalty

        self.detection_history.append(detection_success)
        if len(self.detection_history) > 2000:
            self.detection_history = self.detection_history[-2000:]

        self.ptr += 1
        self.current_step += 1
        done = self.ptr >= len(self.indices) or self.current_step >= self.episode_length
        info = {
            'jamming_active': jam_active,
            'detection_confidence': detection_confidence,
            'threshold': detection_threshold
        }
        next_state = self._get_state() if not done else np.zeros(self.state_dim, dtype=np.float32)
        return next_state, reward, done, info

    def get_performance_metrics(self) -> Dict[str, float]:
        if len(self.detection_history) < 10:
            return {'detection_rate': 0.0, 'episode_count': 0, 'total_detections': len(self.detection_history)}
        recent = self.detection_history[-500:]
        return {
            'detection_rate': float(np.mean(recent)),
            'episode_count': 1,
            'total_detections': len(self.detection_history)
        }

    def validate_calibrated_performance(self) -> Dict[str, Any]:
        metrics = self.get_performance_metrics()
        if metrics['total_detections'] < 50:
            return {'insufficient_data': True}
        f1_est = metrics['detection_rate']
        improvement = max(0.0, f1_est - 0.954)
        return {
            'overall_success': f1_est >= 0.97 and improvement >= 0.01,
            'f1_score_target_met': f1_est >= 0.98,
            'improvement_achieved': improvement >= 0.01,
            'f1_estimate': f1_est
        }
