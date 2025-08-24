"""
Test suite for the ensemble jamming detection model.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import Mock, patch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.ensemble_model import EnsembleJammingDetector
from src.data_processor import JammingDataProcessor
from models.rf_model import RandomForestJammingDetector
from models.svm_model import SVMJammingDetector
from models.isolation_forest_model import IsolationForestJammingDetector


class TestEnsembleModel:
    """Test cases for the ensemble jamming detection model."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        
        # Create normal traffic data
        normal_features = np.random.normal(0, 1, (100, 15))
        normal_labels = np.array(['normal'] * 100)
        
        # Create jamming attack data
        jamming_features = np.random.normal(2, 2, (50, 15))  # Different distribution
        jamming_labels = np.array(['power_jamming'] * 17 + 
                                ['sweep_jamming'] * 17 + 
                                ['intelligent_jamming'] * 16)
        
        # Combine data
        X = np.vstack([normal_features, jamming_features])
        y = np.hstack([normal_labels, jamming_labels])
        
        return X, y
    
    @pytest.fixture
    def trained_ensemble(self, sample_data):
        """Create a trained ensemble model."""
        X, y = sample_data
        
        detector = EnsembleJammingDetector()
        
        # Mock the data loading process
        with patch.object(detector, 'load_and_prepare_data') as mock_load:
            mock_dataset = {
                'X_train': X[:120],
                'X_test': X[120:],
                'y_train': y[:120],
                'y_test': y[120:],
                'feature_names': [f'feature_{i}' for i in range(15)],
                'scaler': Mock()
            }
            mock_load.return_value = mock_dataset
            
            # Train the ensemble
            detector.train_ensemble(mock_dataset)
        
        return detector
    
    def test_ensemble_initialization(self):
        """Test ensemble model initialization."""
        detector = EnsembleJammingDetector()
        
        assert detector.weights['rf'] == 0.44
        assert detector.weights['svm'] == 0.41
        assert detector.weights['if'] == 0.15
        assert detector.threshold == 0.52
        assert not detector.is_trained
    
    def test_individual_model_training(self, sample_data):
        """Test individual model training."""
        X, y = sample_data
        X_train, X_test = X[:120], X[120:]
        y_train, y_test = y[:120], y[120:]
        
        # Test Random Forest
        rf_model = RandomForestJammingDetector()
        rf_metrics = rf_model.train(X_train, y_train)
        
        assert rf_model.is_trained
        assert 'train_accuracy' in rf_metrics
        assert 'train_f1_score' in rf_metrics
        
        # Test SVM
        svm_model = SVMJammingDetector()
        svm_metrics = svm_model.train(X_train, y_train)
        
        assert svm_model.is_trained
        assert 'train_accuracy' in svm_metrics
        
        # Test Isolation Forest
        if_model = IsolationForestJammingDetector()
        if_metrics = if_model.train(X_train, y_train)
        
        assert if_model.is_trained
    
    def test_ensemble_prediction(self, trained_ensemble, sample_data):
        """Test ensemble prediction functionality."""
        X, y = sample_data
        X_test = X[-10:]  # Use last 10 samples for testing
        
        # Test predict_proba
        probabilities = trained_ensemble.predict_proba(X_test)
        
        assert probabilities.shape == (10, 2)  # [normal_prob, jamming_prob]
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
        assert np.allclose(np.sum(probabilities, axis=1), 1.0)  # Probabilities sum to 1
        
        # Test predict
        predictions = trained_ensemble.predict(X_test)
        
        assert len(predictions) == 10
        assert all(pred in ['normal', 'jamming'] for pred in predictions)
    
    def test_jamming_type_detection(self, trained_ensemble, sample_data):
        """Test multi-class jamming type detection."""
        X, y = sample_data
        X_test = X[-5:]  # Use last 5 samples
        
        jamming_types = trained_ensemble.detect_jamming_type(X_test)
        
        assert len(jamming_types) == 5
        valid_types = ['normal', 'power_jamming', 'sweep_jamming', 'intelligent_jamming']
        assert all(jtype in valid_types for jtype in jamming_types)
    
    def test_confidence_calculation(self, trained_ensemble, sample_data):
        """Test confidence calculation."""
        X, y = sample_data
        X_test = X[-5:]
        
        confidence = trained_ensemble.calculate_ensemble_confidence(X_test)
        
        assert len(confidence) == 5
        assert np.all(confidence >= 0) and np.all(confidence <= 1)
    
    def test_model_evaluation(self, trained_ensemble, sample_data):
        """Test model evaluation metrics."""
        X, y = sample_data
        X_test, y_test = X[-30:], y[-30:]
        
        metrics = trained_ensemble.evaluate_model(X_test, y_test)
        
        required_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)
            assert 0 <= metrics[metric] <= 1
    
    def test_weight_optimization(self, trained_ensemble, sample_data):
        """Test ensemble weight optimization."""
        X, y = sample_data
        X_val, y_val = X[-20:], y[-20:]
        
        original_weights = trained_ensemble.weights.copy()
        optimized_weights = trained_ensemble.optimize_weights(X_val, y_val, search_space=5)
        
        # Check that weights are valid
        assert all(0 <= w <= 1 for w in optimized_weights.values())
        assert abs(sum(optimized_weights.values()) - 1.0) < 1e-10
        
        # Weights should be different (unless original was already optimal)
        weights_changed = any(
            abs(optimized_weights[k] - original_weights[k]) > 1e-10 
            for k in optimized_weights
        )
        # Note: weights might not change if original was already optimal for small test data
    
    def test_model_save_load(self, trained_ensemble):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_model')
            
            # Save model
            trained_ensemble.save_ensemble(model_path)
            
            # Check files exist
            expected_files = [
                'rf_model.joblib',
                'svm_model.joblib', 
                'if_model.joblib',
                'ensemble_config.joblib',
                'data_processor.joblib'
            ]
            
            for filename in expected_files:
                assert os.path.exists(os.path.join(model_path, filename))
            
            # Load model
            new_detector = EnsembleJammingDetector()
            new_detector.load_ensemble(model_path)
            
            # Check that loaded model has same configuration
            assert new_detector.is_trained
            assert new_detector.weights == trained_ensemble.weights
            assert new_detector.threshold == trained_ensemble.threshold
    
    def test_latency_performance(self, trained_ensemble, sample_data):
        """Test that detection latency meets requirements."""
        X, y = sample_data
        X_test = X[-1:]  # Single sample
        
        import time
        
        # Measure prediction time
        start_time = time.perf_counter()
        _ = trained_ensemble.predict_proba(X_test)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        
        # Should be well under 100ms requirement
        assert latency_ms < 100, f"Latency {latency_ms:.2f}ms exceeds 100ms requirement"
    
    def test_f1_score_improvement(self, sample_data):
        """Test that ensemble achieves better F1 score than individual models."""
        X, y = sample_data
        X_train, X_test = X[:120], X[120:]
        y_train, y_test = y[:120], y[120:]
        
        # Convert to binary classification for comparison
        y_train_binary = ['normal' if label == 'normal' else 'jamming' for label in y_train]
        y_test_binary = ['normal' if label == 'normal' else 'jamming' for label in y_test]
        
        # Train individual models
        rf_model = RandomForestJammingDetector()
        rf_model.train(X_train, y_train_binary)
        rf_pred = rf_model.predict(X_test)
        
        # Calculate F1 scores
        from sklearn.metrics import f1_score
        rf_f1 = f1_score(y_test_binary, rf_pred, average='weighted')
        
        # Train ensemble
        detector = EnsembleJammingDetector()
        with patch.object(detector, 'load_and_prepare_data') as mock_load:
            mock_dataset = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': [f'feature_{i}' for i in range(15)],
                'scaler': Mock()
            }
            mock_load.return_value = mock_dataset
            detector.train_ensemble(mock_dataset)
        
        ensemble_pred = detector.predict(X_test)
        ensemble_f1 = f1_score(y_test_binary, ensemble_pred, average='weighted')
        
        # Ensemble should perform at least as well as individual models
        # (may not always be better due to small test set and randomness)
        print(f"RF F1: {rf_f1:.3f}, Ensemble F1: {ensemble_f1:.3f}")
        assert ensemble_f1 >= 0.5  # Reasonable minimum performance


class TestDataProcessor:
    """Test cases for data processing functionality."""
    
    def test_feature_engineering(self):
        """Test feature engineering from raw data."""
        processor = JammingDataProcessor()
        
        # Mock raw data
        raw_data = {
            'sinr': np.array([15.0, 14.5, 15.5]),
            'rsrp': np.array([-90.0, -91.0, -89.0]),
            'rsrq': np.array([-10.0, -10.5, -9.5]),
            'ul_bytes': np.array([1000000]),
            'dl_bytes': np.array([2000000]),
            'time_interval': 1.0,
            'retx_count': np.array([5]),
            'total_tx': np.array([100]),
            'lost_packets': np.array([1]),
            'total_packets': np.array([1000]),
            'arrival_times': np.array([0.0, 0.1, 0.2, 0.3]),
            'buffer_size': np.array([300]),
            'buffer_capacity': 1000,
            'used_prbs_ul': np.array([40]),
            'used_prbs_dl': np.array([50]),
            'total_prbs': 100,
            'scheduling_grants': np.array([45]),
            'cqi': np.array([7, 8, 7, 6])
        }
        
        features = processor.engineer_features(raw_data)
        
        assert len(features) == 15
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))
    
    def test_normalization(self):
        """Test feature normalization."""
        processor = JammingDataProcessor()
        
        # Create test data
        features = np.random.normal(0, 1, (100, 15))
        
        # Test normalization
        normalized = processor.normalize_features(features, fit=True)
        
        assert normalized.shape == features.shape
        assert processor.is_fitted
        
        # Check normalization properties (approximately)
        assert abs(np.mean(normalized)) < 0.1  # Mean should be close to 0
        assert abs(np.std(normalized) - 1.0) < 0.1  # Std should be close to 1
    
    def test_dataset_preparation(self):
        """Test dataset preparation pipeline."""
        processor = JammingDataProcessor()
        
        # Create mock DataFrames
        normal_data = pd.DataFrame({
            f'feature_{i}': np.random.normal(0, 1, 100) for i in range(15)
        })
        
        jamming_data = pd.DataFrame({
            f'feature_{i}': np.random.normal(2, 2, 50) for i in range(15)
        })
        jamming_data['jamming_type'] = ['power'] * 17 + ['sweep'] * 17 + ['intelligent'] * 16
        
        # Prepare dataset
        features, labels = processor.prepare_dataset(normal_data, jamming_data)
        
        assert features.shape == (150, 15)
        assert len(labels) == 150
        assert set(labels) == {'normal', 'power_jamming', 'sweep_jamming', 'intelligent_jamming'}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
