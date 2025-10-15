#!/usr/bin/env python3
"""
Hyperparameter Optimization for Jamming Detection Ensemble
Uses Bayesian optimization to find optimal parameters
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer
from catboost import CatBoostClassifier
from sklearn.ensemble import IsolationForest
import optuna
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

class EnsembleHyperparameterOptimizer:
    """Optimize hyperparameters for CatBoost + Isolation Forest ensemble"""
    
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray):
        self.X_train = X_train
        self.y_train = y_train
        self.best_params = {}
        
    def optimize_catboost(self, n_trials: int = 50) -> Dict[str, Any]:
        """Optimize CatBoost hyperparameters"""
        
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 500, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                'random_strength': trial.suggest_float('random_strength', 0, 10),
                'verbose': False,
                'random_seed': 42,
                'thread_count': -1,
                'auto_class_weights': 'Balanced'
            }
            
            model = CatBoostClassifier(**params)
            
            # 5-fold cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(
                model, self.X_train, self.y_train,
                cv=cv, scoring=make_scorer(f1_score, average='weighted'),
                n_jobs=-1
            )
            
            return np.mean(scores)
        
        print("="*70)
        print("OPTIMIZING CATBOOST HYPERPARAMETERS")
        print("="*70)
        
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"\n✅ Best F1-Score: {study.best_value:.4f}")
        print(f"\n📊 Best Parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        self.best_params['catboost'] = study.best_params
        return study.best_params
    
    def optimize_isolation_forest(self, n_trials: int = 30) -> Dict[str, Any]:
        """Optimize Isolation Forest hyperparameters"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_samples': trial.suggest_float('max_samples', 0.5, 1.0),
                'contamination': trial.suggest_float('contamination', 0.1, 0.3),
                'max_features': trial.suggest_float('max_features', 0.5, 1.0),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': 42,
                'n_jobs': -1
            }
            
            model = IsolationForest(**params)
            
            # Fit model
            model.fit(self.X_train)
            
            # Evaluate on training data (anomaly detection)
            y_binary = np.where(self.y_train == 0, 1, -1)  # 0=normal(1), jamming=-1
            predictions = model.predict(self.X_train)
            
            # Calculate F1 for anomaly detection
            score = f1_score(y_binary, predictions, pos_label=-1, average='binary')
            
            return score
        
        print("\n" + "="*70)
        print("OPTIMIZING ISOLATION FOREST HYPERPARAMETERS")
        print("="*70)
        
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"\n✅ Best F1-Score: {study.best_value:.4f}")
        print(f"\n📊 Best Parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        self.best_params['isolation_forest'] = study.best_params
        return study.best_params
    
    def optimize_ensemble_weights(self, n_trials: int = 20) -> Dict[str, float]:
        """Optimize ensemble weighting"""
        
        def objective(trial):
            cb_weight = trial.suggest_float('catboost_weight', 0.5, 0.9)
            if_weight = 1 - cb_weight
            
            # Train with current best hyperparameters
            cb_params = self.best_params.get('catboost', {
                'iterations': 1000,
                'learning_rate': 0.1,
                'depth': 6,
                'verbose': False,
                'random_seed': 42
            })
            
            if_params = self.best_params.get('isolation_forest', {
                'n_estimators': 100,
                'contamination': 0.25,
                'random_state': 42
            })
            
            cb_model = CatBoostClassifier(**cb_params)
            if_model = IsolationForest(**if_params)
            
            # 5-fold cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            
            for train_idx, val_idx in cv.split(self.X_train, self.y_train):
                X_tr, X_val = self.X_train[train_idx], self.X_train[val_idx]
                y_tr, y_val = self.y_train[train_idx], self.y_train[val_idx]
                
                # Train models
                cb_model.fit(X_tr, y_tr)
                if_model.fit(X_tr)
                
                # Get predictions
                cb_pred = cb_model.predict(X_val)
                if_pred = if_model.predict(X_val)
                if_pred = np.where(if_pred == 1, 0, 1)  # Convert to class labels
                
                # Weighted ensemble
                ensemble_pred = np.zeros(len(X_val), dtype=int)
                for i in range(len(X_val)):
                    votes = {}
                    votes[cb_pred[i]] = votes.get(cb_pred[i], 0) + cb_weight
                    votes[if_pred[i]] = votes.get(if_pred[i], 0) + if_weight
                    ensemble_pred[i] = max(votes, key=votes.get)
                
                score = f1_score(y_val, ensemble_pred, average='weighted')
                scores.append(score)
            
            return np.mean(scores)
        
        print("\n" + "="*70)
        print("OPTIMIZING ENSEMBLE WEIGHTS")
        print("="*70)
        
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        cb_weight = study.best_params['catboost_weight']
        if_weight = 1 - cb_weight
        
        print(f"\n✅ Best F1-Score: {study.best_value:.4f}")
        print(f"\n📊 Optimal Weights:")
        print(f"  CatBoost: {cb_weight:.3f}")
        print(f"  Isolation Forest: {if_weight:.3f}")
        
        self.best_params['weights'] = {
            'catboost': cb_weight,
            'isolation_forest': if_weight
        }
        return self.best_params['weights']
    
    def optimize_all(self) -> Dict[str, Any]:
        """Run complete optimization pipeline"""
        print("\n🚀 STARTING COMPLETE HYPERPARAMETER OPTIMIZATION")
        print("="*70)
        
        # Step 1: Optimize CatBoost
        self.optimize_catboost(n_trials=50)
        
        # Step 2: Optimize Isolation Forest
        self.optimize_isolation_forest(n_trials=30)
        
        # Step 3: Optimize ensemble weights
        self.optimize_ensemble_weights(n_trials=20)
        
        print("\n" + "="*70)
        print("✅ OPTIMIZATION COMPLETE")
        print("="*70)
        
        return self.best_params


# === EXAMPLE USAGE ===
if __name__ == '__main__':
    # Generate sample data
    np.random.seed(42)
    n_samples = 2000
    n_features = 20
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, 4, n_samples)  # 4 classes
    
    print("HYPERPARAMETER OPTIMIZATION FOR JAMMING DETECTION")
    print("="*70)
    print(f"Training samples: {n_samples}")
    print(f"Features: {n_features}")
    print(f"Classes: 4 (normal, constant, random, reactive)")
    
    # Run optimization
    optimizer = EnsembleHyperparameterOptimizer(X_train, y_train)
    best_params = optimizer.optimize_all()
    
    # Save results
    import json
    with open('results/optimized_hyperparameters.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"\n💾 Optimized parameters saved to: results/optimized_hyperparameters.json")
