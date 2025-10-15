# Model Improvement Roadmap for Jamming Detection

## 🎯 Current Performance Gap

**Current Results:**
- Traffic Flow Experiment: F1 = 0.59 (59%)
- Mobility Experiment: F1 = 0.67 (67%)
- **Target for Deployment:** F1 > 0.95 (95%)
- **Performance Gap:** ~30-36 percentage points

## 🚀 8 Concrete Improvements (Ranked by Impact)

---

### **1. Advanced Feature Engineering** ⭐⭐⭐⭐⭐ (HIGHEST IMPACT)

**Expected Improvement:** +20-25% F1-score

**Current Problem:** Using only 9 basic features (RSRP, SINR, throughput, packet rate, buffer occupancy, spectral entropy, flatness, BLER, latency)

**Solution Implemented:** `improved_feature_engineering.py`
- **Temporal Features (30+ features):**
  - Higher-order moments (skewness, kurtosis)
  - Percentiles (P25, P50, P75)
  - Range, IQR, coefficient of variation
  - 1st/2nd order differences (velocity, acceleration)

- **Spectral Features (15+ features):**
  - Spectral power, centroid, bandwidth
  - Spectral rolloff (85% energy threshold)
  - FFT-based frequency analysis
  
- **Cross-Signal Features (10+ features):**
  - RSRP-SINR correlation
  - Throughput-packet rate correlation
  - Bytes per packet ratio
  - Signal quality index

- **Anomaly Indicators (8+ features):**
  - Z-score outlier ratio
  - Sudden jump detection
  - Signal instability metrics

- **Entropy Features (8+ features):**
  - Approximate entropy (regularity)
  - Sample entropy (complexity)

**Total: ~70+ features vs. current 9 features**

**Why This Works:**
- Jamming creates unique patterns in temporal dynamics (reactive jamming → high acceleration)
- Spectral anomalies distinguish sweep vs. constant jamming
- Entropy features capture predictability changes
- Cross-signal correlations detect coordinated degradation

**Implementation:**
```python
from improved_feature_engineering import extract_advanced_features

# Instead of basic features
features_df = extract_advanced_features(raw_signals)
```

---

### **2. Hyperparameter Optimization** ⭐⭐⭐⭐ (HIGH IMPACT)

**Expected Improvement:** +5-10% F1-score

**Current Problem:** Using default parameters (iterations=200, depth=6, learning_rate=0.1)

**Solution Implemented:** `hyperparameter_optimization.py`
- **Bayesian optimization** using Optuna (smarter than grid search)
- **CatBoost parameters optimized:**
  - Iterations: 500-2000 (vs. current 200)
  - Learning rate: 0.01-0.3 (adaptive)
  - Depth: 4-10 (find sweet spot)
  - L2 regularization: 1-10
  - Border count: 32-255
  - Bagging temperature, random strength

- **Isolation Forest parameters optimized:**
  - n_estimators: 50-300
  - max_samples: 0.5-1.0
  - contamination: 0.1-0.3 (critical!)
  - max_features: 0.5-1.0

- **Ensemble weights optimized:**
  - CatBoost: 0.5-0.9 (find optimal balance)
  - Current 0.75/0.25 may not be optimal

**Implementation:**
```python
from hyperparameter_optimization import EnsembleHyperparameterOptimizer

optimizer = EnsembleHyperparameterOptimizer(X_train, y_train)
best_params = optimizer.optimize_all()  # Takes 30-60 min
```

---

### **3. Data Augmentation** ⭐⭐⭐⭐ (HIGH IMPACT)

**Expected Improvement:** +8-12% F1-score

**Current Problem:** Only 500 samples per class, limited diversity

**Solutions:**
1. **SMOTE (Synthetic Minority Oversampling)**
   ```python
   from imblearn.over_sampling import SMOTE
   smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
   X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
   ```

2. **Time-Series Augmentation:**
   - Jittering: Add small random noise to each feature
   - Scaling: Random amplitude scaling (×0.8 to ×1.2)
   - Time warping: Speed up/slow down temporal sequences
   - Window slicing: Create overlapping windows

3. **Jamming-Specific Augmentation:**
   - Vary jamming power (±10 dB)
   - Mix jamming types (e.g., 70% constant + 30% random)
   - Add realistic noise (thermal, quantization)

**Expected Result:** 500 → 2000+ samples per class

---

### **4. Ensemble Diversity Enhancement** ⭐⭐⭐ (MEDIUM-HIGH IMPACT)

**Expected Improvement:** +5-8% F1-score

**Current Problem:** Only 2 models (CatBoost + Isolation Forest)

**Solution:** Add complementary models
```python
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier

ensemble_models = {
    'catboost': CatBoostClassifier(...),  # Primary (40% weight)
    'lightgbm': LGBMClassifier(...),      # Fast gradient boosting (25%)
    'extratrees': ExtraTreesClassifier(...), # High diversity (15%)
    'svm_rbf': SVC(kernel='rbf', ...),    # Non-linear boundaries (10%)
    'isolation_forest': IsolationForest(...) # Anomaly detection (10%)
}
```

**Voting Strategy:**
- **Soft voting** with confidence weighting (better than hard voting)
- **Stacking:** Use meta-learner (logistic regression) on base model predictions

---

### **5. Class Imbalance Handling** ⭐⭐⭐ (MEDIUM IMPACT)

**Expected Improvement:** +3-5% F1-score (especially for minority classes)

**Current Problem:** Likely unequal performance across jamming types

**Solutions:**
1. **Class weights:**
   ```python
   from sklearn.utils.class_weight import compute_class_weight
   
   class_weights = compute_class_weight(
       'balanced', classes=np.unique(y_train), y=y_train
   )
   # Pass to CatBoost: class_weights=dict(enumerate(class_weights))
   ```

2. **Focal Loss:** Focus on hard-to-classify samples
   ```python
   # CatBoost custom loss
   loss_function='MultiClass:gamma=2.0'  # Focal loss with γ=2
   ```

3. **Cost-Sensitive Learning:**
   - Higher penalty for misclassifying reactive jamming (hardest)
   - Lower penalty for normal traffic (easiest)

---

### **6. Real-Time Feature Selection** ⭐⭐⭐ (MEDIUM IMPACT)

**Expected Improvement:** +2-4% F1-score + faster inference

**Current Problem:** Using all features (some may be redundant/noisy)

**Solutions:**
1. **Feature Importance Analysis:**
   ```python
   importance = ensemble.catboost.get_feature_importance()
   top_features = np.argsort(importance)[-30:]  # Keep top 30
   ```

2. **Recursive Feature Elimination (RFE):**
   ```python
   from sklearn.feature_selection import RFE
   rfe = RFE(estimator=catboost_model, n_features_to_select=40)
   X_selected = rfe.fit_transform(X_train, y_train)
   ```

3. **Correlation-Based Filtering:**
   - Remove features with correlation > 0.95 (redundant)
   - Keep features with highest correlation to target

---

### **7. Cross-Validation Strategy** ⭐⭐ (LOW-MEDIUM IMPACT)

**Expected Improvement:** +2-3% F1-score (robustness)

**Current Problem:** Single 70/30 train/test split

**Solution:** Stratified K-Fold CV
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    ensemble, X, y, cv=cv, scoring='f1_weighted', n_jobs=-1
)
print(f"Mean F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

**Benefits:**
- More reliable performance estimate
- Reduces variance from lucky/unlucky splits
- Ensures all data used for both training and validation

---

### **8. Post-Processing & Calibration** ⭐⭐ (LOW IMPACT)

**Expected Improvement:** +1-2% F1-score

**Solutions:**
1. **Temporal Smoothing:** Avoid rapid class changes
   ```python
   def smooth_predictions(predictions, window_size=5):
       from scipy.ndimage import median_filter
       return median_filter(predictions, size=window_size)
   ```

2. **Confidence Thresholding:**
   ```python
   # Only accept prediction if confidence > threshold
   if max(proba) < 0.7:
       prediction = "uncertain" or previous_prediction
   ```

3. **Probability Calibration:**
   ```python
   from sklearn.calibration import CalibratedClassifierCV
   calibrated = CalibratedClassifierCV(ensemble, method='isotonic', cv=5)
   calibrated.fit(X_train, y_train)
   ```

---

## 📊 Expected Cumulative Impact

| Improvement | Individual Impact | Cumulative F1 |
|-------------|-------------------|---------------|
| **Baseline** | — | **0.60** |
| + Advanced Features | +22% | **0.82** |
| + Hyperparameter Opt | +8% | **0.90** |
| + Data Augmentation | +4% | **0.94** |
| + Ensemble Diversity | +3% | **0.97** |
| + Class Balancing | +1% | **0.98** |
| + Feature Selection | +0.5% | **0.985** |
| **FINAL TARGET** | — | **>0.95 ✅** |

---

## 🛠️ Implementation Priority (3-Phase Plan)

### **Phase 1: Quick Wins (1-2 days) → Expected F1: 0.85-0.90**
1. ✅ Implement advanced feature engineering (`improved_feature_engineering.py`)
2. ✅ Run hyperparameter optimization (`hyperparameter_optimization.py`)
3. Apply SMOTE data augmentation

### **Phase 2: Ensemble Enhancement (2-3 days) → Expected F1: 0.92-0.95**
4. Add LightGBM + ExtraTrees to ensemble
5. Implement class weighting and focal loss
6. Feature selection (keep top 40 features)

### **Phase 3: Fine-Tuning (1-2 days) → Expected F1: 0.96-0.98**
7. Cross-validation for robust evaluation
8. Temporal smoothing and calibration
9. Final ensemble weight optimization

---

## 💻 Quick Start Commands

```bash
# Phase 1: Feature Engineering + Optimization
python improved_feature_engineering.py  # Test new features
python hyperparameter_optimization.py   # Find optimal params (60 min)

# Phase 2: Retrain with improvements
python experiment_traffic_flow_improved.py
python experiment_mobility_improved.py

# Phase 3: Validate
python comprehensive_validation.py
```

---

## 📈 Success Metrics

- **Traffic Flow:** F1 > 0.95 (currently 0.59)
- **Mobility:** F1 > 0.95 (currently 0.67)
- **Inference Latency:** < 1 ms (maintain real-time capability)
- **Cross-scenario Δ:** < 3% (robustness)

---

## 🔬 Why You're Currently at 60% F1

**Root Causes:**
1. **Insufficient features:** 9 basic features cannot capture complex jamming patterns
2. **Suboptimal hyperparameters:** Default CatBoost settings not tuned for this problem
3. **Limited data:** 500 samples/class insufficient for deep learning generalization
4. **No feature interactions:** Missing cross-signal correlations that indicate jamming
5. **No temporal modeling:** Ignoring time-series nature (reactive jamming evolves)

**Why Advanced Features Fix This:**
- Reactive jamming → High 2nd-order differences (acceleration)
- Sweep jamming → High spectral bandwidth, low spectral flatness correlation
- Constant jamming → Low entropy, high signal stability
- Normal traffic → Moderate entropy, strong RSRP-SINR-throughput correlation

---

## 📚 References & Inspiration

1. **Feature Engineering:** "Feature Engineering for Machine Learning" (Zheng & Casari, 2018)
2. **Hyperparameter Tuning:** Optuna paper (Akiba et al., 2019)
3. **Ensemble Methods:** "Ensemble Methods: Foundations and Algorithms" (Zhou, 2012)
4. **Jamming Detection:** Recent works achieving >95% F1 use 50-100 features

---

## ✅ Next Steps

1. **Run Phase 1 implementations** (already created)
2. **Analyze feature importance** from new features
3. **Retrain experiments** with optimized parameters
4. **Validate on test set** to confirm >95% F1
5. **Update paper** with new results and methodology

**Estimated Time to 95% F1:** 5-7 days with dedicated implementation
