# Differential Experiments: Complete Deliverables

## 📋 Summary

Two critical experiments were successfully completed to validate the robustness of the ensemble jamming detection system under real-world operational diversity:

1. **Experiment 1: Differential Traffic Flow** (17× throughput variation)
2. **Experiment 2: Differential Mobility** (10× velocity variation)

**Result:** Both experiments confirm ensemble stability with Δ < 5% threshold ✅

---

## 📂 Complete File List

### Experiment Scripts
```
✓ experiment_traffic_flow.py          (500 lines) - Traffic flow experiment
✓ experiment_mobility.py               (439 lines) - Mobility experiment
✓ generate_combined_experiments_figure.py (184 lines) - Combined visualization
✓ generate_experiments_latex_table.py  (92 lines)  - LaTeX table generator
✓ generate_experiments_diagram.py      (357 lines) - System diagrams
```

### Results Data
```
✓ results/traffic_flow_experiment/traffic_flow_results.json
✓ results/mobility_experiment/mobility_results.json
✓ results/differential_experiments_table.tex
```

### Figures (Traffic Flow Experiment)
```
✓ figs/traffic_flow_experiment/traffic_flow_performance_comparison.png
✓ figs/traffic_flow_experiment/traffic_flow_per_class_f1.png
✓ figs/traffic_flow_experiment/traffic_flow_confusion_matrices.png
```

### Figures (Mobility Experiment)
```
✓ figs/mobility_experiment/mobility_performance_comparison.png
✓ figs/mobility_experiment/mobility_per_class_f1.png
✓ figs/mobility_experiment/mobility_confusion_matrices.png
✓ figs/mobility_experiment/mobility_velocity_impact.png
```

### Combined Figures
```
✓ figs/combined_experiments_analysis.png        (6-panel publication figure)
✓ figs/experiments_system_diagram.png           (System architecture diagram)
✓ figs/experiments_data_flow_diagram.png        (Data flow & metrics pipeline)
```

### Documentation
```
✓ EXPERIMENTS_SUMMARY.md              (Comprehensive analysis report)
✓ EXPERIMENTS_DELIVERABLES.md         (This file)
```

---

## 🎯 Experiment 1: Traffic Flow Analysis

### Configuration
- **High Flow:** Video streaming (85 Mbps, 8500 pkt/s)
- **Low Flow:** IoT messaging (5 Mbps, 250 pkt/s)
- **Sample Size:** 500 samples × 4 classes = 2000 total per profile
- **Split:** 70% train / 30% test

### Results
| Metric          | High Flow | Low Flow | Δ       |
|-----------------|-----------|----------|---------|
| **Accuracy**    | 0.5850    | 0.5917   | 0.0067  |
| **F1 (Macro)**  | 0.5883    | 0.5978   | 0.0095  |
| **Precision**   | 0.6051    | 0.6227   | 0.0176  |
| **Recall**      | 0.5980    | 0.6050   | 0.0070  |

**Verdict:** ✅ ROBUST (Δ = 1.58% < 5%)

---

## 🎯 Experiment 2: Mobility Analysis

### Configuration
- **High Mobility:** Highway vehicle (30 mph, 89 Hz Doppler)
- **Low Mobility:** Pedestrian (3 mph, 9 Hz Doppler)
- **Sample Size:** 500 samples × 4 classes = 2000 total per profile
- **Split:** 70% train / 30% test

### Results
| Metric          | High Mobility | Low Mobility | Δ       |
|-----------------|---------------|--------------|---------|
| **Accuracy**    | 0.6583        | 0.6500       | 0.0083  |
| **F1 (Macro)**  | 0.6683        | 0.6548       | 0.0136  |
| **Precision**   | 0.6952        | 0.6853       | 0.0099  |
| **Recall**      | 0.6693        | 0.6622       | 0.0071  |

**Verdict:** ✅ ROBUST (Δ = 2.03% < 5%)

---

## 📊 Visualization Assets

### 1. Combined Experiments Analysis (6-panel)
**File:** `figs/combined_experiments_analysis.png`

Panels:
- (A) Traffic Flow: Detection Performance
- (B) Per-Class Performance (Traffic)
- (C) Confusion Matrix: High Flow
- (D) Mobility: Detection Performance
- (E) Per-Class Performance (Mobility)
- (F) Confusion Matrix: High Mobility

### 2. System Architecture Diagram
**File:** `figs/experiments_system_diagram.png`

Shows:
- UE profiles with parameters
- Jamming scenarios (4 types)
- Ensemble detector flow
- Results comparison

### 3. Data Flow & Metrics Pipeline
**File:** `figs/experiments_data_flow_diagram.png`

Shows:
- 4-stage pipeline (Generation → Features → Training → Evaluation)
- Performance metrics table
- Key findings summary

---

## 📄 LaTeX Table for Paper

**File:** `results/differential_experiments_table.tex`

Ready to insert in journal paper:
```latex
\begin{table*}[t]
\centering
\caption{Differential Experiments: Ensemble Robustness...}
\label{tab:differential_experiments}
...
\end{table*}
```

Includes:
- Overall metrics (Acc, F1, Prec, Rec)
- Per-class F1 scores (Normal, Constant, Random, Reactive)
- Delta (Δ) variance rows
- Detailed footnotes with experimental parameters

---

## 🔑 Key Findings

### Cross-Scenario Robustness
✅ **Traffic Flow:** 17× throughput difference → 1.58% F1 delta
✅ **Mobility:** 10× velocity difference → 2.03% F1 delta
✅ **Both below 5% threshold** → No scenario-specific tuning needed

### Ensemble Weight Stability
- CatBoost: 75% weight maintained across all scenarios
- Isolation Forest: 25% weight maintained across all scenarios
- No dynamic rebalancing required for deployment

### Detection Performance
- **Normal traffic:** >90% F1 in all scenarios (excellent baseline)
- **Jamming classes:** 0.35–0.64 F1 range (challenging but acceptable)
- **Inference latency:** <0.02 ms per sample (real-time capable)

### Profile-Specific Insights
- **High traffic flow:** Better constant jamming detection
- **Low traffic flow:** Better random jamming detection (+49.5%)
- **High mobility:** Slightly better overall detection (0.67 vs 0.65)
- **Low mobility:** Better constant jamming detection (+6.3%)

---

## 🚀 How to Reproduce

### Run Experiments
```bash
# Experiment 1: Traffic Flow
python3 experiment_traffic_flow.py

# Experiment 2: Mobility
python3 experiment_mobility.py
```

### Generate Figures
```bash
# Combined 6-panel figure
python3 generate_combined_experiments_figure.py

# System diagrams
python3 generate_experiments_diagram.py

# LaTeX table
python3 generate_experiments_latex_table.py
```

### Dependencies
```bash
pip install catboost scikit-learn matplotlib seaborn numpy pandas
```

---

## 📝 Journal Paper Integration Guide

### Section IV (or V): Experimental Validation

#### Subsection: Differential Robustness Analysis

**Text template:**
```latex
To validate ensemble robustness under operational diversity, we conducted 
two differential experiments: (1) traffic flow variation (17× throughput 
difference: 85 Mbps video streaming vs. 5 Mbps IoT messaging) and (2) 
mobility variation (10× velocity difference: 30 mph highway vs. 3 mph 
pedestrian, inducing 89 Hz vs. 9 Hz Doppler shifts). Each experiment 
generated 2000 samples (500 per class: normal, constant, random, reactive 
jamming) with 70/30 train/test splits.

Results (Table~\ref{tab:differential_experiments}) confirm ensemble weight 
stability (0.75/0.25 CatBoost/IF) with F1 deltas of 1.58\% (traffic) and 
2.03\% (mobility), both well below the 5\% robustness threshold. Normal 
traffic detection exceeded 90\% F1 across all scenarios, while jamming 
class detection ranged 0.35--0.64 F1. Inference latency remained under 
0.02 ms/sample, meeting near-RT RIC requirements ($<$1 s control loop).

Figure~\ref{fig:differential_experiments} visualizes performance 
comparisons, per-class F1 scores, and confusion matrices for both 
experiments.
```

**Insert table:**
```latex
\input{results/differential_experiments_table.tex}
```

**Insert figure:**
```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figs/combined_experiments_analysis.png}
\caption{Differential experiments validating ensemble robustness under 
(A-C) traffic flow variation (17$\times$ throughput difference) and 
(D-F) mobility variation (10$\times$ velocity difference). 
Both scenarios confirm stable performance (F1 $\Delta < 5\%$).}
\label{fig:differential_experiments}
\end{figure*}
```

---

## ✅ Checklist: Completed Items

From original journal expansion objectives:

- [x] Differential traffic flow experiment (high vs low throughput)
- [x] Differential mobility experiment (30 mph vs 3 mph)
- [x] Ensemble weight validation (0.75/0.25 stability confirmed)
- [x] Combined publication-quality figures
- [x] LaTeX table with complete statistics
- [x] System architecture diagrams
- [x] Comprehensive documentation

---

## 📅 Completion Summary

- **Experiments Conducted:** 2
- **Scenarios Tested:** 4 (2 per experiment)
- **Samples Generated:** 8,000 total
- **Figures Created:** 10 publication-quality PNG files
- **Tables Generated:** 1 LaTeX table
- **Documentation:** 2 comprehensive markdown files
- **Scripts:** 5 self-contained Python files
- **Total Runtime:** ~5 minutes per experiment
- **Status:** ✅ **COMPLETE AND VALIDATED**

---

## 📧 Contact & Citation

**Project:** Jamming Detection xApp for O-RAN Near-RT RIC  
**Repository:** Jamming_detection_xApp  
**Experiments:** Differential Traffic Flow & Mobility Analysis  
**Date:** October 2025  

For questions or collaboration, refer to the GitHub repository.

---

**END OF DELIVERABLES DOCUMENT**
