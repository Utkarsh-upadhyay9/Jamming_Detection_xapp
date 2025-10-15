# Experiment Figures Summary

## 📊 Final Publication-Ready Figures (MATLAB Style)

Each experiment has **exactly 2 figures**:

### Traffic Flow Experiment
1. **`traffic_flow_performance.png`** - Overall + Per-class metrics
2. **`traffic_flow_confusion.png`** - Confusion matrices

### Mobility Experiment  
1. **`mobility_performance.png`** - Overall + Per-class metrics
2. **`mobility_confusion.png`** - Confusion matrices

---

## 🎨 MATLAB Style Applied

- **Colors:** #0072BD, #D95319, #EDB120, #7E2F8E (MATLAB defaults)
- **Font:** Serif, 10pt base
- **Labels:** Simplified ("High Flow" / "Low Flow", no throughput values)
- **Grid:** Solid lines, alpha=0.3
- **Resolution:** 300 DPI

---

## 📝 LaTeX Captions

```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figs/traffic_flow_experiment/traffic_flow_performance.png}
\caption{Traffic flow experiment: (A) Overall metrics; (B) Per-class F1-scores. 
F1 $\Delta=1.58\%$ confirms robustness.}
\label{fig:traffic_flow_performance}
\end{figure*}
```

---

✅ **Status:** Complete and ready for journal submission
