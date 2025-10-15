#!/usr/bin/env python3
"""
Generate LaTeX parameter table for experiment section.
Lists all key experimental parameters: frequency, bandwidth, jamming power, etc.
"""

latex_table = r"""\begin{table*}[t]
\centering
\caption{Experimental Parameters and System Configuration}
\label{tab:experimental_parameters}
\renewcommand{\arraystretch}{1.3}
\begin{tabular}{l|l|p{8cm}}
\toprule
\textbf{Category} & \textbf{Parameter} & \textbf{Value/Range} \\
\midrule
\multicolumn{3}{l}{\textit{RF \& Channel Parameters}} \\
\midrule
Carrier Frequency & $f_c$ & 2.4 GHz (ISM band) / 5 GHz (UNII bands) \\
Bandwidth & $B$ & 20 MHz \\
Sample Rate & $f_s$ & 20 MS/s \\
Channel Model & --- & Rayleigh fading with AWGN \\
Noise Floor & $N_0$ & -110 dBm (ideal), -105 dBm (moderate), -100 dBm (realistic) \\
SNR Range & --- & -10 to 30 dB \\
\midrule
\multicolumn{3}{l}{\textit{Jamming Attack Parameters}} \\
\midrule
Jamming Types & --- & Normal (no jamming), Constant, Random, Reactive \\
Jamming Power & $P_J$ & -40 to +5 dBm (HackRF configurable) \\
\multirow{2}{*}{Jamming Intensity} & Training & 60--90\% impact on throughput/SINR \\
 & Testing & Variable (0--100\% for robustness analysis) \\
Waveforms & --- & CW tone (1 kHz), FM-modulated, Gaussian noise \\
Frequency Agility & --- & Fixed, Sequential sweep, Random hopping \\
Sensing Threshold & $\tau$ & 0.0002 (energy detection, reactive mode) \\
\midrule
\multicolumn{3}{l}{\textit{UE Traffic \& Mobility Parameters}} \\
\midrule
\multirow{2}{*}{Throughput} & High flow & 85 $\pm$ 15 Mbps (video streaming) \\
 & Low flow & 5 $\pm$ 2 Mbps (IoT messaging) \\
\multirow{2}{*}{Packet Rate} & High flow & 8500 $\pm$ 1200 pkt/s \\
 & Low flow & 250 $\pm$ 80 pkt/s \\
\multirow{2}{*}{Velocity} & High mobility & 30 mph (48 km/h, highway vehicle) \\
 & Low mobility & 3 mph (5 km/h, pedestrian) \\
\multirow{2}{*}{Doppler Shift} & High mobility & 89 Hz @ 2 GHz carrier \\
 & Low mobility & 9 Hz @ 2 GHz carrier \\
Handover Rate & --- & 15\% (high mobility), 2\% (low mobility) \\
\midrule
\multicolumn{3}{l}{\textit{Signal Quality Metrics}} \\
\midrule
RSRP (Reference Signal Received Power) & --- & -110 to -60 dBm \\
RSRQ (Reference Signal Received Quality) & --- & -20 to -3 dB \\
SINR (Signal-to-Interference-plus-Noise Ratio) & --- & -5 to 30 dB \\
BLER (Block Error Rate) & --- & 0 to 1 (exponential decay with SINR) \\
Latency & $L$ & 10 to 500 ms \\
Buffer Occupancy & --- & 0 to 1 (normalized) \\
\midrule
\multicolumn{3}{l}{\textit{Feature Engineering}} \\
\midrule
Temporal Features & --- & Throughput variance, packet rate variance \\
Spectral Features & --- & Entropy, flatness, peak-to-average ratio \\
Statistical Features & --- & Mean, std, skewness, kurtosis (sliding window) \\
Feature Window & --- & 50--100 samples (adaptive) \\
\midrule
\multicolumn{3}{l}{\textit{Ensemble Model Hyperparameters}} \\
\midrule
\multirow{3}{*}{CatBoost} & Iterations & 200 \\
 & Depth & 6 \\
 & Learning rate & 0.1 \\
\multirow{2}{*}{Isolation Forest} & Contamination & 0.25 (expect 75\% normal traffic) \\
 & Estimators & 100 \\
Ensemble Weights & $w_{CB}$ / $w_{IF}$ & 0.75 / 0.25 (optimized, constrained $w_{IF} \geq 0.10$) \\
\midrule
\multicolumn{3}{l}{\textit{Dataset \& Training}} \\
\midrule
Samples per Class & --- & 500 (training + validation) \\
Total Classes & --- & 4 (normal, constant, random, reactive) \\
Train/Test Split & --- & 70\% / 30\% \\
Cross-Validation & --- & 5-fold stratified \\
Random Seed & --- & 42 (reproducibility) \\
\midrule
\multicolumn{3}{l}{\textit{Performance Targets}} \\
\midrule
Detection Accuracy & --- & $> 90\%$ (normal class), $> 60\%$ (jamming classes) \\
F1-Score & --- & $> 0.95$ (overall), $\Delta < 5\%$ (cross-scenario robustness) \\
Latency & --- & $< 1$ ms inference (near-RT RIC requirement) \\
False Positive Rate & --- & $< 5\%$ (normal misclassified as jamming) \\
False Negative Rate & --- & $< 10\%$ (jamming misclassified as normal) \\
\bottomrule
\end{tabular}
\vspace{0.2cm}
\begin{tablenotes}
\footnotesize
\item \textit{Hardware platform: HackRF One SDR for jamming generation; srsRAN/USRP for O-RAN data collection.}
\item \textit{Carrier frequencies follow ITU-R recommendations for ISM (2.4 GHz) and UNII (5 GHz) unlicensed bands.}
\item \textit{Jamming power range (-40 to +5 dBm) mapped to HackRF RF/IF gain ladder; effective EIRP depends on antenna configuration.}
\item \textit{Traffic profiles designed to span typical 5G use cases: eMBB (high flow) and mMTC (low flow).}
\item \textit{Mobility profiles validated against 3GPP channel models (pedestrian: 3 km/h; vehicular: 30--120 km/h).}
\item \textit{Ensemble weights (0.75/0.25) determined via constrained optimization on validation set (Section IV-C).}
\end{tablenotes}
\end{table*}
"""

# Save to file
output_path = 'results/experimental_parameters_table.tex'
with open(output_path, 'w') as f:
    f.write(latex_table)

print(f"✅ Experimental parameters table saved to: {output_path}")
print("\n" + "="*70)
print("TABLE PREVIEW (first 40 lines):")
print("="*70)
print('\n'.join(latex_table.split('\n')[:40]))
print("...")
print(f"\nTotal lines: {len(latex_table.split(chr(10)))}")
