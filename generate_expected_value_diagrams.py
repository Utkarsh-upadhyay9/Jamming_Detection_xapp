#!/usr/bin/env python3
"""
Generate diagrams of EXPECTED values under paper's environments (Ideal, Moderate, Realistic)
without relying on any trained model. Uses simple analytic expectations from our
signal/jamming generative assumptions to visualize what we should expect.

Outputs (saved under figs/):
- expected_metrics_envs.png   (RSRP, SINR, Throughput, Latency per class across environments)
- expected_traffic_metrics.png (Packet Rate, BLER, Buffer)
- expected_radar_realistic.png (Radar per class in realistic environment)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
np.random.seed(42)

# Environments from the paper
ENVIRONMENTS = {
    'Ideal':    dict(noise_var=1.0,  sigma=1.00, beta=1.30, delta=0.00, I0=-115),
    'Moderate': dict(noise_var=3.5,  sigma=0.85, beta=1.10, delta=0.12, I0=-108),
    'Realistic':dict(noise_var=4.5,  sigma=0.75, beta=0.65, delta=0.40, I0=-95)
}

CLASSES = ['Normal', 'Constant', 'Random', 'Reactive']
# Expected jamming factor E[jf] per class (consistent with our generators)
E_JF = {
    'Normal':  (0.00 + 0.15)/2.0,   # ~0.075
    'Constant':(0.50 + 0.80)/2.0,   # ~0.65
    'Random':  (0.30 + 0.90)/2.0,   # ~0.60
    'Reactive':(0.40 + 0.95)/2.0    # ~0.675
}

# Helper: expected (1 - Rayleigh(sigma)) term used in RSRP mean adjustment
# E[Rayleigh(sigma)] = sigma * sqrt(pi/2)
SQRT_PI_OVER_2 = np.sqrt(np.pi/2)

def expected_metrics(env_name: str):
    p = ENVIRONMENTS[env_name]
    out = {cls:{} for cls in CLASSES}
    for cls in CLASSES:
        jf = E_JF[cls]
        # Means (matching the synthetic formulas used in our experiments)
        rsrp_mean = -70 - jf*25 + (1 - p['sigma']*SQRT_PI_OVER_2)*10  # noise is zero-mean
        sinr_mean = 15*p['beta'] - jf*30                              # noise zero-mean
        thr_mean = 70 * (1 - jf*0.7) * p['beta']
        pkr_mean = 7000 * (1 - jf*0.8) * p['beta']
        bler_mean = np.clip(jf*0.7, 0, 1)
        buf_mean = np.clip(jf*0.8, 0, 1)
        lat_mean = 30 + jf*250
        
        # Rough std models driven by noise variance (for error bars)
        nv = p['noise_var']
        rsrp_std = np.sqrt(nv)*2.0
        sinr_std = np.sqrt(nv)*3.0
        thr_std  = 10*(1 + p['delta'])
        pkr_std  = 1200*(1 + p['delta'])
        bler_std = 0.07*(1 + p['delta'])
        buf_std  = 0.08*(1 + p['delta'])
        lat_std  = 30*(1 + p['delta'])
        
        out[cls] = dict(
            rsrp=(rsrp_mean, rsrp_std),
            sinr=(sinr_mean, sinr_std),
            thr=(thr_mean, thr_std),
            pkr=(pkr_mean, pkr_std),
            bler=(bler_mean, bler_std),
            buf=(buf_mean, buf_std),
            lat=(lat_mean, lat_std)
        )
    return out

# 1) Bar plots for RSRP, SINR, THR, LAT across environments
os.makedirs('figs', exist_ok=True)
metrics = ['rsrp', 'sinr', 'thr', 'lat']
metric_titles = {
    'rsrp':'Expected RSRP (dBm)',
    'sinr':'Expected SINR (dB)',
    'thr':'Expected Throughput (Mbps)',
    'lat':'Expected Latency (ms)'
}

fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharex=False)
colors = {'Ideal':'#4c72b0', 'Moderate':'#dd8452', 'Realistic':'#55a868'}
x = np.arange(len(CLASSES))
width = 0.25

for mi, m in enumerate(metrics):
    ax = axes[mi]
    for ei, env in enumerate(['Ideal','Moderate','Realistic']):
        E = expected_metrics(env)
        means = [E[c][m][0] for c in CLASSES]
        stds  = [E[c][m][1] for c in CLASSES]
        ax.bar(x + (ei-1)*width, means, width, yerr=stds, capsize=3,
               color=colors[env], alpha=0.85, label=env)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=15)
    ax.set_title(metric_titles[m])
    ax.grid(alpha=0.25)
    if m in ('thr','lat'):
        ax.set_ylim(bottom=0)
    if mi==0:
        ax.legend(fontsize=9)

fig.tight_layout()
plt.savefig('figs/expected_metrics_envs.png', dpi=300)
plt.close()

# 2) Traffic metrics: Packet Rate, BLER, Buffer in one figure (Realistic only)
E_real = expected_metrics('Realistic')
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, key, title in zip(
    axes,
    ['pkr','bler','buf'],
    ['Expected Packet Rate (pps)','Expected BLER','Expected Buffer Occupancy']
):
    means = [E_real[c][key][0] for c in CLASSES]
    stds  = [E_real[c][key][1] for c in CLASSES]
    ax.bar(CLASSES, means, yerr=stds, capsize=3, color='#4c72b0', alpha=0.85)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    if key in ('bler','buf'):
        ax.set_ylim(0, 1)
fig.tight_layout()
plt.savefig('figs/expected_traffic_metrics.png', dpi=300)
plt.close()

# 3) Radar plot (Realistic): normalize metrics to [0,1] for visualization
radar_keys = ['thr','pkr','sinr','rsrp','bler','lat']
labels = ['Throughput','PacketRate','SINR','RSRP','BLER(1-x)','Latency(1-x)']
angles = np.linspace(0, 2*np.pi, len(radar_keys), endpoint=False)
angles = np.concatenate([angles, angles[:1]])

plt.figure(figsize=(6,6))
ax = plt.subplot(111, polar=True)
ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

# Normalize helper (per-metric across classes)
vals = {k: np.array([E_real[c][k][0] for c in CLASSES]) for k in radar_keys}
# invert for bler/lat (lower is better) when plotting
vals['bler'] = 1 - (vals['bler'] - np.min(vals['bler'])) / (np.ptp(vals['bler']) + 1e-12)
vals['lat']  = 1 - (vals['lat']  - np.min(vals['lat']))  / (np.ptp(vals['lat'])  + 1e-12)
for k in ['thr','pkr','sinr','rsrp']:
    vals[k] = (vals[k] - np.min(vals[k])) / (np.ptp(vals[k]) + 1e-12)

palette = ['#4c72b0','#dd8452','#55a868','#c44e52']
for i, cls in enumerate(CLASSES):
    data = np.array([vals[k][i] for k in radar_keys])
    data = np.concatenate([data, data[:1]])
    ax.plot(angles, data, color=palette[i], linewidth=2, label=cls)
    ax.fill(angles, data, color=palette[i], alpha=0.15)

ax.set_rlabel_position(0)
ax.set_yticks([0.2,0.4,0.6,0.8])
ax.set_yticklabels(['0.2','0.4','0.6','0.8'])
ax.set_ylim(0,1)
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
plt.tight_layout()
plt.savefig('figs/expected_radar_realistic.png', dpi=300, bbox_inches='tight')
plt.close()

print('✅ Expected value diagrams saved:')
print('  - figs/expected_metrics_envs.png')
print('  - figs/expected_traffic_metrics.png')
print('  - figs/expected_radar_realistic.png')
