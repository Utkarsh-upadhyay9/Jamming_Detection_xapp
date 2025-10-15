#!/usr/bin/env python3
"""
Regenerate mobility figures with MATLAB style (simplified labels).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load results
results = json.load(open('results/mobility_experiment/mobility_results.json'))

# Set MATLAB style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3

fig_dir = Path('figs/mobility_experiment')

# MATLAB default colors
matlab_colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE']

# =============================================================================
# FIGURE 1: Combined Performance
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

profiles = ['high_mobility', 'low_mobility']
profile_names = ['High Mobility', 'Low Mobility']  # Removed speed labels

# Left: Overall metrics
ax1 = axes[0]
metrics = ['accuracy', 'f1_macro', 'precision', 'recall']
metric_labels = ['Accuracy', 'F1', 'Precision', 'Recall']

x = np.arange(len(profile_names))
width = 0.18

for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
    values = [results[p][metric] for p in profiles]
    offset = (i - 1.5) * width
    bars = ax1.bar(x + offset, values, width, label=label, 
                  color=matlab_colors[i], edgecolor='black', 
                  linewidth=0.8, alpha=0.9)
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.2f}', ha='center', va='bottom', fontsize=8)

ax1.set_ylabel('Score', fontsize=11, fontweight='bold')
ax1.set_title('(A) Overall Performance Metrics', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(profile_names, fontsize=10, fontweight='bold')
ax1.legend(fontsize=9, loc='lower right', frameon=True, fancybox=False, edgecolor='black')
ax1.set_ylim(0, 1.1)
ax1.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.8)

# Right: Per-class F1
ax2 = axes[1]
jamming_types = ['normal', 'constant', 'random', 'reactive']
x_jam = np.arange(len(jamming_types))
width_jam = 0.35

high_mob_f1 = [results['high_mobility']['f1_per_class'][jt] for jt in jamming_types]
low_mob_f1 = [results['low_mobility']['f1_per_class'][jt] for jt in jamming_types]

bars1 = ax2.bar(x_jam - width_jam/2, high_mob_f1, width_jam, label='High Mobility', 
               color=matlab_colors[0], edgecolor='black', linewidth=0.8, alpha=0.9)
bars2 = ax2.bar(x_jam + width_jam/2, low_mob_f1, width_jam, label='Low Mobility', 
               color=matlab_colors[1], edgecolor='black', linewidth=0.8, alpha=0.9)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.2f}', ha='center', va='bottom', fontsize=8)

ax2.set_xlabel('Jamming Type', fontsize=11, fontweight='bold')
ax2.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
ax2.set_title('(B) Per-Class Detection Performance', fontsize=12, fontweight='bold')
ax2.set_xticks(x_jam)
ax2.set_xticklabels([jt.capitalize() for jt in jamming_types], fontsize=10)
ax2.legend(fontsize=9, loc='upper right', frameon=True, fancybox=False, edgecolor='black')
ax2.set_ylim(0, 1.1)
ax2.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.8)

plt.suptitle('Mobility Experiment: Performance Analysis', 
            fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(fig_dir / 'mobility_performance.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# FIGURE 2: Confusion Matrices
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

class_labels = ['NRM', 'CON', 'RND', 'REA']

for idx, (profile_key, profile_name) in enumerate(zip(profiles, profile_names)):
    cm = np.array(results[profile_key]['confusion_matrix'])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=axes[idx],
               xticklabels=class_labels, yticklabels=class_labels,
               cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray',
               annot_kws={'fontsize': 10, 'fontweight': 'bold'})
    
    axes[idx].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('True Label', fontsize=11, fontweight='bold')
    axes[idx].set_title(f'({chr(65+idx)}) {profile_name}', fontsize=12, fontweight='bold')

plt.suptitle('Mobility Experiment: Confusion Matrices', 
            fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(fig_dir / 'mobility_confusion.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Mobility figures regenerated (MATLAB style)")
