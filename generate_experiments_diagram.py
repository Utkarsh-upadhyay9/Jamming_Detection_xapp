#!/usr/bin/env python3
"""
Generate conceptual diagram illustrating both differential experiments.
Shows experimental setup, parameters, and flow for traffic & mobility analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# Create figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.25)

# =============================================================================
# EXPERIMENT 1: TRAFFIC FLOW DIAGRAM
# =============================================================================

ax1 = fig.add_subplot(gs[0])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 6)
ax1.axis('off')

# Title
ax1.text(5, 5.7, 'Experiment 1: Differential Traffic Flow Analysis', 
         ha='center', fontsize=14, fontweight='bold')

# High Flow Path
high_y = 4.2

# UE icon (High Flow)
ue_high = FancyBboxPatch((0.3, high_y-0.3), 0.6, 0.6, 
                         boxstyle="round,pad=0.05", 
                         facecolor='#2ecc71', edgecolor='black', linewidth=2)
ax1.add_patch(ue_high)
ax1.text(0.6, high_y, 'UE₁', ha='center', va='center', fontsize=10, fontweight='bold')

# High Flow Parameters Box
params_high = FancyBboxPatch((1.3, high_y-0.45), 2.0, 0.9,
                            boxstyle="round,pad=0.1",
                            facecolor='#d5f4e6', edgecolor='#2ecc71', linewidth=2)
ax1.add_patch(params_high)
ax1.text(2.3, high_y+0.25, 'High Traffic Flow', ha='center', fontsize=9, fontweight='bold')
ax1.text(2.3, high_y, '• Throughput: 85 Mbps', ha='center', fontsize=7.5)
ax1.text(2.3, high_y-0.25, '• Packet Rate: 8500 pkt/s', ha='center', fontsize=7.5)

# Arrow to jamming
arrow1 = FancyArrowPatch((3.5, high_y), (4.2, high_y),
                        arrowstyle='->', mutation_scale=20, 
                        color='black', linewidth=2)
ax1.add_patch(arrow1)

# Jamming Box
jamming_box = FancyBboxPatch((4.2, 3.0), 1.6, 2.4,
                            boxstyle="round,pad=0.1",
                            facecolor='#ffe6e6', edgecolor='#e74c3c', linewidth=2.5)
ax1.add_patch(jamming_box)
ax1.text(5.0, 5.0, 'Jamming', ha='center', fontsize=10, fontweight='bold', color='#c0392b')
ax1.text(5.0, 4.6, 'Scenarios', ha='center', fontsize=9, fontweight='bold', color='#c0392b')
ax1.text(5.0, 4.15, '① Normal', ha='center', fontsize=7.5)
ax1.text(5.0, 3.85, '② Constant', ha='center', fontsize=7.5)
ax1.text(5.0, 3.55, '③ Random', ha='center', fontsize=7.5)
ax1.text(5.0, 3.25, '④ Reactive', ha='center', fontsize=7.5)

# Arrow to ensemble
arrow2 = FancyArrowPatch((5.9, high_y), (6.5, high_y),
                        arrowstyle='->', mutation_scale=20,
                        color='black', linewidth=2)
ax1.add_patch(arrow2)

# Low Flow Path
low_y = 1.8

# UE icon (Low Flow)
ue_low = FancyBboxPatch((0.3, low_y-0.3), 0.6, 0.6,
                       boxstyle="round,pad=0.05",
                       facecolor='#3498db', edgecolor='black', linewidth=2)
ax1.add_patch(ue_low)
ax1.text(0.6, low_y, 'UE₂', ha='center', va='center', fontsize=10, fontweight='bold')

# Low Flow Parameters Box
params_low = FancyBboxPatch((1.3, low_y-0.45), 2.0, 0.9,
                           boxstyle="round,pad=0.1",
                           facecolor='#d6eaf8', edgecolor='#3498db', linewidth=2)
ax1.add_patch(params_low)
ax1.text(2.3, low_y+0.25, 'Low Traffic Flow', ha='center', fontsize=9, fontweight='bold')
ax1.text(2.3, low_y, '• Throughput: 5 Mbps', ha='center', fontsize=7.5)
ax1.text(2.3, low_y-0.25, '• Packet Rate: 250 pkt/s', ha='center', fontsize=7.5)

# Arrow to jamming
arrow3 = FancyArrowPatch((3.5, low_y), (4.2, low_y),
                        arrowstyle='->', mutation_scale=20,
                        color='black', linewidth=2)
ax1.add_patch(arrow3)

# Arrow from low to ensemble
arrow4 = FancyArrowPatch((5.9, low_y), (6.5, low_y),
                        arrowstyle='->', mutation_scale=20,
                        color='black', linewidth=2)
ax1.add_patch(arrow4)

# Ensemble Model Box
ensemble_box = FancyBboxPatch((6.5, 2.2), 1.8, 2.4,
                             boxstyle="round,pad=0.1",
                             facecolor='#fff3cd', edgecolor='#f39c12', linewidth=2.5)
ax1.add_patch(ensemble_box)
ax1.text(7.4, 4.3, 'Ensemble', ha='center', fontsize=10, fontweight='bold', color='#d68910')
ax1.text(7.4, 4.0, 'Detector', ha='center', fontsize=9, fontweight='bold', color='#d68910')
ax1.text(7.4, 3.5, 'CatBoost', ha='center', fontsize=8)
ax1.text(7.4, 3.2, '(75%)', ha='center', fontsize=7.5, style='italic')
ax1.text(7.4, 2.8, '+', ha='center', fontsize=8)
ax1.text(7.4, 2.5, 'Isolation Forest', ha='center', fontsize=8)
ax1.text(7.4, 2.3, '(25%)', ha='center', fontsize=7.5, style='italic')

# Arrow to results
arrow5 = FancyArrowPatch((8.4, 3.4), (9.0, 3.4),
                        arrowstyle='->', mutation_scale=20,
                        color='black', linewidth=2)
ax1.add_patch(arrow5)

# Results Box
results_box = FancyBboxPatch((9.0, 2.6), 0.85, 1.6,
                            boxstyle="round,pad=0.08",
                            facecolor='#e8f8f5', edgecolor='#16a085', linewidth=2)
ax1.add_patch(results_box)
ax1.text(9.425, 4.0, 'Results', ha='center', fontsize=9, fontweight='bold', color='#117a65')
ax1.text(9.425, 3.6, 'ΔF1:', ha='center', fontsize=8)
ax1.text(9.425, 3.35, '1.58%', ha='center', fontsize=8, fontweight='bold')
ax1.text(9.425, 3.0, '✓', ha='center', fontsize=12, color='#16a085')
ax1.text(9.425, 2.75, 'Robust', ha='center', fontsize=7, style='italic')

# Add comparison annotation
ax1.annotate('17× Throughput\nDifference', xy=(2.3, 3.0), fontsize=8, ha='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.7, edgecolor='orange'))

# =============================================================================
# EXPERIMENT 2: MOBILITY DIAGRAM
# =============================================================================

ax2 = fig.add_subplot(gs[1])
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 6)
ax2.axis('off')

# Title
ax2.text(5, 5.7, 'Experiment 2: Differential Mobility Analysis', 
         ha='center', fontsize=14, fontweight='bold')

# High Mobility Path
high_mob_y = 4.2

# UE icon (High Mobility) - with motion lines
ue_high_mob = FancyBboxPatch((0.3, high_mob_y-0.3), 0.6, 0.6,
                            boxstyle="round,pad=0.05",
                            facecolor='#e74c3c', edgecolor='black', linewidth=2)
ax2.add_patch(ue_high_mob)
ax2.text(0.6, high_mob_y, 'UE₁', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
# Motion lines
for i in range(3):
    ax2.plot([0.15-i*0.08, 0.25-i*0.08], [high_mob_y+0.1, high_mob_y+0.1], 
            'r-', linewidth=2, alpha=0.3+i*0.2)

# High Mobility Parameters Box
params_high_mob = FancyBboxPatch((1.3, high_mob_y-0.5), 2.0, 1.0,
                                boxstyle="round,pad=0.1",
                                facecolor='#fadbd8', edgecolor='#e74c3c', linewidth=2)
ax2.add_patch(params_high_mob)
ax2.text(2.3, high_mob_y+0.3, 'High Mobility', ha='center', fontsize=9, fontweight='bold')
ax2.text(2.3, high_mob_y+0.05, '• Speed: 30 mph (48 km/h)', ha='center', fontsize=7.5)
ax2.text(2.3, high_mob_y-0.2, '• Doppler: 89 Hz', ha='center', fontsize=7.5)
ax2.text(2.3, high_mob_y-0.4, '• Scenario: Highway', ha='center', fontsize=7, style='italic')

# Arrow to jamming
arrow_mob1 = FancyArrowPatch((3.5, high_mob_y), (4.2, high_mob_y),
                            arrowstyle='->', mutation_scale=20,
                            color='black', linewidth=2)
ax2.add_patch(arrow_mob1)

# Jamming Box (same as above)
jamming_box_mob = FancyBboxPatch((4.2, 3.0), 1.6, 2.4,
                                boxstyle="round,pad=0.1",
                                facecolor='#ffe6e6', edgecolor='#e74c3c', linewidth=2.5)
ax2.add_patch(jamming_box_mob)
ax2.text(5.0, 5.0, 'Jamming', ha='center', fontsize=10, fontweight='bold', color='#c0392b')
ax2.text(5.0, 4.6, 'Scenarios', ha='center', fontsize=9, fontweight='bold', color='#c0392b')
ax2.text(5.0, 4.15, '① Normal', ha='center', fontsize=7.5)
ax2.text(5.0, 3.85, '② Constant', ha='center', fontsize=7.5)
ax2.text(5.0, 3.55, '③ Random', ha='center', fontsize=7.5)
ax2.text(5.0, 3.25, '④ Reactive', ha='center', fontsize=7.5)

# Arrow to ensemble
arrow_mob2 = FancyArrowPatch((5.9, high_mob_y), (6.5, high_mob_y),
                            arrowstyle='->', mutation_scale=20,
                            color='black', linewidth=2)
ax2.add_patch(arrow_mob2)

# Low Mobility Path
low_mob_y = 1.8

# UE icon (Low Mobility) - slower motion
ue_low_mob = FancyBboxPatch((0.3, low_mob_y-0.3), 0.6, 0.6,
                           boxstyle="round,pad=0.05",
                           facecolor='#9b59b6', edgecolor='black', linewidth=2)
ax2.add_patch(ue_low_mob)
ax2.text(0.6, low_mob_y, 'UE₂', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
# Slow motion line
ax2.plot([0.15, 0.25], [low_mob_y+0.1, low_mob_y+0.1], 'purple', linewidth=2, alpha=0.4)

# Low Mobility Parameters Box
params_low_mob = FancyBboxPatch((1.3, low_mob_y-0.5), 2.0, 1.0,
                               boxstyle="round,pad=0.1",
                               facecolor='#ebdef0', edgecolor='#9b59b6', linewidth=2)
ax2.add_patch(params_low_mob)
ax2.text(2.3, low_mob_y+0.3, 'Low Mobility', ha='center', fontsize=9, fontweight='bold')
ax2.text(2.3, low_mob_y+0.05, '• Speed: 3 mph (5 km/h)', ha='center', fontsize=7.5)
ax2.text(2.3, low_mob_y-0.2, '• Doppler: 9 Hz', ha='center', fontsize=7.5)
ax2.text(2.3, low_mob_y-0.4, '• Scenario: Pedestrian', ha='center', fontsize=7, style='italic')

# Arrow to jamming
arrow_mob3 = FancyArrowPatch((3.5, low_mob_y), (4.2, low_mob_y),
                            arrowstyle='->', mutation_scale=20,
                            color='black', linewidth=2)
ax2.add_patch(arrow_mob3)

# Arrow from low to ensemble
arrow_mob4 = FancyArrowPatch((5.9, low_mob_y), (6.5, low_mob_y),
                            arrowstyle='->', mutation_scale=20,
                            color='black', linewidth=2)
ax2.add_patch(arrow_mob4)

# Ensemble Model Box (same)
ensemble_box_mob = FancyBboxPatch((6.5, 2.2), 1.8, 2.4,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#fff3cd', edgecolor='#f39c12', linewidth=2.5)
ax2.add_patch(ensemble_box_mob)
ax2.text(7.4, 4.3, 'Ensemble', ha='center', fontsize=10, fontweight='bold', color='#d68910')
ax2.text(7.4, 4.0, 'Detector', ha='center', fontsize=9, fontweight='bold', color='#d68910')
ax2.text(7.4, 3.5, 'CatBoost', ha='center', fontsize=8)
ax2.text(7.4, 3.2, '(75%)', ha='center', fontsize=7.5, style='italic')
ax2.text(7.4, 2.8, '+', ha='center', fontsize=8)
ax2.text(7.4, 2.5, 'Isolation Forest', ha='center', fontsize=8)
ax2.text(7.4, 2.3, '(25%)', ha='center', fontsize=7.5, style='italic')

# Arrow to results
arrow_mob5 = FancyArrowPatch((8.4, 3.4), (9.0, 3.4),
                            arrowstyle='->', mutation_scale=20,
                            color='black', linewidth=2)
ax2.add_patch(arrow_mob5)

# Results Box
results_box_mob = FancyBboxPatch((9.0, 2.6), 0.85, 1.6,
                                boxstyle="round,pad=0.08",
                                facecolor='#e8f8f5', edgecolor='#16a085', linewidth=2)
ax2.add_patch(results_box_mob)
ax2.text(9.425, 4.0, 'Results', ha='center', fontsize=9, fontweight='bold', color='#117a65')
ax2.text(9.425, 3.6, 'ΔF1:', ha='center', fontsize=8)
ax2.text(9.425, 3.35, '2.03%', ha='center', fontsize=8, fontweight='bold')
ax2.text(9.425, 3.0, '✓', ha='center', fontsize=12, color='#16a085')
ax2.text(9.425, 2.75, 'Robust', ha='center', fontsize=7, style='italic')

# Add comparison annotation
ax2.annotate('10× Velocity\nDifference', xy=(2.3, 3.0), fontsize=8, ha='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.7, edgecolor='orange'))

# Add legend at bottom
legend_elements = [
    mpatches.Patch(facecolor='#2ecc71', edgecolor='black', label='High Traffic Flow (85 Mbps)'),
    mpatches.Patch(facecolor='#3498db', edgecolor='black', label='Low Traffic Flow (5 Mbps)'),
    mpatches.Patch(facecolor='#e74c3c', edgecolor='black', label='High Mobility (30 mph)'),
    mpatches.Patch(facecolor='#9b59b6', edgecolor='black', label='Low Mobility (3 mph)'),
    mpatches.Patch(facecolor='#fff3cd', edgecolor='#f39c12', label='Ensemble Model (0.75/0.25)'),
    mpatches.Patch(facecolor='#e8f8f5', edgecolor='#16a085', label='Robust Result (Δ < 5%)')
]

fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
          fontsize=9, frameon=True, fancybox=True, shadow=True,
          bbox_to_anchor=(0.5, -0.02))

# Overall title
fig.suptitle('Differential Experiments: System Architecture and Data Flow', 
            fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0.02, 1, 0.97])

# Save
output_path = 'figs/experiments_system_diagram.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Experiments diagram saved to: {output_path}")

plt.close()

# =============================================================================
# Create a second diagram showing data flow and metrics
# =============================================================================

fig2, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(6, 9.5, 'Data Flow and Performance Metrics Pipeline', 
       ha='center', fontsize=14, fontweight='bold')

# Stage 1: Data Generation
stage1_box = FancyBboxPatch((0.5, 7.0), 2.5, 1.8,
                           boxstyle="round,pad=0.15",
                           facecolor='#d5f4e6', edgecolor='#2ecc71', linewidth=2.5)
ax.add_patch(stage1_box)
ax.text(1.75, 8.5, 'Stage 1:', ha='center', fontsize=10, fontweight='bold')
ax.text(1.75, 8.15, 'Data Generation', ha='center', fontsize=9, fontweight='bold')
ax.text(1.75, 7.7, '• 500 samples/class', ha='left', fontsize=7.5)
ax.text(1.75, 7.45, '• 4 jamming types', ha='left', fontsize=7.5)
ax.text(1.75, 7.2, '• Profile features', ha='left', fontsize=7.5)

# Arrow
ax.annotate('', xy=(3.3, 7.9), xytext=(3.1, 7.9),
           arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))

# Stage 2: Feature Engineering
stage2_box = FancyBboxPatch((3.3, 7.0), 2.5, 1.8,
                           boxstyle="round,pad=0.15",
                           facecolor='#d6eaf8', edgecolor='#3498db', linewidth=2.5)
ax.add_patch(stage2_box)
ax.text(4.55, 8.5, 'Stage 2:', ha='center', fontsize=10, fontweight='bold')
ax.text(4.55, 8.15, 'Feature Engineering', ha='center', fontsize=9, fontweight='bold')
ax.text(4.55, 7.7, '• RSRP, SINR, SNR', ha='left', fontsize=7.5)
ax.text(4.55, 7.45, '• Throughput metrics', ha='left', fontsize=7.5)
ax.text(4.55, 7.2, '• Spectral features', ha='left', fontsize=7.5)

# Arrow
ax.annotate('', xy=(6.1, 7.9), xytext=(5.9, 7.9),
           arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))

# Stage 3: Training
stage3_box = FancyBboxPatch((6.1, 7.0), 2.5, 1.8,
                           boxstyle="round,pad=0.15",
                           facecolor='#fadbd8', edgecolor='#e74c3c', linewidth=2.5)
ax.add_patch(stage3_box)
ax.text(7.35, 8.5, 'Stage 3:', ha='center', fontsize=10, fontweight='bold')
ax.text(7.35, 8.15, 'Model Training', ha='center', fontsize=9, fontweight='bold')
ax.text(7.35, 7.7, '• CatBoost classifier', ha='left', fontsize=7.5)
ax.text(7.35, 7.45, '• Isolation Forest', ha='left', fontsize=7.5)
ax.text(7.35, 7.2, '• 70/30 train/test', ha='left', fontsize=7.5)

# Arrow
ax.annotate('', xy=(8.9, 7.9), xytext=(8.7, 7.9),
           arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))

# Stage 4: Evaluation
stage4_box = FancyBboxPatch((8.9, 7.0), 2.5, 1.8,
                           boxstyle="round,pad=0.15",
                           facecolor='#ebdef0', edgecolor='#9b59b6', linewidth=2.5)
ax.add_patch(stage4_box)
ax.text(10.15, 8.5, 'Stage 4:', ha='center', fontsize=10, fontweight='bold')
ax.text(10.15, 8.15, 'Evaluation', ha='center', fontsize=9, fontweight='bold')
ax.text(10.15, 7.7, '• F1, Accuracy', ha='left', fontsize=7.5)
ax.text(10.15, 7.45, '• Confusion matrix', ha='left', fontsize=7.5)
ax.text(10.15, 7.2, '• Per-class metrics', ha='left', fontsize=7.5)

# Performance Metrics Table
metrics_box = FancyBboxPatch((1.5, 3.5), 9.0, 2.8,
                            boxstyle="round,pad=0.2",
                            facecolor='white', edgecolor='#34495e', linewidth=3)
ax.add_patch(metrics_box)
ax.text(6.0, 6.0, 'Performance Metrics Summary', ha='center', fontsize=11, fontweight='bold')

# Table headers
headers = ['Scenario', 'Accuracy', 'F1 (Macro)', 'Normal F1', 'Inference (ms)']
x_positions = [2.0, 4.5, 6.5, 8.5, 10.0]
for i, header in enumerate(headers):
    ax.text(x_positions[i], 5.5, header, ha='center', fontsize=8, fontweight='bold')

# Experiment 1 data
ax.text(2.0, 5.1, 'Traffic Flow:', ha='left', fontsize=8, style='italic')
ax.text(2.0, 4.75, 'High Flow', ha='left', fontsize=7.5, color='#2ecc71')
ax.text(4.5, 4.75, '0.585', ha='center', fontsize=7.5)
ax.text(6.5, 4.75, '0.588', ha='center', fontsize=7.5, fontweight='bold')
ax.text(8.5, 4.75, '0.931', ha='center', fontsize=7.5)
ax.text(10.0, 4.75, '0.01', ha='center', fontsize=7.5)

ax.text(2.0, 4.4, 'Low Flow', ha='left', fontsize=7.5, color='#3498db')
ax.text(4.5, 4.4, '0.592', ha='center', fontsize=7.5)
ax.text(6.5, 4.4, '0.598', ha='center', fontsize=7.5, fontweight='bold')
ax.text(8.5, 4.4, '0.906', ha='center', fontsize=7.5)
ax.text(10.0, 4.4, '0.01', ha='center', fontsize=7.5)

ax.text(4.5, 4.05, 'Δ = 1.58%', ha='center', fontsize=7.5, 
       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

# Experiment 2 data
ax.text(2.0, 3.6, 'Mobility:', ha='left', fontsize=8, style='italic')
ax.text(2.0, 3.25, 'High (30mph)', ha='left', fontsize=7.5, color='#e74c3c')
ax.text(4.5, 3.25, '0.658', ha='center', fontsize=7.5)
ax.text(6.5, 3.25, '0.668', ha='center', fontsize=7.5, fontweight='bold')
ax.text(8.5, 3.25, '0.950', ha='center', fontsize=7.5)
ax.text(10.0, 3.25, '0.01', ha='center', fontsize=7.5)

ax.text(2.0, 2.9, 'Low (3mph)', ha='left', fontsize=7.5, color='#9b59b6')
ax.text(4.5, 2.9, '0.650', ha='center', fontsize=7.5)
ax.text(6.5, 2.9, '0.655', ha='center', fontsize=7.5, fontweight='bold')
ax.text(8.5, 2.9, '0.950', ha='center', fontsize=7.5)
ax.text(10.0, 2.9, '0.01', ha='center', fontsize=7.5)

ax.text(4.5, 2.55, 'Δ = 2.03%', ha='center', fontsize=7.5,
       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

# Key Findings Box
findings_box = FancyBboxPatch((1.5, 0.3), 9.0, 1.8,
                             boxstyle="round,pad=0.15",
                             facecolor='#e8f8f5', edgecolor='#16a085', linewidth=2.5)
ax.add_patch(findings_box)
ax.text(6.0, 1.85, 'Key Findings', ha='center', fontsize=10, fontweight='bold', color='#117a65')
ax.text(2.5, 1.5, '✓ Both experiments: Δ < 5% threshold → ROBUST', ha='left', fontsize=8)
ax.text(2.5, 1.2, '✓ Ensemble weights (0.75/0.25) stable across all scenarios', ha='left', fontsize=8)
ax.text(2.5, 0.9, '✓ Normal traffic detection: >90% F1 in all cases', ha='left', fontsize=8)
ax.text(2.5, 0.6, '✓ Real-time capable: <0.02 ms inference latency', ha='left', fontsize=8)

plt.tight_layout()

# Save
output_path2 = 'figs/experiments_data_flow_diagram.png'
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"✅ Data flow diagram saved to: {output_path2}")

plt.close()

print("\n✅ All experiment diagrams generated successfully!")
print(f"\n📊 Generated files:")
print(f"  1. figs/experiments_system_diagram.png")
print(f"  2. figs/experiments_data_flow_diagram.png")
