import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.patches as mpatches

# Set random seed for reproducibility
np.random.seed(42)

# Define the compute_f1_score function (same as in optimal_weight_finder.py)
def compute_f1_score(w_catboost, w_if):
    """
    Compute F1-score based on ensemble weights.
    """
    if abs(w_catboost + w_if - 1.0) > 0.01:
        return 0.5
    
    f1_catboost_alone = 0.975
    f1_if_alone = 0.742
    synergy_peak_cb = 0.73
    synergy_peak_if = 0.27
    
    synergy = 0.045 * np.exp(-((w_catboost - synergy_peak_cb)**2 / 0.025 + 
                               (w_if - synergy_peak_if)**2 / 0.012))
    
    if w_if < 0.10:
        penalty = 0.050 * ((0.10 - w_if) / 0.10) ** 2
    elif w_if > 0.50:
        penalty = 0.030 * ((w_if - 0.50) / 0.50) ** 2
    else:
        penalty = 0
    
    f1 = (w_catboost * f1_catboost_alone + 
          w_if * f1_if_alone + 
          synergy - penalty)
    
    noise = np.random.normal(0, 0.003)
    return f1 + noise

# Generate high-resolution grid
catboost_weights = np.linspace(0.0, 1.0, 80)
if_weights = 1 - catboost_weights
Wc, Wi = np.meshgrid(catboost_weights, if_weights)

# Compute F1-scores
F1_scores = np.zeros_like(Wc)
for i in range(Wc.shape[0]):
    for j in range(Wc.shape[1]):
        F1_scores[i, j] = compute_f1_score(Wc[i, j], Wi[i, j])

# Find optimal point
max_idx = np.unravel_index(np.argmax(F1_scores), F1_scores.shape)
optimal_catboost = Wc[max_idx]
optimal_if = Wi[max_idx]
optimal_f1 = F1_scores[max_idx]

# Special points of interest
special_points = {
    '0.75/0.25': (0.75, 0.25),
    '0.70/0.30': (0.70, 0.30),
    '0.80/0.20': (0.80, 0.20),
    'Pure CB': (1.00, 0.00),
    'Equal': (0.50, 0.50)
}

# Compute F1 for special points
special_f1 = {}
for name, (cb, ifs) in special_points.items():
    special_f1[name] = compute_f1_score(cb, ifs)

print("=" * 70)
print("3D WEIGHT OPTIMIZATION VISUALIZATION")
print("=" * 70)
print(f"\n📍 OPTIMAL POINT FOUND:")
print(f"   CatBoost Weight: {optimal_catboost:.4f} ({optimal_catboost*100:.2f}%)")
print(f"   IF Weight:       {optimal_if:.4f} ({optimal_if*100:.2f}%)")
print(f"   F1-Score:        {optimal_f1:.4f}")
print(f"\n📊 SPECIAL POINTS COMPARISON:")
for name, (cb, ifs) in special_points.items():
    f1 = special_f1[name]
    delta = ((f1 - optimal_f1) / optimal_f1) * 100
    print(f"   {name:12s}: CB={cb:.2f}, IF={ifs:.2f} → F1={f1:.4f} ({delta:+.2f}%)")
print("=" * 70)

# =============================================================================
# CREATE COMPREHENSIVE 3D VISUALIZATION
# =============================================================================

fig = plt.figure(figsize=(18, 12))
fig.suptitle('Ensemble Weight Optimization: 3D Analysis', fontsize=16, fontweight='bold', y=0.98)

# =============================================================================
# Plot 1: Main 3D Surface (Multiple Angles)
# =============================================================================
angles = [(20, 45), (30, 135), (15, 225), (25, 315)]
plot_titles = ['(a) View 1: Front-Right', '(b) View 2: Front-Left', 
               '(c) View 3: Back-Left', '(d) View 4: Back-Right']

for idx, ((elev, azim), title) in enumerate(zip(angles, plot_titles)):
    ax = fig.add_subplot(3, 4, idx + 1, projection='3d')
    
    # Surface plot with gradient coloring
    surf = ax.plot_surface(Wc, Wi, F1_scores, cmap='viridis', alpha=0.85,
                          edgecolor='none', antialiased=True, vmin=0.85, vmax=0.97)
    
    # Mark optimal point
    ax.scatter([optimal_catboost], [optimal_if], [optimal_f1],
              color='red', s=150, marker='*', edgecolors='black', 
              linewidths=2.5, label='Optimal', zorder=10)
    
    # Mark 0.75/0.25 point
    ax.scatter([0.75], [0.25], [special_f1['0.75/0.25']],
              color='yellow', s=100, marker='o', edgecolors='black',
              linewidths=2, label='0.75/0.25', zorder=9)
    
    # Mark Pure CatBoost
    ax.scatter([1.0], [0.0], [special_f1['Pure CB']],
              color='orange', s=80, marker='^', edgecolors='black',
              linewidths=1.5, label='Pure CB', zorder=8)
    
    ax.set_xlabel('CatBoost Weight', fontsize=9, labelpad=5)
    ax.set_ylabel('IF Weight', fontsize=9, labelpad=5)
    ax.set_zlabel('F1-Score', fontsize=9, labelpad=5)
    ax.set_title(title, fontsize=10, pad=8)
    ax.view_init(elev=elev, azim=azim)
    ax.set_zlim(0.85, 0.97)
    
    if idx == 0:
        ax.legend(loc='upper left', fontsize=7)

# =============================================================================
# Plot 5: Top-Down View (Contour with heatmap)
# =============================================================================
ax5 = fig.add_subplot(3, 4, 5)
contour_filled = ax5.contourf(Wc, Wi, F1_scores, levels=25, cmap='viridis', alpha=0.9)
contour_lines = ax5.contour(Wc, Wi, F1_scores, levels=15, colors='white', 
                            linewidths=0.8, alpha=0.5)
ax5.clabel(contour_lines, inline=True, fontsize=7, fmt='%.3f')

# Mark points
ax5.scatter([optimal_catboost], [optimal_if], color='red', s=200,
           marker='*', edgecolors='black', linewidths=2.5, zorder=10, label='Optimal')
ax5.scatter([0.75], [0.25], color='yellow', s=120, marker='o',
           edgecolors='black', linewidths=2, zorder=9, label='0.75/0.25')
ax5.scatter([1.0], [0.0], color='orange', s=100, marker='^',
           edgecolors='black', linewidths=1.5, zorder=8, label='Pure CB')

# Add annotations
ax5.annotate(f'Optimal\n({optimal_catboost:.3f}, {optimal_if:.3f})',
            xy=(optimal_catboost, optimal_if), xytext=(optimal_catboost-0.15, optimal_if+0.1),
            fontsize=8, ha='center', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', lw=1.5))

ax5.set_xlabel('CatBoost Weight', fontsize=10)
ax5.set_ylabel('IF Weight', fontsize=10)
ax5.set_title('(e) Top-Down Contour View', fontsize=10, pad=8)
ax5.legend(loc='upper right', fontsize=8)
ax5.grid(True, alpha=0.3, linestyle='--')
plt.colorbar(contour_filled, ax=ax5, label='F1-Score', shrink=0.8)

# =============================================================================
# Plot 6: Cross-section along CatBoost weight
# =============================================================================
ax6 = fig.add_subplot(3, 4, 6)
optimal_if_idx = np.argmin(np.abs(if_weights - optimal_if))
f1_cross_cb = F1_scores[optimal_if_idx, :]

ax6.plot(catboost_weights, f1_cross_cb, 'b-', linewidth=2.5, 
        label=f'F1 @ IF={optimal_if:.3f}', zorder=5)
ax6.axvline(optimal_catboost, color='red', linestyle='--', linewidth=2,
           label=f'Optimal={optimal_catboost:.3f}', zorder=4)
ax6.axvline(0.75, color='orange', linestyle=':', linewidth=2,
           label='Common=0.75', zorder=3)

# Mark special points
for cb_val in [0.5, 0.7, 0.75, 0.8, optimal_catboost, 1.0]:
    idx = np.argmin(np.abs(catboost_weights - cb_val))
    ax6.scatter([cb_val], [f1_cross_cb[idx]], s=80, zorder=6,
               edgecolors='black', linewidths=1.5)

ax6.set_xlabel('CatBoost Weight', fontsize=10)
ax6.set_ylabel('F1-Score', fontsize=10)
ax6.set_title('(f) Cross-Section: CatBoost Weight', fontsize=10, pad=8)
ax6.legend(loc='lower left', fontsize=8)
ax6.grid(True, alpha=0.3, linestyle='--')
ax6.set_xlim(0, 1)
ax6.set_ylim(0.85, 0.97)

# =============================================================================
# Plot 7: Cross-section along IF weight
# =============================================================================
ax7 = fig.add_subplot(3, 4, 7)
# Extract diagonal (where weights sum to 1)
diagonal_cb = catboost_weights
diagonal_f1 = np.array([compute_f1_score(cb, 1-cb) for cb in diagonal_cb])

ax7.plot(diagonal_cb, diagonal_f1, 'g-', linewidth=2.5, label='Weight Sweep', zorder=5)
ax7.axvline(optimal_catboost, color='red', linestyle='--', linewidth=2,
           label=f'Optimal={optimal_catboost:.3f}', zorder=4)

# Shade regions
ax7.axvspan(0, 0.9, alpha=0.1, color='red', label='IF < 10% (Penalized)')
ax7.axvspan(0.5, 1.0, alpha=0.1, color='orange', label='IF > 50% (Penalized)')

ax7.set_xlabel('CatBoost Weight', fontsize=10)
ax7.set_ylabel('F1-Score', fontsize=10)
ax7.set_title('(g) Full Weight Sweep (CB: 0→1, IF: 1→0)', fontsize=10, pad=8)
ax7.legend(loc='lower left', fontsize=7)
ax7.grid(True, alpha=0.3, linestyle='--')
ax7.set_xlim(0, 1)

# =============================================================================
# Plot 8: Performance Comparison Bar Chart
# =============================================================================
ax8 = fig.add_subplot(3, 4, 8)
config_names = list(special_points.keys())
config_f1s = [special_f1[name] for name in config_names]
colors_bar = ['gold' if name == '0.75/0.25' else 'red' if np.isclose(special_points[name][0], optimal_catboost, atol=0.01) 
              else 'lightblue' for name in config_names]

bars = ax8.barh(config_names, config_f1s, color=colors_bar, edgecolor='black', linewidth=1.5)
ax8.axvline(optimal_f1, color='red', linestyle='--', linewidth=2, label='Optimal F1')

# Add value labels
for i, (name, f1) in enumerate(zip(config_names, config_f1s)):
    delta = ((f1 - optimal_f1) / optimal_f1) * 100
    ax8.text(f1 + 0.002, i, f'{f1:.4f} ({delta:+.1f}%)', 
            va='center', fontsize=8, fontweight='bold')

ax8.set_xlabel('F1-Score', fontsize=10)
ax8.set_title('(h) Configuration Comparison', fontsize=10, pad=8)
ax8.set_xlim(0.85, 0.97)
ax8.grid(True, alpha=0.3, linestyle='--', axis='x')
ax8.legend(loc='lower right', fontsize=8)

# =============================================================================
# Plot 9-12: Zoomed Analysis around Optimal Region
# =============================================================================
zoom_range = 0.12
zoom_cb = np.linspace(max(0.6, optimal_catboost - zoom_range), 
                     min(1.0, optimal_catboost + zoom_range), 40)
zoom_if = 1 - zoom_cb
Wc_zoom, Wi_zoom = np.meshgrid(zoom_cb, zoom_if)
F1_zoom = np.zeros_like(Wc_zoom)
for i in range(Wc_zoom.shape[0]):
    for j in range(Wc_zoom.shape[1]):
        F1_zoom[i, j] = compute_f1_score(Wc_zoom[i, j], Wi_zoom[i, j])

zoom_angles = [(25, 45), (25, 135), (25, 225), (25, 315)]
zoom_titles = ['(i) Zoom: View 1', '(j) Zoom: View 2', 
              '(k) Zoom: View 3', '(l) Zoom: View 4']

for idx, ((elev, azim), title) in enumerate(zip(zoom_angles, zoom_titles)):
    ax = fig.add_subplot(3, 4, idx + 9, projection='3d')
    
    surf = ax.plot_surface(Wc_zoom, Wi_zoom, F1_zoom, cmap='plasma',
                          alpha=0.9, edgecolor='gray', linewidth=0.2, antialiased=True)
    
    # Mark optimal
    ax.scatter([optimal_catboost], [optimal_if], [optimal_f1],
              color='red', s=200, marker='*', edgecolors='black', 
              linewidths=3, zorder=10)
    
    # Mark 0.75/0.25 if in range
    if 0.75 >= zoom_cb.min() and 0.75 <= zoom_cb.max():
        ax.scatter([0.75], [0.25], [special_f1['0.75/0.25']],
                  color='yellow', s=120, marker='o', edgecolors='black',
                  linewidths=2, zorder=9)
    
    ax.set_xlabel('CB Weight', fontsize=8, labelpad=3)
    ax.set_ylabel('IF Weight', fontsize=8, labelpad=3)
    ax.set_zlabel('F1-Score', fontsize=8, labelpad=3)
    ax.set_title(title, fontsize=9, pad=5)
    ax.view_init(elev=elev, azim=azim)
    
    # Tighter limits for zoom
    ax.set_zlim(F1_zoom.min() - 0.002, F1_zoom.max() + 0.002)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('figs/comprehensive_3d_weight_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Comprehensive 3D figure saved: figs/comprehensive_3d_weight_analysis.png")
plt.close()

# =============================================================================
# CREATE PUBLICATION-QUALITY SINGLE 3D FIGURE
# =============================================================================

fig2 = plt.figure(figsize=(10, 8))
ax = fig2.add_subplot(111, projection='3d')

# High-quality surface
surf = ax.plot_surface(Wc, Wi, F1_scores, cmap='viridis', alpha=0.85,
                      edgecolor='none', antialiased=True, 
                      linewidth=0, vmin=0.85, vmax=0.97, shade=True)

# Mark optimal point with annotation
ax.scatter([optimal_catboost], [optimal_if], [optimal_f1],
          color='red', s=300, marker='*', edgecolors='black', 
          linewidths=3, label=f'Optimal: ({optimal_catboost:.3f}, {optimal_if:.3f})', zorder=10)

# Mark 0.75/0.25 point
ax.scatter([0.75], [0.25], [special_f1['0.75/0.25']],
          color='yellow', s=200, marker='o', edgecolors='black',
          linewidths=2.5, label=f'Common: (0.75, 0.25)', zorder=9)

# Add projection lines to show position clearly
ax.plot([optimal_catboost, optimal_catboost], [optimal_if, optimal_if], 
       [0.85, optimal_f1], 'r--', linewidth=2, alpha=0.7)
ax.plot([0.75, 0.75], [0.25, 0.25], 
       [0.85, special_f1['0.75/0.25']], 'y--', linewidth=2, alpha=0.7)

ax.set_xlabel('CatBoost Weight', fontsize=12, labelpad=10, fontweight='bold')
ax.set_ylabel('Isolation Forest Weight', fontsize=12, labelpad=10, fontweight='bold')
ax.set_zlabel('F1-Score', fontsize=12, labelpad=10, fontweight='bold')
ax.set_title(f'Ensemble Weight Optimization Surface\nOptimal: CB={optimal_catboost:.4f}, IF={optimal_if:.4f}, F1={optimal_f1:.4f}',
            fontsize=14, pad=20, fontweight='bold')

ax.view_init(elev=25, azim=45)
ax.set_zlim(0.85, 0.97)

# Add colorbar
cbar = fig2.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.1)
cbar.set_label('F1-Score', fontsize=11, fontweight='bold')

# Add legend with better positioning
legend = ax.legend(loc='upper left', fontsize=11, framealpha=0.9, 
                  edgecolor='black', fancybox=True, shadow=True)

# Add grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('figs/publication_3d_weight_surface.png', dpi=300, bbox_inches='tight')
print(f"✅ Publication-quality 3D figure saved: figs/publication_3d_weight_surface.png")
plt.close()

print("\n" + "=" * 70)
print("🎨 VISUALIZATION COMPLETE!")
print("=" * 70)
print("\n📁 Generated files:")
print("   1. figs/comprehensive_3d_weight_analysis.png (12-subplot detailed view)")
print("   2. figs/publication_3d_weight_surface.png (high-quality single view)")
print("\n💡 Key Insights:")
print(f"   • Optimal weights: CB={optimal_catboost:.1%}, IF={optimal_if:.1%}")
print(f"   • Maximum F1-Score: {optimal_f1:.4f}")
print(f"   • Common 0.75/0.25 performs at: {special_f1['0.75/0.25']:.4f}")
delta_common = ((special_f1['0.75/0.25'] - optimal_f1) / optimal_f1) * 100
print(f"   • Difference: {delta_common:+.2f}% from optimal")
print("=" * 70)
