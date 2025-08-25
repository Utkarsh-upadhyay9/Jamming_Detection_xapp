#!/usr/bin/env python3
"""
High-Quality MATLAB-Style Figure Generator
Creates 3 publication-ready figures with proper text spacing and no overlaps
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from datetime import datetime
import os

# Configure matplotlib for high-quality MATLAB-style plots
plt.rcParams.update({
    'font.size': 13,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.shadow': False,
    'legend.framealpha': 1.0,
    'legend.edgecolor': 'black',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.3,
    'figure.autolayout': True
})

def create_figure_1_performance_comparison():
    """Figure 1: Clean Performance Metrics Comparison"""
    
    # Performance data
    metrics = ['F1-Score', 'Accuracy', 'Precision', 'Recall']
    paper_values = [95.4, 95.6, 94.8, 96.1]
    our_rf_values = [97.5, 97.5, 97.6, 97.5]
    our_svm_values = [97.7, 97.7, 97.7, 97.7]
    our_ensemble_values = [97.5, 97.5, 97.6, 97.5]
    
    # Create figure with generous spacing
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Bar positions with ample spacing
    x = np.arange(len(metrics))
    width = 0.18
    
    # Create bars with distinct MATLAB colors
    bars1 = ax.bar(x - 1.5*width, paper_values, width, 
                   label='Paper Baseline', color='#0072BD', 
                   edgecolor='black', linewidth=1.2, alpha=0.9)
    bars2 = ax.bar(x - 0.5*width, our_rf_values, width, 
                   label='Random Forest', color='#D95319', 
                   edgecolor='black', linewidth=1.2, alpha=0.9)
    bars3 = ax.bar(x + 0.5*width, our_svm_values, width, 
                   label='SVM', color='#EDB120', 
                   edgecolor='black', linewidth=1.2, alpha=0.9)
    bars4 = ax.bar(x + 1.5*width, our_ensemble_values, width, 
                   label='Ensemble', color='#7E2F8E', 
                   edgecolor='black', linewidth=1.2, alpha=0.9)
    
    # Add clean value labels with proper spacing
    def add_clean_labels(bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                   f'{value:.1f}%', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold', 
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    add_clean_labels(bars1, paper_values)
    add_clean_labels(bars2, our_rf_values)
    add_clean_labels(bars3, our_svm_values)
    add_clean_labels(bars4, our_ensemble_values)
    
    # Clean formatting with proper spacing
    ax.set_xlabel('Performance Metrics', fontsize=16, fontweight='bold', labelpad=15)
    ax.set_ylabel('Performance (%)', fontsize=16, fontweight='bold', labelpad=15)
    ax.set_title('Jamming Detection Performance Comparison\nPaper Baseline vs Our Implementation', 
                fontsize=18, fontweight='bold', pad=25)
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=14)
    ax.set_ylim(90, 101)
    
    # Legend with proper positioning
    ax.legend(loc='lower right', fontsize=12, ncol=2, 
             bbox_to_anchor=(0.98, 0.02), framealpha=1.0)
    
    # Clean grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig

def create_figure_2_confusion_matrix():
    """Figure 2: Clean Confusion Matrix Heatmap"""
    
    # Confusion matrix data from our results
    confusion_matrix = np.array([
        [3500, 0, 0, 0],      # Normal: Perfect
        [0, 477, 12, 11],     # Power Jamming  
        [0, 25, 423, 52],     # Sweep Jamming
        [0, 11, 12, 477]      # Intelligent Jamming
    ])
    
    # Normalize to percentages
    confusion_matrix_norm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure with proper size
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create clean heatmap
    im = ax.imshow(confusion_matrix_norm, interpolation='nearest', 
                   cmap='Blues', vmin=0, vmax=100, aspect='equal')
    
    # Add colorbar with proper spacing
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.05)
    cbar.set_label('Classification Accuracy (%)', fontsize=14, 
                   fontweight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=12)
    
    # Clean class labels
    class_names = ['Normal\nTraffic', 'Power\nJamming', 'Sweep\nJamming', 'Intelligent\nJamming']
    
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=13, ha='center')
    ax.set_yticklabels(class_names, fontsize=13, va='center')
    
    # Add clean text annotations with proper contrast
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = confusion_matrix_norm[i, j]
            text_color = "white" if value > 50 else "black"
            ax.text(j, i, f'{value:.1f}%', ha="center", va="center", 
                   color=text_color, fontsize=12, fontweight='bold')
    
    # Clean title and labels
    ax.set_title('Confusion Matrix - Jamming Attack Classification\nEnsemble Model Performance', 
                fontsize=18, fontweight='bold', pad=25)
    ax.set_xlabel('Predicted Label', fontsize=16, fontweight='bold', labelpad=15)
    ax.set_ylabel('True Label', fontsize=16, fontweight='bold', labelpad=15)
    
    plt.tight_layout()
    return fig

def create_figure_3_dual_analysis():
    """Figure 3: Dual Performance Analysis (2 subplots)"""
    
    # Create figure with two clean subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # LEFT SUBPLOT: Training Efficiency vs Performance
    models = ['Random\nForest', 'SVM', 'Ensemble']
    training_times = [0.38, 3.78, 0.38]
    f1_improvements = [2.1, 2.3, 2.1]  # Over paper baseline
    colors = ['#D95319', '#EDB120', '#7E2F8E']
    sizes = [300, 300, 400]  # Bubble sizes
    
    scatter = ax1.scatter(training_times, f1_improvements, s=sizes, 
                         c=colors, alpha=0.7, edgecolors='black', 
                         linewidth=2)
    
    # Clean annotations with spacing
    annotations = [
        (0.38, 2.1, 'Random Forest\n(Fast & Accurate)'),
        (3.78, 2.3, 'SVM\n(Highest Accuracy)'),
        (0.38, 2.1, 'Ensemble\n(Best Balance)')
    ]
    
    offsets = [(20, 20), (-80, 20), (-20, -40)]
    for i, (x, y, label) in enumerate(annotations):
        if i == 2:  # Ensemble - adjust position to avoid overlap
            ax1.annotate(label, (x, y), xytext=offsets[i], 
                        textcoords='offset points', fontsize=11, 
                        fontweight='bold', ha='center',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                edgecolor=colors[i], alpha=0.9))
    
    # Only annotate RF and SVM to avoid overlap
    for i in range(2):
        x, y, label = annotations[i]
        ax1.annotate(label, (x, y), xytext=offsets[i], 
                    textcoords='offset points', fontsize=11, 
                    fontweight='bold', ha='center',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                            edgecolor=colors[i], alpha=0.9))
    
    ax1.set_xlabel('Training Time (seconds)', fontsize=14, fontweight='bold', labelpad=10)
    ax1.set_ylabel('F1-Score Improvement (%)', fontsize=14, fontweight='bold', labelpad=10)
    ax1.set_title('Training Efficiency vs Performance Gain', fontsize=16, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.3, 4.5)
    ax1.set_ylim(1.8, 2.6)
    
    # RIGHT SUBPLOT: Attack-Specific Performance
    attack_types = ['Normal', 'Power\nJamming', 'Sweep\nJamming', 'Intelligent\nJamming']
    f1_scores = [100.0, 95.6, 88.1, 91.6]
    colors_2 = ['#2E8B57', '#FF6347', '#FFA500', '#9370DB']
    
    bars = ax2.bar(attack_types, f1_scores, color=colors_2, 
                   edgecolor='black', linewidth=1.5, alpha=0.8, width=0.6)
    
    # Clean value labels
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{score:.1f}%', ha='center', va='bottom', 
                fontsize=12, fontweight='bold')
    
    # Paper baseline reference line
    ax2.axhline(y=95.4, color='red', linestyle='--', linewidth=2.5, alpha=0.8)
    ax2.text(1.5, 96.2, 'Paper Baseline (95.4%)', fontsize=11, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='red', alpha=0.9))
    
    ax2.set_xlabel('Attack Types', fontsize=14, fontweight='bold', labelpad=10)
    ax2.set_ylabel('F1-Score (%)', fontsize=14, fontweight='bold', labelpad=10)
    ax2.set_title('Attack-Specific Detection Performance', fontsize=16, fontweight='bold', pad=15)
    ax2.set_ylim(80, 105)
    ax2.grid(True, alpha=0.3)
    
    # Remove top and right spines
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Main title with proper spacing
    fig.suptitle('Comprehensive Performance Analysis\nJamming Detection System Performance', 
                fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, wspace=0.3)
    return fig

def generate_all_figures():
    """Generate all three high-quality figures"""
    
    print("🎨 Generating high-quality MATLAB-style figures...")
    print("🎯 Focus: Clean layout, no text overlaps, publication-ready")
    print("=" * 65)
    
    # Create output directory
    os.makedirs('figures', exist_ok=True)
    
    # Generate Figure 1: Performance Comparison
    print("\n📊 Creating Figure 1: Performance Comparison...")
    fig1 = create_figure_1_performance_comparison()
    
    # Save with high quality
    fig1.savefig('figures/figure1_clean_performance.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    fig1.savefig('figures/figure1_clean_performance.pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig1)
    print("   ✅ Figure 1 saved successfully")
    
    # Generate Figure 2: Confusion Matrix
    print("\n📊 Creating Figure 2: Confusion Matrix...")
    fig2 = create_figure_2_confusion_matrix()
    
    fig2.savefig('figures/figure2_clean_confusion.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig2.savefig('figures/figure2_clean_confusion.pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig2)
    print("   ✅ Figure 2 saved successfully")
    
    # Generate Figure 3: Dual Analysis
    print("\n📊 Creating Figure 3: Performance Analysis...")
    fig3 = create_figure_3_dual_analysis()
    
    fig3.savefig('figures/figure3_clean_analysis.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig3.savefig('figures/figure3_clean_analysis.pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig3)
    print("   ✅ Figure 3 saved successfully")
    
    print("\n" + "="*65)
    print("🎉 ALL FIGURES GENERATED SUCCESSFULLY!")
    print("📁 Location: ./figures/")
    print("📊 Files created:")
    print("   • figure1_clean_performance.png/pdf - Performance comparison")
    print("   • figure2_clean_confusion.png/pdf - Confusion matrix") 
    print("   • figure3_clean_analysis.png/pdf - Dual performance analysis")
    print("\n✨ High-quality MATLAB-style figures with:")
    print("   ✓ No text overlaps")
    print("   ✓ Proper spacing and padding")
    print("   ✓ Clean professional layout")
    print("   ✓ Publication-ready quality (300 DPI)")
    print("   ✓ Both PNG and PDF formats")

if __name__ == "__main__":
    generate_all_figures()
