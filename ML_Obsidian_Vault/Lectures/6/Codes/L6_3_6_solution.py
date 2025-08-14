import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from math import log2

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_6")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("=" * 80)
print("QUESTION 6: C4.5 GAIN RATIO ANALYSIS")
print("=" * 80)

# Problem setup
print("\n1. PROBLEM WITH ID3's INFORMATION GAIN")
print("-" * 50)
print("ID3's information gain has a bias toward features with many values because:")
print("• Features with more unique values tend to create smaller, purer subsets")
print("• This gives them artificially high information gain scores")
print("• Features with many values may not generalize well (overfitting)")
print("• Example: A unique ID feature would have perfect information gain but zero predictive value")

# Given data for the problem
total_samples = 12
subset_sizes = [3, 5, 4]
feature_values = ['A', 'B', 'C']
information_gain = 0.8

print(f"\n2. CALCULATING SPLIT INFORMATION")
print("-" * 50)
print(f"Feature values: {feature_values}")
print(f"Dataset size: {total_samples}")
print(f"Subset sizes: {subset_sizes}")

# Calculate split information step by step
print(f"\nSplit Information Formula:")
print(f"Split Info = -∑(|Si|/|S|) * log2(|Si|/|S|)")
print(f"\nStep-by-step calculation:")

split_info_terms = []
for i, (value, size) in enumerate(zip(feature_values, subset_sizes)):
    proportion = size / total_samples
    log_term = log2(proportion)
    term_value = proportion * log_term
    split_info_terms.append(-term_value)
    
    print(f"For value {value}: |S{i+1}|/|S| = {size}/{total_samples} = {proportion:.3f}")
    print(f"  -({proportion:.3f}) * log2({proportion:.3f}) = -({proportion:.3f}) * ({log_term:.3f}) = {-term_value:.3f}")

split_information = sum(split_info_terms)
print(f"\nSplit Information = {' + '.join([f'{term:.3f}' for term in split_info_terms])}")
print(f"Split Information = {split_information:.3f}")

# Calculate gain ratio
print(f"\n3. CALCULATING GAIN RATIO")
print("-" * 50)
print(f"Information Gain = {information_gain}")
print(f"Split Information = {split_information:.3f}")
print(f"\nGain Ratio Formula:")
print(f"Gain Ratio = Information Gain / Split Information")
print(f"Gain Ratio = {information_gain} / {split_information:.3f} = {information_gain/split_information:.3f}")

gain_ratio_1 = information_gain / split_information

# Why split information corrects bias
print(f"\n4. WHY SPLIT INFORMATION CORRECTS BIAS")
print("-" * 50)
print("Split information penalizes features that create many small subsets by")
print("measuring the entropy of the split itself, thus normalizing information gain.")

# Binary feature comparison
print(f"\n5. BINARY FEATURE COMPARISON")
print("-" * 50)
binary_subset_sizes = [7, 5]
binary_information_gain = 0.6

print(f"Binary feature subset sizes: {binary_subset_sizes}")
print(f"Binary feature information gain: {binary_information_gain}")

# Calculate split information for binary feature
print(f"\nCalculating split information for binary feature:")
binary_split_info_terms = []
for i, size in enumerate(binary_subset_sizes):
    proportion = size / total_samples
    log_term = log2(proportion)
    term_value = proportion * log_term
    binary_split_info_terms.append(-term_value)
    
    print(f"For subset {i+1}: |S{i+1}|/|S| = {size}/{total_samples} = {proportion:.3f}")
    print(f"  -({proportion:.3f}) * log2({proportion:.3f}) = {-term_value:.3f}")

binary_split_information = sum(binary_split_info_terms)
binary_gain_ratio = binary_information_gain / binary_split_information

print(f"\nBinary Split Information = {binary_split_information:.3f}")
print(f"Binary Gain Ratio = {binary_information_gain} / {binary_split_information:.3f} = {binary_gain_ratio:.3f}")

print(f"\nCOMPARISON:")
print(f"3-valued feature: Gain Ratio = {gain_ratio_1:.3f}")
print(f"Binary feature:   Gain Ratio = {binary_gain_ratio:.3f}")
print(f"\nC4.5 would prefer the {'binary' if binary_gain_ratio > gain_ratio_1 else '3-valued'} feature!")

# Create visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Information Gain vs Split Information comparison
features = ['3-valued\nFeature', 'Binary\nFeature']
info_gains = [information_gain, binary_information_gain]
split_infos = [split_information, binary_split_information]
gain_ratios = [gain_ratio_1, binary_gain_ratio]

x_pos = np.arange(len(features))
width = 0.25

bars1 = ax1.bar(x_pos - width, info_gains, width, label='Information Gain', color='skyblue', alpha=0.8)
bars2 = ax1.bar(x_pos, split_infos, width, label='Split Information', color='lightcoral', alpha=0.8)
bars3 = ax1.bar(x_pos + width, gain_ratios, width, label='Gain Ratio', color='lightgreen', alpha=0.8)

ax1.set_xlabel('Feature Type')
ax1.set_ylabel('Value')
ax1.set_title('Information Gain vs Split Information vs Gain Ratio')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(features)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

# Plot 2: Split Information calculation breakdown
categories = ['Value A\n(3 samples)', 'Value B\n(5 samples)', 'Value C\n(4 samples)']
proportions = [size/total_samples for size in subset_sizes]
split_terms = split_info_terms

bars = ax2.bar(categories, split_terms, color=['orange', 'purple', 'brown'], alpha=0.7)
ax2.set_xlabel('Feature Values')
ax2.set_ylabel('Split Information Term')
ax2.set_title('Split Information Breakdown\nfor 3-valued Feature')
ax2.grid(True, alpha=0.3)

for bar, prop in zip(bars, proportions):
    height = bar.get_height()
    ax2.annotate(f'{height:.3f}\n(p={prop:.3f})',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

# Plot 3: Bias illustration - Information Gain vs Number of Values
num_values = np.arange(2, 11)
# Simulate how information gain might increase with more values (biased behavior)
biased_ig = 0.3 + 0.08 * num_values + 0.02 * num_values**2
# Simulate how gain ratio corrects this bias
corrected_gr = biased_ig / (0.5 + 0.15 * num_values)

ax3.plot(num_values, biased_ig, 'r-o', label='Information Gain (biased)', linewidth=2, markersize=6)
ax3.plot(num_values, corrected_gr, 'g-s', label='Gain Ratio (corrected)', linewidth=2, markersize=6)
ax3.set_xlabel('Number of Feature Values')
ax3.set_ylabel('Score')
ax3.set_title('ID3 Bias vs C4.5 Correction')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Comparison table visualization
comparison_data = {
    'Metric': ['Information Gain', 'Split Information', 'Gain Ratio'],
    '3-valued Feature': [information_gain, split_information, gain_ratio_1],
    'Binary Feature': [binary_information_gain, binary_split_information, binary_gain_ratio]
}

df = pd.DataFrame(comparison_data)
ax4.axis('tight')
ax4.axis('off')

table = ax4.table(cellText=[[f'{val:.3f}' if isinstance(val, float) else val for val in row] 
                           for row in df.values],
                 colLabels=df.columns,
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color code the header
for i in range(len(df.columns)):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Highlight the gain ratio row
for i in range(len(df.columns)):
    table[(3, i)].set_facecolor('#FFE082')

ax4.set_title('Detailed Comparison Table', pad=20)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'gain_ratio_analysis.png'), dpi=300, bbox_inches='tight')

# Create a second figure showing the step-by-step calculation
fig2, ax = plt.subplots(1, 1, figsize=(12, 8))

# Create a visual representation of the calculation steps
steps_text = [
    "Step 1: Calculate proportions",
    f"A: {subset_sizes[0]}/{total_samples} = {subset_sizes[0]/total_samples:.3f}",
    f"B: {subset_sizes[1]}/{total_samples} = {subset_sizes[1]/total_samples:.3f}",
    f"C: {subset_sizes[2]}/{total_samples} = {subset_sizes[2]/total_samples:.3f}",
    "",
    "Step 2: Calculate log terms",
    f"log_2({subset_sizes[0]/total_samples:.3f}) = {log2(subset_sizes[0]/total_samples):.3f}",
    f"log_2({subset_sizes[1]/total_samples:.3f}) = {log2(subset_sizes[1]/total_samples):.3f}",
    f"log_2({subset_sizes[2]/total_samples:.3f}) = {log2(subset_sizes[2]/total_samples):.3f}",
    "",
    "Step 3: Calculate split info terms",
    f"A: -{subset_sizes[0]/total_samples:.3f} x {log2(subset_sizes[0]/total_samples):.3f} = {split_info_terms[0]:.3f}",
    f"B: -{subset_sizes[1]/total_samples:.3f} x {log2(subset_sizes[1]/total_samples):.3f} = {split_info_terms[1]:.3f}",
    f"C: -{subset_sizes[2]/total_samples:.3f} x {log2(subset_sizes[2]/total_samples):.3f} = {split_info_terms[2]:.3f}",
    "",
    "Step 4: Sum split information",
    f"Split Info = {split_info_terms[0]:.3f} + {split_info_terms[1]:.3f} + {split_info_terms[2]:.3f} = {split_information:.3f}",
    "",
    "Step 5: Calculate gain ratio",
    f"Gain Ratio = {information_gain} / {split_information:.3f} = {gain_ratio_1:.3f}"
]

ax.text(0.05, 0.95, '\n'.join(steps_text), transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('Step-by-Step Calculation of Split Information and Gain Ratio', fontsize=14, fontweight='bold')

plt.savefig(os.path.join(save_dir, 'calculation_steps.png'), dpi=300, bbox_inches='tight')

print(f"\n" + "="*80)
print(f"SUMMARY AND CONCLUSIONS")
print("="*80)
print(f"1. ID3's bias: Features with more values get artificially high information gain")
print(f"2. Split Information for 3-valued feature: {split_information:.3f}")
print(f"3. Gain Ratio for 3-valued feature: {gain_ratio_1:.3f}")
print(f"4. Split Information corrects bias by normalizing with split entropy")
print(f"5. Binary feature Gain Ratio: {binary_gain_ratio:.3f}")
print(f"6. C4.5 prefers: {'Binary' if binary_gain_ratio > gain_ratio_1 else '3-valued'} feature")
print(f"\nImages saved to: {save_dir}")
