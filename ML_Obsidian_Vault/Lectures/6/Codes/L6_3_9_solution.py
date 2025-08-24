import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import log2
from itertools import combinations
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_9")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

print("=" * 80)
print("QUESTION 9: CART BINARY SPLITTING STRATEGY")
print("=" * 80)

# Helper function
def calculate_gini(labels):
    """Calculate Gini impurity of a list of labels"""
    if len(labels) == 0:
        return 0
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    gini = 1 - sum(p**2 for p in probabilities)
    return gini

def generate_binary_splits(values):
    """Generate all possible binary splits for categorical values"""
    n = len(values)
    if n <= 1:
        return []
    
    splits = []
    # Generate all possible non-empty proper subsets
    for r in range(1, n):
        for subset in combinations(values, r):
            set1 = list(subset)
            set2 = [v for v in values if v not in subset]
            splits.append((set1, set2))
    
    return splits

print("1. BINARY SPLITS FOR GRADE FEATURE")
print("=" * 50)

# Grade feature with values {A, B, C, D}
grade_values = ['A', 'B', 'C', 'D']
print(f"Feature 'Grade' values: {grade_values}")

# Generate all binary splits
binary_splits = generate_binary_splits(grade_values)

print(f"\nAll possible binary splits:")
print(f"Number of splits: {len(binary_splits)}")
print()

for i, (set1, set2) in enumerate(binary_splits, 1):
    print(f"Split {i}: {{{', '.join(sorted(set1))}}} vs {{{', '.join(sorted(set2))}}}")

print(f"\n2. FORMULA FOR NUMBER OF BINARY SPLITS")
print("=" * 50)
print(f"For a categorical feature with k values:")
print(f"Number of binary splits = 2^(k-1) - 1")
print(f"\nExplanation:")
print(f"- Total ways to partition k items into 2 non-empty sets = 2^k - 2")
print(f"- Since {{A}} vs {{B,C,D}} is the same as {{B,C,D}} vs {{A}}, divide by 2")
print(f"- Formula: (2^k - 2) / 2 = 2^(k-1) - 1")

# Verify for different k values
k_values = range(2, 8)
formula_results = []
actual_results = []

print(f"\nVerification for different k values:")
for k in k_values:
    values = [f"V{i}" for i in range(k)]
    actual_splits = generate_binary_splits(values)
    actual_count = len(actual_splits)
    formula_count = 2**(k-1) - 1
    
    formula_results.append(formula_count)
    actual_results.append(actual_count)
    
    print(f"k={k}: Formula={formula_count}, Actual={actual_count}, Match={formula_count==actual_count}")

print(f"\nFor our Grade feature (k=4):")
print(f"Number of binary splits = 2^(4-1) - 1 = 2^3 - 1 = 8 - 1 = 7")

print(f"\n3. WHAT CART STANDS FOR")
print("=" * 50)
print(f"CART = Classification And Regression Trees")
print(f"\nKey characteristics:")
print(f"• Can handle both classification AND regression problems")
print(f"• Uses binary splits only (unlike ID3/C4.5 which use multi-way splits)")
print(f"• For classification: uses Gini impurity")
print(f"• For regression: uses mean squared error (MSE)")
print(f"• Can handle missing values through surrogate splits")
print(f"• Includes built-in pruning mechanisms")

print(f"\nWhy CART can handle regression:")
print(f"• For regression, CART predicts the mean value of target in each leaf")
print(f"• Uses MSE to evaluate splits: MSE = Σ(yi - ŷ)² / n")
print(f"• Chooses splits that minimize weighted MSE of child nodes")
print(f"• ID3 and C4.5 are designed only for categorical targets (classification)")

print(f"\n4. OPTIMAL BINARY SPLIT USING GINI IMPURITY")
print("=" * 50)

# Given class distributions for Grade values
grade_distributions = {
    'A': {'class_0': 3, 'class_1': 1},  # A(3,1)
    'B': {'class_0': 2, 'class_1': 2},  # B(2,2)
    'C': {'class_0': 1, 'class_1': 3},  # C(1,3)
    'D': {'class_0': 4, 'class_1': 0}   # D(4,0)
}

print("Given class distributions:")
total_samples = 0
total_class_0 = 0
total_class_1 = 0

for grade, dist in grade_distributions.items():
    samples = dist['class_0'] + dist['class_1']
    total_samples += samples
    total_class_0 += dist['class_0']
    total_class_1 += dist['class_1']
    print(f"  Grade {grade}: {samples} samples → Class 0: {dist['class_0']}, Class 1: {dist['class_1']}")

print(f"\nTotal dataset: {total_samples} samples")
print(f"Overall distribution: Class 0: {total_class_0}, Class 1: {total_class_1}")

# Calculate baseline Gini
baseline_gini = calculate_gini(['0']*total_class_0 + ['1']*total_class_1)
print(f"Baseline Gini impurity: {baseline_gini:.4f}")

# Evaluate all binary splits
print(f"\nEvaluating all binary splits:")
print(f"Split evaluation: Gini_gain = Gini_parent - (weight_left × Gini_left + weight_right × Gini_right)")

best_split = None
best_gini_gain = -1
split_results = []

for i, (set1, set2) in enumerate(binary_splits, 1):
    # Calculate samples and class distributions for each side
    left_class_0 = sum(grade_distributions[grade]['class_0'] for grade in set1)
    left_class_1 = sum(grade_distributions[grade]['class_1'] for grade in set1)
    left_total = left_class_0 + left_class_1
    
    right_class_0 = sum(grade_distributions[grade]['class_0'] for grade in set2)
    right_class_1 = sum(grade_distributions[grade]['class_1'] for grade in set2)
    right_total = right_class_0 + right_class_1
    
    # Calculate Gini impurities
    left_gini = calculate_gini(['0']*left_class_0 + ['1']*left_class_1)
    right_gini = calculate_gini(['0']*right_class_0 + ['1']*right_class_1)
    
    # Calculate weighted Gini
    weighted_gini = (left_total/total_samples * left_gini + 
                    right_total/total_samples * right_gini)
    
    # Calculate Gini gain
    gini_gain = baseline_gini - weighted_gini
    
    split_results.append({
        'split_num': i,
        'set1': set1,
        'set2': set2,
        'left_dist': (left_class_0, left_class_1),
        'right_dist': (right_class_0, right_class_1),
        'left_gini': left_gini,
        'right_gini': right_gini,
        'weighted_gini': weighted_gini,
        'gini_gain': gini_gain
    })
    
    print(f"\nSplit {i}: {{{', '.join(sorted(set1))}}} vs {{{', '.join(sorted(set2))}}}")
    print(f"  Left:  {left_total} samples ({left_class_0}, {left_class_1}) → Gini = {left_gini:.4f}")
    print(f"  Right: {right_total} samples ({right_class_0}, {right_class_1}) → Gini = {right_gini:.4f}")
    print(f"  Weighted Gini = {left_total}/{total_samples} × {left_gini:.4f} + {right_total}/{total_samples} × {right_gini:.4f} = {weighted_gini:.4f}")
    print(f"  Gini Gain = {baseline_gini:.4f} - {weighted_gini:.4f} = {gini_gain:.4f}")
    
    if gini_gain > best_gini_gain:
        best_gini_gain = gini_gain
        best_split = (set1, set2)

print(f"\n" + "="*60)
print(f"OPTIMAL SPLIT RESULT")
print("="*60)
print(f"Best split: {{{', '.join(sorted(best_split[0]))}}} vs {{{', '.join(sorted(best_split[1]))}}}")
print(f"Best Gini gain: {best_gini_gain:.4f}")

# Create visualizations
fig = plt.figure(figsize=(20, 15))

# Plot 1: Number of binary splits vs k
ax1 = plt.subplot(2, 4, 1)
ax1.plot(k_values, formula_results, 'bo-', label='Formula: $2^{k-1} - 1$', linewidth=2, markersize=8)
ax1.plot(k_values, actual_results, 'rs--', label='Actual count', linewidth=2, markersize=8)
ax1.set_xlabel('Number of values (k)')
ax1.set_ylabel('Number of binary splits')
ax1.set_title('Binary Splits vs Feature Cardinality')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Highlight k=4
ax1.axvline(x=4, color='red', linestyle=':', alpha=0.7)
ax1.text(4, 5, 'Grade\n(k=4)', ha='center', va='center', 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

# Plot 2: All binary splits for Grade
ax2 = plt.subplot(2, 4, 2)
split_labels = [f"Split {i}" for i in range(1, len(binary_splits) + 1)]
gini_gains = [result['gini_gain'] for result in split_results]
colors = ['red' if gain == max(gini_gains) else 'lightblue' for gain in gini_gains]

bars = ax2.bar(split_labels, gini_gains, color=colors, alpha=0.7, edgecolor='black')
ax2.set_xlabel('Binary Split')
ax2.set_ylabel('Gini Gain')
ax2.set_title('Gini Gain for All Binary Splits')
ax2.grid(True, alpha=0.3)

for bar, gain in zip(bars, gini_gains):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{gain:.3f}', ha='center', va='bottom', fontsize=8)

# Plot 3: Grade distribution visualization
ax3 = plt.subplot(2, 4, 3)
grades = list(grade_distributions.keys())
class_0_counts = [grade_distributions[g]['class_0'] for g in grades]
class_1_counts = [grade_distributions[g]['class_1'] for g in grades]

x = np.arange(len(grades))
width = 0.35

bars1 = ax3.bar(x - width/2, class_0_counts, width, label='Class 0', color='lightcoral', alpha=0.7)
bars2 = ax3.bar(x + width/2, class_1_counts, width, label='Class 1', color='lightblue', alpha=0.7)

ax3.set_xlabel('Grade')
ax3.set_ylabel('Count')
ax3.set_title('Class Distribution by Grade')
ax3.set_xticks(x)
ax3.set_xticklabels(grades)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                f'{int(height)}', ha='center', va='bottom')

# Plot 4: Optimal split visualization
ax4 = plt.subplot(2, 4, 4)
best_result = max(split_results, key=lambda x: x['gini_gain'])

# Show the optimal split
left_grades = sorted(best_result['set1'])
right_grades = sorted(best_result['set2'])

left_data = [f"{g}\n({grade_distributions[g]['class_0']},{grade_distributions[g]['class_1']})" 
            for g in left_grades]
right_data = [f"{g}\n({grade_distributions[g]['class_0']},{grade_distributions[g]['class_1']})" 
             for g in right_grades]

# Create boxes for left and right splits
left_box = plt.Rectangle((1, 4), 3, 3, facecolor='lightgreen', alpha=0.5, edgecolor='black')
right_box = plt.Rectangle((6, 4), 3, 3, facecolor='lightcoral', alpha=0.5, edgecolor='black')
ax4.add_patch(left_box)
ax4.add_patch(right_box)

ax4.text(2.5, 5.5, f"Left Split\n{{{', '.join(left_grades)}}}", 
         ha='center', va='center', fontweight='bold')
ax4.text(7.5, 5.5, f"Right Split\n{{{', '.join(right_grades)}}}", 
         ha='center', va='center', fontweight='bold')

ax4.text(2.5, 4.5, f"Gini: {best_result['left_gini']:.3f}", ha='center', va='center')
ax4.text(7.5, 4.5, f"Gini: {best_result['right_gini']:.3f}", ha='center', va='center')

ax4.text(5, 2, f"Best Gini Gain: {best_result['gini_gain']:.4f}", 
         ha='center', va='center', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"))

ax4.set_xlim(0, 10)
ax4.set_ylim(0, 8)
ax4.set_title('Optimal Binary Split')
ax4.axis('off')

# Plot 5: CART algorithm overview
ax5 = plt.subplot(2, 4, 5)
ax5.text(0.5, 0.9, 'CART Algorithm', ha='center', va='top', fontsize=14, fontweight='bold',
         transform=ax5.transAxes)

cart_features = [
    '• Classification And Regression Trees',
    '• Binary splits only',
    '• Classification: Gini impurity',
    '• Regression: Mean Squared Error',
    '• Handles missing values',
    '• Built-in pruning',
    '• Robust to outliers',
    '• No bias toward high-cardinality'
]

ax5.text(0.05, 0.8, '\n'.join(cart_features), ha='left', va='top', fontsize=10,
         transform=ax5.transAxes,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

ax5.axis('off')

# Plot 6: Split evaluation table
ax6 = plt.subplot(2, 4, 6)
ax6.axis('tight')
ax6.axis('off')

# Create table data
table_data = []
for result in split_results:
    set1_str = '{' + ', '.join(sorted(result['set1'])) + '}'
    set2_str = '{' + ', '.join(sorted(result['set2'])) + '}'
    table_data.append([
        f"Split {result['split_num']}",
        f"{set1_str} vs {set2_str}",
        f"{result['gini_gain']:.4f}"
    ])

table = ax6.table(cellText=table_data,
                 colLabels=['Split', 'Sets', 'Gini Gain'],
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.5)

# Highlight best split
best_split_idx = max(range(len(split_results)), key=lambda i: split_results[i]['gini_gain'])
for j in range(3):
    table[(best_split_idx + 1, j)].set_facecolor('yellow')

# Header styling
for j in range(3):
    table[(0, j)].set_facecolor('#1976D2')
    table[(0, j)].set_text_props(weight='bold', color='white')

ax6.set_title('Split Evaluation Results', pad=20)

# Plot 7: Comparison with other algorithms
ax7 = plt.subplot(2, 4, 7)
algorithms = ['ID3', 'C4.5', 'CART']
capabilities = ['Multi-way\nSplits', 'Gain Ratio\nCorrection', 'Binary Splits\nOnly']
colors = ['lightblue', 'lightgreen', 'lightcoral']

bars = ax7.bar(algorithms, [1, 1, 1], color=colors, alpha=0.7, edgecolor='black')
ax7.set_title('Algorithm Splitting Strategies')
ax7.set_ylabel('Strategy')

for bar, capability in zip(bars, capabilities):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
             capability, ha='center', va='center', fontweight='bold')

ax7.set_ylim(0, 1.2)

# Plot 8: Regression capability
ax8 = plt.subplot(2, 4, 8)
problems = ['Classification', 'Regression']
id3_c45 = [1, 0]  # Can only do classification
cart_cap = [1, 1]  # Can do both

x = np.arange(len(problems))
width = 0.35

bars1 = ax8.bar(x - width/2, id3_c45, width, label='ID3/C4.5', color='lightgray', alpha=0.7)
bars2 = ax8.bar(x + width/2, cart_cap, width, label='CART', color='lightcoral', alpha=0.7)

ax8.set_xlabel('Problem Type')
ax8.set_ylabel('Capability')
ax8.set_title('Algorithm Capabilities')
ax8.set_xticks(x)
ax8.set_xticklabels(problems)
ax8.legend()
ax8.set_ylim(0, 1.2)

# Add checkmarks and X marks
for i, (v1, v2) in enumerate(zip(id3_c45, cart_cap)):
    if v1 > 0:
        ax8.text(i - width/2, v1/2, r'$\checkmark$', ha='center', va='center', fontsize=16, color='green')
    else:
        ax8.text(i - width/2, 0.1, r'$\times$', ha='center', va='center', fontsize=16, color='red')
    
    if v2 > 0:
        ax8.text(i + width/2, v2/2, r'$\checkmark$', ha='center', va='center', fontsize=16, color='green')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'cart_binary_splitting.png'), dpi=300, bbox_inches='tight')

# Create a detailed split calculation figure
fig2, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')

# Create detailed calculation text
calc_text = f"""
DETAILED GINI CALCULATION FOR OPTIMAL SPLIT

Given Grade distributions: A(3,1), B(2,2), C(1,3), D(4,0)
Total samples: {total_samples}
Baseline Gini: {baseline_gini:.4f}

Optimal Split: {{{', '.join(sorted(best_split[0]))}}} vs {{{', '.join(sorted(best_split[1]))}}}

Left side calculations:
"""

best_result = max(split_results, key=lambda x: x['gini_gain'])
left_class_0, left_class_1 = best_result['left_dist']
right_class_0, right_class_1 = best_result['right_dist']
left_total = left_class_0 + left_class_1
right_total = right_class_0 + right_class_1

calc_text += f"""  Grades: {', '.join(sorted(best_result['set1']))}
  Samples: {left_total} ({left_class_0} class 0, {left_class_1} class 1)
  P(class 0) = {left_class_0}/{left_total} = {left_class_0/left_total:.3f}
  P(class 1) = {left_class_1}/{left_total} = {left_class_1/left_total:.3f}
  Gini = 1 - ({left_class_0/left_total:.3f})² - ({left_class_1/left_total:.3f})² = {best_result['left_gini']:.4f}

Right side calculations:
  Grades: {', '.join(sorted(best_result['set2']))}
  Samples: {right_total} ({right_class_0} class 0, {right_class_1} class 1)
  P(class 0) = {right_class_0}/{right_total} = {right_class_0/right_total:.3f}
  P(class 1) = {right_class_1}/{right_total} = {right_class_1/right_total:.3f}
  Gini = 1 - ({right_class_0/right_total:.3f})² - ({right_class_1/right_total:.3f})² = {best_result['right_gini']:.4f}

Weighted Gini:
  ({left_total}/{total_samples}) × {best_result['left_gini']:.4f} + ({right_total}/{total_samples}) × {best_result['right_gini']:.4f} = {best_result['weighted_gini']:.4f}

Gini Gain:
  {baseline_gini:.4f} - {best_result['weighted_gini']:.4f} = {best_result['gini_gain']:.4f}
"""

ax.text(0.05, 0.95, calc_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))

ax.set_title('Detailed Gini Calculation for Optimal Split', fontsize=14, fontweight='bold')

plt.savefig(os.path.join(save_dir, 'detailed_gini_calculation.png'), dpi=300, bbox_inches='tight')

print(f"\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"1. Grade feature has 7 possible binary splits")
print(f"2. Formula: 2^(k-1) - 1 = 2^3 - 1 = 7 splits for k=4 values")
print(f"3. CART = Classification And Regression Trees")
print(f"4. CART can handle regression by using MSE instead of Gini")
print(f"5. Optimal split: {{{', '.join(sorted(best_split[0]))}}} vs {{{', '.join(sorted(best_split[1]))}}} with Gini gain {best_gini_gain:.4f}")

print(f"\nImages saved to: {save_dir}")
