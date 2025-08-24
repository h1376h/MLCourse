import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import log2
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_10")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("=" * 80)
print("QUESTION 10: SPLITTING CRITERIA COMPARISON")
print("=" * 80)

def calculate_entropy(class_counts):
    """Calculate entropy from class counts"""
    total = sum(class_counts)
    if total == 0:
        return 0
    
    probabilities = [count / total for count in class_counts]
    entropy = -sum(p * log2(p) for p in probabilities if p > 0)
    return entropy

def calculate_gini(class_counts):
    """Calculate Gini impurity from class counts"""
    total = sum(class_counts)
    if total == 0:
        return 0
    
    probabilities = [count / total for count in class_counts]
    gini = 1 - sum(p**2 for p in probabilities)
    return gini

print("1. ENTROPY AND GINI FOR DISTRIBUTION [6, 2]")
print("=" * 50)

dist1 = [6, 2]
total1 = sum(dist1)
p1_class0 = dist1[0] / total1
p1_class1 = dist1[1] / total1

# Calculate entropy
entropy1 = calculate_entropy(dist1)
print(f"Distribution: {dist1}")
print(f"Total samples: {total1}")
print(f"P(class 0) = {dist1[0]}/{total1} = {p1_class0:.3f}")
print(f"P(class 1) = {dist1[1]}/{total1} = {p1_class1:.3f}")
print(f"\nEntropy calculation:")
print(f"H(S) = -sum p_i log_2(p_i)")
print(f"H(S) = -({p1_class0:.3f} x log_2({p1_class0:.3f}) + {p1_class1:.3f} x log_2({p1_class1:.3f}))")
print(f"H(S) = -({p1_class0:.3f} x {log2(p1_class0):.3f} + {p1_class1:.3f} x {log2(p1_class1):.3f})")
print(f"H(S) = -({p1_class0 * log2(p1_class0):.3f} + {p1_class1 * log2(p1_class1):.3f})")
print(f"H(S) = {entropy1:.4f}")

# Calculate Gini
gini1 = calculate_gini(dist1)
print(f"\nGini impurity calculation:")
print(f"Gini(S) = 1 - sum p_i^2")
print(f"Gini(S) = 1 - ({p1_class0:.3f}² + {p1_class1:.3f}²)")
print(f"Gini(S) = 1 - ({p1_class0**2:.3f} + {p1_class1**2:.3f})")
print(f"Gini(S) = 1 - {p1_class0**2 + p1_class1**2:.3f}")
print(f"Gini(S) = {gini1:.4f}")

print(f"\n2. ENTROPY AND GINI FOR DISTRIBUTION [4, 4]")
print("=" * 50)

dist2 = [4, 4]
total2 = sum(dist2)
p2_class0 = dist2[0] / total2
p2_class1 = dist2[1] / total2

# Calculate entropy
entropy2 = calculate_entropy(dist2)
print(f"Distribution: {dist2}")
print(f"Total samples: {total2}")
print(f"P(class 0) = {dist2[0]}/{total2} = {p2_class0:.3f}")
print(f"P(class 1) = {dist2[1]}/{total2} = {p2_class1:.3f}")
print(f"\nEntropy calculation:")
print(f"H(S) = -({p2_class0:.3f} x log_2({p2_class0:.3f}) + {p2_class1:.3f} x log_2({p2_class1:.3f}))")
print(f"H(S) = -({p2_class0:.3f} x {log2(p2_class0):.3f} + {p2_class1:.3f} x {log2(p2_class1):.3f})")
print(f"H(S) = -({p2_class0 * log2(p2_class0):.3f} + {p2_class1 * log2(p2_class1):.3f})")
print(f"H(S) = {entropy2:.4f}")

# Calculate Gini
gini2 = calculate_gini(dist2)
print(f"\nGini impurity calculation:")
print(f"Gini(S) = 1 - ({p2_class0:.3f}² + {p2_class1:.3f}²)")
print(f"Gini(S) = 1 - ({p2_class0**2:.3f} + {p2_class1**2:.3f})")
print(f"Gini(S) = 1 - {p2_class0**2 + p2_class1**2:.3f}")
print(f"Gini(S) = {gini2:.4f}")

print(f"\n3. MAXIMUM VALUES FOR BALANCED DISTRIBUTIONS")
print("=" * 50)

# Theoretical maximum values
max_entropy_2class = 1.0  # log_2(2) = 1 for balanced binary distribution
max_gini_2class = 0.5     # 1 - 2*(0.5)² = 0.5 for balanced binary distribution

print(f"For binary classification:")
print(f"Maximum entropy = log_2(2) = 1.0 (achieved when P(class 0) = P(class 1) = 0.5)")
print(f"Maximum Gini = 0.5 (achieved when P(class 0) = P(class 1) = 0.5)")
print(f"\nOur [4, 4] distribution has equal probabilities (0.5, 0.5):")
print(f"Entropy = {entropy2:.4f} ≈ {max_entropy_2class:.1f} (maximum)")
print(f"Gini = {gini2:.4f} = {max_gini_2class:.1f} (maximum)")
print(f"\nBoth measures reach their maximum for perfectly balanced distributions!")

# Generate data for visualization
print(f"\n4. COMPARATIVE ANALYSIS")
print("=" * 50)

# Create range of probability distributions
p_values = np.linspace(0.01, 0.99, 100)
entropies = []
ginis = []

for p in p_values:
    # Binary distribution with probabilities [p, 1-p]
    dist = [p, 1-p]
    entropy = -p * log2(p) - (1-p) * log2(1-p)
    gini = 1 - (p**2 + (1-p)**2)
    
    entropies.append(entropy)
    ginis.append(gini)

# Find correlation
correlation = np.corrcoef(entropies, ginis)[0, 1]
print(f"Correlation between entropy and Gini impurity: {correlation:.4f}")

# Practical differences
print(f"\nPractical differences in decision tree construction:")
print(f"• Both measures typically lead to very similar trees")
print(f"• Gini is computationally faster (no logarithm)")
print(f"• Entropy has stronger theoretical foundation in information theory")
print(f"• Gini is slightly more robust to outliers")
print(f"• In practice, the choice rarely matters significantly")

# Generate some example distributions to compare
example_distributions = [
    [10, 0],   # Pure
    [9, 1],    # Mostly pure
    [7, 3],    # Imbalanced
    [6, 4],    # Slightly imbalanced
    [5, 5],    # Balanced
]

print(f"\nComparison across different distributions:")
print(f"{'Distribution':<12} {'Entropy':<8} {'Gini':<8} {'Difference':<10}")
print("-" * 45)

for dist in example_distributions:
    ent = calculate_entropy(dist)
    gin = calculate_gini(dist)
    diff = abs(ent - gin)
    print(f"{str(dist):<12} {ent:<8.4f} {gin:<8.4f} {diff:<10.4f}")

# Create visualizations
fig = plt.figure(figsize=(20, 12))

# Plot 1: Entropy vs probability
ax1 = plt.subplot(2, 4, 1)
ax1.plot(p_values, entropies, 'b-', linewidth=2, label='Entropy')
ax1.set_xlabel('P(Class 0)')
ax1.set_ylabel('Entropy')
ax1.set_title('Entropy vs Class Probability')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Maximum (1.0)')
ax1.axvline(x=0.5, color='r', linestyle='--', alpha=0.7)
ax1.legend()

# Mark our specific points
ax1.plot(p1_class0, entropy1, 'ro', markersize=10, label=f'[6,2]: {entropy1:.3f}')
ax1.plot(p2_class0, entropy2, 'go', markersize=10, label=f'[4,4]: {entropy2:.3f}')
ax1.legend()

# Plot 2: Gini vs probability
ax2 = plt.subplot(2, 4, 2)
ax2.plot(p_values, ginis, 'g-', linewidth=2, label='Gini Impurity')
ax2.set_xlabel('P(Class 0)')
ax2.set_ylabel('Gini Impurity')
ax2.set_title('Gini Impurity vs Class Probability')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Maximum (0.5)')
ax2.axvline(x=0.5, color='r', linestyle='--', alpha=0.7)
ax2.legend()

# Mark our specific points
ax2.plot(p1_class0, gini1, 'ro', markersize=10, label=f'[6,2]: {gini1:.3f}')
ax2.plot(p2_class0, gini2, 'go', markersize=10, label=f'[4,4]: {gini2:.3f}')
ax2.legend()

# Plot 3: Both measures together
ax3 = plt.subplot(2, 4, 3)
ax3.plot(p_values, entropies, 'b-', linewidth=2, label='Entropy')
ax3.plot(p_values, ginis, 'g-', linewidth=2, label='Gini Impurity')
ax3.set_xlabel('P(Class 0)')
ax3.set_ylabel('Impurity Measure')
ax3.set_title('Entropy vs Gini Impurity')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Plot 4: Difference between measures
ax4 = plt.subplot(2, 4, 4)
differences = [abs(e - g) for e, g in zip(entropies, ginis)]
ax4.plot(p_values, differences, 'r-', linewidth=2)
ax4.set_xlabel('P(Class 0)')
ax4.set_ylabel('|Entropy - Gini|')
ax4.set_title('Absolute Difference')
ax4.grid(True, alpha=0.3)

# Plot 5: Comparison bar chart for specific distributions
ax5 = plt.subplot(2, 4, 5)
dist_labels = [str(d) for d in example_distributions]
entropy_values = [calculate_entropy(d) for d in example_distributions]
gini_values = [calculate_gini(d) for d in example_distributions]

x = np.arange(len(dist_labels))
width = 0.35

bars1 = ax5.bar(x - width/2, entropy_values, width, label='Entropy', color='skyblue', alpha=0.7)
bars2 = ax5.bar(x + width/2, gini_values, width, label='Gini', color='lightgreen', alpha=0.7)

ax5.set_xlabel('Distribution')
ax5.set_ylabel('Impurity Value')
ax5.set_title('Entropy vs Gini for Different Distributions')
ax5.set_xticks(x)
ax5.set_xticklabels(dist_labels, rotation=45)
ax5.legend()
ax5.grid(True, alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# Plot 6: Correlation scatter plot
ax6 = plt.subplot(2, 4, 6)
ax6.scatter(entropies, ginis, alpha=0.6, s=20)
ax6.set_xlabel('Entropy')
ax6.set_ylabel('Gini Impurity')
ax6.set_title(f'Entropy vs Gini\n(Correlation: {correlation:.4f})')
ax6.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(entropies, ginis, 1)
p = np.poly1d(z)
ax6.plot(entropies, p(entropies), "r--", alpha=0.8)

# Plot 7: Algorithm usage comparison
ax7 = plt.subplot(2, 4, 7)
algorithms = ['ID3', 'C4.5', 'CART']
criteria = ['Entropy', 'Entropy', 'Gini']
colors = ['skyblue', 'lightgreen', 'lightcoral']

bars = ax7.bar(algorithms, [1, 1, 1], color=colors, alpha=0.7, edgecolor='black')
ax7.set_title('Splitting Criteria by Algorithm')
ax7.set_ylabel('Usage')

for bar, criterion in zip(bars, criteria):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
             criterion, ha='center', va='center', fontweight='bold')

# Plot 8: Computational comparison
ax8 = plt.subplot(2, 4, 8)
operations = ['Logarithm', 'Square', 'Summation']
entropy_ops = [2, 0, 3]  # 2 log operations, 0 squares, 3 summations
gini_ops = [0, 2, 3]     # 0 log operations, 2 squares, 3 summations

x = np.arange(len(operations))
width = 0.35

bars1 = ax8.bar(x - width/2, entropy_ops, width, label='Entropy', color='skyblue', alpha=0.7)
bars2 = ax8.bar(x + width/2, gini_ops, width, label='Gini', color='lightgreen', alpha=0.7)

ax8.set_xlabel('Operation Type')
ax8.set_ylabel('Count per Calculation')
ax8.set_title('Computational Complexity')
ax8.set_xticks(x)
ax8.set_xticklabels(operations)
ax8.legend()
ax8.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'entropy_vs_gini_comparison.png'), dpi=300, bbox_inches='tight')

# Create detailed calculation figure
fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Detailed calculation for [6,2]
ax1.axis('off')
calc_text_1 = f"""
Distribution [6, 2] - Detailed Calculations

Total samples: 8
P(class 0) = 6/8 = 0.750
P(class 1) = 2/8 = 0.250

ENTROPY:
H(S) = -sum p_i log_2(p_i)
H(S) = -(0.750 x log_2(0.750) + 0.250 x log_2(0.250))
H(S) = -(0.750 x (-0.415) + 0.250 x (-2.000))
H(S) = -(-0.311 + -0.500)
H(S) = 0.811

GINI:
Gini(S) = 1 - sum p_i^2
Gini(S) = 1 - (0.750^2 + 0.250^2)
Gini(S) = 1 - (0.563 + 0.063)
Gini(S) = 1 - 0.625
Gini(S) = 0.375
"""

ax1.text(0.05, 0.95, calc_text_1, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
ax1.set_title('Distribution [6, 2] Calculations', fontsize=12, fontweight='bold')

# Detailed calculation for [4,4]
ax2.axis('off')
calc_text_2 = f"""
Distribution [4, 4] - Detailed Calculations

Total samples: 8
P(class 0) = 4/8 = 0.500
P(class 1) = 4/8 = 0.500

ENTROPY:
H(S) = -sum p_i log_2(p_i)
H(S) = -(0.500 x log_2(0.500) + 0.500 x log_2(0.500))
H(S) = -(0.500 x (-1.000) + 0.500 x (-1.000))
H(S) = -(-0.500 + -0.500)
H(S) = 1.000

GINI:
Gini(S) = 1 - sum p_i^2
Gini(S) = 1 - (0.500^2 + 0.500^2)
Gini(S) = 1 - (0.250 + 0.250)
Gini(S) = 1 - 0.500
Gini(S) = 0.500

Maximum values achieved!
"""

ax2.text(0.05, 0.95, calc_text_2, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
ax2.set_title('Distribution [4, 4] Calculations', fontsize=12, fontweight='bold')

# Comparison table
ax3.axis('tight')
ax3.axis('off')

comparison_data = {
    'Measure': ['Entropy', 'Gini Impurity'],
    'Formula': ['$-\\sum p_i \\log_2(p_i)$', '$1 - \\sum p_i^2$'],
    '[6,2] Value': [f'{entropy1:.4f}', f'{gini1:.4f}'],
    '[4,4] Value': [f'{entropy2:.4f}', f'{gini2:.4f}'],
    'Max Value': ['1.000', '0.500'],
    'Achieved at': ['Balanced dist.', 'Balanced dist.']
}

df = pd.DataFrame(comparison_data)
table = ax3.table(cellText=df.values,
                 colLabels=df.columns,
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color coding
for j in range(len(df.columns)):
    table[(0, j)].set_facecolor('#2E8B57')
    table[(0, j)].set_text_props(weight='bold', color='white')

# Highlight maximum values
for i in range(1, 3):
    table[(i, 4)].set_facecolor('yellow')
    table[(i, 5)].set_facecolor('yellow')

ax3.set_title('Comprehensive Comparison', fontsize=12, fontweight='bold', pad=20)

# Practical implications
ax4.axis('off')
practical_text = """
PRACTICAL IMPLICATIONS

1. Maximum Values:
   • Both reach maximum for balanced distributions
   • Entropy max = 1.0, Gini max = 0.5
   • Both minimize for pure distributions

2. Computational Efficiency:
   • Gini: Only requires squaring and addition
   • Entropy: Requires logarithm computation
   • Gini is ~2-3x faster in practice

3. Tree Construction:
   • Usually lead to identical or very similar trees
   • Differences are typically minor
   • Choice often based on implementation preference

4. Algorithm Usage:
   • ID3, C4.5: Use entropy (information gain)
   • CART: Uses Gini impurity
   • Both are theoretically sound

5. When to Choose:
   • Gini: When speed is critical
   • Entropy: When theoretical foundation matters
   • In practice: Either works well
"""

ax4.text(0.05, 0.95, practical_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
ax4.set_title('Practical Decision Making', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'detailed_comparison_analysis.png'), dpi=300, bbox_inches='tight')

print(f"\n" + "="*80)
print("SUMMARY AND CONCLUSIONS")
print("="*80)
print(f"1. Distribution [6, 2]: Entropy = {entropy1:.4f}, Gini = {gini1:.4f}")
print(f"2. Distribution [4, 4]: Entropy = {entropy2:.4f}, Gini = {gini2:.4f}")
print(f"3. Both entropy and Gini reach maximum for balanced distributions")
print(f"4. Correlation between measures: {correlation:.4f} (very high)")
print(f"5. In practice, both usually lead to similar decision trees")
print(f"6. Gini is computationally faster, entropy has theoretical foundation")

print(f"\nImages saved to: {save_dir}")
