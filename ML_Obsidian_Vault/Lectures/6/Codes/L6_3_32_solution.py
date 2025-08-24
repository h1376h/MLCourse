import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_32")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

print("=" * 80)
print("QUESTION 32: CART COST FUNCTION APPROACH")
print("=" * 80)

# Define the problem data
colors = ['Red', 'Blue', 'Green', 'Yellow']
class_distributions = {
    'Red': [2, 1],      # [Class0, Class1]
    'Blue': [1, 2],
    'Green': [3, 0],
    'Yellow': [1, 1]
}

print("\nPROBLEM DATA:")
print("-" * 40)
print("Feature: Color with values {Red, Blue, Green, Yellow}")
print("Class distributions (Class0, Class1):")
for color, dist in class_distributions.items():
    print(f"  {color}: {dist}")

# Calculate total samples
total_samples = sum(sum(dist) for dist in class_distributions.values())
print(f"\nTotal samples: {total_samples}")

# Calculate overall class distribution
overall_class0 = sum(dist[0] for dist in class_distributions.values())
overall_class1 = sum(dist[1] for dist in class_distributions.values())
overall_dist = [overall_class0, overall_class1]
print(f"Overall class distribution: {overall_dist}")

# Step 1: CART Cost Function
print("\n" + "="*60)
print("STEP 1: CART COST FUNCTION")
print("="*60)

cost_function = r"$\text{Cost}(T) = \sum_{\text{leaves}} N_t \cdot \text{Impurity}(t) + \alpha \cdot |\text{leaves}|$"
print(f"CART Cost Function: {cost_function}")

print("\nComponents:")
print("- N_t: Number of samples in leaf t")
print("- Impurity(t): Impurity measure (Gini or Entropy) in leaf t")
print("- α: Complexity penalty parameter")
print("- |leaves|: Number of leaf nodes")

# Step 2: All Possible Binary Splits
print("\n" + "="*60)
print("STEP 2: ALL POSSIBLE BINARY SPLITS")
print("="*60)

# For k=4 values, we have 2^(4-1) - 1 = 7 possible binary splits
binary_splits = [
    ['Red', 'Blue,Green,Yellow'],
    ['Blue', 'Red,Green,Yellow'],
    ['Green', 'Red,Blue,Yellow'],
    ['Yellow', 'Red,Blue,Green'],
    ['Red,Blue', 'Green,Yellow'],
    ['Red,Green', 'Blue,Yellow'],
    ['Red,Yellow', 'Blue,Green']
]

print(f"For k={len(colors)} values, we have 2^({len(colors)-1}) - 1 = {2**(len(colors)-1) - 1} possible binary splits:")
for i, split in enumerate(binary_splits, 1):
    print(f"  {i}. {split[0]} vs {split[1]}")

# Step 3: Gini Impurity Calculations
print("\n" + "="*60)
print("STEP 3: GINI IMPURITY CALCULATIONS")
print("="*60)

def gini_impurity(class_dist):
    """Calculate Gini impurity for a class distribution"""
    total = sum(class_dist)
    if total == 0:
        return 0
    probabilities = [count/total for count in class_dist]
    return 1 - sum(p**2 for p in probabilities)

def entropy_impurity(class_dist):
    """Calculate entropy for a class distribution"""
    total = sum(class_dist)
    if total == 0:
        return 0
    probabilities = [count/total for count in class_dist]
    entropy = 0
    for p in probabilities:
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy

# Calculate overall impurity
overall_gini = gini_impurity(overall_dist)
overall_entropy = entropy_impurity(overall_dist)
print(f"Overall Gini impurity: {overall_gini:.4f}")
print(f"Overall Entropy: {overall_entropy:.4f}")

# Calculate impurity for each color
print("\nIndividual color impurities:")
for color, dist in class_distributions.items():
    gini = gini_impurity(dist)
    entropy = entropy_impurity(dist)
    print(f"  {color}: Gini={gini:.4f}, Entropy={entropy:.4f}")

# Calculate weighted impurity for each binary split
print("\nBinary split analysis using Gini impurity:")
gini_results = []

for i, split in enumerate(binary_splits):
    left_colors = split[0].split(',')
    right_colors = split[1].split(',')
    
    # Aggregate distributions for left and right sides
    left_dist = [0, 0]
    right_dist = [0, 0]
    
    for color in left_colors:
        left_dist[0] += class_distributions[color][0]
        left_dist[1] += class_distributions[color][1]
    
    for color in right_colors:
        right_dist[0] += class_distributions[color][0]
        right_dist[1] += class_distributions[color][1]
    
    # Calculate weighted impurity
    left_samples = sum(left_dist)
    right_samples = sum(right_dist)
    
    left_gini = gini_impurity(left_dist)
    right_gini = gini_impurity(right_dist)
    
    weighted_gini = (left_samples * left_gini + right_samples * right_gini) / total_samples
    gini_gain = overall_gini - weighted_gini
    
    gini_results.append({
        'split': split,
        'left_dist': left_dist,
        'right_dist': right_dist,
        'left_gini': left_gini,
        'right_gini': right_gini,
        'weighted_gini': weighted_gini,
        'gini_gain': gini_gain
    })
    
    print(f"\n  Split {i+1}: {split[0]} vs {split[1]}")
    print(f"    Left: {left_dist} (Gini={left_gini:.4f})")
    print(f"    Right: {right_dist} (Gini={right_gini:.4f})")
    print(f"    Weighted Gini: {weighted_gini:.4f}")
    print(f"    Gini Gain: {gini_gain:.4f}")

# Find optimal Gini split
optimal_gini_idx = max(range(len(gini_results)), key=lambda i: gini_results[i]['gini_gain'])
optimal_gini_split = gini_results[optimal_gini_idx]
print(f"\nOptimal Gini split: {optimal_gini_split['split']}")
print(f"Gini Gain: {optimal_gini_split['gini_gain']:.4f}")

# Step 4: Entropy-based Information Gain
print("\n" + "="*60)
print("STEP 4: ENTROPY-BASED INFORMATION GAIN")
print("="*60)

print("Binary split analysis using Entropy:")
entropy_results = []

for i, split in enumerate(binary_splits):
    left_colors = split[0].split(',')
    right_colors = split[1].split(',')
    
    # Aggregate distributions for left and right sides
    left_dist = [0, 0]
    right_dist = [0, 0]
    
    for color in left_colors:
        left_dist[0] += class_distributions[color][0]
        left_dist[1] += class_distributions[color][1]
    
    for color in right_colors:
        right_dist[0] += class_distributions[color][0]
        right_dist[1] += class_distributions[color][1]
    
    # Calculate weighted entropy
    left_samples = sum(left_dist)
    right_samples = sum(right_dist)
    
    left_entropy = entropy_impurity(left_dist)
    right_entropy = entropy_impurity(right_dist)
    
    weighted_entropy = (left_samples * left_entropy + right_samples * right_entropy) / total_samples
    information_gain = overall_entropy - weighted_entropy
    
    entropy_results.append({
        'split': split,
        'left_dist': left_dist,
        'right_dist': right_dist,
        'left_entropy': left_entropy,
        'right_entropy': right_entropy,
        'weighted_entropy': weighted_entropy,
        'information_gain': information_gain
    })
    
    print(f"\n  Split {i+1}: {split[0]} vs {split[1]}")
    print(f"    Left: {left_dist} (Entropy={left_entropy:.4f})")
    print(f"    Right: {right_dist} (Entropy={right_entropy:.4f})")
    print(f"    Weighted Entropy: {weighted_entropy:.4f}")
    print(f"    Information Gain: {information_gain:.4f}")

# Find optimal entropy split
optimal_entropy_idx = max(range(len(entropy_results)), key=lambda i: entropy_results[i]['information_gain'])
optimal_entropy_split = entropy_results[optimal_entropy_idx]
print(f"\nOptimal Entropy split: {optimal_entropy_split['split']}")
print(f"Information Gain: {optimal_entropy_split['information_gain']:.4f}")

# Step 5: Comparison
print("\n" + "="*60)
print("STEP 5: COMPARISON")
print("="*60)

print(f"Gini optimal split: {optimal_gini_split['split']}")
print(f"Entropy optimal split: {optimal_entropy_split['split']}")

if optimal_gini_idx == optimal_entropy_idx:
    print("✓ The optimal splits are IDENTICAL!")
else:
    print("✗ The optimal splits are DIFFERENT!")
    print("This demonstrates that Gini impurity and entropy can lead to different optimal splits.")

# Step 6: Cost Function Analysis
print("\n" + "="*60)
print("STEP 6: COST FUNCTION ANALYSIS")
print("="*60)

print("If we use entropy instead of Gini impurity in the cost function:")
print("Cost(T) = Σ(leaves) N_t × Entropy(t) + α × |leaves|")

print("\nKey differences:")
print("1. Entropy values are generally higher than Gini impurity")
print("2. Entropy has different scaling properties")
print("3. The relative importance of the complexity penalty (α) may change")
print("4. Tree structure may differ due to different optimal splits")

# Create visualizations
# Plot 1: Class distribution visualization
fig1, ax1 = plt.subplots(figsize=(10, 6))
color_names = list(class_distributions.keys())
class0_counts = [class_distributions[color][0] for color in color_names]
class1_counts = [class_distributions[color][1] for color in color_names]

x = np.arange(len(color_names))
width = 0.35

bars1 = ax1.bar(x - width/2, class0_counts, width, label='Class 0', color='skyblue', alpha=0.7)
bars2 = ax1.bar(x + width/2, class1_counts, width, label='Class 1', color='lightcoral', alpha=0.7)

ax1.set_xlabel('Color Values')
ax1.set_ylabel('Count')
ax1.set_title('Class Distribution by Color')
ax1.set_xticks(x)
ax1.set_xticklabels(color_names)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Impurity comparison
fig2, ax2 = plt.subplots(figsize=(10, 6))
gini_values = [gini_impurity(class_distributions[color]) for color in color_names]
entropy_values = [entropy_impurity(class_distributions[color]) for color in color_names]

x = np.arange(len(color_names))
width = 0.35

bars1 = ax2.bar(x - width/2, gini_values, width, label='Gini Impurity', color='lightgreen', alpha=0.7)
bars2 = ax2.bar(x + width/2, entropy_values, width, label='Entropy', color='orange', alpha=0.7)

ax2.set_xlabel('Color Values')
ax2.set_ylabel('Impurity Value')
ax2.set_title('Gini vs Entropy by Color')
ax2.set_xticks(x)
ax2.set_xticklabels(color_names)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'impurity_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Gini gain comparison
fig3, ax3 = plt.subplots(figsize=(12, 6))
split_labels = [f"Split {i+1}" for i in range(len(binary_splits))]
gini_gains = [result['gini_gain'] for result in gini_results]
entropy_gains = [result['information_gain'] for result in entropy_results]

x = np.arange(len(split_labels))
width = 0.35

bars1 = ax3.bar(x - width/2, gini_gains, width, label='Gini Gain', color='lightgreen', alpha=0.7)
bars2 = ax3.bar(x + width/2, entropy_gains, width, label='Information Gain', color='orange', alpha=0.7)

ax3.set_xlabel('Binary Splits')
ax3.set_ylabel('Gain Value')
ax3.set_title('Gini Gain vs Information Gain')
ax3.set_xticks(x)
ax3.set_xticklabels(split_labels, rotation=45, ha='right')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Highlight optimal splits
optimal_gini_gain = max(gini_gains)
optimal_entropy_gain = max(entropy_gains)

for i, (gini_gain, entropy_gain) in enumerate(zip(gini_gains, entropy_gains)):
    if gini_gain == optimal_gini_gain:
        bars1[i].set_color('red')
        bars1[i].set_alpha(0.9)
    if entropy_gain == optimal_entropy_gain:
        bars2[i].set_color('red')
        bars2[i].set_alpha(0.9)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'gain_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Binary split visualization
fig4, ax4 = plt.subplots(figsize=(10, 8))
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)

# Draw tree structure
tree_elements = [
    {'pos': (5, 9), 'text': 'Color', 'color': 'lightblue', 'type': 'root'},
    {'pos': (2, 7), 'text': 'Red', 'color': 'red', 'type': 'leaf'},
    {'pos': (8, 7), 'text': 'Not Red', 'color': 'lightgray', 'type': 'leaf'},
    {'pos': (5, 5), 'text': 'Optimal\nSplit', 'color': 'yellow', 'type': 'highlight'}
]

# Draw nodes
for element in tree_elements:
    if element['type'] == 'root':
        rect = FancyBboxPatch((element['pos'][0]-0.5, element['pos'][1]-0.3), 1, 0.6,
                             boxstyle="round,pad=0.1", facecolor=element['color'], 
                             edgecolor='black', alpha=0.7)
        ax4.add_patch(rect)
    elif element['type'] == 'leaf':
        rect = FancyBboxPatch((element['pos'][0]-0.4, element['pos'][1]-0.2), 0.8, 0.4,
                             boxstyle="round,pad=0.05", facecolor=element['color'], 
                             edgecolor='black', alpha=0.7)
        ax4.add_patch(rect)
    else:
        rect = FancyBboxPatch((element['pos'][0]-0.6, element['pos'][1]-0.3), 1.2, 0.6,
                             boxstyle="round,pad=0.1", facecolor=element['color'], 
                             edgecolor='black', alpha=0.9)
        ax4.add_patch(rect)
    
    ax4.text(element['pos'][0], element['pos'][1], element['text'], ha='center', va='center', 
            fontsize=9, fontweight='bold')

# Draw arrows
arrows = [
    ((5, 8.7), (2, 7.3), 'Red'),
    ((5, 8.7), (8, 7.3), 'Not Red')
]

for start, end, label in arrows:
    ax4.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    mid_x, mid_y = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
    ax4.text(mid_x + 0.2, mid_y + 0.1, label, fontsize=8, color='red', fontweight='bold')

ax4.set_title('Optimal Binary Split Structure')
ax4.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'binary_split_structure.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 5: Cost function components
fig5, ax5 = plt.subplots(figsize=(10, 6))
cost_components = ['Impurity Cost', 'Complexity Penalty']
gini_cost = optimal_gini_split['weighted_gini'] * total_samples
entropy_cost = optimal_entropy_split['weighted_entropy'] * total_samples

# Example complexity penalty (α = 0.1, 2 leaves)
alpha = 0.1
complexity_penalty = alpha * 2

gini_total = gini_cost + complexity_penalty
entropy_total = entropy_cost + complexity_penalty

x = np.arange(len(cost_components))
width = 0.35

bars1 = ax5.bar(x - width/2, [gini_cost, complexity_penalty], width, 
                label='Gini-based', color='lightgreen', alpha=0.7)
bars2 = ax5.bar(x + width/2, [entropy_cost, complexity_penalty], width, 
                label='Entropy-based', color='orange', alpha=0.7)

ax5.set_xlabel('Cost Components')
ax5.set_ylabel('Cost Value')
ax5.set_title('Cost Function Components')
ax5.set_xticks(x)
ax5.set_xticklabels(cost_components)
ax5.legend()
ax5.grid(True, alpha=0.3)

# Add total cost annotations
ax5.text(0.5, gini_total + 0.1, f'Total: {gini_total:.3f}', ha='center', va='bottom', 
         fontweight='bold', color='green')
ax5.text(1.5, entropy_total + 0.1, f'Total: {entropy_total:.3f}', ha='center', va='bottom', 
         fontweight='bold', color='orange')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'cost_function_components.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 6: Summary table
fig6, ax6 = plt.subplots(figsize=(12, 6))
ax6.axis('tight')
ax6.axis('off')

summary_data = {
    'Metric': ['Optimal Split', 'Gain Value', 'Weighted Impurity', 'Cost Impact'],
    'Gini': [
        f"{optimal_gini_split['split'][0]} vs {optimal_gini_split['split'][1]}",
        f"{optimal_gini_split['gini_gain']:.4f}",
        f"{optimal_gini_split['weighted_gini']:.4f}",
        'Lower complexity penalty'
    ],
    'Entropy': [
        f"{optimal_entropy_split['split'][0]} vs {optimal_entropy_split['split'][1]}",
        f"{optimal_entropy_split['information_gain']:.4f}",
        f"{optimal_entropy_split['weighted_entropy']:.4f}",
        'Higher impurity cost'
    ]
}

df_summary = pd.DataFrame(summary_data)
table = ax6.table(cellText=df_summary.values,
                 colLabels=df_summary.columns,
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Color code based on algorithm
for j in range(1, len(df_summary.columns)):
    if j == 1:  # Gini column
        for i in range(1, len(df_summary) + 1):
            table[(i, j)].set_facecolor('lightgreen')
    else:  # Entropy column
        for i in range(1, len(df_summary) + 1):
            table[(i, j)].set_facecolor('orange')

# Header styling
for j in range(len(df_summary.columns)):
    table[(0, j)].set_facecolor('#2E8B57')
    table[(0, j)].set_text_props(weight='bold', color='white')

ax6.set_title('Gini vs Entropy Comparison Summary', pad=20)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'comparison_summary.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create detailed calculation figure
fig2, ax = plt.subplots(figsize=(14, 10))
ax.axis('tight')
ax.axis('off')

# Detailed calculation table
calculation_data = {
    'Split': [f"Split {i+1}" for i in range(len(binary_splits))],
    'Left Side': [f"{split[0]}" for split in binary_splits],
    'Right Side': [f"{split[1]}" for split in binary_splits],
    'Gini Gain': [f"{result['gini_gain']:.4f}" for result in gini_results],
    'Info Gain': [f"{result['information_gain']:.4f}" for result in entropy_results],
    'Best Gini': ['Yes' if i == optimal_gini_idx else 'No' for i in range(len(binary_splits))],
    'Best Entropy': ['Yes' if i == optimal_entropy_idx else 'No' for i in range(len(binary_splits))]
}

df_calc = pd.DataFrame(calculation_data)
table2 = ax.table(cellText=df_calc.values,
                 colLabels=df_calc.columns,
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])

table2.auto_set_font_size(False)
table2.set_fontsize(10)
table2.scale(1, 2.5)

# Color coding for best splits
for i in range(1, len(df_calc) + 1):
    if i-1 == optimal_gini_idx:
        table2[(i, 5)].set_facecolor('lightgreen')
        table2[(i, 5)].set_text_props(weight='bold')
    if i-1 == optimal_entropy_idx:
        table2[(i, 6)].set_facecolor('orange')
        table2[(i, 6)].set_text_props(weight='bold')

# Header styling
for j in range(len(df_calc.columns)):
    table2[(0, j)].set_facecolor('#1976D2')
    table2[(0, j)].set_text_props(weight='bold', color='white')

ax.set_title('Detailed Binary Split Analysis', fontsize=16, fontweight='bold', pad=20)

plt.savefig(os.path.join(save_dir, 'detailed_binary_split_analysis.png'), dpi=300, bbox_inches='tight')

print(f"\n" + "="*80)
print(f"KEY INSIGHTS AND CONCLUSIONS")
print("="*80)
print("1. CART's cost function balances impurity reduction with tree complexity")
print("2. Binary splits provide more flexibility than multi-way splits")
print("3. Gini impurity and entropy can lead to different optimal splits")
print("4. The choice of impurity measure affects both tree structure and cost")
print("5. Complexity penalty (α) helps prevent overfitting")
print("6. Binary splitting strategy avoids bias toward high-cardinality features")

print(f"\nImages saved to: {save_dir}")
