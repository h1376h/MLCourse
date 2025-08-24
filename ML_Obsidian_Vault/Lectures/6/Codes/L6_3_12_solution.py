import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from itertools import combinations

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_12")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['text.latex.preamble'] = r'\usepackage{amssymb}'

print("=" * 80)
print("QUESTION 12: CONTINUOUS FEATURE HANDLING")
print("=" * 80)

# Given data: ages and corresponding classes
ages = np.array([22, 25, 30, 35, 40])
classes = np.array(['No', 'No', 'Yes', 'Yes', 'No'])

print("Given Dataset:")
print("Ages:", ages)
print("Classes:", classes)

# Function to calculate entropy
def entropy(y):
    """Calculate entropy of a set of labels"""
    if len(y) == 0:
        return 0
    
    unique, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    
    # Avoid log(0) by filtering out zero probabilities
    probabilities = probabilities[probabilities > 0]
    
    return -np.sum(probabilities * np.log2(probabilities))

# Function to calculate information gain
def information_gain(y, y_left, y_right):
    """Calculate information gain for a split"""
    n = len(y)
    n_left = len(y_left)
    n_right = len(y_right)
    
    if n_left == 0 or n_right == 0:
        return 0
    
    entropy_before = entropy(y)
    entropy_after = (n_left/n * entropy(y_left)) + (n_right/n * entropy(y_right))
    
    return entropy_before - entropy_after

# 1. Why can't ID3 handle continuous features directly?
print("\n1. Why can't ID3 handle continuous features directly?")
print("-" * 50)
print("ID3 can't handle continuous features directly because it was designed")
print("to work with categorical features that have discrete, finite sets of values.")
print("Continuous features have infinite possible values, making it impossible")
print("to create discrete splits without preprocessing.")

# 2. Find all candidate threshold values
print("\n2. Candidate Threshold Values:")
print("-" * 50)

# Sort the data by age
sorted_indices = np.argsort(ages)
sorted_ages = ages[sorted_indices]
sorted_classes = classes[sorted_indices]

print("Sorted data:")
for i, (age, cls) in enumerate(zip(sorted_ages, sorted_classes)):
    print(f"  {age}: {cls}")

# Candidate thresholds are midpoints between consecutive different-class values
candidates = []
for i in range(len(sorted_ages) - 1):
    # Only consider thresholds between different classes
    if sorted_classes[i] != sorted_classes[i + 1]:
        threshold = (sorted_ages[i] + sorted_ages[i + 1]) / 2
        candidates.append(threshold)

print(f"\nCandidate thresholds: {candidates}")

# Also consider all possible midpoints for completeness
all_candidates = []
for i in range(len(sorted_ages) - 1):
    threshold = (sorted_ages[i] + sorted_ages[i + 1]) / 2
    all_candidates.append(threshold)

print(f"All possible thresholds: {all_candidates}")

# 3. Calculate information gain for threshold Age ≤ 27.5
print("\n3. Information Gain for Age ≤ 27.5:")
print("-" * 50)

threshold = 27.5
left_mask = ages <= threshold
right_mask = ages > threshold

y_left = classes[left_mask]
y_right = classes[right_mask]

print(f"Threshold: Age ≤ {threshold}")
print(f"Left subset (Age < {threshold}): Ages = {ages[left_mask]}, Classes = {y_left}")
print(f"Right subset (Age ≥ {threshold}): Ages = {ages[right_mask]}, Classes = {y_right}")

# Calculate entropies
entropy_total = entropy(classes)
entropy_left = entropy(y_left)
entropy_right = entropy(y_right)
ig = information_gain(classes, y_left, y_right)

print(f"\nEntropy calculations:")
print(f"Total entropy: {entropy_total:.4f}")
print(f"Left entropy: {entropy_left:.4f}")
print(f"Right entropy: {entropy_right:.4f}")
print(f"Information Gain: {ig:.4f}")

# Show detailed calculation
n_total = len(classes)
n_left = len(y_left)
n_right = len(y_right)

print(f"\nDetailed calculation:")
print(f"IG = H(S) - (|S_left|/|S|) * H(S_left) - (|S_right|/|S|) * H(S_right)")
print(f"IG = {entropy_total:.4f} - ({n_left}/{n_total}) * {entropy_left:.4f} - ({n_right}/{n_total}) * {entropy_right:.4f}")
print(f"IG = {entropy_total:.4f} - {n_left/n_total:.3f} * {entropy_left:.4f} - {n_right/n_total:.3f} * {entropy_right:.4f}")
print(f"IG = {entropy_total:.4f} - {(n_left/n_total) * entropy_left:.4f} - {(n_right/n_total) * entropy_right:.4f}")
print(f"IG = {ig:.4f}")

# 4. C4.5's approach vs manual discretization
print("\n4. C4.5's Approach vs Manual Discretization:")
print("-" * 50)
print("C4.5's approach:")
print("- Automatically finds optimal thresholds by evaluating all candidate splits")
print("- Uses information gain to select the best threshold")
print("- Dynamic and data-driven threshold selection")
print("- No prior knowledge needed about feature distribution")
print("\nManual discretization:")
print("- Requires domain knowledge to choose appropriate bins/thresholds")
print("- Fixed thresholds regardless of class distribution")
print("- May lose important information or create suboptimal splits")
print("- Static approach that doesn't adapt to the data")

# 5. Find optimal threshold
print("\n5. Finding Optimal Threshold:")
print("-" * 50)

# Calculate information gain for all candidate thresholds
results = []
for threshold in all_candidates:
    left_mask = ages <= threshold
    right_mask = ages > threshold
    
    y_left = classes[left_mask]
    y_right = classes[right_mask]
    
    if len(y_left) > 0 and len(y_right) > 0:  # Valid split
        ig = information_gain(classes, y_left, y_right)
        results.append((threshold, ig, y_left, y_right))

# Sort by information gain
results.sort(key=lambda x: x[1], reverse=True)

print("Information Gain for all thresholds:")
for threshold, ig, y_left, y_right in results:
    print(f"Threshold ≤ {threshold:4.1f}: IG = {ig:.4f}, Left: {y_left}, Right: {y_right}")

optimal_threshold, optimal_ig, _, _ = results[0]
print(f"\nOptimal threshold: Age ≤ {optimal_threshold}")
print(f"Maximum Information Gain: {optimal_ig:.4f}")

# Create separate focused visualizations

# Plot 1: Original data visualization
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['red' if c == 'No' else 'green' for c in classes]
ax.scatter(ages, [1]*len(ages), c=colors, s=100, alpha=0.7, edgecolors='black')
for i, (age, cls) in enumerate(zip(ages, classes)):
    ax.annotate(f'{age}\n({cls})', (age, 1), xytext=(0, 20), 
                textcoords='offset points', ha='center', fontsize=10)

ax.set_xlabel('Age')
ax.set_title('Original Dataset')
ax.set_ylim(0.5, 1.5)
ax.set_yticks([])
ax.grid(True, alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='red', label='No'), 
                   Patch(facecolor='green', label='Yes')]
ax.legend(handles=legend_elements, loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'data_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Split visualization for Age ≤ 27.5
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(ages[left_mask], [1]*sum(left_mask), c='blue', s=100, alpha=0.7, 
           label=f'Age $\\leq$ {threshold}', edgecolors='black')
ax.scatter(ages[right_mask], [1]*sum(right_mask), c='orange', s=100, alpha=0.7, 
           label=f'Age $>$ {threshold}', edgecolors='black')

for i, (age, cls) in enumerate(zip(ages, classes)):
    ax.annotate(f'{age}\n({cls})', (age, 1), xytext=(0, 20), 
                textcoords='offset points', ha='center', fontsize=10)

ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
ax.set_xlabel('Age')
ax.set_title(f'Split at Age $\\leq$ {threshold}')
ax.set_ylim(0.5, 1.5)
ax.set_yticks([])
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'split_visualization.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Information gain for all thresholds
fig, ax = plt.subplots(figsize=(10, 6))
thresholds = [r[0] for r in results]
gains = [r[1] for r in results]

bars = ax.bar(range(len(thresholds)), gains, alpha=0.7, color='skyblue', edgecolor='black')
ax.set_xlabel('Threshold Index')
ax.set_ylabel('Information Gain')
ax.set_title('Information Gain for Different Thresholds')
ax.set_xticks(range(len(thresholds)))
ax.set_xticklabels([f'{t:.1f}' for t in thresholds], rotation=45)
ax.grid(True, alpha=0.3)

# Highlight optimal threshold
optimal_idx = gains.index(max(gains))
ax.bar(optimal_idx, gains[optimal_idx], color='red', alpha=0.8, 
        label=f'Optimal: {optimal_threshold:.1f}')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'information_gain_thresholds.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Entropy breakdown
fig, ax = plt.subplots(figsize=(10, 6))
entropy_data = {
    'Total': entropy_total,
    f'Left ($\\leq${threshold})': entropy_left,
    f'Right ($>${threshold})': entropy_right
}

bars = ax.bar(entropy_data.keys(), entropy_data.values(), 
               color=['gray', 'blue', 'orange'], alpha=0.7, edgecolor='black')
ax.set_ylabel('Entropy')
ax.set_title(f'Entropy Breakdown for Threshold {threshold}')
ax.grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, entropy_data.values()):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{value:.3f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'entropy_breakdown.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create decision tree visualization showing the split
fig, ax = plt.subplots(figsize=(12, 8))

# Draw the tree structure
from matplotlib.patches import Rectangle, FancyBboxPatch

# Root node
root_box = FancyBboxPatch((0.4, 0.8), 0.2, 0.1, 
                         boxstyle="round,pad=0.01", 
                         facecolor='lightblue', edgecolor='black')
ax.add_patch(root_box)
ax.text(0.5, 0.85, f'Age $\\leq$ {optimal_threshold}?', ha='center', va='center', fontsize=12, weight='bold')

# Left child (Age < optimal_threshold)
left_mask_opt = ages < optimal_threshold
left_classes_opt = classes[left_mask_opt]
unique_classes, counts = np.unique(left_classes_opt, return_counts=True)
left_counts = {str(cls): int(count) for cls, count in zip(unique_classes, counts)}

left_box = FancyBboxPatch((0.1, 0.5), 0.2, 0.1,
                         boxstyle="round,pad=0.01",
                         facecolor='lightgreen', edgecolor='black')
ax.add_patch(left_box)
left_text = f"Age $<$ {optimal_threshold}\n"
left_text += f"Classes: {left_counts}\n"
left_text += f"Entropy: {entropy(left_classes_opt):.3f}"
ax.text(0.2, 0.55, left_text, ha='center', va='center', fontsize=10)

# Right child (Age ≥ optimal_threshold)
right_mask_opt = ages >= optimal_threshold
right_classes_opt = classes[right_mask_opt]
unique_classes, counts = np.unique(right_classes_opt, return_counts=True)
right_counts = {str(cls): int(count) for cls, count in zip(unique_classes, counts)}

right_box = FancyBboxPatch((0.7, 0.5), 0.2, 0.1,
                          boxstyle="round,pad=0.01",
                          facecolor='lightcoral', edgecolor='black')
ax.add_patch(right_box)
right_text = f"Age $\\geq$ {optimal_threshold}\n"
right_text += f"Classes: {right_counts}\n"
right_text += f"Entropy: {entropy(right_classes_opt):.3f}"
ax.text(0.8, 0.55, right_text, ha='center', va='center', fontsize=10)

# Draw edges
ax.plot([0.45, 0.25], [0.8, 0.6], 'k-', linewidth=2)
ax.plot([0.55, 0.75], [0.8, 0.6], 'k-', linewidth=2)

# Add labels on edges
ax.text(0.32, 0.72, 'Yes', ha='center', fontsize=10, weight='bold', color='green')
ax.text(0.68, 0.72, 'No', ha='center', fontsize=10, weight='bold', color='red')

# Add information gain annotation
ax.text(0.5, 0.3, f'Information Gain = {optimal_ig:.4f}', 
        ha='center', va='center', fontsize=14, weight='bold',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Optimal Decision Tree Split for Continuous Feature', fontsize=14, weight='bold')

plt.savefig(os.path.join(save_dir, 'optimal_decision_tree.png'), dpi=300, bbox_inches='tight')

# Create algorithm comparison visualization
fig, ax = plt.subplots(figsize=(16, 10))

# Create comparison table
comparison_data = {
    'Aspect': ['Feature Types', 'Threshold Selection', 'Split Evaluation', 
               'Preprocessing Required', 'Adaptability', 'Computational Cost'],
    'ID3': ['Categorical only', 'Not applicable', 'Information Gain',
            'Yes (discretization)', 'Low', 'Low'],
    'C4.5': ['Mixed types', 'Automatic optimal', 'Gain Ratio',
             'No', 'High', 'Medium'],
    'Manual Discretization': ['Categorical after preprocessing', 'Fixed/predetermined', 'Various',
                             'Yes (manual binning)', 'Low', 'Low']
}

df_comparison = pd.DataFrame(comparison_data)

# Create table visualization
table = ax.table(cellText=df_comparison.values, colLabels=df_comparison.columns,
                cellLoc='center', loc='center', cellColours=None)

table.auto_set_font_size(False)
table.set_fontsize(16)
table.scale(1.5, 3.0)

# Style the table
for i in range(len(df_comparison.columns)):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, len(df_comparison) + 1):
    for j in range(len(df_comparison.columns)):
        if j == 1:  # ID3 column
            table[(i, j)].set_facecolor('#FFE6E6')
        elif j == 2:  # C4.5 column
            table[(i, j)].set_facecolor('#E6F3FF')
        elif j == 3:  # Manual discretization column
            table[(i, j)].set_facecolor('#F0F0F0')

ax.set_title('Continuous Feature Handling: Algorithm Comparison', 
             fontsize=16, weight='bold', pad=20)
ax.axis('off')

plt.savefig(os.path.join(save_dir, 'algorithm_comparison.png'), dpi=300, bbox_inches='tight')

print(f"\nVisualization files saved to: {save_dir}")
print("Files created:")
print("- data_distribution.png")
print("- split_visualization.png")
print("- information_gain_thresholds.png")
print("- entropy_breakdown.png")
print("- optimal_decision_tree.png")
print("- algorithm_comparison.png")
