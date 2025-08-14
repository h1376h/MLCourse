import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_24")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Understanding the Problem
print_step_header(1, "Understanding the Problem")

print("Question 24: ID3 Algorithm Implementation and Execution Tracing")
print("We need to write complete pseudocode for the ID3 algorithm and trace its execution")
print("on a given dataset to understand how it builds decision trees.")
print()
print("Tasks:")
print("1. Write complete pseudocode for ID3 including stopping criteria")
print("2. For the given dataset, trace the first split decision")
print("3. Calculate information gain for both features")
print("4. Show which feature ID3 would select and why")
print()

# Step 2: Given Dataset Analysis
print_step_header(2, "Given Dataset Analysis")

print("Dataset:")
print("| Feature1 | Feature2 | Class |")
print("|----------|----------|-------|")
print("| A        | X        | +     |")
print("| A        | Y        | +     |")
print("| B        | X        | -     |")
print("| B        | Y        | -     |")
print()
print("Dataset Summary:")
print("- Total samples: 4")
print("- Feature1: A (2 samples), B (2 samples)")
print("- Feature2: X (2 samples), Y (2 samples)")
print("- Class: + (2 samples), - (2 samples)")
print()

# Step 3: ID3 Algorithm Pseudocode
print_step_header(3, "ID3 Algorithm Pseudocode")

print("ID3(D, A, T)")
print("Input:")
print("  D: Dataset")
print("  A: Set of attributes")
print("  T: Target attribute")
print("Output: Decision tree")
print()
print("1. Create a root node")
print("2. If all examples in D belong to the same class C:")
print("     Return a leaf node labeled with class C")
print("3. If A is empty:")
print("     Return a leaf node labeled with the majority class in D")
print("4. Select the attribute a from A with the highest information gain")
print("5. For each possible value v of attribute a:")
print("     Create a branch from the root node for the test a = v")
print("     Let Dv be the subset of examples in D where a = v")
print("     If Dv is empty:")
print("         Attach a leaf node labeled with the majority class in D")
print("     Else:")
print("         Attach the subtree ID3(Dv, A - {a}, T)")
print("6. Return the tree")
print()

print("Key Stopping Criteria:")
print("1. All examples belong to the same class (pure node)")
print("2. No more attributes available for splitting")
print("3. Empty subset after splitting")
print()

# Step 4: Understanding Information Gain
print_step_header(4, "Understanding Information Gain")

print("Information Gain (IG) is the key metric used by ID3 to select the best feature:")
print()
print("IG(S, A) = H(S) - H(S|A)")
print("where:")
print("- H(S) is the entropy of the dataset")
print("- H(S|A) is the conditional entropy given feature A")
print("- Higher IG means better feature for splitting")
print()
print("Entropy formula:")
print("H(S) = -Σ p(i) * log2(p(i))")
print("where p(i) is the probability of class i")
print()

# Step 5: Calculate Entropy of the Dataset
print_step_header(5, "Calculate Entropy of the Dataset")

print("Step 1: Calculate H(S) - Entropy of the entire dataset")
print("H(S) = -Σ p(i) * log2(p(i))")
print()

# Count target classes
total_samples = 4
positive_count = 2
negative_count = 2

p_positive = positive_count / total_samples
p_negative = negative_count / total_samples

print(f"P(+) = {positive_count}/{total_samples} = {p_positive}")
print(f"P(-) = {negative_count}/{total_samples} = {p_negative}")
print()

# Calculate entropy
import math
H_S = -p_positive * math.log2(p_positive) - p_negative * math.log2(p_negative)
print(f"H(S) = -{p_positive} * log2({p_positive}) - {p_negative} * log2({p_negative})")
print(f"H(S) = -{p_positive} * {math.log2(p_positive):.3f} - {p_negative} * {math.log2(p_negative):.3f}")
print(f"H(S) = {H_S:.3f}")
print()

# Step 6: Calculate Information Gain for Feature1
print_step_header(6, "Calculate Information Gain for Feature1")

print("Step 2: Calculate Information Gain for Feature1")
print("IG(S, Feature1) = H(S) - H(S|Feature1)")
print()

print("Feature1 has values: A (2 samples), B (2 samples)")
print()

# For Feature1 = A: 2 samples, both positive
print("For Feature1 = A:")
print("  - 2 samples, both positive")
print("  - P(+|A) = 2/2 = 1.0")
print("  - P(-|A) = 0/2 = 0.0")
print("  - H(S|A) = -1.0 * log2(1.0) - 0.0 * log2(0.0) = 0")
print()

# For Feature1 = B: 2 samples, both negative
print("For Feature1 = B:")
print("  - 2 samples, both negative")
print("  - P(+|B) = 0/2 = 0.0")
print("  - P(-|B) = 2/2 = 1.0")
print("  - H(S|B) = -0.0 * log2(0.0) - 1.0 * log2(1.0) = 0")
print()

# Calculate conditional entropy
H_S_given_Feature1 = (2/4) * 0 + (2/4) * 0  # Weighted average
print(f"H(S|Feature1) = (2/4) * 0 + (2/4) * 0 = {H_S_given_Feature1:.3f}")
print()

# Calculate information gain
IG_Feature1 = H_S - H_S_given_Feature1
print(f"IG(S, Feature1) = {H_S:.3f} - {H_S_given_Feature1:.3f} = {IG_Feature1:.3f}")
print()

# Step 7: Calculate Information Gain for Feature2
print_step_header(7, "Calculate Information Gain for Feature2")

print("Step 3: Calculate Information Gain for Feature2")
print("IG(S, Feature2) = H(S) - H(S|Feature2)")
print()

print("Feature2 has values: X (2 samples), Y (2 samples)")
print()

# For Feature2 = X: 2 samples, 1 positive, 1 negative
print("For Feature2 = X:")
print("  - 2 samples: 1 positive, 1 negative")
print("  - P(+|X) = 1/2 = 0.5")
print("  - P(-|X) = 1/2 = 0.5")
print("  - H(S|X) = -0.5 * log2(0.5) - 0.5 * log2(0.5) = 1.0")
print()

# For Feature2 = Y: 2 samples, 1 positive, 1 negative
print("For Feature2 = Y:")
print("  - 2 samples: 1 positive, 1 negative")
print("  - P(+|Y) = 1/2 = 0.5")
print("  - P(-|Y) = 1/2 = 0.5")
print("  - H(S|Y) = -0.5 * log2(0.5) - 0.5 * log2(0.5) = 1.0")
print()

# Calculate conditional entropy
H_S_given_Feature2 = (2/4) * 1.0 + (2/4) * 1.0
print(f"H(S|Feature2) = (2/4) * 1.0 + (2/4) * 1.0 = {H_S_given_Feature2:.3f}")
print()

# Calculate information gain
IG_Feature2 = H_S - H_S_given_Feature2
print(f"IG(S, Feature2) = {H_S:.3f} - {H_S_given_Feature2:.3f} = {IG_Feature2:.3f}")
print()

# Step 8: Feature Selection Decision
print_step_header(8, "Feature Selection Decision")

print("Step 4: Compare Information Gain values and select the best feature")
print()

print("Information Gain Results:")
print("-" * 50)
print(f"{'Feature':<15} {'Information Gain':<15}")
print("-" * 50)
print(f"{'Feature1':<15} {IG_Feature1:<15.3f}")
print(f"{'Feature2':<15} {IG_Feature2:<15.3f}")
print("-" * 50)
print()

print("Decision:")
print(f"Feature1 has higher information gain ({IG_Feature1:.3f} > {IG_Feature2:.3f})")
print("Therefore, ID3 will select Feature1 as the root node for splitting.")
print()

# Step 9: Tree Construction After First Split
print_step_header(9, "Tree Construction After First Split")

print("After selecting Feature1 as the root node, the tree structure becomes:")
print()
print("Root: Feature1")
print("├── A → [Class: +, +] (2 samples)")
print("└── B → [Class: -, -] (2 samples)")
print()
print("Analysis:")
print("- Both branches lead to pure nodes (all samples have the same class)")
print("- No further splitting is needed")
print("- The tree is complete after the first split")
print()

# Step 10: Why Feature1 is Better
print_step_header(10, "Why Feature1 is Better Than Feature2")

print("Feature1 provides perfect separation:")
print("- A values → all positive (100% pure)")
print("- B values → all negative (100% pure)")
print("- Information gain = 1.000 (maximum possible)")
print()
print("Feature2 provides no separation:")
print("- X values → 50% positive, 50% negative (impure)")
print("- Y values → 50% positive, 50% negative (impure)")
print("- Information gain = 0.000 (no improvement)")
print()
print("Feature1 creates a perfect decision tree with just one split!")
print()

# Step 11: Visualizing the Decision Tree
print_step_header(11, "Visualizing the Decision Tree")

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Draw the decision tree structure
def draw_node(x, y, text, color='lightblue', size=0.8):
    box = FancyBboxPatch(
        (x - size/2, y - size/2), size, size,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

def draw_arrow(start, end, label='', color='black'):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=2))
    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y, label, ha='center', va='center', 
                fontsize=8, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.8))

# Root node
draw_node(5, 9, 'Feature1\n(IG=1.000)', 'lightgreen', 1.2)

# Branches
draw_arrow((5, 8.4), (3, 7.5))
draw_arrow((5, 8.4), (7, 7.5))

# Leaf nodes
draw_node(3, 7, 'A\n[+, +]\n(2 samples)', 'lightcoral')
draw_node(7, 7, 'B\n[-, -]\n(2 samples)', 'lightblue')

# Add branch labels
ax.text(4, 8, 'A', ha='center', va='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", alpha=0.8))
ax.text(6, 8, 'B', ha='center', va='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", alpha=0.8))

# Add title and explanation
ax.text(5, 5, 'Perfect Decision Tree: Feature1 provides complete separation', 
        ha='center', va='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", ec="black", alpha=0.8))

ax.text(5, 4, 'Both branches lead to pure nodes - no further splitting needed', 
        ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="black", alpha=0.8))

plt.savefig(os.path.join(save_dir, 'id3_decision_tree.png'), dpi=300, bbox_inches='tight')

# Step 12: Information Gain Calculation Visualization
print_step_header(12, "Information Gain Calculation Visualization")

# Create separate plots for each aspect of information gain calculation

# Plot 1: Dataset visualization
fig, ax1 = plt.subplots(figsize=(8, 6))
ax1.set_xlabel('Feature1', fontsize=12)
ax1.set_ylabel('Feature2', fontsize=12)
ax1.set_title('Dataset Visualization', fontsize=14, fontweight='bold')

# Plot the data points
ax1.scatter(['A', 'A'], ['X', 'Y'], c=['red', 'red'], s=200, marker='o', label='Class +')
ax1.scatter(['B', 'B'], ['X', 'Y'], c=['blue', 'blue'], s=200, marker='s', label='Class -')

ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xlim(-0.5, 1.5)
ax1.set_ylim(-0.5, 1.5)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'dataset_visualization.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Entropy calculation
fig, ax2 = plt.subplots(figsize=(8, 6))
ax2.set_xlabel('Class', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Class Distribution (Entropy Calculation)', fontsize=14, fontweight='bold')

classes = ['+', '-']
counts = [2, 2]
colors = ['red', 'blue']

bars = ax2.bar(classes, counts, color=colors, alpha=0.7)
ax2.set_ylim(0, 3)

# Add entropy calculation
entropy_text = f'H(S) = -0.5 $\\times$ $\\log_2$(0.5) - 0.5 $\\times$ $\\log_2$(0.5) = {H_S:.3f}'
ax2.text(0.5, 2.5, entropy_text, ha='center', va='center', fontsize=10, fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="black", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'entropy_calculation.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Feature1 splitting
fig, ax3 = plt.subplots(figsize=(8, 6))
ax3.set_xlabel('Feature1 Value', fontsize=12)
ax3.set_ylabel('Count', fontsize=12)
ax3.set_title('Feature1 Splitting (Perfect Separation)', fontsize=14, fontweight='bold')

# Create subplots for A and B values
feature1_values = ['A', 'B']
a_counts = [2, 0]  # A: 2 positive, 0 negative
b_counts = [0, 2]  # B: 0 positive, 2 negative

x = np.arange(len(feature1_values))
width = 0.35

bars1 = ax3.bar(x - width/2, a_counts, width, label='Class +', color='red', alpha=0.7)
bars2 = ax3.bar(x + width/2, b_counts, width, label='Class -', color='blue', alpha=0.7)

ax3.set_xticks(x)
ax3.set_xticklabels(feature1_values)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Add information gain
ig_text = f'IG(Feature1) = {IG_Feature1:.3f}'
ax3.text(0.5, 2.5, ig_text, ha='center', va='center', fontsize=10, fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="black", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature1_splitting.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Feature2 splitting
fig, ax4 = plt.subplots(figsize=(8, 6))
ax4.set_xlabel('Feature2 Value', fontsize=12)
ax4.set_ylabel('Count', fontsize=12)
ax4.set_title('Feature2 Splitting (No Separation)', fontsize=14, fontweight='bold')

# Create subplots for X and Y values
feature2_values = ['X', 'Y']
x_counts = [1, 1]  # X: 1 positive, 1 negative
y_counts = [1, 1]  # Y: 1 positive, 1 negative

bars3 = ax4.bar(x - width/2, x_counts, width, label='Class +', color='red', alpha=0.7)
bars4 = ax4.bar(x + width/2, y_counts, width, label='Class -', color='blue', alpha=0.7)

ax4.set_xticks(x)
ax4.set_xticklabels(feature2_values)
ax4.set_ylim(0, 1.5)  # Reduce whitespace by setting appropriate y limits
ax4.legend()
ax4.grid(True, alpha=0.3)

# Add information gain
ig_text2 = f'IG(Feature2) = {IG_Feature2:.3f}'
ax4.text(0.5, 1.3, ig_text2, ha='center', va='center', fontsize=10, fontweight='bold',
          bbox=dict(boxstyle="round,pad=0.3", fc="lightcoral", ec="black", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature2_splitting.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create separate plots for each aspect of information gain calculation

# Plot 1: Dataset visualization
fig, ax1 = plt.subplots(figsize=(8, 6))
ax1.set_xlabel('Feature1', fontsize=12)
ax1.set_ylabel('Feature2', fontsize=12)
ax1.set_title('Dataset Visualization', fontsize=14, fontweight='bold')

# Plot the data points
ax1.scatter(['A', 'A'], ['X', 'Y'], c=['red', 'red'], s=200, marker='o', label='Class +')
ax1.scatter(['B', 'B'], ['X', 'Y'], c=['blue', 'blue'], s=200, marker='s', label='Class -')

ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xlim(-0.5, 1.5)
ax1.set_ylim(-0.5, 1.5)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'dataset_visualization.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Entropy calculation
fig, ax2 = plt.subplots(figsize=(8, 6))
ax2.set_xlabel('Class', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Class Distribution (Entropy Calculation)', fontsize=14, fontweight='bold')

classes = ['+', '-']
counts = [2, 2]
colors = ['red', 'blue']

bars = ax2.bar(classes, counts, color=colors, alpha=0.7)
ax2.set_ylim(0, 3)

# Add entropy calculation
entropy_text = f'H(S) = -0.5 $\\times$ $\\log_2$(0.5) - 0.5 $\\times$ $\\log_2$(0.5) = {H_S:.3f}'
ax2.text(0.5, 2.5, entropy_text, ha='center', va='center', fontsize=10, fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="black", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'entropy_calculation.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Feature1 splitting
fig, ax3 = plt.subplots(figsize=(8, 6))
ax3.set_xlabel('Feature1 Value', fontsize=12)
ax3.set_ylabel('Count', fontsize=12)
ax3.set_title('Feature1 Splitting (Perfect Separation)', fontsize=14, fontweight='bold')

# Create subplots for A and B values
feature1_values = ['A', 'B']
a_counts = [2, 0]  # A: 2 positive, 0 negative
b_counts = [0, 2]  # B: 0 positive, 2 negative

x = np.arange(len(feature1_values))
width = 0.35

bars1 = ax3.bar(x - width/2, a_counts, width, label='Class +', color='red', alpha=0.7)
bars2 = ax3.bar(x + width/2, b_counts, width, label='Class -', color='blue', alpha=0.7)

ax3.set_xticks(x)
ax3.set_xticklabels(feature1_values)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Add information gain
ig_text = f'IG(Feature1) = {IG_Feature1:.3f}'
ax3.text(0.5, 2.5, ig_text, ha='center', va='center', fontsize=10, fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="black", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature1_splitting.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Feature2 splitting
fig, ax4 = plt.subplots(figsize=(8, 6))
ax4.set_xlabel('Feature2 Value', fontsize=12)
ax4.set_ylabel('Count', fontsize=12)
ax4.set_title('Feature2 Splitting (No Separation)', fontsize=14, fontweight='bold')

# Create subplots for X and Y values
feature2_values = ['X', 'Y']
x_counts = [1, 1]  # X: 1 positive, 1 negative
y_counts = [1, 1]  # Y: 1 positive, 1 negative

bars3 = ax4.bar(x - width/2, x_counts, width, label='Class +', color='red', alpha=0.7)
bars4 = ax4.bar(x + width/2, y_counts, width, label='Class -', color='blue', alpha=0.7)

ax4.set_xticks(x)
ax4.set_xticklabels(feature2_values)
ax4.legend()
ax4.grid(True, alpha=0.3)

# Add information gain
ig_text2 = f'IG(Feature2) = {IG_Feature2:.3f}'
ax4.text(0.5, 2.5, ig_text2, ha='center', va='center', fontsize=10, fontweight='bold',
          bbox=dict(boxstyle="round,pad=0.3", fc="lightcoral", ec="black", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature2_splitting.png'), dpi=300, bbox_inches='tight')
plt.close()

# Step 13: Key Insights
print_step_header(13, "Key Insights")

print("1. Feature Selection:")
print("   - ID3 always selects the feature with highest information gain")
print("   - Information gain measures how well a feature separates classes")
print("   - Perfect separation results in maximum information gain (1.0)")
print()
print("2. Tree Construction:")
print("   - ID3 builds trees recursively from the root")
print("   - Each split creates child nodes for each feature value")
print("   - The process stops when nodes become pure or no features remain")
print()
print("3. Algorithm Properties:")
print("   - ID3 is a greedy algorithm (makes locally optimal choices)")
print("   - It may not find the globally optimal tree")
print("   - Information gain favors features with many values")
print()

# Step 14: Final Answer
print_step_header(14, "Final Answer")

print("1. Complete ID3 pseudocode:")
print("   - Provided above with stopping criteria")
print("   - Key steps: entropy calculation, information gain, recursive splitting")
print()
print("2. First split decision:")
print("   - Feature1 selected as root node")
print("   - Information gain: Feature1 = 1.000, Feature2 = 0.000")
print()
print("3. Information gain calculations:")
print(f"   - Feature1: IG = {IG_Feature1:.3f} (perfect separation)")
print(f"   - Feature2: IG = {IG_Feature2:.3f} (no separation)")
print()
print("4. Feature selection reason:")
print("   - Feature1 provides complete class separation")
print("   - Feature2 provides no useful information")
print("   - ID3 selects the feature with highest information gain")
print()

print(f"\nVisualizations saved to: {save_dir}")
print("The plots show the decision tree structure and information gain calculations.")
