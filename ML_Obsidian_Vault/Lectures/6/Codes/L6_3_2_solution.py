import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.patches import Circle, Arrow
import matplotlib.patches as mpatches

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_2")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Understanding the Problem
print_step_header(1, "Understanding the Problem")

print("Question 2: ID3 Split Selection")
print("Consider a dataset with the following class distribution:")
print()
print("| Class | Count |")
print("|-------|-------|")
print("| Yes   | 8     |")
print("| No    | 4     |")
print()
print("Tasks:")
print("1. Calculate the entropy of this dataset")
print("2. If a feature splits this into two branches with distributions [6,2] and [2,2], calculate the information gain")
print("3. Would this be a good split according to ID3?")
print("4. What is the next step in ID3 after finding the best split?")
print()

# Step 2: Calculate Dataset Entropy
print_step_header(2, "Calculate Dataset Entropy")

# Given class distribution
total_samples = 12
yes_count = 8
no_count = 4

# Calculate probabilities
p_yes = yes_count / total_samples
p_no = no_count / total_samples

print(f"Class distribution:")
print(f"  Yes: {yes_count}/{total_samples} = {p_yes:.3f}")
print(f"  No:  {no_count}/{total_samples} = {p_no:.3f}")
print()

# Calculate entropy
def entropy(probabilities):
    """Calculate entropy given a list of probabilities."""
    h = 0
    for p in probabilities:
        if p > 0:
            h -= p * np.log2(p)
    return h

h_dataset = entropy([p_yes, p_no])
print(f"Entropy calculation:")
print(f"  H(Dataset) = -{p_yes:.3f} × log₂({p_yes:.3f}) - {p_no:.3f} × log₂({p_no:.3f})")
print(f"  H(Dataset) = {h_dataset:.4f} bits")
print()

# Visualize the entropy calculation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Class distribution
ax1.bar(['Yes', 'No'], [p_yes, p_no], color=['green', 'red'], alpha=0.7)
ax1.set_title('Class Distribution')
ax1.set_ylabel('Probability')
ax1.set_ylim(0, 1)
for i, v in enumerate([p_yes, p_no]):
    ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot 2: Entropy calculation
entropy_terms = [-p_yes * np.log2(p_yes) if p_yes > 0 else 0, 
                 -p_no * np.log2(p_no) if p_no > 0 else 0]
colors = ['green', 'red']
labels = ['Yes', 'No']

bars = ax2.bar(labels, entropy_terms, color=colors, alpha=0.7)
ax2.set_title('Entropy Terms')
ax2.set_ylabel('-P(Class) × log₂(P(Class))')
ax2.set_ylim(0, max(entropy_terms) * 1.1)

# Add value labels on bars
for i, (bar, term) in enumerate(zip(bars, entropy_terms)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{term:.4f}', ha='center', va='bottom', fontweight='bold')

# Add total entropy line
ax2.axhline(y=h_dataset, color='blue', linestyle='--', alpha=0.7,
           label=f'Total Entropy = {h_dataset:.4f} bits')
ax2.legend()

plt.tight_layout()
file_path = os.path.join(save_dir, "dataset_entropy.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 3: Analyze the Proposed Split
print_step_header(3, "Analyze the Proposed Split")

print("Proposed split: A feature splits the dataset into two branches:")
print("  Branch 1: [6, 2] (6 Yes, 2 No)")
print("  Branch 2: [2, 2] (2 Yes, 2 No)")
print()

# Calculate conditional entropy for each branch
branch1_total = 8  # 6 + 2
branch2_total = 4  # 2 + 2

# Branch 1 probabilities
p_yes_b1 = 6 / branch1_total
p_no_b1 = 2 / branch1_total
h_branch1 = entropy([p_yes_b1, p_no_b1])

print(f"Branch 1 (6 Yes, 2 No):")
print(f"  P(Yes|Branch1) = 6/8 = {p_yes_b1:.3f}")
print(f"  P(No|Branch1) = 2/8 = {p_no_b1:.3f}")
print(f"  H(Class|Branch1) = -{p_yes_b1:.3f} × log₂({p_yes_b1:.3f}) - {p_no_b1:.3f} × log₂({p_no_b1:.3f})")
print(f"  H(Class|Branch1) = {h_branch1:.4f} bits")
print()

# Branch 2 probabilities
p_yes_b2 = 2 / branch2_total
p_no_b2 = 2 / branch2_total
h_branch2 = entropy([p_yes_b2, p_no_b2])

print(f"Branch 2 (2 Yes, 2 No):")
print(f"  P(Yes|Branch2) = 2/4 = {p_yes_b2:.3f}")
print(f"  P(No|Branch2) = 2/4 = {p_no_b2:.3f}")
print(f"  H(Class|Branch2) = -{p_yes_b2:.3f} × log₂({p_yes_b2:.3f}) - {p_no_b2:.3f} × log₂({p_no_b2:.3f})")
print(f"  H(Class|Branch2) = {h_branch2:.4f} bits")
print()

# Calculate weighted conditional entropy
weight_b1 = branch1_total / total_samples
weight_b2 = branch2_total / total_samples

h_conditional = weight_b1 * h_branch1 + weight_b2 * h_branch2

print(f"Weighted conditional entropy:")
print(f"  Weight(Branch1) = {branch1_total}/{total_samples} = {weight_b1:.3f}")
print(f"  Weight(Branch2) = {branch2_total}/{total_samples} = {weight_b2:.3f}")
print(f"  H(Class|Feature) = {weight_b1:.3f} × {h_branch1:.4f} + {weight_b2:.3f} × {h_branch2:.4f}")
print(f"  H(Class|Feature) = {h_conditional:.4f} bits")
print()

# Step 4: Calculate Information Gain
print_step_header(4, "Calculate Information Gain")

# Calculate information gain
info_gain = h_dataset - h_conditional

print(f"Information Gain calculation:")
print(f"  IG = H(Dataset) - H(Class|Feature)")
print(f"  IG = {h_dataset:.4f} - {h_conditional:.4f}")
print(f"  IG = {info_gain:.4f} bits")
print()

# Visualize the information gain calculation
fig, ax = plt.subplots(figsize=(12, 8))

# Create a comparison chart
categories = ['Original Dataset', 'After Split']
entropy_values = [h_dataset, h_conditional]
colors = ['blue', 'red']

bars = ax.bar(categories, entropy_values, color=colors, alpha=0.7)
ax.set_title('Entropy Before and After Split')
ax.set_ylabel('Entropy (bits)')
ax.set_ylim(0, max(entropy_values) * 1.1)

# Add value labels on bars
for bar, value in zip(bars, entropy_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

# Add information gain arrow and label
ax.annotate(f'Information Gain = {info_gain:.4f} bits',
            xy=(0.5, (h_dataset + h_conditional) / 2),
            xytext=(0.5, h_dataset + 0.1),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            ha='center', fontsize=12, fontweight='bold', color='green')

# Add a horizontal line for the reduction
ax.axhline(y=h_conditional, color='red', linestyle='--', alpha=0.7)

plt.tight_layout()
file_path = os.path.join(save_dir, "information_gain_calculation.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 5: Evaluate Split Quality
print_step_header(5, "Evaluate Split Quality")

print("Is this a good split according to ID3?")
print()

# Calculate percentage reduction in entropy
entropy_reduction = (info_gain / h_dataset) * 100

print(f"Analysis:")
print(f"  Information Gain: {info_gain:.4f} bits")
print(f"  Original Entropy: {h_dataset:.4f} bits")
print(f"  Percentage Reduction: {entropy_reduction:.2f}%")
print()

if info_gain > 0:
    print("✓ This is a GOOD split because:")
    print("  - Information gain is positive (> 0)")
    print("  - The split reduces uncertainty about the class")
    print(f"  - Entropy is reduced by {entropy_reduction:.2f}%")
else:
    print("✗ This is a BAD split because:")
    print("  - Information gain is not positive")
    print("  - The split does not reduce uncertainty")
    print("  - No improvement in classification")

print()

# Compare with other potential splits
print("Comparison with other potential splits:")
print()

# Split 1: Perfect split (all Yes in one branch, all No in another)
perfect_split_entropy = 0
perfect_info_gain = h_dataset - perfect_split_entropy
perfect_reduction = (perfect_info_gain / h_dataset) * 100

print(f"Perfect split (pure branches):")
print(f"  IG = {h_dataset:.4f} - 0 = {perfect_info_gain:.4f} bits")
print(f"  Reduction = {perfect_reduction:.2f}%")
print()

# Split 2: No improvement (same distribution in both branches)
no_improvement_entropy = h_dataset
no_improvement_info_gain = h_dataset - no_improvement_entropy

print(f"No improvement split (same distribution):")
print(f"  IG = {h_dataset:.4f} - {h_dataset:.4f} = {no_improvement_info_gain:.4f} bits")
print(f"  Reduction = 0.00%")
print()

# Split 3: Our proposed split
print(f"Our proposed split:")
print(f"  IG = {info_gain:.4f} bits")
print(f"  Reduction = {entropy_reduction:.2f}%")
print()

# Visualize split quality comparison
fig, ax = plt.subplots(figsize=(12, 8))

split_types = ['Perfect Split', 'Our Split', 'No Improvement']
info_gains = [perfect_info_gain, info_gain, no_improvement_info_gain]
colors = ['green', 'orange', 'red']

bars = ax.bar(split_types, info_gains, color=colors, alpha=0.7)
ax.set_title('Information Gain Comparison for Different Splits')
ax.set_ylabel('Information Gain (bits)')
ax.set_ylim(0, max(info_gains) * 1.1)

# Add value labels on bars
for bar, value in zip(bars, info_gains):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

# Add horizontal line at zero
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

plt.tight_layout()
file_path = os.path.join(save_dir, "split_quality_comparison.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 6: Next Steps in ID3
print_step_header(6, "Next Steps in ID3")

print("What is the next step in ID3 after finding the best split?")
print()

print("After identifying the best split (highest information gain), ID3 proceeds as follows:")
print()

print("1. CREATE THE SPLIT:")
print(f"   - Split the dataset using the feature that gave IG = {info_gain:.4f} bits")
print(f"   - Create child nodes for each branch")
print(f"   - Branch 1: {branch1_total} samples")
print(f"   - Branch 2: {branch2_total} samples")
print()

print("2. RECURSIVE TREE BUILDING:")
print("   - For each child node, repeat the ID3 process:")
print("     a) Check stopping criteria")
print("     b) Calculate information gain for remaining features")
print("     c) Choose best feature for splitting")
print("     d) Continue until stopping criteria are met")
print()

print("3. STOPPING CRITERIA CHECK:")
print("   - Pure node: All samples belong to the same class")
print("   - No features: All features have been used")
print("   - Empty dataset: No samples remain")
print()

# Visualize the next steps
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_title('Next Steps in ID3 After Finding Best Split', fontsize=16, fontweight='bold')

# Draw the tree structure
# Root
root = Circle((0.5, 0.9), 0.08, facecolor='gold', edgecolor='orange', linewidth=3)
ax.add_patch(root)
ax.text(0.5, 0.9, 'Root\nDataset\n(12 samples)', ha='center', va='center', fontweight='bold', fontsize=10)

# First level branches
branch1 = Circle((0.2, 0.7), 0.07, facecolor='lightblue', edgecolor='blue', linewidth=2)
branch2 = Circle((0.8, 0.7), 0.07, facecolor='lightblue', edgecolor='blue', linewidth=2)

ax.add_patch(branch1)
ax.add_patch(branch2)

ax.text(0.2, 0.7, 'Branch 1\n(6 Yes, 2 No)\n8 samples', ha='center', va='center', fontweight='bold', fontsize=9)
ax.text(0.8, 0.7, 'Branch 2\n(2 Yes, 2 No)\n4 samples', ha='center', va='center', fontweight='bold', fontsize=9)

# Second level for Branch 1
b1_leaf1 = Circle((0.1, 0.5), 0.05, facecolor='lightgreen', edgecolor='green', linewidth=2)
b1_leaf2 = Circle((0.3, 0.5), 0.05, facecolor='lightgreen', edgecolor='green', linewidth=2)

ax.add_patch(b1_leaf1)
ax.add_patch(b1_leaf2)

ax.text(0.1, 0.5, 'Yes\n(6)', ha='center', va='center', fontweight='bold', fontsize=8)
ax.text(0.3, 0.5, 'No\n(2)', ha='center', va='center', fontweight='bold', fontsize=8)

# Second level for Branch 2
b2_leaf1 = Circle((0.7, 0.5), 0.05, facecolor='lightgreen', edgecolor='green', linewidth=2)
b2_leaf2 = Circle((0.9, 0.5), 0.05, facecolor='lightgreen', edgecolor='green', linewidth=2)

ax.add_patch(b2_leaf1)
ax.add_patch(b2_leaf2)

ax.text(0.7, 0.5, 'Yes\n(2)', ha='center', va='center', fontweight='bold', fontsize=8)
ax.text(0.9, 0.5, 'No\n(2)', ha='center', va='center', fontweight='bold', fontsize=8)

# Add edges
ax.plot([0.5, 0.2], [0.82, 0.77], 'k-', linewidth=2)
ax.plot([0.5, 0.8], [0.82, 0.77], 'k-', linewidth=2)
ax.plot([0.2, 0.1], [0.63, 0.55], 'k-', linewidth=2)
ax.plot([0.2, 0.3], [0.63, 0.55], 'k-', linewidth=2)
ax.plot([0.8, 0.7], [0.63, 0.55], 'k-', linewidth=2)
ax.plot([0.8, 0.9], [0.63, 0.55], 'k-', linewidth=2)

# Add labels
ax.text(0.35, 0.76, 'Feature Split\n(IG = 0.9183)', ha='center', va='center', fontsize=10, color='blue')
ax.text(0.15, 0.53, 'Pure\n(Yes)', ha='center', va='center', fontsize=9, color='green')
ax.text(0.25, 0.53, 'Pure\n(No)', ha='center', va='center', fontsize=9, color='green')
ax.text(0.65, 0.53, 'Mixed\n(2,2)', ha='center', va='center', fontsize=9, color='orange')
ax.text(0.85, 0.53, 'Mixed\n(2,2)', ha='center', va='center', fontsize=9, color='orange')

# Add next steps text
ax.text(0.5, 0.3, 'Next Steps:', ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(0.5, 0.25, '1. Branch 1: Pure node (stop)', ha='center', va='center', fontsize=10, color='green')
ax.text(0.5, 0.2, '2. Branch 2: Continue splitting', ha='center', va='center', fontsize=10, color='orange')
ax.text(0.5, 0.15, '3. Apply ID3 recursively', ha='center', va='center', fontsize=10, color='blue')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.tight_layout()
file_path = os.path.join(save_dir, "next_steps_in_ID3.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 7: Summary and Key Insights
print_step_header(7, "Summary and Key Insights")

print("Question 2 Summary:")
print("=" * 50)
print()
print("1. Dataset Entropy:")
print(f"   ✓ H(Dataset) = {h_dataset:.4f} bits")
print(f"   ✓ Class distribution: Yes={yes_count}/{total_samples}, No={no_count}/{total_samples}")
print()
print("2. Information Gain Calculation:")
print(f"   ✓ H(Class|Feature) = {h_conditional:.4f} bits")
print(f"   ✓ Information Gain = {info_gain:.4f} bits")
print(f"   ✓ Entropy reduction = {entropy_reduction:.2f}%")
print()
print("3. Split Quality Evaluation:")
print("   ✓ This is a GOOD split (IG > 0)")
print("   ✓ Significantly reduces uncertainty")
print("   ✓ Provides meaningful information gain")
print()
print("4. Next Steps in ID3:")
print("   ✓ Create child nodes for each branch")
print("   ✓ Check stopping criteria for each child")
print("   ✓ Recursively apply ID3 to mixed branches")
print("   ✓ Continue until all nodes are pure or stopping criteria met")
print()

print("Key Insights:")
print("- Information gain measures how much a split reduces uncertainty")
print("- Positive information gain indicates a useful split")
print("- ID3 recursively builds the tree by finding the best splits")
print("- The algorithm continues until all branches reach stopping conditions")
print()

print("All figures have been saved to:", save_dir)
print("This example demonstrates how ID3 evaluates potential splits and")
print("makes decisions about tree construction based on information gain.")
