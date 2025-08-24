import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Arrow
import matplotlib.patches as patches

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_30")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True  # Enable LaTeX rendering
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign rendering

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Understanding the Problem
print_step_header(1, "Understanding the Problem")

print("Question 30: Algorithm Stopping Criteria")
print("We need to analyze the stopping criteria used by different decision tree algorithms")
print("and understand when and why algorithms stop building trees.")
print()
print("Tasks:")
print("1. List three stopping criteria used by ID3")
print("2. Identify additional stopping criterion in C4.5 beyond ID3's criteria")
print("3. Name two stopping criteria specific to CART")
print("4. Analyze whether ID3 should continue splitting for a specific node")
print()

# Step 2: ID3 Stopping Criteria Analysis
print_step_header(2, "ID3 Stopping Criteria Analysis")

print("ID3 (Iterative Dichotomiser 3) uses the following stopping criteria:")
print()
print("1. All samples belong to the same class (Pure Node):")
print("   - If all samples in a node have the same class label")
print("   - No further splitting is needed")
print("   - Node becomes a leaf with that class label")
print("   - Example: Node with [10 positive, 0 negative] → Stop, assign 'positive'")
print()
print("2. No more attributes available (Feature Exhaustion):")
print("   - If all features have been used in the path from root to current node")
print("   - Cannot create additional splits")
print("   - Node becomes a leaf with majority class")
print("   - Example: All features used → Stop, assign majority class")
print()
print("3. Empty attribute list (No Features Left):")
print("   - If the list of available attributes is empty")
print("   - Similar to feature exhaustion")
print("   - Node becomes a leaf with majority class")
print("   - Example: A = ∅ → Stop, assign majority class")
print()

print("ID3 Algorithm Pseudo-code:")
print("```")
print("function ID3(D, A):")
print("    if all samples in D belong to same class:")
print("        return leaf node with that class")
print("    if A is empty:")
print("        return leaf node with majority class in D")
print("    # Continue with feature selection and splitting")
print("```")
print()

# Step 3: C4.5 Additional Stopping Criterion
print_step_header(3, "C4.5 Additional Stopping Criterion")

print("C4.5 includes all of ID3's stopping criteria PLUS additional ones:")
print()
print("Additional C4.5 Stopping Criterion:")
print()
print("4. Minimum Sample Threshold:")
print("   - Stop splitting if node contains fewer than a minimum number of samples")
print("   - Prevents overfitting on very small subsets")
print("   - Configurable parameter (e.g., min_samples = 5)")
print("   - Example: Node with 3 samples → Stop if min_samples = 5")
print()
print("5. Maximum Depth Limit:")
print("   - Stop splitting if tree reaches maximum allowed depth")
print("   - Prevents extremely deep trees")
print("   - Configurable parameter (e.g., max_depth = 10)")
print("   - Example: Current depth = 10, max_depth = 10 → Stop")
print()
print("6. Information Gain Threshold:")
print("   - Stop splitting if best information gain is below a threshold")
print("   - Prevents splits that provide minimal improvement")
print("   - Configurable parameter (e.g., min_gain = 0.01)")
print("   - Example: Best IG = 0.005, threshold = 0.01 → Stop")
print()

print("C4.5 Enhanced Algorithm:")
print("```")
print("function C4.5(D, A, depth, min_samples, max_depth, min_gain):")
print("    if all samples in D belong to same class:")
print("        return leaf node with that class")
print("    if A is empty:")
print("        return leaf node with majority class in D")
print("    if |D| < min_samples:")
print("        return leaf node with majority class in D")
print("    if depth >= max_depth:")
print("        return leaf node with majority class in D")
print("    if best_information_gain < min_gain:")
print("        return leaf node with majority class in D")
print("    # Continue with feature selection and splitting")
print("```")
print()

# Step 4: CART-Specific Stopping Criteria
print_step_header(4, "CART-Specific Stopping Criteria")

print("CART (Classification and Regression Trees) has its own stopping criteria:")
print()
print("1. Minimum Impurity Decrease:")
print("   - Stop splitting if the decrease in impurity is below a threshold")
print("   - Uses Gini impurity or MSE instead of entropy")
print("   - Prevents splits that don't significantly improve the tree")
print("   - Example: Gini decrease = 0.005, threshold = 0.01 → Stop")
print()
print("2. Cost-Complexity Pruning:")
print("   - CART doesn't stop early based on simple thresholds")
print("   - Instead, grows a full tree and then prunes it")
print("   - Uses cost-complexity parameter α to control pruning")
print("   - Pruning removes branches that don't improve generalization")
print()
print("3. Cross-Validation for Optimal Tree Size:")
print("   - CART uses cross-validation to find optimal tree size")
print("   - Tests different values of α to minimize validation error")
print("   - Automatically determines when to stop pruning")
print("   - More sophisticated than simple stopping criteria")
print()

print("CART Algorithm Approach:")
print("```")
print("function CART(D, A):")
print("    # Grow full tree with minimal stopping criteria")
print("    if all samples in D belong to same class:")
print("        return leaf node with that class")
print("    if A is empty:")
print("        return leaf node with majority class in D")
print("    # Continue growing until no more splits possible")
print("    # Then apply cost-complexity pruning")
print("```")
print()

# Step 5: Analyzing the Specific Node Case
print_step_header(5, "Analyzing the Specific Node Case")

print("Given: Node with 5 samples (3 positive, 2 negative)")
print()
print("Analysis for ID3:")
print()
print("1. Check Pure Node Criterion:")
print("   - Samples: [3 positive, 2 negative]")
print("   - Not pure (mixed classes)")
print("   - Continue to next criterion")
print()
print("2. Check Feature Availability:")
print("   - Assume features are still available")
print("   - Continue to next criterion")
print()
print("3. Check Attribute List:")
print("   - Assume attributes are still available")
print("   - Continue to next criterion")
print()
print("4. ID3 Decision:")
print("   - All stopping criteria are FALSE")
print("   - ID3 should CONTINUE splitting")
print("   - Node is not pure and features are available")
print()

print("Analysis for C4.5:")
print()
print("1. Check ID3 Criteria (all FALSE as above)")
print("2. Check Minimum Sample Threshold:")
print("   - Current samples: 5")
print("   - If min_samples = 5: Continue splitting")
print("   - If min_samples > 5: Stop splitting")
print("3. Check Maximum Depth (depends on current depth)")
print("4. Check Information Gain Threshold (depends on best split)")
print()

print("Analysis for CART:")
print()
print("1. Check Basic Stopping Criteria (all FALSE as above)")
print("2. CART would continue growing the tree")
print("3. Apply post-pruning to find optimal size")
print("4. Use cross-validation to determine final tree structure")
print()

# Step 6: Decision Tree Visualization
print_step_header(6, "Decision Tree Visualization")

# Create meaningful visualizations

# Plot 1: Decision Tree Structure for ID3
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Root node
root_circle = Circle((5, 9), 0.8, fill=True, color='lightblue', ec='black', linewidth=2)
ax.add_patch(root_circle)
ax.text(5, 9, r'Root', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(5, 8.7, r'[3+, 2-]', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(5, 8.4, r'$H(S)=0.971$', ha='center', va='center', fontsize=9, fontweight='bold')

# Decision branches - arrows from root to children
ax.arrow(5, 8.2, -2.5, -1.5, head_width=0.2, head_length=0.2, fc='black', ec='black')
ax.arrow(5, 8.2, 2.5, -1.5, head_width=0.2, head_length=0.2, fc='black', ec='black')

# Left child (Feature A = 0)
left_circle = Circle((2.5, 6.5), 0.6, fill=True, color='lightgreen', ec='black', linewidth=2)
ax.add_patch(left_circle)
ax.text(2.5, 6.5, r'$A=0$', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(2.5, 6.3, r'[2+, 1-]', ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(2.5, 6.1, r'$H=0.918$', ha='center', va='center', fontsize=8, fontweight='bold')

# Right child (Feature A = 1)
right_circle = Circle((7.5, 6.5), 0.6, fill=True, color='lightgreen', ec='black', linewidth=2)
ax.add_patch(right_circle)
ax.text(7.5, 6.5, r'$A=1$', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(7.5, 6.3, r'[1+, 1-]', ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(7.5, 6.1, r'$H=1.0$', ha='center', va='center', fontsize=8, fontweight='bold')

# Continue arrows - from left child to leaves
ax.arrow(2.5, 5.9, -1.5, -1.2, head_width=0.15, head_length=0.15, fc='black', ec='black')
ax.arrow(2.5, 5.9, 1.5, -1.2, head_width=0.15, head_length=0.15, fc='black', ec='black')
# Continue arrows - from right child to leaves
ax.arrow(7.5, 5.9, -1.5, -1.2, head_width=0.15, head_length=0.15, fc='black', ec='black')
ax.arrow(7.5, 5.9, 1.5, -1.2, head_width=0.15, head_length=0.15, fc='black', ec='black')

# Leaf nodes
leaf1 = Circle((1, 4.5), 0.4, fill=True, color='lightcoral', ec='black', linewidth=2)
ax.add_patch(leaf1)
ax.text(1, 4.5, r'Leaf', ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(1, 4.3, r'[1+, 0-]', ha='center', va='center', fontsize=7, fontweight='bold')

leaf2 = Circle((4, 4.5), 0.4, fill=True, color='lightcoral', ec='black', linewidth=2)
ax.add_patch(leaf2)
ax.text(4, 4.5, r'Leaf', ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(4, 4.3, r'[1+, 1-]', ha='center', va='center', fontsize=7, fontweight='bold')

leaf3 = Circle((6, 4.5), 0.4, fill=True, color='lightcoral', ec='black', linewidth=2)
ax.add_patch(leaf3)
ax.text(6, 4.5, r'Leaf', ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(6, 4.3, r'[0+, 1-]', ha='center', va='center', fontsize=7, fontweight='bold')

leaf4 = Circle((9, 4.5), 0.4, fill=True, color='lightcoral', ec='black', linewidth=2)
ax.add_patch(leaf4)
ax.text(9, 4.5, r'Leaf', ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(9, 4.3, r'[1+, 0-]', ha='center', va='center', fontsize=7, fontweight='bold')

# Labels
ax.text(5, 7.5, r'Feature $A$', ha='center', va='center', fontsize=12, fontweight='bold', 
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", alpha=0.7))

ax.set_title(r'ID3 Decision Tree Growth Process', fontsize=16, fontweight='bold')
ax.set_xlabel('Tree Structure', fontsize=12)
ax.set_ylabel('Depth Level', fontsize=12)
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'id3_decision_tree.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Entropy vs Class Distribution
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
p_values = np.linspace(0.01, 0.99, 100)
entropy_values = -p_values * np.log2(p_values) - (1-p_values) * np.log2(1-p_values)

ax.plot(p_values, entropy_values, 'b-', linewidth=3, label=r'Binary Entropy')
ax.axhline(y=0.971, color='red', linestyle='--', linewidth=2, label=r'Our case: $H(S) = 0.971$')
ax.axvline(x=0.6, color='green', linestyle='--', linewidth=2, label=r'Our case: $p = 3/5 = 0.6$')

ax.scatter([0.6], [0.971], color='red', s=100, zorder=5, label=r'Node [3+, 2-]')

ax.set_xlabel(r'Probability of Positive Class ($p$)', fontsize=12)
ax.set_ylabel(r'Entropy $H(S)$ [bits]', fontsize=12)
ax.set_title(r'Entropy vs Class Distribution for Binary Classification', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.1)

# Add mathematical annotation
ax.text(0.7, 0.8, r'$H(S) = -p\log_2(p) - (1-p)\log_2(1-p)$', 
        fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", ec="black", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'entropy_vs_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Algorithm Comparison Chart
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Define algorithms and their characteristics
algorithms = ['ID3', 'C4.5', 'CART']
stopping_criteria = [3, 6, 3]  # Number of stopping criteria
overfitting_risk = [0.9, 0.5, 0.2]  # Overfitting risk (0-1)
complexity = [0.2, 0.5, 0.8]  # Algorithm complexity (0-1)

# Create bar chart
x = np.arange(len(algorithms))
width = 0.25

bars1 = ax.bar(x - width, stopping_criteria, width, label='Stopping Criteria', color='lightblue', alpha=0.8)
bars2 = ax.bar(x, [r*10 for r in overfitting_risk], width, label=r'Overfitting Risk ($\times 10$)', color='lightcoral', alpha=0.8)
bars3 = ax.bar(x + width, [c*10 for c in complexity], width, label=r'Complexity ($\times 10$)', color='lightgreen', alpha=0.8)

# Customize the plot
ax.set_xlabel('Algorithm', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title(r'Algorithm Comparison: Stopping Criteria, Overfitting Risk, and Complexity', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(algorithms)
ax.legend()
ax.grid(True, alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'algorithm_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Stopping Criteria Decision Flow
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 12)

# Decision flow diagram
def draw_decision_box(x, y, text, color='lightblue'):
    rect = FancyBboxPatch((x-1, y-0.5), 2, 1, boxstyle="round,pad=0.1", 
                          facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

def draw_arrow(start_x, start_y, end_x, end_y, label=''):
    ax.arrow(start_x, start_y, end_x-start_x, end_y-start_y, 
             head_width=0.2, head_length=0.2, fc='black', ec='black', linewidth=1.5)
    if label:
        mid_x, mid_y = (start_x + end_x) / 2, (start_y + end_y) / 2
        ax.text(mid_x, mid_y, label, ha='center', va='center', fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.2", fc="yellow", ec="black", alpha=0.7))

# Start
draw_decision_box(7, 11, r'Start', 'lightyellow')
ax.text(7, 10.7, r'Node [3+, 2-]', ha='center', va='center', fontsize=8, fontweight='bold')

# ID3 criteria
draw_decision_box(2, 9, r'Pure Node?', 'lightcoral')
ax.text(2, 8.7, r'$H(S) = 0.971$', ha='center', va='center', fontsize=8, fontweight='bold')
draw_decision_box(7, 9, r'Features', 'lightcoral')
ax.text(7, 8.7, r'Available?', ha='center', va='center', fontsize=8, fontweight='bold')
draw_decision_box(12, 9, r'Attribute List', 'lightcoral')
ax.text(12, 8.7, r'Empty?', ha='center', va='center', fontsize=8, fontweight='bold')

# C4.5 additional criteria
draw_decision_box(2, 7, r'Min Samples', 'lightgreen')
ax.text(2, 6.7, r'$\geq 5$?', ha='center', va='center', fontsize=8, fontweight='bold')
draw_decision_box(7, 7, r'Max Depth', 'lightgreen')
ax.text(7, 6.7, r'Reached?', ha='center', va='center', fontsize=8, fontweight='bold')
draw_decision_box(12, 7, r'Min Gain', 'lightgreen')
ax.text(12, 6.7, r'Threshold?', ha='center', va='center', fontsize=8, fontweight='bold')

# CART approach
draw_decision_box(7, 5, r'CART: Grow', 'lightblue')
ax.text(7, 4.7, r'Full Tree', ha='center', va='center', fontsize=8, fontweight='bold')

# Decisions
draw_decision_box(2, 3, r'STOP', 'red')
ax.text(2, 2.7, r'Create Leaf', ha='center', va='center', fontsize=8, fontweight='bold')
draw_decision_box(7, 3, r'CONTINUE', 'green')
ax.text(7, 2.7, r'Splitting', ha='center', va='center', fontsize=8, fontweight='bold')
draw_decision_box(12, 3, r'STOP', 'red')
ax.text(12, 2.7, r'Create Leaf', ha='center', va='center', fontsize=8, fontweight='bold')

# Final decision
draw_decision_box(7, 1, r'Final Decision:', 'green')
ax.text(7, 0.7, r'CONTINUE', ha='center', va='center', fontsize=8, fontweight='bold')

# Draw arrows
# Start to ID3 criteria
draw_arrow(7, 10.5, 2, 9.5, 'ID3')
draw_arrow(7, 10.5, 7, 9.5, 'ID3')
draw_arrow(7, 10.5, 12, 9.5, 'ID3')

# ID3 to C4.5
draw_arrow(2, 8.5, 2, 7.5, 'NO')
draw_arrow(7, 8.5, 7, 7.5, 'NO')
draw_arrow(12, 8.5, 12, 7.5, 'NO')

# C4.5 to decisions
draw_arrow(2, 6.5, 2, 3.5, 'NO')
draw_arrow(7, 6.5, 7, 3.5, 'NO')
draw_arrow(12, 6.5, 12, 3.5, 'NO')

# To final decision
draw_arrow(7, 2.5, 7, 1.5)

ax.set_title(r'Stopping Criteria Decision Flow for Node [3+, 2-]', fontsize=16, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'stopping_criteria_flow.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 5: Cost-Complexity Pruning Visualization
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Simulate cost-complexity function
alpha_values = np.linspace(0, 0.1, 100)
tree_sizes = 20 * np.exp(-alpha_values * 50) + 5  # Simulated tree size
misclassification_rates = 0.15 + 0.05 * np.exp(-alpha_values * 30)  # Simulated error rate
total_cost = misclassification_rates + alpha_values * tree_sizes

# Find optimal alpha
optimal_idx = np.argmin(total_cost)
optimal_alpha = alpha_values[optimal_idx]

# Plot the curves
ax.plot(alpha_values, misclassification_rates, 'b-', linewidth=2, label=r'Misclassification Rate $R(T)$')
ax.plot(alpha_values, alpha_values * tree_sizes, 'g-', linewidth=2, label=r'Complexity Penalty $\alpha|T|$')
ax.plot(alpha_values, total_cost, 'r-', linewidth=3, label=r'Total Cost $R_\alpha(T)$')

# Mark optimal point
ax.axvline(x=optimal_alpha, color='red', linestyle='--', alpha=0.7, label=r'Optimal $\alpha = {:.4f}$'.format(optimal_alpha))
ax.scatter([optimal_alpha], [total_cost[optimal_idx]], color='red', s=100, zorder=5)

ax.set_xlabel(r'Complexity Parameter $\alpha$', fontsize=12)
ax.set_ylabel('Cost', fontsize=12)
ax.set_title(r'Cost-Complexity Pruning: $R_\alpha(T) = R(T) + \alpha|T|$', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Add mathematical annotation
ax.text(0.07, 0.3, r'$R_\alpha(T) = R(T) + \alpha|T|$', 
        fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", ec="black", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'cost_complexity_pruning.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nPlots saved to: {save_dir}")
print("\nStopping criteria analysis complete!")
print("\nCreated meaningful visualizations:")
print("1. ID3 Decision Tree Growth Process")
print("2. Entropy vs Class Distribution")
print("3. Algorithm Comparison Chart")
print("4. Stopping Criteria Decision Flow")
print("5. Cost-Complexity Pruning Visualization")
