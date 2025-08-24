import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.patches import Circle, Arrow
import matplotlib.patches as mpatches

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_4")
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

print("Question 4: ID3 Tree Construction")
print("Consider building a decision tree for a weather dataset with features:")
print()
print("| Feature | Values |")
print("|---------|--------|")
print("| Outlook | Sunny, Rainy, Cloudy |")
print("| Temperature | Hot, Mild, Cool |")
print("| Humidity | High, Normal |")
print("| Windy | True, False |")
print()
print("Tasks:")
print("1. How many possible leaf nodes could this tree have?")
print("2. What is the maximum depth of the tree?")
print("3. How would ID3 handle categorical features with many values?")
print("4. What are the limitations of ID3 for this dataset?")
print()

# Step 2: Calculate Possible Leaf Nodes
print_step_header(2, "Calculate Possible Leaf Nodes")

print("To determine the maximum number of possible leaf nodes, we need to consider:")
print("1. The number of unique combinations of feature values")
print("2. The branching factor at each level")
print("3. The maximum depth of the tree")
print()

# Feature values
features = {
    'Outlook': ['Sunny', 'Rainy', 'Cloudy'],
    'Temperature': ['Hot', 'Mild', 'Cool'],
    'Humidity': ['High', 'Normal'],
    'Windy': [True, False]
}

print("Feature Analysis:")
for feature, values in features.items():
    print(f"  {feature}: {len(values)} values - {values}")

# Calculate maximum possible leaf nodes
max_leaf_nodes = 1
for feature, values in features.items():
    max_leaf_nodes *= len(values)

print(f"\nMaximum possible leaf nodes:")
print(f"  Product of all feature values: {max_leaf_nodes}")
print(f"  This represents all possible combinations of feature values")
print()

# Visualize the branching structure
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_title('Maximum Possible Leaf Nodes in Weather Dataset', fontsize=16, fontweight='bold')

# Draw a tree showing the maximum possible structure
# Root
root = Circle((0.5, 0.9), 0.08, facecolor='lightblue', edgecolor='blue', linewidth=2)
ax.add_patch(root)
ax.text(0.5, 0.9, 'Outlook\n(3 branches)', ha='center', va='center', fontweight='bold', fontsize=10)

# First level - Outlook branches
sunny = Circle((0.2, 0.7), 0.06, facecolor='lightblue', edgecolor='blue', linewidth=2)
rainy = Circle((0.5, 0.7), 0.06, facecolor='lightblue', edgecolor='blue', linewidth=2)
cloudy = Circle((0.8, 0.7), 0.06, facecolor='lightblue', edgecolor='blue', linewidth=2)

ax.add_patch(sunny)
ax.add_patch(rainy)
ax.add_patch(cloudy)

ax.text(0.2, 0.7, 'Sunny\n(3 branches)', ha='center', va='center', fontweight='bold', fontsize=9)
ax.text(0.5, 0.7, 'Rainy\n(3 branches)', ha='center', va='center', fontweight='bold', fontsize=9)
ax.text(0.8, 0.7, 'Cloudy\n(3 branches)', ha='center', va='center', fontweight='bold', fontsize=9)

# Second level - Temperature branches for each outlook
# Sunny branch
sunny_hot = Circle((0.1, 0.5), 0.05, facecolor='lightblue', edgecolor='blue', linewidth=2)
sunny_mild = Circle((0.2, 0.5), 0.05, facecolor='lightblue', edgecolor='blue', linewidth=2)
sunny_cool = Circle((0.3, 0.5), 0.05, facecolor='lightblue', edgecolor='blue', linewidth=2)

ax.add_patch(sunny_hot)
ax.add_patch(sunny_mild)
ax.add_patch(sunny_cool)

ax.text(0.1, 0.5, 'Hot\n(2 branches)', ha='center', va='center', fontweight='bold', fontsize=8)
ax.text(0.2, 0.5, 'Mild\n(2 branches)', ha='center', va='center', fontweight='bold', fontsize=8)
ax.text(0.3, 0.5, 'Cool\n(2 branches)', ha='center', va='center', fontweight='bold', fontsize=8)

# Rainy branch
rainy_hot = Circle((0.4, 0.5), 0.05, facecolor='lightblue', edgecolor='blue', linewidth=2)
rainy_mild = Circle((0.5, 0.5), 0.05, facecolor='lightblue', edgecolor='blue', linewidth=2)
rainy_cool = Circle((0.6, 0.5), 0.05, facecolor='lightblue', edgecolor='blue', linewidth=2)

ax.add_patch(rainy_hot)
ax.add_patch(rainy_mild)
ax.add_patch(rainy_cool)

ax.text(0.4, 0.5, 'Hot\n(2 branches)', ha='center', va='center', fontweight='bold', fontsize=8)
ax.text(0.5, 0.5, 'Mild\n(2 branches)', ha='center', va='center', fontweight='bold', fontsize=8)
ax.text(0.6, 0.5, 'Cool\n(2 branches)', ha='center', va='center', fontweight='bold', fontsize=8)

# Cloudy branch
cloudy_hot = Circle((0.7, 0.5), 0.05, facecolor='lightblue', edgecolor='blue', linewidth=2)
cloudy_mild = Circle((0.8, 0.5), 0.05, facecolor='lightblue', edgecolor='blue', linewidth=2)
cloudy_cool = Circle((0.9, 0.5), 0.05, facecolor='lightblue', edgecolor='blue', linewidth=2)

ax.add_patch(cloudy_hot)
ax.add_patch(cloudy_mild)
ax.add_patch(cloudy_cool)

ax.text(0.7, 0.5, 'Hot\n(2 branches)', ha='center', va='center', fontweight='bold', fontsize=8)
ax.text(0.8, 0.5, 'Mild\n(2 branches)', ha='center', va='center', fontweight='bold', fontsize=8)
ax.text(0.9, 0.5, 'Cool\n(2 branches)', ha='center', va='center', fontweight='bold', fontsize=8)

# Third level - Humidity branches (simplified representation)
# Show a few examples of leaf nodes
leaf1 = Circle((0.05, 0.3), 0.04, facecolor='lightgreen', edgecolor='green', linewidth=2)
leaf2 = Circle((0.15, 0.3), 0.04, facecolor='lightgreen', edgecolor='green', linewidth=2)
leaf3 = Circle((0.25, 0.3), 0.04, facecolor='lightgreen', edgecolor='green', linewidth=2)

ax.add_patch(leaf1)
ax.add_patch(leaf2)
ax.add_patch(leaf3)

ax.text(0.05, 0.3, 'Leaf', ha='center', va='center', fontweight='bold', fontsize=7)
ax.text(0.15, 0.3, 'Leaf', ha='center', va='center', fontweight='bold', fontsize=7)
ax.text(0.25, 0.3, 'Leaf', ha='center', va='center', fontweight='bold', fontsize=7)

# Add edges
# Root to first level
ax.plot([0.5, 0.2], [0.82, 0.76], 'k-', linewidth=2)
ax.plot([0.5, 0.5], [0.82, 0.76], 'k-', linewidth=2)
ax.plot([0.5, 0.8], [0.82, 0.76], 'k-', linewidth=2)

# First to second level
ax.plot([0.2, 0.1], [0.64, 0.55], 'k-', linewidth=1.5)
ax.plot([0.2, 0.2], [0.64, 0.55], 'k-', linewidth=1.5)
ax.plot([0.2, 0.3], [0.64, 0.55], 'k-', linewidth=1.5)

ax.plot([0.5, 0.4], [0.64, 0.55], 'k-', linewidth=1.5)
ax.plot([0.5, 0.5], [0.64, 0.55], 'k-', linewidth=1.5)
ax.plot([0.5, 0.6], [0.64, 0.55], 'k-', linewidth=1.5)

ax.plot([0.8, 0.7], [0.64, 0.55], 'k-', linewidth=1.5)
ax.plot([0.8, 0.8], [0.64, 0.55], 'k-', linewidth=1.5)
ax.plot([0.8, 0.9], [0.64, 0.55], 'k-', linewidth=1.5)

# Second to third level (sample)
ax.plot([0.1, 0.05], [0.45, 0.34], 'k-', linewidth=1)
ax.plot([0.2, 0.15], [0.45, 0.34], 'k-', linewidth=1)
ax.plot([0.3, 0.25], [0.45, 0.34], 'k-', linewidth=1)

# Add labels
ax.text(0.35, 0.76, '3 branches', ha='center', va='center', fontsize=10, color='blue')
ax.text(0.15, 0.53, '3 branches', ha='center', va='center', fontsize=9, color='blue')
ax.text(0.5, 0.53, '3 branches', ha='center', va='center', fontsize=9, color='blue')
ax.text(0.85, 0.53, '3 branches', ha='center', va='center', fontsize=9, color='blue')

# Add calculation text
ax.text(0.5, 0.15, f'Maximum Leaf Nodes = 3 × 3 × 2 × 2 = {max_leaf_nodes}', 
        ha='center', va='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.tight_layout()
file_path = os.path.join(save_dir, "maximum_leaf_nodes.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 3: Maximum Tree Depth
print_step_header(3, "Maximum Tree Depth")

print("What is the maximum depth of the tree?")
print()
print("The maximum depth depends on:")
print("1. The number of features available")
print("2. Whether features can be reused")
print("3. The stopping criteria")
print()

# Calculate maximum depth
max_depth = len(features)
print(f"Maximum depth calculation:")
print(f"  Number of features: {max_depth}")
print(f"  Features: {list(features.keys())}")
print(f"  Maximum depth = {max_depth}")
print()

print("Explanation:")
print("- ID3 uses each feature only once in a path from root to leaf")
print("- This prevents infinite loops and ensures termination")
print("- The maximum depth equals the number of features")
print("- In practice, the actual depth may be less due to stopping criteria")
print()

# Visualize maximum depth
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_title('Maximum Tree Depth Analysis', fontsize=16, fontweight='bold')

# Draw a simplified tree showing maximum depth
depths = ['Root', 'Level 1', 'Level 2', 'Level 3', 'Level 4']
feature_names = ['', 'Outlook', 'Temperature', 'Humidity', 'Windy']
y_positions = [0.9, 0.7, 0.5, 0.3, 0.1]

for i, (depth, feature, y_pos) in enumerate(zip(depths, feature_names, y_positions)):
    # Draw node
    if i == 0:  # Root
        node = Circle((0.5, y_pos), 0.08, facecolor='gold', edgecolor='orange', linewidth=2)
        ax.add_patch(node)
        ax.text(0.5, y_pos, f'{depth}\n(Start)', ha='center', va='center', fontweight='bold', fontsize=10)
    elif i == len(depths) - 1:  # Leaf
        node = Circle((0.5, y_pos), 0.08, facecolor='lightgreen', edgecolor='green', linewidth=2)
        ax.add_patch(node)
        ax.text(0.5, y_pos, f'{depth}\n(Leaf)', ha='center', va='center', fontweight='bold', fontsize=10)
    else:  # Internal nodes
        node = Circle((0.5, y_pos), 0.08, facecolor='lightblue', edgecolor='blue', linewidth=2)
        ax.add_patch(node)
        ax.text(0.5, y_pos, f'{depth}\n{feature}', ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Add depth label
    ax.text(0.1, y_pos, f'Depth {i}', ha='center', va='center', fontsize=11, fontweight='bold')

# Add edges
for i in range(len(depths) - 1):
    ax.plot([0.5, 0.5], [y_positions[i] - 0.08, y_positions[i + 1] + 0.08], 'k-', linewidth=2)

# Add depth analysis
ax.text(0.8, 0.5, 'Depth Analysis:', ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(0.8, 0.4, f'• Root: Depth 0', ha='center', va='center', fontsize=10)
ax.text(0.8, 0.3, f'• Level 1: Depth 1', ha='center', va='center', fontsize=10)
ax.text(0.8, 0.2, f'• Level 2: Depth 2', ha='center', va='center', fontsize=10)
ax.text(0.8, 0.1, f'• Level 3: Depth 3', ha='center', va='center', fontsize=10)
ax.text(0.8, 0.0, f'• Maximum Depth = {max_depth}', ha='center', va='center', fontsize=11, fontweight='bold', color='red')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.tight_layout()
file_path = os.path.join(save_dir, "maximum_tree_depth.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 4: Handling Categorical Features with Many Values
print_step_header(4, "Handling Categorical Features with Many Values")

print("How would ID3 handle categorical features with many values?")
print()
print("ID3 creates multi-way splits for categorical features:")
print("1. Each unique value becomes a separate branch")
print("2. This can lead to overfitting with high-cardinality features")
print("3. Features with many values may be preferred due to higher information gain")
print("4. This is a known limitation of ID3")
print()

# Demonstrate with examples
print("Examples of categorical feature handling:")
print()

# Example 1: Low cardinality (good)
print("Example 1: Low cardinality (good)")
print("  Feature: Outlook")
print("  Values: ['Sunny', 'Rainy', 'Cloudy']")
print("  Branches: 3 (reasonable)")
print("  Information gain: Moderate")
print()

# Example 2: High cardinality (problematic)
print("Example 2: High cardinality (problematic)")
print("  Feature: Customer ID")
print("  Values: ['C001', 'C002', 'C003', ..., 'C1000']")
print("  Branches: 1000 (excessive)")
print("  Information gain: Very high (but misleading)")
print()

# Example 3: Medium cardinality (moderate)
print("Example 3: Medium cardinality (moderate)")
print("  Feature: City")
print("  Values: ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix', ...]")
print("  Branches: 50 (moderate)")
print("  Information gain: High (may be overfitting)")
print()

# Visualize the problem
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Categorical Feature Handling in ID3', fontsize=16, fontweight='bold')

# Good case: Low cardinality
ax1.set_title('Low Cardinality (Good)', fontweight='bold', color='green')
ax1.text(0.5, 0.9, 'Feature: Outlook', ha='center', va='center', fontsize=12)
ax1.text(0.5, 0.8, 'Values: 3', ha='center', va='center', fontsize=11)
ax1.text(0.5, 0.7, 'Branches: 3', ha='center', va='center', fontsize=11)
ax1.text(0.5, 0.6, '→ Reasonable splitting', ha='center', va='center', fontsize=11, color='green')
ax1.text(0.5, 0.5, '→ Good generalization', ha='center', va='center', fontsize=11, color='green')

# Draw simple tree
root = Circle((0.5, 0.3), 0.06, facecolor='lightgreen', edgecolor='green', linewidth=2)
ax1.add_patch(root)
ax1.text(0.5, 0.3, 'Outlook', ha='center', va='center', fontweight='bold', fontsize=10)

# Branches
sunny = Circle((0.2, 0.15), 0.04, facecolor='lightgreen', edgecolor='green', linewidth=2)
rainy = Circle((0.5, 0.15), 0.04, facecolor='lightgreen', edgecolor='green', linewidth=2)
cloudy = Circle((0.8, 0.15), 0.04, facecolor='lightgreen', edgecolor='green', linewidth=2)

ax1.add_patch(sunny)
ax1.add_patch(rainy)
ax1.add_patch(cloudy)

ax1.text(0.2, 0.15, 'Sunny', ha='center', va='center', fontweight='bold', fontsize=8)
ax1.text(0.5, 0.15, 'Rainy', ha='center', va='center', fontweight='bold', fontsize=8)
ax1.text(0.8, 0.15, 'Cloudy', ha='center', va='center', fontweight='bold', fontsize=8)

# Add edges
ax1.plot([0.5, 0.2], [0.24, 0.19], 'k-', linewidth=2)
ax1.plot([0.5, 0.5], [0.24, 0.19], 'k-', linewidth=2)
ax1.plot([0.5, 0.8], [0.24, 0.19], 'k-', linewidth=2)

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')

# Problematic case: High cardinality
ax2.set_title('High Cardinality (Problematic)', fontweight='bold', color='red')
ax2.text(0.5, 0.9, 'Feature: Customer ID', ha='center', va='center', fontsize=12)
ax2.text(0.5, 0.8, 'Values: 1000', ha='center', va='center', fontsize=11)
ax2.text(0.5, 0.7, 'Branches: 1000', ha='center', va='center', fontsize=11)
ax2.text(0.5, 0.6, '→ Excessive splitting', ha='center', va='center', fontsize=11, color='red')
ax2.text(0.5, 0.5, '→ Overfitting risk', ha='center', va='center', fontsize=11, color='red')

# Draw problematic tree
root2 = Circle((0.5, 0.3), 0.06, facecolor='lightcoral', edgecolor='red', linewidth=2)
ax2.add_patch(root2)
ax2.text(0.5, 0.3, 'Customer ID', ha='center', va='center', fontweight='bold', fontsize=10)

# Show many branches (simplified)
for i in range(10):
    x_pos = 0.1 + i * 0.08
    if x_pos <= 0.9:
        branch = Circle((x_pos, 0.15), 0.02, facecolor='lightcoral', edgecolor='red', linewidth=1)
        ax2.add_patch(branch)
        ax2.text(x_pos, 0.15, f'C{i+1:03d}', ha='center', va='center', fontweight='bold', fontsize=6)
        ax2.plot([0.5, x_pos], [0.24, 0.17], 'k-', linewidth=1, alpha=0.5)

# Add ellipsis
ax2.text(0.5, 0.05, '... and 990 more branches', ha='center', va='center', fontsize=10, color='red')

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')

plt.tight_layout()
file_path = os.path.join(save_dir, "categorical_feature_handling.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 5: ID3 Limitations for This Dataset
print_step_header(5, "ID3 Limitations for This Dataset")

print("What are the limitations of ID3 for this dataset?")
print()
print("ID3 has several limitations when applied to the weather dataset:")
print()

limitations = [
    "1. Categorical Feature Bias:",
    "   - Features with more values may be preferred",
    "   - Outlook (3 values) vs Windy (2 values)",
    "   - May not choose the most meaningful feature",
    "",
    "2. No Continuous Feature Handling:",
    "   - Temperature could be continuous (e.g., 75.5°F)",
    "   - ID3 cannot handle numerical ranges",
    "   - Requires discretization preprocessing",
    "",
    "3. Overfitting Risk:",
    "   - Can create overly complex trees",
    "   - May memorize training data",
    "   - Poor generalization to new data",
    "",
    "4. No Pruning:",
    "   - No mechanism to remove unnecessary branches",
    "   - Trees can become unnecessarily deep",
    "   - No regularization techniques",
    "",
    "5. Binary Classification Focus:",
    "   - Designed primarily for binary classification",
    "   - May not handle multi-class problems optimally",
    "   - Limited to classification tasks"
]

for limitation in limitations:
    print(limitation)

# Visualize limitations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('ID3 Limitations for Weather Dataset', fontsize=16, fontweight='bold')

# Limitation 1: Categorical Feature Bias
ax1 = axes[0, 0]
ax1.set_title('1. Categorical Feature Bias', fontweight='bold', color='red')
ax1.text(0.5, 0.8, 'Problem:', ha='center', va='center', fontsize=11, fontweight='bold')
ax1.text(0.5, 0.7, 'Features with more values', ha='center', va='center', fontsize=10)
ax1.text(0.5, 0.6, 'get higher information gain', ha='center', va='center', fontsize=10)
ax1.text(0.5, 0.5, 'Example:', ha='center', va='center', fontsize=10, fontweight='bold')
ax1.text(0.5, 0.4, 'Outlook (3 values) vs', ha='center', va='center', fontsize=10)
ax1.text(0.5, 0.3, 'Windy (2 values)', ha='center', va='center', fontsize=10)

# Draw comparison
ax1.bar(['Outlook', 'Windy'], [3, 2], color=['red', 'blue'], alpha=0.7)
ax1.set_ylabel('Number of Values')
ax1.set_ylim(0, 4)

ax1.set_xlim(-0.5, 1.5)
ax1.axis('off')

# Limitation 2: No Continuous Features
ax2 = axes[0, 1]
ax2.set_title('2. No Continuous Features', fontweight='bold', color='orange')
ax2.text(0.5, 0.8, 'Problem:', ha='center', va='center', fontsize=11, fontweight='bold')
ax2.text(0.5, 0.7, 'Cannot handle numerical', ha='center', va='center', fontsize=10)
ax2.text(0.5, 0.6, 'ranges or continuous values', ha='center', va='center', fontsize=10)
ax2.text(0.5, 0.5, 'Example:', ha='center', va='center', fontsize=10, fontweight='bold')
ax2.text(0.5, 0.4, 'Temperature: 75.5°F', ha='center', va='center', fontsize=10)
ax2.text(0.5, 0.3, 'Must be discretized first', ha='center', va='center', fontsize=10)

# Draw temperature example
temps = [70, 75, 80, 85, 90]
ax2.plot(temps, [0.2, 0.4, 0.6, 0.4, 0.2], 'o-', color='orange', linewidth=2)
ax2.set_xlabel('Temperature (°F)')
ax2.set_ylabel('Play Probability')
ax2.set_title('Continuous Temperature')

# Limitation 3: Overfitting Risk
ax3 = axes[0, 2]
ax3.set_title('3. Overfitting Risk', fontweight='bold', color='red')
ax3.text(0.5, 0.8, 'Problem:', ha='center', va='center', fontsize=11, fontweight='bold')
ax3.text(0.5, 0.7, 'Can create overly complex', ha='center', va='center', fontsize=10)
ax3.text(0.5, 0.6, 'trees that memorize data', ha='center', va='center', fontsize=10)
ax3.text(0.5, 0.5, 'Result:', ha='center', va='center', fontsize=10, fontweight='bold')
ax3.text(0.5, 0.4, 'Poor generalization', ha='center', va='center', fontsize=10)
ax3.text(0.5, 0.3, 'to new data', ha='center', va='center', fontsize=10)

# Draw overfitting example
x = np.linspace(0, 1, 100)
y_true = 0.5 + 0.3 * np.sin(2 * np.pi * x)
y_overfit = y_true + 0.1 * np.random.randn(100)

ax3.plot(x, y_true, 'b-', linewidth=2, label='True Pattern')
ax3.scatter(x[::10], y_overfit[::10], color='red', alpha=0.7, label='Training Data')
ax3.set_xlabel('Feature Value')
ax3.set_ylabel('Target')
ax3.legend()

# Limitation 4: No Pruning
ax4 = axes[1, 0]
ax4.set_title('4. No Pruning', fontweight='bold', color='purple')
ax4.text(0.5, 0.8, 'Problem:', ha='center', va='center', fontsize=11, fontweight='bold')
ax4.text(0.5, 0.7, 'No mechanism to remove', ha='center', va='center', fontsize=10)
ax4.text(0.5, 0.6, 'unnecessary branches', ha='center', va='center', fontsize=10)
ax4.text(0.5, 0.5, 'Result:', ha='center', va='center', fontsize=10, fontweight='bold')
ax4.text(0.5, 0.4, 'Trees can become', ha='center', va='center', fontsize=10)
ax4.text(0.5, 0.3, 'unnecessarily deep', ha='center', va='center', fontsize=10)

# Draw deep tree example
root = Circle((0.5, 0.9), 0.05, facecolor='lightblue', edgecolor='blue', linewidth=2)
ax4.add_patch(root)
ax4.text(0.5, 0.9, 'Root', ha='center', va='center', fontweight='bold', fontsize=8)

# Add many levels
for i in range(1, 6):
    y_pos = 0.9 - i * 0.15
    for j in range(2**i):
        x_pos = 0.3 + j * 0.4 / (2**i - 1) if 2**i > 1 else 0.5
        node = Circle((x_pos, y_pos), 0.03, facecolor='lightblue', edgecolor='blue', linewidth=1)
        ax4.add_patch(node)
        if i == 5:  # Last level
            node.set_facecolor('lightgreen')
            node.set_edgecolor('green')

ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')

# Limitation 5: Binary Classification Focus
ax5 = axes[1, 1]
ax5.set_title('5. Binary Classification Focus', fontweight='bold', color='brown')
ax5.text(0.5, 0.8, 'Problem:', ha='center', va='center', fontsize=11, fontweight='bold')
ax5.text(0.5, 0.7, 'Designed primarily for', ha='center', va='center', fontsize=10)
ax5.text(0.5, 0.6, 'binary classification', ha='center', va='center', fontsize=10)
ax5.text(0.5, 0.5, 'May not handle multi-class', ha='center', va='center', fontsize=10)
ax5.text(0.5, 0.4, 'problems optimally', ha='center', va='center', fontsize=10)

# Draw binary vs multi-class
ax5.bar(['Binary', 'Multi-Class'], [1, 0.7], color=['green', 'orange'], alpha=0.7)
ax5.set_ylabel('Performance')
ax5.set_ylim(0, 1.2)

# Limitation 6: Summary
ax6 = axes[1, 2]
ax6.set_title('6. Summary of Limitations', fontweight='bold', color='darkred')
ax6.text(0.5, 0.8, 'Key Issues:', ha='center', va='center', fontsize=11, fontweight='bold')
ax6.text(0.5, 0.7, '• Feature bias', ha='center', va='center', fontsize=10)
ax6.text(0.5, 0.6, '• No continuous features', ha='center', va='center', fontsize=10)
ax6.text(0.5, 0.5, '• Overfitting risk', ha='center', va='center', fontsize=10)
ax6.text(0.5, 0.4, '• No pruning', ha='center', va='center', fontsize=10)
ax6.text(0.5, 0.3, '• Binary focus', ha='center', va='center', fontsize=10)
ax6.text(0.5, 0.2, 'Solutions: C4.5, CART', ha='center', va='center', fontsize=10, color='green')

ax6.set_xlim(0, 1)
ax6.set_ylim(0, 1)
ax6.axis('off')

plt.tight_layout()
file_path = os.path.join(save_dir, "ID3_limitations.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 6: Summary and Key Insights
print_step_header(6, "Summary and Key Insights")

print("Question 4 Summary:")
print("=" * 50)
print()
print("1. Maximum Possible Leaf Nodes:")
print(f"   ✓ Product of all feature values: {max_leaf_nodes}")
print(f"   ✓ 3 × 3 × 2 × 2 = {max_leaf_nodes} possible combinations")
print(f"   ✓ Represents all possible feature value combinations")
print()
print("2. Maximum Tree Depth:")
print(f"   ✓ Equal to number of features: {max_depth}")
print(f"   ✓ Each feature used at most once per path")
print(f"   ✓ Prevents infinite loops and ensures termination")
print()
print("3. Categorical Feature Handling:")
print("   ✓ Multi-way splits for each unique value")
print("   ✓ High-cardinality features may be preferred")
print("   ✓ Risk of overfitting with many values")
print()
print("4. ID3 Limitations:")
print("   ✓ Categorical feature bias")
print("   ✓ No continuous feature handling")
print("   ✓ Overfitting risk")
print("   ✓ No pruning mechanisms")
print("   ✓ Binary classification focus")
print()

print("Key Insights:")
print("- ID3 has theoretical maximums based on feature characteristics")
print("- Categorical features with many values can cause problems")
print("- The algorithm has several practical limitations")
print("- Understanding limitations helps choose appropriate algorithms")
print()

print("All figures have been saved to:", save_dir)
print("This analysis shows how ID3's theoretical capabilities")
print("compare to its practical limitations for real-world datasets.")
