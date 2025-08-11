import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.patches import Circle, Arrow
import matplotlib.patches as mpatches

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_3")
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

print("Question 3: ID3 Stopping Criteria")
print("ID3 uses stopping criteria to prevent infinite recursion.")
print()
print("Tasks:")
print("1. What are the three main stopping criteria in ID3?")
print("2. Why is it important to have stopping criteria?")
print("3. What happens when all features have been used?")
print("4. How do you handle cases where no features remain but the node is not pure?")
print()

# Step 2: Three Main Stopping Criteria
print_step_header(2, "Three Main Stopping Criteria")

print("The three main stopping criteria in ID3 are:")
print()
print("1. PURE NODE: All samples belong to the same class")
print("2. NO FEATURES: All features have been used")
print("3. EMPTY DATASET: No samples remain after splitting")
print()

# Create a comprehensive visualization of stopping criteria
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('ID3 Stopping Criteria - Complete Overview', fontsize=16, fontweight='bold')

# 1. Pure Node
ax1 = axes[0, 0]
ax1.set_title('1. Pure Node - All samples same class', fontweight='bold', color='green')
ax1.text(0.5, 0.8, 'Node contains only "Yes" samples', ha='center', va='center', fontsize=12)
ax1.text(0.5, 0.7, '→ Create leaf node with class "Yes"', ha='center', va='center', fontsize=11)
ax1.text(0.5, 0.6, '→ Stop recursion', ha='center', va='center', fontsize=11, color='green')
ax1.text(0.5, 0.5, '→ Perfect classification', ha='center', va='center', fontsize=11, color='green')

# Draw a leaf node
leaf1 = Circle((0.5, 0.2), 0.08, facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
ax1.add_patch(leaf1)
ax1.text(0.5, 0.2, 'Yes', ha='center', va='center', fontweight='bold', fontsize=14)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')

# 2. No Features
ax2 = axes[0, 1]
ax2.set_title('2. No Features - All features used', fontweight='bold', color='orange')
ax2.text(0.5, 0.8, 'Features used: Outlook, Temperature, Humidity, Windy', ha='center', va='center', fontsize=11)
ax2.text(0.5, 0.7, '→ No more features to split on', ha='center', va='center', fontsize=11)
ax2.text(0.5, 0.6, '→ Use majority class rule', ha='center', va='center', fontsize=11, color='orange')
ax2.text(0.5, 0.5, '→ May not be pure', ha='center', va='center', fontsize=11, color='orange')

# Draw a leaf node
leaf2 = Circle((0.5, 0.2), 0.08, facecolor='orange', edgecolor='darkorange', linewidth=2)
ax2.add_patch(leaf2)
ax2.text(0.5, 0.2, 'Majority\nClass', ha='center', va='center', fontweight='bold', fontsize=10)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')

# 3. Empty Dataset
ax3 = axes[1, 0]
ax3.set_title('3. Empty Dataset - No samples after split', fontweight='bold', color='red')
ax3.text(0.5, 0.8, 'Split resulted in empty branch', ha='center', va='center', fontsize=11)
ax3.text(0.5, 0.7, '→ No samples to classify', ha='center', va='center', fontsize=11)
ax3.text(0.5, 0.6, '→ Use parent node majority class', ha='center', va='center', fontsize=11, color='red')
ax3.text(0.5, 0.5, '→ Handle edge cases gracefully', ha='center', va='center', fontsize=11, color='red')

# Draw an empty node
empty_node = Circle((0.5, 0.2), 0.08, facecolor='lightcoral', edgecolor='red', linewidth=2)
ax3.add_patch(empty_node)
ax3.text(0.5, 0.2, 'Empty\nBranch', ha='center', va='center', fontweight='bold', fontsize=10)
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.axis('off')

# 4. Decision Tree Example
ax4 = axes[1, 1]
ax4.set_title('4. Complete Decision Tree with Stopping Criteria', fontweight='bold')

# Draw a tree structure showing stopping criteria
# Root
root = Circle((0.5, 0.9), 0.08, facecolor='lightblue', edgecolor='blue', linewidth=2)
ax4.add_patch(root)
ax4.text(0.5, 0.9, 'Outlook', ha='center', va='center', fontweight='bold', fontsize=10)

# First level
sunny = Circle((0.2, 0.7), 0.06, facecolor='lightblue', edgecolor='blue', linewidth=2)
overcast = Circle((0.5, 0.7), 0.06, facecolor='lightgreen', edgecolor='green', linewidth=2)
rainy = Circle((0.8, 0.7), 0.06, facecolor='lightblue', edgecolor='blue', linewidth=2)
ax4.add_patch(sunny)
ax4.add_patch(overcast)
ax4.add_patch(rainy)
ax4.text(0.2, 0.7, 'Sunny', ha='center', va='center', fontweight='bold', fontsize=9)
ax4.text(0.5, 0.7, 'Overcast', ha='center', va='center', fontweight='bold', fontsize=9)
ax4.text(0.8, 0.7, 'Rainy', ha='center', va='center', fontweight='bold', fontsize=9)

# Second level
sunny_temp = Circle((0.2, 0.5), 0.06, facecolor='lightblue', edgecolor='blue', linewidth=2)
overcast_leaf = Circle((0.5, 0.5), 0.06, facecolor='lightgreen', edgecolor='green', linewidth=2)
rainy_leaf = Circle((0.8, 0.5), 0.06, facecolor='lightgreen', edgecolor='green', linewidth=2)

ax4.add_patch(sunny_temp)
ax4.add_patch(overcast_leaf)
ax4.add_patch(rainy_leaf)

ax4.text(0.2, 0.5, 'Temp', ha='center', va='center', fontweight='bold', fontsize=9)
ax4.text(0.5, 0.5, 'Yes\n(Pure)', ha='center', va='center', fontweight='bold', fontsize=8)
ax4.text(0.8, 0.5, 'Yes\n(Pure)', ha='center', va='center', fontweight='bold', fontsize=8)

# Third level for Sunny branch
sunny_hot = Circle((0.1, 0.3), 0.05, facecolor='lightgreen', edgecolor='green', linewidth=2)
sunny_cool = Circle((0.3, 0.3), 0.05, facecolor='lightgreen', edgecolor='green', linewidth=2)

ax4.add_patch(sunny_hot)
ax4.add_patch(sunny_cool)

ax4.text(0.1, 0.3, 'No\n(Pure)', ha='center', va='center', fontweight='bold', fontsize=8)
ax4.text(0.3, 0.3, 'Yes\n(Pure)', ha='center', va='center', fontweight='bold', fontsize=8)

# Add edges
ax4.plot([0.5, 0.2], [0.82, 0.76], 'k-', linewidth=2)
ax4.plot([0.5, 0.5], [0.82, 0.76], 'k-', linewidth=2)
ax4.plot([0.5, 0.8], [0.82, 0.76], 'k-', linewidth=2)
ax4.plot([0.2, 0.2], [0.64, 0.56], 'k-', linewidth=2)
ax4.plot([0.2, 0.1], [0.44, 0.35], 'k-', linewidth=2)
ax4.plot([0.2, 0.3], [0.44, 0.35], 'k-', linewidth=2)

# Add stopping criteria labels
ax4.text(0.5, 0.76, 'Feature Split', ha='center', va='center', fontsize=10, color='blue')
ax4.text(0.5, 0.56, 'Pure Node\n(Stop)', ha='center', va='center', fontsize=9, color='green')
ax4.text(0.8, 0.56, 'Pure Node\n(Stop)', ha='center', va='center', fontsize=9, color='green')
ax4.text(0.15, 0.36, 'Pure Node\n(Stop)', ha='center', va='center', fontsize=9, color='green')
ax4.text(0.25, 0.36, 'Pure Node\n(Stop)', ha='center', va='center', fontsize=9, color='green')

ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')

plt.tight_layout()
file_path = os.path.join(save_dir, "stopping_criteria_overview.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 3: Why Stopping Criteria Are Important
print_step_header(3, "Why Stopping Criteria Are Important")

print("Stopping criteria are crucial in ID3 for several reasons:")
print()
print("1. PREVENT INFINITE RECURSION:")
print("   - Without stopping criteria, the algorithm would continue indefinitely")
print("   - Each recursive call would create more nodes without end")
print("   - The tree would grow infinitely deep")
print()
print("2. COMPUTATIONAL EFFICIENCY:")
print("   - Stopping criteria ensure the algorithm terminates")
print("   - Prevents unnecessary computation on pure or empty nodes")
print("   - Controls the size and complexity of the resulting tree")
print()
print("3. PREVENT OVERFITTING:")
print("   - Stops tree growth when no more meaningful splits are possible")
print("   - Prevents the tree from memorizing the training data")
print("   - Ensures generalization to unseen data")
print()
print("4. PRACTICAL IMPLEMENTATION:")
print("   - Real-world datasets have finite features and samples")
print("   - Stopping criteria handle edge cases gracefully")
print("   - Ensures robust and reliable tree construction")
print()

# Visualize the importance of stopping criteria
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Importance of Stopping Criteria', fontsize=16, fontweight='bold')

# Without stopping criteria
ax1.set_title('Without Stopping Criteria (BAD)', fontweight='bold', color='red')
ax1.text(0.5, 0.9, 'Infinite Recursion', ha='center', va='center', fontsize=14, color='red')
ax1.text(0.5, 0.8, '→ Tree grows forever', ha='center', va='center', fontsize=11)
ax1.text(0.5, 0.7, '→ Never terminates', ha='center', va='center', fontsize=11)
ax1.text(0.5, 0.6, '→ Infinite memory usage', ha='center', va='center', fontsize=11)
ax1.text(0.5, 0.5, '→ Algorithm crashes', ha='center', va='center', fontsize=11)

# Draw infinite recursion symbol
for i in range(5):
    y_pos = 0.4 - i * 0.05
    node = Circle((0.5, y_pos), 0.03, facecolor='red', edgecolor='darkred', linewidth=1)
    ax1.add_patch(node)
    ax1.text(0.5, y_pos, f'...', ha='center', va='center', fontsize=8)

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')

# With stopping criteria
ax2.set_title('With Stopping Criteria (GOOD)', fontweight='bold', color='green')
ax2.text(0.5, 0.9, 'Controlled Growth', ha='center', va='center', fontsize=14, color='green')
ax2.text(0.5, 0.8, '→ Tree terminates naturally', ha='center', va='center', fontsize=11)
ax2.text(0.5, 0.7, '→ Finite size and depth', ha='center', va='center', fontsize=11)
ax2.text(0.5, 0.6, '→ Efficient computation', ha='center', va='center', fontsize=11)
ax2.text(0.5, 0.5, '→ Generalizable model', ha='center', va='center', fontsize=11)

# Draw controlled tree
root = Circle((0.5, 0.4), 0.06, facecolor='lightgreen', edgecolor='green', linewidth=2)
ax2.add_patch(root)
ax2.text(0.5, 0.4, 'Root', ha='center', va='center', fontweight='bold', fontsize=10)

# Add leaf nodes
leaf1 = Circle((0.3, 0.2), 0.04, facecolor='lightgreen', edgecolor='green', linewidth=2)
leaf2 = Circle((0.7, 0.2), 0.04, facecolor='lightgreen', edgecolor='green', linewidth=2)
ax2.add_patch(leaf1)
ax2.add_patch(leaf2)
ax2.text(0.3, 0.2, 'Leaf', ha='center', va='center', fontweight='bold', fontsize=8)
ax2.text(0.7, 0.2, 'Leaf', ha='center', va='center', fontweight='bold', fontsize=8)

# Add edges
ax2.plot([0.5, 0.3], [0.34, 0.24], 'k-', linewidth=2)
ax2.plot([0.5, 0.7], [0.34, 0.24], 'k-', linewidth=2)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')

plt.tight_layout()
file_path = os.path.join(save_dir, "stopping_criteria_importance.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 4: Handling No Features Remaining
print_step_header(4, "Handling No Features Remaining")

print("What happens when all features have been used?")
print()
print("When all features have been used in a branch, ID3 must make a decision:")
print()
print("1. CHECK IF NODE IS PURE:")
print("   - If all samples belong to the same class → Create leaf node")
print("   - If samples belong to different classes → Use majority class rule")
print()
print("2. MAJORITY CLASS RULE:")
print("   - Count the frequency of each class in the node")
print("   - Assign the most frequent class to the leaf")
print("   - This is a reasonable fallback when no more splits are possible")
print()
print("3. EXAMPLE SCENARIO:")
print("   - Node has 5 samples: 3 Yes, 2 No")
print("   - All features have been used")
print("   - Node is not pure (mixed classes)")
print("   - Solution: Assign class 'Yes' (majority)")
print()

# Create example dataset to demonstrate
example_data = {
    'Outlook': ['Sunny', 'Sunny', 'Rainy', 'Rainy', 'Rainy'],
    'Temperature': ['Hot', 'Hot', 'Mild', 'Cool', 'Cool'],
    'Humidity': ['High', 'High', 'High', 'Normal', 'Normal'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes']
}

print("Example: Node where all features have been used")
print("Features: Outlook, Temperature, Humidity")
print("Samples: 5 samples with mixed classes")
print()
print("Class distribution in this node:")
play_counts = {}
for play in example_data['Play']:
    play_counts[play] = play_counts.get(play, 0) + 1

for play, count in play_counts.items():
    print(f"  {play}: {count}/{len(example_data['Play'])} = {count/len(example_data['Play']):.1%}")

majority_class = max(play_counts, key=play_counts.get)
print(f"\nMajority class: {majority_class}")
print(f"Decision: Create leaf node with class '{majority_class}'")
print()

# Visualize the majority class rule
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_title('Majority Class Rule When No Features Remain', fontsize=16, fontweight='bold')

# Create a tree structure showing this scenario
# Root with all features used
root = Circle((0.5, 0.9), 0.08, facecolor='orange', edgecolor='darkorange', linewidth=2)
ax.add_patch(root)
ax.text(0.5, 0.9, 'All Features\nUsed', ha='center', va='center', fontweight='bold', fontsize=10)

# Show the samples
samples_text = "Samples:\n"
for i, (outlook, temp, humidity, play) in enumerate(zip(example_data['Outlook'], 
                                                       example_data['Temperature'], 
                                                       example_data['Humidity'], 
                                                       example_data['Play'])):
    samples_text += f"{i+1}. {outlook}, {temp}, {humidity} → {play}\n"

ax.text(0.5, 0.7, samples_text, ha='center', va='center', fontsize=10, 
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

# Show class distribution
class_dist_text = f"Class Distribution:\n"
for play, count in play_counts.items():
    class_dist_text += f"{play}: {count}/{len(example_data['Play'])} = {count/len(example_data['Play']):.1%}\n"
class_dist_text += f"\nMajority Class: {majority_class}"

ax.text(0.5, 0.4, class_dist_text, ha='center', va='center', fontsize=11, 
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))

# Show the decision
decision_text = f"Decision:\nCreate leaf node\nwith class '{majority_class}'"
ax.text(0.5, 0.15, decision_text, ha='center', va='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="gold", alpha=0.7))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.tight_layout()
file_path = os.path.join(save_dir, "majority_class_rule.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 5: Edge Case Handling
print_step_header(5, "Edge Case Handling")

print("How do you handle cases where no features remain but the node is not pure?")
print()
print("This is a common scenario in ID3 that requires careful handling:")
print()
print("1. IDENTIFY THE SITUATION:")
print("   - Node contains samples with different classes")
print("   - All available features have been used")
print("   - No more splits are possible")
print()
print("2. EVALUATE OPTIONS:")
print("   - Option A: Use majority class rule (most common approach)")
print("   - Option B: Use weighted average of class probabilities")
print("   - Option C: Use parent node's majority class")
print("   - Option D: Create a probabilistic leaf node")
print()
print("3. RECOMMENDED APPROACH:")
print("   - Use majority class rule for simplicity and interpretability")
print("   - This provides a clear, deterministic classification")
print("   - Alternative approaches can be implemented for specific use cases")
print()

# Visualize edge case handling
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Edge Case Handling Strategies', fontsize=16, fontweight='bold')

# Option A: Majority Class Rule
ax1 = axes[0, 0]
ax1.set_title('Option A: Majority Class Rule', fontweight='bold', color='green')
ax1.text(0.5, 0.8, 'Most Common Approach', ha='center', va='center', fontsize=12)
ax1.text(0.5, 0.7, '→ Count class frequencies', ha='center', va='center', fontsize=11)
ax1.text(0.5, 0.6, '→ Choose most frequent class', ha='center', va='center', fontsize=11)
ax1.text(0.5, 0.5, '→ Simple and interpretable', ha='center', va='center', fontsize=11)

# Draw example
ax1.text(0.5, 0.3, 'Example:', ha='center', va='center', fontsize=11, fontweight='bold')
ax1.text(0.5, 0.25, 'Yes: 3, No: 2', ha='center', va='center', fontsize=10)
ax1.text(0.5, 0.2, '→ Choose "Yes"', ha='center', va='center', fontsize=10, color='green')

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')

# Option B: Weighted Average
ax2 = axes[0, 1]
ax2.set_title('Option B: Weighted Average', fontweight='bold', color='blue')
ax2.text(0.5, 0.8, 'Probabilistic Approach', ha='center', va='center', fontsize=12)
ax2.text(0.5, 0.7, '→ Calculate class probabilities', ha='center', va='center', fontsize=11)
ax2.text(0.5, 0.6, '→ Use as confidence scores', ha='center', va='center', fontsize=11)
ax2.text(0.5, 0.5, '→ More nuanced but complex', ha='center', va='center', fontsize=11)

# Draw example
ax2.text(0.5, 0.3, 'Example:', ha='center', va='center', fontsize=11, fontweight='bold')
ax2.text(0.5, 0.25, 'P(Yes) = 3/5 = 0.6', ha='center', va='center', fontsize=10)
ax2.text(0.5, 0.2, 'P(No) = 2/5 = 0.4', ha='center', va='center', fontsize=10)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')

# Option C: Parent Node Class
ax3 = axes[1, 0]
ax3.set_title('Option C: Parent Node Class', fontweight='bold', color='orange')
ax3.text(0.5, 0.8, 'Inheritance Approach', ha='center', va='center', fontsize=12)
ax3.text(0.5, 0.7, '→ Use parent node majority', ha='center', va='center', fontsize=11)
ax3.text(0.5, 0.6, '→ Maintain consistency', ha='center', va='center', fontsize=11)
ax3.text(0.5, 0.5, '→ May not be optimal', ha='center', va='center', fontsize=11)

# Draw example
ax3.text(0.5, 0.3, 'Example:', ha='center', va='center', fontsize=11, fontweight='bold')
ax3.text(0.5, 0.25, 'Parent: Yes (60%)', ha='center', va='center', fontsize=10)
ax3.text(0.5, 0.2, '→ Use "Yes"', ha='center', va='center', fontsize=10, color='orange')

ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.axis('off')

# Option D: Probabilistic Leaf
ax4 = axes[1, 1]
ax4.set_title('Option D: Probabilistic Leaf', fontweight='bold', color='purple')
ax4.text(0.5, 0.8, 'Advanced Approach', ha='center', va='center', fontsize=12)
ax4.text(0.5, 0.7, '→ Store class probabilities', ha='center', va='center', fontsize=11)
ax4.text(0.5, 0.6, '→ Use for ensemble methods', ha='center', va='center', fontsize=11)
ax4.text(0.5, 0.5, '→ Most flexible but complex', ha='center', va='center', fontsize=11)

# Draw example
ax4.text(0.5, 0.3, 'Example:', ha='center', va='center', fontsize=11, fontweight='bold')
ax4.text(0.5, 0.25, 'Leaf: [0.6, 0.4]', ha='center', va='center', fontsize=10)
ax4.text(0.5, 0.2, '→ Probabilistic output', ha='center', va='center', fontsize=10, color='purple')

ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')

plt.tight_layout()
file_path = os.path.join(save_dir, "edge_case_handling.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 6: Summary and Key Insights
print_step_header(6, "Summary and Key Insights")

print("Question 3 Summary:")
print("=" * 50)
print()
print("1. Three Main Stopping Criteria:")
print("   ✓ Pure node (all samples same class)")
print("   ✓ No features remaining")
print("   ✓ Empty dataset after split")
print()
print("2. Importance of Stopping Criteria:")
print("   ✓ Prevent infinite recursion")
print("   ✓ Ensure computational efficiency")
print("   ✓ Prevent overfitting")
print("   ✓ Handle edge cases gracefully")
print()
print("3. Handling No Features Remaining:")
print("   ✓ Check if node is pure")
print("   ✓ Use majority class rule if mixed")
print("   ✓ Create deterministic leaf node")
print()
print("4. Edge Case Handling:")
print("   ✓ Multiple strategies available")
print("   ✓ Majority class rule is standard")
print("   ✓ Consider application requirements")
print()

print("Key Insights:")
print("- Stopping criteria are essential for ID3 to work properly")
print("- They prevent infinite recursion and ensure termination")
print("- The majority class rule handles mixed nodes gracefully")
print("- Different handling strategies offer trade-offs")
print()

print("All figures have been saved to:", save_dir)
print("Understanding stopping criteria is crucial for implementing ID3")
print("and ensuring robust decision tree construction.")
