import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_25")
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

print("Question 25: ID3's Behavior When All Features Are Exhausted")
print("We need to understand how ID3 handles the scenario where all features have been used")
print("but some nodes remain impure, and how to assign class labels in such cases.")
print()
print("Tasks:")
print("1. Describe the scenario where ID3 exhausts all features but has impure nodes")
print("2. Analyze the given partially constructed tree")
print("3. Explain how ID3 should handle impure nodes")
print("4. Determine the decision rule for leaf node class assignment")
print("5. Calculate entropy and determine majority class rules")
print()

# Step 2: Understanding the Scenario
print_step_header(2, "Understanding the Scenario")

print("The scenario occurs when:")
print("1. ID3 has used all available features for splitting")
print("2. Some leaf nodes still contain samples from multiple classes")
print("3. No more features are available to further split these impure nodes")
print("4. The algorithm must make a decision about class assignment")
print()
print("This is a common situation in real-world datasets where:")
print("- Features have limited discriminatory power")
print("- Data is noisy or contains conflicting examples")
print("- The problem is inherently non-linearly separable")
print("- Feature space is insufficient for perfect separation")
print()

# Step 3: Given Tree Analysis
print_step_header(3, "Given Tree Analysis")

print("Partially constructed tree where all features are used:")
print()
print("Root: Outlook")
print("├── Sunny → [Yes: 2, No: 3]")
print("├── Cloudy → [Yes: 4, No: 0]")
print("└── Rain → [Yes: 1, No: 2]")
print()
print("Analysis:")
print("- Total samples: 12")
print("- Sunny branch: 5 samples (2 Yes, 3 No) - IMPURE")
print("- Cloudy branch: 4 samples (4 Yes, 0 No) - PURE")
print("- Rain branch: 3 samples (1 Yes, 2 No) - IMPURE")
print()
print("Features used: Outlook (all possible values exhausted)")
print("No more features available for further splitting")
print()

# Step 4: How ID3 Should Handle Impure Nodes
print_step_header(4, "How ID3 Should Handle Impure Nodes")

print("When ID3 exhausts all features but has impure nodes:")
print()
print("1. Stop further splitting (no more features available)")
print("2. Convert impure nodes to leaf nodes")
print("3. Assign class labels using majority voting")
print("4. Use the most frequent class as the prediction")
print()
print("This is a stopping criterion in ID3:")
print("- If A is empty (no more attributes):")
print("  Return a leaf node labeled with the majority class in D")
print()
print("The algorithm cannot continue building the tree")
print("and must make the best possible prediction with available information")
print()

# Step 5: Decision Rule for Leaf Node Class Assignment
print_step_header(5, "Decision Rule for Leaf Node Class Assignment")

print("Decision Rule: Majority Class Assignment")
print()
print("For each impure leaf node:")
print("1. Count the number of samples from each class")
print("2. Identify the class with the highest count")
print("3. Assign that class as the leaf node's prediction")
print("4. In case of ties, use a tie-breaking strategy")
print()
print("Mathematical formulation:")
print("Class(leaf) = argmax_c Count(c) for c ∈ Classes")
print("where Count(c) is the number of samples of class c in the leaf")
print()

# Step 6: Entropy Calculation for Each Leaf
print_step_header(6, "Entropy Calculation for Each Leaf")

print("Step 1: Calculate entropy for each impure leaf node")
print("H(S) = -Σ p(i) * log2(p(i))")
print()

# Sunny branch: [Yes: 2, No: 3]
print("Sunny branch: [Yes: 2, No: 3]")
sunny_total = 2 + 3
p_yes_sunny = 2 / sunny_total
p_no_sunny = 3 / sunny_total

print(f"P(Yes|Sunny) = {2}/{sunny_total} = {p_yes_sunny:.3f}")
print(f"P(No|Sunny) = {3}/{sunny_total} = {p_no_sunny:.3f}")

import math
H_sunny = -p_yes_sunny * math.log2(p_yes_sunny) - p_no_sunny * math.log2(p_no_sunny)
print(f"H(Sunny) = -{p_yes_sunny:.3f} * log2({p_yes_sunny:.3f}) - {p_no_sunny:.3f} * log2({p_no_sunny:.3f})")
print(f"H(Sunny) = -{p_yes_sunny:.3f} * {math.log2(p_yes_sunny):.3f} - {p_no_sunny:.3f} * {math.log2(p_no_sunny):.3f}")
print(f"H(Sunny) = {H_sunny:.3f}")
print()

# Cloudy branch: [Yes: 4, No: 0] - Pure node
print("Cloudy branch: [Yes: 4, No: 0] - Pure node")
print("P(Yes|Cloudy) = 4/4 = 1.0")
print("P(No|Cloudy) = 0/4 = 0.0")
print("H(Cloudy) = -1.0 * log2(1.0) - 0.0 * log2(0.0) = 0")
print("This is a pure node with zero entropy")
print()

# Rain branch: [Yes: 1, No: 2]
print("Rain branch: [Yes: 1, No: 2]")
rain_total = 1 + 2
p_yes_rain = 1 / rain_total
p_no_rain = 2 / rain_total

print(f"P(Yes|Rain) = {1}/{rain_total} = {p_yes_rain:.3f}")
print(f"P(No|Rain) = {2}/{rain_total} = {p_no_rain:.3f}")

H_rain = -p_yes_rain * math.log2(p_yes_rain) - p_no_rain * math.log2(p_no_rain)
print(f"H(Rain) = -{p_yes_rain:.3f} * log2({p_yes_rain:.3f}) - {p_no_rain:.3f} * log2({p_no_rain:.3f})")
print(f"H(Rain) = -{p_yes_rain:.3f} * {math.log2(p_yes_rain):.3f} - {p_no_rain:.3f} * {math.log2(p_no_rain):.3f}")
print(f"H(Rain) = {H_rain:.3f}")
print()

# Step 7: Majority Class Assignment
print_step_header(7, "Majority Class Assignment")

print("Step 2: Determine majority class for each impure leaf")
print()

print("Sunny branch: [Yes: 2, No: 3]")
print(f"  - Yes count: 2")
print(f"  - No count: 3")
print(f"  - Majority class: No (3 > 2)")
print(f"  - Decision rule: If Outlook = Sunny, predict No")
print()

print("Rain branch: [Yes: 1, No: 2]")
print(f"  - Yes count: 1")
print(f"  - No count: 2")
print(f"  - Majority class: No (2 > 1)")
print(f"  - Decision rule: If Outlook = Rain, predict No")
print()

print("Cloudy branch: [Yes: 4, No: 0]")
print(f"  - Yes count: 4")
print(f"  - No count: 0")
print(f"  - Pure node: All samples are Yes")
print(f"  - Decision rule: If Outlook = Cloudy, predict Yes")
print()

# Step 8: Final Tree Structure
print_step_header(8, "Final Tree Structure")

print("Final decision tree after handling impure nodes:")
print()
print("Root: Outlook")
print("├── Sunny → Predict: No (majority: 3/5)")
print("├── Cloudy → Predict: Yes (pure: 4/4)")
print("└── Rain → Predict: No (majority: 2/3)")
print()
print("Complete decision rules:")
print("1. If Outlook = Sunny, predict No")
print("2. If Outlook = Cloudy, predict Yes")
print("3. If Outlook = Rain, predict No")
print()
print("Note: The tree cannot be further refined due to feature exhaustion")
print()

# Step 9: Visualizing the Decision Tree
print_step_header(9, "Visualizing the Decision Tree")

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
draw_node(5, 9, 'Outlook', 'lightgreen', 1.2)

# Branches
draw_arrow((5, 8.4), (2, 7.5))
draw_arrow((5, 8.4), (5, 7.5))
draw_arrow((5, 8.4), (8, 7.5))

# Leaf nodes
draw_node(2, 7, 'Sunny\n[Yes: 2, No: 3]\nH=0.971\nPredict: No', 'lightcoral')
draw_node(5, 7, 'Cloudy\n[Yes: 4, No: 0]\nH=0.000\nPredict: Yes', 'lightgreen')
draw_node(8, 7, 'Rain\n[Yes: 1, No: 2]\nH=0.918\nPredict: No', 'lightcoral')

# Add branch labels
ax.text(3.5, 8, 'Sunny', ha='center', va='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", alpha=0.8))
ax.text(5, 8, 'Cloudy', ha='center', va='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", alpha=0.8))
ax.text(6.5, 8, 'Rain', ha='center', va='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", alpha=0.8))

# Add explanation
ax.text(5, 5, 'Feature Exhaustion: No more features available for splitting', 
        ha='center', va='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", ec="black", alpha=0.8))

ax.text(5, 4, 'Impure nodes use majority class voting for predictions', 
        ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="black", alpha=0.8))

plt.savefig(os.path.join(save_dir, 'feature_exhaustion_tree.png'), dpi=300, bbox_inches='tight')

# Step 10: Entropy and Information Visualization
print_step_header(10, "Entropy and Information Visualization")

# Create separate plots for each aspect of entropy and majority class assignment

# Plot 1: Class distribution in each branch
fig, ax1 = plt.subplots(figsize=(8, 6))
ax1.set_xlabel('Outlook Value', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('Class Distribution by Outlook Value', fontsize=14, fontweight='bold')

outlook_values = ['Sunny', 'Cloudy', 'Rain']
yes_counts = [2, 4, 1]
no_counts = [3, 0, 2]

x = np.arange(len(outlook_values))
width = 0.35

bars1 = ax1.bar(x - width/2, yes_counts, width, label='Yes', color='green', alpha=0.7)
bars2 = ax1.bar(x + width/2, no_counts, width, label='No', color='red', alpha=0.7)

ax1.set_xticks(x)
ax1.set_xticklabels(outlook_values)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                     f'{int(height)}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Entropy comparison
fig, ax2 = plt.subplots(figsize=(8, 6))
ax2.set_xlabel('Outlook Value', fontsize=12)
ax2.set_ylabel('Entropy', fontsize=12)
ax2.set_title('Entropy by Outlook Value', fontsize=14, fontweight='bold')

entropies = [H_sunny, 0, H_rain]  # Cloudy has 0 entropy (pure node)
colors = ['red', 'green', 'red']  # Red for impure, green for pure

bars = ax2.bar(outlook_values, entropies, color=colors, alpha=0.7)
ax2.set_ylim(0, 1.1)
ax2.grid(True, alpha=0.3)

# Add entropy values
for bar, entropy in zip(bars, entropies):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{entropy:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'entropy_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Majority class assignment
fig, ax3 = plt.subplots(figsize=(8, 6))
ax3.set_xlabel('Outlook Value', fontsize=12)
ax3.set_ylabel('Prediction', fontsize=12)
ax3.set_title('Majority Class Assignment', fontsize=14, fontweight='bold')

predictions = ['No', 'Yes', 'No']
prediction_colors = ['red', 'green', 'red']

bars = ax3.bar(outlook_values, [1, 1, 1], color=prediction_colors, alpha=0.7)
ax3.set_ylim(0, 1.5)
ax3.set_yticks([])

# Add prediction labels
for bar, pred in zip(bars, predictions):
    ax3.text(bar.get_x() + bar.get_width()/2., 0.5,
             pred, ha='center', va='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'majority_class_assignment.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Feature exhaustion explanation
fig, ax4 = plt.subplots(figsize=(8, 6))
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')

# Draw a simple diagram showing feature exhaustion
ax4.text(5, 8, 'Feature Space Exhaustion', fontsize=14, fontweight='bold')

# Draw feature space
ax4.add_patch(Rectangle((1, 4), 8, 3, facecolor='lightblue', alpha=0.3, edgecolor='black'))
ax4.text(5, 6.5, 'All features used\n(Outlook exhausted)', ha='center', va='center', fontsize=12)

# Draw impure regions
ax4.add_patch(Rectangle((2, 2), 2, 1.5, facecolor='lightcoral', alpha=0.7, edgecolor='red'))
ax4.text(3, 2.75, 'Sunny\n[Yes:2, No:3]', ha='center', va='center', fontsize=10, fontweight='bold')

ax4.add_patch(Rectangle((6, 2), 2, 1.5, facecolor='lightcoral', alpha=0.7, edgecolor='red'))
ax4.text(7, 2.75, 'Rain\n[Yes:1, No:2]', ha='center', va='center', fontsize=10, fontweight='bold')

ax4.add_patch(Rectangle((4, 2), 2, 1.5, facecolor='lightgreen', alpha=0.7, edgecolor='green'))
ax4.text(5, 2.75, 'Cloudy\n[Yes:4, No:0]', ha='center', va='center', fontsize=10, fontweight='bold')

ax4.text(5, 1, 'No more features available for further splitting', ha='center', va='center', fontsize=10,
         bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_space_exhaustion.png'), dpi=300, bbox_inches='tight')
plt.close()

# Step 11: Key Insights
print_step_header(11, "Key Insights")

print("1. Feature Exhaustion:")
print("   - Occurs when all available features have been used")
print("   - Impure nodes cannot be further split")
print("   - Tree construction must stop")
print()
print("2. Handling Impure Nodes:")
print("   - Convert to leaf nodes with majority class assignment")
print("   - Use voting mechanism for class prediction")
print("   - Accept some impurity as inevitable")
print()
print("3. Practical Implications:")
print("   - Trees may not achieve perfect classification")
print("   - Model performance depends on feature quality")
print("   - Feature engineering becomes crucial")
print()

# Step 12: Final Answer
print_step_header(12, "Final Answer")

print("1. Scenario description:")
print("   ID3 exhausts all features when no more attributes are available")
print("   for splitting, leaving some nodes impure with mixed class samples.")
print()
print("2. Given tree analysis:")
print("   - Sunny: [Yes:2, No:3] - impure, entropy = 0.971")
print("   - Cloudy: [Yes:4, No:0] - pure, entropy = 0.000")
print("   - Rain: [Yes:1, No:2] - impure, entropy = 0.918")
print()
print("3. Handling impure nodes:")
print("   - Stop further splitting (no features available)")
print("   - Convert impure nodes to leaf nodes")
print("   - Use majority class voting for predictions")
print()
print("4. Decision rule:")
print("   - Sunny → Predict No (majority: 3/5)")
print("   - Cloudy → Predict Yes (pure: 4/4)")
print("   - Rain → Predict No (majority: 2/3)")
print()
print("5. Entropy and majority class:")
print("   - Sunny: H = 0.971, majority = No")
print("   - Cloudy: H = 0.000, majority = Yes")
print("   - Rain: H = 0.918, majority = No")
print()

print(f"\nVisualizations saved to: {save_dir}")
print("The plots show the feature exhaustion scenario and majority class assignment.")
