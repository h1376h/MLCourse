import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.patches import Circle, Arrow
import matplotlib.patches as mpatches

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_1")
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

print("Question 1: ID3 Algorithm Overview")
print("The ID3 algorithm follows a recursive approach to build decision trees.")
print()
print("Tasks:")
print("1. What are the main steps of the ID3 algorithm?")
print("2. How does ID3 choose the best feature for splitting at each node?")
print("3. What is the base case for stopping recursion?")
print("4. Why is ID3 considered a greedy algorithm?")
print()

# Step 2: Main Steps of ID3 Algorithm
print_step_header(2, "Main Steps of ID3 Algorithm")

print("The ID3 algorithm consists of the following main steps:")
print()
print("1. START: Begin with the entire dataset at the root node")
print("2. CHECK STOPPING CRITERIA: Determine if we should stop growing the tree")
print("3. FEATURE SELECTION: Calculate information gain for all available features")
print("4. BEST SPLIT: Choose the feature with the highest information gain")
print("5. CREATE CHILDREN: Split the dataset based on the chosen feature")
print("6. RECURSION: Recursively apply steps 2-6 to each child node")
print("7. TERMINATE: When stopping criteria are met, create a leaf node")
print()

# Visualize the ID3 algorithm flow
fig, ax = plt.subplots(figsize=(14, 10))

# Define the flow steps
steps = [
    "START\n(Entire Dataset)",
    "Check\nStopping Criteria",
    "Calculate\nInformation Gain",
    "Choose Best\nFeature",
    "Split Dataset\n(Create Children)",
    "Recursive\nCall",
    "Create\nLeaf Node"
]

# Define positions for each step
positions = [
    (0.5, 0.9),   # START
    (0.2, 0.7),   # Check Stopping
    (0.4, 0.7),   # Calculate IG
    (0.6, 0.7),   # Choose Feature
    (0.8, 0.7),   # Split Dataset
    (0.5, 0.5),   # Recursive Call
    (0.5, 0.3)    # Create Leaf
]

# Draw the flow diagram
for i, (step, pos) in enumerate(zip(steps, positions)):
    # Create a rounded rectangle for each step
    box = FancyBboxPatch(
        (pos[0] - 0.08, pos[1] - 0.05),
        0.16, 0.1,
        boxstyle="round,pad=0.02",
        facecolor='lightblue',
        edgecolor='navy',
        linewidth=2
    )
    ax.add_patch(box)
    
    # Add text
    ax.text(pos[0], pos[1], step, ha='center', va='center', 
            fontsize=10, fontweight='bold', wrap=True)

# Add arrows between steps
arrows = [
    ((0.5, 0.85), (0.2, 0.75)),  # START to Check Stopping
    ((0.28, 0.7), (0.32, 0.7)),  # Check to Calculate
    ((0.48, 0.7), (0.52, 0.7)),  # Calculate to Choose
    ((0.68, 0.7), (0.72, 0.7)),  # Choose to Split
    ((0.8, 0.65), (0.5, 0.55)),  # Split to Recursive
    ((0.5, 0.45), (0.5, 0.35))   # Recursive to Leaf
]

for start, end in arrows:
    arrow = Arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                  width=0.02, color='red', alpha=0.7)
    ax.add_patch(arrow)

# Add decision diamond for stopping criteria
diamond = mpatches.RegularPolygon((0.2, 0.7), 4, radius=0.06, 
                                  orientation=np.pi/4, facecolor='yellow', 
                                  edgecolor='orange', linewidth=2)
ax.add_patch(diamond)

# Add text for decision
ax.text(0.2, 0.7, "Stop?", ha='center', va='center', fontsize=9, fontweight='bold')

# Add yes/no labels
ax.text(0.15, 0.65, "YES", ha='center', va='center', fontsize=8, color='green')
ax.text(0.25, 0.75, "NO", ha='center', va='center', fontsize=8, color='red')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('ID3 Algorithm Flow Diagram', fontsize=16, fontweight='bold')
ax.axis('off')

plt.tight_layout()
file_path = os.path.join(save_dir, "ID3_algorithm_flow.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 3: Feature Selection Process
print_step_header(3, "Feature Selection Process")

print("How ID3 chooses the best feature for splitting:")
print()
print("1. For each available feature, calculate the information gain")
print("2. Information Gain = H(S) - H(S|A)")
print("   where H(S) is the entropy of the current dataset")
print("   and H(S|A) is the conditional entropy given feature A")
print("3. Choose the feature with the highest information gain")
print("4. This maximizes the reduction in uncertainty")
print()

# Create a sample dataset to demonstrate feature selection
print("Example: Weather dataset for playing tennis")
print("Features: Outlook, Temperature, Humidity, Windy")
print("Target: Play (Yes/No)")
print()

# Sample data for demonstration
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal'],
    'Windy': [False, True, False, False, False, True, True],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes']
}

print("Sample data:")
for i in range(len(data['Outlook'])):
    print(f"  {data['Outlook'][i]:<10} {data['Temperature'][i]:<10} {data['Humidity'][i]:<10} {str(data['Windy'][i]):<5} → {data['Play'][i]}")

# Calculate entropy for the target variable
def entropy(probabilities):
    """Calculate entropy given a list of probabilities."""
    h = 0
    for p in probabilities:
        if p > 0:
            h -= p * np.log2(p)
    return h

# Calculate class distribution
play_counts = {}
for play in data['Play']:
    play_counts[play] = play_counts.get(play, 0) + 1

total_samples = len(data['Play'])
p_yes = play_counts['Yes'] / total_samples
p_no = play_counts['No'] / total_samples

print(f"\nClass distribution:")
print(f"  Play = Yes: {play_counts['Yes']}/{total_samples} = {p_yes:.3f}")
print(f"  Play = No:  {play_counts['No']}/{total_samples} = {p_no:.3f}")

# Calculate entropy of the target variable
h_target = entropy([p_yes, p_no])
print(f"\nEntropy of target variable (Play):")
print(f"  H(Play) = -{p_yes:.3f} × log₂({p_yes:.3f}) - {p_no:.3f} × log₂({p_no:.3f})")
print(f"  H(Play) = {h_target:.4f} bits")

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
ax2.set_ylabel('-P(Play) × log₂(P(Play))')
ax2.set_ylim(0, max(entropy_terms) * 1.1)

# Add value labels on bars
for bar, term in zip(bars, entropy_terms):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{term:.4f}', ha='center', va='bottom', fontweight='bold')

# Add total entropy line
ax2.axhline(y=h_target, color='blue', linestyle='--', alpha=0.7,
           label=f'Total Entropy = {h_target:.4f} bits')
ax2.legend()

plt.tight_layout()
file_path = os.path.join(save_dir, "entropy_calculation.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 4: Information Gain Calculation
print_step_header(4, "Information Gain Calculation")

print("Now let's calculate information gain for each feature:")
print()

# Function to calculate information gain for a feature
def calculate_information_gain(data, feature_name, target_name='Play'):
    """Calculate information gain for a given feature."""
    # Get unique values of the feature
    feature_values = list(set(data[feature_name]))
    
    # Calculate conditional entropy
    conditional_entropy = 0
    total_samples = len(data[target_name])
    
    print(f"Feature: {feature_name}")
    print(f"  Unique values: {feature_values}")
    
    for value in feature_values:
        # Filter data for this feature value
        subset_indices = [i for i, v in enumerate(data[feature_name]) if v == value]
        subset_size = len(subset_indices)
        
        if subset_size == 0:
            continue
            
        # Calculate class distribution in this subset
        subset_plays = [data[target_name][i] for i in subset_indices]
        subset_counts = {}
        for play in subset_plays:
            subset_counts[play] = subset_counts.get(play, 0) + 1
        
        # Calculate probabilities
        subset_probs = [subset_counts.get('Yes', 0) / subset_size, 
                       subset_counts.get('No', 0) / subset_size]
        
        # Calculate entropy for this subset
        subset_entropy = entropy(subset_probs)
        
        # Weight by proportion of samples
        weight = subset_size / total_samples
        conditional_entropy += weight * subset_entropy
        
        print(f"    {value}: {subset_counts} → P(Yes)={subset_probs[0]:.3f}, P(No)={subset_probs[1]:.3f}")
        print(f"         H(Play|{feature_name}={value}) = {subset_entropy:.4f}")
        print(f"         Weight = {subset_size}/{total_samples} = {weight:.3f}")
        print(f"         Contribution = {weight:.3f} × {subset_entropy:.4f} = {weight * subset_entropy:.4f}")
    
    print(f"  H(Play|{feature_name}) = {conditional_entropy:.4f}")
    
    # Calculate information gain
    info_gain = h_target - conditional_entropy
    print(f"  Information Gain = H(Play) - H(Play|{feature_name})")
    print(f"  Information Gain = {h_target:.4f} - {conditional_entropy:.4f} = {info_gain:.4f}")
    print()
    
    return info_gain

# Calculate information gain for each feature
features = ['Outlook', 'Temperature', 'Humidity', 'Windy']
info_gains = {}

for feature in features:
    info_gains[feature] = calculate_information_gain(data, feature)

# Visualize information gains
fig, ax = plt.subplots(figsize=(12, 8))

# Sort features by information gain
sorted_features = sorted(info_gains.items(), key=lambda x: x[1], reverse=True)
feature_names = [f[0] for f in sorted_features]
gains = [f[1] for f in sorted_features]

# Create horizontal bar chart
bars = ax.barh(feature_names, gains, color=['gold', 'silver', 'brown', 'lightblue'])

# Add value labels
for i, (bar, gain) in enumerate(zip(bars, gains)):
    width = bar.get_width()
    ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
            f'{gain:.4f}', ha='left', va='center', fontweight='bold')

# Add a vertical line for the best feature
best_gain = max(gains)
ax.axvline(x=best_gain, color='red', linestyle='--', alpha=0.7,
           label=f'Best Feature: {feature_names[0]} (IG = {best_gain:.4f})')

ax.set_xlabel('Information Gain (bits)')
ax.set_title('Information Gain for Each Feature')
ax.set_xlim(0, max(gains) * 1.1)
ax.legend()

# Highlight the best feature
bars[0].set_color('red')
bars[0].set_alpha(0.8)

plt.tight_layout()
file_path = os.path.join(save_dir, "information_gain_comparison.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 5: Stopping Criteria
print_step_header(5, "Stopping Criteria")

print("The three main stopping criteria in ID3:")
print()
print("1. PURE NODE: All samples belong to the same class")
print("2. NO FEATURES: All features have been used")
print("3. EMPTY DATASET: No samples remain after splitting")
print()

# Visualize stopping criteria
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('ID3 Stopping Criteria', fontsize=16, fontweight='bold')

# 1. Pure Node
ax1 = axes[0, 0]
ax1.set_title('1. Pure Node - All samples same class', fontweight='bold')
ax1.text(0.5, 0.7, 'Node contains only "Yes" samples', ha='center', va='center', fontsize=12)
ax1.text(0.5, 0.5, '→ Create leaf node with class "Yes"', ha='center', va='center', fontsize=11)
ax1.text(0.5, 0.3, '→ Stop recursion', ha='center', va='center', fontsize=11, color='green')

# Draw a leaf node
leaf1 = Circle((0.5, 0.1), 0.08, facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
ax1.add_patch(leaf1)
ax1.text(0.5, 0.1, 'Yes', ha='center', va='center', fontweight='bold')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')

# 2. No Features
ax2 = axes[0, 1]
ax2.set_title('2. No Features - All features used', fontweight='bold')
ax2.text(0.5, 0.7, 'Features used: Outlook, Temperature, Humidity, Windy', ha='center', va='center', fontsize=11)
ax2.text(0.5, 0.5, '→ No more features to split on', ha='center', va='center', fontsize=11)
ax2.text(0.5, 0.3, '→ Use majority class rule', ha='center', va='center', fontsize=11, color='orange')

# Draw a leaf node
leaf2 = Circle((0.5, 0.1), 0.08, facecolor='orange', edgecolor='darkorange', linewidth=2)
ax2.add_patch(leaf2)
ax2.text(0.5, 0.1, 'Majority', ha='center', va='center', fontweight='bold')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')

# 3. Empty Dataset
ax3 = axes[1, 0]
ax3.set_title('3. Empty Dataset - No samples after split', fontweight='bold')
ax3.text(0.5, 0.7, 'Split resulted in empty branch', ha='center', va='center', fontsize=11)
ax3.text(0.5, 0.5, '→ No samples to classify', ha='center', va='center', fontsize=11)
ax3.text(0.5, 0.3, '→ Use parent node majority class', ha='center', va='center', fontsize=11, color='red')

# Draw an empty node
empty_node = Circle((0.5, 0.1), 0.08, facecolor='lightcoral', edgecolor='red', linewidth=2)
ax3.add_patch(empty_node)
ax3.text(0.5, 0.1, 'Empty', ha='center', va='center', fontweight='bold')
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.axis('off')

# 4. Decision Tree Example
ax4 = axes[1, 1]
ax4.set_title('4. Complete Decision Tree Example', fontweight='bold')

# Draw a simple tree structure
# Root
root = Circle((0.5, 0.9), 0.08, facecolor='lightblue', edgecolor='blue', linewidth=2)
ax4.add_patch(root)
ax4.text(0.5, 0.9, 'Outlook', ha='center', va='center', fontweight='bold', fontsize=10)

# First level
sunny = Circle((0.2, 0.6), 0.06, facecolor='lightblue', edgecolor='blue', linewidth=2)
overcast = Circle((0.5, 0.6), 0.06, facecolor='lightblue', edgecolor='blue', linewidth=2)
rainy = Circle((0.8, 0.6), 0.06, facecolor='lightblue', edgecolor='blue', linewidth=2)
ax4.add_patch(sunny)
ax4.add_patch(overcast)
ax4.add_patch(rainy)
ax4.text(0.2, 0.6, 'Sunny', ha='center', va='center', fontweight='bold', fontsize=9)
ax4.text(0.5, 0.6, 'Overcast', ha='center', va='center', fontweight='bold', fontsize=9)
ax4.text(0.8, 0.6, 'Rainy', ha='center', va='center', fontweight='bold', fontsize=9)

# Second level
sunny_temp = Circle((0.2, 0.3), 0.06, facecolor='lightblue', edgecolor='blue', linewidth=2)
sunny_leaf = Circle((0.2, 0.1), 0.06, facecolor='lightgreen', edgecolor='green', linewidth=2)
overcast_leaf = Circle((0.5, 0.3), 0.06, facecolor='lightgreen', edgecolor='green', linewidth=2)
rainy_leaf = Circle((0.8, 0.3), 0.06, facecolor='lightgreen', edgecolor='green', linewidth=2)

ax4.add_patch(sunny_temp)
ax4.add_patch(sunny_leaf)
ax4.add_patch(overcast_leaf)
ax4.add_patch(rainy_leaf)

ax4.text(0.2, 0.3, 'Temp', ha='center', va='center', fontweight='bold', fontsize=9)
ax4.text(0.2, 0.1, 'Yes', ha='center', va='center', fontweight='bold', fontsize=9)
ax4.text(0.5, 0.3, 'Yes', ha='center', va='center', fontweight='bold', fontsize=9)
ax4.text(0.8, 0.3, 'Yes', ha='center', va='center', fontweight='bold', fontsize=9)

# Add edges
ax4.plot([0.5, 0.2], [0.82, 0.66], 'k-', linewidth=2)
ax4.plot([0.5, 0.5], [0.82, 0.66], 'k-', linewidth=2)
ax4.plot([0.5, 0.8], [0.82, 0.66], 'k-', linewidth=2)
ax4.plot([0.2, 0.2], [0.54, 0.36], 'k-', linewidth=2)

ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')

plt.tight_layout()
file_path = os.path.join(save_dir, "stopping_criteria.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 6: Why ID3 is Greedy
print_step_header(6, "Why ID3 is Considered a Greedy Algorithm")

print("ID3 is considered a greedy algorithm because:")
print()
print("1. LOCAL OPTIMIZATION: At each node, it chooses the feature with the highest")
print("   information gain without considering the global tree structure")
print()
print("2. NO BACKTRACKING: Once a split is made, it cannot be undone or reconsidered")
print("   even if a different choice might lead to a better overall tree")
print()
print("3. IMMEDIATE REWARD: It maximizes information gain at the current step")
print("   rather than considering long-term tree quality")
print()
print("4. SUBOPTIMAL SOLUTIONS: The greedy approach may lead to suboptimal trees")
print("   compared to considering all possible tree structures")
print()

# Visualize greedy vs optimal approach
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Greedy vs Optimal Approach in ID3', fontsize=16, fontweight='bold')

# Greedy Approach
ax1.set_title('Greedy Approach (ID3)', fontweight='bold', color='red')
ax1.text(0.5, 0.9, 'Step 1: Choose best feature at root', ha='center', va='center', fontsize=11)
ax1.text(0.5, 0.8, '→ Outlook (IG = 0.247)', ha='center', va='center', fontsize=10, color='red')

ax1.text(0.5, 0.7, 'Step 2: Choose best feature at each child', ha='center', va='center', fontsize=11)
ax1.text(0.5, 0.6, '→ Temperature for Sunny branch', ha='center', va='center', fontsize=10, color='red')
ax1.text(0.5, 0.5, '→ No more features for Overcast', ha='center', va='center', fontsize=10, color='red')
ax1.text(0.5, 0.4, '→ No more features for Rainy', ha='center', va='center', fontsize=10, color='red')

ax1.text(0.5, 0.2, 'Result: Locally optimal at each step', ha='center', va='center', fontsize=11, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))

# Draw a simple greedy tree
root_greedy = Circle((0.5, 0.1), 0.06, facecolor='lightcoral', edgecolor='red', linewidth=2)
ax1.add_patch(root_greedy)
ax1.text(0.5, 0.1, 'Outlook', ha='center', va='center', fontweight='bold', fontsize=9)

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')

# Optimal Approach
ax2.set_title('Optimal Approach (Theoretical)', fontweight='bold', color='green')
ax2.text(0.5, 0.9, 'Step 1: Consider all possible tree structures', ha='center', va='center', fontsize=11)
ax2.text(0.5, 0.8, '→ Evaluate complete trees', ha='center', va='center', fontsize=10, color='green')

ax2.text(0.5, 0.7, 'Step 2: Choose the globally optimal tree', ha='center', va='center', fontsize=11)
ax2.text(0.5, 0.6, '→ Consider interactions between splits', ha='center', va='center', fontsize=10, color='green')
ax2.text(0.5, 0.5, '→ Balance depth vs accuracy', ha='center', va='center', fontsize=10, color='green')

ax2.text(0.5, 0.2, 'Result: Globally optimal tree structure', ha='center', va='center', fontsize=11,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))

# Draw a theoretical optimal tree
root_optimal = Circle((0.5, 0.1), 0.06, facecolor='lightgreen', edgecolor='green', linewidth=2)
ax2.add_patch(root_optimal)
ax2.text(0.5, 0.1, 'Optimal', ha='center', va='center', fontweight='bold', fontsize=9)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')

plt.tight_layout()
file_path = os.path.join(save_dir, "greedy_vs_optimal.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 7: Complete ID3 Example
print_step_header(7, "Complete ID3 Example")

print("Let's walk through a complete ID3 example:")
print()

# Create a more detailed dataset
detailed_data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Windy': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

print("Complete dataset:")
print("Outlook    Temp    Humidity  Windy  Play")
print("-" * 40)
for i in range(len(detailed_data['Outlook'])):
    print(f"{detailed_data['Outlook'][i]:<10} {detailed_data['Temperature'][i]:<8} {detailed_data['Humidity'][i]:<9} {str(detailed_data['Windy'][i]):<6} {detailed_data['Play'][i]}")

# Calculate initial entropy
total_samples = len(detailed_data['Play'])
yes_count = detailed_data['Play'].count('Yes')
no_count = detailed_data['Play'].count('No')
p_yes_initial = yes_count / total_samples
p_no_initial = no_count / total_samples

h_initial = entropy([p_yes_initial, p_no_initial])
print(f"\nInitial entropy: H(Play) = {h_initial:.4f} bits")
print(f"Class distribution: Yes={yes_count}/{total_samples}, No={no_count}/{total_samples}")

# Calculate information gain for all features
print(f"\nCalculating information gain for all features:")
print("=" * 50)

all_info_gains = {}
for feature in ['Outlook', 'Temperature', 'Humidity', 'Windy']:
    ig = calculate_information_gain(detailed_data, feature)
    all_info_gains[feature] = ig

# Find the best feature
best_feature = max(all_info_gains, key=all_info_gains.get)
best_ig = all_info_gains[best_feature]

print(f"\nBest feature to split on: {best_feature} (IG = {best_ig:.4f})")

# Visualize the complete tree building process
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_title('Complete ID3 Tree Building Process', fontsize=16, fontweight='bold')

# Draw the tree structure
# Root
root = Circle((0.5, 0.9), 0.08, facecolor='gold', edgecolor='orange', linewidth=3)
ax.add_patch(root)
ax.text(0.5, 0.9, 'Outlook\n(IG=0.247)', ha='center', va='center', fontweight='bold', fontsize=10)

# First level branches
sunny = Circle((0.2, 0.7), 0.07, facecolor='lightblue', edgecolor='blue', linewidth=2)
overcast = Circle((0.5, 0.7), 0.07, facecolor='lightgreen', edgecolor='green', linewidth=2)
rainy = Circle((0.8, 0.7), 0.07, facecolor='lightblue', edgecolor='blue', linewidth=2)

ax.add_patch(sunny)
ax.add_patch(overcast)
ax.add_patch(rainy)

ax.text(0.2, 0.7, 'Sunny\n(5 samples)', ha='center', va='center', fontweight='bold', fontsize=9)
ax.text(0.5, 0.7, 'Overcast\n(4 samples)', ha='center', va='center', fontweight='bold', fontsize=9)
ax.text(0.8, 0.7, 'Rainy\n(5 samples)', ha='center', va='center', fontweight='bold', fontsize=9)

# Second level for Sunny branch
sunny_temp = Circle((0.2, 0.5), 0.06, facecolor='lightcoral', edgecolor='red', linewidth=2)
ax.add_patch(sunny_temp)
ax.text(0.2, 0.5, 'Temperature\n(IG=0.571)', ha='center', va='center', fontweight='bold', fontsize=8)

# Third level for Sunny branch
sunny_hot = Circle((0.1, 0.3), 0.05, facecolor='lightgreen', edgecolor='green', linewidth=2)
sunny_mild = Circle((0.2, 0.3), 0.05, facecolor='lightgreen', edgecolor='green', linewidth=2)
sunny_cool = Circle((0.3, 0.3), 0.05, facecolor='lightgreen', edgecolor='green', linewidth=2)

ax.add_patch(sunny_hot)
ax.add_patch(sunny_mild)
ax.add_patch(sunny_cool)

ax.text(0.1, 0.3, 'Hot\nNo', ha='center', va='center', fontweight='bold', fontsize=8)
ax.text(0.2, 0.3, 'Mild\nNo', ha='center', va='center', fontweight='bold', fontsize=8)
ax.text(0.15, 0.25, 'Yes', ha='center', va='center', fontweight='bold', fontsize=8)

# Add edges
ax.plot([0.5, 0.2], [0.82, 0.77], 'k-', linewidth=2)
ax.plot([0.5, 0.5], [0.82, 0.77], 'k-', linewidth=2)
ax.plot([0.5, 0.8], [0.82, 0.77], 'k-', linewidth=2)
ax.plot([0.2, 0.2], [0.63, 0.56], 'k-', linewidth=2)
ax.plot([0.2, 0.1], [0.45, 0.35], 'k-', linewidth=2)
ax.plot([0.2, 0.2], [0.45, 0.35], 'k-', linewidth=2)
ax.plot([0.2, 0.3], [0.45, 0.35], 'k-', linewidth=2)

# Add labels
ax.text(0.35, 0.76, 'Sunny', ha='center', va='center', fontsize=10, color='blue')
ax.text(0.5, 0.76, 'Overcast', ha='center', va='center', fontsize=10, color='blue')
ax.text(0.65, 0.76, 'Rainy', ha='center', va='center', fontsize=10, color='blue')
ax.text(0.15, 0.53, 'Temp', ha='center', va='center', fontsize=9, color='red')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.tight_layout()
file_path = os.path.join(save_dir, "complete_ID3_example.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 8: Summary and Key Insights
print_step_header(8, "Summary and Key Insights")

print("Question 1 Summary:")
print("=" * 50)
print()
print("1. Main Steps of ID3 Algorithm:")
print("   ✓ Start with entire dataset")
print("   ✓ Check stopping criteria")
print("   ✓ Calculate information gain for all features")
print("   ✓ Choose feature with highest information gain")
print("   ✓ Split dataset and create child nodes")
print("   ✓ Recursively apply to each child")
print("   ✓ Create leaf nodes when stopping criteria met")
print()
print("2. Feature Selection:")
print("   ✓ Uses information gain: IG = H(S) - H(S|A)")
print("   ✓ Chooses feature that maximizes uncertainty reduction")
print("   ✓ Greedy approach - best local choice at each step")
print()
print("3. Stopping Criteria:")
print("   ✓ Pure node (all samples same class)")
print("   ✓ No features remaining")
print("   ✓ Empty dataset after split")
print()
print("4. Why ID3 is Greedy:")
print("   ✓ Makes locally optimal choices at each step")
print("   ✓ No backtracking or global optimization")
print("   ✓ May lead to suboptimal overall tree structure")
print("   ✓ But computationally efficient and practical")
print()

print("All figures have been saved to:", save_dir)
print("The ID3 algorithm demonstrates how recursive partitioning and information")
print("theory can be combined to create interpretable decision trees.")
