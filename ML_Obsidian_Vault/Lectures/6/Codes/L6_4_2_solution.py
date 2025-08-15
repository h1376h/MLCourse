import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_4_Quiz_2")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("DECISION TREE PRE-PRUNING THRESHOLD PUZZLE - COMPREHENSIVE SOLUTION")
print("=" * 80)

# Given parameters
total_samples = 1000
n_features = 8
min_samples_per_leaf = 50

print(f"Given Parameters:")
print(f"- Total samples: {total_samples}")
print(f"- Number of binary features: {n_features}")
print(f"- Minimum samples per leaf: {min_samples_per_leaf}")
print()

# Question 1: Maximum number of leaf nodes possible
print("QUESTION 1: Maximum number of leaf nodes possible")
print("-" * 50)

# If each leaf must have at least 50 samples, the maximum number of leaves is:
max_leaves = total_samples // min_samples_per_leaf
print(f"Maximum leaves = Total samples ÷ Minimum samples per leaf")
print(f"Maximum leaves = {total_samples} ÷ {min_samples_per_leaf} = {max_leaves}")

# Show the calculation
remaining_samples = total_samples % min_samples_per_leaf
if remaining_samples > 0:
    print(f"Note: {remaining_samples} samples would remain unassigned")
    print(f"To use all samples, you could have {max_leaves} leaves with {min_samples_per_leaf} samples each")
    print(f"and 1 leaf with {min_samples_per_leaf + remaining_samples} samples")

print(f"Answer: Maximum number of leaf nodes = {max_leaves}")
print()

# Question 2: Theoretical maximum depth before pre-pruning
print("QUESTION 2: Theoretical maximum depth before pre-pruning")
print("-" * 50)

# For binary features, each split can create 2 branches
# Maximum depth occurs when each split creates a new level
# At each level, the number of nodes doubles: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
# We need to find the level where we exceed our sample limit

def calculate_max_depth(n_samples, n_features):
    """Calculate maximum depth considering sample constraints"""
    max_depth = 0
    nodes_at_level = 1
    
    while nodes_at_level <= n_samples and max_depth <= n_features:
        max_depth += 1
        nodes_at_level *= 2
    
    return max_depth - 1

max_depth_theoretical = calculate_max_depth(total_samples, n_features)
print(f"Maximum depth calculation:")
print(f"- At depth 0: 1 node (root)")
print(f"- At depth 1: 2 nodes")
print(f"- At depth 2: 4 nodes")
print(f"- At depth 3: 8 nodes")
print(f"- At depth 4: 16 nodes")
print(f"- At depth 5: 32 nodes")
print(f"- At depth 6: 64 nodes")
print(f"- At depth 7: 128 nodes")
print(f"- At depth 8: 256 nodes")
print(f"- At depth 9: 512 nodes")
print(f"- At depth 10: 1024 nodes (exceeds our {total_samples} samples)")

print(f"\nMaximum depth = {max_depth_theoretical}")
print(f"Answer: Theoretical maximum depth = {max_depth_theoretical}")
print()

# Question 3: Appropriate minimum Gini impurity threshold
print("QUESTION 3: Appropriate minimum Gini impurity threshold")
print("-" * 50)

# Gini impurity ranges from 0 (pure) to 0.5 (maximum impurity for binary classification)
# For binary classification, Gini impurity = 2 * p * (1-p) where p is probability of class 1
print("Gini impurity calculation for binary classification:")
print("Gini = 2 × p × (1-p) where p is probability of class 1")

# Show examples
probabilities = [0.1, 0.2, 0.3, 0.4, 0.5]
print("\nGini impurity examples:")
for p in probabilities:
    gini = 2 * p * (1 - p)
    print(f"- p = {p}: Gini = 2 × {p} × {1-p} = {gini:.3f}")

# For medical diagnosis, we want high purity
print(f"\nFor medical diagnosis (high stakes), recommended threshold:")
print(f"- Conservative: 0.01 (very pure nodes)")
print(f"- Moderate: 0.05 (moderately pure nodes)")
print(f"- Liberal: 0.1 (somewhat pure nodes)")

recommended_threshold = 0.05
print(f"\nAnswer: Recommended minimum Gini impurity threshold = {recommended_threshold}")
print()

# Question 4: Which pre-pruning parameter to adjust first
print("QUESTION 4: Which pre-pruning parameter to adjust first")
print("-" * 50)

training_acc = 0.95
validation_acc = 0.82
gap = training_acc - validation_acc

print(f"Training accuracy: {training_acc:.3f}")
print(f"Validation accuracy: {validation_acc:.3f}")
print(f"Generalization gap: {gap:.3f}")

print(f"\nAnalysis:")
print(f"- Large gap ({gap:.3f}) indicates overfitting")
print(f"- Training accuracy is very high ({training_acc:.3f})")
print(f"- Validation accuracy is much lower ({validation_acc:.3f})")

print(f"\nRecommended parameter to adjust first:")
print(f"1. max_depth - Most effective for overfitting")
print(f"2. min_samples_split - Prevents splitting small nodes")
print(f"3. min_samples_leaf - Ensures sufficient samples per leaf")
print(f"4. min_impurity_decrease - Prevents unnecessary splits")

print(f"\nAnswer: Adjust max_depth first to reduce overfitting")
print()

# Question 5: Minimum samples per leaf for exactly 20 leaf nodes
print("QUESTION 5: Minimum samples per leaf for exactly 20 leaf nodes")
print("-" * 50)

target_leaves = 20
min_samples_needed = total_samples // target_leaves
remaining = total_samples % target_leaves

print(f"To have exactly {target_leaves} leaves:")
print(f"Minimum samples per leaf = {total_samples} ÷ {target_leaves} = {min_samples_needed}")
print(f"Remaining samples: {remaining}")

if remaining > 0:
    print(f"\nDistribution:")
    print(f"- {target_leaves - 1} leaves with {min_samples_needed} samples each")
    print(f"- 1 leaf with {min_samples_needed + remaining} samples")
    print(f"Total: {target_leaves - 1} × {min_samples_needed} + {min_samples_needed + remaining} = {total_samples}")

print(f"\nAnswer: Minimum samples per leaf = {min_samples_needed}")
print()

# Question 6: Additional pre-pruning constraints for medical diagnosis
print("QUESTION 6: Additional pre-pruning constraints for medical diagnosis")
print("-" * 50)

print("Medical diagnosis requires high reliability and interpretability:")
print("\n1. max_depth:")
print(f"   - Conservative: 3-4 levels")
print(f"   - Moderate: 5-6 levels")
print(f"   - Liberal: 7-8 levels")

print(f"\n2. min_samples_split:")
print(f"   - Conservative: 100+ samples")
print(f"   - Moderate: 50+ samples")
print(f"   - Liberal: 20+ samples")

print(f"\n3. min_samples_leaf:")
print(f"   - Conservative: 30+ samples")
print(f"   - Moderate: 20+ samples")
print(f"   - Liberal: 10+ samples")

print(f"\n4. min_impurity_decrease:")
print(f"   - Conservative: 0.01")
print(f"   - Moderate: 0.05")
print(f"   - Liberal: 0.1")

print(f"\n5. Additional constraints:")
print(f"   - max_leaf_nodes: Limit total complexity")
print(f"   - class_weight: Handle class imbalance")
print(f"   - random_state: Ensure reproducibility")

print(f"\nAnswer: Use conservative thresholds for medical diagnosis")
print()

# Question 7: Calculate minimum impurity decrease threshold
print("QUESTION 7: Calculate minimum impurity decrease threshold")
print("-" * 50)

# Given scenario: 100 samples split into 45 and 55
parent_samples = 100
left_samples = 45
right_samples = 55

print(f"Parent node: {parent_samples} samples")
print(f"Left child: {left_samples} samples")
print(f"Right child: {right_samples} samples")

# Calculate Gini impurity for parent (assuming balanced classes for demonstration)
parent_gini = 0.5  # Maximum impurity for binary classification
print(f"\nParent Gini impurity = {parent_gini} (assuming balanced classes)")

# Calculate weighted average of child impurities
# For demonstration, assume some impurity in children
left_gini = 0.4  # Example value
right_gini = 0.3  # Example value

weighted_child_gini = (left_samples * left_gini + right_samples * right_gini) / parent_samples
print(f"Left child Gini = {left_gini}")
print(f"Right child Gini = {right_gini}")
print(f"Weighted average child Gini = ({left_samples} × {left_gini} + {right_samples} × {right_gini}) ÷ {parent_samples}")
print(f"Weighted average child Gini = {weighted_child_gini:.3f}")

# Calculate impurity decrease
impurity_decrease = parent_gini - weighted_child_gini
print(f"\nImpurity decrease = Parent Gini - Weighted child Gini")
print(f"Impurity decrease = {parent_gini} - {weighted_child_gini:.3f} = {impurity_decrease:.3f}")

# To prevent this split, set threshold higher than calculated decrease
threshold = impurity_decrease + 0.001
print(f"\nTo prevent this split, set min_impurity_decrease > {impurity_decrease:.3f}")
print(f"Recommended threshold = {threshold:.3f}")

print(f"\nAnswer: Minimum impurity decrease threshold = {threshold:.3f}")
print()

# Create visualizations
print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

# Visualization 1: Decision tree structure and depth
plt.figure(figsize=(15, 10))

# Create a sample decision tree for visualization
X, y = make_classification(n_samples=100, n_features=4, n_informative=2, 
                          n_redundant=1, n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create tree with specific parameters
tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42)
tree.fit(X_train, y_train)

# Plot the tree
plt.subplot(2, 2, 1)
plot_tree(tree, filled=True, rounded=True, feature_names=[f'X{i+1}' for i in range(4)])
plt.title('Sample Decision Tree Structure\n(max_depth=4, min_samples_leaf=5)')

# Visualization 2: Gini impurity vs probability
plt.subplot(2, 2, 2)
p_values = np.linspace(0, 1, 100)
gini_values = 2 * p_values * (1 - p_values)
plt.plot(p_values, gini_values, 'b-', linewidth=2)
plt.xlabel('Probability of Class 1 (p)')
plt.ylabel('Gini Impurity')
plt.title('Gini Impurity vs Probability\nfor Binary Classification')
plt.grid(True, alpha=0.3)
plt.axhline(y=0.05, color='r', linestyle='--', label='Threshold = 0.05')
plt.legend()

# Visualization 3: Training vs Validation accuracy
plt.subplot(2, 2, 3)
depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
train_accs = []
val_accs = []

for depth in depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, tree.predict(X_train))
    val_acc = accuracy_score(y_test, tree.predict(X_test))
    train_accs.append(train_acc)
    val_accs.append(val_acc)

plt.plot(depths, train_accs, 'b-o', label='Training Accuracy', linewidth=2)
plt.plot(depths, val_accs, 'r-s', label='Validation Accuracy', linewidth=2)
plt.xlabel('Maximum Depth')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy\nvs Maximum Depth')
plt.legend()
plt.grid(True, alpha=0.3)

# Visualization 4: Pre-pruning parameters comparison
plt.subplot(2, 2, 4)
parameters = ['max_depth', 'min_samples_split', 'min_samples_leaf', 'min_impurity_decrease']
effectiveness = [0.9, 0.7, 0.6, 0.5]  # Relative effectiveness for overfitting
colors = ['red', 'orange', 'yellow', 'green']

bars = plt.bar(parameters, effectiveness, color=colors, alpha=0.7)
plt.xlabel('Pre-pruning Parameters')
plt.ylabel('Effectiveness for Overfitting')
plt.title('Relative Effectiveness of Pre-pruning\nParameters for Reducing Overfitting')
plt.ylim(0, 1)
plt.xticks(rotation=45)

# Add value labels on bars
for bar, value in zip(bars, effectiveness):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{value:.1f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'decision_tree_pre_pruning_analysis.png'), dpi=300, bbox_inches='tight')

# Create additional detailed visualization
plt.figure(figsize=(16, 12))

# Visualization 5: Detailed impurity decrease calculation
plt.subplot(2, 3, 1)
# Create a tree visualization showing the split scenario
plt.text(0.5, 0.5, f'Parent Node\n100 samples\nGini = 0.5', 
         ha='center', va='center', transform=plt.gca().transAxes,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
plt.text(0.2, 0.2, f'Left Child\n45 samples\nGini = 0.4', 
         ha='center', va='center', transform=plt.gca().transAxes,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
plt.text(0.8, 0.2, f'Right Child\n55 samples\nGini = 0.3', 
         ha='center', va='center', transform=plt.gca().transAxes,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
plt.title('Impurity Decrease Calculation\nExample')
plt.axis('off')

# Visualization 6: Sample distribution across leaves
plt.subplot(2, 3, 2)
leaf_counts = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
leaf_indices = list(range(1, 21))
plt.bar(leaf_indices, leaf_counts, color='skyblue', alpha=0.7)
plt.xlabel('Leaf Node Index')
plt.ylabel('Number of Samples')
plt.title('Sample Distribution\n(20 leaves, 50 samples each)')
plt.xticks(leaf_indices[::2])

# Visualization 7: Feature importance and depth relationship
plt.subplot(2, 3, 3)
feature_importance = [0.4, 0.3, 0.2, 0.1]
feature_names = ['X1', 'X2', 'X3', 'X4']
plt.pie(feature_importance, labels=feature_names, autopct='%1.1f%%', startangle=90)
plt.title('Feature Importance\nDistribution')

# Visualization 8: Overfitting visualization
plt.subplot(2, 3, 4)
complexity = np.linspace(1, 10, 100)
train_error = 0.1 + 0.05 * np.exp(-complexity/3)
val_error = 0.1 + 0.3 * np.exp(-complexity/8) + 0.1 * (complexity/10)**2

plt.plot(complexity, train_error, 'b-', label='Training Error', linewidth=2)
plt.plot(complexity, val_error, 'r-', label='Validation Error', linewidth=2)
plt.axvline(x=4, color='g', linestyle='--', label='Optimal Complexity')
plt.xlabel('Model Complexity (Depth)')
plt.ylabel('Error Rate')
plt.title('Overfitting Visualization')
plt.legend()
plt.grid(True, alpha=0.3)

# Visualization 9: Pre-pruning parameter effects
plt.subplot(2, 3, 5)
param_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
tree_sizes = [100, 95, 85, 70, 50, 30, 15, 8, 4, 2]
plt.plot(param_values, tree_sizes, 'g-o', linewidth=2, markersize=8)
plt.xlabel('max_depth')
plt.ylabel('Number of Nodes')
plt.title('Tree Size vs max_depth')
plt.grid(True, alpha=0.3)

# Visualization 10: Medical diagnosis constraints
plt.subplot(2, 3, 6)
constraints = ['Conservative', 'Moderate', 'Liberal']
max_depths = [3, 6, 9]
min_samples = [100, 50, 20]
min_impurity = [0.01, 0.05, 0.1]

x = np.arange(len(constraints))
width = 0.25

plt.bar(x - width, max_depths, width, label='max_depth', color='lightblue')
plt.bar(x, min_samples, width, label='min_samples_split', color='lightgreen')
plt.bar(x + width, [x*100 for x in min_impurity], width, label='min_impurity_decrease×100', color='lightcoral')

plt.xlabel('Constraint Level')
plt.ylabel('Parameter Value')
plt.title('Medical Diagnosis\nPre-pruning Parameters')
plt.xticks(x, constraints)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'decision_tree_detailed_analysis.png'), dpi=300, bbox_inches='tight')

print(f"Visualizations saved to: {save_dir}")
print("\n" + "=" * 80)
print("SOLUTION COMPLETE!")
print("=" * 80)
print(f"\nSummary of answers:")
print(f"1. Maximum leaf nodes: {max_leaves}")
print(f"2. Maximum depth: {max_depth_theoretical}")
print(f"3. Gini threshold: {recommended_threshold}")
print(f"4. Adjust first: max_depth")
print(f"5. Min samples per leaf: {min_samples_needed}")
print(f"6. Use conservative thresholds for medical diagnosis")
print(f"7. Impurity decrease threshold: {threshold:.3f}")
