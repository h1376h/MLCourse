import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import os
from collections import Counter
import math

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_45")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("QUESTION 45: DECISION TREE CONSTRUCTION WITH CONDITIONAL ENTROPY")
print("=" * 80)

# Define the dataset
data = {
    'Weight': ['N', 'N', 'N', 'U', 'U', 'U', 'N', 'N', 'U', 'U'],
    'Eye_Color': ['A', 'V', 'V', 'V', 'V', 'A', 'A', 'V', 'A', 'A'],
    'Num_Eyes': [2, 2, 2, 3, 3, 4, 4, 4, 3, 3],
    'Output': ['L', 'L', 'L', 'L', 'L', 'D', 'D', 'D', 'D', 'D']
}

df = pd.DataFrame(data)
print("\nDATASET:")
print(df.to_string(index=False))

# Convert to numpy arrays for easier manipulation
X = df[['Weight', 'Eye_Color', 'Num_Eyes']].values
y = df['Output'].values

print(f"\nTotal samples: {len(df)}")
print(f"Features: {list(df.columns[:-1])}")
print(f"Target classes: {list(set(y))}")

# Function to calculate entropy
def entropy(y):
    """Calculate entropy of a target variable"""
    if len(y) == 0:
        return 0
    
    # Count occurrences of each class
    counts = Counter(y)
    total = len(y)
    
    # Calculate entropy
    entropy_val = 0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy_val -= p * math.log2(p)
    
    return entropy_val

# Function to calculate conditional entropy
def conditional_entropy(X, y, feature_idx, feature_value):
    """Calculate conditional entropy H(Y|X=value)"""
    # Filter data where feature equals the given value
    mask = X[:, feature_idx] == feature_value
    filtered_y = y[mask]
    
    if len(filtered_y) == 0:
        return 0
    
    return entropy(filtered_y)

# Function to calculate information gain
def information_gain(X, y, feature_idx):
    """Calculate information gain for a feature"""
    # Calculate parent entropy
    parent_entropy = entropy(y)
    
    # Get unique values of the feature
    unique_values = set(X[:, feature_idx])
    
    # Calculate weighted average of conditional entropies
    weighted_entropy = 0
    total_samples = len(y)
    
    for value in unique_values:
        mask = X[:, feature_idx] == value
        subset_size = np.sum(mask)
        subset_y = y[mask]
        
        if subset_size > 0:
            weighted_entropy += (subset_size / total_samples) * entropy(subset_y)
    
    return parent_entropy - weighted_entropy

# Function to calculate split information (for gain ratio)
def split_information(X, feature_idx):
    """Calculate split information for gain ratio calculation"""
    unique_values = set(X[:, feature_idx])
    total_samples = len(X)
    
    split_info = 0
    for value in unique_values:
        count = np.sum(X[:, feature_idx] == value)
        if count > 0:
            p = count / total_samples
            split_info -= p * math.log2(p)
    
    return split_info

# Function to calculate gain ratio
def gain_ratio(X, y, feature_idx):
    """Calculate gain ratio for a feature"""
    info_gain = information_gain(X, y, feature_idx)
    split_info = split_information(X, feature_idx)
    
    if split_info == 0:
        return 0
    
    return info_gain / split_info

print("\n" + "=" * 60)
print("STEP 1: CALCULATING CONDITIONAL ENTROPY")
print("=" * 60)

# Calculate conditional entropy H(eye color|weight=N)
print("\n1. Conditional Entropy H(Eye_Color|Weight=N):")

# Filter data where Weight = N
weight_n_mask = X[:, 0] == 'N'
weight_n_data = X[weight_n_mask]
weight_n_outputs = y[weight_n_mask]

print(f"   Data where Weight = N:")
print(f"   {weight_n_data}")
print(f"   Corresponding outputs: {weight_n_outputs}")

# Calculate entropy of eye colors when weight = N
eye_colors_weight_n = weight_n_data[:, 1]  # Eye_Color column
print(f"   Eye colors when Weight = N: {eye_colors_weight_n}")

# Count eye colors
eye_color_counts = Counter(eye_colors_weight_n)
print(f"   Eye color counts: {dict(eye_color_counts)}")

# Calculate conditional entropy using the CORRECT formula
# H(Eye_Color|Weight=N) = entropy of Eye_Color distribution when Weight=N
total_weight_n = len(eye_colors_weight_n)
conditional_entropy_val = 0

print(f"\n   CORRECT CONDITIONAL ENTROPY CALCULATION:")
print(f"   H(Eye_Color|Weight=N) = entropy of Eye_Color distribution when Weight=N")
print(f"   This is the entropy of the Eye_Color values themselves, not the Output given Eye_Color")

# Calculate entropy of the Eye_Color distribution when Weight=N
for eye_color, count in eye_color_counts.items():
    p = count / total_weight_n
    print(f"   P(Eye_Color={eye_color}|Weight=N) = {count}/{total_weight_n} = {p:.6f}")
    
    if p > 0:
        log_p = math.log2(p)
        term = -p * log_p
        conditional_entropy_val += term
        print(f"   -{p:.6f} × log2({p:.6f}) = -{p:.6f} × {log_p:.6f} = {term:.6f}")

print(f"\n   H(Eye_Color|Weight=N) = {conditional_entropy_val:.6f}")

# Let's also verify this matches the standard formula
print(f"\n   VERIFICATION USING STANDARD FORMULA:")
print(f"   H(Y|X) = Σ P(X=i) × H(Y|X=i)")
print(f"   In our case: H(Eye_Color|Weight=N) = entropy of Eye_Color distribution when Weight=N")
print(f"   This is the entropy of the distribution: {dict(eye_color_counts)}")

# Manual calculation to verify
p_a = 2/5
p_v = 3/5
h_manual = -p_a * math.log2(p_a) - p_v * math.log2(p_v)
print(f"   Manual calculation: H = -{p_a:.6f} × log2({p_a:.6f}) - {p_v:.6f} × log2({p_v:.6f})")
print(f"   H = -{p_a:.6f} × {math.log2(p_a):.6f} - {p_v:.6f} × {math.log2(p_v):.6f}")
print(f"   H = {p_a * abs(math.log2(p_a)):.6f} + {p_v * abs(math.log2(p_v)):.6f} = {h_manual:.6f}")

print(f"\n   FINAL RESULT: H(Eye_Color|Weight=N) = {conditional_entropy_val:.6f}")
print(f"   This matches your friend's answer of 0.971!")

print("\n" + "=" * 60)
print("STEP 2: ID3 ALGORITHM - INFORMATION GAIN CALCULATION")
print("=" * 60)

# Calculate parent entropy
parent_entropy = entropy(y)
print(f"\nParent entropy H(Output) = {parent_entropy:.3f}")

# Calculate information gain for each feature
feature_names = ['Weight', 'Eye_Color', 'Num_Eyes']
information_gains = []

print(f"\nInformation Gain calculations:")
print("-" * 50)

for i, feature_name in enumerate(feature_names):
    print(f"\nFeature: {feature_name}")
    
    # Get unique values
    unique_values = set(X[:, i])
    print(f"Unique values: {unique_values}")
    
    # Calculate weighted entropy
    weighted_entropy = 0
    total_samples = len(y)
    
    for value in unique_values:
        mask = X[:, i] == value
        subset_size = np.sum(mask)
        subset_y = y[mask]
        
        print(f"  Value '{value}': {subset_size} samples, outputs: {list(subset_y)}")
        
        if subset_size > 0:
            subset_entropy = entropy(subset_y)
            weight = subset_size / total_samples
            weighted_entropy += weight * subset_entropy
            
            print(f"    Entropy = {subset_entropy:.3f}, Weight = {weight:.3f}")
    
    info_gain = parent_entropy - weighted_entropy
    information_gains.append(info_gain)
    
    print(f"  Weighted entropy = {weighted_entropy:.3f}")
    print(f"  Information Gain = {parent_entropy:.3f} - {weighted_entropy:.3f} = {info_gain:.3f}")

# Find best feature
best_feature_idx = np.argmax(information_gains)
best_feature = feature_names[best_feature_idx]
best_info_gain = information_gains[best_feature_idx]

print(f"\n" + "=" * 50)
print(f"ID3 ROOT SELECTION:")
print(f"Best feature: {best_feature}")
print(f"Information gain: {best_info_gain:.3f}")
print("=" * 50)

print("\n" + "=" * 60)
print("STEP 3: DECISION TREE CONSTRUCTION")
print("=" * 60)

# Simple decision tree class for visualization
class DecisionTree:
    def __init__(self):
        self.root = None
    
    def fit(self, X, y, feature_names):
        self.feature_names = feature_names
        self.root = self._build_tree(X, y, feature_names, depth=0)
    
    def _build_tree(self, X, y, feature_names, depth):
        if len(X) == 0:
            return None
        
        # If all samples have same class, return leaf
        if len(set(y)) == 1:
            return {'type': 'leaf', 'class': y[0], 'count': len(y)}
        
        # If no features left or max depth reached, return majority class
        if len(feature_names) == 0 or depth > 10:
            majority_class = Counter(y).most_common(1)[0][0]
            return {'type': 'leaf', 'class': majority_class, 'count': len(y)}
        
        # Find best feature
        best_feature_idx = 0
        best_info_gain = -1
        
        for i in range(len(feature_names)):
            info_gain = information_gain(X, y, i)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature_idx = i
        
        if best_info_gain <= 0:
            # No information gain, return majority class
            majority_class = Counter(y).most_common(1)[0][0]
            return {'type': 'leaf', 'class': majority_class, 'count': len(y)}
        
        best_feature = feature_names[best_feature_idx]
        unique_values = sorted(set(X[:, best_feature_idx]))
        
        # Create node
        node = {
            'type': 'node',
            'feature': best_feature,
            'feature_idx': best_feature_idx,
            'children': {}
        }
        
        # Create children for each value
        remaining_features = [f for j, f in enumerate(feature_names) if j != best_feature_idx]
        
        for value in unique_values:
            mask = X[:, best_feature_idx] == value
            subset_X = X[mask]
            subset_y = y[mask]
            
            # Remove the used feature from subset
            subset_X_remaining = np.column_stack([subset_X[:, j] for j in range(len(feature_names)) if j != best_feature_idx])
            
            node['children'][value] = self._build_tree(subset_X_remaining, subset_y, remaining_features, depth + 1)
        
        return node
    
    def predict(self, x):
        return self._predict_recursive(x, self.root, self.feature_names)
    
    def _predict_recursive(self, x, node, feature_names):
        if node['type'] == 'leaf':
            return node['class']
        
        feature_idx = node['feature_idx']
        feature_value = x[feature_idx]
        
        if feature_value in node['children']:
            # Remove the used feature for recursive call
            remaining_x = [x[j] for j in range(len(x)) if j != feature_idx]
            remaining_features = [f for j, f in enumerate(feature_names) if j != feature_idx]
            return self._predict_recursive(remaining_x, node['children'][feature_value], remaining_features)
        else:
            # Value not seen in training, return majority class of this node
            return self._get_majority_class(node)
    
    def _get_majority_class(self, node):
        if node['type'] == 'leaf':
            return node['class']
        
        # Collect all classes from children
        classes = []
        for child in node['children'].values():
            if child['type'] == 'leaf':
                classes.extend([child['class']] * child['count'])
            else:
                classes.extend([self._get_majority_class(child)])
        
        return Counter(classes).most_common(1)[0][0]

# Build the decision tree
tree = DecisionTree()
tree.fit(X, y, feature_names)

print(f"\nDecision tree built successfully!")
print(f"Root feature: {tree.root['feature']}")

# Function to visualize decision tree
def visualize_decision_tree(tree, save_path):
    # Calculate tree depth and number of leaves for proper sizing
    def get_tree_info(node, level=0):
        if node['type'] == 'leaf':
            return level, 1
        else:
            max_depth = level
            total_leaves = 0
            for child in node['children'].values():
                child_depth, child_leaves = get_tree_info(child, level + 1)
                max_depth = max(max_depth, child_depth)
                total_leaves += child_leaves
            return max_depth, total_leaves
    
    max_depth, total_leaves = get_tree_info(tree.root)
    
    # Adjust figure size based on tree structure
    fig_width = max(12, total_leaves * 2)
    fig_height = max(8, (max_depth + 1) * 2)
    
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    ax.set_xlim(0, fig_width)
    ax.set_ylim(0, fig_height)
    ax.axis('off')
    
    def draw_node(node, x, y, width, height, level=0):
        if node['type'] == 'leaf':
            # Draw leaf node with smaller size to avoid overlap
            node_width = min(width, 1.5)
            node_height = 0.8
            node_x = x + (width - node_width) / 2
            
            rect = FancyBboxPatch((node_x, y), node_width, node_height, 
                                boxstyle="round,pad=0.1", 
                                facecolor='lightgreen', 
                                edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
            # Add text
            ax.text(node_x + node_width/2, y + node_height/2, 
                   f"Class: {node['class']}\nCount: {node['count']}", 
                   ha='center', va='center', fontsize=9, fontweight='bold')
        else:
            # Draw decision node
            node_width = min(width, 2.0)
            node_height = 0.8
            node_x = x + (width - node_width) / 2
            
            rect = FancyBboxPatch((node_x, y), node_width, node_height, 
                                boxstyle="round,pad=0.1", 
                                facecolor='lightblue', 
                                edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
            # Add text
            ax.text(node_x + node_width/2, y + node_height/2, 
                   f"Feature: {node['feature']}", 
                   ha='center', va='center', fontsize=9, fontweight='bold')
            
            # Draw children
            children = list(node['children'].items())
            if children:
                child_width = width / len(children)
                child_y = y - 1.2  # Reduced spacing
                
                for i, (value, child) in enumerate(children):
                    child_x = x + i * child_width
                    
                    # Draw edge
                    ax.plot([node_x + node_width/2, child_x + child_width/2], 
                           [y, child_y + height], 'k-', linewidth=2)
                    
                    # Add edge label
                    ax.text((node_x + node_width/2 + child_x + child_width/2)/2, 
                           (y + child_y + height)/2,
                           f"= {value}", ha='center', va='center', fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                    
                    draw_node(child, child_x, child_y, child_width, height, level + 1)
    
    # Start drawing from the top
    draw_node(tree.root, 1, fig_height - 1, fig_width - 2, 0.8)
    
    plt.title('Decision Tree Visualization', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

# Visualize the tree
visualize_decision_tree(tree, os.path.join(save_dir, 'decision_tree.png'))

print(f"\nDecision tree visualization saved to: {os.path.join(save_dir, 'decision_tree.png')}")

print("\n" + "=" * 60)
print("STEP 4: TRAINING SET ERROR CALCULATION")
print("=" * 60)

# Calculate training set error
correct_predictions = 0
total_predictions = len(X)

print(f"\nPredictions on training set:")
print("-" * 40)

for i, (sample, true_label) in enumerate(zip(X, y)):
    prediction = tree.predict(sample)
    is_correct = prediction == true_label
    if is_correct:
        correct_predictions += 1
    
    print(f"Sample {i+1}: {sample} -> Predicted: {prediction}, True: {true_label}, Correct: {is_correct}")

training_error = 1 - (correct_predictions / total_predictions)
training_accuracy = correct_predictions / total_predictions

print(f"\n" + "=" * 50)
print(f"TRAINING SET PERFORMANCE:")
print(f"Correct predictions: {correct_predictions}/{total_predictions}")
print(f"Training accuracy: {training_accuracy:.3f}")
print(f"Training error: {training_error:.3f}")
print("=" * 50)

# Create detailed analysis visualization
def create_analysis_plots():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Information Gain Comparison
    features = ['Weight', 'Eye_Color', 'Num_Eyes']
    ax1.bar(features, information_gains, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax1.set_title('Information Gain for Each Feature', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Information Gain')
    ax1.set_ylim(0, max(information_gains) * 1.1)
    
    # Add value labels on bars
    for i, v in enumerate(information_gains):
        ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Class Distribution
    class_counts = Counter(y)
    ax2.pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%', 
            colors=['lightblue', 'lightcoral'])
    ax2.set_title('Overall Class Distribution', fontsize=14, fontweight='bold')
    
    # Plot 3: Feature Value Distribution
    feature_values = {}
    for i, feature in enumerate(features):
        feature_values[feature] = Counter(X[:, i])
    
    x = np.arange(len(features))
    width = 0.35
    
    # Get unique values across all features
    all_values = set()
    for values in feature_values.values():
        all_values.update(values.keys())
    
    # Create grouped bar chart
    unique_values_list = sorted(all_values, key=lambda x: str(x))
    for j, value in enumerate(unique_values_list):
        counts = []
        for feature in features:
            counts.append(feature_values[feature].get(value, 0))
        
        ax3.bar(x + j * width/len(unique_values_list), counts, 
                width/len(unique_values_list), label=f'Value: {value}')
    
    ax3.set_title('Feature Value Distribution', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Count')
    ax3.set_xticks(x)
    ax3.set_xticklabels(features)
    ax3.legend()
    
    # Plot 4: Conditional Entropy Analysis
    # Calculate conditional entropy for each feature value
    conditional_entropies = {}
    for i, feature in enumerate(features):
        conditional_entropies[feature] = {}
        unique_values = set(X[:, i])
        for value in unique_values:
            mask = X[:, i] == value
            subset_y = y[mask]
            conditional_entropies[feature][value] = entropy(subset_y)
    
    # Create heatmap-like visualization
    feature_names_plot = []
    value_names_plot = []
    entropy_values = []
    
    for feature in features:
        for value, entropy_val in conditional_entropies[feature].items():
            feature_names_plot.append(feature)
            value_names_plot.append(f"{feature}={value}")
            entropy_values.append(entropy_val)
    
    bars = ax4.barh(range(len(entropy_values)), entropy_values, 
                    color=['lightblue', 'lightcoral', 'lightgreen'] * (len(entropy_values)//3 + 1))
    ax4.set_yticks(range(len(entropy_values)))
    ax4.set_yticklabels(value_names_plot)
    ax4.set_xlabel('Conditional Entropy')
    ax4.set_title('Conditional Entropy by Feature Value', fontsize=14, fontweight='bold')
    
    # Add value labels
    for i, v in enumerate(entropy_values):
        ax4.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'analysis_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()

create_analysis_plots()
print(f"\nAnalysis plots saved to: {os.path.join(save_dir, 'analysis_plots.png')}")

# Print summary
print("\n" + "=" * 80)
print("SUMMARY OF RESULTS")
print("=" * 80)
print(f"1. Conditional entropy H(Eye_Color|Weight=N) = {conditional_entropy_val:.3f}")
print(f"2. ID3 root selection: {best_feature} (Information Gain = {best_info_gain:.3f})")
print(f"3. Decision tree constructed and visualized")
print(f"4. Training set error = {training_error:.3f} ({training_accuracy:.3f} accuracy)")
print("=" * 80)

print(f"\nAll visualizations saved to: {save_dir}")
