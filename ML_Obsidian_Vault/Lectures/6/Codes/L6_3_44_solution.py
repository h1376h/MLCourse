import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter
from math import log2
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_44")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Dataset
data = [
    {'A': 'F', 'B': 'F', 'C': 'F', 'Y': 'F'},
    {'A': 'T', 'B': 'F', 'C': 'T', 'Y': 'T'},
    {'A': 'T', 'B': 'T', 'C': 'F', 'Y': 'T'},
    {'A': 'T', 'B': 'T', 'C': 'T', 'Y': 'F'}
]

print("=" * 60)
print("DECISION TREE ANALYSIS - QUESTION 44")
print("=" * 60)

# Task 1: Calculate entropy of entire dataset
def calculate_entropy(labels):
    """Calculate entropy of a set of labels"""
    if len(labels) == 0:
        return 0

    label_counts = Counter(labels)
    entropy = 0

    for count in label_counts.values():
        p = count / len(labels)
        if p > 0:
            entropy -= p * log2(p)

    return entropy

# Calculate overall entropy
all_labels = [sample['Y'] for sample in data]
overall_entropy = calculate_entropy(all_labels)

print("\n1. ENTROPY OF ENTIRE DATASET")
print(f"Dataset: {len(data)} samples")
print(f"Labels: {[sample['Y'] for sample in data]}")
print(".4f")
print(".4f")
print("Since entropy > 0, there is uncertainty in the classification")
print("The closer to 1, the more mixed the classes are")

# Task 2: Calculate information gain for each feature
def calculate_information_gain(data, feature):
    """Calculate information gain for a feature"""
    # Overall entropy
    overall_entropy = calculate_entropy([sample['Y'] for sample in data])

    # Get unique values of the feature
    feature_values = set(sample[feature] for sample in data)

    weighted_entropy = 0

    print(f"\nSplitting on {feature}:")
    print(f"  Unique values: {sorted(feature_values)}")

    for value in sorted(feature_values):
        # Get subset where feature == value
        subset = [sample for sample in data if sample[feature] == value]
        subset_labels = [sample['Y'] for sample in subset]

        # Calculate entropy of subset
        subset_entropy = calculate_entropy(subset_labels)

        # Weight by proportion
        weight = len(subset) / len(data)
        weighted_entropy += weight * subset_entropy

        print(f"  {feature} = {value}: {len(subset)} samples, labels = {subset_labels}, entropy = {subset_entropy:.4f}")

    information_gain = overall_entropy - weighted_entropy

    print(f"  Information Gain for {feature}: {overall_entropy:.4f} - {weighted_entropy:.4f} = {information_gain:.4f}")

    return information_gain

# Calculate information gain for each feature
features = ['A', 'B', 'C']
information_gains = {}

print("\n2. INFORMATION GAIN CALCULATION")
print("-" * 40)

for feature in features:
    ig = calculate_information_gain(data, feature)
    information_gains[feature] = ig

# Find optimal root split
optimal_feature = max(information_gains, key=information_gains.get)
print("\nOptimal root split:")
for feature in features:
    print(".4f")
print(f"  -> Best feature: {optimal_feature} (highest information gain)")

# Task 3: Build complete decision tree
def build_decision_tree(data, features, depth=0):
    """Recursively build decision tree"""
    labels = [sample['Y'] for sample in data]

    # Base cases
    if len(set(labels)) == 1:
        return {'type': 'leaf', 'class': labels[0], 'samples': len(data)}

    if len(features) == 0:
        majority_class = Counter(labels).most_common(1)[0][0]
        return {'type': 'leaf', 'class': majority_class, 'samples': len(data)}

    # Find best feature to split on
    best_feature = None
    best_ig = -1

    for feature in features:
        ig = calculate_information_gain(data, feature)
        if ig > best_ig:
            best_ig = ig
            best_feature = feature

    if best_ig == 0:
        majority_class = Counter(labels).most_common(1)[0][0]
        return {'type': 'leaf', 'class': majority_class, 'samples': len(data)}

    # Create node
    node = {
        'type': 'node',
        'feature': best_feature,
        'samples': len(data),
        'children': {}
    }

    remaining_features = [f for f in features if f != best_feature]
    feature_values = set(sample[best_feature] for sample in data)

    for value in feature_values:
        subset = [sample for sample in data if sample[best_feature] == value]
        if len(subset) > 0:
            node['children'][value] = build_decision_tree(subset, remaining_features, depth + 1)

    return node

# Helper functions for optimality analysis
def check_zero_error(tree, test_data):
    """Check if tree classifies all data correctly"""
    correct = 0
    for sample in test_data:
        predicted = predict_sample(tree, sample)
        if predicted == sample['Y']:
            correct += 1
    return correct == len(test_data)

def predict_sample(tree, sample):
    """Predict class for a sample using the tree"""
    if tree['type'] == 'leaf':
        return tree['class']

    feature = tree['feature']
    if feature in sample:
        value = sample[feature]
        if value in tree['children']:
            return predict_sample(tree['children'][value], sample)

    # Default to majority class if path not found
    return 'F'  # Conservative default

# Build the tree
print("\n3. DECISION TREE CONSTRUCTION")
print("-" * 40)

decision_tree = build_decision_tree(data, features.copy())

def print_tree(node, indent=""):
    """Print tree structure"""
    if node['type'] == 'leaf':
        print(f"{indent}Leaf: Class {node['class']} ({node['samples']} samples)")
    else:
        print(f"{indent}Node: {node['feature']} ({node['samples']} samples)")
        for value, child in node['children'].items():
            print(f"{indent}  {node['feature']} = {value}:")
            print_tree(child, indent + "    ")

print("Decision Tree Structure:")
print_tree(decision_tree)

# Task 4: Check if tree is optimal - try alternative structures
print("\n4. OPTIMALITY ANALYSIS")
print("-" * 40)

print("Testing alternative tree structures that might achieve zero error...")

# Try the alternative structure: B as root, C in both subtrees
print("\nTesting alternative tree structure: B as root, C in both subtrees")

def build_alternative_tree():
    """Build the alternative tree structure: B -> C in both branches"""
    tree = {
        'type': 'node',
        'feature': 'B',
        'samples': 4,
        'children': {}
    }

    # B = F branch: use C to split
    b_f_samples = [data[0], data[1]]  # Samples 1 and 2
    tree['children']['F'] = {
        'type': 'node',
        'feature': 'C',
        'samples': 2,
        'children': {}
    }
    # C = F: Sample 1 (F,F,F) -> F
    tree['children']['F']['children']['F'] = {
        'type': 'leaf',
        'class': 'F',
        'samples': 1
    }
    # C = T: Sample 2 (T,F,T) -> T
    tree['children']['F']['children']['T'] = {
        'type': 'leaf',
        'class': 'T',
        'samples': 1
    }

    # B = T branch: use C to split
    b_t_samples = [data[2], data[3]]  # Samples 3 and 4
    tree['children']['T'] = {
        'type': 'node',
        'feature': 'C',
        'samples': 2,
        'children': {}
    }
    # C = F: Sample 3 (T,T,F) -> T
    tree['children']['T']['children']['F'] = {
        'type': 'leaf',
        'class': 'T',
        'samples': 1
    }
    # C = T: Sample 4 (T,T,T) -> F
    tree['children']['T']['children']['T'] = {
        'type': 'leaf',
        'class': 'F',
        'samples': 1
    }

    return tree

# Test the alternative tree
alternative_tree = build_alternative_tree()
print("\nAlternative Tree Structure (B as root, C in both subtrees):")
print_tree(alternative_tree)

# Check if alternative tree achieves zero error
alternative_zero_error = check_zero_error(alternative_tree, data)
print(f"\nAlternative tree achieves zero error: {alternative_zero_error}")

if alternative_zero_error:
    print("  ✓ We can achieve zero error with a different tree structure!")
    print("  ✓ The original tree is NOT optimal")
    print("  ✓ Alternative tree structure found:")
    print("    - Root: B (depth 1)")
    print("    - B=F branch: C splits to F and T (depth 2)")
    print("    - B=T branch: C splits to F and T (depth 2)")
    print("    - Maximum depth: 2 (vs 3 in original tree)")

print("\n" + "-" * 50)
print("Original analysis (removing feature A):")
print("-" * 50)

# Dataset without feature A
data_no_a = [
    {'B': 'F', 'C': 'F', 'Y': 'F'},  # First sample: A=F, B=F, C=F, Y=F
    {'B': 'F', 'C': 'T', 'Y': 'T'},  # Second sample: A=T, B=F, C=T, Y=T
    {'B': 'T', 'C': 'F', 'Y': 'T'},  # Third sample: A=T, B=T, C=F, Y=T
    {'B': 'T', 'C': 'T', 'Y': 'F'}   # Fourth sample: A=T, B=T, C=T, Y=F
]

print("Dataset without feature A:")
for i, sample in enumerate(data_no_a):
    print(f"  Sample {i+1}: {sample}")

# Check if samples are still unique without A
print("\nChecking if all samples are still distinguishable without A:")
sample_strings = []
for i, sample in enumerate(data_no_a):
    sample_str = f"B={sample['B']}, C={sample['C']}, Y={sample['Y']}"
    sample_strings.append(sample_str)
    print(f"  Sample {i+1}: {sample_str}")

unique_samples = len(set(sample_strings))
print(f"  Total samples: {len(data_no_a)}, Unique samples: {unique_samples}")

if unique_samples == len(data_no_a):
    print("  ✓ All samples remain distinguishable without feature A")

    # Try to build tree with B and C only
    features_no_a = ['B', 'C']
    tree_no_a = build_decision_tree(data_no_a, features_no_a.copy())

    print("\nDecision tree using only B and C:")
    print_tree(tree_no_a)

    # Check if it achieves zero error

    zero_error_no_a = check_zero_error(tree_no_a, data_no_a)
    print(f"\nTree with B and C only achieves zero error: {zero_error_no_a}")

    if zero_error_no_a:
        print("  ✓ We can achieve zero error without feature A!")
        print("  ✓ The original tree is NOT optimal")
    else:
        print("  ✗ Cannot achieve zero error without feature A")
        print("  ✓ The original tree IS optimal")

else:
    print("  ✗ Samples are not distinguishable without feature A")
    print("  ✓ Feature A is necessary for perfect classification")

# Task 5: Minimum possible depth
print("\n5. MINIMUM DEPTH ANALYSIS")
print("-" * 40)

def calculate_min_depth(data, features):
    """Calculate minimum depth needed for perfect classification"""

    # If all samples have same label, depth 0
    labels = [sample['Y'] for sample in data]
    if len(set(labels)) == 1:
        return 0

    # Try each feature
    min_depth = float('inf')

    for feature in features:
        feature_values = set(sample[feature] for sample in data)

        # Check if this feature can separate the data perfectly
        can_separate = True
        max_child_depth = 0

        for value in feature_values:
            subset = [sample for sample in data if sample[feature] == value]
            if len(subset) == 0:
                continue

            subset_labels = [sample['Y'] for sample in subset]

            # If subset has mixed labels, need to recurse
            if len(set(subset_labels)) > 1:
                remaining_features = [f for f in features if f != feature]
                if len(remaining_features) == 0:
                    can_separate = False
                    break
                child_depth = calculate_min_depth(subset, remaining_features)
                if child_depth == float('inf'):
                    can_separate = False
                    break
                max_child_depth = max(max_child_depth, child_depth)

        if can_separate:
            min_depth = min(min_depth, 1 + max_child_depth)

    return min_depth if min_depth != float('inf') else float('inf')

# Calculate minimum depth for original dataset
min_depth_original = calculate_min_depth(data, features.copy())
print(f"Minimum depth for original dataset: {min_depth_original}")

# Calculate minimum depth without A
min_depth_no_a = calculate_min_depth(data_no_a, ['B', 'C'])
print(f"Minimum depth without feature A: {min_depth_no_a}")

# Generate visualizations
print("\n6. GENERATING VISUALIZATIONS")
print("-" * 40)

# Create individual information gain visualizations

# 1. Information gain bar plot
fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
bars = ax1.bar(features, [information_gains[f] for f in features],
               color=['darkorange', 'lightblue', 'lightgreen'], alpha=0.8, width=0.6)
ax1.set_title('Information Gain by Feature', fontsize=14, fontweight='bold', pad=20)
ax1.set_xlabel('Feature', fontsize=12)
ax1.set_ylabel('Information Gain', fontsize=12)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim(0, 0.35)

# Add values on bars with better formatting
for i, (bar, feature) in enumerate(zip(bars, features)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             '.3f', ha='center', va='bottom', fontweight='bold', fontsize=11)
    # Add feature explanation
    if feature == 'A':
        ax1.text(bar.get_x() + bar.get_width()/2., height/2, 'Best\nRoot\nSplit',
                 ha='center', va='center', fontweight='bold', color='white', fontsize=9)
    elif feature in ['B', 'C']:
        ax1.text(bar.get_x() + bar.get_width()/2., height/2, 'No\nSplit\nPower',
                 ha='center', va='center', fontweight='bold', color='black', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'information_gain_bar.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Dataset visualization with color coding
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
ax2.axis('off')
ax2.set_title('Dataset Overview', fontsize=14, fontweight='bold', pad=20)

# Create enhanced table with color coding
table_data = [['Sample', 'A', 'B', 'C', 'Y']]
colors = []
for i, sample in enumerate(data):
    row = [f'{i+1}', sample['A'], sample['B'], sample['C'], sample['Y']]
    table_data.append(row)
    # Color code rows based on class
    colors.append(['lightcoral' if sample['Y'] == 'F' else 'lightblue'] * 5)

table = ax2.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.3, 2.5)

# Color the cells
for i in range(len(table_data)):
    for j in range(len(table_data[0])):
        cell = table[(i, j)]
        if i > 0:  # Skip header row
            cell.set_facecolor(colors[i-1][j])
            if j == 4:  # Target column
                cell.set_facecolor('lightcoral' if table_data[i][j] == 'F' else 'lightblue')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'dataset_overview.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Feature correlation heatmaps - separate plots for each feature
feature_class_matrix = np.zeros((3, 2, 2))  # 3 features, 2 values each, 2 classes

for sample in data:
    for f_idx, feature in enumerate(features):
        f_val = 0 if sample[feature] == 'F' else 1
        class_val = 0 if sample['Y'] == 'F' else 1
        feature_class_matrix[f_idx, f_val, class_val] += 1

# Plot heatmap for each feature separately
for f_idx, feature in enumerate(features):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.set_title(f'{feature} vs Class Relationship', fontsize=14, fontweight='bold', pad=20)

    matrix = feature_class_matrix[f_idx]
    im = ax.imshow(matrix, cmap='Blues', aspect='auto', alpha=0.8)

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, int(matrix[i, j]),
                         ha="center", va="center", color="black", fontweight='bold')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['F', 'T'])
    ax.set_yticklabels(['F', 'T'])
    ax.set_xlabel('Class')
    ax.set_ylabel(f'{feature} Value')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{feature.lower()}_class_relationship.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Create individual decision tree visualizations

def create_tree_graph(node, G=None, parent=None, edge_label="", pos=None, x=0, y=0, level_spacing=2, node_spacing=3, node_counter=None):
    """Create NetworkX graph from decision tree"""
    if G is None:
        G = nx.DiGraph()
        pos = {}
        node_counter = {'count': 0}

    if node['type'] == 'leaf':
        node_counter['count'] += 1
        node_id = f"Leaf_{node['class']}_{node_counter['count']}"
        label = f"Class {node['class']}\n({node['samples']} samples)"
        G.add_node(node_id, label=label, shape='box', style='filled', fillcolor='lightblue')
        pos[node_id] = (x, y)
    else:
        node_counter['count'] += 1
        node_id = f"Node_{node['feature']}_{node_counter['count']}"
        label = f"{node['feature']}\n({node['samples']} samples)"
        G.add_node(node_id, label=label, shape='ellipse', style='filled', fillcolor='lightgreen')
        pos[node_id] = (x, y)

    if parent is not None:
        G.add_edge(parent, node_id, label=edge_label)

    if node['type'] == 'node':
        children = list(node['children'].items())
        n_children = len(children)
        start_x = x - (n_children - 1) * node_spacing / 2

        for i, (value, child) in enumerate(children):
            child_x = start_x + i * node_spacing
            child_y = y + level_spacing  # Changed from y - level_spacing to y + level_spacing
            create_tree_graph(child, G, node_id, f"{node['feature']}={value}",
                            pos, child_x, child_y, level_spacing, node_spacing/2, node_counter)

    return G, pos

# 1. Original tree visualization
fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
ax1.axis('off')
ax1.set_title('Original Decision Tree\n(A as root, depth 3)', fontsize=14, fontweight='bold', pad=20)

if 'decision_tree' in locals():
    G1, pos1 = create_tree_graph(decision_tree)
    pos_attrs = {}
    for node, coords in pos1.items():
        pos_attrs[node] = (coords[0] * 100 + 400, coords[1] * 80 + 100)  # Changed from -80 + 300 to 80 + 100

    nx.draw(G1, pos_attrs, ax=ax1, with_labels=True, node_color='lightcoral',
            node_size=1800, font_size=9, font_weight='bold',
            arrows=True, arrowstyle='->', arrowsize=15)

    edge_labels = nx.get_edge_attributes(G1, 'label')
    nx.draw_networkx_edge_labels(G1, pos_attrs, edge_labels, ax=ax1, font_size=8)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'original_decision_tree.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Alternative tree visualization
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
ax2.axis('off')
ax2.set_title('Optimal Decision Tree\n(B as root, depth 2)', fontsize=14, fontweight='bold', pad=20)

if 'alternative_tree' in locals():
    G2, pos2 = create_tree_graph(alternative_tree)
    pos_attrs2 = {}
    for node, coords in pos2.items():
        pos_attrs2[node] = (coords[0] * 100 + 400, coords[1] * 80 + 100)  # Changed from -80 + 300 to 80 + 100

    nx.draw(G2, pos_attrs2, ax=ax2, with_labels=True, node_color='lightblue',
            node_size=1800, font_size=9, font_weight='bold',
            arrows=True, arrowstyle='->', arrowsize=15)

    edge_labels2 = nx.get_edge_attributes(G2, 'label')
    nx.draw_networkx_edge_labels(G2, pos_attrs2, edge_labels2, ax=ax2, font_size=8)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'optimal_decision_tree.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Tree structure comparison
fig3, ax3 = plt.subplots(1, 1, figsize=(8, 6))
ax3.set_title('Tree Structure Comparison', fontsize=14, fontweight='bold', pad=20)

methods = ['Original Tree\n(A→B→C)', 'Optimal Tree\n(B→C in both)', 'Minimum\nPossible']
depths = [3, 2, 2]
colors = ['lightcoral', 'lightblue', 'lightgreen']

bars = ax3.bar(methods, depths, color=colors, alpha=0.8, width=0.6)
ax3.set_ylabel('Tree Depth', fontsize=12)
ax3.set_xlabel('Tree Structure', fontsize=12)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.set_ylim(0, 3.5)

for bar, depth in zip(bars, depths):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'Depth {depth}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'tree_structure_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. Performance comparison
fig4, ax4 = plt.subplots(1, 1, figsize=(8, 6))
ax4.set_title('Performance Comparison', fontsize=14, fontweight='bold', pad=20)

methods = ['Original Tree', 'Optimal Tree']
accuracy = [100, 100]  # Both achieve 100% accuracy
complexity = [3, 2]  # Depth complexity

x = np.arange(len(methods))
width = 0.35

bars1 = ax4.bar(x - width/2, accuracy, width, label='Accuracy (%)', color='lightgreen', alpha=0.8)
bars2 = ax4.bar(x + width/2, complexity, width, label='Depth', color='lightcoral', alpha=0.8)

ax4.set_xlabel('Tree Structure')
ax4.set_ylabel('Value')
ax4.set_xticks(x)
ax4.set_xticklabels(methods)
ax4.legend()
ax4.grid(True, alpha=0.3, linestyle='--')

# Add value labels
for bars, values in [(bars1, accuracy), (bars2, complexity)]:
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{value}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create individual depth and performance comparison plots

# 1. Depth comparison with visual indicators
fig1, ax1 = plt.subplots(1, 1, figsize=(10, 7))
methods = ['Original Tree\n(A→B→C)', 'Optimal Tree\n(B→C in both)', 'Tree without A\n(B,C only)']
depths = [3, 2, min_depth_no_a if min_depth_no_a != float('inf') else 0]
colors = ['lightcoral', 'lightblue', 'lightgray']

bars = ax1.bar(methods, depths, color=colors, alpha=0.8, width=0.6)
ax1.set_title('Tree Depth Comparison', fontsize=14, fontweight='bold', pad=20)
ax1.set_ylabel('Depth', fontsize=12)
ax1.set_xlabel('Tree Structure', fontsize=12)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim(0, 3.5)

for i, (method, depth) in enumerate(zip(methods, depths)):
    ax1.text(i, depth + 0.1, f'{depth}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    if depth == 2:
        ax1.text(i, depth/2, 'Optimal', ha='center', va='center', fontweight='bold', color='darkgreen')
    elif depth > 2:
        ax1.text(i, depth/2, 'Suboptimal', ha='center', va='center', fontweight='bold', color='darkred')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'depth_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Performance metrics comparison
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 7))
ax2.set_title('Performance Metrics', fontsize=14, fontweight='bold', pad=20)

methods_perf = ['Original Tree', 'Optimal Tree', 'Tree without A']
accuracy = [100, 100, 0]  # Only first two achieve 100% accuracy
depths_perf = [3, 2, min_depth_no_a if min_depth_no_a != float('inf') else 0]

x = np.arange(len(methods_perf))
width = 0.35

bars1 = ax2.bar(x - width/2, accuracy, width, label='Accuracy (%)', color='lightgreen', alpha=0.8)
bars2 = ax2.bar(x + width/2, depths_perf, width, label='Depth', color='lightcoral', alpha=0.8)

ax2.set_xlabel('Tree Structure')
ax2.set_ylabel('Value')
ax2.set_xticks(x)
ax2.set_xticklabels(methods_perf)
ax2.legend()
ax2.grid(True, alpha=0.3, linestyle='--')

# Add value labels
for bars, values in [(bars1, accuracy), (bars2, depths_perf)]:
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                 f'{value}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Efficiency comparison (accuracy per depth)
fig3, ax3 = plt.subplots(1, 1, figsize=(8, 6))
ax3.set_title('Efficiency: Accuracy per Depth', fontsize=14, fontweight='bold', pad=20)

methods_eff = ['Original Tree', 'Optimal Tree']
efficiency = [100/3, 100/2]  # Accuracy divided by depth
colors_eff = ['lightcoral', 'lightblue']

bars_eff = ax3.bar(methods_eff, efficiency, color=colors_eff, alpha=0.8, width=0.6)
ax3.set_ylabel('Efficiency (Accuracy/Depth)', fontsize=12)
ax3.set_xlabel('Tree Structure', fontsize=12)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.set_ylim(0, 55)

for i, (bar, eff) in enumerate(zip(bars_eff, efficiency)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
             '.1f', ha='center', va='bottom', fontweight='bold')
    if i == 1:  # Optimal tree
        ax3.text(bar.get_x() + bar.get_width()/2., height/2,
                 'Most\nEfficient', ha='center', va='center', fontweight='bold', color='darkgreen')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'efficiency_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. Feature usage comparison
fig4, ax4 = plt.subplots(1, 1, figsize=(8, 6))
ax4.set_title('Feature Usage', fontsize=14, fontweight='bold', pad=20)

features = ['A', 'B', 'C']
original_usage = [1, 1, 1]  # All features used in original tree
optimal_usage = [0, 1, 1]   # Only B and C used in optimal tree

x = np.arange(len(features))
width = 0.35

bars_orig = ax4.bar(x - width/2, original_usage, width, label='Original Tree', color='lightcoral', alpha=0.8)
bars_opt = ax4.bar(x + width/2, optimal_usage, width, label='Optimal Tree', color='lightblue', alpha=0.8)

ax4.set_xlabel('Feature')
ax4.set_ylabel('Usage (1=Used, 0=Not Used)')
ax4.set_xticks(x)
ax4.set_xticklabels(features)
ax4.legend()
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.set_ylim(0, 1.2)

# Add usage indicators
for i, (orig, opt) in enumerate(zip(original_usage, optimal_usage)):
    if orig == 1:
        ax4.text(i - width/2, 0.5, 'Used', ha='center', va='center', fontweight='bold', color='white')
    if opt == 1:
        ax4.text(i + width/2, 0.5, 'Used', ha='center', va='center', fontweight='bold', color='white')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_usage.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nImages saved to: {save_dir}")
print("Generated files:")
print("  - information_gain_bar.png")
print("  - dataset_overview.png")
print("  - a_class_relationship.png")
print("  - b_class_relationship.png")
print("  - c_class_relationship.png")
print("  - original_decision_tree.png")
print("  - optimal_decision_tree.png")
print("  - tree_structure_comparison.png")
print("  - performance_comparison.png")
print("  - depth_comparison.png")
print("  - performance_metrics.png")
print("  - efficiency_comparison.png")
print("  - feature_usage.png")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
