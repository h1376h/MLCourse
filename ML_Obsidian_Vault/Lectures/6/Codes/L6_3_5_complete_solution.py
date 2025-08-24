import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_5")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('ggplot')
sns.set_style("whitegrid")
np.random.seed(42)  # For reproducibility

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

def calculate_entropy(probabilities):
    """Calculate entropy given a list of probabilities."""
    entropy = 0
    for p in probabilities:
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy

def calculate_gini(probabilities):
    """Calculate Gini impurity given a list of probabilities."""
    gini = 1
    for p in probabilities:
        gini -= p ** 2
    return gini

def statement1_id3_continuous_features():
    """Statement 1: ID3 can handle continuous features directly without any preprocessing"""
    print_step_header(1, "Statement 1: ID3 and Continuous Features")
    
    print("Statement: ID3 can handle continuous features directly without any preprocessing")
    print("Analysis: ID3 was originally designed for categorical features only")
    
    # Create visualization showing the need for discretization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ID3 and Continuous Features: Need for Discretization', fontsize=14, fontweight='bold')
    
    # Generate continuous data
    np.random.seed(42)
    X_continuous = np.random.normal(0, 1, (100, 1))
    y = (X_continuous.ravel() > 0).astype(int)
    
    # Original continuous data
    ax1 = axes[0, 0]
    ax1.scatter(X_continuous, y, alpha=0.6, c=y, cmap='viridis')
    ax1.set_xlabel('Continuous Feature Value')
    ax1.set_ylabel('Class Label')
    ax1.set_title('Original Continuous Data')
    ax1.grid(True)
    
    # Discretized data
    ax2 = axes[0, 1]
    bins = [-3, -1, 1, 3]
    X_discretized = np.digitize(X_continuous.ravel(), bins)
    ax2.scatter(X_discretized, y, alpha=0.6, c=y, cmap='viridis')
    ax2.set_xlabel('Discretized Feature (Bin Number)')
    ax2.set_ylabel('Class Label')
    ax2.set_title('After Discretization (3 bins)')
    ax2.set_xticks([1, 2, 3])
    ax2.grid(True)
    
    # Information gain calculation
    ax3 = axes[1, 0]
    p_class_0 = np.sum(y == 0) / len(y)
    p_class_1 = np.sum(y == 1) / len(y)
    original_entropy = calculate_entropy([p_class_0, p_class_1])
    
    conditional_entropy = 0
    bin_entropies = []
    
    for bin_val in [1, 2, 3]:
        mask = X_discretized == bin_val
        if np.sum(mask) > 0:
            bin_y = y[mask]
            bin_p0 = np.sum(bin_y == 0) / len(bin_y)
            bin_p1 = np.sum(bin_y == 1) / len(bin_y)
            bin_entropy = calculate_entropy([bin_p0, bin_p1]) if bin_p0 > 0 and bin_p1 > 0 else 0
            weight = np.sum(mask) / len(y)
            conditional_entropy += weight * bin_entropy
            bin_entropies.append(bin_entropy)
    
    information_gain = original_entropy - conditional_entropy
    
    ax3.bar(['Bin 1', 'Bin 2', 'Bin 3'], bin_entropies, alpha=0.7)
    ax3.set_ylabel('Entropy')
    ax3.set_title(f'Entropy by Bin\nInformation Gain: {information_gain:.3f}')
    ax3.grid(True)
    
    # Discretization methods
    ax4 = axes[1, 1]
    ax4.text(0.1, 0.9, 'Common Discretization Methods:', fontsize=12, fontweight='bold', transform=ax4.transAxes)
    ax4.text(0.1, 0.8, '1. Equal-width binning', fontsize=10, transform=ax4.transAxes)
    ax4.text(0.1, 0.7, '2. Equal-frequency binning', fontsize=10, transform=ax4.transAxes)
    ax4.text(0.1, 0.6, '3. Entropy-based discretization', fontsize=10, transform=ax4.transAxes)
    ax4.text(0.1, 0.5, '4. Chi-square discretization', fontsize=10, transform=ax4.transAxes)
    ax4.text(0.1, 0.3, 'Alternative: Use C4.5 or CART', fontsize=12, fontweight='bold', color='red', transform=ax4.transAxes)
    ax4.text(0.1, 0.2, 'These can handle continuous features directly', fontsize=10, color='red', transform=ax4.transAxes)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement1_id3_continuous.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'statement': "ID3 can handle continuous features directly without any preprocessing",
        'is_true': False,
        'explanation': "ID3 was originally designed for categorical features and cannot handle continuous features directly. Continuous features must be discretized (binned) before ID3 can use them."
    }

def statement2_gain_ratio_vs_information_gain():
    """Statement 2: C4.5's gain ratio always produces the same feature ranking as ID3's information gain"""
    print_step_header(2, "Statement 2: Gain Ratio vs Information Gain")
    
    # Create demonstration of different rankings
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Gain Ratio vs Information Gain: Different Rankings', fontsize=14, fontweight='bold')
    
    # Sample calculation showing difference
    feature_many_values = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'] * 12 + ['A', 'B', 'C', 'D']
    feature_few_values = ['X', 'Y'] * 50
    target = []
    
    for i, val in enumerate(feature_many_values):
        if val in ['A', 'B', 'C', 'D']:
            target.append('Yes' if np.random.random() > 0.3 else 'No')
        else:
            target.append('Yes' if np.random.random() > 0.7 else 'No')
    
    def calculate_metrics(feature_values, target_values):
        total_samples = len(target_values)
        
        # Original entropy
        target_counts = {}
        for t in target_values:
            target_counts[t] = target_counts.get(t, 0) + 1
        
        p_yes = target_counts.get('Yes', 0) / total_samples
        p_no = target_counts.get('No', 0) / total_samples
        original_entropy = calculate_entropy([p_yes, p_no])
        
        # Conditional entropy and split info
        unique_values = list(set(feature_values))
        conditional_entropy = 0
        split_info = 0
        
        for value in unique_values:
            subset_indices = [i for i, v in enumerate(feature_values) if v == value]
            subset_size = len(subset_indices)
            
            if subset_size == 0:
                continue
                
            subset_target = [target_values[i] for i in subset_indices]
            subset_counts = {}
            for t in subset_target:
                subset_counts[t] = subset_counts.get(t, 0) + 1
            
            subset_p_yes = subset_counts.get('Yes', 0) / subset_size
            subset_p_no = subset_counts.get('No', 0) / subset_size
            subset_entropy = calculate_entropy([subset_p_yes, subset_p_no])
            
            weight = subset_size / total_samples
            conditional_entropy += weight * subset_entropy
            
            if weight > 0:
                split_info -= weight * np.log2(weight)
        
        information_gain = original_entropy - conditional_entropy
        gain_ratio = information_gain / split_info if split_info > 0 else 0
        
        return information_gain, gain_ratio, split_info
    
    ig_many, gr_many, si_many = calculate_metrics(feature_many_values, target)
    ig_few, gr_few, si_few = calculate_metrics(feature_few_values, target)
    
    # Plot comparisons
    ax1 = axes[0, 0]
    features = ['Many Values\n(A-H)', 'Few Values\n(X-Y)']
    ig_values = [ig_many, ig_few]
    bars1 = ax1.bar(features, ig_values, color=['orange', 'blue'], alpha=0.7)
    ax1.set_ylabel('Information Gain')
    ax1.set_title('Information Gain Comparison')
    
    for bar, value in zip(bars1, ig_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    ax2 = axes[0, 1]
    gr_values = [gr_many, gr_few]
    bars2 = ax2.bar(features, gr_values, color=['orange', 'blue'], alpha=0.7)
    ax2.set_ylabel('Gain Ratio')
    ax2.set_title('Gain Ratio Comparison')
    
    for bar, value in zip(bars2, gr_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Show rankings
    ax3 = axes[1, 0]
    ig_ranking = ['Many Values', 'Few Values'] if ig_many > ig_few else ['Few Values', 'Many Values']
    gr_ranking = ['Many Values', 'Few Values'] if gr_many > gr_few else ['Few Values', 'Many Values']
    
    ax3.text(0.1, 0.8, 'Feature Rankings:', fontsize=14, fontweight='bold', transform=ax3.transAxes)
    ax3.text(0.1, 0.6, 'By Information Gain:', fontsize=12, color='blue', transform=ax3.transAxes)
    ax3.text(0.1, 0.5, f'1st: {ig_ranking[0]}', fontsize=10, transform=ax3.transAxes)
    ax3.text(0.1, 0.3, 'By Gain Ratio:', fontsize=12, color='red', transform=ax3.transAxes)
    ax3.text(0.1, 0.2, f'1st: {gr_ranking[0]}', fontsize=10, transform=ax3.transAxes)
    
    rankings_differ = ig_ranking != gr_ranking
    ax3.text(0.1, 0.05, f'Rankings Differ: {"YES" if rankings_differ else "NO"}', 
             fontsize=12, fontweight='bold', 
             color='red' if rankings_differ else 'green', 
             transform=ax3.transAxes)
    ax3.axis('off')
    
    # Explanation
    ax4 = axes[1, 1]
    ax4.text(0.1, 0.9, 'Why Rankings Can Differ:', fontsize=12, fontweight='bold', transform=ax4.transAxes)
    ax4.text(0.1, 0.8, '• Gain Ratio = IG / Split Information', fontsize=10, transform=ax4.transAxes)
    ax4.text(0.1, 0.7, '• Split Info penalizes many values', fontsize=10, transform=ax4.transAxes)
    ax4.text(0.1, 0.6, '• High IG + High Split Info = Low GR', fontsize=10, transform=ax4.transAxes)
    ax4.text(0.1, 0.5, '• Prevents bias toward fragmentation', fontsize=10, transform=ax4.transAxes)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement2_gain_ratio_vs_ig.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'statement': "C4.5's gain ratio always produces the same feature ranking as ID3's information gain",
        'is_true': False,
        'explanation': "Gain ratio can produce different rankings because it penalizes features with many values through split information normalization."
    }

def statement3_cart_binary_splits():
    """Statement 3: CART uses only binary splits regardless of the number of values in a categorical feature"""
    print_step_header(3, "Statement 3: CART Binary Splits")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CART: Binary Splits for All Feature Types', fontsize=14, fontweight='bold')
    
    # Multi-way vs Binary split comparison
    ax1 = axes[0, 0]
    ax1.set_title('Multi-way Split (ID3/C4.5)', fontweight='bold')
    
    # Root and children for multi-way
    root = Circle((0.5, 0.8), 0.08, facecolor='lightblue', edgecolor='blue', linewidth=2)
    ax1.add_patch(root)
    ax1.text(0.5, 0.8, 'Color', ha='center', va='center', fontweight='bold')
    
    positions = [(0.2, 0.4), (0.4, 0.4), (0.6, 0.4), (0.8, 0.4)]
    colors = ['Red', 'Blue', 'Green', 'Yellow']
    
    for pos, color in zip(positions, colors):
        child = Circle(pos, 0.06, facecolor='lightgreen', edgecolor='green', linewidth=2)
        ax1.add_patch(child)
        ax1.text(pos[0], pos[1], color, ha='center', va='center', fontweight='bold', fontsize=9)
        ax1.plot([0.5, pos[0]], [0.72, pos[1] + 0.06], 'k-', linewidth=2)
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.text(0.5, 0.1, '4 branches for 4 values', ha='center', va='center', fontsize=11)
    
    # Binary split
    ax2 = axes[0, 1]
    ax2.set_title('Binary Split (CART)', fontweight='bold')
    
    root = Circle((0.5, 0.8), 0.08, facecolor='lightcoral', edgecolor='red', linewidth=2)
    ax2.add_patch(root)
    ax2.text(0.5, 0.8, 'Color ∈\n{Red, Blue}?', ha='center', va='center', fontweight='bold', fontsize=9)
    
    left_child = Circle((0.3, 0.4), 0.07, facecolor='lightgreen', edgecolor='green', linewidth=2)
    right_child = Circle((0.7, 0.4), 0.07, facecolor='lightgreen', edgecolor='green', linewidth=2)
    ax2.add_patch(left_child)
    ax2.add_patch(right_child)
    
    ax2.text(0.3, 0.4, 'Yes\n{Red, Blue}', ha='center', va='center', fontweight='bold', fontsize=9)
    ax2.text(0.7, 0.4, 'No\n{Green, Yellow}', ha='center', va='center', fontweight='bold', fontsize=9)
    
    ax2.plot([0.5, 0.3], [0.72, 0.47], 'k-', linewidth=2)
    ax2.plot([0.5, 0.7], [0.72, 0.47], 'k-', linewidth=2)
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.text(0.5, 0.1, 'Always 2 branches', ha='center', va='center', fontsize=11)
    
    # Comparison table
    ax3 = axes[1, 0]
    ax3.set_title('Algorithm Comparison', fontweight='bold')
    
    comparison_data = [
        ['Algorithm', 'Split Type', 'Categorical Features'],
        ['ID3', 'Multi-way', 'One branch per value'],
        ['C4.5', 'Multi-way', 'One branch per value'],
        ['CART', 'Binary', 'Binary partition of values']
    ]
    
    table = ax3.table(cellText=comparison_data[1:], colLabels=comparison_data[0],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(3):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(3):
        table[(3, i)].set_facecolor('#ffcccc')
    
    ax3.axis('off')
    
    # Pros and cons
    ax4 = axes[1, 1]
    ax4.set_title('CART Binary Splits: Pros and Cons', fontweight='bold')
    
    ax4.text(0.05, 0.9, 'Advantages:', fontsize=12, fontweight='bold', color='green', transform=ax4.transAxes)
    ax4.text(0.05, 0.8, '• Consistent tree structure', fontsize=10, transform=ax4.transAxes)
    ax4.text(0.05, 0.75, '• Handles missing values better', fontsize=10, transform=ax4.transAxes)
    ax4.text(0.05, 0.7, '• Can find optimal partitions', fontsize=10, transform=ax4.transAxes)
    
    ax4.text(0.05, 0.5, 'Disadvantages:', fontsize=12, fontweight='bold', color='red', transform=ax4.transAxes)
    ax4.text(0.05, 0.4, '• May create deeper trees', fontsize=10, transform=ax4.transAxes)
    ax4.text(0.05, 0.35, '• Less intuitive for categorical data', fontsize=10, transform=ax4.transAxes)
    
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement3_cart_binary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'statement': "CART uses only binary splits regardless of the number of values in a categorical feature",
        'is_true': True,
        'explanation': "CART always creates binary splits, even for categorical features with multiple values by finding optimal binary partitions."
    }

def statement4_pure_node_entropy():
    """Statement 4: The entropy of a pure node (all samples belong to one class) is always zero"""
    print_step_header(4, "Statement 4: Pure Node Entropy")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Pure Node Entropy Analysis', fontsize=14, fontweight='bold')
    
    # Pure node examples
    ax1 = axes[0, 0]
    scenarios = [("All Class A", [1, 0, 0]), ("All Class B", [0, 1, 0]), ("All Class C", [0, 0, 1])]
    entropies = [calculate_entropy(probs) for _, probs in scenarios]
    
    bars1 = ax1.bar([s[0] for s in scenarios], entropies, color=['red', 'blue', 'green'], alpha=0.7)
    ax1.set_ylabel('Entropy')
    ax1.set_title('Pure Nodes: Entropy = 0')
    ax1.set_ylim(0, 0.1)
    
    for bar, entropy in zip(bars1, entropies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{entropy:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Entropy curve for binary classification
    ax2 = axes[0, 1]
    p_values = np.linspace(0, 1, 11)
    entropies_binary = [calculate_entropy([p, 1-p]) if 0 < p < 1 else 0 for p in p_values]
    
    ax2.plot(p_values, entropies_binary, 'ro-', linewidth=2, markersize=6)
    ax2.set_xlabel('Probability of Class 1')
    ax2.set_ylabel('Entropy')
    ax2.set_title('Entropy vs Class Distribution')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.text(0.1, 0.1, 'Pure\n(p=0)', ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    ax2.text(0.9, 0.1, 'Pure\n(p=1)', ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    
    # Mathematical proof
    ax3 = axes[1, 0]
    ax3.set_title('Mathematical Proof', fontweight='bold')
    
    proof_text = [
        "For a pure node with class C:",
        "P(class C) = 1",
        "P(other classes) = 0",
        "",
        "H(S) = -Σ pᵢ × log₂(pᵢ)",
        "H(S) = -(1 × log₂(1)) - Σ(0 × log₂(0))",
        "",
        "Since log₂(1) = 0 and 0 × log₂(0) = 0:",
        "H(S) = 0"
    ]
    
    for i, line in enumerate(proof_text):
        fontweight = 'bold' if line.startswith('H(S)') else 'normal'
        ax3.text(0.05, 0.9 - i*0.1, line, fontsize=10, transform=ax3.transAxes, fontweight=fontweight)
    
    ax3.axis('off')
    
    # Node examples
    ax4 = axes[1, 1]
    ax4.set_title('Decision Tree Node Examples', fontweight='bold')
    
    examples = [
        ("Pure Leaf\n100% Yes", 0, 'lightgreen'),
        ("Pure Leaf\n100% No", 0, 'lightcoral'),
        ("Mixed Node\n60% Yes, 40% No", calculate_entropy([0.6, 0.4]), 'yellow'),
        ("Balanced Node\n50% Yes, 50% No", calculate_entropy([0.5, 0.5]), 'lightblue')
    ]
    
    y_positions = [0.8, 0.6, 0.4, 0.2]
    
    for i, (label, entropy, color) in enumerate(examples):
        node = Circle((0.3, y_positions[i]), 0.08, facecolor=color, edgecolor='black', linewidth=2)
        ax4.add_patch(node)
        ax4.text(0.3, y_positions[i], label, ha='center', va='center', fontweight='bold', fontsize=8)
        ax4.text(0.6, y_positions[i], f'Entropy = {entropy:.3f}', ha='left', va='center', fontsize=10)
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement4_pure_node_entropy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'statement': "The entropy of a pure node (all samples belong to one class) is always zero",
        'is_true': True,
        'explanation': "A pure node has entropy of zero because there is no uncertainty. Mathematically, when p=1 for one class, H(S) = -1 × log₂(1) = 0."
    }

def statement5_c45_split_info_penalty():
    """Statement 5: C4.5's split information penalizes features with many values to reduce bias"""
    print_step_header(5, "Statement 5: C4.5 Split Information Penalty")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("C4.5's Split Information Penalty", fontsize=14, fontweight='bold')
    
    # Demonstrate bias toward multi-valued features
    n_samples = 100
    
    # Binary feature with predictive power
    feature1 = np.random.choice(['A', 'B'], n_samples, p=[0.6, 0.4])
    target1 = []
    for f in feature1:
        if f == 'A':
            target1.append(np.random.choice(['Yes', 'No'], p=[0.7, 0.3]))
        else:
            target1.append(np.random.choice(['Yes', 'No'], p=[0.3, 0.7]))
    
    # Multi-valued feature with no predictive power
    feature2 = [f'ID_{i}' for i in range(n_samples)]
    target2 = np.random.choice(['Yes', 'No'], n_samples, p=[0.5, 0.5])
    
    def calculate_split_metrics(feature_values, target_values):
        unique_values = list(set(feature_values))
        total_samples = len(target_values)
        
        # Calculate information gain and split information
        target_counts = {}
        for t in target_values:
            target_counts[t] = target_counts.get(t, 0) + 1
        
        p_yes = target_counts.get('Yes', 0) / total_samples
        p_no = target_counts.get('No', 0) / total_samples
        original_entropy = calculate_entropy([p_yes, p_no])
        
        conditional_entropy = 0
        split_info = 0
        
        for value in unique_values:
            subset_mask = [f == value for f in feature_values]
            subset_size = sum(subset_mask)
            
            if subset_size == 0:
                continue
                
            subset_target = [target_values[i] for i, mask in enumerate(subset_mask) if mask]
            subset_counts = {}
            for t in subset_target:
                subset_counts[t] = subset_counts.get(t, 0) + 1
            
            subset_p_yes = subset_counts.get('Yes', 0) / subset_size
            subset_p_no = subset_counts.get('No', 0) / subset_size
            subset_entropy = calculate_entropy([subset_p_yes, subset_p_no])
            
            weight = subset_size / total_samples
            conditional_entropy += weight * subset_entropy
            
            if weight > 0:
                split_info -= weight * np.log2(weight)
        
        information_gain = original_entropy - conditional_entropy
        gain_ratio = information_gain / split_info if split_info > 0 else 0
        
        return information_gain, split_info, gain_ratio, len(unique_values)
    
    ig1, si1, gr1, unique1 = calculate_split_metrics(feature1, target1)
    ig2, si2, gr2, unique2 = calculate_split_metrics(feature2, target2)
    
    # Information Gain comparison
    ax1 = axes[0, 0]
    features = ['Binary\nFeature', 'Multi-valued\nFeature']
    ig_values = [ig1, ig2]
    bars1 = ax1.bar(features, ig_values, color=['blue', 'red'], alpha=0.7)
    ax1.set_ylabel('Information Gain')
    ax1.set_title('Information Gain Comparison')
    
    for bar, value in zip(bars1, ig_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    if ig2 > ig1:
        ax1.text(0.5, max(ig_values) * 0.8, 'Multi-valued feature\nhas higher IG!\n(Bias)', 
                ha='center', va='center', fontsize=10, color='red', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Split Information
    ax2 = axes[0, 1]
    si_values = [si1, si2]
    bars2 = ax2.bar(features, si_values, color=['blue', 'red'], alpha=0.7)
    ax2.set_ylabel('Split Information (Penalty)')
    ax2.set_title('Split Information Comparison')
    
    for bar, value in zip(bars2, si_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Gain Ratio
    ax3 = axes[1, 0]
    gr_values = [gr1, gr2]
    bars3 = ax3.bar(features, gr_values, color=['blue', 'red'], alpha=0.7)
    ax3.set_ylabel('Gain Ratio')
    ax3.set_title('Gain Ratio: After Penalty Applied')
    
    for bar, value in zip(bars3, gr_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    if gr1 > gr2:
        ax3.text(0.5, max(gr_values) * 0.7, 'Binary feature now\nhas higher gain ratio!\n(Bias corrected)', 
                ha='center', va='center', fontsize=10, color='green', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    # Mathematical explanation
    ax4 = axes[1, 1]
    ax4.set_title('Split Information Formula', fontweight='bold')
    
    explanation = [
        "Split Information:",
        "SI = -Σ (|Sᵢ|/|S|) × log₂(|Sᵢ|/|S|)",
        "",
        f"Binary feature ({unique1} values):",
        f"• Split information: {si1:.3f}",
        "",
        f"Multi-valued feature ({unique2} values):",
        f"• Split information: {si2:.3f}",
        "",
        "Gain Ratio = IG / SI",
        "→ Higher SI penalizes the feature",
        "→ Reduces bias toward fragmentation"
    ]
    
    for i, line in enumerate(explanation):
        fontweight = 'bold' if line.startswith('Split Information') or line.startswith('Gain Ratio') else 'normal'
        ax4.text(0.05, 0.95 - i*0.07, line, fontsize=9, transform=ax4.transAxes, fontweight=fontweight)
    
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement5_split_info_penalty.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'statement': "C4.5's split information penalizes features with many values to reduce bias",
        'is_true': True,
        'explanation': "C4.5 uses gain ratio (IG/Split Info) where split information increases with number of feature values, penalizing multi-valued attributes."
    }

def run_all_statements():
    """Run analysis for all statements"""
    print("Decision Tree Algorithms: Evaluating 10 Statements")
    print("=" * 60)
    
    results = []
    
    # Run statements 1-5 (implemented above)
    results.append(statement1_id3_continuous_features())
    results.append(statement2_gain_ratio_vs_information_gain())
    results.append(statement3_cart_binary_splits())
    results.append(statement4_pure_node_entropy())
    results.append(statement5_c45_split_info_penalty())
    
    # Statements 6-10 (basic implementations for completeness)
    results.append({
        'statement': "CART can handle both classification and regression problems using the same tree structure",
        'is_true': True,
        'explanation': "CART uses the same binary tree structure for both problems, with different splitting criteria and leaf predictions."
    })
    
    results.append({
        'statement': "ID3 includes built-in pruning mechanisms to prevent overfitting",
        'is_true': False,
        'explanation': "Original ID3 has no pruning mechanisms. Pruning was added in later algorithms like C4.5."
    })
    
    results.append({
        'statement': "C4.5's handling of missing values is more sophisticated than ID3's approach",
        'is_true': True,
        'explanation': "C4.5 has built-in probabilistic missing value support, while ID3 requires preprocessing."
    })
    
    results.append({
        'statement': "Information gain and Gini impurity always select the same feature for splitting",
        'is_true': False,
        'explanation': "Different mathematical formulations can lead to different feature rankings in certain scenarios."
    })
    
    results.append({
        'statement': "CART's binary splits always result in more interpretable trees than multi-way splits",
        'is_true': False,
        'explanation': "Interpretability is context-dependent. Multi-way splits can be more intuitive for categorical data."
    })
    
    return results

if __name__ == "__main__":
    results = run_all_statements()
    
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        print(f"\nStatement {i}: {result['statement']}")
        print(f"Answer: {'TRUE' if result['is_true'] else 'FALSE'}")
        print(f"Explanation: {result['explanation']}")
        
    print(f"\nAll visualizations saved to: {save_dir}")
