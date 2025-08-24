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
    
    # Create comprehensive visualization showing the need for discretization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ID3 and Continuous Features: Discretization Requirement', fontsize=14, fontweight='bold')
    
    # Generate continuous data with clear pattern
    np.random.seed(42)
    X_continuous = np.random.normal(0, 1, (200, 1))
    y = (X_continuous.ravel() > 0).astype(int)
    
    # Add some noise for realism
    noise_indices = np.random.choice(len(y), size=20, replace=False)
    y[noise_indices] = 1 - y[noise_indices]
    
    # Original continuous data
    ax1 = axes[0, 0]
    colors = ['red' if label == 0 else 'blue' for label in y]
    ax1.scatter(X_continuous, y + np.random.normal(0, 0.05, len(y)), alpha=0.6, c=colors)
    ax1.set_xlabel('Continuous Feature Value')
    ax1.set_ylabel('Class Label (with jitter)')
    ax1.set_title('Original Continuous Data\n(ID3 Cannot Process This)')
    ax1.grid(True)
    ax1.axvline(x=0, color='green', linestyle='--', alpha=0.7, label='Natural Boundary')
    ax1.legend()
    
    # Multiple discretization approaches
    ax2 = axes[0, 1]
    
    # Equal-width discretization
    bins_equal_width = np.linspace(X_continuous.min(), X_continuous.max(), 4)
    X_disc_equal = np.digitize(X_continuous.ravel(), bins_equal_width)
    
    # Equal-frequency discretization  
    bins_equal_freq = np.percentile(X_continuous, [0, 33.33, 66.67, 100])
    X_disc_freq = np.digitize(X_continuous.ravel(), bins_equal_freq)
    
    # Plot discretized versions
    unique_bins = np.unique(X_disc_equal)
    colors_disc = plt.cm.Set3(np.linspace(0, 1, len(unique_bins)))
    
    for i, bin_val in enumerate(unique_bins):
        mask = X_disc_equal == bin_val
        if np.sum(mask) > 0:
            ax2.scatter(X_disc_equal[mask] + np.random.normal(0, 0.1, np.sum(mask)), 
                       y[mask] + np.random.normal(0, 0.05, np.sum(mask)), 
                       alpha=0.6, c=[colors_disc[i]], label=f'Bin {bin_val}', s=30)
    
    ax2.set_xlabel('Discretized Feature (Bin Number)')
    ax2.set_ylabel('Class Label (with jitter)')
    ax2.set_title('After Equal-Width Discretization\n(ID3 Can Process This)')
    ax2.legend()
    ax2.grid(True)
    
    # Information gain calculation for discretized version
    ax3 = axes[1, 0]
    
    # Calculate entropy for original problem
    p_class_0 = np.sum(y == 0) / len(y)
    p_class_1 = np.sum(y == 1) / len(y)
    original_entropy = calculate_entropy([p_class_0, p_class_1])
    
    # Calculate conditional entropy for discretized version
    conditional_entropy = 0
    bin_entropies = []
    bin_sizes = []
    bin_labels = []
    
    for bin_val in unique_bins:
        mask = X_disc_equal == bin_val
        if np.sum(mask) > 0:
            bin_y = y[mask]
            bin_p0 = np.sum(bin_y == 0) / len(bin_y)
            bin_p1 = np.sum(bin_y == 1) / len(bin_y)
            bin_entropy = calculate_entropy([bin_p0, bin_p1]) if bin_p0 > 0 and bin_p1 > 0 else 0
            weight = np.sum(mask) / len(y)
            conditional_entropy += weight * bin_entropy
            bin_entropies.append(bin_entropy)
            bin_sizes.append(np.sum(mask))
            bin_labels.append(f'Bin {bin_val}\n({np.sum(mask)} samples)')
    
    information_gain = original_entropy - conditional_entropy
    
    bars = ax3.bar(range(len(bin_entropies)), bin_entropies, alpha=0.7, color=colors_disc[:len(bin_entropies)])
    ax3.set_ylabel('Entropy')
    ax3.set_xlabel('Discretized Bins')
    ax3.set_title(f'Entropy by Bin\nInformation Gain: {information_gain:.4f}')
    ax3.set_xticks(range(len(bin_entropies)))
    ax3.set_xticklabels(bin_labels, rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Add entropy values on bars
    for bar, entropy in zip(bars, bin_entropies):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{entropy:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Comparison of algorithms and discretization methods
    ax4 = axes[1, 1]
    ax4.set_title('Solutions for Continuous Features', fontweight='bold')
    
    solutions_text = [
        "ID3 Limitations:",
        "❌ Cannot handle continuous features",
        "❌ Requires preprocessing",
        "❌ Information loss through binning",
        "",
        "Discretization Methods:",
        "1️⃣ Equal-width binning",
        "   • Divide range into equal intervals",
        f"   • Simple but may miss data distribution",
        "",
        "2️⃣ Equal-frequency binning", 
        "   • Equal samples per bin",
        "   • Better for skewed distributions",
        "",
        "3️⃣ Entropy-based discretization",
        "   • Maximize information gain",
        "   • Optimal for decision trees",
        "",
        "4️⃣ Domain knowledge discretization",
        "   • Use expert knowledge",
        "   • Most interpretable",
        "",
        "Modern Alternatives:",
        "✅ C4.5: Handles continuous natively",
        "✅ CART: Binary splits on thresholds",
        "✅ Random Forest: Built-in handling"
    ]
    
    for i, line in enumerate(solutions_text):
        if line.startswith('❌'):
            color = 'red'
            fontweight = 'bold'
        elif line.startswith('✅'):
            color = 'green'
            fontweight = 'bold'
        elif any(line.startswith(x) for x in ['1️⃣', '2️⃣', '3️⃣', '4️⃣']):
            color = 'blue'
            fontweight = 'bold'
        elif line.endswith(':'):
            color = 'black'
            fontweight = 'bold'
        else:
            color = 'black'
            fontweight = 'normal'
            
        ax4.text(0.05, 0.95 - i*0.035, line, fontsize=8, transform=ax4.transAxes,
                color=color, fontweight=fontweight)
    
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement1_id3_continuous.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'statement': "ID3 can handle continuous features directly without any preprocessing",
        'is_true': False,
        'explanation': "ID3 was originally designed for categorical features and cannot handle continuous features directly. Continuous features must be discretized (binned) before ID3 can use them, as the algorithm's information gain calculation relies on discrete partitions of the feature space.",
        'key_points': [
            "ID3 uses categorical splits only",
            "Requires discretization of continuous features", 
            "Information gain calculation needs discrete partitions",
            "Modern algorithms like C4.5 and CART handle continuous features directly"
        ]
    }

def statement2_gain_ratio_vs_information_gain():
    """Statement 2: C4.5's gain ratio always produces the same feature ranking as ID3's information gain"""
    print_step_header(2, "Statement 2: Gain Ratio vs Information Gain")
    
    print("Statement: C4.5's gain ratio always produces the same feature ranking as ID3's information gain")
    print("Analysis: Examining how gain ratio normalization affects feature selection")
    
    # Create a comprehensive dataset where gain ratio and information gain disagree
    np.random.seed(42)
    n_samples = 200
    
    # Feature 1: Binary feature with good predictive power
    feature1 = np.random.choice(['High', 'Low'], n_samples, p=[0.6, 0.4])
    target1 = []
    for f in feature1:
        if f == 'High':
            target1.append(np.random.choice(['Success', 'Failure'], p=[0.8, 0.2]))
        else:
            target1.append(np.random.choice(['Success', 'Failure'], p=[0.2, 0.8]))
    
    # Feature 2: Multi-valued feature (customer ID-like) with some predictive power
    feature2 = [f'ID_{i//4}' for i in range(n_samples)]  # Creates ~50 unique IDs
    target2 = []
    for f in feature2:
        # Add slight predictive power based on ID number
        id_num = int(f.split('_')[1])
        if id_num % 3 == 0:
            target2.append(np.random.choice(['Success', 'Failure'], p=[0.7, 0.3]))
        else:
            target2.append(np.random.choice(['Success', 'Failure'], p=[0.45, 0.55]))
    
    # Feature 3: Categorical feature with moderate number of values
    feature3 = np.random.choice(['A', 'B', 'C', 'D', 'E'], n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1])
    target3 = []
    for f in feature3:
        if f in ['A', 'B']:
            target3.append(np.random.choice(['Success', 'Failure'], p=[0.65, 0.35]))
        else:
            target3.append(np.random.choice(['Success', 'Failure'], p=[0.4, 0.6]))
    
    def calculate_comprehensive_metrics(feature_values, target_values, feature_name):
        """Calculate detailed metrics for a feature"""
        unique_values = list(set(feature_values))
        total_samples = len(target_values)
        
        print(f"\n--- {feature_name} Analysis ---")
        print(f"Unique values: {len(unique_values)}")
        
        # Calculate original entropy
        target_counts = {}
        for t in target_values:
            target_counts[t] = target_counts.get(t, 0) + 1
        
        p_success = target_counts.get('Success', 0) / total_samples
        p_failure = target_counts.get('Failure', 0) / total_samples
        original_entropy = calculate_entropy([p_success, p_failure])
        
        print(f"Original entropy: {original_entropy:.4f}")
        
        # Calculate conditional entropy and split information
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
            
            subset_p_success = subset_counts.get('Success', 0) / subset_size
            subset_p_failure = subset_counts.get('Failure', 0) / subset_size
            subset_entropy = calculate_entropy([subset_p_success, subset_p_failure])
            
            weight = subset_size / total_samples
            conditional_entropy += weight * subset_entropy
            
            # Calculate split information
            if weight > 0:
                split_info -= weight * np.log2(weight)
        
        information_gain = original_entropy - conditional_entropy
        gain_ratio = information_gain / split_info if split_info > 0 else 0
        
        print(f"Information gain: {information_gain:.4f}")
        print(f"Split information: {split_info:.4f}")
        print(f"Gain ratio: {gain_ratio:.4f}")
        
        return information_gain, gain_ratio, split_info, len(unique_values)
    
    # Calculate metrics for all features using the same target
    # (to ensure fair comparison, we'll use target1 for all)
    ig1, gr1, si1, unique1 = calculate_comprehensive_metrics(feature1, target1, "Binary Feature")
    ig2, gr2, si2, unique2 = calculate_comprehensive_metrics(feature2, target1, "Multi-ID Feature") 
    ig3, gr3, si3, unique3 = calculate_comprehensive_metrics(feature3, target1, "Categorical Feature")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Information Gain vs Gain Ratio: Bias Correction Analysis', fontsize=14, fontweight='bold')
    
    # Metrics comparison
    ax1 = axes[0, 0]
    features = ['Binary\n(2 values)', 'Multi-ID\n(~50 values)', 'Categorical\n(5 values)']
    x = np.arange(len(features))
    width = 0.35
    
    ig_values = [ig1, ig2, ig3]
    gr_values = [gr1, gr2, gr3]
    
    bars1 = ax1.bar(x - width/2, ig_values, width, label='Information Gain', alpha=0.8, color='blue')
    bars2 = ax1.bar(x + width/2, gr_values, width, label='Gain Ratio', alpha=0.8, color='red')
    
    ax1.set_ylabel('Metric Value')
    ax1.set_title('Information Gain vs Gain Ratio')
    ax1.set_xticks(x)
    ax1.set_xticklabels(features)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                    f'{height:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # Split information analysis
    ax2 = axes[0, 1]
    si_values = [si1, si2, si3]
    unique_counts = [unique1, unique2, unique3]
    
    # Create scatter plot showing relationship
    colors = ['blue', 'red', 'green']
    for i, (si, unique, feature, color) in enumerate(zip(si_values, unique_counts, features, colors)):
        ax2.scatter(unique, si, s=100, c=color, alpha=0.7, label=feature.split('\n')[0])
        ax2.annotate(f'{si:.3f}', (unique, si), xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax2.set_xlabel('Number of Unique Values')
    ax2.set_ylabel('Split Information (Penalty)')
    ax2.set_title('Split Information vs Feature Cardinality')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(unique_counts, si_values, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(unique_counts), max(unique_counts), 100)
    ax2.plot(x_trend, p(x_trend), 'k--', alpha=0.5, label='Trend')
    
    # Ranking comparison
    ax3 = axes[1, 0]
    
    # Determine rankings
    feature_data = [
        ('Binary', ig1, gr1),
        ('Multi-ID', ig2, gr2), 
        ('Categorical', ig3, gr3)
    ]
    
    ig_ranking = sorted(feature_data, key=lambda x: x[1], reverse=True)
    gr_ranking = sorted(feature_data, key=lambda x: x[2], reverse=True)
    
    rankings_agree = [item[0] for item in ig_ranking] == [item[0] for item in gr_ranking]
    
    # Visualize rankings
    y_positions = [2, 1, 0]
    rank_labels = ['1st (Best)', '2nd', '3rd (Worst)']
    
    for i, (ig_item, gr_item) in enumerate(zip(ig_ranking, gr_ranking)):
        # Information gain ranking
        ax3.barh(y_positions[i] + 0.15, 1, height=0.25, color='blue', alpha=0.7, 
                label='Information Gain' if i == 0 else "")
        ax3.text(0.5, y_positions[i] + 0.15, f"{ig_item[0]} ({ig_item[1]:.4f})", 
                ha='center', va='center', fontweight='bold', fontsize=9)
        
        # Gain ratio ranking
        ax3.barh(y_positions[i] - 0.15, 1, height=0.25, color='red', alpha=0.7,
                label='Gain Ratio' if i == 0 else "")
        ax3.text(0.5, y_positions[i] - 0.15, f"{gr_item[0]} ({gr_item[2]:.4f})", 
                ha='center', va='center', fontweight='bold', fontsize=9)
    
    ax3.set_yticks(y_positions)
    ax3.set_yticklabels(rank_labels)
    ax3.set_xlabel('Feature Rankings')
    ax3.set_xlim(0, 1)
    ax3.set_title('Feature Ranking Comparison')
    ax3.legend()
    
    # Add agreement status
    agreement_color = 'green' if rankings_agree else 'red'
    agreement_text = 'RANKINGS AGREE' if rankings_agree else 'RANKINGS DISAGREE'
    ax3.text(0.5, -0.7, agreement_text, ha='center', va='center', 
             fontsize=12, fontweight='bold', color=agreement_color,
             bbox=dict(boxstyle="round,pad=0.3", facecolor=agreement_color, alpha=0.2))
    
    # Mathematical explanation
    ax4 = axes[1, 1]
    ax4.set_title('Mathematical Foundation', fontweight='bold')
    
    explanation = [
        "Information Gain (ID3):",
        "IG = H(S) - H(S|A)",
        "• Measures entropy reduction",
        "• Biased toward high-cardinality features",
        "",
        "Gain Ratio (C4.5):",
        "GR = IG / SplitInfo(S,A)", 
        "• Normalizes by split information",
        "• Penalizes fragmentation",
        "",
        "Split Information:",
        "SI = -Σ (|Sᵢ|/|S|) × log₂(|Sᵢ|/|S|)",
        "• Increases with # of partitions",
        "• Acts as normalization factor",
        "",
        f"Current Results:",
        f"• Binary feature: {unique1} values, SI={si1:.3f}",
        f"• Multi-ID feature: {unique2} values, SI={si2:.3f}",
        f"• Categorical: {unique3} values, SI={si3:.3f}",
        "",
        f"Rankings {'agree' if rankings_agree else 'disagree'} ➜ {'Expected' if not rankings_agree else 'Interesting case!'}"
    ]
    
    for i, line in enumerate(explanation):
        if line.startswith('IG =') or line.startswith('GR =') or line.startswith('SI ='):
            color = 'blue'
            fontweight = 'bold'
        elif 'disagree' in line.lower():
            color = 'red'
            fontweight = 'bold'
        elif 'agree' in line.lower():
            color = 'green'
            fontweight = 'bold'
        elif line.endswith(':'):
            color = 'black'
            fontweight = 'bold'
        else:
            color = 'black'
            fontweight = 'normal'
            
        ax4.text(0.05, 0.95 - i*0.04, line, fontsize=8, transform=ax4.transAxes,
                color=color, fontweight=fontweight)
    
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement2_gain_ratio_vs_ig.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'statement': "C4.5's gain ratio always produces the same feature ranking as ID3's information gain",
        'is_true': False,
        'explanation': "Gain ratio and information gain can produce different feature rankings because gain ratio normalizes information gain by split information, which penalizes features with many values. This correction mechanism can change the relative ordering of features.",
        'key_points': [
            "Gain ratio = Information Gain / Split Information",
            "Split information penalizes features with many values",
            "Normalization can change feature rankings",
            "Addresses ID3's bias toward high-cardinality features"
        ]
    }

def statement3_cart_binary_splits():
    """Statement 3: CART uses only binary splits regardless of the number of values in a categorical feature"""
    print_step_header(3, "Statement 3: CART Binary Splits")
    
    print("Statement: CART uses only binary splits regardless of the number of values in a categorical feature")
    print("Analysis: Examining CART's consistent binary splitting approach")
    
    # Comprehensive demonstration of binary vs multi-way splitting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CART Binary Splits vs Multi-way Splits: Complete Analysis', fontsize=14, fontweight='bold')
    
    # Multi-way split visualization (ID3/C4.5 style)
    ax1 = axes[0, 0]
    ax1.set_title('Multi-way Split (ID3/C4.5)', fontweight='bold', color='blue')
    
    # Root node
    root = Circle((0.5, 0.85), 0.08, facecolor='lightblue', edgecolor='blue', linewidth=2)
    ax1.add_patch(root)
    ax1.text(0.5, 0.85, 'Color\nFeature', ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Multi-way children (one for each value)
    positions = [(0.15, 0.5), (0.35, 0.5), (0.55, 0.5), (0.75, 0.5)]
    colors = ['Red', 'Green', 'Blue', 'Yellow']
    decisions = ['Class A', 'Class B', 'Class A', 'Class C']
    node_colors = ['lightcoral', 'lightgreen', 'lightcoral', 'yellow']
    
    for pos, color, decision, node_color in zip(positions, colors, decisions, node_colors):
        child = Circle(pos, 0.06, facecolor=node_color, edgecolor='black', linewidth=2)
        ax1.add_patch(child)
        ax1.text(pos[0], pos[1], f'{color}\n→{decision}', ha='center', va='center', fontweight='bold', fontsize=7)
        ax1.plot([0.5, pos[0]], [0.77, pos[1] + 0.06], 'k-', linewidth=2)
    
    # Add rule summary
    ax1.text(0.5, 0.25, 'Single Decision Rule:\n"If Color = X, then Class = Y"', 
             ha='center', va='center', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    ax1.text(0.5, 0.1, '✓ Natural categorical mapping\n✓ One branch per value\n✓ 4 values → 4 branches', 
             ha='center', va='center', fontsize=9, color='blue', fontweight='bold')
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # Binary split visualization (CART style)
    ax2 = axes[0, 1]
    ax2.set_title('Binary Split (CART)', fontweight='bold', color='red')
    
    # Root node with binary question
    root_bin = Circle((0.5, 0.85), 0.08, facecolor='lightcoral', edgecolor='red', linewidth=2)
    ax2.add_patch(root_bin)
    ax2.text(0.5, 0.85, 'Color ∈\n{Red,Blue}?', ha='center', va='center', fontweight='bold', fontsize=8)
    
    # First level binary split
    left1 = Circle((0.3, 0.65), 0.07, facecolor='lightcoral', edgecolor='red', linewidth=2)
    right1 = Circle((0.7, 0.65), 0.07, facecolor='lightcoral', edgecolor='red', linewidth=2)
    ax2.add_patch(left1)
    ax2.add_patch(right1)
    ax2.text(0.3, 0.65, 'Yes\n{Red,Blue}', ha='center', va='center', fontweight='bold', fontsize=8)
    ax2.text(0.7, 0.65, 'No\n{Green,Yellow}', ha='center', va='center', fontweight='bold', fontsize=8)
    
    # Second level splits
    left2a = Circle((0.2, 0.45), 0.06, facecolor='lightgreen', edgecolor='green', linewidth=2)
    left2b = Circle((0.4, 0.45), 0.06, facecolor='lightgreen', edgecolor='green', linewidth=2)
    right2 = Circle((0.7, 0.45), 0.06, facecolor='lightcoral', edgecolor='red', linewidth=2)
    
    ax2.add_patch(left2a)
    ax2.add_patch(left2b)
    ax2.add_patch(right2)
    
    ax2.text(0.2, 0.45, 'Red\n→Class A', ha='center', va='center', fontweight='bold', fontsize=7)
    ax2.text(0.4, 0.45, 'Blue\n→Class A', ha='center', va='center', fontweight='bold', fontsize=7)
    ax2.text(0.7, 0.45, 'Green∈\n{Green}?', ha='center', va='center', fontweight='bold', fontsize=7)
    
    # Third level
    right3a = Circle((0.65, 0.25), 0.05, facecolor='lightgreen', edgecolor='green', linewidth=2)
    right3b = Circle((0.75, 0.25), 0.05, facecolor='lightgreen', edgecolor='green', linewidth=2)
    ax2.add_patch(right3a)
    ax2.add_patch(right3b)
    
    ax2.text(0.65, 0.25, 'Green\n→B', ha='center', va='center', fontweight='bold', fontsize=6)
    ax2.text(0.75, 0.25, 'Yellow\n→C', ha='center', va='center', fontweight='bold', fontsize=6)
    
    # Draw all edges
    edges = [
        ((0.5, 0.77), (0.3, 0.72)), ((0.5, 0.77), (0.7, 0.72)),  # Root to level 1
        ((0.3, 0.58), (0.2, 0.51)), ((0.3, 0.58), (0.4, 0.51)),  # Left level 1 to level 2
        ((0.7, 0.58), (0.7, 0.51)),  # Right level 1 to level 2
        ((0.7, 0.39), (0.65, 0.31)), ((0.7, 0.39), (0.75, 0.31))  # Level 2 to level 3
    ]
    
    for start, end in edges:
        ax2.plot([start[0], end[0]], [start[1], end[1]], 'k-', linewidth=1.5)
    
    # Add rule summary
    ax2.text(0.5, 0.1, 'Multiple Binary Rules:\n✓ Always 2 branches\n✓ 4 values → 3 binary questions', 
             ha='center', va='center', fontsize=9, color='red', fontweight='bold')
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # Comprehensive algorithm comparison
    ax3 = axes[1, 0]
    ax3.set_title('Complete Algorithm Comparison', fontweight='bold')
    
    comparison_data = [
        ['Aspect', 'ID3/C4.5', 'CART'],
        ['Split Type', 'Multi-way', 'Binary'],
        ['Categorical Handling', 'One branch per value', 'Binary partitions'],
        ['Continuous Features', 'Requires discretization', 'Natural thresholds'],
        ['Tree Depth', 'Often shallower', 'Often deeper'],
        ['Decision Complexity', 'Simple per node', 'Multiple nodes'],
        ['Missing Values', 'Complex (C4.5)', 'Surrogate splits'],
        ['Interpretability', 'Domain-natural', 'Consistent structure'],
        ['Feature Selection', 'All values at once', 'Optimal binary splits']
    ]
    
    # Create table
    table = ax3.table(cellText=comparison_data[1:], colLabels=comparison_data[0],
                     cellLoc='center', loc='center', colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    
    # Color header
    for i in range(3):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight key differences
    key_rows = [1, 2, 8]  # Split Type, Categorical Handling, Feature Selection
    for row in key_rows:
        for col in [1, 2]:
            table[(row, col)].set_facecolor('#fff2cc')
    
    ax3.axis('off')
    
    # Detailed pros and cons analysis
    ax4 = axes[1, 1]
    ax4.set_title('CART Binary Splits: Detailed Analysis', fontweight='bold')
    
    analysis_text = [
        "CART Binary Split Advantages:",
        "✅ Consistent structure across all feature types",
        "✅ Can find optimal partitions for categories",
        "✅ Handles continuous and categorical uniformly",
        "✅ Better for automated processing",
        "✅ Robust missing value handling",
        "",
        "Potential Disadvantages:",
        "❌ May create deeper trees",
        "❌ Less intuitive for natural categories",
        "❌ More complex for simple categorical rules",
        "❌ May fragment natural groupings",
        "",
        "Key CART Principle:",
        "🎯 Always exactly 2 branches per node",
        "🎯 Finds best binary partition for any feature",
        "🎯 Optimizes split quality over interpretability",
        "",
        "Example with 6 categories:",
        "• Multi-way: 6 branches",
        "• CART: Max 5 binary questions (log₂(6)≈2.6)",
    ]
    
    for i, line in enumerate(analysis_text):
        if line.startswith('✅'):
            color = 'green'
            fontweight = 'bold'
        elif line.startswith('❌'):
            color = 'red'
            fontweight = 'bold'
        elif line.startswith('🎯'):
            color = 'blue'
            fontweight = 'bold'
        elif line.endswith(':'):
            color = 'black'
            fontweight = 'bold'
        else:
            color = 'black'
            fontweight = 'normal'
            
        ax4.text(0.05, 0.95 - i*0.04, line, fontsize=8, transform=ax4.transAxes,
                color=color, fontweight=fontweight)
    
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement3_cart_binary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'statement': "CART uses only binary splits regardless of the number of values in a categorical feature",
        'is_true': True,
        'explanation': "CART always creates exactly two branches at each split, regardless of the feature type or number of possible values. For categorical features with multiple values, CART finds the optimal way to partition all possible values into two groups, unlike ID3 and C4.5 which create one branch for each possible value.",
        'key_points': [
            "CART creates exactly two branches at each split",
            "For categorical features, finds optimal binary partitions",
            "Differs from ID3/C4.5 multi-way splits",
            "Allows for more flexible handling of categorical variables"
        ]
    }

def run_enhanced_statements():
    """Run the enhanced analysis for statements 1-3 as a demonstration"""
    print("Decision Tree Algorithms: Enhanced Analysis (Statements 1-3)")
    print("=" * 60)
    
    results = []
    
    # Run enhanced statements 1-3
    results.append(statement1_id3_continuous_features())
    results.append(statement2_gain_ratio_vs_information_gain())
    results.append(statement3_cart_binary_splits())
    
    return results

if __name__ == "__main__":
    results = run_enhanced_statements()
    
    print("\n" + "="*80)
    print("ENHANCED ANALYSIS RESULTS (Statements 1-3)")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        print(f"\nStatement {i}: {result['statement']}")
        print(f"Answer: {'TRUE' if result['is_true'] else 'FALSE'}")
        print(f"Explanation: {result['explanation']}")
        if 'key_points' in result:
            print("Key Points:")
            for point in result['key_points']:
                print(f"  • {point}")
        
    print(f"\nEnhanced visualizations saved to: {save_dir}")
    print("Note: This demonstrates the enhanced approach for statements 1-3.")
    print("The complete solution would include all 10 statements with this level of detail.")
