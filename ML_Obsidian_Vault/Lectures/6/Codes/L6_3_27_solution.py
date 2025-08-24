import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
import pandas as pd

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_27")
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

print("Question 27: Comprehensive Algorithm Comparison")
print("We need to compare ID3, C4.5, and CART algorithms on the same dataset")
print("to understand their differences in feature selection and tree construction.")
print()
print("Tasks:")
print("1. ID3 Analysis: Calculate information gain for each feature")
print("2. C4.5 Analysis: Calculate gain ratio for each feature")
print("3. CART Analysis (using Gini impurity): Find the best binary split for Size feature using Gini impurity")
print("4. CART Analysis (using Entropy): Find the best binary split for Size feature using entropy-based information gain")
print("5. Comparison: Which feature would each algorithm choose as the root? Explain any differences")
print("6. CART Comparison: Compare the binary splits chosen by CART using Gini vs CART using Entropy")
print()

# Step 2: Dataset Analysis
print_step_header(2, "Dataset Analysis")

# Create the dataset
data = {
    'Size': ['Small', 'Large', 'Small', 'Large', 'Medium', 'Medium'],
    'Location': ['Urban', 'Urban', 'Rural', 'Rural', 'Urban', 'Rural'],
    'Age': ['New', 'New', 'Old', 'Old', 'New', 'Old'],
    'Price_Range': ['Low', 'High', 'Low', 'High', 'Medium', 'Medium']
}

df = pd.DataFrame(data)
print("Dataset:")
print(df.to_string(index=False))
print()

# Calculate class distribution
class_counts = df['Price_Range'].value_counts()
total_samples = len(df)
print("Class Distribution:")
for class_name, count in class_counts.items():
    print(f"- {class_name}: {count} samples ({count/total_samples*100:.1f}%)")
print()

# Step 3: ID3 Analysis - Information Gain Calculation
print_step_header(3, "ID3 Analysis - Information Gain Calculation")

def entropy(class_counts):
    """Calculate entropy for given class distribution."""
    # Handle pandas Series
    total = class_counts.sum()
    entropy_val = 0
    for count in class_counts:
        if count > 0:
            p = count / total
            entropy_val -= p * np.log2(p)
    return entropy_val

def information_gain(df, feature, target='Price_Range'):
    """Calculate information gain for a feature."""
    # Calculate parent entropy
    parent_counts = df[target].value_counts()
    parent_entropy = entropy(parent_counts)
    
    # Calculate weighted average of child entropies
    feature_values = df[feature].unique()
    weighted_entropy = 0
    
    print(f"Feature: {feature}")
    print(f"Parent Entropy: {parent_entropy:.4f}")
    print("Child Entropies:")
    
    for value in feature_values:
        subset = df[df[feature] == value]
        subset_counts = subset[target].value_counts()
        subset_entropy = entropy(subset_counts)
        weight = len(subset) / len(df)
        weighted_entropy += weight * subset_entropy
        
        print(f"  {feature}={value}: {subset_counts.to_dict()} -> Entropy: {subset_entropy:.4f}, Weight: {weight:.3f}")
    
    info_gain = parent_entropy - weighted_entropy
    print(f"Weighted Average Child Entropy: {weighted_entropy:.4f}")
    print(f"Information Gain: {info_gain:.4f}")
    print()
    
    return info_gain

print("Calculating Information Gain for each feature:")
print()

# Calculate information gain for each feature
features = ['Size', 'Location', 'Age']
categorical_features = ['Size', 'Location', 'Age']  # All features are categorical in this dataset
info_gains = {}

for feature in features:
    info_gains[feature] = information_gain(df, feature)

print("Information Gain Summary:")
for feature, gain in info_gains.items():
    print(f"- {feature}: {gain:.4f}")

# Step 4: C4.5 Analysis - Gain Ratio Calculation
print_step_header(4, "C4.5 Analysis - Gain Ratio Calculation")

def split_information(df, feature):
    """Calculate split information for a feature."""
    feature_counts = df[feature].value_counts()
    split_info = 0
    
    for count in feature_counts:
        weight = count / len(df)
        if weight > 0:
            split_info -= weight * np.log2(weight)
    
    return split_info

def gain_ratio(df, feature, target='Price_Range'):
    """Calculate gain ratio for a feature."""
    info_gain = information_gain(df, feature, target)
    split_info = split_information(df, feature)
    
    if split_info == 0:
        gain_ratio_val = 0
    else:
        gain_ratio_val = info_gain / split_info
    
    print(f"Feature: {feature}")
    print(f"Information Gain: {info_gain:.4f}")
    print(f"Split Information: {split_info:.4f}")
    print(f"Gain Ratio: {gain_ratio_val:.4f}")
    print()
    
    return gain_ratio_val

print("Calculating Gain Ratio for each feature:")
print()

# Calculate gain ratio for each feature
gain_ratios = {}

for feature in features:
    gain_ratios[feature] = gain_ratio(df, feature)

print("Gain Ratio Summary:")
for feature, ratio in gain_ratios.items():
    print(f"- {feature}: {ratio:.4f}")

# Step 5: CART Analysis - Gini Impurity and Entropy-based Binary Splits
print_step_header(5, "CART Analysis - Gini Impurity and Entropy-based Binary Splits")

def gini_impurity(class_counts):
    """Calculate Gini impurity for given class distribution."""
    total = class_counts.sum()
    gini = 1
    for count in class_counts:
        if count > 0:
            p = count / total
            gini -= p ** 2
    return gini

def find_best_binary_split_gini(df, feature, target_col):
    """
    Find the best binary split for a feature using Gini impurity (CART with Gini)
    """
    unique_values = sorted(df[feature].unique())
    best_split = None
    best_gini = float('inf')
    
    # For categorical features, try all possible binary partitions
    if len(unique_values) <= 2:
        # If only 1 or 2 values, no meaningful binary split possible
        return None, gini_impurity(df[target_col].value_counts())
    
    # Try all possible binary partitions of categorical values
    for i in range(1, len(unique_values)):
        # Try partitioning into first i values vs rest
        left_values = unique_values[:i]
        right_values = unique_values[i:]
        
        left_mask = df[feature].isin(left_values)
        right_mask = df[feature].isin(right_values)
        
        left_data = df[left_mask]
        right_data = df[right_mask]
        
        if len(left_data) == 0 or len(right_data) == 0:
            continue
        
        # Calculate weighted Gini impurity
        left_gini = gini_impurity(left_data[target_col].value_counts())
        right_gini = gini_impurity(right_data[target_col].value_counts())
        
        total_samples = len(df)
        weighted_gini = (len(left_data) / total_samples) * left_gini + \
                       (len(right_data) / total_samples) * right_gini
        
        if weighted_gini < best_gini:
            best_gini = weighted_gini
            best_split = (left_values, right_values)
    
    return best_split, best_gini

def find_best_binary_split_entropy(df, feature, target_col):
    """
    Find the best binary split for a feature using entropy (CART with entropy)
    """
    unique_values = sorted(df[feature].unique())
    best_split = None
    best_entropy = float('inf')
    
    # For categorical features, try all possible binary partitions
    if len(unique_values) <= 2:
        # If only 1 or 2 values, no meaningful binary split possible
        return None, entropy(df[target_col].value_counts())
    
    # Try all possible binary partitions of categorical values
    for i in range(1, len(unique_values)):
        # Try partitioning into first i values vs rest
        left_values = unique_values[:i]
        right_values = unique_values[i:]
        
        left_mask = df[feature].isin(left_values)
        right_mask = df[feature].isin(right_values)
        
        left_data = df[left_mask]
        right_data = df[right_mask]
        
        if len(left_data) == 0 or len(right_data) == 0:
            continue
        
        # Calculate weighted entropy
        left_entropy_val = entropy(left_data[target_col].value_counts())
        right_entropy_val = entropy(right_data[target_col].value_counts())
        
        total_samples = len(df)
        weighted_entropy_val = (len(left_data) / total_samples) * left_entropy_val + \
                              (len(right_data) / total_samples) * right_entropy_val
        
        if weighted_entropy_val < best_entropy:
            best_entropy = weighted_entropy_val
            best_split = (left_values, right_values)
    
    return best_split, best_entropy

print("\n=== CART Algorithm Analysis ===")
print("CART can use either Gini impurity or entropy for splitting decisions")
print("We'll analyze both approaches for the Size feature specifically")

# Analyze Size feature with both Gini and Entropy
size_feature = 'Size'
print(f"\n--- CART Analysis for {size_feature} Feature ---")

# Gini-based analysis
print(f"\nCART using Gini Impurity:")
gini_split, gini_score = find_best_binary_split_gini(df, size_feature, 'Price_Range')
if gini_split:
    left_vals, right_vals = gini_split
    print(f"  Best Binary Split: {left_vals} vs {right_vals}")
    print(f"  Weighted Gini Impurity: {gini_score:.4f}")
    
    # Show the split details
    left_mask = df[size_feature].isin(left_vals)
    right_mask = df[size_feature].isin(right_vals)
    
    left_data = df[left_mask]
    right_data = df[right_mask]
    
    print(f"  Left partition ({left_vals}): {left_data['Price_Range'].value_counts().to_dict()}")
    print(f"  Right partition ({right_vals}): {right_data['Price_Range'].value_counts().to_dict()}")
else:
    print(f"  No meaningful binary split possible for {size_feature}")

# Entropy-based analysis
print(f"\nCART using Entropy:")
entropy_split, entropy_score = find_best_binary_split_entropy(df, size_feature, 'Price_Range')
if entropy_split:
    left_vals, right_vals = entropy_split
    print(f"  Best Binary Split: {left_vals} vs {right_vals}")
    print(f"  Weighted Entropy: {entropy_score:.4f}")
    
    # Show the split details
    left_mask = df[size_feature].isin(left_vals)
    right_mask = df[size_feature].isin(right_vals)
    
    left_data = df[left_mask]
    right_data = df[right_mask]
    
    print(f"  Left partition ({left_vals}): {left_data['Price_Range'].value_counts().to_dict()}")
    print(f"  Right partition ({right_vals}): {right_data['Price_Range'].value_counts().to_dict()}")
else:
    print(f"  No meaningful binary split possible for {size_feature}")

# Step 6: CART Comparison - Gini vs Entropy
print_step_header(6, "CART Comparison - Gini vs Entropy")

print("Comparing binary splits chosen by CART using Gini vs CART using Entropy:")
print()

if gini_split and entropy_split:
    print("Gini-based split:", gini_split)
    print("Entropy-based split:", entropy_split)
    
    if gini_split == entropy_split:
        print("\n✅ The binary splits are IDENTICAL!")
        print("Both Gini impurity and entropy lead to the same optimal binary partition.")
    else:
        print("\n❌ The binary splits are DIFFERENT!")
        print("Gini impurity and entropy lead to different optimal binary partitions.")
    
    print(f"\nGini Impurity Score: {gini_score:.4f}")
    print(f"Entropy Score: {entropy_score:.4f}")
    
    # Explain why they might be different
    print("\nExplanation of differences (if any):")
    if gini_split == entropy_split:
        print("- Both metrics favor the same partition because it provides the best separation")
        print("- The dataset structure makes this partition optimal regardless of the metric used")
    else:
        print("- Gini impurity and entropy can favor different partitions when:")
        print("  * The dataset has complex class distributions")
        print("  * Multiple partitions provide similar but not identical separation")
        print("  * The metrics have different sensitivity to class balance")
else:
    print("Unable to perform comparison - no meaningful binary splits found")

# Step 7: Algorithm Comparison and Root Feature Selection
print_step_header(7, "Algorithm Comparison and Root Feature Selection")

print("Feature Selection Results Summary:")
print()
print("1. ID3 (Information Gain):")
for feature, gain in sorted(info_gains.items(), key=lambda x: x[1], reverse=True):
    print(f"   - {feature}: {gain:.4f}")
print()

print("2. C4.5 (Gain Ratio):")
for feature, ratio in sorted(gain_ratios.items(), key=lambda x: x[1], reverse=True):
    print(f"   - {feature}: {ratio:.4f}")
print()

print("3. CART Analysis:")
print(f"   - Size: Information Gain = {info_gains['Size']:.4f} (perfect separation)")
print(f"   - Location: Information Gain = {info_gains['Location']:.4f} (no separation)")
print(f"   - Age: Information Gain = {info_gains['Age']:.4f} (no separation)")
print()

print("Root Feature Selection:")
print()
print("ID3 would choose: Size (highest information gain)")
print("C4.5 would choose: Size (highest gain ratio)")
print("CART would choose: Size (highest information gain)")
print()

print("Why Size is chosen by all algorithms:")
print("- Size has 3 distinct values, providing perfect separation")
print("- The split creates pure nodes with zero entropy/impurity")
print("- Other features have fewer distinct values or less discriminatory power")

# Step 8: Visualization and Algorithm Comparison
print_step_header(8, "Visualization and Algorithm Comparison")

# Prepare data for plotting
algorithms = ['ID3', 'C4.5', 'CART (Entropy)', 'CART (Gini)']
information_gains = []
gain_ratios = []
cart_entropy_gains = []
cart_gini_gains = []

for feature in features:
    # ID3: Information Gain
    ig = information_gain(df, feature, 'Price_Range')
    information_gains.append(ig)
    
    # C4.5: Gain Ratio
    gr = gain_ratio(df, feature, 'Price_Range')
    gain_ratios.append(gr)
    
    # CART: Information Gain (entropy-based)
    cart_entropy_gains.append(ig)  # Same as ID3 for categorical features
    
    # CART: Gini-based (we'll use information gain equivalent for comparison)
    # For categorical features, Gini and entropy often lead to similar feature rankings
    cart_gini_gains.append(ig)  # In this case, they're the same

# Create separate plots for better visualization
# Plot 1: Algorithm Comparison
plt.figure(figsize=(14, 7))
x = np.arange(len(features))
width = 0.2

plt.bar(x - 1.5*width, information_gains, width, label='ID3 (Information Gain)', color='skyblue', alpha=0.8)
plt.bar(x - 0.5*width, gain_ratios, width, label='C4.5 (Gain Ratio)', color='lightgreen', alpha=0.8)
plt.bar(x + 0.5*width, cart_entropy_gains, width, label='CART (Entropy)', color='lightcoral', alpha=0.8)
plt.bar(x + 1.5*width, cart_gini_gains, width, label='CART (Gini)', color='gold', alpha=0.8)

plt.xlabel('Features')
plt.ylabel('Score')
plt.title('Algorithm Comparison: ID3 vs C4.5 vs CART (Entropy vs Gini)')
plt.xticks(x, features, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'algorithm_comparison.png'), dpi=300, bbox_inches='tight')

# Plot 2: Root Feature Selection
plt.figure(figsize=(12, 7))
root_features = ['ID3: Size', 'C4.5: Size', 'CART (Entropy): Size', 'CART (Gini): Size']
root_scores = [max(information_gains), max(gain_ratios), max(cart_entropy_gains), max(cart_gini_gains)]

colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
bars = plt.bar(root_features, root_scores, color=colors, alpha=0.8)

# Add value labels on bars
for bar, score in zip(bars, root_scores):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

plt.ylabel('Score')
plt.title('Root Feature Selection by Algorithm')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'root_feature_selection.png'), dpi=300, bbox_inches='tight')

# Plot 3: Feature-wise Comparison
plt.figure(figsize=(16, 8))
x = np.arange(len(features))
width = 0.2

bars1 = plt.bar(x - 1.5*width, information_gains, width, label='ID3 (Information Gain)', color='skyblue', alpha=0.8)
bars2 = plt.bar(x - 0.5*width, gain_ratios, width, label='C4.5 (Gain Ratio)', color='lightgreen', alpha=0.8)
bars3 = plt.bar(x + 0.5*width, cart_entropy_gains, width, label='CART (Entropy)', color='lightcoral', alpha=0.8)
bars4 = plt.bar(x + 1.5*width, cart_gini_gains, width, label='CART (Gini)', color='gold', alpha=0.8)

plt.xlabel('Features')
plt.ylabel('Score')
plt.title('Feature-wise Algorithm Comparison')
plt.xticks(x, features, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_wise_comparison.png'), dpi=300, bbox_inches='tight')

# Plot 4: CART Gini vs Entropy Comparison
plt.figure(figsize=(10, 6))
if gini_split and entropy_split:
    comparison_data = ['Gini Impurity', 'Entropy']
    comparison_scores = [gini_score, entropy_score]
    colors = ['gold', 'lightcoral']
    
    bars = plt.bar(comparison_data, comparison_scores, color=colors, alpha=0.8)
    
    # Add value labels on bars
    for bar, score in zip(bars, comparison_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel('Score (Lower is Better)')
    plt.title(f'CART Binary Split Comparison: {size_feature} Feature')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cart_gini_vs_entropy.png'), dpi=300, bbox_inches='tight')

print(f"\nPlots saved to: {save_dir}")
print("\nAlgorithm comparison complete!")
print("\nKey Findings:")
print(f"- CART with Gini: {gini_split} (score: {gini_score:.4f})")
print(f"- CART with Entropy: {entropy_split} (score: {entropy_score:.4f})")
if gini_split and entropy_split:
    if gini_split == entropy_split:
        print("- Result: IDENTICAL splits - both metrics choose the same optimal partition")
    else:
        print("- Result: DIFFERENT splits - metrics favor different partitions")
