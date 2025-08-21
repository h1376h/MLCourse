import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.patches as mpatches

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_43")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Dataset
data = {
    'Study_Time': ['Short', 'Short', 'Short', 'Short', 'Long', 'Long', 'Long', 'Long'],
    'Study_Location': ['Library', 'Home', 'Cafe', 'Dorm', 'Office', 'Park', 'Lab', 'Study_Room'],
    'Coffee_Consumption': ['None', 'High', 'None', 'High', 'None', 'High', 'None', 'High'],
    'Exam_Result': ['Fail', 'Fail', 'Fail', 'Fail', 'Pass', 'Pass', 'Pass', 'Fail']
}

df = pd.DataFrame(data)
print("Dataset:")
print(df)
print("\n" + "="*80 + "\n")

# Function to calculate entropy
def entropy(y):
    """Calculate entropy of a target variable"""
    if len(y) == 0:
        return 0
    
    # Count unique values and their frequencies
    unique, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    
    # Calculate entropy: -sum(p * log2(p))
    entropy_val = -np.sum(probabilities * np.log2(probabilities))
    return entropy_val

# Function to calculate information gain (ID3 approach)
def information_gain(X, y, feature):
    """Calculate information gain for a feature using ID3 approach"""
    # Calculate entropy of the parent node
    parent_entropy = entropy(y)
    
    # Get unique values of the feature
    unique_values = df[feature].unique()
    
    # Calculate weighted entropy for each value
    weighted_entropy = 0
    total_samples = len(y)
    
    print(f"Calculating Information Gain for feature: {feature}")
    print(f"Parent entropy: H(S) = {parent_entropy:.4f}")
    print(f"Feature values: {unique_values}")
    
    for value in unique_values:
        # Get samples where feature equals this value
        mask = df[feature] == value
        subset_y = y[mask]
        subset_size = len(subset_y)
        
        if subset_size > 0:
            subset_entropy = entropy(subset_y)
            weight = subset_size / total_samples
            weighted_entropy += weight * subset_entropy
            
            print(f"  Value '{value}': {subset_size} samples, entropy = {subset_entropy:.4f}, weight = {weight:.4f}")
            print(f"    Subset: {list(subset_y)}")
    
    # Calculate information gain
    info_gain = parent_entropy - weighted_entropy
    print(f"Weighted entropy: {weighted_entropy:.4f}")
    print(f"Information Gain: IG({feature}) = {parent_entropy:.4f} - {weighted_entropy:.4f} = {info_gain:.4f}")
    print()
    
    return info_gain

# Function to calculate split information (C4.5 approach)
def split_information(X, feature):
    """Calculate split information for a feature using C4.5 approach"""
    unique_values = df[feature].unique()
    total_samples = len(X)
    
    split_info = 0
    print(f"Calculating Split Information for feature: {feature}")
    
    for value in unique_values:
        mask = df[feature] == value
        subset_size = np.sum(mask)
        
        if subset_size > 0:
            p = subset_size / total_samples
            split_info -= p * np.log2(p)
            print(f"  Value '{value}': {subset_size} samples, p = {p:.4f}")
    
    print(f"Split Information: SI({feature}) = {split_info:.4f}")
    print()
    return split_info

# Function to calculate gain ratio (C4.5 approach)
def gain_ratio(X, y, feature):
    """Calculate gain ratio for a feature using C4.5 approach"""
    info_gain = information_gain(X, y, feature)
    split_info = split_information(X, feature)
    
    if split_info == 0:
        gain_ratio_val = 0
    else:
        gain_ratio_val = info_gain / split_info
    
    print(f"Gain Ratio: GR({feature}) = IG({feature}) / SI({feature}) = {info_gain:.4f} / {split_info:.4f} = {gain_ratio_val:.4f}")
    print()
    
    return gain_ratio_val

# Calculate entropy of the entire dataset
print("STEP 1: Calculate the entropy of the entire dataset")
print("Target variable: Exam_Result")
print(f"Values: {list(df['Exam_Result'])}")
print(f"Counts: Pass = {list(df['Exam_Result']).count('Pass')}, Fail = {list(df['Exam_Result']).count('Fail')}")

total_entropy = entropy(df['Exam_Result'])
print(f"Total entropy H(S) = {total_entropy:.4f}")
print("\n" + "="*80 + "\n")

# Calculate information gain for each feature (ID3 approach)
print("STEP 2: Calculate Information Gain for each feature (ID3 approach)")
features = ['Study_Time', 'Study_Location', 'Coffee_Consumption']
info_gains = {}

for feature in features:
    info_gains[feature] = information_gain(df, df['Exam_Result'], feature)

print("="*80 + "\n")

# Calculate gain ratio for each feature (C4.5 approach)
print("STEP 3: Calculate Gain Ratio for each feature (C4.5 approach)")
gain_ratios = {}

for feature in features:
    gain_ratios[feature] = gain_ratio(df, df['Exam_Result'], feature)

print("="*80 + "\n")

# Create summary table
print("STEP 4: Summary Table")
print("-" * 80)
print(f"{'Feature':<20} {'Info Gain (ID3)':<15} {'Split Info':<12} {'Gain Ratio (C4.5)':<18} {'Preferred By':<15}")
print("-" * 80)

for feature in features:
    ig = info_gains[feature]
    si = split_information(df, feature)
    gr = gain_ratios[feature]
    
    # Determine which algorithm prefers this feature
    if ig == max(info_gains.values()):
        preferred_id3 = "ID3"
    else:
        preferred_id3 = ""
        
    if gr == max(gain_ratios.values()):
        preferred_c45 = "C4.5"
    else:
        preferred_c45 = ""
    
    preferred = f"{preferred_id3}/{preferred_c45}".strip("/")
    if not preferred:
        preferred = "Neither"
    
    print(f"{feature:<20} {ig:<15.4f} {si:<12.4f} {gr:<18.4f} {preferred:<15}")

print("-" * 80)
print()

# Determine root nodes for each algorithm
id3_root = max(info_gains, key=info_gains.get)
c45_root = max(gain_ratios, key=gain_ratios.get)

print(f"ID3 would choose: {id3_root} (Information Gain = {info_gains[id3_root]:.4f})")
print(f"C4.5 would choose: {c45_root} (Gain Ratio = {gain_ratios[c45_root]:.4f})")
print()

# Create visualization of the first level decision trees
def create_decision_tree_visualization():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # ID3 Decision Tree
    ax1.set_title('ID3 Decision Tree - First Level', fontsize=16, fontweight='bold')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    # Root node
    root_box = FancyBboxPatch((4, 8), 2, 1, boxstyle="round,pad=0.1", 
                              facecolor='lightblue', edgecolor='black', linewidth=2)
    ax1.add_patch(root_box)
    ax1.text(5, 8.5, f'{id3_root}', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Branches and child nodes
    unique_values = df[id3_root].unique()
    num_values = len(unique_values)
    
    # Calculate positions dynamically based on number of values
    if num_values == 2:
        x_positions = [2, 8]
        y_positions = [6.5, 6.5]
    elif num_values == 3:
        x_positions = [1, 5, 9]
        y_positions = [6.5, 6.5, 6.5]
    else:  # For more values, distribute evenly
        x_positions = np.linspace(1, 9, num_values)
        y_positions = [6.5] * num_values
    
    for i, value in enumerate(unique_values):
        # Draw branch
        ax1.plot([5, x_positions[i]], [8, y_positions[i]], 'k-', linewidth=2)
        
        # Draw child node
        child_box = FancyBboxPatch((x_positions[i]-0.5, y_positions[i]-0.5), 1, 1, 
                                  boxstyle="round,pad=0.1", facecolor='lightgreen', 
                                  edgecolor='black', linewidth=2)
        ax1.add_patch(child_box)
        
        # Add value label on branch
        mid_x = (5 + x_positions[i]) / 2
        mid_y = (8 + y_positions[i]) / 2
        ax1.text(mid_x, mid_y, value, ha='center', va='center', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"))
        
        # Add entropy information
        mask = df[id3_root] == value
        subset_y = df['Exam_Result'][mask]
        subset_entropy = entropy(subset_y)
        subset_counts = subset_y.value_counts()
        
        info_text = f"Entropy: {subset_entropy:.3f}\n"
        for label, count in subset_counts.items():
            info_text += f"{label}: {count}\n"
        
        ax1.text(x_positions[i], y_positions[i]-1.5, info_text.strip(), ha='center', va='top', 
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"))
    
    # C4.5 Decision Tree
    ax2.set_title('C4.5 Decision Tree - First Level', fontsize=16, fontweight='bold')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    # Root node
    root_box = FancyBboxPatch((4, 8), 2, 1, boxstyle="round,pad=0.1", 
                              facecolor='lightcoral', edgecolor='black', linewidth=2)
    ax2.add_patch(root_box)
    ax2.text(5, 8.5, f'{c45_root}', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Branches and child nodes
    unique_values = df[c45_root].unique()
    num_values = len(unique_values)
    
    # Calculate positions dynamically based on number of values
    if num_values == 2:
        x_positions = [2, 8]
        y_positions = [6.5, 6.5]
    elif num_values == 3:
        x_positions = [1, 5, 9]
        y_positions = [6.5, 6.5, 6.5]
    else:  # For more values, distribute evenly
        x_positions = np.linspace(1, 9, num_values)
        y_positions = [6.5] * num_values
    
    for i, value in enumerate(unique_values):
        # Draw branch
        ax2.plot([5, x_positions[i]], [8, y_positions[i]], 'k-', linewidth=2)
        
        # Draw child node
        child_box = FancyBboxPatch((x_positions[i]-0.5, y_positions[i]-0.5), 1, 1, 
                                  boxstyle="round,pad=0.1", facecolor='lightyellow', 
                                  edgecolor='black', linewidth=2)
        ax2.add_patch(child_box)
        
        # Add value label on branch
        mid_x = (5 + x_positions[i]) / 2
        mid_y = (8 + y_positions[i]) / 2
        ax2.text(mid_x, mid_y, value, ha='center', va='center', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"))
        
        # Add entropy information
        mask = df[c45_root] == value
        subset_y = df['Exam_Result'][mask]
        subset_entropy = entropy(subset_y)
        subset_counts = subset_y.value_counts()
        
        info_text = f"Entropy: {subset_entropy:.3f}\n"
        for label, count in subset_counts.items():
            info_text += f"{label}: {count}\n"
        
        ax2.text(x_positions[i], y_positions[i]-1.5, info_text.strip(), ha='center', va='top', 
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'decision_trees_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Create feature comparison visualization
def create_feature_comparison_visualization():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Information Gain comparison
    features_list = list(info_gains.keys())
    ig_values = list(info_gains.values())
    
    bars1 = ax1.bar(features_list, ig_values, color=['lightblue', 'lightgreen', 'lightcoral'], alpha=0.8)
    ax1.set_title('Information Gain Comparison (ID3)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Information Gain')
    ax1.set_ylim(0, max(ig_values) * 1.1)
    
    # Add value labels on bars
    for bar, value in zip(bars1, ig_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Highlight the best feature
    best_ig_idx = np.argmax(ig_values)
    bars1[best_ig_idx].set_color('gold')
    bars1[best_ig_idx].set_edgecolor('black')
    bars1[best_ig_idx].set_linewidth(2)
    
    # Gain Ratio comparison
    gr_values = list(gain_ratios.values())
    
    bars2 = ax2.bar(features_list, gr_values, color=['lightblue', 'lightgreen', 'lightcoral'], alpha=0.8)
    ax2.set_title('Gain Ratio Comparison (C4.5)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Gain Ratio')
    ax2.set_ylim(0, max(gr_values) * 1.1)
    
    # Add value labels on bars
    for bar, value in zip(bars2, gr_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Highlight the best feature
    best_gr_idx = np.argmax(gr_values)
    bars2[best_gr_idx].set_color('gold')
    bars2[best_gr_idx].set_edgecolor('black')
    bars2[best_gr_idx].set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Create entropy distribution visualization
def create_entropy_distribution_visualization():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for plotting
    feature_names = []
    entropy_values = []
    colors = []
    
    for feature in features:
        unique_values = df[feature].unique()
        for value in unique_values:
            mask = df[feature] == value
            subset_y = df['Exam_Result'][mask]
            subset_entropy = entropy(subset_y)
            subset_size = len(subset_y)
            
            feature_names.append(f"{feature}: {value}")
            entropy_values.append(subset_entropy)
            colors.append('lightcoral' if subset_entropy == 0 else 'lightblue')
    
    # Create horizontal bar chart
    bars = ax.barh(range(len(feature_names)), entropy_values, color=colors, alpha=0.8)
    
    # Customize the plot
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=10)
    ax.set_xlabel('Entropy', fontsize=12, fontweight='bold')
    ax.set_title('Entropy Distribution for Each Feature Value', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, entropy_values)):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', ha='left', va='center', fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='lightcoral', label='Pure Split (Entropy = 0)'),
                      Patch(facecolor='lightblue', label='Mixed Split (Entropy > 0)')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'entropy_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Create algorithm comparison visualization
def create_algorithm_comparison_visualization():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data for comparison
    x = np.arange(len(features))
    width = 0.35
    
    # Create grouped bar chart
    bars1 = ax.bar(x - width/2, list(info_gains.values()), width, label='Information Gain (ID3)', 
                   color='lightblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, list(gain_ratios.values()), width, label='Gain Ratio (C4.5)', 
                   color='lightcoral', alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('ID3 vs C4.5: Feature Selection Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(features, fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Highlight the best choice for each algorithm
    best_ig_idx = np.argmax(list(info_gains.values()))
    best_gr_idx = np.argmax(list(gain_ratios.values()))
    
    bars1[best_ig_idx].set_edgecolor('darkblue')
    bars1[best_ig_idx].set_linewidth(2)
    bars2[best_gr_idx].set_edgecolor('darkred')
    bars2[best_gr_idx].set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'algorithm_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Generate all visualizations
print("Generating visualizations...")
create_decision_tree_visualization()
create_feature_comparison_visualization()
create_entropy_distribution_visualization()
create_algorithm_comparison_visualization()

print(f"\nAll visualizations saved to: {save_dir}")

# Answer the specific questions
print("\n" + "="*80)
print("ANSWERS TO SPECIFIC QUESTIONS")
print("="*80)

print("\n1. Information Gain for each feature (ID3 approach):")
for feature, ig in info_gains.items():
    print(f"   {feature}: {ig:.4f}")

print(f"\n   ID3 would choose: {id3_root} (highest information gain)")

print("\n2. Gain Ratio for each feature (C4.5 approach):")
for feature, gr in gain_ratios.items():
    print(f"   {feature}: {gr:.4f}")

print(f"\n   C4.5 would choose: {c45_root} (highest gain ratio)")

print("\n3. First level decision trees:")
print(f"   ID3: Root = {id3_root}")
print(f"   C4.5: Root = {c45_root}")

print("\n4. Summary table created above showing all metrics")

print("\n5. Student advice:")
if id3_root != c45_root:
    print(f"   The algorithms disagree! ID3 suggests focusing on {id3_root}, while C4.5 suggests {c45_root}")
    print(f"   This shows how different splitting criteria can lead to different tree structures")
else:
    print(f"   Both algorithms agree on {id3_root} as the root node")

print("\n6. Fundamental differences:")
print("   - ID3 uses Information Gain which can favor features with many values")
print("   - C4.5 uses Gain Ratio which normalizes by Split Information to reduce bias")
print("   - This normalization helps C4.5 avoid overfitting to features with many unique values")
