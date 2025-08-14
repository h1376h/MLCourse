import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import log2
import os
from itertools import combinations

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_8")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

print("=" * 80)
print("QUESTION 8: MULTI-ALGORITHM CONSTRUCTION TRACE")
print("=" * 80)

# Dataset: Restaurant Recommendation
data = {
    'Cuisine': ['Italian', 'Chinese', 'Italian', 'Mexican', 'Chinese', 'Mexican'],
    'Price': ['Low', 'High', 'High', 'Low', 'Low', 'High'],
    'Rating': ['Good', 'Poor', 'Good', 'Poor', 'Good', 'Good'],
    'Busy': ['No', 'Yes', 'No', 'Yes', 'Yes', 'No'],
    'Recommend': ['Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
}

df = pd.DataFrame(data)
print("DATASET: Restaurant Recommendation")
print("=" * 40)
print(df.to_string(index=True))

# Helper functions
def calculate_entropy(labels):
    """Calculate entropy of a list of labels"""
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropy = -sum(p * log2(p) for p in probabilities if p > 0)
    return entropy

def calculate_gini(labels):
    """Calculate Gini impurity of a list of labels"""
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    gini = 1 - sum(p**2 for p in probabilities)
    return gini

def calculate_information_gain(parent_labels, children_labels_list):
    """Calculate information gain for a split"""
    parent_entropy = calculate_entropy(parent_labels)
    total_samples = len(parent_labels)
    
    weighted_child_entropy = 0
    for child_labels in children_labels_list:
        if len(child_labels) > 0:
            child_weight = len(child_labels) / total_samples
            child_entropy = calculate_entropy(child_labels)
            weighted_child_entropy += child_weight * child_entropy
    
    information_gain = parent_entropy - weighted_child_entropy
    return information_gain, parent_entropy, weighted_child_entropy

def calculate_gain_ratio(parent_labels, children_labels_list):
    """Calculate gain ratio (C4.5 criterion)"""
    information_gain, _, _ = calculate_information_gain(parent_labels, children_labels_list)
    
    # Calculate split information
    total_samples = len(parent_labels)
    split_info = 0
    for child_labels in children_labels_list:
        if len(child_labels) > 0:
            proportion = len(child_labels) / total_samples
            split_info -= proportion * log2(proportion)
    
    if split_info == 0:
        return 0
    
    gain_ratio = information_gain / split_info
    return gain_ratio, information_gain, split_info

def calculate_gini_gain(parent_labels, children_labels_list):
    """Calculate Gini gain for CART"""
    parent_gini = calculate_gini(parent_labels)
    total_samples = len(parent_labels)
    
    weighted_child_gini = 0
    for child_labels in children_labels_list:
        if len(child_labels) > 0:
            child_weight = len(child_labels) / total_samples
            child_gini = calculate_gini(child_labels)
            weighted_child_gini += child_weight * child_gini
    
    gini_gain = parent_gini - weighted_child_gini
    return gini_gain, parent_gini, weighted_child_gini

# Calculate baseline entropy and Gini
target = df['Recommend'].tolist()
baseline_entropy = calculate_entropy(target)
baseline_gini = calculate_gini(target)

print(f"\nBASELINE METRICS")
print("=" * 40)
print(f"Total samples: {len(target)}")
print(f"Class distribution: {dict(pd.Series(target).value_counts())}")
print(f"Baseline entropy: {baseline_entropy:.4f}")
print(f"Baseline Gini impurity: {baseline_gini:.4f}")

# Features to evaluate
features = ['Cuisine', 'Price', 'Rating', 'Busy']

print(f"\n" + "="*80)
print("1. ID3 APPROACH - INFORMATION GAIN")
print("="*80)

id3_results = {}
for feature in features:
    print(f"\nFeature: {feature}")
    print("-" * 30)
    
    unique_values = df[feature].unique()
    children_labels = []
    
    print("Split breakdown:")
    for value in unique_values:
        subset = df[df[feature] == value]['Recommend'].tolist()
        children_labels.append(subset)
        subset_counts = dict(pd.Series(subset).value_counts())
        print(f"  {value}: {subset} → {subset_counts}")
    
    ig, parent_ent, weighted_child_ent = calculate_information_gain(target, children_labels)
    
    print(f"Calculation:")
    print(f"  Parent entropy: {parent_ent:.4f}")
    print(f"  Weighted child entropy: {weighted_child_ent:.4f}")
    print(f"  Information Gain: {parent_ent:.4f} - {weighted_child_ent:.4f} = {ig:.4f}")
    
    id3_results[feature] = ig

# Find best feature for ID3
best_id3_feature = max(id3_results, key=id3_results.get)
print(f"\nID3 RESULTS:")
for feature, ig in sorted(id3_results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {feature}: {ig:.4f}")
print(f"ID3 would choose: {best_id3_feature}")

print(f"\n" + "="*80)
print("2. C4.5 APPROACH - GAIN RATIO")
print("="*80)

c45_results = {}
for feature in features:
    print(f"\nFeature: {feature}")
    print("-" * 30)
    
    unique_values = df[feature].unique()
    children_labels = []
    
    for value in unique_values:
        subset = df[df[feature] == value]['Recommend'].tolist()
        children_labels.append(subset)
    
    gr, ig, split_info = calculate_gain_ratio(target, children_labels)
    
    print(f"Information Gain: {ig:.4f}")
    print(f"Split Information: {split_info:.4f}")
    print(f"Gain Ratio: {ig:.4f} / {split_info:.4f} = {gr:.4f}")
    
    c45_results[feature] = gr

# Find best feature for C4.5
best_c45_feature = max(c45_results, key=c45_results.get)
print(f"\nC4.5 RESULTS:")
for feature, gr in sorted(c45_results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {feature}: {gr:.4f}")
print(f"C4.5 would choose: {best_c45_feature}")

print(f"\n" + "="*80)
print("3. CART APPROACH - BINARY SPLITS WITH GINI")
print("="*80)

cart_results = {}

# For Cuisine feature - demonstrate all binary splits
print(f"\nDetailed analysis for Cuisine feature:")
print("-" * 40)
cuisine_values = df['Cuisine'].unique()
print(f"Cuisine values: {list(cuisine_values)}")

# Generate all possible binary splits for Cuisine
all_binary_splits = []
for r in range(1, len(cuisine_values)):
    for subset in combinations(cuisine_values, r):
        split_1 = list(subset)
        split_2 = [v for v in cuisine_values if v not in subset]
        all_binary_splits.append((split_1, split_2))

print(f"\nAll possible binary splits for Cuisine:")
cuisine_gini_gains = []

for i, (set1, set2) in enumerate(all_binary_splits):
    print(f"\nSplit {i+1}: {set1} vs {set2}")
    
    # Get labels for each split
    mask1 = df['Cuisine'].isin(set1)
    mask2 = df['Cuisine'].isin(set2)
    
    labels1 = df[mask1]['Recommend'].tolist()
    labels2 = df[mask2]['Recommend'].tolist()
    
    children_labels = [labels1, labels2]
    gini_gain, parent_gini, weighted_child_gini = calculate_gini_gain(target, children_labels)
    cuisine_gini_gains.append(gini_gain)
    
    print(f"  Group 1 ({set1}): {labels1} → {dict(pd.Series(labels1).value_counts())}")
    print(f"  Group 2 ({set2}): {labels2} → {dict(pd.Series(labels2).value_counts())}")
    print(f"  Gini Gain: {gini_gain:.4f}")

best_cuisine_split_idx = np.argmax(cuisine_gini_gains)
best_cuisine_split = all_binary_splits[best_cuisine_split_idx]
best_cuisine_gini = cuisine_gini_gains[best_cuisine_split_idx]

print(f"\nBest Cuisine split: {best_cuisine_split[0]} vs {best_cuisine_split[1]}")
print(f"Best Cuisine Gini Gain: {best_cuisine_gini:.4f}")

cart_results['Cuisine'] = best_cuisine_gini

# For other features, calculate best binary split
for feature in ['Price', 'Rating', 'Busy']:
    feature_values = df[feature].unique()
    
    if len(feature_values) == 2:
        # Already binary
        labels1 = df[df[feature] == feature_values[0]]['Recommend'].tolist()
        labels2 = df[df[feature] == feature_values[1]]['Recommend'].tolist()
        children_labels = [labels1, labels2]
        gini_gain, _, _ = calculate_gini_gain(target, children_labels)
        cart_results[feature] = gini_gain
    else:
        # Find best binary split
        best_gini = 0
        for r in range(1, len(feature_values)):
            for subset in combinations(feature_values, r):
                set1 = list(subset)
                set2 = [v for v in feature_values if v not in subset]
                
                mask1 = df[feature].isin(set1)
                mask2 = df[feature].isin(set2)
                
                labels1 = df[mask1]['Recommend'].tolist()
                labels2 = df[mask2]['Recommend'].tolist()
                
                children_labels = [labels1, labels2]
                gini_gain, _, _ = calculate_gini_gain(target, children_labels)
                
                if gini_gain > best_gini:
                    best_gini = gini_gain
        
        cart_results[feature] = best_gini

# Find best feature for CART
best_cart_feature = max(cart_results, key=cart_results.get)
print(f"\nCART RESULTS (best binary splits):")
for feature, gini_gain in sorted(cart_results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {feature}: {gini_gain:.4f}")
print(f"CART would choose: {best_cart_feature}")

print(f"\n" + "="*80)
print("4. ALGORITHM COMPARISON")
print("="*80)
print(f"ID3 choice:  {best_id3_feature} (Information Gain: {id3_results[best_id3_feature]:.4f})")
print(f"C4.5 choice: {best_c45_feature} (Gain Ratio: {c45_results[best_c45_feature]:.4f})")
print(f"CART choice: {best_cart_feature} (Gini Gain: {cart_results[best_cart_feature]:.4f})")

# Check if choices are different
all_choices = [best_id3_feature, best_c45_feature, best_cart_feature]
if len(set(all_choices)) == 1:
    print(f"\nAll algorithms agree on choosing: {best_id3_feature}")
else:
    print(f"\nAlgorithms disagree! Different splitting criteria lead to different choices.")

# Create visualizations
fig = plt.figure(figsize=(20, 15))

# Plot 1: Information Gain comparison (ID3)
ax1 = plt.subplot(2, 4, 1)
features_sorted = sorted(id3_results.keys(), key=lambda x: id3_results[x], reverse=True)
ig_values = [id3_results[f] for f in features_sorted]
colors = ['red' if f == best_id3_feature else 'skyblue' for f in features_sorted]

bars1 = ax1.bar(features_sorted, ig_values, color=colors, alpha=0.7, edgecolor='black')
ax1.set_title('ID3: Information Gain')
ax1.set_ylabel('Information Gain')
ax1.set_xticklabels(features_sorted, rotation=45)
ax1.grid(True, alpha=0.3)

for bar, value in zip(bars1, ig_values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot 2: Gain Ratio comparison (C4.5)
ax2 = plt.subplot(2, 4, 2)
features_sorted_c45 = sorted(c45_results.keys(), key=lambda x: c45_results[x], reverse=True)
gr_values = [c45_results[f] for f in features_sorted_c45]
colors_c45 = ['red' if f == best_c45_feature else 'lightgreen' for f in features_sorted_c45]

bars2 = ax2.bar(features_sorted_c45, gr_values, color=colors_c45, alpha=0.7, edgecolor='black')
ax2.set_title('C4.5: Gain Ratio')
ax2.set_ylabel('Gain Ratio')
ax2.set_xticklabels(features_sorted_c45, rotation=45)
ax2.grid(True, alpha=0.3)

for bar, value in zip(bars2, gr_values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot 3: Gini Gain comparison (CART)
ax3 = plt.subplot(2, 4, 3)
features_sorted_cart = sorted(cart_results.keys(), key=lambda x: cart_results[x], reverse=True)
gini_values = [cart_results[f] for f in features_sorted_cart]
colors_cart = ['red' if f == best_cart_feature else 'lightcoral' for f in features_sorted_cart]

bars3 = ax3.bar(features_sorted_cart, gini_values, color=colors_cart, alpha=0.7, edgecolor='black')
ax3.set_title('CART: Gini Gain')
ax3.set_ylabel('Gini Gain')
ax3.set_xticklabels(features_sorted_cart, rotation=45)
ax3.grid(True, alpha=0.3)

for bar, value in zip(bars3, gini_values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: Algorithm choices summary
ax4 = plt.subplot(2, 4, 4)
algorithms = ['ID3', 'C4.5', 'CART']
chosen_features = [best_id3_feature, best_c45_feature, best_cart_feature]
alg_colors = ['skyblue', 'lightgreen', 'lightcoral']

bars4 = ax4.bar(algorithms, [1, 1, 1], color=alg_colors, alpha=0.7, edgecolor='black')
ax4.set_title('Algorithm Choices')
ax4.set_ylabel('Selection')
ax4.set_ylim(0, 1.2)

for bar, choice in zip(bars4, chosen_features):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
             choice, ha='center', va='center', fontweight='bold', rotation=90)

# Plot 5: Dataset visualization
ax5 = plt.subplot(2, 4, 5)
ax5.axis('tight')
ax5.axis('off')

# Create dataset table with color coding
table_data = df.copy()
table_data.index = range(1, len(table_data) + 1)

table = ax5.table(cellText=table_data.values,
                 colLabels=table_data.columns,
                 rowLabels=table_data.index,
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)

# Color code target column
for i in range(len(table_data)):
    if table_data.iloc[i]['Recommend'] == 'Yes':
        table[(i+1, -1)].set_facecolor('lightgreen')
    else:
        table[(i+1, -1)].set_facecolor('lightcoral')

# Header styling
for j in range(len(table_data.columns)):
    table[(0, j)].set_facecolor('#1976D2')
    table[(0, j)].set_text_props(weight='bold', color='white')

ax5.set_title('Restaurant Dataset', pad=20)

# Plot 6: Cuisine binary splits for CART
ax6 = plt.subplot(2, 4, 6)
split_labels = [f"Split {i+1}" for i in range(len(all_binary_splits))]
bars6 = ax6.bar(split_labels, cuisine_gini_gains, 
               color=['red' if i == best_cuisine_split_idx else 'lightcoral' 
                     for i in range(len(cuisine_gini_gains))],
               alpha=0.7, edgecolor='black')

ax6.set_title('CART: Cuisine Binary Splits')
ax6.set_ylabel('Gini Gain')
ax6.set_xlabel('Binary Split')
ax6.grid(True, alpha=0.3)

for bar, value in zip(bars6, cuisine_gini_gains):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{value:.3f}', ha='center', va='bottom', fontsize=8)

# Plot 7 & 8: Decision tree sketches for each algorithm
# This would be the first level of trees that each algorithm would construct

# For ID3 tree
ax7 = plt.subplot(2, 4, 7)
ax7.set_xlim(0, 10)
ax7.set_ylim(0, 8)
ax7.axis('off')

# Root node
root_rect = plt.Rectangle((4, 6), 2, 1, facecolor='skyblue', edgecolor='black')
ax7.add_patch(root_rect)
ax7.text(5, 6.5, best_id3_feature, ha='center', va='center', fontweight='bold')

# Child nodes for ID3's choice
unique_vals = df[best_id3_feature].unique()
child_positions = [(1, 3), (5, 3), (8, 3)]
for i, val in enumerate(unique_vals[:3]):  # Show up to 3 children
    if i < len(child_positions):
        x, y = child_positions[i]
        child_rect = plt.Rectangle((x-0.7, y-0.4), 1.4, 0.8, 
                                 facecolor='lightblue', edgecolor='black')
        ax7.add_patch(child_rect)
        ax7.text(x, y, val, ha='center', va='center', fontsize=9)
        
        # Draw edge
        ax7.plot([5, x], [6, y+0.4], 'k-', linewidth=1)

ax7.set_title(f'ID3 Tree\n(Root: {best_id3_feature})', fontweight='bold')

# For C4.5 tree
ax8 = plt.subplot(2, 4, 8)
ax8.set_xlim(0, 10)
ax8.set_ylim(0, 8)
ax8.axis('off')

# Root node
root_rect = plt.Rectangle((4, 6), 2, 1, facecolor='lightgreen', edgecolor='black')
ax8.add_patch(root_rect)
ax8.text(5, 6.5, best_c45_feature, ha='center', va='center', fontweight='bold')

# Child nodes for C4.5's choice
unique_vals_c45 = df[best_c45_feature].unique()
for i, val in enumerate(unique_vals_c45[:3]):  # Show up to 3 children
    if i < len(child_positions):
        x, y = child_positions[i]
        child_rect = plt.Rectangle((x-0.7, y-0.4), 1.4, 0.8, 
                                 facecolor='#C8E6C9', edgecolor='black')
        ax8.add_patch(child_rect)
        ax8.text(x, y, val, ha='center', va='center', fontsize=9)
        
        # Draw edge
        ax8.plot([5, x], [6, y+0.4], 'k-', linewidth=1)

ax8.set_title(f'C4.5 Tree\n(Root: {best_c45_feature})', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'multi_algorithm_comparison.png'), dpi=300, bbox_inches='tight')

# Create detailed calculation breakdown figure
fig2, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')

# Create detailed calculation text
calculation_text = f"""
DETAILED CALCULATION BREAKDOWN

Dataset: 6 samples, Target: {dict(pd.Series(target).value_counts())}
Baseline Entropy: {baseline_entropy:.4f}
Baseline Gini: {baseline_gini:.4f}

ID3 (Information Gain):
"""

for feature in features:
    unique_values = df[feature].unique()
    children_labels = []
    for value in unique_values:
        subset = df[df[feature] == value]['Recommend'].tolist()
        children_labels.append(subset)
    
    ig, parent_ent, weighted_child_ent = calculate_information_gain(target, children_labels)
    calculation_text += f"\n{feature}: IG = {parent_ent:.4f} - {weighted_child_ent:.4f} = {ig:.4f}"

calculation_text += f"\nBest: {best_id3_feature}\n\nC4.5 (Gain Ratio):"

for feature in features:
    unique_values = df[feature].unique()
    children_labels = []
    for value in unique_values:
        subset = df[df[feature] == value]['Recommend'].tolist()
        children_labels.append(subset)
    
    gr, ig, split_info = calculate_gain_ratio(target, children_labels)
    calculation_text += f"\n{feature}: GR = {ig:.4f} / {split_info:.4f} = {gr:.4f}"

calculation_text += f"\nBest: {best_c45_feature}\n\nCART (Gini Gain):"

for feature in features:
    gini_gain = cart_results[feature]
    calculation_text += f"\n{feature}: Gini Gain = {gini_gain:.4f}"

calculation_text += f"\nBest: {best_cart_feature}"

ax.text(0.05, 0.95, calculation_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

ax.set_title('Detailed Algorithm Calculations', fontsize=14, fontweight='bold')

plt.savefig(os.path.join(save_dir, 'detailed_calculations.png'), dpi=300, bbox_inches='tight')

print(f"\n" + "="*80)
print("5. FIRST LEVEL DECISION TREES")
print("="*80)
print("All three algorithms would create different tree structures based on their")
print("chosen root features. The specific splits would be:")
print(f"- ID3: Split on {best_id3_feature}")
print(f"- C4.5: Split on {best_c45_feature}") 
print(f"- CART: Split on {best_cart_feature}")

print(f"\nImages saved to: {save_dir}")
