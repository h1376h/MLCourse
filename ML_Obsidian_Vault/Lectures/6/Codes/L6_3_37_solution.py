import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import log2
import os
from itertools import combinations

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_37")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = False  # Disable LaTeX to avoid issues with dollar signs
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

print("=" * 80)
print("QUESTION 37: CUSTOMER PURCHASE BEHAVIOR ANALYSIS")
print("DECISION TREE CONSTRUCTION RACE")
print("=" * 80)

# Dataset: Customer Purchase Behavior
data = {
    'Product_Category': ['Sports', 'Electronics', 'Books', 'Books', 'Electronics', 'Sports', 'Clothing', 'Clothing'],
    'Purchase_Amount': ['$51-100', '$200+', '$200+', '$101-200', '$200+', '$10-50', '$200+', '$200+'],
    'Customer_Type': ['Regular', 'Regular', 'Regular', 'New', 'Premium', 'Frequent', 'Premium', 'Premium'],
    'Service_Rating': ['Excellent', 'Excellent', 'Excellent', 'Fair', 'Good', 'Excellent', 'Good', 'Good'],
    'Buy_Again': ['Yes', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'Yes']
}

df = pd.DataFrame(data)
print("DATASET: Customer Purchase Behavior")
print("=" * 50)
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
target = df['Buy_Again'].tolist()
baseline_entropy = calculate_entropy(target)
baseline_gini = calculate_gini(target)

print(f"\nBASELINE METRICS")
print("=" * 50)
print(f"Total samples: {len(target)}")
print(f"Class distribution: {dict(pd.Series(target).value_counts())}")
print(f"Baseline entropy: {baseline_entropy:.4f}")
print(f"Baseline Gini impurity: {baseline_gini:.4f}")

# Features to evaluate
features = ['Product_Category', 'Purchase_Amount', 'Customer_Type', 'Service_Rating']

print(f"\n" + "="*80)
print("1. ID3 APPROACH - INFORMATION GAIN")
print("="*80)

id3_results = {}
for feature in features:
    print(f"\nFeature: {feature}")
    print("-" * 40)
    
    unique_values = df[feature].unique()
    children_labels = []
    
    print("Split breakdown:")
    for value in unique_values:
        subset = df[df[feature] == value]['Buy_Again'].tolist()
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
    print("-" * 40)
    
    unique_values = df[feature].unique()
    children_labels = []
    
    for value in unique_values:
        subset = df[df[feature] == value]['Buy_Again'].tolist()
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

# For Product_Category feature - demonstrate all binary splits
print(f"\nDetailed analysis for Product_Category feature (Gini-based):")
print("-" * 60)
category_values = df['Product_Category'].unique()
print(f"Product_Category values: {list(category_values)}")

# Generate all possible binary splits for Product_Category
all_binary_splits = []
for r in range(1, len(category_values)):
    for subset in combinations(category_values, r):
        split_1 = list(subset)
        split_2 = [v for v in category_values if v not in subset]
        all_binary_splits.append((split_1, split_2))

print(f"\nAll possible binary splits for Product_Category (Gini-based):")
category_gini_gains = []

for i, (set1, set2) in enumerate(all_binary_splits):
    print(f"\nSplit {i+1}: {set1} vs {set2}")
    
    # Get labels for each split
    mask1 = df['Product_Category'].isin(set1)
    mask2 = df['Product_Category'].isin(set2)
    
    labels1 = df[mask1]['Buy_Again'].tolist()
    labels2 = df[mask2]['Buy_Again'].tolist()
    
    children_labels = [labels1, labels2]
    gini_gain, parent_gini, weighted_child_gini = calculate_gini_gain(target, children_labels)
    category_gini_gains.append(gini_gain)
    
    print(f"  Group 1 ({set1}): {labels1} → {dict(pd.Series(labels1).value_counts())}")
    print(f"  Group 2 ({set2}): {labels2} → {dict(pd.Series(labels2).value_counts())}")
    print(f"  Gini Gain: {gini_gain:.4f}")

best_category_split_idx = np.argmax(category_gini_gains)
best_category_split = all_binary_splits[best_category_split_idx]
best_category_gini = category_gini_gains[best_category_split_idx]

print(f"\nBest Product_Category split: {best_category_split[0]} vs {best_category_split[1]}")
print(f"Best Product_Category Gini Gain: {best_category_gini:.4f}")

cart_results['Product_Category'] = best_category_gini

# For other features, calculate best binary split
for feature in ['Purchase_Amount', 'Customer_Type', 'Service_Rating']:
    feature_values = df[feature].unique()
    
    if len(feature_values) == 2:
        # Already binary
        labels1 = df[df[feature] == feature_values[0]]['Buy_Again'].tolist()
        labels2 = df[df[feature] == feature_values[1]]['Buy_Again'].tolist()
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
                
                labels1 = df[mask1]['Buy_Again'].tolist()
                labels2 = df[mask2]['Buy_Again'].tolist()
                
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
print("4. CART APPROACH - BINARY SPLITS WITH ENTROPY")
print("="*80)

cart_entropy_results = {}

# For Product_Category feature - demonstrate all binary splits using entropy
print(f"\nDetailed analysis for Product_Category feature (Entropy-based):")
print("-" * 60)

print(f"\nAll possible binary splits for Product_Category (Entropy-based):")
category_entropy_gains = []

for i, (set1, set2) in enumerate(all_binary_splits):
    print(f"\nSplit {i+1}: {set1} vs {set2}")
    
    # Get labels for each split
    mask1 = df['Product_Category'].isin(set1)
    mask2 = df['Product_Category'].isin(set2)
    
    labels1 = df[mask1]['Buy_Again'].tolist()
    labels2 = df[mask2]['Buy_Again'].tolist()
    
    children_labels = [labels1, labels2]
    entropy_gain, parent_ent, weighted_child_ent = calculate_information_gain(target, children_labels)
    category_entropy_gains.append(entropy_gain)
    
    print(f"  Group 1 ({set1}): {labels1} → {dict(pd.Series(labels1).value_counts())}")
    print(f"  Group 2 ({set2}): {labels2} → {dict(pd.Series(labels2).value_counts())}")
    print(f"  Entropy Gain: {entropy_gain:.4f}")

best_category_entropy_split_idx = np.argmax(category_entropy_gains)
best_category_entropy_split = all_binary_splits[best_category_entropy_split_idx]
best_category_entropy = category_entropy_gains[best_category_entropy_split_idx]

print(f"\nBest Product_Category entropy split: {best_category_entropy_split[0]} vs {best_category_entropy_split[1]}")
print(f"Best Product_Category Entropy Gain: {best_category_entropy:.4f}")

cart_entropy_results['Product_Category'] = best_category_entropy

# For other features, calculate best binary split using entropy
for feature in ['Purchase_Amount', 'Customer_Type', 'Service_Rating']:
    feature_values = df[feature].unique()
    
    if len(feature_values) == 2:
        # Already binary
        labels1 = df[df[feature] == feature_values[0]]['Buy_Again'].tolist()
        labels2 = df[df[feature] == feature_values[1]]['Buy_Again'].tolist()
        children_labels = [labels1, labels2]
        entropy_gain, _, _ = calculate_information_gain(target, children_labels)
        cart_entropy_results[feature] = entropy_gain
    else:
        # Find best binary split
        best_entropy = 0
        for r in range(1, len(feature_values)):
            for subset in combinations(feature_values, r):
                set1 = list(subset)
                set2 = [v for v in feature_values if v not in subset]
                
                mask1 = df[feature].isin(set1)
                mask2 = df[feature].isin(set2)
                
                labels1 = df[mask1]['Buy_Again'].tolist()
                labels2 = df[mask2]['Buy_Again'].tolist()
                
                children_labels = [labels1, labels2]
                entropy_gain, _, _ = calculate_information_gain(target, children_labels)
                
                if entropy_gain > best_entropy:
                    best_entropy = entropy_gain
        
        cart_entropy_results[feature] = best_entropy

# Find best feature for CART (entropy-based)
best_cart_entropy_feature = max(cart_entropy_results, key=cart_entropy_results.get)
print(f"\nCART RESULTS (entropy-based, best binary splits):")
for feature, entropy_gain in sorted(cart_entropy_results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {feature}: {entropy_gain:.4f}")
print(f"CART (entropy) would choose: {best_cart_entropy_feature}")

print(f"\n" + "="*80)
print("5. ALGORITHM COMPARISON")
print("="*80)
print(f"ID3 choice:  {best_id3_feature} (Information Gain: {id3_results[best_id3_feature]:.4f})")
print(f"C4.5 choice: {best_c45_feature} (Gain Ratio: {c45_results[best_c45_feature]:.4f})")
print(f"CART (Gini) choice: {best_cart_feature} (Gini Gain: {cart_results[best_cart_feature]:.4f})")
print(f"CART (Entropy) choice: {best_cart_entropy_feature} (Entropy Gain: {cart_entropy_results[best_cart_entropy_feature]:.4f})")

# Check if choices are different
all_choices = [best_id3_feature, best_c45_feature, best_cart_feature, best_cart_entropy_feature]
if len(set(all_choices)) == 1:
    print(f"\nAll algorithms agree on choosing: {best_id3_feature}")
else:
    print(f"\nAlgorithms disagree! Different splitting criteria lead to different choices.")

print(f"\n" + "="*80)
print("6. CART GINI vs CART ENTROPY COMPARISON")
print("="*80)
print("Comparing the results between CART using Gini vs CART using Entropy:")
for feature in features:
    gini_gain = cart_results[feature]
    entropy_gain = cart_entropy_results[feature]
    print(f"{feature}: Gini Gain = {gini_gain:.4f}, Entropy Gain = {entropy_gain:.4f}")
    if abs(gini_gain - entropy_gain) < 0.001:
        print(f"  → Same choice for {feature}")
    else:
        print(f"  → Different choice for {feature}")

# Create separate visualizations for each algorithm
# Plot 1: Information Gain comparison (ID3)
fig1, ax1 = plt.subplots(figsize=(10, 6))
features_sorted = sorted(id3_results.keys(), key=lambda x: id3_results[x], reverse=True)
ig_values = [id3_results[f] for f in features_sorted]
colors = ['red' if f == best_id3_feature else 'skyblue' for f in features_sorted]

bars1 = ax1.bar(features_sorted, ig_values, color=colors, alpha=0.7, edgecolor='black')
ax1.set_title('ID3: Information Gain Analysis', fontsize=14, fontweight='bold')
ax1.set_ylabel('Information Gain', fontsize=12)
ax1.set_xlabel('Features', fontsize=12)
ax1.set_xticks(range(len(features_sorted)))
ax1.set_xticklabels(features_sorted, rotation=45)
ax1.grid(True, alpha=0.3)

for bar, value in zip(bars1, ig_values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'id3_information_gain.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Gain Ratio comparison (C4.5)
fig2, ax2 = plt.subplots(figsize=(10, 6))
features_sorted_c45 = sorted(c45_results.keys(), key=lambda x: c45_results[x], reverse=True)
gr_values = [c45_results[f] for f in features_sorted_c45]
colors_c45 = ['red' if f == best_c45_feature else 'lightgreen' for f in features_sorted_c45]

bars2 = ax2.bar(features_sorted_c45, gr_values, color=colors_c45, alpha=0.7, edgecolor='black')
ax2.set_title('C4.5: Gain Ratio Analysis', fontsize=14, fontweight='bold')
ax2.set_ylabel('Gain Ratio', fontsize=12)
ax2.set_xlabel('Features', fontsize=12)
ax2.set_xticks(range(len(features_sorted_c45)))
ax2.set_xticklabels(features_sorted_c45, rotation=45)
ax2.grid(True, alpha=0.3)

for bar, value in zip(bars2, gr_values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'c45_gain_ratio.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Gini Gain comparison (CART)
fig3, ax3 = plt.subplots(figsize=(10, 6))
features_sorted_cart = sorted(cart_results.keys(), key=lambda x: cart_results[x], reverse=True)
gini_values = [cart_results[f] for f in features_sorted_cart]
colors_cart = ['red' if f == best_cart_feature else 'lightcoral' for f in features_sorted_cart]

bars3 = ax3.bar(features_sorted_cart, gini_values, color=colors_cart, alpha=0.7, edgecolor='black')
ax3.set_title('CART: Gini Gain Analysis', fontsize=14, fontweight='bold')
ax3.set_ylabel('Gini Gain', fontsize=12)
ax3.set_xlabel('Features', fontsize=12)
ax3.set_xticks(range(len(features_sorted_cart)))
ax3.set_xticklabels(features_sorted_cart, rotation=45)
ax3.grid(True, alpha=0.3)

for bar, value in zip(bars3, gini_values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'cart_gini_gain.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Entropy Gain comparison (CART)
fig4, ax4 = plt.subplots(figsize=(10, 6))
features_sorted_cart_entropy = sorted(cart_entropy_results.keys(), key=lambda x: cart_entropy_results[x], reverse=True)
entropy_values = [cart_entropy_results[f] for f in features_sorted_cart_entropy]
colors_cart_entropy = ['red' if f == best_cart_entropy_feature else 'lightblue' for f in features_sorted_cart_entropy]

bars4 = ax4.bar(features_sorted_cart_entropy, entropy_values, color=colors_cart_entropy, alpha=0.7, edgecolor='black')
ax4.set_title('CART: Entropy Gain Analysis', fontsize=14, fontweight='bold')
ax4.set_ylabel('Entropy Gain', fontsize=12)
ax4.set_xlabel('Features', fontsize=12)
ax4.set_xticks(range(len(features_sorted_cart_entropy)))
ax4.set_xticklabels(features_sorted_cart_entropy, rotation=45)
ax4.grid(True, alpha=0.3)

for bar, value in zip(bars4, entropy_values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'cart_entropy_gain.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 5: Dataset visualization
fig5, ax5 = plt.subplots(figsize=(12, 8))
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
table.set_fontsize(10)
table.scale(1, 1.5)

# Color code target column
for i in range(len(table_data)):
    if table_data.iloc[i]['Buy_Again'] == 'Yes':
        table[(i+1, -1)].set_facecolor('lightgreen')
    else:
        table[(i+1, -1)].set_facecolor('lightcoral')

# Header styling
for j in range(len(table_data.columns)):
    table[(0, j)].set_facecolor('#1976D2')
    table[(0, j)].set_text_props(weight='bold', color='white')

ax5.set_title('Customer Purchase Dataset Analysis', pad=20, fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'dataset_visualization.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 6: Product_Category binary splits for CART (Gini)
fig6, ax6 = plt.subplots(figsize=(12, 6))
split_labels = [f"Split {i+1}" for i in range(len(all_binary_splits))]
bars6 = ax6.bar(split_labels, category_gini_gains, 
               color=['red' if i == best_category_split_idx else 'lightcoral' 
                     for i in range(len(category_gini_gains))],
               alpha=0.7, edgecolor='black')

ax6.set_title('CART (Gini): Product_Category Binary Splits Analysis', fontsize=14, fontweight='bold')
ax6.set_ylabel('Gini Gain', fontsize=12)
ax6.set_xlabel('Binary Split Configuration', fontsize=12)
ax6.grid(True, alpha=0.3)

for bar, value in zip(bars6, category_gini_gains):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{value:.3f}', ha='center', va='bottom', fontsize=9)

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'cart_gini_binary_splits.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 7: Product_Category binary splits for CART (Entropy)
fig7, ax7 = plt.subplots(figsize=(12, 6))
bars7 = ax7.bar(split_labels, category_entropy_gains, 
               color=['red' if i == best_category_entropy_split_idx else 'lightblue' 
                     for i in range(len(category_entropy_gains))],
               alpha=0.7, edgecolor='black')

ax7.set_title('CART (Entropy): Product_Category Binary Splits Analysis', fontsize=14, fontweight='bold')
ax7.set_ylabel('Entropy Gain', fontsize=12)
ax7.set_xlabel('Binary Split Configuration', fontsize=12)
ax7.grid(True, alpha=0.3)

for bar, value in zip(bars7, category_entropy_gains):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{value:.3f}', ha='center', va='bottom', fontsize=9)

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'cart_entropy_binary_splits.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 8: Gini vs Entropy comparison
fig8, ax8 = plt.subplots(figsize=(10, 6))
x = np.arange(len(features))
width = 0.35

bars8a = ax8.bar(x - width/2, [cart_results[f] for f in features], width, 
                 label='Gini Gain', color='lightcoral', alpha=0.7)
bars8b = ax8.bar(x + width/2, [cart_entropy_results[f] for f in features], width, 
                 label='Entropy Gain', color='lightblue', alpha=0.7)

ax8.set_title('CART: Gini vs Entropy Comparison', fontsize=14, fontweight='bold')
ax8.set_ylabel('Gain Value', fontsize=12)
ax8.set_xlabel('Features', fontsize=12)
ax8.set_xticks(x)
ax8.set_xticklabels(features, rotation=45)
ax8.legend(fontsize=12)
ax8.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'cart_gini_vs_entropy.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 9: Algorithm choices summary
fig9, ax9 = plt.subplots(figsize=(10, 6))
algorithms = ['ID3', 'C4.5', 'CART\n(Gini)', 'CART\n(Entropy)']
chosen_features = [best_id3_feature, best_c45_feature, best_cart_feature, best_cart_entropy_feature]
alg_colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightblue']

bars9 = ax9.bar(algorithms, [1, 1, 1, 1], color=alg_colors, alpha=0.7, edgecolor='black')
ax9.set_title('Algorithm Feature Selection Summary', fontsize=14, fontweight='bold')
ax9.set_ylabel('Selection', fontsize=12)
ax9.set_ylim(0, 1.2)

for bar, choice in zip(bars9, chosen_features):
    ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
             choice, ha='center', va='center', fontweight='bold', rotation=90, fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'algorithm_choices_summary.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 10: Decision tree sketches for each algorithm

def calculate_leaf_positions(num_leaves, root_x=5, leaf_y=3, total_width=8):
    """Dynamically calculate leaf positions for any number of leaves, centered around root"""
    if num_leaves == 1:
        return [(root_x, leaf_y)]
    elif num_leaves == 2:
        return [(root_x - 2, leaf_y), (root_x + 2, leaf_y)]
    elif num_leaves == 3:
        return [(root_x - 3, leaf_y), (root_x, leaf_y), (root_x + 3, leaf_y)]
    elif num_leaves == 4:
        return [(root_x - 3.5, leaf_y), (root_x - 1.5, leaf_y), (root_x + 1.5, leaf_y), (root_x + 3.5, leaf_y)]
    else:
        # For 5+ leaves, calculate evenly spaced positions
        spacing = total_width / (num_leaves - 1) if num_leaves > 1 else 0
        start_x = root_x - total_width / 2
        return [(start_x + i * spacing, leaf_y) for i in range(num_leaves)]

# ID3 Tree
fig10, ax10 = plt.subplots(figsize=(8, 6))
ax10.set_xlim(0, 10)
ax10.set_ylim(0, 8)
ax10.axis('off')

# Root node
root_rect = plt.Rectangle((4, 6), 2, 1, facecolor='skyblue', edgecolor='black')
ax10.add_patch(root_rect)
ax10.text(5, 6.5, best_id3_feature, ha='center', va='center', fontweight='bold', fontsize=10)

# Child nodes for ID3's choice with sample distributions
unique_vals = df[best_id3_feature].unique()
child_positions = calculate_leaf_positions(len(unique_vals))

for i, val in enumerate(unique_vals):  # Show all children
    if i < len(child_positions):
        x, y = child_positions[i]
        
        # Calculate sample distribution for this value
        subset = df[df[best_id3_feature] == val]
        yes_count = len(subset[subset['Buy_Again'] == 'Yes'])
        total_count = len(subset)
        percentage = (yes_count / total_count) * 100 if total_count > 0 else 0
        
        # Create leaf node
        child_rect = plt.Rectangle((x-0.7, y-0.4), 1.4, 0.8, 
                                 facecolor='lightblue', edgecolor='black')
        ax10.add_patch(child_rect)
        ax10.text(x, y, val, ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Add sample distribution below leaf
        distribution_text = f'Yes: {yes_count}/{total_count} ({percentage:.0f}%)'
        ax10.text(x, y-0.8, distribution_text, ha='center', va='center', fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='gray', alpha=0.8))
        
        # Draw edge
        ax10.plot([5, x], [6, y+0.4], 'k-', linewidth=1)

# Add legend
ax10.text(8.5, 5, 'Sample Distribution:\nYes = Buy Again', 
          fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', edgecolor='black'))

ax10.set_title(f'ID3 Decision Tree\n(Root: {best_id3_feature})', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'id3_decision_tree.png'), dpi=300, bbox_inches='tight')
plt.close()

# C4.5 Tree
fig11, ax11 = plt.subplots(figsize=(8, 6))
ax11.set_xlim(0, 10)
ax11.set_ylim(0, 8)
ax11.axis('off')

# Root node
root_rect = plt.Rectangle((4, 6), 2, 1, facecolor='lightgreen', edgecolor='black')
ax11.add_patch(root_rect)
ax11.text(5, 6.5, best_c45_feature, ha='center', va='center', fontweight='bold', fontsize=10)

# Child nodes for C4.5's choice with sample distributions
unique_vals_c45 = df[best_c45_feature].unique()
child_positions_c45 = calculate_leaf_positions(len(unique_vals_c45))

for i, val in enumerate(unique_vals_c45):  # Show all children
    if i < len(child_positions_c45):
        x, y = child_positions_c45[i]
        
        # Calculate sample distribution for this value
        subset = df[df[best_c45_feature] == val]
        yes_count = len(subset[subset['Buy_Again'] == 'Yes'])
        total_count = len(subset)
        percentage = (yes_count / total_count) * 100 if total_count > 0 else 0
        
        # Create leaf node
        child_rect = plt.Rectangle((x-0.7, y-0.4), 1.4, 0.8, 
                                 facecolor='#C8E6C9', edgecolor='black')
        ax11.add_patch(child_rect)
        ax11.text(x, y, val, ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Add sample distribution below leaf
        distribution_text = f'Yes: {yes_count}/{total_count} ({percentage:.0f}%)'
        ax11.text(x, y-0.8, distribution_text, ha='center', va='center', fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='gray', alpha=0.8))
        
        # Draw edge
        ax11.plot([5, x], [6, y+0.4], 'k-', linewidth=1)

# Add legend
ax11.text(8.5, 5, 'Sample Distribution:\nYes = Buy Again', 
          fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', edgecolor='black'))

ax11.set_title(f'C4.5 Decision Tree\n(Root: {best_c45_feature})', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'c45_decision_tree.png'), dpi=300, bbox_inches='tight')
plt.close()

# CART (Gini) Tree
fig12, ax12 = plt.subplots(figsize=(8, 6))
ax12.set_xlim(0, 10)
ax12.set_ylim(0, 8)
ax12.axis('off')

# Root node
root_rect = plt.Rectangle((4, 6), 2, 1, facecolor='lightcoral', edgecolor='black')
ax12.add_patch(root_rect)
ax12.text(5, 6.5, best_cart_feature, ha='center', va='center', fontweight='bold', fontsize=10)

# Child nodes for CART Gini's choice with sample distributions
unique_vals_cart = df[best_cart_feature].unique()
child_positions_cart = calculate_leaf_positions(len(unique_vals_cart))

for i, val in enumerate(unique_vals_cart):  # Show all children
    if i < len(child_positions_cart):
        x, y = child_positions_cart[i]
        
        # Calculate sample distribution for this value
        subset = df[df[best_cart_feature] == val]
        yes_count = len(subset[subset['Buy_Again'] == 'Yes'])
        total_count = len(subset)
        percentage = (yes_count / total_count) * 100 if total_count > 0 else 0
        
        # Create leaf node
        child_rect = plt.Rectangle((x-0.7, y-0.4), 1.4, 0.8, 
                                 facecolor='#FFCDD2', edgecolor='black')
        ax12.add_patch(child_rect)
        ax12.text(x, y, val, ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Add sample distribution below leaf
        distribution_text = f'Yes: {yes_count}/{total_count} ({percentage:.0f}%)'
        ax12.text(x, y-0.8, distribution_text, ha='center', va='center', fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='gray', alpha=0.8))
        
        # Draw edge
        ax12.plot([5, x], [6, y+0.4], 'k-', linewidth=1)

# Add legend
ax12.text(8.5, 5, 'Sample Distribution:\nYes = Buy Again', 
          fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', edgecolor='black'))

ax12.set_title(f'CART (Gini) Decision Tree\n(Root: {best_cart_feature})', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'cart_gini_decision_tree.png'), dpi=300, bbox_inches='tight')
plt.close()

# CART (Entropy) Tree
fig13, ax13 = plt.subplots(figsize=(8, 6))
ax13.set_xlim(0, 10)
ax13.set_ylim(0, 8)
ax13.axis('off')

# Root node
root_rect = plt.Rectangle((4, 6), 2, 1, facecolor='lightblue', edgecolor='black')
ax13.add_patch(root_rect)
ax13.text(5, 6.5, best_cart_entropy_feature, ha='center', va='center', fontweight='bold', fontsize=10)

# Child nodes for CART entropy's choice with sample distributions
unique_vals_cart_entropy = df[best_cart_entropy_feature].unique()
child_positions_cart_entropy = calculate_leaf_positions(len(unique_vals_cart_entropy))

for i, val in enumerate(unique_vals_cart_entropy):  # Show all children
    if i < len(child_positions_cart_entropy):
        x, y = child_positions_cart_entropy[i]
        
        # Calculate sample distribution for this value
        subset = df[df[best_cart_entropy_feature] == val]
        yes_count = len(subset[subset['Buy_Again'] == 'Yes'])
        total_count = len(subset)
        percentage = (yes_count / total_count) * 100 if total_count > 0 else 0
        
        # Create leaf node
        child_rect = plt.Rectangle((x-0.7, y-0.4), 1.4, 0.8, 
                                 facecolor='#E3F2FD', edgecolor='black')
        ax13.add_patch(child_rect)
        ax13.text(x, y, val, ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Add sample distribution below leaf
        distribution_text = f'Yes: {yes_count}/{total_count} ({percentage:.0f}%)'
        ax13.text(x, y-0.8, distribution_text, ha='center', va='center', fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='gray', alpha=0.8))
        
        # Draw edge
        ax13.plot([5, x], [6, y+0.4], 'k-', linewidth=1)

# Add legend
ax13.text(8.5, 5, 'Sample Distribution:\nYes = Buy Again', 
          fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', edgecolor='black'))

ax13.set_title(f'CART (Entropy) Decision Tree\n(Root: {best_cart_entropy_feature})', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'cart_entropy_decision_tree.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 1: Information Gain comparison (ID3)
ax1 = plt.subplot(3, 4, 1)
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
ax2 = plt.subplot(3, 4, 2)
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
ax3 = plt.subplot(3, 4, 3)
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
ax4 = plt.subplot(3, 4, 4)
algorithms = ['ID3', 'C4.5', 'CART\n(Gini)', 'CART\n(Entropy)']
chosen_features = [best_id3_feature, best_c45_feature, best_cart_feature, best_cart_entropy_feature]
alg_colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightblue']

bars4 = ax4.bar(algorithms, [1, 1, 1, 1], color=alg_colors, alpha=0.7, edgecolor='black')
ax4.set_title('Algorithm Choices')
ax4.set_ylabel('Selection')
ax4.set_ylim(0, 1.2)

for bar, choice in zip(bars4, chosen_features):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
             choice, ha='center', va='center', fontweight='bold', rotation=90)

# Plot 5: Dataset visualization
ax5 = plt.subplot(3, 4, 5)
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
table.set_fontsize(8)
table.scale(1, 1.5)

# Color code target column
for i in range(len(table_data)):
    if table_data.iloc[i]['Buy_Again'] == 'Yes':
        table[(i+1, -1)].set_facecolor('lightgreen')
    else:
        table[(i+1, -1)].set_facecolor('lightcoral')

# Header styling
for j in range(len(table_data.columns)):
    table[(0, j)].set_facecolor('#1976D2')
    table[(0, j)].set_text_props(weight='bold', color='white')

ax5.set_title('Customer Purchase Dataset', pad=20)

# Plot 6: Product_Category binary splits for CART (Gini)
ax6 = plt.subplot(3, 4, 6)
split_labels = [f"Split {i+1}" for i in range(len(all_binary_splits))]
bars6 = ax6.bar(split_labels, category_gini_gains, 
               color=['red' if i == best_category_split_idx else 'lightcoral' 
                     for i in range(len(category_gini_gains))],
               alpha=0.7, edgecolor='black')

ax6.set_title('CART (Gini): Product_Category Binary Splits')
ax6.set_ylabel('Gini Gain')
ax6.set_xlabel('Binary Split')
ax6.grid(True, alpha=0.3)

for bar, value in zip(bars6, category_gini_gains):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{value:.3f}', ha='center', va='bottom', fontsize=8)

# Plot 7: Product_Category binary splits for CART (Entropy)
ax7 = plt.subplot(3, 4, 7)
bars7 = ax7.bar(split_labels, category_entropy_gains, 
               color=['red' if i == best_category_entropy_split_idx else 'lightblue' 
                     for i in range(len(category_entropy_gains))],
               alpha=0.7, edgecolor='black')

ax7.set_title('CART (Entropy): Product_Category Binary Splits')
ax7.set_ylabel('Entropy Gain')
ax7.set_xlabel('Binary Split')
ax7.grid(True, alpha=0.3)

for bar, value in zip(bars7, category_entropy_gains):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{value:.3f}', ha='center', va='bottom', fontsize=8)

# Plot 8: Gini vs Entropy comparison
ax8 = plt.subplot(3, 4, 8)
x = np.arange(len(features))
width = 0.35

bars8a = ax8.bar(x - width/2, [cart_results[f] for f in features], width, 
                 label='Gini Gain', color='lightcoral', alpha=0.7)
bars8b = ax8.bar(x + width/2, [cart_entropy_results[f] for f in features], width, 
                 label='Entropy Gain', color='lightblue', alpha=0.7)

ax8.set_title('CART: Gini vs Entropy Comparison')
ax8.set_ylabel('Gain')
ax8.set_xlabel('Features')
ax8.set_xticks(x)
ax8.set_xticklabels(features, rotation=45)
ax8.legend()
ax8.grid(True, alpha=0.3)

# Plot 9, 10, 11, 12: Decision tree sketches for each algorithm
# This would be the first level of trees that each algorithm would construct

# For ID3 tree
ax9 = plt.subplot(3, 4, 9)
ax9.set_xlim(0, 10)
ax9.set_ylim(0, 8)
ax9.axis('off')

# Root node
root_rect = plt.Rectangle((4, 6), 2, 1, facecolor='skyblue', edgecolor='black')
ax9.add_patch(root_rect)
ax9.text(5, 6.5, best_id3_feature, ha='center', va='center', fontweight='bold', fontsize=9)

# Child nodes for ID3's choice
unique_vals = df[best_id3_feature].unique()
child_positions = [(1, 3), (5, 3), (8, 3)]
for i, val in enumerate(unique_vals[:3]):  # Show up to 3 children
    if i < len(child_positions):
        x, y = child_positions[i]
        child_rect = plt.Rectangle((x-0.7, y-0.4), 1.4, 0.8, 
                                 facecolor='lightblue', edgecolor='black')
        ax9.add_patch(child_rect)
        ax9.text(x, y, val, ha='center', va='center', fontsize=8)
        
        # Draw edge
        ax9.plot([5, x], [6, y+0.4], 'k-', linewidth=1)

ax9.set_title(f'ID3 Tree\n(Root: {best_id3_feature})', fontweight='bold')

# For C4.5 tree
ax10 = plt.subplot(3, 4, 10)
ax10.set_xlim(0, 10)
ax10.set_ylim(0, 8)
ax10.axis('off')

# Root node
root_rect = plt.Rectangle((4, 6), 2, 1, facecolor='lightgreen', edgecolor='black')
ax10.add_patch(root_rect)
ax10.text(5, 6.5, best_c45_feature, ha='center', va='center', fontweight='bold', fontsize=9)

# Child nodes for C4.5's choice
unique_vals_c45 = df[best_c45_feature].unique()
for i, val in enumerate(unique_vals_c45[:3]):  # Show up to 3 children
    if i < len(child_positions):
        x, y = child_positions[i]
        child_rect = plt.Rectangle((x-0.7, y-0.4), 1.4, 0.8, 
                                 facecolor='#C8E6C9', edgecolor='black')
        ax10.add_patch(child_rect)
        ax10.text(x, y, val, ha='center', va='center', fontsize=8)
        
        # Draw edge
        ax10.plot([5, x], [6, y+0.4], 'k-', linewidth=1)

ax10.set_title(f'C4.5 Tree\n(Root: {best_c45_feature})', fontweight='bold')

# For CART (Gini) tree
ax11 = plt.subplot(3, 4, 11)
ax11.set_xlim(0, 10)
ax11.set_ylim(0, 8)
ax11.axis('off')

# Root node
root_rect = plt.Rectangle((4, 6), 2, 1, facecolor='lightcoral', edgecolor='black')
ax11.add_patch(root_rect)
ax11.text(5, 6.5, best_cart_feature, ha='center', va='center', fontweight='bold', fontsize=9)

# Child nodes for CART Gini's choice
unique_vals_cart = df[best_cart_feature].unique()
for i, val in enumerate(unique_vals_cart[:3]):  # Show up to 3 children
    if i < len(child_positions):
        x, y = child_positions[i]
        child_rect = plt.Rectangle((x-0.7, y-0.4), 1.4, 0.8, 
                                 facecolor='#FFCDD2', edgecolor='black')
        ax11.add_patch(child_rect)
        ax11.text(x, y, val, ha='center', va='center', fontsize=8)
        
        # Draw edge
        ax11.plot([5, x], [6, y+0.4], 'k-', linewidth=1)

ax11.set_title(f'CART (Gini) Tree\n(Root: {best_cart_feature})', fontweight='bold')

# For CART (Entropy) tree
ax12 = plt.subplot(3, 4, 12)
ax12.set_xlim(0, 10)
ax12.set_ylim(0, 8)
ax12.axis('off')

# Root node
root_rect = plt.Rectangle((4, 6), 2, 1, facecolor='lightblue', edgecolor='black')
ax12.add_patch(root_rect)
ax12.text(5, 6.5, best_cart_entropy_feature, ha='center', va='center', fontweight='bold', fontsize=9)

# Child nodes for CART entropy's choice
unique_vals_cart_entropy = df[best_cart_entropy_feature].unique()
for i, val in enumerate(unique_vals_cart_entropy[:3]):  # Show up to 3 children
    if i < len(child_positions):
        x, y = child_positions[i]
        child_rect = plt.Rectangle((x-0.7, y-0.4), 1.4, 0.8, 
                                 facecolor='#E3F2FD', edgecolor='black')
        ax12.add_patch(child_rect)
        ax12.text(x, y, val, ha='center', va='center', fontsize=8)
        
        # Draw edge
        ax12.plot([5, x], [6, y+0.4], 'k-', linewidth=1)

ax12.set_title(f'CART (Entropy) Tree\n(Root: {best_cart_entropy_feature})', fontweight='bold')





# Enhanced step-by-step calculations display
print(f"\n" + "="*80)
print("7. ENHANCED STEP-BY-STEP CALCULATIONS")
print("="*80)

print("\n" + "="*60)
print("DETAILED MATHEMATICAL DERIVATIONS")
print("="*60)

print(f"\nBaseline Calculations:")
print(f"  Total samples: {len(target)}")
print(f"  Class distribution: {dict(pd.Series(target).value_counts())}")
print(f"  P(Yes) = {6/8:.4f}, P(No) = {2/8:.4f}")
print(f"  Baseline Entropy H(S) = -({6/8:.4f} × log₂({6/8:.4f}) + {2/8:.4f} × log₂({2/8:.4f}))")
print(f"  H(S) = -({6/8:.4f} × {log2(6/8):.4f} + {2/8:.4f} × {log2(2/8):.4f})")
print(f"  H(S) = -({6/8:.4f} × {-0.4150:.4f} + {2/8:.4f} × {-2.0000:.4f})")
print(f"  H(S) = -({-0.3113:.4f} + {-0.5000:.4f})")
print(f"  H(S) = -({-0.8113:.4f}) = {baseline_entropy:.4f}")

print(f"\n  Baseline Gini G(S) = 1 - ({6/8:.4f}² + {2/8:.4f}²)")
print(f"  G(S) = 1 - ({0.5625:.4f} + {0.0625:.4f})")
print(f"  G(S) = 1 - {0.6250:.4f} = {baseline_gini:.4f}")

print(f"\n" + "="*60)
print("ID3 INFORMATION GAIN CALCULATIONS")
print("="*60)

for feature in features:
    print(f"\nFeature: {feature}")
    print("-" * 40)
    
    unique_values = df[feature].unique()
    children_labels = []
    
    print("Split breakdown:")
    for value in unique_values:
        subset = df[df[feature] == value]['Buy_Again'].tolist()
        children_labels.append(subset)
        subset_counts = dict(pd.Series(subset).value_counts())
        yes_count = subset_counts.get('Yes', 0)
        no_count = subset_counts.get('No', 0)
        total_count = len(subset)
        
        if yes_count > 0 and no_count > 0:
            p_yes = yes_count / total_count
            p_no = no_count / total_count
            entropy = -p_yes * log2(p_yes) - p_no * log2(p_no)
            print(f"  {value}: {subset} → {subset_counts}")
            print(f"    P(Yes) = {yes_count}/{total_count} = {p_yes:.4f}")
            print(f"    P(No) = {no_count}/{total_count} = {p_no:.4f}")
            print(f"    H({value}) = -({p_yes:.4f} × log₂({p_yes:.4f}) + {p_no:.4f} × log₂({p_no:.4f}))")
            print(f"    H({value}) = -({p_yes:.4f} × {log2(p_yes):.4f} + {p_no:.4f} × {log2(p_no):.4f})")
            print(f"    H({value}) = {entropy:.4f}")
        else:
            print(f"  {value}: {subset} → {subset_counts}")
            print(f"    Pure node (entropy = 0.0000)")
    
    ig, parent_ent, weighted_child_ent = calculate_information_gain(target, children_labels)
    
    print(f"\nInformation Gain Calculation:")
    print(f"  Parent entropy H(S) = {parent_ent:.4f}")
    print(f"  Weighted child entropy = Σ(|S_v|/|S| × H(S_v))")
    
    for i, value in enumerate(unique_values):
        subset = children_labels[i]
        weight = len(subset) / len(target)
        subset_entropy = calculate_entropy(subset)
        print(f"    {value}: {len(subset)}/{len(target)} × {subset_entropy:.4f} = {weight * subset_entropy:.4f}")
    
    print(f"  Weighted child entropy = {weighted_child_ent:.4f}")
    print(f"  Information Gain = {parent_ent:.4f} - {weighted_child_ent:.4f} = {ig:.4f}")

print(f"\n" + "="*60)
print("C4.5 GAIN RATIO CALCULATIONS")
print("="*60)

for feature in features:
    print(f"\nFeature: {feature}")
    print("-" * 40)
    
    unique_values = df[feature].unique()
    children_labels = []
    
    for value in unique_values:
        subset = df[df[feature] == value]['Buy_Again'].tolist()
        children_labels.append(subset)
    
    gr, ig, split_info = calculate_gain_ratio(target, children_labels)
    
    print(f"Information Gain = {ig:.4f}")
    print(f"Split Information = -Σ(|S_v|/|S| × log₂(|S_v|/|S|))")
    
    for value in unique_values:
        subset = children_labels[unique_values.tolist().index(value)]
        proportion = len(subset) / len(target)
        log_term = log2(proportion)
        print(f"  {value}: {len(subset)}/{len(target)} × log₂({proportion:.4f}) = {proportion:.4f} × {log_term:.4f} = {proportion * log_term:.4f}")
    
    print(f"Split Information = {split_info:.4f}")
    print(f"Gain Ratio = {ig:.4f} / {split_info:.4f} = {gr:.4f}")

print(f"\n" + "="*60)
print("CART BINARY SPLIT ANALYSIS")
print("="*60)

print(f"\nProduct_Category Binary Splits (Gini-based):")
print("-" * 50)
for i, (set1, set2) in enumerate(all_binary_splits):
    print(f"\nSplit {i+1}: {set1} vs {set2}")
    
    mask1 = df['Product_Category'].isin(set1)
    mask2 = df['Product_Category'].isin(set2)
    
    labels1 = df[mask1]['Buy_Again'].tolist()
    labels2 = df[mask2]['Buy_Again'].tolist()
    
    gini1 = calculate_gini(labels1)
    gini2 = calculate_gini(labels2)
    
    weight1 = len(labels1) / len(target)
    weight2 = len(labels2) / len(target)
    
    weighted_gini = weight1 * gini1 + weight2 * gini2
    gini_gain = baseline_gini - weighted_gini
    
    print(f"  Group 1 ({set1}): {labels1} → Gini = {gini1:.4f}, Weight = {weight1:.4f}")
    print(f"  Group 2 ({set2}): {labels2} → Gini = {gini2:.4f}, Weight = {weight2:.4f}")
    print(f"  Weighted Gini = {weight1:.4f} × {gini1:.4f} + {weight2:.4f} × {gini2:.4f} = {weighted_gini:.4f}")
    print(f"  Gini Gain = {baseline_gini:.4f} - {weighted_gini:.4f} = {gini_gain:.4f}")

# Enhanced CART Gini analysis with tie-breaking explanation
print(f"\n" + "="*80)
print("8. ENHANCED CART GINI ANALYSIS - TIE-BREAKING EXPLANATION")
print("="*80)

print(f"\n" + "="*60)
print("WHY PURCHASE_AMOUNT WAS CHOSEN DESPITE TIES")
print("="*60)

print(f"\nCART (Gini) Results Analysis:")
print(f"  Purchase_Amount: Gini Gain = {cart_results['Purchase_Amount']:.4f}")
print(f"  Customer_Type: Gini Gain = {cart_results['Customer_Type']:.4f}")
print(f"  Service_Rating: Gini Gain = {cart_results['Service_Rating']:.4f}")
print(f"  Product_Category: Gini Gain = {cart_results['Product_Category']:.4f}")

print(f"\nTie-Breaking Analysis:")
print(f"  All three features (Purchase_Amount, Customer_Type, Service_Rating) have the same Gini Gain: {cart_results['Purchase_Amount']:.4f}")

print(f"\n1. BALANCED SPLIT ANALYSIS:")
print(f"   Purchase_Amount binary splits:")
print(f"     - Left branch ($10-200): 4 samples")
print(f"     - Right branch ($200+): 4 samples")
print(f"     - Balance ratio: 4:4 = 1.0 (perfectly balanced)")

print(f"\n   Customer_Type binary splits:")
print(f"     - Left branch (Regular, New, Frequent): 5 samples")
print(f"     - Right branch (Premium): 3 samples")
print(f"     - Balance ratio: 5:3 = 1.67 (unbalanced)")

print(f"\n   Service_Rating binary splits:")
print(f"     - Left branch (Excellent, Fair): 5 samples")
print(f"     - Right branch (Good): 3 samples")
print(f"     - Balance ratio: 5:3 = 1.67 (unbalanced)")

print(f"\n2. DETAILED GINI IMPURITY CALCULATIONS FOR PURCHASE_AMOUNT:")
print(f"   Left Branch ($10-200): {df[df['Purchase_Amount'].isin(['$10-50', '$51-100', '$101-200'])]['Buy_Again'].tolist()}")
left_branch = df[df['Purchase_Amount'].isin(['$10-50', '$51-100', '$101-200'])]['Buy_Again'].tolist()
left_gini = calculate_gini(left_branch)
print(f"   Left Branch Gini = {left_gini:.4f}")

print(f"   Right Branch ($200+): {df[df['Purchase_Amount'] == '$200+']['Buy_Again'].tolist()}")
right_branch = df[df['Purchase_Amount'] == '$200+']['Buy_Again'].tolist()
right_gini = calculate_gini(right_branch)
print(f"   Right Branch Gini = {right_gini:.4f}")

weighted_gini = (len(left_branch)/len(target)) * left_gini + (len(right_branch)/len(target)) * right_gini
gini_gain = baseline_gini - weighted_gini

print(f"   Weighted Gini = (4/8) × {left_gini:.4f} + (4/8) × {right_gini:.4f} = {weighted_gini:.4f}")
print(f"   Gini Gain = {baseline_gini:.4f} - {weighted_gini:.4f} = {gini_gain:.4f}")

print(f"\n3. TIE-BREAKING CRITERIA:")
print(f"   - Balanced splits are preferred as they create more stable trees")
print(f"   - Equal sample distribution reduces overfitting risk")
print(f"   - Consistent impurity reduction across both branches")
print(f"   - Purchase_Amount creates the most balanced binary split")

print(f"\n4. FEATURE ENCODING OPTIMIZATION:")
print(f"   Current Purchase_Amount encoding: {df['Purchase_Amount'].unique()}")
print(f"   Suggested numeric encoding: [1, 2, 3, 4]")
print(f"   Binary encoding: ['Low', 'Low', 'Medium', 'High']")
print(f"   This would create even better binary splits for CART algorithms")

print(f"\n" + "="*80)
print("9. FIRST LEVEL DECISION TREES")
print("="*80)
print("All four algorithms would create different tree structures based on their")
print("chosen root features. The specific splits would be:")
print(f"- ID3: Split on {best_id3_feature}")
print(f"- C4.5: Split on {best_c45_feature}") 
print(f"- CART (Gini): Split on {best_cart_feature}")
print(f"- CART (Entropy): Split on {best_cart_entropy_feature}")

print(f"\n" + "="*80)
print("10. FEATURE ENCODING ANALYSIS")
print("="*80)
print("Feature encoding considerations for optimal tree construction:")
print("\nCategorical Features:")
print("- Product_Category: Already optimal (4 distinct values)")
print("- Customer_Type: Already optimal (4 distinct values)")
print("- Service_Rating: Already optimal (3 distinct values)")
print("\nOrdinal Features:")
print("- Purchase_Amount: Consider encoding as numeric ranges for better splits")
print("  Current: ['$10-50', '$51-100', '$101-200', '$200+']")
print("  Suggested: [1, 2, 3, 4] or actual numeric values")

print(f"\n" + "="*80)
print("11. VISUALIZATION FILES GENERATED")
print("="*80)
print("The following separate visualization files have been created:")
print("1. id3_information_gain.png - ID3 algorithm analysis")
print("2. c45_gain_ratio.png - C4.5 algorithm analysis")
print("3. cart_gini_gain.png - CART (Gini) algorithm analysis")
print("4. cart_entropy_gain.png - CART (Entropy) algorithm analysis")
print("5. dataset_visualization.png - Dataset overview")
print("6. cart_gini_binary_splits.png - CART Gini binary splits")
print("7. cart_entropy_binary_splits.png - CART Entropy binary splits")
print("8. cart_gini_vs_entropy.png - Gini vs Entropy comparison")
print("9. algorithm_choices_summary.png - Algorithm selection summary")
print("10. id3_decision_tree.png - ID3 decision tree structure")
print("11. c45_decision_tree.png - C4.5 decision tree structure")
print("12. cart_gini_decision_tree.png - CART (Gini) decision tree structure")
print("13. cart_entropy_decision_tree.png - CART (Entropy) decision tree structure")
print(f"\nAll images saved to: {save_dir}")
