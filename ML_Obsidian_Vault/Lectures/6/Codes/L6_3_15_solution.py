import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.patches import FancyBboxPatch
from itertools import combinations

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_15")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['text.latex.preamble'] = r'\usepackage{amssymb}'

print("=" * 80)
print("QUESTION 15: CART REGRESSION TREES")
print("=" * 80)

# Create the regression dataset
data = {
    'Feature1': ['Low', 'High', 'Low', 'High', 'Medium', 'Medium'],
    'Feature2': ['A', 'A', 'B', 'B', 'A', 'B'],
    'Target': [10.5, 15.2, 12.8, 18.1, 13.0, 16.5]
}

df = pd.DataFrame(data)
print("Regression Dataset:")
print(df.to_string(index=False))

# Function to calculate variance
def variance(values):
    """Calculate variance of a set of values"""
    if len(values) == 0:
        return 0
    
    mean_val = np.mean(values)
    var = np.sum((values - mean_val) ** 2) / len(values)
    return var

# Function to calculate variance reduction
def variance_reduction(y_total, y_left, y_right):
    """Calculate variance reduction for a split"""
    n_total = len(y_total)
    n_left = len(y_left)
    n_right = len(y_right)
    
    if n_left == 0 or n_right == 0:
        return 0
    
    var_total = variance(y_total)
    var_left = variance(y_left)
    var_right = variance(y_right)
    
    weighted_var = (n_left / n_total) * var_left + (n_right / n_total) * var_right
    
    return var_total - weighted_var

# 1. Calculate variance of the entire dataset
print("\n1. Variance of the Entire Dataset:")
print("-" * 35)

target_values = np.array(df['Target'])
total_variance = variance(target_values)
mean_target = np.mean(target_values)

print(f"Target values: {target_values}")
print(f"Mean: {mean_target:.3f}")

print(f"\nVariance calculation:")
print(f"Var(S) = (1/n) × Σ(yi - ȳ)²")
print(f"Where n = {len(target_values)}, ȳ = {mean_target:.3f}")

print(f"\nIndividual terms:")
variance_terms = []
for i, value in enumerate(target_values):
    term = (value - mean_target) ** 2
    variance_terms.append(term)
    print(f"  (y{i+1} - ȳ)² = ({value} - {mean_target:.3f})² = {term:.4f}")

total_sum = sum(variance_terms)
print(f"\nVar(S) = (1/{len(target_values)}) × ({' + '.join([f'{term:.4f}' for term in variance_terms])})")
print(f"Var(S) = (1/{len(target_values)}) × {total_sum:.4f}")
print(f"Var(S) = {total_variance:.4f}")

# 2. Calculate variance reduction for splitting on Feature1 (Low vs {Medium, High})
print("\n2. Variance Reduction for Feature1 Split (Low vs {Medium, High}):")
print("-" * 70)

# Create the split
left_mask = df['Feature1'] == 'Low'
right_mask = df['Feature1'] != 'Low'

left_subset = df[left_mask]
right_subset = df[right_mask]

y_left = left_subset['Target'].values
y_right = right_subset['Target'].values

print(f"Split: Feature1 = 'Low' vs Feature1 ∈ {{'Medium', 'High'}}")
print(f"\nLeft subset (Feature1 = 'Low'):")
print(left_subset.to_string(index=False))
print(f"Target values: {y_left}")
print(f"Mean: {np.mean(y_left):.3f}")

print(f"\nRight subset (Feature1 ∈ {{'Medium', 'High'}}):")
print(right_subset.to_string(index=False))
print(f"Target values: {y_right}")
print(f"Mean: {np.mean(y_right):.3f}")

# Calculate variances
var_left = variance(y_left)
var_right = variance(y_right)
var_reduction = variance_reduction(target_values, y_left, y_right)

print(f"\nVariance calculations:")
print(f"Var(S_left) = {var_left:.4f}")
print(f"Var(S_right) = {var_right:.4f}")

n_left = len(y_left)
n_right = len(y_right)
n_total = len(target_values)

print(f"\nWeighted variance after split:")
weighted_var = (n_left / n_total) * var_left + (n_right / n_total) * var_right
print(f"Weighted_Var = ({n_left}/{n_total}) × {var_left:.4f} + ({n_right}/{n_total}) × {var_right:.4f}")
print(f"Weighted_Var = {n_left/n_total:.3f} × {var_left:.4f} + {n_right/n_total:.3f} × {var_right:.4f}")
print(f"Weighted_Var = {weighted_var:.4f}")

print(f"\nVariance Reduction:")
print(f"Variance_Reduction = Var(S) - Weighted_Var")
print(f"Variance_Reduction = {total_variance:.4f} - {weighted_var:.4f}")
print(f"Variance_Reduction = {var_reduction:.4f}")

# 3. Predicted values for each leaf node
print("\n3. Predicted Values for Each Leaf Node:")
print("-" * 40)

print(f"Left leaf (Feature1 = 'Low'):")
print(f"  Predicted value = mean of target values = {np.mean(y_left):.3f}")
print(f"  This will be the prediction for any new sample with Feature1 = 'Low'")

print(f"\nRight leaf (Feature1 ∈ {{'Medium', 'High'}}):")
print(f"  Predicted value = mean of target values = {np.mean(y_right):.3f}")
print(f"  This will be the prediction for any new sample with Feature1 ∈ {{'Medium', 'High'}}")

# 4. How CART's regression criterion differs from classification
print("\n4. CART Regression vs Classification Criteria:")
print("-" * 45)

print("CART Regression (MSE/Variance criterion):")
print("• Minimizes variance (or Mean Squared Error) within nodes")
print("• Split quality measured by variance reduction")
print("• Leaf predictions are the mean of target values")
print("• Handles continuous target variables")
print("• Optimal for minimizing squared error loss")

print("\nCART Classification (Gini criterion):")
print("• Minimizes Gini impurity within nodes")
print("• Split quality measured by Gini reduction") 
print("• Leaf predictions are the majority class")
print("• Handles categorical target variables")
print("• Optimal for minimizing classification error")

print("\nKey Differences:")
print("• Target type: Continuous vs Categorical")
print("• Split criterion: Variance/MSE vs Gini impurity")
print("• Prediction method: Mean vs Majority vote")
print("• Objective: Minimize squared error vs Minimize misclassification")

# Calculate variance reduction for all possible splits
print("\n5. Variance Reduction for All Possible Splits:")
print("-" * 50)

# Feature1 splits
feature1_values = df['Feature1'].unique()
print(f"Feature1 possible splits:")

# All possible binary splits for Feature1
for value in feature1_values:
    left_mask = df['Feature1'] == value
    right_mask = df['Feature1'] != value
    
    if sum(left_mask) > 0 and sum(right_mask) > 0:  # Valid split
        y_left = df[left_mask]['Target'].values
        y_right = df[right_mask]['Target'].values
        var_red = variance_reduction(target_values, y_left, y_right)
        
        print(f"  {value} vs others: Variance Reduction = {var_red:.4f}")

# Feature2 splits
feature2_values = df['Feature2'].unique()
print(f"\nFeature2 possible splits:")

for value in feature2_values:
    left_mask = df['Feature2'] == value
    right_mask = df['Feature2'] != value
    
    if sum(left_mask) > 0 and sum(right_mask) > 0:  # Valid split
        y_left = df[left_mask]['Target'].values
        y_right = df[right_mask]['Target'].values
        var_red = variance_reduction(target_values, y_left, y_right)
        
        print(f"  {value} vs others: Variance Reduction = {var_red:.4f}")

# Create separate focused visualizations

# Visualization 1: Dataset overview
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('tight')
ax.axis('off')
table_data = df.values.tolist()
table = ax.table(cellText=table_data, colLabels=df.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1.5, 2.0)

# Color code by Feature1 value
colors = {'Low': '#FFB6C1', 'Medium': '#98FB98', 'High': '#87CEEB'}
for i in range(len(df)):
    feature1_val = df.iloc[i]['Feature1']
    table[(i+1, 0)].set_facecolor(colors[feature1_val])

# Header styling
for j in range(len(df.columns)):
    table[(0, j)].set_facecolor('#4472C4')
    table[(0, j)].set_text_props(weight='bold', color='white')

ax.set_title('Regression Dataset', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'regression_dataset_overview.png'), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 2: Target value distribution
fig, ax = plt.subplots(figsize=(10, 6))
feature1_groups = df.groupby('Feature1')['Target'].apply(list).to_dict()

x_pos = 0
colors_list = []
values_list = []
labels_list = []

for group, values in feature1_groups.items():
    for i, value in enumerate(values):
        values_list.append(value)
        colors_list.append(colors[group])
        labels_list.append(f'{group}_{i+1}')
        x_pos += 1

bars = ax.bar(range(len(values_list)), values_list, color=colors_list, alpha=0.7, edgecolor='black')
ax.set_xlabel('Samples')
ax.set_ylabel('Target Value')
ax.set_title('Target Values by Feature1 Groups')
ax.set_xticks(range(len(values_list)))
ax.set_xticklabels([f'S{i+1}' for i in range(len(values_list))])

# Add value labels on bars
for bar, value in zip(bars, values_list):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
             f'{value}', ha='center', va='bottom', fontsize=10)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color, label=group) for group, color in colors.items()]
ax.legend(handles=legend_elements, title='Feature1')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'target_values_by_feature1.png'), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 3: Variance analysis
fig, ax = plt.subplots(figsize=(10, 6))
variance_data = {
    'Total': total_variance,
    'Left (Low)': var_left,
    'Right (Med+High)': var_right
}

bars3 = ax.bar(variance_data.keys(), variance_data.values(), 
                color=['gray', 'lightblue', 'lightcoral'], alpha=0.7, edgecolor='black')
ax.set_ylabel('Variance')
ax.set_title('Variance Analysis for Feature1 Split')
ax.grid(True, alpha=0.3)

# Add value labels
for bar, (label, value) in zip(bars3, variance_data.items()):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{value:.3f}', ha='center', va='bottom', fontsize=10, weight='bold')

# Add variance reduction annotation
ax.text(1, max(variance_data.values()) * 0.8, 
         f'Variance Reduction\n= {var_reduction:.4f}', 
         ha='center', va='center', fontsize=12, weight='bold',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'variance_analysis_split.png'), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 4: Regression tree structure
fig, ax = plt.subplots(figsize=(10, 8))
ax.text(0.5, 0.95, 'CART Regression Tree', ha='center', fontsize=14, weight='bold')

# Root node
root_box = FancyBboxPatch((0.35, 0.75), 0.3, 0.1,
                         boxstyle="round,pad=0.01",
                         facecolor='lightblue', edgecolor='black', linewidth=2)
ax.add_patch(root_box)
ax.text(0.5, 0.8, f'Feature1 = Low?\nVariance = {total_variance:.3f}', 
         ha='center', va='center', fontsize=10, weight='bold')

# Left child (Feature1 = Low)
left_box = FancyBboxPatch((0.1, 0.45), 0.25, 0.15,
                         boxstyle="round,pad=0.01",
                         facecolor='lightgreen', edgecolor='black')
ax.add_patch(left_box)
left_text = f"Feature1 = Low\nSamples: {len(y_left)}\nValues: {y_left}\nMean: {np.mean(y_left):.1f}\nVariance: {var_left:.3f}"
ax.text(0.225, 0.525, left_text, ha='center', va='center', fontsize=9)

# Right child (Feature1 ≠ Low)
right_box = FancyBboxPatch((0.65, 0.45), 0.25, 0.15,
                          boxstyle="round,pad=0.01",
                          facecolor='lightcoral', edgecolor='black')
ax.add_patch(right_box)
right_text = f"Feature1 $\\neq$ Low\nSamples: {len(y_right)}\nValues: {y_right}\nMean: {np.mean(y_right):.1f}\nVariance: {var_right:.3f}"
ax.text(0.775, 0.525, right_text, ha='center', va='center', fontsize=9)

# Draw edges
ax.plot([0.45, 0.275], [0.75, 0.6], 'k-', linewidth=2)
ax.plot([0.55, 0.725], [0.75, 0.6], 'k-', linewidth=2)

# Edge labels
ax.text(0.35, 0.68, 'Yes', ha='center', fontsize=10, weight='bold', color='green')
ax.text(0.65, 0.68, 'No', ha='center', fontsize=10, weight='bold', color='red')

# Variance reduction annotation
ax.text(0.5, 0.3, f'Variance Reduction = {var_reduction:.4f}', 
         ha='center', va='center', fontsize=12, weight='bold',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'cart_regression_tree.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create detailed variance calculation visualization
fig, ax = plt.subplots(figsize=(14, 10))

ax.text(0.5, 0.95, 'Detailed Variance Calculations', ha='center', fontsize=16, weight='bold')

# Total variance calculation
y_pos = 0.85
ax.text(0.05, y_pos, '1. Total Dataset Variance:', fontsize=14, weight='bold')
y_pos -= 0.05
ax.text(0.1, y_pos, f'Target values: {target_values}', fontsize=12)
y_pos -= 0.04
ax.text(0.1, y_pos, f'Mean (ȳ): {mean_target:.3f}', fontsize=12)
y_pos -= 0.04

for i, (value, term) in enumerate(zip(target_values, variance_terms)):
    ax.text(0.1, y_pos, f'(y{i+1} - ȳ)² = ({value} - {mean_target:.3f})² = {term:.4f}', fontsize=11)
    y_pos -= 0.03

ax.text(0.1, y_pos, f'Var(S) = (1/{len(target_values)}) × {sum(variance_terms):.4f} = {total_variance:.4f}', 
        fontsize=12, weight='bold', color='blue')

# Left subset variance
y_pos -= 0.08
ax.text(0.05, y_pos, '2. Left Subset Variance (Feature1 = Low):', fontsize=14, weight='bold')
y_pos -= 0.05
ax.text(0.1, y_pos, f'Values: {y_left}, Mean: {np.mean(y_left):.3f}', fontsize=12)
y_pos -= 0.04

left_mean = np.mean(y_left)
for i, value in enumerate(y_left):
    term = (value - left_mean) ** 2
    ax.text(0.1, y_pos, f'({value} - {left_mean:.3f})² = {term:.4f}', fontsize=11)
    y_pos -= 0.03

ax.text(0.1, y_pos, f'Var(S_left) = {var_left:.4f}', fontsize=12, weight='bold', color='blue')

# Right subset variance
y_pos -= 0.06
ax.text(0.05, y_pos, '3. Right Subset Variance (Feature1 $\\neq$ Low):', fontsize=14, weight='bold')
y_pos -= 0.05
ax.text(0.1, y_pos, f'Values: {y_right}, Mean: {np.mean(y_right):.3f}', fontsize=12)
y_pos -= 0.04

right_mean = np.mean(y_right)
for i, value in enumerate(y_right):
    term = (value - right_mean) ** 2
    ax.text(0.1, y_pos, f'({value} - {right_mean:.3f})² = {term:.4f}', fontsize=11)
    y_pos -= 0.03

ax.text(0.1, y_pos, f'Var(S_right) = {var_right:.4f}', fontsize=12, weight='bold', color='blue')

# Final calculation
y_pos -= 0.06
ax.text(0.05, y_pos, '4. Variance Reduction:', fontsize=14, weight='bold')
y_pos -= 0.05
ax.text(0.1, y_pos, f'Weighted_Var = ({n_left}/{n_total}) × {var_left:.4f} + ({n_right}/{n_total}) × {var_right:.4f} = {weighted_var:.4f}', 
        fontsize=12)
y_pos -= 0.04
ax.text(0.1, y_pos, f'Variance_Reduction = {total_variance:.4f} - {weighted_var:.4f} = {var_reduction:.4f}', 
        fontsize=12, weight='bold', color='red')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.savefig(os.path.join(save_dir, 'detailed_variance_calculations.png'), dpi=300, bbox_inches='tight')

# Create comparison between regression and classification criteria
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Regression criteria
regression_aspects = ['Target Type', 'Split Criterion', 'Leaf Prediction', 'Optimization Goal', 
                     'Loss Function', 'Node Purity Measure']
regression_values = ['Continuous', 'Variance/MSE', 'Mean of targets', 'Minimize squared error',
                    'Squared loss', 'Variance']

y_pos = np.arange(len(regression_aspects))
bars1 = ax1.barh(y_pos, [1]*len(regression_aspects), color='lightblue', alpha=0.7)

for i, (aspect, value) in enumerate(zip(regression_aspects, regression_values)):
    ax1.text(0.1, i, f"{aspect}:\n{value}", va='center', fontsize=10, weight='bold')

ax1.set_yticks(y_pos)
ax1.set_yticklabels(regression_aspects)
ax1.set_title('CART Regression Criteria', fontsize=14, weight='bold')
ax1.set_xlim(0, 1)
ax1.set_xticks([])

# Classification criteria
classification_values = ['Categorical', 'Gini/Entropy', 'Majority class', 'Minimize misclassification',
                        'Zero-one loss', 'Gini impurity']

bars2 = ax2.barh(y_pos, [1]*len(regression_aspects), color='lightcoral', alpha=0.7)

for i, (aspect, value) in enumerate(zip(regression_aspects, classification_values)):
    ax2.text(0.1, i, f"{aspect}:\n{value}", va='center', fontsize=10, weight='bold')

ax2.set_yticks(y_pos)
ax2.set_yticklabels(regression_aspects)
ax2.set_title('CART Classification Criteria', fontsize=14, weight='bold')
ax2.set_xlim(0, 1)
ax2.set_xticks([])

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'regression_vs_classification.png'), dpi=300, bbox_inches='tight')

print(f"\nVisualization files saved to: {save_dir}")
print("Files created:")
print("- regression_dataset_overview.png")
print("- target_values_by_feature1.png")
print("- variance_analysis_split.png")
print("- cart_regression_tree.png")
print("- detailed_variance_calculations.png")
print("- regression_vs_classification.png")
