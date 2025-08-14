import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.patches import FancyBboxPatch, Rectangle
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_18")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['text.latex.preamble'] = r'\usepackage{amssymb}'

print("=" * 80)
print("QUESTION 18: GAIN RATIO CALCULATION")
print("=" * 80)

# Define the loan approval dataset
data = {
    'Income': ['High', 'High', 'Medium', 'Low', 'Low', 'Medium'],
    'Age_Group': ['Young', 'Young', 'Middle', 'Old', 'Young', 'Old'],
    'Credit': ['Good', 'Poor', 'Good', 'Good', 'Poor', 'Good'],
    'Approved': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)
print("Dataset:")
print(df.to_string(index=False))
print()

# 1. Calculate entropy of the dataset
print("1. Calculate entropy of the dataset")
print("-" * 50)

# Count target classes
target_counts = df['Approved'].value_counts()
total_samples = len(df)
print(f"Target class distribution:")
print(f"  Yes: {target_counts.get('Yes', 0)} samples")
print(f"  No:  {target_counts.get('No', 0)} samples")
print(f"  Total: {total_samples} samples")

# Calculate probabilities
p_yes = target_counts.get('Yes', 0) / total_samples
p_no = target_counts.get('No', 0) / total_samples

print(f"\nProbabilities:")
print(f"  P(Yes) = {p_yes:.3f}")
print(f"  P(No)  = {p_no:.3f}")

# Calculate entropy using the formula: H(S) = -Σ p_i * log2(p_i)
def entropy(probabilities):
    """Calculate entropy given a list of probabilities"""
    entropy_val = 0
    for p in probabilities:
        if p > 0:  # Avoid log(0)
            entropy_val -= p * np.log2(p)
    return entropy_val

dataset_entropy = entropy([p_yes, p_no])
print(f"\nEntropy calculation:")
print(f"  $H(S) = -P(\\text{{Yes}}) \\times \\log_2(P(\\text{{Yes}})) - P(\\text{{No}}) \\times \\log_2(P(\\text{{No}}))$")
print(f"  $H(S) = -{p_yes:.3f} \\times \\log_2({p_yes:.3f}) - {p_no:.3f} \\times \\log_2({p_no:.3f})$")
print(f"  $H(S) = -{p_yes:.3f} \\times {np.log2(p_yes):.3f} - {p_no:.3f} \\times {np.log2(p_no):.3f}$")
print(f"  $H(S) = {p_yes * np.log2(p_yes):.3f} + {p_no * np.log2(p_no):.3f}$")
print(f"  $H(S) = {dataset_entropy:.3f}$")

# 2. Calculate information gain for Income feature
print("\n\n2. Calculate information gain for Income feature")
print("-" * 50)

# Get unique values and their distributions
income_values = df['Income'].unique()
print(f"Income feature has {len(income_values)} unique values: {income_values}")

# Calculate entropy for each income value
income_entropies = {}
income_counts = {}
weighted_entropy = 0

print(f"\nCalculating entropy for each income value:")
for income_val in income_values:
    # Filter data for this income value
    subset = df[df['Income'] == income_val]
    subset_count = len(subset)
    income_counts[income_val] = subset_count
    
    # Count target classes in this subset
    subset_target_counts = subset['Approved'].value_counts()
    subset_p_yes = subset_target_counts.get('Yes', 0) / subset_count
    subset_p_no = subset_target_counts.get('No', 0) / subset_count
    
    # Calculate entropy for this subset
    subset_entropy = entropy([subset_p_yes, subset_p_no])
    income_entropies[income_val] = subset_entropy
    
    print(f"\n  {income_val}:")
    print(f"    Count: {subset_count} samples")
    print(f"    Yes: {subset_target_counts.get('Yes', 0)}, No: {subset_target_counts.get('No', 0)}")
    print(f"    P(Yes) = {subset_p_yes:.3f}, P(No) = {subset_p_no:.3f}")
    print(f"    Entropy = {subset_entropy:.3f}")
    
    # Add to weighted entropy
    weighted_entropy += (subset_count / total_samples) * subset_entropy

print(f"\nWeighted entropy calculation:")
print(f"  $H(S|\\text{{Income}}) = \\sum(\\frac{{|S_v|}}{{|S|}} \\times H(S_v))$")
print(f"  H(S|Income) = ({income_counts.get('High', 0)}/{total_samples}) × {income_entropies.get('High', 0):.3f} + "
      f"({income_counts.get('Medium', 0)}/{total_samples}) × {income_entropies.get('Medium', 0):.3f} + "
      f"({income_counts.get('Low', 0)}/{total_samples}) × {income_entropies.get('Low', 0):.3f}")
print(f"  H(S|Income) = {income_counts.get('High', 0)/total_samples:.3f} × {income_entropies.get('High', 0):.3f} + "
      f"{income_counts.get('Medium', 0)/total_samples:.3f} × {income_entropies.get('Medium', 0):.3f} + "
      f"{income_counts.get('Low', 0)/total_samples:.3f} × {income_entropies.get('Low', 0):.3f}")
print(f"  H(S|Income) = {weighted_entropy:.3f}")

# Calculate information gain
information_gain = dataset_entropy - weighted_entropy
print(f"\nInformation Gain calculation:")
print(f"  $IG(S, \\text{{Income}}) = H(S) - H(S|\\text{{Income}})$")
print(f"  $IG(S, \\text{{Income}}) = {dataset_entropy:.3f} - {weighted_entropy:.3f}$")
print(f"  $IG(S, \\text{{Income}}) = {information_gain:.3f}$")

# 3. Calculate split information for Income feature
print("\n\n3. Calculate split information for Income feature")
print("-" * 50)

# Split information measures how balanced the split is
split_info = 0
for income_val in income_values:
    p_val = income_counts[income_val] / total_samples
    if p_val > 0:
        split_info -= p_val * np.log2(p_val)

print(f"Split Information calculation:")
print(f"  $\\text{{SplitInfo}}(S, \\text{{Income}}) = -\\sum(\\frac{{|S_v|}}{{|S|}} \\times \\log_2(\\frac{{|S_v|}}{{|S|}}))$")
for income_val in income_values:
    p_val = income_counts[income_val] / total_samples
    print(f"  -({p_val:.3f} × log2({p_val:.3f}))", end="")
print()
print(f"  SplitInfo(S, Income) = {split_info:.3f}")

# 4. Calculate gain ratio and compare with information gain
print("\n\n4. Calculate gain ratio and compare with information gain")
print("-" * 50)

gain_ratio = information_gain / split_info if split_info > 0 else 0
print(f"Gain Ratio calculation:")
print(f"  $\\text{{GainRatio}}(S, \\text{{Income}}) = IG(S, \\text{{Income}}) / \\text{{SplitInfo}}(S, \\text{{Income}})$")
print(f"  $\\text{{GainRatio}}(S, \\text{{Income}}) = {information_gain:.3f} / {split_info:.3f}$")
print(f"  $\\text{{GainRatio}}(S, \\text{{Income}}) = {gain_ratio:.3f}$")

print(f"\nComparison:")
print(f"  Information Gain: {information_gain:.3f}")
print(f"  Gain Ratio: {gain_ratio:.3f}")
print(f"  Difference: {information_gain - gain_ratio:.3f}")

# Create separate visualizations for each aspect

# Plot 1: Dataset Distribution
plt.figure(figsize=(8, 6))
target_labels = ['Yes', 'No']
target_values = [target_counts.get('Yes', 0), target_counts.get('No', 0)]
colors = ['lightgreen', 'lightcoral']

bars1 = plt.bar(target_labels, target_values, color=colors, alpha=0.7, edgecolor='black')
plt.ylabel('Number of Samples')
plt.grid(True, alpha=0.3)
plt.title('Dataset Target Distribution', fontsize=14, fontweight='bold')

# Add value labels on bars
for bar, value in zip(bars1, target_values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{value}', ha='center', va='bottom', fontweight='bold')
    plt.text(bar.get_x() + bar.get_width()/2., height/2,
             f'P = {value/total_samples:.3f}', ha='center', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'dataset_target_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Income Feature Distribution
plt.figure(figsize=(8, 6))
income_labels = list(income_counts.keys())
income_values_plot = list(income_counts.values())
income_colors = ['lightblue', 'lightyellow', 'lightpink']

bars2 = plt.bar(income_labels, income_values_plot, color=income_colors, alpha=0.7, edgecolor='black')
plt.ylabel('Number of Samples')
plt.grid(True, alpha=0.3)
plt.title('Income Feature Distribution', fontsize=14, fontweight='bold')

# Add value labels on bars
for bar, value in zip(bars2, income_values_plot):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{value}', ha='center', va='bottom', fontweight='bold')
    plt.text(bar.get_x() + bar.get_width()/2., height/2,
             f'P = {value/total_samples:.3f}', ha='center', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'income_feature_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Conditional Entropy by Income
plt.figure(figsize=(8, 6))
entropy_values = [income_entropies[val] for val in income_labels]
entropy_colors = ['red' if e > 0.5 else 'green' for e in entropy_values]

bars3 = plt.bar(income_labels, entropy_values, color=entropy_colors, alpha=0.7, edgecolor='black')
plt.ylabel('Entropy')
plt.ylim(0, 1.1)
plt.grid(True, alpha=0.3)
plt.title('Conditional Entropy by Income Value', fontsize=14, fontweight='bold')

# Add value labels on bars
for bar, value in zip(bars3, entropy_values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

# Add horizontal line for dataset entropy
plt.axhline(y=dataset_entropy, color='blue', linestyle='--', alpha=0.7, 
             label=f'Dataset Entropy = {dataset_entropy:.3f}')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'conditional_entropy_by_income.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Information Gain vs Gain Ratio
plt.figure(figsize=(8, 6))
metrics = ['Information\nGain', 'Gain\nRatio']
metric_values = [information_gain, gain_ratio]
metric_colors = ['orange', 'purple']

bars4 = plt.bar(metrics, metric_values, color=metric_colors, alpha=0.7, edgecolor='black')
plt.ylabel('Value')
plt.grid(True, alpha=0.3)
plt.title('Information Gain vs Gain Ratio', fontsize=14, fontweight='bold')

# Add value labels on bars
for bar, value in zip(bars4, metric_values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{value:.3f}', ha='center', va='bottom', fontsize=10, rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'information_gain_vs_gain_ratio.png'), dpi=300, bbox_inches='tight')
plt.close()



# Create detailed calculation table
print("\n" + "=" * 80)
print("DETAILED CALCULATION SUMMARY")
print("=" * 80)

calculation_summary = {
    'Step': [
        '1. Dataset Entropy',
        '2. Income = High Entropy',
        '3. Income = Medium Entropy', 
        '4. Income = Low Entropy',
        '5. Weighted Conditional Entropy',
        '6. Information Gain',
        '7. Split Information',
        '8. Gain Ratio'
    ],
    'Calculation': [
        f'H(S) = -{p_yes:.3f}×log2({p_yes:.3f}) - {p_no:.3f}×log2({p_no:.3f})',
        f'H(S|Income=High) = {income_entropies.get("High", 0):.3f}',
        f'H(S|Income=Medium) = {income_entropies.get("Medium", 0):.3f}',
        f'H(S|Income=Low) = {income_entropies.get("Low", 0):.3f}',
        f'({income_counts.get("High", 0)}/{total_samples})×{income_entropies.get("High", 0):.3f} + ...',
        f'{dataset_entropy:.3f} - {weighted_entropy:.3f}',
        f'-Σ(|S_v|/|S|)×log2(|S_v|/|S|)',
        f'{information_gain:.3f} / {split_info:.3f}'
    ],
    'Result': [
        f'{dataset_entropy:.3f}',
        f'{income_entropies.get("High", 0):.3f}',
        f'{income_entropies.get("Medium", 0):.3f}',
        f'{income_entropies.get("Low", 0):.3f}',
        f'{weighted_entropy:.3f}',
        f'{information_gain:.3f}',
        f'{split_info:.3f}',
        f'{gain_ratio:.3f}'
    ]
}

df_summary = pd.DataFrame(calculation_summary)
print(df_summary.to_string(index=False))

# Mathematical interpretation
print("\n" + "=" * 80)
print("MATHEMATICAL INTERPRETATION")
print("=" * 80)

print("1. Dataset Entropy H(S):")
print(f"   • Measures the impurity of the entire dataset")
print(f"   • Current value: {dataset_entropy:.3f} (closer to 1 = more impure)")
print(f"   • Interpretation: Moderate impurity, some class imbalance")

print("\n2. Information Gain IG(S, Income):")
print(f"   • Measures how much the Income feature reduces uncertainty")
print(f"   • Current value: {information_gain:.3f}")
print(f"   • Interpretation: {'Good' if information_gain > 0.3 else 'Moderate' if information_gain > 0.1 else 'Poor'} feature for splitting")

print("\n3. Split Information SplitInfo(S, Income):")
print(f"   • Measures how balanced the split is across feature values")
print(f"   • Current value: {split_info:.3f}")
print(f"   • Interpretation: {'Balanced' if split_info > 1.5 else 'Moderately balanced' if split_info > 1.0 else 'Unbalanced'} split")

print("\n4. Gain Ratio GainRatio(S, Income):")
print(f"   • Normalized information gain that penalizes multi-valued features")
print(f"   • Current value: {gain_ratio:.3f}")
print(f"   • Interpretation: {'Better' if gain_ratio > information_gain else 'Worse'} than information gain due to split information penalty")

print(f"\nPlots saved to: {save_dir}")
