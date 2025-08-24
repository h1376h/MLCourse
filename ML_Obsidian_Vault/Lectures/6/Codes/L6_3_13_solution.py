import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.patches import Rectangle, FancyBboxPatch
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_13")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['text.latex.preamble'] = r'\usepackage{amssymb}'

print("=" * 80)
print("QUESTION 13: MISSING VALUE STRATEGIES")
print("=" * 80)

# Create sample dataset with missing values
np.random.seed(42)
n_samples = 100
dataset = {
    'Feature_A': np.random.choice(['Low', 'Medium', 'High', np.nan], n_samples, p=[0.3, 0.3, 0.1, 0.3]),
    'Feature_B': np.random.normal(50, 15, n_samples),
    'Feature_C': np.random.choice(['Red', 'Blue', 'Green', np.nan], n_samples, p=[0.25, 0.25, 0.2, 0.3]),
    'Target': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4])
}

# Introduce some missing values in Feature_B
missing_indices = np.random.choice(n_samples, int(0.3 * n_samples), replace=False)
for idx in missing_indices:
    dataset['Feature_B'][idx] = np.nan

df = pd.DataFrame(dataset)

print("Sample Dataset with Missing Values:")
print(f"Total samples: {len(df)}")
print(f"Missing values per feature:")
for col in df.columns:
    missing_count = df[col].isna().sum()
    missing_pct = (missing_count / len(df)) * 100
    print(f"  {col}: {missing_count} ({missing_pct:.1f}%)")

# 1. How does ID3 typically handle missing values in practice?
print("\n1. How does ID3 typically handle missing values in practice?")
print("-" * 60)
print("ID3 typically handles missing values through preprocessing approaches:")
print("• Remove samples with missing values (complete case analysis)")
print("• Remove features with high missing rates") 
print("• Impute missing values using mode (categorical) or mean (numerical)")
print("• Use 'Unknown' as a separate category for categorical features")
print("• ID3 itself doesn't have built-in missing value handling")

# 2. Describe C4.5's "fractional instance" method
print("\n2. C4.5's 'Fractional Instance' Method:")
print("-" * 50)
print("C4.5 uses probabilistic fractional instances to handle missing values:")
print("• When a feature value is missing, the instance is split proportionally")
print("• Each branch receives a fraction of the instance based on training data distribution")
print("• Fractions sum to 1.0 across all branches for that feature")
print("• Information gain calculations are adjusted for fractional instances")
print("• This allows using all available information without discarding samples")

# 3. CART's surrogate splits
print("\n3. CART's Surrogate Splits:")
print("-" * 30)
print("CART uses surrogate splits as backup decision rules:")
print("• Primary split uses the best feature (highest information gain/Gini reduction)")
print("• Surrogate splits use other features that correlate with the primary split")
print("• When primary feature is missing, use the best surrogate split")
print("• Multiple surrogates can be ranked by their agreement with primary split")
print("• Provides robustness and maintains tree structure even with missing values")

# Create detailed visualization of missing value strategies

# Create separate focused visualizations for missing value strategies

# Visualization 1: ID3 missing value strategies
fig, ax = plt.subplots(figsize=(10, 8))
id3_strategies = ['Remove Samples', 'Remove Features', 'Mode/Mean Imputation', 'Unknown Category']
id3_pros = ['Simple', 'Simple', 'Preserves all data', 'Explicit handling']
id3_cons = ['Data loss', 'Feature loss', 'May introduce bias', 'May create noise']

y_pos = np.arange(len(id3_strategies))
bars1 = ax.barh(y_pos, [1]*len(id3_strategies), color=['red', 'orange', 'yellow', 'lightblue'], alpha=0.7)

for i, (strategy, pro, con) in enumerate(zip(id3_strategies, id3_pros, id3_cons)):
    ax.text(0.1, i, f"{strategy}\nPro: {pro}\nCon: {con}", 
             va='center', fontsize=9, weight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(id3_strategies)
ax.set_title('ID3: Missing Value Strategies', fontsize=12, weight='bold')
ax.set_xlim(0, 1)
ax.set_xticks([])
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'id3_missing_value_strategies.png'), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 2: C4.5 fractional instance method
fig, ax = plt.subplots(figsize=(10, 8))
ax.text(0.5, 0.8, 'C4.5 Fractional Instance Method', ha='center', fontsize=12, weight='bold')
ax.text(0.1, 0.6, 'Original Sample: A=?, B=High, Class=Yes', fontsize=10)
ax.text(0.1, 0.5, 'Split on Feature A:', fontsize=10, weight='bold')
ax.text(0.1, 0.4, '• Branch A=Low: 0.4 × Sample', fontsize=9, color='blue')
ax.text(0.1, 0.35, '• Branch A=Medium: 0.3 × Sample', fontsize=9, color='green')
ax.text(0.1, 0.3, '• Branch A=High: 0.3 × Sample', fontsize=9, color='red')
ax.text(0.1, 0.2, 'Fractions based on training data distribution', fontsize=9, style='italic')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'c45_fractional_instance_method.png'), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 3: CART surrogate splits
fig, ax = plt.subplots(figsize=(10, 8))
ax.text(0.5, 0.9, 'CART Surrogate Splits', ha='center', fontsize=12, weight='bold')

# Draw tree structure
root_box = FancyBboxPatch((0.35, 0.7), 0.3, 0.1, 
                         boxstyle="round,pad=0.01", 
                         facecolor='lightblue', edgecolor='black')
ax.add_patch(root_box)
ax.text(0.5, 0.75, 'Primary: A $\\leq$ 5?', ha='center', va='center', fontsize=10, weight='bold')

# Surrogate splits
ax.text(0.1, 0.5, 'Surrogate 1:', fontsize=9, weight='bold')
ax.text(0.1, 0.45, 'B $\\leq$ 10? (Agreement: 85%)', fontsize=8)
ax.text(0.1, 0.35, 'Surrogate 2:', fontsize=9, weight='bold')  
ax.text(0.1, 0.3, 'C = Red? (Agreement: 78%)', fontsize=8)

ax.text(0.1, 0.15, 'When A is missing:', fontsize=9, weight='bold', color='red')
ax.text(0.1, 0.1, 'Use best available surrogate', fontsize=8, style='italic')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'cart_surrogate_splits.png'), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 4: Algorithm robustness comparison
fig, ax = plt.subplots(figsize=(10, 6))
algorithms = ['ID3', 'C4.5', 'CART']
robustness_scores = [2, 8, 9]  # Subjective robustness scores out of 10
colors = ['red', 'orange', 'green']

bars4 = ax.bar(algorithms, robustness_scores, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Robustness Score (1-10)')
ax.set_title('Missing Value Robustness Comparison')
ax.set_ylim(0, 10)
ax.grid(True, alpha=0.3)

# Add score labels on bars
for bar, score in zip(bars4, robustness_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{score}/10', ha='center', va='bottom', fontsize=10, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'missing_value_robustness_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 2: Detailed fractional instance example
fig, ax = plt.subplots(figsize=(14, 10))

# Create example scenario
ax.text(0.5, 0.95, 'C4.5 Fractional Instance Method: Detailed Example', 
        ha='center', fontsize=16, weight='bold')

# Original sample
orig_box = FancyBboxPatch((0.1, 0.8), 0.8, 0.08,
                         boxstyle="round,pad=0.01",
                         facecolor='lightgray', edgecolor='black')
ax.add_patch(orig_box)
ax.text(0.5, 0.84, 'Original Sample: Feature_A=?, Feature_B=High, Target=Yes (Weight=1.0)', 
        ha='center', va='center', fontsize=12, weight='bold')

# Show distribution from training data
ax.text(0.1, 0.7, 'Training Data Distribution for Feature_A:', fontsize=12, weight='bold')
ax.text(0.1, 0.65, '• Low: 40% of samples', fontsize=10)
ax.text(0.1, 0.6, '• Medium: 35% of samples', fontsize=10)
ax.text(0.1, 0.55, '• High: 25% of samples', fontsize=10)

# Show fractional splits
ax.text(0.1, 0.45, 'Fractional Instance Creation:', fontsize=12, weight='bold')

# Three fractional instances
fractions = [0.4, 0.35, 0.25]
labels = ['Low', 'Medium', 'High']
colors = ['lightblue', 'lightgreen', 'lightpink']
y_positions = [0.35, 0.25, 0.15]

for i, (frac, label, color, y_pos) in enumerate(zip(fractions, labels, colors, y_positions)):
    box = FancyBboxPatch((0.1, y_pos), 0.8, 0.06,
                        boxstyle="round,pad=0.01",
                        facecolor=color, edgecolor='black')
    ax.add_patch(box)
    ax.text(0.5, y_pos + 0.03, 
            f'Branch A={label}: Weight={frac}, Feature_B=High, Target=Yes', 
            ha='center', va='center', fontsize=11)

# Show information gain calculation
ax.text(0.1, 0.05, 'Information gain calculated using weighted instances', 
        fontsize=11, style='italic', color='blue')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.savefig(os.path.join(save_dir, 'fractional_instance_example.png'), dpi=300, bbox_inches='tight')

# Visualization 3: Surrogate splits detailed example
fig, ax = plt.subplots(figsize=(14, 10))

ax.text(0.5, 0.95, 'CART Surrogate Splits: Detailed Example', 
        ha='center', fontsize=16, weight='bold')

# Primary split
primary_box = FancyBboxPatch((0.3, 0.8), 0.4, 0.08,
                            boxstyle="round,pad=0.01",
                            facecolor='lightblue', edgecolor='black')
ax.add_patch(primary_box)
ax.text(0.5, 0.84, 'Primary Split: Age $\\leq$ 30?', ha='center', va='center', fontsize=14, weight='bold')

# Draw branches
ax.plot([0.4, 0.2], [0.8, 0.65], 'k-', linewidth=2)
ax.plot([0.6, 0.8], [0.8, 0.65], 'k-', linewidth=2)

# Left and right nodes
left_box = FancyBboxPatch((0.05, 0.6), 0.3, 0.1,
                         boxstyle="round,pad=0.01",
                         facecolor='lightgreen', edgecolor='black')
ax.add_patch(left_box)
ax.text(0.2, 0.65, 'Age $<$ 30\n(Young)', ha='center', va='center', fontsize=11, weight='bold')

right_box = FancyBboxPatch((0.65, 0.6), 0.3, 0.1,
                          boxstyle="round,pad=0.01",
                          facecolor='lightcoral', edgecolor='black')
ax.add_patch(right_box)
ax.text(0.8, 0.65, 'Age $\\geq$ 30\n(Older)', ha='center', va='center', fontsize=11, weight='bold')

# Surrogate splits information
ax.text(0.1, 0.45, 'Surrogate Splits (ranked by agreement with primary):', 
        fontsize=12, weight='bold')

surrogate_info = [
    ('Income $\\leq$ 50K?', '88%', 'Best surrogate'),
    ('Education = Bachelors?', '76%', 'Second best'),
    ('Married = Yes?', '65%', 'Third option')
]

for i, (split, agreement, note) in enumerate(surrogate_info):
    y_pos = 0.35 - i*0.08
    ax.text(0.1, y_pos, f'{i+1}. {split} (Agreement: {agreement}) - {note}', 
            fontsize=10, color='blue' if i == 0 else 'black')

# Missing value handling
ax.text(0.1, 0.1, 'When Age is missing:', fontsize=12, weight='bold', color='red')
ax.text(0.1, 0.05, '→ Use Income $\\leq$ 50K? (best available surrogate)', 
        fontsize=11, style='italic')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.savefig(os.path.join(save_dir, 'surrogate_splits_example.png'), dpi=300, bbox_inches='tight')

# 4. Which algorithm would be most robust for 30% missing values?
print("\n4. Algorithm Robustness for 30% Missing Values in Feature A:")
print("-" * 65)

# Create comprehensive comparison
robustness_analysis = {
    'Algorithm': ['ID3', 'C4.5', 'CART'],
    'Native Support': ['No', 'Yes', 'Yes'],
    'Data Loss': ['High (30%)', 'None', 'None'],
    'Information Preservation': ['Low', 'High', 'High'],
    'Computational Overhead': ['Low', 'Medium', 'Medium'],
    'Accuracy Impact': ['High', 'Low', 'Low'],
    'Robustness Score': [3, 8, 9]
}

df_robustness = pd.DataFrame(robustness_analysis)
print(df_robustness.to_string(index=False))

print(f"\nRecommendation: CART would be most robust because:")
print("• Native surrogate split mechanism handles missing values elegantly")
print("• No data loss - all samples can be used")
print("• Maintains prediction accuracy even with missing values")
print("• Surrogate splits provide multiple backup decision paths")
print("• Well-tested approach in practice")

# Create final comparison visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Robustness comparison chart
algorithms = robustness_analysis['Algorithm']
scores = robustness_analysis['Robustness Score']
colors = ['red', 'orange', 'green']

bars = ax1.bar(algorithms, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel('Robustness Score (1-10)', fontsize=12)
ax1.set_title('Missing Value Robustness: 30% Missing Rate', fontsize=14, weight='bold')
ax1.set_ylim(0, 10)
ax1.grid(True, alpha=0.3)

# Add score labels and explanations
for bar, score, alg in zip(bars, scores, algorithms):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.2,
             f'{score}/10', ha='center', va='bottom', fontsize=12, weight='bold')
    
    # Add explanation below
    if alg == 'ID3':
        explanation = 'Requires\npreprocessing'
    elif alg == 'C4.5':
        explanation = 'Fractional\ninstances'
    else:
        explanation = 'Surrogate\nsplits'
    
    ax1.text(bar.get_x() + bar.get_width()/2., -0.8,
             explanation, ha='center', va='top', fontsize=10, style='italic')

# Data utilization comparison
utilization_data = {
    'ID3 (Remove)': 70,  # Only 70% of data used
    'ID3 (Impute)': 100,  # All data used but with bias
    'C4.5': 100,  # All data used effectively
    'CART': 100   # All data used effectively
}

methods = list(utilization_data.keys())
utilization = list(utilization_data.values())
colors2 = ['red', 'orange', 'lightblue', 'green']

bars2 = ax2.bar(methods, utilization, color=colors2, alpha=0.7, edgecolor='black', linewidth=2)
ax2.set_ylabel('Data Utilization (%)', fontsize=12)
ax2.set_title('Data Utilization with Missing Values', fontsize=14, weight='bold')
ax2.set_ylim(0, 110)
ax2.grid(True, alpha=0.3)

# Add percentage labels
for bar, util in zip(bars2, utilization):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{util}%', ha='center', va='bottom', fontsize=11, weight='bold')

# Rotate x-axis labels for better readability
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'robustness_comparison.png'), dpi=300, bbox_inches='tight')

print(f"\nVisualization files saved to: {save_dir}")
print("Files created:")
print("- id3_missing_value_strategies.png")
print("- c45_fractional_instance_method.png")
print("- cart_surrogate_splits.png")
print("- missing_value_robustness_comparison.png")
print("- fractional_instance_example.png")
print("- surrogate_splits_example.png")
print("- robustness_comparison.png")
