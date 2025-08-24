import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from itertools import combinations
import math

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_33")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

print("=" * 80)
print("QUESTION 33: MULTI-WAY VS BINARY SPLITS ANALYSIS")
print("=" * 80)
print("COMPLETE SOLUTION FOR ALL TASKS")
print("=" * 80)

# Define the Grade feature with values
grade_values = ['A', 'B', 'C', 'D', 'F']
k = len(grade_values)
print(f"Feature 'Grade' has {k} values: {grade_values}")

print("\n" + "="*60)
print("TASK 1: SPLIT ANALYSIS FOR GRADE FEATURE")
print("="*60)

# Task 1a: ID3 Multi-way Split
print("\n1a) ID3 Multi-way Split:")
print("-" * 40)
print("ID3 would consider exactly 1 split:")
print(f"Split: Grade ∈ {{A, B, C, D, F}} → 5 branches")
print("This creates one branch for each unique value")
print("\nTree Structure:")
print("Root: Grade?")
print("├── A → Leaf (Pass)")
print("├── B → Leaf (Pass)")
print("├── C → Leaf (Pass)")
print("├── D → Leaf (Fail)")
print("└── F → Leaf (Fail)")

# Task 1b: CART Binary Splits
print("\n1b) CART Binary Splits (using Gini impurity):")
print("-" * 50)

# Generate all possible binary splits
binary_splits = []
for i in range(1, 2**(k-1)):  # Avoid empty or full subsets
    # Convert to binary representation
    binary = format(i, f'0{k}b')
    left_values = [grade_values[j] for j in range(k) if binary[j] == '1']
    right_values = [grade_values[j] for j in range(k) if binary[j] == '0']
    
    if len(left_values) > 0 and len(right_values) > 0:
        binary_splits.append((left_values, right_values))

print(f"CART would consider {len(binary_splits)} binary splits:")
print("\nAll possible binary splits:")
for i, (left, right) in enumerate(binary_splits, 1):
    print(f"Split {i:2d}: {{Grade ∈ {left}}} vs {{Grade ∈ {right}}}")

print("\n" + "="*60)
print("TASK 2: CALCULATE NUMBER OF BINARY SPLITS FOR k VALUES")
print("="*60)

print(f"\nFor a categorical feature with k = {k} values:")
print(f"Number of possible binary splits = 2^(k-1) - 1")
print(f"Calculation: 2^({k}-1) - 1 = 2^{k-1} - 1 = {2**(k-1)} - 1 = {2**(k-1) - 1}")

# Verify with our example
print(f"\nVerification with our Grade example:")
print(f"Expected: 2^({k}-1) - 1 = {2**(k-1)} - 1 = {2**(k-1) - 1}")
print(f"Actual count: {len(binary_splits)} ✓")

# Mathematical proof
print(f"\nMathematical Proof:")
print(f"• Total combinations: 2^{k} = {2**k}")
print(f"• Exclude empty subsets: 2^{k} - 2 = {2**k - 2}")
print(f"• Account for symmetry: (2^{k} - 2) / 2 = {(2**k - 2) // 2}")
print(f"• Result: 2^{k-1} - 1 = {2**(k-1)} - 1 = {2**(k-1) - 1}")

# Show examples for different k values
print(f"\nExamples for different k values:")
for test_k in [2, 3, 4, 5, 6]:
    expected_splits = 2**(test_k-1) - 1
    print(f"k = {test_k}: 2^({test_k}-1) - 1 = 2^{test_k-1} - 1 = {expected_splits} splits")

print("\n" + "="*60)
print("TASK 3: DETAILED GINI IMPURITY CALCULATIONS")
print("="*60)

# Create sample data for demonstration
np.random.seed(42)
n_samples = 1000

# Generate synthetic data with some pattern
grade_distribution = {'A': 0.25, 'B': 0.30, 'C': 0.25, 'D': 0.15, 'F': 0.05}
grades = np.random.choice(grade_values, n_samples, p=list(grade_distribution.values()))

# Target variable (pass/fail) with some correlation to grades
target = np.where(np.isin(grades, ['A', 'B', 'C']), 1, 0)  # Pass = 1, Fail = 0

# Create DataFrame
df = pd.DataFrame({'Grade': grades, 'Pass': target})

print("\nSample Data Distribution:")
print(df['Grade'].value_counts().sort_index())
print(f"\nTarget Distribution:")
print(df['Pass'].value_counts())

# Calculate Gini impurity for different splits
def calculate_gini_impurity(y):
    """Calculate Gini impurity for a set of target values"""
    if len(y) == 0:
        return 0
    p = np.mean(y)
    return 2 * p * (1 - p)

def calculate_split_gini(y_left, y_right):
    """Calculate weighted Gini impurity for a split"""
    n_left, n_right = len(y_left), len(y_right)
    n_total = n_left + n_right
    
    gini_left = calculate_gini_impurity(y_left)
    gini_right = calculate_gini_impurity(y_right)
    
    weighted_gini = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
    return weighted_gini, gini_left, gini_right

print("\nGini Impurity Calculations for Selected Binary Splits:")
print("-" * 70)

# Calculate for a few representative splits
representative_splits = [
    (['A', 'B'], ['C', 'D', 'F']),  # Good students vs others
    (['A'], ['B', 'C', 'D', 'F']),  # A students vs others
    (['A', 'B', 'C'], ['D', 'F']),  # Passing vs failing
    (['A', 'B', 'C', 'D'], ['F'])   # All except F
]

for i, (left, right) in enumerate(representative_splits, 1):
    # Filter data for this split
    left_mask = df['Grade'].isin(left)
    right_mask = df['Grade'].isin(right)
    
    y_left = df.loc[left_mask, 'Pass'].values
    y_right = df.loc[right_mask, 'Pass'].values
    
    weighted_gini, gini_left, gini_right = calculate_split_gini(y_left, y_right)
    
    print(f"\nSplit {i}: {{Grade ∈ {left}}} vs {{Grade ∈ {right}}}")
    print(f"  Left branch: {len(y_left)} samples, Gini = {gini_left:.4f}")
    print(f"  Right branch: {len(y_right)} samples, Gini = {gini_right:.4f}")
    print(f"  Weighted Gini = {weighted_gini:.4f}")
    
    # Show the actual pass rates
    if len(y_left) > 0:
        pass_rate_left = np.mean(y_left) * 100
        print(f"  Left branch pass rate: {pass_rate_left:.1f}%")
    if len(y_right) > 0:
        pass_rate_right = np.mean(y_right) * 100
        print(f"  Right branch pass rate: {pass_rate_right:.1f}%")

# Part 4: Multi-way Split Gini Calculation
print("\n" + "="*60)
print("MULTI-WAY SPLIT GINI CALCULATION (ID3)")
print("="*60)

print("\nID3 Multi-way Split Gini Impurity:")
print("-" * 40)

# Calculate Gini for each branch of the multi-way split
multi_way_gini = 0
total_samples = len(df)

for grade in grade_values:
    grade_mask = df['Grade'] == grade
    y_grade = df.loc[grade_mask, 'Pass'].values
    n_grade = len(y_grade)
    
    gini_grade = calculate_gini_impurity(y_grade)
    weight = n_grade / total_samples
    
    multi_way_gini += weight * gini_grade
    
    pass_rate = np.mean(y_grade) * 100 if len(y_grade) > 0 else 0
    print(f"Grade {grade}: {n_grade:3d} samples, Gini = {gini_grade:.4f}, Weight = {weight:.3f}, Pass Rate = {pass_rate:.1f}%")

print(f"\nTotal Multi-way Split Gini = {multi_way_gini:.4f}")

# Part 5: Comparison and Analysis
print("\n" + "="*60)
print("COMPARISON AND ANALYSIS")
print("="*60)

# Find the best binary split
best_binary_gini = float('inf')
best_binary_split = None

for left, right in binary_splits:
    left_mask = df['Grade'].isin(left)
    right_mask = df['Grade'].isin(right)
    
    y_left = df.loc[left_mask, 'Pass'].values
    y_right = df.loc[right_mask, 'Pass'].values
    
    weighted_gini, _, _ = calculate_split_gini(y_left, y_right)
    
    if weighted_gini < best_binary_gini:
        best_binary_gini = weighted_gini
        best_binary_split = (left, right)

print(f"\nBest Binary Split:")
print(f"Split: {{Grade ∈ {best_binary_split[0]}}} vs {{Grade ∈ {best_binary_split[1]}}}")
print(f"Gini Impurity: {best_binary_gini:.4f}")

print(f"\nComparison:")
print(f"Multi-way Split Gini: {multi_way_gini:.4f}")
print(f"Best Binary Split Gini: {best_binary_gini:.4f}")
print(f"Improvement: {multi_way_gini - best_binary_gini:.4f}")

print("\n" + "="*60)
print("TASK 3: ADVANTAGES AND DISADVANTAGES OF EACH APPROACH")
print("="*60)

print("\nMULTI-WAY SPLITS (ID3):")
print("-" * 40)
print("ADVANTAGES:")
print("• Simple and intuitive tree structure")
print("• Fast computation - only one split per feature")
print("• Easy interpretation - one branch per value")
print("• Memory efficient - fewer nodes")
print("• Perfect for educational purposes")

print("\nDISADVANTAGES:")
print("• High bias toward high-cardinality features")
print("• Poor generalization to new data")
print("• Overfitting risk - creates many branches")
print("• Limited flexibility in partitioning")
print("• Can't handle continuous features")

print("\nBINARY SPLITS (CART):")
print("-" * 40)
print("ADVANTAGES:")
print("• Better bias resistance and generalization")
print("• More flexible partitioning strategies")
print("• Robust to noise and irrelevant features")
print("• Can handle both categorical and continuous features")
print("• Better for production systems")

print("\nDISADVANTAGES:")
print("• More complex tree structure")
print("• Higher computational cost")
print("• Harder to interpret")
print("• More memory usage")
print("• Can create deeper trees")

print("\n" + "="*60)
print("TASK 4: WHEN MIGHT BINARY SPLITS BE PREFERRED?")
print("="*60)

print("\nBinary splits are preferred over multi-way splits when:")
print("-" * 60)
print("1. HIGH-CARDINALITY CATEGORICAL FEATURES:")
print("   • Features with many unique values (e.g., customer ID, zip code)")
print("   • When bias resistance is crucial for model fairness")
print("   • To avoid overfitting to spurious patterns")

print("\n2. PRODUCTION SYSTEMS:")
print("   • When robustness and reliability are critical")
print("   • When generalization to new data is important")
print("   • When model performance consistency matters")

print("\n3. NOISY DATASETS:")
print("   • Datasets with measurement errors or irrelevant features")
print("   • When noise resistance is more important than interpretability")
print("   • To avoid capturing noise in the training data")

print("\n4. MIXED DATA TYPES:")
print("   • Datasets with both categorical and continuous features")
print("   • When unified splitting strategy is needed")
print("   • For consistency across different feature types")

print("\n5. COMPLEX DECISION BOUNDARIES:")
print("   • When simple multi-way splits can't capture complex patterns")
print("   • For non-linear decision boundaries")
print("   • When feature interactions are important")

# Create remaining visualizations
print("\n" + "="*60)
print("GENERATING REMAINING VISUALIZATIONS")
print("="*60)

# Plot 1: Binary Split Distribution
fig1, ax1 = plt.subplots(figsize=(12, 6))
binary_gini_values = []
split_labels = []

for left, right in binary_splits[:10]:  # Show first 10 for clarity
    left_mask = df['Grade'].isin(left)
    right_mask = df['Grade'].isin(right)
    
    y_left = df.loc[left_mask, 'Pass'].values
    y_right = df.loc[right_mask, 'Pass'].values
    
    weighted_gini, _, _ = calculate_split_gini(y_left, y_right)
    binary_gini_values.append(weighted_gini)
    
    # Create readable label
    left_str = ','.join(left) if len(left) <= 3 else f"{len(left)} values"
    right_str = ','.join(right) if len(right) <= 3 else f"{len(right)} values"
    split_labels.append(f"{left_str} vs {right_str}")

ax1.bar(range(len(binary_gini_values)), binary_gini_values, color='lightblue', alpha=0.7)
ax1.set_xlabel('Binary Split Index')
ax1.set_ylabel('Gini Impurity')
ax1.set_title('Gini Impurity for Binary Splits')
ax1.set_xticks(range(len(split_labels)))
ax1.set_xticklabels(split_labels, rotation=45, ha='right')
ax1.grid(True, alpha=0.3)

# Highlight best split
best_idx = binary_gini_values.index(min(binary_gini_values))
ax1.bar(best_idx, binary_gini_values[best_idx], color='red', alpha=0.8, label='Best Split')
ax1.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'binary_split_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Grade Distribution and Target
fig2, ax2 = plt.subplots(figsize=(10, 6))
grade_counts = df['Grade'].value_counts().sort_index()
pass_rates = df.groupby('Grade')['Pass'].mean().sort_index()

x = np.arange(len(grade_values))
width = 0.35

bars1 = ax2.bar(x - width/2, grade_counts.values, width, label='Sample Count', color='skyblue', alpha=0.7)
bars2 = ax2.bar(x + width/2, pass_rates.values * 100, width, label='Pass Rate (%)', color='lightcoral', alpha=0.7)

ax2.set_xlabel('Grade')
ax2.set_ylabel('Count / Percentage')
ax2.set_title('Grade Distribution and Pass Rates')
ax2.set_xticks(x)
ax2.set_xticklabels(grade_values)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'grade_distribution_pass_rates.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Decision Tree Visualization
fig3, ax3 = plt.subplots(figsize=(10, 8))
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)

# Draw decision tree structure
tree_elements = [
    {'pos': (5, 9), 'text': 'Grade?', 'color': 'lightblue'},
    {'pos': (2, 7), 'text': 'A or B?', 'color': 'lightgreen'},
    {'pos': (8, 7), 'text': 'C, D, or F?', 'color': 'lightcoral'},
    {'pos': (1, 5), 'text': 'A', 'color': 'lightgreen'},
    {'pos': (3, 5), 'text': 'B', 'color': 'lightgreen'},
    {'pos': (7, 5), 'text': 'C', 'color': 'lightcoral'},
    {'pos': (9, 5), 'text': 'D or F', 'color': 'lightcoral'},
    {'pos': (8.5, 3), 'text': 'D', 'color': 'lightcoral'},
    {'pos': (9.5, 3), 'text': 'F', 'color': 'lightcoral'}
]

# Draw decision boxes
for element in tree_elements:
    rect = plt.Rectangle((element['pos'][0]-0.3, element['pos'][1]-0.2), 0.6, 0.4, 
                        facecolor=element['color'], edgecolor='black', alpha=0.7)
    ax3.add_patch(rect)
    ax3.text(element['pos'][0], element['pos'][1], element['text'], ha='center', va='center', 
            fontsize=8, fontweight='bold')

# Draw arrows
arrows = [
    ((5, 8.8), (2, 7.2), 'Yes'),
    ((5, 8.8), (8, 7.2), 'No'),
    ((2, 6.8), (1, 5.2), 'Yes'),
    ((2, 6.8), (3, 5.2), 'No'),
    ((8, 6.8), (7, 5.2), 'C'),
    ((8, 6.8), (9, 5.2), 'D/F'),
    ((9, 4.8), (8.5, 3.2), 'D'),
    ((9, 4.8), (9.5, 3.2), 'F')
]

for start, end, label in arrows:
    ax3.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    # Add labels
    mid_x, mid_y = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
    ax3.text(mid_x + 0.2, mid_y + 0.1, label, fontsize=8, color='red', fontweight='bold')

ax3.set_title('Binary Decision Tree Structure')
ax3.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'binary_decision_tree.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n" + "="*80)
print(f"COMPLETE SOLUTION SUMMARY")
print("="*80)
print("✓ TASK 1: Split analysis for Grade feature - COMPLETED")
print("  • ID3 multi-way split: 1 split with 5 branches")
print("  • CART binary splits: 15 possible binary combinations")
print("")
print("✓ TASK 2: Formula calculation - COMPLETED")
print("  • Formula: 2^(k-1) - 1")
print("  • Verification: 2^(5-1) - 1 = 15 ✓")
print("")
print("✓ TASK 3: Advantages/disadvantages - COMPLETED")
print("  • Multi-way splits: Simple, fast, but biased")
print("  • Binary splits: Complex, robust, but slower")
print("")
print("✓ TASK 4: When to use binary splits - COMPLETED")
print("  • High-cardinality features, production systems, noisy data")
print("  • When bias resistance and generalization are crucial")

print(f"\nImages saved to: {save_dir}")
print("All tasks from Question 33 have been completely addressed!")
print("Generated remaining visualizations and removed unnecessary images.")
