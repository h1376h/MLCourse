import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import comb
import os
from matplotlib.patches import Rectangle
import time

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_1_Quiz_3")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("=" * 80)
print("QUESTION 3: SUPERVISED vs UNSUPERVISED FEATURE SELECTION")
print("=" * 80)

# ============================================================================
# TASK 1: Main advantage of supervised feature selection
# ============================================================================
print("\n" + "="*60)
print("TASK 1: Main advantage of supervised feature selection")
print("="*60)

print("The main advantage of supervised feature selection is that it can identify")
print("features that are most relevant to the target variable by using the labels.")
print("This leads to better predictive performance and more interpretable models.")

# Create visualization showing supervised vs unsupervised
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Supervised feature selection
ax1.set_title('Supervised Feature Selection', fontsize=14, fontweight='bold')
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)

# Draw features and target
features = ['F1', 'F2', 'F3', 'F4', 'F5']
feature_positions = [(2, 8), (4, 8), (6, 8), (8, 8), (5, 5)]
target_pos = (5, 2)

# Draw features
for i, (feature, pos) in enumerate(zip(features, feature_positions)):
    if i < 3:  # Relevant features
        color = 'lightgreen'
        edge_color = 'darkgreen'
    else:  # Irrelevant features
        color = 'lightcoral'
        edge_color = 'darkred'
    
    ax1.add_patch(Rectangle((pos[0]-0.5, pos[1]-0.5), 1, 1, 
                           facecolor=color, edgecolor=edge_color, linewidth=2))
    ax1.text(pos[0], pos[1], feature, ha='center', va='center', fontweight='bold')

# Draw target
ax1.add_patch(Rectangle((target_pos[0]-0.5, target_pos[1]-0.5), 1, 1, 
                       facecolor='lightblue', edgecolor='darkblue', linewidth=2))
ax1.text(target_pos[0], target_pos[1], 'Target', ha='center', va='center', fontweight='bold')

# Draw connections for relevant features
for i in range(3):
    pos = feature_positions[i]
    ax1.arrow(pos[0], pos[1]-0.5, target_pos[0]-pos[0], target_pos[1]+0.5-pos[1]+0.5,
              head_width=0.2, head_length=0.2, fc='darkgreen', ec='darkgreen', linewidth=2)

ax1.set_xlabel('Features')
ax1.set_ylabel('Relevance')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Unsupervised feature selection
ax2.set_title('Unsupervised Feature Selection', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)

# Draw features without target
for i, (feature, pos) in enumerate(zip(features, feature_positions)):
    if i < 2:  # Features with high variance/correlation
        color = 'lightgreen'
        edge_color = 'darkgreen'
    else:  # Features with low variance/correlation
        color = 'lightcoral'
        edge_color = 'darkred'
    
    ax2.add_patch(Rectangle((pos[0]-0.5, pos[1]-0.5), 1, 1, 
                           facecolor=color, edgecolor=edge_color, linewidth=2))
    ax2.text(pos[0], pos[1], feature, ha='center', va='center', fontweight='bold')

# Draw connections between features (correlations)
ax2.arrow(2, 8, 2, 0, head_width=0.2, head_length=0.2, fc='darkgreen', ec='darkgreen', linewidth=2)
ax2.arrow(4, 8, 2, 0, head_width=0.2, head_length=0.2, fc='darkgreen', ec='darkgreen', linewidth=2)

ax2.set_xlabel('Features')
ax2.set_ylabel('Correlation')
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'supervised_vs_unsupervised.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# TASK 2: When to use unsupervised feature selection
# ============================================================================
print("\n" + "="*60)
print("TASK 2: When to use unsupervised feature selection")
print("="*60)

print("Unsupervised feature selection is used when:")
print("1. No labels are available (e.g., clustering problems)")
print("2. Labels are unreliable or noisy")
print("3. You want to reduce dimensionality before applying supervised methods")
print("4. You need to identify redundant or highly correlated features")
print("5. You want to preserve data structure and variance")

# ============================================================================
# TASK 3: Measuring feature relevance in unsupervised scenarios
# ============================================================================
print("\n" + "="*60)
print("TASK 3: Measuring feature relevance in unsupervised scenarios")
print("="*60)

print("In unsupervised scenarios, feature relevance is measured by:")
print("1. Variance: Features with low variance provide little information")
print("2. Correlation: Highly correlated features are redundant")
print("3. Mutual information between features")
print("4. Clustering quality when using the feature")
print("5. Information gain based on feature distributions")

# Create visualization of unsupervised metrics
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Variance-based selection
np.random.seed(42)
n_samples = 100
features = ['F1', 'F2', 'F3', 'F4', 'F5']
variances = [0.1, 0.5, 1.0, 2.0, 3.0]

# Generate sample data
data = np.random.randn(n_samples, 5)
for i in range(5):
    data[:, i] *= np.sqrt(variances[i])

# Plot variance
axes[0, 0].bar(features, variances, color=['lightgreen' if v < 1 else 'lightcoral' for v in variances])
axes[0, 0].set_title('Feature Variance')
axes[0, 0].set_ylabel('Variance')
axes[0, 0].axhline(y=1, color='red', linestyle='--', label='Threshold')
axes[0, 0].legend()

# Correlation matrix
corr_matrix = np.corrcoef(data.T)
im = axes[0, 1].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
axes[0, 1].set_title('Feature Correlation Matrix')
axes[0, 1].set_xticks(range(5))
axes[0, 1].set_yticks(range(5))
axes[0, 1].set_xticklabels(features)
axes[0, 1].set_yticklabels(features)
plt.colorbar(im, ax=axes[0, 1])

# Feature distributions
for i in range(5):
    axes[1, 0].hist(data[:, i], alpha=0.7, label=f'{features[i]} (var={variances[i]:.1f})')
axes[1, 0].set_title('Feature Distributions')
axes[1, 0].set_xlabel('Value')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()

# Feature ranking
relevance_scores = [1/v if v > 0 else 0 for v in variances]  # Inverse variance
axes[1, 1].bar(features, relevance_scores, color=['lightgreen' if s > 1 else 'lightcoral' for s in relevance_scores])
axes[1, 1].set_title('Feature Relevance Scores (1/Variance)')
axes[1, 1].set_ylabel('Relevance Score')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'unsupervised_metrics.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# TASK 4: Number of possible feature subsets
# ============================================================================
print("\n" + "="*60)
print("TASK 4: Number of possible feature subsets")
print("="*60)

n_samples = 1000
n_features = 50

print(f"Given: {n_samples} samples with {n_features} features")
print(f"Formula: Number of subsets = 2^{n_features} - 1")
print(f"Calculation: 2^{n_features} - 1 = 2^{n_features} - 1")

print("\nDETAILED CALCULATION:")
print("Step 1: Understand the problem")
print("   - We have n = 50 features")
print("   - Each feature can be either included (1) or excluded (0) from a subset")
print("   - This gives us 2^50 possible combinations")
print("   - We subtract 1 to exclude the empty set (no features selected)")

print("\nStep 2: Break down the calculation")
print("   2^50 = 2^10 × 2^10 × 2^10 × 2^10 × 2^10")
print("   We know: 2^10 = 1,024")
print("   Therefore: 2^50 = 1,024^5")

print("\nStep 3: Calculate step by step")
print("   1,024^2 = 1,024 × 1,024 = 1,048,576")
print("   1,048,576^2 = 1,048,576 × 1,048,576 = 1,099,511,627,776")
print("   1,099,511,627,776 × 1,024 = 1,125,899,906,842,624")
print("   Final result: 2^50 = 1,125,899,906,842,624")

print("\nStep 4: Subtract 1 for the empty set")
print("   Total subsets = 2^50 - 1 = 1,125,899,906,842,624 - 1")
print("   Total subsets = 1,125,899,906,842,623")

# Calculate using Python
total_subsets = 2**n_features - 1
print(f"\nVerification with Python: {total_subsets:,}")

# Scientific notation for large numbers
scientific_notation = f"{total_subsets:.2e}"
print(f"In scientific notation: {scientific_notation}")

print(f"\nINTERPRETATION:")
print(f"- This means we have over 1.1 quadrillion possible feature subsets")
print(f"- Even if we could evaluate 1 million subsets per second,")
print(f"  it would take over 35 years to evaluate all possibilities!")
print(f"- This demonstrates why exhaustive search is impractical")

# Create visualization of subset growth
n_features_range = np.arange(1, 21)
subset_counts = 2**n_features_range - 1

plt.figure(figsize=(12, 8))
plt.semilogy(n_features_range, subset_counts, 'b-o', linewidth=2, markersize=8)
plt.axhline(y=total_subsets, color='red', linestyle='--', 
            label=f'50 features: {scientific_notation}')
plt.xlabel('Number of Features (n)')
plt.ylabel(r'Number of Feature Subsets ($2^n - 1$)')
plt.title('Exponential Growth of Feature Subset Space')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(n_features_range[::2])

# Add annotations
plt.annotate(f'10 features: {2**10 - 1:,}', xy=(10, 2**10 - 1), 
             xytext=(12, 2**10 - 1), arrowprops=dict(arrowstyle='->'))
plt.annotate(f'20 features: {2**20 - 1:,}', xy=(20, 2**20 - 1), 
             xytext=(18, 2**20 - 1), arrowprops=dict(arrowstyle='->'))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'subset_growth.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# TASK 5: Subsets with exactly 10 features
# ============================================================================
print("\n" + "="*60)
print("TASK 5: Subsets with exactly 10 features")
print("="*60)

k = 10
print(f"Given: {n_features} total features, want exactly {k} features")
print(f"Formula: C({n_features}, {k}) = {n_features}! / ({k}! × ({n_features}-{k})!)")
print(f"Calculation: C({n_features}, {k}) = {n_features}! / ({k}! × {n_features-k}!)")

print("\nDETAILED CALCULATION:")
print("Step 1: Understand the combination formula")
print("   C(n,k) = n! / (k! × (n-k)!)")
print(f"   C({n_features},{k}) = {n_features}! / ({k}! × {n_features-k}!)")
print(f"   C({n_features},{k}) = {n_features}! / ({k}! × {n_features-k}!)")

print("\nStep 2: Break down the factorials")
print(f"   {n_features}! = {n_features} × {n_features-1} × {n_features-2} × ... × 2 × 1")
print(f"   {k}! = {k} × {k-1} × {k-2} × ... × 2 × 1 = {k}!")
print(f"   {n_features-k}! = {n_features-k} × {n_features-k-1} × ... × 2 × 1")

print("\nStep 3: Simplify using cancellation")
print(f"   C({n_features},{k}) = {n_features} × {n_features-1} × {n_features-2} × ... × {n_features-k+1}")
print(f"   C({n_features},{k}) = {n_features} × {n_features-1} × {n_features-2} × ... × {n_features-k+1}")
print(f"   C({n_features},{k}) = {n_features} × {n_features-1} × {n_features-2} × ... × {n_features-k+1}")

print("\nStep 4: Calculate the product")
print(f"   Numerator: {n_features} × {n_features-1} × {n_features-2} × ... × {n_features-k+1}")
print(f"   Numerator: {n_features} × {n_features-1} × {n_features-2} × ... × {n_features-k+1}")

# Calculate using scipy
exact_k_subsets = comb(n_features, k, exact=True)
print(f"\nResult from scipy: {exact_k_subsets:,} feature subsets with exactly {k} features")

# Calculate manually to show the formula
numerator = 1
for i in range(n_features - k + 1, n_features + 1):
    numerator *= i

denominator = 1
for i in range(1, k + 1):
    denominator *= i

manual_result = numerator // denominator
print(f"Manual calculation: {manual_result:,}")

print(f"\nVERIFICATION:")
print(f"- Both methods give the same result: {exact_k_subsets:,}")
print(f"- This represents the number of ways to choose {k} features from {n_features} total features")
print(f"- It's a subset of the total {total_subsets:,} possible feature combinations")
print(f"- Percentage: {exact_k_subsets/total_subsets*100:.10f}% of all possible subsets")

# Create visualization of subset sizes
subset_sizes = np.arange(0, n_features + 1)
subset_counts_by_size = [comb(n_features, k, exact=True) for k in subset_sizes]

plt.figure(figsize=(12, 8))
plt.bar(subset_sizes, subset_counts_by_size, color='skyblue', edgecolor='navy')
plt.axvline(x=k, color='red', linestyle='--', linewidth=2, 
            label=f'Exactly {k} features: {exact_k_subsets:,}')
plt.xlabel('Number of Features in Subset')
plt.ylabel('Number of Subsets')
plt.title(f'Distribution of Feature Subset Sizes for {n_features} Total Features')
plt.grid(True, alpha=0.3)
plt.legend()

# Highlight the peak
peak_size = n_features // 2
peak_count = comb(n_features, peak_size, exact=True)
plt.annotate(f'Peak at {peak_size} features:\n{peak_count:,} subsets', 
             xy=(peak_size, peak_count), xytext=(peak_size + 5, peak_count * 0.8),
             arrowprops=dict(arrowstyle='->', color='red'),
             bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'subset_size_distribution.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# TASK 6: Time constraint calculation
# ============================================================================
print("\n" + "="*60)
print("TASK 6: Time constraint calculation")
print("="*60)

evaluation_rate = 1000  # combinations per second
total_time = 1 * 60 * 60  # 1 hour in seconds
target_percentage = 0.80  # 80%

print(f"Given:")
print(f"- Evaluation rate: {evaluation_rate:,} combinations/second")
print(f"- Total time: {total_time:,} seconds ({total_time/3600:.1f} hours)")
print(f"- Target: Evaluate at least {target_percentage*100:.0f}% of all combinations")
print(f"- Formula: Total combinations = 2^n - 1")

print(f"\nDETAILED CALCULATION:")
print("Step 1: Calculate maximum combinations we can evaluate")
print(f"   Evaluation rate: {evaluation_rate:,} combinations/second")
print(f"   Total time: {total_time:,} seconds = {total_time/3600:.1f} hours")
print(f"   Max combinations = Rate × Time")
print(f"   Max combinations = {evaluation_rate:,} × {total_time:,}")
max_combinations = evaluation_rate * total_time
print(f"   Max combinations = {max_combinations:,}")

print(f"\nStep 2: Find n where 2^n - 1 ≤ {max_combinations:,}")
print(f"   We need: 2^n - 1 ≤ {max_combinations:,}")
print(f"   Therefore: 2^n ≤ {max_combinations + 1:,}")
print(f"   We need to find the largest n such that 2^n ≤ {max_combinations + 1:,}")

print(f"\nStep 3: Systematic approach to find n")
print(f"   Let's test values of n systematically:")
print(f"   n = 20: 2^20 = {2**20:,} ≤ {max_combinations + 1:,}? {'Yes' if 2**20 <= max_combinations + 1 else 'No'}")
print(f"   n = 21: 2^21 = {2**21:,} ≤ {max_combinations + 1:,}? {'Yes' if 2**21 <= max_combinations + 1 else 'No'}")
print(f"   n = 22: 2^22 = {2**22:,} ≤ {max_combinations + 1:,}? {'Yes' if 2**22 <= max_combinations + 1 else 'No'}")

# Find the maximum n
n_max = 0
while 2**n_max <= max_combinations + 1:
    n_max += 1
n_max -= 1  # Adjust for the last increment

print(f"\nStep 4: Determine maximum n")
print(f"   Maximum n = {n_max}")
print(f"   Verification: 2^{n_max} - 1 = {2**n_max - 1:,}")
print(f"   Check: {2**n_max - 1:,} ≤ {max_combinations:,}? {'Yes' if 2**n_max - 1 <= max_combinations else 'No'}")

print(f"\nStep 5: Check if we meet the 80% requirement")
combinations_at_n_max = 2**n_max - 1
percentage_evaluated = combinations_at_n_max / max_combinations
print(f"   Combinations at max n: {combinations_at_n_max:,}")
print(f"   Max combinations possible: {max_combinations:,}")
print(f"   Percentage evaluated: {combinations_at_n_max:,} / {max_combinations:,} = {percentage_evaluated:.1%}")

if percentage_evaluated >= target_percentage:
    print(f"   $\\checkmark$ We can evaluate {percentage_evaluated:.1%} ≥ {target_percentage:.1%}")
else:
    print(f"   $\\times$ We can only evaluate {percentage_evaluated:.1%} < {target_percentage:.1%}")

print(f"\nStep 6: What if we want to meet the 80% target?")
print(f"   Target: {target_percentage*100:.0f}% of {max_combinations:,} = {target_percentage * max_combinations:.0f}")
print(f"   We need: 2^n - 1 ≥ {target_percentage * max_combinations:.0f}")
print(f"   Therefore: 2^n ≥ {target_percentage * max_combinations + 1:.0f}")

# Find what n would be needed for 80%
n_needed_for_target = 0
while 2**n_needed_for_target < target_percentage * max_combinations + 1:
    n_needed_for_target += 1

print(f"   n needed for 80%: {n_needed_for_target}")
print(f"   Verification: 2^{n_needed_for_target} - 1 = {2**n_needed_for_target - 1:,}")
print(f"   This would require: {2**n_needed_for_target - 1:,} / {evaluation_rate:,} = {(2**n_needed_for_target - 1) / evaluation_rate:.1f} seconds")
print(f"   Time needed: {(2**n_needed_for_target - 1) / evaluation_rate / 3600:.1f} hours")

# Create visualization of time constraints
n_range = np.arange(1, 26)
combinations_range = 2**n_range - 1
time_required = combinations_range / evaluation_rate

plt.figure(figsize=(15, 10))

# Plot 1: Combinations vs Features
plt.subplot(2, 2, 1)
plt.semilogy(n_range, combinations_range, 'b-o', linewidth=2, markersize=6)
plt.axhline(y=max_combinations, color='red', linestyle='--', 
            label=f'Max combinations: {max_combinations:,}')
plt.axvline(x=n_max, color='green', linestyle='--', 
            label=f'Max features: {n_max}')
plt.xlabel('Number of Features (n)')
plt.ylabel(r'Number of Combinations ($2^n - 1$)')
plt.title('Feature Combinations vs Time Constraint')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 2: Time required vs Features
plt.subplot(2, 2, 2)
plt.semilogy(n_range, time_required, 'g-o', linewidth=2, markersize=6)
plt.axhline(y=total_time, color='red', linestyle='--', 
            label=f'Available time: {total_time/3600:.1f} hours')
plt.axvline(x=n_max, color='green', linestyle='--', 
            label=f'Max features: {n_max}')
plt.xlabel('Number of Features (n)')
plt.ylabel('Time Required (seconds)')
plt.title('Time Required vs Features')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 3: Percentage evaluated vs Features
plt.subplot(2, 2, 3)
percentage_range = np.minimum(combinations_range / max_combinations, 1.0)
plt.plot(n_range, percentage_range * 100, 'm-o', linewidth=2, markersize=6)
plt.axhline(y=target_percentage * 100, color='red', linestyle='--', 
            label=f'Target: {target_percentage*100:.0f}%')
plt.axvline(x=n_max, color='green', linestyle='--', 
            label=f'Max features: {n_max}')
plt.xlabel('Number of Features (n)')
plt.ylabel('Percentage of Combinations Evaluated (%)')
plt.title('Percentage Evaluated vs Features')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 4: Summary table
plt.subplot(2, 2, 4)
plt.axis('off')
table_data = [
    ['Metric', 'Value'],
    ['Max combinations', f'{max_combinations:,}'],
    ['Max features (n)', f'{n_max}'],
    ['Combinations at max n', f'{2**n_max - 1:,}'],
    ['Percentage evaluated', f'{percentage_evaluated:.1%}'],
    ['Meets 80% target', '$\\checkmark$' if percentage_evaluated >= target_percentage else '$\\times$']
]

table = plt.table(cellText=table_data, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'time_constraints.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# SUMMARY AND CONCLUSIONS
# ============================================================================
print("\n" + "="*80)
print("SUMMARY AND CONCLUSIONS")
print("="*80)

print("\nKey Results:")
print(f"1. Supervised feature selection uses labels to identify relevant features")
print(f"2. Unsupervised methods use variance, correlation, and information measures")
print(f"3. With {n_features} features: {total_subsets:,} total subsets")
print(f"4. Exactly {k} features: {exact_k_subsets:,} subsets")
print(f"5. Time constraint: Can handle up to {n_max} features in 1 hour")
print(f"6. Percentage evaluated: {percentage_evaluated:.1%}")

print(f"\nPRACTICAL IMPLICATIONS:")
print(f"- Feature selection space grows exponentially (2^n)")
print(f"- Exhaustive search becomes impractical beyond ~20 features")
print(f"- Need for efficient algorithms (greedy, genetic, etc.)")
print(f"- Time constraints severely limit feature set size for exhaustive search")

print(f"\nMATHEMATICAL INSIGHTS:")
print(f"- The growth rate 2^n means doubling the features quadruples the search space")
print(f"- For n = 50, we have 1.13 × 10^15 subsets")
print(f"- For n = 100, we would have 1.27 × 10^30 subsets")
print(f"- This demonstrates why feature selection is a fundamental challenge in ML")

print(f"\nALGORITHM RECOMMENDATIONS:")
print(f"- n < 20: Exhaustive search may be feasible")
print(f"- 20 ≤ n < 50: Greedy algorithms (forward/backward selection)")
print(f"- 50 ≤ n < 100: Genetic algorithms, randomized methods")
print(f"- n ≥ 100: Heuristic approaches, domain knowledge integration")

print(f"\nPlots saved to: {save_dir}")

# Don't show plots, just save them
print("All plots saved successfully!")
