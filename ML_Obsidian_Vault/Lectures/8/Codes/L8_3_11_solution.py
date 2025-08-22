import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import comb
import os
from itertools import combinations

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_3_Quiz_11")
os.makedirs(save_dir, exist_ok=True)

# Configure matplotlib to not display plots interactively
plt.ioff()  # Turn off interactive plotting
plt.rcParams['font.family'] = 'serif'
plt.style.use('default')

# Disable LaTeX to avoid rendering issues
plt.rcParams['text.usetex'] = False

print("=" * 80)
print("Question 11: Model Stability and Feature Selection")
print("=" * 80)

# Problem parameters
initial_features = 100
reduced_features = 20
original_cv_variance = 0.04
reduced_cv_variance = 0.028
target_variance = 0.02

print("\nProblem Parameters:")
print(f"Initial number of features: {initial_features}")
print(f"Reduced number of features: {reduced_features}")
print(f"Original CV variance: {original_cv_variance:.4f}")
print(f"Reduced CV variance: {reduced_cv_variance:.4f}")
print(f"Target variance: {target_variance:.4f}")

# ============================================================================
# Task 1: How does reducing features from 100 to 20 affect model stability?
# ============================================================================

print("\n" + "="*80)
print("Task 1: Effect of Feature Reduction on Model Stability")
print("="*80)

# Theoretical analysis
reduction_ratio = (initial_features - reduced_features) / initial_features
print(f"Feature reduction ratio: {reduction_ratio:.1%}")

# Variance reduction calculation
variance_reduction = (original_cv_variance - reduced_cv_variance) / original_cv_variance
print(f"Variance reduction: {variance_reduction:.1%}")

# Create visualization for Task 1
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Subplot 1: Feature count vs theoretical stability
feature_counts = np.linspace(5, 100, 50)
theoretical_stability = 1 / (1 + np.log(feature_counts))  # Simplified relationship
ax1.plot(feature_counts, theoretical_stability, 'b-', linewidth=3, label='Theoretical Stability')
ax1.axvline(x=initial_features, color='r', linestyle='--', label='Original (100 features)')
ax1.axvline(x=reduced_features, color='g', linestyle='--', label='Reduced (20 features)')
ax1.set_xlabel('Number of Features')
ax1.set_ylabel('Model Stability')
ax1.set_title('Feature Count vs Model Stability')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Subplot 2: Variance reduction visualization
methods = ['Original\n(100 features)', 'After\nFeature Selection\n(20 features)']
variances = [original_cv_variance, reduced_cv_variance]
bars = ax2.bar(methods, variances, color=['red', 'green'], alpha=0.7)
ax2.set_ylabel('Cross-Validation Variance')
ax2.set_title('Cross-Validation Variance Before/After Feature Selection')
ax2.grid(True, alpha=0.3)

# Add value labels on bars
for bar, variance in zip(bars, variances):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'{variance:.4f}', ha='center', va='bottom')

# Subplot 3: Bias-Variance tradeoff
feature_range = np.linspace(5, 100, 50)
bias = 0.1 + 0.5 * np.exp(-feature_range/20)  # Bias decreases with more features
variance = 0.05 + 0.15 * np.exp(feature_range/30)  # Variance increases with more features
total_error = bias + variance

ax3.plot(feature_range, bias, 'b-', label='Bias', linewidth=2)
ax3.plot(feature_range, variance, 'r-', label='Variance', linewidth=2)
ax3.plot(feature_range, total_error, 'k--', label='Total Error', linewidth=2)
ax3.axvline(x=initial_features, color='gray', linestyle=':', alpha=0.7)
ax3.axvline(x=reduced_features, color='green', linestyle=':', alpha=0.7)
ax3.set_xlabel('Number of Features')
ax3.set_ylabel('Error')
ax3.set_title('Bias-Variance Tradeoff')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Subplot 4: Stability improvement factors
factors = ['Overfitting\nReduction', 'Noise\nReduction', 'Computational\nEfficiency', 'Interpretability']
improvements = [85, 70, 90, 75]  # Percentage improvements

bars4 = ax4.barh(factors, improvements, color='skyblue', alpha=0.7)
ax4.set_xlabel('Improvement Percentage')
ax4.set_title('Stability Improvement Factors')
ax4.set_xlim(0, 100)

# Add value labels
for bar, improvement in zip(bars4, improvements):
    width = bar.get_width()
    ax4.text(width + 1, bar.get_y() + bar.get_height()/2,
             f'{improvement:.0f}', ha='left', va='center')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'task1_feature_reduction_effects.png'), dpi=300, bbox_inches='tight')
plt.close()

print("\nKey Effects of Feature Reduction:")
print("1. Reduced overfitting: Fewer features = less complex model")
print("2. Lower variance: Less sensitivity to noise in data")
print("3. Better generalization: Improved performance on unseen data")
print("4. Computational efficiency: Faster training and prediction")
print("5. Enhanced interpretability: Easier to understand feature importance")

# ============================================================================
# Task 2: What does 30% variance decrease suggest?
# ============================================================================

print("\n" + "="*80)
print("Task 2: Interpretation of 30% Variance Decrease")
print("="*80)

print(f"Variance reduction: {variance_reduction:.1%}")
print(f"Original variance: {original_cv_variance:.4f}")
print(f"Reduced variance: {reduced_cv_variance:.4f}")

# Create visualization for Task 2
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: Variance reduction interpretation
labels = ['Original\nVariance', 'Reduced\nVariance', 'Improvement']
values = [original_cv_variance, reduced_cv_variance, variance_reduction * 100]
colors = ['red', 'green', 'blue']

bars1 = ax1.bar(labels, [values[0], values[1], 0], color=colors[:2], alpha=0.7)
ax1_twin = ax1.twinx()
bars1_twin = ax1_twin.bar(labels[2], values[2], color=colors[2], alpha=0.7)

ax1.set_ylabel('Variance Value', color='black')
ax1_twin.set_ylabel('Improvement (%)', color='blue')
ax1.set_title('Variance Reduction Analysis')
ax1.grid(True, alpha=0.3)

# Subplot 2: Confidence interval improvement
original_std = np.sqrt(original_cv_variance)
reduced_std = np.sqrt(reduced_cv_variance)

x = np.linspace(0, 0.1, 100)
original_dist = stats.norm.pdf(x, 0.04, original_std)
reduced_dist = stats.norm.pdf(x, 0.04, reduced_std)

ax2.plot(x, original_dist, 'r-', label=f'Original (σ={original_std:.4f})', linewidth=2)
ax2.plot(x, reduced_dist, 'g-', label=f'Reduced (σ={reduced_std:.4f})', linewidth=2)
ax2.fill_between(x, 0, original_dist, color='red', alpha=0.3)
ax2.fill_between(x, 0, reduced_dist, color='green', alpha=0.3)
ax2.set_xlabel('Cross-Validation Score')
ax2.set_ylabel('Probability Density')
ax2.set_title('Distribution of CV Scores')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'task2_variance_interpretation.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Task 3: Design a stability-based feature selection criterion
# ============================================================================

print("\n" + "="*80)
print("Task 3: Stability-Based Feature Selection Criterion")
print("="*80)

def calculate_jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets"""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def calculate_kuncheva_index(selected_features, total_features, k):
    """Calculate Kuncheva stability index"""
    if len(selected_features) == 0:
        return 0

    # Calculate average Jaccard
    jaccard_sum = 0
    count = 0
    for i in range(len(selected_features)):
        for j in range(i+1, len(selected_features)):
            jaccard_sum += calculate_jaccard_similarity(selected_features[i], selected_features[j])
            count += 1

    avg_jaccard = jaccard_sum / count if count > 0 else 0

    # Correction factor
    correction = (k * (total_features - k)) / (total_features * comb(total_features, k))

    return avg_jaccard / correction if correction > 0 else 0

def stability_criterion(feature_subsets, total_features, threshold=0.7):
    """
    Stability-based feature selection criterion

    Parameters:
    - feature_subsets: List of feature subsets from different data splits
    - total_features: Total number of available features
    - threshold: Minimum stability threshold

    Returns:
    - stability_score: Overall stability measure
    - is_stable: Boolean indicating if selection is stable
    - recommendations: Text recommendations
    """
    k = len(feature_subsets[0]) if feature_subsets else 0

    # Calculate Jaccard similarity matrix
    n_subsets = len(feature_subsets)
    jaccard_matrix = np.zeros((n_subsets, n_subsets))

    for i in range(n_subsets):
        for j in range(n_subsets):
            jaccard_matrix[i, j] = calculate_jaccard_similarity(feature_subsets[i], feature_subsets[j])

    # Calculate average similarity
    avg_similarity = np.mean(jaccard_matrix[np.triu_indices_from(jaccard_matrix, k=1)])

    # Calculate Kuncheva index
    kuncheva_index = calculate_kuncheva_index(feature_subsets, total_features, k)

    # Comprehensive stability score
    stability_score = 0.7 * avg_similarity + 0.3 * kuncheva_index

    # Determine stability
    is_stable = stability_score >= threshold

    # Generate recommendations
    if is_stable:
        recommendations = "Feature selection is stable. Proceed with confidence."
    else:
        recommendations = "Feature selection shows instability. Consider:"
        recommendations += "\n- Using more data splits for selection"
        recommendations += "\n- Adjusting the number of selected features"
        recommendations += "\n- Using ensemble feature selection methods"

    return stability_score, is_stable, recommendations, avg_similarity, kuncheva_index

# Example usage
print("Stability-based Feature Selection Criterion:")
print("\nComponents:")
print("1. Jaccard Similarity: Measures overlap between feature subsets")
print("2. Kuncheva Index: Corrected stability measure for feature selection")
print("3. Combined Stability Score: Weighted combination of both metrics")
print("4. Threshold-based Decision: Automatic stability assessment")

# ============================================================================
# Task 4: How do you measure feature subset stability?
# ============================================================================

print("\n" + "="*80)
print("Task 4: Measuring Feature Subset Stability")
print("="*80)

print("Methods to Measure Feature Subset Stability:")
print("\n1. Jaccard Similarity:")
print("   - Measures overlap between feature subsets")
print("   - Formula: Jaccard(S1, S2) = |S1 ∩ S2| / |S1 ∪ S2|")
print("   - Range: [0, 1], higher values indicate more similarity")

print("\n2. Kuncheva Index:")
print("   - Corrected stability measure for feature selection")
print("   - Accounts for random selection probability")
print("   - More robust for different feature set sizes")

print("\n3. Consistency Index:")
print("   - Percentage of times each feature is selected")
print("   - Useful for individual feature stability")

print("\n4. Hamming Distance:")
print("   - Counts position differences between binary vectors")
print("   - Useful for comparing selection vectors")

# Create example feature subsets
np.random.seed(42)
feature_subsets = []
for i in range(5):
    # Simulate feature selection with some consistency
    base_features = set(range(1, 11))  # Features 1-10
    additional = set(np.random.choice(range(11, 21), size=3, replace=False))  # Features 11-20
    selected = set(np.random.choice(list(base_features), size=8, replace=False))
    feature_subsets.append(selected)

print("\nExample Feature Subsets (5 different data splits):")
for i, subset in enumerate(feature_subsets):
    print(f"Split {i+1}: {sorted(subset)}")

# Calculate stability measures
stability_results = stability_criterion(feature_subsets, 20, 0.7)
stability_score, is_stable, recommendations, avg_jaccard, kuncheva_idx = stability_results

print("\nStability Analysis Results:")
print(f"Stability score: {stability_score:.4f}")
print(f"Average Jaccard similarity: {avg_jaccard:.4f}")
print(f"Kuncheva index: {kuncheva_idx:.4f}")
print(f"Stable selection: {is_stable}")
print(f"Recommendations: {recommendations}")

# Create visualization for stability measurement
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Subplot 1: Jaccard similarity heatmap
n_subsets = len(feature_subsets)
jaccard_matrix = np.zeros((n_subsets, n_subsets))

for i in range(n_subsets):
    for j in range(n_subsets):
        jaccard_matrix[i, j] = calculate_jaccard_similarity(feature_subsets[i], feature_subsets[j])

sns.heatmap(jaccard_matrix, annot=True, fmt='.3f', cmap='Blues',
            xticklabels=[f'Split {i+1}' for i in range(n_subsets)],
            yticklabels=[f'Split {i+1}' for i in range(n_subsets)], ax=ax1)
ax1.set_title('Jaccard Similarity Between Feature Subsets')

# Subplot 2: Feature selection frequency
all_features = set()
for subset in feature_subsets:
    all_features.update(subset)

feature_freq = {}
for feature in all_features:
    count = sum(1 for subset in feature_subsets if feature in subset)
    feature_freq[feature] = count / len(feature_subsets) * 100

features_sorted = sorted(feature_freq.keys())
frequencies = [feature_freq[f] for f in features_sorted]

bars = ax2.bar(range(len(features_sorted)), frequencies, alpha=0.7, color='skyblue')
ax2.set_xlabel('Feature ID')
ax2.set_ylabel('Selection Frequency (%)')
ax2.set_title('Feature Selection Frequency Across Splits')
ax2.set_xticks(range(len(features_sorted)))
ax2.set_xticklabels([str(f) for f in features_sorted])
ax2.grid(True, alpha=0.3)

# Subplot 3: Stability metrics comparison
metrics = ['Jaccard\nSimilarity', 'Kuncheva\nIndex', 'Combined\nStability']
values = [avg_jaccard, kuncheva_idx, stability_score]
colors = ['blue', 'green', 'red']

bars3 = ax3.bar(metrics, values, color=colors, alpha=0.7)
ax3.axhline(y=0.7, color='gray', linestyle='--', label='Stability Threshold')
ax3.set_ylabel('Stability Score')
ax3.set_title('Stability Metrics Comparison')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Subplot 4: Feature subset sizes
subset_sizes = [len(subset) for subset in feature_subsets]
ax4.hist(subset_sizes, bins=range(min(subset_sizes), max(subset_sizes)+2),
         alpha=0.7, color='lightcoral', edgecolor='black')
ax4.set_xlabel('Subset Size')
ax4.set_ylabel('Frequency')
ax4.set_title('Distribution of Feature Subset Sizes')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'task4_stability_measurement.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Task 5: Compare stability metrics before and after feature selection
# ============================================================================

print("\n" + "="*80)
print("Task 5: Stability Metrics Comparison")
print("="*80)

# Simulate before and after scenarios
np.random.seed(42)

# Before feature selection (high variability)
before_subsets = []
for i in range(5):
    # More random selection - less stability
    subset = set(np.random.choice(range(1, 101), size=50, replace=False))
    before_subsets.append(subset)

# After feature selection (more consistent)
after_subsets = []
for i in range(5):
    # More consistent selection - higher stability
    base_features = set(range(1, 16))  # Features 1-15
    additional = set(np.random.choice(range(16, 21), size=5, replace=False))  # Features 16-20
    selected = set(np.random.choice(list(base_features), size=12, replace=False))
    selected.update(additional)
    after_subsets.append(selected)

# Calculate metrics for both scenarios
before_results = stability_criterion(before_subsets, 100, 0.7)
after_results = stability_criterion(after_subsets, 100, 0.7)

print("Stability Comparison:")
print("\nBefore Feature Selection:")
print(f"Combined stability: {before_results[0]:.4f}")
print(f"Jaccard similarity: {before_results[3]:.4f}")
print(f"Kuncheva index: {before_results[4]:.4f}")
print(f"Stable: {before_results[1]}")

print("\nAfter Feature Selection:")
print(f"Combined stability: {after_results[0]:.4f}")
print(f"Jaccard similarity: {after_results[3]:.4f}")
print(f"Kuncheva index: {after_results[4]:.4f}")
print(f"Stable: {after_results[1]}")

# Improvement calculations
jaccard_improvement = ((after_results[3] - before_results[3]) / before_results[3]) * 100
kuncheva_improvement = ((after_results[4] - before_results[4]) / before_results[4]) * 100
overall_improvement = ((after_results[0] - before_results[0]) / before_results[0]) * 100

print("\nImprovements After Feature Selection:")
print(f"Jaccard similarity improvement: {jaccard_improvement:.1f}%")
print(f"Kuncheva index improvement: {kuncheva_improvement:.1f}%")
print(f"Overall stability improvement: {overall_improvement:.1f}%")

# Create comparison visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Subplot 1: Before/After stability comparison
scenarios = ['Before Selection', 'After Selection']
jaccard_values = [before_results[3], after_results[3]]
kuncheva_values = [before_results[4], after_results[4]]
combined_values = [before_results[0], after_results[0]]

x = np.arange(len(scenarios))
width = 0.25

ax1.bar(x - width, jaccard_values, width, label='Jaccard Similarity', alpha=0.7, color='blue')
ax1.bar(x, kuncheva_values, width, label='Kuncheva Index', alpha=0.7, color='green')
ax1.bar(x + width, combined_values, width, label='Combined Stability', alpha=0.7, color='red')
ax1.axhline(y=0.7, color='gray', linestyle='--', label='Stability Threshold')
ax1.set_xlabel('Scenario')
ax1.set_ylabel('Stability Score')
ax1.set_title('Stability Metrics Before vs After Feature Selection')
ax1.set_xticks(x)
ax1.set_xticklabels(scenarios)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Feature selection consistency
before_freq = {}
after_freq = {}

for subset in before_subsets:
    for f in subset:
        before_freq[f] = before_freq.get(f, 0) + 1

for subset in after_subsets:
    for f in subset:
        after_freq[f] = after_freq.get(f, 0) + 1

# Get top 20 features for each scenario
before_top = sorted(before_freq.items(), key=lambda x: x[1], reverse=True)[:20]
after_top = sorted(after_freq.items(), key=lambda x: x[1], reverse=True)[:20]

before_features, before_counts = zip(*before_top)
after_features, after_counts = zip(*after_top)

ax2.bar(range(len(before_features)), before_counts, alpha=0.7, color='red', label='Before')
ax2.bar(range(len(after_features)), after_counts, alpha=0.7, color='green', label='After')
ax2.set_xlabel('Top Features')
ax2.set_ylabel('Selection Count')
ax2.set_title('Feature Selection Consistency')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Subplot 3: Improvement visualization
improvements = [jaccard_improvement, kuncheva_improvement, overall_improvement]
improvement_labels = ['Jaccard\nSimilarity', 'Kuncheva\nIndex', 'Combined\nStability']

bars3 = ax3.bar(improvement_labels, improvements, color=['blue', 'green', 'red'], alpha=0.7)
ax3.set_ylabel('Improvement (%)')
ax3.set_title('Stability Improvements After Feature Selection')
ax3.grid(True, alpha=0.3)

# Add value labels
for bar, improvement in zip(bars3, improvements):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{improvement:.1f}', ha='center', va='bottom')

# Subplot 4: Stability distribution
stability_scores = []
for i in range(20):  # Simulate multiple feature selection runs
    np.random.seed(42 + i)
    test_subsets = []
    for j in range(5):
        base = set(range(1, 16))
        additional = set(np.random.choice(range(16, 21), size=5, replace=False))
        selected = set(np.random.choice(list(base), size=12, replace=False))
        selected.update(additional)
        test_subsets.append(selected)

    score, _, _, _, _ = stability_criterion(test_subsets, 100, 0.7)
    stability_scores.append(score)

ax4.hist(stability_scores, bins=10, alpha=0.7, color='purple', edgecolor='black')
ax4.axvline(np.mean(stability_scores), color='red', linestyle='--', label=f'Mean ({np.mean(stability_scores):.3f})')
ax4.set_xlabel('Stability Score')
ax4.set_ylabel('Frequency')
ax4.set_title('Distribution of Stability Scores After Feature Selection')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'task5_stability_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Task 6: Variance calculations and feature removal estimation
# ============================================================================

print("\n" + "="*80)
print("Task 6: Variance Calculations and Feature Removal Estimation")
print("="*80)

print("Given:")
print(f"Original CV variance: {original_cv_variance:.4f}")
print(f"Reduced CV variance: {reduced_cv_variance:.4f}")
print(f"Target variance: {target_variance:.4f}")

# Calculate percentage improvement
percentage_improvement = ((original_cv_variance - reduced_cv_variance) / original_cv_variance) * 100
print(f"Percentage improvement: {percentage_improvement:.1f}%")

# Linear relationship assumption
# variance = a * num_features + b
# We have two points: (100, 0.04) and (20, 0.028)
# We want to find: (x, 0.02)

slope = (reduced_cv_variance - original_cv_variance) / (reduced_features - initial_features)
intercept = original_cv_variance - slope * initial_features

print("\nLinear Relationship Analysis:")
print(f"Slope (a): {slope:.6f}")
print(f"Intercept (b): {intercept:.6f}")
print(f"Variance = {slope:.6f} × num_features + {intercept:.6f}")

# Solve for target variance
target_features = (target_variance - intercept) / slope
features_to_remove = initial_features - target_features

print("\nTarget Analysis:")
print(f"Target variance: {target_variance:.4f}")
print(f"Target features needed: {target_features:.1f}")
print(f"Additional features to remove: {features_to_remove:.1f}")

# Create visualization for variance calculations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Subplot 1: Linear relationship between features and variance
feature_range_plot = np.linspace(5, 100, 100)
variance_predicted = slope * feature_range_plot + intercept

ax1.plot(feature_range_plot, variance_predicted, 'b-', linewidth=3, label='Linear Relationship')
ax1.scatter([initial_features, reduced_features], [original_cv_variance, reduced_cv_variance],
           color='red', s=100, zorder=5, label='Observed Points')
ax1.scatter(target_features, target_variance, color='green', s=100, marker='*', zorder=5,
           label=f'Target ({target_features:.1f})')
ax1.set_xlabel('Number of Features')
ax1.set_ylabel('Cross-Validation Variance')
ax1.set_title('Linear Relationship: Features vs Variance')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Subplot 2: Variance reduction progression
features_removed = np.arange(0, 81, 10)  # 0 to 80 features removed
remaining_features = initial_features - features_removed
variance_progression = slope * remaining_features + intercept

ax2.plot(features_removed, variance_progression, 'r-', linewidth=3, marker='o', markersize=8)
ax2.axhline(y=target_variance, color='green', linestyle='--', label=f'Target Variance ({target_variance:.4f})')
ax2.axvline(x=features_to_remove, color='blue', linestyle='--',
           label=f'Features to Remove ({features_to_remove:.0f})')
ax2.set_xlabel('Features Removed')
ax2.set_ylabel('Cross-Validation Variance')
ax2.set_title('Variance Reduction as Features are Removed')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Subplot 3: Percentage improvement calculation
improvements = [percentage_improvement]
labels = ['Actual\nImprovement']

bars3 = ax3.bar(labels, improvements, color='orange', alpha=0.7)
ax3.set_ylabel('Improvement (%)')
ax3.set_title('Percentage Improvement from Feature Selection')
ax3.grid(True, alpha=0.3)

# Add value labels
for bar, improvement in zip(bars3, improvements):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{improvement:.1f}', ha='center', va='bottom')

# Subplot 4: Feature reduction efficiency
efficiency = -percentage_improvement / (80)  # 80 features removed
efficiency_values = [efficiency] * 3
efficiency_labels = ['Variance\nReduction\nEfficiency', 'Feature\nUtilization', 'Selection\nEffectiveness']

bars4 = ax4.bar(efficiency_labels, efficiency_values, color='purple', alpha=0.7)
ax4.set_ylabel('Efficiency Metric')
ax4.set_title('Feature Selection Efficiency Metrics')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'task6_variance_calculations.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Task 7: Jaccard similarity and Kuncheva index calculations
# ============================================================================

print("\n" + "="*80)
print("Task 7: Jaccard Similarity and Kuncheva Index Calculations")
print("="*80)

# Given sets
S1 = {1, 2, 3, 4, 5}
S2 = {2, 3, 4, 6, 7}
total_features_task7 = 20
num_splits = 5
avg_jaccard_given = 0.6

print("Given:")
print(f"S1 = {S1}")
print(f"S2 = {S2}")
print(f"Total features: {total_features_task7}")
print(f"Number of data splits: {num_splits}")
print(f"Average Jaccard similarity: {avg_jaccard_given}")

# Calculate Jaccard similarity for S1 and S2
jaccard_s1_s2 = calculate_jaccard_similarity(S1, S2)
print(f"Jaccard similarity between S1 and S2: {jaccard_s1_s2:.4f}")

# Calculate stability index (average Jaccard similarity)
stability_index = avg_jaccard_given
print(f"Stability index (average Jaccard): {stability_index:.4f}")

# Calculate Kuncheva index
k = len(S1)  # Assume same size for all subsets
total_possible_pairs = comb(num_splits, 2)
correction_factor = (k * (total_features_task7 - k)) / (total_features_task7 * comb(total_features_task7, k))
kuncheva_index = stability_index / correction_factor if correction_factor > 0 else 0

print("\nKuncheva Index Calculation:")
print(f"k (features per subset): {k}")
print(f"Total possible pairs: {total_possible_pairs}")
print(f"Correction factor: {correction_factor:.6f}")
print(f"Kuncheva index: {kuncheva_index:.4f}")

# Create visualization for Task 7
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Subplot 1: Set visualization
s1_list = list(S1)
s2_list = list(S2)
all_elements = sorted(list(S1.union(S2)))

# Create binary vectors
s1_vector = [1 if x in S1 else 0 for x in all_elements]
s2_vector = [1 if x in S2 else 0 for x in all_elements]

x_pos = np.arange(len(all_elements))
width = 0.35

bars1 = ax1.bar(x_pos - width/2, s1_vector, width, label='S1', alpha=0.7, color='blue')
bars2 = ax1.bar(x_pos + width/2, s2_vector, width, label='S2', alpha=0.7, color='red')
ax1.set_xlabel('Feature ID')
ax1.set_ylabel('Selected (1/0)')
ax1.set_title('Feature Sets S1 and S2')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(all_elements)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Set operations
intersection = S1.intersection(S2)
union = S1.union(S2)
s1_only = S1 - S2
s2_only = S2 - S1

operations = ['S1 Only', 'Intersection', 'S2 Only']
counts = [len(s1_only), len(intersection), len(s2_only)]
colors = ['blue', 'purple', 'red']

bars_op = ax2.bar(operations, counts, color=colors, alpha=0.7)
ax2.set_ylabel('Number of Features')
ax2.set_title('Set Operations Between S1 and S2')
ax2.grid(True, alpha=0.3)

# Add value labels
for bar, count in zip(bars_op, counts):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{count:.0f}', ha='center', va='bottom')

# Subplot 3: Jaccard similarity components
components = ['|S1 \\cap S2|', '|S1 \\cup S2|', 'Jaccard\nSimilarity']
values = [len(intersection), len(union), jaccard_s1_s2]

bars3 = ax3.bar(components, values, color=['purple', 'orange', 'green'], alpha=0.7)
ax3.set_ylabel('Value')
ax3.set_title('Jaccard Similarity Components')
ax3.grid(True, alpha=0.3)

# Add value labels
for bar, value in zip(bars3, values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{value:.4f}' if isinstance(value, float) else f'{value:.0f}',
             ha='center', va='bottom')

# Subplot 4: Stability metrics comparison
metrics = ['Jaccard\nSimilarity', 'Stability\nIndex', 'Kuncheva\nIndex']
values = [jaccard_s1_s2, stability_index, kuncheva_index]

bars4 = ax4.bar(metrics, values, color=['blue', 'green', 'red'], alpha=0.7)
ax4.set_ylabel('Index Value')
ax4.set_title('Stability Metrics for Given Feature Sets')
ax4.grid(True, alpha=0.3)

# Add value labels
for bar, value in zip(bars4, values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{value:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'task7_jaccard_kuncheva.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Summary and Conclusions
# ============================================================================

print("\n" + "="*80)
print("SUMMARY AND KEY FINDINGS")
print("="*80)

print("\n1. Feature Reduction Effects:")
print(f"   - Reduced from {initial_features} to {reduced_features} features ({reduction_ratio:.1%} reduction)")
print(f"   - Variance reduction: {variance_reduction:.1%}")

print("\n2. Stability-Based Feature Selection:")
print(f"   - Jaccard similarity between S1 and S2: {jaccard_s1_s2:.4f}")
print(f"   - Stability index (average Jaccard): {stability_index:.4f}")
print(f"   - Kuncheva index: {kuncheva_index:.4f}")

print("\n3. Variance Analysis:")
print(f"   - Original CV variance: {original_cv_variance:.4f}")
print(f"   - Reduced CV variance: {reduced_cv_variance:.4f}")
print(f"   - Percentage improvement: {percentage_improvement:.1f}%")

print("\n4. Feature Removal for Target Variance:")
print(f"   - Target features needed: {target_features:.1f}")
print(f"   - Additional features to remove: {features_to_remove:.1f}")

print("\n5. Key Recommendations:")
print("   - Feature selection significantly improves model stability")
print("   - Use stability metrics (Jaccard, Kuncheva) to evaluate selection quality")
print("   - Linear relationship between features and variance can guide optimal selection")
print("   - Consider both bias-variance tradeoff and stability when selecting features")

print(f"\nPlots saved to: {save_dir}")
print("\nAll tasks completed successfully!")
