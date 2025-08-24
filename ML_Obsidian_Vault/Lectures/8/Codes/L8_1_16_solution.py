import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_1_Quiz_16")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("Question 16: Statistical Significance Testing in Feature Selection")
print("=" * 70)

# Generate synthetic dataset for demonstration
np.random.seed(42)
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                          n_redundant=5, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# 1. What is statistical significance in feature selection?
print("\n1. Statistical Significance in Feature Selection")
print("-" * 50)
print("Statistical significance helps determine if feature selection results are reliable")
print("and not due to random chance. It quantifies the probability that observed")
print("improvements in model performance occurred by random variation rather than")
print("true feature importance.")

# 2. How do you test if a selected feature is truly important?
print("\n2. Testing Feature Importance")
print("-" * 30)

# Train model with all features
rf_all = RandomForestClassifier(n_estimators=100, random_state=42)
rf_all.fit(X_train, y_train)
y_pred_all = rf_all.predict(X_test)
accuracy_all = accuracy_score(y_test, y_pred_all)

# Train model without a specific feature (feature 0)
feature_to_remove = 0
X_train_reduced = np.delete(X_train, feature_to_remove, axis=1)
X_test_reduced = np.delete(X_test, feature_to_remove, axis=1)

rf_reduced = RandomForestClassifier(n_estimators=100, random_state=42)
rf_reduced.fit(X_train_reduced, y_train)
y_pred_reduced = rf_reduced.predict(X_test_reduced)
accuracy_reduced = accuracy_score(y_test, y_pred_reduced)

improvement = accuracy_all - accuracy_reduced
print(f"Accuracy with all features: {accuracy_all:.4f}")
print(f"Accuracy without feature {feature_to_remove}: {accuracy_reduced:.4f}")
print(f"Improvement: {improvement:.4f} ({improvement*100:.2f}%)")

# 3. Role of p-values in feature selection
print("\n3. Role of P-values in Feature Selection")
print("-" * 40)
print("P-value: Probability of observing the data (or more extreme) under the null hypothesis")
print("Null hypothesis: The feature has no effect on model performance")
print("Small p-value (< α) suggests the feature is truly important")

# 4. Testing significance of 0.5% improvement
print("\n4. Testing Significance of 0.5% Improvement")
print("-" * 45)

# Simulate multiple runs to get distribution
n_runs = 100
accuracies_with_feature = []
accuracies_without_feature = []

for _ in range(n_runs):
    # Bootstrap sampling
    indices = np.random.choice(len(X_train), len(X_train), replace=True)
    X_boot, y_boot = X_train[indices], y_train[indices]
    
    # With feature
    rf_with = RandomForestClassifier(n_estimators=50, random_state=None)
    rf_with.fit(X_boot, y_boot)
    acc_with = accuracy_score(y_test, rf_with.predict(X_test))
    accuracies_with_feature.append(acc_with)
    
    # Without feature
    X_boot_reduced = np.delete(X_boot, feature_to_remove, axis=1)
    rf_without = RandomForestClassifier(n_estimators=50, random_state=None)
    rf_without.fit(X_boot_reduced, y_boot)
    acc_without = accuracy_score(y_test, rf_without.predict(X_test_reduced))
    accuracies_without_feature.append(acc_without)

accuracies_with_feature = np.array(accuracies_with_feature)
accuracies_without_feature = np.array(accuracies_without_feature)
improvements = accuracies_with_feature - accuracies_without_feature

print(f"Mean improvement: {np.mean(improvements):.4f} ({np.mean(improvements)*100:.2f}%)")
print(f"Standard deviation: {np.std(improvements):.4f}")

# 5. Compare different significance testing approaches
print("\n5. Comparing Significance Testing Approaches")
print("-" * 50)

# T-test
t_stat, p_value_t = stats.ttest_rel(accuracies_with_feature, accuracies_without_feature)
print(f"T-test:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value_t:.6f}")
print(f"  Significant at α=0.05: {'Yes' if p_value_t < 0.05 else 'No'}")

# Permutation test
print(f"\nPermutation test:")
n_permutations = 1000
permutation_improvements = []

for _ in range(n_permutations):
    # Randomly shuffle the feature values
    X_permuted = X_train.copy()
    np.random.shuffle(X_permuted[:, feature_to_remove])
    
    # Train with permuted feature
    rf_perm = RandomForestClassifier(n_estimators=50, random_state=None)
    rf_perm.fit(X_permuted, y_train)
    acc_perm = accuracy_score(y_test, rf_perm.predict(X_test))
    
    # Compare with reduced model
    permutation_improvement = acc_perm - accuracy_reduced
    permutation_improvements.append(permutation_improvement)

permutation_improvements = np.array(permutation_improvements)
observed_improvement = np.mean(improvements)

# Count permutations with improvement >= observed
count_extreme = np.sum(permutation_improvements >= observed_improvement)
p_value_perm = count_extreme / n_permutations

print(f"  Observed improvement: {observed_improvement:.4f}")
print(f"  Extreme permutations: {count_extreme}/{n_permutations}")
print(f"  p-value: {p_value_perm:.6f}")
print(f"  Significant at α=0.05: {'Yes' if p_value_perm < 0.05 else 'No'}")

# Bootstrap confidence interval
print(f"\nBootstrap confidence interval:")
bootstrap_means = []
for _ in range(1000):
    indices = np.random.choice(len(improvements), len(improvements), replace=True)
    bootstrap_means.append(np.mean(improvements[indices]))

bootstrap_means = np.array(bootstrap_means)
ci_lower = np.percentile(bootstrap_means, 2.5)
ci_upper = np.percentile(bootstrap_means, 97.5)

print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"  Significant (CI doesn't contain 0): {'Yes' if ci_lower > 0 else 'No'}")

# 6. Specific permutation test example
print("\n6. Specific Permutation Test Example")
print("-" * 40)
print("Given: Original accuracy improvement = 0.05")
print("      1000 permutations")
print("      25 permutations give improvement ≥ 0.05")

original_improvement = 0.05
n_permutations_given = 1000
extreme_count_given = 25

p_value_given = extreme_count_given / n_permutations_given
print(f"P-value = {extreme_count_given}/{n_permutations_given} = {p_value_given:.4f}")

alpha_05 = 0.05
alpha_01 = 0.01

print(f"Significant at α = 0.05: {'Yes' if p_value_given < alpha_05 else 'No'}")
print(f"Significant at α = 0.01: {'Yes' if p_value_given < alpha_01 else 'No'}")

# Create visualizations
print("\nGenerating visualizations...")

# Plot 1: Distribution of improvements
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.hist(improvements, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(observed_improvement, color='red', linestyle='--', linewidth=2, 
            label=f'Observed: {observed_improvement:.4f}')
plt.axvline(0, color='green', linestyle='-', linewidth=2, label='No improvement')
plt.xlabel('Accuracy Improvement')
plt.ylabel('Frequency')
plt.title('Distribution of Accuracy Improvements')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: T-test visualization
plt.subplot(2, 3, 2)
plt.hist(accuracies_with_feature, bins=20, alpha=0.7, color='lightgreen', 
         label='With feature', edgecolor='black')
plt.hist(accuracies_without_feature, bins=20, alpha=0.7, color='lightcoral', 
         label='Without feature', edgecolor='black')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.title('T-test: Accuracy Distributions')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Permutation test
plt.subplot(2, 3, 3)
plt.hist(permutation_improvements, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
plt.axvline(observed_improvement, color='red', linestyle='--', linewidth=2, 
            label=f'Observed: {observed_improvement:.4f}')
plt.axvline(0, color='green', linestyle='-', linewidth=2, label='No improvement')
plt.xlabel('Permutation Improvements')
plt.ylabel('Frequency')
plt.title('Permutation Test Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Bootstrap confidence interval
plt.subplot(2, 3, 4)
plt.hist(bootstrap_means, bins=30, alpha=0.7, color='lightyellow', edgecolor='black')
plt.axvline(observed_improvement, color='red', linestyle='--', linewidth=2, 
            label=f'Observed: {observed_improvement:.4f}')
plt.axvline(ci_lower, color='orange', linestyle=':', linewidth=2, 
            label=f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
plt.axvline(ci_upper, color='orange', linestyle=':', linewidth=2)
plt.axvline(0, color='green', linestyle='-', linewidth=2, label='No improvement')
plt.xlabel('Bootstrap Mean Improvements')
plt.ylabel('Frequency')
plt.title('Bootstrap Confidence Interval')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 5: P-value comparison
plt.subplot(2, 3, 5)
methods = ['T-test', 'Permutation', 'Bootstrap']
p_values = [p_value_t, p_value_perm, 1 - (ci_lower > 0)]  # Bootstrap significance
colors = ['skyblue', 'lightgreen', 'lightcoral']

bars = plt.bar(methods, p_values, color=colors, alpha=0.7, edgecolor='black')
plt.axhline(y=0.05, color='red', linestyle='--', label='$\\alpha = 0.05$')
plt.axhline(y=0.01, color='darkred', linestyle='--', label='$\\alpha = 0.01$')
plt.ylabel('P-value')
plt.title('P-values Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar, p_val in zip(bars, p_values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'{p_val:.4f}', ha='center', va='bottom')

# Plot 6: Feature importance comparison
plt.subplot(2, 3, 6)
feature_importances = rf_all.feature_importances_
feature_indices = np.argsort(feature_importances)[::-1]

plt.bar(range(len(feature_importances)), feature_importances[feature_indices], 
        color='lightblue', alpha=0.7, edgecolor='black')
plt.axhline(y=feature_importances[feature_to_remove], color='red', linestyle='--', 
            linewidth=2, label=f'Removed feature: {feature_importances[feature_to_remove]:.4f}')
plt.xlabel('Feature Index (sorted by importance)')
plt.ylabel('Feature Importance')
plt.title('Feature Importance Ranking')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'statistical_significance_analysis.png'), 
            dpi=300, bbox_inches='tight')

# Create detailed comparison table
print("\nDetailed Comparison of Significance Testing Methods:")
print("-" * 60)
print(f"{'Method':<15} {'P-value':<12} {'Significant (α=0.05)':<20} {'Significant (α=0.01)':<20}")
print("-" * 60)
print(f"{'T-test':<15} {p_value_t:<12.6f} {'Yes' if p_value_t < 0.05 else 'No':<20} {'Yes' if p_value_t < 0.01 else 'No':<20}")
print(f"{'Permutation':<15} {p_value_perm:<12.6f} {'Yes' if p_value_perm < 0.05 else 'No':<20} {'Yes' if p_value_perm < 0.01 else 'No':<20}")
print(f"{'Bootstrap':<15} {'N/A':<12} {'Yes' if ci_lower > 0 else 'No':<20} {'Yes' if ci_lower > 0 else 'No':<20}")

print(f"\nBootstrap 95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"Observed improvement: {observed_improvement:.4f}")

print(f"\nPlots saved to: {save_dir}")
print("\nAnalysis complete!")
