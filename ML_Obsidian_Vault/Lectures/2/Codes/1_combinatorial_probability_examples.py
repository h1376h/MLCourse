import matplotlib.pyplot as plt
import numpy as np
import os
from math import factorial
from scipy.stats import binom
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

print("\n=== COMBINATORIAL PROBABILITY EXAMPLES: STEP-BY-STEP SOLUTIONS ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the parent directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Helper functions for combinatorial calculations
def permutation(n, k):
    """Calculate P(n,k) - number of ways to arrange k items from n distinct items."""
    return factorial(n) // factorial(n - k)

def combination(n, k):
    """Calculate C(n,k) - number of ways to select k items from n distinct items."""
    return factorial(n) // (factorial(k) * factorial(n - k))

def multinomial(n, *k_tuple):
    """Calculate multinomial coefficient - number of ways to partition n items into groups of sizes k1, k2, ..., kj."""
    denominator = 1
    for k in k_tuple:
        denominator *= factorial(k)
    return factorial(n) // denominator

# Example 1: Basic Feature Selection
print("Example 1: Basic Feature Selection")
total_features = 15
features_to_select = 5
important_features = 3

# Calculate total number of possible feature subsets
total_subsets = combination(total_features, features_to_select)
print(f"Total features: {total_features}")
print(f"Features to select: {features_to_select}")
print(f"Number of important features: {important_features}")

print("\nStep 1: Calculate the total number of possible feature subsets")
print(f"C({total_features},{features_to_select}) = {total_features}!/(({features_to_select}!)*({total_features-features_to_select})!)")
print(f"  = {total_features}!/({features_to_select}!*{total_features-features_to_select}!)")

# Detailed calculation for C(15,5)
num_str = " × ".join(str(i) for i in range(total_features, total_features-features_to_select, -1))
den_str = " × ".join(str(i) for i in range(1, features_to_select+1))
print(f"  = ({num_str})/({den_str})")
print(f"  = {total_subsets}")

print("\nStep 2: Calculate the number of subsets containing all important features")
# Calculate number of favorable subsets
favorable_subsets = combination(total_features - important_features, features_to_select - important_features)
print(f"If all {important_features} important features must be included, we need to select {features_to_select - important_features} more features from the remaining {total_features - important_features} features.")
print(f"C({total_features - important_features},{features_to_select - important_features}) = {favorable_subsets}")

print("\nStep 3: Calculate the probability")
probability = favorable_subsets / total_subsets
print(f"P(subset contains all important features) = {favorable_subsets}/{total_subsets} = {probability:.4f}")
print(f"Therefore, the probability is {probability:.4f} or {probability*100:.2f}%")
print("\nVisualization: feature_selection_probability.png")
print(f"- Total possible feature subsets: {total_subsets:,}")
print(f"- Subsets containing all important features: {favorable_subsets:,}")
print(f"- Probability: {probability:.4f} ({probability*100:.2f}%)")

# Create a simplified visual representation
plt.figure(figsize=(8, 5))
plt.bar(['Total Subsets', 'Favorable Subsets'], [total_subsets, favorable_subsets], color=['skyblue', 'lightgreen'])
plt.yscale('log')  # Using log scale due to large numbers
plt.ylabel('Number of Subsets (log scale)')
plt.title('Feature Selection Probability')

# Create a simplified visual representation of the feature selection problem
ax = plt.axes([0.15, 0.55, 0.7, 0.3])
ax.set_xlim(0, 15)
ax.set_ylim(0, 3)
ax.set_yticks([])
ax.set_xticks(range(total_features))
ax.set_xticklabels([f'F{i+1}' for i in range(total_features)])
ax.set_title('Features')

# Draw rectangles for each feature
for i in range(total_features):
    if i < important_features:
        color = 'red'
    else:
        color = 'gray'
    rect = Rectangle((i, 0.5), 0.8, 1, linewidth=1, edgecolor='black', facecolor=color, alpha=0.6)
    ax.add_patch(rect)

# Add legend
important_patch = plt.Rectangle((0, 0), 1, 1, fc="red", alpha=0.6)
regular_patch = plt.Rectangle((0, 0), 1, 1, fc="gray", alpha=0.6)
selected_patch = plt.Rectangle((0, 0), 1, 1, fc="green", alpha=0.6)
plt.legend([important_patch, regular_patch, selected_patch], 
           ['Important', 'Regular', 'Selected'], 
           loc='upper center', bbox_to_anchor=(0.5, 0.4))

# Highlight an example selection
example_selection = [0, 1, 2, 5, 8]  # 3 important and 2 regular features
for i in example_selection:
    rect = Rectangle((i, 0.5), 0.8, 1, linewidth=2, edgecolor='green', facecolor='none')
    ax.add_patch(rect)

plt.savefig(os.path.join(images_dir, 'feature_selection_probability.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Cross-Validation Fold Assignment
print("\n\nExample 2: Cross-Validation Fold Assignment")
total_samples = 20
num_folds = 4
samples_per_fold = total_samples // num_folds
positive_samples = 8
negative_samples = 12

print(f"Total samples: {total_samples}")
print(f"Number of folds: {num_folds}")
print(f"Samples per fold: {samples_per_fold}")
print(f"Positive samples: {positive_samples}")
print(f"Negative samples: {negative_samples}")

print("\nStep 1: Calculate the total number of possible fold assignments")
total_assignments = multinomial(total_samples, *[samples_per_fold]*num_folds)
print(f"Total number of ways to divide {total_samples} samples into {num_folds} folds of {samples_per_fold} samples each:")
print(f"  = {total_samples}!/({samples_per_fold}!)^{num_folds}")
print(f"  ≈ {total_assignments:.2e}")

print("\nStep 2: Calculate the number of fold assignments with balanced class distribution")
# For positive samples distribution
pos_distribution = multinomial(positive_samples, *[positive_samples//num_folds]*num_folds)
print(f"Ways to distribute {positive_samples} positive samples evenly ({positive_samples//num_folds} per fold):")
print(f"  = {positive_samples}!/({positive_samples//num_folds}!)^{num_folds}")
print(f"  = {pos_distribution:,}")

# For negative samples distribution
neg_distribution = multinomial(negative_samples, *[negative_samples//num_folds]*num_folds)
print(f"Ways to distribute {negative_samples} negative samples evenly ({negative_samples//num_folds} per fold):")
print(f"  = {negative_samples}!/({negative_samples//num_folds}!)^{num_folds}")
print(f"  = {neg_distribution:,}")

# Total favorable assignments
favorable_assignments = pos_distribution * neg_distribution
print(f"Total number of favorable assignments: {pos_distribution:,} × {neg_distribution:,} = {favorable_assignments:,}")

print("\nStep 3: Calculate the probability")
probability = favorable_assignments / total_assignments
print(f"P(balanced folds) = {favorable_assignments}/{total_assignments:.2e} ≈ {probability:.4f}")
print(f"Therefore, the probability is approximately {probability:.4f} or {probability*100:.2f}%")
print("\nVisualization: cross_validation_probability.png")
print(f"- Total possible fold assignments: {total_assignments:.2e}")
print(f"- Favorable balanced fold assignments: {favorable_assignments:,}")
print(f"- Probability of balanced folds: {probability:.4f} ({probability*100:.2f}%)")
print("- Each balanced fold contains 2 positive samples and 3 negative samples")

# Create a simplified visualization
plt.figure(figsize=(10, 6))

# Create a grid showing the fold structure
plt.subplot(2, 1, 1)
grid = np.zeros((4, 5))  # 4 folds, 5 samples each

# Color first 2 elements of each row as positive (class 1)
for i in range(4):
    grid[i, 0:2] = 1

plt.imshow(grid, cmap=plt.cm.coolwarm, aspect='auto')
plt.title('Balanced Cross-Validation Folds')
plt.yticks(range(4), [f'Fold {i+1}' for i in range(4)])
plt.xticks(range(5))
plt.xlabel('Sample Index')
plt.ylabel('Fold')

# Create a simplified probability chart
plt.subplot(2, 1, 2)
plt.bar(['Random Assignment', 'Balanced Folds'], 
        [1, probability], 
        color=['gray', 'green'])
plt.ylabel('Probability')
plt.title('Probability of Balanced Fold Assignment')
plt.ylim(0, 1.1)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'cross_validation_probability.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Random Forest Feature Sampling
print("\n\nExample 3: Random Forest Feature Sampling")
total_features = 12
features_per_split = 3  # sqrt(12) rounded to nearest integer
strong_features = 4

print(f"Total features: {total_features}")
print(f"Features considered per split: {features_per_split}")
print(f"Number of strong predictive features: {strong_features}")

print("\nStep 1: Calculate the total number of possible feature subsets")
total_subsets = combination(total_features, features_per_split)
print(f"C({total_features},{features_per_split}) = {total_subsets}")

print("\nStep 2: Calculate the number of subsets with at least one strong predictive feature")
# Calculate number of subsets with no strong features
subsets_no_strong = combination(total_features - strong_features, features_per_split)
print(f"Number of subsets with no strong features: C({total_features - strong_features},{features_per_split}) = {subsets_no_strong}")

# Calculate number of subsets with at least one strong feature
subsets_with_strong = total_subsets - subsets_no_strong
print(f"Number of subsets with at least one strong feature: {total_subsets} - {subsets_no_strong} = {subsets_with_strong}")

print("\nStep 3: Calculate the probability")
probability = subsets_with_strong / total_subsets
print(f"P(at least one strong feature) = {subsets_with_strong}/{total_subsets} = {probability:.4f}")
print(f"Therefore, the probability is {probability:.4f} or {probability*100:.2f}%")
print("\nVisualization: random_forest_feature_sampling.png")
print(f"- Total possible feature subsets: {total_subsets}")
print(f"- Subsets with at least one strong feature: {subsets_with_strong}")
print(f"- Probability: {probability:.4f} ({probability*100:.2f}%)")
print("- The visualization shows a pie chart of the probability and a bar representation of features")

# Create a simplified visual representation
plt.figure(figsize=(10, 6))

# Create a pie chart showing the probability
plt.subplot(2, 1, 1)
plt.pie([probability, 1-probability], 
        labels=[f'At least one\nstrong feature', 
                f'No strong\nfeatures'],
        colors=['green', 'red'],
        autopct='%1.1f%%',
        explode=(0.1, 0),
        shadow=True)
plt.axis('equal')
plt.title('Probability of Including Strong Features')

# Create a simplified visual of the feature subset selection
plt.subplot(2, 1, 2)
feature_names = [f'F{i+1}' for i in range(total_features)]
feature_strengths = ['Strong'] * strong_features + ['Weak'] * (total_features - strong_features)
feature_colors = ['darkred' if s == 'Strong' else 'lightblue' for s in feature_strengths]

# Create a bar chart
bars = plt.bar(feature_names, height=1, color=feature_colors)

# Highlight a few example selections
example_idx = [0, 5, 9]  # One strong feature, two weak features
for i, bar in enumerate(bars):
    if i in example_idx:
        bar.set_edgecolor('black')
        bar.set_linewidth(2)
        plt.text(i, 0.5, '✓', fontsize=20, ha='center', va='center')

plt.title('Feature Selection Example')
plt.ylim(0, 1.2)
plt.yticks([])

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'random_forest_feature_sampling.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 4: Ensemble Model Majority Voting
print("\n\nExample 4: Ensemble Model Majority Voting")
num_models = 7
model_accuracy = 0.8
majority_threshold = (num_models // 2) + 1  # Majority = 4 out of 7

print(f"Number of models in ensemble: {num_models}")
print(f"Individual model accuracy: {model_accuracy:.1f} or {model_accuracy*100:.0f}%")
print(f"Majority threshold: {majority_threshold} models")

print("\nStep 1: Define the problem")
print(f"For the ensemble to make a correct prediction with majority voting, at least {majority_threshold} out of {num_models} models must be correct.")

print("\nStep 2: Calculate the probability for each successful case")
ensemble_correct_prob = 0
for k in range(majority_threshold, num_models + 1):
    prob_k_correct = binom.pmf(k, num_models, model_accuracy)
    print(f"  P({k} models correct) = C({num_models},{k}) × ({model_accuracy:.1f})^{k} × ({1-model_accuracy:.1f})^{num_models-k} = {prob_k_correct:.4f}")
    ensemble_correct_prob += prob_k_correct

print("\nStep 3: Calculate the total probability")
print(f"P(ensemble correct) = sum of all individual probabilities = {ensemble_correct_prob:.4f}")
print(f"Therefore, the probability is {ensemble_correct_prob:.4f} or {ensemble_correct_prob*100:.2f}%")
print("\nVisualization: ensemble_majority_voting.png")
print(f"- Individual model accuracy: {model_accuracy*100:.0f}%")
print(f"- Ensemble model accuracy: {ensemble_correct_prob*100:.2f}%")
print(f"- Improvement: {((ensemble_correct_prob - model_accuracy) / model_accuracy * 100):.1f}%")
print("- The visualization shows the distribution of correct models and accuracy comparison")

# Create a simplified visual representation
plt.figure(figsize=(10, 6))

# Plot binomial distribution
x = np.arange(0, num_models + 1)
pmf = binom.pmf(x, num_models, model_accuracy)

plt.subplot(2, 1, 1)
bars = plt.bar(x, pmf, alpha=0.7)

# Color bars based on majority threshold
for i, bar in enumerate(bars):
    if i >= majority_threshold:
        bar.set_color('green')
    else:
        bar.set_color('red')

plt.axvline(x=majority_threshold-0.5, color='black', linestyle='--')
plt.xticks(x)
plt.xlabel('Number of Correct Models')
plt.ylabel('Probability')
plt.title('Binomial Distribution of Correct Models')

# Create a simplified accuracy comparison
plt.subplot(2, 1, 2)
plt.bar(['Individual Model', 'Ensemble Model'],
        [model_accuracy, ensemble_correct_prob],
        color=['lightblue', 'green'])
plt.ylim(0, 1.1)
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'ensemble_majority_voting.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 5: Permutation Test for Feature Importance
print("\n\nExample 5: Permutation Test for Feature Importance")
num_permutations = 50
alpha = 0.05

print(f"Number of permutations: {num_permutations}")
print(f"Significance level α: {alpha}")

print("\nStep 1: Understand the permutation test")
print("In a permutation test, if the feature has no predictive power (null hypothesis), then any permutation")
print("of the feature values should be equally likely to produce any particular model performance.")

print("\nStep 2: Define the significance criteria")
total_tests = num_permutations + 1  # Original + permutations
significant_rank = int(alpha * total_tests)
if significant_rank < 1:
    significant_rank = 1
print(f"For significance at α = {alpha}, the original feature's performance must rank in top {alpha} × {total_tests} ≈ {alpha * total_tests}")
print(f"This means it must rank {significant_rank}{'' if significant_rank == 1 else ' or ' + str(significant_rank-1)}")

print("\nStep 3: Calculate the probability")
p_significant = significant_rank / total_tests
print(f"P(significant at α = {alpha}) = {significant_rank}/{total_tests} = {p_significant:.4f}")
print(f"Therefore, the probability is {p_significant:.4f} or {p_significant*100:.2f}%")
print("\nVisualization: permutation_test_probability.png")
print(f"- Number of permutations: {num_permutations}")
print(f"- Significance level: {alpha}")
print(f"- Probability of false significance: {p_significant:.4f} ({p_significant*100:.2f}%)")
print("- Even with no real predictive power, a feature may appear significant by chance")

# Create a simplified visualization
plt.figure(figsize=(10, 6))

# Create a simplified visualization of the permutation test concept
plt.subplot(2, 1, 1)
np.random.seed(42)  # For reproducibility
original_perf = 0.75
perm_perfs = np.random.normal(0.7, 0.03, num_permutations)
all_perfs = np.append(perm_perfs, original_perf)
all_perfs.sort()

# Find where original performance ranks
original_rank = np.where(all_perfs == original_perf)[0][0]
is_significant = original_rank >= len(all_perfs) - significant_rank

# Plot the performance distribution
x = np.arange(len(all_perfs))
colors = ['lightgray'] * len(all_perfs)
if is_significant:
    colors[original_rank] = 'green'
else:
    colors[original_rank] = 'red'

plt.bar(x, all_perfs, color=colors)
plt.axhline(y=all_perfs[len(all_perfs) - significant_rank], color='black', linestyle='--')
plt.xlabel('Permutation Rank')
plt.ylabel('Model Performance')
plt.title('Permutation Test Distribution')

# Create a simplified visualization of the significance probability
plt.subplot(2, 1, 2)
plt.bar(['Significant', 'Not Significant'], 
        [p_significant, 1-p_significant], 
        color=['green', 'lightgray'])
plt.ylim(0, 1)
plt.ylabel('Probability')
plt.title('Chance of False Significance')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'permutation_test_probability.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll combinatorial probability example images created successfully.") 