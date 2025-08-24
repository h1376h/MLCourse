import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
from matplotlib.patches import Rectangle
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_2_Quiz_19")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("Question 19: Feature Selection and Domain Insights")
print("=" * 60)

# 1. How can feature selection help understand data relationships?
print("\n1. How can feature selection help understand data relationships?")
print("-" * 60)

print("Feature selection can reveal data relationships through:")
print("• Correlation analysis between selected features")
print("• Identification of redundant or highly correlated features")
print("• Discovery of feature importance rankings")
print("• Understanding of feature interactions")
print("• Revealing underlying data structure and patterns")

# 2. Binomial distribution calculation
print("\n2. Binomial Distribution Analysis")
print("-" * 60)

# Given parameters
n_features = 100  # Total number of features
n_selected = 80   # Number of features selected from one category
n_categories = 5  # Total number of domain categories
alpha = 0.05      # Significance level

print(f"Given:")
print(f"• Total features: {n_features}")
print(f"• Features selected from one category: {n_selected}")
print(f"• Total domain categories: {n_categories}")
print(f"• Significance level: α = {alpha}")

# Calculate probability of selection for each category under random distribution
p_random = 1/n_categories
print(f"\nUnder random distribution:")
print(f"• Probability of selecting from any category: p = 1/{n_categories} = {p_random}")

# Calculate expected number of features from one category under random distribution
expected_random = n_selected * p_random
print(f"• Expected features from one category: E[X] = {n_selected} × {p_random} = {expected_random}")

# Calculate observed proportion
observed_proportion = n_selected / n_selected  # 80/80 = 1.0
print(f"• Observed proportion from one category: {n_selected}/{n_selected} = {observed_proportion:.1%}")

# Calculate probability of getting 80 or more features from one category by chance
# Using binomial distribution: P(X ≥ 80) where X ~ Bin(n=80, p=0.2)
k_values = np.arange(n_selected, n_selected + 1)  # Just 80 in this case
binomial_pmf = stats.binom.pmf(k_values, n_selected, p_random)
cumulative_prob = 1 - stats.binom.cdf(n_selected - 1, n_selected, p_random)

print(f"\nBinomial Distribution Analysis:")
print(f"• X ~ Bin(n={n_selected}, p={p_random})")
print(f"• P(X ≥ {n_selected}) = {cumulative_prob:.10f}")

# Check statistical significance
is_significant = cumulative_prob < alpha
print(f"• Is this clustering statistically significant at α = {alpha}? {is_significant}")
print(f"• p-value = {cumulative_prob:.10f} {'<' if is_significant else '>'} {alpha}")

# 3. Visualize the binomial distribution
plt.figure(figsize=(12, 8))

# Create subplots
plt.subplot(2, 2, 1)
k_range = np.arange(0, n_selected + 1)
pmf_values = stats.binom.pmf(k_range, n_selected, p_random)
cdf_values = stats.binom.cdf(k_range, n_selected, p_random)

# Plot PMF
plt.bar(k_range, pmf_values, alpha=0.7, color='skyblue', edgecolor='navy')
plt.axvline(x=n_selected, color='red', linestyle='--', linewidth=2, label=f'Observed: {n_selected}')
plt.axvline(x=expected_random, color='green', linestyle='--', linewidth=2, label=f'Expected: {expected_random}')
plt.xlabel('Number of Features from One Category')
plt.ylabel('Probability')
plt.title('Binomial Distribution PMF\n$X \\sim \\mathrm{Bin}(80, 0.2)$')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot CDF
plt.subplot(2, 2, 2)
plt.plot(k_range, cdf_values, 'b-', linewidth=2)
plt.axvline(x=n_selected, color='red', linestyle='--', linewidth=2, label=f'Observed: {n_selected}')
plt.axhline(y=1-cumulative_prob, color='orange', linestyle='--', alpha=0.7, label=f'P(X < {n_selected})')
plt.xlabel('Number of Features from One Category')
plt.ylabel('Cumulative Probability')
plt.title('Binomial Distribution CDF')
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Feature category distribution visualization
plt.subplot(2, 2, 3)
categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
random_distribution = [n_selected * p_random] * n_categories
observed_distribution = [n_selected, 0, 0, 0, 0]  # All 80 from Category A

x = np.arange(len(categories))
width = 0.35

plt.bar(x - width/2, random_distribution, width, label='Random Distribution', alpha=0.7, color='lightcoral')
plt.bar(x + width/2, observed_distribution, width, label='Observed Distribution', alpha=0.7, color='lightblue')

plt.xlabel('Domain Categories')
plt.ylabel('Number of Selected Features')
plt.title('Feature Distribution Across Categories')
plt.xticks(x, categories)
plt.legend()
plt.grid(True, alpha=0.3)

# 5. Statistical significance visualization
plt.subplot(2, 2, 4)
# Create a more detailed view around the observed value
k_detailed = np.arange(60, n_selected + 1)
pmf_detailed = stats.binom.pmf(k_detailed, n_selected, p_random)

plt.bar(k_detailed, pmf_detailed, alpha=0.7, color='lightcoral', edgecolor='darkred')
plt.axvline(x=n_selected, color='red', linestyle='--', linewidth=2, label=f'Observed: {n_selected}')
plt.axvline(x=expected_random, color='green', linestyle='--', linewidth=2, label=f'Expected: {expected_random}')

# Shade the critical region
critical_region = k_detailed[k_detailed >= n_selected]
if len(critical_region) > 0:
    critical_pmf = stats.binom.pmf(critical_region, n_selected, p_random)
    plt.bar(critical_region, critical_pmf, alpha=0.9, color='red', edgecolor='darkred', label='Critical Region')

plt.xlabel('Number of Features from One Category')
plt.ylabel('Probability')
plt.title('Critical Region for Statistical Test')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'binomial_analysis.png'), dpi=300, bbox_inches='tight')

# 6. Feature importance and correlation visualization
plt.figure(figsize=(15, 10))

# Simulate feature importance scores for different categories
np.random.seed(42)
category_sizes = [80, 5, 5, 5, 5]  # 80 from Category A, 5 from each other
all_features = []
all_importances = []
all_categories = []

for i, (cat, size) in enumerate(zip(categories, category_sizes)):
    # Generate feature names
    feature_names = [f'{cat}_Feature_{j+1}' for j in range(size)]
    # Generate importance scores (Category A features have higher importance)
    if i == 0:  # Category A
        importances = np.random.normal(0.8, 0.1, size)
    else:  # Other categories
        importances = np.random.normal(0.3, 0.2, size)
    
    all_features.extend(feature_names)
    all_importances.extend(importances)
    all_categories.extend([cat] * size)

# Create feature importance plot
plt.subplot(2, 3, 1)
colors = ['red', 'blue', 'green', 'orange', 'purple']
for i, cat in enumerate(categories):
    mask = [c == cat for c in all_categories]
    cat_importances = [imp for j, imp in enumerate(all_importances) if mask[j]]
    plt.hist(cat_importances, alpha=0.7, label=cat, color=colors[i], bins=20)

plt.xlabel('Feature Importance Score')
plt.ylabel('Frequency')
plt.title('Feature Importance Distribution by Category')
plt.legend()
plt.grid(True, alpha=0.3)

# Feature correlation heatmap (simulated)
plt.subplot(2, 3, 2)
# Create a correlation matrix for selected features
n_selected_features = 20  # Show top 20 features
top_indices = np.argsort(all_importances)[-n_selected_features:]
top_features = [all_features[i] for i in top_indices]
top_categories = [all_categories[i] for i in top_indices]

# Simulate correlation matrix
np.random.seed(42)
corr_matrix = np.random.uniform(-0.3, 0.8, (n_selected_features, n_selected_features))
np.fill_diagonal(corr_matrix, 1.0)  # Diagonal should be 1
corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric

# Create heatmap
sns.heatmap(corr_matrix, xticklabels=top_features, yticklabels=top_features, 
            cmap='RdBu_r', center=0, annot=False, cbar_kws={'label': 'Correlation'})
plt.title('Feature Correlation Matrix\n(Top 20 Features)')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Category distribution pie chart
plt.subplot(2, 3, 3)
category_counts = [category_sizes[0], sum(category_sizes[1:])]
category_labels = ['Category A', 'Other Categories']
colors_pie = ['red', 'lightgray']
plt.pie(category_counts, labels=category_labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Selected Features')

# Feature importance by category (box plot)
plt.subplot(2, 3, 4)
category_importances = []
category_names = []
for cat in categories:
    mask = [c == cat for c in all_categories]
    cat_imp = [imp for j, imp in enumerate(all_importances) if mask[j]]
    if cat_imp:
        category_importances.append(cat_imp)
        category_names.append(cat)

plt.boxplot(category_importances, labels=category_names)
plt.ylabel('Feature Importance Score')
plt.title('Feature Importance by Category')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Cumulative feature importance
plt.subplot(2, 3, 5)
sorted_importances = np.sort(all_importances)[::-1]
cumulative_importance = np.cumsum(sorted_importances)
cumulative_percentage = cumulative_importance / cumulative_importance[-1] * 100

plt.plot(range(1, len(sorted_importances) + 1), cumulative_percentage, 'b-', linewidth=2)
plt.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% threshold')
plt.xlabel('Number of Features')
plt.ylabel('Cumulative Importance (%)')
plt.title('Cumulative Feature Importance')
plt.legend()
plt.grid(True, alpha=0.3)

# Feature selection timeline (simulated)
plt.subplot(2, 3, 6)
# Simulate feature selection process over iterations
iterations = np.arange(1, 21)
features_selected = np.cumsum([4, 3, 5, 2, 6, 4, 3, 5, 2, 6, 4, 3, 5, 2, 6, 4, 3, 5, 2, 6])
category_a_selected = np.cumsum([3, 2, 4, 1, 5, 3, 2, 4, 1, 5, 3, 2, 4, 1, 5, 3, 2, 4, 1, 5])

plt.plot(iterations, features_selected, 'b-', linewidth=2, label='Total Features Selected')
plt.plot(iterations, category_a_selected, 'r-', linewidth=2, label='Category A Features Selected')
plt.xlabel('Iteration')
plt.ylabel('Number of Features')
plt.title('Feature Selection Progress')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_analysis.png'), dpi=300, bbox_inches='tight')

# 7. Summary statistics
print("\n3. Insights from Consistently Selected Features")
print("-" * 60)

print("Key insights from the analysis:")
print(f"• Category A dominates with {n_selected} out of {n_selected} selected features ({observed_proportion:.1%})")
print(f"• This clustering is {'statistically significant' if is_significant else 'not statistically significant'} (p = {cumulative_prob:.10f})")
print(f"• Expected random distribution: {expected_random:.1f} features per category")
print(f"• Observed distribution: {n_selected} features from Category A")

print("\n4. Feature Engineering Decisions")
print("-" * 60)

print("Feature selection helps with feature engineering by:")
print("• Identifying which feature categories are most informative")
print("• Revealing redundant features that can be combined or removed")
print("• Suggesting new features based on selected feature patterns")
print("• Optimizing feature extraction from the most relevant domains")
print("• Guiding resource allocation for feature development")

# 8. Additional statistical analysis
print("\nAdditional Statistical Analysis")
print("-" * 60)

# Calculate confidence interval for the proportion (approximate)
# Using normal approximation for large n
z_alpha = stats.norm.ppf(1 - alpha/2)
margin_of_error = z_alpha * np.sqrt(p_random * (1 - p_random) / n_selected)
confidence_interval = (p_random - margin_of_error, p_random + margin_of_error)
print(f"• 95% Confidence Interval for proportion: [{confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}]")

# Calculate effect size (Cohen's h)
p_observed = n_selected / n_selected  # 1.0
p_expected = p_random  # 0.2
cohens_h = 2 * (np.arcsin(np.sqrt(p_observed)) - np.arcsin(np.sqrt(p_expected)))
print(f"• Effect size (Cohen's h): {cohens_h:.3f}")

# Interpret effect size
if abs(cohens_h) < 0.2:
    effect_size_desc = "small"
elif abs(cohens_h) < 0.5:
    effect_size_desc = "small to medium"
elif abs(cohens_h) < 0.8:
    effect_size_desc = "medium to large"
else:
    effect_size_desc = "large"

print(f"• Effect size interpretation: {effect_size_desc}")

print(f"\nAll plots saved to: {save_dir}")
print("=" * 60)
