import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import math
from scipy import stats
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import time

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_2_Quiz_2")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("Question 2: Univariate Filter Scoring - Step-by-Step Solution")
print("=" * 60)

# 1. Purpose of Filter Scoring
print("\n1. Purpose of Filter Scoring:")
print("- Filter scoring evaluates individual features independently")
print("- Ranks features based on their statistical relationship with the target")
print("- Fast and computationally efficient")
print("- Model-agnostic approach")
print("- Helps identify most relevant features before model training")

# 2. Filter vs Wrapper Methods
print("\n2. Filter vs Wrapper Methods:")
print("- Filter methods: Evaluate features independently, fast, model-agnostic")
print("- Wrapper methods: Use model performance, slower, model-specific")
print("- Filter: Feature-by-feature evaluation")
print("- Wrapper: Subset-by-subset evaluation")

# 3. Computational Time Comparison
print("\n3. Computational Time Comparison:")
print("Given:")
print("- Wrapper method: 5 minutes per feature subset")
print("- Filter method: 30 seconds per feature")
print("- 100 features")
print("- Evaluate subsets of size 1, 2, and 3")

# Calculate number of subsets
def calculate_subsets(n, max_size):
    total = 0
    for k in range(1, max_size + 1):
        total += int(math.comb(n, k))
    return total

n_features = 100
max_subset_size = 3
total_subsets = calculate_subsets(n_features, max_subset_size)

print(f"\nNumber of subsets to evaluate:")
print(f"- Size 1: {int(math.comb(n_features, 1))} subsets")
print(f"- Size 2: {int(math.comb(n_features, 2))} subsets")
print(f"- Size 3: {int(math.comb(n_features, 3))} subsets")
print(f"- Total: {total_subsets} subsets")

# Calculate time for each method
wrapper_time_minutes = total_subsets * 5
filter_time_minutes = n_features * 0.5  # 30 seconds = 0.5 minutes

print(f"\nTime calculations:")
print(f"- Wrapper method: {total_subsets} × 5 minutes = {wrapper_time_minutes} minutes")
print(f"- Filter method: {n_features} × 0.5 minutes = {filter_time_minutes} minutes")

time_difference = wrapper_time_minutes - filter_time_minutes
speedup_factor = wrapper_time_minutes / filter_time_minutes

print(f"\nResults:")
print(f"- Filter method is faster by {time_difference} minutes")
print(f"- Filter method is {speedup_factor:.1f}x faster than wrapper method")

# 4. False Positive Calculation
print("\n4. False Positive Calculation:")
print("Given:")
print("- Filter accuracy: 80%")
print("- Total features: 100")
print("- Truly relevant features: 20")
print("- Truly irrelevant features: 80")

# Calculate expected values
true_positives = 20 * 0.8  # 80% of truly relevant features correctly identified
false_negatives = 20 - true_positives  # Missed relevant features
false_positives = 80 * 0.2  # 20% of irrelevant features incorrectly identified as relevant
true_negatives = 80 - false_positives  # Correctly identified irrelevant features

print(f"\nExpected outcomes:")
print(f"- True Positives: {true_positives:.1f} features")
print(f"- False Negatives: {false_negatives:.1f} features")
print(f"- False Positives: {false_positives:.1f} features")
print(f"- True Negatives: {true_negatives:.1f} features")

# Create synthetic dataset for demonstration
print("\n" + "="*60)
print("DEMONSTRATION: Creating synthetic dataset and applying filter methods")
print("="*60)

# Generate synthetic dataset
np.random.seed(42)
n_samples = 1000
n_features = 20
n_informative = 8  # Only 8 features are truly relevant

X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=n_informative,
    n_redundant=5,
    n_clusters_per_class=1,
    random_state=42
)

# Create feature names
feature_names = [f'Feature_{i+1}' for i in range(n_features)]

print(f"\nDataset created:")
print(f"- Samples: {n_samples}")
print(f"- Features: {n_features}")
print(f"- Informative features: {n_informative}")
print(f"- Target classes: {len(np.unique(y))}")

# Calculate different filter scores
print("\nCalculating filter scores...")

# 1. F-statistic (ANOVA)
f_scores, f_pvalues = f_classif(X, y)

# 2. Mutual Information
mi_scores = mutual_info_classif(X, y, random_state=42)

# 3. Correlation
corr_scores = np.abs([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])

# 4. Chi-square (for categorical-like data)
# Discretize continuous features for chi-square
X_discrete = pd.cut(X.flatten(), bins=10, labels=False).reshape(X.shape)
chi2_scores, chi2_pvalues = chi2(X_discrete, y)

# Create results dataframe
results_df = pd.DataFrame({
    'Feature': feature_names,
    'F_Score': f_scores,
    'F_PValue': f_pvalues,
    'MI_Score': mi_scores,
    'Correlation': corr_scores,
    'Chi2_Score': chi2_scores,
    'Chi2_PValue': chi2_pvalues
})

# Add rankings
for metric in ['F_Score', 'MI_Score', 'Correlation', 'Chi2_Score']:
    results_df[f'{metric}_Rank'] = results_df[metric].rank(ascending=False)

print("\nFilter scores calculated:")
print(results_df.round(4))

# Visualize the results
plt.figure(figsize=(15, 12))

# 1. F-scores
plt.subplot(2, 2, 1)
bars = plt.bar(range(n_features), f_scores, color='skyblue', alpha=0.7)
# Highlight informative features
for i in range(n_informative):
    bars[i].set_color('red')
    bars[i].set_alpha(0.8)
plt.xlabel('Feature Index')
plt.ylabel('F-Score')
plt.title('F-Statistic Scores')
plt.xticks(range(n_features), [f'F{i+1}' for i in range(n_features)], rotation=45)
plt.grid(True, alpha=0.3)

# 2. Mutual Information
plt.subplot(2, 2, 2)
bars = plt.bar(range(n_features), mi_scores, color='lightgreen', alpha=0.7)
for i in range(n_informative):
    bars[i].set_color('red')
    bars[i].set_alpha(0.8)
plt.xlabel('Feature Index')
plt.ylabel('Mutual Information')
plt.title('Mutual Information Scores')
plt.xticks(range(n_features), [f'F{i+1}' for i in range(n_features)], rotation=45)
plt.grid(True, alpha=0.3)

# 3. Correlation
plt.subplot(2, 2, 3)
bars = plt.bar(range(n_features), corr_scores, color='orange', alpha=0.7)
for i in range(n_informative):
    bars[i].set_color('red')
    bars[i].set_alpha(0.8)
plt.xlabel('Feature Index')
plt.ylabel('|Correlation|')
plt.title('Absolute Correlation Scores')
plt.xticks(range(n_features), [f'F{i+1}' for i in range(n_features)], rotation=45)
plt.grid(True, alpha=0.3)

# 4. Chi-square
plt.subplot(2, 2, 4)
bars = plt.bar(range(n_features), chi2_scores, color='purple', alpha=0.7)
for i in range(n_informative):
    bars[i].set_color('red')
    bars[i].set_alpha(0.8)
plt.xlabel('Feature Index')
plt.ylabel('Chi-Square Score')
plt.title('Chi-Square Scores')
plt.xticks(range(n_features), [f'F{i+1}' for i in range(n_features)], rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'filter_scores_comparison.png'), dpi=300, bbox_inches='tight')

# Create ranking comparison
plt.figure(figsize=(12, 8))

# Create heatmap of rankings
ranking_data = results_df[['F_Score_Rank', 'MI_Score_Rank', 'Correlation_Rank', 'Chi2_Score_Rank']].values
ranking_df = pd.DataFrame(ranking_data, 
                         index=feature_names,
                         columns=['F-Score', 'MI', 'Correlation', 'Chi-Square'])

# Create heatmap
sns.heatmap(ranking_df, annot=True, cmap='RdYlBu_r', fmt='.0f', cbar_kws={'label': 'Rank (Lower = Better)'})
plt.title('Feature Rankings Across Different Filter Methods')
plt.xlabel('Filter Method')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_rankings_heatmap.png'), dpi=300, bbox_inches='tight')

# Create composite scoring demonstration
print("\n" + "="*60)
print("COMPOSITE SCORING DEMONSTRATION")
print("="*60)

# Normalize scores to 0-1 scale
def normalize_scores(scores):
    min_val = np.min(scores)
    max_val = np.max(scores)
    if max_val == min_val:
        return np.ones_like(scores)
    return (scores - min_val) / (max_val - min_val)

# Normalize all scores
f_normalized = normalize_scores(f_scores)
mi_normalized = normalize_scores(mi_scores)
corr_normalized = normalize_scores(corr_scores)
chi2_normalized = normalize_scores(chi2_scores)

# Create composite score with different weightings
weights = {
    'F-Score': 0.4,
    'MI': 0.3,
    'Correlation': 0.2,
    'Chi-Square': 0.1
}

composite_scores = (weights['F-Score'] * f_normalized + 
                   weights['MI'] * mi_normalized + 
                   weights['Correlation'] * corr_normalized + 
                   weights['Chi-Square'] * chi2_normalized)

# Add to results dataframe
results_df['Composite_Score'] = composite_scores
results_df['Composite_Rank'] = results_df['Composite_Score'].rank(ascending=False)

print("\nComposite scoring with weights:")
for method, weight in weights.items():
    print(f"- {method}: {weight:.1f}")

print(f"\nTop 5 features by composite score:")
top_features = results_df.nlargest(5, 'Composite_Score')[['Feature', 'Composite_Score', 'Composite_Rank']]
print(top_features.round(4))

# Visualize composite scoring
plt.figure(figsize=(14, 10))

# 1. Individual normalized scores
plt.subplot(2, 2, 1)
x = np.arange(n_features)
width = 0.2
plt.bar(x - 1.5*width, f_normalized, width, label='F-Score', alpha=0.7)
plt.bar(x - 0.5*width, mi_normalized, width, label='MI', alpha=0.7)
plt.bar(x + 0.5*width, corr_normalized, width, label='Correlation', alpha=0.7)
plt.bar(x + 1.5*width, chi2_normalized, width, label='Chi-Square', alpha=0.7)
plt.xlabel('Feature Index')
plt.ylabel('Normalized Score')
plt.title('Normalized Individual Scores')
plt.legend()
plt.xticks(x, [f'F{i+1}' for i in range(n_features)], rotation=45)
plt.grid(True, alpha=0.3)

# 2. Composite scores
plt.subplot(2, 2, 2)
bars = plt.bar(range(n_features), composite_scores, color='gold', alpha=0.7)
for i in range(n_informative):
    bars[i].set_color('red')
    bars[i].set_alpha(0.8)
plt.xlabel('Feature Index')
plt.ylabel('Composite Score')
plt.title('Composite Scores (Weighted Average)')
plt.xticks(range(n_features), [f'F{i+1}' for i in range(n_features)], rotation=45)
plt.grid(True, alpha=0.3)

# 3. Score comparison for top features
plt.subplot(2, 2, 3)
top_indices = results_df.nlargest(5, 'Composite_Score').index
top_names = [f'F{i+1}' for i in top_indices]
top_scores = results_df.loc[top_indices, ['F_Score', 'MI_Score', 'Correlation', 'Chi2_Score']].values.T

x_pos = np.arange(len(top_names))
width = 0.2
plt.bar(x_pos - 1.5*width, top_scores[0], width, label='F-Score', alpha=0.7)
plt.bar(x_pos - 0.5*width, top_scores[1], width, label='MI', alpha=0.7)
plt.bar(x_pos + 0.5*width, top_scores[2], width, label='Correlation', alpha=0.7)
plt.bar(x_pos + 1.5*width, top_scores[3], width, label='Chi-Square', alpha=0.7)
plt.xlabel('Top Features')
plt.ylabel('Score')
plt.title('Score Comparison for Top 5 Features')
plt.legend()
plt.xticks(x_pos, top_names)
plt.grid(True, alpha=0.3)

# 4. Ranking stability
plt.subplot(2, 2, 4)
ranking_stability = results_df[['F_Score_Rank', 'MI_Score_Rank', 'Correlation_Rank', 'Chi2_Score_Rank']].std(axis=1)
bars = plt.bar(range(n_features), ranking_stability, color='lightcoral', alpha=0.7)
for i in range(n_informative):
    bars[i].set_color('red')
    bars[i].set_alpha(0.8)
plt.xlabel('Feature Index')
plt.ylabel('Ranking Standard Deviation')
plt.title('Ranking Stability (Lower = More Stable)')
plt.xticks(range(n_features), [f'F{i+1}' for i in range(n_features)], rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'composite_scoring_analysis.png'), dpi=300, bbox_inches='tight')

# Performance comparison visualization
plt.figure(figsize=(12, 8))

# Create performance comparison
methods = ['F-Score', 'MI', 'Correlation', 'Chi-Square', 'Composite']
performance_metrics = []

# Calculate performance metrics (using top 8 features as ground truth)
for method in ['F_Score_Rank', 'MI_Score_Rank', 'Correlation_Rank', 'Chi2_Score_Rank', 'Composite_Rank']:
    top_features_method = results_df.nsmallest(n_informative, method)
    relevant_features_found = sum(i < n_informative for i in top_features_method.index)
    precision = relevant_features_found / n_informative
    performance_metrics.append(precision)

# Create bar plot
bars = plt.bar(methods, performance_metrics, color=['skyblue', 'lightgreen', 'orange', 'purple', 'gold'], alpha=0.7)
plt.xlabel('Filter Method')
plt.ylabel('Precision (Relevant Features Found)')
plt.title('Performance Comparison: Precision of Top 8 Features')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, performance_metrics):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{value:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'filter_methods_performance.png'), dpi=300, bbox_inches='tight')

# Summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print(f"\nFeature Selection Results:")
print(f"- Total features analyzed: {n_features}")
print(f"- Truly relevant features: {n_informative}")
print(f"- Features selected by each method (top {n_informative}):")

for method in ['F_Score', 'MI_Score', 'Correlation', 'Chi2_Score', 'Composite_Score']:
    top_features_method = results_df.nlargest(n_informative, method)
    relevant_features_found = sum(i < n_informative for i in top_features_method.index)
    precision = relevant_features_found / n_informative
    print(f"  {method}: {relevant_features_found}/{n_informative} relevant features (Precision: {precision:.2f})")

print(f"\nRanking stability analysis:")
ranking_stability = results_df[['F_Score_Rank', 'MI_Score_Rank', 'Correlation_Rank', 'Chi2_Score_Rank']].std(axis=1)
most_stable = results_df.loc[ranking_stability.idxmin(), 'Feature']
least_stable = results_df.loc[ranking_stability.idxmax(), 'Feature']
print(f"- Most stable ranking: {most_stable} (std: {ranking_stability.min():.2f})")
print(f"- Least stable ranking: {least_stable} (std: {ranking_stability.max():.2f})")

print(f"\nPlots saved to: {save_dir}")
print("\nQuestion 2 solution completed!")
