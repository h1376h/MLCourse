import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
import os
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_2_Quiz_1")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=== Univariate Feature Selection: Step-by-Step Analysis ===\n")

# 1. Generate synthetic dataset for demonstration
print("1. GENERATING SYNTHETIC DATASET")
print("-" * 50)

# Create a dataset with 20 features (10 informative, 10 noise)
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_repeated=0,
    n_clusters_per_class=2,
    random_state=42
)

# Create feature names
feature_names = [f'X_{i+1}' for i in range(20)]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print(f"Dataset shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")
print(f"Target distribution: {np.bincount(y)}")
print()

# 2. Demonstrate univariate feature selection process
print("2. UNIVARIATE FEATURE SELECTION PROCESS")
print("-" * 50)

# Calculate individual feature scores using different methods
print("Calculating feature scores using different univariate methods...")

# Method 1: F-statistic (ANOVA F-test)
f_scores, f_pvalues = f_classif(X, y)
print(f"F-statistic scores calculated for all {len(f_scores)} features")

# Method 2: Mutual Information
mi_scores = mutual_info_classif(X, y, random_state=42)
print(f"Mutual Information scores calculated for all {len(mi_scores)} features")

# Method 3: Correlation with target
corr_scores = []
for i in range(X.shape[1]):
    corr = np.corrcoef(X[:, i], y)[0, 1]
    corr_scores.append(abs(corr) if not np.isnan(corr) else 0)
corr_scores = np.array(corr_scores)
print(f"Correlation scores calculated for all {len(corr_scores)} features")

# Create results dataframe
results_df = pd.DataFrame({
    'Feature': feature_names,
    'F_Score': f_scores,
    'F_Pvalue': f_pvalues,
    'MI_Score': mi_scores,
    'Correlation': corr_scores
})

# Sort by F-score for ranking
results_df_sorted = results_df.sort_values('F_Score', ascending=False)
print("\nTop 10 features by F-score:")
print(results_df_sorted.head(10)[['Feature', 'F_Score', 'F_Pvalue', 'MI_Score', 'Correlation']].round(4))
print()

# 3. Visualize feature scores
print("3. CREATING VISUALIZATIONS")
print("-" * 50)

# Create a comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Univariate Feature Selection Analysis', fontsize=16, fontweight='bold')

# Plot 1: F-scores ranking
axes[0, 0].bar(range(len(feature_names)), results_df_sorted['F_Score'], 
                color='skyblue', edgecolor='navy', alpha=0.7)
axes[0, 0].set_title('Feature Ranking by F-Score')
axes[0, 0].set_xlabel('Feature Rank')
axes[0, 0].set_ylabel('F-Score')
axes[0, 0].set_xticks(range(len(feature_names)))
axes[0, 0].set_xticklabels(results_df_sorted['Feature'], rotation=45, ha='right')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: P-values (log scale)
axes[0, 1].semilogy(range(len(feature_names)), results_df_sorted['F_Pvalue'], 
                     'ro-', markersize=6, linewidth=2)
axes[0, 1].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label=r'$\alpha = 0.05$')
axes[0, 1].set_title('Feature P-values (Log Scale)')
axes[0, 1].set_xlabel('Feature Rank')
axes[0, 1].set_ylabel('P-value (log scale)')
axes[0, 1].set_xticks(range(len(feature_names)))
axes[0, 1].set_xticklabels(results_df_sorted['Feature'], rotation=45, ha='right')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# Plot 3: Mutual Information scores
axes[1, 0].bar(range(len(feature_names)), results_df_sorted['MI_Score'], 
                color='lightgreen', edgecolor='darkgreen', alpha=0.7)
axes[1, 0].set_title('Feature Ranking by Mutual Information')
axes[1, 0].set_xlabel('Feature Rank')
axes[1, 0].set_ylabel('Mutual Information Score')
axes[1, 0].set_xticks(range(len(feature_names)))
axes[1, 0].set_xticklabels(results_df_sorted['Feature'], rotation=45, ha='right')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Correlation scores
axes[1, 1].bar(range(len(feature_names)), results_df_sorted['Correlation'], 
                color='salmon', edgecolor='darkred', alpha=0.7)
axes[1, 1].set_title('Feature Ranking by Absolute Correlation')
axes[1, 1].set_xlabel('Feature Rank')
axes[1, 1].set_ylabel('|Correlation|')
axes[1, 1].set_xticks(range(len(feature_names)))
axes[1, 1].set_xticklabels(results_df_sorted['Feature'], rotation=45, ha='right')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'univariate_feature_analysis.png'), dpi=300, bbox_inches='tight')

# 4. Demonstrate computational complexity
print("4. COMPUTATIONAL COMPLEXITY ANALYSIS")
print("-" * 50)

# Simulate feature evaluation time
n_features_list = [10, 50, 100, 200, 500, 1000]
evaluation_times = []

print("Simulating feature evaluation times...")
for n_features in n_features_list:
    start_time = time.time()
    
    # Simulate evaluating each feature individually
    for i in range(n_features):
        # Simulate some computation time
        _ = np.random.rand(1000, 1000).sum()
    
    end_time = time.time()
    evaluation_times.append(end_time - start_time)
    print(f"Features: {n_features:4d} | Time: {evaluation_times[-1]:.4f} seconds")

# Calculate theoretical time based on 2 seconds per feature
theoretical_times = [n_features * 2 for n_features in n_features_list]

# Create complexity visualization
plt.figure(figsize=(12, 8))

# Plot actual vs theoretical times
plt.subplot(2, 1, 1)
plt.plot(n_features_list, evaluation_times, 'bo-', linewidth=2, markersize=8, label='Actual Time')
plt.plot(n_features_list, theoretical_times, 'r--', linewidth=2, label='Theoretical Time (2s/feature)')
plt.xlabel('Number of Features')
plt.ylabel('Time (seconds)')
plt.title('Feature Selection Time Complexity')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.yscale('log')

# Plot time per feature
plt.subplot(2, 1, 2)
time_per_feature = [t/n for t, n in zip(evaluation_times, n_features_list)]
plt.plot(n_features_list, time_per_feature, 'go-', linewidth=2, markersize=8)
plt.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='2 seconds per feature')
plt.xlabel('Number of Features')
plt.ylabel('Time per Feature (seconds)')
plt.title('Time per Feature vs Number of Features')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xscale('log')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'computational_complexity.png'), dpi=300, bbox_inches='tight')

# 5. Answer the specific questions
print("\n5. ANSWERING THE SPECIFIC QUESTIONS")
print("-" * 50)

print("Question 1: What is the main advantage of univariate methods?")
print("Answer: Univariate methods evaluate each feature independently, making them:")
print("  - Simple and easy to understand")
print("  - Computationally efficient (O(n) complexity)")
print("  - Interpretable (each feature's importance is clear)")
print("  - Robust to feature interactions")
print()

print("Question 2: What is the main limitation of univariate methods?")
print("Answer: Univariate methods have several limitations:")
print("  - They cannot detect feature interactions or redundancy")
print("  - They may miss features that are only useful in combination")
print("  - They assume features are independent")
print("  - They may select redundant features")
print()

print("Question 3: If you have 100 features, how many individual evaluations does univariate selection require?")
print(f"Answer: {100} individual evaluations")
print("  - Each feature is evaluated independently")
print("  - Total evaluations = number of features = 100")
print()

print("Question 4: Computational complexity and time calculation")
print("Given: 500 features, 2 seconds per feature evaluation")
print("Computational complexity: O(n) where n = number of features")
print("Total time = 500 features × 2 seconds/feature = 1000 seconds")
print("1000 seconds = 16 minutes and 40 seconds")
print()

# 6. Demonstrate feature selection with different k values
print("6. FEATURE SELECTION WITH DIFFERENT K VALUES")
print("-" * 50)

k_values = [5, 10, 15]
for k in k_values:
    # Select top k features using F-score
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
    
    print(f"Top {k} features selected:")
    for i, feature in enumerate(selected_features):
        score = results_df[results_df['Feature'] == feature]['F_Score'].iloc[0]
        print(f"  {i+1:2d}. {feature}: F-score = {score:.4f}")
    print()

# 7. Create final summary visualization
plt.figure(figsize=(14, 10))

# Create a heatmap of selected features
k = 10
selector = SelectKBest(score_func=f_classif, k=k)
X_selected = selector.fit_transform(X, y)
selected_indices = selector.get_support(indices=True)
selected_features = [feature_names[i] for i in selected_indices]

# Create correlation matrix for selected features
selected_df = df[selected_features + ['target']]
corr_matrix = selected_df.corr()

# Plot heatmap
plt.subplot(2, 2, 1)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title(f'Correlation Matrix of Top {k} Selected Features')

# Plot feature importance comparison
plt.subplot(2, 2, 2)
top_k_df = results_df.iloc[selected_indices]
x_pos = np.arange(len(selected_features))
width = 0.35

plt.bar(x_pos - width/2, top_k_df['F_Score'], width, label='F-Score', color='skyblue', alpha=0.7)
plt.bar(x_pos + width/2, top_k_df['MI_Score'], width, label='MI-Score', color='lightgreen', alpha=0.7)
plt.xlabel('Features')
plt.ylabel('Scores')
plt.title(f'Top {k} Features: F-Score vs Mutual Information')
plt.xticks(x_pos, selected_features, rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot cumulative importance
plt.subplot(2, 2, 3)
cumulative_importance = np.cumsum(top_k_df['F_Score']) / np.sum(top_k_df['F_Score']) * 100
plt.plot(range(1, k+1), cumulative_importance, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Features')
plt.ylabel('Cumulative Importance (%)')
plt.title('Cumulative Feature Importance')
plt.grid(True, alpha=0.3)
plt.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% threshold')
plt.legend()

# Plot feature distribution
plt.subplot(2, 2, 4)
for i, feature in enumerate(selected_features[:5]):  # Show first 5 features
    feature_data = df[feature]
    plt.hist(feature_data[df['target'] == 0], alpha=0.5, label=f'{feature} (Class 0)', bins=20)
    plt.hist(feature_data[df['target'] == 1], alpha=0.5, label=f'{feature} (Class 1)', bins=20)

plt.xlabel('Feature Values')
plt.ylabel('Frequency')
plt.title('Distribution of Top 5 Features by Class')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_selection_summary.png'), dpi=300, bbox_inches='tight')

print("7. VISUALIZATIONS COMPLETED")
print("-" * 50)
print(f"All plots saved to: {save_dir}")
print("\nGenerated visualizations:")
print("1. univariate_feature_analysis.png - Comprehensive feature analysis")
print("2. computational_complexity.png - Time complexity analysis")
print("3. feature_selection_summary.png - Final summary and insights")

print("\n=== Analysis Complete ===")
print("The code demonstrates:")
print("- How univariate feature selection works step by step")
print("- Computational complexity analysis")
print("- Feature ranking and selection")
print("- Visualization of results")
print("- Answers to all quiz questions")
