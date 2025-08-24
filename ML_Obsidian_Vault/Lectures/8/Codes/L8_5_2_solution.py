import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_5_Quiz_2")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("UNIVARIATE vs MULTIVARIATE FILTERS COMPARISON")
print("=" * 80)

# 1. Generate synthetic dataset for demonstration
print("\n1. GENERATING SYNTHETIC DATASET")
print("-" * 50)

# Create a dataset with feature interactions
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    n_repeated=1,
    n_clusters_per_class=2,
    random_state=42
)

# Create feature names
feature_names = [f'F{i+1}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print(f"Dataset shape: {X.shape}")
print(f"Features: {feature_names}")
print(f"Target distribution: {np.bincount(y)}")

# 2. Univariate Filter Analysis
print("\n2. UNIVARIATE FILTER ANALYSIS")
print("-" * 50)

# Calculate various univariate scores
print("Calculating univariate filter scores...")

# Correlation-based scores
correlation_scores = []
for i in range(X.shape[1]):
    corr = np.corrcoef(X[:, i], y)[0, 1]
    correlation_scores.append(abs(corr))

# F-statistic scores
f_scores, _ = f_classif(X, y)

# Mutual information scores
mi_scores = mutual_info_classif(X, y, random_state=42)

# Create univariate scores dataframe
univariate_df = pd.DataFrame({
    'Feature': feature_names,
    'Correlation': correlation_scores,
    'F_Score': f_scores,
    'Mutual_Info': mi_scores
})

print("\nUnivariate Filter Scores:")
print(univariate_df.round(4))

# 3. Multivariate Filter Analysis
print("\n3. MULTIVARIATE FILTER ANALYSIS")
print("-" * 50)

print("Applying multivariate filter methods...")

# Random Forest importance (multivariate)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
rf_importance = rf.feature_importances_

# Recursive Feature Elimination (multivariate)
lr = LogisticRegression(random_state=42, max_iter=1000)
rfe = RFE(estimator=lr, n_features_to_select=5)
rfe.fit(X, y)
rfe_ranking = rfe.ranking_

# Create multivariate scores dataframe
multivariate_df = pd.DataFrame({
    'Feature': feature_names,
    'RF_Importance': rf_importance,
    'RFE_Ranking': rfe_ranking
})

print("\nMultivariate Filter Scores:")
print(multivariate_df.round(4))

# 4. Performance Comparison
print("\n4. PERFORMANCE COMPARISON")
print("-" * 50)

# Split data for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Function to evaluate feature selection
def evaluate_selection(X_train, X_test, y_train, y_test, selected_features):
    if len(selected_features) == 0:
        return 0.0
    
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    return accuracy_score(y_test, y_pred)

# Test different feature selection approaches
print("Evaluating different feature selection approaches...")

# Univariate: Top 5 features by correlation
top_corr_features = np.argsort(correlation_scores)[-5:]
corr_accuracy = evaluate_selection(X_train, X_test, y_train, y_test, top_corr_features)

# Univariate: Top 5 features by F-score
top_f_features = np.argsort(f_scores)[-5:]
f_accuracy = evaluate_selection(X_train, X_test, y_train, y_test, top_f_features)

# Univariate: Top 5 features by mutual information
top_mi_features = np.argsort(mi_scores)[-5:]
mi_accuracy = evaluate_selection(X_train, X_test, y_train, y_test, top_mi_features)

# Multivariate: Top 5 features by Random Forest
top_rf_features = np.argsort(rf_importance)[-5:]
rf_accuracy = evaluate_selection(X_train, X_test, y_train, y_test, top_rf_features)

# Multivariate: Top 5 features by RFE
top_rfe_features = np.argsort(rfe_ranking)[:5]
rfe_accuracy = evaluate_selection(X_train, X_test, y_train, y_test, top_rfe_features)

# Create performance comparison dataframe
performance_df = pd.DataFrame({
    'Method': ['Correlation', 'F-Score', 'Mutual Info', 'Random Forest', 'RFE'],
    'Type': ['Univariate', 'Univariate', 'Univariate', 'Multivariate', 'Multivariate'],
    'Accuracy': [corr_accuracy, f_accuracy, mi_accuracy, rf_accuracy, rfe_accuracy],
    'Selected_Features': [
        [feature_names[i] for i in top_corr_features],
        [feature_names[i] for i in top_f_features],
        [feature_names[i] for i in top_mi_features],
        [feature_names[i] for i in top_rf_features],
        [feature_names[i] for i in top_rfe_features]
    ]
})

print("\nPerformance Comparison:")
print(performance_df.round(4))

# 5. Computational Complexity Analysis
print("\n5. COMPUTATIONAL COMPLEXITY ANALYSIS")
print("-" * 50)

# Measure execution time for different methods
print("Measuring execution time...")

# Univariate methods
start_time = time.time()
_ = SelectKBest(score_func=f_classif, k=5).fit(X, y)
univariate_time = time.time() - start_time

# Multivariate methods
start_time = time.time()
_ = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
multivariate_rf_time = time.time() - start_time

start_time = time.time()
_ = RFE(estimator=LogisticRegression(random_state=42), n_features_to_select=5).fit(X, y)
multivariate_rfe_time = time.time() - start_time

complexity_df = pd.DataFrame({
    'Method': ['F-Score (Univariate)', 'Random Forest (Multivariate)', 'RFE (Multivariate)'],
    'Execution_Time_Seconds': [univariate_time, multivariate_rf_time, multivariate_rfe_time],
    'Relative_Time': [1, multivariate_rf_time/univariate_time, multivariate_rfe_time/univariate_time]
})

print("\nComputational Complexity:")
print(complexity_df.round(4))

# 6. Feature Interaction Analysis
print("\n6. FEATURE INTERACTION ANALYSIS")
print("-" * 50)

# Calculate correlation matrix between features
correlation_matrix = np.corrcoef(X.T)

# Find highly correlated feature pairs
high_corr_pairs = []
for i in range(len(feature_names)):
    for j in range(i+1, len(feature_names)):
        corr_val = abs(correlation_matrix[i, j])
        if corr_val > 0.7:  # High correlation threshold
            high_corr_pairs.append((feature_names[i], feature_names[j], corr_val))

print(f"Found {len(high_corr_pairs)} highly correlated feature pairs (|correlation| > 0.7):")
for pair in high_corr_pairs:
    print(f"  {pair[0]} - {pair[1]}: {pair[2]:.3f}")

# 7. Visualization
print("\n7. CREATING VISUALIZATIONS")
print("-" * 50)

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Univariate vs Multivariate Filters Comparison', fontsize=16, fontweight='bold')

# Plot 1: Univariate scores comparison
ax1 = axes[0, 0]
x_pos = np.arange(len(feature_names))
width = 0.25

ax1.bar(x_pos - width, univariate_df['Correlation'], width, label='Correlation', alpha=0.8)
ax1.bar(x_pos, univariate_df['F_Score']/np.max(univariate_df['F_Score']), width, label='F-Score (Normalized)', alpha=0.8)
ax1.bar(x_pos + width, univariate_df['Mutual_Info']/np.max(univariate_df['Mutual_Info']), width, label='Mutual Info (Normalized)', alpha=0.8)

ax1.set_xlabel('Features')
ax1.set_ylabel('Score')
ax1.set_title('Univariate Filter Scores')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(feature_names, rotation=45)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Multivariate scores comparison
ax2 = axes[0, 1]
ax2.bar(x_pos, multivariate_df['RF_Importance'], alpha=0.8, color='orange', label='Random Forest')
ax2_twin = ax2.twinx()
ax2_twin.bar(x_pos, multivariate_df['RFE_Ranking'], alpha=0.6, color='red', label='RFE Ranking')

ax2.set_xlabel('Features')
ax2.set_ylabel('RF Importance', color='orange')
ax2_twin.set_ylabel('RFE Ranking (Lower is Better)', color='red')
ax2.set_title('Multivariate Filter Scores')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(feature_names, rotation=45)
ax2.legend(loc='upper left')
ax2_twin.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# Plot 3: Performance comparison
ax3 = axes[0, 2]
methods = performance_df['Method']
accuracies = performance_df['Accuracy']
colors = ['blue' if t == 'Univariate' else 'red' for t in performance_df['Type']]

bars = ax3.bar(methods, accuracies, color=colors, alpha=0.7)
ax3.set_xlabel('Method')
ax3.set_ylabel('Accuracy')
ax3.set_title('Classification Accuracy Comparison')
ax3.set_ylim(0, 1)
ax3.tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{acc:.3f}', ha='center', va='bottom')

# Plot 4: Computational complexity
ax4 = axes[1, 0]
methods_comp = complexity_df['Method']
times = complexity_df['Execution_Time_Seconds']
colors_comp = ['blue', 'red', 'red']

bars_comp = ax4.bar(methods_comp, times, color=colors_comp, alpha=0.7)
ax4.set_xlabel('Method')
ax4.set_ylabel('Execution Time (seconds)')
ax4.set_title('Computational Complexity')
ax4.tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar, time_val in zip(bars_comp, times):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'{time_val:.3f}s', ha='center', va='bottom')

# Plot 5: Feature correlation heatmap
ax5 = axes[1, 1]
im = ax5.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
ax5.set_xticks(range(len(feature_names)))
ax5.set_yticks(range(len(feature_names)))
ax5.set_xticklabels(feature_names, rotation=45)
ax5.set_yticklabels(feature_names)
ax5.set_title('Feature Correlation Matrix')
plt.colorbar(im, ax=ax5)

# Plot 6: Selected features comparison
ax6 = axes[1, 2]
# Count how many times each feature is selected
feature_selection_count = {name: 0 for name in feature_names}
for features in performance_df['Selected_Features']:
    for feature in features:
        feature_selection_count[feature] += 1

features_list = list(feature_selection_count.keys())
counts = list(feature_selection_count.values())

bars_sel = ax6.bar(features_list, counts, alpha=0.7, color='green')
ax6.set_xlabel('Features')
ax6.set_ylabel('Selection Frequency')
ax6.set_title('Feature Selection Frequency Across Methods')
ax6.set_xticklabels(features_list, rotation=45)
ax6.set_ylim(0, max(counts) + 1)

# Add value labels on bars
for bar, count in zip(bars_sel, counts):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             str(count), ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'univariate_vs_multivariate_comparison.png'), dpi=300, bbox_inches='tight')

# Create detailed comparison plots
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('Detailed Analysis of Univariate vs Multivariate Filters', fontsize=16, fontweight='bold')

# Plot A: Score distribution comparison
ax_a = axes2[0, 0]
univariate_scores = np.concatenate([
    univariate_df['Correlation'],
    univariate_df['F_Score']/np.max(univariate_df['F_Score']),
    univariate_df['Mutual_Info']/np.max(univariate_df['Mutual_Info'])
])
multivariate_scores = np.concatenate([
    multivariate_df['RF_Importance'],
    multivariate_df['RFE_Ranking']/np.max(multivariate_df['RFE_Ranking'])
])

ax_a.hist(univariate_scores, bins=15, alpha=0.7, label='Univariate', color='blue')
ax_a.hist(multivariate_scores, bins=15, alpha=0.7, label='Multivariate', color='red')
ax_a.set_xlabel('Normalized Scores')
ax_a.set_ylabel('Frequency')
ax_a.set_title('Score Distribution Comparison')
ax_a.legend()
ax_a.grid(True, alpha=0.3)

# Plot B: Feature ranking stability
ax_b = axes2[0, 1]
# Create ranking comparison
corr_ranking = np.argsort(correlation_scores)[::-1]
rf_ranking = np.argsort(rf_importance)[::-1]

ax_b.scatter(range(len(feature_names)), corr_ranking, label='Correlation', s=100, alpha=0.7)
ax_b.scatter(range(len(feature_names)), rf_ranking, label='Random Forest', s=100, alpha=0.7)
ax_b.set_xlabel('Feature Index')
ax_b.set_ylabel('Ranking Position')
ax_b.set_title('Feature Ranking Comparison')
ax_b.legend()
ax_b.grid(True, alpha=0.3)
ax_b.invert_yaxis()

# Plot C: Performance vs complexity trade-off
ax_c = axes2[1, 0]
# Normalize accuracy and time for fair comparison
norm_accuracy = performance_df['Accuracy'] / np.max(performance_df['Accuracy'])
norm_time = [1, 1, 1, multivariate_rf_time/univariate_time, multivariate_rfe_time/univariate_time]

colors_tradeoff = ['blue' if t == 'Univariate' else 'red' for t in performance_df['Type']]
ax_c.scatter(norm_time, norm_accuracy, c=colors_tradeoff, s=100, alpha=0.7)

# Add labels for each point
for i, method in enumerate(performance_df['Method']):
    ax_c.annotate(method, (norm_time[i], norm_accuracy[i]), 
                  xytext=(5, 5), textcoords='offset points', fontsize=8)

ax_c.set_xlabel('Normalized Execution Time')
ax_c.set_ylabel('Normalized Accuracy')
ax_c.set_title('Performance vs Complexity Trade-off')
ax_c.grid(True, alpha=0.3)

# Plot D: Feature selection overlap
ax_d = axes2[1, 1]
# Create Venn diagram-like visualization
from matplotlib_venn import venn2

# Get unique features selected by each approach
univariate_features = set()
for features in performance_df[performance_df['Type'] == 'Univariate']['Selected_Features']:
    univariate_features.update(features)

multivariate_features = set()
for features in performance_df[performance_df['Type'] == 'Multivariate']['Selected_Features']:
    multivariate_features.update(features)

# Create simple overlap visualization
overlap = len(univariate_features.intersection(multivariate_features))
only_univariate = len(univariate_features - multivariate_features)
only_multivariate = len(multivariate_features - univariate_features)

labels = ['Only Univariate', 'Overlap', 'Only Multivariate']
sizes = [only_univariate, overlap, only_multivariate]
colors_pie = ['lightblue', 'lightgreen', 'lightcoral']

ax_d.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.0f', startangle=90)
ax_d.set_title('Feature Selection Overlap')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'detailed_analysis.png'), dpi=300, bbox_inches='tight')

# 8. Summary and Insights
print("\n8. SUMMARY AND KEY INSIGHTS")
print("-" * 50)

print("\nKey Differences:")
print(f"• Univariate filters evaluate features independently")
print(f"• Multivariate filters consider feature interactions and dependencies")
print(f"• Univariate methods are faster but may miss feature interactions")
print(f"• Multivariate methods are slower but capture feature relationships")

print("\nPerformance Summary:")
print(f"• Best univariate method: {performance_df.loc[performance_df['Type'] == 'Univariate', 'Accuracy'].idxmax()}")
print(f"• Best multivariate method: {performance_df.loc[performance_df['Type'] == 'Multivariate', 'Accuracy'].idxmax()}")
print(f"• Overall best method: {performance_df.loc[performance_df['Accuracy'].idxmax(), 'Method']}")

print("\nComputational Efficiency:")
print(f"• Univariate methods are {complexity_df.loc[0, 'Relative_Time']:.1f}x faster than multivariate")
print(f"• Random Forest is {complexity_df.loc[1, 'Relative_Time']:.1f}x slower than univariate")
print(f"• RFE is {complexity_df.loc[2, 'Relative_Time']:.1f}x slower than univariate")

print(f"\nPlots saved to: {save_dir}")
print("=" * 80)
