import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_1_Quiz_18")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting (disabled for compatibility)
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'sans-serif'

print("=== Ensemble Feature Selection Demonstration ===\n")

# 1. Generate synthetic dataset for demonstration
print("1. Generating synthetic dataset...")
X, y = make_classification(n_samples=1000, n_features=20, n_informative=8, 
                          n_redundant=6, n_clusters_per_class=1, random_state=42)
feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Dataset shape: {X.shape}")
print(f"Number of informative features: 8")
print(f"Number of redundant features: 6")
print(f"Number of noise features: 6\n")

# 2. Apply different feature selection methods
print("2. Applying different feature selection methods...")

# Method 1: Statistical tests (F-test)
print("Method 1: F-test (ANOVA)")
f_scores, f_pvalues = f_classif(X_scaled, y)
f_selector = SelectKBest(score_func=f_classif, k=10)
f_selector.fit(X_scaled, y)
f_scores_selected = f_selector.scores_
f_pvalues_selected = f_selector.pvalues_
print(f"F-scores shape: {f_scores.shape}")
print(f"Selected features: {np.where(f_selector.get_support())[0] + 1}")

# Method 2: Mutual Information
print("\nMethod 2: Mutual Information")
mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
mi_selector = SelectKBest(score_func=mutual_info_classif, k=10)
mi_selector.fit(X_scaled, y)
mi_scores_selected = mi_selector.scores_
print(f"MI scores shape: {mi_scores.shape}")
print(f"Selected features: {np.where(mi_selector.get_support())[0] + 1}")

# Method 3: Recursive Feature Elimination with Random Forest
print("\nMethod 3: Recursive Feature Elimination (RFE)")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rfe_selector = RFE(estimator=rf, n_features_to_select=10, step=1)
rfe_selector.fit(X_scaled, y)
rfe_ranking = rfe_selector.ranking_
rfe_support = rfe_selector.support_
print(f"RFE ranking shape: {rfe_ranking.shape}")
print(f"Selected features: {np.where(rfe_selector.support_)[0] + 1}")

# 3. Create feature importance dataframe
print("\n3. Creating feature importance comparison...")
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'F_Test_Score': f_scores,
    'F_Test_PValue': f_pvalues,
    'Mutual_Info_Score': mi_scores,
    'RFE_Ranking': rfe_ranking,
    'RFE_Support': rfe_support
})

# Normalize scores to [0,1] range for comparison
feature_importance_df['F_Test_Normalized'] = (f_scores - f_scores.min()) / (f_scores.max() - f_scores.min())
feature_importance_df['MI_Normalized'] = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min())
feature_importance_df['RFE_Normalized'] = 1 - (rfe_ranking - 1) / (rfe_ranking.max() - 1)

print("Feature importance comparison (first 10 features):")
print(feature_importance_df.head(10).round(4))

# 4. Ensemble aggregation methods
print("\n4. Ensemble aggregation methods...")

# Simple voting
print("Simple Voting:")
# Create support columns for each method
feature_importance_df['F_Test_Support'] = f_selector.get_support()
feature_importance_df['MI_Support'] = mi_selector.get_support()
feature_importance_df['RFE_Support'] = rfe_selector.support_

simple_voting = (feature_importance_df['F_Test_Support'] + 
                feature_importance_df['MI_Support'] + 
                feature_importance_df['RFE_Support'])
simple_voting_support = simple_voting >= 2
print(f"Features selected by at least 2 methods: {np.where(simple_voting_support)[0] + 1}")

# Weighted voting
print("\nWeighted Voting:")
weights = [0.4, 0.3, 0.3]  # F-test, MI, RFE
weighted_scores = (weights[0] * feature_importance_df['F_Test_Normalized'] + 
                   weights[1] * feature_importance_df['MI_Normalized'] + 
                   weights[2] * feature_importance_df['RFE_Normalized'])

feature_importance_df['Ensemble_Score'] = weighted_scores
feature_importance_df['Ensemble_Rank'] = feature_importance_df['Ensemble_Score'].rank(ascending=False)

print("Top 10 features by ensemble score:")
print(feature_importance_df.nlargest(10, 'Ensemble_Score')[['Feature', 'Ensemble_Score', 'Ensemble_Rank']].round(4))

# 5. Specific example from the question
print("\n5. Specific example from the question:")
print("Feature A scores: [0.8, 0.6, 0.9]")
print("Feature B scores: [0.7, 0.8, 0.5]")
print("Weights: [0.4, 0.3, 0.3]")

feature_a_scores = np.array([0.8, 0.6, 0.9])
feature_b_scores = np.array([0.7, 0.8, 0.5])
weights = np.array([0.4, 0.3, 0.3])

# Calculate weighted ensemble scores
ensemble_a = np.sum(weights * feature_a_scores)
ensemble_b = np.sum(weights * feature_b_scores)

print(f"\nFeature A ensemble score: {ensemble_a:.3f}")
print(f"Feature B ensemble score: {ensemble_b:.3f}")
print(f"Feature A has {'higher' if ensemble_a > ensemble_b else 'lower'} ensemble score")

# 6. Visualizations
print("\n6. Generating visualizations...")

# Plot 1: Individual method scores comparison
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
top_features = feature_importance_df.nlargest(10, 'Ensemble_Score')
x_pos = np.arange(len(top_features))
width = 0.25

plt.bar(x_pos - width, top_features['F_Test_Normalized'], width, label='F-Test', alpha=0.8)
plt.bar(x_pos, top_features['MI_Normalized'], width, label='Mutual Info', alpha=0.8)
plt.bar(x_pos + width, top_features['RFE_Normalized'], width, label='RFE', alpha=0.8)

plt.xlabel('Features')
plt.ylabel('Normalized Scores')
plt.title('Feature Selection Scores by Method')
plt.xticks(x_pos, [f'F{i+1}' for i in range(len(top_features))], rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Ensemble scores
plt.subplot(2, 2, 2)
plt.bar(range(len(top_features)), top_features['Ensemble_Score'], color='skyblue', alpha=0.8)
plt.xlabel('Features')
plt.ylabel('Ensemble Score')
plt.title('Ensemble Feature Selection Scores')
plt.xticks(range(len(top_features)), [f'F{i+1}' for i in range(len(top_features))], rotation=45)
plt.grid(True, alpha=0.3)

# Plot 3: Method agreement heatmap
plt.subplot(2, 2, 3)
method_agreement = np.zeros((len(feature_names), 3))
method_agreement[:, 0] = feature_importance_df['F_Test_Normalized']
method_agreement[:, 1] = feature_importance_df['MI_Normalized']
method_agreement[:, 2] = feature_importance_df['RFE_Normalized']

# Create heatmap
im = plt.imshow(method_agreement.T, cmap='YlOrRd', aspect='auto')
plt.colorbar(im)
plt.xlabel('Features')
plt.ylabel('Methods')
plt.title('Feature Selection Method Agreement')
plt.yticks([0, 1, 2], ['F-Test', 'MI', 'RFE'])
plt.xticks(range(0, len(feature_names), 2), [f'F{i+1}' for i in range(0, len(feature_names), 2)], rotation=45)

# Plot 4: Feature selection overlap
plt.subplot(2, 2, 4)

# Get selected features for each method
f_test_selected = set(np.where(f_selector.get_support())[0])
mi_selected = set(np.where(mi_selector.get_support())[0])
rfe_selected = set(np.where(rfe_selector.support_)[0])

# Create a simple overlap visualization instead of Venn diagram
overlap_12 = len(f_test_selected.intersection(mi_selected))
overlap_13 = len(f_test_selected.intersection(rfe_selected))
overlap_23 = len(mi_selected.intersection(rfe_selected))
overlap_all = len(f_test_selected.intersection(mi_selected).intersection(rfe_selected))

plt.text(0.5, 0.5, f'F-Test & MI: {overlap_12}\nF-Test & RFE: {overlap_13}\nMI & RFE: {overlap_23}\nAll: {overlap_all}', 
         ha='center', va='center', transform=plt.gca().transAxes, fontsize=12,
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))
plt.title('Feature Selection Method Overlap')
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'ensemble_feature_selection_overview.png'), dpi=300, bbox_inches='tight')

# Plot 5: Detailed ensemble score analysis
plt.figure(figsize=(12, 8))

# Sort features by ensemble score
sorted_features = feature_importance_df.sort_values('Ensemble_Score', ascending=False)

plt.subplot(2, 1, 1)
plt.bar(range(len(sorted_features)), sorted_features['Ensemble_Score'], 
        color='lightcoral', alpha=0.8, edgecolor='black')
plt.xlabel('Features (sorted by ensemble score)')
plt.ylabel('Ensemble Score')
plt.title('Ensemble Feature Selection Scores (Sorted)')
plt.xticks(range(0, len(sorted_features), 2), 
           [sorted_features.iloc[i]['Feature'] for i in range(0, len(sorted_features), 2)], 
           rotation=45)
plt.grid(True, alpha=0.3)

# Add threshold line for top features
threshold = sorted_features.iloc[9]['Ensemble_Score']  # Top 10 features
plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, 
            label=f'Top 10 threshold: {threshold:.3f}')
plt.legend()

# Plot 6: Method comparison for top features
plt.subplot(2, 1, 2)
top_10_features = sorted_features.head(10)
x_pos = np.arange(len(top_10_features))
width = 0.25

plt.bar(x_pos - width, top_10_features['F_Test_Normalized'], width, 
        label='F-Test', alpha=0.8, color='lightblue')
plt.bar(x_pos, top_10_features['MI_Normalized'], width, 
        label='Mutual Info', alpha=0.8, color='lightgreen')
plt.bar(x_pos + width, top_10_features['RFE_Normalized'], width, 
        label='RFE', alpha=0.8, color='lightcoral')

plt.xlabel('Top 10 Features')
plt.ylabel('Normalized Scores')
plt.title('Method Comparison for Top 10 Features')
plt.xticks(x_pos, [f'F{i+1}' for i in range(len(top_10_features))], rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'ensemble_detailed_analysis.png'), dpi=300, bbox_inches='tight')

# 7. Summary statistics
print("\n7. Summary statistics:")
print(f"Total features: {len(feature_names)}")
print(f"Features selected by F-Test: {np.sum(f_selector.get_support())}")
print(f"Features selected by Mutual Info: {np.sum(mi_selector.get_support())}")
print(f"Features selected by RFE: {np.sum(rfe_selector.support_)}")
print(f"Features selected by ensemble (top 10): 10")

# Agreement analysis
agreement_matrix = np.zeros((len(feature_names), len(feature_names)))
for i in range(len(feature_names)):
    for j in range(len(feature_names)):
        if i == j:
            agreement_matrix[i, j] = 1
        else:
            # Calculate agreement based on selection patterns
            method1_selected = [f_selector.get_support()[i], mi_selector.get_support()[i], rfe_selector.support_[i]]
            method2_selected = [f_selector.get_support()[j], mi_selector.get_support()[j], rfe_selector.support_[j]]
            agreement = np.mean([m1 == m2 for m1, m2 in zip(method1_selected, method2_selected)])
            agreement_matrix[i, j] = agreement

print(f"\nAverage method agreement: {np.mean(agreement_matrix[np.triu_indices_from(agreement_matrix, k=1)]):.3f}")

# Save results to CSV
results_file = os.path.join(save_dir, 'ensemble_feature_selection_results.csv')
feature_importance_df.to_csv(results_file, index=False)
print(f"\nDetailed results saved to: {results_file}")
print(f"Visualizations saved to: {save_dir}")

print("\n=== Ensemble Feature Selection Demonstration Complete ===")
