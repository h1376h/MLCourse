import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_4_Quiz_11")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("Feature Selection Stability Analysis")
print("=" * 50)

# 1. Generate synthetic dataset for demonstration
print("\n1. Generating synthetic dataset...")
np.random.seed(42)
X, y = make_classification(
    n_samples=1000, 
    n_features=20, 
    n_informative=8, 
    n_redundant=6, 
    n_clusters_per_class=2,
    random_state=42
)

feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
print(f"Dataset shape: {X.shape}")
print(f"Number of informative features: 8")
print(f"Number of redundant features: 6")
print(f"Number of noise features: 6")

# 2. Demonstrate why stability is important
print("\n2. Why stability is important...")
print("Stability ensures that selected features are robust and not artifacts of specific data samples.")
print("Unstable feature selection can lead to:")
print("- Poor generalization performance")
print("- Inconsistent model interpretations")
print("- Unreliable feature importance rankings")

# 3. Bootstrap sampling for stability measurement
print("\n3. Performing bootstrap sampling for stability analysis...")
n_bootstrap = 10
n_samples = X.shape[0]
bootstrap_size = int(0.8 * n_samples)

# Store selected features for each bootstrap sample
selected_features_bootstrap = []
feature_importance_bootstrap = []

for i in range(n_bootstrap):
    # Generate bootstrap indices
    bootstrap_indices = np.random.choice(n_samples, size=bootstrap_size, replace=True)
    X_bootstrap = X[bootstrap_indices]
    y_bootstrap = y[bootstrap_indices]
    
    # Feature selection using Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=i)
    rf.fit(X_bootstrap, y_bootstrap)
    
    # Get top 8 features (same as number of informative features)
    top_k = 8
    feature_importance = rf.feature_importances_
    top_features_idx = np.argsort(feature_importance)[-top_k:]
    
    selected_features_bootstrap.append(top_features_idx)
    feature_importance_bootstrap.append(feature_importance)

print(f"Generated {n_bootstrap} bootstrap samples")
print(f"Each sample selects top {top_k} features")

# 4. Calculate stability measures
print("\n4. Calculating stability measures...")

# Create feature selection matrix
feature_selection_matrix = np.zeros((n_bootstrap, X.shape[1]))
for i, selected_features in enumerate(selected_features_bootstrap):
    feature_selection_matrix[i, selected_features] = 1

# Calculate feature frequency (stability score)
feature_frequency = np.mean(feature_selection_matrix, axis=0)

# Calculate Jaccard similarity between different bootstrap samples
jaccard_similarities = []
for i in range(n_bootstrap):
    for j in range(i+1, n_bootstrap):
        set_i = set(selected_features_bootstrap[i])
        set_j = set(selected_features_bootstrap[j])
        
        intersection = len(set_i.intersection(set_j))
        union = len(set_i.union(set_j))
        
        jaccard = intersection / union if union > 0 else 0
        jaccard_similarities.append(jaccard)

mean_jaccard = np.mean(jaccard_similarities)
std_jaccard = np.std(jaccard_similarities)

print(f"Mean Jaccard similarity: {mean_jaccard:.3f} ± {std_jaccard:.3f}")

# 5. Answer specific question about feature appearing in 7 out of 10 samples
print("\n5. Stability calculation for feature appearing in 7 out of 10 samples...")
stability_score = 7 / 10
print(f"Stability score = 7/10 = {stability_score:.1f} = {stability_score*100:.0f}%")
print("This indicates the feature is relatively stable across bootstrap samples.")

# 6. Visualizations
print("\n6. Generating visualizations...")

# Plot 1: Feature selection frequency across bootstrap samples
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)

# Sort features by frequency
sorted_indices = np.argsort(feature_frequency)[::-1]
sorted_frequencies = feature_frequency[sorted_indices]
sorted_names = [feature_names[i] for i in sorted_indices]

bars = plt.bar(range(len(sorted_frequencies)), sorted_frequencies, 
               color='skyblue', edgecolor='navy', alpha=0.7)
plt.xlabel('Features')
plt.ylabel('Selection Frequency')
plt.title('Feature Selection Stability Across Bootstrap Samples')
plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha='right')

# Highlight the 7/10 case
for i, (freq, name) in enumerate(zip(sorted_frequencies, sorted_names)):
    if abs(freq - 0.7) < 0.01:  # Close to 0.7
        bars[i].set_color('red')
        bars[i].set_alpha(0.8)
        plt.annotate(f'7/10 = {freq:.1f}', 
                    xy=(i, freq), xytext=(0, 10),
                    textcoords='offset points', ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))

plt.grid(True, alpha=0.3)

# Plot 2: Feature selection heatmap
plt.subplot(2, 2, 2)
heatmap_data = feature_selection_matrix[:, sorted_indices]
sns.heatmap(heatmap_data, 
            xticklabels=[f'F{i+1}' for i in sorted_indices],
            yticklabels=[f'B{i+1}' for i in range(n_bootstrap)],
            cmap='Blues', cbar_kws={'label': 'Selected (1) / Not Selected (0)'})
plt.title('Feature Selection Pattern Across Bootstrap Samples')
plt.xlabel('Features (sorted by frequency)')
plt.ylabel('Bootstrap Sample')

# Plot 3: Jaccard similarity distribution
plt.subplot(2, 2, 3)
plt.hist(jaccard_similarities, bins=10, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
plt.axvline(mean_jaccard, color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {mean_jaccard:.3f}')
plt.xlabel('Jaccard Similarity')
plt.ylabel('Frequency')
plt.title('Distribution of Jaccard Similarities')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Feature importance consistency
plt.subplot(2, 2, 4)
feature_importance_array = np.array(feature_importance_bootstrap)
feature_importance_mean = np.mean(feature_importance_array, axis=0)
feature_importance_std = np.std(feature_importance_array, axis=0)

# Sort by mean importance
sorted_imp_indices = np.argsort(feature_importance_mean)[::-1]
sorted_imp_means = feature_importance_mean[sorted_imp_indices]
sorted_imp_stds = feature_importance_std[sorted_imp_indices]
sorted_imp_names = [feature_names[i] for i in sorted_imp_indices]

plt.errorbar(range(len(sorted_imp_means)), sorted_imp_means, 
            yerr=sorted_imp_stds, fmt='o-', capsize=5, capthick=2,
            color='orange', ecolor='red', alpha=0.7)
plt.xlabel('Features')
plt.ylabel('Feature Importance')
plt.title('Feature Importance Consistency Across Bootstrap Samples')
plt.xticks(range(len(sorted_imp_names)), sorted_imp_names, rotation=45, ha='right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_selection_stability_analysis.png'), 
            dpi=300, bbox_inches='tight')

# 7. Design stability evaluation protocol
print("\n7. Stability Evaluation Protocol Design...")
print("Step 1: Data Preparation")
print("  - Use bootstrap sampling with replacement")
print("  - Maintain consistent sample size (80% of original data)")
print("  - Ensure class balance preservation")

print("\nStep 2: Feature Selection")
print("  - Apply same feature selection method across all samples")
print("  - Use consistent parameters (e.g., k=8 for top-k selection)")
print("  - Record both selected features and their importance scores")

print("\nStep 3: Stability Measurement")
print("  - Calculate feature selection frequency")
print("  - Compute pairwise Jaccard similarities")
print("  - Analyze feature importance consistency")

print("\nStep 4: Interpretation")
print("  - High frequency (>0.8): Very stable features")
print("  - Medium frequency (0.5-0.8): Moderately stable features")
print("  - Low frequency (<0.5): Unstable features")

# 8. Factors affecting stability
print("\n8. Factors affecting feature selection stability...")
print("- Dataset size: Larger datasets generally more stable")
print("- Feature correlation: Highly correlated features reduce stability")
print("- Feature selection method: Some methods more stable than others")
print("- Data quality: Noise and outliers reduce stability")
print("- Class imbalance: Can affect feature importance rankings")

# 9. Summary statistics
print("\n9. Summary Statistics...")
print(f"Most stable feature: {sorted_names[0]} (frequency: {sorted_frequencies[0]:.3f})")
print(f"Least stable feature: {sorted_names[-1]} (frequency: {sorted_frequencies[-1]:.3f})")
print(f"Features with frequency > 0.7: {np.sum(sorted_frequencies > 0.7)}")
print(f"Features with frequency < 0.3: {np.sum(sorted_frequencies < 0.3)}")

# Save detailed results to CSV
results_df = pd.DataFrame({
    'Feature': sorted_names,
    'Selection_Frequency': sorted_frequencies,
    'Stability_Score': sorted_frequencies,
    'Stability_Category': ['High' if f > 0.7 else 'Medium' if f > 0.3 else 'Low' for f in sorted_frequencies]
})

results_df.to_csv(os.path.join(save_dir, 'stability_analysis_results.csv'), index=False)

print(f"\nResults saved to: {save_dir}")
print("Analysis complete!")
