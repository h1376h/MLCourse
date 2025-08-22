import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_4_Quiz_4")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=== Question 4: Dependency Measures for Feature Selection ===\n")

# 1. Generate synthetic dataset for demonstration
np.random.seed(42)
n_samples = 1000

# Create features with different correlation patterns
x1 = np.random.normal(0, 1, n_samples)  # Feature 1: independent
x2 = 0.8 * x1 + np.random.normal(0, 0.6, n_samples)  # Feature 2: correlated with x1
x3 = np.random.normal(0, 1, n_samples)  # Feature 3: independent
x4 = 0.6 * x1 + 0.4 * x3 + np.random.normal(0, 0.7, n_samples)  # Feature 4: correlated with both

# Create target variable (class)
y = np.where(x1 + 0.5 * x3 + np.random.normal(0, 0.3, n_samples) > 0, 1, 0)

# Create DataFrame
data = pd.DataFrame({
    'Feature_1': x1,
    'Feature_2': x2,
    'Feature_3': x3,
    'Feature_4': x4,
    'Target': y
})

print("1. GOAL OF DEPENDENCY MEASURES")
print("=" * 50)
print("The goal is to select features that are:")
print("- Highly correlated with the target class (high predictive power)")
print("- Uncorrelated with each other (low redundancy)")
print("- This maximizes information gain while minimizing redundancy\n")

print("2. MEASURING FEATURE-CLASS CORRELATION")
print("=" * 50)

# Calculate different correlation measures
correlation_measures = {}

# Pearson correlation
pearson_corr = {}
for col in ['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4']:
    corr, p_value = pearsonr(data[col], data['Target'])
    pearson_corr[col] = corr
    print(f"Pearson correlation {col} vs Target: {corr:.4f} (p-value: {p_value:.4f})")

# Spearman correlation
spearman_corr = {}
for col in ['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4']:
    corr, p_value = spearmanr(data[col], data['Target'])
    spearman_corr[col] = corr
    print(f"Spearman correlation {col} vs Target: {corr:.4f} (p-value: {p_value:.4f})")

# Mutual Information
mi_scores = mutual_info_classif(data[['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4']], data['Target'])
mi_dict = dict(zip(['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4'], mi_scores))
print("\nMutual Information scores:")
for feature, mi in mi_dict.items():
    print(f"{feature}: {mi:.4f}")

# F-statistic
f_scores, f_pvalues = f_classif(data[['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4']], data['Target'])
f_dict = dict(zip(['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4'], f_scores))
print("\nF-statistic scores:")
for feature, f_score in f_dict.items():
    print(f"{feature}: {f_score:.4f}")

correlation_measures['pearson'] = pearson_corr
correlation_measures['spearman'] = spearman_corr
correlation_measures['mutual_info'] = mi_dict
correlation_measures['f_statistic'] = f_dict

print("\n3. MEASURING FEATURE-FEATURE CORRELATION")
print("=" * 50)

# Calculate feature-feature correlation matrix
feature_cols = ['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4']
correlation_matrix = data[feature_cols].corr()

print("Feature-Feature Correlation Matrix:")
print(correlation_matrix.round(4))

# Extract specific correlations mentioned in question
print(f"\nSpecific correlations:")
print(f"Feature A (Feature_1) vs Target: {pearson_corr['Feature_1']:.4f}")
print(f"Feature A (Feature_1) vs Feature B (Feature_2): {correlation_matrix.loc['Feature_1', 'Feature_2']:.4f}")

print("\n4. EVALUATING FEATURE A vs FEATURE B")
print("=" * 50)
print("Given: Feature A has 0.8 correlation with class and 0.1 with Feature B")
print("Analysis:")
print("- High class correlation (0.8): EXCELLENT - Feature A is highly predictive")
print("- Low feature correlation (0.1): EXCELLENT - Features A and B are nearly independent")
print("- This combination is IDEAL for feature selection!")
print("- Feature A provides unique information not captured by Feature B\n")

print("5. DESIGNING DEPENDENCY-BASED SELECTION STRATEGY")
print("=" * 50)

# Create a comprehensive feature selection strategy
def dependency_based_feature_selection(data, target_col, threshold_correlation=0.7, threshold_redundancy=0.5):
    """
    Dependency-based feature selection strategy
    """
    feature_cols = [col for col in data.columns if col != target_col]
    
    # Step 1: Calculate feature-class correlations
    class_correlations = {}
    for col in feature_cols:
        corr, _ = pearsonr(data[col], data[target_col])
        class_correlations[col] = abs(corr)
    
    # Step 2: Calculate feature-feature correlations
    feature_correlations = data[feature_cols].corr().abs()
    
    # Step 3: Select features based on criteria
    selected_features = []
    rejected_features = []
    
    # Sort features by class correlation (descending)
    sorted_features = sorted(class_correlations.items(), key=lambda x: x[1], reverse=True)
    
    for feature, class_corr in sorted_features:
        if class_corr < threshold_correlation:
            rejected_features.append((feature, f"Low class correlation: {class_corr:.4f}"))
            continue
            
        # Check redundancy with already selected features
        is_redundant = False
        for selected_feature in selected_features:
            redundancy = feature_correlations.loc[feature, selected_feature]
            if redundancy > threshold_redundancy:
                rejected_features.append((feature, f"High redundancy with {selected_feature}: {redundancy:.4f}"))
                is_redundant = True
                break
        
        if not is_redundant:
            selected_features.append(feature)
            print(f"✓ Selected {feature}: Class correlation = {class_corr:.4f}")
    
    return selected_features, rejected_features

# Apply the strategy
print("Applying dependency-based selection strategy:")
print("Thresholds: Class correlation > 0.3, Feature redundancy < 0.7")
selected, rejected = dependency_based_feature_selection(data, 'Target', threshold_correlation=0.3, threshold_redundancy=0.7)

print(f"\nSelected features: {selected}")
print(f"Rejected features: {len(rejected)}")

if rejected:
    print("\nRejection reasons:")
    for feature, reason in rejected:
        print(f"  {feature}: {reason}")

# Create visualizations
print("\nGenerating visualizations...")

# Visualization 1: Feature-Class Correlation Comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Feature Selection Analysis: Dependency Measures', fontsize=16)

# Plot 1: Pearson vs Spearman correlation
features = list(pearson_corr.keys())
pearson_vals = [pearson_corr[f] for f in features]
spearman_vals = [spearman_corr[f] for f in features]

x = np.arange(len(features))
width = 0.35

axes[0, 0].bar(x - width/2, pearson_vals, width, label='Pearson', alpha=0.8, color='skyblue')
axes[0, 0].bar(x + width/2, spearman_vals, width, label='Spearman', alpha=0.8, color='lightcoral')
axes[0, 0].set_xlabel('Features')
axes[0, 0].set_ylabel('Correlation Coefficient')
axes[0, 0].set_title('Feature-Class Correlation: Pearson vs Spearman')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(features)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Mutual Information vs F-statistic
mi_vals = [mi_dict[f] for f in features]
f_vals = [f_dict[f] for f in features]

# Normalize F-statistic for comparison
f_vals_norm = np.array(f_vals) / max(f_vals)

axes[0, 1].bar(x - width/2, mi_vals, width, label='Mutual Information', alpha=0.8, color='lightgreen')
axes[0, 1].bar(x + width/2, f_vals_norm, width, label='F-statistic (normalized)', alpha=0.8, color='gold')
axes[0, 1].set_xlabel('Features')
axes[0, 1].set_ylabel('Score (normalized)')
axes[0, 1].set_title('Feature-Class Correlation: MI vs F-statistic')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(features)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Feature-Feature Correlation Heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, 
            ax=axes[1, 0], cbar_kws={'label': 'Correlation Coefficient'})
axes[1, 0].set_title('Feature-Feature Correlation Matrix')
axes[1, 0].set_xlabel('Features')
axes[1, 0].set_ylabel('Features')

# Plot 4: Scatter plots showing relationships
for i, feature in enumerate(features):
    row = i // 2
    col = i % 2
    if row < 2 and col < 2:
        continue
    
    if row == 1 and col == 1:  # Use the last subplot for scatter
        axes[1, 1].scatter(data[feature], data['Target'], alpha=0.6, s=20)
        axes[1, 1].set_xlabel(feature)
        axes[1, 1].set_ylabel('Target')
        axes[1, 1].set_title(f'{feature} vs Target')
        axes[1, 1].grid(True, alpha=0.3)
        break

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'dependency_measures_analysis.png'), dpi=300, bbox_inches='tight')

# Visualization 2: Feature Selection Process
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Feature Selection Process: Step-by-Step Analysis', fontsize=16)

# Step 1: Class correlation ranking
class_corr_ranked = sorted(pearson_corr.items(), key=lambda x: abs(x[1]), reverse=True)
features_ranked = [item[0] for item in class_corr_ranked]
corr_ranked = [abs(item[1]) for item in class_corr_ranked]

axes[0].barh(features_ranked, corr_ranked, color='lightblue', alpha=0.8)
axes[0].set_xlabel('Absolute Correlation with Target')
axes[0].set_title('Step 1: Rank Features by Class Correlation')
axes[0].grid(True, alpha=0.3)

# Step 2: Redundancy analysis
redundancy_scores = []
for feature in features_ranked:
    # Calculate average correlation with other features
    other_features = [f for f in features_ranked if f != feature]
    avg_corr = correlation_matrix.loc[feature, other_features].abs().mean()
    redundancy_scores.append(avg_corr)

axes[1].barh(features_ranked, redundancy_scores, color='lightcoral', alpha=0.8)
axes[1].set_xlabel('Average Correlation with Other Features')
axes[1].set_title('Step 2: Analyze Feature Redundancy')
axes[1].grid(True, alpha=0.3)

# Step 3: Final selection
selection_scores = []
for i, feature in enumerate(features_ranked):
    # Combine class correlation and redundancy (lower redundancy is better)
    score = corr_ranked[i] * (1 - redundancy_scores[i])
    selection_scores.append(score)

axes[2].barh(features_ranked, selection_scores, color='lightgreen', alpha=0.8)
axes[2].set_xlabel('Selection Score (Class Corr × Independence)')
axes[2].set_title('Step 3: Final Selection Score')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_selection_process.png'), dpi=300, bbox_inches='tight')

# Visualization 3: Decision Boundary Analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Feature Pair Analysis: Decision Boundaries', fontsize=16)

# Create meshgrid for decision boundaries
x_min, x_max = data[['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4']].min().min(), data[['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4']].max().max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(x_min, x_max, 100))

# Plot different feature pairs
feature_pairs = [('Feature_1', 'Feature_2'), ('Feature_1', 'Feature_3'), 
                 ('Feature_2', 'Feature_4'), ('Feature_3', 'Feature_4')]

for idx, (feat1, feat2) in enumerate(feature_pairs):
    row = idx // 2
    col = idx % 2
    
    # Scatter plot
    scatter = axes[row, col].scatter(data[feat1], data[feat2], c=data['Target'], 
                                    cmap='RdYlBu', alpha=0.6, s=20)
    
    # Add correlation info
    corr_val = correlation_matrix.loc[feat1, feat2]
    axes[row, col].set_title(f'{feat1} vs {feat2}\nCorrelation: {corr_val:.3f}')
    axes[row, col].set_xlabel(feat1)
    axes[row, col].set_ylabel(feat2)
    axes[row, col].grid(True, alpha=0.3)
    
    # Add colorbar
    plt.colorbar(scatter, ax=axes[row, col])

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_pair_analysis.png'), dpi=300, bbox_inches='tight')

print(f"\nAll visualizations saved to: {save_dir}")

# Summary of results
print("\n" + "="*60)
print("SUMMARY OF DEPENDENCY-BASED FEATURE SELECTION")
print("="*60)

print(f"\n1. GOAL ACHIEVED:")
print(f"   - Selected {len(selected)} features out of {len(features)}")
print(f"   - Features selected: {selected}")

print(f"\n2. FEATURE-CLASS CORRELATION ANALYSIS:")
for feature in features:
    print(f"   {feature}: Pearson={pearson_corr[feature]:.4f}, MI={mi_dict[feature]:.4f}")

print(f"\n3. FEATURE REDUNDANCY ANALYSIS:")
for feature in features:
    other_features = [f for f in features if f != feature]
    avg_corr = correlation_matrix.loc[feature, other_features].abs().mean()
    print(f"   {feature}: Average correlation with others = {avg_corr:.4f}")

print(f"\n4. OPTIMAL FEATURE COMBINATION:")
print(f"   - High class correlation: {[f for f in selected if abs(pearson_corr[f]) > 0.5]}")
print(f"   - Low redundancy: {[f for f in selected if correlation_matrix.loc[f, [s for s in selected if s != f]].abs().max() < 0.5]}")

print(f"\n5. RECOMMENDED SELECTION STRATEGY:")
print(f"   - Use mutual information for non-linear relationships")
print(f"   - Use correlation for linear relationships")
print(f"   - Set redundancy threshold at 0.5-0.7")
print(f"   - Prioritize features with high class correlation and low redundancy")
