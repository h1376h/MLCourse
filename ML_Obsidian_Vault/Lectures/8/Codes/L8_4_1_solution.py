import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_4_Quiz_1")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=== Distance Measures in Feature Selection - Step by Step Solution ===\n")

# Step 1: Generate synthetic dataset with different feature qualities
print("Step 1: Generating synthetic dataset with different feature qualities")
np.random.seed(42)

# Create dataset with 4 features: 2 good separators, 1 moderate, 1 poor
X, y = make_classification(
    n_samples=200,
    n_features=4,
    n_informative=2,  # Only 2 features are actually informative
    n_redundant=1,    # 1 redundant feature
    n_repeated=0,
    n_clusters_per_class=1,
    n_classes=2,
    random_state=42
)

# Add some noise to make it more realistic
X += np.random.normal(0, 0.1, X.shape)

feature_names = ['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4']
print(f"Dataset shape: {X.shape}")
print(f"Features: {feature_names}")
print(f"Classes: {np.unique(y)}")
print(f"Class distribution: {np.bincount(y)}")
print()

# Step 2: Calculate distance measures for each feature
print("Step 2: Calculating distance measures for each feature")

def calculate_distance_measures(X, y):
    """Calculate various distance measures for feature evaluation"""
    n_features = X.shape[1]
    results = {}
    
    for i in range(n_features):
        feature = X[:, i]
        
        # Separate classes
        class_0 = feature[y == 0]
        class_1 = feature[y == 1]
        
        # Calculate means and standard deviations
        mean_0, mean_1 = np.mean(class_0), np.mean(class_1)
        std_0, std_1 = np.std(class_0), np.std(class_1)
        
        # 1. Euclidean Distance between class means
        euclidean_dist = abs(mean_1 - mean_0)
        
        # 2. Manhattan Distance between class means
        manhattan_dist = abs(mean_1 - mean_0)
        
        # 3. Mahalanobis Distance (simplified version)
        # Using pooled standard deviation
        pooled_std = np.sqrt(((len(class_0) - 1) * std_0**2 + (len(class_1) - 1) * std_1**2) / (len(class_0) + len(class_1) - 2))
        mahalanobis_dist = abs(mean_1 - mean_0) / pooled_std if pooled_std > 0 else 0
        
        # 4. Fisher's Discriminant Ratio (FDR)
        fdr = (mean_1 - mean_0)**2 / (std_0**2 + std_1**2) if (std_0**2 + std_1**2) > 0 else 0
        
        # 5. Bhattacharyya Distance
        bhattacharyya = 0.25 * ((mean_1 - mean_0)**2 / (std_0**2 + std_1**2)) + 0.5 * np.log((std_0**2 + std_1**2) / (2 * std_0 * std_1)) if (std_0 > 0 and std_1 > 0 and (std_0**2 + std_1**2) > 0) else 0
        
        results[feature_names[i]] = {
            'Euclidean': euclidean_dist,
            'Manhattan': manhattan_dist,
            'Mahalanobis': mahalanobis_dist,
            'FDR': fdr,
            'Bhattacharyya': bhattacharyya,
            'Mean_Class_0': mean_0,
            'Mean_Class_1': mean_1,
            'Std_Class_0': std_0,
            'Std_Class_1': std_1
        }
    
    return results

distance_results = calculate_distance_measures(X, y)

# Display results
print("Distance Measures for each feature:")
print("-" * 80)
for feature, metrics in distance_results.items():
    print(f"\n{feature}:")
    print(f"  Class 0: Mean = {metrics['Mean_Class_0']:.3f}, Std = {metrics['Std_Class_0']:.3f}")
    print(f"  Class 1: Mean = {metrics['Mean_Class_1']:.3f}, Std = {metrics['Std_Class_1']:.3f}")
    print(f"  Euclidean Distance: {metrics['Euclidean']:.3f}")
    print(f"  Manhattan Distance: {metrics['Manhattan']:.3f}")
    print(f"  Mahalanobis Distance: {metrics['Mahalanobis']:.3f}")
    print(f"  Fisher's Discriminant Ratio: {metrics['FDR']:.3f}")
    print(f"  Bhattacharyya Distance: {metrics['Bhattacharyya']:.3f}")
print()

# Step 3: Visualize feature distributions and separability
print("Step 3: Creating visualizations of feature distributions and separability")

# Create subplots for each feature
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Feature Distributions and Class Separability', fontsize=16)

for i, feature_name in enumerate(feature_names):
    row, col = i // 2, i % 2
    ax = axes[row, col]
    
    # Plot histograms for each class
    class_0_data = X[y == 0, i]
    class_1_data = X[y == 1, i]
    
    ax.hist(class_0_data, bins=20, alpha=0.7, label='Class 0', color='blue', density=True)
    ax.hist(class_1_data, bins=20, alpha=0.7, label='Class 1', color='red', density=True)
    
    # Add vertical lines for means
    mean_0 = distance_results[feature_name]['Mean_Class_0']
    mean_1 = distance_results[feature_name]['Mean_Class_1']
    ax.axvline(mean_0, color='blue', linestyle='--', alpha=0.8, label=f'Class 0 Mean: {mean_0:.2f}')
    ax.axvline(mean_1, color='red', linestyle='--', alpha=0.8, label=f'Class 1 Mean: {mean_1:.2f}')
    
    # Add distance measure information
    euclidean = distance_results[feature_name]['Euclidean']
    fdr = distance_results[feature_name]['FDR']
    ax.set_title(f'{feature_name}\nEuclidean Dist: {euclidean:.3f}, FDR: {fdr:.3f}')
    ax.set_xlabel('Feature Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
print("Saved: feature_distributions.png")

# Step 4: Create distance measures comparison heatmap
print("\nStep 4: Creating distance measures comparison heatmap")

# Prepare data for heatmap
distance_metrics = ['Euclidean', 'Manhattan', 'Mahalanobis', 'FDR', 'Bhattacharyya']
heatmap_data = []

for feature in feature_names:
    row = [distance_results[feature][metric] for metric in distance_metrics]
    heatmap_data.append(row)

# Create heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, 
            xticklabels=distance_metrics, 
            yticklabels=feature_names,
            annot=True, 
            fmt='.3f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Distance Measure Value'})

plt.title('Distance Measures Comparison Across Features')
plt.xlabel('Distance Measure Type')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'distance_measures_heatmap.png'), dpi=300, bbox_inches='tight')
print("Saved: distance_measures_heatmap.png")

# Step 5: Feature ranking based on different distance measures
print("\nStep 5: Feature ranking based on different distance measures")

# Create ranking dataframe
ranking_data = []
for feature in feature_names:
    row = [feature]
    for metric in distance_metrics:
        row.append(distance_results[feature][metric])
    ranking_data.append(row)

ranking_df = pd.DataFrame(ranking_data, columns=['Feature'] + distance_metrics)
print("Feature ranking based on distance measures:")
print(ranking_df.to_string(index=False, float_format='%.3f'))
print()

# Rank features for each metric
print("Feature rankings (1 = best, 4 = worst):")
print("-" * 60)
for metric in distance_metrics:
    # Sort by metric value (higher is better for most distance measures)
    sorted_features = sorted(feature_names, key=lambda x: distance_results[x][metric], reverse=True)
    print(f"{metric}: {', '.join([f'{f}({i+1})' for i, f in enumerate(sorted_features)])}")
print()

# Step 6: Compare with sklearn's F-test
print("Step 6: Comparing with sklearn's F-test implementation")

# Use sklearn's f_classif for comparison
f_scores, p_values = f_classif(X, y)

print("Sklearn F-test results:")
print("-" * 40)
for i, feature in enumerate(feature_names):
    print(f"{feature}: F-score = {f_scores[i]:.3f}, p-value = {p_values[i]:.3f}")

# Create comparison plot
plt.figure(figsize=(12, 6))
x_pos = np.arange(len(feature_names))
width = 0.35

# Our FDR calculation
our_fdr = [distance_results[f]['FDR'] for f in feature_names]

plt.bar(x_pos - width/2, our_fdr, width, label='Our FDR Calculation', alpha=0.8)
plt.bar(x_pos + width/2, f_scores, width, label='Sklearn F-scores', alpha=0.8)

plt.xlabel('Features')
plt.ylabel('Score')
plt.title('Comparison: Our FDR vs Sklearn F-scores')
plt.xticks(x_pos, feature_names)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'fdr_comparison.png'), dpi=300, bbox_inches='tight')
print("Saved: fdr_comparison.png")

# Step 7: Feature selection demonstration
print("\nStep 7: Demonstrating feature selection using distance measures")

# Select top 2 features using different criteria
print("Top 2 features selected by different criteria:")

# By Euclidean distance
euclidean_ranking = sorted(feature_names, key=lambda x: distance_results[x]['Euclidean'], reverse=True)
print(f"Euclidean Distance: {euclidean_ranking[:2]}")

# By FDR
fdr_ranking = sorted(feature_names, key=lambda x: distance_results[x]['FDR'], reverse=True)
print(f"Fisher's Discriminant Ratio: {fdr_ranking[:2]}")

# By Mahalanobis distance
mahal_ranking = sorted(feature_names, key=lambda x: distance_results[x]['Mahalanobis'], reverse=True)
print(f"Mahalanobis Distance: {mahal_ranking[:2]}")

# By sklearn F-test
sklearn_ranking = [feature_names[i] for i in np.argsort(f_scores)[::-1]]
print(f"Sklearn F-test: {sklearn_ranking[:2]}")

# Step 8: Create final summary visualization
print("\nStep 8: Creating final summary visualization")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Distance measures comparison
x = np.arange(len(feature_names))
width = 0.15

for i, metric in enumerate(distance_metrics):
    values = [distance_results[f][metric] for f in feature_names]
    ax1.bar(x + i*width, values, width, label=metric, alpha=0.8)

ax1.set_xlabel('Features')
ax1.set_ylabel('Distance Measure Value')
ax1.set_title('Distance Measures Comparison')
ax1.set_xticks(x + width * 2)
ax1.set_xticklabels(feature_names)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Feature importance ranking
importance_scores = []
for feature in feature_names:
    # Average rank across all metrics
    ranks = []
    for metric in distance_metrics:
        sorted_features = sorted(feature_names, key=lambda x: distance_results[x][metric], reverse=True)
        ranks.append(sorted_features.index(feature) + 1)
    avg_rank = np.mean(ranks)
    importance_scores.append(1/avg_rank)  # Invert rank so lower rank = higher importance

ax2.bar(feature_names, importance_scores, color=['green', 'blue', 'orange', 'red'], alpha=0.7)
ax2.set_xlabel('Features')
ax2.set_ylabel('Importance Score (1/Average Rank)')
ax2.set_title('Feature Importance Based on Average Ranking')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'final_summary.png'), dpi=300, bbox_inches='tight')
print("Saved: final_summary.png")

# Step 9: Summary and conclusions
print("\n" + "="*80)
print("SUMMARY AND CONCLUSIONS")
print("="*80)

print("\n1. Purpose of Distance Measures in Feature Selection:")
print("   - Evaluate how well features separate different classes")
print("   - Provide quantitative measures of feature quality")
print("   - Help rank features for selection")

print("\n2. Relationship to Class Separability:")
print("   - Higher distance values indicate better class separation")
print("   - Features with larger between-class distances and smaller within-class variances are preferred")
print("   - Distance measures help identify discriminative features")

print("\n3. Most Common Distance Measures:")
print("   - Euclidean Distance: Simple geometric distance between class means")
print("   - Fisher's Discriminant Ratio (FDR): Most widely used in practice")
print("   - Mahalanobis Distance: Accounts for feature correlations")
print("   - Bhattacharyya Distance: Information-theoretic measure")

print("\n4. Well-Separated Classes Indicate:")
print("   - Features have strong discriminative power")
print("   - Good potential for classification performance")
print("   - Lower risk of overfitting")

print("\n5. Distance Measures vs Other Criteria:")
print("   - Distance measures: Focus on class separability")
print("   - Correlation-based: Focus on feature independence")
print("   - Information-based: Focus on mutual information")
print("   - Wrapper methods: Focus on actual classification performance")

print(f"\nAll visualizations saved to: {save_dir}")
print("="*80)
