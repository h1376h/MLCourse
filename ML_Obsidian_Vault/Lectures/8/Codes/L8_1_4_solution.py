import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_1_Quiz_4")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting and configure for file saving only
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300
# Disable interactive display - only save to files
plt.ioff()

print("Feature Selection vs Feature Extraction Analysis")
print("=" * 60)

# ============================================================================
# Section 1: Conceptual Comparison
# ============================================================================
print("\n1. CONCEPTUAL COMPARISON")
print("-" * 40)

# Create a visual comparison table
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Feature Selection visualization
ax1.text(0.5, 0.9, 'Feature Selection', ha='center', va='center', 
         fontsize=16, fontweight='bold', transform=ax1.transAxes)

# Draw original features
features = ['$X_1$', '$X_2$', '$X_3$', '$X_4$', '$X_5$']
selected = [True, False, True, False, True]  # Example selection
colors = ['green' if sel else 'red' for sel in selected]

for i, (feat, color) in enumerate(zip(features, colors)):
    ax1.add_patch(plt.Rectangle((i*0.15 + 0.1, 0.6), 0.1, 0.15, 
                               facecolor=color, alpha=0.7, edgecolor='black'))
    ax1.text(i*0.15 + 0.15, 0.675, feat, ha='center', va='center', fontsize=12)

ax1.arrow(0.5, 0.5, 0, -0.15, head_width=0.03, head_length=0.02, 
          fc='black', ec='black', transform=ax1.transAxes)

# Draw selected features
selected_idx = [0, 2, 4]
for j, i in enumerate(selected_idx):
    ax1.add_patch(plt.Rectangle((j*0.2 + 0.2, 0.15), 0.1, 0.15, 
                               facecolor='green', alpha=0.7, edgecolor='black'))
    ax1.text(j*0.2 + 0.25, 0.225, features[i], ha='center', va='center', fontsize=12)

ax1.text(0.5, 0.05, 'Subset of Original Features', ha='center', va='center', 
         fontsize=12, transform=ax1.transAxes)

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')

# Feature Extraction visualization
ax2.text(0.5, 0.9, 'Feature Extraction', ha='center', va='center', 
         fontsize=16, fontweight='bold', transform=ax2.transAxes)

# Draw original features
for i, feat in enumerate(features):
    ax2.add_patch(plt.Rectangle((i*0.15 + 0.1, 0.6), 0.1, 0.15, 
                               facecolor='blue', alpha=0.7, edgecolor='black'))
    ax2.text(i*0.15 + 0.15, 0.675, feat, ha='center', va='center', fontsize=12)

ax2.arrow(0.5, 0.5, 0, -0.15, head_width=0.03, head_length=0.02, 
          fc='black', ec='black', transform=ax2.transAxes)

# Draw transformed features
new_features = ['$PC_1$', '$PC_2$', '$PC_3$']
for j, feat in enumerate(new_features):
    ax2.add_patch(plt.Rectangle((j*0.2 + 0.2, 0.15), 0.1, 0.15, 
                               facecolor='purple', alpha=0.7, edgecolor='black'))
    ax2.text(j*0.2 + 0.25, 0.225, feat, ha='center', va='center', fontsize=12)

ax2.text(0.5, 0.05, 'Linear Combinations of Original Features', ha='center', va='center', 
         fontsize=12, transform=ax2.transAxes)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'selection_vs_extraction_concept.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Section 2: PCA Variance Calculation Example
# ============================================================================
print("\n2. PCA VARIANCE CALCULATION")
print("-" * 40)

# Given eigenvalues from the problem
eigenvalues = np.array([50, 30, 15, 5])
total_variance = np.sum(eigenvalues)
cumulative_variance = np.cumsum(eigenvalues)
cumulative_variance_ratio = cumulative_variance / total_variance

print(f"Given eigenvalues: {eigenvalues}")
print(f"Total variance: {total_variance}")
print()

# Calculate cumulative variance explained
print("Principal Component Analysis:")
for i, (eigenval, cum_var, cum_ratio) in enumerate(zip(eigenvalues, cumulative_variance, cumulative_variance_ratio)):
    print(f"PC{i+1}: λ = {eigenval:2d}, Cumulative Variance = {cum_var:3d}, Ratio = {cum_ratio:.3f} ({cum_ratio*100:.1f}%)")

# Find number of components for 95% variance
target_variance = 0.95
n_components_needed = np.argmax(cumulative_variance_ratio >= target_variance) + 1
print(f"\nTo retain {target_variance*100:.0f}% of variance, need {n_components_needed} principal components")
print(f"Actual variance retained: {cumulative_variance_ratio[n_components_needed-1]*100:.1f}%")

# Visualize eigenvalues and cumulative variance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Individual eigenvalues
ax1.bar(range(1, len(eigenvalues)+1), eigenvalues, alpha=0.7, color='skyblue', edgecolor='navy')
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Eigenvalue ($\\lambda_i$)')
ax1.set_title('Individual Eigenvalues')
ax1.grid(True, alpha=0.3)
for i, val in enumerate(eigenvalues):
    ax1.text(i+1, val + 1, f'{val}', ha='center', va='bottom', fontweight='bold')

# Cumulative variance explained
ax2.plot(range(1, len(eigenvalues)+1), cumulative_variance_ratio*100, 'o-', 
         linewidth=2, markersize=8, color='darkred')
ax2.axhline(y=95, color='green', linestyle='--', alpha=0.7, label='95% Threshold')
ax2.axvline(x=n_components_needed, color='green', linestyle='--', alpha=0.7)
ax2.fill_between(range(1, n_components_needed+1), 0, 
                [cumulative_variance_ratio[i]*100 for i in range(n_components_needed)], 
                alpha=0.3, color='green', label='Selected Components')
ax2.set_xlabel('Number of Principal Components')
ax2.set_ylabel('Cumulative Variance Explained (\\%)')
ax2.set_title('Cumulative Variance Explained')
ax2.grid(True, alpha=0.3)
ax2.legend()
for i, val in enumerate(cumulative_variance_ratio):
    ax2.text(i+1, val*100 + 2, f'{val*100:.1f}\\%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'pca_variance_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Section 3: Practical Example with Real Data
# ============================================================================
print("\n3. PRACTICAL COMPARISON WITH IRIS DATASET")
print("-" * 50)

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names

print(f"Original dataset shape: {X.shape}")
print(f"Feature names: {feature_names}")

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# ============================================================================
# Feature Selection Approach
# ============================================================================
print("\n3.1 Feature Selection Results:")
print("-" * 30)

# Use SelectKBest with f_classif
selector = SelectKBest(score_func=f_classif, k=2)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Get selected feature indices and scores
selected_features = selector.get_support(indices=True)
feature_scores = selector.scores_

print("Feature Selection (SelectKBest with f_classif):")
for i, (feat_idx, score) in enumerate(zip(range(len(feature_names)), feature_scores)):
    selected = "✓" if feat_idx in selected_features else "✗"
    print(f"  {selected} {feature_names[feat_idx]}: F-score = {score:.3f}")

print(f"\nSelected features: {[feature_names[i] for i in selected_features]}")
print(f"Reduced shape: {X_train_selected.shape}")

# Train classifier on selected features
clf_selection = LogisticRegression(random_state=42)
clf_selection.fit(X_train_selected, y_train)
y_pred_selection = clf_selection.predict(X_test_selected)
accuracy_selection = accuracy_score(y_test, y_pred_selection)
print(f"Classification accuracy: {accuracy_selection:.3f}")

# ============================================================================
# Feature Extraction Approach
# ============================================================================
print("\n3.2 Feature Extraction Results:")
print("-" * 30)

# Apply PCA
pca = PCA(n_components=2, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("Feature Extraction (PCA):")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.3f}")

print("\nPrincipal Component Loadings:")
components_df = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
    index=feature_names
)
print(components_df.round(3))

print(f"\nReduced shape: {X_train_pca.shape}")

# Train classifier on PCA features
clf_pca = LogisticRegression(random_state=42)
clf_pca.fit(X_train_pca, y_train)
y_pred_pca = clf_pca.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test, y_pred_pca)
print(f"Classification accuracy: {accuracy_pca:.3f}")

# ============================================================================
# Visualization of Feature Selection vs Extraction
# ============================================================================
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Original feature importance
ax1.bar(range(len(feature_names)), feature_scores, alpha=0.7, color='lightblue', edgecolor='navy')
ax1.set_xlabel('Features')
ax1.set_ylabel('F-score')
ax1.set_title('Feature Importance (Selection)')
ax1.set_xticks(range(len(feature_names)))
ax1.set_xticklabels([name.replace(' ', '\n') for name in feature_names], rotation=45)
ax1.grid(True, alpha=0.3)

# Highlight selected features
for i, score in enumerate(feature_scores):
    color = 'green' if i in selected_features else 'lightblue'
    ax1.bar(i, score, alpha=0.7, color=color, edgecolor='navy')

# PCA component loadings heatmap
im = ax2.imshow(pca.components_, cmap='RdBu_r', aspect='auto')
ax2.set_xlabel('Original Features')
ax2.set_ylabel('Principal Components')
ax2.set_title('PCA Component Loadings (Extraction)')
ax2.set_xticks(range(len(feature_names)))
ax2.set_xticklabels([name.replace(' ', '\n') for name in feature_names], rotation=45)
ax2.set_yticks(range(pca.n_components_))
ax2.set_yticklabels([f'PC{i+1}' for i in range(pca.n_components_)])

# Add colorbar
cbar = plt.colorbar(im, ax=ax2)
cbar.set_label('Loading Value')

# Add loading values as text
for i in range(pca.n_components_):
    for j in range(len(feature_names)):
        text = ax2.text(j, i, f'{pca.components_[i, j]:.2f}', 
                       ha="center", va="center", color="black", fontweight='bold')

# Data visualization - Feature Selection
colors = ['red', 'green', 'blue']
for i, class_label in enumerate(np.unique(y)):
    mask = y_test == class_label
    ax3.scatter(X_test_selected[mask, 0], X_test_selected[mask, 1], 
               c=colors[i], label=f'Class {class_label}', alpha=0.7, s=60)

ax3.set_xlabel(f'{feature_names[selected_features[0]]}')
ax3.set_ylabel(f'{feature_names[selected_features[1]]}')
ax3.set_title(f'Selected Features Space\nAccuracy: {accuracy_selection:.3f}')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Data visualization - PCA
for i, class_label in enumerate(np.unique(y)):
    mask = y_test == class_label
    ax4.scatter(X_test_pca[mask, 0], X_test_pca[mask, 1], 
               c=colors[i], label=f'Class {class_label}', alpha=0.7, s=60)

ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
ax4.set_title(f'PCA Feature Space\nAccuracy: {accuracy_pca:.3f}')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_selection_vs_extraction_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Computational Cost and Interpretability Analysis
# ============================================================================
print("\n4. COMPUTATIONAL COST AND INTERPRETABILITY ANALYSIS")
print("-" * 55)

# Create comparison table
comparison_data = {
    'Aspect': [
        'Interpretability',
        'Original Features',
        'Dimensionality',
        'Information Loss',
        'Computational Cost',
        'Reversibility',
        'Noise Handling',
        'Feature Correlation'
    ],
    'Feature Selection': [
        'High - original features preserved',
        'Subset of original features',
        'Reduced (subset)',
        'Complete loss of unselected features',
        'Low - simple filtering/ranking',
        'Not applicable',
        'Limited - keeps noise in selected features',
        'Ignores feature correlations'
    ],
    'Feature Extraction': [
        'Low - transformed features hard to interpret',
        'Linear combinations of original features',
        'Reduced (transformation)',
        'Minimal if done properly',
        'Higher - requires transformation computation',
        'Possible with transformation matrix',
        'Good - combines information, reduces noise',
        'Considers feature correlations'
    ]
}

comparison_df = pd.DataFrame(comparison_data)

# Create a visual comparison chart
fig, ax = plt.subplots(figsize=(14, 8))

# Create a heatmap-style visualization of the comparison
aspects = comparison_data['Aspect']
methods = ['Feature Selection', 'Feature Extraction']

# Scoring system for visualization (higher is better)
scores = np.array([
    [5, 2],  # Interpretability
    [5, 3],  # Original Features  
    [4, 4],  # Dimensionality
    [2, 4],  # Information Loss (inverted - less loss is better)
    [5, 3],  # Computational Cost (inverted - lower cost is better)
    [1, 4],  # Reversibility
    [2, 4],  # Noise Handling
    [2, 4]   # Feature Correlation
])

im = ax.imshow(scores, cmap='RdYlGn', aspect='auto', vmin=1, vmax=5)

# Set ticks and labels
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods)
ax.set_yticks(range(len(aspects)))
ax.set_yticklabels(aspects)

# Add text annotations
for i in range(len(aspects)):
    for j in range(len(methods)):
        text = ax.text(j, i, scores[i, j], ha="center", va="center", 
                      color="white", fontweight='bold', fontsize=14)

ax.set_title('Feature Selection vs Extraction Comparison\n(5=Best, 1=Worst)', fontsize=14, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Score (1-5 scale)', rotation=270, labelpad=20)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'comparison_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

# Print detailed comparison table
print("\nDetailed Comparison Table:")
print("=" * 80)
for _, row in comparison_df.iterrows():
    print(f"\n{row['Aspect']}:")
    print(f"  Selection: {row['Feature Selection']}")
    print(f"  Extraction: {row['Feature Extraction']}")

# ============================================================================
# When to Use Each Approach
# ============================================================================
print("\n5. WHEN TO USE EACH APPROACH")
print("-" * 35)

use_cases = {
    'Feature Selection': [
        'Need to maintain interpretability of results',
        'Working with domain experts who need to understand features',
        'Regulatory requirements for explainable models',
        'Limited computational resources',
        'Features have clear physical/business meaning',
        'Want to identify most important original features',
        'Working with sparse data where feature meaning matters'
    ],
    'Feature Extraction': [
        'High-dimensional data with many correlated features',
        'Computational efficiency is more important than interpretability',
        'Data is noisy and need to reduce noise',
        'Want to visualize high-dimensional data',
        'Features are highly correlated (multicollinearity)',
        'Need to preserve maximum information with fewer dimensions',
        'Working with image, signal, or text data'
    ]
}

for approach, cases in use_cases.items():
    print(f"\n{approach}:")
    for i, case in enumerate(cases, 1):
        print(f"  {i}. {case}")

print(f"\nAnalysis complete! All visualizations saved to: {save_dir}")
