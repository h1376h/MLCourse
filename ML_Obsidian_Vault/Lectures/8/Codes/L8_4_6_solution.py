import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.datasets import make_classification
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_4_Quiz_6")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=== Feature Selection Stability Analysis ===\n")

# 1. Generate synthetic dataset for demonstration
print("1. Generating synthetic dataset...")
np.random.seed(42)
X, y = make_classification(n_samples=1000, n_features=20, n_informative=8, 
                          n_redundant=5, n_clusters_per_class=1, random_state=42)

feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
print(f"Dataset shape: {X.shape}")
print(f"Number of informative features: 8")
print(f"Number of redundant features: 5")
print(f"Number of noise features: 7\n")

# 2. Demonstrate different feature selection methods
print("2. Applying different feature selection methods...")

# Method 1: Statistical tests (F-test)
print("Method 1: Statistical tests (F-test)")
selector_f = SelectKBest(score_func=f_classif, k=10)
X_f_selected = selector_f.fit_transform(X, y)
f_scores = selector_f.scores_
f_pvalues = selector_f.pvalues_

# Method 2: Recursive Feature Elimination
print("Method 2: Recursive Feature Elimination (RFE)")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
selector_rfe = RFE(estimator=rf, n_features_to_select=10)
X_rfe_selected = selector_rfe.fit_transform(X, y)
rfe_ranking = selector_rfe.ranking_

# Method 3: Random Forest feature importance
print("Method 3: Random Forest feature importance")
rf.fit(X, y)
rf_importance = rf.feature_importances_

print("Feature selection methods applied successfully!\n")

# 3. Cross-validation stability analysis
print("3. Performing cross-validation stability analysis...")
n_folds = 10
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Store feature selection results for each fold
fold_selections = {
    'f_test': [],
    'rfe': [],
    'rf_importance': []
}

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # F-test selection
    selector_f_fold = SelectKBest(score_func=f_classif, k=10)
    selector_f_fold.fit(X_train, y_train)
    f_selected = selector_f_fold.get_support()
    fold_selections['f_test'].append(f_selected)
    
    # RFE selection
    selector_rfe_fold = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=42), 
                           n_features_to_select=10)
    selector_rfe_fold.fit(X_train, y_train)
    rfe_selected = selector_rfe_fold.get_support()
    fold_selections['rfe'].append(rfe_selected)
    
    # RF importance selection
    rf_fold = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_fold.fit(X_train, y_train)
    rf_importance_fold = rf_fold.feature_importances_
    # Select top 10 features
    top_10_idx = np.argsort(rf_importance_fold)[-10:]
    rf_selected = np.zeros(20, dtype=bool)
    rf_selected[top_10_idx] = True
    fold_selections['rf_importance'].append(rf_selected)

print(f"Cross-validation completed with {n_folds} folds\n")

# 4. Calculate stability measures
print("4. Calculating stability measures...")

def calculate_stability(fold_selections):
    """Calculate stability measures for feature selection"""
    n_folds = len(fold_selections)
    n_features = len(fold_selections[0])
    
    # Convert to numpy array for easier manipulation
    selections = np.array(fold_selections)
    
    # Calculate selection frequency for each feature
    selection_frequency = np.mean(selections, axis=0)
    
    # Calculate pairwise stability (Jaccard similarity between folds)
    pairwise_stabilities = []
    for i in range(n_folds):
        for j in range(i+1, n_folds):
            intersection = np.sum(selections[i] & selections[j])
            union = np.sum(selections[i] | selections[j])
            jaccard = intersection / union if union > 0 else 0
            pairwise_stabilities.append(jaccard)
    
    # Calculate overall stability measures
    mean_pairwise_stability = np.mean(pairwise_stabilities)
    
    # Calculate feature-wise stability
    feature_stability = selection_frequency
    
    return {
        'selection_frequency': selection_frequency,
        'pairwise_stabilities': pairwise_stabilities,
        'mean_pairwise_stability': mean_pairwise_stability,
        'feature_stability': feature_stability
    }

# Calculate stability for each method
stability_results = {}
for method, selections in fold_selections.items():
    stability_results[method] = calculate_stability(selections)
    print(f"{method.upper()}:")
    print(f"  Mean pairwise stability: {stability_results[method]['mean_pairwise_stability']:.4f}")
    print(f"  Feature selection frequencies: {stability_results[method]['selection_frequency']}")

print("\n5. Creating visualizations...")

# Visualization 1: Feature selection frequency comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Feature Selection Stability Analysis', fontsize=16)

# Plot 1: Feature selection frequency for each method
methods = ['f_test', 'rfe', 'rf_importance']
colors = ['blue', 'red', 'green']
labels = ['F-test', 'RFE', 'Random Forest']

for i, method in enumerate(methods):
    freq = stability_results[method]['selection_frequency']
    axes[0, 0].bar(np.arange(len(freq)) + i*0.25, freq, width=0.25, 
                   label=labels[i], alpha=0.7, color=colors[i])

axes[0, 0].set_xlabel('Feature Index')
axes[0, 0].set_ylabel('Selection Frequency')
axes[0, 0].set_title('Feature Selection Frequency Across Folds')
axes[0, 0].set_xticks(np.arange(20))
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Stability comparison between methods
stability_scores = [stability_results[method]['mean_pairwise_stability'] for method in methods]
bars = axes[0, 1].bar(labels, stability_scores, color=colors, alpha=0.7)
axes[0, 1].set_ylabel('Mean Pairwise Stability (Jaccard)')
axes[0, 1].set_title('Overall Stability Comparison')
axes[0, 1].set_ylim(0, 1)
axes[0, 1].grid(True, alpha=0.3)

# Add value labels on bars
for bar, score in zip(bars, stability_scores):
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{score:.3f}', ha='center', va='bottom')

# Plot 3: Heatmap of feature selection across folds for F-test
selection_matrix = np.array(fold_selections['f_test'])
im = axes[1, 0].imshow(selection_matrix, cmap='RdYlBu_r', aspect='auto')
axes[1, 0].set_xlabel('Feature Index')
axes[1, 0].set_ylabel('Fold Index')
axes[1, 0].set_title('Feature Selection Pattern Across Folds (F-test)')
axes[1, 0].set_yticks(range(n_folds))
axes[1, 0].set_xticks(range(0, 20, 2))
plt.colorbar(im, ax=axes[1, 0], label='Selected (1) / Not Selected (0)')

# Plot 4: Stability score distribution
all_stabilities = []
for method in methods:
    all_stabilities.extend(stability_results[method]['pairwise_stabilities'])

axes[1, 1].hist(all_stabilities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
axes[1, 1].set_xlabel('Pairwise Stability (Jaccard)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Distribution of Pairwise Stability Scores')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'stability_analysis_overview.png'), dpi=300, bbox_inches='tight')

# Visualization 2: Detailed stability analysis
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Detailed Stability Analysis by Method', fontsize=16)

for i, method in enumerate(methods):
    freq = stability_results[method]['selection_frequency']
    
    # Create bar plot with color coding based on stability
    colors_stability = ['red' if f < 0.5 else 'orange' if f < 0.8 else 'green' for f in freq]
    bars = axes[i].bar(range(len(freq)), freq, color=colors_stability, alpha=0.7)
    
    axes[i].set_xlabel('Feature Index')
    axes[i].set_ylabel('Selection Frequency')
    axes[i].set_title(f'{labels[i]} Stability')
    axes[i].set_xticks(range(0, 20, 2))
    axes[i].set_ylim(0, 1)
    axes[i].grid(True, alpha=0.3)
    
    # Add stability score as text
    stability_score = stability_results[method]['mean_pairwise_stability']
    axes[i].text(0.02, 0.98, f'Stability: {stability_score:.3f}', 
                 transform=axes[i].transAxes, fontsize=12,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'detailed_stability_analysis.png'), dpi=300, bbox_inches='tight')

# Visualization 3: Stability vs. Feature Importance
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Stability vs. Feature Importance', fontsize=16)

for i, method in enumerate(methods):
    freq = stability_results[method]['selection_frequency']
    
    if method == 'f_test':
        importance = f_scores
        importance_name = 'F-score'
    elif method == 'rf_importance':
        importance = rf_importance
        importance_name = 'RF Importance'
    else:  # RFE
        importance = 1.0 / rfe_ranking  # Convert ranking to importance-like score
        importance_name = '1/Ranking'
    
    # Normalize importance to [0,1] for comparison
    importance_norm = (importance - importance.min()) / (importance.max() - importance.min())
    
    axes[i].scatter(importance_norm, freq, alpha=0.7, s=50)
    axes[i].set_xlabel(f'Normalized {importance_name}')
    axes[i].set_ylabel('Selection Frequency')
    axes[i].set_title(f'{labels[i]}: Stability vs. Importance')
    axes[i].grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = np.corrcoef(importance_norm, freq)[0, 1]
    axes[i].text(0.02, 0.98, f'Correlation: {correlation:.3f}', 
                 transform=axes[i].transAxes, fontsize=12,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'stability_vs_importance.png'), dpi=300, bbox_inches='tight')

print("Visualizations created successfully!")

# 6. Answer specific questions
print("\n=== ANSWERS TO SPECIFIC QUESTIONS ===")

print("\nQ1: What is feature selection stability and why is it important?")
print("Feature selection stability measures how consistent feature selection is across different data samples.")
print("Importance:")
print("- Ensures model robustness and generalizability")
print("- Reduces overfitting to specific data splits")
print("- Provides confidence in selected features")
print("- Critical for reproducible research")

print("\nQ2: How do you measure stability across different samples?")
print("Methods used in this analysis:")
print(f"- Pairwise Jaccard similarity between folds: {stability_results['f_test']['mean_pairwise_stability']:.4f}")
print(f"- Feature selection frequency across folds")
print(f"- Cross-validation with {n_folds} folds")

print("\nQ3: What causes instability in feature selection?")
print("Causes of instability:")
print("- Small sample sizes")
print("- High feature correlation")
print("- Noise in the data")
print("- Different feature selection algorithms")
print("- Random variations in data splits")

print("\nQ4: If a feature is selected in 8 out of 10 cross-validation folds, what's its stability score?")
print("Stability score = 8/10 = 0.8")
print("This indicates high stability for that feature.")

print("\nQ5: Design a stability measurement approach")
print("Our implemented approach:")
print("1. Perform k-fold cross-validation")
print("2. Apply feature selection in each fold")
print("3. Calculate pairwise Jaccard similarities between folds")
print("4. Compute mean pairwise stability")
print("5. Analyze feature-wise selection frequency")
print("6. Visualize stability patterns")

# 7. Summary statistics
print("\n=== SUMMARY STATISTICS ===")
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Cross-validation: {n_folds} folds")
print("\nStability scores by method:")
for method, label in zip(methods, labels):
    stability = stability_results[method]['mean_pairwise_stability']
    print(f"{label}: {stability:.4f}")

print(f"\nAll plots saved to: {save_dir}")
print("\nAnalysis complete!")
