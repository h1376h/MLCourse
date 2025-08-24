import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import accuracy_score
import pandas as pd
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_1_Quiz_13")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("Question 13: Model Stability and Feature Selection")
print("=" * 60)

# Set random seed for reproducibility
np.random.seed(42)

# 1. Generate synthetic dataset with noisy and useful features
print("\n1. Generating synthetic dataset with noisy and useful features...")
n_samples, n_features = 200, 100
n_informative = 20  # Only 20 features are truly useful
n_redundant = 30    # 30 redundant features
n_repeated = 10     # 10 repeated features
n_useless = n_features - n_informative - n_redundant - n_repeated  # 40 useless features

X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=n_informative,
    n_redundant=n_redundant,
    n_repeated=n_repeated,
    n_clusters_per_class=2,
    random_state=42
)

print(f"Dataset created: {n_samples} samples, {n_features} features")
print(f"- Informative features: {n_informative}")
print(f"- Redundant features: {n_redundant}")
print(f"- Repeated features: {n_repeated}")
print(f"- Useless features: {n_useless}")

# 2. Demonstrate how feature selection improves model stability
print("\n2. Analyzing model stability with different feature sets...")

# Define models to test
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

# Feature selection methods
selection_methods = {
    'All Features (100)': None,
    'SelectKBest (20)': SelectKBest(f_classif, k=20),
    'RFE (20)': RFE(RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=20)
}

# Cross-validation setup
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Results storage
results = {}

print("\nPerforming cross-validation with different feature sets...")
for model_name, model in models.items():
    print(f"\n{model_name}:")
    results[model_name] = {}
    
    for selection_name, selector in selection_methods.items():
        print(f"  Testing {selection_name}...")
        
        if selector is None:
            X_selected = X
            feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        else:
            if selection_name == 'RFE (20)':
                X_selected = selector.fit_transform(X, y)
                feature_names = [f"Feature_{i}" for i in range(X_selected.shape[1])]
            else:
                X_selected = selector.fit_transform(X, y)
                feature_names = [f"Feature_{i}" for i in range(X_selected.shape[1])]
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_selected, y, cv=cv, scoring='accuracy')
        
        results[model_name][selection_name] = {
            'cv_scores': cv_scores,
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'n_features': X_selected.shape[1]
        }
        
        print(f"    CV Scores: {cv_scores}")
        print(f"    Mean: {np.mean(cv_scores):.4f}")
        print(f"    Std: {np.std(cv_scores):.4f}")

# 3. Calculate stability improvement (Question 6)
print("\n3. Calculating stability improvement...")
print("Given CV scores:")
print("100 features: [0.75, 0.78, 0.72, 0.76, 0.74]")
print("20 features:  [0.73, 0.74, 0.72, 0.73, 0.74]")

cv_100 = np.array([0.75, 0.78, 0.72, 0.76, 0.74])
cv_20 = np.array([0.73, 0.74, 0.72, 0.73, 0.74])

std_100 = np.std(cv_100)
std_20 = np.std(cv_20)

stability_improvement = ((std_100 - std_20) / std_100) * 100

print(f"\nStandard deviation with 100 features: {std_100:.4f}")
print(f"Standard deviation with 20 features: {std_20:.4f}")
print(f"Stability improvement: {stability_improvement:.2f}% reduction in standard deviation")

# 4. Create comprehensive visualizations
print("\n4. Creating visualizations...")

# Figure 1: CV Score Comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Model Stability Analysis: Feature Selection Impact', fontsize=16, fontweight='bold')

# Plot 1: CV Scores Comparison
ax1 = axes[0, 0]
for model_name in models.keys():
    for selection_name in selection_methods.keys():
        scores = results[model_name][selection_name]['cv_scores']
        n_features = results[model_name][selection_name]['n_features']
        ax1.scatter([n_features] * len(scores), scores, alpha=0.7, 
                    label=f'{model_name} - {selection_name}', s=100)

ax1.set_xlabel('Number of Features')
ax1.set_ylabel('Cross-Validation Accuracy')
ax1.set_title('CV Scores vs Number of Features')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3)

# Plot 2: Standard Deviation Comparison
ax2 = axes[0, 1]
x_pos = np.arange(len(selection_methods))
width = 0.35

for i, model_name in enumerate(models.keys()):
    stds = [results[model_name][sel]['std_score'] for sel in selection_methods.keys()]
    ax2.bar(x_pos + i*width, stds, width, label=model_name, alpha=0.8)

ax2.set_xlabel('Feature Selection Method')
ax2.set_ylabel('Standard Deviation of CV Scores')
ax2.set_title('Model Stability (Lower is Better)')
ax2.set_xticks(x_pos + width/2)
ax2.set_xticklabels([sel.split('(')[0].strip() for sel in selection_methods.keys()], rotation=45)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Stability Improvement
ax3 = axes[1, 0]
stability_data = []
labels = []

for model_name in models.keys():
    all_features_std = results[model_name]['All Features (100)']['std_score']
    for sel_name, sel in selection_methods.items():
        if sel is not None:
            selected_std = results[model_name][sel_name]['std_score']
            improvement = ((all_features_std - selected_std) / all_features_std) * 100
            stability_data.append(improvement)
            labels.append(f'{model_name}\n{sel_name.split("(")[0].strip()}')

bars = ax3.bar(range(len(stability_data)), stability_data, color='skyblue', alpha=0.8)
ax3.set_xlabel('Model and Selection Method')
ax3.set_ylabel('Stability Improvement (%)')
ax3.set_title('Stability Improvement After Feature Selection')
ax3.set_xticks(range(len(labels)))
ax3.set_xticklabels(labels, rotation=45, ha='right')
ax3.grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, stability_data):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{value:.1f}%', ha='center', va='bottom')

# Plot 4: Feature Importance Distribution
ax4 = axes[1, 1]
# Use Random Forest feature importance as example
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)
feature_importance = rf_model.feature_importances_

# Sort features by importance
sorted_indices = np.argsort(feature_importance)[::-1]
sorted_importance = feature_importance[sorted_indices]

# Plot top 30 features
top_n = 30
ax4.bar(range(top_n), sorted_importance[:top_n], color='lightcoral', alpha=0.8)
ax4.set_xlabel('Feature Rank')
ax4.set_ylabel('Feature Importance')
ax4.set_title(f'Top {top_n} Most Important Features')
ax4.grid(True, alpha=0.3)

# Add horizontal line for average importance
avg_importance = np.mean(feature_importance)
ax4.axhline(y=avg_importance, color='red', linestyle='--', 
            label=f'Average Importance: {avg_importance:.4f}')
ax4.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'model_stability_analysis.png'), dpi=300, bbox_inches='tight')

# Figure 2: Detailed CV Score Analysis
fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))
fig2.suptitle('Cross-Validation Score Analysis', fontsize=16, fontweight='bold')

# Plot 1: CV Score Distributions
ax1 = axes2[0]
for model_name in models.keys():
    for selection_name in selection_methods.keys():
        scores = results[model_name][selection_name]['cv_scores']
        n_features = results[model_name][selection_name]['n_features']
        ax1.violinplot(scores, positions=[n_features], widths=5, 
                      showmeans=True, showextrema=True)

ax1.set_xlabel('Number of Features')
ax1.set_ylabel('Cross-Validation Accuracy')
ax1.set_title('CV Score Distributions')
ax1.grid(True, alpha=0.3)

# Plot 2: Stability Metrics
ax2 = axes2[1]
metrics_data = []
metric_labels = []

for model_name in models.keys():
    for selection_name in selection_methods.keys():
        mean_score = results[model_name][selection_name]['mean_score']
        std_score = results[model_name][selection_name]['std_score']
        n_features = results[model_name][selection_name]['n_features']
        
        # Calculate coefficient of variation (CV/mean)
        cv_coeff = std_score / mean_score if mean_score > 0 else 0
        
        metrics_data.append([mean_score, std_score, cv_coeff, n_features])
        metric_labels.append(f'{model_name}\n{selection_name.split("(")[0].strip()}')

metrics_data = np.array(metrics_data)
scatter = ax2.scatter(metrics_data[:, 0], metrics_data[:, 1], 
                      s=metrics_data[:, 3]*2, c=metrics_data[:, 2], 
                      cmap='viridis', alpha=0.7)

ax2.set_xlabel('Mean CV Score')
ax2.set_ylabel('Standard Deviation')
ax2.set_title('Stability vs Performance Trade-off\n(Size = Features, Color = CV Coefficient)')
ax2.grid(True, alpha=0.3)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label('Coefficient of Variation (CV/Mean)')

# Add annotations for each point
for i, (x, y, label) in enumerate(zip(metrics_data[:, 0], metrics_data[:, 1], metric_labels)):
    ax2.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points', 
                 fontsize=8, ha='left', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'cv_score_analysis.png'), dpi=300, bbox_inches='tight')

# Figure 3: Feature Selection Process Visualization
fig3, axes3 = plt.subplots(2, 2, figsize=(15, 10))
fig3.suptitle('Feature Selection Process and Impact', fontsize=16, fontweight='bold')

# Plot 1: Feature Selection Methods Comparison
ax1 = axes3[0, 0]
methods = list(selection_methods.keys())
rf_means = [results['Random Forest'][method]['mean_score'] for method in methods]
lr_means = [results['Logistic Regression'][method]['mean_score'] for method in methods]

x = np.arange(len(methods))
width = 0.35

ax1.bar(x - width/2, rf_means, width, label='Random Forest', alpha=0.8)
ax1.bar(x + width/2, lr_means, width, label='Logistic Regression', alpha=0.8)

ax1.set_xlabel('Feature Selection Method')
ax1.set_ylabel('Mean CV Score')
ax1.set_title('Performance Comparison Across Methods')
ax1.set_xticks(x)
ax1.set_xticklabels([method.split('(')[0].strip() for method in methods], rotation=45)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Stability vs Performance Trade-off
ax2 = axes3[0, 1]
for model_name in models.keys():
    for selection_name in selection_methods.keys():
        mean_score = results[model_name][selection_name]['mean_score']
        std_score = results[model_name][selection_name]['std_score']
        n_features = results[model_name][selection_name]['n_features']
        
        ax2.scatter(mean_score, std_score, s=n_features*3, alpha=0.7,
                    label=f'{model_name} - {selection_name.split("(")[0].strip()}')

ax2.set_xlabel('Mean CV Score (Performance)')
ax2.set_ylabel('Standard Deviation (Instability)')
ax2.set_title('Stability vs Performance Trade-off\n(Size = Number of Features)')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(True, alpha=0.3)

# Plot 3: Feature Count vs Stability
ax3 = axes3[1, 0]
for model_name in models.keys():
    n_features_list = []
    std_list = []
    
    for selection_name in selection_methods.keys():
        n_features_list.append(results[model_name][selection_name]['n_features'])
        std_list.append(results[model_name][selection_name]['std_score'])
    
    ax3.plot(n_features_list, std_list, 'o-', label=model_name, linewidth=2, markersize=8)

ax3.set_xlabel('Number of Features')
ax3.set_ylabel('Standard Deviation of CV Scores')
ax3.set_title('Feature Count vs Model Stability')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.invert_xaxis()  # Invert to show fewer features on the right

# Plot 4: Stability Improvement Heatmap
ax4 = axes3[1, 1]
improvement_matrix = []

for model_name in models.keys():
    row = []
    all_features_std = results[model_name]['All Features (100)']['std_score']
    
    for selection_name in selection_methods.keys():
        if selection_name == 'All Features (100)':
            row.append(0)  # No improvement for baseline
        else:
            selected_std = results[model_name][selection_name]['std_score']
            improvement = ((all_features_std - selected_std) / all_features_std) * 100
            row.append(improvement)
    
    improvement_matrix.append(row)

# Create heatmap
im = ax4.imshow(improvement_matrix, cmap='RdYlGn', aspect='auto')
ax4.set_xticks(range(len(selection_methods)))
ax4.set_yticks(range(len(models)))
ax4.set_xticklabels([method.split('(')[0].strip() for method in selection_methods.keys()], rotation=45)
ax4.set_yticklabels(list(models.keys()))

# Add text annotations
for i in range(len(models)):
    for j in range(len(selection_methods)):
        text = ax4.text(j, i, f'{improvement_matrix[i][j]:.1f}%',
                       ha="center", va="center", color="black", fontweight='bold')

ax4.set_title('Stability Improvement Heatmap\n(Percentage reduction in std dev)')
plt.colorbar(im, ax=ax4, label='Improvement (%)')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_selection_process.png'), dpi=300, bbox_inches='tight')

# 5. Summary Statistics
print("\n5. Summary Statistics:")
print("-" * 40)

summary_df = pd.DataFrame()

for model_name in models.keys():
    for selection_name in selection_methods.keys():
        result = results[model_name][selection_name]
        summary_df = pd.concat([summary_df, pd.DataFrame({
            'Model': [model_name],
            'Selection Method': [selection_name],
            'Features': [result['n_features']],
            'Mean CV Score': [result['mean_score']],
            'Std CV Score': [result['std_score']],
            'CV Coefficient': [result['std_score'] / result['mean_score'] if result['mean_score'] > 0 else 0]
        })], ignore_index=True)

print(summary_df.to_string(index=False))

# 6. Key Insights
print("\n6. Key Insights:")
print("-" * 40)

# Find best performing method for each model
for model_name in models.keys():
    best_method = min(results[model_name].keys(), 
                     key=lambda x: results[model_name][x]['std_score'])
    best_result = results[model_name][best_method]
    
    print(f"\n{model_name}:")
    print(f"  Most stable method: {best_method}")
    print(f"  Stability (std): {best_result['std_score']:.4f}")
    print(f"  Performance (mean): {best_result['mean_score']:.4f}")
    print(f"  Features used: {best_result['n_features']}")

# Calculate overall stability improvement
print(f"\nOverall Stability Analysis:")
print(f"  Question 6 Answer: {stability_improvement:.2f}% reduction in standard deviation")
print(f"  This demonstrates significant improvement in model stability")

print(f"\nAll plots saved to: {save_dir}")
print("Analysis complete!")
