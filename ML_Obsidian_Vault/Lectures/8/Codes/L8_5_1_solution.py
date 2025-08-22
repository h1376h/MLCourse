import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.feature_selection import (
    SelectKBest, chi2, f_classif, mutual_info_classif,
    VarianceThreshold, SelectPercentile
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import time
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_5_Quiz_1")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("="*80)
print("FILTER METHODS IN FEATURE SELECTION - COMPREHENSIVE ANALYSIS")
print("="*80)

# 1. Create synthetic dataset to demonstrate filter methods
print("\n1. CREATING SYNTHETIC DATASET")
print("-" * 50)

# Generate synthetic dataset with known relevant/irrelevant features
np.random.seed(42)
X_synthetic, y_synthetic = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=5,
    n_redundant=5,
    n_clusters_per_class=1,
    random_state=42
)

# Add some noise features
noise_features = np.random.randn(1000, 10)
X_synthetic = np.column_stack([X_synthetic, noise_features])

# Create feature names
feature_names = [f'Feature_{i+1}' for i in range(X_synthetic.shape[1])]
print(f"Dataset created with {X_synthetic.shape[0]} samples and {X_synthetic.shape[1]} features")
print(f"- Informative features: 5")
print(f"- Redundant features: 5") 
print(f"- Original features: 20")
print(f"- Added noise features: 10")
print(f"- Total features: {X_synthetic.shape[1]}")

# 2. Load real-world dataset for comparison
print("\n2. LOADING REAL-WORLD DATASET (BREAST CANCER)")
print("-" * 50)

breast_cancer = load_breast_cancer()
X_real = breast_cancer.data
y_real = breast_cancer.target
feature_names_real = breast_cancer.feature_names

print(f"Breast Cancer dataset: {X_real.shape[0]} samples, {X_real.shape[1]} features")
print(f"Classes: {np.unique(y_real)} (0: malignant, 1: benign)")

# 3. MAIN CHARACTERISTICS OF FILTER METHODS
print("\n3. MAIN CHARACTERISTICS OF FILTER METHODS")
print("-" * 50)

filter_characteristics = {
    "Independence": "Evaluate features independently of learning algorithm",
    "Speed": "Fast computation - no model training required",
    "Scalability": "Can handle high-dimensional data efficiently",
    "Generality": "Results are algorithm-agnostic",
    "Univariate": "Most filter methods consider one feature at a time",
    "Statistical": "Based on statistical measures and correlations"
}

for char, description in filter_characteristics.items():
    print(f"• {char}: {description}")

# 4. DEMONSTRATE DIFFERENT FILTER METHODS
print("\n4. DEMONSTRATING DIFFERENT FILTER METHODS")
print("-" * 50)

# Prepare data
X_scaled = StandardScaler().fit_transform(X_real)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_real, test_size=0.3, random_state=42)

# Define filter methods
filter_methods = {
    'Variance Threshold': VarianceThreshold(threshold=0.1),
    'Chi-Square': SelectKBest(chi2, k=10),
    'F-Score': SelectKBest(f_classif, k=10),
    'Mutual Information': SelectKBest(mutual_info_classif, k=10)
}

# Apply filter methods and collect results
filter_results = {}
filter_times = {}

for method_name, method in filter_methods.items():
    print(f"\nApplying {method_name}...")
    
    start_time = time.time()
    
    if method_name == 'Variance Threshold':
        # Variance threshold works differently
        X_filtered = method.fit_transform(X_train)
        selected_features = method.get_support(indices=True)
        scores = method.variances_
    elif method_name == 'Chi-Square':
        # Chi-square requires non-negative features
        X_train_pos = X_train - X_train.min() + 1e-6
        X_filtered = method.fit_transform(X_train_pos, y_train)
        selected_features = method.get_support(indices=True)
        scores = method.scores_
    else:
        X_filtered = method.fit_transform(X_train, y_train)
        selected_features = method.get_support(indices=True)
        scores = method.scores_
    
    end_time = time.time()
    
    filter_results[method_name] = {
        'selected_features': selected_features,
        'scores': scores,
        'n_features': len(selected_features)
    }
    filter_times[method_name] = end_time - start_time
    
    print(f"  Selected {len(selected_features)} features in {filter_times[method_name]:.4f} seconds")

# 5. WRAPPER METHOD COMPARISON (for comparison with filter methods)
print("\n5. WRAPPER METHOD COMPARISON")
print("-" * 50)

from sklearn.feature_selection import RFE

print("Applying Recursive Feature Elimination (RFE) - Wrapper Method...")
start_time = time.time()

# Use Random Forest as the estimator for RFE
rf_estimator = RandomForestClassifier(n_estimators=50, random_state=42)
rfe = RFE(estimator=rf_estimator, n_features_to_select=10)
X_rfe = rfe.fit_transform(X_train, y_train)
rfe_features = rfe.get_support(indices=True)

end_time = time.time()
rfe_time = end_time - start_time

print(f"  RFE selected {len(rfe_features)} features in {rfe_time:.4f} seconds")

# 6. SPEED COMPARISON VISUALIZATION
print("\n6. CREATING SPEED COMPARISON VISUALIZATION")
print("-" * 50)

plt.figure(figsize=(12, 8))

# Prepare data for plotting
methods = list(filter_times.keys()) + ['RFE (Wrapper)']
times = list(filter_times.values()) + [rfe_time]
colors = ['lightblue'] * len(filter_times) + ['lightcoral']

bars = plt.bar(methods, times, color=colors, alpha=0.7, edgecolor='black')

# Add value labels on bars
for bar, time_val in zip(bars, times):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'{time_val:.4f}s', ha='center', va='bottom', fontweight='bold')

plt.xlabel('Feature Selection Methods')
plt.ylabel('Execution Time (seconds)')
plt.title('Speed Comparison: Filter Methods vs Wrapper Methods')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)

# Add legend
legend_elements = [plt.Rectangle((0,0),1,1, facecolor='lightblue', alpha=0.7, label='Filter Methods'),
                   plt.Rectangle((0,0),1,1, facecolor='lightcoral', alpha=0.7, label='Wrapper Methods')]
plt.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'speed_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# 7. FEATURE RANKING VISUALIZATION
print("\n7. CREATING FEATURE RANKING VISUALIZATIONS")
print("-" * 50)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for idx, (method_name, results) in enumerate(filter_results.items()):
    if idx >= 4:
        break
    
    ax = axes[idx]
    
    if method_name == 'Variance Threshold':
        # For variance threshold, show all variances
        feature_indices = np.arange(len(results['scores']))
        scores = results['scores']
        selected = results['selected_features']
        
        colors = ['lightcoral' if i in selected else 'lightblue' for i in feature_indices]
        bars = ax.bar(feature_indices, scores, color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Variance')
        ax.set_title(f'{method_name}\n(Red = Selected, Blue = Rejected)')
        
    else:
        # For other methods, show top features
        if len(results['scores']) > 15:
            # Show top 15 features for clarity
            top_indices = np.argsort(results['scores'])[-15:]
            top_scores = results['scores'][top_indices]
            top_feature_names = [f'F{i}' for i in top_indices]
        else:
            top_indices = np.argsort(results['scores'])
            top_scores = results['scores'][top_indices]
            top_feature_names = [f'F{i}' for i in top_indices]
        
        colors = ['lightcoral' if i in results['selected_features'] else 'lightblue' 
                 for i in top_indices]
        
        bars = ax.barh(range(len(top_scores)), top_scores, color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_yticks(range(len(top_feature_names)))
        ax.set_yticklabels(top_feature_names)
        ax.set_xlabel('Score')
        ax.set_title(f'{method_name}\n(Red = Selected, Blue = Rejected)')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_rankings.png'), dpi=300, bbox_inches='tight')
plt.close()

# 8. EVALUATION CRITERIA DEMONSTRATION
print("\n8. EVALUATION CRITERIA USED BY FILTER METHODS")
print("-" * 50)

evaluation_criteria = {
    'Variance Threshold': {
        'Type': 'Statistical Measure',
        'Formula': r'$\text{Var}(X) = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2$',
        'Purpose': 'Remove features with low variance (quasi-constant)',
        'Range': '[0, ∞)',
        'Higher_Better': True
    },
    'Chi-Square': {
        'Type': 'Statistical Test',
        'Formula': r'$\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$',
        'Purpose': 'Test independence between categorical variables',
        'Range': '[0, ∞)',
        'Higher_Better': True
    },
    'F-Score (ANOVA)': {
        'Type': 'Statistical Test',
        'Formula': r'$F = \frac{\text{MS}_{\text{between}}}{\text{MS}_{\text{within}}}$',
        'Purpose': 'Compare means between groups',
        'Range': '[0, ∞)',
        'Higher_Better': True
    },
    'Mutual Information': {
        'Type': 'Information Theory',
        'Formula': r'$I(X;Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$',
        'Purpose': 'Measure dependence between variables',
        'Range': '[0, ∞)',
        'Higher_Better': True
    }
}

for method, criteria in evaluation_criteria.items():
    print(f"\n{method}:")
    for key, value in criteria.items():
        if key != 'Formula':
            print(f"  {key.replace('_', ' ')}: {value}")
        else:
            print(f"  {key}: {value}")

# 9. PERFORMANCE COMPARISON
print("\n9. PERFORMANCE COMPARISON OF SELECTED FEATURES")
print("-" * 50)

# Test performance with different feature selections
performance_results = {}

# Test with all features
lr_all = LogisticRegression(random_state=42, max_iter=1000)
scores_all = cross_val_score(lr_all, X_train, y_train, cv=5)
performance_results['All Features'] = {
    'mean_score': scores_all.mean(),
    'std_score': scores_all.std(),
    'n_features': X_train.shape[1]
}

# Test with filter method selections
for method_name, results in filter_results.items():
    if method_name == 'Variance Threshold':
        continue  # Skip variance threshold for this comparison
    
    selected_indices = results['selected_features']
    X_selected = X_train[:, selected_indices]
    
    lr_filtered = LogisticRegression(random_state=42, max_iter=1000)
    scores_filtered = cross_val_score(lr_filtered, X_selected, y_train, cv=5)
    
    performance_results[method_name] = {
        'mean_score': scores_filtered.mean(),
        'std_score': scores_filtered.std(),
        'n_features': len(selected_indices)
    }

# Test with RFE selection
X_rfe_selected = X_train[:, rfe_features]
lr_rfe = LogisticRegression(random_state=42, max_iter=1000)
scores_rfe = cross_val_score(lr_rfe, X_rfe_selected, y_train, cv=5)
performance_results['RFE (Wrapper)'] = {
    'mean_score': scores_rfe.mean(),
    'std_score': scores_rfe.std(),
    'n_features': len(rfe_features)
}

print("Performance Results (5-fold CV):")
for method, results in performance_results.items():
    print(f"  {method}: {results['mean_score']:.4f} ± {results['std_score']:.4f} "
          f"({results['n_features']} features)")

# 10. PERFORMANCE VISUALIZATION
print("\n10. CREATING PERFORMANCE COMPARISON VISUALIZATION")
print("-" * 50)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Performance comparison
methods = list(performance_results.keys())
means = [performance_results[m]['mean_score'] for m in methods]
stds = [performance_results[m]['std_score'] for m in methods]
n_features = [performance_results[m]['n_features'] for m in methods]

colors = ['lightgreen'] + ['lightblue'] * (len(methods)-2) + ['lightcoral']

bars1 = ax1.bar(methods, means, yerr=stds, color=colors, alpha=0.7, 
                edgecolor='black', capsize=5)

# Add value labels
for bar, mean_val, n_feat in zip(bars1, means, n_features):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{mean_val:.3f}\n({n_feat} feat)', ha='center', va='bottom',
             fontweight='bold', fontsize=10)

ax1.set_ylabel('Cross-Validation Accuracy')
ax1.set_title('Performance Comparison: Filter vs Wrapper Methods')
ax1.set_xticklabels(methods, rotation=45, ha='right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.1)

# Feature count vs performance scatter plot
filter_methods_only = [m for m in methods if m not in ['All Features', 'RFE (Wrapper)']]
filter_means = [performance_results[m]['mean_score'] for m in filter_methods_only]
filter_n_features = [performance_results[m]['n_features'] for m in filter_methods_only]

ax2.scatter(filter_n_features, filter_means, c='lightblue', s=100, alpha=0.7, 
           edgecolor='black', label='Filter Methods')

# Add wrapper method
wrapper_mean = performance_results['RFE (Wrapper)']['mean_score']
wrapper_n_features = performance_results['RFE (Wrapper)']['n_features']
ax2.scatter(wrapper_n_features, wrapper_mean, c='lightcoral', s=100, alpha=0.7,
           edgecolor='black', label='Wrapper Method')

# Add all features
all_mean = performance_results['All Features']['mean_score']
all_n_features = performance_results['All Features']['n_features']
ax2.scatter(all_n_features, all_mean, c='lightgreen', s=100, alpha=0.7,
           edgecolor='black', label='All Features')

# Add method labels
for method, x, y in zip(filter_methods_only + ['RFE (Wrapper)', 'All Features'], 
                       filter_n_features + [wrapper_n_features, all_n_features],
                       filter_means + [wrapper_mean, all_mean]):
    ax2.annotate(method.replace(' ', '\n'), (x, y), xytext=(5, 5), 
                textcoords='offset points', fontsize=8, ha='left')

ax2.set_xlabel('Number of Features')
ax2.set_ylabel('Cross-Validation Accuracy')
ax2.set_title('Performance vs Number of Features')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# 11. WHEN TO USE FILTER METHODS
print("\n11. WHEN TO USE FILTER METHODS")
print("-" * 50)

usage_scenarios = {
    "High-dimensional data": "When you have many features (>>1000) and need fast preprocessing",
    "Exploratory analysis": "Initial feature selection before applying more sophisticated methods",
    "Computational constraints": "When computational resources are limited",
    "Algorithm independence": "When you want feature selection that works across multiple algorithms",
    "Large datasets": "When dataset size makes wrapper methods computationally prohibitive",
    "Preprocessing step": "As a first step before applying wrapper or embedded methods",
    "Real-time applications": "When feature selection needs to be done quickly in production"
}

print("Filter methods are most appropriate when:")
for scenario, description in usage_scenarios.items():
    print(f"• {scenario}: {description}")

# 12. SUMMARY COMPARISON TABLE
print("\n12. FILTER VS WRAPPER METHODS COMPARISON")
print("-" * 50)

comparison_data = {
    'Aspect': ['Speed', 'Accuracy', 'Computational Cost', 'Algorithm Dependence', 
               'Feature Interactions', 'Overfitting Risk', 'Scalability'],
    'Filter Methods': ['Fast', 'Good', 'Low', 'Independent', 'Limited', 'Low', 'High'],
    'Wrapper Methods': ['Slow', 'Better', 'High', 'Dependent', 'Captured', 'Higher', 'Lower']
}

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Create comparison visualization
fig, ax = plt.subplots(figsize=(12, 8))

# Create a heatmap-style comparison
metrics = ['Speed', 'Computational\nEfficiency', 'Algorithm\nIndependence', 'Scalability']
filter_scores = [5, 5, 5, 5]  # High scores for filter methods
wrapper_scores = [2, 2, 2, 2]  # Lower scores for wrapper methods

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, filter_scores, width, label='Filter Methods', 
               color='lightblue', alpha=0.7, edgecolor='black')
bars2 = ax.bar(x + width/2, wrapper_scores, width, label='Wrapper Methods', 
               color='lightcoral', alpha=0.7, edgecolor='black')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height}', ha='center', va='bottom', fontweight='bold')

ax.set_xlabel('Comparison Metrics')
ax.set_ylabel('Score (1-5 scale)')
ax.set_title('Filter Methods vs Wrapper Methods: Key Characteristics')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 6)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'filter_vs_wrapper_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE!")
print(f"All visualizations saved to: {save_dir}")
print(f"{'='*80}")

# Final summary statistics
print(f"\nSUMMARY STATISTICS:")
print(f"- Filter methods average time: {np.mean(list(filter_times.values())):.4f} seconds")
print(f"- Wrapper method (RFE) time: {rfe_time:.4f} seconds")
print(f"- Speed improvement: {rfe_time / np.mean(list(filter_times.values())):.1f}x faster")

filter_performance = np.mean([performance_results[m]['mean_score'] 
                             for m in performance_results.keys() 
                             if m not in ['All Features', 'RFE (Wrapper)']])
wrapper_performance = performance_results['RFE (Wrapper)']['mean_score']

print(f"- Average filter method performance: {filter_performance:.4f}")
print(f"- Wrapper method performance: {wrapper_performance:.4f}")
print(f"- Performance difference: {wrapper_performance - filter_performance:.4f}")
