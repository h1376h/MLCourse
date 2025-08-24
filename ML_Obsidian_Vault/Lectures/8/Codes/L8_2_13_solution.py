import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_2_Quiz_13")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("Question 13: Feature Selection Effects on Different ML Algorithms")
print("=" * 70)

# 1. How does feature selection affect linear models vs tree-based models?
print("\n1. Feature Selection Effects: Linear Models vs Tree-Based Models")
print("-" * 60)

# Generate synthetic dataset for demonstration
np.random.seed(42)
X, y = make_classification(n_samples=1000, n_features=20, n_informative=8, 
                          n_redundant=8, n_clusters_per_class=1, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Dataset created with:")
print(f"- {X.shape[0]} samples, {X.shape[1]} features")
print(f"- {np.sum(y)} positive samples, {len(y) - np.sum(y)} negative samples")

# Test different feature selection methods
feature_counts = [20, 15, 10, 8, 5, 3]
linear_scores = []
tree_scores = []
linear_times = []
tree_times = []

print("\nTesting different numbers of selected features:")
print(f"{'Features':<10} {'Linear Acc':<12} {'Tree Acc':<12} {'Linear Time':<12} {'Tree Time':<12}")

for n_features in feature_counts:
    # Feature selection using mutual information
    selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Linear model
    start_time = time.time()
    linear_model = LogisticRegression(random_state=42, max_iter=1000)
    linear_model.fit(X_train_selected, y_train)
    linear_score = accuracy_score(y_test, linear_model.predict(X_test_selected))
    linear_time = time.time() - start_time
    
    # Tree model
    start_time = time.time()
    tree_model = DecisionTreeClassifier(random_state=42, max_depth=5)
    tree_model.fit(X_train_selected, y_train)
    tree_score = accuracy_score(y_test, tree_model.predict(X_test_selected))
    tree_time = time.time() - start_time
    
    linear_scores.append(linear_score)
    tree_scores.append(tree_score)
    linear_times.append(linear_time)
    tree_times.append(tree_time)
    
    print(f"{n_features:<10} {linear_score:<12.4f} {tree_score:<12.4f} {linear_time:<12.4f} {tree_time:<12.4f}")

# 2. Which algorithm type benefits most from univariate selection?
print("\n2. Univariate Feature Selection Analysis")
print("-" * 50)

# Test univariate selection methods
univariate_methods = ['f_classif', 'mutual_info_classif']
method_names = ['F-statistic', 'Mutual Information']

print("Testing univariate selection methods:")
for i, method in enumerate(univariate_methods):
    print(f"\n{method_names[i]}:")
    
    if method == 'f_classif':
        selector = SelectKBest(score_func=f_classif, k=8)
    else:
        selector = SelectKBest(score_func=mutual_info_classif, k=8)
    
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Get feature scores
    scores = selector.scores_
    selected_features = selector.get_support()
    
    print(f"Selected features: {np.where(selected_features)[0]}")
    print(f"Feature scores: {scores[selected_features]}")
    
    # Test performance
    linear_model = LogisticRegression(random_state=42, max_iter=1000)
    linear_model.fit(X_train_selected, y_train)
    linear_acc = accuracy_score(y_test, linear_model.predict(X_test_selected))
    
    tree_model = DecisionTreeClassifier(random_state=42, max_depth=5)
    tree_model.fit(X_train_selected, y_train)
    tree_acc = accuracy_score(y_test, tree_model.predict(X_test_selected))
    
    print(f"Linear model accuracy: {linear_acc:.4f}")
    print(f"Tree model accuracy: {tree_acc:.4f}")

# 3. Calculate training time savings
print("\n3. Training Time Savings Calculation")
print("-" * 50)

# Given formulas
def linear_time_func(n):
    return 0.1 * n**2

def tree_time_func(n):
    return 0.05 * n * np.log(n)

n_original = 100
n_reduced = 20

# Calculate times
linear_original = linear_time_func(n_original)
linear_reduced = linear_time_func(n_reduced)
linear_savings = linear_original - linear_reduced
linear_savings_pct = (linear_savings / linear_original) * 100

tree_original = tree_time_func(n_original)
tree_reduced = tree_time_func(n_reduced)
tree_savings = tree_original - tree_reduced
tree_savings_pct = (tree_savings / tree_original) * 100

print(f"Original features: {n_original}")
print(f"Reduced features: {n_reduced}")
print()

print("Linear Model Training Times:")
print(f"  T(100) = 0.1 × 100² = {linear_original:.1f} seconds")
print(f"  T(20) = 0.1 × 20² = {linear_reduced:.1f} seconds")
print(f"  Time savings: {linear_savings:.1f} seconds ({linear_savings_pct:.1f}%)")
print()

print("Tree Model Training Times:")
print(f"  T(100) = 0.05 × 100 × log(100) = {tree_original:.3f} seconds")
print(f"  T(20) = 0.05 × 20 × log(20) = {tree_reduced:.3f} seconds")
print(f"  Time savings: {tree_savings:.3f} seconds ({tree_savings_pct:.1f}%)")

# 4. Which model benefits more from feature selection?
print("\n4. Model Comparison: Training Time Reduction")
print("-" * 55)

if linear_savings_pct > tree_savings_pct:
    print(f"Linear models benefit more from feature selection!")
    print(f"Linear: {linear_savings_pct:.1f}% reduction vs Tree: {tree_savings_pct:.1f}% reduction")
else:
    print(f"Tree models benefit more from feature selection!")
    print(f"Tree: {tree_savings_pct:.1f}% reduction vs Linear: {linear_savings_pct:.1f}% reduction")

# Create visualizations
print("\nGenerating visualizations...")

# Plot 1: Accuracy vs Number of Features
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(feature_counts, linear_scores, 'b-o', linewidth=2, markersize=8, label='Linear Model')
plt.plot(feature_counts, tree_scores, 'r-s', linewidth=2, markersize=8, label='Tree Model')
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.title('Model Accuracy vs Feature Count')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(feature_counts)

# Plot 2: Training Time vs Number of Features
plt.subplot(2, 2, 2)
plt.plot(feature_counts, linear_times, 'b-o', linewidth=2, markersize=8, label='Linear Model')
plt.plot(feature_counts, tree_times, 'r-s', linewidth=2, markersize=8, label='Tree Model')
plt.xlabel('Number of Features')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time vs Feature Count')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(feature_counts)

# Plot 3: Theoretical Time Complexity
plt.subplot(2, 2, 3)
n_range = np.linspace(1, 100, 1000)
plt.plot(n_range, linear_time_func(n_range), 'b-', linewidth=2, label='Linear: $T = 0.1n^2$')
plt.plot(n_range, tree_time_func(n_range), 'r-', linewidth=2, label='Tree: $T = 0.05n\\log n$')
plt.axvline(x=n_original, color='g', linestyle='--', alpha=0.7, label=f'Original: {n_original} features')
plt.axvline(x=n_reduced, color='orange', linestyle='--', alpha=0.7, label=f'Reduced: {n_reduced} features')
plt.xlabel('Number of Features (n)')
plt.ylabel('Training Time (seconds)')
plt.title('Theoretical Time Complexity')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Plot 4: Time Savings Comparison
plt.subplot(2, 2, 4)
models = ['Linear Model', 'Tree Model']
savings = [linear_savings, tree_savings]
savings_pct = [linear_savings_pct, tree_savings_pct]

bars = plt.bar(models, savings, color=['blue', 'red'], alpha=0.7)
plt.ylabel('Time Savings (seconds)')
plt.title('Training Time Savings: 100 → 20 Features')

# Add percentage labels on bars
for i, (bar, pct) in enumerate(zip(bars, savings_pct)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + max(savings)*0.01,
             f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_selection_analysis.png'), dpi=300, bbox_inches='tight')

# Plot 5: Feature Importance Comparison
plt.figure(figsize=(14, 6))

# Get feature importance for different selection methods
selector_f = SelectKBest(score_func=f_classif, k=8)
selector_mi = SelectKBest(score_func=mutual_info_classif, k=8)

X_train_f = selector_f.fit_transform(X_train, y_train)
X_train_mi = selector_mi.fit_transform(X_train, y_train)

# Get selected feature indices and scores
f_features = np.where(selector_f.get_support())[0]
f_scores = selector_f.scores_[f_features]
mi_features = np.where(selector_mi.get_support())[0]
mi_scores = selector_mi.scores_[mi_features]

# Plot F-statistic selection
plt.subplot(1, 2, 1)
bars1 = plt.bar(range(len(f_features)), f_scores, color='skyblue', alpha=0.8)
plt.xlabel('Feature Index')
plt.ylabel('F-statistic Score')
plt.title('F-statistic Feature Selection (Top 8)')
plt.xticks(range(len(f_features)), f_features)
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar, score in zip(bars1, f_scores):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + max(f_scores)*0.01,
             f'{score:.2f}', ha='center', va='bottom', fontsize=9)

# Plot Mutual Information selection
plt.subplot(1, 2, 2)
bars2 = plt.bar(range(len(mi_features)), mi_scores, color='lightcoral', alpha=0.8)
plt.xlabel('Feature Index')
plt.ylabel('Mutual Information Score')
plt.title('Mutual Information Feature Selection (Top 8)')
plt.xticks(range(len(mi_features)), mi_features)
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar, score in zip(bars2, mi_scores):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + max(mi_scores)*0.01,
             f'{score:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_importance_comparison.png'), dpi=300, bbox_inches='tight')

# Plot 6: Performance vs Feature Count (Detailed)
plt.figure(figsize=(15, 10))

# Create a more detailed analysis
feature_counts_detailed = list(range(3, 21))
linear_acc_detailed = []
tree_acc_detailed = []
linear_time_detailed = []
tree_time_detailed = []

for n_features in feature_counts_detailed:
    # Feature selection
    selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Linear model
    start_time = time.time()
    linear_model = LogisticRegression(random_state=42, max_iter=1000)
    linear_model.fit(X_train_selected, y_train)
    linear_acc = accuracy_score(y_test, linear_model.predict(X_test_selected))
    linear_time = time.time() - start_time
    
    # Tree model
    start_time = time.time()
    tree_model = DecisionTreeClassifier(random_state=42, max_depth=5)
    tree_model.fit(X_train_selected, y_train)
    tree_acc = accuracy_score(y_test, tree_model.predict(X_test_selected))
    tree_time = time.time() - start_time
    
    linear_acc_detailed.append(linear_acc)
    tree_acc_detailed.append(tree_acc)
    linear_time_detailed.append(linear_time)
    tree_time_detailed.append(tree_time)

# Plot accuracy trends
plt.subplot(2, 3, 1)
plt.plot(feature_counts_detailed, linear_acc_detailed, 'b-o', linewidth=2, markersize=6, label='Linear Model')
plt.plot(feature_counts_detailed, tree_acc_detailed, 'r-s', linewidth=2, markersize=6, label='Tree Model')
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Feature Count')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot training time trends
plt.subplot(2, 3, 2)
plt.plot(feature_counts_detailed, linear_time_detailed, 'b-o', linewidth=2, markersize=6, label='Linear Model')
plt.plot(feature_counts_detailed, tree_time_detailed, 'r-s', linewidth=2, markersize=6, label='Tree Model')
plt.xlabel('Number of Features')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time vs Feature Count')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot time savings percentage
plt.subplot(2, 3, 3)
linear_savings_pct_detailed = [(linear_time_detailed[0] - t) / linear_time_detailed[0] * 100 for t in linear_time_detailed]
tree_savings_pct_detailed = [(tree_time_detailed[0] - t) / tree_time_detailed[0] * 100 for t in tree_time_detailed]

plt.plot(feature_counts_detailed, linear_savings_pct_detailed, 'b-o', linewidth=2, markersize=6, label='Linear Model')
plt.plot(feature_counts_detailed, tree_savings_pct_detailed, 'r-s', linewidth=2, markersize=6, label='Tree Model')
plt.xlabel('Number of Features')
plt.ylabel('Time Savings (%)')
plt.title('Time Savings vs Feature Count')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot feature efficiency (accuracy per feature)
plt.subplot(2, 3, 4)
linear_efficiency = [acc/n for acc, n in zip(linear_acc_detailed, feature_counts_detailed)]
tree_efficiency = [acc/n for acc, n in zip(tree_acc_detailed, feature_counts_detailed)]

plt.plot(feature_counts_detailed, linear_efficiency, 'b-o', linewidth=2, markersize=6, label='Linear Model')
plt.plot(feature_counts_detailed, tree_efficiency, 'r-s', linewidth=2, markersize=6, label='Tree Model')
plt.xlabel('Number of Features')
plt.ylabel('Accuracy per Feature')
plt.title('Feature Efficiency')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot theoretical vs actual time complexity
plt.subplot(2, 3, 5)
theoretical_linear = [linear_time_func(n) for n in feature_counts_detailed]
theoretical_tree = [tree_time_func(n) for n in feature_counts_detailed]

# Normalize actual times to theoretical scale for comparison
scale_factor_linear = theoretical_linear[0] / linear_time_detailed[0]
scale_factor_tree = theoretical_tree[0] / tree_time_detailed[0]

scaled_linear_time = [t * scale_factor_linear for t in linear_time_detailed]
scaled_tree_time = [t * scale_factor_tree for t in tree_time_detailed]

plt.plot(feature_counts_detailed, theoretical_linear, 'b--', linewidth=2, label='Theoretical Linear')
plt.plot(feature_counts_detailed, theoretical_tree, 'r--', linewidth=2, label='Theoretical Tree')
plt.plot(feature_counts_detailed, scaled_linear_time, 'b-o', linewidth=2, markersize=6, label='Scaled Actual Linear')
plt.plot(feature_counts_detailed, scaled_tree_time, 'r-s', linewidth=2, markersize=6, label='Scaled Actual Tree')
plt.xlabel('Number of Features')
plt.ylabel('Scaled Training Time')
plt.title('Theoretical vs Actual Time Complexity')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot optimal feature count analysis
plt.subplot(2, 3, 6)
# Calculate a combined score (accuracy - normalized time penalty)
time_penalty_linear = np.array(linear_time_detailed) / max(linear_time_detailed)
time_penalty_tree = np.array(tree_time_detailed) / max(tree_time_detailed)

combined_score_linear = np.array(linear_acc_detailed) - 0.1 * time_penalty_linear
combined_score_tree = np.array(tree_acc_detailed) - 0.1 * time_penalty_tree

plt.plot(feature_counts_detailed, combined_score_linear, 'b-o', linewidth=2, markersize=6, label='Linear Model')
plt.plot(feature_counts_detailed, combined_score_tree, 'r-s', linewidth=2, markersize=6, label='Tree Model')
plt.xlabel('Number of Features')
plt.ylabel('Combined Score (Accuracy - Time Penalty)')
plt.title('Optimal Feature Count Analysis')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'detailed_feature_analysis.png'), dpi=300, bbox_inches='tight')

print(f"\nAll visualizations saved to: {save_dir}")

# Summary of key findings
print("\n" + "="*70)
print("SUMMARY OF KEY FINDINGS")
print("="*70)

print("\n1. Feature Selection Effects:")
print(f"   - Linear models show {'consistent' if np.std(linear_scores) < 0.05 else 'variable'} accuracy across feature counts")
print(f"   - Tree models show {'consistent' if np.std(tree_scores) < 0.05 else 'variable'} accuracy across feature counts")
print(f"   - Linear models are more sensitive to feature selection")

print("\n2. Univariate Selection Benefits:")
print("   - Both F-statistic and Mutual Information methods select informative features")
print("   - Linear models benefit more from univariate selection due to linear assumptions")
print("   - Tree models can handle non-linear relationships better")

print("\n3. Training Time Savings (100 → 20 features):")
print(f"   - Linear model: {linear_savings:.1f} seconds ({linear_savings_pct:.1f}%)")
print(f"   - Tree model: {tree_savings:.3f} seconds ({tree_savings_pct:.1f}%)")

print("\n4. Model Comparison:")
if linear_savings_pct > tree_savings_pct:
    print("   - Linear models benefit MORE from feature selection")
    print("   - This is due to O(n²) vs O(n log n) complexity")
else:
    print("   - Tree models benefit MORE from feature selection")
    print("   - This is due to O(n²) vs O(n log n) complexity")

print("\n5. Optimal Feature Count:")
optimal_linear = feature_counts_detailed[np.argmax(combined_score_linear)]
optimal_tree = feature_counts_detailed[np.argmax(combined_score_tree)]
print(f"   - Linear model optimal features: {optimal_linear}")
print(f"   - Tree model optimal features: {optimal_tree}")

print("\nAnalysis complete! Check the generated images for visual insights.")
