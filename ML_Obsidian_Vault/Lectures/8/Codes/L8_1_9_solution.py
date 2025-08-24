import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_1_Quiz_9")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("=" * 80)
print("QUESTION 9: FEATURE SELECTION EFFECTS ON DIFFERENT ALGORITHMS")
print("=" * 80)

# Generate synthetic datasets for demonstration
np.random.seed(42)

# Create a dataset with different feature relevance levels
n_samples = 1000
n_features = 20
n_informative = 8
n_redundant = 6
n_useless = 6

# Generate classification dataset
X_class, y_class = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=n_informative,
    n_redundant=n_redundant,
    n_clusters_per_class=1,
    random_state=42
)

# Generate regression dataset
X_reg, y_reg = make_regression(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=n_informative,
    random_state=42
)

print(f"\nDataset created:")
print(f"- Samples: {n_samples}")
print(f"- Total features: {n_features}")
print(f"- Informative features: {n_informative}")
print(f"- Redundant features: {n_redundant}")
print(f"- Useless features: {n_useless}")

# Split datasets
X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(
    X_class, y_class, test_size=0.3, random_state=42
)
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

# Standardize features
scaler_class = StandardScaler()
scaler_reg = StandardScaler()

X_class_train_scaled = scaler_class.fit_transform(X_class_train)
X_class_test_scaled = scaler_class.transform(X_class_test)
X_reg_train_scaled = scaler_reg.fit_transform(X_reg_train)
X_reg_test_scaled = scaler_reg.transform(X_reg_test)

print("\n" + "="*60)
print("PART 1: LINEAR MODELS (Linear Regression)")
print("="*60)

# Test linear regression with different numbers of features
feature_counts = [5, 10, 15, 20]
linear_scores = []
linear_coefficients = []

for k in feature_counts:
    # Select top k features
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = selector.fit_transform(X_class_train_scaled, y_class_train)
    X_test_selected = selector.transform(X_class_test_scaled)
    
    # Train linear regression
    lr = LinearRegression()
    lr.fit(X_train_selected, y_class_train)
    
    # Predict and evaluate
    y_pred = lr.predict(X_test_selected)
    score = r2_score(y_class_test, y_pred)
    linear_scores.append(score)
    
    # Store coefficients
    linear_coefficients.append(np.abs(lr.coef_))
    
    print(f"Features: {k:2d} | R² Score: {score:.4f}")

# Visualize linear model performance
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(feature_counts, linear_scores, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Features')
plt.ylabel('R² Score')
plt.title('Linear Regression Performance vs Feature Count')
plt.grid(True, alpha=0.3)
plt.xticks(feature_counts)

# Show coefficient magnitudes
plt.subplot(2, 2, 2)
for i, k in enumerate(feature_counts):
    plt.plot(range(1, k+1), linear_coefficients[i], 'o-', 
             label=f'{k} features', alpha=0.7)
plt.xlabel('Feature Index')
plt.ylabel('|Coefficient|')
plt.title('Coefficient Magnitudes by Feature Count')
plt.legend()
plt.grid(True, alpha=0.3)

print("\n" + "="*60)
print("PART 2: TREE-BASED MODELS (Decision Trees)")
print("="*60)

# Test decision tree with different numbers of features
tree_scores = []
tree_importances = []

for k in feature_counts:
    # Select top k features
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_train_selected = selector.fit_transform(X_class_train_scaled, y_class_train)
    X_test_selected = selector.transform(X_class_test_scaled)
    
    # Train decision tree
    dt = DecisionTreeClassifier(random_state=42, max_depth=10)
    dt.fit(X_train_selected, y_class_train)
    
    # Predict and evaluate
    y_pred = dt.predict(X_test_selected)
    score = accuracy_score(y_class_test, y_pred)
    tree_scores.append(score)
    
    # Store feature importances
    tree_importances.append(dt.feature_importances_)
    
    print(f"Features: {k:2d} | Accuracy: {score:.4f}")

# Visualize tree model performance
plt.subplot(2, 2, 3)
plt.plot(feature_counts, tree_scores, 'go-', linewidth=2, markersize=8)
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.title('Decision Tree Performance vs Feature Count')
plt.grid(True, alpha=0.3)
plt.xticks(feature_counts)

# Show feature importances
plt.subplot(2, 2, 4)
for i, k in enumerate(feature_counts):
    plt.plot(range(1, k+1), tree_importances[i], 'o-', 
             label=f'{k} features', alpha=0.7)
plt.xlabel('Feature Index')
plt.ylabel('Feature Importance')
plt.title('Feature Importances by Feature Count')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'linear_vs_tree_performance.png'), dpi=300, bbox_inches='tight')

print("\n" + "="*60)
print("PART 3: NEURAL NETWORKS")
print("="*60)

# Test neural network with different numbers of features
nn_scores = []
nn_training_times = []

for k in feature_counts:
    # Select top k features
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_train_selected = selector.fit_transform(X_class_train_scaled, y_class_train)
    X_test_selected = selector.transform(X_class_test_scaled)
    
    # Train neural network
    nn = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
    nn.fit(X_train_selected, y_class_train)
    
    # Predict and evaluate
    y_pred = nn.predict(X_test_selected)
    score = accuracy_score(y_class_test, y_pred)
    nn_scores.append(score)
    
    print(f"Features: {k:2d} | Accuracy: {score:.4f}")

# Visualize neural network performance
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(feature_counts, nn_scores, 'ro-', linewidth=2, markersize=8)
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.title('Neural Network Performance vs Feature Count')
plt.grid(True, alpha=0.3)
plt.xticks(feature_counts)

# Compare all three algorithms
plt.subplot(2, 2, 2)
plt.plot(feature_counts, linear_scores, 'bo-', linewidth=2, markersize=8, label='Linear Regression')
plt.plot(feature_counts, tree_scores, 'go-', linewidth=2, markersize=8, label='Decision Tree')
plt.plot(feature_counts, nn_scores, 'ro-', linewidth=2, markersize=8, label='Neural Network')
plt.xlabel('Number of Features')
plt.ylabel('Performance Score')
plt.title('Algorithm Comparison: Performance vs Feature Count')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(feature_counts)

print("\n" + "="*60)
print("PART 4: WHICH ALGORITHM BENEFITS MOST?")
print("="*60)

# Calculate improvement ratios
def calculate_improvement_ratio(scores):
    """Calculate improvement ratio: (final_score - initial_score) / initial_score"""
    return (scores[-1] - scores[0]) / scores[0]

linear_improvement = calculate_improvement_ratio(linear_scores)
tree_improvement = calculate_improvement_ratio(tree_scores)
nn_improvement = calculate_improvement_ratio(nn_scores)

print(f"Linear Regression improvement ratio: {linear_improvement:.4f}")
print(f"Decision Tree improvement ratio: {tree_improvement:.4f}")
print(f"Neural Network improvement ratio: {nn_improvement:.4f}")

# Visualize improvement ratios
plt.subplot(2, 2, 3)
algorithms = ['Linear\nRegression', 'Decision\nTree', 'Neural\nNetwork']
improvements = [linear_improvement, tree_improvement, nn_improvement]
colors = ['blue', 'green', 'red']

bars = plt.bar(algorithms, improvements, color=colors, alpha=0.7)
plt.ylabel('Improvement Ratio')
plt.title('Performance Improvement with Feature Selection')
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, value in zip(bars, improvements):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{value:.3f}', ha='center', va='bottom')

print("\n" + "="*60)
print("PART 5: COMPARISON WITH SPECIFIC EXAMPLES")
print("="*60)

# Create a concrete example with feature correlation analysis
print("\nFeature Correlation Analysis:")
feature_correlations = np.corrcoef(X_class_train_scaled.T)
np.fill_diagonal(feature_correlations, 0)  # Remove self-correlations

# Find highly correlated features
high_corr_pairs = []
for i in range(len(feature_correlations)):
    for j in range(i+1, len(feature_correlations)):
        if abs(feature_correlations[i, j]) > 0.8:
            high_corr_pairs.append((i, j, feature_correlations[i, j]))

print(f"Found {len(high_corr_pairs)} pairs of highly correlated features (|r| > 0.8)")
for pair in high_corr_pairs[:5]:  # Show first 5
    print(f"  Features {pair[0]} and {pair[1]}: r = {pair[2]:.3f}")

# Visualize correlation matrix
plt.subplot(2, 2, 4)
sns.heatmap(feature_correlations[:10, :10], cmap='RdBu_r', center=0, 
            square=True, cbar_kws={'label': 'Correlation'})
plt.title('Feature Correlation Matrix (First 10 Features)')
plt.xlabel('Feature Index')
plt.ylabel('Feature Index')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'neural_network_and_comparison.png'), dpi=300, bbox_inches='tight')

print("\n" + "="*60)
print("PART 6: INFORMATION GAIN CALCULATION")
print("="*60)

# Given information gain values
ig_values = [0.8, 0.6, 0.3]
n_features_total = len(ig_values)
n_features_selected = 2

print(f"Given Information Gain values: {ig_values}")
print(f"Total features: {n_features_total}")
print(f"Features to select: {n_features_selected}")

# Sort features by information gain (descending)
sorted_ig = sorted(ig_values, reverse=True)
print(f"Sorted IG values: {sorted_ig}")

# Select top k features
selected_ig = sorted_ig[:n_features_selected]
total_ig = sum(selected_ig)
max_possible_ig = sum(ig_values)
ig_percentage = (total_ig / max_possible_ig) * 100

print(f"\nTop {n_features_selected} features selected: {selected_ig}")
print(f"Total Information Gain: {total_ig:.1f}")
print(f"Maximum possible IG: {max_possible_ig:.1f}")
print(f"Percentage of maximum IG retained: {ig_percentage:.1f}%")

# Mathematical verification
print(f"\nMathematical verification:")
print(f"IG(S,A) = H(S) - Σ(|S_v|/|S|) * H(S_v)")
print(f"Selected features contribute: {selected_ig[0]:.1f} + {selected_ig[1]:.1f} = {total_ig:.1f}")
print(f"Information retention ratio: {total_ig:.1f}/{max_possible_ig:.1f} = {ig_percentage:.1f}%")

# Visualize information gain
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
features = [f'F{i+1}' for i in range(len(ig_values))]
bars = plt.bar(features, ig_values, color=['red', 'orange', 'blue'], alpha=0.7)
plt.ylabel('Information Gain')
plt.title('Information Gain by Feature')
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, value in zip(bars, ig_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{value:.1f}', ha='center', va='bottom')

plt.subplot(1, 2, 2)
# Pie chart showing information retention
labels = ['Selected\nFeatures', 'Unselected\nFeatures']
sizes = [total_ig, max_possible_ig - total_ig]
colors = ['lightgreen', 'lightcoral']
explode = (0.1, 0)

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90)
plt.title(f'Information Retention\n({n_features_selected}/{n_features_total} features)')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'information_gain_analysis.png'), dpi=300, bbox_inches='tight')

print("\n" + "="*60)
print("DETAILED ANALYSIS AND INSIGHTS")
print("="*60)

# Performance analysis with different feature selection strategies
print("\nFeature Selection Strategy Analysis:")

# Strategy 1: Select by correlation with target
correlation_scores = []
for i in range(X_class_train_scaled.shape[1]):
    corr = np.corrcoef(X_class_train_scaled[:, i], y_class_train)[0, 1]
    correlation_scores.append(abs(corr))

# Strategy 2: Select by mutual information
mutual_info_scores = mutual_info_classif(X_class_train_scaled, y_class_train, random_state=42)

# Strategy 3: Select by variance
variance_scores = np.var(X_class_train_scaled, axis=0)

# Compare strategies
strategies = ['Correlation', 'Mutual Info', 'Variance']
strategy_scores = [correlation_scores, mutual_info_scores, variance_scores]

plt.figure(figsize=(15, 10))

for i, (strategy, scores) in enumerate(zip(strategies, strategy_scores)):
    plt.subplot(2, 3, i+1)
    plt.bar(range(len(scores)), scores, alpha=0.7)
    plt.xlabel('Feature Index')
    plt.ylabel('Score')
    plt.title(f'{strategy} Scores')
    plt.grid(True, alpha=0.3)

# Show top features by each strategy
plt.subplot(2, 3, 4)
top_features_corr = np.argsort(correlation_scores)[-10:]
top_features_mi = np.argsort(mutual_info_scores)[-10:]
top_features_var = np.argsort(variance_scores)[-10:]

plt.bar(range(10), [correlation_scores[i] for i in top_features_corr], 
        alpha=0.7, label='Correlation', color='blue')
plt.bar(range(10), [mutual_info_scores[i] for i in top_features_mi], 
        alpha=0.7, label='Mutual Info', color='green')
plt.bar(range(10), [variance_scores[i] for i in top_features_var], 
        alpha=0.7, label='Variance', color='red')
plt.xlabel('Top 10 Features')
plt.ylabel('Score')
plt.title('Top 10 Features by Different Strategies')
plt.legend()
plt.grid(True, alpha=0.3)

# Feature overlap analysis
plt.subplot(2, 3, 5)
overlap_corr_mi = len(set(top_features_corr) & set(top_features_mi))
overlap_corr_var = len(set(top_features_corr) & set(top_features_var))
overlap_mi_var = len(set(top_features_mi) & set(top_features_var))

overlaps = [overlap_corr_mi, overlap_corr_var, overlap_mi_var]
overlap_labels = ['Corr vs MI', 'Corr vs Var', 'MI vs Var']

bars = plt.bar(overlap_labels, overlaps, color=['purple', 'orange', 'brown'], alpha=0.7)
plt.ylabel('Number of Overlapping Features')
plt.title('Feature Selection Strategy Overlap')
plt.grid(True, alpha=0.3, axis='y')

for bar, value in zip(bars, overlaps):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             str(value), ha='center', va='bottom')

# Performance comparison with different strategies
plt.subplot(2, 3, 6)
strategy_performances = []

for strategy_name, scores in zip(strategies, strategy_scores):
    # Select top 10 features
    top_indices = np.argsort(scores)[-10:]
    X_train_selected = X_class_train_scaled[:, top_indices]
    X_test_selected = X_class_test_scaled[:, top_indices]
    
    # Train and evaluate
    dt = DecisionTreeClassifier(random_state=42, max_depth=8)
    dt.fit(X_train_selected, y_class_train)
    y_pred = dt.predict(X_test_selected)
    accuracy = accuracy_score(y_class_test, y_pred)
    strategy_performances.append(accuracy)
    
    print(f"{strategy} strategy - Top 10 features: Accuracy = {accuracy:.4f}")

bars = plt.bar(strategies, strategy_performances, color=['blue', 'green', 'red'], alpha=0.7)
plt.ylabel('Accuracy')
plt.title('Performance by Selection Strategy')
plt.grid(True, alpha=0.3, axis='y')

for bar, value in zip(bars, strategy_performances):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{value:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_selection_strategies.png'), dpi=300, bbox_inches='tight')

print(f"\nAll plots saved to: {save_dir}")
print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
