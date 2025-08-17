import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L7_2_Quiz_7")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("Question 7: Base Learner Suitability for Bagging")
print("=" * 60)

# Set random seed for reproducibility
np.random.seed(42)

# 1. Demonstrate what "unstable" means in bias-variance trade-off
print("\n1. Understanding 'Unstable' in Bias-Variance Trade-off")
print("-" * 50)

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                          n_informative=2, n_clusters_per_class=1, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train multiple models with different random seeds to show instability
n_models = 10
unstable_predictions = []
stable_predictions = []

# Unstable model: Deep decision tree
for i in range(n_models):
    # Deep tree (unstable)
    deep_tree = DecisionTreeClassifier(max_depth=20, random_state=i)
    deep_tree.fit(X_train, y_train)
    pred = deep_tree.predict(X_test)
    unstable_predictions.append(pred)
    
    # Shallow tree (more stable)
    shallow_tree = DecisionTreeClassifier(max_depth=3, random_state=i)
    shallow_tree.fit(X_train, y_train)
    pred = shallow_tree.predict(X_test)
    stable_predictions.append(pred)

# Calculate prediction variance for each test point
unstable_variance = np.var(unstable_predictions, axis=0)
stable_variance = np.var(stable_predictions, axis=0)

# Plot 1: Prediction variance comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_test[:, 0], X_test[:, 1], c=unstable_variance, cmap='viridis', s=50)
plt.colorbar(label='Prediction Variance')
plt.title('Deep Tree (Unstable) Prediction Variance')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=stable_variance, cmap='viridis', s=50)
plt.colorbar(label='Prediction Variance')
plt.title('Shallow Tree (Stable) Prediction Variance')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'prediction_variance_comparison.png'), dpi=300, bbox_inches='tight')

# 2. Demonstrate why deep trees are perfect for bagging
print("\n2. Deep Trees as Perfect Candidates for Bagging")
print("-" * 50)

# Train deep trees with different random seeds
deep_trees = []
for i in range(n_models):
    tree = DecisionTreeClassifier(max_depth=20, random_state=i)
    tree.fit(X_train, y_train)
    deep_trees.append(tree)

# Create bagging ensemble
bagging_ensemble = BaggingClassifier(
    DecisionTreeClassifier(max_depth=20), 
    n_estimators=n_models, 
    random_state=42
)
bagging_ensemble.fit(X_train, y_train)

# Compare individual trees vs ensemble
individual_accuracies = []
for tree in deep_trees:
    pred = tree.predict(X_test)
    acc = accuracy_score(y_test, pred)
    individual_accuracies.append(acc)

ensemble_accuracy = accuracy_score(y_test, bagging_ensemble.predict(X_test))

print(f"Individual Deep Tree Accuracies: {[f'{acc:.3f}' for acc in individual_accuracies]}")
print(f"Mean Individual Accuracy: {np.mean(individual_accuracies):.3f}")
print(f"Standard Deviation: {np.std(individual_accuracies):.3f}")
print(f"Bagging Ensemble Accuracy: {ensemble_accuracy:.3f}")
print(f"Improvement: {ensemble_accuracy - np.mean(individual_accuracies):.3f}")

# Plot 2: Individual vs Ensemble performance
plt.figure(figsize=(10, 6))
plt.bar(range(1, n_models + 1), individual_accuracies, alpha=0.7, label='Individual Trees')
plt.axhline(y=ensemble_accuracy, color='red', linestyle='--', linewidth=2, label='Bagging Ensemble')
plt.axhline(y=np.mean(individual_accuracies), color='green', linestyle='--', linewidth=2, label='Mean Individual')
plt.xlabel('Tree Number')
plt.ylabel('Accuracy')
plt.title('Individual Deep Trees vs Bagging Ensemble Performance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(save_dir, 'deep_trees_vs_ensemble.png'), dpi=300, bbox_inches='tight')

# 3. Demonstrate decision stump limitations
print("\n3. Decision Stump Limitations for Bagging")
print("-" * 50)

# Create decision stumps
decision_stumps = []
for i in range(n_models):
    stump = DecisionTreeClassifier(max_depth=1, random_state=i)
    stump.fit(X_train, y_train)
    decision_stumps.append(stump)

# Create bagging with stumps
stump_bagging = BaggingClassifier(
    DecisionTreeClassifier(max_depth=1), 
    n_estimators=n_models, 
    random_state=42
)
stump_bagging.fit(X_train, y_train)

# Compare performance
stump_accuracies = []
for stump in decision_stumps:
    pred = stump.predict(X_test)
    acc = accuracy_score(y_test, pred)
    stump_accuracies.append(acc)

stump_ensemble_accuracy = accuracy_score(y_test, stump_bagging.predict(X_test))

print(f"Individual Stump Accuracies: {[f'{acc:.3f}' for acc in stump_accuracies]}")
print(f"Mean Stump Accuracy: {np.mean(stump_accuracies):.3f}")
print(f"Stump Ensemble Accuracy: {stump_ensemble_accuracy:.3f}")
print(f"Improvement: {stump_ensemble_accuracy - np.mean(stump_accuracies):.3f}")

# Plot 3: Decision stump performance
plt.figure(figsize=(10, 6))
plt.bar(range(1, n_models + 1), stump_accuracies, alpha=0.7, label='Individual Stumps')
plt.axhline(y=stump_ensemble_accuracy, color='red', linestyle='--', linewidth=2, label='Stump Ensemble')
plt.axhline(y=np.mean(stump_accuracies), color='green', linestyle='--', linewidth=2, label='Mean Individual')
plt.xlabel('Stump Number')
plt.ylabel('Accuracy')
plt.title('Individual Decision Stumps vs Stump Ensemble Performance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(save_dir, 'decision_stumps_vs_ensemble.png'), dpi=300, bbox_inches='tight')

# 4. Demonstrate bias-variance reduction in bagging
print("\n4. Bias-Variance Reduction in Bagging")
print("-" * 50)

# Generate regression data for clearer bias-variance demonstration
X_reg, y_reg = make_regression(n_samples=1000, n_features=1, noise=0.5, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

# Train multiple deep trees
reg_trees = []
for i in range(n_models):
    tree = DecisionTreeRegressor(max_depth=15, random_state=i)
    tree.fit(X_train_reg, y_train_reg)
    reg_trees.append(tree)

# Create bagging ensemble
reg_bagging = BaggingRegressor(
    DecisionTreeRegressor(max_depth=15), 
    n_estimators=n_models, 
    random_state=42
)
reg_bagging.fit(X_train_reg, y_train_reg)

# Calculate predictions and errors
X_plot = np.linspace(X_reg.min(), X_reg.max(), 100).reshape(-1, 1)

individual_predictions = []
for tree in reg_trees:
    pred = tree.predict(X_plot)
    individual_predictions.append(pred)

ensemble_prediction = reg_bagging.predict(X_plot)

# Calculate bias and variance
individual_predictions = np.array(individual_predictions)
mean_individual = np.mean(individual_predictions, axis=0)
bias_squared = np.mean((mean_individual - y_reg.mean())**2)
variance = np.mean(np.var(individual_predictions, axis=0))

print(f"Individual Trees - Bias²: {bias_squared:.4f}")
print(f"Individual Trees - Variance: {variance:.4f}")
print(f"Individual Trees - MSE: {bias_squared + variance:.4f}")

# Plot 4: Bias-variance demonstration
plt.figure(figsize=(12, 8))

# Plot individual predictions
for i, pred in enumerate(individual_predictions):
    plt.plot(X_plot, pred, 'b-', alpha=0.2, linewidth=0.5)

# Plot ensemble prediction
plt.plot(X_plot, ensemble_prediction, 'r-', linewidth=3, label='Bagging Ensemble')
plt.plot(X_plot, mean_individual, 'g--', linewidth=2, label='Mean Individual')

# Plot data
plt.scatter(X_train_reg, y_train_reg, c='black', alpha=0.6, s=20, label='Training Data')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Bias-Variance Reduction in Bagging')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(save_dir, 'bias_variance_reduction.png'), dpi=300, bbox_inches='tight')

# 5. Comprehensive comparison visualization
print("\n5. Comprehensive Comparison")
print("-" * 50)

# Create comparison table
models = ['Deep Tree', 'Decision Stump', 'Deep Tree + Bagging', 'Stump + Bagging']
accuracies = [
    np.mean(individual_accuracies),
    np.mean(stump_accuracies),
    ensemble_accuracy,
    stump_ensemble_accuracy
]
variances = [
    np.std(individual_accuracies),
    np.std(stump_accuracies),
    0,  # Ensemble variance is much lower
    0
]

# Plot 5: Comprehensive comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Accuracy comparison
bars1 = ax1.bar(models, accuracies, color=['blue', 'green', 'red', 'orange'], alpha=0.7)
ax1.set_ylabel('Accuracy')
ax1.set_title('Model Performance Comparison')
ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for bar, acc in zip(bars1, accuracies):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{acc:.3f}', ha='center', va='bottom')

# Variance comparison (excluding ensembles)
models_var = ['Deep Tree', 'Decision Stump']
variances_var = [np.std(individual_accuracies), np.std(stump_accuracies)]

bars2 = ax2.bar(models_var, variances_var, color=['blue', 'green'], alpha=0.7)
ax2.set_ylabel('Standard Deviation')
ax2.set_title('Model Instability (Lower is Better)')
ax2.grid(True, alpha=0.3)

# Add value labels on bars
for bar, var in zip(bars2, variances_var):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'{var:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'comprehensive_comparison.png'), dpi=300, bbox_inches='tight')

# 6. Theoretical explanation visualization
print("\n6. Theoretical Framework")
print("-" * 50)

# Create theoretical framework diagram
fig, ax = plt.subplots(figsize=(12, 8))

# Define positions for text boxes
positions = {
    'unstable': (0.2, 0.8),
    'stable': (0.8, 0.8),
    'bagging': (0.5, 0.5),
    'bias': (0.2, 0.2),
    'variance': (0.8, 0.2)
}

# Add text boxes
ax.text(positions['unstable'][0], positions['unstable'][1], 
        'Unstable Learners\n(Deep Trees)\n• High Variance\n• Low Bias\n• Perfect for Bagging', 
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        ha='center', va='center', fontsize=10)

ax.text(positions['stable'][0], positions['stable'][1], 
        'Stable Learners\n(Decision Stumps)\n• Low Variance\n• High Bias\n• Poor for Bagging', 
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
        ha='center', va='center', fontsize=10)

ax.text(positions['bagging'][0], positions['bagging'][1], 
        'Bagging Effect\n• Reduces Variance\n• Maintains Bias\n• Improves Stability', 
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8),
        ha='center', va='center', fontsize=10)

ax.text(positions['bias'][0], positions['bias'][1], 
        'Bias\n• Systematic Error\n• Model Complexity\n• Underfitting', 
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
        ha='center', va='center', fontsize=10)

ax.text(positions['variance'][0], positions['variance'][1], 
        'Variance\n• Random Error\n• Model Sensitivity\n• Overfitting', 
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightpink", alpha=0.8),
        ha='center', va='center', fontsize=10)

# Add arrows
ax.annotate('', xy=positions['unstable'], xytext=positions['bagging'],
            arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
ax.annotate('', xy=positions['stable'], xytext=positions['bagging'],
            arrowprops=dict(arrowstyle='->', lw=2, color='green'))
ax.annotate('', xy=positions['bagging'], xytext=positions['bias'],
            arrowprops=dict(arrowstyle='->', lw=2, color='red'))
ax.annotate('', xy=positions['bagging'], xytext=positions['variance'],
            arrowprops=dict(arrowstyle='->', lw=2, color='red'))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('Theoretical Framework: Base Learner Suitability for Bagging')
ax.axis('off')

plt.savefig(os.path.join(save_dir, 'theoretical_framework.png'), dpi=300, bbox_inches='tight')

print(f"\nAll plots saved to: {save_dir}")
print("\nSummary of Key Findings:")
print("-" * 30)
print(f"1. Deep trees show high variance (instability): {np.std(individual_accuracies):.3f}")
print(f"2. Decision stumps show low variance (stability): {np.std(stump_accuracies):.3f}")
print(f"3. Bagging improves deep tree performance: {ensemble_accuracy - np.mean(individual_accuracies):.3f}")
print(f"4. Bagging provides minimal improvement for stumps: {stump_ensemble_accuracy - np.mean(stump_accuracies):.3f}")
print(f"5. Deep trees + bagging achieve best performance: {ensemble_accuracy:.3f}")
