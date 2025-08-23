import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification, make_moons
import warnings
warnings.filterwarnings('ignore')

# Create directory to save figures
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_2_Quiz_14")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("Question 14: Practical Implementation Considerations for Soft Margin SVMs")
print("=" * 80)

# 1. Handling the case where all points are outliers (all ξi > 0)
print("\n1. HANDLING ALL OUTLIERS CASE")
print("-" * 40)

def create_all_outliers_dataset(n_samples=100, noise_level=0.8):
    """Create a dataset where all points are essentially outliers"""
    # Generate two well-separated clusters
    X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, 
                             n_informative=2, n_clusters_per_class=1, 
                             random_state=42)
    
    # Add extreme noise to make all points outliers
    noise = np.random.normal(0, noise_level, X.shape)
    X_noisy = X + noise
    
    return X_noisy, y

# Create datasets with different outlier levels
X_clean, y_clean = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                                      n_informative=2, n_clusters_per_class=1, 
                                      random_state=42)

X_outliers, y_outliers = create_all_outliers_dataset(n_samples=100, noise_level=0.8)

# Test different C values on outlier dataset
C_values = [0.01, 0.1, 1, 10, 100]
outlier_results = []

for C in C_values:
    svm = SVC(C=C, kernel='linear', random_state=42)
    svm.fit(X_outliers, y_outliers)
    
    # Calculate slack variables (approximation)
    decision_values = svm.decision_function(X_outliers)
    slack_vars = np.maximum(0, 1 - y_outliers * decision_values)
    total_slack = np.sum(slack_vars)
    avg_slack = np.mean(slack_vars)
    
    outlier_results.append({
        'C': C,
        'total_slack': total_slack,
        'avg_slack': avg_slack,
        'n_support_vectors': len(svm.support_vectors_),
        'accuracy': accuracy_score(y_outliers, svm.predict(X_outliers))
    })
    
    print(f"C = {C:>6}: Total slack = {total_slack:>8.2f}, "
          f"Avg slack = {avg_slack:>6.3f}, "
          f"Support vectors = {len(svm.support_vectors_):>2}")

# Visualize outlier handling
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Handling All Outliers Case: Effect of C Parameter', fontsize=16)

# Plot clean dataset
axes[0, 0].scatter(X_clean[:, 0], X_clean[:, 1], c=y_clean, cmap='viridis', alpha=0.7)
axes[0, 0].set_title('Clean Dataset')
axes[0, 0].set_xlabel('$x_1$')
axes[0, 0].set_ylabel('$x_2$')

# Plot outlier dataset
axes[0, 1].scatter(X_outliers[:, 0], X_outliers[:, 1], c=y_outliers, cmap='viridis', alpha=0.7)
axes[0, 1].set_title('Dataset with All Outliers')
axes[0, 1].set_xlabel('$x_1$')
axes[0, 1].set_ylabel('$x_2$')

# Plot slack variables vs C
C_vals = [r['C'] for r in outlier_results]
slack_vals = [r['total_slack'] for r in outlier_results]
axes[0, 2].semilogx(C_vals, slack_vals, 'bo-', linewidth=2, markersize=8)
axes[0, 2].set_xlabel('C Parameter')
axes[0, 2].set_ylabel('Total Slack Variables')
axes[0, 2].set_title('Total Slack vs C')
axes[0, 2].grid(True, alpha=0.3)

# Plot support vectors vs C
sv_counts = [r['n_support_vectors'] for r in outlier_results]
axes[1, 0].semilogx(C_vals, sv_counts, 'ro-', linewidth=2, markersize=8)
axes[1, 0].set_xlabel('C Parameter')
axes[1, 0].set_ylabel('Number of Support Vectors')
axes[1, 0].set_title('Support Vectors vs C')
axes[1, 0].grid(True, alpha=0.3)

# Plot accuracy vs C
accuracies = [r['accuracy'] for r in outlier_results]
axes[1, 1].semilogx(C_vals, accuracies, 'go-', linewidth=2, markersize=8)
axes[1, 1].set_xlabel('C Parameter')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_title('Accuracy vs C')
axes[1, 1].grid(True, alpha=0.3)

# Decision boundaries for different C values
for i, C in enumerate([0.01, 1, 100]):
    svm = SVC(C=C, kernel='linear', random_state=42)
    svm.fit(X_outliers, y_outliers)
    
    # Create mesh grid
    x_min, x_max = X_outliers[:, 0].min() - 0.5, X_outliers[:, 0].max() + 0.5
    y_min, y_max = X_outliers[:, 1].min() - 0.5, X_outliers[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axes[1, 2].contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    axes[1, 2].scatter(X_outliers[:, 0], X_outliers[:, 1], c=y_outliers, 
                       cmap='viridis', alpha=0.7, s=20)
    axes[1, 2].set_title(f'Decision Boundary (C={C})')
    axes[1, 2].set_xlabel('$x_1$')
    axes[1, 2].set_ylabel('$x_2$')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'outlier_handling.png'), dpi=300, bbox_inches='tight')

# 2. Essential preprocessing steps
print("\n2. ESSENTIAL PREPROCESSING STEPS")
print("-" * 40)

def demonstrate_preprocessing():
    """Demonstrate essential preprocessing steps for SVM"""
    
    # Create dataset with different scales
    np.random.seed(42)
    X_unscaled = np.random.randn(200, 2)
    X_unscaled[:, 0] *= 100  # First feature has much larger scale
    X_unscaled[:, 1] *= 0.1  # Second feature has much smaller scale
    y_unscaled = np.sign(X_unscaled[:, 0] + X_unscaled[:, 1])
    
    # Apply different preprocessing methods
    scalers = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'No Scaling': None
    }
    
    preprocessing_results = {}
    
    for name, scaler in scalers.items():
        if scaler is not None:
            X_scaled = scaler.fit_transform(X_unscaled)
        else:
            X_scaled = X_unscaled
            
        # Train SVM
        svm = SVC(C=1, kernel='linear', random_state=42)
        scores = cross_val_score(svm, X_scaled, y_unscaled, cv=5)
        
        preprocessing_results[name] = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'X_scaled': X_scaled
        }
        
        print(f"{name:>15}: CV Score = {scores.mean():.3f} ± {scores.std():.3f}")
    
    return preprocessing_results, X_unscaled, y_unscaled

preprocessing_results, X_unscaled, y_unscaled = demonstrate_preprocessing()

# Visualize preprocessing effects
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Essential Preprocessing Steps for SVM', fontsize=16)

# Original data
axes[0, 0].scatter(X_unscaled[:, 0], X_unscaled[:, 1], c=y_unscaled, cmap='viridis', alpha=0.7)
axes[0, 0].set_title('Original Data (Unscaled)')
axes[0, 0].set_xlabel('$x_1$ (scale: 100)')
axes[0, 0].set_ylabel('$x_2$ (scale: 0.1)')

# StandardScaler
X_std = preprocessing_results['StandardScaler']['X_scaled']
axes[0, 1].scatter(X_std[:, 0], X_std[:, 1], c=y_unscaled, cmap='viridis', alpha=0.7)
axes[0, 1].set_title('StandardScaler (Z-score)')
axes[0, 1].set_xlabel('$x_1$ (standardized)')
axes[0, 1].set_ylabel('$x_2$ (standardized)')

# MinMaxScaler
X_minmax = preprocessing_results['MinMaxScaler']['X_scaled']
axes[0, 2].scatter(X_minmax[:, 0], X_minmax[:, 1], c=y_unscaled, cmap='viridis', alpha=0.7)
axes[0, 2].set_title('MinMaxScaler (0-1)')
axes[0, 2].set_xlabel('$x_1$ (normalized)')
axes[0, 2].set_ylabel('$x_2$ (normalized)')

# Feature scales comparison
feature_scales = {
    'Original': [np.std(X_unscaled[:, 0]), np.std(X_unscaled[:, 1])],
    'StandardScaler': [np.std(X_std[:, 0]), np.std(X_std[:, 1])],
    'MinMaxScaler': [np.std(X_minmax[:, 0]), np.std(X_minmax[:, 1])]
}

x_pos = np.arange(len(feature_scales))
width = 0.35

axes[1, 0].bar(x_pos - width/2, [scales[0] for scales in feature_scales.values()], 
               width, label='Feature 1', alpha=0.8)
axes[1, 0].bar(x_pos + width/2, [scales[1] for scales in feature_scales.values()], 
               width, label='Feature 2', alpha=0.8)
axes[1, 0].set_xlabel('Preprocessing Method')
axes[1, 0].set_ylabel('Standard Deviation')
axes[1, 0].set_title('Feature Scale Comparison')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(feature_scales.keys())
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# CV scores comparison
methods = list(preprocessing_results.keys())
scores = [preprocessing_results[method]['mean_score'] for method in methods]
score_stds = [preprocessing_results[method]['std_score'] for method in methods]

axes[1, 1].bar(methods, scores, yerr=score_stds, capsize=5, alpha=0.8)
axes[1, 1].set_xlabel('Preprocessing Method')
axes[1, 1].set_ylabel('Cross-Validation Score')
axes[1, 1].set_title('Performance Comparison')
axes[1, 1].grid(True, alpha=0.3)

# Decision boundaries comparison
for i, (name, scaler) in enumerate([('No Scaling', None), ('StandardScaler', StandardScaler())]):
    if scaler is not None:
        X_plot = scaler.fit_transform(X_unscaled)
    else:
        X_plot = X_unscaled
    
    svm = SVC(C=1, kernel='linear', random_state=42)
    svm.fit(X_plot, y_unscaled)
    
    # Create mesh grid
    x_min, x_max = X_plot[:, 0].min() - 0.5, X_plot[:, 0].max() + 0.5
    y_min, y_max = X_plot[:, 1].min() - 0.5, X_plot[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axes[1, 2].contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    axes[1, 2].scatter(X_plot[:, 0], X_plot[:, 1], c=y_unscaled, 
                       cmap='viridis', alpha=0.7, s=20)
    axes[1, 2].set_title(f'Decision Boundary ({name})')
    axes[1, 2].set_xlabel('$x_1$')
    axes[1, 2].set_ylabel('$x_2$')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'preprocessing_steps.png'), dpi=300, bbox_inches='tight')

# 3. Grid search strategy for optimal C values
print("\n3. GRID SEARCH STRATEGY FOR OPTIMAL C VALUES")
print("-" * 40)

def demonstrate_grid_search():
    """Demonstrate different grid search strategies for C parameter"""
    
    # Create a complex dataset
    X, y = make_moons(n_samples=200, noise=0.3, random_state=42)
    
    # Different grid search strategies
    strategies = {
        'Linear Grid': np.linspace(0.1, 10, 20),
        'Logarithmic Grid': np.logspace(-3, 3, 20),
        'Exponential Grid': np.exp(np.linspace(-2, 2, 20)),
        'Custom Grid': np.array([0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100])
    }
    
    grid_results = {}
    
    for strategy_name, C_values in strategies.items():
        print(f"\n{strategy_name}:")
        print(f"C values: {C_values[:5]}...{C_values[-5:]}")
        
        # Perform grid search
        param_grid = {'C': C_values}
        grid_search = GridSearchCV(SVC(kernel='rbf', random_state=42), 
                                 param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X, y)
        
        grid_results[strategy_name] = {
            'best_C': grid_search.best_params_['C'],
            'best_score': grid_search.best_score_,
            'C_values': C_values,
            'scores': grid_search.cv_results_['mean_test_score']
        }
        
        print(f"Best C: {grid_search.best_params_['C']:.4f}")
        print(f"Best CV Score: {grid_search.best_score_:.4f}")
    
    return grid_results, X, y

grid_results, X_grid, y_grid = demonstrate_grid_search()

# Visualize grid search results
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Grid Search Strategies for C Parameter Optimization', fontsize=16)

for i, (strategy_name, results) in enumerate(grid_results.items()):
    row, col = i // 2, i % 2
    
    # Plot C vs CV score
    axes[row, col].semilogx(results['C_values'], results['scores'], 'bo-', 
                           linewidth=2, markersize=6)
    axes[row, col].axvline(results['best_C'], color='red', linestyle='--', 
                          linewidth=2, label=f"Best C = {results['best_C']:.4f}")
    axes[row, col].set_xlabel('C Parameter')
    axes[row, col].set_ylabel('Cross-Validation Score')
    axes[row, col].set_title(f'{strategy_name}')
    axes[row, col].grid(True, alpha=0.3)
    axes[row, col].legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'grid_search_strategies.png'), dpi=300, bbox_inches='tight')

# 4. Detecting when C is too small or too large
print("\n4. DETECTING WHEN C IS TOO SMALL OR TOO LARGE")
print("-" * 40)

def analyze_C_effects():
    """Analyze the effects of different C values on SVM characteristics"""
    
    # Create dataset with some noise
    X, y = make_classification(n_samples=150, n_features=2, n_redundant=0, 
                             n_informative=2, n_clusters_per_class=1, 
                             random_state=42)
    X += np.random.normal(0, 0.3, X.shape)  # Add noise
    
    C_values = np.logspace(-3, 3, 20)
    analysis_results = []
    
    for C in C_values:
        svm = SVC(C=C, kernel='linear', random_state=42)
        svm.fit(X, y)
        
        # Calculate characteristics
        decision_values = svm.decision_function(X)
        slack_vars = np.maximum(0, 1 - y * decision_values)
        total_slack = np.sum(slack_vars)
        margin_width = 2 / np.linalg.norm(svm.coef_[0]) if len(svm.support_vectors_) > 0 else float('inf')
        
        analysis_results.append({
            'C': C,
            'total_slack': total_slack,
            'margin_width': margin_width,
            'n_support_vectors': len(svm.support_vectors_),
            'training_accuracy': accuracy_score(y, svm.predict(X)),
            'avg_slack': np.mean(slack_vars)
        })
    
    return analysis_results, X, y

analysis_results, X_analysis, y_analysis = analyze_C_effects()

# Print analysis for extreme C values
print("C too small indicators:")
small_C_results = [r for r in analysis_results if r['C'] <= 0.1]
for r in small_C_results[-3:]:
    print(f"C = {r['C']:.3f}: Total slack = {r['total_slack']:.2f}, "
          f"Margin width = {r['margin_width']:.2f}, "
          f"Support vectors = {r['n_support_vectors']}")

print("\nC too large indicators:")
large_C_results = [r for r in analysis_results if r['C'] >= 10]
for r in large_C_results[:3]:
    print(f"C = {r['C']:.1f}: Total slack = {r['total_slack']:.2f}, "
          f"Margin width = {r['margin_width']:.2f}, "
          f"Support vectors = {r['n_support_vectors']}")

# Visualize C analysis
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Detecting When C is Too Small or Too Large', fontsize=16)

C_vals = [r['C'] for r in analysis_results]

# Total slack vs C
slack_vals = [r['total_slack'] for r in analysis_results]
axes[0, 0].semilogx(C_vals, slack_vals, 'bo-', linewidth=2, markersize=6)
axes[0, 0].axvline(0.1, color='red', linestyle='--', alpha=0.7, label='C too small')
axes[0, 0].axvline(10, color='orange', linestyle='--', alpha=0.7, label='C too large')
axes[0, 0].set_xlabel('C Parameter')
axes[0, 0].set_ylabel('Total Slack Variables')
axes[0, 0].set_title('Total Slack vs C')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# Margin width vs C
margin_vals = [r['margin_width'] for r in analysis_results]
axes[0, 1].semilogx(C_vals, margin_vals, 'ro-', linewidth=2, markersize=6)
axes[0, 1].axvline(0.1, color='red', linestyle='--', alpha=0.7)
axes[0, 1].axvline(10, color='orange', linestyle='--', alpha=0.7)
axes[0, 1].set_xlabel('C Parameter')
axes[0, 1].set_ylabel('Margin Width')
axes[0, 1].set_title('Margin Width vs C')
axes[0, 1].grid(True, alpha=0.3)

# Support vectors vs C
sv_vals = [r['n_support_vectors'] for r in analysis_results]
axes[0, 2].semilogx(C_vals, sv_vals, 'go-', linewidth=2, markersize=6)
axes[0, 2].axvline(0.1, color='red', linestyle='--', alpha=0.7)
axes[0, 2].axvline(10, color='orange', linestyle='--', alpha=0.7)
axes[0, 2].set_xlabel('C Parameter')
axes[0, 2].set_ylabel('Number of Support Vectors')
axes[0, 2].set_title('Support Vectors vs C')
axes[0, 2].grid(True, alpha=0.3)

# Training accuracy vs C
acc_vals = [r['training_accuracy'] for r in analysis_results]
axes[1, 0].semilogx(C_vals, acc_vals, 'mo-', linewidth=2, markersize=6)
axes[1, 0].axvline(0.1, color='red', linestyle='--', alpha=0.7)
axes[1, 0].axvline(10, color='orange', linestyle='--', alpha=0.7)
axes[1, 0].set_xlabel('C Parameter')
axes[1, 0].set_ylabel('Training Accuracy')
axes[1, 0].set_title('Training Accuracy vs C')
axes[1, 0].grid(True, alpha=0.3)

# Average slack vs C
avg_slack_vals = [r['avg_slack'] for r in analysis_results]
axes[1, 1].semilogx(C_vals, avg_slack_vals, 'co-', linewidth=2, markersize=6)
axes[1, 1].axvline(0.1, color='red', linestyle='--', alpha=0.7)
axes[1, 1].axvline(10, color='orange', linestyle='--', alpha=0.7)
axes[1, 1].set_xlabel('C Parameter')
axes[1, 1].set_ylabel('Average Slack')
axes[1, 1].set_title('Average Slack vs C')
axes[1, 1].grid(True, alpha=0.3)

# Decision boundaries for different C values
C_examples = [0.01, 1, 100]
for i, C in enumerate(C_examples):
    svm = SVC(C=C, kernel='linear', random_state=42)
    svm.fit(X_analysis, y_analysis)
    
    # Create mesh grid
    x_min, x_max = X_analysis[:, 0].min() - 0.5, X_analysis[:, 0].max() + 0.5
    y_min, y_max = X_analysis[:, 1].min() - 0.5, X_analysis[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axes[1, 2].contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    axes[1, 2].scatter(X_analysis[:, 0], X_analysis[:, 1], c=y_analysis, 
                       cmap='viridis', alpha=0.7, s=20)
    axes[1, 2].set_title(f'Decision Boundaries\n(C=0.01, 1, 100)')
    axes[1, 2].set_xlabel('$x_1$')
    axes[1, 2].set_ylabel('$x_2$')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'C_analysis.png'), dpi=300, bbox_inches='tight')

# 5. Stopping criteria for iterative optimization algorithms
print("\n5. STOPPING CRITERIA FOR ITERATIVE OPTIMIZATION")
print("-" * 40)

def demonstrate_stopping_criteria():
    """Demonstrate different stopping criteria for SVM optimization"""
    
    # Create a challenging dataset
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                             n_informative=2, n_clusters_per_class=1, 
                             random_state=42)
    X += np.random.normal(0, 0.2, X.shape)
    
    # Test different stopping criteria
    stopping_criteria = {
        'Tolerance': 1e-3,
        'Max Iterations': 1000,
        'Convergence': 'auto'
    }
    
    stopping_results = {}
    
    for criterion_name, value in stopping_criteria.items():
        print(f"\n{criterion_name} = {value}:")
        
        if criterion_name == 'Tolerance':
            svm = SVC(C=1, kernel='linear', tol=value, random_state=42)
        elif criterion_name == 'Max Iterations':
            svm = SVC(C=1, kernel='linear', max_iter=value, random_state=42)
        else:
            svm = SVC(C=1, kernel='linear', random_state=42)
        
        # Time the training
        import time
        start_time = time.time()
        svm.fit(X, y)
        training_time = time.time() - start_time
        
        # Get optimization info
        n_iter = getattr(svm, 'n_iter_', 'N/A')
        accuracy = accuracy_score(y, svm.predict(X))
        
        stopping_results[criterion_name] = {
            'training_time': training_time,
            'n_iterations': n_iter,
            'accuracy': accuracy,
            'n_support_vectors': len(svm.support_vectors_)
        }
        
        print(f"Training time: {training_time:.4f} seconds")
        print(f"Iterations: {n_iter}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Support vectors: {len(svm.support_vectors_)}")
    
    return stopping_results, X, y

stopping_results, X_stop, y_stop = demonstrate_stopping_criteria()

# Visualize stopping criteria effects
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Stopping Criteria Analysis', fontsize=16)

# Training time comparison
criteria = list(stopping_results.keys())
times = [stopping_results[c]['training_time'] for c in criteria]
iterations = [stopping_results[c]['n_iterations'] for c in criteria]
accuracies = [stopping_results[c]['accuracy'] for c in criteria]

axes[0].bar(criteria, times, alpha=0.8, color=['blue', 'green', 'red'])
axes[0].set_xlabel('Stopping Criterion')
axes[0].set_ylabel('Training Time (seconds)')
axes[0].set_title('Training Time Comparison')
axes[0].grid(True, alpha=0.3)

axes[1].bar(criteria, accuracies, alpha=0.8, color=['blue', 'green', 'red'])
axes[1].set_xlabel('Stopping Criterion')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy Comparison')
axes[1].grid(True, alpha=0.3)

# Support vectors comparison
sv_counts = [stopping_results[c]['n_support_vectors'] for c in criteria]
axes[2].bar(criteria, sv_counts, alpha=0.8, color=['blue', 'green', 'red'])
axes[2].set_xlabel('Stopping Criterion')
axes[2].set_ylabel('Number of Support Vectors')
axes[2].set_title('Support Vectors Comparison')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'stopping_criteria.png'), dpi=300, bbox_inches='tight')

print(f"\nAll plots saved to: {save_dir}")
print("\nSummary of Implementation Considerations:")
print("1. For all outliers: Use very small C values or robust preprocessing")
print("2. Essential preprocessing: Standardization, handling outliers, feature scaling")
print("3. Grid search: Use logarithmic grid for C parameter optimization")
print("4. C too small: High slack variables, wide margins, many support vectors")
print("5. C too large: Low slack variables, narrow margins, overfitting")
print("6. Stopping criteria: Balance between convergence and computational cost")
