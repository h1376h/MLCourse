import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
import os
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_7_Quiz_6")
os.makedirs(save_dir, exist_ok=True)

# Set the style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Function to create correlated features
def create_correlated_features(n_samples=100, n_features=10, corr_strength=0.85, noise=0.5, random_state=42):
    """
    Create a dataset with correlated features.
    
    Parameters:
    - n_samples: Number of samples
    - n_features: Number of features
    - corr_strength: Correlation strength between features
    - noise: Noise level
    - random_state: Random seed
    
    Returns:
    - X: Feature matrix
    - y: Target vector
    - beta: True coefficients
    """
    np.random.seed(random_state)
    
    # Create a covariance matrix with high correlation
    cov_matrix = np.ones((n_features, n_features)) * corr_strength
    np.fill_diagonal(cov_matrix, 1)
    
    # Generate correlated features
    X = np.random.multivariate_normal(mean=np.zeros(n_features), cov=cov_matrix, size=n_samples)
    
    # Create the true coefficients (only a few non-zero)
    beta = np.zeros(n_features)
    beta[:3] = [3, 1.5, 2]  # Only first 3 coefficients are non-zero
    
    # Generate the target with noise
    y = X.dot(beta) + np.random.normal(0, noise, n_samples)
    
    return X, y, beta

# Print explanations
print("\nElastic Net Regression - Balancing L1 and L2 Regularization")
print("===========================================================")
print("\nElastic Net Objective Function:")
print("J(w) = ||y - Xw||^2 + λ1||w||₁ + λ2||w||₂²")
print("\nWhere:")
print("- ||y - Xw||^2 is the sum of squared errors")
print("- λ1||w||₁ is the L1 penalty (sum of absolute values of coefficients)")
print("- λ2||w||₂² is the L2 penalty (sum of squared coefficients)")

# Example 1: Performance on Data with Highly Correlated Features
print("\n\nExample 1: Performance with Highly Correlated Features")
print("----------------------------------------------------")

# Create a dataset with highly correlated features
n_samples = 200
n_features = 10
X, y, true_coef = create_correlated_features(n_samples=n_samples, n_features=n_features, 
                                            corr_strength=0.85, noise=0.5)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
alphas = [0.01, 0.1, 1.0, 10.0]
l1_ratios = [0.1, 0.5, 0.9]

# Initialize dictionaries to store results
coef_paths = {}
mse_results = {"Ridge": [], "Lasso": [], "ElasticNet": []}

# Ridge Regression (L2 penalty)
print("\nRidge Regression Results:")
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    y_pred = ridge.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    mse_results["Ridge"].append(mse)
    coef_paths[f"Ridge (α={alpha})"] = ridge.coef_
    print(f"  α={alpha:.2f}, MSE={mse:.4f}, Coefficients: {ridge.coef_.round(2)}")

# Lasso Regression (L1 penalty)
print("\nLasso Regression Results:")
for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    y_pred = lasso.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    mse_results["Lasso"].append(mse)
    coef_paths[f"Lasso (α={alpha})"] = lasso.coef_
    print(f"  α={alpha:.2f}, MSE={mse:.4f}, Coefficients: {lasso.coef_.round(2)}")

# Elastic Net
print("\nElastic Net Results:")
for alpha in alphas:
    for l1_ratio in l1_ratios:
        elastic = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
        elastic.fit(X_train_scaled, y_train)
        y_pred = elastic.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        if l1_ratio == 0.5:  # Only track middle l1_ratio for plotting
            mse_results["ElasticNet"].append(mse)
        coef_paths[f"ElasticNet (α={alpha}, L1={l1_ratio})"] = elastic.coef_
        print(f"  α={alpha:.2f}, L1 ratio={l1_ratio:.1f}, MSE={mse:.4f}, Coefficients: {elastic.coef_.round(2)}")

# Visualize results

# Plot 1: Coefficient Paths Comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Ridge coefficient comparison
ax = axes[0]
for key, coef in coef_paths.items():
    if 'Ridge' in key:
        ax.plot(range(len(coef)), coef, 'o-', label=key)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.set_ylabel('Coefficient Value')
ax.set_xlabel('Feature Index')
ax.set_title('Ridge Coefficients (L2 Penalty)')
ax.set_xticks(range(n_features))
ax.legend()

# Lasso coefficient comparison
ax = axes[1]
for key, coef in coef_paths.items():
    if 'Lasso' in key and 'Elastic' not in key:
        ax.plot(range(len(coef)), coef, 'o-', label=key)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.set_ylabel('Coefficient Value')
ax.set_xlabel('Feature Index')
ax.set_title('Lasso Coefficients (L1 Penalty)')
ax.set_xticks(range(n_features))
ax.legend()

# ElasticNet coefficient comparison
ax = axes[2]
for key, coef in coef_paths.items():
    if 'ElasticNet' in key and 'L1=0.5' in key:
        ax.plot(range(len(coef)), coef, 'o-', label=key)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.set_ylabel('Coefficient Value')
ax.set_xlabel('Feature Index')
ax.set_title('Elastic Net Coefficients (L1=0.5)')
ax.set_xticks(range(n_features))
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "elastic_net_coefficient_paths.png"), dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Compare MSE for different alpha values
plt.figure(figsize=(10, 6))
for method, mses in mse_results.items():
    plt.plot(alphas, mses, 'o-', label=method)
plt.xscale('log')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs. Regularization Strength')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, "elastic_net_mse_comparison.png"), dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: L1 Ratio Impact on Elastic Net
alphas_fine = np.logspace(-3, 1, 30)
l1_ratios = [0.1, 0.5, 0.9]
colors = ['blue', 'green', 'red']

plt.figure(figsize=(10, 6))
for i, l1_ratio in enumerate(l1_ratios):
    mses = []
    for alpha in alphas_fine:
        elastic = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
        elastic.fit(X_train_scaled, y_train)
        y_pred = elastic.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        mses.append(mse)
    plt.plot(alphas_fine, mses, 'o-', color=colors[i], label=f'L1 ratio={l1_ratio}')

plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Mean Squared Error')
plt.title('Impact of L1 Ratio on Elastic Net Performance')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, "elastic_net_l1_ratio_impact.png"), dpi=300, bbox_inches='tight')
plt.close()

# Example 2: Feature Selection Properties
print("\n\nExample 2: Feature Selection Properties")
print("-------------------------------------")

# Create a dataset with more features and stronger correlations
X, y, true_coef = create_correlated_features(n_samples=200, n_features=20, 
                                           corr_strength=0.9, noise=0.7)

# Compute correlation matrix
corr_matrix = np.corrcoef(X.T)

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Feature Correlation Matrix')
plt.savefig(os.path.join(save_dir, "feature_correlation_matrix.png"), dpi=300, bbox_inches='tight')
plt.close()

# Split data and standardize
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models with fixed alpha but varying L1 ratios for Elastic Net
alpha = 0.1
l1_ratios = np.linspace(0, 1, 11)  # From 0 (Ridge) to 1 (Lasso)

coefs = []
for l1_ratio in l1_ratios:
    if l1_ratio == 0:
        # Pure Ridge regression
        model = Ridge(alpha=alpha)
    elif l1_ratio == 1:
        # Pure Lasso regression
        model = Lasso(alpha=alpha, max_iter=10000)
    else:
        # Elastic Net with varying L1 ratio
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
    
    model.fit(X_train_scaled, y_train)
    coefs.append(model.coef_)

# Visualize coefficient paths as L1 ratio changes
plt.figure(figsize=(12, 8))
coefs_array = np.array(coefs)

for i in range(X.shape[1]):
    plt.plot(l1_ratios, coefs_array[:, i], '-', label=f'Feature {i+1}')

plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel('L1 Ratio (0=Ridge, 1=Lasso)')
plt.ylabel('Coefficient Value')
plt.title('Coefficient Paths as L1 Ratio Changes (α=0.1)')
plt.grid(True)

# Highlight the true non-zero coefficients
for i in range(len(true_coef)):
    if true_coef[i] != 0:
        plt.plot(l1_ratios, coefs_array[:, i], 'o-', linewidth=2, label=f'True Feature {i+1}')

plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1), ncol=2)
plt.savefig(os.path.join(save_dir, "elastic_net_coefficient_l1_ratio_paths.png"), dpi=300, bbox_inches='tight')
plt.close()

# Example 3: Parameter Selection with Grid Search
print("\n\nExample 3: Parameter Selection with Grid Search")
print("--------------------------------------------")

# Create a new dataset
X, y, _ = create_correlated_features(n_samples=200, n_features=15, corr_strength=0.8, noise=0.6)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Set up grid search
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}

elastic_net = ElasticNet(max_iter=10000)
grid_search = GridSearchCV(estimator=elastic_net, param_grid=param_grid, 
                          scoring='neg_mean_squared_error', cv=5)

# Run grid search
grid_search.fit(X_train_scaled, y_train)

# Print results
print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best MSE: {-grid_search.best_score_:.4f}")

# Create heatmap of results
results = pd.DataFrame(grid_search.cv_results_)
scores = results.pivot(index='param_alpha', columns='param_l1_ratio', values='mean_test_score')
scores = -scores  # Convert back to MSE from negative MSE

plt.figure(figsize=(10, 8))
sns.heatmap(scores, annot=True, fmt='.4f', cmap='viridis_r')
plt.title('Grid Search Results: MSE for Different α and L1 Ratio Values')
plt.xlabel('L1 Ratio')
plt.ylabel('Alpha')
plt.savefig(os.path.join(save_dir, "elastic_net_grid_search.png"), dpi=300, bbox_inches='tight')
plt.close()

# Example 4: 3D Visualization of L1 and L2 Constraints
print("\n\nExample 4: Geometric Interpretation of Elastic Net")
print("----------------------------------------------")

# Create a simple 3D visualization of combined L1 and L2 constraints
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Generate points on the surface of L1 and L2 constraints
n_points = 100
theta = np.linspace(0, 2 * np.pi, n_points)
x = np.cos(theta)
y = np.sin(theta)

# L2 unit ball (circle in 2D)
ax.plot(x, y, 0, 'b-', linewidth=2, label='L2 Constraint (Ridge)')

# L1 unit ball (diamond in 2D)
diamond_x = np.array([1, 0, -1, 0, 1])
diamond_y = np.array([0, 1, 0, -1, 0])
ax.plot(diamond_x, diamond_y, 0, 'r-', linewidth=2, label='L1 Constraint (Lasso)')

# Elastic Net constraint (mixture of L1 and L2)
l1_ratio = 0.5
elastic_x = []
elastic_y = []

for t in theta:
    # Parametric equation for elastic net constraint in 2D
    px = np.cos(t)
    py = np.sin(t)
    p_norm = l1_ratio * (abs(px) + abs(py)) + (1 - l1_ratio) * np.sqrt(px**2 + py**2)
    elastic_x.append(px / p_norm)
    elastic_y.append(py / p_norm)

ax.plot(elastic_x, elastic_y, 0, 'g-', linewidth=2, label='Elastic Net Constraint')

# Set labels using LaTeX for subscripts instead of Unicode
ax.set_xlabel(r'$w_1$')
ax.set_ylabel(r'$w_2$')
ax.set_zlabel('')
ax.set_title('Geometric Interpretation of Regularization Constraints')
ax.set_box_aspect([1, 1, 0.01])  # Flatten the z-dimension
ax.view_init(elev=30, azim=-45)
ax.legend()

plt.savefig(os.path.join(save_dir, "elastic_net_geometry.png"), dpi=300, bbox_inches='tight')
plt.close()

# Example 5: Elastic Net Behavior in the Presence of Correlated Features
print("\n\nExample 5: Elastic Net Behavior with Grouped Features")
print("--------------------------------------------------")

# Generate a highly correlated group of features
np.random.seed(42)
n_samples = 150
n_features = 10

# Generate a base feature
X_base = np.random.randn(n_samples, 1)

# Generate correlated features by adding noise to the base feature
noise_levels = [0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
X = np.zeros((n_samples, n_features))

for i in range(n_features):
    X[:, i] = X_base.flatten() + np.random.normal(0, noise_levels[i], n_samples)

# True coefficients - first three features are important
beta = np.zeros(n_features)
beta[:3] = [1.5, 1.3, 1.4]  # Similar coefficients for correlated features

# Generate response
y = X.dot(beta) + np.random.normal(0, 0.5, n_samples)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Calculate correlation matrix for the first few features
corr_subset = np.corrcoef(X[:, :5].T)

# Visualize correlations
plt.figure(figsize=(8, 6))
sns.heatmap(corr_subset, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix for First 5 Features')
plt.savefig(os.path.join(save_dir, "correlated_features_group.png"), dpi=300, bbox_inches='tight')
plt.close()

# Compare models on correlated groups
alphas = [0.01, 0.1, 1.0]
models = {}

for alpha in alphas:
    # Ridge
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    models[f'Ridge (α={alpha})'] = ridge.coef_
    
    # Lasso
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    models[f'Lasso (α={alpha})'] = lasso.coef_
    
    # Elastic Net with different L1 ratios
    for l1_ratio in [0.2, 0.5, 0.8]:
        elastic = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
        elastic.fit(X_train_scaled, y_train)
        models[f'ElasticNet (α={alpha}, L1={l1_ratio})'] = elastic.coef_

# Visualize coefficients for correlated groups
plt.figure(figsize=(15, 10))

# Plot true coefficients
plt.subplot(3, 1, 1)
plt.bar(range(n_features), beta, color='green', alpha=0.7)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.title('True Coefficients')
plt.ylabel('Value')
plt.xticks(range(n_features))

# Plot coefficients from different models (α=0.1)
plt.subplot(3, 1, 2)
width = 0.15
x = np.arange(n_features)
plt.bar(x - 2*width, models['Ridge (α=0.1)'], width=width, label='Ridge', color='blue', alpha=0.7)
plt.bar(x - width, models['Lasso (α=0.1)'], width=width, label='Lasso', color='red', alpha=0.7)
plt.bar(x, models['ElasticNet (α=0.1, L1=0.2)'], width=width, label='ElasticNet (L1=0.2)', color='purple', alpha=0.7)
plt.bar(x + width, models['ElasticNet (α=0.1, L1=0.5)'], width=width, label='ElasticNet (L1=0.5)', color='orange', alpha=0.7)
plt.bar(x + 2*width, models['ElasticNet (α=0.1, L1=0.8)'], width=width, label='ElasticNet (L1=0.8)', color='green', alpha=0.7)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.title('Model Coefficients (α=0.1)')
plt.ylabel('Value')
plt.xticks(range(n_features))
plt.legend()

# Focus on the correlated group (first 3 features)
plt.subplot(3, 1, 3)
for i, model_name in enumerate(['Ridge (α=0.1)', 'Lasso (α=0.1)', 'ElasticNet (α=0.1, L1=0.5)']):
    plt.bar(np.arange(3) + (i-1)*0.2, models[model_name][:3], width=0.2, 
            label=model_name, alpha=0.7)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.title('Coefficients for Correlated Features (First 3)')
plt.ylabel('Value')
plt.xticks([0, 1, 2], ['Feature 1', 'Feature 2', 'Feature 3'])
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "elastic_net_group_effect.png"), dpi=300, bbox_inches='tight')
plt.close()

# Example 6: Feature Selection Stability Comparison
print("\n\nExample 6: Feature Selection Stability Across Bootstrap Samples")
print("-----------------------------------------------------------")

# Generate a dataset with correlated features
np.random.seed(123)
n_samples = 200
n_features = 15
X, y, true_coef = create_correlated_features(n_samples=n_samples, n_features=n_features, 
                                           corr_strength=0.85, noise=0.7)

# Parameters for the experiment
n_bootstraps = 50
bootstrap_size = int(0.8 * n_samples)
alpha_value = 0.1
methods = ['Ridge', 'Lasso', 'ElasticNet-0.5']

# Storage for feature selection results
selection_matrix = {method: np.zeros((n_bootstraps, n_features)) for method in methods}

# Run bootstrap experiment
for b in range(n_bootstraps):
    # Create bootstrap sample
    indices = np.random.choice(n_samples, size=bootstrap_size, replace=True)
    X_boot, y_boot = X[indices], y[indices]
    
    # Standardize
    X_boot_scaled = scaler.fit_transform(X_boot)
    
    # Apply each method
    # Ridge
    ridge = Ridge(alpha=alpha_value)
    ridge.fit(X_boot_scaled, y_boot)
    selection_matrix['Ridge'][b] = np.abs(ridge.coef_) > 0.1  # Threshold for "selection"
    
    # Lasso
    lasso = Lasso(alpha=alpha_value, max_iter=10000)
    lasso.fit(X_boot_scaled, y_boot)
    selection_matrix['Lasso'][b] = np.abs(lasso.coef_) > 0.01  # Smaller threshold as Lasso zeros out coefficients
    
    # Elastic Net
    elastic = ElasticNet(alpha=alpha_value, l1_ratio=0.5, max_iter=10000)
    elastic.fit(X_boot_scaled, y_boot)
    selection_matrix['ElasticNet-0.5'][b] = np.abs(elastic.coef_) > 0.05

# Calculate selection frequencies
selection_freq = {method: np.mean(selection_matrix[method], axis=0) for method in methods}

# Create a heatmap of the selection frequencies
plt.figure(figsize=(12, 8))
data = np.vstack([selection_freq[method] for method in methods])
# Fix the colorbar issue by assigning the heatmap to a variable
heatmap = sns.heatmap(data, annot=False, cmap='viridis', vmin=0, vmax=1, 
            xticklabels=range(1, n_features+1), yticklabels=methods)
plt.title('Feature Selection Stability Across Bootstrap Samples')
plt.xlabel('Feature Index')
plt.ylabel('Method')
# The colorbar is now automatically created by the heatmap function
plt.savefig(os.path.join(save_dir, "elastic_net_selection_stability.png"), dpi=300, bbox_inches='tight')
plt.close()

# Create a detailed variation visualization 
plt.figure(figsize=(15, 10))

# Plot selection consistency for top features
for i, method in enumerate(methods):
    plt.subplot(3, 1, i+1)
    # For each feature, create a small plot of selections across bootstraps
    feature_selections = selection_matrix[method]
    plt.imshow(feature_selections, aspect='auto', cmap='binary', interpolation='none')
    plt.colorbar(label='Selected', ticks=[0, 1])
    plt.title(f'{method} Selection Patterns Across Bootstrap Samples')
    plt.xlabel('Feature Index')
    plt.ylabel('Bootstrap Sample')
    plt.yticks(np.arange(0, n_bootstraps, 10))
    
    # Highlight the true non-zero coefficient features
    true_features = np.where(true_coef != 0)[0]
    for feat in true_features:
        plt.axvline(x=feat, color='red', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "elastic_net_selection_patterns.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nVisualizations saved to: {save_dir}") 