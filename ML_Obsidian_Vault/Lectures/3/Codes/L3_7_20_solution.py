import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
import os
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d, Axes3D

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_7_Quiz_20")
os.makedirs(save_dir, exist_ok=True)

# Set seed for reproducibility
np.random.seed(42)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Define function for saving plots
def save_plot(filename):
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# Print explanation header
print("\nRegularization Methods in Linear Regression")
print("==========================================")

# Part 1: Elastic Net vs Lasso vs Ridge regression
print("\n1. Elastic Net vs Lasso vs Ridge Regression")
print("------------------------------------------")

# Generate synthetic data with some irrelevant features
X, y = make_regression(n_samples=100, n_features=20, n_informative=5, 
                      noise=20.0, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models with different regularization approaches
alphas = np.logspace(-2, 2, 50)

# Lists to store results
ridge_coefs = []
lasso_coefs = []
elastic_coefs = []
ridge_mse = []
lasso_mse = []
elastic_mse = []

# Calculate results for different alpha values
for alpha in alphas:
    # Ridge Regression
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    ridge_coefs.append(ridge.coef_)
    y_pred_ridge = ridge.predict(X_test_scaled)
    ridge_mse.append(mean_squared_error(y_test, y_pred_ridge))
    
    # Lasso Regression
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    lasso_coefs.append(lasso.coef_)
    y_pred_lasso = lasso.predict(X_test_scaled)
    lasso_mse.append(mean_squared_error(y_test, y_pred_lasso))
    
    # Elastic Net (with alpha distribution: 0.5 L1, 0.5 L2)
    elastic = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000)
    elastic.fit(X_train_scaled, y_train)
    elastic_coefs.append(elastic.coef_)
    y_pred_elastic = elastic.predict(X_test_scaled)
    elastic_mse.append(mean_squared_error(y_test, y_pred_elastic))

# Print results
best_ridge_idx = np.argmin(ridge_mse)
best_lasso_idx = np.argmin(lasso_mse)
best_elastic_idx = np.argmin(elastic_mse)

print(f"Best Ridge alpha: {alphas[best_ridge_idx]:.4f}, MSE: {ridge_mse[best_ridge_idx]:.4f}")
print(f"Best Lasso alpha: {alphas[best_lasso_idx]:.4f}, MSE: {lasso_mse[best_lasso_idx]:.4f}")
print(f"Best Elastic Net alpha: {alphas[best_elastic_idx]:.4f}, MSE: {elastic_mse[best_elastic_idx]:.4f}")

print(f"Number of non-zero coefficients:")
print(f"Ridge: {np.sum(np.abs(ridge_coefs[best_ridge_idx]) > 1e-3)}")
print(f"Lasso: {np.sum(np.abs(lasso_coefs[best_lasso_idx]) > 1e-3)}")
print(f"Elastic Net: {np.sum(np.abs(elastic_coefs[best_elastic_idx]) > 1e-3)}")

# Plot MSE vs regularization strength
plt.figure(figsize=(10, 6))
plt.plot(alphas, ridge_mse, '-o', label='Ridge', markersize=4)
plt.plot(alphas, lasso_mse, '-o', label='Lasso', markersize=4)
plt.plot(alphas, elastic_mse, '-o', label='Elastic Net', markersize=4)
plt.xscale('log')
plt.xlabel('Regularization parameter (alpha)', fontsize=12)
plt.ylabel('Mean Squared Error (Test)', fontsize=12)
plt.title('Comparison of Regularization Methods: Test Error vs Regularization Strength', fontsize=14)
plt.legend()
plt.grid(True)
save_plot("elastic_vs_lasso_ridge_mse.png")

# Plot coefficients vs regularization strength for each method
plt.figure(figsize=(15, 12))

plt.subplot(3, 1, 1)
ridge_coefs_array = np.array(ridge_coefs)
for i in range(X.shape[1]):
    plt.semilogx(alphas, ridge_coefs_array[:, i], label=f'Feature {i+1}' if i < 5 else '')
plt.xlabel('Regularization parameter (alpha)', fontsize=12)
plt.ylabel('Coefficient value', fontsize=12)
plt.title('Ridge Regression: Coefficient Paths', fontsize=14)
if X.shape[1] <= 10:
    plt.legend(loc='upper right')
else:
    plt.plot([], [], '-', label=f'Features 1-5 shown', color='black')
    plt.legend(loc='upper right')
plt.grid(True)

plt.subplot(3, 1, 2)
lasso_coefs_array = np.array(lasso_coefs)
for i in range(X.shape[1]):
    plt.semilogx(alphas, lasso_coefs_array[:, i], label=f'Feature {i+1}' if i < 5 else '')
plt.xlabel('Regularization parameter (alpha)', fontsize=12)
plt.ylabel('Coefficient value', fontsize=12)
plt.title('Lasso Regression: Coefficient Paths', fontsize=14)
if X.shape[1] <= 10:
    plt.legend(loc='upper right')
else:
    plt.plot([], [], '-', label=f'Features 1-5 shown', color='black')
    plt.legend(loc='upper right')
plt.grid(True)

plt.subplot(3, 1, 3)
elastic_coefs_array = np.array(elastic_coefs)
for i in range(X.shape[1]):
    plt.semilogx(alphas, elastic_coefs_array[:, i], label=f'Feature {i+1}' if i < 5 else '')
plt.xlabel('Regularization parameter (alpha)', fontsize=12)
plt.ylabel('Coefficient value', fontsize=12)
plt.title('Elastic Net: Coefficient Paths', fontsize=14)
if X.shape[1] <= 10:
    plt.legend(loc='upper right')
else:
    plt.plot([], [], '-', label=f'Features 1-5 shown', color='black')
    plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
save_plot("regularization_coefficient_paths.png")

# Part 2: Early stopping as regularization
print("\n2. Early Stopping as Regularization")
print("----------------------------------")

# Function to perform gradient descent with tracking
def gradient_descent(X, y, alpha=0.01, max_iter=1000):
    n_samples, n_features = X.shape
    # Initialize parameters randomly
    theta = np.random.randn(n_features)
    
    # Track errors and parameters
    train_errors = []
    test_errors = []
    thetas = []
    
    # Create a small validation set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    for i in range(max_iter):
        # Calculate predictions and errors for current parameters
        y_pred_train = X_train @ theta
        error_train = y_train - y_pred_train
        mse_train = np.sum(error_train**2) / len(y_train)
        train_errors.append(mse_train)
        
        y_pred_val = X_val @ theta
        error_val = y_val - y_pred_val
        mse_val = np.sum(error_val**2) / len(y_val)
        test_errors.append(mse_val)
        
        # Store current parameters
        thetas.append(theta.copy())
        
        # Update parameters
        gradient = -2 * X_train.T @ error_train / len(y_train)
        theta = theta - alpha * gradient
    
    return train_errors, test_errors, thetas

# Create a noisy regression problem
X, y = make_regression(n_samples=200, n_features=15, n_informative=10, 
                      noise=30.0, random_state=42)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform gradient descent
train_errors, val_errors, thetas = gradient_descent(X_scaled, y, alpha=0.01, max_iter=1000)

# Find early stopping point (minimum validation error)
early_stop_idx = np.argmin(val_errors)
print(f"Early stopping point: Iteration {early_stop_idx}")
print(f"Training MSE at early stopping: {train_errors[early_stop_idx]:.2f}")
print(f"Validation MSE at early stopping: {val_errors[early_stop_idx]:.2f}")
print(f"Final training MSE: {train_errors[-1]:.2f}")
print(f"Final validation MSE: {val_errors[-1]:.2f}")

# Calculate L2 norm of weights over iterations
weight_norms = [np.linalg.norm(theta) for theta in thetas]

# Plot errors and early stopping point
plt.figure(figsize=(10, 6))
plt.plot(train_errors, label='Training Error')
plt.plot(val_errors, label='Validation Error')
plt.axvline(x=early_stop_idx, color='r', linestyle='--', 
            label=f'Early Stopping (iter {early_stop_idx})')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.title('Early Stopping as Regularization', fontsize=14)
plt.legend()
plt.grid(True)
save_plot("early_stopping.png")

# Plot L2 norm of weights over iterations
plt.figure(figsize=(10, 6))
plt.plot(weight_norms, label='L2 Norm of Weights')
plt.axvline(x=early_stop_idx, color='r', linestyle='--', 
            label=f'Early Stopping (iter {early_stop_idx})')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('L2 Norm of Weights', fontsize=12)
plt.title('Weight Growth During Gradient Descent', fontsize=14)
plt.legend()
plt.grid(True)
save_plot("early_stopping_weight_norm.png")

# Part 3: Bayesian interpretation of regularization
print("\n3. Bayesian Interpretation of Regularization")
print("------------------------------------------")

# Generate some simple 1D data for Bayesian visualization
np.random.seed(42)
X_1d = np.sort(np.random.uniform(-3, 3, 10))
y_1d = 2 * X_1d + np.random.normal(0, 1.5, len(X_1d))
X_1d = X_1d.reshape(-1, 1)

def plot_bayesian_regression(X, y, prior_var, title_suffix=""):
    # Calculate posterior parameters analytically for linear regression
    # with Gaussian prior on weights with variance prior_var
    
    # Convert to numpy arrays
    X = np.asarray(X)
    y = np.asarray(y)
    
    # Assumed noise variance (likelihood)
    noise_var = 1.0
    
    # Prior parameters (mean and precision matrix)
    prior_mean = np.zeros(2)  # We include a bias term
    prior_precision = np.diag([1.0/prior_var, 1.0/prior_var])
    
    # Design matrix with intercept
    X_design = np.column_stack([np.ones(len(X)), X])
    
    # Posterior precision matrix
    posterior_precision = prior_precision + (1.0/noise_var) * X_design.T @ X_design
    
    # Posterior covariance matrix
    posterior_cov = np.linalg.inv(posterior_precision)
    
    # Posterior mean
    posterior_mean = posterior_cov @ ((1.0/noise_var) * X_design.T @ y)
    
    # Plot data and regression line with uncertainty
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.scatter(X, y, color='blue', s=50, label='Data points')
    
    # Plot true function
    x_line = np.linspace(-4, 4, 100)
    y_true = 2 * x_line
    plt.plot(x_line, y_true, 'g-', alpha=0.5, label='True function')
    
    # Plot posterior mean function
    X_line_design = np.column_stack([np.ones(len(x_line)), x_line.reshape(-1, 1)])
    y_posterior_mean = X_line_design @ posterior_mean
    plt.plot(x_line, y_posterior_mean, 'r-', label='Posterior mean')
    
    # Calculate uncertainty (1 standard deviation)
    std_dev = np.array([np.sqrt(X_line_design[i] @ posterior_cov @ X_line_design[i].T) 
                         for i in range(len(x_line))])
    
    # Plot uncertainty
    plt.fill_between(x_line, y_posterior_mean - 2*std_dev, 
                     y_posterior_mean + 2*std_dev, alpha=0.2, color='red',
                     label='95% confidence interval')
    
    # Also fit Ridge regression for comparison
    alpha = 1.0 / prior_var
    ridge = Ridge(alpha=alpha)
    X_ridge = X.copy()
    ridge.fit(X_ridge, y)
    
    # Plot Ridge prediction
    y_ridge = ridge.predict(x_line.reshape(-1, 1))
    plt.plot(x_line, y_ridge, 'b--', label='Ridge regression')
    
    # Display MAP estimate and regularization parameter
    intercept, slope = posterior_mean
    plt.annotate(f"MAP: y = {intercept:.2f} + {slope:.2f}x\nPrior Var: {prior_var:.2f}, λ = {1/prior_var:.2f}", 
                 xy=(0.05, 0.9), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.xlabel('X', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title(f'Bayesian Linear Regression: {title_suffix}', fontsize=14)
    plt.legend()
    plt.grid(True)
    save_plot(f"bayesian_regression_{prior_var:.2f}".replace(".", "_"))
    
    return posterior_mean, posterior_cov

# Plot with different prior variances (corresponding to different regularization strengths)
print("Bayesian Regression with different prior variances:")
results = {}

for prior_var in [0.1, 1.0, 10.0]:
    posterior_mean, posterior_cov = plot_bayesian_regression(
        X_1d, y_1d, prior_var, f"Prior Variance = {prior_var}")
    
    results[prior_var] = {
        'posterior_mean': posterior_mean,
        'posterior_cov': posterior_cov,
        'l2_reg': 1.0 / prior_var
    }
    
    print(f"Prior Variance: {prior_var}")
    print(f"  Regularization parameter (λ): {1.0 / prior_var}")
    print(f"  Posterior mean (intercept, slope): {posterior_mean}")
    print(f"  Posterior variance (slope): {posterior_cov[1, 1]}")
    print()

# Part 4: L1 vs L2 regularization and sparsity
print("\n4. L1 vs L2 Regularization and Sparsity")
print("--------------------------------------")

# Generate a high-dimensional dataset with only a few relevant features
X, y, coef = make_regression(n_samples=100, n_features=50, n_informative=5, 
                           coef=True, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply both Lasso and Ridge with the same alpha
alpha_value = 0.1

lasso = Lasso(alpha=alpha_value)
ridge = Ridge(alpha=alpha_value)

lasso.fit(X_train_scaled, y_train)
ridge.fit(X_train_scaled, y_train)

# Get predictions and errors
y_pred_lasso = lasso.predict(X_test_scaled)
y_pred_ridge = ridge.predict(X_test_scaled)

lasso_mse = mean_squared_error(y_test, y_pred_lasso)
ridge_mse = mean_squared_error(y_test, y_pred_ridge)

# Count non-zero coefficients
lasso_nonzero = np.sum(np.abs(lasso.coef_) > 1e-10)
ridge_nonzero = np.sum(np.abs(ridge.coef_) > 1e-10)

print(f"Lasso (L1) - Number of non-zero coefficients: {lasso_nonzero} / {len(lasso.coef_)}")
print(f"Ridge (L2) - Number of non-zero coefficients: {ridge_nonzero} / {len(ridge.coef_)}")
print(f"Lasso (L1) - Test MSE: {lasso_mse:.4f}")
print(f"Ridge (L2) - Test MSE: {ridge_mse:.4f}")

# Plot coefficients sorted by true importance
sorted_idx = np.argsort(np.abs(coef))[::-1]  # Sort by magnitude of true coefficients
top_idx = sorted_idx[:20]  # Show top 20 coefficients

plt.figure(figsize=(12, 8))
width = 0.35
x = np.arange(len(top_idx))

plt.bar(x - width/2, coef[top_idx], width, label='True', alpha=0.7)
plt.bar(x + width/2, lasso.coef_[top_idx], width, label='Lasso (L1)', alpha=0.7)
plt.bar(x + 3*width/2, ridge.coef_[top_idx], width, label='Ridge (L2)', alpha=0.7)

plt.title('Comparison of Coefficients: True vs. L1 vs. L2', fontsize=14)
plt.xlabel('Feature Index (sorted by true importance)', fontsize=12)
plt.ylabel('Coefficient Value', fontsize=12)
plt.legend()
plt.xticks(x, [str(i) for i in top_idx], rotation=90)
plt.grid(True, axis='y')
plt.tight_layout()
save_plot("l1_vs_l2_sparsity.png")

# Create regularization path visualization
plt.figure(figsize=(12, 8))

alphas_path = np.logspace(-5, 2, 200)
coefs_lasso = []
coefs_ridge = []

for alpha in alphas_path:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    coefs_lasso.append(lasso.coef_.copy())
    
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    coefs_ridge.append(ridge.coef_.copy())

coefs_lasso = np.array(coefs_lasso)
coefs_ridge = np.array(coefs_ridge)

# Select only a few important features to visualize
colors = plt.cm.rainbow(np.linspace(0, 1, 10))
plt.figure(figsize=(12, 10))

plt.subplot(2, 1, 1)
for i, c in zip(range(10), colors):
    plt.semilogx(alphas_path, coefs_lasso[:, top_idx[i]], color=c, 
                 label=f'Feature {top_idx[i]}')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.xlabel('Regularization parameter (alpha)', fontsize=12)
plt.ylabel('Coefficient value', fontsize=12)
plt.title('Lasso (L1) Regularization Path', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True)

plt.subplot(2, 1, 2)
for i, c in zip(range(10), colors):
    plt.semilogx(alphas_path, coefs_ridge[:, top_idx[i]], color=c, 
                 label=f'Feature {top_idx[i]}')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.xlabel('Regularization parameter (alpha)', fontsize=12)
plt.ylabel('Coefficient value', fontsize=12)
plt.title('Ridge (L2) Regularization Path', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True)

plt.tight_layout()
save_plot("l1_vs_l2_path.png")

# Visualize geometric interpretation
def plot_contours(ax, f, xx, yy, **params):
    Z = np.zeros_like(xx)
    for i in range(len(xx)):
        for j in range(len(xx[0])):
            Z[i, j] = f(xx[i, j], yy[i, j])
    ax.contour(xx, yy, Z, **params)

def l1_norm(w1, w2):
    return np.abs(w1) + np.abs(w2)

def l2_norm(w1, w2):
    return np.sqrt(w1**2 + w2**2)

def objective_func(w1, w2, norm_func, alpha, x0=0.5, y0=0.5):
    # Simplified objective function: (w1 - x0)² + (w2 - y0)² + alpha * norm_func(w1, w2)
    return (w1 - x0)**2 + (w2 - y0)**2 + alpha * norm_func(w1, w2)

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Create a grid of values
xx, yy = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))

# L1 norm contours
ax = axes[0, 0]
levels = [0.5, 1.0, 1.5, 2.0, 2.5]
plot_contours(ax, l1_norm, xx, yy, levels=levels, cmap=plt.cm.coolwarm)
ax.set_title('L1 Norm Contours', fontsize=14)
ax.set_xlabel('w₁', fontsize=12)
ax.set_ylabel('w₂', fontsize=12)
ax.grid(True)

# L2 norm contours
ax = axes[0, 1]
plot_contours(ax, l2_norm, xx, yy, levels=levels, cmap=plt.cm.coolwarm)
ax.set_title('L2 Norm Contours', fontsize=14)
ax.set_xlabel('w₁', fontsize=12)
ax.set_ylabel('w₂', fontsize=12)
ax.grid(True)

# L1 regularized objective
ax = axes[1, 0]
alpha = 1.0
l1_obj = lambda w1, w2: objective_func(w1, w2, l1_norm, alpha)
levels = np.logspace(0, 1, 8) 
plot_contours(ax, l1_obj, xx, yy, levels=levels, cmap=plt.cm.coolwarm)
ax.scatter(0.5, 0.5, color='r', marker='*', s=100, label='Optimal w/o reg')
ax.set_title('L1 Regularized Objective', fontsize=14)
ax.set_xlabel('w₁', fontsize=12)
ax.set_ylabel('w₂', fontsize=12)
ax.grid(True)
ax.legend()

# L2 regularized objective
ax = axes[1, 1]
l2_obj = lambda w1, w2: objective_func(w1, w2, l2_norm, alpha)
plot_contours(ax, l2_obj, xx, yy, levels=levels, cmap=plt.cm.coolwarm)
ax.scatter(0.5, 0.5, color='r', marker='*', s=100, label='Optimal w/o reg')
ax.set_title('L2 Regularized Objective', fontsize=14)
ax.set_xlabel('w₁', fontsize=12)
ax.set_ylabel('w₂', fontsize=12)
ax.grid(True)
ax.legend()

plt.tight_layout()
save_plot("l1_l2_geometric.png")

# Part 5: Cross-validation for regularization parameter selection
print("\n5. Cross-Validation for Selecting Regularization Parameter")
print("-------------------------------------------------------")

# Generate dataset
X, y = make_regression(n_samples=100, n_features=10, n_informative=5,
                      random_state=42, noise=30.0)

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define alpha range
alphas = np.logspace(-3, 3, 20)

# Function for cross-validation
def perform_cv(model_class, alphas, X, y, cv=5, **model_params):
    mean_train_scores = []
    mean_test_scores = []
    std_test_scores = []
    
    for alpha in alphas:
        model = model_class(alpha=alpha, **model_params)
        
        # Cross-validation
        cv_obj = KFold(n_splits=cv, shuffle=True, random_state=42)
        train_scores = []
        test_scores = []
        
        for train_idx, test_idx in cv_obj.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_score = mean_squared_error(y_train, y_train_pred)
            test_score = mean_squared_error(y_test, y_test_pred)
            
            train_scores.append(train_score)
            test_scores.append(test_score)
        
        mean_train_scores.append(np.mean(train_scores))
        mean_test_scores.append(np.mean(test_scores))
        std_test_scores.append(np.std(test_scores))
    
    return mean_train_scores, mean_test_scores, std_test_scores

# Perform cross-validation for Ridge
ridge_train_scores, ridge_test_scores, ridge_test_std = perform_cv(
    Ridge, alphas, X_scaled, y)

# Perform cross-validation for Lasso
lasso_train_scores, lasso_test_scores, lasso_test_std = perform_cv(
    Lasso, alphas, X_scaled, y, max_iter=10000)

# Find best alpha
best_ridge_idx = np.argmin(ridge_test_scores)
best_lasso_idx = np.argmin(lasso_test_scores)

print(f"Best Ridge alpha: {alphas[best_ridge_idx]:.4f}, CV MSE: {ridge_test_scores[best_ridge_idx]:.4f}")
print(f"Best Lasso alpha: {alphas[best_lasso_idx]:.4f}, CV MSE: {lasso_test_scores[best_lasso_idx]:.4f}")

# Plot cross-validation results
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.semilogx(alphas, ridge_train_scores, 'b--', label='Training error')
plt.semilogx(alphas, ridge_test_scores, 'r-', label='Cross-validation error')
plt.fill_between(alphas, 
                 np.array(ridge_test_scores) - np.array(ridge_test_std),
                 np.array(ridge_test_scores) + np.array(ridge_test_std),
                 alpha=0.2, color='r')
plt.axvline(x=alphas[best_ridge_idx], color='green', linestyle='--',
            label=f'Best alpha: {alphas[best_ridge_idx]:.4f}')
plt.xlabel('Regularization parameter (alpha)', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.title('Ridge Regression: Cross-validation for Regularization Parameter', fontsize=14)
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.semilogx(alphas, lasso_train_scores, 'b--', label='Training error')
plt.semilogx(alphas, lasso_test_scores, 'r-', label='Cross-validation error')
plt.fill_between(alphas, 
                 np.array(lasso_test_scores) - np.array(lasso_test_std),
                 np.array(lasso_test_scores) + np.array(lasso_test_std),
                 alpha=0.2, color='r')
plt.axvline(x=alphas[best_lasso_idx], color='green', linestyle='--',
            label=f'Best alpha: {alphas[best_lasso_idx]:.4f}')
plt.xlabel('Regularization parameter (alpha)', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.title('Lasso Regression: Cross-validation for Regularization Parameter', fontsize=14)
plt.legend()
plt.grid(True)

plt.tight_layout()
save_plot("cross_validation.png")

print(f"\nAll visualizations saved to: {save_dir}") 