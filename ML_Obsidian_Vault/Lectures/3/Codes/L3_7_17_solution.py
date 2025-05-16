import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import os
from scipy import stats
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_7_Quiz_17")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Generate synthetic data with multicollinearity
np.random.seed(42)
X, y, coef = make_regression(n_samples=200, n_features=20, n_informative=10, 
                             noise=30, coef=True, random_state=42)

# Add multicollinearity
X[:, 5] = X[:, 1] * 0.9 + np.random.normal(0, 0.1, size=X.shape[0])
X[:, 15] = X[:, 10] * 0.8 + X[:, 8] * 0.3 + np.random.normal(0, 0.1, size=X.shape[0])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit models with different regularization types
# For question 1: Different regularization methods
def fit_models():
    ridge = Ridge(alpha=1.0)
    lasso = Lasso(alpha=0.1)
    elasticnet = ElasticNet(alpha=0.1, l1_ratio=0.5)
    
    ridge.fit(X_train_scaled, y_train)
    lasso.fit(X_train_scaled, y_train)
    elasticnet.fit(X_train_scaled, y_train)
    
    return {
        'Ridge': ridge,
        'Lasso': lasso,
        'ElasticNet': elasticnet
    }

models = fit_models()

# Visualization 1: Compare coefficients of different regularization methods
plt.figure(figsize=(14, 7))
features = range(X.shape[1])
plt.plot(features, models['Ridge'].coef_, 'o-', label='Ridge', alpha=0.7)
plt.plot(features, models['Lasso'].coef_, 's-', label='Lasso', alpha=0.7)
plt.plot(features, models['ElasticNet'].coef_, '^-', label='ElasticNet', alpha=0.7)

plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.title('Coefficient Values for Different Regularization Methods', fontsize=14)
plt.xlabel('Feature Index', fontsize=12)
plt.ylabel('Coefficient Value', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "regularization_comparison.png"), dpi=300, bbox_inches='tight')
plt.close()

# For question 2: Bias-variance tradeoff with increasing regularization
alphas = np.logspace(-3, 3, 30)
ridge_train_errors = []
ridge_test_errors = []
ridge_coefs = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    
    # Store train and test errors
    y_train_pred = ridge.predict(X_train_scaled)
    y_test_pred = ridge.predict(X_test_scaled)
    
    train_error = mean_squared_error(y_train, y_train_pred)
    test_error = mean_squared_error(y_test, y_test_pred)
    
    ridge_train_errors.append(train_error)
    ridge_test_errors.append(test_error)
    ridge_coefs.append(ridge.coef_.copy())

# Visualization 2: Bias-variance tradeoff
plt.figure(figsize=(12, 7))
plt.semilogx(alphas, ridge_train_errors, 'b-', label='Training Error (↑ Bias)')
plt.semilogx(alphas, ridge_test_errors, 'r-', label='Test Error (↓ Variance)')
plt.axvline(x=alphas[np.argmin(ridge_test_errors)], color='k', linestyle='--', 
            label=f'Optimal Alpha = {alphas[np.argmin(ridge_test_errors)]:.4f}')

plt.title('Bias-Variance Tradeoff in Ridge Regression', fontsize=14)
plt.xlabel('Regularization Parameter (Alpha)', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "bias_variance_tradeoff.png"), dpi=300, bbox_inches='tight')
plt.close()

# For question 3: Zero coefficients in Lasso vs Ridge
alphas_comparison = np.logspace(-2, 2, 10)
lasso_num_zeros = []
ridge_num_zeros = []

for alpha in alphas_comparison:
    ridge = Ridge(alpha=alpha)
    lasso = Lasso(alpha=alpha)
    
    ridge.fit(X_train_scaled, y_train)
    lasso.fit(X_train_scaled, y_train)
    
    # Count zero coefficients (using a small threshold for numerical stability)
    ridge_zero = np.sum(np.abs(ridge.coef_) < 1e-10)
    lasso_zero = np.sum(np.abs(lasso.coef_) < 1e-10)
    
    ridge_num_zeros.append(ridge_zero)
    lasso_num_zeros.append(lasso_zero)

# Visualization 3: Number of zero coefficients
plt.figure(figsize=(12, 7))
plt.semilogx(alphas_comparison, ridge_num_zeros, 'bo-', label='Ridge')
plt.semilogx(alphas_comparison, lasso_num_zeros, 'ro-', label='Lasso')
plt.title('Number of Zero Coefficients vs. Regularization Strength', fontsize=14)
plt.xlabel('Regularization Parameter (Alpha)', fontsize=12)
plt.ylabel('Number of Zero Coefficients', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "zero_coefficients.png"), dpi=300, bbox_inches='tight')
plt.close()

# For question 4: Bayesian interpretation with different priors
def plot_prior_posterior(prior_name, prior_dist, alpha, save_path):
    """Plot prior, likelihood, and posterior distributions for a parameter"""
    plt.figure(figsize=(12, 7))
    
    # Parameter values to evaluate
    beta_range = np.linspace(-5, 5, 1000)
    
    # Prior distribution
    prior = prior_dist(beta_range)
    
    # Simplified likelihood (normal)
    likelihood_mean = 2.0
    likelihood_std = 1.0
    likelihood = stats.norm.pdf(beta_range, loc=likelihood_mean, scale=likelihood_std)
    
    # Posterior calculation (unnormalized)
    posterior = likelihood * prior
    posterior = posterior / np.trapz(posterior, beta_range)  # Normalize
    
    # Plot
    plt.plot(beta_range, prior, 'b-', label='Prior', linewidth=2)
    plt.plot(beta_range, likelihood, 'r-', label='Likelihood', linewidth=2)
    plt.plot(beta_range, posterior, 'g-', label='Posterior', linewidth=2)
    
    if prior_name == "Gaussian (Ridge)":
        map_estimate = (likelihood_mean / likelihood_std**2) / (1/likelihood_std**2 + 1/alpha)
    else:  # For demonstration, use mode of posterior
        map_estimate = beta_range[np.argmax(posterior)]
    
    plt.axvline(x=map_estimate, color='k', linestyle='--', 
                label=f'MAP Estimate: {map_estimate:.2f}')
    
    plt.title(f'Bayesian Interpretation: {prior_name} Prior', fontsize=14)
    plt.xlabel('Parameter Value (β)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Define different priors
alpha_ridge = 1.0  # Ridge regularization strength
alpha_lasso = 1.0  # Lasso regularization strength

# Gaussian prior (Ridge)
gaussian_prior = lambda beta: stats.norm.pdf(beta, loc=0, scale=np.sqrt(alpha_ridge))
plot_prior_posterior("Gaussian (Ridge)", gaussian_prior, alpha_ridge, 
                     os.path.join(save_dir, "gaussian_prior.png"))

# Laplace prior (Lasso)
laplace_prior = lambda beta: stats.laplace.pdf(beta, loc=0, scale=alpha_lasso/2)
plot_prior_posterior("Laplace (Lasso)", laplace_prior, alpha_lasso, 
                     os.path.join(save_dir, "laplace_prior.png"))

# Uniform prior (No regularization)
uniform_prior = lambda beta: np.ones_like(beta) * 0.1  # Constant value for uniform
plot_prior_posterior("Uniform (No regularization)", uniform_prior, 1.0, 
                     os.path.join(save_dir, "uniform_prior.png"))

# Visualization 5: Coefficient paths for Lasso
alphas_path = np.logspace(-3, 1, 100)
lasso_coefs = []

for alpha in alphas_path:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train_scaled, y_train)
    lasso_coefs.append(lasso.coef_)

lasso_coefs = np.array(lasso_coefs)

plt.figure(figsize=(12, 7))
for i in range(X.shape[1]):
    plt.semilogx(alphas_path, lasso_coefs[:, i], '-', alpha=0.7)

plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.xlabel('Regularization Parameter (Alpha)', fontsize=12)
plt.ylabel('Coefficient Value', fontsize=12)
plt.title('Lasso Regularization Path', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "lasso_path.png"), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 6: L1 vs L2 norm constraint visualization for 2D case
def plot_constraint_regions():
    plt.figure(figsize=(10, 10))
    
    # Create data
    x = np.linspace(-2, 2, 1000)
    y = np.linspace(-2, 2, 1000)
    X, Y = np.meshgrid(x, y)
    
    # Create contours for L1 and L2 norms
    L1_norm = np.abs(X) + np.abs(Y)
    L2_norm = np.sqrt(X**2 + Y**2)
    
    # Plot constraint regions
    plt.contour(X, Y, L1_norm, levels=[1], colors='r', linewidths=2)
    plt.contour(X, Y, L2_norm, levels=[1], colors='b', linewidths=2)
    
    # Make it look like a proper coordinate system
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Add some loss function contours (circular for simplicity)
    for r in np.arange(0.2, 2, 0.4):
        plt.contour(X, Y, X**2 + Y**2, levels=[r**2], colors='green', alpha=0.5, linestyles='--')
    
    # Add intersection points
    plt.plot([0, 1, 0, -1, 0], [1, 0, -1, 0, 1], 'ro', markersize=8)  # L1 vertices
    plt.plot([0, 1/np.sqrt(2), 0, -1/np.sqrt(2), 0], 
             [1, 1/np.sqrt(2), -1, -1/np.sqrt(2), 1], 'bo', markersize=8, alpha=0.6)  # L2 points
    
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.title('L1 vs L2 Regularization Constraint Regions', fontsize=14)
    plt.xlabel('β₁', fontsize=12)
    plt.ylabel('β₂', fontsize=12)
    plt.legend(['L1 Norm = 1 (Lasso)', 'L2 Norm = 1 (Ridge)'], fontsize=12)
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "regularization_geometry.png"), dpi=300, bbox_inches='tight')
    plt.close()

plot_constraint_regions()

print(f"\nVisualizations saved to: {save_dir}")

# Print key results and insights for each question
print("\nQuestion 1: Common Regularization Methods")
print("=============================================")
print("- L1 Regularization (Lasso): Adds penalty term λ∑|βj|")
print("- L2 Regularization (Ridge): Adds penalty term λ∑βj²")
print("- Elastic Net: Combines L1 and L2 penalties")
print("- L0 Regularization: Directly penalizes number of non-zero coefficients")
print("\nObservation: L0 is rarely used directly due to computational complexity (non-convex).")

print("\nQuestion 2: Effect of Increasing Regularization Parameter in Ridge")
print("=====================================================================")
print(f"Optimal alpha value: {alphas[np.argmin(ridge_test_errors)]:.4f}")
print(f"Minimum test error: {min(ridge_test_errors):.2f}")
print("\nAs alpha increases:")
print("- Training error increases (higher bias)")
print("- Test error initially decreases (lower variance), then increases (too much bias)")
print("- The bias-variance tradeoff is clearly visible in the plot")

print("\nQuestion 3: Zero Coefficients in Regularization Methods")
print("========================================================")
for i, alpha in enumerate(alphas_comparison):
    print(f"Alpha = {alpha:.4f}: Ridge zeros = {ridge_num_zeros[i]}, Lasso zeros = {lasso_num_zeros[i]}")
print("\nObservation: Lasso produces exactly zero coefficients even at moderate regularization strengths,")
print("whereas Ridge tends to shrink coefficients close to, but not exactly, zero.")

print("\nQuestion 4: Bayesian Interpretation of Ridge Regression")
print("========================================================")
print("Ridge regression corresponds to Maximum A Posteriori (MAP) estimation with:")
print("- Gaussian prior on coefficients: β ~ N(0, σ²/λ)")
print("- Maximizing posterior p(β|X,y) ∝ p(y|X,β) × p(β)")
print("- This yields the same estimate as ridge regression")
print("\nLasso corresponds to a Laplace prior, while uniform prior gives ordinary least squares.")
print("Ridge's Gaussian prior is bell-shaped and encourages small but non-zero values.") 