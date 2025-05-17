import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_7_Quiz_22")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Print explanations and formulas for Ridge Regression
print("\nRidge Regression - Mathematical Properties")
print("==================================================")
print("Standard Linear Regression: min ||y - Xw||²")
print("Ridge Regression: min ||y - Xw||² + λ||w||²")
print("\nClosed-form solution:")
print("w_ridge = (X^T X + λI)^(-1) X^T y")

print("\nKey Properties:")
print("1. Regularization: Penalizes large coefficients")
print("2. Shrinkage: Coefficient values are shrunk toward zero")
print("3. Bias-Variance trade-off: Increases bias, reduces variance")
print("4. Multicollinearity: Stabilizes solutions with highly correlated features")

# -----------------------
# Mathematical Derivation
# -----------------------
print("\n\nMathematical Derivation of the Ridge Solution:")
print("Starting with the ridge cost function: J(w) = ||y - Xw||² + λ||w||²")
print("Taking the gradient with respect to w and setting to zero:")
print("∇J(w) = -2X^T(y - Xw) + 2λw = 0")
print("X^T(y - Xw) = λw")
print("X^Ty - X^TXw = λw")
print("X^Ty = X^TXw + λw")
print("X^Ty = (X^TX + λI)w")
print("w_ridge = (X^TX + λI)^(-1)X^Ty")

# -----------------------
# Invertibility Proof
# -----------------------
print("\n\nInvertibility of (X^TX + λI):")
print("For any matrix X, X^TX is positive semi-definite")
print("All eigenvalues of X^TX are non-negative")
print("For λ > 0, adding λI to X^TX increases all eigenvalues by λ")
print("Therefore, all eigenvalues of (X^TX + λI) are at least λ > 0")
print("Since all eigenvalues are positive, (X^TX + λI) is positive definite")
print("Positive definite matrices are always invertible")

# ----------------------------
# Multicollinearity Explanation
# ----------------------------
print("\n\nMulticollinearity Effect:")
print("With multicollinearity, X^TX has some eigenvalues close to zero")
print("Ridge adds λ to all eigenvalues: (X^TX + λI) has eigenvalues (λ_i + λ)")
print("This stabilizes directions with near-zero eigenvalues")
print("Eigenvectors with small eigenvalues are penalized more strongly")
print("Makes the solution less sensitive to small changes in the data")

# ----------------------------
# Limiting behavior as λ → ∞
# ----------------------------
print("\n\nBehavior as λ → ∞:")
print("w_ridge = (X^TX + λI)^(-1)X^Ty")
print("As λ → ∞, (X^TX + λI) ≈ λI")
print("Therefore, (X^TX + λI)^(-1) ≈ (1/λ)I")
print("w_ridge ≈ (1/λ)X^Ty → 0 as λ → ∞")
print("So all ridge coefficients approach zero as λ → ∞")

# ----------------------------
# Eigenvalue perspective
# ----------------------------
print("\n\nEigenvalue Perspective:")
print("In the eigenbasis of X^TX:")
print("Let X^TX = UDU^T where D is diagonal with eigenvalues λ_1, λ_2, ... λ_p")
print("The ridge estimator can be written as:")
print("w_ridge = U(D + λI)^(-1)U^TX^Ty")
print("For each direction (eigenvector), the shrinkage factor is λ_i/(λ_i + λ)")
print("Directions with small eigenvalues (λ_i ≈ 0) are shrunk more: 0/(0+λ) ≈ 0")
print("Directions with large eigenvalues are shrunk less: λ_i/(λ_i+λ) ≈ 1 for λ_i >> λ")

# ----------------------------
# Numerical Examples and Visualizations
# ----------------------------

# Example 1: Generate multicollinear data
np.random.seed(42)
n = 100  # number of samples
X_orig = np.random.normal(0, 1, (n, 2))  # two independent features
X_multicol = np.column_stack([X_orig[:, 0], 0.95 * X_orig[:, 0] + 0.05 * X_orig[:, 1]])  # second feature highly correlated with first
X = np.column_stack([np.ones(n), X_multicol])  # add intercept

# Debug shapes
print(f"\nData shapes: X: {X.shape}, X_multicol: {X_multicol.shape}")

# True coefficients
true_w = np.array([1, 2, -1.5])

# Generate response with some noise
y = X @ true_w + np.random.normal(0, 1, n)
print(f"y shape: {y.shape}")

# Compute OLS and Ridge solutions for different λ values
lambdas = [0, 0.1, 1, 10, 100, 1000]
ridge_solutions = []

X_centered = X.copy()
X_centered[:, 1:] = X_centered[:, 1:] - X_centered[:, 1:].mean(axis=0)
XtX = X_centered.T @ X_centered
Xty = X_centered.T @ y

print("\n\nNumerical Example with Multicollinear Data:")
print(f"Correlation between features: {np.corrcoef(X_multicol.T)[0, 1]:.4f}")
print("\nEigenvalues of X^TX:")
eigvals = np.linalg.eigvals(XtX[1:, 1:])  # excluding intercept
print(eigvals)
print("\nCondition number of X^TX:", np.linalg.cond(XtX[1:, 1:]))

print("\nRidge solutions for different λ values:")
for lambda_val in lambdas:
    # Ridge solution
    ridge_w = np.linalg.inv(XtX + lambda_val * np.eye(3)) @ Xty
    ridge_solutions.append(ridge_w)
    print(f"λ = {lambda_val}: w = {ridge_w}")

# 1. Plot of coefficient paths
plt.figure(figsize=(10, 6))
lambda_range = np.logspace(-2, 3, 100)
coef_paths = []
for lambda_val in lambda_range:
    # Ridge solution
    ridge_w = np.linalg.inv(XtX + lambda_val * np.eye(3)) @ Xty
    coef_paths.append(ridge_w)

coef_paths = np.array(coef_paths)
plt.semilogx(lambda_range, coef_paths[:, 0], '-', linewidth=2, label='Intercept')
plt.semilogx(lambda_range, coef_paths[:, 1], '--', linewidth=2, label='Feature 1')
plt.semilogx(lambda_range, coef_paths[:, 2], '-.', linewidth=2, label='Feature 2')
plt.axhline(y=0, color='k', linestyle=':', alpha=0.3)
plt.xlabel('λ (regularization strength)', fontsize=12)
plt.ylabel('Coefficient value', fontsize=12)
plt.title('Ridge Regression Coefficient Paths', fontsize=14)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, "ridge_coefficient_paths.png"), dpi=300, bbox_inches='tight')
plt.close()

# Skip contour plots and 3D plots for now
print("\nSkipping contour and 3D plots due to shape issues")

# 4. Eigenvalue effects - How ridge affects different eigendirections
# Let's create a SVD decomposition to visualize this
U, s, Vt = np.linalg.svd(X_centered[:, 1:], full_matrices=False)
V = Vt.T

# Print the eigenvalues (squared singular values)
print("\nSingular values of X:", s)
print("Eigenvalues of X^TX:", s**2)

# Compute shrinkage factors for different eigenvalues and lambda values
eigenvalues = s**2
shrinkage_factors = {}
for lambda_val in [0.1, 1, 10, 100]:
    shrinkage_factors[lambda_val] = eigenvalues / (eigenvalues + lambda_val)

print("\nShrinkage factors (eigenvalue / (eigenvalue + λ)):")
for lambda_val, factors in shrinkage_factors.items():
    print(f"λ = {lambda_val}: {factors}")

# Plot the shrinkage factor for different eigenvalues and lambda values
plt.figure(figsize=(10, 6))
lambda_range_eigen = np.logspace(-2, 3, 100)
eigenvalue_range = np.logspace(-2, 2, 10)

for eigenvalue in eigenvalue_range:
    shrinkage = eigenvalue / (eigenvalue + lambda_range_eigen)
    plt.semilogx(lambda_range_eigen, shrinkage, '-', linewidth=1.5, 
                label=f'Eigenvalue={eigenvalue:.2f}')

plt.xlabel('λ (regularization strength)', fontsize=12)
plt.ylabel('Shrinkage factor: λ_i / (λ_i + λ)', fontsize=12)
plt.title('Ridge Shrinkage Factor by Eigenvalue', fontsize=14)
plt.grid(True)
plt.legend(loc='best', fontsize=8)
plt.savefig(os.path.join(save_dir, "ridge_shrinkage_factors.png"), dpi=300, bbox_inches='tight')
plt.close()

# 5. Ridge solution in the eigenbasis
plt.figure(figsize=(12, 6))

# Transform the data to eigenbasis
X_eigen = X_centered[:, 1:] @ V
Xty_eigen = X_eigen.T @ y

# Calculate ridge solutions in eigenbasis
lambda_range_plot = np.logspace(-2, 3, 20)
ridge_sol_eigen = []

for lambda_val in lambda_range_plot:
    # Ridge solution in eigenbasis
    weights_eigen = np.zeros(2)
    for i in range(2):
        weights_eigen[i] = Xty_eigen[i] / (eigenvalues[i] + lambda_val)
    
    # Transform back to original basis
    weights_orig = V @ weights_eigen
    ridge_sol_eigen.append((weights_eigen, weights_orig))

# Plot coefficients in eigenbasis (canonical directions)
plt.subplot(1, 2, 1)
# Change from semilogx to regular plot for the eigenbasis
plt.plot([r[0][0] for r in ridge_sol_eigen], [r[0][1] for r in ridge_sol_eigen], 
        'o-', linewidth=2, markersize=8, label='Ridge path in eigenbasis')
plt.scatter(ridge_sol_eigen[0][0][0], ridge_sol_eigen[0][0][1], s=150, c='red', 
           marker='o', label='OLS solution')
plt.xlabel('Coefficient for 1st eigenvector', fontsize=12)
plt.ylabel('Coefficient for 2nd eigenvector', fontsize=12)
plt.title('Ridge Path in Eigenbasis', fontsize=14)
plt.grid(True)
plt.legend()

# Plot coefficients in original basis
plt.subplot(1, 2, 2)
plt.plot([r[1][0] for r in ridge_sol_eigen], [r[1][1] for r in ridge_sol_eigen], 
        'o-', linewidth=2, markersize=8, label='Ridge path in original basis')
plt.scatter(ridge_sol_eigen[0][1][0], ridge_sol_eigen[0][1][1], s=150, c='red', 
           marker='o', label='OLS solution')
plt.xlabel('Coefficient for Feature 1', fontsize=12)
plt.ylabel('Coefficient for Feature 2', fontsize=12)
plt.title('Ridge Path in Original Basis', fontsize=14)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "ridge_eigenbasis.png"), dpi=300, bbox_inches='tight')
plt.close()

# Add a new plot: Ridge regression prediction vs. OLS prediction
plt.figure(figsize=(10, 6))

# Create a test set with a single feature for visualization
x_test = np.linspace(-3, 3, 100)
X_test = np.column_stack([np.ones_like(x_test), x_test, 0.95*x_test + 0.05*np.random.normal(0, 0.1, 100)])

# Calculate predictions for different lambda values
y_ols = X_test @ ridge_solutions[0]  # lambda = 0
y_pred = {}
for i, lambda_val in enumerate([0.1, 1, 10, 100]):
    y_pred[lambda_val] = X_test @ ridge_solutions[i+1]  # +1 because index 0 is OLS

# Plot the predictions
plt.scatter(X_test[:, 1], y_ols, s=10, color='blue', alpha=0.5, label='OLS')

colors = ['green', 'orange', 'red', 'purple']
lambda_vals = [0.1, 1, 10, 100]

for i, (lambda_val, color) in enumerate(zip(lambda_vals, colors)):
    plt.scatter(X_test[:, 1], y_pred[lambda_val], s=10, color=color, alpha=0.5, 
               label=f'Ridge λ={lambda_val}')

plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Predicted Response', fontsize=12)
plt.title('Ridge Regression Predictions', fontsize=14)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, "ridge_predictions.png"), dpi=300, bbox_inches='tight')
plt.close()

# 6. Variance shrinkage visualization
plt.figure(figsize=(10, 6))

# Generate new simulated datasets
n_datasets = 100
n_samples = 50
beta_ols = np.zeros((n_datasets, 2))
beta_ridge = {}
for lambda_val in [0.1, 1, 10]:
    beta_ridge[lambda_val] = np.zeros((n_datasets, 2))

X_orig_small = np.random.normal(0, 1, (n_samples, 2))
X_multicol_small = np.column_stack([X_orig_small[:, 0], 
                                 0.95 * X_orig_small[:, 0] + 0.05 * X_orig_small[:, 1]])
X_small = np.column_stack([np.ones(n_samples), X_multicol_small])

for i in range(n_datasets):
    # Generate a new response with noise
    y_small = X_small @ true_w + np.random.normal(0, 1, n_samples)
    
    # Center X (not including intercept)
    X_small_centered = X_small.copy()
    X_small_centered[:, 1:] = X_small_centered[:, 1:] - X_small_centered[:, 1:].mean(axis=0)
    
    # Calculate OLS solution
    XtX_small = X_small_centered.T @ X_small_centered
    Xty_small = X_small_centered.T @ y_small
    
    w_ols_small = np.linalg.inv(XtX_small) @ Xty_small
    beta_ols[i] = w_ols_small[1:]  # excluding intercept
    
    # Calculate ridge solutions
    for lambda_val in [0.1, 1, 10]:
        w_ridge_small = np.linalg.inv(XtX_small + lambda_val * np.eye(3)) @ Xty_small
        beta_ridge[lambda_val][i] = w_ridge_small[1:]

# Create an ellipse to represent the distribution of coefficients
def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the covariance of `points`.
    Extra keyword arguments are passed to matplotlib's `patch.Ellipse`.
    """
    from matplotlib.patches import Ellipse
    
    if ax is None:
        ax = plt.gca()
    
    # Calculate covariance matrix
    cov = np.cov(points, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    
    # Calculate center (mean)
    mu = points.mean(axis=0)
    
    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=mu, width=width, height=height, 
                   angle=np.degrees(np.arctan2(*vecs[:,0][::-1])), **kwargs)
    
    ax.add_artist(ellip)
    return ellip

plt.scatter(beta_ols[:, 0], beta_ols[:, 1], marker='.', alpha=0.3, label='OLS solutions')
plot_point_cov(beta_ols, nstd=2, alpha=0.2, color='blue')

colors = ['green', 'orange', 'red']
for i, lambda_val in enumerate([0.1, 1, 10]):
    plt.scatter(beta_ridge[lambda_val][:, 0], beta_ridge[lambda_val][:, 1], 
               marker='.', alpha=0.3, label=f'Ridge λ={lambda_val}')
    plot_point_cov(beta_ridge[lambda_val], nstd=2, alpha=0.2, color=colors[i])

plt.scatter(true_w[1], true_w[2], s=150, c='black', marker='*', label='True coefficients')
plt.xlabel('Coefficient for Feature 1', fontsize=12)
plt.ylabel('Coefficient for Feature 2', fontsize=12)
plt.title('Variance Reduction with Ridge Regression', fontsize=14)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, "ridge_variance_reduction.png"), dpi=300, bbox_inches='tight')
plt.close()

# Better version of contour plot that should work
# 7. Create simpler contour plot
plt.figure(figsize=(10, 8))
resolution = 100
x_range = np.linspace(-1.5, 8, resolution)
y_range = np.linspace(-8, 2, resolution)
xx, yy = np.meshgrid(x_range, y_range)

# Calculate OLS loss surface
Z_ols = np.zeros((resolution, resolution))
Z_ridge = {}
for lambda_val in [0.1, 1, 10, 100]:
    Z_ridge[lambda_val] = np.zeros((resolution, resolution))

# Get intercept from OLS solution (assumed constant)
intercept = coef_paths[0, 0]

for i in range(resolution):
    for j in range(resolution):
        # Create coefficient vector [w1, w2] (excluding intercept)
        w_test = np.array([xx[i, j], yy[i, j]])
        
        # Full coefficient vector [w0, w1, w2]
        w_full = np.array([intercept, xx[i, j], yy[i, j]])
        
        # OLS loss: ||y - Xw||^2
        residuals = y - X @ w_full
        ols_loss = np.sum(residuals**2)
        Z_ols[i, j] = ols_loss
        
        # Ridge losses for different lambda values
        for lambda_val in [0.1, 1, 10, 100]:
            # Ridge loss: ||y - Xw||^2 + λ||w||^2
            ridge_loss = ols_loss + lambda_val * np.sum(w_test**2)
            Z_ridge[lambda_val][i, j] = ridge_loss

# Find minimum values for better contour levels
min_ols = np.min(Z_ols)
max_display = min_ols * 10  # Limit the maximum for better visualization

# Get reasonable contour levels
levels_ols = np.linspace(min_ols, max_display, 15)

# Plot OLS contours
cs_ols = plt.contour(xx, yy, Z_ols, levels=levels_ols, colors='blue', linestyles='-', alpha=0.5)

# Plot Ridge contours with different lambdas
colors = ['green', 'orange', 'red', 'purple']
lambda_vals = [0.1, 1, 10, 100]

for i, (lambda_val, color) in enumerate(zip(lambda_vals, colors)):
    min_ridge = np.min(Z_ridge[lambda_val])
    levels_ridge = np.linspace(min_ridge, min_ridge*10, 15)
    cs_ridge = plt.contour(xx, yy, Z_ridge[lambda_val], levels=levels_ridge, 
                          colors=color, linestyles='-', alpha=0.7)

# Plot the coefficient values
plt.scatter(coef_paths[0, 1], coef_paths[0, 2], s=100, color='blue', marker='o', label='OLS')

for i, (lambda_val, color) in enumerate(zip(lambda_vals, colors)):
    idx = np.where(lambda_range >= lambda_val)[0][0]
    plt.scatter(coef_paths[idx, 1], coef_paths[idx, 2], s=100, color=color, 
               marker='o', label=f'Ridge λ={lambda_val}')

# Add true coefficients
plt.scatter(true_w[1], true_w[2], s=150, color='black', marker='*', label='True coefficients')

plt.xlabel('Coefficient for Feature 1', fontsize=12)
plt.ylabel('Coefficient for Feature 2', fontsize=12)
plt.title('Ridge Regression Contours', fontsize=14)
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig(os.path.join(save_dir, "ridge_contours_fixed.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nVisualizations saved to: {save_dir}") 