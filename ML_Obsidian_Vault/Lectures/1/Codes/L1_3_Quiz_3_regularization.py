import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split # Needed for fit plot example

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L1_3_Quiz_3")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

def plot_coef_comparison(coef_ridge, coef_lasso, feature_names, alpha_val, filename):
    """Compare Ridge and Lasso coefficients for a specific alpha."""
    n_features = len(coef_ridge)
    indices = np.arange(n_features)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    
    # Ridge plot
    axes[0].stem(indices, coef_ridge, linefmt='b-', markerfmt='bo', basefmt=' ')
    axes[0].set_title(f'Ridge Coefficients (α={alpha_val:.2f})')
    axes[0].set_ylabel('Coefficient Value')
    axes[0].grid(True)
    axes[0].set_xticks(indices)
    axes[0].set_xticklabels(feature_names, rotation=90, fontsize=8)
    axes[0].set_xlabel('Feature')

    # Lasso plot
    axes[1].stem(indices, coef_lasso, linefmt='r-', markerfmt='ro', basefmt=' ')
    axes[1].set_title(f'Lasso Coefficients (α={alpha_val:.2f})')
    axes[1].grid(True)
    axes[1].set_xticks(indices)
    axes[1].set_xticklabels(feature_names, rotation=90, fontsize=8)
    axes[1].set_xlabel('Feature')
    
    plt.suptitle(f'Ridge (L2) vs. Lasso (L1) Coefficient Comparison @ α={alpha_val:.2f}', fontsize=14)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout
    
    file_path = os.path.join(save_dir, filename)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Coefficient comparison plot saved to: {file_path}")

def plot_regularized_fit(degree=15, alpha=0.1, filename="regularized_fit.png"):
    """Compare polynomial fit with and without Ridge regularization."""
    # Generate some noisy data
    np.random.seed(0)
    n_samples = 30
    X = np.sort(np.random.rand(n_samples))
    y_true_func = np.cos(1.5 * np.pi * X)
    y = y_true_func + np.random.randn(n_samples) * 0.2
    X = X.reshape(-1, 1)
    X_plot = np.linspace(0, 1, 100).reshape(-1, 1)

    # Model without regularization
    model_unreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model_unreg.fit(X, y)
    y_plot_unreg = model_unreg.predict(X_plot)

    # Model with Ridge regularization
    # Note: PolynomialFeatures can create large values, scaling helps Ridge
    model_reg = make_pipeline(PolynomialFeatures(degree), StandardScaler(), Ridge(alpha=alpha))
    model_reg.fit(X, y)
    y_plot_reg = model_reg.predict(X_plot)

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, edgecolor='k', facecolor='none', s=50, label='Data Points')
    plt.plot(X_plot, np.cos(1.5 * np.pi * X_plot), color='black', linestyle='--', label='True Function')
    plt.plot(X_plot, y_plot_unreg, color='red', linewidth=2, alpha=0.7, label=f'Degree {degree} Fit (No Regularization)')
    plt.plot(X_plot, y_plot_reg, color='blue', linewidth=2, label=f'Degree {degree} Fit (Ridge α={alpha})')
    
    plt.title('Effect of L2 Regularization on High-Degree Polynomial Fit', fontsize=14)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.ylim(-1.5, 1.5)
    plt.grid(True)
    plt.tight_layout()

    file_path = os.path.join(save_dir, filename)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Regularized fit comparison plot saved to: {file_path}")

# --- Step 1: Purpose of Regularization ---
print_step_header(1, "Purpose of Regularization")

print("Regularization is a set of techniques used to prevent overfitting in machine learning models, particularly those with high variance (complex models).")
print("How it works:")
print("- It adds a penalty term to the model's loss function.")
print("- This penalty discourages the model from learning overly complex patterns or assigning excessively large weights/coefficients to features.")
print("- By constraining the model complexity, regularization helps improve the model\'s ability to generalize to unseen data.")
print("Common Loss Function with Regularization:")
print("  Loss = Empirical Loss (e.g., MSE) + λ * Regularization Term")
print("- λ (lambda) is the regularization parameter: controls the strength of the penalty.")
print("  - λ = 0: No regularization.")
print("  - λ > 0: Regularization applied. Higher λ means stronger penalty, simpler model.")

# --- Step 2: L1 (Lasso) vs L2 (Ridge) Regularization ---
print_step_header(2, "Comparing L1 (Lasso) and L2 (Ridge) Regularization")

print("L2 Regularization (Ridge Regression):")
print("- Penalty Term: Sum of the squares of the coefficients (L2 norm): λ * Σ(wᵢ²)")
print("- Effect: Shrinks coefficients towards zero, but rarely makes them exactly zero.")
print("- Use Case: Good when most features are expected to be relevant; reduces coefficient magnitude smoothly.")
print("- Geometric View: Constrains coefficients to lie within a hypersphere (circle in 2D).")
print()
print("L1 Regularization (Lasso Regression):")
print("- Penalty Term: Sum of the absolute values of the coefficients (L1 norm): λ * Σ|wᵢ|")
print("- Effect: Shrinks coefficients towards zero and can force some coefficients to become exactly zero.")
print("- Use Case: Useful for feature selection, especially when many features might be irrelevant; produces sparse models.")
print("- Geometric View: Constrains coefficients to lie within a hyperdiamond (diamond in 2D).")

# Visualize the L1 and L2 constraints geometrically (conceptual)
plt.figure(figsize=(6, 6))
t = np.linspace(-1, 1, 100)
plt.plot(t, np.sqrt(1 - t**2), color='blue')
plt.plot(t, -np.sqrt(1 - t**2), color='blue', label='L2 Constraint (Ridge, w₁² + w₂² ≤ C)')
plt.plot(t, 1 - np.abs(t), color='red')
plt.plot(t, -(1 - np.abs(t)), color='red', label='L1 Constraint (Lasso, |w₁| + |w₂| ≤ C)')

# Simulate an unregularized solution and level curves of loss function (conceptual)
w1, w2 = np.meshgrid(np.linspace(-1.5, 1.5, 100), np.linspace(-1.5, 1.5, 100))
loss = (w1 - 0.8)**2 + (w2 - 0.7)**2 # Example loss centered at (0.8, 0.7)
contour_plot = plt.contour(w1, w2, loss, levels=[0.1, 0.5, 1.0], colors='gray', linestyles='dashed')
# contour_plot.collections[0].set_label('Loss Contours') # Add label properly
plt.scatter([0.8], [0.7], color='black', label='Unregularized Minimum')

plt.title('Geometric Interpretation of L1 and L2 Regularization', fontsize=14)
plt.xlabel('Coefficient w₁', fontsize=12)
plt.ylabel('Coefficient w₂', fontsize=12)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend(fontsize=9)
plt.grid(True)
plt.axis('equal')
plt.tight_layout()

file_path = os.path.join(save_dir, "l1_l2_geometry.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Geometric interpretation plot saved to: {file_path}")
print("- The intersection of the loss contours and the constraint region determines the regularized solution.")
print("- L1's sharp corners make intersections along the axes (zero coefficient) more likely.")

# --- Step 3: Effect of Regularization Parameter (λ) ---
print_step_header(3, "Effect of Regularization Parameter (λ)")

# Generate synthetic data with some irrelevant features
np.random.seed(1)
n_samples, n_features_reg = 50, 20 # Reduced features for coefficient plot readability
feature_names = [f'Feat_{i+1}' for i in range(n_features_reg)]
X = np.random.randn(n_samples, n_features_reg)

# True coefficients - only first 5 are non-zero
coef_true = np.zeros(n_features_reg)
coef_true[:5] = np.random.randn(5) * 5 
y = X @ coef_true + np.random.randn(n_samples) * 2 # Add noise

# Standardize features - important for regularization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply Ridge (L2) and Lasso (L1) with varying alpha (lambda)
alphas = np.logspace(-3, 1.5, 100) # Adjusted alpha range
coefs_ridge = []
coefs_lasso = []

for a in alphas:
    ridge = Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X_scaled, y)
    coefs_ridge.append(ridge.coef_)
    
    lasso = Lasso(alpha=a, fit_intercept=False, max_iter=10000, tol=1e-3) # Adjust tolerance
    lasso.fit(X_scaled, y)
    coefs_lasso.append(lasso.coef_)

coefs_ridge = np.array(coefs_ridge)
coefs_lasso = np.array(coefs_lasso)

# Plot coefficient paths
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(alphas, coefs_ridge)
plt.xscale('log')
plt.xlabel('λ (alpha)')
plt.ylabel('Coefficients')
plt.title('Ridge (L2) Coefficients vs. λ')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(alphas, coefs_lasso)
plt.xscale('log')
plt.xlabel('λ (alpha)')
plt.ylabel('Coefficients')
plt.title('Lasso (L1) Coefficients vs. λ')
plt.grid(True)

plt.suptitle('Regularization Path: Coefficient Magnitudes vs. λ', fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle

file_path = os.path.join(save_dir, "regularization_paths.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Coefficient path plot saved to: {file_path}")

print("Observations from the paths plot:")
print("- As λ increases:")
print("  - Model Complexity: Decreases (coefficients are shrunk towards zero).")
print("  - Coefficient Magnitudes: Decrease for both Ridge and Lasso.")
print("  - Sparsity (Lasso): More coefficients become exactly zero.")
print("  - Performance: Typically, bias increases, and variance decreases.")
print("    - Very low λ: Risk of overfitting (high variance).")
print("    - Very high λ: Risk of underfitting (high bias).")
print("    - Optimal λ balances bias and variance, often found using cross-validation.")

# Add plot comparing coefficients at a specific alpha
alpha_compare_idx = 60 # Choose an index for a moderate alpha from the logspace
alpha_compare_val = alphas[alpha_compare_idx]
plot_coef_comparison(coefs_ridge[alpha_compare_idx], 
                     coefs_lasso[alpha_compare_idx], 
                     feature_names,
                     alpha_compare_val, 
                     "ridge_lasso_coef_comparison.png")
print("- Coefficient comparison plot shows Ridge shrinks all coeffs, Lasso sets many (esp. irrelevant ones) to zero.")

# Add plot showing effect on fit
plot_regularized_fit(degree=15, alpha=0.01, filename="regularized_fit_example.png") # Use a small alpha
print("- Fit plot shows how Ridge prevents extreme overfitting of high-degree polynomial.")

# --- Step 4: Recommendation for Irrelevant Features ---
print_step_header(4, "Recommendation for Scenario with Irrelevant Features")

print("Problem: Linear regression model with 100 features, many suspected to be irrelevant.")
print("Recommendation: **L1 (Lasso) Regularization**")
print("Reasoning:")
print("- Lasso\'s L1 penalty has the property of shrinking some coefficients exactly to zero.")
print("- This acts as an automatic form of feature selection, effectively removing the irrelevant features (those with zero coefficients) from the model.")
print("- Ridge (L2) shrinks coefficients but rarely sets them to exactly zero, so it keeps all features in the model, just with smaller weights.")
print("- In a high-dimensional setting with potentially many irrelevant features, Lasso\'s ability to produce sparse models is highly advantageous.")

# Show number of non-zero coefficients for Lasso
plt.figure(figsize=(8, 5))
n_nonzero_coefs = np.sum(np.abs(coefs_lasso) > 1e-6, axis=1) # Count non-zeros (adjust threshold if needed)
plt.plot(alphas, n_nonzero_coefs)
plt.xscale('log')
plt.xlabel('λ (alpha)')
plt.ylabel('Number of Non-Zero Coefficients')
plt.title('Lasso: Sparsity vs. λ')
plt.grid(True)
plt.axhline(5, color='red', linestyle='--', label='True Non-Zero Features (5)') # Mark true number (updated for reduced features)
plt.legend()
plt.tight_layout()

file_path = os.path.join(save_dir, "lasso_sparsity.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Lasso sparsity plot saved to: {file_path}")
print("- The plot shows how increasing λ in Lasso reduces the number of active features.")

print("\nScript finished. Plots saved in:", save_dir) 