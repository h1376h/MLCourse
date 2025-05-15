import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
import scipy.linalg as la
from numpy.linalg import svd, pinv
from sklearn.linear_model import Ridge
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_4_Quiz_19")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'DejaVu Sans',
    'mathtext.fontset': 'dejavusans'
})

# Step 1: Define pseudo-inverse and its formula
def define_pseudo_inverse():
    """Define what the pseudo-inverse is in linear regression and provide its formula."""
    print("Step 1: Defining the Pseudo-Inverse in Linear Regression")
    print("-" * 80)
    print("In linear regression, we try to solve the normal equations:")
    print("(X^T X)β = X^T y")
    print()
    print("If X^T X is invertible, we can find β directly:")
    print("β = (X^T X)^(-1) X^T y")
    print()
    print("The pseudo-inverse (Moore-Penrose inverse) provides a solution when X^T X is not invertible.")
    print("The pseudo-inverse of matrix X is denoted as X^+ and has the following formula:")
    print("X^+ = (X^T X)^+ X^T")
    print()
    print("Using the pseudo-inverse, we can solve for β as:")
    print("β = X^+ y")
    print()
    print("The pseudo-inverse has these important properties:")
    print("1. If X has full rank, X^+ = (X^T X)^(-1) X^T")
    print("2. X^+ minimizes ||Xβ - y||_2")
    print("3. Among all solutions that minimize ||Xβ - y||_2, X^+ y has the minimum norm ||β||_2")
    print()

# Step 2: Explain when pseudo-inverse is necessary
def when_pseudo_inverse_needed():
    """Explain when the pseudo-inverse becomes necessary instead of the normal inverse."""
    print("Step 2: When the Pseudo-Inverse Becomes Necessary in Linear Regression")
    print("-" * 80)
    print("The pseudo-inverse becomes necessary when X^T X is not invertible (singular), which happens when:")
    print()
    print("1. Multicollinearity: When features are perfectly correlated (linearly dependent).")
    print("   This makes X^T X rank-deficient and singular.")
    print()
    print("2. High-dimensional data: When the number of features (p) exceeds the number of samples (n),")
    print("   making X^T X singular because at most min(n,p) columns can be linearly independent.")
    print()
    print("3. Numerical instability: When X^T X is nearly singular (ill-conditioned), leading to")
    print("   numerical problems when computing the inverse.")
    print()
    print("In these cases, the standard inverse (X^T X)^(-1) doesn't exist, but the pseudo-inverse")
    print("still provides a meaningful solution with desirable properties.")
    print()

# Step 3: Specific scenarios leading to non-invertible X^T X
def non_invertible_scenarios():
    """Identify and demonstrate specific scenarios that lead to non-invertible X^T X."""
    print("Step 3: Scenarios Leading to Non-invertible X^T X Matrix")
    print("-" * 80)
    
    # Scenario 1: Perfect multicollinearity
    print("Scenario 1: Perfect Multicollinearity")
    print("-" * 50)
    
    # Create example data with perfectly correlated features
    np.random.seed(42)
    n_samples = 20
    x1 = np.random.normal(0, 1, n_samples)
    x2 = 2 * x1  # x2 is perfectly correlated with x1
    x3 = np.random.normal(0, 1, n_samples)
    
    # Create design matrix
    X_mc = np.column_stack((np.ones(n_samples), x1, x2, x3))
    
    # Calculate X^T X and check its properties
    XTX_mc = X_mc.T @ X_mc
    eigenvalues_mc = np.linalg.eigvals(XTX_mc)
    condition_number_mc = np.linalg.cond(XTX_mc)
    rank_mc = np.linalg.matrix_rank(XTX_mc)
    
    print(f"Design matrix X shape: {X_mc.shape}")
    print(f"Rank of X^T X: {rank_mc} (should be {X_mc.shape[1]} for full rank)")
    print(f"Smallest eigenvalue: {min(eigenvalues_mc):.2e}")
    print(f"Condition number: {condition_number_mc:.2e}")
    print()
    print("Since x2 = 2*x1, the features are linearly dependent, making X^T X singular.")
    print("In this case, rank(X^T X) < p, where p is the number of features.")
    print()
    
    # Visual demonstration of collinearity
    plt.figure(figsize=(10, 6))
    plt.scatter(x1, x2, c='blue', s=50, alpha=0.7, label='Data points')
    
    # Plot the exact relationship
    x1_line = np.linspace(min(x1), max(x1), 100)
    x2_line = 2 * x1_line
    plt.plot(x1_line, x2_line, 'r-', linewidth=2, label='x2 = 2*x1')
    
    plt.title('Perfect Multicollinearity Example', fontsize=14)
    plt.xlabel('Feature x1', fontsize=12)
    plt.ylabel('Feature x2', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=12)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "multicollinearity_example.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Scenario 2: More features than samples (p > n)
    print("\nScenario 2: More Features than Samples (p > n)")
    print("-" * 50)
    
    # Create example with more features than samples
    np.random.seed(42)
    n_samples = 5  # Small number of samples
    n_features = 10  # Larger number of features
    
    # Generate random data
    X_p_gt_n = np.random.normal(0, 1, (n_samples, n_features))
    X_p_gt_n = np.column_stack((np.ones(n_samples), X_p_gt_n))  # Add intercept
    
    # Calculate X^T X and check its properties
    XTX_p_gt_n = X_p_gt_n.T @ X_p_gt_n
    eigenvalues_p_gt_n = np.linalg.eigvals(XTX_p_gt_n)
    condition_number_p_gt_n = np.linalg.cond(XTX_p_gt_n)
    rank_p_gt_n = np.linalg.matrix_rank(XTX_p_gt_n)
    
    print(f"Design matrix X shape: {X_p_gt_n.shape}")
    print(f"Rank of X^T X: {rank_p_gt_n} (should be {X_p_gt_n.shape[1]} for full rank)")
    print(f"Smallest eigenvalue: {min(eigenvalues_p_gt_n):.2e}")
    print(f"Condition number: {condition_number_p_gt_n:.2e}")
    print()
    print("Since p (features + intercept) = 11 > n (samples) = 5, X^T X is singular.")
    print("The maximum possible rank is 5 (the number of samples), but we have 11 columns.")
    print()
    
    # Visual demonstration of p > n
    plt.figure(figsize=(10, 6))
    
    # Create a heatmap of X
    plt.imshow(X_p_gt_n, aspect='auto', cmap='viridis')
    plt.colorbar(label='Value')
    plt.title('High-dimensional Data: More Features than Samples', fontsize=14)
    plt.xlabel('Features (p=11 including intercept)', fontsize=12)
    plt.ylabel('Samples (n=5)', fontsize=12)
    
    # Add grid to show the matrix structure
    plt.grid(False)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "p_gt_n_example.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return example data for further use
    return X_mc, X_p_gt_n

# Step 4: Calculate pseudo-inverse using SVD
def pseudo_inverse_svd(X_mc, X_p_gt_n):
    """Describe and demonstrate how the pseudo-inverse can be calculated using SVD."""
    print("Step 4: Calculating Pseudo-Inverse using Singular Value Decomposition (SVD)")
    print("-" * 80)
    print("The pseudo-inverse can be calculated using Singular Value Decomposition (SVD).")
    print("For a matrix X, the SVD is X = UΣV^T, where:")
    print("- U is an orthogonal matrix of left singular vectors")
    print("- Σ is a diagonal matrix of singular values")
    print("- V is an orthogonal matrix of right singular vectors")
    print()
    print("The pseudo-inverse X^+ is then calculated as:")
    print("X^+ = VΣ^+U^T")
    print()
    print("Where Σ^+ is obtained by taking the reciprocal of each non-zero singular value in Σ")
    print("and leaving the zeros as zeros, then transposing the resulting matrix.")
    print()
    
    # Demonstrate SVD pseudo-inverse calculation for multicollinearity example
    print("Example: Calculating pseudo-inverse for the multicollinearity case")
    print("-" * 50)
    
    # Calculate SVD
    U, s, Vt = np.linalg.svd(X_mc, full_matrices=False)
    
    # Display singular values
    print("Singular values of X:")
    print(s)
    print()
    
    # Calculate pseudo-inverse manually
    s_plus = np.zeros_like(s)
    s_plus[s > 1e-10] = 1/s[s > 1e-10]  # Threshold for considering a singular value as non-zero
    X_plus_manual = Vt.T @ np.diag(s_plus) @ U.T
    
    # Calculate using numpy's pinv
    X_plus_numpy = np.linalg.pinv(X_mc)
    
    # Check how close they are
    pinv_diff = np.linalg.norm(X_plus_manual - X_plus_numpy)
    
    print(f"Norm of difference between manual and numpy's pinv: {pinv_diff:.2e}")
    
    # Create a visual representation of SVD for the multicollinearity example
    plt.figure(figsize=(12, 8))
    
    # Plot singular values
    plt.subplot(2, 1, 1)
    plt.stem(range(1, len(s) + 1), s)
    plt.title('Singular Values of X (Multicollinearity Example)', fontsize=14)
    plt.xlabel('Index', fontsize=12)
    plt.ylabel('Singular Value', fontsize=12)
    plt.grid(True)
    
    # Highlight the nearly zero singular value
    plt.annotate('Near-zero singular value\ndue to collinearity',
                xy=(3, s[2]), xytext=(3.3, s[2] + 2),
                arrowprops=dict(facecolor='red', shrink=0.05),
                fontsize=10)
    
    # Plot singular values in log scale to better see the small values
    plt.subplot(2, 1, 2)
    plt.stem(range(1, len(s) + 1), s)
    plt.yscale('log')
    plt.title('Singular Values (Log Scale)', fontsize=14)
    plt.xlabel('Index', fontsize=12)
    plt.ylabel('Singular Value (log scale)', fontsize=12)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "svd_example.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Demonstrate SVD for p > n case
    print("\nExample: Calculating pseudo-inverse for p > n case")
    print("-" * 50)
    
    # Calculate SVD for p > n case
    U_p, s_p, Vt_p = np.linalg.svd(X_p_gt_n, full_matrices=False)
    
    # Display singular values
    print("Singular values of X (p > n case):")
    print(s_p)
    print()
    print(f"Number of non-zero singular values: {np.sum(s_p > 1e-10)}")
    print(f"This equals the rank of X, which is limited by the number of samples ({X_p_gt_n.shape[0]}).")
    print()
    
    # Visualize the singular values for p > n case
    plt.figure(figsize=(10, 6))
    plt.stem(range(1, len(s_p) + 1), s_p)
    plt.title('Singular Values for p > n Example', fontsize=14)
    plt.xlabel('Index', fontsize=12)
    plt.ylabel('Singular Value', fontsize=12)
    plt.grid(True)
    
    # Highlight the zero singular values
    plt.axhline(y=1e-10, color='r', linestyle='--', label='Effective zero threshold')
    plt.annotate('Zero singular values\n(p > n case)',
                xy=(3, s_p[2]), xytext=(3, s_p[0]/2),
                arrowprops=dict(facecolor='red', shrink=0.05),
                fontsize=10)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "svd_p_gt_n.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Step 5: Relationship between ridge regression and pseudo-inverse
def ridge_pseudo_inverse_relationship():
    """Explain and demonstrate the relationship between ridge regression and pseudo-inverse."""
    print("Step 5: Relationship Between Ridge Regression and the Pseudo-Inverse")
    print("-" * 80)
    print("Ridge regression and the pseudo-inverse are both approaches to handle non-invertible matrices.")
    print()
    print("Ridge regression adds a regularization term λI to X^T X:")
    print("β_ridge = (X^T X + λI)^(-1) X^T y")
    print()
    print("The pseudo-inverse can be related to ridge regression through SVD:")
    print("For X = UΣV^T:")
    print("1. Pseudo-inverse: X^+ = VΣ^+U^T where Σ^+ has 1/σᵢ for non-zero singular values")
    print("2. Ridge solution: (X^T X + λI)^(-1) X^T = V(Σ^T Σ + λI)^(-1)Σ^T U^T")
    print()
    print("This gives diagonal elements 1/σᵢ for SVD vs. σᵢ/(σᵢ² + λ) for ridge regression.")
    print()
    print("As λ → 0, ridge regression approaches the pseudo-inverse solution.")
    print("As λ increases, ridge regression increasingly shrinks coefficients toward zero.")
    print()
    
    # Demonstrate with a toy example
    np.random.seed(42)
    n = 50
    p = 3
    
    # Create multicollinearity
    X = np.random.normal(0, 1, (n, p))
    X[:, 2] = 0.95 * X[:, 0] + 0.05 * X[:, 1] + np.random.normal(0, 0.1, n)  # Near collinearity
    X = np.column_stack((np.ones(n), X))  # Add intercept
    
    # True coefficients
    beta_true = np.array([3, 1.5, -2, 2.5])
    
    # Generate target variable
    y = X @ beta_true + np.random.normal(0, 1, n)
    
    # Calculate condition number to show ill-conditioning
    cond_num = np.linalg.cond(X.T @ X)
    
    print(f"Condition number of X^T X: {cond_num:.2e}")
    print("High condition number indicates near-singularity (ill-conditioning).")
    print()
    
    # SVD of X
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    
    # Compute pseudo-inverse solution
    beta_pinv = np.linalg.pinv(X) @ y
    
    # Compute ridge solutions for different lambdas
    lambdas = [0.01, 0.1, 1, 10, 100]
    ridge_betas = []
    
    for lam in lambdas:
        # Ridge coefficients using formula
        beta_ridge = np.linalg.inv(X.T @ X + lam * np.eye(X.shape[1])) @ X.T @ y
        ridge_betas.append(beta_ridge)
    
    # Calculate SVD-based ridge solutions for visualization
    sigma_sq = s**2
    ridge_sol_by_svd = []
    lambdas_dense = np.logspace(-3, 2, 100)
    
    for lam in lambdas_dense:
        # For each singular value, calculate the ridge ratio
        ridge_factors = s / (sigma_sq + lam)
        ridge_sol_by_svd.append(ridge_factors)
    
    # Create DataFrame for comparison
    beta_df = pd.DataFrame({
        'True': beta_true,
        'Pseudo-inverse': beta_pinv
    })
    
    for i, lam in enumerate(lambdas):
        beta_df[f'Ridge (λ={lam})'] = ridge_betas[i]
    
    print("Comparison of coefficients:")
    print(beta_df)
    print()
    
    # Visualize the relationship between pseudo-inverse and ridge regression
    plt.figure(figsize=(12, 8))
    
    # Plot singular values
    plt.subplot(2, 1, 1)
    plt.stem(range(1, len(s) + 1), s)
    plt.title('Singular Values of X', fontsize=14)
    plt.xlabel('Index', fontsize=12)
    plt.ylabel('Singular Value (σᵢ)', fontsize=12)
    plt.grid(True)
    
    # Plot the "filtering" factors for different lambdas
    plt.subplot(2, 1, 2)
    
    # Pseudo-inverse factors (1/σᵢ for non-zero σᵢ)
    pinv_factors = np.zeros_like(s)
    pinv_factors[s > 1e-10] = 1/s[s > 1e-10]
    plt.scatter(range(1, len(s) + 1), pinv_factors, marker='o', s=100, 
                label='Pseudo-inverse (1/σ_i)', zorder=5)
    
    # Ridge factors for different lambdas
    colors = plt.cm.viridis(np.linspace(0, 1, len(lambdas)))
    for i, lam in enumerate(lambdas):
        ridge_factors = s / (s**2 + lam)
        plt.scatter(range(1, len(s) + 1), ridge_factors, marker='x', s=100, 
                    color=colors[i], label=f'Ridge (λ={lam})', zorder=4-i)
    
    plt.title('Filtering Factors: Pseudo-inverse vs. Ridge Regression', fontsize=14)
    plt.xlabel('Index (i)', fontsize=12)
    plt.ylabel('Factor Applied to Singular Value', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=10)
    
    # Add extra space for the legend
    plt.subplots_adjust(right=0.85, hspace=0.3)
    plt.savefig(os.path.join(save_dir, "ridge_vs_pinv.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a 3D plot showing how ridge regression approaches pseudo-inverse as lambda → 0
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface for ridge factors
    X_grid, Y_grid = np.meshgrid(range(1, len(s) + 1), np.log10(lambdas_dense))
    Z_grid = np.array(ridge_sol_by_svd)
    
    surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis', alpha=0.7)
    
    # Plot pseudo-inverse factors
    x_idx = np.arange(1, len(s) + 1)
    y_idx = np.ones_like(s) * np.log10(lambdas_dense[0])  # Lowest lambda
    ax.scatter(x_idx, y_idx, pinv_factors, color='red', s=100, label='Pseudo-inverse')
    
    ax.set_xlabel('Singular Value Index (i)', fontsize=12)
    ax.set_ylabel('log10(λ)', fontsize=12)
    ax.set_zlabel('Factor Applied to Singular Value', fontsize=12)
    ax.set_title('Ridge Regression Approaches Pseudo-inverse as λ → 0', fontsize=14)
    
    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10)
    cbar.set_label('Factor Value', fontsize=12)
    
    # Add annotation for pseudo-inverse
    ax.text(4, np.log10(lambdas_dense[0]), np.max(pinv_factors)*1.1, 
            "Pseudo-inverse\n(λ → 0)", color='red', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ridge_to_pinv_3d.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Comparison of solutions: error norm vs. coefficient norm
    residual_norms = []
    coef_norms = []
    
    # Compute for pseudo-inverse
    residual_pinv = np.linalg.norm(X @ beta_pinv - y)
    coef_norm_pinv = np.linalg.norm(beta_pinv)
    residual_norms.append(residual_pinv)
    coef_norms.append(coef_norm_pinv)
    
    # Compute for ridge regression with different lambdas
    for beta_ridge in ridge_betas:
        residual_ridge = np.linalg.norm(X @ beta_ridge - y)
        coef_norm_ridge = np.linalg.norm(beta_ridge)
        residual_norms.append(residual_ridge)
        coef_norms.append(coef_norm_ridge)
    
    # Plot the L-curve
    plt.figure(figsize=(12, 8))
    
    plt.plot(coef_norm_pinv, residual_pinv, 'ro', markersize=10, label='Pseudo-inverse')
    for i, lam in enumerate(lambdas):
        plt.plot(coef_norms[i+1], residual_norms[i+1], 'bx', markersize=10, 
                label=f'Ridge (λ={lam})')
    
    # Connect the points
    plt.plot([coef_norm_pinv] + coef_norms[1:], [residual_pinv] + residual_norms[1:], 'k--', alpha=0.5)
    
    plt.title('L-curve: Trade-off Between Residual Norm and Coefficient Norm', fontsize=14)
    plt.xlabel('Coefficient Norm ||β||₂', fontsize=12)
    plt.ylabel('Residual Norm ||Xβ - y||₂', fontsize=12)
    plt.grid(True)
    
    # Move legend to a better position to avoid overlap
    plt.legend(fontsize=10, loc='upper right')
    
    # Adjust arrow annotations to be shorter and cleaner
    plt.annotate('Minimum residual,\npotentially larger coefficients',
                xy=(coef_norm_pinv, residual_pinv), 
                xytext=(coef_norm_pinv + 0.5, residual_pinv + 0.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                fontsize=10)
    
    plt.annotate('Higher regularization,\nsmaller coefficients,\nlarger residual',
                xy=(coef_norms[-1], residual_norms[-1]), 
                xytext=(coef_norms[-1] - 1.5, residual_norms[-1] + 0.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                fontsize=10)
    
    # Adjust the figure layout
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig(os.path.join(save_dir, "l_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Execute all steps
def main():
    """Execute all steps of the solution."""
    print("LINEAR REGRESSION: PSEUDO-INVERSE EXPLANATION")
    print("=" * 80)
    print()
    
    # Step 1: Define pseudo-inverse
    define_pseudo_inverse()
    print()
    
    # Step 2: When pseudo-inverse is needed
    when_pseudo_inverse_needed()
    print()
    
    # Step 3: Specific scenarios
    X_mc, X_p_gt_n = non_invertible_scenarios()
    print()
    
    # Step 4: Calculate pseudo-inverse using SVD
    pseudo_inverse_svd(X_mc, X_p_gt_n)
    print()
    
    # Step 5: Ridge regression and pseudo-inverse relationship
    ridge_pseudo_inverse_relationship()
    print()
    
    print("=" * 80)
    print("Summary of saved visualizations:")
    print("-" * 80)
    for i, filename in enumerate(os.listdir(save_dir)):
        print(f"{i+1}. {filename}")

if __name__ == "__main__":
    main() 