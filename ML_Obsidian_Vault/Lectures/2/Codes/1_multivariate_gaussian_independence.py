import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images/Multivariate_Gaussian relative to the parent directory
images_dir = os.path.join(parent_dir, "Images", "Multivariate_Gaussian")

# Create images directory if it doesn't exist
os.makedirs(images_dir, exist_ok=True)

def print_step(step_num, description):
    """Print a step in the calculation process."""
    print(f"\nStep {step_num}: {description}")

def print_matrix(name, matrix):
    """Print a matrix with proper formatting."""
    print(f"\n{name}:")
    print(np.array2string(matrix, precision=4, suppress_small=True))

def calculate_covariance(X, Y, Sigma):
    """Calculate covariance between linear combinations of variables."""
    if isinstance(X, np.ndarray):
        X = X.reshape(-1, 1)
    if isinstance(Y, np.ndarray):
        Y = Y.reshape(-1, 1)
    return X.T @ Sigma @ Y

def plot_2d_gaussian(mean, cov, title, filename, show_conditional=False, conditional_x=None, x_label='X₁', y_label='X₂'):
    """Plot a 2D Gaussian distribution with improved visualization."""
    # Create a grid of points
    x = np.linspace(mean[0] - 3*np.sqrt(cov[0,0]), mean[0] + 3*np.sqrt(cov[0,0]), 100)
    y = np.linspace(mean[1] - 3*np.sqrt(cov[1,1]), mean[1] + 3*np.sqrt(cov[1,1]), 100)
    X, Y = np.meshgrid(x, y)
    
    # Create the multivariate normal distribution
    pos = np.dstack((X, Y))
    rv = stats.multivariate_normal(mean, cov)
    
    # Calculate PDF
    Z = rv.pdf(pos)
    
    # Plot
    plt.figure(figsize=(12, 10))
    plt.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(label='Probability Density')
    
    # Add contour lines
    plt.contour(X, Y, Z, levels=10, colors='white', alpha=0.3)
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    
    # Add mean point
    plt.plot(mean[0], mean[1], 'r*', markersize=15, label='Mean')
    
    # Add eigenvectors
    eigenvals, eigenvecs = np.linalg.eig(cov)
    for i in range(2):
        plt.quiver(mean[0], mean[1],
                  eigenvecs[0,i]*np.sqrt(eigenvals[i])*2,
                  eigenvecs[1,i]*np.sqrt(eigenvals[i])*2,
                  color=['red', 'blue'][i], alpha=0.5,
                  scale=1.0, scale_units='xy',
                  label=f'Eigenvector {i+1}')
    
    if show_conditional and conditional_x is not None:
        plt.axvline(x=conditional_x, color='r', linestyle='--', 
                   label=f'Conditional on {x_label} = {conditional_x}')
    
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add text box with distribution parameters
    info_text = f'μ = [{mean[0]:.2f}, {mean[1]:.2f}]\n'
    info_text += f'Σ = [[{cov[0,0]:.2f}, {cov[0,1]:.2f}],\n     [{cov[1,0]:.2f}, {cov[1,1]:.2f}]]'
    plt.text(0.95, 0.05, info_text, transform=plt.gca().transAxes,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, f'{filename}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def calculate_conditional_distribution(mu, Sigma, conditioning_vars, given_values):
    """Calculate conditional distribution parameters."""
    n = len(mu)
    mask = np.zeros(n, dtype=bool)
    mask[conditioning_vars] = True
    
    mu_1 = mu[~mask]
    mu_2 = mu[mask]
    Sigma_11 = Sigma[~mask][:, ~mask]
    Sigma_12 = Sigma[~mask][:, mask]
    Sigma_21 = Sigma[mask][:, ~mask]
    Sigma_22 = Sigma[mask][:, mask]
    
    # Calculate conditional mean and covariance
    Sigma_22_inv = np.linalg.inv(Sigma_22)
    mu_cond = mu_1 + Sigma_12 @ Sigma_22_inv @ (given_values - mu_2)
    Sigma_cond = Sigma_11 - Sigma_12 @ Sigma_22_inv @ Sigma_21
    
    return mu_cond, Sigma_cond

def example1():
    """Example 1: Independence in Multivariate Normal Variables"""
    print("\n=== Example 1: Independence in Multivariate Normal Variables ===")
    
    print_step(1, "Define parameters")
    mu = np.array([1, 2, 3])
    Sigma = np.array([[4, 0, 2],
                     [0, 3, 0],
                     [2, 0, 6]])
    
    print_matrix("Mean vector μ", mu)
    print_matrix("Covariance matrix Σ", Sigma)
    
    print_step(2, "Check independence between pairs")
    pairs = [(0,1), (1,2), (0,2)]
    pair_names = ["(X₁,X₂)", "(X₂,X₃)", "(X₁,X₃)"]
    
    for (i,j), name in zip(pairs, pair_names):
        cov = Sigma[i,j]
        print(f"\nChecking {name}:")
        print(f"Cov{name} = Σ[{i},{j}] = {cov}")
        print(f"→ Variables are {'independent' if cov == 0 else 'not independent'}")
    
    print_step(3, "Calculate covariance between Z = 3X₁ - 6X₃ and X₂")
    # Define coefficients for Z = 3X₁ - 6X₃
    z_coef = np.array([3, 0, -6])
    x2_coef = np.array([0, 1, 0])
    
    # Calculate covariance using matrix operations
    cov_z_x2 = calculate_covariance(z_coef, x2_coef, Sigma)
    
    # Show detailed calculation steps
    print("\nCov(Z,X₂) = Cov(3X₁ - 6X₃, X₂)")
    print(f"         = 3·Cov(X₁,X₂) - 6·Cov(X₃,X₂)")
    print(f"         = 3·({Sigma[0,1]}) - 6·({Sigma[2,1]})")
    print(f"         = {3 * Sigma[0,1]} - {6 * Sigma[2,1]}")
    print(f"         = {cov_z_x2[0,0]}")
    
    print_step(4, "Calculate conditional independence of X₁ and X₃ given X₂")
    # Extract relevant submatrices
    Sigma_aa = np.array([[Sigma[0,0], Sigma[0,2]],
                        [Sigma[2,0], Sigma[2,2]]])  # Covariance of (X₁,X₃)
    Sigma_ab = np.array([[Sigma[0,1]],
                        [Sigma[2,1]]])  # Covariance between (X₁,X₃) and X₂
    Sigma_bb = np.array([[Sigma[1,1]]])  # Variance of X₂
    
    # Calculate conditional covariance matrix
    Sigma_bb_inv = np.linalg.inv(Sigma_bb)
    Sigma_cond = Sigma_aa - Sigma_ab @ Sigma_bb_inv @ Sigma_ab.T
    
    print("\nStep-by-step conditional covariance calculation:")
    print_matrix("Σ_aa (Covariance of X₁,X₃)", Sigma_aa)
    print_matrix("Σ_ab (Covariance between (X₁,X₃) and X₂)", Sigma_ab)
    print_matrix("Σ_bb (Variance of X₂)", Sigma_bb)
    print_matrix("Σ_bb⁻¹", Sigma_bb_inv)
    print_matrix("Σ_ab · Σ_bb⁻¹ · Σ_ba", Sigma_ab @ Sigma_bb_inv @ Sigma_ab.T)
    print_matrix("Conditional covariance matrix", Sigma_cond)
    
    print("\nSince the off-diagonal element in the conditional covariance matrix " +
          f"is {Sigma_cond[0,1]:.2f} (not 0), X₁ and X₃ are not conditionally independent given X₂")
    
    # Plot marginal distributions
    plot_2d_gaussian(mu[[1,2]], Sigma[1:3,1:3][[0,1]][:,[0,1]], 
                    'Joint Distribution of X₂ and X₃\n(Independent Variables)',
                    'example1_independent_pair',
                    x_label='X₂', y_label='X₃')

def example2():
    """Example 2: Creating Independent Variables Through Linear Transformations"""
    print("\n=== Example 2: Creating Independent Variables Through Linear Transformations ===")
    
    print_step(1, "Define parameters")
    mu = np.array([0, 0, 0])
    Sigma = np.array([[4, 2, 0],
                     [2, 5, 1],
                     [0, 1, 3]])
    
    print_matrix("Mean vector μ", mu)
    print_matrix("Covariance matrix Σ", Sigma)
    
    print_step(2, "Check independence between pairs")
    pairs = [(0,1), (1,2), (0,2)]
    pair_names = ["(X₁,X₂)", "(X₂,X₃)", "(X₁,X₃)"]
    
    for (i,j), name in zip(pairs, pair_names):
        cov = Sigma[i,j]
        print(f"\nChecking {name}:")
        print(f"Cov{name} = Σ[{i},{j}] = {cov}")
        print(f"→ Variables are {'independent' if cov == 0 else 'not independent'}")
    
    print_step(3, "Calculate covariance between Z₁ = X₁ - ½X₂ and Z₂ = X₂ - ⅕X₃")
    # Define coefficients for Z₁ and Z₂
    z1_coef = np.array([1, -0.5, 0])    # Z₁ = X₁ - ½X₂
    z2_coef = np.array([0, 1, -0.2])    # Z₂ = X₂ - ⅕X₃
    
    # Calculate covariance using matrix operations
    cov_z1_z2 = calculate_covariance(z1_coef, z2_coef, Sigma)
    
    # Show detailed calculation steps
    print("\nCov(Z₁,Z₂) = Cov(X₁ - ½X₂, X₂ - ⅕X₃)")
    print("         = Cov(X₁,X₂) - ⅕Cov(X₁,X₃) - ½Cov(X₂,X₂) + ⅒Cov(X₂,X₃)")
    print(f"         = {Sigma[0,1]} - ⅕·{Sigma[0,2]} - ½·{Sigma[1,1]} + ⅒·{Sigma[1,2]}")
    print(f"         = {Sigma[0,1]} - {0.2 * Sigma[0,2]} - {0.5 * Sigma[1,1]} + {0.1 * Sigma[1,2]}")
    print(f"         = {cov_z1_z2[0,0]}")
    
    print_step(4, "Find linear transformation for independence")
    # Extract 2x2 submatrix for X₁,X₂
    Sigma_12 = Sigma[:2,:2]
    print("\nCovariance matrix of (X₁,X₂):")
    print_matrix("Σ₁₂", Sigma_12)
    
    # Perform eigendecomposition
    eigenvals, eigenvecs = np.linalg.eig(Sigma_12)
    
    print("\nEigendecomposition of Σ₁₂:")
    print_matrix("Eigenvalues", eigenvals)
    print_matrix("Eigenvectors", eigenvecs)
    
    # Calculate transformation matrix A
    A = eigenvecs.T
    print_matrix("Transformation matrix A = eigenvectors^T", A)
    
    # Calculate transformed covariance
    transformed_cov = A @ Sigma_12 @ A.T
    print("\nVerifying independence in transformed space:")
    print_matrix("Transformed covariance matrix = AΣ₁₂A^T", transformed_cov)
    print("\nNote: The off-diagonal elements are effectively zero (up to numerical precision)")
    
    # Calculate transformed mean
    transformed_mean = A @ mu[:2]
    print_matrix("Transformed mean = Aμ", transformed_mean)
    
    # Plot original and transformed distributions
    plot_2d_gaussian(mu[:2], Sigma_12, 
                    'Original Joint Distribution of X₁ and X₂\nCorrelated Variables',
                    'example2_original')
    
    plot_2d_gaussian(transformed_mean, transformed_cov,
                    'Transformed Distribution\nIndependent Variables Y₁ and Y₂',
                    'example2_transformed',
                    x_label='Y₁', y_label='Y₂')

def example3():
    """Example 3: Independence Properties in Statistical Inference"""
    print("\n=== Example 3: Independence Properties in Statistical Inference ===")
    
    print_step(1, "Define parameters")
    mu = np.array([1, 2, 3])
    Sigma = np.array([[3, 1, 2],
                     [1, 4, 0],
                     [2, 0, 5]])
    
    print_matrix("Mean vector μ", mu)
    print_matrix("Covariance matrix Σ", Sigma)
    
    print_step(2, "Check independence between pairs")
    pairs = [(0,1), (1,2), (0,2)]
    pair_names = ["(X₁,X₂)", "(X₂,X₃)", "(X₁,X₃)"]
    
    for (i,j), name in zip(pairs, pair_names):
        cov = Sigma[i,j]
        print(f"\nChecking {name}:")
        print(f"Cov{name} = Σ[{i},{j}] = {cov}")
        print(f"→ Variables are {'independent' if cov == 0 else 'not independent'}")
    
    print_step(3, "Calculate covariance between Z = 3X₁ - 6X₃ and X₂")
    # Define coefficients for Z = 3X₁ - 6X₃
    z_coef = np.array([3, 0, -6])
    x2_coef = np.array([0, 1, 0])
    
    # Calculate covariance using matrix operations
    cov_z_x2 = calculate_covariance(z_coef, x2_coef, Sigma)
    
    # Show detailed calculation steps
    print("\nCov(Z,X₂) = Cov(3X₁ - 6X₃, X₂)")
    print(f"         = 3·Cov(X₁,X₂) - 6·Cov(X₃,X₂)")
    print(f"         = 3·({Sigma[0,1]}) - 6·({Sigma[2,1]})")
    print(f"         = {3 * Sigma[0,1]} - {6 * Sigma[2,1]}")
    print(f"         = {cov_z_x2[0,0]}")
    
    print_step(4, "Calculate conditional independence of X₁ and X₃ given X₂")
    print("\nCalculating conditional covariance matrix:")
    print("Σ₁₁ = covariance matrix of (X₁,X₃)")
    
    # Extract relevant submatrices
    Sigma_11 = np.array([[Sigma[0,0], Sigma[0,2]],
                        [Sigma[2,0], Sigma[2,2]]])  # Covariance of (X₁,X₃)
    print_matrix("Σ₁₁", Sigma_11)
    
    print("\nΣ₁₂ = covariance vector between (X₁,X₃) and X₂")
    Sigma_12 = np.array([[Sigma[0,1]],
                        [Sigma[2,1]]])  # Covariance between (X₁,X₃) and X₂
    print_matrix("Σ₁₂", Sigma_12)
    
    print("\nΣ₂₂ = variance of X₂")
    Sigma_22 = np.array([[Sigma[1,1]]])  # Variance of X₂
    print_matrix("Σ₂₂", Sigma_22)
    
    # Calculate conditional covariance matrix
    Sigma_22_inv = np.linalg.inv(Sigma_22)
    Sigma_cond = Sigma_11 - Sigma_12 @ Sigma_22_inv @ Sigma_12.T
    
    print("\nConditional covariance matrix:")
    print_matrix("Σ₁₁|₂ = Σ₁₁ - Σ₁₂Σ₂₂⁻¹Σ₂₁", Sigma_cond)
    
    print("\nSince the off-diagonal element is not 0, X₁ and X₃ are not conditionally independent given X₂")
    
    # Plot distributions
    plot_2d_gaussian(mu[[0,1]], Sigma[:2,:2],
                    'Original Joint Distribution of X₁ and X₂\nCorrelated Variables',
                    'example3_original')
    
    # Plot conditional distribution
    x2_value = 2  # Conditioning on X₂ = 2
    # Calculate conditional mean
    mu_cond = mu[[0,2]] + Sigma_12 @ Sigma_22_inv @ np.array([[x2_value - mu[1]]])
    mu_cond = mu_cond.flatten()  # Convert to 1D array
    
    # Plot conditional distribution
    plot_2d_gaussian(mu_cond[[0,1]], Sigma_cond,
                    f'Conditional Distribution of X₁ and X₃ given X₂ = {x2_value}\nStill Dependent',
                    'example3_conditional',
                    x_label='X₁', y_label='X₃')

def example4():
    """Example 4: Multivariate Normal with Partitioned Vectors"""
    print("\n=== Example 4: Multivariate Normal with Partitioned Vectors ===")
    
    print_step(1, "Define parameters")
    mu = np.array([4, 45, 30, 35, 40])
    Sigma = np.array([[1, 1, 0, 0, 0],
                     [1, 15, 1, 4, 0],
                     [0, 1, 5, 4, 0],
                     [0, 4, 4, 8, 0],
                     [0, 0, 0, 0, 9]])
    
    print_matrix("Mean vector μ", mu)
    print_matrix("Covariance matrix Σ", Sigma)
    
    print_step(2, "Calculate PDF of X₂ = [X₁, X₃]")
    # Extract relevant components
    idx_X2 = [0, 2]  # Indices for X₁ and X₃
    mu_2 = mu[idx_X2]
    Sigma_2 = Sigma[np.ix_(idx_X2, idx_X2)]
    
    print("\nMarginal distribution parameters for X₂:")
    print_matrix("Mean vector μ₂", mu_2)
    print_matrix("Covariance matrix Σ₂", Sigma_2)
    
    print_step(3, "Calculate covariance matrix of Y = [X₁, X₂, X₃, X₄, X₅]")
    # Reorder variables according to Y = [X₂, X₄, X₅, X₁, X₃]
    idx_Y = [1, 3, 4, 0, 2]
    mu_Y = mu[idx_Y]
    Sigma_Y = Sigma[np.ix_(idx_Y, idx_Y)]
    
    print("\nReordered parameters for Y:")
    print_matrix("Mean vector μY", mu_Y)
    print_matrix("Covariance matrix ΣY", Sigma_Y)
    
    print_step(4, "Calculate conditional distribution of X₁ given X₂ = [6, 24]")
    # X₁ = [X₂, X₄, X₅], X₂ = [X₁, X₃]
    x2_values = np.array([6, 24])
    
    # Calculate conditional distribution
    idx_X1 = [1, 3, 4]  # Indices for X₂, X₄, X₅
    idx_cond = [0, 2]   # Indices for X₁, X₃
    
    mu_cond, Sigma_cond = calculate_conditional_distribution(
        mu, Sigma, idx_cond, x2_values)
    
    print("\nStep-by-step calculation of conditional distribution:")
    print("\nΣ₁₁ = covariance matrix of X₁ = [X₂, X₄, X₅]")
    Sigma_11 = Sigma[np.ix_(idx_X1, idx_X1)]
    print_matrix("Σ₁₁", Sigma_11)
    
    print("\nΣ₁₂ = covariance between X₁ and X₂")
    Sigma_12 = Sigma[np.ix_(idx_X1, idx_cond)]
    print_matrix("Σ₁₂", Sigma_12)
    
    print("\nΣ₂₂ = covariance matrix of X₂ = [X₁, X₃]")
    Sigma_22 = Sigma[np.ix_(idx_cond, idx_cond)]
    print_matrix("Σ₂₂", Sigma_22)
    
    print("\nConditional distribution parameters:")
    print_matrix("Conditional mean μ₁|₂", mu_cond)
    print_matrix("Conditional covariance Σ₁|₂", Sigma_cond)
    
    # Plot original and conditional distributions
    plot_2d_gaussian(mu[[1,3]], Sigma[np.ix_([1,3], [1,3])],
                    'Joint Distribution of X₂ and X₄',
                    'example4_original',
                    x_label='X₂', y_label='X₄')
    
    # Plot conditional distribution for X₂ and X₄
    plot_2d_gaussian(mu_cond[[0,1]], Sigma_cond[np.ix_([0,1], [0,1])],
                    'Conditional Distribution of X₂ and X₄\ngiven X₁ = 6, X₃ = 24',
                    'example4_conditional',
                    x_label='X₂', y_label='X₄')

def example5():
    """Example 5: Independent Variables with Inverse of Covariance Matrix"""
    print("\n=== Example 5: Independent Variables with Inverse of Covariance Matrix ===")
    
    print_step(1, "Define parameters")
    mu = np.array([0, 1, -2])
    Sigma = np.array([[4, 1, -1],
                     [1, 1, 0],
                     [-1, 0, 1]])
    Sigma_inv = np.array([[1/2, -1/2, 1/2],
                         [-1/2, 3/2, -1/2],
                         [1/2, -1/2, 3/2]])
    
    print_matrix("Mean vector μ", mu)
    print_matrix("Covariance matrix Σ", Sigma)
    print_matrix("Inverse covariance matrix Σ⁻¹", Sigma_inv)
    
    # Verify the inverse
    print("\nVerifying Σ·Σ⁻¹ = I:")
    print_matrix("Σ·Σ⁻¹", Sigma @ Sigma_inv)
    
    print_step(2, "Check independence between pairs")
    pairs = [(0,1), (1,2), (0,2)]
    pair_names = ["(X₁,X₂)", "(X₂,X₃)", "(X₁,X₃)"]
    
    for (i,j), name in zip(pairs, pair_names):
        cov = Sigma[i,j]
        print(f"\nChecking {name}:")
        print(f"Cov{name} = Σ[{i},{j}] = {cov}")
        print(f"→ Variables are {'independent' if cov == 0 else 'not independent'}")
    
    print_step(3, "Find values of a and b for Z = X₁ - aX₂ - bX₃ to be independent of X₁")
    print("\nFor Z to be independent of X₁, we need Cov(Z,X₁) = 0:")
    print("Cov(X₁ - aX₂ - bX₃, X₁) = 0")
    print("Var(X₁) - a·Cov(X₂,X₁) - b·Cov(X₃,X₁) = 0")
    print(f"4 - a·1 - b·(-1) = 0")
    print("4 - a + b = 0")
    print("Therefore: a = 4 + b")
    
    # Example solution
    b = 0
    a = 4 + b
    print(f"\nExample solution: b = {b}, a = {a}")
    
    # Verify the solution
    z_coef = np.array([1, -a, -b])
    x1_coef = np.array([1, 0, 0])
    cov_z_x1 = calculate_covariance(z_coef, x1_coef, Sigma)
    print(f"\nVerification - Cov(Z,X₁) = {cov_z_x1[0,0]}")
    
    print_step(4, "Check conditional independence given X₃")
    # Calculate conditional covariance matrix
    mu_cond, Sigma_cond = calculate_conditional_distribution(
        mu, Sigma, [2], np.array([-2]))
    
    print("\nStep-by-step calculation of conditional distribution:")
    print("\nΣ₁₁ = covariance matrix of (X₁,X₂)")
    Sigma_11 = Sigma[:2,:2]
    print_matrix("Σ₁₁", Sigma_11)
    
    print("\nΣ₁₂ = covariance vector between (X₁,X₂) and X₃")
    Sigma_12 = Sigma[:2,2:]
    print_matrix("Σ₁₂", Sigma_12)
    
    print("\nΣ₂₂ = variance of X₃")
    Sigma_22 = Sigma[2:,2:]
    print_matrix("Σ₂₂", Sigma_22)
    
    print("\nConditional covariance matrix:")
    print_matrix("Σ₁₁|₂ = Σ₁₁ - Σ₁₂Σ₂₂⁻¹Σ₂₁", Sigma_cond)
    
    print("\nFor conditional independence, we need a = 3")
    
    # Plot distributions
    plot_2d_gaussian(mu[:2], Sigma[:2,:2],
                    'Joint Distribution of X₁ and X₂',
                    'example5_original')
    
    # Plot conditional distribution
    plot_2d_gaussian(mu_cond, Sigma_cond,
                    'Conditional Distribution of X₁ and X₂\ngiven X₃ = -2',
                    'example5_conditional',
                    show_conditional=True,
                    conditional_x=-2)

def main():
    print("=== Multivariate Gaussian Independence Examples ===")
    
    example1()
    example2()
    example3()
    example4()
    example5()
    
    print("\nAll examples completed. Check the Images/Multivariate_Gaussian directory for visualizations.")

if __name__ == "__main__":
    main() 