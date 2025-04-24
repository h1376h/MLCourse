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

def calculate_determinant_2x2(matrix):
    """Calculate determinant of 2x2 matrix with steps."""
    a, b = matrix[0,0], matrix[0,1]
    c, d = matrix[1,0], matrix[1,1]
    det = a*d - b*c
    steps = [
        "For 2x2 matrix:",
        f"|{a} {b}| = ({a})({d}) - ({b})({c})",
        f"|{c} {d}|",
        f"= {a*d} - {b*c}",
        f"= {det}"
    ]
    return det, "\n".join(steps)

def calculate_inverse_2x2(matrix):
    """Calculate inverse of 2x2 matrix with steps."""
    a, b = matrix[0,0], matrix[0,1]
    c, d = matrix[1,0], matrix[1,1]
    det = a*d - b*c
    if det == 0:
        return None, "Matrix is not invertible (determinant = 0)"
    
    inv = np.array([[d/det, -b/det], [-c/det, a/det]])
    steps = [
        "For 2x2 matrix inverse:",
        f"1. Calculate determinant = {det}",
        "2. Adjugate matrix:",
        f"   |{d} {-b}|",
        f"   |{-c} {a}|",
        "3. Multiply by 1/determinant:",
        f"   |{d/det:.4f} {-b/det:.4f}|",
        f"   |{-c/det:.4f} {a/det:.4f}|"
    ]
    return inv, "\n".join(steps)

def calculate_conditional_distribution_steps(mu, Sigma, idx_X1, idx_cond, x2_values):
    """Calculate conditional distribution with detailed steps."""
    # Extract submatrices with steps
    Sigma_11 = Sigma[np.ix_(idx_X1, idx_X1)]
    Sigma_12 = Sigma[np.ix_(idx_X1, idx_cond)]
    Sigma_21 = Sigma[np.ix_(idx_cond, idx_X1)]
    Sigma_22 = Sigma[np.ix_(idx_cond, idx_cond)]
    
    # Calculate inverse with steps
    Sigma_22_inv, inv_steps = calculate_inverse_2x2(Sigma_22)
    
    # Calculate conditional mean
    mu_1 = mu[idx_X1]
    mu_2_cond = mu[idx_cond]
    diff = x2_values - mu_2_cond
    
    # Calculate Sigma_12 @ Sigma_22_inv with steps
    S12_S22inv = Sigma_12 @ Sigma_22_inv
    S12_S22inv_steps = [
        "Calculate Σ₁₂Σ₂₂⁻¹:",
        f"[{Sigma_12[0,0]} {Sigma_12[0,1]}] [{Sigma_22_inv[0,0]:.4f} {Sigma_22_inv[0,1]:.4f}]",
        f"[{Sigma_12[1,0]} {Sigma_12[1,1]}] [{Sigma_22_inv[1,0]:.4f} {Sigma_22_inv[1,1]:.4f}]",
        f"= [{S12_S22inv[0,0]:.4f} {S12_S22inv[0,1]:.4f}]",
        f"  [{S12_S22inv[1,0]:.4f} {S12_S22inv[1,1]:.4f}]"
    ]
    
    # Calculate conditional mean with steps
    mu_cond = mu_1 + S12_S22inv @ diff
    mu_cond_steps = [
        "Calculate conditional mean μ₁|₂ = μ₁ + Σ₁₂Σ₂₂⁻¹(x₂ - μ₂):",
        f"μ₁|₂ = [{mu_1[0]}] + [{S12_S22inv[0,0]:.4f} {S12_S22inv[0,1]:.4f}] [{diff[0]}]",
        f"       [{mu_1[1]}]   [{S12_S22inv[1,0]:.4f} {S12_S22inv[1,1]:.4f}] [{diff[1]}]",
        f"= [{mu_cond[0]:.4f}]",
        f"  [{mu_cond[1]:.4f}]"
    ]
    
    # Calculate conditional covariance
    Sigma_cond = Sigma_11 - S12_S22inv @ Sigma_21
    Sigma_cond_steps = [
        "Calculate conditional covariance Σ₁|₂ = Σ₁₁ - Σ₁₂Σ₂₂⁻¹Σ₂₁:",
        f"Σ₁|₂ = [{Sigma_11[0,0]} {Sigma_11[0,1]}] - [{S12_S22inv[0,0]:.4f} {S12_S22inv[0,1]:.4f}] [{Sigma_21[0,0]} {Sigma_21[0,1]}]",
        f"       [{Sigma_11[1,0]} {Sigma_11[1,1]}]   [{S12_S22inv[1,0]:.4f} {S12_S22inv[1,1]:.4f}] [{Sigma_21[1,0]} {Sigma_21[1,1]}]",
        f"= [{Sigma_cond[0,0]:.4f} {Sigma_cond[0,1]:.4f}]",
        f"  [{Sigma_cond[1,0]:.4f} {Sigma_cond[1,1]:.4f}]"
    ]
    
    return {
        'mu_cond': mu_cond,
        'Sigma_cond': Sigma_cond,
        'steps': {
            'inverse': inv_steps,
            'S12_S22inv': "\n".join(S12_S22inv_steps),
            'mu_cond': "\n".join(mu_cond_steps),
            'Sigma_cond': "\n".join(Sigma_cond_steps)
        }
    }

def calculate_3x3_inverse(matrix):
    """Calculate inverse of 3x3 matrix with detailed steps."""
    # Step 1: Calculate cofactor matrix
    def cofactor(i, j):
        # Get the 2x2 submatrix
        sub_matrix = np.delete(np.delete(matrix, i, 0), j, 1)
        # Calculate determinant of 2x2 matrix
        det = sub_matrix[0,0]*sub_matrix[1,1] - sub_matrix[0,1]*sub_matrix[1,0]
        # Apply (-1)^(i+j)
        return (-1)**(i+j) * det
    
    cofactor_matrix = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            cofactor_matrix[i,j] = cofactor(i,j)
    
    # Step 2: Calculate determinant using first row expansion
    det = sum(matrix[0,j] * cofactor_matrix[0,j] for j in range(3))
    
    # Step 3: Calculate adjugate (transpose of cofactor matrix)
    adjugate = cofactor_matrix.T
    
    # Step 4: Divide by determinant
    inverse = adjugate / det
    
    steps = [
        "1. Calculate cofactor matrix:",
        f"   {cofactor_matrix[0,0]:.4f} {cofactor_matrix[0,1]:.4f} {cofactor_matrix[0,2]:.4f}",
        f"   {cofactor_matrix[1,0]:.4f} {cofactor_matrix[1,1]:.4f} {cofactor_matrix[1,2]:.4f}",
        f"   {cofactor_matrix[2,0]:.4f} {cofactor_matrix[2,1]:.4f} {cofactor_matrix[2,2]:.4f}",
        "\n2. Calculate determinant using first row expansion:",
        f"   det = {matrix[0,0]}·{cofactor_matrix[0,0]:.4f} + {matrix[0,1]}·{cofactor_matrix[0,1]:.4f} + {matrix[0,2]}·{cofactor_matrix[0,2]:.4f}",
        f"   det = {det:.4f}",
        "\n3. Calculate adjugate (transpose of cofactor matrix):",
        f"   {adjugate[0,0]:.4f} {adjugate[0,1]:.4f} {adjugate[0,2]:.4f}",
        f"   {adjugate[1,0]:.4f} {adjugate[1,1]:.4f} {adjugate[1,2]:.4f}",
        f"   {adjugate[2,0]:.4f} {adjugate[2,1]:.4f} {adjugate[2,2]:.4f}",
        "\n4. Divide by determinant to get inverse:",
        f"   {inverse[0,0]:.4f} {inverse[0,1]:.4f} {inverse[0,2]:.4f}",
        f"   {inverse[1,0]:.4f} {inverse[1,1]:.4f} {inverse[1,2]:.4f}",
        f"   {inverse[2,0]:.4f} {inverse[2,1]:.4f} {inverse[2,2]:.4f}"
    ]
    
    return inverse, det, "\n".join(steps)

def calculate_independence_conditions(Sigma, a, b):
    """Calculate covariance between Z and X₁ with detailed steps."""
    # Z = X₁ - aX₂ - bX₃
    z_coef = np.array([1, -a, -b])
    x1_coef = np.array([1, 0, 0])
    
    # Calculate Cov(Z,X₁)
    cov = z_coef.T @ Sigma @ x1_coef
    
    steps = [
        "Calculate Cov(Z,X₁) = Cov(X₁ - aX₂ - bX₃, X₁):",
        "= Var(X₁) - a·Cov(X₂,X₁) - b·Cov(X₃,X₁)",
        f"= {Sigma[0,0]} - ({a})·({Sigma[0,1]}) - ({b})·({Sigma[0,2]})",
        f"= {Sigma[0,0]} - {a*Sigma[0,1]} - {b*Sigma[0,2]}",
        f"= {cov}"
    ]
    
    return cov, "\n".join(steps)

def calculate_conditional_independence(Sigma, a, x3_value=None):
    """Calculate conditional covariance between Z and X₁ given X₃."""
    # Extract relevant submatrices for X₁,X₂ given X₃
    Sigma_11 = Sigma[:2,:2]
    Sigma_12 = Sigma[:2,2:3]
    Sigma_21 = Sigma[2:3,:2]
    Sigma_22 = Sigma[2:3,2:3]
    
    # Calculate conditional covariance matrix
    Sigma_22_inv = 1/Sigma_22[0,0]
    Sigma_cond = Sigma_11 - Sigma_12 @ np.array([[Sigma_22_inv]]) @ Sigma_21
    
    # Calculate conditional covariance between Z and X₁
    z_coef = np.array([1, -a])
    x1_coef = np.array([1, 0])
    cond_cov = z_coef.T @ Sigma_cond @ x1_coef
    
    steps = [
        "1. Extract submatrices:",
        f"Σ₁₁ = [{Sigma_11[0,0]} {Sigma_11[0,1]}]",
        f"     [{Sigma_11[1,0]} {Sigma_11[1,1]}]",
        f"\nΣ₁₂ = [{Sigma_12[0,0]}]",
        f"     [{Sigma_12[1,0]}]",
        f"\nΣ₂₂ = [{Sigma_22[0,0]}]",
        "\n2. Calculate conditional covariance matrix:",
        "Σ₁₁|₃ = Σ₁₁ - Σ₁₂Σ₂₂⁻¹Σ₂₁",
        f"= [{Sigma_cond[0,0]:.4f} {Sigma_cond[0,1]:.4f}]",
        f"  [{Sigma_cond[1,0]:.4f} {Sigma_cond[1,1]:.4f}]",
        "\n3. Calculate conditional covariance between Z and X₁:",
        f"Cov(Z,X₁|X₃) = {cond_cov:.4f}"
    ]
    
    return cond_cov, Sigma_cond, "\n".join(steps)

def calculate_eigendecomposition_2x2(matrix):
    """Calculate eigenvalues and eigenvectors of 2x2 matrix with detailed steps."""
    a, b = matrix[0,0], matrix[0,1]
    c, d = matrix[1,0], matrix[1,1]
    
    # Step 1: Characteristic equation
    # |A - λI| = 0
    # |(a-λ  b  )| = 0
    # |(c    d-λ)|
    # (a-λ)(d-λ) - bc = 0
    # λ² - (a+d)λ + (ad-bc) = 0
    
    # Calculate coefficients
    sum_diag = a + d  # trace
    det = a*d - b*c
    
    # Step 2: Solve quadratic equation
    # λ = [-(-trace) ± √(trace² - 4det)] / 2
    discriminant = sum_diag**2 - 4*det
    sqrt_disc = np.sqrt(discriminant)
    lambda1 = (sum_diag + sqrt_disc) / 2
    lambda2 = (sum_diag - sqrt_disc) / 2
    eigenvals = np.array([lambda1, lambda2])
    
    # Step 3: Find eigenvectors
    eigenvecs = np.zeros((2,2))
    for i, lambda_i in enumerate(eigenvals):
        # For each λ, solve (A - λI)v = 0
        # (a-λ)v₁ + bv₂ = 0
        # cv₁ + (d-λ)v₂ = 0
        
        # Use first equation: v₁ = [-b/(a-λ)]v₂
        # Normalize to get unit vector
        v2 = 1.0
        v1 = -b/(a-lambda_i) if abs(a-lambda_i) > 1e-10 else 1.0
        norm = np.sqrt(v1*v1 + v2*v2)
        eigenvecs[:,i] = np.array([v1/norm, v2/norm])
    
    steps = [
        "1. Form characteristic equation |A - λI| = 0:",
        f"   |(a-λ  b  )| = |(({a}-λ)  {b}  )| = 0",
        f"   |(c    d-λ)| = |({c}    ({d}-λ))| = 0",
        f"   ({a}-λ)({d}-λ) - ({b})({c}) = 0",
        f"   λ² - ({a}+{d})λ + ({a}·{d}-{b}·{c}) = 0",
        "\n2. Solve quadratic equation:",
        f"   trace = {sum_diag}",
        f"   det = {det}",
        f"   discriminant = {discriminant}",
        f"   λ₁ = {lambda1:.4f}",
        f"   λ₂ = {lambda2:.4f}",
        "\n3. Find eigenvectors:",
        "   For each λᵢ, solve (A - λᵢI)v = 0:",
        f"   For λ₁ = {lambda1:.4f}:",
        f"   ({a}-{lambda1:.4f})v₁ + {b}v₂ = 0",
        f"   {c}v₁ + ({d}-{lambda1:.4f})v₂ = 0",
        f"   v₁ = [{eigenvecs[0,0]:.4f}]",
        f"   v₂ = [{eigenvecs[1,0]:.4f}]",
        f"\n   For λ₂ = {lambda2:.4f}:",
        f"   ({a}-{lambda2:.4f})v₁ + {b}v₂ = 0",
        f"   {c}v₁ + ({d}-{lambda2:.4f})v₂ = 0",
        f"   v₁ = [{eigenvecs[0,1]:.4f}]",
        f"   v₂ = [{eigenvecs[1,1]:.4f}]"
    ]
    
    return eigenvals, eigenvecs, "\n".join(steps)

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
    
    # Perform eigendecomposition with detailed steps
    eigenvals, eigenvecs, eig_steps = calculate_eigendecomposition_2x2(Sigma_12)
    print("\nDetailed eigendecomposition steps:")
    print(eig_steps)
    
    # Calculate transformation matrix A
    A = eigenvecs.T
    print_matrix("\nTransformation matrix A = eigenvectors^T", A)
    
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
    
    # Calculate determinant with steps
    det_Sigma_2, det_steps = calculate_determinant_2x2(Sigma_2)
    print("\nDeterminant calculation:")
    print(det_steps)
    
    # Calculate inverse with steps
    inv_Sigma_2, inv_steps = calculate_inverse_2x2(Sigma_2)
    print("\nInverse calculation:")
    print(inv_steps)
    
    print_step(3, "Calculate covariance matrix of Y = [X₁, X₂, X₃, X₄, X₅]")
    # Reorder variables according to Y = [X₂, X₄, X₅, X₁, X₃]
    idx_Y = [1, 3, 4, 0, 2]
    mu_Y = mu[idx_Y]
    Sigma_Y = Sigma[np.ix_(idx_Y, idx_Y)]
    
    print("\nReordered parameters for Y:")
    print_matrix("Mean vector μY", mu_Y)
    print_matrix("Covariance matrix ΣY", Sigma_Y)
    
    print_step(4, "Calculate conditional distribution of X₁ given X₂ = [6, 24]")
    x2_values = np.array([6, 24])
    idx_X1 = [1, 3]  # Indices for X₂, X₄
    idx_cond = [0, 2]   # Indices for X₁, X₃
    
    # Calculate conditional distribution with detailed steps
    cond_dist = calculate_conditional_distribution_steps(mu, Sigma, idx_X1, idx_cond, x2_values)
    
    print("\nStep-by-step calculations:")
    print("\n1. Calculate Σ₂₂⁻¹:")
    print(cond_dist['steps']['inverse'])
    print("\n2. Calculate Σ₁₂Σ₂₂⁻¹:")
    print(cond_dist['steps']['S12_S22inv'])
    print("\n3. Calculate conditional mean:")
    print(cond_dist['steps']['mu_cond'])
    print("\n4. Calculate conditional covariance:")
    print(cond_dist['steps']['Sigma_cond'])
    
    # Plot distributions
    plot_2d_gaussian(mu[[1,3]], Sigma[np.ix_([1,3], [1,3])],
                    'Joint Distribution of X₂ and X₄',
                    'example4_original',
                    x_label='X₂', y_label='X₄')
    
    # Plot conditional distribution for X₂ and X₄
    plot_2d_gaussian(cond_dist['mu_cond'], cond_dist['Sigma_cond'],
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
    
    print_matrix("Mean vector μ", mu)
    print_matrix("Covariance matrix Σ", Sigma)
    
    # Calculate inverse with detailed steps
    Sigma_inv, det, inv_steps = calculate_3x3_inverse(Sigma)
    print("\nDetailed inverse calculation:")
    print(inv_steps)
    
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
    # Example solution with b = 0
    b = 0
    a = 4 + b
    
    # Calculate covariance with detailed steps
    cov_z_x1, cov_steps = calculate_independence_conditions(Sigma, a, b)
    print("\nDetailed covariance calculation:")
    print(cov_steps)
    
    print(f"\nExample solution: b = {b}, a = {a}")
    print(f"Verification - Cov(Z,X₁) = {cov_z_x1}")
    
    print_step(4, "Check conditional independence given X₃")
    # Calculate conditional independence with a = 3
    a_cond = 3
    cond_cov, Sigma_cond, cond_steps = calculate_conditional_independence(Sigma, a_cond)
    
    print("\nDetailed conditional independence calculation:")
    print(cond_steps)
    
    # Plot distributions
    plot_2d_gaussian(mu[:2], Sigma[:2,:2], 
                    'Joint Distribution of X₁ and X₂',
                    'example5_original')
    
    # Plot conditional distribution
    x3_value = -2  # Conditioning on X₃ = -2
    mu_cond = mu[:2] + Sigma[:2,2:3] @ np.array([[1/Sigma[2,2]]]) @ np.array([x3_value - mu[2]])
    plot_2d_gaussian(mu_cond, Sigma_cond,
                    'Conditional Distribution of X₁ and X₂\ngiven X₃ = -2',
                    'example5_conditional',
                    show_conditional=True,
                    conditional_x=x3_value)

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