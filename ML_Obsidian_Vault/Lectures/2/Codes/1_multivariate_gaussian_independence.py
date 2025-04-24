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
    """Calculate inverse of 3x3 matrix with very detailed steps."""
    steps = ["Calculating inverse of 3x3 matrix using cofactor method:"]
    
    # Step 1: Calculate minors and cofactors
    steps.append("\n1. Calculate minors and cofactors for each element:")
    cofactor_matrix = np.zeros((3,3))
    minor_steps = []
    
    for i in range(3):
        for j in range(3):
            # Get the 2x2 submatrix
            sub_matrix = np.delete(np.delete(matrix, i, 0), j, 1)
            minor_steps.append(f"\nPosition ({i+1},{j+1}):")
            minor_steps.append(f"Submatrix:")
            minor_steps.append(f"[{sub_matrix[0,0]} {sub_matrix[0,1]}]")
            minor_steps.append(f"[{sub_matrix[1,0]} {sub_matrix[1,1]}]")
            
            # Calculate determinant of 2x2 matrix
            det = sub_matrix[0,0]*sub_matrix[1,1] - sub_matrix[0,1]*sub_matrix[1,0]
            minor_steps.append(f"Minor = ({sub_matrix[0,0]}×{sub_matrix[1,1]}) - ({sub_matrix[0,1]}×{sub_matrix[1,0]}) = {det}")
            
            # Apply (-1)^(i+j)
            cofactor = (-1)**(i+j) * det
            minor_steps.append(f"Cofactor = (-1)^({i+1}+{j+1}) × {det} = {cofactor}")
            
            cofactor_matrix[i,j] = cofactor
    
    steps.extend(minor_steps)
    
    steps.append("\nCofactor matrix:")
    steps.append(f"[{cofactor_matrix[0,0]:.4f} {cofactor_matrix[0,1]:.4f} {cofactor_matrix[0,2]:.4f}]")
    steps.append(f"[{cofactor_matrix[1,0]:.4f} {cofactor_matrix[1,1]:.4f} {cofactor_matrix[1,2]:.4f}]")
    steps.append(f"[{cofactor_matrix[2,0]:.4f} {cofactor_matrix[2,1]:.4f} {cofactor_matrix[2,2]:.4f}]")
    
    # Step 2: Calculate determinant using first row expansion
    steps.append("\n2. Calculate determinant using first row expansion:")
    det_terms = [matrix[0,j] * cofactor_matrix[0,j] for j in range(3)]
    det = sum(det_terms)
    det_expansion = " + ".join([f"({matrix[0,j]}×{cofactor_matrix[0,j]:.4f})" for j in range(3)])
    steps.append(f"|A| = {det_expansion} = {det:.4f}")
    
    # Step 3: Calculate adjugate (transpose of cofactor matrix)
    adjugate = cofactor_matrix.T
    steps.append("\n3. Calculate adjugate (transpose of cofactor matrix):")
    steps.append(f"[{adjugate[0,0]:.4f} {adjugate[0,1]:.4f} {adjugate[0,2]:.4f}]")
    steps.append(f"[{adjugate[1,0]:.4f} {adjugate[1,1]:.4f} {adjugate[1,2]:.4f}]")
    steps.append(f"[{adjugate[2,0]:.4f} {adjugate[2,1]:.4f} {adjugate[2,2]:.4f}]")
    
    # Step 4: Divide by determinant
    inverse = adjugate / det
    steps.append("\n4. Divide adjugate by determinant to get inverse:")
    steps.append(f"A⁻¹ = (1/{det:.4f}) × adjugate =")
    steps.append(f"[{inverse[0,0]:.4f} {inverse[0,1]:.4f} {inverse[0,2]:.4f}]")
    steps.append(f"[{inverse[1,0]:.4f} {inverse[1,1]:.4f} {inverse[1,2]:.4f}]")
    steps.append(f"[{inverse[2,0]:.4f} {inverse[2,1]:.4f} {inverse[2,2]:.4f}]")
    
    # Step 5: Verify the inverse
    product = matrix @ inverse
    steps.append("\n5. Verify A×A⁻¹ = I:")
    steps.append("A×A⁻¹ =")
    steps.append(f"[{product[0,0]:.4f} {product[0,1]:.4f} {product[0,2]:.4f}]")
    steps.append(f"[{product[1,0]:.4f} {product[1,1]:.4f} {product[1,2]:.4f}]")
    steps.append(f"[{product[2,0]:.4f} {product[2,1]:.4f} {product[2,2]:.4f}]")
    
    return inverse, det, "\n".join(steps)

def calculate_covariance_detailed(matrix, vector1, vector2, name1="v1", name2="v2"):
    """Calculate covariance between two linear combinations with detailed steps."""
    steps = [f"Calculating covariance between {name1} and {name2}:"]
    
    # Step 1: Show the vectors
    steps.append("\n1. Input vectors:")
    steps.append(f"{name1} = [{vector1[0]:.4f} {vector1[1]:.4f} {vector1[2]:.4f}]")
    steps.append(f"{name2} = [{vector2[0]:.4f} {vector2[1]:.4f} {vector2[2]:.4f}]")
    
    # Step 2: Show the covariance matrix
    steps.append("\n2. Covariance matrix Σ:")
    steps.append(f"[{matrix[0,0]:.4f} {matrix[0,1]:.4f} {matrix[0,2]:.4f}]")
    steps.append(f"[{matrix[1,0]:.4f} {matrix[1,1]:.4f} {matrix[1,2]:.4f}]")
    steps.append(f"[{matrix[2,0]:.4f} {matrix[2,1]:.4f} {matrix[2,2]:.4f}]")
    
    # Step 3: Calculate v1ᵀΣ
    temp1 = vector1.T @ matrix
    steps.append("\n3. Calculate first multiplication v1ᵀΣ:")
    steps.append(f"[{vector1[0]:.4f} {vector1[1]:.4f} {vector1[2]:.4f}] × Σ =")
    steps.append(f"[{temp1[0]:.4f} {temp1[1]:.4f} {temp1[2]:.4f}]")
    
    # Step 4: Calculate final result (v1ᵀΣv2)
    result = temp1 @ vector2
    steps.append("\n4. Calculate final result (v1ᵀΣv2):")
    steps.append(f"[{temp1[0]:.4f} {temp1[1]:.4f} {temp1[2]:.4f}] × [{vector2[0]:.4f}]")
    steps.append(f"                                    [{vector2[1]:.4f}]")
    steps.append(f"                                    [{vector2[2]:.4f}]")
    steps.append(f"= {result:.4f}")
    
    return result, "\n".join(steps)

def calculate_conditional_covariance_detailed(Sigma, idx1, idx2, cond_idx):
    """Calculate conditional covariance with detailed steps."""
    steps = ["Calculating conditional covariance:"]
    
    # Step 1: Partition the covariance matrix
    steps.append("\n1. Partition the covariance matrix:")
    Sigma_11 = Sigma[np.ix_(idx1, idx1)]
    Sigma_12 = Sigma[np.ix_(idx1, cond_idx)]
    Sigma_21 = Sigma[np.ix_(cond_idx, idx1)]
    Sigma_22 = Sigma[np.ix_(cond_idx, cond_idx)]
    
    steps.append("Σ₁₁ (Covariance of variables of interest):")
    steps.append(f"[{Sigma_11[0,0]:.4f} {Sigma_11[0,1]:.4f}]")
    steps.append(f"[{Sigma_11[1,0]:.4f} {Sigma_11[1,1]:.4f}]")
    
    steps.append("\nΣ₁₂ (Cross-covariance):")
    steps.append(f"[{Sigma_12[0,0]:.4f}]")
    steps.append(f"[{Sigma_12[1,0]:.4f}]")
    
    steps.append("\nΣ₂₂ (Covariance of conditioning variable):")
    steps.append(f"[{Sigma_22[0,0]:.4f}]")
    
    # Step 2: Calculate Σ₂₂⁻¹
    Sigma_22_inv = 1/Sigma_22[0,0]
    steps.append("\n2. Calculate Σ₂₂⁻¹:")
    steps.append(f"Σ₂₂⁻¹ = 1/{Sigma_22[0,0]:.4f} = {Sigma_22_inv:.4f}")
    
    # Step 3: Calculate Σ₁₂Σ₂₂⁻¹Σ₂₁
    temp = Sigma_12 @ np.array([[Sigma_22_inv]]) @ Sigma_21
    steps.append("\n3. Calculate Σ₁₂Σ₂₂⁻¹Σ₂₁:")
    steps.append("First multiply Σ₁₂ × Σ₂₂⁻¹:")
    temp1 = Sigma_12 @ np.array([[Sigma_22_inv]])
    steps.append(f"[{Sigma_12[0,0]:.4f}] × [{Sigma_22_inv:.4f}] = [{temp1[0,0]:.4f}]")
    steps.append(f"[{Sigma_12[1,0]:.4f}]                = [{temp1[1,0]:.4f}]")
    
    steps.append("\nThen multiply by Σ₂₁:")
    steps.append(f"[{temp1[0,0]:.4f}] × [{Sigma_21[0,0]} {Sigma_21[0,1]}] =")
    steps.append(f"[{temp1[1,0]:.4f}]")
    steps.append(f"[{temp[0,0]:.4f} {temp[0,1]:.4f}]")
    steps.append(f"[{temp[1,0]:.4f} {temp[1,1]:.4f}]")
    
    # Step 4: Calculate conditional covariance
    Sigma_cond = Sigma_11 - temp
    steps.append("\n4. Calculate conditional covariance matrix:")
    steps.append("Σ₁|₂ = Σ₁₁ - Σ₁₂Σ₂₂⁻¹Σ₂₁")
    steps.append(f"= [{Sigma_11[0,0]:.4f} {Sigma_11[0,1]:.4f}] - [{temp[0,0]:.4f} {temp[0,1]:.4f}]")
    steps.append(f"  [{Sigma_11[1,0]:.4f} {Sigma_11[1,1]:.4f}]   [{temp[1,0]:.4f} {temp[1,1]:.4f}]")
    steps.append(f"= [{Sigma_cond[0,0]:.4f} {Sigma_cond[0,1]:.4f}]")
    steps.append(f"  [{Sigma_cond[1,0]:.4f} {Sigma_cond[1,1]:.4f}]")
    
    return Sigma_cond, "\n".join(steps)

def calculate_independence_conditions_detailed(Sigma, a, b):
    """Calculate independence conditions with detailed steps."""
    steps = ["Calculating independence conditions for Z = X₁ - aX₂ - bX₃:"]
    
    # Step 1: Write out vectors
    z_coef = np.array([1, -a, -b])
    x1_coef = np.array([1, 0, 0])
    
    # Calculate covariance with detailed steps
    cov, cov_steps = calculate_covariance_detailed(Sigma, z_coef, x1_coef, "Z", "X₁")
    steps.append("\nCovariance calculation:")
    steps.extend(cov_steps.split('\n'))
    
    # Add interpretation
    steps.append("\nInterpretation:")
    if abs(cov) < 1e-10:
        steps.append("Since Cov(Z,X₁) ≈ 0, Z and X₁ are independent")
    else:
        steps.append(f"Since Cov(Z,X₁) = {cov:.4f} ≠ 0, Z and X₁ are not independent")
        steps.append("\nTo achieve independence, we need:")
        steps.append("Cov(Z,X₁) = Var(X₁) - aCov(X₂,X₁) - bCov(X₃,X₁) = 0")
        steps.append(f"{Sigma[0,0]:.4f} - {a:.4f}({Sigma[1,0]:.4f}) - {b:.4f}({Sigma[2,0]:.4f}) = 0")
    
    return cov, "\n".join(steps)

def calculate_conditional_independence(Sigma, a, x3_value=None):
    """Calculate conditional covariance between Z and X₁ given X₃ with very detailed steps."""
    steps = ["Calculating conditional independence between Z and X₁ given X₃:"]
    
    # Step 1: Extract relevant submatrices
    steps.append("\n1. Partition the covariance matrix:")
    Sigma_11 = Sigma[:2,:2]  # Covariance of (X₁,X₂)
    Sigma_12 = Sigma[:2,2:3]  # Covariance between (X₁,X₂) and X₃
    Sigma_21 = Sigma[2:3,:2]  # Covariance between X₃ and (X₁,X₂)
    Sigma_22 = Sigma[2:3,2:3]  # Variance of X₃
    
    steps.append("Σ₁₁ (Covariance of X₁,X₂):")
    steps.append(f"[{Sigma_11[0,0]} {Sigma_11[0,1]}]")
    steps.append(f"[{Sigma_11[1,0]} {Sigma_11[1,1]}]")
    
    steps.append("\nΣ₁₂ (Covariance between (X₁,X₂) and X₃):")
    steps.append(f"[{Sigma_12[0,0]}]")
    steps.append(f"[{Sigma_12[1,0]}]")
    
    steps.append("\nΣ₂₂ (Variance of X₃):")
    steps.append(f"[{Sigma_22[0,0]}]")
    
    # Step 2: Calculate Σ₂₂⁻¹
    steps.append("\n2. Calculate Σ₂₂⁻¹:")
    Sigma_22_inv = 1/Sigma_22[0,0]
    steps.append(f"Σ₂₂⁻¹ = 1/{Sigma_22[0,0]} = {Sigma_22_inv:.4f}")
    
    # Step 3: Calculate Σ₁₂Σ₂₂⁻¹Σ₂₁
    steps.append("\n3. Calculate Σ₁₂Σ₂₂⁻¹Σ₂₁:")
    temp1 = Sigma_12 @ np.array([[Sigma_22_inv]])
    steps.append("First multiply Σ₁₂Σ₂₂⁻¹:")
    steps.append(f"[{Sigma_12[0,0]}]·[{Sigma_22_inv:.4f}] = [{temp1[0,0]:.4f}]")
    steps.append(f"[{Sigma_12[1,0]}]                = [{temp1[1,0]:.4f}]")
    
    temp2 = temp1 @ Sigma_21
    steps.append("\nThen multiply by Σ₂₁:")
    steps.append(f"[{temp1[0,0]:.4f}]·[{Sigma_21[0,0]} {Sigma_21[0,1]}] =")
    steps.append(f"[{temp1[1,0]:.4f}]")
    steps.append(f"[{temp2[0,0]:.4f} {temp2[0,1]:.4f}]")
    steps.append(f"[{temp2[1,0]:.4f} {temp2[1,1]:.4f}]")
    
    # Step 4: Calculate conditional covariance matrix
    steps.append("\n4. Calculate conditional covariance matrix:")
    steps.append("Σ₁₁|₃ = Σ₁₁ - Σ₁₂Σ₂₂⁻¹Σ₂₁")
    Sigma_cond = Sigma_11 - temp2
    steps.append(f"= [   {Sigma_11[0,0]} {Sigma_11[0,1]}   ] - [   {temp2[0,0]:.4f} {temp2[0,1]:.4f}   ]")
    steps.append(f"  [   {Sigma_11[1,0]} {Sigma_11[1,1]}   ]   [   {temp2[1,0]:.4f} {temp2[1,1]:.4f}   ]")
    steps.append(f"= [   {Sigma_cond[0,0]:.4f} {Sigma_cond[0,1]:.4f}   ]")
    steps.append(f"  [   {Sigma_cond[1,0]:.4f} {Sigma_cond[1,1]:.4f}   ]")
    
    # Step 5: Calculate conditional covariance between Z and X₁
    steps.append("\n5. Calculate conditional covariance between Z and X₁:")
    z_coef = np.array([1, -a])  # Coefficients for Z in terms of X₁,X₂
    x1_coef = np.array([1, 0])  # Coefficients for X₁
    
    steps.append("For Z = X₁ - aX₂, calculate Cov(Z,X₁|X₃):")
    steps.append(f"Using vector form: [1 {-a:.4f}]·Σ₁₁|₃·[1]")
    steps.append("                                    [0]")
    
    temp3 = z_coef.T @ Sigma_cond
    steps.append("\nFirst multiply [1 -a]·Σ₁₁|₃:")
    steps.append(f"[1 {-a:.4f}]·[{Sigma_cond[0,0]:.4f} {Sigma_cond[0,1]:.4f}]")
    steps.append(f"          [{Sigma_cond[1,0]:.4f} {Sigma_cond[1,1]:.4f}]")
    steps.append(f"= [{temp3[0]:.4f} {temp3[1]:.4f}]")
    
    cond_cov = temp3 @ x1_coef
    steps.append("\nThen multiply by [1 0]ᵀ:")
    steps.append(f"[{temp3[0]:.4f} {temp3[1]:.4f}]·[1] = {cond_cov:.4f}")
    steps.append("                           [0]")
    
    # Step 6: Interpret the result
    steps.append("\n6. Interpretation:")
    if abs(cond_cov) < 1e-10:
        steps.append("Since conditional covariance ≈ 0, Z and X₁ are conditionally independent given X₃")
    else:
        steps.append(f"Since conditional covariance = {cond_cov:.4f} ≠ 0, Z and X₁ are not conditionally independent given X₃")
    
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
    
    # Plot independent pair (X₂,X₃)
    plot_2d_gaussian(mu[[1,2]], Sigma[1:3,1:3][[0,1]][:,[0,1]], 
                    'Joint Distribution of X₂ and X₃\n(Independent Variables)',
                    'example1_independent_pair',
                    x_label='X₂', y_label='X₃')
    
    # Plot dependent pair (X₁,X₃) before conditioning
    plot_2d_gaussian(mu[[0,2]], np.array([[Sigma[0,0], Sigma[0,2]], [Sigma[2,0], Sigma[2,2]]]),
                    'Joint Distribution of X₁ and X₃\n(Dependent Variables)',
                    'example1_dependent_pair',
                    x_label='X₁', y_label='X₃')
    
    # Plot conditional distribution of (X₁,X₃) given X₂=2
    plot_2d_gaussian(mu[[0,2]], Sigma_cond,
                    'Conditional Distribution of X₁ and X₃ given X₂=2\n(Still Dependent)',
                    'example1_conditional',
                    x_label='X₁', y_label='X₃',
                    show_conditional=True,
                    conditional_x=2)

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
    
    # Calculate correlations for each pair
    print("\nCalculating correlations:")
    for (i,j), name in zip(pairs, pair_names):
        cov = Sigma[i,j]
        var_i = Sigma[i,i]
        var_j = Sigma[j,j]
        corr = cov / np.sqrt(var_i * var_j)
        print(f"\nPair {name}:")
        print(f"Cov{name} = Σ[{i},{j}] = {cov}")
        print(f"Var(X_{i+1}) = Σ[{i},{i}] = {var_i}")
        print(f"Var(X_{j+1}) = Σ[{j},{j}] = {var_j}")
        print(f"Correlation = {cov}/√({var_i}×{var_j}) = {corr:.4f}")
        print(f"→ Variables are {'independent' if cov == 0 else 'not independent'}")
    
    print_step(3, "Calculate covariance between Z = 3X₁ - 6X₃ and X₂")
    # Define coefficients for Z = 3X₁ - 6X₃
    z_coef = np.array([3, 0, -6])
    x2_coef = np.array([0, 1, 0])
    
    # Calculate covariance using matrix operations
    cov_z_x2 = z_coef.T @ Sigma @ x2_coef
    
    # Show detailed calculation steps
    print("\nDetailed calculation of Cov(Z,X₂):")
    print("Z = 3X₁ - 6X₃")
    print("Cov(Z,X₂) = Cov(3X₁ - 6X₃, X₂)")
    print("          = 3Cov(X₁,X₂) - 6Cov(X₃,X₂)")
    print(f"          = 3·({Sigma[0,1]}) - 6·({Sigma[2,1]})")
    print(f"          = {3 * Sigma[0,1]} - {6 * Sigma[2,1]}")
    print(f"          = {cov_z_x2}")
    
    # Calculate correlation between Z and X₂
    var_z = z_coef.T @ Sigma @ z_coef
    var_x2 = Sigma[1,1]
    corr_z_x2 = cov_z_x2 / np.sqrt(var_z * var_x2)
    
    print("\nCalculating correlation between Z and X₂:")
    print(f"Var(Z) = Var(3X₁ - 6X₃)")
    print(f"       = 9Var(X₁) + 36Var(X₃) - 36Cov(X₁,X₃)")
    print(f"       = 9·({Sigma[0,0]}) + 36·({Sigma[2,2]}) - 36·({Sigma[0,2]})")
    print(f"       = {var_z}")
    print(f"Var(X₂) = {var_x2}")
    print(f"Correlation = {cov_z_x2}/√({var_z}×{var_x2}) = {corr_z_x2:.4f}")
    
    print_step(4, "Calculate conditional independence of X₁ and X₃ given X₂")
    print("\nStep-by-step calculation of conditional covariance matrix:")
    
    # Step 1: Partition the covariance matrix
    print("\n1. Partition the covariance matrix:")
    Sigma_11 = np.array([[Sigma[0,0], Sigma[0,2]],
                        [Sigma[2,0], Sigma[2,2]]])  # Covariance of (X₁,X₃)
    print_matrix("Σ₁₁ (Covariance of X₁,X₃)", Sigma_11)
    
    Sigma_12 = np.array([[Sigma[0,1]],
                        [Sigma[2,1]]])  # Covariance between (X₁,X₃) and X₂
    print_matrix("Σ₁₂ (Covariance between (X₁,X₃) and X₂)", Sigma_12)
    
    Sigma_22 = np.array([[Sigma[1,1]]])  # Variance of X₂
    print_matrix("Σ₂₂ (Variance of X₂)", Sigma_22)
    
    # Step 2: Calculate inverse of Sigma_22
    print("\n2. Calculate Σ₂₂⁻¹:")
    Sigma_22_inv = 1/Sigma[1,1]  # For 1x1 matrix, inverse is reciprocal
    print(f"Σ₂₂⁻¹ = 1/{Sigma[1,1]} = {Sigma_22_inv}")
    
    # Step 3: Calculate Sigma_12 @ Sigma_22_inv @ Sigma_12.T
    print("\n3. Calculate Σ₁₂Σ₂₂⁻¹Σ₂₁:")
    temp = Sigma_12 @ np.array([[Sigma_22_inv]]) @ Sigma_12.T
    print_matrix("Σ₁₂Σ₂₂⁻¹Σ₂₁", temp)
    
    # Step 4: Calculate conditional covariance matrix
    print("\n4. Calculate conditional covariance matrix:")
    print("Σ₁₁|₂ = Σ₁₁ - Σ₁₂Σ₂₂⁻¹Σ₂₁")
    Sigma_cond = Sigma_11 - temp
    print_matrix("Σ₁₁|₂", Sigma_cond)
    
    # Calculate conditional correlation
    cond_cov = Sigma_cond[0,1]
    cond_var1 = Sigma_cond[0,0]
    cond_var2 = Sigma_cond[1,1]
    cond_corr = cond_cov / np.sqrt(cond_var1 * cond_var2)
    
    print("\n5. Calculate conditional correlation:")
    print(f"Conditional covariance = {cond_cov}")
    print(f"Conditional variance of X₁|X₂ = {cond_var1}")
    print(f"Conditional variance of X₃|X₂ = {cond_var2}")
    print(f"Conditional correlation = {cond_cov}/√({cond_var1}×{cond_var2}) = {cond_corr:.4f}")
    
    print("\nSince the conditional covariance is not 0, X₁ and X₃ are not conditionally independent given X₂")
    
    # Plot distributions
    plot_2d_gaussian(mu[[0,1]], Sigma[:2,:2],
                    'Original Joint Distribution of X₁ and X₂\nCorrelated Variables',
                    'example3_original')
    
    # Plot conditional distribution
    x2_value = 2  # Conditioning on X₂ = 2
    # Calculate conditional mean
    mu_cond = np.array([mu[0], mu[2]]) + Sigma_12 @ np.array([[Sigma_22_inv]]) @ np.array([[x2_value - mu[1]]])
    mu_cond = mu_cond.flatten()  # Convert to 1D array of length 2
    
    print("\n6. Calculate conditional mean:")
    print(f"μ₁|₂ = μ₁ + Σ₁₂Σ₂₂⁻¹(x₂ - μ₂)")
    print(f"    = [{mu[0]}] + [{Sigma_12[0,0]}] · {Sigma_22_inv} · ({x2_value} - {mu[1]})")
    print(f"    = {mu_cond[0]}")
    print(f"μ₃|₂ = μ₃ + Σ₃₂Σ₂₂⁻¹(x₂ - μ₂)")
    print(f"    = [{mu[2]}] + [{Sigma_12[1,0]}] · {Sigma_22_inv} · ({x2_value} - {mu[1]})")
    print(f"    = {mu_cond[1]}")
    
    # Plot conditional distribution
    plot_2d_gaussian(mu_cond[:2], Sigma_cond[:2,:2],  # Ensure both mean and covariance are 2D
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
    
    print_step(2, "Calculate marginal distribution of X₂ = [X₁, X₃]")
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
    
    # Calculate PDF components
    print("\nPDF calculation components:")
    print("f(x₂) = (2π)^(-n/2)|Σ₂|^(-1/2)exp(-½(x₂-μ₂)ᵀΣ₂⁻¹(x₂-μ₂))")
    print(f"n = 2 (dimension)")
    print(f"|Σ₂| = {det_Sigma_2}")
    print(f"(2π)^(-n/2) = {(2*np.pi)**(-1):.6f}")
    print(f"|Σ₂|^(-1/2) = {det_Sigma_2**(-0.5):.6f}")
    print("Normalizing constant = (2π)^(-n/2)|Σ₂|^(-1/2) = " + 
          f"{((2*np.pi)**(-1) * det_Sigma_2**(-0.5)):.6f}")
    
    print_step(3, "Calculate covariance matrix of Y = [X₁, X₂, X₃, X₄, X₅]")
    # Reorder variables according to Y = [X₂, X₄, X₅, X₁, X₃]
    idx_Y = [1, 3, 4, 0, 2]
    mu_Y = mu[idx_Y]
    Sigma_Y = Sigma[np.ix_(idx_Y, idx_Y)]
    
    print("\nReordered parameters for Y:")
    print_matrix("Mean vector μY", mu_Y)
    print_matrix("Covariance matrix ΣY", Sigma_Y)
    
    print("\nVerifying properties of reordered covariance matrix:")
    print("1. Symmetry check:")
    is_symmetric = np.allclose(Sigma_Y, Sigma_Y.T)
    print(f"   Is symmetric: {is_symmetric}")
    
    print("2. Positive definiteness check:")
    eigenvals = np.linalg.eigvals(Sigma_Y)
    print("   Eigenvalues:")
    for i, ev in enumerate(eigenvals):
        print(f"   λ{i+1} = {ev:.4f}")
    is_pos_def = np.all(eigenvals > 0)
    print(f"   Is positive definite: {is_pos_def}")
    
    print_step(4, "Calculate conditional distribution of X₁ given X₂ = [6, 24]")
    x2_values = np.array([6, 24])
    idx_X1 = [1, 3]  # Indices for X₂, X₄
    idx_cond = [0, 2]   # Indices for X₁, X₃
    
    # Calculate conditional distribution with detailed steps
    cond_dist = calculate_conditional_distribution_steps(mu, Sigma, idx_X1, idx_cond, x2_values)
    
    print("\nStep-by-step conditional distribution calculation:")
    print("\n1. Partition the covariance matrix:")
    print("   Σ₁₁ (covariance of X₂,X₄):")
    print_matrix("   ", Sigma[np.ix_(idx_X1, idx_X1)])
    print("   Σ₁₂ (covariance between (X₂,X₄) and (X₁,X₃)):")
    print_matrix("   ", Sigma[np.ix_(idx_X1, idx_cond)])
    print("   Σ₂₂ (covariance of X₁,X₃):")
    print_matrix("   ", Sigma[np.ix_(idx_cond, idx_cond)])
    
    print("\n2. Calculate Σ₂₂⁻¹:")
    print(cond_dist['steps']['inverse'])
    
    print("\n3. Calculate Σ₁₂Σ₂₂⁻¹:")
    print(cond_dist['steps']['S12_S22inv'])
    
    print("\n4. Calculate conditional mean:")
    print(cond_dist['steps']['mu_cond'])
    
    print("\n5. Calculate conditional covariance:")
    print(cond_dist['steps']['Sigma_cond'])
    
    # Calculate correlation in conditional distribution
    cond_cov = cond_dist['Sigma_cond'][0,1]
    cond_var1 = cond_dist['Sigma_cond'][0,0]
    cond_var2 = cond_dist['Sigma_cond'][1,1]
    cond_corr = cond_cov / np.sqrt(cond_var1 * cond_var2)
    
    print("\n6. Calculate conditional correlation:")
    print(f"   Conditional covariance = {cond_cov:.4f}")
    print(f"   Conditional variance of X₂|X₁,X₃ = {cond_var1:.4f}")
    print(f"   Conditional variance of X₄|X₁,X₃ = {cond_var2:.4f}")
    print(f"   Conditional correlation = {cond_cov}/√({cond_var1}×{cond_var2}) = {cond_corr:.4f}")
    
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
    
    print("\nGeometric interpretation:")
    print("1. The original joint distribution shows the unconstrained relationship")
    print("   between X₂ and X₄.")
    print("2. Conditioning on X₁ = 6 and X₃ = 24 creates a slice through the")
    print("   5-dimensional distribution, resulting in a new bivariate normal")
    print("   distribution with adjusted mean and covariance.")
    print("3. The conditional correlation coefficient of {:.4f} indicates".format(cond_corr))
    print("   a moderate positive relationship between X₂ and X₄ even after")
    print("   conditioning on X₁ and X₃.")

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
    
    print_step(2, "Calculate inverse of covariance matrix")
    # Calculate inverse with detailed steps
    Sigma_inv, det, inv_steps = calculate_3x3_inverse(Sigma)
    print("\nDetailed inverse calculation:")
    print(inv_steps)
    
    print_step(3, "Check independence between pairs")
    pairs = [(0,1), (1,2), (0,2)]
    pair_names = ["(X₁,X₂)", "(X₂,X₃)", "(X₁,X₃)"]
    
    for (i,j), name in zip(pairs, pair_names):
        # Create vectors for each pair
        v1 = np.zeros(3)
        v2 = np.zeros(3)
        v1[i] = 1
        v2[j] = 1
        
        # Calculate covariance with detailed steps
        cov, cov_steps = calculate_covariance_detailed(Sigma, v1, v2, f"X_{i+1}", f"X_{j+1}")
        print(f"\nChecking independence for {name}:")
        print(cov_steps)
        
        # Calculate correlation
        var1 = Sigma[i,i]
        var2 = Sigma[j,j]
        corr = cov / np.sqrt(var1 * var2)
        print(f"\nCorrelation calculation:")
        print(f"ρ = Cov(X_{i+1},X_{j+1})/√(Var(X_{i+1})·Var(X_{j+1}))")
        print(f"  = {cov:.4f}/√({var1:.4f}·{var2:.4f})")
        print(f"  = {corr:.4f}")
        print(f"→ Variables are {'independent' if abs(cov) < 1e-10 else 'not independent'}")
    
    print_step(4, "Find values of a and b for Z = X₁ - aX₂ - bX₃ to be independent of X₁")
    # First try with b = 0
    b = 0
    a = 4 + b  # This comes from solving Cov(Z,X₁) = 0
    
    # Calculate covariance with detailed steps
    cov_z_x1, cov_steps = calculate_independence_conditions_detailed(Sigma, a, b)
    print("\nDetailed independence calculation for b = 0:")
    print(cov_steps)
    
    # Try another value of b to show multiple solutions
    b = 1
    a = 4 + b
    cov_z_x1, cov_steps = calculate_independence_conditions_detailed(Sigma, a, b)
    print("\nDetailed independence calculation for b = 1:")
    print(cov_steps)
    
    print_step(5, "Check conditional independence given X₃")
    # Calculate conditional independence with a = 3
    a_cond = 3
    idx1 = [0, 1]  # Indices for X₁,X₂
    cond_idx = [2]  # Index for X₃
    
    # Calculate conditional covariance with detailed steps
    Sigma_cond, cond_steps = calculate_conditional_covariance_detailed(Sigma, idx1, idx1, cond_idx)
    print("\nDetailed conditional covariance calculation:")
    print(cond_steps)
    
    # Calculate conditional covariance between Z and X₁
    z_coef = np.array([1, -a_cond])
    x1_coef = np.array([1, 0])
    cond_cov = z_coef.T @ Sigma_cond @ x1_coef
    
    print("\nConditional covariance between Z and X₁:")
    print(f"Cov(Z,X₁|X₃) = Cov(X₁ - {a_cond}X₂,X₁|X₃)")
    print(f"              = Var(X₁|X₃) - {a_cond}·Cov(X₂,X₁|X₃)")
    print(f"              = {Sigma_cond[0,0]:.4f} - {a_cond}·{Sigma_cond[0,1]:.4f}")
    print(f"              = {cond_cov:.4f}")
    
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
    
    print("\nGeometric interpretation:")
    print("1. The original joint distribution shows correlation between X₁ and X₂.")
    print("2. Conditioning on X₃ = -2 changes the correlation structure.")
    print("3. With a = 3, we can create a linear combination Z = X₁ - 3X₂")
    print("   that is conditionally independent of X₁ given X₃.")
    print("4. The plots visualize how conditioning affects the joint distribution.")

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