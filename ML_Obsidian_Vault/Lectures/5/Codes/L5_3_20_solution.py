import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import eigvals
import os
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
import warnings
warnings.filterwarnings('ignore')

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_3_Quiz_20")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'

print("=" * 80)
print("QUESTION 20: KERNEL VALIDITY TESTING - DETAILED CALCULATIONS")
print("=" * 80)

# ============================================================================
# TASK 1: Check validity of specific kernels with detailed calculations
# ============================================================================
print("\n" + "="*60)
print("TASK 1: Checking Validity of Specific Kernels")
print("="*60)

def check_kernel_validity_detailed(kernel_func, name, test_points):
    """
    Check if a kernel function is valid by computing its Gram matrix
    and checking if it's positive semi-definite (PSD) with detailed steps
    """
    n = len(test_points)
    K = np.zeros((n, n))
    
    print(f"\n{name}")
    print("-" * 50)
    
    # Step 1: Compute individual kernel values
    print("Step 1: Computing individual kernel values")
    print("K(x_i, x_j) for all pairs:")
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel_func(test_points[i], test_points[j])
            print(f"  K(x_{i+1}, x_{j+1}) = K({test_points[i]}, {test_points[j]}) = {K[i, j]:.6f}")
    
    # Step 2: Display the complete Gram matrix
    print(f"\nStep 2: Complete Gram Matrix K")
    print("K = ")
    for i in range(n):
        row_str = "  ["
        for j in range(n):
            row_str += f"{K[i, j]:8.6f}"
            if j < n-1:
                row_str += ", "
        row_str += "]"
        print(row_str)
    
    # Step 3: Check symmetry
    print(f"\nStep 3: Checking symmetry")
    is_symmetric = np.allclose(K, K.T)
    print(f"K = K^T: {'✓ YES' if is_symmetric else '✗ NO'}")
    if not is_symmetric:
        print("Matrix is not symmetric, which violates kernel properties!")
    
    # Step 4: Compute eigenvalues
    print(f"\nStep 4: Computing eigenvalues")
    eigenvalues = eigvals(K)
    print(f"Eigenvalues of K: {eigenvalues}")
    
    # Step 5: Check PSD property
    print(f"\nStep 5: Checking Positive Semi-Definiteness")
    min_eigenvalue = np.min(eigenvalues)
    print(f"Minimum eigenvalue: {min_eigenvalue:.6f}")
    is_psd = np.all(eigenvalues >= -1e-10)  # Small tolerance for numerical errors
    print(f"All eigenvalues ≥ 0: {'✓ YES' if is_psd else '✗ NO'}")
    
    # Step 6: Final conclusion
    print(f"\nStep 6: Conclusion")
    is_valid = is_symmetric and is_psd
    print(f"Valid kernel: {'✓ YES' if is_valid else '✗ NO'}")
    
    if is_valid:
        print("Reason: Matrix is symmetric and positive semi-definite")
    else:
        if not is_symmetric:
            print("Reason: Matrix is not symmetric")
        if not is_psd:
            print("Reason: Matrix has negative eigenvalues")
    
    return K, eigenvalues, is_valid

# Test points for kernel evaluation
test_points = np.array([
    [0, 0],    # x₁
    [1, 0],    # x₂
    [0, 1],    # x₃
    [1, 1],    # x₄
    [-1, 0]    # x₅
])

print(f"Test points:")
for i, point in enumerate(test_points):
    print(f"  x_{i+1} = {point}")

# 1.1 K(x,z) = (x^T z)^2 + (x^T z)^3
print(f"\n" + "="*60)
print("1.1 K(x,z) = (x^T z)^2 + (x^T z)^3")
print("="*60)

def kernel_1(x, z):
    dot_product = np.dot(x, z)
    return dot_product**2 + dot_product**3

print("Mathematical analysis:")
print("K(x,z) = (x^T z)^2 + (x^T z)^3")
print("This is a polynomial kernel combining quadratic and cubic terms.")
print("Since both (x^T z)^2 and (x^T z)^3 are valid polynomial kernels,")
print("their sum with positive coefficients should also be valid.")

K1, eig1, valid1 = check_kernel_validity_detailed(kernel_1, 
                                                "K(x,z) = (x^T z)^2 + (x^T z)^3", 
                                                test_points)

# 1.2 K(x,z) = exp(x^T z)
print(f"\n" + "="*60)
print("1.2 K(x,z) = exp(x^T z)")
print("="*60)

def kernel_2(x, z):
    return np.exp(np.dot(x, z))

print("Mathematical analysis:")
print("K(x,z) = exp(x^T z)")
print("This is the exponential kernel.")
print("It can be written as: exp(x^T z) = Σ_{k=0}^∞ (x^T z)^k / k!")
print("This is an infinite sum of polynomial kernels with positive coefficients,")
print("so it should be a valid kernel.")

K2, eig2, valid2 = check_kernel_validity_detailed(kernel_2, 
                                                "K(x,z) = exp(x^T z)", 
                                                test_points)

# 1.3 K(x,z) = sin(x^T z)
print(f"\n" + "="*60)
print("1.3 K(x,z) = sin(x^T z)")
print("="*60)

def kernel_3(x, z):
    return np.sin(np.dot(x, z))

print("Mathematical analysis:")
print("K(x,z) = sin(x^T z)")
print("This is the sine kernel.")
print("The sine function oscillates between -1 and 1,")
print("and can produce negative values, which may violate")
print("the positive semi-definiteness requirement.")

K3, eig3, valid3 = check_kernel_validity_detailed(kernel_3, 
                                                "K(x,z) = sin(x^T z)", 
                                                test_points)

# ============================================================================
# TASK 2: Compute Gram matrices for specific 3 points with detailed steps
# ============================================================================
print("\n" + "="*60)
print("TASK 2: Gram Matrices for Points (0,0), (1,0), (0,1)")
print("="*60)

specific_points = np.array([[0, 0], [1, 0], [0, 1]])
print(f"Points: x₁ = {specific_points[0]}, x₂ = {specific_points[1]}, x₃ = {specific_points[2]}")

def compute_gram_matrix_detailed(kernel_func, points, kernel_name):
    """Compute Gram matrix with detailed steps"""
    n = len(points)
    K = np.zeros((n, n))
    
    print(f"\n{kernel_name}")
    print("-" * 40)
    
    # Compute each element with explanation
    for i in range(n):
        for j in range(n):
            x_i = points[i]
            x_j = points[j]
            k_val = kernel_func(x_i, x_j)
            K[i, j] = k_val
            
            # Detailed calculation for specific cases
            if kernel_name == "Polynomial Kernel":
                dot_prod = np.dot(x_i, x_j)
                print(f"K(x_{i+1}, x_{j+1}) = ({dot_prod})² + ({dot_prod})³ = {dot_prod**2} + {dot_prod**3} = {k_val}")
            elif kernel_name == "Exponential Kernel":
                dot_prod = np.dot(x_i, x_j)
                print(f"K(x_{i+1}, x_{j+1}) = exp({dot_prod}) = {k_val}")
            elif kernel_name == "Sine Kernel":
                dot_prod = np.dot(x_i, x_j)
                print(f"K(x_{i+1}, x_{j+1}) = sin({dot_prod}) = {k_val}")
    
    print(f"\nGram Matrix:")
    for i in range(n):
        row_str = "  ["
        for j in range(n):
            row_str += f"{K[i, j]:8.6f}"
            if j < n-1:
                row_str += ", "
        row_str += "]"
        print(row_str)
    
    # Check eigenvalues
    eigenvalues = eigvals(K)
    print(f"Eigenvalues: {eigenvalues}")
    is_psd = np.all(eigenvalues >= -1e-10)
    print(f"PSD: {'✓ YES' if is_psd else '✗ NO'}")
    
    return K, eigenvalues

# Compute Gram matrices for all three kernels
K1_specific, eig1_spec = compute_gram_matrix_detailed(kernel_1, specific_points, "Polynomial Kernel")
K2_specific, eig2_spec = compute_gram_matrix_detailed(kernel_2, specific_points, "Exponential Kernel")
K3_specific, eig3_spec = compute_gram_matrix_detailed(kernel_3, specific_points, "Sine Kernel")

# ============================================================================
# TASK 3: Show linear combination of valid kernels is valid
# ============================================================================
print("\n" + "="*60)
print("TASK 3: Linear Combination of Valid Kernels")
print("="*60)

# Define two valid kernels (linear and RBF)
def K1_valid(x, z):
    return np.dot(x, z)  # Linear kernel

def K2_valid(x, z):
    return np.exp(-0.1 * np.sum((np.array(x) - np.array(z))**2))  # RBF kernel

# Linear combination: K = 2*K1 + 3*K2
def K_combined(x, z):
    return 2 * K1_valid(x, z) + 3 * K2_valid(x, z)

print("Mathematical analysis:")
print("K(x,z) = 2K₁(x,z) + 3K₂(x,z)")
print("where K₁(x,z) = x^T z (linear kernel)")
print("and K₂(x,z) = exp(-0.1||x-z||²) (RBF kernel)")
print("\nTheoretical justification:")
print("1. If K₁ and K₂ are valid kernels, their Gram matrices K₁ and K₂ are PSD")
print("2. For any positive constants a, b > 0, aK₁ + bK₂ is also PSD")
print("3. This follows from: (aK₁ + bK₂)ᵀ = aK₁ᵀ + bK₂ᵀ = aK₁ + bK₂ (symmetry)")
print("   and all eigenvalues of aK₁ + bK₂ are non-negative (PSD)")

# Check validity of individual kernels
print(f"\nChecking individual kernels:")
K1_individual, eig1_ind, valid1_ind = check_kernel_validity_detailed(K1_valid, 
                                                                    "K₁(x,z) = x^T z (Linear)", 
                                                                    test_points)

K2_individual, eig2_ind, valid2_ind = check_kernel_validity_detailed(K2_valid, 
                                                                    "K₂(x,z) = exp(-0.1||x-z||²) (RBF)", 
                                                                    test_points)

# Check validity of combined kernel
print(f"\nChecking combined kernel:")
K_comb, eig_comb, valid_comb = check_kernel_validity_detailed(K_combined, 
                                                             "K(x,z) = 2K₁(x,z) + 3K₂(x,z)", 
                                                             test_points)

# Verification: K_combined = 2*K₁ + 3*K₂
print(f"\nVerification: K_combined = 2*K₁ + 3*K₂")
print("Computing 2*K₁ + 3*K₂:")
K_expected = 2*K1_individual + 3*K2_individual
print("Expected matrix:")
for i in range(K_expected.shape[0]):
    row_str = "  ["
    for j in range(K_expected.shape[1]):
        row_str += f"{K_expected[i, j]:8.6f}"
        if j < K_expected.shape[1]-1:
            row_str += ", "
    row_str += "]"
    print(row_str)

print(f"Matrices are equal: {np.allclose(K_comb, K_expected)}")

# ============================================================================
# TASK 4: Example of invalid kernel with detailed analysis
# ============================================================================
print("\n" + "="*60)
print("TASK 4: Example of Invalid Kernel")
print("="*60)

# Invalid kernel: K(x,z) = -||x-z||^2 (negative squared distance)
def invalid_kernel(x, z):
    return -np.sum((np.array(x) - np.array(z))**2)

print("4.1 K(x,z) = -||x-z||²")
print("Mathematical analysis:")
print("This kernel is the negative of the squared Euclidean distance.")
print("While ||x-z||² is a valid kernel (it's the negative of the RBF kernel),")
print("the negative sign makes it invalid because it violates PSD property.")

K_invalid, eig_invalid, valid_invalid = check_kernel_validity_detailed(invalid_kernel, 
                                                                      "K(x,z) = -||x-z||² (Invalid)", 
                                                                      test_points)

# Another invalid kernel: K(x,z) = x[0] * z[1] (asymmetric)
def invalid_kernel2(x, z):
    return x[0] * z[1]

print(f"\n4.2 K(x,z) = x₀ * z₁ (Asymmetric)")
print("Mathematical analysis:")
print("This kernel is asymmetric: K(x,z) ≠ K(z,x)")
print("For example: K([1,0], [0,1]) = 1*1 = 1")
print("but K([0,1], [1,0]) = 0*0 = 0")
print("This violates the symmetry requirement for valid kernels.")

K_invalid2, eig_invalid2, valid_invalid2 = check_kernel_validity_detailed(invalid_kernel2, 
                                                                         "K(x,z) = x₀ * z₁ (Asymmetric)", 
                                                                         test_points)

# ============================================================================
# TASK 5: Design kernel for comparing sets of different sizes
# ============================================================================
print("\n" + "="*60)
print("TASK 5: Kernel for Comparing Sets of Different Sizes")
print("="*60)

def set_kernel(set_A, set_B, base_kernel='rbf', gamma=1.0):
    """
    Design a valid kernel for comparing sets of different sizes.
    Uses the average pairwise kernel between all elements.
    """
    if len(set_A) == 0 or len(set_B) == 0:
        return 0.0
    
    total_kernel = 0.0
    count = 0
    
    for a in set_A:
        for b in set_B:
            if base_kernel == 'rbf':
                k_val = np.exp(-gamma * np.sum((np.array(a) - np.array(b))**2))
            elif base_kernel == 'linear':
                k_val = np.dot(a, b)
            elif base_kernel == 'poly':
                k_val = (np.dot(a, b) + 1)**2
            else:
                k_val = np.dot(a, b)
            
            total_kernel += k_val
            count += 1
    
    return total_kernel / count

# Test sets of different sizes
set_A = [[0, 0], [1, 0]]
set_B = [[0, 1], [1, 1], [0.5, 0.5]]
set_C = [[-1, 0]]

print(f"Set A: {set_A} (|A| = {len(set_A)})")
print(f"Set B: {set_B} (|B| = {len(set_B)})")
print(f"Set C: {set_C} (|C| = {len(set_C)})")

print(f"\nSet Kernel Definition:")
print("K(A,B) = (1/|A|·|B|) Σ_{a∈A} Σ_{b∈B} k(a,b)")
print("where k(a,b) is a base kernel (RBF or linear)")

# Compute set kernels with detailed steps
print(f"\nComputing set kernels with RBF base kernel:")
print("k(a,b) = exp(-γ||a-b||²) with γ = 1.0")

K_AB_rbf = set_kernel(set_A, set_B, 'rbf')
K_AC_rbf = set_kernel(set_A, set_C, 'rbf')
K_BC_rbf = set_kernel(set_B, set_C, 'rbf')

print(f"K(A,B) = {K_AB_rbf:.4f}")
print(f"K(A,C) = {K_AC_rbf:.4f}")
print(f"K(B,C) = {K_BC_rbf:.4f}")

print(f"\nComputing set kernels with linear base kernel:")
print("k(a,b) = a^T b")

K_AB_linear = set_kernel(set_A, set_B, 'linear')
K_AC_linear = set_kernel(set_A, set_C, 'linear')
K_BC_linear = set_kernel(set_B, set_C, 'linear')

print(f"K(A,B) = {K_AB_linear:.4f}")
print(f"K(A,C) = {K_AC_linear:.4f}")
print(f"K(B,C) = {K_BC_linear:.4f}")

# Check validity by creating Gram matrix
sets = [set_A, set_B, set_C]
n_sets = len(sets)

K_set_rbf = np.zeros((n_sets, n_sets))
K_set_linear = np.zeros((n_sets, n_sets))

for i in range(n_sets):
    for j in range(n_sets):
        K_set_rbf[i, j] = set_kernel(sets[i], sets[j], 'rbf')
        K_set_linear[i, j] = set_kernel(sets[i], sets[j], 'linear')

print(f"\nSet Kernel Gram Matrix (RBF):")
for i in range(n_sets):
    row_str = "  ["
    for j in range(n_sets):
        row_str += f"{K_set_rbf[i, j]:8.6f}"
        if j < n_sets-1:
            row_str += ", "
    row_str += "]"
    print(row_str)

eig_set_rbf = eigvals(K_set_rbf)
print(f"Eigenvalues: {eig_set_rbf}")
print(f"PSD: {'✓ YES' if np.all(eig_set_rbf >= -1e-10) else '✗ NO'}")

print(f"\nSet Kernel Gram Matrix (Linear):")
for i in range(n_sets):
    row_str = "  ["
    for j in range(n_sets):
        row_str += f"{K_set_linear[i, j]:8.6f}"
        if j < n_sets-1:
            row_str += ", "
    row_str += "]"
    print(row_str)

eig_set_linear = eigvals(K_set_linear)
print(f"Eigenvalues: {eig_set_linear}")
print(f"PSD: {'✓ YES' if np.all(eig_set_linear >= -1e-10) else '✗ NO'}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

# Create a comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(r'Kernel Validity Analysis - Detailed Calculations', fontsize=16, fontweight='bold')

# 1. Eigenvalue plots for the three main kernels
eigenvalues_data = [eig1, eig2, eig3]
kernel_names = [r'$K(\mathbf{x},\mathbf{z}) = (\mathbf{x}^T\mathbf{z})^2 + (\mathbf{x}^T\mathbf{z})^3$', 
                r'$K(\mathbf{x},\mathbf{z}) = \exp(\mathbf{x}^T\mathbf{z})$', 
                r'$K(\mathbf{x},\mathbf{z}) = \sin(\mathbf{x}^T\mathbf{z})$']
colors = ['blue', 'green', 'red']

for i, (eig, name, color) in enumerate(zip(eigenvalues_data, kernel_names, colors)):
    ax = axes[0, i]
    ax.bar(range(len(eig)), eig, color=color, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel(r'Eigenvalue Index $i$')
    ax.set_ylabel(r'Eigenvalue $\lambda_i$')
    ax.grid(True, alpha=0.3)
    
    # Add validity indicator
    is_valid = np.all(eig >= -1e-10)
    ax.text(0.05, 0.95, f'Valid: {"YES" if is_valid else "NO"}', 
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"))
    
    # Add min eigenvalue annotation (handle complex numbers)
    min_eig = np.min(eig)
    if np.iscomplexobj(min_eig):
        min_eig_real = min_eig.real
        ax.text(0.05, 0.85, f'$\\min\\lambda = {min_eig_real:.3f}$', 
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", fc="lightblue", ec="black", alpha=0.8))
    else:
        ax.text(0.05, 0.85, f'$\\min\\lambda = {min_eig:.3f}$', 
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", fc="lightblue", ec="black", alpha=0.8))

# 2. Heatmaps of Gram matrices
gram_matrices = [K1, K2, K3]
for i, (K, name) in enumerate(zip(gram_matrices, kernel_names)):
    ax = axes[1, i]
    im = ax.imshow(K, cmap='viridis', aspect='auto')
    ax.set_title(r'Gram Matrix $\mathbf{K}$', fontsize=10)
    ax.set_xlabel(r'Point Index $j$')
    ax.set_ylabel(r'Point Index $i$')
    
    # Add text annotations
    for j in range(K.shape[0]):
        for k in range(K.shape[1]):
            ax.text(k, j, f'{K[j, k]:.2f}', ha='center', va='center', 
                   color='white' if K[j, k] > 0.5 else 'black', fontsize=8)
    
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'kernel_validity_analysis_detailed.png'), dpi=300, bbox_inches='tight')

# Create additional visualization for set kernels
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Set kernel heatmaps
set_kernels = [K_set_rbf, K_set_linear]
set_names = [r'RBF Set Kernel: $k(a,b) = \exp(-\gamma\|a-b\|^2)$', 
             r'Linear Set Kernel: $k(a,b) = a^T b$']

for i, (K, name) in enumerate(zip(set_kernels, set_names)):
    ax = axes[i]
    im = ax.imshow(K, cmap='plasma', aspect='auto')
    ax.set_title(name, fontsize=10)
    ax.set_xlabel(r'Set Index')
    ax.set_ylabel(r'Set Index')
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(['A', 'B', 'C'])
    ax.set_yticklabels(['A', 'B', 'C'])
    
    # Add text annotations
    for j in range(K.shape[0]):
        for k in range(K.shape[1]):
            ax.text(k, j, f'{K[j, k]:.3f}', ha='center', va='center', 
                   color='white' if K[j, k] > 0.5 else 'black', fontsize=10)
    
    # Add eigenvalue information
    eig_vals = eigvals(K)
    is_psd = np.all(eig_vals >= -1e-10)
    ax.text(0.05, 0.95, f'PSD: {"YES" if is_psd else "NO"}', 
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"))
    
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'set_kernel_analysis_detailed.png'), dpi=300, bbox_inches='tight')

# Create visualization showing the test points
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Plot test points
ax.scatter(test_points[:, 0], test_points[:, 1], s=200, c='red', alpha=0.7, 
          edgecolors='black', linewidth=2, label=r'Test Points $\{\mathbf{x}_1, \ldots, \mathbf{x}_5\}$')

# Plot specific points with different colors
ax.scatter(specific_points[:, 0], specific_points[:, 1], s=300, c='blue', alpha=0.7,
          edgecolors='black', linewidth=2, marker='s', label=r'Specific Points $\{\mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3\}$')

# Add point labels with LaTeX formatting
for i, point in enumerate(test_points):
    ax.annotate(r'$\mathbf{x}_{' + f'{i+1}' + r'}$', 
                (point[0], point[1]), xytext=(10, 10), 
                textcoords='offset points', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_title(r'Test Points for Kernel Evaluation')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-0.5, 1.5)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'test_points_detailed.png'), dpi=300, bbox_inches='tight')

# Create a mathematical summary visualization
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Create a text-based summary with LaTeX formatting (avoiding Unicode issues)
summary_text = r'''
Kernel Validity Summary

Valid Kernels:
• $K(\mathbf{x},\mathbf{z}) = (\mathbf{x}^T\mathbf{z})^2 + (\mathbf{x}^T\mathbf{z})^3$ (VALID)
• $K(\mathbf{x},\mathbf{z}) = \exp(\mathbf{x}^T\mathbf{z})$ (VALID)

Invalid Kernels:
• $K(\mathbf{x},\mathbf{z}) = \sin(\mathbf{x}^T\mathbf{z})$ (INVALID - negative eigenvalues)
• $K(\mathbf{x},\mathbf{z}) = -\|\mathbf{x}-\mathbf{z}\|^2$ (INVALID - negative eigenvalues)
• $K(\mathbf{x},\mathbf{z}) = x_0 \cdot z_1$ (INVALID - asymmetric)

Mercer's Theorem:
A kernel $K(\mathbf{x},\mathbf{z})$ is valid if and only if:
1. $\mathbf{K} = \mathbf{K}^T$ (symmetry)
2. $\lambda_i \geq 0$ for all eigenvalues $\lambda_i$ of $\mathbf{K}$ (PSD)

where $\mathbf{K}_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$ is the Gram matrix.
'''

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
        fc="lightblue", ec="black", alpha=0.9))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'kernel_summary_latex.png'), dpi=300, bbox_inches='tight')

# Create a simple 3D surface visualization of kernel functions
fig = plt.figure(figsize=(15, 5))

# Create a grid of points for visualization
x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
X, Y = np.meshgrid(x, y)

# Define kernel functions for 3D visualization
def kernel_3d_poly(x, y):
    return (x**2 + y**2)**2 + (x**2 + y**2)**3

def kernel_3d_exp(x, y):
    return np.exp(x**2 + y**2)

def kernel_3d_sin(x, y):
    return np.sin(x**2 + y**2)

# Compute kernel values
Z_poly = kernel_3d_poly(X, Y)
Z_exp = kernel_3d_exp(X, Y)
Z_sin = kernel_3d_sin(X, Y)

# Create subplots
kernels_3d = [Z_poly, Z_exp, Z_sin]
titles = ['Polynomial Kernel', 'Exponential Kernel', 'Sine Kernel']
colors = ['viridis', 'plasma', 'coolwarm']

for i, (Z, title, cmap) in enumerate(zip(kernels_3d, titles, colors)):
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.8, linewidth=0, antialiased=True)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('K(x,y)')
    ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'kernel_3d_surfaces.png'), dpi=300, bbox_inches='tight')

print(f"\nVisualizations saved to: {save_dir}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*60)
print("SUMMARY OF RESULTS")
print("="*60)

print(f"\nTask 1 - Kernel Validity:")
print(f"  K(x,z) = (x^T z)^2 + (x^T z)^3: {'✓ Valid' if valid1 else '✗ Invalid'}")
print(f"  K(x,z) = exp(x^T z): {'✓ Valid' if valid2 else '✗ Invalid'}")
print(f"  K(x,z) = sin(x^T z): {'✓ Valid' if valid3 else '✗ Invalid'}")

print(f"\nTask 2 - Specific Points Analysis:")
print(f"  All three kernels tested on points (0,0), (1,0), (0,1)")

print(f"\nTask 3 - Linear Combination:")
print(f"  K = 2*K1 + 3*K2: {'✓ Valid' if valid_comb else '✗ Invalid'}")

print(f"\nTask 4 - Invalid Kernel Examples:")
print(f"  K(x,z) = -||x-z||^2: {'✓ Valid' if valid_invalid else '✗ Invalid'}")
print(f"  K(x,z) = x[0]*z[1]: {'✓ Valid' if valid_invalid2 else '✗ Invalid'}")

print(f"\nTask 5 - Set Kernel:")
print(f"  RBF-based set kernel: {'✓ Valid' if np.all(eigvals(K_set_rbf) >= -1e-10) else '✗ Invalid'}")
print(f"  Linear-based set kernel: {'✓ Valid' if np.all(eigvals(K_set_linear) >= -1e-10) else '✗ Invalid'}")

print(f"\nAll detailed calculations and visualizations completed!")
