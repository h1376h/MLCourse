import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import eigh
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_3_Quiz_27")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
# Use non-interactive backend to avoid displaying plots
plt.ioff()

print("=" * 80)
print("QUESTION 27: UNDERSTANDING MERCER'S THEOREM AND KERNEL VALIDITY")
print("=" * 80)

# ============================================================================
# PART 1: CORE REQUIREMENT FOR A VALID KERNEL
# ============================================================================

print("\n" + "="*60)
print("PART 1: CORE REQUIREMENT FOR A VALID KERNEL")
print("="*60)

print("""
A function K(x, z) is a valid kernel if and only if it corresponds to an inner product 
in some feature space. Mathematically, this means:

K(x, z) = <φ(x), φ(z)>

where φ is a feature mapping from the input space to a (possibly infinite-dimensional) 
feature space.

This is the fundamental requirement - the kernel must represent a dot product 
in some transformed space.
""")

# ============================================================================
# PART 2: MERCER'S THEOREM CONDITION
# ============================================================================

print("\n" + "="*60)
print("PART 2: MERCER'S THEOREM CONDITION")
print("="*60)

print("""
MERCER'S THEOREM:
A function K(x, z) is a valid kernel if and only if for any finite set of points 
{x₁, x₂, ..., xₙ}, the corresponding Gram matrix K is positive semi-definite (PSD).

The Gram matrix K is defined as:
K_ij = K(x_i, x_j)

A matrix is positive semi-definite if all its eigenvalues are non-negative.
""")

# ============================================================================
# PART 3: EXAMPLES AND DEMONSTRATIONS
# ============================================================================

print("\n" + "="*60)
print("PART 3: EXAMPLES AND DEMONSTRATIONS")
print("="*60)

def compute_gram_matrix(X, kernel_func):
    """Compute the Gram matrix for given points and kernel function."""
    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel_func(X[i], X[j])
    return K

def check_psd(K):
    """Check if a matrix is positive semi-definite."""
    eigenvalues = eigh(K, eigvals_only=True)
    is_psd = np.all(eigenvalues >= -1e-10)  # Small tolerance for numerical errors
    return is_psd, eigenvalues

def plot_eigenvalues(eigenvalues, title, filename):
    """Plot eigenvalues to visualize PSD property."""
    plt.figure(figsize=(10, 6))
    
    # Plot eigenvalues
    plt.subplot(1, 2, 1)
    plt.bar(range(len(eigenvalues)), eigenvalues, color='skyblue', alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero line')
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue')
    plt.title(f'Eigenvalues of {title}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot eigenvalues on number line
    plt.subplot(1, 2, 2)
    plt.scatter(eigenvalues, np.zeros_like(eigenvalues), s=100, color='blue', alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero line')
    plt.xlabel('Eigenvalue Value')
    plt.ylabel('')
    plt.title(f'Eigenvalue Distribution for {title}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(min(eigenvalues) - 0.1, max(eigenvalues) + 0.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def plot_gram_matrix(K, title, filename):
    """Plot the Gram matrix as a heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(K, annot=True, fmt='.3f', cmap='viridis', 
                xticklabels=[f'x{i+1}' for i in range(K.shape[0])],
                yticklabels=[f'x{i+1}' for i in range(K.shape[0])])
    plt.title(f'Gram Matrix: {title}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# Example 1: Linear Kernel (Valid)
print("\n--- Example 1: Linear Kernel (Valid) ---")
print("K(x, z) = x^T z")

# Define sample points
X_linear = np.array([[1, 2], [3, 4], [0, 1], [2, 3]])

def linear_kernel(x, z):
    return np.dot(x, z)

# Compute Gram matrix
K_linear = compute_gram_matrix(X_linear, linear_kernel)
print(f"Sample points: {X_linear}")
print(f"Gram matrix:\n{K_linear}")

# Check PSD property
is_psd_linear, eigenvals_linear = check_psd(K_linear)
print(f"Is PSD: {is_psd_linear}")
print(f"Eigenvalues: {eigenvals_linear}")

# Plot results
plot_gram_matrix(K_linear, "Linear Kernel", "linear_kernel_gram_matrix.png")
plot_eigenvalues(eigenvals_linear, "Linear Kernel", "linear_kernel_eigenvalues.png")

# Example 2: RBF Kernel (Valid)
print("\n--- Example 2: RBF Kernel (Valid) ---")
print("K(x, z) = exp(-γ ||x - z||²)")

def rbf_kernel(x, z, gamma=1.0):
    diff = np.array(x) - np.array(z)
    return np.exp(-gamma * np.dot(diff, diff))

# Compute Gram matrix
K_rbf = compute_gram_matrix(X_linear, lambda x, z: rbf_kernel(x, z, gamma=0.5))
print(f"Gram matrix (RBF, γ=0.5):\n{K_rbf}")

# Check PSD property
is_psd_rbf, eigenvals_rbf = check_psd(K_rbf)
print(f"Is PSD: {is_psd_rbf}")
print(f"Eigenvalues: {eigenvals_rbf}")

# Plot results
plot_gram_matrix(K_rbf, "RBF Kernel", "rbf_kernel_gram_matrix.png")
plot_eigenvalues(eigenvals_rbf, "RBF Kernel", "rbf_kernel_eigenvalues.png")

# Example 3: Polynomial Kernel (Valid)
print("\n--- Example 3: Polynomial Kernel (Valid) ---")
print("K(x, z) = (x^T z + c)^d")

def polynomial_kernel(x, z, d=2, c=1):
    return (np.dot(x, z) + c) ** d

# Compute Gram matrix
K_poly = compute_gram_matrix(X_linear, lambda x, z: polynomial_kernel(x, z, d=2, c=1))
print(f"Gram matrix (Polynomial, d=2, c=1):\n{K_poly}")

# Check PSD property
is_psd_poly, eigenvals_poly = check_psd(K_poly)
print(f"Is PSD: {is_psd_poly}")
print(f"Eigenvalues: {eigenvals_poly}")

# Plot results
plot_gram_matrix(K_poly, "Polynomial Kernel", "polynomial_kernel_gram_matrix.png")
plot_eigenvalues(eigenvals_poly, "Polynomial Kernel", "polynomial_kernel_eigenvalues.png")

# Example 4: Invalid Kernel
print("\n--- Example 4: Invalid Kernel ---")
print("K(x, z) = sin(x^T z)")

def invalid_kernel(x, z):
    return np.sin(np.dot(x, z))

# Compute Gram matrix
K_invalid = compute_gram_matrix(X_linear, invalid_kernel)
print(f"Gram matrix (Invalid):\n{K_invalid}")

# Check PSD property
is_psd_invalid, eigenvals_invalid = check_psd(K_invalid)
print(f"Is PSD: {is_psd_invalid}")
print(f"Eigenvalues: {eigenvals_invalid}")

# Plot results
plot_gram_matrix(K_invalid, "Invalid Kernel", "invalid_kernel_gram_matrix.png")
plot_eigenvalues(eigenvals_invalid, "Invalid Kernel", "invalid_kernel_eigenvalues.png")

# ============================================================================
# PART 4: VISUALIZATION OF KERNEL PROPERTIES
# ============================================================================

print("\n" + "="*60)
print("PART 4: VISUALIZATION OF KERNEL PROPERTIES")
print("="*60)

def plot_kernel_surface(kernel_func, kernel_name, filename):
    """Plot kernel function as a surface."""
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    
    # Fix one point and compute kernel values
    fixed_point = np.array([1.0, 1.0])
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i, j], Y[i, j]])
            Z[i, j] = kernel_func(fixed_point, point)
    
    fig = plt.figure(figsize=(12, 5))
    
    # 3D surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.scatter([fixed_point[0]], [fixed_point[1]], [kernel_func(fixed_point, fixed_point)], 
                color='red', s=100, label='Fixed point')
    ax1.set_xlabel(r'$x_1$')
    ax1.set_ylabel(r'$x_2$')
    ax1.set_zlabel(f'$K(x, [{fixed_point[0]}, {fixed_point[1]}])$')
    ax1.set_title(f'{kernel_name} Kernel Surface')
    ax1.legend()
    
    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, Z, levels=20, cmap='viridis')
    ax2.scatter([fixed_point[0]], [fixed_point[1]], color='red', s=100, label='Fixed point')
    ax2.set_xlabel(r'$x_1$')
    ax2.set_ylabel(r'$x_2$')
    ax2.set_title(f'{kernel_name} Kernel Contours')
    ax2.legend()
    plt.colorbar(contour, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# Plot kernel surfaces
plot_kernel_surface(linear_kernel, "Linear", "linear_kernel_surface.png")
plot_kernel_surface(lambda x, z: rbf_kernel(x, z, gamma=0.5), "RBF", "rbf_kernel_surface.png")
plot_kernel_surface(lambda x, z: polynomial_kernel(x, z, d=2, c=1), "Polynomial", "polynomial_kernel_surface.png")
plot_kernel_surface(invalid_kernel, "Invalid", "invalid_kernel_surface.png")

# ============================================================================
# PART 5: IMPACT OF INVALID KERNELS ON SVM OPTIMIZATION
# ============================================================================

print("\n" + "="*60)
print("PART 5: IMPACT OF INVALID KERNELS ON SVM OPTIMIZATION")
print("="*60)

def demonstrate_svm_issues():
    """Demonstrate what happens when using invalid kernels in SVM."""
    
    # Create a simple 2D dataset
    np.random.seed(42)
    X_pos = np.random.multivariate_normal([1, 1], [[0.5, 0], [0, 0.5]], 20)
    X_neg = np.random.multivariate_normal([-1, -1], [[0.5, 0], [0, 0.5]], 20)
    X_data = np.vstack([X_pos, X_neg])
    y_data = np.hstack([np.ones(20), -np.ones(20)])
    
    print(f"Dataset shape: {X_data.shape}")
    print(f"Labels: {np.unique(y_data, return_counts=True)}")
    
    # Test different kernels
    kernels = {
        'Linear': linear_kernel,
        'RBF': lambda x, z: rbf_kernel(x, z, gamma=1.0),
        'Polynomial': lambda x, z: polynomial_kernel(x, z, d=2, c=1),
        'Invalid': invalid_kernel
    }
    
    results = {}
    
    for name, kernel_func in kernels.items():
        print(f"\n--- Testing {name} Kernel ---")
        
        # Compute Gram matrix
        K = compute_gram_matrix(X_data, kernel_func)
        
        # Check PSD property
        is_psd, eigenvals = check_psd(K)
        min_eigenval = np.min(eigenvals)
        
        print(f"Minimum eigenvalue: {min_eigenval:.6f}")
        print(f"Is PSD: {is_psd}")
        
        # Simulate SVM dual problem
        # The dual problem involves solving: max_α Σαᵢ - 1/2 ΣαᵢαⱼyᵢyⱼK(xᵢ, xⱼ)
        # subject to: 0 ≤ αᵢ ≤ C and Σαᵢyᵢ = 0
        
        # For demonstration, we'll check if the quadratic term is well-behaved
        n = len(X_data)
        alpha = np.random.random(n)  # Random dual variables
        alpha = alpha / np.sum(alpha)  # Normalize
        
        # Compute the quadratic term
        quadratic_term = 0
        for i in range(n):
            for j in range(n):
                quadratic_term += alpha[i] * alpha[j] * y_data[i] * y_data[j] * K[i, j]
        
        print(f"Quadratic term value: {quadratic_term:.6f}")
        
        # Check if the problem is convex
        is_convex = quadratic_term >= 0
        print(f"Is convex: {is_convex}")
        
        results[name] = {
            'is_psd': is_psd,
            'min_eigenval': min_eigenval,
            'quadratic_term': quadratic_term,
            'is_convex': is_convex
        }
    
    # Create summary plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Minimum eigenvalues
    names = list(results.keys())
    min_eigenvals = [results[name]['min_eigenval'] for name in names]
    colors = ['green' if results[name]['is_psd'] else 'red' for name in names]
    
    bars1 = ax1.bar(names, min_eigenvals, color=colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax1.set_ylabel('Minimum Eigenvalue')
    ax1.set_title('Minimum Eigenvalues of Gram Matrices')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars1, min_eigenvals):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.4f}', ha='center', va='bottom')
    
    # Plot 2: PSD status
    psd_status = [1 if results[name]['is_psd'] else 0 for name in names]
    bars2 = ax2.bar(names, psd_status, color=colors, alpha=0.7)
    ax2.set_ylabel('PSD Status (1=Valid, 0=Invalid)')
    ax2.set_title('Positive Semi-Definite Status')
    ax2.set_ylim(0, 1.2)
    ax2.grid(True, alpha=0.3)
    
    # Add labels
    for bar, status in zip(bars2, psd_status):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                'Valid' if status else 'Invalid', ha='center', va='bottom')
    
    # Plot 3: Quadratic term values
    quad_terms = [results[name]['quadratic_term'] for name in names]
    bars3 = ax3.bar(names, quad_terms, color=colors, alpha=0.7)
    ax3.set_ylabel('Quadratic Term Value')
    ax3.set_title('SVM Dual Quadratic Term')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars3, quad_terms):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.4f}', ha='center', va='bottom')
    
    # Plot 4: Dataset visualization
    ax4.scatter(X_data[y_data == 1, 0], X_data[y_data == 1, 1], 
                c='blue', marker='o', s=50, label='Class +1', alpha=0.7)
    ax4.scatter(X_data[y_data == -1, 0], X_data[y_data == -1, 1], 
                c='red', marker='s', s=50, label='Class -1', alpha=0.7)
    ax4.set_xlabel(r'$x_1$')
    ax4.set_ylabel(r'$x_2$')
    ax4.set_title('Binary Classification Dataset')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'kernel_comparison_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return results

# Run the demonstration
svm_results = demonstrate_svm_issues()

# ============================================================================
# PART 6: SUMMARY AND CONCLUSIONS
# ============================================================================

print("\n" + "="*60)
print("PART 6: SUMMARY AND CONCLUSIONS")
print("="*60)

print("""
SUMMARY OF FINDINGS:

1. CORE REQUIREMENT:
   - A valid kernel must correspond to an inner product in some feature space
   - K(x, z) = <φ(x), φ(z)> for some feature mapping φ

2. MERCER'S THEOREM CONDITION:
   - The Gram matrix K must be positive semi-definite for any finite set of points
   - All eigenvalues of K must be non-negative

3. IMPACT OF INVALID KERNELS:
   - Non-PSD kernels lead to non-convex optimization problems
   - The SVM dual objective function becomes unbounded
   - Numerical instability and convergence issues
   - No guarantee of finding a global optimum

VALID KERNELS TESTED:
- Linear kernel: K(x, z) = x^T z ✓
- RBF kernel: K(x, z) = exp(-γ ||x - z||²) ✓  
- Polynomial kernel: K(x, z) = (x^T z + c)^d ✓

INVALID KERNEL TESTED:
- Sin kernel: K(x, z) = sin(x^T z) ✗
""")

print(f"\nAll plots and visualizations have been saved to: {save_dir}")
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
