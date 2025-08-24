import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.linalg import eigvals
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_3_Quiz_6")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

print("Question 6: Mercer's Theorem and Kernel Validity")
print("=" * 50)

# Task 1: State Mercer's theorem precisely
print("\n1. Mercer's Theorem Statement")
print("-" * 30)
print("Mercer's Theorem:")
print("A symmetric function K(x,z) can be expressed as an inner product")
print("K(x,z) = ⟨φ(x), φ(z)⟩ in some feature space if and only if")
print("the kernel matrix K is positive semi-definite for any finite set of points.")
print("\nMathematically: K is a valid kernel ⟺ K ⪰ 0 (positive semi-definite)")

# Task 2: Verify that the given kernel matrix is positive semi-definite
print("\n2. Kernel Matrix PSD Verification")
print("-" * 35)

K_matrix = np.array([[1, 0.5], 
                     [0.5, 1]])

print("Given kernel matrix:")
print("K =", K_matrix)

# Check if matrix is symmetric
is_symmetric = np.allclose(K_matrix, K_matrix.T)
print(f"\nSymmetric: {is_symmetric}")

# Compute eigenvalues
eigenvalues = eigvals(K_matrix)
eigenvalues_real = np.real(eigenvalues)  # Extract real parts to avoid complex warnings
print(f"Eigenvalues: {eigenvalues_real}")

# Check if all eigenvalues are non-negative
is_psd = np.all(eigenvalues_real >= -1e-10)  # Allow for small numerical errors
print(f"All eigenvalues ≥ 0: {is_psd}")
print(f"Therefore, K is positive semi-definite: {is_psd}")

# Visualize the kernel matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot kernel matrix as heatmap
sns.heatmap(K_matrix, annot=True, cmap='RdYlBu_r', center=0, 
            square=True, ax=axes[0], cbar_kws={'label': 'Kernel Value'})
axes[0].set_title('Kernel Matrix K')
axes[0].set_xlabel('Data Point Index')
axes[0].set_ylabel('Data Point Index')

# Plot eigenvalues
axes[1].bar(range(len(eigenvalues_real)), eigenvalues_real, color='skyblue',
            edgecolor='black', alpha=0.7)
axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7, 
                label='Zero Line')
axes[1].set_title('Eigenvalues of Kernel Matrix')
axes[1].set_xlabel('Eigenvalue Index')
axes[1].set_ylabel('Eigenvalue')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'kernel_matrix_analysis.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# Task 3: Check if K(x,z) = exp(x^T z) is a valid kernel
print("\n3. Validity of K(x,z) = exp(x^T z)")
print("-" * 35)

# Test with specific points
test_points = np.array([[1, 0], [0, 1], [-1, 0]])
n_points = len(test_points)

# Compute kernel matrix for exp(x^T z)
K_exp = np.zeros((n_points, n_points))
for i in range(n_points):
    for j in range(n_points):
        K_exp[i, j] = np.exp(np.dot(test_points[i], test_points[j]))

print("Test points:", test_points)
print("Kernel matrix for K(x,z) = exp(x^T z):")
print(K_exp)

# Check eigenvalues
eigenvals_exp = eigvals(K_exp)
eigenvals_exp_real = np.real(eigenvals_exp)  # Extract real parts
print(f"Eigenvalues: {eigenvals_exp_real}")
is_psd_exp = np.all(eigenvals_exp_real >= -1e-10)
print(f"Positive semi-definite: {is_psd_exp}")

if is_psd_exp:
    print("K(x,z) = exp(x^T z) appears to be a valid kernel for this test set.")
else:
    print("K(x,z) = exp(x^T z) is NOT a valid kernel.")

# Task 4: Prove that K(x,z) = -||x-z||² is not a valid kernel
print("\n4. Invalidity of K(x,z) = -||x-z||²")
print("-" * 40)

print("Mathematical proof that K(x,z) = -||x-z||² is invalid:")
print("Step 1: Expand the squared distance")
print("||x-z||² = (x-z)ᵀ(x-z) = xᵀx - 2xᵀz + zᵀz")
print("So K(x,z) = -(xᵀx - 2xᵀz + zᵀz) = -xᵀx + 2xᵀz - zᵀz")
print("\nStep 2: Check if this can be written as ⟨φ(x), φ(z)⟩")
print("For any valid kernel, we need K(x,z) = φ(x)ᵀφ(z)")
print("But K(x,z) = -xᵀx + 2xᵀz - zᵀz cannot be factored this way")
print("The negative terms -xᵀx and -zᵀz prevent valid factorization")

# Test with simple 1D points
test_points_1d = np.array([[0], [1], [2]])
n_points_1d = len(test_points_1d)

# Compute kernel matrix for -||x-z||²
K_negative = np.zeros((n_points_1d, n_points_1d))
for i in range(n_points_1d):
    for j in range(n_points_1d):
        diff = test_points_1d[i] - test_points_1d[j]
        K_negative[i, j] = -np.dot(diff, diff)

print("Test points (1D):", test_points_1d.flatten())
print("Kernel matrix for K(x,z) = -||x-z||²:")
print(K_negative)

# Check eigenvalues
eigenvals_neg = eigvals(K_negative)
eigenvals_neg_real = np.real(eigenvals_neg)  # Extract real parts
print(f"Eigenvalues: {eigenvals_neg_real}")
is_psd_neg = np.all(eigenvals_neg_real >= -1e-10)
print(f"Positive semi-definite: {is_psd_neg}")

if not is_psd_neg:
    print("K(x,z) = -||x-z||² is NOT a valid kernel (has negative eigenvalues).")
    print("This violates Mercer's condition.")

# Task 5: 2D example showing why non-PSD kernels lead to optimization problems
print("\n5. Optimization Problems with Invalid Kernels")
print("-" * 45)

# Create visualization comparing valid vs invalid kernels
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Valid kernel matrices
kernels_info = [
    ("Valid: Linear", np.array([[1, 0.5], [0.5, 1]])),
    ("Valid: RBF-like", np.array([[1, 0.8], [0.8, 1]])),
    ("Invalid: Negative", np.array([[1, -1.5], [-1.5, 1]]))
]

for i, (title, K) in enumerate(kernels_info):
    # Plot kernel matrix
    sns.heatmap(K, annot=True, cmap='RdBu_r', center=0, 
                square=True, ax=axes[0, i], 
                cbar_kws={'label': 'Kernel Value'})
    axes[0, i].set_title(title)
    
    # Plot eigenvalues
    eigs = eigvals(K)
    eigs_real = np.real(eigs)  # Extract real parts
    colors = ['green' if e >= 0 else 'red' for e in eigs_real]
    axes[1, i].bar(range(len(eigs_real)), eigs_real, color=colors,
                   edgecolor='black', alpha=0.7)
    axes[1, i].axhline(y=0, color='black', linestyle='--', alpha=0.7)
    # Format eigenvalues for display (round to 3 decimal places)
    eigs_formatted = [f'{e:.3f}' for e in eigs_real]
    axes[1, i].set_title(f'Eigenvalues: {eigs_formatted}')
    axes[1, i].set_ylabel('Eigenvalue')
    axes[1, i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'valid_vs_invalid_kernels.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# Demonstrate optimization issues
print("\nWhy non-PSD kernels cause optimization problems:")
print("1. SVM optimization requires solving a quadratic programming problem")
print("2. The objective function involves the kernel matrix K")
print("3. For convex optimization, we need K to be positive semi-definite")
print("4. Negative eigenvalues make the problem non-convex")
print("5. Non-convex problems may have multiple local minima")
print("6. Standard QP solvers may fail or give incorrect solutions")

# Mathematical explanation
print("\nMathematical explanation:")
print("SVM dual problem: maximize Σαᵢ - ½ΣΣαᵢαⱼyᵢyⱼK(xᵢ,xⱼ)")
print("The Hessian matrix H = [yᵢyⱼK(xᵢ,xⱼ)] must be PSD for convexity")
print("If K has negative eigenvalues, H may not be PSD")

# Create a visualization of the optimization landscape
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

# Convex function (PSD case)
Z_convex = X**2 + Y**2

# Non-convex function (non-PSD case)
Z_nonconvex = X**2 - Y**2

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot convex landscape
contour1 = axes[0].contour(X, Y, Z_convex, levels=20, cmap='viridis')
axes[0].set_title('Convex Optimization Landscape\n(PSD Kernel)')
axes[0].set_xlabel('Parameter 1')
axes[0].set_ylabel('Parameter 2')
axes[0].plot(0, 0, 'r*', markersize=15, label='Global Minimum')
axes[0].legend()

# Plot non-convex landscape
contour2 = axes[1].contour(X, Y, Z_nonconvex, levels=20, cmap='viridis')
axes[1].set_title('Non-convex Optimization Landscape\n(Non-PSD Kernel)')
axes[1].set_xlabel('Parameter 1')
axes[1].set_ylabel('Parameter 2')
axes[1].plot(0, 0, 'r*', markersize=15, label='Saddle Point')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'optimization_landscapes.png'),
            dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAll plots saved to: {save_dir}")
print("\nSummary:")
print("- Mercer's theorem provides the mathematical foundation for valid kernels")
print("- Valid kernels must produce positive semi-definite kernel matrices")
print("- Invalid kernels lead to non-convex optimization problems")
print("- PSD condition ensures unique, globally optimal solutions")
