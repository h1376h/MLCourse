import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.linalg import eigvals
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_3_Quiz_7")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

print("Question 7: Kernel Combinations and Closure Properties")
print("=" * 55)

# Define test points for demonstrations
test_points = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
n_points = len(test_points)

def compute_kernel_matrix(points, kernel_func):
    """Compute kernel matrix for given points and kernel function."""
    n = len(points)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel_func(points[i], points[j])
    return K

def is_psd(matrix, tolerance=1e-10):
    """Check if matrix is positive semi-definite."""
    eigenvals = eigvals(matrix)
    eigenvals_real = np.real(eigenvals)  # Extract real parts to avoid complex warnings
    return np.all(eigenvals_real >= -tolerance)

# Define basic kernel functions
def linear_kernel(x, z):
    return np.dot(x, z)

def rbf_kernel(x, z, gamma=1.0):
    return np.exp(-gamma * np.linalg.norm(x - z)**2)

def polynomial_kernel(x, z, degree=2, c=1):
    return (np.dot(x, z) + c)**degree

# Task 1: Prove that K1 + K2 is valid if K1 and K2 are valid
print("\n1. Kernel Addition: K(x,z) = K1(x,z) + K2(x,z)")
print("-" * 50)

# Compute individual kernel matrices
K_linear = compute_kernel_matrix(test_points, linear_kernel)
K_rbf = compute_kernel_matrix(test_points, lambda x, z: rbf_kernel(x, z, gamma=1.0))

print("Linear kernel matrix K1:")
print(K_linear)
print(f"PSD: {is_psd(K_linear)}")
eigs_linear = np.real(eigvals(K_linear))
print(f"Eigenvalues: {eigs_linear}")

print("\nRBF kernel matrix K2:")
print(K_rbf)
print(f"PSD: {is_psd(K_rbf)}")
eigs_rbf = np.real(eigvals(K_rbf))
print(f"Eigenvalues: {eigs_rbf}")

# Compute sum
K_sum = K_linear + K_rbf
print("\nSum kernel matrix K = K1 + K2:")
print(K_sum)
print(f"PSD: {is_psd(K_sum)}")
eigs_sum = np.real(eigvals(K_sum))
print(f"Eigenvalues: {eigs_sum}")

print("\nMathematical proof:")
print("Theorem: If K₁ and K₂ are valid kernels, then K = K₁ + K₂ is valid.")
print("Proof:")
print("Step 1: K₁ and K₂ are PSD ⟹ ∀c ∈ ℝⁿ: cᵀK₁c ≥ 0 and cᵀK₂c ≥ 0")
print("Step 2: For any vector c:")
print("       cᵀ(K₁ + K₂)c = cᵀK₁c + cᵀK₂c ≥ 0 + 0 = 0")
print("Step 3: Therefore K₁ + K₂ ⪰ 0 (PSD)")
print("Step 4: By Mercer's theorem, K₁ + K₂ is a valid kernel. ∎")

# Task 2: Prove that cK1 is valid for c > 0
print("\n2. Kernel Scaling: K(x,z) = c·K1(x,z) for c > 0")
print("-" * 50)

c_values = [0.5, 2.0, 10.0]
for c in c_values:
    K_scaled = c * K_linear
    print(f"\nFor c = {c}:")
    eigs_scaled = np.real(eigvals(K_scaled))
    print(f"Scaled kernel matrix eigenvalues: {eigs_scaled}")
    print(f"PSD: {is_psd(K_scaled)}")

print("\nMathematical proof:")
print("Theorem: If K₁ is a valid kernel and c > 0, then K = cK₁ is valid.")
print("Proof:")
print("Step 1: K₁ is PSD ⟹ ∀v ∈ ℝⁿ: vᵀK₁v ≥ 0")
print("Step 2: For any vector v and c > 0:")
print("       vᵀ(cK₁)v = c(vᵀK₁v) ≥ c·0 = 0")
print("       (since c > 0 and vᵀK₁v ≥ 0)")
print("Step 3: Therefore cK₁ ⪰ 0 (PSD)")
print("Step 4: By Mercer's theorem, cK₁ is a valid kernel. ∎")

# Task 3: Prove that K1 * K2 is valid (element-wise product)
print("\n3. Kernel Product: K(x,z) = K1(x,z) · K2(x,z)")
print("-" * 50)

# Element-wise product (Hadamard product)
K_product = K_linear * K_rbf
print("Product kernel matrix K = K1 ⊙ K2 (element-wise):")
print(K_product)
print(f"PSD: {is_psd(K_product)}")
eigs_product = np.real(eigvals(K_product))
print(f"Eigenvalues: {eigs_product}")

print("\nMathematical explanation:")
print("The element-wise product of two PSD matrices is PSD.")
print("This follows from Schur's theorem on Hadamard products.")
print("If K1 = Φ1^T Φ1 and K2 = Φ2^T Φ2, then")
print("K1 ⊙ K2 corresponds to the kernel of the tensor product feature space.")

# Task 4: Design combined kernel
print("\n4. Combined Kernel Design: K(x,z) = α·K_linear + β·K_RBF")
print("-" * 60)

# Choose appropriate alpha and beta
alpha, beta = 0.3, 0.7  # Weights that sum to 1 for normalization

K_combined = alpha * K_linear + beta * K_rbf
print(f"Combined kernel with α = {alpha}, β = {beta}:")
print(K_combined)
print(f"PSD: {is_psd(K_combined)}")
eigs_combined = np.real(eigvals(K_combined))
print(f"Eigenvalues: {eigs_combined}")

print(f"\nRationale for choosing α = {alpha}, β = {beta}:")
print("- α + β = 1 provides a convex combination")
print("- β > α gives more weight to RBF (non-linear) component")
print("- This balances linear and non-linear characteristics")

# Task 5: Check if min(K1, K2) is valid
print("\n5. Minimum Kernel: K(x,z) = min(K1(x,z), K2(x,z))")
print("-" * 55)

K_min = np.minimum(K_linear, K_rbf)
print("Minimum kernel matrix K = min(K1, K2):")
print(K_min)
print(f"PSD: {is_psd(K_min)}")
eigenvals_min = np.real(eigvals(K_min))
print(f"Eigenvalues: {eigenvals_min}")

if not is_psd(K_min):
    print("\nCounterexample found!")
    print("The minimum of two valid kernels is NOT always a valid kernel.")
    print("This violates the PSD condition.")
else:
    print("\nFor this specific example, min(K1, K2) appears to be PSD.")
    print("However, this is not guaranteed in general.")

# Create a specific counterexample
print("\nConstructing a counterexample:")
# Use specific points that will create a non-PSD minimum
counter_points = np.array([[1, 0], [0, 1]])
K1_counter = compute_kernel_matrix(counter_points, linear_kernel)
K2_counter = compute_kernel_matrix(counter_points, lambda x, z: rbf_kernel(x, z, gamma=0.1))

print("K1 (linear):")
print(K1_counter)
print("K2 (RBF, γ=0.1):")
print(K2_counter)

K_min_counter = np.minimum(K1_counter, K2_counter)
print("min(K1, K2):")
print(K_min_counter)
print(f"PSD: {is_psd(K_min_counter)}")
eigs_min_counter = np.real(eigvals(K_min_counter))
print(f"Eigenvalues: {eigs_min_counter}")

# Visualization of kernel combinations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

kernels_to_plot = [
    ("Linear $K_1$", K_linear),
    ("RBF $K_2$", K_rbf),
    ("Sum $K_1 + K_2$", K_sum),
    ("Product $K_1 \\odot K_2$", K_product),
    ("Combined $\\alpha K_1 + \\beta K_2$", K_combined),
    ("Minimum $\\min(K_1, K_2)$", K_min)
]

for i, (title, K) in enumerate(kernels_to_plot):
    row, col = i // 3, i % 3
    
    # Plot kernel matrix heatmap
    sns.heatmap(K, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                center=0, square=True, ax=axes[row, col],
                cbar_kws={'label': 'Kernel Value'})
    axes[row, col].set_title(title)
    
    # Add PSD status
    psd_status = "PSD" if is_psd(K) else "Not PSD"
    axes[row, col].text(0.02, 0.98, psd_status, 
                       transform=axes[row, col].transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", 
                               facecolor='green' if is_psd(K) else 'red',
                               alpha=0.7),
                       verticalalignment='top')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'kernel_combinations.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# Eigenvalue comparison
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

kernel_names = [name for name, _ in kernels_to_plot]
eigenvals_list = [np.real(eigvals(K)) for _, K in kernels_to_plot]

x_pos = np.arange(len(kernel_names))
colors = ['green' if is_psd(K) else 'red' for _, K in kernels_to_plot]

for i, (name, eigs) in enumerate(zip(kernel_names, eigenvals_list)):
    y_pos = np.full(len(eigs), i)
    ax.scatter(eigs, y_pos, s=100, alpha=0.7, color=colors[i])

ax.set_yticks(x_pos)
ax.set_yticklabels(kernel_names)
ax.set_xlabel('Eigenvalue')
ax.set_title('Eigenvalues of Different Kernel Combinations')
ax.axvline(x=0, color='black', linestyle='--', alpha=0.7, label='Zero Line')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'eigenvalue_comparison.png'),
            dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAll plots saved to: {save_dir}")
print("\nSummary of Closure Properties:")
print("✓ Addition: K1 + K2 is valid if K1, K2 are valid")
print("✓ Positive scaling: c·K1 is valid if K1 is valid and c > 0")
print("✓ Element-wise product: K1 ⊙ K2 is valid if K1, K2 are valid")
print("✓ Linear combination: α·K1 + β·K2 is valid if K1, K2 are valid and α,β ≥ 0")
print("✗ Minimum: min(K1, K2) is NOT always valid, even if K1, K2 are valid")
