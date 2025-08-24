import numpy as np
import matplotlib.pyplot as plt
import os
from math import comb
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_3_Quiz_8")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

print("Question 8: Feature Space Dimensionality and the Kernel Trick")
print("=" * 60)

# Task 1: Calculate dimensionality for polynomial kernels
print("\n1. Polynomial Kernel Feature Space Dimensionality")
print("-" * 50)

def polynomial_feature_dim(n, d):
    """
    Calculate the dimensionality of polynomial feature space.
    For polynomial kernel of degree d in n dimensions:
    Dimensionality = C(n+d, d) = (n+d)! / (d! * n!)

    Mathematical derivation:
    - We need to count monomials x₁^i₁ x₂^i₂ ... xₙ^iₙ where i₁+i₂+...+iₙ ≤ d
    - This is equivalent to distributing d identical balls into n+1 bins
    - The extra bin accounts for the "unused degree"
    - By stars and bars: C(n+d, d) = C(n+d, n)
    """
    return comb(n + d, d)

n = 5  # 5 dimensions
degrees = [1, 2, 3, 4]

print(f"For n = {n} input dimensions:")
dimensions = []
for d in degrees:
    dim = polynomial_feature_dim(n, d)
    dimensions.append(dim)
    print(f"Degree d = {d}: Feature space dimension = {dim}")

# Visualize the growth
plt.figure(figsize=(10, 6))
plt.plot(degrees, dimensions, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Polynomial Degree (d)')
plt.ylabel('Feature Space Dimension')
plt.title(f'Feature Space Dimensionality Growth\n(n = {n} input dimensions)')
plt.grid(True, alpha=0.3)
plt.yscale('log')
for i, (d, dim) in enumerate(zip(degrees, dimensions)):
    plt.annotate(f'{dim}', (d, dim), textcoords="offset points", 
                xytext=(0,10), ha='center')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'polynomial_dimensionality_growth.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# Task 2: RBF kernel dimensionality
print("\n2. RBF Kernel Feature Space Dimensionality")
print("-" * 40)

print("The RBF kernel K(x,z) = exp(-γ||x-z||²) corresponds to an")
print("INFINITE-dimensional feature space.")
print("\nMathematical explanation:")
print("The RBF kernel can be expanded using the Taylor series:")
print("exp(-γ||x-z||²) = exp(-γ||x||²) * exp(-γ||z||²) * exp(2γx^T z)")
print("The term exp(2γx^T z) = Σ(n=0 to ∞) (2γx^T z)^n / n!")
print("Each term (x^T z)^n corresponds to polynomial features of degree n")
print("Since the sum goes to infinity, the feature space is infinite-dimensional")

# Task 3: How SVMs handle infinite-dimensional spaces
print("\n3. Computational Handling of Infinite-Dimensional Spaces")
print("-" * 55)

print("SVMs can handle infinite-dimensional feature spaces through:")
print("1. KERNEL TRICK: Never compute φ(x) explicitly")
print("2. DUAL FORMULATION: Work only with kernel evaluations K(xi, xj)")
print("3. FINITE SUPPORT: Only support vectors matter for the solution")
print("4. REPRESENTER THEOREM: Solution lies in span of training data")

print("\nKey insight: Even though the feature space is infinite-dimensional,")
print("the solution can be represented using only the finite training set!")

# Task 4: Prove kernel trick allows high-dimensional computation
print("\n4. Kernel Trick: Avoiding Explicit High-Dimensional Computation")
print("-" * 65)

print("Mathematical proof:")
print("Consider polynomial kernel K(x,z) = (x^T z + c)^d")
print("\nDirect computation:")
print("- Map to feature space: φ(x) ∈ R^D where D = C(n+d, d)")
print("- Compute inner product: φ(x)^T φ(z)")
print("- Complexity: O(D) where D grows exponentially with d")

print("\nKernel trick computation:")
print("- Compute K(x,z) = (x^T z + c)^d directly")
print("- Complexity: O(n) for inner product + O(1) for exponentiation")
print("- Total: O(n) regardless of feature space dimension!")

# Demonstrate computational complexity
n_values = [2, 5, 10, 20, 50]
d_values = [2, 3, 4, 5]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Feature space dimension vs input dimension
for d in d_values:
    dims = [polynomial_feature_dim(n, d) for n in n_values]
    axes[0].plot(n_values, dims, 'o-', label=f'd = {d}', linewidth=2)

axes[0].set_xlabel('Input Dimension (n)')
axes[0].set_ylabel('Feature Space Dimension')
axes[0].set_title('Feature Space Growth with Input Dimension')
axes[0].set_yscale('log')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Computational complexity comparison
n_range = np.arange(2, 21)
for d in [2, 3, 4]:
    # Direct computation complexity (proportional to feature space dimension)
    direct_complexity = [polynomial_feature_dim(n, d) for n in n_range]
    # Kernel trick complexity (linear in input dimension)
    kernel_complexity = n_range
    
    axes[1].plot(n_range, direct_complexity, '--', 
                label=f'Direct (d={d})', linewidth=2)
    if d == 2:  # Only plot kernel complexity once
        axes[1].plot(n_range, kernel_complexity, 'k-', 
                    label='Kernel Trick', linewidth=3)

axes[1].set_xlabel('Input Dimension (n)')
axes[1].set_ylabel('Computational Complexity')
axes[1].set_title('Computational Complexity: Direct vs Kernel Trick')
axes[1].set_yscale('log')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'computational_complexity_comparison.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# Task 5: Decision function in terms of kernel evaluations
print("\n5. Decision Function in Terms of Kernel Evaluations")
print("-" * 50)

print("SVM decision function can be expressed entirely using kernels:")
print("\nPrimal form (requires explicit feature mapping):")
print("f(x) = w^T φ(x) + b")
print("where w ∈ R^D (D-dimensional weight vector)")

print("\nDual form (kernel-based):")
print("f(x) = Σ(i=1 to n) αi yi K(xi, x) + b")
print("where:")
print("- αi are Lagrange multipliers (learned from training)")
print("- yi are training labels")
print("- K(xi, x) are kernel evaluations")
print("- Only support vectors (αi > 0) contribute to the sum")

print("\nKey advantages of dual formulation:")
print("1. No explicit weight vector w needed")
print("2. Computation depends only on kernel evaluations")
print("3. Works for infinite-dimensional feature spaces")
print("4. Sparse representation (only support vectors matter)")

# Demonstrate sparsity of support vectors
print("\nSupport Vector Sparsity Demonstration:")
# Simulate SVM solution sparsity
np.random.seed(42)
n_samples = 100
n_support_vectors = 15

# Create visualization of sparse solution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: All training points vs support vectors
all_points = np.random.randn(n_samples, 2)
sv_indices = np.random.choice(n_samples, n_support_vectors, replace=False)
sv_points = all_points[sv_indices]
non_sv_points = np.delete(all_points, sv_indices, axis=0)

axes[0].scatter(non_sv_points[:, 0], non_sv_points[:, 1], 
               c='lightblue', alpha=0.6, s=30, label='Training Points')
axes[0].scatter(sv_points[:, 0], sv_points[:, 1], 
               c='red', s=100, marker='s', edgecolor='black',
               label='Support Vectors')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].set_title(f'Support Vector Sparsity\n({n_support_vectors}/{n_samples} = {n_support_vectors/n_samples:.1%} are SVs)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Alpha coefficients (most are zero)
alphas = np.zeros(n_samples)
alphas[sv_indices] = np.random.exponential(1, n_support_vectors)

axes[1].bar(range(n_samples), alphas, color='skyblue', alpha=0.7)
axes[1].set_xlabel('Training Sample Index')
axes[1].set_ylabel('Alpha Coefficient')
axes[1].set_title('Lagrange Multipliers ($\\alpha$)\nMost are zero (non-support vectors)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'support_vector_sparsity.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# Summary table
print("\n" + "="*60)
print("SUMMARY TABLE: Feature Space Dimensions")
print("="*60)
print(f"{'Kernel Type':<20} {'Input Dim':<10} {'Feature Dim':<15} {'Complexity':<15}")
print("-"*60)
print(f"{'Linear':<20} {'n':<10} {'n':<15} {'O(n)':<15}")
for d in [2, 3, 4]:
    dim_formula = f"C(n+{d},{d})"
    print(f"{'Polynomial d='+str(d):<20} {'n':<10} {dim_formula:<15} {'O(n)':<15}")
print(f"{'RBF (Gaussian)':<20} {'n':<10} {'∞':<15} {'O(n)':<15}")

print(f"\nFor n=5 dimensions:")
for d in [1, 2, 3, 4]:
    dim = polynomial_feature_dim(5, d)
    print(f"  Polynomial d={d}: {dim} features")

print(f"\nAll plots saved to: {save_dir}")
print("\nKey Insights:")
print("- Polynomial feature spaces grow exponentially with degree")
print("- RBF kernels correspond to infinite-dimensional spaces")
print("- Kernel trick makes high-dimensional computation feasible")
print("- SVM solutions are sparse (only support vectors matter)")
print("- Decision functions depend only on kernel evaluations")
