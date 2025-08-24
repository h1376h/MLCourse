import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
from sklearn.datasets import make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
import time
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_3_Quiz_26")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("=== Question 26: The Essence of the Kernel Trick ===\n")

# ============================================================================
# PART 1: Understanding the Kernel Trick - Main Idea
# ============================================================================

print("1. EXPLAINING THE MAIN IDEA BEHIND THE KERNEL TRICK")
print("=" * 60)

# Create non-linearly separable data
np.random.seed(42)
X, y = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)

# Plot original data
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='red', s=50, alpha=0.7, label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='blue', s=50, alpha=0.7, label='Class 1')
plt.title('Original 2D Data (Non-linearly Separable)')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.grid(True, alpha=0.3)

# Show explicit feature mapping for polynomial kernel (x^T z + 1)^2
print("\n1.1 Explicit Feature Mapping for Polynomial Kernel")
print("-" * 50)

# Take a few sample points
sample_points = X[:5]
print(f"Sample points (first 5):\n{sample_points}")

# Show explicit mapping for polynomial kernel (x^T z + 1)^2
# For 2D input [x1, x2], the mapping is:
# φ(x) = [1, √2*x1, √2*x2, x1², √2*x1*x2, x2²]
def explicit_polynomial_mapping(X):
    """Explicit mapping for polynomial kernel (x^T z + 1)^2"""
    n_samples = X.shape[0]
    # For 2D input, this creates 6D feature space
    mapped = np.zeros((n_samples, 6))
    mapped[:, 0] = 1  # constant term
    mapped[:, 1] = np.sqrt(2) * X[:, 0]  # √2 * x1
    mapped[:, 2] = np.sqrt(2) * X[:, 1]  # √2 * x2
    mapped[:, 3] = X[:, 0]**2  # x1²
    mapped[:, 4] = np.sqrt(2) * X[:, 0] * X[:, 1]  # √2 * x1 * x2
    mapped[:, 5] = X[:, 1]**2  # x2²
    return mapped

# Apply explicit mapping
X_mapped = explicit_polynomial_mapping(sample_points)
print(f"\nExplicit mapping to 6D space:\n{X_mapped}")

# Show kernel computation
print(f"\n1.2 Kernel Computation Comparison")
print("-" * 40)

# Compute kernel using explicit mapping
def kernel_explicit_mapping(x1, x2):
    """Compute kernel using explicit mapping"""
    phi1 = explicit_polynomial_mapping(x1.reshape(1, -1)).flatten()
    phi2 = explicit_polynomial_mapping(x2.reshape(1, -1)).flatten()
    return np.dot(phi1, phi2)

# Compute kernel using kernel trick
def kernel_trick(x1, x2):
    """Compute kernel using kernel trick"""
    return (np.dot(x1, x2) + 1)**2

# Compare both methods
x1, x2 = sample_points[0], sample_points[1]
kernel_explicit = kernel_explicit_mapping(x1, x2)
kernel_trick_result = kernel_trick(x1, x2)

print(f"Point 1: {x1}")
print(f"Point 2: {x2}")
print(f"Kernel using explicit mapping: {kernel_explicit:.6f}")
print(f"Kernel using kernel trick: {kernel_trick_result:.6f}")
print(f"Difference: {abs(kernel_explicit - kernel_trick_result):.10f}")

# ============================================================================
# PART 2: Dual Formulation and Dot Product Property
# ============================================================================

print(f"\n\n2. DUAL FORMULATION AND DOT PRODUCT PROPERTY")
print("=" * 60)

# Show dual formulation
print("\n2.1 Dual Formulation of SVM")
print("-" * 35)

print("Primal formulation:")
print("minimize: (1/2) ||w||² + C Σξᵢ")
print("subject to: yᵢ(wᵀφ(xᵢ) + b) ≥ 1 - ξᵢ")

print("\nDual formulation:")
print("maximize: Σαᵢ - (1/2) Σᵢⱼ αᵢαⱼyᵢyⱼφ(xᵢ)ᵀφ(xⱼ)")
print("subject to: 0 ≤ αᵢ ≤ C, Σαᵢyᵢ = 0")

print("\nKey insight: The dual depends only on φ(xᵢ)ᵀφ(xⱼ), not φ(x) individually!")

# Demonstrate with Gram matrix
print(f"\n2.2 Gram Matrix Construction")
print("-" * 35)

# Compute Gram matrix using explicit mapping
n_samples = 10
X_small = X[:n_samples]
y_small = y[:n_samples]

# Method 1: Explicit mapping
X_mapped_small = explicit_polynomial_mapping(X_small)
gram_explicit = X_mapped_small @ X_mapped_small.T

# Method 2: Kernel trick
gram_kernel = polynomial_kernel(X_small, degree=2, coef0=1)

print(f"Gram matrix using explicit mapping (first 5x5):")
print(gram_explicit[:5, :5])
print(f"\nGram matrix using kernel trick (first 5x5):")
print(gram_kernel[:5, :5])
print(f"\nMaximum difference: {np.max(np.abs(gram_explicit - gram_kernel)):.10f}")

# ============================================================================
# PART 3: Computational Efficiency Comparison
# ============================================================================

print(f"\n\n3. COMPUTATIONAL EFFICIENCY COMPARISON")
print("=" * 60)

# Compare computational costs
print("\n3.1 Time Complexity Analysis")
print("-" * 35)

def time_explicit_mapping(X):
    """Time the explicit mapping approach"""
    start_time = time.time()
    X_mapped = explicit_polynomial_mapping(X)
    gram_matrix = X_mapped @ X_mapped.T
    end_time = time.time()
    return end_time - start_time, gram_matrix

def time_kernel_trick(X):
    """Time the kernel trick approach"""
    start_time = time.time()
    gram_matrix = polynomial_kernel(X, degree=2, coef0=1)
    end_time = time.time()
    return end_time - start_time, gram_matrix

# Test with different dataset sizes
sizes = [100, 500, 1000]
print("Dataset Size | Explicit (s) | Kernel Trick (s) | Speedup")
print("-" * 55)

for size in sizes:
    X_test = X[:size]
    
    time_explicit, _ = time_explicit_mapping(X_test)
    time_kernel, _ = time_kernel_trick(X_test)
    speedup = time_explicit / time_kernel
    
    print(f"{size:11d} | {time_explicit:11.4f} | {time_kernel:13.4f} | {speedup:7.2f}x")

print(f"\n3.2 Memory Complexity Analysis")
print("-" * 35)

print("For polynomial kernel (x^T z + 1)^2 with 2D input:")
print("- Input dimension: 2")
print("- Explicit mapping dimension: 6")
print("- Memory for explicit mapping: O(n × 6)")
print("- Memory for kernel trick: O(n × 2)")

# Show the mapping dimensions
print(f"\nExplicit mapping dimensions:")
print(f"φ(x) = [1, √2*x₁, √2*x₂, x₁², √2*x₁*x₂, x₂²]")
print(f"Total dimensions: 6")

# ============================================================================
# PART 4: Infinite-Dimensional Feature Spaces
# ============================================================================

print(f"\n\n4. INFINITE-DIMENSIONAL FEATURE SPACES")
print("=" * 60)

print("\n4.1 RBF Kernel Example")
print("-" * 25)

# RBF kernel: K(x, z) = exp(-γ||x - z||²)
# This corresponds to an infinite-dimensional feature space

def rbf_kernel_manual(X, gamma=1.0):
    """Manual implementation of RBF kernel"""
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            diff = X[i] - X[j]
            K[i, j] = np.exp(-gamma * np.dot(diff, diff))
    return K

# Compare with sklearn implementation
X_small = X[:20]
rbf_manual = rbf_kernel_manual(X_small, gamma=1.0)
rbf_sklearn = rbf_kernel(X_small, gamma=1.0)

print(f"RBF kernel computation:")
print(f"Manual implementation max diff: {np.max(np.abs(rbf_manual - rbf_sklearn)):.10f}")

print(f"\n4.2 Why RBF Kernel is Infinite-Dimensional")
print("-" * 40)

print("RBF kernel: K(x, z) = exp(-γ||x - z||²)")
print("This can be expanded as:")
print("exp(-γ||x - z||²) = exp(-γ||x||²) exp(-γ||z||²) exp(2γx^T z)")
print("= exp(-γ||x||²) exp(-γ||z||²) Σₖ (2γ)^k/k! (x^T z)^k")

print("\nThe expansion contains infinitely many terms, each corresponding to")
print("a different polynomial degree, making the feature space infinite-dimensional.")

# ============================================================================
# PART 5: Visualization of Kernel Effects
# ============================================================================

print(f"\n\n5. VISUALIZATION OF KERNEL EFFECTS")
print("=" * 60)

# Create different datasets
np.random.seed(42)
X_circles, y_circles = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)
X_moons, y_moons = make_moons(n_samples=200, noise=0.1, random_state=42)

# Train SVMs with different kernels
kernels = ['linear', 'poly', 'rbf']
datasets = [(X_circles, y_circles, 'Circles'), (X_moons, y_moons, 'Moons')]

plt.figure(figsize=(15, 10))

for i, (X_data, y_data, dataset_name) in enumerate(datasets):
    for j, kernel in enumerate(kernels):
        plt.subplot(2, 3, i*3 + j + 1)
        
        # Train SVM
        if kernel == 'poly':
            svm = SVC(kernel='poly', degree=2, C=1.0, random_state=42)
        elif kernel == 'rbf':
            svm = SVC(kernel='rbf', C=1.0, random_state=42)
        else:
            svm = SVC(kernel='linear', C=1.0, random_state=42)
        
        svm.fit(X_data, y_data)
        
        # Create mesh for decision boundary
        x_min, x_max = X_data[:, 0].min() - 0.5, X_data[:, 0].max() + 0.5
        y_min, y_max = X_data[:, 1].min() - 0.5, X_data[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        plt.scatter(X_data[y_data == 0][:, 0], X_data[y_data == 0][:, 1], 
                   c='red', s=30, alpha=0.7, label='Class 0')
        plt.scatter(X_data[y_data == 1][:, 0], X_data[y_data == 1][:, 1], 
                   c='blue', s=30, alpha=0.7, label='Class 1')
        
        # Highlight support vectors
        plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
                   s=100, linewidth=1, facecolors='none', edgecolors='black')
        
        plt.title(f'{dataset_name} - {kernel.upper()} Kernel')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.legend()
        plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'kernel_comparison.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# PART 6: Detailed Polynomial Kernel Analysis
# ============================================================================

print(f"\n\n6. DETAILED POLYNOMIAL KERNEL ANALYSIS")
print("=" * 60)

# Show step-by-step computation for polynomial kernel
print("\n6.1 Step-by-Step Polynomial Kernel Computation")
print("-" * 50)

x1 = np.array([1, 2])
x2 = np.array([3, 4])

print(f"Point 1: x₁ = {x1}")
print(f"Point 2: x₂ = {x2}")

# Step 1: Compute dot product
dot_product = np.dot(x1, x2)
print(f"\nStep 1: x₁^T x₂ = {x1[0]}×{x2[0]} + {x1[1]}×{x2[1]} = {dot_product}")

# Step 2: Add 1
sum_with_one = dot_product + 1
print(f"Step 2: x₁^T x₂ + 1 = {dot_product} + 1 = {sum_with_one}")

# Step 3: Square
kernel_result = sum_with_one**2
print(f"Step 3: (x₁^T x₂ + 1)² = {sum_with_one}² = {kernel_result}")

# Compare with explicit mapping
print(f"\n6.2 Comparison with Explicit Mapping")
print("-" * 40)

# Explicit mapping
phi1 = explicit_polynomial_mapping(x1.reshape(1, -1)).flatten()
phi2 = explicit_polynomial_mapping(x2.reshape(1, -1)).flatten()

print(f"φ(x₁) = {phi1}")
print(f"φ(x₂) = {phi2}")
print(f"φ(x₁)^T φ(x₂) = {np.dot(phi1, phi2)}")

print(f"\nKernel trick result: {kernel_result}")
print(f"Explicit mapping result: {np.dot(phi1, phi2)}")
print(f"Difference: {abs(kernel_result - np.dot(phi1, phi2)):.10f}")

# ============================================================================
# PART 7: Computational Complexity Visualization
# ============================================================================

print(f"\n\n7. COMPUTATIONAL COMPLEXITY VISUALIZATION")
print("=" * 60)

# Measure time for different dataset sizes
sizes = np.array([50, 100, 200, 500, 1000])
times_explicit = []
times_kernel = []

for size in sizes:
    X_test = X[:size]
    
    # Time explicit mapping
    start_time = time.time()
    X_mapped = explicit_polynomial_mapping(X_test)
    gram_explicit = X_mapped @ X_mapped.T
    time_explicit = time.time() - start_time
    times_explicit.append(time_explicit)
    
    # Time kernel trick
    start_time = time.time()
    gram_kernel = polynomial_kernel(X_test, degree=2, coef0=1)
    time_kernel = time.time() - start_time
    times_kernel.append(time_kernel)

# Plot computational complexity
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(sizes, times_explicit, 'ro-', label='Explicit Mapping', linewidth=2, markersize=8)
plt.plot(sizes, times_kernel, 'bs-', label='Kernel Trick', linewidth=2, markersize=8)
plt.xlabel('Dataset Size')
plt.ylabel('Time (seconds)')
plt.title('Computational Time Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.subplot(1, 2, 2)
speedup = np.array(times_explicit) / np.array(times_kernel)
plt.plot(sizes, speedup, 'go-', linewidth=2, markersize=8)
plt.xlabel('Dataset Size')
plt.ylabel('Speedup Factor')
plt.title('Kernel Trick Speedup')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'computational_complexity.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# PART 8: Summary and Key Insights
# ============================================================================

print(f"\n\n8. SUMMARY AND KEY INSIGHTS")
print("=" * 60)

print("\n8.1 Why it's called a 'trick'")
print("-" * 35)
print("• The kernel trick is called a 'trick' because it allows us to work")
print("  in high-dimensional spaces without explicitly computing the mapping")
print("• Instead of computing φ(x) and then φ(x)^T φ(z), we compute K(x,z) directly")
print("• This is computationally much more efficient")

print(f"\n8.2 Key Advantages of the Kernel Trick")
print("-" * 40)
print("1. Computational Efficiency:")
print("   • Avoid explicit feature mapping")
print("   • Work with infinite-dimensional spaces")
print("   • Reduced memory requirements")

print("\n2. Flexibility:")
print("   • Easy to try different kernels")
print("   • No need to design explicit mappings")
print("   • Can work with non-vector data (strings, graphs, etc.)")

print("\n3. Theoretical Benefits:")
print("   • Mercer's theorem guarantees valid kernels")
print("   • Reproducing Kernel Hilbert Space (RKHS) theory")
print("   • Optimality guarantees")

print(f"\n8.3 Computational Complexity Summary")
print("-" * 45)
print("For polynomial kernel (x^T z + 1)^d:")
print(f"• Explicit mapping: O(n × C(d+2,2)) where C(d+2,2) = {(2+2)*(2+1)//2}")
print("• Kernel trick: O(n × d)")
print(f"• Speedup: ~{np.mean(speedup):.1f}x faster")

print(f"\nPlots saved to: {save_dir}")
print("\n=== End of Kernel Trick Demonstration ===")
