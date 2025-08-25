import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_3_Quiz_33")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'

print("=" * 80)
print("QUESTION 33: SVM Decision Boundary with RBF Kernel")
print("=" * 80)

# ============================================================================
# TASK 1: Mathematical Proof that f(x_far; α, b) ≈ b for distant points
# ============================================================================

print("\n" + "="*60)
print("TASK 1: Mathematical Proof")
print("="*60)

print("""
For a distant test point x_far, we need to prove that f(x_far; α, b) ≈ b.

The SVM decision function is:
f(x; α, b) = Σ_{i ∈ SV} y^i α^i K(x^i, x) + b

For the RBF kernel: K(x_i, x_j) = exp(-(1/2) ||x_i - x_j||²)

When x_far is far from all training points:
||x_far - x^i||² is very large for all support vectors i

Therefore:
K(x^i, x_far) = exp(-(1/2) ||x^i - x_far||²) ≈ 0

This means:
Σ_{i ∈ SV} y^i α^i K(x^i, x_far) ≈ 0

Therefore:
f(x_far; α, b) ≈ 0 + b = b

QED: f(x_far; α, b) ≈ b for distant test points.
""")

# ============================================================================
# TASK 2: Compute RBF kernel values for different distances
# ============================================================================

print("\n" + "="*60)
print("TASK 2: RBF Kernel Values for Different Distances")
print("="*60)

def rbf_kernel(x1, x2, sigma=1.0):
    """Compute RBF kernel between two points"""
    distance_squared = np.sum((np.array(x1) - np.array(x2))**2)
    return np.exp(-(1/(2*sigma**2)) * distance_squared)

# Test points
x_i = np.array([0, 0])
test_points = [
    np.array([1, 0]),   # distance = 1
    np.array([2, 0]),   # distance = 2
    np.array([10, 0]),  # distance = 10
]

print("RBF Kernel values for different distances:")
print(f"Reference point: x_i = {x_i}")
print("-" * 50)

for i, x_j in enumerate(test_points):
    distance = np.linalg.norm(x_i - x_j)
    kernel_value = rbf_kernel(x_i, x_j)
    print(f"x_j = {x_j}, distance = {distance:.1f}, K(x_i, x_j) = {kernel_value:.6f}")

# Create visualization of kernel decay
distances = np.linspace(0, 10, 1000)
kernel_values = [rbf_kernel([0, 0], [d, 0]) for d in distances]

plt.figure(figsize=(12, 8))

# Plot 1: Kernel decay with distance
plt.subplot(2, 2, 1)
plt.plot(distances, kernel_values, 'b-', linewidth=2)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel(r'Distance $\|\mathbf{x}_i - \mathbf{x}_j\|$')
plt.ylabel(r'RBF Kernel Value $K(\mathbf{x}_i, \mathbf{x}_j)$')
plt.title(r'RBF Kernel Decay with Distance')
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.ylim(1e-6, 1)

# Mark specific points
for x_j in test_points:
    distance = np.linalg.norm(x_i - x_j)
    kernel_val = rbf_kernel(x_i, x_j)
    plt.plot(distance, kernel_val, 'ro', markersize=8)
    plt.annotate(f'({distance}, {kernel_val:.4f})', 
                (distance, kernel_val), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

# Plot 2: Linear scale for better visualization of small values
plt.subplot(2, 2, 2)
plt.plot(distances, kernel_values, 'b-', linewidth=2)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel(r'Distance $\|\mathbf{x}_i - \mathbf{x}_j\|$')
plt.ylabel(r'RBF Kernel Value $K(\mathbf{x}_i, \mathbf{x}_j)$')
plt.title(r'RBF Kernel Decay (Linear Scale)')
plt.grid(True, alpha=0.3)
plt.xlim(0, 5)
plt.ylim(0, 1)

# Mark specific points
for x_j in test_points:
    distance = np.linalg.norm(x_i - x_j)
    kernel_val = rbf_kernel(x_i, x_j)
    plt.plot(distance, kernel_val, 'ro', markersize=8)
    plt.annotate(f'({distance}, {kernel_val:.4f})', 
                (distance, kernel_val), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

# Plot 3: 2D visualization of kernel values around a point
plt.subplot(2, 2, 3)
x_range = np.linspace(-3, 3, 100)
y_range = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = np.array([X[i, j], Y[i, j]])
        Z[i, j] = rbf_kernel(x_i, point)

contour = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar(contour, label='Kernel Value')
plt.plot(x_i[0], x_i[1], 'ro', markersize=10, label='Reference Point')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title('RBF Kernel Values Around Reference Point')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: 3D surface plot
ax = plt.subplot(2, 2, 4, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.scatter([x_i[0]], [x_i[1]], [1], color='red', s=100, label='Reference Point')
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_zlabel('Kernel Value')
ax.set_title('RBF Kernel Surface')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'rbf_kernel_analysis.png'), dpi=300, bbox_inches='tight')

print("\nPattern observed:")
print("As distance increases, the RBF kernel value decreases exponentially.")
print("For large distances (e.g., distance = 10), the kernel value is very close to 0.")
print("This confirms that distant points contribute negligibly to the decision function.")

# ============================================================================
# TASK 3: Explanation of why f(x_far; α, b) ≈ b
# ============================================================================

print("\n" + "="*60)
print("TASK 3: Explanation of f(x_far; α, b) ≈ b")
print("="*60)

print("""
Based on our observations from Task 2:

1. The RBF kernel K(x_i, x_j) = exp(-(1/2) ||x_i - x_j||²) decreases exponentially with distance.

2. For large distances (e.g., ||x_far - x^i|| = 10), the kernel value becomes extremely small:
   K(x^i, x_far) ≈ 0

3. In the SVM decision function:
   f(x_far; α, b) = Σ_{i ∈ SV} y^i α^i K(x^i, x_far) + b

4. When x_far is far from all support vectors, all kernel terms K(x^i, x_far) ≈ 0

5. Therefore:
   f(x_far; α, b) ≈ Σ_{i ∈ SV} y^i α^i × 0 + b = b

This explains why distant test points are classified based primarily on the bias term b.
""")

# ============================================================================
# TASK 4: 1D Example with specific values
# ============================================================================

print("\n" + "="*60)
print("TASK 4: 1D Example with Specific Values")
print("="*60)

# Given parameters
x1, x2 = -1, 1  # Training points
y1, y2 = 1, 1   # Labels
alpha1, alpha2 = 0.5, 0.5  # Dual weights
b = 0  # Bias

# Test points
test_x = [5, 10]

print(f"Training points: x¹ = {x1}, x² = {x2}")
print(f"Labels: y¹ = {y1}, y² = {y2}")
print(f"Dual weights: α¹ = {alpha1}, α² = {alpha2}")
print(f"Bias: b = {b}")
print("-" * 50)

def svm_decision_function_1d(x, support_vectors, labels, alphas, bias):
    """Compute SVM decision function for 1D case"""
    result = bias
    for i, (sv, label, alpha) in enumerate(zip(support_vectors, labels, alphas)):
        kernel_val = rbf_kernel([sv], [x])
        result += label * alpha * kernel_val
    return result

support_vectors = [x1, x2]
labels = [y1, y2]
alphas = [alpha1, alpha2]

for test_point in test_x:
    f_value = svm_decision_function_1d(test_point, support_vectors, labels, alphas, b)
    
    print(f"\nFor test point x = {test_point}:")
    print(f"  Distance to x¹ = {x1}: ||{test_point} - {x1}|| = {abs(test_point - x1)}")
    print(f"  Distance to x² = {x2}: ||{test_point} - {x2}|| = {abs(test_point - x2)}")
    
    # Calculate individual terms
    kernel1 = rbf_kernel([x1], [test_point])
    kernel2 = rbf_kernel([x2], [test_point])
    term1 = y1 * alpha1 * kernel1
    term2 = y2 * alpha2 * kernel2
    
    print(f"  K(x¹, x) = {kernel1:.6f}")
    print(f"  K(x², x) = {kernel2:.6f}")
    print(f"  y¹α¹K(x¹, x) = {term1:.6f}")
    print(f"  y²α²K(x², x) = {term2:.6f}")
    print(f"  f({test_point}; α, b) = {term1:.6f} + {term2:.6f} + {b} = {f_value:.6f}")
    print(f"  Verification: f({test_point}; α, b) ≈ b = {b} ✓")

# Create visualization for 1D example
plt.figure(figsize=(15, 10))

# Plot 1: Decision function over range
x_range = np.linspace(-5, 15, 1000)
f_values = [svm_decision_function_1d(x, support_vectors, labels, alphas, b) for x in x_range]

plt.subplot(2, 2, 1)
plt.plot(x_range, f_values, 'b-', linewidth=2, label=r'Decision Function $f(\mathbf{x})$')
plt.axhline(y=b, color='r', linestyle='--', alpha=0.7, label=f'Bias b = {b}')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.scatter(support_vectors, [0, 0], color='green', s=100, zorder=5, label='Support Vectors')
plt.scatter(test_x, [0, 0], color='red', s=100, zorder=5, label='Test Points')
plt.xlabel(r'$\mathbf{x}$')
plt.ylabel(r'$f(\mathbf{x})$')
plt.title('SVM Decision Function (1D Example)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-5, 15)

# Mark test points with their f values
for test_point in test_x:
    f_val = svm_decision_function_1d(test_point, support_vectors, labels, alphas, b)
    plt.annotate(r'$f(' + str(test_point) + r') = ' + f'{f_val:.4f}' + r'$', 
                (test_point, f_val), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

# Plot 2: Kernel values for each support vector
plt.subplot(2, 2, 2)
for i, sv in enumerate(support_vectors):
    kernel_vals = [rbf_kernel([sv], [x]) for x in x_range]
    plt.plot(x_range, kernel_vals, label=r'$K(\mathbf{x}^{' + str(i+1) + r'}, \mathbf{x})$')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.scatter(support_vectors, [1, 1], color='green', s=100, zorder=5, label='Support Vectors')
plt.xlabel(r'$\mathbf{x}$')
plt.ylabel(r'Kernel Value')
plt.title('RBF Kernel Values for Each Support Vector')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-5, 15)

# Plot 3: Individual terms in the decision function
plt.subplot(2, 2, 3)
for i, (sv, label, alpha) in enumerate(zip(support_vectors, labels, alphas)):
    term_vals = [label * alpha * rbf_kernel([sv], [x]) for x in x_range]
    plt.plot(x_range, term_vals, label=r'$y^{' + str(i+1) + r'}\alpha^{' + str(i+1) + r'}K(\mathbf{x}^{' + str(i+1) + r'}, \mathbf{x})$')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.xlabel(r'$\mathbf{x}$')
plt.ylabel(r'Term Value')
plt.title('Individual Terms in Decision Function')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-5, 15)

# Plot 4: Sum of kernel terms vs bias
plt.subplot(2, 2, 4)
sum_terms = np.zeros_like(x_range)
for i, (sv, label, alpha) in enumerate(zip(support_vectors, labels, alphas)):
    term_vals = [label * alpha * rbf_kernel([sv], [x]) for x in x_range]
    sum_terms += np.array(term_vals)

plt.plot(x_range, sum_terms, 'g-', linewidth=2, label=r'$\sum_{i \in SV} y^i \alpha^i K(\mathbf{x}^i, \mathbf{x})$')
plt.axhline(y=b, color='r', linestyle='--', alpha=0.7, label=f'Bias b = {b}')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.xlabel(r'$\mathbf{x}$')
plt.ylabel(r'Value')
plt.title('Sum of Kernel Terms vs Bias')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-5, 15)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'svm_1d_example.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# TASK 5: Mathematical justification for infinite distance
# ============================================================================

print("\n" + "="*60)
print("TASK 5: Mathematical Justification for Infinite Distance")
print("="*60)

print("""
Mathematical justification for why f(x; α, b) approaches b as ||x - x^i|| → ∞:

1. For any support vector x^i, as ||x - x^i|| → ∞:
   lim_{||x - x^i|| → ∞} K(x^i, x) = lim_{||x - x^i|| → ∞} exp(-(1/2) ||x - x^i||²) = 0

2. This is because:
   - ||x - x^i||² → ∞ as ||x - x^i|| → ∞
   - exp(-(1/2) × ∞) = exp(-∞) = 0

3. Therefore, for the SVM decision function:
   f(x; α, b) = Σ_{i ∈ SV} y^i α^i K(x^i, x) + b

4. As ||x - x^i|| → ∞ for all support vectors:
   lim_{||x - x^i|| → ∞} f(x; α, b) = Σ_{i ∈ SV} y^i α^i × 0 + b = b

5. This means that very distant points are classified based solely on the bias term b.

This mathematical result explains the behavior we observed in our numerical examples.
""")

# Create a final visualization showing the asymptotic behavior
plt.figure(figsize=(12, 8))

# Generate points at various distances
distances = np.logspace(-1, 3, 1000)  # From 0.1 to 1000
f_values = []

for dist in distances:
    # Place test point at distance 'dist' from origin
    test_point = dist
    f_val = svm_decision_function_1d(test_point, support_vectors, labels, alphas, b)
    f_values.append(f_val)

plt.semilogx(distances, f_values, 'b-', linewidth=2, label=r'$f(\mathbf{x}; \boldsymbol{\alpha}, b)$')
plt.axhline(y=b, color='r', linestyle='--', alpha=0.7, label=f'Asymptote: b = {b}')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.xlabel(r'Distance from Support Vectors')
plt.ylabel(r'Decision Function Value')
plt.title('Asymptotic Behavior of SVM Decision Function')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-0.1, 0.1)

# Add annotations
plt.annotate(r'$f(\mathbf{x}) \to b$ as distance $\to \infty$', 
            xy=(100, b), xytext=(50, b + 0.05),
            arrowprops=dict(arrowstyle='->', lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'asymptotic_behavior.png'), dpi=300, bbox_inches='tight')

print(f"\nAll visualizations saved to: {save_dir}")
print("\n" + "="*80)
print("SOLUTION COMPLETE")
print("="*80)
