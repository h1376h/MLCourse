import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import eigvals
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_3_Quiz_32")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
# Configure LaTeX to handle Unicode characters
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'

print("=" * 80)
print("QUESTION 32: COMPREHENSIVE KERNEL METHODS")
print("=" * 80)

# ============================================================================
# TASK 1 & 2: Kernel Properties - Sum and Product of Valid Kernels
# ============================================================================

print("\n" + "="*60)
print("TASK 1 & 2: KERNEL PROPERTIES")
print("="*60)

def linear_kernel(x1, x2):
    """Linear kernel: k(x1, x2) = x1^T x2"""
    return np.dot(x1, x2)

def polynomial_kernel(x1, x2, degree=2, c=1):
    """Polynomial kernel: k(x1, x2) = (x1^T x2 + c)^d"""
    return (np.dot(x1, x2) + c) ** degree

def rbf_kernel(x1, x2, gamma=0.5):
    """RBF kernel: k(x1, x2) = exp(-gamma * ||x1 - x2||^2)"""
    diff = x1 - x2
    return np.exp(-gamma * np.dot(diff, diff))

def sum_kernel(x1, x2, k1, k2):
    """Sum of two kernels: k(x1, x2) = k1(x1, x2) + k2(x1, x2)"""
    return k1(x1, x2) + k2(x1, x2)

def product_kernel(x1, x2, k1, k2):
    """Product of two kernels: k(x1, x2) = k1(x1, x2) * k2(x1, x2)"""
    return k1(x1, x2) * k2(x1, x2)

# Test points
x1 = np.array([1, 2])
x2 = np.array([3, 0])
x3 = np.array([0, 1])

print(f"Test points:")
print(f"x1 = {x1}")
print(f"x2 = {x2}")
print(f"x3 = {x3}")

# ============================================================================
# DETAILED STEP-BY-STEP CALCULATIONS
# ============================================================================

print(f"\n" + "="*60)
print("DETAILED STEP-BY-STEP CALCULATIONS")
print("="*60)

print(f"\n1. LINEAR KERNEL CALCULATIONS:")
print(f"   k1(x1, x2) = x1^T x2")
print(f"   x1 = [{x1[0]}, {x1[1]}]")
print(f"   x2 = [{x2[0]}, {x2[1]}]")
print(f"   k1(x1, x2) = {x1[0]} × {x2[0]} + {x1[1]} × {x2[1]}")
print(f"   k1(x1, x2) = {x1[0] * x2[0]} + {x1[1] * x2[1]} = {linear_kernel(x1, x2)}")

print(f"\n   k1(x1, x1) = x1^T x1")
print(f"   k1(x1, x1) = {x1[0]}² + {x1[1]}² = {x1[0]**2} + {x1[1]**2} = {linear_kernel(x1, x1)}")

print(f"\n   k1(x2, x2) = x2^T x2")
print(f"   k1(x2, x2) = {x2[0]}² + {x2[1]}² = {x2[0]**2} + {x2[1]**2} = {linear_kernel(x2, x2)}")

print(f"\n   k1(x3, x3) = x3^T x3")
print(f"   k1(x3, x3) = {x3[0]}² + {x3[1]}² = {x3[0]**2} + {x3[1]**2} = {linear_kernel(x3, x3)}")

print(f"\n2. POLYNOMIAL KERNEL CALCULATIONS:")
print(f"   k2(x1, x2) = (x1^T x2 + c)^d where c=1, d=2")
print(f"   k2(x1, x2) = (k1(x1, x2) + 1)²")
print(f"   k2(x1, x2) = ({linear_kernel(x1, x2)} + 1)² = {linear_kernel(x1, x2) + 1}² = {polynomial_kernel(x1, x2)}")

print(f"\n   k2(x1, x1) = (k1(x1, x1) + 1)²")
print(f"   k2(x1, x1) = ({linear_kernel(x1, x1)} + 1)² = {linear_kernel(x1, x1) + 1}² = {polynomial_kernel(x1, x1)}")

print(f"\n   k2(x2, x2) = (k1(x2, x2) + 1)²")
print(f"   k2(x2, x2) = ({linear_kernel(x2, x2)} + 1)² = {linear_kernel(x2, x2) + 1}² = {polynomial_kernel(x2, x2)}")

print(f"\n   k2(x3, x3) = (k1(x3, x3) + 1)²")
print(f"   k2(x3, x3) = ({linear_kernel(x3, x3)} + 1)² = {linear_kernel(x3, x3) + 1}² = {polynomial_kernel(x3, x3)}")

print(f"\n3. RBF KERNEL CALCULATIONS:")
print(f"   k3(x1, x2) = exp(-γ ||x1 - x2||²) where γ = 0.5")
print(f"   ||x1 - x2||² = ||[{x1[0]}, {x1[1]}] - [{x2[0]}, {x2[1]}]||²")
print(f"   ||x1 - x2||² = ||[{x1[0] - x2[0]}, {x1[1] - x2[1]}]||²")
print(f"   ||x1 - x2||² = ||[{x1[0] - x2[0]}, {x1[1] - x2[1]}]||² = ({x1[0] - x2[0]})² + ({x1[1] - x2[1]})²")
print(f"   ||x1 - x2||² = ({x1[0] - x2[0]})² + ({x1[1] - x2[1]})² = {x1[0] - x2[0]}² + {x1[1] - x2[1]}²")
print(f"   ||x1 - x2||² = {(x1[0] - x2[0])**2} + {(x1[1] - x2[1])**2} = {(x1[0] - x2[0])**2 + (x1[1] - x2[1])**2}")
print(f"   k3(x1, x2) = exp(-0.5 × {(x1[0] - x2[0])**2 + (x1[1] - x2[1])**2}) = exp({-0.5 * ((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)}) = {rbf_kernel(x1, x2):.6f}")

print(f"\n   k3(x1, x1) = exp(-0.5 × ||x1 - x1||²) = exp(-0.5 × 0) = exp(0) = 1")

print(f"\n4. SUM KERNEL CALCULATIONS:")
print(f"   k_sum(x1, x2) = k1(x1, x2) + k2(x1, x2)")
print(f"   k_sum(x1, x2) = {linear_kernel(x1, x2)} + {polynomial_kernel(x1, x2)} = {linear_kernel(x1, x2) + polynomial_kernel(x1, x2)}")

print(f"\n5. PRODUCT KERNEL CALCULATIONS:")
print(f"   k_prod(x1, x2) = k1(x1, x2) × k2(x1, x2)")
print(f"   k_prod(x1, x2) = {linear_kernel(x1, x2)} × {polynomial_kernel(x1, x2)} = {linear_kernel(x1, x2) * polynomial_kernel(x1, x2)}")

# Demonstrate kernel properties
print(f"\n" + "="*60)
print("KERNEL MATRIX CONSTRUCTION")
print("="*60)

print(f"\n1. Kernel Values Summary:")
print(f"Linear kernel k1(x1, x2) = {linear_kernel(x1, x2):.3f}")
print(f"Polynomial kernel k2(x1, x2) = {polynomial_kernel(x1, x2):.3f}")
print(f"RBF kernel k3(x1, x2) = {rbf_kernel(x1, x2):.3f}")

print(f"\n2. Sum of kernels k1 + k2:")
sum_val = sum_kernel(x1, x2, linear_kernel, lambda x1, x2: polynomial_kernel(x1, x2))
print(f"k1(x1, x2) + k2(x1, x2) = {linear_kernel(x1, x2):.3f} + {polynomial_kernel(x1, x2):.3f} = {sum_val:.3f}")

print(f"\n3. Product of kernels k1 * k2:")
prod_val = product_kernel(x1, x2, linear_kernel, lambda x1, x2: polynomial_kernel(x1, x2))
print(f"k1(x1, x2) * k2(x1, x2) = {linear_kernel(x1, x2):.3f} * {polynomial_kernel(x1, x2):.3f} = {prod_val:.3f}")

# Create kernel matrices to demonstrate positive semi-definiteness
def create_kernel_matrix(points, kernel_func):
    """Create kernel matrix for given points and kernel function"""
    n = len(points)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel_func(points[i], points[j])
    return K

points = [x1, x2, x3]

# Original kernels
K1 = create_kernel_matrix(points, linear_kernel)
K2 = create_kernel_matrix(points, lambda x1, x2: polynomial_kernel(x1, x2))

# Sum and product kernels
K_sum = create_kernel_matrix(points, lambda x1, x2: sum_kernel(x1, x2, linear_kernel, lambda x1, x2: polynomial_kernel(x1, x2)))
K_prod = create_kernel_matrix(points, lambda x1, x2: product_kernel(x1, x2, linear_kernel, lambda x1, x2: polynomial_kernel(x1, x2)))

print(f"\n4. Kernel Matrices:")
print(f"Linear kernel matrix K1:")
print(K1)
print(f"\nPolynomial kernel matrix K2:")
print(K2)
print(f"\nSum kernel matrix K1 + K2:")
print(K_sum)
print(f"\nProduct kernel matrix K1 * K2:")
print(K_prod)

# Check positive semi-definiteness
def check_psd(K, name):
    """Check if matrix is positive semi-definite"""
    eigenvals = eigvals(K)
    is_psd = np.all(eigenvals >= -1e-10)  # Small tolerance for numerical errors
    print(f"\n{name} eigenvalues: {eigenvals}")
    print(f"{name} is positive semi-definite: {is_psd}")
    return is_psd

check_psd(K1, "Linear kernel")
check_psd(K2, "Polynomial kernel")
check_psd(K_sum, "Sum kernel")
check_psd(K_prod, "Product kernel")

# ============================================================================
# TASK 3: Distance in Feature Space
# ============================================================================

print("\n" + "="*60)
print("TASK 3: DISTANCE IN FEATURE SPACE")
print("="*60)

def distance_in_feature_space(x1, x2, kernel_func):
    """Compute distance between points in feature space"""
    k11 = kernel_func(x1, x1)
    k22 = kernel_func(x2, x2)
    k12 = kernel_func(x1, x2)
    return np.sqrt(k11 + k22 - 2*k12)

print(f"\nDETAILED CALCULATION OF DISTANCE IN FEATURE SPACE:")
print(f"Formula: ||φ(x1) - φ(x2)||² = K(x1, x1) + K(x2, x2) - 2K(x1, x2)")

# Calculate individual terms
k11 = rbf_kernel(x1, x1, gamma=0.5)
k22 = rbf_kernel(x2, x2, gamma=0.5)
k12 = rbf_kernel(x1, x2, gamma=0.5)

print(f"\nStep 1: Calculate K(x1, x1)")
print(f"   K(x1, x1) = exp(-0.5 × ||x1 - x1||²)")
print(f"   ||x1 - x1||² = ||[{x1[0]}, {x1[1]}] - [{x1[0]}, {x1[1]}]||² = ||[0, 0]||² = 0")
print(f"   K(x1, x1) = exp(-0.5 × 0) = exp(0) = 1")

print(f"\nStep 2: Calculate K(x2, x2)")
print(f"   K(x2, x2) = exp(-0.5 × ||x2 - x2||²)")
print(f"   ||x2 - x2||² = ||[{x2[0]}, {x2[1]}] - [{x2[0]}, {x2[1]}]||² = ||[0, 0]||² = 0")
print(f"   K(x2, x2) = exp(-0.5 × 0) = exp(0) = 1")

print(f"\nStep 3: Calculate K(x1, x2)")
print(f"   K(x1, x2) = exp(-0.5 × ||x1 - x2||²)")
print(f"   ||x1 - x2||² = ||[{x1[0]}, {x1[1]}] - [{x2[0]}, {x2[1]}]||²")
print(f"   ||x1 - x2||² = ||[{x1[0] - x2[0]}, {x1[1] - x2[1]}]||²")
print(f"   ||x1 - x2||² = ({x1[0] - x2[0]})² + ({x1[1] - x2[1]})²")
print(f"   ||x1 - x2||² = {x1[0] - x2[0]}² + {x1[1] - x2[1]}² = {(x1[0] - x2[0])**2} + {(x1[1] - x2[1])**2} = {(x1[0] - x2[0])**2 + (x1[1] - x2[1])**2}")
print(f"   K(x1, x2) = exp(-0.5 × {(x1[0] - x2[0])**2 + (x1[1] - x2[1])**2}) = {k12:.6f}")

print(f"\nStep 4: Calculate ||φ(x1) - φ(x2)||²")
print(f"   ||φ(x1) - φ(x2)||² = K(x1, x1) + K(x2, x2) - 2K(x1, x2)")
print(f"   ||φ(x1) - φ(x2)||² = {k11} + {k22} - 2 × {k12}")
print(f"   ||φ(x1) - φ(x2)||² = {k11} + {k22} - {2*k12}")
print(f"   ||φ(x1) - φ(x2)||² = {k11 + k22 - 2*k12:.6f}")

print(f"\nStep 5: Calculate ||φ(x1) - φ(x2)||")
rbf_dist = np.sqrt(k11 + k22 - 2*k12)
print(f"   ||φ(x1) - φ(x2)|| = √({k11 + k22 - 2*k12:.6f}) = {rbf_dist:.6f}")

print(f"\nStep 6: Verify the bound ||φ(x1) - φ(x2)||² ≤ 2")
distance_squared = rbf_dist**2
print(f"   ||φ(x1) - φ(x2)||² = {distance_squared:.6f}")
print(f"   Is {distance_squared:.6f} ≤ 2? {distance_squared <= 2}")

# Test with different gamma values
gamma_values = [0.1, 0.5, 1.0, 2.0]
print(f"\n" + "="*60)
print("TESTING WITH DIFFERENT γ VALUES")
print("="*60)

for gamma in gamma_values:
    print(f"\nFor γ = {gamma}:")
    k11_gamma = rbf_kernel(x1, x1, gamma=gamma)
    k22_gamma = rbf_kernel(x2, x2, gamma=gamma)
    k12_gamma = rbf_kernel(x1, x2, gamma=gamma)
    
    print(f"   K(x1, x1) = exp(-{gamma} × 0) = 1")
    print(f"   K(x2, x2) = exp(-{gamma} × 0) = 1")
    print(f"   K(x1, x2) = exp(-{gamma} × {(x1[0] - x2[0])**2 + (x1[1] - x2[1])**2}) = {k12_gamma:.6f}")
    
    dist_squared = k11_gamma + k22_gamma - 2*k12_gamma
    dist = np.sqrt(dist_squared)
    
    print(f"   ||φ(x1) - φ(x2)||² = 1 + 1 - 2 × {k12_gamma:.6f} = {dist_squared:.6f}")
    print(f"   ||φ(x1) - φ(x2)|| = √{dist_squared:.6f} = {dist:.6f}")
    print(f"   Bound check: {dist_squared:.6f} ≤ 2? {dist_squared <= 2}")

# ============================================================================
# TASK 4: RBF Kernel Values for Different Distances
# ============================================================================

print("\n" + "="*60)
print("TASK 4: RBF KERNEL VALUES FOR DIFFERENT DISTANCES")
print("="*60)

def rbf_kernel_distance(distance, gamma=0.5):
    """RBF kernel value for given distance"""
    return np.exp(-gamma * distance**2)

distances = [0, 1, 10, 100]
print(f"\nDETAILED CALCULATION OF RBF KERNEL VALUES:")
print(f"Formula: K(x_i, x_j) = exp(-γ × ||x_i - x_j||²) where γ = 0.5")

for dist in distances:
    print(f"\nFor distance ||x_i - x_j|| = {dist}:")
    print(f"   K(x_i, x_j) = exp(-0.5 × {dist}²)")
    print(f"   K(x_i, x_j) = exp(-0.5 × {dist**2})")
    print(f"   K(x_i, x_j) = exp({-0.5 * dist**2})")
    kernel_val = rbf_kernel_distance(dist)
    print(f"   K(x_i, x_j) = {kernel_val:.10f}")
    
    if dist == 0:
        print(f"   Note: When distance = 0, K(x_i, x_j) = exp(0) = 1 (maximum similarity)")
    elif dist == 1:
        print(f"   Note: When distance = 1, K(x_i, x_j) = exp(-0.5) ≈ 0.6065 (moderate similarity)")
    else:
        print(f"   Note: When distance = {dist}, K(x_i, x_j) ≈ 0 (negligible similarity)")

print(f"\n" + "="*60)
print("OBSERVATIONS:")
print("="*60)
print(f"1. K(x_i, x_j) = 1 when ||x_i - x_j|| = 0 (identical points)")
print(f"2. K(x_i, x_j) decreases exponentially as ||x_i - x_j|| increases")
print(f"3. For large distances (≥ 10), K(x_i, x_j) ≈ 0 (negligible similarity)")
print(f"4. This means distant points have almost no influence on each other in the feature space")

# Create visualization
plt.figure(figsize=(12, 8))

# Plot 1: RBF kernel values vs distance
plt.subplot(2, 2, 1)
dist_range = np.linspace(0, 10, 1000)
kernel_values = rbf_kernel_distance(dist_range)
plt.plot(dist_range, kernel_values, 'b-', linewidth=2)
plt.scatter(distances, [rbf_kernel_distance(d) for d in distances], 
           color='red', s=100, zorder=5, label='Test points')
plt.xlabel('Distance $\\|x_i - x_j\\|$')
plt.ylabel('RBF Kernel Value $K(x_i, x_j)$')
plt.title('RBF Kernel vs Distance')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 2: Log scale to show behavior for large distances
plt.subplot(2, 2, 2)
dist_range_log = np.logspace(-1, 3, 1000)
kernel_values_log = rbf_kernel_distance(dist_range_log)
plt.semilogx(dist_range_log, kernel_values_log, 'g-', linewidth=2)
plt.scatter(distances, [rbf_kernel_distance(d) for d in distances], 
           color='red', s=100, zorder=5, label='Test points')
plt.xlabel('Distance $\\|x_i - x_j\\|$ (log scale)')
plt.ylabel('RBF Kernel Value $K(x_i, x_j)$')
plt.title('RBF Kernel vs Distance (Log Scale)')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 3: Different gamma values
plt.subplot(2, 2, 3)
gamma_values = [0.1, 0.5, 1.0, 2.0]
colors = ['blue', 'green', 'orange', 'red']
for i, gamma in enumerate(gamma_values):
    kernel_vals = np.exp(-gamma * dist_range**2)
    plt.plot(dist_range, kernel_vals, color=colors[i], linewidth=2, 
             label=f'$\\gamma = {gamma}$')
plt.xlabel('Distance $\\|x_i - x_j\\|$')
plt.ylabel('RBF Kernel Value $K(x_i, x_j)$')
plt.title('RBF Kernel with Different $\\gamma$ Values')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 4: Feature space distance
plt.subplot(2, 2, 4)
feature_distances = []
for dist in dist_range:
    # For RBF kernel, distance in feature space is sqrt(2 - 2*exp(-γ*d²))
    feature_dist = np.sqrt(2 - 2*np.exp(-0.5*dist**2))
    feature_distances.append(feature_dist)

plt.plot(dist_range, feature_distances, 'purple', linewidth=2)
plt.axhline(y=np.sqrt(2), color='red', linestyle='--', 
            label='Upper bound $\\sqrt{2}$')
plt.xlabel('Distance in Input Space $\\|x_i - x_j\\|$')
plt.ylabel('Distance in Feature Space $\\|\\phi(x_i) - \\phi(x_j)\\|$')
plt.title('Distance in Feature Space vs Input Space')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'rbf_kernel_analysis.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# TASK 5: SVM Decision Function for Far Points
# ============================================================================

print("\n" + "="*60)
print("TASK 5: SVM DECISION FUNCTION FOR FAR POINTS")
print("="*60)

# Create synthetic training data
np.random.seed(42)
n_train = 50
X_train = np.random.randn(n_train, 2) * 2
y_train = np.sign(X_train[:, 0] + X_train[:, 1])  # Simple linear separation

# Support vectors (simplified - in practice these would come from SVM training)
support_vectors = X_train[:10]  # Assume first 10 are support vectors
support_labels = y_train[:10]
alphas = np.random.uniform(0.1, 0.5, 10)  # Dual coefficients
b = 0.1  # Bias term

print(f"Number of support vectors: {len(support_vectors)}")
print(f"Support vector alphas: {alphas}")
print(f"Bias term b: {b}")

def svm_decision_function(x, support_vectors, support_labels, alphas, b, kernel_func):
    """Compute SVM decision function"""
    decision = b
    for i, (sv, label, alpha) in enumerate(zip(support_vectors, support_labels, alphas)):
        kernel_val = kernel_func(sv, x)
        decision += alpha * label * kernel_val
    return decision

# Test points at different distances
test_points = [
    np.array([0, 0]),      # Near training data
    np.array([10, 10]),    # Far from training data
    np.array([50, 50]),    # Very far from training data
    np.array([100, 100]),  # Extremely far from training data
]

print(f"\n" + "="*60)
print("DETAILED CALCULATION OF SVM DECISION FUNCTION")
print("="*60)
print(f"Formula: f(x) = Σ(α_i × y_i × K(x_i, x)) + b")

for i, test_point in enumerate(test_points):
    print(f"\nTest point {i+1}: x = {test_point}")
    print(f"   f(x) = b + Σ(α_i × y_i × K(x_i, x))")
    print(f"   f(x) = {b} + Σ(α_i × y_i × K(x_i, x))")
    
    # Calculate minimum distance to any support vector
    min_dist = min([np.linalg.norm(test_point - sv) for sv in support_vectors])
    print(f"   Minimum distance to any support vector: {min_dist:.2f}")
    
    # Calculate individual contributions
    total_kernel_contribution = 0
    print(f"   Individual contributions:")
    
    for j, (sv, label, alpha) in enumerate(zip(support_vectors, support_labels, alphas)):
        kernel_val = rbf_kernel(sv, test_point)
        contribution = alpha * label * kernel_val
        total_kernel_contribution += contribution
        dist_to_sv = np.linalg.norm(test_point - sv)
        
        print(f"     SV {j+1}: α_{j+1} = {alpha:.3f}, y_{j+1} = {label}, K(x_{j+1}, x) = {kernel_val:.6f}")
        print(f"           Contribution = {alpha:.3f} × {label} × {kernel_val:.6f} = {contribution:.6f}")
        print(f"           Distance to SV {j+1}: {dist_to_sv:.2f}")
    
    print(f"   Total kernel contribution: Σ(α_i × y_i × K(x_i, x)) = {total_kernel_contribution:.6f}")
    
    decision_val = b + total_kernel_contribution
    print(f"   Final decision: f(x) = {b} + {total_kernel_contribution:.6f} = {decision_val:.6f}")
    
    if i == 0:
        print(f"   Note: Near training data - significant kernel contributions")
    else:
        print(f"   Note: Far from training data - kernel contributions ≈ 0, f(x) ≈ b = {b}")

print(f"\n" + "="*60)
print("SUMMARY OF RESULTS:")
print("="*60)
print(f"1. For test points near training data: f(x) ≠ b (significant kernel contributions)")
print(f"2. For test points far from training data: f(x) ≈ b (negligible kernel contributions)")
print(f"3. This confirms that f(y_far) ≈ b for far test points")
print(f"4. The convergence to b happens because K(x_i, y_far) ≈ 0 for all support vectors")

# Create visualization
plt.figure(figsize=(15, 10))

# Plot 1: Training data and support vectors
plt.subplot(2, 3, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu', alpha=0.6, s=30)
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], c='red', s=100, 
           marker='s', edgecolor='black', linewidth=2, label='Support Vectors')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Training Data and Support Vectors')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Decision function values vs distance
plt.subplot(2, 3, 2)
distances = []
decision_values = []
for test_point in test_points:
    # Calculate minimum distance to any support vector
    min_dist = min([np.linalg.norm(test_point - sv) for sv in support_vectors])
    distances.append(min_dist)
    decision_val = svm_decision_function(test_point, support_vectors, support_labels, 
                                       alphas, b, lambda x1, x2: rbf_kernel(x1, x2))
    decision_values.append(decision_val)

plt.plot(distances, decision_values, 'bo-', linewidth=2, markersize=8)
plt.axhline(y=b, color='red', linestyle='--', label=f'Bias b = {b:.3f}')
plt.xlabel('Minimum Distance to Support Vectors')
plt.ylabel('Decision Function Value f(x)')
plt.title('Decision Function vs Distance')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Kernel values vs distance
plt.subplot(2, 3, 3)
kernel_values = []
for dist in distances:
    kernel_val = rbf_kernel_distance(dist)
    kernel_values.append(kernel_val)

plt.plot(distances, kernel_values, 'go-', linewidth=2, markersize=8)
plt.xlabel('Distance to Support Vectors')
plt.ylabel('RBF Kernel Value')
plt.title('Kernel Values vs Distance')
plt.grid(True, alpha=0.3)

# Plot 4: Decision function components
plt.subplot(2, 3, 4)
components = []
for test_point in test_points:
    component_sum = 0
    for sv, label, alpha in zip(support_vectors, support_labels, alphas):
        kernel_val = rbf_kernel(sv, test_point)
        component_sum += alpha * label * kernel_val
    components.append(component_sum)

plt.plot(distances, components, 'mo-', linewidth=2, markersize=8)
plt.axhline(y=0, color='red', linestyle='--', label='Zero line')
plt.xlabel('Distance to Support Vectors')
plt.ylabel('Kernel Component Sum')
plt.title('Kernel Component vs Distance')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 5: Decision boundary visualization
plt.subplot(2, 3, 5)
x1_range = np.linspace(-5, 5, 50)
x2_range = np.linspace(-5, 5, 50)
X1, X2 = np.meshgrid(x1_range, x2_range)
Z = np.zeros_like(X1)

for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        test_point = np.array([X1[i, j], X2[i, j]])
        Z[i, j] = svm_decision_function(test_point, support_vectors, support_labels, 
                                      alphas, b, lambda x1, x2: rbf_kernel(x1, x2))

plt.contour(X1, X2, Z, levels=[0], colors='red', linewidths=2)
plt.contourf(X1, X2, Z, levels=20, cmap='RdYlBu', alpha=0.3)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu', alpha=0.6, s=30)
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], c='red', s=100, 
           marker='s', edgecolor='black', linewidth=2)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('SVM Decision Boundary')
plt.grid(True, alpha=0.3)

# Plot 6: Convergence to bias
plt.subplot(2, 3, 6)
plt.plot(distances, np.abs(np.array(decision_values) - b), 'co-', linewidth=2, markersize=8)
plt.xlabel('Distance to Support Vectors')
plt.ylabel('|f(x) - b|')
plt.title('Convergence to Bias Term')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'svm_decision_function_analysis.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# ADDITIONAL SIMPLE INFORMATIVE VISUALIZATION
# ============================================================================

print("\n" + "="*60)
print("ADDITIONAL VISUALIZATION: KERNEL SIMILARITY HEATMAP")
print("="*60)

# Create a simple heatmap showing kernel similarities
plt.figure(figsize=(10, 8))

# Generate a grid of points
x_range = np.linspace(-3, 3, 50)
y_range = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x_range, y_range)

# Reference point
ref_point = np.array([0, 0])

# Calculate RBF kernel values for each point with respect to reference point
kernel_values = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = np.array([X[i, j], Y[i, j]])
        kernel_values[i, j] = rbf_kernel(ref_point, point, gamma=0.5)

# Create the heatmap
plt.imshow(kernel_values, extent=[-3, 3, -3, 3], origin='lower', 
           cmap='viridis', aspect='equal')
plt.colorbar(label='RBF Kernel Value')

# Add reference point
plt.plot(0, 0, 'r*', markersize=15, label='Reference Point', markeredgecolor='white', markeredgewidth=2)

# Add contour lines for specific kernel values
contour_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
contours = plt.contour(X, Y, kernel_values, levels=contour_levels, colors='white', linewidths=1, alpha=0.7)
plt.clabel(contours, inline=True, fontsize=8, fmt='%.1f')

plt.title('RBF Kernel Similarity Heatmap\n(Reference Point at Origin)')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'rbf_kernel_heatmap.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# MATHEMATICAL PROOFS AND INSIGHTS
# ============================================================================

print("\n" + "="*60)
print("MATHEMATICAL PROOFS AND INSIGHTS")
print("="*60)

print("\n1. PROOF: Sum of kernels is a valid kernel")
print("If k1 and k2 are valid kernels, then k(x,x') = k1(x,x') + k2(x,x') is also valid.")
print("Proof: The sum of two positive semi-definite matrices is positive semi-definite.")
print("Therefore, the kernel matrix K = K1 + K2 is positive semi-definite.")

print("\n2. PROOF: Product of kernels is a valid kernel")
print("If k1 and k2 are valid kernels, then k(x,x') = k1(x,x') * k2(x,x') is also valid.")
print("Proof: The element-wise product (Hadamard product) of two positive semi-definite")
print("matrices is positive semi-definite.")

print("\n3. PROOF: ||φ(x_i) - φ(x_j)||^2 ≤ 2 for RBF kernel")
print("For RBF kernel K(x_i, x_j) = exp(-γ||x_i - x_j||^2):")
print("||φ(x_i) - φ(x_j)||^2 = K(x_i, x_i) + K(x_j, x_j) - 2K(x_i, x_j)")
print("                      = 1 + 1 - 2exp(-γ||x_i - x_j||^2)")
print("                      = 2(1 - exp(-γ||x_i - x_j||^2))")
print("Since exp(-γ||x_i - x_j||^2) ≥ 0, we have ||φ(x_i) - φ(x_j)||^2 ≤ 2")

print("\n4. OBSERVATION: RBF kernel values approach 0 for large distances")
print("As ||x_i - x_j|| → ∞, K(x_i, x_j) = exp(-γ||x_i - x_j||^2) → 0")
print("This means distant points have negligible influence on the decision function.")

print("\n5. PROOF: f(y_far) ≈ b for far test points")
print("For a test point y_far far from all training points:")
print("f(y_far) = Σ(α_i * y_i * K(x_i, y_far)) + b")
print("Since K(x_i, y_far) ≈ 0 for all support vectors x_i:")
print("f(y_far) ≈ 0 + b = b")

print(f"\nPlots saved to: {save_dir}")
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
