import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC, LinearSVC
from sklearn.datasets import make_classification, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time
import seaborn as sns
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_6_Quiz_23")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("=" * 80)
print("SVM PRIMAL vs DUAL FORMULATION ANALYSIS")
print("=" * 80)

# ============================================================================
# PART 1: UNDERSTANDING THE FORMULATIONS
# ============================================================================

print("\n1. UNDERSTANDING SVM FORMULATIONS")
print("-" * 50)

print("\n1.1 Primal Formulation:")
print("   Minimize: (1/2)||w||^2 + C*sum(xi_i)")
print("   Subject to: yi*(w^T*xi + b) >= 1 - xi_i, xi_i >= 0")
print("   Variables: w (d-dimensional), b (scalar), xi_i (n slack variables)")
print("   Total variables: d + 1 + n = d + n + 1")
print("   → Number of variables depends on: d (features) + n (samples)")

print("\n1.2 Dual Formulation:")
print("   Maximize: sum(alpha_i) - (1/2)*sum_sum(alpha_i*alpha_j*yi*yj*K(xi, xj))")
print("   Subject to: 0 <= alpha_i <= C, sum(alpha_i*yi) = 0")
print("   Variables: alpha_i (n Lagrange multipliers)")
print("   Total variables: n")
print("   → Number of variables depends on: n (samples)")

# ============================================================================
# PART 2: KERNEL TRICK DEMONSTRATION
# ============================================================================

print("\n\n2. KERNEL TRICK DEMONSTRATION")
print("-" * 50)

# Create non-linearly separable data
X, y = make_circles(n_samples=100, noise=0.1, factor=0.5, random_state=42)
X = StandardScaler().fit_transform(X)

print(f"\n2.1 Dataset Characteristics:")
print(f"   - Number of samples (n): {X.shape[0]}")
print(f"   - Number of features (d): {X.shape[1]}")
print(f"   - Data is non-linearly separable (circles)")

# Visualize original data
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='red', label='Class -1', alpha=0.7)
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='blue', label='Class +1', alpha=0.7)
plt.title('Original 2D Data')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.grid(True, alpha=0.3)

# Demonstrate feature transformation (polynomial kernel)
def polynomial_kernel_2d(X, degree=2):
    """Transform 2D data to higher dimensional space using polynomial kernel"""
    n_samples = X.shape[0]
    X_transformed = np.zeros((n_samples, 6))  # 6 features for degree 2
    
    for i in range(n_samples):
        x1, x2 = X[i, 0], X[i, 1]
        X_transformed[i] = [1, x1, x2, x1**2, x1*x2, x2**2]
    
    return X_transformed

X_transformed = polynomial_kernel_2d(X)

print(f"\n2.2 Feature Transformation:")
print(f"   - Original features: {X.shape[1]} (x1, x2)")
print(f"   - Transformed features: {X_transformed.shape[1]} (1, x1, x2, x1^2, x1*x2, x2^2)")
print(f"   - This demonstrates the 'kernel trick' - implicit mapping to higher dimensions")

# Visualize transformed data (first 3 dimensions)
plt.subplot(1, 3, 2)
plt.scatter(X_transformed[y == 0][:, 1], X_transformed[y == 0][:, 2], 
           c='red', label='Class -1', alpha=0.7)
plt.scatter(X_transformed[y == 1][:, 1], X_transformed[y == 1][:, 2], 
           c='blue', label='Class +1', alpha=0.7)
plt.title('Transformed Data ($x_1$, $x_2$)')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.grid(True, alpha=0.3)

# Show the quadratic terms
plt.subplot(1, 3, 3)
plt.scatter(X_transformed[y == 0][:, 3], X_transformed[y == 0][:, 5], 
           c='red', label='Class -1', alpha=0.7)
plt.scatter(X_transformed[y == 1][:, 3], X_transformed[y == 1][:, 5], 
           c='blue', label='Class +1', alpha=0.7)
plt.title('Transformed Data ($x_1^2$, $x_2^2$)')
plt.xlabel('$x_1^2$')
plt.ylabel('$x_2^2$')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'kernel_trick_demonstration.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n2.3 Why Dual Formulation is Essential for Kernel Trick:")
print(f"   - Primal: Requires explicit feature transformation phi(x)")
print(f"   - Dual: Uses kernel function K(xi, xj) = phi(xi)^T * phi(xj)")
print(f"   - Kernel trick avoids computing phi(x) explicitly")
print(f"   - Only inner products K(xi, xj) are needed in dual formulation")

# ============================================================================
# PART 3: COMPUTATIONAL COMPLEXITY ANALYSIS
# ============================================================================

print("\n\n3. COMPUTATIONAL COMPLEXITY ANALYSIS")
print("-" * 50)

# Create datasets with different characteristics
np.random.seed(42)

# Dataset 1: d >> n (many features, few samples)
n1, d1 = 50, 1000
X1 = np.random.randn(n1, d1)
y1 = np.random.choice([-1, 1], n1)

# Dataset 2: n >> d (many samples, few features)
n2, d2 = 10000, 10
X2 = np.random.randn(n2, d2)
y2 = np.random.choice([-1, 1], n2)

print(f"\n3.1 Dataset Characteristics:")
print(f"   Dataset 1 (d >> n): n={n1}, d={d1}, ratio d/n={d1/n1:.1f}")
print(f"   Dataset 2 (n >> d): n={n2}, d={d2}, ratio n/d={n2/d2:.1f}")

# Analyze computational complexity
def analyze_complexity(n, d, kernel_type='linear'):
    """Analyze computational complexity for different formulations"""
    
    print(f"\n   For {kernel_type} kernel with n={n}, d={d}:")
    
    # Primal formulation
    primal_vars = d + n + 1  # w (d) + ξ (n) + b (1)
    primal_complexity = f"O({d}³)" if d < n else f"O({n}³)"
    
    # Dual formulation
    dual_vars = n
    dual_complexity = f"O({n}³)"
    
    print(f"   Primal: {primal_vars} variables, complexity: {primal_complexity}")
    print(f"   Dual: {dual_vars} variables, complexity: {dual_complexity}")
    
    return primal_vars, dual_vars, primal_complexity, dual_complexity

print("\n3.2 Complexity Analysis:")
analyze_complexity(n1, d1, "linear")
analyze_complexity(n2, d2, "linear")

# ============================================================================
# PART 4: EMPIRICAL COMPARISON
# ============================================================================

print("\n\n4. EMPIRICAL COMPARISON")
print("-" * 50)

def compare_formulations(X, y, dataset_name):
    """Compare training time for different formulations"""
    
    print(f"\n4.1 {dataset_name}:")
    print(f"   n={X.shape[0]}, d={X.shape[1]}")
    
    results = {}
    
    # Linear SVM (can use both primal and dual)
    try:
        # Primal formulation (LinearSVC)
        start_time = time.time()
        svm_primal = LinearSVC(random_state=42, max_iter=1000)
        svm_primal.fit(X, y)
        primal_time = time.time() - start_time
        primal_acc = accuracy_score(y, svm_primal.predict(X))
        results['primal'] = (primal_time, primal_acc)
        print(f"   Primal (LinearSVC): {primal_time:.4f}s, accuracy: {primal_acc:.4f}")
    except Exception as e:
        print(f"   Primal failed: {e}")
        results['primal'] = (float('inf'), 0)
    
    try:
        # Dual formulation (SVC with linear kernel)
        start_time = time.time()
        svm_dual = SVC(kernel='linear', random_state=42, max_iter=1000)
        svm_dual.fit(X, y)
        dual_time = time.time() - start_time
        dual_acc = accuracy_score(y, svm_dual.predict(X))
        results['dual'] = (dual_time, dual_acc)
        print(f"   Dual (SVC linear): {dual_time:.4f}s, accuracy: {dual_acc:.4f}")
    except Exception as e:
        print(f"   Dual failed: {e}")
        results['dual'] = (float('inf'), 0)
    
    return results

# Test with different datasets
results1 = compare_formulations(X1, y1, "Dataset 1 (d >> n)")
results2 = compare_formulations(X2, y2, "Dataset 2 (n >> d)")

# ============================================================================
# PART 5: KERNEL COMPARISON
# ============================================================================

print("\n\n5. KERNEL COMPARISON")
print("-" * 50)

# Use the circle dataset for kernel comparison
X_circles, y_circles = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)
X_circles = StandardScaler().fit_transform(X_circles)

print(f"\n5.1 Non-linear Dataset (Circles):")
print(f"   n={X_circles.shape[0]}, d={X_circles.shape[1]}")

kernels = ['linear', 'rbf', 'poly']
kernel_results = {}

for kernel in kernels:
    try:
        start_time = time.time()
        svm = SVC(kernel=kernel, random_state=42, max_iter=1000)
        svm.fit(X_circles, y_circles)
        train_time = time.time() - start_time
        acc = accuracy_score(y_circles, svm.predict(X_circles))
        n_sv = len(svm.support_vectors_)
        kernel_results[kernel] = (train_time, acc, n_sv)
        print(f"   {kernel.upper()} kernel: {train_time:.4f}s, accuracy: {acc:.4f}, SVs: {n_sv}")
    except Exception as e:
        print(f"   {kernel.upper()} kernel failed: {e}")
        kernel_results[kernel] = (float('inf'), 0, 0)

# ============================================================================
# PART 6: VISUALIZATION OF DECISION BOUNDARIES
# ============================================================================

print("\n\n6. DECISION BOUNDARY VISUALIZATION")
print("-" * 50)

# Create mesh for visualization
x_min, x_max = X_circles[:, 0].min() - 0.5, X_circles[:, 0].max() + 0.5
y_min, y_max = X_circles[:, 1].min() - 0.5, X_circles[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

plt.figure(figsize=(15, 5))

for i, kernel in enumerate(['linear', 'rbf', 'poly']):
    plt.subplot(1, 3, i+1)
    
    # Train SVM
    svm = SVC(kernel=kernel, random_state=42)
    svm.fit(X_circles, y_circles)
    
    # Predict on mesh
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X_circles[y_circles == 0][:, 0], X_circles[y_circles == 0][:, 1], 
               c='red', label='Class -1', alpha=0.7)
    plt.scatter(X_circles[y_circles == 1][:, 0], X_circles[y_circles == 1][:, 1], 
               c='blue', label='Class +1', alpha=0.7)
    
    # Highlight support vectors
    plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
               s=100, linewidth=1, facecolors='none', edgecolors='black', 
               label=f'Support Vectors ({len(svm.support_vectors_)})')
    
    plt.title(f'{kernel.upper()} Kernel')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'decision_boundaries_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PART 7: SCALING ANALYSIS
# ============================================================================

print("\n\n7. SCALING ANALYSIS")
print("-" * 50)

# Test different dataset sizes
sizes = [100, 500, 1000, 2000]
scaling_results = {'primal': [], 'dual': []}

for n in sizes:
    d = 10  # Fixed number of features
    X_test = np.random.randn(n, d)
    y_test = np.random.choice([-1, 1], n)
    
    print(f"\n   Testing n={n}, d={d}:")
    
    # Primal
    try:
        start_time = time.time()
        svm_primal = LinearSVC(random_state=42, max_iter=1000)
        svm_primal.fit(X_test, y_test)
        primal_time = time.time() - start_time
        scaling_results['primal'].append((n, primal_time))
        print(f"   Primal: {primal_time:.4f}s")
    except:
        scaling_results['primal'].append((n, float('inf')))
        print(f"   Primal: failed")
    
    # Dual
    try:
        start_time = time.time()
        svm_dual = SVC(kernel='linear', random_state=42, max_iter=1000)
        svm_dual.fit(X_test, y_test)
        dual_time = time.time() - start_time
        scaling_results['dual'].append((n, dual_time))
        print(f"   Dual: {dual_time:.4f}s")
    except:
        scaling_results['dual'].append((n, float('inf')))
        print(f"   Dual: failed")

# Plot scaling results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
primal_data = np.array(scaling_results['primal'])
dual_data = np.array(scaling_results['dual'])

plt.plot(primal_data[:, 0], primal_data[:, 1], 'o-', label='Primal', linewidth=2)
plt.plot(dual_data[:, 0], dual_data[:, 1], 's-', label='Dual', linewidth=2)
plt.xlabel('Number of Samples ($n$)')
plt.ylabel('Training Time (s)')
plt.title('Scaling with Sample Size')
plt.legend()
plt.grid(True, alpha=0.3)

# Theoretical complexity comparison
plt.subplot(1, 2, 2)
n_range = np.linspace(100, 2000, 100)
primal_theoretical = n_range**3 / 1e6  # Normalized
dual_theoretical = n_range**3 / 1e6    # Same complexity for linear kernel

plt.plot(n_range, primal_theoretical, '--', label='Primal $O(n^3)$', alpha=0.7)
plt.plot(n_range, dual_theoretical, '--', label='Dual $O(n^3)$', alpha=0.7)
plt.xlabel('Number of Samples ($n$)')
plt.ylabel('Theoretical Complexity (normalized)')
plt.title('Theoretical Complexity')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'scaling_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PART 8: SUMMARY AND RECOMMENDATIONS
# ============================================================================

print("\n\n8. SUMMARY AND RECOMMENDATIONS")
print("-" * 50)

print("\n8.1 Key Findings:")
print("   - Primal variables: d + n + 1 (depends on features + samples)")
print("   - Dual variables: n (depends only on samples)")
print("   - Kernel trick requires dual formulation")
print("   - For d >> n: Dual is often more efficient")
print("   - For n >> d: Primal can be more efficient")
print("   - Non-linear kernels always use dual formulation")

print("\n8.2 Recommendations:")
print("   - Linear SVM, d >> n: Use dual formulation")
print("   - Linear SVM, n >> d: Consider primal formulation")
print("   - Non-linear kernels: Always use dual formulation")
print("   - Memory constraints: Consider primal for very large n")
print("   - Accuracy requirements: Dual often provides better results")

print(f"\nPlots saved to: {save_dir}")
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
