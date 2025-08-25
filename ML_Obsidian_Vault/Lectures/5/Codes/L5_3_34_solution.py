import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
from sklearn.datasets import make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_3_Quiz_34")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting with more compatible settings
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'

print("=" * 80)
print("QUESTION 34: KERNEL TRICK AND POLYNOMIAL KERNEL EXAMPLE")
print("=" * 80)

# ============================================================================
# PART 1: UNDERSTANDING THE POLYNOMIAL KERNEL
# ============================================================================

print("\n1. UNDERSTANDING THE POLYNOMIAL KERNEL")
print("-" * 50)

# Define the polynomial kernel function
def polynomial_kernel(t, s, degree=2):
    """Compute the polynomial kernel K(t, s) = (1 + t^T s)^degree"""
    return (1 + np.dot(t, s)) ** degree

# Define the explicit feature mapping
def explicit_feature_mapping(t):
    """Explicit feature mapping for degree-2 polynomial kernel"""
    t1, t2 = t
    return np.array([1, np.sqrt(2)*t1, np.sqrt(2)*t2, t1**2, t2**2, np.sqrt(2)*t1*t2])

# Test points
t = np.array([1, 2])
s = np.array([3, 1])

print(f"Test points:")
print(f"t = {t}")
print(f"s = {s}")

# Step 1: Compute kernel using kernel trick
print(f"\nStep 1: Compute kernel using kernel trick")
print(f"K(t, s) = (1 + t^T s)^2")
print(f"t^T s = {np.dot(t, s)}")
print(f"1 + t^T s = {1 + np.dot(t, s)}")
kernel_value = polynomial_kernel(t, s)
print(f"K(t, s) = ({1 + np.dot(t, s)})^2 = {kernel_value}")

# Step 2: Compute kernel using explicit feature mapping
print(f"\nStep 2: Compute kernel using explicit feature mapping")
print(f"φ(t) = (1, √2*t₁, √2*t₂, t₁², t₂², √2*t₁*t₂)")
phi_t = explicit_feature_mapping(t)
phi_s = explicit_feature_mapping(s)
print(f"φ(t) = {phi_t}")
print(f"φ(s) = {phi_s}")
print(f"φ(t)^T φ(s) = {np.dot(phi_t, phi_s)}")

# Verify they are equal
print(f"\nVerification:")
print(f"Kernel trick result: {kernel_value}")
print(f"Explicit mapping result: {np.dot(phi_t, phi_s)}")
print(f"Are they equal? {np.isclose(kernel_value, np.dot(phi_t, phi_s))}")

# ============================================================================
# PART 2: COMPUTATIONAL COMPLEXITY COMPARISON
# ============================================================================

print(f"\n\n2. COMPUTATIONAL COMPLEXITY COMPARISON")
print("-" * 50)

# Generate larger dataset for complexity comparison
n_samples = 1000
X = np.random.randn(n_samples, 2)
y = np.random.choice([-1, 1], n_samples)

print(f"Dataset size: {n_samples} samples, 2 features")

# Method 1: Explicit feature mapping
print(f"\nMethod 1: Explicit feature mapping")
start_time = time.time()

# Transform all data points
X_transformed = np.zeros((n_samples, 6))
for i in range(n_samples):
    X_transformed[i] = explicit_feature_mapping(X[i])

# Compute kernel matrix using explicit mapping
K_explicit = np.zeros((n_samples, n_samples))
for i in range(n_samples):
    for j in range(n_samples):
        K_explicit[i, j] = np.dot(X_transformed[i], X_transformed[j])

explicit_time = time.time() - start_time
print(f"Time for explicit mapping: {explicit_time:.4f} seconds")
print(f"Memory usage: {X_transformed.nbytes / 1024:.2f} KB for transformed data")

# Method 2: Kernel trick
print(f"\nMethod 2: Kernel trick")
start_time = time.time()

# Compute kernel matrix using kernel trick
K_kernel = np.zeros((n_samples, n_samples))
for i in range(n_samples):
    for j in range(n_samples):
        K_kernel[i, j] = polynomial_kernel(X[i], X[j])

kernel_time = time.time() - start_time
print(f"Time for kernel trick: {kernel_time:.4f} seconds")
print(f"Speedup: {explicit_time / kernel_time:.2f}x")

# Verify results are the same
print(f"Results are identical: {np.allclose(K_explicit, K_kernel)}")

# ============================================================================
# PART 3: VISUALIZATION OF FEATURE SPACE TRANSFORMATION
# ============================================================================

print(f"\n\n3. VISUALIZATION OF FEATURE SPACE TRANSFORMATION")
print("-" * 50)

# Create a simple dataset that's not linearly separable in 2D
X_vis, y_vis = make_circles(n_samples=100, noise=0.1, factor=0.5, random_state=42)

# Plot original data
plt.figure(figsize=(15, 5))

# Original space
plt.subplot(1, 3, 1)
plt.scatter(X_vis[y_vis == 0][:, 0], X_vis[y_vis == 0][:, 1], c='red', label='Class 0', alpha=0.7)
plt.scatter(X_vis[y_vis == 1][:, 0], X_vis[y_vis == 1][:, 1], c='blue', label='Class 1', alpha=0.7)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title(r'Original Feature Space')
plt.legend()
plt.grid(True, alpha=0.3)

# Transformed space (first 3 dimensions)
X_transformed_vis = np.array([explicit_feature_mapping(x) for x in X_vis])
plt.subplot(1, 3, 2)
plt.scatter(X_transformed_vis[y_vis == 0][:, 1], X_transformed_vis[y_vis == 0][:, 2], 
           c='red', label='Class 0', alpha=0.7)
plt.scatter(X_transformed_vis[y_vis == 1][:, 1], X_transformed_vis[y_vis == 1][:, 2], 
           c='blue', label='Class 1', alpha=0.7)
plt.xlabel(r'$\sqrt{2}x_1$')
plt.ylabel(r'$\sqrt{2}x_2$')
plt.title(r'Transformed Space (Dimensions 2-3)')
plt.legend()
plt.grid(True, alpha=0.3)

# Transformed space (dimensions 4-5)
plt.subplot(1, 3, 3)
plt.scatter(X_transformed_vis[y_vis == 0][:, 3], X_transformed_vis[y_vis == 0][:, 4], 
           c='red', label='Class 0', alpha=0.7)
plt.scatter(X_transformed_vis[y_vis == 1][:, 3], X_transformed_vis[y_vis == 1][:, 4], 
           c='blue', label='Class 1', alpha=0.7)
plt.xlabel('$x_1^2$')
plt.ylabel('$x_2^2$')
plt.title(r'Transformed Space (Dimensions 4-5)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_space_transformation.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# PART 4: SVM WITH KERNEL TRICK DEMONSTRATION
# ============================================================================

print(f"\n\n4. SVM WITH KERNEL TRICK DEMONSTRATION")
print("-" * 50)

# Create a more complex dataset
X_svm, y_svm = make_moons(n_samples=200, noise=0.1, random_state=42)

# Split into train and test
train_size = 150
X_train, X_test = X_svm[:train_size], X_svm[train_size:]
y_train, y_test = y_svm[:train_size], y_svm[train_size:]

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Train SVM with polynomial kernel (kernel trick)
print(f"\nTraining SVM with polynomial kernel (kernel trick)...")
svm_kernel = SVC(kernel='poly', degree=2, C=1.0, random_state=42)
svm_kernel.fit(X_train, y_train)

# Predictions
y_pred_kernel = svm_kernel.predict(X_test)
accuracy_kernel = accuracy_score(y_test, y_pred_kernel)
print(f"Accuracy with kernel trick: {accuracy_kernel:.4f}")

# Train SVM with explicit feature mapping
print(f"\nTraining SVM with explicit feature mapping...")
X_train_transformed = np.array([explicit_feature_mapping(x) for x in X_train])
X_test_transformed = np.array([explicit_feature_mapping(x) for x in X_test])

svm_explicit = SVC(kernel='linear', C=1.0, random_state=42)
svm_explicit.fit(X_train_transformed, y_train)

# Predictions
y_pred_explicit = svm_explicit.predict(X_test_transformed)
accuracy_explicit = accuracy_score(y_test, y_pred_explicit)
print(f"Accuracy with explicit mapping: {accuracy_explicit:.4f}")

print(f"Accuracies are equal: {np.isclose(accuracy_kernel, accuracy_explicit)}")

# ============================================================================
# PART 5: DECISION BOUNDARY VISUALIZATION
# ============================================================================

print(f"\n\n5. DECISION BOUNDARY VISUALIZATION")
print("-" * 50)

# Create mesh grid for decision boundary
x_min, x_max = X_svm[:, 0].min() - 0.5, X_svm[:, 0].max() + 0.5
y_min, y_max = X_svm[:, 1].min() - 0.5, X_svm[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predict for mesh grid using kernel trick
Z_kernel = svm_kernel.predict(np.c_[xx.ravel(), yy.ravel()])
Z_kernel = Z_kernel.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(12, 5))

# Kernel trick decision boundary
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z_kernel, alpha=0.4)
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], 
           c='red', label='Class 0 (Train)', alpha=0.7)
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], 
           c='blue', label='Class 1 (Train)', alpha=0.7)
plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], 
           c='red', marker='s', label='Class 0 (Test)', alpha=0.7)
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], 
           c='blue', marker='s', label='Class 1 (Test)', alpha=0.7)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title(r'SVM with Polynomial Kernel (Kernel Trick)')
plt.legend()
plt.grid(True, alpha=0.3)

# Support vectors
support_vectors = svm_kernel.support_vectors_
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
           s=100, linewidth=1, facecolors='none', edgecolors='black', label='Support Vectors')

# Explicit mapping decision boundary (projected back to 2D)
plt.subplot(1, 2, 2)
Z_explicit = svm_explicit.predict(np.array([explicit_feature_mapping([x, y]) for x, y in zip(xx.ravel(), yy.ravel())]))
Z_explicit = Z_explicit.reshape(xx.shape)

plt.contourf(xx, yy, Z_explicit, alpha=0.4)
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], 
           c='red', label='Class 0 (Train)', alpha=0.7)
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], 
           c='blue', label='Class 1 (Train)', alpha=0.7)
plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], 
           c='red', marker='s', label='Class 0 (Test)', alpha=0.7)
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], 
           c='blue', marker='s', label='Class 1 (Test)', alpha=0.7)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title(r'SVM with Explicit Feature Mapping')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'decision_boundaries.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# PART 6: KERNEL MATRIX VISUALIZATION
# ============================================================================

print(f"\n\n6. KERNEL MATRIX VISUALIZATION")
print("-" * 50)

# Compute kernel matrix for a subset of data
n_subset = 50
X_subset = X_svm[:n_subset]
y_subset = y_svm[:n_subset]

# Compute kernel matrix
K_matrix = np.zeros((n_subset, n_subset))
for i in range(n_subset):
    for j in range(n_subset):
        K_matrix[i, j] = polynomial_kernel(X_subset[i], X_subset[j])

# Plot kernel matrix
plt.figure(figsize=(10, 8))
plt.imshow(K_matrix, cmap='viridis', aspect='auto')
plt.colorbar(label='Kernel Value')
plt.title(r'Polynomial Kernel Matrix $K(x_i, x_j) = (1 + x_i^T x_j)^2$')
plt.xlabel('Sample Index $j$')
plt.ylabel('Sample Index $i$')

# Add text annotations for some values
for i in range(0, n_subset, 10):
    for j in range(0, n_subset, 10):
        plt.text(j, i, f'{K_matrix[i, j]:.1f}', 
                ha='center', va='center', color='white', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'kernel_matrix.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# PART 7: COMPARISON OF DIFFERENT KERNELS
# ============================================================================

print(f"\n\n7. COMPARISON OF DIFFERENT KERNELS")
print("-" * 50)

# Test different kernels on the same dataset
kernels = {
    'Linear': 'linear',
    'Polynomial (d=2)': 'poly',
    'RBF': 'rbf'
}

results = {}
for kernel_name, kernel_type in kernels.items():
    print(f"\nTraining SVM with {kernel_name} kernel...")
    
    if kernel_type == 'poly':
        svm = SVC(kernel=kernel_type, degree=2, C=1.0, random_state=42)
    else:
        svm = SVC(kernel=kernel_type, C=1.0, random_state=42)
    
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[kernel_name] = accuracy
    print(f"Accuracy: {accuracy:.4f}")

# Plot comparison
plt.figure(figsize=(10, 6))
kernels_list = list(results.keys())
accuracies = list(results.values())

bars = plt.bar(kernels_list, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
plt.xlabel('Kernel Type')
plt.ylabel('Accuracy')
plt.title(r'SVM Performance with Different Kernels')
plt.ylim(0, 1)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.3f}', ha='center', va='bottom')

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'kernel_comparison.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# PART 8: MATHEMATICAL DERIVATION VISUALIZATION
# ============================================================================

print(f"\n\n8. MATHEMATICAL DERIVATION VISUALIZATION")
print("-" * 50)

# Create a figure showing the mathematical derivation
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# Mathematical derivation text with simplified LaTeX formatting
derivation_text = r"""
\textbf{Polynomial Kernel Derivation}

\textbf{Given:} $K(t, s) = (1 + t^T s)^2$ where $t, s \in \mathbb{R}^2$

\textbf{Step 1:} Expand the kernel expression
$(1 + t^T s)^2 = 1 + 2t^T s + (t^T s)^2$

\textbf{Step 2:} Express in terms of components
$= 1 + 2(t_1s_1 + t_2s_2) + (t_1s_1 + t_2s_2)^2$

\textbf{Step 3:} Expand the squared term
$= 1 + 2t_1s_1 + 2t_2s_2 + t_1^2s_1^2 + t_2^2s_2^2 + 2t_1t_2s_1s_2$

\textbf{Step 4:} Rearrange to match feature mapping
$= 1 \cdot 1 + (\sqrt{2}t_1)(\sqrt{2}s_1) + (\sqrt{2}t_2)(\sqrt{2}s_2) + t_1^2s_1^2 + t_2^2s_2^2 + (\sqrt{2}t_1t_2)(\sqrt{2}s_1s_2)$

\textbf{Step 5:} Identify the feature mapping
$\phi(t) = (1, \sqrt{2}t_1, \sqrt{2}t_2, t_1^2, t_2^2, \sqrt{2}t_1t_2)^T$

\textbf{Result:} $K(t, s) = \phi(t)^T \phi(s)$
"""

ax.text(0.05, 0.95, derivation_text, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'mathematical_derivation.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n\nSUMMARY")
print("=" * 80)
print(f"1. Kernel Trick Verification:")
print(f"   - Polynomial kernel K(t,s) = (1 + t^T s)^2")
print(f"   - Explicit mapping φ(t) = (1, √2*t₁, √2*t₂, t₁², t₂², √2*t₁*t₂)")
print(f"   - Verification: K(t,s) = φ(t)^T φ(s) ✓")

print(f"\n2. Computational Benefits:")
print(f"   - Explicit mapping time: {explicit_time:.4f}s")
print(f"   - Kernel trick time: {kernel_time:.4f}s")
print(f"   - Speedup: {explicit_time/kernel_time:.2f}x")

print(f"\n3. SVM Performance:")
for kernel_name, accuracy in results.items():
    print(f"   - {kernel_name}: {accuracy:.4f}")

print(f"\n4. Files Generated:")
print(f"   - feature_space_transformation.png")
print(f"   - decision_boundaries.png")
print(f"   - kernel_matrix.png")
print(f"   - kernel_comparison.png")
print(f"   - mathematical_derivation.png")

print(f"\nAll visualizations saved to: {save_dir}")
print("=" * 80)
