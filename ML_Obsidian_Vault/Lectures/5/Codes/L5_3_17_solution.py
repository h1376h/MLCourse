import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import eigvals
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_3_Quiz_17")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting (disabled for Unicode compatibility)
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("QUESTION 17: KERNEL MATRIX COMPUTATIONS")
print("=" * 80)

# Given points
X = np.array([[1, 2], [0, 1], [2, 0]])
print(f"Given points:")
print(f"x₁ = {X[0]} = (1, 2)")
print(f"x₂ = {X[1]} = (0, 1)")
print(f"x₃ = {X[2]} = (2, 0)")
print()

# Task 1: Linear Kernel (Gram matrix)
print("TASK 1: Linear Kernel K_ij = x_i^T x_j")
print("-" * 50)

# Compute linear kernel matrix
K_linear = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        K_linear[i, j] = np.dot(X[i], X[j])
        print(f"K_{i+1}{j+1} = x_{i+1}^T x_{j+1} = {X[i]} · {X[j]} = {K_linear[i, j]}")

print(f"\nLinear Kernel Matrix:")
print(K_linear)
print()

# Task 2: Polynomial Kernel
print("TASK 2: Polynomial Kernel K_ij = (x_i^T x_j + 1)²")
print("-" * 50)

# Compute polynomial kernel matrix
K_poly = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        inner_product = np.dot(X[i], X[j])
        K_poly[i, j] = (inner_product + 1)**2
        print(f"K_{i+1}{j+1} = (x_{i+1}^T x_{j+1} + 1)² = ({inner_product} + 1)² = {K_poly[i, j]}")

print(f"\nPolynomial Kernel Matrix:")
print(K_poly)
print()

# Task 3: RBF Kernel
print("TASK 3: RBF Kernel K_ij = exp(-0.5 ||x_i - x_j||²)")
print("-" * 50)

# Compute RBF kernel matrix
K_rbf = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        diff = X[i] - X[j]
        squared_distance = np.dot(diff, diff)
        K_rbf[i, j] = np.exp(-0.5 * squared_distance)
        print(f"K_{i+1}{j+1} = exp(-0.5 ||x_{i+1} - x_{j+1}||²)")
        print(f"  = exp(-0.5 ||{X[i]} - {X[j]}||²)")
        print(f"  = exp(-0.5 ||{diff}||²)")
        print(f"  = exp(-0.5 × {squared_distance})")
        print(f"  = {K_rbf[i, j]:.6f}")

print(f"\nRBF Kernel Matrix:")
print(K_rbf)
print()

# Task 4: Check Positive Semi-Definiteness
print("TASK 4: Verify Positive Semi-Definiteness")
print("-" * 50)

def check_psd(matrix, name):
    eigenvalues = eigvals(matrix)
    is_psd = np.all(eigenvalues >= -1e-10)  # Small tolerance for numerical precision
    print(f"\n{name} Kernel Matrix:")
    print(f"Eigenvalues: {eigenvalues}")
    print(f"All eigenvalues ≥ 0: {is_psd}")
    print(f"Min eigenvalue: {np.min(eigenvalues):.10f}")
    return eigenvalues, is_psd

eig_linear, psd_linear = check_psd(K_linear, "Linear")
eig_poly, psd_poly = check_psd(K_poly, "Polynomial")
eig_rbf, psd_rbf = check_psd(K_rbf, "RBF")

# Task 5: Effective Dimensionality
print("\nTASK 5: Effective Dimensionality")
print("-" * 50)

def effective_dimension(eigenvalues, name):
    # Count non-zero eigenvalues (with small tolerance)
    non_zero_eig = np.sum(np.abs(eigenvalues) > 1e-10)
    print(f"{name} Kernel:")
    print(f"  Number of non-zero eigenvalues: {non_zero_eig}")
    print(f"  Effective dimensionality: {non_zero_eig}")
    return non_zero_eig

dim_linear = effective_dimension(eig_linear, "Linear")
dim_poly = effective_dimension(eig_poly, "Polynomial")
dim_rbf = effective_dimension(eig_rbf, "RBF")

# Visualization 1: Kernel Matrices Heatmaps
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

kernels = [K_linear, K_poly, K_rbf]
titles = ['Linear Kernel', 'Polynomial Kernel', 'RBF Kernel']
names = ['Linear', 'Polynomial', 'RBF']

for i, (kernel, title, name) in enumerate(zip(kernels, titles, names)):
    sns.heatmap(kernel, annot=True, fmt='.4f', cmap='viridis', 
                xticklabels=['x1', 'x2', 'x3'], 
                yticklabels=['x1', 'x2', 'x3'],
                ax=axes[i])
    axes[i].set_title(f'{title}\n$K_{{ij}}$ Matrix')
    axes[i].set_xlabel('Point j')
    axes[i].set_ylabel('Point i')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'kernel_matrices_heatmap.png'), dpi=300, bbox_inches='tight')

# Visualization 2: Eigenvalue Analysis
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

eigenvalues_list = [eig_linear, eig_poly, eig_rbf]

for i, (eig_vals, name) in enumerate(zip(eigenvalues_list, names)):
    axes[i].bar(range(1, len(eig_vals) + 1), eig_vals, alpha=0.7, color='skyblue', edgecolor='navy')
    axes[i].axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Zero line')
    axes[i].set_title(f'{name} Kernel Eigenvalues')
    axes[i].set_xlabel('Eigenvalue Index')
    axes[i].set_ylabel('Eigenvalue')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)
    
    # Add eigenvalue values as text
    for j, val in enumerate(eig_vals):
        axes[i].text(j + 1, val + 0.01 * max(eig_vals), f'{val:.4f}', 
                    ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'eigenvalue_analysis.png'), dpi=300, bbox_inches='tight')

# Visualization 3: Points in 2D Space
fig, ax = plt.subplots(figsize=(10, 8))

# Plot the original points
colors = ['red', 'blue', 'green']
labels = ['x1', 'x2', 'x3']

for i, (point, color, label) in enumerate(zip(X, colors, labels)):
    ax.scatter(point[0], point[1], c=color, s=200, alpha=0.7, label=label)
    ax.annotate(label, (point[0], point[1]), xytext=(10, 10), 
                textcoords='offset points', fontsize=12, fontweight='bold')

# Add distance lines for RBF kernel visualization
for i in range(3):
    for j in range(i+1, 3):
        ax.plot([X[i][0], X[j][0]], [X[i][1], X[j][1]], 'k--', alpha=0.3)
        mid_x = (X[i][0] + X[j][0]) / 2
        mid_y = (X[i][1] + X[j][1]) / 2
        distance = np.sqrt(np.sum((X[i] - X[j])**2))
        ax.annotate(f'd={distance:.2f}', (mid_x, mid_y), 
                   xytext=(0, 5), textcoords='offset points', 
                   ha='center', fontsize=10, alpha=0.7)

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Original Points in 2D Space')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.5, 2.5)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'original_points_2d.png'), dpi=300, bbox_inches='tight')

# Visualization 4: Kernel Value Comparison
fig, ax = plt.subplots(figsize=(12, 8))

# Create comparison table
point_pairs = [(0, 1), (0, 2), (1, 2)]
pair_labels = ['(x1, x2)', '(x1, x3)', '(x2, x3)']

linear_vals = [K_linear[0, 1], K_linear[0, 2], K_linear[1, 2]]
poly_vals = [K_poly[0, 1], K_poly[0, 2], K_poly[1, 2]]
rbf_vals = [K_rbf[0, 1], K_rbf[0, 2], K_rbf[1, 2]]

x = np.arange(len(point_pairs))
width = 0.25

ax.bar(x - width, linear_vals, width, label='Linear', alpha=0.8)
ax.bar(x, poly_vals, width, label='Polynomial', alpha=0.8)
ax.bar(x + width, rbf_vals, width, label='RBF', alpha=0.8)

ax.set_xlabel('Point Pairs')
ax.set_ylabel('Kernel Value')
ax.set_title('Kernel Values for Different Point Pairs')
ax.set_xticks(x)
ax.set_xticklabels(pair_labels)
ax.legend()
ax.grid(True, alpha=0.3)

# Add value labels on bars
for i, (linear, poly, rbf) in enumerate(zip(linear_vals, poly_vals, rbf_vals)):
    ax.text(i - width, linear + 0.01, f'{linear:.3f}', ha='center', va='bottom', fontsize=9)
    ax.text(i, poly + 0.01, f'{poly:.3f}', ha='center', va='bottom', fontsize=9)
    ax.text(i + width, rbf + 0.01, f'{rbf:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'kernel_values_comparison.png'), dpi=300, bbox_inches='tight')

# Summary Table
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"{'Kernel Type':<15} {'PSD':<8} {'Effective Dim':<15} {'Min Eigenvalue':<15}")
print("-" * 60)
print(f"{'Linear':<15} {psd_linear:<8} {dim_linear:<15} {np.min(eig_linear):<15.6f}")
print(f"{'Polynomial':<15} {psd_poly:<8} {dim_poly:<15} {np.min(eig_poly):<15.6f}")
print(f"{'RBF':<15} {psd_rbf:<8} {dim_rbf:<15} {np.min(eig_rbf):<15.6f}")

print(f"\nPlots saved to: {save_dir}")

# Additional Analysis: Feature Space Interpretation
print("\n" + "=" * 80)
print("FEATURE SPACE INTERPRETATION")
print("=" * 80)

print("Linear Kernel:")
print("  - Maps points to their original 2D space")
print("  - Effective dimension = 2 (same as input dimension)")
print("  - Preserves linear relationships")

print("\nPolynomial Kernel (degree 2):")
print("  - Maps points to a higher-dimensional feature space")
print("  - Feature mapping: φ(x) = [1, √2x₁, √2x₂, x₁², x₂², √2x₁x₂]")
print("  - Effective dimension = 3 (captures quadratic relationships)")

print("\nRBF Kernel:")
print("  - Maps points to an infinite-dimensional feature space")
print("  - Each point becomes a Gaussian centered at that point")
print("  - Effective dimension = 3 (full rank for 3 distinct points)")
print("  - Kernel values decrease exponentially with distance")
