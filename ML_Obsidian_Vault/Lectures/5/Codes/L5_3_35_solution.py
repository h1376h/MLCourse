import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_3_Quiz_35")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("=" * 80)
print("QUESTION 35: POLYNOMIAL KERNEL PROOF AND FEATURE MAPPING")
print("=" * 80)

# Define the basis functions
def phi_0(d):
    """φ₀(d) = d₁²"""
    return d[0]**2

def phi_1(d):
    """φ₁(d) = d₂²"""
    return d[1]**2

def phi_2(d):
    """φ₂(d) = √2 · d₁ · d₂"""
    return np.sqrt(2) * d[0] * d[1]

def phi_3(d):
    """φ₃(d) = √2 · d₁"""
    return np.sqrt(2) * d[0]

def phi_4(d):
    """φ₄(d) = √2 · d₂"""
    return np.sqrt(2) * d[1]

def phi_5(d):
    """φ₅(d) = 1"""
    return 1

def phi_mapping(d):
    """Complete feature mapping φ(d) = [φ₀(d), φ₁(d), φ₂(d), φ₃(d), φ₄(d), φ₅(d)]"""
    return np.array([
        phi_0(d),  # d₁²
        phi_1(d),  # d₂²
        phi_2(d),  # √2 · d₁ · d₂
        phi_3(d),  # √2 · d₁
        phi_4(d),  # √2 · d₂
        phi_5(d)   # 1
    ])

def polynomial_kernel(d, q):
    """Polynomial kernel K(d, q) = (1 + d · q)²"""
    dot_product = np.dot(d, q)
    return (1 + dot_product)**2

def dot_product_after_mapping(d, q):
    """Calculate φ(d) · φ(q)"""
    phi_d = phi_mapping(d)
    phi_q = phi_mapping(q)
    return np.dot(phi_d, phi_q)

print("\n1. PROVING THE POLYNOMIAL KERNEL EQUIVALENCE")
print("-" * 50)

# Test with specific vectors
d = np.array([2, 3])
q = np.array([1, 2])

print(f"Support vector d = {d}")
print(f"Query vector q = {q}")

# Calculate polynomial kernel
kernel_value = polynomial_kernel(d, q)
print(f"\nPolynomial kernel K(d, q) = (1 + d · q)² = {kernel_value}")

# Calculate dot product after mapping
mapped_dot_product = dot_product_after_mapping(d, q)
print(f"Dot product after mapping φ(d) · φ(q) = {mapped_dot_product}")

print(f"\nAre they equal? {np.isclose(kernel_value, mapped_dot_product)}")

# Step-by-step proof
print("\n" + "="*60)
print("STEP-BY-STEP PROOF")
print("="*60)

print(f"\nStep 1: Calculate d · q")
dot_dq = np.dot(d, q)
print(f"d · q = {d[0]} × {q[0]} + {d[1]} × {q[1]} = {dot_dq}")

print(f"\nStep 2: Calculate (1 + d · q)²")
step2 = (1 + dot_dq)**2
print(f"(1 + d · q)² = (1 + {dot_dq})² = {step2}")

print(f"\nStep 3: Calculate φ(d) and φ(q)")
phi_d = phi_mapping(d)
phi_q = phi_mapping(q)
print(f"φ(d) = [{', '.join([f'{x:.3f}' for x in phi_d])}]")
print(f"φ(q) = [{', '.join([f'{x:.3f}' for x in phi_q])}]")

print(f"\nStep 4: Calculate φ(d) · φ(q)")
step4 = np.dot(phi_d, phi_q)
print(f"φ(d) · φ(q) = {step4}")

print(f"\nStep 5: Verify equality")
print(f"K(d, q) = {kernel_value}")
print(f"φ(d) · φ(q) = {mapped_dot_product}")
print(f"Difference = {abs(kernel_value - mapped_dot_product)}")

# Mathematical proof
print("\n" + "="*60)
print("MATHEMATICAL PROOF")
print("="*60)

print("\nLet's prove this algebraically:")
print("K(d, q) = (1 + d · q)²")
print("        = (1 + d₁q₁ + d₂q₂)²")
print("        = 1 + 2(d₁q₁ + d₂q₂) + (d₁q₁ + d₂q₂)²")
print("        = 1 + 2d₁q₁ + 2d₂q₂ + d₁²q₁² + d₂²q₂² + 2d₁d₂q₁q₂")

print("\nNow let's calculate φ(d) · φ(q):")
print("φ(d) = [d₁², d₂², √2·d₁d₂, √2·d₁, √2·d₂, 1]")
print("φ(q) = [q₁², q₂², √2·q₁q₂, √2·q₁, √2·q₂, 1]")

print("\nφ(d) · φ(q) = d₁²q₁² + d₂²q₂² + (√2·d₁d₂)(√2·q₁q₂) + (√2·d₁)(√2·q₁) + (√2·d₂)(√2·q₂) + 1·1")
print("             = d₁²q₁² + d₂²q₂² + 2d₁d₂q₁q₂ + 2d₁q₁ + 2d₂q₂ + 1")
print("             = (1 + d₁q₁ + d₂q₂)²")
print("             = K(d, q) ✓")

print("\n2. FEATURE SPACE DIMENSIONALITY")
print("-" * 50)

print(f"Original input space: 2-dimensional (d₁, d₂)")
print(f"Feature space after φ mapping: 6-dimensional")
print(f"φ(d) = [d₁², d₂², √2·d₁d₂, √2·d₁, √2·d₂, 1]")

print("\n3. VERIFICATION OF FEATURE MAPPING")
print("-" * 50)

print(f"φ(d) = [{', '.join([f'{x:.3f}' for x in phi_d])}]")
print(f"Expected: [d₁², d₂², √2·d₁d₂, √2·d₁, √2·d₂, 1]")
print(f"Calculated: [{d[0]**2:.3f}, {d[1]**2:.3f}, {np.sqrt(2)*d[0]*d[1]:.3f}, {np.sqrt(2)*d[0]:.3f}, {np.sqrt(2)*d[1]:.3f}, 1.000]")

# Visualization 1: Original 2D space
print("\n" + "="*60)
print("VISUALIZATION 1: ORIGINAL 2D SPACE")
print("="*60)

fig, ax = plt.subplots(figsize=(10, 8))

# Create a grid of points
x1 = np.linspace(-3, 3, 50)
x2 = np.linspace(-3, 3, 50)
X1, X2 = np.meshgrid(x1, x2)

# Plot some example points
example_points = [
    np.array([1, 1]),
    np.array([-1, 1]),
    np.array([1, -1]),
    np.array([-1, -1]),
    np.array([0, 0]),
    np.array([2, 1]),
    np.array([1, 2])
]

colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

for i, (point, color, label) in enumerate(zip(example_points, colors, labels)):
    ax.scatter(point[0], point[1], c=color, s=100, label=f'Point {label} {point}')
    ax.annotate(label, (point[0], point[1]), xytext=(5, 5), textcoords='offset points')

ax.set_xlabel('$d_1$')
ax.set_ylabel('$d_2$')
ax.set_title('Original 2D Input Space')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.axis('equal')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'original_2d_space.png'), dpi=300, bbox_inches='tight')

# Visualization 2: Feature space projection (3D view of first 3 dimensions)
print("\n" + "="*60)
print("VISUALIZATION 2: FEATURE SPACE PROJECTION (3D)")
print("="*60)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Calculate feature space coordinates for example points
feature_coords = []
for point in example_points:
    phi_point = phi_mapping(point)
    feature_coords.append(phi_point)

feature_coords = np.array(feature_coords)

# Plot in 3D (first 3 dimensions: d₁², d₂², √2·d₁d₂)
ax.scatter(feature_coords[:, 0], feature_coords[:, 1], feature_coords[:, 2], 
           c=colors, s=100)

for i, (coord, label) in enumerate(zip(feature_coords, labels)):
    ax.text(coord[0], coord[1], coord[2], label, fontsize=12)

ax.set_xlabel('$\\phi_0(d) = d_1^2$')
ax.set_ylabel('$\\phi_1(d) = d_2^2$')
ax.set_zlabel('$\\phi_2(d) = \\sqrt{2} \\cdot d_1 \\cdot d_2$')
ax.set_title('Feature Space Projection (First 3 Dimensions)')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_space_3d.png'), dpi=300, bbox_inches='tight')

# Visualization 3: Kernel values heatmap
print("\n" + "="*60)
print("VISUALIZATION 3: KERNEL VALUES HEATMAP")
print("="*60)

# Create a grid for kernel visualization
x1_range = np.linspace(-2, 2, 20)
x2_range = np.linspace(-2, 2, 20)
X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)

# Fixed query point
q_fixed = np.array([1, 0.5])

# Calculate kernel values
kernel_values = np.zeros_like(X1_grid)
for i in range(X1_grid.shape[0]):
    for j in range(X1_grid.shape[1]):
        d_point = np.array([X1_grid[i, j], X2_grid[i, j]])
        kernel_values[i, j] = polynomial_kernel(d_point, q_fixed)

fig, ax = plt.subplots(figsize=(10, 8))
contour = ax.contourf(X1_grid, X2_grid, kernel_values, levels=20, cmap='viridis')
ax.scatter(q_fixed[0], q_fixed[1], c='red', s=200, marker='*', label=f'Query point {q_fixed}')
ax.set_xlabel('$d_1$')
ax.set_ylabel('$d_2$')
ax.set_title(f'Polynomial Kernel Values K(d, {q_fixed}) = (1 + d · {q_fixed})²')
plt.colorbar(contour, ax=ax, label='Kernel Value')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'kernel_values_heatmap.png'), dpi=300, bbox_inches='tight')

# Visualization 4: Decision boundary in original space
print("\n" + "="*60)
print("VISUALIZATION 4: DECISION BOUNDARY IN ORIGINAL SPACE")
print("="*60)

# Create synthetic data that's not linearly separable in 2D
np.random.seed(42)
n_points = 100

# Generate data in a circle pattern
theta = np.linspace(0, 2*np.pi, n_points)
r_inner = 0.5 + 0.2*np.random.randn(n_points)
r_outer = 1.5 + 0.2*np.random.randn(n_points)

# Inner circle (class -1)
x1_inner = r_inner * np.cos(theta)
x2_inner = r_inner * np.sin(theta)
y_inner = -np.ones(n_points)

# Outer circle (class +1)
x1_outer = r_outer * np.cos(theta)
x2_outer = r_outer * np.sin(theta)
y_outer = np.ones(n_points)

# Combine data
X = np.vstack([np.column_stack([x1_inner, x2_inner]), 
               np.column_stack([x1_outer, x2_outer])])
y = np.hstack([y_inner, y_outer])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Original space (not linearly separable)
ax1.scatter(X[y == -1, 0], X[y == -1, 1], c='red', label='Class -1', alpha=0.6)
ax1.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label='Class +1', alpha=0.6)
ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.set_title('Original 2D Space (Not Linearly Separable)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)

# Plot 2: Feature space (first 3 dimensions)
phi_X = np.array([phi_mapping(x) for x in X])
ax2.scatter(phi_X[y == -1, 0], phi_X[y == -1, 1], c='red', label='Class -1', alpha=0.6)
ax2.scatter(phi_X[y == 1, 0], phi_X[y == 1, 1], c='blue', label='Class +1', alpha=0.6)
ax2.set_xlabel('$\\phi_0(x) = x_1^2$')
ax2.set_ylabel('$\\phi_1(x) = x_2^2$')
ax2.set_title('Feature Space (First 2 Dimensions)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'linear_separability_comparison.png'), dpi=300, bbox_inches='tight')

# Visualization 5: Kernel matrix visualization
print("\n" + "="*60)
print("VISUALIZATION 5: KERNEL MATRIX")
print("="*60)

# Select a subset of points for kernel matrix
n_kernel = 10
indices = np.random.choice(len(X), n_kernel, replace=False)
X_kernel = X[indices]
y_kernel = y[indices]

# Calculate kernel matrix
K = np.zeros((n_kernel, n_kernel))
for i in range(n_kernel):
    for j in range(n_kernel):
        K[i, j] = polynomial_kernel(X_kernel[i], X_kernel[j])

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(K, cmap='viridis', aspect='auto')
ax.set_title('Polynomial Kernel Matrix K(d, q) = (1 + d · q)²')
ax.set_xlabel('Query Points')
ax.set_ylabel('Support Vectors')
plt.colorbar(im, ax=ax, label='Kernel Value')

# Add text annotations
for i in range(n_kernel):
    for j in range(n_kernel):
        text = ax.text(j, i, f'{K[i, j]:.2f}',
                      ha="center", va="center", color="white", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'kernel_matrix.png'), dpi=300, bbox_inches='tight')

# Numerical verification with multiple examples
print("\n" + "="*60)
print("NUMERICAL VERIFICATION WITH MULTIPLE EXAMPLES")
print("="*60)

test_cases = [
    (np.array([0, 0]), np.array([1, 1])),
    (np.array([1, 0]), np.array([0, 1])),
    (np.array([2, 3]), np.array([1, 2])),
    (np.array([-1, 1]), np.array([1, -1])),
    (np.array([0.5, 0.5]), np.array([2, 2]))
]

print(f"{'d':<15} {'q':<15} {'K(d,q)':<15} {'φ(d)·φ(q)':<15} {'Equal?':<10}")
print("-" * 75)

for d, q in test_cases:
    kernel_val = polynomial_kernel(d, q)
    mapped_dot = dot_product_after_mapping(d, q)
    is_equal = np.isclose(kernel_val, mapped_dot)
    
    print(f"{str(d):<15} {str(q):<15} {kernel_val:<15.6f} {mapped_dot:<15.6f} {is_equal:<10}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print("\n1. ✓ Polynomial kernel K(d, q) = (1 + d · q)² is equivalent to φ(d) · φ(q)")
print("2. ✓ Feature mapping φ transforms 2D input to 6D feature space")
print("3. ✓ φ(d) = [d₁², d₂², √2·d₁d₂, √2·d₁, √2·d₂, 1] verified")
print("4. ✓ Transformation enables linear separation in higher-dimensional space")

print(f"\nAll visualizations saved to: {save_dir}")
print("\nFiles generated:")
print("- original_2d_space.png: Original 2D input space")
print("- feature_space_3d.png: 3D projection of feature space")
print("- kernel_values_heatmap.png: Kernel values visualization")
print("- linear_separability_comparison.png: Before/after transformation comparison")
print("- kernel_matrix.png: Kernel matrix heatmap")

plt.close()
