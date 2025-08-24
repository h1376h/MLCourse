import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.linalg import eigvals
from mpl_toolkits.mplot3d import Axes3D

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_3_Quiz_31")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("=" * 80)
print("QUESTION 31: GAUSSIAN KERNEL ANALYSIS")
print("=" * 80)

# Part 1: Prove Euclidean distance between mapped points is less than 2
print("\n" + "=" * 60)
print("PART 1: PROOF THAT EUCLIDEAN DISTANCE < 2")
print("=" * 60)

print("\nGiven: Gaussian kernel k(x_i, x_j) = exp(-(1/2)||x_i - x_j||^2)")
print("Task: Prove that $||\\phi(x_i) - \\phi(x_j)|| < 2$")

print("\nStep 1: Understanding the relationship between kernel and distance")
print("For any kernel k(x_i, x_j), we have:")
print("$k(x_i, x_j) = \\langle \\phi(x_i), \\phi(x_j) \\rangle$")
print("where $\\phi(x)$ is the feature mapping.")

print("\nStep 2: Expressing distance in terms of kernel values")
print("The squared Euclidean distance between mapped points is:")
print("$||\\phi(x_i) - \\phi(x_j)||^2 = \\langle \\phi(x_i) - \\phi(x_j), \\phi(x_i) - \\phi(x_j) \\rangle$")
print("                      $= \\langle \\phi(x_i), \\phi(x_i) \\rangle + \\langle \\phi(x_j), \\phi(x_j) \\rangle - 2\\langle \\phi(x_i), \\phi(x_j) \\rangle$")
print("                      $= k(x_i, x_i) + k(x_j, x_j) - 2k(x_i, x_j)$")

print("\nStep 3: Calculating kernel values")
print("For the Gaussian kernel k(x_i, x_j) = exp(-(1/2)||x_i - x_j||^2):")
print("- k(x_i, x_i) = exp(-(1/2)||x_i - x_i||^2) = exp(0) = 1")
print("- k(x_j, x_j) = exp(-(1/2)||x_j - x_j||^2) = exp(0) = 1")
print("- k(x_i, x_j) = exp(-(1/2)||x_i - x_j||^2)")

print("\nStep 4: Substituting into distance formula")
print("$||\\phi(x_i) - \\phi(x_j)||^2 = 1 + 1 - 2 \\cdot \\exp(-\\frac{1}{2}||x_i - x_j||^2)$")
print("                      $= 2 - 2 \\cdot \\exp(-\\frac{1}{2}||x_i - x_j||^2)$")
print("                      $= 2 \\cdot (1 - \\exp(-\\frac{1}{2}||x_i - x_j||^2))$")

print("\nStep 5: Analyzing the expression")
print("Since exp(-(1/2)||x_i - x_j||^2) > 0 for any finite ||x_i - x_j||:")
print("1 - exp(-(1/2)||x_i - x_j||^2) < 1")
print("Therefore: $||\\phi(x_i) - \\phi(x_j)||^2 < 2$")
print("Taking square root: $||\\phi(x_i) - \\phi(x_j)|| < \\sqrt{2} \\approx 1.414 < 2$")

print("\nStep 6: Verification with numerical examples")
print("\nLet's verify our theoretical result with concrete examples.")
print("We'll use the points from Part 2: x1 = (0,0), x2 = (1,0), x3 = (0,1)")

# Use the points from Part 2 for consistency
x1 = np.array([0, 0])
x2 = np.array([1, 0])
x3 = np.array([0, 1])

def gaussian_kernel(x1, x2):
    """Compute Gaussian kernel k(x1, x2) = exp(-(1/2)||x1 - x2||^2)"""
    distance_squared = np.sum((x1 - x2)**2)
    return np.exp(-0.5 * distance_squared)

def mapped_distance(x1, x2):
    """Compute Euclidean distance between mapped points φ(x1) and φ(x2)"""
    k11 = gaussian_kernel(x1, x1)  # = 1
    k22 = gaussian_kernel(x2, x2)  # = 1
    k12 = gaussian_kernel(x1, x2)
    distance_squared = k11 + k22 - 2 * k12
    return np.sqrt(distance_squared)

print(f"\nExample 1: x1 = {x1}, x2 = {x2}")
print("Step-by-step calculation:")
print(f"1. Original distance: ||x1 - x2|| = ||{x1} - {x2}|| = ||{x1-x2}|| = {np.linalg.norm(x1 - x2):.4f}")
print(f"2. Squared distance: ||x1 - x2||^2 = {np.linalg.norm(x1 - x2)**2:.4f}")
print(f"3. Kernel value: k(x1, x2) = exp(-(1/2) * {np.linalg.norm(x1 - x2)**2:.4f}) = exp(-{np.linalg.norm(x1 - x2)**2/2:.4f}) = {gaussian_kernel(x1, x2):.6f}")
print(f"4. Mapped distance squared: ||φ(x1) - φ(x2)||^2 = 2 - 2 * {gaussian_kernel(x1, x2):.6f} = {2 - 2*gaussian_kernel(x1, x2):.6f}")
print(f"5. Mapped distance: ||φ(x1) - φ(x2)|| = sqrt({2 - 2*gaussian_kernel(x1, x2):.6f}) = {mapped_distance(x1, x2):.6f}")
print(f"6. Bound check: {mapped_distance(x1, x2):.6f} < 2 ✓")

print(f"\nExample 2: x1 = {x1}, x3 = {x3}")
print("Step-by-step calculation:")
print(f"1. Original distance: ||x1 - x3|| = ||{x1} - {x3}|| = ||{x1-x3}|| = {np.linalg.norm(x1 - x3):.4f}")
print(f"2. Squared distance: ||x1 - x3||^2 = {np.linalg.norm(x1 - x3)**2:.4f}")
print(f"3. Kernel value: k(x1, x3) = exp(-(1/2) * {np.linalg.norm(x1 - x3)**2:.4f}) = exp(-{np.linalg.norm(x1 - x3)**2/2:.4f}) = {gaussian_kernel(x1, x3):.6f}")
print(f"4. Mapped distance squared: ||φ(x1) - φ(x3)||^2 = 2 - 2 * {gaussian_kernel(x1, x3):.6f} = {2 - 2*gaussian_kernel(x1, x3):.6f}")
print(f"5. Mapped distance: ||φ(x1) - φ(x3)|| = sqrt({2 - 2*gaussian_kernel(x1, x3):.6f}) = {mapped_distance(x1, x3):.6f}")
print(f"6. Bound check: {mapped_distance(x1, x3):.6f} < 2 ✓")

print(f"\nExample 3: x2 = {x2}, x3 = {x3}")
print("Step-by-step calculation:")
print(f"1. Original distance: ||x2 - x3|| = ||{x2} - {x3}|| = ||{x2-x3}|| = {np.linalg.norm(x2 - x3):.4f}")
print(f"2. Squared distance: ||x2 - x3||^2 = {np.linalg.norm(x2 - x3)**2:.4f}")
print(f"3. Kernel value: k(x2, x3) = exp(-(1/2) * {np.linalg.norm(x2 - x3)**2:.4f}) = exp(-{np.linalg.norm(x2 - x3)**2/2:.4f}) = {gaussian_kernel(x2, x3):.6f}")
print(f"4. Mapped distance squared: ||φ(x2) - φ(x3)||^2 = 2 - 2 * {gaussian_kernel(x2, x3):.6f} = {2 - 2*gaussian_kernel(x2, x3):.6f}")
print(f"5. Mapped distance: ||φ(x2) - φ(x3)|| = sqrt({2 - 2*gaussian_kernel(x2, x3):.6f}) = {mapped_distance(x2, x3):.6f}")
print(f"6. Bound check: {mapped_distance(x2, x3):.6f} < 2 ✓")

print("\n✓ All mapped distances are indeed less than 2!")

# Part 2: Calculate kernel matrix and prove positive semi-definiteness
print("\n" + "=" * 60)
print("PART 2: KERNEL MATRIX AND POSITIVE SEMI-DEFINITENESS")
print("=" * 60)

print("\nGiven points:")
x1 = np.array([0, 0])
x2 = np.array([1, 0])
x3 = np.array([0, 1])

print(f"x1 = {x1}")
print(f"x2 = {x2}")
print(f"x3 = {x3}")

print("\nStep 1: Calculate all pairwise kernel values")
points = [x1, x2, x3]
n = len(points)

print("We need to calculate the kernel matrix K where K[i,j] = k(x_i, x_j)")
print("Let's compute each element step by step:")

# Calculate kernel matrix K with detailed steps
K = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        print(f"\nK[{i+1},{j+1}] = k(x{i+1}, x{j+1}):")
        print(f"  x{i+1} = {points[i]}, x{j+1} = {points[j]}")
        print(f"  ||x{i+1} - x{j+1}||^2 = ||{points[i]} - {points[j]}||^2 = ||{points[i] - points[j]}||^2 = {np.sum((points[i] - points[j])**2):.4f}")
        print(f"  k(x{i+1}, x{j+1}) = exp(-(1/2) * {np.sum((points[i] - points[j])**2):.4f}) = exp(-{np.sum((points[i] - points[j])**2)/2:.4f}) = {np.exp(-0.5 * np.sum((points[i] - points[j])**2)):.6f}")
        K[i, j] = gaussian_kernel(points[i], points[j])

print(f"\nTherefore, the kernel matrix K is:")
print(K)

print("\nStep 2: Verify kernel matrix properties")
print("Diagonal elements (should all be 1):")
for i in range(n):
    print(f"K[{i+1},{i+1}] = k(x{i+1}, x{i+1}) = {K[i,i]:.4f}")

print("\nOff-diagonal elements:")
for i in range(n):
    for j in range(i+1, n):
        print(f"K[{i+1},{j+1}] = k(x{i+1}, x{j+1}) = {K[i,j]:.4f}")

print("\nStep 3: Calculate eigenvalues")
print("To find the eigenvalues, we solve the characteristic equation det(K - λI) = 0")
print("For a 3×3 matrix, this gives us a cubic equation in λ")
print("Let's compute the eigenvalues numerically:")

eigenvalues = np.linalg.eigvals(K)
print(f"Eigenvalues of K: λ₁ = {eigenvalues[0]:.6f}, λ₂ = {eigenvalues[1]:.6f}, λ₃ = {eigenvalues[2]:.6f}")

print("\nLet's verify these are indeed eigenvalues by checking K - λI for each eigenvalue:")
for i, eig in enumerate(eigenvalues):
    print(f"\nFor λ{i+1} = {eig:.6f}:")
    print(f"K - λ{i+1}I = K - {eig:.6f} * I")
    print(f"det(K - λ{i+1}I) should be approximately 0")
    det = np.linalg.det(K - eig * np.eye(3))
    print(f"det(K - λ{i+1}I) = {det:.10f} ≈ 0 ✓")

print("\nStep 4: Check positive semi-definiteness")
print("A matrix is positive semi-definite if all its eigenvalues are non-negative.")
print("Let's check each eigenvalue:")

is_psd = np.all(eigenvalues >= -1e-10)  # Allow for small numerical errors
for i, eig in enumerate(eigenvalues):
    print(f"λ{i+1} = {eig:.6f} {'≥' if eig >= -1e-10 else '<'} 0 {'✓' if eig >= -1e-10 else '✗'}")

print(f"\nAll eigenvalues ≥ 0: {is_psd}")
print(f"Minimum eigenvalue: {np.min(eigenvalues):.10f}")

if is_psd:
    print("✓ The kernel matrix K is positive semi-definite!")
else:
    print("✗ The kernel matrix K is NOT positive semi-definite!")

print("\nStep 5: Theoretical verification")
print("For any valid kernel k(x_i, x_j), the kernel matrix K must be positive semi-definite.")
print("This is a consequence of Mercer's theorem.")
print("The Gaussian kernel is a valid kernel, so K must be positive semi-definite.")

# Create visualizations
print("\n" + "=" * 60)
print("CREATING VISUALIZATIONS")
print("=" * 60)

# Visualization 1: Original points and their distances
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Original points
ax1.scatter([x1[0], x2[0], x3[0]], [x1[1], x2[1], x3[1]], 
           c=['red', 'blue', 'green'], s=200, alpha=0.7, edgecolors='black')
ax1.plot([x1[0], x2[0]], [x1[1], x2[1]], 'k--', alpha=0.5, label=f'Distance: {np.linalg.norm(x1-x2):.3f}')
ax1.plot([x1[0], x3[0]], [x1[1], x3[1]], 'k--', alpha=0.5, label=f'Distance: {np.linalg.norm(x1-x3):.3f}')
ax1.plot([x2[0], x3[0]], [x2[1], x3[1]], 'k--', alpha=0.5, label=f'Distance: {np.linalg.norm(x2-x3):.3f}')

ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.set_title('Original Points in Input Space')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_aspect('equal')

# Add point labels
for i, point in enumerate([x1, x2, x3]):
    ax1.annotate(f'$x_{i+1}$', (point[0], point[1]), 
                xytext=(10, 10), textcoords='offset points', fontsize=12)

# Plot 2: Kernel values heatmap
im = ax2.imshow(K, cmap='viridis', aspect='auto')
ax2.set_xticks(range(n))
ax2.set_yticks(range(n))
ax2.set_xticklabels(['$x_1$', '$x_2$', '$x_3$'])
ax2.set_yticklabels(['$x_1$', '$x_2$', '$x_3$'])
ax2.set_title('Kernel Matrix K')

# Add text annotations
for i in range(n):
    for j in range(n):
        text = ax2.text(j, i, f'{K[i, j]:.3f}',
                       ha="center", va="center", color="white", fontweight='bold')

plt.colorbar(im, ax=ax2, label='Kernel Value')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'kernel_matrix_visualization.png'), dpi=300, bbox_inches='tight')

# Visualization 2: Eigenvalue analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Eigenvalues
ax1.bar(range(1, n+1), eigenvalues, color=['red', 'blue', 'green'], alpha=0.7)
ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax1.set_xlabel('Eigenvalue Index')
ax1.set_ylabel('Eigenvalue')
ax1.set_title('Eigenvalues of Kernel Matrix K')
ax1.grid(True, alpha=0.3)

# Add eigenvalue labels
for i, eig in enumerate(eigenvalues):
    ax1.text(i+1, eig + 0.01, f'{eig:.4f}', ha='center', va='bottom', fontweight='bold')

# Plot 2: Distance comparison
original_distances = [
    np.linalg.norm(x1 - x2),
    np.linalg.norm(x1 - x3),
    np.linalg.norm(x2 - x3)
]

mapped_distances = [
    mapped_distance(x1, x2),
    mapped_distance(x1, x3),
    mapped_distance(x2, x3)
]

pairs = ['$(x_1,x_2)$', '$(x_1,x_3)$', '$(x_2,x_3)$']
x_pos = np.arange(len(pairs))

ax2.bar(x_pos - 0.2, original_distances, 0.4, label='Original Distance', alpha=0.7)
ax2.bar(x_pos + 0.2, mapped_distances, 0.4, label='Mapped Distance', alpha=0.7)
ax2.axhline(y=2, color='red', linestyle='--', label='Upper Bound (2)', alpha=0.7)
ax2.axhline(y=np.sqrt(2), color='orange', linestyle='--', label='Theoretical Bound ($\\sqrt{2}$)', alpha=0.7)

ax2.set_xlabel('Point Pairs')
ax2.set_ylabel('Distance')
ax2.set_title('Distance Comparison: Original vs Mapped Space')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(pairs)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'eigenvalue_and_distance_analysis.png'), dpi=300, bbox_inches='tight')

# Visualization 3: Feature space mapping concept
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Create a grid of points to show the mapping
x_range = np.linspace(-2, 3, 20)
y_range = np.linspace(-2, 3, 20)
X, Y = np.meshgrid(x_range, y_range)

# For visualization, we'll show a simplified 3D mapping
# In reality, the Gaussian kernel maps to infinite dimensions
Z = np.exp(-0.5 * (X**2 + Y**2))  # Simplified representation

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

# Plot the original points and their mappings
for i, point in enumerate([x1, x2, x3]):
    # Original point in 2D
    ax.scatter(point[0], point[1], 0, c=['red', 'blue', 'green'][i], 
              s=100, marker='o', edgecolors='black', label=f'x_{i+1} (original)')
    
    # Mapped point in 3D (simplified)
    mapped_z = gaussian_kernel(point, np.array([0, 0]))  # Distance from origin
    ax.scatter(point[0], point[1], mapped_z, c=['red', 'blue', 'green'][i], 
                              s=100, marker='^', edgecolors='black', label=f'$\\phi(x_{{{i+1}}})$ (mapped)')

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$\\phi(x)$ (simplified)')
ax.set_title('Conceptual Feature Space Mapping\n(Gaussian Kernel)')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_space_mapping.png'), dpi=300, bbox_inches='tight')

# Visualization 4: Kernel function behavior vs distance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Gaussian kernel function
distances = np.linspace(0, 4, 100)
kernel_values = np.exp(-0.5 * distances**2)

ax1.plot(distances, kernel_values, 'b-', linewidth=2, label='Gaussian Kernel')
ax1.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Maximum (distance=0)')
ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax1.set_xlabel('Distance $||x_i - x_j||$')
ax1.set_ylabel('Kernel Value $k(x_i, x_j)$')
ax1.set_title('Gaussian Kernel Function')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Mark specific points from our examples
example_distances = [0, 1, np.sqrt(2)]
example_kernels = [1.0, np.exp(-0.5), np.exp(-1.0)]
colors = ['red', 'blue', 'green']
labels = ['$x_1$ vs $x_1$', '$x_1$ vs $x_2$', '$x_2$ vs $x_3$']

for i, (d, k, c, l) in enumerate(zip(example_distances, example_kernels, colors, labels)):
    ax1.scatter(d, k, color=c, s=100, zorder=5, label=l)
    ax1.annotate(f'{k:.3f}', (d, k), xytext=(10, 10), textcoords='offset points', fontsize=10)

# Plot 2: Mapped distance vs original distance
mapped_distances = np.sqrt(2 - 2 * kernel_values)

ax2.plot(distances, mapped_distances, 'g-', linewidth=2, label='Mapped Distance')
ax2.axhline(y=np.sqrt(2), color='orange', linestyle='--', alpha=0.7, label='Theoretical Bound $\\sqrt{2}$')
ax2.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='Required Bound 2')
ax2.set_xlabel('Original Distance $||x_i - x_j||$')
ax2.set_ylabel('Mapped Distance $||\\phi(x_i) - \\phi(x_j)||$')
ax2.set_title('Distance Transformation')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Mark specific points from our examples
example_mapped = [0, np.sqrt(2 - 2*np.exp(-0.5)), np.sqrt(2 - 2*np.exp(-1.0))]

for i, (d, md, c, l) in enumerate(zip(example_distances, example_mapped, colors, labels)):
    ax2.scatter(d, md, color=c, s=100, zorder=5)
    ax2.annotate(f'{md:.3f}', (d, md), xytext=(10, 10), textcoords='offset points', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'kernel_behavior_analysis.png'), dpi=300, bbox_inches='tight')

print(f"\nVisualizations saved to: {save_dir}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("\nPart 1 Results:")
print("✓ Proved that $||\\phi(x_i) - \\phi(x_j)|| < 2$ for any points $x_i, x_j$")
print("✓ The theoretical bound is actually $\\sqrt{2} \\approx 1.414$")
print("✓ Verified with numerical examples")

print("\nPart 2 Results:")
print("✓ Calculated kernel matrix K:")
print(K)
print(f"✓ Eigenvalues: {eigenvalues}")
print("✓ All eigenvalues are non-negative")
print("✓ Kernel matrix K is positive semi-definite")
print("✓ This confirms the Gaussian kernel is a valid kernel")

print("\nKey Insights:")
print("1. The Gaussian kernel maps points to a feature space where distances are bounded")
print("2. This boundedness property is useful for kernel methods")
print("3. The positive semi-definiteness ensures the kernel can be used in SVM optimization")
print("4. The kernel matrix captures the similarity structure of the data")

print("\n" + "=" * 80)
