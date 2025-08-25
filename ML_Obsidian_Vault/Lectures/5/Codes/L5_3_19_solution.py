import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import rbf_kernel
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_3_Quiz_19")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX rendering for mathematical expressions
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
# Use LaTeX with proper math support
plt.rcParams['text.latex.preamble'] = r'''
\usepackage{amsmath}
\usepackage{amssymb}
'''

print("=" * 80)
print("QUESTION 19: RBF KERNEL ANALYSIS")
print("=" * 80)

# Define the RBF kernel function
def rbf_kernel_manual(X, Y=None, gamma=1.0):
    """
    Manual implementation of RBF kernel: K(x,z) = exp(-gamma * ||x-z||^2)
    """
    if Y is None:
        Y = X
    
    # Compute pairwise squared Euclidean distances
    distances = cdist(X, Y, metric='sqeuclidean')
    
    # Apply RBF kernel
    K = np.exp(-gamma * distances)
    return K

# Task 1: Calculate pairwise kernel values for given points
print("\n" + "="*50)
print("TASK 1: Pairwise Kernel Values with $\\gamma = 0.5$")
print("="*50)

# Define the points
points = np.array([[1, 0], [0, 1], [2, 2]])
gamma = 0.5

print(f"Points: {points}")
print(f"$\\gamma = {gamma}$")

# Calculate kernel matrix
K = rbf_kernel_manual(points, gamma=gamma)

print("\nKernel Matrix $K(x,z) = \\exp(-\\gamma \\|x-z\\|^2)$:")
print(K)

# Calculate distances for verification
distances = cdist(points, points, metric='euclidean')
distances_squared = distances**2

print("\nPairwise Euclidean Distances:")
print(distances)

print("\nPairwise Squared Euclidean Distances:")
print(distances_squared)

print("\nDetailed calculations:")
for i in range(len(points)):
    for j in range(len(points)):
        x = points[i]
        z = points[j]
        dist_sq = np.sum((x - z)**2)
        kernel_val = np.exp(-gamma * dist_sq)
        print(f"K({x}, {z}) = exp(-{gamma} * {dist_sq:.2f}) = {kernel_val:.4f}")

# Task 2: Plot kernel value vs distance for different gamma values
print("\n" + "="*50)
print("TASK 2: Kernel Value vs Distance for Different $\\gamma$ Values")
print("="*50)

# Create distance range
distances_range = np.linspace(0, 5, 1000)
gamma_values = [0.1, 1, 10]

plt.figure(figsize=(12, 8))

for gamma_val in gamma_values:
    kernel_values = np.exp(-gamma_val * distances_range**2)
    plt.plot(distances_range, kernel_values, linewidth=2, 
             label=f'$\\gamma = {gamma_val}$')

plt.xlabel(r'Distance $\|\mathbf{x} - \mathbf{z}\|$')
plt.ylabel(r'Kernel Value $K(\mathbf{x}, \mathbf{z})$')
plt.title(r'RBF Kernel: $K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma \|\mathbf{x} - \mathbf{z}\|^2)$')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(0, 5)
plt.ylim(0, 1)

# Add some key points
for gamma_val in gamma_values:
    # Find distance where kernel value = 0.5
    target_kernel = 0.5
    target_distance = np.sqrt(-np.log(target_kernel) / gamma_val)
    plt.plot(target_distance, target_kernel, 'ro', markersize=8)
    plt.annotate(f'$\\gamma={gamma_val}$\nd={target_distance:.2f}', 
                (target_distance, target_kernel), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'rbf_kernel_vs_distance.png'), dpi=300, bbox_inches='tight')

# Task 3: Visualize decision boundary complexity with different gamma
print("\n" + "="*50)
print("TASK 3: Decision Boundary Complexity with Different $\\gamma$")
print("="*50)

# Create synthetic data for visualization
np.random.seed(42)
X_positive = np.random.multivariate_normal([1, 1], [[0.3, 0], [0, 0.3]], 20)
X_negative = np.random.multivariate_normal([3, 3], [[0.3, 0], [0, 0.3]], 20)
X = np.vstack([X_positive, X_negative])
y = np.hstack([np.ones(20), -np.ones(20)])

# Create mesh for decision boundary
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Plot decision boundaries for different gamma values
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
gamma_vis = [0.1, 1, 10]

for idx, gamma_val in enumerate(gamma_vis):
    ax = axes[idx]
    
    # Calculate decision function for all points in mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    # For simplicity, we'll use a simple decision function based on kernel values
    # In practice, this would be the SVM decision function
    decision_values = np.zeros(len(mesh_points))
    
    for i, point in enumerate(mesh_points):
        # Calculate kernel values with all training points
        kernel_vals = rbf_kernel_manual(point.reshape(1, -1), X, gamma=gamma_val).flatten()
        # Simple decision: sum of kernel values weighted by labels
        decision_values[i] = np.sum(kernel_vals * y)
    
    decision_values = decision_values.reshape(xx.shape)
    
    # Plot decision boundary
    contour = ax.contour(xx, yy, decision_values, levels=[0], colors='red', linewidths=2)
    ax.contourf(xx, yy, decision_values, levels=[-100, 0, 100], 
                colors=['lightblue', 'lightpink'], alpha=0.3)
    
    # Plot data points
    ax.scatter(X_positive[:, 0], X_positive[:, 1], c='blue', s=50, label='Class +1')
    ax.scatter(X_negative[:, 0], X_negative[:, 1], c='red', s=50, label='Class -1')
    
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_title(f'Decision Boundary ($\\gamma = {gamma_val}$)')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'decision_boundary_complexity.png'), dpi=300, bbox_inches='tight')

# Task 4: Interpret kernel values as similarity scores
print("\n" + "="*50)
print("TASK 4: Kernel Values as Similarity Scores")
print("="*50)

# Calculate all pairwise similarities
similarities = []
for i in range(len(points)):
    for j in range(i+1, len(points)):
        x = points[i]
        z = points[j]
        similarity = K[i, j]
        distance = distances[i, j]
        similarities.append((i, j, x, z, similarity, distance))

# Sort by similarity (highest first)
similarities.sort(key=lambda x: x[4], reverse=True)

print("Point pairs ranked by similarity (kernel value):")
print("Rank | Points | Coordinates | Distance | Similarity")
print("-" * 60)
for rank, (i, j, x, z, similarity, distance) in enumerate(similarities, 1):
    print(f"{rank:4d} | ({i},{j})  | {x} vs {z} | {distance:8.3f} | {similarity:10.4f}")

# Visualize similarities
plt.figure(figsize=(10, 8))

# Plot points
colors = ['blue', 'red', 'green']
for i, point in enumerate(points):
    plt.scatter(point[0], point[1], c=colors[i], s=200, 
                label=f'Point {i}: {point}', edgecolor='black', linewidth=2)

# Draw lines between points with thickness proportional to similarity
for i, j, x, z, similarity, distance in similarities:
    plt.plot([x[0], z[0]], [x[1], z[1]], 'k-', 
             linewidth=similarity * 5, alpha=0.7)
    # Add similarity label
    mid_x = (x[0] + z[0]) / 2
    mid_y = (x[1] + z[1]) / 2
    plt.annotate(f'{similarity:.3f}', (mid_x, mid_y), 
                xytext=(5, 5), textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'Point Similarities (Line thickness $\propto$ Kernel value)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'point_similarities.png'), dpi=300, bbox_inches='tight')

# Task 5: Estimate appropriate gamma range for dataset with d_avg = 2
print("\n" + "="*50)
print("TASK 5: Appropriate $\\gamma$ Range for $d_{avg} = 2$")
print("="*50)

d_avg = 2.0

# For RBF kernel, we want kernel values to be meaningful
# Typically, we want kernel values to be between 0.1 and 0.9 for average distances
# K(x,z) = exp(-γ * d²) where d is the distance

# For d_avg = 2, we want:
# 0.1 ≤ exp(-γ * 2²) ≤ 0.9
# 0.1 ≤ exp(-4γ) ≤ 0.9

# Solving for γ:
# exp(-4γ) = 0.1 → -4γ = ln(0.1) → γ = -ln(0.1)/4 ≈ 0.576
# exp(-4γ) = 0.9 → -4γ = ln(0.9) → γ = -ln(0.9)/4 ≈ 0.026

gamma_min = -np.log(0.9) / (d_avg**2)
gamma_max = -np.log(0.1) / (d_avg**2)

print(f"Average pairwise distance: d_avg = {d_avg}")
print(f"Target kernel value range: [0.1, 0.9]")
print(f"Appropriate $\\gamma$ range: [{gamma_min:.3f}, {gamma_max:.3f}]")

# Verify with calculations
kernel_min = np.exp(-gamma_min * d_avg**2)
kernel_max = np.exp(-gamma_max * d_avg**2)

print(f"Verification:")
print(f"  $\\gamma = {gamma_min:.3f}$ → K(d={d_avg}) = {kernel_min:.3f}")
print(f"  $\\gamma = {gamma_max:.3f}$ → K(d={d_avg}) = {kernel_max:.3f}")

# Visualize the relationship
plt.figure(figsize=(12, 8))

# Plot kernel values for different distances
distances_test = np.linspace(0, 4, 1000)

plt.subplot(2, 1, 1)
for gamma_val in [gamma_min, gamma_max]:
    kernel_vals = np.exp(-gamma_val * distances_test**2)
    plt.plot(distances_test, kernel_vals, linewidth=2, 
             label=f'$\\gamma = {gamma_val:.3f}$')

plt.axvline(x=d_avg, color='red', linestyle='--', label=f'$d_{{avg}} = {d_avg}$')
plt.axhline(y=0.1, color='green', linestyle=':', label='$K = 0.1$')
plt.axhline(y=0.9, color='green', linestyle=':', label='$K = 0.9$')

plt.xlabel('Distance')
plt.ylabel('Kernel Value')
plt.title(r'Kernel Values for Recommended $\gamma$ Range')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot gamma vs kernel value at d_avg
plt.subplot(2, 1, 2)
gamma_range = np.linspace(0.01, 1, 1000)
kernel_at_davg = np.exp(-gamma_range * d_avg**2)

plt.plot(gamma_range, kernel_at_davg, linewidth=2)
plt.axhline(y=0.1, color='green', linestyle=':', label='$K = 0.1$')
plt.axhline(y=0.9, color='green', linestyle=':', label='$K = 0.9$')
plt.axvline(x=gamma_min, color='red', linestyle='--', label=f'$\\gamma_{{min}} = {gamma_min:.3f}$')
plt.axvline(x=gamma_max, color='red', linestyle='--', label=f'$\\gamma_{{max}} = {gamma_max:.3f}$')

plt.xlabel(r'$\gamma$')
plt.ylabel(f'Kernel Value at $d = {d_avg}$')
plt.title(r'Kernel Value at Average Distance vs $\gamma$')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'gamma_recommendation.png'), dpi=300, bbox_inches='tight')

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("1. Kernel Matrix calculated for $\\gamma = 0.5$")
print("2. Kernel vs distance plots created for $\\gamma = 0.1, 1, 10$")
print("3. Decision boundary complexity visualized")
print("4. Point similarities ranked and visualized")
print("5. Recommended $\\gamma$ range for $d_{avg} = 2$: [0.026, 0.576]")
print(f"\nAll plots saved to: {save_dir}")

plt.close()
