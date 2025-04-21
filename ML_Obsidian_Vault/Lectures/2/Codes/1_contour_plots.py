import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

print("\n=== CONTOUR PLOTS: VISUALIZATIONS ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images/Contour_Plots relative to the parent directory
images_dir = os.path.join(parent_dir, "Images", "Contour_Plots")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Example 1: Simple Contour Plot
print("Example 1: Simple Contour Plot")

# Create grid of points
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

# Create a simple function
Z = np.exp(-(X**2 + Y**2) / 2)

# Create figure
plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, levels=10, colors='black')
plt.title('Simple Contour Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'simple_contour.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Filled Contour Plot
print("\nExample 2: Filled Contour Plot")

plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z, levels=10, cmap='viridis')
plt.colorbar(label='Value')
plt.contour(X, Y, Z, levels=10, colors='black', alpha=0.3)
plt.title('Filled Contour Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'filled_contour.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Labeled Contour Plot
print("\nExample 3: Labeled Contour Plot")

plt.figure(figsize=(10, 8))
cs = plt.contour(X, Y, Z, levels=10, colors='black')
plt.clabel(cs, inline=True, fontsize=10)
plt.title('Labeled Contour Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'labeled_contour.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 4: Different Level Spacings
print("\nExample 4: Different Level Spacings")

plt.figure(figsize=(15, 5))

# Linear spacing
plt.subplot(1, 3, 1)
levels = np.linspace(0, 1, 10)
plt.contourf(X, Y, Z, levels=levels, cmap='viridis')
plt.contour(X, Y, Z, levels=levels, colors='black', alpha=0.3)
plt.title('Linear Levels')
plt.axis('equal')
plt.grid(alpha=0.3)

# Logarithmic spacing
plt.subplot(1, 3, 2)
levels = np.logspace(-5, 0, 10)
plt.contourf(X, Y, Z, levels=levels, cmap='viridis')
plt.contour(X, Y, Z, levels=levels, colors='black', alpha=0.3)
plt.title('Logarithmic Levels')
plt.axis('equal')
plt.grid(alpha=0.3)

# Custom spacing
plt.subplot(1, 3, 3)
levels = [0.1, 0.3, 0.5, 0.7, 0.9]
plt.contourf(X, Y, Z, levels=levels, cmap='viridis')
plt.contour(X, Y, Z, levels=levels, colors='black', alpha=0.3)
plt.title('Custom Levels')
plt.axis('equal')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'contour_level_spacing.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 5: Contour Plot with Different Color Maps
print("\nExample 5: Contour Plot with Different Color Maps")

plt.figure(figsize=(15, 10))
cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Blues']

for i, cmap in enumerate(cmaps):
    plt.subplot(2, 3, i+1)
    plt.contourf(X, Y, Z, levels=10, cmap=cmap)
    plt.contour(X, Y, Z, levels=10, colors='black', alpha=0.3)
    plt.title(f'Colormap: {cmap}')
    plt.axis('equal')
    plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'contour_colormaps.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 6: 3D Surface with Projected Contour
print("\nExample 6: 3D Surface with Projected Contour")

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
cset = ax.contour(X, Y, Z, zdir='z', offset=-0.5, cmap='viridis')
cset = ax.contour(X, Y, Z, zdir='x', offset=-3, cmap='viridis')
cset = ax.contour(X, Y, Z, zdir='y', offset=3, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Surface with Projected Contours')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-0.5, 1)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'contour_3d_projection.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 7: Decision Boundary Visualization
print("\nExample 7: Decision Boundary Visualization")

# Create synthetic data for a binary classification problem
np.random.seed(42)
n_samples = 100
# Class 1: Samples from a normal distribution centered at (1, 1)
X1 = np.random.normal(1, 1, (n_samples, 2))
# Class 2: Samples from a normal distribution centered at (-1, -1)
X2 = np.random.normal(-1, 1, (n_samples, 2))
X_data = np.vstack([X1, X2])
y_data = np.hstack([np.ones(n_samples), np.zeros(n_samples)])

# Create a simple decision boundary function (logistic regression-like)
def decision_function(x, y):
    return 1/(1 + np.exp(-(x + y)))

Z_decision = decision_function(X, Y)

plt.figure(figsize=(10, 8))
# Plot the decision boundary
plt.contourf(X, Y, Z_decision, levels=20, cmap='RdBu', alpha=0.7)
plt.colorbar(label='Probability of class 1')
plt.contour(X, Y, Z_decision, levels=[0.5], colors='black', linewidths=2)
# Plot the data points
plt.scatter(X1[:, 0], X1[:, 1], c='red', label='Class 1', edgecolors='k')
plt.scatter(X2[:, 0], X2[:, 1], c='blue', label='Class 2', edgecolors='k')
plt.title('Decision Boundary Visualization')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'decision_boundary_contour.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 8: Contours for a Saddle Function
print("\nExample 8: Contours for a Saddle Function")

# Create a saddle function
Z_saddle = X**2 - Y**2

plt.figure(figsize=(15, 5))

# 3D Surface
ax1 = plt.subplot(1, 2, 1, projection='3d')
surf = ax1.plot_surface(X, Y, Z_saddle, cmap='coolwarm', alpha=0.8)
ax1.set_title('3D Saddle Function')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
plt.colorbar(surf, ax=ax1, shrink=0.6, aspect=10)

# Contour plot
ax2 = plt.subplot(1, 2, 2)
contour = ax2.contourf(X, Y, Z_saddle, levels=20, cmap='coolwarm')
ax2.contour(X, Y, Z_saddle, levels=20, colors='black', alpha=0.3)
ax2.contour(X, Y, Z_saddle, levels=[0], colors='black', linewidths=2)
ax2.set_title('Contour Plot of Saddle Function')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.grid(alpha=0.3)
plt.colorbar(contour, ax=ax2)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'saddle_function_contour.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll contour plot visualizations created successfully.") 