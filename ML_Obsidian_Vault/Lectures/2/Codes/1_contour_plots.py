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

# Example 9: Confidence Regions for Bivariate Normal
print("\nExample 9: Confidence Regions for Bivariate Normal")

# Create correlated bivariate normal
corr = 0.7
var1, var2 = 1.0, 1.0
cov = np.array([[var1, corr * np.sqrt(var1 * var2)], 
                [corr * np.sqrt(var1 * var2), var2]])
rv = stats.multivariate_normal([0, 0], cov)

# Create grid
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))
Z = rv.pdf(pos)

# Create contours corresponding to confidence regions
plt.figure(figsize=(10, 8))

# Calculate Mahalanobis distance for each point (squared)
# For bivariate normal with mean 0, this is (x,y) * inv(cov) * (x,y)^T
inv_cov = np.linalg.inv(cov)
mahalanobis2 = np.zeros_like(Z)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = np.array([X[i,j], Y[i,j]])
        mahalanobis2[i,j] = point @ inv_cov @ point.T

# Get chi-square values for different confidence levels
confidence_levels = [0.5, 0.75, 0.9, 0.95]
chi2_values = stats.chi2.ppf(confidence_levels, df=2)

# Plot the confidence regions
colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072']
alpha_levels = [0.8, 0.6, 0.4, 0.2]

# Plot from highest confidence (smallest region) to lowest
for i, (chi2, color, alpha) in enumerate(zip(chi2_values, colors, alpha_levels)):
    mask = mahalanobis2 <= chi2
    plt.contourf(X, Y, mask.astype(float), levels=[0.5, 1.5], colors=[color], alpha=alpha)
    plt.contour(X, Y, mask.astype(float), levels=[0.5], colors='black', linewidths=1)

# Add text to indicate each region
region_labels = ["50%", "75%", "90%", "95%"]
for i, label in enumerate(region_labels):
    plt.text(2.0, 2.0 - i*0.3, f"{label} Confidence Region", 
             bbox=dict(facecolor=colors[i], alpha=0.8, boxstyle='round', pad=0.3))

plt.title('Confidence Regions for Bivariate Normal Distribution')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'confidence_regions_contour.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 10: Surface vs Contour Plot
print("\nExample 10: Surface vs Contour Plot")

# Create bivariate normal for visualization
rv = stats.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]])
Z = rv.pdf(pos)

fig = plt.figure(figsize=(15, 6))

# 3D Surface plot
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax1.set_title('3D PDF Surface')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Density')

# Contour plot
ax2 = fig.add_subplot(1, 2, 2)
contour = ax2.contourf(X, Y, Z, levels=20, cmap='viridis')
ax2.contour(X, Y, Z, levels=10, colors='black', alpha=0.3)
ax2.set_title('Equivalent Contour Plot')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.grid(alpha=0.3)
ax2.axis('equal')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'surface_vs_contour.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 11: Bivariate Normal with Different Configurations
print("\nExample 11: Bivariate Normal with Different Configurations")

plt.figure(figsize=(12, 10))

# 1. Uncorrelated with equal variances
plt.subplot(2, 2, 1)
cov1 = np.array([[1.0, 0.0], [0.0, 1.0]])
rv1 = stats.multivariate_normal([0, 0], cov1)
Z1 = rv1.pdf(pos)
plt.contourf(X, Y, Z1, levels=15, cmap='Blues')
plt.contour(X, Y, Z1, levels=10, colors='k', alpha=0.5)
plt.title('Uncorrelated (ρ=0)\nEqual Variances')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.grid(alpha=0.3)

# 2. Positively correlated
plt.subplot(2, 2, 2)
cov2 = np.array([[1.0, 0.7], [0.7, 1.0]])
rv2 = stats.multivariate_normal([0, 0], cov2)
Z2 = rv2.pdf(pos)
plt.contourf(X, Y, Z2, levels=15, cmap='Greens')
plt.contour(X, Y, Z2, levels=10, colors='k', alpha=0.5)
plt.title('Positively Correlated (ρ=0.7)\nEqual Variances')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.grid(alpha=0.3)

# 3. Negatively correlated
plt.subplot(2, 2, 3)
cov3 = np.array([[1.0, -0.7], [-0.7, 1.0]])
rv3 = stats.multivariate_normal([0, 0], cov3)
Z3 = rv3.pdf(pos)
plt.contourf(X, Y, Z3, levels=15, cmap='Reds')
plt.contour(X, Y, Z3, levels=10, colors='k', alpha=0.5)
plt.title('Negatively Correlated (ρ=-0.7)\nEqual Variances')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.grid(alpha=0.3)

# 4. Uncorrelated with different variances
plt.subplot(2, 2, 4)
cov4 = np.array([[2.0, 0.0], [0.0, 0.5]])
rv4 = stats.multivariate_normal([0, 0], cov4)
Z4 = rv4.pdf(pos)
plt.contourf(X, Y, Z4, levels=15, cmap='Purples')
plt.contour(X, Y, Z4, levels=10, colors='k', alpha=0.5)
plt.title('Uncorrelated (ρ=0)\nDifferent Variances')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'bivariate_normal_contours.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 12: Gaussian Mixture Model
print("\nExample 12: Gaussian Mixture Model")

# Define three Gaussian components
means = [[-2, -2], [0, 1], [2, -1]]
covs = [[[0.8, 0.2], [0.2, 0.8]], 
        [[0.7, -0.3], [-0.3, 0.7]], 
        [[0.8, 0.0], [0.0, 0.8]]]
weights = [0.3, 0.4, 0.3]  # Component weights

# Create grid
x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Calculate mixture density
Z_mix = np.zeros_like(X)
for mean, cov, weight in zip(means, covs, weights):
    rv = stats.multivariate_normal(mean, cov)
    Z_mix += weight * rv.pdf(pos)

plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z_mix, levels=20, cmap='viridis')
plt.colorbar(label='Probability Density')
plt.contour(X, Y, Z_mix, levels=15, colors='k', alpha=0.3)

# Mark component centers
for i, mean in enumerate(means):
    plt.plot(mean[0], mean[1], 'ro', markersize=10, label=f'Component {i+1}' if i==0 else f'Component {i+1}')

plt.title('Gaussian Mixture Model with Three Components')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'gaussian_mixture_contours.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 13: Optimization Landscape
print("\nExample 13: Optimization Landscape")

# Create a function with multiple local minima
def optimization_function(x, y):
    return np.sin(x*0.5)**2 + np.sin(y*0.5)**2 + 0.2*np.sin(x*2)*np.sin(y*2) + 0.1*(x**2 + y**2)

# Create grid
x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)
Z_opt = optimization_function(X, Y)

# Find local minima
from scipy.signal import argrelextrema
local_min_indices = argrelextrema(Z_opt, np.less, order=5)
local_min_x = X[local_min_indices]
local_min_y = Y[local_min_indices]
local_min_z = Z_opt[local_min_indices]

plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z_opt, levels=30, cmap='viridis')
plt.colorbar(label='Function Value')
plt.contour(X, Y, Z_opt, levels=20, colors='k', alpha=0.3)

# Mark local minima
plt.scatter(local_min_x, local_min_y, color='red', s=80, marker='o', label='Local Minima')

plt.title('Optimization Landscape with Multiple Local Minima')
plt.xlabel('Parameter 1')
plt.ylabel('Parameter 2')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'optimization_landscape_contour.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 14: Non-Gaussian Distribution
print("\nExample 14: Non-Gaussian Distribution")

# Create a non-Gaussian distribution (gamma-like)
def gamma_like_bivariate(x, y, alpha1=2, beta1=1, alpha2=3, beta2=1.5, rho=0.5):
    # Convert to positive domain
    x_pos = np.maximum(x, 0.001)
    y_pos = np.maximum(y, 0.001)
    
    # Create marginals (gamma-like)
    x_term = (x_pos**(alpha1-1)) * np.exp(-beta1 * x_pos)
    y_term = (y_pos**(alpha2-1)) * np.exp(-beta2 * y_pos)
    
    # Add dependence through a simplified version
    # This is not a proper copula but gives correlation
    interaction = np.exp(rho * np.sqrt(x_pos * y_pos) / (beta1 * beta2))
    
    return x_term * y_term * interaction

# Create grid (positive domain focused)
x = np.linspace(0, 5, 200)
y = np.linspace(0, 5, 200)
X, Y = np.meshgrid(x, y)
Z_gamma = gamma_like_bivariate(X, Y)

# Normalize for visualization
Z_gamma = Z_gamma / np.sum(Z_gamma)
Z_gamma = Z_gamma / np.max(Z_gamma)

plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z_gamma, levels=20, cmap='YlOrRd')
plt.colorbar(label='Relative Density')
plt.contour(X, Y, Z_gamma, levels=15, colors='k', alpha=0.3)
plt.title('Non-Gaussian Bivariate Distribution (Gamma-like)')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'non_gaussian_contours.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 15: Volcano-like Function Contour Plot
print("\nExample 15: Volcano-like Function Contour Plot")

# Create a volcano-like function
def volcano_function(x, y):
    r = np.sqrt(x**2 + y**2)
    return np.exp(-r**2) * np.sin(5*r) * 5

# Create grid
x = np.linspace(-3, 3, 200)
y = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x, y)
Z_volcano = volcano_function(X, Y)

plt.figure(figsize=(15, 6))

# 3D Surface
ax1 = plt.subplot(1, 2, 1, projection='3d')
surf = ax1.plot_surface(X, Y, Z_volcano, cmap='plasma', alpha=0.8)
ax1.set_title('3D Volcano Function')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
plt.colorbar(surf, ax=ax1, shrink=0.6, aspect=10)

# Contour plot
ax2 = plt.subplot(1, 2, 2)
contour = ax2.contourf(X, Y, Z_volcano, levels=30, cmap='plasma')
ax2.contour(X, Y, Z_volcano, levels=15, colors='black', alpha=0.5)
ax2.set_title('Contour Plot of Volcano Function')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.grid(alpha=0.3)
plt.colorbar(contour, ax=ax2)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'volcano_function_contour.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 16: Comparison of Contour Line Styles
print("\nExample 16: Comparison of Contour Line Styles")

# Create a simple function
def ripple_function(x, y):
    return np.sin(x*y/2) * np.exp(-(x**2 + y**2)/10)

# Create grid
x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)
Z_ripple = ripple_function(X, Y)

plt.figure(figsize=(15, 10))

# Standard contour lines
plt.subplot(2, 2, 1)
plt.contour(X, Y, Z_ripple, levels=15, colors='black')
plt.title('Standard Contour Lines')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(alpha=0.3)

# Filled contour with thin lines
plt.subplot(2, 2, 2)
plt.contourf(X, Y, Z_ripple, levels=15, cmap='viridis')
plt.contour(X, Y, Z_ripple, levels=15, colors='black', alpha=0.3, linewidths=0.5)
plt.title('Filled Contour with Thin Lines')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(alpha=0.3)

# Different line styles
plt.subplot(2, 2, 3)
plt.contour(X, Y, Z_ripple, levels=15, colors='black', linestyles=['solid', 'dashed', 'dotted', 'dashdot'])
plt.title('Different Line Styles')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(alpha=0.3)

# Colored contour lines
plt.subplot(2, 2, 4)
cs = plt.contour(X, Y, Z_ripple, levels=15, cmap='coolwarm')
plt.clabel(cs, inline=True, fontsize=8)
plt.title('Colored and Labeled Contour Lines')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'contour_line_styles.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll contour plot visualizations created successfully.") 