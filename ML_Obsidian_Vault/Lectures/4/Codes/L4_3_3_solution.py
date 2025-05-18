import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_3_Quiz_3")
os.makedirs(save_dir, exist_ok=True)

# Set the style for the plots and enable LaTeX support
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['text.usetex'] = True  # Enable LaTeX rendering
plt.rcParams['font.family'] = 'serif'

# Define the problem parameters for the logistic regression model
w0 = -3  # Bias term
w1 = 2   # Weight for x1
w2 = -1  # Weight for x2

# Step 1: Define the sigmoid function (logistic function)
def sigmoid(z):
    """
    Compute the sigmoid (logistic) function: 1 / (1 + exp(-z))
    
    Args:
        z: Input value or array
        
    Returns:
        Sigmoid output between 0 and 1
    """
    return 1.0 / (1.0 + np.exp(-z))

# Step 2: Define the posterior probability function P(y=1|x)
def posterior_probability(x1, x2):
    """
    Compute the posterior probability P(y=1|x) for logistic regression
    
    Args:
        x1: Feature 1 value or array
        x2: Feature 2 value or array
        
    Returns:
        Posterior probability P(y=1|x)
    """
    z = w0 + w1 * x1 + w2 * x2
    return sigmoid(z)

# Step 3: Calculate the decision boundary equation
# For logistic regression, the decision boundary occurs where P(y=1|x) = 0.5
# This happens when the sigmoid argument z = 0, or w0 + w1*x1 + w2*x2 = 0
# So the decision boundary equation is: w0 + w1*x1 + w2*x2 = 0
# Solving for x2: x2 = (-w0 - w1*x1) / w2

def decision_boundary(x1, threshold=0.5):
    """
    Calculate the decision boundary for a given threshold
    
    Args:
        x1: x1 values for the decision boundary
        threshold: Classification threshold (default: 0.5)
    
    Returns:
        x2 values for the decision boundary
    """
    # For threshold=0.5, we need z = 0, which gives w0 + w1*x1 + w2*x2 = 0
    # For other thresholds, we need sigmoid(z) = threshold, which gives:
    # z = log(threshold/(1-threshold))
    z_threshold = np.log(threshold / (1 - threshold))
    
    # Solve for x2: (z_threshold - w0 - w1*x1) / w2
    return (z_threshold - w0 - w1*x1) / w2

# Step 4: Create a grid of points for visualization
x1_min, x1_max = -2, 6
x2_min, x2_max = -2, 6
x1_grid = np.linspace(x1_min, x1_max, 1000)
x2_grid = np.linspace(x2_min, x2_max, 1000)
X1, X2 = np.meshgrid(x1_grid, x2_grid)

# Step 5: Calculate posterior probabilities for every point in the grid
Z = posterior_probability(X1, X2)

# Step 6: Create Figure 1: Decision Boundary with Threshold 0.5
plt.figure(figsize=(10, 8))

# Plot contours of posterior probability
contour = plt.contourf(X1, X2, Z, levels=20, cmap='coolwarm', alpha=0.8)
plt.colorbar(contour, label='$P(y=1|x)$')

# Plot the decision boundary (where P(y=1|x) = 0.5)
x1_boundary = np.linspace(x1_min, x1_max, 100)
x2_boundary = decision_boundary(x1_boundary)
plt.plot(x1_boundary, x2_boundary, 'k--', linewidth=2, label='Decision Boundary: $P(y=1|x) = 0.5$')

# Add the point (2, 1) to check its classification
x1_point, x2_point = 2, 1
prob_point = posterior_probability(x1_point, x2_point)
class_point = 1 if prob_point >= 0.5 else 0
plt.scatter(x1_point, x2_point, color='black', s=100, label=f'Point (2, 1): Class {class_point}')
plt.annotate(f'P(y=1|x) = {prob_point:.4f}', xy=(x1_point, x2_point), 
             xytext=(x1_point+0.5, x2_point+0.5),
             arrowprops=dict(facecolor='black', shrink=0.05))

# Plot the regions where the model predicts each class
decision_region = (Z >= 0.5).astype(int)
plt.contour(X1, X2, decision_region, levels=[0.5], colors='k', linestyles='--', linewidths=2)

# Add labels and legend
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Decision Boundary for Logistic Regression with Threshold = 0.5')
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
plt.grid(True)
plt.legend()

# Set axis limits
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)

# Save the figure
plt.savefig(os.path.join(save_dir, 'decision_boundary_05.png'), dpi=300, bbox_inches='tight')

# Step 7: Create Figure 2: Decision Boundary with Threshold 0.7
plt.figure(figsize=(10, 8))

# Plot contours of posterior probability
contour = plt.contourf(X1, X2, Z, levels=20, cmap='coolwarm', alpha=0.8)
plt.colorbar(contour, label='$P(y=1|x)$')

# Plot the decision boundary for threshold=0.5
x2_boundary_05 = decision_boundary(x1_boundary, threshold=0.5)
plt.plot(x1_boundary, x2_boundary_05, 'k--', linewidth=2, label='Decision Boundary: $P(y=1|x) = 0.5$')

# Plot the decision boundary for threshold=0.7
x2_boundary_07 = decision_boundary(x1_boundary, threshold=0.7)
plt.plot(x1_boundary, x2_boundary_07, 'r-', linewidth=2, label='Decision Boundary: $P(y=1|x) = 0.7$')

# Plot regions for threshold 0.7
decision_region_07 = (Z >= 0.7).astype(int)
plt.contour(X1, X2, decision_region_07, levels=[0.5], colors='r', linestyles='-', linewidths=2)

# Add the point (2, 1) again
class_point_07 = 1 if prob_point >= 0.7 else 0
plt.scatter(x1_point, x2_point, color='black', s=100, label=f'Point (2, 1): Class {class_point_07} with threshold=0.7')

# Add labels and legend
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Comparison of Decision Boundaries with Different Thresholds')
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
plt.grid(True)
plt.legend()

# Set axis limits
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)

# Save the figure
plt.savefig(os.path.join(save_dir, 'decision_boundary_comparison.png'), dpi=300, bbox_inches='tight')

# Step 8: Create Figure 3: 3D visualization of the posterior probability
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Downsample for clearer visualization
step = 20
X1_sparse = X1[::step, ::step]
X2_sparse = X2[::step, ::step]
Z_sparse = Z[::step, ::step]

# Plot the 3D surface of posterior probabilities
surf = ax.plot_surface(X1_sparse, X2_sparse, Z_sparse, cmap='coolwarm', alpha=0.8, linewidth=0, antialiased=True)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='$P(y=1|x)$')

# Add plane at z=0.5 to visualize the decision boundary
X1_plane = np.linspace(x1_min, x1_max, 10)
X2_plane = np.linspace(x2_min, x2_max, 10)
X1_plane, X2_plane = np.meshgrid(X1_plane, X2_plane)
Z_plane = np.full_like(X1_plane, 0.5)
ax.plot_surface(X1_plane, X2_plane, Z_plane, color='gray', alpha=0.5)

# Add plane at z=0.7 to visualize the second threshold
Z_plane_07 = np.full_like(X1_plane, 0.7)
ax.plot_surface(X1_plane, X2_plane, Z_plane_07, color='red', alpha=0.3)

# Add labels
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('Posterior Probability $P(y=1|x)$')
ax.set_title('3D Visualization of Posterior Probability Surface')

# Save the figure
plt.savefig(os.path.join(save_dir, '3d_posterior_probability.png'), dpi=300, bbox_inches='tight')

# Step 9: Create Figure 4: Decision regions comparison
plt.figure(figsize=(12, 10))

# Create a meshgrid for classification regions (sparser for better visualization)
x1_vis = np.linspace(x1_min, x1_max, 100)
x2_vis = np.linspace(x2_min, x2_max, 100)
X1_vis, X2_vis = np.meshgrid(x1_vis, x2_vis)
Z_vis = posterior_probability(X1_vis, X2_vis)

# Create decision regions for threshold 0.5
decision_region_05 = (Z_vis >= 0.5).astype(int)

# Create decision regions for threshold 0.7
decision_region_07 = (Z_vis >= 0.7).astype(int)

# Create a custom colormap for visualization
cmap = ListedColormap(['#FFAAAA', '#AAAAFF'])

# Create a subplot for threshold 0.5
plt.subplot(1, 2, 1)
plt.contourf(X1_vis, X2_vis, decision_region_05, cmap=cmap, alpha=0.8)
plt.contour(X1_vis, X2_vis, Z_vis, levels=[0.5], colors='k', linestyles='--', linewidths=2)
plt.scatter(x1_point, x2_point, color='black', s=100)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Decision Regions with Threshold = 0.5')
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
plt.grid(True)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)

# Create a subplot for threshold 0.7
plt.subplot(1, 2, 2)
plt.contourf(X1_vis, X2_vis, decision_region_07, cmap=cmap, alpha=0.8)
plt.contour(X1_vis, X2_vis, Z_vis, levels=[0.7], colors='r', linestyles='-', linewidths=2)
plt.scatter(x1_point, x2_point, color='black', s=100)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Decision Regions with Threshold = 0.7')
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
plt.grid(True)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)

# Add a common colorbar for both plots
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes([0.85, 0.1, 0.03, 0.8])
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cax)
cbar.set_ticks([0.25, 0.75])
cbar.set_ticklabels(['Class 0', 'Class 1'])

# Save the figure
plt.savefig(os.path.join(save_dir, 'decision_regions_comparison.png'), dpi=300, bbox_inches='tight')

# Step 10: Mathematical derivation (output to console)
print("\nQuestion 3: Logistic Regression Decision Boundary")
print("==================================================")
print("\nMathematical Derivation:")
print("-----------------------")
print("Step 1: The posterior probability is given as:")
print("P(y=1|x) = 1 / (1 + exp(-w₀ - w₁x₁ - w₂x₂))")
print(f"with w₀ = {w0}, w₁ = {w1}, w₂ = {w2}")

print("\nStep 2: For the decision boundary where P(y=1|x) = P(y=0|x) = 0.5:")
print("P(y=1|x) = 0.5")
print("1 / (1 + exp(-w₀ - w₁x₁ - w₂x₂)) = 0.5")
print("1 + exp(-w₀ - w₁x₁ - w₂x₂) = 2")
print("exp(-w₀ - w₁x₁ - w₂x₂) = 1")
print("-w₀ - w₁x₁ - w₂x₂ = 0")
print("w₀ + w₁x₁ + w₂x₂ = 0")

print(f"\nStep 3: Substituting our values (w₀ = {w0}, w₁ = {w1}, w₂ = {w2}):")
print(f"{w0} + {w1}x₁ + ({w2})x₂ = 0")
print(f"-3 + 2x₁ - x₂ = 0")

print("\nStep 4: Solving for x₂:")
print("x₂ = 2x₁ - 3")
print("This is the equation of our decision boundary for threshold = 0.5.")

print("\nStep 5: For the point (x₁, x₂) = (2, 1), we calculate:")
x1_check, x2_check = 2, 1
z_check = w0 + w1*x1_check + w2*x2_check
p_check = sigmoid(z_check)
print(f"z = w₀ + w₁x₁ + w₂x₂ = {w0} + {w1}×{x1_check} + {w2}×{x2_check} = {z_check}")
print(f"P(y=1|x) = 1/(1 + exp(-z)) = 1/(1 + exp({-z_check})) = {p_check:.6f}")
print(f"Since P(y=1|x) = {p_check:.6f} > 0.5, the point belongs to Class 1")

print("\nStep 6: For threshold = 0.7:")
print("P(y=1|x) = 0.7")
print("1 / (1 + exp(-w₀ - w₁x₁ - w₂x₂)) = 0.7")
print("1 + exp(-w₀ - w₁x₁ - w₂x₂) = 1/0.7")
print("exp(-w₀ - w₁x₁ - w₂x₂) = 1/0.7 - 1")
print("-w₀ - w₁x₁ - w₂x₂ = log(1/0.7 - 1)")
print("w₀ + w₁x₁ + w₂x₂ = -log(1/0.7 - 1)")

z_threshold_07 = np.log(0.7 / (1 - 0.7))
print(f"\nStep 7: log(0.7/0.3) = {z_threshold_07:.6f}")
print(f"This gives us: {w0} + {w1}x₁ + ({w2})x₂ = {z_threshold_07:.6f}")
print(f"-3 + 2x₁ - x₂ = {z_threshold_07:.6f}")
print(f"Solving for x₂: x₂ = 2x₁ - 3 - {z_threshold_07:.6f}")
print(f"x₂ = 2x₁ - {3 + z_threshold_07:.6f}")
print(f"This is the equation of our decision boundary for threshold = 0.7.")

print(f"\nFor the point (2, 1) with threshold = 0.7:")
print(f"P(y=1|x) = {p_check:.6f}")
print(f"Since P(y=1|x) = {p_check:.6f} < 0.7, the point now belongs to Class 0")

print("\nVisualization saved to:", save_dir)

# Comment out plt.show() for automation
# plt.show() 