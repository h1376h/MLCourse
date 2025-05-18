import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_5_Quiz_10")
os.makedirs(save_dir, exist_ok=True)

# Set the style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True  # Enable LaTeX rendering

# Part 1: Compare Perceptron and Logistic Regression objectives
# ===================================================================

def perceptron_loss(y, z):
    """
    Perceptron loss function: max(0, -y*z)
    
    Args:
        y: True label {-1, 1}
        z: Model prediction before activation (wx + b)
    """
    return np.maximum(0, -y * z)

def logistic_loss(y, z):
    """
    Logistic regression loss function: log(1 + exp(-y*z))
    
    Args:
        y: True label {-1, 1}
        z: Model prediction before activation (wx + b)
    """
    # Use a numerically stable version of the logistic loss
    return np.log(1 + np.exp(-y * z))

# Figure 1: Compare perceptron and logistic regression loss functions
plt.figure(figsize=(10, 6))

z = np.linspace(-5, 5, 1000)  # Model predictions
y = 1  # For positive class

plt_perceptron = plt.plot(z, perceptron_loss(y, z), 'b-', linewidth=2, label='Perceptron Loss')
plt_logistic = plt.plot(z, logistic_loss(y, z), 'r-', linewidth=2, label='Logistic Loss')

plt.grid(True)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.xlabel(r'$z = \mathbf{w}^T\mathbf{x} + b$', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Comparison of Perceptron and Logistic Regression Loss Functions (y=1)', fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()

# Save the figure
plt.savefig(os.path.join(save_dir, 'perceptron_vs_logistic_loss.png'), dpi=300, bbox_inches='tight')

# Part 2: Visualization of loss landscapes with L1 and L2 regularization
# ======================================================================

# Create a 2D parameter space (w1, w2)
w1 = np.linspace(-5, 5, 100)
w2 = np.linspace(-5, 5, 100)
W1, W2 = np.meshgrid(w1, w2)

# Base loss function (e.g., simplified MSE for illustration)
def base_loss(w1, w2):
    """Simplified base loss function for demonstration"""
    # Quadratic bowl centered at (2, 3)
    return (w1 - 2)**2 + (w2 - 3)**2 + 1

# Loss functions with different regularization
def loss_no_reg(w1, w2):
    return base_loss(w1, w2)

def loss_l1_reg(w1, w2, lambda_=1):
    return base_loss(w1, w2) + lambda_ * (np.abs(w1) + np.abs(w2))

def loss_l2_reg(w1, w2, lambda_=1):
    return base_loss(w1, w2) + lambda_ * (w1**2 + w2**2)

# Compute loss values for each point in the grid
Z_no_reg = loss_no_reg(W1, W2)
Z_l1_reg = loss_l1_reg(W1, W2)
Z_l2_reg = loss_l2_reg(W1, W2)

# Figure 2: 3D visualization of regularization effects
fig = plt.figure(figsize=(18, 6))

# No regularization
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(W1, W2, Z_no_reg, cmap=cm.viridis, alpha=0.8, linewidth=0, antialiased=True)
ax1.set_xlabel(r'$w_1$', fontsize=14)
ax1.set_ylabel(r'$w_2$', fontsize=14)
ax1.set_zlabel('Loss', fontsize=14)
ax1.set_title('Without Regularization', fontsize=16)

# L1 regularization
ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(W1, W2, Z_l1_reg, cmap=cm.viridis, alpha=0.8, linewidth=0, antialiased=True)
ax2.set_xlabel(r'$w_1$', fontsize=14)
ax2.set_ylabel(r'$w_2$', fontsize=14)
ax2.set_zlabel('Loss', fontsize=14)
ax2.set_title('With L1 Regularization', fontsize=16)

# L2 regularization
ax3 = fig.add_subplot(133, projection='3d')
surf3 = ax3.plot_surface(W1, W2, Z_l2_reg, cmap=cm.viridis, alpha=0.8, linewidth=0, antialiased=True)
ax3.set_xlabel(r'$w_1$', fontsize=14)
ax3.set_ylabel(r'$w_2$', fontsize=14)
ax3.set_zlabel('Loss', fontsize=14)
ax3.set_title('With L2 Regularization', fontsize=16)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'regularization_3d.png'), dpi=300, bbox_inches='tight')

# Figure 3: Contour plots for easier visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# No regularization
contour1 = axes[0].contour(W1, W2, Z_no_reg, 20, cmap='viridis')
axes[0].clabel(contour1, inline=1, fontsize=8)
axes[0].set_xlabel(r'$w_1$', fontsize=14)
axes[0].set_ylabel(r'$w_2$', fontsize=14)
axes[0].set_title('Without Regularization', fontsize=16)
axes[0].grid(True)
axes[0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axes[0].axvline(x=0, color='k', linestyle='-', alpha=0.3)

# L1 regularization
contour2 = axes[1].contour(W1, W2, Z_l1_reg, 20, cmap='viridis')
axes[1].clabel(contour2, inline=1, fontsize=8)
axes[1].set_xlabel(r'$w_1$', fontsize=14)
axes[1].set_ylabel(r'$w_2$', fontsize=14)
axes[1].set_title('With L1 Regularization', fontsize=16)
axes[1].grid(True)
axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axes[1].axvline(x=0, color='k', linestyle='-', alpha=0.3)

# L2 regularization
contour3 = axes[2].contour(W1, W2, Z_l2_reg, 20, cmap='viridis')
axes[2].clabel(contour3, inline=1, fontsize=8)
axes[2].set_xlabel(r'$w_1$', fontsize=14)
axes[2].set_ylabel(r'$w_2$', fontsize=14)
axes[2].set_title('With L2 Regularization', fontsize=16)
axes[2].grid(True)
axes[2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axes[2].axvline(x=0, color='k', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'regularization_contours.png'), dpi=300, bbox_inches='tight')

# Part 3: Coordinate Descent vs. Gradient Descent for L1 regularization
# ====================================================================

# Create a function with a sharper minimum to better illustrate the differences
def l1_objective(w1, w2, lambda_=1.0):
    """L1 regularized objective function"""
    return (w1 - 2)**2 + (w2 - 2)**2 + lambda_ * (np.abs(w1) + np.abs(w2))

# Implementation of gradient descent for L1 regularization
def gd_l1_step(w1, w2, lr=0.1, lambda_=1.0):
    """One step of gradient descent for L1 regularized objective"""
    # Compute gradients (using subgradients for L1 term)
    grad_w1 = 2 * (w1 - 2) + lambda_ * np.sign(w1)
    grad_w2 = 2 * (w2 - 2) + lambda_ * np.sign(w2)
    
    # Update parameters
    w1_new = w1 - lr * grad_w1
    w2_new = w2 - lr * grad_w2
    
    return w1_new, w2_new

# Implementation of coordinate descent for L1 regularization
def cd_l1_step(w1, w2, lambda_=1.0):
    """One step of coordinate descent for L1 regularized objective"""
    # Update w1 (keeping w2 fixed)
    if 2 * (w1 - 2) < -lambda_:
        w1_new = w1 + 0.2  # Move w1 upward
    elif 2 * (w1 - 2) > lambda_:
        w1_new = w1 - 0.2  # Move w1 downward
    else:
        w1_new = 0  # Set w1 to exactly zero
    
    # Update w2 (keeping w1 fixed)
    if 2 * (w2 - 2) < -lambda_:
        w2_new = w2 + 0.2  # Move w2 upward
    elif 2 * (w2 - 2) > lambda_:
        w2_new = w2 - 0.2  # Move w2 downward
    else:
        w2_new = 0  # Set w2 to exactly zero
    
    return w1_new, w2_new

# Generate trajectories
def generate_trajectory(start_w1, start_w2, step_fn, num_steps=10, **kwargs):
    """Generate a trajectory of parameter values using the provided step function"""
    w1_hist = [start_w1]
    w2_hist = [start_w2]
    w1, w2 = start_w1, start_w2
    
    for _ in range(num_steps):
        w1, w2 = step_fn(w1, w2, **kwargs)
        w1_hist.append(w1)
        w2_hist.append(w2)
    
    return np.array(w1_hist), np.array(w2_hist)

# Starting point away from the optimum
start_w1, start_w2 = 4.5, 4.5

# Generate trajectories
gd_w1_hist, gd_w2_hist = generate_trajectory(start_w1, start_w2, gd_l1_step, num_steps=15)
cd_w1_hist, cd_w2_hist = generate_trajectory(start_w1, start_w2, cd_l1_step, num_steps=15)

# Calculate contours of the L1 objective function
Z_l1 = np.zeros_like(W1)
for i in range(len(w1)):
    for j in range(len(w2)):
        Z_l1[j, i] = l1_objective(w1[i], w2[j])

# Figure 4: Comparison of Gradient Descent and Coordinate Descent for L1 regularization
plt.figure(figsize=(12, 10))

# Plot contours of the L1 objective
contour = plt.contour(W1, W2, Z_l1, 20, cmap='viridis', alpha=0.6)
plt.clabel(contour, inline=1, fontsize=8)

# Plot trajectories
plt.plot(gd_w1_hist, gd_w2_hist, 'r.-', linewidth=2, markersize=10, label='Gradient Descent')
plt.plot(cd_w1_hist, cd_w2_hist, 'b.-', linewidth=2, markersize=10, label='Coordinate Descent')

# Mark starting point
plt.plot(start_w1, start_w2, 'ko', markersize=12, label='Starting Point')

# Highlight the L1 regularization effect - the sparse region where parameters become exactly zero
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.axhspan(-0.5, 0.5, alpha=0.1, color='green', label='Sparse Region ($w_2\\approx0$)')
plt.axvspan(-0.5, 0.5, alpha=0.1, color='orange', label='Sparse Region ($w_1\\approx0$)')

plt.grid(True)
plt.xlabel(r'$w_1$', fontsize=14)
plt.ylabel(r'$w_2$', fontsize=14)
plt.title('Gradient Descent vs. Coordinate Descent for L1-Regularized Objective', fontsize=16)
plt.legend(fontsize=12, loc='upper right')

# Annotate key observations
plt.annotate('CD reaches sparse\nsolution', xy=(cd_w1_hist[-1], cd_w2_hist[-1]), 
             xytext=(cd_w1_hist[-1]-1, cd_w2_hist[-1]+1), 
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
             fontsize=12)

plt.annotate('GD oscillates near\naxis boundaries', xy=(gd_w1_hist[-1], gd_w2_hist[-1]),
             xytext=(gd_w1_hist[-1]+1, gd_w2_hist[-1]+1),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
             fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'gd_vs_cd_l1.png'), dpi=300, bbox_inches='tight')

# Figure 5: Objective function value comparison
plt.figure(figsize=(10, 6))

# Calculate objective function values along each trajectory
gd_obj_vals = [l1_objective(gd_w1_hist[i], gd_w2_hist[i]) for i in range(len(gd_w1_hist))]
cd_obj_vals = [l1_objective(cd_w1_hist[i], cd_w2_hist[i]) for i in range(len(cd_w1_hist))]

# Plot objective function values
plt.plot(range(len(gd_obj_vals)), gd_obj_vals, 'r.-', linewidth=2, label='Gradient Descent')
plt.plot(range(len(cd_obj_vals)), cd_obj_vals, 'b.-', linewidth=2, label='Coordinate Descent')

plt.grid(True)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Objective Function Value', fontsize=14)
plt.title('Convergence of GD vs CD for L1-Regularized Objective', fontsize=16)
plt.legend(fontsize=12)

# Annotate key observations
plt.annotate('CD reaches lower objective\nvalue and exact sparsity', xy=(len(cd_obj_vals)-1, cd_obj_vals[-1]),
             xytext=(len(cd_obj_vals)-5, cd_obj_vals[-1]+5),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
             fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'gd_vs_cd_convergence.png'), dpi=300, bbox_inches='tight')

# Print key insights to the console
print("\nKey Insights for Optimization Objectives:")
print("==========================================")

print("\n1. Perceptron vs Logistic Regression:")
print("   * Perceptron Objective: Minimizes misclassification errors with hinge loss (max(0, -y*z))")
print("   * Logistic Regression Objective: Minimizes log loss (log(1 + exp(-y*z)))")
print("   * Difference: Perceptron loss is zero for correctly classified points, while logistic regression")
print("     always has non-zero gradients even for correctly classified points, allowing it to")
print("     continue improving the model's confidence.")

print("\n2. L1 Regularization Effect:")
print("   * Adds a term λ||w||₁ to the objective function")
print("   * Creates corners and edges in the loss landscape")
print("   * Promotes sparse solutions by pushing parameters exactly to zero")
print("   * Makes the optimization landscape non-differentiable at w=0")

print("\n3. L2 Regularization Effect:")
print("   * Adds a term λ||w||₂² to the objective function")
print("   * Creates a more bowl-shaped, smoother loss landscape")
print("   * Shrinks weights proportionally but rarely to exactly zero")
print("   * Remains differentiable everywhere")

print("\n4. Coordinate Descent for L1-Regularization:")
print("   * Advantages:")
print("     - Can move exactly to zero along one coordinate")
print("     - Handles the non-differentiability of L1 term naturally")
print("     - Often finds sparser solutions than gradient descent")
print("     - More efficient computation of the soft-thresholding operation")
print("   * Gradient descent oscillates near zero due to subgradient approximation")
print("   * L1 regularization with coordinate descent is the foundation of LASSO")

print("\nFigures saved to:", save_dir)

# plt.show() # Uncomment to display figures 