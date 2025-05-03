import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_4_Quiz_8")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

print("Question 8: Loss Function Convexity")
print("==================================")

# Step 1: Define the loss functions
print("\nStep 1: Define the loss functions")
print("------------------------------")

def zero_one_loss(y, f_x):
    """
    0-1 Loss: 0 if y*f(x) > 0, 1 otherwise
    
    Parameters:
    -----------
    y : scalar or array
        True label (1 or -1)
    f_x : scalar or array
        Model prediction
        
    Returns:
    --------
    loss : scalar or array
        0-1 loss value
    """
    return np.where(y * f_x > 0, 0, 1)

def hinge_loss(y, f_x):
    """
    Hinge Loss: max(0, 1 - y*f(x))
    
    Parameters:
    -----------
    y : scalar or array
        True label (1 or -1)
    f_x : scalar or array
        Model prediction
        
    Returns:
    --------
    loss : scalar or array
        Hinge loss value
    """
    return np.maximum(0, 1 - y * f_x)

def logistic_loss(y, f_x):
    """
    Logistic Loss: log(1 + exp(-y*f(x)))
    
    Parameters:
    -----------
    y : scalar or array
        True label (1 or -1)
    f_x : scalar or array
        Model prediction
        
    Returns:
    --------
    loss : scalar or array
        Logistic loss value
    """
    # Use a stable implementation to avoid overflow
    z = -y * f_x
    # Use np.log1p (log(1+x)) for numerical stability
    return np.log1p(np.exp(z))

def squared_error_loss(y, f_x):
    """
    Squared Error Loss: (y - f(x))^2
    Note: This assumes y is {-1, 1} for consistency with other losses
    
    Parameters:
    -----------
    y : scalar or array
        True label (1 or -1)
    f_x : scalar or array
        Model prediction
        
    Returns:
    --------
    loss : scalar or array
        Squared error loss value
    """
    return (y - f_x) ** 2

print("Defined the following loss functions:")
print("1. 0-1 Loss: L(y, f(x)) = 0 if y*f(x) > 0, 1 otherwise")
print("2. Hinge Loss: L(y, f(x)) = max(0, 1 - y*f(x))")
print("3. Logistic Loss: L(y, f(x)) = log(1 + exp(-y*f(x)))")
print("4. Squared Error Loss: L(y, f(x)) = (y - f(x))^2")

# Step 2: Calculate losses for given example points
print("\nStep 2: Calculate losses for the given examples")
print("--------------------------------------------")

# Example 1: y = 1, f(x) = 0.5
y1 = 1
f_x1 = 0.5

# Example 2: y = -1, f(x) = -2
y2 = -1
f_x2 = -2

# Calculate losses for Example 1
zero_one_loss_1 = zero_one_loss(y1, f_x1)
hinge_loss_1 = hinge_loss(y1, f_x1)
logistic_loss_1 = logistic_loss(y1, f_x1)
squared_error_loss_1 = squared_error_loss(y1, f_x1)

# Calculate losses for Example 2
zero_one_loss_2 = zero_one_loss(y2, f_x2)
hinge_loss_2 = hinge_loss(y2, f_x2)
logistic_loss_2 = logistic_loss(y2, f_x2)
squared_error_loss_2 = squared_error_loss(y2, f_x2)

print("Example 1: y = 1, f(x) = 0.5")
print(f"0-1 Loss: {zero_one_loss_1}")
print(f"Hinge Loss: {hinge_loss_1}")
print(f"Logistic Loss: {logistic_loss_1:.4f}")
print(f"Squared Error Loss: {squared_error_loss_1}")

print("\nExample 2: y = -1, f(x) = -2")
print(f"0-1 Loss: {zero_one_loss_2}")
print(f"Hinge Loss: {hinge_loss_2}")
print(f"Logistic Loss: {logistic_loss_2:.4f}")
print(f"Squared Error Loss: {squared_error_loss_2}")

# Step 3: Visualize the loss functions for different values of z = y*f(x)
print("\nStep 3: Visualize the loss functions")
print("---------------------------------")

z = np.linspace(-3, 3, 1000)  # z = y*f(x)

# Calculate losses for each z
zero_one_losses = np.where(z > 0, 0, 1)
hinge_losses = np.maximum(0, 1 - z)
logistic_losses = np.log1p(np.exp(-z))
squared_error_losses = (1 - z) ** 2  # Assuming y=1 for simplicity

# Create a figure to compare all loss functions
plt.figure(figsize=(12, 8))
plt.plot(z, zero_one_losses, 'b-', linewidth=2, label='0-1 Loss')
plt.plot(z, hinge_losses, 'r-', linewidth=2, label='Hinge Loss')
plt.plot(z, logistic_losses, 'g-', linewidth=2, label='Logistic Loss')
plt.plot(z, squared_error_losses, 'm-', linewidth=2, label='Squared Error Loss')

# Add vertical line at z = 0 (decision boundary)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)

# Add horizontal line at y = 1 to show reference
plt.axhline(y=1, color='k', linestyle='--', alpha=0.5)

# Add markers for the example points
plt.scatter([y1 * f_x1], [hinge_loss_1], color='red', s=100, zorder=5, label='Example 1: y=1, f(x)=0.5')
plt.scatter([y2 * f_x2], [hinge_loss_2], color='blue', s=100, zorder=5, label='Example 2: y=-1, f(x)=-2')

plt.xlabel('z = y·f(x)', fontsize=14)
plt.ylabel('Loss Value', fontsize=14)
plt.title('Comparison of Loss Functions', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(-3, 3)
plt.ylim(-0.5, 4)

# Save the comparison plot
plt.savefig(os.path.join(save_dir, "loss_functions_comparison.png"), dpi=300, bbox_inches='tight')

# Step 4: Analyze convexity by looking at second derivatives
print("\nStep 4: Analyze convexity properties")
print("--------------------------------")

# Let's look at the second derivatives with respect to z = y*f(x)
def second_derivative_hinge(z):
    # Hinge loss is not differentiable at z = 1, but second derivative is 0 elsewhere
    return np.zeros_like(z)

def second_derivative_logistic(z):
    # Second derivative of logistic loss
    exp_z = np.exp(z)
    return exp_z / ((1 + exp_z) ** 2)

def second_derivative_squared_error(z):
    # Second derivative of squared error loss
    return 2 * np.ones_like(z)

# 0-1 loss has undefined derivatives (it's discontinuous)

# Create a figure to visualize the second derivatives
plt.figure(figsize=(12, 6))
z = np.linspace(-3, 3, 1000)

# Plot second derivatives
plt.plot(z, second_derivative_hinge(z), 'r-', linewidth=2, label='Hinge Loss (2nd deriv)')
plt.plot(z, second_derivative_logistic(-z), 'g-', linewidth=2, label='Logistic Loss (2nd deriv)')
plt.plot(z, second_derivative_squared_error(z), 'm-', linewidth=2, label='Squared Error (2nd deriv)')

plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.xlabel('z = y·f(x)', fontsize=14)
plt.ylabel('Second Derivative', fontsize=14)
plt.title('Second Derivatives of Loss Functions', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(-0.1, 2.1)

# Save the second derivatives plot
plt.savefig(os.path.join(save_dir, "second_derivatives.png"), dpi=300, bbox_inches='tight')

# Step 5: Show non-differentiability of hinge loss
print("\nStep 5: Analyze non-differentiability")
print("----------------------------------")

# Create a figure to focus on the non-differentiability of the hinge loss
plt.figure(figsize=(10, 6))
z = np.linspace(0.7, 1.3, 1000)
hinge_losses = np.maximum(0, 1 - z)

plt.plot(z, hinge_losses, 'r-', linewidth=3, label='Hinge Loss')
plt.axvline(x=1, color='k', linestyle='--', alpha=0.7, label='Non-differentiable point at z=1')

# Add zoomed inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
axins = zoomed_inset_axes(plt.gca(), 4, loc='upper right')
axins.plot(z, hinge_losses, 'r-', linewidth=3)
axins.axvline(x=1, color='k', linestyle='--', alpha=0.7)
axins.set_xlim(0.95, 1.05)
axins.set_ylim(-0.01, 0.06)
axins.grid(True, alpha=0.3)
axins.set_xticks([0.95, 1, 1.05])
axins.set_xticklabels(['0.95', '1.00', '1.05'])
mark_inset(plt.gca(), axins, loc1=2, loc2=4, fc="none", ec="0.5")

plt.xlabel('z = y·f(x)', fontsize=14)
plt.ylabel('Loss Value', fontsize=14)
plt.title('Non-differentiability of Hinge Loss at z=1', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(-0.05, 0.35)

# Save the non-differentiability plot
plt.savefig(os.path.join(save_dir, "hinge_loss_non_differentiability.png"), dpi=300, bbox_inches='tight')

# Step 6: Sketch the logistic loss function in the range [-3, 3]
print("\nStep 6: Sketch the logistic loss function")
print("--------------------------------------")

plt.figure(figsize=(10, 6))
z = np.linspace(-3, 3, 1000)
logistic_losses = np.log1p(np.exp(-z))

plt.plot(z, logistic_losses, 'g-', linewidth=3, label='Logistic Loss')
plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)

# Mark key points
plt.scatter([0], [logistic_losses[500]], color='red', s=100, zorder=5, label='z=0: L=log(2)≈0.693')
plt.annotate(f'L={logistic_losses[500]:.3f}', xy=(0, logistic_losses[500]), 
             xytext=(0.3, 1.0), fontsize=12,
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))

plt.xlabel('z = y·f(x)', fontsize=14)
plt.ylabel('Logistic Loss Value', fontsize=14)
plt.title('Logistic Loss Function: L(z) = log(1 + e^(-z))', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(-0.2, 3.2)

# Add description of convexity
plt.text(1.5, 2.5, "Logistic loss is convex\nThe second derivative is\nalways positive", 
         fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# Save the logistic loss plot
plt.savefig(os.path.join(save_dir, "logistic_loss.png"), dpi=300, bbox_inches='tight')

# Step 7: Create a 3D visualization to illustrate convexity
print("\nStep 7: 3D visualization of loss functions")
print("---------------------------------------")

# Create a mesh grid for 3D visualization
w1 = np.linspace(-2, 2, 100)
w2 = np.linspace(-2, 2, 100)
W1, W2 = np.meshgrid(w1, w2)

# Example data points for visualization
X = np.array([[1, 1], [2, 0], [0, 2], [-1, 1]])
y = np.array([1, 1, -1, -1])

# Calculate loss values for each (w1, w2) pair for logistic loss
def calc_logistic_loss_surface(W1, W2, X, y):
    loss_surface = np.zeros_like(W1)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            w = np.array([W1[i, j], W2[i, j]])
            loss = 0
            for k in range(X.shape[0]):
                f_x = np.dot(w, X[k])
                loss += logistic_loss(y[k], f_x)
            loss_surface[i, j] = loss / X.shape[0]
    return loss_surface

# Calculate loss values for each (w1, w2) pair for hinge loss
def calc_hinge_loss_surface(W1, W2, X, y):
    loss_surface = np.zeros_like(W1)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            w = np.array([W1[i, j], W2[i, j]])
            loss = 0
            for k in range(X.shape[0]):
                f_x = np.dot(w, X[k])
                loss += hinge_loss(y[k], f_x)
            loss_surface[i, j] = loss / X.shape[0]
    return loss_surface

# Calculate loss surfaces
logistic_loss_surface = calc_logistic_loss_surface(W1, W2, X, y)
hinge_loss_surface = calc_hinge_loss_surface(W1, W2, X, y)

# Create 3D plots
fig = plt.figure(figsize=(15, 7))

# Logistic loss surface
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf1 = ax1.plot_surface(W1, W2, logistic_loss_surface, cmap='viridis', alpha=0.8)
ax1.set_xlabel('w1')
ax1.set_ylabel('w2')
ax1.set_zlabel('Loss')
ax1.set_title('Logistic Loss Surface (Convex)', fontsize=14)

# Hinge loss surface
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf2 = ax2.plot_surface(W1, W2, hinge_loss_surface, cmap='viridis', alpha=0.8)
ax2.set_xlabel('w1')
ax2.set_ylabel('w2')
ax2.set_zlabel('Loss')
ax2.set_title('Hinge Loss Surface (Convex but Non-Differentiable)', fontsize=14)

plt.tight_layout()

# Save the 3D visualization
plt.savefig(os.path.join(save_dir, "3d_loss_surfaces.png"), dpi=300, bbox_inches='tight')

# Step 8: Summarize findings
print("\nStep 8: Summarize findings on convexity")
print("-------------------------------------")
print("Convexity of the different loss functions:")
print("1. 0-1 Loss: Not convex. It has a discontinuity at z=0 and is constant elsewhere.")
print("2. Hinge Loss: Convex but non-differentiable at z=1. Second derivative is 0 where defined.")
print("3. Logistic Loss: Convex and differentiable everywhere. Second derivative is always positive.")
print("4. Squared Error Loss: Convex and differentiable everywhere. Constant positive second derivative.")

print("\nImportance of convexity in optimization:")
print("- Convex functions have a single global minimum (no local minima).")
print("- Gradient descent is guaranteed to find the global minimum for convex functions.")
print("- Non-convex functions (like 0-1 loss) make optimization challenging.")
print("- Non-differentiable points (like in hinge loss) require special optimization techniques.")

print("\nConclusion:")
print("-----------")
print("Of the four loss functions analyzed:")
print("- 0-1 Loss is NOT convex")
print("- Hinge Loss IS convex (but non-differentiable at z=1)")
print("- Logistic Loss IS convex")
print("- Squared Error Loss IS convex")
print("\nConvexity is crucial in machine learning optimization as it guarantees that algorithms")
print("like gradient descent will converge to the global minimum, making training more reliable.") 