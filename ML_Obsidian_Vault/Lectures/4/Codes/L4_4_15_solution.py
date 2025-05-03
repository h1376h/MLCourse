import numpy as np
import matplotlib.pyplot as plt
import os
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_4_Quiz_15")
os.makedirs(save_dir, exist_ok=True)

print("Question 15: Exponential Loss Function Analysis")
print("==============================================")

# Define the loss functions
def exponential_loss(y, fx):
    """
    Exponential loss function: L(y, f(x)) = exp(-y * f(x))
    
    Parameters:
        y: true label (1 or -1)
        fx: model prediction f(x) = w^T x + b
    
    Returns:
        The loss value
    """
    return np.exp(-y * fx)

def hinge_loss(y, fx):
    """
    Hinge loss function: L(y, f(x)) = max(0, 1 - y * f(x))
    
    Parameters:
        y: true label (1 or -1)
        fx: model prediction f(x) = w^T x + b
    
    Returns:
        The loss value
    """
    return np.maximum(0, 1 - y * fx)

def logistic_loss(y, fx):
    """
    Logistic loss function: L(y, f(x)) = log(1 + exp(-y * f(x)))
    
    Parameters:
        y: true label (1 or -1)
        fx: model prediction f(x) = w^T x + b
    
    Returns:
        The loss value
    """
    return np.log(1 + np.exp(-y * fx))

def zero_one_loss(y, fx):
    """
    0-1 loss function: L(y, f(x)) = 0 if y*f(x) > 0, 1 otherwise
    
    Parameters:
        y: true label (1 or -1)
        fx: model prediction f(x) = w^T x + b
    
    Returns:
        The loss value (0 or 1)
    """
    return np.where(y * fx > 0, 0, 1)

# Task 1: Compute the gradient of the exponential loss with respect to w
print("\nTask 1: Computing the Gradient of Exponential Loss")
print("--------------------------------------------------")

# Use symbolic math for clear derivation
w, x, y, b = sp.symbols('w x y b')
fx = w * x + b  # Simplified to 1D for clarity in derivation
L = sp.exp(-y * fx)

# Compute gradient with respect to w
dL_dw = sp.diff(L, w)

print("Given:")
print("  L(y, f(x)) = exp(-y * f(x))")
print("  f(x) = w^T x + b")
print("\nDerivation of gradient with respect to w:")
print(f"  ∂L/∂w = {dL_dw}")
print("  ∂L/∂w = -y * x * exp(-y * f(x))")
print("\nIn vector form:")
print("  ∇_w L(y, f(x)) = -y * x * exp(-y * f(x))")
print("\nExplanation:")
print("  The gradient is the direction of steepest increase in the loss function.")
print("  For minimizing the loss, we move in the opposite direction of the gradient.")
print("  The magnitude of the gradient depends on the classification margin (-y * f(x)).")
print("  For misclassified points, the gradient magnitude can be large due to the exponential.")

# Generate explanatory plots
# Plot the loss functions for comparison
z = np.linspace(-3, 3, 1000)  # z = y * f(x) represents the margin

exp_loss = exponential_loss(1, z)
hinge_loss_vals = hinge_loss(1, z)
logistic_loss_vals = logistic_loss(1, z)
zero_one_loss_vals = zero_one_loss(1, z)

plt.figure(figsize=(10, 6))
plt.plot(z, exp_loss, 'r-', linewidth=2, label='Exponential Loss')
plt.plot(z, hinge_loss_vals, 'b-', linewidth=2, label='Hinge Loss')
plt.plot(z, logistic_loss_vals, 'g-', linewidth=2, label='Logistic Loss')
plt.plot(z, zero_one_loss_vals, 'k--', linewidth=2, label='0-1 Loss')

plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.xlabel('$y \cdot f(x)$ (Margin)', fontsize=14)
plt.ylabel('Loss Value', fontsize=14)
plt.title('Comparison of Loss Functions', fontsize=16)
plt.legend(fontsize=12)
plt.xlim(-3, 3)
plt.ylim(0, 5)

# Add annotations
plt.annotate('Correct classification\n(y·f(x) > 0)', xy=(1.5, 0.5), xytext=(1.5, 1.5),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8), 
             fontsize=12, ha='center')
plt.annotate('Incorrect classification\n(y·f(x) < 0)', xy=(-1, 2), xytext=(-1.5, 3.5),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8), 
             fontsize=12, ha='center')

plt.savefig(os.path.join(save_dir, "loss_functions_comparison.png"), dpi=300, bbox_inches='tight')

# Task 2: Compare exponential loss with hinge loss
print("\nTask 2: Comparing Exponential Loss with Hinge Loss")
print("--------------------------------------------------")
print("Exponential Loss:")
print("  L(y, f(x)) = exp(-y * f(x))")
print("\nHinge Loss:")
print("  L(y, f(x)) = max(0, 1 - y * f(x))")
print("\nKey differences:")
print("1. Behavior for misclassified points (y·f(x) < 0):")
print("   - Exponential loss grows exponentially as margins become more negative")
print("   - Hinge loss grows linearly with the negative margin")
print("2. Behavior for correctly classified points (y·f(x) > 0):")
print("   - Exponential loss continues to decrease but never reaches zero")
print("   - Hinge loss becomes zero when the margin exceeds 1 (y·f(x) > 1)")
print("3. Derivative behavior:")
print("   - Exponential loss: Always has a non-zero gradient")
print("   - Hinge loss: Zero gradient for correctly classified points with margin > 1")
print("4. Sensitivity to outliers:")
print("   - Exponential loss is more sensitive to outliers due to exponential growth")
print("   - Hinge loss is less sensitive as it penalizes linearly")

# Compare the gradients of both losses
z = np.linspace(-3, 3, 1000)
exp_loss_gradient = -1 * np.exp(-1 * z)  # for y=1, simplified gradient magnitude
hinge_loss_gradient = np.where(z < 1, -1, 0)  # for y=1, simplified gradient magnitude

plt.figure(figsize=(10, 6))
plt.plot(z, exp_loss_gradient, 'r-', linewidth=2, label='Exponential Loss Gradient')
plt.plot(z, hinge_loss_gradient, 'b-', linewidth=2, label='Hinge Loss Gradient')

plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
plt.axvline(x=1, color='gray', linestyle='--', alpha=0.3, label='Hinge Loss Margin (y·f(x)=1)')
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.xlabel('$y \cdot f(x)$ (Margin)', fontsize=14)
plt.ylabel('Gradient Magnitude (w.r.t. f(x))', fontsize=14)
plt.title('Comparison of Loss Function Gradients', fontsize=16)
plt.legend(fontsize=12)
plt.xlim(-3, 3)
plt.ylim(-1.1, 0.1)

plt.savefig(os.path.join(save_dir, "loss_gradients_comparison.png"), dpi=300, bbox_inches='tight')

# Task 3: Calculate loss for a correctly classified point with margin 2
print("\nTask 3: Loss Value for Correctly Classified Point")
print("------------------------------------------------")
margin = 2  # y·f(x) = 2
exp_loss_val = exponential_loss(1, margin)
hinge_loss_val = hinge_loss(1, margin)

print(f"For a point with y·f(x) = {margin} (correctly classified with margin 2):")
print(f"  Exponential Loss: exp(-{margin}) = {exp_loss_val:.6f}")
print(f"  Hinge Loss: max(0, 1-{margin}) = {hinge_loss_val}")
print("\nObservation:")
print("  - The exponential loss is very small but still positive")
print("  - The hinge loss is exactly zero since the margin exceeds 1")
print("  - This shows how exponential loss continues to reward higher margins, while")
print("    hinge loss doesn't distinguish between margins once they exceed the threshold")

# Visualize the loss value at margin = 2
plt.figure(figsize=(10, 6))
plt.plot(z, exp_loss, 'r-', linewidth=2, label='Exponential Loss')
plt.plot(z, hinge_loss_vals, 'b-', linewidth=2, label='Hinge Loss')

plt.axvline(x=margin, color='green', linestyle='--', linewidth=2, label=f'y·f(x) = {margin}')
plt.plot(margin, exp_loss_val, 'go', markersize=10)
plt.annotate(f'({margin}, {exp_loss_val:.6f})', xy=(margin, exp_loss_val), xytext=(margin+0.3, exp_loss_val+0.3),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12)

plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.xlabel('$y \cdot f(x)$ (Margin)', fontsize=14)
plt.ylabel('Loss Value', fontsize=14)
plt.title('Loss Value for Margin y·f(x) = 2', fontsize=16)
plt.legend(fontsize=12)
plt.xlim(-3, 3)
plt.ylim(0, 5)

plt.savefig(os.path.join(save_dir, "loss_at_margin_2.png"), dpi=300, bbox_inches='tight')

# Task 4: Determine if the exponential loss function is convex
print("\nTask 4: Convexity of Exponential Loss Function")
print("---------------------------------------------")

# Analyze convexity of exponential loss
print("Mathematical analysis of convexity:")
print("A function is convex if its second derivative is non-negative everywhere.")
print("\nLet's calculate the second derivative of L(z) = exp(-z) with respect to z = y·f(x):")
print("  First derivative: dL/dz = -exp(-z)")
print("  Second derivative: d²L/dz² = exp(-z)")
print("\nSince exp(-z) > 0 for all z, the second derivative is always positive.")
print("Therefore, the exponential loss function is strictly convex.")

# Visualize convexity using a 3D plot for a 2D weight space
w1 = np.linspace(-2, 2, 100)
w2 = np.linspace(-2, 2, 100)
W1, W2 = np.meshgrid(w1, w2)

# Sample data point and label
x = np.array([1, 2])  # Sample feature vector
y_label = 1  # Sample label

# Calculate the model output f(x) = w^T x + b for each weight combination
# For simplicity, we set bias b = 0
F = W1 * x[0] + W2 * x[1]  # Simplified f(x) = w1*x1 + w2*x2

# Calculate exponential loss for each weight combination
Loss = np.exp(-y_label * F)

# Create a 3D plot
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Plot the loss surface
surf = ax.plot_surface(W1, W2, Loss, cmap=cm.coolwarm, alpha=0.8, rstride=5, cstride=5)

# Add a color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Set labels and title
ax.set_xlabel('Weight $w_1$', fontsize=12)
ax.set_ylabel('Weight $w_2$', fontsize=12)
ax.set_zlabel('Exponential Loss', fontsize=12)
ax.set_title('Exponential Loss Function Surface in Weight Space', fontsize=16)

# Save the 3D plot
plt.savefig(os.path.join(save_dir, "exponential_loss_convexity.png"), dpi=300, bbox_inches='tight')

# Show additional evidence of convexity by highlighting some convex combinations
print("\nVisual evidence of convexity:")
print("- The 3D plot shows the loss function surface in weight space")
print("- Notice the bowl-like shape, which is characteristic of convex functions")
print("- For any two points on the surface, the line segment connecting them lies above the surface")
print("- This property confirms the convexity of the exponential loss function")

# Summary
print("\nSummary of Findings")
print("------------------")
print("1. Gradient of exponential loss: -y * x * exp(-y * f(x))")
print("2. Compared to hinge loss, exponential loss:")
print("   - Has exponential rather than linear growth for misclassifications")
print("   - Never becomes exactly zero for correct classifications")
print("   - Is more sensitive to outliers")
print("   - Always has a non-zero gradient")
print("3. For a correctly classified point with margin y·f(x) = 2:")
print(f"   - Exponential loss value = {exp_loss_val:.6f}")
print("   - This is small but positive, showing that the loss continues to reward larger margins")
print("4. The exponential loss function is strictly convex because:")
print("   - Its second derivative exp(-z) is always positive")
print("   - The 3D visualization shows a convex surface in weight space")

print("\nConclusion:")
print("Exponential loss provides a smooth, continuously differentiable alternative to the")
print("hinge loss, with stronger penalties for misclassifications and continued rewards for")
print("larger margins. Its strict convexity ensures that gradient-based optimization will")
print("converge to a unique global minimum, making it suitable for algorithms like gradient descent.") 