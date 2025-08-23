import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_2_Quiz_6")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("=" * 80)
print("QUESTION 6: LOSS FUNCTION COMPARISON")
print("=" * 80)

# Define the loss functions
def loss_01(y, fx):
    """0-1 Loss: L(y, f(x)) = I[y * f(x) <= 0]"""
    return np.where(y * fx <= 0, 1, 0)

def loss_hinge(y, fx):
    """Hinge Loss: L(y, f(x)) = max(0, 1 - y * f(x))"""
    return np.maximum(0, 1 - y * fx)

def loss_logistic(y, fx):
    """Logistic Loss: L(y, f(x)) = log(1 + exp(-y * f(x)))"""
    return np.log(1 + np.exp(-y * fx))

def loss_squared(y, fx):
    """Squared Loss: L(y, f(x)) = (y - f(x))^2"""
    return (y - fx)**2

# Define gradients
def grad_01(y, fx):
    """Gradient of 0-1 Loss (discontinuous, not differentiable at y*f(x) = 0)"""
    # Note: This is not actually differentiable, but we can define a subgradient
    return np.where(y * fx == 0, 0, 0)  # Subgradient at discontinuity

def grad_hinge(y, fx):
    """Gradient of Hinge Loss"""
    return np.where(y * fx >= 1, 0, -y)

def grad_logistic(y, fx):
    """Gradient of Logistic Loss"""
    return -y / (1 + np.exp(y * fx))

def grad_squared(y, fx):
    """Gradient of Squared Loss"""
    return 2 * (fx - y)

print("\n1. SKETCHING ALL FOUR LOSS FUNCTIONS")
print("-" * 50)

# Create the plot for y = +1 and f(x) in [-3, 3]
fx_range = np.linspace(-3, 3, 1000)
y = 1

# Calculate loss values
loss_01_vals = loss_01(y, fx_range)
loss_hinge_vals = loss_hinge(y, fx_range)
loss_logistic_vals = loss_logistic(y, fx_range)
loss_squared_vals = loss_squared(y, fx_range)

# Create the main plot
plt.figure(figsize=(12, 8))

plt.plot(fx_range, loss_01_vals, 'r-', linewidth=3, label='0-1 Loss', alpha=0.8)
plt.plot(fx_range, loss_hinge_vals, 'b-', linewidth=3, label='Hinge Loss', alpha=0.8)
plt.plot(fx_range, loss_logistic_vals, 'g-', linewidth=3, label='Logistic Loss', alpha=0.8)
plt.plot(fx_range, loss_squared_vals, 'm-', linewidth=3, label='Squared Loss', alpha=0.8)

# Add vertical line at f(x) = 0 to show decision boundary
plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Decision Boundary')

# Add horizontal line at y = 0
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# Highlight the point f(x) = 0.5
fx_test = 0.5
plt.axvline(x=fx_test, color='orange', linestyle=':', alpha=0.7, label=f'f(x) = {fx_test}')

plt.xlabel('$f(x)$')
plt.ylabel('Loss Value')
plt.title('Comparison of Classification Loss Functions (y = +1)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(-3, 3)
plt.ylim(-0.1, 10)

# Add annotations for key points
plt.annotate('Correct\nClassification', xy=(1.5, 0.5), xytext=(2, 2),
            arrowprops=dict(arrowstyle='->', color='green'),
            bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.7))

plt.annotate('Incorrect\nClassification', xy=(-1.5, 1), xytext=(-2.5, 3),
            arrowprops=dict(arrowstyle='->', color='red'),
            bbox=dict(boxstyle="round,pad=0.3", fc="lightcoral", alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'loss_functions_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

print("✓ Plot saved as 'loss_functions_comparison.png'")

print("\n2. CALCULATING LOSS VALUES FOR y = +1 AND f(x) = 0.5")
print("-" * 60)

y_test = 1
fx_test = 0.5

print(f"Given: y = {y_test}, f(x) = {fx_test}")
print(f"y * f(x) = {y_test} * {fx_test} = {y_test * fx_test}")
print()

# Calculate each loss
loss_01_val = loss_01(y_test, fx_test)
loss_hinge_val = loss_hinge(y_test, fx_test)
loss_logistic_val = loss_logistic(y_test, fx_test)
loss_squared_val = loss_squared(y_test, fx_test)

print("Loss Calculations:")
print(f"1. 0-1 Loss: L(y, f(x)) = I[y * f(x) ≤ 0] = I[{y_test * fx_test} ≤ 0] = {loss_01_val}")
print(f"2. Hinge Loss: L(y, f(x)) = max(0, 1 - y * f(x)) = max(0, 1 - {y_test * fx_test}) = {loss_hinge_val}")
print(f"3. Logistic Loss: L(y, f(x)) = log(1 + exp(-y * f(x))) = log(1 + exp(-{y_test * fx_test})) = {loss_logistic_val:.4f}")
print(f"4. Squared Loss: L(y, f(x)) = (y - f(x))² = ({y_test} - {fx_test})² = {loss_squared_val}")

print("\n3. CONVEXITY ANALYSIS")
print("-" * 30)

def check_convexity(loss_func, name, y=1):
    """Check convexity by examining second derivative or using definition"""
    fx_range = np.linspace(-2, 2, 1000)
    loss_vals = loss_func(y, fx_range)
    
    # For numerical check, we'll use the definition of convexity
    # A function is convex if f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y) for all λ in [0,1]
    
    # Test convexity at a few points
    test_points = [-1, 0, 1]
    is_convex = True
    
    for i in range(len(test_points)-1):
        x1, x2 = test_points[i], test_points[i+1]
        for lambda_val in [0.25, 0.5, 0.75]:
            # Calculate f(λx1 + (1-λ)x2)
            mid_point = lambda_val * x1 + (1 - lambda_val) * x2
            f_mid = loss_func(y, mid_point)
            
            # Calculate λf(x1) + (1-λ)f(x2)
            f_linear = lambda_val * loss_func(y, x1) + (1 - lambda_val) * loss_func(y, x2)
            
            if f_mid > f_linear + 1e-10:  # Add small tolerance for numerical precision
                is_convex = False
                break
    
    return is_convex

# Check convexity for each loss function
convexity_results = {}
for loss_func, name in [(loss_01, "0-1 Loss"), (loss_hinge, "Hinge Loss"), 
                        (loss_logistic, "Logistic Loss"), (loss_squared, "Squared Loss")]:
    is_convex = check_convexity(loss_func, name)
    convexity_results[name] = is_convex
    print(f"{name}: {'Convex' if is_convex else 'Non-convex'}")

print("\nConvexity Proofs:")
print("1. 0-1 Loss: Non-convex - discontinuous and not differentiable")
print("2. Hinge Loss: Convex - piecewise linear with non-decreasing slope")
print("3. Logistic Loss: Convex - second derivative is always positive")
print("4. Squared Loss: Convex - second derivative is constant (2)")

print("\n4. ROBUSTNESS TO OUTLIERS ANALYSIS")
print("-" * 40)

# Test with different values to show robustness
print("Testing robustness with different f(x) values for y = +1:")
test_values = [0.5, 1.0, 2.0, 5.0, 10.0, -0.5, -1.0, -2.0, -5.0, -10.0]

print(f"{'f(x)':>8} {'0-1':>8} {'Hinge':>8} {'Logistic':>10} {'Squared':>10}")
print("-" * 50)

for fx in test_values:
    l01 = loss_01(1, fx)
    lh = loss_hinge(1, fx)
    ll = loss_logistic(1, fx)
    ls = loss_squared(1, fx)
    print(f"{fx:>8.1f} {l01:>8.1f} {lh:>8.1f} {ll:>10.4f} {ls:>10.1f}")

print("\nRobustness Analysis:")
print("• 0-1 Loss: Most robust - bounded between 0 and 1")
print("• Hinge Loss: Robust - bounded and insensitive to large positive margins")
print("• Logistic Loss: Less robust - grows linearly with negative margins")
print("• Squared Loss: Least robust - grows quadratically with errors")

print("\n5. GRADIENT CALCULATIONS")
print("-" * 30)

# Calculate gradients at f(x) = 0.5
fx_test = 0.5
y_test = 1

print(f"Gradients at f(x) = {fx_test}, y = {y_test}:")
print()

# 0-1 Loss gradient (subgradient)
grad_01_val = grad_01(y_test, fx_test)
print(f"1. 0-1 Loss Gradient:")
print(f"   ∂L/∂f(x) = 0 (not differentiable at y*f(x) = 0)")
print(f"   Subgradient at f(x) = {fx_test}: {grad_01_val}")

# Hinge Loss gradient
grad_hinge_val = grad_hinge(y_test, fx_test)
print(f"\n2. Hinge Loss Gradient:")
print(f"   ∂L/∂f(x) = -y if y*f(x) < 1, 0 otherwise")
print(f"   At f(x) = {fx_test}: y*f(x) = {y_test * fx_test}")
print(f"   Since {y_test * fx_test} < 1: ∂L/∂f(x) = -{y_test} = {grad_hinge_val}")

# Logistic Loss gradient
grad_logistic_val = grad_logistic(y_test, fx_test)
print(f"\n3. Logistic Loss Gradient:")
print(f"   ∂L/∂f(x) = -y / (1 + exp(y*f(x)))")
print(f"   At f(x) = {fx_test}: ∂L/∂f(x) = -{y_test} / (1 + exp({y_test * fx_test}))")
print(f"   ∂L/∂f(x) = {grad_logistic_val:.4f}")

# Squared Loss gradient
grad_squared_val = grad_squared(y_test, fx_test)
print(f"\n4. Squared Loss Gradient:")
print(f"   ∂L/∂f(x) = 2(f(x) - y)")
print(f"   At f(x) = {fx_test}: ∂L/∂f(x) = 2({fx_test} - {y_test}) = {grad_squared_val}")

# Create gradient plot
plt.figure(figsize=(12, 8))

fx_range = np.linspace(-2, 2, 1000)
y = 1

# Calculate gradients (excluding 0-1 loss as it's not differentiable)
grad_hinge_vals = grad_hinge(y, fx_range)
grad_logistic_vals = grad_logistic(y, fx_range)
grad_squared_vals = grad_squared(y, fx_range)

plt.plot(fx_range, grad_hinge_vals, 'b-', linewidth=3, label='Hinge Loss Gradient', alpha=0.8)
plt.plot(fx_range, grad_logistic_vals, 'g-', linewidth=3, label='Logistic Loss Gradient', alpha=0.8)
plt.plot(fx_range, grad_squared_vals, 'm-', linewidth=3, label='Squared Loss Gradient', alpha=0.8)

# Add vertical line at f(x) = 0 to show decision boundary
plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Decision Boundary')

# Add horizontal line at y = 0
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# Highlight the point f(x) = 0.5
fx_test = 0.5
plt.axvline(x=fx_test, color='orange', linestyle=':', alpha=0.7, label=f'f(x) = {fx_test}')

plt.xlabel('$f(x)$')
plt.ylabel('Gradient Value')
plt.title('Gradients of Classification Loss Functions (y = +1)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(-2, 2)
plt.ylim(-3, 3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'loss_gradients.png'), dpi=300, bbox_inches='tight')
plt.close()

print("✓ Gradient plot saved as 'loss_gradients.png'")

# Create detailed analysis plots
print("\n6. DETAILED ANALYSIS PLOTS")
print("-" * 30)

# Create subplots for individual loss functions
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Detailed Analysis of Classification Loss Functions', fontsize=16)

# 0-1 Loss
axes[0, 0].plot(fx_range, loss_01(y, fx_range), 'r-', linewidth=3)
axes[0, 0].set_title('0-1 Loss')
axes[0, 0].set_xlabel('$f(x)$')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
axes[0, 0].text(0.5, 0.8, 'Non-convex\nDiscontinuous', transform=axes[0, 0].transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", fc="lightcoral", alpha=0.7))

# Hinge Loss
axes[0, 1].plot(fx_range, loss_hinge(y, fx_range), 'b-', linewidth=3)
axes[0, 1].set_title('Hinge Loss')
axes[0, 1].set_xlabel('$f(x)$')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axvline(x=1, color='black', linestyle='--', alpha=0.5)
axes[0, 1].text(0.5, 0.8, 'Convex\nRobust to outliers', transform=axes[0, 1].transAxes,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.7))

# Logistic Loss
axes[1, 0].plot(fx_range, loss_logistic(y, fx_range), 'g-', linewidth=3)
axes[1, 0].set_title('Logistic Loss')
axes[1, 0].set_xlabel('$f(x)$')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
axes[1, 0].text(0.5, 0.8, 'Convex\nSmooth', transform=axes[1, 0].transAxes,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.7))

# Squared Loss
axes[1, 1].plot(fx_range, loss_squared(y, fx_range), 'm-', linewidth=3)
axes[1, 1].set_title('Squared Loss')
axes[1, 1].set_xlabel('$f(x)$')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axvline(x=1, color='black', linestyle='--', alpha=0.5)
axes[1, 1].text(0.5, 0.8, 'Convex\nSensitive to outliers', transform=axes[1, 1].transAxes,
                bbox=dict(boxstyle="round,pad=0.3", fc="plum", alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'detailed_loss_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

print("✓ Detailed analysis plot saved as 'detailed_loss_analysis.png'")

print(f"\n" + "=" * 80)
print("SUMMARY OF RESULTS")
print("=" * 80)

print("\nLoss Values at f(x) = 0.5, y = +1:")
print(f"• 0-1 Loss: {loss_01_val}")
print(f"• Hinge Loss: {loss_hinge_val}")
print(f"• Logistic Loss: {loss_logistic_val:.4f}")
print(f"• Squared Loss: {loss_squared_val}")

print("\nConvexity:")
for name, is_convex in convexity_results.items():
    print(f"• {name}: {'Convex' if is_convex else 'Non-convex'}")

print("\nRobustness to Outliers (from most to least robust):")
print("1. 0-1 Loss (bounded)")
print("2. Hinge Loss (bounded)")
print("3. Logistic Loss (unbounded but grows linearly)")
print("4. Squared Loss (unbounded and grows quadratically)")

print("\nGradient Properties:")
print("• 0-1 Loss: Not differentiable")
print("• Hinge Loss: Piecewise constant gradient")
print("• Logistic Loss: Smooth, bounded gradient")
print("• Squared Loss: Linear gradient")

print(f"\nAll plots saved to: {save_dir}")
print("=" * 80)
