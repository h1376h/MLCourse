import numpy as np
import matplotlib.pyplot as plt
import os
import math

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_4_Quiz_2")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

print("Question 2: Loss Functions Comparison")
print("====================================")

# Define the loss functions
def zero_one_loss(y, f_x):
    """
    0-1 Loss: L(y, f(x)) = 0 if y*f(x) > 0, 1 otherwise
    """
    return 0 if y * f_x > 0 else 1

def hinge_loss(y, f_x):
    """
    Hinge Loss: L(y, f(x)) = max(0, 1 - y*f(x))
    """
    return max(0, 1 - y * f_x)

def logistic_loss(y, f_x):
    """
    Logistic Loss: L(y, f(x)) = log(1 + exp(-y*f(x)))
    """
    return np.log(1 + np.exp(-y * f_x))

# Print very detailed step-by-step calculations for Case 1: y=1, f(x)=0.5
print("\nStep 1: Detailed calculations for y=1, f(x)=0.5")
print("--------------------------------------------")

y_1 = 1
f_x_1 = 0.5
y_f_x_1 = y_1 * f_x_1

print(f"Given: y = {y_1}, f(x) = {f_x_1}")
print(f"Calculate y·f(x) = {y_1} × {f_x_1} = {y_f_x_1}")

print("\n0-1 Loss calculation:")
print(f"L(y, f(x)) = 0 if y·f(x) > 0, else 1")
print(f"Since y·f(x) = {y_f_x_1} > 0, the prediction has the correct sign")
print(f"Therefore, L(y, f(x)) = 0")
zero_one_1 = zero_one_loss(y_1, f_x_1)
print(f"0-1 Loss = {zero_one_1}")

print("\nHinge Loss calculation:")
print(f"L(y, f(x)) = max(0, 1 - y·f(x))")
print(f"Step 1: Calculate 1 - y·f(x) = 1 - {y_f_x_1} = {1 - y_f_x_1}")
print(f"Step 2: Take the maximum of 0 and {1 - y_f_x_1}")
print(f"max(0, {1 - y_f_x_1}) = {max(0, 1 - y_f_x_1)}")
hinge_1 = hinge_loss(y_1, f_x_1)
print(f"Hinge Loss = {hinge_1}")

print("\nLogistic Loss calculation:")
print(f"L(y, f(x)) = log(1 + e^(-y·f(x)))")
print(f"Step 1: Calculate -y·f(x) = -({y_f_x_1}) = {-y_f_x_1}")
print(f"Step 2: Calculate e^(-y·f(x)) = e^({-y_f_x_1})")
exp_val = np.exp(-y_f_x_1)
print(f"e^({-y_f_x_1}) = {exp_val:.6f}")
print(f"Step 3: Add 1 to the result: 1 + {exp_val:.6f} = {1 + exp_val:.6f}")
log_arg = 1 + exp_val
print(f"Step 4: Take the natural logarithm: log({log_arg:.6f})")
logistic_1 = logistic_loss(y_1, f_x_1)
print(f"log({log_arg:.6f}) = {logistic_1:.6f}")
print(f"Logistic Loss = {logistic_1:.4f}")

# Print very detailed step-by-step calculations for Case 2: y=-1, f(x)=-2
print("\nStep 2: Detailed calculations for y=-1, f(x)=-2")
print("--------------------------------------------")

y_2 = -1
f_x_2 = -2
y_f_x_2 = y_2 * f_x_2

print(f"Given: y = {y_2}, f(x) = {f_x_2}")
print(f"Calculate y·f(x) = {y_2} × {f_x_2} = {y_f_x_2}")

print("\n0-1 Loss calculation:")
print(f"L(y, f(x)) = 0 if y·f(x) > 0, else 1")
print(f"Since y·f(x) = {y_f_x_2} > 0, the prediction has the correct sign")
print(f"Therefore, L(y, f(x)) = 0")
zero_one_2 = zero_one_loss(y_2, f_x_2)
print(f"0-1 Loss = {zero_one_2}")

print("\nHinge Loss calculation:")
print(f"L(y, f(x)) = max(0, 1 - y·f(x))")
print(f"Step 1: Calculate 1 - y·f(x) = 1 - {y_f_x_2} = {1 - y_f_x_2}")
print(f"Step 2: Take the maximum of 0 and {1 - y_f_x_2}")
print(f"max(0, {1 - y_f_x_2}) = {max(0, 1 - y_f_x_2)}")
hinge_2 = hinge_loss(y_2, f_x_2)
print(f"Hinge Loss = {hinge_2}")

print("\nLogistic Loss calculation:")
print(f"L(y, f(x)) = log(1 + e^(-y·f(x)))")
print(f"Step 1: Calculate -y·f(x) = -({y_f_x_2}) = {-y_f_x_2}")
print(f"Step 2: Calculate e^(-y·f(x)) = e^({-y_f_x_2})")
exp_val = np.exp(-y_f_x_2)
print(f"e^({-y_f_x_2}) = {exp_val:.6f}")
print(f"Step 3: Add 1 to the result: 1 + {exp_val:.6f} = {1 + exp_val:.6f}")
log_arg = 1 + exp_val
print(f"Step 4: Take the natural logarithm: log({log_arg:.6f})")
logistic_2 = logistic_loss(y_2, f_x_2)
print(f"log({log_arg:.6f}) = {logistic_2:.6f}")
print(f"Logistic Loss = {logistic_2:.4f}")

# Step 3: Analyze non-differentiability
print("\nStep 3: Analyze non-differentiability of the loss functions")
print("-------------------------------------------------------")

print("0-1 Loss: Non-differentiable at y·f(x) = 0 (decision boundary)")
print("Hinge Loss: Non-differentiable at y·f(x) = 1 (margin boundary)")
print("Logistic Loss: Differentiable everywhere")

# Create array of f(x) values for plotting
f_x_range = np.linspace(-3, 3, 1000)

# Create a cleaner visualization for y=1
print("\nStep 4: Creating ultra-clean visualizations without text")
print("--------------------------------------------")

# Calculate loss values for y=1 
zero_one_values = [zero_one_loss(1, fx) for fx in f_x_range]
hinge_values = [hinge_loss(1, fx) for fx in f_x_range]
logistic_values = [logistic_loss(1, fx) for fx in f_x_range]

# Ultra-clean visualization for y=1 with NO text in the image
plt.figure(figsize=(10, 6))
plt.plot(f_x_range, zero_one_values, 'r-', linewidth=3, label='0-1 Loss')
plt.plot(f_x_range, hinge_values, 'g-', linewidth=3, label='Hinge Loss')
plt.plot(f_x_range, logistic_values, 'b-', linewidth=3, label='Logistic Loss')

# Mark key locations with subtle vertical lines
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=f_x_1, color='black', linestyle=':', alpha=0.5)

# Mark the data point
plt.scatter([f_x_1], [zero_one_1], color='red', s=100, zorder=5)
plt.scatter([f_x_1], [hinge_1], color='green', s=100, zorder=5)
plt.scatter([f_x_1], [logistic_1], color='blue', s=100, zorder=5)

plt.xlabel('f(x)', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Loss Functions for y=1', fontsize=16)
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, alpha=0.3)
plt.ylim(-0.1, 2.0)

# Save the plot with no text annotations
plt.savefig(os.path.join(save_dir, "loss_functions_y1_simple.png"), dpi=300, bbox_inches='tight')

# Ultra-clean visualization for y=-1 with NO text in the image
zero_one_values_neg = [zero_one_loss(-1, fx) for fx in f_x_range]
hinge_values_neg = [hinge_loss(-1, fx) for fx in f_x_range]
logistic_values_neg = [logistic_loss(-1, fx) for fx in f_x_range]

plt.figure(figsize=(10, 6))
plt.plot(f_x_range, zero_one_values_neg, 'r-', linewidth=3, label='0-1 Loss')
plt.plot(f_x_range, hinge_values_neg, 'g-', linewidth=3, label='Hinge Loss')
plt.plot(f_x_range, logistic_values_neg, 'b-', linewidth=3, label='Logistic Loss')

# Mark key locations with subtle vertical lines
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=-1, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=f_x_2, color='black', linestyle=':', alpha=0.5)

# Mark the data point
plt.scatter([f_x_2], [zero_one_2], color='red', s=100, zorder=5)
plt.scatter([f_x_2], [hinge_2], color='green', s=100, zorder=5)
plt.scatter([f_x_2], [logistic_2], color='blue', s=100, zorder=5)

plt.xlabel('f(x)', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Loss Functions for y=-1', fontsize=16)
plt.legend(fontsize=12, loc='upper left')
plt.grid(True, alpha=0.3)
plt.ylim(-0.1, 2.0)

# Save the plot with no text annotations
plt.savefig(os.path.join(save_dir, "loss_functions_y-1_simple.png"), dpi=300, bbox_inches='tight')

# Create ultra-simple visualizations for each loss function
print("\nStep 5: Creating ultra-simple visualizations for each loss function")
print("----------------------------------------------------------")

# 0-1 Loss - ultra simple
plt.figure(figsize=(7, 4))
f_x_zoom = np.linspace(-2, 2, 1000)
zero_one_zoom = [zero_one_loss(1, fx) for fx in f_x_zoom]

plt.plot(f_x_zoom, zero_one_zoom, 'r-', linewidth=4)
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
plt.title('0-1 Loss', fontsize=14)
plt.xlabel('f(x)', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(-0.1, 1.2)

plt.savefig(os.path.join(save_dir, "zero_one_simple.png"), dpi=300, bbox_inches='tight')

# Hinge Loss - ultra simple
plt.figure(figsize=(7, 4))
f_x_zoom = np.linspace(-1, 3, 1000)
hinge_zoom = [hinge_loss(1, fx) for fx in f_x_zoom]

plt.plot(f_x_zoom, hinge_zoom, 'g-', linewidth=4)
plt.axvline(x=1, color='gray', linestyle='--', alpha=0.7)
plt.scatter([1], [0], color='black', s=100, marker='x', zorder=5)
plt.title('Hinge Loss', fontsize=14)
plt.xlabel('f(x)', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(-0.1, 2.1)

plt.savefig(os.path.join(save_dir, "hinge_simple.png"), dpi=300, bbox_inches='tight')

# Logistic Loss - ultra simple
plt.figure(figsize=(7, 4))
f_x_zoom = np.linspace(-2, 4, 1000)
logistic_zoom = [logistic_loss(1, fx) for fx in f_x_zoom]

plt.plot(f_x_zoom, logistic_zoom, 'b-', linewidth=4)
plt.title('Logistic Loss', fontsize=14)
plt.xlabel('f(x)', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(-0.1, 2.1)

plt.savefig(os.path.join(save_dir, "logistic_simple.png"), dpi=300, bbox_inches='tight')

# Very simple visualization for the specific data points
print("\nStep 6: Creating ultra-simple visualizations for our data points")
print("-----------------------------------------------------")

# Data point 1: (y=1, f(x)=0.5) - Ultra simple visualization
plt.figure(figsize=(7, 4))
plt.scatter(f_x_1, zero_one_1, color='red', s=120, zorder=5, label='0-1 Loss')
plt.scatter(f_x_1, hinge_1, color='green', s=120, zorder=5, label='Hinge Loss')
plt.scatter(f_x_1, logistic_1, color='blue', s=120, zorder=5, label='Logistic Loss')
plt.vlines(x=f_x_1, ymin=0, ymax=hinge_1, color='black', linestyle=':', linewidth=2)
plt.axvline(x=f_x_1, color='black', linestyle=':', alpha=0.7)
plt.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

plt.title('Loss Values at f(x)=0.5, y=1', fontsize=14)
plt.xlabel('f(x)', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(0, 1.5)
plt.ylim(-0.1, 1.0)

plt.savefig(os.path.join(save_dir, "point_1_simple.png"), dpi=300, bbox_inches='tight')

# Data point 2: (y=-1, f(x)=-2) - Ultra simple visualization
plt.figure(figsize=(7, 4))
plt.scatter(f_x_2, zero_one_2, color='red', s=120, zorder=5, label='0-1 Loss')
plt.scatter(f_x_2, hinge_2, color='green', s=120, zorder=5, label='Hinge Loss')
plt.scatter(f_x_2, logistic_2, color='blue', s=120, zorder=5, label='Logistic Loss')
plt.vlines(x=f_x_2, ymin=0, ymax=logistic_2, color='black', linestyle=':', linewidth=2)
plt.axvline(x=f_x_2, color='black', linestyle=':', alpha=0.7)
plt.axvline(x=-1, color='gray', linestyle='--', alpha=0.5)
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

plt.title('Loss Values at f(x)=-2, y=-1', fontsize=14)
plt.xlabel('f(x)', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(-2.5, -0.5)
plt.ylim(-0.1, 0.5)

plt.savefig(os.path.join(save_dir, "point_2_simple.png"), dpi=300, bbox_inches='tight')

# Create a very simple geometric interpretation visualization
print("\nStep 7: Creating a simplified geometric interpretation")
print("-----------------------------------------------------")

# Simple geometric interpretation - minimal annotations
plt.figure(figsize=(10, 4))

# Draw the 1D feature space
plt.axhline(y=0, color='k', linestyle='-', linewidth=2)
plt.arrow(-3, 0, 6.5, 0, head_width=0.1, head_length=0.2, fc='k', ec='k', linewidth=2)

# Mark key boundaries
plt.scatter([0], [0], color='purple', s=150, marker='o', zorder=5)  # Decision boundary
plt.scatter([-1], [0], color='green', s=100, marker='o', zorder=5)  # Negative margin
plt.scatter([1], [0], color='green', s=100, marker='o', zorder=5)   # Positive margin

# Mark data points
plt.scatter([f_x_1], [0], color='blue', s=120, marker='o', zorder=5)
plt.scatter([f_x_2], [0], color='blue', s=120, marker='o', zorder=5)

# Add minimal labels
plt.text(0, 0.15, 'Decision\nBoundary', ha='center', fontsize=10)
plt.text(-1, 0.15, 'Negative\nMargin', ha='center', fontsize=10)
plt.text(1, 0.15, 'Positive\nMargin', ha='center', fontsize=10)
plt.text(f_x_1, -0.15, 'f(x)=0.5', ha='center', fontsize=10)
plt.text(f_x_2, -0.15, 'f(x)=-2', ha='center', fontsize=10)
plt.text(3.5, 0, 'f(x)', fontsize=12, ha='center', va='center')

plt.ylim(-0.3, 0.3)
plt.xlim(-3.2, 3.7)
plt.title('Geometric Interpretation', fontsize=14)
plt.gca().set_yticks([])
plt.gca().set_xticks([-2, -1, 0, 0.5, 1, 2])
plt.grid(False)

plt.savefig(os.path.join(save_dir, "geometric_simple.png"), dpi=300, bbox_inches='tight')

# Create a very simple visualization showing derivatives
print("\nStep 8: Creating visualizations of derivatives/non-differentiability")
print("-----------------------------------------------------")

# 0-1 Loss with derivative illustration
plt.figure(figsize=(7, 4))
f_x_zoom = np.linspace(-2, 2, 1000)
zero_one_zoom = [zero_one_loss(1, fx) for fx in f_x_zoom]

plt.plot(f_x_zoom, zero_one_zoom, 'r-', linewidth=3)
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
plt.scatter([0], [1], color='black', s=100, marker='x', zorder=5)

# Add arrows to show left and right derivatives
plt.arrow(-0.5, 1, 0.4, 0, head_width=0.1, head_length=0.05, fc='black', ec='black')
plt.arrow(0.5, 0, -0.4, 0, head_width=0.1, head_length=0.05, fc='black', ec='black')

plt.title('0-1 Loss: Non-differentiable at y·f(x)=0', fontsize=14)
plt.xlabel('f(x)', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(-0.1, 1.5)

plt.savefig(os.path.join(save_dir, "zero_one_derivative.png"), dpi=300, bbox_inches='tight')

# Hinge Loss with derivative illustration
plt.figure(figsize=(7, 4))
f_x_zoom = np.linspace(-1, 3, 1000)
hinge_zoom = [hinge_loss(1, fx) for fx in f_x_zoom]

plt.plot(f_x_zoom, hinge_zoom, 'g-', linewidth=3)
plt.axvline(x=1, color='gray', linestyle='--', alpha=0.7)
plt.scatter([1], [0], color='black', s=100, marker='x', zorder=5)

# Add arrows to show left and right derivatives
plt.arrow(0.5, 0.5, 0.4, -0.4, head_width=0.1, head_length=0.05, fc='black', ec='black')
plt.arrow(1.5, 0, -0.4, 0, head_width=0.1, head_length=0.05, fc='black', ec='black')

plt.title('Hinge Loss: Non-differentiable at y·f(x)=1', fontsize=14)
plt.xlabel('f(x)', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(-0.1, 2.1)

plt.savefig(os.path.join(save_dir, "hinge_derivative.png"), dpi=300, bbox_inches='tight')

# Logistic Loss with derivative illustration
plt.figure(figsize=(7, 4))
f_x_zoom = np.linspace(-2, 4, 1000)
logistic_zoom = [logistic_loss(1, fx) for fx in f_x_zoom]

# Calculate derivatives at various points for visualization
derivative_points = np.array([-1, 0, 1, 2])
derivatives = [-np.exp(-dp)/(1 + np.exp(-dp)) for dp in derivative_points]
logistic_values_at_points = [logistic_loss(1, dp) for dp in derivative_points]

# Plot the function and derivatives
plt.plot(f_x_zoom, logistic_zoom, 'b-', linewidth=3)
plt.scatter(derivative_points, logistic_values_at_points, color='black', s=60, zorder=5)

# Add arrows to show derivatives (tangent lines)
for i, (dp, dv, lv) in enumerate(zip(derivative_points, derivatives, logistic_values_at_points)):
    # Add arrow to show derivative direction
    plt.arrow(dp, lv, 0.5, 0.5*dv, head_width=0.08, head_length=0.05, fc='black', ec='black')

plt.title('Logistic Loss: Differentiable Everywhere', fontsize=14)
plt.xlabel('f(x)', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(-0.1, 2.1)

plt.savefig(os.path.join(save_dir, "logistic_derivative.png"), dpi=300, bbox_inches='tight')

# Create a comparative visualization for all loss functions
print("\nStep 9: Creating a comparative visualization of all loss functions")
print("-----------------------------------------------------")

# Plot all losses on the same axes for y=1
plt.figure(figsize=(9, 6))
plt.plot(f_x_range, zero_one_values, 'r-', linewidth=3, label='0-1 Loss')
plt.plot(f_x_range, hinge_values, 'g-', linewidth=3, label='Hinge Loss')
plt.plot(f_x_range, logistic_values, 'b-', linewidth=3, label='Logistic Loss')

# Mark key points
plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
plt.axvline(x=1, color='black', linestyle='--', alpha=0.5)
plt.scatter([0], [1], color='red', s=100, marker='x', zorder=5)
plt.scatter([1], [0], color='green', s=100, marker='x', zorder=5)

plt.title('Comparison of Loss Functions (y=1)', fontsize=16)
plt.xlabel('f(x)', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(-0.1, 2.0)
plt.xlim(-3, 3)

plt.savefig(os.path.join(save_dir, "loss_comparison.png"), dpi=300, bbox_inches='tight')

# Summary
print("\nStep 10: Summary of loss values")
print("--------------------------")

print("For y=1, f(x)=0.5:")
print(f"0-1 Loss: {zero_one_1}")
print(f"Hinge Loss: {hinge_1}")
print(f"Logistic Loss: {logistic_1:.4f}")

print("\nFor y=-1, f(x)=-2:")
print(f"0-1 Loss: {zero_one_2}")
print(f"Hinge Loss: {hinge_2}")
print(f"Logistic Loss: {logistic_2:.4f}")

print("\nNon-differentiability points:")
print("- 0-1 loss: Non-differentiable at y·f(x) = 0 (decision boundary)")
print("- Hinge loss: Non-differentiable at y·f(x) = 1 (margin boundary)")
print("- Logistic loss: Differentiable everywhere") 