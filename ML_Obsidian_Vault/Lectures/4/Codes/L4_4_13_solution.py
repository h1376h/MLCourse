import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_4_Quiz_13")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

print("Question 13: Decision Boundary Geometry")
print("====================================")

# Given information
w1 = 2
w2 = -3
b = 1

# Create a weight vector and bias term
w = np.array([w1, w2])
print(f"Weight vector w = [{w1}, {w2}]")
print(f"Bias term b = {b}")

# Task 1: Calculate the distance from point (2, 3) to the decision boundary
print("\nTask 1: Distance from point (2, 3) to the decision boundary")
print("------------------------------------------------------")

# The point (2, 3)
x_point = np.array([2, 3])

# The formula for the distance from a point to a line ax + by + c = 0 is:
# d = |ax0 + by0 + c| / sqrt(a^2 + b^2)
# In our case, a = w1, b = w2, c = b, (x0, y0) = (2, 3)
numerator = abs(w1 * x_point[0] + w2 * x_point[1] + b)
denominator = np.sqrt(w1**2 + w2**2)
distance = numerator / denominator

print(f"Point: ({x_point[0]}, {x_point[1]})")
print(f"Decision boundary equation: {w1}x₁ + {w2}x₂ + {b} = 0")
print(f"Distance calculation: |{w1}×{x_point[0]} + {w2}×{x_point[1]} + {b}| / √({w1}² + {w2}²)")
print(f"Distance = |{w1 * x_point[0] + w2 * x_point[1] + b}| / √{w1**2 + w2**2}")
print(f"Distance = {numerator} / {denominator:.4f}")
print(f"Distance = {distance:.4f}")

# Task 2: Determine the class prediction for a new data point (0, 1)
print("\nTask 2: Class prediction for point (0, 1)")
print("--------------------------------------")

# New data point (0, 1)
x_new = np.array([0, 1])

# Calculate the decision function value
decision_value = w1 * x_new[0] + w2 * x_new[1] + b
# Determine the class (positive or negative)
prediction = 1 if decision_value > 0 else -1

print(f"New point: ({x_new[0]}, {x_new[1]})")
print(f"Decision function: f(x) = {w1}x₁ + {w2}x₂ + {b}")
print(f"f({x_new[0]}, {x_new[1]}) = {w1}×{x_new[0]} + {w2}×{x_new[1]} + {b} = {decision_value}")
print(f"Since f(x) = {decision_value} < 0, the predicted class is: {prediction}")

# Task 3: Normalize the weight vector to unit length
print("\nTask 3: Normalize the weight vector to unit length")
print("----------------------------------------------")

# Calculate the norm of the weight vector
w_norm = np.linalg.norm(w)
# Normalize the weight vector
w_normalized = w / w_norm
# Normalize the bias term
b_normalized = b / w_norm

print(f"Original weight vector: w = [{w1}, {w2}]")
print(f"Original bias term: b = {b}")
print(f"Norm of the weight vector: ||w|| = √({w1}² + {w2}²) = {w_norm:.4f}")
print(f"Normalized weight vector: w_normalized = w / ||w|| = [{w_normalized[0]:.4f}, {w_normalized[1]:.4f}]")
print(f"Normalized bias term: b_normalized = b / ||w|| = {b_normalized:.4f}")
print(f"New decision boundary equation: {w_normalized[0]:.4f}x₁ + {w_normalized[1]:.4f}x₂ + {b_normalized:.4f} = 0")

# Verify that the decision boundary is the same
print("\nVerification that both equations represent the same decision boundary:")

# For a point on the decision boundary, both equations should give the same result (equal to zero)
x_verify = np.array([3, 2])  # Arbitrarily chosen point
original_result = w1 * x_verify[0] + w2 * x_verify[1] + b
normalized_result = w_normalized[0] * x_verify[0] + w_normalized[1] * x_verify[1] + b_normalized

print(f"For point ({x_verify[0]}, {x_verify[1]})")
print(f"Original equation: {w1}×{x_verify[0]} + {w2}×{x_verify[1]} + {b} = {original_result}")
print(f"Normalized equation: {w_normalized[0]:.4f}×{x_verify[0]} + {w_normalized[1]:.4f}×{x_verify[1]} + {b_normalized:.4f} = {normalized_result:.4f}")
print(f"Ratio of results: {original_result / normalized_result:.4f}")

# Task 4: Sketch the decision boundary and indicate positive and negative regions
print("\nTask 4: Sketch the decision boundary")
print("----------------------------------")

# Create data for visualization
x1_range = np.linspace(-3, 6, 100)
x2_range = np.linspace(-3, 6, 100)
X1, X2 = np.meshgrid(x1_range, x2_range)
Z = w1 * X1 + w2 * X2 + b

# Create a figure
plt.figure(figsize=(10, 8))

# Plot the decision boundary and regions
plt.contourf(X1, X2, Z, levels=[-10, 0, 10], colors=['#FFAAAA', '#AAAAFF'], alpha=0.5)
plt.contour(X1, X2, Z, levels=[0], colors='k', linewidths=2)

# Calculate some points along the decision boundary for better understanding
# 2x - 3y + 1 = 0 => y = (2x + 1) / 3
x1_boundary = np.array([-1, 5])
x2_boundary = (w1 * x1_boundary + b) / (-w2)

# Plot the original points with class labels
plt.scatter(2, 3, color='blue', s=100, marker='o', label='Task 1: (2, 3)')
plt.scatter(0, 1, color='red', s=100, marker='x', label='Task 2: (0, 1)')

# Plot the weight vector (perpendicular to the decision boundary)
plt.arrow(0, 0, w[0], w[1], head_width=0.2, head_length=0.3, fc='green', ec='green', label='Weight Vector w')
# Plot the normalized weight vector
plt.arrow(0, 0, w_normalized[0], w_normalized[1], head_width=0.1, head_length=0.15, 
          fc='purple', ec='purple', label='Normalized w', alpha=0.7)

# Annotate the distance
# Calculate a point on the decision boundary closest to (2, 3)
# This is the projection of (2, 3) onto the decision boundary
# First, find a vector perpendicular to the weight vector
perpendicular = w / np.linalg.norm(w)
# Calculate how far along this perpendicular to go
t = -(w1 * x_point[0] + w2 * x_point[1] + b) / (w1**2 + w2**2)
# Find the closest point on the boundary
closest_x = x_point[0] + t * w1
closest_y = x_point[1] + t * w2

# Plot the distance line
plt.plot([x_point[0], closest_x], [x_point[1], closest_y], 'k--', linewidth=1.5)
# Annotate the distance value
plt.annotate(f'Distance = {distance:.4f}', 
             xy=((x_point[0] + closest_x)/2, (x_point[1] + closest_y)/2), 
             xytext=(3, 3.5), 
             arrowprops=dict(arrowstyle='->'))

# Add region labels
plt.annotate('Positive Region (y = +1)', xy=(4, 0.5), fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", fc='#AAAAFF', ec="b", alpha=0.3))
plt.annotate('Negative Region (y = -1)', xy=(0, 4), fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", fc='#FFAAAA', ec="r", alpha=0.3))

# Add decision boundary equation
plt.annotate(f'Decision Boundary: {w1}x₁ + {w2}x₂ + {b} = 0',
             xy=(0.05, 0.95), xycoords='axes fraction',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

# Customize the plot
plt.xlim(-3, 6)
plt.ylim(-3, 6)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel('Feature $x_1$', fontsize=12)
plt.ylabel('Feature $x_2$', fontsize=12)
plt.title('Decision Boundary Geometry', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')

# Save the figure
plt.savefig(os.path.join(save_dir, "decision_boundary.png"), dpi=300, bbox_inches='tight')

# Create a 3D visualization to better understand the concept
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Create a 3D surface of the decision function
Z_3d = w1 * X1 + w2 * X2 + b
ax.plot_surface(X1, X2, Z_3d, alpha=0.5, cmap='viridis')

# Plot the decision boundary (where Z = 0)
ax.contour(X1, X2, Z_3d, levels=[0], colors='k', linewidths=2)

# Mark the points on the 3D surface
ax.scatter(2, 3, w1*2 + w2*3 + b, color='blue', s=100, marker='o', label='Task 1: (2, 3)')
ax.scatter(0, 1, w1*0 + w2*1 + b, color='red', s=100, marker='x', label='Task 2: (0, 1)')

# Plot the plane Z = 0 (horizontal plane)
xx, yy = np.meshgrid(x1_range, x2_range)
ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.2, color='gray')

# Customize the 3D plot
ax.set_xlabel('Feature $x_1$', fontsize=12)
ax.set_ylabel('Feature $x_2$', fontsize=12)
ax.set_zlabel('Decision Function Value', fontsize=12)
ax.set_title('3D Visualization of Decision Function', fontsize=14)
ax.legend()

# Save the 3D figure
plt.savefig(os.path.join(save_dir, "decision_function_3d.png"), dpi=300, bbox_inches='tight')

print("\nImages saved:")
print(f"1. Decision boundary: {os.path.join(save_dir, 'decision_boundary.png')}")
print(f"2. 3D decision function: {os.path.join(save_dir, 'decision_function_3d.png')}")

print("\nSummary of Findings:")
print("------------------")
print(f"1. The distance from point (2, 3) to the decision boundary is {distance:.4f}")
print(f"2. For the data point (0, 1), the model predicts class {prediction}")
print(f"3. When normalized, the decision boundary equation becomes: {w_normalized[0]:.4f}x₁ + {w_normalized[1]:.4f}x₂ + {b_normalized:.4f} = 0")
print("4. The decision boundary has been sketched with positive region (y = +1) and negative region (y = -1) clearly indicated") 