import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
import matplotlib.font_manager as fm  # For font management

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_4_Quiz_13")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
# Use a different font that supports subscripts, or use regular notation
plt.rcParams['font.family'] = 'DejaVu Sans'

print("Question 13: Decision Boundary Geometry")
print("====================================")

# Define the decision boundary parameters
w1 = 2
w2 = -3
b = 1

# Create a weight vector and bias term
w = np.array([w1, w2])
print(f"Weight vector w = [{w1}, {w2}]")
print(f"Bias term b = {b}")
print(f"Decision boundary equation: {w1}x_1 + {w2}x_2 + {b} = 0")

# Task 1: Calculate the distance from point (2, 3) to the decision boundary
print("\nTask 1: Distance from point (2, 3) to the decision boundary")
print("------------------------------------------------------")

# The point (2, 3)
x_point = np.array([2, 3])
print(f"Point: ({x_point[0]}, {x_point[1]})")

# Step 1: Calculate the value of the decision function at the point
print("Step 1: Calculate the decision function value at the point")
decision_value = w1 * x_point[0] + w2 * x_point[1] + b
print(f"f(x) = w·x + b = {w1}×{x_point[0]} + {w2}×{x_point[1]} + {b}")
print(f"f({x_point[0]}, {x_point[1]}) = {w1 * x_point[0]} + {w2 * x_point[1]} + {b} = {decision_value}")

# Step 2: Calculate the norm of the weight vector
print("\nStep 2: Calculate the norm of the weight vector")
w_norm_squared = w1**2 + w2**2
w_norm = np.sqrt(w_norm_squared)
print(f"||w||² = w_1² + w_2² = {w1}² + {w2}² = {w1**2} + {w2**2} = {w_norm_squared}")
print(f"||w|| = √({w_norm_squared}) = {w_norm:.4f}")

# Step 3: Calculate the distance using the formula
print("\nStep 3: Calculate the distance using the formula")
print("The formula for the distance from a point to a hyperplane is:")
print("d = |w·x + b| / ||w||")
numerator = abs(decision_value)
denominator = w_norm
distance = numerator / denominator
print(f"d = |f(x)| / ||w|| = |{decision_value}| / {w_norm:.4f} = {numerator} / {denominator:.4f} = {distance:.4f}")

# Step 4: Verify the result by finding the closest point on the boundary
print("\nStep 4: Verify the result by finding the closest point on the boundary")
print("The closest point on the boundary is found by projecting from the given point in the direction of w:")
# The formula: x_closest = x - (w·x + b)w / ||w||²
projection_factor = decision_value / w_norm_squared
print(f"Projection factor = (w·x + b) / ||w||² = {decision_value} / {w_norm_squared} = {projection_factor:.4f}")
closest_point = x_point - projection_factor * w
print(f"Closest point = point - factor × w = [{x_point[0]}, {x_point[1]}] - {projection_factor:.4f} × [{w1}, {w2}]")
print(f"Closest point = [{x_point[0] - projection_factor * w1:.4f}, {x_point[1] - projection_factor * w2:.4f}]")

# Verify the closest point is on the boundary
closest_point_decision_value = w1 * closest_point[0] + w2 * closest_point[1] + b
print(f"Decision function value at closest point: {w1}×{closest_point[0]:.4f} + {w2}×{closest_point[1]:.4f} + {b} = {closest_point_decision_value:.10f}")
print("The value is very close to zero, confirming the point is on the decision boundary.")

# Calculate Euclidean distance between the original point and the closest point
euclidean_distance = np.sqrt((x_point[0] - closest_point[0])**2 + (x_point[1] - closest_point[1])**2)
print(f"Euclidean distance between original point and closest point = {euclidean_distance:.4f}")
print(f"This matches our formula-based calculation: {distance:.4f}")

print("\nTherefore, the distance from the point ({x_point[0]}, {x_point[1]}) to the decision boundary {w1}x_1 + {w2}x_2 + {b} = 0 is {distance:.4f} units.")

# Task 2: Determine the class prediction for a new data point (0, 1)
print("\nTask 2: Class prediction for point (0, 1)")
print("--------------------------------------")

# New data point (0, 1)
x_new = np.array([0, 1])
print(f"New point: ({x_new[0]}, {x_new[1]})")

# Step 1: Calculate the decision function value
print("Step 1: Calculate the decision function value")
decision_value_new = w1 * x_new[0] + w2 * x_new[1] + b
print(f"f(x) = w·x + b = {w1}×{x_new[0]} + {w2}×{x_new[1]} + {b}")
print(f"f({x_new[0]}, {x_new[1]}) = {w1 * x_new[0]} + {w2 * x_new[1]} + {b} = {decision_value_new}")

# Step 2: Determine the class based on the sign of the decision function
print("\nStep 2: Determine the class based on the sign of the decision function")
print("Classification rule:")
print("- If f(x) > 0, predict class +1")
print("- If f(x) < 0, predict class -1")
print("- If f(x) = 0, the point is on the decision boundary")

prediction = 1 if decision_value_new > 0 else -1
print(f"Since f({x_new[0]}, {x_new[1]}) = {decision_value_new} {'>' if decision_value_new > 0 else '<'} 0")
print(f"The predicted class is: {prediction}")

# Step 3: Calculate the distance to the decision boundary
print("\nStep 3: Calculate the point's distance to the decision boundary")
distance_new = abs(decision_value_new) / w_norm
print(f"Distance = |f(x)| / ||w|| = |{decision_value_new}| / {w_norm:.4f} = {abs(decision_value_new)} / {w_norm:.4f} = {distance_new:.4f}")
print(f"The distance from point ({x_new[0]}, {x_new[1]}) to the decision boundary is {distance_new:.4f} units")
print(f"This means the point is {distance_new:.4f} units into the {'positive' if prediction > 0 else 'negative'} region")

# Task 3: Normalize the weight vector to unit length
print("\nTask 3: Normalize the weight vector to unit length")
print("----------------------------------------------")

# Step 1: Calculate the norm of the weight vector (already done)
print("Step 1: Calculate the norm of the weight vector")
print(f"||w|| = √(w_1² + w_2²) = √({w1}² + {w2}²) = √({w_norm_squared}) = {w_norm:.4f}")

# Step 2: Normalize the weight vector
print("\nStep 2: Normalize the weight vector by dividing by its norm")
w_normalized = w / w_norm
print(f"w_normalized = w / ||w|| = [{w1}, {w2}] / {w_norm:.4f}")
print(f"w_normalized = [{w1 / w_norm:.4f}, {w2 / w_norm:.4f}]")

# Step 3: Normalize the bias term
print("\nStep 3: Normalize the bias term")
b_normalized = b / w_norm
print(f"b_normalized = b / ||w|| = {b} / {w_norm:.4f} = {b_normalized:.4f}")

# Step 4: Write the new decision boundary equation
print("\nStep 4: Write the new normalized decision boundary equation")
print(f"Original equation: {w1}x_1 + {w2}x_2 + {b} = 0")
print(f"Normalized equation: {w_normalized[0]:.4f}x_1 + {w_normalized[1]:.4f}x_2 + {b_normalized:.4f} = 0")

# Step 5: Verify that both equations represent the same boundary
print("\nStep 5: Verify that both equations represent the same decision boundary")
print("A test point should yield proportional results with both equations")

# Choose a test point that's close to the boundary
x_verify = np.array([3, 2])
original_result = w1 * x_verify[0] + w2 * x_verify[1] + b
normalized_result = w_normalized[0] * x_verify[0] + w_normalized[1] * x_verify[1] + b_normalized

print(f"Test point: ({x_verify[0]}, {x_verify[1]})")
print("Original equation:")
print(f"{w1}×{x_verify[0]} + {w2}×{x_verify[1]} + {b} = {w1 * x_verify[0]} + ({w2 * x_verify[1]}) + {b} = {original_result}")
print("Normalized equation:")
print(f"{w_normalized[0]:.4f}×{x_verify[0]} + {w_normalized[1]:.4f}×{x_verify[1]} + {b_normalized:.4f} = {normalized_result:.4f}")

ratio = original_result / normalized_result
print(f"Ratio of results: {original_result} / {normalized_result:.4f} = {ratio:.4f}")
print(f"This is very close to the weight norm {w_norm:.4f}, confirming both equations represent the same boundary")

# Task 4: Sketch the decision boundary and indicate positive and negative regions
print("\nTask 4: Sketch the decision boundary")
print("----------------------------------")

# Step 1: Rearrange the equation to express x_2 in terms of x_1
print("Step 1: Rearrange the decision boundary equation to express x_2 in terms of x_1")
print(f"Original equation: {w1}x_1 + {w2}x_2 + {b} = 0")
print(f"Rearranging: {w2}x_2 = -{w1}x_1 - {b}")
print(f"Therefore: x_2 = ({-w1}x_1 - {b}) / {w2}")
slope = -w1 / w2
intercept = -b / w2
print(f"This is a line with slope {slope:.4f} and y-intercept {intercept:.4f}")
print(f"Equation: x_2 = {slope:.4f}x_1 + {intercept:.4f}")

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

# Annotate the distance from (2, 3) to the boundary
plt.plot([x_point[0], closest_point[0]], [x_point[1], closest_point[1]], 'k--', linewidth=1.5)
plt.annotate(f'Distance = {distance:.4f}', 
             xy=((x_point[0] + closest_point[0])/2, (x_point[1] + closest_point[1])/2), 
             xytext=(3, 3.5), 
             arrowprops=dict(arrowstyle='->'))

# Add region labels
plt.annotate('Positive Region (y = +1)', xy=(4, 0.5), fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", fc='#AAAAFF', ec="b", alpha=0.3))
plt.annotate('Negative Region (y = -1)', xy=(0, 4), fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", fc='#FFAAAA', ec="r", alpha=0.3))

# Add decision boundary equation
plt.annotate(f'Decision Boundary: {w1}x_1 + {w2}x_2 + {b} = 0',
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

# NEW VISUALIZATION: Showing multiple points with distances to decision boundary
plt.figure(figsize=(12, 10))

# Plot the decision boundary and regions
plt.contourf(X1, X2, Z, levels=[-10, 0, 10], colors=['#FFAAAA', '#AAAAFF'], alpha=0.5)
cs = plt.contour(X1, X2, np.abs(Z)/w_norm, levels=np.arange(0, 3, 0.5), 
                  colors='gray', linestyles='--', alpha=0.7)
plt.clabel(cs, inline=1, fontsize=10, fmt='%.1f')
plt.contour(X1, X2, Z, levels=[0], colors='k', linewidths=2)

# Generate random points
np.random.seed(42)  # For reproducibility
num_points = 10
points_x1 = np.random.uniform(-2, 5, num_points)
points_x2 = np.random.uniform(-2, 5, num_points)
points = np.column_stack((points_x1, points_x2))

# Calculate distances and classify points
distances = []
classes = []
closest_points = []

for i, point in enumerate(points):
    # Calculate distance to decision boundary
    decision_val = w1 * point[0] + w2 * point[1] + b
    distance_i = abs(decision_val) / w_norm
    distances.append(distance_i)
    
    # Determine class
    class_i = 1 if decision_val > 0 else -1
    classes.append(class_i)
    
    # Find closest point on boundary
    proj_factor = decision_val / w_norm_squared
    closest_point_i = point - proj_factor * w
    closest_points.append(closest_point_i)
    
    # Plot line connecting point to decision boundary
    plt.plot([point[0], closest_point_i[0]], [point[1], closest_point_i[1]], 
             'k--', alpha=0.5, linewidth=1)
    
    # Annotate with distance
    mid_point = (point + closest_point_i) / 2
    plt.annotate(f'{distance_i:.2f}', xy=(mid_point[0], mid_point[1]), 
                 fontsize=9, ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))

# Plot the points
for i, (point, cls) in enumerate(zip(points, classes)):
    color = 'blue' if cls == 1 else 'red'
    plt.scatter(point[0], point[1], color=color, s=100, alpha=0.7,
                edgecolors='black', linewidths=1)
    plt.annotate(f'P{i+1}', xy=(point[0], point[1]), xytext=(5, 5), 
                 textcoords='offset points', fontsize=10)

# Plot the weight vector (perpendicular to the decision boundary)
plt.arrow(0, 0, w[0], w[1], head_width=0.2, head_length=0.3, 
          fc='green', ec='green', label='Weight Vector w')

# Add a title and labels with clear explanation
plt.title('Multiple Points and Their Distances to Decision Boundary', fontsize=14)
plt.xlabel('Feature $x_1$', fontsize=12)
plt.ylabel('Feature $x_2$', fontsize=12)

# Add a special annotation explaining the distance contours
plt.annotate('Contour lines show equal distances\nto the decision boundary', 
             xy=(0.05, 0.05), xycoords='axes fraction',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

# Add an explanation of points
plt.annotate('Red points = Negative class (-1)\nBlue points = Positive class (+1)', 
             xy=(0.70, 0.05), xycoords='axes fraction',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

plt.grid(True, alpha=0.3)
plt.legend()

# Save this new visualization
plt.savefig(os.path.join(save_dir, "distance_visualization.png"), dpi=300, bbox_inches='tight')

print("\nImages saved:")
print(f"1. Decision boundary: {os.path.join(save_dir, 'decision_boundary.png')}")
print(f"2. 3D decision function: {os.path.join(save_dir, 'decision_function_3d.png')}")
print(f"3. Distance visualization: {os.path.join(save_dir, 'distance_visualization.png')}")

print("\nSummary of Findings:")
print("------------------")
print(f"1. The distance from point (2, 3) to the decision boundary {w1}x_1 + {w2}x_2 + {b} = 0 is {distance:.4f} units")
print(f"2. For the data point (0, 1), the model predicts class {prediction}")
print(f"   This is because the decision function value f(0, 1) = {decision_value_new} is {'positive' if decision_value_new > 0 else 'negative'}")
print(f"3. When normalized, the decision boundary equation becomes: {w_normalized[0]:.4f}x_1 + {w_normalized[1]:.4f}x_2 + {b_normalized:.4f} = 0")
print("   Both equations represent the same geometric boundary but with the constraint that ||w|| = 1")
print("4. The decision boundary divides the feature space into:")
print("   - Positive region: points where decision function > 0, classified as y = +1")
print("   - Negative region: points where decision function < 0, classified as y = -1") 