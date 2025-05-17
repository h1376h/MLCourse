import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_4_Quiz_1")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

print("Question 1: Linear Separability in 2D")
print("====================================")

# Step 1: Plot the data points
print("\nStep 1: Plot the data points")
print("--------------------------")

# Given data points
class_A = np.array([[1, 1], [2, 3]])
class_B = np.array([[-1, 0], [0, -2]])

# Create the plot
plt.figure(figsize=(10, 8))
plt.scatter(class_A[:, 0], class_A[:, 1], color='blue', s=100, marker='o', label='Class A')
plt.scatter(class_B[:, 0], class_B[:, 1], color='red', s=100, marker='x', label='Class B')

# Label the points
for i, point in enumerate(class_A):
    plt.annotate(f'A{i+1}({point[0]}, {point[1]})', (point[0], point[1]), 
                 xytext=(10, 5), textcoords='offset points', fontsize=12)
for i, point in enumerate(class_B):
    plt.annotate(f'B{i+1}({point[0]}, {point[1]})', (point[0], point[1]), 
                 xytext=(10, 5), textcoords='offset points', fontsize=12)

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('Plotted Data Points', fontsize=16)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Save the plot without decision boundary
plt.savefig(os.path.join(save_dir, "data_points.png"), dpi=300, bbox_inches='tight')

print("Data points plotted:")
print("Class A: (1, 1), (2, 3)")
print("Class B: (-1, 0), (0, -2)")
print("The points are well separated in the 2D feature space.")

# Step 2: Find a linear decision boundary using basic geometry
print("\nStep 2: Find a linear decision boundary")
print("----------------------------------")

# Calculate the center of each class (centroid)
center_A = np.mean(class_A, axis=0)
center_B = np.mean(class_B, axis=0)
print(f"Center of Class A: ({center_A[0]}, {center_A[1]})")
print(f"Center of Class B: ({center_B[0]}, {center_B[1]})")

# Calculate the slope of the line connecting the centroids
# m = (y2 - y1) / (x2 - x1)
m_centroids = (center_B[1] - center_A[1]) / (center_B[0] - center_A[0])
print(f"Slope of line connecting centroids: {m_centroids:.4f}")

# The perpendicular line will be our decision boundary
# m1 * m2 = -1 => m2 = -1/m1
m_boundary = -1 / m_centroids
print(f"Slope of the decision boundary: {m_boundary:.4f}")

# Calculate the midpoint between the centroids
midpoint = (center_A + center_B) / 2
print(f"Midpoint between centroids: ({midpoint[0]}, {midpoint[1]})")

# Use the point-slope form to get the decision boundary equation
# y - y1 = m(x - x1)
# Where (x1, y1) is the midpoint and m is the slope of the boundary
# Rearranging to standard form: y = mx + b => -mx + y - b = 0
# Therefore: w1 = -m, w2 = 1, b = -b_intercept
b_intercept = midpoint[1] - m_boundary * midpoint[0]
w1 = -m_boundary
w2 = 1
b = -b_intercept

# Convert to the standard form: w1*x1 + w2*x2 + b = 0
print(f"The decision boundary parameters are:")
print(f"w1 = {w1:.4f}, w2 = {w2:.4f}, b = {b:.4f}")
print(f"Decision boundary equation: {w1:.4f}x₁ + {w2:.4f}x₂ + {b:.4f} = 0")

# Step 3: Plot the decision boundary
print("\nStep 3: Plot the decision boundary")
print("-------------------------------")

# Create a meshgrid to plot the decision boundary
X = np.vstack([class_A, class_B])
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Calculate decision function values for the meshgrid
Z = w1 * xx + w2 * yy + b
Z = Z.reshape(xx.shape)

# Create the final plot with decision boundary
plt.figure(figsize=(10, 8))
plt.scatter(class_A[:, 0], class_A[:, 1], color='blue', s=100, marker='o', label='Class A')
plt.scatter(class_B[:, 0], class_B[:, 1], color='red', s=100, marker='x', label='Class B')
plt.scatter([midpoint[0]], [midpoint[1]], color='green', s=100, marker='*', label='Midpoint')
plt.scatter([center_A[0]], [center_A[1]], color='cyan', s=100, marker='D', label='Center A')
plt.scatter([center_B[0]], [center_B[1]], color='magenta', s=100, marker='D', label='Center B')

# Label the points
for i, point in enumerate(class_A):
    plt.annotate(f'A{i+1}({point[0]}, {point[1]})', (point[0], point[1]), 
                 xytext=(10, 5), textcoords='offset points', fontsize=12)
for i, point in enumerate(class_B):
    plt.annotate(f'B{i+1}({point[0]}, {point[1]})', (point[0], point[1]), 
                 xytext=(10, 5), textcoords='offset points', fontsize=12)

# Plot the decision boundary
plt.contour(xx, yy, Z, levels=[0], colors=['black'], linestyles=['-'], linewidths=[2])

# Fill the regions
plt.contourf(xx, yy, Z, levels=[-float('inf'), 0, float('inf')], colors=['#FFAAAA', '#AAAAFF'], alpha=0.3)

# Add decision boundary equation to the plot
boundary_eq = f"${w1:.2f}x_1 + {w2:.2f}x_2 + {b:.2f} = 0$"
plt.text(0.05, 0.9, f"Decision Boundary:\n{boundary_eq}", transform=plt.gca().transAxes, 
         fontsize=14, bbox=dict(facecolor='white', alpha=0.8))

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('Linear Decision Boundary for the Classification Task', fontsize=16)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Save the final plot
plt.savefig(os.path.join(save_dir, "decision_boundary.png"), dpi=300, bbox_inches='tight')

print("The linear decision boundary has been plotted.")
print("The line is positioned to separate the two classes.")

# Step 4: Check classification accuracy and separability
print("\nStep 4: Check if the data is linearly separable")
print("------------------------------------------")

# Calculate decision function values for data points
decision_values = np.dot(X, [w1, w2]) + b
y = np.array([1, 1, -1, -1])  # 1 for class A, -1 for class B
y_pred = np.sign(decision_values)

# Check if all points are correctly classified
is_separable = np.all(y == y_pred)

print("Checking if each point is correctly classified by the decision boundary:")
for i, (point, true_label, pred_label, score) in enumerate(zip(X, y, y_pred, decision_values)):
    is_correct = true_label == pred_label
    print(f"Point {i+1}: {point} - True label: {true_label}, Predicted label: {pred_label}, Score: {score:.4f}, Correctly classified: {is_correct}")

# Verification of linear separability
if is_separable:
    print("\nThe dataset is linearly separable!")
    print("This means a linear decision boundary can perfectly separate the two classes.")
else:
    print("\nThe dataset is NOT linearly separable.")
    print("Some points cannot be correctly classified by any linear decision boundary.")

# Additional visualization: Manually verifying a different decision boundary
print("\nStep 5: Manual verification with an alternative decision boundary")
print("----------------------------------------------------------")

# Let's try a different decision boundary (manually chosen)
w_manual = np.array([1, 1])  # A different weight vector
b_manual = -1.5             # A different bias term

# Calculate the decision scores for each point
scores_manual = np.dot(X, w_manual) + b_manual
pred_manual = np.sign(scores_manual)

# Check if this decision boundary also separates the data
is_separable_manual = np.all(y == pred_manual)

print(f"Alternative decision boundary: {w_manual[0]}x₁ + {w_manual[1]}x₂ + {b_manual} = 0")
print("Classification results with this boundary:")
for i, (point, true_label, pred_label, score) in enumerate(zip(X, y, pred_manual, scores_manual)):
    is_correct = true_label == pred_label
    print(f"Point {i+1}: {point} - True label: {true_label}, Predicted label: {pred_label}, Score: {score:.4f}, Correctly classified: {is_correct}")

print(f"\nIs the data linearly separable with this alternative boundary? {is_separable_manual}")

# Plot the alternative decision boundary
plt.figure(figsize=(10, 8))
plt.scatter(class_A[:, 0], class_A[:, 1], color='blue', s=100, marker='o', label='Class A')
plt.scatter(class_B[:, 0], class_B[:, 1], color='red', s=100, marker='x', label='Class B')

# Label the points
for i, point in enumerate(class_A):
    plt.annotate(f'A{i+1}({point[0]}, {point[1]})', (point[0], point[1]), 
                 xytext=(10, 5), textcoords='offset points', fontsize=12)
for i, point in enumerate(class_B):
    plt.annotate(f'B{i+1}({point[0]}, {point[1]})', (point[0], point[1]), 
                 xytext=(10, 5), textcoords='offset points', fontsize=12)

# Function to plot a decision boundary line
def plot_line(w, b, color='purple', label='Alternative'):
    # Create a line based on the weights and bias
    if w[1] != 0:  # Check if w2 is not zero to avoid division by zero
        x_vals = np.array([x_min, x_max])
        y_vals = (-w[0] * x_vals - b) / w[1]
    else:  # If w2 is zero, it's a vertical line
        x_vals = np.array([-b / w[0], -b / w[0]])
        y_vals = np.array([y_min, y_max])
    
    plt.plot(x_vals, y_vals, color=color, linestyle='-', linewidth=2, label=label)
    
    # Shade the regions
    if w[1] != 0:
        xx_fill = np.linspace(x_min, x_max, 100)
        yy_boundary = (-w[0] * xx_fill - b) / w[1]
        
        # Get points above and below the boundary
        y_above = np.maximum(yy_boundary, y_min)
        y_below = np.minimum(yy_boundary, y_max)
        
        if (w[0]/w[1]) < 0:  # Adjust based on the orientation of the boundary
            plt.fill_between(xx_fill, y_above, y_max, alpha=0.2, color='blue')
            plt.fill_between(xx_fill, y_min, y_below, alpha=0.2, color='red')
        else:
            plt.fill_between(xx_fill, y_above, y_max, alpha=0.2, color='red')
            plt.fill_between(xx_fill, y_min, y_below, alpha=0.2, color='blue')

# Plot both the geometric and manual decision boundaries
plot_line([w1, w2], b, color='black', label='Geometric Boundary')
plot_line(w_manual, b_manual, color='purple', label='Alternative Boundary')

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('Multiple Linear Decision Boundaries', fontsize=16)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

boundary_eq_geom = f"Geom: ${w1:.2f}x_1 + {w2:.2f}x_2 + {b:.2f} = 0$"
boundary_eq_manual = f"Alt: ${w_manual[0]}x_1 + {w_manual[1]}x_2 + {b_manual} = 0$"
plt.text(0.05, 0.95, boundary_eq_geom, transform=plt.gca().transAxes, 
         fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
plt.text(0.05, 0.87, boundary_eq_manual, transform=plt.gca().transAxes, 
         fontsize=14, bbox=dict(facecolor='white', alpha=0.8))

# Save the alternative boundary plot
plt.savefig(os.path.join(save_dir, "multiple_boundaries.png"), dpi=300, bbox_inches='tight')

print("\nConclusion:")
print("-----------")
print("The dataset is linearly separable because:")
print("1. We found multiple linear decision boundaries that perfectly separate the classes")
print("2. All points are correctly classified by both boundaries")
print("3. Each class lies entirely on one side of the decision boundary")
print(f"4. The geometric boundary equation is: {w1:.4f}x₁ + {w2:.4f}x₂ + {b:.4f} = 0")
print(f"5. The alternative boundary equation is: {w_manual[0]}x₁ + {w_manual[1]}x₂ + {b_manual} = 0") 