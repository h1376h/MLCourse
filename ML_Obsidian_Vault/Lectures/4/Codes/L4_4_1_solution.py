import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
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

# Step 2: Find a linear decision boundary
print("\nStep 2: Find a linear decision boundary")
print("----------------------------------")

# Prepare data for the SVM
X = np.vstack([class_A, class_B])
y = np.array([1, 1, -1, -1])  # 1 for class A, -1 for class B

# Train a linear SVM
clf = LinearSVC(dual="auto", loss='hinge', C=100)
clf.fit(X, y)

# Get the parameters of the decision boundary
w = clf.coef_[0]  # The coefficients w1, w2
b = clf.intercept_[0]  # The bias term

print(f"The decision boundary parameters are:")
print(f"w1 = {w[0]:.4f}, w2 = {w[1]:.4f}, b = {b:.4f}")
print(f"Decision boundary equation: {w[0]:.4f}x₁ + {w[1]:.4f}x₂ + {b:.4f} = 0")

# Step 3: Plot the decision boundary
print("\nStep 3: Plot the decision boundary")
print("-------------------------------")

# Create a meshgrid to plot the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Get the prediction for each point in the meshgrid
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Create the final plot with decision boundary
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

# Plot the decision boundary and margins
plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['red', 'black', 'blue'], 
            linestyles=['--', '-', '--'], linewidths=[1, 2, 1])

# Fill the regions
plt.contourf(xx, yy, Z, levels=[-float('inf'), 0, float('inf')], colors=['#FFAAAA', '#AAAAFF'], alpha=0.3)

# Add decision boundary equation to the plot
boundary_eq = f"${w[0]:.2f}x_1 + {w[1]:.2f}x_2 + {b:.2f} = 0$"
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

# Predict the class labels
y_pred = clf.predict(X)

# Check if all points are correctly classified
is_separable = np.all(y == y_pred)

print("Checking if each point is correctly classified by the decision boundary:")
for i, (point, true_label, pred_label) in enumerate(zip(X, y, y_pred)):
    is_correct = true_label == pred_label
    print(f"Point {i+1}: {point} - True label: {true_label}, Predicted label: {pred_label}, Correctly classified: {is_correct}")

# Distance of each point to the decision boundary
distances = (np.dot(X, w) + b) / np.linalg.norm(w)
print("\nDistance of each point to the decision boundary:")
for i, (point, dist) in enumerate(zip(X, distances)):
    print(f"Point {i+1}: {point} - Distance: {dist:.4f}")

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

# Plot the manual decision boundary
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
        
        if w[1] > 0:  # Adjust based on the orientation of the boundary
            plt.fill_between(xx_fill, y_above, y_max, alpha=0.2, color='blue')
            plt.fill_between(xx_fill, y_min, y_below, alpha=0.2, color='red')
        else:
            plt.fill_between(xx_fill, y_above, y_max, alpha=0.2, color='red')
            plt.fill_between(xx_fill, y_min, y_below, alpha=0.2, color='blue')

# Plot both the SVM and manual decision boundaries
plot_line(w, b, color='black', label='SVM Boundary')
plot_line(w_manual, b_manual, color='purple', label='Alternative Boundary')

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('Multiple Linear Decision Boundaries', fontsize=16)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

boundary_eq_svm = f"SVM: ${w[0]:.2f}x_1 + {w[1]:.2f}x_2 + {b:.2f} = 0$"
boundary_eq_manual = f"Alt: ${w_manual[0]}x_1 + {w_manual[1]}x_2 + {b_manual} = 0$"
plt.text(0.05, 0.95, boundary_eq_svm, transform=plt.gca().transAxes, 
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
print(f"4. The SVM boundary equation is: {w[0]:.4f}x₁ + {w[1]:.4f}x₂ + {b:.4f} = 0")
print(f"5. The alternative boundary equation is: {w_manual[0]}x₁ + {w_manual[1]}x₂ + {b_manual} = 0") 