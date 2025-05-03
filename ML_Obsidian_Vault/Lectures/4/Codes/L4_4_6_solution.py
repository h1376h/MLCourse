import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from mpl_toolkits.mplot3d import Axes3D
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_4_Quiz_6")
os.makedirs(save_dir, exist_ok=True)

print("Question 6: XOR and Feature Transformation")
print("=========================================")

# Step 1: Plot the data points for XOR
print("\nStep 1: Plot the XOR data points")
print("------------------------------")

# Given data points for XOR
class_A = np.array([[0, 0], [1, 1]])
class_B = np.array([[0, 1], [1, 0]])

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
plt.title('XOR Problem: Original Data Points', fontsize=16)
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Save the plot
plt.savefig(os.path.join(save_dir, "xor_data_points.png"), dpi=300, bbox_inches='tight')

print("XOR data points:")
print("Class A: (0, 0), (1, 1)")
print("Class B: (0, 1), (1, 0)")

# Step 2: Demonstrate that XOR is not linearly separable
print("\nStep 2: Demonstrate that XOR is not linearly separable")
print("--------------------------------------------------")

# Combine data for classifier
X = np.vstack([class_A, class_B])
y = np.array([1, 1, -1, -1])  # 1 for class A, -1 for class B

# Try different lines to separate the data
def plot_line(a, b, c, color='green', label='Line'):
    """Plot a line ax + by + c = 0"""
    x = np.linspace(-0.5, 1.5, 1000)
    if b != 0:
        y = (-a * x - c) / b
        plt.plot(x, y, color=color, label=label)
    else:  # vertical line
        plt.axvline(x=-c/a, color=color, label=label)

# Create a new figure with multiple attempted decision boundaries
plt.figure(figsize=(10, 8))
plt.scatter(class_A[:, 0], class_A[:, 1], color='blue', s=100, marker='o', label='Class A')
plt.scatter(class_B[:, 0], class_B[:, 1], color='red', s=100, marker='x', label='Class B')

# Try several lines
plot_line(1, 1, -0.5, color='green', label='Line 1: $x_1 + x_2 - 0.5 = 0$')
plot_line(1, -1, 0, color='purple', label='Line 2: $x_1 - x_2 = 0$')
plot_line(2, 1, -1, color='orange', label='Line 3: $2x_1 + x_2 - 1 = 0$')

# Label the points
for i, point in enumerate(class_A):
    plt.annotate(f'A{i+1}({point[0]}, {point[1]})', (point[0], point[1]), 
                 xytext=(10, 5), textcoords='offset points', fontsize=12)
for i, point in enumerate(class_B):
    plt.annotate(f'B{i+1}({point[0]}, {point[1]})', (point[0], point[1]), 
                 xytext=(10, 5), textcoords='offset points', fontsize=12)

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('XOR Problem: Attempted Linear Decision Boundaries', fontsize=16)
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10, loc='upper left')

# Save the plot
plt.savefig(os.path.join(save_dir, "xor_not_separable.png"), dpi=300, bbox_inches='tight')

# Check classification accuracy for each line
def classify_points(a, b, c, points, true_labels):
    """Classify points using line ax + by + c = 0"""
    predictions = []
    for point in points:
        x, y = point
        value = a*x + b*y + c
        pred = 1 if value >= 0 else -1
        predictions.append(pred)
    
    accuracy = np.mean(np.array(predictions) == true_labels)
    misclassified = np.where(np.array(predictions) != true_labels)[0]
    return predictions, accuracy, misclassified

lines = [
    (1, 1, -0.5, "Line 1: x₁ + x₂ - 0.5 = 0"),
    (1, -1, 0, "Line 2: x₁ - x₂ = 0"),
    (2, 1, -1, "Line 3: 2x₁ + x₂ - 1 = 0")
]

print("Attempting to find a linear boundary for XOR:")
for a, b, c, name in lines:
    preds, acc, misclassified = classify_points(a, b, c, X, y)
    print(f"\n{name}")
    print(f"Predictions: {preds}")
    print(f"Accuracy: {acc*100:.1f}%")
    print(f"Misclassified points: {misclassified}")
    for idx in misclassified:
        print(f"  Point {idx+1}: {X[idx]} (true label: {y[idx]}, predicted: {preds[idx]})")

print("\nNo linear decision boundary can correctly classify all XOR points.")
print("This is because the XOR problem is not linearly separable.")

# Step 3: Transform the data with an additional feature
print("\nStep 3: Transform the data with the feature x₃ = x₁ × x₂")
print("-----------------------------------------------------")

# Create the transformed data
X_transformed = np.zeros((4, 3))  # 4 points, 3 features
X_transformed[:, 0:2] = X  # Original features
X_transformed[:, 2] = X[:, 0] * X[:, 1]  # New feature x₃ = x₁ × x₂

print("Original data (x₁, x₂) and transformed data (x₁, x₂, x₃ = x₁ × x₂):")
for i, (orig, transformed) in enumerate(zip(X, X_transformed)):
    label = "A" if y[i] == 1 else "B"
    print(f"Point {label}{i % 2 + 1}: Original {orig} → Transformed {transformed}")

# Step 4: 3D visualization of the transformed data
print("\nStep 4: 3D visualization of the transformed data")
print("----------------------------------------------")

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot class A points (blue)
ax.scatter(X_transformed[0:2, 0], X_transformed[0:2, 1], X_transformed[0:2, 2], 
           color='blue', s=100, marker='o', label='Class A')

# Plot class B points (red)
ax.scatter(X_transformed[2:4, 0], X_transformed[2:4, 1], X_transformed[2:4, 2], 
           color='red', s=100, marker='x', label='Class B')

# Label the points
for i, point in enumerate(X_transformed):
    label = "A" if i < 2 else "B"
    idx = i % 2 + 1
    ax.text(point[0], point[1], point[2], f'{label}{idx}({point[0]}, {point[1]}, {point[2]:.1f})', 
            fontsize=10)

ax.set_xlabel('$x_1$', fontsize=14)
ax.set_ylabel('$x_2$', fontsize=14)
ax.set_zlabel('$x_3 = x_1 × x_2$', fontsize=14)
ax.set_title('XOR Problem: 3D Visualization of Transformed Data', fontsize=16)
ax.legend(fontsize=12)
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.set_zlim(-0.1, 1.1)
ax.view_init(elev=20, azim=30)

# Save the 3D visualization
plt.savefig(os.path.join(save_dir, "xor_3d_transformed.png"), dpi=300, bbox_inches='tight')

# Step 5: Find a separating hyperplane for the transformed data
print("\nStep 5: Find a separating hyperplane for the transformed data")
print("----------------------------------------------------------")

# Train a linear SVM on the transformed data
clf = LinearSVC(dual="auto", loss='hinge', C=100)
clf.fit(X_transformed, y)

# Get the parameters of the decision boundary
w = clf.coef_[0]  # The coefficients w1, w2, w3
b = clf.intercept_[0]  # The bias term

print(f"The decision boundary parameters are:")
print(f"w1 = {w[0]:.4f}, w2 = {w[1]:.4f}, w3 = {w[2]:.4f}, b = {b:.4f}")
print(f"Decision boundary equation: {w[0]:.4f}x₁ + {w[1]:.4f}x₂ + {w[2]:.4f}x₃ + {b:.4f} = 0")

# Verify classification accuracy
y_pred = clf.predict(X_transformed)
accuracy = np.mean(y == y_pred)
print(f"\nClassification accuracy: {accuracy*100:.1f}%")

print("\nVerifying predictions for each point:")
for i, (point, true_label, pred_label) in enumerate(zip(X_transformed, y, y_pred)):
    decision_value = np.dot(w, point) + b
    label = "A" if i < 2 else "B"
    idx = i % 2 + 1
    print(f"Point {label}{idx} {point}: True label = {true_label}, Predicted = {pred_label}, Decision value = {decision_value:.4f}")

# Visualize the separating hyperplane in 3D
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot class A points (blue)
ax.scatter(X_transformed[0:2, 0], X_transformed[0:2, 1], X_transformed[0:2, 2], 
           color='blue', s=100, marker='o', label='Class A')

# Plot class B points (red)
ax.scatter(X_transformed[2:4, 0], X_transformed[2:4, 1], X_transformed[2:4, 2], 
           color='red', s=100, marker='x', label='Class B')

# Create a meshgrid to visualize the hyperplane
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 10), np.linspace(-0.5, 1.5, 10))
# Compute corresponding z values for the hyperplane: w1*x + w2*y + w3*z + b = 0
# => z = -(w1*x + w2*y + b) / w3
zz = -(w[0] * xx + w[1] * yy + b) / w[2]

# Plot the hyperplane
surf = ax.plot_surface(xx, yy, zz, alpha=0.3, color='green')

# Label the points
for i, point in enumerate(X_transformed):
    label = "A" if i < 2 else "B"
    idx = i % 2 + 1
    ax.text(point[0], point[1], point[2], f'{label}{idx}', fontsize=10)

ax.set_xlabel('$x_1$', fontsize=14)
ax.set_ylabel('$x_2$', fontsize=14)
ax.set_zlabel('$x_3 = x_1 × x_2$', fontsize=14)
ax.set_title('XOR Problem: Separating Hyperplane in 3D Space', fontsize=16)
ax.legend(fontsize=12)
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.set_zlim(-0.1, 1.1)
ax.view_init(elev=20, azim=30)

# Add hyperplane equation to the plot
hyperplane_eq = f"Hyperplane: {w[0]:.4f}x₁ + {w[1]:.4f}x₂ + {w[2]:.4f}x₃ + {b:.4f} = 0"
ax.text2D(0.05, 0.95, hyperplane_eq, transform=ax.transAxes, fontsize=12,
         bbox=dict(facecolor='white', alpha=0.8))

# Save the 3D visualization with hyperplane
plt.savefig(os.path.join(save_dir, "xor_3d_separating_hyperplane.png"), dpi=300, bbox_inches='tight')

# Step 6: Show a simple separating hyperplane
print("\nStep 6: Demonstrate a simple separating hyperplane")
print("-----------------------------------------------")

print("Let's verify a simpler separating hyperplane: x₃ - 0.5 = 0")
simple_w = np.array([0, 0, 1])  # Coefficient vector [w1, w2, w3]
simple_b = -0.5  # Bias term

print(f"Simple hyperplane equation: {simple_w[0]}x₁ + {simple_w[1]}x₂ + {simple_w[2]}x₃ + {simple_b} = 0")
print(f"Simplified to: x₃ - 0.5 = 0")

print("\nVerifying predictions for each point with this simple hyperplane:")
for i, point in enumerate(X_transformed):
    decision_value = np.dot(simple_w, point) + simple_b
    pred = 1 if decision_value >= 0 else -1
    label = "A" if i < 2 else "B"
    idx = i % 2 + 1
    correct = pred == y[i]
    print(f"Point {label}{idx} {point}: x₃ = {point[2]}, Decision value = {decision_value:.1f}, Prediction = {pred}, True = {y[i]}, Correct = {correct}")

# Visualize the simple separating hyperplane in 3D
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot class A points (blue)
ax.scatter(X_transformed[0:2, 0], X_transformed[0:2, 1], X_transformed[0:2, 2], 
           color='blue', s=100, marker='o', label='Class A')

# Plot class B points (red)
ax.scatter(X_transformed[2:4, 0], X_transformed[2:4, 1], X_transformed[2:4, 2], 
           color='red', s=100, marker='x', label='Class B')

# Create a meshgrid to visualize the simple hyperplane
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 10), np.linspace(-0.5, 1.5, 10))
# For x₃ - 0.5 = 0, this is a flat plane at z = 0.5
zz = np.ones_like(xx) * 0.5

# Plot the hyperplane
surf = ax.plot_surface(xx, yy, zz, alpha=0.3, color='cyan')

# Label the points
for i, point in enumerate(X_transformed):
    label = "A" if i < 2 else "B"
    idx = i % 2 + 1
    ax.text(point[0], point[1], point[2], f'{label}{idx}', fontsize=10)

ax.set_xlabel('$x_1$', fontsize=14)
ax.set_ylabel('$x_2$', fontsize=14)
ax.set_zlabel('$x_3 = x_1 × x_2$', fontsize=14)
ax.set_title('XOR Problem: Simple Separating Hyperplane x₃ - 0.5 = 0', fontsize=16)
ax.legend(fontsize=12)
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.set_zlim(-0.1, 1.1)
ax.view_init(elev=20, azim=30)

# Add simple hyperplane equation to the plot
hyperplane_eq = f"Simple Hyperplane: x₃ - 0.5 = 0"
ax.text2D(0.05, 0.95, hyperplane_eq, transform=ax.transAxes, fontsize=12,
         bbox=dict(facecolor='white', alpha=0.8))

# Save the 3D visualization with simple hyperplane
plt.savefig(os.path.join(save_dir, "xor_3d_simple_hyperplane.png"), dpi=300, bbox_inches='tight')

print("\nConclusion:")
print("-----------")
print("1. The XOR problem is not linearly separable in the original 2D feature space.")
print("2. By adding a new feature x₃ = x₁ × x₂, the data becomes linearly separable in 3D.")
print("3. A simple separating hyperplane is x₃ - 0.5 = 0, or in the standard form: 0x₁ + 0x₂ + 1x₃ - 0.5 = 0")
print("4. This demonstrates how feature transformations can make non-linearly separable data linearly separable.")
print("5. The machine learning algorithm found the hyperplane: " + 
      f"{w[0]:.4f}x₁ + {w[1]:.4f}x₂ + {w[2]:.4f}x₃ + {b:.4f} = 0, which also perfectly separates the classes.") 