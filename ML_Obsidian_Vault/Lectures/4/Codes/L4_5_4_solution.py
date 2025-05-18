import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_5_Quiz_4")
os.makedirs(save_dir, exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'

# Define the dataset
data = np.array([
    [1, 2, 1],  # [x1, x2, y]
    [2, 1, 1],
    [3, 3, 1],
    [6, 4, 0],
    [5, 6, 0],
    [7, 5, 0]
])

# Extract features and target
X = data[:, :2]
y = data[:, 2]

# Step 1: Add a column of 1s for the bias term
X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))

# Define functions for logistic regression
def sigmoid(z):
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-z))

def predict_proba(X, w):
    """Compute the probability predictions."""
    z = np.dot(X, w)
    return sigmoid(z)

def compute_gradient(X, y, w):
    """Compute the gradient of the loss function."""
    y_pred = predict_proba(X, w)
    return np.dot(X.T, (y_pred - y))

def compute_loss(X, y, w):
    """Compute the logistic loss."""
    y_pred = predict_proba(X, w)
    return -np.mean(y * np.log(y_pred + 1e-10) + (1 - y) * np.log(1 - y_pred + 1e-10))

# Initialize parameters
w = np.zeros(3)  # [w0, w1, w2]
eta = 0.1  # Learning rate

print("Initial weights:", w)

# Task 2: Perform the first SGD update using the first data point
first_point = X_with_bias[0]
first_target = y[0]

# Compute predicted probability for the first point
prob_before_update = predict_proba(first_point, w)
print(f"Predicted probability for first point before update: {prob_before_update:.6f}")

# Compute gradient for the first point (single instance SGD)
gradient_first = (predict_proba(first_point, w) - first_target) * first_point
print(f"Gradient from first point: {gradient_first}")

# Update weights using the gradient (SGD update)
w_updated = w - eta * gradient_first
print(f"Updated weights after first SGD step: {w_updated}")

# Task 3: Calculate predicted probability for second data point using updated weights
second_point = X_with_bias[1]
prob_second_point = predict_proba(second_point, w_updated)
print(f"Predicted probability for second point using updated weights: {prob_second_point:.6f}")

# Calculate the decision boundary line
# For a logistic regression model, the decision boundary is w0 + w1*x1 + w2*x2 = 0
# Solving for x2: x2 = (-w0 - w1*x1) / w2
def decision_boundary(x1, weights):
    return (-weights[0] - weights[1] * x1) / weights[2]

# Visualize the data and the initial decision boundary
plt.figure(figsize=(10, 8))

# Plot data points
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='o', label='Class 1')
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', marker='x', label='Class 0')

# Highlight the first and second data points
plt.scatter(X[0, 0], X[0, 1], c='purple', marker='*', s=200, 
            label='First point for SGD update', zorder=5)
plt.scatter(X[1, 0], X[1, 1], c='green', marker='*', s=200, 
            label='Second point for prediction', zorder=5)

# Plot the decision boundaries
x1_range = np.linspace(0, 8, 100)
plt.plot(x1_range, decision_boundary(x1_range, w), 'k--', 
         label='Initial Decision Boundary (w = [0, 0, 0])')
plt.plot(x1_range, decision_boundary(x1_range, w_updated), 'g-', 
         label=f'Updated Decision Boundary (w = {w_updated.round(4)})')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Logistic Regression: Data Points and Decision Boundaries')
plt.legend()
plt.grid(True)
plt.axis([0, 8, 0, 8])

# Save the figure
plt.savefig(os.path.join(save_dir, 'decision_boundaries.png'), dpi=300, bbox_inches='tight')

# Visualize the weights before and after update
plt.figure(figsize=(8, 6))
bar_width = 0.35
index = np.arange(3)

plt.bar(index, w, bar_width, label='Initial weights', color='skyblue')
plt.bar(index + bar_width, w_updated, bar_width, label='Updated weights', color='salmon')

plt.xlabel('Weight components')
plt.ylabel('Weight values')
plt.title('Weights Before and After SGD Update')
plt.xticks(index + bar_width/2, ['$w_0$ (bias)', '$w_1$', '$w_2$'])
plt.legend()
plt.grid(True, axis='y')

# Save the figure
plt.savefig(os.path.join(save_dir, 'weights_comparison.png'), dpi=300, bbox_inches='tight')

# Create a more detailed visualization of the SGD update
plt.figure(figsize=(10, 8))

# Compute a grid of probability values to visualize the change
x1_grid = np.linspace(0, 8, 100)
x2_grid = np.linspace(0, 8, 100)
X1, X2 = np.meshgrid(x1_grid, x2_grid)

# Create grid points with bias term
grid_points = np.column_stack((np.ones(X1.ravel().shape), X1.ravel(), X2.ravel()))

# Compute probability predictions for the grid before and after update
Z_before = predict_proba(grid_points, w).reshape(X1.shape)
Z_after = predict_proba(grid_points, w_updated).reshape(X1.shape)

# Create contour plots for probabilities before update
plt.subplot(1, 2, 1)
plt.contourf(X1, X2, Z_before, levels=20, cmap='coolwarm', alpha=0.7)
plt.colorbar(label='Probability of class 1')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='o', label='Class 1')
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', marker='x', label='Class 0')
plt.scatter(X[0, 0], X[0, 1], c='purple', marker='*', s=200)
plt.title('Before SGD Update (w = [0, 0, 0])')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.grid(True)
plt.axis([0, 8, 0, 8])

# Create contour plots for probabilities after update
plt.subplot(1, 2, 2)
plt.contourf(X1, X2, Z_after, levels=20, cmap='coolwarm', alpha=0.7)
plt.colorbar(label='Probability of class 1')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='o', label='Class 1')
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', marker='x', label='Class 0')
plt.scatter(X[1, 0], X[1, 1], c='green', marker='*', s=200)
plt.title(f'After SGD Update (w = {w_updated.round(4)})')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.grid(True)
plt.axis([0, 8, 0, 8])

plt.tight_layout()

# Save the figure
plt.savefig(os.path.join(save_dir, 'probability_contours.png'), dpi=300, bbox_inches='tight')

# Computational complexity analysis
print("\nComputational Complexity Analysis:")
print("---------------------------------")
n = len(X_with_bias)  # number of data points
d = X_with_bias.shape[1]  # number of features (including bias)

print(f"Dataset size: {n} data points, {d} features (including bias)")
print("\nBatch Gradient Descent (BGD):")
print(f"- Forward pass: O({n} × {d}) = O({n*d}) operations per iteration")
print(f"- Gradient calculation: O({n} × {d}) = O({n*d}) operations per iteration")
print("- Total operations per epoch: O(nd)")
print("\nStochastic Gradient Descent (SGD):")
print(f"- Forward pass: O({d}) operations per data point")
print(f"- Gradient calculation: O({d}) operations per data point")
print(f"- Total operations per epoch: O({n} × {d}) = O({n*d})")
print("\nComparison:")
print("- BGD: One large gradient computation using all data points")
print("- SGD: Multiple small gradient computations, one for each data point")
print("- Both have O(nd) operations per epoch theoretically")
print("- However, SGD usually converges faster in practice with fewer epochs")
print("- SGD has better memory efficiency as it processes one point at a time")
print("- BGD has more stable convergence but might be slower to reach optimum")

print("\nVisualization saved to:", save_dir) 