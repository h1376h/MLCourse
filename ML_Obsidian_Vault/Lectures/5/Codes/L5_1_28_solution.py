import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_1_Quiz_28")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("=== SVM Terminology Demonstration for Question 28 ===\n")

# 1. Support Vectors Demonstration
print("1. SUPPORT VECTORS")
print("=" * 50)

# Generate linearly separable data
np.random.seed(42)
X, y = make_blobs(n_samples=100, centers=2, cluster_std=1.5, random_state=42)
# Ensure binary labels
y = np.where(y == 0, -1, 1)

# Train SVM
svm = SVC(kernel='linear', C=1000)  # High C for hard margin
svm.fit(X, y)

# Get support vectors
support_vectors = svm.support_vectors_
support_vector_indices = svm.support_

print(f"Total training points: {len(X)}")
print(f"Number of support vectors: {len(support_vectors)}")
print(f"Support vector indices: {support_vector_indices}")
print(f"Support vectors:\n{support_vectors}")

# Visualize support vectors
plt.figure(figsize=(12, 10))

# Plot all points
plt.scatter(X[y == -1, 0], X[y == -1, 1], c='lightblue', s=100, 
           label='Class -1 (Non-Support Vectors)', alpha=0.7, edgecolors='blue')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='lightcoral', s=100, 
           label='Class 1 (Non-Support Vectors)', alpha=0.7, edgecolors='red')

# Highlight support vectors
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
           c='yellow', s=200, marker='s', edgecolors='black', linewidth=2,
           label='Support Vectors')

# Plot decision boundary
w = svm.coef_[0]
b = svm.intercept_[0]
slope = -w[0] / w[1]
xx = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
yy = slope * xx - b / w[1]

plt.plot(xx, yy, 'g-', linewidth=3, label='Decision Boundary')

# Plot margin boundaries
margin_1 = yy + 1 / np.linalg.norm(w)
margin_2 = yy - 1 / np.linalg.norm(w)
plt.plot(xx, margin_1, 'g--', alpha=0.7, label='Margin Boundary (+1)')
plt.plot(xx, margin_2, 'g--', alpha=0.7, label='Margin Boundary (-1)')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Support Vectors in SVM')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

# Add text annotations
plt.annotate('Support vectors are the points\nclosest to the decision boundary\nthat define the margin', 
             xy=(0.02, 0.98), xycoords='axes fraction',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

plt.savefig(os.path.join(save_dir, 'support_vectors.png'), dpi=300, bbox_inches='tight')
plt.show()

# 2. Linear Separability Demonstration
print("\n2. LINEAR SEPARABILITY")
print("=" * 50)

# Create linearly separable data
np.random.seed(123)
X_lin, y_lin = make_blobs(n_samples=50, centers=2, cluster_std=1.0, random_state=123)
y_lin = np.where(y_lin == 0, -1, 1)

# Create non-linearly separable data (XOR-like)
np.random.seed(456)
X_nonlin = np.random.randn(100, 2) * 1.5
y_nonlin = np.where((X_nonlin[:, 0] > 0) & (X_nonlin[:, 1] > 0) | 
                    (X_nonlin[:, 0] < 0) & (X_nonlin[:, 1] < 0), 1, -1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Linearly separable data
svm_lin = SVC(kernel='linear', C=1000)
svm_lin.fit(X_lin, y_lin)

ax1.scatter(X_lin[y_lin == -1, 0], X_lin[y_lin == -1, 1], c='lightblue', s=100, 
           label='Class -1', alpha=0.7, edgecolors='blue')
ax1.scatter(X_lin[y_lin == 1, 0], X_lin[y_lin == 1, 1], c='lightcoral', s=100, 
           label='Class 1', alpha=0.7, edgecolors='red')

# Plot decision boundary for linearly separable
w_lin = svm_lin.coef_[0]
b_lin = svm_lin.intercept_[0]
slope_lin = -w_lin[0] / w_lin[1]
xx_lin = np.linspace(X_lin[:, 0].min() - 1, X_lin[:, 0].max() + 1, 100)
yy_lin = slope_lin * xx_lin - b_lin / w_lin[1]
ax1.plot(xx_lin, yy_lin, 'g-', linewidth=3, label='Linear Separator')

ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.set_title('Linearly Separable Data')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Non-linearly separable data
ax2.scatter(X_nonlin[y_nonlin == -1, 0], X_nonlin[y_nonlin == -1, 1], c='lightblue', s=100, 
           label='Class -1', alpha=0.7, edgecolors='blue')
ax2.scatter(X_nonlin[y_nonlin == 1, 0], X_nonlin[y_nonlin == 1, 1], c='lightcoral', s=100, 
           label='Class 1', alpha=0.7, edgecolors='red')

ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.set_title('Non-Linearly Separable Data (XOR-like)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'linear_separability.png'), dpi=300, bbox_inches='tight')
plt.show()

print("Linearly separable: Data can be perfectly separated by a straight line")
print("Non-linearly separable: Data cannot be separated by a straight line")

# 3. Primal vs Dual Formulation
print("\n3. PRIMAL vs DUAL FORMULATION")
print("=" * 50)

# Demonstrate the relationship between primal and dual
w_primal = svm.coef_[0]
b_primal = svm.intercept_[0]
alphas = svm.dual_coef_[0]  # These are the Lagrange multipliers

print("Primal Formulation:")
print(f"  Weight vector w = {w_primal}")
print(f"  Bias term b = {b_primal}")
print(f"  ||w||² = {np.linalg.norm(w_primal)**2:.4f}")

print("\nDual Formulation:")
print(f"  Number of support vectors: {len(alphas)}")
print(f"  Lagrange multipliers (α): {alphas}")
print(f"  Sum of α_i * y_i: {np.sum(alphas * y[support_vector_indices]):.6f}")

# Verify the relationship: w = Σ(α_i * y_i * x_i)
w_from_dual = np.zeros(2)
for i, alpha in enumerate(alphas):
    w_from_dual += alpha * y[support_vector_indices[i]] * support_vectors[i]

print(f"\nVerification:")
print(f"  w from primal: {w_primal}")
print(f"  w from dual: {w_from_dual}")
print(f"  Difference: {np.linalg.norm(w_primal - w_from_dual):.10f}")

# 4. Margin Demonstration
print("\n4. MARGIN IN MAXIMUM MARGIN CLASSIFICATION")
print("=" * 50)

# Calculate margin
margin_width = 2 / np.linalg.norm(w_primal)
print(f"Margin width = 2 / ||w|| = {margin_width:.4f}")

# Visualize margin
plt.figure(figsize=(12, 10))

# Plot all points
plt.scatter(X[y == -1, 0], X[y == -1, 1], c='lightblue', s=100, 
           label='Class -1', alpha=0.7, edgecolors='blue')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='lightcoral', s=100, 
           label='Class 1', alpha=0.7, edgecolors='red')

# Highlight support vectors
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
           c='yellow', s=200, marker='s', edgecolors='black', linewidth=2,
           label='Support Vectors')

# Plot decision boundary
plt.plot(xx, yy, 'g-', linewidth=3, label='Decision Boundary')

# Plot margin boundaries with shading
plt.plot(xx, margin_1, 'g--', linewidth=2, label='Margin Boundary (+1)')
plt.plot(xx, margin_2, 'g--', linewidth=2, label='Margin Boundary (-1)')

# Shade the margin region
plt.fill_between(xx, margin_2, margin_1, alpha=0.2, color='green', label='Margin Region')

# Add margin width annotation
mid_x = (xx[0] + xx[-1]) / 2
mid_y = slope * mid_x - b_primal / w_primal[1]
plt.annotate(f'Margin Width = {margin_width:.3f}', 
             xy=(mid_x, mid_y), xytext=(mid_x + 1, mid_y + 1),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=12, color='red')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Maximum Margin Classification')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

# Add text box explaining margin
plt.annotate('The margin is the distance between\nthe decision boundary and the\nclosest data points (support vectors).\nLarger margin = better generalization.', 
             xy=(0.02, 0.98), xycoords='axes fraction',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

plt.savefig(os.path.join(save_dir, 'margin_demonstration.png'), dpi=300, bbox_inches='tight')
plt.show()

# 5. Hyperplane Definition and 2D Example
print("\n5. HYPERPLANE DEFINITION")
print("=" * 50)

# Example hyperplane: 2x₁ + 3x₂ - 6 = 0
w_example = np.array([2, 3])
b_example = -6

print(f"Example 2D hyperplane: {w_example[0]}x₁ + {w_example[1]}x₂ + {b_example} = 0")
print(f"  Weight vector w = {w_example}")
print(f"  Bias term b = {b_example}")
print(f"  ||w|| = {np.linalg.norm(w_example):.4f}")

# Visualize the hyperplane
plt.figure(figsize=(10, 8))

# Generate some points
x1_range = np.linspace(-2, 5, 100)
x2_range = np.linspace(-2, 5, 100)
X1, X2 = np.meshgrid(x1_range, x2_range)

# Calculate which side each point is on
Z = w_example[0] * X1 + w_example[1] * X2 + b_example

# Plot the hyperplane
plt.contour(X1, X2, Z, levels=[0], colors='red', linewidths=3, label='Hyperplane')

# Shade the regions
plt.contourf(X1, X2, Z, levels=[-100, 0], colors=['lightblue'], alpha=0.3)
plt.contourf(X1, X2, Z, levels=[0, 100], colors=['lightcoral'], alpha=0.3)

# Add some example points
example_points = np.array([[1, 1], [2, 2], [0, 0], [3, 1]])
for point in example_points:
    side = w_example[0] * point[0] + w_example[1] * point[1] + b_example
    color = 'blue' if side < 0 else 'red'
    plt.scatter(point[0], point[1], c=color, s=100, edgecolors='black', linewidth=2)
    plt.annotate(f'({point[0]}, {point[1]})', (point[0], point[1]), 
                 xytext=(5, 5), textcoords='offset points')

# Plot weight vector (perpendicular to hyperplane)
origin = np.array([1, 1])  # Point on the hyperplane
weight_end = origin + w_example / np.linalg.norm(w_example) * 0.5
plt.arrow(origin[0], origin[1], w_example[0]/np.linalg.norm(w_example)*0.5, 
          w_example[1]/np.linalg.norm(w_example)*0.5, 
          head_width=0.1, head_length=0.1, fc='green', ec='green', linewidth=2,
          label='Weight Vector w')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('2D Hyperplane Example: $2x_1 + 3x_2 - 6 = 0$')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

# Add text box explaining hyperplane
plt.annotate('A hyperplane in 2D is a line.\nThe weight vector w is perpendicular\nto the hyperplane and points toward\nthe positive region.', 
             xy=(0.02, 0.98), xycoords='axes fraction',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

plt.savefig(os.path.join(save_dir, 'hyperplane_example.png'), dpi=300, bbox_inches='tight')
plt.show()

# 6. Comprehensive Summary Visualization
print("\n6. COMPREHENSIVE SUMMARY")
print("=" * 50)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 1. Support Vectors
ax1.scatter(X[y == -1, 0], X[y == -1, 1], c='lightblue', s=80, alpha=0.7, edgecolors='blue')
ax1.scatter(X[y == 1, 0], X[y == 1, 1], c='lightcoral', s=80, alpha=0.7, edgecolors='red')
ax1.scatter(support_vectors[:, 0], support_vectors[:, 1], c='yellow', s=150, 
           marker='s', edgecolors='black', linewidth=2)
ax1.plot(xx, yy, 'g-', linewidth=2)
ax1.set_title('1. Support Vectors\n(Points closest to decision boundary)')
ax1.grid(True, alpha=0.3)

# 2. Linear Separability
ax2.scatter(X_lin[y_lin == -1, 0], X_lin[y_lin == -1, 1], c='lightblue', s=80, alpha=0.7, edgecolors='blue')
ax2.scatter(X_lin[y_lin == 1, 0], X_lin[y_lin == 1, 1], c='lightcoral', s=80, alpha=0.7, edgecolors='red')
ax2.plot(xx_lin, yy_lin, 'g-', linewidth=2)
ax2.set_title('2. Linear Separability\n(Data can be separated by a line)')
ax2.grid(True, alpha=0.3)

# 3. Margin
ax3.scatter(X[y == -1, 0], X[y == -1, 1], c='lightblue', s=80, alpha=0.7, edgecolors='blue')
ax3.scatter(X[y == 1, 0], X[y == 1, 1], c='lightcoral', s=80, alpha=0.7, edgecolors='red')
ax3.scatter(support_vectors[:, 0], support_vectors[:, 1], c='yellow', s=150, 
           marker='s', edgecolors='black', linewidth=2)
ax3.plot(xx, yy, 'g-', linewidth=2)
ax3.plot(xx, margin_1, 'g--', alpha=0.7)
ax3.plot(xx, margin_2, 'g--', alpha=0.7)
ax3.fill_between(xx, margin_2, margin_1, alpha=0.2, color='green')
ax3.set_title('3. Maximum Margin\n(Distance between boundary and closest points)')
ax3.grid(True, alpha=0.3)

# 4. Hyperplane
ax4.contour(X1, X2, Z, levels=[0], colors='red', linewidths=2)
ax4.contourf(X1, X2, Z, levels=[-100, 0], colors=['lightblue'], alpha=0.3)
ax4.contourf(X1, X2, Z, levels=[0, 100], colors=['lightcoral'], alpha=0.3)
ax4.arrow(origin[0], origin[1], w_example[0]/np.linalg.norm(w_example)*0.5, 
          w_example[1]/np.linalg.norm(w_example)*0.5, 
          head_width=0.1, head_length=0.1, fc='green', ec='green', linewidth=2)
ax4.set_title('4. Hyperplane\n(Decision boundary in high dimensions)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'comprehensive_summary.png'), dpi=300, bbox_inches='tight')
plt.show()

print(f"\nAll visualizations saved to: {save_dir}")
print("\n=== Summary of Key Concepts ===")
print("1. Support Vectors: Points closest to decision boundary that define the margin")
print("2. Linear Separability: Data can be perfectly separated by a straight line")
print("3. Primal vs Dual: Different formulations of the same optimization problem")
print("4. Margin: Distance between decision boundary and closest data points")
print("5. Hyperplane: Decision boundary in high-dimensional space (line in 2D)")
