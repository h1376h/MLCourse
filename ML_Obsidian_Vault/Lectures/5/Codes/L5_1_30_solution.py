import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as patches

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_1_Quiz_30")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("=== SVM Concept Demonstration for Question 30 ===\n")

# 1. Demonstrate that NOT all training points are support vectors
print("1. Support Vectors vs All Training Points")
print("=" * 50)

# Generate a simple linearly separable dataset
np.random.seed(42)
X, y = make_blobs(n_samples=20, centers=2, cluster_std=1.0, random_state=42)
y = 2 * y - 1  # Convert to {-1, 1}

# Train SVM
svm = SVC(kernel='linear', C=1000)  # High C for hard margin
svm.fit(X, y)

# Get support vectors
support_vectors = svm.support_vectors_
support_vector_indices = svm.support_

print(f"Total training points: {len(X)}")
print(f"Number of support vectors: {len(support_vectors)}")
print(f"Support vector indices: {support_vector_indices}")
print(f"Percentage of support vectors: {len(support_vectors)/len(X)*100:.1f}%")

# Visualize
plt.figure(figsize=(12, 8))

# Plot all training points
plt.scatter(X[:, 0], X[:, 1], c=['red' if label == -1 else 'blue' for label in y], 
           s=100, alpha=0.6, label='Training Points')

# Highlight support vectors
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
           c=['darkred' if y[i] == -1 else 'darkblue' for i in support_vector_indices],
           s=200, marker='o', edgecolors='black', linewidth=2, 
           label='Support Vectors')

# Plot decision boundary
w = svm.coef_[0]
b = svm.intercept_[0]
slope = -w[0] / w[1]
xx = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
yy = slope * xx - b / w[1]

plt.plot(xx, yy, 'k-', linewidth=2, label='Decision Boundary')

# Plot margin boundaries
margin = 1 / np.sqrt(np.sum(w**2))
yy_upper = yy + margin * np.sqrt(1 + slope**2)
yy_lower = yy - margin * np.sqrt(1 + slope**2)

plt.plot(xx, yy_upper, 'k--', alpha=0.5, label='Margin Boundary')
plt.plot(xx, yy_lower, 'k--', alpha=0.5)

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Support Vectors vs All Training Points')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

# Add text box with statistics
textstr = f'Total Points: {len(X)}\nSupport Vectors: {len(support_vectors)}\nPercentage: {len(support_vectors)/len(X)*100:.1f}%'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=props)

plt.savefig(os.path.join(save_dir, 'support_vectors_vs_all_points.png'), dpi=300, bbox_inches='tight')
plt.show()

# 2. Demonstrate uniqueness of maximum margin hyperplane
print("\n2. Uniqueness of Maximum Margin Hyperplane")
print("=" * 50)

# Create a dataset with clear maximum margin
np.random.seed(123)
X_unique, y_unique = make_blobs(n_samples=10, centers=2, cluster_std=0.5, random_state=123)
y_unique = 2 * y_unique - 1

# Train multiple SVMs with different random states
svm1 = SVC(kernel='linear', C=1000, random_state=1)
svm2 = SVC(kernel='linear', C=1000, random_state=2)
svm3 = SVC(kernel='linear', C=1000, random_state=3)

svm1.fit(X_unique, y_unique)
svm2.fit(X_unique, y_unique)
svm3.fit(X_unique, y_unique)

# Check if all give same solution
w1, b1 = svm1.coef_[0], svm1.intercept_[0]
w2, b2 = svm2.coef_[0], svm2.intercept_[0]
w3, b3 = svm3.coef_[0], svm3.intercept_[0]

print(f"Solution 1: w = {w1}, b = {b1:.4f}")
print(f"Solution 2: w = {w2}, b = {b2:.4f}")
print(f"Solution 3: w = {w3}, b = {b3:.4f}")

# Check if solutions are identical (within numerical precision)
tolerance = 1e-10
identical_12 = np.allclose(w1, w2, atol=tolerance) and np.allclose(b1, b2, atol=tolerance)
identical_13 = np.allclose(w1, w3, atol=tolerance) and np.allclose(b1, b3, atol=tolerance)
identical_23 = np.allclose(w2, w3, atol=tolerance) and np.allclose(b2, b3, atol=tolerance)

print(f"Solutions 1 and 2 identical: {identical_12}")
print(f"Solutions 1 and 3 identical: {identical_13}")
print(f"Solutions 2 and 3 identical: {identical_23}")

# Visualize
plt.figure(figsize=(12, 8))

# Plot data points
plt.scatter(X_unique[:, 0], X_unique[:, 1], 
           c=['red' if label == -1 else 'blue' for label in y_unique], 
           s=150, alpha=0.7, label='Training Points')

# Plot decision boundaries (should be identical)
xx = np.linspace(X_unique[:, 0].min() - 0.5, X_unique[:, 0].max() + 0.5, 100)

for i, (svm, color, label) in enumerate([(svm1, 'green', 'SVM 1'), 
                                        (svm2, 'orange', 'SVM 2'), 
                                        (svm3, 'purple', 'SVM 3')]):
    w, b = svm.coef_[0], svm.intercept_[0]
    slope = -w[0] / w[1]
    yy = slope * xx - b / w[1]
    plt.plot(xx, yy, color=color, linewidth=2, label=f'{label} (Decision Boundary)')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Uniqueness of Maximum Margin Hyperplane')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

# Add text box
textstr = f'All solutions identical: {identical_12 and identical_13 and identical_23}\nMaximum margin hyperplane is unique'
props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=props)

plt.savefig(os.path.join(save_dir, 'uniqueness_maximum_margin.png'), dpi=300, bbox_inches='tight')
plt.show()

# 3. Demonstrate margin vs generalization relationship
print("\n3. Margin vs Generalization")
print("=" * 50)

# Create datasets with different margins
np.random.seed(456)

# Dataset 1: Large margin (well-separated)
X1, y1 = make_blobs(n_samples=20, centers=2, cluster_std=0.3, random_state=456)
y1 = 2 * y1 - 1

# Dataset 2: Small margin (closer classes)
X2, y2 = make_blobs(n_samples=20, centers=2, cluster_std=1.5, random_state=456)
y2 = 2 * y2 - 1

# Train SVMs
svm_large_margin = SVC(kernel='linear', C=1000)
svm_small_margin = SVC(kernel='linear', C=1000)

svm_large_margin.fit(X1, y1)
svm_small_margin.fit(X2, y2)

# Calculate margins
w1, b1 = svm_large_margin.coef_[0], svm_large_margin.intercept_[0]
w2, b2 = svm_small_margin.coef_[0], svm_small_margin.intercept_[0]

margin1 = 1 / np.sqrt(np.sum(w1**2))
margin2 = 1 / np.sqrt(np.sum(w2**2))

print(f"Large margin dataset - Margin: {margin1:.4f}")
print(f"Small margin dataset - Margin: {margin2:.4f}")
print(f"Margin ratio (large/small): {margin1/margin2:.2f}")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Large margin
ax1.scatter(X1[:, 0], X1[:, 1], c=['red' if label == -1 else 'blue' for label in y1], 
           s=100, alpha=0.7)
w, b = svm_large_margin.coef_[0], svm_large_margin.intercept_[0]
slope = -w[0] / w[1]
xx = np.linspace(X1[:, 0].min() - 0.5, X1[:, 0].max() + 0.5, 100)
yy = slope * xx - b / w[1]
ax1.plot(xx, yy, 'k-', linewidth=2, label='Decision Boundary')

# Plot margin
yy_upper = yy + margin1 * np.sqrt(1 + slope**2)
yy_lower = yy - margin1 * np.sqrt(1 + slope**2)
ax1.plot(xx, yy_upper, 'k--', alpha=0.7, label='Margin')
ax1.plot(xx, yy_lower, 'k--', alpha=0.7)
ax1.fill_between(xx, yy_lower, yy_upper, alpha=0.2, color='gray')

ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.set_title(f'Large Margin (Margin = {margin1:.3f})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Small margin
ax2.scatter(X2[:, 0], X2[:, 1], c=['red' if label == -1 else 'blue' for label in y2], 
           s=100, alpha=0.7)
w, b = svm_small_margin.coef_[0], svm_small_margin.intercept_[0]
slope = -w[0] / w[1]
xx = np.linspace(X2[:, 0].min() - 1, X2[:, 0].max() + 1, 100)
yy = slope * xx - b / w[1]
ax2.plot(xx, yy, 'k-', linewidth=2, label='Decision Boundary')

# Plot margin
yy_upper = yy + margin2 * np.sqrt(1 + slope**2)
yy_lower = yy - margin2 * np.sqrt(1 + slope**2)
ax2.plot(xx, yy_upper, 'k--', alpha=0.7, label='Margin')
ax2.plot(xx, yy_lower, 'k--', alpha=0.7)
ax2.fill_between(xx, yy_lower, yy_upper, alpha=0.2, color='gray')

ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.set_title(f'Small Margin (Margin = {margin2:.3f})')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'margin_vs_generalization.png'), dpi=300, bbox_inches='tight')
plt.show()

# 4. Demonstrate effect of removing non-support vectors
print("\n4. Effect of Removing Non-Support Vectors")
print("=" * 50)

# Use the first dataset
X_original = X.copy()
y_original = y.copy()

# Get support vector indices
sv_indices = svm.support_
non_sv_indices = [i for i in range(len(X)) if i not in sv_indices]

print(f"Original dataset size: {len(X_original)}")
print(f"Support vector indices: {sv_indices}")
print(f"Non-support vector indices: {non_sv_indices}")

# Remove a non-support vector
if len(non_sv_indices) > 0:
    remove_idx = non_sv_indices[0]
    X_reduced = np.delete(X_original, remove_idx, axis=0)
    y_reduced = np.delete(y_original, remove_idx, axis=0)
    
    print(f"Removing point {remove_idx}: {X_original[remove_idx]} (label: {y_original[remove_idx]})")
    print(f"Reduced dataset size: {len(X_reduced)}")
    
    # Train SVM on reduced dataset
    svm_reduced = SVC(kernel='linear', C=1000)
    svm_reduced.fit(X_reduced, y_reduced)
    
    # Compare solutions
    w_orig, b_orig = svm.coef_[0], svm.intercept_[0]
    w_reduced, b_reduced = svm_reduced.coef_[0], svm_reduced.intercept_[0]
    
    print(f"Original solution: w = {w_orig}, b = {b_orig:.4f}")
    print(f"Reduced solution: w = {w_reduced}, b = {b_reduced:.4f}")
    
    # Check if solutions are identical
    identical = np.allclose(w_orig, w_reduced, atol=1e-10) and np.allclose(b_orig, b_reduced, atol=1e-10)
    print(f"Solutions identical: {identical}")
    
    # Visualize
    plt.figure(figsize=(12, 8))
    
    # Plot original data
    plt.scatter(X_original[:, 0], X_original[:, 1], 
               c=['red' if label == -1 else 'blue' for label in y_original], 
               s=100, alpha=0.6, label='Original Points')
    
    # Highlight removed point
    plt.scatter(X_original[remove_idx, 0], X_original[remove_idx, 1], 
               c='yellow', s=200, marker='x', linewidth=3, 
               label=f'Removed Point {remove_idx}')
    
    # Plot original decision boundary
    xx = np.linspace(X_original[:, 0].min() - 1, X_original[:, 0].max() + 1, 100)
    slope_orig = -w_orig[0] / w_orig[1]
    yy_orig = slope_orig * xx - b_orig / w_orig[1]
    plt.plot(xx, yy_orig, 'g-', linewidth=2, label='Original Decision Boundary')
    
    # Plot reduced decision boundary
    slope_reduced = -w_reduced[0] / w_reduced[1]
    yy_reduced = slope_reduced * xx - b_reduced / w_reduced[1]
    plt.plot(xx, yy_reduced, 'r--', linewidth=2, label='Reduced Decision Boundary')
    
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Effect of Removing Non-Support Vector')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Add text box
    textstr = f'Removed point: {remove_idx}\nSolutions identical: {identical}\nDecision boundary unchanged'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=props)
    
    plt.savefig(os.path.join(save_dir, 'removing_non_support_vector.png'), dpi=300, bbox_inches='tight')
    plt.show()

# 5. Demonstrate why we minimize ||w||^2 instead of ||w||
print("\n5. Why Minimize ||w||^2 Instead of ||w||")
print("=" * 50)

# Create a simple 2D example
np.random.seed(789)
X_simple, y_simple = make_blobs(n_samples=8, centers=2, cluster_std=0.8, random_state=789)
y_simple = 2 * y_simple - 1

# Train SVM
svm_simple = SVC(kernel='linear', C=1000)
svm_simple.fit(X_simple, y_simple)

w, b = svm_simple.coef_[0], svm_simple.intercept_[0]
margin = 1 / np.sqrt(np.sum(w**2))

print(f"Optimal weight vector: w = {w}")
print(f"Optimal bias: b = {b:.4f}")
print(f"||w|| = {np.sqrt(np.sum(w**2)):.4f}")
print(f"||w||^2 = {np.sum(w**2):.4f}")
print(f"Margin = {margin:.4f}")

# Demonstrate scaling invariance
print(f"\nScaling demonstration:")
print(f"Original ||w||^2 = {np.sum(w**2):.4f}")
print(f"Scaled ||w||^2 = {np.sum((2*w)**2):.4f} (2x scaling)")
print(f"Ratio = {np.sum((2*w)**2)/np.sum(w**2):.1f}")

# Visualize
plt.figure(figsize=(12, 8))

# Plot data points
plt.scatter(X_simple[:, 0], X_simple[:, 1], 
           c=['red' if label == -1 else 'blue' for label in y_simple], 
           s=150, alpha=0.7, label='Training Points')

# Plot optimal decision boundary
xx = np.linspace(X_simple[:, 0].min() - 1, X_simple[:, 0].max() + 1, 100)
slope = -w[0] / w[1]
yy = slope * xx - b / w[1]
plt.plot(xx, yy, 'k-', linewidth=3, label='Optimal Decision Boundary')

# Plot margin boundaries
yy_upper = yy + margin * np.sqrt(1 + slope**2)
yy_lower = yy - margin * np.sqrt(1 + slope**2)
plt.plot(xx, yy_upper, 'k--', alpha=0.7, label='Margin Boundaries')
plt.plot(xx, yy_lower, 'k--', alpha=0.7)
plt.fill_between(xx, yy_lower, yy_upper, alpha=0.2, color='gray')

# Plot weight vector
center_x = (X_simple[:, 0].min() + X_simple[:, 0].max()) / 2
center_y = (X_simple[:, 1].min() + X_simple[:, 1].max()) / 2
plt.arrow(center_x, center_y, w[0], w[1], head_width=0.1, head_length=0.1, 
          fc='green', ec='green', linewidth=2, label='Weight Vector w')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Why Minimize $||\\mathbf{w}||^2$ Instead of $||\\mathbf{w}||$')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

# Add text box with explanation
textstr = """Why $||\\mathbf{w}||^2$ instead of $||\\mathbf{w}||$:

1. Differentiability: $||\\mathbf{w}||^2$ is differentiable everywhere
2. Convexity: $||\\mathbf{w}||^2$ is strictly convex
3. Optimization: Easier to optimize with gradient methods
4. Scaling: $||\\mathbf{w}||^2$ scales quadratically with weight scaling
5. Margin: Maximizing margin = minimizing $||\\mathbf{w}||^2$"""
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
plt.text(0.02, 0.02, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='bottom', bbox=props)

plt.savefig(os.path.join(save_dir, 'why_minimize_w_squared.png'), dpi=300, bbox_inches='tight')
plt.show()

print(f"\nAll visualizations saved to: {save_dir}")
print("\n=== Summary of Key Findings ===")
print("1. Not all training points are support vectors")
print("2. Maximum margin hyperplane is unique (for linearly separable data)")
print("3. Larger margins generally improve generalization, but not always")
print("4. Removing non-support vectors doesn't change the decision boundary")
print("5. We minimize ||w||^2 for better optimization properties")
