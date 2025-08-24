import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
from sklearn.datasets import make_classification, make_circles
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_3_Quiz_5")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("Question 5: RBF Kernel Parameter Effects")
print("=" * 50)

# Task 1: 1D dataset with points x1 = -1, x2 = 1 (different classes)
print("\n1. 1D Dataset Decision Boundaries for Different Gamma Values")
print("-" * 60)

# Define the 1D dataset
X_1d = np.array([[-1], [1]])
y_1d = np.array([-1, 1])

# Define gamma values to test
gamma_values = [0.1, 1, 10]

# Create a fine grid for plotting decision boundaries
x_plot = np.linspace(-3, 3, 1000).reshape(-1, 1)

# Create figure for 1D decision boundaries
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, gamma in enumerate(gamma_values):
    # Train SVM with RBF kernel
    svm = SVC(kernel='rbf', gamma=gamma, C=1000)  # High C to minimize regularization
    svm.fit(X_1d, y_1d)
    
    # Get decision function values
    decision_values = svm.decision_function(x_plot)
    
    # Plot decision boundary
    axes[i].plot(x_plot.flatten(), decision_values, 'b-', linewidth=2, 
                 label='Decision Function')
    axes[i].axhline(y=0, color='k', linestyle='--', alpha=0.7, 
                    label='Decision Boundary')
    
    # Plot training points
    axes[i].scatter(X_1d[y_1d == -1], [0], c='red', s=100, marker='s', 
                    label='Class -1', edgecolor='black', linewidth=1.5)
    axes[i].scatter(X_1d[y_1d == 1], [0], c='blue', s=100, marker='o', 
                    label='Class +1', edgecolor='black', linewidth=1.5)
    
    # Shade regions
    axes[i].fill_between(x_plot.flatten(), -2, 0, where=(decision_values < 0), 
                         alpha=0.3, color='red', label='Class -1 Region')
    axes[i].fill_between(x_plot.flatten(), 0, 2, where=(decision_values > 0), 
                         alpha=0.3, color='blue', label='Class +1 Region')
    
    axes[i].set_title(f'$\\gamma = {gamma}$')
    axes[i].set_xlabel('$x$')
    axes[i].set_ylabel('Decision Function Value')
    axes[i].grid(True, alpha=0.3)
    axes[i].set_ylim(-2, 2)
    axes[i].legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'rbf_1d_decision_boundaries.png'), 
            dpi=300, bbox_inches='tight')
plt.show()

# Task 2: Analyze overfitting and underfitting
print("\n2. Overfitting and Underfitting Analysis")
print("-" * 40)

print("Effect of gamma on model complexity:")
print("- Small gamma (γ → 0): Creates smooth, simple decision boundaries")
print("  → Tends to underfit (high bias, low variance)")
print("- Large gamma (γ → ∞): Creates complex, localized decision boundaries")
print("  → Tends to overfit (low bias, high variance)")

# Task 3: Limit behavior analysis
print("\n3. Limit Behavior Analysis")
print("-" * 30)

print("As γ → 0:")
print("- RBF kernel K(x,z) = exp(-γ||x-z||²) → exp(0) = 1 for all x,z")
print("- All points become equally similar")
print("- Decision boundary becomes linear (similar to linear kernel)")
print("- Model underfits")

print("\nAs γ → ∞:")
print("- RBF kernel becomes very localized")
print("- K(x,z) → 0 unless x ≈ z (very close)")
print("- Each training point creates its own 'island' of influence")
print("- Decision boundary becomes very complex and wiggly")
print("- Model overfits")

# Task 4: Design synthetic 2D dataset where small gamma performs better
print("\n4. Synthetic 2D Dataset: Small vs Large Gamma Performance")
print("-" * 55)

# Create a dataset with a simple linear separation
np.random.seed(42)
X_linear, y_linear = make_classification(n_samples=100, n_features=2, 
                                        n_redundant=0, n_informative=2,
                                        n_clusters_per_class=1, 
                                        class_sep=1.5, random_state=42)

# Create a dataset with complex non-linear patterns
X_circles, y_circles = make_circles(n_samples=100, noise=0.1, factor=0.3, 
                                   random_state=42)

# Test different gamma values on both datasets
gamma_test = [0.01, 0.1, 1, 10, 100]

fig, axes = plt.subplots(2, 5, figsize=(20, 8))

# Plot linear dataset with different gammas
for i, gamma in enumerate(gamma_test):
    svm = SVC(kernel='rbf', gamma=gamma, C=1)
    svm.fit(X_linear, y_linear)
    
    # Create mesh for decision boundary
    h = 0.02
    x_min, x_max = X_linear[:, 0].min() - 1, X_linear[:, 0].max() + 1
    y_min, y_max = X_linear[:, 1].min() - 1, X_linear[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axes[0, i].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    scatter = axes[0, i].scatter(X_linear[:, 0], X_linear[:, 1], c=y_linear, 
                                cmap=plt.cm.RdYlBu, edgecolors='black')
    axes[0, i].set_title(f'Linear Data: $\\gamma = {gamma}$')
    axes[0, i].set_xlabel('$x_1$')
    axes[0, i].set_ylabel('$x_2$')

# Plot circles dataset with different gammas
for i, gamma in enumerate(gamma_test):
    svm = SVC(kernel='rbf', gamma=gamma, C=1)
    svm.fit(X_circles, y_circles)
    
    # Create mesh for decision boundary
    h = 0.02
    x_min, x_max = X_circles[:, 0].min() - 1, X_circles[:, 0].max() + 1
    y_min, y_max = X_circles[:, 1].min() - 1, X_circles[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axes[1, i].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    scatter = axes[1, i].scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, 
                                cmap=plt.cm.RdYlBu, edgecolors='black')
    axes[1, i].set_title(f'Circles Data: $\\gamma = {gamma}$')
    axes[1, i].set_xlabel('$x_1$')
    axes[1, i].set_ylabel('$x_2$')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'gamma_comparison_datasets.png'), 
            dpi=300, bbox_inches='tight')
plt.show()

# Task 5: Calculate effective "width" of influence
print("\n5. Effective Width of Influence Analysis")
print("-" * 40)

# The RBF kernel is K(x,z) = exp(-γ||x-z||²)
# The kernel value drops to 1/e ≈ 0.368 when γ||x-z||² = 1
# This gives us ||x-z|| = 1/√γ as the characteristic length scale

print("RBF Kernel: K(x,z) = exp(-γ||x-z||²)")
print("Characteristic width (distance where K drops to 1/e):")

for gamma in [0.1, 1, 10]:
    width = 1 / np.sqrt(gamma)
    print(f"γ = {gamma:4.1f} → width = 1/√γ = {width:.3f}")

# Visualize kernel influence for different gamma values
x_center = 0
x_range = np.linspace(-3, 3, 1000)
distances = np.abs(x_range - x_center)

plt.figure(figsize=(12, 8))

for gamma in [0.1, 1, 10]:
    kernel_values = np.exp(-gamma * distances**2)
    width = 1 / np.sqrt(gamma)
    
    plt.plot(x_range, kernel_values, linewidth=2, 
             label=f'$\\gamma = {gamma}$ (width = {width:.2f})')
    
    # Mark the characteristic width
    plt.axvline(x=width, color=plt.gca().lines[-1].get_color(), 
                linestyle='--', alpha=0.7)
    plt.axvline(x=-width, color=plt.gca().lines[-1].get_color(), 
                linestyle='--', alpha=0.7)

plt.axhline(y=1/np.e, color='black', linestyle=':', alpha=0.7, 
            label='$1/e \\approx 0.368$')
plt.xlabel('Distance from Center')
plt.ylabel('Kernel Value')
plt.title('RBF Kernel Influence vs Distance for Different $\\gamma$ Values')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.1)
plt.savefig(os.path.join(save_dir, 'rbf_influence_width.png'), 
            dpi=300, bbox_inches='tight')
plt.show()

print(f"\nAll plots saved to: {save_dir}")
print("\nSummary:")
print("- Small γ: Wide influence, smooth boundaries, may underfit")
print("- Large γ: Narrow influence, complex boundaries, may overfit")
print("- Optimal γ depends on data complexity and noise level")
