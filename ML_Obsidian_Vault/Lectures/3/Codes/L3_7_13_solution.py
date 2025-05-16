import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, Polygon

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_7_Quiz_13")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Define the weight vector from the problem
w = np.array([0.5, -1.5, 2.0])

# Define lambda values
lambda_1 = 2
lambda_2 = 4  # Doubled lambda

# Calculate L1 and L2 norms for the given weight vector
l1_norm = np.sum(np.abs(w))
l2_norm = np.sqrt(np.sum(w**2))
l2_norm_squared = np.sum(w**2)

# Calculate the penalty terms for both models with lambda = 2
penalty_A_lambda1 = lambda_1 * l2_norm_squared
penalty_B_lambda1 = lambda_1 * l1_norm

# Calculate the penalty terms for both models with lambda = 4
penalty_A_lambda2 = lambda_2 * l2_norm_squared
penalty_B_lambda2 = lambda_2 * l1_norm

# Print the results
print("\nRegularization and Penalty Terms Analysis")
print("=========================================")
print(f"Weight vector w = {w}")
print(f"L1 norm of w: ||w||₁ = |{w[0]}| + |{w[1]}| + |{w[2]}| = {l1_norm:.2f}")
print(f"L2 norm of w: ||w||₂ = √({w[0]}² + {w[1]}² + {w[2]}²) = {l2_norm:.2f}")
print(f"L2 norm squared: ||w||₂² = {w[0]}² + {w[1]}² + {w[2]}² = {l2_norm_squared:.2f}")

print("\nQuestion 1: Calculate penalty terms with λ = 2")
print(f"Model A (L2 penalty): λ||w||₂² = {lambda_1} × {l2_norm_squared:.2f} = {penalty_A_lambda1:.2f}")
print(f"Model B (L1 penalty): λ||w||₁ = {lambda_1} × {l1_norm:.2f} = {penalty_B_lambda1:.2f}")

print("\nQuestion 4: Calculate penalty terms with λ = 4")
print(f"Model A (L2 penalty): λ||w||₂² = {lambda_2} × {l2_norm_squared:.2f} = {penalty_A_lambda2:.2f}")
print(f"Model B (L1 penalty): λ||w||₁ = {lambda_2} × {l1_norm:.2f} = {penalty_B_lambda2:.2f}")
print(f"Effect: Both penalty terms doubled when λ doubled.")

# Visual explanation of question 2: L1 vs L2 regularization in 2D
plt.figure(figsize=(10, 8))

# Function to generate points on L1 and L2 norm boundaries
theta = np.linspace(0, 2*np.pi, 100)
l2_x = np.cos(theta)
l2_y = np.sin(theta)

# Plot L2 norm unit circle
plt.plot(l2_x, l2_y, 'b-', linewidth=2, label='L2 norm ||w||₂ = 1')
# Plot L1 norm diamond
l1_points = np.array([
    [1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]
])
plt.plot(l1_points[:, 0], l1_points[:, 1], 'r-', linewidth=2, label='L1 norm ||w||₁ = 1')

# Add contours of the loss function (assumed to be circular for illustration)
for r in [0.5, 1.0, 1.5, 2.0]:
    circle = plt.Circle((0, 0), r, color='green', fill=False, linestyle='--', alpha=0.5)
    plt.gca().add_patch(circle)

# Show where L1 and L2 norms typically intersect with loss function
plt.plot([0], [1.9], 'ro', markersize=10, label='L1 solution (sparse)')
plt.plot([1.2], [1.5], 'bo', markersize=10, label='L2 solution')

# Draw coordinate axes
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# Annotate axes
plt.text(2.1, 0, 'w₁', fontsize=12)
plt.text(0, 2.1, 'w₂', fontsize=12)

plt.grid(True)
plt.axis('equal')
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.title('Geometric Interpretation of L1 vs L2 Regularization', fontsize=14)
plt.legend(loc='best')
plt.savefig(os.path.join(save_dir, "l1_vs_l2_geometry.png"), dpi=300, bbox_inches='tight')
plt.close()

# Visualization of penalty functions
w_range = np.linspace(-2, 2, 200)
l1_penalties = np.abs(w_range)
l2_penalties = w_range**2

plt.figure(figsize=(10, 6))
plt.plot(w_range, l1_penalties, 'r-', linewidth=2, label='L1 penalty: |w|')
plt.plot(w_range, l2_penalties, 'b-', linewidth=2, label='L2 penalty: w²')

plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

plt.grid(True)
plt.xlim(-2.1, 2.1)
plt.ylim(-0.1, 4.1)
plt.title('Comparison of L1 and L2 Penalty Functions', fontsize=14)
plt.xlabel('Weight value w', fontsize=12)
plt.ylabel('Penalty', fontsize=12)
plt.legend(loc='best')
plt.savefig(os.path.join(save_dir, "l1_vs_l2_penalties.png"), dpi=300, bbox_inches='tight')
plt.close()

# 3D Visualization of L1 and L2 penalties in 2D weight space
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

# L2 penalty surface
Z_l2 = X**2 + Y**2

# L1 penalty surface
Z_l1 = np.abs(X) + np.abs(Y)

# Create 3D plots
fig = plt.figure(figsize=(15, 7))

# L2 penalty surface
ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(X, Y, Z_l2, cmap='viridis', alpha=0.8)
ax1.set_title('L2 Penalty Surface: ||w||₂²', fontsize=14)
ax1.set_xlabel('w₁', fontsize=12)
ax1.set_ylabel('w₂', fontsize=12)
ax1.set_zlabel('Penalty', fontsize=12)

# L1 penalty surface
ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(X, Y, Z_l1, cmap='plasma', alpha=0.8)
ax2.set_title('L1 Penalty Surface: ||w||₁', fontsize=14)
ax2.set_xlabel('w₁', fontsize=12)
ax2.set_ylabel('w₂', fontsize=12)
ax2.set_zlabel('Penalty', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "l1_vs_l2_3d_surfaces.png"), dpi=300, bbox_inches='tight')
plt.close()

# Simulate regularization path to show sparsity
# Generate some synthetic data
np.random.seed(42)
n_features = 10
n_samples = 100
X = np.random.randn(n_samples, n_features)
true_w = np.array([1.5, 0.8, 0, 0, 2.0, 0, 0, -1.0, 0, 0.5])
y = X @ true_w + np.random.randn(n_samples) * 0.5

# Define a range of lambda values
lambda_values = np.logspace(-2, 3, 20)
l1_weights = np.zeros((len(lambda_values), n_features))
l2_weights = np.zeros((len(lambda_values), n_features))

# Simplified regularization path calculation
# In reality, this would use proper solvers for L1 and L2 regularization
X_centered = X - X.mean(axis=0)
X_normalized = X_centered / np.linalg.norm(X_centered, axis=0)
XTX = X_normalized.T @ X_normalized
XTy = X_normalized.T @ y

for i, lam in enumerate(lambda_values):
    # L2 solution (closed form for Ridge)
    l2_weights[i] = np.linalg.solve(XTX + lam * np.eye(n_features), XTy)
    
    # L1 solution (simplified approximation - not accurate but illustrative)
    # In practice, would use proper Lasso solver
    soft_threshold = lambda z, t: np.sign(z) * np.maximum(np.abs(z) - t, 0)
    l1_w = np.linalg.solve(XTX + 0.001 * np.eye(n_features), XTy)  # starting point
    for _ in range(100):  # Coordinate descent iterations
        for j in range(n_features):
            r = y - X_normalized @ l1_w + X_normalized[:, j] * l1_w[j]
            z = X_normalized[:, j] @ r
            l1_w[j] = soft_threshold(z, lam / 2) 
    l1_weights[i] = l1_w

# Plot the regularization paths
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
for j in range(n_features):
    plt.semilogx(lambda_values, l1_weights[:, j], '-', linewidth=1.5, label=f'w{j+1}')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel('Regularization parameter λ', fontsize=12)
plt.ylabel('Coefficient value', fontsize=12)
plt.title('L1 Regularization Path (Lasso)', fontsize=14)
plt.grid(True)

plt.subplot(1, 2, 2)
for j in range(n_features):
    plt.semilogx(lambda_values, l2_weights[:, j], '-', linewidth=1.5, label=f'w{j+1}')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel('Regularization parameter λ', fontsize=12)
plt.ylabel('Coefficient value', fontsize=12)
plt.title('L2 Regularization Path (Ridge)', fontsize=14)
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "l1_vs_l2_regularization_paths.png"), dpi=300, bbox_inches='tight')
plt.close()

# Plot how the number of non-zero coefficients changes with lambda for L1 and L2
l1_nonzero = np.sum(np.abs(l1_weights) > 1e-4, axis=1)
l2_nonzero = np.sum(np.abs(l2_weights) > 1e-4, axis=1)

plt.figure(figsize=(10, 6))
plt.semilogx(lambda_values, l1_nonzero, 'r-', linewidth=2, marker='o', label='L1 (Lasso)')
plt.semilogx(lambda_values, l2_nonzero, 'b-', linewidth=2, marker='s', label='L2 (Ridge)')
plt.xlabel('Regularization parameter λ', fontsize=12)
plt.ylabel('Number of non-zero coefficients', fontsize=12)
plt.title('Sparsity Effect: Non-zero Coefficients vs Regularization Strength', fontsize=14)
plt.legend(loc='best')
plt.grid(True)
plt.savefig(os.path.join(save_dir, "l1_vs_l2_sparsity.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nVisualizations saved to: {save_dir}")
print("\nQuestion 2: Which model would likely produce more zero coefficients and why?")
print("Model B (L1 regularization) would produce more zero coefficients.")
print("Reason: L1 penalty creates corners at zero, encouraging exact zeros in the solution.")
print("This is evident in the diamond shape of the L1 constraint, which intersects axes more readily.")

print("\nQuestion 3: One advantage of Model A (L2 regularization) over Model B (L1 regularization)")
print("Advantage: Model A (Ridge) has a unique, stable solution even with correlated features.")
print("Ridge regression handles multicollinearity better because it shrinks correlated features together,")
print("whereas Lasso tends to arbitrarily select one feature from a group of correlated features.")
print("Additionally, L2 regularization has a closed-form solution, making it computationally more efficient.") 