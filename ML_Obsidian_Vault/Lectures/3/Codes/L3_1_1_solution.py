import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_1_Quiz_1")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Define a 3D arrow class for better visualizations
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return np.min(zs)

# Print explanations and formulas
print("\nHat Matrix in Linear Regression - Mathematical Properties")
print("==================================================")
print("Definition: H = X(X'X)^(-1)X'")
print("\nKey Properties:")
print("1. Projection matrix: Projects y onto column space of X")
print("2. Symmetric: H = H'")
print("3. Idempotent: H² = H")
print("4. Fitted values: ŷ = Hy")
print("5. Residuals: e = y - ŷ = (I - H)y")

print("\nEigenvalue Properties:")
print("- Eigenvalues can only be 0 or 1 (due to idempotence)")
print("- Sum of eigenvalues equals rank of X")
print("- Number of eigenvalues = 1 equals rank of X")

print("\nGeometric Interpretation:")
print("- Projects y onto closest point in column space of X")
print("- Eigenvalue 1: directions preserved by projection")
print("- Eigenvalue 0: directions eliminated by projection")

# Create example data
np.random.seed(42)
n = 10
X = np.column_stack((np.ones(n), np.random.normal(0, 1, n)))
H = X @ np.linalg.inv(X.T @ X) @ X.T
y = np.random.normal(X[:, 1], 1)
y_fitted = H @ y
residuals = y - y_fitted

# Print numerical results
print("\nNumerical Results:")
print("Eigenvalues of H:", np.linalg.eigvals(H).round(10))
print("Rank of X:", np.linalg.matrix_rank(X))

# Visualization 1: 2D Projection
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 1], y, s=100, color='blue', label='Original data')
plt.scatter(X[:, 1], y_fitted, s=100, color='red', label='Projected data')
for i in range(n):
    plt.plot([X[i, 1], X[i, 1]], [y[i], y_fitted[i]], 'k--', alpha=0.5)
plt.title('Projection of y onto the Column Space of X', fontsize=14)
plt.xlabel('Predictor (X)', fontsize=12)
plt.ylabel('Response (y)', fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, "hat_matrix_2d_projection.png"), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 2: 3D Projection
X_small = X[:3, :]
H_small = X_small @ np.linalg.inv(X_small.T @ X_small) @ X_small.T
y_small = y[:3]
y_fitted_small = H_small @ y_small

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

max_extent = 1.5 * max(np.max(np.abs(y_small)), 1)
xx, yy = np.meshgrid([-max_extent, max_extent], [-max_extent, max_extent])
z_plane = np.zeros(xx.shape)

col1, col2 = X_small[:, 0], X_small[:, 1]
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        point = col1 * xx[i, j] + col2 * yy[i, j]
        z_plane[i, j] = point[2]

ax.plot_surface(xx, yy, z_plane, alpha=0.3, color='grey')
ax.scatter(y_small[0], y_small[1], y_small[2], s=200, c='blue', label='Original')
ax.scatter(y_fitted_small[0], y_fitted_small[1], y_fitted_small[2], s=200, c='red', label='Projected')

ax.plot([y_small[0], y_fitted_small[0]], [y_small[1], y_fitted_small[1]], 
        [y_small[2], y_fitted_small[2]], 'k--', alpha=0.7)
a = Arrow3D([y_fitted_small[0], y_small[0]], [y_fitted_small[1], y_small[1]], 
           [y_fitted_small[2], y_small[2]], mutation_scale=20, lw=2, 
           arrowstyle="-|>", color="green", alpha=0.7)
ax.add_artist(a)

ax.set_xlabel('Component 1', fontsize=12)
ax.set_ylabel('Component 2', fontsize=12)
ax.set_zlabel('Component 3', fontsize=12)
ax.set_title('3D Geometric Interpretation of the Hat Matrix', fontsize=14)
ax.legend(['Column Space of X', 'Original y', 'Projected ŷ = Hy'], loc='upper left', fontsize=12)

plt.savefig(os.path.join(save_dir, "hat_matrix_3d_projection.png"), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 3: Unit Circle Transformation
X_2 = np.array([[1, 0], [1, 1]])
H_2 = X_2 @ np.linalg.inv(X_2.T @ X_2) @ X_2.T
eigenvalues_2, eigenvectors_2 = np.linalg.eig(H_2)

# Image 1: Original unit circle
plt.figure(figsize=(8, 8))
theta = np.linspace(0, 2*np.pi, 100)
x, y = np.cos(theta), np.sin(theta)
circle_points = np.vstack((x, y))

plt.plot(x, y, 'b-', linewidth=2, label='Original Unit Circle')

for i in range(2):
    v = eigenvectors_2[:, i] * 1
    plt.arrow(0, 0, v[0], v[1], head_width=0.05, head_length=0.1, 
              fc='green', ec='green', label=f'λ={eigenvalues_2[i]:.1f}' if i == 0 else "")

plt.grid(True)
plt.axis('equal')
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.title('Original Unit Circle with Eigenvectors', fontsize=14)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig(os.path.join(save_dir, "hat_matrix_original_circle.png"), dpi=300, bbox_inches='tight')
plt.close()

# Image 2: Transformed unit circle
plt.figure(figsize=(8, 8))
transformed_points = H_2 @ circle_points

plt.plot(x, y, 'b-', alpha=0.5, label='Original')
plt.plot(transformed_points[0, :], transformed_points[1, :], 'r-', linewidth=2, label='Transformed')

for i in range(2):
    v = eigenvectors_2[:, i] * 1
    plt.arrow(0, 0, v[0], v[1], head_width=0.05, head_length=0.1, 
              fc='green', ec='green', label=f'λ={eigenvalues_2[i]:.1f}' if i == 0 else "")

plt.grid(True)
plt.axis('equal')
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.title('Effect of Hat Matrix on Unit Circle', fontsize=14)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig(os.path.join(save_dir, "hat_matrix_eigenvalues.png"), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 4: Eigenspaces
plt.figure(figsize=(10, 6))
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

plt.fill_between([-2, 2], [-2, 2], [2, -2], color='red', alpha=0.2)
plt.plot([-2, 2], [-2, 2], 'r-', linewidth=2, label='λ=1 space')
plt.plot([-2, 2], [2, -2], 'b-', linewidth=2, label='λ=0 space')

v1 = np.array([1.5, 0.5])
projection = np.array([1, 1])
residual = v1 - projection

plt.arrow(0, 0, v1[0], v1[1], head_width=0.1, head_length=0.1, 
          fc='purple', ec='purple', label='Original')
plt.arrow(0, 0, projection[0], projection[1], head_width=0.1, head_length=0.1, 
          fc='red', ec='red', label='Projection')
plt.arrow(projection[0], projection[1], residual[0], residual[1], 
          head_width=0.1, head_length=0.1, fc='blue', ec='blue', label='Residual')

plt.grid(True)
plt.axis('equal')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.title('Eigenspaces of the Hat Matrix', fontsize=14)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig(os.path.join(save_dir, "hat_matrix_eigenspaces.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nVisualizations saved to: {save_dir}")