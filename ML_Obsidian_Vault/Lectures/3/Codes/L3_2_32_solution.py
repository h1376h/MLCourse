import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_2_Quiz_32")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

print("Geometric Interpretation of Linear Regression in n-Dimensional Space")
print("===================================================================")

def save_figure(fig, filename):
    """Save figure to the specified directory."""
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {filepath}")
    return filepath

# Part 1: Column Space Visualization in 2D
print("\nPart 1: Understanding the Column Space of X")
print("------------------------------------------")

# Create a simple 2D example first
np.random.seed(42)
n_samples = 8  # Number of observations

# Define X with two basis vectors (columns)
x1 = np.array([1, 1, 0, 1, 0, 0, 1, 0])  # First column of X (including intercept)
x2 = np.array([0, 1, 1, 2, 1, 2, 3, 2])  # Second column of X (feature)

X = np.column_stack((x1, x2))
print(f"Design matrix X (shape {X.shape}):")
print(X)

# Define a true relationship
true_weights = np.array([2, 1.5])  # [intercept, feature coefficient]
y_true = X @ true_weights

# Add some noise to create observed y
np.random.seed(42)
noise = np.random.normal(0, 1, n_samples)
y = y_true + noise

print(f"\nResponse vector y (shape {y.shape}):")
print(y)

# Compute the least squares solution
X_pinv = np.linalg.pinv(X)
w_hat = X_pinv @ y
print(f"\nLeast squares solution w_hat: {w_hat}")

# Calculate the projection of y onto the column space
y_proj = X @ w_hat
print(f"\nProjection of y onto column space (y_hat): {y_proj}")

# Calculate the residual vector
residual = y - y_proj
print(f"\nResidual vector (e): {residual}")

# Check orthogonality of residual with columns of X
ortho_check_x1 = np.dot(residual, x1)
ortho_check_x2 = np.dot(residual, x2)
print(f"\nOrthogonality check with x1: {ortho_check_x1:.10f}")
print(f"\nOrthogonality check with x2: {ortho_check_x2:.10f}")

# Verify the least squares solution using the normal equations
normal_eq_solution = np.linalg.inv(X.T @ X) @ X.T @ y
print(f"\nSolution via normal equations: {normal_eq_solution}")

# Add detailed manual calculations after the normal equation check
print("\nPart 1.5: Detailed Step-by-Step Calculations")
print("--------------------------------------------")
print("Let's perform the detailed calculations by hand for our example:")

# Manual calculation of X^T X
X_transpose = X.T
XTX = X_transpose @ X
print("\nStep A: Calculate X^T X by hand:")
print(f"X^T = \n{X_transpose}")
print(f"X^T X = \n{XTX}")

# Manual calculation of X^T y
XTy = X_transpose @ y
print("\nStep B: Calculate X^T y by hand:")
print(f"X^T y = \n{XTy}")

# Manual calculation of (X^T X)^(-1)
XTX_inv = np.linalg.inv(XTX)
print("\nStep C: Calculate (X^T X)^(-1) by hand:")
print(f"(X^T X)^(-1) = \n{XTX_inv}")

# Manual calculation of (X^T X)^(-1) X^T y
w_hat_manual = XTX_inv @ XTy
print("\nStep D: Calculate (X^T X)^(-1) X^T y to get w_hat by hand:")
print(f"w_hat = (X^T X)^(-1) X^T y = \n{w_hat_manual}")

# Manual calculation of the projection
y_hat_manual = X @ w_hat_manual
print("\nStep E: Calculate X w_hat to get y_hat (the projection) by hand:")
print(f"y_hat = X w_hat = \n{y_hat_manual}")

# Manual calculation of the residual
e_manual = y - y_hat_manual
print("\nStep F: Calculate e = y - y_hat (the residual) by hand:")
print(f"e = y - y_hat = \n{e_manual}")

# Verify orthogonality manually
ortho_manual_1 = X_transpose[0] @ e_manual
ortho_manual_2 = X_transpose[1] @ e_manual
print("\nStep G: Verify that X^T e = 0 (orthogonality):")
print(f"x1^T e = {ortho_manual_1:.10f}")
print(f"x2^T e = {ortho_manual_2:.10f}")

# Calculate the squared norm of the residual
residual_norm_squared = np.dot(e_manual, e_manual)
print("\nStep H: Calculate the squared norm of the residual:")
print(f"||e||^2 = e^T e = {residual_norm_squared:.6f}")

# Calculate the squared distance from y to its projection
squared_distance = np.sum((y - y_hat_manual) ** 2)
print("\nStep I: Calculate the squared distance from y to its projection:")
print(f"||y - y_hat||^2 = {squared_distance:.6f}")

# Verify this is the minimum distance
print("\nStep J: Verify this is the minimum distance:")
print("Let's try a different point in the column space by slightly changing the coefficients:")
w_perturbed = w_hat_manual + np.array([0.1, -0.1])
y_perturbed = X @ w_perturbed
squared_distance_perturbed = np.sum((y - y_perturbed) ** 2)
print(f"With perturbed w = {w_perturbed}, the squared distance is {squared_distance_perturbed:.6f}")
print(f"This is larger than the original squared distance: {squared_distance:.6f} < {squared_distance_perturbed:.6f}")

print("\nVisualization: Column Space and Projection in R^n")

# Create a 3D visualization to illustrate the concept in 3D (simplifying the 8D reality)
# We'll use PCA to reduce to 3D for visualization purposes
# This is just for illustration - the geometric concepts are the same in higher dimensions

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# We'll create a simplified 3D representation just for visualization
# In reality, this is an 8-dimensional space with a 2D subspace
# Create unit vectors in 3D space to represent our simplified view
e1 = np.array([1, 0, 0])
e2 = np.array([0, 1, 0])
e3 = np.array([0, 0, 1])

# Let's say x1 and x2 span a plane in this 3D space
# These are different from the actual x1, x2 - just simplified for visualization
x1_3d = np.array([1, 0.5, 0])
x2_3d = np.array([0.5, 1, 0])

# Normalize them for better visualization
x1_3d = x1_3d / np.linalg.norm(x1_3d)
x2_3d = x2_3d / np.linalg.norm(x2_3d)

# Create a grid on the plane spanned by x1 and x2
xx, yy = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
zz = np.zeros_like(xx)

for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        # Every point on this grid is a linear combination of x1_3d and x2_3d
        point = xx[i,j] * x1_3d + yy[i,j] * x2_3d
        zz[i,j] = point[2]  # z-coordinate

# Plot the plane (representing column space)
ax.plot_surface(
    xx[0,0] * x1_3d[0] + yy * x2_3d[0], 
    xx[0,0] * x1_3d[1] + yy * x2_3d[1], 
    xx[0,0] * x1_3d[2] + yy * x2_3d[2], 
    alpha=0.2, color='blue'
)

# Create a point to represent y (outside the plane)
y_3d = np.array([0.7, 0.5, 0.8])

# Create its projection onto the plane
# The projection formula is: proj = (y·x1)x1 + (y·x2)x2
proj_coef1 = np.dot(y_3d, x1_3d)
proj_coef2 = np.dot(y_3d, x2_3d)
y_proj_3d = proj_coef1 * x1_3d + proj_coef2 * x2_3d

# Calculate the residual
residual_3d = y_3d - y_proj_3d

# Plot basis vectors
ax.quiver(0, 0, 0, x1_3d[0], x1_3d[1], x1_3d[2], color='r', label='x₁', arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, x2_3d[0], x2_3d[1], x2_3d[2], color='g', label='x₂', arrow_length_ratio=0.1)

# Plot y vector, its projection, and the residual
ax.quiver(0, 0, 0, y_3d[0], y_3d[1], y_3d[2], color='blue', label='y', arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, y_proj_3d[0], y_proj_3d[1], y_proj_3d[2], color='purple', label='ŷ (projection)', arrow_length_ratio=0.1)
ax.quiver(y_proj_3d[0], y_proj_3d[1], y_proj_3d[2], 
          residual_3d[0], residual_3d[1], residual_3d[2], 
          color='orange', label='e (residual)', arrow_length_ratio=0.1)

# Draw a dashed line from y to its projection to illustrate the shortest distance
ax.plot([y_3d[0], y_proj_3d[0]], [y_3d[1], y_proj_3d[1]], [y_3d[2], y_proj_3d[2]], 
        'k--', label='Shortest distance')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Geometric Interpretation of Linear Regression in 3D')

# Set equal aspect ratio
ax.set_box_aspect([1, 1, 1])

# Add legend
ax.legend()

file_path1 = save_figure(fig, "column_space_projection_3d.png")
plt.close(fig)

# Part 2: 2D Visualization for better understanding
print("\nPart 2: 2D Visualization of Linear Regression Geometry")
print("---------------------------------------------------")

# Create a simple 2D dataset
np.random.seed(123)
n_points = 20
X_2d = np.column_stack([np.ones(n_points), np.random.uniform(0, 10, n_points)])
true_beta = np.array([2, 0.5])
y_2d = X_2d @ true_beta + np.random.normal(0, 1, n_points)

# Fit the model
model = LinearRegression(fit_intercept=False)  # We already added the intercept column
model.fit(X_2d, y_2d)
y_pred = model.predict(X_2d)
residuals = y_2d - y_pred

print(f"Coefficients: {model.coef_}")
print(f"Mean squared residual: {np.mean(residuals**2)}")

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))

# Plot original data points
ax.scatter(X_2d[:, 1], y_2d, color='blue', label='Observed data')

# Plot the fitted line
x_range = np.array([min(X_2d[:, 1]), max(X_2d[:, 1])])
y_fit = model.intercept_ + model.coef_[1] * x_range
ax.plot(x_range, y_fit, color='red', label='Fitted line')

# Plot residuals
for i in range(n_points):
    ax.plot([X_2d[i, 1], X_2d[i, 1]], [y_2d[i], y_pred[i]], 'g--', alpha=0.5)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Linear Regression: Data, Fitted Line, and Residuals')
ax.legend()
ax.grid(True)

file_path2 = save_figure(fig, "linear_regression_2d.png")
plt.close(fig)

# Part 3: Visual proof of orthogonality
print("\nPart 3: Visual Proof of Residual Orthogonality")
print("-----------------------------------------")

# Generate a new example
np.random.seed(456)
n_samples = 100
X_ortho = np.column_stack([np.ones(n_samples), np.random.uniform(0, 10, n_samples)])
true_beta = np.array([3, 0.7])
y_ortho = X_ortho @ true_beta + np.random.normal(0, 2, n_samples)

# Fit model
model = LinearRegression(fit_intercept=False)
model.fit(X_ortho, y_ortho)
y_pred_ortho = model.predict(X_ortho)
residuals_ortho = y_ortho - y_pred_ortho

# Check orthogonality of residuals with X columns
ortho_check1 = np.dot(residuals_ortho, X_ortho[:, 0])
ortho_check2 = np.dot(residuals_ortho, X_ortho[:, 1])

print(f"Dot product of residuals with intercept column: {ortho_check1:.10f}")
print(f"Dot product of residuals with feature column: {ortho_check2:.10f}")
print("\nThis confirms that the residual vector is orthogonal to each column of X.")

# Mathematical proof
print("\nMathematical Proof of Orthogonality:")
print("The normal equations: X^T X β̂ = X^T y")
print("Rearranging: X^T X β̂ - X^T y = 0")
print("Factoring out X^T: X^T (X β̂ - y) = 0")
print("Since (X β̂ - y) = -e (the negative of the residual vector):")
print("X^T (-e) = 0")
print("Therefore: X^T e = 0")
print("This means that each column of X is orthogonal to the residual vector e.")

# Plot showing orthogonality in a simplified 2D case
fig, ax = plt.subplots(figsize=(10, 6))

# We'll use a simplified example with just a few points
n_small = 5
X_small = X_ortho[:n_small]
y_small = y_ortho[:n_small]
y_pred_small = y_pred_ortho[:n_small]
residuals_small = residuals_ortho[:n_small]

# Project data into a 2D space for visualization
# Use the first feature and the residuals as basis
feature = X_small[:, 1].reshape(-1, 1)
constant = X_small[:, 0].reshape(-1, 1)

# Normalize to unit length for visualization
feature_norm = feature / np.linalg.norm(feature)
constant_norm = constant / np.linalg.norm(constant)
residuals_norm = residuals_small / np.linalg.norm(residuals_small)

# Plot vectors
ax.quiver(0, 0, feature_norm[0, 0], 0, color='red', angles='xy', scale_units='xy', scale=1, label='Feature direction')
ax.quiver(0, 0, 0, constant_norm[0, 0], color='green', angles='xy', scale_units='xy', scale=1, label='Constant term direction')
ax.quiver(0, 0, residuals_norm[0], residuals_norm[1], color='blue', angles='xy', scale_units='xy', scale=1, label='Residual vector')

# Draw lines at 90 degrees to show orthogonality
ax.axhline(0, color='r', linestyle='--', alpha=0.3)
ax.axvline(0, color='g', linestyle='--', alpha=0.3)

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.grid(True)
ax.legend()
ax.set_title('Orthogonality of Residual Vector to Columns of X')

file_path3 = save_figure(fig, "residual_orthogonality.png")
plt.close(fig)

# Part 4: Adding a new feature visualization
print("\nPart 4: Effect of Adding a New Feature")
print("------------------------------------")

# Create data with potential for nonlinear relationship
np.random.seed(789)
n_data = 100
x_non = np.random.uniform(0, 10, n_data)
y_non = 2 + 0.5 * x_non + 0.2 * x_non**2 + np.random.normal(0, 1, n_data)

# Model 1: Linear model with just intercept and x
X1 = np.column_stack([np.ones(n_data), x_non])
beta1 = np.linalg.lstsq(X1, y_non, rcond=None)[0]
y_pred1 = X1 @ beta1
residuals1 = y_non - y_pred1
mse1 = np.mean(residuals1**2)

# Model 2: Adding x^2 as a feature
X2 = np.column_stack([np.ones(n_data), x_non, x_non**2])
beta2 = np.linalg.lstsq(X2, y_non, rcond=None)[0]
y_pred2 = X2 @ beta2
residuals2 = y_non - y_pred2
mse2 = np.mean(residuals2**2)

print(f"Model 1 (intercept + x): MSE = {mse1:.4f}, Coefficients = {beta1}")
print(f"Model 2 (intercept + x + x²): MSE = {mse2:.4f}, Coefficients = {beta2}")
print(f"Reduction in MSE: {mse1 - mse2:.4f} ({(1 - mse2/mse1) * 100:.2f}%)")

# Visualization of the two models
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# First row: Data and fitted curves
axes[0, 0].scatter(x_non, y_non, color='blue', alpha=0.6, label='Data')
x_sorted = np.sort(x_non)
axes[0, 0].plot(x_sorted, beta1[0] + beta1[1] * x_sorted, 'r-', label='Linear model')
axes[0, 0].plot(x_sorted, beta2[0] + beta2[1] * x_sorted + beta2[2] * x_sorted**2, 'g-', label='Quadratic model')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('y')
axes[0, 0].set_title('Data and Fitted Models')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Second plot: Residuals for both models
axes[0, 1].scatter(x_non, residuals1, color='red', alpha=0.6, label='Linear model residuals')
axes[0, 1].scatter(x_non, residuals2, color='green', alpha=0.6, label='Quadratic model residuals')
axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('Residual')
axes[0, 1].set_title('Residuals Comparison')
axes[0, 1].legend()
axes[0, 1].grid(True)

# 3D Visualization to show the expanded column space
from mpl_toolkits.mplot3d import Axes3D

# Create a 3D representation of the column spaces
ax3d1 = fig.add_subplot(223, projection='3d')
ax3d2 = fig.add_subplot(224, projection='3d')

# Simplify to 3D for visualization by taking just 3 data points
indices = [10, 30, 50]  # Just pick some indices
X_simple = X1[indices]
y_simple = y_non[indices]

# Basis vectors for the first model (normalized for visualization)
x0 = X_simple[:, 0] / np.linalg.norm(X_simple[:, 0])
x1 = X_simple[:, 1] / np.linalg.norm(X_simple[:, 1])

# Plot the plane for the first model
u = np.linspace(-1, 1, 10)
v = np.linspace(-1, 1, 10)
U, V = np.meshgrid(u, v)
X_plane = np.outer(x0, U.flatten()) + np.outer(x1, V.flatten())
X_plane = X_plane.reshape(3, U.shape[0], V.shape[0])

ax3d1.plot_surface(X_plane[0], X_plane[1], X_plane[2], alpha=0.2, color='blue')
ax3d1.set_title('Column Space: Model 1 (2D)')
ax3d1.set_xlabel('Dimension 1')
ax3d1.set_ylabel('Dimension 2')
ax3d1.set_zlabel('Dimension 3')

# For the second model, add an additional basis vector (x^2)
x2 = X2[indices, 2] / np.linalg.norm(X2[indices, 2])

# To visualize a 3D subspace in 3D, we would need a 4D plot, which is not possible
# So we'll show part of the expanded subspace by plots points from the plane and the new direction

# Plot the points from the first model's plane (blue)
ax3d2.plot_surface(X_plane[0], X_plane[1], X_plane[2], alpha=0.2, color='blue')

# Plot the new dimension/direction that comes from adding x^2
for i in np.linspace(-1, 1, 5):
    for j in np.linspace(-1, 1, 5):
        point_on_plane = i * x0 + j * x1
        # Plot lines going in the x2 direction from points on the plane
        ax3d2.plot([point_on_plane[0], point_on_plane[0] + x2[0]],
                  [point_on_plane[1], point_on_plane[1] + x2[1]],
                  [point_on_plane[2], point_on_plane[2] + x2[2]],
                  'g-', alpha=0.3)

# Plot the expanded space representation
ax3d2.set_title('Column Space: Model 2 (3D)')
ax3d2.set_xlabel('Dimension 1')
ax3d2.set_ylabel('Dimension 2')
ax3d2.set_zlabel('Dimension 3')

plt.tight_layout()
file_path4 = save_figure(fig, "adding_feature_visualization.png")
plt.close(fig)

# Create a new 3D visualization specifically showing the projection improvement
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Create a point to represent y in 3D
y_point = np.array([1, 1, 1.5])

# Create two subspaces
# First, a line (1D subspace) representing a simple model
x1_dir = np.array([1, 0, 0])
x2_dir = np.array([0, 1, 0.5])  # Direction that will be added

# Points along the first subspace (line)
line_points = np.array([t * x1_dir for t in np.linspace(-2, 2, 20)])

# Create a grid for the 2D subspace (plane) created by adding the second direction
# Fix the broadcasting issue
xx, yy = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
plane_points = np.zeros((10, 10, 3))
for i in range(10):
    for j in range(10):
        plane_points[i, j] = xx[i, j] * x1_dir + yy[i, j] * x2_dir

# Calculate projections
proj1_coef = np.dot(y_point, x1_dir) / np.dot(x1_dir, x1_dir)
proj1 = proj1_coef * x1_dir

# For 2D case, we need to solve a 2x2 system
A = np.column_stack([x1_dir, x2_dir])
coefs, _, _, _ = np.linalg.lstsq(A, y_point, rcond=None)
proj2 = coefs[0] * x1_dir + coefs[1] * x2_dir

# Calculate residuals
residual1 = y_point - proj1
residual2 = y_point - proj2

# Plot the point y
ax.scatter([y_point[0]], [y_point[1]], [y_point[2]], color='blue', s=100, label='y (data point)')

# Plot the subspaces
ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], 'r-', linewidth=2, label='1D subspace (Model 1)')
ax.plot_surface(plane_points[:, :, 0], plane_points[:, :, 1], plane_points[:, :, 2], 
               color='green', alpha=0.2, label='2D subspace (Model 2)')

# Plot the projections
ax.scatter([proj1[0]], [proj1[1]], [proj1[2]], color='red', s=80, label='ŷ (Model 1)')
ax.scatter([proj2[0]], [proj2[1]], [proj2[2]], color='green', s=80, label='ŷ (Model 2)')

# Plot the residuals
ax.plot([y_point[0], proj1[0]], [y_point[1], proj1[1]], [y_point[2], proj1[2]], 
       'r--', linewidth=2, label='Residual (Model 1)')
ax.plot([y_point[0], proj2[0]], [y_point[1], proj2[1]], [y_point[2], proj2[2]], 
       'g--', linewidth=2, label='Residual (Model 2)')

# Beautify
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.view_init(elev=20, azim=30)
ax.set_title('Improved Projection with Additional Feature')

# Legend outside the plot
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

file_path5 = save_figure(fig, "projection_improvement_3d.png")
plt.close(fig)

# Final summary
print("\nFinal Summary")
print("============")
print("1. Column Space Interpretation: The column space of X represents all possible predictions that can be made")
print("   by linear combinations of the feature vectors.")
print("2. Least Squares Solution: Finds the point in the column space closest to y, minimizing the Euclidean distance.")
print("3. Orthogonality of Residuals: The residual vector is orthogonal to each column of X, confirmed both")
print("   geometrically and mathematically.")
print("4. Geometric Perspective: Explains 'fitting' as finding the optimal projection of y onto the column space.")
print("5. Adding Features: Expands the column space, potentially allowing a closer approximation to y by providing")
print("   more directions for projection.")
print("6. Step-by-Step Calculations: We performed detailed calculations to demonstrate the mathematical foundation")
print("   of these geometric interpretations.")

# Print output for images
print("\nGenerated Images:")
print(f"1. {file_path1}")
print(f"2. {file_path2}")
print(f"3. {file_path3}")
print(f"4. {file_path4}")
print(f"5. {file_path5}")

print("\nThe above visualizations and analyses help understand the geometric interpretation of linear regression.")
print("Use these insights to create an explanation for the five tasks in the question.")

# Add a new visualization showing perpendicular projection more clearly
print("\nPart 6: Additional visualization showing subspace projections without text labels")
print("----------------------------------------------------------------------------")

# Create a new figure for a clean visualization with minimal text
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Create a more complex 3D space to work with
np.random.seed(42)
n_samples = 30
X_vis = np.random.rand(n_samples, 3)
# Make the third column a linear combination of the first two to ensure we have a 2D subspace
X_vis[:, 2] = 0.7 * X_vis[:, 0] + 0.3 * X_vis[:, 1] + np.random.normal(0, 0.05, n_samples)

# PCA to find the principal components (the 2D subspace)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_projected = pca.fit_transform(X_vis)
components = pca.components_

# Create a point that's not in the subspace
point_outside = np.array([0.5, 0.5, 0.9])

# Project the point onto the 2D subspace
projection_coefs = np.array([np.dot(point_outside, components[0]), 
                             np.dot(point_outside, components[1])])
point_projected = projection_coefs[0] * components[0] + projection_coefs[1] * components[1]
point_projected_origin = pca.inverse_transform(np.array([projection_coefs]))[0]

# Calculate the residual
residual = point_outside - point_projected_origin

# Create a grid for the 2D subspace
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 20), np.linspace(-0.5, 1.5, 20))
subspace_points = np.zeros((20, 20, 3))
for i in range(20):
    for j in range(20):
        # Linear combinations of the principal components
        coef1, coef2 = xx[i, j], yy[i, j]
        subspace_points[i, j] = coef1 * components[0] + coef2 * components[1]

# Plot the subspace as a surface with high alpha for clarity
ax.plot_surface(subspace_points[:, :, 0], 
                subspace_points[:, :, 1], 
                subspace_points[:, :, 2], 
                alpha=0.3, color='blue', shade=False)

# Plot the original data points
ax.scatter(X_vis[:, 0], X_vis[:, 1], X_vis[:, 2], color='black', s=20, alpha=0.5)

# Plot the point outside the subspace
ax.scatter([point_outside[0]], [point_outside[1]], [point_outside[2]], 
           color='red', s=100)

# Plot the projected point
ax.scatter([point_projected_origin[0]], [point_projected_origin[1]], [point_projected_origin[2]], 
           color='green', s=100)

# Plot the residual vector
ax.quiver(point_projected_origin[0], point_projected_origin[1], point_projected_origin[2],
          residual[0], residual[1], residual[2], 
          color='orange', arrow_length_ratio=0.1)

# Connect the point to its projection
ax.plot([point_outside[0], point_projected_origin[0]],
        [point_outside[1], point_projected_origin[1]],
        [point_outside[2], point_projected_origin[2]],
        'k--', linewidth=1)

# No labels, let the visualization speak for itself
ax.set_axis_off()
ax.view_init(elev=20, azim=30)

file_path6 = save_figure(fig, "subspace_projection_clean.png")
plt.close(fig)

# Update the final summary to include the new information
print("\nFinal Summary")
print("============")
print("1. Column Space Interpretation: The column space of X represents all possible predictions that can be made")
print("   by linear combinations of the feature vectors.")
print("2. Least Squares Solution: Finds the point in the column space closest to y, minimizing the Euclidean distance.")
print("3. Orthogonality of Residuals: The residual vector is orthogonal to each column of X, confirmed both")
print("   geometrically and mathematically.")
print("4. Geometric Perspective: Explains 'fitting' as finding the optimal projection of y onto the column space.")
print("5. Adding Features: Expands the column space, potentially allowing a closer approximation to y by providing")
print("   more directions for projection.")
print("6. Step-by-Step Calculations: We performed detailed calculations to demonstrate the mathematical foundation")
print("   of these geometric interpretations.")

# Print output for images
print("\nGenerated Images:")
print(f"1. {file_path1}")
print(f"2. {file_path2}")
print(f"3. {file_path3}")
print(f"4. {file_path4}")
print(f"5. {file_path5}")
print(f"6. {file_path6}")

print("\nThe above visualizations and analyses help understand the geometric interpretation of linear regression.")
print("Use these insights to create an explanation for the five tasks in the question.") 