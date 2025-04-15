import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_1_Quiz_3")
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
    
print("Question 3: Geometric Interpretation of Least Squares")
print("====================================================")

# Step 1: Explain the mathematical interpretation of regression as projection
print("\nStep 1: Regression as Projection onto Column Space")
print("-----------------------------------------------")
print("In linear regression, we have a model of the form:")
print("y = Xβ + ε")
print()
print("where y is the target vector, X is the design matrix, β is the vector of")
print("coefficients, and ε is the error vector.")
print()
print("The least squares solution seeks to minimize ||y - Xβ||^2, which is the")
print("squared Euclidean distance between y and Xβ.")
print()
print("The fitted values are ŷ = Xβ̂ = X(X'X)^(-1)X'y = Hy, where H is the hat matrix.")
print()
print("Geometrically, this means that ŷ is the orthogonal projection of y onto the")
print("column space of X, which is the space spanned by the columns of X.")
print()
print("This projection interpretation means that ŷ is the point in the column space of X")
print("that is closest to y in terms of Euclidean distance.")
print()

# Step 2: Explain why residuals are orthogonal to column space
print("\nStep 2: Orthogonality of Residuals to Column Space")
print("-----------------------------------------------")
print("The residual vector is defined as:")
print("e = y - ŷ = y - Xβ̂")
print()
print("For any vector v in the column space of X, v = Xα for some α.")
print()
print("The orthogonality property states that the inner product of the residual vector")
print("with any vector in the column space of X is zero:")
print("<e, v> = e'v = (y - Xβ̂)'(Xα) = 0")
print()
print("This is a fundamental property of least squares estimation and is equivalent to")
print("the normal equations: X'(y - Xβ̂) = 0, which can be rearranged to X'y = X'Xβ̂,")
print("leading to the familiar solution β̂ = (X'X)^(-1)X'y.")
print()
print("Geometrically, this means that the residual vector is perpendicular to the column")
print("space of X, making a right angle with any vector in this space.")
print()

# Step 3: Explain the mathematical property ensuring orthogonality
print("\nStep 3: Mathematical Property Ensuring Orthogonality")
print("------------------------------------------------")
print("The orthogonality of the residual vector to the column space of X is ensured by")
print("the normal equations:")
print("X'(y - Xβ̂) = 0")
print()
print("These equations arise from setting the gradient of the least squares objective")
print("function to zero:")
print("∇β||y - Xβ||^2 = -2X'(y - Xβ) = 0")
print()
print("This leads to X'y = X'Xβ̂, which has the solution β̂ = (X'X)^(-1)X'y when X'X is invertible.")
print()
print("The normal equations ensure that the projection of the residual vector onto each")
print("column of X is zero, which means the residual is orthogonal to the column space of X.")
print()
print("This orthogonality principle is a direct consequence of the least squares criterion")
print("and is the reason why least squares provides the 'best' linear approximation of y")
print("within the column space of X.")
print()

# Create example data for visualization
np.random.seed(42)
n = 20  # Number of observations
p = 2   # Number of predictors (plus intercept)

# Create design matrix with intercept
X = np.column_stack((np.ones(n), np.random.normal(0, 1, (n, p-1))))

# Create true coefficients (fixed to match dimensions)
beta_true = np.array([3, 2])  # Intercept and one predictor coefficient

# Create target vector with noise
y = X @ beta_true + np.random.normal(0, 2, n)

# Compute least squares solution
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y

# Compute fitted values and residuals
y_hat = X @ beta_hat
residuals = y - y_hat

# Compute projection matrix (hat matrix)
H = X @ np.linalg.inv(X.T @ X) @ X.T

print("\nStep 4: Numerical Example")
print("----------------------")
print(f"Design matrix X (first few rows):")
print(X[:5, :])
print()
print(f"Target vector y (first few elements):")
print(y[:5])
print()
print(f"Estimated coefficients β̂:")
print(beta_hat)
print()
print(f"Fitted values ŷ (first few elements):")
print(y_hat[:5])
print()
print(f"Residuals e (first few elements):")
print(residuals[:5])
print()

# Check orthogonality of residuals to column space
ortho_check = X.T @ residuals
print(f"Orthogonality check (X'e), should be close to zero:")
print(ortho_check)
print()

# Compute dot product of residuals with fitted values (should be close to zero)
resid_yhat_dot = np.dot(residuals, y_hat)
print(f"Dot product of residuals and fitted values: {resid_yhat_dot:.10f}")
print(f"(Should be close to zero, confirming orthogonality)")
print()

# Print vector norms
print(f"Norm of y: {np.linalg.norm(y):.4f}")
print(f"Norm of ŷ: {np.linalg.norm(y_hat):.4f}")
print(f"Norm of e: {np.linalg.norm(residuals):.4f}")
print()

# Verify Pythagorean theorem (||y||^2 = ||ŷ||^2 + ||e||^2)
pythagorean_check = np.linalg.norm(y)**2 - (np.linalg.norm(y_hat)**2 + np.linalg.norm(residuals)**2)
print(f"Pythagorean theorem check (should be close to zero): {pythagorean_check:.10f}")
print()

# Create visualizations

# Visualization 1: 2D Representation for Simple Linear Regression
plt.figure(figsize=(10, 6))

# Use only the first predictor for 2D visualization
X_simple = X[:, :2]  # Use intercept and first predictor
beta_simple = np.linalg.inv(X_simple.T @ X_simple) @ X_simple.T @ y
y_hat_simple = X_simple @ beta_simple
residuals_simple = y - y_hat_simple

plt.scatter(X[:, 1], y, color='blue', alpha=0.7, label='Original data points')
plt.plot(X[:, 1], y_hat_simple, 'r-', linewidth=2, label='Regression line (projection)')

# Draw projection lines
for i in range(n):
    plt.plot([X[i, 1], X[i, 1]], [y[i], y_hat_simple[i]], 'k--', alpha=0.5)

plt.title('Projection Interpretation of Linear Regression (2D)', fontsize=14)
plt.xlabel('Predictor (x)', fontsize=12)
plt.ylabel('Response (y)', fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, "regression_2d_projection.png"), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 2: 3D Representation with Column Space
# We'll use a simpler example with 3 observations and 2 predictors to visualize in 3D
n_small = 3
X_small = np.array([[1, 0.5], [1, 1.5], [1, 2.5]])
y_small = np.array([2, 4, 5])

# Compute least squares solution for small example
beta_small = np.linalg.inv(X_small.T @ X_small) @ X_small.T @ y_small
y_hat_small = X_small @ beta_small
residuals_small = y_small - y_hat_small

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the column space of X (a plane in 3D)
# First, find two basis vectors for the column space
col1 = X_small[:, 0]  # Intercept column (all ones)
col2 = X_small[:, 1]  # Predictor column

# Calculate corners of the plane for visualization
max_extent = 1.5 * max(np.max(np.abs(y_small)), 1)
xx, yy = np.meshgrid([-max_extent, max_extent], [-max_extent, max_extent])
z_plane = np.zeros(xx.shape)

# For each point in the grid, find the corresponding point in the column space
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        # Linear combination of the columns of X
        point = col1 * xx[i, j] + col2 * yy[i, j]
        z_plane[i, j] = point[2]  # z-coordinate of the point

# Plot the column space plane
ax.plot_surface(xx, yy, z_plane, alpha=0.3, color='lightblue', label='Column Space of X')

# Plot column vectors of X
ax.quiver(0, 0, 0, col1[0], col1[1], col1[2], color='green', arrow_length_ratio=0.1, label='Column 1 (Intercept)')
ax.quiver(0, 0, 0, col2[0], col2[1], col2[2], color='orange', arrow_length_ratio=0.1, label='Column 2 (Predictor)')

# Plot the original data point (y)
ax.scatter(y_small[0], y_small[1], y_small[2], s=100, c='blue', label='Target Vector y')

# Plot the projected point (ŷ)
ax.scatter(y_hat_small[0], y_hat_small[1], y_hat_small[2], s=100, c='red', label='Projection ŷ = Xβ̂')

# Draw a line from y to its projection
ax.plot([y_small[0], y_hat_small[0]], 
        [y_small[1], y_hat_small[1]], 
        [y_small[2], y_hat_small[2]], 'k--', alpha=0.7)

# Add a 3D arrow for the residual vector
a = Arrow3D([y_hat_small[0], y_small[0]], 
           [y_hat_small[1], y_small[1]], 
           [y_hat_small[2], y_small[2]], 
           mutation_scale=20, lw=2, arrowstyle="-|>", color="purple", alpha=0.7)
ax.add_artist(a)
ax.text((y_hat_small[0] + y_small[0])/2 + 0.1, 
        (y_hat_small[1] + y_small[1])/2 + 0.1, 
        (y_hat_small[2] + y_small[2])/2 + 0.1, 
        "Residual: e = y - ŷ", fontsize=12)

# Set labels and title
ax.set_xlabel('Component 1', fontsize=12)
ax.set_ylabel('Component 2', fontsize=12)
ax.set_zlabel('Component 3', fontsize=12)
ax.set_title('Geometric Interpretation of Least Squares in 3D', fontsize=14)

# Adjust view for clarity
ax.view_init(elev=20, azim=30)

# Create a custom legend
ax.legend(loc='upper left', fontsize=10)

plt.savefig(os.path.join(save_dir, "regression_3d_projection.png"), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 3: Orthogonality Visualization
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the column space plane as before
ax.plot_surface(xx, yy, z_plane, alpha=0.2, color='lightblue')

# Plot column vectors of X
ax.quiver(0, 0, 0, col1[0], col1[1], col1[2], color='green', arrow_length_ratio=0.1, label='Column 1')
ax.quiver(0, 0, 0, col2[0], col2[1], col2[2], color='orange', arrow_length_ratio=0.1, label='Column 2')

# Plot the original data point (y)
ax.scatter(y_small[0], y_small[1], y_small[2], s=100, c='blue', label='y')

# Plot the projected point (ŷ)
ax.scatter(y_hat_small[0], y_hat_small[1], y_hat_small[2], s=100, c='red', label='ŷ')

# Plot the residual vector
ax.quiver(y_hat_small[0], y_hat_small[1], y_hat_small[2],
          residuals_small[0], residuals_small[1], residuals_small[2], 
          color='purple', arrow_length_ratio=0.1, label='Residual e')

# Highlight orthogonality with a right angle marker
# Create a small cube at the projection point to represent the right angle
size = 0.2
ax.plot([y_hat_small[0], y_hat_small[0]+size], 
        [y_hat_small[1], y_hat_small[1]], 
        [y_hat_small[2], y_hat_small[2]], 'k-', linewidth=2)
ax.plot([y_hat_small[0], y_hat_small[0]], 
        [y_hat_small[1], y_hat_small[1]+size], 
        [y_hat_small[2], y_hat_small[2]], 'k-', linewidth=2)
ax.text(y_hat_small[0]+size/2, y_hat_small[1]+size/2, y_hat_small[2], 
        "90°", fontsize=12)

# Add annotations about orthogonality
ax.text(0, 0, max_extent, 
        "Orthogonality: e ⊥ Column Space of X\n" + 
        f"Dot product of e with columns of X:\n" +
        f"e·col1 = {np.dot(residuals_small, col1):.10f}\n" +
        f"e·col2 = {np.dot(residuals_small, col2):.10f}",
        fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

# Set labels and title
ax.set_xlabel('Component 1', fontsize=12)
ax.set_ylabel('Component 2', fontsize=12)
ax.set_zlabel('Component 3', fontsize=12)
ax.set_title('Orthogonality of Residuals to Column Space', fontsize=14)

# Adjust view for clarity
ax.view_init(elev=20, azim=120)

# Create a custom legend
ax.legend(loc='upper left', fontsize=10)

plt.savefig(os.path.join(save_dir, "residual_orthogonality.png"), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 4: Normal Equations Visualization
plt.figure(figsize=(12, 8))

# Create a simple 2D example for visualization
x = np.linspace(0, 10, 100)
y_true = 2 + 3 * x
y_noise = y_true + np.random.normal(0, 5, size=len(x))

# Fit linear model
X_viz = np.column_stack((np.ones_like(x), x))
beta_viz = np.linalg.inv(X_viz.T @ X_viz) @ X_viz.T @ y_noise
y_fit = X_viz @ beta_viz
residuals_viz = y_noise - y_fit

# Plot data and fit
plt.scatter(x, y_noise, alpha=0.6, label='Data points')
plt.plot(x, y_fit, 'r-', linewidth=2, label='Fitted line')

# Draw residuals for a subset of points
for i in range(0, len(x), 10):
    plt.plot([x[i], x[i]], [y_noise[i], y_fit[i]], 'k--', alpha=0.5)

# Add explanation of normal equations
plt.text(5, np.max(y_noise) + 5,
         "Normal Equations: X'(y - Xβ̂) = 0\n" +
         "This ensures that residuals are orthogonal to X\n" +
         "Leading to β̂ = (X'X)^(-1)X'y",
         horizontalalignment='center',
         bbox=dict(facecolor='white', alpha=0.8),
         fontsize=12)

plt.title('Visual Representation of Normal Equations', fontsize=14)
plt.xlabel('Predictor (x)', fontsize=12)
plt.ylabel('Response (y)', fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, "normal_equations.png"), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 5: Pythagorean Theorem in Regression
plt.figure(figsize=(10, 8))

# Create a diagram showing the Pythagorean relationship
# Draw a right triangle
plt.plot([0, 5], [0, 0], 'b-', linewidth=3, label='||ŷ|| (Projection)')
plt.plot([5, 5], [0, 4], 'r-', linewidth=3, label='||e|| (Residuals)')
plt.plot([0, 5], [0, 4], 'g-', linewidth=3, label='||y|| (Target)')

# Add annotations
plt.text(2.5, -0.5, f"||ŷ|| = {np.linalg.norm(y_hat):.2f}", fontsize=12, ha='center')
plt.text(5.5, 2, f"||e|| = {np.linalg.norm(residuals):.2f}", fontsize=12, va='center')
plt.text(2, 2.5, f"||y|| = {np.linalg.norm(y):.2f}", fontsize=12, ha='center')

# Add the Pythagorean theorem formula
plt.text(2.5, 5.5,
         "Pythagorean Theorem in Regression:\n" +
         "||y||² = ||ŷ||² + ||e||²\n" +
         f"{np.linalg.norm(y)**2:.2f} = {np.linalg.norm(y_hat)**2:.2f} + {np.linalg.norm(residuals)**2:.2f}\n" +
         f"{np.linalg.norm(y)**2:.2f} ≈ {np.linalg.norm(y_hat)**2 + np.linalg.norm(residuals)**2:.2f}",
         horizontalalignment='center',
         bbox=dict(facecolor='yellow', alpha=0.2),
         fontsize=12)

plt.plot([5, 5.5], [0, 0], 'k-')  # Right angle marker
plt.plot([5, 5], [0, 0.5], 'k-')

plt.xlim(-1, 8)
plt.ylim(-1, 7)
plt.gca().set_aspect('equal')
plt.axis('off')
plt.title('Pythagorean Theorem in Regression', fontsize=14)
plt.legend(loc='upper left')
plt.savefig(os.path.join(save_dir, "pythagorean_theorem.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nVisualizations saved to directory: {save_dir}")
print("\nQuestion 3 Answers:")
print("1. The interpretation of regression as a projection means that the fitted values")
print("   ŷ = Xβ̂ represent the orthogonal projection of the target vector y onto the column")
print("   space of X. This means ŷ is the closest point to y (in terms of Euclidean distance)")
print("   that lies within the space of all possible linear combinations of the columns of X.")
print()
print("2. The residual vector e = y - ŷ is orthogonal to the column space of X because")
print("   it represents the shortest path from y to its projection ŷ. By definition, the")
print("   shortest path from a point to a subspace is perpendicular to that subspace.")
print("   This orthogonality is mathematically expressed as X'e = 0.")
print()
print("3. The orthogonality is ensured by the normal equations X'(y - Xβ̂) = 0, which")
print("   arise from minimizing the sum of squared residuals (||y - Xβ||²). When we")
print("   differentiate this expression with respect to β and set it to zero, we get")
print("   the normal equations, which guarantee the orthogonality property.") 