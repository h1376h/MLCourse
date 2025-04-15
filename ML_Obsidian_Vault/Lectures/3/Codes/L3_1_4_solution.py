import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import os
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_1_Quiz_4")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# For 3D arrows
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"{title.center(80)}")
    print("="*80 + "\n")

# Start with an introduction
print_section_header("QUESTION 4: HAT MATRIX IN REGRESSION")

print("""Problem Statement:
Consider a linear regression model with design matrix X and response vector y.
The hat matrix H is defined as H = X(X'X)^-1X'.

Tasks:
1. State two key properties of the hat matrix H
2. Explain why H is called a projection matrix
3. What is the relationship between H and the fitted values ŷ?
""")

# Create a simple example to illustrate the hat matrix properties
print_section_header("GENERATING A SIMPLE EXAMPLE")

np.random.seed(42)  # For reproducibility
n_samples = 20  # Number of observations
n_features = 2  # Number of features (excluding intercept)

# Generate some data
X_raw = np.random.rand(n_samples, n_features)  # Random features
X = np.column_stack([np.ones(n_samples), X_raw])  # Add intercept column
true_beta = np.array([3, 1.5, -2])  # True coefficients: intercept, beta_1, beta_2

# Create response with some noise
y = X @ true_beta + np.random.normal(0, 1, n_samples)

print(f"Generated a dataset with {n_samples} observations and {n_features} features (plus intercept).")
print(f"Sample X (first 5 rows):")
for i in range(min(5, n_samples)):
    print(f"  Row {i+1}: {X[i]}")
print(f"Sample y (first 5 values): {y[:5]}")
print(f"True coefficients: {true_beta}")

# Calculate the hat matrix H
print_section_header("STEP 1: CALCULATE THE HAT MATRIX")

print("The hat matrix H is defined as:")
print("H = X(X'X)^-1X'")
print("\nCalculating H step by step:")

# Calculate X'X
X_transpose = X.T
X_transpose_X = X_transpose @ X
print(f"\nX' (transpose of X) shape: {X_transpose.shape}")
print(f"X'X shape: {X_transpose_X.shape}")
print("X'X sample (3x3 matrix):")
print(X_transpose_X)

# Calculate (X'X)^-1
X_transpose_X_inv = np.linalg.inv(X_transpose_X)
print(f"\n(X'X)^-1 shape: {X_transpose_X_inv.shape}")
print("(X'X)^-1 sample (3x3 matrix):")
print(X_transpose_X_inv)

# Calculate X(X'X)^-1X'
hat_matrix = X @ X_transpose_X_inv @ X_transpose
print(f"\nHat matrix H shape: {hat_matrix.shape}")
print("Hat matrix H is a square {n_samples}x{n_samples} matrix.")
print("H sample (first 5x5 submatrix):")
print(hat_matrix[:5, :5])

# Step 2: Demonstrate key properties of the hat matrix
print_section_header("STEP 2: KEY PROPERTIES OF THE HAT MATRIX")

# Property 1: H is symmetric (H = H')
print("Property 1: H is symmetric (H = H')")
H_transpose = hat_matrix.T
is_symmetric = np.allclose(hat_matrix, H_transpose)
print(f"Is H symmetric? {is_symmetric}")
print("Proof: ||H - H'|| (Frobenius norm) = {:.2e}".format(np.linalg.norm(hat_matrix - H_transpose)))

# Property 2: H is idempotent (H² = H)
print("\nProperty 2: H is idempotent (H² = H)")
H_squared = hat_matrix @ hat_matrix
is_idempotent = np.allclose(hat_matrix, H_squared)
print(f"Is H idempotent? {is_idempotent}")
print("Proof: ||H² - H|| (Frobenius norm) = {:.2e}".format(np.linalg.norm(H_squared - hat_matrix)))

# Property 3: H has rank p (same as rank of X)
print("\nProperty 3: The rank of H is p (number of columns in X)")
rank_H = np.linalg.matrix_rank(hat_matrix)
rank_X = np.linalg.matrix_rank(X)
p = X.shape[1]  # Number of columns in X
print(f"Rank of H: {rank_H}")
print(f"Rank of X: {rank_X}")
print(f"Number of columns in X (p): {p}")
print(f"Are rank(H) and rank(X) equal? {rank_H == rank_X}")

# Property 4: Eigenvalues of H are either 0 or 1
print("\nProperty 4: Eigenvalues of H are either 0 or 1")
eigenvalues = np.linalg.eigvals(hat_matrix)
print(f"Eigenvalues of H (first 10): {np.round(eigenvalues[:10], 6)}")
print(f"Number of eigenvalues close to 1: {np.sum(np.isclose(eigenvalues, 1))}")
print(f"Number of eigenvalues close to 0: {np.sum(np.isclose(eigenvalues, 0))}")
print(f"Total number of eigenvalues: {len(eigenvalues)}")

# Step 3: Projection properties
print_section_header("STEP 3: WHY H IS CALLED A PROJECTION MATRIX")

print("The hat matrix H projects the response vector y onto the column space of X.")
print("This means that Hy is the closest point to y in the column space of X.")
print("Mathematically, Hy is the orthogonal projection of y onto the column space of X.")

# Calculate fitted values and residuals
y_hat = hat_matrix @ y
residuals = y - y_hat

print(f"\nOriginal response y (first 5 values): {y[:5]}")
print(f"Fitted values ŷ = Hy (first 5 values): {y_hat[:5]}")
print(f"Residuals (first 5 values): {residuals[:5]}")

# Check orthogonality of residuals to X
ortho_check = X.T @ residuals
print("\nOrthogonality check (X'(y - ŷ)):")
print(ortho_check)
print(f"Are residuals orthogonal to column space of X? {np.allclose(ortho_check, np.zeros_like(ortho_check))}")

# Calculate OLS estimates for comparison
beta_ols = np.linalg.inv(X.T @ X) @ X.T @ y
y_hat_direct = X @ beta_ols
print("\nFor comparison, we can also calculate ŷ directly:")
print(f"OLS estimates β: {beta_ols}")
print(f"Fitted values ŷ = Xβ (first 5 values): {y_hat_direct[:5]}")
print(f"Are both ways of calculating ŷ equal? {np.allclose(y_hat, y_hat_direct)}")

# Create visualizations
print_section_header("VISUALIZATIONS")

# Visualization 1: Hat matrix heatmap
plt.figure(figsize=(8, 6))
heatmap = plt.imshow(hat_matrix, cmap='viridis')
plt.colorbar(heatmap, label='Value')
plt.title('Hat Matrix (H) Heatmap', fontsize=14)
plt.xlabel('Observation Index', fontsize=12)
plt.ylabel('Observation Index', fontsize=12)

# Add textual information
textbox = f"Key Properties:\n- Symmetric: H = H'\n- Idempotent: H² = H\n- Rank = {p}\n- Eigenvalues are 0 or 1"
plt.text(0.02, 0.02, textbox, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='bottom', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
hat_matrix_heatmap_path = os.path.join(save_dir, "hat_matrix_heatmap.png")
plt.savefig(hat_matrix_heatmap_path, dpi=300, bbox_inches='tight')
plt.close()

# Visualization 2: Eigenvalues of the hat matrix
plt.figure(figsize=(8, 6))
eig_vals_sorted = np.sort(eigenvalues)[::-1]  # Sort in descending order
plt.plot(range(1, len(eig_vals_sorted) + 1), eig_vals_sorted, 'o-', markersize=6)
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, 
            label='Threshold Value (0.5)')

plt.title('Eigenvalues of the Hat Matrix', fontsize=14)
plt.xlabel('Index', fontsize=12)
plt.ylabel('Eigenvalue', fontsize=12)
plt.ylim(-0.1, 1.1)
plt.grid(True, alpha=0.3)
plt.legend()

textbox = f"Hat Matrix Properties:\n- {np.sum(np.isclose(eig_vals_sorted, 1))} eigenvalues ≈ 1\n- {np.sum(np.isclose(eig_vals_sorted, 0))} eigenvalues ≈ 0\n- Total eigenvalues: {len(eig_vals_sorted)}"
plt.text(0.98, 0.02, textbox, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
eigenvalues_plot_path = os.path.join(save_dir, "eigenvalues_plot.png")
plt.savefig(eigenvalues_plot_path, dpi=300, bbox_inches='tight')
plt.close()

# For simplicity, let's do a 2D projection example to visualize
print("\nGenerating a 2D projection visualization for easier understanding...")

# Generate a simple 2D dataset
np.random.seed(123)
n_samples_2d = 50
X_2d = np.column_stack([np.ones(n_samples_2d), np.random.uniform(-3, 3, n_samples_2d)])
true_beta_2d = np.array([2, 1.5])
y_2d = X_2d @ true_beta_2d + np.random.normal(0, 2, n_samples_2d)

# Calculate projection
H_2d = X_2d @ np.linalg.inv(X_2d.T @ X_2d) @ X_2d.T
y_hat_2d = H_2d @ y_2d
residuals_2d = y_2d - y_hat_2d

# Plotting
plt.figure(figsize=(10, 6))
x_feature = X_2d[:, 1]  # The non-intercept feature

# Plot the data points
plt.scatter(x_feature, y_2d, color='blue', label='Original Data Points (y)', s=50, alpha=0.7)

# Plot the fitted line
x_range = np.linspace(min(x_feature), max(x_feature), 100)
X_range = np.column_stack([np.ones(100), x_range])
y_range = X_range @ np.linalg.inv(X_2d.T @ X_2d) @ X_2d.T @ y_2d
plt.plot(x_range, y_range, 'r-', linewidth=2, label='Fitted Line (Column Space of X)')

# Plot the projections and residuals
for i in range(n_samples_2d):
    plt.plot([x_feature[i], x_feature[i]], [y_2d[i], y_hat_2d[i]], 'g--', alpha=0.4)

# Plot the fitted points
plt.scatter(x_feature, y_hat_2d, color='red', label='Fitted Points (Hy = ŷ)', s=50)

plt.title('Projection Interpretation of the Hat Matrix', fontsize=14)
plt.xlabel('X Feature', fontsize=12)
plt.ylabel('Y Value', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# Add explanatory text
textbox = "The hat matrix H projects y onto\nthe column space of X (the line),\nproducing the fitted values ŷ = Hy.\nThe green lines represent the residuals,\nwhich are orthogonal to the column space."
plt.text(0.02, 0.98, textbox, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
projection_2d_path = os.path.join(save_dir, "projection_2d.png")
plt.savefig(projection_2d_path, dpi=300, bbox_inches='tight')
plt.close()

# Visualization 4: 3D projection for a more complete picture
# We'll use a simplified 3D case where X has 2 features (plus intercept)
np.random.seed(456)
n_samples_3d = 50
X_3d_raw = np.random.uniform(-3, 3, (n_samples_3d, 2))
X_3d = np.column_stack([np.ones(n_samples_3d), X_3d_raw])
true_beta_3d = np.array([2, 1, -1])
y_3d = X_3d @ true_beta_3d + np.random.normal(0, 1, n_samples_3d)

# Calculate the hat matrix and projection
H_3d = X_3d @ np.linalg.inv(X_3d.T @ X_3d) @ X_3d.T
y_hat_3d = H_3d @ y_3d
residuals_3d = y_3d - y_hat_3d

# 3D plotting
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the plane (column space of X)
xx, yy = np.meshgrid(np.linspace(-3, 3, 10), np.linspace(-3, 3, 10))
z = true_beta_3d[0] + true_beta_3d[1] * xx + true_beta_3d[2] * yy
ax.plot_surface(xx, yy, z, alpha=0.3, color='gray')

# Plot original points
ax.scatter(X_3d_raw[:, 0], X_3d_raw[:, 1], y_3d, color='blue', 
           label='Original Points (y)', s=50, alpha=0.7)

# Plot projected points
ax.scatter(X_3d_raw[:, 0], X_3d_raw[:, 1], y_hat_3d, color='red',
           label='Projected Points (Hy = ŷ)', s=50)

# Plot residual lines
for i in range(min(15, n_samples_3d)):  # Limit to 15 lines for clarity
    ax.plot([X_3d_raw[i, 0], X_3d_raw[i, 0]], 
            [X_3d_raw[i, 1], X_3d_raw[i, 1]], 
            [y_3d[i], y_hat_3d[i]], 
            'g-', alpha=0.4)

ax.set_xlabel('X₁', fontsize=12)
ax.set_ylabel('X₂', fontsize=12)
ax.set_zlabel('Y', fontsize=12)
ax.set_title('3D Projection via Hat Matrix', fontsize=14)
ax.legend()

# Adjust viewing angle for better visibility
ax.view_init(elev=30, azim=45)

plt.tight_layout()
projection_3d_path = os.path.join(save_dir, "projection_3d.png")
plt.savefig(projection_3d_path, dpi=300, bbox_inches='tight')
plt.close()

# Visualization 5: Relationship diagram between y, ŷ, and H
plt.figure(figsize=(10, 6))

# Create a flowchart-like diagram
gs = GridSpec(1, 3, width_ratios=[1, 1, 1])

ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
ax3 = plt.subplot(gs[2])

# Draw y vector (original responses)
ax1.text(0.5, 0.5, r"$\mathbf{y}$", fontsize=36, ha='center', va='center')
ax1.text(0.5, 0.1, "Original Response\nVector", ha='center', va='center', fontsize=12)
ax1.axis('off')

# Draw H (hat matrix)
ax2.text(0.5, 0.5, r"$\mathbf{H}$", fontsize=36, ha='center', va='center')
ax2.text(0.5, 0.1, "Hat Matrix\nH = X(X'X)⁻¹X'", ha='center', va='center', fontsize=12)
ax2.axis('off')

# Draw ŷ (fitted values)
ax3.text(0.5, 0.5, r"$\hat{\mathbf{y}}$", fontsize=36, ha='center', va='center')
ax3.text(0.5, 0.1, "Fitted Values\nŷ = Hy", ha='center', va='center', fontsize=12)
ax3.axis('off')

# Add arrows
plt.annotate("", xy=(0.33, 0.5), xytext=(0.02, 0.5), 
             xycoords=ax2.transAxes, textcoords=ax1.transAxes,
             arrowprops=dict(arrowstyle="->", color="black", linewidth=2))

plt.annotate("", xy=(0.02, 0.5), xytext=(0.33, 0.5), 
             xycoords=ax3.transAxes, textcoords=ax2.transAxes,
             arrowprops=dict(arrowstyle="->", color="black", linewidth=2))

# Add the direct relationship y -> ŷ
plt.annotate("", xy=(0.15, 0.3), xytext=(0.15, 0.3), 
             xycoords=ax3.transAxes, textcoords=ax1.transAxes,
             arrowprops=dict(arrowstyle="->", color="red", linewidth=2, 
                            connectionstyle="arc3,rad=0.3"))

plt.text(0.5, 0.85, "Hat Matrix: The Projector", fontsize=16, ha='center', transform=plt.gcf().transFigure)
plt.text(0.5, 0.78, "ŷ = Hy: The hat matrix projects y onto the column space of X", 
         fontsize=12, ha='center', transform=plt.gcf().transFigure)

plt.tight_layout()
relationship_path = os.path.join(save_dir, "relationship_diagram.png")
plt.savefig(relationship_path, dpi=300, bbox_inches='tight')
plt.close()

# Visualization 6: Diagonal elements of H (leverage)
plt.figure(figsize=(10, 6))
leverage = np.diag(hat_matrix)

plt.bar(range(1, n_samples+1), leverage, color='steelblue')
plt.axhline(y=p/n_samples, color='red', linestyle='--', label=f'Average Leverage: {p/n_samples:.3f}')

plt.title('Leverage (Diagonal Elements of Hat Matrix)', fontsize=14)
plt.xlabel('Observation Index', fontsize=12)
plt.ylabel('Leverage (h_ii)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()

# Add explanatory text
textbox = "Leverage measures how much an observation\ninfluences its own prediction.\n- Higher values indicate more influence\n- Sum of all leverages equals p (rank of X)\n- Average leverage = p/n"
plt.text(0.02, 0.98, textbox, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
leverage_path = os.path.join(save_dir, "leverage_plot.png")
plt.savefig(leverage_path, dpi=300, bbox_inches='tight')
plt.close()

print_section_header("SUMMARY AND CONCLUSIONS")

print("""Properties of the Hat Matrix (H):

1. H is symmetric: H = H'
   - This means H is equal to its transpose, which is a key property of projection matrices.

2. H is idempotent: H² = H
   - Applying H twice yields the same result as applying it once, confirming it's a projection.

3. The rank of H equals p (number of columns in X)
   - This tells us the dimensionality of the projection space.

4. Eigenvalues of H are either 0 or 1
   - This is characteristic of projection matrices; eigenvectors with eigenvalue 1 are in the column space of X.

5. H projects y onto the column space of X
   - The fitted values ŷ = Hy are the orthogonal projection of y onto the space spanned by the columns of X.

6. The diagonal elements of H (h_ii) represent leverage
   - These values measure how much each observation influences its own prediction.

Relationship between H and ŷ:
   - The hat matrix H transforms the response vector y into the fitted values ŷ: ŷ = Hy
   - This is equivalent to calculating ŷ = Xβ where β = (X'X)⁻¹X'y are the OLS estimates
   - H "puts the hat on y" by projecting it onto the column space of X

The hat matrix encapsulates the entire fitting process of linear regression in a single matrix,
making it a fundamental concept for understanding the geometric interpretation of regression.
""")

print(f"\nVisualizations saved in: {save_dir}")
print(f"1. Hat Matrix Heatmap: {hat_matrix_heatmap_path}")
print(f"2. Eigenvalues Plot: {eigenvalues_plot_path}")
print(f"3. 2D Projection: {projection_2d_path}")
print(f"4. 3D Projection: {projection_3d_path}")
print(f"5. Relationship Diagram: {relationship_path}")
print(f"6. Leverage Plot: {leverage_path}") 