import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from scipy.stats import multivariate_normal, norm
from scipy import linalg
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_1_35")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

def print_statement_result(statement_number, is_true, explanation):
    """Print a formatted statement result with explanation."""
    verdict = "TRUE" if is_true else "FALSE"
    print(f"\nStatement {statement_number} is {verdict}.")
    print(explanation)

# Step 1: Set up the problem
print_step_header(1, "Understanding the Problem")

print("Question 35: Evaluate whether each of the following statements is TRUE or FALSE.")
print("Justify your answer with a brief explanation.\n")
print("1. For a multivariate normal distribution, a diagonal covariance matrix implies that the variables are uncorrelated, resulting in probability density contours that are axis-aligned ellipses (or circles if variances are equal).")
print("2. Covariance measures the tendency of two random variables to vary together; a positive value indicates they tend to increase or decrease together, while a negative value indicates one tends to increase as the other decreases.")
print("3. All valid covariance matrices must be positive semi-definite, meaning Var(a^T X) = a^T Σ a ≥ 0 for any vector a.")
print("4. A covariance matrix is strictly positive definite if and only if all its eigenvalues are strictly positive; this condition guarantees the matrix is invertible.")
print("5. Covariance only quantifies the strength and direction of the linear relationship between two random variables.")
print("6. Zero covariance (Cov(X,Y) = 0) guarantees that the random variables X and Y are statistically independent.")
print("7. The covariance between X and Y can be calculated using the formula Cov(X,Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y].")
print("8. The covariance of a random variable X with itself, Cov(X,X), is equal to its variance, Var(X).")
print("9. In a bivariate normal distribution, negative correlation corresponds to probability density contours being tilted primarily along the line y = -x.")
print("10. The principal axes of the probability density contours for a multivariate normal distribution align with the eigenvectors of its covariance matrix.")
print("11. Contour lines on a probability density plot connect points having the same probability density value.")
print("12. For any n×n covariance matrix with eigenvalues λ₁ ≥ λ₂ ≥ ... ≥ λₙ, the volume of the ellipsoid representing the region within one standard deviation is directly proportional to the sum of eigenvalues rather than their product.")
print("13. In a multivariate normal distribution, the angle of rotation of probability density contours in a 2D plane is always given by θ = (1/2)tan⁻¹(2σₓᵧ/(σₓ²-σᵧ²)), regardless of whether σₓ² = σᵧ².")

# Step 2: Visualize diagonal covariance matrices and axis-aligned contours
print_step_header(2, "Visualizing Diagonal Covariance Matrices and Axis-Aligned Contours")

# Create a figure with three covariance matrices
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Create a meshgrid for plotting
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Case 1: Diagonal covariance matrix with equal variances (circle)
mean = [0, 0]
cov1 = np.array([[1, 0], [0, 1]])  # Identity covariance (equal variances)
rv1 = multivariate_normal(mean, cov1)
Z1 = rv1.pdf(pos)

axs[0].contour(X, Y, Z1, levels=10, colors='blue')
axs[0].set_title("Diagonal Covariance\nEqual Variances")
axs[0].set_xlabel("X")
axs[0].set_ylabel("Y")
axs[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axs[0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axs[0].set_aspect('equal')
axs[0].grid(True)
axs[0].text(-2.5, 2.5, f"Σ = {cov1[0,0]:.1f}  {cov1[0,1]:.1f}\n    {cov1[1,0]:.1f}  {cov1[1,1]:.1f}", bbox=dict(facecolor='white', alpha=0.8))

# Case 2: Diagonal covariance matrix with unequal variances (axis-aligned ellipse)
cov2 = np.array([[2, 0], [0, 0.5]])  # Diagonal but unequal variances
rv2 = multivariate_normal(mean, cov2)
Z2 = rv2.pdf(pos)

axs[1].contour(X, Y, Z2, levels=10, colors='green')
axs[1].set_title("Diagonal Covariance\nUnequal Variances")
axs[1].set_xlabel("X")
axs[1].set_ylabel("Y")
axs[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axs[1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axs[1].set_aspect('equal')
axs[1].grid(True)
axs[1].text(-2.5, 2.5, f"Σ = {cov2[0,0]:.1f}  {cov2[0,1]:.1f}\n    {cov2[1,0]:.1f}  {cov2[1,1]:.1f}", bbox=dict(facecolor='white', alpha=0.8))

# Case 3: Non-diagonal covariance matrix (tilted ellipse)
cov3 = np.array([[1, 0.8], [0.8, 1]])  # Non-diagonal (correlated variables)
rv3 = multivariate_normal(mean, cov3)
Z3 = rv3.pdf(pos)

axs[2].contour(X, Y, Z3, levels=10, colors='red')
axs[2].set_title("Non-Diagonal Covariance")
axs[2].set_xlabel("X")
axs[2].set_ylabel("Y")
axs[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axs[2].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axs[2].set_aspect('equal')
axs[2].grid(True)
axs[2].text(-2.5, 2.5, f"Σ = {cov3[0,0]:.1f}  {cov3[0,1]:.1f}\n    {cov3[1,0]:.1f}  {cov3[1,1]:.1f}", bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
file_path = os.path.join(save_dir, "1_diagonal_covariance.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Detailed explanation in text rather than in the image
print("\nStatement 1 Analysis:")
print("When a covariance matrix is diagonal (all off-diagonal elements are zero):")
print("- The variables are uncorrelated")
print("- The probability density contours are aligned with the coordinate axes")
print("- If diagonal elements are equal, contours form circles (as in plot 1)")
print("- If diagonal elements are unequal, contours form axis-aligned ellipses (as in plot 2)")
print("- When off-diagonal elements are non-zero, contours form tilted ellipses (as in plot 3)")
print("- The contour lines connect points with equal probability density")

print_statement_result(1, True, "A diagonal covariance matrix implies uncorrelated variables and results in axis-aligned elliptical contours (or circular contours if the diagonal elements are equal).")

# Step 3: Visualize positive and negative covariance
print_step_header(3, "Visualizing Positive and Negative Covariance")

# Generate samples with positive, negative, and zero covariance
np.random.seed(42)
n_samples = 500

# Positive covariance
mean = [0, 0]
cov_pos = [[1, 0.8], [0.8, 1]]
samples_pos = np.random.multivariate_normal(mean, cov_pos, n_samples)

# Negative covariance
cov_neg = [[1, -0.8], [-0.8, 1]]
samples_neg = np.random.multivariate_normal(mean, cov_neg, n_samples)

# Zero covariance
cov_zero = [[1, 0], [0, 1]]
samples_zero = np.random.multivariate_normal(mean, cov_zero, n_samples)

# Create figure with three scatter plots
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Plot positive covariance
axs[0].scatter(samples_pos[:, 0], samples_pos[:, 1], alpha=0.5)
axs[0].set_title(f"Positive Covariance\n(Cov = {cov_pos[0][1]:.1f})")
axs[0].set_xlabel("X")
axs[0].set_ylabel("Y")
axs[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axs[0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axs[0].set_xlim(-4, 4)
axs[0].set_ylim(-4, 4)
axs[0].grid(True)
z = np.polyfit(samples_pos[:, 0], samples_pos[:, 1], 1)
p = np.poly1d(z)
axs[0].plot([-4, 4], [p(-4), p(4)], "r--", alpha=0.8)

# Plot negative covariance
axs[1].scatter(samples_neg[:, 0], samples_neg[:, 1], alpha=0.5)
axs[1].set_title(f"Negative Covariance\n(Cov = {cov_neg[0][1]:.1f})")
axs[1].set_xlabel("X")
axs[1].set_ylabel("Y")
axs[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axs[1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axs[1].set_xlim(-4, 4)
axs[1].set_ylim(-4, 4)
axs[1].grid(True)
z = np.polyfit(samples_neg[:, 0], samples_neg[:, 1], 1)
p = np.poly1d(z)
axs[1].plot([-4, 4], [p(-4), p(4)], "r--", alpha=0.8)

# Plot zero covariance
axs[2].scatter(samples_zero[:, 0], samples_zero[:, 1], alpha=0.5)
axs[2].set_title(f"Zero Covariance\n(Cov = {cov_zero[0][1]:.1f})")
axs[2].set_xlabel("X")
axs[2].set_ylabel("Y")
axs[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axs[2].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axs[2].set_xlim(-4, 4)
axs[2].set_ylim(-4, 4)
axs[2].grid(True)

plt.tight_layout()
file_path = os.path.join(save_dir, "2_positive_negative_covariance.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Detailed explanation in text
print("\nStatement 2 Analysis:")
print("Covariance measures how two random variables change together:")
print("- Positive covariance (left plot): Variables tend to increase or decrease together")
print("  • When X increases, Y tends to increase")
print("  • When X decreases, Y tends to decrease")
print("  • The trend follows a line with positive slope")
print("- Negative covariance (middle plot): Variables tend to change in opposite directions")
print("  • When X increases, Y tends to decrease")
print("  • When X decreases, Y tends to increase") 
print("  • The trend follows a line with negative slope")
print("- Zero covariance (right plot): No linear relationship between variables")
print("  • Changes in X are not linearly associated with changes in Y")
print("  • The variables appear to be scattered randomly")
print("- The magnitude of covariance indicates the strength of the linear relationship")

print_statement_result(2, True, "Covariance measures how variables vary together, with positive/negative values indicating their directional relationship.")

# Step 4: Positive semi-definite covariance matrices
print_step_header(4, "Demonstrating Positive Semi-Definite Covariance Matrices")

# Create examples of positive definite, positive semi-definite, and indefinite matrices
pos_def_matrix = np.array([[2, 0.5], [0.5, 1]])
pos_semi_def_matrix = np.array([[1, 1], [1, 1]])  # rank-deficient matrix
indef_matrix = np.array([[1, 2], [2, 1]])  # indefinite matrix (not a valid covariance)

# Function to check if a matrix is positive definite/semi-definite
def check_matrix_properties(matrix, matrix_name):
    # Calculate eigenvalues
    eigvals = np.linalg.eigvals(matrix)
    
    # Check properties
    is_symmetric = np.allclose(matrix, matrix.T)
    is_pos_def = np.all(eigvals > 0) and is_symmetric
    is_pos_semi_def = np.all(eigvals >= 0) and is_symmetric
    is_invertible = np.linalg.det(matrix) != 0
    
    print(f"\n{matrix_name}:")
    print(f"Matrix = [[{matrix[0,0]}, {matrix[0,1]}], [{matrix[1,0]}, {matrix[1,1]}]]")
    print(f"Eigenvalues = {eigvals}")
    print(f"Is symmetric: {is_symmetric}")
    print(f"Is positive definite: {is_pos_def}")
    print(f"Is positive semi-definite: {is_pos_semi_def}")
    print(f"Is invertible: {is_invertible}")
    
    # Test the quadratic form for various vectors a
    vectors = [np.array([1, 0]), np.array([0, 1]), np.array([1, 1]), np.array([1, -1])]
    print("\nQuadratic form a^T Σ a:")
    for a in vectors:
        quad_form = a.T @ matrix @ a
        print(f"  a = [{a[0]}, {a[1]}]: a^T Σ a = {quad_form}")
    
    return eigvals, is_pos_def, is_pos_semi_def, is_invertible

# Test the matrices
eigvals_1, is_pos_def_1, is_pos_semi_def_1, is_invertible_1 = check_matrix_properties(pos_def_matrix, "Positive Definite Matrix")
eigvals_2, is_pos_def_2, is_pos_semi_def_2, is_invertible_2 = check_matrix_properties(pos_semi_def_matrix, "Positive Semi-Definite Matrix")
eigvals_3, is_pos_def_3, is_pos_semi_def_3, is_invertible_3 = check_matrix_properties(indef_matrix, "Indefinite Matrix")

# Visualize quadratic forms for each matrix
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Create a meshgrid for plotting
a1_range = np.linspace(-2, 2, 100)
a2_range = np.linspace(-2, 2, 100)
A1, A2 = np.meshgrid(a1_range, a2_range)

# Calculate quadratic form for each point in the grid
Z1 = np.zeros_like(A1)
Z2 = np.zeros_like(A1)
Z3 = np.zeros_like(A1)

for i in range(len(a1_range)):
    for j in range(len(a2_range)):
        a = np.array([A1[i, j], A2[i, j]])
        Z1[i, j] = a.T @ pos_def_matrix @ a
        Z2[i, j] = a.T @ pos_semi_def_matrix @ a
        Z3[i, j] = a.T @ indef_matrix @ a

# Plot each quadratic form
cmap = plt.cm.viridis

# Plot the positive definite case
contour1 = axs[0].contourf(A1, A2, Z1, 20, cmap=cmap)
axs[0].set_title("Positive Definite")
axs[0].set_xlabel("a1")
axs[0].set_ylabel("a2")
axs[0].grid(True)
plt.colorbar(contour1, ax=axs[0])
axs[0].contour(A1, A2, Z1, levels=[0], colors='red', linewidths=2)

# Plot the positive semi-definite case
contour2 = axs[1].contourf(A1, A2, Z2, 20, cmap=cmap)
axs[1].set_title("Positive Semi-Definite")
axs[1].set_xlabel("a1")
axs[1].set_ylabel("a2")
axs[1].grid(True)
plt.colorbar(contour2, ax=axs[1])
axs[1].contour(A1, A2, Z2, levels=[0], colors='red', linewidths=2)

# Plot the indefinite case
contour3 = axs[2].contourf(A1, A2, Z3, 20, cmap=cmap)
axs[2].set_title("Indefinite")
axs[2].set_xlabel("a1")
axs[2].set_ylabel("a2")
axs[2].grid(True)
plt.colorbar(contour3, ax=axs[2])
axs[2].contour(A1, A2, Z3, levels=[0], colors='red', linewidths=2)

plt.tight_layout()
file_path = os.path.join(save_dir, "3_positive_semi_definite.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Detailed explanation in text
print("\nStatement 3 Analysis:")
print("For a matrix to be a valid covariance matrix, it must be positive semi-definite:")
print("- Positive definite matrix (left plot):")
print("  • All eigenvalues are positive")
print("  • For any non-zero vector a, a^T Σ a > 0")
print("  • The quadratic form creates a paraboloid that never crosses below zero")
print("- Positive semi-definite matrix (middle plot):")
print("  • All eigenvalues are non-negative (some can be zero)")
print("  • For any vector a, a^T Σ a ≥ 0")
print("  • The quadratic form creates a paraboloid that touches but never goes below zero")
print("- Indefinite matrix (right plot):")
print("  • Some eigenvalues are negative")
print("  • For some vectors a, a^T Σ a < 0")
print("  • The quadratic form crosses below zero (red contour shows where it equals zero)")
print("  • Cannot be a valid covariance matrix since variance cannot be negative")
print("- This property ensures that Var(a^T X) = a^T Σ a ≥ 0 for any linear combination of random variables")

print_statement_result(3, True, "All valid covariance matrices must be positive semi-definite to ensure variances are non-negative.")

# Step 5: Eigenvalues and positive definiteness
print_step_header(5, "Eigenvalues and Positive Definiteness of Covariance Matrices")

# Create a range of covariance matrices with varying eigenvalues
eig_1 = [2.0, 0.5]  # Positive definite
eig_2 = [1.0, 0.0]  # Positive semi-definite (not invertible)
eig_3 = [0.0, 0.0]  # Zero matrix (degenerate)

# Create matrices with specified eigenvalues
def create_matrix_with_eigenvalues(eigenvalues, rotation=0):
    # Create diagonal matrix with eigenvalues
    D = np.diag(eigenvalues)
    
    # Create rotation matrix
    theta = rotation * np.pi / 180
    Q = np.array([[np.cos(theta), -np.sin(theta)], 
                  [np.sin(theta), np.cos(theta)]])
    
    # Create matrix M = Q D Q^T
    M = Q @ D @ Q.T
    
    return M, Q, D

# Create matrices
matrix_1, Q1, D1 = create_matrix_with_eigenvalues(eig_1, rotation=30)
matrix_2, Q2, D2 = create_matrix_with_eigenvalues(eig_2, rotation=45)
matrix_3, Q3, D3 = create_matrix_with_eigenvalues(eig_3, rotation=0)

# Check properties of each matrix
eigvals_4, is_pos_def_4, is_pos_semi_def_4, is_invertible_4 = check_matrix_properties(matrix_1, "Matrix with positive eigenvalues")
eigvals_5, is_pos_def_5, is_pos_semi_def_5, is_invertible_5 = check_matrix_properties(matrix_2, "Matrix with one zero eigenvalue")
eigvals_6, is_pos_def_6, is_pos_semi_def_6, is_invertible_6 = check_matrix_properties(matrix_3, "Matrix with all zero eigenvalues")

# Visualize how eigenvalues and eigenvectors relate to probability ellipses
# Create a figure
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Create meshgrid
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Case 1: Positive definite (all eigenvalues > 0)
mean = [0, 0]
rv1 = multivariate_normal(mean, matrix_1)
Z1 = rv1.pdf(pos)

# Plot contours
ax1 = axs[0]
ax1.contour(X, Y, Z1, levels=10, colors='blue')
ax1.set_title("Positive Definite\n(All λ > 0)")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_xlim(-3, 3)
ax1.set_ylim(-3, 3)
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)

# Plot eigenvectors scaled by sqrt(eigenvalues)
eigvals, eigvecs = np.linalg.eigh(matrix_1)
ax1.arrow(0, 0, eigvecs[0, 0] * np.sqrt(eigvals[0]), eigvecs[1, 0] * np.sqrt(eigvals[0]), 
          head_width=0.1, head_length=0.1, fc='red', ec='red', label=f"λ₁ = {eigvals[0]:.2f}")
ax1.arrow(0, 0, eigvecs[0, 1] * np.sqrt(eigvals[1]), eigvecs[1, 1] * np.sqrt(eigvals[1]), 
          head_width=0.1, head_length=0.1, fc='green', ec='green', label=f"λ₂ = {eigvals[1]:.2f}")

ax1.text(-2.5, 2.5, f"Invertible: Yes", bbox=dict(facecolor='white', alpha=0.8))
ax1.legend()
ax1.grid(True)
ax1.set_aspect('equal')

# Case 2: Positive semi-definite (one eigenvalue = 0)
rv2 = multivariate_normal(mean, matrix_2 + 1e-10 * np.eye(2), allow_singular=True)  # Add small epsilon for numerical stability
Z2 = rv2.pdf(pos)

# Plot contours
ax2 = axs[1]
ax2.contour(X, Y, Z2, levels=10, colors='blue')
ax2.set_title("Positive Semi-Definite\n(Some λ = 0)")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_xlim(-3, 3)
ax2.set_ylim(-3, 3)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)

# Plot eigenvectors scaled by sqrt(eigenvalues)
eigvals, eigvecs = np.linalg.eigh(matrix_2)
ax2.arrow(0, 0, eigvecs[0, 1] * np.sqrt(eigvals[1]), eigvecs[1, 1] * np.sqrt(eigvals[1]), 
          head_width=0.1, head_length=0.1, fc='red', ec='red', label=f"λ₁ = {eigvals[1]:.2f}")
ax2.arrow(0, 0, eigvecs[0, 0] * 0.2, eigvecs[1, 0] * 0.2, 
          head_width=0.1, head_length=0.1, fc='green', ec='green', label=f"λ₂ = {eigvals[0]:.2f}")

ax2.text(-2.5, 2.5, f"Invertible: No", bbox=dict(facecolor='white', alpha=0.8))
ax2.legend()
ax2.grid(True)
ax2.set_aspect('equal')

# Case 3: Visual representation of invertibility
# Create a custom matrix for demonstration
custom_matrix = np.array([[1.0, 0.5], [0.5, 0.3]])  # Low condition number, poorly conditioned
eigvals, eigvecs = np.linalg.eigh(custom_matrix)

ax3 = axs[2]
ax3.set_title("Eigenvalue Magnitudes")
ax3.set_xlabel("Eigenvalue Index")
ax3.set_ylabel("Eigenvalue Magnitude")
ax3.bar([1, 2], eigvals, color=['red', 'green'])
ax3.axhline(y=0, color='k', linestyle='-', alpha=0.5)
ax3.set_xticks([1, 2])
ax3.set_xticklabels(['λ₁', 'λ₂'])
ax3.grid(True)

# Add condition number
cond_num = np.linalg.cond(custom_matrix)
ax3.text(1.5, eigvals.max()/2, f"Condition Number: {cond_num:.2f}", 
         bbox=dict(facecolor='white', alpha=0.8), ha='center')

plt.tight_layout()
file_path = os.path.join(save_dir, "4_eigenvalues_definiteness.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Detailed explanation in text
print("\nStatement 4 Analysis:")
print("A covariance matrix is strictly positive definite if and only if all its eigenvalues are strictly positive:")
print("- Left plot: Positive definite matrix")
print("  • All eigenvalues are strictly positive")
print("  • The determinant is non-zero (product of eigenvalues)")
print("  • The matrix is invertible")
print("  • The probability density contours form a non-degenerate ellipse")
print("  • The eigenvectors (arrows) show the principal axes of the ellipse")
print("- Middle plot: Positive semi-definite matrix (not positive definite)")
print("  • At least one eigenvalue equals zero")
print("  • The determinant is zero")
print("  • The matrix is not invertible")
print("  • The probability density contours are degenerate (flattened in one direction)")
print("- Right plot: Relationship between eigenvalues and invertibility")
print("  • A matrix is invertible if and only if all eigenvalues are non-zero")
print("  • The determinant equals the product of all eigenvalues")
print("  • The condition number (ratio of largest to smallest eigenvalue) affects numerical stability")

print_statement_result(4, True, "A covariance matrix is strictly positive definite if and only if all eigenvalues are positive, making it invertible.")

# Step 6: Linear relationship in covariance
print_step_header(6, "Demonstrating that Covariance Captures Linear Relationships")

# Create data for both linear and non-linear relationships
np.random.seed(42)
n_samples = 300

# Linear relationship
x_lin = np.random.uniform(-3, 3, n_samples)
y_lin = 2 * x_lin + 0.5 * np.random.normal(size=n_samples)

# Non-linear relationship (quadratic)
x_nonlin = np.random.uniform(-3, 3, n_samples)
y_nonlin = x_nonlin**2 + 0.5 * np.random.normal(size=n_samples)
# Center the data to have zero means for proper covariance calculation
x_nonlin_centered = x_nonlin - np.mean(x_nonlin)
y_nonlin_centered = y_nonlin - np.mean(y_nonlin)

# Compute covariances
cov_linear = np.cov(x_lin, y_lin)[0, 1]
corr_linear = np.corrcoef(x_lin, y_lin)[0, 1]
cov_nonlinear = np.cov(x_nonlin, y_nonlin)[0, 1]
corr_nonlinear = np.corrcoef(x_nonlin, y_nonlin)[0, 1]

# Compute covariance manually to show E[XY] - E[X]E[Y]
E_x_lin = np.mean(x_lin)
E_y_lin = np.mean(y_lin)
E_xy_lin = np.mean(x_lin * y_lin)
manual_cov_lin = E_xy_lin - E_x_lin * E_y_lin

E_x_nonlin = np.mean(x_nonlin)
E_y_nonlin = np.mean(y_nonlin)
E_xy_nonlin = np.mean(x_nonlin * y_nonlin)
manual_cov_nonlin = E_xy_nonlin - E_x_nonlin * E_y_nonlin

# Create figure with subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Plot linear relationship
axs[0, 0].scatter(x_lin, y_lin, alpha=0.5)
axs[0, 0].set_title(f"Linear Relationship\nCov = {cov_linear:.3f}, Corr = {corr_linear:.3f}")
axs[0, 0].set_xlabel("X")
axs[0, 0].set_ylabel("Y")
axs[0, 0].grid(True)

# Fit and plot a linear model
z = np.polyfit(x_lin, y_lin, 1)
p = np.poly1d(z)
x_range = np.linspace(-3, 3, 100)
axs[0, 0].plot(x_range, p(x_range), "r--", alpha=0.8)

# Plot non-linear relationship
axs[0, 1].scatter(x_nonlin, y_nonlin, alpha=0.5)
axs[0, 1].set_title(f"Non-linear Relationship\nCov = {cov_nonlinear:.3f}, Corr = {corr_nonlinear:.3f}")
axs[0, 1].set_xlabel("X")
axs[0, 1].set_ylabel("Y")
axs[0, 1].grid(True)

# Fit and plot a quadratic model to show the true relationship
z_nonlin = np.polyfit(x_nonlin, y_nonlin, 2)
p_nonlin = np.poly1d(z_nonlin)
axs[0, 1].plot(x_range, p_nonlin(x_range), "r--", alpha=0.8)

# Fit a linear model to show what covariance captures
z_lin_nonlin = np.polyfit(x_nonlin, y_nonlin, 1)
p_lin_nonlin = np.poly1d(z_lin_nonlin)
axs[0, 1].plot(x_range, p_lin_nonlin(x_range), "g--", alpha=0.8, label="Linear fit")
axs[0, 1].legend()

# Visualize the covariance computation for linear relationship
axs[1, 0].scatter(x_lin - E_x_lin, y_lin - E_y_lin, alpha=0.5)
axs[1, 0].set_title("Centered Linear Data")
axs[1, 0].set_xlabel("X - E[X]")
axs[1, 0].set_ylabel("Y - E[Y]")
axs[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axs[1, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axs[1, 0].grid(True)

# Divide the plot into quadrants and color points accordingly
q1 = ((x_lin - E_x_lin) > 0) & ((y_lin - E_y_lin) > 0)  # Quadrant 1: positive contribution
q3 = ((x_lin - E_x_lin) < 0) & ((y_lin - E_y_lin) < 0)  # Quadrant 3: positive contribution
q2 = ((x_lin - E_x_lin) > 0) & ((y_lin - E_y_lin) < 0)  # Quadrant 2: negative contribution
q4 = ((x_lin - E_x_lin) < 0) & ((y_lin - E_y_lin) > 0)  # Quadrant 4: negative contribution

axs[1, 0].scatter((x_lin - E_x_lin)[q1], (y_lin - E_y_lin)[q1], color='green', alpha=0.5)
axs[1, 0].scatter((x_lin - E_x_lin)[q3], (y_lin - E_y_lin)[q3], color='green', alpha=0.5)
axs[1, 0].scatter((x_lin - E_x_lin)[q2], (y_lin - E_y_lin)[q2], color='red', alpha=0.5)
axs[1, 0].scatter((x_lin - E_x_lin)[q4], (y_lin - E_y_lin)[q4], color='red', alpha=0.5)

# Visualize the covariance computation for non-linear relationship
axs[1, 1].scatter(x_nonlin - E_x_nonlin, y_nonlin - E_y_nonlin, alpha=0.5)
axs[1, 1].set_title("Centered Non-linear Data")
axs[1, 1].set_xlabel("X - E[X]")
axs[1, 1].set_ylabel("Y - E[Y]")
axs[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axs[1, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axs[1, 1].grid(True)

# Divide the plot into quadrants and color points for non-linear data
q1_nonlin = ((x_nonlin - E_x_nonlin) > 0) & ((y_nonlin - E_y_nonlin) > 0)
q3_nonlin = ((x_nonlin - E_x_nonlin) < 0) & ((y_nonlin - E_y_nonlin) < 0)
q2_nonlin = ((x_nonlin - E_x_nonlin) > 0) & ((y_nonlin - E_y_nonlin) < 0)
q4_nonlin = ((x_nonlin - E_x_nonlin) < 0) & ((y_nonlin - E_y_nonlin) > 0)

axs[1, 1].scatter((x_nonlin - E_x_nonlin)[q1_nonlin], (y_nonlin - E_y_nonlin)[q1_nonlin], color='green', alpha=0.5)
axs[1, 1].scatter((x_nonlin - E_x_nonlin)[q3_nonlin], (y_nonlin - E_y_nonlin)[q3_nonlin], color='green', alpha=0.5)
axs[1, 1].scatter((x_nonlin - E_x_nonlin)[q2_nonlin], (y_nonlin - E_y_nonlin)[q2_nonlin], color='red', alpha=0.5)
axs[1, 1].scatter((x_nonlin - E_x_nonlin)[q4_nonlin], (y_nonlin - E_y_nonlin)[q4_nonlin], color='red', alpha=0.5)

plt.tight_layout()
file_path = os.path.join(save_dir, "5_linear_relationship_covariance.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Calculate the contribution of each quadrant to covariance for linear relationship
cov_q1 = np.mean((x_lin[q1] - E_x_lin) * (y_lin[q1] - E_y_lin)) * np.sum(q1) / n_samples if np.sum(q1) > 0 else 0
cov_q2 = np.mean((x_lin[q2] - E_x_lin) * (y_lin[q2] - E_y_lin)) * np.sum(q2) / n_samples if np.sum(q2) > 0 else 0
cov_q3 = np.mean((x_lin[q3] - E_x_lin) * (y_lin[q3] - E_y_lin)) * np.sum(q3) / n_samples if np.sum(q3) > 0 else 0
cov_q4 = np.mean((x_lin[q4] - E_x_lin) * (y_lin[q4] - E_y_lin)) * np.sum(q4) / n_samples if np.sum(q4) > 0 else 0

# Calculate the contribution of each quadrant to covariance for non-linear data
cov_q1_nonlin = np.mean((x_nonlin[q1_nonlin] - E_x_nonlin) * (y_nonlin[q1_nonlin] - E_y_nonlin)) * np.sum(q1_nonlin) / n_samples if np.sum(q1_nonlin) > 0 else 0
cov_q2_nonlin = np.mean((x_nonlin[q2_nonlin] - E_x_nonlin) * (y_nonlin[q2_nonlin] - E_y_nonlin)) * np.sum(q2_nonlin) / n_samples if np.sum(q2_nonlin) > 0 else 0
cov_q3_nonlin = np.mean((x_nonlin[q3_nonlin] - E_x_nonlin) * (y_nonlin[q3_nonlin] - E_y_nonlin)) * np.sum(q3_nonlin) / n_samples if np.sum(q3_nonlin) > 0 else 0
cov_q4_nonlin = np.mean((x_nonlin[q4_nonlin] - E_x_nonlin) * (y_nonlin[q4_nonlin] - E_y_nonlin)) * np.sum(q4_nonlin) / n_samples if np.sum(q4_nonlin) > 0 else 0

# Detailed explanation in text
print("\nStatement 5 Analysis:")
print("Covariance only quantifies the strength and direction of linear relationships between variables:")
print("- Linear relationship (top-left plot):")
print(f"  • Strong linear trend with covariance = {cov_linear:.3f} and correlation = {corr_linear:.3f}")
print("  • Covariance/correlation effectively captures the relationship")
print("- Non-linear relationship (top-right plot):")
print(f"  • Strong quadratic trend but covariance is only {cov_nonlinear:.3f} and correlation is {corr_nonlinear:.3f}")
print("  • The red dashed line shows the true quadratic relationship")
print("  • The green dashed line shows what covariance 'sees' (linear approximation)")
print("  • Covariance misses the non-linear structure completely")
print("\nHow covariance is calculated (centered data plots):")
print("- Green points contribute positively to covariance (points in Q1 and Q3)")
print("- Red points contribute negatively to covariance (points in Q2 and Q4)")
print("- For the linear case, positive contributions (green) dominate:")
print(f"  • Q1 contribution: {cov_q1:.3f}")
print(f"  • Q3 contribution: {cov_q3:.3f}")
print(f"  • Q2 contribution: {cov_q2:.3f}")
print(f"  • Q4 contribution: {cov_q4:.3f}")
print(f"  • Total: {cov_q1+cov_q2+cov_q3+cov_q4:.3f}")
print("- For the non-linear case, positive and negative contributions nearly cancel out:")
print(f"  • Q1 contribution: {cov_q1_nonlin:.3f}")
print(f"  • Q3 contribution: {cov_q3_nonlin:.3f}")
print(f"  • Q2 contribution: {cov_q2_nonlin:.3f}")
print(f"  • Q4 contribution: {cov_q4_nonlin:.3f}")
print(f"  • Total: {cov_q1_nonlin+cov_q2_nonlin+cov_q3_nonlin+cov_q4_nonlin:.3f}")

print_statement_result(5, True, "Covariance only quantifies the strength and direction of linear relationships between variables, missing non-linear dependencies.")

# Step 7: Zero covariance vs. independence
print_step_header(7, "Zero Covariance vs. Independence")

# Create two examples:
# 1. Independent variables (with zero covariance)
# 2. Dependent variables but with zero covariance (e.g., quadratic relationship)

# Independent variables
np.random.seed(42)
n_samples = 1000
x_indep = np.random.normal(0, 1, n_samples)
y_indep = np.random.normal(0, 1, n_samples)

# Dependent variables with zero linear correlation
x_dep = np.random.uniform(-3, 3, n_samples)
y_dep = x_dep**2 + 0.5 * np.random.normal(size=n_samples)
# Center y_dep to ensure zero covariance
y_dep = y_dep - np.polyval(np.polyfit(x_dep, y_dep, 1), x_dep)

# Calculate covariances and correlations
cov_indep = np.cov(x_indep, y_indep)[0, 1]
corr_indep = np.corrcoef(x_indep, y_indep)[0, 1]
cov_dep = np.cov(x_dep, y_dep)[0, 1]
corr_dep = np.corrcoef(x_dep, y_dep)[0, 1]

# Create figure
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Plot independent variables
axs[0].scatter(x_indep, y_indep, alpha=0.3)
axs[0].set_title(f"Independent Variables\nCov = {cov_indep:.4f}")
axs[0].set_xlabel("X")
axs[0].set_ylabel("Y")
axs[0].set_xlim(-4, 4)
axs[0].set_ylim(-4, 4)
axs[0].grid(True)

# Plot dependent variables with zero covariance
axs[1].scatter(x_dep, y_dep, alpha=0.3)
axs[1].set_title(f"Dependent with Zero Covariance\nCov = {cov_dep:.4f}")
axs[1].set_xlabel("X")
axs[1].set_ylabel("Y")
axs[1].set_xlim(-4, 4)
axs[1].set_ylim(-4, 4)
axs[1].grid(True)

# Fit a quadratic curve to show the relationship
z_quad = np.polyfit(x_dep, y_dep, 2)
p_quad = np.poly1d(z_quad)
x_range = np.linspace(-4, 4, 100)
axs[1].plot(x_range, p_quad(x_range), "r--", alpha=0.8)

# Create a joint distribution visualization for the dependent case
bins = 20
hist, x_edges, y_edges = np.histogram2d(x_dep, y_dep, bins=bins)
x_centers = (x_edges[:-1] + x_edges[1:]) / 2
y_centers = (y_edges[:-1] + y_edges[1:]) / 2
X_grid, Y_grid = np.meshgrid(x_centers, y_centers)

# Compute marginal distributions
p_x = np.sum(hist, axis=0) / np.sum(hist)
p_y = np.sum(hist, axis=1) / np.sum(hist)

# Compute product of marginals to compare with joint distribution
joint_hist = hist / np.sum(hist)
product_marginals = np.outer(p_y, p_x)

# Compute difference between joint and product of marginals
diff = joint_hist - product_marginals

# Plot the difference (should be zero for independence)
c = axs[2].pcolormesh(X_grid, Y_grid, diff.T, cmap='coolwarm', vmin=-0.01, vmax=0.01)
axs[2].set_title("Joint - Product of Marginals")
axs[2].set_xlabel("X")
axs[2].set_ylabel("Y")
axs[2].grid(True)
plt.colorbar(c, ax=axs[2])

plt.tight_layout()
file_path = os.path.join(save_dir, "6_zero_covariance_vs_independence.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Detailed explanation in text
print("\nStatement 6 Analysis:")
print("Zero covariance between X and Y does not guarantee they are statistically independent:")
print("- Left plot: Truly independent variables")
print(f"  • X and Y are independent Gaussian random variables")
print(f"  • Their covariance is close to zero: {cov_indep:.4f}")
print(f"  • Their correlation is close to zero: {corr_indep:.4f}")
print("  • No pattern is visible in the scatter plot")
print("- Middle plot: Dependent variables with zero covariance")
print(f"  • Y depends on X through a quadratic relationship (Y ≈ X²)")
print(f"  • Their covariance is close to zero: {cov_dep:.4f}")
print(f"  • Their correlation is close to zero: {corr_dep:.4f}")
print("  • A clear quadratic pattern is visible (red dashed curve)")
print("  • These variables are strongly dependent despite zero covariance")
print("- Right plot: Difference between joint distribution and product of marginals")
print("  • For independent variables, this difference would be zero everywhere")
print("  • For the quadratic relationship, we see significant deviations (non-zero values)")
print("  • This confirms that X and Y are not independent")
print("\nOnly for multivariate normal distributions does zero covariance imply independence.")

print_statement_result(6, False, "Zero covariance does not guarantee statistical independence. Variables can have zero covariance yet be dependent through non-linear relationships.")

# Step 8: Covariance formula and self-covariance
print_step_header(8, "Covariance Formula and Self-Covariance")

# Generate bivariate data
np.random.seed(42)
n_samples = 1000
x = np.random.normal(0, 1, n_samples)
y = 0.7 * x + 0.5 * np.random.normal(0, 1, n_samples)

# Compute covariance and correlation using different formulas
mean_x = np.mean(x)
mean_y = np.mean(y)

# Method 1: E[(X-E[X])(Y-E[Y])]
cov_method1 = np.mean((x - mean_x) * (y - mean_y))

# Method 2: E[XY] - E[X]E[Y]
cov_method2 = np.mean(x * y) - mean_x * mean_y

# Compute variance as self-covariance
var_x_cov = np.cov(x, x)[0, 1]  # This should equal Var(X)
var_x_direct = np.var(x)

# Create visualization
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot the data
axs[0].scatter(x, y, alpha=0.3)
axs[0].set_title("Bivariate Data")
axs[0].set_xlabel("X")
axs[0].set_ylabel("Y")
axs[0].grid(True)

# Visualize the computational equivalence with a scatter plot
axs[1].scatter((x - mean_x) * (y - mean_y), x * y - mean_x * mean_y, alpha=0.3)
axs[1].set_title("Covariance Formula Equivalence")
axs[1].set_xlabel("(X-E[X])(Y-E[Y])")
axs[1].set_ylabel("XY - E[X]E[Y]")
axs[1].grid(True)

# Add y=x line to show they're equal
min_val = min(np.min((x - mean_x) * (y - mean_y)), np.min(x * y - mean_x * mean_y))
max_val = max(np.max((x - mean_x) * (y - mean_y)), np.max(x * y - mean_x * mean_y))
axs[1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)

plt.tight_layout()
file_path = os.path.join(save_dir, "7_covariance_formula.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Detailed explanation in text
print("\nStatement 7 and 8 Analysis:")
print("Covariance calculation methods:")
print(f"1. Using definition E[(X-E[X])(Y-E[Y])] = {cov_method1:.4f}")
print(f"2. Using algebraic form E[XY] - E[X]E[Y] = {cov_method2:.4f}")
print("The right plot shows each point's contribution calculated both ways - they're identical (points fall on y=x line).")
print("\nSelf-covariance equals variance:")
print(f"- Cov(X,X) = {var_x_cov:.4f}")
print(f"- Var(X) = {var_x_direct:.4f}")
print("\nMathematical proof:")
print("Cov(X,X) = E[(X-E[X])(X-E[X])] = E[(X-E[X])²] = Var(X)")
print("Alternative form: Cov(X,X) = E[XX] - E[X]E[X] = E[X²] - (E[X])² = Var(X)")

print_statement_result(7, True, "The covariance between X and Y can be calculated using either formula: Cov(X,Y) = E[(X-E[X])(Y-E[Y])] = E[XY] - E[X]E[Y].")
print_statement_result(8, True, "The covariance of a random variable with itself equals its variance: Cov(X,X) = Var(X).")

# Step 9: Negative correlation and contour orientation
print_step_header(9, "Negative Correlation and Contour Orientation")

# Create a range of correlation values from negative to positive
correlations = [-0.8, 0.0, 0.8]  # Negative, Zero, Positive
titles = ["Negative Correlation", "Zero Correlation", "Positive Correlation"]
colors = ['red', 'green', 'blue']

# Create a figure with multiple subplots
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# Define the mean and standard deviations
mean = [0, 0]
std_devs = [1, 1]  # Equal standard deviations

# Create a meshgrid for plotting contours
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Generate contour plots and scatter plots
for i, (correlation, title, color) in enumerate(zip(correlations, titles, colors)):
    # Create the covariance matrix
    cov = [[std_devs[0]**2, correlation*std_devs[0]*std_devs[1]], 
           [correlation*std_devs[0]*std_devs[1], std_devs[1]**2]]
    
    # Generate multivariate normal distribution
    rv = multivariate_normal(mean, cov)
    Z = rv.pdf(pos)
    
    # Generate samples
    samples = np.random.multivariate_normal(mean, cov, 500)
    
    # Plot density contours
    ax_top = axs[0, i]
    ax_top.contour(X, Y, Z, levels=10, colors=color)
    ax_top.set_title(f"{title}\nρ = {correlation}")
    ax_top.set_xlabel("X")
    ax_top.set_ylabel("Y")
    ax_top.grid(True)
    ax_top.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax_top.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax_top.set_aspect('equal')
    
    # Plot sample points
    ax_bottom = axs[1, i]
    ax_bottom.scatter(samples[:, 0], samples[:, 1], alpha=0.3, color=color)
    ax_bottom.set_title(f"Samples with {title}")
    ax_bottom.set_xlabel("X")
    ax_bottom.set_ylabel("Y")
    ax_bottom.grid(True)
    ax_bottom.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax_bottom.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax_bottom.set_aspect('equal')
    
    # Draw the principal axes (eigenvectors) of the covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # Scale eigenvectors by eigenvalues for visualization
    for j, eigvec in enumerate(eigvecs.T):
        ax_bottom.arrow(0, 0, eigvec[0] * np.sqrt(eigvals[j]), eigvec[1] * np.sqrt(eigvals[j]), 
                     head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Add reference lines for y = x and y = -x
    ax_bottom.plot([-3, 3], [3, -3], 'k:', alpha=0.3, label='y = -x')
    ax_bottom.plot([-3, 3], [-3, 3], 'k:', alpha=0.3, label='y = x')

plt.tight_layout()
file_path = os.path.join(save_dir, "8_negative_correlation_principal_axes.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Detailed explanation in text
print("\nStatement 9 and 10 Analysis:")
print("Correlation and contour orientation:")
print("- Left column: Negative correlation (ρ = -0.8)")
print("  • Contours are tilted along the line y = -x")
print("  • When X increases, Y tends to decrease (negative relationship)")
print("  • Principal axes (eigenvectors shown as black arrows) align with the contour ellipse axes")
print("- Middle column: Zero correlation (ρ = 0.0)")
print("  • Contours are axis-aligned (no tilt)")
print("  • X and Y vary independently")
print("  • Principal axes align with the coordinate axes")
print("- Right column: Positive correlation (ρ = 0.8)")
print("  • Contours are tilted along the line y = x")
print("  • When X increases, Y tends to increase (positive relationship)")
print("  • Principal axes align with the contour ellipse axes")
print("\nThe principal axes of the probability contours align with the eigenvectors of the covariance matrix.")
print("The length of each arrow is proportional to the square root of the corresponding eigenvalue.")

print_statement_result(9, True, "In a bivariate normal distribution, negative correlation corresponds to probability density contours being tilted primarily along the line y = -x.")
print_statement_result(10, True, "The principal axes of the probability density contours for a multivariate normal distribution align with the eigenvectors of its covariance matrix.")

# Step 10: Contour lines
print_step_header(10, "Contour Lines and Probability Density")

# Create a 3D plot with contour lines
fig = plt.figure(figsize=(15, 6))
gs = GridSpec(1, 2, width_ratios=[1.5, 1])

# Create 3D surface plot in the first subplot
ax1 = fig.add_subplot(gs[0], projection='3d')

# Define the bivariate normal parameters
mean = [0, 0]
cov = [[1, 0.7], [0.7, 1]]

# Create a meshgrid for plotting
x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Calculate PDF values
rv = multivariate_normal(mean, cov)
Z = rv.pdf(pos)

# Plot the 3D surface
surf = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.8)
ax1.set_title("3D Probability Density Surface")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Probability Density")

# Create contour plot in the second subplot
ax2 = fig.add_subplot(gs[1])
contour = ax2.contour(X, Y, Z, levels=10, colors='blue')
ax2.set_title("Contour Plot of the PDF")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.grid(True)
plt.colorbar(contour, ax=ax2, label="Probability Density")

plt.tight_layout()
file_path = os.path.join(save_dir, "9_contour_lines.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Detailed explanation in text
print("\nStatement 11 Analysis:")
print("Contour lines on a probability density plot:")
print("- Left plot: 3D surface of a bivariate normal probability density function")
print("  • The height of the surface represents the probability density at each point (X,Y)")
print("  • The peak occurs at the mean of the distribution")
print("- Right plot: Contour plot of the same probability density function")
print("  • Each contour line connects points having the same probability density value")
print("  • These can be thought of as horizontal 'slices' through the 3D surface")
print("  • Closely spaced contours indicate steep changes in probability density")
print("  • For multivariate normal distributions, these contours form ellipses")

print_statement_result(11, True, "Contour lines on a probability density plot connect points having the same probability density value.")

# Step 11: Volume of ellipsoid and eigenvalues
print_step_header(11, "Volume of Ellipsoid and Eigenvalues")

# Create ellipsoids with different eigenvalue combinations
# Two pairs with same sum but different products
eig_pair1 = [4, 1]  # Sum = 5, Product = 4
eig_pair2 = [2.5, 2.5]  # Sum = 5, Product = 6.25

# Create corresponding covariance matrices
cov1, _, _ = create_matrix_with_eigenvalues(eig_pair1, rotation=30)
cov2, _, _ = create_matrix_with_eigenvalues(eig_pair2, rotation=30)

# Calculate volumes (proportional to sqrt of determinant)
vol1 = np.sqrt(np.linalg.det(cov1))
vol2 = np.sqrt(np.linalg.det(cov2))

# Create meshgrid for plotting
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Create figure for visualization
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot first ellipse
mean = [0, 0]
rv1 = multivariate_normal(mean, cov1)
Z1 = rv1.pdf(pos)

axs[0].contour(X, Y, Z1, levels=10, colors='blue')
axs[0].contour(X, Y, Z1, levels=[rv1.pdf(np.array([0, 0])) / np.e], colors='red', linewidths=2)
axs[0].set_title(f"λ₁ = {eig_pair1[0]}, λ₂ = {eig_pair1[1]}\nSum = {sum(eig_pair1)}, Product = {eig_pair1[0]*eig_pair1[1]}\nVolume ∝ √det = {vol1:.2f}")
axs[0].set_xlabel("X")
axs[0].set_ylabel("Y")
axs[0].grid(True)
axs[0].set_aspect('equal')

# Plot second ellipse
rv2 = multivariate_normal(mean, cov2)
Z2 = rv2.pdf(pos)

axs[1].contour(X, Y, Z2, levels=10, colors='green')
axs[1].contour(X, Y, Z2, levels=[rv2.pdf(np.array([0, 0])) / np.e], colors='red', linewidths=2)
axs[1].set_title(f"λ₁ = {eig_pair2[0]}, λ₂ = {eig_pair2[1]}\nSum = {sum(eig_pair2)}, Product = {eig_pair2[0]*eig_pair2[1]}\nVolume ∝ √det = {vol2:.2f}")
axs[1].set_xlabel("X")
axs[1].set_ylabel("Y")
axs[1].grid(True)
axs[1].set_aspect('equal')

plt.tight_layout()
file_path = os.path.join(save_dir, "10_ellipsoid_volume.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Detailed explanation in text
print("\nStatement 12 Analysis:")
print("Volume of the standard deviation ellipsoid and eigenvalues:")
print("- The volume of an n-dimensional ellipsoid is proportional to the product of its semi-axes lengths")
print("- For a covariance matrix, the semi-axes lengths are proportional to the square roots of eigenvalues")
print("- Therefore, the volume is proportional to the square root of the determinant, which equals the square root of the product of eigenvalues")
print(f"- Left ellipse: eigenvalues λ₁ = {eig_pair1[0]}, λ₂ = {eig_pair1[1]}")
print(f"  • Sum of eigenvalues = {sum(eig_pair1)}")
print(f"  • Product of eigenvalues = {eig_pair1[0]*eig_pair1[1]}")
print(f"  • Volume proportional to √(λ₁·λ₂) = {vol1:.2f}")
print(f"- Right ellipse: eigenvalues λ₁ = {eig_pair2[0]}, λ₂ = {eig_pair2[1]}")
print(f"  • Sum of eigenvalues = {sum(eig_pair2)}")
print(f"  • Product of eigenvalues = {eig_pair2[0]*eig_pair2[1]}")
print(f"  • Volume proportional to √(λ₁·λ₂) = {vol2:.2f}")
print("- Both ellipses have the same sum of eigenvalues, but different products and therefore different volumes")
print("- The red contour shows one standard deviation from the mean (corresponds to the standard deviation ellipsoid)")

print_statement_result(12, False, "The volume of the ellipsoid representing the region within one standard deviation is proportional to the square root of the product of eigenvalues (determinant), not the sum of eigenvalues (trace).")

# Step 13: Angle of rotation of probability density contours
print_step_header(13, "Angle of Rotation of Probability Density Contours")

# Function to calculate angle of rotation given covariance parameters
def calculate_angle(sigma_x, sigma_y, sigma_xy):
    if sigma_x**2 == sigma_y**2:
        # For equal variances, the angle depends only on the sign of covariance
        if sigma_xy > 0:
            return 45  # pi/4 radians
        elif sigma_xy < 0:
            return 135  # 3*pi/4 radians
        else:
            return "undefined (circular contours)"
    else:
        # Standard formula for unequal variances
        return 0.5 * np.arctan2(2*sigma_xy, sigma_x**2 - sigma_y**2) * 180 / np.pi

# Create a set of covariance matrices with different parameters
# Case 1: Unequal variances
sigma_x1, sigma_y1, sigma_xy1 = 2.0, 0.5, 0.5
cov1 = np.array([[sigma_x1**2, sigma_xy1], [sigma_xy1, sigma_y1**2]])
angle1 = calculate_angle(sigma_x1, sigma_y1, sigma_xy1)

# Case 2: Equal variances, positive covariance
sigma_x2, sigma_y2, sigma_xy2 = 1.0, 1.0, 0.5
cov2 = np.array([[sigma_x2**2, sigma_xy2], [sigma_xy2, sigma_y2**2]])
angle2 = calculate_angle(sigma_x2, sigma_y2, sigma_xy2)

# Case 3: Equal variances, negative covariance
sigma_x3, sigma_y3, sigma_xy3 = 1.0, 1.0, -0.5
cov3 = np.array([[sigma_x3**2, sigma_xy3], [sigma_xy3, sigma_y3**2]])
angle3 = calculate_angle(sigma_x3, sigma_y3, sigma_xy3)

# Create a figure for visualization
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Create meshgrid for plotting
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Plot case 1: Unequal variances
mean = [0, 0]
rv1 = multivariate_normal(mean, cov1)
Z1 = rv1.pdf(pos)

axs[0].contour(X, Y, Z1, levels=10, colors='blue')
axs[0].set_title(f"Unequal Variances\nσ²ₓ = {sigma_x1**2}, σ²ᵧ = {sigma_y1**2}, σₓᵧ = {sigma_xy1}")
axs[0].set_xlabel("X")
axs[0].set_ylabel("Y")
axs[0].grid(True)
axs[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axs[0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axs[0].set_aspect('equal')

# Add the calculated angle line
angle_rad = angle1 * np.pi / 180
axs[0].plot([0, np.cos(angle_rad)*3], [0, np.sin(angle_rad)*3], 'r-', linewidth=2)
axs[0].text(2.2, 1, f"θ = {angle1:.1f}°", color='red')

# Plot case 2: Equal variances, positive covariance
rv2 = multivariate_normal(mean, cov2)
Z2 = rv2.pdf(pos)

axs[1].contour(X, Y, Z2, levels=10, colors='green')
axs[1].set_title(f"Equal Variances, Positive Covariance\nσ²ₓ = {sigma_x2**2}, σ²ᵧ = {sigma_y2**2}, σₓᵧ = {sigma_xy2}")
axs[1].set_xlabel("X")
axs[1].set_ylabel("Y")
axs[1].grid(True)
axs[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axs[1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axs[1].set_aspect('equal')

# Add y = x line (45 degrees)
axs[1].plot([-3, 3], [-3, 3], 'r-', linewidth=2)
axs[1].text(2, 2, "θ = 45°", color='red')

# Plot case 3: Equal variances, negative covariance
rv3 = multivariate_normal(mean, cov3)
Z3 = rv3.pdf(pos)

axs[2].contour(X, Y, Z3, levels=10, colors='red')
axs[2].set_title(f"Equal Variances, Negative Covariance\nσ²ₓ = {sigma_x3**2}, σ²ᵧ = {sigma_y3**2}, σₓᵧ = {sigma_xy3}")
axs[2].set_xlabel("X")
axs[2].set_ylabel("Y")
axs[2].grid(True)
axs[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axs[2].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axs[2].set_aspect('equal')

# Add y = -x line (135 degrees)
axs[2].plot([-3, 3], [3, -3], 'r-', linewidth=2)
axs[2].text(2, -2, "θ = 135°", color='red')

plt.tight_layout()
file_path = os.path.join(save_dir, "11_rotation_angles.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Detailed explanation in text
print("\nStatement 13 Analysis:")
print("Angle of rotation of probability density contours:")
print("- The formula θ = (1/2)tan⁻¹(2σₓᵧ/(σₓ²-σᵧ²)) only applies when variances are unequal (σₓ² ≠ σᵧ²)")
print("- Left plot: Unequal variances")
print(f"  • σₓ² = {sigma_x1**2}, σᵧ² = {sigma_y1**2}, σₓᵧ = {sigma_xy1}")
print(f"  • Using the formula: θ = {angle1:.1f}°")
print(f"  • The red line shows this calculated angle")
print("- Middle plot: Equal variances, positive covariance")
print(f"  • σₓ² = {sigma_x2**2}, σᵧ² = {sigma_y2**2}, σₓᵧ = {sigma_xy2}")
print(f"  • The formula is undefined (division by zero)")
print(f"  • With equal variances and positive covariance, θ is always 45°")
print(f"  • The contours align with the line y = x (red line)")
print("- Right plot: Equal variances, negative covariance")
print(f"  • σₓ² = {sigma_x3**2}, σᵧ² = {sigma_y3**2}, σₓᵧ = {sigma_xy3}")
print(f"  • The formula is undefined (division by zero)")
print(f"  • With equal variances and negative covariance, θ is always 135°")
print(f"  • The contours align with the line y = -x (red line)")
print("\nIn the special case where variances are equal, the orientation depends only on the sign of the covariance:")
print("- If σₓᵧ > 0, θ = 45° (along y = x)")
print("- If σₓᵧ < 0, θ = 135° (along y = -x)")
print("- If σₓᵧ = 0, contours are circular with no preferred orientation")

print_statement_result(13, False, "The given formula for the angle of rotation is not valid when σₓ² = σᵧ². In this special case, the orientation is determined only by the sign of σₓᵧ.")

# Step 14: Summarize all statements
print_step_header(14, "Summary of All Statements")

statements = [
    "1. For a multivariate normal distribution, a diagonal covariance matrix implies that the variables are uncorrelated, resulting in probability density contours that are axis-aligned ellipses (or circles if variances are equal).",
    "2. Covariance measures the tendency of two random variables to vary together; a positive value indicates they tend to increase or decrease together, while a negative value indicates one tends to increase as the other decreases.",
    "3. All valid covariance matrices must be positive semi-definite, meaning Var(a^T X) = a^T Σ a ≥ 0 for any vector a.",
    "4. A covariance matrix is strictly positive definite if and only if all its eigenvalues are strictly positive; this condition guarantees the matrix is invertible.",
    "5. Covariance only quantifies the strength and direction of the linear relationship between two random variables.",
    "6. Zero covariance (Cov(X,Y) = 0) guarantees that the random variables X and Y are statistically independent.",
    "7. The covariance between X and Y can be calculated using the formula Cov(X,Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y].",
    "8. The covariance of a random variable X with itself, Cov(X,X), is equal to its variance, Var(X).",
    "9. In a bivariate normal distribution, negative correlation corresponds to probability density contours being tilted primarily along the line y = -x.",
    "10. The principal axes of the probability density contours for a multivariate normal distribution align with the eigenvectors of its covariance matrix.",
    "11. Contour lines on a probability density plot connect points having the same probability density value.",
    "12. For any n×n covariance matrix with eigenvalues λ₁ ≥ λ₂ ≥ ... ≥ λₙ, the volume of the ellipsoid representing the region within one standard deviation is directly proportional to the sum of eigenvalues rather than their product.",
    "13. In a multivariate normal distribution, the angle of rotation of probability density contours in a 2D plane is always given by θ = (1/2)tan⁻¹(2σₓᵧ/(σₓ²-σᵧ²)), regardless of whether σₓ² = σᵧ²."
]

verdicts = [
    "TRUE",
    "TRUE",
    "TRUE",
    "TRUE",
    "TRUE",
    "FALSE",
    "TRUE",
    "TRUE",
    "TRUE",
    "TRUE",
    "TRUE",
    "FALSE",
    "FALSE"
]

explanations = [
    "A diagonal covariance matrix means zero correlation between variables, resulting in axis-aligned contours.",
    "Covariance measures linear relationships between variables with sign indicating direction.",
    "All valid covariance matrices must be positive semi-definite to ensure non-negative variances.",
    "A matrix is positive definite if and only if all eigenvalues are positive, which guarantees invertibility.",
    "Covariance only captures linear relationships, not more complex nonlinear dependencies.",
    "Zero covariance only implies no linear relationship. Variables can still be dependent in nonlinear ways.",
    "These two formulas for covariance are mathematically equivalent.",
    "The variance is a special case of covariance when the two variables are identical.",
    "Negative correlation results in contours aligned along the negative slope diagonal.",
    "Eigenvectors of the covariance matrix define the directions of maximum and minimum variance.",
    "Contour lines represent 'slices' of constant height through the probability density function.",
    "Ellipsoid volume is proportional to the square root of the product (determinant), not the sum (trace) of eigenvalues.",
    "The formula is undefined when variances are equal due to division by zero."
]

print("\n\nSummary of Verdicts:")
print("-" * 100)
print(f"{'Statement':<80} {'Verdict':<10} {'Brief Explanation'}")
print("-" * 100)
for i, (statement, verdict, explanation) in enumerate(zip(statements, verdicts, explanations)):
    print(f"{i+1}. {statement:<78} {verdict:<10} {explanation}")
print("-" * 100)

# Count true and false statements
true_count = sum([1 for v in verdicts if v == "TRUE"])
false_count = sum([1 for v in verdicts if v == "FALSE"])

print(f"\nFinal count: {true_count} TRUE statements, {false_count} FALSE statements")
print(f"The FALSE statements are 6, 12, and 13.")

print("\nThis completes the exploration of covariance concepts and properties.") 