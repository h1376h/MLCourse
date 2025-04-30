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

# Create a figure with three covariance matrices: diagonal with equal variances, 
# diagonal with unequal variances, and non-diagonal
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
axs[0].set_title("Diagonal Covariance with Equal Variances\n(Circle)")
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
axs[1].set_title("Diagonal Covariance with Unequal Variances\n(Axis-Aligned Ellipse)")
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
axs[2].set_title("Non-Diagonal Covariance\n(Tilted Ellipse)")
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

print("\nKey Insights for Statement 1:")
print("- When a covariance matrix is diagonal (all off-diagonal elements are zero), the variables are uncorrelated")
print("- This results in axis-aligned contours in the probability density function")
print("- If the diagonal elements are equal, the contours form circles")
print("- If the diagonal elements are unequal, the contours form axis-aligned ellipses")
print("- Non-diagonal covariance matrices result in tilted ellipses")
print("\nStatement 1 is TRUE. A diagonal covariance matrix means variables are uncorrelated, and contours are axis-aligned.")

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
axs[0].set_title(f"Positive Covariance (Cov = {cov_pos[0][1]:.1f})")
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
axs[0].text(-3.5, 3, "Variables increase\nand decrease together", bbox=dict(facecolor='white', alpha=0.8))

# Plot negative covariance
axs[1].scatter(samples_neg[:, 0], samples_neg[:, 1], alpha=0.5)
axs[1].set_title(f"Negative Covariance (Cov = {cov_neg[0][1]:.1f})")
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
axs[1].text(-3.5, 3, "One variable increases\nas the other decreases", bbox=dict(facecolor='white', alpha=0.8))

# Plot zero covariance
axs[2].scatter(samples_zero[:, 0], samples_zero[:, 1], alpha=0.5)
axs[2].set_title(f"Zero Covariance (Cov = {cov_zero[0][1]:.1f})")
axs[2].set_xlabel("X")
axs[2].set_ylabel("Y")
axs[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axs[2].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axs[2].set_xlim(-4, 4)
axs[2].set_ylim(-4, 4)
axs[2].grid(True)
axs[2].text(-3.5, 3, "No linear relationship\nbetween variables", bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
file_path = os.path.join(save_dir, "2_positive_negative_covariance.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

print("\nKey Insights for Statement 2:")
print("- Positive covariance means variables tend to increase or decrease together")
print("- Negative covariance means one variable tends to increase while the other decreases")
print("- Zero covariance indicates no linear relationship between variables")
print("- The magnitude of covariance indicates the strength of this linear relationship")
print("\nStatement 2 is TRUE. Covariance measures how variables vary together, with positive/negative values indicating their directional relationship.")

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
axs[0].set_title("Positive Definite Matrix\nQuadratic Form a^T Σ a")
axs[0].set_xlabel("a1")
axs[0].set_ylabel("a2")
axs[0].grid(True)
plt.colorbar(contour1, ax=axs[0])
axs[0].contour(A1, A2, Z1, levels=[0], colors='red', linewidths=2)
axs[0].text(-1.8, 1.8, "All values > 0", bbox=dict(facecolor='white', alpha=0.8))

# Plot the positive semi-definite case
contour2 = axs[1].contourf(A1, A2, Z2, 20, cmap=cmap)
axs[1].set_title("Positive Semi-Definite Matrix\nQuadratic Form a^T Σ a")
axs[1].set_xlabel("a1")
axs[1].set_ylabel("a2")
axs[1].grid(True)
plt.colorbar(contour2, ax=axs[1])
axs[1].contour(A1, A2, Z2, levels=[0], colors='red', linewidths=2)
axs[1].text(-1.8, 1.8, "All values ≥ 0", bbox=dict(facecolor='white', alpha=0.8))

# Plot the indefinite case
contour3 = axs[2].contourf(A1, A2, Z3, 20, cmap=cmap)
axs[2].set_title("Indefinite Matrix\nQuadratic Form a^T Σ a")
axs[2].set_xlabel("a1")
axs[2].set_ylabel("a2")
axs[2].grid(True)
plt.colorbar(contour3, ax=axs[2])
axs[2].contour(A1, A2, Z3, levels=[0], colors='red', linewidths=2)
axs[2].text(-1.8, 1.8, "Some values < 0\n(Not a valid covariance)", bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
file_path = os.path.join(save_dir, "3_positive_semi_definite.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

print("\nKey Insights for Statement 3:")
print("- Valid covariance matrices must be positive semi-definite, meaning a^T Σ a ≥ 0 for any vector a")
print("- Positive semi-definite matrices have non-negative eigenvalues")
print("- This property ensures variances are always non-negative (since a^T Σ a = Var(a^T X))")
print("- A positive semi-definite matrix can have some eigenvalues equal to zero")
print("- Indefinite matrices, with some negative eigenvalues, cannot be valid covariance matrices")
print("\nStatement 3 is TRUE. All valid covariance matrices must be positive semi-definite to ensure variances are non-negative.")

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
ax1.set_title("Positive Definite\n(All eigenvalues > 0)")
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

ax1.text(-2.5, 2.5, f"Det(Σ) = {np.linalg.det(matrix_1):.2f}\nInvertible: Yes", 
         bbox=dict(facecolor='white', alpha=0.8))
ax1.legend()
ax1.grid(True)
ax1.set_aspect('equal')

# Case 2: Positive semi-definite (one eigenvalue = 0)
rv2 = multivariate_normal(mean, matrix_2 + 1e-10 * np.eye(2), allow_singular=True)  # Add small epsilon for numerical stability
Z2 = rv2.pdf(pos)

# Plot contours
ax2 = axs[1]
ax2.contour(X, Y, Z2, levels=10, colors='blue')
ax2.set_title("Positive Semi-Definite\n(Some eigenvalues = 0)")
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

ax2.text(-2.5, 2.5, f"Det(Σ) = {np.linalg.det(matrix_2):.2f}\nInvertible: No", 
         bbox=dict(facecolor='white', alpha=0.8))
ax2.legend()
ax2.grid(True)
ax2.set_aspect('equal')

# Case 3: Visual representation of invertibility
# Create a custom matrix for demonstration
custom_matrix = np.array([[1.0, 0.5], [0.5, 0.3]])  # Low condition number, poorly conditioned
eigvals, eigvecs = np.linalg.eigh(custom_matrix)

ax3 = axs[2]
ax3.set_title("Eigenvalue Magnitudes and Invertibility")
ax3.set_xlabel("Eigenvalue Index")
ax3.set_ylabel("Eigenvalue Magnitude")
ax3.bar([1, 2], eigvals, color=['red', 'green'])
ax3.axhline(y=0, color='k', linestyle='-', alpha=0.5)
ax3.set_xticks([1, 2])
ax3.set_xticklabels(['λ₁', 'λ₂'])
ax3.grid(True)

# Add condition number and other information
cond_num = np.linalg.cond(custom_matrix)
ax3.text(1.5, eigvals.max()/2, f"Condition Number: {cond_num:.2f}\n" + 
         f"Det(Σ) = {np.linalg.det(custom_matrix):.2f}\n" +
         f"All λ > 0: {np.all(eigvals > 0)}\n" +
         f"Invertible: {np.linalg.det(custom_matrix) != 0}", 
         bbox=dict(facecolor='white', alpha=0.8), ha='center')

plt.tight_layout()
file_path = os.path.join(save_dir, "4_eigenvalues_definiteness.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

print("\nKey Insights for Statement 4:")
print("- A covariance matrix is strictly positive definite if and only if all eigenvalues are strictly positive")
print("- Positive definiteness guarantees the matrix is invertible")
print("- If any eigenvalue is zero, the matrix is only positive semi-definite and not invertible")
print("- The determinant of a matrix is the product of its eigenvalues, so positive eigenvalues ensure non-zero determinant")
print("- In a multivariate normal distribution, positive definiteness is required for calculating the density function")
print("\nStatement 4 is TRUE. A covariance matrix is strictly positive definite iff all eigenvalues are positive, making it invertible.")

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

# Center the data 
x_nonlin_centered = x_nonlin - np.mean(x_nonlin)
y_nonlin_centered = y_nonlin - np.mean(y_nonlin)

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
axs[0, 0].set_title(f"Linear Relationship\nCovariance = {cov_linear:.3f}, Correlation = {corr_linear:.3f}")
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
axs[0, 1].set_title(f"Non-linear Relationship\nCovariance = {cov_nonlinear:.3f}, Correlation = {corr_nonlinear:.3f}")
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
axs[1, 0].set_title("Centered Linear Data\nE[(X-E[X])(Y-E[Y])]")
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

axs[1, 0].scatter((x_lin - E_x_lin)[q1], (y_lin - E_y_lin)[q1], color='green', alpha=0.5, label='+ contribution')
axs[1, 0].scatter((x_lin - E_x_lin)[q3], (y_lin - E_y_lin)[q3], color='green', alpha=0.5)
axs[1, 0].scatter((x_lin - E_x_lin)[q2], (y_lin - E_y_lin)[q2], color='red', alpha=0.5, label='- contribution')
axs[1, 0].scatter((x_lin - E_x_lin)[q4], (y_lin - E_y_lin)[q4], color='red', alpha=0.5)

# Calculate the contribution of each quadrant to covariance
cov_q1 = np.mean((x_lin[q1] - E_x_lin) * (y_lin[q1] - E_y_lin)) * np.sum(q1) / n_samples
cov_q2 = np.mean((x_lin[q2] - E_x_lin) * (y_lin[q2] - E_y_lin)) * np.sum(q2) / n_samples
cov_q3 = np.mean((x_lin[q3] - E_x_lin) * (y_lin[q3] - E_y_lin)) * np.sum(q3) / n_samples
cov_q4 = np.mean((x_lin[q4] - E_x_lin) * (y_lin[q4] - E_y_lin)) * np.sum(q4) / n_samples

axs[1, 0].legend()
axs[1, 0].text(-2, 2, f"Quadrant contributions to covariance:\nQ1 (green): {cov_q1:.3f}\nQ3 (green): {cov_q3:.3f}\nQ2 (red): {cov_q2:.3f}\nQ4 (red): {cov_q4:.3f}\nTotal: {cov_q1+cov_q2+cov_q3+cov_q4:.3f}", 
               bbox=dict(facecolor='white', alpha=0.8))

# Visualize the covariance computation for non-linear relationship
axs[1, 1].scatter(x_nonlin - E_x_nonlin, y_nonlin - E_y_nonlin, alpha=0.5)
axs[1, 1].set_title("Centered Non-linear Data\nE[(X-E[X])(Y-E[Y])]")
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

axs[1, 1].scatter((x_nonlin - E_x_nonlin)[q1_nonlin], (y_nonlin - E_y_nonlin)[q1_nonlin], color='green', alpha=0.5, label='+ contribution')
axs[1, 1].scatter((x_nonlin - E_x_nonlin)[q3_nonlin], (y_nonlin - E_y_nonlin)[q3_nonlin], color='green', alpha=0.5)
axs[1, 1].scatter((x_nonlin - E_x_nonlin)[q2_nonlin], (y_nonlin - E_y_nonlin)[q2_nonlin], color='red', alpha=0.5, label='- contribution')
axs[1, 1].scatter((x_nonlin - E_x_nonlin)[q4_nonlin], (y_nonlin - E_y_nonlin)[q4_nonlin], color='red', alpha=0.5)

# Calculate the contribution of each quadrant to covariance for non-linear data
cov_q1_nonlin = np.mean((x_nonlin[q1_nonlin] - E_x_nonlin) * (y_nonlin[q1_nonlin] - E_y_nonlin)) * np.sum(q1_nonlin) / n_samples if np.sum(q1_nonlin) > 0 else 0
cov_q2_nonlin = np.mean((x_nonlin[q2_nonlin] - E_x_nonlin) * (y_nonlin[q2_nonlin] - E_y_nonlin)) * np.sum(q2_nonlin) / n_samples if np.sum(q2_nonlin) > 0 else 0
cov_q3_nonlin = np.mean((x_nonlin[q3_nonlin] - E_x_nonlin) * (y_nonlin[q3_nonlin] - E_y_nonlin)) * np.sum(q3_nonlin) / n_samples if np.sum(q3_nonlin) > 0 else 0
cov_q4_nonlin = np.mean((x_nonlin[q4_nonlin] - E_x_nonlin) * (y_nonlin[q4_nonlin] - E_y_nonlin)) * np.sum(q4_nonlin) / n_samples if np.sum(q4_nonlin) > 0 else 0

axs[1, 1].legend()
axs[1, 1].text(-2, 5, f"Quadrant contributions to covariance:\nQ1 (green): {cov_q1_nonlin:.3f}\nQ3 (green): {cov_q3_nonlin:.3f}\nQ2 (red): {cov_q2_nonlin:.3f}\nQ4 (red): {cov_q4_nonlin:.3f}\nTotal: {cov_q1_nonlin+cov_q2_nonlin+cov_q3_nonlin+cov_q4_nonlin:.3f}", 
               bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
file_path = os.path.join(save_dir, "5_linear_relationship_covariance.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

print("\nKey Insights for Statement 5:")
print("- Covariance quantifies only the linear relationship between variables")
print("- When a relationship is perfectly linear, covariance/correlation accurately capture the relationship's strength")
print("- For non-linear relationships (like quadratic), covariance may be low or even zero despite strong dependence")
print("- Covariance measures the average product of centered variables: E[(X-E[X])(Y-E[Y])]")
print("- Positive contributions come from quadrants where both variables are above or below their means")
print("- Negative contributions come from quadrants where one variable is above while the other is below its mean")
print("\nStatement 5 is TRUE. Covariance only quantifies the strength and direction of linear relationships between variables.")

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
axs[0].set_title(f"Independent Variables\nCov(X,Y) = {cov_indep:.4f}, Corr = {corr_indep:.4f}")
axs[0].set_xlabel("X")
axs[0].set_ylabel("Y")
axs[0].set_xlim(-4, 4)
axs[0].set_ylim(-4, 4)
axs[0].grid(True)
axs[0].text(-3.5, 3.5, "X and Y are independent\nand have zero covariance", bbox=dict(facecolor='white', alpha=0.8))

# Plot dependent variables with zero covariance
axs[1].scatter(x_dep, y_dep, alpha=0.3)
axs[1].set_title(f"Dependent Variables with Zero Covariance\nCov(X,Y) = {cov_dep:.4f}, Corr = {corr_dep:.4f}")
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
axs[1].text(-3.5, 3.5, "X and Y are dependent\nbut have zero covariance", bbox=dict(facecolor='white', alpha=0.8))

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
axs[2].set_title("Difference: Joint Distribution - Product of Marginals\nShould be zero everywhere for independence")
axs[2].set_xlabel("X")
axs[2].set_ylabel("Y")
axs[2].grid(True)
plt.colorbar(c, ax=axs[2])

plt.tight_layout()
file_path = os.path.join(save_dir, "6_zero_covariance_vs_independence.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

print("\nKey Insights for Statement 6:")
print("- Zero covariance between X and Y means they have no linear relationship")
print("- Independence of X and Y implies zero covariance, but zero covariance does not imply independence")
print("- Variables can be dependent yet have zero covariance (e.g., quadratic relationship y = x²)")
print("- For independent variables, the joint distribution equals the product of marginal distributions")
print("- For dependent variables with zero covariance, the joint distribution differs from the product of marginals")
print("- Only for certain distributions (like multivariate normal) does zero covariance imply independence")
print("\nStatement 6 is FALSE. Zero covariance does not guarantee statistical independence, only the absence of linear relationship.")

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

# Highlight the different formulas
formula_text = (
    f"Covariance Calculation Methods:\n\n"
    f"1. E[(X-E[X])(Y-E[Y])] = {cov_method1:.4f}\n"
    f"2. E[XY] - E[X]E[Y] = {cov_method2:.4f}\n\n"
    f"Self-Covariance equals Variance:\n"
    f"Cov(X,X) = {var_x_cov:.4f}\n"
    f"Var(X) = {var_x_direct:.4f}"
)
axs[0].text(2, -2, formula_text, bbox=dict(facecolor='white', alpha=0.8))

# Visualize the computational equivalence with a scatter plot
axs[1].scatter((x - mean_x) * (y - mean_y), x * y - mean_x * mean_y, alpha=0.3)
axs[1].set_title("Equivalence of Covariance Formulas")
axs[1].set_xlabel("(X-E[X])(Y-E[Y])")
axs[1].set_ylabel("XY - E[X]E[Y]")
axs[1].grid(True)

# Add y=x line to show they're equal
min_val = min(np.min((x - mean_x) * (y - mean_y)), np.min(x * y - mean_x * mean_y))
max_val = max(np.max((x - mean_x) * (y - mean_y)), np.max(x * y - mean_x * mean_y))
axs[1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
axs[1].text(min_val + 0.1 * (max_val - min_val), 
           max_val - 0.1 * (max_val - min_val), 
           "y = x (perfect equivalence)", 
           bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
file_path = os.path.join(save_dir, "7_covariance_formula.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

print("\nKey Insights for Statements 7 and 8:")
print("- The covariance formula can be expressed as either E[(X-E[X])(Y-E[Y])] or E[XY] - E[X]E[Y]")
print("- These formulations are mathematically equivalent")
print("- When X = Y, covariance becomes variance: Cov(X,X) = Var(X)")
print("- Cov(X,X) = E[(X-E[X])(X-E[X])] = E[(X-E[X])²] = Var(X)")
print("\nStatement 7 is TRUE. The covariance can be calculated using either formula.")
print("\nStatement 8 is TRUE. The covariance of a random variable with itself equals its variance.")

# Step 9: Negative correlation and principal axes
print_step_header(9, "Negative Correlation and Principal Axes")

# Create a range of correlation values to visualize
correlations = [-0.9, 0, 0.9]  # Negative, zero, and positive correlation
titles = ["Negative Correlation", "Zero Correlation", "Positive Correlation"]
colors = ["red", "blue", "green"]

# Create figure
fig, axs = plt.subplots(2, 3, figsize=(18, 12))

# Plot bivariate distributions with different correlations
for i, (corr, title, color) in enumerate(zip(correlations, titles, colors)):
    # Create covariance matrix
    cov = np.array([[1, corr], [corr, 1]])
    
    # Generate data
    np.random.seed(42)
    data = np.random.multivariate_normal([0, 0], cov, 500)
    
    # Plot scatter
    axs[0, i].scatter(data[:, 0], data[:, 1], alpha=0.5, color=color)
    axs[0, i].set_title(f"{title}\nCorrelation = {corr}")
    axs[0, i].set_xlabel("X")
    axs[0, i].set_ylabel("Y")
    axs[0, i].set_xlim(-3, 3)
    axs[0, i].set_ylim(-3, 3)
    axs[0, i].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axs[0, i].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    axs[0, i].grid(True)
    axs[0, i].set_aspect('equal')
    
    # Add y = x and y = -x lines to show alignment
    if corr == -0.9:
        axs[0, i].plot([-3, 3], [3, -3], 'r--', alpha=0.8, label="y = -x")
    elif corr == 0.9:
        axs[0, i].plot([-3, 3], [-3, 3], 'g--', alpha=0.8, label="y = x")
    axs[0, i].legend()
    
    # Generate contour plot for PDF
    x_range = np.linspace(-3, 3, 100)
    y_range = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    pos = np.dstack((X, Y))
    rv = multivariate_normal([0, 0], cov)
    Z = rv.pdf(pos)
    
    # Plot contours
    axs[1, i].contour(X, Y, Z, levels=10, colors=color)
    axs[1, i].set_title(f"Contours for {title}")
    axs[1, i].set_xlabel("X")
    axs[1, i].set_ylabel("Y")
    axs[1, i].set_xlim(-3, 3)
    axs[1, i].set_ylim(-3, 3)
    axs[1, i].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axs[1, i].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    axs[1, i].grid(True)
    axs[1, i].set_aspect('equal')
    
    # Find eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # Plot principal axes (eigenvectors scaled by sqrt(eigenvalues))
    for j in range(2):
        axs[1, i].arrow(0, 0, 
                       eigvecs[0, j] * np.sqrt(eigvals[j]), 
                       eigvecs[1, j] * np.sqrt(eigvals[j]),
                       head_width=0.2, head_length=0.2, fc=color, ec=color, 
                       alpha=0.8, label=f"Eigenvector {j+1}")
    
    # Add y = x and y = -x lines to show alignment
    if corr == -0.9:
        axs[1, i].plot([-3, 3], [3, -3], 'r--', alpha=0.5, label="y = -x")
    elif corr == 0.9:
        axs[1, i].plot([-3, 3], [-3, 3], 'g--', alpha=0.5, label="y = x")
    axs[1, i].legend()

plt.tight_layout()
file_path = os.path.join(save_dir, "8_negative_correlation_principal_axes.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

print("\nKey Insights for Statements 9 and 10:")
print("- In a bivariate normal distribution, negative correlation results in contours tilted along y = -x")
print("- Positive correlation results in contours tilted along y = x")
print("- Zero correlation results in axis-aligned contours (no tilt)")
print("- The principal axes of probability density contours align with the eigenvectors of the covariance matrix")
print("- Eigenvectors point in the directions of maximum and minimum variance")
print("- Eigenvalues represent the amount of variance in the direction of the corresponding eigenvector")
print("\nStatement 9 is TRUE. Negative correlation corresponds to contours tilted along the line y = -x.")
print("\nStatement 10 is TRUE. The principal axes of density contours align with the eigenvectors of the covariance matrix.")

# Step 10: Contour lines on probability density plots
print_step_header(10, "Contour Lines on Probability Density Plots")

# Create a bivariate normal distribution
mean = [0, 0]
cov = [[2, 0.5], [0.5, 1]]  # Non-diagonal covariance matrix
rv = multivariate_normal(mean, cov)

# Create a meshgrid for plotting
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Calculate PDF values
Z = rv.pdf(pos)

# Create 3D surface and contour plots
fig = plt.figure(figsize=(15, 10))

# 3D surface
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.8)
ax1.set_title("3D PDF Surface")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("PDF Value")

# Calculate some points with the same PDF value for demonstration
pdf_value = 0.05
points = []
for i in range(0, 100, 10):
    for j in range(0, 100, 10):
        if abs(Z[i, j] - pdf_value) < 0.01:
            points.append([X[i, j], Y[i, j], Z[i, j]])

# Add points with the same PDF value
if points:
    points = np.array(points)
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c='red', s=50, label=f"PDF = {pdf_value:.3f}")

# Add a horizontal plane at the specific PDF value
x_grid, y_grid = np.meshgrid([-3, 3], [-3, 3])
z_grid = np.ones_like(x_grid) * pdf_value
ax1.plot_surface(x_grid, y_grid, z_grid, color='red', alpha=0.3)

# 2D contour
ax2 = fig.add_subplot(1, 2, 2)
contour = ax2.contour(X, Y, Z, levels=10, colors='blue')
ax2.set_title("Contour Plot (Horizontal Slices of PDF)")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.grid(True)

# Highlight the contour line with PDF = pdf_value
specific_contour = ax2.contour(X, Y, Z, levels=[pdf_value], colors='red', linewidths=3)
ax2.clabel(specific_contour, inline=True, fontsize=10, fmt=f'PDF = %1.3f')

# Plot points with the same PDF value
if points.size > 0:
    ax2.scatter(points[:, 0], points[:, 1], c='red', s=50, label=f"PDF = {pdf_value:.3f}")

ax2.legend()

# Add text explanation
ax2.text(2, 2, "Contour lines connect points\nwith the same PDF value", 
         bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
file_path = os.path.join(save_dir, "9_contour_lines.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

print("\nKey Insights for Statement 11:")
print("- Contour lines on a probability density plot connect points with the same PDF value")
print("- They represent 'slices' through the 3D probability density surface at constant height")
print("- In a multivariate normal distribution, these contours form ellipses")
print("- Each contour represents a region with equal probability density")
print("\nStatement 11 is TRUE. Contour lines connect points with the same probability density value.")

# Step 11: Ellipsoid Volume and Eigenvalues
print_step_header(11, "Ellipsoid Volume and Eigenvalues")

print("Now we'll demonstrate statement 12 about the volume of the standard deviation ellipsoid...")

# Create several 2D covariance matrices with different eigenvalues but same sum
eigenvalues_1 = np.array([2.0, 2.0])  # Sum = 4, Product = 4
eigenvalues_2 = np.array([3.0, 1.0])  # Sum = 4, Product = 3
eigenvalues_3 = np.array([3.9, 0.1])  # Sum = 4, Product = 0.39

# Create matrices with these eigenvalues (using identity eigenvectors for simplicity)
matrix_1 = np.diag(eigenvalues_1)
matrix_2 = np.diag(eigenvalues_2)
matrix_3 = np.diag(eigenvalues_3)

# Calculate actual volumes (proportional to sqrt(determinant))
volume_1 = np.sqrt(np.linalg.det(matrix_1))
volume_2 = np.sqrt(np.linalg.det(matrix_2))
volume_3 = np.sqrt(np.linalg.det(matrix_3))

# Create figure
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Plot ellipses corresponding to each covariance matrix
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Calculate PDFs for each matrix
rv1 = multivariate_normal([0, 0], matrix_1)
Z1 = rv1.pdf(pos)
rv2 = multivariate_normal([0, 0], matrix_2)
Z2 = rv2.pdf(pos)
rv3 = multivariate_normal([0, 0], matrix_3)
Z3 = rv3.pdf(pos)

# Plot standard deviation contours
axs[0].contour(X, Y, Z1, levels=[np.exp(-0.5/1)/(2*np.pi*np.sqrt(np.linalg.det(matrix_1)))], colors='blue', linewidths=2, label=f"Sum={np.sum(eigenvalues_1)}, Vol={volume_1:.2f}")
axs[0].contour(X, Y, Z2, levels=[np.exp(-0.5/1)/(2*np.pi*np.sqrt(np.linalg.det(matrix_2)))], colors='green', linewidths=2, label=f"Sum={np.sum(eigenvalues_2)}, Vol={volume_2:.2f}")
axs[0].contour(X, Y, Z3, levels=[np.exp(-0.5/1)/(2*np.pi*np.sqrt(np.linalg.det(matrix_3)))], colors='red', linewidths=2, label=f"Sum={np.sum(eigenvalues_3)}, Vol={volume_3:.2f}")

axs[0].set_title("Ellipsoids with Same Sum of Eigenvalues\nBut Different Volumes")
axs[0].set_xlabel("X")
axs[0].set_ylabel("Y")
axs[0].legend()
axs[0].set_aspect('equal')
axs[0].grid(True)

# Create bar plot comparing sum vs volume
matrices = ["Matrix 1", "Matrix 2", "Matrix 3"]
eigenvalue_sums = [np.sum(eigenvalues_1), np.sum(eigenvalues_2), np.sum(eigenvalues_3)]
volumes = [volume_1, volume_2, volume_3]

x = np.arange(len(matrices))
width = 0.35

axs[1].bar(x - width/2, eigenvalue_sums, width, label='Sum of Eigenvalues')
axs[1].bar(x + width/2, volumes, width, label='Ellipsoid Volume')
axs[1].set_ylabel("Value")
axs[1].set_title("Comparison of Sum and Volume")
axs[1].set_xticks(x)
axs[1].set_xticklabels(matrices)
axs[1].legend()

plt.tight_layout()
file_path = os.path.join(save_dir, "10_ellipsoid_volume.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

print("\nStep-by-step calculation of ellipsoid volumes:")
for i, (evals, matrix, vol) in enumerate(zip([eigenvalues_1, eigenvalues_2, eigenvalues_3], 
                                           [matrix_1, matrix_2, matrix_3],
                                           [volume_1, volume_2, volume_3])):
    print(f"\nMatrix {i+1}:")
    print(f"  Eigenvalues: {evals}")
    print(f"  Sum of eigenvalues: {np.sum(evals)}")
    print(f"  Product of eigenvalues: {np.prod(evals)}")
    print(f"  Determinant of matrix: {np.linalg.det(matrix)}")
    print(f"  Volume of ellipsoid (∝ √det): {vol}")
    
print("\nKey Insights for Statement 12:")
print("- The volume of an ellipsoid with semi-axes a₁, a₂, ..., aₙ is proportional to ∏ᵢ aᵢ")
print("- For a covariance matrix, the semi-axes are proportional to √λᵢ where λᵢ are eigenvalues")
print("- Therefore, ellipsoid volume ∝ ∏ᵢ √λᵢ = √(∏ᵢ λᵢ) = √det(Σ)")
print("- The sum of eigenvalues (trace of the matrix) does not determine the volume")
print("- Matrices with the same trace can have very different volumes")
print("\nStatement 12 is FALSE. Ellipsoid volume is proportional to the square root of the product (not sum) of eigenvalues.")

# Step 12: Angle of Rotation of Probability Contours
print_step_header(12, "Angle of Rotation of Probability Contours")

print("Now we'll examine statement 13 about the angle of rotation formula...")

# Create several 2D covariance matrices with different properties
# Case 1: Unequal variances (formula works)
sigma_x1, sigma_y1 = 3.0, 1.0
sigma_xy1 = 0.5
cov1 = np.array([[sigma_x1, sigma_xy1], [sigma_xy1, sigma_y1]])

# Case 2: Equal variances, positive covariance (45° rotation)
sigma_x2, sigma_y2 = 2.0, 2.0  
sigma_xy2 = 1.0
cov2 = np.array([[sigma_x2, sigma_xy2], [sigma_xy2, sigma_y2]])

# Case 3: Equal variances, negative covariance (135° rotation)
sigma_x3, sigma_y3 = 2.0, 2.0
sigma_xy3 = -1.0
cov3 = np.array([[sigma_x3, sigma_xy3], [sigma_xy3, sigma_y3]])

# Calculate angles using the formula (with error handling for case 2 & 3)
def calculate_angle(sigma_x, sigma_y, sigma_xy):
    if abs(sigma_x - sigma_y) < 1e-10:  # Equal variances
        if sigma_xy > 0:
            return 45  # degrees
        elif sigma_xy < 0:
            return 135  # degrees
        else:
            return None  # No preferred orientation (circle)
    else:
        # Apply the formula
        return np.degrees(0.5 * np.arctan(2 * sigma_xy / (sigma_x - sigma_y)))

angle1 = calculate_angle(sigma_x1, sigma_y1, sigma_xy1)
angle2 = calculate_angle(sigma_x2, sigma_y2, sigma_xy2)
angle3 = calculate_angle(sigma_x3, sigma_y3, sigma_xy3)

# Create figure
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Plot contours for each case
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Calculate PDFs
rv1 = multivariate_normal([0, 0], cov1)
Z1 = rv1.pdf(pos)
rv2 = multivariate_normal([0, 0], cov2)
Z2 = rv2.pdf(pos)
rv3 = multivariate_normal([0, 0], cov3)
Z3 = rv3.pdf(pos)

# Plot contours
axs[0].contour(X, Y, Z1, levels=10, colors='blue')
axs[0].set_title(f"Unequal Variances\nσ²ₓ={sigma_x1}, σ²ᵧ={sigma_y1}, σₓᵧ={sigma_xy1}\nAngle = {angle1:.1f}°")
axs[0].set_xlabel("X")
axs[0].set_ylabel("Y")
axs[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axs[0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axs[0].set_aspect('equal')
axs[0].grid(True)

# Plot contours for equal variances, positive covariance
axs[1].contour(X, Y, Z2, levels=10, colors='green')
axs[1].set_title(f"Equal Variances, Positive Covariance\nσ²ₓ={sigma_x2}, σ²ᵧ={sigma_y2}, σₓᵧ={sigma_xy2}\nAngle = {angle2}°")
axs[1].set_xlabel("X")
axs[1].set_ylabel("Y")
axs[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axs[1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axs[1].plot([-4, 4], [-4, 4], 'r--', alpha=0.8)  # y = x line
axs[1].set_aspect('equal')
axs[1].grid(True)

# Plot contours for equal variances, negative covariance
axs[2].contour(X, Y, Z3, levels=10, colors='red')
axs[2].set_title(f"Equal Variances, Negative Covariance\nσ²ₓ={sigma_x3}, σ²ᵧ={sigma_y3}, σₓᵧ={sigma_xy3}\nAngle = {angle3}°")
axs[2].set_xlabel("X")
axs[2].set_ylabel("Y")
axs[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axs[2].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axs[2].plot([-4, 4], [4, -4], 'r--', alpha=0.8)  # y = -x line
axs[2].set_aspect('equal')
axs[2].grid(True)

plt.tight_layout()
file_path = os.path.join(save_dir, "11_rotation_angles.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

print("\nStep-by-step calculation of rotation angles:")
print("\nCase 1: Unequal Variances")
print(f"  σ²ₓ = {sigma_x1}, σ²ᵧ = {sigma_y1}, σₓᵧ = {sigma_xy1}")
print(f"  Formula: θ = (1/2)tan⁻¹(2σₓᵧ/(σ²ₓ-σ²ᵧ))")
print(f"  Calculation: θ = (1/2)tan⁻¹(2×{sigma_xy1}/({sigma_x1}-{sigma_y1}))")
print(f"  = (1/2)tan⁻¹(2×{sigma_xy1}/{sigma_x1-sigma_y1})")
print(f"  = (1/2)tan⁻¹({2*sigma_xy1}/{sigma_x1-sigma_y1})")
print(f"  = (1/2)tan⁻¹({2*sigma_xy1/(sigma_x1-sigma_y1)})")
print(f"  = (1/2) × {np.degrees(np.arctan(2*sigma_xy1/(sigma_x1-sigma_y1)))}°")
print(f"  = {angle1}°")

print("\nCase 2: Equal Variances, Positive Covariance")
print(f"  σ²ₓ = {sigma_x2}, σ²ᵧ = {sigma_y2}, σₓᵧ = {sigma_xy2}")
print(f"  Formula: θ = (1/2)tan⁻¹(2σₓᵧ/(σ²ₓ-σ²ᵧ))")
print(f"  Calculation: θ = (1/2)tan⁻¹(2×{sigma_xy2}/({sigma_x2}-{sigma_y2}))")
print(f"  = (1/2)tan⁻¹(2×{sigma_xy2}/{sigma_x2-sigma_y2})")
print(f"  = (1/2)tan⁻¹(2×{sigma_xy2}/0)")
print("  Problem: Division by zero! Formula is undefined.")
print("  For equal variances with positive covariance: θ = 45°")

print("\nCase 3: Equal Variances, Negative Covariance")
print(f"  σ²ₓ = {sigma_x3}, σ²ᵧ = {sigma_y3}, σₓᵧ = {sigma_xy3}")
print(f"  Formula breaks similarly")
print("  For equal variances with negative covariance: θ = 135°")

print("\nKey Insights for Statement 13:")
print("- The formula θ = (1/2)tan⁻¹(2σₓᵧ/(σ²ₓ-σ²ᵧ)) only works when σ²ₓ ≠ σ²ᵧ")
print("- When variances are equal (σ²ₓ = σ²ᵧ), formula is undefined (division by zero)")
print("- For equal variances:")
print("  * If σₓᵧ > 0: ellipse is oriented at 45° (along y = x)")
print("  * If σₓᵧ < 0: ellipse is oriented at 135° (along y = -x)")
print("  * If σₓᵧ = 0: contours form a circle (no preferred orientation)")
print("\nStatement 13 is FALSE. The formula is not valid for all cases, particularly when variances are equal.")

# Step 13: Summarize all statements
print_step_header(13, "Summary of All Statements")

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
    "Ellipsoid volume is proportional to the square root of the product (determinant), not sum (trace) of eigenvalues.",
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