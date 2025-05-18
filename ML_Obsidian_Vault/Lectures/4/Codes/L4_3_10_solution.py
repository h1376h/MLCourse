import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
import os
from mpl_toolkits.mplot3d import Axes3D

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_3_Quiz_10")
os.makedirs(save_dir, exist_ok=True)

# Set the style for the plots and enable LaTeX support
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['text.usetex'] = True  # Enable LaTeX rendering
plt.rcParams['font.family'] = 'serif'

# Define the problem parameters for Question 10
# Step 1: Define the mean vectors and covariance matrices for both classes
mu0 = np.array([0, 0])  # Mean for class 0
mu1 = np.array([0, 0])  # Mean for class 1 (equal to mu0)

Sigma0 = np.array([[1, 0.8], [0.8, 1]])  # Covariance matrix for class 0 (positive correlation)
Sigma1 = np.array([[1, -0.8], [-0.8, 1]])  # Covariance matrix for class 1 (negative correlation)

prior0 = 0.5  # Prior probability for class 0
prior1 = 0.5  # Prior probability for class 1

# Calculate the inverse of covariance matrices (precision matrices)
inv_Sigma0 = np.linalg.inv(Sigma0)
inv_Sigma1 = np.linalg.inv(Sigma1)

# Calculate determinants
det_Sigma0 = np.linalg.det(Sigma0)
det_Sigma1 = np.linalg.det(Sigma1)

# Step 2: Define the multivariate Gaussian PDF function
def gaussian_pdf(x, mu, Sigma):
    """
    Compute multivariate Gaussian PDF value at point x.
    
    Args:
        x: The point at which to evaluate the PDF
        mu: Mean vector of the Gaussian distribution
        Sigma: Covariance matrix of the Gaussian distribution
        
    Returns:
        The probability density at point x
    """
    n = len(mu)
    det = np.linalg.det(Sigma)
    inv = np.linalg.inv(Sigma)
    
    # Calculate the normalization factor
    factor = 1.0 / ((2 * np.pi) ** (n / 2) * np.sqrt(det))
    
    # Calculate the exponent term
    exponent = -0.5 * np.dot(np.dot((x - mu).T, inv), (x - mu))
    
    return factor * np.exp(exponent)

# Step 3: Create a grid of points for visualization
# Create a grid of x and y values for plotting
x = np.linspace(-3, 3, 300)
y = np.linspace(-3, 3, 300)
X, Y = np.meshgrid(x, y)
pos = np.stack((X, Y), axis=2)  # Stack all (x,y) pairs

# Step 4: Compute probability density functions for both classes
Z0 = np.zeros_like(X)
Z1 = np.zeros_like(X)

# Calculate PDF values at each grid point
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = np.array([X[i, j], Y[i, j]])
        Z0[i, j] = gaussian_pdf(point, mu0, Sigma0)
        Z1[i, j] = gaussian_pdf(point, mu1, Sigma1)

# Step 5: Compute log-likelihood ratio for decision boundary
# log[p(x|y=1)/p(x|y=0)] = log[p(x|y=1)] - log[p(x|y=0)]
# For Bayes optimal with equal priors, decision boundary is where this equals 0
log_ratio = np.log(Z1) - np.log(Z0)

# Step 6: Compute posterior probabilities
# p(y=0|x) = p(x|y=0)p(y=0) / [p(x|y=0)p(y=0) + p(x|y=1)p(y=1)]
posterior0 = prior0 * Z0 / (prior0 * Z0 + prior1 * Z1)
posterior1 = prior1 * Z1 / (prior0 * Z0 + prior1 * Z1)

# Step 7: Theoretical derivation of decision boundary
# For multivariate Gaussians with equal means, the log ratio simplifies to:
# log[p(x|y=1)/p(x|y=0)] = -0.5 * x^T (Σ1^-1 - Σ0^-1) x + 0.5 * log(|Σ0|/|Σ1|)

# For our specific covariance matrices:
# Compute Σ1^-1 - Σ0^-1 explicitly for the analysis
precision_diff = inv_Sigma1 - inv_Sigma0
print("Σ1^-1 - Σ0^-1 =")
print(precision_diff)

# Log determinant ratio
log_det_ratio = np.log(det_Sigma0 / det_Sigma1)
print(f"log(|Σ0|/|Σ1|) = {log_det_ratio}")

# Decision boundary equation: x^T (Σ1^-1 - Σ0^-1) x = log(|Σ0|/|Σ1|)
# Which simplifies to: c11*x1^2 + 2*c12*x1*x2 + c22*x2^2 = log(|Σ0|/|Σ1|)
# Where c11, c12, c22 are elements of (Σ1^-1 - Σ0^-1)

# Step 8: Calculate analytical decision boundary
# For equal means and equal determinants, the decision boundary is quadratic
# It represents a conic section (in this case, a hyperbola)
# We'll create a contour where the log-ratio equals 0

# Define consistent axis limits for all plots
x_lim = (-3, 3)
y_lim = (-3, 3)

# Step 9: Create Figure 1: Gaussian contours and decision boundary
plt.figure(figsize=(10, 8))

# Plot contours for p(x|y=0)
contour0 = plt.contour(X, Y, Z0, levels=5, colors='blue', alpha=0.7, linestyles='solid')
plt.clabel(contour0, inline=True, fontsize=8, fmt='%.2f')

# Plot contours for p(x|y=1)
contour1 = plt.contour(X, Y, Z1, levels=5, colors='red', alpha=0.7, linestyles='solid')
plt.clabel(contour1, inline=True, fontsize=8, fmt='%.2f')

# Plot decision boundary (where log_ratio = 0)
plt.contour(X, Y, log_ratio, levels=[0], colors='k', linestyles='--', linewidths=2)

# Step 10: Add ellipses to represent the covariance matrices
def add_covariance_ellipse(mean, cov, color, label):
    """
    Add an ellipse representing the covariance matrix to the current plot.
    
    Args:
        mean: Mean vector of the Gaussian distribution
        cov: Covariance matrix of the Gaussian distribution
        color: Color for the ellipse
        label: Label for the legend
    """
    # Calculate eigenvalues and eigenvectors of the covariance matrix
    evals, evecs = np.linalg.eigh(cov)
    # Sort eigenvalues in decreasing order
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    
    # Calculate the angle of the ellipse
    angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
    
    # Create the ellipse patch (2 standard deviations)
    ellipse = Ellipse(xy=mean, width=2*2*np.sqrt(evals[0]), height=2*2*np.sqrt(evals[1]),
                      angle=angle, edgecolor=color, facecolor='none', lw=2, label=label)
    plt.gca().add_patch(ellipse)

add_covariance_ellipse(mu0, Sigma0, 'blue', r'Class 0: $2\sigma$ region (Positive Correlation)')
add_covariance_ellipse(mu1, Sigma1, 'red', r'Class 1: $2\sigma$ region (Negative Correlation)')

# Add labels and legend
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Gaussian Contours and Decision Boundary')
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
plt.grid(True)
plt.xlim(x_lim)
plt.ylim(y_lim)
plt.legend()

# Save the figure
plt.savefig(os.path.join(save_dir, 'gaussian_contours_decision_boundary.png'), dpi=300, bbox_inches='tight')

# Step 11: Create Figure 2: Decision regions
plt.figure(figsize=(10, 8))

# Calculate the classified regions
decision_regions = np.zeros_like(X)
decision_regions[(log_ratio < 0)] = 0  # Class 0 regions
decision_regions[(log_ratio >= 0)] = 1  # Class 1 regions

# Plot decision regions
plt.contourf(X, Y, decision_regions, levels=[0, 0.5, 1], colors=['skyblue', 'salmon'], alpha=0.3)

# Plot decision boundary
boundary = plt.contour(X, Y, log_ratio, levels=[0], colors='k', linestyles='--', linewidths=2)
# Add label manually to avoid warning
plt.plot([], [], 'k--', linewidth=2, label='Decision Boundary')

# Add ellipses to represent the covariance matrices
add_covariance_ellipse(mu0, Sigma0, 'blue', r'Class 0: $2\sigma$ region (Positive Correlation)')
add_covariance_ellipse(mu1, Sigma1, 'red', r'Class 1: $2\sigma$ region (Negative Correlation)')

# Add labels and legend
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Decision Regions')
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
plt.grid(True)
plt.xlim(x_lim)
plt.ylim(y_lim)
plt.legend()

# Save the figure
plt.savefig(os.path.join(save_dir, 'decision_regions.png'), dpi=300, bbox_inches='tight')

# Step 12: Visualize correlation with scatter plots
plt.figure(figsize=(10, 8))

# Generate random samples from both distributions
np.random.seed(42)  # For reproducibility
num_samples = 1000

# Generate samples from each distribution
samples0 = np.random.multivariate_normal(mu0, Sigma0, num_samples)
samples1 = np.random.multivariate_normal(mu1, Sigma1, num_samples)

# Plot samples
plt.scatter(samples0[:, 0], samples0[:, 1], color='blue', alpha=0.3, label='Class 0 (Positive Correlation)')
plt.scatter(samples1[:, 0], samples1[:, 1], color='red', alpha=0.3, label='Class 1 (Negative Correlation)')

# Plot decision boundary more clearly
plt.axhline(y=0, color='k', linestyle='--', linewidth=2)
plt.axvline(x=0, color='k', linestyle='--', linewidth=2)

# Add labels and legend
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Effect of Feature Correlation on Class Distribution')
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
plt.grid(True)
plt.xlim(x_lim)
plt.ylim(y_lim)
plt.legend()

# Save the figure
plt.savefig(os.path.join(save_dir, 'correlation_scatter.png'), dpi=300, bbox_inches='tight')

# Step 13: Visualize Linear Classifier Comparison
plt.figure(figsize=(10, 8))

# Plot decision regions with explicit extent to ensure full coloring
plt.contourf(X, Y, decision_regions, levels=[0, 0.5, 1], colors=['skyblue', 'salmon'], alpha=0.3, extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]])

# Plot Bayes-optimal decision boundary
boundary = plt.contour(X, Y, log_ratio, levels=[0], colors='k', linestyles='--', linewidths=2)
# Add label manually to avoid warning
plt.plot([], [], 'k--', linewidth=2, label='Bayes-Optimal Boundary')

# Plot a linear decision boundary (the best approximation)
# Since the optimal boundary requires two perpendicular lines, a single line will be suboptimal
# We'll draw the diagonal line x1 = x2 as an example of a linear boundary
x_diag = np.linspace(-3, 3, 100)
plt.plot(x_diag, x_diag, 'g-', linewidth=2, label='Linear Boundary Example ($x_1 = x_2$)')

# Add samples
plt.scatter(samples0[:, 0], samples0[:, 1], color='blue', alpha=0.3, s=10)
plt.scatter(samples1[:, 0], samples1[:, 1], color='red', alpha=0.3, s=10)

# Add covariance ellipses
add_covariance_ellipse(mu0, Sigma0, 'blue', r'Class 0 (Positive Correlation)')
add_covariance_ellipse(mu1, Sigma1, 'red', r'Class 1 (Negative Correlation)')

# Add text annotations to clarify the quadrants
plt.text(2, 2, "Class 0", fontsize=12, ha='center', va='center', color='blue')
plt.text(-2, -2, "Class 0", fontsize=12, ha='center', va='center', color='blue')
plt.text(-2, 2, "Class 1", fontsize=12, ha='center', va='center', color='red')
plt.text(2, -2, "Class 1", fontsize=12, ha='center', va='center', color='red')

# Plot decision boundary more clearly
plt.axhline(y=0, color='k', linestyle='--', linewidth=2)
plt.axvline(x=0, color='k', linestyle='--', linewidth=2)

# Add labels and legend
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Comparison: Bayes-Optimal vs. Linear Decision Boundary')
plt.grid(True)
plt.xlim(x_lim)
plt.ylim(y_lim)
plt.legend()

# Save the figure
plt.savefig(os.path.join(save_dir, 'linear_vs_bayes.png'), dpi=300, bbox_inches='tight')

# Step 14: Create 3D visualization of the probability densities
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Downsample for clearer visualization
step = 5
X_sparse = X[::step, ::step]
Y_sparse = Y[::step, ::step]
Z0_sparse = Z0[::step, ::step]
Z1_sparse = Z1[::step, ::step]

# Plot the two probability densities
surf0 = ax.plot_surface(X_sparse, Y_sparse, Z0_sparse, cmap='Blues', alpha=0.7, linewidth=0, antialiased=True)
surf1 = ax.plot_surface(X_sparse, Y_sparse, Z1_sparse, cmap='Reds', alpha=0.7, linewidth=0, antialiased=True)

# Add labels
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('Probability Density')
ax.set_title('3D Visualization of Probability Densities')

# Save the figure
plt.savefig(os.path.join(save_dir, '3d_probability_densities.png'), dpi=300, bbox_inches='tight')

# Step 15: Create posterior probability heatmap
plt.figure(figsize=(10, 8))

# Create a heatmap of the posterior probability for class 1
plt.imshow(posterior1, extent=[-3, 3, -3, 3], origin='lower', cmap='coolwarm', vmin=0, vmax=1)
plt.colorbar(label='$P(y=1|x)$')

# Plot decision boundary
plt.contour(X, Y, log_ratio, levels=[0], colors='k', linestyles='--', linewidths=2)

# Add labels
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Posterior Probability Heatmap for Class 1')
plt.grid(False)

# Save the figure
plt.savefig(os.path.join(save_dir, 'posterior_probability_heatmap.png'), dpi=300, bbox_inches='tight')

# Step 16: Analytical derivation of the decision boundary
# Decision boundary for equal priors is where: x^T (Σ1^-1 - Σ0^-1) x = log(|Σ0|/|Σ1|)

# Print analysis of the decision boundary
print("\nAnalytical Derivation of the Decision Boundary:")
print("-----------------------------------------------")
print("For equal means, the Bayes-optimal decision boundary is where:")
print("    x^T (Σ1^-1 - Σ0^-1) x = log(|Σ0|/|Σ1|)")

print("\nCalculating precision matrices (inverse covariance):")
print("Σ0^-1 =")
print(inv_Sigma0)
print("\nΣ1^-1 =")
print(inv_Sigma1)

print("\nCalculating determinants:")
print(f"|Σ0| = {det_Sigma0}")
print(f"|Σ1| = {det_Sigma1}")

print("\nThe full decision boundary equation is:")
c11 = precision_diff[0, 0]
c12 = precision_diff[0, 1]
c22 = precision_diff[1, 1]
print(f"    {c11:.4f}*x1^2 + {2*c12:.4f}*x1*x2 + {c22:.4f}*x2^2 = {log_det_ratio:.4f}")

print("\nThis is the equation of a quadratic curve (specifically a hyperbola).")
print("The key observation is that the decision boundary is NOT linear.")
print("This is primarily due to the *opposite* correlations in the two classes.")
print("Linear classifiers can only create straight line decision boundaries,")
print("making them ineffective for this type of problem where the optimal")
print("boundary is quadratic.")

# Print the eigendecomposition to analyze the curvature
eigvals, eigvecs = np.linalg.eigh(precision_diff)
print("\nEigendecomposition of (Σ1^-1 - Σ0^-1):")
print(f"Eigenvalues: {eigvals}")
print("Eigenvectors:")
print(eigvecs)

print("\nVisualization saved to:", save_dir)

# Display the plots (commented out for automation)
# plt.show() 