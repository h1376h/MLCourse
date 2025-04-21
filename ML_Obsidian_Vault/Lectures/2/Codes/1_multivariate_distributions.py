import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.gridspec import GridSpec

print("\n=== MULTIVARIATE DISTRIBUTIONS: VISUALIZATIONS ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Example 1: Basic Bivariate Normal Distribution
print("Example 1: Basic Bivariate Normal Distribution")

# Create grid of points
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Define bivariate normal parameters
mu = np.array([0.0, 0.0])  # mean
var1, var2, corr = 1.0, 1.0, 0.0  # parameters
cov = np.array([[var1, corr * np.sqrt(var1 * var2)], 
                [corr * np.sqrt(var1 * var2), var2]])

# Create multivariate normal distribution
rv = stats.multivariate_normal(mu, cov)
Z = rv.pdf(pos)

# Plot PDF
plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar(label='Probability Density')
plt.contour(X, Y, Z, levels=10, colors='k', alpha=0.5)
plt.title('Basic Bivariate Normal Distribution')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'basic_bivariate_normal.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Visualization of Correlation
print("\nExample 2: Visualization of Correlation")

correlations = [-0.8, -0.4, 0.0, 0.4, 0.8]
plt.figure(figsize=(15, 10))

for i, corr in enumerate(correlations):
    # Create covariance matrix with correlation
    cov = np.array([[var1, corr * np.sqrt(var1 * var2)], 
                    [corr * np.sqrt(var1 * var2), var2]])
    
    # Create multivariate normal distribution
    rv = stats.multivariate_normal(mu, cov)
    Z = rv.pdf(pos)
    
    plt.subplot(2, 3, i+1)
    plt.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.contour(X, Y, Z, levels=10, colors='k', alpha=0.5)
    plt.title(f'Correlation: {corr}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'correlation_visualization.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Scatter Plot vs. Contour Plot
print("\nExample 3: Scatter Plot vs. Contour Plot")

# Generate random samples from bivariate normal
np.random.seed(42)
corr = 0.7
cov = np.array([[var1, corr * np.sqrt(var1 * var2)], 
                [corr * np.sqrt(var1 * var2), var2]])
samples = np.random.multivariate_normal(mu, cov, 1000)

# Calculate PDF for contour
rv = stats.multivariate_normal(mu, cov)
Z = rv.pdf(pos)

plt.figure(figsize=(15, 6))

# Scatter plot
plt.subplot(1, 2, 1)
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, edgecolor='none')
plt.title('Scatter Plot of Bivariate Normal Samples')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.grid(alpha=0.3)

# Contour plot
plt.subplot(1, 2, 2)
plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.contour(X, Y, Z, levels=10, colors='k', alpha=0.5)
plt.title('Contour Plot of Bivariate Normal PDF')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'scatter_vs_contour.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 4: Conditional Distributions
print("\nExample 4: Conditional Distributions")

# Create bivariate normal with correlation
corr = 0.7
cov = np.array([[var1, corr * np.sqrt(var1 * var2)], 
                [corr * np.sqrt(var1 * var2), var2]])
rv = stats.multivariate_normal(mu, cov)
Z = rv.pdf(pos)

# Function to compute conditional distribution
def conditional_normal(x_val, mean, cov):
    mu1, mu2 = mean
    sigma11, sigma12 = cov[0, 0], cov[0, 1]
    sigma21, sigma22 = cov[1, 0], cov[1, 1]
    
    # Conditional mean: mu2|1 = mu2 + sigma21 * sigma11^-1 * (x - mu1)
    cond_mean = mu2 + sigma21 / sigma11 * (x_val - mu1)
    
    # Conditional variance: sigma2|1 = sigma22 - sigma21 * sigma11^-1 * sigma12
    cond_var = sigma22 - sigma21 * sigma12 / sigma11
    
    return cond_mean, np.sqrt(cond_var)

# Compute several conditional distributions
x_values = [-2, -1, 0, 1, 2]
y_range = np.linspace(-5, 5, 1000)

plt.figure(figsize=(10, 8))

# Plot joint distribution
plt.contourf(X, Y, Z, levels=20, cmap='Blues', alpha=0.3)
plt.contour(X, Y, Z, levels=10, colors='k', alpha=0.2)

# Plot conditional distributions
for x_val in x_values:
    # Get conditional mean and std for this x value
    cond_mean, cond_std = conditional_normal(x_val, mu, cov)
    
    # Create conditional PDF (univariate normal)
    y_pdf = stats.norm.pdf(y_range, loc=cond_mean, scale=cond_std)
    
    # Scale PDF for visualization
    y_pdf = y_pdf / y_pdf.max() * 0.8
    
    # Plot vertical line at x value
    plt.axvline(x=x_val, color='r', linestyle='--', alpha=0.5)
    
    # Plot conditional distribution
    plt.plot(x_val + y_pdf, y_range, 'r-', linewidth=2)
    plt.fill_betweenx(y_range, x_val, x_val + y_pdf, alpha=0.2, color='red')

plt.title('Conditional Distributions of Bivariate Normal')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'conditional_distributions.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 5: Multivariate KDE from Data
print("\nExample 5: Multivariate KDE from Data")

# Generate data from a mixture of two bivariate normals
np.random.seed(42)
n_samples = 500

# First component
mean1 = [1, 1]
cov1 = [[1, 0.6], [0.6, 1]]
samples1 = np.random.multivariate_normal(mean1, cov1, n_samples)

# Second component
mean2 = [-1, -1]
cov2 = [[1, -0.6], [-0.6, 1]]
samples2 = np.random.multivariate_normal(mean2, cov2, n_samples)

# Combine samples
samples = np.vstack([samples1, samples2])

# Compute kernel density estimate
kde = stats.gaussian_kde(samples.T)
Z_kde = np.zeros_like(Z)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z_kde[i, j] = kde([X[i, j], Y[i, j]])

plt.figure(figsize=(15, 6))

# Scatter plot of data
plt.subplot(1, 2, 1)
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, edgecolor='none')
plt.title('Scattered Data from Mixture Distribution')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.grid(alpha=0.3)

# KDE contour plot
plt.subplot(1, 2, 2)
plt.contourf(X, Y, Z_kde, levels=20, cmap='viridis')
plt.contour(X, Y, Z_kde, levels=10, colors='k', alpha=0.5)
plt.title('Kernel Density Estimate (KDE)')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'multivariate_kde.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 6: Joint, Marginal, and Conditional Distributions
print("\nExample 6: Joint, Marginal, and Conditional Distributions")

# Create bivariate normal with correlation
corr = 0.7
cov = np.array([[var1, corr * np.sqrt(var1 * var2)], 
                [corr * np.sqrt(var1 * var2), var2]])
rv = stats.multivariate_normal(mu, cov)

# Create grid of points
x_fine = np.linspace(-5, 5, 200)
y_fine = np.linspace(-5, 5, 200)
X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
pos_fine = np.dstack((X_fine, Y_fine))
Z_fine = rv.pdf(pos_fine)

# Calculate marginal distributions
margin_x = np.sum(Z_fine, axis=0) * (y_fine[1] - y_fine[0])  # approximate integration over y
margin_y = np.sum(Z_fine, axis=1) * (x_fine[1] - x_fine[0])  # approximate integration over x

fig = plt.figure(figsize=(12, 10))
gs = GridSpec(4, 4, figure=fig)

# Main contour plot (center)
ax_joint = fig.add_subplot(gs[1:4, 0:3])
contour = ax_joint.contourf(X_fine, Y_fine, Z_fine, levels=20, cmap='viridis')
ax_joint.contour(X_fine, Y_fine, Z_fine, levels=10, colors='k', alpha=0.3)
ax_joint.set_xlabel('X')
ax_joint.set_ylabel('Y')
ax_joint.grid(alpha=0.3)

# Marginal plot for x (top)
ax_marg_x = fig.add_subplot(gs[0, 0:3])
ax_marg_x.plot(x_fine, margin_x, 'r-', linewidth=2)
ax_marg_x.fill_between(x_fine, margin_x, alpha=0.3, color='red')
ax_marg_x.set_title('Joint, Marginal & Conditional Distributions')
ax_marg_x.set_ylabel('P(X)')
ax_marg_x.grid(alpha=0.3)
ax_marg_x.set_xlim(ax_joint.get_xlim())
ax_marg_x.set_xticklabels([])

# Marginal plot for y (right)
ax_marg_y = fig.add_subplot(gs[1:4, 3])
ax_marg_y.plot(margin_y, y_fine, 'r-', linewidth=2)
ax_marg_y.fill_betweenx(y_fine, margin_y, alpha=0.3, color='red')
ax_marg_y.set_xlabel('P(Y)')
ax_marg_y.grid(alpha=0.3)
ax_marg_y.set_ylim(ax_joint.get_ylim())
ax_marg_y.set_yticklabels([])

# Select a conditional slice for x = 1
x_cond = 1.0
idx = np.abs(x_fine - x_cond).argmin()
cond_mean, cond_std = conditional_normal(x_cond, mu, cov)
y_cond_pdf = stats.norm.pdf(y_fine, loc=cond_mean, scale=cond_std)
y_cond_pdf = y_cond_pdf / y_cond_pdf.max() * margin_y.max() * 0.8

# Plot the conditional distribution on the joint plot
ax_joint.axvline(x=x_cond, color='blue', linestyle='--', label=f'Conditional at x={x_cond}')
ax_joint.plot(x_cond + y_cond_pdf, y_fine, 'b-', linewidth=2)
ax_joint.fill_betweenx(y_fine, x_cond, x_cond + y_cond_pdf, alpha=0.2, color='blue')
ax_joint.legend(loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'joint_marginal_conditional.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 7: Different Multivariate Distributions
print("\nExample 7: Different Multivariate Distributions")

plt.figure(figsize=(15, 10))

# 1. Standard Bivariate Normal
plt.subplot(2, 2, 1)
cov = np.array([[1.0, 0.0], [0.0, 1.0]])
rv = stats.multivariate_normal(mu, cov)
Z1 = rv.pdf(pos)
plt.contourf(X, Y, Z1, levels=20, cmap='Blues')
plt.contour(X, Y, Z1, levels=10, colors='k', alpha=0.5)
plt.title('Standard Bivariate Normal')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.grid(alpha=0.3)

# 2. Student's t-distribution (bivariate)
# We'll use samples and KDE for visualization
plt.subplot(2, 2, 2)
np.random.seed(42)
# Generate from bivariate t with 3 degrees of freedom
df = 3
samples_t = np.random.standard_t(df, size=(2000, 2))
# Apply correlation
corr_matrix = np.array([[1.0, 0.7], [0.7, 1.0]])
L = np.linalg.cholesky(corr_matrix)
samples_t = np.dot(samples_t, L.T)

# Compute KDE for t-distribution
kde_t = stats.gaussian_kde(samples_t.T)
Z_t = np.zeros_like(Z)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z_t[i, j] = kde_t([X[i, j], Y[i, j]])

plt.contourf(X, Y, Z_t, levels=20, cmap='Greens')
plt.contour(X, Y, Z_t, levels=10, colors='k', alpha=0.5)
plt.title('Bivariate Student\'s t-Distribution')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.grid(alpha=0.3)

# 3. Bivariate Uniform Distribution
plt.subplot(2, 2, 3)
Z_unif = np.zeros_like(Z)
Z_unif[(X >= -2) & (X <= 2) & (Y >= -2) & (Y <= 2)] = 1.0
Z_unif = Z_unif / np.sum(Z_unif)  # Normalize

plt.contourf(X, Y, Z_unif, levels=20, cmap='Oranges')
plt.contour(X, Y, Z_unif, levels=[0.0001], colors='k', linewidths=2)
plt.title('Bivariate Uniform Distribution')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.grid(alpha=0.3)

# 4. Bivariate Exponential-like Distribution
plt.subplot(2, 2, 4)
Z_exp = np.exp(-np.abs(X) - np.abs(Y))
Z_exp = Z_exp / np.sum(Z_exp)  # Normalize

plt.contourf(X, Y, Z_exp, levels=20, cmap='Reds')
plt.contour(X, Y, Z_exp, levels=10, colors='k', alpha=0.5)
plt.title('Bivariate Exponential-like Distribution')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'different_multivariate_distributions.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 8: 3D Comparison of Multivariate Distributions
print("\nExample 8: 3D Comparison of Multivariate Distributions")

plt.figure(figsize=(15, 10))

# Create focused grid for better visualization
x_focus = np.linspace(-3, 3, 100)
y_focus = np.linspace(-3, 3, 100)
X_focus, Y_focus = np.meshgrid(x_focus, y_focus)
pos_focus = np.dstack((X_focus, Y_focus))

# 1. Standard Bivariate Normal
ax1 = plt.subplot(2, 2, 1, projection='3d')
cov = np.array([[1.0, 0.0], [0.0, 1.0]])
rv = stats.multivariate_normal(mu, cov)
Z1 = rv.pdf(pos_focus)
surf1 = ax1.plot_surface(X_focus, Y_focus, Z1, cmap='Blues', alpha=0.8, linewidth=0, antialiased=True)
ax1.set_title('Standard Bivariate Normal')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Density')

# 2. Correlated Bivariate Normal
ax2 = plt.subplot(2, 2, 2, projection='3d')
cov = np.array([[1.0, 0.7], [0.7, 1.0]])
rv = stats.multivariate_normal(mu, cov)
Z2 = rv.pdf(pos_focus)
surf2 = ax2.plot_surface(X_focus, Y_focus, Z2, cmap='Greens', alpha=0.8, linewidth=0, antialiased=True)
ax2.set_title('Correlated Bivariate Normal')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Density')

# 3. Bivariate Uniform
ax3 = plt.subplot(2, 2, 3, projection='3d')
Z3 = np.zeros_like(Z1)
Z3[(X_focus >= -1) & (X_focus <= 1) & (Y_focus >= -1) & (Y_focus <= 1)] = 0.25
surf3 = ax3.plot_surface(X_focus, Y_focus, Z3, cmap='Oranges', alpha=0.8, linewidth=0, antialiased=True)
ax3.set_title('Bivariate Uniform')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Density')

# 4. Bivariate Exponential-like
ax4 = plt.subplot(2, 2, 4, projection='3d')
Z4 = np.exp(-np.abs(X_focus) - np.abs(Y_focus))
Z4 = Z4 / np.sum(Z4) * 100  # Normalize and scale for visualization
surf4 = ax4.plot_surface(X_focus, Y_focus, Z4, cmap='Reds', alpha=0.8, linewidth=0, antialiased=True)
ax4.set_title('Bivariate Exponential-like')
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Density')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, '3d_multivariate_distributions.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll multivariate distribution visualizations created successfully.") 