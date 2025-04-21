import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_1_Quiz_30")
os.makedirs(save_dir, exist_ok=True)

# Set general plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})

print("\n## Solution to Question 30: Contour Plot Interpretation\n")

# Original bivariate normal distribution parameters (positive correlation)
mu = [0, 0]          # Mean vector at the origin
rho_original = 0.6   # Positive correlation coefficient
sigma_x = 2          # Higher variance for x
sigma_y = 1          # Lower variance for y

# Create a grid for all plots
x = np.linspace(-5, 5, 300)
y = np.linspace(-3, 3, 300)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Function to create bivariate normal distribution
def create_bivariate_normal(mu, rho, sigma_x, sigma_y):
    cov = [[sigma_x**2, rho*sigma_x*sigma_y], 
           [rho*sigma_x*sigma_y, sigma_y**2]]
    rv = stats.multivariate_normal(mu, cov)
    return rv, cov

# Standalone zero correlation visualization
print("Generating standalone zero correlation visualization...")
plt.figure(figsize=(10, 8))

# Create a bivariate normal with zero correlation
rv_zero, _ = create_bivariate_normal(mu, 0, sigma_x, sigma_y)
Z_zero = rv_zero.pdf(pos)

# Add contour lines and filled contours
contour_levels = np.linspace(0, 0.15, 10)
plt.contourf(X, Y, Z_zero, levels=contour_levels, cmap='viridis', alpha=0.7)
CS = plt.contour(X, Y, Z_zero, levels=contour_levels, colors='black', linewidths=0.8)

# Label the innermost contour
plt.clabel(CS, inline=True, fontsize=8, fmt='%.2f', manual=[(0, 0)])

# Add grid, axis labels and title
plt.grid(linestyle='--', alpha=0.6)
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.title('Bivariate Normal Distribution with Zero Correlation')
plt.colorbar(label='Probability Density')

# Add annotation to highlight zero correlation
plt.annotate('Ellipses aligned with axes\nindicate zero correlation', 
             xy=(2, 0), xytext=(2.5, 1), fontsize=9,
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

# Add a mark at (0,0) to show the distribution center
plt.scatter(0, 0, color='red', s=50, marker='x', label='Mean Vector (μ)')
plt.legend(loc='upper right')

# Keep axis equal to prevent distortion
plt.axis('equal')
plt.tight_layout()

# Save the figure
plt.savefig(os.path.join(save_dir, 'zero_correlation.png'), dpi=300, bbox_inches='tight')
plt.close()

# 1. Visualization 1: Correlation comparison (original, negative, zero)
print("Generating correlation comparison visualization...")
plt.figure(figsize=(15, 5))

# Plot original positive correlation
plt.subplot(1, 3, 1)
rv_pos, _ = create_bivariate_normal(mu, rho_original, sigma_x, sigma_y)
Z_pos = rv_pos.pdf(pos)
contour_levels = np.linspace(0, 0.15, 10)
plt.contourf(X, Y, Z_pos, levels=contour_levels, cmap='viridis', alpha=0.7)
plt.contour(X, Y, Z_pos, levels=contour_levels, colors='black', linewidths=0.8)
plt.title(f'Positive Correlation (ρ = {rho_original})')
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.axis('equal')
plt.grid(alpha=0.3)
plt.scatter(0, 0, color='red', s=50, marker='x')

# Plot negative correlation
plt.subplot(1, 3, 2)
rv_neg, _ = create_bivariate_normal(mu, -rho_original, sigma_x, sigma_y)
Z_neg = rv_neg.pdf(pos)
plt.contourf(X, Y, Z_neg, levels=contour_levels, cmap='viridis', alpha=0.7)
plt.contour(X, Y, Z_neg, levels=contour_levels, colors='black', linewidths=0.8)
plt.title(f'Negative Correlation (ρ = {-rho_original})')
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.axis('equal')
plt.grid(alpha=0.3)
plt.scatter(0, 0, color='red', s=50, marker='x')

# Plot zero correlation
plt.subplot(1, 3, 3)
rv_zero, _ = create_bivariate_normal(mu, 0, sigma_x, sigma_y)
Z_zero = rv_zero.pdf(pos)
plt.contourf(X, Y, Z_zero, levels=contour_levels, cmap='viridis', alpha=0.7)
plt.contour(X, Y, Z_zero, levels=contour_levels, colors='black', linewidths=0.8)
plt.title('Zero Correlation (ρ = 0)')
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.axis('equal')
plt.grid(alpha=0.3)
plt.scatter(0, 0, color='red', s=50, marker='x')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'correlation_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Visualization 2: Variance comparison (equal vs different variances)
print("Generating variance comparison visualization...")
plt.figure(figsize=(15, 5))

# Original with different variances
plt.subplot(1, 3, 1)
plt.contourf(X, Y, Z_pos, levels=contour_levels, cmap='viridis', alpha=0.7)
plt.contour(X, Y, Z_pos, levels=contour_levels, colors='black', linewidths=0.8)
plt.title(f'Different Variances\n$\\sigma_x = {sigma_x}, \\sigma_y = {sigma_y}$')
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.axis('equal')
plt.grid(alpha=0.3)
plt.scatter(0, 0, color='red', s=50, marker='x')

# Equal variances (larger both)
plt.subplot(1, 3, 2)
sigma_equal_large = 2
rv_equal_large, _ = create_bivariate_normal(mu, rho_original, sigma_equal_large, sigma_equal_large)
Z_equal_large = rv_equal_large.pdf(pos)
equal_large_levels = np.linspace(0, 0.08, 10)  # Adjusted for the different max density
plt.contourf(X, Y, Z_equal_large, levels=equal_large_levels, cmap='viridis', alpha=0.7)
plt.contour(X, Y, Z_equal_large, levels=equal_large_levels, colors='black', linewidths=0.8)
plt.title(f'Equal Variances (Large)\n$\\sigma_x = \\sigma_y = {sigma_equal_large}$')
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.axis('equal')
plt.grid(alpha=0.3)
plt.scatter(0, 0, color='red', s=50, marker='x')

# Equal variances (smaller both)
plt.subplot(1, 3, 3)
sigma_equal_small = 1
rv_equal_small, _ = create_bivariate_normal(mu, rho_original, sigma_equal_small, sigma_equal_small)
Z_equal_small = rv_equal_small.pdf(pos)
equal_small_levels = np.linspace(0, 0.15, 10)
plt.contourf(X, Y, Z_equal_small, levels=equal_small_levels, cmap='viridis', alpha=0.7)
plt.contour(X, Y, Z_equal_small, levels=equal_small_levels, colors='black', linewidths=0.8)
plt.title(f'Equal Variances (Small)\n$\\sigma_x = \\sigma_y = {sigma_equal_small}$')
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.axis('equal')
plt.grid(alpha=0.3)
plt.scatter(0, 0, color='red', s=50, marker='x')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'variance_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Visualization 3: Mean vector movement
print("Generating mean vector visualization...")
plt.figure(figsize=(15, 5))

# Original at origin
plt.subplot(1, 3, 1)
plt.contourf(X, Y, Z_pos, levels=contour_levels, cmap='viridis', alpha=0.7)
plt.contour(X, Y, Z_pos, levels=contour_levels, colors='black', linewidths=0.8)
plt.title('Mean at Origin\n$\\mu = [0, 0]$')
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.axis('equal')
plt.grid(alpha=0.3)
plt.scatter(0, 0, color='red', s=50, marker='x')

# Mean shifted to [2, 1]
plt.subplot(1, 3, 2)
mu_shifted1 = [2, 1]
rv_shifted1, _ = create_bivariate_normal(mu_shifted1, rho_original, sigma_x, sigma_y)
Z_shifted1 = rv_shifted1.pdf(pos)
plt.contourf(X, Y, Z_shifted1, levels=contour_levels, cmap='viridis', alpha=0.7)
plt.contour(X, Y, Z_shifted1, levels=contour_levels, colors='black', linewidths=0.8)
plt.title(f'Mean Shifted\n$\\mu = {mu_shifted1}$')
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.axis('equal')
plt.grid(alpha=0.3)
plt.scatter(mu_shifted1[0], mu_shifted1[1], color='red', s=50, marker='x')

# Mean shifted to [-1, 2]
plt.subplot(1, 3, 3)
mu_shifted2 = [-1, 2]
rv_shifted2, _ = create_bivariate_normal(mu_shifted2, rho_original, sigma_x, sigma_y)
Z_shifted2 = rv_shifted2.pdf(pos)
plt.contourf(X, Y, Z_shifted2, levels=contour_levels, cmap='viridis', alpha=0.7)
plt.contour(X, Y, Z_shifted2, levels=contour_levels, colors='black', linewidths=0.8)
plt.title(f'Mean Shifted\n$\\mu = {mu_shifted2}$')
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.axis('equal')
plt.grid(alpha=0.3)
plt.scatter(mu_shifted2[0], mu_shifted2[1], color='red', s=50, marker='x')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'mean_vector_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. Visualization 4: 3D Surface vs Contour Plot
print("Generating 3D surface vs contour plot visualization...")
fig = plt.figure(figsize=(15, 6))

# 3D Surface plot
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax1.plot_surface(X, Y, Z_pos, cmap='viridis', alpha=0.8)
ax1.set_title('3D PDF Surface')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Probability Density')
ax1.view_init(elev=30, azim=30)

# Contour plot
ax2 = fig.add_subplot(1, 2, 2)
contour = ax2.contourf(X, Y, Z_pos, levels=contour_levels, cmap='viridis')
ax2.contour(X, Y, Z_pos, levels=contour_levels, colors='black', alpha=0.7)
CS = ax2.contour(X, Y, Z_pos, levels=[contour_levels[-1]], colors='red', linewidths=2)
ax2.clabel(CS, inline=True, fontsize=10, fmt='%.2f')
ax2.set_title('Equivalent Contour Plot')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.grid(alpha=0.3)
ax2.axis('equal')
ax2.scatter(0, 0, color='red', s=50, marker='x')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'surface_vs_contour.png'), dpi=300, bbox_inches='tight')
plt.close()

# 5. Visualization 5: Multiple correlation values
print("Generating multiple correlation values visualization...")
fig = plt.figure(figsize=(12, 10))
correlations = [-0.9, -0.5, 0, 0.5, 0.9]
num_plots = len(correlations)
num_rows = (num_plots + 2) // 3  # Calculate number of rows needed

for i, rho in enumerate(correlations):
    ax = fig.add_subplot(num_rows, 3, i+1)
    rv, cov = create_bivariate_normal(mu, rho, sigma_x, sigma_y)
    Z = rv.pdf(pos)
    ax.contourf(X, Y, Z, levels=np.linspace(0, 0.15, 10), cmap='viridis', alpha=0.7)
    ax.contour(X, Y, Z, levels=np.linspace(0, 0.15, 10), colors='black', linewidths=0.8)
    ax.set_title(f'ρ = {rho}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(alpha=0.3)
    ax.axis('equal')
    ax.scatter(0, 0, color='red', s=30, marker='x')
    
    # Add eigenvalue arrows to show principal axes
    evals, evecs = np.linalg.eigh(cov)
    for eval, evec in zip(evals, evecs.T):
        scale = np.sqrt(eval) * 2  # Scale arrow by sqrt of eigenvalue
        ax.arrow(0, 0, scale*evec[0], scale*evec[1], head_width=0.2, 
                 head_length=0.3, fc='blue', ec='blue', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'multiple_correlations.png'), dpi=300, bbox_inches='tight')
plt.close()

# 6. Visualization 6: Covariance matrix visualization
print("Generating covariance matrix visualization...")
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# For positive correlation
rho = rho_original
_, cov_pos = create_bivariate_normal(mu, rho, sigma_x, sigma_y)

# For zero correlation
_, cov_zero = create_bivariate_normal(mu, 0, sigma_x, sigma_y)

# Convert covariance matrices to numpy arrays to ensure proper indexing
cov_pos = np.array(cov_pos)
cov_zero = np.array(cov_zero)

# Create visual representation of covariance matrices
im1 = ax[0].imshow(cov_pos, cmap='Blues', vmin=0)
ax[0].set_title(f'Covariance Matrix (ρ = {rho})')
ax[0].set_xticks([0, 1])
ax[0].set_yticks([0, 1])
ax[0].set_xticklabels(['X', 'Y'])
ax[0].set_yticklabels(['X', 'Y'])

# Add text annotations to the matrix
for i in range(2):
    for j in range(2):
        text = ax[0].text(j, i, f'{cov_pos[i, j]:.2f}',
                          ha='center', va='center', color='black')

im2 = ax[1].imshow(cov_zero, cmap='Blues', vmin=0)
ax[1].set_title('Covariance Matrix (ρ = 0)')
ax[1].set_xticks([0, 1])
ax[1].set_yticks([0, 1])
ax[1].set_xticklabels(['X', 'Y'])
ax[1].set_yticklabels(['X', 'Y'])

# Add text annotations to the matrix
for i in range(2):
    for j in range(2):
        text = ax[1].text(j, i, f'{cov_zero[i, j]:.2f}',
                          ha='center', va='center', color='black')

plt.colorbar(im1, ax=ax[0], label='Covariance Value')
plt.colorbar(im2, ax=ax[1], label='Covariance Value')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'covariance_matrices.png'), dpi=300, bbox_inches='tight')
plt.close()

# 7. Visualization 7: Annotated contour plot explaining key features
print("Generating annotated educational visualization...")
plt.figure(figsize=(10, 8))

# Generate the contour plot
rv, cov = create_bivariate_normal(mu, rho_original, sigma_x, sigma_y)
Z = rv.pdf(pos)
contour_filled = plt.contourf(X, Y, Z, levels=np.linspace(0, 0.15, 15), cmap='viridis', alpha=0.7)
contours = plt.contour(X, Y, Z, levels=np.linspace(0, 0.15, 10), colors='black', linewidths=0.8)
plt.colorbar(contour_filled, label='Probability Density')

# Label the peak
plt.scatter(0, 0, color='red', s=80, marker='x')
plt.annotate('Mean Vector μ = [0, 0]\n(Peak Density)', 
             xy=(0, 0), xytext=(-3, 2), fontsize=10,
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

# Add annotations for correlation
plt.annotate('Positive Correlation (ρ = 0.6)\nTilted Ellipses', 
             xy=(2, 1), xytext=(2, 2), fontsize=10,
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

# Add annotations for variances
plt.annotate('Greater Variance in X-direction\n(σ$_x$ = 2 > σ$_y$ = 1)', 
             xy=(-2, 0), xytext=(-4, -1), fontsize=10,
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

# Mark principal axes
evals, evecs = np.linalg.eigh(cov)
for i, (eval, evec) in enumerate(zip(evals, evecs.T)):
    scale = np.sqrt(eval) * 2
    plt.arrow(0, 0, scale*evec[0], scale*evec[1], head_width=0.2, 
             head_length=0.3, fc='blue', ec='blue', alpha=0.7)
    plt.annotate(f'Principal Axis {i+1}', 
                 xy=(scale*evec[0]/2, scale*evec[1]/2), 
                 xytext=(scale*evec[0]/2 + (-1 if i==0 else 1), scale*evec[1]/2 + 0.5), 
                 fontsize=9,
                 arrowprops=dict(facecolor='blue', shrink=0.05, width=1))

# Label the innermost contour
plt.clabel(contours, inline=True, fontsize=8, fmt='%.2f', manual=[(0, 0)])

# Add title and labels
plt.title('Anatomy of a Bivariate Normal Contour Plot', fontsize=14)
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.grid(alpha=0.3)
plt.axis('equal')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'annotated_contour_plot.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAll solution visualizations saved to '{save_dir}'")
print("\nKey findings from the visualizations:")
print("1. The positive correlation (ρ = 0.6) is indicated by the upward tilt of the ellipses.")
print("2. The mean vector [0, 0] is located at the point of highest density (center of innermost contour).")
print("3. The greater spread along the x-axis indicates Var(X) > Var(Y) (σ$_x$ = 2 > σ$_y$ = 1).")
print("4. When correlation is zero, the ellipses align with the coordinate axes with no tilting.")
print("5. The principal axes of the ellipses are determined by the eigenvectors of the covariance matrix.") 