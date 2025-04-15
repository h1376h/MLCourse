import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns

print("\n=== EXAMPLE 1: BIVARIATE NORMAL DISTRIBUTION VISUALIZATION ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the Lectures/2 directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images", "Multivariate_Normal")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Define the distributions
print("\nStep 1: Define the distributions")

# Define the parameters for the correlated and uncorrelated distributions
mu = np.array([0, 0])  # Mean vector (same for both distributions)

# Covariance matrices
cov_correlated = np.array([[1.0, 0.7],
                           [0.7, 1.0]])  # Correlated case
cov_uncorrelated = np.array([[1.0, 0.0],
                            [0.0, 1.0]])  # Uncorrelated case

print(f"Mean vector μ = {mu}")
print(f"\nCorrelated covariance matrix:")
print(f"{cov_correlated}")
print(f"\nUncorrelated covariance matrix:")
print(f"{cov_uncorrelated}")

# Step 2: Generate data points from these distributions
print("\nStep 2: Generate data points from these distributions")

# Number of samples
n_samples = 1000

# Generate samples from the multivariate normal distributions
samples_correlated = np.random.multivariate_normal(mu, cov_correlated, n_samples)
samples_uncorrelated = np.random.multivariate_normal(mu, cov_uncorrelated, n_samples)

print(f"Generated {n_samples} samples from each distribution")
print(f"First 5 samples from correlated distribution:")
for i in range(5):
    print(f"Sample {i+1}: {samples_correlated[i]}")

# Step 3: Visualize the probability density functions
print("\nStep 3: Visualize the probability density functions")

# Create a meshgrid for 3D plotting
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Calculate the PDF values
rv_correlated = stats.multivariate_normal(mu, cov_correlated)
Z_correlated = rv_correlated.pdf(pos)

rv_uncorrelated = stats.multivariate_normal(mu, cov_uncorrelated)
Z_uncorrelated = rv_uncorrelated.pdf(pos)

# Create separate plots instead of subplots

# Plot 1: 3D plot for correlated variables
fig1 = plt.figure(figsize=(10, 8))
ax1 = fig1.add_subplot(111, projection='3d')
surf1 = ax1.plot_surface(X, Y, Z_correlated, cmap=cm.viridis, linewidth=0, antialiased=True)
ax1.set_xlabel('X₁')
ax1.set_ylabel('X₂')
ax1.set_zlabel('Density')
ax1.set_title('PDF of Correlated Bivariate Normal Distribution\n(ρ = 0.7)')
fig1.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
plt.tight_layout()
fig1.savefig(os.path.join(images_dir, 'bivariate_normal_3d_correlated.png'), dpi=100, bbox_inches='tight')
plt.close(fig1)
print(f"Generated visualization '{os.path.join(images_dir, 'bivariate_normal_3d_correlated.png')}'")

# Plot 2: 3D plot for uncorrelated variables
fig2 = plt.figure(figsize=(10, 8))
ax2 = fig2.add_subplot(111, projection='3d')
surf2 = ax2.plot_surface(X, Y, Z_uncorrelated, cmap=cm.viridis, linewidth=0, antialiased=True)
ax2.set_xlabel('X₁')
ax2.set_ylabel('X₂')
ax2.set_zlabel('Density')
ax2.set_title('PDF of Uncorrelated Bivariate Normal Distribution\n(ρ = 0)')
fig2.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
plt.tight_layout()
fig2.savefig(os.path.join(images_dir, 'bivariate_normal_3d_uncorrelated.png'), dpi=100, bbox_inches='tight')
plt.close(fig2)
print(f"Generated visualization '{os.path.join(images_dir, 'bivariate_normal_3d_uncorrelated.png')}'")

# Plot 3: Contour plot for correlated variables
fig3 = plt.figure(figsize=(10, 8))
ax3 = fig3.add_subplot(111)
contour1 = ax3.contour(X, Y, Z_correlated, levels=10, cmap=cm.viridis)
ax3.scatter(samples_correlated[:, 0], samples_correlated[:, 1], alpha=0.3, c='red', s=8)
ax3.set_xlabel('X₁')
ax3.set_ylabel('X₂')
ax3.grid(alpha=0.3)
ax3.set_title('Contour Plot with Samples (ρ = 0.7)')
fig3.colorbar(contour1, ax=ax3, shrink=0.7)
plt.tight_layout()
fig3.savefig(os.path.join(images_dir, 'bivariate_normal_contour_correlated.png'), dpi=100, bbox_inches='tight')
plt.close(fig3)
print(f"Generated visualization '{os.path.join(images_dir, 'bivariate_normal_contour_correlated.png')}'")

# Plot 4: Contour plot for uncorrelated variables
fig4 = plt.figure(figsize=(10, 8))
ax4 = fig4.add_subplot(111)
contour2 = ax4.contour(X, Y, Z_uncorrelated, levels=10, cmap=cm.viridis)
ax4.scatter(samples_uncorrelated[:, 0], samples_uncorrelated[:, 1], alpha=0.3, c='blue', s=8)
ax4.set_xlabel('X₁')
ax4.set_ylabel('X₂')
ax4.grid(alpha=0.3)
ax4.set_title('Contour Plot with Samples (ρ = 0)')
fig4.colorbar(contour2, ax=ax4, shrink=0.7)
plt.tight_layout()
fig4.savefig(os.path.join(images_dir, 'bivariate_normal_contour_uncorrelated.png'), dpi=100, bbox_inches='tight')
plt.close(fig4)
print(f"Generated visualization '{os.path.join(images_dir, 'bivariate_normal_contour_uncorrelated.png')}'")

# Plot 5: 3D view of both distributions
fig5 = plt.figure(figsize=(10, 8))
ax5 = fig5.add_subplot(111, projection='3d')
# Plot wireframe for both distributions
ax5.plot_wireframe(X, Y, Z_correlated, color='red', alpha=0.3, linewidth=0.5, label='ρ = 0.7')
ax5.plot_wireframe(X, Y, Z_uncorrelated, color='blue', alpha=0.3, linewidth=0.5, label='ρ = 0')
ax5.set_xlabel('X₁')
ax5.set_ylabel('X₂')
ax5.set_zlabel('Density')
ax5.set_title('3D Comparison of Distributions')
ax5.legend()
plt.tight_layout()
fig5.savefig(os.path.join(images_dir, 'bivariate_normal_3d_comparison.png'), dpi=100, bbox_inches='tight')
plt.close(fig5)
print(f"Generated visualization '{os.path.join(images_dir, 'bivariate_normal_3d_comparison.png')}'")

# Plot 6: Comparison of the two distributions (contours only)
fig6 = plt.figure(figsize=(10, 8))
ax6 = fig6.add_subplot(111)
contour_corr = ax6.contour(X, Y, Z_correlated, levels=5, colors='red', alpha=0.7)
contour_uncorr = ax6.contour(X, Y, Z_uncorrelated, levels=5, colors='blue', alpha=0.7)
ax6.set_xlabel('X₁')
ax6.set_ylabel('X₂')
ax6.grid(alpha=0.3)
ax6.set_title('Comparison of Contours\nRed: ρ = 0.7, Blue: ρ = 0')

# Add explanatory text
ax6.text(0.5, -2.5, 
         "Correlation affects the elliptical shape of the distribution.\n"
         "With ρ = 0.7, the distribution is elongated along the y=x line,\n"
         "indicating that high values of X₁ tend to occur with high values of X₂.",
         ha='center', fontsize=9)

plt.tight_layout()
fig6.savefig(os.path.join(images_dir, 'bivariate_normal_contour_comparison.png'), dpi=100, bbox_inches='tight')
plt.close(fig6)
print(f"Generated visualization '{os.path.join(images_dir, 'bivariate_normal_contour_comparison.png')}'")

# Also keep the original combined figure for reference
fig = plt.figure(figsize=(18, 12))

# 3D plot for correlated variables
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
surf1 = ax1.plot_surface(X, Y, Z_correlated, cmap=cm.viridis, linewidth=0, antialiased=True)
ax1.set_xlabel('X₁')
ax1.set_ylabel('X₂')
ax1.set_zlabel('Density')
ax1.set_title('PDF of Correlated Bivariate Normal Distribution\n(ρ = 0.7)')
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

# 3D plot for uncorrelated variables
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
surf2 = ax2.plot_surface(X, Y, Z_uncorrelated, cmap=cm.viridis, linewidth=0, antialiased=True)
ax2.set_xlabel('X₁')
ax2.set_ylabel('X₂')
ax2.set_zlabel('Density')
ax2.set_title('PDF of Uncorrelated Bivariate Normal Distribution\n(ρ = 0)')
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

# Contour plot for correlated variables
ax3 = fig.add_subplot(2, 3, 3)
contour1 = ax3.contour(X, Y, Z_correlated, levels=10, cmap=cm.viridis)
ax3.scatter(samples_correlated[:, 0], samples_correlated[:, 1], alpha=0.3, c='red', s=8)
ax3.set_xlabel('X₁')
ax3.set_ylabel('X₂')
ax3.grid(alpha=0.3)
ax3.set_title('Contour Plot with Samples (ρ = 0.7)')
fig.colorbar(contour1, ax=ax3, shrink=0.5)

# Contour plot for uncorrelated variables
ax4 = fig.add_subplot(2, 3, 4)
contour2 = ax4.contour(X, Y, Z_uncorrelated, levels=10, cmap=cm.viridis)
ax4.scatter(samples_uncorrelated[:, 0], samples_uncorrelated[:, 1], alpha=0.3, c='blue', s=8)
ax4.set_xlabel('X₁')
ax4.set_ylabel('X₂')
ax4.grid(alpha=0.3)
ax4.set_title('Contour Plot with Samples (ρ = 0)')
fig.colorbar(contour2, ax=ax4, shrink=0.5)

# 3D view of both distributions
ax5 = fig.add_subplot(2, 3, 5, projection='3d')
# Plot wireframe for both distributions
ax5.plot_wireframe(X, Y, Z_correlated, color='red', alpha=0.3, linewidth=0.5, label='ρ = 0.7')
ax5.plot_wireframe(X, Y, Z_uncorrelated, color='blue', alpha=0.3, linewidth=0.5, label='ρ = 0')
ax5.set_xlabel('X₁')
ax5.set_ylabel('X₂')
ax5.set_zlabel('Density')
ax5.set_title('3D Comparison of Distributions')
ax5.legend()

# Comparison of the two distributions (contours only)
ax6 = fig.add_subplot(2, 3, 6)
contour_corr = ax6.contour(X, Y, Z_correlated, levels=5, colors='red', alpha=0.7)
contour_uncorr = ax6.contour(X, Y, Z_uncorrelated, levels=5, colors='blue', alpha=0.7)
ax6.set_xlabel('X₁')
ax6.set_ylabel('X₂')
ax6.grid(alpha=0.3)
ax6.set_title('Comparison of Contours\nRed: ρ = 0.7, Blue: ρ = 0')

# Add explanatory text
ax6.text(0.5, -2.5, 
         "Correlation affects the elliptical shape of the distribution.\n"
         "With ρ = 0.7, the distribution is elongated along the y=x line,\n"
         "indicating that high values of X₁ tend to occur with high values of X₂.",
         ha='center', fontsize=9)


plt.tight_layout()
fig.savefig(os.path.join(images_dir, 'bivariate_normal_comparison.png'), dpi=100, bbox_inches='tight')
plt.close(fig)

print(f"Generated visualization '{os.path.join(images_dir, 'bivariate_normal_comparison.png')}'")

# Create a separate figure for joint and marginal distributions
plt.figure(figsize=(10, 10))
# Create a 2x2 grid for joint and marginal distributions
g = sns.jointplot(
    x=samples_correlated[:, 0],
    y=samples_correlated[:, 1],
    kind="scatter",
    color="red",
    alpha=0.3,
    s=8,
    height=8,
    ratio=3,
    marginal_kws=dict(bins=30, fill=True),
)
g.plot_joint(sns.kdeplot, levels=10, color="k", zorder=1)
g.fig.suptitle('Joint and Marginal Distributions (ρ = 0.7)', y=1.02, fontsize=12)
g.set_axis_labels('X₁', 'X₂')

# Save just the jointplot figure
g.savefig(os.path.join(images_dir, 'bivariate_normal_joint_marginal.png'), dpi=100, bbox_inches='tight')
plt.close(g.fig)

# Add a new plot: Density comparison along X₁ axis where Y=0
fig7 = plt.figure(figsize=(12, 6))
ax7 = fig7.add_subplot(111)

# Create a cross-section of the PDFs along x-axis (y=0)
x_crosssection = np.linspace(-3, 3, 1000)
y_crosssection = 0
pdf_correlated = np.zeros_like(x_crosssection)
pdf_uncorrelated = np.zeros_like(x_crosssection)

# Calculate PDF values for each point along the x-axis
for i, x_val in enumerate(x_crosssection):
    point = np.array([x_val, y_crosssection])
    pdf_correlated[i] = rv_correlated.pdf(point)
    pdf_uncorrelated[i] = rv_uncorrelated.pdf(point)

# Plot the cross-sections
ax7.plot(x_crosssection, pdf_correlated, 'r-', linewidth=2.5, label='Correlated (ρ = 0.7)')
ax7.plot(x_crosssection, pdf_uncorrelated, 'b-', linewidth=2.5, label='Uncorrelated (ρ = 0)')

# Add vertical lines for clarity
ax7.axvline(x=0, color='k', linestyle='--', alpha=0.5)
ax7.axhline(y=0, color='k', linestyle='--', alpha=0.5)

# Enhance the plot
ax7.set_xlabel('X₁', fontsize=12)
ax7.set_ylabel('Probability Density', fontsize=12)
ax7.set_title('Density Comparison Along X₁-axis (X₂=0)', fontsize=14)
ax7.grid(alpha=0.3)
ax7.legend(fontsize=12)

# Add explanatory text
ax7.text(1.5, 0.3, 
         "When X₂=0, both distributions have\nidentical densities along the X₁-axis.\n"
         "This demonstrates that the correlation\nonly affects the joint distribution,\n"
         "not the marginal distributions.",
         fontsize=10, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

plt.tight_layout()
fig7.savefig(os.path.join(images_dir, 'bivariate_normal_density_slice.png'), dpi=100, bbox_inches='tight')
plt.close(fig7)
print(f"Generated visualization '{os.path.join(images_dir, 'bivariate_normal_density_slice.png')}'")

# Mathematical explanation
print("\nMathematical Explanation:")
print("The bivariate normal PDF with mean vector μ and covariance matrix Σ is given by:")
print("f(x) = (1/2π|Σ|^(1/2)) * exp(-1/2 * (x-μ)ᵀΣ⁻¹(x-μ))")

print("\nFor the correlated case (ρ = 0.7):")
det_corr = np.linalg.det(cov_correlated)
inv_corr = np.linalg.inv(cov_correlated)
print(f"Determinant |Σ| = {det_corr:.4f}")
print(f"Inverse Σ⁻¹ = \n{inv_corr}")

print("\nFor the uncorrelated case (ρ = 0):")
det_uncorr = np.linalg.det(cov_uncorrelated)
inv_uncorr = np.linalg.inv(cov_uncorrelated)
print(f"Determinant |Σ| = {det_uncorr:.4f}")
print(f"Inverse Σ⁻¹ = \n{inv_uncorr}")

print("\nInterpretation:")
print("1. The correlated distribution has elliptical contours tilted at an angle, showing the relationship between variables.")
print("2. The uncorrelated distribution has circular contours as the variables vary independently.")
print("3. The probability density is highest at the mean (0,0) and decreases as we move away.")
print("4. The correlation coefficient ρ = 0.7 indicates a strong positive relationship between X₁ and X₂.")
print("5. In the correlated case, knowing the value of one variable provides information about the likely value of the other.")

# Display plots if running in interactive mode
plt.show() 