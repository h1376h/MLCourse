import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import multivariate_normal, norm
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_7_Quiz_12")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Understanding the Problem
print_step_header(1, "Understanding the Problem")

print("Given:")
print("- Bivariate normal posterior distribution for parameters θ₁ and θ₂:")
print("  p(θ₁, θ₂|D) ∝ exp(-0.5 * [θ₁-3, θ₂-2]ᵀ * [4 1; 1 2]⁻¹ * [θ₁-3, θ₂-2])")
print("\nTask:")
print("1. Write a factorized variational approximation q(θ₁, θ₂) = q₁(θ₁)q₂(θ₂) where both q₁ and q₂ are normal")
print("2. Explain the key limitation of this factorized approximation for this particular posterior")
print("3. Briefly describe how the ELBO is used in variational inference")

# Step 2: Analyzing the true posterior distribution
print_step_header(2, "Analyzing the True Posterior Distribution")

# Define the parameters of the true posterior
mean = np.array([3, 2])
cov_matrix = np.array([[4, 1], [1, 2]])
precision_matrix = np.linalg.inv(cov_matrix)

print("The true posterior is a bivariate normal distribution:")
print(f"Mean vector: μ = {mean}")
print(f"Covariance matrix: Σ = {cov_matrix}")
print(f"Precision matrix: Λ = {precision_matrix}")
print(f"Correlation coefficient: ρ = {cov_matrix[0,1] / np.sqrt(cov_matrix[0,0] * cov_matrix[1,1]):.4f}")

# Create a grid of points
x = np.linspace(-2, 8, 300)
y = np.linspace(-3, 7, 300)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Calculate the density at each point
rv = multivariate_normal(mean, cov_matrix)
Z = rv.pdf(pos)

# Plot the true posterior
plt.figure(figsize=(10, 8))
contour = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar(contour, label='Density')
plt.contour(X, Y, Z, levels=5, colors='white', alpha=0.5, linestyles='--')
plt.scatter(mean[0], mean[1], color='red', s=100, label='Mean (3, 2)')

# Add confidence ellipses
def confidence_ellipse(ax, x0, y0, cov, n_std=2.0, **kwargs):
    """
    Create a plot of the covariance confidence ellipse.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    x0, y0 : float
        The center of the ellipse.
    cov : array-like, shape (2, 2)
        The covariance matrix.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    **kwargs
        Additional arguments passed to the ellipse patch.
    """
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    
    ellipse = Ellipse((0, 0), 
                       width=ell_radius_x * 2, 
                       height=ell_radius_y * 2, 
                       **kwargs)
    
    # Compute standard deviation scaling
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(x0, y0)
    
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

confidence_ellipse(plt.gca(), mean[0], mean[1], cov_matrix, n_std=1.0, 
                   edgecolor='red', facecolor='none', label='68% Confidence')
confidence_ellipse(plt.gca(), mean[0], mean[1], cov_matrix, n_std=2.0, 
                   edgecolor='red', facecolor='none', linestyle='--', label='95% Confidence')

plt.xlabel('θ₁', fontsize=14)
plt.ylabel('θ₂', fontsize=14)
plt.title('True Posterior Distribution p(θ₁, θ₂|D)', fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "true_posterior.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 3: Deriving the Factorized Variational Approximation
print_step_header(3, "Deriving the Factorized Variational Approximation")

print("For a factorized variational approximation q(θ₁, θ₂) = q₁(θ₁)q₂(θ₂), we need to find")
print("normal distributions q₁(θ₁) = N(θ₁|μ₁, σ₁²) and q₂(θ₂) = N(θ₂|μ₂, σ₂²) that best")
print("approximate the true posterior.")
print("\nThe optimal q₁(θ₁) can be derived by taking the expectation of log p(θ₁, θ₂|D) with respect to q₂(θ₂).")
print("Similarly, the optimal q₂(θ₂) can be derived by taking the expectation of log p(θ₁, θ₂|D) with respect to q₁(θ₁).")
print("\nFor a multivariate Gaussian, this gives us the following equations:")

# For a factorized Gaussian variational approximation
# The optimal parameters for the factorized distribution can be obtained 
# from the true distribution parameters
q1_variance = 1/precision_matrix[0, 0]
q1_mean = mean[0]
q2_variance = 1/precision_matrix[1, 1]
q2_mean = mean[1]

print(f"q₁(θ₁) = N(θ₁|{q1_mean:.2f}, {q1_variance:.2f})")
print(f"q₂(θ₂) = N(θ₂|{q2_mean:.2f}, {q2_variance:.2f})")
print("\nNote: In this factorized approximation, we're ignoring the off-diagonal element of the precision matrix,")
print("which represents the correlation between θ₁ and θ₂.")

# Calculate the factorized approximation on the grid
q1 = norm.pdf(X, loc=q1_mean, scale=np.sqrt(q1_variance))
q2 = norm.pdf(Y, loc=q2_mean, scale=np.sqrt(q2_variance))
Z_q = np.multiply.outer(q1[0,:], q2[:,0])

# Plot the comparison between true posterior and variational approximation
plt.figure(figsize=(12, 10))

# First subplot: True posterior
plt.subplot(2, 2, 1)
contour1 = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar(contour1, label='Density')
plt.contour(X, Y, Z, levels=5, colors='white', alpha=0.5, linestyles='--')
plt.scatter(mean[0], mean[1], color='red', s=100)
confidence_ellipse(plt.gca(), mean[0], mean[1], cov_matrix, n_std=2.0, 
                   edgecolor='red', facecolor='none', linestyle='--')
plt.xlabel('θ₁', fontsize=12)
plt.ylabel('θ₂', fontsize=12)
plt.title('True Posterior p(θ₁, θ₂|D)', fontsize=14)
plt.grid(True)

# Second subplot: Variational approximation
plt.subplot(2, 2, 2)
contour2 = plt.contourf(X, Y, Z_q, levels=20, cmap='plasma')
plt.colorbar(contour2, label='Density')
plt.contour(X, Y, Z_q, levels=5, colors='white', alpha=0.5, linestyles='--')
plt.scatter(q1_mean, q2_mean, color='red', s=100)

# Add rectangular confidence regions (since the variables are independent in the approximation)
plt.axhline(y=q2_mean + 2*np.sqrt(q2_variance), color='white', linestyle='-')
plt.axhline(y=q2_mean - 2*np.sqrt(q2_variance), color='white', linestyle='-')
plt.axvline(x=q1_mean + 2*np.sqrt(q1_variance), color='white', linestyle='-')
plt.axvline(x=q1_mean - 2*np.sqrt(q1_variance), color='white', linestyle='-')

plt.xlabel('θ₁', fontsize=12)
plt.ylabel('θ₂', fontsize=12)
plt.title('Variational Approximation q₁(θ₁)q₂(θ₂)', fontsize=14)
plt.grid(True)

# Third subplot: Difference
plt.subplot(2, 2, 3)
diff = Z - Z_q
max_abs_diff = np.max(np.abs(diff))
contour3 = plt.contourf(X, Y, diff, levels=20, cmap='coolwarm', vmin=-max_abs_diff, vmax=max_abs_diff)
plt.colorbar(contour3, label='Difference')
plt.contour(X, Y, diff, levels=5, colors='black', alpha=0.5, linestyles='--')
plt.scatter(mean[0], mean[1], color='red', s=100)
plt.xlabel('θ₁', fontsize=12)
plt.ylabel('θ₂', fontsize=12)
plt.title('Difference (True Posterior - Variational Approximation)', fontsize=14)
plt.grid(True)

# Fourth subplot: Marginal distributions
plt.subplot(2, 2, 4)

# Marginal for θ₁
theta1_range = np.linspace(-2, 8, 1000)
true_marginal_theta1 = norm.pdf(theta1_range, loc=mean[0], scale=np.sqrt(cov_matrix[0, 0]))
var_marginal_theta1 = norm.pdf(theta1_range, loc=q1_mean, scale=np.sqrt(q1_variance))

plt.plot(theta1_range, true_marginal_theta1, 'b-', label='True p(θ₁|D)', linewidth=2)
plt.plot(theta1_range, var_marginal_theta1, 'b--', label='Approx q₁(θ₁)', linewidth=2)

# Marginal for θ₂
theta2_range = np.linspace(-3, 7, 1000)
true_marginal_theta2 = norm.pdf(theta2_range, loc=mean[1], scale=np.sqrt(cov_matrix[1, 1]))
var_marginal_theta2 = norm.pdf(theta2_range, loc=q2_mean, scale=np.sqrt(q2_variance))

plt.plot(theta2_range, true_marginal_theta2, 'r-', label='True p(θ₂|D)', linewidth=2)
plt.plot(theta2_range, var_marginal_theta2, 'r--', label='Approx q₂(θ₂)', linewidth=2)

plt.xlabel('Parameter Value', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Marginal Distributions', fontsize=14)
plt.legend()
plt.grid(True)

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "comparison.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 4: Limitations of the Factorized Approximation
print_step_header(4, "Limitations of the Factorized Approximation")

print("Key limitation: The factorized approximation cannot capture correlations between variables.")
print(f"In the true posterior, θ₁ and θ₂ have a correlation coefficient of {cov_matrix[0,1] / np.sqrt(cov_matrix[0,0] * cov_matrix[1,1]):.4f}.")
print("But in the factorized approximation q(θ₁, θ₂) = q₁(θ₁)q₂(θ₂), we assume independence between θ₁ and θ₂.")
print("\nThis leads to several issues:")
print("1. The approximation is poor when the variables are strongly correlated")
print("2. It cannot represent the correct uncertainty in joint predictions")
print("3. The resulting confidence regions are axis-aligned rectangles instead of ellipses")
print("4. The KL divergence between the true posterior and the approximation will be larger")
print("   when correlations are stronger")

# Visualize the KL divergence and factorization gap
# We'll compute and visualize some metrics to quantify the approximation quality
kl_div = 0.5 * (np.log(q1_variance * q2_variance / np.linalg.det(cov_matrix)) + 
                np.trace(np.linalg.inv(np.diag([q1_variance, q2_variance])) @ cov_matrix) - 2)

print(f"\nKL divergence from approximation to true posterior: {kl_div:.4f}")
print("This value quantifies the information lost when using the factorized approximation.")
print("The larger the correlation in the true posterior, the larger this KL divergence will be.")

# Visualize the samples from both distributions
np.random.seed(42)
true_samples = np.random.multivariate_normal(mean, cov_matrix, 1000)

# Independent samples from factorized distribution
q1_samples = np.random.normal(q1_mean, np.sqrt(q1_variance), 1000)
q2_samples = np.random.normal(q2_mean, np.sqrt(q2_variance), 1000)
var_samples = np.column_stack((q1_samples, q2_samples))

plt.figure(figsize=(12, 6))

# Subplot for true posterior samples
plt.subplot(1, 2, 1)
plt.scatter(true_samples[:, 0], true_samples[:, 1], alpha=0.5, s=10)
plt.axhline(y=mean[1], color='red', linestyle='--')
plt.axvline(x=mean[0], color='red', linestyle='--')
confidence_ellipse(plt.gca(), mean[0], mean[1], cov_matrix, n_std=2.0, 
                   edgecolor='red', facecolor='none', linestyle='-')
plt.xlabel('θ₁', fontsize=12)
plt.ylabel('θ₂', fontsize=12)
plt.title('Samples from True Posterior', fontsize=14)
plt.grid(True)

# Subplot for variational approximation samples
plt.subplot(1, 2, 2)
plt.scatter(var_samples[:, 0], var_samples[:, 1], alpha=0.5, s=10)
plt.axhline(y=q2_mean, color='red', linestyle='--')
plt.axvline(x=q1_mean, color='red', linestyle='--')
# Draw a rectangle for 95% confidence region
rect_width = 2 * 2 * np.sqrt(q1_variance)
rect_height = 2 * 2 * np.sqrt(q2_variance)
rect_left = q1_mean - 2 * np.sqrt(q1_variance)
rect_bottom = q2_mean - 2 * np.sqrt(q2_variance)
plt.gca().add_patch(plt.Rectangle((rect_left, rect_bottom), rect_width, rect_height, 
                                  edgecolor='red', facecolor='none', linestyle='-'))
plt.xlabel('θ₁', fontsize=12)
plt.ylabel('θ₂', fontsize=12)
plt.title('Samples from Variational Approximation', fontsize=14)
plt.grid(True)

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "samples_comparison.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 5: The ELBO in Variational Inference
print_step_header(5, "The ELBO in Variational Inference")

print("The Evidence Lower Bound (ELBO) is a key concept in variational inference.")
print("It represents a lower bound on the log evidence (log marginal likelihood), log p(D).")
print("\nThe ELBO is defined as:")
print("ELBO = E_q[log p(θ, D)] - E_q[log q(θ)]")
print("     = E_q[log p(θ, D) - log q(θ)]")
print("\nIt can be rewritten as:")
print("ELBO = log p(D) - KL(q(θ) || p(θ|D))")
print("\nSince KL divergence is always non-negative, the ELBO is always ≤ log p(D).")
print("\nIn variational inference, we maximize the ELBO with respect to the variational parameters.")
print("This is equivalent to minimizing the KL divergence between q(θ) and p(θ|D).")
print("\nELBO maximization can be done through:")
print("1. Coordinate ascent (iteratively updating each component of q)")
print("2. Gradient-based optimization")
print("3. Stochastic variational inference (for large datasets)")
print("\nFor our factorized approximation q(θ₁, θ₂) = q₁(θ₁)q₂(θ₂), the ELBO calculation")
print("would involve expectations over both q₁ and q₂.")

# Plot a conceptual diagram of ELBO
plt.figure(figsize=(8, 6))

# Create some sample distributions for visualization
x_range = np.linspace(-3, 7, 1000)
true_posterior = norm.pdf(x_range, loc=3, scale=2)
q_dist = norm.pdf(x_range, loc=2.5, scale=1.5)

# Plot the distributions
plt.plot(x_range, true_posterior, 'b-', label='p(θ|D) (True Posterior)', linewidth=2)
plt.plot(x_range, q_dist, 'r--', label='q(θ) (Variational Approximation)', linewidth=2)

# Shade the KL divergence area
plt.fill_between(x_range, q_dist, true_posterior, 
                  where=(q_dist <= true_posterior), 
                  alpha=0.3, color='green', label='ELBO increases as this area decreases')

plt.xlabel('θ', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('ELBO and KL Divergence Visualization', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "elbo_concept.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Summary
print_step_header(6, "Summary")

print("In summary:")
print("1. The true posterior is a bivariate normal with mean [3, 2] and covariance [[4, 1], [1, 2]]")
print("2. Our factorized variational approximation is:")
print(f"   q₁(θ₁) = N(θ₁|{q1_mean:.2f}, {q1_variance:.2f})")
print(f"   q₂(θ₂) = N(θ₂|{q2_mean:.2f}, {q2_variance:.2f})")
print("3. The key limitation is that the factorized approximation cannot capture")
print("   the correlation between θ₁ and θ₂ in the true posterior")
print("4. The ELBO is used as an optimization objective in variational inference,")
print("   allowing us to find the best approximation within our chosen family")