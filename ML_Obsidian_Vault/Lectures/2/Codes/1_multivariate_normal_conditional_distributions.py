import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

print("\n=== EXAMPLE 3: CONDITIONAL DISTRIBUTIONS ===\n")

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

# Problem statement: find the conditional distribution of X₁ given X₂ = 18 and X₃ = 32
print("\nProblem: Given a trivariate normal distribution with variables X₁, X₂, and X₃")

# Mean vector and covariance matrix
mu = np.array([10, 20, 30])
sigma = np.array([
    [16, 8, 4],
    [8, 25, 5],
    [4, 5, 9]
])

print("\nMean vector μ =", mu)
print("\nCovariance matrix Σ =")
print(sigma)

# Step 1: Partition the mean vector and covariance matrix
print("\nStep 1: Partition the mean vector and covariance matrix")

# Partition the mean vector
mu_1 = mu[0]          # Mean of X₁
mu_23 = mu[1:3]       # Mean of [X₂, X₃]

print(f"μ₁ = {mu_1}")
print(f"μ₂₃ = {mu_23}")

# Partition the covariance matrix
sigma_11 = sigma[0, 0]                  # Variance of X₁
sigma_12 = sigma[0, 1:3]                # Covariance between X₁ and [X₂, X₃]
sigma_21 = sigma[1:3, 0]                # Covariance between [X₂, X₃] and X₁
sigma_22 = sigma[1:3, 1:3]              # Covariance matrix of [X₂, X₃]

print(f"\nσ₁₁ = {sigma_11}")
print(f"σ₁₂ = {sigma_12}")
print(f"σ₂₁ = {sigma_21}")
print(f"σ₂₂ = \n{sigma_22}")

# Observed values of X₂ and X₃
x_23 = np.array([18, 32])  # X₂ = 18, X₃ = 32

print(f"\nObserved values: X₂ = {x_23[0]}, X₃ = {x_23[1]}")

# Step 2: Calculate the conditional mean
print("\nStep 2: Calculate the conditional mean")

# Calculate sigma_22_inv
sigma_22_inv = np.linalg.inv(sigma_22)
print(f"Inverse of σ₂₂ = \n{sigma_22_inv}")

# Calculate deviation from mean for observed values
x_23_deviation = x_23 - mu_23
print(f"(x₂₃ - μ₂₃) = {x_23_deviation}")

# Calculate the term sigma_12 @ sigma_22_inv
adjustment_term = sigma_12 @ sigma_22_inv
print(f"σ₁₂ × σ₂₂⁻¹ = {adjustment_term}")

# Calculate the conditional mean
cond_mean = mu_1 + adjustment_term @ x_23_deviation
print(f"\nμ₁|₂₃ = μ₁ + σ₁₂ × σ₂₂⁻¹ × (x₂₃ - μ₂₃)")
print(f"μ₁|₂₃ = {mu_1} + {adjustment_term} × {x_23_deviation}")
print(f"μ₁|₂₃ = {mu_1} + {adjustment_term @ x_23_deviation}")
print(f"μ₁|₂₃ = {cond_mean}")

# Step 3: Calculate the conditional variance
print("\nStep 3: Calculate the conditional variance")

# Calculate the conditional variance
cond_variance = sigma_11 - adjustment_term @ sigma_21
print(f"σ₁|₂₃ = σ₁₁ - σ₁₂ × σ₂₂⁻¹ × σ₂₁")
print(f"σ₁|₂₃ = {sigma_11} - {adjustment_term} × {sigma_21}")
print(f"σ₁|₂₃ = {sigma_11} - {adjustment_term @ sigma_21}")
print(f"σ₁|₂₃ = {cond_variance}")

# Step 4: Interpret the result
print("\nStep 4: Interpret the result")

print(f"The conditional distribution X₁|X₂=18,X₃=32 ~ N({cond_mean:.4f}, {cond_variance:.4f})")
print(f"This means that given X₂ = 18 and X₃ = 32, X₁ follows a normal distribution")
print(f"with mean {cond_mean:.4f} and variance {cond_variance:.4f}.")

# Calculate 95% confidence interval for the conditional distribution
alpha = 0.05
z_score = stats.norm.ppf(1 - alpha/2)  # Two-tailed 95% confidence interval
margin_of_error = z_score * np.sqrt(cond_variance)
ci_lower = cond_mean - margin_of_error
ci_upper = cond_mean + margin_of_error

print(f"\n95% confidence interval for X₁|X₂=18,X₃=32:")
print(f"[{ci_lower:.4f}, {ci_upper:.4f}]")

# Visualize the conditional distribution
plt.figure(figsize=(12, 6))

# Plot the conditional distribution
x = np.linspace(cond_mean - 4*np.sqrt(cond_variance), 
               cond_mean + 4*np.sqrt(cond_variance), 1000)
y = stats.norm.pdf(x, cond_mean, np.sqrt(cond_variance))

plt.plot(x, y, 'b-', linewidth=2, label=f'X₁|X₂=18,X₃=32 ~ N({cond_mean:.2f}, {cond_variance:.2f})')
plt.axvline(cond_mean, color='r', linestyle='--', alpha=0.7, label=f'Conditional Mean = {cond_mean:.2f}')
plt.axvline(ci_lower, color='g', linestyle='--', alpha=0.7)
plt.axvline(ci_upper, color='g', linestyle='--', alpha=0.7, label=f'95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]')

# Fill the 95% confidence interval
plt.fill_between(x, y, where=((x >= ci_lower) & (x <= ci_upper)), color='green', alpha=0.2)

# Fill the unconditional distribution for comparison
x_uncond = np.linspace(mu_1 - 4*np.sqrt(sigma_11), mu_1 + 4*np.sqrt(sigma_11), 1000)
y_uncond = stats.norm.pdf(x_uncond, mu_1, np.sqrt(sigma_11))
plt.plot(x_uncond, y_uncond, 'k--', alpha=0.5, linewidth=1.5, 
         label=f'Unconditional X₁ ~ N({mu_1:.2f}, {sigma_11:.2f})')

plt.legend()
plt.grid(alpha=0.3)
plt.title('Conditional Distribution of X₁ given X₂=18 and X₃=32')
plt.xlabel('X₁')
plt.ylabel('Probability Density')

# Add explanatory text
plt.text(0.02, 0.02, 
         "The conditional distribution is narrower than the unconditional distribution.\n"
         "This shows how knowing X₂ and X₃ reduces uncertainty about X₁.\n"
         "The conditional mean is shifted from the unconditional mean based on the values of X₂ and X₃.",
         transform=plt.gca().transAxes, fontsize=10, va='bottom', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'conditional_distribution.png'), dpi=100)
plt.close()

print(f"\nGenerated visualization for conditional distribution example")

# Generate a visualization showing how the conditional distribution
# of X₁ changes for different observed values of X₂ and X₃
print("\nGenerating visualization of how the conditional distribution changes...")

# Create a grid of different X₂, X₃ values
x2_values = np.array([15, 18, 20, 25])
x3_values = np.array([25, 28, 32, 35])

# Calculate conditional means and variances for each combination
grid_size = len(x2_values) * len(x3_values)
cond_means = np.zeros(grid_size)
cond_vars = np.zeros(grid_size)
x2_labels = []
x3_labels = []

idx = 0
for x2 in x2_values:
    for x3 in x3_values:
        x_obs = np.array([x2, x3])
        x_deviation = x_obs - mu_23
        cond_means[idx] = mu_1 + adjustment_term @ x_deviation
        cond_vars[idx] = sigma_11 - adjustment_term @ sigma_21
        x2_labels.append(x2)
        x3_labels.append(x3)
        idx += 1

# Visualize how different values of X₂ and X₃ affect the conditional distribution
plt.figure(figsize=(14, 8))

colors = plt.cm.viridis(np.linspace(0, 1, grid_size))

# Plot each conditional distribution
for i in range(grid_size):
    x = np.linspace(cond_means[i] - 3*np.sqrt(cond_vars[i]), 
                   cond_means[i] + 3*np.sqrt(cond_vars[i]), 100)
    y = stats.norm.pdf(x, cond_means[i], np.sqrt(cond_vars[i]))
    plt.plot(x, y, '-', color=colors[i], linewidth=1.5, alpha=0.7,
             label=f'X₂={x2_labels[i]}, X₃={x3_labels[i]}')
    plt.axvline(cond_means[i], color=colors[i], linestyle='--', alpha=0.5)

# Plot the unconditioned distribution for reference
x_uncond = np.linspace(mu_1 - 3*np.sqrt(sigma_11), mu_1 + 3*np.sqrt(sigma_11), 100)
y_uncond = stats.norm.pdf(x_uncond, mu_1, np.sqrt(sigma_11))
plt.plot(x_uncond, y_uncond, 'k--', linewidth=2, alpha=0.5, 
         label=f'Unconditional X₁ ~ N({mu_1}, {sigma_11})')
plt.axvline(mu_1, color='k', linestyle='--', alpha=0.5)

plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.grid(alpha=0.3)
plt.title('Conditional Distribution of X₁ for Different Values of X₂ and X₃')
plt.xlabel('X₁')
plt.ylabel('Probability Density')

# Add explanatory text
plt.text(0.02, 0.02, 
         "Each curve represents the distribution of X₁ given different values of X₂ and X₃.\n"
         "Note how the mean shifts based on the observed values, while the variance remains constant.\n"
         "This is a key property of multivariate normal distributions.",
         transform=plt.gca().transAxes, fontsize=10, va='bottom', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'conditional_distributions_comparison.png'), dpi=100)
plt.close()

print(f"Generated comparison visualization for conditional distributions")

print("\nKey insights from conditional distributions example:")
print("1. The conditional distribution of a multivariate normal is also normal.")
print("2. The conditional mean μ₁|₂₃ = μ₁ + Σ₁₂Σ₂₂⁻¹(x₂₃ - μ₂₃) shows how observed values influence our prediction.")
print("3. The conditional variance σ₁|₂₃ = σ₁₁ - Σ₁₂Σ₂₂⁻¹Σ₂₁ depends only on the covariance structure, not on observed values.")
print(f"4. In this example, the variance reduced from {sigma_11} to {cond_variance}, showing the information gain.")
print("5. The conditional mean is a linear function of the observed values, showing the linear relationship in normal distributions.")
print("6. This property makes multivariate normal distributions powerful for linear prediction models.")
print("7. The same principle underlies Gaussian Process regression and Kalman filtering in time series.")

# Add a new visualization: 3D representation of conditioning
print("\nGenerating 3D visualization of conditioning process...")

# Create a figure for the 3D visualization
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Create a meshgrid for X₁ and X₂ (we'll fix X₃ = 32)
x1 = np.linspace(mu[0] - 3*np.sqrt(sigma[0, 0]), mu[0] + 3*np.sqrt(sigma[0, 0]), 30)
x2 = np.linspace(mu[1] - 3*np.sqrt(sigma[1, 1]), mu[1] + 3*np.sqrt(sigma[1, 1]), 30)
X1, X2 = np.meshgrid(x1, x2)

# Fixed value for X₃
x3_fixed = 32

# Calculate the joint PDF for the slice where X₃ = x3_fixed
Z = np.zeros_like(X1)
for i in range(len(x1)):
    for j in range(len(x2)):
        # For each point in the meshgrid, calculate the PDF
        point = np.array([X1[j, i], X2[j, i], x3_fixed])
        Z[j, i] = stats.multivariate_normal.pdf(point, mean=mu, cov=sigma)

# Normalize Z for better visualization
Z = Z / np.max(Z)

# Plot the 3D surface
surf = ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8, 
                       linewidth=0, antialiased=True)

# Add the conditional distribution line at X₂ = 18
x1_line = np.linspace(mu[0] - 3*np.sqrt(sigma[0, 0]), mu[0] + 3*np.sqrt(sigma[0, 0]), 100)
x2_fixed = 18.0
z_line = np.zeros_like(x1_line)

# Calculate conditional PDF values
for i in range(len(x1_line)):
    point = np.array([x1_line[i], x2_fixed, x3_fixed])
    z_line[i] = stats.multivariate_normal.pdf(point, mean=mu, cov=sigma)

# Normalize for better visualization
z_line = z_line / np.max(Z)

# Plot the conditional distribution line
ax.plot(x1_line, np.ones_like(x1_line) * x2_fixed, z_line, 'r-', linewidth=3, 
        label=f'Conditional at X₂={x2_fixed}, X₃={x3_fixed}')

# Add a vertical line at the conditional mean
ax.plot([cond_mean, cond_mean], [x2_fixed, x2_fixed], [0, np.max(z_line)], 
        'r--', linewidth=2, label=f'Conditional Mean={cond_mean:.2f}')

# Set labels and title
ax.set_xlabel('X₁')
ax.set_ylabel('X₂')
ax.set_zlabel('Normalized Probability Density')
ax.set_title('3D Visualization of Conditioning: Joint Distribution with X₃ Fixed at 32')

# Add legend
ax.legend(loc='upper right')

# Add colorbar
cbar = fig.colorbar(surf, ax=ax, shrink=0.7, aspect=10)
cbar.set_label('Normalized PDF')

# Add text annotation
ax.text2D(0.02, 0.02, 
         "This 3D visualization shows how conditioning works geometrically.\n"
         "The surface represents the joint distribution of X₁ and X₂ when X₃=32.\n"
         "The red line shows the conditional distribution of X₁ when X₂=18 and X₃=32.\n"
         "The dashed vertical line marks the conditional mean of X₁.",
         transform=ax.transAxes, fontsize=10, va='bottom',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'conditional_distribution_3d.png'), dpi=100)
plt.close()

print(f"Generated 3D visualization of conditioning process")

# Display plots if running in interactive mode
plt.show() 