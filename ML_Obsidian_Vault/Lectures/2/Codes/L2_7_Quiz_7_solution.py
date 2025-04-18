import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_7_Quiz_7")
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
print("1. Ridge Regression: L(w) = ||y - Xw||^2 + λ||w||^2")
print("2. Lasso Regression: L(w) = ||y - Xw||^2 + λ||w||_1")
print("\nTasks:")
print("1. Show that Ridge Regression can be interpreted as MAP estimation with a specific prior on w")
print("2. Show that Lasso Regression can be interpreted as MAP estimation with a different prior on w")
print("3. For λ = 10, sketch the shape of both priors in 2D (for a 2-dimensional weight vector)")

# Step 2: Ridge Regression as MAP Estimation
print_step_header(2, "Ridge Regression as MAP Estimation")

print("To connect Ridge Regression to MAP estimation, we need to establish the relationship between")
print("the regularized loss function and the posterior distribution in Bayesian inference.")
print("\nIn MAP estimation, we seek to maximize the posterior probability p(w|D), which by Bayes' rule is:")
print("p(w|D) ∝ p(D|w) × p(w)")
print("\nTaking the logarithm, we get:")
print("log(p(w|D)) = log(p(D|w)) + log(p(w)) + const")
print("\nTo maximize log(p(w|D)), we can equivalently minimize -log(p(w|D)):")
print("-log(p(w|D)) = -log(p(D|w)) - log(p(w)) + const")
print("\nIn linear regression with Gaussian noise, the likelihood term is:")
print("p(D|w) = N(y|Xw, σ²I)")
print("\nTaking the negative log-likelihood:")
print("-log(p(D|w)) ∝ ||y - Xw||²/(2σ²)")
print("\nFor Ridge Regression to be interpreted as MAP estimation, we need:")
print("-log(p(w|D)) ∝ ||y - Xw||² + λ||w||²")
print("\nThis means the prior must satisfy:")
print("-log(p(w)) ∝ λ||w||²/(2σ²)")
print("\nTherefore:")
print("p(w) ∝ exp(-λ||w||²/(2σ²))")
print("\nThis is a multivariate Gaussian prior on w:")
print("p(w) = N(w|0, (σ²/λ)I)")
print("\nIn other words, Ridge Regression corresponds to MAP estimation with a zero-mean isotropic")
print("Gaussian prior with covariance matrix (σ²/λ)I. The regularization parameter λ controls the")
print("strength of the prior: larger λ means smaller variance, reflecting stronger prior belief")
print("that weights should be close to zero.")

# Step 3: Lasso Regression as MAP Estimation
print_step_header(3, "Lasso Regression as MAP Estimation")

print("Similar to Ridge Regression, we can interpret Lasso Regression as MAP estimation.")
print("\nLasso Regression minimizes:")
print("L(w) = ||y - Xw||² + λ||w||₁")
print("\nFollowing the same approach as before, we need:")
print("-log(p(w|D)) ∝ ||y - Xw||² + λ||w||₁")
print("\nAssuming the same Gaussian likelihood, we have:")
print("-log(p(D|w)) ∝ ||y - Xw||²/(2σ²)")
print("\nThis means the prior must satisfy:")
print("-log(p(w)) ∝ λ||w||₁/(2σ²)")
print("\nTherefore:")
print("p(w) ∝ exp(-λ||w||₁/(2σ²))")
print("\nThis corresponds to a Laplace (or double exponential) prior on each component of w:")
print("p(w) = ∏ᵢ (λ/(2σ²)) exp(-λ|wᵢ|/(2σ²))")
print("\nIn other words, Lasso Regression corresponds to MAP estimation with a Laplace prior.")
print("The Laplace distribution has heavier tails than the Gaussian, which makes it more likely")
print("to produce sparse solutions (some weights exactly zero).")

# Step 4: Visualizing the Priors
print_step_header(4, "Visualizing the Priors in 2D")

print("We'll now visualize both priors for a 2D weight vector with λ = 10.")
print("For simplicity, we'll assume σ² = 1.")

# Define the prior density functions
def gaussian_prior(w, lambda_val=10, sigma_sq=1):
    """Gaussian prior density for Ridge Regression."""
    var = sigma_sq / lambda_val
    return np.exp(-np.sum(w**2) / (2 * var)) / np.sqrt(2 * np.pi * var)

def laplace_prior(w, lambda_val=10, sigma_sq=1):
    """Laplace prior density for Lasso Regression."""
    scale = 2 * sigma_sq / lambda_val
    return np.exp(-np.sum(np.abs(w)) / scale) / (2 * scale)

# Create a grid of w values
w1 = np.linspace(-1, 1, 100)
w2 = np.linspace(-1, 1, 100)
W1, W2 = np.meshgrid(w1, w2)

# Calculate prior densities at each point
Z_gaussian = np.zeros_like(W1)
Z_laplace = np.zeros_like(W1)

for i in range(len(w1)):
    for j in range(len(w2)):
        w = np.array([W1[i, j], W2[i, j]])
        Z_gaussian[i, j] = gaussian_prior(w)
        Z_laplace[i, j] = laplace_prior(w)

# Plot the Gaussian prior (Ridge)
plt.figure(figsize=(12, 10))
contour_gaussian = plt.contourf(W1, W2, Z_gaussian, 20, cmap='viridis')
plt.colorbar(contour_gaussian, label='Prior density p(w)')
plt.title('Gaussian Prior for Ridge Regression (λ = 10)', fontsize=16)
plt.xlabel('w₁', fontsize=14)
plt.ylabel('w₂', fontsize=14)
plt.grid(True)
plt.axis('equal')
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "gaussian_prior_2d.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {file_path}")

# Plot the Laplace prior (Lasso)
plt.figure(figsize=(12, 10))
contour_laplace = plt.contourf(W1, W2, Z_laplace, 20, cmap='plasma')
plt.colorbar(contour_laplace, label='Prior density p(w)')
plt.title('Laplace Prior for Lasso Regression (λ = 10)', fontsize=16)
plt.xlabel('w₁', fontsize=14)
plt.ylabel('w₂', fontsize=14)
plt.grid(True)
plt.axis('equal')
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "laplace_prior_2d.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Create 3D visualizations for both priors
# Gaussian Prior
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(W1, W2, Z_gaussian, cmap=cm.viridis, linewidth=0, antialiased=True)
ax.set_xlabel('w₁', fontsize=14)
ax.set_ylabel('w₂', fontsize=14)
ax.set_zlabel('Prior density p(w)', fontsize=14)
ax.set_title('3D Visualization of Gaussian Prior (λ = 10)', fontsize=16)

# Save the 3D figure
file_path = os.path.join(save_dir, "gaussian_prior_3d.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Laplace Prior
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(W1, W2, Z_laplace, cmap=cm.plasma, linewidth=0, antialiased=True)
ax.set_xlabel('w₁', fontsize=14)
ax.set_ylabel('w₂', fontsize=14)
ax.set_zlabel('Prior density p(w)', fontsize=14)
ax.set_title('3D Visualization of Laplace Prior (λ = 10)', fontsize=16)

# Save the 3D figure
file_path = os.path.join(save_dir, "laplace_prior_3d.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 5: Comparison of Priors and Regression Methods
print_step_header(5, "Comparing Ridge and Lasso Priors")

print("Let's compare the key properties of these priors and their implications for regression:")
print("\n1. Shape Comparison:")
print("   - Gaussian prior (Ridge): Circular/spherical contours, smooth at the origin")
print("   - Laplace prior (Lasso): Diamond-shaped contours, sharp corners at the axes")
print("\n2. Effect on Sparsity:")
print("   - Gaussian prior: Tends to shrink all weights proportionally toward zero")
print("   - Laplace prior: Can shrink some weights exactly to zero (sparse solutions)")
print("\n3. Practical Implications:")
print("   - Ridge regression is better when many features are expected to contribute")
print("   - Lasso regression performs well for feature selection (identifying important features)")
print("\n4. Mathematical Interpretation:")
print("   - Ridge corresponds to L2 regularization with a differentiable objective")
print("   - Lasso corresponds to L1 regularization with a non-differentiable objective at w=0")

# Create a visualization comparing both priors on the same plot (contour lines only)
plt.figure(figsize=(12, 10))

# Plot contour lines for both priors
contour_gaussian = plt.contour(W1, W2, Z_gaussian, 5, colors='blue', linewidths=2)
contour_laplace = plt.contour(W1, W2, Z_laplace, 5, colors='red', linewidths=2)

# Add labels and formatting
plt.clabel(contour_gaussian, inline=True, fontsize=10, fmt='%.3f')
plt.clabel(contour_laplace, inline=True, fontsize=10, fmt='%.3f')
plt.title('Comparison of Gaussian (Ridge) and Laplace (Lasso) Priors (λ = 10)', fontsize=16)
plt.xlabel('w₁', fontsize=14)
plt.ylabel('w₂', fontsize=14)
plt.grid(True)
plt.axis('equal')

# Add a legend
plt.plot([], [], 'b-', linewidth=2, label='Gaussian Prior (Ridge)')
plt.plot([], [], 'r-', linewidth=2, label='Laplace Prior (Lasso)')
plt.legend(fontsize=12)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "prior_comparison.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {file_path}")

# Step 6: Effect of Lambda on the Shape of Priors
print_step_header(6, "Effect of Lambda on Prior Shapes")

print("Finally, let's examine how the regularization parameter λ affects the shape of the priors.")

# Define lambda values to visualize
lambda_values = [1, 5, 10, 20]
w = np.linspace(-1, 1, 1000)

# Plot Gaussian priors for different lambda values
plt.figure(figsize=(12, 8))

for lambda_val in lambda_values:
    var = 1 / lambda_val
    density = np.exp(-w**2 / (2 * var)) / np.sqrt(2 * np.pi * var)
    plt.plot(w, density, label=f'λ = {lambda_val}', linewidth=2)

plt.title('Effect of λ on Gaussian Prior (1D)', fontsize=16)
plt.xlabel('w', fontsize=14)
plt.ylabel('Prior density p(w)', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "gaussian_prior_lambda.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {file_path}")

# Plot Laplace priors for different lambda values
plt.figure(figsize=(12, 8))

for lambda_val in lambda_values:
    scale = 2 / lambda_val
    density = np.exp(-np.abs(w) / scale) / (2 * scale)
    plt.plot(w, density, label=f'λ = {lambda_val}', linewidth=2)

plt.title('Effect of λ on Laplace Prior (1D)', fontsize=16)
plt.xlabel('w', fontsize=14)
plt.ylabel('Prior density p(w)', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "laplace_prior_lambda.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Summarize the findings
print_step_header(7, "Summary of Findings")

print("1. Ridge Regression corresponds to MAP estimation with a Gaussian prior:")
print("   p(w) ∝ exp(-λ||w||²/(2σ²))")
print("\n2. Lasso Regression corresponds to MAP estimation with a Laplace prior:")
print("   p(w) ∝ exp(-λ||w||₁/(2σ²))")
print("\n3. The shape of the priors explains the different behavior of the methods:")
print("   - Ridge: Smooth, circular contours lead to proportional shrinkage")
print("   - Lasso: Diamond shape with sharp corners leads to sparse solutions")
print("\n4. The parameter λ controls the concentration of both priors around 0:")
print("   - Larger λ = stronger prior = more regularization")
print("   - Smaller λ = weaker prior = less regularization")
print("\n5. This Bayesian interpretation provides a principled way to understand")
print("   regularization methods in terms of prior beliefs about the parameters.") 