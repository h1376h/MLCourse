import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import os
import seaborn as sns

print("\n=== EXAMPLE 4: MAXIMUM LIKELIHOOD ESTIMATION ===\n")

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

# Problem statement
print("\nProblem: Estimate the parameters (mean vector and covariance matrix) of a multivariate normal distribution\n"
      "using maximum likelihood estimation (MLE) from a dataset of observations.")

# Step 1: Generate synthetic data from a known distribution
print("\nStep 1: Generate synthetic data from a known multivariate normal distribution")

# True parameters
true_mean = np.array([5, 10, 15])
true_cov = np.array([
    [4.0, 1.0, 0.5],
    [1.0, 9.0, 2.0],
    [0.5, 2.0, 16.0]
])

print("True mean vector μ =", true_mean)
print("\nTrue covariance matrix Σ =")
print(true_cov)

# Generate synthetic data
n_samples = 200
data = np.random.multivariate_normal(true_mean, true_cov, n_samples)

print(f"\nGenerated {n_samples} samples from multivariate normal distribution")
print("First 5 samples:")
for i in range(5):
    print(f"Sample {i+1}: {data[i]}")

# Step 2: Derive the Maximum Likelihood Estimators
print("\nStep 2: Derive the Maximum Likelihood Estimators")

print("The log-likelihood function for a multivariate normal distribution is:")
print("ln L(μ, Σ | X) = -n/2 ln(2π) - n/2 ln|Σ| - 1/2 Σᵢ(xᵢ - μ)ᵀΣ⁻¹(xᵢ - μ)")

print("\nMaximizing this function with respect to μ and Σ gives the MLEs:")
print("μ̂ = 1/n Σᵢ xᵢ")
print("Σ̂ = 1/n Σᵢ(xᵢ - μ̂)(xᵢ - μ̂)ᵀ")

# Step 3: Calculate the MLEs from the data
print("\nStep 3: Calculate the MLEs from the data")

# Calculate sample mean and covariance
mle_mean = np.mean(data, axis=0)
mle_cov = np.cov(data, rowvar=False, bias=True)  # Using 1/n instead of 1/(n-1)

print("\nMLE of mean vector (sample mean):")
print(mle_mean)
print("\nMLE of covariance matrix (using 1/n):")
print(mle_cov)

# For comparison, show the usual unbiased covariance estimator (1/(n-1))
unbiased_cov = np.cov(data, rowvar=False)
print("\nUnbiased estimate of covariance matrix (using 1/(n-1)):")
print(unbiased_cov)

# Step 4: Compare the estimates with the true parameters
print("\nStep 4: Compare the estimates with the true parameters")

# Mean estimate error
mean_error = mle_mean - true_mean
mean_percent_error = 100 * np.abs(mean_error / true_mean)

print("\nError in mean estimate:")
print(f"Absolute error: {mean_error}")
print(f"Percentage error: {mean_percent_error}%")

# Covariance estimate error (Frobenius norm)
cov_error = mle_cov - true_cov
cov_frob_norm = np.linalg.norm(cov_error, 'fro')
true_cov_frob_norm = np.linalg.norm(true_cov, 'fro')
cov_relative_error = cov_frob_norm / true_cov_frob_norm

print("\nError in covariance estimate:")
print(f"Frobenius norm of error: {cov_frob_norm:.4f}")
print(f"Relative Frobenius norm error: {cov_relative_error:.4f} ({cov_relative_error*100:.2f}%)")

# Create visualizations to compare true vs. estimated parameters
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Heatmap for true covariance
ax1 = axes[0]
sns.heatmap(true_cov, annot=True, fmt=".2f", cmap="viridis", ax=ax1)
ax1.set_title("True Covariance Matrix")

# Heatmap for estimated covariance
ax2 = axes[1]
sns.heatmap(mle_cov, annot=True, fmt=".2f", cmap="viridis", ax=ax2)
ax2.set_title("MLE Covariance Matrix")

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'covariance_estimation_comparison.png'), dpi=100)
plt.close(fig)

# Compare true vs. estimated density contours (for first two dimensions)
plt.figure(figsize=(12, 10))

# Create a 2D grid for contour plotting
x = np.linspace(true_mean[0] - 3*np.sqrt(true_cov[0, 0]), true_mean[0] + 3*np.sqrt(true_cov[0, 0]), 100)
y = np.linspace(true_mean[1] - 3*np.sqrt(true_cov[1, 1]), true_mean[1] + 3*np.sqrt(true_cov[1, 1]), 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# True distribution contours (2D marginal)
true_pdf = stats.multivariate_normal(true_mean[:2], true_cov[:2, :2])
Z_true = true_pdf.pdf(pos)

# Estimated distribution contours (2D marginal)
est_pdf = stats.multivariate_normal(mle_mean[:2], mle_cov[:2, :2])
Z_est = est_pdf.pdf(pos)

# Plot the data, true contours, and estimated contours
plt.scatter(data[:, 0], data[:, 1], alpha=0.5, s=10, color='blue', label='Data Points')
plt.contour(X, Y, Z_true, levels=6, colors='r', linestyles='dashed', linewidths=2, alpha=0.7, label='True Distribution')
plt.contour(X, Y, Z_est, levels=6, colors='g', linewidths=2, alpha=0.7, label='Estimated Distribution')

# Add the means
plt.scatter(true_mean[0], true_mean[1], color='red', s=100, marker='*', label='True Mean')
plt.scatter(mle_mean[0], mle_mean[1], color='green', s=100, marker='*', label='Estimated Mean')

plt.xlabel('X₁')
plt.ylabel('X₂')
plt.title('Comparison of True vs. Estimated Bivariate Normal Distribution')
plt.legend()

# Add text explanation
plt.figtext(0.5, 0.01, 
           "The MLE estimates approximate the true parameters very well with sufficient data.\n"
           "With increasing sample size, the MLE estimates converge to the true parameters.",
           ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'mle_multivariate_normal.png'), dpi=100)
plt.close()

# Demonstrate convergence of MLE with increasing sample size
print("\nDemonstrating convergence of MLE with increasing sample size...")

# Different sample sizes to test
sample_sizes = [10, 50, 100, 500, 1000, 5000]
mean_errors = []
cov_errors = []

for n in sample_sizes:
    # Generate data with size n
    samples = np.random.multivariate_normal(true_mean, true_cov, n)
    
    # Calculate MLEs
    sample_mean = np.mean(samples, axis=0)
    sample_cov = np.cov(samples, rowvar=False, bias=True)
    
    # Calculate errors
    mean_err = np.linalg.norm(sample_mean - true_mean)
    cov_err = np.linalg.norm(sample_cov - true_cov, 'fro') / np.linalg.norm(true_cov, 'fro')
    
    mean_errors.append(mean_err)
    cov_errors.append(cov_err)
    
    print(f"n = {n}: Mean error = {mean_err:.4f}, Covariance relative error = {cov_err:.4f}")

# Plot the convergence
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(sample_sizes, mean_errors, 'bo-', linewidth=2)
plt.xscale('log')
plt.xlabel('Sample Size (log scale)')
plt.ylabel('L2 Norm of Mean Error')
plt.title('Convergence of Mean MLE')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(sample_sizes, cov_errors, 'ro-', linewidth=2)
plt.xscale('log')
plt.xlabel('Sample Size (log scale)')
plt.ylabel('Relative Frobenius Norm of Covariance Error')
plt.title('Convergence of Covariance MLE')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'mle_convergence.png'), dpi=100)
plt.close()

print("\nKey insights from Maximum Likelihood Estimation example:")
print("1. The MLE for the mean vector is the sample mean.")
print("2. The MLE for the covariance matrix is the sample covariance matrix (using 1/n).")
print("3. The MLEs are consistent: as sample size increases, they converge to true parameters.")
print("4. The likelihood function for a multivariate normal is fully determined by the mean vector and covariance matrix.")
print("5. MLEs provide a principled way to fit multivariate normal models to data.")
print("6. The sample covariance matrix must be positive definite, which requires n ≥ d (samples ≥ dimensions).")
print("7. In high-dimensional settings, regularization techniques like shrinkage estimation may be needed.")

# Display plots if running in interactive mode
plt.show() 