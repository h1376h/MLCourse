import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
import math

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_3_1")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Set up the problem
print_step_header(1, "Problem Setup")

# Given data
data = np.array([4.2, 3.8, 5.1, 4.5, 3.2, 4.9, 5.3, 4.0, 4.7, 3.6])
n = len(data)
sigma = 2  # Known standard deviation
print("Given:")
print(f"- Random sample X₁, X₂, ..., X_{n} from N(μ, σ²)")
print(f"- Known standard deviation σ = {sigma}")
print(f"- Unknown mean μ to be estimated")
print(f"- Observed values: {data}")

# Step 2: Likelihood Function
print_step_header(2, "Likelihood Function")

def likelihood(mu, data, sigma):
    """Compute the likelihood function for normal distribution."""
    n = len(data)
    exponent = -np.sum((data - mu)**2) / (2 * sigma**2)
    coefficient = (1 / (sigma * np.sqrt(2 * np.pi)))**n
    return coefficient * np.exp(exponent)

# Create an array of mu values to plot
mu_values = np.linspace(3.0, 5.5, 1000)
likelihood_values = [likelihood(mu, data, sigma) for mu in mu_values]

# Plot the likelihood function
plt.figure(figsize=(10, 6))
plt.plot(mu_values, likelihood_values, 'b-', linewidth=2)
plt.axvline(x=np.mean(data), color='r', linestyle='--', 
            label=f'Sample Mean = {np.mean(data):.4f}')
plt.title('Likelihood Function L(μ)', fontsize=14)
plt.xlabel('μ', fontsize=12)
plt.ylabel('L(μ)', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "likelihood_function.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Display the mathematical expression for the likelihood function
print("The likelihood function for normal distribution is:")
print("L(μ) = ∏_{i=1}^{n} f(xᵢ|μ) = ∏_{i=1}^{n} (1/(σ√(2π))) * exp(-(xᵢ-μ)²/(2σ²))")
print("L(μ) = (1/(σ√(2π)))^n * exp(-(∑(xᵢ-μ)²)/(2σ²))")
print(f"With our data and σ = {sigma}:")
print(f"L(μ) = (1/({sigma}√(2π)))^{n} * exp(-(∑(xᵢ-μ)²)/(2*{sigma}²))")

# Step 3: Log-Likelihood Function
print_step_header(3, "Log-Likelihood Function")

def log_likelihood(mu, data, sigma):
    """Compute the log-likelihood function for normal distribution."""
    n = len(data)
    log_coefficient = n * (-np.log(sigma) - 0.5 * np.log(2 * np.pi))
    log_exponent = -np.sum((data - mu)**2) / (2 * sigma**2)
    return log_coefficient + log_exponent

# Calculate log-likelihood values
log_likelihood_values = [log_likelihood(mu, data, sigma) for mu in mu_values]

# Plot the log-likelihood function
plt.figure(figsize=(10, 6))
plt.plot(mu_values, log_likelihood_values, 'g-', linewidth=2)
plt.axvline(x=np.mean(data), color='r', linestyle='--', 
            label=f'Sample Mean = {np.mean(data):.4f}')
plt.title('Log-Likelihood Function ℓ(μ)', fontsize=14)
plt.xlabel('μ', fontsize=12)
plt.ylabel('ℓ(μ)', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "log_likelihood_function.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Display the mathematical expression for the log-likelihood function
print("The log-likelihood function is:")
print("ℓ(μ) = log(L(μ)) = n*log(1/(σ√(2π))) - (∑(xᵢ-μ)²)/(2σ²)")
print("ℓ(μ) = -n*log(σ) - n/2*log(2π) - (∑(xᵢ-μ)²)/(2σ²)")
print(f"With our data and σ = {sigma}:")
print(f"ℓ(μ) = -{n}*log({sigma}) - {n}/2*log(2π) - (∑(xᵢ-μ)²)/(2*{sigma}²)")

# Step 4: Maximum Likelihood Estimation
print_step_header(4, "Maximum Likelihood Estimation")

# Calculate MLE analytically
mle_mu = np.mean(data)
print(f"The maximum likelihood estimate (MLE) for μ is:")
print(f"μ̂ = (1/n) * ∑(xᵢ) = {mle_mu:.4f}")

# Verify using numerical optimization
from scipy.optimize import minimize

def neg_log_likelihood(mu, data, sigma):
    """Negative log-likelihood function (for minimization)."""
    return -log_likelihood(mu, data, sigma)

# Find the MLE using numerical optimization
result = minimize(neg_log_likelihood, x0=4.0, args=(data, sigma))
print(f"MLE using numerical optimization: {result.x[0]:.4f}")
print(f"This confirms our analytical result: {mle_mu:.4f}")

# Plot the likelihood function with MLE highlighted
plt.figure(figsize=(10, 6))

# Create a subplot grid
gs = GridSpec(2, 1, height_ratios=[2, 1])

# Plot likelihood function
ax1 = plt.subplot(gs[0])
ax1.plot(mu_values, likelihood_values, 'b-', linewidth=2)
ax1.axvline(x=mle_mu, color='r', linestyle='--', 
           label=f'MLE μ̂ = {mle_mu:.4f}')
ax1.set_title('Likelihood Function with MLE', fontsize=14)
ax1.set_ylabel('L(μ)', fontsize=12)
ax1.grid(True)
ax1.legend()

# Plot log-likelihood function
ax2 = plt.subplot(gs[1])
ax2.plot(mu_values, log_likelihood_values, 'g-', linewidth=2)
ax2.axvline(x=mle_mu, color='r', linestyle='--', 
           label=f'MLE μ̂ = {mle_mu:.4f}')
ax2.set_xlabel('μ', fontsize=12)
ax2.set_ylabel('ℓ(μ)', fontsize=12)
ax2.grid(True)
ax2.legend()

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "mle_estimation.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Illustrate how MLE maximizes the likelihood
print("\nTo find the MLE analytically:")
print("1. Take the derivative of the log-likelihood function with respect to μ:")
print("   dℓ(μ)/dμ = (1/σ²) * ∑(xᵢ-μ)")
print("2. Set this equal to zero and solve for μ:")
print("   (1/σ²) * ∑(xᵢ-μ) = 0")
print("   ∑(xᵢ-μ) = 0")
print("   ∑xᵢ - n*μ = 0")
print("   μ = (1/n) * ∑xᵢ")

# Step 5: Likelihood Ratio Test
print_step_header(5, "Likelihood Ratio Test")

# Hypotheses
null_hypothesis = 5.0  # H0: μ = 5
print(f"Null hypothesis H₀: μ = {null_hypothesis}")
print(f"Alternative hypothesis H₁: μ ≠ {null_hypothesis}")

# Calculate likelihood at the MLE and at the null hypothesis
likelihood_mle = likelihood(mle_mu, data, sigma)
likelihood_null = likelihood(null_hypothesis, data, sigma)

# Calculate the likelihood ratio
lr = likelihood_null / likelihood_mle
print(f"Likelihood under H₀ (μ = {null_hypothesis}): {likelihood_null:.10e}")
print(f"Likelihood under MLE (μ = {mle_mu:.4f}): {likelihood_mle:.10e}")
print(f"Likelihood ratio λ = L(μ=5)/L(μ=μ̂): {lr:.10f}")

# Plot the likelihood ratio
plt.figure(figsize=(10, 6))

plt.plot(mu_values, likelihood_values / max(likelihood_values), 'b-', linewidth=2, label='Normalized Likelihood')
plt.axvline(x=mle_mu, color='r', linestyle='--', 
           label=f'MLE μ̂ = {mle_mu:.4f}')
plt.axvline(x=null_hypothesis, color='g', linestyle='--', 
           label=f'H₀: μ = {null_hypothesis}')
plt.axhline(y=lr, color='orange', linestyle='--', 
           label=f'Likelihood Ratio λ = {lr:.4f}')

# Add a point for the likelihood ratio
plt.scatter([null_hypothesis], [lr], color='orange', s=100, zorder=5)

plt.title('Likelihood Ratio Test', fontsize=14)
plt.xlabel('μ', fontsize=12)
plt.ylabel('L(μ)/L(μ̂)', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "likelihood_ratio_test.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Calculate the log-likelihood ratio (for statistical testing)
log_lr = log_likelihood(null_hypothesis, data, sigma) - log_likelihood(mle_mu, data, sigma)
chi_squared_stat = -2 * log_lr
print(f"Log-likelihood ratio: {log_lr:.4f}")
print(f"Chi-squared statistic (-2 log λ): {chi_squared_stat:.4f}")

# Step 6: Visual Summary
print_step_header(6, "Summary of Results")

plt.figure(figsize=(12, 8))
gs = GridSpec(2, 2)

# Plot 1: Data distribution
ax1 = plt.subplot(gs[0, 0])
ax1.hist(data, bins=5, alpha=0.6, color='skyblue', edgecolor='black')
ax1.axvline(x=mle_mu, color='r', linestyle='--', label=f'Sample Mean = {mle_mu:.4f}')
ax1.axvline(x=null_hypothesis, color='g', linestyle='--', label=f'H₀: μ = {null_hypothesis}')
ax1.set_title('Data Distribution', fontsize=12)
ax1.set_xlabel('Value', fontsize=10)
ax1.set_ylabel('Frequency', fontsize=10)
ax1.legend(fontsize=8)
ax1.grid(True)

# Plot 2: Normal PDF with MLE
x_range = np.linspace(min(data) - 2, max(data) + 2, 1000)
ax2 = plt.subplot(gs[0, 1])
# PDF with MLE
pdf_mle = norm.pdf(x_range, loc=mle_mu, scale=sigma)
ax2.plot(x_range, pdf_mle, 'r-', linewidth=2, label=f'N({mle_mu:.2f}, {sigma}²)')
# PDF with null hypothesis
pdf_null = norm.pdf(x_range, loc=null_hypothesis, scale=sigma)
ax2.plot(x_range, pdf_null, 'g--', linewidth=2, label=f'N({null_hypothesis}, {sigma}²)')
# Add data points
for x in data:
    ax2.axvline(x=x, ymax=0.1, color='blue', alpha=0.3)
ax2.set_title('Normal PDFs with Data Points', fontsize=12)
ax2.set_xlabel('Value', fontsize=10)
ax2.set_ylabel('Density', fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(True)

# Plot 3: Likelihood function
ax3 = plt.subplot(gs[1, 0])
ax3.plot(mu_values, likelihood_values, 'b-', linewidth=2)
ax3.axvline(x=mle_mu, color='r', linestyle='--', label=f'MLE μ̂ = {mle_mu:.4f}')
ax3.axvline(x=null_hypothesis, color='g', linestyle='--', label=f'H₀: μ = {null_hypothesis}')
ax3.set_title('Likelihood Function', fontsize=12)
ax3.set_xlabel('μ', fontsize=10)
ax3.set_ylabel('L(μ)', fontsize=10)
ax3.legend(fontsize=8)
ax3.grid(True)

# Plot 4: Likelihood ratio
ax4 = plt.subplot(gs[1, 1])
ax4.plot(mu_values, likelihood_values / max(likelihood_values), 'b-', linewidth=2)
ax4.axvline(x=mle_mu, color='r', linestyle='--', label=f'MLE μ̂ = {mle_mu:.4f}')
ax4.axvline(x=null_hypothesis, color='g', linestyle='--', label=f'H₀: μ = {null_hypothesis}')
ax4.axhline(y=lr, color='orange', linestyle='--', label=f'Likelihood Ratio λ = {lr:.4f}')
ax4.scatter([null_hypothesis], [lr], color='orange', s=80, zorder=5)
ax4.set_title('Normalized Likelihood', fontsize=12)
ax4.set_xlabel('μ', fontsize=10)
ax4.set_ylabel('L(μ)/L(μ̂)', fontsize=10)
ax4.legend(fontsize=8)
ax4.grid(True)

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "summary_results.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Final summary
print("\nSummary of results:")
print(f"1. Sample: {data}")
print(f"2. Sample mean: {mle_mu:.4f}")
print(f"3. MLE for μ: μ̂ = {mle_mu:.4f}")
print(f"4. Likelihood ratio for testing H₀: μ = {null_hypothesis} vs H₁: μ ≠ {null_hypothesis}: λ = {lr:.6f}")
print(f"5. Log-likelihood ratio: {log_lr:.4f}")
print(f"6. Chi-squared statistic (-2 log λ): {chi_squared_stat:.4f}") 