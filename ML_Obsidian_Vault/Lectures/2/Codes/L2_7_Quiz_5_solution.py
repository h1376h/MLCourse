import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import beta
from scipy.optimize import minimize_scalar

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_7_Quiz_5")
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
print("True or False: When using a uniform prior (e.g., Beta(1,1) for a probability parameter),")
print("the MAP estimate is identical to the MLE.")
print("\nTask:")
print("1. Determine whether the statement is true or false")
print("2. Explain the reasoning mathematically")
print("3. Provide a simple example that illustrates the answer")

# Step 2: Theoretical Analysis
print_step_header(2, "Theoretical Analysis")

print("To analyze whether the MAP with a uniform prior equals the MLE, we need to understand the relationship")
print("between the posterior distribution and the likelihood function when using a uniform prior.")
print("\nMathematically, for a parameter θ:")
print("  P(θ|D) ∝ P(D|θ) × P(θ)")
print("Where:")
print("  - P(θ|D) is the posterior distribution")
print("  - P(D|θ) is the likelihood function")
print("  - P(θ) is the prior distribution")
print("\nWhen the prior P(θ) is uniform (constant across the parameter space):")
print("  P(θ|D) ∝ P(D|θ) × constant")
print("  P(θ|D) ∝ P(D|θ)")
print("\nThis means the posterior is proportional to the likelihood, and therefore,")
print("the value of θ that maximizes the posterior (MAP) will be the same as")
print("the value that maximizes the likelihood (MLE).")

print("\nHowever, there's an important consideration: a uniform prior might not be uniform in")
print("all parameterizations, and the boundaries of the parameter space might affect the posterior.")
print("In general, the statement is TRUE when the uniform prior is defined over the same")
print("parameter space as the likelihood function and there are no boundary effects.")

# Step 3: Example 1 - Binomial Model with Beta(1,1) Prior
print_step_header(3, "Example 1: Binomial Model with Beta(1,1) Prior")

# Define a binomial example
n_trials = 20
k_successes = 15

# Calculate MLE
theta_mle = k_successes / n_trials
print(f"Consider a binomial model with {k_successes} successes out of {n_trials} trials.")
print(f"The MLE for the probability parameter θ is: {theta_mle:.4f}")

# Calculate MAP with Beta(1,1) prior (uniform over [0,1])
alpha_prior = 1
beta_prior = 1
alpha_posterior = alpha_prior + k_successes
beta_posterior = beta_prior + n_trials - k_successes

# For Beta(a,b) with a,b > 1, the mode is (a-1)/(a+b-2)
if alpha_posterior > 1 and beta_posterior > 1:
    theta_map = (alpha_posterior - 1) / (alpha_posterior + beta_posterior - 2)
else:
    # If either parameter is ≤ 1, the mode could be at 0 or 1
    if alpha_posterior <= 1:
        theta_map = 0
    else:
        theta_map = 1

print("\nWith a Beta(1,1) prior (uniform over [0,1]):")
print(f"The posterior distribution is Beta({alpha_posterior}, {beta_posterior})")
print(f"The MAP estimate is: {theta_map:.4f}")
print(f"Difference between MAP and MLE: {theta_map - theta_mle:.10f}")

# Visualize the likelihood and posterior
theta_range = np.linspace(0, 1, 1000)
likelihood = theta_range**k_successes * (1-theta_range)**(n_trials-k_successes)
likelihood_normalized = likelihood / np.max(likelihood)
posterior = beta.pdf(theta_range, alpha_posterior, beta_posterior)
posterior_normalized = posterior / np.max(posterior)

plt.figure(figsize=(10, 6))
plt.plot(theta_range, likelihood_normalized, 'g--', label=f'Likelihood', linewidth=2)
plt.plot(theta_range, posterior_normalized, 'b-', label=f'Posterior with Beta(1,1) prior', linewidth=2)

# Mark the MLE and MAP
plt.axvline(x=theta_mle, color='g', linestyle='-', label=f'MLE: {theta_mle:.4f}', linewidth=2)
plt.axvline(x=theta_map, color='b', linestyle='-', label=f'MAP: {theta_map:.4f}', linewidth=2)

plt.xlabel('θ (Probability of Success)', fontsize=12)
plt.ylabel('Density (Normalized)', fontsize=12)
plt.title('Comparing MLE and MAP with Uniform Prior - Binomial Example', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "binomial_example.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {file_path}")

# Step 4: Example 2 - Normal Model with Different Uniform Priors
print_step_header(4, "Example 2: Normal Model with Different Uniform Priors")

# Generate some sample data from a normal distribution
np.random.seed(42)
true_mean = 5.0
true_std = 1.0
data = np.random.normal(true_mean, true_std, 20)
sample_mean = np.mean(data)
sample_var = np.var(data)

print(f"Consider normally distributed data with sample mean = {sample_mean:.4f} and sample variance = {sample_var:.4f}")
print(f"The MLE for the mean parameter μ is: {sample_mean:.4f}")

# Case 1: Uniform prior over the whole real line
print("\nCase 1: With a uniform prior over the whole real line (-∞, ∞):")
print(f"The MAP estimate is: {sample_mean:.4f}")
print(f"Difference between MAP and MLE: {sample_mean - sample_mean:.10f}")

# Case 2: Uniform prior over [0, 10]
prior_lower = 0
prior_upper = 10

# For a bounded uniform prior, the MAP depends on whether the MLE is within bounds
if sample_mean < prior_lower:
    map_estimate_bounded = prior_lower
elif sample_mean > prior_upper:
    map_estimate_bounded = prior_upper
else:
    map_estimate_bounded = sample_mean

print(f"\nCase 2: With a uniform prior over [{prior_lower}, {prior_upper}]:")
print(f"The MAP estimate is: {map_estimate_bounded:.4f}")
print(f"Difference between MAP and MLE: {map_estimate_bounded - sample_mean:.10f}")

# Visualize the likelihood and posterior for the normal example
mu_range = np.linspace(-2, 12, 1000)
likelihood = np.exp(-0.5 * (len(data) / true_std**2) * (mu_range - sample_mean)**2)
likelihood_normalized = likelihood / np.max(likelihood)

# Create uniform prior over [0, 10]
prior = np.zeros_like(mu_range)
prior[(mu_range >= prior_lower) & (mu_range <= prior_upper)] = 1.0

# Posterior is proportional to likelihood * prior
posterior = likelihood * prior
if np.max(posterior) > 0:  # Avoid division by zero
    posterior_normalized = posterior / np.max(posterior)
else:
    posterior_normalized = posterior

plt.figure(figsize=(10, 6))
plt.plot(mu_range, likelihood_normalized, 'g--', label=f'Likelihood', linewidth=2)
plt.plot(mu_range, prior, 'r:', label=f'Uniform Prior [{prior_lower}, {prior_upper}]', linewidth=2)
plt.plot(mu_range, posterior_normalized, 'b-', label=f'Posterior', linewidth=2)

# Mark the MLE and MAP
plt.axvline(x=sample_mean, color='g', linestyle='-', label=f'MLE: {sample_mean:.4f}', linewidth=2)
plt.axvline(x=map_estimate_bounded, color='b', linestyle='-', label=f'MAP: {map_estimate_bounded:.4f}', linewidth=2)

plt.xlabel('μ (Mean Parameter)', fontsize=12)
plt.ylabel('Density (Normalized)', fontsize=12)
plt.title('Comparing MLE and MAP with Bounded Uniform Prior - Normal Example', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "normal_example.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {file_path}")

# Step 5: Parameterization Example
print_step_header(5, "Example 3: Effect of Parameterization")

# Consider a normal model with unknown variance σ²
# We'll compare using a uniform prior on σ² vs. a uniform prior on σ

# Simulate data with known mean (0) and variance (4)
np.random.seed(42)
true_variance = 4.0
data_var = np.random.normal(0, np.sqrt(true_variance), 30)
sample_variance = np.var(data_var)

print(f"Consider normally distributed data with mean fixed at 0 and sample variance = {sample_variance:.4f}")
print(f"The MLE for the variance parameter σ² is: {sample_variance:.4f}")

# Theoretical behavior for different parameterizations
print("\nParameterization effects:")
print("1. If we place a uniform prior on σ² (variance):")
print("   The posterior is proportional to the likelihood and the MAP equals the MLE.")
print("\n2. If we place a uniform prior on σ (standard deviation):")
print("   This implies a non-uniform prior on σ², and the MAP will NOT equal the MLE.")

# Visualize the effect of different parameterizations
sigma2_range = np.linspace(0.1, 10, 1000)  # Variance
sigma_range = np.sqrt(sigma2_range)        # Standard deviation

# Likelihood as a function of variance (σ²)
n = len(data_var)
sum_x2 = np.sum(data_var**2)  # For zero-mean data, sum of squares
log_likelihood_sigma2 = -n/2 * np.log(2*np.pi*sigma2_range) - sum_x2/(2*sigma2_range)
likelihood_sigma2 = np.exp(log_likelihood_sigma2 - np.max(log_likelihood_sigma2))  # Normalized

# The implied prior on σ² when using a uniform prior on σ
prior_on_sigma2_from_sigma = 1/(2*np.sqrt(sigma2_range))  # p(σ²) = p(σ)|dσ/dσ²| = 1 * 1/(2√σ²)
prior_on_sigma2_from_sigma = prior_on_sigma2_from_sigma / np.max(prior_on_sigma2_from_sigma)  # Normalized

# Posterior with uniform prior on σ² (directly proportional to likelihood)
posterior_uniform_sigma2 = likelihood_sigma2.copy()

# Posterior with uniform prior on σ (not directly proportional to likelihood)
posterior_uniform_sigma = likelihood_sigma2 * prior_on_sigma2_from_sigma
posterior_uniform_sigma = posterior_uniform_sigma / np.max(posterior_uniform_sigma)  # Normalized

# MLE for σ²
mle_sigma2 = sample_variance

# MAP for σ² with uniform prior on σ²
map_uniform_sigma2 = mle_sigma2

# MAP for σ² with uniform prior on σ (analytical result: MAP = (n-2)/n * MLE for n > 2)
if n > 2:
    map_uniform_sigma = (n-2)/n * mle_sigma2
else:
    # For small samples, find the mode numerically
    def neg_posterior(sigma2):
        if sigma2 <= 0:
            return 1e10  # Large penalty for invalid values
        log_likelihood = -n/2 * np.log(2*np.pi*sigma2) - sum_x2/(2*sigma2)
        log_prior = -0.5 * np.log(sigma2)  # Log of 1/(2√σ²)
        return -(log_likelihood + log_prior)
    
    result = minimize_scalar(neg_posterior, bounds=(1e-10, 10), method='bounded')
    map_uniform_sigma = result.x

print(f"\nMLE for σ²: {mle_sigma2:.4f}")
print(f"MAP with uniform prior on σ²: {map_uniform_sigma2:.4f}")
print(f"MAP with uniform prior on σ: {map_uniform_sigma:.4f}")
print(f"Ratio MAP/MLE with uniform prior on σ: {map_uniform_sigma/mle_sigma2:.4f}")
print(f"Theoretical ratio for n={n}: {(n-2)/n:.4f}")

plt.figure(figsize=(12, 8))

# Plot densities as functions of σ²
plt.subplot(2, 1, 1)
plt.plot(sigma2_range, likelihood_sigma2, 'g--', label='Likelihood', linewidth=2)
plt.plot(sigma2_range, prior_on_sigma2_from_sigma, 'r:', label='Implied Prior from Uniform on σ', linewidth=2)
plt.plot(sigma2_range, posterior_uniform_sigma2, 'b-', label='Posterior with Uniform Prior on σ²', linewidth=2)
plt.plot(sigma2_range, posterior_uniform_sigma, 'm-', label='Posterior with Uniform Prior on σ', linewidth=2)

plt.axvline(x=mle_sigma2, color='g', linestyle='-', label=f'MLE: {mle_sigma2:.4f}', linewidth=2)
plt.axvline(x=map_uniform_sigma2, color='b', linestyle='-', label=f'MAP (Uniform on σ²): {map_uniform_sigma2:.4f}', linewidth=2)
plt.axvline(x=map_uniform_sigma, color='m', linestyle='-', label=f'MAP (Uniform on σ): {map_uniform_sigma:.4f}', linewidth=2)

plt.xlabel('σ² (Variance)', fontsize=12)
plt.ylabel('Density (Normalized)', fontsize=12)
plt.title('Effect of Parameterization on Prior and Posterior', fontsize=14)
plt.legend()
plt.grid(True)

# Plot the same densities as functions of σ for clarity
plt.subplot(2, 1, 2)
# Transform the densities to the σ scale
# When changing variables, need to multiply by the Jacobian |dσ²/dσ| = 2σ
likelihood_sigma = likelihood_sigma2[:-1] * 2 * sigma_range[:-1]
likelihood_sigma = likelihood_sigma / np.max(likelihood_sigma)
posterior_uniform_sigma2_on_sigma = posterior_uniform_sigma2[:-1] * 2 * sigma_range[:-1]
posterior_uniform_sigma2_on_sigma = posterior_uniform_sigma2_on_sigma / np.max(posterior_uniform_sigma2_on_sigma)
posterior_uniform_sigma_on_sigma = posterior_uniform_sigma[:-1] * 2 * sigma_range[:-1]
posterior_uniform_sigma_on_sigma = posterior_uniform_sigma_on_sigma / np.max(posterior_uniform_sigma_on_sigma)

# Uniform prior on σ is constant
prior_uniform_sigma = np.ones_like(sigma_range[:-1])
prior_uniform_sigma = prior_uniform_sigma / np.max(prior_uniform_sigma)

plt.plot(sigma_range[:-1], likelihood_sigma, 'g--', label='Likelihood', linewidth=2)
plt.plot(sigma_range[:-1], prior_uniform_sigma, 'r:', label='Uniform Prior on σ', linewidth=2)
plt.plot(sigma_range[:-1], posterior_uniform_sigma2_on_sigma, 'b-', label='Posterior with Uniform Prior on σ²', linewidth=2)
plt.plot(sigma_range[:-1], posterior_uniform_sigma_on_sigma, 'm-', label='Posterior with Uniform Prior on σ', linewidth=2)

plt.axvline(x=np.sqrt(mle_sigma2), color='g', linestyle='-', label=f'MLE: {np.sqrt(mle_sigma2):.4f}', linewidth=2)
plt.axvline(x=np.sqrt(map_uniform_sigma2), color='b', linestyle='-', label=f'MAP (Uniform on σ²): {np.sqrt(map_uniform_sigma2):.4f}', linewidth=2)
plt.axvline(x=np.sqrt(map_uniform_sigma), color='m', linestyle='-', label=f'MAP (Uniform on σ): {np.sqrt(map_uniform_sigma):.4f}', linewidth=2)

plt.xlabel('σ (Standard Deviation)', fontsize=12)
plt.ylabel('Density (Normalized)', fontsize=12)
plt.title('Densities Viewed in Standard Deviation Space', fontsize=14)
plt.legend()
plt.grid(True)

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "parameterization_example.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {file_path}")

# Step 6: Conclusion
print_step_header(6, "Conclusion")

print("Based on our analysis and examples, we can state that the original statement is:")
print("TRUE - but with important caveats.")
print("\nWhen using a uniform prior over the parameter space:")
print("1. If the parameter space is unbounded and the MLE is within the domain, the MAP equals the MLE.")
print("2. If the parameter space is bounded and the MLE falls within these bounds, the MAP equals the MLE.")
print("3. If the MLE falls outside the bounds of the uniform prior, the MAP will be at the boundary closest to the MLE.")
print("4. The equivalence between MAP and MLE depends on the parameterization. A prior that is uniform in one ")
print("   parameterization may not be uniform in another, leading to different MAP estimates.")
print("\nIn summary, a uniform prior makes the posterior proportional to the likelihood within the bounds")
print("of the prior, but parameterization choices and boundary constraints can still cause MAP and MLE to differ.") 