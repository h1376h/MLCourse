import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
from scipy import integrate

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_7_Quiz_2")
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
print("- Normal distribution with unknown mean μ and known variance σ² = 4")
print("- Data X = {7.2, 6.8, 8.3, 7.5, 6.9}")
print("- Prior for μ is N(7, 1)")
print("\nTask:")
print("1. Derive the posterior distribution for μ")
print("2. Calculate the MAP estimate for μ")
print("3. Derive the posterior predictive distribution for a new observation")
print("4. Calculate the 95% prediction interval for a new observation")

# Step 2: Deriving the Posterior Distribution
print_step_header(2, "Deriving the Posterior Distribution")

# Define the parameters
data = np.array([7.2, 6.8, 8.3, 7.5, 6.9])
n = len(data)
prior_mean = 7.0
prior_var = 1.0
likelihood_var = 4.0  # Known variance σ²
data_mean = np.mean(data)

# Calculate posterior parameters
posterior_var = 1 / (1/prior_var + n/likelihood_var)
posterior_mean = posterior_var * (prior_mean/prior_var + sum(data)/likelihood_var)

print("For a normal likelihood with known variance and a normal prior for the mean:")
print("The posterior is also a normal distribution.")
print("\nPosterior calculation steps:")
print(f"Prior: μ ~ N({prior_mean}, {prior_var})")
print(f"Likelihood (for each observation): x_i | μ ~ N(μ, {likelihood_var})")
print(f"Number of observations: n = {n}")
print(f"Sample mean: x̄ = {data_mean:.4f}")
print("\nPosterior variance calculation:")
print(f"1/σ²_posterior = 1/σ²_prior + n/σ²_likelihood")
print(f"1/σ²_posterior = 1/{prior_var} + {n}/{likelihood_var}")
print(f"1/σ²_posterior = {1/prior_var} + {n/likelihood_var}")
print(f"1/σ²_posterior = {1/prior_var + n/likelihood_var}")
print(f"σ²_posterior = {posterior_var:.6f}")
print("\nPosterior mean calculation:")
print(f"μ_posterior = σ²_posterior * (μ_prior/σ²_prior + Σx_i/σ²_likelihood)")
print(f"μ_posterior = {posterior_var:.6f} * ({prior_mean}/{prior_var} + {sum(data)}/{likelihood_var})")
print(f"μ_posterior = {posterior_var:.6f} * ({prior_mean/prior_var} + {sum(data)/likelihood_var})")
print(f"μ_posterior = {posterior_var:.6f} * {prior_mean/prior_var + sum(data)/likelihood_var}")
print(f"μ_posterior = {posterior_mean:.6f}")
print("\nPosterior distribution: μ | X ~ N({:.4f}, {:.6f})".format(posterior_mean, posterior_var))

# Create a visualization of the prior, likelihood, and posterior
mu_range = np.linspace(5, 9, 1000)

# Prior
prior = norm.pdf(mu_range, prior_mean, np.sqrt(prior_var))

# Likelihood (not normalized)
likelihood = np.prod([norm.pdf(x, mu_range[:, np.newaxis], np.sqrt(likelihood_var)) for x in data], axis=0)
likelihood = likelihood / np.max(likelihood)  # Normalize for visualization

# Posterior
posterior = norm.pdf(mu_range, posterior_mean, np.sqrt(posterior_var))
posterior = posterior / np.max(posterior)  # Normalize for visualization

plt.figure(figsize=(10, 6))
plt.plot(mu_range, prior, 'r--', label=f'Prior: N({prior_mean}, {prior_var})', linewidth=2)
plt.plot(mu_range, likelihood, 'g-.', label=f'Likelihood (normalized)', linewidth=2)
plt.plot(mu_range, posterior, 'b-', label=f'Posterior: N({posterior_mean:.4f}, {posterior_var:.6f})', linewidth=2)

plt.xlabel('μ (Mean)', fontsize=12)
plt.ylabel('Density (Normalized)', fontsize=12)
plt.title('Bayesian Inference: Prior, Likelihood, and Posterior for Normal Mean', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "prior_likelihood_posterior.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 3: Calculate the MAP Estimate
print_step_header(3, "Calculating the MAP Estimate")

# For a normal distribution, the MAP estimate is the mean of the posterior
map_estimate = posterior_mean

print("For a normal posterior distribution, the mode (MAP estimate) equals the mean:")
print(f"MAP = μ_posterior = {map_estimate:.6f}")

# Step 4: Derive the Posterior Predictive Distribution
print_step_header(4, "Deriving the Posterior Predictive Distribution")

# For normal likelihood and normal posterior, the posterior predictive is also normal
predictive_mean = posterior_mean
predictive_var = posterior_var + likelihood_var  # Predictive variance = posterior variance + likelihood variance

print("The posterior predictive distribution for a new observation x_new is:")
print("p(x_new | X) = ∫ p(x_new | μ) p(μ | X) dμ")
print("For normal likelihood and normal posterior, this integral has a closed form solution:")
print(f"x_new | X ~ N(μ_posterior, σ²_posterior + σ²_likelihood)")
print(f"x_new | X ~ N({posterior_mean:.6f}, {posterior_var:.6f} + {likelihood_var})")
print(f"x_new | X ~ N({predictive_mean:.6f}, {predictive_var:.6f})")

# Visualize the posterior predictive distribution
x_new_range = np.linspace(0, 15, 1000)
predictive_pdf = norm.pdf(x_new_range, predictive_mean, np.sqrt(predictive_var))

plt.figure(figsize=(10, 6))
plt.plot(x_new_range, predictive_pdf, 'b-', linewidth=2)
plt.axvline(x=predictive_mean, color='r', linestyle='-', 
            label=f'Mean: {predictive_mean:.4f}', linewidth=2)

# Mark the data points on the x-axis
for x in data:
    plt.axvline(x=x, color='g', linestyle='--', alpha=0.5)

plt.annotate('Observed Data Points', xy=(data.mean(), 0.02), 
             xytext=(data.mean(), 0.05), 
             arrowprops=dict(facecolor='green', shrink=0.05),
             ha='center')

plt.xlabel('x_new', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Posterior Predictive Distribution for a New Observation', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "posterior_predictive.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 5: Calculate the 95% Prediction Interval
print_step_header(5, "Calculating the 95% Prediction Interval")

# For a normal distribution, the 95% prediction interval is mean ± 1.96*std
alpha = 0.05  # For 95% interval
z_score = norm.ppf(1 - alpha/2)  # = 1.96 for 95%
lower_bound = predictive_mean - z_score * np.sqrt(predictive_var)
upper_bound = predictive_mean + z_score * np.sqrt(predictive_var)

print(f"For a 95% prediction interval in a normal distribution:")
print(f"Lower bound = μ - z_(α/2) * σ = {predictive_mean:.6f} - {z_score:.4f} * √{predictive_var:.6f}")
print(f"Lower bound = {predictive_mean:.6f} - {z_score:.4f} * {np.sqrt(predictive_var):.6f}")
print(f"Lower bound = {predictive_mean:.6f} - {z_score * np.sqrt(predictive_var):.6f}")
print(f"Lower bound = {lower_bound:.6f}")
print()
print(f"Upper bound = μ + z_(α/2) * σ = {predictive_mean:.6f} + {z_score:.4f} * √{predictive_var:.6f}")
print(f"Upper bound = {predictive_mean:.6f} + {z_score:.4f} * {np.sqrt(predictive_var):.6f}")
print(f"Upper bound = {predictive_mean:.6f} + {z_score * np.sqrt(predictive_var):.6f}")
print(f"Upper bound = {upper_bound:.6f}")
print()
print(f"95% Prediction Interval: [{lower_bound:.6f}, {upper_bound:.6f}]")

# Visualize the prediction interval on the posterior predictive distribution
plt.figure(figsize=(10, 6))
plt.plot(x_new_range, predictive_pdf, 'b-', linewidth=2, label='Posterior Predictive Distribution')

# Shade the 95% prediction interval
interval_x = np.linspace(lower_bound, upper_bound, 1000)
interval_y = norm.pdf(interval_x, predictive_mean, np.sqrt(predictive_var))
plt.fill_between(interval_x, interval_y, color='blue', alpha=0.3, 
                 label=f'95% Prediction Interval: [{lower_bound:.4f}, {upper_bound:.4f}]')

# Mark the bounds of the interval
plt.axvline(x=lower_bound, color='r', linestyle='--', linewidth=2)
plt.axvline(x=upper_bound, color='r', linestyle='--', linewidth=2)

# Mark the mean of the predictive distribution
plt.axvline(x=predictive_mean, color='k', linestyle='-', 
            label=f'Mean: {predictive_mean:.4f}', linewidth=2)

plt.xlabel('x_new', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Posterior Predictive Distribution with 95% Prediction Interval', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "prediction_interval.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 6: Summary
print_step_header(6, "Summary of Results")

print("1. Posterior Distribution for μ:")
print(f"   μ | X ~ N({posterior_mean:.6f}, {posterior_var:.6f})")
print()
print(f"2. MAP Estimate for μ: {map_estimate:.6f}")
print()
print("3. Posterior Predictive Distribution for x_new:")
print(f"   x_new | X ~ N({predictive_mean:.6f}, {predictive_var:.6f})")
print()
print(f"4. 95% Prediction Interval for x_new: [{lower_bound:.6f}, {upper_bound:.6f}]")

# Compare the influence of prior and data on the posterior
print_step_header(7, "Influence of Prior vs. Data")

# Define a range of sample sizes
sample_sizes = [1, 5, 20, 100]

# Define a range of values for μ
mu_range = np.linspace(5, 9, 1000)

plt.figure(figsize=(12, 8))

# Plot the prior
plt.plot(mu_range, norm.pdf(mu_range, prior_mean, np.sqrt(prior_var)), 
         'k--', label=f'Prior: N({prior_mean}, {prior_var})', linewidth=2)

# Generate some synthetic data from a normal with true mean = 7.5
np.random.seed(42)
true_mean = 7.5
synthetic_data = np.random.normal(true_mean, np.sqrt(likelihood_var), 100)

colors = ['red', 'green', 'blue', 'purple']
for i, n in enumerate(sample_sizes):
    # Use subset of data
    data_subset = synthetic_data[:n]
    data_mean = np.mean(data_subset)
    
    # Calculate posterior for this sample size
    posterior_var = 1 / (1/prior_var + n/likelihood_var)
    posterior_mean = posterior_var * (prior_mean/prior_var + sum(data_subset)/likelihood_var)
    
    # Plot the posterior
    plt.plot(mu_range, norm.pdf(mu_range, posterior_mean, np.sqrt(posterior_var)), 
             color=colors[i], linestyle='-', 
             label=f'Posterior with n={n}, μ={posterior_mean:.4f}', linewidth=2)
    
    # Mark the posterior mean
    plt.axvline(x=posterior_mean, color=colors[i], linestyle=':', alpha=0.7)

# Mark the true mean and data mean for reference
plt.axvline(x=true_mean, color='black', linestyle='-', 
            label=f'True Mean: {true_mean}', linewidth=2)

plt.xlabel('μ', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Influence of Sample Size on Posterior Distribution', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "prior_influence.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

print("Figure illustrating how the influence of the prior diminishes with increasing sample size has been saved.")
print("This demonstrates that with more data, the posterior is increasingly dominated by the likelihood,")
print("converging toward the true parameter value regardless of the prior.") 