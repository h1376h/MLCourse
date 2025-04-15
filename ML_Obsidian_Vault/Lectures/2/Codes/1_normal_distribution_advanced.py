import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

print("\n=== ADVANCED NORMAL DISTRIBUTION VISUALIZATIONS ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Example 1: Bayesian Updating with Normal Prior and Likelihood
print("Example 1: Bayesian Updating with Normal Prior and Likelihood")
mu_prior = 0
sigma_prior = 2
mu_likelihood = 3
sigma_likelihood = 1
n_samples = 10

# Calculate posterior parameters
sigma_posterior = 1 / np.sqrt(1/sigma_prior**2 + n_samples/sigma_likelihood**2)
mu_posterior = (mu_prior/sigma_prior**2 + n_samples*mu_likelihood/sigma_likelihood**2) * sigma_posterior**2

x = np.linspace(-5, 8, 1000)
prior = stats.norm.pdf(x, mu_prior, sigma_prior)
likelihood = stats.norm.pdf(x, mu_likelihood, sigma_likelihood/np.sqrt(n_samples))
posterior = stats.norm.pdf(x, mu_posterior, sigma_posterior)

plt.figure(figsize=(10, 6))
plt.plot(x, prior, 'b-', label='Prior N(0,4)', linewidth=2)
plt.plot(x, likelihood, 'g-', label=f'Likelihood N(3,{1/n_samples:.2f})', linewidth=2)
plt.plot(x, posterior, 'r-', label=f'Posterior N({mu_posterior:.2f},{sigma_posterior**2:.2f})', linewidth=2)
plt.title('Bayesian Updating with Normal Distributions')
plt.xlabel('Parameter Value')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'normal_bayesian_updating.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Effect of Sample Size on Posterior
print("\nExample 2: Effect of Sample Size on Posterior")
mu_prior = 0
sigma_prior = 2
mu_likelihood = 3
sigma_likelihood = 1
sample_sizes = [1, 5, 20, 100]

plt.figure(figsize=(10, 6))
for n in sample_sizes:
    sigma_posterior = 1 / np.sqrt(1/sigma_prior**2 + n/sigma_likelihood**2)
    mu_posterior = (mu_prior/sigma_prior**2 + n*mu_likelihood/sigma_likelihood**2) * sigma_posterior**2
    posterior = stats.norm.pdf(x, mu_posterior, sigma_posterior)
    plt.plot(x, posterior, label=f'n = {n}', linewidth=2)

plt.title('Effect of Sample Size on Posterior Distribution')
plt.xlabel('Parameter Value')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'normal_sample_size_effect.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Credible Intervals
print("\nExample 3: Credible Intervals")
mu = 0
sigma = 1
x = np.linspace(-4, 4, 1000)
pdf = stats.norm.pdf(x, mu, sigma)

# Calculate 50%, 80%, and 95% credible intervals
intervals = [0.5, 0.8, 0.95]
colors = ['blue', 'green', 'red']

plt.figure(figsize=(10, 6))
plt.plot(x, pdf, 'k-', linewidth=2)

for interval, color in zip(intervals, colors):
    lower = stats.norm.ppf((1-interval)/2, mu, sigma)
    upper = stats.norm.ppf(1-(1-interval)/2, mu, sigma)
    plt.fill_between(x, pdf, where=(x >= lower) & (x <= upper), 
                    color=color, alpha=0.3, label=f'{interval*100}% CI')
    plt.axvline(x=lower, color=color, linestyle='--', alpha=0.5)
    plt.axvline(x=upper, color=color, linestyle='--', alpha=0.5)

plt.title('Credible Intervals for Normal Distribution')
plt.xlabel('Parameter Value')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'normal_credible_intervals.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 4: MAP vs MLE Estimation
print("\nExample 4: MAP vs MLE Estimation")
mu_prior = 0
sigma_prior = 1
mu_likelihood = 2
sigma_likelihood = 1
n_samples = 10

# Calculate MAP and MLE
sigma_map = 1 / np.sqrt(1/sigma_prior**2 + n_samples/sigma_likelihood**2)
mu_map = (mu_prior/sigma_prior**2 + n_samples*mu_likelihood/sigma_likelihood**2) * sigma_map**2
mu_mle = mu_likelihood

x = np.linspace(-2, 4, 1000)
prior = stats.norm.pdf(x, mu_prior, sigma_prior)
likelihood = stats.norm.pdf(x, mu_likelihood, sigma_likelihood/np.sqrt(n_samples))
posterior = stats.norm.pdf(x, mu_map, sigma_map)

plt.figure(figsize=(10, 6))
plt.plot(x, prior, 'b-', label='Prior', linewidth=2)
plt.plot(x, likelihood, 'g-', label='Likelihood', linewidth=2)
plt.plot(x, posterior, 'r-', label='Posterior', linewidth=2)
plt.axvline(x=mu_mle, color='k', linestyle='--', label='MLE')
plt.axvline(x=mu_map, color='m', linestyle='--', label='MAP')
plt.title('MAP vs MLE Estimation')
plt.xlabel('Parameter Value')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'normal_map_vs_mle.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll advanced normal distribution visualizations created successfully.") 