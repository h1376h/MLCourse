import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

print("\n=== ADVANCED BETA DISTRIBUTION EXAMPLES: STEP-BY-STEP SOLUTIONS ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Example 1: Conjugate Prior Properties
print("Example 1: Conjugate Prior Properties")
# Show how Beta-Bernoulli conjugacy works
alpha_prior, beta_prior = 2, 2
data_sequences = [
    [1, 1, 0, 1],  # 3 successes, 1 failure
    [1, 1, 1, 1, 0, 0],  # 4 successes, 2 failures
    [1, 1, 1, 1, 1, 0, 0, 0]  # 5 successes, 3 failures
]

x = np.linspace(0, 1, 1000)
plt.figure(figsize=(12, 8))

# Plot prior
prior_pdf = stats.beta.pdf(x, alpha_prior, beta_prior)
plt.plot(x, prior_pdf, 'b-', label='Prior: Beta(2,2)', linewidth=2)

# Plot posteriors for different data sequences
colors = ['g', 'r', 'm']  # Changed 'purple' to 'm' (magenta)
for i, data in enumerate(data_sequences):
    successes = sum(data)
    failures = len(data) - successes
    alpha_post = alpha_prior + successes
    beta_post = beta_prior + failures
    post_pdf = stats.beta.pdf(x, alpha_post, beta_post)
    plt.plot(x, post_pdf, f'{colors[i]}-', 
             label=f'Posterior {i+1}: Beta({alpha_post},{beta_post})', 
             linewidth=2)

plt.title('Beta-Bernoulli Conjugate Prior Demonstration')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'beta_conjugate_prior.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: MAP vs MLE Estimation
print("\nExample 2: MAP vs MLE Estimation")
# Compare MAP and MLE estimates for different sample sizes
sample_sizes = [5, 20, 100]
success_ratio = 0.6
alpha_prior, beta_prior = 2, 2

plt.figure(figsize=(12, 8))
x = np.linspace(0, 1, 1000)

for i, n in enumerate(sample_sizes):
    successes = int(n * success_ratio)
    failures = n - successes
    
    # MLE estimate (just the sample proportion)
    mle = successes / n
    
    # MAP estimate (using Beta prior)
    alpha_post = alpha_prior + successes
    beta_post = beta_prior + failures
    map_estimate = (alpha_post - 1) / (alpha_post + beta_post - 2)
    
    # Plot the posterior
    post_pdf = stats.beta.pdf(x, alpha_post, beta_post)
    plt.plot(x, post_pdf, label=f'n={n}, MAP={map_estimate:.3f}, MLE={mle:.3f}', linewidth=2)
    
    # Add vertical lines for estimates
    plt.axvline(x=mle, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=map_estimate, color='b', linestyle='--', alpha=0.5)

plt.title('MAP vs MLE Estimation with Beta Prior')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'beta_map_vs_mle.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Predictive Distribution
print("\nExample 3: Predictive Distribution")
# Show how to make predictions using the posterior
alpha_prior, beta_prior = 2, 2
observed_data = [1, 1, 0, 1]  # 3 successes, 1 failure
alpha_post = alpha_prior + sum(observed_data)
beta_post = beta_prior + len(observed_data) - sum(observed_data)

# Calculate predictive probability for next observation
predictive_success = alpha_post / (alpha_post + beta_post)
predictive_failure = beta_post / (alpha_post + beta_post)

# Plot posterior and predictive probabilities
plt.figure(figsize=(10, 6))
x = np.linspace(0, 1, 1000)
post_pdf = stats.beta.pdf(x, alpha_post, beta_post)
plt.plot(x, post_pdf, 'b-', label='Posterior', linewidth=2)
plt.axvline(x=predictive_success, color='r', linestyle='--', 
            label=f'Predictive Success: {predictive_success:.3f}')
plt.axvline(x=predictive_failure, color='g', linestyle='--', 
            label=f'Predictive Failure: {predictive_failure:.3f}')

plt.title('Predictive Distribution from Beta Posterior')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'beta_predictive.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 4: Effect of Prior Strength
print("\nExample 4: Effect of Prior Strength")
# Compare different prior strengths with same data
data = [1, 1, 0, 1]  # 3 successes, 1 failure
prior_strengths = [1, 2, 5, 10]  # Different prior sample sizes

plt.figure(figsize=(10, 6))
x = np.linspace(0, 1, 1000)

for strength in prior_strengths:
    # Use Beta(2,2) as base prior, scaled by strength
    alpha_prior = 2 * strength
    beta_prior = 2 * strength
    
    alpha_post = alpha_prior + sum(data)
    beta_post = beta_prior + len(data) - sum(data)
    
    post_pdf = stats.beta.pdf(x, alpha_post, beta_post)
    plt.plot(x, post_pdf, label=f'Prior Strength: {strength}', linewidth=2)

plt.title('Effect of Prior Strength on Posterior')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'beta_prior_strength.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 5: Sequential Bayesian Updating
print("\nExample 5: Sequential Bayesian Updating")
# Show how the posterior evolves with sequential data
alpha_prior, beta_prior = 2, 2
sequential_data = [1, 0, 1, 1, 0, 1, 1, 1]

plt.figure(figsize=(12, 8))
x = np.linspace(0, 1, 1000)

# Plot prior
alpha, beta = alpha_prior, beta_prior
plt.plot(x, stats.beta.pdf(x, alpha, beta), 'b-', label='Prior', linewidth=2)

# Update and plot after each observation
colors = ['g', 'r', 'm', 'c', 'y', 'k', 'w', 'b']  # Changed colors to standard matplotlib colors
for i, obs in enumerate(sequential_data):
    if obs == 1:
        alpha += 1
    else:
        beta += 1
    plt.plot(x, stats.beta.pdf(x, alpha, beta), f'{colors[i]}-', 
             label=f'After {i+1} observations', linewidth=2)

plt.title('Sequential Bayesian Updating')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'beta_sequential.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 6: Effect of Different Prior Parameters
print("\nExample 6: Effect of Different Prior Parameters")
# Compare different prior parameter combinations
prior_combinations = [(1, 1), (2, 2), (5, 1), (1, 5), (0.5, 0.5)]
data = [1, 1, 0, 1]  # 3 successes, 1 failure

plt.figure(figsize=(12, 8))
x = np.linspace(0, 1, 1000)

for alpha, beta in prior_combinations:
    # Calculate posterior
    alpha_post = alpha + sum(data)
    beta_post = beta + len(data) - sum(data)
    
    # Plot prior and posterior
    prior_pdf = stats.beta.pdf(x, alpha, beta)
    post_pdf = stats.beta.pdf(x, alpha_post, beta_post)
    
    plt.plot(x, prior_pdf, '--', label=f'Prior: Beta({alpha},{beta})', linewidth=2)
    plt.plot(x, post_pdf, '-', label=f'Posterior: Beta({alpha_post},{beta_post})', linewidth=2)

plt.title('Effect of Different Prior Parameters on Posterior')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'beta_prior_combinations.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 7: Beta Distribution Shape Analysis
print("\nExample 7: Beta Distribution Shape Analysis")
# Show how different parameter combinations affect the shape
parameter_sets = [
    (0.5, 0.5),  # U-shaped
    (1, 1),      # Uniform
    (2, 2),      # Symmetric
    (5, 1),      # Skewed right
    (1, 5),      # Skewed left
    (2, 5),      # Asymmetric
    (5, 2)       # Asymmetric
]

plt.figure(figsize=(12, 8))
x = np.linspace(0, 1, 1000)

for alpha, beta in parameter_sets:
    pdf = stats.beta.pdf(x, alpha, beta)
    plt.plot(x, pdf, label=f'Beta({alpha},{beta})', linewidth=2)

plt.title('Beta Distribution Shape Analysis')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'beta_shape_analysis.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 8: Beta Distribution Moments
print("\nExample 8: Beta Distribution Moments")
# Visualize mean, mode, and variance for different parameters
alpha_values = np.linspace(0.5, 10, 20)
beta_values = np.linspace(0.5, 10, 20)
alpha_grid, beta_grid = np.meshgrid(alpha_values, beta_values)

# Calculate moments
mean = alpha_grid / (alpha_grid + beta_grid)
mode = (alpha_grid - 1) / (alpha_grid + beta_grid - 2)
variance = (alpha_grid * beta_grid) / ((alpha_grid + beta_grid)**2 * (alpha_grid + beta_grid + 1))

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
titles = ['Mean', 'Mode', 'Variance']
data = [mean, mode, variance]

for ax, title, d in zip(axes, titles, data):
    im = ax.imshow(d, origin='lower', extent=[0.5, 10, 0.5, 10])
    ax.set_title(title)
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Beta')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'beta_moments.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 9: Beta Distribution and Binomial Likelihood
print("\nExample 9: Beta Distribution and Binomial Likelihood")
# Show how Beta posterior relates to binomial likelihood
n = 10  # Number of trials
k = 7   # Number of successes
alpha_prior, beta_prior = 2, 2

# Calculate posterior
alpha_post = alpha_prior + k
beta_post = beta_prior + n - k

# Calculate likelihood (scaled for visualization)
x = np.linspace(0, 1, 1000)
likelihood = stats.binom.pmf(k, n, x) * 10  # Scale for better visualization
posterior = stats.beta.pdf(x, alpha_post, beta_post)
prior = stats.beta.pdf(x, alpha_prior, beta_prior)

plt.figure(figsize=(12, 8))
plt.plot(x, prior, 'b-', label='Prior: Beta(2,2)', linewidth=2)
plt.plot(x, likelihood, 'g-', label='Likelihood (scaled)', linewidth=2)
plt.plot(x, posterior, 'r-', label=f'Posterior: Beta({alpha_post},{beta_post})', linewidth=2)

plt.title('Beta Prior, Binomial Likelihood, and Beta Posterior')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'beta_binomial_likelihood.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll advanced beta distribution example images created successfully.") 