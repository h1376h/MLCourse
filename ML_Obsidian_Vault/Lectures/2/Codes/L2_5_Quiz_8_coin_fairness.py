import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from scipy.stats import beta, binom

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_5_Quiz_8")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Set up the problem
print_step_header(1, "Understanding the Problem")

print("Given:")
print("- We have a coin with unknown probability θ of landing heads")
print("- Our prior belief about θ is represented by Beta(3, 3)")
print("- We observe 5 heads out of 8 coin flips")
print()
print("Questions:")
print("1. What is the posterior distribution?")
print("2. What is the posterior mean probability of the coin showing heads?")
print("3. How does this posterior mean compare to the maximum likelihood estimate (5/8)?")
print()

# Step 2: Understanding Beta-Binomial conjugacy
print_step_header(2, "Beta-Binomial Conjugate Prior Relationship")

print("For a binomial likelihood (such as coin flips), the conjugate prior is the Beta distribution.")
print("The Beta PDF for the parameter θ with parameters α and β is:")
print("p(θ | α, β) = [θ^(α-1) * (1-θ)^(β-1)] / B(α, β)")
print("where B(α, β) is the Beta function that normalizes the distribution.")
print()
print("The parameters α and β can be interpreted as:")
print("- α-1: the number of 'prior successes' (heads)")
print("- β-1: the number of 'prior failures' (tails)")
print()
print("Our prior Beta(3, 3) represents a belief that the coin is fair (symmetric around 0.5)")
print("with an effective prior sample size of (3-1) + (3-1) = 4 flips.")
print()

# Visualize the prior Beta distribution
alpha_prior = 3
beta_prior = 3
theta_range = np.linspace(0, 1, 1000)
prior_pdf = beta.pdf(theta_range, alpha_prior, beta_prior)

plt.figure(figsize=(10, 6))
plt.plot(theta_range, prior_pdf, 'b-', linewidth=3, label=f'Beta({alpha_prior}, {beta_prior}) Prior')
plt.axvline(x=0.5, color='k', linestyle='--', label='θ = 0.5 (Fair Coin)')

# Add annotations about the prior
plt.text(0.5, 1.0, f'Prior Mean: {alpha_prior/(alpha_prior+beta_prior):.2f}', fontsize=12, ha='center')
plt.title('Prior Distribution for θ - Beta(3, 3)', fontsize=14)
plt.xlabel('θ (Probability of Heads)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "prior_distribution.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 3: Bayesian Updating with Beta-Binomial
print_step_header(3, "Bayesian Updating with Beta-Binomial Model")

print("For a Beta-Binomial model, the Bayesian updating follows these rules:")
print("If the prior is Beta(α, β) and we observe h heads and t tails,")
print("then the posterior is Beta(α + h, β + t)")
print()
print("Let's apply this to our problem:")
print("- Prior: Beta(α=3, β=3)")
print("- Data: 5 heads, 3 tails out of 8 flips")
print()
print("Therefore, the posterior distribution is:")
print("Beta(α + h, β + t) = Beta(3 + 5, 3 + 3) = Beta(8, 6)")
print()

# Calculate posterior parameters
heads = 5
tails = 8 - heads
alpha_posterior = alpha_prior + heads
beta_posterior = beta_prior + tails

print(f"Posterior parameters: α' = {alpha_posterior}, β' = {beta_posterior}")
print()

# Visualize prior, likelihood, and posterior
theta_range = np.linspace(0, 1, 1000)
prior_pdf = beta.pdf(theta_range, alpha_prior, beta_prior)
likelihood = binom.pmf(heads, heads + tails, theta_range)
posterior_pdf = beta.pdf(theta_range, alpha_posterior, beta_posterior)

# Normalize likelihood for better visualization
likelihood = likelihood / np.max(likelihood) * np.max(posterior_pdf) * 0.8

plt.figure(figsize=(10, 6))
plt.plot(theta_range, prior_pdf, 'b--', linewidth=2, label=f'Prior: Beta({alpha_prior}, {beta_prior})')
plt.plot(theta_range, likelihood, 'g-.', linewidth=2, label='Likelihood (scaled)')
plt.plot(theta_range, posterior_pdf, 'r-', linewidth=3, label=f'Posterior: Beta({alpha_posterior}, {beta_posterior})')

# Add markers for important values
prior_mean = alpha_prior / (alpha_prior + beta_prior)
posterior_mean = alpha_posterior / (alpha_posterior + beta_posterior)
mle = heads / (heads + tails)  # Maximum likelihood estimate

plt.axvline(x=prior_mean, color='b', linestyle=':', label=f'Prior Mean: {prior_mean:.2f}')
plt.axvline(x=posterior_mean, color='r', linestyle=':', label=f'Posterior Mean: {posterior_mean:.2f}')
plt.axvline(x=mle, color='g', linestyle=':', label=f'MLE: {mle:.2f}')
plt.axvline(x=0.5, color='k', linestyle='--', label='Fair Coin: θ = 0.5')

plt.title('Bayesian Updating for Coin Fairness', fontsize=14)
plt.xlabel('θ (Probability of Heads)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.grid(True)
plt.legend(loc='best')
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "bayesian_updating.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 4: Posterior Analysis
print_step_header(4, "Posterior Analysis")

# Calculate posterior statistics
posterior_mean = alpha_posterior / (alpha_posterior + beta_posterior)
posterior_mode = (alpha_posterior - 1) / (alpha_posterior + beta_posterior - 2) if alpha_posterior > 1 and beta_posterior > 1 else 0
posterior_variance = (alpha_posterior * beta_posterior) / ((alpha_posterior + beta_posterior)**2 * (alpha_posterior + beta_posterior + 1))
posterior_std = np.sqrt(posterior_variance)

print(f"Posterior Distribution: Beta({alpha_posterior}, {beta_posterior})")
print(f"Posterior Mean: E[θ|data] = α'/(α'+β') = {alpha_posterior}/({alpha_posterior}+{beta_posterior}) = {posterior_mean:.4f}")
print(f"Posterior Mode: (α'-1)/(α'+β'-2) = {posterior_mode:.4f}")
print(f"Posterior Variance: (α'*β')/((α'+β')²*(α'+β'+1)) = {posterior_variance:.4f}")
print(f"Posterior Standard Deviation: {posterior_std:.4f}")
print()

# Calculate 95% credible interval
lower_ci = beta.ppf(0.025, alpha_posterior, beta_posterior)
upper_ci = beta.ppf(0.975, alpha_posterior, beta_posterior)
print(f"95% Credible Interval: [{lower_ci:.4f}, {upper_ci:.4f}]")
print()

# Calculate probability that the coin is biased toward heads (θ > 0.5)
prob_biased_heads = 1 - beta.cdf(0.5, alpha_posterior, beta_posterior)
print(f"Probability that the coin is biased toward heads (θ > 0.5): {prob_biased_heads:.4f}")
print()

# Visualize posterior with credible interval
theta_range = np.linspace(0, 1, 1000)
posterior_pdf = beta.pdf(theta_range, alpha_posterior, beta_posterior)

plt.figure(figsize=(10, 6))
plt.plot(theta_range, posterior_pdf, 'r-', linewidth=3, label=f'Posterior: Beta({alpha_posterior}, {beta_posterior})')

# Shade the 95% credible interval
ci_x = np.linspace(lower_ci, upper_ci, 1000)
ci_y = beta.pdf(ci_x, alpha_posterior, beta_posterior)
plt.fill_between(ci_x, ci_y, alpha=0.3, color='red', label='95% Credible Interval')

# Add markers for important values
plt.axvline(x=posterior_mean, color='r', linestyle=':', label=f'Posterior Mean: {posterior_mean:.4f}')
plt.axvline(x=posterior_mode, color='b', linestyle=':', label=f'Posterior Mode: {posterior_mode:.4f}')
plt.axvline(x=0.5, color='k', linestyle='--', label='Fair Coin: θ = 0.5')

# Shade the region where coin is biased towards heads
bias_x = np.linspace(0.5, 1, 500)
bias_y = beta.pdf(bias_x, alpha_posterior, beta_posterior)
plt.fill_between(bias_x, bias_y, alpha=0.2, color='green', label=f'P(θ > 0.5) = {prob_biased_heads:.4f}')

plt.title('Posterior Distribution with 95% Credible Interval', fontsize=14)
plt.xlabel('θ (Probability of Heads)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.grid(True)
plt.legend(loc='best')
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "posterior_credible_interval.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 5: Comparison with the MLE
print_step_header(5, "Comparison with Maximum Likelihood Estimate")

# Calculate MLE
mle = heads / (heads + tails)
print(f"Maximum Likelihood Estimate (MLE): {heads}/{heads+tails} = {mle:.4f}")
print(f"Posterior Mean: {posterior_mean:.4f}")
print(f"Difference: {posterior_mean - mle:.4f}")
print()
print("Interpretation of the difference:")
if posterior_mean < mle:
    print(f"The posterior mean is lower than the MLE by {mle - posterior_mean:.4f}.")
    print("This 'shrinkage' is due to the influence of the prior, which pulls the estimate toward 0.5.")
else:
    print(f"The posterior mean is higher than the MLE by {posterior_mean - mle:.4f}.")
    print("This is due to the influence of the prior, which pulls the estimate toward 0.5.")
print()

# Calculate the "effective sample size" of the prior
prior_effective_sample_size = alpha_prior + beta_prior - 2  # Subtract 2 for Beta(1,1) uniform prior
data_sample_size = heads + tails
posterior_weight_data = data_sample_size / (prior_effective_sample_size + data_sample_size)
posterior_weight_prior = prior_effective_sample_size / (prior_effective_sample_size + data_sample_size)

print(f"Effective sample size of the prior: {prior_effective_sample_size}")
print(f"Data sample size: {data_sample_size}")
print(f"In the posterior, the data carries {posterior_weight_data:.2%} of the weight")
print(f"and the prior carries {posterior_weight_prior:.2%} of the weight.")
print()

# Visual demonstration of MLE vs. Posterior Mean
x_vals = [0, 1, 2, 3, 4, 5, 6, 7, 8]
mle_estimates = [x/8 for x in x_vals]
posterior_means = [(alpha_prior + x)/(alpha_prior + beta_prior + 8) for x in x_vals]

plt.figure(figsize=(10, 6))
plt.plot(x_vals, mle_estimates, 'go-', linewidth=2, markersize=8, label='MLE: h/n')
plt.plot(x_vals, posterior_means, 'ro-', linewidth=2, markersize=8, label=f'Posterior Mean: (α+h)/(α+β+n)')

# Add vertical line at our observed data
plt.axvline(x=heads, color='k', linestyle='--', label=f'Observed: {heads} heads')

# Add annotations
plt.text(heads, mle, f'MLE = {mle:.4f}', fontsize=10, ha='left', va='bottom')
plt.text(heads, posterior_mean, f'Posterior Mean = {posterior_mean:.4f}', fontsize=10, ha='left', va='top')

plt.title('MLE vs. Posterior Mean for Different Possible Observations', fontsize=14)
plt.xlabel('Number of Heads (out of 8 flips)', fontsize=12)
plt.ylabel('Probability Estimate for θ', fontsize=12)
plt.grid(True)
plt.legend(loc='best')
plt.xticks(x_vals)
plt.ylim(0, 1)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "mle_vs_posterior.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step A: Sample size effect
print_step_header(6, "Effect of Sample Size on Posterior")

# Define different sample sizes with the same proportion of heads
sample_sizes = [8, 16, 32, 64, 128]
head_proportions = [heads / (heads + tails)] * len(sample_sizes)
head_counts = [int(p * n) for p, n in zip(head_proportions, sample_sizes)]
tail_counts = [n - h for h, n in zip(head_counts, sample_sizes)]

plt.figure(figsize=(12, 8))

for i, (n, h, t) in enumerate(zip(sample_sizes, head_counts, tail_counts)):
    # Calculate posterior with this sample size
    alpha_post = alpha_prior + h
    beta_post = beta_prior + t
    post_mean = alpha_post / (alpha_post + beta_post)
    
    # Plot the posterior
    theta_range = np.linspace(0, 1, 1000)
    posterior_pdf = beta.pdf(theta_range, alpha_post, beta_post)
    plt.plot(theta_range, posterior_pdf, linewidth=2, 
             label=f'n={n}, h={h} → Posterior Mean: {post_mean:.4f}')

# Add vertical lines for reference
plt.axvline(x=0.5, color='k', linestyle='--', label='Fair Coin: θ = 0.5')
plt.axvline(x=mle, color='g', linestyle='--', label=f'MLE: {mle:.4f}')

plt.title('Effect of Sample Size on Posterior Distribution\n(Constant Proportion of Heads)', fontsize=14)
plt.xlabel('θ (Probability of Heads)', fontsize=12)
plt.ylabel('Posterior Probability Density', fontsize=12)
plt.grid(True)
plt.legend(loc='best')
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "sample_size_effect.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 7: Prior sensitivity analysis
print_step_header(7, "Prior Sensitivity Analysis")

# Define different priors
prior_pairs = [
    (1, 1, "Uniform - Beta(1, 1)"),
    (3, 3, "Original - Beta(3, 3)"),
    (10, 10, "Stronger - Beta(10, 10)"),
    (1, 3, "Skeptical - Beta(1, 3)")
]

plt.figure(figsize=(12, 8))

for alpha_p, beta_p, label in prior_pairs:
    # Calculate posterior with this prior
    alpha_post = alpha_p + heads
    beta_post = beta_p + tails
    post_mean = alpha_post / (alpha_post + beta_post)
    
    # Plot the posterior
    theta_range = np.linspace(0, 1, 1000)
    posterior_pdf = beta.pdf(theta_range, alpha_post, beta_post)
    plt.plot(theta_range, posterior_pdf, linewidth=2, 
             label=f'{label} → Posterior Mean: {post_mean:.4f}')

# Add vertical lines for reference
plt.axvline(x=0.5, color='k', linestyle='--', label='Fair Coin: θ = 0.5')
plt.axvline(x=mle, color='g', linestyle='--', label=f'MLE: {mle:.4f}')

plt.title('Effect of Prior Choice on Posterior Distribution', fontsize=14)
plt.xlabel('θ (Probability of Heads)', fontsize=12)
plt.ylabel('Posterior Probability Density', fontsize=12)
plt.grid(True)
plt.legend(loc='best')
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "prior_sensitivity.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 8: Conclusion and Answer
print_step_header(8, "Conclusion")

print("In summary:")
print("1. The posterior distribution is Beta(8, 6)")
print(f"2. The posterior mean probability of the coin showing heads is {posterior_mean:.4f}")
print(f"3. Comparison with the MLE:")
print(f"   - MLE: {mle:.4f}")
print(f"   - Posterior Mean: {posterior_mean:.4f}")
print(f"   - Difference: {posterior_mean - mle:.4f}")
print()
print("The posterior mean is slightly lower than the MLE. This is because:")
print("1. The prior Beta(3, 3) is centered at 0.5 (representing belief in a fair coin)")
print("2. The observed data (5 heads out of 8 flips) suggests a bias toward heads")
print("3. The posterior mean 'shrinks' the MLE toward the prior mean, with the")
print("   amount of shrinkage depending on the relative 'strength' of the prior compared to the data")
print("4. With more data (more coin flips), the posterior would converge toward the MLE")
print()
print("This illustrates the Bayesian approach of combining prior beliefs with observed data,")
print("and how the posterior serves as a compromise between the two sources of information.") 