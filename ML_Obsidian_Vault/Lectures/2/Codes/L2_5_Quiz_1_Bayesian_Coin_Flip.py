import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import os
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_5_Quiz_1")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Understanding the problem
print_step_header(1, "Understanding the Problem")

print("Given:")
print("- We have a coin with unknown probability θ of landing heads")
print("- We assume a prior distribution for θ as Beta(2, 2)")
print("- We toss the coin 10 times and observe 7 heads and 3 tails")
print()
print("Task:")
print("1. Write down the likelihood function for the observed data")
print("2. Calculate the posterior distribution for θ")
print("3. Find the posterior mean, mode, and variance of θ")
print("4. Calculate the 95% credible interval for θ")
print()

# Step 2: Visualize the prior distribution - Beta(2,2)
print_step_header(2, "Visualizing the Prior Distribution")

print("We start with a Beta(2, 2) prior distribution for θ.")
print("The Beta distribution is a conjugate prior for the Bernoulli likelihood.")
print("The probability density function (PDF) of Beta(α, β) is:")
print("f(θ; α, β) = θ^(α-1) * (1-θ)^(β-1) / B(α, β)")
print("where B(α, β) is the Beta function")
print()
print("For Beta(2, 2):")
print("f(θ; 2, 2) = θ^(2-1) * (1-θ)^(2-1) / B(2, 2) = θ * (1-θ) * 6")
print("This is a symmetric distribution with mean = 0.5 and mode = 0.5")
print()

# Plot the prior Beta(2,2) distribution
theta = np.linspace(0, 1, 1000)
prior = beta.pdf(theta, 2, 2)

plt.figure(figsize=(10, 6))
plt.plot(theta, prior, 'b-', linewidth=2, label='Prior: Beta(2, 2)')
plt.fill_between(theta, 0, prior, alpha=0.2, color='blue')
plt.xlabel('θ (Probability of Heads)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title('Prior Distribution: Beta(2, 2)', fontsize=14)
plt.axvline(x=0.5, color='red', linestyle='--', label='Mean & Mode = 0.5')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "prior_distribution.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 3: Derive the likelihood function
print_step_header(3, "Deriving the Likelihood Function")

print("For a sequence of coin flips, the likelihood function follows a binomial distribution:")
print("L(θ | data) = P(data | θ) = C(n,k) * θ^k * (1-θ)^(n-k)")
print("where:")
print("- n is the number of trials (10 in our case)")
print("- k is the number of successes (7 heads)")
print("- C(n,k) is the binomial coefficient (constant for a given n and k)")
print()
print("In our case:")
print("L(θ | 7 heads in 10 tosses) = C(10,7) * θ^7 * (1-θ)^3")
print("L(θ | data) ∝ θ^7 * (1-θ)^3  (dropping the constant)")
print()

# Plot the likelihood function
n, k = 10, 7  # 10 trials, 7 successes
likelihood = beta.pdf(theta, k+1, n-k+1) * beta.cdf(1, k+1, n-k+1) / beta.cdf(1, 1, 1)

plt.figure(figsize=(10, 6))
plt.plot(theta, likelihood, 'g-', linewidth=2, label='Likelihood: θ^7 * (1-θ)^3')
plt.fill_between(theta, 0, likelihood, alpha=0.2, color='green')
plt.xlabel('θ (Probability of Heads)', fontsize=12)
plt.ylabel('Likelihood (unnormalized)', fontsize=12)
plt.title('Likelihood Function for 7 Heads in 10 Tosses', fontsize=14)
plt.axvline(x=k/n, color='red', linestyle='--', label=f'MLE = {k/n}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "likelihood_function.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 4: Calculate the posterior distribution
print_step_header(4, "Calculating the Posterior Distribution")

print("Using Bayes' theorem, the posterior distribution is:")
print("P(θ | data) ∝ P(data | θ) * P(θ)")
print()
print("With a Beta(α, β) prior and binomial likelihood, the posterior is also a Beta distribution:")
print("Posterior: Beta(α + k, β + n - k)")
print()
print("In our case:")
print("Prior: Beta(2, 2)")
print("Likelihood: Binomial with 7 heads out of 10 tosses")
print("Posterior: Beta(2 + 7, 2 + 10 - 7) = Beta(9, 5)")
print()

# Compute the posterior distribution
posterior_alpha = 2 + k  # 9
posterior_beta = 2 + n - k  # 5
posterior = beta.pdf(theta, posterior_alpha, posterior_beta)

# Calculate posterior statistics
posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
posterior_mode = (posterior_alpha - 1) / (posterior_alpha + posterior_beta - 2)
posterior_var = (posterior_alpha * posterior_beta) / ((posterior_alpha + posterior_beta)**2 * (posterior_alpha + posterior_beta + 1))
posterior_std = np.sqrt(posterior_var)

print(f"Posterior Distribution: Beta(9, 5)")
print(f"Posterior Mean: {posterior_mean:.4f}")
print(f"Posterior Mode: {posterior_mode:.4f}")
print(f"Posterior Variance: {posterior_var:.4f}")
print(f"Posterior Standard Deviation: {posterior_std:.4f}")
print()

# Plot the posterior distribution
plt.figure(figsize=(10, 6))
plt.plot(theta, posterior, 'r-', linewidth=2, label='Posterior: Beta(9, 5)')
plt.fill_between(theta, 0, posterior, alpha=0.2, color='red')
plt.xlabel('θ (Probability of Heads)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title('Posterior Distribution after Observing 7 Heads in 10 Tosses', fontsize=14)
plt.axvline(x=posterior_mean, color='darkred', linestyle='--', label=f'Mean = {posterior_mean:.4f}')
plt.axvline(x=posterior_mode, color='orange', linestyle=':', label=f'Mode = {posterior_mode:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "posterior_distribution.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 5: Calculate the 95% credible interval
print_step_header(5, "Calculating the 95% Credible Interval")

print("The 95% credible interval represents the range of values for θ that")
print("contains 95% of the posterior probability mass.")
print()
print("For a Beta distribution, we can calculate percentiles using the inverse CDF.")
print()

# Calculate the 95% credible interval
lower_bound = beta.ppf(0.025, posterior_alpha, posterior_beta)
upper_bound = beta.ppf(0.975, posterior_alpha, posterior_beta)

print(f"95% Credible Interval: [{lower_bound:.4f}, {upper_bound:.4f}]")
print()
print("Interpretation: With 95% probability, the true probability of the coin landing heads")
print(f"is between {lower_bound:.4f} and {upper_bound:.4f} given our prior and observed data.")
print()

# Plot the posterior with the credible interval highlighted
plt.figure(figsize=(10, 6))
plt.plot(theta, posterior, 'r-', linewidth=2, label='Posterior: Beta(9, 5)')

# Highlight the credible interval
idx = (theta >= lower_bound) & (theta <= upper_bound)
plt.fill_between(theta[idx], 0, posterior[idx], alpha=0.4, color='red', 
                 label=f'95% Credible Interval\n[{lower_bound:.4f}, {upper_bound:.4f}]')

plt.xlabel('θ (Probability of Heads)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title('Posterior Distribution with 95% Credible Interval', fontsize=14)
plt.axvline(x=posterior_mean, color='darkred', linestyle='--', label=f'Mean = {posterior_mean:.4f}')
plt.axvline(x=posterior_mode, color='orange', linestyle=':', label=f'Mode = {posterior_mode:.4f}')
plt.axvline(x=lower_bound, color='green', linestyle='-')
plt.axvline(x=upper_bound, color='green', linestyle='-')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "credible_interval.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 6: Compare prior, likelihood, and posterior
print_step_header(6, "Comparing Prior, Likelihood, and Posterior")

print("Now let's compare the prior, likelihood, and posterior distributions")
print("to visualize how our belief about θ was updated by the observed data.")
print()

# Normalize the likelihood for better comparison
likelihood_normalized = likelihood / np.max(likelihood) * np.max(posterior)

plt.figure(figsize=(10, 6))
plt.plot(theta, prior, 'b-', linewidth=2, label='Prior: Beta(2, 2)')
plt.plot(theta, likelihood_normalized, 'g-', linewidth=2, label='Likelihood (scaled)')
plt.plot(theta, posterior, 'r-', linewidth=2, label='Posterior: Beta(9, 5)')
plt.xlabel('θ (Probability of Heads)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Bayesian Updating: Prior → Likelihood → Posterior', fontsize=14)
plt.axvline(x=0.5, color='blue', linestyle='--', label='Prior Mean = 0.5')
plt.axvline(x=0.7, color='green', linestyle='--', label='MLE = 0.7')
plt.axvline(x=posterior_mean, color='red', linestyle='--', label=f'Posterior Mean = {posterior_mean:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "comparison.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 7: Visualize the effect of different priors
print_step_header(7, "Effect of Different Priors")

print("Let's visualize how different prior distributions would affect our posterior inference.")
print("We'll compare our Beta(2, 2) prior with:")
print("1. Uniform prior: Beta(1, 1) - No preference for any value")
print("2. Informative prior favoring heads: Beta(8, 2) - Strongly believing the coin is biased towards heads")
print("3. Informative prior favoring tails: Beta(2, 8) - Strongly believing the coin is biased towards tails")
print()

# Calculate posteriors for different priors
priors = [
    {"alpha": 1, "beta": 1, "label": "Uniform Prior: Beta(1, 1)", "color": "blue"},
    {"alpha": 2, "beta": 2, "label": "Original Prior: Beta(2, 2)", "color": "green"},
    {"alpha": 8, "beta": 2, "label": "Heads-favoring Prior: Beta(8, 2)", "color": "orange"},
    {"alpha": 2, "beta": 8, "label": "Tails-favoring Prior: Beta(2, 8)", "color": "purple"}
]

plt.figure(figsize=(12, 10))
gs = GridSpec(2, 2, figure=plt.gcf())

# Plot all priors
ax1 = plt.subplot(gs[0, 0])
for prior_info in priors:
    prior_alpha, prior_beta = prior_info["alpha"], prior_info["beta"]
    prior_pdf = beta.pdf(theta, prior_alpha, prior_beta)
    ax1.plot(theta, prior_pdf, color=prior_info["color"], linewidth=2, label=prior_info["label"])

ax1.set_xlabel('θ (Probability of Heads)', fontsize=10)
ax1.set_ylabel('Density', fontsize=10)
ax1.set_title('Different Prior Distributions', fontsize=12)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Plot all posteriors
ax2 = plt.subplot(gs[0, 1])
for prior_info in priors:
    prior_alpha, prior_beta = prior_info["alpha"], prior_info["beta"]
    post_alpha, post_beta = prior_alpha + k, prior_beta + n - k
    post_pdf = beta.pdf(theta, post_alpha, post_beta)
    post_mean = post_alpha / (post_alpha + post_beta)
    ax2.plot(theta, post_pdf, color=prior_info["color"], linewidth=2, 
             label=f'Posterior from {prior_info["label"]} - Mean: {post_mean:.4f}')

ax2.set_xlabel('θ (Probability of Heads)', fontsize=10)
ax2.set_ylabel('Density', fontsize=10)
ax2.set_title('Resulting Posterior Distributions', fontsize=12)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Plot the likelihood
ax3 = plt.subplot(gs[1, 0])
ax3.plot(theta, likelihood_normalized, 'r-', linewidth=2, label='Likelihood for 7 heads in 10 tosses')
ax3.fill_between(theta, 0, likelihood_normalized, alpha=0.2, color='red')
ax3.set_xlabel('θ (Probability of Heads)', fontsize=10)
ax3.set_ylabel('Likelihood (scaled)', fontsize=10)
ax3.set_title('Likelihood Function (Data Evidence)', fontsize=12)
ax3.axvline(x=k/n, color='black', linestyle='--', label=f'MLE = {k/n}')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Plot prior and posterior means
ax4 = plt.subplot(gs[1, 1])
prior_means = [p["alpha"]/(p["alpha"]+p["beta"]) for p in priors]
posterior_means = [(p["alpha"]+k)/((p["alpha"]+k)+(p["beta"]+n-k)) for p in priors]
prior_labels = [p["label"].split(":")[0] for p in priors]
colors = [p["color"] for p in priors]

x = np.arange(len(priors))
width = 0.35

ax4.bar(x - width/2, prior_means, width, label='Prior Mean', color=colors, alpha=0.5)
ax4.bar(x + width/2, posterior_means, width, label='Posterior Mean', color=colors)
ax4.axhline(y=k/n, color='red', linestyle='--', label=f'MLE = {k/n}')

ax4.set_xlabel('Prior Choice', fontsize=10)
ax4.set_ylabel('Mean Value of θ', fontsize=10)
ax4.set_title('Prior vs Posterior Means', fontsize=12)
ax4.set_xticks(x)
ax4.set_xticklabels(prior_labels, rotation=45, ha='right')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "different_priors.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 8: Conclusion
print_step_header(8, "Conclusion")

print("Summary of Bayesian inference for coin flip problem:")
print(f"1. Likelihood function: L(θ | data) ∝ θ^{k} * (1-θ)^{n-k}")
print(f"2. Prior distribution: Beta(2, 2)")
print(f"3. Posterior distribution: Beta(9, 5)")
print(f"4. Posterior mean: {posterior_mean:.4f}")
print(f"5. Posterior mode: {posterior_mode:.4f}")
print(f"6. Posterior variance: {posterior_var:.4f}")
print(f"7. 95% credible interval: [{lower_bound:.4f}, {upper_bound:.4f}]")
print()
print("Key Insights:")
print("- The posterior mean (0.6429) is between the prior mean (0.5) and the MLE (0.7)")
print("- The effect of the prior diminishes as we collect more data")
print("- With conjugate priors, posterior calculations are straightforward")
print("- The 95% credible interval gives us a range of plausible values for θ")
print("- Different priors lead to different posteriors, but they converge as data increases") 