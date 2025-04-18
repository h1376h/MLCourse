import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from scipy.stats import gamma, poisson
from scipy.special import factorial

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_5_Quiz_7")
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
print("- We have a manufacturing process where defects occur according to a Poisson process")
print("- The parameter λ represents the average number of defects per batch")
print("- We want to model λ using Bayesian inference")
print("- Prior belief about λ is represented by Gamma(3, 2)")
print("- We observe defect counts in 5 batches: {1, 0, 2, 1, 1}")
print()
print("Questions:")
print("1. What is the conjugate prior for a Poisson likelihood?")
print("2. What is the resulting posterior distribution?")
print("3. What is the posterior mean of λ?")
print("4. What is the advantage of using a conjugate prior in this scenario?")
print()

# Step 2: Understanding Conjugate Priors for Poisson
print_step_header(2, "Conjugate Prior for Poisson Likelihood")

print("The Poisson probability mass function is:")
print("P(X = k | λ) = (λ^k * e^(-λ)) / k!")
print()
print("When we look at the form of this likelihood as a function of λ, we have:")
print("L(λ | data) ∝ λ^(∑k) * e^(-nλ)")
print()
print("This has the form λ^a * e^(-bλ), which matches the kernel of a Gamma distribution.")
print("Therefore, the conjugate prior for the Poisson likelihood is the Gamma distribution.")
print()
print("The Gamma PDF is:")
print("p(λ | α, β) = (β^α / Γ(α)) * λ^(α-1) * e^(-βλ)")
print()
print("When we combine this prior with the Poisson likelihood, the posterior will also be Gamma distributed.")
print()

# Visualize the Gamma prior
alpha_prior = 3
beta_prior = 2

lambda_range = np.linspace(0, 5, 1000)
prior_pdf = gamma.pdf(lambda_range, alpha_prior, scale=1/beta_prior)

plt.figure(figsize=(10, 6))
plt.plot(lambda_range, prior_pdf, 'b-', linewidth=3, label=f'Gamma({alpha_prior}, {beta_prior}) Prior')
plt.title('Prior Distribution for λ - Gamma(3, 2)', fontsize=14)
plt.xlabel('λ (Rate Parameter)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "prior_distribution.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 3: Bayesian Updating with Poisson-Gamma Model
print_step_header(3, "Bayesian Updating with Poisson-Gamma Model")

print("For a Poisson-Gamma model, the Bayesian updating follows these rules:")
print("If the prior is Gamma(α, β) and we observe x₁, x₂, ..., xₙ from Poisson(λ),")
print("then the posterior is Gamma(α + ∑xᵢ, β + n)")
print()
print("Let's apply this to our problem:")
print("- Prior: Gamma(α=3, β=2)")
print("- Data: {1, 0, 2, 1, 1}")
print("- Sum of observations: ∑xᵢ = 1 + 0 + 2 + 1 + 1 = 5")
print("- Number of observations: n = 5")
print()
print("Therefore, the posterior distribution is:")
print("Gamma(α + ∑xᵢ, β + n) = Gamma(3 + 5, 2 + 5) = Gamma(8, 7)")
print()

# Calculate posterior parameters
data = np.array([1, 0, 2, 1, 1])
n = len(data)
sum_data = np.sum(data)

alpha_posterior = alpha_prior + sum_data
beta_posterior = beta_prior + n

print(f"Posterior parameters: α' = {alpha_posterior}, β' = {beta_posterior}")
print()

# Visualize prior, likelihood, and posterior
lambda_range = np.linspace(0, 3, 1000)
prior_pdf = gamma.pdf(lambda_range, alpha_prior, scale=1/beta_prior)
posterior_pdf = gamma.pdf(lambda_range, alpha_posterior, scale=1/beta_posterior)

# Create a function for the unnormalized likelihood
def poisson_likelihood(lambda_val, data):
    return np.prod(np.power(lambda_val, data) * np.exp(-lambda_val) / factorial(data))

# Compute likelihood for plotting
likelihood_values = np.array([poisson_likelihood(l, data) for l in lambda_range])
# Normalize for better visualization
likelihood_values = likelihood_values / np.max(likelihood_values) * np.max(posterior_pdf)

plt.figure(figsize=(10, 6))
plt.plot(lambda_range, prior_pdf, 'b--', linewidth=2, label=f'Prior: Gamma({alpha_prior}, {beta_prior})')
plt.plot(lambda_range, likelihood_values, 'g-.', linewidth=2, label='Likelihood (scaled)')
plt.plot(lambda_range, posterior_pdf, 'r-', linewidth=3, label=f'Posterior: Gamma({alpha_posterior}, {beta_posterior})')

# Add markers for important values
prior_mean = alpha_prior / beta_prior
posterior_mean = alpha_posterior / beta_posterior
mle = np.mean(data)  # Maximum likelihood estimate

plt.axvline(x=prior_mean, color='b', linestyle=':', label=f'Prior Mean: {prior_mean:.2f}')
plt.axvline(x=posterior_mean, color='r', linestyle=':', label=f'Posterior Mean: {posterior_mean:.2f}')
plt.axvline(x=mle, color='g', linestyle=':', label=f'MLE: {mle:.2f}')

plt.title('Bayesian Updating for Poisson Parameter λ', fontsize=14)
plt.xlabel('λ (Rate Parameter)', fontsize=12)
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

posterior_mean = alpha_posterior / beta_posterior
posterior_mode = (alpha_posterior - 1) / beta_posterior if alpha_posterior > 1 else 0
posterior_variance = alpha_posterior / (beta_posterior ** 2)
posterior_std = np.sqrt(posterior_variance)

print(f"Posterior Distribution: Gamma({alpha_posterior}, {beta_posterior})")
print(f"Posterior Mean: E[λ|data] = α'/β' = {alpha_posterior}/{beta_posterior} = {posterior_mean:.4f}")
print(f"Posterior Mode: (α'-1)/β' = {posterior_mode:.4f}")
print(f"Posterior Variance: α'/(β')² = {posterior_variance:.4f}")
print(f"Posterior Standard Deviation: {posterior_std:.4f}")
print()

# Calculate 95% credible interval
lower_ci = gamma.ppf(0.025, alpha_posterior, scale=1/beta_posterior)
upper_ci = gamma.ppf(0.975, alpha_posterior, scale=1/beta_posterior)
print(f"95% Credible Interval: [{lower_ci:.4f}, {upper_ci:.4f}]")
print()

# Visualize posterior with credible interval
lambda_range = np.linspace(0, 3, 1000)
posterior_pdf = gamma.pdf(lambda_range, alpha_posterior, scale=1/beta_posterior)

plt.figure(figsize=(10, 6))
plt.plot(lambda_range, posterior_pdf, 'r-', linewidth=3, label=f'Posterior: Gamma({alpha_posterior}, {beta_posterior})')

# Shade the 95% credible interval
ci_x = np.linspace(lower_ci, upper_ci, 1000)
ci_y = gamma.pdf(ci_x, alpha_posterior, scale=1/beta_posterior)
plt.fill_between(ci_x, ci_y, alpha=0.3, color='red', label='95% Credible Interval')

# Add markers for important values
plt.axvline(x=posterior_mean, color='r', linestyle=':', label=f'Posterior Mean: {posterior_mean:.2f}')
plt.axvline(x=posterior_mode, color='b', linestyle=':', label=f'Posterior Mode: {posterior_mode:.2f}')

plt.title('Posterior Distribution with 95% Credible Interval', fontsize=14)
plt.xlabel('λ (Rate Parameter)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.grid(True)
plt.legend(loc='best')
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "posterior_credible_interval.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 5: Compare different prior strengths
print_step_header(5, "Effect of Prior Strength")

# Define different priors with varying strengths but same mean
prior_pairs = [
    (0.3, 0.2, "Weak prior - Gamma(0.3, 0.2)"),
    (3, 2, "Original prior - Gamma(3, 2)"),
    (30, 20, "Strong prior - Gamma(30, 20)")
]

plt.figure(figsize=(12, 8))

for alpha, beta, label in prior_pairs:
    # Calculate posterior with this prior
    alpha_post = alpha + sum_data
    beta_post = beta + n
    post_mean = alpha_post / beta_post
    
    # Plot the posterior
    lambda_range = np.linspace(0, 3, 1000)
    posterior_pdf = gamma.pdf(lambda_range, alpha_post, scale=1/beta_post)
    plt.plot(lambda_range, posterior_pdf, linewidth=2, label=f'{label} → Posterior Mean: {post_mean:.2f}')

# Add MLE for comparison
plt.axvline(x=mle, color='k', linestyle='--', label=f'MLE: {mle:.2f}')

plt.title('Effect of Prior Strength on Posterior Distribution', fontsize=14)
plt.xlabel('λ (Rate Parameter)', fontsize=12)
plt.ylabel('Posterior Probability Density', fontsize=12)
plt.grid(True)
plt.legend(loc='best')
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "prior_strength_effect.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 6: Predictive distribution
print_step_header(6, "Posterior Predictive Distribution")

print("The posterior predictive distribution tells us the probability of observing a certain")
print("number of defects in the next batch, after updating our belief about λ.")
print()
print("For the Poisson-Gamma model, the posterior predictive follows a negative binomial distribution:")
print("P(X = k | data) = NegBin(k | r=α', p=β'/(β'+1))")
print("Where α' and β' are the posterior parameters.")
print()

# Calculate parameters for the negative binomial predictive distribution
r = alpha_posterior
p = beta_posterior / (beta_posterior + 1)

print(f"Posterior Predictive Distribution: NegBin(r={r}, p={p:.4f})")
print()

# Calculate and plot the predictive PMF manually
def neg_bin_pmf(k, r, p):
    from scipy.special import comb
    return comb(k+r-1, k) * (p**r) * ((1-p)**k)

k_range = np.arange(0, 11)
predictive_pmf = np.array([neg_bin_pmf(k, r, p) for k in k_range])

plt.figure(figsize=(10, 6))
plt.bar(k_range, predictive_pmf, alpha=0.7, label='Posterior Predictive PMF')

# Add expected value marker
expected_value = r * (1-p) / p
plt.axvline(x=expected_value, color='r', linestyle='--', 
            label=f'Expected Value: {expected_value:.2f}')

plt.title('Posterior Predictive Distribution for Number of Defects in Next Batch', fontsize=14)
plt.xlabel('Number of Defects (k)', fontsize=12)
plt.ylabel('Probability Mass', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(k_range)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "posterior_predictive.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 7: Advantages of Conjugate Priors
print_step_header(7, "Advantages of Conjugate Priors")

print("Advantages of using a conjugate prior in this scenario:")
print("1. Analytical Tractability: The posterior has a closed-form solution (Gamma distribution)")
print("2. Computational Efficiency: No need for numerical integration or MCMC methods")
print("3. Interpretability: Prior and posterior parameters have intuitive meanings:")
print("   - α can be interpreted as 'equivalent prior defects'")
print("   - β can be interpreted as 'equivalent prior batches'")
print("4. Sequential Updating: Easy to update as new data arrives - just update the parameters")
print("5. Predictive Distribution: Also has a closed-form solution (Negative Binomial)")
print()

# Step 8: Conclusion and Answer
print_step_header(8, "Conclusion")

print("In summary:")
print("1. The conjugate prior for a Poisson likelihood is the Gamma distribution")
print("2. The posterior distribution is Gamma(8, 7)")
print("3. The posterior mean of λ is 8/7 ≈ 1.143")
print("4. The advantage of using a conjugate prior in this scenario is analytical tractability,")
print("   computational efficiency, interpretability, ease of sequential updating, and")
print("   having a closed-form posterior predictive distribution.")
print()
print("This analysis provides a complete Bayesian framework for modeling the number of defects")
print("in the manufacturing process, allowing us to make probabilistic predictions about")
print("future defect counts while accounting for uncertainty in our estimates.") 