import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from scipy.stats import gamma, poisson, nbinom

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_5_Quiz_3")
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
print("- We have a Poisson likelihood with parameter λ")
print("- We observe data X = {3, 5, 2, 4, 6, 3, 4, 5, 2, 3}")
print("- We use a Gamma(α, β) prior for λ")
print()
print("We need to:")
print("1. Derive the posterior distribution")
print("2. Calculate the posterior with Gamma(2, 1) prior")
print("3. Find posterior mean, mode, and variance")
print("4. Calculate the predictive distribution for a new observation")
print()

# Define the observed data
data = np.array([3, 5, 2, 4, 6, 3, 4, 5, 2, 3])
n = len(data)
sum_x = np.sum(data)

print(f"Sample size: n = {n}")
print(f"Sum of all observations: sum(x_i) = {sum_x}")
print(f"Sample mean: x̄ = {sum_x/n:.2f}")
print()

# Step 2: Poisson-Gamma conjugate prior relationship
print_step_header(2, "Poisson-Gamma Conjugate Prior Relationship")

print("For a Poisson likelihood with parameter λ:")
print("- Likelihood: p(x|λ) = e^(-λ) * λ^x / x!")
print("- Prior: p(λ) = Gamma(α, β) = β^α * λ^(α-1) * e^(-βλ) / Γ(α)")
print("  where α is the shape parameter and β is the rate parameter")
print()
print("When we observe data X = {x_1, x_2, ..., x_n} from a Poisson(λ) distribution:")
print("- The posterior is: p(λ|X) ∝ p(X|λ) * p(λ)")
print("- This gives us: p(λ|X) = Gamma(α + sum(x_i), β + n)")
print()

# Plot the Poisson distribution for different lambda values
lambda_values = [2, 3.7, 5]  # Different lambda values to demonstrate
x = np.arange(0, 15)

plt.figure(figsize=(10, 6))

for lambda_val in lambda_values:
    pmf = poisson.pmf(x, lambda_val)
    plt.plot(x, pmf, 'o-', linewidth=2, label=f'λ = {lambda_val}')
    
plt.title('Poisson Distributions with Different λ Values', fontsize=14)
plt.xlabel('x (Count)', fontsize=12)
plt.ylabel('Probability Mass', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "poisson_distributions.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 3: Derive the posterior distribution
print_step_header(3, "Deriving the Posterior Distribution")

print("Starting with the prior p(λ) = Gamma(α, β) and the Poisson likelihood:")
print("p(λ|X) ∝ p(X|λ) * p(λ)")
print("p(λ|X) ∝ [∏ p(x_i|λ)] * p(λ)")
print("p(λ|X) ∝ [∏ (e^(-λ) * λ^(x_i) / x_i!)] * [β^α * λ^(α-1) * e^(-βλ) / Γ(α)]")
print("p(λ|X) ∝ e^(-nλ) * λ^(sum(x_i)) * λ^(α-1) * e^(-βλ)")
print("p(λ|X) ∝ e^(-(n+β)λ) * λ^(α+sum(x_i)-1)")
print()
print("This is proportional to Gamma(α + sum(x_i), β + n)")
print()
print("Therefore, the posterior distribution is:")
print("p(λ|X) = Gamma(α + sum(x_i), β + n)")
print()

# Step 4: Calculate posterior with specific prior
print_step_header(4, "Calculating the Posterior with Gamma(2, 1) Prior")

# Prior parameters
alpha_prior = 2
beta_prior = 1

# Posterior parameters
alpha_post = alpha_prior + sum_x
beta_post = beta_prior + n

print(f"Prior: Gamma(α={alpha_prior}, β={beta_prior})")
print(f"Data: X = {data}")
print(f"Posterior: Gamma(α={alpha_post}, β={beta_post})")
print()

# Calculate relevant values from the posterior
post_mean = alpha_post / beta_post
post_mode = (alpha_post - 1) / beta_post if alpha_post > 1 else 0
post_var = alpha_post / (beta_post ** 2)
post_std = np.sqrt(post_var)

print(f"Posterior mean: E[λ|X] = α'/β' = {post_mean:.4f}")
print(f"Posterior mode: (α'-1)/β' = {post_mode:.4f}")
print(f"Posterior variance: α'/β'^2 = {post_var:.4f}")
print(f"Posterior standard deviation: {post_std:.4f}")
print()

# Create visualization of prior, likelihood, and posterior
plt.figure(figsize=(12, 7))
lambda_range = np.linspace(0, 8, 1000)

# Prior
prior_pdf = gamma.pdf(lambda_range, alpha_prior, scale=1/beta_prior)
plt.plot(lambda_range, prior_pdf, 'b--', linewidth=2, label=f'Prior: Gamma({alpha_prior}, {beta_prior})')

# Likelihood (using a kernel density estimate from many Poisson samples)
def poisson_likelihood(data, lambda_val):
    """Return the likelihood for a given lambda."""
    return np.prod([poisson.pmf(x, lambda_val) for x in data])

likelihood_values = np.array([poisson_likelihood(data, lam) for lam in lambda_range])
likelihood_values = likelihood_values / np.max(likelihood_values)  # Scale for better visualization
plt.plot(lambda_range, likelihood_values, 'g-', linewidth=2, 
         label=f'Scaled Likelihood (based on {n} observations)')

# Posterior
posterior_pdf = gamma.pdf(lambda_range, alpha_post, scale=1/beta_post)
plt.plot(lambda_range, posterior_pdf, 'r-', linewidth=3, label=f'Posterior: Gamma({alpha_post}, {beta_post})')

# Add vertical lines for the mean and mode
plt.axvline(x=post_mean, color='r', linestyle=':', linewidth=2, label=f'Posterior Mean: {post_mean:.2f}')
plt.axvline(x=post_mode, color='r', linestyle='--', linewidth=2, label=f'Posterior Mode: {post_mode:.2f}')

# Add a vertical line for the sample mean
plt.axvline(x=sum_x/n, color='g', linestyle=':', linewidth=2, label=f'Sample Mean: {sum_x/n:.2f}')

plt.title('Prior, Likelihood, and Posterior Distributions for λ', fontsize=14)
plt.xlabel('λ (Poisson Rate Parameter)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "prior_likelihood_posterior.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 5: Calculate the predictive distribution for a new observation
print_step_header(5, "Calculating the Predictive Distribution")

print("The predictive distribution for a new observation given data X is:")
print("p(x_new|X) = ∫ p(x_new|λ) p(λ|X) dλ")
print("For the Poisson-Gamma model, this is a Negative Binomial distribution:")
print("p(x_new|X) = NegBin(α', β'/(β'+1))")
print(f"p(x_new|X) = NegBin(r={alpha_post}, p={beta_post/(beta_post+1):.4f})")
print()

# Calculate parameters for the predictive distribution (negative binomial)
r = alpha_post  # shape parameter
p = beta_post / (beta_post + 1)  # success probability (in scipy parameterization)

print(f"The predictive distribution parameters are: r={r}, p={p:.4f}")
print(f"Mean of predictive distribution: {r * (1-p) / p:.4f}")
print(f"Variance of predictive distribution: {r * (1-p) / (p**2):.4f}")
print()

# Calculate probabilities for specific new observations
x_new_range = np.arange(0, 15)
pred_probs = nbinom.pmf(x_new_range, r, p)

print("Probabilities for specific new observations:")
for x_new, prob in zip(x_new_range[:10], pred_probs[:10]):
    print(f"P(x_new = {x_new}) = {prob:.4f}")
print()

# Plot the predictive distribution
plt.figure(figsize=(10, 6))
plt.bar(x_new_range, pred_probs, alpha=0.7, width=0.4, label='Predictive PMF')

# Add a line for the Poisson PMF with lambda = posterior mean
poisson_pmf = poisson.pmf(x_new_range, post_mean)
plt.plot(x_new_range, poisson_pmf, 'ro-', linewidth=2, 
         label=f'Poisson PMF with λ = {post_mean:.2f}')

plt.title('Predictive Distribution for a New Observation', fontsize=14)
plt.xlabel('x_new (New Count)', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "predictive_distribution.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 6: Comparison with different priors
print_step_header(6, "Effect of Different Priors")

# Define different priors
prior_params = [
    (1, 1, "Uninformative"),
    (2, 1, "Original Prior"),
    (10, 2, "More informative, E[λ]=5"),
    (20, 10, "Very informative, E[λ]=2")
]

plt.figure(figsize=(12, 8))

for alpha, beta, label in prior_params:
    # Calculate posterior parameters
    alpha_p = alpha + sum_x
    beta_p = beta + n
    post_mean = alpha_p / beta_p
    
    # Plot posterior
    posterior_pdf = gamma.pdf(lambda_range, alpha_p, scale=1/beta_p)
    plt.plot(lambda_range, posterior_pdf, linewidth=2, 
             label=f'Prior: Gamma({alpha}, {beta}) → Posterior Mean: {post_mean:.2f}')

plt.title('Effect of Different Priors on the Posterior Distribution', fontsize=14)
plt.xlabel('λ (Poisson Rate Parameter)', fontsize=12)
plt.ylabel('Posterior Density', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "prior_comparison.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 7: Compare posterior mean with MLE and show credible interval
print_step_header(7, "Posterior Mean vs MLE and Credible Interval")

# MLE for Poisson is just the sample mean
mle = sum_x / n

# Calculate 95% credible interval
lower_ci = gamma.ppf(0.025, alpha_post, scale=1/beta_post)
upper_ci = gamma.ppf(0.975, alpha_post, scale=1/beta_post)

print(f"Maximum Likelihood Estimate (MLE): {mle:.4f}")
print(f"Posterior Mean: {post_mean:.4f}")
print(f"95% Credible Interval: [{lower_ci:.4f}, {upper_ci:.4f}]")
print()

plt.figure(figsize=(10, 6))

# Plot posterior distribution
posterior_pdf = gamma.pdf(lambda_range, alpha_post, scale=1/beta_post)
plt.plot(lambda_range, posterior_pdf, 'b-', linewidth=2, label='Posterior Distribution')

# Shade the 95% credible interval
mask = (lambda_range >= lower_ci) & (lambda_range <= upper_ci)
plt.fill_between(lambda_range[mask], 0, posterior_pdf[mask], color='blue', alpha=0.3,
                 label=f'95% Credible Interval: [{lower_ci:.2f}, {upper_ci:.2f}]')

# Add vertical lines for the mean, MLE
plt.axvline(x=post_mean, color='r', linestyle='-', linewidth=2, label=f'Posterior Mean: {post_mean:.2f}')
plt.axvline(x=mle, color='g', linestyle='--', linewidth=2, label=f'MLE: {mle:.2f}')

plt.title('Posterior Distribution with 95% Credible Interval', fontsize=14)
plt.xlabel('λ (Poisson Rate Parameter)', fontsize=12)
plt.ylabel('Posterior Density', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "credible_interval.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 8: Conclusion and Summary
print_step_header(8, "Conclusion and Summary")

print("Summary of findings:")
print(f"1. Observed data: {data}")
print(f"2. Prior: Gamma({alpha_prior}, {beta_prior})")
print(f"3. Posterior: Gamma({alpha_post}, {beta_post})")
print(f"4. Posterior mean: {post_mean:.4f}")
print(f"5. Posterior mode: {post_mode:.4f}")
print(f"6. Posterior variance: {post_var:.4f}")
print(f"7. 95% Credible interval: [{lower_ci:.4f}, {upper_ci:.4f}]")
print(f"8. Predictive distribution: Negative Binomial(r={r}, p={p:.4f})")
print()
print("Key insights:")
print("1. The Gamma prior is conjugate to the Poisson likelihood, yielding a Gamma posterior")
print("2. The posterior effectively combines the prior knowledge with the observed data")
print("3. The predictive distribution accounts for both parameter uncertainty and Poisson randomness")
print("4. As sample size increases, the influence of the prior diminishes")
print("5. The Bayesian approach provides a natural way to quantify uncertainty through the posterior distribution") 