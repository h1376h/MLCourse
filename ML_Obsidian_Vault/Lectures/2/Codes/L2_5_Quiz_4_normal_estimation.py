import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from scipy.stats import norm, t

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_5_Quiz_4")
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
print("- We want to estimate the mean μ of a normal distribution")
print("- The variance σ² = 4 is known")
print("- We observe data X = {10.2, 8.7, 9.5, 11.3, 10.8}")
print("- We use a normal prior N(9, 1) for μ")
print()
print("We need to:")
print("1. Derive the posterior distribution")
print("2. Calculate the posterior mean and variance")
print("3. Find the 90% credible interval for μ")
print("4. Compare Bayesian estimate with the maximum likelihood estimate (MLE)")
print()

# Define the observed data
data = np.array([10.2, 8.7, 9.5, 11.3, 10.8])
n = len(data)
sample_mean = np.mean(data)
sample_variance = np.var(data, ddof=1)  # Sample variance with Bessel's correction

print(f"Sample size: n = {n}")
print(f"Sample mean: x̄ = {sample_mean:.4f}")
print(f"Sample variance: s² = {sample_variance:.4f}")
print()

# Step 2: Normal-Normal conjugate prior relationship
print_step_header(2, "Normal-Normal Conjugate Prior Relationship")

print("For a normal likelihood with known variance σ²:")
print("- Likelihood: p(x|μ) = N(x|μ, σ²)")
print("- Prior: p(μ) = N(μ|μ₀, σ₀²)")
print()
print("When we observe data X = {x₁, x₂, ..., xₙ} from a N(μ, σ²) distribution:")
print("- The posterior is: p(μ|X) ∝ p(X|μ) * p(μ)")
print("- This gives us: p(μ|X) = N(μ|μₙ, σₙ²)")
print("  where:")
print("  μₙ = (μ₀/σ₀² + n*x̄/σ²) / (1/σ₀² + n/σ²)")
print("  σₙ² = 1 / (1/σ₀² + n/σ²)")
print()

# Define the prior and likelihood parameters
prior_mean = 9.0
prior_variance = 1.0
known_variance = 4.0

# Plot the normal distribution for our data
x = np.linspace(6, 14, 1000)

plt.figure(figsize=(10, 6))

# Plot the normal distribution with the sample mean and known variance
sample_pdf = norm.pdf(x, loc=sample_mean, scale=np.sqrt(known_variance))
plt.plot(x, sample_pdf, 'g-', linewidth=2, 
         label=f'Likelihood (based on data): N({sample_mean:.2f}, {known_variance})')

# Plot the prior distribution
prior_pdf = norm.pdf(x, loc=prior_mean, scale=np.sqrt(prior_variance))
plt.plot(x, prior_pdf, 'b--', linewidth=2, 
         label=f'Prior: N({prior_mean}, {prior_variance})')

# Plot the data points
for value in data:
    plt.axvline(x=value, color='g', linestyle=':', alpha=0.3)

plt.title('Normal Distributions - Prior and Likelihood', fontsize=14)
plt.xlabel('μ (Mean)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "prior_likelihood.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 3: Derive the posterior distribution
print_step_header(3, "Deriving the Posterior Distribution")

print("Starting with the prior p(μ) = N(μ|μ₀, σ₀²) and the normal likelihood:")
print("p(μ|X) ∝ p(X|μ) * p(μ)")
print()
print("For n independent observations from a normal distribution with known variance:")
print("p(X|μ) = ∏ p(xᵢ|μ) = ∏ [(2πσ²)^(-1/2) * exp(-(xᵢ-μ)²/(2σ²))]")
print("       = (2πσ²)^(-n/2) * exp[-Σ(xᵢ-μ)²/(2σ²)]")
print("       = (2πσ²)^(-n/2) * exp[-(n(x̄-μ)² + Σ(xᵢ-x̄)²)/(2σ²)]")
print()
print("The prior is:")
print("p(μ) = (2πσ₀²)^(-1/2) * exp[-(μ-μ₀)²/(2σ₀²)]")
print()
print("Multiplying these and focusing on terms involving μ:")
print("p(μ|X) ∝ exp[-(n(x̄-μ)²)/(2σ²)] * exp[-(μ-μ₀)²/(2σ₀²)]")
print("     ∝ exp[-((n/σ²)(x̄-μ)² + (1/σ₀²)(μ-μ₀)²)/2]")
print()
print("Completing the square and rearranging:")
print("p(μ|X) = N(μ|μₙ, σₙ²)")
print("where:")
print("  σₙ² = 1 / (1/σ₀² + n/σ²)")
print("  μₙ = σₙ² * (μ₀/σ₀² + n*x̄/σ²)")
print()

# Step 4: Calculate posterior with specific prior
print_step_header(4, "Calculating the Posterior Distribution")

# Calculate posterior parameters
precision_0 = 1.0 / prior_variance  # Prior precision
precision_data = n / known_variance  # Data precision
posterior_precision = precision_0 + precision_data  # Posterior precision
posterior_variance = 1.0 / posterior_precision  # Posterior variance

# Calculate posterior mean
posterior_mean = (prior_mean * precision_0 + sample_mean * precision_data) / posterior_precision

print(f"Prior: N(μ₀={prior_mean}, σ₀²={prior_variance})")
print(f"Data: X = {data}")
print(f"Sample mean: x̄ = {sample_mean:.4f}")
print(f"Known variance: σ² = {known_variance}")
print()
print(f"Posterior: N(μₙ={posterior_mean:.4f}, σₙ²={posterior_variance:.4f})")
print()

# Calculate 90% credible interval
alpha = 0.1  # 10% significance level for 90% credible interval
z_critical = norm.ppf(1 - alpha/2)  # z-value for 95% of probability mass
lower_ci = posterior_mean - z_critical * np.sqrt(posterior_variance)
upper_ci = posterior_mean + z_critical * np.sqrt(posterior_variance)

print(f"Posterior mean: E[μ|X] = {posterior_mean:.4f}")
print(f"Posterior variance: Var(μ|X) = {posterior_variance:.4f}")
print(f"Posterior standard deviation: {np.sqrt(posterior_variance):.4f}")
print(f"90% Credible interval: [{lower_ci:.4f}, {upper_ci:.4f}]")
print()

# Create visualization of prior, likelihood, and posterior
plt.figure(figsize=(12, 7))
x = np.linspace(6, 14, 1000)

# Prior
prior_pdf = norm.pdf(x, loc=prior_mean, scale=np.sqrt(prior_variance))
plt.plot(x, prior_pdf, 'b--', linewidth=2, label=f'Prior: N({prior_mean}, {prior_variance})')

# Likelihood (using sample mean)
likelihood_pdf = norm.pdf(x, loc=sample_mean, scale=np.sqrt(known_variance/n))
plt.plot(x, likelihood_pdf, 'g-', linewidth=2, 
         label=f'Likelihood (for mean): N({sample_mean:.2f}, {known_variance/n:.2f})')

# Posterior
posterior_pdf = norm.pdf(x, loc=posterior_mean, scale=np.sqrt(posterior_variance))
plt.plot(x, posterior_pdf, 'r-', linewidth=3, 
         label=f'Posterior: N({posterior_mean:.2f}, {posterior_variance:.2f})')

# Add vertical lines for the means
plt.axvline(x=prior_mean, color='b', linestyle=':', linewidth=1.5, label=f'Prior Mean: {prior_mean}')
plt.axvline(x=sample_mean, color='g', linestyle=':', linewidth=1.5, label=f'Sample Mean: {sample_mean:.2f}')
plt.axvline(x=posterior_mean, color='r', linestyle=':', linewidth=1.5, label=f'Posterior Mean: {posterior_mean:.2f}')

# Shade the 90% credible interval
mask = (x >= lower_ci) & (x <= upper_ci)
plt.fill_between(x[mask], 0, posterior_pdf[mask], color='red', alpha=0.2,
                label=f'90% Credible Interval: [{lower_ci:.2f}, {upper_ci:.2f}]')

plt.title('Prior, Likelihood, and Posterior Distributions for μ', fontsize=14)
plt.xlabel('μ (Mean Parameter)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "posterior_distribution.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 5: Alternative interpretation as weighted average
print_step_header(5, "Alternative Interpretation as Weighted Average")

# Calculate the weights
weight_prior = precision_0 / posterior_precision
weight_data = precision_data / posterior_precision

print("The posterior mean can be interpreted as a precision-weighted average:")
print(f"μₙ = {weight_prior:.4f} × μ₀ + {weight_data:.4f} × x̄")
print(f"   = {weight_prior:.4f} × {prior_mean} + {weight_data:.4f} × {sample_mean:.4f}")
print(f"   = {weight_prior * prior_mean:.4f} + {weight_data * sample_mean:.4f}")
print(f"   = {posterior_mean:.4f}")
print()
print(f"Prior weight: {weight_prior:.4f} ({weight_prior*100:.1f}%)")
print(f"Data weight: {weight_data:.4f} ({weight_data*100:.1f}%)")
print()

# Create visualization of weighted average
plt.figure(figsize=(10, 6))

# Define positions for the means
positions = [0, 1, 2]
means = [prior_mean, sample_mean, posterior_mean]
labels = ['Prior Mean', 'Sample Mean', 'Posterior Mean']
colors = ['blue', 'green', 'red']

plt.bar(positions, means, color=colors, alpha=0.6, width=0.6)

# Add text annotations for the weights
plt.text(positions[0], prior_mean/2, f"Weight: {weight_prior:.2f}", ha='center', va='center', color='white', fontsize=12, fontweight='bold')
plt.text(positions[1], sample_mean/2, f"Weight: {weight_data:.2f}", ha='center', va='center', color='white', fontsize=12, fontweight='bold')

# Add connecting arrows to show the weighted average
plt.annotate("", xy=(positions[2], posterior_mean), xytext=(positions[0], prior_mean),
            arrowprops=dict(arrowstyle="->", color='blue', alpha=0.5, lw=2))
plt.annotate("", xy=(positions[2], posterior_mean), xytext=(positions[1], sample_mean),
            arrowprops=dict(arrowstyle="->", color='green', alpha=0.5, lw=2))

plt.title('Posterior Mean as a Weighted Average', fontsize=14)
plt.xticks(positions, labels)
plt.ylabel('Value', fontsize=12)
plt.grid(True, axis='y')
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "weighted_average.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 6: Compare Bayesian estimate with MLE
print_step_header(6, "Comparing Bayesian Estimate with MLE")

# The MLE for normal mean is the sample mean
mle = sample_mean

# Calculate MLE confidence interval (frequentist)
t_critical = t.ppf(1 - alpha/2, n-1)  # t-value for n-1 degrees of freedom
mle_se = np.sqrt(known_variance / n)  # Standard error of the mean
mle_lower_ci = mle - t_critical * mle_se
mle_upper_ci = mle + t_critical * mle_se

print(f"Maximum Likelihood Estimate (MLE): μ̂ = x̄ = {mle:.4f}")
print(f"Standard error of MLE: SE(μ̂) = σ/√n = {mle_se:.4f}")
print(f"90% Confidence interval (frequentist): [{mle_lower_ci:.4f}, {mle_upper_ci:.4f}]")
print()
print(f"Bayesian posterior mean: {posterior_mean:.4f}")
print(f"Posterior standard deviation: {np.sqrt(posterior_variance):.4f}")
print(f"90% Credible interval (Bayesian): [{lower_ci:.4f}, {upper_ci:.4f}]")
print()

# Comparison
print("Comparing Bayesian and Frequentist approaches:")
print(f"1. Point estimate: The Bayesian posterior mean ({posterior_mean:.4f}) is a compromise between")
print(f"   the prior mean ({prior_mean}) and the MLE ({mle:.4f}), weighted by their respective precisions.")
print("2. The Bayesian credible interval has a direct probability interpretation:")
print("   'There is a 90% probability that μ lies within this interval, given our data and prior.'")
print("3. The frequentist confidence interval has a different interpretation:")
print("   'If we repeated the experiment many times, 90% of the computed intervals would contain the true μ.'")
print()

# Create visualization comparing MLE and Bayesian estimates
plt.figure(figsize=(12, 7))
x = np.linspace(6, 14, 1000)

# Likelihood for the mean (MLE)
likelihood_pdf = norm.pdf(x, loc=mle, scale=mle_se)
plt.plot(x, likelihood_pdf, 'g-', linewidth=2, 
         label=f'Likelihood (for mean): N({mle:.2f}, {mle_se**2:.4f})')

# Posterior
posterior_pdf = norm.pdf(x, loc=posterior_mean, scale=np.sqrt(posterior_variance))
plt.plot(x, posterior_pdf, 'r-', linewidth=2, 
         label=f'Posterior: N({posterior_mean:.2f}, {posterior_variance:.4f})')

# Shade the MLE confidence interval
plt.fill_between(x[(x >= mle_lower_ci) & (x <= mle_upper_ci)], 
                0, likelihood_pdf[(x >= mle_lower_ci) & (x <= mle_upper_ci)],
                color='green', alpha=0.2,
                label=f'90% Confidence Interval: [{mle_lower_ci:.2f}, {mle_upper_ci:.2f}]')

# Shade the Bayesian credible interval
plt.fill_between(x[(x >= lower_ci) & (x <= upper_ci)], 
                0, posterior_pdf[(x >= lower_ci) & (x <= upper_ci)],
                color='red', alpha=0.2,
                label=f'90% Credible Interval: [{lower_ci:.2f}, {upper_ci:.2f}]')

# Add vertical lines for the means
plt.axvline(x=mle, color='g', linestyle='-', linewidth=1, label=f'MLE: {mle:.2f}')
plt.axvline(x=posterior_mean, color='r', linestyle='-', linewidth=1, label=f'Posterior Mean: {posterior_mean:.2f}')

plt.title('Comparison of MLE and Bayesian Estimation', fontsize=14)
plt.xlabel('μ (Mean Parameter)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "mle_vs_bayesian.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 7: Effect of different priors
print_step_header(7, "Effect of Different Priors")

# Define different priors
prior_params = [
    (9, 0.1, "Strong, centered at 9"),    # Strong prior centered at 9
    (9, 1, "Original prior"),             # Original prior
    (9, 10, "Weak, centered at 9"),       # Weak prior centered at 9
    (11, 1, "Strong, centered at 11")     # Different prior mean
]

plt.figure(figsize=(12, 8))
x = np.linspace(6, 14, 1000)

# MLE for reference
likelihood_pdf = norm.pdf(x, loc=mle, scale=mle_se)
plt.plot(x, likelihood_pdf, 'g-', linewidth=3, alpha=0.5,
         label=f'Likelihood (for mean): N({mle:.2f}, {mle_se**2:.4f})')

# Plot posteriors for different priors
for i, (prior_mu, prior_var, label) in enumerate(prior_params):
    # Calculate posterior parameters
    prior_precision = 1/prior_var
    post_precision = prior_precision + n/known_variance
    post_var = 1/post_precision
    post_mean = (prior_mu * prior_precision + sample_mean * n/known_variance) / post_precision
    
    # Plot posterior
    posterior_pdf = norm.pdf(x, loc=post_mean, scale=np.sqrt(post_var))
    plt.plot(x, posterior_pdf, linewidth=2, 
             label=f'Prior: N({prior_mu}, {prior_var}) → Posterior Mean: {post_mean:.2f}')

plt.title('Effect of Different Priors on the Posterior Distribution', fontsize=14)
plt.xlabel('μ (Mean Parameter)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "prior_effect.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 8: Effect of increasing sample size
print_step_header(8, "Effect of Increasing Sample Size")

# Define different sample sizes to simulate
sample_sizes = [1, 5, 20, 100]

plt.figure(figsize=(12, 8))
x = np.linspace(7, 13, 1000)

# Original prior
prior_pdf = norm.pdf(x, loc=prior_mean, scale=np.sqrt(prior_variance))
plt.plot(x, prior_pdf, 'b--', linewidth=2, label=f'Prior: N({prior_mean}, {prior_variance})')

# Plot posteriors for different sample sizes
colors = ['purple', 'teal', 'orange', 'red']
for i, n_samples in enumerate(sample_sizes):
    # Calculate posterior parameters
    post_precision = precision_0 + n_samples/known_variance
    post_var = 1/post_precision
    post_mean = (prior_mean * precision_0 + sample_mean * n_samples/known_variance) / post_precision
    
    # Plot posterior
    posterior_pdf = norm.pdf(x, loc=post_mean, scale=np.sqrt(post_var))
    plt.plot(x, posterior_pdf, color=colors[i], linewidth=2, 
             label=f'n = {n_samples}, Posterior: N({post_mean:.2f}, {post_var:.4f})')

# Add vertical line for the MLE/sample mean
plt.axvline(x=sample_mean, color='g', linestyle='--', linewidth=1.5, 
           label=f'Sample Mean: {sample_mean:.2f}')

plt.title('Effect of Sample Size on Posterior Distribution', fontsize=14)
plt.xlabel('μ (Mean Parameter)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "sample_size_effect.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 9: Summary and conclusion
print_step_header(9, "Conclusion and Summary")

print("Summary of findings:")
print(f"1. Observed data: {data}")
print(f"2. Prior: N({prior_mean}, {prior_variance})")
print(f"3. Posterior: N({posterior_mean:.4f}, {posterior_variance:.4f})")
print(f"4. Posterior mean: {posterior_mean:.4f}")
print(f"5. Posterior variance: {posterior_variance:.4f}")
print(f"6. 90% Credible interval: [{lower_ci:.4f}, {upper_ci:.4f}]")
print(f"7. MLE (sample mean): {mle:.4f}")
print(f"8. 90% Confidence interval (frequentist): [{mle_lower_ci:.4f}, {mle_upper_ci:.4f}]")
print()
print("Key insights:")
print("1. The Normal distribution is conjugate to itself for the mean parameter with known variance")
print("2. The posterior mean is a precision-weighted average of the prior mean and the sample mean")
print("3. The posterior variance decreases with increasing sample size")
print("4. With more data, the posterior converges to the MLE, and the prior's influence diminishes")
print("5. Bayesian credible intervals have a direct probability interpretation, unlike frequentist confidence intervals")
print("6. The choice of prior can significantly impact inference with small sample sizes") 