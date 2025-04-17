import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_5_Quiz_11")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Set up the problem
print_step_header(1, "Understanding the Soil Scientist's Problem")

print("Given:")
print("- A soil scientist is developing a Bayesian model for soil nutrient content")
print("- Nutrient concentration follows a normal distribution with unknown mean μ")
print("- Prior for μ: Normal with mean 25 ppm and variance 4")
print("- Collected 6 samples with measurements: {22, 27, 24, 23, 26, 25} ppm")
print("- Known measurement variance σ² = 9")
print()
print("We need to:")
print("1. Find the posterior distribution for μ")
print("2. Calculate the posterior mean and variance")
print("3. Compare how the posterior would differ with an uninformative prior")
print()

# Define the observed data
data = np.array([22, 27, 24, 23, 26, 25])
n = len(data)
sample_mean = np.mean(data)
sample_variance = np.var(data, ddof=1)  # Sample variance with Bessel's correction

print(f"Sample data: {data.tolist()}")
print(f"Sample size: n = {n}")
print(f"Sample mean: x̄ = {sample_mean:.4f}")
print(f"Sample variance: s² = {sample_variance:.4f}")
print()

# Step 2: Normal-Normal conjugate prior relationship
print_step_header(2, "Applying Normal-Normal Conjugate Prior Relationship")

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
prior_mean = 25.0
prior_variance = 4.0
known_variance = 9.0

# Plot the normal distribution for our data
x = np.linspace(15, 35, 1000)

plt.figure(figsize=(10, 6))

# Plot the normal distribution with the sample mean and known variance
likelihood_mean = sample_mean
likelihood_variance = known_variance / n  # Variance of the sample mean
likelihood_pdf = stats.norm.pdf(x, loc=likelihood_mean, scale=np.sqrt(likelihood_variance))
plt.plot(x, likelihood_pdf, 'g-', linewidth=2, 
         label=f'Likelihood: N({likelihood_mean:.2f}, {likelihood_variance:.2f})')

# Plot the prior normal distribution
prior_pdf = stats.norm.pdf(x, loc=prior_mean, scale=np.sqrt(prior_variance))
plt.plot(x, prior_pdf, 'b-', linewidth=2, 
         label=f'Prior: N({prior_mean:.2f}, {prior_variance:.2f})')

plt.title('Prior Distribution and Likelihood Function', fontsize=14)
plt.xlabel('μ (Mean Nutrient Concentration in ppm)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)

# Save the figure
file_path = os.path.join(save_dir, "prior_likelihood.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 3: Calculate the posterior distribution
print_step_header(3, "Calculating the Posterior Distribution")

# Calculate posterior parameters
posterior_precision = 1/prior_variance + n/known_variance
posterior_variance = 1 / posterior_precision
posterior_mean = posterior_variance * (prior_mean/prior_variance + n*sample_mean/known_variance)

# Calculate the 95% credible interval
posterior_std = np.sqrt(posterior_variance)
credible_interval = stats.norm.ppf([0.025, 0.975], loc=posterior_mean, scale=posterior_std)

print(f"Prior Distribution:")
print(f"- Mean: μ₀ = {prior_mean:.4f} ppm")
print(f"- Variance: σ₀² = {prior_variance:.4f}")
print(f"- Standard Deviation: σ₀ = {np.sqrt(prior_variance):.4f}")
print()

print(f"Likelihood (based on data):")
print(f"- Sample Mean: x̄ = {sample_mean:.4f} ppm")
print(f"- Known Variance (per measurement): σ² = {known_variance:.4f}")
print(f"- Variance of Sample Mean: σ²/n = {known_variance/n:.4f}")
print()

print(f"Posterior Distribution:")
print(f"- Mean: μₙ = {posterior_mean:.4f} ppm")
print(f"- Variance: σₙ² = {posterior_variance:.4f}")
print(f"- Standard Deviation: σₙ = {posterior_std:.4f}")
print(f"- 95% Credible Interval: [{credible_interval[0]:.4f}, {credible_interval[1]:.4f}] ppm")
print()

# Step 4: Visualize prior, likelihood, and posterior
plt.figure(figsize=(10, 6))

# Plot the prior normal distribution
plt.plot(x, prior_pdf, 'b-', linewidth=2, 
         label=f'Prior: N({prior_mean:.2f}, {prior_variance:.2f})')

# Plot the likelihood
plt.plot(x, likelihood_pdf, 'g-', linewidth=2, 
         label=f'Likelihood: N({likelihood_mean:.2f}, {likelihood_variance:.2f})')

# Plot the posterior normal distribution
posterior_pdf = stats.norm.pdf(x, loc=posterior_mean, scale=posterior_std)
plt.plot(x, posterior_pdf, 'r-', linewidth=2, 
         label=f'Posterior: N({posterior_mean:.2f}, {posterior_variance:.2f})')

# Shade the 95% credible interval
plt.fill_between(x, 0, posterior_pdf, where=(x >= credible_interval[0]) & (x <= credible_interval[1]), 
                 color='red', alpha=0.3, label=f'95% Credible Interval: [{credible_interval[0]:.2f}, {credible_interval[1]:.2f}]')

plt.title('Prior, Likelihood, and Posterior Distributions', fontsize=14)
plt.xlabel('μ (Mean Nutrient Concentration in ppm)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)

# Save the figure
file_path = os.path.join(save_dir, "posterior_distribution.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 5: Interpret the posterior as a weighted average
print_step_header(5, "Interpreting the Posterior Mean as a Weighted Average")

# Calculate weights for weighted average interpretation
prior_weight = (1/prior_variance) / (1/prior_variance + n/known_variance)
data_weight = (n/known_variance) / (1/prior_variance + n/known_variance)

print("The posterior mean can be interpreted as a precision-weighted average:")
print(f"μₙ = w₁·μ₀ + w₂·x̄")
print()
print(f"Weight of prior (w₁): {prior_weight:.4f} ({prior_weight*100:.1f}%)")
print(f"Weight of data (w₂): {data_weight:.4f} ({data_weight*100:.1f}%)")
print()
print(f"μₙ = {prior_weight:.4f} × {prior_mean:.2f} + {data_weight:.4f} × {sample_mean:.2f}")
print(f"μₙ = {prior_weight*prior_mean:.4f} + {data_weight*sample_mean:.4f}")
print(f"μₙ = {posterior_mean:.4f} ppm")
print()

# Visualize the weighted average interpretation
plt.figure(figsize=(10, 6))

# Create a bar plot to show the weighted average
weights = [prior_weight, data_weight]
means = [prior_mean, sample_mean]
contributions = [prior_weight*prior_mean, data_weight*sample_mean]
sources = ['Prior', 'Data']

# Plot the weighted components
plt.bar(sources, contributions, alpha=0.7, 
        color=['blue', 'green'], 
        label=f'Contributions: {contributions[0]:.2f} + {contributions[1]:.2f} = {sum(contributions):.2f}')

# Add weight percentages on top of the bars
for i, (weight, contrib) in enumerate(zip(weights, contributions)):
    plt.text(i, contrib + 0.1, f"{weight*100:.1f}%", ha='center', fontsize=12)

# Add mean values inside the bars
for i, (mean, contrib) in enumerate(zip(means, contributions)):
    plt.text(i, contrib/2, f"μ = {mean:.1f}", ha='center', color='white', fontsize=12, fontweight='bold')

plt.axhline(y=posterior_mean, color='red', linestyle='-', 
           label=f'Posterior Mean: {posterior_mean:.2f}')

plt.title('Posterior Mean as a Weighted Average', fontsize=14)
plt.ylabel('Contribution to Posterior Mean', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, axis='y', alpha=0.3)

# Save the figure
file_path = os.path.join(save_dir, "weighted_average.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 6: Compare with uninformative prior
print_step_header(6, "Comparing with Uninformative Prior")

# Define an uninformative (flat) prior with large variance
uninformative_prior_mean = 25.0  # Same center but doesn't matter much
uninformative_prior_variance = 1000.0  # Very large variance

# Calculate posterior parameters with uninformative prior
uninformative_posterior_precision = 1/uninformative_prior_variance + n/known_variance
uninformative_posterior_variance = 1 / uninformative_posterior_precision
uninformative_posterior_mean = uninformative_posterior_variance * (uninformative_prior_mean/uninformative_prior_variance + n*sample_mean/known_variance)

# Calculate the 95% credible interval with uninformative prior
uninformative_posterior_std = np.sqrt(uninformative_posterior_variance)
uninformative_credible_interval = stats.norm.ppf([0.025, 0.975], loc=uninformative_posterior_mean, scale=uninformative_posterior_std)

print("Comparing the posteriors with informative vs. uninformative priors:")
print()
print("Informative Prior: N(25, 4)")
print(f"- Posterior Mean: {posterior_mean:.4f} ppm")
print(f"- Posterior Variance: {posterior_variance:.4f}")
print(f"- Posterior Standard Deviation: {posterior_std:.4f}")
print(f"- 95% Credible Interval: [{credible_interval[0]:.4f}, {credible_interval[1]:.4f}] ppm")
print()
print("Uninformative Prior: N(25, 1000) [effectively flat]")
print(f"- Posterior Mean: {uninformative_posterior_mean:.4f} ppm")
print(f"- Posterior Variance: {uninformative_posterior_variance:.4f}")
print(f"- Posterior Standard Deviation: {uninformative_posterior_std:.4f}")
print(f"- 95% Credible Interval: [{uninformative_credible_interval[0]:.4f}, {uninformative_credible_interval[1]:.4f}] ppm")
print()

print("Note: With an uninformative prior, the posterior mean approaches the sample mean,")
print("and the posterior variance approaches the variance of the sample mean (σ²/n).")
print()

# Calculate MLE (maximum likelihood estimate)
mle_mean = sample_mean
mle_variance = known_variance / n

print(f"Maximum Likelihood Estimate (MLE):")
print(f"- Mean: μ_MLE = {mle_mean:.4f} ppm")
print(f"- Variance: {mle_variance:.4f}")
print(f"- Standard Deviation: {np.sqrt(mle_variance):.4f}")
print()

# Visualize comparison between informative and uninformative priors
plt.figure(figsize=(12, 6))

# Define a wider range for the uninformative prior
x_wide = np.linspace(15, 35, 1000)

# Plot the informative prior
informative_prior_pdf = stats.norm.pdf(x_wide, loc=prior_mean, scale=np.sqrt(prior_variance))
plt.plot(x_wide, informative_prior_pdf, 'b-', linewidth=2, 
         label=f'Informative Prior: N({prior_mean:.1f}, {prior_variance:.1f})')

# Plot the uninformative prior (scaled down to be visible)
uninformative_prior_pdf = stats.norm.pdf(x_wide, loc=uninformative_prior_mean, scale=np.sqrt(uninformative_prior_variance))
uninformative_prior_pdf_scaled = uninformative_prior_pdf * (np.max(informative_prior_pdf) / np.max(uninformative_prior_pdf)) * 0.8
plt.plot(x_wide, uninformative_prior_pdf_scaled, 'c--', linewidth=2, 
         label=f'Uninformative Prior: N({uninformative_prior_mean:.1f}, {uninformative_prior_variance:.1f}) [scaled]')

# Plot the likelihood
plt.plot(x_wide, likelihood_pdf, 'g-', linewidth=2, 
         label=f'Likelihood: N({likelihood_mean:.2f}, {likelihood_variance:.2f})')

# Plot the posterior with informative prior
informative_posterior_pdf = stats.norm.pdf(x_wide, loc=posterior_mean, scale=posterior_std)
plt.plot(x_wide, informative_posterior_pdf, 'r-', linewidth=2, 
         label=f'Posterior (Informative): N({posterior_mean:.2f}, {posterior_variance:.2f})')

# Plot the posterior with uninformative prior
uninformative_posterior_pdf = stats.norm.pdf(x_wide, loc=uninformative_posterior_mean, scale=uninformative_posterior_std)
plt.plot(x_wide, uninformative_posterior_pdf, 'm--', linewidth=2, 
         label=f'Posterior (Uninformative): N({uninformative_posterior_mean:.2f}, {uninformative_posterior_variance:.2f})')

plt.title('Effect of Prior Information on Posterior Distribution', fontsize=14)
plt.xlabel('μ (Mean Nutrient Concentration in ppm)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)

# Save the figure
file_path = os.path.join(save_dir, "prior_comparison.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 7: Conclusion
print_step_header(7, "Conclusion")

print("The soil scientist's Bayesian analysis leads to the following conclusions:")
print()
print("1. The posterior distribution for the mean nutrient concentration is:")
print(f"   Normal({posterior_mean:.4f}, {posterior_variance:.4f})")
print()
print("2. With 95% credibility, the true mean nutrient concentration is between")
print(f"   {credible_interval[0]:.4f} and {credible_interval[1]:.4f} ppm.")
print()
print("3. When using an uninformative prior, the posterior mean shifts closer to the sample mean")
print(f"   ({uninformative_posterior_mean:.4f} ppm vs. {posterior_mean:.4f} ppm),")
print("   and the credible interval becomes wider, reflecting increased uncertainty.")
print()
print("4. The informative prior has pulled the estimate slightly toward 25 ppm")
print("   compared to the sample mean of 24.5 ppm, demonstrating how prior knowledge")
print("   influences Bayesian inference, especially with smaller sample sizes.") 