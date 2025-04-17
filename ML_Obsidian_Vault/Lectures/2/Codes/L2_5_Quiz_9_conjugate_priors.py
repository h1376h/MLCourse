import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from scipy.stats import beta, norm, gamma, poisson, bernoulli, binom

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_5_Quiz_9")
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

print("Task: For each of the following likelihoods, identify the corresponding conjugate prior:")
print("1. Bernoulli likelihood (for binary outcomes)")
print("2. Normal likelihood with known variance (for the mean parameter)")
print("3. Poisson likelihood (for the rate parameter)")
print()

# Step 2: Conjugate Prior for Bernoulli Likelihood
print_step_header(2, "Conjugate Prior for Bernoulli Likelihood")

print("The Bernoulli probability mass function is:")
print("P(X = x | θ) = θ^x * (1-θ)^(1-x)")
print("where x ∈ {0, 1} and θ is the probability parameter.")
print()
print("For n independent Bernoulli trials, the likelihood function is:")
print("L(θ | data) ∝ θ^(∑x) * (1-θ)^(n-∑x)")
print()
print("This has the form θ^a * (1-θ)^b, which matches the kernel of a Beta distribution.")
print("Therefore, the conjugate prior for the Bernoulli likelihood is the Beta distribution.")
print()
print("The Beta PDF is:")
print("p(θ | α, β) = [θ^(α-1) * (1-θ)^(β-1)] / B(α, β)")
print("where B(α, β) is the Beta function that normalizes the distribution.")
print()

# Visualize the Beta prior and posterior for Bernoulli
alpha_prior = 2
beta_prior = 2
theta_range = np.linspace(0, 1, 1000)
prior_pdf = beta.pdf(theta_range, alpha_prior, beta_prior)

# Simulate some Bernoulli data
np.random.seed(42)
theta_true = 0.7  # True parameter
n_samples = 20
data = np.random.binomial(1, theta_true, n_samples)
successes = np.sum(data)
failures = n_samples - successes

# Calculate posterior
alpha_posterior = alpha_prior + successes
beta_posterior = beta_prior + failures
posterior_pdf = beta.pdf(theta_range, alpha_posterior, beta_posterior)

# Compute likelihood for plotting
likelihood = theta_range ** successes * (1 - theta_range) ** failures
likelihood = likelihood / np.max(likelihood) * np.max(posterior_pdf) * 0.8

plt.figure(figsize=(12, 6))
gs = GridSpec(1, 2, width_ratios=[1, 1.5])

# First subplot: prior and data
ax1 = plt.subplot(gs[0])
ax1.plot(theta_range, prior_pdf, 'b-', linewidth=2, label=f'Beta({alpha_prior}, {beta_prior}) Prior')
ax1.axvline(x=theta_true, color='r', linestyle='--', label=f'True θ = {theta_true}')

# Generate bars for Bernoulli data
offset = np.linspace(-0.3, 0.3, n_samples)
heights = [0.1] * n_samples
y_positions = [0 if d == 0 else 0.2 for d in data]
ax1.bar(offset, heights, bottom=y_positions, width=0.02, color=['r' if d == 1 else 'b' for d in data],
        label='Data: 1=success, 0=failure')

ax1.set_xlim(-0.4, 1.4)
ax1.set_ylim(0, 3)
ax1.set_title('Beta Prior and Bernoulli Data', fontsize=12)
ax1.set_xlabel('θ (Probability of Success)', fontsize=10)
ax1.set_ylabel('Probability Density', fontsize=10)
ax1.legend(loc='upper right')
ax1.grid(True)

# Second subplot: Bayesian updating
ax2 = plt.subplot(gs[1])
ax2.plot(theta_range, prior_pdf, 'b--', linewidth=2, label=f'Prior: Beta({alpha_prior}, {beta_prior})')
ax2.plot(theta_range, likelihood, 'g-.', linewidth=2, label=f'Likelihood (∝ θ^{successes}(1-θ)^{failures})')
ax2.plot(theta_range, posterior_pdf, 'r-', linewidth=3, label=f'Posterior: Beta({alpha_posterior}, {beta_posterior})')
ax2.axvline(x=theta_true, color='k', linestyle='--', label=f'True θ = {theta_true}')

ax2.set_title('Bayesian Updating for Bernoulli Parameter', fontsize=12)
ax2.set_xlabel('θ (Probability of Success)', fontsize=10)
ax2.set_ylabel('Probability Density', fontsize=10)
ax2.legend(loc='best')
ax2.grid(True)

plt.tight_layout()
file_path = os.path.join(save_dir, "bernoulli_beta.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")
print()

# Step 3: Conjugate Prior for Normal Likelihood (Known Variance)
print_step_header(3, "Conjugate Prior for Normal Likelihood (Known Variance)")

print("The Normal probability density function is:")
print("P(X = x | μ, σ²) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))")
print("where μ is the mean and σ² is the variance.")
print()
print("For n independent observations from Normal(μ, σ²) with known σ², the likelihood function is:")
print("L(μ | data, σ²) ∝ exp(-(∑(x_i-μ)²)/(2σ²))")
print("  = exp(-(nμ² - 2μ∑x_i + ∑x_i²)/(2σ²))")
print("  ∝ exp(-(nμ² - 2μ∑x_i)/(2σ²))")
print("  ∝ exp(-(n/(2σ²))*(μ - (∑x_i)/n)²)")
print()
print("This has the form exp(-(a/2)(μ-b)²), which matches the kernel of a Normal distribution.")
print("Therefore, the conjugate prior for the Normal likelihood (with known variance) is the Normal distribution.")
print()
print("The Normal PDF is:")
print("p(μ | μ₀, σ₀²) = (1/√(2πσ₀²)) * exp(-(μ-μ₀)²/(2σ₀²))")
print("where μ₀ is the prior mean and σ₀² is the prior variance.")
print()

# Visualize the Normal prior and posterior for Normal likelihood
mu_true = 5.0  # True parameter
sigma = 2.0  # Known variance
mu0_prior = 3.0  # Prior mean
sigma0_prior = 1.5  # Prior standard deviation

# Simulate Normal data
np.random.seed(42)
n_samples = 10
data = np.random.normal(mu_true, sigma, n_samples)
sample_mean = np.mean(data)

# Calculate posterior parameters
sigma0_squared = sigma0_prior ** 2
sigma_squared = sigma ** 2
posterior_precision = 1/sigma0_squared + n_samples/sigma_squared
posterior_variance = 1/posterior_precision
posterior_mean = (mu0_prior/sigma0_squared + n_samples*sample_mean/sigma_squared) / posterior_precision
posterior_std = np.sqrt(posterior_variance)

# Range for plotting
mu_range = np.linspace(mu0_prior - 4*sigma0_prior, mu_true + 4*sigma, 1000)
prior_pdf = norm.pdf(mu_range, mu0_prior, sigma0_prior)
posterior_pdf = norm.pdf(mu_range, posterior_mean, posterior_std)

# Compute likelihood for plotting
likelihood = np.exp(-(n_samples/(2*sigma_squared)) * (mu_range - sample_mean)**2)
likelihood = likelihood / np.max(likelihood) * np.max(posterior_pdf) * 0.8

plt.figure(figsize=(12, 6))
gs = GridSpec(1, 2, width_ratios=[1, 1.5])

# First subplot: prior and data
ax1 = plt.subplot(gs[0])
ax1.plot(mu_range, prior_pdf, 'b-', linewidth=2, label=f'Normal({mu0_prior}, {sigma0_prior}²) Prior')
ax1.axvline(x=mu_true, color='r', linestyle='--', label=f'True μ = {mu_true}')

# Plot the data points
y_positions = np.linspace(0, 0.05, n_samples)
ax1.scatter(data, y_positions, color='g', s=50, alpha=0.6, label='Data samples')
ax1.axvline(x=sample_mean, color='g', linestyle='-', label=f'Sample mean = {sample_mean:.2f}')

ax1.set_xlim(mu0_prior - 4*sigma0_prior, mu_true + 4*sigma)
ax1.set_ylim(0, np.max(prior_pdf)*1.2)
ax1.set_title('Normal Prior and Normal Data', fontsize=12)
ax1.set_xlabel('μ (Mean Parameter)', fontsize=10)
ax1.set_ylabel('Probability Density', fontsize=10)
ax1.legend(loc='upper right')
ax1.grid(True)

# Second subplot: Bayesian updating
ax2 = plt.subplot(gs[1])
ax2.plot(mu_range, prior_pdf, 'b--', linewidth=2, label=f'Prior: Normal({mu0_prior}, {sigma0_prior}²)')
ax2.plot(mu_range, likelihood, 'g-.', linewidth=2, label=f'Likelihood (∝ exp(-n(μ-x̄)²/(2σ²)))')
ax2.plot(mu_range, posterior_pdf, 'r-', linewidth=3, 
         label=f'Posterior: Normal({posterior_mean:.2f}, {posterior_std:.2f}²)')
ax2.axvline(x=mu_true, color='k', linestyle='--', label=f'True μ = {mu_true}')
ax2.axvline(x=sample_mean, color='g', linestyle='-', label=f'Sample mean = {sample_mean:.2f}')

ax2.set_title('Bayesian Updating for Normal Mean (Known Variance)', fontsize=12)
ax2.set_xlabel('μ (Mean Parameter)', fontsize=10)
ax2.set_ylabel('Probability Density', fontsize=10)
ax2.legend(loc='best')
ax2.grid(True)

plt.tight_layout()
file_path = os.path.join(save_dir, "normal_normal.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")
print()

# Step 4: Conjugate Prior for Poisson Likelihood
print_step_header(4, "Conjugate Prior for Poisson Likelihood")

print("The Poisson probability mass function is:")
print("P(X = k | λ) = (λ^k * e^(-λ)) / k!")
print("where k is a non-negative integer and λ is the rate parameter.")
print()
print("For n independent observations from Poisson(λ), the likelihood function is:")
print("L(λ | data) ∝ λ^(∑k_i) * e^(-nλ)")
print()
print("This has the form λ^a * e^(-bλ), which matches the kernel of a Gamma distribution.")
print("Therefore, the conjugate prior for the Poisson likelihood is the Gamma distribution.")
print()
print("The Gamma PDF is:")
print("p(λ | α, β) = (β^α / Γ(α)) * λ^(α-1) * e^(-βλ)")
print("where α and β are the shape and rate parameters, respectively.")
print()

# Visualize the Gamma prior and posterior for Poisson likelihood
lambda_true = 4.0  # True parameter
alpha_prior = 2.0  # Prior shape
beta_prior = 0.5   # Prior rate

# Simulate Poisson data
np.random.seed(42)
n_samples = 15
data = np.random.poisson(lambda_true, n_samples)
sum_data = np.sum(data)

# Calculate posterior parameters
alpha_posterior = alpha_prior + sum_data
beta_posterior = beta_prior + n_samples

# Range for plotting
lambda_range = np.linspace(0, lambda_true*2, 1000)
prior_pdf = gamma.pdf(lambda_range, alpha_prior, scale=1/beta_prior)
posterior_pdf = gamma.pdf(lambda_range, alpha_posterior, scale=1/beta_posterior)

# Compute likelihood for plotting
from scipy.special import factorial
def poisson_likelihood(lambda_val, data):
    return np.prod(np.power(lambda_val, data) * np.exp(-lambda_val) / factorial(data))

likelihood_values = np.array([poisson_likelihood(l, data) for l in lambda_range])
likelihood_values = likelihood_values / np.max(likelihood_values) * np.max(posterior_pdf) * 0.8

plt.figure(figsize=(12, 6))
gs = GridSpec(1, 2, width_ratios=[1, 1.5])

# First subplot: prior and data
ax1 = plt.subplot(gs[0])
ax1.plot(lambda_range, prior_pdf, 'b-', linewidth=2, label=f'Gamma({alpha_prior}, {beta_prior}) Prior')
ax1.axvline(x=lambda_true, color='r', linestyle='--', label=f'True λ = {lambda_true}')

# Create histogram of the Poisson data
ax1.hist(data, bins=range(0, int(np.max(data))+2), alpha=0.5, density=True, color='g', label='Data histogram')
sample_mean = np.mean(data)
ax1.axvline(x=sample_mean, color='g', linestyle='-', label=f'Sample mean = {sample_mean:.2f}')

ax1.set_xlim(0, lambda_true*2)
ax1.set_ylim(0, np.max(prior_pdf)*1.2)
ax1.set_title('Gamma Prior and Poisson Data', fontsize=12)
ax1.set_xlabel('λ (Rate Parameter)', fontsize=10)
ax1.set_ylabel('Probability Density', fontsize=10)
ax1.legend(loc='best')
ax1.grid(True)

# Second subplot: Bayesian updating
ax2 = plt.subplot(gs[1])
ax2.plot(lambda_range, prior_pdf, 'b--', linewidth=2, label=f'Prior: Gamma({alpha_prior}, {beta_prior})')
ax2.plot(lambda_range, likelihood_values, 'g-.', linewidth=2, label=f'Likelihood (∝ λ^{sum_data} * e^(-{n_samples}λ))')
ax2.plot(lambda_range, posterior_pdf, 'r-', linewidth=3, 
         label=f'Posterior: Gamma({alpha_posterior}, {beta_posterior})')
ax2.axvline(x=lambda_true, color='k', linestyle='--', label=f'True λ = {lambda_true}')
ax2.axvline(x=sample_mean, color='g', linestyle='-', label=f'Sample mean = {sample_mean:.2f}')

ax2.set_title('Bayesian Updating for Poisson Rate Parameter', fontsize=12)
ax2.set_xlabel('λ (Rate Parameter)', fontsize=10)
ax2.set_ylabel('Probability Density', fontsize=10)
ax2.legend(loc='best')
ax2.grid(True)

plt.tight_layout()
file_path = os.path.join(save_dir, "poisson_gamma.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")
print()

# Step 5: Summary of Conjugate Prior Relationships
print_step_header(5, "Summary of Conjugate Prior Relationships")

# Create a table summarizing conjugate relationships
plt.figure(figsize=(12, 8))
ax = plt.subplot(111)

# Hide axes
ax.axis('off')
ax.axis('tight')

# Create table data
data = [
    ["Likelihood", "Parameter", "Conjugate Prior", "Posterior Parameters"],
    ["Bernoulli(θ)", "θ (probability)", "Beta(α, β)", "Beta(α + ∑x, β + n - ∑x)"],
    ["Normal(μ, σ²)", "μ (mean)\nwith known σ²", "Normal(μ₀, σ₀²)", "Normal(μ', σ'²)\nwhere μ' and σ'² are weighted combinations"],
    ["Poisson(λ)", "λ (rate)", "Gamma(α, β)", "Gamma(α + ∑x, β + n)"]
]

# Create table
table = ax.table(cellText=data, colWidths=[0.2, 0.2, 0.25, 0.35], loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)

# Style the table
for i in range(len(data)):
    for j in range(len(data[0])):
        cell = table[i, j]
        if i == 0:  # Header row
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            if j == 0:  # Likelihood column
                cell.set_facecolor('#D9E1F2')
            elif j == 2:  # Conjugate prior column
                cell.set_facecolor('#E2EFDA')
            else:
                cell.set_facecolor('#F2F2F2')

plt.title("Conjugate Prior Relationships", fontsize=16, pad=20)
plt.tight_layout()

file_path = os.path.join(save_dir, "conjugate_summary.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")
print()

# Step 6: Conclusion
print_step_header(6, "Conclusion")

print("In summary, the conjugate priors for the given likelihoods are:")
print("1. Bernoulli likelihood: Beta distribution")
print("2. Normal likelihood with known variance: Normal distribution")
print("3. Poisson likelihood: Gamma distribution")
print()
print("Using conjugate priors provides analytical tractability in Bayesian inference,")
print("as the posterior distribution has the same functional form as the prior.")
print("This eliminates the need for numerical integration or sampling methods,")
print("making the updating process more efficient.") 