import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm, poisson
from scipy.special import factorial
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_4_Quiz_21")
os.makedirs(save_dir, exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')

# Set plot style parameters
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'text.usetex': False,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

# Print header for pretty output
def print_section_header(title):
    separator = "=" * 60
    print(f"\n{separator}")
    print(f"{title.center(60)}")
    print(f"{separator}\n")

print_section_header("SOLUTION: Visual Analysis of Maximum Likelihood Estimation")

# Load the data from the question script 
# (in a real solution, you'd use the actual data or regenerate with the same seed)
np.random.seed(42)  # For reproducibility - same as question
true_mu = 25
true_sigma = 5
n_obs = 50  # Sample size
normal_data = np.random.normal(true_mu, true_sigma, n_obs)

true_lambda = 3.5
poisson_data = np.random.poisson(true_lambda, n_obs)

# ========= NORMAL DISTRIBUTION ANALYSIS =========

print_section_header("PART 1: Normal Distribution MLE Analysis")

# Calculate MLE for normal
mle_mu = np.mean(normal_data)
print(f"Sample size: {n_obs}")
print(f"True mean (μ): {true_mu}")
print(f"MLE estimate of mean (μ̂): {mle_mu:.4f}")
print(f"Discrepancy: {abs(true_mu - mle_mu):.4f}")

# Calculate standard error of the mean
sem = true_sigma / np.sqrt(n_obs)
print(f"\nStandard Error of the Mean: {sem:.4f}")
print(f"95% Confidence Interval: ({mle_mu - 1.96*sem:.4f}, {mle_mu + 1.96*sem:.4f})")

print("\nTheoretical Results:")
print("1. For a normal distribution with known variance, the MLE of μ is the sample mean.")
print("2. The MLE is an unbiased estimator of the mean: E[μ̂] = μ")
print("3. Variance of the MLE: Var(μ̂) = σ²/n = {:.4f}".format(true_sigma**2/n_obs))
print("4. As n increases, the MLE converges to the true mean (consistency).")

# Visual explanation of why sample mean is MLE
# Create a simplified example to show the geometric interpretation
plt.figure(figsize=(12, 6))

# Theoretical demonstration with simple case
sample_means = np.linspace(20, 30, 200)
x_vals = np.linspace(10, 40, 1000)

# Compute log-likelihood for a fixed dataset and varying means
def compute_log_likelihood(data, sample_means, sigma):
    """Compute log-likelihood for various sample means"""
    log_likelihood = np.zeros_like(sample_means)
    n = len(data)
    for i, mean in enumerate(sample_means):
        log_likelihood[i] = -n/2 * np.log(2 * np.pi * sigma**2) - \
                           sum((data - mean)**2) / (2 * sigma**2)
    return log_likelihood

# Show how log-likelihood peaks at sample mean
log_likelihood = compute_log_likelihood(normal_data, sample_means, true_sigma)
max_idx = np.argmax(log_likelihood)

plt.plot(sample_means, log_likelihood, 'b-', linewidth=2, label='Log-Likelihood')
plt.axvline(sample_means[max_idx], color='r', linestyle='--', 
            label=f'MLE = Sample Mean = {sample_means[max_idx]:.4f}')
plt.axvline(true_mu, color='g', linestyle=':', 
            label=f'True Mean = {true_mu}')
plt.grid(True, alpha=0.3)
plt.xlabel('μ')
plt.ylabel('Log-Likelihood')
plt.title('Log-Likelihood Function for Normal Distribution')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'normal_mle_theoretical.png'), dpi=300, bbox_inches='tight')

# ========= POISSON DISTRIBUTION ANALYSIS =========

print_section_header("PART 2: Poisson Distribution MLE Analysis")

# Calculate MLE for Poisson
mle_lambda = np.mean(poisson_data)
print(f"Sample size: {n_obs}")
print(f"True lambda (λ): {true_lambda}")
print(f"MLE estimate of lambda (λ̂): {mle_lambda:.4f}")
print(f"Discrepancy: {abs(true_lambda - mle_lambda):.4f}")

# Calculate standard error for Poisson
poisson_sem = np.sqrt(mle_lambda / n_obs)
print(f"\nStandard Error of λ̂: {poisson_sem:.4f}")
print(f"95% Confidence Interval: ({mle_lambda - 1.96*poisson_sem:.4f}, {mle_lambda + 1.96*poisson_sem:.4f})")

print("\nTheoretical Results:")
print("1. For a Poisson distribution, the MLE of λ is the sample mean.")
print("2. The MLE is an unbiased estimator: E[λ̂] = λ")
print("3. Variance of the MLE: Var(λ̂) = λ/n = {:.4f}".format(true_lambda/n_obs))
print("4. The Poisson MLE has a narrower likelihood function than the normal MLE (for this data),")
print("   indicating greater precision in the estimate.")

# Visual explanation for Poisson MLE
plt.figure(figsize=(12, 6))

# Simplified Poisson log-likelihood
lambda_vals = np.linspace(2, 5, 200)

def compute_poisson_log_likelihood(data, lambda_vals):
    """Compute Poisson log-likelihood for various lambdas"""
    log_likelihood = np.zeros_like(lambda_vals)
    for i, lambda_val in enumerate(lambda_vals):
        log_likelihood[i] = sum(data) * np.log(lambda_val) - len(data) * lambda_val
    return log_likelihood

poisson_log_likelihood = compute_poisson_log_likelihood(poisson_data, lambda_vals)
poisson_max_idx = np.argmax(poisson_log_likelihood)

plt.plot(lambda_vals, poisson_log_likelihood, 'b-', linewidth=2, label='Log-Likelihood')
plt.axvline(lambda_vals[poisson_max_idx], color='r', linestyle='--', 
            label=f'MLE = Sample Mean = {lambda_vals[poisson_max_idx]:.4f}')
plt.axvline(true_lambda, color='g', linestyle=':', 
            label=f'True λ = {true_lambda}')
plt.grid(True, alpha=0.3)
plt.xlabel('λ')
plt.ylabel('Log-Likelihood (ignoring constant terms)')
plt.title('Log-Likelihood Function for Poisson Distribution')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'poisson_mle_theoretical.png'), dpi=300, bbox_inches='tight')

# ========= COMPARISON OF FISHER INFORMATION =========

print_section_header("PART 3: Comparison of Estimation Uncertainty")

# Show visual comparison of relative widths of likelihood functions
plt.figure(figsize=(14, 7))

# Normalize both log-likelihoods to have max=0 for comparison
norm_ll = log_likelihood - np.max(log_likelihood)
pois_ll = poisson_log_likelihood - np.max(poisson_log_likelihood)

# Create relative scales for x-axis
norm_x_rel = (sample_means - mle_mu) / mle_mu
pois_x_rel = (lambda_vals - mle_lambda) / mle_lambda

plt.subplot(121)
plt.plot(norm_x_rel, np.exp(norm_ll), 'b-', linewidth=2, label='Normal')
plt.axvline(0, color='r', linestyle='--')
plt.xlabel('Relative Difference from MLE (μ)')
plt.ylabel('Normalized Likelihood')
plt.title('Normal Distribution Likelihood')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(122)
plt.plot(pois_x_rel, np.exp(pois_ll), 'g-', linewidth=2, label='Poisson')
plt.axvline(0, color='r', linestyle='--')
plt.xlabel('Relative Difference from MLE (λ)')
plt.ylabel('Normalized Likelihood')
plt.title('Poisson Distribution Likelihood')
plt.grid(True, alpha=0.3)
plt.legend()

plt.suptitle('Comparison of Likelihood Function Widths', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'likelihood_width_comparison.png'), dpi=300, bbox_inches='tight')

# ========= SAMPLING DISTRIBUTION SIMULATION =========

print_section_header("PART 4: Sampling Distribution Simulation")

# Simulate multiple samples to show convergence to true parameters
n_simulations = 1000
normal_mle_samples = []
poisson_mle_samples = []

for _ in range(n_simulations):
    # Normal sample
    norm_sample = np.random.normal(true_mu, true_sigma, n_obs)
    normal_mle_samples.append(np.mean(norm_sample))
    
    # Poisson sample
    pois_sample = np.random.poisson(true_lambda, n_obs)
    poisson_mle_samples.append(np.mean(pois_sample))

normal_mle_samples = np.array(normal_mle_samples)
poisson_mle_samples = np.array(poisson_mle_samples)

plt.figure(figsize=(14, 7))

plt.subplot(121)
plt.hist(normal_mle_samples, bins=30, alpha=0.7, color='skyblue')
plt.axvline(true_mu, color='r', linestyle='--', label=f'True μ = {true_mu}')
plt.axvline(np.mean(normal_mle_samples), color='g', linestyle='-', 
            label=f'Mean of MLEs = {np.mean(normal_mle_samples):.4f}')
plt.xlabel('Sample Mean (MLE)')
plt.ylabel('Frequency')
plt.title(f'Sampling Distribution of Normal MLE (n={n_obs})')
plt.legend()

plt.subplot(122)
plt.hist(poisson_mle_samples, bins=30, alpha=0.7, color='lightgreen')
plt.axvline(true_lambda, color='r', linestyle='--', label=f'True λ = {true_lambda}')
plt.axvline(np.mean(poisson_mle_samples), color='g', linestyle='-', 
            label=f'Mean of MLEs = {np.mean(poisson_mle_samples):.4f}')
plt.xlabel('Sample Mean (MLE)')
plt.ylabel('Frequency')
plt.title(f'Sampling Distribution of Poisson MLE (n={n_obs})')
plt.legend()

plt.suptitle('Sampling Distributions of Maximum Likelihood Estimators', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'sampling_distributions.png'), dpi=300, bbox_inches='tight')

# Display theoretical vs observed properties
print("\nEmpirical Results from Simulation:")
print(f"Normal MLE - Mean: {np.mean(normal_mle_samples):.4f}, Std Dev: {np.std(normal_mle_samples):.4f}")
print(f"Poisson MLE - Mean: {np.mean(poisson_mle_samples):.4f}, Std Dev: {np.std(poisson_mle_samples):.4f}")

print("\nComparison with Theoretical Values:")
print(f"Normal - Theoretical Mean: {true_mu}, Theoretical Std Dev: {true_sigma/np.sqrt(n_obs):.4f}")
print(f"Poisson - Theoretical Mean: {true_lambda}, Theoretical Std Dev: {np.sqrt(true_lambda/n_obs):.4f}")

# Conclusion
print_section_header("SUMMARY AND CONCLUSIONS")

print("1. Maximum Likelihood Estimation provides a principled way to estimate parameters")
print("   from observed data by maximizing the likelihood function.")
print()
print("2. For both normal and Poisson distributions, the MLE of the main parameter")
print("   is simply the sample mean, though this is not true for all distributions.")
print()
print("3. The width of the likelihood function around the MLE relates to the uncertainty")
print("   of the estimate and is connected to the Fisher Information.")
print()
print("4. MLEs have desirable properties:")
print("   - Consistency: Converge to the true parameter value as sample size increases")
print("   - Asymptotic normality: MLEs are approximately normally distributed for large samples")
print("   - Efficiency: MLEs achieve the Cramér-Rao lower bound asymptotically")
print()
print("5. For finite samples, MLEs will typically differ from the true parameter values")
print("   due to sampling variability, as demonstrated in our visual analysis.")

print(f"\nAll visualizations saved in '{save_dir}'")
plt.close('all') 