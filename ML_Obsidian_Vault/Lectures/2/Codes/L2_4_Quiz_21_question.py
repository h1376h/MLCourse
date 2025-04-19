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

# Set basic plot styles, avoiding LaTeX
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'text.usetex': False,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

# Generate synthetic data for a normal distribution
np.random.seed(42)  # For reproducibility
true_mu = 25
true_sigma = 5
n_obs = 50  # Sample size
data = np.random.normal(true_mu, true_sigma, n_obs)

# Function to calculate likelihood for different means
def normal_likelihood(mu_values, data, sigma):
    n = len(data)
    log_likelihoods = []
    
    for mu in mu_values:
        log_likelihood = -n/2 * np.log(2 * np.pi * sigma**2) - 1/(2*sigma**2) * np.sum((data - mu)**2)
        log_likelihoods.append(log_likelihood)
    
    return np.array(log_likelihoods)

# Generate range of possible mean values
mu_range = np.linspace(20, 30, 200)

# Calculate log-likelihood for each mean value
log_likelihoods = normal_likelihood(mu_range, data, true_sigma)

# Convert to likelihood (exponentiating is unnecessary as it can lead to numerical overflow)
# Normalized log-likelihoods for visualization
normalized_log_likelihoods = log_likelihoods - np.max(log_likelihoods)
likelihoods = np.exp(normalized_log_likelihoods)

# Calculate the MLE (should be the sample mean)
mle_mu = np.mean(data)
print(f"True mu: {true_mu}")
print(f"MLE estimate of mu: {mle_mu:.4f}")

# Method of Moments (MoM) - for normal, this is the same as MLE
mom_mu = np.mean(data)
print(f"Method of Moments estimate of mu: {mom_mu:.4f}")

# Median - robust estimator
median_mu = np.median(data)
print(f"Median estimate of mu: {median_mu:.4f}")

# Trimmed mean (removing 10% from each end)
trim_size = int(0.1 * n_obs)
if trim_size > 0:
    sorted_data = np.sort(data)
    trimmed_data = sorted_data[trim_size:-trim_size]
    trimmed_mu = np.mean(trimmed_data)
else:
    trimmed_mu = np.mean(data)
print(f"10% Trimmed mean estimate of mu: {trimmed_mu:.4f}")

# Part 2: Poisson Distribution
# Generate synthetic data for a Poisson distribution
true_lambda = 3.5
poisson_data = np.random.poisson(true_lambda, n_obs)

# Function to calculate Poisson log-likelihood
def poisson_log_likelihood(lambda_values, data):
    n = len(data)
    log_likelihoods = []
    
    for lambda_val in lambda_values:
        log_likelihood = np.sum(data * np.log(lambda_val) - lambda_val - np.log(factorial(data)))
        log_likelihoods.append(log_likelihood)
    
    return np.array(log_likelihoods)

# Generate range of possible lambda values
lambda_range = np.linspace(2, 5, 100)

# Calculate log-likelihood for each lambda
poisson_log_likelihoods = poisson_log_likelihood(lambda_range, poisson_data)

# Normalize for visualization
normalized_poisson_log_likelihoods = poisson_log_likelihoods - np.max(poisson_log_likelihoods)
poisson_likelihoods = np.exp(normalized_poisson_log_likelihoods)

# Calculate the MLE for lambda (should be the sample mean for Poisson)
mle_lambda = np.mean(poisson_data)
print(f"\nTrue lambda: {true_lambda}")
print(f"MLE estimate of lambda: {mle_lambda:.4f}")

# Calculate the median for comparison
median_lambda = np.median(poisson_data)
print(f"Median estimate of lambda: {median_lambda:.4f}")

# Count occurrences of each value for Poisson data
unique_values, counts = np.unique(poisson_data, return_counts=True)
probabilities = counts / n_obs

# Calculate theoretical PMF values for both MLE and true values
x_range = np.arange(0, max(unique_values) + 3)
pmf_mle = poisson.pmf(x_range, mle_lambda)
pmf_true = poisson.pmf(x_range, true_lambda)

# Create a combined figure for the question
plt.figure(figsize=(12, 10))
gs = GridSpec(2, 2, figure=plt.gcf())

# Normal distribution histogram and density
ax1 = plt.subplot(gs[0, 0])
ax1.hist(data, bins=15, density=True, alpha=0.6, color='skyblue', label='Data')
x = np.linspace(np.min(data) - 5, np.max(data) + 5, 1000)
ax1.plot(x, norm.pdf(x, mle_mu, true_sigma), 'r-', linewidth=2, label='MLE Fit')
ax1.plot(x, norm.pdf(x, true_mu, true_sigma), 'k:', linewidth=2, label='True Distribution')
ax1.set_xlabel('Value')
ax1.set_ylabel('Density')
ax1.set_title('(a) Normal Distribution Data and MLE Fit')
ax1.legend()

# Normal likelihood function
ax2 = plt.subplot(gs[0, 1])
ax2.plot(mu_range, likelihoods, 'b-', linewidth=2)
ax2.axvline(x=mle_mu, color='r', linestyle='--', linewidth=2, label='MLE')
ax2.axvline(x=true_mu, color='k', linestyle=':', linewidth=2, label='True Value')
ax2.set_xlabel('mu')
ax2.set_ylabel('Likelihood')
ax2.set_title('(b) Normal Distribution Likelihood Function')
ax2.legend()

# Poisson PMF
ax3 = plt.subplot(gs[1, 0])
ax3.bar(unique_values, probabilities, color='skyblue', alpha=0.6, label='Data')
ax3.plot(x_range, pmf_mle, 'ro-', linewidth=2, markersize=6, label='MLE Fit')
ax3.plot(x_range, pmf_true, 'ko--', linewidth=2, markersize=6, label='True Distribution')
ax3.set_xlabel('Count')
ax3.set_ylabel('Probability')
ax3.set_title('(c) Poisson Distribution Data and MLE Fit')
ax3.legend()

# Poisson likelihood function
ax4 = plt.subplot(gs[1, 1])
ax4.plot(lambda_range, poisson_likelihoods, 'b-', linewidth=2)
ax4.axvline(x=mle_lambda, color='r', linestyle='--', linewidth=2, label='MLE')
ax4.axvline(x=true_lambda, color='k', linestyle=':', linewidth=2, label='True Value')
ax4.set_xlabel('lambda')
ax4.set_ylabel('Likelihood')
ax4.set_title('(d) Poisson Distribution Likelihood Function')
ax4.legend()

plt.tight_layout()
# Only save the combined figure, not individual plots
plt.savefig(os.path.join(save_dir, 'mle_visual_question.png'), dpi=300, bbox_inches='tight')
plt.close('all')

print(f"\nVisualization saved in '{save_dir}'") 