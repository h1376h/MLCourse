import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon
import os
import matplotlib as mpl

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_3_Quiz_14")
os.makedirs(save_dir, exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

# Function to create and save a plot
def create_and_save_plot(filename, x, y, title, xlabel, ylabel, color='blue'):
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, color=color, linewidth=2.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# Create sample data from an exponential distribution
np.random.seed(42)
true_lambda = 2.0  # True parameter
sample_size = 100
data = np.random.exponential(scale=1/true_lambda, size=sample_size)

# Generate lambda values for plotting likelihood functions
lambda_values = np.linspace(0.5, 4, 1000)

# 1. Likelihood function
def log_likelihood(lambda_val, data):
    return sample_size * np.log(lambda_val) - lambda_val * np.sum(data)

# Calculate log-likelihood values for different lambdas
log_likelihood_values = [log_likelihood(lambda_val, data) for lambda_val in lambda_values]

# 2. Score function (derivative of log-likelihood)
def score_function(lambda_val, data):
    return sample_size / lambda_val - np.sum(data)

# Calculate score function values
score_values = [score_function(lambda_val, data) for lambda_val in lambda_values]

# 3. Fisher Information Function
def fisher_information(lambda_val):
    return sample_size / (lambda_val**2)

# Calculate Fisher information values
fisher_values = [fisher_information(lambda_val) for lambda_val in lambda_values]

# 4. Method of Moments vs. MLE bias comparison
def mle_bias(lambda_val, n):
    return 0  # MLE is unbiased for exponential

def mom_bias(lambda_val, n):
    # For small samples, we'll simulate some bias for illustration
    return -0.15 * lambda_val / np.sqrt(n)

# Sample sizes for bias comparison
sample_sizes = np.linspace(10, 100, 100)
mle_biases = [mle_bias(true_lambda, n) for n in sample_sizes]
mom_biases = [mom_bias(true_lambda, n) for n in sample_sizes]

# Create all the required plots
# 1. Log-likelihood function
create_and_save_plot(
    "graph1_log_likelihood.png",
    lambda_values,
    log_likelihood_values,
    r"Log-Likelihood Function $\ell(\lambda)$",
    r"$\lambda$",
    r"$\ell(\lambda)$"
)

# 2. Score function
create_and_save_plot(
    "graph2_score_function.png",
    lambda_values,
    score_values,
    r"Score Function $\frac{d\ell(\lambda)}{d\lambda}$",
    r"$\lambda$",
    r"$\frac{d\ell(\lambda)}{d\lambda}$",
    color='red'
)

# 3. Fisher Information function
create_and_save_plot(
    "graph3_fisher_information.png",
    lambda_values,
    fisher_values,
    r"Fisher Information $I(\lambda)$",
    r"$\lambda$",
    r"$I(\lambda)$",
    color='green'
)

# 4. MLE vs MoM bias comparison
plt.figure(figsize=(8, 6))
plt.plot(sample_sizes, mle_biases, 'b-', linewidth=2.5, label="MLE Bias")
plt.plot(sample_sizes, mom_biases, 'r-', linewidth=2.5, label="MoM Bias")
plt.axhline(y=0, color='k', linestyle='--', alpha=0.7)
plt.title(r"Bias Comparison: MLE vs. Method of Moments")
plt.xlabel(r"Sample Size $n$")
plt.ylabel(r"Bias")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "graph4_bias_comparison.png"), dpi=300, bbox_inches='tight')
plt.close()

# 5. Cramér-Rao bound visualization
def variance_bound(lambda_val, n):
    return lambda_val**2 / n

variance_bounds = [variance_bound(true_lambda, n) for n in sample_sizes]

plt.figure(figsize=(8, 6))
plt.plot(sample_sizes, variance_bounds, 'g-', linewidth=2.5)
plt.fill_between(sample_sizes, variance_bounds, alpha=0.3, color='green')
plt.title(r"Cramér-Rao Lower Bound for Exponential MLE")
plt.xlabel(r"Sample Size $n$")
plt.ylabel(r"Minimum Variance")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "graph5_cramer_rao.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"All visualizations have been saved to {save_dir}") 