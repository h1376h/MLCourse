import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma, beta
import os
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_3_Quiz_15")
os.makedirs(save_dir, exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
# Disable LaTeX rendering to avoid issues
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'text.usetex': False,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

# Function to save plots
def save_plot(filename, dpi=300):
    file_path = os.path.join(save_dir, filename)
    plt.savefig(file_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {file_path}")

print("Generating visualizations for L2_3_Quiz_15...")

# Data for our demonstrations - simulating a dataset with unknown distribution
np.random.seed(42)  # For reproducibility

# Scenario: We have data points and want to determine which distribution fits best
# This is perfect for likelihood comparison

# Generate data from a beta distribution (but students won't know this)
true_alpha, true_beta = 2.5, 1.5
data_points = beta.rvs(true_alpha, true_beta, size=50)

# Let's create 5 useful visualizations

# 1. Plot the data histogram
plt.figure(figsize=(10, 6))
plt.hist(data_points, bins=15, density=True, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Observed Data Distribution')
save_plot('graph1_data_histogram.png')
plt.close()

# 2. Likelihood functions for different distributions with different parameters
# Define parameter grids
alphas = np.linspace(1, 4, 100)  # For beta distribution
means = np.linspace(0.2, 0.8, 100)  # For normal distribution
rates = np.linspace(1, 5, 100)  # For exponential distribution

# Calculate log-likelihoods
beta_logliks = np.zeros((len(alphas), len(alphas)))
normal_logliks = np.zeros(len(means))
exp_logliks = np.zeros(len(rates))

# Beta distribution (varying alpha and beta)
for i, alpha in enumerate(alphas):
    for j, b in enumerate(alphas):  # Using the same grid for beta parameter
        beta_logliks[i, j] = np.sum(np.log(beta.pdf(data_points, alpha, b) + 1e-10))

# Normal distribution (varying mean, fixed std=0.2)
for i, mean in enumerate(means):
    normal_logliks[i] = np.sum(np.log(norm.pdf(data_points, mean, 0.2) + 1e-10))

# Exponential distribution (varying rate)
for i, rate in enumerate(rates):
    # For exponential, we use scale=1/rate in scipy
    exp_logliks[i] = np.sum(np.log(gamma.pdf(data_points, a=1, scale=1/rate) + 1e-10))

# Find the maximum likelihood parameters
beta_max_idx = np.unravel_index(np.argmax(beta_logliks), beta_logliks.shape)
beta_mle_alpha = alphas[beta_max_idx[0]]
beta_mle_beta = alphas[beta_max_idx[1]]

normal_mle_mean = means[np.argmax(normal_logliks)]
exp_mle_rate = rates[np.argmax(exp_logliks)]

# 2. Plot likelihood contours for beta distribution
plt.figure(figsize=(10, 8))
contour = plt.contourf(alphas, alphas, beta_logliks, 50, cmap='viridis')
plt.colorbar(contour, label='Log-Likelihood')
plt.scatter(beta_mle_alpha, beta_mle_beta, color='red', s=100, marker='*', 
            label=f'MLE: alpha={beta_mle_alpha:.2f}, beta={beta_mle_beta:.2f}')
plt.xlabel('alpha parameter')
plt.ylabel('beta parameter')
plt.title('Log-Likelihood Contour Plot for Beta Distribution')
plt.legend()
save_plot('graph2_beta_loglik_contour.png')
plt.close()

# 3. Plot log-likelihood curves for different distributions
plt.figure(figsize=(12, 8))

# Normalize log-likelihoods for better visualization
beta_max_loglik = np.max(beta_logliks)
normal_max_loglik = np.max(normal_logliks)
exp_max_loglik = np.max(exp_logliks)

plt.subplot(3, 1, 1)
plt.plot(means, normal_logliks - normal_max_loglik, 'g-', linewidth=2)
plt.axvline(x=normal_mle_mean, color='r', linestyle='--', 
            label=f'MLE: mean={normal_mle_mean:.2f}')
plt.ylabel('Log-Likelihood')
plt.title('Normal Distribution (std=0.2)')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(rates, exp_logliks - exp_max_loglik, 'b-', linewidth=2)
plt.axvline(x=exp_mle_rate, color='r', linestyle='--', 
            label=f'MLE: rate={exp_mle_rate:.2f}')
plt.ylabel('Log-Likelihood')
plt.title('Exponential Distribution')
plt.legend()

plt.subplot(3, 1, 3)
# For beta, show the slice where beta=mle_beta
slice_idx = beta_max_idx[1]
plt.plot(alphas, beta_logliks[:, slice_idx] - beta_max_loglik, 'purple', linewidth=2)
plt.axvline(x=beta_mle_alpha, color='r', linestyle='--',
            label=f'MLE: alpha={beta_mle_alpha:.2f} (beta={beta_mle_beta:.2f})')
plt.xlabel('Parameter Value')
plt.ylabel('Log-Likelihood')
plt.title(f'Beta Distribution (slice at beta={beta_mle_beta:.2f})')
plt.legend()

plt.tight_layout()
save_plot('graph3_loglik_comparisons.png')
plt.close()

# 4. Compare the probability density functions with fitted parameters
x = np.linspace(0, 1, 1000)
beta_pdf = beta.pdf(x, beta_mle_alpha, beta_mle_beta)
normal_pdf = norm.pdf(x, normal_mle_mean, 0.2)
exp_pdf = gamma.pdf(x, a=1, scale=1/exp_mle_rate)  # Exponential as gamma with a=1

plt.figure(figsize=(10, 6))
plt.hist(data_points, bins=15, density=True, alpha=0.5, color='skyblue', edgecolor='black', label='Data')
plt.plot(x, beta_pdf, 'purple', linewidth=2, label=f'Beta(alpha={beta_mle_alpha:.2f}, beta={beta_mle_beta:.2f})')
plt.plot(x, normal_pdf, 'g', linewidth=2, label=f'Normal(mean={normal_mle_mean:.2f}, std=0.2)')
plt.plot(x, exp_pdf, 'b', linewidth=2, label=f'Exponential(rate={exp_mle_rate:.2f})')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Fitted Probability Density Functions')
plt.legend()
save_plot('graph4_fitted_pdfs.png')
plt.close()

# 5. Illustrate probability vs likelihood
# For a single data point, show how we interpret it differently
plt.figure(figsize=(12, 10))

# First subplot: Fixed distribution, varying data point (probability)
plt.subplot(2, 1, 1)
single_x = np.linspace(0, 1, 1000)
fixed_dist = beta.pdf(single_x, true_alpha, true_beta)

plt.plot(single_x, fixed_dist, 'b-', linewidth=2, label=f'Fixed Beta({true_alpha}, {true_beta})')

# Highlight a few points to demonstrate probability
highlighted_points = [0.2, 0.5, 0.8]
for point in highlighted_points:
    prob = beta.pdf(point, true_alpha, true_beta)
    plt.scatter(point, prob, s=100, color='red')
    plt.annotate(f'P(X={point:.1f}) = {prob:.3f}', 
                xy=(point, prob), xytext=(point+0.05, prob+0.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

plt.xlabel('x (Data Value)')
plt.ylabel('Probability Density')
plt.title('Probability: Fixed Distribution, Varying Data')
plt.legend()

# Second subplot: Fixed data point, varying distribution parameter (likelihood)
plt.subplot(2, 1, 2)
fixed_point = 0.5  # A specific data point
varying_param = np.linspace(0.5, 5, 1000)  # Varying alpha parameter
likelihoods = np.zeros_like(varying_param)

# Calculate the likelihood of the fixed point for different alpha values (keeping beta=true_beta)
for i, alpha in enumerate(varying_param):
    likelihoods[i] = beta.pdf(fixed_point, alpha, true_beta)

plt.plot(varying_param, likelihoods, 'r-', linewidth=2, 
        label=f'Likelihood for x={fixed_point} with beta={true_beta}')

# Highlight a few parameter values
highlighted_params = [1.0, 2.5, 4.0]
for param in highlighted_params:
    lik = beta.pdf(fixed_point, param, true_beta)
    plt.scatter(param, lik, s=100, color='blue')
    plt.annotate(f'L(alpha={param:.1f}) = {lik:.3f}', 
                xy=(param, lik), xytext=(param+0.2, lik+0.2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

plt.xlabel('alpha Parameter')
plt.ylabel('Likelihood')
plt.title('Likelihood: Fixed Data Point, Varying Distribution Parameter')
plt.legend()

plt.tight_layout()
save_plot('graph5_probability_vs_likelihood.png')
plt.close()

# Bonus graph: Likelihood ratio comparison between distributions
# Calculate the maximum log-likelihoods
max_beta_ll = beta_max_loglik
max_normal_ll = normal_max_loglik
max_exp_ll = exp_max_loglik

# Convert to linear scale for ratio
max_ll_values = [max_beta_ll, max_normal_ll, max_exp_ll]
max_l_values = np.exp(max_ll_values - np.max(max_ll_values))  # Avoid numerical issues

plt.figure(figsize=(10, 6))
bars = plt.bar(['Beta', 'Normal', 'Exponential'], max_l_values, color=['purple', 'green', 'blue'])
plt.ylabel('Relative Likelihood')
plt.title('Likelihood Ratio Comparison Between Distributions')

# Add the log-likelihood values as text
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'Log-L = {max_ll_values[i]:.2f}', 
            ha='center', va='bottom', fontsize=12)

save_plot('graph6_likelihood_ratio.png')
plt.close()

print(f"All visualizations for L2_3_Quiz_15 saved to: {save_dir}")
print(f"True distribution: Beta(alpha={true_alpha}, beta={true_beta})")
print(f"MLE estimates:")
print(f"Beta: alpha={beta_mle_alpha:.4f}, beta={beta_mle_beta:.4f}")
print(f"Normal: mean={normal_mle_mean:.4f}, std=0.2 (fixed)")
print(f"Exponential: rate={exp_mle_rate:.4f}") 