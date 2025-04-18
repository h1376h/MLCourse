import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import gamma, expon
from scipy.optimize import minimize_scalar

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_7_Quiz_15")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Understanding the Problem
print_step_header(1, "Understanding the Problem")

print("Given:")
print("- Inter-arrival times at a hospital emergency room (in minutes):")
data = np.array([12.1, 8.3, 15.7, 9.2, 10.5, 7.8, 14.2, 11.9, 13.4, 9.8])
print(f"- Data: {data}")
print("- Assumed distribution: Exponential with parameter λ (rate)")
print("- Prior for λ: Gamma(2, 4) - using shape-scale parameterization")
print("\nTask:")
print("1. Calculate the Maximum Likelihood Estimate (MLE) for λ")
print("2. Derive the posterior distribution")
print("3. Calculate the Maximum A Posteriori (MAP) estimate for λ")
print("4. Find the probability that the next inter-arrival time will be greater than 15 minutes")

# Step 2: Calculating the MLE
print_step_header(2, "Calculating the MLE")

n = len(data)
data_sum = np.sum(data)
mle = n / data_sum

print("For an exponential distribution with parameter λ (rate), the MLE is:")
print("MLE = n / Σx_i")
print(f"MLE = {n} / {data_sum:.1f}")
print(f"MLE = {mle:.6f}")
print(f"Mean of the data (1/MLE): {data_sum/n:.4f} minutes")

# Step 3: Deriving the Posterior Distribution
print_step_header(3, "Deriving the Posterior Distribution")

# Prior parameters (shape, scale) for Gamma distribution
alpha_prior = 2
beta_prior = 4

# Posterior parameters
alpha_posterior = alpha_prior + n
beta_posterior = 1 / (1/beta_prior + data_sum)

# Convert to shape-rate for some calculations
rate_posterior = 1 / beta_posterior

print("For an exponential likelihood with a Gamma prior, the posterior is also a Gamma distribution.")
print("Using the formula for the posterior with Gamma-Exponential conjugacy:")
print("\nLikelihood: L(λ|data) ∝ λ^n * exp(-λ * Σx_i)")
print(f"Prior: p(λ) = Gamma(λ|{alpha_prior}, {beta_prior}) ∝ λ^({alpha_prior-1}) * exp(-λ/{beta_prior})")
print("Posterior: p(λ|data) ∝ Likelihood × Prior")
print(f"p(λ|data) ∝ λ^n * exp(-λ * Σx_i) × λ^({alpha_prior-1}) * exp(-λ/{beta_prior})")
print(f"p(λ|data) ∝ λ^({alpha_prior+n-1}) * exp(-λ * (Σx_i + 1/{beta_prior}))")
print(f"p(λ|data) ∝ λ^({alpha_posterior-1}) * exp(-λ * {1/beta_posterior:.6f})")
print(f"Posterior = Gamma({alpha_posterior}, {beta_posterior:.6f}) (shape-scale)")
print(f"Posterior = Gamma({alpha_posterior}, {rate_posterior:.6f}) (shape-rate)")

# Create a visualization of the prior, likelihood, and posterior
lambda_range = np.linspace(0.01, 0.3, 1000)

# Prior (shape, scale parameterization)
prior = gamma.pdf(lambda_range, alpha_prior, scale=beta_prior)

# Likelihood (not normalized)
likelihood = lambda_range**n * np.exp(-lambda_range * data_sum)

# Normalize the likelihood for better visualization
likelihood = likelihood / np.max(likelihood)

# Posterior (shape, scale parameterization)
posterior = gamma.pdf(lambda_range, alpha_posterior, scale=beta_posterior)

# Normalize the posterior for better visualization
posterior = posterior / np.max(posterior)

plt.figure(figsize=(10, 6))
plt.plot(lambda_range, prior / np.max(prior), 'r--', 
         label=f'Prior: Gamma({alpha_prior}, {beta_prior}) (shape-scale)', linewidth=2)
plt.plot(lambda_range, likelihood, 'g-.', 
         label=f'Likelihood (normalized)', linewidth=2)
plt.plot(lambda_range, posterior, 'b-', 
         label=f'Posterior: Gamma({alpha_posterior}, {beta_posterior:.4f})', linewidth=2)

plt.xlabel('λ (Rate parameter)', fontsize=12)
plt.ylabel('Density (Normalized)', fontsize=12)
plt.title('Bayesian Inference: Prior, Likelihood, and Posterior', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "prior_likelihood_posterior.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 4: Calculate the MAP Estimate
print_step_header(4, "Calculating the MAP Estimate")

# For Gamma(a, b) with shape-scale parameterization, the mode is (a-1)*b when a > 1
map_estimate = (alpha_posterior - 1) * beta_posterior

# Alternative calculation using optimization
def negative_log_posterior(lambda_val):
    return -((alpha_posterior - 1) * np.log(lambda_val) - lambda_val / beta_posterior)

result = minimize_scalar(negative_log_posterior, bracket=[0.01, 0.3])
map_estimate_optimizer = result.x

print("For a Gamma(a, b) distribution with shape-scale parameterization and a > 1, the mode (MAP estimate) is:")
print("MAP = (a - 1) * b")
print(f"MAP = ({alpha_posterior} - 1) * {beta_posterior:.6f}")
print(f"MAP = {alpha_posterior - 1} * {beta_posterior:.6f}")
print(f"MAP = {map_estimate:.6f}")
print(f"MAP (via optimization) = {map_estimate_optimizer:.6f}")
print(f"Mean of the posterior = {alpha_posterior * beta_posterior:.6f}")

# Step 5: Compare MAP and MLE
print_step_header(5, "Comparing MAP and MLE Estimates")

print(f"MAP estimate: {map_estimate:.6f}")
print(f"MLE estimate: {mle:.6f}")
print(f"Difference (MAP - MLE): {map_estimate - mle:.6f}")
print(f"Mean of data (1/MLE): {1/mle:.4f} minutes")
print(f"Mean of posterior exponential: {1/map_estimate:.4f} minutes")
print("\nExplanation of the difference:")
print("The MAP estimate incorporates the prior belief (Gamma(2, 4)), while the MLE only considers the data.")
print(f"The prior Gamma(2, 4) has mean {alpha_prior * beta_prior:.4f}, which pulls the MAP estimate away from the MLE.")

# Visualize the comparison of MAP and MLE
plt.figure(figsize=(10, 6))
plt.plot(lambda_range, posterior / np.max(posterior), 'b-', label='Posterior Density', linewidth=2)
plt.axvline(x=map_estimate, color='r', linestyle='-', label=f'MAP Estimate: {map_estimate:.6f}', linewidth=2)
plt.axvline(x=mle, color='g', linestyle='--', label=f'MLE Estimate: {mle:.6f}', linewidth=2)

# Fill the area between MAP and MLE
plt.fill_between(lambda_range, 0, posterior / np.max(posterior), 
                 where=(lambda_range >= min(map_estimate, mle)) & 
                       (lambda_range <= max(map_estimate, mle)),
                 color='gray', alpha=0.3)

plt.annotate(f'Difference: {abs(map_estimate - mle):.6f}',
             xy=((map_estimate + mle) / 2, 0.5),
             xytext=((map_estimate + mle) / 2, 0.7),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
             fontsize=10, ha='center')

plt.xlabel('λ (Rate parameter)', fontsize=12)
plt.ylabel('Normalized Density', fontsize=12)
plt.title('Comparison of MAP and MLE Estimates', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "map_vs_mle.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 6: Probability that the next inter-arrival time will be greater than 15 minutes
print_step_header(6, "Probability that Next Inter-arrival Time > 15 Minutes")

# Approach 1: Using the posterior predictive distribution
# For exponential likelihood with gamma posterior, the posterior predictive follows a Lomax distribution
# But we can compute this directly by integrating over the posterior:
# P(X > 15) = ∫ P(X > 15 | λ) × p(λ|data) dλ

# We'll use Monte Carlo integration by sampling from the posterior
np.random.seed(42)
n_samples = 100000
lambda_samples = np.random.gamma(alpha_posterior, beta_posterior, n_samples)
prob_gt_15_samples = np.mean(np.random.exponential(1/lambda_samples) > 15)

# Approach 2: Using the closed-form expression for the posterior predictive
def lomax_survival(x, alpha, beta):
    """Survival function for Lomax distribution: P(X > x)"""
    return (1 + x/(alpha*beta))**(-alpha)

prob_gt_15_exact = lomax_survival(15, alpha_posterior, beta_posterior)

# Approach 3: Using the MAP estimate (point estimate approach)
prob_gt_15_map = np.exp(-map_estimate * 15)

# Approach 4: Using the MLE (point estimate approach)
prob_gt_15_mle = np.exp(-mle * 15)

print("We can calculate the probability that the next inter-arrival time is > 15 minutes in several ways:")
print("\n1. Using Monte Carlo integration over the posterior (most accurate):")
print(f"P(X > 15) = {prob_gt_15_samples:.6f}")

print("\n2. Using the closed-form posterior predictive (Lomax distribution):")
print(f"P(X > 15) = {prob_gt_15_exact:.6f}")

print("\n3. Using the MAP estimate (point estimate approach):")
print(f"P(X > 15 | λ = λ_MAP) = exp(-λ_MAP * 15) = exp(-{map_estimate:.6f} * 15) = {prob_gt_15_map:.6f}")

print("\n4. Using the MLE (point estimate approach):")
print(f"P(X > 15 | λ = λ_MLE) = exp(-λ_MLE * 15) = exp(-{mle:.6f} * 15) = {prob_gt_15_mle:.6f}")

print("\nNotice that the Bayesian approaches (1 and 2) account for parameter uncertainty,")
print("while the point estimate approaches (3 and 4) do not.")

# Visualize the posterior predictive distribution
x_range = np.linspace(0, 30, 1000)

# Generate samples from the posterior predictive
pp_samples = np.zeros(n_samples)
for i in range(n_samples):
    pp_samples[i] = np.random.exponential(1/lambda_samples[i])

# Plot histogram of posterior predictive samples
plt.figure(figsize=(10, 6))
plt.hist(pp_samples, bins=50, density=True, alpha=0.5, label='Posterior Predictive Samples')

# Add vertical line at x = 15
plt.axvline(x=15, color='r', linestyle='--', linewidth=2, 
            label=f'x = 15, P(X > 15) ≈ {prob_gt_15_exact:.4f}')

# Fill area for P(X > 15)
hist, bin_edges = np.histogram(pp_samples, bins=50, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
valid_indices = bin_centers >= 15
if np.any(valid_indices):
    plt.fill_between(bin_centers[valid_indices], 0, hist[valid_indices], 
                     color='lightcoral', alpha=0.3,
                     label='P(X > 15)')

plt.xlabel('Arrival Time (minutes)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Posterior Predictive Distribution with P(X > 15) Highlighted', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "posterior_predictive.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

print("\nConclusion:")
print("1. The MLE for λ is directly related to the sample mean: λ_MLE = 1/(sample mean).")
print("2. The Gamma prior combined with exponential likelihood gives a Gamma posterior.")
print("3. The MAP estimate is influenced by both the prior and the data.")
print("4. The full posterior predictive distribution accounts for parameter uncertainty,")
print("   giving a more accurate probability estimate for future data.") 