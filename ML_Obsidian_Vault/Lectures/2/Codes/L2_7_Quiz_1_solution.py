import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import beta
from scipy.optimize import minimize_scalar

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_7_Quiz_1")
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
print("- Binomial likelihood with parameter θ (probability of success)")
print("- Data D = 8 successes out of n = 20 trials")
print("- Prior is Beta(2, 2)")
print("\nTask:")
print("1. Derive the posterior distribution")
print("2. Calculate the MAP estimate")
print("3. Calculate the MLE")
print("4. Compare the MAP and MLE estimates")

# Step 2: Deriving the Posterior Distribution
print_step_header(2, "Deriving the Posterior Distribution")

# Define the parameters
n = 20       # Total number of trials
k = 8        # Number of successes
alpha = 2    # Prior parameter
beta_param = 2  # Prior parameter

print("For a binomial likelihood with a Beta prior, the posterior is also a Beta distribution.")
print("Using the formula for the posterior with Beta-Binomial conjugacy:")
print("\nPosterior = Beta(α + k, β + n - k)")
print(f"Posterior = Beta({alpha} + {k}, {beta_param} + {n} - {k})")
print(f"Posterior = Beta({alpha+k}, {beta_param+n-k})")
print(f"Posterior = Beta(10, 14)")

# Create a visualization of the prior, likelihood, and posterior
theta_range = np.linspace(0, 1, 1000)

# Prior
prior = beta.pdf(theta_range, alpha, beta_param)

# Likelihood (not normalized)
# Note: The binomial coefficient is a constant scaling factor that doesn't affect the shape
likelihood = theta_range**k * (1-theta_range)**(n-k)

# Normalize the likelihood for better visualization
likelihood = likelihood / np.max(likelihood)

# Posterior
posterior = beta.pdf(theta_range, alpha+k, beta_param+n-k)

# Normalize the posterior for better visualization
posterior = posterior / np.max(posterior)

plt.figure(figsize=(10, 6))
plt.plot(theta_range, prior, 'r--', label=f'Prior: Beta({alpha}, {beta_param})', linewidth=2)
plt.plot(theta_range, likelihood, 'g-.', label=f'Likelihood: Bin({n}, {k})', linewidth=2)
plt.plot(theta_range, posterior, 'b-', label=f'Posterior: Beta({alpha+k}, {beta_param+n-k})', linewidth=2)

plt.xlabel('θ (Probability of Success)', fontsize=12)
plt.ylabel('Density (Normalized)', fontsize=12)
plt.title('Bayesian Inference: Prior, Likelihood, and Posterior', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "prior_likelihood_posterior.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 3: Calculate the MAP Estimate
print_step_header(3, "Calculating the MAP Estimate")

# For Beta(a, b), the mode is (a-1)/(a+b-2) when a, b > 1
a_posterior = alpha + k
b_posterior = beta_param + n - k
map_estimate = (a_posterior - 1) / (a_posterior + b_posterior - 2)

print("For a Beta(a, b) distribution with a > 1 and b > 1, the mode (MAP estimate) is:")
print("MAP = (a - 1) / (a + b - 2)")
print(f"MAP = ({a_posterior} - 1) / ({a_posterior} + {b_posterior} - 2)")
print(f"MAP = {a_posterior - 1} / {a_posterior + b_posterior - 2}")
print(f"MAP = {map_estimate:.4f}")

# Step 4: Calculate the MLE Estimate
print_step_header(4, "Calculating the MLE Estimate")

# For binomial distribution, MLE = k/n
mle_estimate = k / n

print("For a binomial distribution, the MLE is simply the sample proportion:")
print("MLE = k / n")
print(f"MLE = {k} / {n}")
print(f"MLE = {mle_estimate:.4f}")

# Step 5: Compare MAP and MLE
print_step_header(5, "Comparing MAP and MLE Estimates")

print(f"MAP estimate: {map_estimate:.4f}")
print(f"MLE estimate: {mle_estimate:.4f}")
print(f"Difference (MAP - MLE): {map_estimate - mle_estimate:.4f}")
print("\nExplanation of the difference:")
print("The MAP estimate incorporates the prior belief (Beta(2, 2)), while the MLE only considers the data.")
print(f"The prior Beta(2, 2) has mean {alpha/(alpha+beta_param):.4f}, which pulls the MAP estimate away from the MLE.")

# Visualize the comparison of MAP and MLE
plt.figure(figsize=(10, 6))
plt.plot(theta_range, posterior / np.max(posterior), 'b-', label='Posterior Density', linewidth=2)
plt.axvline(x=map_estimate, color='r', linestyle='-', label=f'MAP Estimate: {map_estimate:.4f}', linewidth=2)
plt.axvline(x=mle_estimate, color='g', linestyle='--', label=f'MLE Estimate: {mle_estimate:.4f}', linewidth=2)

# Fill the area between MAP and MLE
plt.fill_between(theta_range, 0, posterior / np.max(posterior), 
                 where=(theta_range >= min(map_estimate, mle_estimate)) & 
                       (theta_range <= max(map_estimate, mle_estimate)),
                 color='gray', alpha=0.3)

plt.annotate(f'Difference: {abs(map_estimate - mle_estimate):.4f}',
             xy=((map_estimate + mle_estimate) / 2, 0.5),
             xytext=((map_estimate + mle_estimate) / 2, 0.7),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
             fontsize=10, ha='center')

plt.xlabel('θ (Probability of Success)', fontsize=12)
plt.ylabel('Normalized Density', fontsize=12)
plt.title('Comparison of MAP and MLE Estimates', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "map_vs_mle.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 6: Effect of different priors on the MAP estimate
print_step_header(6, "Effect of Different Priors on the MAP Estimate")

# Define different priors
prior_params = [
    (1, 1, 'Uniform Prior: Beta(1, 1)'),
    (2, 2, 'Symmetric Prior: Beta(2, 2)'),
    (8, 2, 'Informative Prior: Beta(8, 2)'),
    (2, 8, 'Informative Prior: Beta(2, 8)'),
    (20, 20, 'Strong Symmetric Prior: Beta(20, 20)')
]

plt.figure(figsize=(12, 8))

# Plot the likelihood
plt.plot(theta_range, likelihood, 'k-.', label='Likelihood (Data Only)', linewidth=2)

map_estimates = []
colors = ['red', 'blue', 'green', 'purple', 'orange']
linestyles = ['-', '--', '-.', ':', '--']

for i, (a, b, label) in enumerate(prior_params):
    # Calculate and store the MAP estimate for this prior
    map_est = (a + k - 1) / (a + b + n - 2) if a + k > 1 and b + n - k > 1 else float('nan')
    map_estimates.append((a, b, map_est))
    
    # Plot the posterior for this prior
    posterior = beta.pdf(theta_range, a + k, b + n - k)
    posterior = posterior / np.max(posterior)  # Normalize
    plt.plot(theta_range, posterior, color=colors[i], linestyle=linestyles[i], 
             label=f'{label} → MAP: {map_est:.4f}', linewidth=2)
    
    # Mark the MAP estimate on the plot
    plt.axvline(x=map_est, color=colors[i], linestyle='-', alpha=0.5)

plt.axvline(x=mle_estimate, color='black', linestyle='-', 
            label=f'MLE: {mle_estimate:.4f}', linewidth=2)

plt.xlabel('θ (Probability of Success)', fontsize=12)
plt.ylabel('Normalized Density', fontsize=12)
plt.title('Effect of Different Priors on MAP Estimates', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "different_priors.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Print a comparison table of MAP estimates for different priors
print("\nComparison of MAP Estimates for Different Priors:")
print("------------------------------------------------")
print("Prior           MAP Estimate  Difference from MLE")
print("------------------------------------------------")
for a, b, map_est in map_estimates:
    print(f"Beta({a:2d}, {b:2d})      {map_est:.4f}       {map_est - mle_estimate:+.4f}")
print("------------------------------------------------")
print(f"MLE            {mle_estimate:.4f}       {0:.4f}")

print("\nConclusion:")
print("1. The MAP estimate is influenced by both the prior and the data.")
print("2. With a uniform prior, the MAP estimate equals the MLE.")
print("3. As the prior becomes more informative, the MAP estimate deviates more from the MLE.")
print("4. The direction of deviation depends on the shape of the prior.")
print("5. With larger sample sizes, the influence of the prior diminishes.") 