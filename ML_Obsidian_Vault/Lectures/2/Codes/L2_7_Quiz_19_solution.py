import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import gamma, poisson
from scipy.optimize import minimize_scalar

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_7_Quiz_19")
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
print("- Count data D = {x_1, x_2, ..., x_N} from Poisson distribution with rate parameter λ")
print("- Prior for λ is a Gamma distribution with shape parameter α and rate parameter β")
print("\nTask:")
print("1. Write out the log-posterior log P(λ|D)")
print("2. Take the derivative of the log-posterior with respect to λ")
print("3. Solve for λ_MAP by setting the derivative equal to zero")

# Step 2: Deriving the Log-Posterior
print_step_header(2, "Deriving the Log-Posterior")

print("The Poisson likelihood function for a single observation x_i is:")
print("P(x_i|λ) = (λ^x_i * e^(-λ)) / x_i!")
print("\nFor N independent observations, the joint likelihood is:")
print("P(D|λ) = ∏_{i=1}^N P(x_i|λ) = ∏_{i=1}^N (λ^x_i * e^(-λ)) / x_i!")
print("      = (λ^(∑x_i) * e^(-Nλ)) / ∏x_i!")

print("\nThe Gamma prior for λ is:")
print("P(λ) = (β^α / Γ(α)) * λ^(α-1) * e^(-βλ)")

print("\nUsing Bayes' rule, the posterior is proportional to the likelihood times the prior:")
print("P(λ|D) ∝ P(D|λ) * P(λ)")

print("\nTaking the logarithm of both sides:")
print("log P(λ|D) ∝ log P(D|λ) + log P(λ)")
print("          ∝ log[(λ^(∑x_i) * e^(-Nλ)) / ∏x_i!] + log[(β^α / Γ(α)) * λ^(α-1) * e^(-βλ)]")
print("          ∝ ∑x_i * log(λ) - Nλ - log(∏x_i!) + α*log(β) - log(Γ(α)) + (α-1)*log(λ) - βλ")
print("          ∝ ∑x_i * log(λ) - Nλ + (α-1)*log(λ) - βλ")
print("          ∝ (∑x_i + α - 1) * log(λ) - (N + β)λ")

# We'll use some example data to visualize
# Define example parameters
alpha = 3  # Shape parameter of the Gamma prior
beta = 2   # Rate parameter of the Gamma prior
N = 10     # Number of observations
# Let's say the sum of observed counts is 15
sum_x = 15

# Step 3: Taking the Derivative of the Log-Posterior
print_step_header(3, "Taking the Derivative of the Log-Posterior")

print("The log-posterior is:")
print("log P(λ|D) ∝ (∑x_i + α - 1) * log(λ) - (N + β)λ")

print("\nTaking the derivative with respect to λ:")
print("d/dλ[log P(λ|D)] = (∑x_i + α - 1) / λ - (N + β)")

# Step 4: Solving for λ_MAP
print_step_header(4, "Solving for λ_MAP")

print("Setting the derivative equal to zero:")
print("(∑x_i + α - 1) / λ - (N + β) = 0")

print("\nSolving for λ:")
print("(∑x_i + α - 1) / λ = (N + β)")
print("λ = (∑x_i + α - 1) / (N + β)")

# Calculate λ_MAP for our example
lambda_map = (sum_x + alpha - 1) / (N + beta)

print(f"\nUsing the example values:")
print(f"∑x_i = {sum_x}, α = {alpha}, β = {beta}, N = {N}")
print(f"λ_MAP = ({sum_x} + {alpha} - 1) / ({N} + {beta})")
print(f"λ_MAP = {sum_x + alpha - 1} / {N + beta}")
print(f"λ_MAP = {lambda_map:.4f}")

# Verify with numerical optimization
def negative_log_posterior(lambda_val, sum_x, alpha, beta, N):
    if lambda_val <= 0:
        return float('inf')  # Log is undefined for non-positive values
    return -((sum_x + alpha - 1) * np.log(lambda_val) - (N + beta) * lambda_val)

# Find the minimum of the negative log posterior (equivalent to maximum of log posterior)
result = minimize_scalar(negative_log_posterior, args=(sum_x, alpha, beta, N), 
                         bounds=(0.001, 10), method='bounded')
lambda_numerical = result.x

print(f"\nVerification using numerical optimization:")
print(f"λ_MAP (numerical) = {lambda_numerical:.4f}")

# Step 5: Visualizing the Posterior and MAP Estimate
print_step_header(5, "Visualizing the Posterior and MAP Estimate")

# Create a range of lambda values
lambda_range = np.linspace(0.1, 5, 1000)

# Calculate the unnormalized log posterior for each lambda
log_posterior = [(sum_x + alpha - 1) * np.log(lam) - (N + beta) * lam for lam in lambda_range]

# Convert to posterior (unnormalized)
posterior = np.exp(log_posterior - np.max(log_posterior))  # Subtract max for numerical stability

# Normalize for plotting
posterior = posterior / np.max(posterior)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(lambda_range, posterior, 'b-', linewidth=2, label='Posterior Distribution')
plt.axvline(x=lambda_map, color='r', linestyle='-', 
            label=f'MAP Estimate: λ_MAP = {lambda_map:.4f}', linewidth=2)

# Add prior and likelihood for comparison
prior = gamma.pdf(lambda_range, alpha, scale=1/beta)
prior = prior / np.max(prior)  # Normalize for visualization

# Simple approximation of the likelihood (unnormalized)
# Using the fact that the likelihood peaks at sum_x / N for Poisson
likelihood = np.exp(sum_x * np.log(lambda_range) - N * lambda_range)
likelihood = likelihood / np.max(likelihood)  # Normalize for visualization

plt.plot(lambda_range, prior, 'g--', linewidth=2, label='Prior (Gamma)')
plt.plot(lambda_range, likelihood, 'k-.', linewidth=2, label='Likelihood (Poisson)')

plt.xlabel('λ (Rate Parameter)', fontsize=12)
plt.ylabel('Normalized Density', fontsize=12)
plt.title('Posterior Distribution with MAP Estimate for Poisson-Gamma Model', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "posterior_map_estimate.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 6: Effect of Different Priors and Data on the MAP Estimate
print_step_header(6, "Effect of Different Priors and Data on the MAP Estimate")

# Define different prior parameters to investigate
prior_params = [
    (1, 1, 'Weak Prior: α=1, β=1'),
    (3, 2, 'Moderate Prior: α=3, β=2'),
    (10, 5, 'Stronger Prior: α=10, β=5'),
    (20, 2, 'Informative Prior: α=20, β=2'),
]

# Define different data scenarios
data_scenarios = [
    (5, 10, 'Small Count: sum=5, N=10'),
    (15, 10, 'Medium Count: sum=15, N=10'),
    (50, 20, 'Large Count: sum=50, N=20'),
]

# Create a plot to show how different priors affect the MAP estimate
plt.figure(figsize=(12, 8))

colors = ['red', 'blue', 'green', 'purple']
linestyles = ['-', '--', '-.', ':']

# Fix the data scenario for this plot
data_idx = 1  # Use the medium count scenario
sum_x_fixed, N_fixed, data_label = data_scenarios[data_idx]

for i, (alpha, beta, prior_label) in enumerate(prior_params):
    # Calculate the MAP estimate
    lambda_map = (sum_x_fixed + alpha - 1) / (N_fixed + beta)
    
    # Calculate the posterior for this prior
    log_posterior = [(sum_x_fixed + alpha - 1) * np.log(lam) - (N_fixed + beta) * lam 
                     for lam in lambda_range]
    posterior = np.exp(log_posterior - np.max(log_posterior))
    posterior = posterior / np.max(posterior)
    
    # Plot the posterior
    plt.plot(lambda_range, posterior, color=colors[i], linestyle=linestyles[i],
             label=f'{prior_label} → MAP: {lambda_map:.4f}', linewidth=2)
    
    # Mark the MAP estimate
    plt.axvline(x=lambda_map, color=colors[i], linestyle='-', alpha=0.5)

# Also show the MLE (which is sum_x/N for Poisson)
mle = sum_x_fixed / N_fixed
plt.axvline(x=mle, color='black', linestyle='-',
            label=f'MLE: {mle:.4f}', linewidth=2)

plt.xlabel('λ (Rate Parameter)', fontsize=12)
plt.ylabel('Normalized Posterior Density', fontsize=12)
plt.title(f'Effect of Different Priors on MAP Estimate ({data_label})', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "different_priors.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Create a plot to show how different data affects the MAP estimate
plt.figure(figsize=(12, 8))

# Fix the prior for this plot
prior_idx = 1  # Use the moderate prior
alpha_fixed, beta_fixed, prior_label = prior_params[prior_idx]

for i, (sum_x, N, data_label) in enumerate(data_scenarios):
    # Calculate the MAP estimate
    lambda_map = (sum_x + alpha_fixed - 1) / (N + beta_fixed)
    
    # Calculate the posterior for this data
    log_posterior = [(sum_x + alpha_fixed - 1) * np.log(lam) - (N + beta_fixed) * lam 
                     for lam in lambda_range]
    posterior = np.exp(log_posterior - np.max(log_posterior))
    posterior = posterior / np.max(posterior)
    
    # Plot the posterior
    plt.plot(lambda_range, posterior, color=colors[i], linestyle=linestyles[i],
             label=f'{data_label} → MAP: {lambda_map:.4f}', linewidth=2)
    
    # Mark the MAP estimate
    plt.axvline(x=lambda_map, color=colors[i], linestyle='-', alpha=0.5)
    
    # Also show the MLE for this data
    mle = sum_x / N
    plt.axvline(x=mle, color=colors[i], linestyle=':', alpha=0.5,
                label=f'MLE ({data_label}): {mle:.4f}')

plt.xlabel('λ (Rate Parameter)', fontsize=12)
plt.ylabel('Normalized Posterior Density', fontsize=12)
plt.title(f'Effect of Different Data on MAP Estimate ({prior_label})', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "different_data.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 7: Special case - Extremely Large Prior Variance
print_step_header(7, "Special Case: Extremely Large Prior Variance")

print("In the special case where the prior variance becomes extremely large (β → 0 for Gamma prior),")
print("the MAP estimate approaches the MLE.")
print("\nFor a Gamma(α, β) prior, the variance is α/β². As β approaches 0, the variance approaches infinity.")
print("In this case, the prior becomes non-informative and has little influence on the posterior.")
print("\nThe MAP estimate is:")
print("λ_MAP = (∑x_i + α - 1) / (N + β)")
print("\nAs β → 0:")
print("λ_MAP ≈ (∑x_i + α - 1) / N")
print("\nAnd if α is also small compared to ∑x_i, then:")
print("λ_MAP ≈ ∑x_i / N = MLE")

# Let's visualize this by using a very small beta
small_beta = 0.001
lambda_map_small_beta = (sum_x + alpha - 1) / (N + small_beta)
mle = sum_x / N

print(f"\nFor our example with β = {small_beta}:")
print(f"λ_MAP = ({sum_x} + {alpha} - 1) / ({N} + {small_beta})")
print(f"λ_MAP = {sum_x + alpha - 1} / {N + small_beta}")
print(f"λ_MAP = {lambda_map_small_beta:.4f}")
print(f"MLE = {sum_x} / {N} = {mle:.4f}")
print(f"Difference = {lambda_map_small_beta - mle:.6f}")

# Step 8: Summary and Conclusion
print_step_header(8, "Summary and Conclusion")

print("In this problem, we derived the MAP estimator for a Poisson rate parameter λ with a Gamma prior.")
print("The key results are:")
print("\n1. The log-posterior is proportional to (∑x_i + α - 1) * log(λ) - (N + β)λ")
print("2. Setting the derivative equal to zero gives us the MAP estimator:")
print("   λ_MAP = (∑x_i + α - 1) / (N + β)")
print("\n3. This can be interpreted as a weighted average between:")
print("   - The MLE: ∑x_i / N")
print("   - The prior mean: (α - 1) / β (adjusted)")
print("   with weights determined by the relative strengths of the data and prior")
print("\n4. As the prior becomes more diffuse (β → 0), the MAP estimate approaches the MLE")
print("5. As we collect more data (N and ∑x_i increase), the influence of the prior diminishes")

# Create a comparison table for different scenarios
print("\nComparison of MAP Estimates for Different Scenarios:")
print("------------------------------------------------------------------")
print("Scenario                Prior               MAP        MLE    Diff")
print("------------------------------------------------------------------")

for alpha, beta, prior_label in prior_params:
    for sum_x, N, data_label in data_scenarios:
        lambda_map = (sum_x + alpha - 1) / (N + beta)
        mle = sum_x / N
        diff = lambda_map - mle
        print(f"{data_label:<20} {prior_label:<20} {lambda_map:.4f}  {mle:.4f}  {diff:+.4f}")

print("------------------------------------------------------------------") 