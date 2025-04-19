import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import bernoulli
from datetime import datetime

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_7_Quiz_26")
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
print("- Samples x₁=0, x₂=0, x₃=1, x₄=1, x₅=0 from a Bernoulli distribution")
print("- Unknown parameter θ ∈ (0, 1)")
print("\nTask:")
print("1. Find the maximum likelihood estimator (MLE) for θ")
print("2. Select θ from {0.2, 0.5, 0.7} using MLE principle")
print("3. Find MAP with discrete prior π_θ(0.2)=0.1, π_θ(0.5)=0.01, π_θ(0.7)=0.89")
print("4. Compare MAP and MLE, explain differences")
print("5. Calculate MAP with more data (100 samples: 40 ones, 60 zeros)")

# Save the data
data = np.array([0, 0, 1, 1, 0])
print("\nData summary:")
print(f"Number of samples: {len(data)}")
print(f"Number of successes (1's): {np.sum(data)}")
print(f"Number of failures (0's): {len(data) - np.sum(data)}")
print(f"Sample: {data}")

# Step 2: Calculating the MLE
print_step_header(2, "Calculating the Maximum Likelihood Estimator (MLE)")

# For Bernoulli, MLE is simply the sample proportion of successes
theta_mle = np.mean(data)
print(f"MLE for θ: {theta_mle:.4f}")

# Derive the MLE formula
print("\nDerivation of MLE for Bernoulli distribution:")
print("For Bernoulli samples x₁, x₂, ..., xₙ with parameter θ:")
print("1. Likelihood function: L(θ) = ∏ᵢ p(xᵢ|θ) = ∏ᵢ θ^(xᵢ) × (1-θ)^(1-xᵢ)")
print("2. Log-likelihood: log L(θ) = ∑ᵢ xᵢ log(θ) + ∑ᵢ (1-xᵢ) log(1-θ)")
print("3. Differentiate log L(θ) with respect to θ and set to zero:")
print("   d/dθ log L(θ) = ∑ᵢ xᵢ/θ - ∑ᵢ (1-xᵢ)/(1-θ) = 0")
print("4. Solve for θ:")
print("   ∑ᵢ xᵢ/θ = ∑ᵢ (1-xᵢ)/(1-θ)")
print("   (1-θ)∑ᵢ xᵢ = θ∑ᵢ (1-xᵢ)")
print("   ∑ᵢ xᵢ - θ∑ᵢ xᵢ = θn - θ∑ᵢ xᵢ")
print("   ∑ᵢ xᵢ = θn")
print("   θ = ∑ᵢ xᵢ/n = sample mean")

# Create a visualization of the likelihood function
theta_range = np.linspace(0.01, 0.99, 1000)
likelihood = np.array([np.prod(bernoulli.pmf(data, p)) for p in theta_range])
log_likelihood = np.log(likelihood)

plt.figure(figsize=(10, 6))
plt.plot(theta_range, likelihood, 'b-', linewidth=2)
plt.axvline(x=theta_mle, color='r', linestyle='--', linewidth=2, 
            label=f'MLE: θ = {theta_mle:.4f}')
plt.xlabel('θ (probability of success)', fontsize=12)
plt.ylabel('Likelihood L(θ)', fontsize=12)
plt.title('Likelihood Function for Bernoulli Distribution', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "likelihood_function.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {file_path}")

# Also plot the log-likelihood for clarity
plt.figure(figsize=(10, 6))
plt.plot(theta_range, log_likelihood, 'g-', linewidth=2)
plt.axvline(x=theta_mle, color='r', linestyle='--', linewidth=2,
            label=f'MLE: θ = {theta_mle:.4f}')
plt.xlabel('θ (probability of success)', fontsize=12)
plt.ylabel('Log-Likelihood log L(θ)', fontsize=12)
plt.title('Log-Likelihood Function for Bernoulli Distribution', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "log_likelihood_function.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 3: MLE from Restricted Set
print_step_header(3, "MLE from Restricted Set {0.2, 0.5, 0.7}")

# Calculate likelihood for each value in the restricted set
theta_restricted = np.array([0.2, 0.5, 0.7])
likelihoods_restricted = np.array([np.prod(bernoulli.pmf(data, p)) for p in theta_restricted])
log_likelihoods_restricted = np.log(likelihoods_restricted)

# Find the maximum
theta_mle_restricted_idx = np.argmax(likelihoods_restricted)
theta_mle_restricted = theta_restricted[theta_mle_restricted_idx]

print("Likelihood values for restricted set:")
for i, theta in enumerate(theta_restricted):
    print(f"θ = {theta:.1f}: L(θ) = {likelihoods_restricted[i]:.6f}, log L(θ) = {log_likelihoods_restricted[i]:.6f}")

print(f"\nMLE from restricted set: θ = {theta_mle_restricted:.1f}")

# Visualize the restricted MLE
plt.figure(figsize=(10, 6))
plt.plot(theta_range, likelihood, 'b-', alpha=0.5, linewidth=2, label='Likelihood Function')
plt.scatter(theta_restricted, likelihoods_restricted, c='red', s=100, zorder=5, label='Restricted Set')
plt.axvline(x=theta_mle, color='g', linestyle='--', linewidth=1.5, 
            label=f'Unrestricted MLE: θ = {theta_mle:.4f}')
plt.axvline(x=theta_mle_restricted, color='r', linestyle='-', linewidth=2, 
            label=f'Restricted MLE: θ = {theta_mle_restricted:.1f}')

for i, theta in enumerate(theta_restricted):
    plt.annotate(f'θ = {theta:.1f}\nL(θ) = {likelihoods_restricted[i]:.6f}',
                 xy=(theta, likelihoods_restricted[i]),
                 xytext=(theta+0.05, likelihoods_restricted[i]+0.001),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                 fontsize=10)

plt.xlabel('θ (probability of success)', fontsize=12)
plt.ylabel('Likelihood L(θ)', fontsize=12)
plt.title('MLE from Restricted Set {0.2, 0.5, 0.7}', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "restricted_mle.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {file_path}")

# Step 4: MAP Estimation with Discrete Prior
print_step_header(4, "MAP Estimation with Discrete Prior")

# Define the prior probabilities
prior_values = np.array([0.2, 0.5, 0.7])
prior_probs = np.array([0.1, 0.01, 0.89])

print("Prior distribution:")
for i, theta in enumerate(prior_values):
    print(f"π_θ({theta:.1f}) = {prior_probs[i]:.2f}")

# Calculate posterior probabilities
posterior_unnormalized = likelihoods_restricted * prior_probs
posterior_probs = posterior_unnormalized / np.sum(posterior_unnormalized)

print("\nPosterior probabilities:")
for i, theta in enumerate(prior_values):
    print(f"P(θ={theta:.1f}|data) ∝ L(θ) × π_θ(θ) = {likelihoods_restricted[i]:.6f} × {prior_probs[i]:.2f} = {posterior_unnormalized[i]:.6f}")
    print(f"Normalized: P(θ={theta:.1f}|data) = {posterior_probs[i]:.6f}")

# Find the MAP estimate
theta_map_idx = np.argmax(posterior_probs)
theta_map = prior_values[theta_map_idx]

print(f"\nMAP estimate: θ = {theta_map:.1f}")

# Visualize prior, likelihood, and posterior
plt.figure(figsize=(12, 8))

# Create a subplot for the prior
plt.subplot(3, 1, 1)
plt.bar(prior_values, prior_probs, width=0.05, alpha=0.7, color='blue')
plt.xticks(prior_values)
plt.title('Prior Distribution π_θ(θ)', fontsize=12)
plt.ylabel('Probability', fontsize=10)
plt.grid(True, axis='y')

# Create a subplot for the likelihood
plt.subplot(3, 1, 2)
plt.bar(prior_values, likelihoods_restricted, width=0.05, alpha=0.7, color='green')
plt.xticks(prior_values)
plt.title('Likelihood Function L(θ|data)', fontsize=12)
plt.ylabel('Likelihood', fontsize=10)
plt.grid(True, axis='y')

# Create a subplot for the posterior
plt.subplot(3, 1, 3)
plt.bar(prior_values, posterior_probs, width=0.05, alpha=0.7, color='red')
plt.axvline(x=theta_map, color='r', linestyle='--', linewidth=2, 
            label=f'MAP: θ = {theta_map:.1f}')
plt.axvline(x=theta_mle_restricted, color='g', linestyle='--', linewidth=2, 
            label=f'MLE: θ = {theta_mle_restricted:.1f}')
plt.xticks(prior_values)
plt.title('Posterior Distribution P(θ|data)', fontsize=12)
plt.xlabel('θ (probability of success)', fontsize=10)
plt.ylabel('Probability', fontsize=10)
plt.legend()
plt.grid(True, axis='y')

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "map_estimation.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {file_path}")

# Step 5: MAP vs MLE Comparison
print_step_header(5, "MAP vs MLE Comparison")

print("Comparison of MAP and MLE estimates from restricted set:")
print(f"MLE: θ = {theta_mle_restricted:.1f}")
print(f"MAP: θ = {theta_map:.1f}")

if theta_mle_restricted != theta_map:
    print("\nThe MAP and MLE estimates differ. This is because:")
    print("1. The MAP takes into account the prior distribution, while the MLE only considers the likelihood.")
    print("2. The prior strongly favors θ = 0.7 (with probability 0.89).")
    print("3. With small sample size (n = 5), the prior has a significant influence on the posterior.")
else:
    print("\nThe MAP and MLE estimates are the same. This suggests that:")
    print("1. The likelihood strongly dominates the prior.")
    print("2. The prior happens to align with the likelihood maximum.")

# Step 6: MAP with Larger Sample Size
print_step_header(6, "MAP with Larger Sample Size")

# New data: 100 samples with 40 ones and 60 zeros
large_data_size = 100
large_data_ones = 40
large_data_zeros = 60

print(f"New data: {large_data_size} samples with {large_data_ones} ones and {large_data_zeros} zeros")

# Calculate likelihoods for the new data
likelihoods_large = np.array([
    bernoulli.pmf(1, p)**large_data_ones * bernoulli.pmf(0, p)**large_data_zeros 
    for p in prior_values
])

# Calculate posterior probabilities
posterior_unnormalized_large = likelihoods_large * prior_probs
posterior_probs_large = posterior_unnormalized_large / np.sum(posterior_unnormalized_large)

print("\nPosterior probabilities with large sample:")
for i, theta in enumerate(prior_values):
    print(f"P(θ={theta:.1f}|data) = {posterior_probs_large[i]:.6f}")

# Find the MAP estimate for large sample
theta_map_large_idx = np.argmax(posterior_probs_large)
theta_map_large = prior_values[theta_map_large_idx]

print(f"\nMAP estimate with large sample: θ = {theta_map_large:.1f}")

# For comparison, calculate the MLE for large sample
theta_mle_large = large_data_ones / large_data_size
print(f"MLE with large sample: θ = {theta_mle_large:.4f}")

# Create a separate calculation to more clearly see the impact of data on posteriors
print("\nImpact of increasing sample size on posterior probabilities:")

# Function to calculate posteriors for different sample sizes
def calculate_posterior(n_samples, proportion_ones, prior_values, prior_probs):
    n_ones = int(n_samples * proportion_ones)
    n_zeros = n_samples - n_ones
    likelihoods = np.array([
        bernoulli.pmf(1, p)**n_ones * bernoulli.pmf(0, p)**n_zeros 
        for p in prior_values
    ])
    posterior_unnorm = likelihoods * prior_probs
    return posterior_unnorm / np.sum(posterior_unnorm)

# Sample sizes to explore (from small to large)
sample_sizes = [5, 10, 20, 50, 100, 500, 1000]
proportion_ones = 0.4  # 40% ones, consistent with the large sample in the problem

# Calculate posteriors for each sample size
posteriors_by_size = []
map_estimates_by_size = []

print("\nPosterior probabilities and MAP estimates for different sample sizes (keeping 40% ones):")
for n in sample_sizes:
    posterior = calculate_posterior(n, proportion_ones, prior_values, prior_probs)
    posteriors_by_size.append(posterior)
    map_idx = np.argmax(posterior)
    map_estimates_by_size.append(prior_values[map_idx])
    
    print(f"\nSample size n = {n}")
    for i, theta in enumerate(prior_values):
        print(f"P(θ={theta:.1f}|data) = {posterior[i]:.6f}")
    print(f"MAP estimate: θ = {prior_values[map_idx]:.1f}")

# Visualize how posteriors change with sample size
plt.figure(figsize=(12, 8))

# Plot posteriors for each sample size
for i, n in enumerate(sample_sizes):
    plt.subplot(len(sample_sizes), 1, i+1)
    plt.bar(prior_values, posteriors_by_size[i], width=0.05, alpha=0.7, color='blue')
    plt.axvline(x=map_estimates_by_size[i], color='r', linestyle='--', linewidth=2, 
                label=f'MAP: θ = {map_estimates_by_size[i]:.1f}')
    plt.axvline(x=proportion_ones, color='g', linestyle='--', linewidth=1.5, 
                label=f'MLE: θ = {proportion_ones:.1f}')
    plt.xticks(prior_values)
    plt.yticks([0, 0.5, 1.0])
    plt.title(f'Posterior Distribution (n = {n})', fontsize=10)
    if i == len(sample_sizes) - 1:
        plt.xlabel('θ (probability of success)', fontsize=10)
    if i == 3:  # Middle plot
        plt.ylabel('Probability', fontsize=10)
    plt.legend(fontsize=8)
    plt.grid(True, axis='y')

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "posterior_evolution.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {file_path}")

# Step 7: Summary and Conclusion
print_step_header(7, "Summary and Conclusion")

print("Summary of findings:")
print(f"1. MLE for the original sample (5 points): θ = {theta_mle:.4f}")
print(f"2. MLE from restricted set {prior_values}: θ = {theta_mle_restricted:.1f}")
print(f"3. MAP with discrete prior: θ = {theta_map:.1f}")
print(f"4. MAP with larger sample (100 points): θ = {theta_map_large:.1f}")
print(f"5. MLE for larger sample: θ = {theta_mle_large:.4f}")

print("\nConclusions:")
print("1. The influence of the prior diminishes as sample size increases")
print("2. With small samples, the MAP estimate can be significantly influenced by the prior")
print("3. As sample size increases, both MAP and MLE converge toward the true parameter value")
print("4. For very large samples, the likelihood dominates the prior, making MAP and MLE similar")
print("5. The convergence of MAP to MLE with increasing data is an example of the principle")
print("   that 'the data overwhelms the prior' as more information becomes available")

# Calculate the influence ratio of prior vs data for different sample sizes
print("\nQuantifying prior influence for different sample sizes:")
print("(Measured by how far MAP is from MLE toward the prior mode)")

prior_mode = prior_values[np.argmax(prior_probs)]  # Mode of the prior distribution
print(f"Prior mode: θ = {prior_mode:.1f}")

for n in sample_sizes:
    mle = proportion_ones  # MLE is always 0.4 (40% ones)
    posterior = calculate_posterior(n, proportion_ones, prior_values, prior_probs)
    map_idx = np.argmax(posterior)
    map_estimate = prior_values[map_idx]
    
    # If MAP equals MLE, prior influence is 0
    # If MAP equals prior mode, prior influence is 1
    # Otherwise, calculate relative position
    if map_estimate == mle:
        prior_influence = 0
    elif map_estimate == prior_mode:
        prior_influence = 1
    else:
        if abs(prior_mode - mle) > 0:  # Avoid division by zero
            prior_influence = abs(map_estimate - mle) / abs(prior_mode - mle)
        else:
            prior_influence = 0
            
    print(f"n = {n}: MAP = {map_estimate:.1f}, Prior influence ≈ {prior_influence:.4f}")

print("\nAs we can see, the influence of the prior diminishes as the sample size increases,")
print("demonstrating the convergence of Bayesian methods to frequentist methods with large data.")

# Add a visualization of prior influence vs sample size
# Calculate influence for more sample sizes for a smoother curve
extended_sample_sizes = np.logspace(0, 4, 20).astype(int)  # From 1 to 10000
extended_sample_sizes = np.unique(extended_sample_sizes)  # Remove duplicates

prior_influences = []
for n in extended_sample_sizes:
    posterior = calculate_posterior(n, proportion_ones, prior_values, prior_probs)
    map_idx = np.argmax(posterior)
    map_estimate = prior_values[map_idx]
    
    if map_estimate == proportion_ones:
        prior_influences.append(0)
    elif map_estimate == prior_mode:
        prior_influences.append(1)
    else:
        if abs(prior_mode - proportion_ones) > 0:
            prior_influences.append(abs(map_estimate - proportion_ones) / 
                                  abs(prior_mode - proportion_ones))
        else:
            prior_influences.append(0)

plt.figure(figsize=(10, 6))
plt.semilogx(extended_sample_sizes, prior_influences, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Sample Size (log scale)', fontsize=12)
plt.ylabel('Prior Influence on MAP Estimate', fontsize=12)
plt.title('Diminishing Influence of Prior with Increasing Sample Size', fontsize=14)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.axhline(y=0, color='r', linestyle='--', linewidth=1, 
           label='No Prior Influence (MAP = MLE)')
plt.axhline(y=1, color='g', linestyle='--', linewidth=1, 
           label='Maximum Prior Influence (MAP = Prior Mode)')
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "prior_influence.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {file_path}")

print("\nThis analysis clearly demonstrates how the influence of the prior")
print("diminishes with increasing sample size, illustrating a fundamental")
print("principle of Bayesian statistics.")

print("\nAnalysis completed successfully!")
print(f"All figures saved to: {save_dir}") 