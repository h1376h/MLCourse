import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, norm
import os
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_5_Quiz_14")
solutions_dir = os.path.join(save_dir, "solutions")
os.makedirs(solutions_dir, exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Understanding the Problem
print_step_header(1, "Understanding the Problem")

print("Given:")
print("- Five visualizations showing different aspects of Bayesian inference")
print("- We are working with a coin flip scenario with 6 heads out of 10 flips")
print("- Three different prior distributions: Beta(2,5), Beta(5,2), and Beta(3,3)")
print("\nTask:")
print("1. Determine which prior has the strongest influence on the posterior")
print("2. Find the 90% credible interval for the Beta(3,3) prior")
print("3. Assess whether the prior or data has greater impact on the final posterior")
print("4. Calculate the expected value of θ for the posterior with Beta(5,2) prior")

# Step 2: Analyze Prior Influence on Posterior
print_step_header(2, "Analyzing Prior Influence on Posterior")

# Set up parameters
x = np.linspace(0, 1, 1000)
h, n = 6, 10  # 6 heads out of 10 flips

# Define priors and calculate posteriors
prior_params = [(2, 5, "Prior 1: Beta(2,5)"), 
                (5, 2, "Prior 2: Beta(5,2)"), 
                (3, 3, "Prior 3: Beta(3,3)")]
colors = ['r', 'g', 'b']

# Calculate likelihood function
def likelihood(theta, h, n):
    return theta**h * (1-theta)**(n-h)

likelihood_values = likelihood(x, h, n)
likelihood_norm = likelihood_values / np.max(likelihood_values)

# Plot priors, likelihood and posteriors together
plt.figure(figsize=(12, 8))

# Plot likelihood
plt.plot(x, likelihood_norm, 'k--', linewidth=2, label='Likelihood (6 heads out of 10)')

# Plot priors and posteriors
for i, (a, b, label) in enumerate(prior_params):
    # Plot prior
    prior = beta(a, b)
    prior_values = prior.pdf(x)
    prior_norm = prior_values / np.max(prior_values) * 0.7  # Scale for visualization
    plt.plot(x, prior_norm, f'{colors[i]}:', linewidth=2, 
             label=f'{label} (prior)')
    
    # Calculate posterior
    a_post = a + h
    b_post = b + (n - h)
    posterior = beta(a_post, b_post)
    
    # Plot posterior
    posterior_values = posterior.pdf(x)
    posterior_norm = posterior_values / np.max(posterior_values)
    plt.plot(x, posterior_norm, f'{colors[i]}-', linewidth=2, 
             label=f'Posterior with {label}')
    
    # Calculate distances
    prior_mode = (a-1)/(a+b-2) if a > 1 and b > 1 else (0 if a < 1 else 1)
    posterior_mode = (a_post-1)/(a_post+b_post-2)
    likelihood_mode = h/n
    
    print(f"\n{label}:")
    print(f"  Prior mode: {prior_mode:.4f}")
    print(f"  Likelihood mode: {likelihood_mode:.4f}")
    print(f"  Posterior mode: {posterior_mode:.4f}")
    print(f"  Shift from likelihood to posterior: {abs(posterior_mode - likelihood_mode):.4f}")

# Determine strongest prior influence
shifts = [(label, abs((a+h-1)/(a+b+n-2) - h/n)) 
          for a, b, label in prior_params]
strongest_prior = max(shifts, key=lambda x: x[1])

print(f"\nThe prior with the strongest influence is {strongest_prior[0]}")
print(f"with a shift of {strongest_prior[1]:.4f} from the likelihood mode.")

plt.xlabel('θ (probability of heads)', fontsize=12)
plt.ylabel('Normalized Density', fontsize=12)
plt.title('Comparison of Priors, Likelihood, and Posteriors', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

file_path = os.path.join(solutions_dir, "prior_influence_analysis.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"\nPrior influence analysis saved to: {file_path}")

# Step 3: Calculate Credible Interval
print_step_header(3, "Calculating 90% Credible Interval")

# Focus on Beta(3,3) prior
a, b = 3, 3
a_post, b_post = a + h, b + (n - h)
posterior = beta(a_post, b_post)

# Calculate 90% credible interval
lower_bound = posterior.ppf(0.05)
upper_bound = posterior.ppf(0.95)

print(f"For the posterior based on Beta(3,3) prior:")
print(f"  Posterior distribution: Beta({a_post}, {b_post})")
print(f"  90% credible interval: [{lower_bound:.4f}, {upper_bound:.4f}]")

# Visualize the credible interval
plt.figure(figsize=(10, 6))

# Plot the posterior
plt.plot(x, posterior.pdf(x), 'b-', linewidth=2, 
         label=f'Posterior: Beta({a_post}, {b_post})')

# Shade the credible interval
idx = (x >= lower_bound) & (x <= upper_bound)
plt.fill_between(x[idx], 0, posterior.pdf(x)[idx], color='b', alpha=0.3)

# Add vertical lines for interval bounds
plt.axvline(x=lower_bound, color='r', linestyle='--', linewidth=2,
            label=f'Lower bound: {lower_bound:.4f}')
plt.axvline(x=upper_bound, color='r', linestyle='--', linewidth=2,
            label=f'Upper bound: {upper_bound:.4f}')

# Add annotation
plt.annotate(f'90% Credible Interval:\n[{lower_bound:.4f}, {upper_bound:.4f}]',
             xy=(0.5, posterior.pdf(0.5)*0.7), xycoords='data',
             bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
             ha='center', fontsize=12)

plt.xlabel('θ (probability of heads)', fontsize=12)
plt.ylabel('Posterior Density', fontsize=12)
plt.title('90% Credible Interval for Posterior with Beta(3,3) Prior', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

file_path = os.path.join(solutions_dir, "credible_interval_analysis.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Credible interval analysis saved to: {file_path}")

# Step 4: Analyze Prior vs. Data Impact
print_step_header(4, "Analyzing Prior vs. Data Impact")

# Set up parameters for Bayesian updating
a_prior, b_prior = 2, 2
data_sequence = [(1, 1), (2, 3), (5, 6), (8, 9), (12, 12)]  # (heads, total flips)
colors = ['k', 'm', 'c', 'g', 'r']

plt.figure(figsize=(12, 8))

# Plot the distributions
for i, (cum_h, cum_n) in enumerate(data_sequence):
    # Calculate posterior parameters
    a_post = a_prior + cum_h
    b_post = b_prior + (cum_n - cum_h)
    
    # Create distribution
    dist = beta(a_post, b_post)
    
    # Plot distribution
    label = "Prior: Beta(2,2)" if i == 0 else f"After {cum_n} flips ({cum_h} heads)"
    plt.plot(x, dist.pdf(x), f'{colors[i]}-', linewidth=2, label=label)
    
    # Calculate and print statistics
    mean = a_post / (a_post + b_post)
    mode = (a_post-1)/(a_post+b_post-2) if a_post > 1 and b_post > 1 else 0.5
    variance = (a_post * b_post) / ((a_post + b_post)**2 * (a_post + b_post + 1))
    
    print(f"\nDistribution {i+1} - {label}:")
    print(f"  Parameters: Beta({a_post}, {b_post})")
    print(f"  Mean: {mean:.4f}")
    print(f"  Mode: {mode:.4f}")
    print(f"  Variance: {variance:.4f}")
    print(f"  Standard Deviation: {np.sqrt(variance):.4f}")

# Calculate influence metrics
initial_mean = a_prior / (a_prior + b_prior)
final_mean = (a_prior + data_sequence[-1][0]) / (a_prior + b_prior + data_sequence[-1][1])
data_only_mean = data_sequence[-1][0] / data_sequence[-1][1]

print(f"\nInfluence analysis:")
print(f"  Initial prior mean: {initial_mean:.4f}")
print(f"  Final posterior mean: {final_mean:.4f}")
print(f"  Data-only estimate (MLE): {data_only_mean:.4f}")
print(f"  Distance from prior to posterior: {abs(final_mean - initial_mean):.4f}")
print(f"  Distance from MLE to posterior: {abs(final_mean - data_only_mean):.4f}")

if abs(final_mean - initial_mean) > abs(final_mean - data_only_mean):
    print("\nConclusion: The data has a greater impact on the final posterior than the prior.")
else:
    print("\nConclusion: The prior has a greater impact on the final posterior than the data.")

plt.annotate("Prior's influence\ndiminishes as\ndata accumulates",
             xy=(0.3, 2.5), xycoords='data',
             bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
             ha='center', fontsize=12)

plt.annotate("Final posterior\nconcentrated around data",
             xy=(0.8, 4), xycoords='data',
             bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
             ha='center', fontsize=12)

plt.xlabel('θ (probability of heads)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Evolution of Belief Through Bayesian Updating', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

file_path = os.path.join(solutions_dir, "bayesian_updating_analysis.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Bayesian updating analysis saved to: {file_path}")

# Step 5: Calculate Expected Value
print_step_header(5, "Calculating Expected Value of θ")

# Parameters for Beta(5,2) prior
a, b = 5, 2
h, n = 6, 10  # 6 heads out of 10 flips

# Calculate posterior parameters
a_post = a + h
b_post = b + (n - h)

# Calculate expected value
expected_value = a_post / (a_post + b_post)

print(f"For the posterior based on Beta(5,2) prior:")
print(f"  Posterior distribution: Beta({a_post}, {b_post})")
print(f"  Expected value: E[θ] = {a_post} / ({a_post} + {b_post}) = {expected_value:.4f}")

# Visualize the expected value
plt.figure(figsize=(10, 6))

# Plot the prior and posterior
prior = beta(a, b)
posterior = beta(a_post, b_post)

plt.plot(x, prior.pdf(x), 'g:', linewidth=2, 
         label=f'Prior: Beta({a}, {b})')
plt.plot(x, posterior.pdf(x), 'g-', linewidth=2, 
         label=f'Posterior: Beta({a_post}, {b_post})')

# Add vertical line for expected value
plt.axvline(x=expected_value, color='r', linestyle='-', linewidth=2,
            label=f'Expected value: {expected_value:.4f}')

# Add annotation
plt.annotate(f'E[θ] = {expected_value:.4f}',
             xy=(expected_value, posterior.pdf(expected_value)*0.7), xycoords='data',
             xytext=(expected_value-0.15, posterior.pdf(expected_value)*0.9),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
             bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
             ha='center', fontsize=12)

plt.xlabel('θ (probability of heads)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Expected Value of θ for Posterior with Beta(5,2) Prior', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

file_path = os.path.join(solutions_dir, "expected_value_analysis.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Expected value analysis saved to: {file_path}")

# Step 6: Summary Visualization
print_step_header(6, "Creating Summary Visualization")

plt.figure(figsize=(15, 10))
gs = GridSpec(2, 2, figure=plt)

# 1. Prior influence subplot
ax1 = plt.subplot(gs[0, 0])
for i, (a, b, label) in enumerate(prior_params):
    # Calculate posterior
    a_post = a + h
    b_post = b + (n - h)
    posterior = beta(a_post, b_post)
    
    # Plot posterior
    ax1.plot(x, posterior.pdf(x), f'{colors[i]}-', linewidth=2, 
             label=f'Posterior with {label}')

ax1.set_title('1. Prior Influence on Posterior', fontsize=12)
ax1.set_xlabel('θ (probability of heads)', fontsize=10)
ax1.set_ylabel('Posterior Density', fontsize=10)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# 2. Credible interval subplot
ax2 = plt.subplot(gs[0, 1])
a, b = 3, 3
a_post, b_post = a + h, b + (n - h)
posterior = beta(a_post, b_post)

# Plot the posterior
ax2.plot(x, posterior.pdf(x), 'b-', linewidth=2, 
         label=f'Posterior with Beta(3,3) prior')

# Shade the credible interval
idx = (x >= lower_bound) & (x <= upper_bound)
ax2.fill_between(x[idx], 0, posterior.pdf(x)[idx], color='b', alpha=0.3)

# Add vertical lines for interval bounds
ax2.axvline(x=lower_bound, color='r', linestyle='--', linewidth=2,
            label=f'90% CI: [{lower_bound:.2f}, {upper_bound:.2f}]')
ax2.axvline(x=upper_bound, color='r', linestyle='--', linewidth=2)

ax2.set_title('2. 90% Credible Interval', fontsize=12)
ax2.set_xlabel('θ (probability of heads)', fontsize=10)
ax2.set_ylabel('Posterior Density', fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# 3. Bayesian updating subplot
ax3 = plt.subplot(gs[1, 0])
a_prior, b_prior = 2, 2
data_sequence = [(1, 1), (2, 3), (5, 6), (8, 9), (12, 12)]

for i, (cum_h, cum_n) in enumerate(data_sequence):
    # Calculate posterior parameters
    a_post = a_prior + cum_h
    b_post = b_prior + (cum_n - cum_h)
    
    # Create distribution
    dist = beta(a_post, b_post)
    
    # Plot distribution
    label = "Prior" if i == 0 else f"After {cum_n} flips"
    ax3.plot(x, dist.pdf(x), f'{colors[i]}-', linewidth=2, label=label)

ax3.set_title('3. Bayesian Updating Process', fontsize=12)
ax3.set_xlabel('θ (probability of heads)', fontsize=10)
ax3.set_ylabel('Density', fontsize=10)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# 4. Expected value subplot
ax4 = plt.subplot(gs[1, 1])
a, b = 5, 2
a_post, b_post = a + h, b + (n - h)
expected_value = a_post / (a_post + b_post)

posterior = beta(a_post, b_post)
ax4.plot(x, posterior.pdf(x), 'g-', linewidth=2, 
         label=f'Posterior with Beta(5,2) prior')
ax4.axvline(x=expected_value, color='r', linestyle='-', linewidth=2,
            label=f'Expected value: {expected_value:.4f}')

ax4.set_title('4. Expected Value of θ', fontsize=12)
ax4.set_xlabel('θ (probability of heads)', fontsize=10)
ax4.set_ylabel('Posterior Density', fontsize=10)
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

plt.suptitle('Visual Bayesian Inference and Prior Selection - Solution Summary', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle

file_path = os.path.join(solutions_dir, "summary_visualization.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Summary visualization saved to: {file_path}")

print("\nAll solution visualizations saved in:", solutions_dir) 