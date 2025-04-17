import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from scipy.stats import beta

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_5_Quiz_5")
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

print("Given:")
print("- We want to estimate the probability p of an observation belonging to class 1")
print("- We have observed 5 instances belonging to class 1 out of 20 total observations")
print()
print("We need to:")
print("1. Calculate the posterior distribution using a uniform prior Beta(1,1) for p")
print("2. Find the posterior mean of p")
print("3. Determine how the posterior would change with an informative prior Beta(10,30)")
print("4. Explain the practical significance of using an informative prior vs a uniform prior")
print()

# Define the observed data
successes = 5  # Number of class 1 instances
total = 20     # Total number of observations
failures = total - successes  # Number of class 0 instances

print(f"Class 1 instances: {successes}")
print(f"Class 0 instances: {failures}")
print(f"Total observations: {total}")
print(f"Observed proportion: {successes/total:.4f}")
print()

# Step 2: Beta-Binomial conjugate prior relationship
print_step_header(2, "Beta-Binomial Conjugate Prior Relationship")

print("For a binomial likelihood with parameter p:")
print("- Likelihood: p(data|p) = Binomial(k=successes, n=total, p)")
print("- Prior: p(p) = Beta(α, β)")
print()
print("When we observe k successes out of n trials:")
print("- The posterior is: p(p|data) ∝ p(data|p) × p(p)")
print("- This gives us: p(p|data) = Beta(α + k, β + n - k)")
print()

# Step 3: Uniform Prior - Calculate posterior
print_step_header(3, "Calculating Posterior with Uniform Prior")

# Define the uniform prior parameters
alpha_uniform = 1
beta_uniform = 1

# Calculate the posterior parameters with uniform prior
alpha_post_uniform = alpha_uniform + successes
beta_post_uniform = beta_uniform + failures

print(f"Uniform prior: Beta({alpha_uniform}, {beta_uniform})")
print(f"Posterior with uniform prior: Beta({alpha_post_uniform}, {beta_post_uniform})")
print()

# Calculate posterior statistics
post_mean_uniform = alpha_post_uniform / (alpha_post_uniform + beta_post_uniform)
post_mode_uniform = (alpha_post_uniform - 1) / (alpha_post_uniform + beta_post_uniform - 2) if alpha_post_uniform > 1 and beta_post_uniform > 1 else None
post_var_uniform = (alpha_post_uniform * beta_post_uniform) / ((alpha_post_uniform + beta_post_uniform)**2 * (alpha_post_uniform + beta_post_uniform + 1))

print(f"Posterior mean with uniform prior: E[p|data] = {post_mean_uniform:.4f}")
if post_mode_uniform is not None:
    print(f"Posterior mode with uniform prior: {post_mode_uniform:.4f}")
print(f"Posterior variance with uniform prior: {post_var_uniform:.6f}")
print(f"Posterior standard deviation with uniform prior: {np.sqrt(post_var_uniform):.4f}")
print()

# Calculate 95% credible interval
lower_ci_uniform = beta.ppf(0.025, alpha_post_uniform, beta_post_uniform)
upper_ci_uniform = beta.ppf(0.975, alpha_post_uniform, beta_post_uniform)
print(f"95% Credible interval with uniform prior: [{lower_ci_uniform:.4f}, {upper_ci_uniform:.4f}]")
print()

# Create visualization of the prior, likelihood, and posterior
plt.figure(figsize=(12, 7))
p_range = np.linspace(0, 1, 1000)

# Prior (uniform)
prior_pdf_uniform = beta.pdf(p_range, alpha_uniform, beta_uniform)
plt.plot(p_range, prior_pdf_uniform, 'b--', linewidth=2, 
         label=f'Uniform Prior: Beta({alpha_uniform}, {beta_uniform})')

# Likelihood 
# We'll compute a scaled version of the likelihood for visualization
def scaled_binomial_likelihood(p, k, n):
    # Using log to handle numerical issues with factorials
    log_likelihood = k * np.log(p) + (n - k) * np.log(1 - p)
    # Normalize to a reasonable scale for plotting
    likelihood = np.exp(log_likelihood - np.max(log_likelihood))
    return likelihood / np.max(likelihood)  # Scale to max = 1 for better visualization

likelihood = scaled_binomial_likelihood(p_range, successes, total)
plt.plot(p_range, likelihood, 'g-', linewidth=2, 
         label=f'Scaled Likelihood: Bin({successes}, {total}, p)')

# Posterior with uniform prior
posterior_pdf_uniform = beta.pdf(p_range, alpha_post_uniform, beta_post_uniform)
plt.plot(p_range, posterior_pdf_uniform, 'r-', linewidth=3, 
         label=f'Posterior with Uniform Prior: Beta({alpha_post_uniform}, {beta_post_uniform})')

# Add vertical lines for MLE and posterior mean
mle = successes / total
plt.axvline(x=mle, color='g', linestyle=':', linewidth=1.5, 
            label=f'MLE (sample proportion): {mle:.4f}')
plt.axvline(x=post_mean_uniform, color='r', linestyle=':', linewidth=1.5, 
            label=f'Posterior Mean: {post_mean_uniform:.4f}')

plt.title('Prior, Likelihood, and Posterior with Uniform Prior', fontsize=14)
plt.xlabel('p (Probability of Class 1)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "uniform_prior_analysis.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 4: Informative Prior - Calculate posterior
print_step_header(4, "Calculating Posterior with Informative Prior")

# Define the informative prior parameters
alpha_informative = 10
beta_informative = 30

# Calculate the posterior parameters with informative prior
alpha_post_informative = alpha_informative + successes
beta_post_informative = beta_informative + failures

print(f"Informative prior: Beta({alpha_informative}, {beta_informative})")
print(f"Prior mean: {alpha_informative / (alpha_informative + beta_informative):.4f}")
print(f"Posterior with informative prior: Beta({alpha_post_informative}, {beta_post_informative})")
print()

# Calculate posterior statistics
post_mean_informative = alpha_post_informative / (alpha_post_informative + beta_post_informative)
post_mode_informative = (alpha_post_informative - 1) / (alpha_post_informative + beta_post_informative - 2) if alpha_post_informative > 1 and beta_post_informative > 1 else None
post_var_informative = (alpha_post_informative * beta_post_informative) / ((alpha_post_informative + beta_post_informative)**2 * (alpha_post_informative + beta_post_informative + 1))

print(f"Posterior mean with informative prior: E[p|data] = {post_mean_informative:.4f}")
if post_mode_informative is not None:
    print(f"Posterior mode with informative prior: {post_mode_informative:.4f}")
print(f"Posterior variance with informative prior: {post_var_informative:.6f}")
print(f"Posterior standard deviation with informative prior: {np.sqrt(post_var_informative):.4f}")
print()

# Calculate 95% credible interval
lower_ci_informative = beta.ppf(0.025, alpha_post_informative, beta_post_informative)
upper_ci_informative = beta.ppf(0.975, alpha_post_informative, beta_post_informative)
print(f"95% Credible interval with informative prior: [{lower_ci_informative:.4f}, {upper_ci_informative:.4f}]")
print()

# Create visualization comparing the two posteriors
plt.figure(figsize=(12, 7))
p_range = np.linspace(0, 1, 1000)

# Uniform prior
prior_pdf_uniform = beta.pdf(p_range, alpha_uniform, beta_uniform)
plt.plot(p_range, prior_pdf_uniform, 'b--', linewidth=2, alpha=0.7,
         label=f'Uniform Prior: Beta({alpha_uniform}, {beta_uniform})')

# Informative prior
prior_pdf_informative = beta.pdf(p_range, alpha_informative, beta_informative)
plt.plot(p_range, prior_pdf_informative, 'c--', linewidth=2, alpha=0.7,
         label=f'Informative Prior: Beta({alpha_informative}, {beta_informative})')

# Posterior with uniform prior
posterior_pdf_uniform = beta.pdf(p_range, alpha_post_uniform, beta_post_uniform)
plt.plot(p_range, posterior_pdf_uniform, 'r-', linewidth=3, 
         label=f'Posterior with Uniform Prior: Beta({alpha_post_uniform}, {beta_post_uniform})')

# Posterior with informative prior
posterior_pdf_informative = beta.pdf(p_range, alpha_post_informative, beta_post_informative)
plt.plot(p_range, posterior_pdf_informative, 'm-', linewidth=3, 
         label=f'Posterior with Informative Prior: Beta({alpha_post_informative}, {beta_post_informative})')

# Add vertical lines for posterior means
plt.axvline(x=post_mean_uniform, color='r', linestyle=':', linewidth=1.5, 
            label=f'Posterior Mean (Uniform): {post_mean_uniform:.4f}')
plt.axvline(x=post_mean_informative, color='m', linestyle=':', linewidth=1.5, 
            label=f'Posterior Mean (Informative): {post_mean_informative:.4f}')

# Add MLE
plt.axvline(x=mle, color='g', linestyle=':', linewidth=1.5, 
            label=f'MLE: {mle:.4f}')

plt.title('Comparison of Posteriors with Different Priors', fontsize=14)
plt.xlabel('p (Probability of Class 1)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "prior_comparison.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 5: Interpretation of prior influence
print_step_header(5, "Interpreting Prior Influence")

print("Analysis of prior influence:")
print(f"1. Uniform Prior: Beta(1, 1)")
print(f"   - Prior weight: {alpha_uniform + beta_uniform} = 2 (equivalent to 2 pseudo-observations)")
print(f"   - Data weight: {successes + failures} = {total} (20 actual observations)")
print(f"   - Posterior: Beta({alpha_post_uniform}, {beta_post_uniform})")
print(f"   - Posterior Mean: {post_mean_uniform:.4f}")
print()
print(f"2. Informative Prior: Beta(10, 30)")
print(f"   - Prior weight: {alpha_informative + beta_informative} = 40 (equivalent to 40 pseudo-observations)")
print(f"   - Data weight: {successes + failures} = {total} (20 actual observations)")
print(f"   - Posterior: Beta({alpha_post_informative}, {beta_post_informative})")
print(f"   - Posterior Mean: {post_mean_informative:.4f}")
print()

# Calculate relative weights
uniform_prior_weight = (alpha_uniform + beta_uniform) / (alpha_uniform + beta_uniform + total)
uniform_data_weight = total / (alpha_uniform + beta_uniform + total)

informative_prior_weight = (alpha_informative + beta_informative) / (alpha_informative + beta_informative + total)
informative_data_weight = total / (alpha_informative + beta_informative + total)

print("Relative influence:")
print(f"1. With Uniform Prior:")
print(f"   - Prior influence: {uniform_prior_weight:.2%}")
print(f"   - Data influence: {uniform_data_weight:.2%}")
print()
print(f"2. With Informative Prior:")
print(f"   - Prior influence: {informative_prior_weight:.2%}")
print(f"   - Data influence: {informative_data_weight:.2%}")
print()

# Create visualization of the effective sample sizes and weights
plt.figure(figsize=(10, 6))

# Set up data for stacked bar chart
labels = ['Uniform Prior', 'Informative Prior']
prior_weights = [alpha_uniform + beta_uniform, alpha_informative + beta_informative]
data_weight = [total, total]

# Create stacked bar chart
plt.bar(labels, prior_weights, color='blue', alpha=0.7, label='Prior (Pseudo-observations)')
plt.bar(labels, data_weight, bottom=prior_weights, color='green', alpha=0.7, label='Data (Actual observations)')

# Add annotations
for i, label in enumerate(labels):
    total_height = prior_weights[i] + data_weight[i]
    prior_text = f"{prior_weights[i]} ({prior_weights[i]/total_height:.1%})"
    data_text = f"{data_weight[i]} ({data_weight[i]/total_height:.1%})"
    
    # Position the text in the middle of each segment
    plt.text(i, prior_weights[i]/2, prior_text, ha='center', va='center', color='white', fontweight='bold')
    plt.text(i, prior_weights[i] + data_weight[i]/2, data_text, ha='center', va='center', color='white', fontweight='bold')

plt.title('Effective Sample Sizes and Relative Weights', fontsize=14)
plt.ylabel('Number of Observations (Real + Pseudo)', fontsize=12)
plt.grid(True, axis='y')
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "effective_sample_sizes.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 6: Effect of increasing real observations
print_step_header(6, "Effect of Increasing Sample Size")

# Define different sample sizes to simulate
sample_sizes = [20, 100, 500]
success_rate = successes / total  # Use the same observed proportion for all samples

plt.figure(figsize=(12, 8))
p_range = np.linspace(0, 1, 1000)

# Original posteriors
plt.plot(p_range, posterior_pdf_uniform, 'r-', linewidth=2, alpha=0.7,
         label=f'Uniform Prior, n={total}: Beta({alpha_post_uniform}, {beta_post_uniform})')
plt.plot(p_range, posterior_pdf_informative, 'm-', linewidth=2, alpha=0.7,
         label=f'Informative Prior, n={total}: Beta({alpha_post_informative}, {beta_post_informative})')

# Simulate posteriors for larger sample sizes
colors_uniform = ['salmon', 'darkred', 'firebrick']
colors_informative = ['orchid', 'purple', 'indigo']

for i, n in enumerate(sample_sizes[1:]):  # Skip the first, which we already plotted
    # Calculate new number of successes and failures, keeping the same proportion
    new_successes = int(n * success_rate)
    new_failures = n - new_successes
    
    # Calculate new posteriors
    new_alpha_uniform = alpha_uniform + new_successes
    new_beta_uniform = beta_uniform + new_failures
    new_posterior_uniform = beta.pdf(p_range, new_alpha_uniform, new_beta_uniform)
    
    new_alpha_informative = alpha_informative + new_successes
    new_beta_informative = beta_informative + new_failures
    new_posterior_informative = beta.pdf(p_range, new_alpha_informative, new_beta_informative)
    
    # Plot new posteriors
    plt.plot(p_range, new_posterior_uniform, color=colors_uniform[i], linestyle='-', linewidth=2,
             label=f'Uniform Prior, n={n}: Beta({new_alpha_uniform}, {new_beta_uniform})')
    plt.plot(p_range, new_posterior_informative, color=colors_informative[i], linestyle='-', linewidth=2,
             label=f'Informative Prior, n={n}: Beta({new_alpha_informative}, {new_beta_informative})')

# Add vertical line for the true proportion
plt.axvline(x=success_rate, color='g', linestyle='--', linewidth=1.5, 
            label=f'Observed Proportion: {success_rate:.4f}')

plt.title('Effect of Sample Size on Posterior Distributions', fontsize=14)
plt.xlabel('p (Probability of Class 1)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "sample_size_effect.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 7: Practical significance of prior choice
print_step_header(7, "Practical Significance of Prior Choice")

print("Practical implications of prior choice in classification problems:")
print()
print("1. Informative priors are most influential when:")
print("   - Sample size is small (limited data)")
print("   - Prior knowledge is strong and reliable")
print("   - Class distributions are highly imbalanced")
print()
print("2. Using the informative prior Beta(10, 30) in our problem:")
print(f"   - Prior mean: {alpha_informative/(alpha_informative+beta_informative):.4f}")
print(f"   - Prior effective sample size: {alpha_informative+beta_informative}")
print(f"   - This prior represents a strong belief that class 1 is rarer (about 25%)")
print(f"   - Posterior mean: {post_mean_informative:.4f} (vs. {post_mean_uniform:.4f} with uniform prior)")
print(f"   - Credible interval: [{lower_ci_informative:.4f}, {upper_ci_informative:.4f}] (narrower than uniform)")
print()
print("3. Practical benefits of informative priors in classification:")
print("   - Regularizes predictions when data is limited")
print("   - Can help correct for class imbalance and sampling bias")
print("   - Provides more stable and robust probability estimates")
print("   - Especially useful for rare event prediction (fraud, disease, etc.)")
print()
print("4. When to prefer the uniform prior:")
print("   - When no reliable prior information exists")
print("   - When you want the data to 'speak for itself'")
print("   - With large datasets where the data will dominate anyway")
print("   - For maximum objectivity in reporting results")
print()

# Create individual visualizations for each threshold
thresholds = [0.2, 0.3, 0.4]

for threshold in thresholds:
    plt.figure(figsize=(10, 7))
    
    # Plot the posteriors
    plt.plot(p_range, posterior_pdf_uniform, 'r-', linewidth=3, 
             label=f'Posterior with Uniform Prior: Beta({alpha_post_uniform}, {beta_post_uniform})')
    plt.plot(p_range, posterior_pdf_informative, 'm-', linewidth=3, 
             label=f'Posterior with Informative Prior: Beta({alpha_post_informative}, {beta_post_informative})')
    
    # Calculate posterior probabilities of exceeding threshold
    # P(p > threshold | data, uniform prior)
    prob_exceed_uniform = 1 - beta.cdf(threshold, alpha_post_uniform, beta_post_uniform)
    
    # P(p > threshold | data, informative prior)
    prob_exceed_informative = 1 - beta.cdf(threshold, alpha_post_informative, beta_post_informative)
    
    # Add vertical line for the threshold
    plt.axvline(x=threshold, color='gray', linestyle='--', linewidth=2,
                label=f'Threshold = {threshold}')
    
    # Add annotations with arrows for uniform posterior
    plt.annotate(f"P(p > {threshold} | uniform) = {prob_exceed_uniform:.2f}", 
                xy=(threshold, posterior_pdf_uniform[np.abs(p_range - threshold).argmin()]), 
                xytext=(threshold + 0.1, 4.5),
                color='red', fontsize=10, fontweight='bold',
                arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, alpha=0.7, 
                               connectionstyle="arc3,rad=.3"))
    
    # Add annotations with arrows for informative posterior
    plt.annotate(f"P(p > {threshold} | informative) = {prob_exceed_informative:.2f}", 
                xy=(threshold, posterior_pdf_informative[np.abs(p_range - threshold).argmin()]), 
                xytext=(threshold - 0.1, 6.5),
                color='purple', fontsize=10, fontweight='bold',
                arrowprops=dict(facecolor='purple', shrink=0.05, width=1.5, alpha=0.7,
                               connectionstyle="arc3,rad=-.3"))
    
    plt.title(f'Decision-Making with Threshold {threshold}', fontsize=14)
    plt.xlabel('p (Probability of Class 1)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.ylim(0, 8)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    file_path = os.path.join(save_dir, f"decision_threshold_{int(threshold*10)}.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {file_path}")

# Create a combined visualization for all thresholds (keeping the original for reference)
plt.figure(figsize=(12, 8))

# Plot the posteriors
plt.plot(p_range, posterior_pdf_uniform, 'r-', linewidth=3, 
         label=f'Posterior with Uniform Prior: Beta({alpha_post_uniform}, {beta_post_uniform})')
plt.plot(p_range, posterior_pdf_informative, 'm-', linewidth=3, 
         label=f'Posterior with Informative Prior: Beta({alpha_post_informative}, {beta_post_informative})')

# Add vertical lines for thresholds with different styles
for threshold in thresholds:
    plt.axvline(x=threshold, color='gray', linestyle='--', linewidth=2,
                label=f'Threshold = {threshold}')

# Add annotations for uniform posterior probabilities
for threshold in thresholds:
    prob_exceed_uniform = 1 - beta.cdf(threshold, alpha_post_uniform, beta_post_uniform)
    if threshold == 0.2:
        xytext = (threshold, 4.2)
    elif threshold == 0.3:
        xytext = (threshold + 0.05, 3.8)
    else:  # threshold == 0.4
        xytext = (threshold + 0.05, 3.4)
    
    plt.annotate(f"P(p > {threshold} | uniform) = {prob_exceed_uniform:.2f}", 
                xy=(threshold, posterior_pdf_uniform[np.abs(p_range - threshold).argmin()]), 
                xytext=xytext,
                color='red', fontsize=10, fontweight='bold',
                arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, alpha=0.7, 
                               connectionstyle="arc3,rad=.3"))

# Add annotations for informative posterior probabilities
for threshold in thresholds:
    prob_exceed_informative = 1 - beta.cdf(threshold, alpha_post_informative, beta_post_informative)
    if threshold == 0.2:
        xytext = (threshold, 7.2)
    elif threshold == 0.3:
        xytext = (threshold - 0.05, 2.8)
    else:  # threshold == 0.4
        xytext = (threshold - 0.05, 1.8)
    
    plt.annotate(f"P(p > {threshold} | informative) = {prob_exceed_informative:.2f}", 
                xy=(threshold, posterior_pdf_informative[np.abs(p_range - threshold).argmin()]), 
                xytext=xytext,
                color='purple', fontsize=10, fontweight='bold',
                arrowprops=dict(facecolor='purple', shrink=0.05, width=1.5, alpha=0.7,
                               connectionstyle="arc3,rad=-.3"))

plt.title('Decision-Making with Different Priors', fontsize=14)
plt.xlabel('p (Probability of Class 1)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.ylim(0, 8)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "decision_thresholds_combined.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 8: Conclusion
print_step_header(8, "Conclusion and Summary")

print("Summary of findings:")
print(f"1. Using a uniform prior Beta(1, 1):")
print(f"   - Posterior: Beta({alpha_post_uniform}, {beta_post_uniform})")
print(f"   - Posterior mean: {post_mean_uniform:.4f}")
print(f"   - 95% Credible interval: [{lower_ci_uniform:.4f}, {upper_ci_uniform:.4f}]")
print()
print(f"2. Using an informative prior Beta(10, 30):")
print(f"   - Posterior: Beta({alpha_post_informative}, {beta_post_informative})")
print(f"   - Posterior mean: {post_mean_informative:.4f}")
print(f"   - 95% Credible interval: [{lower_ci_informative:.4f}, {upper_ci_informative:.4f}]")
print()
print("3. Comparison:")
print(f"   - The informative prior pulls the estimate toward {alpha_informative/(alpha_informative+beta_informative):.4f}")
print(f"   - This results in a posterior mean {post_mean_uniform-post_mean_informative:.4f} lower than with uniform prior")
print(f"   - The informative prior produces a narrower credible interval")
print()
print("4. Practical significance:")
print("   - Choice of prior matters most with small datasets")
print("   - Informative priors can improve predictions when prior knowledge is reliable")
print("   - As data accumulates, both approaches converge to the true parameter value")
print("   - The decision between priors should consider the application context and consequences") 