import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_5_Quiz_6")
os.makedirs(save_dir, exist_ok=True)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Introduction to the problem
print_step_header(1, "Understanding the Question")
print("Question 6: Evaluate whether each of the following statements is TRUE or FALSE.")
print("Justify your answer with a brief explanation.")
print("1. In Bayesian statistics, the posterior distribution represents our updated belief about a parameter after observing data.")
print("2. Conjugate priors always lead to the most accurate Bayesian inference results.")
print("3. Bayesian credible intervals and frequentist confidence intervals have identical interpretations.")
print("4. The posterior predictive distribution incorporates both the uncertainty in the parameter estimates and the inherent randomness in generating new data.")
print("5. Hierarchical Bayesian models are useful only when we have a large amount of data.")

# Statement 1: Posterior distributions represent updated belief
print_step_header(2, "Statement 1: Posterior Distribution as Updated Belief")
print("STATEMENT: In Bayesian statistics, the posterior distribution represents our updated belief about a parameter after observing data.")
print("\nEXPLANATION:")
print("This statement is TRUE. The fundamental concept of Bayesian inference is updating prior beliefs with data.")
print("The posterior distribution is obtained by applying Bayes' theorem:")
print("P(θ|data) ∝ P(data|θ) × P(θ)")
print("where:")
print("- P(θ|data) is the posterior distribution - our updated belief about parameter θ given the observed data")
print("- P(data|θ) is the likelihood function - how probable the observed data is for different values of θ")
print("- P(θ) is the prior distribution - our initial belief about parameter θ before seeing the data")
print("\nThe posterior distribution mathematically combines our prior knowledge with the information from the data.")

# Visualize the Bayesian updating process
# Example: Coin flip with Beta prior, Binomial likelihood
theta = np.linspace(0, 1, 1000)
prior_alpha, prior_beta = 2, 2
heads, tails = 7, 3
posterior_alpha = prior_alpha + heads
posterior_beta = prior_beta + tails

# Calculate prior, likelihood, and posterior
prior = stats.beta.pdf(theta, prior_alpha, prior_beta)
likelihood = stats.binom.pmf(heads, heads + tails, theta) * (10000 / max(stats.binom.pmf(heads, heads + tails, theta) * np.ones_like(theta)))
posterior = stats.beta.pdf(theta, posterior_alpha, posterior_beta)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(theta, prior, 'b--', linewidth=2, label='Prior: Beta(2, 2)')
plt.plot(theta, likelihood, 'g-', linewidth=2, label='Scaled Likelihood: 7 heads in 10 flips')
plt.plot(theta, posterior, 'r-', linewidth=3, label='Posterior: Beta(9, 5)')
plt.fill_between(theta, 0, posterior, alpha=0.1, color='red')

plt.annotate('Prior Mean\n0.5', xy=(0.5, stats.beta.pdf(0.5, prior_alpha, prior_beta)), 
             xytext=(0.35, 3), arrowprops=dict(facecolor='blue', shrink=0.05))
plt.annotate('MLE\n0.7', xy=(0.7, stats.binom.pmf(heads, heads + tails, 0.7) * (10000 / max(stats.binom.pmf(heads, heads + tails, theta) * np.ones_like(theta)))), 
             xytext=(0.75, 3), arrowprops=dict(facecolor='green', shrink=0.05))
plt.annotate('Posterior Mean\n0.64', xy=(9/14, stats.beta.pdf(9/14, posterior_alpha, posterior_beta)), 
             xytext=(0.55, 4), arrowprops=dict(facecolor='red', shrink=0.05))

plt.xlabel('θ (Probability of Heads)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title('Bayesian Updating: Prior to Posterior', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

file_path = os.path.join(save_dir, "statement1_bayesian_updating.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Statement 2: Conjugate priors and accuracy
print_step_header(3, "Statement 2: Conjugate Priors and Accuracy")
print("STATEMENT: Conjugate priors always lead to the most accurate Bayesian inference results.")
print("\nEXPLANATION:")
print("This statement is FALSE. While conjugate priors are mathematically convenient, they don't always provide the most accurate results.")
print("\nConjugate priors are chosen so that the posterior distribution follows the same parametric form as the prior distribution.")
print("For example, the Beta distribution is conjugate to the Binomial likelihood, leading to a Beta posterior distribution.")
print("\nHowever, accuracy in Bayesian inference depends on how well the prior represents actual prior knowledge:")
print("1. If our true prior beliefs don't match the conjugate form, forcing a conjugate prior may reduce accuracy")
print("2. Non-conjugate priors that better represent actual knowledge can lead to more accurate results")
print("3. With large datasets, the likelihood dominates and the choice of prior becomes less important")
print("4. In complex models, conjugate priors might be too restrictive to capture the true parameter relationships")

# Visualize misspecified conjugate prior vs. appropriate non-conjugate prior
theta = np.linspace(0, 1, 1000)
true_theta = 0.7

# Define different priors
conjugate_prior = stats.beta.pdf(theta, 1, 5)  # Misspecified conjugate Beta prior
nonconjugate_prior = np.exp(-10 * (theta - 0.7)**2)  # More accurate non-conjugate prior (Gaussian-like)
nonconjugate_prior = nonconjugate_prior / np.sum(nonconjugate_prior) * 1000 * 0.001  # Normalize

# With small data
heads_small, tails_small = 2, 1
likelihood_small = theta**heads_small * (1-theta)**tails_small
likelihood_small = likelihood_small / np.max(likelihood_small) * 5  # Scale for visibility

# Resulting posteriors (unnormalized)
conj_posterior_small = conjugate_prior * likelihood_small
nonconj_posterior_small = nonconjugate_prior * likelihood_small
# Normalize for comparison
conj_posterior_small = conj_posterior_small / np.max(conj_posterior_small) * 5
nonconj_posterior_small = nonconj_posterior_small / np.max(nonconj_posterior_small) * 5

# With large data
heads_large, tails_large = 70, 30
likelihood_large = theta**heads_large * (1-theta)**tails_large
likelihood_large = likelihood_large / np.max(likelihood_large) * 5  # Scale for visibility

# Resulting posteriors (unnormalized)
conj_posterior_large = conjugate_prior * likelihood_large
nonconj_posterior_large = nonconjugate_prior * likelihood_large
# Normalize for comparison
conj_posterior_large = conj_posterior_large / np.max(conj_posterior_large) * 5
nonconj_posterior_large = nonconj_posterior_large / np.max(nonconj_posterior_large) * 5

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Small data scenario
axs[0, 0].plot(theta, conjugate_prior, 'b-', label='Misspecified Conjugate Prior')
axs[0, 0].plot(theta, nonconjugate_prior, 'g-', label='Appropriate Non-conjugate Prior')
axs[0, 0].axvline(x=true_theta, color='r', linestyle='--', label='True θ = 0.7')
axs[0, 0].set_title('Different Types of Priors', fontsize=12)
axs[0, 0].set_ylabel('Density', fontsize=10)
axs[0, 0].legend()

axs[0, 1].plot(theta, likelihood_small, 'k-', label='Likelihood (Small Data)')
axs[0, 1].axvline(x=true_theta, color='r', linestyle='--', label='True θ = 0.7')
axs[0, 1].set_title('Likelihood from Small Dataset', fontsize=12)
axs[0, 1].legend()

axs[1, 0].plot(theta, conj_posterior_small, 'b-', label='Posterior with Conjugate Prior')
axs[1, 0].plot(theta, nonconj_posterior_small, 'g-', label='Posterior with Non-conjugate Prior')
axs[1, 0].axvline(x=true_theta, color='r', linestyle='--', label='True θ = 0.7')
axs[1, 0].set_title('Posteriors with Small Data', fontsize=12)
axs[1, 0].set_xlabel('θ', fontsize=10)
axs[1, 0].set_ylabel('Density', fontsize=10)
axs[1, 0].legend()

axs[1, 1].plot(theta, conj_posterior_large, 'b-', label='Posterior with Conjugate Prior')
axs[1, 1].plot(theta, nonconj_posterior_large, 'g-', label='Posterior with Non-conjugate Prior')
axs[1, 1].axvline(x=true_theta, color='r', linestyle='--', label='True θ = 0.7')
axs[1, 1].set_title('Posteriors with Large Data', fontsize=12)
axs[1, 1].set_xlabel('θ', fontsize=10)
axs[1, 1].legend()

plt.tight_layout()
file_path = os.path.join(save_dir, "statement2_conjugate_accuracy.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Statement 3: Credible vs. Confidence Intervals
print_step_header(4, "Statement 3: Bayesian vs. Frequentist Intervals")
print("STATEMENT: Bayesian credible intervals and frequentist confidence intervals have identical interpretations.")
print("\nEXPLANATION:")
print("This statement is FALSE. Bayesian credible intervals and frequentist confidence intervals have fundamentally different interpretations.")
print("\nBayesian 95% credible interval:")
print("- A range within which the parameter has a 95% probability of lying, given the observed data and prior")
print("- Makes direct probability statements about where the parameter is likely to be")
print("- Is conditional on the observed data")
print("- Example interpretation: 'Given our data, there's a 95% probability that the parameter lies within this interval'")
print("\nFrequentist 95% confidence interval:")
print("- A range calculated using a procedure that will contain the true parameter in 95% of repeated experiments")
print("- Makes statements about the procedure, not the parameter directly")
print("- Is based on hypothetical repeated sampling")
print("- Example interpretation: 'If we repeat the experiment many times, 95% of the calculated intervals will contain the true parameter'")
print("\nThese differences reflect the fundamental philosophical distinction between Bayesian and frequentist paradigms.")

# Visualize the differences between credible and confidence intervals
np.random.seed(42)
true_mean = 5
std_dev = 2
prior_mean = 6
prior_std = 1.5

# Function to generate samples and calculate intervals
def generate_sample_and_intervals(n=20):
    # Generate sample
    sample = np.random.normal(true_mean, std_dev, n)
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)
    
    # Frequentist confidence interval
    conf_margin = 1.96 * sample_std / np.sqrt(n)
    conf_lower = sample_mean - conf_margin
    conf_upper = sample_mean + conf_margin
    
    # Bayesian posterior (assuming normal-normal model)
    posterior_var = 1 / (1/prior_std**2 + n/std_dev**2)
    posterior_mean = posterior_var * (prior_mean/prior_std**2 + n*sample_mean/std_dev**2)
    posterior_std = np.sqrt(posterior_var)
    
    # Bayesian credible interval
    cred_lower = posterior_mean - 1.96 * posterior_std
    cred_upper = posterior_mean + 1.96 * posterior_std
    
    return {
        'sample_mean': sample_mean,
        'conf_lower': conf_lower,
        'conf_upper': conf_upper,
        'posterior_mean': posterior_mean,
        'cred_lower': cred_lower,
        'cred_upper': cred_upper,
        'contains_true_conf': conf_lower <= true_mean <= conf_upper,
        'contains_true_cred': cred_lower <= true_mean <= cred_upper
    }

# Generate multiple samples and intervals
n_experiments = 50
results = [generate_sample_and_intervals() for _ in range(n_experiments)]

# Create a figure showing the difference
plt.figure(figsize=(12, 10))

# Setup the grid
gs = GridSpec(2, 2, height_ratios=[2, 1])
ax1 = plt.subplot(gs[0, :])
ax2 = plt.subplot(gs[1, 0])
ax3 = plt.subplot(gs[1, 1])

# Plot the intervals for each experiment
for i, res in enumerate(results):
    # Plot confidence intervals
    color = 'green' if res['contains_true_conf'] else 'red'
    ax1.plot([i, i], [res['conf_lower'], res['conf_upper']], 'o-', color=color, alpha=0.5)
    
    # Plot credible intervals
    color = 'green' if res['contains_true_cred'] else 'red'
    ax1.plot([i+0.2, i+0.2], [res['cred_lower'], res['cred_upper']], 's-', color=color, alpha=0.5)

# Add true mean line and legend
ax1.axhline(y=true_mean, color='black', linestyle='--', label='True Mean')
ax1.plot([], [], 'o-', color='green', label='Confidence Interval (Contains True)')
ax1.plot([], [], 'o-', color='red', label='Confidence Interval (Misses True)')
ax1.plot([], [], 's-', color='green', label='Credible Interval (Contains True)')
ax1.plot([], [], 's-', color='red', label='Credible Interval (Misses True)')

ax1.set_xlabel('Experiment Number', fontsize=10)
ax1.set_ylabel('Interval Range', fontsize=10)
ax1.set_title('Comparison of 95% Confidence and Credible Intervals Across 50 Experiments', fontsize=12)
ax1.legend(loc='upper right')
ax1.set_xlim(-1, n_experiments)

# Calculate coverage rate
conf_coverage = sum(res['contains_true_conf'] for res in results) / n_experiments
cred_coverage = sum(res['contains_true_cred'] for res in results) / n_experiments

# Add illustrative distributions for confidence interval
x = np.linspace(2, 8, 1000)
sample_dist = stats.norm.pdf(x, 5, 2/np.sqrt(20))
ax2.plot(x, sample_dist, 'b-', label='Sampling Distribution')
ax2.axvline(x=true_mean, color='black', linestyle='--', label='True Mean')
ax2.axvline(x=true_mean - 1.96*2/np.sqrt(20), color='green', linestyle='-', label='95% CI Bounds')
ax2.axvline(x=true_mean + 1.96*2/np.sqrt(20), color='green', linestyle='-')
ax2.set_title(f'Frequentist Perspective: {conf_coverage:.1%} Coverage', fontsize=12)
ax2.legend()

# Add illustrative distributions for credible interval
posterior_dist = stats.norm.pdf(x, results[0]['posterior_mean'], np.sqrt(1/(1/prior_std**2 + 20/std_dev**2)))
ax3.plot(x, posterior_dist, 'r-', label='Posterior Distribution')
ax3.axvline(x=true_mean, color='black', linestyle='--', label='True Mean')
ax3.axvline(x=results[0]['cred_lower'], color='purple', linestyle='-', label='95% Credible Bounds')
ax3.axvline(x=results[0]['cred_upper'], color='purple', linestyle='-')
ax3.set_title(f'Bayesian Perspective: {cred_coverage:.1%} Coverage', fontsize=12)
ax3.legend()

plt.tight_layout()
file_path = os.path.join(save_dir, "statement3_interval_comparison.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Statement 4: Posterior Predictive Distribution
print_step_header(5, "Statement 4: Posterior Predictive Distribution")
print("STATEMENT: The posterior predictive distribution incorporates both the uncertainty in the parameter estimates and the inherent randomness in generating new data.")
print("\nEXPLANATION:")
print("This statement is TRUE. The posterior predictive distribution accounts for two sources of uncertainty.")
print("\nThe posterior predictive distribution is given by:")
print("p(y_new | data) = ∫ p(y_new | θ) p(θ | data) dθ")
print("\nThis distribution incorporates:")
print("1. Parameter uncertainty - by integrating over the posterior distribution p(θ | data)")
print("2. Inherent randomness - through the likelihood function p(y_new | θ) which models the stochastic data-generating process")
print("\nThis makes the posterior predictive distribution wider than predictions made using just a point estimate.")
print("It provides more realistic uncertainty quantification for future observations by accounting for all sources of uncertainty.")

# Visualize posterior predictive distribution
np.random.seed(42)

# Normal-Normal example (known variance)
mu_true = 10
sigma = 2

# Generate observed data
n_obs = 5
data = np.random.normal(mu_true, sigma, n_obs)
data_mean = np.mean(data)

# Prior
mu_prior = 8
sigma_prior = 3

# Posterior
sigma_posterior_sq = 1 / (1/sigma_prior**2 + n_obs/sigma**2)
mu_posterior = sigma_posterior_sq * (mu_prior/sigma_prior**2 + n_obs*data_mean/sigma**2)
sigma_posterior = np.sqrt(sigma_posterior_sq)

# Generate posterior samples
n_samples = 1000
posterior_samples = np.random.normal(mu_posterior, sigma_posterior, n_samples)

# Generate predictive samples (for each posterior sample, generate a predictive sample)
predictive_samples = np.random.normal(posterior_samples, sigma, n_samples)

# Plot
plt.figure(figsize=(10, 6))

# Plot distributions
x = np.linspace(0, 20, 1000)
posterior_pdf = stats.norm.pdf(x, mu_posterior, sigma_posterior)
fixed_param_predictive_pdf = stats.norm.pdf(x, mu_posterior, sigma)
plt.plot(x, posterior_pdf, 'b-', linewidth=2, label='Posterior Distribution p(θ|data)')
plt.plot(x, fixed_param_predictive_pdf, 'g--', linewidth=2, label='Fixed-Parameter Prediction p(y|θ=E[θ|data])')

# Plot histogram of posterior predictive samples
plt.hist(predictive_samples, bins=30, density=True, alpha=0.3, color='red', label='Posterior Predictive p(y|data)')

# Add annotations
plt.annotate('Parameter\nUncertainty', xy=(mu_posterior-1.5*sigma_posterior, posterior_pdf[300]), 
             xytext=(mu_posterior-5, posterior_pdf[300]), 
             arrowprops=dict(facecolor='blue', shrink=0.05))

plt.annotate('Inherent\nRandomness', xy=(mu_posterior+2*sigma, fixed_param_predictive_pdf[700]), 
             xytext=(mu_posterior+4, fixed_param_predictive_pdf[700]), 
             arrowprops=dict(facecolor='green', shrink=0.05))

plt.annotate('Combined\nUncertainty', xy=(mu_posterior+2.5*sigma, 0.08), 
             xytext=(mu_posterior+4, 0.05), 
             arrowprops=dict(facecolor='red', shrink=0.05))

plt.xlabel('Value', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title('Posterior Predictive Distribution Incorporates Multiple Sources of Uncertainty', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

file_path = os.path.join(save_dir, "statement4_predictive_distribution.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Statement 5: Hierarchical Bayesian Models
print_step_header(6, "Statement 5: Hierarchical Bayesian Models")
print("STATEMENT: Hierarchical Bayesian models are useful only when we have a large amount of data.")
print("\nEXPLANATION:")
print("This statement is FALSE. Hierarchical Bayesian models are particularly valuable when we have limited data per group or entity.")
print("\nHierarchical (or multilevel) models:")
print("1. Allow sharing of information across groups through partial pooling")
print("2. Reduce overfitting for groups with small sample sizes")
print("3. Provide more regularized and stable estimates when data is sparse")
print("4. Balance between complete pooling (treating all groups as identical) and no pooling (treating all groups as independent)")
print("\nIn fact, with large amounts of data for each group, simple non-hierarchical models may perform well.")
print("It's precisely when we have limited data that hierarchical models provide the greatest benefit by borrowing strength across groups.")

# Visualize the advantage of hierarchical modeling with limited data
np.random.seed(42)

# Generate data for multiple groups with varying sample sizes
group_means = [5, 7, 9, 11, 13]
n_groups = len(group_means)
group_stds = [2] * n_groups
group_sample_sizes = [3, 5, 30, 7, 4]  # Intentionally unbalanced

# Generate data
group_data = []
for i in range(n_groups):
    group_data.append(np.random.normal(group_means[i], group_stds[i], group_sample_sizes[i]))

# Calculate estimates under different approaches
# No pooling: use only group-specific data
no_pooling_estimates = [np.mean(group) for group in group_data]
no_pooling_se = [np.std(group, ddof=1) / np.sqrt(len(group)) for group in group_data]

# Complete pooling: ignore group structure
all_data = np.concatenate(group_data)
complete_pooling_estimate = np.mean(all_data)
complete_pooling_estimates = [complete_pooling_estimate] * n_groups

# Partial pooling (simplified hierarchical modeling)
# We'll simulate this with a weighted average between no pooling and complete pooling
grand_mean = np.mean(all_data)
total_n = sum(group_sample_sizes)
partial_pooling_estimates = []

for i in range(n_groups):
    # Weight based on sample size - smaller samples pull more toward the grand mean
    weight = min(0.5, group_sample_sizes[i] / total_n * 5)  # Cap weight at 0.5 for visualization
    partial_estimate = weight * no_pooling_estimates[i] + (1 - weight) * grand_mean
    partial_pooling_estimates.append(partial_estimate)

# True hierarchical modeling would use full Bayesian inference with MCMC

# Plot the comparison
plt.figure(figsize=(12, 8))

# Plot true group means
plt.scatter(range(n_groups), group_means, s=100, color='black', marker='*', label='True Group Means')

# Plot sample sizes as text
for i in range(n_groups):
    plt.text(i, group_means[i] - 0.5, f"n={group_sample_sizes[i]}", ha='center')

# Plot the estimates
x = np.arange(n_groups)
width = 0.25

plt.bar(x - width, no_pooling_estimates, width, alpha=0.6, color='blue', label='No Pooling (Group-specific)')
plt.bar(x, partial_pooling_estimates, width, alpha=0.6, color='green', label='Hierarchical (Partial Pooling)')
plt.bar(x + width, complete_pooling_estimates, width, alpha=0.6, color='red', label='Complete Pooling (Ignore Groups)')

# Add error bars for no pooling to show estimation uncertainty
plt.errorbar(x - width, no_pooling_estimates, yerr=1.96 * np.array(no_pooling_se), fmt='none', color='black', capsize=5)

# Annotations
plt.annotate('High uncertainty\nwith small n', xy=(0, no_pooling_estimates[0]), 
             xytext=(0, no_pooling_estimates[0] - 2), ha='center',
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.annotate('Hierarchical model\nreduces extreme estimates', xy=(0, partial_pooling_estimates[0]), 
             xytext=(-1, partial_pooling_estimates[0] - 1), ha='left',
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.annotate('With large n, estimates\nconverge to true mean', xy=(2, no_pooling_estimates[2]), 
             xytext=(2, no_pooling_estimates[2] + 1.5), ha='center',
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.xticks(range(n_groups), [f'Group {i+1}' for i in range(n_groups)])
plt.xlabel('Group', fontsize=12)
plt.ylabel('Estimated Mean', fontsize=12)
plt.title('Effect of Different Modeling Approaches with Limited Data per Group', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
file_path = os.path.join(save_dir, "statement5_hierarchical_models.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Summary of answers
print_step_header(7, "Summary of Answers")
print("1. In Bayesian statistics, the posterior distribution represents our updated belief about a parameter after observing data. - TRUE")
print("2. Conjugate priors always lead to the most accurate Bayesian inference results. - FALSE")
print("3. Bayesian credible intervals and frequentist confidence intervals have identical interpretations. - FALSE")
print("4. The posterior predictive distribution incorporates both the uncertainty in the parameter estimates and the inherent randomness in generating new data. - TRUE")
print("5. Hierarchical Bayesian models are useful only when we have a large amount of data. - FALSE")

# Conclusion
print_step_header(8, "Conclusion")
print("These concepts highlight key aspects of Bayesian statistical thinking:")
print("- Bayesian inference updates prior beliefs with observed data to form posterior distributions")
print("- The choice of prior is important and should be based on actual knowledge, not just mathematical convenience")
print("- Bayesian and frequentist approaches differ fundamentally in how they interpret probability and uncertainty")
print("- Bayesian methods naturally handle multiple sources of uncertainty in predictions")
print("- Hierarchical models provide powerful tools for analyzing grouped data, especially with limited samples per group")
print("\nUnderstanding these concepts is essential for correctly applying Bayesian methods in machine learning and statistical analysis.") 