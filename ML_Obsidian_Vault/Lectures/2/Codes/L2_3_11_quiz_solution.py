import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from scipy.stats import norm

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_3_11")
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

print("Question 11: Evaluate whether each of the following statements is TRUE or FALSE.")
print("Justify your answer with a brief explanation.")
print()
print("1. The likelihood function represents the probability of observing the data given the parameters.")
print("2. If two estimators have the same variance, the one with lower bias will always have lower Mean Squared Error (MSE).")
print()

# Step 2: Analyze Statement 1 - Likelihood Function
print_step_header(2, "Analyzing Statement 1: The Likelihood Function")

# Create a visual explanation of the likelihood function
plt.figure(figsize=(10, 6))

# Generate some data from a normal distribution
np.random.seed(42)
true_mu = 5
true_sigma = 1.5
sample_size = 10
data = np.random.normal(true_mu, true_sigma, sample_size)

# Compute the likelihood for a range of parameter values
mu_range = np.linspace(2, 8, 1000)
likelihood = np.zeros_like(mu_range)

for i, mu in enumerate(mu_range):
    # Calculate the likelihood for this parameter value
    likelihood[i] = np.prod(norm.pdf(data, mu, true_sigma))

# Plot the likelihood function
plt.plot(mu_range, likelihood, 'b-', linewidth=2)
plt.axvline(x=true_mu, color='r', linestyle='--', label='True Parameter Value')
sample_mean = np.mean(data)
plt.axvline(x=sample_mean, color='g', linestyle='-', label='MLE (Sample Mean)')

plt.title('Likelihood Function L(μ|Data) for Normal Distribution', fontsize=14)
plt.xlabel('Parameter μ', fontsize=12)
plt.ylabel('Likelihood', fontsize=12)
plt.legend()
plt.grid(True)

# Highlight and annotate that this is not a PDF over the parameter
plt.annotate('Not a PDF over μ\n(area ≠ 1)', xy=(7, likelihood[900]), 
             xytext=(7, likelihood[900] * 1.5),
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='center', verticalalignment='center')

# Save the figure
file_path = os.path.join(save_dir, "likelihood_function.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Explain the difference between likelihood and probability
print("\nAnalysis of Statement 1:")
print("------------------------")
print("The likelihood function L(θ|x) and the probability function P(x|θ) are mathematically the same expression,")
print("but they have fundamentally different interpretations:")
print()
print("- P(x|θ): Probability of observing data x given fixed parameters θ")
print("  * This is a function of x, with θ fixed")
print("  * For discrete random variables, represents the probability mass function")
print("  * For continuous random variables, represents the probability density function")
print("  * Integrates/sums to 1 over all possible values of x")
print()
print("- L(θ|x): Likelihood of parameters θ given fixed observed data x")
print("  * This is a function of θ, with x fixed")
print("  * Does NOT generally integrate/sum to 1 over values of θ")
print("  * Not a proper probability distribution over θ")
print("  * Used for parameter estimation, not for calculating probabilities")
print()
print("Conclusion: Statement 1 is technically TRUE, but only when referring to P(x|θ) as a function of x.")
print("It's important to understand that the likelihood function L(θ|x) is not a probability distribution over θ.")
print("For Bayesian analysis, we need the posterior P(θ|x), which combines the likelihood with a prior P(θ).")

# Step 3: Create a visual comparison of likelihood, prior, and posterior
print_step_header(3, "Visual Comparison: Likelihood, Prior, and Posterior")

# Create a figure with multiple subplots
plt.figure(figsize=(12, 8))
gs = GridSpec(2, 2)

# Define parameter ranges and priors
mu_range = np.linspace(2, 8, 1000)
prior_mu = 4
prior_sigma = 1.0

# Calculate likelihood (same as before)
likelihood = np.zeros_like(mu_range)
for i, mu in enumerate(mu_range):
    likelihood[i] = np.prod(norm.pdf(data, mu, true_sigma))

# Define the prior (normal distribution)
prior = norm.pdf(mu_range, prior_mu, prior_sigma)

# Calculate the posterior (unnormalized and normalized)
unnormalized_posterior = likelihood * prior
posterior = unnormalized_posterior / np.trapz(unnormalized_posterior, mu_range)

# Plot the likelihood
ax1 = plt.subplot(gs[0, 0])
ax1.plot(mu_range, likelihood / np.max(likelihood), 'b-', linewidth=2)
ax1.axvline(x=sample_mean, color='g', linestyle='-', label='MLE')
ax1.set_title('Likelihood L(θ|x)', fontsize=12)
ax1.set_xlabel('Parameter θ', fontsize=10)
ax1.set_ylabel('Relative Magnitude', fontsize=10)
ax1.legend(loc='best')
ax1.grid(True)

# Plot the prior
ax2 = plt.subplot(gs[0, 1])
ax2.plot(mu_range, prior, 'r-', linewidth=2)
ax2.axvline(x=prior_mu, color='r', linestyle='--', label='Prior Mean')
ax2.set_title('Prior P(θ)', fontsize=12)
ax2.set_xlabel('Parameter θ', fontsize=10)
ax2.set_ylabel('Density', fontsize=10)
ax2.legend(loc='best')
ax2.grid(True)

# Plot the posterior
ax3 = plt.subplot(gs[1, :])
ax3.plot(mu_range, posterior, 'g-', linewidth=2, label='Posterior')
ax3.axvline(x=prior_mu, color='r', linestyle='--', label='Prior Mean')
ax3.axvline(x=sample_mean, color='b', linestyle='--', label='Sample Mean (MLE)')
posterior_mean = np.sum(mu_range * posterior) * (mu_range[1] - mu_range[0])
ax3.axvline(x=posterior_mean, color='g', linestyle='-', label='Posterior Mean')
ax3.set_title('Posterior P(θ|x) ∝ L(θ|x) × P(θ)', fontsize=12)
ax3.set_xlabel('Parameter θ', fontsize=10)
ax3.set_ylabel('Density', fontsize=10)
ax3.legend(loc='best')
ax3.grid(True)

# Add equations to explain the relationships
plt.figtext(0.5, 0.01, 'Bayes\' Theorem: P(θ|x) = P(x|θ)P(θ)/P(x) ∝ L(θ|x)P(θ)', 
            ha='center', fontsize=12, bbox={'facecolor':'white', 'alpha':0.8, 'pad':5})

plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# Save the figure
file_path = os.path.join(save_dir, "likelihood_vs_probability.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 4: Analyze Statement 2 - MSE, Bias, and Variance relationship
print_step_header(4, "Analyzing Statement 2: MSE, Bias, and Variance Relationship")

print("For an estimator θ̂ of a parameter θ, the Mean Squared Error (MSE) is defined as:")
print("MSE(θ̂) = E[(θ̂ - θ)²]")
print()
print("The MSE can be decomposed into bias and variance components:")
print("MSE(θ̂) = (Bias(θ̂))² + Var(θ̂)")
print("where Bias(θ̂) = E[θ̂] - θ")
print()

# Create visualizations of bias, variance, and MSE
plt.figure(figsize=(12, 8))

# Define a parameter θ for which we'll calculate MSE for different estimators
theta = 5.0  # True parameter value

# Create a range of possible bias values
bias_range = np.linspace(-2, 2, 100)

# Set a fixed variance for analysis of statement 2
fixed_variance = 1.0

# Calculate MSE for different bias values with fixed variance
mse_values = np.square(bias_range) + fixed_variance

# Plot MSE as a function of bias (with fixed variance)
plt.subplot(2, 1, 1)
plt.plot(bias_range, mse_values, 'b-', linewidth=2, label=f'MSE (Variance={fixed_variance})')
plt.plot(bias_range, np.square(bias_range), 'r--', linewidth=2, label='Squared Bias Component')
plt.axhline(y=fixed_variance, color='g', linestyle='--', linewidth=2, label='Variance Component')
plt.axvline(x=0, color='k', linestyle='--')
plt.text(0.1, 0.5, "Unbiased\nEstimator", transform=plt.gca().transAxes, fontsize=10)

plt.title('MSE as a Function of Bias (Fixed Variance)', fontsize=14)
plt.xlabel('Bias', fontsize=12)
plt.ylabel('MSE', fontsize=12)
plt.legend()
plt.grid(True)

# Now compare two estimators with different biases but same variance
bias_A = 0.5  # Bias of estimator A
bias_B = 0.0  # Bias of estimator B (unbiased)
variance = 1.0  # Same variance for both

# Calculate MSE
mse_A = bias_A**2 + variance
mse_B = bias_B**2 + variance

# For illustration, create normal distributions centered at E[θ̂]
x_range = np.linspace(theta - 4, theta + 4, 1000)
estimator_A_pdf = norm.pdf(x_range, theta + bias_A, np.sqrt(variance))
estimator_B_pdf = norm.pdf(x_range, theta + bias_B, np.sqrt(variance))

# Plot the sampling distributions of the estimators
plt.subplot(2, 1, 2)
plt.fill_between(x_range, estimator_A_pdf, alpha=0.3, color='red', label=f'Estimator A (Bias={bias_A}, Var={variance})')
plt.fill_between(x_range, estimator_B_pdf, alpha=0.3, color='blue', label=f'Estimator B (Bias={bias_B}, Var={variance})')
plt.axvline(x=theta, color='k', linestyle='-', linewidth=2, label='True θ')
plt.axvline(x=theta + bias_A, color='r', linestyle='--', linewidth=2, label='E[θ̂A]')
plt.axvline(x=theta + bias_B, color='b', linestyle='--', linewidth=2, label='E[θ̂B]')

# Annotate the MSE
plt.annotate(f'MSE(A) = {mse_A:.2f}', xy=(theta + bias_A, 0.1), xytext=(theta + bias_A, 0.2),
             arrowprops=dict(facecolor='red', shrink=0.05), color='red',
             horizontalalignment='center', verticalalignment='bottom')
plt.annotate(f'MSE(B) = {mse_B:.2f}', xy=(theta + bias_B, 0.1), xytext=(theta + bias_B, 0.3),
             arrowprops=dict(facecolor='blue', shrink=0.05), color='blue',
             horizontalalignment='center', verticalalignment='bottom')

plt.title('Sampling Distributions of Two Estimators (Same Variance)', fontsize=14)
plt.xlabel('Estimator Value', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.grid(True)

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "mse_bias_variance.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Now visualize a counterexample where MSE is the same despite different biases
plt.figure(figsize=(10, 6))

# Create two estimators with different biases and different variances but same MSE
bias_C = 0.5  # Small bias
variance_C = 0.75  # Higher variance
mse_C = bias_C**2 + variance_C

bias_D = 1.0  # Larger bias
variance_D = 0.0  # Zero variance (deterministic estimator)
mse_D = bias_D**2 + variance_D

# Create PDFs for the estimators
x_range = np.linspace(theta - 4, theta + 4, 1000)
estimator_C_pdf = norm.pdf(x_range, theta + bias_C, np.sqrt(variance_C))
# For estimator D, we use a very narrow normal (since it's deterministic)
estimator_D_pdf = norm.pdf(x_range, theta + bias_D, 0.05)

# Plot the sampling distributions
plt.fill_between(x_range, estimator_C_pdf, alpha=0.3, color='purple', 
                label=f'Estimator C (Bias={bias_C}, Var={variance_C}, MSE={mse_C:.2f})')
plt.fill_between(x_range, estimator_D_pdf, alpha=0.3, color='orange', 
                label=f'Estimator D (Bias={bias_D}, Var={variance_D}, MSE={mse_D:.2f})')
plt.axvline(x=theta, color='k', linestyle='-', linewidth=2, label='True θ')

plt.title('Same MSE, Different Bias-Variance Combinations', fontsize=14)
plt.xlabel('Estimator Value', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "same_mse_different_bias_variance.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Analyze statement 2
print("\nAnalysis of Statement 2:")
print("------------------------")
print("Statement: If two estimators have the same variance, the one with lower bias will always have lower Mean Squared Error (MSE).")
print()
print("For an estimator θ̂, the MSE can be decomposed as:")
print("MSE(θ̂) = (Bias(θ̂))² + Var(θ̂)")
print()
print("When two estimators have the same variance:")
print("- MSE_A = (Bias_A)² + Var")
print("- MSE_B = (Bias_B)² + Var")
print()
print("If |Bias_A| < |Bias_B|, then (Bias_A)² < (Bias_B)²")
print("Therefore: MSE_A < MSE_B")
print()
print("Conclusion: Statement 2 is TRUE. When variance is equal, lower bias always results in lower MSE.")
print("This is because MSE is the sum of squared bias and variance, so with fixed variance, MSE depends only on squared bias.")

# Step 5: Demonstrate the MSE decomposition with simulated data
print_step_header(5, "Demonstrating MSE Decomposition with Simulated Data")

# Simulate estimation with different estimators
np.random.seed(42)
true_param = 5.0
sample_size = 20
n_simulations = 10000

# Define several estimators with different bias-variance properties
def unbiased_estimator(data):
    """Sample mean - unbiased estimator."""
    return np.mean(data)

def biased_estimator_1(data):
    """Biased estimator with multiplication factor."""
    return 0.9 * np.mean(data) + 0.2  # Has bias of -0.1*mu + 0.2

def biased_estimator_2(data):
    """Biased estimator with larger bias."""
    return 0.8 * np.mean(data) + 0.6  # Has bias of -0.2*mu + 0.6

def high_variance_estimator(data):
    """Unbiased but higher variance estimator."""
    # Take only half the sample randomly
    subsample = np.random.choice(data, size=max(1, len(data)//2), replace=False)
    return np.mean(subsample)

# Run simulations to estimate bias, variance, and MSE
estimators = {
    "Unbiased": unbiased_estimator,
    "Biased (Small)": biased_estimator_1,
    "Biased (Large)": biased_estimator_2,
    "High Variance": high_variance_estimator
}

results = {}
for name, estimator_func in estimators.items():
    estimates = []
    for _ in range(n_simulations):
        data = np.random.normal(true_param, 1.0, sample_size)
        estimates.append(estimator_func(data))
    
    estimates = np.array(estimates)
    mean_estimate = np.mean(estimates)
    bias = mean_estimate - true_param
    variance = np.var(estimates)
    mse = np.mean((estimates - true_param)**2)
    
    results[name] = {
        "mean_estimate": mean_estimate,
        "bias": bias,
        "variance": variance,
        "mse": mse,
        "expected_mse": bias**2 + variance  # Theoretical MSE
    }

# Display simulation results
print("\nSimulation Results (10,000 simulations, sample size = 20):")
print(f"True parameter value = {true_param}")
print("\n{:<15} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
    "Estimator", "Mean Est.", "Bias", "Variance", "MSE", "Bias²+Var"))
print("-" * 65)
for name, result in results.items():
    print("{:<15} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
        name, 
        result["mean_estimate"], 
        result["bias"], 
        result["variance"], 
        result["mse"],
        result["bias"]**2 + result["variance"]
    ))

# Create a visualization of the MSE decomposition
plt.figure(figsize=(10, 6))

# Prepare data for the stacked bar chart
names = list(results.keys())
biases_squared = [results[name]["bias"]**2 for name in names]
variances = [results[name]["variance"] for name in names]

# Create stacked bar chart
bar_width = 0.35
indices = np.arange(len(names))

plt.bar(indices, biases_squared, bar_width, label='Squared Bias')
plt.bar(indices, variances, bar_width, bottom=biases_squared, label='Variance')

# Add MSE values on top of each bar
for i, name in enumerate(names):
    mse = results[name]["mse"]
    plt.text(i, mse + 0.02, f'MSE = {mse:.4f}', ha='center')

plt.xticks(indices, names)
plt.ylabel('Value')
plt.title('MSE Decomposition for Different Estimators')
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "mse_decomposition.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 6: Conclude and summarize
print_step_header(6, "Conclusion and Summary")

print("Question 11 Analysis Summary:")
print()
print("Statement 1: The likelihood function represents the probability of observing the data given the parameters.")
print("Verdict: TRUE")
print("The likelihood function mathematically represents P(data|parameters), which is the probability or")
print("probability density of observing the data given the parameters. However, it's important to note that")
print("the likelihood function views this mathematical expression as a function of the parameters with fixed data,")
print("rather than as a probability distribution over the parameters.")
print()
print("Statement 2: If two estimators have the same variance, the one with lower bias will always have lower MSE.")
print("Verdict: TRUE")
print("Given the MSE decomposition: MSE = (Bias)² + Variance")
print("When variance is fixed, MSE depends solely on the squared bias. Therefore, lower bias will always result")
print("in lower MSE when comparing estimators with equal variance.")
print()
print("These concepts are fundamental to statistical estimation theory and help guide the choice of estimators")
print("in practical applications. While unbiased estimators are often preferred, the bias-variance tradeoff")
print("sometimes makes biased estimators with sufficiently low variance more desirable in terms of overall MSE.") 