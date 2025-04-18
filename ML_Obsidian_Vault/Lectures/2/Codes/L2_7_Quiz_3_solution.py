import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_7_Quiz_3")
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
print("- Two competing models:")
print("  - Model 1: Normal distribution with unknown mean μ₁, known variance σ₁² = 2")
print("  - Model 2: Normal distribution with unknown mean μ₂, known variance σ₂² = 4")
print("- Priors:")
print("  - μ₁ ~ N(0, 1)")
print("  - μ₂ ~ N(0, 2)")
print("- Data X = {1.5, 2.3, 1.8, 2.5, 1.9}")
print("\nTask:")
print("1. Calculate the posterior distribution for μ₁ under Model 1")
print("2. Calculate the posterior distribution for μ₂ under Model 2")
print("3. Calculate the marginal likelihood (evidence) for each model")
print("4. Calculate the Bayes factor and interpret the result")

# Step 2: Calculate Posterior Distributions
print_step_header(2, "Calculating Posterior Distributions")

# Define the parameters
data = np.array([1.5, 2.3, 1.8, 2.5, 1.9])
n = len(data)
data_mean = np.mean(data)

# Model 1 parameters
prior_mean1 = 0.0
prior_var1 = 1.0
likelihood_var1 = 2.0  # Known variance σ₁²

# Model 2 parameters
prior_mean2 = 0.0
prior_var2 = 2.0
likelihood_var2 = 4.0  # Known variance σ₂²

# Calculate posterior parameters for Model 1
posterior_var1 = 1 / (1/prior_var1 + n/likelihood_var1)
posterior_mean1 = posterior_var1 * (prior_mean1/prior_var1 + sum(data)/likelihood_var1)

# Calculate posterior parameters for Model 2
posterior_var2 = 1 / (1/prior_var2 + n/likelihood_var2)
posterior_mean2 = posterior_var2 * (prior_mean2/prior_var2 + sum(data)/likelihood_var2)

print("Model 1 Posterior Calculation:")
print(f"Prior: μ₁ ~ N({prior_mean1}, {prior_var1})")
print(f"Likelihood (for each observation): x_i | μ₁ ~ N(μ₁, {likelihood_var1})")
print(f"Number of observations: n = {n}")
print(f"Sample mean: x̄ = {data_mean:.4f}")
print("\nPosterior variance calculation:")
print(f"1/σ²_posterior1 = 1/σ²_prior1 + n/σ²_likelihood1")
print(f"1/σ²_posterior1 = 1/{prior_var1} + {n}/{likelihood_var1}")
print(f"1/σ²_posterior1 = {1/prior_var1} + {n/likelihood_var1}")
print(f"1/σ²_posterior1 = {1/prior_var1 + n/likelihood_var1}")
print(f"σ²_posterior1 = {posterior_var1:.6f}")
print("\nPosterior mean calculation:")
print(f"μ_posterior1 = σ²_posterior1 * (μ_prior1/σ²_prior1 + Σx_i/σ²_likelihood1)")
print(f"μ_posterior1 = {posterior_var1:.6f} * ({prior_mean1}/{prior_var1} + {sum(data)}/{likelihood_var1})")
print(f"μ_posterior1 = {posterior_var1:.6f} * ({prior_mean1/prior_var1} + {sum(data)/likelihood_var1})")
print(f"μ_posterior1 = {posterior_var1:.6f} * {prior_mean1/prior_var1 + sum(data)/likelihood_var1}")
print(f"μ_posterior1 = {posterior_mean1:.6f}")
print("\nPosterior distribution for Model 1: μ₁ | X ~ N({:.4f}, {:.6f})".format(posterior_mean1, posterior_var1))

print("\nModel 2 Posterior Calculation:")
print(f"Prior: μ₂ ~ N({prior_mean2}, {prior_var2})")
print(f"Likelihood (for each observation): x_i | μ₂ ~ N(μ₂, {likelihood_var2})")
print(f"Number of observations: n = {n}")
print(f"Sample mean: x̄ = {data_mean:.4f}")
print("\nPosterior variance calculation:")
print(f"1/σ²_posterior2 = 1/σ²_prior2 + n/σ²_likelihood2")
print(f"1/σ²_posterior2 = 1/{prior_var2} + {n}/{likelihood_var2}")
print(f"1/σ²_posterior2 = {1/prior_var2} + {n/likelihood_var2}")
print(f"1/σ²_posterior2 = {1/prior_var2 + n/likelihood_var2}")
print(f"σ²_posterior2 = {posterior_var2:.6f}")
print("\nPosterior mean calculation:")
print(f"μ_posterior2 = σ²_posterior2 * (μ_prior2/σ²_prior2 + Σx_i/σ²_likelihood2)")
print(f"μ_posterior2 = {posterior_var2:.6f} * ({prior_mean2}/{prior_var2} + {sum(data)}/{likelihood_var2})")
print(f"μ_posterior2 = {posterior_var2:.6f} * ({prior_mean2/prior_var2} + {sum(data)/likelihood_var2})")
print(f"μ_posterior2 = {posterior_var2:.6f} * {prior_mean2/prior_var2 + sum(data)/likelihood_var2}")
print(f"μ_posterior2 = {posterior_mean2:.6f}")
print("\nPosterior distribution for Model 2: μ₂ | X ~ N({:.4f}, {:.6f})".format(posterior_mean2, posterior_var2))

# Create a visualization of the posterior distributions for both models
mu_range = np.linspace(-1, 3, 1000)

# Posterior distributions
posterior1 = norm.pdf(mu_range, posterior_mean1, np.sqrt(posterior_var1))
posterior2 = norm.pdf(mu_range, posterior_mean2, np.sqrt(posterior_var2))

plt.figure(figsize=(10, 6))
plt.plot(mu_range, posterior1, 'b-', label=f'Model 1 Posterior: N({posterior_mean1:.4f}, {posterior_var1:.6f})', linewidth=2)
plt.plot(mu_range, posterior2, 'r--', label=f'Model 2 Posterior: N({posterior_mean2:.4f}, {posterior_var2:.6f})', linewidth=2)

# Mark the posterior means
plt.axvline(x=posterior_mean1, color='blue', linestyle=':', alpha=0.7)
plt.axvline(x=posterior_mean2, color='red', linestyle=':', alpha=0.7)

# Mark the data points on the x-axis
for x in data:
    plt.axvline(x=x, color='g', linestyle='--', alpha=0.2)

plt.xlabel('μ (Mean)', fontsize=12)
plt.ylabel('Posterior Density', fontsize=12)
plt.title('Posterior Distributions for Model 1 and Model 2', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "posterior_distributions.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {file_path}")

# Step 3: Calculate Marginal Likelihoods (Evidence)
print_step_header(3, "Calculating Marginal Likelihoods (Evidence)")

# For normal likelihood and normal prior, we can compute the log marginal likelihood analytically
def log_marginal_likelihood(data, prior_mean, prior_var, likelihood_var):
    n = len(data)
    s2 = likelihood_var
    t2 = prior_var
    
    # Calculate sufficient statistics
    sum_x = np.sum(data)
    sum_x2 = np.sum(data**2)
    
    # Calculate log marginal likelihood components
    log_ml = -0.5 * n * np.log(2 * np.pi * s2)
    log_ml -= 0.5 * (1/s2) * sum_x2
    
    # Add the contribution from integrating out mu
    precision_post = 1/t2 + n/s2
    var_post = 1/precision_post
    mean_post = var_post * (prior_mean/t2 + sum_x/s2)
    
    log_ml += 0.5 * np.log(t2) - 0.5 * np.log(var_post)
    log_ml += 0.5 * (mean_post**2 / var_post - prior_mean**2 / t2)
    
    return log_ml

# Compute marginal likelihoods
log_ml1 = log_marginal_likelihood(data, prior_mean1, prior_var1, likelihood_var1)
log_ml2 = log_marginal_likelihood(data, prior_mean2, prior_var2, likelihood_var2)

# Convert to natural scale
ml1 = np.exp(log_ml1)
ml2 = np.exp(log_ml2)

print("To calculate the marginal likelihood (evidence), we integrate out the parameter:")
print("p(X | Model) = ∫ p(X | μ, Model) p(μ | Model) dμ")
print("\nFor normal likelihood and normal prior, this has a closed form solution.")
print("\nModel 1 Marginal Likelihood calculation:")
print(f"Log Marginal Likelihood for Model 1: {log_ml1:.6f}")
print(f"Marginal Likelihood for Model 1: {ml1:.10e}")

print("\nModel 2 Marginal Likelihood calculation:")
print(f"Log Marginal Likelihood for Model 2: {log_ml2:.6f}")
print(f"Marginal Likelihood for Model 2: {ml2:.10e}")

# Step 4: Calculate Bayes Factor
print_step_header(4, "Calculating Bayes Factor and Interpretation")

# Calculate Bayes factor
bayes_factor = ml1 / ml2
log_bayes_factor = log_ml1 - log_ml2

print("The Bayes factor is the ratio of marginal likelihoods:")
print("BF₁₂ = p(X | Model 1) / p(X | Model 2)")
print(f"BF₁₂ = {ml1:.10e} / {ml2:.10e}")
print(f"BF₁₂ = {bayes_factor:.6f}")
print(f"Log(BF₁₂) = {log_bayes_factor:.6f}")

# Interpret the Bayes factor
def interpret_bayes_factor(bf):
    if bf > 100:
        return "Extreme evidence for Model 1"
    elif bf > 30:
        return "Very strong evidence for Model 1"
    elif bf > 10:
        return "Strong evidence for Model 1"
    elif bf > 3:
        return "Moderate evidence for Model 1"
    elif bf > 1:
        return "Weak evidence for Model 1"
    elif bf > 1/3:
        return "Weak evidence for Model 2"
    elif bf > 1/10:
        return "Moderate evidence for Model 2"
    elif bf > 1/30:
        return "Strong evidence for Model 2"
    elif bf > 1/100:
        return "Very strong evidence for Model 2"
    else:
        return "Extreme evidence for Model 2"

interpretation = interpret_bayes_factor(bayes_factor)
print(f"\nInterpretation: {interpretation}")

if bayes_factor > 1:
    print(f"The data favor Model 1 over Model 2 by a factor of {bayes_factor:.2f}.")
else:
    print(f"The data favor Model 2 over Model 1 by a factor of {1/bayes_factor:.2f}.")

# Step 5: Summary
print_step_header(5, "Summary of Results")

print("1. Posterior Distribution for μ₁ under Model 1:")
print(f"   μ₁ | X ~ N({posterior_mean1:.6f}, {posterior_var1:.6f})")
print()
print("2. Posterior Distribution for μ₂ under Model 2:")
print(f"   μ₂ | X ~ N({posterior_mean2:.6f}, {posterior_var2:.6f})")
print()
print("3. Marginal Likelihoods:")
print(f"   p(X | Model 1) = {ml1:.10e}")
print(f"   p(X | Model 2) = {ml2:.10e}")
print()
print("4. Bayes Factor:")
print(f"   BF₁₂ = {bayes_factor:.6f}")
print(f"   Interpretation: {interpretation}")
print()
print("Conclusion:")
if bayes_factor > 1:
    print(f"The data favor Model 1 over Model 2 by a factor of {bayes_factor:.2f}.")
    print("Model 1 has a smaller variance (σ₁² = 2) which fits the observed data better than")
    print("Model 2 with larger variance (σ₂² = 4).")
else:
    print(f"The data favor Model 2 over Model 1 by a factor of {1/bayes_factor:.2f}.")
    print("Model 2 has a larger variance (σ₂² = 4) which fits the observed data better than")
    print("Model 1 with smaller variance (σ₁² = 2).") 