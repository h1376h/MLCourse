import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import beta
from scipy.special import beta as beta_function
from scipy.special import gamma

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_7_Quiz_11")
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
print("- A coin-flipping model with parameter θ representing the probability of heads")
print("- Observed data D = {H, H, T, H, T} (3 heads, 2 tails)")
print("\nTask:")
print("1. Compute the marginal likelihood p(D) using a uniform prior (Beta(1,1))")
print("2. Compute the marginal likelihood p(D) using an informative prior (Beta(10,10))")
print("3. Explain how these values could be used for model comparison")

# Step 2: Computing the Marginal Likelihood with a Uniform Prior
print_step_header(2, "Computing the Marginal Likelihood with a Uniform Prior")

# Define parameters for the uniform prior and data
alpha_uniform = 1
beta_uniform = 1
heads = 3
tails = 2

# For a Beta-Binomial model, the marginal likelihood is:
# p(D) = B(α + h, β + t) / B(α, β)
# where B is the beta function, h is number of heads, t is number of tails

def compute_marginal_likelihood(alpha, beta, heads, tails):
    """Compute the marginal likelihood for a Beta-Binomial model."""
    # Use the beta function to calculate B(α,β)
    prior_beta = beta_function(alpha, beta)
    
    # Calculate B(α + h, β + t)
    posterior_beta = beta_function(alpha + heads, beta + tails)
    
    # Calculate the marginal likelihood
    marginal_likelihood = posterior_beta / prior_beta
    
    return marginal_likelihood

# Calculate the marginal likelihood with uniform prior
marginal_likelihood_uniform = compute_marginal_likelihood(alpha_uniform, beta_uniform, heads, tails)

print("For a Beta-Binomial model, the marginal likelihood is:")
print("p(D) = B(α + h, β + t) / B(α, β)")
print("where:")
print("- B is the beta function")
print("- α and β are the parameters of the Beta prior")
print("- h is the number of heads")
print("- t is the number of tails")
print("\nWith a uniform prior Beta(1,1):")
print(f"p(D) = B({alpha_uniform + heads}, {beta_uniform + tails}) / B({alpha_uniform}, {beta_uniform})")
print(f"p(D) = B(4, 3) / B(1, 1)")
print(f"p(D) = {beta_function(alpha_uniform + heads, beta_uniform + tails):.6f} / {beta_function(alpha_uniform, beta_uniform):.6f}")
print(f"p(D) = {marginal_likelihood_uniform:.6f}")

# Step 3: Computing the Marginal Likelihood with an Informative Prior
print_step_header(3, "Computing the Marginal Likelihood with an Informative Prior")

# Define parameters for the informative prior
alpha_informative = 10
beta_informative = 10

# Calculate the marginal likelihood with informative prior
marginal_likelihood_informative = compute_marginal_likelihood(alpha_informative, beta_informative, heads, tails)

print("With an informative prior Beta(10,10):")
print(f"p(D) = B({alpha_informative + heads}, {beta_informative + tails}) / B({alpha_informative}, {beta_informative})")
print(f"p(D) = B(13, 12) / B(10, 10)")
print(f"p(D) = {beta_function(alpha_informative + heads, beta_informative + tails):.6f} / {beta_function(alpha_informative, beta_informative):.6f}")
print(f"p(D) = {marginal_likelihood_informative:.6f}")

# Create a visualization comparing the two marginal likelihoods
plt.figure(figsize=(10, 6))

# Create a bar chart to compare the marginal likelihoods
priors = ['Uniform Prior\nBeta(1,1)', 'Informative Prior\nBeta(10,10)']
marginal_likelihoods = [marginal_likelihood_uniform, marginal_likelihood_informative]

plt.bar(priors, marginal_likelihoods, color=['blue', 'green'])

plt.ylabel('Marginal Likelihood p(D)', fontsize=12)
plt.title('Comparison of Marginal Likelihoods for Different Priors', fontsize=14)
plt.grid(True, alpha=0.3)

# Add text annotations
for i, v in enumerate(marginal_likelihoods):
    plt.text(i, v + 0.001, f'{v:.6f}', ha='center', fontsize=10)

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "marginal_likelihood_comparison.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 4: Visual representation of the priors and posteriors
print_step_header(4, "Visualization of Priors and Posteriors")

# Define a range of theta values
theta = np.linspace(0, 1, 1000)

# Compute the Beta PDF values for prior and posterior distributions
prior_uniform = beta.pdf(theta, alpha_uniform, beta_uniform)
posterior_uniform = beta.pdf(theta, alpha_uniform + heads, beta_uniform + tails)

prior_informative = beta.pdf(theta, alpha_informative, beta_informative)
posterior_informative = beta.pdf(theta, alpha_informative + heads, beta_informative + tails)

# Plot the priors and posteriors
plt.figure(figsize=(12, 10))

# Plot for uniform prior
plt.subplot(2, 1, 1)
plt.plot(theta, prior_uniform, 'b-', label=f'Prior: Beta({alpha_uniform}, {beta_uniform})', linewidth=2)
plt.plot(theta, posterior_uniform, 'r-', label=f'Posterior: Beta({alpha_uniform + heads}, {beta_uniform + tails})', linewidth=2)
plt.axvline(x=0.5, color='k', linestyle='--', alpha=0.5, label='Prior Mean: 0.5')
plt.axvline(x=(alpha_uniform + heads) / (alpha_uniform + beta_uniform + heads + tails), 
          color='darkred', linestyle='--', alpha=0.5, 
          label=f'Posterior Mean: {(alpha_uniform + heads) / (alpha_uniform + beta_uniform + heads + tails):.3f}')

plt.xlabel('θ (Probability of Heads)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Uniform Prior: Beta(1,1) and Resulting Posterior', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Plot for informative prior
plt.subplot(2, 1, 2)
plt.plot(theta, prior_informative, 'g-', label=f'Prior: Beta({alpha_informative}, {beta_informative})', linewidth=2)
plt.plot(theta, posterior_informative, 'm-', label=f'Posterior: Beta({alpha_informative + heads}, {beta_informative + tails})', linewidth=2)
plt.axvline(x=0.5, color='k', linestyle='--', alpha=0.5, label='Prior Mean: 0.5')
plt.axvline(x=(alpha_informative + heads) / (alpha_informative + beta_informative + heads + tails), 
          color='darkmagenta', linestyle='--', alpha=0.5, 
          label=f'Posterior Mean: {(alpha_informative + heads) / (alpha_informative + beta_informative + heads + tails):.3f}')

plt.xlabel('θ (Probability of Heads)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Informative Prior: Beta(10,10) and Resulting Posterior', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "priors_and_posteriors.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 5: Visualizing the likelihood function
print_step_header(5, "Visualizing the Likelihood Function")

# Define the binomial likelihood function
def binomial_likelihood(theta, heads, tails):
    """Compute the binomial likelihood for a given theta."""
    # Likelihood is proportional to theta^h * (1-theta)^t
    return theta**heads * (1-theta)**tails

# Compute the likelihood values
likelihood_values = binomial_likelihood(theta, heads, tails)

# Normalize for better visualization
likelihood_values = likelihood_values / np.max(likelihood_values)

plt.figure(figsize=(10, 6))

plt.plot(theta, likelihood_values, 'k-', label='Likelihood p(D|θ)', linewidth=2)
plt.axvline(x=heads/(heads+tails), color='r', linestyle='--', 
           label=f'MLE: θ = {heads/(heads+tails):.3f}')

plt.xlabel('θ (Probability of Heads)', fontsize=12)
plt.ylabel('Normalized Likelihood', fontsize=12)
plt.title('Likelihood Function for 3 Heads and 2 Tails', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "likelihood_function.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 6: Model comparison using Bayes factor
print_step_header(6, "Model Comparison using Bayes Factor")

# Calculate the Bayes factor
bayes_factor = marginal_likelihood_uniform / marginal_likelihood_informative

print("The Bayes factor for comparing the models with uniform vs. informative prior is:")
print(f"BF = p(D|M₁) / p(D|M₂) = {marginal_likelihood_uniform:.6f} / {marginal_likelihood_informative:.6f} = {bayes_factor:.6f}")

print("\nInterpretation of the Bayes factor:")
if bayes_factor > 1:
    print(f"BF = {bayes_factor:.6f} > 1: Evidence favors the model with uniform prior")
    
    # Provide a more detailed interpretation
    if bayes_factor < 3:
        print("1 < BF < 3: Weak evidence for the uniform prior model")
    elif bayes_factor < 10:
        print("3 < BF < 10: Substantial evidence for the uniform prior model")
    elif bayes_factor < 30:
        print("10 < BF < 30: Strong evidence for the uniform prior model")
    elif bayes_factor < 100:
        print("30 < BF < 100: Very strong evidence for the uniform prior model")
    else:
        print("BF > 100: Decisive evidence for the uniform prior model")
else:
    inverse_bf = 1 / bayes_factor
    print(f"BF = {bayes_factor:.6f} < 1: Evidence favors the model with informative prior")
    
    # Provide a more detailed interpretation
    if inverse_bf < 3:
        print("1 < 1/BF < 3: Weak evidence for the informative prior model")
    elif inverse_bf < 10:
        print("3 < 1/BF < 10: Substantial evidence for the informative prior model")
    elif inverse_bf < 30:
        print("10 < 1/BF < 30: Strong evidence for the informative prior model")
    elif inverse_bf < 100:
        print("30 < 1/BF < 100: Very strong evidence for the informative prior model")
    else:
        print("1/BF > 100: Decisive evidence for the informative prior model")

# Create a visualization for the Bayes factor interpretation
plt.figure(figsize=(10, 5))

# Create a simple visualization showing the Bayes factor scale
bf_value = bayes_factor
bf_scale = np.array([0.01, 0.033, 0.1, 0.33, 1.0, 3.0, 10.0, 30.0, 100.0])
bf_labels = ['1/100', '1/30', '1/10', '1/3', '1', '3', '10', '30', '100']
bf_categories = [
    'Decisive\nfor M₂', 
    'V. Strong\nfor M₂', 
    'Strong\nfor M₂', 
    'Substantial\nfor M₂', 
    'Weak\neither way',
    'Substantial\nfor M₁', 
    'Strong\nfor M₁', 
    'V. Strong\nfor M₁', 
    'Decisive\nfor M₁'
]

colors = plt.cm.RdYlBu(np.linspace(0, 1, len(bf_scale)))

# Plot the bar chart for evidence categories
plt.bar(np.arange(len(bf_categories)), [1]*len(bf_categories), width=0.8, 
       color=colors, alpha=0.7)

# Add labels
for i, (cat, val) in enumerate(zip(bf_categories, bf_scale)):
    plt.text(i, 0.5, cat, ha='center', va='center', fontsize=8, fontweight='bold')
    if i < len(bf_scale)-1:
        plt.text(i+0.5, -0.1, bf_labels[i], ha='center', va='top', fontsize=8, rotation=90)

# Highlight where our Bayes factor falls
bf_position = np.interp(bf_value, bf_scale, np.arange(len(bf_scale)))
plt.scatter([bf_position], [0.5], s=200, color='black', marker='*', 
           label=f'BF = {bf_value:.4f}', zorder=10)
plt.legend(loc='upper center')

plt.ylim(0, 1)
plt.xlim(-0.5, len(bf_categories)-0.5)
plt.title('Interpretation of Bayes Factor', fontsize=14)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "bayes_factor_interpretation.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 7: How Different Prior Choices Affect Marginal Likelihood
print_step_header(7, "Effect of Different Priors on Marginal Likelihood")

# Define a range of different priors
alpha_values = [1, 2, 5, 10, 20]
beta_values = [1, 2, 5, 10, 20]

# Create a matrix of marginal likelihoods for different pairs of alpha and beta
marginal_matrix = np.zeros((len(alpha_values), len(beta_values)))

for i, alpha in enumerate(alpha_values):
    for j, beta in enumerate(beta_values):
        marginal_matrix[i, j] = compute_marginal_likelihood(alpha, beta, heads, tails)

# Plot a heatmap of marginal likelihoods
plt.figure(figsize=(10, 8))

plt.imshow(marginal_matrix, cmap='viridis', origin='lower')

# Add text annotations
for i in range(len(alpha_values)):
    for j in range(len(beta_values)):
        plt.text(j, i, f'{marginal_matrix[i, j]:.5f}', 
                ha='center', va='center', color='white' if marginal_matrix[i, j] < 0.05 else 'black')

plt.colorbar(label='Marginal Likelihood p(D)')
plt.xticks(np.arange(len(beta_values)), [f'β={b}' for b in beta_values])
plt.yticks(np.arange(len(alpha_values)), [f'α={a}' for a in alpha_values])
plt.xlabel('Beta Parameter of Prior', fontsize=12)
plt.ylabel('Alpha Parameter of Prior', fontsize=12)
plt.title('Marginal Likelihood for Different Prior Choices', fontsize=14)

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "marginal_likelihood_heatmap.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 8: Summary and Conclusions
print_step_header(8, "Summary and Conclusions")

print("In this analysis, we've computed and compared the marginal likelihoods for two models:")
print(f"1. Model with uniform prior (Beta(1,1)): p(D) = {marginal_likelihood_uniform:.6f}")
print(f"2. Model with informative prior (Beta(10,10)): p(D) = {marginal_likelihood_informative:.6f}")
print("\nKey insights:")
print(f"1. The model with the informative prior has a higher marginal likelihood.")
print(f"2. The Bayes factor of {bayes_factor:.6f} suggests", end=" ")

if bayes_factor > 100:
    print("decisive evidence in favor of the uniform prior model.")
elif bayes_factor > 30:
    print("very strong evidence in favor of the uniform prior model.")
elif bayes_factor > 10:
    print("strong evidence in favor of the uniform prior model.")
elif bayes_factor > 3:
    print("substantial evidence in favor of the uniform prior model.")
elif bayes_factor > 1:
    print("weak evidence in favor of the uniform prior model.")
elif bayes_factor > 1/3:
    print("weak evidence in favor of the informative prior model.")
elif bayes_factor > 1/10:
    print("substantial evidence in favor of the informative prior model.")
elif bayes_factor > 1/30:
    print("strong evidence in favor of the informative prior model.")
elif bayes_factor > 1/100:
    print("very strong evidence in favor of the informative prior model.")
else:
    print("decisive evidence in favor of the informative prior model.")

print("3. The data (3 heads, 2 tails) is more compatible with the model using the uniform prior.")
print("4. The informative prior (Beta(10,10)) is centered at 0.5, which is close to the observed")
print("   frequency of heads (3/5 = 0.6), but its stronger concentration around 0.5 makes the")
print("   data less likely under this model.")
print("5. The marginal likelihood represents the probability of observing the data, averaged")
print("   over all possible parameter values weighted by the prior.")
print("6. In model comparison, higher marginal likelihood indicates a better balance between")
print("   model fit and complexity.")