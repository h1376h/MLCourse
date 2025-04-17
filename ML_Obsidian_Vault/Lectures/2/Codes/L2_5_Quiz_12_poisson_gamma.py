import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_5_Quiz_12")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Set up the problem
print_step_header(1, "Understanding the Astronomy Problem")

print("Given:")
print("- Researchers are modeling the number of exoplanets in different star systems")
print("- The count follows a Poisson distribution with unknown rate parameter λ")
print("- Gamma is the conjugate prior for Poisson")
print("- Prior: Gamma(α=2, β=0.5)")
print("- Observations from 5 star systems: {3, 1, 4, 2, 5} exoplanets")
print()
print("We need to:")
print("1. Express the posterior distribution")
print("2. Find the posterior mean of λ")
print("3. Calculate the posterior predictive probability of finding exactly 3 exoplanets")
print("   in the next observed star system")
print()

# Define the prior parameters and observed data
prior_alpha = 2
prior_beta = 0.5
data = np.array([3, 1, 4, 2, 5])
n = len(data)
total_count = np.sum(data)
sample_mean = np.mean(data)

print(f"Sample data: {data.tolist()}")
print(f"Number of observations: n = {n}")
print(f"Total exoplanet count: sum(x) = {total_count}")
print(f"Average exoplanets per system: {sample_mean:.2f}")
print()

# Step 2: Understand the Poisson-Gamma conjugate relationship
print_step_header(2, "Poisson-Gamma Conjugate Relationship")

print("The Poisson distribution models count data:")
print("P(X = k | λ) = e^(-λ) * λ^k / k!")
print("where λ is the rate parameter (average number of events).")
print()
print("The Gamma distribution is the conjugate prior for the Poisson likelihood:")
print("p(λ) = Gamma(λ | α, β) = (β^α / Γ(α)) * λ^(α-1) * e^(-βλ)")
print("where α is the shape parameter and β is the rate parameter.")
print()
print("When we observe data X = {x₁, x₂, ..., xₙ} from a Poisson(λ) distribution:")
print("- The posterior is: p(λ|X) ∝ p(X|λ) * p(λ)")
print("- This gives us: p(λ|X) = Gamma(λ | α + sum(x), β + n)")
print()

# Plot the prior Gamma distribution
lambda_range = np.linspace(0, 15, 1000)
prior_pdf = stats.gamma.pdf(lambda_range, a=prior_alpha, scale=1/prior_beta)

plt.figure(figsize=(10, 6))
plt.plot(lambda_range, prior_pdf, 'b-', linewidth=2, 
         label=f'Prior: Gamma(α={prior_alpha}, β={prior_beta})')

plt.title('Prior Distribution for λ (Rate of Exoplanets per Star System)', fontsize=14)
plt.xlabel('λ (Exoplanets per Star System)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)

# Save the figure
file_path = os.path.join(save_dir, "prior_distribution.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 3: Calculate the posterior distribution
print_step_header(3, "Calculating the Posterior Distribution")

# Calculate posterior parameters
posterior_alpha = prior_alpha + total_count
posterior_beta = prior_beta + n

# Calculate posterior statistics
posterior_mean = posterior_alpha / posterior_beta
posterior_mode = (posterior_alpha - 1) / posterior_beta if posterior_alpha > 1 else 0
posterior_var = posterior_alpha / (posterior_beta ** 2)
posterior_std = np.sqrt(posterior_var)

# Calculate the 95% credible interval
credible_interval = stats.gamma.ppf([0.025, 0.975], a=posterior_alpha, scale=1/posterior_beta)

print(f"Prior Distribution:")
print(f"- Gamma(α={prior_alpha}, β={prior_beta})")
print(f"- Prior mean: E[λ] = α/β = {prior_alpha/prior_beta:.4f}")
print(f"- Prior mode: (α-1)/β = {(prior_alpha-1)/prior_beta if prior_alpha > 1 else 0:.4f}")
print(f"- Prior variance: α/β² = {prior_alpha/(prior_beta**2):.4f}")
print()

print(f"Posterior Distribution:")
print(f"- Gamma(α={posterior_alpha}, β={posterior_beta})")
print(f"- Posterior mean: E[λ|X] = α'/β' = {posterior_mean:.4f}")
print(f"- Posterior mode: (α'-1)/β' = {posterior_mode:.4f}")
print(f"- Posterior variance: α'/β'² = {posterior_var:.4f}")
print(f"- Posterior standard deviation: {posterior_std:.4f}")
print(f"- 95% Credible Interval: [{credible_interval[0]:.4f}, {credible_interval[1]:.4f}]")
print()

# Plot the prior and posterior Gamma distributions
plt.figure(figsize=(10, 6))

# Plot the prior
plt.plot(lambda_range, prior_pdf, 'b-', linewidth=2, 
         label=f'Prior: Gamma({prior_alpha}, {prior_beta})')

# Plot the posterior
posterior_pdf = stats.gamma.pdf(lambda_range, a=posterior_alpha, scale=1/posterior_beta)
plt.plot(lambda_range, posterior_pdf, 'r-', linewidth=2, 
         label=f'Posterior: Gamma({posterior_alpha}, {posterior_beta})')

# Shade the 95% credible interval
plt.fill_between(lambda_range, 0, posterior_pdf, 
                 where=(lambda_range >= credible_interval[0]) & (lambda_range <= credible_interval[1]), 
                 color='red', alpha=0.3, 
                 label=f'95% Credible Interval: [{credible_interval[0]:.2f}, {credible_interval[1]:.2f}]')

# Mark the posterior mean and mode
plt.axvline(x=posterior_mean, color='darkred', linestyle='--', 
           label=f'Posterior Mean: {posterior_mean:.2f}')
plt.axvline(x=posterior_mode, color='firebrick', linestyle=':', 
           label=f'Posterior Mode: {posterior_mode:.2f}')

plt.title('Prior and Posterior Distributions for λ', fontsize=14)
plt.xlabel('λ (Exoplanets per Star System)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)

# Save the figure
file_path = os.path.join(save_dir, "posterior_distribution.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 4: Calculate the posterior predictive distribution
print_step_header(4, "Calculating the Posterior Predictive Distribution")

print("The posterior predictive distribution gives the probability of future observations")
print("after accounting for the parameter uncertainty.")
print()
print("For the Poisson-Gamma model, the posterior predictive follows a negative binomial distribution:")
print("P(X_new = k | data) = NegBin(k | α', β'/(β'+1))")
print("or equivalently:")
print("P(X_new = k | data) = Choose(k+α'-1, k) * (β'/(β'+1))^α' * (1/(β'+1))^k")
print()

# Calculate posterior predictive distribution for different k values
k_values = np.arange(0, 11)  # Consider 0 to 10 exoplanets
pred_probs = stats.nbinom.pmf(k_values, posterior_alpha, posterior_beta/(posterior_beta+1))

# Find probability of exactly 3 exoplanets
prob_3_exoplanets = stats.nbinom.pmf(3, posterior_alpha, posterior_beta/(posterior_beta+1))

print(f"Posterior Predictive Distribution:")
print(f"- Follows Negative Binomial(r={posterior_alpha}, p={posterior_beta/(posterior_beta+1):.4f})")
print()
print(f"Probabilities for different numbers of exoplanets in the next star system:")
for k, prob in zip(k_values, pred_probs):
    star = '*' if k == 3 else ''
    print(f"  P(X_new = {k}) = {prob:.4f} {star}")
print()
print(f"The probability of finding exactly 3 exoplanets in the next star system is {prob_3_exoplanets:.4f} (or {prob_3_exoplanets*100:.2f}%)")
print()

# Plot the posterior predictive distribution
plt.figure(figsize=(10, 6))

# Create bar plot for the predictive probabilities
plt.bar(k_values, pred_probs, color='purple', alpha=0.7, 
        label='Posterior Predictive Distribution')

# Highlight the bar for k=3
plt.bar(3, pred_probs[3], color='orange', 
        label=f'P(X_new = 3) = {prob_3_exoplanets:.4f}')

plt.title('Posterior Predictive Distribution for Exoplanet Counts', fontsize=14)
plt.xlabel('Number of Exoplanets (k)', fontsize=12)
plt.ylabel('Probability P(X_new = k | data)', fontsize=12)
plt.xticks(k_values)
plt.legend(fontsize=12)
plt.grid(True, axis='y')

# Save the figure
file_path = os.path.join(save_dir, "predictive_distribution.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 5: Visual comparison of empirical data with posterior mean
print_step_header(5, "Comparing Empirical Data with Posterior Inference")

# Create figure with two subplots
plt.figure(figsize=(12, 10))
gs = GridSpec(2, 1, height_ratios=[1, 1.5])

# Plot 1: Observed data with posterior mean
ax1 = plt.subplot(gs[0])
ax1.bar(range(1, n+1), data, color='skyblue', alpha=0.7, label='Observed Exoplanet Counts')
ax1.axhline(y=posterior_mean, color='red', linestyle='-', linewidth=2, 
           label=f'Posterior Mean: {posterior_mean:.2f}')
ax1.axhline(y=sample_mean, color='green', linestyle='--', linewidth=2, 
           label=f'Sample Mean: {sample_mean:.2f}')
ax1.axhline(y=prior_alpha/prior_beta, color='blue', linestyle=':', linewidth=2, 
           label=f'Prior Mean: {prior_alpha/prior_beta:.2f}')

ax1.set_title('Observed Exoplanet Counts with Estimated Rate', fontsize=14)
ax1.set_xlabel('Star System', fontsize=12)
ax1.set_ylabel('Number of Exoplanets', fontsize=12)
ax1.set_xticks(range(1, n+1))
ax1.legend(fontsize=10)
ax1.grid(True, axis='y')

# Plot 2: Evolution of posterior mean as data arrives
ax2 = plt.subplot(gs[1])

# Initialize with prior
evolving_alpha = [prior_alpha]
evolving_beta = [prior_beta]
evolving_mean = [prior_alpha / prior_beta]
evolving_lower_ci = [stats.gamma.ppf(0.025, a=prior_alpha, scale=1/prior_beta)]
evolving_upper_ci = [stats.gamma.ppf(0.975, a=prior_alpha, scale=1/prior_beta)]

# Update with each observation
for i in range(n):
    new_alpha = evolving_alpha[-1] + data[i]
    new_beta = evolving_beta[-1] + 1
    evolving_alpha.append(new_alpha)
    evolving_beta.append(new_beta)
    evolving_mean.append(new_alpha / new_beta)
    evolving_lower_ci.append(stats.gamma.ppf(0.025, a=new_alpha, scale=1/new_beta))
    evolving_upper_ci.append(stats.gamma.ppf(0.975, a=new_alpha, scale=1/new_beta))

# Plot the evolution of posterior mean and credible interval
observations = ['Prior'] + [f'+ System {i+1}: {count} exoplanets' for i, count in enumerate(data)]

ax2.plot(range(n+1), evolving_mean, 'ro-', linewidth=2, markersize=8, label='Posterior Mean')
ax2.fill_between(range(n+1), evolving_lower_ci, evolving_upper_ci, 
                 color='red', alpha=0.2, label='95% Credible Interval')
ax2.scatter(range(1, n+1), data, color='blue', s=100, zorder=10, label='Observed Counts')

# Set x-ticks to show the updates
ax2.set_xticks(range(n+1))
ax2.set_xticklabels(observations, rotation=45, ha='right')
ax2.set_ylim(0, max(evolving_upper_ci) * 1.1)

ax2.set_title('Evolution of Posterior as Data Arrives', fontsize=14)
ax2.set_ylabel('Exoplanet Rate (λ)', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True)

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "posterior_evolution.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 6: Conclusion
print_step_header(6, "Conclusion")

print("The astronomers' Bayesian analysis leads to the following conclusions:")
print()
print("1. The posterior distribution for the exoplanet rate parameter λ is:")
print(f"   Gamma({posterior_alpha}, {posterior_beta})")
print()
print("2. The posterior mean rate of exoplanets per star system is:")
print(f"   E[λ|data] = {posterior_mean:.4f}")
print()
print("3. The probability of observing exactly 3 exoplanets in the next star system is:")
print(f"   P(X_new = 3 | data) = {prob_3_exoplanets:.4f} ({prob_3_exoplanets*100:.2f}%)")
print()
print("4. With 95% credibility, the true exoplanet rate λ is between")
print(f"   {credible_interval[0]:.4f} and {credible_interval[1]:.4f} exoplanets per star system.")
print()
print("5. The posterior distribution effectively combines the prior knowledge (Gamma(2, 0.5))")
print("   with the observed data (15 exoplanets in 5 star systems) to provide an updated")
print("   estimate of the exoplanet rate parameter.") 