import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
from matplotlib.patches import Patch

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

def print_substep(substep_title):
    """Print a formatted substep header."""
    print(f"\n{'-' * 50}")
    print(f"{substep_title}")
    print(f"{'-' * 50}")

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_5_Quiz_18")
os.makedirs(save_dir, exist_ok=True)

# Function to plot beta distributions
def plot_beta(alphas, betas, labels, title, filename, x_label="Probability of Rain", include_mean=False, include_mode=False, include_legend=True, xlim=None):
    plt.figure(figsize=(10, 6))
    x = np.linspace(0, 1, 1000)
    
    # Custom colors for better distinction
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (a, b, label) in enumerate(zip(alphas, betas, labels)):
        y = stats.beta.pdf(x, a, b)
        plt.plot(x, y, label=label, color=colors[i % len(colors)], linewidth=2)
        
        if include_mean:
            mean = a / (a + b)
            plt.axvline(mean, color=colors[i % len(colors)], linestyle='--', alpha=0.7)
            plt.text(mean+0.01, plt.ylim()[1]*0.9, f'Mean: {mean:.3f}', 
                     color=colors[i % len(colors)], rotation=90, verticalalignment='top')
        
        if include_mode and a > 1:  # Mode exists only when alpha > 1
            mode = (a - 1) / (a + b - 2) if (a + b - 2) > 0 else a / (a + b)
            plt.axvline(mode, color=colors[i % len(colors)], linestyle=':', alpha=0.7)
            plt.text(mode-0.01, plt.ylim()[1]*0.7, f'Mode: {mode:.3f}', 
                     color=colors[i % len(colors)], rotation=90, verticalalignment='top', horizontalalignment='right')
    
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(title, fontsize=14)
    if include_legend:
        plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    if xlim:
        plt.xlim(xlim)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {os.path.join(save_dir, filename)}")

# Function to calculate beta distribution statistics
def beta_stats(alpha, beta):
    mean = alpha / (alpha + beta)
    var = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
    std = np.sqrt(var)
    
    # Mode exists only when alpha > 1
    mode = (alpha - 1) / (alpha + beta - 2) if alpha > 1 and (alpha + beta - 2) > 0 else "N/A"
    
    # Calculate 95% credible interval
    lower = stats.beta.ppf(0.025, alpha, beta)
    upper = stats.beta.ppf(0.975, alpha, beta)
    
    return {
        "mean": mean,
        "mode": mode,
        "variance": var,
        "std_dev": std,
        "95%_CI": (lower, upper)
    }

# Function to calculate equivalent sample sizes
def beta_effective_sample_size(alpha, beta):
    return alpha + beta - 2  # Subtract 2 for the uniform prior contribution

# ==============================
# STEP 1: Understanding the Problem
# ==============================
print_step_header(1, "Understanding the Problem")

print("Problem Context:")
print("- A meteorologist is forecasting rain probability")
print("- Initial Beta(2, 8) prior based on historical data")
print("- New information: storm system is approaching")
print("- Tasks: update prior, adjust to 40% probability, handle uncertainty, calculate posteriors")

# Calculate and display prior statistics
original_prior = (2, 8)
prior_stats = beta_stats(*original_prior)

print("\nOriginal Prior - Beta(2, 8):")
print(f"Mean (Expected probability): {prior_stats['mean']:.4f}")
print(f"Mode (Most likely probability): {prior_stats['mode']}")
print(f"Standard Deviation: {prior_stats['std_dev']:.4f}")
print(f"95% Credible Interval: ({prior_stats['95%_CI'][0]:.4f}, {prior_stats['95%_CI'][1]:.4f})")
print(f"Effective Sample Size: {beta_effective_sample_size(*original_prior)}")

# Visualize the original prior
plot_beta([2], [8], ["Beta(2, 8)"], "Original Beta(2, 8) Prior", "original_prior.png", 
          include_mean=True, include_mode=True)

# ==============================
# STEP 2: Adjusting the Prior for New Information
# ==============================
print_step_header(2, "Adjusting the Prior for New Information")

print_substep("2.1: Calculating New Prior With Same Certainty but 40% Mean")

# Original prior has mean = α/(α+β) = 2/(2+8) = 2/10 = 0.2
# We want a new prior with mean = 0.4 and same effective sample size
original_strength = original_prior[0] + original_prior[1]  # α+β
new_mean = 0.4  # Target mean

# If mean = α/(α+β) = 0.4 and (α+β) = 10, then α = 0.4*10 = 4
new_alpha = new_mean * original_strength
new_beta = original_strength - new_alpha

print(f"New prior calculations:")
print(f"- Original prior: Beta({original_prior[0]}, {original_prior[1]})")
print(f"- Original mean: {prior_stats['mean']}")
print(f"- Original strength (α+β): {original_strength}")
print(f"- Target mean: {new_mean}")
print(f"- Calculated new α: {new_alpha}")
print(f"- Calculated new β: {new_beta}")
print(f"- New prior: Beta({new_alpha}, {new_beta})")

# Verify the new prior has the desired properties
new_prior = (new_alpha, new_beta)
new_prior_stats = beta_stats(*new_prior)

print("\nNew Prior - Beta(4, 6):")
print(f"Mean (Expected probability): {new_prior_stats['mean']:.4f}")
print(f"Mode (Most likely probability): {new_prior_stats['mode']}")
print(f"Standard Deviation: {new_prior_stats['std_dev']:.4f}")
print(f"95% Credible Interval: ({new_prior_stats['95%_CI'][0]:.4f}, {new_prior_stats['95%_CI'][1]:.4f})")
print(f"Effective Sample Size: {beta_effective_sample_size(*new_prior)}")

# Compare original and new priors
plot_beta([2, 4], [8, 6], ["Original Beta(2, 8)", "Adjusted Beta(4, 6)"],
          "Comparing Original and Adjusted Priors",
          "comparing_priors.png", include_mean=True, include_mode=True)

print_substep("2.2: Handling Increased Uncertainty")

# Create a more uncertain prior with same mean but less certainty
less_certain_alpha = 1.2  # These values maintain mean ≈ 0.4 but with higher variance
less_certain_beta = 1.8
less_certain_prior = (less_certain_alpha, less_certain_beta)
less_certain_stats = beta_stats(*less_certain_prior)

print("For increased uncertainty with same mean (0.4):")
print(f"- Less certain prior: Beta({less_certain_alpha}, {less_certain_beta})")
print(f"- Mean: {less_certain_stats['mean']:.4f}")
print(f"- Standard Deviation: {less_certain_stats['std_dev']:.4f} (higher than adjusted prior)")
print(f"- 95% Credible Interval: ({less_certain_stats['95%_CI'][0]:.4f}, {less_certain_stats['95%_CI'][1]:.4f})")
print(f"- Effective Sample Size: {beta_effective_sample_size(*less_certain_prior)}")

# Compare all three priors
plot_beta([2, 4, 1.2], [8, 6, 1.8], 
          ["Original Beta(2, 8)", "Adjusted Beta(4, 6)", "Uncertain Beta(1.2, 1.8)"],
          "Comparing Priors with Different Certainty Levels",
          "comparing_certainty.png", include_mean=True)

# ==============================
# STEP 3: Calculating Posteriors After Observing Data
# ==============================
print_step_header(3, "Calculating Posteriors After Observing Data")

# Observed data: 3 rainy days out of 5
rainy_days = 3
total_days = 5

print(f"Observed data: {rainy_days} rainy days out of {total_days} days")

# For Beta(α, β) prior with data (successes, failures), posterior is Beta(α+successes, β+failures)
original_posterior = (original_prior[0] + rainy_days, original_prior[1] + (total_days - rainy_days))
adjusted_posterior = (new_prior[0] + rainy_days, new_prior[1] + (total_days - rainy_days))
uncertain_posterior = (less_certain_prior[0] + rainy_days, less_certain_prior[1] + (total_days - rainy_days))

# Calculate posterior statistics
original_posterior_stats = beta_stats(*original_posterior)
adjusted_posterior_stats = beta_stats(*adjusted_posterior)
uncertain_posterior_stats = beta_stats(*uncertain_posterior)

# Display posterior results
print("\nPosterior Results:")
print("1. From Original Prior Beta(2, 8):")
print(f"   - Posterior: Beta({original_posterior[0]}, {original_posterior[1]})")
print(f"   - Mean: {original_posterior_stats['mean']:.4f}")
print(f"   - 95% Credible Interval: ({original_posterior_stats['95%_CI'][0]:.4f}, {original_posterior_stats['95%_CI'][1]:.4f})")

print("\n2. From Adjusted Prior Beta(4, 6):")
print(f"   - Posterior: Beta({adjusted_posterior[0]}, {adjusted_posterior[1]})")
print(f"   - Mean: {adjusted_posterior_stats['mean']:.4f}")
print(f"   - 95% Credible Interval: ({adjusted_posterior_stats['95%_CI'][0]:.4f}, {adjusted_posterior_stats['95%_CI'][1]:.4f})")

print("\n3. From Uncertain Prior Beta(1.2, 1.8):")
print(f"   - Posterior: Beta({uncertain_posterior[0]}, {uncertain_posterior[1]})")
print(f"   - Mean: {uncertain_posterior_stats['mean']:.4f}")
print(f"   - 95% Credible Interval: ({uncertain_posterior_stats['95%_CI'][0]:.4f}, {uncertain_posterior_stats['95%_CI'][1]:.4f})")

# Visualize the posteriors
plot_beta([original_posterior[0], adjusted_posterior[0], uncertain_posterior[0]],
          [original_posterior[1], adjusted_posterior[1], uncertain_posterior[1]],
          [f"From Beta(2, 8): Beta({original_posterior[0]}, {original_posterior[1]})",
           f"From Beta(4, 6): Beta({adjusted_posterior[0]}, {adjusted_posterior[1]})",
           f"From Beta(1.2, 1.8): Beta({uncertain_posterior[0]}, {uncertain_posterior[1]})"],
          "Posterior Distributions After Observing 3/5 Rainy Days",
          "posteriors.png", include_mean=True)

# ==============================
# STEP 4: Comparing Prior Influence
# ==============================
print_step_header(4, "Comparing Prior Influence")

# Visualize prior-to-posterior updating for each case
# Original Prior --> Posterior
plot_beta([2, original_posterior[0]], 
          [8, original_posterior[1]],
          ["Prior Beta(2, 8)", f"Posterior Beta({original_posterior[0]}, {original_posterior[1]})"],
          "Bayesian Updating with Original Prior",
          "updating_original.png", include_mean=True)

# Adjusted Prior --> Posterior
plot_beta([4, adjusted_posterior[0]], 
          [6, adjusted_posterior[1]],
          ["Prior Beta(4, 6)", f"Posterior Beta({adjusted_posterior[0]}, {adjusted_posterior[1]})"],
          "Bayesian Updating with Adjusted Prior",
          "updating_adjusted.png", include_mean=True)

# Uncertain Prior --> Posterior
plot_beta([1.2, uncertain_posterior[0]], 
          [1.8, uncertain_posterior[1]],
          ["Prior Beta(1.2, 1.8)", f"Posterior Beta({uncertain_posterior[0]}, {uncertain_posterior[1]})"],
          "Bayesian Updating with Uncertain Prior",
          "updating_uncertain.png", include_mean=True)

# Create a summary visualization showing all three prior-posterior pairs
plt.figure(figsize=(12, 8))
x = np.linspace(0, 1, 1000)

# Define styles for priors and posteriors
prior_styles = ['--', '--', '--']
posterior_styles = ['-', '-', '-']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
alpha_levels = [0.6, 0.6, 0.6]

# Plot each prior and its posterior
priors = [(2, 8), (4, 6), (1.2, 1.8)]
posteriors = [original_posterior, adjusted_posterior, uncertain_posterior]
prior_labels = ["Original Prior Beta(2, 8)", "Adjusted Prior Beta(4, 6)", "Uncertain Prior Beta(1.2, 1.8)"]
posterior_labels = [f"Posterior Beta({original_posterior[0]}, {original_posterior[1]})",
                    f"Posterior Beta({adjusted_posterior[0]}, {adjusted_posterior[1]})",
                    f"Posterior Beta({uncertain_posterior[0]}, {uncertain_posterior[1]})"]

for i, ((a_prior, b_prior), (a_post, b_post)) in enumerate(zip(priors, posteriors)):
    # Plot prior
    y_prior = stats.beta.pdf(x, a_prior, b_prior)
    plt.plot(x, y_prior, linestyle=prior_styles[i], color=colors[i], 
             label=prior_labels[i], alpha=alpha_levels[i])
    
    # Plot posterior
    y_post = stats.beta.pdf(x, a_post, b_post)
    plt.plot(x, y_post, linestyle=posterior_styles[i], color=colors[i], 
             label=posterior_labels[i])

# Add data likelihood
likelihood_x = np.linspace(0, 1, 1000)
likelihood_y = [x**rainy_days * (1-x)**(total_days-rainy_days) * 100 for x in likelihood_x]  # Scaled for visibility
plt.plot(likelihood_x, likelihood_y, 'k--', label=f"Likelihood (3/5 rainy days)", alpha=0.7)

plt.xlabel("Probability of Rain", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.title("Prior and Posterior Distributions with Different Starting Points", fontsize=14)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "prior_posterior_comparison.png"), dpi=300, bbox_inches='tight')
plt.close()

# ==============================
# STEP 5: Key Insights and Conclusions
# ==============================
print_step_header(5, "Key Insights and Conclusions")

print("1. Prior Selection Impact:")
print("   - Different priors lead to different posteriors even with the same data")
print("   - The original prior (Beta(2, 8)) had a strong influence pulling down the estimate")
print("   - The adjusted prior (Beta(4, 6)) led to higher posterior probabilities of rain")
print("   - The uncertain prior (Beta(1.2, 1.8)) allowed the data to have more influence")

print("\n2. Data vs Prior Influence:")
print("   - The more uncertain prior led to a posterior that was more heavily influenced by the data")
print("   - The stronger priors (original and adjusted) maintained more influence in the posterior")
print(f"   - Original prior mean: {prior_stats['mean']:.4f} → Posterior mean: {original_posterior_stats['mean']:.4f}")
print(f"   - Adjusted prior mean: {new_prior_stats['mean']:.4f} → Posterior mean: {adjusted_posterior_stats['mean']:.4f}")
print(f"   - Uncertain prior mean: {less_certain_stats['mean']:.4f} → Posterior mean: {uncertain_posterior_stats['mean']:.4f}")

print("\n3. The Importance of Prior Updates:")
print("   - When new information arrives (like a storm system), Bayesian analysis allows rational prior updates")
print("   - Maintaining the same distribution family (Beta) with updated parameters provides a clean framework")
print("   - Adjusting certainty is as important as adjusting the expected value")

print("\n4. Practical Application in Weather Forecasting:")
print("   - Meteorologists can incorporate both historical data (prior) and current conditions")
print("   - The Beta distribution is ideal for binary events like 'rain' or 'no rain'")
print("   - Continuous updating allows adaptive forecasting as more information becomes available")

# List all generated images
print("\nGenerated Images:")
for img in sorted(os.listdir(save_dir)):
    if img.endswith('.png'):
        print(f"- {img}") 