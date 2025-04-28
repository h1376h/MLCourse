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

def print_task(task_number, task_description):
    """Print a formatted task header."""
    print(f"\n{'*' * 60}")
    print(f"TASK {task_number}: {task_description}")
    print(f"{'*' * 60}")

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_5_Quiz_18")
os.makedirs(save_dir, exist_ok=True)

# Function to plot beta distributions
def plot_beta(alphas, betas, labels, title, filename, x_label="Probability of Rain", include_mean=False, 
              include_mode=False, include_legend=True, xlim=None, highlight_points=None, data_line=None):
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
    
    # Add data frequency line if provided
    if data_line is not None:
        plt.axvline(data_line, color='red', linestyle='-', alpha=0.5, linewidth=2)
        plt.text(data_line+0.01, plt.ylim()[1]*0.8, f'Data: {data_line:.2f}', 
                color='red', rotation=90, verticalalignment='top')
    
    # Add highlight points if provided
    if highlight_points is not None:
        for point, label, color in highlight_points:
            plt.plot(point, 0, 'o', markersize=10, color=color)
            plt.text(point, 0.5, label, color=color, ha='center')
    
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
    """Calculate key statistics for a Beta distribution."""
    mean = alpha / (alpha + beta)
    var = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
    std = np.sqrt(var)
    
    # Mode exists only when alpha > 1
    if alpha > 1 and (alpha + beta - 2) > 0:
        mode = (alpha - 1) / (alpha + beta - 2)
    else:
        mode = "N/A"
    
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
    """Calculate the effective sample size of a Beta distribution."""
    return alpha + beta - 2  # Subtract 2 for the uniform prior contribution

# Function to calculate the influence of prior vs data in a posterior
def calculate_influence(prior_alpha, prior_beta, data_successes, data_failures):
    """Calculate the relative influence of prior vs data in the posterior."""
    prior_strength = prior_alpha + prior_beta
    data_strength = data_successes + data_failures
    
    prior_influence = prior_strength / (prior_strength + data_strength)
    data_influence = data_strength / (prior_strength + data_strength)
    
    return prior_influence, data_influence

# ==============================
# STEP 0: Presenting the Problem
# ==============================
print_step_header(0, "The Problem")

print("A meteorologist is forecasting whether it will rain tomorrow.")
print("Based on historical data for this time of year, she initially uses a Beta(2, 8) prior")
print("for the probability of rain. However, she receives a special weather report indicating")
print("a storm system is approaching.")
print("\nThe tasks are:")
print("1. Explain why the meteorologist should update her prior given the new information")
print("2. Calculate new Beta parameters to reflect a 40% probability while maintaining certainty")
print("3. Suggest an approach for increased uncertainty while maintaining 40% probability")
print("4. Calculate posteriors after observing 3/5 rainy days and compare effects of priors")

# ==============================
# STEP 1: Understanding the Problem - Initial Prior Analysis
# ==============================
print_step_header(1, "Understanding the Problem - Initial Prior Analysis")

# Calculate and display prior statistics
original_prior = (2, 8)
prior_stats = beta_stats(*original_prior)

print("Original Beta(2, 8) Prior Analysis:")
print(f"Mean (Expected probability): α/(α+β) = {original_prior[0]}/{sum(original_prior)} = {prior_stats['mean']:.4f}")
print(f"Mode (Most likely probability): (α-1)/(α+β-2) = {original_prior[0]-1}/{sum(original_prior)-2} = {prior_stats['mode']}")
print(f"Standard Deviation: {prior_stats['std_dev']:.4f}")
print(f"95% Credible Interval: ({prior_stats['95%_CI'][0]:.4f}, {prior_stats['95%_CI'][1]:.4f})")
print(f"Effective Sample Size: α+β-2 = {beta_effective_sample_size(*original_prior)}")

# Visualize the original prior
plot_beta([2], [8], ["Beta(2, 8)"], "Original Beta(2, 8) Prior for Rain Probability", 
          "original_prior.png", include_mean=True, include_mode=True)

# ==============================
# TASK 1: Explain Why the Prior Should be Updated
# ==============================
print_task(1, "Explaining Why the Prior Should be Updated")

print("The meteorologist should update her prior distribution because:")
print("\n1. Bayesian Principle: When new relevant information becomes available, beliefs should be")
print("   updated accordingly, even before collecting new numerical data.")
print("\n2. Specificity: The new information (approaching storm system) is specific to this forecast")
print("   period, while the original prior based on historical data represents average conditions.")
print("\n3. Ignoring Information: Using only the historical prior when a storm is approaching would")
print("   mean ignoring valuable information, leading to potentially inaccurate forecasts.")
print("\n4. Information Relevance: The approaching storm directly affects the physical conditions")
print("   that influence rain probability, making it highly relevant information.")
print("\n5. Expert Knowledge Integration: Meteorological expertise about storm systems and their")
print("   effects on local weather should be incorporated into the forecast.")

# ==============================
# TASK 2: Calculating New Prior Parameters for 40% Probability
# ==============================
print_task(2, "Calculating New Prior Parameters for 40% Probability")

print_substep("Mathematical Derivation")
print("To adjust the prior to reflect a 40% probability of rain while maintaining certainty:")
print("1. For a Beta(α, β) distribution, the mean is: μ = α/(α+β)")
print("2. The strength/certainty is represented by: s = α+β")
print("3. We want to maintain the same strength as the original prior: α+β = 2+8 = 10")
print("4. We want to adjust the mean to 40%: α/(α+β) = 0.4")
print("5. Solving for α with these constraints:")
print("   α/(α+β) = 0.4")
print("   α = 0.4(α+β)")
print("   α = 0.4(10)")
print("   α = 4")
print("6. And β = 10-α = 10-4 = 6")

# Original prior has mean = α/(α+β) = 2/(2+8) = 2/10 = 0.2
# We want a new prior with mean = 0.4 and same effective sample size
original_strength = original_prior[0] + original_prior[1]  # α+β
new_mean = 0.4  # Target mean

# If mean = α/(α+β) = 0.4 and (α+β) = 10, then α = 0.4*10 = 4
new_alpha = new_mean * original_strength
new_beta = original_strength - new_alpha
new_prior = (new_alpha, new_beta)
new_prior_stats = beta_stats(*new_prior)

print("\nNew Prior Calculations:")
print(f"- Original prior: Beta({original_prior[0]}, {original_prior[1]})")
print(f"- Original strength (α+β): {original_strength}")
print(f"- Target mean: {new_mean}")
print(f"- New α = target_mean × strength = {new_mean} × {original_strength} = {new_alpha}")
print(f"- New β = strength - α = {original_strength} - {new_alpha} = {new_beta}")
print(f"- New prior: Beta({new_alpha}, {new_beta})")

print("\nVerifying the New Beta(4, 6) Prior:")
print(f"Mean: α/(α+β) = {new_alpha}/{new_alpha+new_beta} = {new_prior_stats['mean']:.4f} ✓")
print(f"Mode: (α-1)/(α+β-2) = {new_alpha-1}/{new_alpha+new_beta-2} = {new_prior_stats['mode']}")
print(f"Standard Deviation: {new_prior_stats['std_dev']:.4f}")
print(f"95% Credible Interval: ({new_prior_stats['95%_CI'][0]:.4f}, {new_prior_stats['95%_CI'][1]:.4f})")
print(f"Effective Sample Size: α+β-2 = {beta_effective_sample_size(*new_prior)}")

# Compare original and new priors
plot_beta([2, 4], [8, 6], ["Original Beta(2, 8)", "Adjusted Beta(4, 6)"],
          "Comparing Original and Adjusted Priors",
          "comparing_priors.png", include_mean=True, include_mode=True)

# ==============================
# TASK 3: Handling Increased Uncertainty
# ==============================
print_task(3, "Handling Increased Uncertainty")

print_substep("Mathematical Approach")
print("To increase uncertainty while maintaining the same mean of 0.4:")
print("1. The mean of a Beta(α, β) is: μ = α/(α+β) = 0.4")
print("2. This gives us: α = 0.4(α+β), or α = 0.4β/0.6")
print("3. The strength (α+β) represents certainty - smaller values mean less certainty")
print("4. We want α/(α+β) = 0.4 but with lower α+β than the original 10")
print("5. If we choose (α+β) = 3 (much smaller than 10):")
print("   α = 0.4(3) = 1.2")
print("   β = 3-1.2 = 1.8")
print("6. This gives Beta(1.2, 1.8) with the same mean but much less certainty")

# Create a more uncertain prior with same mean but less certainty
less_certain_alpha = 1.2  # These values maintain mean ≈ 0.4 but with higher variance
less_certain_beta = 1.8
less_certain_prior = (less_certain_alpha, less_certain_beta)
less_certain_stats = beta_stats(*less_certain_prior)

print("\nUncertain Prior Calculations:")
print(f"- Target mean: {new_mean}")
print(f"- Selected reduced strength (α+β): {less_certain_alpha + less_certain_beta}")
print(f"- New α = {less_certain_alpha}")
print(f"- New β = {less_certain_beta}")
print(f"- Uncertain prior: Beta({less_certain_alpha}, {less_certain_beta})")

print("\nVerifying the Uncertain Beta(1.2, 1.8) Prior:")
print(f"Mean: α/(α+β) = {less_certain_alpha}/{less_certain_alpha+less_certain_beta} = {less_certain_stats['mean']:.4f} ✓")
print(f"Standard Deviation: {less_certain_stats['std_dev']:.4f} (higher than adjusted prior's {new_prior_stats['std_dev']:.4f})")
print(f"95% Credible Interval: ({less_certain_stats['95%_CI'][0]:.4f}, {less_certain_stats['95%_CI'][1]:.4f})")
print(f"Effective Sample Size: α+β-2 = {beta_effective_sample_size(*less_certain_prior)} (much lower than 8)")

# Compare all three priors
plot_beta([2, 4, 1.2], [8, 6, 1.8], 
          ["Original Beta(2, 8)", "Adjusted Beta(4, 6)", "Uncertain Beta(1.2, 1.8)"],
          "Comparing Priors with Different Certainty Levels",
          "comparing_certainty.png", include_mean=True)

# ==============================
# TASK 4: Posterior Calculation and Comparison
# ==============================
print_task(4, "Posterior Calculation and Comparison")

# Observed data: 3 rainy days out of 5
rainy_days = 3
total_days = 5
dry_days = total_days - rainy_days
data_frequency = rainy_days / total_days

print(f"Observed data: {rainy_days} rainy days out of {total_days} days (frequency: {data_frequency:.2f})")

print_substep("Mathematical Calculation of Posteriors")
print("For a Beta(α, β) prior with observed data of s successes and f failures:")
print("The posterior is Beta(α+s, β+f)")
print("1. Original Prior Beta(2, 8) with data (3 rainy, 2 dry):")
print("   Posterior = Beta(2+3, 8+2) = Beta(5, 10)")
print("2. Adjusted Prior Beta(4, 6) with data (3 rainy, 2 dry):")
print("   Posterior = Beta(4+3, 6+2) = Beta(7, 8)")
print("3. Uncertain Prior Beta(1.2, 1.8) with data (3 rainy, 2 dry):")
print("   Posterior = Beta(1.2+3, 1.8+2) = Beta(4.2, 3.8)")

# For Beta(α, β) prior with data (successes, failures), posterior is Beta(α+successes, β+failures)
original_posterior = (original_prior[0] + rainy_days, original_prior[1] + dry_days)
adjusted_posterior = (new_prior[0] + rainy_days, new_prior[1] + dry_days)
uncertain_posterior = (less_certain_prior[0] + rainy_days, less_certain_prior[1] + dry_days)

# Calculate posterior statistics
original_posterior_stats = beta_stats(*original_posterior)
adjusted_posterior_stats = beta_stats(*adjusted_posterior)
uncertain_posterior_stats = beta_stats(*uncertain_posterior)

# Calculate the relative influence of prior vs data
original_prior_influence, original_data_influence = calculate_influence(
    original_prior[0], original_prior[1], rainy_days, dry_days)
adjusted_prior_influence, adjusted_data_influence = calculate_influence(
    new_prior[0], new_prior[1], rainy_days, dry_days)
uncertain_prior_influence, uncertain_data_influence = calculate_influence(
    less_certain_prior[0], less_certain_prior[1], rainy_days, dry_days)

# Display posterior results
print("\nPosterior Results with Detailed Statistics:")
print("\n1. From Original Prior Beta(2, 8):")
print(f"   - Posterior: Beta({original_posterior[0]}, {original_posterior[1]})")
print(f"   - Mean: {original_posterior_stats['mean']:.4f} (vs data frequency {data_frequency:.4f})")
print(f"   - Difference from data: {abs(original_posterior_stats['mean'] - data_frequency):.4f}")
print(f"   - 95% Credible Interval: ({original_posterior_stats['95%_CI'][0]:.4f}, {original_posterior_stats['95%_CI'][1]:.4f})")
print(f"   - Prior influence: {original_prior_influence:.2%}, Data influence: {original_data_influence:.2%}")

print("\n2. From Adjusted Prior Beta(4, 6):")
print(f"   - Posterior: Beta({adjusted_posterior[0]}, {adjusted_posterior[1]})")
print(f"   - Mean: {adjusted_posterior_stats['mean']:.4f} (vs data frequency {data_frequency:.4f})")
print(f"   - Difference from data: {abs(adjusted_posterior_stats['mean'] - data_frequency):.4f}")
print(f"   - 95% Credible Interval: ({adjusted_posterior_stats['95%_CI'][0]:.4f}, {adjusted_posterior_stats['95%_CI'][1]:.4f})")
print(f"   - Prior influence: {adjusted_prior_influence:.2%}, Data influence: {adjusted_data_influence:.2%}")

print("\n3. From Uncertain Prior Beta(1.2, 1.8):")
print(f"   - Posterior: Beta({uncertain_posterior[0]}, {uncertain_posterior[1]})")
print(f"   - Mean: {uncertain_posterior_stats['mean']:.4f} (vs data frequency {data_frequency:.4f})")
print(f"   - Difference from data: {abs(uncertain_posterior_stats['mean'] - data_frequency):.4f}")
print(f"   - 95% Credible Interval: ({uncertain_posterior_stats['95%_CI'][0]:.4f}, {uncertain_posterior_stats['95%_CI'][1]:.4f})")
print(f"   - Prior influence: {uncertain_prior_influence:.2%}, Data influence: {uncertain_data_influence:.2%}")

# Visualize the posteriors with data line
plot_beta([original_posterior[0], adjusted_posterior[0], uncertain_posterior[0]],
          [original_posterior[1], adjusted_posterior[1], uncertain_posterior[1]],
          [f"From Beta(2, 8): Beta({original_posterior[0]}, {original_posterior[1]})",
           f"From Beta(4, 6): Beta({adjusted_posterior[0]}, {adjusted_posterior[1]})",
           f"From Beta(1.2, 1.8): Beta({uncertain_posterior[0]}, {uncertain_posterior[1]})"],
          "Posterior Distributions After Observing 3/5 Rainy Days",
          "posteriors.png", include_mean=True, data_line=data_frequency)

# Accuracy analysis
posteriors = [
    ("Original Prior (Beta(2,8))", original_posterior_stats['mean'], abs(original_posterior_stats['mean'] - data_frequency)),
    ("Adjusted Prior (Beta(4,6))", adjusted_posterior_stats['mean'], abs(adjusted_posterior_stats['mean'] - data_frequency)),
    ("Uncertain Prior (Beta(1.2,1.8))", uncertain_posterior_stats['mean'], abs(uncertain_posterior_stats['mean'] - data_frequency))
]
posteriors.sort(key=lambda x: x[2])  # Sort by accuracy (smallest difference)

print("\nAccuracy Ranking (closest to observed data frequency):")
for i, (prior_name, posterior_mean, difference) in enumerate(posteriors, 1):
    print(f"{i}. {prior_name}: posterior mean = {posterior_mean:.4f}, difference = {difference:.4f}")

print(f"\nThe most accurate posterior comes from {posteriors[0][0]}, with posterior mean {posteriors[0][1]:.4f}")
print(f"This is closest to the observed frequency of {data_frequency:.4f}")

# Create bar chart comparing posterior means to data frequency
plt.figure(figsize=(10, 6))
labels = ['Original\nBeta(2,8)', 'Adjusted\nBeta(4,6)', 'Uncertain\nBeta(1.2,1.8)', 'Data\nFrequency']
values = [original_posterior_stats['mean'], adjusted_posterior_stats['mean'], uncertain_posterior_stats['mean'], data_frequency]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', 'red']
bars = plt.bar(labels, values, color=colors, alpha=0.7)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom')

plt.ylim(0, 0.7)
plt.ylabel('Probability of Rain', fontsize=12)
plt.title('Comparing Posterior Means with Observed Frequency', fontsize=14)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "posterior_accuracy.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"Plot saved to: {os.path.join(save_dir, 'posterior_accuracy.png')}")

# Create visualization showing prior-to-posterior updating for each case
# Original Prior --> Posterior
plot_beta([2, original_posterior[0]], 
          [8, original_posterior[1]],
          ["Prior Beta(2, 8)", f"Posterior Beta({original_posterior[0]}, {original_posterior[1]})"],
          "Bayesian Updating with Original Prior",
          "updating_original.png", include_mean=True, data_line=data_frequency)

# Adjusted Prior --> Posterior
plot_beta([4, adjusted_posterior[0]], 
          [6, adjusted_posterior[1]],
          ["Prior Beta(4, 6)", f"Posterior Beta({adjusted_posterior[0]}, {adjusted_posterior[1]})"],
          "Bayesian Updating with Adjusted Prior",
          "updating_adjusted.png", include_mean=True, data_line=data_frequency)

# Uncertain Prior --> Posterior
plot_beta([1.2, uncertain_posterior[0]], 
          [1.8, uncertain_posterior[1]],
          ["Prior Beta(1.2, 1.8)", f"Posterior Beta({uncertain_posterior[0]}, {uncertain_posterior[1]})"],
          "Bayesian Updating with Uncertain Prior",
          "updating_uncertain.png", include_mean=True, data_line=data_frequency)

# Create a visualization showing influence of prior vs data
plt.figure(figsize=(10, 6))
priors = ['Original\nBeta(2,8)', 'Adjusted\nBeta(4,6)', 'Uncertain\nBeta(1.2,1.8)']
prior_influences = [original_prior_influence, adjusted_prior_influence, uncertain_prior_influence]
data_influences = [original_data_influence, adjusted_data_influence, uncertain_data_influence]

x = np.arange(len(priors))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, prior_influences, width, label='Prior Influence', color='#1f77b4', alpha=0.7)
rects2 = ax.bar(x + width/2, data_influences, width, label='Data Influence', color='#ff7f0e', alpha=0.7)

# Add labels and titles
ax.set_ylabel('Influence Proportion', fontsize=12)
ax.set_title('Relative Influence of Prior vs Data in Posterior', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(priors)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1%}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "influence_analysis.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"Plot saved to: {os.path.join(save_dir, 'influence_analysis.png')}")

# Create a summary visualization showing all three prior-posterior pairs with likelihood
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

# Add vertical line at the data frequency
plt.axvline(data_frequency, color='red', linestyle='-', linewidth=2, alpha=0.5, 
            label=f"Data Frequency ({data_frequency:.2f})")

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

print("\n3. Most Accurate Posterior Estimate:")
print("   - The Uncertain Prior (Beta(1.2,1.8)) resulted in the most accurate posterior")
print(f"   - Its posterior mean of {uncertain_posterior_stats['mean']:.4f} was closest to the observed frequency of {data_frequency:.4f}")
print("   - This is because this prior had the right balance between prior expectations and allowing data influence")

print("\n4. The Importance of Prior Updates:")
print("   - When new information arrives (like a storm system), Bayesian analysis allows rational prior updates")
print("   - Maintaining the same distribution family (Beta) with updated parameters provides a clean framework")
print("   - Adjusting certainty is as important as adjusting the expected value")

print("\n5. Practical Application in Weather Forecasting:")
print("   - Meteorologists can incorporate both historical data (prior) and current conditions")
print("   - The Beta distribution is ideal for binary events like 'rain' or 'no rain'")
print("   - Continuous updating allows adaptive forecasting as more information becomes available")

# List all generated images
print("\nGenerated Images:")
for img in sorted(os.listdir(save_dir)):
    if img.endswith('.png'):
        print(f"- {img}")

print("\nThis completes the solution to Question 18 on Updating Prior Distributions in Weather Forecasting.") 