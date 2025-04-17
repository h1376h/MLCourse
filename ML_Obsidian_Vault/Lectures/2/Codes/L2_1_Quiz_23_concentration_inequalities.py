import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_1_Quiz_23")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Define a function to save figures
def save_figure(filename):
    file_path = os.path.join(save_dir, filename)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {file_path}")
    
# Step 1: Markov's Inequality
print_step_header(1, "Markov's Inequality")

print("Markov's Inequality states that for a non-negative random variable X with mean μ:")
print("P(X ≥ a) ≤ μ/a for any a > 0")
print()
print("Practical implications:")
print("- It provides an upper bound on the probability of a random variable exceeding")
print("  some threshold without knowing the full distribution")
print("- The bound is often loose, especially for symmetric distributions")
print("- Only requires knowledge of the mean")
print()

# Visualize Markov's Inequality
plt.figure(figsize=(10, 6))

# Define a random variable (exponential distribution)
lambda_val = 0.5
mean = 1/lambda_val
x = np.linspace(0, 15, 1000)
pdf = stats.expon.pdf(x, scale=mean)

# Create threshold values
thresholds = np.array([1, 2, 3, 5, 10])
markov_bounds = mean / thresholds
actual_probs = 1 - stats.expon.cdf(thresholds, scale=mean)

# Plot PDF
plt.plot(x, pdf, 'b-', linewidth=2, label='Exponential PDF (μ=2)')

# Plot thresholds and bounds
colors = cm.viridis(np.linspace(0, 0.8, len(thresholds)))
for i, (threshold, bound, actual, color) in enumerate(zip(thresholds, markov_bounds, actual_probs, colors)):
    # Mark the threshold on x-axis
    plt.axvline(x=threshold, color=color, linestyle='--', alpha=0.7)
    
    # Fill the area beyond threshold
    plt.fill_between(x[x >= threshold], 0, pdf[x >= threshold], color=color, alpha=0.3)
    
    # Add annotation with the bound and actual probability
    plt.annotate(f'a={threshold}\nMarkov: P(X≥{threshold}) ≤ {bound:.3f}\nActual: {actual:.3f}',
                xy=(threshold, pdf[x >= threshold][0]/2), 
                xytext=(threshold + 0.5, pdf[x >= threshold][0] + 0.05),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, alpha=0.5),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.title("Visualization of Markov's Inequality", fontsize=15)
plt.xlabel('x', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.xlim(0, 15)
plt.ylim(0, 0.6)
plt.legend()
plt.grid(True)
plt.tight_layout()

save_figure("markov_inequality.png")

# Step 2: Chebyshev's Inequality
print_step_header(2, "Chebyshev's Inequality")

print("Chebyshev's Inequality states that for a random variable X with mean μ and variance σ²:")
print("P(|X - μ| ≥ k·σ) ≤ 1/k² for any k > 0")
print()
print("Improvements over Markov's Inequality:")
print("- It provides tighter bounds for distributions with finite variance")
print("- It accounts for the spread of the distribution via variance")
print("- It's applicable to both sides of the mean (two-sided bound)")
print("- Works for both positive and negative random variables")
print()

# Visualize Chebyshev's Inequality
plt.figure(figsize=(12, 8))

# Define normal distributions with different variances
mean = 0
std_devs = [1, 2]  # Standard deviations
k_values = [1, 2, 3]  # Number of standard deviations away from mean

x = np.linspace(-8, 8, 1000)

for i, std in enumerate(std_devs):
    plt.subplot(2, 1, i+1)
    
    pdf = stats.norm.pdf(x, loc=mean, scale=std)
    plt.plot(x, pdf, 'b-', linewidth=2, label=f'Normal PDF (μ={mean}, σ={std})')
    
    colors = cm.viridis(np.linspace(0, 0.8, len(k_values)))
    for j, (k, color) in enumerate(zip(k_values, colors)):
        # Calculate bounds
        lower_bound = mean - k*std
        upper_bound = mean + k*std
        
        # Calculate Chebyshev bound and actual probability
        chebyshev_bound = 1/(k**2)
        actual_prob = 1 - (stats.norm.cdf(upper_bound, loc=mean, scale=std) - 
                         stats.norm.cdf(lower_bound, loc=mean, scale=std))
        
        # Fill the areas beyond bounds
        mask_below = x <= lower_bound
        mask_above = x >= upper_bound
        
        plt.fill_between(x[mask_below], 0, pdf[mask_below], color=color, alpha=0.3)
        plt.fill_between(x[mask_above], 0, pdf[mask_above], color=color, alpha=0.3)
        
        # Add vertical lines
        plt.axvline(x=lower_bound, color=color, linestyle='--', alpha=0.7)
        plt.axvline(x=upper_bound, color=color, linestyle='--', alpha=0.7)
        
        # Add annotations
        if i == 0:  # Only annotate on the first subplot to avoid clutter
            plt.annotate(f'k={k}\nChebyshev: P(|X-μ|≥{k}σ) ≤ {chebyshev_bound:.3f}\nActual: {actual_prob:.3f}',
                        xy=(upper_bound, pdf[np.abs(x - upper_bound).argmin()]/2), 
                        xytext=(upper_bound + 0.5, pdf[np.abs(x - upper_bound).argmin()] + 0.05),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1, alpha=0.5),
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.title(f"Chebyshev's Inequality for Normal Distribution (σ={std})", fontsize=13)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.xlim(-8, 8)
    plt.legend()
    plt.grid(True)

plt.tight_layout()
save_figure("chebyshev_inequality.png")

# Step 3: Hoeffding's Inequality
print_step_header(3, "Hoeffding's Inequality")

print("Hoeffding's Inequality provides a bound on the probability that the sum of random")
print("variables deviates from its expected value.")
print()
print("For independent bounded random variables X₁, X₂, ..., Xₙ where aᵢ ≤ Xᵢ ≤ bᵢ, and")
print("S = X₁ + X₂ + ... + Xₙ with E[S] = μ:")
print()
print("P(|S - μ| ≥ t) ≤ 2exp(-2t²/Σ(bᵢ-aᵢ)²)")
print()
print("For the sample mean X̄ = S/n with E[X̄] = μ:")
print("P(|X̄ - μ| ≥ ε) ≤ 2exp(-2nε²/Σ((bᵢ-aᵢ)²/n²))")
print()
print("If all Xᵢ are in [0,1], this simplifies to:")
print("P(|X̄ - μ| ≥ ε) ≤ 2exp(-2nε²)")
print()
print("This tells us that the probability of the sample mean deviating from")
print("the true mean by more than ε decreases exponentially with sample size n.")
print()

# Visualize Hoeffding's Inequality for sample means
plt.figure(figsize=(12, 6))

# Setup
epsilon = 0.1  # Deviation from mean
sample_sizes = np.arange(10, 1000, 10)
hoeffding_bounds = 2 * np.exp(-2 * sample_sizes * epsilon**2)

# Create distribution bounds for different scenarios
bounds = [0.5, 1.0]  # Range width: X in [0,0.5] and X in [0,1]
colors = ['b', 'r']
labels = ['X in [0,0.5]', 'X in [0,1]']

for i, (bound, color, label) in enumerate(zip(bounds, colors, labels)):
    # Calculate Hoeffding bound
    hoeffding_bound = 2 * np.exp(-2 * sample_sizes * epsilon**2 / bound**2)
    plt.plot(sample_sizes, hoeffding_bound, color=color, linewidth=2, label=label)

plt.title(f"Hoeffding's Inequality: P(|X̄ - μ| ≥ {epsilon}) vs. Sample Size", fontsize=15)
plt.xlabel('Sample Size (n)', fontsize=12)
plt.ylabel('Probability Bound', fontsize=12)
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.tight_layout()

save_figure("hoeffding_inequality.png")

# Step 4: Concentration Inequalities in ML
print_step_header(4, "Concentration Inequalities in Machine Learning")

print("Applications of Concentration Inequalities in ML:")
print("1. Generalization Error Bounds:")
print("   - Help estimate how well a model performs on unseen data")
print("   - Provide theoretical guarantees for learning algorithms")
print()
print("2. Sample Complexity Analysis:")
print("   - Determine how many samples are needed to achieve a desired error rate")
print("   - Guide dataset size decisions in practical ML applications")
print()
print("3. Feature Selection and Dimensionality Reduction:")
print("   - Provide guarantees for approximation errors when reducing dimensions")
print()
print("4. Model Selection and Regularization:")
print("   - Theoretical foundation for techniques like cross-validation")
print("   - Guidance for selecting hyperparameters")
print()

# Visualize generalization error bounds
plt.figure(figsize=(12, 6))

# Setup
training_sizes = np.arange(10, 10000, 100)
delta = 0.05  # Confidence parameter (95% confidence)

# Generalization error bounds
vc_dim_values = [5, 10, 20, 50]  # Various VC dimensions
colors = cm.viridis(np.linspace(0, 0.8, len(vc_dim_values)))

for i, (d, color) in enumerate(zip(vc_dim_values, colors)):
    # Calculate generalization error bound using VC dimension
    # Based on simplified form of the VC bound
    gen_error = np.sqrt((d * (np.log(2*training_sizes/d) + 1) + np.log(4/delta)) / (2 * training_sizes))
    plt.plot(training_sizes, gen_error, color=color, linewidth=2, 
             label=f'VC Dimension = {d}')

plt.title("Generalization Error Bounds vs. Training Set Size", fontsize=15)
plt.xlabel('Training Set Size', fontsize=12)
plt.ylabel('Generalization Error Bound', fontsize=12)
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.tight_layout()

save_figure("ml_generalization_bounds.png")

# Step 5: Law of Large Numbers vs. Central Limit Theorem
print_step_header(5, "Law of Large Numbers vs. Central Limit Theorem")

print("Law of Large Numbers (LLN):")
print("- Assumptions: Random variables X₁, X₂, ... are independent and identically distributed (i.i.d.)")
print("  with finite mean μ")
print("- Statement: The sample mean X̄ₙ converges to the population mean μ as n → ∞")
print("- Types: Weak LLN (convergence in probability) and Strong LLN (almost sure convergence)")
print()
print("Central Limit Theorem (CLT):")
print("- Assumptions: Random variables X₁, X₂, ... are i.i.d. with finite mean μ and")
print("  finite variance σ²")
print("- Statement: The distribution of √n(X̄ₙ - μ)/σ converges to a standard normal distribution")
print("- Provides information about the rate of convergence and distribution shape")
print()
print("Key Differences:")
print("1. LLN describes convergence of the sample mean to the population mean")
print("2. CLT describes the distribution of the sample mean and its convergence to normality")
print("3. CLT requires finite variance, while weak LLN only requires finite mean")
print("4. CLT gives us confidence intervals and hypothesis tests")
print()

# Visualize the Law of Large Numbers and Central Limit Theorem
plt.figure(figsize=(15, 10))
gs = GridSpec(2, 2, width_ratios=[2, 1])

# Define distributions to sample from
distributions = [
    {"name": "Uniform", "dist": stats.uniform(0, 1), "mean": 0.5, "std": 1/np.sqrt(12)},
    {"name": "Exponential", "dist": stats.expon(scale=1), "mean": 1, "std": 1}
]

np.random.seed(42)  # For reproducibility

for i, dist_info in enumerate(distributions):
    # Sample means for Law of Large Numbers
    ax1 = plt.subplot(gs[i, 0])
    
    # Generate random samples
    max_samples = 10000
    samples = dist_info["dist"].rvs(max_samples)
    
    # Calculate running mean
    sample_indices = np.arange(1, max_samples+1)
    running_means = np.cumsum(samples) / sample_indices
    
    # Plot running mean
    ax1.plot(sample_indices, running_means, 'b-', alpha=0.7, linewidth=1)
    ax1.axhline(y=dist_info["mean"], color='r', linestyle='-', linewidth=2, 
               label=f'Population Mean (μ={dist_info["mean"]})')
    
    # Plot bounds using Hoeffding
    if i == 0:  # Only for uniform which is bounded
        epsilon = 0.05
        hoeffding_upper = dist_info["mean"] + epsilon
        hoeffding_lower = dist_info["mean"] - epsilon
        
        sample_points = np.logspace(1, 4, 100)
        hoeffding_bounds = dist_info["mean"] + epsilon * np.sqrt(np.log(2/0.05) / (2*sample_points))
        
        ax1.fill_between(sample_indices, 
                        dist_info["mean"] - np.sqrt(np.log(2/0.05) / (2*sample_indices)), 
                        dist_info["mean"] + np.sqrt(np.log(2/0.05) / (2*sample_indices)), 
                        color='g', alpha=0.2, label='Hoeffding Bounds (95%)')
    
    ax1.set_title(f"Law of Large Numbers: {dist_info['name']} Distribution", fontsize=14)
    ax1.set_xlabel('Number of Samples (n)', fontsize=12)
    ax1.set_ylabel('Sample Mean', fontsize=12)
    ax1.set_xscale('log')
    ax1.grid(True)
    ax1.legend()
    
    # CLT visualization - distribution of sample means
    ax2 = plt.subplot(gs[i, 1])
    
    # Generate multiple batches and calculate sample means
    n_batches = 10000
    batch_size = 30
    batch_means = np.zeros(n_batches)
    
    for j in range(n_batches):
        batch_samples = dist_info["dist"].rvs(batch_size)
        batch_means[j] = np.mean(batch_samples)
    
    # Standardize means according to CLT
    standardized_means = np.sqrt(batch_size) * (batch_means - dist_info["mean"]) / dist_info["std"]
    
    # Plot histogram of standardized means and compare to standard normal
    ax2.hist(standardized_means, bins=50, density=True, alpha=0.6, 
            label='Standardized Sample Means')
    
    # Plot standard normal PDF
    x = np.linspace(-4, 4, 1000)
    ax2.plot(x, stats.norm.pdf(x), 'r-', linewidth=2, 
            label='Standard Normal PDF')
    
    ax2.set_title(f"Central Limit Theorem: {dist_info['name']} Distribution", fontsize=14)
    ax2.set_xlabel('Standardized Sample Mean', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.grid(True)
    ax2.legend()

plt.tight_layout()
save_figure("lln_vs_clt.png")

# Step 6: Summary and conclusion
print_step_header(6, "Summary of Concentration Inequalities")

print("Concentration inequalities are powerful tools in probability and statistics that provide")
print("bounds on how a random variable deviates from its expected value:")
print()
print("1. Markov's Inequality: Most basic form, requires only finite mean")
print("   P(X ≥ a) ≤ E[X]/a for non-negative X")
print()
print("2. Chebyshev's Inequality: Improves on Markov by considering variance")
print("   P(|X - E[X]| ≥ kσ) ≤ 1/k² for any random variable with finite variance")
print()
print("3. Hoeffding's Inequality: Provides exponentially decreasing bounds for sums")
print("   of bounded independent random variables")
print("   P(|X̄ - μ| ≥ ε) ≤ 2exp(-2nε²) for bounded random variables")
print()
print("4. Applications in Machine Learning:")
print("   - Generalization error bounds")
print("   - PAC learning theory")
print("   - Feature selection guarantees")
print("   - Model complexity analysis")
print()
print("5. Law of Large Numbers vs. Central Limit Theorem:")
print("   - LLN: Sample mean converges to true mean (convergence guarantee)")
print("   - CLT: Provides the distribution of the sample mean (distribution information)")
print()
print("These inequalities are fundamental for providing theoretical guarantees in")
print("statistical learning theory and algorithm design.")

# Create a single comprehensive figure summarizing all inequalities
plt.figure(figsize=(12, 10))
gs = GridSpec(3, 1, height_ratios=[1, 1, 1])

# Sample size and deviation parameter
n_values = np.logspace(1, 4, 100)
epsilon = 0.1

# Markov's Inequality (for a specific case)
ax1 = plt.subplot(gs[0])
# For a random variable with mean 1, P(X >= a)
a_values = np.linspace(1, 5, 100)
markov_bound = 1/a_values
ax1.plot(a_values, markov_bound, 'b-', linewidth=2)
ax1.set_title("Markov's Inequality: P(X ≥ a) ≤ μ/a", fontsize=14)
ax1.set_xlabel('Threshold (a)', fontsize=12)
ax1.set_ylabel('Probability Bound', fontsize=12)
ax1.grid(True)

# Chebyshev's Inequality 
ax2 = plt.subplot(gs[1])
# For different k values (standard deviations)
k_values = np.linspace(1, 5, 100)
chebyshev_bound = 1/(k_values**2)
ax2.plot(k_values, chebyshev_bound, 'g-', linewidth=2)
ax2.set_title("Chebyshev's Inequality: P(|X - μ| ≥ kσ) ≤ 1/k²", fontsize=14)
ax2.set_xlabel('Number of Standard Deviations (k)', fontsize=12)
ax2.set_ylabel('Probability Bound', fontsize=12)
ax2.grid(True)

# Hoeffding's Inequality
ax3 = plt.subplot(gs[2])
# For varying sample sizes
hoeffding_bound = 2 * np.exp(-2 * n_values * epsilon**2)
ax3.plot(n_values, hoeffding_bound, 'r-', linewidth=2)
ax3.set_title(f"Hoeffding's Inequality: P(|X̄ - μ| ≥ {epsilon}) ≤ 2exp(-2nε²)", fontsize=14)
ax3.set_xlabel('Sample Size (n)', fontsize=12)
ax3.set_ylabel('Probability Bound', fontsize=12)
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.grid(True)

plt.tight_layout()
save_figure("summary_inequalities.png")

print("\nAll visualizations and explanations for concentration inequalities are complete!") 