import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_1_Quiz_20")
os.makedirs(save_dir, exist_ok=True)

# Set a clean style for plots
plt.style.use('seaborn-v0_8-whitegrid')

print("# Statement 5: The Law of Large Numbers guarantees that as sample size increases, the sample mean will exactly equal the population mean.")

# Image 1: Simulate the Law of Large Numbers with multiple trials
np.random.seed(42)
population_mean = 50
population_std = 10

# Generate increasing sample sizes
sample_sizes = np.unique(np.logspace(1, 4, 40).astype(int))
num_trials = 3
sample_means = np.zeros((num_trials, len(sample_sizes)))

# Generate multiple trials
for trial in range(num_trials):
    for i, n in enumerate(sample_sizes):
        sample = np.random.normal(population_mean, population_std, n)
        sample_means[trial, i] = np.mean(sample)

# Create the plot
fig1, ax1 = plt.subplots(figsize=(10, 6))

# Plot multiple sample paths
colors = ['blue', 'green', 'purple']
for i in range(num_trials):
    ax1.plot(sample_sizes, sample_means[i], '-', color=colors[i], alpha=0.7, 
            label=f'Sample Path {i+1}')

# Add population mean line
ax1.axhline(y=population_mean, color='red', linestyle='-', linewidth=2, 
           label=f'Population Mean (μ = {population_mean})')

# Add confidence bands
std_errors = population_std / np.sqrt(sample_sizes)
ax1.fill_between(sample_sizes, 
                population_mean - 2*std_errors,
                population_mean + 2*std_errors,
                color='gray', alpha=0.2,
                label='95% Confidence Band')

ax1.set_xscale('log')
ax1.set_xlabel('Sample Size (log scale)', fontsize=12)
ax1.set_ylabel('Sample Mean', fontsize=12)
ax1.set_title('Law of Large Numbers: Convergence of Sample Means', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# Save the figure
lln_img_path1 = os.path.join(save_dir, "statement5_sample_paths.png")
plt.savefig(lln_img_path1, dpi=300, bbox_inches='tight')
plt.close()

# Image 2: Distribution of sample means for different sample sizes
np.random.seed(123)
population = stats.norm(loc=50, scale=10)

# Generate samples of different sizes
small_samples = [population.rvs(size=10) for _ in range(1000)]
medium_samples = [population.rvs(size=100) for _ in range(1000)]
large_samples = [population.rvs(size=1000) for _ in range(1000)]

# Calculate sample means
small_means = [np.mean(sample) for sample in small_samples]
medium_means = [np.mean(sample) for sample in medium_samples]
large_means = [np.mean(sample) for sample in large_samples]

# Plot distributions of sample means
fig2, axes = plt.subplots(3, 1, figsize=(10, 12))

# Small samples
axes[0].hist(small_means, bins=30, alpha=0.7, color='blue')
axes[0].axvline(x=population_mean, color='red', linestyle='--', linewidth=2, label='Population Mean')
axes[0].set_title(f'Distribution of Sample Means (n=10)\nStd Dev: {np.std(small_means):.4f}', fontsize=12)
axes[0].legend()

# Medium samples
axes[1].hist(medium_means, bins=30, alpha=0.7, color='green')
axes[1].axvline(x=population_mean, color='red', linestyle='--', linewidth=2, label='Population Mean')
axes[1].set_title(f'Distribution of Sample Means (n=100)\nStd Dev: {np.std(medium_means):.4f}', fontsize=12)
axes[1].legend()

# Large samples
axes[2].hist(large_means, bins=30, alpha=0.7, color='purple')
axes[2].axvline(x=population_mean, color='red', linestyle='--', linewidth=2, label='Population Mean')
axes[2].set_title(f'Distribution of Sample Means (n=1000)\nStd Dev: {np.std(large_means):.4f}', fontsize=12)
axes[2].legend()

fig2.suptitle('Convergence of Sample Means Distribution', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save the figure
lln_img_path2 = os.path.join(save_dir, "statement5_mean_distributions.png")
plt.savefig(lln_img_path2, dpi=300, bbox_inches='tight')
plt.close()

# Image 3: Distribution of single sample for different trials
np.random.seed(42)

# Create a plot to show the distance from true mean over trials
fig3, ax3 = plt.subplots(figsize=(10, 6))

# Fix a large sample size
sample_size = 10000
num_experiments = 50

# Generate absolute errors for multiple experiments
errors = []
for i in range(num_experiments):
    sample = np.random.normal(population_mean, population_std, sample_size)
    sample_mean = np.mean(sample)
    error = abs(sample_mean - population_mean)
    errors.append(error)

# Plot the errors
ax3.bar(range(1, num_experiments+1), errors, alpha=0.7, color='blue')
ax3.axhline(y=np.mean(errors), color='red', linestyle='--', 
           label=f'Average Error: {np.mean(errors):.4f}')

ax3.set_xlabel('Experiment Number', fontsize=12)
ax3.set_ylabel('|Sample Mean - Population Mean|', fontsize=12)
ax3.set_title(f'Absolute Error with Large Sample Size (n={sample_size})', fontsize=14)
ax3.grid(True, alpha=0.3)
ax3.legend()

# Save the figure
lln_img_path3 = os.path.join(save_dir, "statement5_absolute_errors.png")
plt.savefig(lln_img_path3, dpi=300, bbox_inches='tight')
plt.close()

# Explain the Law of Large Numbers - Print to terminal
print("#### Mathematical Analysis of the Law of Large Numbers")
print("The Law of Large Numbers (LLN) states that as the sample size n increases,")
print("the sample mean X̄ₙ converges in probability to the population mean μ.")
print("")
print("Mathematically, for any ε > 0:")
print("lim[n→∞] P(|X̄ₙ - μ| > ε) = 0")
print("")
print("#### Types of Convergence:")
print("1. Convergence in probability: The probability of the sample mean deviating from")
print("   the population mean by more than any fixed amount approaches zero as n increases")
print("2. This is different from exact equality, which would require |X̄ₙ - μ| = 0")
print("")
print("#### Numerical Demonstration:")
print("Population parameters: μ = 50, σ = 10")
print("")
print("Looking at sample means for different sample sizes across three trials:")
for trial in range(num_trials):
    print(f"Trial {trial+1}:")
    print(f"  n = {sample_sizes[0]:6d}: X̄ₙ = {sample_means[trial, 0]:.4f}, |X̄ₙ - μ| = {abs(sample_means[trial, 0] - population_mean):.4f}")
    print(f"  n = {sample_sizes[19]:6d}: X̄ₙ = {sample_means[trial, 19]:.4f}, |X̄ₙ - μ| = {abs(sample_means[trial, 19] - population_mean):.4f}")
    print(f"  n = {sample_sizes[-1]:6d}: X̄ₙ = {sample_means[trial, -1]:.4f}, |X̄ₙ - μ| = {abs(sample_means[trial, -1] - population_mean):.4f}")
print("")
print("Standard error (σ/√n) for different sample sizes:")
print(f"  n = 10: σ/√n = {10/np.sqrt(10):.4f}")
print(f"  n = 100: σ/√n = {10/np.sqrt(100):.4f}")
print(f"  n = 1000: σ/√n = {10/np.sqrt(1000):.4f}")
print(f"  n = 10000: σ/√n = {10/np.sqrt(10000):.4f}")
print("")
print("Even with very large samples (n = 10,000), we observe:")
print(f"Average absolute error across {num_experiments} experiments: {np.mean(errors):.6f}")
print(f"Maximum absolute error: {max(errors):.6f}")
print(f"Minimum absolute error: {min(errors):.6f}")
print("")
print("#### Key Properties of the Law of Large Numbers:")
print("1. Sample means converge toward the population mean as sample size increases")
print("2. The convergence is probabilistic, not deterministic")
print("3. The standard error decreases at a rate of 1/√n")
print("4. Even with large samples, some non-zero error typically remains")
print("5. Different samples of the same size will yield different means")
print("")
print("#### Visual Verification:")
print(f"1. Convergence of sample means: {lln_img_path1}")
print(f"2. Distribution of sample means for different sample sizes: {lln_img_path2}")
print(f"3. Absolute errors with large sample size: {lln_img_path3}")
print("")
print("#### Conclusion:")
print("The Law of Large Numbers ensures that sample means get arbitrarily close to the")
print("population mean with high probability as sample size increases, but it does not")
print("guarantee exact equality for any finite sample size.")
print("")
print("Therefore, Statement 5 is FALSE.") 