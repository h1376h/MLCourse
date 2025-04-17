import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from scipy.stats import norm

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_3_13")
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
print("- We have a random sample X₁, X₂, ..., Xₙ from a distribution with unknown mean μ and known variance σ²")
print("- The point estimator for μ is the sample mean: μ̂ = (1/n)∑ᵢXᵢ")
print()
print("Task: Calculate the bias and variance of this estimator")
print()

# Step 2: Define bias and variance
print_step_header(2, "Understanding Bias and Variance")

print("For an estimator θ̂ of a parameter θ:")
print()
print("Bias:")
print("- Bias(θ̂) = E[θ̂] - θ")
print("- It measures the systematic error or deviation of the estimator from the true parameter")
print("- An estimator is unbiased if its bias is 0, i.e., E[θ̂] = θ")
print()
print("Variance:")
print("- Var(θ̂) = E[(θ̂ - E[θ̂])²]")
print("- It measures the dispersion or spread of the estimator around its expected value")
print("- Lower variance means the estimator is more precise")
print()

# Create a visual explanation of bias and variance
plt.figure(figsize=(12, 6))

# Define true parameter and sample size
mu_true = 5
sigma = 2
n_samples = 5
n_simulations = 10000

# Create 1000 simulations of the sample mean
np.random.seed(42)
sample_means = [np.mean(np.random.normal(mu_true, sigma, n_samples)) for _ in range(n_simulations)]

# Calculate the expected value of the sample mean
mean_of_means = np.mean(sample_means)
variance_of_means = np.var(sample_means)

# Plot the sampling distribution
plt.subplot(1, 2, 1)
plt.hist(sample_means, bins=50, alpha=0.7, density=True, color='skyblue')
plt.axvline(mu_true, color='red', linestyle='--', linewidth=2, label='True μ = ' + str(mu_true))
plt.axvline(mean_of_means, color='green', linestyle='-', linewidth=2, label='E[μ̂] = ' + f"{mean_of_means:.2f}")
plt.title('Sampling Distribution of μ̂ = (1/n)∑ᵢXᵢ', fontsize=12)
plt.xlabel('Sample Mean (μ̂)', fontsize=10)
plt.ylabel('Density', fontsize=10)
plt.legend()
plt.grid(True)

# Plot bias and variance illustration
plt.subplot(1, 2, 2)
x = np.linspace(mu_true - 4*sigma/np.sqrt(n_samples), mu_true + 4*sigma/np.sqrt(n_samples), 1000)
y = norm.pdf(x, mean_of_means, np.sqrt(variance_of_means))
plt.plot(x, y, 'k-', linewidth=2, label=f'N({mean_of_means:.2f}, {variance_of_means:.2f})')
plt.axvline(mu_true, color='red', linestyle='--', linewidth=2, label='True μ = ' + str(mu_true))
plt.axvline(mean_of_means, color='green', linestyle='-', linewidth=2, label='E[μ̂] = ' + f"{mean_of_means:.2f}")

# Represent the bias
bias = mean_of_means - mu_true
plt.annotate('', xy=(mu_true, 0.15), xytext=(mean_of_means, 0.15),
             arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
plt.text((mu_true + mean_of_means)/2, 0.16, f'Bias = {bias:.3f}', 
         fontsize=12, ha='center', va='bottom', color='purple')

# Represent the variance
std_dev = np.sqrt(variance_of_means)
plt.annotate('', xy=(mean_of_means - std_dev, 0.1), xytext=(mean_of_means + std_dev, 0.1),
             arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
plt.text(mean_of_means, 0.11, f'σ² = {variance_of_means:.3f}', 
         fontsize=12, ha='center', va='bottom', color='blue')

plt.title('Bias and Variance Illustration', fontsize=12)
plt.xlabel('Sample Mean (μ̂)', fontsize=10)
plt.ylabel('Density', fontsize=10)
plt.legend()
plt.grid(True)

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "bias_variance_illustration.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 3: Calculate bias for sample mean
print_step_header(3, "Calculating the Bias of the Sample Mean")

print("For the sample mean μ̂ = (1/n)∑ᵢXᵢ, the bias is given by:")
print("Bias(μ̂) = E[μ̂] - μ")
print()
print("Step 1: Calculate E[μ̂]")
print("E[μ̂] = E[(1/n)∑ᵢXᵢ]")
print("     = (1/n)∑ᵢE[Xᵢ]  (by linearity of expectation)")
print("     = (1/n)∑ᵢμ      (since E[Xᵢ] = μ for all i)")
print("     = (1/n)(n·μ)")
print("     = μ")
print()
print("Step 2: Calculate the bias")
print("Bias(μ̂) = E[μ̂] - μ")
print("        = μ - μ")
print("        = 0")
print()
print("Therefore, the sample mean is an unbiased estimator of μ.")
print()

# Step 4: Verify unbiasedness with simulations
print_step_header(4, "Verifying Unbiasedness with Simulations")

# Simulate for different sample sizes
np.random.seed(42)
sample_sizes = [5, 10, 50, 100, 500, 1000]
num_simulations = 10000
true_mean = 5
sigma = 2

results = {}
for n in sample_sizes:
    sample_means = []
    for _ in range(num_simulations):
        sample = np.random.normal(true_mean, sigma, n)
        sample_means.append(np.mean(sample))
    
    mean_of_means = np.mean(sample_means)
    bias = mean_of_means - true_mean
    results[n] = {'mean': mean_of_means, 'bias': bias}
    
    print(f"Sample size n = {n}:")
    print(f"  - E[μ̂] (empirical) = {mean_of_means:.6f}")
    print(f"  - Bias = {bias:.6f}")
    print()

# Plot the results
plt.figure(figsize=(10, 6))
ns = list(results.keys())
biases = [results[n]['bias'] for n in ns]

plt.plot(ns, biases, 'o-', color='purple', linewidth=2, markersize=8)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Bias')
plt.title('Bias of Sample Mean for Different Sample Sizes', fontsize=14)
plt.xlabel('Sample Size (n)', fontsize=12)
plt.ylabel('Bias', fontsize=12)
plt.xscale('log')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "bias_vs_sample_size.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 5: Calculate variance for sample mean
print_step_header(5, "Calculating the Variance of the Sample Mean")

print("For the sample mean μ̂ = (1/n)∑ᵢXᵢ, the variance is given by:")
print("Var(μ̂) = Var((1/n)∑ᵢXᵢ)")
print()
print("Step 1: Apply variance properties")
print("Var((1/n)∑ᵢXᵢ) = (1/n)² · Var(∑ᵢXᵢ)")
print()
print("Step 2: Since X₁, X₂, ..., Xₙ are independent,")
print("Var(∑ᵢXᵢ) = ∑ᵢVar(Xᵢ) = n·σ²  (since Var(Xᵢ) = σ² for all i)")
print()
print("Step 3: Substitute back")
print("Var(μ̂) = (1/n)² · (n·σ²)")
print("       = σ²/n")
print()
print("Therefore, the variance of the sample mean is σ²/n.")
print()
print("This result shows that as the sample size n increases, the variance of the sample mean decreases,")
print("making it a more precise estimator for larger samples.")
print()

# Step 6: Verify variance with simulations
print_step_header(6, "Verifying Variance with Simulations")

# Simulate for different sample sizes
np.random.seed(42)
sample_sizes = [5, 10, 50, 100, 500, 1000]
num_simulations = 10000
true_mean = 5
sigma = 2

results = {}
for n in sample_sizes:
    sample_means = []
    for _ in range(num_simulations):
        sample = np.random.normal(true_mean, sigma, n)
        sample_means.append(np.mean(sample))
    
    variance_of_means = np.var(sample_means)
    theoretical_variance = sigma**2 / n
    ratio = variance_of_means / theoretical_variance
    results[n] = {
        'empirical_variance': variance_of_means, 
        'theoretical_variance': theoretical_variance,
        'ratio': ratio
    }
    
    print(f"Sample size n = {n}:")
    print(f"  - Var(μ̂) (empirical) = {variance_of_means:.6f}")
    print(f"  - Var(μ̂) (theoretical) = σ²/n = {theoretical_variance:.6f}")
    print(f"  - Ratio = {ratio:.4f}")
    print()

# Plot the results
plt.figure(figsize=(12, 10))

# Plot 1: Variance vs Sample Size
plt.subplot(2, 1, 1)
ns = list(results.keys())
empirical_vars = [results[n]['empirical_variance'] for n in ns]
theoretical_vars = [results[n]['theoretical_variance'] for n in ns]

plt.plot(ns, empirical_vars, 'o-', color='blue', linewidth=2, markersize=8, label='Empirical Variance')
plt.plot(ns, theoretical_vars, 's--', color='red', linewidth=2, markersize=8, label='Theoretical Variance (σ²/n)')
plt.title('Variance of Sample Mean for Different Sample Sizes', fontsize=14)
plt.xlabel('Sample Size (n)', fontsize=12)
plt.ylabel('Variance', fontsize=12)
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.legend()

# Plot 2: Ratio of Empirical to Theoretical Variance
plt.subplot(2, 1, 2)
ratios = [results[n]['ratio'] for n in ns]

plt.plot(ns, ratios, 'o-', color='green', linewidth=2, markersize=8)
plt.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Perfect Match (Ratio = 1)')
plt.title('Ratio of Empirical to Theoretical Variance', fontsize=14)
plt.xlabel('Sample Size (n)', fontsize=12)
plt.ylabel('Ratio (Empirical / Theoretical)', fontsize=12)
plt.xscale('log')
plt.grid(True)
plt.legend()

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "variance_vs_sample_size.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 7: Visualize sampling distribution for different sample sizes
print_step_header(7, "Visualizing Sampling Distributions")

# Simulate for selected sample sizes
np.random.seed(42)
selected_sample_sizes = [5, 50, 500]
num_simulations = 10000
true_mean = 5
sigma = 2

plt.figure(figsize=(12, 5))

for i, n in enumerate(selected_sample_sizes):
    sample_means = []
    for _ in range(num_simulations):
        sample = np.random.normal(true_mean, sigma, n)
        sample_means.append(np.mean(sample))
    
    mean_of_means = np.mean(sample_means)
    variance_of_means = np.var(sample_means)
    
    plt.subplot(1, 3, i+1)
    plt.hist(sample_means, bins=50, alpha=0.7, density=True, color='skyblue')
    
    # Plot the theoretical normal curve
    x = np.linspace(true_mean - 4*sigma/np.sqrt(n), true_mean + 4*sigma/np.sqrt(n), 1000)
    theoretical_pdf = norm.pdf(x, true_mean, sigma/np.sqrt(n))
    plt.plot(x, theoretical_pdf, 'r-', linewidth=2, label='Theoretical N(μ, σ²/n)')
    
    plt.axvline(true_mean, color='green', linestyle='--', linewidth=2, label='True μ = ' + str(true_mean))
    plt.title(f'Sampling Distribution (n={n})', fontsize=12)
    plt.xlabel('Sample Mean (μ̂)', fontsize=10)
    plt.ylabel('Density', fontsize=10)
    plt.legend(fontsize=8)
    plt.grid(True)

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "sampling_distributions.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 8: Conclusion
print_step_header(8, "Conclusion and Answer")

print("For a random sample X₁, X₂, ..., Xₙ from a distribution with unknown mean μ and known variance σ²,")
print("the point estimator μ̂ = (1/n)∑ᵢXᵢ has the following properties:")
print()
print("1. Bias: Bias(μ̂) = E[μ̂] - μ = μ - μ = 0")
print("   → The sample mean is an unbiased estimator of μ.")
print()
print("2. Variance: Var(μ̂) = σ²/n")
print("   → The variance decreases as the sample size n increases.")
print()
print("These properties make the sample mean an important estimator in statistics:")
print("- It is unbiased, so it doesn't systematically over- or under-estimate the true parameter")
print("- Its variance decreases at a rate of 1/n, so it becomes more precise with larger samples")
print("- By the Central Limit Theorem, the sampling distribution of μ̂ approaches a normal distribution")
print("  as n increases, regardless of the original distribution of the data")
print()
print("Therefore, the sample mean μ̂ is not only unbiased but also a consistent estimator of μ.") 