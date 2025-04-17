import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
import matplotlib.patches as patches
from scipy.special import factorial

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_3_5")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Set up the problem
print_step_header(1, "Understanding the Question")

print("QUESTION 5: Is the sample mean a sufficient statistic for the parameter μ of a normal distribution with known variance?")
print("\nLet's understand what this question is asking:")
print("- We have a random sample X₁, X₂, ..., Xₙ from a normal distribution N(μ, σ²)")
print("- The variance σ² is known")
print("- The mean μ is unknown and we want to estimate it")
print("- We want to determine if the sample mean X̄ = (1/n) Σ Xᵢ contains all the information about μ that is in the data")

# Step 2: Explain what a sufficient statistic is
print_step_header(2, "Understanding Sufficient Statistics")

print("A statistic T(X) is sufficient for a parameter θ if the conditional distribution of the data X given T(X) does not depend on θ.")
print("\nIntuitively, this means that once we know the value of a sufficient statistic, the original data doesn't provide any additional information about the parameter.")
print("\nWe can use the factorization theorem to show that a statistic T is sufficient for θ if and only if the likelihood function can be factored as:")
print("L(θ|x) = g(T(x),θ) × h(x)")
print("where g depends on the data only through T(x), and h does not depend on θ at all.")

# Step 3: Derive the likelihood function for a normal sample
print_step_header(3, "Deriving the Likelihood Function")

print("For a random sample X₁, X₂, ..., Xₙ from a normal distribution N(μ, σ²), the likelihood function is:")
print("\nL(μ|x) = ∏ f(xᵢ|μ)")
print("where f(xᵢ|μ) is the PDF of a normal distribution:")
print("f(xᵢ|μ) = (1/√(2πσ²)) × exp(-(xᵢ-μ)²/(2σ²))")
print("\nTherefore:")
print("L(μ|x) = (1/√(2πσ²))ⁿ × exp(-(Σ(xᵢ-μ)²)/(2σ²))")
print("\nWe can expand the sum in the exponent:")
print("Σ(xᵢ-μ)² = Σ(xᵢ² - 2μxᵢ + μ²) = Σxᵢ² - 2μΣxᵢ + nμ²")
print("\nThis gives us:")
print("L(μ|x) = (1/√(2πσ²))ⁿ × exp(-(Σxᵢ² - 2μΣxᵢ + nμ²)/(2σ²))")
print("L(μ|x) = (1/√(2πσ²))ⁿ × exp(-Σxᵢ²/(2σ²)) × exp(2μΣxᵢ/(2σ²)) × exp(-nμ²/(2σ²))")
print("L(μ|x) = (1/√(2πσ²))ⁿ × exp(-Σxᵢ²/(2σ²)) × exp((μΣxᵢ - nμ²/2)/σ²)")
print("\nNote that Σxᵢ = n×x̄, so:")
print("L(μ|x) = (1/√(2πσ²))ⁿ × exp(-Σxᵢ²/(2σ²)) × exp((nμx̄ - nμ²/2)/σ²)")
print("\nWe can now factor this as:")
print("L(μ|x) = h(x) × g(x̄,μ)")
print("where:")
print("h(x) = (1/√(2πσ²))ⁿ × exp(-Σxᵢ²/(2σ²))")
print("g(x̄,μ) = exp((nμx̄ - nμ²/2)/σ²)")
print("\nBy the factorization theorem, since the likelihood depends on the data only through x̄, the sample mean x̄ is a sufficient statistic for μ.")

# Step 4: Visual Demonstration
print_step_header(4, "Visual Demonstration")

# Generate random samples from different normal distributions
np.random.seed(42)
n_samples = 10
mu_true = 5
sigma = 2

# Generate 3 different samples from the same distribution
samples_1 = np.random.normal(mu_true, sigma, n_samples)
samples_2 = np.random.normal(mu_true, sigma, n_samples)
samples_3 = np.random.normal(mu_true, sigma, n_samples)

# Calculate sample means
mean_1 = np.mean(samples_1)
mean_2 = np.mean(samples_2)
mean_3 = np.mean(samples_3)

print(f"True μ = {mu_true}, σ = {sigma}")
print(f"\nSample 1: {samples_1}")
print(f"Sample Mean 1: {mean_1:.4f}")
print(f"\nSample 2: {samples_2}")
print(f"Sample Mean 2: {mean_2:.4f}")
print(f"\nSample 3: {samples_3}")
print(f"Sample Mean 3: {mean_3:.4f}")

# FIGURE 1: Plot the samples
plt.figure(figsize=(10, 6))
sample_indices = np.arange(1, n_samples + 1)

plt.plot(sample_indices, samples_1, 'o-', label=f'Sample 1 (Mean = {mean_1:.4f})')
plt.plot(sample_indices, samples_2, 's-', label=f'Sample 2 (Mean = {mean_2:.4f})')
plt.plot(sample_indices, samples_3, '^-', label=f'Sample 3 (Mean = {mean_3:.4f})')
plt.axhline(y=mu_true, color='r', linestyle='--', label=f'True μ = {mu_true}')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Three Different Samples from N(5, 4)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save Figure 1
file_path_1 = os.path.join(save_dir, "1_sample_comparison.png")
plt.savefig(file_path_1, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure 1 saved to: {file_path_1}")

# Create an artificial sample with exactly the same mean as samples_1
samples_4 = np.array([mean_1] * n_samples)  # All values equal to the mean
variance_adjustment = np.linspace(-2, 2, n_samples)
samples_4 += variance_adjustment  # Add variance while preserving the mean
samples_4 -= np.mean(samples_4) - mean_1  # Correct any deviation in mean due to floating point

# Define the likelihood function (without constants)
def likelihood(mu, data, sigma):
    n = len(data)
    return np.exp(-np.sum((data - mu)**2) / (2 * sigma**2))

# Mu range for likelihood plots
mu_range = np.linspace(mu_true - 3, mu_true + 3, 1000)

# Compute and normalize likelihoods
l1 = [likelihood(mu, samples_1, sigma) for mu in mu_range]
l1 = l1 / np.max(l1)
l2 = [likelihood(mu, samples_2, sigma) for mu in mu_range]
l2 = l2 / np.max(l2)
l3 = [likelihood(mu, samples_3, sigma) for mu in mu_range]
l3 = l3 / np.max(l3)
l4 = [likelihood(mu, samples_4, sigma) for mu in mu_range]
l4 = l4 / np.max(l4)

# FIGURE 2: Plot likelihood functions for same mean, different data
plt.figure(figsize=(10, 6))
plt.plot(mu_range, l1, '-', label=f'Sample 1 (Mean = {mean_1:.4f})')
plt.plot(mu_range, l4, ':', label=f'Different Sample with Same Mean')
plt.axvline(x=mean_1, color='k', linestyle='--')
plt.xlabel('μ')
plt.ylabel('Normalized Likelihood')
plt.title('Likelihood Functions: Same Mean, Different Data')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save Figure 2
file_path_2 = os.path.join(save_dir, "2_same_mean_comparison.png")
plt.savefig(file_path_2, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure 2 saved to: {file_path_2}")

# FIGURE 3: Plot likelihood functions for different means
plt.figure(figsize=(10, 6))
plt.plot(mu_range, l1, '-', label=f'Sample 1 (Mean = {mean_1:.4f})')
plt.plot(mu_range, l2, '-', label=f'Sample 2 (Mean = {mean_2:.4f})')
plt.plot(mu_range, l3, '-', label=f'Sample 3 (Mean = {mean_3:.4f})')
plt.axvline(x=mu_true, color='r', linestyle='--', label=f'True μ = {mu_true}')
plt.xlabel('μ')
plt.ylabel('Normalized Likelihood')
plt.title('Likelihood Functions for Different Samples')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save Figure 3
file_path_3 = os.path.join(save_dir, "3_different_means_comparison.png")
plt.savefig(file_path_3, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure 3 saved to: {file_path_3}")

# FIGURE 4: Create a simple visual demonstration of sufficiency
# Generate a larger dataset for visualization
n_large = 1000
samples_large = np.random.normal(mu_true, sigma, n_large)
sample_mean_large = np.mean(samples_large)

# Create a figure showing the data compression aspect
plt.figure(figsize=(10, 6))

# Left side: Original data
plt.subplot(1, 2, 1)
plt.hist(samples_large, bins=30, alpha=0.7, color='skyblue')
plt.axvline(x=sample_mean_large, color='r', linestyle='--', 
            label=f'Sample Mean = {sample_mean_large:.4f}')
plt.axvline(x=mu_true, color='g', linestyle='-', 
            label=f'True μ = {mu_true}')
plt.title(f'Original Data: {n_large} Points')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()

# Right side: Sufficient statistic
plt.subplot(1, 2, 2)
plt.axvline(x=sample_mean_large, color='r', linestyle='--', linewidth=3,
            label=f'Sample Mean = {sample_mean_large:.4f}')
plt.axvline(x=mu_true, color='g', linestyle='-', 
            label=f'True μ = {mu_true}')
plt.title('Sufficient Statistic: Just 1 Point')
plt.xlabel('Value')
plt.yticks([])
plt.xlim(plt.xlim())  # Match x-axis limits with the left subplot
plt.legend()

plt.suptitle('Data Compression with Sufficient Statistics', fontsize=14)
plt.tight_layout()
plt.subplots_adjust(top=0.85)

# Save Figure 4
file_path_4 = os.path.join(save_dir, "4_data_compression.png")
plt.savefig(file_path_4, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure 4 saved to: {file_path_4}")

# Step 5: Formal Proof using Fisher-Neyman Factorization Theorem
print_step_header(5, "Formal Proof using Fisher-Neyman Factorization Theorem")

print("The Fisher-Neyman factorization theorem states that T(X) is a sufficient statistic for θ if and only if the likelihood function can be written as:")
print("L(θ|x) = g(T(x),θ) × h(x)")
print("\nFor a normal distribution with known variance σ², we've shown that:")
print("L(μ|x) = (1/√(2πσ²))ⁿ × exp(-Σxᵢ²/(2σ²)) × exp((nμx̄ - nμ²/2)/σ²)")
print("\nWhich we can factor as:")
print("L(μ|x) = h(x) × g(x̄,μ)")
print("where:")
print("h(x) = (1/√(2πσ²))ⁿ × exp(-Σxᵢ²/(2σ²)) - depends only on the data, not on μ")
print("g(x̄,μ) = exp((nμx̄ - nμ²/2)/σ²) - depends on the data only through x̄")
print("\nSince we can express the likelihood in this factorized form, where the parameter μ only interacts with the data through the sample mean x̄, by the factorization theorem, the sample mean is a sufficient statistic for μ.")

# Step 6: Visual explanation of efficiency
print_step_header(6, "Comparing with Other Statistics")

print("Let's compare the sample mean with other statistics to see why the sample mean is sufficient:")
print("\n1. Sample Median: While the median is robust to outliers, it doesn't use all the information about μ")
print("2. Sample Range: The range only considers the minimum and maximum values, losing information about the distribution")
print("3. Sample Variance: The variance measures spread around the mean, but doesn't directly tell us about the center")
print("\nThe sample mean is not only sufficient but also the most efficient unbiased estimator for μ in a normal distribution.")

# Step 7: Conclusion
print_step_header(7, "Conclusion")

print("ANSWER: YES, the sample mean is a sufficient statistic for μ in a normal distribution with known variance.")
print("\nReason: By the factorization theorem, we've shown that the likelihood function depends on the data only through the sample mean. Therefore, once we know the sample mean, the original data provides no additional information about μ.")
print("\nImportance: This result tells us that when we want to estimate the mean of a normal distribution with known variance, we can compress all the information from our n data points into a single value—the sample mean—without losing any information about μ. This is a powerful result for statistical efficiency and data compression.") 