import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

print("\n=== NORMAL DISTRIBUTION EXAMPLES: STEP-BY-STEP SOLUTIONS ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Example 1: Basic Properties of Normal Distribution
print("Example 1: Basic Properties of Normal Distribution")
mu = 0
sigma = 1
x = np.linspace(-4, 4, 1000)
pdf = stats.norm.pdf(x, mu, sigma)
cdf = stats.norm.cdf(x, mu, sigma)

print(f"Parameters:")
print(f"  Mean (μ) = {mu}")
print(f"  Standard Deviation (σ) = {sigma}")
print(f"  Variance (σ²) = {sigma**2}")

# Calculate key properties
mean = np.mean(x * pdf * (x[1] - x[0]))
variance = np.sum((x - mean)**2 * pdf * (x[1] - x[0]))
print(f"\nCalculated Properties:")
print(f"  Mean from PDF: {mean:.4f}")
print(f"  Variance from PDF: {variance:.4f}")

# Plot PDF and CDF
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x, pdf, 'b-', linewidth=2)
plt.fill_between(x, pdf, alpha=0.2)
plt.title('Normal Distribution PDF')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(x, cdf, 'r-', linewidth=2)
plt.title('Normal Distribution CDF')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'normal_distribution_basic.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Standard Normal vs Non-Standard Normal
print("\nExample 2: Standard Normal vs Non-Standard Normal")
mu1, sigma1 = 0, 1  # Standard normal
mu2, sigma2 = 2, 1.5  # Non-standard normal
x = np.linspace(-5, 7, 1000)
pdf1 = stats.norm.pdf(x, mu1, sigma1)
pdf2 = stats.norm.pdf(x, mu2, sigma2)

print(f"Standard Normal (N(0,1)):")
print(f"  Mean = {mu1}")
print(f"  Standard Deviation = {sigma1}")

print(f"\nNon-Standard Normal (N(2,2.25)):")
print(f"  Mean = {mu2}")
print(f"  Standard Deviation = {sigma2}")
print(f"  Variance = {sigma2**2}")

plt.figure(figsize=(10, 6))
plt.plot(x, pdf1, 'b-', label=f'N({mu1},{sigma1**2})', linewidth=2)
plt.plot(x, pdf2, 'r-', label=f'N({mu2},{sigma2**2})', linewidth=2)
plt.title('Standard vs Non-Standard Normal Distributions')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'normal_distribution_comparison.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: 68-95-99.7 Rule
print("\nExample 3: 68-95-99.7 Rule")
mu = 0
sigma = 1
x = np.linspace(-4, 4, 1000)
pdf = stats.norm.pdf(x, mu, sigma)

# Calculate probabilities
p_1sigma = stats.norm.cdf(1) - stats.norm.cdf(-1)
p_2sigma = stats.norm.cdf(2) - stats.norm.cdf(-2)
p_3sigma = stats.norm.cdf(3) - stats.norm.cdf(-3)

print(f"Probabilities within standard deviations:")
print(f"  Within ±1σ: {p_1sigma:.4f} (68.27%)")
print(f"  Within ±2σ: {p_2sigma:.4f} (95.45%)")
print(f"  Within ±3σ: {p_3sigma:.4f} (99.73%)")

plt.figure(figsize=(10, 6))
plt.plot(x, pdf, 'b-', linewidth=2)
plt.fill_between(x, pdf, where=(x >= -1) & (x <= 1), color='blue', alpha=0.2, label='±1σ (68.27%)')
plt.fill_between(x, pdf, where=(x >= -2) & (x <= 2), color='green', alpha=0.2, label='±2σ (95.45%)')
plt.fill_between(x, pdf, where=(x >= -3) & (x <= 3), color='red', alpha=0.2, label='±3σ (99.73%)')
plt.title('68-95-99.7 Rule for Normal Distribution')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'normal_distribution_rule.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 4: Z-scores and Standardization
print("\nExample 4: Z-scores and Standardization")
# Original data
mu = 100
sigma = 15
x = np.array([85, 100, 115, 130])

print("Original data:")
print(f"  Mean = {mu}")
print(f"  Standard Deviation = {sigma}")
print(f"  Values = {x}")

# Calculate z-scores
z_scores = (x - mu) / sigma
print("\nZ-scores:")
for i, (value, z) in enumerate(zip(x, z_scores)):
    print(f"  x = {value} → z = {z:.2f}")

# Plot original and standardized distributions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
x_range = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
pdf = stats.norm.pdf(x_range, mu, sigma)
plt.plot(x_range, pdf, 'b-', linewidth=2)
plt.scatter(x, np.zeros_like(x), color='red', s=100)
plt.title('Original Distribution')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
z_range = np.linspace(-4, 4, 1000)
pdf_z = stats.norm.pdf(z_range, 0, 1)
plt.plot(z_range, pdf_z, 'b-', linewidth=2)
plt.scatter(z_scores, np.zeros_like(z_scores), color='red', s=100)
plt.title('Standardized Distribution')
plt.xlabel('z')
plt.ylabel('f(z)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'normal_distribution_standardization.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 5: Probability Calculations
print("\nExample 5: Probability Calculations")
mu = 100
sigma = 15
x1, x2 = 85, 115

print(f"Parameters:")
print(f"  Mean = {mu}")
print(f"  Standard Deviation = {sigma}")

# Calculate probabilities
p_less_than_x1 = stats.norm.cdf(x1, mu, sigma)
p_between_x1_x2 = stats.norm.cdf(x2, mu, sigma) - stats.norm.cdf(x1, mu, sigma)
p_greater_than_x2 = 1 - stats.norm.cdf(x2, mu, sigma)

print(f"\nProbabilities:")
print(f"  P(X < {x1}) = {p_less_than_x1:.4f}")
print(f"  P({x1} < X < {x2}) = {p_between_x1_x2:.4f}")
print(f"  P(X > {x2}) = {p_greater_than_x2:.4f}")

plt.figure(figsize=(10, 6))
x_range = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
pdf = stats.norm.pdf(x_range, mu, sigma)
plt.plot(x_range, pdf, 'b-', linewidth=2)

# Fill areas
plt.fill_between(x_range, pdf, where=(x_range <= x1), color='red', alpha=0.3, label=f'P(X < {x1})')
plt.fill_between(x_range, pdf, where=(x_range >= x1) & (x_range <= x2), color='green', alpha=0.3, label=f'P({x1} < X < {x2})')
plt.fill_between(x_range, pdf, where=(x_range >= x2), color='blue', alpha=0.3, label=f'P(X > {x2})')

plt.title('Probability Calculations for Normal Distribution')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'normal_distribution_probabilities.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 6: Quantiles and Percentiles
print("\nExample 6: Quantiles and Percentiles")
mu = 0
sigma = 1
percentiles = [0.25, 0.5, 0.75, 0.95]

print(f"Standard Normal Distribution (N(0,1)):")
for p in percentiles:
    quantile = stats.norm.ppf(p, mu, sigma)
    print(f"  {p*100}th percentile: {quantile:.4f}")

# Plot with quantiles
plt.figure(figsize=(10, 6))
x = np.linspace(-4, 4, 1000)
pdf = stats.norm.pdf(x, mu, sigma)
plt.plot(x, pdf, 'b-', linewidth=2)

# Add quantile lines
for p in percentiles:
    q = stats.norm.ppf(p, mu, sigma)
    plt.axvline(x=q, color='r', linestyle='--', alpha=0.5)
    plt.text(q, 0.1, f'{p*100}%', rotation=90, va='bottom')

plt.title('Quantiles of Standard Normal Distribution')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'normal_distribution_quantiles.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 7: Effect of Standard Deviation
print("\nExample 7: Effect of Standard Deviation")
mu = 0
sigmas = [0.5, 1.0, 2.0]
x = np.linspace(-5, 5, 1000)

plt.figure(figsize=(10, 6))
for sigma in sigmas:
    pdf = stats.norm.pdf(x, mu, sigma)
    plt.plot(x, pdf, label=f'σ = {sigma}', linewidth=2)

plt.title('Effect of Standard Deviation on Normal Distribution')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'normal_distribution_sigma_effect.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 8: Central Limit Theorem - Sample Means
print("\nExample 8: Central Limit Theorem - Sample Means")
# Generate samples from a non-normal distribution (exponential)
np.random.seed(42)
population = np.random.exponential(scale=1, size=10000)
sample_means = []
sample_sizes = [1, 5, 30, 100]

for n in sample_sizes:
    means = [np.mean(np.random.choice(population, size=n)) for _ in range(1000)]
    sample_means.append(means)

plt.figure(figsize=(12, 8))
for i, (n, means) in enumerate(zip(sample_sizes, sample_means)):
    plt.subplot(2, 2, i+1)
    plt.hist(means, bins=30, density=True, alpha=0.6, color='blue')
    plt.title(f'Sample Size = {n}')
    plt.xlabel('Sample Mean')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'normal_distribution_clt_means.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 9: Central Limit Theorem - Different Distributions
print("\nExample 9: Central Limit Theorem - Different Distributions")
distributions = [
    ('Exponential', lambda: np.random.exponential(scale=1, size=1000)),
    ('Uniform', lambda: np.random.uniform(0, 1, size=1000)),
    ('Poisson', lambda: np.random.poisson(lam=5, size=1000)),
    ('Binomial', lambda: np.random.binomial(n=10, p=0.5, size=1000))
]

plt.figure(figsize=(12, 8))
for i, (name, dist_func) in enumerate(distributions):
    samples = [np.mean(dist_func()) for _ in range(1000)]
    plt.subplot(2, 2, i+1)
    plt.hist(samples, bins=30, density=True, alpha=0.6, color='green')
    plt.title(f'{name} Distribution')
    plt.xlabel('Sample Mean')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'normal_distribution_clt_distributions.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll normal distribution example images created successfully.") 