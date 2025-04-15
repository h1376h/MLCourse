import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

print("\n=== UNIFORM DISTRIBUTION EXAMPLES: STEP-BY-STEP SOLUTIONS ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Example 1: Basic Properties of Uniform Distribution
print("Example 1: Basic Properties of Uniform Distribution")
a, b = 0, 1  # Standard uniform distribution
x = np.linspace(a - 0.5, b + 0.5, 1000)
pdf = stats.uniform.pdf(x, a, b - a)
cdf = stats.uniform.cdf(x, a, b - a)

print(f"Parameters:")
print(f"  Lower bound (a) = {a}")
print(f"  Upper bound (b) = {b}")

# Calculate key properties
mean = (a + b) / 2
variance = (b - a)**2 / 12

print(f"\nCalculated Properties:")
print(f"  Mean: {mean:.4f}")
print(f"  Variance: {variance:.4f}")

# Plot PDF and CDF
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x, pdf, 'b-', linewidth=2)
plt.fill_between(x, pdf, alpha=0.2)
plt.title('Uniform Distribution PDF')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(x, cdf, 'r-', linewidth=2)
plt.title('Uniform Distribution CDF')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'uniform_distribution_basic.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Different Intervals
print("\nExample 2: Different Intervals")
intervals = [
    (0, 1, "Standard"),
    (1, 3, "Shifted"),
    (-1, 1, "Centered"),
    (0, 2, "Wider")
]

plt.figure(figsize=(10, 6))
for a, b, label in intervals:
    x = np.linspace(a - 0.5, b + 0.5, 1000)
    pdf = stats.uniform.pdf(x, a, b - a)
    plt.plot(x, pdf, linewidth=2, label=f'Uniform({a},{b}) - {label}')

plt.title('Uniform Distributions with Different Intervals')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'uniform_distribution_intervals.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Sampling Properties
print("\nExample 3: Sampling Properties")
a, b = 0, 1
sample_sizes = [10, 100, 1000]

plt.figure(figsize=(15, 5))
for i, n in enumerate(sample_sizes, 1):
    samples = np.random.uniform(a, b, n)
    plt.subplot(1, 3, i)
    plt.hist(samples, bins=30, density=True, alpha=0.7)
    x = np.linspace(a - 0.5, b + 0.5, 1000)
    pdf = stats.uniform.pdf(x, a, b - a)
    plt.plot(x, pdf, 'r-', linewidth=2)
    plt.title(f'n = {n} samples')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'uniform_distribution_sampling.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 4: Transformation Examples
print("\nExample 4: Transformation Examples")
a, b = 0, 1
n_samples = 10000
uniform_samples = np.random.uniform(a, b, n_samples)

plt.figure(figsize=(15, 5))

# Transform to exponential
plt.subplot(1, 3, 1)
lambda_exp = 1
exp_samples = -np.log(1 - uniform_samples) / lambda_exp
plt.hist(exp_samples, bins=50, density=True, alpha=0.7)
x = np.linspace(0, 5, 1000)
pdf = stats.expon.pdf(x, scale=1/lambda_exp)
plt.plot(x, pdf, 'r-', linewidth=2)
plt.title('Uniform → Exponential')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

# Transform to normal
plt.subplot(1, 3, 2)
normal_samples = stats.norm.ppf(uniform_samples)
plt.hist(normal_samples, bins=50, density=True, alpha=0.7)
x = np.linspace(-4, 4, 1000)
pdf = stats.norm.pdf(x)
plt.plot(x, pdf, 'r-', linewidth=2)
plt.title('Uniform → Normal')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

# Transform to beta
plt.subplot(1, 3, 3)
alpha_beta, beta_beta = 2, 5
beta_samples = stats.beta.ppf(uniform_samples, alpha_beta, beta_beta)
plt.hist(beta_samples, bins=50, density=True, alpha=0.7)
x = np.linspace(0, 1, 1000)
pdf = stats.beta.pdf(x, alpha_beta, beta_beta)
plt.plot(x, pdf, 'r-', linewidth=2)
plt.title('Uniform → Beta')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'uniform_distribution_transformations.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll uniform distribution example images created successfully.") 