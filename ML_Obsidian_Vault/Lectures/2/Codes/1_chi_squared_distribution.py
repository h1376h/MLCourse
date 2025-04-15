import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

print("\n=== CHI-SQUARED DISTRIBUTION EXAMPLES: STEP-BY-STEP SOLUTIONS ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Example 1: Basic Properties of Chi-Squared Distribution
print("Example 1: Basic Properties of Chi-Squared Distribution")
k = 3  # Degrees of freedom
x = np.linspace(0, 20, 1000)
pdf = stats.chi2.pdf(x, k)
cdf = stats.chi2.cdf(x, k)

print(f"Parameters:")
print(f"  Degrees of freedom (k) = {k}")

# Calculate key properties
mean = k
variance = 2 * k
mode = max(0, k - 2)

print(f"\nCalculated Properties:")
print(f"  Mean: {mean:.4f}")
print(f"  Variance: {variance:.4f}")
print(f"  Mode: {mode:.4f}")

# Plot PDF and CDF
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x, pdf, 'b-', linewidth=2)
plt.fill_between(x, pdf, alpha=0.2)
plt.title('Chi-Squared Distribution PDF')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(x, cdf, 'r-', linewidth=2)
plt.title('Chi-Squared Distribution CDF')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'chi_squared_distribution_basic.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Different Degrees of Freedom
print("\nExample 2: Different Degrees of Freedom")
degrees_of_freedom = [1, 2, 3, 5, 10]

plt.figure(figsize=(10, 6))
for k in degrees_of_freedom:
    pdf = stats.chi2.pdf(x, k)
    plt.plot(x, pdf, linewidth=2, label=f'k = {k}')

plt.title('Chi-Squared Distributions with Different Degrees of Freedom')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'chi_squared_distribution_dof.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Relationship to Normal Distribution
print("\nExample 3: Relationship to Normal Distribution")
k = 5
n_samples = 10000

# Generate standard normal samples
normal_samples = np.random.normal(0, 1, (n_samples, k))
# Square and sum to get chi-squared samples
chi_squared_samples = np.sum(normal_samples**2, axis=1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(normal_samples[:, 0], bins=50, density=True, alpha=0.7)
x_normal = np.linspace(-4, 4, 1000)
plt.plot(x_normal, stats.norm.pdf(x_normal), 'r-', linewidth=2)
plt.title('Standard Normal Distribution')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(chi_squared_samples, bins=50, density=True, alpha=0.7)
x_chi2 = np.linspace(0, 20, 1000)
plt.plot(x_chi2, stats.chi2.pdf(x_chi2, k), 'r-', linewidth=2)
plt.title(f'Sum of {k} Squared Normal Variables')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'chi_squared_normal_relationship.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 4: Critical Values
print("\nExample 4: Critical Values")
k_values = [1, 2, 3, 5, 10]
percentiles = [0.90, 0.95, 0.99]

plt.figure(figsize=(10, 6))
for k in k_values:
    critical_values = [stats.chi2.ppf(p, k) for p in percentiles]
    plt.plot([k] * len(percentiles), critical_values, 'o-', label=f'k = {k}')

plt.title('Critical Values of Chi-Squared Distribution')
plt.xlabel('Degrees of Freedom (k)')
plt.ylabel('Critical Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'chi_squared_critical_values.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll chi-squared distribution example images created successfully.") 