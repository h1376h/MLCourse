import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

print("\n=== CENTRAL LIMIT THEOREM VISUALIZATIONS ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Example 1: CLT with Different Sample Sizes
print("Example 1: CLT with Different Sample Sizes")
np.random.seed(42)
population = np.random.exponential(scale=1, size=10000)
sample_sizes = [1, 5, 10, 30, 100, 1000]

plt.figure(figsize=(15, 10))
for i, n in enumerate(sample_sizes):
    means = [np.mean(np.random.choice(population, size=n)) for _ in range(1000)]
    plt.subplot(2, 3, i+1)
    plt.hist(means, bins=30, density=True, alpha=0.6, color='blue')
    plt.title(f'Sample Size = {n}')
    plt.xlabel('Sample Mean')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'clt_sample_sizes.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: CLT with Different Distributions
print("\nExample 2: CLT with Different Distributions")
distributions = [
    ('Exponential (λ=1)', lambda: np.random.exponential(scale=1, size=1000)),
    ('Uniform (0,1)', lambda: np.random.uniform(0, 1, size=1000)),
    ('Poisson (λ=5)', lambda: np.random.poisson(lam=5, size=1000)),
    ('Binomial (n=10, p=0.5)', lambda: np.random.binomial(n=10, p=0.5, size=1000)),
    ('Gamma (k=2, θ=1)', lambda: np.random.gamma(shape=2, scale=1, size=1000)),
    ('Beta (α=2, β=2)', lambda: np.random.beta(a=2, b=2, size=1000))
]

plt.figure(figsize=(15, 10))
for i, (name, dist_func) in enumerate(distributions):
    samples = [np.mean(dist_func()) for _ in range(1000)]
    plt.subplot(2, 3, i+1)
    plt.hist(samples, bins=30, density=True, alpha=0.6, color='green')
    plt.title(f'{name}')
    plt.xlabel('Sample Mean')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'clt_distributions.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: CLT Convergence Rate
print("\nExample 3: CLT Convergence Rate")
np.random.seed(42)
population = np.random.exponential(scale=1, size=10000)
sample_sizes = np.logspace(1, 3, 20).astype(int)
convergence_metrics = []

for n in sample_sizes:
    means = [np.mean(np.random.choice(population, size=n)) for _ in range(1000)]
    # Calculate skewness and kurtosis
    skewness = stats.skew(means)
    kurtosis = stats.kurtosis(means)
    convergence_metrics.append((n, skewness, kurtosis))

n_values, skewness_values, kurtosis_values = zip(*convergence_metrics)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(n_values, skewness_values, 'b-o')
plt.xscale('log')
plt.title('Skewness vs Sample Size')
plt.xlabel('Sample Size (log scale)')
plt.ylabel('Skewness')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(n_values, kurtosis_values, 'r-o')
plt.xscale('log')
plt.title('Kurtosis vs Sample Size')
plt.xlabel('Sample Size (log scale)')
plt.ylabel('Kurtosis')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'clt_convergence_rate.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 4: CLT with Different Population Parameters
print("\nExample 4: CLT with Different Population Parameters")
np.random.seed(42)
sample_size = 30
parameters = [
    ('Exponential (λ=0.5)', lambda: np.random.exponential(scale=0.5, size=1000)),
    ('Exponential (λ=1)', lambda: np.random.exponential(scale=1, size=1000)),
    ('Exponential (λ=2)', lambda: np.random.exponential(scale=2, size=1000))
]

plt.figure(figsize=(15, 5))
for i, (name, dist_func) in enumerate(parameters):
    samples = [np.mean(dist_func()) for _ in range(1000)]
    plt.subplot(1, 3, i+1)
    plt.hist(samples, bins=30, density=True, alpha=0.6, color='purple')
    plt.title(f'{name}')
    plt.xlabel('Sample Mean')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'clt_parameters.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll CLT visualization images created successfully.") 