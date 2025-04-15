import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

print("\n=== ADVANCED UNIFORM DISTRIBUTION EXAMPLES ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Example 1: Transformation Examples
print("Example 1: Transformation Examples")
n_samples = 10000
uniform_samples = np.random.uniform(0, 1, n_samples)

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
plt.savefig(os.path.join(images_dir, 'uniform_transformations.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Monte Carlo Integration
print("\nExample 2: Monte Carlo Integration")
def f(x):
    return np.sin(x) * np.exp(-x)

a, b = 0, np.pi
x = np.linspace(a, b, 1000)
true_integral = np.trapz(f(x), x)

sample_sizes = [10, 100, 1000, 10000]
mc_estimates = []
mc_errors = []

plt.figure(figsize=(15, 5))
for i, n in enumerate(sample_sizes, 1):
    uniform_samples = np.random.uniform(a, b, n)
    mc_estimate = (b - a) * np.mean(f(uniform_samples))
    mc_estimates.append(mc_estimate)
    mc_errors.append(np.abs(mc_estimate - true_integral))
    
    plt.subplot(1, 4, i)
    plt.plot(x, f(x), 'b-', linewidth=2)
    plt.scatter(uniform_samples, f(uniform_samples), color='red', alpha=0.5)
    plt.title(f'n = {n}\nEstimate = {mc_estimate:.4f}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'uniform_monte_carlo.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Random Number Generation
print("\nExample 3: Random Number Generation")
n_samples = 10000

plt.figure(figsize=(15, 5))

# Linear Congruential Generator
plt.subplot(1, 3, 1)
a, c, m = 1664525, 1013904223, 2**32
x = 1
lcg_samples = []
for _ in range(n_samples):
    x = (a * x + c) % m
    lcg_samples.append(x / m)
plt.hist(lcg_samples, bins=50, density=True, alpha=0.7)
plt.title('Linear Congruential Generator')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

# Mersenne Twister
plt.subplot(1, 3, 2)
mt_samples = np.random.uniform(0, 1, n_samples)
plt.hist(mt_samples, bins=50, density=True, alpha=0.7)
plt.title('Mersenne Twister')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

# Box-Muller Transform
plt.subplot(1, 3, 3)
u1 = np.random.uniform(0, 1, n_samples)
u2 = np.random.uniform(0, 1, n_samples)
z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
plt.hist(z1, bins=50, density=True, alpha=0.7)
x = np.linspace(-4, 4, 1000)
pdf = stats.norm.pdf(x)
plt.plot(x, pdf, 'r-', linewidth=2)
plt.title('Box-Muller Transform')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'uniform_rng.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 4: Order Statistics
print("\nExample 4: Order Statistics")
n = 10  # Sample size
n_samples = 10000
samples = np.random.uniform(0, 1, (n_samples, n))
sorted_samples = np.sort(samples, axis=1)

plt.figure(figsize=(15, 5))

# Minimum
plt.subplot(1, 3, 1)
min_samples = sorted_samples[:, 0]
plt.hist(min_samples, bins=50, density=True, alpha=0.7)
x = np.linspace(0, 1, 1000)
pdf = stats.beta.pdf(x, 1, n)
plt.plot(x, pdf, 'r-', linewidth=2)
plt.title('Minimum (Beta(1,n))')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

# Maximum
plt.subplot(1, 3, 2)
max_samples = sorted_samples[:, -1]
plt.hist(max_samples, bins=50, density=True, alpha=0.7)
pdf = stats.beta.pdf(x, n, 1)
plt.plot(x, pdf, 'r-', linewidth=2)
plt.title('Maximum (Beta(n,1))')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

# Median
plt.subplot(1, 3, 3)
median_samples = sorted_samples[:, n//2]
plt.hist(median_samples, bins=50, density=True, alpha=0.7)
pdf = stats.beta.pdf(x, n//2 + 1, n//2 + 1)
plt.plot(x, pdf, 'r-', linewidth=2)
plt.title('Median (Beta(k,n-k+1))')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'uniform_order_stats.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 5: Multidimensional Uniform
print("\nExample 5: Multidimensional Uniform")
n_samples = 1000

plt.figure(figsize=(15, 5))

# 2D Uniform
plt.subplot(1, 3, 1)
x2d = np.random.uniform(0, 1, n_samples)
y2d = np.random.uniform(0, 1, n_samples)
plt.scatter(x2d, y2d, alpha=0.5)
plt.title('2D Uniform Distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True, alpha=0.3)

# 3D Uniform
plt.subplot(1, 3, 2)
x3d = np.random.uniform(0, 1, n_samples)
y3d = np.random.uniform(0, 1, n_samples)
z3d = np.random.uniform(0, 1, n_samples)
ax = plt.axes(projection='3d')
ax.scatter(x3d, y3d, z3d, alpha=0.5)
plt.title('3D Uniform Distribution')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Volume Estimation
plt.subplot(1, 3, 3)
n_inside = 0
n_total = 10000
for _ in range(n_total):
    x = np.random.uniform(-1, 1)
    y = np.random.uniform(-1, 1)
    z = np.random.uniform(-1, 1)
    if x**2 + y**2 + z**2 <= 1:
        n_inside += 1
volume_estimate = 8 * n_inside / n_total
true_volume = 4/3 * np.pi
plt.bar(['Estimated', 'True'], [volume_estimate, true_volume])
plt.title(f'Volume Estimation\nEstimate: {volume_estimate:.4f}\nTrue: {true_volume:.4f}')
plt.ylabel('Volume')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'uniform_multidimensional.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll advanced uniform distribution example images created successfully.") 