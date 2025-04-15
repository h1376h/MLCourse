import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

print("\n=== ADVANCED EXPONENTIAL DISTRIBUTION EXAMPLES ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Example 1: Memoryless Property
print("Example 1: Memoryless Property")
lambda_exp = 1
n_samples = 10000

plt.figure(figsize=(15, 5))

# Original distribution
plt.subplot(1, 3, 1)
samples = np.random.exponential(1/lambda_exp, n_samples)
plt.hist(samples, bins=50, density=True, alpha=0.7)
x = np.linspace(0, 5, 1000)
pdf = stats.expon.pdf(x, scale=1/lambda_exp)
plt.plot(x, pdf, 'r-', linewidth=2)
plt.title('Original Exponential Distribution')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

# Conditional distribution
plt.subplot(1, 3, 2)
s = 1  # conditioning value
conditional_samples = samples[samples > s] - s
plt.hist(conditional_samples, bins=50, density=True, alpha=0.7)
plt.plot(x, pdf, 'r-', linewidth=2)
plt.title(f'Conditional Distribution (X > {s})')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

# Comparison with non-memoryless distribution
plt.subplot(1, 3, 3)
weibull_samples = np.random.weibull(2, n_samples)
weibull_conditional = weibull_samples[weibull_samples > s] - s
plt.hist(weibull_conditional, bins=50, density=True, alpha=0.7, label='Weibull')
plt.hist(conditional_samples, bins=50, density=True, alpha=0.7, label='Exponential')
plt.title('Comparison with Weibull (non-memoryless)')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'exponential_memoryless.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Poisson Process Relationship
print("\nExample 2: Poisson Process Relationship")
lambda_poisson = 2
t = 5  # time interval
n_simulations = 1000

plt.figure(figsize=(15, 5))

# Inter-arrival times
plt.subplot(1, 3, 1)
inter_arrival_times = np.random.exponential(1/lambda_poisson, n_samples)
plt.hist(inter_arrival_times, bins=50, density=True, alpha=0.7)
x = np.linspace(0, 5, 1000)
pdf = stats.expon.pdf(x, scale=1/lambda_poisson)
plt.plot(x, pdf, 'r-', linewidth=2)
plt.title('Inter-arrival Times')
plt.xlabel('Time between events')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

# Number of events in time t
plt.subplot(1, 3, 2)
event_counts = np.random.poisson(lambda_poisson * t, n_simulations)
plt.hist(event_counts, bins=range(max(event_counts)+1), density=True, alpha=0.7)
x = range(max(event_counts)+1)
pmf = stats.poisson.pmf(x, lambda_poisson * t)
plt.plot(x, pmf, 'ro-', linewidth=2)
plt.title(f'Number of Events in Time {t}')
plt.xlabel('Number of events')
plt.ylabel('Probability')
plt.grid(True, alpha=0.3)

# Joint distribution visualization
plt.subplot(1, 3, 3)
n_events = 3
joint_samples = np.random.exponential(1/lambda_poisson, (n_simulations, n_events))
plt.hist(joint_samples.flatten(), bins=50, density=True, alpha=0.7)
plt.title(f'Joint Distribution of First {n_events} Inter-arrival Times')
plt.xlabel('Time')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'exponential_poisson.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Minimum of Exponentials
print("\nExample 3: Minimum of Exponentials")
n_distributions = 3
lambda_values = [1, 2, 3]
n_samples = 10000

plt.figure(figsize=(15, 5))

# Individual distributions
plt.subplot(1, 3, 1)
for i, lam in enumerate(lambda_values):
    samples = np.random.exponential(1/lam, n_samples)
    plt.hist(samples, bins=50, density=True, alpha=0.7, label=f'λ={lam}')
    x = np.linspace(0, 5, 1000)
    pdf = stats.expon.pdf(x, scale=1/lam)
    plt.plot(x, pdf, linewidth=2)
plt.title('Individual Exponential Distributions')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

# Minimum distribution
plt.subplot(1, 3, 2)
samples = np.array([np.random.exponential(1/lam, n_samples) for lam in lambda_values])
min_samples = np.min(samples, axis=0)
plt.hist(min_samples, bins=50, density=True, alpha=0.7)
total_lambda = sum(lambda_values)
x = np.linspace(0, 5, 1000)
pdf = stats.expon.pdf(x, scale=1/total_lambda)
plt.plot(x, pdf, 'r-', linewidth=2)
plt.title(f'Minimum Distribution (λ={total_lambda})')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

# Comparison with theoretical
plt.subplot(1, 3, 3)
plt.hist(min_samples, bins=50, density=True, alpha=0.7, label='Empirical')
plt.plot(x, pdf, 'r-', linewidth=2, label='Theoretical')
plt.title('Comparison with Theoretical Distribution')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'exponential_minimum.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 4: Hazard Function Analysis
print("\nExample 4: Hazard Function Analysis")
lambda_exp = 1
t = np.linspace(0, 5, 1000)

plt.figure(figsize=(15, 5))

# Hazard function
plt.subplot(1, 3, 1)
hazard = np.ones_like(t) * lambda_exp
plt.plot(t, hazard, 'b-', linewidth=2)
plt.title('Hazard Function')
plt.xlabel('t')
plt.ylabel('h(t)')
plt.grid(True, alpha=0.3)

# Comparison with other distributions
plt.subplot(1, 3, 2)
# Weibull hazard
k = 2
weibull_hazard = k * t**(k-1)
plt.plot(t, hazard, 'b-', linewidth=2, label='Exponential')
plt.plot(t, weibull_hazard, 'r-', linewidth=2, label='Weibull')
plt.title('Comparison of Hazard Functions')
plt.xlabel('t')
plt.ylabel('h(t)')
plt.legend()
plt.grid(True, alpha=0.3)

# Survival function
plt.subplot(1, 3, 3)
survival = np.exp(-lambda_exp * t)
plt.plot(t, survival, 'b-', linewidth=2)
plt.title('Survival Function')
plt.xlabel('t')
plt.ylabel('S(t)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'exponential_hazard.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 5: Transformations and Relationships
print("\nExample 5: Transformations and Relationships")
n_samples = 10000
lambda_exp = 1

plt.figure(figsize=(15, 5))

# Sum of exponentials (Gamma)
plt.subplot(1, 3, 1)
n = 3  # number of exponentials to sum
samples = np.sum(np.random.exponential(1/lambda_exp, (n_samples, n)), axis=1)
plt.hist(samples, bins=50, density=True, alpha=0.7)
x = np.linspace(0, 10, 1000)
pdf = stats.gamma.pdf(x, n, scale=1/lambda_exp)
plt.plot(x, pdf, 'r-', linewidth=2)
plt.title('Sum of Exponentials (Gamma)')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

# Power transformation (Weibull)
plt.subplot(1, 3, 2)
k = 2  # shape parameter
exp_samples = np.random.exponential(1/lambda_exp, n_samples)
weibull_samples = exp_samples**(1/k)
plt.hist(weibull_samples, bins=50, density=True, alpha=0.7)
x = np.linspace(0, 3, 1000)
pdf = stats.weibull_min.pdf(x, k, scale=1/lambda_exp**(1/k))
plt.plot(x, pdf, 'r-', linewidth=2)
plt.title('Power Transformation (Weibull)')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

# Square root transformation (Rayleigh)
plt.subplot(1, 3, 3)
rayleigh_samples = np.sqrt(2 * exp_samples)
plt.hist(rayleigh_samples, bins=50, density=True, alpha=0.7)
x = np.linspace(0, 3, 1000)
pdf = stats.rayleigh.pdf(x)
plt.plot(x, pdf, 'r-', linewidth=2)
plt.title('Square Root Transformation (Rayleigh)')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'exponential_transformations.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll advanced exponential distribution example images created successfully.") 