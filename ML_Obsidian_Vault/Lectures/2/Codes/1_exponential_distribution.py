import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

print("\n=== EXPONENTIAL DISTRIBUTION EXAMPLES: STEP-BY-STEP SOLUTIONS ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Example 1: Basic Properties of Exponential Distribution
print("Example 1: Basic Properties of Exponential Distribution")
lambda_param = 1  # Rate parameter
x = np.linspace(0, 5, 1000)
pdf = stats.expon.pdf(x, scale=1/lambda_param)
cdf = stats.expon.cdf(x, scale=1/lambda_param)
survival = 1 - cdf
hazard = pdf / survival

print(f"Parameters:")
print(f"  Rate (λ) = {lambda_param}")

# Calculate key properties
mean = 1 / lambda_param
variance = 1 / lambda_param**2

print(f"\nCalculated Properties:")
print(f"  Mean: {mean:.4f}")
print(f"  Variance: {variance:.4f}")

# Plot PDF, CDF, Survival, and Hazard
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(x, pdf, 'b-', linewidth=2)
plt.fill_between(x, pdf, alpha=0.2)
plt.title('Exponential Distribution PDF')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(x, cdf, 'r-', linewidth=2)
plt.title('Exponential Distribution CDF')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.plot(x, survival, 'g-', linewidth=2)
plt.title('Survival Function')
plt.xlabel('x')
plt.ylabel('S(x)')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.plot(x, hazard, 'm-', linewidth=2)
plt.title('Hazard Function')
plt.xlabel('x')
plt.ylabel('h(x)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'exponential_distribution_basic.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Different Rate Parameters
print("\nExample 2: Different Rate Parameters")
rates = [0.5, 1, 2, 3]

plt.figure(figsize=(10, 6))
for rate in rates:
    pdf = stats.expon.pdf(x, scale=1/rate)
    plt.plot(x, pdf, linewidth=2, label=f'λ = {rate}')

plt.title('Exponential Distributions with Different Rates')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'exponential_distribution_rates.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Memoryless Property
print("\nExample 3: Memoryless Property")
lambda_param = 1
s = 1  # Elapsed time
t = 2  # Additional time

# Calculate probabilities
P_X_gt_s_plus_t = stats.expon.sf(s + t, scale=1/lambda_param)
P_X_gt_s = stats.expon.sf(s, scale=1/lambda_param)
P_X_gt_t = stats.expon.sf(t, scale=1/lambda_param)
conditional_prob = P_X_gt_s_plus_t / P_X_gt_s

print(f"Memoryless Property Verification:")
print(f"  P(X > {s} + {t} | X > {s}) = {conditional_prob:.4f}")
print(f"  P(X > {t}) = {P_X_gt_t:.4f}")

# Visualize memoryless property
plt.figure(figsize=(10, 6))
x = np.linspace(0, 5, 1000)
pdf = stats.expon.pdf(x, scale=1/lambda_param)
plt.plot(x, pdf, 'b-', linewidth=2, label='Original PDF')

# Highlight the conditional probability
x_cond = np.linspace(s, s + t, 100)
pdf_cond = stats.expon.pdf(x_cond, scale=1/lambda_param) / P_X_gt_s
plt.plot(x_cond, pdf_cond, 'r-', linewidth=2, label='Conditional PDF')

plt.axvline(x=s, color='k', linestyle='--', label=f'Elapsed time = {s}')
plt.title('Memoryless Property of Exponential Distribution')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'exponential_distribution_memoryless.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 4: Survival Analysis
print("\nExample 4: Survival Analysis")
lambda_param = 1
x = np.linspace(0, 5, 1000)
pdf = stats.expon.pdf(x, scale=1/lambda_param)
survival = stats.expon.sf(x, scale=1/lambda_param)
hazard = pdf / survival

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(x, pdf, 'b-', linewidth=2)
plt.title('Probability Density Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(x, survival, 'g-', linewidth=2)
plt.title('Survival Function')
plt.xlabel('x')
plt.ylabel('S(x)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(x, hazard, 'm-', linewidth=2)
plt.title('Hazard Function')
plt.xlabel('x')
plt.ylabel('h(x)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'exponential_distribution_survival.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll exponential distribution example images created successfully.") 