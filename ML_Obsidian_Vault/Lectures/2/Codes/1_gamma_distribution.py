import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

print("\n=== GAMMA DISTRIBUTION EXAMPLES: STEP-BY-STEP SOLUTIONS ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Example 1: Basic Properties of Gamma Distribution
print("Example 1: Basic Properties of Gamma Distribution")
alpha = 2  # Shape parameter
lambda_param = 1  # Rate parameter
x = np.linspace(0, 10, 1000)
pdf = stats.gamma.pdf(x, alpha, scale=1/lambda_param)
cdf = stats.gamma.cdf(x, alpha, scale=1/lambda_param)

print(f"Parameters:")
print(f"  Shape (α) = {alpha}")
print(f"  Rate (λ) = {lambda_param}")

# Calculate key properties
mean = alpha / lambda_param
variance = alpha / lambda_param**2
mode = (alpha - 1) / lambda_param if alpha > 1 else 0

print(f"\nCalculated Properties:")
print(f"  Mean: {mean:.4f}")
print(f"  Variance: {variance:.4f}")
print(f"  Mode: {mode:.4f}")

# Plot PDF and CDF
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x, pdf, 'b-', linewidth=2)
plt.fill_between(x, pdf, alpha=0.2)
plt.title('Gamma Distribution PDF')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(x, cdf, 'r-', linewidth=2)
plt.title('Gamma Distribution CDF')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'gamma_distribution_basic.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Shape Parameter Effect
print("\nExample 2: Shape Parameter Effect")
lambda_param = 1
shapes = [0.5, 1, 2, 5, 10]

plt.figure(figsize=(10, 6))
for alpha in shapes:
    pdf = stats.gamma.pdf(x, alpha, scale=1/lambda_param)
    plt.plot(x, pdf, linewidth=2, label=f'α = {alpha}')

plt.title('Gamma Distributions with Different Shape Parameters')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'gamma_distribution_shape.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Rate Parameter Effect
print("\nExample 3: Rate Parameter Effect")
alpha = 2
rates = [0.5, 1, 2, 3]

plt.figure(figsize=(10, 6))
for lambda_param in rates:
    pdf = stats.gamma.pdf(x, alpha, scale=1/lambda_param)
    plt.plot(x, pdf, linewidth=2, label=f'λ = {lambda_param}')

plt.title('Gamma Distributions with Different Rate Parameters')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'gamma_distribution_rate.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 4: Special Cases
print("\nExample 4: Special Cases")
x = np.linspace(0, 10, 1000)

plt.figure(figsize=(15, 5))

# Exponential distribution (α = 1)
plt.subplot(1, 3, 1)
lambda_exp = 1
pdf_exp = stats.gamma.pdf(x, 1, scale=1/lambda_exp)
plt.plot(x, pdf_exp, 'b-', linewidth=2)
plt.title('Exponential (α = 1)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True, alpha=0.3)

# Chi-squared distribution (α = k/2, λ = 1/2)
plt.subplot(1, 3, 2)
k = 4  # degrees of freedom
pdf_chi2 = stats.gamma.pdf(x, k/2, scale=2)
plt.plot(x, pdf_chi2, 'g-', linewidth=2)
plt.title(f'Chi-squared (k = {k})')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True, alpha=0.3)

# Additive property
plt.subplot(1, 3, 3)
alpha1, lambda1 = 2, 1
alpha2, lambda2 = 3, 1
pdf1 = stats.gamma.pdf(x, alpha1, scale=1/lambda1)
pdf2 = stats.gamma.pdf(x, alpha2, scale=1/lambda2)
pdf_sum = stats.gamma.pdf(x, alpha1 + alpha2, scale=1/lambda1)
plt.plot(x, pdf1, 'b--', linewidth=2, label='Gamma(2,1)')
plt.plot(x, pdf2, 'g--', linewidth=2, label='Gamma(3,1)')
plt.plot(x, pdf_sum, 'r-', linewidth=2, label='Gamma(5,1)')
plt.title('Additive Property')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'gamma_distribution_special.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll gamma distribution example images created successfully.") 