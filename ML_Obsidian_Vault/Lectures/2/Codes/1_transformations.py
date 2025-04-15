import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from scipy.special import gamma  # Use scipy's gamma function

print("\n=== PROBABILITY DISTRIBUTION TRANSFORMATIONS ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Example 1: Uniform to Exponential Transformation
print("Example 1: Uniform to Exponential Transformation")
n_samples = 10000

plt.figure(figsize=(15, 5))

# Original Uniform
plt.subplot(1, 3, 1)
uniform_samples = np.random.uniform(0, 1, n_samples)
plt.hist(uniform_samples, bins=50, density=True, alpha=0.7)
plt.plot([0, 0, 1, 1], [0, 1, 1, 0], 'r-', linewidth=2)
plt.title('Original: Uniform(0,1)')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

# Transformation Process
plt.subplot(1, 3, 2)
lambda_exp = 1.5
x = np.linspace(0.001, 0.999, 1000)  # Avoid 0 and 1
y = -np.log(1-x) / lambda_exp
plt.plot(x, y, 'b-', linewidth=2)
plt.title('Transformation Function\nX = -ln(1-U)/λ')
plt.xlabel('u')
plt.ylabel('x')
plt.grid(True, alpha=0.3)

# Resulting Exponential
plt.subplot(1, 3, 3)
exp_samples = -np.log(1-uniform_samples) / lambda_exp
plt.hist(exp_samples, bins=50, density=True, alpha=0.7)
x = np.linspace(0, 5, 1000)
plt.plot(x, lambda_exp * np.exp(-lambda_exp * x), 'r-', linewidth=2)
plt.title('Result: Exponential(λ)')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'transform_uniform_exponential.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Box-Muller Transform (Uniform to Normal)
print("\nExample 2: Box-Muller Transform")

plt.figure(figsize=(15, 5))

# Original Uniforms
plt.subplot(1, 3, 1)
u1 = np.random.uniform(0, 1, n_samples)
u2 = np.random.uniform(0, 1, n_samples)
plt.scatter(u1[:1000], u2[:1000], alpha=0.5)
plt.title('Original: Two Uniform(0,1)')
plt.xlabel('U₁')
plt.ylabel('U₂')
plt.grid(True, alpha=0.3)

# Transformation Process
plt.subplot(1, 3, 2)
r = np.sqrt(-2 * np.log(u1))
theta = 2 * np.pi * u2
plt.scatter(r[:1000], theta[:1000], alpha=0.5)
plt.title('Intermediate: (R,θ)\nR = √(-2ln(U₁)), θ = 2πU₂')
plt.xlabel('R')
plt.ylabel('θ')
plt.grid(True, alpha=0.3)

# Resulting Normal
plt.subplot(1, 3, 3)
z1 = r * np.cos(theta)
z2 = r * np.sin(theta)
plt.hist2d(z1, z2, bins=50, density=True)
plt.title('Result: Two Normal(0,1)')
plt.xlabel('Z₁')
plt.ylabel('Z₂')
plt.colorbar(label='Density')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'transform_uniform_normal.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Uniform to Beta
print("\nExample 3: Uniform to Beta Transformation")

plt.figure(figsize=(15, 5))

# Original Uniform
plt.subplot(1, 3, 1)
uniform_samples = np.random.uniform(0, 1, n_samples)
plt.hist(uniform_samples, bins=50, density=True, alpha=0.7)
plt.plot([0, 0, 1, 1], [0, 1, 1, 0], 'r-', linewidth=2)
plt.title('Original: Uniform(0,1)')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

# Transformation Process
plt.subplot(1, 3, 2)
x = np.linspace(0.001, 0.999, 1000)
alpha, beta = 2, 5
y = stats.beta.ppf(x, alpha, beta)
plt.plot(x, y, 'b-', linewidth=2)
plt.title('Transformation Function\nX = F⁻¹ᵦₑₜₐ(U)')
plt.xlabel('u')
plt.ylabel('x')
plt.grid(True, alpha=0.3)

# Resulting Beta
plt.subplot(1, 3, 3)
beta_samples = stats.beta.ppf(uniform_samples, alpha, beta)
plt.hist(beta_samples, bins=50, density=True, alpha=0.7)
x = np.linspace(0, 1, 1000)
plt.plot(x, stats.beta.pdf(x, alpha, beta), 'r-', linewidth=2)
plt.title('Result: Beta(α,β)')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'transform_uniform_beta.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 4: Uniform to Gamma (Rejection Sampling)
print("\nExample 4: Uniform to Gamma Transformation")

plt.figure(figsize=(15, 5))

# Generate samples using rejection sampling
k = 3  # shape parameter
accepted_samples = []
u1_accepted = []
u2_accepted = []
u1_rejected = []
u2_rejected = []

while len(accepted_samples) < n_samples:
    u1 = np.random.uniform(0, 1)
    u2 = np.random.uniform(0, 1)
    y = -np.log(u1)
    if u2 <= (y**(k-1) * np.exp(-y)) / (gamma(k) * np.exp(-k+1) * ((k-1)**(k-1))):
        accepted_samples.append(y)
        u1_accepted.append(u1)
        u2_accepted.append(u2)
    else:
        u1_rejected.append(u1)
        u2_rejected.append(u2)

# Original Space with Accept/Reject
plt.subplot(1, 3, 1)
plt.scatter(u1_rejected[:1000], u2_rejected[:1000], c='red', alpha=0.2, label='Rejected')
plt.scatter(u1_accepted[:1000], u2_accepted[:1000], c='blue', alpha=0.5, label='Accepted')
plt.title('Rejection Sampling\nin Uniform Space')
plt.xlabel('U₁')
plt.ylabel('U₂')
plt.legend()
plt.grid(True, alpha=0.3)

# Acceptance Region
plt.subplot(1, 3, 2)
x = np.linspace(0, 5, 1000)
y = (x**(k-1) * np.exp(-x)) / (gamma(k) * np.exp(-k+1) * ((k-1)**(k-1)))
y = np.minimum(y, 1)  # clip for visualization
plt.plot(x, y, 'b-', linewidth=2)
plt.fill_between(x, 0, y, alpha=0.3)
plt.title('Acceptance Region')
plt.xlabel('y = -ln(U₁)')
plt.ylabel('Acceptance Threshold')
plt.grid(True, alpha=0.3)

# Resulting Gamma
plt.subplot(1, 3, 3)
accepted_samples = np.array(accepted_samples)
plt.hist(accepted_samples, bins=50, density=True, alpha=0.7)
x = np.linspace(0, 10, 1000)
plt.plot(x, stats.gamma.pdf(x, k), 'r-', linewidth=2)
plt.title('Result: Gamma(k,1)')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'transform_uniform_gamma.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll transformation visualizations created successfully.") 