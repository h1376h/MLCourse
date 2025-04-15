import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import geom
import os

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

print("\n=== GEOMETRIC DISTRIBUTION EXAMPLES ===\n")

# Example 1: Coin Flips
print("Example 1: Coin Flips")
p = 0.5  # Probability of getting heads
k_values = np.arange(1, 11)  # Number of flips until first head
pmf_values = geom.pmf(k_values, p)
cdf_values = geom.cdf(k_values, p)

print(f"Probability of first head on k-th flip:")
for k, prob in zip(k_values, pmf_values):
    print(f"P(X = {k}) = {prob:.4f}")

print(f"\nExpected number of flips until first head: {1/p}")
print(f"Variance: {(1-p)/p**2:.2f}")

# Plot PMF
plt.figure(figsize=(10, 6))
plt.bar(k_values, pmf_values, color='skyblue', alpha=0.7)
plt.title('Geometric Distribution: Coin Flips (p=0.5)')
plt.xlabel('Number of Flips Until First Head')
plt.ylabel('Probability')
plt.grid(True, alpha=0.3)
plt.xticks(k_values)
plt.savefig(os.path.join(images_dir, 'geometric_coin_flips.png'))
plt.close()

# Example 2: Quality Control
print("\nExample 2: Quality Control")
p = 0.1  # Probability of finding a defective item
k = 5  # We want first success on 5th trial
prob = geom.pmf(k, p)
expected = 1/p
variance = (1-p)/p**2

print(f"Probability of first defect on 5th inspection: {prob:.4f}")
print(f"Expected number of inspections until first defect: {expected}")
print(f"Variance: {variance:.2f}")

# Plot PMF for quality control
k_values = np.arange(1, 21)
pmf_values = geom.pmf(k_values, p)
plt.figure(figsize=(10, 6))
plt.bar(k_values, pmf_values, color='lightgreen', alpha=0.7)
plt.title('Geometric Distribution: Quality Control (p=0.1)')
plt.xlabel('Number of Inspections Until First Defect')
plt.ylabel('Probability')
plt.grid(True, alpha=0.3)
plt.xticks(k_values)
plt.savefig(os.path.join(images_dir, 'geometric_quality_control.png'))
plt.close()

# Example 3: Website Conversion
print("\nExample 3: Website Conversion")
p = 0.02  # Probability of conversion
k = 50  # We want first success within 50 trials
prob = geom.cdf(k, p)
expected = 1/p
variance = (1-p)/p**2

print(f"Probability of first conversion within 50 visitors: {prob:.4f}")
print(f"Expected number of visitors until first conversion: {expected}")
print(f"Variance: {variance:.2f}")

# Plot CDF for website conversion
k_values = np.arange(1, 101)
cdf_values = geom.cdf(k_values, p)
plt.figure(figsize=(10, 6))
plt.plot(k_values, cdf_values, 'b-', linewidth=2)
plt.title('Geometric CDF: Website Conversion (p=0.02)')
plt.xlabel('Number of Visitors')
plt.ylabel('Cumulative Probability')
plt.grid(True, alpha=0.3)
plt.axvline(x=50, color='r', linestyle='--', label='50 visitors')
plt.legend()
plt.savefig(os.path.join(images_dir, 'geometric_website_conversion.png'))
plt.close()

# Example 4: Memoryless Property
print("\nExample 4: Memoryless Property")
p = 0.3
n = 5
k = 3

# Calculate P(X > n + k | X > n)
prob_conditional = (1 - geom.cdf(n + k, p)) / (1 - geom.cdf(n, p))
prob_unconditional = 1 - geom.cdf(k, p)

print(f"P(X > {n + k} | X > {n}) = {prob_conditional:.4f}")
print(f"P(X > {k}) = {prob_unconditional:.4f}")
print(f"Memoryless property holds: {abs(prob_conditional - prob_unconditional) < 1e-10}")

# Plot to demonstrate memoryless property
k_values = np.arange(1, 16)
pmf_values = geom.pmf(k_values, p)
plt.figure(figsize=(10, 6))
plt.bar(k_values, pmf_values, color='purple', alpha=0.7)
plt.title('Geometric Distribution: Memoryless Property (p=0.3)')
plt.xlabel('Number of Trials Until First Success')
plt.ylabel('Probability')
plt.grid(True, alpha=0.3)
plt.xticks(k_values)
plt.axvline(x=n, color='r', linestyle='--', label=f'n = {n}')
plt.axvline(x=n+k, color='g', linestyle='--', label=f'n+k = {n+k}')
plt.legend()
plt.savefig(os.path.join(images_dir, 'geometric_memoryless.png'))
plt.close()

print("\nAll geometric distribution example images created successfully.") 