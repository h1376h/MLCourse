import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, poisson
import os

print("\n=== EXPONENTIAL DISTRIBUTION EXAMPLES ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Create a subdirectory for exponential distribution images
images_dir = os.path.join(parent_dir, "Images", "Exponential_Distribution")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Example 1: Customer Service Waiting Times
print("\nExample 1: Customer Service Waiting Times")
lambda_calls = 2  # calls per hour
times = np.linspace(0, 2, 1000)  # 0 to 2 hours
pdf = expon.pdf(times, scale=1/lambda_calls)
cdf = expon.cdf(times, scale=1/lambda_calls)

# Calculate probabilities
p_30min = expon.cdf(0.5, scale=1/lambda_calls)
p_1hour = 1 - expon.cdf(1, scale=1/lambda_calls)
p_between = expon.cdf(1, scale=1/lambda_calls) - expon.cdf(0.5, scale=1/lambda_calls)

print(f"1. Probability next call within 30 minutes: {p_30min:.3f}")
print(f"2. Probability waiting more than 1 hour: {p_1hour:.3f}")
print(f"3. Probability between 30 min and 1 hour: {p_between:.3f}")

# Plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(times, pdf, 'b-', label='PDF')
plt.fill_between(times[times <= 0.5], pdf[times <= 0.5], alpha=0.3, color='blue')
plt.axvline(x=0.5, color='r', linestyle='--', label='30 minutes')
plt.title('PDF: Probability Density Function')
plt.xlabel('Time (hours)')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(times, cdf, 'g-', label='CDF')
plt.axvline(x=0.5, color='r', linestyle='--', label='30 minutes')
plt.axhline(y=p_30min, color='r', linestyle=':', label=f'P(X ≤ 0.5) = {p_30min:.3f}')
plt.title('CDF: Cumulative Distribution Function')
plt.xlabel('Time (hours)')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'customer_service.png'))
plt.close()

# Example 2: Radioactive Decay
print("\nExample 2: Radioactive Decay")
half_life = 5  # years
lambda_decay = np.log(2) / half_life
times = np.linspace(0, 15, 1000)  # 0 to 15 years
pdf = expon.pdf(times, scale=1/lambda_decay)
cdf = expon.cdf(times, scale=1/lambda_decay)

# Calculate probabilities
p_3years = expon.cdf(3, scale=1/lambda_decay)
p_8years = 1 - expon.cdf(8, scale=1/lambda_decay)

print(f"1. Decay rate parameter: {lambda_decay:.3f} decays/year")
print(f"2. Probability of decay within 3 years: {p_3years:.3f}")
print(f"3. Probability of surviving more than 8 years: {p_8years:.3f}")

# Plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(times, pdf, 'b-', label='PDF')
plt.fill_between(times[times <= 3], pdf[times <= 3], alpha=0.3, color='blue')
plt.axvline(x=3, color='r', linestyle='--', label='3 years')
plt.title('PDF: Probability Density Function')
plt.xlabel('Time (years)')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(times, cdf, 'g-', label='CDF')
plt.axvline(x=3, color='r', linestyle='--', label='3 years')
plt.axhline(y=p_3years, color='r', linestyle=':', label=f'P(X ≤ 3) = {p_3years:.3f}')
plt.title('CDF: Cumulative Distribution Function')
plt.xlabel('Time (years)')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'radioactive_decay.png'))
plt.close()

# Example 3: Memoryless Property
print("\nExample 3: Memoryless Property")
mean_lifetime = 1000  # hours
lambda_bulb = 1 / mean_lifetime
times = np.linspace(0, 2000, 1000)  # 0 to 2000 hours
pdf = expon.pdf(times, scale=1/lambda_bulb)
cdf = expon.cdf(times, scale=1/lambda_bulb)

# Calculate probabilities
p_1000_new = 1 - expon.cdf(1000, scale=1/lambda_bulb)
p_1000_after_500 = 1 - expon.cdf(1000, scale=1/lambda_bulb)  # Same due to memoryless property

print(f"1. Rate parameter: {lambda_bulb:.6f} failures/hour")
print(f"2. Probability new bulb lasts 1000 hours: {p_1000_new:.3f}")
print(f"3. Probability bulb lasts another 1000 hours after 500 hours: {p_1000_after_500:.3f}")

# Plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(times, pdf, 'b-', label='PDF')
plt.axvline(x=500, color='r', linestyle='--', label='500 hours')
plt.axvline(x=1500, color='g', linestyle='--', label='1500 hours')
plt.title('PDF: Probability Density Function')
plt.xlabel('Time (hours)')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(times, cdf, 'g-', label='CDF')
plt.axvline(x=500, color='r', linestyle='--', label='500 hours')
plt.axvline(x=1500, color='g', linestyle='--', label='1500 hours')
plt.title('CDF: Cumulative Distribution Function')
plt.xlabel('Time (hours)')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'memoryless_property.png'))
plt.close()

# Additional Visualization 1: Rate Parameter Comparison
print("\nAdditional Visualization 1: Rate Parameter Comparison")
lambdas = [0.5, 1, 2, 4]
times = np.linspace(0, 5, 1000)

plt.figure(figsize=(10, 6))
for lam in lambdas:
    pdf = expon.pdf(times, scale=1/lam)
    plt.plot(times, pdf, label=f'λ = {lam}')
plt.title('Exponential Distribution PDFs for Different Rate Parameters')
plt.xlabel('Time')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'rate_comparison.png'))
plt.close()

# Additional Visualization 2: Survival Function and Hazard Rate
print("\nAdditional Visualization 2: Survival Function and Hazard Rate")
lambda_val = 2
times = np.linspace(0, 3, 1000)
survival = 1 - expon.cdf(times, scale=1/lambda_val)
hazard = np.ones_like(times) * lambda_val

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(times, survival, 'b-', label='Survival Function')
plt.title('Survival Function')
plt.xlabel('Time')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(times, hazard, 'r-', label='Hazard Rate')
plt.title('Hazard Rate')
plt.xlabel('Time')
plt.ylabel('Hazard Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'survival_hazard.png'))
plt.close()

# Additional Visualization 3: Relationship with Poisson
print("\nAdditional Visualization 3: Relationship with Poisson")
lambda_val = 2
t = 1  # time interval
k_values = np.arange(0, 10)
poisson_pmf = poisson.pmf(k_values, lambda_val * t)

plt.figure(figsize=(10, 6))
plt.bar(k_values, poisson_pmf, alpha=0.7, label=f'Poisson(λ={lambda_val})')
plt.title('Poisson Distribution for Events in Time Interval')
plt.xlabel('Number of Events')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'poisson_relationship.png'))
plt.close()

# Additional Visualization 4: Minimum of Exponential Random Variables
print("\nAdditional Visualization 4: Minimum of Exponential Random Variables")
lambda1, lambda2 = 1, 2
times = np.linspace(0, 3, 1000)
min_pdf = (lambda1 + lambda2) * np.exp(-(lambda1 + lambda2) * times)

plt.figure(figsize=(10, 6))
plt.plot(times, min_pdf, 'b-', label='Minimum PDF')
plt.title('PDF of Minimum of Two Exponential Random Variables')
plt.xlabel('Time')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'minimum_exponential.png'))
plt.close()

print("\nAll examples completed and visualizations saved to Images directory.")
print(f"Images saved to: {images_dir}") 