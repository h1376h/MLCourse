import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import nbinom
import os

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

def plot_negative_binomial(r, p, title, xlabel, ylabel, filename):
    # Calculate the PMF for a range of k values
    k_values = np.arange(0, 50)
    pmf_values = nbinom.pmf(k_values, r, p)
    
    # Calculate mean and variance
    mean = nbinom.mean(r, p)
    variance = nbinom.var(r, p)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(k_values, pmf_values, color='skyblue', alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    
    # Add mean and variance information
    plt.axvline(x=mean, color='red', linestyle='--', label=f'Mean = {mean:.2f}')
    plt.axvline(x=mean + np.sqrt(variance), color='green', linestyle=':', label=f'Mean ± SD')
    plt.axvline(x=mean - np.sqrt(variance), color='green', linestyle=':')
    
    # Add probability values for the first few bars
    for i, prob in enumerate(pmf_values[:10]):
        if prob > 0.01:
            plt.text(k_values[i], prob + 0.01, f'{prob:.3f}', ha='center', fontsize=8)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, filename), dpi=100, bbox_inches='tight')
    plt.close()
    
    return mean, variance

print("\n=== NEGATIVE BINOMIAL DISTRIBUTION EXAMPLES ===\n")

# Example 1: Website Visits
print("\nExample 1: Website Visits")
r1, p1 = 5, 0.3
mean1, var1 = plot_negative_binomial(
    r1, p1,
    "Negative Binomial Distribution: Website Conversions\n(r=5, p=0.3)",
    "Number of Non-Converting Visits",
    "Probability",
    "negative_binomial_website.png"
)
print(f"Parameters: r = {r1}, p = {p1}")
print(f"Mean (expected non-converting visits): {mean1:.2f}")
print(f"Variance: {var1:.2f}")
print(f"Standard Deviation: {np.sqrt(var1):.2f}")

# Example 2: Customer Support Calls
print("\nExample 2: Customer Support Calls")
r2, p2 = 3, 0.4
mean2, var2 = plot_negative_binomial(
    r2, p2,
    "Negative Binomial Distribution: Support Call Resolutions\n(r=3, p=0.4)",
    "Number of Unsuccessful Calls",
    "Probability",
    "negative_binomial_support.png"
)
print(f"Parameters: r = {r2}, p = {p2}")
print(f"Mean (expected unsuccessful calls): {mean2:.2f}")
print(f"Variance: {var2:.2f}")
print(f"Standard Deviation: {np.sqrt(var2):.2f}")

# Example 3: Machine Learning Model Training
print("\nExample 3: Machine Learning Model Training")
r3, p3 = 4, 0.25
mean3, var3 = plot_negative_binomial(
    r3, p3,
    "Negative Binomial Distribution: Model Training\n(r=4, p=0.25)",
    "Number of Unsuccessful Epochs",
    "Probability",
    "negative_binomial_training.png"
)
print(f"Parameters: r = {r3}, p = {p3}")
print(f"Mean (expected unsuccessful epochs): {mean3:.2f}")
print(f"Variance: {var3:.2f}")
print(f"Standard Deviation: {np.sqrt(var3):.2f}")

# Create a comparison plot
plt.figure(figsize=(15, 5))

# Plot 1: Website Visits
plt.subplot(1, 3, 1)
k_values = np.arange(0, 50)
pmf1 = nbinom.pmf(k_values, r1, p1)
plt.bar(k_values, pmf1, color='skyblue', alpha=0.7)
plt.title(f"Website (r={r1}, p={p1})")
plt.xlabel("Non-Converting Visits")
plt.ylabel("Probability")
plt.grid(True, alpha=0.3)

# Plot 2: Customer Support
plt.subplot(1, 3, 2)
pmf2 = nbinom.pmf(k_values, r2, p2)
plt.bar(k_values, pmf2, color='lightgreen', alpha=0.7)
plt.title(f"Support (r={r2}, p={p2})")
plt.xlabel("Unsuccessful Calls")
plt.grid(True, alpha=0.3)

# Plot 3: Model Training
plt.subplot(1, 3, 3)
pmf3 = nbinom.pmf(k_values, r3, p3)
plt.bar(k_values, pmf3, color='salmon', alpha=0.7)
plt.title(f"Training (r={r3}, p={p3})")
plt.xlabel("Unsuccessful Epochs")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, "negative_binomial_comparison.png"), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll negative binomial distribution plots created successfully.") 