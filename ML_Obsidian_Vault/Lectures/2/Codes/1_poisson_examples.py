import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import os

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

print("\n=== POISSON DISTRIBUTION EXAMPLES ===\n")

# Example 1: Website Traffic Analysis
print("\nExample 1: Website Traffic Analysis")
lambda1 = 5  # average visitors per minute

# Calculate probabilities
p_exactly_3 = poisson.pmf(3, lambda1)
p_at_most_2 = poisson.cdf(2, lambda1)
p_more_than_4 = 1 - poisson.cdf(4, lambda1)

print(f"1. Probability of exactly 3 visitors: {p_exactly_3:.4f}")
print(f"2. Probability of at most 2 visitors: {p_at_most_2:.4f}")
print(f"3. Probability of more than 4 visitors: {p_more_than_4:.4f}")

# Example 2: Customer Service Calls
print("\nExample 2: Customer Service Calls")
lambda2 = 10  # average calls per hour

# Calculate probabilities
p_exactly_8 = poisson.pmf(8, lambda2)
p_between_7_12 = poisson.cdf(12, lambda2) - poisson.cdf(6, lambda2)
p_at_least_15 = 1 - poisson.cdf(14, lambda2)

print(f"1. Probability of exactly 8 calls: {p_exactly_8:.4f}")
print(f"2. Probability of between 7 and 12 calls: {p_between_7_12:.4f}")
print(f"3. Probability of at least 15 calls: {p_at_least_15:.4f}")

# Example 3: Defective Products
print("\nExample 3: Defective Products")
lambda3 = 10  # average defects per 5000 units

# Calculate probabilities
p_exactly_8_defects = poisson.pmf(8, lambda3)
p_at_most_5_defects = poisson.cdf(5, lambda3)
p_more_than_12_defects = 1 - poisson.cdf(12, lambda3)

print(f"1. Probability of exactly 8 defects: {p_exactly_8_defects:.4f}")
print(f"2. Probability of at most 5 defects: {p_at_most_5_defects:.4f}")
print(f"3. Probability of more than 12 defects: {p_more_than_12_defects:.4f}")

# Create visualizations

# Plot 1: Different lambda values
plt.figure(figsize=(10, 6))
x = np.arange(0, 20)
for l in [1, 4, 10]:
    plt.plot(x, poisson.pmf(x, l), 'o-', label=f'λ = {l}')
plt.title('Poisson Distribution for Different λ Values')
plt.xlabel('Number of Events')
plt.ylabel('Probability')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'poisson_lambda_comparison.png'), dpi=100, bbox_inches='tight')
plt.close()

# Plot 2: Example 1 - Website Traffic
plt.figure(figsize=(10, 6))
x = np.arange(0, 15)
plt.bar(x, poisson.pmf(x, lambda1), alpha=0.7)
plt.axvline(x=3, color='red', linestyle='--', label='Exactly 3 visitors')
plt.axvline(x=2, color='green', linestyle='--', label='At most 2 visitors')
plt.axvline(x=4, color='blue', linestyle='--', label='More than 4 visitors')
plt.title('Website Traffic Analysis (λ = 5)')
plt.xlabel('Number of Visitors')
plt.ylabel('Probability')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'poisson_website_traffic.png'), dpi=100, bbox_inches='tight')
plt.close()

# Plot 3: Example 2 - Call Center
plt.figure(figsize=(10, 6))
x = np.arange(0, 25)
plt.bar(x, poisson.pmf(x, lambda2), alpha=0.7)
plt.axvline(x=8, color='red', linestyle='--', label='Exactly 8 calls')
plt.axvline(x=7, color='green', linestyle='--', label='Between 7-12 calls')
plt.axvline(x=12, color='green', linestyle='--')
plt.axvline(x=15, color='blue', linestyle='--', label='At least 15 calls')
plt.title('Call Center Analysis (λ = 10)')
plt.xlabel('Number of Calls')
plt.ylabel('Probability')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'poisson_call_center.png'), dpi=100, bbox_inches='tight')
plt.close()

# Plot 4: Example 3 - Defective Products
plt.figure(figsize=(10, 6))
x = np.arange(0, 25)
plt.bar(x, poisson.pmf(x, lambda3), alpha=0.7)
plt.axvline(x=8, color='red', linestyle='--', label='Exactly 8 defects')
plt.axvline(x=5, color='green', linestyle='--', label='At most 5 defects')
plt.axvline(x=12, color='blue', linestyle='--', label='More than 12 defects')
plt.title('Defective Products Analysis (λ = 10)')
plt.xlabel('Number of Defects')
plt.ylabel('Probability')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'poisson_defective_products.png'), dpi=100, bbox_inches='tight')
plt.close()

# Plot 5: Probability Comparison Across Examples
plt.figure(figsize=(12, 6))
examples = ['Website Traffic', 'Call Center', 'Defective Products']
probabilities = {
    'Exactly': [p_exactly_3, p_exactly_8, p_exactly_8_defects],
    'At Most': [p_at_most_2, p_between_7_12, p_at_most_5_defects],
    'More Than': [p_more_than_4, p_at_least_15, p_more_than_12_defects]
}

x = np.arange(len(examples))
width = 0.25

for i, (label, probs) in enumerate(probabilities.items()):
    plt.bar(x + i*width, probs, width, label=label)

plt.xlabel('Example')
plt.ylabel('Probability')
plt.title('Probability Comparison Across Examples')
plt.xticks(x + width, examples)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'poisson_probability_comparison.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nVisualizations have been saved as separate images:")
print("1. poisson_lambda_comparison.png")
print("2. poisson_website_traffic.png")
print("3. poisson_call_center.png")
print("4. poisson_defective_products.png")
print("5. poisson_probability_comparison.png") 