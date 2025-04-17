import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_2_Quiz_6")
os.makedirs(save_dir, exist_ok=True)

# Set a clean style for plots
plt.style.use('seaborn-v0_8-whitegrid')

print("# Statement 2: The entropy of a uniform distribution is always higher than any other distribution over the same set of values.")

# Helper function to calculate entropy
def entropy(p):
    p = p[p > 0] # Avoid log2(0)
    return -np.sum(p * np.log2(p))

# Define the number of outcomes
num_outcomes = 5

# 1. Uniform Distribution (U)
uniform_p = np.ones(num_outcomes) / num_outcomes
entropy_uniform = entropy(uniform_p)

# 2. Non-Uniform Distribution 1 (P1)
non_uniform_p1 = np.array([0.4, 0.2, 0.2, 0.1, 0.1])
non_uniform_p1 = non_uniform_p1 / np.sum(non_uniform_p1) # Ensure sums to 1
entropy_p1 = entropy(non_uniform_p1)

# 3. Non-Uniform Distribution 2 (P2 - more skewed)
non_uniform_p2 = np.array([0.7, 0.1, 0.1, 0.05, 0.05])
non_uniform_p2 = non_uniform_p2 / np.sum(non_uniform_p2) # Ensure sums to 1
entropy_p2 = entropy(non_uniform_p2)

# Mathematical explanation
print("\n#### Mathematical Definition of Entropy:")
print("H(P) = - Σ P(x) * log2(P(x))")
print("Entropy measures the uncertainty or randomness of a distribution.")
print("")
print("#### Maximum Entropy Principle:")
print("For a discrete random variable with K possible outcomes, the entropy is maximized")
print("when the distribution is uniform, i.e., P(x) = 1/K for all x.")
print("The maximum entropy value is log2(K).")
print("")
print("#### Numerical Calculation (K=5 outcomes):")
print(f"Uniform Distribution U: {uniform_p}")
print(f"  Entropy H(U) = {entropy_uniform:.4f}")
print(f"  Theoretical Maximum Entropy = log2({num_outcomes}) = {np.log2(num_outcomes):.4f}")
print("")
print(f"Non-Uniform Dist. P1: {non_uniform_p1}")
print(f"  Entropy H(P1) = {entropy_p1:.4f}")
print("")
print(f"Non-Uniform Dist. P2: {non_uniform_p2}")
print(f"  Entropy H(P2) = {entropy_p2:.4f}")
print("")
print(f"Comparison: H(U) ({entropy_uniform:.4f}) >= H(P1) ({entropy_p1:.4f}) and H(U) ({entropy_uniform:.4f}) >= H(P2) ({entropy_p2:.4f})")

# --- Visualization ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

x_labels = [f'{i+1}' for i in range(num_outcomes)]
x_pos = np.arange(num_outcomes)

# Plot distributions
axes[0].bar(x_pos - 0.25, uniform_p, width=0.25, label=f'Uniform (H={entropy_uniform:.2f})', color='green', alpha=0.7)
axes[0].bar(x_pos, non_uniform_p1, width=0.25, label=f'Non-Uniform 1 (H={entropy_p1:.2f})', color='skyblue', alpha=0.7)
axes[0].bar(x_pos + 0.25, non_uniform_p2, width=0.25, label=f'Non-Uniform 2 (H={entropy_p2:.2f})', color='lightcoral', alpha=0.7)

axes[0].set_xlabel('Outcomes', fontsize=12)
axes[0].set_ylabel('Probability', fontsize=12)
axes[0].set_title('Distributions and Their Entropies', fontsize=14)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(x_labels)
axes[0].legend()
axes[0].grid(True, axis='y', linestyle='--', alpha=0.7)

# Plot Entropy vs. Skewness (using binary example for simplicity)
probs = np.linspace(0.001, 0.999, 200)
binary_entropies = [- (p * np.log2(p) + (1-p) * np.log2(1-p)) for p in probs]

axes[1].plot(probs, binary_entropies, lw=2, color='purple')
axes[1].axvline(0.5, color='green', linestyle='--', label='Uniform (p=0.5)')
axes[1].set_xlabel('Probability of Outcome 1 (p)', fontsize=12)
axes[1].set_ylabel('Entropy H', fontsize=12)
axes[1].set_title('Entropy of a Binary Variable', fontsize=14)
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.7)
axes[1].text(0.5, max(binary_entropies), f' Max Entropy = {max(binary_entropies):.2f}', va='bottom', ha='center')

plt.tight_layout()

# Save the figure
img_path = os.path.join(save_dir, "statement2_max_entropy.png")
plt.savefig(img_path, dpi=300, bbox_inches='tight')
plt.close()

print("#### Visual Verification:")
print(f"Plot comparing entropies saved to: {img_path}")
print("- Left plot shows distributions: Uniform has highest entropy.")
print("- Right plot (binary case) shows entropy peaks at p=0.5 (uniform) and decreases as distribution becomes skewed.")
print("")
print("#### Conclusion:")
print("Entropy quantifies the uncertainty or surprise in a distribution.")
print("A uniform distribution represents maximum uncertainty because every outcome is equally likely.")
print("Any deviation from uniformity reduces uncertainty (some outcomes become more predictable) and thus lowers entropy.")
print("The maximum possible entropy for K outcomes is log2(K), achieved only by the uniform distribution.")
print("")
print("Therefore, Statement 2 is TRUE.") 