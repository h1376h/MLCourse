import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_2_Quiz_6")
os.makedirs(save_dir, exist_ok=True)

# Set a clean style for plots
plt.style.use('seaborn-v0_8-whitegrid')

print("# Statement 1: The KL divergence between two identical probability distributions is always zero.")

# Define a sample probability distribution P
p = np.array([0.1, 0.2, 0.3, 0.4])
# Ensure it sums to 1
p = p / np.sum(p)

# Create an identical distribution Q
q = p

# Calculate KL divergence D_KL(P || Q)
# D_KL(P || Q) = Σ P(x) * log2(P(x) / Q(x))
kl_divergence = np.sum(p * np.log2(p / q))

# Mathematical explanation
print("\n#### Mathematical Definition of KL Divergence:")
print("D_KL(P || Q) = Σ P(x) * log2(P(x) / Q(x))")
print("")
print("#### Applying to Identical Distributions (P = Q):")
print("If P = Q, then P(x) = Q(x) for all x.")
print("The ratio P(x) / Q(x) = P(x) / P(x) = 1 (assuming P(x) > 0).")
print("log2(P(x) / Q(x)) = log2(1) = 0.")
print("Therefore, D_KL(P || P) = Σ P(x) * 0 = 0.")
print("")
print("#### Numerical Calculation:")
print(f"Distribution P: {p}")
print(f"Distribution Q: {q}")
print(f"KL Divergence D_KL(P || Q): {kl_divergence:.4f}")
print("(Note: Any small non-zero value is due to floating-point precision)")
print("")

# --- Visualization ---
fig, ax = plt.subplots(figsize=(8, 5))

x_labels = [f'Outcome {i+1}' for i in range(len(p))]
x_pos = np.arange(len(p))

# Plot distribution P
ax.bar(x_pos - 0.2, p, width=0.4, label='Distribution P', color='skyblue', alpha=0.8)
# Plot distribution Q (identical)
ax.bar(x_pos + 0.2, q, width=0.4, label='Distribution Q (Identical to P)', color='lightcoral', alpha=0.8)

ax.set_xlabel('Outcomes', fontsize=12)
ax.set_ylabel('Probability', fontsize=12)
ax.set_title('Two Identical Probability Distributions', fontsize=14)
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels)
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()

# Save the figure
img_path = os.path.join(save_dir, "statement1_identical_distributions.png")
plt.savefig(img_path, dpi=300, bbox_inches='tight')
plt.close()

print("#### Visual Verification:")
print(f"Plot showing identical distributions P and Q saved to: {img_path}")
print("")
print("#### Conclusion:")
print("The KL divergence D_KL(P || Q) measures the 'distance' or difference between")
print("probability distributions P and Q. When the distributions are identical (P = Q),")
print("the ratio P(x)/Q(x) is always 1, making log2(P(x)/Q(x)) = 0.")
print("Thus, the sum representing the KL divergence becomes zero.")
print("")
print("Therefore, Statement 1 is TRUE.") 