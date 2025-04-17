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

print("# Statement 5: If the entropy of a random variable X is zero, then X must be deterministic.")

# Helper function to calculate entropy
def entropy(p):
    # Use the convention that 0 * log2(0) = 0
    p_safe = np.array(p)
    p_safe = p_safe[p_safe > 1e-12] # Consider only non-zero probabilities
    return -np.sum(p_safe * np.log2(p_safe))

# Define distributions
num_outcomes = 4

# 1. Deterministic Distribution (P_det)
# Outcome 2 has probability 1, others have 0
p_det = np.zeros(num_outcomes)
p_det[1] = 1.0
entropy_det = entropy(p_det)

# 2. Near-Deterministic Distribution (P_near_det)
p_near_det = np.array([0.01, 0.97, 0.01, 0.01])
p_near_det = p_near_det / np.sum(p_near_det)
entropy_near_det = entropy(p_near_det)

# 3. Uniform Distribution (P_uniform) - Max entropy for comparison
p_uniform = np.ones(num_outcomes) / num_outcomes
entropy_uniform = entropy(p_uniform)

# Mathematical explanation
print("\n#### Mathematical Definition of Entropy:")
print("H(P) = - Σ P(x) * log2(P(x))")
print("(Using the convention 0 * log2(0) = 0)")
print("")
print("#### Condition for Zero Entropy:")
print("H(P) = 0 if and only if all terms P(x) * log2(P(x)) are zero.")
print("Since log2(P(x)) is non-zero for 0 < P(x) < 1, this requires P(x) to be either 0 or 1.")
print("Because Σ P(x) must equal 1, exactly one P(x_i) must be 1, and all other P(x_j) (j≠i) must be 0.")
print("This describes a deterministic distribution, where the outcome is certain.")
print("")
print("#### Numerical Calculations:")
print(f"Deterministic Dist P_det: {p_det}")
print(f"  Entropy H(P_det) = {entropy_det:.4f}")
print("")
print(f"Near-Deterministic P_near_det: {p_near_det}")
print(f"  Entropy H(P_near_det) = {entropy_near_det:.4f} (Small but positive)")
print("")
print(f"Uniform Dist P_uniform: {p_uniform}")
print(f"  Entropy H(P_uniform) = {entropy_uniform:.4f} (Maximum)")

# --- Visualization ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

x_labels = [f'{i+1}' for i in range(num_outcomes)]
x_pos = np.arange(num_outcomes)

# Plot distributions
axes[0].bar(x_pos - 0.25, p_det, width=0.25, label=f'Deterministic (H={entropy_det:.2f})', color='red', alpha=0.8)
axes[0].bar(x_pos, p_near_det, width=0.25, label=f'Near-Deterministic (H={entropy_near_det:.2f})', color='orange', alpha=0.7)
axes[0].bar(x_pos + 0.25, p_uniform, width=0.25, label=f'Uniform (H={entropy_uniform:.2f})', color='green', alpha=0.7)

axes[0].set_xlabel('Outcomes', fontsize=12)
axes[0].set_ylabel('Probability', fontsize=12)
axes[0].set_title('Distributions and Their Entropies', fontsize=14)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(x_labels)
axes[0].legend()
axes[0].grid(True, axis='y', linestyle='--', alpha=0.7)

# Plot Entropy vs. Probability for a Binary Variable
probs = np.linspace(0, 1, 200)
# Handle p=0 and p=1 cases explicitly due to log2(0)
binary_entropies = []
for p in probs:
    if p == 0 or p == 1:
        binary_entropies.append(0)
    else:
        binary_entropies.append(- (p * np.log2(p) + (1-p) * np.log2(1-p)))

axes[1].plot(probs, binary_entropies, lw=2, color='purple')
axes[1].set_xlabel('Probability of Outcome 1 (p)', fontsize=12)
axes[1].set_ylabel('Entropy H', fontsize=12)
axes[1].set_title('Entropy of a Binary Variable vs. Determinism', fontsize=14)
axes[1].grid(True, linestyle='--', alpha=0.7)
axes[1].axhline(0, color='red', linestyle=':')
axes[1].text(0, 0.05, 'H=0\n(Deterministic p=0)', va='bottom', ha='center', color='red')
axes[1].text(1, 0.05, 'H=0\n(Deterministic p=1)', va='bottom', ha='center', color='red')
axes[1].text(0.5, max(binary_entropies), f'Max Entropy\n(Uniform p=0.5)', va='bottom', ha='center', color='green')

plt.tight_layout()

# Save the figure
img_path = os.path.join(save_dir, "statement5_zero_entropy.png")
plt.savefig(img_path, dpi=300, bbox_inches='tight')
plt.close()

print("\n#### Visual Verification:")
print(f"Plot illustrating zero entropy saved to: {img_path}")
print("- Left plot shows that the deterministic distribution has H=0, while others have H>0.")
print("- Right plot (binary case) clearly shows entropy is zero only at the extremes (p=0 or p=1), representing deterministic outcomes.")
print("")
print("#### Conclusion:")
print("Entropy measures uncertainty. Zero entropy implies zero uncertainty.")
print("This occurs only when the outcome is completely predictable, meaning one outcome")
print("has a probability of 1 and all others have a probability of 0.")
print("This is the definition of a deterministic random variable.")
print("")
print("Therefore, Statement 5 is TRUE.") 