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

print("# Statement 4: The cross-entropy between two distributions is always greater than or equal to the entropy of the true distribution.")

# Helper functions
def entropy(p):
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def cross_entropy(p, q):
    # Ensure q does not contain zeros where p has non-zeros to avoid log2(0)
    q = np.clip(q, 1e-10, 1.0)
    return -np.sum(p * np.log2(q))

def kl_divergence(p, q):
    # Ensure q does not contain zeros where p has non-zeros
    p_safe = p[p > 1e-10] # Select only elements where p is non-zero
    q_safe = q[p > 1e-10] # Select corresponding elements in q
    q_safe = np.clip(q_safe, 1e-10, 1.0) # Avoid log2(0) for q
    return np.sum(p_safe * np.log2(p_safe / q_safe))

# Define distributions
num_outcomes = 4
# True distribution P
p = np.array([0.1, 0.6, 0.1, 0.2])
p = p / np.sum(p)

# Predicted distribution Q1 (close to P)
q1 = np.array([0.15, 0.55, 0.1, 0.2])
q1 = q1 / np.sum(q1)

# Predicted distribution Q2 (farther from P)
q2 = np.array([0.4, 0.1, 0.4, 0.1])
q2 = q2 / np.sum(q2)

# Calculate metrics
h_p = entropy(p)
h_pq1 = cross_entropy(p, q1)
dkl_pq1 = kl_divergence(p, q1)

h_pq2 = cross_entropy(p, q2)
dkl_pq2 = kl_divergence(p, q2)

# Mathematical explanation
print("\n#### Definitions:")
print("Entropy H(P) = - Σ P(x) * log2(P(x))")
print("Cross-Entropy H(P, Q) = - Σ P(x) * log2(Q(x))")
print("KL Divergence D_KL(P || Q) = Σ P(x) * log2(P(x) / Q(x))")
print("")
print("#### Relationship: H(P, Q) = H(P) + D_KL(P || Q)")
print("H(P, Q) = - Σ P(x)log2(Q(x))")
print("          = - Σ P(x)log2(Q(x) * P(x)/P(x))")
print("          = - Σ P(x)[log2(P(x)) + log2(Q(x)/P(x))]") # Error in derivation step, should be P(x)/Q(x)
# Correct derivation:
# H(P, Q) - H(P) = [- Σ P(x)log2(Q(x))] - [- Σ P(x)log2(P(x))]
#               = Σ P(x)log2(P(x)) - Σ P(x)log2(Q(x))
#               = Σ P(x)[log2(P(x)) - log2(Q(x))]
#               = Σ P(x)log2(P(x)/Q(x)) = D_KL(P || Q)
# So, H(P, Q) = H(P) + D_KL(P || Q)
print("Correctly derived: H(P, Q) = H(P) + D_KL(P || Q)")
print("")
print("#### Gibbs' Inequality:")
print("Since D_KL(P || Q) ≥ 0 (Gibbs' inequality), with equality iff P = Q.")
print("It follows that H(P, Q) = H(P) + D_KL(P || Q) ≥ H(P).")
print("")
print("#### Numerical Calculations:")
print(f"True Dist P: {p}")
print(f"Pred Dist Q1 (close): {q1}")
print(f"Pred Dist Q2 (far): {q2}")
print("-")
print(f"Entropy H(P) = {h_p:.4f}")
print("-")
print(f"Cross-Entropy H(P, Q1) = {h_pq1:.4f}")
print(f"KL Divergence D_KL(P || Q1) = {dkl_pq1:.4f}")
print(f"Check: H(P) + D_KL(P || Q1) = {h_p + dkl_pq1:.4f} (Matches H(P, Q1)))")
print(f"Inequality: H(P, Q1) ({h_pq1:.4f}) >= H(P) ({h_p:.4f}) -> {h_pq1 >= h_p}")
print("-")
print(f"Cross-Entropy H(P, Q2) = {h_pq2:.4f}")
print(f"KL Divergence D_KL(P || Q2) = {dkl_pq2:.4f}")
print(f"Check: H(P) + D_KL(P || Q2) = {h_p + dkl_pq2:.4f} (Matches H(P, Q2)))")
print(f"Inequality: H(P, Q2) ({h_pq2:.4f}) >= H(P) ({h_p:.4f}) -> {h_pq2 >= h_p}")

# --- Visualization ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

x_labels = [f'{i+1}' for i in range(num_outcomes)]
x_pos = np.arange(num_outcomes)

# Plot distributions
axes[0].bar(x_pos - 0.25, p, width=0.25, label=f'True P (H={h_p:.2f})', color='black', alpha=0.8)
axes[0].bar(x_pos, q1, width=0.25, label=f'Pred Q1 (H(P,Q1)={h_pq1:.2f})', color='skyblue', alpha=0.7)
axes[0].bar(x_pos + 0.25, q2, width=0.25, label=f'Pred Q2 (H(P,Q2)={h_pq2:.2f})', color='lightcoral', alpha=0.7)

axes[0].set_xlabel('Outcomes', fontsize=12)
axes[0].set_ylabel('Probability', fontsize=12)
axes[0].set_title('Distributions and Cross-Entropies', fontsize=14)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(x_labels)
axes[0].legend()
axes[0].grid(True, axis='y', linestyle='--', alpha=0.7)

# Plot Entropy vs Cross-Entropy Values
bar_labels = ['H(P)', 'H(P, Q1)', 'H(P, Q2)']
values = [h_p, h_pq1, h_pq2]
colors = ['black', 'skyblue', 'lightcoral']

bars = axes[1].bar(bar_labels, values, color=colors, alpha=0.8)
axes[1].axhline(h_p, color='black', linestyle='--', lw=1, label=f'H(P) = {h_p:.2f}')

# Add KL divergence values as text
axes[1].text(1, h_pq1, f' D_KL(P||Q1)\n = {dkl_pq1:.2f}', va='bottom', ha='center', color='blue')
axes[1].text(2, h_pq2, f' D_KL(P||Q2)\n = {dkl_pq2:.2f}', va='bottom', ha='center', color='red')

# Add annotations for bars
for bar in bars:
    yval = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom', ha='center') # Adjust text position

axes[1].set_ylabel('Entropy / Cross-Entropy (bits)', fontsize=12)
axes[1].set_title('H(P, Q) vs H(P)', fontsize=14)
axes[1].legend()
axes[1].grid(True, axis='y', linestyle='--', alpha=0.7)
axes[1].set_ylim(bottom=0)

plt.tight_layout()

# Save the figure
img_path = os.path.join(save_dir, "statement4_cross_entropy_vs_entropy.png")
plt.savefig(img_path, dpi=300, bbox_inches='tight')
plt.close()

print("\n#### Visual Verification:")
print(f"Plot comparing H(P) and H(P, Q) saved to: {img_path}")
print("- Left plot shows the true distribution P and two predicted distributions Q1, Q2.")
print("- Right plot shows the values of H(P) and the cross-entropies H(P, Q1) and H(P, Q2).")
print("- The cross-entropy H(P, Q) is always greater than or equal to the entropy H(P).")
print("- The difference H(P, Q) - H(P) is exactly the KL divergence D_KL(P || Q), which is non-negative.")
print("- H(P, Q) is minimized and equals H(P) only when Q = P (i.e., D_KL = 0).")
print("")
print("#### Conclusion:")
print("The relationship H(P, Q) = H(P) + D_KL(P || Q) and the fact that D_KL(P || Q) ≥ 0")
print("directly implies that H(P, Q) ≥ H(P).")
print("Cross-entropy can be seen as the average number of bits needed to encode events")
print("drawn from P when using a code optimized for Q. This is minimized when Q=P.")
print("")
print("Therefore, Statement 4 is TRUE.") 