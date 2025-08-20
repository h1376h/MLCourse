import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import entropy
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_2_Quiz_3")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("QUESTION 3: FEATURE IRRELEVANCE AND KL DIVERGENCE")
print("=" * 80)

# ============================================================================
# PART 1: What does it mean for a feature to be irrelevant?
# ============================================================================
print("\n" + "="*60)
print("PART 1: What does it mean for a feature to be irrelevant?")
print("="*60)

print("A feature is considered irrelevant when it provides no useful information")
print("for predicting the target variable. Mathematically, this means:")
print("• P(y|x) = P(y) for all values of x")
print("• The conditional probability of y given x equals the marginal probability of y")
print("• The feature x and target y are statistically independent")

# Create visualization for feature irrelevance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Example 1: Relevant feature (strong correlation)
np.random.seed(42)
x_relevant = np.random.normal(0, 1, 1000)
y_relevant = 2 * x_relevant + np.random.normal(0, 0.5, 1000)

ax1.scatter(x_relevant, y_relevant, alpha=0.6, color='blue')
ax1.set_xlabel('$x$ (Relevant Feature)')
ax1.set_ylabel('$y$ (Target)')
ax1.set_title('Feature Relevance: Strong Linear Correlation')
ax1.grid(True, alpha=0.3)

# Example 2: Irrelevant feature (no correlation)
x_irrelevant = np.random.normal(0, 1, 1000)
y_irrelevant = np.random.normal(0, 1, 1000)

ax2.scatter(x_irrelevant, y_irrelevant, alpha=0.6, color='red')
ax2.set_xlabel('$x$ (Irrelevant Feature)')
ax2.set_ylabel('$y$ (Target)')
ax2.set_title('Feature Irrelevance: No Correlation')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_relevance_comparison.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# PART 2: If P(y|x) = P(y) for all values of x, what does this suggest?
# ============================================================================
print("\n" + "="*60)
print("PART 2: If P(y|x) = P(y) for all values of x, what does this suggest?")
print("="*60)

print("If P(y|x) = P(y) for all values of x, this suggests:")
print("1. Statistical Independence: X and Y are independent random variables")
print("2. No Information Gain: Knowing x provides no information about y")
print("3. Feature Irrelevance: x is completely irrelevant for predicting y")
print("4. Zero Mutual Information: I(X;Y) = 0")

# Create visualization showing independence
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Example: Independent variables
np.random.seed(42)
x_indep = np.random.choice([0, 1, 2, 3], size=1000, p=[0.25, 0.25, 0.25, 0.25])
y_indep = np.random.choice([0, 1], size=1000, p=[0.6, 0.4])

# Create joint distribution table
joint_dist = np.zeros((4, 2))
for i in range(4):
    for j in range(2):
        joint_dist[i, j] = np.sum((x_indep == i) & (y_indep == j)) / 1000

# Marginal distributions
p_x = np.sum(joint_dist, axis=1)
p_y = np.sum(joint_dist, axis=0)

# Conditional distributions P(y|x)
cond_y_given_x = joint_dist / p_x[:, np.newaxis]

# Plot joint distribution
im1 = ax1.imshow(joint_dist, cmap='Blues', aspect='auto')
ax1.set_xlabel('$y$ values')
ax1.set_ylabel('$x$ values')
ax1.set_title('Joint Distribution P(x,y) - Independence Structure')
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1, 2, 3])
ax1.set_xticklabels(['0', '1'])
ax1.set_yticklabels(['0', '1', '2', '3'])

# Add text annotations
for i in range(4):
    for j in range(2):
        text = ax1.text(j, i, f'{joint_dist[i, j]:.3f}', 
                        ha="center", va="center", color="black", fontweight='bold')

# Plot conditional distributions
im2 = ax2.imshow(cond_y_given_x, cmap='Reds', aspect='auto')
ax2.set_xlabel('$y$ values')
ax2.set_ylabel('$x$ values')
ax2.set_title('Conditional Distributions P(y|x) - Independence Check')
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1, 2, 3])
ax2.set_xticklabels(['0', '1'])
ax2.set_yticklabels(['0', '1', '2', '3'])

# Add text annotations
for i in range(4):
    for j in range(2):
        text = ax2.text(j, i, f'{cond_y_given_x[i, j]:.3f}', 
                        ha="center", va="center", color="black", fontweight='bold')

# Add colorbar
plt.colorbar(im1, ax=ax1, label='Probability')
plt.colorbar(im2, ax=ax2, label='Probability')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'independence_visualization.png'), dpi=300, bbox_inches='tight')

print(f"\nMarginal distribution P(y): {p_y}")
print(f"Conditional distributions P(y|x):")
for i in range(4):
    print(f"  P(y|x={i}) = {cond_y_given_x[i, :]}")

# ============================================================================
# PART 3: KL divergence calculation and interpretation
# ============================================================================
print("\n" + "="*60)
print("PART 3: KL divergence calculation and interpretation")
print("="*60)

# Given distributions
P = np.array([0.3, 0.4, 0.3])
Q = np.array([0.2, 0.5, 0.3])

print(f"Distribution P: {P}")
print(f"Distribution Q: {Q}")

# Calculate KL divergence D_KL(P||Q)
def kl_divergence(p, q):
    """Calculate KL divergence D_KL(P||Q)"""
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    p_safe = p + epsilon
    q_safe = q + epsilon
    
    # Normalize to ensure they sum to 1
    p_safe = p_safe / np.sum(p_safe)
    q_safe = q_safe / np.sum(q_safe)
    
    return np.sum(p_safe * np.log(p_safe / q_safe))

kl_div = kl_divergence(P, Q)
print(f"\nCalculated KL divergence D_KL(P||Q) = {kl_div:.6f}")

# Verify with scipy
kl_div_scipy = entropy(P, Q)
print(f"Verified with scipy: D_KL(P||Q) = {kl_div_scipy:.6f}")

# Given KL divergence = 0.05
given_kl = 0.05
print(f"\nGiven KL divergence = {given_kl}")

print("\nInterpretation:")
print("• D_KL(P||Q) = 0.05 indicates that distributions P and Q are relatively similar")
print("• The small value suggests that Q is a good approximation of P")
print("• However, they are not identical (which would give D_KL = 0)")
print("• This suggests that the features represented by P and Q are related but not identical")

# Create visualization of the distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot distributions
x_pos = np.arange(len(P))
width = 0.35

ax1.bar(x_pos - width/2, P, width, label='Distribution P', alpha=0.7, color='blue')
ax1.bar(x_pos + width/2, Q, width, label='Distribution Q', alpha=0.7, color='red')
ax1.set_xlabel('Outcome')
ax1.set_ylabel('Probability')
ax1.set_title('Probability Distributions P and Q Comparison')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(['1', '2', '3'])
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add probability values on bars
for i, (p_val, q_val) in enumerate(zip(P, Q)):
    ax1.text(i - width/2, p_val + 0.01, f'{p_val:.2f}', ha='center', va='bottom')
    ax1.text(i + width/2, q_val + 0.01, f'{q_val:.2f}', ha='center', va='bottom')

# Plot KL divergence components
kl_components = P * np.log(P / Q)
# Handle cases where P[i] = 0 (log(0) = -inf, but 0 * -inf = 0)
kl_components = np.where(P == 0, 0, kl_components)

ax2.bar(x_pos, kl_components, color='green', alpha=0.7)
ax2.set_xlabel('Outcome')
ax2.set_ylabel('Contribution to KL Divergence')
ax2.set_title('KL Divergence Components: P[i] * log(P[i]/Q[i])')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(['1', '2', '3'])
ax2.grid(True, alpha=0.3)

# Add values on bars
for i, comp in enumerate(kl_components):
    ax2.text(i, comp + 0.001, f'{comp:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'kl_divergence_analysis.png'), dpi=300, bbox_inches='tight')

print(f"\nIndividual contributions to KL divergence:")
for i, comp in enumerate(kl_components):
    print(f"  Outcome {i+1}: P[{i}] * log(P[{i}]/Q[{i}]) = {P[i]:.2f} * log({P[i]:.2f}/{Q[i]:.2f}) = {comp:.6f}")

# ============================================================================
# PART 4: KL divergence for uniform distributions
# ============================================================================
print("\n" + "="*60)
print("PART 4: KL divergence for uniform distributions")
print("="*60)

# Given uniform distributions
P_uniform = np.array([0.25, 0.25, 0.25, 0.25])
Q_uniform = np.array([0.5, 0.5, 0, 0])

print(f"Uniform distribution P: {P_uniform}")
print(f"Distribution Q: {Q_uniform}")

# Calculate KL divergence
kl_div_uniform = kl_divergence(P_uniform, Q_uniform)
print(f"\nKL divergence D_KL(P||Q) = {kl_div_uniform:.6f}")

# Verify with scipy
kl_div_uniform_scipy = entropy(P_uniform, Q_uniform)
print(f"Verified with scipy: D_KL(P||Q) = {kl_div_uniform_scipy:.6f}")

print("\nInterpretation:")
print("• D_KL(P||Q) is infinite because Q has zero probabilities where P has non-zero probabilities")
print("• This violates the absolute continuity requirement for KL divergence")
print("• It indicates that Q cannot approximate P well in regions where P has mass but Q doesn't")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot distributions
x_pos = np.arange(len(P_uniform))
width = 0.35

ax1.bar(x_pos - width/2, P_uniform, width, label='Distribution P (Uniform)', alpha=0.7, color='blue')
ax1.bar(x_pos + width/2, Q_uniform, width, label='Distribution Q', alpha=0.7, color='red')
ax1.set_xlabel('Outcome')
ax1.set_ylabel('Probability')
ax1.set_title('Uniform Distribution P vs Distribution Q')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(['1', '2', '3', '4'])
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add probability values on bars
for i, (p_val, q_val) in enumerate(zip(P_uniform, Q_uniform)):
    ax1.text(i - width/2, p_val + 0.01, f'{p_val:.2f}', ha='center', va='bottom')
    ax1.text(i + width/2, q_val + 0.01, f'{q_val:.2f}', ha='center', va='bottom')

# Plot log ratios (showing why KL divergence is infinite)
# Handle division by zero more gracefully
log_ratios = np.zeros_like(P_uniform, dtype=float)
for i in range(len(P_uniform)):
    if Q_uniform[i] > 0:
        log_ratios[i] = np.log(P_uniform[i] / Q_uniform[i])
    else:
        log_ratios[i] = np.nan  # Use NaN instead of inf for plotting

ax2.bar(x_pos, log_ratios, color='orange', alpha=0.7)
ax2.set_xlabel('Outcome')
ax2.set_ylabel('log(P[i]/Q[i])')
ax2.set_title('Log Ratios: Why KL Divergence is Infinite')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(['1', '2', '3', '4'])
ax2.grid(True, alpha=0.3)

# Add values on bars
for i, ratio in enumerate(log_ratios):
    if np.isfinite(ratio):
        ax2.text(i, ratio + 0.1, f'{ratio:.2f}', ha='center', va='bottom')
    else:
        ax2.text(i, 0.5, 'inf', ha='center', va='bottom', fontsize=12, fontweight='bold', color='red')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'uniform_kl_divergence.png'), dpi=300, bbox_inches='tight')

print(f"\nLog ratios log(P[i]/Q[i]):")
for i, ratio in enumerate(log_ratios):
    if np.isfinite(ratio):
        print(f"  Outcome {i+1}: log({P_uniform[i]:.2f}/{Q_uniform[i]:.2f}) = {ratio:.2f}")
    else:
        print(f"  Outcome {i+1}: log({P_uniform[i]:.2f}/{Q_uniform[i]:.2f}) = inf (undefined)")

# ============================================================================
# SUMMARY AND KEY INSIGHTS
# ============================================================================
print("\n" + "="*80)
print("SUMMARY AND KEY INSIGHTS")
print("="*80)

print("\n1. Feature Irrelevance:")
print("   • A feature is irrelevant when P(y|x) = P(y) for all x")
print("   • This means X and Y are statistically independent")
print("   • Irrelevant features provide no information for prediction")

print("\n2. Statistical Independence:")
print("   • When P(y|x) = P(y), knowing x doesn't help predict y")
print("   • The joint distribution factors as P(x,y) = P(x)P(y)")
print("   • Mutual information I(X;Y) = 0")

print("\n3. KL Divergence Interpretation:")
print("   • D_KL(P||Q) measures how different distribution Q is from P")
print("   • D_KL = 0 means P and Q are identical")
print("   • Small values indicate Q is a good approximation of P")
print("   • Large values indicate Q is a poor approximation of P")

print("\n4. KL Divergence Properties:")
print("   • D_KL(P||Q) ≥ 0 (always non-negative)")
print("   • D_KL(P||Q) = 0 if and only if P = Q")
print("   • D_KL(P||Q) ≠ D_KL(Q||P) (not symmetric)")
print("   • Infinite when Q has zero probability where P has non-zero probability")

print(f"\nAll plots have been saved to: {save_dir}")
print("=" * 80)
