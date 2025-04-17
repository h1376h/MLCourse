import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
import math

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_2_Quiz_2")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Set up the problem
print_step_header(1, "Understanding the Problem")

print("Given:")
print("- Distribution P: P(X = 0) = 0.7, P(X = 1) = 0.3")
print("- Distribution Q: Q(X = 0) = 0.5, Q(X = 1) = 0.5")
print()
print("Tasks:")
print("1. Calculate the KL divergence D_KL(P||Q)")
print("2. Calculate the KL divergence D_KL(Q||P)")
print("3. Calculate the cross-entropy H(P, Q)")
print("4. Explain why D_KL(P||Q) ≠ D_KL(Q||P)")
print()

# Step 2: Define the distributions
print_step_header(2, "Defining the Distributions")

# Define the distributions
P = np.array([0.7, 0.3])  # P(X = 0) = 0.7, P(X = 1) = 0.3
Q = np.array([0.5, 0.5])  # Q(X = 0) = 0.5, Q(X = 1) = 0.5

print("Distribution P:")
print(f"P(X = 0) = {P[0]}, P(X = 1) = {P[1]}")
print("\nDistribution Q:")
print(f"Q(X = 0) = {Q[0]}, Q(X = 1) = {Q[1]}")

# Visualize the distributions
plt.figure(figsize=(10, 6))
x = np.array([0, 1])
width = 0.35
plt.bar(x - width/2, P, width, label='P', color='blue', alpha=0.7)
plt.bar(x + width/2, Q, width, label='Q', color='red', alpha=0.7)
plt.xticks(x, ['X = 0', 'X = 1'])
plt.ylim(0, 1)
plt.ylabel('Probability')
plt.title('Distributions P and Q')
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()

file_path = os.path.join(save_dir, "distributions.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 3: Calculate KL divergence D_KL(P||Q)
print_step_header(3, "Calculating KL Divergence D_KL(P||Q)")

# KL divergence D_KL(P||Q) = sum(P(x) * log(P(x)/Q(x)))
def kl_divergence(p, q):
    """Calculate KL divergence between distributions p and q."""
    # Handle zeros in calculations
    result = 0
    for p_i, q_i in zip(p, q):
        if p_i > 0:  # Only consider non-zero elements of P
            if q_i > 0:
                result += p_i * np.log2(p_i / q_i)
            else:
                # If q_i is 0, KL divergence is infinity
                return float('inf')
    return result

kl_pq = kl_divergence(P, Q)
print(f"D_KL(P||Q) = {kl_pq:.6f} bits")

# Show the calculation steps
print("\nCalculation steps:")
for i, (p_i, q_i) in enumerate(zip(P, Q)):
    if p_i > 0:
        term = p_i * np.log2(p_i/q_i)
        print(f"- P(X = {i}) * log2(P(X = {i})/Q(X = {i})) = {p_i} * log2({p_i}/{q_i}) = {term:.6f}")

print(f"\nSum of all terms = {kl_pq:.6f} bits")

# Visualize the KL divergence calculation
plt.figure(figsize=(10, 6))
bar_width = 0.4
x = np.array([0, 1])

# Calculate individual terms
terms = np.zeros_like(P)
for i in range(len(P)):
    if P[i] > 0:
        terms[i] = P[i] * np.log2(P[i] / Q[i])

plt.bar(x, terms, bar_width, color=['blue', 'green'])
plt.xticks(x, ['X = 0', 'X = 1'])
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.grid(True, axis='y')
plt.title('Terms in D_KL(P||Q) Calculation')
plt.ylabel('P(x) * log2(P(x)/Q(x))')

# Add text annotations
for i, term in enumerate(terms):
    plt.text(i, term + 0.01, f'{term:.6f}', ha='center')

plt.tight_layout()
file_path = os.path.join(save_dir, "kl_pq_terms.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 4: Calculate KL divergence D_KL(Q||P)
print_step_header(4, "Calculating KL Divergence D_KL(Q||P)")

kl_qp = kl_divergence(Q, P)
print(f"D_KL(Q||P) = {kl_qp:.6f} bits")

# Show the calculation steps
print("\nCalculation steps:")
for i, (q_i, p_i) in enumerate(zip(Q, P)):
    if q_i > 0:
        term = q_i * np.log2(q_i/p_i)
        print(f"- Q(X = {i}) * log2(Q(X = {i})/P(X = {i})) = {q_i} * log2({q_i}/{p_i}) = {term:.6f}")

print(f"\nSum of all terms = {kl_qp:.6f} bits")

# Visualize the KL divergence calculation
plt.figure(figsize=(10, 6))
bar_width = 0.4
x = np.array([0, 1])

# Calculate individual terms
terms = np.zeros_like(Q)
for i in range(len(Q)):
    if Q[i] > 0:
        terms[i] = Q[i] * np.log2(Q[i] / P[i])

plt.bar(x, terms, bar_width, color=['red', 'orange'])
plt.xticks(x, ['X = 0', 'X = 1'])
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.grid(True, axis='y')
plt.title('Terms in D_KL(Q||P) Calculation')
plt.ylabel('Q(x) * log2(Q(x)/P(x))')

# Add text annotations
for i, term in enumerate(terms):
    plt.text(i, term + 0.01, f'{term:.6f}', ha='center')

plt.tight_layout()
file_path = os.path.join(save_dir, "kl_qp_terms.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 5: Compare D_KL(P||Q) and D_KL(Q||P)
print_step_header(5, "Comparing D_KL(P||Q) and D_KL(Q||P)")

print(f"D_KL(P||Q) = {kl_pq:.6f} bits")
print(f"D_KL(Q||P) = {kl_qp:.6f} bits")
print(f"Difference: {abs(kl_pq - kl_qp):.6f} bits")

if kl_pq != kl_qp:
    print("\nObservation: D_KL(P||Q) ≠ D_KL(Q||P), showing that KL divergence is not symmetric.")
else:
    print("\nObservation: D_KL(P||Q) = D_KL(Q||P) for this specific case.")

# Visualize the comparison
plt.figure(figsize=(8, 6))
plt.bar(['D_KL(P||Q)', 'D_KL(Q||P)'], [kl_pq, kl_qp], color=['blue', 'red'], alpha=0.7)
plt.grid(True, axis='y')
plt.title('Comparison of KL Divergences')
plt.ylabel('Value (bits)')

# Add text annotations
plt.text(0, kl_pq + 0.01, f'{kl_pq:.6f}', ha='center')
plt.text(1, kl_qp + 0.01, f'{kl_qp:.6f}', ha='center')

plt.tight_layout()
file_path = os.path.join(save_dir, "kl_comparison.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 6: Calculate Cross-Entropy H(P, Q)
print_step_header(6, "Calculating Cross-Entropy H(P, Q)")

def cross_entropy(p, q):
    """Calculate cross-entropy between distributions p and q."""
    result = 0
    for p_i, q_i in zip(p, q):
        if p_i > 0:  # Only consider non-zero elements of P
            if q_i > 0:
                result += p_i * (-np.log2(q_i))
            else:
                # If q_i is 0, cross-entropy is infinity
                return float('inf')
    return result

# Calculate entropy of P
def entropy(p):
    """Calculate entropy of distribution p."""
    result = 0
    for p_i in p:
        if p_i > 0:  # Only consider non-zero elements
            result += p_i * (-np.log2(p_i))
    return result

h_p = entropy(P)
h_p_q = cross_entropy(P, Q)
    
print(f"Entropy of P: H(P) = {h_p:.6f} bits")
print(f"Cross-Entropy: H(P, Q) = {h_p_q:.6f} bits")

# Show the calculation steps
print("\nCalculation steps for Cross-Entropy H(P, Q):")
for i, (p_i, q_i) in enumerate(zip(P, Q)):
    if p_i > 0:
        term = p_i * (-np.log2(q_i))
        print(f"- P(X = {i}) * (-log2(Q(X = {i}))) = {p_i} * (-log2({q_i})) = {term:.6f}")

print(f"\nSum of all terms = {h_p_q:.6f} bits")

# Visualize the cross-entropy calculation
plt.figure(figsize=(10, 6))
bar_width = 0.4
x = np.array([0, 1])

# Calculate individual terms
cross_entropy_terms = np.zeros_like(P)
for i in range(len(P)):
    if P[i] > 0:
        cross_entropy_terms[i] = P[i] * (-np.log2(Q[i]))

plt.bar(x, cross_entropy_terms, bar_width, color=['purple', 'magenta'])
plt.xticks(x, ['X = 0', 'X = 1'])
plt.grid(True, axis='y')
plt.title('Terms in Cross-Entropy H(P, Q) Calculation')
plt.ylabel('P(x) * (-log2(Q(x)))')

# Add text annotations
for i, term in enumerate(cross_entropy_terms):
    plt.text(i, term + 0.01, f'{term:.6f}', ha='center')

plt.tight_layout()
file_path = os.path.join(save_dir, "cross_entropy_terms.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 7: Verify that H(P, Q) = H(P) + D_KL(P||Q)
print_step_header(7, "Verifying H(P, Q) = H(P) + D_KL(P||Q)")

sum_h_p_kl_pq = h_p + kl_pq
print(f"H(P) = {h_p:.6f} bits")
print(f"D_KL(P||Q) = {kl_pq:.6f} bits")
print(f"H(P) + D_KL(P||Q) = {sum_h_p_kl_pq:.6f} bits")
print(f"H(P, Q) = {h_p_q:.6f} bits")

if np.isclose(sum_h_p_kl_pq, h_p_q):
    print("\nVerification: H(P, Q) = H(P) + D_KL(P||Q) ✓")
else:
    print("\nVerification: H(P, Q) ≠ H(P) + D_KL(P||Q) ✗")

# Visualize the relationship
plt.figure(figsize=(10, 6))
bar_width = 0.3
x = np.array([0, 1, 2])

bars = plt.bar(x, [h_p, kl_pq, h_p_q], bar_width, 
               color=['green', 'blue', 'purple'],
               tick_label=['H(P)', 'D_KL(P||Q)', 'H(P, Q)'])

# Add a bar for the sum
plt.bar(3, sum_h_p_kl_pq, bar_width, color='red', 
        tick_label=['H(P) + D_KL(P||Q)'])

# Draw a horizontal line at H(P, Q) for comparison
plt.axhline(y=h_p_q, color='purple', linestyle='--', alpha=0.7)

plt.grid(True, axis='y')
plt.title('Relationship: H(P, Q) = H(P) + D_KL(P||Q)')
plt.ylabel('Value (bits)')

# Add text annotations
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.6f}', ha='center')
plt.text(3, sum_h_p_kl_pq + 0.01, f'{sum_h_p_kl_pq:.6f}', ha='center')

plt.tight_layout()
file_path = os.path.join(save_dir, "relationship.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 8: Create a comprehensive visualization showing all components
print_step_header(8, "Comprehensive Visualization")

# Create a 3D figure showing the relationship between distributions and divergences
fig = plt.figure(figsize=(12, 10))
gs = GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], hspace=0.3, wspace=0.3)

# Plot 1: Distributions
ax1 = fig.add_subplot(gs[0, 0])
x = np.array([0, 1])
width = 0.35
ax1.bar(x - width/2, P, width, label='P', color='blue', alpha=0.7)
ax1.bar(x + width/2, Q, width, label='Q', color='red', alpha=0.7)
ax1.set_xticks(x)
ax1.set_xticklabels(['X = 0', 'X = 1'])
ax1.set_ylim(0, 1)
ax1.set_ylabel('Probability')
ax1.set_title('Distributions P and Q')
ax1.legend()
ax1.grid(True, axis='y')

# Plot 2: KL Divergences
ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(['D_KL(P||Q)', 'D_KL(Q||P)'], [kl_pq, kl_qp], color=['blue', 'red'], alpha=0.7)
ax2.grid(True, axis='y')
ax2.set_title('KL Divergences')
ax2.set_ylabel('Value (bits)')
ax2.text(0, kl_pq + 0.01, f'{kl_pq:.4f}', ha='center')
ax2.text(1, kl_qp + 0.01, f'{kl_qp:.4f}', ha='center')

# Plot 3: Cross-Entropy Components
ax3 = fig.add_subplot(gs[1, 0])
components = [h_p, kl_pq, h_p_q]
ax3.bar([0, 1, 2], components, color=['green', 'blue', 'purple'], 
        tick_label=['H(P)', 'D_KL(P||Q)', 'H(P, Q)'])
ax3.grid(True, axis='y')
ax3.set_title('Cross-Entropy Decomposition')
ax3.set_ylabel('Value (bits)')
for i, val in enumerate(components):
    ax3.text(i, val + 0.01, f'{val:.4f}', ha='center')

# Plot 4: Interpretation and visual explanation
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')  # Turn off axis
ax4.text(0.5, 0.95, "Key Insights", ha='center', va='top', fontweight='bold', fontsize=12)
ax4.text(0.5, 0.85, "1. KL Divergence is asymmetric:", ha='center', va='top', fontsize=10)
ax4.text(0.5, 0.80, f"   D_KL(P||Q) = {kl_pq:.4f} ≠ D_KL(Q||P) = {kl_qp:.4f}", ha='center', va='top', fontsize=10)
ax4.text(0.5, 0.70, "2. Cross-entropy can be decomposed:", ha='center', va='top', fontsize=10)
ax4.text(0.5, 0.65, f"   H(P, Q) = H(P) + D_KL(P||Q)", ha='center', va='top', fontsize=10)
ax4.text(0.5, 0.60, f"   {h_p_q:.4f} = {h_p:.4f} + {kl_pq:.4f}", ha='center', va='top', fontsize=10)
ax4.text(0.5, 0.50, "3. Interpretations:", ha='center', va='top', fontsize=10)
ax4.text(0.5, 0.45, "• KL divergence measures how P differs from Q", ha='center', va='top', fontsize=10)
ax4.text(0.5, 0.40, "• D_KL(P||Q) is the extra bits needed when", ha='center', va='top', fontsize=10)
ax4.text(0.5, 0.35, "  using Q to encode samples from P", ha='center', va='top', fontsize=10)
ax4.text(0.5, 0.25, "• The asymmetry reveals that the cost of", ha='center', va='top', fontsize=10)
ax4.text(0.5, 0.20, "  a wrong model depends on which way the", ha='center', va='top', fontsize=10)
ax4.text(0.5, 0.15, "  approximation is going", ha='center', va='top', fontsize=10)

plt.tight_layout()
file_path = os.path.join(save_dir, "comprehensive_visualization.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 9: Conclusion and Answer Summary
print_step_header(9, "Conclusion and Answer Summary")

print("Question 2 Solution Summary:")
print("1. KL Divergence D_KL(P||Q) = %.6f bits" % kl_pq)
print("2. KL Divergence D_KL(Q||P) = %.6f bits" % kl_qp)
print("3. Cross-Entropy H(P, Q) = %.6f bits" % h_p_q)
print("4. D_KL(P||Q) ≠ D_KL(Q||P) because KL divergence is asymmetric. It measures")
print("   the extra bits needed when using distribution Q to encode samples from P.")
print("   This asymmetry is important in practice because it means that the 'cost'")
print("   of approximating one distribution with another depends on which way the")
print("   approximation is going.")
print("\nBonus: We verified that H(P, Q) = H(P) + D_KL(P||Q)")
print(f"  {h_p_q:.6f} = {h_p:.6f} + {kl_pq:.6f}") 