import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
import math

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_2_Quiz_7")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Given data for the problem
P = {"y=1": 0.7, "y=0": 0.3}  # True distribution
Q = {"y=1": 0.6, "y=0": 0.4}  # Predicted distribution

# Step 1: Define the problem
print_step_header(1, "Understanding the Problem")

print("Given:")
print(f"- True distribution P: P(y=1) = {P['y=1']}, P(y=0) = {P['y=0']}")
print(f"- Predicted distribution Q: Q(y=1) = {Q['y=1']}, Q(y=0) = {Q['y=0']}")
print()
print("We need to calculate:")
print("1. Cross-entropy H(P, Q)")
print("2. Entropy H(P)")
print("3. KL Divergence D_KL(P||Q)")
print("4. Verify that H(P, Q) = H(P) + D_KL(P||Q)")
print()

# Step 2: Calculate Cross-entropy H(P, Q)
print_step_header(2, "Calculating Cross-entropy H(P, Q)")

print("The cross-entropy between distributions P and Q is defined as:")
print("H(P, Q) = -∑ P(x) log Q(x)")
print()
print("For our binary classification problem:")
print("H(P, Q) = -[P(y=1) log Q(y=1) + P(y=0) log Q(y=0)]")
print()

# Calculate cross-entropy
h_p_q = -1 * (P["y=1"] * math.log2(Q["y=1"]) + P["y=0"] * math.log2(Q["y=0"]))

print(f"H(P, Q) = -[{P['y=1']} × log₂({Q['y=1']}) + {P['y=0']} × log₂({Q['y=0']})]")
print(f"H(P, Q) = -[{P['y=1']} × {math.log2(Q['y=1']):.6f} + {P['y=0']} × {math.log2(Q['y=0']):.6f}]")
print(f"H(P, Q) = -[{P['y=1'] * math.log2(Q['y=1']):.6f} + {P['y=0'] * math.log2(Q['y=0']):.6f}]")
print(f"H(P, Q) = -{(P['y=1'] * math.log2(Q['y=1']) + P['y=0'] * math.log2(Q['y=0'])):.6f}")
print(f"H(P, Q) = {h_p_q:.6f}")
print()

# Step 3: Calculate Entropy H(P)
print_step_header(3, "Calculating Entropy H(P)")

print("The entropy of distribution P is defined as:")
print("H(P) = -∑ P(x) log P(x)")
print()
print("For our binary classification problem:")
print("H(P) = -[P(y=1) log P(y=1) + P(y=0) log P(y=0)]")
print()

# Calculate entropy
h_p = -1 * (P["y=1"] * math.log2(P["y=1"]) + P["y=0"] * math.log2(P["y=0"]))

print(f"H(P) = -[{P['y=1']} × log₂({P['y=1']}) + {P['y=0']} × log₂({P['y=0']})]")
print(f"H(P) = -[{P['y=1']} × {math.log2(P['y=1']):.6f} + {P['y=0']} × {math.log2(P['y=0']):.6f}]")
print(f"H(P) = -[{P['y=1'] * math.log2(P['y=1']):.6f} + {P['y=0'] * math.log2(P['y=0']):.6f}]")
print(f"H(P) = -{(P['y=1'] * math.log2(P['y=1']) + P['y=0'] * math.log2(P['y=0'])):.6f}")
print(f"H(P) = {h_p:.6f}")
print()

# Step 4: Calculate KL Divergence D_KL(P||Q)
print_step_header(4, "Calculating KL Divergence D_KL(P||Q)")

print("The KL Divergence between distributions P and Q is defined as:")
print("D_KL(P||Q) = ∑ P(x) log(P(x)/Q(x))")
print()
print("For our binary classification problem:")
print("D_KL(P||Q) = P(y=1) log(P(y=1)/Q(y=1)) + P(y=0) log(P(y=0)/Q(y=0))")
print()

# Calculate KL divergence
d_kl = P["y=1"] * math.log2(P["y=1"]/Q["y=1"]) + P["y=0"] * math.log2(P["y=0"]/Q["y=0"])

print(f"D_KL(P||Q) = {P['y=1']} × log₂({P['y=1']}/{Q['y=1']}) + {P['y=0']} × log₂({P['y=0']}/{Q['y=0']})")
print(f"D_KL(P||Q) = {P['y=1']} × log₂({P['y=1']/Q['y=1']}) + {P['y=0']} × log₂({P['y=0']/Q['y=0']})")
print(f"D_KL(P||Q) = {P['y=1']} × {math.log2(P['y=1']/Q['y=1']):.6f} + {P['y=0']} × {math.log2(P['y=0']/Q['y=0']):.6f}")
print(f"D_KL(P||Q) = {P['y=1'] * math.log2(P['y=1']/Q['y=1']):.6f} + {P['y=0'] * math.log2(P['y=0']/Q['y=0']):.6f}")
print(f"D_KL(P||Q) = {d_kl:.6f}")
print()

# Step 5: Verify the relationship
print_step_header(5, "Verifying H(P, Q) = H(P) + D_KL(P||Q)")

print(f"We have calculated:")
print(f"H(P, Q) = {h_p_q:.6f}")
print(f"H(P) = {h_p:.6f}")
print(f"D_KL(P||Q) = {d_kl:.6f}")
print()
print(f"To verify the relationship, we calculate H(P) + D_KL(P||Q):")
print(f"H(P) + D_KL(P||Q) = {h_p:.6f} + {d_kl:.6f} = {h_p + d_kl:.6f}")
print()

# Check if the relationship holds
relationship_check = abs(h_p_q - (h_p + d_kl)) < 1e-10
print(f"Is H(P, Q) = H(P) + D_KL(P||Q)? {'Yes' if relationship_check else 'No'}")
print(f"Difference: {abs(h_p_q - (h_p + d_kl)):.10f}")
print()

# Step 6: Visualize the distributions
print_step_header(6, "Visualizing the Distributions")

# Create a bar chart comparing P and Q
plt.figure(figsize=(10, 6))
x = ['y=0', 'y=1']
plt.bar([0, 2], [P['y=0'], P['y=1']], width=0.4, label='True Distribution (P)', color='blue', alpha=0.7)
plt.bar([0.5, 2.5], [Q['y=0'], Q['y=1']], width=0.4, label='Predicted Distribution (Q)', color='orange', alpha=0.7)

# Add text labels
plt.text(0, P['y=0'] + 0.01, f"{P['y=0']}", ha='center', va='bottom', fontsize=12)
plt.text(2, P['y=1'] + 0.01, f"{P['y=1']}", ha='center', va='bottom', fontsize=12)
plt.text(0.5, Q['y=0'] + 0.01, f"{Q['y=0']}", ha='center', va='bottom', fontsize=12)
plt.text(2.5, Q['y=1'] + 0.01, f"{Q['y=1']}", ha='center', va='bottom', fontsize=12)

plt.xticks([0.25, 2.25], x)
plt.ylim(0, 1)
plt.title('Comparison of True (P) and Predicted (Q) Distributions', fontsize=14)
plt.ylabel('Probability', fontsize=12)
plt.legend(loc='upper right')
plt.grid(axis='y', alpha=0.3)

# Save the figure
file_path = os.path.join(save_dir, "distributions_comparison.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 7: Visualize the calculated metrics
print_step_header(7, "Visualizing the Information Theory Metrics")

plt.figure(figsize=(12, 7))
gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[2, 1])

# Upper left: Visual representation of entropy H(P)
ax1 = plt.subplot(gs[0, 0])
labels = ['y=0', 'y=1']
sizes = [P['y=0'], P['y=1']]
explode = (0.1, 0)
colors = ['#ff9999', '#66b3ff']
ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
ax1.set_title(f'True Distribution P\nEntropy H(P) = {h_p:.4f}')

# Upper right: Visual representation of cross-entropy H(P, Q)
ax2 = plt.subplot(gs[0, 1])
x = np.arange(2)
width = 0.35
ax2.bar(x - width/2, [P['y=0'], P['y=1']], width, label='P', color='blue', alpha=0.7)
ax2.bar(x + width/2, [Q['y=0'], Q['y=1']], width, label='Q', color='red', alpha=0.7)
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_title(f'Cross-Entropy H(P, Q) = {h_p_q:.4f}')
ax2.legend()

# Bottom: Relationship between metrics
ax3 = plt.subplot(gs[1, :])
metrics = ['H(P)', 'D_KL(P||Q)', 'H(P) + D_KL(P||Q)', 'H(P, Q)']
values = [h_p, d_kl, h_p + d_kl, h_p_q]
ax3.bar(metrics, values, color=['blue', 'green', 'purple', 'red'])
for i, v in enumerate(values):
    ax3.text(i, v + 0.01, f"{v:.4f}", ha='center', va='bottom', fontsize=10)
ax3.set_title('Relationship: H(P, Q) = H(P) + D_KL(P||Q)')
ax3.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.subplots_adjust(hspace=0.4)

# Save the figure
file_path = os.path.join(save_dir, "metrics_visualization.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 8: Visualize the KL divergence interpretation
print_step_header(8, "KL Divergence Interpretation")

plt.figure(figsize=(10, 6))
# Plot for binary values (0 and 1)
x = np.array([0, 1])
p_values = np.array([P['y=0'], P['y=1']])
q_values = np.array([Q['y=0'], Q['y=1']])

# Calculate contribution of each term to KL divergence
kl_terms = [P['y=0'] * math.log2(P['y=0']/Q['y=0']), 
            P['y=1'] * math.log2(P['y=1']/Q['y=1'])]

bar_positions = np.array([0, 1])
plt.bar(bar_positions, kl_terms, color=['#ff9999', '#66b3ff'], 
        alpha=0.8, label='KL Divergence Terms')

for i, v in enumerate(kl_terms):
    plt.text(i, v + 0.001, f"{v:.4f}", ha='center', va='bottom', fontsize=12)

plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.xticks([0, 1], ['y=0', 'y=1'])
plt.title(f'Contribution to KL Divergence D_KL(P||Q) = {d_kl:.4f}', fontsize=14)
plt.ylabel('P(x) log(P(x)/Q(x))', fontsize=12)
plt.grid(axis='y', alpha=0.3)

# Add annotations explaining KL divergence
plt.annotate('KL Divergence measures how much information is lost\nwhen Q is used to approximate P',
             xy=(0.5, max(kl_terms) * 0.5), xytext=(0.5, max(kl_terms) * 0.7),
             ha='center', va='center', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.3))

# Save the figure
file_path = os.path.join(save_dir, "kl_divergence_interpretation.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 9: Summary and conclusion
print_step_header(9, "Summary and Conclusion")

print("Summary of our calculations:")
print(f"1. Cross-entropy H(P, Q) = {h_p_q:.6f}")
print(f"2. Entropy H(P) = {h_p:.6f}")
print(f"3. KL Divergence D_KL(P||Q) = {d_kl:.6f}")
print(f"4. H(P) + D_KL(P||Q) = {h_p + d_kl:.6f}")
print()
print("We have verified that H(P, Q) = H(P) + D_KL(P||Q)")
print()
print("Key insights:")
print("- Cross-entropy H(P, Q) measures the average number of bits needed if we use distribution Q")
print("  to encode events from distribution P")
print("- Entropy H(P) is the theoretical minimum average number of bits needed to encode events from P")
print("- KL Divergence D_KL(P||Q) quantifies the information lost when using Q to approximate P")
print("- The relationship H(P, Q) = H(P) + D_KL(P||Q) shows that cross-entropy equals the true entropy")
print("  plus the extra bits needed due to using the wrong distribution") 