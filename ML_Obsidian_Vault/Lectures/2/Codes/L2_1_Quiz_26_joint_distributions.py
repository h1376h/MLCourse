import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_1_Quiz_26")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

def save_figure(fig, filename):
    """Save figure to the specified directory."""
    file_path = os.path.join(save_dir, filename)
    fig.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {file_path}")
    plt.close(fig)

# ==============================
# STEP 1: Finding the Constant c
# ==============================
print_step_header(1, "Finding the Constant c")

# Problem statement
print("Joint PMF: P(X=x, Y=y) = c(x+y) for x ∈ {0,1,2} and y ∈ {0,1}")

# Create joint probability table with symbolic c
x_values = [0, 1, 2]
y_values = [0, 1]

print("\nTo find c, we require that the sum of all probabilities equals 1:")
print(f"∑ P(X=x, Y=y) = c(0+0) + c(1+0) + c(2+0) + c(0+1) + c(1+1) + c(2+1) = 1")
print(f"∑ P(X=x, Y=y) = c(0 + 1 + 2 + 1 + 2 + 3) = c*9 = 1")
print(f"Therefore, c = 1/9")

c = 1/9
print(f"\nValue of c = {c}")

# Now create joint probability table with the constant c
joint_pmf_with_c = {}
joint_pmf_sum_with_c = 0

print("\nJoint probabilities with constant c:")
print("x | y | P(X=x, Y=y)")
print("-" * 22)

for x in x_values:
    for y in y_values:
        joint_pmf_with_c[(x, y)] = f"{c}({x+y})"
        joint_pmf_sum_with_c += (x + y)
        print(f"{x} | {y} | {c}({x+y})")

# Create joint PMF table with the actual probabilities
joint_pmf = {}
for x in x_values:
    for y in y_values:
        joint_pmf[(x, y)] = c * (x + y)

print("\nJoint probability mass function:")
print("x | y | P(X=x, Y=y)")
print("-" * 22)
for x in x_values:
    for y in y_values:
        print(f"{x} | {y} | {joint_pmf[(x, y)]:.4f}")

# Verify that probabilities sum to 1
total_prob = sum(joint_pmf.values())
print(f"\nSum of all probabilities: {total_prob:.6f}")
print(f"This confirms that c = 1/9 is correct.")

# Visualization of the joint PMF
fig1 = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 2, width_ratios=[2, 1])

# 3D bar chart
ax1 = fig1.add_subplot(gs[0, 0], projection='3d')
x_pos, y_pos = np.meshgrid(x_values, y_values)
x_pos = x_pos.flatten()
y_pos = y_pos.flatten()
z_pos = np.zeros_like(x_pos)
dx = dy = 0.8
dz = [joint_pmf[(x, y)] for x, y in zip(x_pos, y_pos)]

ax1.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color='skyblue', alpha=0.7)
ax1.set_xlabel('X', fontsize=12)
ax1.set_ylabel('Y', fontsize=12)
ax1.set_zlabel('P(X=x, Y=y)', fontsize=12)
ax1.set_title('Joint PMF P(X=x, Y=y) = c(x+y)', fontsize=14)
ax1.set_xticks(x_values)
ax1.set_yticks(y_values)

# Heat map
ax2 = fig1.add_subplot(gs[0, 1])
heatmap_data = np.zeros((len(y_values), len(x_values)))
for i, y in enumerate(y_values):
    for j, x in enumerate(x_values):
        heatmap_data[i, j] = joint_pmf[(x, y)]
        
sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlGnBu', 
            xticklabels=x_values, yticklabels=y_values, ax=ax2)
ax2.set_title('Joint PMF Heatmap', fontsize=14)
ax2.set_xlabel('X', fontsize=12)
ax2.set_ylabel('Y', fontsize=12)

# Table format
ax3 = fig1.add_subplot(gs[1, :])
table_data = []
for y in y_values:
    row = [f"Y={y}"]
    for x in x_values:
        row.append(f"{joint_pmf[(x, y)]:.4f}")
    table_data.append(row)
column_labels = [""] + [f"X={x}" for x in x_values]
table = ax3.table(cellText=table_data, colLabels=column_labels, loc='center',
                 cellLoc='center', colColours=['#f5f5f5']*len(column_labels))
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.5)
ax3.axis('off')
ax3.set_title('Joint PMF Table Format', fontsize=14)

plt.tight_layout()
save_figure(fig1, "step1_joint_pmf.png")

# ==============================
# STEP 2: Marginal PMFs
# ==============================
print_step_header(2, "Calculating Marginal PMFs")

# Calculate marginal PMFs
marginal_x = {}
for x in x_values:
    marginal_x[x] = sum(joint_pmf[(x, y)] for y in y_values)
    
marginal_y = {}
for y in y_values:
    marginal_y[y] = sum(joint_pmf[(x, y)] for x in x_values)

# Display marginal PMFs
print("Marginal PMF for X:")
print("x | P(X=x)")
print("-" * 12)
for x in x_values:
    print(f"{x} | {marginal_x[x]:.4f}")
    
print("\nMarginal PMF for Y:")
print("y | P(Y=y)")
print("-" * 12)
for y in y_values:
    print(f"{y} | {marginal_y[y]:.4f}")

# Verify that marginal PMFs sum to 1
sum_x = sum(marginal_x.values())
sum_y = sum(marginal_y.values())
print(f"\nSum of P(X=x): {sum_x:.6f}")
print(f"Sum of P(Y=y): {sum_y:.6f}")

# Visualization of the marginal PMFs
fig2 = plt.figure(figsize=(12, 6))

# Marginal PMF of X
ax1 = fig2.add_subplot(121)
ax1.bar(x_values, [marginal_x[x] for x in x_values], color='skyblue', alpha=0.7)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('P(X=x)', fontsize=12)
ax1.set_title('Marginal PMF of X', fontsize=14)
ax1.set_xticks(x_values)
ax1.grid(True)

# Add text labels with probabilities
for x in x_values:
    ax1.text(x, marginal_x[x] + 0.01, f"{marginal_x[x]:.4f}", 
             ha='center', va='bottom', fontsize=10)

# Marginal PMF of Y
ax2 = fig2.add_subplot(122)
ax2.bar(y_values, [marginal_y[y] for y in y_values], color='salmon', alpha=0.7)
ax2.set_xlabel('y', fontsize=12)
ax2.set_ylabel('P(Y=y)', fontsize=12)
ax2.set_title('Marginal PMF of Y', fontsize=14)
ax2.set_xticks(y_values)
ax2.grid(True)

# Add text labels with probabilities
for y in y_values:
    ax2.text(y, marginal_y[y] + 0.01, f"{marginal_y[y]:.4f}", 
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
save_figure(fig2, "step2_marginal_pmfs.png")

# Combined visualization showing marginals and joint
fig3 = plt.figure(figsize=(10, 8))
gs = GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3])

# Marginal PMF of X (top)
ax_top = fig3.add_subplot(gs[0, 0])
ax_top.bar(x_values, [marginal_x[x] for x in x_values], color='skyblue', alpha=0.7)
ax_top.set_title('Marginal PMF of X', fontsize=12)
ax_top.set_xticks([])
ax_top.set_ylim(0, max(marginal_x.values()) * 1.2)
ax_top.grid(True)

# Marginal PMF of Y (right)
ax_right = fig3.add_subplot(gs[1, 1])
ax_right.barh(y_values, [marginal_y[y] for y in y_values], color='salmon', alpha=0.7)
ax_right.set_title('Marginal PMF of Y', fontsize=12)
ax_right.set_yticks([])
ax_right.set_xlim(0, max(marginal_y.values()) * 1.2)
ax_right.grid(True)

# Joint PMF (center)
ax_center = fig3.add_subplot(gs[1, 0])
joint_data = np.zeros((len(y_values), len(x_values)))
for i, y in enumerate(y_values):
    for j, x in enumerate(x_values):
        joint_data[i, j] = joint_pmf[(x, y)]

im = ax_center.imshow(joint_data, cmap='YlGnBu', aspect='auto',
                      extent=[-0.5, 2.5, -0.5, 1.5], origin='lower')
for i, y in enumerate(y_values):
    for j, x in enumerate(x_values):
        ax_center.text(x, y, f"{joint_pmf[(x, y)]:.4f}", 
                      ha='center', va='center', color='black', fontsize=10)
ax_center.set_title('Joint PMF P(X=x, Y=y)', fontsize=14)
ax_center.set_xlabel('X', fontsize=12)
ax_center.set_ylabel('Y', fontsize=12)
ax_center.set_xticks(x_values)
ax_center.set_yticks(y_values)

# Color bar
cbar = plt.colorbar(im, ax=ax_center)
cbar.set_label('Probability')

plt.tight_layout()
save_figure(fig3, "step2_joint_and_marginals.png")

# ==============================
# STEP 3: Testing for Independence
# ==============================
print_step_header(3, "Testing for Independence")

print("To determine if X and Y are independent, we need to check if P(X=x, Y=y) = P(X=x) · P(Y=y) for all x, y")

# Check independence for each combination
independence_check = {}
for x in x_values:
    for y in y_values:
        product = marginal_x[x] * marginal_y[y]
        joint = joint_pmf[(x, y)]
        independence_check[(x, y)] = {
            "P(X=x, Y=y)": joint,
            "P(X=x) · P(Y=y)": product,
            "Equal?": np.isclose(joint, product)
        }

# Print results
print("\nIndependence Check:")
print("x | y | P(X=x, Y=y) | P(X=x) · P(Y=y) | Equal?")
print("-" * 62)
for x in x_values:
    for y in y_values:
        check = independence_check[(x, y)]
        print(f"{x} | {y} | {check['P(X=x, Y=y)']:.6f} | {check['P(X=x) · P(Y=y)']:.6f} | {check['Equal?']}")

# Determine independence
are_independent = all(check["Equal?"] for check in independence_check.values())
print(f"\nAre X and Y independent? {are_independent}")
if not are_independent:
    print("Since P(X=x, Y=y) ≠ P(X=x) · P(Y=y) for at least one combination, X and Y are dependent.")

# Visualization comparing joint and product of marginals
fig4 = plt.figure(figsize=(12, 6))

# Bar chart comparing joint PMF and product of marginals
ax = fig4.add_subplot(111)
bar_width = 0.35
index = np.arange(len(x_values) * len(y_values))

joint_probs = []
product_probs = []
bar_labels = []

for x in x_values:
    for y in y_values:
        joint_probs.append(joint_pmf[(x, y)])
        product_probs.append(marginal_x[x] * marginal_y[y])
        bar_labels.append(f"({x},{y})")

ax.bar(index, joint_probs, bar_width, label='Joint P(X=x, Y=y)', alpha=0.7)
ax.bar(index + bar_width, product_probs, bar_width, label='Product P(X=x)·P(Y=y)', alpha=0.7)

ax.set_xlabel('(x, y) Combinations', fontsize=12)
ax.set_ylabel('Probability', fontsize=12)
ax.set_title('Comparing Joint PMF vs. Product of Marginals', fontsize=14)
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(bar_labels)
ax.legend()
ax.grid(True)

# Add significance markers for differences
for i in range(len(index)):
    if not np.isclose(joint_probs[i], product_probs[i]):
        ax.text(index[i] + bar_width/2, max(joint_probs[i], product_probs[i]) + 0.01,
                "≠", ha='center', va='bottom', fontsize=14, color='red')

plt.tight_layout()
save_figure(fig4, "step3_independence_check.png")

# ==============================
# STEP 4: Conditional PMF
# ==============================
print_step_header(4, "Calculating Conditional PMF P(Y|X=1)")

print("To find P(Y|X=1), we use the formula: P(Y=y|X=1) = P(X=1, Y=y) / P(X=1)")

# Calculate conditional PMF
conditional_y_given_x1 = {}
for y in y_values:
    conditional_y_given_x1[y] = joint_pmf[(1, y)] / marginal_x[1]

# Display conditional PMF
print("\nConditional PMF P(Y|X=1):")
print("y | P(Y=y|X=1)")
print("-" * 18)
for y in y_values:
    print(f"{y} | {conditional_y_given_x1[y]:.6f}")

# Verify that conditional PMF sums to 1
sum_conditional = sum(conditional_y_given_x1.values())
print(f"\nSum of P(Y=y|X=1): {sum_conditional:.6f}")

# Visualization of the conditional PMF
fig5 = plt.figure(figsize=(12, 6))

# Bar chart of conditional PMF
ax1 = fig5.add_subplot(121)
ax1.bar(y_values, [conditional_y_given_x1[y] for y in y_values], color='green', alpha=0.7)
ax1.set_xlabel('y', fontsize=12)
ax1.set_ylabel('P(Y=y|X=1)', fontsize=12)
ax1.set_title('Conditional PMF P(Y|X=1)', fontsize=14)
ax1.set_xticks(y_values)
ax1.grid(True)

# Add text labels with probabilities
for y in y_values:
    ax1.text(y, conditional_y_given_x1[y] + 0.01, f"{conditional_y_given_x1[y]:.4f}", 
             ha='center', va='bottom', fontsize=10)

# Comparison with marginal PMF of Y
ax2 = fig5.add_subplot(122)
bar_width = 0.35
index = np.arange(len(y_values))

ax2.bar(index, [marginal_y[y] for y in y_values], bar_width, 
         label='Marginal P(Y=y)', alpha=0.7, color='salmon')
ax2.bar(index + bar_width, [conditional_y_given_x1[y] for y in y_values], bar_width,
         label='Conditional P(Y=y|X=1)', alpha=0.7, color='green')

ax2.set_xlabel('y', fontsize=12)
ax2.set_ylabel('Probability', fontsize=12)
ax2.set_title('Comparing P(Y) and P(Y|X=1)', fontsize=14)
ax2.set_xticks(index + bar_width / 2)
ax2.set_xticklabels(y_values)
ax2.legend()
ax2.grid(True)

plt.tight_layout()
save_figure(fig5, "step4_conditional_pmf.png")

# ==============================
# STEP 5: Conditional Expectation
# ==============================
print_step_header(5, "Computing E[X|Y=1]")

print("To find E[X|Y=1], we first need the conditional PMF P(X|Y=1).")
print("Then we compute E[X|Y=1] = ∑ x * P(X=x|Y=1) for all x")

# Calculate conditional PMF P(X|Y=1)
conditional_x_given_y1 = {}
for x in x_values:
    conditional_x_given_y1[x] = joint_pmf[(x, 1)] / marginal_y[1]

# Display conditional PMF
print("\nConditional PMF P(X|Y=1):")
print("x | P(X=x|Y=1)")
print("-" * 18)
for x in x_values:
    print(f"{x} | {conditional_x_given_y1[x]:.6f}")

# Calculate conditional expectation
conditional_expectation = sum(x * conditional_x_given_y1[x] for x in x_values)
print(f"\nE[X|Y=1] = ∑ x * P(X=x|Y=1) = {conditional_expectation:.6f}")

# Interpretation
print("\nInterpretation of E[X|Y=1]:")
print(f"When Y=1, the expected value of X is {conditional_expectation:.6f}.")
print("This means that when we observe Y=1, X tends to be closer to this value on average.")

# Visualization of the conditional expectation
fig6 = plt.figure(figsize=(12, 6))

# Bar chart of conditional PMF P(X|Y=1)
ax1 = fig6.add_subplot(121)
ax1.bar(x_values, [conditional_x_given_y1[x] for x in x_values], color='purple', alpha=0.7)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('P(X=x|Y=1)', fontsize=12)
ax1.set_title('Conditional PMF P(X|Y=1)', fontsize=14)
ax1.set_xticks(x_values)
ax1.grid(True)

# Add text labels with probabilities
for x in x_values:
    ax1.text(x, conditional_x_given_y1[x] + 0.01, f"{conditional_x_given_y1[x]:.4f}", 
             ha='center', va='bottom', fontsize=10)

# Visualization of conditional expectation
ax2 = fig6.add_subplot(122)
ax2.bar(x_values, [conditional_x_given_y1[x] for x in x_values], color='purple', alpha=0.7)
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('P(X=x|Y=1)', fontsize=12)
ax2.set_title(f'E[X|Y=1] = {conditional_expectation:.4f}', fontsize=14)
ax2.set_xticks(x_values)
ax2.grid(True)

# Add vertical line for conditional expectation
ax2.axvline(x=conditional_expectation, color='red', linestyle='--', linewidth=2,
            label=f'E[X|Y=1] = {conditional_expectation:.4f}')
ax2.legend()

plt.tight_layout()
save_figure(fig6, "step5_conditional_expectation.png")

# Comparison with the unconditional expectation of X
unconditional_expectation = sum(x * marginal_x[x] for x in x_values)

fig7 = plt.figure(figsize=(10, 6))
ax = fig7.add_subplot(111)

# Prepare data
comparison_data = {
    'x': x_values + x_values,
    'probability': [marginal_x[x] for x in x_values] + [conditional_x_given_y1[x] for x in x_values],
    'type': ['Marginal P(X=x)'] * len(x_values) + ['Conditional P(X=x|Y=1)'] * len(x_values)
}
df = pd.DataFrame(comparison_data)

# Create grouped bar chart
sns.barplot(x='x', y='probability', hue='type', data=df, ax=ax, alpha=0.7)

# Add vertical lines for expectations
ax.axvline(x=-0.2, ymin=0, ymax=1, color='blue', linestyle='--', linewidth=2,
          label=f'E[X] = {unconditional_expectation:.4f}')
ax.axvline(x=2.2, ymin=0, ymax=1, color='red', linestyle='--', linewidth=2,
          label=f'E[X|Y=1] = {conditional_expectation:.4f}')

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Probability', fontsize=12)
ax.set_title('Comparing P(X) and P(X|Y=1) with their Expectations', fontsize=14)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True)

plt.tight_layout()
save_figure(fig7, "step5_expectations_comparison.png")

# ==============================
# SUMMARY
# ==============================
print_step_header(6, "SUMMARY")

print("Key findings from the joint distribution analysis:")
print("\n1. Constant c:")
print(f"   • c = 1/9 = {c:.6f}")

print("\n2. Marginal PMFs:")
print("   • P(X=0) =", marginal_x[0])
print("   • P(X=1) =", marginal_x[1])
print("   • P(X=2) =", marginal_x[2])
print("   • P(Y=0) =", marginal_y[0])
print("   • P(Y=1) =", marginal_y[1])

print("\n3. Independence:")
print(f"   • X and Y are {'independent' if are_independent else 'dependent'}")
print("   • This is determined by checking if P(X=x, Y=y) = P(X=x)·P(Y=y) for all x, y")

print("\n4. Conditional PMF P(Y|X=1):")
print("   • P(Y=0|X=1) =", conditional_y_given_x1[0])
print("   • P(Y=1|X=1) =", conditional_y_given_x1[1])

print("\n5. Conditional Expectation:")
print(f"   • E[X|Y=1] = {conditional_expectation:.6f}")
print(f"   • (Compared to unconditional E[X] = {unconditional_expectation:.6f})")
print("   • This demonstrates how conditioning on Y=1 shifts our expectation of X")

print("\nThis analysis illustrates important concepts in joint probability distributions,")
print("including marginal and conditional distributions, independence testing, and conditional expectations.") 