import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from matplotlib_venn import venn2, venn3

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_1_Quiz_36")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Define the problem
print_step_header(1, "Problem Statement")
print("Problem: Prove that P(X,Y|Z) = P(X|Z)P(Y|Z) if P(X|Y,Z) = P(X|Z)")
print("\nThis is a proof of a property of conditional independence.")
print("When X and Y are conditionally independent given Z, we can factor the joint probability.")

# Step 2: Recall the definitions
print_step_header(2, "Basic Definitions")
print("Let's recall some basic definitions from probability theory:")
print("1. Conditional probability: P(A|B) = P(A,B)/P(B)")
print("2. Joint probability: P(A,B|C) = P(A,B,C)/P(C)")
print("3. Chain rule: P(A,B) = P(A|B)P(B) = P(B|A)P(A)")

# Step 3: Start the proof
print_step_header(3, "Starting the Proof")
print("Given: P(X|Y,Z) = P(X|Z)")
print("\nWe need to prove: P(X,Y|Z) = P(X|Z)P(Y|Z)")
print("\nLet's start from the left side: P(X,Y|Z)")

# Step 4: Apply definition of conditional probability
print_step_header(4, "Applying Definition of Conditional Probability")
print("By definition of conditional probability:")
print("P(X,Y|Z) = P(X,Y,Z)/P(Z)")
print("\nUsing the chain rule, we can rewrite P(X,Y,Z) as:")
print("P(X,Y,Z) = P(X|Y,Z)P(Y,Z)")
print("\nSubstituting this into our expression:")
print("P(X,Y|Z) = P(X|Y,Z)P(Y,Z)/P(Z)")

# Step 5: Use the given condition
print_step_header(5, "Using the Given Condition")
print("Given that P(X|Y,Z) = P(X|Z), we substitute:")
print("P(X,Y|Z) = P(X|Z)P(Y,Z)/P(Z)")
print("\nNow we can further simplify using the definition of conditional probability:")
print("P(Y,Z)/P(Z) = P(Y|Z)")
print("\nTherefore:")
print("P(X,Y|Z) = P(X|Z)P(Y|Z)")
print("\nThis completes the proof, showing that if P(X|Y,Z) = P(X|Z), then P(X,Y|Z) = P(X|Z)P(Y|Z).")

# Step 6: Visualize conditional independence with Venn diagrams
print_step_header(6, "Visualizing Conditional Independence with Venn Diagrams")

# Create a figure showing the concept of conditional independence
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# First subplot - Conditional dependence
ax1.set_title("Conditional Dependence: P(X|Y,Z) ≠ P(X|Z)", fontsize=14)

# Draw three overlapping circles
circle1 = plt.Circle((0.3, 0.6), 0.25, fc='red', alpha=0.5, label='X')
circle2 = plt.Circle((0.5, 0.4), 0.25, fc='blue', alpha=0.5, label='Y')
circle3 = plt.Circle((0.7, 0.6), 0.25, fc='green', alpha=0.5, label='Z')

ax1.add_patch(circle1)
ax1.add_patch(circle2)
ax1.add_patch(circle3)

# Add labels
ax1.text(0.3, 0.6, "X", fontsize=20, ha='center', va='center')
ax1.text(0.5, 0.4, "Y", fontsize=20, ha='center', va='center')
ax1.text(0.7, 0.6, "Z", fontsize=20, ha='center', va='center')

ax1.text(0.5, 0.1, "When X and Y share information beyond Z\nP(X,Y|Z) ≠ P(X|Z)P(Y|Z)", fontsize=12, ha='center')

# Second subplot - Conditional independence
ax2.set_title("Conditional Independence: P(X|Y,Z) = P(X|Z)", fontsize=14)

# Draw the circles with Z separating X and Y
circle1 = plt.Circle((0.3, 0.5), 0.25, fc='red', alpha=0.5, label='X')
circle3 = plt.Circle((0.5, 0.5), 0.25, fc='green', alpha=0.5, label='Z')
circle2 = plt.Circle((0.7, 0.5), 0.25, fc='blue', alpha=0.5, label='Y')

ax2.add_patch(circle1)
ax2.add_patch(circle3)
ax2.add_patch(circle2)

# Add labels
ax2.text(0.3, 0.5, "X", fontsize=20, ha='center', va='center')
ax2.text(0.5, 0.5, "Z", fontsize=20, ha='center', va='center')
ax2.text(0.7, 0.5, "Y", fontsize=20, ha='center', va='center')

ax2.text(0.5, 0.1, "When Z separates X and Y\nP(X,Y|Z) = P(X|Z)P(Y|Z)", fontsize=12, ha='center')

for ax in (ax1, ax2):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

plt.tight_layout()

# Save the visualization
file_path = os.path.join(save_dir, "conditional_independence_venn.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Venn diagram visualization saved to: {file_path}")

# Step 7: Create a visualization of Bayesian networks
print_step_header(7, "Visualizing Conditional Independence with Bayesian Networks")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# First subnet - Conditional dependence
ax1.set_title("Conditional Dependence", fontsize=14)

# Draw nodes
ax1.add_patch(patches.Circle((0.3, 0.6), 0.1, fc='red', ec='black'))
ax1.add_patch(patches.Circle((0.7, 0.6), 0.1, fc='blue', ec='black'))
ax1.add_patch(patches.Circle((0.5, 0.2), 0.1, fc='green', ec='black'))

# Add labels
ax1.text(0.3, 0.6, "X", fontsize=14, ha='center', va='center')
ax1.text(0.7, 0.6, "Y", fontsize=14, ha='center', va='center')
ax1.text(0.5, 0.2, "Z", fontsize=14, ha='center', va='center')

# Draw arrows
ax1.arrow(0.5, 0.28, -0.12, 0.24, head_width=0.03, head_length=0.05, fc='black', ec='black')
ax1.arrow(0.5, 0.28, 0.12, 0.24, head_width=0.03, head_length=0.05, fc='black', ec='black')
ax1.arrow(0.39, 0.6, 0.22, 0, head_width=0.03, head_length=0.05, fc='black', ec='black')

ax1.text(0.5, 0.8, "X and Y are connected\n(not conditionally independent given Z)", fontsize=10, ha='center')

# Second subnet - Conditional independence
ax2.set_title("Conditional Independence", fontsize=14)

# Draw nodes
ax2.add_patch(patches.Circle((0.3, 0.6), 0.1, fc='red', ec='black'))
ax2.add_patch(patches.Circle((0.7, 0.6), 0.1, fc='blue', ec='black'))
ax2.add_patch(patches.Circle((0.5, 0.2), 0.1, fc='green', ec='black'))

# Add labels
ax2.text(0.3, 0.6, "X", fontsize=14, ha='center', va='center')
ax2.text(0.7, 0.6, "Y", fontsize=14, ha='center', va='center')
ax2.text(0.5, 0.2, "Z", fontsize=14, ha='center', va='center')

# Draw arrows
ax2.arrow(0.5, 0.28, -0.12, 0.24, head_width=0.03, head_length=0.05, fc='black', ec='black')
ax2.arrow(0.5, 0.28, 0.12, 0.24, head_width=0.03, head_length=0.05, fc='black', ec='black')

ax2.text(0.5, 0.8, "X and Y are separated by Z\n(conditionally independent given Z)", fontsize=10, ha='center')

for ax in (ax1, ax2):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

plt.tight_layout()

# Save the visualization
file_path = os.path.join(save_dir, "conditional_independence_bayes_net.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Bayesian network visualization saved to: {file_path}")

# Step 8: Create a numeric example with a simple joint probability distribution
print_step_header(8, "Numerical Example")

# Create a simple joint probability distribution
print("Let's create a numerical example with a simple joint probability distribution")
print("Consider binary random variables X, Y, and Z with the following joint distribution:")

# Define a joint distribution where X and Y are conditionally independent given Z
# P(X,Y,Z) for all combinations of X, Y, Z
# Using values where P(X|Y,Z) = P(X|Z) is satisfied

# Format: [x, y, z, p(x,y,z)]
joint_dist = [
    [0, 0, 0, 0.12],
    [0, 1, 0, 0.18],
    [1, 0, 0, 0.28],
    [1, 1, 0, 0.42],
    [0, 0, 1, 0.24],
    [0, 1, 1, 0.06],
    [1, 0, 1, 0.16],
    [1, 1, 1, 0.04]
]

# Display the joint distribution as a table
print("\nJoint Probability Distribution P(X,Y,Z):")
print("-" * 40)
print("| X | Y | Z | P(X,Y,Z) |")
print("-" * 40)
for x, y, z, p in joint_dist:
    print(f"| {x} | {y} | {z} |  {p:.2f}    |")
print("-" * 40)

# Calculate P(Z)
p_z0 = sum(p for x, y, z, p in joint_dist if z == 0)
p_z1 = sum(p for x, y, z, p in joint_dist if z == 1)

print(f"\nP(Z=0) = {p_z0:.2f}")
print(f"P(Z=1) = {p_z1:.2f}")

# Calculate P(X|Z)
p_x0_z0 = sum(p for x, y, z, p in joint_dist if x == 0 and z == 0) / p_z0
p_x1_z0 = sum(p for x, y, z, p in joint_dist if x == 1 and z == 0) / p_z0
p_x0_z1 = sum(p for x, y, z, p in joint_dist if x == 0 and z == 1) / p_z1
p_x1_z1 = sum(p for x, y, z, p in joint_dist if x == 1 and z == 1) / p_z1

print("\nP(X|Z):")
print(f"P(X=0|Z=0) = {p_x0_z0:.2f}")
print(f"P(X=1|Z=0) = {p_x1_z0:.2f}")
print(f"P(X=0|Z=1) = {p_x0_z1:.2f}")
print(f"P(X=1|Z=1) = {p_x1_z1:.2f}")

# Calculate P(Y|Z)
p_y0_z0 = sum(p for x, y, z, p in joint_dist if y == 0 and z == 0) / p_z0
p_y1_z0 = sum(p for x, y, z, p in joint_dist if y == 1 and z == 0) / p_z0
p_y0_z1 = sum(p for x, y, z, p in joint_dist if y == 0 and z == 1) / p_z1
p_y1_z1 = sum(p for x, y, z, p in joint_dist if y == 1 and z == 1) / p_z1

print("\nP(Y|Z):")
print(f"P(Y=0|Z=0) = {p_y0_z0:.2f}")
print(f"P(Y=1|Z=0) = {p_y1_z0:.2f}")
print(f"P(Y=0|Z=1) = {p_y0_z1:.2f}")
print(f"P(Y=1|Z=1) = {p_y1_z1:.2f}")

# Calculate P(X|Y,Z) for each combination
for y in [0, 1]:
    for z in [0, 1]:
        p_yz = sum(p for x, y_val, z_val, p in joint_dist if y_val == y and z_val == z)
        if p_yz > 0:  # Avoid division by zero
            p_x0_yz = sum(p for x, y_val, z_val, p in joint_dist if x == 0 and y_val == y and z_val == z) / p_yz
            p_x1_yz = sum(p for x, y_val, z_val, p in joint_dist if x == 1 and y_val == y and z_val == z) / p_yz
            print(f"\nP(X|Y={y},Z={z}):")
            print(f"P(X=0|Y={y},Z={z}) = {p_x0_yz:.2f}")
            print(f"P(X=1|Y={y},Z={z}) = {p_x1_yz:.2f}")

# Calculate P(X,Y|Z) and compare with P(X|Z)P(Y|Z)
for z in [0, 1]:
    print(f"\nFor Z={z}:")
    for x in [0, 1]:
        for y in [0, 1]:
            # Calculate P(X,Y|Z) directly
            p_xy_z = sum(p for x_val, y_val, z_val, p in joint_dist if x_val == x and y_val == y and z_val == z) / (p_z0 if z == 0 else p_z1)
            
            # Calculate P(X|Z)P(Y|Z)
            p_x_z = p_x0_z0 if x == 0 and z == 0 else p_x1_z0 if x == 1 and z == 0 else p_x0_z1 if x == 0 and z == 1 else p_x1_z1
            p_y_z = p_y0_z0 if y == 0 and z == 0 else p_y1_z0 if y == 1 and z == 0 else p_y0_z1 if y == 0 and z == 1 else p_y1_z1
            
            product = p_x_z * p_y_z
            
            print(f"P(X={x},Y={y}|Z={z}) = {p_xy_z:.2f}")
            print(f"P(X={x}|Z={z})P(Y={y}|Z={z}) = {p_x_z:.2f} × {p_y_z:.2f} = {product:.2f}")
            print(f"Difference: {abs(p_xy_z - product):.4f}")

# Step 9: Visualize the numerical example
print_step_header(9, "Visualizing the Numerical Example")

# Create a heatmap of the conditional distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Reshape data for heatmaps
# P(X,Y|Z=0)
p_xy_given_z0 = np.zeros((2, 2))
for x in [0, 1]:
    for y in [0, 1]:
        p_xy_given_z0[x, y] = sum(p for x_val, y_val, z_val, p in joint_dist if x_val == x and y_val == y and z_val == 0) / p_z0

# P(X|Z=0)P(Y|Z=0)
p_x_z0_times_p_y_z0 = np.zeros((2, 2))
for x in [0, 1]:
    for y in [0, 1]:
        p_x_z0 = sum(p for x_val, y_val, z_val, p in joint_dist if x_val == x and z_val == 0) / p_z0
        p_y_z0 = sum(p for x_val, y_val, z_val, p in joint_dist if y_val == y and z_val == 0) / p_z0
        p_x_z0_times_p_y_z0[x, y] = p_x_z0 * p_y_z0

# P(X,Y|Z=1)
p_xy_given_z1 = np.zeros((2, 2))
for x in [0, 1]:
    for y in [0, 1]:
        p_xy_given_z1[x, y] = sum(p for x_val, y_val, z_val, p in joint_dist if x_val == x and y_val == y and z_val == 1) / p_z1

# P(X|Z=1)P(Y|Z=1)
p_x_z1_times_p_y_z1 = np.zeros((2, 2))
for x in [0, 1]:
    for y in [0, 1]:
        p_x_z1 = sum(p for x_val, y_val, z_val, p in joint_dist if x_val == x and z_val == 1) / p_z1
        p_y_z1 = sum(p for x_val, y_val, z_val, p in joint_dist if y_val == y and z_val == 1) / p_z1
        p_x_z1_times_p_y_z1[x, y] = p_x_z1 * p_y_z1

# Create heatmaps
im1 = axes[0, 0].imshow(p_xy_given_z0, cmap='Blues', vmin=0, vmax=0.5)
axes[0, 0].set_title('P(X,Y|Z=0)', fontsize=12)
axes[0, 0].set_xlabel('Y')
axes[0, 0].set_ylabel('X')
axes[0, 0].set_xticks([0, 1])
axes[0, 0].set_yticks([0, 1])

for i in range(2):
    for j in range(2):
        axes[0, 0].text(j, i, f'{p_xy_given_z0[i, j]:.2f}', ha='center', va='center', color='black')

im2 = axes[0, 1].imshow(p_x_z0_times_p_y_z0, cmap='Blues', vmin=0, vmax=0.5)
axes[0, 1].set_title('P(X|Z=0)P(Y|Z=0)', fontsize=12)
axes[0, 1].set_xlabel('Y')
axes[0, 1].set_ylabel('X')
axes[0, 1].set_xticks([0, 1])
axes[0, 1].set_yticks([0, 1])

for i in range(2):
    for j in range(2):
        axes[0, 1].text(j, i, f'{p_x_z0_times_p_y_z0[i, j]:.2f}', ha='center', va='center', color='black')

im3 = axes[1, 0].imshow(p_xy_given_z1, cmap='Reds', vmin=0, vmax=0.5)
axes[1, 0].set_title('P(X,Y|Z=1)', fontsize=12)
axes[1, 0].set_xlabel('Y')
axes[1, 0].set_ylabel('X')
axes[1, 0].set_xticks([0, 1])
axes[1, 0].set_yticks([0, 1])

for i in range(2):
    for j in range(2):
        axes[1, 0].text(j, i, f'{p_xy_given_z1[i, j]:.2f}', ha='center', va='center', color='black')

im4 = axes[1, 1].imshow(p_x_z1_times_p_y_z1, cmap='Reds', vmin=0, vmax=0.5)
axes[1, 1].set_title('P(X|Z=1)P(Y|Z=1)', fontsize=12)
axes[1, 1].set_xlabel('Y')
axes[1, 1].set_ylabel('X')
axes[1, 1].set_xticks([0, 1])
axes[1, 1].set_yticks([0, 1])

for i in range(2):
    for j in range(2):
        axes[1, 1].text(j, i, f'{p_x_z1_times_p_y_z1[i, j]:.2f}', ha='center', va='center', color='black')

plt.tight_layout()

# Save the visualization
file_path = os.path.join(save_dir, "numerical_example_heatmaps.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Numerical example heatmaps saved to: {file_path}")

# Step 10: Conclusion
print_step_header(10, "Conclusion")

print("Key findings from our proof and examples:")
print("1. We've proven that P(X,Y|Z) = P(X|Z)P(Y|Z) if P(X|Y,Z) = P(X|Z)")
print("2. This means that if knowing Y doesn't add any information about X beyond what Z already tells us,")
print("   then X and Y are conditionally independent given Z")
print("3. Our numerical example demonstrated this equality by showing that P(X,Y|Z) ≈ P(X|Z)P(Y|Z)")
print("4. This property is important in Bayesian networks, causal inference, and probabilistic graphical models")
print("5. Conditional independence allows us to simplify joint probability distributions and reduce")
print("   the number of parameters needed to specify complex models") 