import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from matplotlib import colors

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_2_Quiz_3")
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

print("Given: Joint probability distribution of X and Y")
print("┌───────┬───────┬───────┐")
print("│       │ Y = 0 │ Y = 1 │")
print("├───────┼───────┼───────┤")
print("│ X = 0 │  0.3  │  0.2  │")
print("├───────┼───────┼───────┤")
print("│ X = 1 │  0.1  │  0.4  │")
print("└───────┴───────┴───────┘")
print()
print("Tasks:")
print("1. Calculate the entropy of X, H(X)")
print("2. Calculate the entropy of Y, H(Y)")
print("3. Calculate the joint entropy H(X, Y)")
print("4. Calculate the mutual information I(X; Y) and interpret what it means")
print()

# Step 2: Define the joint probability distribution
print_step_header(2, "Defining the Joint Probability Distribution")

# Define the joint probability distribution as a 2D array
joint_prob = np.array([
    [0.3, 0.2],  # P(X=0, Y=0), P(X=0, Y=1)
    [0.1, 0.4]   # P(X=1, Y=0), P(X=1, Y=1)
])

print("Joint probability distribution P(X, Y):")
for i in range(2):
    for j in range(2):
        print(f"P(X = {i}, Y = {j}) = {joint_prob[i, j]}")

# Calculate the marginal probabilities
p_x = np.sum(joint_prob, axis=1)  # Sum over Y to get P(X)
p_y = np.sum(joint_prob, axis=0)  # Sum over X to get P(Y)

print("\nMarginal probability distribution P(X):")
for i in range(2):
    print(f"P(X = {i}) = {p_x[i]}")

print("\nMarginal probability distribution P(Y):")
for j in range(2):
    print(f"P(Y = {j}) = {p_y[j]}")

# Visualize the joint and marginal distributions
fig = plt.figure(figsize=(12, 9))
gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4])

# Joint distribution (main heatmap)
ax_joint = fig.add_subplot(gs[1, 0])
im = ax_joint.imshow(joint_prob, cmap='Blues', 
                     norm=colors.Normalize(vmin=0, vmax=np.max(joint_prob)))
for i in range(2):
    for j in range(2):
        text = ax_joint.text(j, i, f'{joint_prob[i, j]:.1f}',
                           ha="center", va="center", color="black", fontweight='bold')

ax_joint.set_xticks([0, 1])
ax_joint.set_yticks([0, 1])
ax_joint.set_xticklabels(['Y = 0', 'Y = 1'])
ax_joint.set_yticklabels(['X = 0', 'X = 1'])
ax_joint.set_xlabel('Y')
ax_joint.set_ylabel('X')
ax_joint.set_title('Joint Probability Distribution P(X, Y)')

# Colorbar
cbar = fig.colorbar(im, ax=ax_joint)
cbar.set_label('Probability')

# X marginal (top)
ax_x = fig.add_subplot(gs[0, 0], sharex=ax_joint)
ax_x.bar(np.arange(2), p_y, color='skyblue', width=0.8)
ax_x.set_title('Marginal Distribution P(Y)')
ax_x.set_ylim(0, 1)
ax_x.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax_x.tick_params(labelbottom=False)

# Y marginal (right)
ax_y = fig.add_subplot(gs[1, 1], sharey=ax_joint)
ax_y.barh(np.arange(2), p_x, color='skyblue', height=0.8)
ax_y.set_title('Marginal Distribution P(X)')
ax_y.set_xlim(0, 1)
ax_y.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax_y.tick_params(labelleft=False)

# Empty plot (top-right corner)
ax_empty = fig.add_subplot(gs[0, 1])
ax_empty.axis('off')

plt.tight_layout()
file_path = os.path.join(save_dir, "joint_distribution.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 3: Calculate the entropy of X, H(X)
print_step_header(3, "Calculating the Entropy of X, H(X)")

def entropy(p):
    """Calculate the entropy of a probability distribution."""
    # Handle 0 probabilities (0 * log(0) = 0)
    h = 0
    for p_i in p:
        if p_i > 0:
            h -= p_i * np.log2(p_i)
    return h

h_x = entropy(p_x)
print(f"H(X) = {h_x:.6f} bits")

# Show the calculation steps
print("\nCalculation steps:")
for i, p_i in enumerate(p_x):
    if p_i > 0:
        term = -p_i * np.log2(p_i)
        print(f"- P(X = {i}) * (-log2(P(X = {i}))) = {p_i} * (-log2({p_i})) = {term:.6f}")

# Visualize the entropy calculation for X
plt.figure(figsize=(10, 6))
x_vals = np.arange(2)
terms = np.zeros_like(p_x, dtype=float)
for i, p_i in enumerate(p_x):
    if p_i > 0:
        terms[i] = -p_i * np.log2(p_i)

plt.bar(x_vals, terms, color=['blue', 'green'], alpha=0.7)
plt.xlabel('X values')
plt.ylabel('- P(X) * log₂(P(X))')
plt.title('Terms in H(X) Entropy Calculation')
plt.xticks(x_vals, ['X = 0', 'X = 1'])
plt.grid(axis='y')

# Add a horizontal line for the sum (entropy)
plt.axhline(y=h_x, color='r', linestyle='--', alpha=0.7,
           label=f'H(X) = {h_x:.4f} bits')

# Add value annotations
for i, term in enumerate(terms):
    plt.text(i, term + 0.01, f'{term:.4f}', ha='center')

plt.legend()
plt.tight_layout()
file_path = os.path.join(save_dir, "entropy_x.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 4: Calculate the entropy of Y, H(Y)
print_step_header(4, "Calculating the Entropy of Y, H(Y)")

h_y = entropy(p_y)
print(f"H(Y) = {h_y:.6f} bits")

# Show the calculation steps
print("\nCalculation steps:")
for j, p_j in enumerate(p_y):
    if p_j > 0:
        term = -p_j * np.log2(p_j)
        print(f"- P(Y = {j}) * (-log2(P(Y = {j}))) = {p_j} * (-log2({p_j})) = {term:.6f}")

# Visualize the entropy calculation for Y
plt.figure(figsize=(10, 6))
y_vals = np.arange(2)
terms = np.zeros_like(p_y, dtype=float)
for j, p_j in enumerate(p_y):
    if p_j > 0:
        terms[j] = -p_j * np.log2(p_j)

plt.bar(y_vals, terms, color=['red', 'orange'], alpha=0.7)
plt.xlabel('Y values')
plt.ylabel('- P(Y) * log₂(P(Y))')
plt.title('Terms in H(Y) Entropy Calculation')
plt.xticks(y_vals, ['Y = 0', 'Y = 1'])
plt.grid(axis='y')

# Add a horizontal line for the sum (entropy)
plt.axhline(y=h_y, color='r', linestyle='--', alpha=0.7,
           label=f'H(Y) = {h_y:.4f} bits')

# Add value annotations
for j, term in enumerate(terms):
    plt.text(j, term + 0.01, f'{term:.4f}', ha='center')

plt.legend()
plt.tight_layout()
file_path = os.path.join(save_dir, "entropy_y.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 5: Calculate the joint entropy H(X, Y)
print_step_header(5, "Calculating the Joint Entropy H(X, Y)")

def joint_entropy(p_xy):
    """Calculate the joint entropy of a joint probability distribution."""
    h = 0
    for i in range(p_xy.shape[0]):
        for j in range(p_xy.shape[1]):
            p_ij = p_xy[i, j]
            if p_ij > 0:
                h -= p_ij * np.log2(p_ij)
    return h

h_xy = joint_entropy(joint_prob)
print(f"H(X, Y) = {h_xy:.6f} bits")

# Show the calculation steps
print("\nCalculation steps:")
terms = np.zeros_like(joint_prob, dtype=float)
for i in range(joint_prob.shape[0]):
    for j in range(joint_prob.shape[1]):
        p_ij = joint_prob[i, j]
        if p_ij > 0:
            term = -p_ij * np.log2(p_ij)
            terms[i, j] = term
            print(f"- P(X = {i}, Y = {j}) * (-log2(P(X = {i}, Y = {j}))) = {p_ij} * (-log2({p_ij})) = {term:.6f}")

# Visualize the joint entropy calculation
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(terms, cmap='Reds')

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.set_label('- P(X,Y) * log₂(P(X,Y))')

# Show terms in each cell
for i in range(terms.shape[0]):
    for j in range(terms.shape[1]):
        ax.text(j, i, f'{terms[i, j]:.4f}',
               ha="center", va="center", color="black")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Y = 0', 'Y = 1'])
ax.set_yticklabels(['X = 0', 'X = 1'])
ax.set_xlabel('Y')
ax.set_ylabel('X')
ax.set_title(f'Terms in Joint Entropy H(X, Y) = {h_xy:.4f} bits')

plt.tight_layout()
file_path = os.path.join(save_dir, "joint_entropy.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 6: Calculate the mutual information I(X; Y)
print_step_header(6, "Calculating the Mutual Information I(X; Y)")

# Calculate mutual information via the formula: I(X; Y) = H(X) + H(Y) - H(X, Y)
mi = h_x + h_y - h_xy
print(f"I(X; Y) = H(X) + H(Y) - H(X, Y)")
print(f"I(X; Y) = {h_x:.6f} + {h_y:.6f} - {h_xy:.6f} = {mi:.6f} bits")

# Alternative calculation using the KL divergence formula
print("\nAlternative calculation using the KL divergence formula:")
print("I(X; Y) = sum_{x,y} P(x,y) * log2(P(x,y) / (P(x)P(y)))")

terms = np.zeros_like(joint_prob, dtype=float)
for i in range(joint_prob.shape[0]):
    for j in range(joint_prob.shape[1]):
        p_ij = joint_prob[i, j]
        p_i = p_x[i]
        p_j = p_y[j]
        if p_ij > 0:
            term = p_ij * np.log2(p_ij / (p_i * p_j))
            terms[i, j] = term
            print(f"- P(X = {i}, Y = {j}) * log2(P(X = {i}, Y = {j}) / (P(X = {i}) * P(Y = {j}))) = {p_ij} * log2({p_ij} / ({p_i} * {p_j})) = {term:.6f}")

mi_direct = np.sum(terms)
print(f"Sum of all terms = {mi_direct:.6f} bits")

if np.isclose(mi, mi_direct):
    print(f"\nVerification: The two calculations match ✓")
else:
    print(f"\nVerification: The two calculations do not match ✗")

# Visualize the mutual information calculation
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(terms, cmap='PuBu')

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.set_label('P(X,Y) * log₂(P(X,Y) / (P(X)P(Y)))')

# Show terms in each cell
for i in range(terms.shape[0]):
    for j in range(terms.shape[1]):
        ax.text(j, i, f'{terms[i, j]:.4f}',
               ha="center", va="center", color="black")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Y = 0', 'Y = 1'])
ax.set_yticklabels(['X = 0', 'X = 1'])
ax.set_xlabel('Y')
ax.set_ylabel('X')
ax.set_title(f'Terms in Mutual Information I(X; Y) = {mi:.4f} bits')

plt.tight_layout()
file_path = os.path.join(save_dir, "mutual_information.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 7: Visualize the relationships between entropies and mutual information
print_step_header(7, "Visualizing the Relationships between Entropy and Mutual Information")

# Create a Venn diagram-style visualization
plt.figure(figsize=(10, 8))

# Parameters for the circles
h_x_only = h_x - mi
h_y_only = h_y - mi

# Draw a text-based Venn diagram
plt.text(0.25, 0.7, f'H(X|Y) = {h_x_only:.4f}', fontsize=12, ha='center')
plt.text(0.75, 0.7, f'H(Y|X) = {h_y_only:.4f}', fontsize=12, ha='center')
plt.text(0.5, 0.5, f'I(X;Y) = {mi:.4f}', fontsize=14, ha='center', bbox=dict(facecolor='yellow', alpha=0.3))
plt.text(0.5, 0.9, f'H(X,Y) = {h_xy:.4f}', fontsize=14, ha='center', bbox=dict(facecolor='green', alpha=0.3))

# Create the circles
from matplotlib.patches import Ellipse
circle1 = Ellipse((0.3, 0.5), 0.4, 0.6, alpha=0.3, color='blue')
circle2 = Ellipse((0.7, 0.5), 0.4, 0.6, alpha=0.3, color='red')

plt.gca().add_patch(circle1)
plt.gca().add_patch(circle2)

# Add labels to the circles
plt.text(0.3, 0.5, f'H(X) = {h_x:.4f}', fontsize=14, ha='center', va='center')
plt.text(0.7, 0.5, f'H(Y) = {h_y:.4f}', fontsize=14, ha='center', va='center')

# Set axis properties
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis('off')  # Hide axes
plt.title('Information Theory Measures for X and Y', fontsize=16)

# Add explanatory text
plt.text(0.5, 0.1, 'Mutual Information I(X;Y) measures how much knowing one\n'
                   'variable reduces uncertainty about the other.', 
         fontsize=12, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8))

# Add relationship equations
plt.text(0.5, 0.25, 'Key Relationships:', fontsize=14, ha='center', va='center')
plt.text(0.5, 0.20, f'I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)', fontsize=12, ha='center', va='center')
plt.text(0.5, 0.15, f'H(X,Y) = H(X) + H(Y) - I(X;Y)', fontsize=12, ha='center', va='center')
plt.text(0.5, 0.0, f'I(X;Y) > 0 indicates X and Y are dependent', fontsize=12, ha='center', va='center')

plt.tight_layout()
file_path = os.path.join(save_dir, "information_venn.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Create a bar chart to compare the different information measures
plt.figure(figsize=(12, 6))
measures = ['H(X)', 'H(Y)', 'H(X,Y)', 'I(X;Y)', 'H(X|Y)', 'H(Y|X)']
values = [h_x, h_y, h_xy, mi, h_x_only, h_y_only]
bars = plt.bar(measures, values, color=['blue', 'red', 'green', 'yellow', 'lightblue', 'lightcoral'])

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.4f}', ha='center', va='bottom')

plt.xlabel('Information Measures')
plt.ylabel('Value (bits)')
plt.title('Comparison of Information Measures')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
file_path = os.path.join(save_dir, "information_measures.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 8: Interpret the meaning of mutual information
print_step_header(8, "Interpreting the Meaning of Mutual Information")

# Calculate conditional entropies for interpretation
h_x_given_y = h_xy - h_y  # H(X|Y) = H(X,Y) - H(Y)
h_y_given_x = h_xy - h_x  # H(Y|X) = H(X,Y) - H(X)

print(f"Mutual Information I(X; Y) = {mi:.6f} bits")
print(f"Normalized Mutual Information (I(X;Y) / min(H(X), H(Y))) = {mi / min(h_x, h_y):.6f} or {mi / min(h_x, h_y) * 100:.2f}%")

print("\nInterpretation:")
print(f"1. The mutual information of {mi:.6f} bits quantifies how much knowing one variable")
print(f"   reduces uncertainty about the other variable.")

if np.isclose(mi, 0):
    print(f"2. Since I(X;Y) ≈ 0, the variables X and Y are statistically independent.")
elif mi > 0:
    print(f"2. Since I(X;Y) > 0, the variables X and Y are dependent.")

print(f"3. Reduction in uncertainty about X when Y is known: {h_x - h_x_given_y:.6f} bits")
print(f"   ({(h_x - h_x_given_y) / h_x * 100:.2f}% reduction in uncertainty)")

print(f"4. Reduction in uncertainty about Y when X is known: {h_y - h_y_given_x:.6f} bits")
print(f"   ({(h_y - h_y_given_x) / h_y * 100:.2f}% reduction in uncertainty)")

# Generate a final comprehensive visualization
plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plt.imshow(joint_prob, cmap='Blues')
for i in range(joint_prob.shape[0]):
    for j in range(joint_prob.shape[1]):
        plt.text(j, i, f'{joint_prob[i, j]:.1f}', ha='center', va='center', color='black')
plt.xticks([0, 1], ['Y = 0', 'Y = 1'])
plt.yticks([0, 1], ['X = 0', 'X = 1'])
plt.title('Joint Probability P(X,Y)')
plt.colorbar(label='Probability')

plt.subplot(2, 2, 2)
plt.imshow(terms, cmap='PuBu')
for i in range(terms.shape[0]):
    for j in range(terms.shape[1]):
        plt.text(j, i, f'{terms[i, j]:.4f}', ha='center', va='center', color='black')
plt.xticks([0, 1], ['Y = 0', 'Y = 1'])
plt.yticks([0, 1], ['X = 0', 'X = 1'])
plt.title('Mutual Information Terms')
plt.colorbar(label='P(X,Y) * log₂(P(X,Y) / (P(X)P(Y)))')

plt.subplot(2, 2, 3)
# Compute and plot the conditional probabilities P(Y|X)
p_y_given_x = joint_prob / p_x[:, np.newaxis]
plt.imshow(p_y_given_x, cmap='Greens')
for i in range(p_y_given_x.shape[0]):
    for j in range(p_y_given_x.shape[1]):
        plt.text(j, i, f'{p_y_given_x[i, j]:.4f}', ha='center', va='center', color='black')
plt.xticks([0, 1], ['Y = 0', 'Y = 1'])
plt.yticks([0, 1], ['X = 0', 'X = 1'])
plt.title('Conditional Probability P(Y|X)')
plt.colorbar(label='P(Y|X)')

plt.subplot(2, 2, 4)
# Compute and plot the conditional probabilities P(X|Y)
p_x_given_y = (joint_prob.T / p_y[:, np.newaxis]).T
plt.imshow(p_x_given_y, cmap='Oranges')
for i in range(p_x_given_y.shape[0]):
    for j in range(p_x_given_y.shape[1]):
        plt.text(j, i, f'{p_x_given_y[i, j]:.4f}', ha='center', va='center', color='black')
plt.xticks([0, 1], ['Y = 0', 'Y = 1'])
plt.yticks([0, 1], ['X = 0', 'X = 1'])
plt.title('Conditional Probability P(X|Y)')
plt.colorbar(label='P(X|Y)')

plt.tight_layout()
file_path = os.path.join(save_dir, "comprehensive_view.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 9: Conclusion and Answer Summary
print_step_header(9, "Conclusion and Answer Summary")

print("Question 3 Solution Summary:")
print("1. Entropy of X: H(X) = %.6f bits" % h_x)
print("2. Entropy of Y: H(Y) = %.6f bits" % h_y)
print("3. Joint Entropy: H(X, Y) = %.6f bits" % h_xy)
print("4. Mutual Information: I(X; Y) = %.6f bits" % mi)
print()
print("Interpretation of Mutual Information:")
print("The mutual information of %.6f bits indicates the amount of information" % mi)
print("shared between variables X and Y. It quantifies how much knowing one")
print("variable reduces uncertainty about the other variable.")
print()
print("Since I(X; Y) > 0, variables X and Y are dependent. This means that")
print("knowing the value of one variable gives us information about the other.")
print()
print("Specifically, knowing Y reduces uncertainty about X by %.2f%%, and" % ((h_x - h_x_given_y) / h_x * 100))
print("knowing X reduces uncertainty about Y by %.2f%%." % ((h_y - h_y_given_x) / h_y * 100)) 