import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib_venn import venn2, venn3

print("\n=== JOINT PROBABILITY EXAMPLES: STEP-BY-STEP SOLUTIONS ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Set style for all plots
plt.style.use('default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'

# Example 1: Employee Education and Position
print("\nExample 1: Employee Education and Position")
total_employees = 200
grad_degree = 80
management = 120
both = 60

# Calculate probabilities
p_grad = grad_degree / total_employees
p_management = management / total_employees
p_both = both / total_employees

print(f"Total employees: {total_employees}")
print(f"Employees with graduate degree: {grad_degree}")
print(f"Employees in management: {management}")
print(f"Employees with both: {both}")

print("\nStep-by-step calculation:")
print(f"P(G) = {grad_degree}/{total_employees} = {p_grad:.2f}")
print(f"P(M) = {management}/{total_employees} = {p_management:.2f}")
print(f"P(G ∩ M) = {both}/{total_employees} = {p_both:.2f}")

# Check independence
p_independent = p_grad * p_management
print(f"\nIf independent, P(G ∩ M) would be: {p_grad:.2f} × {p_management:.2f} = {p_independent:.2f}")
print(f"Since {p_both:.2f} ≠ {p_independent:.2f}, the events are not independent")

# Create Venn diagram
plt.figure(figsize=(8, 6))
v = venn2(subsets=(grad_degree - both, management - both, both),
         set_labels=('Graduate Degree', 'Management'),
         set_colors=('#1f77b4', '#ff7f0e'),
         alpha=0.7)
plt.title('Employee Education and Position', pad=20)
plt.savefig(os.path.join(images_dir, 'employee_venn.png'), dpi=100, bbox_inches='tight')
plt.close()

# Create bar chart for probabilities
plt.figure(figsize=(8, 6))
labels = ['P(G)', 'P(M)', 'P(G ∩ M)', 'P(G) × P(M)']
values = [p_grad, p_management, p_both, p_independent]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
plt.bar(labels, values, color=colors, alpha=0.7)
plt.axhline(y=p_independent, color='r', linestyle='--', label='Independent Value')
plt.title('Probability Comparison')
plt.ylabel('Probability')
plt.ylim(0, 0.7)
plt.legend()
plt.savefig(os.path.join(images_dir, 'employee_probabilities.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Plant Genetics
print("\n\nExample 2: Plant Genetics")
p_a = 0.40
p_b = 0.35
p_c = 0.25
p_ab = 0.15
p_ac = 0.10
p_bc = 0.05
p_abc = 0.02

print("Given probabilities:")
print(f"P(A) = {p_a:.2f}")
print(f"P(B) = {p_b:.2f}")
print(f"P(C) = {p_c:.2f}")
print(f"P(A ∩ B) = {p_ab:.2f}")
print(f"P(A ∩ C) = {p_ac:.2f}")
print(f"P(B ∩ C) = {p_bc:.2f}")
print(f"P(A ∩ B ∩ C) = {p_abc:.2f}")

# Calculate P(A ∪ B ∪ C)
p_union = p_a + p_b + p_c - p_ab - p_ac - p_bc + p_abc

print("\nStep-by-step calculation using inclusion-exclusion:")
print(f"P(A ∪ B ∪ C) = P(A) + P(B) + P(C) - P(A ∩ B) - P(A ∩ C) - P(B ∩ C) + P(A ∩ B ∩ C)")
print(f"              = {p_a:.2f} + {p_b:.2f} + {p_c:.2f} - {p_ab:.2f} - {p_ac:.2f} - {p_bc:.2f} + {p_abc:.2f}")
print(f"              = {p_union:.2f}")

# Create Venn diagram for three events
plt.figure(figsize=(8, 6))
v = venn3(subsets=(p_a - p_ab - p_ac + p_abc, 
                  p_b - p_ab - p_bc + p_abc,
                  p_ab - p_abc,
                  p_c - p_ac - p_bc + p_abc,
                  p_ac - p_abc,
                  p_bc - p_abc,
                  p_abc),
         set_labels=('Allele A', 'Allele B', 'Allele C'),
         set_colors=('#1f77b4', '#ff7f0e', '#2ca02c'),
         alpha=0.7)

# Format the numbers in the Venn diagram
for idx, subset in enumerate(v.subset_labels):
    if subset is not None:
        v.subset_labels[idx].set_text(f"{float(subset.get_text()):.2f}")

plt.title('Plant Genetics', pad=20)
plt.savefig(os.path.join(images_dir, 'plant_venn.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Joint Distribution
print("\n\nExample 3: Joint Distribution")
# Create joint probability table
joint_prob = np.array([
    [0.10, 0.05, 0.05],
    [0.20, 0.30, 0.05],
    [0.05, 0.15, 0.05]
])

print("Joint Probability Mass Function:")
df = pd.DataFrame(joint_prob, 
                 index=['x=1', 'x=2', 'x=3'],
                 columns=['y=1', 'y=2', 'y=3'])
print(df)

# Calculate marginal distributions
marginal_x = joint_prob.sum(axis=1)
marginal_y = joint_prob.sum(axis=0)

print("\nMarginal Distribution of X:")
for i, p in enumerate(marginal_x, 1):
    print(f"P(X={i}) = {p:.2f}")

print("\nMarginal Distribution of Y:")
for i, p in enumerate(marginal_y, 1):
    print(f"P(Y={i}) = {p:.2f}")

# Check independence
x2_y2 = joint_prob[1, 1]  # P(X=2, Y=2)
x2 = marginal_x[1]        # P(X=2)
y2 = marginal_y[1]        # P(Y=2)
independent_value = x2 * y2

print("\nChecking independence:")
print(f"P(X=2, Y=2) = {x2_y2:.2f}")
print(f"P(X=2) × P(Y=2) = {x2:.2f} × {y2:.2f} = {independent_value:.2f}")
print(f"Since {x2_y2:.2f} ≠ {independent_value:.2f}, X and Y are not independent")

# Create heatmap of joint distribution
plt.figure(figsize=(8, 6))
plt.imshow(joint_prob, cmap='YlOrRd', interpolation='nearest')
plt.colorbar(label='Probability')

plt.xticks(range(3), ['y=1', 'y=2', 'y=3'])
plt.yticks(range(3), ['x=1', 'x=2', 'x=3'])
plt.title('Joint Probability Distribution')
plt.xlabel('Y')
plt.ylabel('X')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'joint_distribution.png'), dpi=100, bbox_inches='tight')
plt.close()

# Create marginal distributions plot
plt.figure(figsize=(8, 6))
x = np.arange(1, 4)
width = 0.35

plt.bar(x - width/2, marginal_x, width, label='P(X)', color='#1f77b4', alpha=0.7)
plt.bar(x + width/2, marginal_y, width, label='P(Y)', color='#ff7f0e', alpha=0.7)

plt.xlabel('Value')
plt.ylabel('Probability')
plt.title('Marginal Distributions')
plt.xticks(x, ['1', '2', '3'])
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(images_dir, 'marginal_distributions.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll joint probability example images created successfully.") 