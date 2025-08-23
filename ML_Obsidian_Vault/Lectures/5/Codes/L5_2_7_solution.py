import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, diff, solve, Eq, simplify
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_2_Quiz_7")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("=" * 80)
print("SOFT MARGIN SVM - KKT CONDITIONS DERIVATION")
print("=" * 80)

# Define symbolic variables
w1, w2, b = symbols('w1 w2 b')
alpha1, alpha2, alpha3, alpha4, alpha5 = symbols('alpha1 alpha2 alpha3 alpha4 alpha5')
mu1, mu2, mu3, mu4, mu5 = symbols('mu1 mu2 mu3 mu4 mu5')
xi1, xi2, xi3, xi4, xi5 = symbols('xi1 xi2 xi3 xi4 xi5')
C = symbols('C')

# Sample data points for demonstration
x_data = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [0.1, 0.1]])
y_data = np.array([1, 1, -1, -1, -1])

print("\n1. KKT STATIONARITY CONDITIONS")
print("-" * 40)

# Lagrangian function
print("Given Lagrangian:")
print("L = (1/2)||w||² + C∑ξᵢ - ∑αᵢ[yᵢ(w^T xᵢ + b) - 1 + ξᵢ] - ∑μᵢξᵢ")

# For demonstration, let's use 5 data points
n = 5
w = sp.Matrix([w1, w2])

# Construct the Lagrangian symbolically
lagrangian = (1/2) * (w1**2 + w2**2) + C * (xi1 + xi2 + xi3 + xi4 + xi5)

# Add the constraint terms
for i in range(n):
    x_i = sp.Matrix(x_data[i])
    y_i = y_data[i]
    alpha_i = [alpha1, alpha2, alpha3, alpha4, alpha5][i]
    xi_i = [xi1, xi2, xi3, xi4, xi5][i]
    
    constraint_term = alpha_i * (y_i * (w1 * x_i[0] + w2 * x_i[1] + b) - 1 + xi_i)
    lagrangian -= constraint_term

# Add the non-negativity constraint terms
for i in range(n):
    mu_i = [mu1, mu2, mu3, mu4, mu5][i]
    xi_i = [xi1, xi2, xi3, xi4, xi5][i]
    lagrangian -= mu_i * xi_i

print(f"\nExpanded Lagrangian:")
print(f"L = {lagrangian}")

# KKT Stationarity Conditions
print("\n\nKKT Stationarity Conditions (∂L/∂variable = 0):")

# 1. ∂L/∂w = 0
print("\n1. ∂L/∂w = 0:")
dl_dw1 = diff(lagrangian, w1)
dl_dw2 = diff(lagrangian, w2)
print(f"∂L/∂w₁ = {dl_dw1} = 0")
print(f"∂L/∂w₂ = {dl_dw2} = 0")

# Solve for w in terms of alpha
print("\nSolving for w:")
w1_solution = solve(dl_dw1, w1)[0]
w2_solution = solve(dl_dw2, w2)[0]
print(f"w₁ = {w1_solution}")
print(f"w₂ = {w2_solution}")

# 2. ∂L/∂b = 0
print("\n2. ∂L/∂b = 0:")
dl_db = diff(lagrangian, b)
print(f"∂L/∂b = {dl_db} = 0")
print(f"This gives us: ∑αᵢyᵢ = 0")

# 3. ∂L/∂ξᵢ = 0
print("\n3. ∂L/∂ξᵢ = 0:")
for i in range(n):
    xi_i = [xi1, xi2, xi3, xi4, xi5][i]
    alpha_i = [alpha1, alpha2, alpha3, alpha4, alpha5][i]
    mu_i = [mu1, mu2, mu3, mu4, mu5][i]
    dl_dxi = diff(lagrangian, xi_i)
    print(f"∂L/∂ξ_{i+1} = {dl_dxi} = 0")
    print(f"This gives us: C - α_{i+1} - μ_{i+1} = 0")
    print(f"Therefore: α_{i+1} + μ_{i+1} = C")

print("\n" + "=" * 80)
print("2. DERIVING THE CONSTRAINT ∑αᵢyᵢ = 0")
print("=" * 80)

print("\nFrom ∂L/∂b = 0:")
print(f"{dl_db} = 0")
print("This directly gives us the constraint:")
print("∑αᵢyᵢ = 0")

# Demonstrate with our data
print(f"\nFor our data points:")
for i in range(n):
    print(f"Point {i+1}: x = {x_data[i]}, y = {y_data[i]}")
print(f"Constraint: {alpha1}*{y_data[0]} + {alpha2}*{y_data[1]} + {alpha3}*{y_data[2]} + {alpha4}*{y_data[3]} + {alpha5}*{y_data[4]} = 0")
print(f"Constraint: {alpha1} + {alpha2} - {alpha3} - {alpha4} - {alpha5} = 0")

print("\n" + "=" * 80)
print("3. SHOWING THAT αᵢ + μᵢ = C FOR ALL i")
print("=" * 80)

print("\nFrom ∂L/∂ξᵢ = 0 for each i:")
for i in range(n):
    print(f"∂L/∂ξ_{i+1} = C - α_{i+1} - μ_{i+1} = 0")
    print(f"Therefore: α_{i+1} + μ_{i+1} = C")

print("\nThis relationship holds for all i = 1, 2, ..., n")

print("\n" + "=" * 80)
print("4. PROVING THAT 0 ≤ αᵢ ≤ C FOR ALL i")
print("=" * 80)

print("\nWe know from KKT conditions:")
print("1. αᵢ ≥ 0 (non-negativity constraint)")
print("2. μᵢ ≥ 0 (non-negativity constraint)")
print("3. αᵢ + μᵢ = C (from stationarity condition)")

print("\nFrom αᵢ + μᵢ = C and μᵢ ≥ 0:")
print("αᵢ = C - μᵢ")
print("Since μᵢ ≥ 0, we have:")
print("αᵢ ≤ C")

print("\nCombining with αᵢ ≥ 0:")
print("0 ≤ αᵢ ≤ C")

print("\n" + "=" * 80)
print("5. CLASSIFYING TRAINING POINTS BASED ON αᵢ AND ξᵢ VALUES")
print("=" * 80)

# Create visualization of different point types with actual examples
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle(r'Classification of Training Points in Soft Margin SVM', fontsize=16)

# Common setup for all subplots
for i, ax in enumerate(axes.flat):
    # Draw decision boundary and margins
    x_line = np.linspace(-3, 3, 100)
    y_boundary = x_line  # Simple diagonal boundary
    y_margin_upper = x_line + 1
    y_margin_lower = x_line - 1
    
    # Plot decision boundary and margins
    ax.plot(x_line, y_boundary, 'g-', linewidth=2, label='Decision Boundary')
    ax.plot(x_line, y_margin_upper, 'g--', linewidth=1, alpha=0.7, label='Margin')
    ax.plot(x_line, y_margin_lower, 'g--', linewidth=1, alpha=0.7)
    
    # Shade regions
    ax.fill_between(x_line, y_margin_lower, y_margin_upper, alpha=0.2, color='green')
    ax.fill_between(x_line, -5, y_boundary, alpha=0.1, color='blue')
    ax.fill_between(x_line, y_boundary, 5, alpha=0.1, color='red')
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')

# Case 1: α_i = 0 (correctly classified, not support vector)
ax1 = axes[0, 0]
ax1.set_title(r'Case 1: $\alpha_i = 0$ (Correctly Classified)')

# Add points well outside the margin (correctly classified)
# Positive class points (above boundary + margin)
pos_points_x = np.array([-2, -1, 0, 1])
pos_points_y = np.array([0, 1, 2, 3])
ax1.scatter(pos_points_x, pos_points_y, c='red', s=100, marker='o', 
           edgecolors='black', linewidth=1, label='Class +1')

# Negative class points (below boundary - margin)
neg_points_x = np.array([0, 1, 2, -1])
neg_points_y = np.array([-2, -1, 0, -3])
ax1.scatter(neg_points_x, neg_points_y, c='blue', s=100, marker='s', 
           edgecolors='black', linewidth=1, label='Class -1')

ax1.text(0.02, 0.98, r'$\alpha_i = 0$, $\xi_i = 0$' + '\n' + 
         r'Outside margin' + '\n' + r'Not support vectors',
         transform=ax1.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

# Case 2: 0 < α_i < C (support vector on margin)
ax2 = axes[0, 1]
ax2.set_title(r'Case 2: $0 < \alpha_i < C$ (Support Vector on Margin)')

# Add points exactly on the margin
margin_pos_x = np.array([-2, -1, 0, 1])
margin_pos_y = margin_pos_x + 1  # On upper margin
ax2.scatter(margin_pos_x, margin_pos_y, c='red', s=150, marker='o', 
           edgecolors='black', linewidth=3, label='Class +1 SV')

margin_neg_x = np.array([-1, 0, 1, 2])
margin_neg_y = margin_neg_x - 1  # On lower margin
ax2.scatter(margin_neg_x, margin_neg_y, c='blue', s=150, marker='s', 
           edgecolors='black', linewidth=3, label='Class -1 SV')

ax2.text(0.02, 0.98, r'$0 < \alpha_i < C$, $\xi_i = 0$' + '\n' + 
         r'On margin boundary' + '\n' + r'Support vectors',
         transform=ax2.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

# Case 3: α_i = C, ξ_i > 0 (misclassified or within margin)
ax3 = axes[1, 0]
ax3.set_title(r'Case 3: $\alpha_i = C$, $\xi_i > 0$ (Misclassified)')

# Add misclassified points (on wrong side of boundary)
misc_pos_x = np.array([-1.5, -0.5, 0.5])
misc_pos_y = np.array([-2, -1, -0.5])  # Positive class on negative side
ax3.scatter(misc_pos_x, misc_pos_y, c='red', s=150, marker='o', 
           edgecolors='black', linewidth=3, label='Misclassified +1')

misc_neg_x = np.array([0.5, 1.5, 2])
misc_neg_y = np.array([1, 2, 2.5])  # Negative class on positive side
ax3.scatter(misc_neg_x, misc_neg_y, c='blue', s=150, marker='s', 
           edgecolors='black', linewidth=3, label='Misclassified -1')

# Add points within margin but correctly classified
within_pos_x = np.array([-1, 0])
within_pos_y = np.array([0, 0.5])  # Within margin but correct side
ax3.scatter(within_pos_x, within_pos_y, c='red', s=120, marker='o', 
           edgecolors='orange', linewidth=2, alpha=0.7, label='Within margin +1')

within_neg_x = np.array([0, 1])
within_neg_y = np.array([-0.5, 0])  # Within margin but correct side
ax3.scatter(within_neg_x, within_neg_y, c='blue', s=120, marker='s', 
           edgecolors='orange', linewidth=2, alpha=0.7, label='Within margin -1')

ax3.text(0.02, 0.98, r'$\alpha_i = C$, $\xi_i > 0$' + '\n' + 
         r'Violating margin' + '\n' + r'Support vectors',
         transform=ax3.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))

# Case 4: α_i = C, ξ_i = 0 (boundary case)
ax4 = axes[1, 1]
ax4.set_title(r'Case 4: $\alpha_i = C$, $\xi_i = 0$ (Boundary Case)')

# Add points exactly on margin (boundary case)
boundary_pos_x = np.array([-1.5, -0.5, 0.5])
boundary_pos_y = boundary_pos_x + 1  # On upper margin
ax4.scatter(boundary_pos_x, boundary_pos_y, c='red', s=150, marker='o', 
           edgecolors='purple', linewidth=3, label='Boundary +1 SV')

boundary_neg_x = np.array([-0.5, 0.5, 1.5])
boundary_neg_y = boundary_neg_x - 1  # On lower margin
ax4.scatter(boundary_neg_x, boundary_neg_y, c='blue', s=150, marker='s', 
           edgecolors='purple', linewidth=3, label='Boundary -1 SV')

ax4.text(0.02, 0.98, r'$\alpha_i = C$, $\xi_i = 0$' + '\n' + 
         r'On margin boundary' + '\n' + r'Limiting case',
         transform=ax4.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

# Add legends to each subplot
for ax in axes.flat:
    ax.legend(loc='lower right', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'point_classification.png'), dpi=300, bbox_inches='tight')

# Create decision tree visualization
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_title(r'Decision Tree for Classifying Points Based on $(\alpha_i, \xi_i)$', fontsize=14)

# Decision tree structure
y_positions = [0.9, 0.7, 0.5, 0.3, 0.1]
x_positions = [0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875]

# Root node
ax.text(0.5, 0.9, r'$\alpha_i = 0$?', ha='center', va='center', 
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8), fontsize=12)

# First level
ax.text(0.25, 0.7, 'Yes\nCorrectly\nClassified\n(Not SV)', ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8), fontsize=10)

ax.text(0.75, 0.7, 'No\nSupport\nVector', ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.8), fontsize=10)

# Second level
ax.text(0.125, 0.5, r'$\xi_i = 0$' + '\nOn Margin', ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8), fontsize=10)

ax.text(0.375, 0.5, r'$\xi_i > 0$' + '\nMisclassified', ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8), fontsize=10)

ax.text(0.625, 0.5, r'$\alpha_i = C$?', ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8), fontsize=10)

ax.text(0.875, 0.5, r'$0 < \alpha_i < C$' + '\nOn Margin', ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8), fontsize=10)

# Third level
ax.text(0.125, 0.3, r'$\mu_i > 0$' + '\n' + r'$0 < \alpha_i < C$', ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8), fontsize=9)

ax.text(0.375, 0.3, r'$\mu_i = 0$' + '\n' + r'$\alpha_i = C$', ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8), fontsize=9)

ax.text(0.625, 0.3, r'Yes' + '\n' + r'$\xi_i > 0$' + '\nMisclassified', ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8), fontsize=9)

ax.text(0.875, 0.3, r'No' + '\n' + r'$\xi_i = 0$' + '\nOn Margin', ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8), fontsize=9)

# Fourth level
ax.text(0.125, 0.1, r'$\mu_i = 0$' + '\n' + r'$\alpha_i = C$', ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8), fontsize=9)

ax.text(0.375, 0.1, r'$\mu_i = 0$' + '\n' + r'$\alpha_i = C$', ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8), fontsize=9)

ax.text(0.625, 0.1, r'$\mu_i = 0$' + '\n' + r'$\alpha_i = C$', ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8), fontsize=9)

ax.text(0.875, 0.1, r'$\mu_i = 0$' + '\n' + r'$\alpha_i = C$', ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8), fontsize=9)

# Draw connecting lines
lines = [
    ((0.5, 0.9), (0.25, 0.7)),  # Root to Yes
    ((0.5, 0.9), (0.75, 0.7)),  # Root to No
    ((0.25, 0.7), (0.125, 0.5)), # Yes to ξᵢ = 0
    ((0.25, 0.7), (0.375, 0.5)), # Yes to ξᵢ > 0
    ((0.75, 0.7), (0.625, 0.5)), # No to αᵢ = C?
    ((0.75, 0.7), (0.875, 0.5)), # No to 0 < αᵢ < C
    ((0.125, 0.5), (0.125, 0.3)), # ξᵢ = 0 to μᵢ > 0
    ((0.375, 0.5), (0.375, 0.3)), # ξᵢ > 0 to μᵢ = 0
    ((0.625, 0.5), (0.625, 0.3)), # αᵢ = C? to Yes
    ((0.875, 0.5), (0.875, 0.3)), # 0 < αᵢ < C to No
    ((0.125, 0.3), (0.125, 0.1)), # μᵢ > 0 to μᵢ = 0
    ((0.375, 0.3), (0.375, 0.1)), # μᵢ = 0 to μᵢ = 0
    ((0.625, 0.3), (0.625, 0.1)), # Yes to μᵢ = 0
    ((0.875, 0.3), (0.875, 0.1)), # No to μᵢ = 0
]

for start, end in lines:
    ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', alpha=0.5)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'decision_tree.png'), dpi=300, bbox_inches='tight')

# Create a simple SVM margin visualization
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Generate sample data
np.random.seed(42)
n_samples = 50

# Create two classes with some overlap
class1_x = np.random.normal(2, 1, n_samples)
class1_y = np.random.normal(2, 1, n_samples)
class2_x = np.random.normal(4, 1, n_samples)
class2_y = np.random.normal(4, 1, n_samples)

# Add some overlap
class1_x = np.concatenate([class1_x, np.random.normal(3, 0.5, 10)])
class1_y = np.concatenate([class1_y, np.random.normal(3, 0.5, 10)])
class2_x = np.concatenate([class2_x, np.random.normal(3, 0.5, 10)])
class2_y = np.concatenate([class2_y, np.random.normal(3, 0.5, 10)])

# Plot the data points
ax.scatter(class1_x, class1_y, c='blue', s=50, alpha=0.7, label='Class 1')
ax.scatter(class2_x, class2_y, c='red', s=50, alpha=0.7, label='Class 2')

# Draw a simple decision boundary (diagonal line)
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()
x_line = np.linspace(x_min, x_max, 100)
y_line = x_line  # Simple diagonal boundary

# Plot decision boundary
ax.plot(x_line, y_line, 'g-', linewidth=2, label='Decision Boundary')

# Draw margin lines
margin = 0.5
ax.plot(x_line, y_line + margin, 'g--', linewidth=1, alpha=0.7, label='Margin')
ax.plot(x_line, y_line - margin, 'g--', linewidth=1, alpha=0.7)

# Highlight support vectors (points close to the boundary)
support_vectors_x = []
support_vectors_y = []
support_vectors_color = []

for i in range(len(class1_x)):
    dist = abs(class1_x[i] - class1_y[i]) / np.sqrt(2)
    if dist < 1.0:
        support_vectors_x.append(class1_x[i])
        support_vectors_y.append(class1_y[i])
        support_vectors_color.append('blue')

for i in range(len(class2_x)):
    dist = abs(class2_x[i] - class2_y[i]) / np.sqrt(2)
    if dist < 1.0:
        support_vectors_x.append(class2_x[i])
        support_vectors_y.append(class2_y[i])
        support_vectors_color.append('red')

# Plot support vectors with larger markers
ax.scatter(support_vectors_x, support_vectors_y, c=support_vectors_color, s=100, 
           edgecolors='black', linewidth=2, alpha=0.8, label='Support Vectors')

# Shade the margin region
ax.fill_between(x_line, y_line - margin, y_line + margin, alpha=0.2, color='green')

ax.set_xlabel(r'Feature 1')
ax.set_ylabel(r'Feature 2')
ax.set_title(r'Soft Margin SVM Visualization')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'svm_margin_visualization.png'), dpi=300, bbox_inches='tight')

print(f"\nAll visualizations saved to: {save_dir}")

# Print summary of key results
print("\n" + "=" * 80)
print("SUMMARY OF KKT CONDITIONS")
print("=" * 80)

print("\nKey Results:")
print("1. Stationarity Conditions:")
print("   • ∂L/∂w = 0 → w = ∑αᵢyᵢxᵢ")
print("   • ∂L/∂b = 0 → ∑αᵢyᵢ = 0")
print("   • ∂L/∂ξᵢ = 0 → αᵢ + μᵢ = C")

print("\n2. Constraint ∑αᵢyᵢ = 0:")
print("   • Derived directly from ∂L/∂b = 0")
print("   • Ensures the bias term is properly constrained")

print("\n3. Relationship αᵢ + μᵢ = C:")
print("   • Derived from ∂L/∂ξᵢ = 0 for each i")
print("   • Links Lagrange multipliers for different constraints")

print("\n4. Bounds 0 ≤ αᵢ ≤ C:")
print("   • Lower bound: αᵢ ≥ 0 (non-negativity)")
print("   • Upper bound: αᵢ ≤ C (from αᵢ + μᵢ = C and μᵢ ≥ 0)")

print("\n5. Point Classification:")
print("   • αᵢ = 0: Correctly classified, not support vector")
print("   • 0 < αᵢ < C: Support vector on margin")
print("   • αᵢ = C, ξᵢ > 0: Misclassified support vector")
print("   • αᵢ = C, ξᵢ = 0: Support vector on margin")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
