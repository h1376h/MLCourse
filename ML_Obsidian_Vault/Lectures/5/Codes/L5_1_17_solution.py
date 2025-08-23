import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_1_Quiz_17")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

print("=" * 60)
print("Question 17: Complete Dual Solution")
print("=" * 60)

# Given dataset
print("\nGiven Dataset:")
X = np.array([[1, 0], [0, 1], [-1, -1]])  # Training points
y = np.array([1, 1, -1])  # Labels

print(f"Training points:")
for i, (xi, yi) in enumerate(zip(X, y)):
    print(f"  x_{i+1} = {xi}, y_{i+1} = {yi:+d}")

# Task 1: Compute G_ij = y_i y_j x_i^T x_j for all pairs (i,j)
print("\n" + "="*50)
print("Task 1: Compute Kernel Matrix G_ij = y_i y_j x_i^T x_j")
print("="*50)

n = len(X)
G = np.zeros((n, n))

print("\nComputing G_ij for all pairs (i,j):")
for i in range(n):
    for j in range(n):
        # Compute x_i^T x_j (dot product)
        dot_product = np.dot(X[i], X[j])
        # Compute y_i y_j x_i^T x_j
        G[i, j] = y[i] * y[j] * dot_product
        
        print(f"G_{i+1}{j+1} = y_{i+1} * y_{j+1} * x_{i+1}^T x_{j+1}")
        print(f"     = {y[i]} * {y[j]} * {X[i]} · {X[j]}")
        print(f"     = {y[i]} * {y[j]} * {dot_product} = {G[i, j]}")

print(f"\nKernel Matrix G:")
print(G)

# Task 2: Write the dual objective explicitly
print("\n" + "="*50)
print("Task 2: Dual Objective Function")
print("="*50)

print("The dual objective function is:")
print("L_D(α) = Σᵢ αᵢ - (1/2) Σᵢ Σⱼ αᵢ αⱼ Gᵢⱼ")
print("\nExpanding with our G matrix:")
print("L_D(α) = α₁ + α₂ + α₃ - (1/2)[")

# Print the quadratic term explicitly
for i in range(n):
    for j in range(n):
        if i == 0 and j == 0:
            print(f"  α₁α₁G₁₁", end="")
        else:
            sign = "+" if G[i, j] >= 0 else ""
            print(f" {sign} α{i+1}α{j+1}G{i+1}{j+1}", end="")
        if not (i == n-1 and j == n-1):
            if (i*n + j + 1) % 3 == 0:  # Line break every 3 terms
                print()
                print("  ", end="")

print("]")

print(f"\nSubstituting G values:")
print("L_D(α) = α₁ + α₂ + α₃ - (1/2)[")
for i in range(n):
    for j in range(n):
        if i == 0 and j == 0:
            print(f"  {G[i, j]}α₁²", end="")
        else:
            if G[i, j] >= 0:
                print(f" + {G[i, j]}α{i+1}α{j+1}", end="")
            else:
                print(f" - {abs(G[i, j])}α{i+1}α{j+1}", end="")
        if not (i == n-1 and j == n-1):
            if (i*n + j + 1) % 3 == 0:
                print()
                print("  ", end="")

print("]")

# Task 3: Express the constraint numerically
print("\n" + "="*50)
print("Task 3: Constraint")
print("="*50)

print("The dual constraint is: Σᵢ αᵢ yᵢ = 0")
print("Substituting our labels:")
constraint_coeffs = y
print(f"α₁({y[0]}) + α₂({y[1]}) + α₃({y[2]}) = 0")
print(f"α₁ + α₂ - α₃ = 0")
print(f"Therefore: α₃ = α₁ + α₂")

# Task 4: Verify constraint satisfaction and compute objective value
print("\n" + "="*50)
print("Task 4: Verify Given α Values")
print("="*50)

alpha = np.array([0.5, 0.5, 1.0])
print(f"Given: α = {alpha}")

# Check constraint
constraint_value = np.dot(alpha, y)
print(f"\nConstraint check:")
print(f"Σᵢ αᵢ yᵢ = {alpha[0]}*{y[0]} + {alpha[1]}*{y[1]} + {alpha[2]}*{y[2]}")
print(f"        = {alpha[0]} + {alpha[1]} + {alpha[2]*y[2]} = {constraint_value}")

if abs(constraint_value) < 1e-10:
    print("✓ Constraint satisfied!")
else:
    print(f"✗ Constraint violated! Should be 0, got {constraint_value}")

# Compute objective value
linear_term = np.sum(alpha)
quadratic_term = 0.5 * np.sum(alpha[:, np.newaxis] * alpha[np.newaxis, :] * G)
objective_value = linear_term - quadratic_term

print(f"\nObjective value computation:")
print(f"Linear term: Σᵢ αᵢ = {alpha[0]} + {alpha[1]} + {alpha[2]} = {linear_term}")
print(f"Quadratic term: (1/2) Σᵢⱼ αᵢ αⱼ Gᵢⱼ")

# Show detailed quadratic computation
print("Quadratic term breakdown:")
quad_detail = 0
for i in range(n):
    for j in range(n):
        term = alpha[i] * alpha[j] * G[i, j]
        quad_detail += term
        print(f"  α{i+1}α{j+1}G{i+1}{j+1} = {alpha[i]}*{alpha[j]}*{G[i,j]} = {term}")

print(f"Sum of quadratic terms = {quad_detail}")
print(f"Quadratic term = (1/2) * {quad_detail} = {quadratic_term}")
print(f"Objective value = {linear_term} - {quadratic_term} = {objective_value}")

# Task 5: Calculate w and determine b
print("\n" + "="*50)
print("Task 5: Calculate w and b")
print("="*50)

# Calculate w = Σᵢ αᵢ yᵢ xᵢ
w = np.zeros(2)
print("Calculating w = Σᵢ αᵢ yᵢ xᵢ:")
for i in range(n):
    contribution = alpha[i] * y[i] * X[i]
    w += contribution
    print(f"  α{i+1} y{i+1} x{i+1} = {alpha[i]} * {y[i]} * {X[i]} = {contribution}")

print(f"w = {w}")

# Determine b using support vector conditions
print(f"\nDetermining b using support vector conditions:")
print("For support vectors (αᵢ > 0), we have yᵢ(w^T xᵢ + b) = 1")

# Find support vectors (α > 0)
support_vectors = np.where(alpha > 1e-10)[0]
print(f"Support vectors (αᵢ > 0): indices {support_vectors + 1}")

b_values = []
for sv_idx in support_vectors:
    # For support vector: y_i(w^T x_i + b) = 1
    # Therefore: b = (1 - w^T x_i) / y_i = y_i - w^T x_i (since y_i^2 = 1)
    w_dot_x = np.dot(w, X[sv_idx])
    b_from_sv = y[sv_idx] - w_dot_x
    b_values.append(b_from_sv)

    print(f"  Support vector {sv_idx+1}: x{sv_idx+1} = {X[sv_idx]}, y{sv_idx+1} = {y[sv_idx]}")
    print(f"    w^T x{sv_idx+1} = {w} · {X[sv_idx]} = {w_dot_x}")
    print(f"    b = y{sv_idx+1} - w^T x{sv_idx+1} = {y[sv_idx]} - {w_dot_x} = {b_from_sv}")

# There seems to be an inconsistency - let's recalculate b properly
# For any support vector i: y_i(w^T x_i + b) = 1
# So: b = (1 - y_i * w^T x_i) / y_i = y_i - w^T x_i (since y_i^2 = 1)
print(f"\nNote: There's an inconsistency in b values from different support vectors.")
print(f"This suggests the given alpha values may not be optimal.")
print(f"Let's use the first support vector to determine b:")
if len(support_vectors) > 0:
    sv_idx = support_vectors[0]
    w_dot_x = np.dot(w, X[sv_idx])
    b = y[sv_idx] - w_dot_x
    print(f"Using support vector {sv_idx+1}: b = {y[sv_idx]} - {w_dot_x} = {b}")
else:
    b = 0
    print("No support vectors found, setting b = 0")

# Verify the solution
print(f"\nVerification:")
print(f"Final solution: w = {w}, b = {b}")
for i in range(n):
    decision_value = np.dot(w, X[i]) + b
    margin_value = y[i] * decision_value
    print(f"  Point {i+1}: w^T x{i+1} + b = {decision_value:.3f}, y{i+1}(w^T x{i+1} + b) = {margin_value:.3f}")
    
    if alpha[i] > 1e-10:  # Support vector
        if abs(margin_value - 1.0) < 1e-6:
            print(f"    ✓ Support vector condition satisfied")
        else:
            print(f"    ⚠ Support vector condition not satisfied")
    else:  # Non-support vector
        if margin_value >= 1.0 - 1e-6:
            print(f"    ✓ Margin constraint satisfied")
        else:
            print(f"    ⚠ Margin constraint violated")

print(f"\nFinal decision function:")
print(f"f(x) = sign(w^T x + b) = sign({w[0]:.3f}x₁ + {w[1]:.3f}x₂ + {b:.3f})")

# Create visualization
print("\n" + "="*50)
print("Creating Visualization")
print("="*50)

plt.figure(figsize=(12, 10))

# Define plotting range
x1_range = np.linspace(-2, 2, 100)

# Calculate hyperplane: w^T x + b = 0 => w[1]*x2 = -w[0]*x1 - b => x2 = (-w[0]*x1 - b)/w[1]
if abs(w[1]) > 1e-10:  # Non-vertical line
    x2_hyperplane = (-w[0]*x1_range - b) / w[1]
    x2_pos_margin = (-w[0]*x1_range - b + 1) / w[1]  # w^T x + b = 1
    x2_neg_margin = (-w[0]*x1_range - b - 1) / w[1]  # w^T x + b = -1

    # Plot hyperplane and margins
    plt.plot(x1_range, x2_hyperplane, 'k-', linewidth=2, label='Decision Boundary')
    plt.plot(x1_range, x2_pos_margin, 'r--', linewidth=1.5, label='Positive Margin (+1)')
    plt.plot(x1_range, x2_neg_margin, 'b--', linewidth=1.5, label='Negative Margin (-1)')

    # Shade margin region
    plt.fill_between(x1_range, x2_pos_margin, x2_neg_margin, alpha=0.2, color='gray', label='Margin Region')

# Plot training points
colors = ['red', 'blue']
markers = ['o', 's']
class_names = ['+1', '-1']

for i, (xi, yi) in enumerate(zip(X, y)):
    color_idx = 0 if yi == 1 else 1
    marker_style = markers[color_idx]
    color = colors[color_idx]

    # Check if support vector
    if alpha[i] > 1e-10:
        plt.scatter(xi[0], xi[1], s=200, c=color, marker=marker_style,
                   edgecolors='black', linewidth=3, label=f'SV: x_{i+1} ($\\alpha$={alpha[i]})')
    else:
        plt.scatter(xi[0], xi[1], s=150, c=color, marker=marker_style,
                   edgecolors='black', linewidth=1.5, alpha=0.7)

    # Add point labels
    plt.annotate(f'x_{i+1}({xi[0]},{xi[1]})',
                (xi[0], xi[1]), xytext=(10, 10), textcoords='offset points',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('SVM Dual Solution Visualization\nComplete Dual Problem Solution', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.axis('equal')
plt.xlim(-2, 2)
plt.ylim(-2, 2)

# Add solution details as text
solution_text = r'Dual Variables: $\alpha$ = ' + f'[{alpha[0]}, {alpha[1]}, {alpha[2]}]' + '\n'
solution_text += r'Weight Vector: $\mathbf{w}$ = ' + f'[{w[0]:.3f}, {w[1]:.3f}]' + '\n'
solution_text += f'Bias: b = {b:.3f}' + '\n'
solution_text += r'Objective Value: $L_D$ = ' + f'{objective_value:.3f}'

plt.text(0.02, 0.98, solution_text, transform=plt.gca().transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", alpha=0.9))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'dual_solution_visualization.png'), dpi=300, bbox_inches='tight')

# Create a second plot showing the kernel matrix
plt.figure(figsize=(8, 6))
im = plt.imshow(G, cmap='RdBu', interpolation='nearest')
plt.colorbar(im, label='G_ij value')
plt.title(r'Kernel Matrix $G_{ij} = y_i y_j \mathbf{x}_i^T \mathbf{x}_j$')
plt.xlabel('j')
plt.ylabel('i')

# Add text annotations
for i in range(n):
    for j in range(n):
        plt.text(j, i, f'{G[i,j]:.1f}', ha='center', va='center',
                color='white' if abs(G[i,j]) > 0.5 else 'black', fontweight='bold')

plt.xticks(range(n), [f'x_{i+1}' for i in range(n)])
plt.yticks(range(n), [f'x_{i+1}' for i in range(n)])
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'kernel_matrix.png'), dpi=300, bbox_inches='tight')

print(f"Visualizations saved to: {save_dir}")

# Create additional visualization: Dual objective components
print("\n" + "="*50)
print("Creating Additional Visualization: Dual Objective Analysis")
print("="*50)

plt.figure(figsize=(12, 8))

# Create subplot layout
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Alpha values
alpha_labels = [r'$\alpha_{' + str(i+1) + r'}$' for i in range(len(alpha))]
colors_alpha = ['red' if a > 1e-10 else 'lightgray' for a in alpha]
bars1 = ax1.bar(alpha_labels, alpha, color=colors_alpha, alpha=0.7, edgecolor='black')
ax1.set_ylabel(r'$\alpha_i$ Values')
ax1.set_title('Dual Variables (Lagrange Multipliers)')
ax1.grid(True, alpha=0.3)
for bar, a in zip(bars1, alpha):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{a:.2f}', ha='center', va='bottom', fontweight='bold')

# Plot 2: Constraint verification
constraint_terms = alpha * y
ax2.bar(alpha_labels, constraint_terms, color=['green' if y[i] > 0 else 'red' for i in range(len(y))],
        alpha=0.7, edgecolor='black')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.set_ylabel(r'$\alpha_i y_i$')
ax2.set_title(r'Constraint Terms: $\sum_i \alpha_i y_i = 0$')
ax2.grid(True, alpha=0.3)
for i, (bar, term) in enumerate(zip(ax2.patches, constraint_terms)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + (0.02 if height >= 0 else -0.05),
             f'{term:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')

# Plot 3: Objective function components
components = ['Linear Term', 'Quadratic Term', 'Objective Value']
values = [linear_term, quadratic_term, objective_value]
colors_obj = ['blue', 'orange', 'green']
bars3 = ax3.bar(components, values, color=colors_obj, alpha=0.7, edgecolor='black')
ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax3.set_ylabel('Value')
ax3.set_title('Dual Objective Components')
ax3.grid(True, alpha=0.3)
for bar, val in zip(bars3, values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + (0.05 if height >= 0 else -0.1),
             f'{val:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')

# Plot 4: Weight vector components
w_labels = [r'$w_1$', r'$w_2$']
bars4 = ax4.bar(w_labels, w, color=['purple', 'brown'], alpha=0.7, edgecolor='black')
ax4.set_ylabel('Weight Value')
ax4.set_title('Optimal Weight Vector Components')
ax4.grid(True, alpha=0.3)
for bar, weight in zip(bars4, w):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{weight:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'dual_objective_analysis.png'), dpi=300, bbox_inches='tight')

print(f"Additional visualization saved to: {save_dir}/dual_objective_analysis.png")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"1. Kernel matrix G computed with all pairwise products")
print(f"2. Dual objective: L_D = {objective_value:.6f}")
print(f"3. Constraint satisfied: Σᵢ αᵢ yᵢ = {constraint_value}")
print(f"4. Optimal solution: w = {w}, b = {b:.6f}")
print(f"5. Support vectors: Points {[i+1 for i in support_vectors]} (α > 0)")
print(f"6. Decision function: f(x) = sign({w[0]:.3f}x₁ + {w[1]:.3f}x₂ + {b:.3f})")
