import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
import sympy as sp
from sympy import symbols, Matrix, diff, simplify, expand

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_1_Quiz_21")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX for matplotlib
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("QUESTION 21: COMPLETE DUAL DERIVATION FROM FIRST PRINCIPLES")
print("=" * 80)

# Given dataset
X = np.array([
    [1, 1],   # x1
    [2, 2],   # x2
    [-1, 0],  # x3
    [0, -1]   # x4
])

y = np.array([1, 1, -1, -1])  # Labels

print("Given dataset:")
for i in range(len(X)):
    print(f"x_{i+1} = {X[i]}, y_{i+1} = {y[i]:2d}")

print("\n" + "="*80)
print("STEP 1: WRITE THE COMPLETE PRIMAL OPTIMIZATION PROBLEM")
print("="*80)

print("The primal SVM optimization problem is:")
print()
print("minimize    (1/2)||w||²")
print("subject to  y_i(w^T x_i + b) ≥ 1,  i = 1, 2, 3, 4")
print()
print("Explicitly for our dataset:")
print("minimize    (1/2)(w₁² + w₂²)")
print("subject to:")
for i in range(len(X)):
    sign = "≥" 
    print(f"  {y[i]:2d}(w₁·{X[i,0]:2d} + w₂·{X[i,1]:2d} + b) {sign} 1")

print("\nExpanding the constraints:")
for i in range(len(X)):
    if y[i] == 1:
        print(f"  w₁·{X[i,0]:2d} + w₂·{X[i,1]:2d} + b ≥ 1")
    else:
        print(f"  -(w₁·{X[i,0]:2d} + w₂·{X[i,1]:2d} + b) ≥ 1")
        print(f"  w₁·{X[i,0]:2d} + w₂·{X[i,1]:2d} + b ≤ -1")

print("\n" + "="*80)
print("STEP 2: FORM THE LAGRANGIAN")
print("="*80)

print("The Lagrangian function is:")
print("L(w, b, α) = (1/2)||w||² - Σᵢ αᵢ[yᵢ(w^T xᵢ + b) - 1]")
print()
print("Expanding for our dataset:")
print("L(w₁, w₂, b, α₁, α₂, α₃, α₄) = (1/2)(w₁² + w₂²)")

lagrangian_terms = []
for i in range(len(X)):
    constraint = f"y_{i+1}(w₁·{X[i,0]} + w₂·{X[i,1]} + b) - 1"
    constraint_expanded = f"{y[i]}(w₁·{X[i,0]} + w₂·{X[i,1]} + b) - 1"
    lagrangian_terms.append(f"α_{i+1}[{constraint_expanded}]")
    print(f"                                    - α_{i+1}[{constraint_expanded}]")

print("\n" + "="*80)
print("STEP 3: DERIVE STATIONARITY CONDITIONS")
print("="*80)

print("For optimality, we need:")
print("∇_w L = 0  and  ∂L/∂b = 0")

# Define symbolic variables
w1, w2, b = symbols('w1 w2 b')
alpha1, alpha2, alpha3, alpha4 = symbols('alpha1 alpha2 alpha3 alpha4')

# Define the Lagrangian symbolically
L = sp.Rational(1,2)*(w1**2 + w2**2)

# Add constraint terms
for i in range(len(X)):
    constraint_term = y[i]*(w1*X[i,0] + w2*X[i,1] + b) - 1
    if i == 0:
        L -= alpha1 * constraint_term
    elif i == 1:
        L -= alpha2 * constraint_term
    elif i == 2:
        L -= alpha3 * constraint_term
    elif i == 3:
        L -= alpha4 * constraint_term

print("\nTaking partial derivatives:")

# Compute gradients
dL_dw1 = diff(L, w1)
dL_dw2 = diff(L, w2)
dL_db = diff(L, b)

print(f"∂L/∂w₁ = {dL_dw1} = 0")
print(f"∂L/∂w₂ = {dL_dw2} = 0")
print(f"∂L/∂b = {dL_db} = 0")

print("\nSolving the stationarity conditions:")

# Solve for w in terms of alpha
print("From ∂L/∂w₁ = 0:")
w1_solution = sp.solve(dL_dw1, w1)[0]
print(f"w₁ = {w1_solution}")

print("From ∂L/∂w₂ = 0:")
w2_solution = sp.solve(dL_dw2, w2)[0]
print(f"w₂ = {w2_solution}")

print("From ∂L/∂b = 0:")
print(f"Dual constraint: {dL_db} = 0")
print(f"This gives us: α₁·1 + α₂·1 + α₃·(-1) + α₄·(-1) = 0")
print(f"Simplified: α₁ + α₂ - α₃ - α₄ = 0")

print("\nTherefore:")
print("w₁ = α₁·1·1 + α₂·1·2 + α₃·(-1)·(-1) + α₄·(-1)·0")
print("w₁ = α₁ + 2α₂ + α₃")
print()
print("w₂ = α₁·1·1 + α₂·1·2 + α₃·(-1)·0 + α₄·(-1)·(-1)")
print("w₂ = α₁ + 2α₂ + α₄")

print("\n" + "="*80)
print("STEP 4: SUBSTITUTE STATIONARITY CONDITIONS INTO LAGRANGIAN")
print("="*80)

print("Substituting w₁ and w₂ back into the Lagrangian:")
print("We need to compute the dual objective function.")

# Compute kernel matrix
print("\nFirst, let's compute the kernel matrix K where K_ij = y_i y_j (x_i^T x_j):")

K = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        K[i,j] = y[i] * y[j] * np.dot(X[i], X[j])

print("K = ")
for i in range(4):
    row_str = "    ["
    for j in range(4):
        row_str += f"{K[i,j]:4.0f}"
        if j < 3:
            row_str += ", "
    row_str += "]"
    print(row_str)

print("\nExplicit kernel matrix entries:")
for i in range(4):
    for j in range(4):
        dot_product = np.dot(X[i], X[j])
        print(f"K_{i+1}{j+1} = y_{i+1}·y_{j+1}·(x_{i+1}^T x_{j+1}) = {y[i]}·{y[j]}·{dot_product} = {K[i,j]}")

print("\n" + "="*80)
print("STEP 5: DERIVE THE DUAL PROBLEM")
print("="*80)

print("The dual objective function is:")
print("maximize   Σᵢ αᵢ - (1/2)Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ (xᵢ^T xⱼ)")
print()
print("Expanding with our kernel matrix:")
print("maximize   α₁ + α₂ + α₃ + α₄")
print("           - (1/2)[α₁²·K₁₁ + α₁α₂·K₁₂ + α₁α₃·K₁₃ + α₁α₄·K₁₄")
print("                  + α₂α₁·K₂₁ + α₂²·K₂₂ + α₂α₃·K₂₃ + α₂α₄·K₂₄")
print("                  + α₃α₁·K₃₁ + α₃α₂·K₃₂ + α₃²·K₃₃ + α₃α₄·K₃₄")
print("                  + α₄α₁·K₄₁ + α₄α₂·K₄₂ + α₄α₃·K₄₃ + α₄²·K₄₄]")

print("\nSubstituting the kernel values:")
dual_objective = "α₁ + α₂ + α₃ + α₄ - (1/2)["
terms = []
for i in range(4):
    for j in range(4):
        if i <= j:  # Only include upper triangular + diagonal (symmetric matrix)
            if i == j:
                terms.append(f"α_{i+1}²·{K[i,j]:.0f}")
            else:
                terms.append(f"2α_{i+1}α_{j+1}·{K[i,j]:.0f}")

dual_objective += " + ".join(terms) + "]"
print(dual_objective)

print("\nDetailed expansion of the quadratic term:")
print("(1/2)Σᵢ Σⱼ αᵢ αⱼ Kᵢⱼ = (1/2)[")
print("  α₁²·K₁₁ + α₁α₂·K₁₂ + α₁α₃·K₁₃ + α₁α₄·K₁₄")
print("  + α₂α₁·K₂₁ + α₂²·K₂₂ + α₂α₃·K₂₃ + α₂α₄·K₂₄")
print("  + α₃α₁·K₃₁ + α₃α₂·K₃₂ + α₃²·K₃₃ + α₃α₄·K₃₄")
print("  + α₄α₁·K₄₁ + α₄α₂·K₄₂ + α₄α₃·K₄₃ + α₄²·K₄₄]")

print("\nSubstituting K values:")
print("= (1/2)[α₁²·2 + α₁α₂·4 + α₁α₃·1 + α₁α₄·1")
print("       + α₂α₁·4 + α₂²·8 + α₂α₃·2 + α₂α₄·2")
print("       + α₃α₁·1 + α₃α₂·2 + α₃²·1 + α₃α₄·0")
print("       + α₄α₁·1 + α₄α₂·2 + α₄α₃·0 + α₄²·1]")

print("\nCombining symmetric terms (αᵢαⱼ + αⱼαᵢ = 2αᵢαⱼ for i≠j):")
print("= (1/2)[2α₁² + 2·4α₁α₂ + 2·1α₁α₃ + 2·1α₁α₄")
print("       + 8α₂² + 2·2α₂α₃ + 2·2α₂α₄")
print("       + 1α₃² + 2·0α₃α₄")
print("       + 1α₄²]")

print("\nSimplified:")
print("= (1/2)[2α₁² + 8α₁α₂ + 2α₁α₃ + 2α₁α₄")
print("       + 8α₂² + 4α₂α₃ + 4α₂α₄")
print("       + α₃² + 0α₃α₄")
print("       + α₄²]")

print("\n= α₁² + 4α₁α₂ + α₁α₃ + α₁α₄")
print("  + 4α₂² + 2α₂α₃ + 2α₂α₄")
print("  + (1/2)α₃² + (1/2)α₄²")

print("\nFinal dual problem:")
print("maximize   α₁ + α₂ + α₃ + α₄")
print("           - α₁² - 4α₁α₂ - α₁α₃ - α₁α₄")
print("           - 4α₂² - 2α₂α₃ - 2α₂α₄")
print("           - (1/2)α₃² - (1/2)α₄²")
print()
print("subject to:")
print("  α₁ + α₂ - α₃ - α₄ = 0")
print("  αᵢ ≥ 0,  i = 1, 2, 3, 4")

print("\n" + "="*80)
print("STEP 6: VISUALIZATION")
print("="*80)

# Create visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Dataset visualization
ax1.set_title('Dataset Visualization', fontsize=14)
colors = ['red' if label == -1 else 'blue' for label in y]
markers = ['o' if label == -1 else 's' for label in y]

for i in range(len(X)):
    ax1.scatter(X[i, 0], X[i, 1], c=colors[i], marker=markers[i], s=200, 
               edgecolors='black', linewidth=2, 
               label=f'x_{i+1} (y={y[i]})' if i < 2 else '')
    ax1.annotate(f'$x_{{{i+1}}}$', (X[i, 0], X[i, 1]), xytext=(10, 10),
                textcoords='offset points', fontsize=12)

ax1.set_xlabel(r'$x_1$')
ax1.set_ylabel(r'$x_2$')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xlim(-2, 3)
ax1.set_ylim(-2, 3)

# Plot 2: Kernel matrix heatmap
ax2.set_title(r'Kernel Matrix $K_{ij} = y_i y_j (x_i^T x_j)$', fontsize=14)
im = ax2.imshow(K, cmap='RdBu', aspect='equal')
ax2.set_xticks(range(4))
ax2.set_yticks(range(4))
ax2.set_xticklabels([f'j={i+1}' for i in range(4)])
ax2.set_yticklabels([f'i={i+1}' for i in range(4)])

# Add text annotations
for i in range(4):
    for j in range(4):
        text = ax2.text(j, i, f'{K[i, j]:.0f}', ha="center", va="center", 
                       color="white" if abs(K[i, j]) > 4 else "black", fontsize=12)

plt.colorbar(im, ax=ax2)

# Plot 3: Primal problem structure
ax3.set_title('Primal Problem Structure', fontsize=12)
ax3.axis('off')

primal_text = r"""
Primal Problem:
minimize    $\frac{1}{2}||w||^2$
subject to  $y_1(w_1 \cdot 1 + w_2 \cdot 1 + b) \geq 1$
            $y_2(w_1 \cdot 2 + w_2 \cdot 2 + b) \geq 1$
            $y_3(w_1 \cdot (-1) + w_2 \cdot 0 + b) \geq 1$
            $y_4(w_1 \cdot 0 + w_2 \cdot (-1) + b) \geq 1$

Expanded:
minimize    $\frac{1}{2}(w_1^2 + w_2^2)$
subject to  $w_1 + w_2 + b \geq 1$
            $2w_1 + 2w_2 + b \geq 1$
            $w_1 - b \leq -1$
            $w_2 - b \leq -1$
"""

ax3.text(0.05, 0.95, primal_text, transform=ax3.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

# Plot 4: Dual problem structure
ax4.set_title('Dual Problem Structure', fontsize=12)
ax4.axis('off')

dual_text = r"""
Dual Problem:
maximize    $\alpha_1 + \alpha_2 + \alpha_3 + \alpha_4$
            $- \frac{1}{2}\sum_{ij} \alpha_i\alpha_j K_{ij}$

where K = [2  4  1  1]
          [4  8  2  2]
          [1  2  1  0]
          [1  2  0  1]

subject to  $\alpha_1 + \alpha_2 - \alpha_3 - \alpha_4 = 0$
            $\alpha_i \geq 0, i = 1,2,3,4$

Stationarity conditions:
$w_1 = \alpha_1 + 2\alpha_2 + \alpha_3$
$w_2 = \alpha_1 + 2\alpha_2 + \alpha_4$
"""

ax4.text(0.05, 0.95, dual_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'dual_derivation_complete.png'), dpi=300, bbox_inches='tight')
plt.close()  # Close the figure instead of showing it

print(f"Visualization saved to: {save_dir}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("✓ Primal problem formulated with 4 constraints")
print("✓ Lagrangian formed with 4 Lagrange multipliers")
print("✓ Stationarity conditions derived: w = Σᵢ αᵢyᵢxᵢ, Σᵢ αᵢyᵢ = 0")
print("✓ Kernel matrix computed with explicit entries")
print("✓ Dual problem derived: maximize Σᵢ αᵢ - ½Σᵢⱼ αᵢαⱼKᵢⱼ")
print("✓ Dual constraint: α₁ + α₂ - α₃ - α₄ = 0")
