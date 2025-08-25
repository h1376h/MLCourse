import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_1_Quiz_5")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("QUESTION 5: ANALYTICAL SOLUTION OF DUAL SVM PROBLEM")
print("=" * 80)

# Dataset
X = np.array([
    [0, 1],   # x1
    [1, 0],   # x2
    [-1, -1]  # x3
])

y = np.array([1, 1, -1])  # Labels

print("\nDataset:")
for i in range(len(X)):
    print(f"x_{i+1} = {X[i]}, y_{i+1} = {y[i]:+d}")

print("\n" + "="*50)
print("STEP 1: SET UP THE DUAL OPTIMIZATION PROBLEM")
print("="*50)

# Compute kernel matrix K_ij = y_i * y_j * x_i^T * x_j
K = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        K[i, j] = y[i] * y[j] * np.dot(X[i], X[j])

print("\nKernel Matrix K_ij = y_i * y_j * x_i^T * x_j:")
print("K =")
for i in range(3):
    row_str = "["
    for j in range(3):
        row_str += f"{K[i,j]:6.1f}"
        if j < 2:
            row_str += ", "
    row_str += "]"
    print(f"    {row_str}")

print("\nDetailed calculations:")
for i in range(3):
    for j in range(3):
        dot_product = np.dot(X[i], X[j])
        print(f"K_{i+1}{j+1} = y_{i+1} * y_{j+1} * x_{i+1}^T * x_{j+1} = {y[i]:+d} * {y[j]:+d} * {dot_product:4.1f} = {K[i,j]:6.1f}")

print("\nDual Problem:")
print("maximize: Σ α_i - (1/2) Σ_i Σ_j α_i α_j K_ij")
print("subject to: Σ α_i y_i = 0, α_i ≥ 0")

print("\n" + "="*50)
print("STEP 2: SOLVE THE DUAL PROBLEM ANALYTICALLY")
print("="*50)

print("Constraint: α1*y1 + α2*y2 + α3*y3 = α1*(1) + α2*(1) + α3*(-1) = α1 + α2 - α3 = 0")
print("Therefore: α3 = α1 + α2")

print("\n" + "-"*60)
print("DETAILED OBJECTIVE FUNCTION EXPANSION")
print("-"*60)

print("\nThe dual objective function is:")
print("L(α1, α2, α3) = α1 + α2 + α3 - (1/2) * [α1²*K11 + α2²*K22 + α3²*K33 + 2*α1*α2*K12 + 2*α1*α3*K13 + 2*α2*α3*K23]")

print("\nSubstituting the kernel matrix values:")
print("L(α1, α2, α3) = α1 + α2 + α3 - (1/2) * [α1²*1 + α2²*1 + α3²*2 + 2*α1*α2*0 + 2*α1*α3*1 + 2*α2*α3*1]")
print("L(α1, α2, α3) = α1 + α2 + α3 - (1/2) * [α1² + α2² + 2*α3² + 2*α1*α3 + 2*α2*α3]")

print("\nNow substitute α3 = α1 + α2:")
print("L(α1, α2) = α1 + α2 + (α1 + α2) - (1/2) * [α1² + α2² + 2*(α1 + α2)² + 2*α1*(α1 + α2) + 2*α2*(α1 + α2)]")

print("\nExpand the quadratic terms step by step:")
print("1. (α1 + α2)² = α1² + 2*α1*α2 + α2²")
print("2. 2*(α1 + α2)² = 2*α1² + 4*α1*α2 + 2*α2²")
print("3. 2*α1*(α1 + α2) = 2*α1² + 2*α1*α2")
print("4. 2*α2*(α1 + α2) = 2*α1*α2 + 2*α2²")

print("\nSubstituting these expansions:")
print("L(α1, α2) = 2*α1 + 2*α2 - (1/2) * [α1² + α2² + 2*α1² + 4*α1*α2 + 2*α2² + 2*α1² + 2*α1*α2 + 2*α1*α2 + 2*α2²]")

print("\nCollecting like terms:")
print("L(α1, α2) = 2*α1 + 2*α2 - (1/2) * [α1² + 2*α1² + 2*α1² + α2² + 2*α2² + 2*α2² + 4*α1*α2 + 2*α1*α2 + 2*α1*α2]")
print("L(α1, α2) = 2*α1 + 2*α2 - (1/2) * [5*α1² + 5*α2² + 8*α1*α2]")
print("L(α1, α2) = 2*α1 + 2*α2 - (5/2)*α1² - (5/2)*α2² - 4*α1*α2")

print("\n" + "-"*60)
print("FINDING CRITICAL POINTS")
print("-"*60)

print("\nTaking partial derivatives:")
print("∂L/∂α1 = 2 - 5*α1 - 4*α2")
print("∂L/∂α2 = 2 - 5*α2 - 4*α1")

print("\nSetting partial derivatives to zero:")
print("2 - 5*α1 - 4*α2 = 0  →  5*α1 + 4*α2 = 2  (Equation 1)")
print("2 - 5*α2 - 4*α1 = 0  →  4*α1 + 5*α2 = 2  (Equation 2)")

print("\n" + "-"*60)
print("SOLVING THE LINEAR SYSTEM")
print("-"*60)

print("\nWe have the system:")
print("5*α1 + 4*α2 = 2  (Equation 1)")
print("4*α1 + 5*α2 = 2  (Equation 2)")

print("\nMethod 1: Elimination")
print("Multiply Equation 1 by 4:  20*α1 + 16*α2 = 8")
print("Multiply Equation 2 by 5:  20*α1 + 25*α2 = 10")
print("Subtract:  -9*α2 = -2")
print("Therefore: α2 = 2/9")

print("\nSubstitute α2 = 2/9 back into Equation 1:")
print("5*α1 + 4*(2/9) = 2")
print("5*α1 + 8/9 = 2")
print("5*α1 = 2 - 8/9 = 18/9 - 8/9 = 10/9")
print("α1 = (10/9)/5 = 2/9")

print("\nMethod 2: Matrix solution")
print("The system can be written as:")
print("[[5, 4], [4, 5]] * [[α1], [α2]] = [[2], [2]]")

A = np.array([[5, 4], [4, 5]])
b = np.array([2, 2])
alpha_solution = np.linalg.solve(A, b)

print(f"\nMatrix solution: α1 = {alpha_solution[0]:.6f}, α2 = {alpha_solution[1]:.6f}")
print(f"Fractional form: α1 = {alpha_solution[0]}, α2 = {alpha_solution[1]}")

print("\n" + "-"*60)
print("VERIFICATION OF SOLUTION")
print("-"*60)

# Calculate the optimal solution
alpha_optimal = np.array([alpha_solution[0], alpha_solution[1], alpha_solution[0] + alpha_solution[1]])

print(f"\nOptimal solution:")
for i in range(3):
    print(f"α_{i+1}* = {alpha_optimal[i]:.6f}")

print("\nVerifying constraints:")
print(f"α1 = {alpha_optimal[0]:.6f} ≥ 0? {alpha_optimal[0] >= 0} ✓")
print(f"α2 = {alpha_optimal[1]:.6f} ≥ 0? {alpha_optimal[1] >= 0} ✓")
print(f"α3 = {alpha_optimal[2]:.6f} ≥ 0? {alpha_optimal[2] >= 0} ✓")

constraint_sum = np.sum(alpha_optimal * y)
print(f"Constraint check: Σ α_i y_i = {constraint_sum:.6f} ≈ 0? {abs(constraint_sum) < 1e-10} ✓")

print("\nVerifying the solution satisfies the original equations:")
print(f"Equation 1: 5*α1 + 4*α2 = 5*{alpha_optimal[0]:.6f} + 4*{alpha_optimal[1]:.6f} = {5*alpha_optimal[0] + 4*alpha_optimal[1]:.6f} = 2? {abs(5*alpha_optimal[0] + 4*alpha_optimal[1] - 2) < 1e-10} ✓")
print(f"Equation 2: 4*α1 + 5*α2 = 4*{alpha_optimal[0]:.6f} + 5*{alpha_optimal[1]:.6f} = {4*alpha_optimal[0] + 5*alpha_optimal[1]:.6f} = 2? {abs(4*alpha_optimal[0] + 5*alpha_optimal[1] - 2) < 1e-10} ✓")

# Calculate objective value
obj_value = 2*alpha_optimal[0] + 2*alpha_optimal[1] - (5/2)*alpha_optimal[0]**2 - (5/2)*alpha_optimal[1]**2 - 4*alpha_optimal[0]*alpha_optimal[1]
print(f"\nObjective value: L({alpha_optimal[0]:.6f}, {alpha_optimal[1]:.6f}) = {obj_value:.6f}")

print("\n" + "="*50)
print("STEP 3: CALCULATE THE OPTIMAL WEIGHT VECTOR")
print("="*50)

# Calculate w* = Σ α_i y_i x_i
w_optimal = np.zeros(2)
for i in range(3):
    w_optimal += alpha_optimal[i] * y[i] * X[i]

print("w* = Σ α_i* y_i x_i")
for i in range(3):
    contribution = alpha_optimal[i] * y[i] * X[i]
    print(f"   + α_{i+1}* * y_{i+1} * x_{i+1} = {alpha_optimal[i]:.6f} * {y[i]:+d} * {X[i]} = {contribution}")

print(f"\nw* = {w_optimal}")

print("\n" + "="*50)
print("STEP 4: FIND THE BIAS TERM b*")
print("="*50)

# Find support vectors (α_i > 0)
support_vectors = []
for i in range(3):
    if alpha_optimal[i] > 1e-6:  # Numerical tolerance
        support_vectors.append(i)
        print(f"Point {i+1} is a support vector (α_{i+1}* = {alpha_optimal[i]:.6f} > 0)")

# Calculate b using support vector conditions
b_values = []
for sv in support_vectors:
    # For support vectors: y_i(w^T x_i + b) = 1
    # Therefore: b = y_i - w^T x_i
    b_val = y[sv] - np.dot(w_optimal, X[sv])
    b_values.append(b_val)
    print(f"Using support vector {sv+1}: b = y_{sv+1} - w*^T x_{sv+1} = {y[sv]} - {np.dot(w_optimal, X[sv]):.6f} = {b_val:.6f}")

b_optimal = np.mean(b_values)
print(f"\nb* = {b_optimal:.6f}")

print("\n" + "="*50)
print("STEP 5: WRITE THE FINAL DECISION FUNCTION")
print("="*50)

print("Decision function: f(x) = sign(w*^T x + b*)")
print(f"f(x) = sign({w_optimal[0]:.6f} * x1 + {w_optimal[1]:.6f} * x2 + {b_optimal:.6f})")

# Verify the solution
print("\nVerification - checking all training points:")
for i in range(3):
    decision_value = np.dot(w_optimal, X[i]) + b_optimal
    prediction = np.sign(decision_value)
    margin = y[i] * decision_value
    print(f"Point {i+1}: f(x_{i+1}) = {decision_value:.6f}, prediction = {prediction:+.0f}, margin = {margin:.6f}")

print("\n" + "="*50)
print("STEP 6: VISUALIZATION")
print("="*50)

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Data points and decision boundary
x1_range = np.linspace(-2.5, 2.5, 100)
if abs(w_optimal[1]) > 1e-10:
    x2_boundary = -(w_optimal[0] * x1_range + b_optimal) / w_optimal[1]
    ax1.plot(x1_range, x2_boundary, 'k-', linewidth=2, label='Decision Boundary')

    # Plot margin boundaries
    margin_width = 1.0 / np.linalg.norm(w_optimal)
    x2_pos_margin = -(w_optimal[0] * x1_range + b_optimal - 1) / w_optimal[1]
    x2_neg_margin = -(w_optimal[0] * x1_range + b_optimal + 1) / w_optimal[1]
    ax1.plot(x1_range, x2_pos_margin, 'k--', alpha=0.7, label='Positive Margin')
    ax1.plot(x1_range, x2_neg_margin, 'k--', alpha=0.7, label='Negative Margin')

# Plot data points
colors = ['red', 'blue']
markers = ['o', 's']
for i in range(3):
    color_idx = 0 if y[i] == 1 else 1
    marker_size = 150 if i in support_vectors else 100
    edge_width = 3 if i in support_vectors else 1

    ax1.scatter(X[i, 0], X[i, 1], c=colors[color_idx], marker=markers[color_idx],
               s=marker_size, edgecolors='black', linewidth=edge_width,
               label=f'Class {y[i]:+d}' if i < 2 else None)

    # Add point labels
    ax1.annotate(f'x_{i+1}', (X[i, 0], X[i, 1]), xytext=(10, 10),
                textcoords='offset points', fontsize=12, fontweight='bold')

ax1.set_xlabel('$x_1$', fontsize=14)
ax1.set_ylabel('$x_2$', fontsize=14)
ax1.set_title('SVM Solution: Data Points and Decision Boundary', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.axis('equal')
ax1.set_xlim(-2.5, 2.5)
ax1.set_ylim(-2.5, 2.5)

# Plot 2: Dual variables
ax2.bar(range(1, 4), alpha_optimal, color=['lightblue', 'lightgreen', 'lightcoral'])
ax2.set_xlabel('Data Point Index', fontsize=14)
ax2.set_ylabel('$\\alpha_i^*$', fontsize=14)
ax2.set_title('Optimal Dual Variables', fontsize=14)
ax2.set_xticks(range(1, 4))
ax2.set_xticklabels([f'$\\alpha_{i}^*$' for i in range(1, 4)])
ax2.grid(True, alpha=0.3)

# Add value labels on bars
for i, v in enumerate(alpha_optimal):
    ax2.text(i+1, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'svm_analytical_solution.png'), dpi=300, bbox_inches='tight')

print(f"Visualization saved to: {os.path.join(save_dir, 'svm_analytical_solution.png')}")

print("\n" + "="*50)
print("STRATEGIC GAME INTERPRETATION")
print("="*50)

print("In the strategy game context:")
print("- Red Army units (Class +1) are at (0,1) and (1,0)")
print("- Blue Army unit (Class -1) is at (-1,-1)")
print(f"- Optimal defensive wall equation: {w_optimal[0]:.3f}x₁ + {w_optimal[1]:.3f}x₂ + {b_optimal:.3f} = 0")

margin_width = 2.0 / np.linalg.norm(w_optimal)
print(f"- Safety margin (distance between armies): {margin_width:.3f} units")

print(f"- Support vectors (critical positions): Points {[i+1 for i in support_vectors]}")
print("- For maximum advantage, place additional Red unit away from the decision boundary")
print("  but in the positive region (where w^T x + b > 0)")

print("\n" + "="*80)
print("SOLUTION COMPLETE")
print("="*80)