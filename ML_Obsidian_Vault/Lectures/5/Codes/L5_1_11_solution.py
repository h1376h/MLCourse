import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_1_Quiz_11")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amssymb}'

print("=" * 80)
print("Question 11: Optimization Theory for SVM")
print("=" * 80)

# Task 1: Type of optimization problem
print("\n1. TYPE OF OPTIMIZATION PROBLEM")
print("-" * 50)

print("SVM Primal Formulation:")
print("minimize: (1/2)||w||²")
print("subject to: y_i(w^T x_i + b) ≥ 1, for i = 1, ..., n")
print("\nClassification:")
print("• QUADRATIC PROGRAMMING (QP) problem")
print("• Quadratic objective function: (1/2)||w||²")
print("• Linear inequality constraints: y_i(w^T x_i + b) ≥ 1")
print("• CONVEX optimization problem")

# Task 2: Prove feasible region is convex
print("\n2. CONVEXITY OF FEASIBLE REGION")
print("-" * 50)

print("Proof that feasible region is convex:")
print("1. Each constraint y_i(w^T x_i + b) ≥ 1 defines a half-space")
print("2. A half-space is a convex set")
print("3. The intersection of convex sets is convex")
print("4. Therefore, the feasible region (intersection of all half-spaces) is convex")
print("\nMathematical proof:")
print("Let F = {(w,b) : y_i(w^T x_i + b) ≥ 1, ∀i}")
print("For any (w₁,b₁), (w₂,b₂) ∈ F and λ ∈ [0,1]:")
print("λ(w₁,b₁) + (1-λ)(w₂,b₂) = (λw₁ + (1-λ)w₂, λb₁ + (1-λ)b₂)")
print("For each constraint i:")
print("y_i[(λw₁ + (1-λ)w₂)^T x_i + (λb₁ + (1-λ)b₂)]")
print("= λ[y_i(w₁^T x_i + b₁)] + (1-λ)[y_i(w₂^T x_i + b₂)]")
print("≥ λ(1) + (1-λ)(1) = 1")
print("Therefore, F is convex.")

# Task 3: Show objective function is strictly convex
print("\n3. STRICT CONVEXITY OF OBJECTIVE FUNCTION")
print("-" * 50)

print("Objective function: f(w,b) = (1/2)||w||² = (1/2)w^T w")
print("\nProof of strict convexity:")
print("1. Compute Hessian matrix:")
print("   ∇²f = [∂²f/∂w∂w^T  ∂²f/∂w∂b]")
print("         [∂²f/∂b∂w^T  ∂²f/∂b²  ]")
print("   = [I  0]  where I is the identity matrix")
print("     [0  0]")
print("\n2. The Hessian is positive semidefinite")
print("3. For strict convexity in w: ∇²f_w = I > 0 (positive definite)")
print("4. The objective is strictly convex in w, convex in b")
print("5. Since we're minimizing over w (the main variables), the problem is strictly convex")

# Visualization of optimization problem structure
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Feasible region visualization (2D example)
# Simple 2D case: minimize w₁² + w₂² subject to constraints
w1_range = np.linspace(-2, 3, 100)
w2_range = np.linspace(-2, 3, 100)
W1, W2 = np.meshgrid(w1_range, w2_range)

# Example constraints: w₁ + w₂ ≥ 1, w₁ - w₂ ≥ -1, -w₁ + w₂ ≥ -1
constraint1 = W1 + W2 >= 1
constraint2 = W1 - W2 >= -1
constraint3 = -W1 + W2 >= -1
feasible = constraint1 & constraint2 & constraint3

# Plot feasible region
ax1.contourf(W1, W2, feasible.astype(int), levels=[0.5, 1.5], colors=['lightblue'], alpha=0.7)
ax1.contour(W1, W2, feasible.astype(int), levels=[0.5], colors=['blue'], linewidths=2)

# Plot constraint boundaries
ax1.plot(w1_range, 1 - w1_range, 'r-', linewidth=2, label='$w_1 + w_2 = 1$')
ax1.plot(w1_range, w1_range + 1, 'g-', linewidth=2, label='$w_1 - w_2 = -1$')
ax1.plot(w1_range, w1_range - 1, 'b-', linewidth=2, label='$-w_1 + w_2 = -1$')

# Plot objective function contours
objective = W1**2 + W2**2
ax1.contour(W1, W2, objective, levels=[0.5, 1, 2, 4], colors='black', alpha=0.5, linestyles='--')

# Mark optimal point
w_opt = np.array([0.5, 0.5])  # Example optimal point
ax1.scatter(w_opt[0], w_opt[1], c='red', s=150, marker='*', 
           edgecolor='black', linewidth=2, label='Optimal Point')

ax1.set_xlabel('$w_1$')
ax1.set_ylabel('$w_2$')
ax1.set_title('Convex Feasible Region')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xlim(-1, 2)
ax1.set_ylim(-1, 2)

# Plot 2: Objective function convexity
w_range = np.linspace(-2, 2, 100)
objective_1d = 0.5 * w_range**2

ax2.plot(w_range, objective_1d, 'b-', linewidth=3, label='$f(w) = \\frac{1}{2}w^2$')

# Show convexity property
w1, w2 = -1, 1
lambda_val = 0.3
w_combo = lambda_val * w1 + (1 - lambda_val) * w2
f_combo_actual = 0.5 * w_combo**2
f_combo_convex = lambda_val * (0.5 * w1**2) + (1 - lambda_val) * (0.5 * w2**2)

ax2.scatter([w1, w2], [0.5 * w1**2, 0.5 * w2**2], c='red', s=100, zorder=5)
ax2.scatter(w_combo, f_combo_actual, c='blue', s=100, zorder=5, label='Actual value')
ax2.scatter(w_combo, f_combo_convex, c='green', s=100, zorder=5, label='Convex combination')

# Draw line showing convex combination
ax2.plot([w1, w2], [0.5 * w1**2, 0.5 * w2**2], 'r--', alpha=0.7)
ax2.plot([w_combo, w_combo], [f_combo_actual, f_combo_convex], 'g-', linewidth=3, alpha=0.7)

ax2.set_xlabel('$w$')
ax2.set_ylabel('$f(w) = \\frac{1}{2}w^2$')
ax2.set_title('Strict Convexity of Objective Function')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: Complexity comparison
problem_sizes = [10, 50, 100, 500, 1000, 5000]
qp_complexity = [n**3 for n in problem_sizes]  # O(n³) for interior point methods
smo_complexity = [n**2 for n in problem_sizes]  # O(n²) for SMO algorithm

ax3.loglog(problem_sizes, qp_complexity, 'b-o', linewidth=2, markersize=8, label='Standard QP: $O(n^3)$')
ax3.loglog(problem_sizes, smo_complexity, 'r-s', linewidth=2, markersize=8, label='SMO Algorithm: $O(n^2)$')

ax3.set_xlabel('Number of Training Points (n)')
ax3.set_ylabel('Time Complexity')
ax3.set_title('SVM Solver Complexity Comparison')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Plot 4: Uniqueness conditions
conditions = ['Linearly\nSeparable', 'Strictly Convex\nObjective', 'Convex\nFeasible Region', 'Non-degenerate\nData']
importance = [1, 1, 1, 1]  # All equally important for uniqueness
colors = ['lightgreen', 'lightblue', 'lightyellow', 'lightcoral']

bars = ax4.bar(conditions, importance, color=colors, edgecolor='black', linewidth=2)
ax4.set_ylabel('Required for Uniqueness')
ax4.set_title('Conditions for Unique Solution')
ax4.set_ylim(0, 1.2)

# Add checkmarks
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax4.annotate('$\\checkmark$', xy=(bar.get_x() + bar.get_width()/2, height/2),
                ha='center', va='center', fontsize=20, fontweight='bold', color='darkgreen')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'optimization_theory_overview.png'), dpi=300, bbox_inches='tight')
plt.close()

# Task 4: Time complexity analysis
print("\n4. TIME COMPLEXITY ANALYSIS")
print("-" * 50)

print("Standard QP Solvers:")
print("• Interior Point Methods: O(n³) per iteration")
print("• Active Set Methods: O(n³) worst case")
print("• Total complexity: O(n³) to O(n⁴) depending on method")
print("\nSpecialized SVM Solvers:")
print("• SMO (Sequential Minimal Optimization): O(n²) to O(n³)")
print("• Decomposition methods: O(n²) to O(n³)")
print("• Online/incremental methods: O(n) per update")
print("\nPractical considerations:")
print("• For large n (>10,000): specialized algorithms preferred")
print("• For small to medium n: standard QP solvers acceptable")
print("• Memory requirements: O(n²) for kernel matrix storage")

# Task 5: Conditions for unique solution
print("\n5. CONDITIONS FOR UNIQUE SOLUTION")
print("-" * 50)

print("The SVM optimization problem has a unique solution when:")
print("\n1. LINEAR SEPARABILITY:")
print("   • Training data must be linearly separable")
print("   • Ensures feasible region is non-empty")
print("\n2. STRICT CONVEXITY:")
print("   • Objective function (1/2)||w||² is strictly convex in w")
print("   • Guarantees unique optimal w*")
print("\n3. NON-DEGENERATE DATA:")
print("   • Training points are in general position")
print("   • No redundant or collinear support vectors")
print("\n4. SUFFICIENT CONSTRAINTS:")
print("   • At least d+1 linearly independent constraints")
print("   • Ensures unique determination of hyperplane")
print("\nMathematical guarantee:")
print("Under these conditions, the KKT system has a unique solution (w*, b*, α*)")

# Simple visualization: Convex optimization landscape
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Create a 3D-like visualization of convex objective function
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = 0.5 * (X**2 + Y**2)  # Convex quadratic function

# Plot contours
contours = ax.contour(X, Y, Z, levels=15, colors='blue', alpha=0.6)
ax.contourf(X, Y, Z, levels=15, cmap='Blues', alpha=0.3)

# Mark global minimum
ax.scatter(0, 0, c='red', s=200, marker='*', edgecolor='black',
           linewidth=3, label='Global Minimum')

# Show feasible region (example constraints)
# Constraint 1: x + y >= -1
x_constraint = np.linspace(-3, 3, 100)
y_constraint1 = -1 - x_constraint
ax.fill_between(x_constraint, y_constraint1, 3, alpha=0.2, color='green',
                label='Feasible Region')

# Constraint 2: x - y >= -2
y_constraint2 = x_constraint + 2
mask = y_constraint2 <= 3
ax.fill_between(x_constraint[mask], -3, y_constraint2[mask], alpha=0.2, color='green')

ax.set_xlabel('$w_1$')
ax.set_ylabel('$w_2$')
ax.set_title('Convex Optimization Landscape')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'convex_optimization_simple.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAll visualizations saved to: {save_dir}")
print("=" * 80)
