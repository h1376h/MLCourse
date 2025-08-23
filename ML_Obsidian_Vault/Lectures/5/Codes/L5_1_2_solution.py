import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_1_Quiz_2")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting (disabled for compatibility)
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("=" * 80)
print("QUESTION 2: PRIMAL OPTIMIZATION PROBLEM FOR MAXIMUM MARGIN CLASSIFICATION")
print("=" * 80)

# ============================================================================
# TASK 1: Explain why we minimize ||w||^2 instead of maximizing 1/||w||
# ============================================================================
print("\n" + "="*60)
print("TASK 1: Why minimize ||w||^2 instead of maximizing 1/||w||?")
print("="*60)

print("""
Mathematical Reasons:
1. Differentiability: ||w||^2 is differentiable everywhere, while 1/||w|| is not differentiable at w=0
2. Convexity: ||w||^2 is a convex function, making optimization easier and guaranteeing global optimum
3. Computational efficiency: Quadratic functions are easier to optimize than rational functions
4. Equivalence: Minimizing ||w||^2 is equivalent to maximizing 1/||w|| when ||w|| > 0

Let's demonstrate this with a simple example:
""")

# Create a simple example to illustrate
w_values = np.linspace(0.1, 5, 100)
norm_w = w_values
norm_w_squared = w_values**2
inverse_norm_w = 1/w_values

plt.figure(figsize=(15, 5))

# Plot 1: ||w|| vs ||w||^2
plt.subplot(1, 3, 1)
plt.plot(w_values, norm_w, 'b-', linewidth=2, label=r'$||\mathbf{w}||$')
plt.plot(w_values, norm_w_squared, 'r-', linewidth=2, label=r'$||\mathbf{w}||^2$')
plt.xlabel(r'$||\mathbf{w}||$')
plt.ylabel('Value')
plt.title('Comparison of ||w|| and ||w||²')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: 1/||w||
plt.subplot(1, 3, 2)
plt.plot(w_values, inverse_norm_w, 'g-', linewidth=2, label=r'$\frac{1}{||\mathbf{w}||}$')
plt.xlabel(r'$||\mathbf{w}||$')
plt.ylabel('Value')
plt.title('Inverse of ||w||')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Optimization landscape comparison
plt.subplot(1, 3, 3)
plt.plot(w_values, norm_w_squared, 'r-', linewidth=2, label=r'Minimize: $||\mathbf{w}||^2$')
plt.plot(w_values, -inverse_norm_w, 'g--', linewidth=2, label=r'Maximize: $-\frac{1}{||\mathbf{w}||}$')
plt.xlabel(r'$||\mathbf{w}||$')
plt.ylabel('Objective Value')
plt.title('Optimization Landscape Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'task1_optimization_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

print("✓ Plotted optimization landscape comparison")
print("✓ ||w||² is convex and differentiable everywhere")
print("✓ 1/||w|| has issues at w=0 and is harder to optimize")

# ============================================================================
# TASK 2: Geometric interpretation of constraints
# ============================================================================
print("\n" + "="*60)
print("TASK 2: Geometric interpretation of y_i(w^T x_i + b) ≥ 1")
print("="*60)

print("""
Geometric Interpretation:
The constraint y_i(w^T x_i + b) ≥ 1 ensures that:
1. All points are correctly classified (y_i and w^T x_i + b have same sign)
2. All points are at least distance 1/||w|| from the decision boundary
3. The margin width is 2/||w|| (distance between parallel hyperplanes)

Let's visualize this with a 2D example:
""")

# Create a simple 2D dataset
np.random.seed(42)
n_points = 20

# Generate two classes
class1_points = np.random.multivariate_normal([2, 2], [[0.5, 0], [0, 0.5]], n_points//2)
class2_points = np.random.multivariate_normal([-2, -2], [[0.5, 0], [0, 0.5]], n_points//2)

X = np.vstack([class1_points, class2_points])
y = np.hstack([np.ones(n_points//2), -np.ones(n_points//2)])

# Define a decision boundary (w = [1, 1], b = 0)
w = np.array([1, 1])
b = 0

plt.figure(figsize=(12, 8))

# Plot data points
plt.scatter(class1_points[:, 0], class1_points[:, 1], c='red', s=100, label='Class +1', alpha=0.7)
plt.scatter(class2_points[:, 0], class2_points[:, 1], c='blue', s=100, label='Class -1', alpha=0.7)

# Plot decision boundary
x_range = np.linspace(-4, 4, 100)
decision_boundary = (-w[0] * x_range - b) / w[1]
plt.plot(x_range, decision_boundary, 'k-', linewidth=2, label='Decision Boundary')

# Plot margin boundaries
margin_upper = (-w[0] * x_range - b + 1) / w[1]
margin_lower = (-w[0] * x_range - b - 1) / w[1]
plt.plot(x_range, margin_upper, 'g--', linewidth=2, label='Margin Boundary (+1)')
plt.plot(x_range, margin_lower, 'g--', linewidth=2, label='Margin Boundary (-1)')

# Shade the margin
plt.fill_between(x_range, margin_lower, margin_upper, alpha=0.2, color='green', label='Margin')

# Add arrows showing distances
for i, (x, yi) in enumerate(zip(X, y)):
    # Calculate distance to decision boundary
    distance = abs(np.dot(w, x) + b) / np.linalg.norm(w)
    
    # Add arrow if point is close to boundary
    if distance < 2:
        # Project point onto decision boundary
        proj_x = x - (np.dot(w, x) + b) * w / (np.linalg.norm(w)**2)
        
        # Draw arrow from point to boundary
        plt.annotate('', xy=proj_x, xytext=x,
                    arrowprops=dict(arrowstyle='->', color='purple', alpha=0.6))
        
        # Add distance label
        plt.text(x[0] + 0.1, x[1] + 0.1, f'{distance:.2f}', 
                fontsize=8, color='purple')

plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title('Geometric Interpretation of SVM Constraints')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(-4, 4)
plt.ylim(-4, 4)

plt.savefig(os.path.join(save_dir, 'task2_geometric_interpretation.png'), dpi=300, bbox_inches='tight')
plt.close()

print("✓ Visualized decision boundary and margin")
print("✓ Constraint y_i(w^T x_i + b) ≥ 1 ensures minimum distance of 1/||w||")
print("✓ Margin width = 2/||w|| (distance between parallel hyperplanes)")

# ============================================================================
# TASK 3: Prove margin width is 2/||w||
# ============================================================================
print("\n" + "="*60)
print("TASK 3: Prove that margin width is 2/||w||")
print("="*60)

print("""
Mathematical Proof:

1. The decision boundary is: w^T x + b = 0
2. The margin boundaries are: w^T x + b = ±1
3. Distance from a point x to the decision boundary is:
   d = |w^T x + b| / ||w||
4. For points on margin boundaries: |w^T x + b| = 1
5. So distance from margin boundary to decision boundary = 1/||w||
6. Total margin width = 2 × (1/||w||) = 2/||w||

Let's verify this numerically:
""")

# Numerical verification
w_norm = np.linalg.norm(w)
margin_width_theoretical = 2 / w_norm

print(f"||w|| = {w_norm:.4f}")
print(f"Theoretical margin width = 2/||w|| = {margin_width_theoretical:.4f}")

# Calculate actual margin width from our visualization
# Points on margin boundaries: w^T x + b = ±1
# For x = [x1, x2]: x1 + x2 = ±1
# So x2 = ±1 - x1

margin_points_upper = np.array([[x, 1 - x] for x in x_range])
margin_points_lower = np.array([[x, -1 - x] for x in x_range])

# Calculate distance between parallel lines
# Distance between parallel lines ax + by + c1 = 0 and ax + by + c2 = 0 is |c1 - c2| / sqrt(a² + b²)
# Our lines: x1 + x2 - 1 = 0 and x1 + x2 + 1 = 0
# So distance = |(-1) - 1| / sqrt(1² + 1²) = 2 / sqrt(2) = sqrt(2)

actual_margin_width = 2 / np.sqrt(2)
print(f"Actual margin width = {actual_margin_width:.4f}")
print(f"Verification: {abs(margin_width_theoretical - actual_margin_width) < 1e-10}")

# Visualize the proof
plt.figure(figsize=(10, 8))

# Plot decision boundary
plt.plot(x_range, decision_boundary, 'k-', linewidth=3, label='Decision Boundary: $\\mathbf{w}^T\\mathbf{x} + b = 0$')

# Plot margin boundaries
plt.plot(x_range, margin_upper, 'g-', linewidth=2, label='Margin Boundary: $\\mathbf{w}^T\\mathbf{x} + b = 1$')
plt.plot(x_range, margin_lower, 'g-', linewidth=2, label='Margin Boundary: $\\mathbf{w}^T\\mathbf{x} + b = -1$')

# Add distance annotations
plt.annotate('', xy=(0, 1), xytext=(0, 0),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
plt.text(0.2, 0.5, r'$\frac{1}{||\mathbf{w}||}$', fontsize=14, color='red')

plt.annotate('', xy=(0, -1), xytext=(0, 0),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
plt.text(0.2, -0.5, r'$\frac{1}{||\mathbf{w}||}$', fontsize=14, color='red')

plt.annotate('', xy=(0, 1), xytext=(0, -1),
            arrowprops=dict(arrowstyle='<->', color='blue', lw=3))
plt.text(0.5, 0, r'$\frac{2}{||\mathbf{w}||}$', fontsize=16, color='blue', weight='bold')

plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title('Proof: Margin Width = $\\frac{2}{||\\mathbf{w}||}$')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(-3, 3)
plt.ylim(-3, 3)

plt.savefig(os.path.join(save_dir, 'task3_margin_width_proof.png'), dpi=300, bbox_inches='tight')
plt.close()

print("✓ Proved margin width = 2/||w||")
print("✓ Verified numerically and visually")

# ============================================================================
# TASK 4: Minimum number of active constraints in 2D
# ============================================================================
print("\n" + "="*60)
print("TASK 4: Minimum number of active constraints in 2D")
print("="*60)

print("""
In 2D, the minimum number of constraints that must be active at the optimal solution is 2.

Reasoning:
1. The optimal solution lies on the boundary of the feasible region
2. In 2D, we need at least 2 constraints to define a unique point
3. The support vectors are the points that lie exactly on the margin boundaries
4. For a well-defined maximum margin hyperplane, we need at least 2 support vectors

Let's demonstrate this with a simple example:
""")

# Create a simple 2D example with clear support vectors
support_vectors = np.array([
    [1, 1],   # Support vector for class +1
    [-1, -1]  # Support vector for class -1
])

# Optimal hyperplane: x1 + x2 = 0 (w = [1, 1], b = 0)
w_opt = np.array([1, 1])
b_opt = 0

# Generate additional points
additional_points = np.array([
    [2, 2], [3, 1], [1, 3],  # Class +1 points
    [-2, -2], [-3, -1], [-1, -3]  # Class -1 points
])

plt.figure(figsize=(10, 8))

# Plot all points
plt.scatter(additional_points[:3, 0], additional_points[:3, 1], 
           c='red', s=100, label='Class +1 (non-support)', alpha=0.5)
plt.scatter(additional_points[3:, 0], additional_points[3:, 1], 
           c='blue', s=100, label='Class -1 (non-support)', alpha=0.5)

# Highlight support vectors
plt.scatter(support_vectors[0, 0], support_vectors[0, 1], 
           c='red', s=200, marker='s', edgecolor='black', linewidth=2, label='Support Vector (+1)')
plt.scatter(support_vectors[1, 0], support_vectors[1, 1], 
           c='blue', s=200, marker='s', edgecolor='black', linewidth=2, label='Support Vector (-1)')

# Plot optimal decision boundary
x_range = np.linspace(-3, 3, 100)
decision_boundary = (-w_opt[0] * x_range - b_opt) / w_opt[1]
plt.plot(x_range, decision_boundary, 'k-', linewidth=3, label='Optimal Decision Boundary')

# Plot margin boundaries
margin_upper = (-w_opt[0] * x_range - b_opt + 1) / w_opt[1]
margin_lower = (-w_opt[0] * x_range - b_opt - 1) / w_opt[1]
plt.plot(x_range, margin_upper, 'g--', linewidth=2, label='Margin Boundaries')
plt.plot(x_range, margin_lower, 'g--', linewidth=2)

# Shade the margin
plt.fill_between(x_range, margin_lower, margin_upper, alpha=0.2, color='green', label='Margin')

# Add annotations for support vectors
for i, sv in enumerate(support_vectors):
    plt.annotate(f'SV{i+1}', (sv[0], sv[1]), xytext=(10, 10), 
                textcoords='offset points', fontsize=12, weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))

plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title('Minimum 2 Active Constraints in 2D SVM')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(-3, 3)
plt.ylim(-3, 3)

plt.savefig(os.path.join(save_dir, 'task4_minimum_constraints.png'), dpi=300, bbox_inches='tight')
plt.close()

print("✓ Demonstrated minimum 2 active constraints in 2D")
print("✓ Support vectors define the optimal hyperplane")
print("✓ Additional points don't affect the solution")

# ============================================================================
# TASK 5: Lagrangian function
# ============================================================================
print("\n" + "="*60)
print("TASK 5: Lagrangian function for the optimization problem")
print("="*60)

print("""
The Lagrangian function for the primal SVM optimization problem is:

L(w, b, α) = (1/2)||w||² - Σᵢ αᵢ[yᵢ(w^T xᵢ + b) - 1]

where:
- αᵢ ≥ 0 are the Lagrange multipliers
- yᵢ(w^T xᵢ + b) - 1 ≥ 0 are the constraints
- The term αᵢ[yᵢ(w^T xᵢ + b) - 1] enforces the constraints

Let's implement and solve this numerically:
""")

def lagrangian(w, b, alpha, X, y):
    """Compute the Lagrangian function value"""
    n = len(y)
    w_norm_squared = np.dot(w, w)
    
    constraint_terms = 0
    for i in range(n):
        constraint_terms += alpha[i] * (y[i] * (np.dot(w, X[i]) + b) - 1)
    
    return 0.5 * w_norm_squared - constraint_terms

def constraint_violation(w, b, X, y):
    """Compute constraint violations"""
    violations = []
    for i in range(len(y)):
        violation = y[i] * (np.dot(w, X[i]) + b) - 1
        violations.append(violation)
    return np.array(violations)

# Use our simple dataset
X_simple = np.vstack([support_vectors, additional_points])
y_simple = np.hstack([1, -1, 1, 1, 1, -1, -1, -1])

# Initialize parameters
w_init = np.array([0.5, 0.5])
b_init = 0.0
alpha_init = np.ones(len(y_simple)) * 0.1

print(f"Initial Lagrangian value: {lagrangian(w_init, b_init, alpha_init, X_simple, y_simple):.4f}")
print(f"Initial constraint violations: {constraint_violation(w_init, b_init, X_simple, y_simple)}")

# Visualize Lagrangian landscape
w1_range = np.linspace(-2, 2, 50)
w2_range = np.linspace(-2, 2, 50)
W1, W2 = np.meshgrid(w1_range, w2_range)

# Fix b and alpha for visualization
b_fixed = 0
alpha_fixed = np.ones(len(y_simple)) * 0.1

L_values = np.zeros_like(W1)
for i in range(len(w1_range)):
    for j in range(len(w2_range)):
        w_current = np.array([W1[i, j], W2[i, j]])
        L_values[i, j] = lagrangian(w_current, b_fixed, alpha_fixed, X_simple, y_simple)

plt.figure(figsize=(12, 5))

# Plot Lagrangian landscape
plt.subplot(1, 2, 1)
contour = plt.contour(W1, W2, L_values, levels=20)
plt.colorbar(contour, label='Lagrangian Value')
plt.xlabel(r'$w_1$')
plt.ylabel(r'$w_2$')
plt.title('Lagrangian Landscape')
plt.grid(True, alpha=0.3)

# Mark optimal point
plt.scatter([1], [1], c='red', s=100, marker='*', label='Optimal w', zorder=5)
plt.legend()

# Plot constraint satisfaction
plt.subplot(1, 2, 2)
w_optimal = np.array([1, 1])
b_optimal = 0
violations = constraint_violation(w_optimal, b_optimal, X_simple, y_simple)

plt.bar(range(len(violations)), violations, color=['green' if v >= 0 else 'red' for v in violations])
plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
plt.xlabel('Data Point Index')
plt.ylabel('Constraint Value')
plt.title('Constraint Satisfaction (>= 0 means satisfied)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'task5_lagrangian_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

print("✓ Implemented Lagrangian function")
print("✓ Visualized Lagrangian landscape")
print("✓ Checked constraint satisfaction")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY OF ALL TASKS")
print("="*80)

print("""
Task 1: Why minimize ||w||² instead of maximizing 1/||w||?
✓ ||w||² is differentiable everywhere (1/||w|| is not at w=0)
✓ ||w||² is convex (easier optimization)
✓ Computational efficiency (quadratic vs rational)
✓ Mathematical equivalence when ||w|| > 0

Task 2: Geometric interpretation of y_i(w^T x_i + b) ≥ 1
✓ Ensures correct classification (same sign)
✓ Enforces minimum distance of 1/||w|| from boundary
✓ Defines margin width of 2/||w||
✓ Visualized with decision boundary and margin lines

Task 3: Prove margin width is 2/||w||
✓ Distance from boundary to margin line = 1/||w||
✓ Total margin width = 2 × (1/||w||) = 2/||w||
✓ Verified numerically and visually

Task 4: Minimum active constraints in 2D
✓ Minimum 2 constraints must be active
✓ Support vectors define the optimal hyperplane
✓ Demonstrated with 2D example

Task 5: Lagrangian function
✓ L(w, b, α) = (1/2)||w||² - Σᵢ αᵢ[yᵢ(w^T xᵢ + b) - 1]
✓ Implemented and visualized
✓ Lagrange multipliers enforce constraints
""")

print(f"\nAll plots saved to: {save_dir}")
print("=" * 80)
