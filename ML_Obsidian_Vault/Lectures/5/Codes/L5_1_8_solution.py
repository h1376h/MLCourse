import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_1_Quiz_8")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("Question 8: Geometric Properties of Maximum Margin Hyperplane")
print("=" * 80)

# Task 1: Prove that the optimal hyperplane is equidistant from the closest points of each class
print("\n1. PROVING EQUIDISTANCE FROM CLOSEST POINTS")
print("-" * 50)

print("Mathematical Proof:")
print("For a maximum margin hyperplane w^T x + b = 0:")
print("- The margin boundaries are: w^T x + b = +1 and w^T x + b = -1")
print("- Distance from hyperplane to positive margin: d+ = 1/||w||")
print("- Distance from hyperplane to negative margin: d- = 1/||w||")
print("- Therefore: d+ = d- = 1/||w||")
print("- Total margin width = d+ + d- = 2/||w||")

# Visualization of equidistance property
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Create a simple 2D example
# Hyperplane: x1 + x2 = 0 (w = [1, 1], b = 0)
w = np.array([1, 1])
b = 0
w_norm = np.linalg.norm(w)

# Plot hyperplane
x1_range = np.linspace(-3, 3, 100)
x2_hyperplane = -x1_range  # x1 + x2 = 0 => x2 = -x1

# Plot margin boundaries
x2_pos_margin = -x1_range + 1/w_norm  # x1 + x2 = 1
x2_neg_margin = -x1_range - 1/w_norm  # x1 + x2 = -1

ax.plot(x1_range, x2_hyperplane, 'k-', linewidth=2, label='Decision Boundary')
ax.plot(x1_range, x2_pos_margin, 'r--', linewidth=1.5, label='Positive Margin')
ax.plot(x1_range, x2_neg_margin, 'b--', linewidth=1.5, label='Negative Margin')

# Add support vectors
support_vectors_pos = np.array([[1/np.sqrt(2), -1/np.sqrt(2)]])
support_vectors_neg = np.array([[-1/np.sqrt(2), 1/np.sqrt(2)]])

ax.scatter(support_vectors_pos[:, 0], support_vectors_pos[:, 1], 
           c='red', s=100, marker='o', edgecolor='black', linewidth=2,
           label='Positive Support Vectors')
ax.scatter(support_vectors_neg[:, 0], support_vectors_neg[:, 1], 
           c='blue', s=100, marker='s', edgecolor='black', linewidth=2,
           label='Negative Support Vectors')

# Draw distance lines
for sv in support_vectors_pos:
    # Project onto hyperplane
    proj = sv - (np.dot(w, sv) + b) / (w_norm**2) * w
    ax.plot([sv[0], proj[0]], [sv[1], proj[1]], 'g-', linewidth=2, alpha=0.7)
    
for sv in support_vectors_neg:
    # Project onto hyperplane
    proj = sv - (np.dot(w, sv) + b) / (w_norm**2) * w
    ax.plot([sv[0], proj[0]], [sv[1], proj[1]], 'g-', linewidth=2, alpha=0.7)

# Add distance annotations
ax.annotate(f'$d = \\frac{{1}}{{||w||}} = \\frac{{1}}{{{w_norm:.2f}}} = {1/w_norm:.2f}$',
            xy=(0.5, 0.5), xytext=(1.5, 1.5),
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
            fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.8))

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Equidistance Property of Maximum Margin Hyperplane')
ax.grid(True, alpha=0.3)
ax.legend()
ax.axis('equal')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'equidistance_property.png'), dpi=300, bbox_inches='tight')
plt.close()

# Task 2: Show that the decision boundary is uniquely determined
print("\n2. UNIQUENESS OF DECISION BOUNDARY")
print("-" * 50)

print("Proof of Uniqueness:")
print("1. The SVM optimization problem is:")
print("   min (1/2)||w||² subject to y_i(w^T x_i + b) ≥ 1")
print("2. This is a convex quadratic programming problem")
print("3. The objective function (1/2)||w||² is strictly convex")
print("4. The feasible region (defined by linear constraints) is convex")
print("5. A strictly convex function over a convex set has a unique global minimum")
print("6. Therefore, the optimal (w*, b*) is unique (up to scaling)")

# Visualization showing uniqueness
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Left plot: Multiple possible separating hyperplanes
x1_range = np.linspace(-2, 2, 100)

# Sample data points
pos_points = np.array([[1, 1], [1.5, 0.5]])
neg_points = np.array([[-1, -1], [-0.5, -1.5]])

# Multiple separating hyperplanes
hyperplanes = [
    (1, 1, 0),      # x1 + x2 = 0
    (1, 0.5, 0.2),  # x1 + 0.5*x2 = -0.2
    (0.8, 1.2, 0.1) # 0.8*x1 + 1.2*x2 = -0.1
]

colors = ['red', 'blue', 'green']
labels = ['Hyperplane 1', 'Hyperplane 2', 'Hyperplane 3']

for i, (w1, w2, b) in enumerate(hyperplanes):
    if w2 != 0:
        x2_line = (-w1 * x1_range - b) / w2
        ax1.plot(x1_range, x2_line, color=colors[i], linewidth=2, 
                label=labels[i], linestyle='--' if i > 0 else '-')

ax1.scatter(pos_points[:, 0], pos_points[:, 1], c='red', s=100, marker='o', 
           edgecolor='black', linewidth=2, label='Class +1')
ax1.scatter(neg_points[:, 0], neg_points[:, 1], c='blue', s=100, marker='s', 
           edgecolor='black', linewidth=2, label='Class -1')

ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.set_title('Multiple Possible Separating Hyperplanes')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xlim(-2.5, 2.5)
ax1.set_ylim(-2.5, 2.5)

# Right plot: Unique maximum margin hyperplane
w_opt = np.array([1, 1])
b_opt = 0
w_norm_opt = np.linalg.norm(w_opt)

x2_opt = -x1_range  # Optimal hyperplane
x2_pos_margin = -x1_range + 1/w_norm_opt
x2_neg_margin = -x1_range - 1/w_norm_opt

ax2.plot(x1_range, x2_opt, 'k-', linewidth=3, label='Maximum Margin Hyperplane')
ax2.plot(x1_range, x2_pos_margin, 'r--', linewidth=2, label='Positive Margin')
ax2.plot(x1_range, x2_neg_margin, 'b--', linewidth=2, label='Negative Margin')

# Support vectors for maximum margin
sv_pos = np.array([[0.707, -0.707]])
sv_neg = np.array([[-0.707, 0.707]])

ax2.scatter(sv_pos[:, 0], sv_pos[:, 1], c='red', s=150, marker='o', 
           edgecolor='black', linewidth=3, label='Support Vectors (+)')
ax2.scatter(sv_neg[:, 0], sv_neg[:, 1], c='blue', s=150, marker='s', 
           edgecolor='black', linewidth=3, label='Support Vectors (-)')

# Fill margin area
ax2.fill_between(x1_range, x2_pos_margin, x2_neg_margin, alpha=0.2, color='yellow',
                label='Margin Region')

ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.set_title('Unique Maximum Margin Hyperplane')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_xlim(-2.5, 2.5)
ax2.set_ylim(-2.5, 2.5)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'uniqueness_demonstration.png'), dpi=300, bbox_inches='tight')
plt.close()

# Task 3: Minimum number of support vectors in d dimensions
print("\n3. MINIMUM NUMBER OF SUPPORT VECTORS")
print("-" * 50)

print("Theoretical Analysis:")
print("For a d-dimensional problem:")
print("- A hyperplane in R^d has d+1 parameters (w₁, w₂, ..., wₐ, b)")
print("- Each support vector provides one constraint: yᵢ(w^T xᵢ + b) = 1")
print("- We also have the constraint: Σ αᵢ yᵢ = 0 from dual formulation")
print("- Minimum number of support vectors needed: d+1")
print("- This assumes general position (non-degenerate data)")

# Visualization for different dimensions
dimensions = [1, 2, 3, 4, 5]
min_support_vectors = [d + 1 for d in dimensions]

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.bar(dimensions, min_support_vectors, color='skyblue', edgecolor='navy', linewidth=2)
ax.set_xlabel('Dimension (d)')
ax.set_ylabel('Minimum Support Vectors')
ax.set_title('Minimum Number of Support Vectors vs Dimension')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, v in enumerate(min_support_vectors):
    ax.text(dimensions[i], v + 0.1, str(v), ha='center', va='bottom', fontweight='bold')

# Add formula annotation
ax.annotate('Minimum SVs = d + 1', xy=(3, 4), xytext=(4, 5),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=14, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'minimum_support_vectors.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nFor d=1: minimum {1+1} = 2 support vectors")
print(f"For d=2: minimum {2+1} = 3 support vectors") 
print(f"For d=3: minimum {3+1} = 4 support vectors")
print(f"For d=d: minimum d+1 support vectors")

# Task 4: Maximum number of support vectors
print("\n4. MAXIMUM NUMBER OF SUPPORT VECTORS")
print("-" * 50)

print("Theoretical Analysis:")
print("- Maximum number of support vectors = total number of training points (n)")
print("- This occurs when all points lie on the margin boundaries")
print("- In practice, this is rare for well-separated data")
print("- More common: small fraction of points are support vectors")

# Visualization of different scenarios
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Scenario 1: Few support vectors (typical case)
np.random.seed(42)
n_points = 20
pos_points_1 = np.random.normal([2, 2], 0.3, (n_points//2, 2))
neg_points_1 = np.random.normal([-2, -2], 0.3, (n_points//2, 2))

ax1.scatter(pos_points_1[:, 0], pos_points_1[:, 1], c='red', s=50, marker='o', alpha=0.7)
ax1.scatter(neg_points_1[:, 0], neg_points_1[:, 1], c='blue', s=50, marker='s', alpha=0.7)

# Highlight a few support vectors
sv_indices = [0, 1]  # Just for illustration
ax1.scatter(pos_points_1[sv_indices, 0], pos_points_1[sv_indices, 1], 
           c='red', s=150, marker='o', edgecolor='black', linewidth=3)
ax1.scatter(neg_points_1[sv_indices, 0], neg_points_1[sv_indices, 1], 
           c='blue', s=150, marker='s', edgecolor='black', linewidth=3)

ax1.plot([-1, 1], [1, -1], 'k-', linewidth=2)  # Decision boundary
ax1.set_title(f'Few Support Vectors\n({len(sv_indices)*2}/{n_points} points are SVs)')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-4, 4)
ax1.set_ylim(-4, 4)

# Scenario 2: Many support vectors
pos_points_2 = np.array([[1, 0.5], [0.8, 0.8], [1.2, 0.2]])
neg_points_2 = np.array([[-1, -0.5], [-0.8, -0.8], [-1.2, -0.2]])

ax2.scatter(pos_points_2[:, 0], pos_points_2[:, 1], c='red', s=150, marker='o',
           edgecolor='black', linewidth=3)
ax2.scatter(neg_points_2[:, 0], neg_points_2[:, 1], c='blue', s=150, marker='s',
           edgecolor='black', linewidth=3)

ax2.plot([-1.5, 1.5], [1.5, -1.5], 'k-', linewidth=2)  # Decision boundary
ax2.plot([-1.5, 1.5], [0.79, -0.79], 'r--', linewidth=1.5)  # Positive margin
ax2.plot([-1.5, 1.5], [2.21, -2.21], 'b--', linewidth=1.5)  # Negative margin

ax2.set_title(f'Many Support Vectors\n({len(pos_points_2)+len(neg_points_2)}/{len(pos_points_2)+len(neg_points_2)} points are SVs)')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-2, 2)
ax2.set_ylim(-2, 2)

# Scenario 3: Maximum case - all points are support vectors
points_on_margin_pos = np.array([[0.707, -0.707], [1.414, -1.414]])
points_on_margin_neg = np.array([[-0.707, 0.707], [-1.414, 1.414]])

ax3.scatter(points_on_margin_pos[:, 0], points_on_margin_pos[:, 1],
           c='red', s=150, marker='o', edgecolor='black', linewidth=3)
ax3.scatter(points_on_margin_neg[:, 0], points_on_margin_neg[:, 1],
           c='blue', s=150, marker='s', edgecolor='black', linewidth=3)

ax3.plot([-2, 2], [2, -2], 'k-', linewidth=2)  # Decision boundary
ax3.plot([-2, 2], [1.29, -1.29], 'r--', linewidth=1.5)  # Positive margin
ax3.plot([-2, 2], [2.71, -2.71], 'b--', linewidth=1.5)  # Negative margin

total_points = len(points_on_margin_pos) + len(points_on_margin_neg)
ax3.set_title(f'Maximum Support Vectors\n({total_points}/{total_points} points are SVs)')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-2.5, 2.5)
ax3.set_ylim(-2.5, 2.5)

# Scenario 4: Comparison chart
scenarios = ['Few SVs\n(Typical)', 'Many SVs\n(Dense)', 'All SVs\n(Maximum)']
sv_percentages = [20, 60, 100]  # Example percentages

ax4.bar(scenarios, sv_percentages, color=['lightgreen', 'orange', 'red'],
        edgecolor='black', linewidth=2)
ax4.set_ylabel('Percentage of Support Vectors')
ax4.set_title('Support Vector Scenarios')
ax4.grid(True, alpha=0.3, axis='y')

# Add percentage labels
for i, v in enumerate(sv_percentages):
    ax4.text(i, v + 2, f'{v}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'support_vector_scenarios.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"Scenario 1: {len(sv_indices)*2}/{n_points} = {len(sv_indices)*2/n_points*100:.1f}% are support vectors")
print(f"Scenario 2: {len(pos_points_2)+len(neg_points_2)}/{len(pos_points_2)+len(neg_points_2)} = 100% are support vectors")
print(f"Maximum possible: n/n = 100% (all training points)")

# Task 5: Distance formula derivation
print("\n5. DISTANCE FORMULA DERIVATION")
print("-" * 50)

print("Deriving distance from point x₀ to hyperplane w^T x + b = 0:")
print("1. The hyperplane equation: w^T x + b = 0")
print("2. Normal vector to hyperplane: w (points perpendicular to hyperplane)")
print("3. Unit normal vector: w/||w||")
print("4. For any point x₀, project onto hyperplane:")
print("   - Vector from hyperplane to x₀: (w^T x₀ + b)/||w||² * w")
print("   - Distance = ||(w^T x₀ + b)/||w||² * w|| = |w^T x₀ + b|/||w||")

# Visualization of distance formula
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Define hyperplane: x1 + 2*x2 - 3 = 0
w_demo = np.array([1, 2])
b_demo = -3
w_norm_demo = np.linalg.norm(w_demo)

# Plot hyperplane
x1_demo = np.linspace(-1, 5, 100)
x2_demo = (-w_demo[0] * x1_demo - b_demo) / w_demo[1]

ax.plot(x1_demo, x2_demo, 'k-', linewidth=3, label='Hyperplane: $x_1 + 2x_2 - 3 = 0$')

# Test points
test_points = np.array([[2, 1], [4, 2], [1, 0]])
colors = ['red', 'blue', 'green']

for i, point in enumerate(test_points):
    # Calculate distance
    distance = abs(np.dot(w_demo, point) + b_demo) / w_norm_demo

    # Find projection on hyperplane
    # Point on hyperplane closest to test point
    t = -(np.dot(w_demo, point) + b_demo) / np.dot(w_demo, w_demo)
    projection = point + t * w_demo

    # Plot point and projection
    ax.scatter(point[0], point[1], c=colors[i], s=150, marker='o',
              edgecolor='black', linewidth=2, label=f'Point {i+1}: ({point[0]}, {point[1]})')
    ax.scatter(projection[0], projection[1], c=colors[i], s=100, marker='x',
              linewidth=3)

    # Draw distance line
    ax.plot([point[0], projection[0]], [point[1], projection[1]],
            color=colors[i], linewidth=2, linestyle='--', alpha=0.8)

    # Add distance annotation
    mid_point = (point + projection) / 2
    ax.annotate(f'd = {distance:.2f}', xy=mid_point, xytext=(mid_point[0]+0.3, mid_point[1]+0.3),
                arrowprops=dict(arrowstyle='->', color=colors[i], lw=1.5),
                fontsize=10, bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

# Draw normal vector
origin_on_hyperplane = np.array([3, 0])  # Point on hyperplane
normal_end = origin_on_hyperplane + 0.5 * w_demo / w_norm_demo
ax.arrow(origin_on_hyperplane[0], origin_on_hyperplane[1],
         normal_end[0] - origin_on_hyperplane[0], normal_end[1] - origin_on_hyperplane[1],
         head_width=0.1, head_length=0.1, fc='purple', ec='purple', linewidth=2)
ax.annotate('Normal vector $\\mathbf{w}$', xy=normal_end, xytext=(normal_end[0]+0.3, normal_end[1]+0.3),
            fontsize=12, color='purple', fontweight='bold')

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Distance from Point to Hyperplane: $d = \\frac{|\\mathbf{w}^T\\mathbf{x}_0 + b|}{||\\mathbf{w}||}$')
ax.grid(True, alpha=0.3)
ax.legend()
ax.axis('equal')
ax.set_xlim(-0.5, 5.5)
ax.set_ylim(-1, 3)

# Add formula box
formula_text = ('Distance Formula:\n'
                '$d = \\frac{|\\mathbf{w}^T\\mathbf{x}_0 + b|}{||\\mathbf{w}||}$\n\n'
                f'For hyperplane: ${w_demo[0]}x_1 + {w_demo[1]}x_2 + {b_demo} = 0$\n'
                f'$||\\mathbf{{w}}|| = \\sqrt{{{w_demo[0]}^2 + {w_demo[1]}^2}} = {w_norm_demo:.2f}$')

ax.text(0.02, 0.98, formula_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", alpha=0.9))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'distance_formula_visualization.png'), dpi=300, bbox_inches='tight')
plt.close()

# Numerical verification
print("\nNumerical Verification:")
for i, point in enumerate(test_points):
    distance = abs(np.dot(w_demo, point) + b_demo) / w_norm_demo
    print(f"Point {i+1} ({point[0]}, {point[1]}): d = |{w_demo[0]}×{point[0]} + {w_demo[1]}×{point[1]} + {b_demo}|/{w_norm_demo:.2f} = {distance:.3f}")

# Simple visualization: SVM geometry overview
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Create a clean 2D SVM example
np.random.seed(42)
# Support vectors
sv_pos = np.array([[1.2, 0.8], [0.9, 1.1]])
sv_neg = np.array([[-1.2, -0.8], [-0.9, -1.1]])
# Non-support vectors
nsv_pos = np.array([[2.1, 1.5], [1.8, 2.0], [2.5, 1.2]])
nsv_neg = np.array([[-2.1, -1.5], [-1.8, -2.0], [-2.5, -1.2]])

# Decision boundary and margins
x_range = np.linspace(-3, 3, 100)
decision_boundary = -0.8 * x_range  # Simple line
pos_margin = -0.8 * x_range + 1.0
neg_margin = -0.8 * x_range - 1.0

# Plot margins
ax.fill_between(x_range, pos_margin, neg_margin, alpha=0.2, color='yellow', label='Margin')
ax.plot(x_range, decision_boundary, 'k-', linewidth=3, label='Decision Boundary')
ax.plot(x_range, pos_margin, 'r--', linewidth=2, alpha=0.7)
ax.plot(x_range, neg_margin, 'b--', linewidth=2, alpha=0.7)

# Plot points
ax.scatter(sv_pos[:, 0], sv_pos[:, 1], c='red', s=150, marker='o',
           edgecolor='black', linewidth=3, label='Support Vectors (+)')
ax.scatter(sv_neg[:, 0], sv_neg[:, 1], c='blue', s=150, marker='s',
           edgecolor='black', linewidth=3, label='Support Vectors (-)')
ax.scatter(nsv_pos[:, 0], nsv_pos[:, 1], c='red', s=80, marker='o',
           alpha=0.6, edgecolor='gray', linewidth=1)
ax.scatter(nsv_neg[:, 0], nsv_neg[:, 1], c='blue', s=80, marker='s',
           alpha=0.6, edgecolor='gray', linewidth=1)

# Draw distance lines from support vectors to decision boundary
for sv in np.vstack([sv_pos, sv_neg]):
    # Find closest point on decision boundary
    # For line ax + by + c = 0, closest point to (x0,y0) is:
    # (x0,y0) - ((ax0 + by0 + c)/(a² + b²)) * (a,b)
    a, b, c = 0.8, 1, 0  # 0.8x + y = 0
    closest_x = sv[0] - (a * sv[0] + b * sv[1] + c) / (a**2 + b**2) * a
    closest_y = sv[1] - (a * sv[0] + b * sv[1] + c) / (a**2 + b**2) * b
    ax.plot([sv[0], closest_x], [sv[1], closest_y], 'g-', linewidth=2, alpha=0.7)

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('SVM Geometric Properties')
ax.grid(True, alpha=0.3)
ax.legend()
ax.axis('equal')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'svm_geometry_simple.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAll visualizations saved to: {save_dir}")
print("=" * 80)
