import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_1_Quiz_27")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{newunicodechar} \newunicodechar{γ}{\gamma} \newunicodechar{̂}{\hat} \newunicodechar{ᵢ}{_i}'

print("=" * 80)
print("SVM FORMULAS AND TERMINOLOGY - QUESTION 27")
print("=" * 80)

# ============================================================================
# TASK 1: Geometric interpretation of ||w|| in SVM objective function
# ============================================================================
print("\n1. GEOMETRIC INTERPRETATION OF ||w||")
print("-" * 50)

# Define a weight vector w = [w1, w2]
w = np.array([3, 4])
w_norm = np.linalg.norm(w)

print("STEP-BY-STEP CALCULATION:")
print("Given: Weight vector w = [w₁, w₂] = [3, 4]")
print("")
print("Step 1: Write the formula for ||w||")
print("   ||w|| = √(w₁² + w₂²)")
print("")
print("Step 2: Substitute the values")
print("   ||w|| = √(3² + 4²)")
print("   ||w|| = √(9 + 16)")
print("   ||w|| = √25")
print("   ||w|| = 5")
print("")
print("Step 3: Geometric interpretation")
print("   • ||w|| represents the magnitude (length) of the weight vector")
print("   • The weight vector w is perpendicular to the decision boundary")
print("   • In SVM objective function min(½||w||²), minimizing ||w|| maximizes the margin")
print("   • The direction of w points toward the positive class")
print("")
print(f"Result: ||w|| = {w_norm:.2f}")

# Create visualization for Task 1
plt.figure(figsize=(10, 8))

# Plot the weight vector
plt.quiver(0, 0, w[0], w[1], angles='xy', scale_units='xy', scale=1, 
           color='red', linewidth=3, label=f'Weight vector w = [{w[0]}, {w[1]}]')

# Plot the magnitude
plt.plot([0, w[0]], [0, w[1]], 'r-', linewidth=2)

# Add annotations
plt.annotate(f'$||\\mathbf{{w}}|| = {w_norm:.2f}$', 
             xy=(w[0]/2, w[1]/2), 
             xytext=(w[0]/2 + 0.5, w[1]/2 + 0.5),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=12, color='red')

# Add grid and labels
plt.grid(True, alpha=0.3)
plt.xlabel('$w_1$')
plt.ylabel('$w_2$')
plt.title('Geometric Interpretation of $||\\mathbf{w}||$')
plt.axis('equal')
plt.xlim(-1, 6)
plt.ylim(-1, 6)

# Add text explanation
plt.text(0.05, 0.95, 
         r'$||\mathbf{w}||$ represents the magnitude (length) of the weight vector.\\'
         r'It is perpendicular to the hyperplane and determines the margin width.',
         transform=plt.gca().transAxes, 
         bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", ec="black"),
         verticalalignment='top')

plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'task1_weight_vector_interpretation.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# TASK 2: Constraint y_i(w^T x_i + b) >= 1 and when it equals 1
# ============================================================================
print("\n2. CONSTRAINT ANALYSIS: y_i(w^T x_i + b) >= 1")
print("-" * 50)

# Define a simple 2D example
w_example = np.array([1, -1])
b_example = 0

# Create some data points
x1_positive = np.array([2, 1])  # Positive class point
x2_negative = np.array([1, 2])  # Negative class point
x3_support = np.array([1, 1])   # Support vector (margin point)

print("STEP-BY-STEP ANALYSIS:")
print("Given: w = [1, -1], b = 0")
print("Constraint: y_i(w^T x_i + b) ≥ 1")
print("")
print("When y_i(w^T x_i + b) = 1 exactly:")
print("   • The point lies exactly on the margin boundary")
print("   • The point is a SUPPORT VECTOR")
print("   • These points define the optimal hyperplane")
print("")

# Calculate functional margins
def functional_margin(x, y, w, b):
    return y * (np.dot(w, x) + b)

# Test different points
points = [
    (x1_positive, 1, "Point A (Positive class)"),
    (x2_negative, -1, "Point B (Negative class)"),
    (x3_support, 1, "Point C (On boundary)")
]

for x, y, desc in points:
    print(f"{desc}:")
    print(f"   Given: x = [{x[0]}, {x[1]}], y = {y}")
    print(f"   Step 1: Calculate w^T x + b")
    print(f"      w^T x + b = [{w_example[0]}, {w_example[1]}] · [{x[0]}, {x[1]}] + {b_example}")
    print(f"      w^T x + b = {w_example[0]}×{x[0]} + {w_example[1]}×{x[1]} + {b_example}")
    print(f"      w^T x + b = {np.dot(w_example, x)} + {b_example} = {np.dot(w_example, x) + b_example}")
    print(f"   Step 2: Calculate y(w^T x + b)")
    print(f"      y(w^T x + b) = {y} × {np.dot(w_example, x) + b_example} = {y * (np.dot(w_example, x) + b_example)}")
    
    margin = functional_margin(x, y, w_example, b_example)
    if abs(margin - 1) < 1e-10:
        print(f"   ✓ RESULT: This is a support vector (margin = 1)")
    elif margin > 1:
        print(f"   ✓ RESULT: Correctly classified with margin > 1")
    else:
        print(f"   ✗ RESULT: Violates constraint (margin < 1)")
    print("")

# Create visualization for Task 2
plt.figure(figsize=(12, 10))

# Plot decision boundary: w^T x + b = 0
# For w = [1, -1], b = 0: x1 - x2 = 0, so x2 = x1
x_line = np.linspace(-1, 4, 100)
y_line = x_line  # x2 = x1
plt.plot(x_line, y_line, 'g-', linewidth=3, label='Decision Boundary: $\\mathbf{w}^T\\mathbf{x} + b = 0$')

# Plot margin boundaries: w^T x + b = ±1
y_margin_plus = x_line + 1  # x2 = x1 + 1
y_margin_minus = x_line - 1  # x2 = x1 - 1
plt.plot(x_line, y_margin_plus, 'b--', linewidth=2, label='Margin Boundary: $\\mathbf{w}^T\\mathbf{x} + b = 1$')
plt.plot(x_line, y_margin_minus, 'b--', linewidth=2, label='Margin Boundary: $\\mathbf{w}^T\\mathbf{x} + b = -1$')

# Plot points
colors = ['red', 'blue', 'green']
markers = ['o', 's', '^']
labels = ['Positive Class', 'Negative Class', 'Support Vector']

for i, (x, y, desc) in enumerate(points):
    margin = functional_margin(x, y, w_example, b_example)
    plt.scatter(x[0], x[1], s=200, c=colors[i], marker=markers[i], 
                edgecolor='black', linewidth=2, label=f'{desc} (margin={margin:.1f})')
    
    # Add point labels
    plt.annotate(f'({x[0]}, {x[1]})', (x[0], x[1]), 
                 xytext=(10, 10), textcoords='offset points',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"))

# Shade regions
x_grid = np.linspace(-1, 4, 50)
y_grid = np.linspace(-1, 4, 50)
X, Y = np.meshgrid(x_grid, y_grid)
Z = w_example[0]*X + w_example[1]*Y + b_example

plt.contourf(X, Y, Z, levels=[-10, 0], colors=['lightcoral'], alpha=0.3)
plt.contourf(X, Y, Z, levels=[0, 10], colors=['lightblue'], alpha=0.3)

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('SVM Constraint Analysis: $y_i(\\mathbf{w}^T\\mathbf{x}_i + b) \\geq 1$')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(-1, 4)
plt.ylim(-1, 4)

# Add explanation text
plt.text(0.02, 0.98, 
         r'When $y_i(\mathbf{w}^T\mathbf{x}_i + b) = 1$:\\'
         r'Point lies exactly on the margin boundary, is a support vector,\\'
         r'and these points define the optimal hyperplane.',
         transform=plt.gca().transAxes, 
         bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="black"),
         verticalalignment='top')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'task2_constraint_analysis.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# TASK 3: Distance from point to line formula
# ============================================================================
print("\n3. DISTANCE FROM POINT TO LINE FORMULA")
print("-" * 50)

# Example: line ax + by + c = 0, point (x0, y0)
a, b, c = 2, -3, 6  # Line: 2x - 3y + 6 = 0
x0, y0 = 1, 2       # Point: (1, 2)

print("STEP-BY-STEP CALCULATION:")
print("Given: Line ax + by + c = 0 where a = 2, b = -3, c = 6")
print("       Point (x₀, y₀) = (1, 2)")
print("")
print("Step 1: Write the distance formula")
print("   d = |ax₀ + by₀ + c| / √(a² + b²)")
print("")
print("Step 2: Substitute the values")
print("   d = |2×1 + (-3)×2 + 6| / √(2² + (-3)²)")
print("   d = |2 - 6 + 6| / √(4 + 9)")
print("   d = |2| / √13")
print("   d = 2 / √13")
print("")
print("Step 3: Calculate the final value")
print("   d = 2 / 3.606")
print("   d ≈ 0.555")
print("")
print("Step 4: Geometric interpretation")
print("   • This is the shortest distance from the point to the line")
print("   • The distance is measured along the perpendicular direction")
print("   • The formula works for any line in the form ax + by + c = 0")
print("")

# Formula: |ax0 + by0 + c| / sqrt(a^2 + b^2)
numerator = abs(a*x0 + b*y0 + c)
denominator = np.sqrt(a**2 + b**2)
distance = numerator / denominator

print(f"Result: d = {distance:.3f}")

# Create visualization for Task 3
plt.figure(figsize=(10, 8))

# Plot the line
x_line = np.linspace(-2, 5, 100)
y_line = (-a*x_line - c) / b  # Solve for y: y = (-ax - c)/b
plt.plot(x_line, y_line, 'g-', linewidth=3, label=f'Line: ${a}x + {b}y + {c} = 0$')

# Plot the point
plt.scatter(x0, y0, s=200, c='red', marker='o', edgecolor='black', linewidth=2, 
           label=f'Point: $({x0}, {y0})$')

# Draw perpendicular line from point to line
# Find the foot of perpendicular
# The perpendicular line has slope -a/b (negative reciprocal)
perp_slope = -a/b
perp_intercept = y0 - perp_slope * x0

# Find intersection
# y = perp_slope * x + perp_intercept
# y = (-a/b) * x - c/b
# Set them equal and solve for x
if abs(perp_slope - (-a/b)) > 1e-10:  # Check for division by zero
    x_intersect = (perp_intercept - (-c/b)) / (perp_slope - (-a/b))
    y_intersect = perp_slope * x_intersect + perp_intercept
else:
    # Lines are parallel, use a different approach
    x_intersect = x0
    y_intersect = (-a*x0 - c) / b

# Draw perpendicular line
plt.plot([x0, x_intersect], [y0, y_intersect], 'r--', linewidth=2, 
         label=f'Perpendicular distance = {distance:.3f}')

# Add annotations
plt.annotate(f'Distance = {distance:.3f}', 
             xy=((x0 + x_intersect)/2, (y0 + y_intersect)/2),
             xytext=((x0 + x_intersect)/2 + 0.5, (y0 + y_intersect)/2 + 0.5),
             arrowprops=dict(arrowstyle='<->', color='red'),
             fontsize=12, color='red')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Distance from Point to Line')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(-2, 5)
plt.ylim(-2, 5)

# Add formula text
plt.text(0.02, 0.98, 
         r'Distance formula: $d = \frac{|ax_0 + by_0 + c|}{\sqrt{a^2 + b^2}}$\\'
         rf'$d = \frac{{|{a} \times {x0} + {b} \times {y0} + {c}|}}{{\sqrt{{{a}^2 + {b}^2}}}}$\\'
         rf'$d = \frac{{{numerator}}}{{{denominator:.3f}}} = {distance:.3f}$',
         transform=plt.gca().transAxes, 
         bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="black"),
         verticalalignment='top')

plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'task3_point_to_line_distance.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# TASK 4: Functional margin definition and formula
# ============================================================================
print("\n4. FUNCTIONAL MARGIN DEFINITION AND FORMULA")
print("-" * 50)

# Example with multiple points
w_func = np.array([1, -1])
b_func = 0

# Create sample data points
data_points = [
    (np.array([2, 1]), 1, "Point A"),
    (np.array([1, 2]), -1, "Point B"),
    (np.array([1, 1]), 1, "Point C"),
    (np.array([3, 2]), 1, "Point D")
]

print("STEP-BY-STEP ANALYSIS:")
print("Given: w = [1, -1], b = 0")
print("")
print("Definition: Functional margin measures the confidence of the classifier's prediction")
print("Formula: \\hat{\\gamma}_i = y_i(w^T x_i + b)")
print("")
print("Key Properties:")
print("   • Positive when point is correctly classified")
print("   • Larger values indicate higher confidence")
print("   • Support vectors have \\hat{\\gamma}_i = 1")
print("   • Invariant to scaling of the weight vector")
print("")

for x, y, name in data_points:
    print(f"{name}:")
    print(f"   Given: x = [{x[0]}, {x[1]}], y = {y}")
    print(f"   Step 1: Calculate w^T x + b")
    print(f"      w^T x + b = [{w_func[0]}, {w_func[1]}] · [{x[0]}, {x[1]}] + {b_func}")
    print(f"      w^T x + b = {w_func[0]}×{x[0]} + {w_func[1]}×{x[1]} + {b_func}")
    activation = np.dot(w_func, x) + b_func
    print(f"      w^T x + b = {np.dot(w_func, x)} + {b_func} = {activation}")
    print(f"   Step 2: Calculate \\hat{{\\gamma}}_i = y(w^T x + b)")
    func_margin = y * activation
    print(f"      \\hat{{\\gamma}}_i = {y} × {activation} = {func_margin}")
    
    # Interpret the result
    if func_margin > 0:
        print(f"   ✓ RESULT: Correctly classified with confidence {func_margin}")
    elif func_margin == 0:
        print(f"   ⚠ RESULT: On decision boundary (margin = 0)")
    else:
        print(f"   ✗ RESULT: Misclassified (negative margin)")
    print("")

# Create visualization for Task 4
plt.figure(figsize=(12, 10))

# Plot decision boundary
x_line = np.linspace(-1, 4, 100)
y_line = x_line
plt.plot(x_line, y_line, 'g-', linewidth=3, label='Decision Boundary')

# Plot margin boundaries
y_margin_plus = x_line + 1
y_margin_minus = x_line - 1
plt.plot(x_line, y_margin_plus, 'b--', linewidth=2, label='Margin Boundaries')
plt.plot(x_line, y_margin_minus, 'b--', linewidth=2)

# Plot points with functional margins
colors = ['red', 'blue', 'green', 'orange']
markers = ['o', 's', '^', 'D']

for i, (x, y, name) in enumerate(data_points):
    func_margin = y * (np.dot(w_func, x) + b_func)
    plt.scatter(x[0], x[1], s=200, c=colors[i], marker=markers[i], 
                edgecolor='black', linewidth=2, 
                label=f'{name}: $\\hat{{\\gamma}} = {func_margin}$')
    
    # Add point labels
    plt.annotate(f'{name}\\\\$\\hat{{\\gamma}} = {func_margin}$', (x[0], x[1]), 
                 xytext=(10, 10), textcoords='offset points',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"))

# Shade regions
x_grid = np.linspace(-1, 4, 50)
y_grid = np.linspace(-1, 4, 50)
X, Y = np.meshgrid(x_grid, y_grid)
Z = w_func[0]*X + w_func[1]*Y + b_func

plt.contourf(X, Y, Z, levels=[-10, 0], colors=['lightcoral'], alpha=0.3)
plt.contourf(X, Y, Z, levels=[0, 10], colors=['lightblue'], alpha=0.3)

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Functional Margin Examples')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(-1, 4)
plt.ylim(-1, 4)

# Add explanation text
plt.text(0.02, 0.98, 
         r'Functional Margin Definition: $\hat{\gamma}_i = y_i(\mathbf{w}^T\mathbf{x}_i + b)$.\\'
         r'Measures confidence of prediction. Positive when correctly classified.\\'
         r'Larger values = higher confidence. Support vectors have $\hat{\gamma}_i = 1$.',
         transform=plt.gca().transAxes, 
         bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="black"),
         verticalalignment='top')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'task4_functional_margin.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# TASK 5: Relationship between functional and geometric margin
# ============================================================================
print("\n5. RELATIONSHIP BETWEEN FUNCTIONAL AND GEOMETRIC MARGIN")
print("-" * 50)

# Example calculation
w_geom = np.array([2, -1])
b_geom = 0
x_example = np.array([1, 1])
y_example = 1

print("STEP-BY-STEP ANALYSIS:")
print("Given: w = [2, -1], b = 0, x = [1, 1], y = 1")
print("")
print("Relationship Formula: \\gamma = \\hat{\\gamma}/||w||")
print("")
print("Key Insights:")
print("   • \\hat{\\gamma} (functional margin): Measures prediction confidence")
print("   • \\gamma (geometric margin): Actual Euclidean distance")
print("   • ||w|| (weight norm): Normalizes the margin")
print("   • \\gamma is invariant to scaling of the weight vector")
print("")

# Functional margin
func_margin = y_example * (np.dot(w_geom, x_example) + b_geom)

print("Step 1: Calculate functional margin \\hat{\\gamma}")
print(f"   \\hat{{\\gamma}} = y(w^T x + b)")
print(f"   \\hat{{\\gamma}} = {y_example} × ([{w_geom[0]}, {w_geom[1]}] · [{x_example[0]}, {x_example[1]}] + {b_geom})")
print(f"   \\hat{{\\gamma}} = {y_example} × ({np.dot(w_geom, x_example)} + {b_geom})")
print(f"   \\hat{{\\gamma}} = {y_example} × {np.dot(w_geom, x_example) + b_geom}")
print(f"   \\hat{{\\gamma}} = {func_margin}")
print("")

# Geometric margin
w_norm = np.linalg.norm(w_geom)
geom_margin = func_margin / w_norm

print("Step 2: Calculate weight norm ||w||")
print(f"   ||w|| = √(w₁² + w₂²)")
print(f"   ||w|| = √({w_geom[0]}² + {w_geom[1]}²)")
print(f"   ||w|| = √({w_geom[0]**2} + {w_geom[1]**2})")
print(f"   ||w|| = √{w_geom[0]**2 + w_geom[1]**2}")
print(f"   ||w|| = {w_norm:.3f}")
print("")

print("Step 3: Calculate geometric margin \\gamma")
print(f"   \\gamma = \\hat{{\\gamma}}/||w||")
print(f"   \\gamma = {func_margin}/{w_norm:.3f}")
print(f"   \\gamma = {geom_margin:.3f}")
print("")

print("Step 4: Interpretation")
print(f"   • Functional margin \\hat{{\\gamma}} = {func_margin} (confidence measure)")
print(f"   • Geometric margin \\gamma = {geom_margin:.3f} (actual distance)")
print(f"   • The point is {geom_margin:.3f} units away from the decision boundary")
print("")

# Create visualization for Task 5
plt.figure(figsize=(12, 10))

# Plot decision boundary
x_line = np.linspace(-2, 4, 100)
y_line = (-w_geom[0]*x_line - b_geom) / w_geom[1]  # Solve for y
plt.plot(x_line, y_line, 'g-', linewidth=3, label='Decision Boundary')

# Plot margin boundaries
y_margin_plus = (-w_geom[0]*x_line - b_geom + 1) / w_geom[1]
y_margin_minus = (-w_geom[0]*x_line - b_geom - 1) / w_geom[1]
plt.plot(x_line, y_margin_plus, 'b--', linewidth=2, label='Margin Boundaries')
plt.plot(x_line, y_margin_minus, 'b--', linewidth=2)

# Plot the example point
plt.scatter(x_example[0], x_example[1], s=200, c='red', marker='o', 
            edgecolor='black', linewidth=2, label=f'Example Point: $\\hat{{\\gamma}} = {func_margin}$')

# Draw geometric margin
# Find the foot of perpendicular
perp_slope = w_geom[0]/w_geom[1]  # Perpendicular to w
perp_intercept = x_example[1] - perp_slope * x_example[0]

# Find intersection with decision boundary
if abs(perp_slope - (-w_geom[0]/w_geom[1])) > 1e-10:  # Check for division by zero
    x_intersect = (perp_intercept - (-b_geom/w_geom[1])) / (perp_slope - (-w_geom[0]/w_geom[1]))
    y_intersect = perp_slope * x_intersect + perp_intercept
else:
    # Lines are parallel, use a different approach
    x_intersect = x_example[0]
    y_intersect = (-w_geom[0]*x_example[0] - b_geom) / w_geom[1]

# Draw geometric margin line
plt.plot([x_example[0], x_intersect], [x_example[1], y_intersect], 'r--', linewidth=2, 
         label=f'Geometric margin = {geom_margin:.3f}')

# Add annotations
plt.annotate(f'$\\hat{{\\gamma}} = {func_margin}$\\\\$\\gamma = {geom_margin:.3f}$', 
             xy=((x_example[0] + x_intersect)/2, (x_example[1] + y_intersect)/2),
             xytext=((x_example[0] + x_intersect)/2 + 0.5, (x_example[1] + y_intersect)/2 + 0.5),
             arrowprops=dict(arrowstyle='<->', color='red'),
             fontsize=12, color='red')

# Shade regions
x_grid = np.linspace(-2, 4, 50)
y_grid = np.linspace(-2, 4, 50)
X, Y = np.meshgrid(x_grid, y_grid)
Z = w_geom[0]*X + w_geom[1]*Y + b_geom

plt.contourf(X, Y, Z, levels=[-10, 0], colors=['lightcoral'], alpha=0.3)
plt.contourf(X, Y, Z, levels=[0, 10], colors=['lightblue'], alpha=0.3)

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Functional vs Geometric Margin')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(-2, 4)
plt.ylim(-2, 4)

# Add relationship explanation
plt.text(0.02, 0.98, 
         r'Relationship: $\gamma = \hat{\gamma}/||\mathbf{w}||$.\\'
         r'Functional margin ($\hat{\gamma}$): confidence measure.\\'
         r'Geometric margin ($\gamma$): actual distance.\\'
         r'$||\mathbf{w}||$ normalizes the margin.\\'
         r'$\gamma$ is invariant to scaling of $\mathbf{w}$.',
         transform=plt.gca().transAxes, 
         bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="black"),
         verticalalignment='top')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'task5_margin_relationship.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# SUMMARY AND COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY OF SVM FORMULAS AND TERMINOLOGY")
print("=" * 80)

print("\n1. ||w|| represents the magnitude (length) of the weight vector")
print("   - It's perpendicular to the decision boundary")
print("   - Minimizing ||w|| maximizes the geometric margin")

print("\n2. When y_i(w^T x_i + b) = 1:")
print("   - Point lies exactly on the margin boundary")
print("   - Point is a support vector")
print("   - These points define the optimal hyperplane")

print("\n3. Distance from point (x0, y0) to line ax + by + c = 0:")
print("   d = |ax0 + by0 + c| / √(a² + b²)")

print("\n4. Functional margin \\hat{\\gamma}_i = y_i(w^T x_i + b):")
print("   - Measures prediction confidence")
print("   - Positive for correct classification")
print("   - Larger values = higher confidence")

print("\n5. Relationship: \\gamma = \\hat{\\gamma}/||w||")
print("   - Geometric margin = Functional margin / Weight norm")
print("   - Geometric margin is the actual distance")
print("   - Invariant to scaling of the weight vector")

print(f"\nAll visualizations saved to: {save_dir}")
print("=" * 80)
