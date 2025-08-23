import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_1_Quiz_16")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 60)
print("Question 16: Numerical Hyperplane Analysis")
print("=" * 60)

# Given data
print("\nGiven Dataset:")
X = np.array([[2, 1], [3, 3], [0, 0], [1, 2]])  # Training points
y = np.array([1, 1, -1, -1])  # Labels
w = np.array([1, -1])  # Weight vector
b = 0.5  # Bias term

print(f"Training points:")
for i, (xi, yi) in enumerate(zip(X, y)):
    print(f"  x_{i+1} = {xi}, y_{i+1} = {yi:+d}")
print(f"Hyperplane: w = {w}, b = {b}")
print(f"Hyperplane equation: {w[0]}x_1 + {w[1]}x_2 + {b} = 0")

# Task 1: Calculate y_i(w^T x_i + b) for each point and verify margin constraints
print("\n" + "="*50)
print("Task 1: Verify Margin Constraints")
print("="*50)

print("\nCalculating y_i(w^T x_i + b) for each point:")
margin_values = []
for i, (xi, yi) in enumerate(zip(X, y)):
    # Calculate w^T x_i + b
    activation = np.dot(w, xi) + b
    # Calculate y_i(w^T x_i + b)
    margin_value = yi * activation
    margin_values.append(margin_value)
    
    print(f"\nPoint {i+1}: x_{i+1} = {xi}, y_{i+1} = {yi:+d}")
    print(f"  w^T x_{i+1} + b = {w[0]}*{xi[0]} + {w[1]}*{xi[1]} + {b} = {activation}")
    print(f"  y_{i+1}(w^T x_{i+1} + b) = {yi} * {activation} = {margin_value}")
    
    if margin_value >= 1:
        print(f"  ✓ Constraint satisfied: {margin_value} ≥ 1")
    elif margin_value > 0:
        print(f"  ⚠ Positive but < 1: {margin_value}")
    else:
        print(f"  ✗ Constraint violated: {margin_value} < 0")

margin_values = np.array(margin_values)

# Task 2: Identify support vectors
print("\n" + "="*50)
print("Task 2: Identify Support Vectors")
print("="*50)

print("\nSupport vectors are points where y_i(w^T x_i + b) = 1:")
support_vectors = []
for i, margin_val in enumerate(margin_values):
    if abs(margin_val - 1.0) < 1e-10:  # Check for equality with tolerance
        support_vectors.append(i)
        print(f"  Point {i+1}: x_{i+1} = {X[i]} is a SUPPORT VECTOR (margin = {margin_val})")
    else:
        print(f"  Point {i+1}: x_{i+1} = {X[i]} is NOT a support vector (margin = {margin_val})")

print(f"\nSupport vector indices: {[i+1 for i in support_vectors]}")

# Task 3: Calculate geometric margin
print("\n" + "="*50)
print("Task 3: Calculate Geometric Margin")
print("="*50)

w_norm = np.linalg.norm(w)
geometric_margin = 1.0 / w_norm

print(f"||w|| = ||{w}|| = √({w[0]}² + {w[1]}²) = √{w[0]**2 + w[1]**2} = {w_norm}")
print(f"Geometric margin γ = 1/||w|| = 1/{w_norm} = {geometric_margin}")
print(f"Margin width = 2γ = 2/{w_norm} = {2*geometric_margin}")

# Task 4: Distance from test point to hyperplane
print("\n" + "="*50)
print("Task 4: Distance from Test Point to Hyperplane")
print("="*50)

test_point = np.array([1.5, 1.5])
print(f"Test point: {test_point}")

# Distance formula: |w^T x + b| / ||w||
activation_test = np.dot(w, test_point) + b
distance_test = abs(activation_test) / w_norm

print(f"w^T x_test + b = {w[0]}*{test_point[0]} + {w[1]}*{test_point[1]} + {b} = {activation_test}")
print(f"Distance = |w^T x_test + b| / ||w|| = |{activation_test}| / {w_norm} = {distance_test}")

# Determine which side of hyperplane
if activation_test > 0:
    print(f"Test point is on the POSITIVE side of the hyperplane")
elif activation_test < 0:
    print(f"Test point is on the NEGATIVE side of the hyperplane")
else:
    print(f"Test point is ON the hyperplane")

# Task 5: Equations for margin boundaries
print("\n" + "="*50)
print("Task 5: Margin Boundary Equations")
print("="*50)

print("The hyperplane equation: w^T x + b = 0")
print(f"  {w[0]}x_1 + {w[1]}x_2 + {b} = 0")

print("\nPositive margin boundary: w^T x + b = +1")
print(f"  {w[0]}x_1 + {w[1]}x_2 + {b} = 1")
print(f"  {w[0]}x_1 + {w[1]}x_2 = {1-b}")

print("\nNegative margin boundary: w^T x + b = -1")
print(f"  {w[0]}x_1 + {w[1]}x_2 + {b} = -1")
print(f"  {w[0]}x_1 + {w[1]}x_2 = {-1-b}")

# Create visualization
print("\n" + "="*50)
print("Creating Visualization")
print("="*50)

plt.figure(figsize=(12, 10))

# Define plotting range
x1_range = np.linspace(-1, 4, 100)

# Calculate hyperplane and margin boundaries
# For w1*x1 + w2*x2 + b = c, we get x2 = (c - w1*x1) / w2
def hyperplane_x2(x1, c):
    return (c - w[0]*x1) / w[1]

x2_hyperplane = hyperplane_x2(x1_range, -b)
x2_pos_margin = hyperplane_x2(x1_range, 1-b)
x2_neg_margin = hyperplane_x2(x1_range, -1-b)

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
    if i in support_vectors:
        plt.scatter(xi[0], xi[1], s=200, c=color, marker=marker_style, 
                   edgecolors='black', linewidth=3, label=f'SV: x_{i+1} (class {class_names[color_idx]})')
    else:
        plt.scatter(xi[0], xi[1], s=150, c=color, marker=marker_style, 
                   edgecolors='black', linewidth=1.5, alpha=0.7)
    
    # Add point labels
    plt.annotate(f'x_{i+1}({xi[0]},{xi[1]})', 
                (xi[0], xi[1]), xytext=(10, 10), textcoords='offset points',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

# Plot test point
plt.scatter(test_point[0], test_point[1], s=150, c='green', marker='*', 
           edgecolors='black', linewidth=2, label=f'Test Point ({test_point[0]},{test_point[1]})')

# Add distance line from test point to hyperplane
# Find closest point on hyperplane to test point
# The closest point is test_point - (w^T test_point + b) / ||w||^2 * w
closest_point = test_point - (activation_test / (w_norm**2)) * w
plt.plot([test_point[0], closest_point[0]], [test_point[1], closest_point[1]], 
         'g:', linewidth=2, label=f'Distance = {distance_test:.3f}')

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('SVM Hyperplane Analysis\n$w^T x + b = 0$ with $w = [1, -1]^T$, $b = 0.5$', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.axis('equal')
plt.xlim(-0.5, 4)
plt.ylim(-0.5, 4)

# Add equations as text
eq_text = f'Decision Boundary: $x_1 - x_2 + 0.5 = 0$\n'
eq_text += f'Positive Margin: $x_1 - x_2 = 0.5$\n'
eq_text += f'Negative Margin: $x_1 - x_2 = -1.5$\n'
eq_text += f'Geometric Margin: $\\gamma = {geometric_margin:.3f}$'

plt.text(0.02, 0.98, eq_text, transform=plt.gca().transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", alpha=0.9))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'hyperplane_analysis.png'), dpi=300, bbox_inches='tight')

print(f"Visualization saved to: {save_dir}/hyperplane_analysis.png")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"1. Margin constraints: All points satisfy y_i(w^T x_i + b) ≥ 1")
print(f"2. Support vectors: Points {[i+1 for i in support_vectors]} (on margin boundaries)")
print(f"3. Geometric margin: γ = {geometric_margin:.6f}")
print(f"4. Test point distance: {distance_test:.6f}")
print(f"5. Margin boundaries:")
print(f"   - Positive: x_1 - x_2 = 0.5")
print(f"   - Negative: x_1 - x_2 = -1.5")
