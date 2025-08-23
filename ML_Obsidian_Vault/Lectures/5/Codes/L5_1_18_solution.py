import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_1_Quiz_18")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

print("=" * 60)
print("Question 18: 3D Geometric Analysis")
print("=" * 60)

# Given hyperplane equation: 2x₁ - x₂ + 3x₃ = 6
print("\nGiven hyperplane equation: 2x₁ - x₂ + 3x₃ = 6")

# Task 1: Identify w and b from the hyperplane equation
print("\n" + "="*50)
print("Task 1: Identify w and b")
print("="*50)

# Standard form: w^T x + b = 0
# Given: 2x₁ - x₂ + 3x₃ = 6
# Rearrange: 2x₁ - x₂ + 3x₃ - 6 = 0
# So: w = [2, -1, 3]^T and b = -6

w = np.array([2, -1, 3])
b = -6

print(f"From 2x₁ - x₂ + 3x₃ = 6:")
print(f"Rearranging: 2x₁ - x₂ + 3x₃ - 6 = 0")
print(f"Standard form: w^T x + b = 0")
print(f"Therefore:")
print(f"  w = {w}")
print(f"  b = {b}")

# Task 2: Compute margin width
print("\n" + "="*50)
print("Task 2: Compute Margin Width")
print("="*50)

w_norm = np.linalg.norm(w)
margin_width = 2 / w_norm

print(f"||w|| = ||{w}|| = √({w[0]}² + {w[1]}² + {w[2]}²)")
print(f"     = √({w[0]**2} + {w[1]**2} + {w[2]**2}) = √{np.sum(w**2)} = {w_norm}")
print(f"")
print(f"Margin width = 2/||w|| = 2/{w_norm} = {margin_width}")

# Task 3: Classify test points
print("\n" + "="*50)
print("Task 3: Classify Test Points")
print("="*50)

test_points = np.array([
    [1, 0, 2],    # Point a
    [3, 1, 1],    # Point b  
    [0, -2, 2]    # Point c
])
point_names = ['a', 'b', 'c']

print("For classification, we evaluate w^T x + b:")
print("If w^T x + b > 0: positive class")
print("If w^T x + b < 0: negative class")
print("If w^T x + b = 0: on the hyperplane")

classifications = []
for i, (point, name) in enumerate(zip(test_points, point_names)):
    # Calculate w^T x + b
    activation = np.dot(w, point) + b
    
    if activation > 0:
        classification = "Positive"
    elif activation < 0:
        classification = "Negative"
    else:
        classification = "On hyperplane"
    
    classifications.append(classification)
    
    print(f"\nPoint {name} = {point}:")
    print(f"  w^T x + b = {w[0]}*{point[0]} + {w[1]}*{point[1]} + {w[2]}*{point[2]} + {b}")
    print(f"            = {w[0]*point[0]} + {w[1]*point[1]} + {w[2]*point[2]} + {b}")
    print(f"            = {activation}")
    print(f"  Classification: {classification}")

# Task 4: Calculate signed distances
print("\n" + "="*50)
print("Task 4: Calculate Signed Distances")
print("="*50)

print("Signed distance formula: d = (w^T x + b) / ||w||")
print("Note: We use w^T x - 6 since the original equation is 2x₁ - x₂ + 3x₃ = 6")
print("This is equivalent to w^T x + b where b = -6")

distances = []
for i, (point, name) in enumerate(zip(test_points, point_names)):
    # Calculate signed distance using the original form: w^T x - 6
    numerator = np.dot(w, point) - 6  # This is w^T x + b where b = -6
    signed_distance = numerator / w_norm
    distances.append(signed_distance)
    
    print(f"\nPoint {name} = {point}:")
    print(f"  w^T x - 6 = {w[0]}*{point[0]} + {w[1]}*{point[1]} + {w[2]}*{point[2]} - 6")
    print(f"            = {w[0]*point[0]} + {w[1]*point[1]} + {w[2]*point[2]} - 6")
    print(f"            = {numerator}")
    print(f"  d = {numerator} / {w_norm} = {signed_distance}")
    
    if signed_distance > 0:
        print(f"  Point is {abs(signed_distance):.3f} units on the POSITIVE side")
    elif signed_distance < 0:
        print(f"  Point is {abs(signed_distance):.3f} units on the NEGATIVE side")
    else:
        print(f"  Point is ON the hyperplane")

# Task 5: Write equations for margin boundaries
print("\n" + "="*50)
print("Task 5: Margin Boundary Equations")
print("="*50)

print("For SVM margin boundaries, we have:")
print("Positive margin: w^T x + b = +||w||")
print("Negative margin: w^T x + b = -||w||")
print("")
print("Using our values:")
print(f"w^T x + b = ±{w_norm}")
print("")
print("Substituting w and b:")
print(f"Positive margin: {w[0]}x₁ + {w[1]}x₂ + {w[2]}x₃ + {b} = +{w_norm}")
print(f"                {w[0]}x₁ + {w[1]}x₂ + {w[2]}x₃ = {w_norm - b}")
print(f"                2x₁ - x₂ + 3x₃ = {w_norm - b}")
print("")
print(f"Negative margin: {w[0]}x₁ + {w[1]}x₂ + {w[2]}x₃ + {b} = -{w_norm}")
print(f"                {w[0]}x₁ + {w[1]}x₂ + {w[2]}x₃ = {-w_norm - b}")
print(f"                2x₁ - x₂ + 3x₃ = {-w_norm - b}")

# Create 3D visualization
print("\n" + "="*50)
print("Creating 3D Visualization")
print("="*50)

fig = plt.figure(figsize=(15, 5))

# Plot 1: 3D scatter plot with hyperplane
ax1 = fig.add_subplot(131, projection='3d')

# Create a mesh for the hyperplane
x1_range = np.linspace(-2, 4, 20)
x2_range = np.linspace(-3, 3, 20)
X1, X2 = np.meshgrid(x1_range, x2_range)

# From 2x₁ - x₂ + 3x₃ = 6, solve for x₃: x₃ = (6 - 2x₁ + x₂) / 3
X3 = (6 - 2*X1 + X2) / 3

# Plot the hyperplane
ax1.plot_surface(X1, X2, X3, alpha=0.3, color='gray', label='Hyperplane')

# Plot test points
colors = ['red', 'blue', 'green']
for i, (point, name, color) in enumerate(zip(test_points, point_names, colors)):
    ax1.scatter(point[0], point[1], point[2], c=color, s=100, 
               label=f'Point {name} ({classifications[i]})')
    
    # Add text labels
    ax1.text(point[0], point[1], point[2], f'  {name}', fontsize=10)

ax1.set_xlabel(r'$x_1$')
ax1.set_ylabel(r'$x_2$')
ax1.set_zlabel(r'$x_3$')
ax1.set_title('3D Hyperplane and Test Points')
ax1.legend()

# Plot 2: Distance visualization
ax2 = fig.add_subplot(132)

point_names_full = ['Point a', 'Point b', 'Point c']
colors_bar = ['red' if d > 0 else 'blue' for d in distances]

bars = ax2.bar(point_names_full, distances, color=colors_bar, alpha=0.7)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.set_ylabel('Signed Distance')
ax2.set_title('Signed Distances to Hyperplane')
ax2.grid(True, alpha=0.3)

# Add value labels on bars
for bar, dist in zip(bars, distances):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.1),
             f'{dist:.3f}', ha='center', va='bottom' if height >= 0 else 'top')

# Plot 3: Margin boundaries illustration
ax3 = fig.add_subplot(133)

# For 2D illustration, fix x₃ = 2 and show x₁-x₂ plane
x3_fixed = 2
print(f"\nFor visualization, fixing x₃ = {x3_fixed}:")
print(f"Hyperplane becomes: 2x₁ - x₂ + 3({x3_fixed}) = 6")
print(f"                   2x₁ - x₂ = {6 - 3*x3_fixed}")

x1_2d = np.linspace(-1, 4, 100)
# 2x₁ - x₂ = 6 - 3*x3_fixed => x₂ = 2x₁ - (6 - 3*x3_fixed)
x2_hyperplane_2d = 2*x1_2d - (6 - 3*x3_fixed)
x2_pos_margin_2d = 2*x1_2d - (6 - 3*x3_fixed - w_norm)
x2_neg_margin_2d = 2*x1_2d - (6 - 3*x3_fixed + w_norm)

ax3.plot(x1_2d, x2_hyperplane_2d, 'k-', linewidth=2, label='Hyperplane')
ax3.plot(x1_2d, x2_pos_margin_2d, 'r--', linewidth=1.5, label='Positive Margin')
ax3.plot(x1_2d, x2_neg_margin_2d, 'b--', linewidth=1.5, label='Negative Margin')

# Fill margin region
ax3.fill_between(x1_2d, x2_pos_margin_2d, x2_neg_margin_2d, alpha=0.2, color='gray', label='Margin Region')

# Plot test points that have x₃ = 2
for i, (point, name, color) in enumerate(zip(test_points, point_names, colors)):
    if point[2] == x3_fixed:
        ax3.scatter(point[0], point[1], c=color, s=100, label=f'Point {name}')
        ax3.text(point[0], point[1], f'  {name}', fontsize=10)

ax3.set_xlabel(r'$x_1$')
ax3.set_ylabel(r'$x_2$')
ax3.set_title(f'2D Cross-section ($x_3 = {x3_fixed}$)')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.axis('equal')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, '3d_geometric_analysis.png'), dpi=300, bbox_inches='tight')

print(f"3D visualization saved to: {save_dir}/3d_geometric_analysis.png")

# Create additional visualization: Vector analysis
print("\n" + "="*50)
print("Creating Additional Visualization: Vector and Distance Analysis")
print("="*50)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Weight vector components
w_labels = [r'$w_1$', r'$w_2$', r'$w_3$']
colors_w = ['red', 'green', 'blue']
bars1 = ax1.bar(w_labels, w, color=colors_w, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Weight Value')
ax1.set_title('Weight Vector Components')
ax1.grid(True, alpha=0.3)
for bar, weight in zip(bars1, w):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.2),
             f'{weight}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')

# Plot 2: Point coordinates
point_coords = test_points.T  # Transpose to get coordinates by dimension
x_pos = np.arange(len(point_names))
width = 0.25
for i, (coord, label, color) in enumerate(zip(point_coords, [r'$x_1$', r'$x_2$', r'$x_3$'], colors_w)):
    ax2.bar(x_pos + i*width, coord, width, label=label, color=color, alpha=0.7, edgecolor='black')

ax2.set_xlabel('Points')
ax2.set_ylabel('Coordinate Value')
ax2.set_title('Test Point Coordinates')
ax2.set_xticks(x_pos + width)
ax2.set_xticklabels([f'Point {name}' for name in point_names])
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Distance analysis
bars3 = ax3.bar([f'Point {name}' for name in point_names], distances,
                color=['orange' if d > 0 else 'purple' for d in distances], alpha=0.7, edgecolor='black')
ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax3.set_ylabel('Signed Distance')
ax3.set_title('Signed Distances from Hyperplane')
ax3.grid(True, alpha=0.3)
for bar, dist in zip(bars3, distances):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{dist:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: Geometric properties
properties = [r'$\|\mathbf{w}\|$', 'Margin Width', r'Bias $|b|$']
values = [w_norm, margin_width, abs(b)]
colors_prop = ['navy', 'darkgreen', 'darkred']
bars4 = ax4.bar(properties, values, color=colors_prop, alpha=0.7, edgecolor='black')
ax4.set_ylabel('Value')
ax4.set_title('Geometric Properties')
ax4.grid(True, alpha=0.3)
for bar, val in zip(bars4, values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'vector_distance_analysis.png'), dpi=300, bbox_inches='tight')

print(f"Additional visualization saved to: {save_dir}/vector_distance_analysis.png")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"1. Weight vector: w = {w}")
print(f"2. Bias term: b = {b}")
print(f"3. Margin width: 2/||w|| = {margin_width:.6f}")
print(f"4. Classifications:")
for i, (name, classification) in enumerate(zip(point_names, classifications)):
    print(f"   Point {name}: {classification}")
print(f"5. Signed distances:")
for i, (name, distance) in enumerate(zip(point_names, distances)):
    print(f"   Point {name}: {distance:.6f}")
print(f"6. Margin boundaries:")
print(f"   Positive: 2x₁ - x₂ + 3x₃ = {w_norm - b:.6f}")
print(f"   Negative: 2x₁ - x₂ + 3x₃ = {-w_norm - b:.6f}")
