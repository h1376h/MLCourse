import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_1_Quiz_29")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = False  # Disable LaTeX to avoid Unicode issues
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("=" * 80)
print("QUESTION 29: HYPERPLANE CALCULATIONS")
print("=" * 80)

# Given hyperplane: 2x_1 + 3x_2 - 6 = 0
print("\nGiven hyperplane: 2x_1 + 3x_2 - 6 = 0")
print("Point: (1, 1)")
print("Label: y = +1")

# Extract coefficients
a, b, c = 2, 3, -6
point = np.array([1, 1])
label = 1

print(f"\nCoefficients: a = {a}, b = {b}, c = {c}")

# Task 1: Calculate distance from point (1,1) to hyperplane
print("\n" + "="*50)
print("TASK 1: Distance from point (1,1) to hyperplane")
print("="*50)

# Distance formula: |ax_0 + by_0 + c| / sqrt(a^2 + b^2)
numerator = abs(a * point[0] + b * point[1] + c)
denominator = np.sqrt(a**2 + b**2)
distance = numerator / denominator

print(f"Distance formula: |ax_0 + by_0 + c| / sqrt(a^2 + b^2)")
print(f"Numerator: |{a} × {point[0]} + {b} × {point[1]} + {c}| = |{a*point[0]} + {b*point[1]} + {c}| = |{a*point[0] + b*point[1] + c}| = {numerator}")
print(f"Denominator: sqrt({a}^2 + {b}^2) = sqrt({a**2} + {b**2}) = sqrt{a**2 + b**2} = {denominator}")
print(f"Distance = {numerator} / {denominator} = {distance:.4f}")

# Task 2: Functional margin
print("\n" + "="*50)
print("TASK 2: Functional margin")
print("="*50)

# Functional margin = y * (w^T * x + b)
# First, convert to w^T * x + b = 0 form
w = np.array([a, b])  # weight vector
bias = c              # bias term

functional_margin = label * (np.dot(w, point) + bias)
print(f"Weight vector w = [{a}, {b}]")
print(f"Bias term b = {bias}")
print(f"Functional margin = y × (w^T × x + b)")
print(f"Functional margin = {label} × ([{a}, {b}] · [{point[0]}, {point[1]}] + {bias})")
print(f"Functional margin = {label} × ({a*point[0] + b*point[1]} + {bias})")
print(f"Functional margin = {label} × {a*point[0] + b*point[1] + bias}")
print(f"Functional margin = {functional_margin}")

# Task 3: Magnitude of weight vector
print("\n" + "="*50)
print("TASK 3: Magnitude of weight vector ||w||")
print("="*50)

w_magnitude = np.linalg.norm(w)
print(f"||w|| = sqrt(w_1^2 + w_2^2)")
print(f"||w|| = sqrt({a}^2 + {b}^2)")
print(f"||w|| = sqrt({a**2} + {b**2})")
print(f"||w|| = sqrt{a**2 + b**2}")
print(f"||w|| = {w_magnitude:.4f}")

# Task 4: Hyperplane in w^T * x + b = 0 form
print("\n" + "="*50)
print("TASK 4: Hyperplane in w^T * x + b = 0 form")
print("="*50)

print(f"Original form: {a}x_1 + {b}x_2 + {c} = 0")
print(f"w^T * x + b = 0 form: [{a}, {b}]^T * [x_1, x_2] + {bias} = 0")
print(f"Or: {a}x_1 + {b}x_2 + {bias} = 0")

# Task 5: Effect of doubling coefficients
print("\n" + "="*50)
print("TASK 5: Effect of doubling coefficients")
print("="*50)

# Doubled coefficients
a_doubled, b_doubled, c_doubled = 2*a, 2*b, 2*c
w_doubled = np.array([a_doubled, b_doubled])
bias_doubled = c_doubled

print(f"Doubled hyperplane: {a_doubled}x_1 + {b_doubled}x_2 + {c_doubled} = 0")
print(f"Doubled weight vector: w' = [{a_doubled}, {b_doubled}]")

# Calculate new distance
numerator_doubled = abs(a_doubled * point[0] + b_doubled * point[1] + c_doubled)
denominator_doubled = np.sqrt(a_doubled**2 + b_doubled**2)
distance_doubled = numerator_doubled / denominator_doubled

print(f"\nNew distance calculation:")
print(f"Numerator: |{a_doubled} × {point[0]} + {b_doubled} × {point[1]} + {c_doubled}| = {numerator_doubled}")
print(f"Denominator: sqrt({a_doubled}^2 + {b_doubled}^2) = sqrt{a_doubled**2 + b_doubled**2} = {denominator_doubled}")
print(f"New distance = {numerator_doubled} / {denominator_doubled} = {distance_doubled:.4f}")

print(f"\nOriginal distance: {distance:.4f}")
print(f"New distance: {distance_doubled:.4f}")
print(f"Ratio: {distance_doubled/distance:.4f}")

# Geometric margin = functional margin / ||w||
geometric_margin_original = functional_margin / w_magnitude
geometric_margin_doubled = (label * (np.dot(w_doubled, point) + bias_doubled)) / np.linalg.norm(w_doubled)

print(f"\nGeometric margin analysis:")
print(f"Original geometric margin = functional margin / ||w|| = {functional_margin} / {w_magnitude:.4f} = {geometric_margin_original:.4f}")
print(f"New geometric margin = {label * (np.dot(w_doubled, point) + bias_doubled)} / {np.linalg.norm(w_doubled):.4f} = {geometric_margin_doubled:.4f}")
print(f"Geometric margin ratio: {geometric_margin_doubled/geometric_margin_original:.4f}")

# Visualization 1: Original hyperplane and point
plt.figure(figsize=(12, 10))

# Create grid for plotting
x1 = np.linspace(-2, 5, 100)
x2 = np.linspace(-2, 5, 100)
X1, X2 = np.meshgrid(x1, x2)

# Plot hyperplane: 2x_1 + 3x_2 - 6 = 0
# Solve for x_2: x_2 = (-2x_1 + 6) / 3
x2_hyperplane = (-a * x1 + abs(c)) / b
plt.plot(x1, x2_hyperplane, 'g-', linewidth=3, label=f'Hyperplane: {a}x_1 + {b}x_2 + {c} = 0')

# Plot the point
plt.scatter(point[0], point[1], color='red', s=200, zorder=5, label=f'Point ({point[0]}, {point[1]})')

# Draw perpendicular line from point to hyperplane
# Vector from point to hyperplane (perpendicular direction)
perpendicular = w / np.linalg.norm(w)
# Projection of point onto hyperplane
t = -(np.dot(w, point) + bias) / np.dot(w, w)
projection = point + t * w

plt.plot([point[0], projection[0]], [point[1], projection[1]], 'r--', linewidth=2, label=f'Distance = {distance:.4f}')

# Shade regions
Z = a * X1 + b * X2 + c
plt.contourf(X1, X2, Z, levels=[-100, 0], colors=['lightblue'], alpha=0.3)
plt.contourf(X1, X2, Z, levels=[0, 100], colors=['lightpink'], alpha=0.3)

# Add weight vector arrow - place it on the hyperplane and point perpendicular to it
# Choose a point on the hyperplane (e.g., x1 = 1, then x2 = (-a*x1 + abs(c))/b)
arrow_start_x1 = 1
arrow_start_x2 = (-a * arrow_start_x1 + abs(c)) / b
arrow_start = np.array([arrow_start_x1, arrow_start_x2])

# The weight vector should point toward the positive region (where ax1 + bx2 + c > 0)
# Since w = [a, b] = [2, 3], it points in the direction of increasing x1 and x2
arrow_length = 0.8
plt.arrow(arrow_start[0], arrow_start[1], w[0]/np.linalg.norm(w)*arrow_length, w[1]/np.linalg.norm(w)*arrow_length, 
          head_width=0.15, head_length=0.15, fc='blue', ec='blue', linewidth=3, label=f'Weight vector w = [{a}, {b}]')

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('Hyperplane and Point Distance', fontsize=16)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(-1, 4)
plt.ylim(-1, 4)
plt.legend(loc='upper left', fontsize=12)

# Add text annotations
plt.annotate(f'Distance = {distance:.4f}', xy=(point[0], point[1]), xytext=(point[0]+0.5, point[1]+0.5),
             arrowprops=dict(arrowstyle='->', color='red'), fontsize=12, color='red')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'hyperplane_distance.png'), dpi=300, bbox_inches='tight')

# Visualization 2: Comparison of original and doubled hyperplanes
plt.figure(figsize=(12, 10))

# Original hyperplane
x2_original = (-a * x1 + abs(c)) / b
plt.plot(x1, x2_original, 'g-', linewidth=3, label=f'Original: {a}x_1 + {b}x_2 + {c} = 0')

# Doubled hyperplane
x2_doubled = (-a_doubled * x1 + abs(c_doubled)) / b_doubled
plt.plot(x1, x2_doubled, 'b--', linewidth=3, label=f'Doubled: {a_doubled}x_1 + {b_doubled}x_2 + {c_doubled} = 0')

# Plot the point
plt.scatter(point[0], point[1], color='red', s=200, zorder=5, label=f'Point ({point[0]}, {point[1]})')

# Draw distances
# Original distance
t_original = -(np.dot(w, point) + bias) / np.dot(w, w)
projection_original = point + t_original * w
plt.plot([point[0], projection_original[0]], [point[1], projection_original[1]], 'g--', linewidth=2, label=f'Original distance = {distance:.4f}')

# Doubled distance
t_doubled = -(np.dot(w_doubled, point) + bias_doubled) / np.dot(w_doubled, w_doubled)
projection_doubled = point + t_doubled * w_doubled
plt.plot([point[0], projection_doubled[0]], [point[1], projection_doubled[1]], 'b--', linewidth=2, label=f'Doubled distance = {distance_doubled:.4f}')

# Shade regions for original hyperplane
Z_original = a * X1 + b * X2 + c
plt.contourf(X1, X2, Z_original, levels=[-100, 0], colors=['lightgreen'], alpha=0.2)
plt.contourf(X1, X2, Z_original, levels=[0, 100], colors=['lightyellow'], alpha=0.2)

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('Comparison: Original vs Doubled Hyperplane', fontsize=16)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(-1, 4)
plt.ylim(-1, 4)
plt.legend(loc='upper left', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'hyperplane_comparison.png'), dpi=300, bbox_inches='tight')

# Visualization 3: Functional and geometric margins
plt.figure(figsize=(12, 10))

# Plot hyperplane
plt.plot(x1, x2_hyperplane, 'g-', linewidth=3, label=f'Hyperplane: {a}x_1 + {b}x_2 + {c} = 0')

# Plot the point
plt.scatter(point[0], point[1], color='red', s=200, zorder=5, label=f'Point ({point[0]}, {point[1]})')

# Draw functional margin (scaled weight vector)
functional_margin_vector = functional_margin * w / np.linalg.norm(w)
plt.arrow(point[0], point[1], functional_margin_vector[0], functional_margin_vector[1], 
          head_width=0.1, head_length=0.1, fc='orange', ec='orange', linewidth=3, 
          label=f'Functional margin = {functional_margin:.4f}')

# Draw geometric margin (perpendicular distance)
plt.plot([point[0], projection[0]], [point[1], projection[1]], 'purple', linewidth=3, 
         label=f'Geometric margin = {geometric_margin_original:.4f}')

# Shade regions
Z = a * X1 + b * X2 + c
plt.contourf(X1, X2, Z, levels=[-100, 0], colors=['lightblue'], alpha=0.3)
plt.contourf(X1, X2, Z, levels=[0, 100], colors=['lightpink'], alpha=0.3)

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('Functional vs Geometric Margin', fontsize=16)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(-1, 4)
plt.ylim(-1, 4)
plt.legend(loc='upper left', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'margin_visualization.png'), dpi=300, bbox_inches='tight')

print(f"\n" + "="*80)
print("SUMMARY OF RESULTS")
print("="*80)
print(f"1. Distance from (1,1) to hyperplane: {distance:.4f}")
print(f"2. Functional margin: {functional_margin:.4f}")
print(f"3. Weight vector magnitude ||w||: {w_magnitude:.4f}")
print(f"4. Hyperplane form: {a}x_1 + {b}x_2 + {bias} = 0")
print(f"5. After doubling coefficients:")
print(f"   - New distance: {distance_doubled:.4f}")
print(f"   - Distance ratio: {distance_doubled/distance:.4f}")
print(f"   - Geometric margin ratio: {geometric_margin_doubled/geometric_margin_original:.4f}")

print(f"\nPlots saved to: {save_dir}")
print("=" * 80)
