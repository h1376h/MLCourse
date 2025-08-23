import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_1_Quiz_10")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("Question 10: Margin Calculations")
print("=" * 80)

# Given hyperplane: 2x₁ + 3x₂ - 6 = 0
w = np.array([2, 3])
b = -6
w_norm = np.linalg.norm(w)

print(f"Given hyperplane: {w[0]}x₁ + {w[1]}x₂ + {b} = 0")
print(f"Weight vector w = {w}")
print(f"Bias term b = {b}")
print(f"||w|| = {w_norm:.4f}")

# Task 1: Calculate distance from point (1, 2) to hyperplane
print("\n1. DISTANCE FROM POINT (1, 2) TO HYPERPLANE")
print("-" * 50)

point = np.array([1, 2])
distance = abs(np.dot(w, point) + b) / w_norm

print(f"Point: ({point[0]}, {point[1]})")
print(f"Distance formula: d = |w^T x + b| / ||w||")
print(f"d = |{w[0]}×{point[0]} + {w[1]}×{point[1]} + {b}| / {w_norm:.4f}")
print(f"d = |{w[0]*point[0]} + {w[1]*point[1]} + {b}| / {w_norm:.4f}")
print(f"d = |{np.dot(w, point) + b}| / {w_norm:.4f}")
print(f"d = {distance:.4f}")

# Task 2: Functional margin for point (1, 2) with label y = +1
print("\n2. FUNCTIONAL MARGIN")
print("-" * 50)

y_label = 1
functional_margin = y_label * (np.dot(w, point) + b)

print(f"Point: ({point[0]}, {point[1]}) with label y = {y_label}")
print(f"Functional margin = y × (w^T x + b)")
print(f"Functional margin = {y_label} × ({np.dot(w, point) + b})")
print(f"Functional margin = {functional_margin}")

# Task 3: Normalize hyperplane equation
print("\n3. NORMALIZE HYPERPLANE EQUATION")
print("-" * 50)

# For normalization, we want the closest points to have functional margin ±1
# This means we need to scale w and b by 1/min_functional_margin
# Since we don't have the actual closest points, we'll demonstrate the concept

print("To normalize the hyperplane so closest points have functional margin ±1:")
print("We need to scale the hyperplane equation by a factor k such that:")
print("min |y_i(k×w^T x_i + k×b)| = 1")

# For demonstration, let's assume the minimum functional margin is the current one
min_functional_margin = abs(functional_margin)
k = 1 / min_functional_margin

w_normalized = k * w
b_normalized = k * b

print(f"\nAssuming minimum functional margin = {min_functional_margin}")
print(f"Scaling factor k = 1/{min_functional_margin} = {k:.4f}")
print(f"Normalized hyperplane: {w_normalized[0]:.4f}x₁ + {w_normalized[1]:.4f}x₂ + {b_normalized:.4f} = 0")

# Verify normalization
functional_margin_normalized = y_label * (np.dot(w_normalized, point) + b_normalized)
print(f"Functional margin after normalization = {functional_margin_normalized:.4f}")

# Task 4: Geometric margin of normalized hyperplane
print("\n4. GEOMETRIC MARGIN OF NORMALIZED HYPERPLANE")
print("-" * 50)

w_normalized_norm = np.linalg.norm(w_normalized)
geometric_margin = 1 / w_normalized_norm

print(f"For normalized hyperplane:")
print(f"||w_normalized|| = {w_normalized_norm:.4f}")
print(f"Geometric margin = 1/||w_normalized|| = {geometric_margin:.4f}")

# Task 5: Effect of scaling dataset by factor of 2
print("\n5. EFFECT OF SCALING DATASET BY FACTOR 2")
print("-" * 50)

scale_factor = 2
point_scaled = scale_factor * point

print(f"Original point: ({point[0]}, {point[1]})")
print(f"Scaled point: ({point_scaled[0]}, {point_scaled[1]})")

# When we scale the dataset, the hyperplane equation changes
# If all points are scaled by factor s, then w remains the same but b scales by s
# Actually, let's think about this more carefully...

print("\nWhen scaling all data points by factor s:")
print("- If x_new = s × x_old, then for the same hyperplane:")
print("- w^T x_new + b = w^T (s × x_old) + b = s × (w^T x_old) + b")
print("- To maintain the same classification, we need:")
print("- w_new^T x_new + b_new = w^T x_old + b_old")
print("- This gives us: w_new = w/s and b_new = b")

w_scaled = w / scale_factor
b_scaled = b  # b doesn't change when we scale coordinates

print(f"\nScaled hyperplane parameters:")
print(f"w_scaled = {w_scaled}")
print(f"b_scaled = {b_scaled}")

# Calculate new geometric margin
w_scaled_norm = np.linalg.norm(w_scaled)
geometric_margin_scaled = 2 / w_scaled_norm  # Factor of 2 because margin width = 2/||w||

print(f"||w_scaled|| = {w_scaled_norm:.4f}")
print(f"New geometric margin = 2/||w_scaled|| = {geometric_margin_scaled:.4f}")
print(f"Original geometric margin = 2/||w|| = {2/w_norm:.4f}")
print(f"Margin scaling factor = {geometric_margin_scaled/(2/w_norm):.4f}")

# Visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Distance calculation
x1_range = np.linspace(-1, 4, 100)
x2_hyperplane = (-w[0] * x1_range - b) / w[1]

ax1.plot(x1_range, x2_hyperplane, 'k-', linewidth=3, label='Hyperplane')
ax1.scatter(point[0], point[1], c='red', s=150, marker='o', 
           edgecolor='black', linewidth=2, label='Point (1, 2)')

# Find projection of point onto hyperplane
t = -(np.dot(w, point) + b) / np.dot(w, w)
projection = point + t * w

ax1.scatter(projection[0], projection[1], c='blue', s=100, marker='x', 
           linewidth=3, label='Projection')
ax1.plot([point[0], projection[0]], [point[1], projection[1]], 
         'r--', linewidth=2, alpha=0.8, label=f'Distance = {distance:.3f}')

# Add normal vector
normal_start = projection
normal_end = normal_start + 0.3 * w / w_norm
ax1.arrow(normal_start[0], normal_start[1], 
          normal_end[0] - normal_start[0], normal_end[1] - normal_start[1],
          head_width=0.1, head_length=0.1, fc='green', ec='green', linewidth=2)

ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.set_title('Distance from Point to Hyperplane')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.axis('equal')
ax1.set_xlim(-0.5, 3.5)
ax1.set_ylim(-0.5, 3.5)

# Plot 2: Functional vs Geometric margin
margins_demo = np.array([functional_margin, distance])
margin_types = ['Functional\nMargin', 'Geometric\nMargin']
colors = ['lightblue', 'lightcoral']

bars = ax2.bar(margin_types, margins_demo, color=colors, edgecolor='black', linewidth=2)
ax2.set_ylabel('Margin Value')
ax2.set_title('Functional vs Geometric Margin')
ax2.grid(True, alpha=0.3, axis='y')

# Add value annotations
for bar, value in zip(bars, margins_demo):
    height = bar.get_height()
    ax2.annotate(f'{value:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 5), textcoords='offset points', ha='center', va='bottom',
                fontsize=12, fontweight='bold')

# Plot 3: Normalization effect
x1_norm_range = np.linspace(-1, 4, 100)
x2_original = (-w[0] * x1_norm_range - b) / w[1]
x2_normalized = (-w_normalized[0] * x1_norm_range - b_normalized) / w_normalized[1]

ax3.plot(x1_norm_range, x2_original, 'b-', linewidth=2, label='Original Hyperplane')
ax3.plot(x1_norm_range, x2_normalized, 'r--', linewidth=2, label='Normalized Hyperplane')
ax3.scatter(point[0], point[1], c='red', s=150, marker='o', 
           edgecolor='black', linewidth=2, label='Point (1, 2)')

ax3.set_xlabel('$x_1$')
ax3.set_ylabel('$x_2$')
ax3.set_title('Original vs Normalized Hyperplane')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.set_xlim(-0.5, 3.5)
ax3.set_ylim(-0.5, 3.5)

# Plot 4: Scaling effect
x1_scale_range = np.linspace(-2, 8, 100)
x2_original_scale = (-w[0] * x1_scale_range - b) / w[1]
x2_scaled_coords = (-w_scaled[0] * x1_scale_range - b_scaled) / w_scaled[1]

ax4.plot(x1_scale_range, x2_original_scale, 'b-', linewidth=2, label='Original Scale')
ax4.plot(x1_scale_range, x2_scaled_coords, 'g--', linewidth=2, label='After 2× Scaling')

ax4.scatter(point[0], point[1], c='red', s=150, marker='o', 
           edgecolor='black', linewidth=2, label='Original Point')
ax4.scatter(point_scaled[0], point_scaled[1], c='orange', s=150, marker='s', 
           edgecolor='black', linewidth=2, label='Scaled Point')

ax4.set_xlabel('$x_1$')
ax4.set_ylabel('$x_2$')
ax4.set_title('Effect of Dataset Scaling')
ax4.grid(True, alpha=0.3)
ax4.legend()
ax4.set_xlim(-1, 7)
ax4.set_ylim(-1, 5)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'margin_calculations_overview.png'), dpi=300, bbox_inches='tight')
plt.close()

# Simple visualization: Margin concepts
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Hyperplane: 2x₁ + 3x₂ - 6 = 0
x1_simple = np.linspace(-1, 4, 100)
x2_simple = (6 - 2 * x1_simple) / 3

# Plot hyperplane
ax.plot(x1_simple, x2_simple, 'k-', linewidth=3, label='Hyperplane')

# Test point
test_point = np.array([1, 2])
ax.scatter(test_point[0], test_point[1], c='red', s=200, marker='o',
           edgecolor='black', linewidth=3, label='Test Point (1,2)')

# Find projection point on hyperplane
# For line 2x + 3y - 6 = 0, projection of point (x0,y0) is:
w = np.array([2, 3])
b = -6
t = -(np.dot(w, test_point) + b) / np.dot(w, w)
projection = test_point + t * w

ax.scatter(projection[0], projection[1], c='blue', s=150, marker='x',
           linewidth=4, label='Projection')

# Draw distance line
ax.plot([test_point[0], projection[0]], [test_point[1], projection[1]],
        'g--', linewidth=3, alpha=0.8, label='Distance')

# Draw normal vector
normal_scale = 0.5
normal_end = projection + normal_scale * w / np.linalg.norm(w)
ax.arrow(projection[0], projection[1],
         normal_end[0] - projection[0], normal_end[1] - projection[1],
         head_width=0.1, head_length=0.1, fc='purple', ec='purple', linewidth=2)

ax.set_xlabel('x₁')
ax.set_ylabel('x₂')
ax.set_title('Distance from Point to Hyperplane')
ax.grid(True, alpha=0.3)
ax.legend()
ax.axis('equal')
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(-0.5, 3.5)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'margin_simple.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAll visualizations saved to: {save_dir}")
print("=" * 80)
