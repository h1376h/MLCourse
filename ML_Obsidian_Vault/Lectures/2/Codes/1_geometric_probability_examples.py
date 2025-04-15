import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Circle, Rectangle, ConnectionPatch, Polygon
from matplotlib.collections import PatchCollection

print("\n=== GEOMETRIC PROBABILITY EXAMPLES: VISUALIZATIONS ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the parent directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Example 1: Random Point in a Rectangle
print("Example 1: Random Point in a Rectangle")
print("A point is selected uniformly at random within a rectangle with width 4 and height 3.")
print("What is the probability that the point lies within a circle of radius 1 centered at the origin?")

# Calculate the analytical result
rectangle_area = 4 * 3  # width * height
circle_area = np.pi * 1**2  # π * r²
quarter_circle_area = circle_area / 4  # Only 1/4 of the circle is in the rectangle
probability = quarter_circle_area / rectangle_area

print(f"Rectangle area: {rectangle_area} square units")
print(f"Quarter circle area: {quarter_circle_area:.4f} square units")
print(f"Probability (analytical): {probability:.6f} or {probability*100:.4f}%")

# Create visualization for Example 1
fig, ax = plt.subplots(figsize=(8, 6))

# Draw rectangle
rectangle = Rectangle((0, 0), 4, 3, fill=False, edgecolor='blue', linewidth=2)
ax.add_patch(rectangle)

# Draw quarter circle
quarter_circle = Circle((0, 0), 1, fill=True, alpha=0.3, color='red')
ax.add_patch(quarter_circle)

# Set axis limits
ax.set_xlim(-0.5, 4.5)
ax.set_ylim(-0.5, 3.5)
ax.set_aspect('equal')

# Add grid
ax.grid(True, linestyle='--', alpha=0.7)

# Add labels and title
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Example 1: Random Point in a Rectangle', fontsize=14)

# Save the figure
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'geometric_prob_rectangle_circle.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Random Points on a Line Segment
print("\nExample 2: Random Points on a Line Segment")
print("Three points are selected randomly and independently on a line segment of length L.")
print("What is the probability that all three points lie in the same half of the segment?")

# Calculate the analytical result
p_left_half = 0.5**3  # Probability all three points are in left half
p_right_half = 0.5**3  # Probability all three points are in right half
probability = p_left_half + p_right_half

print(f"Probability all three points in left half: {p_left_half}")
print(f"Probability all three points in right half: {p_right_half}")
print(f"Probability all three points in same half: {probability} or {probability*100}%")

# Create visualization for Example 2
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

# Line segment length
L = 10

# Draw full line segment
ax1.plot([0, L], [0, 0], 'k-', linewidth=2)
ax1.axvline(x=L/2, color='red', linestyle='--', linewidth=1)
ax1.scatter([L/4, L/3, 3*L/8], [0, 0, 0], color='blue', s=80)
ax1.set_xlim(-1, L+1)
ax1.set_ylim(-1, 1)
ax1.set_title("Case 1: All points in left half")
ax1.set_yticks([])

# Draw full line segment - right half example
ax2.plot([0, L], [0, 0], 'k-', linewidth=2)
ax2.axvline(x=L/2, color='red', linestyle='--', linewidth=1)
ax2.scatter([2*L/3, 3*L/4, 7*L/8], [0, 0, 0], color='green', s=80)
ax2.set_xlim(-1, L+1)
ax2.set_ylim(-1, 1)
ax2.set_title("Case 2: All points in right half")
ax2.set_yticks([])

# Draw full line segment - mixed example (non-matching case)
ax3.plot([0, L], [0, 0], 'k-', linewidth=2)
ax3.axvline(x=L/2, color='red', linestyle='--', linewidth=1)
ax3.scatter([L/4, 2*L/3, 7*L/8], [0, 0, 0], color='red', s=80)
ax3.set_xlim(-1, L+1)
ax3.set_ylim(-1, 1)
ax3.set_title("Non-matching case: Points in both halves")
ax3.set_yticks([])

# Add title
fig.suptitle('Example 2: Random Points on a Line Segment', fontsize=16)

# Save the figure
plt.tight_layout(rect=[0, 0.01, 1, 0.95])
plt.savefig(os.path.join(images_dir, 'geometric_prob_line_segment.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Overlapping Squares
print("\nExample 3: Overlapping Squares")
print("Two squares, each with side length 1, are positioned such that one corner of the second square")
print("is at the center of the first square. What is the probability that a point selected uniformly")
print("at random from the first square also lies within the second square?")

# Calculate the analytical result
overlap_area = 0.5 * 0.5  # Area of overlapping region
first_square_area = 1  # Area of first square
probability = overlap_area / first_square_area

print(f"First square area: {first_square_area} square units")
print(f"Overlapping area: {overlap_area} square units")
print(f"Probability: {probability} or {probability*100}%")

# Create visualization for Example 3
fig, ax = plt.subplots(figsize=(8, 8))

# Draw first square
first_square = Rectangle((0, 0), 1, 1, fill=False, edgecolor='blue', linewidth=2, label='First Square')
ax.add_patch(first_square)

# Draw second square
second_square = Rectangle((0.5, 0.5), 1, 1, fill=False, edgecolor='red', linewidth=2, label='Second Square')
ax.add_patch(second_square)

# Fill overlap region
overlap = Rectangle((0.5, 0.5), 0.5, 0.5, fill=True, alpha=0.3, color='purple', label='Overlap')
ax.add_patch(overlap)

# Set axis limits
ax.set_xlim(-0.1, 1.6)
ax.set_ylim(-0.1, 1.6)
ax.set_aspect('equal')

# Add grid
ax.grid(True, linestyle='--', alpha=0.7)

# Add labels and title
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Example 3: Overlapping Squares', fontsize=14)

# Add legend
ax.legend(loc='upper right')

# Save the figure
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'geometric_prob_overlapping_squares.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 4: Random Chord Length
print("\nExample 4: Random Chord Length")
print("A chord is drawn randomly in a circle of radius R. What is the probability")
print("that the length of the chord is greater than the radius of the circle?")

# Calculate the analytical result
probability = 3/4

print(f"Probability (analytical): {probability} or {probability*100}%")

# Create visualization for Example 4
fig, ax = plt.subplots(figsize=(10, 8))

# Draw the circle
circle = Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=2)
ax.add_patch(circle)

# Draw the inner circle that determines which chords are longer than radius
inner_circle = Circle((0, 0), np.sqrt(3)/2, fill=True, alpha=0.3, color='lightblue')
ax.add_patch(inner_circle)

# Draw coordinate axes
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.7)
ax.axvline(x=0, color='gray', linestyle='-', alpha=0.7)

# Draw some example chords (perpendicular to lines from center to points)
np.random.seed(42)  # For reproducibility
n_examples = 5
theta_samples = np.random.uniform(0, 2*np.pi, n_examples)
r_samples = np.random.uniform(0, 1, n_examples)

chord_colors = ['red', 'blue', 'green', 'orange', 'purple']

for i, (theta, r) in enumerate(zip(theta_samples, r_samples)):
    # Point on the circle
    x_point = r * np.cos(theta)
    y_point = r * np.sin(theta)
    
    # Direction perpendicular to radius
    perp_angle = theta + np.pi/2
    dx = np.cos(perp_angle)
    dy = np.sin(perp_angle)
    
    # Calculate chord endpoints
    chord_half_length = np.sqrt(1 - r**2)
    x1 = x_point + chord_half_length * dx
    y1 = y_point + chord_half_length * dy
    x2 = x_point - chord_half_length * dx
    y2 = y_point - chord_half_length * dy
    
    # Draw the chord
    ax.plot([x1, x2], [y1, y2], color=chord_colors[i], linewidth=2)
    
    # Draw the radius to the random point
    ax.plot([0, x_point], [0, y_point], 'k--', alpha=0.5)

# Set axis limits
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')

# Add labels and title
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Example 4: Random Chord Length', fontsize=14)

# Save the figure
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'geometric_prob_chord_length.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 5: Buffon's Needle Problem
print("\nExample 5: Buffon's Needle Problem")
print("A needle of length L is dropped randomly onto a floor with parallel lines spaced a distance D apart")
print("(where D ≥ L). What is the probability that the needle will cross one of the lines?")

# Calculate the analytical result for L = D
L = 1  # Needle length
D = 1  # Line spacing
probability = 2*L/(D*np.pi)

print(f"Needle length (L): {L}")
print(f"Line spacing (D): {D}")
print(f"Probability (analytical): {probability:.6f} or {probability*100:.4f}%")

# Create visualization for Example 5
fig, ax = plt.subplots(figsize=(10, 8))

# Draw parallel lines
n_lines = 5
line_positions = np.arange(n_lines) * D
for pos in line_positions:
    ax.axhline(y=pos, color='black', linestyle='-', linewidth=1)

# Draw some example needles
np.random.seed(42)  # For reproducibility
n_examples = 20
x_positions = np.random.uniform(0, (n_lines-1)*D, n_examples)
y_positions = np.random.uniform(0, (n_lines-1)*D, n_examples)
angles = np.random.uniform(0, np.pi, n_examples)

# Count needles crossing lines
crossings = 0

for x, y, angle in zip(x_positions, y_positions, angles):
    # Calculate needle endpoints
    x1 = x - L/2 * np.cos(angle)
    y1 = y - L/2 * np.sin(angle)
    x2 = x + L/2 * np.cos(angle)
    y2 = y + L/2 * np.sin(angle)
    
    # Check if needle crosses a line
    closest_line = np.floor(y / D) * D
    next_line = closest_line + D
    
    crosses = False
    if (y1 <= closest_line and y2 >= closest_line) or (y1 >= closest_line and y2 <= closest_line) or \
       (y1 <= next_line and y2 >= next_line) or (y1 >= next_line and y2 <= next_line):
        crosses = True
        crossings += 1
        
    # Draw the needle
    color = 'red' if crosses else 'blue'
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=2)

# Set axis limits
ax.set_xlim(-0.5, (n_lines-1)*D + 0.5)
ax.set_ylim(-0.5, (n_lines-1)*D + 0.5)
ax.set_aspect('equal')

# Add labels and title
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Example 5: Buffon\'s Needle Problem', fontsize=14)

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='red', lw=2, label='Crosses a line'),
    Line2D([0], [0], color='blue', lw=2, label='Does not cross')
]
ax.legend(handles=legend_elements, loc='upper right')

# Print statistics for markdown
empirical_prob = crossings / n_examples
pi_estimate = 2*L/(D*empirical_prob) if empirical_prob > 0 else "undefined"
print(f"Monte Carlo simulation: {crossings} crossings out of {n_examples} needles")
print(f"Empirical probability: {empirical_prob:.4f}")
print(f"π estimate from simulation: {pi_estimate if isinstance(pi_estimate, str) else pi_estimate:.4f}")

# Save the figure
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'geometric_prob_buffon_needle.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll geometric probability example images created successfully.") 