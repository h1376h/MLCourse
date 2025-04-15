import matplotlib.pyplot as plt
import numpy as np
import os

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Create a figure with a clean, modern look
plt.figure(figsize=(10, 8))

# Set up the plot with simpler style
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
plt.xlim(-1, 8)
plt.ylim(-1, 8)
plt.grid(False)

# Increase text size and use a cleaner font
plt.rcParams.update({'font.size': 14})
plt.rcParams['text.usetex'] = False

# Draw hyperplane
x = np.linspace(-1, 7, 100)
y = -x + 6  # hyperplane with w = [1, 1], b = -6
plt.plot(x, y, color='magenta', linewidth=3)

# Draw positive examples (blue +)
pos_x = [4.2, 5.5, 3.2, 5.8, 6.2]
pos_y = [6.3, 5.2, 4.3, 3.8, 2.7]
plt.scatter(pos_x, pos_y, color='blue', marker='+', s=200, linewidths=2)

# Draw negative examples (red -)
neg_x = [0.8, 1.7, 2.7, 1.3, 2.3]
neg_y = [0.9, 2.3, 0.8, 2.9, 3.3]
plt.scatter(neg_x, neg_y, color='red', marker='_', s=200, linewidths=2)

# Draw the weight vector w - make it thicker and more prominent
# Start the arrow from a point on the hyperplane
# For hyperplane y = -x + 6, let's pick a point (2.5, 3.5)
# w = [1, 1] so the arrow should be perpendicular to the hyperplane
plt.arrow(2.5, 3.5, 1, 1, head_width=0.3, head_length=0.4, fc='black', ec='black', linewidth=2)
plt.text(3.5, 4.8, r'$\mathbf{w}$', fontsize=24, fontweight='bold')

# Add text annotations with better positioning and larger text
plt.text(5, 6.5, r'On this side:', fontsize=16)
plt.text(5, 6, r'$\mathbf{w}^T\mathbf{x} + b > 0$', fontsize=16)

plt.text(0.5, 0.7, r'On this side:', fontsize=16)
plt.text(0.5, 0.2, r'$\mathbf{w}^T\mathbf{x} + b < 0$', fontsize=16)

# Add hyperplane label
plt.text(2.5, -1, r'Hyperplane perpendicular to $\mathbf{w}$', color='magenta', fontsize=16, fontweight='bold')
plt.text(2.5, -1.5, r'$H = \{\mathbf{x}: \mathbf{w}^T\mathbf{x} + b = 0\}$', color='magenta', fontsize=16)

# Add tick marks for better readability
plt.xticks(range(0, 8))
plt.yticks(range(0, 8))

# Save the figure with high quality
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'perceptron_hyperplane.png'), dpi=300, bbox_inches='tight')
print("Updated perceptron diagram saved to perceptron_hyperplane.png") 