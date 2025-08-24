import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_1_Quiz_22")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("=" * 80)
print("QUESTION 22: DERIVING THE GEOMETRIC MARGIN FORMULA")
print("=" * 80)

# Step 1: Define the hyperplane equation
print("\nSTEP 1: Starting with the hyperplane equation")
print("-" * 50)

# Let's use a simple 2D example for visualization
w = np.array([2, 1])  # Weight vector
b = -3               # Bias term

print(f"Hyperplane equation: w^T x + b = 0")
print(f"Where w = {w}, b = {b}")
print(f"Equation: {w[0]}x₁ + {w[1]}x₂ + {b} = 0")
print(f"Or: x₂ = {-w[0]/w[1]}x₁ + {-b/w[1]}")

# Step 2: Define the margin boundaries
print("\nSTEP 2: Defining the margin boundaries")
print("-" * 50)

print("Margin boundaries are parallel hyperplanes:")
print(f"Upper boundary: w^T x + b = 1")
print(f"  {w[0]}x₁ + {w[1]}x₂ + {b} = 1")
print(f"  x₂ = {-w[0]/w[1]}x₁ + {(1-b)/w[1]}")

print(f"Lower boundary: w^T x + b = -1")
print(f"  {w[0]}x₁ + {w[1]}x₂ + {b} = -1")
print(f"  x₂ = {-w[0]/w[1]}x₁ + {(-1-b)/w[1]}")

# Step 3: Calculate the distance between parallel hyperplanes
print("\nSTEP 3: Deriving the distance between parallel hyperplanes")
print("-" * 50)

print("The distance between two parallel hyperplanes w^T x + b = c₁ and w^T x + b = c₂ is:")
print("d = |c₂ - c₁| / ||w||")

print(f"In our case:")
print(f"  c₁ = 1, c₂ = -1")
print(f"  d = |(-1) - 1| / ||w|| = 2 / ||w||")

# Calculate ||w||
w_norm = np.linalg.norm(w)
print(f"  ||w|| = √({w[0]}² + {w[1]}²) = √({w[0]**2} + {w[1]**2}) = √{w[0]**2 + w[1]**2} = {w_norm}")
print(f"  d = 2 / {w_norm} = {2/w_norm:.4f}")

# Step 4: Geometric interpretation
print("\nSTEP 4: Geometric interpretation")
print("-" * 50)

print("The weight vector w is perpendicular to the hyperplane.")
print("The distance from a point x₀ to the hyperplane w^T x + b = 0 is:")
print("d = |w^T x₀ + b| / ||w||")

print("For the margin boundaries:")
print("  Distance to upper boundary: |w^T x + b - 1| / ||w||")
print("  Distance to lower boundary: |w^T x + b + 1| / ||w||")

# Create comprehensive visualization
def create_comprehensive_visualization():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Basic hyperplane and margin boundaries
    x1_range = np.linspace(-2, 4, 100)
    
    # Main hyperplane
    x2_main = (-w[0]*x1_range - b) / w[1]
    ax1.plot(x1_range, x2_main, 'b-', linewidth=2, label='Main Hyperplane: $\\mathbf{w}^T\\mathbf{x} + b = 0$')
    
    # Upper margin boundary
    x2_upper = (-w[0]*x1_range - b + 1) / w[1]
    ax1.plot(x1_range, x2_upper, 'g--', linewidth=2, label='Upper Margin: $\\mathbf{w}^T\\mathbf{x} + b = 1$')
    
    # Lower margin boundary
    x2_lower = (-w[0]*x1_range - b - 1) / w[1]
    ax1.plot(x1_range, x2_lower, 'r--', linewidth=2, label='Lower Margin: $\\mathbf{w}^T\\mathbf{x} + b = -1$')
    
    # Add weight vector
    center_point = np.array([1, 0])
    arrow = FancyArrowPatch(center_point, center_point + w/2, 
                           arrowstyle='->', mutation_scale=20, color='purple', linewidth=3)
    ax1.add_patch(arrow)
    ax1.text(center_point[0] + w[0]/2 + 0.1, center_point[1] + w[1]/2, 
             '$\\mathbf{w}$', fontsize=14, color='purple')
    
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.set_title('Hyperplane and Margin Boundaries')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(-2, 4)
    ax1.set_ylim(-2, 4)
    ax1.axis('equal')
    
    # Plot 2: Distance calculation
    # Choose a point on the upper boundary
    x0_upper = np.array([0, (1-b)/w[1]])
    x0_lower = np.array([0, (-1-b)/w[1]])
    
    # Plot the boundaries
    ax2.plot(x1_range, x2_upper, 'g--', linewidth=2, label='Upper Boundary')
    ax2.plot(x1_range, x2_lower, 'r--', linewidth=2, label='Lower Boundary')
    
    # Plot the points
    ax2.scatter(x0_upper[0], x0_upper[1], color='green', s=100, zorder=5, label='Point on Upper Boundary')
    ax2.scatter(x0_lower[0], x0_lower[1], color='red', s=100, zorder=5, label='Point on Lower Boundary')
    
    # Draw the distance line
    ax2.plot([x0_upper[0], x0_lower[0]], [x0_upper[1], x0_lower[1]], 'k-', linewidth=2, label='Distance')
    
    # Add distance annotation
    mid_point = (x0_upper + x0_lower) / 2
    ax2.annotate(f'$d = \\frac{{2}}{{||\\mathbf{{w}}||}} = {2/w_norm:.3f}$', 
                xy=mid_point, xytext=(mid_point[0] + 0.5, mid_point[1]),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))
    
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
    ax2.set_title('Distance Between Margin Boundaries')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(-2, 4)
    ax2.set_ylim(-2, 4)
    ax2.axis('equal')
    
    # Plot 3: Perpendicular distance from point to hyperplane
    # Choose a test point
    test_point = np.array([2, 2])
    
    # Calculate distance from test point to main hyperplane
    distance_to_main = abs(np.dot(w, test_point) + b) / w_norm
    
    # Projection of test point onto hyperplane
    projection = test_point - distance_to_main * w / w_norm
    
    ax3.plot(x1_range, x2_main, 'b-', linewidth=2, label='Hyperplane')
    ax3.scatter(test_point[0], test_point[1], color='red', s=100, zorder=5, label='Test Point')
    ax3.scatter(projection[0], projection[1], color='blue', s=100, zorder=5, label='Projection')
    
    # Draw perpendicular line
    ax3.plot([test_point[0], projection[0]], [test_point[1], projection[1]], 'k--', linewidth=2, label='Perpendicular Distance')
    
    # Add distance annotation
    ax3.annotate(f'$d = \\frac{{|\\mathbf{{w}}^T\\mathbf{{x}} + b|}}{{||\\mathbf{{w}}||}} = {distance_to_main:.3f}$', 
                xy=((test_point + projection)/2), xytext=(test_point[0] + 0.5, test_point[1] + 0.5),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))
    
    ax3.set_xlabel('$x_1$')
    ax3.set_ylabel('$x_2$')
    ax3.set_title('Distance from Point to Hyperplane')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlim(-2, 4)
    ax3.set_ylim(-2, 4)
    ax3.axis('equal')
    
    # Plot 4: Margin width visualization
    # Create a rectangle to show the margin width
    margin_width = 2 / w_norm
    
    # Find points on both boundaries
    x1_margin = np.linspace(-1, 3, 50)
    x2_upper_margin = (-w[0]*x1_margin - b + 1) / w[1]
    x2_lower_margin = (-w[0]*x1_margin - b - 1) / w[1]
    
    ax4.plot(x1_margin, x2_upper_margin, 'g-', linewidth=3, label='Upper Margin Boundary')
    ax4.plot(x1_margin, x2_lower_margin, 'r-', linewidth=3, label='Lower Margin Boundary')
    
    # Fill the margin region
    ax4.fill_between(x1_margin, x2_lower_margin, x2_upper_margin, alpha=0.3, color='yellow', label='Margin Region')
    
    # Add margin width annotation
    mid_x = 1
    mid_y_upper = (-w[0]*mid_x - b + 1) / w[1]
    mid_y_lower = (-w[0]*mid_x - b - 1) / w[1]
    
    ax4.annotate(f'Margin Width = $\\frac{{2}}{{||\\mathbf{{w}}||}} = {margin_width:.3f}$', 
                xy=(mid_x, (mid_y_upper + mid_y_lower)/2), xytext=(mid_x + 1, (mid_y_upper + mid_y_lower)/2),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2),
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    ax4.set_xlabel('$x_1$')
    ax4.set_ylabel('$x_2$')
    ax4.set_title('Margin Width Visualization')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_xlim(-2, 4)
    ax4.set_ylim(-2, 4)
    ax4.axis('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'geometric_margin_derivation.png'), dpi=300, bbox_inches='tight')
    plt.show()

# Step 5: Mathematical proof
print("\nSTEP 5: Mathematical proof")
print("-" * 50)

print("Let's prove that the distance between two parallel hyperplanes is |c₂ - c₁| / ||w||")
print("\nProof:")
print("1. Consider two parallel hyperplanes:")
print("   H₁: w^T x + b = c₁")
print("   H₂: w^T x + b = c₂")
print("\n2. The distance from a point x₀ to hyperplane H₁ is:")
print("   d₁ = |w^T x₀ + b - c₁| / ||w||")
print("\n3. For a point x₀ on H₂, we have w^T x₀ + b = c₂")
print("   So, d₁ = |c₂ - c₁| / ||w||")
print("\n4. This is the perpendicular distance between H₁ and H₂")

# Step 6: Verification with numerical example
print("\nSTEP 6: Numerical verification")
print("-" * 50)

# Choose a point on the upper boundary
x0 = np.array([0, (1-b)/w[1]])
print(f"Point on upper boundary: x₀ = {x0}")

# Calculate distance from this point to lower boundary
distance_calc = abs(np.dot(w, x0) + b - (-1)) / w_norm
print(f"Distance from x₀ to lower boundary:")
print(f"  d = |w^T x₀ + b - (-1)| / ||w||")
print(f"  d = |{np.dot(w, x0)} + {b} + 1| / {w_norm}")
print(f"  d = |{np.dot(w, x0) + b + 1}| / {w_norm}")
print(f"  d = {distance_calc:.4f}")

print(f"\nThis matches our formula: 2 / ||w|| = {2/w_norm:.4f}")

# Create the comprehensive visualization
create_comprehensive_visualization()

# Step 7: Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("We have successfully derived the geometric margin formula:")
print("1. Started with hyperplane equation: w^T x + b = 0")
print("2. Defined margin boundaries: w^T x + b = ±1")
print("3. Used the distance formula between parallel hyperplanes")
print("4. Proved that the margin width is 2 / ||w||")
print("5. Verified the result numerically")

print(f"\nKey insights:")
print("- The weight vector w is perpendicular to the hyperplane")
print("- The margin boundaries are parallel to the main hyperplane")
print("- The distance between parallel hyperplanes is |c₂ - c₁| / ||w||")
print("- For SVM, we use c₁ = 1 and c₂ = -1, giving margin width = 2 / ||w||")
print("- Maximizing the margin is equivalent to minimizing ||w||")

print(f"\nPlots saved to: {save_dir}")
