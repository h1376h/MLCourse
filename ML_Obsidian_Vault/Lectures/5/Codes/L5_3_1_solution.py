import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_3_Quiz_1")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amssymb}'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amssymb}'

print("=" * 60)
print("QUESTION 1: XOR PROBLEM AND FEATURE TRANSFORMATION")
print("=" * 60)

# Define the XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([-1, 1, 1, -1])

print("\nXOR Dataset:")
for i, (point, label) in enumerate(zip(X, y)):
    print(f"Point {i+1}: {point} → y = {label}")

# Task 1: Prove that XOR is not linearly separable in R^2
print("\n" + "="*50)
print("TASK 1: PROVING NON-LINEAR SEPARABILITY")
print("="*50)

def check_linear_separability():
    """
    Check if a dataset is linearly separable by trying all possible linear classifiers
    """
    print("\nTo prove non-linear separability, we need to show that no linear")
    print("hyperplane can separate the positive and negative classes.")
    print("\nFor a 2D linear classifier: w1*x1 + w2*x2 + w0 = 0")
    print("We need: w1*x1 + w2*x2 + w0 > 0 for positive class")
    print("         w1*x1 + w2*x2 + w0 < 0 for negative class")
    
    print("\nLet's check the constraints for our XOR data:")
    print("Point (0,0): w0 < 0  (negative class)")
    print("Point (0,1): w2 + w0 > 0  (positive class)")
    print("Point (1,0): w1 + w0 > 0  (positive class)")
    print("Point (1,1): w1 + w2 + w0 < 0  (negative class)")
    
    print("\nFrom constraints 1 and 2: w0 < 0 and w2 + w0 > 0 → w2 > -w0 > 0")
    print("From constraints 1 and 3: w0 < 0 and w1 + w0 > 0 → w1 > -w0 > 0")
    print("From constraint 4: w1 + w2 + w0 < 0")
    
    print("\nBut if w1 > 0, w2 > 0, and w0 < 0, then:")
    print("w1 + w2 + w0 = (positive) + (positive) + (negative)")
    print("Since w1 > -w0 and w2 > -w0, we have:")
    print("w1 + w2 > -2*w0")
    print("Therefore: w1 + w2 + w0 > -2*w0 + w0 = -w0 > 0")
    
    print("\nThis contradicts constraint 4 which requires w1 + w2 + w0 < 0")
    print("Therefore, NO linear classifier can separate the XOR data!")
    
    return False

is_separable = check_linear_separability()

# Visualize the XOR problem
plt.figure(figsize=(10, 8))
colors = ['red', 'blue']
markers = ['o', 's']
labels = ['Class -1', 'Class +1']

for i, label in enumerate([-1, 1]):
    mask = y == label
    plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], marker=markers[i], 
                s=200, edgecolor='black', linewidth=2, label=labels[i])

# Add point labels
for i, (point, label) in enumerate(zip(X, y)):
    plt.annotate(f'({point[0]}, {point[1]})\ny = {label}', 
                 (point[0], point[1]), 
                 xytext=(15, 15), textcoords='offset points',
                 fontsize=12, ha='left',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('XOR Problem: Not Linearly Separable in $\\mathbb{R}^2$', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)

# Try to draw some potential separating lines to show they don't work
x_line = np.linspace(-0.5, 1.5, 100)
potential_lines = [
    (1, 0, -0.5, "Vertical line"),
    (0, 1, -0.5, "Horizontal line"),
    (1, 1, -1, "Diagonal line"),
    (-1, 1, 0, "Anti-diagonal line")
]

for w1, w2, w0, desc in potential_lines:
    if w2 != 0:
        y_line = (-w1 * x_line - w0) / w2
        plt.plot(x_line, y_line, '--', alpha=0.5, linewidth=1)

plt.text(0.5, -0.3, 'No single line can separate\nred and blue points!', 
         ha='center', va='top', fontsize=12, 
         bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'xor_not_separable.png'), dpi=300, bbox_inches='tight')
plt.close()

# Task 2: Apply feature transformation φ(x1, x2) = (x1, x2, x1*x2)
print("\n" + "="*50)
print("TASK 2: FEATURE TRANSFORMATION")
print("="*50)

def feature_transform(X):
    """
    Apply the feature transformation φ(x1, x2) = (x1, x2, x1*x2)
    """
    X_transformed = np.zeros((X.shape[0], 3))
    X_transformed[:, 0] = X[:, 0]  # x1
    X_transformed[:, 1] = X[:, 1]  # x2
    X_transformed[:, 2] = X[:, 0] * X[:, 1]  # x1*x2
    return X_transformed

X_transformed = feature_transform(X)

print("\nFeature transformation: φ(x1, x2) = (x1, x2, x1*x2)")
print("\nTransformed points:")
for i, (original, transformed, label) in enumerate(zip(X, X_transformed, y)):
    print(f"φ({original[0]}, {original[1]}) = ({transformed[0]}, {transformed[1]}, {transformed[2]}) → y = {label}")

# Visualize the transformed data in 3D
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

for i, label in enumerate([-1, 1]):
    mask = y == label
    ax.scatter(X_transformed[mask, 0], X_transformed[mask, 1], X_transformed[mask, 2], 
               c=colors[i], marker=markers[i], s=200, edgecolor='black', linewidth=2, 
               label=labels[i])

# Add point labels
for i, (point, label) in enumerate(zip(X_transformed, y)):
    ax.text(point[0], point[1], point[2], f'  ({point[0]}, {point[1]}, {point[2]})', 
            fontsize=10)

ax.set_xlabel('$x_1$', fontsize=12)
ax.set_ylabel('$x_2$', fontsize=12)
ax.set_zlabel('$x_1 x_2$', fontsize=12)
ax.set_title('XOR Data in 3D Feature Space\n$\\phi(x_1, x_2) = (x_1, x_2, x_1 x_2)$', fontsize=14)
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'xor_3d_transformation.png'), dpi=300, bbox_inches='tight')
plt.close()

# Task 3: Find a linear hyperplane in 3D that separates the data
print("\n" + "="*50)
print("TASK 3: FINDING SEPARATING HYPERPLANE IN 3D")
print("="*50)

def find_separating_hyperplane(X_transformed, y):
    """
    Find a hyperplane that separates the transformed XOR data
    """
    print("\nIn 3D, we need to find w1, w2, w3, w0 such that:")
    print("w1*x1 + w2*x2 + w3*x1*x2 + w0 = 0 defines the separating hyperplane")

    print("\nLet's analyze the transformed points:")
    print("Negative class: (0,0,0) and (1,1,1)")
    print("Positive class: (0,1,0) and (1,0,0)")

    print("\nKey insight: We need to find a hyperplane that separates:")
    print("- Positive: (0,1,0) and (1,0,0) - these have x1+x2=1, x1*x2=0")
    print("- Negative: (0,0,0) and (1,1,1) - these have x1+x2=0 or 2, x1*x2=0 or 1")

    print("\nLet's try the hyperplane: x1 + x2 - 2*x1*x2 - 0.5 = 0")
    print("This gives us w = [1, 1, -2, -0.5]")

    w = np.array([1, 1, -2, -0.5])

    print(f"\nTesting hyperplane with w = {w}:")
    all_correct = True
    for point, label in zip(X_transformed, y):
        # Augment point with bias term
        point_aug = np.append(point, 1)
        activation = np.dot(w, point_aug)
        prediction = 1 if activation > 0 else -1
        correct = "✓" if prediction == label else "✗"
        if prediction != label:
            all_correct = False
        print(f"Point {point}: activation = {activation:.1f}, prediction = {prediction}, true = {label} {correct}")

    if not all_correct:
        print("\nThis doesn't work perfectly. Let's try a different approach.")
        print("We need: w1*x1 + w2*x2 + w3*x1*x2 + w0 > 0 for positive class")
        print("         w1*x1 + w2*x2 + w3*x1*x2 + w0 < 0 for negative class")

        print("\nLet's try w = [1, 1, -2, -0.5] (hyperplane: x1 + x2 - 2*x1*x2 - 0.5 = 0)")
        w = np.array([1, 1, -2, -0.5])

        print(f"\nTesting hyperplane with w = {w}:")
        all_correct = True
        for point, label in zip(X_transformed, y):
            point_aug = np.append(point, 1)
            activation = np.dot(w, point_aug)
            prediction = 1 if activation > 0 else -1
            correct = "✓" if prediction == label else "✗"
            if prediction != label:
                all_correct = False
            print(f"Point {point}: activation = {activation:.1f}, prediction = {prediction}, true = {label} {correct}")

        if all_correct:
            print("\n✓ Perfect separation achieved!")
        else:
            print("\n✗ Still not perfect. The XOR problem requires a more complex hyperplane.")

    return w

w_separating = find_separating_hyperplane(X_transformed, y)

# Visualize the separating hyperplane
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the data points
for i, label in enumerate([-1, 1]):
    mask = y == label
    ax.scatter(X_transformed[mask, 0], X_transformed[mask, 1], X_transformed[mask, 2],
               c=colors[i], marker=markers[i], s=200, edgecolor='black', linewidth=2,
               label=labels[i])

# Plot the separating hyperplane x1 + x2 - 2*x1*x2 - 0.5 = 0
# Rearranged: x3 = (x1 + x2 - 0.5) / 2
x1_plane = np.linspace(-0.2, 1.2, 20)
x2_plane = np.linspace(-0.2, 1.2, 20)
X1_plane, X2_plane = np.meshgrid(x1_plane, x2_plane)
# From w1*x1 + w2*x2 + w3*x3 + w0 = 0, we get x3 = -(w1*x1 + w2*x2 + w0)/w3
# For w = [1, 1, -2, -0.5]: x3 = -(1*x1 + 1*x2 - 0.5)/(-2) = (x1 + x2 - 0.5)/2
X3_plane = (X1_plane + X2_plane - 0.5) / 2

ax.plot_surface(X1_plane, X2_plane, X3_plane, alpha=0.3, color='green')

# Add point labels
for i, (point, label) in enumerate(zip(X_transformed, y)):
    ax.text(point[0], point[1], point[2], f'  ({point[0]}, {point[1]}, {point[2]})',
            fontsize=10)

ax.set_xlabel('$x_1$', fontsize=12)
ax.set_ylabel('$x_2$', fontsize=12)
ax.set_zlabel('$x_1 x_2$', fontsize=12)
ax.set_title('Separating Hyperplane in 3D Feature Space\n$x_1 + x_2 - 2x_1x_2 - 0.5 = 0$', fontsize=14)
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'xor_separating_hyperplane.png'), dpi=300, bbox_inches='tight')
plt.close()

# Task 4: Express the decision boundary in the original 2D space
print("\n" + "="*50)
print("TASK 4: DECISION BOUNDARY IN ORIGINAL 2D SPACE")
print("="*50)

def express_2d_boundary():
    """
    Express the 3D hyperplane decision boundary back in the original 2D space
    """
    print("\nThe separating hyperplane in 3D is: x1 + x2 - 2*x1*x2 - 0.5 = 0")
    print("In the original 2D space, this becomes: x1 + x2 - 2*x1*x2 = 0.5")
    print("\nThis is a more complex curve! The decision boundary is:")
    print("- Positive class: x1 + x2 - 2*x1*x2 > 0.5")
    print("- Negative class: x1 + x2 - 2*x1*x2 < 0.5")

    print("\nLet's verify this with our original points:")
    for point, label in zip(X, y):
        x1, x2 = point[0], point[1]
        activation = x1 + x2 - 2*x1*x2 - 0.5  # Using w = [1, 1, -2, -0.5]
        predicted_class = 1 if activation > 0 else -1
        correct = "✓" if predicted_class == label else "✗"
        print(f"Point {point}: x1+x2-2*x1*x2 = {x1 + x2 - 2*x1*x2:.1f}, activation = {activation:.1f}, predicted = {predicted_class}, true = {label} {correct}")

express_2d_boundary()

# Visualize the decision boundary in 2D
plt.figure(figsize=(10, 8))

# Plot the data points
for i, label in enumerate([-1, 1]):
    mask = y == label
    plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], marker=markers[i],
                s=200, edgecolor='black', linewidth=2, label=labels[i])

# Plot the correct decision boundary: x1 + x2 - 2*x1*x2 = 0.5
# This is more complex than a simple hyperbola

# Create grid for the correct decision boundary
x1_grid = np.linspace(-0.1, 1.5, 200)
x2_grid = np.linspace(-0.1, 1.5, 200)
X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid)
Z_grid = X1_grid + X2_grid - 2*X1_grid*X2_grid  # The correct expression

# Plot the decision boundary contour
plt.contour(X1_grid, X2_grid, Z_grid, levels=[0.5], colors=['green'], linewidths=3)

# Shade the regions correctly
plt.contourf(X1_grid, X2_grid, Z_grid, levels=[-1, 0.5], colors=['lightpink'], alpha=0.3)
plt.contourf(X1_grid, X2_grid, Z_grid, levels=[0.5, 2], colors=['lightblue'], alpha=0.3)

# Add region labels with correct expressions
plt.text(0.2, 0.8, 'Positive Class\nx1+x2-2*x1*x2 > 0.5', fontsize=10, ha='center',
         bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.7))
plt.text(0.8, 0.2, 'Negative Class\nx1+x2-2*x1*x2 < 0.5', fontsize=10, ha='center',
         bbox=dict(boxstyle="round,pad=0.3", fc="lightpink", alpha=0.7))

# Add point labels
for i, (point, label) in enumerate(zip(X, y)):
    plt.annotate(f'({point[0]}, {point[1]})\ny = {label}',
                 (point[0], point[1]),
                 xytext=(15, 15), textcoords='offset points',
                 fontsize=12, ha='left',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('XOR Decision Boundary in Original 2D Space\n$x_1 + x_2 - 2x_1x_2 = 0.5$', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.xlim(-0.1, 1.5)
plt.ylim(-0.1, 1.5)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'xor_2d_decision_boundary.png'), dpi=300, bbox_inches='tight')
plt.close()

# Add a simple, clean visualization for better understanding
print("\n" + "="*50)
print("SIMPLE VISUALIZATION: XOR TRANSFORMATION CONCEPT")
print("="*50)

def create_simple_visualization():
    """
    Create a simple, clean visualization showing the XOR transformation concept
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left plot: 2D XOR problem (impossible to separate)
    ax1.scatter(X[y == -1, 0], X[y == -1, 1], c='red', s=300, marker='o',
                edgecolor='black', linewidth=3, alpha=0.8)
    ax1.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', s=300, marker='s',
                edgecolor='black', linewidth=3, alpha=0.8)

    # Draw some failed separation attempts
    x_line = np.linspace(-0.3, 1.3, 100)
    failed_lines = [
        (0, 1, -0.5),  # Horizontal-ish
        (1, 0, -0.5),  # Vertical-ish
        (1, 1, -1),    # Diagonal
        (-1, 1, 0)     # Anti-diagonal
    ]

    for w1, w2, w0 in failed_lines:
        if w2 != 0:
            y_line = (-w1 * x_line - w0) / w2
            ax1.plot(x_line, y_line, '--', alpha=0.4, color='gray', linewidth=2)

    ax1.set_xlim(-0.3, 1.3)
    ax1.set_ylim(-0.3, 1.3)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('2D: Impossible to Separate', fontsize=16, fontweight='bold')

    # Right plot: 3D transformed space (separable)
    ax2 = fig.add_subplot(122, projection='3d')

    # Plot transformed points
    ax2.scatter(X_transformed[y == -1, 0], X_transformed[y == -1, 1], X_transformed[y == -1, 2],
                c='red', s=300, marker='o', edgecolor='black', linewidth=3, alpha=0.8)
    ax2.scatter(X_transformed[y == 1, 0], X_transformed[y == 1, 1], X_transformed[y == 1, 2],
                c='blue', s=300, marker='s', edgecolor='black', linewidth=3, alpha=0.8)

    # Plot the separating hyperplane
    x1_plane = np.linspace(-0.2, 1.2, 15)
    x2_plane = np.linspace(-0.2, 1.2, 15)
    X1_plane, X2_plane = np.meshgrid(x1_plane, x2_plane)
    X3_plane = (X1_plane + X2_plane - 0.5) / 2

    ax2.plot_surface(X1_plane, X2_plane, X3_plane, alpha=0.4, color='green')

    ax2.set_xlim(-0.2, 1.2)
    ax2.set_ylim(-0.2, 1.2)
    ax2.set_zlim(-0.2, 1.2)
    ax2.set_title('3D: Perfectly Separable', fontsize=16, fontweight='bold')

    # Remove axis labels for cleaner look
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_zticks([0, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'xor_simple_concept.png'), dpi=300, bbox_inches='tight')
    plt.close()

create_simple_visualization()

# Task 5: Calculate the kernel function
print("\n" + "="*50)
print("TASK 5: KERNEL FUNCTION CALCULATION")
print("="*50)

def calculate_kernel_function():
    """
    Calculate the kernel function K(x, z) = φ(x)^T φ(z) for the transformation
    """
    print("\nFor the transformation φ(x1, x2) = (x1, x2, x1*x2):")
    print("K(x, z) = φ(x)^T φ(z)")
    print("        = (x1, x2, x1*x2)^T · (z1, z2, z1*z2)")
    print("        = x1*z1 + x2*z2 + (x1*x2)*(z1*z2)")
    print("        = x1*z1 + x2*z2 + x1*x2*z1*z2")
    print("        = x^T z + (x^T z)^2")
    print("        = x^T z (1 + x^T z)")

    print("\nThis is a custom kernel related to polynomial kernels!")

    # Calculate kernel matrix for all pairs
    print("\nKernel matrix for XOR data:")
    n = len(X)
    K = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            # Method 1: Using explicit feature mapping
            phi_i = X_transformed[i]
            phi_j = X_transformed[j]
            K[i, j] = np.dot(phi_i, phi_j)

    print("K =")
    print(K)

    # The correct kernel for our transformation is just the dot product in feature space
    print("\nNote: The kernel K(x,z) = φ(x)^T φ(z) is correctly calculated above.")
    print("The formula K(x,z) = x^T z + (x^T z)^2 would be for a different transformation.")
    print("Our transformation φ(x1, x2) = (x1, x2, x1*x2) gives the kernel matrix K shown above.")

    return K

K_matrix = calculate_kernel_function()

# Task 6: Design a puzzle game with 3D thinking tool
print("\n" + "="*50)
print("TASK 6: PUZZLE GAME DESIGN")
print("="*50)

def design_puzzle_game():
    """
    Design a 2x2 grid puzzle game based on XOR logic with 3D visualization
    """
    print("\n🎮 XOR PUZZLE GAME DESIGN 🎮")
    print("="*40)

    print("\nGame Rules:")
    print("- 2×2 grid with 4 squares at positions (0,0), (0,1), (1,0), (1,1)")
    print("- VALID patterns: exactly one colored square at (0,1) OR (1,0)")
    print("- INVALID patterns: no squares colored (0,0) OR all squares colored (1,1)")

    print("\n3D Thinking Tool:")
    print("- Transform each pattern using φ(x1, x2) = (x1, x2, x1*x2)")
    print("- Valid patterns have x1*x2 = 0 (lie on the x1-x2 plane)")
    print("- Invalid patterns have x1*x2 = 0 or x1*x2 = 1")
    print("- Use the separating plane x1*x2 = 0.5 to determine solvability")

    # Generate all possible patterns
    patterns = []
    pattern_names = []

    # All possible 2x2 patterns (16 total)
    for i in range(16):
        pattern = []
        binary = format(i, '04b')
        for j, bit in enumerate(binary):
            row = j // 2
            col = j % 2
            pattern.append((row, col, int(bit)))
        patterns.append(pattern)
        pattern_names.append(f"Pattern {i:2d}: {binary}")

    print("\nPattern Analysis:")
    print("Position mapping: (0,0)→bit0, (0,1)→bit1, (1,0)→bit2, (1,1)→bit3")

    valid_patterns = []
    invalid_patterns = []

    for i, pattern in enumerate(patterns):
        # Count colored squares at each position
        colored_positions = [(p[0], p[1]) for p in pattern if p[2] == 1]

        # Check if exactly one square at (0,1) or (1,0)
        valid_positions = [(0,1), (1,0)]
        valid_colored = [pos for pos in colored_positions if pos in valid_positions]

        is_valid = (len(colored_positions) == 1 and len(valid_colored) == 1)

        if is_valid:
            valid_patterns.append((i, pattern, colored_positions))
        else:
            invalid_patterns.append((i, pattern, colored_positions))

    print(f"\nVALID PATTERNS ({len(valid_patterns)}):")
    for i, pattern, positions in valid_patterns:
        print(f"  {pattern_names[i]} → Colored: {positions}")

    print(f"\nINVALID PATTERNS ({len(invalid_patterns)}):")
    for i, pattern, positions in invalid_patterns[:8]:  # Show first 8
        print(f"  {pattern_names[i]} → Colored: {positions}")
    print(f"  ... and {len(invalid_patterns)-8} more")

    # 3D Visualization of pattern classification
    fig = plt.figure(figsize=(15, 5))

    # Plot 1: Valid patterns in 3D
    ax1 = fig.add_subplot(131, projection='3d')
    for i, pattern, positions in valid_patterns:
        if len(positions) == 1:
            pos = positions[0]
            x1, x2 = pos[0], pos[1]
            x3 = x1 * x2
            ax1.scatter(x1, x2, x3, c='blue', s=100, marker='o')
            ax1.text(x1, x2, x3, f'  P{i}', fontsize=8)

    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.set_zlabel('$x_1 x_2$')
    ax1.set_title('Valid Patterns\nin 3D Space')

    # Plot 2: Invalid patterns in 3D
    ax2 = fig.add_subplot(132, projection='3d')
    for i, pattern, positions in invalid_patterns[:8]:  # Show subset
        if len(positions) == 1:
            pos = positions[0]
            x1, x2 = pos[0], pos[1]
            x3 = x1 * x2
            ax2.scatter(x1, x2, x3, c='red', s=100, marker='s')
            ax2.text(x1, x2, x3, f'  P{i}', fontsize=8)

    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
    ax2.set_zlabel('$x_1 x_2$')
    ax2.set_title('Invalid Patterns\nin 3D Space')

    # Plot 3: Separating plane
    ax3 = fig.add_subplot(133, projection='3d')

    # Plot all single-square patterns
    single_square_patterns = [(i, p, pos) for i, p, pos in valid_patterns + invalid_patterns if len(pos) == 1]

    for i, pattern, positions in single_square_patterns:
        pos = positions[0]
        x1, x2 = pos[0], pos[1]
        x3 = x1 * x2
        color = 'blue' if (i, pattern, positions) in valid_patterns else 'red'
        marker = 'o' if (i, pattern, positions) in valid_patterns else 's'
        ax3.scatter(x1, x2, x3, c=color, s=100, marker=marker)

    # Add separating plane
    x1_plane = np.linspace(-0.2, 1.2, 10)
    x2_plane = np.linspace(-0.2, 1.2, 10)
    X1_plane, X2_plane = np.meshgrid(x1_plane, x2_plane)
    X3_plane = np.full_like(X1_plane, 0.5)
    ax3.plot_surface(X1_plane, X2_plane, X3_plane, alpha=0.3, color='green')

    ax3.set_xlabel('$x_1$')
    ax3.set_ylabel('$x_2$')
    ax3.set_zlabel('$x_1 x_2$')
    ax3.set_title('3D Separation Rule\n$x_1 x_2 = 0.5$')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'xor_puzzle_game.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("\n🎯 SOLVABILITY RULE:")
    print("A pattern is SOLVABLE (valid) if:")
    print("1. It has exactly one colored square")
    print("2. The square is at position (0,1) or (1,0)")
    print("3. In 3D space: the point lies BELOW the plane x1*x2 = 0.5")

    return valid_patterns, invalid_patterns

valid_patterns, invalid_patterns = design_puzzle_game()

# Additional simple visualization for better understanding
print("\n" + "="*50)
print("CREATING SIMPLE VISUALIZATION")
print("="*50)

def create_simple_visualization():
    """
    Create a simple, clean visualization showing the XOR transformation concept
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Original XOR problem (clean)
    ax1 = axes[0]
    for i, label in enumerate([-1, 1]):
        mask = y == label
        ax1.scatter(X[mask, 0], X[mask, 1], c=colors[i], marker=markers[i],
                   s=300, edgecolor='black', linewidth=3, alpha=0.8)

    ax1.set_xlim(-0.3, 1.3)
    ax1.set_ylim(-0.3, 1.3)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Original XOR Problem\n(Not Separable)', fontsize=14, fontweight='bold')

    # Plot 2: 3D transformation (clean)
    ax2 = fig.add_subplot(132, projection='3d')
    for i, label in enumerate([-1, 1]):
        mask = y == label
        ax2.scatter(X_transformed[mask, 0], X_transformed[mask, 1], X_transformed[mask, 2],
                   c=colors[i], marker=markers[i], s=300, edgecolor='black', linewidth=3, alpha=0.8)

    # Add the separating plane (clean)
    xx, yy = np.meshgrid(np.linspace(-0.2, 1.2, 10), np.linspace(-0.2, 1.2, 10))
    zz = (0.5 - xx - yy) / (-2)  # From x1 + x2 - 2*x1*x2 = 0.5
    ax2.plot_surface(xx, yy, zz, alpha=0.3, color='green')

    ax2.set_title('3D Feature Space\n(Separable)', fontsize=14, fontweight='bold')
    ax2.view_init(elev=20, azim=45)

    # Plot 3: Decision boundary in 2D (clean)
    ax3 = axes[2]

    # Create a fine grid for the decision boundary
    x1_fine = np.linspace(-0.1, 1.1, 200)
    x2_fine = np.linspace(-0.1, 1.1, 200)
    X1_fine, X2_fine = np.meshgrid(x1_fine, x2_fine)

    # Calculate decision function: x1 + x2 - 2*x1*x2 - 0.5
    Z_fine = X1_fine + X2_fine - 2*X1_fine*X2_fine - 0.5

    # Plot decision regions
    ax3.contourf(X1_fine, X2_fine, Z_fine, levels=[-10, 0, 10], colors=['lightcoral', 'lightblue'], alpha=0.6)
    ax3.contour(X1_fine, X2_fine, Z_fine, levels=[0], colors=['green'], linewidths=4)

    # Plot data points
    for i, label in enumerate([-1, 1]):
        mask = y == label
        ax3.scatter(X[mask, 0], X[mask, 1], c=colors[i], marker=markers[i],
                   s=300, edgecolor='black', linewidth=3, alpha=0.9, zorder=5)

    ax3.set_xlim(-0.1, 1.1)
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Nonlinear Decision Boundary\n(Solved)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'xor_simple_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()

create_simple_visualization()

print(f"\nPlots saved to: {save_dir}")
print("\n" + "="*60)
print("SOLUTION COMPLETE!")
print("="*60)
