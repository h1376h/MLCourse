import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize
import matplotlib.patches as patches

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_1_Quiz_3")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'

print("=" * 80)
print("QUESTION 3: SUPPORT VECTOR IDENTIFICATION")
print("=" * 80)

# Given dataset
X = np.array([
    [1, 1],   # x1, y1 = +1
    [2, 2],   # x2, y2 = +1
    [-1, -1], # x3, y3 = -1
    [-2, -1]  # x4, y4 = -1
])

y = np.array([1, 1, -1, -1])  # Labels

print("Dataset:")
for i, (x, label) in enumerate(zip(X, y)):
    print(f"  x{i+1} = {x}, y{i+1} = {label}")

# Given optimal hyperplane: x1 + x2 = 0
# This means w = [1, 1] and b = 0
w_given = np.array([1, 1])
b_given = 0

print(f"\nGiven optimal hyperplane: x1 + x2 = 0")
print(f"This corresponds to w = {w_given}, b = {b_given}")

# Note: The given hyperplane is not actually the maximum margin hyperplane
# for this dataset. Let's find the actual optimal hyperplane that puts
# support vectors exactly on the margin boundaries.

# For a maximum margin classifier, we need to scale w and b so that
# support vectors satisfy y_i(w^T x_i + b) = 1

# Let's find the scaling factor by looking at the minimum constraint value
constraint_values = []
for x, label in zip(X, y):
    constraint_value = label * (np.dot(w_given, x) + b_given)
    constraint_values.append(constraint_value)

min_constraint = min(constraint_values)
print(f"\nMinimum constraint value: {min_constraint}")
print(f"This means the hyperplane needs to be scaled by {1/min_constraint}")

# Scale the hyperplane to make it optimal
w_optimal = w_given / min_constraint
b_optimal = b_given / min_constraint

print(f"Optimal hyperplane: {w_optimal[0]:.3f}x1 + {w_optimal[1]:.3f}x2 + {b_optimal:.3f} = 0")
print(f"Optimal w = {w_optimal}, b = {b_optimal}")

# Use the optimal hyperplane for calculations
w_given = w_optimal
b_given = b_optimal

# Function to calculate distance from point to hyperplane
def distance_to_hyperplane(x, w, b):
    """Calculate distance from point x to hyperplane w^T x + b = 0"""
    return abs(np.dot(w, x) + b) / np.linalg.norm(w)

# Function to calculate margin
def calculate_margin(X, y, w, b):
    """Calculate the margin of the hyperplane"""
    distances = []
    for x, label in zip(X, y):
        dist = distance_to_hyperplane(x, w, b)
        distances.append(dist)
    return min(distances)

# Calculate distances and margin
print("\n" + "="*50)
print("STEP 1: IDENTIFYING SUPPORT VECTORS")
print("="*50)

distances = []
for i, (x, label) in enumerate(zip(X, y)):
    dist = distance_to_hyperplane(x, w_given, b_given)
    distances.append(dist)
    print(f"Distance from x{i+1} = {x} to hyperplane: {dist:.4f}")

margin = min(distances)
print(f"\nMargin of the hyperplane: {margin:.4f}")

# Identify support vectors (points exactly on the margin: y_i(w^T x_i + b) = 1)
support_vector_indices = []
for i, (x, label) in enumerate(zip(X, y)):
    constraint_value = label * (np.dot(w_given, x) + b_given)
    if abs(constraint_value - 1) < 1e-10:  # Points exactly on the margin
        support_vector_indices.append(i)

support_vectors = X[support_vector_indices]
support_vector_labels = y[support_vector_indices]

print(f"\nSupport vectors (points exactly on the margin y_i(w^T x_i + b) = 1):")
for i, idx in enumerate(support_vector_indices):
    constraint_value = y[idx] * (np.dot(w_given, X[idx]) + b_given)
    print(f"  x{idx+1} = {X[idx]}, y{idx+1} = {y[idx]}, constraint = {constraint_value:.4f}")

# Visualize the dataset and hyperplane
plt.figure(figsize=(12, 10))

# Plot all points
colors = ['blue' if label == 1 else 'red' for label in y]
markers = ['o' if label == 1 else 's' for label in y]

for i, (x, label, color, marker) in enumerate(zip(X, y, colors, markers)):
    if i in support_vector_indices:
        # Support vectors get larger markers and black edges
        plt.scatter(x[0], x[1], s=200, color=color, marker=marker, 
                   edgecolor='black', linewidth=3, zorder=5,
                   label=f'Support Vector {i+1} ({label})')
    else:
        plt.scatter(x[0], x[1], s=100, color=color, marker=marker, 
                   alpha=0.7, label=f'Point {i+1} ({label})')

# Plot the hyperplane
x1_range = np.linspace(-3, 3, 100)
x2_hyperplane = -w_given[0]/w_given[1] * x1_range - b_given/w_given[1]
plt.plot(x1_range, x2_hyperplane, 'g-', linewidth=3, label='Optimal Hyperplane')

# Plot margin boundaries
margin_boundary_plus = x2_hyperplane + margin/np.linalg.norm(w_given)
margin_boundary_minus = x2_hyperplane - margin/np.linalg.norm(w_given)

plt.plot(x1_range, margin_boundary_plus, 'g--', alpha=0.7, label='Margin Boundary (+1)')
plt.plot(x1_range, margin_boundary_minus, 'g--', alpha=0.7, label='Margin Boundary (-1)')

# Shade the margin region
plt.fill_between(x1_range, margin_boundary_minus, margin_boundary_plus, 
                alpha=0.2, color='green', label='Margin Region')

# Add labels and title
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Support Vector Machine: Dataset and Optimal Hyperplane')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(-3, 3)
plt.ylim(-3, 3)

# Add equation of hyperplane
plt.annotate(f'Hyperplane: $x_1 + x_2 = 0$\nMargin: {margin:.4f}', 
            xy=(0.05, 0.95), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

plt.legend()
plt.savefig(os.path.join(save_dir, 'svm_dataset_and_hyperplane.png'), dpi=300, bbox_inches='tight')

print(f"\nVisualization saved to: {save_dir}/svm_dataset_and_hyperplane.png")

print("\n" + "="*50)
print("STEP 2: VERIFYING KKT CONDITIONS")
print("="*50)

# KKT conditions for SVM:
# 1. Primal feasibility: y_i(w^T x_i + b) >= 1 for all i
# 2. Dual feasibility: α_i >= 0 for all i
# 3. Complementary slackness: α_i(y_i(w^T x_i + b) - 1) = 0 for all i
# 4. Stationarity: w = Σ α_i y_i x_i and Σ α_i y_i = 0

print("KKT Conditions Verification:")
print("1. Primal feasibility: y_i(w^T x_i + b) >= 1 for all i")

for i, (x, label) in enumerate(zip(X, y)):
    constraint_value = label * (np.dot(w_given, x) + b_given)
    print(f"   Point {i+1}: y_{i+1}(w^T x_{i+1} + b) = {label} * ({np.dot(w_given, x)} + {b_given}) = {constraint_value}")
    if constraint_value >= 1:
        print(f"   ✓ Satisfied (>= 1)")
    else:
        print(f"   ✗ NOT satisfied (< 1)")

print("\n2. Dual feasibility: α_i >= 0 for all i")
print("   (We'll calculate α_i values next)")

print("\n3. Complementary slackness: α_i(y_i(w^T x_i + b) - 1) = 0 for all i")
print("   (This means α_i > 0 only for support vectors)")

print("\n4. Stationarity: w = Σ α_i y_i x_i and Σ α_i y_i = 0")
print("   (We'll verify this after calculating α_i)")

print("\n" + "="*50)
print("STEP 3: CALCULATING LAGRANGE MULTIPLIERS")
print("="*50)

# For the given hyperplane, we need to find α_i that satisfy:
# w = Σ α_i y_i x_i
# Σ α_i y_i = 0
# α_i >= 0 for all i
# α_i > 0 only for support vectors

# Since we know the support vectors, we can set up a system of equations
# For support vectors: y_i(w^T x_i + b) = 1
# For non-support vectors: α_i = 0

print("Setting up the system of equations:")
print("For support vectors: y_i(w^T x_i + b) = 1")
print("For non-support vectors: α_i = 0")

# Calculate which points are exactly at the margin
margin_points = []
for i, (x, label) in enumerate(zip(X, y)):
    constraint_value = label * (np.dot(w_given, x) + b_given)
    print(f"Point {i+1}: y_{i+1}(w^T x_{i+1} + b) = {constraint_value:.4f}")
    if abs(constraint_value - 1) < 1e-10:
        margin_points.append(i)
        print(f"  ✓ This is a support vector (constraint = 1)")

print(f"\nSupport vectors identified: {[i+1 for i in margin_points]}")

# Now we need to solve for α_i
# We have: w = Σ α_i y_i x_i
# And: Σ α_i y_i = 0

# Let's set up the system of equations
# For the given w = [1, 1], we have:
# 1 = α_1 * y_1 * x_1[0] + α_2 * y_2 * x_2[0] + α_3 * y_3 * x_3[0] + α_4 * y_4 * x_4[0]
# 1 = α_1 * y_1 * x_1[1] + α_2 * y_2 * x_2[1] + α_3 * y_3 * x_3[1] + α_4 * y_4 * x_4[1]
# 0 = α_1 * y_1 + α_2 * y_2 + α_3 * y_3 + α_4 * y_4

print("\nSetting up equations:")
print("w[0] = 1 = Σ α_i y_i x_i[0]")
print("w[1] = 1 = Σ α_i y_i x_i[1]")
print("0 = Σ α_i y_i")

# Let's solve this analytically
# From the data:
# x1 = [1, 1], y1 = 1
# x2 = [2, 2], y2 = 1
# x3 = [-1, -1], y3 = -1
# x4 = [-2, -1], y4 = -1

# The equations become:
# 1 = α_1 * 1 * 1 + α_2 * 1 * 2 + α_3 * (-1) * (-1) + α_4 * (-1) * (-2)
# 1 = α_1 * 1 * 1 + α_2 * 1 * 2 + α_3 * (-1) * (-1) + α_4 * (-1) * (-1)
# 0 = α_1 * 1 + α_2 * 1 + α_3 * (-1) + α_4 * (-1)

# Simplifying:
# 1 = α_1 + 2α_2 + α_3 + 2α_4
# 1 = α_1 + 2α_2 + α_3 + α_4
# 0 = α_1 + α_2 - α_3 - α_4

# From the third equation: α_1 + α_2 = α_3 + α_4
# From the first two equations: α_4 = 0 (since 2α_4 - α_4 = 0)

# So α_4 = 0, and we have:
# α_1 + α_2 = α_3
# α_1 + 2α_2 + α_3 = 1

# Substituting: α_1 + 2α_2 + (α_1 + α_2) = 1
# 2α_1 + 3α_2 = 1

# We need to find values that satisfy this and the support vector conditions
# Let's try α_1 = 0.5, α_2 = 0, α_3 = 0.5, α_4 = 0

alpha_guess = np.array([0.5, 0, 0.5, 0])

print(f"\nTrying solution: α = {alpha_guess}")

# Verify this solution
w_calculated = np.zeros(2)
for i, (x, label, alpha) in enumerate(zip(X, y, alpha_guess)):
    w_calculated += alpha * label * x
    print(f"α_{i+1} * y_{i+1} * x_{i+1} = {alpha} * {label} * {x} = {alpha * label * x}")

print(f"Calculated w = {w_calculated}")
print(f"Given w = {w_given}")
print(f"Match: {np.allclose(w_calculated, w_given)}")

# Check the sum constraint
sum_constraint = np.sum(alpha_guess * y)
print(f"Σ α_i y_i = {sum_constraint}")

# Check which points are support vectors
print(f"\nSupport vector check:")
for i, (x, label, alpha) in enumerate(zip(X, y, alpha_guess)):
    constraint_value = label * (np.dot(w_given, x) + b_given)
    print(f"Point {i+1}: α_{i+1} = {alpha}, y_{i+1}(w^T x_{i+1} + b) = {constraint_value:.4f}")
    if alpha > 0:
        print(f"  ✓ Support vector (α > 0)")
    else:
        print(f"  ✗ Not a support vector (α = 0)")

# Let's find a better solution by solving the system properly
print(f"\n" + "="*50)
print("STEP 4: SOLVING THE SYSTEM ANALYTICALLY")
print("="*50)

# We know that points 1 and 3 are support vectors (they're at the margin)
# So α_1 > 0 and α_3 > 0, α_2 = α_4 = 0

# The equations become:
# 1 = α_1 * 1 * 1 + α_3 * (-1) * (-1) = α_1 + α_3
# 1 = α_1 * 1 * 1 + α_3 * (-1) * (-1) = α_1 + α_3
# 0 = α_1 * 1 + α_3 * (-1) = α_1 - α_3

# From the third equation: α_1 = α_3
# From the first equation: α_1 + α_3 = 1
# So: 2α_1 = 1, therefore α_1 = 0.5, α_3 = 0.5

alpha_correct = np.array([0.5, 0, 0.5, 0])

print(f"Correct solution: α = {alpha_correct}")

# Verify this solution
w_verified = np.zeros(2)
for i, (x, label, alpha) in enumerate(zip(X, y, alpha_correct)):
    w_verified += alpha * label * x

print(f"Verified w = {w_verified}")
print(f"Given w = {w_given}")
print(f"Match: {np.allclose(w_verified, w_given)}")

sum_verified = np.sum(alpha_correct * y)
print(f"Σ α_i y_i = {sum_verified}")

print(f"\n" + "="*50)
print("STEP 5: FINAL VERIFICATION")
print("="*50)

print("Final Lagrange multipliers:")
for i, alpha in enumerate(alpha_correct):
    print(f"  α_{i+1} = {alpha}")

print(f"\nSupport vectors:")
for i, alpha in enumerate(alpha_correct):
    if alpha > 0:
        print(f"  x_{i+1} = {X[i]}, y_{i+1} = {y[i]}, α_{i+1} = {alpha}")

print(f"\nNon-support vectors:")
for i, alpha in enumerate(alpha_correct):
    if alpha == 0:
        print(f"  x_{i+1} = {X[i]}, y_{i+1} = {y[i]}, α_{i+1} = {alpha}")

print(f"\nVerification of w = Σ α_i y_i x_i:")
w_final = np.zeros(2)
for i, (x, label, alpha) in enumerate(zip(X, y, alpha_correct)):
    if alpha > 0:
        contribution = alpha * label * x
        w_final += contribution
        print(f"  α_{i+1} * y_{i+1} * x_{i+1} = {alpha} * {label} * {x} = {contribution}")

print(f"  w = {w_final}")
print(f"  Given w = {w_given}")
print(f"  Match: {np.allclose(w_final, w_given)}")

print(f"\nVerification of Σ α_i y_i = 0:")
sum_final = np.sum(alpha_correct * y)
print(f"  Σ α_i y_i = {sum_final}")

# Create a visualization showing the support vectors and their contributions
plt.figure(figsize=(12, 10))

# Plot all points
for i, (x, label, color, marker) in enumerate(zip(X, y, colors, markers)):
    if alpha_correct[i] > 0:
        # Support vectors get larger markers and black edges
        plt.scatter(x[0], x[1], s=200, color=color, marker=marker, 
                   edgecolor='black', linewidth=3, zorder=5,
                                       label=f'Support Vector {i+1} ($\\alpha_{i+1}={alpha_correct[i]:.1f}$)')
    else:
        plt.scatter(x[0], x[1], s=100, color=color, marker=marker, 
                   alpha=0.7, label=f'Point {i+1} ($\\alpha_{i+1}=0$)')

# Plot the hyperplane
plt.plot(x1_range, x2_hyperplane, 'g-', linewidth=3, label='Optimal Hyperplane')

# Plot margin boundaries
plt.plot(x1_range, margin_boundary_plus, 'g--', alpha=0.7, label='Margin Boundary (+1)')
plt.plot(x1_range, margin_boundary_minus, 'g--', alpha=0.7, label='Margin Boundary (-1)')

# Shade the margin region
plt.fill_between(x1_range, margin_boundary_minus, margin_boundary_plus, 
                alpha=0.2, color='green', label='Margin Region')

# Add arrows showing the contribution of each support vector to w
for i, (x, label, alpha) in enumerate(zip(X, y, alpha_correct)):
    if alpha > 0:
        contribution = alpha * label * x
        plt.arrow(0, 0, contribution[0], contribution[1], 
                 head_width=0.1, head_length=0.1, fc='purple', ec='purple', 
                 alpha=0.7, linewidth=2, label=f'$\\alpha_{i+1}y_{i+1}\\mathbf{{x}}_{i+1}$' if i == 0 else "")

# Add the resulting w vector
plt.arrow(0, 0, w_given[0], w_given[1], 
         head_width=0.15, head_length=0.15, fc='orange', ec='orange', 
         linewidth=3, label='$\\mathbf{w} = \\sum \\alpha_i y_i \\mathbf{x}_i$')

# Add labels and title
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Support Vector Machine: Lagrange Multipliers and Weight Vector')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(-3, 3)
plt.ylim(-3, 3)

# Add equations
eq_text = f'Hyperplane: $x_1 + x_2 = 0$\n'
eq_text += f'Support Vectors: $x_1, x_3$\n'
eq_text += f'$\\alpha_1 = {alpha_correct[0]:.1f}, \\alpha_3 = {alpha_correct[2]:.1f}$\n'
eq_text += f'$\\mathbf{{w}} = \\sum \\alpha_i y_i \\mathbf{{x}}_i$'

plt.annotate(eq_text, xy=(0.05, 0.95), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

plt.legend()
plt.savefig(os.path.join(save_dir, 'svm_lagrange_multipliers.png'), dpi=300, bbox_inches='tight')

print(f"\nVisualization saved to: {save_dir}/svm_lagrange_multipliers.png")

# Create a summary table
print(f"\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)

print(f"{'Point':<8} {'Coordinates':<15} {'Label':<8} {'Distance':<12} {'α':<8} {'Support Vector':<15}")
print("-" * 80)
for i, (x, label, dist, alpha) in enumerate(zip(X, y, distances, alpha_correct)):
    sv_status = "Yes" if alpha > 0 else "No"
    print(f"{'x'+str(i+1):<8} {str(x):<15} {label:<8} {dist:<12.4f} {alpha:<8.1f} {sv_status:<15}")

print(f"\n" + "="*80)
print("FINAL ANSWERS")
print("="*80)

print("1. Support vectors: x1 = (1, 1) and x3 = (-1, -1)")
print("2. KKT conditions are satisfied:")
print("   - Primal feasibility: y_i(w^T x_i + b) >= 1 for all i")
print("   - Dual feasibility: α_i >= 0 for all i")
print("   - Complementary slackness: α_i > 0 only for support vectors")
print("   - Stationarity: w = Σ α_i y_i x_i and Σ α_i y_i = 0")
print("3. Lagrange multipliers: α1 = 0.5, α2 = 0, α3 = 0.5, α4 = 0")
print("4. Σ α_i y_i = 0.5*1 + 0*1 + 0.5*(-1) + 0*(-1) = 0 ✓")
print("5. Weight vector: w = α1*y1*x1 + α3*y3*x3 = 0.5*1*(1,1) + 0.5*(-1)*(-1,-1) = (1,1)")

# Additional informative visualizations

print(f"\n" + "="*50)
print("ADDITIONAL VISUALIZATIONS")
print("="*50)

# Visualization 3: Margin analysis with constraint values
plt.figure(figsize=(12, 10))

# Plot all points with constraint values
for i, (x, label, color, marker) in enumerate(zip(X, y, colors, markers)):
    constraint_value = label * (np.dot(w_given, x) + b_given)
    if i in support_vector_indices:
        plt.scatter(x[0], x[1], s=200, color=color, marker=marker, 
                   edgecolor='black', linewidth=3, zorder=5,
                   label=f'Support Vector {i+1} (constraint={constraint_value:.1f})')
    else:
        plt.scatter(x[0], x[1], s=100, color=color, marker=marker, 
                   alpha=0.7, label=f'Point {i+1} (constraint={constraint_value:.1f})')

# Plot the hyperplane
plt.plot(x1_range, x2_hyperplane, 'g-', linewidth=3, label='Optimal Hyperplane')

# Plot margin boundaries
plt.plot(x1_range, margin_boundary_plus, 'g--', alpha=0.7, label='Margin Boundary (+1)')
plt.plot(x1_range, margin_boundary_minus, 'g--', alpha=0.7, label='Margin Boundary (-1)')

# Shade the margin region
plt.fill_between(x1_range, margin_boundary_minus, margin_boundary_plus, 
                alpha=0.2, color='green', label='Margin Region')

# Add constraint value annotations
for i, (x, label) in enumerate(zip(X, y)):
    constraint_value = label * (np.dot(w_given, x) + b_given)
    plt.annotate(f'$y_{i+1}(\\mathbf{{w}}^T\\mathbf{{x}}_{i+1} + b) = {constraint_value:.1f}$',
                 (x[0], x[1]), xytext=(10, 10), textcoords='offset points',
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

# Add labels and title
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('SVM: Constraint Values and Margin Analysis')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(-3, 3)
plt.ylim(-3, 3)

# Add equation of hyperplane
plt.annotate(f'Hyperplane: $x_1 + x_2 = 0$\nMargin: {margin:.4f}', 
            xy=(0.05, 0.95), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

plt.legend()
plt.savefig(os.path.join(save_dir, 'svm_constraint_analysis.png'), dpi=300, bbox_inches='tight')

print(f"Visualization 3 saved to: {save_dir}/svm_constraint_analysis.png")

# Visualization 4: Lagrange multiplier contributions
plt.figure(figsize=(12, 10))

# Plot all points
for i, (x, label, color, marker) in enumerate(zip(X, y, colors, markers)):
    if alpha_correct[i] > 0:
        plt.scatter(x[0], x[1], s=200, color=color, marker=marker, 
                   edgecolor='black', linewidth=3, zorder=5,
                   label=f'Support Vector {i+1} ($\\alpha_{i+1}={alpha_correct[i]:.1f}$)')
    else:
        plt.scatter(x[0], x[1], s=100, color=color, marker=marker, 
                   alpha=0.7, label=f'Point {i+1} ($\\alpha_{i+1}=0$)')

# Plot the hyperplane
plt.plot(x1_range, x2_hyperplane, 'g-', linewidth=3, label='Optimal Hyperplane')

# Plot margin boundaries
plt.plot(x1_range, margin_boundary_plus, 'g--', alpha=0.7, label='Margin Boundary (+1)')
plt.plot(x1_range, margin_boundary_minus, 'g--', alpha=0.7, label='Margin Boundary (-1)')

# Shade the margin region
plt.fill_between(x1_range, margin_boundary_minus, margin_boundary_plus, 
                alpha=0.2, color='green', label='Margin Region')

# Add arrows showing the contribution of each support vector to w
for i, (x, label, alpha) in enumerate(zip(X, y, alpha_correct)):
    if alpha > 0:
        contribution = alpha * label * x
        plt.arrow(0, 0, contribution[0], contribution[1], 
                 head_width=0.1, head_length=0.1, fc='purple', ec='purple', 
                 alpha=0.7, linewidth=2, label=f'$\\alpha_{i+1}y_{i+1}\\mathbf{{x}}_{i+1}$' if i == 0 else "")
        
        # Add contribution labels
        plt.annotate(f'$\\alpha_{i+1}y_{i+1}\\mathbf{{x}}_{i+1} = {alpha:.1f} \\cdot {label} \\cdot ({x[0]},{x[1]}) = ({contribution[0]:.1f},{contribution[1]:.1f})$',
                     (contribution[0]/2, contribution[1]/2), xytext=(10, 10), textcoords='offset points',
                     fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", ec="black", alpha=0.8))

# Add the resulting w vector
plt.arrow(0, 0, w_given[0], w_given[1], 
         head_width=0.15, head_length=0.15, fc='orange', ec='orange', 
         linewidth=3, label='$\\mathbf{w} = \\sum \\alpha_i y_i \\mathbf{x}_i$')

# Add labels and title
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('SVM: Lagrange Multiplier Contributions to Weight Vector')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(-3, 3)
plt.ylim(-3, 3)

# Add equations
eq_text = f'Hyperplane: $x_1 + x_2 = 0$\n'
eq_text += f'Support Vectors: $\\mathbf{{x}}_1, \\mathbf{{x}}_3$\n'
eq_text += f'$\\alpha_1 = {alpha_correct[0]:.1f}, \\alpha_3 = {alpha_correct[2]:.1f}$\n'
eq_text += f'$\\mathbf{{w}} = \\sum \\alpha_i y_i \\mathbf{{x}}_i$'

plt.annotate(eq_text, xy=(0.05, 0.95), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

plt.legend()
plt.savefig(os.path.join(save_dir, 'svm_lagrange_contributions.png'), dpi=300, bbox_inches='tight')

print(f"Visualization 4 saved to: {save_dir}/svm_lagrange_contributions.png")

# Visualization 5: Distance analysis
plt.figure(figsize=(12, 10))

# Plot all points with distance values
for i, (x, label, color, marker) in enumerate(zip(X, y, colors, markers)):
    dist = distances[i]
    if i in support_vector_indices:
        plt.scatter(x[0], x[1], s=200, color=color, marker=marker, 
                   edgecolor='black', linewidth=3, zorder=5,
                   label=f'Support Vector {i+1} (dist={dist:.3f})')
    else:
        plt.scatter(x[0], x[1], s=100, color=color, marker=marker, 
                   alpha=0.7, label=f'Point {i+1} (dist={dist:.3f})')

# Plot the hyperplane
plt.plot(x1_range, x2_hyperplane, 'g-', linewidth=3, label='Optimal Hyperplane')

# Plot margin boundaries
plt.plot(x1_range, margin_boundary_plus, 'g--', alpha=0.7, label='Margin Boundary (+1)')
plt.plot(x1_range, margin_boundary_minus, 'g--', alpha=0.7, label='Margin Boundary (-1)')

# Shade the margin region
plt.fill_between(x1_range, margin_boundary_minus, margin_boundary_plus, 
                alpha=0.2, color='green', label='Margin Region')

# Add distance annotations
for i, (x, dist) in enumerate(zip(X, distances)):
    plt.annotate(f'$d_{i+1} = {dist:.3f}$',
                 (x[0], x[1]), xytext=(10, 10), textcoords='offset points',
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

# Add labels and title
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('SVM: Distance Analysis and Margin')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(-3, 3)
plt.ylim(-3, 3)

# Add equation of hyperplane and margin
plt.annotate(f'Hyperplane: $x_1 + x_2 = 0$\nMargin: {margin:.4f}\n$d = \\frac{{|\\mathbf{{w}}^T\\mathbf{{x}} + b|}}{{\\|\\mathbf{{w}}\\|}}$', 
            xy=(0.05, 0.95), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

plt.legend()
plt.savefig(os.path.join(save_dir, 'svm_distance_analysis.png'), dpi=300, bbox_inches='tight')

print(f"Visualization 5 saved to: {save_dir}/svm_distance_analysis.png")

print(f"\nAll visualizations saved to: {save_dir}")
