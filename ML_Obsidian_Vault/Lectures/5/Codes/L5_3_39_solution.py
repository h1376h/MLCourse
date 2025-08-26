import numpy as np
import matplotlib.pyplot as plt
import os
from fractions import Fraction

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_3_Quiz_39")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amssymb}'

print("=" * 80)
print("Question 39: Feature Transformation and Linear Separation")
print("DETAILED MATHEMATICAL SOLUTION (Pen-and-Paper Style)")
print("=" * 80)

# Dataset with 7 points
data = np.array([
    [-3, 1],   # (x, y) = (x, label)
    [-2, 1],
    [-1, -1],
    [0, -1],
    [1, -1],
    [2, 1],
    [3, 1]
])

X = data[:, 0].reshape(-1, 1)  # Input features (x values)
y = data[:, 1]                 # Labels

print("Given Dataset:")
print("D = {(x_i, y_i)} = {(-3,1), (-2,1), (-1,-1), (0,-1), (1,-1), (2,1), (3,1)}")
print("\nWhere:")
print("- x_i in R is the input feature")
print("- y_i in {-1, +1} is the class label")

print("\n" + "="*80)
print("STEP 1: ANALYZE LINEAR SEPARABILITY IN ORIGINAL SPACE")
print("="*80)

print("In the original 1D space, we need to find if there exists a threshold theta such that:")
print("- All points with x < theta belong to one class")
print("- All points with x > theta belong to the other class")
print("\nLet's examine the data points in order:")

sorted_data = sorted(data, key=lambda x: x[0])
print("Sorted by x-coordinate:")
for x_val, label in sorted_data:
    print(f"x = {x_val:2d}, y = {label:2d}")

print("\nAnalysis:")
print("x = -3: y = +1")
print("x = -2: y = +1")
print("x = -1: y = -1  <- Class changes")
print("x =  0: y = -1")
print("x =  1: y = -1")
print("x =  2: y = +1  <- Class changes again")
print("x =  3: y = +1")

print("\nConclusion: The classes alternate, so NO single threshold can separate them.")
print("The data is NOT linearly separable in 1D.")

print("\n" + "="*80)
print("STEP 2: APPLY FEATURE TRANSFORMATION phi(x) = (x, x^2)")
print("="*80)

print("We apply the transformation phi: R -> R^2 defined by:")
print("phi(x) = (x, x^2)")
print("\nThis maps each 1D point to a 2D point in the feature space.")

# Apply feature transformation phi(x) = (x, x^2)
X_transformed = np.column_stack([X.flatten(), X.flatten()**2])

print("\nTransformation results:")
print("Original x | phi(x) = (x, x^2) | Label y")
print("-" * 45)
for i, (x_val, label) in enumerate(data):
    x_trans = X_transformed[i]
    print(f"{x_val:8.0f}   | ({x_trans[0]:4.0f}, {x_trans[1]:4.0f})     | {label:5.0f}")

print("\n" + "="*80)
print("STEP 3: ANALYZE SEPARABILITY IN TRANSFORMED SPACE")
print("="*80)

print("Let's group the transformed points by class:")
print("\nClass +1 points (positive class):")
pos_points = []
neg_points = []
for i, (x_val, label) in enumerate(data):
    x_trans = X_transformed[i]
    if label == 1:
        pos_points.append((x_trans[0], x_trans[1]))
        print(f"  phi({x_val}) = ({x_trans[0]:4.0f}, {x_trans[1]:4.0f})")
    else:
        neg_points.append((x_trans[0], x_trans[1]))

print("\nClass -1 points (negative class):")
for i, (x_val, label) in enumerate(data):
    x_trans = X_transformed[i]
    if label == -1:
        print(f"  phi({x_val}) = ({x_trans[0]:4.0f}, {x_trans[1]:4.0f})")

print("\nKey Observation:")
print("Looking at the x^2 coordinate (second dimension):")
pos_x2_values = [p[1] for p in pos_points]
neg_x2_values = [p[1] for p in neg_points]
print(f"- Positive class: x^2 in {{{', '.join(map(str, map(int, pos_x2_values)))}}}")
print(f"- Negative class: x^2 in {{{', '.join(map(str, map(int, neg_x2_values)))}}}")

min_pos_x2 = min(pos_x2_values)
max_neg_x2 = max(neg_x2_values)
print(f"\n- Minimum x^2 for positive class: {min_pos_x2}")
print(f"- Maximum x^2 for negative class: {max_neg_x2}")
print(f"- Gap exists: {max_neg_x2} < x^2 < {min_pos_x2}")

print(f"\nTherefore, a horizontal line x^2 = c where {max_neg_x2} < c < {min_pos_x2}")
print("can perfectly separate the two classes!")

# Choose separation threshold
separation_threshold = (max_neg_x2 + min_pos_x2) / 2
print(f"\nChoose c = ({max_neg_x2} + {min_pos_x2})/2 = {separation_threshold}")

print("\n" + "="*80)
print("STEP 4: DERIVE THE LINEAR DECISION BOUNDARY MATHEMATICALLY")
print("="*80)

print("In the transformed 2D space, we seek a linear decision boundary of the form:")
print("w1*x + w2*x^2 + b = 0")
print("\nWhere the decision rule is:")
print("f(x) = sign(w1*x + w2*x^2 + b)")
print("  = +1 if w1*x + w2*x^2 + b > 0")
print("  = -1 if w1*x + w2*x^2 + b < 0")

print("\nFrom our analysis, we know that x^2 = 2.5 separates the classes.")
print("This suggests the decision boundary should be approximately:")
print("x^2 - 2.5 = 0")
print("or equivalently: 0*x + 1*x^2 - 2.5 = 0")

print("\nSo we can choose:")
print("w1 = 0")
print("w2 = 1")
print("b = -2.5")

w1_exact = 0
w2_exact = 1
b_exact = -2.5

print(f"\nDecision boundary: {w1_exact}*x + {w2_exact}*x^2 + ({b_exact}) = 0")
print(f"Simplified: x^2 = {-b_exact/w2_exact}")

print("\n" + "="*80)
print("STEP 5: VERIFY THE DECISION BOUNDARY")
print("="*80)

print("Let's verify that our decision boundary correctly classifies all points:")
print("Decision function: f(x) = sign(0*x + 1*x^2 - 2.5) = sign(x^2 - 2.5)")
print("\nVerification:")
print("Point | x  | x^2 | x^2 - 2.5 | sign(x^2 - 2.5) | True Label | Correct?")
print("-" * 75)

all_correct = True
for i, (x_val, true_label) in enumerate(data):
    x_squared = x_val**2
    decision_value = x_squared - 2.5
    prediction = 1 if decision_value > 0 else -1
    correct = prediction == true_label
    all_correct = all_correct and correct

    print(f"{i+1:5d} | {x_val:2.0f} | {x_squared:2.0f} | {decision_value:8.1f} | {prediction:15d} | {true_label:10d} | {correct}")

print(f"\nResult: All points correctly classified: {all_correct}")

print("\n" + "="*80)
print("STEP 6: FIND THE OPTIMAL SEPARATING HYPERPLANE (SVM APPROACH)")
print("="*80)

print("For a more rigorous approach, we can find the optimal separating hyperplane")
print("that maximizes the margin between the classes.")
print("\nThe SVM optimization problem is:")
print("minimize: (1/2)||w||^2")
print("subject to: y_i(w1*x_i + w2*x_i^2 + b) >= 1 for all i")

print("\nLet's identify the support vectors (points closest to the decision boundary):")

# Find points closest to the boundary x^2 = 2.5
distances_to_boundary = []
for i, (x_val, label) in enumerate(data):
    x_squared = x_val**2
    distance = abs(x_squared - 2.5)
    distances_to_boundary.append((i, x_val, x_squared, label, distance))

# Sort by distance to boundary
distances_to_boundary.sort(key=lambda x: x[4])

print("\nDistances from decision boundary x^2 = 2.5:")
print("Point | x  | x^2 | Label | |x^2 - 2.5|")
print("-" * 40)
for i, x_val, x_squared, label, distance in distances_to_boundary:
    print(f"{i+1:5d} | {x_val:2.0f} | {x_squared:2.0f} | {label:5.0f} | {distance:9.1f}")

# The closest points from each class will be support vectors
closest_pos = min([d for d in distances_to_boundary if d[3] == 1], key=lambda x: x[4])
closest_neg = min([d for d in distances_to_boundary if d[3] == -1], key=lambda x: x[4])

print(f"\nClosest positive point: x = {closest_pos[1]}, x^2 = {closest_pos[2]}")
print(f"Closest negative point: x = {closest_neg[1]}, x^2 = {closest_neg[2]}")

# The optimal boundary should be equidistant from the closest points of each class
optimal_boundary = (closest_pos[2] + closest_neg[2]) / 2
margin_half_width = (closest_pos[2] - closest_neg[2]) / 2

print(f"\nOptimal decision boundary: x^2 = {optimal_boundary}")
print(f"Margin (half-width): {margin_half_width}")
print(f"Full margin width: {2 * margin_half_width}")

# Update our weights for the optimal boundary
w1_optimal = 0
w2_optimal = 1
b_optimal = -optimal_boundary

print(f"\nOptimal hyperplane parameters:")
print(f"w1 = {w1_optimal}")
print(f"w2 = {w2_optimal}")
print(f"b = {b_optimal}")

print(f"\nOptimal decision boundary: {w1_optimal}*x + {w2_optimal}*x^2 + ({b_optimal}) = 0")

print("\n" + "="*80)
print("STEP 7: CALCULATE THE GEOMETRIC MARGIN")
print("="*80)

print("The geometric margin is the distance from the decision boundary to the")
print("closest points (support vectors).")
print("\nFor a hyperplane w1*x + w2*x^2 + b = 0, the distance from a point (x0, x0^2)")
print("to the hyperplane is:")
print("distance = |w1*x0 + w2*x0^2 + b| / sqrt(w1^2 + w2^2)")

# For the standard SVM formulation, we need to normalize the hyperplane
# so that the support vectors satisfy: y_i(w1*x_i + w2*x_i^2 + b) = 1

# The distance from support vectors to the boundary is 1.5
# For standard SVM, this should be 1/||w||, so ||w|| = 1/1.5 = 2/3
w1_normalized = 0
w2_normalized = 2/3
b_normalized = -2.5 * (2/3)

w_norm = np.sqrt(w1_normalized**2 + w2_normalized**2)
print(f"\nFor standard SVM formulation:")
print(f"Normalized weights: w1 = {w1_normalized}, w2 = {w2_normalized:.3f}, b = {b_normalized:.3f}")
print(f"||w|| = sqrt(w1^2 + w2^2) = sqrt({w1_normalized}^2 + {w2_normalized:.3f}^2) = {w_norm:.3f}")

geometric_margin = 1/w_norm
print(f"\nGeometric margin = 1/||w|| = 1/{w_norm:.3f} = {geometric_margin:.3f}")

print("\nSupport vectors (points on the margin boundary):")
support_vectors = []
for i, (x_val, label) in enumerate(data):
    x_squared = x_val**2
    # Check if point is a support vector (closest to boundary)
    distance_to_boundary = abs(x_squared - 2.5)
    if abs(distance_to_boundary - 1.5) < 1e-10:  # Points at distance 1.5 from boundary
        support_vectors.append((i+1, x_val, x_squared, label))
        print(f"  Point {i+1}: x = {x_val}, phi(x) = ({x_val}, {x_squared}), label = {label}")

print(f"\nVerification: Support vectors are at distance {geometric_margin:.3f} from the decision boundary")

print("\n" + "="*80)
print("STEP 8: EXPRESS THE FINAL DECISION FUNCTION")
print("="*80)

print("The final decision function in the original space is:")
print(f"f(x) = sign({w1_optimal}*x + {w2_optimal}*x^2 + ({b_optimal}))")
print(f"f(x) = sign(x^2 - {-b_optimal})")

print(f"\nThis means:")
print(f"- If x^2 > {-b_optimal}, predict class +1")
print(f"- If x^2 < {-b_optimal}, predict class -1")
print(f"- Equivalently: If |x| > sqrt({-b_optimal}) = {np.sqrt(-b_optimal):.3f}, predict +1")
print(f"                If |x| < {np.sqrt(-b_optimal):.3f}, predict -1")

print("\n" + "="*80)
print("STEP 9: GEOMETRIC INTERPRETATION")
print("="*80)

print("In the original 1D space, our decision boundary corresponds to:")
print(f"|x| = {np.sqrt(-b_optimal):.3f}")
print("\nThis creates two regions:")
print(f"- Inner region: |x| < {np.sqrt(-b_optimal):.3f} -> Class -1")
print(f"- Outer regions: |x| > {np.sqrt(-b_optimal):.3f} -> Class +1")

print("\nThis is a 'band' or 'ring' classifier that selects points based on their")
print("distance from the origin.")

# Plot original data in 1D
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
colors = ['red' if label == -1 else 'blue' for label in y]
markers = ['s' if label == -1 else 'o' for label in y]

for i, (x_val, label) in enumerate(data):
    color = 'red' if label == -1 else 'blue'
    marker = 's' if label == -1 else 'o'
    plt.scatter(x_val, 0, c=color, marker=marker, s=100, edgecolor='black', linewidth=1)
    plt.annotate(f'({x_val})', (x_val, 0), xytext=(0, 15),
                textcoords='offset points', ha='center', fontsize=10)

# Add decision boundaries
threshold = np.sqrt(-b_optimal)
plt.axvline(x=threshold, color='green', linestyle='--', alpha=0.7, label=f'x = {threshold:.2f}')
plt.axvline(x=-threshold, color='green', linestyle='--', alpha=0.7, label=f'x = {-threshold:.2f}')

plt.axhline(y=0, color='black', linewidth=0.5)
plt.xlabel(r'$x$')
plt.ylabel('')
plt.title(r'Original 1D Data with Decision Boundaries')
plt.grid(True, alpha=0.3)
plt.ylim(-0.5, 0.5)
plt.xlim(-4, 4)

# Create legend
plt.scatter([], [], c='blue', marker='o', s=100, edgecolor='black', label='Class +1')
plt.scatter([], [], c='red', marker='s', s=100, edgecolor='black', label='Class -1')
plt.legend()

# Plot transformed data in 2D
plt.subplot(2, 2, 2)
for i, (x_val, label) in enumerate(data):
    x_trans = X_transformed[i]
    color = 'red' if label == -1 else 'blue'
    marker = 's' if label == -1 else 'o'
    size = 150 if any(sv[0] == i+1 for sv in support_vectors) else 100
    edge_width = 3 if any(sv[0] == i+1 for sv in support_vectors) else 1

    plt.scatter(x_trans[0], x_trans[1], c=color, marker=marker, s=size,
               edgecolor='black', linewidth=edge_width)
    plt.annotate(f'({x_trans[0]:.0f},{x_trans[1]:.0f})',
                (x_trans[0], x_trans[1]), xytext=(5, 5),
                textcoords='offset points', fontsize=9)

# Plot decision boundary
x_range = np.linspace(-4, 4, 100)
boundary_line = np.full_like(x_range, -b_optimal)
plt.plot(x_range, boundary_line, 'g-', linewidth=2, label=rf'Decision Boundary: $x^2 = {-b_optimal}$')

# Plot margin boundaries
margin_distance = 1/w_norm
plt.plot(x_range, boundary_line + margin_distance, 'g--', alpha=0.7, label='Margin Boundaries')
plt.plot(x_range, boundary_line - margin_distance, 'g--', alpha=0.7)

plt.xlabel(r'$x$')
plt.ylabel(r'$x^2$')
plt.title(r'Transformed 2D Data with Decision Boundary')
plt.grid(True, alpha=0.3)
plt.xlim(-4, 4)
plt.ylim(-1, 10)

# Create legend
plt.scatter([], [], c='blue', marker='o', s=100, edgecolor='black', label='Class +1')
plt.scatter([], [], c='red', marker='s', s=100, edgecolor='black', label='Class -1')
plt.scatter([], [], c='gray', marker='o', s=150, edgecolor='black', linewidth=3, label='Support Vectors')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'detailed_mathematical_solution.png'), dpi=300, bbox_inches='tight')

# Create comprehensive visualization
plt.subplot(2, 2, 3)
# Plot decision regions in original space
x_plot = np.linspace(-4, 4, 1000)
threshold = np.sqrt(-b_optimal)

# Color regions
x_neg_region = x_plot[np.abs(x_plot) < threshold]
x_pos_region1 = x_plot[x_plot <= -threshold]
x_pos_region2 = x_plot[x_plot >= threshold]

plt.fill_between(x_neg_region, -0.3, 0.3, color='lightcoral', alpha=0.3, label='Class -1 Region')
plt.fill_between(x_pos_region1, -0.3, 0.3, color='lightblue', alpha=0.3, label='Class +1 Region')
plt.fill_between(x_pos_region2, -0.3, 0.3, color='lightblue', alpha=0.3)

# Plot data points
for i, (x_val, label) in enumerate(data):
    color = 'red' if label == -1 else 'blue'
    marker = 's' if label == -1 else 'o'
    plt.scatter(x_val, 0, c=color, marker=marker, s=100, edgecolor='black', linewidth=1, zorder=5)
    plt.annotate(f'{x_val}', (x_val, 0), xytext=(0, 20),
                textcoords='offset points', ha='center', fontsize=10)

# Plot decision boundaries
plt.axvline(x=threshold, color='green', linestyle='-', linewidth=2, label=f'Decision Boundaries')
plt.axvline(x=-threshold, color='green', linestyle='-', linewidth=2)

plt.axhline(y=0, color='black', linewidth=0.5)
plt.xlabel(r'$x$')
plt.ylabel('')
plt.title(r'Decision Regions in Original Space')
plt.grid(True, alpha=0.3)
plt.ylim(-0.4, 0.4)
plt.xlim(-4, 4)
plt.legend(loc='upper right')

# Summary plot
plt.subplot(2, 2, 4)
plt.text(0.1, 0.9, 'SOLUTION SUMMARY', fontsize=14, fontweight='bold', transform=plt.gca().transAxes)

summary_text = f"""Mathematical Solution:

1. Feature Transformation:
   phi(x) = (x, x²)

2. Decision Boundary:
   x² = {-b_optimal}

3. Decision Function:
   f(x) = sign(x² - {-b_optimal})

4. Geometric Interpretation:
   |x| < {threshold:.3f} -> Class -1
   |x| > {threshold:.3f} -> Class +1

5. Margin: {geometric_margin:.3f}

6. Support Vectors:
   x in {{{', '.join([str(int(sv[1])) for sv in support_vectors])}}}

7. Classification Accuracy: 100%"""

plt.text(0.1, 0.8, summary_text, fontsize=10, transform=plt.gca().transAxes,
         verticalalignment='top', fontfamily='monospace')
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'complete_mathematical_solution.png'), dpi=300, bbox_inches='tight')

print("\n" + "="*80)
print("FINAL VERIFICATION")
print("="*80)

print("Final verification of our mathematical solution:")
print(f"Decision function: f(x) = sign(x^2 - {-b_optimal})")
print("\nTesting on all data points:")
print("Point | x  | x^2 | x^2 - 2.5 | Prediction | True Label | Correct?")
print("-" * 70)

final_all_correct = True
for i, (x_val, true_label) in enumerate(data):
    x_squared = x_val**2
    decision_value = x_squared - (-b_optimal)
    prediction = 1 if decision_value > 0 else -1
    correct = prediction == true_label
    final_all_correct = final_all_correct and correct

    print(f"{i+1:5d} | {x_val:2.0f} | {x_squared:2.0f} | {decision_value:8.1f} | {prediction:10d} | {true_label:10d} | {correct}")

print(f"\nFINAL RESULT: All points correctly classified: {final_all_correct}")

print(f"\nAll plots saved to: {save_dir}")
print("="*80)
print("MATHEMATICAL SOLUTION COMPLETE")
print("="*80)
