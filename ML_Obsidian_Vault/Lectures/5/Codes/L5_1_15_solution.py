import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_1_Quiz_15")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("Question 15: Complete Maximum Margin Classification Example")
print("=" * 80)

# Task 1: Create a 2D dataset with 6 points (3 per class) requiring exactly 3 support vectors
print("Task 1: Designing Dataset with Exactly 3 Support Vectors")
print("="*60)

# Design strategy: Create a dataset where exactly 3 points lie on the margin boundaries
# We'll place points such that the optimal hyperplane has 3 support vectors

# Positive class points
X_pos = np.array([
    [3, 3],    # Support vector on positive margin
    [4, 4],    # Support vector on positive margin  
    [5, 2]     # Non-support vector (inside margin)
])

# Negative class points
X_neg = np.array([
    [1, 1],    # Support vector on negative margin
    [0, 2],    # Non-support vector (inside margin)
    [2, 0]     # Non-support vector (inside margin)
])

X = np.vstack([X_pos, X_neg])
y = np.array([1, 1, 1, -1, -1, -1])

print("Designed dataset:")
print("Positive class points:", X_pos)
print("Negative class points:", X_neg)
print("Labels:", y)

# Verify using sklearn SVM
svm = SVC(kernel='linear', C=1e6)  # Large C for hard margin
svm.fit(X, y)

support_vector_indices = svm.support_
n_support_vectors = len(support_vector_indices)

print(f"\nVerification with sklearn:")
print(f"Number of support vectors: {n_support_vectors}")
print(f"Support vector indices: {support_vector_indices}")
print(f"Support vectors:")
for i, idx in enumerate(support_vector_indices):
    print(f"  SV {i+1}: Point {idx} = {X[idx]}, label = {y[idx]}")

# Task 2: Solve optimization problem analytically
print("\n" + "="*60)
print("Task 2: Analytical Solution")
print("="*60)

# For a 2D problem with 3 support vectors, we can solve analytically
# The support vectors should be points 0, 1, and 3 based on our design

# Let's assume the hyperplane is w1*x1 + w2*x2 + b = 0
# For support vectors, we have: y_i(w1*x1_i + w2*x2_i + b) = 1

# From sklearn solution
w_sklearn = svm.coef_[0]
b_sklearn = svm.intercept_[0]

print(f"sklearn solution:")
print(f"w = [{w_sklearn[0]:.6f}, {w_sklearn[1]:.6f}]")
print(f"b = {b_sklearn:.6f}")

# Analytical approach: Set up system of equations for support vectors
# Assuming points 0, 1, 3 are support vectors (we'll verify this)
sv_points = X[support_vector_indices]
sv_labels = y[support_vector_indices]

print(f"\nSetting up analytical solution:")
print(f"Support vector constraints:")
for i, (point, label) in enumerate(zip(sv_points, sv_labels)):
    print(f"  {label}*({w_sklearn[0]:.3f}*{point[0]} + {w_sklearn[1]:.3f}*{point[1]} + {b_sklearn:.3f}) = 1")

# Verify margins for all points
margins = y * (X.dot(w_sklearn) + b_sklearn)
print(f"\nMargin verification for all points:")
for i, (point, label, margin) in enumerate(zip(X, y, margins)):
    is_sv = i in support_vector_indices
    status = "SV" if is_sv else "Non-SV"
    print(f"  Point {i}: {point}, y={label:2d}, margin={margin:.6f}, {status}")

# Task 3: Verify KKT conditions
print("\n" + "="*60)
print("Task 3: KKT Conditions Verification")
print("="*60)

# KKT conditions for SVM:
# 1. Stationarity: ∇_w L = w - Σ α_i y_i x_i = 0
# 2. Primal feasibility: y_i(w^T x_i + b) >= 1
# 3. Dual feasibility: α_i >= 0
# 4. Complementary slackness: α_i[y_i(w^T x_i + b) - 1] = 0

# Get dual coefficients from sklearn
dual_coef = svm.dual_coef_[0]  # α_i * y_i for support vectors
alphas = np.zeros(len(X))

# Extract α_i values
for i, (idx, coef) in enumerate(zip(support_vector_indices, dual_coef)):
    alphas[idx] = coef / y[idx]  # α_i = (α_i * y_i) / y_i

print("KKT Conditions Verification:")
print("\n1. Stationarity condition: w = Σ α_i y_i x_i")
w_reconstructed = np.zeros(2)
for i in range(len(X)):
    w_reconstructed += alphas[i] * y[i] * X[i]

print(f"   Original w: [{w_sklearn[0]:.6f}, {w_sklearn[1]:.6f}]")
print(f"   Reconstructed w: [{w_reconstructed[0]:.6f}, {w_reconstructed[1]:.6f}]")
print(f"   Difference: {np.linalg.norm(w_sklearn - w_reconstructed):.8f}")
print(f"   Stationarity satisfied: {np.linalg.norm(w_sklearn - w_reconstructed) < 1e-6}")

print("\n2. Primal feasibility: y_i(w^T x_i + b) >= 1")
all_feasible = True
for i, margin in enumerate(margins):
    feasible = margin >= 0.999  # Allow small numerical error
    print(f"   Point {i}: margin = {margin:.6f}, feasible = {feasible}")
    if not feasible:
        all_feasible = False
print(f"   All constraints satisfied: {all_feasible}")

print("\n3. Dual feasibility: α_i >= 0")
all_dual_feasible = True
for i, alpha in enumerate(alphas):
    feasible = alpha >= -1e-6  # Allow small numerical error
    print(f"   Point {i}: α_i = {alpha:.6f}, feasible = {feasible}")
    if not feasible:
        all_dual_feasible = False
print(f"   All dual constraints satisfied: {all_dual_feasible}")

print("\n4. Complementary slackness: α_i[y_i(w^T x_i + b) - 1] = 0")
all_complementary = True
for i, (alpha, margin) in enumerate(zip(alphas, margins)):
    slack = margin - 1.0
    complementary_product = alpha * slack
    satisfied = abs(complementary_product) < 1e-6
    print(f"   Point {i}: α_i = {alpha:.6f}, slack = {slack:.6f}, product = {complementary_product:.8f}, satisfied = {satisfied}")
    if not satisfied:
        all_complementary = False
print(f"   All complementary slackness satisfied: {all_complementary}")

print(f"\nOverall KKT conditions satisfied: {all_feasible and all_dual_feasible and all_complementary}")

# Task 4: Generalization bound using VC dimension theory
print("\n" + "="*60)
print("Task 4: Generalization Bound (VC Dimension Theory)")
print("="*60)

n_samples = len(X)
n_features = X.shape[1]
n_support_vectors = len(support_vector_indices)

# VC dimension for linear classifiers in d dimensions is d+1
vc_dimension = n_features + 1

print(f"Dataset characteristics:")
print(f"  Number of samples (n): {n_samples}")
print(f"  Number of features (d): {n_features}")
print(f"  Number of support vectors: {n_support_vectors}")
print(f"  VC dimension: {vc_dimension}")

# Generalization bounds
# 1. VC bound: R(h) <= R_emp(h) + sqrt((VC_dim * log(2n/VC_dim) + log(4/δ)) / (2n))
# 2. Support vector bound: R(h) <= E[# support vectors] / n

delta = 0.05  # Confidence parameter (95% confidence)
empirical_risk = 0  # Perfect classification on training set

# VC bound
import math
vc_term = vc_dimension * math.log(2 * n_samples / vc_dimension) + math.log(4 / delta)
vc_bound = empirical_risk + math.sqrt(vc_term / (2 * n_samples))

# Support vector bound
sv_bound = n_support_vectors / n_samples

print(f"\nGeneralization bounds (with δ = {delta}):")
print(f"  Empirical risk: {empirical_risk:.6f}")
print(f"  VC bound: R(h) <= {vc_bound:.6f}")
print(f"  Support vector bound: R(h) <= {sv_bound:.6f}")

print(f"\nInterpretation:")
print(f"  - VC bound is quite loose for small datasets")
print(f"  - SV bound suggests good generalization (only {n_support_vectors}/{n_samples} = {sv_bound:.1%} of data are SVs)")
print(f"  - Fewer support vectors generally indicate better generalization")

# Task 5: Compare analytical solution with geometric construction
print("\n" + "="*60)
print("Task 5: Geometric Construction Comparison")
print("="*60)

# Geometric approach: Find the line equidistant from closest points of each class
print("Geometric construction approach:")

# Find closest points between classes
min_distance = float('inf')
closest_pair = None

for i, pos_point in enumerate(X_pos):
    for j, neg_point in enumerate(X_neg):
        distance = np.linalg.norm(pos_point - neg_point)
        if distance < min_distance:
            min_distance = distance
            closest_pair = (i, j, pos_point, neg_point)

i_pos, j_neg, closest_pos, closest_neg = closest_pair
print(f"Closest points: Pos[{i_pos}] = {closest_pos}, Neg[{j_neg}] = {closest_neg}")
print(f"Distance between closest points: {min_distance:.6f}")

# The optimal hyperplane is perpendicular to the line connecting closest points
# and passes through their midpoint
direction_vector = closest_pos - closest_neg
midpoint = (closest_pos + closest_neg) / 2

# Normal vector to the hyperplane (perpendicular to direction vector)
w_geometric = direction_vector / np.linalg.norm(direction_vector)
b_geometric = -np.dot(w_geometric, midpoint)

print(f"\nGeometric construction:")
print(f"Direction vector: {direction_vector}")
print(f"Midpoint: {midpoint}")
print(f"Normal vector (normalized): [{w_geometric[0]:.6f}, {w_geometric[1]:.6f}]")
print(f"Bias: {b_geometric:.6f}")

# Compare with analytical solution (normalize for comparison)
w_analytical_norm = w_sklearn / np.linalg.norm(w_sklearn)
b_analytical_norm = b_sklearn / np.linalg.norm(w_sklearn)

print(f"\nComparison (normalized):")
print(f"Analytical: w = [{w_analytical_norm[0]:.6f}, {w_analytical_norm[1]:.6f}], b = {b_analytical_norm:.6f}")
print(f"Geometric:  w = [{w_geometric[0]:.6f}, {w_geometric[1]:.6f}], b = {b_geometric:.6f}")

# Check if they're the same (up to sign)
diff1 = np.linalg.norm(w_analytical_norm - w_geometric) + abs(b_analytical_norm - b_geometric)
diff2 = np.linalg.norm(w_analytical_norm + w_geometric) + abs(b_analytical_norm + b_geometric)
min_diff = min(diff1, diff2)

print(f"Difference (considering sign): {min_diff:.6f}")
print(f"Solutions match: {min_diff < 1e-3}")

# Calculate margin width
margin_width = 2.0 / np.linalg.norm(w_sklearn)
print(f"\nMargin width: {margin_width:.6f}")
print(f"Half margin: {margin_width/2:.6f}")

print(f"\nGeometric verification:")
print(f"Distance from closest points to hyperplane should equal half margin:")
dist_pos = abs(np.dot(w_sklearn, closest_pos) + b_sklearn) / np.linalg.norm(w_sklearn)
dist_neg = abs(np.dot(w_sklearn, closest_neg) + b_sklearn) / np.linalg.norm(w_sklearn)
print(f"Distance from closest positive point: {dist_pos:.6f}")
print(f"Distance from closest negative point: {dist_neg:.6f}")
print(f"Expected distance (half margin): {margin_width/2:.6f}")
print(f"Geometric consistency: {abs(dist_pos - margin_width/2) < 1e-6 and abs(dist_neg - margin_width/2) < 1e-6}")

# Create comprehensive visualizations
print("\n" + "="*60)
print("Creating Visualizations...")
print("="*60)

# Figure 1: Complete SVM analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Dataset and SVM solution
ax = axes[0, 0]

# Plot data points
pos_mask = y == 1
neg_mask = y == -1

ax.scatter(X[pos_mask, 0], X[pos_mask, 1], c='red', s=150, marker='o',
           edgecolor='black', linewidth=2, label='Class +1', zorder=5)
ax.scatter(X[neg_mask, 0], X[neg_mask, 1], c='blue', s=150, marker='s',
           edgecolor='black', linewidth=2, label='Class -1', zorder=5)

# Highlight support vectors
for i, idx in enumerate(support_vector_indices):
    ax.scatter(X[idx, 0], X[idx, 1], s=300, facecolors='none',
               edgecolors='green', linewidth=4, zorder=6)

# Plot decision boundary and margins
x1_range = np.linspace(-1, 6, 100)
if abs(w_sklearn[1]) > 1e-6:
    x2_boundary = (-w_sklearn[0] * x1_range - b_sklearn) / w_sklearn[1]
    x2_margin_pos = (-w_sklearn[0] * x1_range - b_sklearn + 1) / w_sklearn[1]
    x2_margin_neg = (-w_sklearn[0] * x1_range - b_sklearn - 1) / w_sklearn[1]

    ax.plot(x1_range, x2_boundary, 'k-', linewidth=3, label='Decision Boundary')
    ax.plot(x1_range, x2_margin_pos, 'k--', linewidth=2, alpha=0.7, label='Margin Boundaries')
    ax.plot(x1_range, x2_margin_neg, 'k--', linewidth=2, alpha=0.7)

# Add point labels
for i, (point, label) in enumerate(zip(X, y)):
    ax.annotate(f'P{i}', (point[0], point[1]), xytext=(8, 8),
                textcoords='offset points', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

ax.set_xlim(-1, 6)
ax.set_ylim(-1, 5)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Complete SVM Solution')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: KKT conditions visualization
ax = axes[0, 1]

# Create bar plot for KKT conditions
points = [f'P{i}' for i in range(len(X))]
alpha_values = alphas
margin_values = margins - 1  # Slack variables

x_pos = np.arange(len(points))
width = 0.35

bars1 = ax.bar(x_pos - width/2, alpha_values, width, label='$\\alpha_i$', alpha=0.7)
bars2 = ax.bar(x_pos + width/2, margin_values, width, label='Slack $(y_i f(x_i) - 1)$', alpha=0.7)

# Color bars based on support vector status
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    if i in support_vector_indices:
        bar1.set_color('green')
        bar2.set_color('green')
    else:
        bar1.set_color('gray')
        bar2.set_color('gray')

ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax.set_xlabel('Data Points')
ax.set_ylabel('Values')
ax.set_title('KKT Conditions: $\\alpha_i \\cdot \\mathrm{slack}_i = 0$')
ax.set_xticks(x_pos)
ax.set_xticklabels(points)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Geometric construction
ax = axes[1, 0]

# Plot data points
ax.scatter(X[pos_mask, 0], X[pos_mask, 1], c='red', s=150, marker='o',
           edgecolor='black', linewidth=2, label='Class +1', zorder=5)
ax.scatter(X[neg_mask, 0], X[neg_mask, 1], c='blue', s=150, marker='s',
           edgecolor='black', linewidth=2, label='Class -1', zorder=5)

# Highlight closest points
ax.scatter(closest_pos[0], closest_pos[1], s=300, facecolors='none',
           edgecolors='orange', linewidth=4, zorder=6, label='Closest Points')
ax.scatter(closest_neg[0], closest_neg[1], s=300, facecolors='none',
           edgecolors='orange', linewidth=4, zorder=6)

# Draw line between closest points
ax.plot([closest_pos[0], closest_neg[0]], [closest_pos[1], closest_neg[1]],
        'orange', linewidth=3, alpha=0.7, label='Closest Point Connection')

# Draw midpoint
ax.scatter(midpoint[0], midpoint[1], c='orange', s=100, marker='*',
           edgecolor='black', linewidth=2, zorder=7, label='Midpoint')

# Plot geometric decision boundary
if abs(w_geometric[1]) > 1e-6:
    x2_geom_boundary = (-w_geometric[0] * x1_range - b_geometric) / w_geometric[1]
    ax.plot(x1_range, x2_geom_boundary, 'purple', linewidth=3,
            linestyle=':', label='Geometric Boundary')

# Plot analytical boundary for comparison
if abs(w_sklearn[1]) > 1e-6:
    ax.plot(x1_range, x2_boundary, 'k-', linewidth=2, alpha=0.7, label='Analytical Boundary')

ax.set_xlim(-1, 6)
ax.set_ylim(-1, 5)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Geometric Construction')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Generalization analysis
ax = axes[1, 1]

# Create visualization of generalization bounds
methods = ['Empirical\nRisk', 'VC Bound', 'SV Bound']
bounds = [empirical_risk, vc_bound, sv_bound]
colors = ['green', 'red', 'blue']

bars = ax.bar(methods, bounds, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

# Add value labels on bars
for bar, bound in zip(bars, bounds):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{bound:.3f}', ha='center', va='bottom', fontweight='bold')

ax.set_ylabel('Risk Bound')
ax.set_title('Generalization Bounds')
ax.grid(True, alpha=0.3)

# Add text annotations
ax.text(0.02, 0.98, f'Dataset: {n_samples} points, {n_features}D',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
ax.text(0.02, 0.88, f'Support vectors: {n_support_vectors}/{n_samples} = {sv_bound:.1%}',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
ax.text(0.02, 0.78, f'VC dimension: {vc_dimension}',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'complete_svm_analysis.png'), dpi=300, bbox_inches='tight')

# Figure 2: Detailed margin analysis
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Margin distances
ax = axes[0]

# Calculate distances for all points
distances = []
for i, point in enumerate(X):
    dist = abs(np.dot(w_sklearn, point) + b_sklearn) / np.linalg.norm(w_sklearn)
    distances.append(dist)

# Create bar plot
point_labels = [f'P{i}' for i in range(len(X))]
colors = ['green' if i in support_vector_indices else 'gray' for i in range(len(X))]

bars = ax.bar(point_labels, distances, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

# Add horizontal line for margin boundary
ax.axhline(y=margin_width/2, color='red', linestyle='--', linewidth=2,
           label=f'Margin boundary ({margin_width/2:.3f})')

# Add value labels
for bar, dist in zip(bars, distances):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{dist:.3f}', ha='center', va='bottom', fontweight='bold')

ax.set_xlabel('Data Points')
ax.set_ylabel('Distance to Hyperplane')
ax.set_title('Point Distances to Decision Boundary')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Convergence analysis
ax = axes[1]

# Show how the solution converges with different numbers of points
subset_sizes = [3, 4, 5, 6]  # Minimum 3 points needed
margin_widths_subset = []

for size in subset_sizes:
    if size <= len(X):
        X_subset = X[:size]
        y_subset = y[:size]

        try:
            svm_subset = SVC(kernel='linear', C=1e6)
            svm_subset.fit(X_subset, y_subset)
            w_subset = svm_subset.coef_[0]
            margin_subset = 2.0 / np.linalg.norm(w_subset)
            margin_widths_subset.append(margin_subset)
        except:
            margin_widths_subset.append(0)
    else:
        margin_widths_subset.append(margin_width)

ax.plot(subset_sizes, margin_widths_subset, 'bo-', linewidth=2, markersize=8)
ax.axhline(y=margin_width, color='red', linestyle='--', alpha=0.7,
           label=f'Final margin: {margin_width:.3f}')

ax.set_xlabel('Number of Points')
ax.set_ylabel('Margin Width')
ax.set_title('Solution Convergence')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'margin_analysis.png'), dpi=300, bbox_inches='tight')

# Figure 3: Optimization Landscape Visualization
plt.figure(figsize=(12, 8))

# Create a grid around the optimal solution for visualization
w_opt_norm = np.linalg.norm(w_sklearn)
w1_range = np.linspace(w_sklearn[0] - 0.3, w_sklearn[0] + 0.3, 50)
w2_range = np.linspace(w_sklearn[1] - 0.3, w_sklearn[1] + 0.3, 50)
W1, W2 = np.meshgrid(w1_range, w2_range)

# For each point in the grid, compute the objective value and constraint violations
objective_values = np.zeros_like(W1)
max_violation = np.zeros_like(W1)

for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        w_test = np.array([W1[i,j], W2[i,j]])

        # Objective: 1/2 ||w||^2
        objective_values[i,j] = 0.5 * np.dot(w_test, w_test)

        # Constraint violations: max(0, 1 - y_i(w^T x_i + b))
        # Use the optimal b for this w
        margins = y * (X.dot(w_test) + b_sklearn)
        violations = np.maximum(0, 1 - margins)
        max_violation[i,j] = np.max(violations)

# Plot the optimization landscape
plt.subplot(2, 2, 1)
# Only show feasible region (where max_violation <= 0.01)
feasible_mask = max_violation <= 0.01
objective_feasible = np.where(feasible_mask, objective_values, np.nan)

contour = plt.contourf(W1, W2, objective_feasible, levels=20, cmap='viridis', alpha=0.8)
plt.colorbar(contour, label='Objective $\\frac{1}{2}\\|\\mathbf{w}\\|^2$')

# Mark the optimal solution
plt.scatter(w_sklearn[0], w_sklearn[1], c='red', s=200, marker='*',
           edgecolor='white', linewidth=2, zorder=10, label='Optimal $\\mathbf{w}^*$')

plt.xlabel('$w_1$')
plt.ylabel('$w_2$')
plt.title('Optimization Landscape (Feasible Region)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot constraint violations
plt.subplot(2, 2, 2)
violation_contour = plt.contourf(W1, W2, max_violation, levels=20, cmap='Reds', alpha=0.8)
plt.colorbar(violation_contour, label='Max Constraint Violation')

# Mark feasible region boundary
plt.contour(W1, W2, max_violation, levels=[0.01], colors='blue', linewidths=2)

plt.scatter(w_sklearn[0], w_sklearn[1], c='blue', s=200, marker='*',
           edgecolor='white', linewidth=2, zorder=10)

plt.xlabel('$w_1$')
plt.ylabel('$w_2$')
plt.title('Constraint Violation Map')
plt.grid(True, alpha=0.3)

# Plot dual variables and support vectors
plt.subplot(2, 2, 3)
point_indices = np.arange(len(X))
alpha_values = np.zeros(len(X))

# Get dual coefficients
for i, (idx, coef) in enumerate(zip(support_vector_indices, dual_coef)):
    alpha_values[idx] = coef / y[idx]

colors = ['green' if i in support_vector_indices else 'gray' for i in range(len(X))]
bars = plt.bar(point_indices, alpha_values, color=colors, alpha=0.7, edgecolor='black')

plt.xlabel('Data Point Index')
plt.ylabel('Dual Variable $\\alpha_i$')
plt.title('Dual Solution')
plt.grid(True, alpha=0.3)

# Add value labels on bars
for i, (bar, alpha) in enumerate(zip(bars, alpha_values)):
    if alpha > 1e-6:  # Only label non-zero values
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{alpha:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot generalization bounds comparison
plt.subplot(2, 2, 4)
sample_sizes = np.arange(6, 101, 5)
vc_bounds = []
sv_bounds = []

for n in sample_sizes:
    # VC bound (simplified)
    vc_term = vc_dimension * np.log(2 * n / vc_dimension) + np.log(4 / delta)
    vc_bound_n = np.sqrt(vc_term / (2 * n))
    vc_bounds.append(vc_bound_n)

    # SV bound (assuming same ratio of support vectors)
    sv_bound_n = n_support_vectors / n * (len(X) / n)  # Scale by current ratio
    sv_bounds.append(sv_bound_n)

plt.plot(sample_sizes, vc_bounds, 'r-', linewidth=2, label='VC Bound')
plt.plot(sample_sizes, sv_bounds, 'b-', linewidth=2, label='SV Bound')

# Mark current dataset
plt.scatter([len(X)], [vc_bound], c='red', s=100, zorder=5)
plt.scatter([len(X)], [sv_bound], c='blue', s=100, zorder=5)

plt.xlabel('Sample Size $n$')
plt.ylabel('Generalization Bound')
plt.title('Bounds vs Sample Size')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'optimization_landscape.png'), dpi=300, bbox_inches='tight')

print(f"Visualizations saved to: {save_dir}")
print("Files created:")
print("- complete_svm_analysis.png")
print("- margin_analysis.png")
print("- optimization_landscape.png")
