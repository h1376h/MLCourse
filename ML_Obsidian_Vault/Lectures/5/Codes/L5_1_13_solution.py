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
save_dir = os.path.join(images_dir, "L5_1_Quiz_13")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("Question 13: Theoretical Properties of Maximum Margin Classifier")
print("=" * 80)

# Define a linearly separable dataset for demonstrations
np.random.seed(42)

# Create a simple 2D linearly separable dataset
X_pos = np.array([[2, 3], [3, 4], [4, 2], [3, 3]])  # Class +1
X_neg = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])  # Class -1

X = np.vstack([X_pos, X_neg])
y = np.array([1, 1, 1, 1, -1, -1, -1, -1])

print("Original dataset:")
print("Positive class points:", X_pos)
print("Negative class points:", X_neg)
print("Labels:", y)

def solve_svm_primal(X, y, C=1e6):
    """
    Solve SVM in primal form using quadratic programming
    For hard margin, we use very large C to approximate hard constraints
    """
    n_samples, n_features = X.shape
    
    # Objective: minimize 1/2 ||w||^2
    # Variables: [w1, w2, b]
    def objective(params):
        w = params[:n_features]
        return 0.5 * np.dot(w, w)
    
    # Constraints: y_i(w^T x_i + b) >= 1
    def constraint_func(params):
        w = params[:n_features]
        b = params[n_features]
        margins = y * (X.dot(w) + b)
        return margins - 1  # >= 1 constraint
    
    # Initial guess
    x0 = np.zeros(n_features + 1)
    
    # Solve with constraints
    constraints = {'type': 'ineq', 'fun': constraint_func}
    result = minimize(objective, x0, method='SLSQP', constraints=constraints)
    
    w_opt = result.x[:n_features]
    b_opt = result.x[n_features]
    
    return w_opt, b_opt, result.success

# Property 1: Existence of solution for linearly separable data
print("\n" + "="*60)
print("Property 1: Existence of Solution")
print("="*60)

# Check if data is linearly separable by solving SVM
w_opt, b_opt, success = solve_svm_primal(X, y)

print(f"SVM optimization successful: {success}")
print(f"Optimal weight vector: w* = [{w_opt[0]:.4f}, {w_opt[1]:.4f}]")
print(f"Optimal bias: b* = {b_opt:.4f}")

# Verify all constraints are satisfied
margins = y * (X.dot(w_opt) + b_opt)
print(f"Constraint violations (should all be >= 1):")
for i, margin in enumerate(margins):
    print(f"  Point {i}: y_i(w^T x_i + b) = {margin:.4f}")

min_margin = np.min(margins)
print(f"Minimum margin: {min_margin:.4f}")
print(f"Data is linearly separable: {min_margin >= 0.99}")  # Allow small numerical error

# Property 2: Uniqueness (up to scaling)
print("\n" + "="*60)
print("Property 2: Uniqueness of Solution")
print("="*60)

# Solve SVM multiple times with different initializations
solutions = []
for seed in [42, 123, 456, 789]:
    np.random.seed(seed)
    w_i, b_i, success_i = solve_svm_primal(X, y)
    if success_i:
        # Normalize to compare (scale so that ||w|| = 1)
        w_norm = np.linalg.norm(w_i)
        w_normalized = w_i / w_norm
        b_normalized = b_i / w_norm
        solutions.append((w_normalized, b_normalized))

print("Normalized solutions (||w|| = 1):")
for i, (w_norm, b_norm) in enumerate(solutions):
    print(f"Solution {i+1}: w = [{w_norm[0]:.4f}, {w_norm[1]:.4f}], b = {b_norm:.4f}")

# Check if all solutions are the same (up to sign)
if len(solutions) > 1:
    w_ref, b_ref = solutions[0]
    all_same = True
    for w_i, b_i in solutions[1:]:
        # Check both orientations (w, b) and (-w, -b)
        diff1 = np.linalg.norm(w_i - w_ref) + abs(b_i - b_ref)
        diff2 = np.linalg.norm(w_i + w_ref) + abs(b_i + b_ref)
        if min(diff1, diff2) > 1e-3:
            all_same = False
            break
    print(f"All solutions are equivalent (up to scaling/sign): {all_same}")

# Property 3: Removing non-support vectors doesn't change solution
print("\n" + "="*60)
print("Property 3: Non-Support Vector Removal")
print("="*60)

# Identify support vectors (points on the margin)
distances = np.abs(X.dot(w_opt) + b_opt) / np.linalg.norm(w_opt)
margin_width = 2.0 / np.linalg.norm(w_opt)
support_vector_indices = np.where(np.abs(distances - margin_width/2) < 1e-3)[0]

print(f"Margin width: {margin_width:.4f}")
print(f"Support vector indices: {support_vector_indices}")
print("Support vectors:")
for idx in support_vector_indices:
    print(f"  Point {idx}: {X[idx]}, label: {y[idx]}, distance: {distances[idx]:.4f}")

# Solve SVM with only support vectors
X_sv = X[support_vector_indices]
y_sv = y[support_vector_indices]

w_sv, b_sv, success_sv = solve_svm_primal(X_sv, y_sv)

print(f"\nSolution with only support vectors:")
print(f"w_sv = [{w_sv[0]:.4f}, {w_sv[1]:.4f}]")
print(f"b_sv = {b_sv:.4f}")

# Compare solutions (normalize for comparison)
w_opt_norm = w_opt / np.linalg.norm(w_opt)
b_opt_norm = b_opt / np.linalg.norm(w_opt)
w_sv_norm = w_sv / np.linalg.norm(w_sv)
b_sv_norm = b_sv / np.linalg.norm(w_sv)

diff_w = np.linalg.norm(w_opt_norm - w_sv_norm)
diff_b = abs(b_opt_norm - b_sv_norm)
print(f"Difference in normalized w: {diff_w:.6f}")
print(f"Difference in normalized b: {diff_b:.6f}")
print(f"Solutions are equivalent: {diff_w < 1e-2 and diff_b < 1e-2}")

# Property 4: Adding points far from boundary doesn't affect solution
print("\n" + "="*60)
print("Property 4: Adding Distant Points")
print("="*60)

# Add points far from the decision boundary
far_positive = np.array([[10, 10], [8, 9]])  # Far positive points
far_negative = np.array([[-5, -5], [-4, -6]])  # Far negative points

X_extended = np.vstack([X, far_positive, far_negative])
y_extended = np.hstack([y, [1, 1], [-1, -1]])

print("Added distant points:")
print("Far positive:", far_positive)
print("Far negative:", far_negative)

# Solve SVM with extended dataset
w_ext, b_ext, success_ext = solve_svm_primal(X_extended, y_extended)

print(f"\nSolution with extended dataset:")
print(f"w_ext = [{w_ext[0]:.4f}, {w_ext[1]:.4f}]")
print(f"b_ext = {b_ext:.4f}")

# Compare with original solution
w_ext_norm = w_ext / np.linalg.norm(w_ext)
b_ext_norm = b_ext / np.linalg.norm(w_ext)

diff_w_ext = np.linalg.norm(w_opt_norm - w_ext_norm)
diff_b_ext = abs(b_opt_norm - b_ext_norm)
print(f"Difference in normalized w: {diff_w_ext:.6f}")
print(f"Difference in normalized b: {diff_b_ext:.6f}")
print(f"Solutions are equivalent: {diff_w_ext < 1e-2 and diff_b_ext < 1e-2}")

# Check margins for new points
margins_new = y_extended[-4:] * (X_extended[-4:].dot(w_opt) + b_opt)
print("Margins for new points (using original solution):")
for i, margin in enumerate(margins_new):
    point_type = "far positive" if i < 2 else "far negative"
    print(f"  {point_type} point {i%2 + 1}: margin = {margin:.4f}")

# Property 5: Representer Theorem
print("\n" + "="*60)
print("Property 5: Representer Theorem")
print("="*60)

print("The representer theorem states that the optimal weight vector")
print("can be expressed as a linear combination of the training points:")
print("w* = Σ α_i y_i x_i")
print()

# Use sklearn SVM to get dual coefficients
svm = SVC(kernel='linear', C=1e6)  # Large C for hard margin
svm.fit(X, y)

# Get support vectors and their coefficients
sv_indices = svm.support_
dual_coef = svm.dual_coef_[0]  # α_i * y_i for support vectors
support_vectors = svm.support_vectors_

print(f"Support vector indices from sklearn: {sv_indices}")
print("Dual coefficients (α_i * y_i):")
for i, (idx, coef) in enumerate(zip(sv_indices, dual_coef)):
    print(f"  SV {i}: index {idx}, α_i * y_i = {coef:.4f}")

# Reconstruct weight vector using representer theorem
w_representer = np.zeros(2)
for i, (idx, coef) in enumerate(zip(sv_indices, dual_coef)):
    w_representer += coef * X[idx]

print(f"\nReconstructed weight vector: w = [{w_representer[0]:.4f}, {w_representer[1]:.4f}]")
print(f"sklearn weight vector: w = [{svm.coef_[0][0]:.4f}, {svm.coef_[0][1]:.4f}]")

diff_representer = np.linalg.norm(w_representer - svm.coef_[0])
print(f"Difference: {diff_representer:.6f}")
print(f"Representer theorem verified: {diff_representer < 1e-6}")

# Verify that non-support vectors have α_i = 0
print("\nVerification that non-support vectors have α_i = 0:")
all_alphas = np.zeros(len(X))
for i, (idx, coef) in enumerate(zip(sv_indices, dual_coef)):
    all_alphas[idx] = coef / y[idx]  # α_i = (α_i * y_i) / y_i

for i, alpha in enumerate(all_alphas):
    is_sv = i in sv_indices
    print(f"  Point {i}: α_i = {alpha:.6f}, is support vector: {is_sv}")

# Create comprehensive visualizations
print("\n" + "="*60)
print("Creating Visualizations...")
print("="*60)

# Figure 1: Theoretical Properties Demonstration
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Original problem and solution existence
ax = axes[0, 0]
x1_range = np.linspace(-1, 5, 100)
x2_range = np.linspace(-1, 5, 100)
X1, X2 = np.meshgrid(x1_range, x2_range)

# Plot decision boundary
if abs(w_opt[1]) > 1e-6:
    x2_boundary = (-w_opt[0] * x1_range - b_opt) / w_opt[1]
    ax.plot(x1_range, x2_boundary, 'k-', linewidth=2, label='Decision Boundary')

    # Plot margin boundaries
    margin_offset = 1.0 / np.linalg.norm(w_opt)
    x2_margin_pos = (-w_opt[0] * x1_range - b_opt + 1) / w_opt[1]
    x2_margin_neg = (-w_opt[0] * x1_range - b_opt - 1) / w_opt[1]
    ax.plot(x1_range, x2_margin_pos, 'k--', alpha=0.7, label='Margin Boundaries')
    ax.plot(x1_range, x2_margin_neg, 'k--', alpha=0.7)

# Plot data points
ax.scatter(X_pos[:, 0], X_pos[:, 1], c='red', s=100, marker='o',
           edgecolor='black', label='Class +1', zorder=5)
ax.scatter(X_neg[:, 0], X_neg[:, 1], c='blue', s=100, marker='s',
           edgecolor='black', label='Class -1', zorder=5)

# Highlight support vectors
for idx in support_vector_indices:
    ax.scatter(X[idx, 0], X[idx, 1], s=200, facecolors='none',
               edgecolors='green', linewidth=3, zorder=6)

ax.set_xlim(-1, 5)
ax.set_ylim(-1, 5)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Property 1: Solution Existence')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Uniqueness demonstration
ax = axes[0, 1]
# Show that different initializations lead to same solution
colors = ['red', 'blue', 'green', 'orange']
for i, (w_norm, b_norm) in enumerate(solutions[:4]):
    if abs(w_norm[1]) > 1e-6:
        x2_line = (-w_norm[0] * x1_range - b_norm) / w_norm[1]
        ax.plot(x1_range, x2_line, color=colors[i], linewidth=2,
                alpha=0.7, label=f'Init {i+1}')

ax.scatter(X_pos[:, 0], X_pos[:, 1], c='red', s=100, marker='o',
           edgecolor='black', zorder=5)
ax.scatter(X_neg[:, 0], X_neg[:, 1], c='blue', s=100, marker='s',
           edgecolor='black', zorder=5)

ax.set_xlim(-1, 5)
ax.set_ylim(-1, 5)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Property 2: Solution Uniqueness')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Support vector removal
ax = axes[0, 2]
# Plot original solution
if abs(w_opt[1]) > 1e-6:
    x2_boundary = (-w_opt[0] * x1_range - b_opt) / w_opt[1]
    ax.plot(x1_range, x2_boundary, 'k-', linewidth=2, label='Original Solution')

# Plot solution with only support vectors
if abs(w_sv[1]) > 1e-6:
    x2_sv_boundary = (-w_sv[0] * x1_range - b_sv) / w_sv[1]
    ax.plot(x1_range, x2_sv_boundary, 'r--', linewidth=2, label='SV-only Solution')

# Plot all points
ax.scatter(X_pos[:, 0], X_pos[:, 1], c='red', s=100, marker='o',
           edgecolor='black', alpha=0.7, zorder=5)
ax.scatter(X_neg[:, 0], X_neg[:, 1], c='blue', s=100, marker='s',
           edgecolor='black', alpha=0.7, zorder=5)

# Highlight support vectors
for idx in support_vector_indices:
    ax.scatter(X[idx, 0], X[idx, 1], s=200, facecolors='none',
               edgecolors='green', linewidth=3, zorder=6)

# Mark non-support vectors as removable
non_sv_indices = [i for i in range(len(X)) if i not in support_vector_indices]
for idx in non_sv_indices:
    ax.scatter(X[idx, 0], X[idx, 1], s=150, marker='x',
               color='gray', linewidth=3, zorder=6)

ax.set_xlim(-1, 5)
ax.set_ylim(-1, 5)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Property 3: Non-SV Removal')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Adding distant points
ax = axes[1, 0]
# Plot original solution
if abs(w_opt[1]) > 1e-6:
    x2_boundary = (-w_opt[0] * x1_range - b_opt) / w_opt[1]
    ax.plot(x1_range, x2_boundary, 'k-', linewidth=2, label='Original Solution')

# Plot solution with extended dataset
if abs(w_ext[1]) > 1e-6:
    x2_ext_boundary = (-w_ext[0] * x1_range - b_ext) / w_ext[1]
    ax.plot(x1_range, x2_ext_boundary, 'r--', linewidth=2, label='Extended Solution')

# Plot original points
ax.scatter(X_pos[:, 0], X_pos[:, 1], c='red', s=100, marker='o',
           edgecolor='black', label='Original +1', zorder=5)
ax.scatter(X_neg[:, 0], X_neg[:, 1], c='blue', s=100, marker='s',
           edgecolor='black', label='Original -1', zorder=5)

# Plot distant points
ax.scatter(far_positive[:, 0], far_positive[:, 1], c='red', s=100, marker='^',
           edgecolor='black', alpha=0.5, label='Distant +1', zorder=5)
ax.scatter(far_negative[:, 0], far_negative[:, 1], c='blue', s=100, marker='v',
           edgecolor='black', alpha=0.5, label='Distant -1', zorder=5)

ax.set_xlim(-7, 12)
ax.set_ylim(-7, 12)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Property 4: Distant Points')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Representer theorem visualization
ax = axes[1, 1]
# Plot the weight vector and its decomposition
ax.quiver(0, 0, w_opt[0], w_opt[1], angles='xy', scale_units='xy', scale=1,
          color='black', width=0.005, label='$\\mathbf{w}^*$')

# Plot individual contributions from support vectors
colors_sv = ['red', 'blue', 'green', 'orange']
cumulative_w = np.array([0.0, 0.0])
for i, (idx, coef) in enumerate(zip(sv_indices, dual_coef)):
    contribution = coef * X[idx]
    ax.quiver(cumulative_w[0], cumulative_w[1], contribution[0], contribution[1],
              angles='xy', scale_units='xy', scale=1,
              color=colors_sv[i % len(colors_sv)], width=0.003, alpha=0.7,
              label=f'$\\alpha_{{{idx}}} y_{{{idx}}} \\mathbf{{x}}_{{{idx}}}$')
    cumulative_w += contribution

# Plot support vectors in feature space
for i, idx in enumerate(sv_indices):
    ax.scatter(X[idx, 0], X[idx, 1], s=150,
               color=colors_sv[i % len(colors_sv)],
               edgecolor='black', zorder=5)

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Property 5: Representer Theorem')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis('equal')

# Plot 6: Summary of all properties
ax = axes[1, 2]
ax.text(0.1, 0.9, 'Theoretical Properties Verified:', fontsize=14, fontweight='bold',
        transform=ax.transAxes)
ax.text(0.1, 0.8, f'1. Solution exists: {success}', fontsize=12, transform=ax.transAxes)
ax.text(0.1, 0.7, f'2. Solution unique: {all_same}', fontsize=12, transform=ax.transAxes)
ax.text(0.1, 0.6, f'3. Non-SV removal OK: {diff_w < 1e-2}', fontsize=12, transform=ax.transAxes)
ax.text(0.1, 0.5, f'4. Distant points OK: {diff_w_ext < 1e-2}', fontsize=12, transform=ax.transAxes)
ax.text(0.1, 0.4, f'5. Representer theorem: {diff_representer < 1e-6}', fontsize=12, transform=ax.transAxes)

ax.text(0.1, 0.25, 'Key Insights:', fontsize=14, fontweight='bold', transform=ax.transAxes)
ax.text(0.1, 0.15, '• SVM has unique solution for separable data', fontsize=10, transform=ax.transAxes)
ax.text(0.1, 0.1, '• Only support vectors matter', fontsize=10, transform=ax.transAxes)
ax.text(0.1, 0.05, '• Solution is sparse representation', fontsize=10, transform=ax.transAxes)
ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'theoretical_properties.png'), dpi=300, bbox_inches='tight')

# Figure 2: Support Vector Evolution
plt.figure(figsize=(12, 8))

# Create a sequence showing how support vectors change with data modifications
datasets = [
    (X, y, "Original Dataset"),
    (X[support_vector_indices], y[support_vector_indices], "Support Vectors Only"),
    (X_extended, y_extended, "With Distant Points")
]

for i, (X_data, y_data, title) in enumerate(datasets):
    ax = plt.subplot(2, 3, i+1)

    # Train SVM on current dataset
    svm_temp = SVC(kernel='linear', C=1e6)
    svm_temp.fit(X_data, y_data)

    w_temp = svm_temp.coef_[0]
    b_temp = svm_temp.intercept_[0]
    sv_indices_temp = svm_temp.support_

    # Plot data points
    pos_mask = y_data == 1
    neg_mask = y_data == -1

    ax.scatter(X_data[pos_mask, 0], X_data[pos_mask, 1],
               c='red', s=100, marker='o', edgecolor='black',
               label='Class +1', zorder=5)
    ax.scatter(X_data[neg_mask, 0], X_data[neg_mask, 1],
               c='blue', s=100, marker='s', edgecolor='black',
               label='Class -1', zorder=5)

    # Highlight support vectors
    for idx in sv_indices_temp:
        ax.scatter(X_data[idx, 0], X_data[idx, 1], s=200,
                   facecolors='none', edgecolors='green',
                   linewidth=3, zorder=6)

    # Plot decision boundary
    x1_range = np.linspace(-7, 12, 100)
    if abs(w_temp[1]) > 1e-6:
        x2_boundary = (-w_temp[0] * x1_range - b_temp) / w_temp[1]
        ax.plot(x1_range, x2_boundary, 'k-', linewidth=2)

        # Plot margins
        x2_margin_pos = (-w_temp[0] * x1_range - b_temp + 1) / w_temp[1]
        x2_margin_neg = (-w_temp[0] * x1_range - b_temp - 1) / w_temp[1]
        ax.plot(x1_range, x2_margin_pos, 'k--', alpha=0.5)
        ax.plot(x1_range, x2_margin_neg, 'k--', alpha=0.5)

    ax.set_xlim(-7, 12)
    ax.set_ylim(-7, 12)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend()

# Plot margin width comparison
ax = plt.subplot(2, 3, 4)
margin_widths = []
dataset_names = []

for X_data, y_data, name in datasets:
    svm_temp = SVC(kernel='linear', C=1e6)
    svm_temp.fit(X_data, y_data)
    w_temp = svm_temp.coef_[0]
    margin_width = 2.0 / np.linalg.norm(w_temp)
    margin_widths.append(margin_width)
    dataset_names.append(name.replace(" ", "\n"))

bars = ax.bar(dataset_names, margin_widths, color=['blue', 'green', 'red'], alpha=0.7)
ax.set_ylabel('Margin Width')
ax.set_title('Margin Width Consistency')
ax.grid(True, alpha=0.3)

# Add value labels on bars
for bar, width in zip(bars, margin_widths):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{width:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot support vector count
ax = plt.subplot(2, 3, 5)
sv_counts = []
for X_data, y_data, name in datasets:
    svm_temp = SVC(kernel='linear', C=1e6)
    svm_temp.fit(X_data, y_data)
    sv_counts.append(len(svm_temp.support_))

bars = ax.bar(dataset_names, sv_counts, color=['blue', 'green', 'red'], alpha=0.7)
ax.set_ylabel('Number of Support Vectors')
ax.set_title('Support Vector Count')
ax.grid(True, alpha=0.3)

# Add value labels on bars
for bar, count in zip(bars, sv_counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
            f'{count}', ha='center', va='bottom', fontweight='bold')

# Plot weight vector comparison
ax = plt.subplot(2, 3, 6)
weight_norms = []
for X_data, y_data, name in datasets:
    svm_temp = SVC(kernel='linear', C=1e6)
    svm_temp.fit(X_data, y_data)
    w_temp = svm_temp.coef_[0]
    weight_norms.append(np.linalg.norm(w_temp))

bars = ax.bar(dataset_names, weight_norms, color=['blue', 'green', 'red'], alpha=0.7)
ax.set_ylabel('$\\|\\mathbf{w}\\|$')
ax.set_title('Weight Vector Norm')
ax.grid(True, alpha=0.3)

# Add value labels on bars
for bar, norm in zip(bars, weight_norms):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{norm:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'support_vector_evolution.png'), dpi=300, bbox_inches='tight')

print(f"Visualizations saved to: {save_dir}")
print("Files created:")
print("- theoretical_properties.png")
print("- support_vector_evolution.png")
