import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_1_Quiz_14")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("Question 14: Scaling and Invariance Properties")
print("=" * 80)

# Create a sample dataset with different scales
np.random.seed(42)

# Original dataset with different feature scales
X_original = np.array([
    [1, 10],    # Feature 1: small scale, Feature 2: large scale
    [2, 20],
    [3, 15],
    [4, 25],
    [8, 5],     # Different region
    [9, 8],
    [10, 3],
    [11, 7]
])

y_original = np.array([1, 1, 1, 1, -1, -1, -1, -1])

print("Original dataset:")
print("X_original =")
print(X_original)
print("y_original =", y_original)

# Train SVM on original data
svm_original = SVC(kernel='linear', C=1e6)  # Large C for hard margin
svm_original.fit(X_original, y_original)

w_original = svm_original.coef_[0]
b_original = svm_original.intercept_[0]

print(f"\nOriginal SVM solution:")
print(f"w_original = [{w_original[0]:.6f}, {w_original[1]:.6f}]")
print(f"b_original = {b_original:.6f}")
print(f"||w_original|| = {np.linalg.norm(w_original):.6f}")

# Task 1: Effect of scaling features by constant c
print("\n" + "="*60)
print("Task 1: Scaling Features by Constant c")
print("="*60)

scaling_factors = [0.5, 2.0, 5.0, 10.0]
scaling_results = []

for c in scaling_factors:
    # Scale all features by c
    X_scaled = c * X_original
    
    # Train SVM on scaled data
    svm_scaled = SVC(kernel='linear', C=1e6)
    svm_scaled.fit(X_scaled, y_original)
    
    w_scaled = svm_scaled.coef_[0]
    b_scaled = svm_scaled.intercept_[0]
    
    scaling_results.append((c, w_scaled, b_scaled))
    
    print(f"\nScaling factor c = {c}:")
    print(f"X_scaled = c * X_original")
    print(f"w_scaled = [{w_scaled[0]:.6f}, {w_scaled[1]:.6f}]")
    print(f"b_scaled = {b_scaled:.6f}")
    print(f"||w_scaled|| = {np.linalg.norm(w_scaled):.6f}")
    
    # Check the relationship: w_scaled = w_original / c
    w_expected = w_original / c
    b_expected = b_original / c
    
    print(f"Expected: w = w_original / c = [{w_expected[0]:.6f}, {w_expected[1]:.6f}]")
    print(f"Expected: b = b_original / c = {b_expected:.6f}")
    
    w_diff = np.linalg.norm(w_scaled - w_expected)
    b_diff = abs(b_scaled - b_expected)
    
    print(f"Difference in w: {w_diff:.8f}")
    print(f"Difference in b: {b_diff:.8f}")
    print(f"Relationship verified: {w_diff < 1e-6 and b_diff < 1e-6}")

# Task 2: Effect of adding constant to labels
print("\n" + "="*60)
print("Task 2: Adding Constant to Labels")
print("="*60)

print("Original labels:", y_original)
print("Unique values:", np.unique(y_original))

# Try adding different constants
label_shifts = [1, 2, 5, -1]

for k in label_shifts:
    y_shifted = y_original + k
    print(f"\nShifting labels by k = {k}:")
    print(f"y_shifted = y_original + {k} = {y_shifted}")
    print(f"Unique values: {np.unique(y_shifted)}")
    
    # Check if this is still a valid binary classification problem
    unique_vals = np.unique(y_shifted)
    if len(unique_vals) == 2:
        print("Still binary classification - can train SVM")
        
        # Train SVM (but this violates SVM assumptions about labels being ±1)
        try:
            svm_shifted = SVC(kernel='linear', C=1e6)
            svm_shifted.fit(X_original, y_shifted)
            print("SVM training successful (but theoretically invalid)")
        except:
            print("SVM training failed - labels must be binary")
    else:
        print("No longer binary classification - SVM not applicable")

print("\nKey insight: SVM requires labels to be ±1 (or at least two distinct values)")
print("Adding constants to labels breaks the SVM formulation!")

# Task 3: Effect of standardization
print("\n" + "="*60)
print("Task 3: Feature Standardization")
print("="*60)

# Standardize features (zero mean, unit variance)
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X_original)

print("Original feature statistics:")
print(f"Feature 1: mean = {np.mean(X_original[:, 0]):.3f}, std = {np.std(X_original[:, 0]):.3f}")
print(f"Feature 2: mean = {np.mean(X_original[:, 1]):.3f}, std = {np.std(X_original[:, 1]):.3f}")

print("\nStandardized feature statistics:")
print(f"Feature 1: mean = {np.mean(X_standardized[:, 0]):.3f}, std = {np.std(X_standardized[:, 0]):.3f}")
print(f"Feature 2: mean = {np.mean(X_standardized[:, 1]):.3f}, std = {np.std(X_standardized[:, 1]):.3f}")

# Train SVM on standardized data
svm_standardized = SVC(kernel='linear', C=1e6)
svm_standardized.fit(X_standardized, y_original)

w_standardized = svm_standardized.coef_[0]
b_standardized = svm_standardized.intercept_[0]

print(f"\nOriginal SVM solution:")
print(f"w_original = [{w_original[0]:.6f}, {w_original[1]:.6f}]")
print(f"b_original = {b_original:.6f}")

print(f"\nStandardized SVM solution:")
print(f"w_standardized = [{w_standardized[0]:.6f}, {w_standardized[1]:.6f}]")
print(f"b_standardized = {b_standardized:.6f}")

# Compare decision boundaries
print(f"\nDecision boundary comparison:")
print(f"Original: {w_original[0]:.3f}*x1 + {w_original[1]:.3f}*x2 + {b_original:.3f} = 0")
print(f"Standardized: {w_standardized[0]:.3f}*z1 + {w_standardized[1]:.3f}*z2 + {b_standardized:.3f} = 0")
print("where z1, z2 are standardized features")

# Task 4: Importance of feature scaling
print("\n" + "="*60)
print("Task 4: Importance of Feature Scaling")
print("="*60)

# Create an extreme example with very different scales
X_extreme = np.array([
    [1, 1000],      # Feature 1: scale 1, Feature 2: scale 1000
    [2, 2000],
    [3, 1500],
    [4, 2500],
    [100, 50],      # Different region
    [110, 80],
    [120, 30],
    [130, 70]
])

y_extreme = np.array([1, 1, 1, 1, -1, -1, -1, -1])

print("Extreme scale dataset:")
print("Feature 1 range:", np.min(X_extreme[:, 0]), "to", np.max(X_extreme[:, 0]))
print("Feature 2 range:", np.min(X_extreme[:, 1]), "to", np.max(X_extreme[:, 1]))
print("Scale ratio:", np.max(X_extreme[:, 1]) / np.max(X_extreme[:, 0]))

# Train SVM without scaling
svm_extreme = SVC(kernel='linear', C=1e6)
svm_extreme.fit(X_extreme, y_extreme)

w_extreme = svm_extreme.coef_[0]
b_extreme = svm_extreme.intercept_[0]

print(f"\nSVM without scaling:")
print(f"w = [{w_extreme[0]:.6f}, {w_extreme[1]:.6f}]")
print(f"b = {b_extreme:.6f}")
print(f"Weight ratio |w1/w2| = {abs(w_extreme[0]/w_extreme[1]):.6f}")

# Train SVM with scaling
scaler_extreme = StandardScaler()
X_extreme_scaled = scaler_extreme.fit_transform(X_extreme)

svm_extreme_scaled = SVC(kernel='linear', C=1e6)
svm_extreme_scaled.fit(X_extreme_scaled, y_extreme)

w_extreme_scaled = svm_extreme_scaled.coef_[0]
b_extreme_scaled = svm_extreme_scaled.intercept_[0]

print(f"\nSVM with scaling:")
print(f"w = [{w_extreme_scaled[0]:.6f}, {w_extreme_scaled[1]:.6f}]")
print(f"b = {b_extreme_scaled:.6f}")
print(f"Weight ratio |w1/w2| = {abs(w_extreme_scaled[0]/w_extreme_scaled[1]):.6f}")

print(f"\nKey insights:")
print(f"- Without scaling: Feature 2 dominates (large scale)")
print(f"- With scaling: Features have balanced influence")
print(f"- Weight ratios show the effect of scaling")

# Task 5: Affine transformation invariance
print("\n" + "="*60)
print("Task 5: Affine Transformation Invariance")
print("="*60)

# Apply affine transformation: X_new = A * X + b_vec
A = np.array([[2, 1], [0, 3]])  # Linear transformation matrix
b_vec = np.array([5, -2])       # Translation vector

X_affine = X_original @ A.T + b_vec  # Apply transformation

print("Affine transformation:")
print("A =", A)
print("b_vec =", b_vec)
print("X_affine = X_original @ A.T + b_vec")

# Train SVM on transformed data
svm_affine = SVC(kernel='linear', C=1e6)
svm_affine.fit(X_affine, y_original)

w_affine = svm_affine.coef_[0]
b_affine = svm_affine.intercept_[0]

print(f"\nOriginal SVM: w = [{w_original[0]:.6f}, {w_original[1]:.6f}], b = {b_original:.6f}")
print(f"Affine SVM: w = [{w_affine[0]:.6f}, {w_affine[1]:.6f}], b = {b_affine:.6f}")

# The relationship should be: w_affine = A^(-T) @ w_original
A_inv_T = np.linalg.inv(A).T
w_expected = A_inv_T @ w_original

print(f"\nExpected transformation: w_affine = A^(-T) @ w_original")
print(f"A^(-T) =", A_inv_T)
print(f"Expected w_affine = [{w_expected[0]:.6f}, {w_expected[1]:.6f}]")

w_diff = np.linalg.norm(w_affine - w_expected)
print(f"Difference: {w_diff:.8f}")
print(f"Transformation relationship verified: {w_diff < 1e-6}")

print(f"\nKey insight: SVM is NOT invariant to affine transformations")
print(f"The decision boundary changes under affine transformations")

# Create comprehensive visualizations
print("\n" + "="*60)
print("Creating Visualizations...")
print("="*60)

# Figure 1: Scaling effects
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Original data and solution
ax = axes[0, 0]
pos_mask = y_original == 1
neg_mask = y_original == -1

ax.scatter(X_original[pos_mask, 0], X_original[pos_mask, 1],
           c='red', s=100, marker='o', edgecolor='black', label='Class +1', zorder=5)
ax.scatter(X_original[neg_mask, 0], X_original[neg_mask, 1],
           c='blue', s=100, marker='s', edgecolor='black', label='Class -1', zorder=5)

# Plot decision boundary
x1_range = np.linspace(0, 12, 100)
if abs(w_original[1]) > 1e-6:
    x2_boundary = (-w_original[0] * x1_range - b_original) / w_original[1]
    ax.plot(x1_range, x2_boundary, 'k-', linewidth=2, label='Decision Boundary')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('Original Data')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Scaled data (c=2)
ax = axes[0, 1]
c_demo = 2.0
X_demo_scaled = c_demo * X_original
w_demo, b_demo = None, None

for c, w_s, b_s in scaling_results:
    if c == c_demo:
        w_demo, b_demo = w_s, b_s
        break

ax.scatter(X_demo_scaled[pos_mask, 0], X_demo_scaled[pos_mask, 1],
           c='red', s=100, marker='o', edgecolor='black', label='Class +1', zorder=5)
ax.scatter(X_demo_scaled[neg_mask, 0], X_demo_scaled[neg_mask, 1],
           c='blue', s=100, marker='s', edgecolor='black', label='Class -1', zorder=5)

# Plot decision boundary for scaled data
x1_scaled_range = np.linspace(0, 24, 100)
if w_demo is not None and abs(w_demo[1]) > 1e-6:
    x2_scaled_boundary = (-w_demo[0] * x1_scaled_range - b_demo) / w_demo[1]
    ax.plot(x1_scaled_range, x2_scaled_boundary, 'k-', linewidth=2, label='Decision Boundary')

ax.set_xlabel('Feature 1 (scaled by 2)')
ax.set_ylabel('Feature 2 (scaled by 2)')
ax.set_title(f'Scaled Data (c = {c_demo})')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Standardized data
ax = axes[0, 2]
ax.scatter(X_standardized[pos_mask, 0], X_standardized[pos_mask, 1],
           c='red', s=100, marker='o', edgecolor='black', label='Class +1', zorder=5)
ax.scatter(X_standardized[neg_mask, 0], X_standardized[neg_mask, 1],
           c='blue', s=100, marker='s', edgecolor='black', label='Class -1', zorder=5)

# Plot decision boundary for standardized data
x1_std_range = np.linspace(-2, 2, 100)
if abs(w_standardized[1]) > 1e-6:
    x2_std_boundary = (-w_standardized[0] * x1_std_range - b_standardized) / w_standardized[1]
    ax.plot(x1_std_range, x2_std_boundary, 'k-', linewidth=2, label='Decision Boundary')

ax.set_xlabel('Feature 1 (standardized)')
ax.set_ylabel('Feature 2 (standardized)')
ax.set_title('Standardized Data')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Extreme scaling example
ax = axes[1, 0]
pos_mask_extreme = y_extreme == 1
neg_mask_extreme = y_extreme == -1

ax.scatter(X_extreme[pos_mask_extreme, 0], X_extreme[pos_mask_extreme, 1],
           c='red', s=100, marker='o', edgecolor='black', label='Class +1', zorder=5)
ax.scatter(X_extreme[neg_mask_extreme, 0], X_extreme[neg_mask_extreme, 1],
           c='blue', s=100, marker='s', edgecolor='black', label='Class -1', zorder=5)

# Plot decision boundary
x1_extreme_range = np.linspace(0, 140, 100)
if abs(w_extreme[1]) > 1e-6:
    x2_extreme_boundary = (-w_extreme[0] * x1_extreme_range - b_extreme) / w_extreme[1]
    ax.plot(x1_extreme_range, x2_extreme_boundary, 'k-', linewidth=2, label='Decision Boundary')

ax.set_xlabel('Feature 1 (small scale)')
ax.set_ylabel('Feature 2 (large scale)')
ax.set_title('Extreme Scaling (No Preprocessing)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Extreme scaling with standardization
ax = axes[1, 1]
ax.scatter(X_extreme_scaled[pos_mask_extreme, 0], X_extreme_scaled[pos_mask_extreme, 1],
           c='red', s=100, marker='o', edgecolor='black', label='Class +1', zorder=5)
ax.scatter(X_extreme_scaled[neg_mask_extreme, 0], X_extreme_scaled[neg_mask_extreme, 1],
           c='blue', s=100, marker='s', edgecolor='black', label='Class -1', zorder=5)

# Plot decision boundary
x1_extreme_std_range = np.linspace(-2, 2, 100)
if abs(w_extreme_scaled[1]) > 1e-6:
    x2_extreme_std_boundary = (-w_extreme_scaled[0] * x1_extreme_std_range - b_extreme_scaled) / w_extreme_scaled[1]
    ax.plot(x1_extreme_std_range, x2_extreme_std_boundary, 'k-', linewidth=2, label='Decision Boundary')

ax.set_xlabel('Feature 1 (standardized)')
ax.set_ylabel('Feature 2 (standardized)')
ax.set_title('Extreme Scaling (With Standardization)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 6: Affine transformation
ax = axes[1, 2]
ax.scatter(X_affine[pos_mask, 0], X_affine[pos_mask, 1],
           c='red', s=100, marker='o', edgecolor='black', label='Class +1', zorder=5)
ax.scatter(X_affine[neg_mask, 0], X_affine[neg_mask, 1],
           c='blue', s=100, marker='s', edgecolor='black', label='Class -1', zorder=5)

# Plot decision boundary for affine transformed data
x1_affine_range = np.linspace(0, 20, 100)
if abs(w_affine[1]) > 1e-6:
    x2_affine_boundary = (-w_affine[0] * x1_affine_range - b_affine) / w_affine[1]
    ax.plot(x1_affine_range, x2_affine_boundary, 'k-', linewidth=2, label='Decision Boundary')

ax.set_xlabel('Feature 1 (affine transformed)')
ax.set_ylabel('Feature 2 (affine transformed)')
ax.set_title('Affine Transformation')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'scaling_effects.png'), dpi=300, bbox_inches='tight')

# Figure 2: Weight vector analysis
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Weight magnitude vs scaling factor
ax = axes[0, 0]
scales = [result[0] for result in scaling_results]
w_norms = [np.linalg.norm(result[1]) for result in scaling_results]
expected_norms = [np.linalg.norm(w_original) / c for c in scales]

ax.plot(scales, w_norms, 'bo-', linewidth=2, markersize=8, label='Actual ||w||')
ax.plot(scales, expected_norms, 'r--', linewidth=2, label='Expected ||w|| = ||w_0||/c')
ax.set_xlabel('Scaling Factor c')
ax.set_ylabel('||w||')
ax.set_title('Weight Norm vs Scaling Factor')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_yscale('log')

# Plot 2: Weight components comparison
ax = axes[0, 1]
methods = ['Original', 'Standardized', 'Extreme\n(no scaling)', 'Extreme\n(scaled)']
w1_values = [w_original[0], w_standardized[0], w_extreme[0], w_extreme_scaled[0]]
w2_values = [w_original[1], w_standardized[1], w_extreme[1], w_extreme_scaled[1]]

x_pos = np.arange(len(methods))
width = 0.35

ax.bar(x_pos - width/2, w1_values, width, label='w1', alpha=0.7)
ax.bar(x_pos + width/2, w2_values, width, label='w2', alpha=0.7)

ax.set_xlabel('Method')
ax.set_ylabel('Weight Value')
ax.set_title('Weight Components Comparison')
ax.set_xticks(x_pos)
ax.set_xticklabels(methods)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Feature importance (weight magnitude)
ax = axes[1, 0]
feature_importance_orig = np.abs(w_original)
feature_importance_std = np.abs(w_standardized)
feature_importance_extreme = np.abs(w_extreme)
feature_importance_extreme_std = np.abs(w_extreme_scaled)

features = ['Feature 1', 'Feature 2']
x_pos = np.arange(len(features))

ax.bar(x_pos - 0.3, feature_importance_orig, 0.2, label='Original', alpha=0.7)
ax.bar(x_pos - 0.1, feature_importance_std, 0.2, label='Standardized', alpha=0.7)
ax.bar(x_pos + 0.1, feature_importance_extreme, 0.2, label='Extreme (no scaling)', alpha=0.7)
ax.bar(x_pos + 0.3, feature_importance_extreme_std, 0.2, label='Extreme (scaled)', alpha=0.7)

ax.set_xlabel('Features')
ax.set_ylabel('|Weight|')
ax.set_title('Feature Importance (|w_i|)')
ax.set_xticks(x_pos)
ax.set_xticklabels(features)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# Plot 4: Transformation matrix visualization
ax = axes[1, 1]
# Show the affine transformation effect graphically
original_points = np.array([[1, 1], [2, 2], [3, 1]])
transformed_points = original_points @ A.T + b_vec

ax.scatter(original_points[:, 0], original_points[:, 1],
           c='blue', s=100, marker='o', label='Original', alpha=0.7)
ax.scatter(transformed_points[:, 0], transformed_points[:, 1],
           c='red', s=100, marker='s', label='Transformed', alpha=0.7)

# Draw transformation arrows
for orig, trans in zip(original_points, transformed_points):
    ax.annotate('', xy=trans, xytext=orig,
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7))

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Affine Transformation Effect')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'weight_analysis.png'), dpi=300, bbox_inches='tight')

# Figure 3: Feature Scale Impact Visualization
plt.figure(figsize=(14, 8))

# Create datasets with different scale ratios
scale_ratios = [1, 10, 100, 1000]
base_data = np.array([[1, 1], [2, 1], [1, 2], [4, 4], [5, 4], [4, 5]])
base_labels = np.array([1, 1, 1, -1, -1, -1])

for i, ratio in enumerate(scale_ratios):
    ax = plt.subplot(2, 4, i+1)

    # Scale the second feature
    scaled_data = base_data.copy()
    scaled_data[:, 1] *= ratio

    # Train SVM
    svm_scaled = SVC(kernel='linear', C=1e6)
    svm_scaled.fit(scaled_data, base_labels)

    w_scaled = svm_scaled.coef_[0]
    b_scaled = svm_scaled.intercept_[0]

    # Plot data
    pos_mask = base_labels == 1
    neg_mask = base_labels == -1

    ax.scatter(scaled_data[pos_mask, 0], scaled_data[pos_mask, 1],
               c='red', s=100, marker='o', edgecolor='black', zorder=5)
    ax.scatter(scaled_data[neg_mask, 0], scaled_data[neg_mask, 1],
               c='blue', s=100, marker='s', edgecolor='black', zorder=5)

    # Plot decision boundary
    x1_range = np.linspace(0, 6, 100)
    if abs(w_scaled[1]) > 1e-6:
        x2_boundary = (-w_scaled[0] * x1_range - b_scaled) / w_scaled[1]
        ax.plot(x1_range, x2_boundary, 'k-', linewidth=2)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel(f'$x_2$ (scale $\\times${ratio})')
    ax.set_title(f'Scale Ratio 1:{ratio}')
    ax.grid(True, alpha=0.3)

# Plot weight component ratios
ax = plt.subplot(2, 4, 5)
weight_ratios = []
for ratio in scale_ratios:
    scaled_data = base_data.copy()
    scaled_data[:, 1] *= ratio
    svm_scaled = SVC(kernel='linear', C=1e6)
    svm_scaled.fit(scaled_data, base_labels)
    w_scaled = svm_scaled.coef_[0]
    weight_ratios.append(abs(w_scaled[0] / w_scaled[1]) if abs(w_scaled[1]) > 1e-6 else float('inf'))

ax.loglog(scale_ratios, weight_ratios, 'bo-', linewidth=2, markersize=8)
ax.set_xlabel('Feature Scale Ratio')
ax.set_ylabel('$|w_1 / w_2|$')
ax.set_title('Weight Ratio vs Scale Ratio')
ax.grid(True, alpha=0.3)

# Plot margin widths
ax = plt.subplot(2, 4, 6)
margin_widths = []
for ratio in scale_ratios:
    scaled_data = base_data.copy()
    scaled_data[:, 1] *= ratio
    svm_scaled = SVC(kernel='linear', C=1e6)
    svm_scaled.fit(scaled_data, base_labels)
    w_scaled = svm_scaled.coef_[0]
    margin_width = 2.0 / np.linalg.norm(w_scaled)
    margin_widths.append(margin_width)

ax.semilogx(scale_ratios, margin_widths, 'ro-', linewidth=2, markersize=8)
ax.set_xlabel('Feature Scale Ratio')
ax.set_ylabel('Margin Width')
ax.set_title('Margin Width vs Scale Ratio')
ax.grid(True, alpha=0.3)

# Plot standardized comparison
ax = plt.subplot(2, 4, 7)
standardized_weights = []
for ratio in scale_ratios:
    scaled_data = base_data.copy()
    scaled_data[:, 1] *= ratio

    # Standardize
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(scaled_data)

    svm_std = SVC(kernel='linear', C=1e6)
    svm_std.fit(standardized_data, base_labels)
    w_std = svm_std.coef_[0]
    standardized_weights.append(abs(w_std[0] / w_std[1]) if abs(w_std[1]) > 1e-6 else 1.0)

ax.semilogx(scale_ratios, standardized_weights, 'go-', linewidth=2, markersize=8)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
ax.set_xlabel('Original Scale Ratio')
ax.set_ylabel('$|w_1 / w_2|$ (Standardized)')
ax.set_title('Standardization Effect')
ax.grid(True, alpha=0.3)

# Summary comparison
ax = plt.subplot(2, 4, 8)
ax.plot(scale_ratios, weight_ratios, 'bo-', linewidth=2, markersize=8, label='Original')
ax.plot(scale_ratios, standardized_weights, 'go-', linewidth=2, markersize=8, label='Standardized')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Feature Scale Ratio')
ax.set_ylabel('$|w_1 / w_2|$')
ax.set_title('Scaling Impact Summary')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_scale_impact.png'), dpi=300, bbox_inches='tight')

print(f"Visualizations saved to: {save_dir}")
print("Files created:")
print("- scaling_effects.png")
print("- weight_analysis.png")
print("- feature_scale_impact.png")
