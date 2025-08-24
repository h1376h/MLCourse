import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_1_Quiz_6")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("QUESTION 6: SUPPORT VECTOR SPARSITY ANALYSIS")
print("=" * 80)

print("\nProblem Setup:")
print("- Dataset: n = 500 training points in R^10")
print("- After training: k = 50 support vectors")
print("- Non-support vectors: 500 - 50 = 450 points")

print("\n" + "="*60)
print("STEP 1: UNDERSTANDING NON-SUPPORT VECTORS")
print("="*60)

print("\nWhat does this tell us about the remaining 450 points?")
print("\nFor the 450 non-support vector points:")
print("1. Their Lagrange multipliers α_i = 0")
print("2. They satisfy the margin constraint with STRICT inequality:")
print("   y_i(w^T x_i + b) > 1")
print("3. They lie OUTSIDE the margin boundaries")
print("4. They do NOT contribute to defining the decision boundary")
print("5. Their removal would NOT change the optimal hyperplane")

# Create a synthetic example to demonstrate
np.random.seed(42)
print("\n" + "="*60)
print("STEP 2: DEMONSTRATION WITH SYNTHETIC DATA")
print("="*60)

# Generate synthetic data similar to the problem description
X, y = make_classification(n_samples=500, n_features=10, n_redundant=0,
                          n_informative=10, n_clusters_per_class=1,
                          class_sep=1.5, random_state=42)

# Convert labels to -1, +1
y = 2*y - 1

print(f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Class distribution: {np.sum(y == 1)} positive, {np.sum(y == -1)} negative")

# Train SVM
svm = SVC(kernel='linear', C=1.0)
svm.fit(X, y)

# Get support vectors
support_vector_indices = svm.support_
n_support_vectors = len(support_vector_indices)
sparsity_ratio = n_support_vectors / len(y)

print(f"\nSVM Results:")
print(f"Number of support vectors: {n_support_vectors}")
print(f"Sparsity ratio: {sparsity_ratio:.3f} ({sparsity_ratio*100:.1f}%)")
print(f"Non-support vectors: {len(y) - n_support_vectors}")

print("\n" + "="*60)
print("STEP 3: EFFECT OF REMOVING NON-SUPPORT VECTORS")
print("="*60)

# Remove 100 non-support vector points
non_support_indices = np.setdiff1d(np.arange(len(y)), support_vector_indices)
remove_indices = np.random.choice(non_support_indices, size=min(100, len(non_support_indices)), replace=False)

# Create reduced dataset
X_reduced = np.delete(X, remove_indices, axis=0)
y_reduced = np.delete(y, remove_indices)

print(f"Original dataset size: {len(y)}")
print(f"Reduced dataset size: {len(y_reduced)}")
print(f"Removed {len(remove_indices)} non-support vector points")

# Train SVM on reduced dataset
svm_reduced = SVC(kernel='linear', C=1.0)
svm_reduced.fit(X_reduced, y_reduced)

# Compare decision boundaries
w_original = svm.coef_[0]
b_original = svm.intercept_[0]
w_reduced = svm_reduced.coef_[0]
b_reduced = svm_reduced.intercept_[0]

# Normalize for comparison
w_original_norm = w_original / np.linalg.norm(w_original)
w_reduced_norm = w_reduced / np.linalg.norm(w_reduced)

cosine_similarity = np.dot(w_original_norm, w_reduced_norm)
print(f"\nDecision boundary comparison:")
print(f"Cosine similarity between weight vectors: {cosine_similarity:.6f}")
print(f"Difference in normalized weights: {np.linalg.norm(w_original_norm - w_reduced_norm):.6f}")

if cosine_similarity > 0.999:
    print("✓ Decision boundaries are essentially identical!")
else:
    print("⚠ Decision boundaries differ (unexpected for non-support vector removal)")

print("\n" + "="*60)
print("STEP 4: SPARSITY RATIO CALCULATION")
print("="*60)

original_sparsity = 50 / 500
print(f"Original sparsity ratio: {original_sparsity:.3f} = {original_sparsity*100:.1f}%")
print(f"This means {original_sparsity*100:.1f}% of training points are support vectors")
print(f"And {(1-original_sparsity)*100:.1f}% are non-support vectors")

print(f"\nSparsity interpretation:")
print(f"- Only {50} out of {500} points are needed to define the decision boundary")
print(f"- The solution is highly sparse: most training data is 'redundant'")
print(f"- This demonstrates SVM's ability to find compact representations")

print("\n" + "="*60)
print("STEP 5: ADDING NEW POINTS FAR FROM BOUNDARY")
print("="*60)

# Add 50 new points far from the decision boundary
print("Adding 50 new training points far from the decision boundary...")

# Generate points far from the boundary in both classes
# For positive class: generate points with large positive margin
# For negative class: generate points with large negative margin

new_points_pos = []
new_points_neg = []

# Generate positive class points far from boundary
for _ in range(25):
    # Start with a random point
    point = np.random.randn(10)
    # Ensure it's far on the positive side
    while (np.dot(w_original, point) + b_original) < 3.0:  # Far from boundary
        point = np.random.randn(10)
    new_points_pos.append(point)

# Generate negative class points far from boundary
for _ in range(25):
    # Start with a random point
    point = np.random.randn(10)
    # Ensure it's far on the negative side
    while (np.dot(w_original, point) + b_original) > -3.0:  # Far from boundary
        point = np.random.randn(10)
    new_points_neg.append(point)

# Combine new points
X_new = np.vstack([np.array(new_points_pos), np.array(new_points_neg)])
y_new = np.hstack([np.ones(25), -np.ones(25)])

# Add to original dataset
X_extended = np.vstack([X, X_new])
y_extended = np.hstack([y, y_new])

print(f"Extended dataset size: {len(y_extended)}")

# Train SVM on extended dataset
svm_extended = SVC(kernel='linear', C=1.0)
svm_extended.fit(X_extended, y_extended)

n_support_extended = len(svm_extended.support_)
expected_support_vectors = n_support_vectors  # Should remain the same

print(f"\nResults after adding 50 new points:")
print(f"Expected support vectors: {expected_support_vectors}")
print(f"Actual support vectors: {n_support_extended}")
print(f"Difference: {n_support_extended - expected_support_vectors}")

if abs(n_support_extended - expected_support_vectors) <= 2:  # Allow small numerical differences
    print("✓ Number of support vectors remained essentially the same!")
    print("  This confirms that points far from the boundary don't become support vectors")
else:
    print("⚠ Unexpected change in support vector count")

print("\n" + "="*60)
print("STEP 6: WHY SVMs ARE CALLED 'SPARSE' CLASSIFIERS")
print("="*60)

print("SVMs are called 'sparse' classifiers because:")
print("\n1. SOLUTION SPARSITY:")
print("   - Only a subset of training points (support vectors) determine the solution")
print("   - Most training points have α_i = 0 (don't contribute to the decision function)")
print("   - The decision function depends only on support vectors:")
print("     f(x) = Σ α_i y_i K(x_i, x) + b  (sum only over support vectors)")

print("\n2. GEOMETRIC SPARSITY:")
print("   - Only points on or within the margin boundaries matter")
print("   - Points far from the boundary are 'ignored' by the algorithm")
print("   - The solution is determined by the 'most difficult' points to classify")

print("\n3. COMPUTATIONAL SPARSITY:")
print("   - Prediction time depends only on the number of support vectors")
print("   - Storage requirements scale with support vectors, not total training size")
print("   - This makes SVMs efficient for large datasets with few support vectors")

print("\n4. ROBUSTNESS SPARSITY:")
print("   - Removing non-support vectors doesn't change the solution")
print("   - The model is robust to 'redundant' training data")
print("   - Focus on 'boundary' cases leads to good generalization")

print("\n" + "="*60)
print("STEP 7: VISUALIZATION")
print("="*60)

# Create comprehensive visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Sparsity comparison across different scenarios
scenarios = ['Original\n(500 points)', 'After removing\n100 non-SV', 'After adding\n50 far points']
support_counts = [n_support_vectors, len(svm_reduced.support_), n_support_extended]
total_counts = [500, len(y_reduced), len(y_extended)]
sparsity_ratios = [sc/tc for sc, tc in zip(support_counts, total_counts)]

x_pos = np.arange(len(scenarios))
bars1 = ax1.bar(x_pos - 0.2, support_counts, 0.4, label='Support Vectors', color='red', alpha=0.7)
bars2 = ax1.bar(x_pos + 0.2, [tc - sc for tc, sc in zip(total_counts, support_counts)], 0.4,
                label='Non-Support Vectors', color='blue', alpha=0.7)

ax1.set_xlabel('Scenario')
ax1.set_ylabel('Number of Points')
ax1.set_title('Support Vector Counts Across Scenarios')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(scenarios)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for i, (sv, total) in enumerate(zip(support_counts, total_counts)):
    ax1.text(i - 0.2, sv + 5, str(sv), ha='center', va='bottom', fontweight='bold')
    ax1.text(i + 0.2, total - sv + 5, str(total - sv), ha='center', va='bottom', fontweight='bold')

# Plot 2: Sparsity ratios
ax2.bar(x_pos, [sr * 100 for sr in sparsity_ratios], color='green', alpha=0.7)
ax2.set_xlabel('Scenario')
ax2.set_ylabel('Sparsity Ratio (%)')
ax2.set_title('Sparsity Ratio Across Scenarios')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(scenarios)
ax2.grid(True, alpha=0.3)

# Add value labels
for i, sr in enumerate(sparsity_ratios):
    ax2.text(i, sr * 100 + 0.5, f'{sr*100:.1f}%', ha='center', va='bottom', fontweight='bold')

# Plot 3: Decision boundary stability (2D projection for visualization)
# Project to 2D using first two principal components for visualization
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)
X_reduced_2d = pca.transform(X_reduced)

# Plot original data
support_mask = np.zeros(len(y), dtype=bool)
support_mask[support_vector_indices] = True

ax3.scatter(X_2d[~support_mask, 0], X_2d[~support_mask, 1],
           c=y[~support_mask], cmap='RdYlBu', alpha=0.6, s=30, label='Non-Support Vectors')
ax3.scatter(X_2d[support_mask, 0], X_2d[support_mask, 1],
           c=y[support_mask], cmap='RdYlBu', s=100, edgecolors='black',
           linewidth=2, label='Support Vectors')

ax3.set_xlabel('First Principal Component')
ax3.set_ylabel('Second Principal Component')
ax3.set_title('Support Vectors vs Non-Support Vectors\n(2D PCA Projection)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Effect of dataset size on sparsity
dataset_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
sparsity_trends = []

print("\nAnalyzing sparsity trends with dataset size...")
for size in dataset_sizes:
    if size <= len(X):
        X_subset = X[:size]
        y_subset = y[:size]
    else:
        # Generate additional data for larger sizes
        X_extra, y_extra = make_classification(n_samples=size-len(X), n_features=10,
                                             n_redundant=0, n_informative=10,
                                             n_clusters_per_class=1, class_sep=1.5,
                                             random_state=size)
        y_extra = 2*y_extra - 1
        X_subset = np.vstack([X, X_extra])
        y_subset = np.hstack([y, y_extra])

    svm_temp = SVC(kernel='linear', C=1.0)
    svm_temp.fit(X_subset, y_subset)
    sparsity = len(svm_temp.support_) / len(y_subset)
    sparsity_trends.append(sparsity)

ax4.plot(dataset_sizes, [s*100 for s in sparsity_trends], 'o-', linewidth=2, markersize=6)
ax4.set_xlabel('Dataset Size')
ax4.set_ylabel('Sparsity Ratio (%)')
ax4.set_title('Sparsity Ratio vs Dataset Size')
ax4.grid(True, alpha=0.3)

# Add trend annotation
if sparsity_trends[-1] < sparsity_trends[0]:
    trend_text = "Sparsity decreases with dataset size"
else:
    trend_text = "Sparsity remains relatively stable"
ax4.text(0.05, 0.95, trend_text, transform=ax4.transAxes,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'svm_sparsity_analysis.png'), dpi=300, bbox_inches='tight')

print(f"Visualization saved to: {os.path.join(save_dir, 'svm_sparsity_analysis.png')}")

print("\n" + "="*80)
print("SUMMARY OF FINDINGS")
print("="*80)

print(f"\n1. NON-SUPPORT VECTORS ({500-50} points):")
print("   - Have α_i = 0 (don't contribute to decision function)")
print("   - Satisfy y_i(w^T x_i + b) > 1 (lie outside margin)")
print("   - Can be removed without affecting the decision boundary")

print(f"\n2. REMOVING NON-SUPPORT VECTORS:")
print("   - Decision boundary remains unchanged")
print("   - Computational efficiency improves")
print("   - Storage requirements reduce")

print(f"\n3. SPARSITY RATIO:")
print(f"   - Original: {original_sparsity:.3f} ({original_sparsity*100:.1f}%)")
print("   - Interpretation: Only 10% of data points are 'essential'")

print(f"\n4. ADDING DISTANT POINTS:")
print(f"   - Expected support vectors: ~{expected_support_vectors}")
print(f"   - Actual support vectors: {n_support_extended}")
print("   - Points far from boundary don't become support vectors")

print(f"\n5. SVM SPARSITY:")
print("   - Solution depends only on boundary points")
print("   - Prediction complexity: O(n_support_vectors)")
print("   - Memory complexity: O(n_support_vectors)")
print("   - Robust to redundant training data")

print("\n" + "="*80)
print("SOLUTION COMPLETE")
print("="*80)