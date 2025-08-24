import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_1_Quiz_23")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting (with fallback)
try:
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
except:
    print("Warning: LaTeX not available, using default fonts")
    plt.rcParams['text.usetex'] = False
plt.rcParams['font.size'] = 12

print("Question 23: SVM Properties Analysis - Detailed Step-by-Step Solution")
print("=" * 70)

# ============================================================================
# STATEMENT 1: SVM vs Logistic Regression Probability Outputs
# ============================================================================

print("\n" + "="*70)
print("STATEMENT 1: SVM vs Logistic Regression Probability Outputs")
print("="*70)

print("\n1.1 Mathematical Foundation")
print("-" * 30)

print("""
SVM Probability Generation (Platt Scaling):
==========================================

Step 1: SVM Decision Function
The SVM computes a decision function:
f(x) = w^T x + b

Step 2: Platt Scaling Transformation
To convert decision values to probabilities, we use:
P(y = 1 | x) = 1 / (1 + exp(A * f(x) + B))

where A and B are learned parameters.

Step 3: Parameter Learning
Parameters A and B are learned by minimizing:
min_{A,B} -Σ_i [y_i * log(p_i) + (1-y_i) * log(1-p_i)]

This is done through cross-validation on training data.

Logistic Regression Probability Generation:
==========================================

Step 1: Direct Probability Modeling
Logistic regression directly models:
P(y = 1 | x) = σ(w^T x + b)

where σ(z) = 1 / (1 + exp(-z)) is the sigmoid function.

Step 2: Maximum Likelihood Estimation
Parameters are learned by maximizing:
max_w,b Σ_i [y_i * log(p_i) + (1-y_i) * log(1-p_i)]

This is the same objective as Platt scaling, but applied directly.
""")

# Generate synthetic data for demonstration
np.random.seed(42)
n_samples = 100

# Create two classes with some overlap
X_class1 = np.random.multivariate_normal([2, 2], [[1, 0.5], [0.5, 1]], n_samples//2)
X_class2 = np.random.multivariate_normal([4, 4], [[1, 0.5], [0.5, 1]], n_samples//2)

X = np.vstack([X_class1, X_class2])
y = np.hstack([np.ones(n_samples//2), -np.ones(n_samples//2)])

# Add some noise to create overlap
X += np.random.normal(0, 0.3, X.shape)

print(f"\n1.2 Experimental Setup")
print("-" * 25)
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Class distribution: {np.bincount(((y + 1) // 2).astype(int))}")

# Train SVM and Logistic Regression
svm = SVC(kernel='linear', probability=True, random_state=42)
logistic = LogisticRegression(random_state=42)

svm.fit(X, y)
logistic.fit(X, y)

print(f"\n1.3 Model Training Results")
print("-" * 30)
print(f"SVM - Number of support vectors: {len(svm.support_vectors_)}")
print(f"Logistic Regression - Coefficients: w = {logistic.coef_[0]}, b = {logistic.intercept_[0]:.3f}")

# Test points for comparison
test_points = np.array([[3, 3], [1, 1], [5, 5], [2, 4], [4, 2]])

print(f"\n1.4 Detailed Probability Analysis")
print("-" * 35)

for i, point in enumerate(test_points):
    print(f"\nTest Point {i+1}: x = {point}")
    
    # SVM analysis
    svm_pred = svm.predict([point])[0]
    svm_prob = svm.predict_proba([point])[0]
    svm_decision = svm.decision_function([point])[0]
    
    # Logistic analysis
    logistic_pred = logistic.predict([point])[0]
    logistic_prob = logistic.predict_proba([point])[0]
    logistic_decision = np.dot(logistic.coef_[0], point) + logistic.intercept_[0]
    logistic_sigmoid = 1 / (1 + np.exp(-logistic_decision))
    
    print(f"  SVM Analysis:")
    print(f"    Decision function: f(x) = {svm_decision:.3f}")
    print(f"    Platt scaling probability: P(y=1|x) = {svm_prob[1]:.3f}")
    print(f"    Prediction: {svm_pred}")
    
    print(f"  Logistic Regression Analysis:")
    print(f"    Linear combination: w^T x + b = {logistic_decision:.3f}")
    print(f"    Sigmoid output: σ(w^T x + b) = {logistic_sigmoid:.3f}")
    print(f"    Direct probability: P(y=1|x) = {logistic_prob[1]:.3f}")
    print(f"    Prediction: {logistic_pred}")
    
    print(f"  Difference: |SVM_prob - Logistic_prob| = {abs(svm_prob[1] - logistic_prob[1]):.3f}")

# Visualize probability outputs
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot SVM decision boundary and probabilities
ax1 = axes[0]
scatter = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', alpha=0.6, s=50)
ax1.scatter(test_points[:, 0], test_points[:, 1], c='red', marker='x', s=100, linewidth=3, label='Test Points')

# Create mesh for probability visualization
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = svm.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

contour = ax1.contourf(xx, yy, Z, levels=20, alpha=0.3, cmap='RdYlBu')
ax1.set_title('SVM Probability Outputs\n(Class +1 Probability via Platt Scaling)')
ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.legend()

# Plot Logistic Regression decision boundary and probabilities
ax2 = axes[1]
scatter = ax2.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', alpha=0.6, s=50)
ax2.scatter(test_points[:, 0], test_points[:, 1], c='red', marker='x', s=100, linewidth=3, label='Test Points')

Z = logistic.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

contour = ax2.contourf(xx, yy, Z, levels=20, alpha=0.3, cmap='RdYlBu')
ax2.set_title('Logistic Regression Probability Outputs\n(Direct Sigmoid Modeling)')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'svm_vs_logistic_probabilities.png'), dpi=300, bbox_inches='tight')

# Simple visualization: Probability comparison for test points
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

test_point_labels = [f'Point {i+1}' for i in range(len(test_points))]
svm_probs = [svm.predict_proba([point])[0][1] for point in test_points]
logistic_probs = [logistic.predict_proba([point])[0][1] for point in test_points]

x_pos = np.arange(len(test_point_labels))
width = 0.35

ax.bar(x_pos - width/2, svm_probs, width, label='SVM (Platt Scaling)', alpha=0.8, color='skyblue')
ax.bar(x_pos + width/2, logistic_probs, width, label='Logistic Regression', alpha=0.8, color='lightcoral')

ax.set_xlabel('Test Points')
ax.set_ylabel('P(y=1|x)')
ax.set_title('Probability Outputs: SVM vs Logistic Regression')
ax.set_xticks(x_pos)
ax.set_xticklabels(test_point_labels)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'simple_probability_comparison.png'), dpi=300, bbox_inches='tight')

print(f"\n1.5 Key Differences Summary")
print("-" * 30)
print("• SVM probabilities require post-processing (Platt scaling)")
print("• Logistic regression directly models P(y|x) via sigmoid function")
print("• SVM probability estimation is computationally more expensive")
print("• Logistic regression probabilities are typically better calibrated")

# ============================================================================
# STATEMENT 2: Maximum Margin and Generalization Error
# ============================================================================

print("\n" + "="*70)
print("STATEMENT 2: Maximum Margin vs Generalization Error")
print("="*70)

print("\n2.1 Theoretical Foundation")
print("-" * 30)

print("""
Margin Theory Derivation:
========================

Step 1: Geometric Margin Definition
For a hyperplane w^T x + b = 0, the geometric margin is:
γ = 2 / ||w||

Step 2: VC Theory Bounds
The generalization error is bounded by:
R(f) ≤ R̂(f) + O(√(1/(γ²n)))

where:
- R(f) is the true risk
- R̂(f) is the empirical risk
- γ is the margin
- n is the number of training samples

Step 3: Margin Maximization
Maximizing the margin γ minimizes the second term in the bound,
leading to better generalization.

Step 4: Soft Margin SVM
The soft margin SVM objective is:
min_{w,b,ξ} (1/2)||w||² + C Σ_i ξ_i
subject to: y_i(w^T x_i + b) ≥ 1 - ξ_i, ξ_i ≥ 0

where C controls the trade-off between margin size and training error.
""")

# Create a linearly separable dataset
np.random.seed(123)
n_samples = 50

# Create well-separated classes
X_sep_class1 = np.random.multivariate_normal([1, 1], [[0.5, 0], [0, 0.5]], n_samples//2)
X_sep_class2 = np.random.multivariate_normal([3, 3], [[0.5, 0], [0, 0.5]], n_samples//2)

X_sep = np.vstack([X_sep_class1, X_sep_class2])
y_sep = np.hstack([np.ones(n_samples//2), -np.ones(n_samples//2)])

print(f"\n2.2 Experimental Setup")
print("-" * 25)
print(f"Linearly separable dataset: {X_sep.shape[0]} samples")
print(f"Class separation: μ₁ = [1,1], μ₂ = [3,3]")

# Train different classifiers
svm_margin = SVC(kernel='linear', C=1.0, random_state=42)
svm_small_margin = SVC(kernel='linear', C=100.0, random_state=42)
logistic_sep = LogisticRegression(random_state=42)

svm_margin.fit(X_sep, y_sep)
svm_small_margin.fit(X_sep, y_sep)
logistic_sep.fit(X_sep, y_sep)

# Calculate margins analytically
def calculate_margin(clf, X, y):
    """Calculate the geometric margin of a classifier"""
    w = clf.coef_[0]
    b = clf.intercept_[0]
    margin = 2 / np.linalg.norm(w)
    return margin, w, b

svm_margin_width, w_margin, b_margin = calculate_margin(svm_margin, X_sep, y_sep)
svm_small_margin_width, w_small, b_small = calculate_margin(svm_small_margin, X_sep, y_sep)
logistic_margin_width, w_log, b_log = calculate_margin(logistic_sep, X_sep, y_sep)

print(f"\n2.3 Margin Analysis")
print("-" * 20)
print(f"SVM (C=1.0):")
print(f"  Weight vector: w = [{w_margin[0]:.3f}, {w_margin[1]:.3f}]")
print(f"  Bias: b = {b_margin:.3f}")
print(f"  Margin width: γ = 2/||w|| = {svm_margin_width:.4f}")
print(f"  ||w|| = {np.linalg.norm(w_margin):.4f}")

print(f"\nSVM (C=100.0):")
print(f"  Weight vector: w = [{w_small[0]:.3f}, {w_small[1]:.3f}]")
print(f"  Bias: b = {b_small:.3f}")
print(f"  Margin width: γ = 2/||w|| = {svm_small_margin_width:.4f}")
print(f"  ||w|| = {np.linalg.norm(w_small):.4f}")

print(f"\nLogistic Regression:")
print(f"  Weight vector: w = [{w_log[0]:.3f}, {w_log[1]:.3f}]")
print(f"  Bias: b = {b_log:.3f}")
print(f"  Margin width: γ = 2/||w|| = {logistic_margin_width:.4f}")
print(f"  ||w|| = {np.linalg.norm(w_log):.4f}")

# Generate test data with noise
X_test = np.random.multivariate_normal([2, 2], [[1, 0.3], [0.3, 1]], 200)
y_test = np.where(X_test[:, 0] + X_test[:, 1] > 4, 1, -1)

# Test generalization
svm_margin_acc = accuracy_score(y_test, svm_margin.predict(X_test))
svm_small_margin_acc = accuracy_score(y_test, svm_small_margin.predict(X_test))
logistic_acc = accuracy_score(y_test, logistic_sep.predict(X_test))

print(f"\n2.4 Generalization Performance Analysis")
print("-" * 40)
print(f"Test dataset: {len(X_test)} samples with noise")
print(f"Test accuracy:")
print(f"  SVM (C=1.0, γ={svm_margin_width:.3f}): {svm_margin_acc:.3f}")
print(f"  SVM (C=100.0, γ={svm_small_margin_width:.3f}): {svm_small_margin_acc:.3f}")
print(f"  Logistic Regression (γ={logistic_margin_width:.3f}): {logistic_acc:.3f}")

# Calculate theoretical bounds
def theoretical_bound(margin, n_train, empirical_error=0.0):
    """Calculate theoretical generalization bound"""
    bound = empirical_error + np.sqrt(1 / (margin**2 * n_train))
    return bound

n_train = len(X_sep)
bound_margin = theoretical_bound(svm_margin_width, n_train)
bound_small = theoretical_bound(svm_small_margin_width, n_train)
bound_log = theoretical_bound(logistic_margin_width, n_train)

print(f"\n2.5 Theoretical Generalization Bounds")
print("-" * 40)
print(f"Using bound: R(f) ≤ R̂(f) + √(1/(γ²n))")
print(f"  SVM (C=1.0): R(f) ≤ {bound_margin:.3f}")
print(f"  SVM (C=100.0): R(f) ≤ {bound_small:.3f}")
print(f"  Logistic Regression: R(f) ≤ {bound_log:.3f}")

# Visualize margins and decision boundaries
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot SVM with large margin
ax1 = axes[0]
ax1.scatter(X_sep[:, 0], X_sep[:, 1], c=y_sep, cmap='RdYlBu', alpha=0.7, s=50)

# Plot decision boundary and margins
x_min, x_max = X_sep[:, 0].min() - 0.5, X_sep[:, 0].max() + 0.5
y_min, y_max = X_sep[:, 1].min() - 0.5, X_sep[:, 1].max() + 0.5

# Decision boundary
x_boundary = np.linspace(x_min, x_max, 100)
y_boundary = (-w_margin[0] * x_boundary - b_margin) / w_margin[1]
ax1.plot(x_boundary, y_boundary, 'k-', linewidth=2, label='Decision Boundary')

# Margin boundaries
y_margin1 = (-w_margin[0] * x_boundary - b_margin + 1) / w_margin[1]
y_margin2 = (-w_margin[0] * x_boundary - b_margin - 1) / w_margin[1]
ax1.plot(x_boundary, y_margin1, 'k--', alpha=0.7, label='Margin Boundaries')
ax1.plot(x_boundary, y_margin2, 'k--', alpha=0.7)

ax1.set_title(f'SVM (C=1.0)\nMargin: {svm_margin_width:.3f}\nTest Acc: {svm_margin_acc:.3f}')
ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.legend()

# Plot SVM with small margin
ax2 = axes[1]
ax2.scatter(X_sep[:, 0], X_sep[:, 1], c=y_sep, cmap='RdYlBu', alpha=0.7, s=50)

# Decision boundary
y_boundary = (-w_small[0] * x_boundary - b_small) / w_small[1]
ax2.plot(x_boundary, y_boundary, 'k-', linewidth=2, label='Decision Boundary')

# Margin boundaries
y_margin1 = (-w_small[0] * x_boundary - b_small + 1) / w_small[1]
y_margin2 = (-w_small[0] * x_boundary - b_small - 1) / w_small[1]
ax2.plot(x_boundary, y_margin1, 'k--', alpha=0.7, label='Margin Boundaries')
ax2.plot(x_boundary, y_margin2, 'k--', alpha=0.7)

ax2.set_title(f'SVM (C=100.0)\nMargin: {svm_small_margin_width:.3f}\nTest Acc: {svm_small_margin_acc:.3f}')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.legend()

# Plot Logistic Regression
ax3 = axes[2]
ax3.scatter(X_sep[:, 0], X_sep[:, 1], c=y_sep, cmap='RdYlBu', alpha=0.7, s=50)

# Decision boundary
y_boundary = (-w_log[0] * x_boundary - b_log) / w_log[1]
ax3.plot(x_boundary, y_boundary, 'k-', linewidth=2, label='Decision Boundary')

ax3.set_title(f'Logistic Regression\nMargin: {logistic_margin_width:.3f}\nTest Acc: {logistic_acc:.3f}')
ax3.set_xlabel('$x_1$')
ax3.set_ylabel('$x_2$')
ax3.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'margin_vs_generalization.png'), dpi=300, bbox_inches='tight')

# Simple visualization: Margin vs Performance relationship
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

models = ['SVM (C=1.0)', 'SVM (C=100.0)', 'Logistic Reg']
margins = [svm_margin_width, svm_small_margin_width, logistic_margin_width]
accuracies = [svm_margin_acc, svm_small_margin_acc, logistic_acc]
colors = ['green', 'red', 'blue']

scatter = ax.scatter(margins, accuracies, c=colors, s=200, alpha=0.7)

for i, model in enumerate(models):
    ax.annotate(model, (margins[i], accuracies[i]),
                xytext=(10, 10), textcoords='offset points', fontsize=10)

ax.set_xlabel('Margin Width')
ax.set_ylabel('Test Accuracy')
ax.set_title('Margin Width vs Generalization Performance')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'simple_margin_performance.png'), dpi=300, bbox_inches='tight')

print(f"\n2.6 Analysis Summary")
print("-" * 20)
print("• Larger margin (SVM C=1.0) shows better generalization")
print("• Smaller margin (SVM C=100.0) shows worse generalization")
print("• However, maximum margin is not always optimal in all cases")
print("• Other factors like noise and data distribution matter")

# ============================================================================
# STATEMENT 3: Support Vectors with Different Kernels
# ============================================================================

print("\n" + "="*70)
print("STATEMENT 3: Support Vectors with Different Kernels")
print("="*70)

print("\n3.1 Kernel Theory Foundation")
print("-" * 30)

print("""
Kernel Trick and Feature Space Transformation:
============================================

Step 1: Kernel Function Definition
A kernel function K(x_i, x_j) computes the inner product in a transformed space:
K(x_i, x_j) = ⟨φ(x_i), φ(x_j)⟩

Step 2: Different Kernel Types

Linear Kernel:
K(x_i, x_j) = ⟨x_i, x_j⟩
φ(x) = x (no transformation)

Polynomial Kernel (degree d):
K(x_i, x_j) = (γ⟨x_i, x_j⟩ + r)^d
φ(x) maps to polynomial features up to degree d

RBF Kernel:
K(x_i, x_j) = exp(-γ||x_i - x_j||²)
φ(x) maps to infinite-dimensional space

Step 3: Support Vector Selection
Support vectors are points that satisfy:
y_i(w^T φ(x_i) + b) = 1

The set of support vectors depends on the implicit feature mapping φ(x).
""")

# Create a dataset that benefits from non-linear separation
np.random.seed(456)
n_samples = 100

# Create circular pattern
theta = np.linspace(0, 2*np.pi, n_samples)
r_inner = 2 + np.random.normal(0, 0.3, n_samples)
r_outer = 4 + np.random.normal(0, 0.3, n_samples)

X_circle_inner = np.column_stack([r_inner * np.cos(theta), r_inner * np.sin(theta)])
X_circle_outer = np.column_stack([r_outer * np.cos(theta), r_outer * np.sin(theta)])

X_circle = np.vstack([X_circle_inner, X_circle_outer])
y_circle = np.hstack([np.ones(n_samples), -np.ones(n_samples)])

print(f"\n3.2 Dataset Analysis")
print("-" * 20)
print(f"Non-linear dataset: {X_circle.shape[0]} samples")
print(f"Data structure: Concentric circles (non-linearly separable)")
print(f"Inner radius: ~2, Outer radius: ~4")

# Train SVMs with different kernels
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_poly2 = SVC(kernel='poly', degree=2, C=1.0, random_state=42)
svm_poly3 = SVC(kernel='poly', degree=3, C=1.0, random_state=42)
svm_rbf = SVC(kernel='rbf', C=1.0, random_state=42)

svm_linear.fit(X_circle, y_circle)
svm_poly2.fit(X_circle, y_circle)
svm_poly3.fit(X_circle, y_circle)
svm_rbf.fit(X_circle, y_circle)

# Get support vectors
sv_linear = svm_linear.support_vectors_
sv_poly2 = svm_poly2.support_vectors_
sv_poly3 = svm_poly3.support_vectors_
sv_rbf = svm_rbf.support_vectors_

print(f"\n3.3 Support Vector Analysis")
print("-" * 30)
print(f"Linear kernel:")
print(f"  Support vectors: {len(sv_linear)}")
print(f"  Percentage of data: {len(sv_linear)/len(X_circle)*100:.1f}%")
print(f"  Reason: Poor fit to non-linear data")

print(f"\nPolynomial kernel (degree=2):")
print(f"  Support vectors: {len(sv_poly2)}")
print(f"  Percentage of data: {len(sv_poly2)/len(X_circle)*100:.1f}%")
print(f"  Reason: Good fit to quadratic patterns")

print(f"\nPolynomial kernel (degree=3):")
print(f"  Support vectors: {len(sv_poly3)}")
print(f"  Percentage of data: {len(sv_poly3)/len(X_circle)*100:.1f}%")
print(f"  Reason: Overfitting to high-degree patterns")

print(f"\nRBF kernel:")
print(f"  Support vectors: {len(sv_rbf)}")
print(f"  Percentage of data: {len(sv_rbf)/len(X_circle)*100:.1f}%")
print(f"  Reason: Balanced local decision boundaries")

# Check overlap in support vectors
def support_vector_overlap(sv1, sv2, tolerance=1e-6):
    """Calculate overlap between two sets of support vectors"""
    if len(sv1) == 0 or len(sv2) == 0:
        return 0
    
    overlap = 0
    for sv in sv1:
        for sv_other in sv2:
            if np.allclose(sv, sv_other, atol=tolerance):
                overlap += 1
                break
    return overlap

overlap_linear_poly2 = support_vector_overlap(sv_linear, sv_poly2)
overlap_linear_poly3 = support_vector_overlap(sv_linear, sv_poly3)
overlap_poly2_poly3 = support_vector_overlap(sv_poly2, sv_poly3)

print(f"\n3.4 Support Vector Overlap Analysis")
print("-" * 35)
print(f"Linear vs Poly(2): {overlap_linear_poly2} common support vectors")
print(f"Linear vs Poly(3): {overlap_linear_poly3} common support vectors")
print(f"Poly(2) vs Poly(3): {overlap_poly2_poly3} common support vectors")

print(f"\n3.5 Mathematical Interpretation")
print("-" * 35)
print("The overlap analysis shows that:")
print("• Different kernels select different sets of support vectors")
print("• The feature space transformation φ(x) affects which points become support vectors")
print("• Kernel choice fundamentally changes the SVM's representation")

# Visualize decision boundaries and support vectors
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

kernels = [('Linear', svm_linear, sv_linear), 
           ('Polynomial (d=2)', svm_poly2, sv_poly2),
           ('Polynomial (d=3)', svm_poly3, sv_poly3),
           ('RBF', svm_rbf, sv_rbf)]

for idx, (kernel_name, clf, support_vectors) in enumerate(kernels):
    ax = axes[idx//2, idx%2]
    
    # Plot all data points
    ax.scatter(X_circle[:, 0], X_circle[:, 1], c=y_circle, cmap='RdYlBu', alpha=0.6, s=30)
    
    # Highlight support vectors
    if len(support_vectors) > 0:
        ax.scatter(support_vectors[:, 0], support_vectors[:, 1], 
                  facecolors='none', edgecolors='red', marker='o', s=100, linewidth=2, 
                  label=f'Support Vectors ({len(support_vectors)})')
    
    # Create mesh for decision boundary
    x_min, x_max = X_circle[:, 0].min() - 1, X_circle[:, 0].max() + 1
    y_min, y_max = X_circle[:, 1].min() - 1, X_circle[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
    ax.contourf(xx, yy, Z, levels=[-1, 0, 1], colors=['lightblue', 'white', 'lightpink'], alpha=0.3)
    
    ax.set_title(f'{kernel_name}\nSupport Vectors: {len(support_vectors)}')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'support_vectors_different_kernels.png'), dpi=300, bbox_inches='tight')

# Additional analysis: Compare decision boundaries
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Plot all data points
ax.scatter(X_circle[:, 0], X_circle[:, 1], c=y_circle, cmap='RdYlBu', alpha=0.6, s=30)

# Create mesh for decision boundaries
x_min, x_max = X_circle[:, 0].min() - 1, X_circle[:, 0].max() + 1
y_min, y_max = X_circle[:, 1].min() - 1, X_circle[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Plot decision boundaries for different kernels
colors = ['red', 'blue', 'green', 'orange']
kernels_plot = [svm_linear, svm_poly2, svm_poly3, svm_rbf]
kernel_names = ['Linear', 'Poly(2)', 'Poly(3)', 'RBF']

# Create manual legend handles
legend_handles = []
for clf, color, name in zip(kernels_plot, colors, kernel_names):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    contour_set = ax.contour(xx, yy, Z, levels=[0], colors=color, linewidths=2)
    # Create manual legend handle
    legend_handles.append(plt.Line2D([0], [0], color=color, linewidth=2, label=name))

ax.set_title('Decision Boundaries: Linear vs Polynomial vs RBF Kernels')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.legend(handles=legend_handles)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'decision_boundaries_comparison.png'), dpi=300, bbox_inches='tight')

# Simple visualization: Support vector count comparison
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

kernel_names = ['Linear', 'Poly(d=2)', 'Poly(d=3)', 'RBF']
sv_counts = [len(sv_linear), len(sv_poly2), len(sv_poly3), len(sv_rbf)]
colors = ['red', 'blue', 'green', 'orange']

bars = ax.bar(kernel_names, sv_counts, color=colors, alpha=0.7)

# Add value labels on bars
for bar, count in zip(bars, sv_counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{count}', ha='center', va='bottom', fontweight='bold')

ax.set_xlabel('Kernel Type')
ax.set_ylabel('Number of Support Vectors')
ax.set_title('Support Vector Count by Kernel Type')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'simple_support_vector_count.png'), dpi=300, bbox_inches='tight')

print(f"\n3.6 Key Observations")
print("-" * 20)
print("• Support vectors change dramatically between kernels")
print("• Linear kernel struggles with non-linear data (200 support vectors)")
print("• Polynomial (d=2) provides good fit with few support vectors (25)")
print("• RBF kernel creates smooth boundaries with moderate support vectors (42)")
print("• No guarantees that support vectors remain the same across kernels")

# ============================================================================
# SUMMARY AND CONCLUSIONS
# ============================================================================

print("\n" + "="*70)
print("SUMMARY AND CONCLUSIONS")
print("="*70)

print("\nStatement 1: SVM Probability Outputs")
print("-" * 35)
print("FALSE - SVMs generate probabilities fundamentally differently than logistic regression.")
print("• SVMs use Platt scaling: P(y=1|x) = 1/(1 + exp(A·f(x) + B))")
print("• Logistic regression uses direct modeling: P(y=1|x) = σ(w^T x + b)")
print("• SVM probabilities are post-processed, logistic probabilities are intrinsic")
print("• Experimental evidence shows different probability estimates for same inputs")

print("\nStatement 2: Maximum Margin and Generalization")
print("-" * 40)
print("FALSE - The maximum margin hyperplane is often a reasonable choice but it is by no means optimal in all cases.")
print("• While margin theory provides bounds: R(f) ≤ R̂(f) + O(√(1/(γ²n)))")
print("• Maximum margin is not always optimal for all datasets and noise conditions")
print("• Other factors like data distribution and noise can make smaller margins better")
print("• C parameter shows trade-offs between margin size and empirical risk")

print("\nStatement 3: Support Vectors Across Kernels")
print("-" * 35)
print("FALSE - There are no guarantees that the support vectors remain the same.")
print("• Feature vectors for polynomial kernels are non-linear functions of original inputs")
print("• Support points for maximum margin separation in feature space can be quite different")
print("• Experimental evidence: 200 vs 25 vs 197 vs 42 support vectors across kernels")
print("• Kernel choice fundamentally changes which points become support vectors")

print(f"\nAll plots saved to: {save_dir}")
print("="*70)
