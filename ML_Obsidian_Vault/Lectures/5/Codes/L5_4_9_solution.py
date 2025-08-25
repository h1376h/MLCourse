import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from cvxopt import matrix, solvers
import os
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_4_Quiz_9")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

# Suppress CVXOPT output
solvers.options['show_progress'] = False

print("=" * 80)
print("DIRECT MULTI-CLASS SVM FORMULATION ANALYSIS")
print("=" * 80)

# ============================================================================
# PART 1: UNDERSTANDING THE CONSTRAINT
# ============================================================================

print("\n1. INTERPRETING THE CONSTRAINT")
print("-" * 50)

# Generate sample data for demonstration
np.random.seed(42)
X, y = make_blobs(n_samples=100, centers=3, cluster_std=1.0, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Sample point and weights for demonstration
sample_point = X[0]  # First point
true_class = y[0]    # True class of the point
K = 3               # Number of classes

print(f"Sample point: {sample_point}")
print(f"True class: {true_class}")
print(f"Number of classes: {K}")

# Create sample weight vectors for demonstration
np.random.seed(42)
W = np.random.randn(K, X.shape[1]) * 0.5  # K x d weight matrix

print(f"\nWeight matrix W (K x d):")
print(W)

# Demonstrate the constraint for the sample point
print(f"\nConstraint analysis for point {sample_point} (true class {true_class}):")

for k in range(K):
    if k != true_class:
        # Calculate w_yi^T * x_i - w_k^T * x_i
        margin = np.dot(W[true_class], sample_point) - np.dot(W[k], sample_point)
        print(f"  w_{true_class}^T * x - w_{k}^T * x = {margin:.3f}")
        print(f"  This should be >= 1 - \\xi_{{{true_class}{k}}}")
        print(f"  Current margin: {margin:.3f}")
        if margin < 1:
            slack_needed = 1 - margin
            print(f"  Slack variable \\xi_{{{true_class}{k}}} needed: {slack_needed:.3f}")
        else:
            print(f"  No slack needed (margin >= 1)")
        print()

# ============================================================================
# PART 2: COUNTING SLACK VARIABLES
# ============================================================================

print("\n2. COUNTING SLACK VARIABLES")
print("-" * 50)

n = len(X)  # Number of samples
K = 3       # Number of classes

# For each sample, we need slack variables for all incorrect classes
slack_vars_per_sample = K - 1
total_slack_vars = n * (K - 1)

print(f"Number of samples (n): {n}")
print(f"Number of classes (K): {K}")
print(f"Slack variables per sample: {slack_vars_per_sample}")
print(f"Total slack variables needed: {total_slack_vars}")

# Create a visualization of slack variable structure
fig, ax = plt.subplots(figsize=(12, 8))

# Create a matrix showing which slack variables are needed
slack_matrix = np.zeros((n, K))
for i in range(n):
    true_class = y[i]
    for k in range(K):
        if k != true_class:
            slack_matrix[i, k] = 1

# Plot the slack variable structure
im = ax.imshow(slack_matrix, cmap='Blues', aspect='auto')
ax.set_xlabel('Class Index')
ax.set_ylabel('Sample Index')
ax.set_title('Slack Variable Structure\n(1 = slack variable needed, 0 = no slack needed)')

# Add text annotations
for i in range(n):
    for k in range(K):
        if slack_matrix[i, k] == 1:
            ax.text(k, i, f'$\\xi_{{{i}{k}}}$', ha='center', va='center', fontsize=8)
        else:
            ax.text(k, i, '0', ha='center', va='center', fontsize=8)

plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'slack_variable_structure.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PART 3: COMPARISON WITH OVR AND OVO
# ============================================================================

print("\n3. COMPARISON WITH OVR AND OVO APPROACHES")
print("-" * 50)

# Calculate complexities for different approaches
n = len(X)
K = 3

# Direct multi-class SVM
direct_constraints = n * (K - 1)
direct_slack_vars = n * (K - 1)

# One-vs-Rest (OvR)
ovr_constraints = n * K
ovr_slack_vars = n * K

# One-vs-One (OvO)
ovo_constraints = n * (K * (K - 1) // 2)
ovo_slack_vars = n * (K * (K - 1) // 2)

print(f"Direct Multi-class SVM:")
print(f"  Constraints: {direct_constraints}")
print(f"  Slack variables: {direct_slack_vars}")
print()

print(f"One-vs-Rest (OvR):")
print(f"  Constraints: {ovr_constraints}")
print(f"  Slack variables: {ovr_slack_vars}")
print()

print(f"One-vs-One (OvO):")
print(f"  Constraints: {ovo_constraints}")
print(f"  Slack variables: {ovo_slack_vars}")
print()

# Create comparison visualization
methods = ['Direct', 'OvR', 'OvO']
constraints = [direct_constraints, ovr_constraints, ovo_constraints]
slack_vars = [direct_slack_vars, ovr_slack_vars, ovo_slack_vars]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot constraints comparison
bars1 = ax1.bar(methods, constraints, color=['skyblue', 'lightcoral', 'lightgreen'])
ax1.set_ylabel('Number of Constraints')
ax1.set_title('Constraint Complexity Comparison')
ax1.set_ylim(0, max(constraints) * 1.1)

# Add value labels on bars
for bar, value in zip(bars1, constraints):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             str(value), ha='center', va='bottom')

# Plot slack variables comparison
bars2 = ax2.bar(methods, slack_vars, color=['skyblue', 'lightcoral', 'lightgreen'])
ax2.set_ylabel('Number of Slack Variables')
ax2.set_title('Slack Variable Complexity Comparison')
ax2.set_ylim(0, max(slack_vars) * 1.1)

# Add value labels on bars
for bar, value in zip(bars2, slack_vars):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             str(value), ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'complexity_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PART 4: DUAL FORMULATION DERIVATION
# ============================================================================

print("\n4. DUAL FORMULATION DERIVATION")
print("-" * 50)

print("Primal formulation:")
print("min_{W,\\xi} (1/2)||W||_F^2 + C * Σ_{i=1}^n Σ_{k≠y_i} \\xi_{ik}")
print("subject to: w_{y_i}^T * x_i - w_k^T * x_i ≥ 1 - \\xi_{ik}, \\xi_{ik} ≥ 0")
print()

print("Step-by-step dual derivation:")
print("1. Lagrangian function:")
print("   L(W,\\xi,\\alpha,\\beta) = (1/2)||W||_F^2 + C * Σ_{i=1}^n Σ_{k≠y_i} \\xi_{ik}")
print("                - Σ_{i=1}^n Σ_{k≠y_i} \\alpha_{ik}(w_{y_i}^T * x_i - w_k^T * x_i - 1 + \\xi_{ik})")
print("                - Σ_{i=1}^n Σ_{k≠y_i} \\beta_{ik} * \\xi_{ik}")
print()

print("2. Stationarity conditions:")
print("   ∂L/∂W = 0 → W = Σ_{i=1}^n Σ_{k≠y_i} \\alpha_{ik}(x_i * e_{y_i}^T - x_i * e_k^T)")
print("   ∂L/∂\\xi_{ik} = 0 → C = \\alpha_{ik} + \\beta_{ik}")
print()

print("3. Dual formulation:")
print("   max_{\\alpha} Σ_{i=1}^n Σ_{k≠y_i} \\alpha_{ik}")
print("   - (1/2) Σ_{i,j=1}^n Σ_{k≠y_i} Σ_{l≠y_j} \\alpha_{ik} * \\alpha_{jl} * K(x_i, x_j) * (\\delta_{y_i,y_j} - \\delta_{y_i,l} - \\delta_{k,y_j} + \\delta_{k,l})")
print("   subject to: 0 ≤ \\alpha_{ik} ≤ C")
print()

# ============================================================================
# PART 5: ADVANTAGES AND DISADVANTAGES
# ============================================================================

print("\n5. ADVANTAGES AND DISADVANTAGES")
print("-" * 50)

print("ADVANTAGES of Direct Multi-class SVM:")
print("1. Single optimization problem - no need to combine multiple classifiers")
print("2. Directly optimizes the multi-class objective")
print("3. Guaranteed convergence to global optimum")
print("4. No ambiguity in decision making")
print("5. Theoretical guarantees for multi-class generalization")
print()

print("DISADVANTAGES of Direct Multi-class SVM:")
print("1. Higher computational complexity - O(nK) constraints vs O(n) for binary")
print("2. More memory requirements due to larger constraint matrix")
print("3. Slower training time compared to decomposition methods")
print("4. Less scalable to large datasets")
print("5. More complex implementation")
print()

# ============================================================================
# PART 6: PRACTICAL IMPLEMENTATION AND VISUALIZATION
# ============================================================================

print("\n6. PRACTICAL IMPLEMENTATION AND VISUALIZATION")
print("-" * 50)

# Generate synthetic data for visualization
np.random.seed(42)
X_viz, y_viz = make_blobs(n_samples=150, centers=3, cluster_std=1.2, random_state=42)
X_viz = StandardScaler().fit_transform(X_viz)

# Train different multi-class SVM approaches
print("Training different SVM approaches...")

# Direct multi-class SVM (using sklearn's SVC with 'ovr' decision function)
svm_direct = SVC(kernel='linear', C=1.0, decision_function_shape='ovr')
svm_direct.fit(X_viz, y_viz)

# One-vs-Rest approach
svm_ovr = SVC(kernel='linear', C=1.0, decision_function_shape='ovr')
svm_ovr.fit(X_viz, y_viz)

# One-vs-One approach
svm_ovo = SVC(kernel='linear', C=1.0, decision_function_shape='ovo')
svm_ovo.fit(X_viz, y_viz)

print("Training completed!")

# Create decision boundary visualization
def plot_decision_boundaries(X, y, classifiers, titles, filename):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    colors = ['red', 'blue', 'green']
    
    for idx, (clf, title) in enumerate(zip(classifiers, titles)):
        ax = axes[idx]
        
        # Get predictions for mesh points
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        ax.contourf(xx, yy, Z, alpha=0.3, colors=colors)
        ax.contour(xx, yy, Z, colors='black', linewidths=0.5)
        
        # Plot data points
        for i, color in enumerate(colors):
            mask = y == i
            ax.scatter(X[mask, 0], X[mask, 1], c=color, s=30, alpha=0.7, 
                      edgecolors='black', linewidth=0.5, label=f'Class {i}')
        
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# Plot decision boundaries
classifiers = [svm_direct, svm_ovr, svm_ovo]
titles = ['Direct Multi-class SVM', 'One-vs-Rest (OvR)', 'One-vs-One (OvO)']
plot_decision_boundaries(X_viz, y_viz, classifiers, titles, 'decision_boundaries_comparison.png')

# ============================================================================
# PART 7: COMPLEXITY ANALYSIS VISUALIZATION
# ============================================================================

print("\n7. COMPLEXITY ANALYSIS VISUALIZATION")
print("-" * 50)

# Create complexity analysis for different numbers of classes
K_values = np.arange(2, 11)
n = 100  # Fixed number of samples

direct_constraints = n * (K_values - 1)
ovr_constraints = n * K_values
ovo_constraints = n * (K_values * (K_values - 1) // 2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot constraints vs number of classes
ax1.plot(K_values, direct_constraints, 'o-', label='Direct Multi-class', linewidth=2, markersize=8)
ax1.plot(K_values, ovr_constraints, 's-', label='One-vs-Rest (OvR)', linewidth=2, markersize=8)
ax1.plot(K_values, ovo_constraints, '^-', label='One-vs-One (OvO)', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Classes (K)')
ax1.set_ylabel('Number of Constraints')
ax1.set_title('Constraint Complexity vs Number of Classes')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot computational complexity (big O notation)
ax2.plot(K_values, K_values - 1, 'o-', label='Direct Multi-class: O(K)', linewidth=2, markersize=8)
ax2.plot(K_values, K_values, 's-', label='One-vs-Rest: O(K)', linewidth=2, markersize=8)
ax2.plot(K_values, K_values * (K_values - 1) / 2, '^-', label='One-vs-One: O(K²)', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Classes (K)')
ax2.set_ylabel('Computational Complexity')
ax2.set_title('Computational Complexity vs Number of Classes')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'complexity_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PART 8: SUMMARY AND CONCLUSIONS
# ============================================================================

print("\n8. SUMMARY AND CONCLUSIONS")
print("-" * 50)

print("Key Findings:")
print(f"1. Direct multi-class SVM requires {direct_constraints[2]} constraints for {K_values[2]} classes")
print(f"2. OvR requires {ovr_constraints[2]} constraints for {K_values[2]} classes")
print(f"3. OvO requires {ovo_constraints[2]} constraints for {K_values[2]} classes")
print()

print("Recommendations:")
print("1. Use Direct Multi-class SVM for small to medium datasets with few classes")
print("2. Use OvR for large datasets or when computational efficiency is crucial")
print("3. Use OvO when you need more precise pairwise comparisons")
print("4. Consider the trade-off between accuracy and computational cost")
print()

print(f"All visualizations saved to: {save_dir}")
print("=" * 80)
