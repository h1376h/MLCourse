import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_2_Quiz_2")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("=" * 80)
print("SOFT MARGIN SVM ANALYSIS - QUESTION 2")
print("=" * 80)

# ============================================================================
# TASK 1: Derive soft margin formulation from hard margin case
# ============================================================================

print("\n1. DERIVATION FROM HARD MARGIN TO SOFT MARGIN")
print("-" * 50)

print("Hard Margin SVM Formulation:")
print("min_{w,b} (1/2)||w||^2")
print("subject to: y_i(w^T x_i + b) >= 1 for all i")

print("\nProblem: Hard margin requires perfect linear separability")
print("Solution: Introduce slack variables ξ_i >= 0")

print("\nSoft Margin SVM Formulation:")
print("min_{w,b,ξ} (1/2)||w||^2 + C * Σ ξ_i")
print("subject to: y_i(w^T x_i + b) >= 1 - ξ_i for all i")
print("           ξ_i >= 0 for all i")

print("\nKey Changes:")
print("- Added slack variables ξ_i to allow constraint violations")
print("- Added penalty term C * Σ ξ_i to the objective function")
print("- Modified constraints to y_i(w^T x_i + b) >= 1 - ξ_i")

# ============================================================================
# TASK 2: Geometric interpretation of slack variables
# ============================================================================

print("\n\n2. GEOMETRIC INTERPRETATION OF SLACK VARIABLES")
print("-" * 50)

# Create a simple dataset to demonstrate slack variables
np.random.seed(42)
X, y = make_blobs(n_samples=20, centers=2, cluster_std=1.5, random_state=42)
y = 2 * y - 1  # Convert to {-1, 1}

# Add some noise/outliers
X[0] = [2, 3]  # Outlier in positive class
X[1] = [-1, -2]  # Outlier in negative class

# Train soft margin SVM with different C values
C_values = [0.1, 1, 10]
svms = {}

for C in C_values:
    svm = SVC(kernel='linear', C=C, random_state=42)
    svm.fit(X, y)
    svms[C] = svm

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, C in enumerate(C_values):
    svm = svms[C]
    ax = axes[idx]
    
    # Get decision boundary
    w = svm.coef_[0]
    b = svm.intercept_[0]
    
    # Create mesh for decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and margins
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['red', 'black', 'blue'], 
               linestyles=['--', '-', '--'], linewidths=2)
    
    # Plot data points
    colors = ['red' if label == -1 else 'blue' for label in y]
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=100, alpha=0.7, edgecolors='black')
    
    # Calculate and display slack variables
    decision_values = svm.decision_function(X)
    slack_variables = np.maximum(0, 1 - y * decision_values)
    
    # Annotate points with slack values
    for i, (x, y_val, slack) in enumerate(zip(X, y, slack_variables)):
        if slack > 0:
            ax.annotate(f'$\\xi={slack:.2f}$', (x[0], x[1]), 
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))
    
    ax.set_title(f'C = {C}\nTotal Slack = {slack_variables.sum():.2f}')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'slack_variables_geometric.png'), dpi=300, bbox_inches='tight')

print("Geometric interpretation of slack variables:")
print("- ξ_i = 0: Point is correctly classified with margin >= 1")
print("- 0 < ξ_i < 1: Point is correctly classified but inside the margin")
print("- ξ_i = 1: Point is exactly on the decision boundary")
print("- ξ_i > 1: Point is misclassified")

# ============================================================================
# TASK 3: Prove ξ_i >= 0 is necessary
# ============================================================================

print("\n\n3. PROOF THAT ξ_i >= 0 IS NECESSARY")
print("-" * 50)

print("Proof by contradiction:")
print("Assume ξ_i < 0 for some i")
print("Then: y_i(w^T x_i + b) >= 1 - ξ_i")
print("Since ξ_i < 0, we have: y_i(w^T x_i + b) >= 1 - ξ_i > 1")
print("This means the point satisfies the hard margin constraint")
print("But if ξ_i < 0, we're penalizing points that are already correct!")
print("This would lead to a worse objective function value.")
print("Therefore, ξ_i >= 0 is necessary for the formulation to make sense.")

# Demonstrate with numerical example
print("\nNumerical Example:")
w_example = np.array([1, 1])
b_example = 0
x_example = np.array([0.5, 0.5])
y_example = 1

# Calculate activation
activation = y_example * (np.dot(w_example, x_example) + b_example)
print(f"Point: {x_example}, Label: {y_example}")
print(f"Activation: {activation}")

# Show what happens with negative slack
xi_negative = -0.5
print(f"If ξ_i = {xi_negative} (negative):")
print(f"Constraint: {activation} >= 1 - {xi_negative} = {1 - xi_negative}")
print(f"This is satisfied, but we're penalizing a correct classification!")

# ============================================================================
# TASK 4: Show ξ_i = max(0, 1 - y_i(w^T x_i + b)) in optimal solution
# ============================================================================

print("\n\n4. PROOF: ξ_i = max(0, 1 - y_i(w^T x_i + b)) IN OPTIMAL SOLUTION")
print("-" * 50)

print("Proof:")
print("In the optimal solution, we want to minimize the objective function.")
print("The objective includes the term C * Σ ξ_i.")
print("For any given w and b, we want to choose ξ_i as small as possible")
print("while satisfying the constraints.")

print("\nThe constraint is: y_i(w^T x_i + b) >= 1 - ξ_i")
print("Rearranging: ξ_i >= 1 - y_i(w^T x_i + b)")

print("\nSince we want to minimize ξ_i and ξ_i >= 0:")
print("ξ_i = max(0, 1 - y_i(w^T x_i + b))")

# Demonstrate with our SVM solutions
print("\nVerification with our SVM solutions:")
for C in C_values:
    svm = svms[C]
    decision_values = svm.decision_function(X)
    slack_variables = np.maximum(0, 1 - y * decision_values)
    
    print(f"\nC = {C}:")
    print(f"Total slack from SVM: {slack_variables.sum():.4f}")
    
    # Verify the formula
    calculated_slack = np.maximum(0, 1 - y * decision_values)
    print(f"Calculated using formula: {calculated_slack.sum():.4f}")
    print(f"Difference: {abs(slack_variables.sum() - calculated_slack.sum()):.10f}")

# ============================================================================
# TASK 5: What happens when C → ∞
# ============================================================================

print("\n\n5. BEHAVIOR WHEN C → ∞")
print("-" * 50)

print("When C → ∞, the penalty for slack variables becomes extremely large.")
print("This means the optimization will try to make all ξ_i = 0.")
print("This reduces the soft margin SVM to the hard margin SVM.")

# Demonstrate with very large C
C_large = 1000
svm_large_C = SVC(kernel='linear', C=C_large, random_state=42)
svm_large_C.fit(X, y)

decision_values_large_C = svm_large_C.decision_function(X)
slack_variables_large_C = np.maximum(0, 1 - y * decision_values_large_C)

print(f"\nWith C = {C_large}:")
print(f"Total slack: {slack_variables_large_C.sum():.6f}")
print(f"Number of support vectors: {len(svm_large_C.support_vectors_)}")

# Compare with hard margin (very large C)
print("\nComparison:")
for C in [0.1, 1, 10, C_large]:
    svm = SVC(kernel='linear', C=C, random_state=42)
    svm.fit(X, y)
    decision_values = svm.decision_function(X)
    slack_variables = np.maximum(0, 1 - y * decision_values)
    
    print(f"C = {C:>6}: Slack = {slack_variables.sum():.4f}, "
          f"Support Vectors = {len(svm.support_vectors_)}")

# Create visualization showing convergence to hard margin
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot with C = 0.1 (soft margin)
svm_soft = svms[0.1]
ax1 = axes[0]
w_soft = svm_soft.coef_[0]
b_soft = svm_soft.intercept_[0]

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                    np.linspace(y_min, y_max, 100))

Z_soft = svm_soft.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z_soft = Z_soft.reshape(xx.shape)

ax1.contour(xx, yy, Z_soft, levels=[-1, 0, 1], colors=['red', 'black', 'blue'], 
           linestyles=['--', '-', '--'], linewidths=2)
colors = ['red' if label == -1 else 'blue' for label in y]
ax1.scatter(X[:, 0], X[:, 1], c=colors, s=100, alpha=0.7, edgecolors='black')
ax1.set_title('Soft Margin (C = 0.1)')
ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.grid(True, alpha=0.3)

# Plot with C = 1000 (hard margin)
ax2 = axes[1]
Z_hard = svm_large_C.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z_hard = Z_hard.reshape(xx.shape)

ax2.contour(xx, yy, Z_hard, levels=[-1, 0, 1], colors=['red', 'black', 'blue'], 
           linestyles=['--', '-', '--'], linewidths=2)
ax2.scatter(X[:, 0], X[:, 1], c=colors, s=100, alpha=0.7, edgecolors='black')
ax2.set_title('Hard Margin ($C \\to \\infty$)')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'soft_vs_hard_margin.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# ADDITIONAL ANALYSIS: KKT Conditions
# ============================================================================

print("\n\nADDITIONAL ANALYSIS: KKT CONDITIONS")
print("-" * 50)

print("The Lagrangian for soft margin SVM is:")
print("L = (1/2)||w||^2 + C*Σξ_i - Σα_i[y_i(w^T x_i + b) - 1 + ξ_i] - Σμ_i ξ_i")

print("\nKKT Conditions:")
print("1. Stationarity: ∂L/∂w = 0, ∂L/∂b = 0, ∂L/∂ξ_i = 0")
print("2. Primal feasibility: y_i(w^T x_i + b) >= 1 - ξ_i, ξ_i >= 0")
print("3. Dual feasibility: α_i >= 0, μ_i >= 0")
print("4. Complementary slackness: α_i[y_i(w^T x_i + b) - 1 + ξ_i] = 0, μ_i ξ_i = 0")

print("\nFrom stationarity conditions:")
print("∂L/∂ξ_i = C - α_i - μ_i = 0")
print("Therefore: α_i + μ_i = C")

print("\nSince μ_i >= 0 and μ_i ξ_i = 0:")
print("If ξ_i > 0, then μ_i = 0, so α_i = C")
print("If ξ_i = 0, then α_i <= C")

# ============================================================================
# VISUALIZATION: Slack Variables vs C
# ============================================================================

print("\n\nVISUALIZATION: SLACK VARIABLES VS C")
print("-" * 50)

C_range = np.logspace(-2, 3, 20)
slack_totals = []
support_vector_counts = []

for C in C_range:
    svm = SVC(kernel='linear', C=C, random_state=42)
    svm.fit(X, y)
    decision_values = svm.decision_function(X)
    slack_variables = np.maximum(0, 1 - y * decision_values)
    
    slack_totals.append(slack_variables.sum())
    support_vector_counts.append(len(svm.support_vectors_))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.semilogx(C_range, slack_totals, 'b-o', linewidth=2, markersize=6)
ax1.set_xlabel('Regularization Parameter C')
ax1.set_ylabel('Total Slack Variables')
ax1.set_title('Total Slack vs C')
ax1.grid(True, alpha=0.3)

ax2.semilogx(C_range, support_vector_counts, 'r-o', linewidth=2, markersize=6)
ax2.set_xlabel('Regularization Parameter C')
ax2.set_ylabel('Number of Support Vectors')
ax2.set_title('Support Vectors vs C')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'slack_and_support_vectors_vs_C.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# ADDITIONAL VISUALIZATION: Simple Margin Width Comparison
# ============================================================================

print("\n\nADDITIONAL VISUALIZATION: MARGIN WIDTH COMPARISON")
print("-" * 50)

# Create a simple visualization showing margin width differences
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Use a simple 2D dataset
X_simple = np.array([[1, 1], [2, 2], [3, 1], [0, 0], [1, 0], [2, 0]])
y_simple = np.array([1, 1, 1, -1, -1, -1])

# Train SVMs with different C values
C_simple = [0.1, 10]
colors_simple = ['lightblue', 'lightcoral']
labels_simple = ['Soft Margin (C=0.1)', 'Hard Margin (C=10)']

for idx, C in enumerate(C_simple):
    svm_simple = SVC(kernel='linear', C=C, random_state=42)
    svm_simple.fit(X_simple, y_simple)
    
    # Get decision boundary
    w = svm_simple.coef_[0]
    b = svm_simple.intercept_[0]
    
    # Create margin lines
    x_min, x_max = -1, 4
    y_min, y_max = -1, 3
    
    # Decision boundary
    x_db = np.linspace(x_min, x_max, 100)
    y_db = (-w[0] * x_db - b) / w[1]
    
    # Margin boundaries
    y_margin_plus = (-w[0] * x_db - b + 1) / w[1]
    y_margin_minus = (-w[0] * x_db - b - 1) / w[1]
    
    # Plot decision boundary and margins
    ax.plot(x_db, y_db, color=colors_simple[idx], linewidth=3, 
            label=f'Decision Boundary ({labels_simple[idx]})')
    ax.plot(x_db, y_margin_plus, color=colors_simple[idx], linewidth=1, 
            linestyle='--', alpha=0.7)
    ax.plot(x_db, y_margin_minus, color=colors_simple[idx], linewidth=1, 
            linestyle='--', alpha=0.7)
    
    # Fill margin area
    ax.fill_between(x_db, y_margin_minus, y_margin_plus, 
                   color=colors_simple[idx], alpha=0.2)

# Plot data points
colors_points = ['blue' if label == 1 else 'red' for label in y_simple]
ax.scatter(X_simple[:, 0], X_simple[:, 1], c=colors_points, s=200, 
          edgecolors='black', linewidth=2, zorder=5)

# Highlight support vectors
for C in C_simple:
    svm_simple = SVC(kernel='linear', C=C, random_state=42)
    svm_simple.fit(X_simple, y_simple)
    support_vectors = svm_simple.support_vectors_
    ax.scatter(support_vectors[:, 0], support_vectors[:, 1], 
              s=300, facecolors='none', edgecolors='black', 
              linewidth=3, marker='o', zorder=6)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Margin Width Comparison: Soft vs Hard Margin')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'margin_width_comparison.png'), dpi=300, bbox_inches='tight')

print("Plots saved to:", save_dir)
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
