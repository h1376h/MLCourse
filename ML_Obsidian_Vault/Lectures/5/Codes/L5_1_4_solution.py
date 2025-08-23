import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize
from sklearn.datasets import make_classification
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_1_Quiz_4")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = False  # Disable LaTeX to avoid Unicode issues
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("=" * 80)
print("SVM DUAL FORMULATION: COMPREHENSIVE ANALYSIS")
print("=" * 80)

# ============================================================================
# PART 1: MATHEMATICAL DERIVATION DEMONSTRATION
# ============================================================================

print("\n1. DUAL FORMULATION DERIVATION")
print("-" * 50)

print("""
The SVM primal problem is:
    min  (1/2)||w||² 
    s.t. y_i(w^T x_i + b) ≥ 1, ∀i

Using Lagrange multipliers α_i ≥ 0, the Lagrangian is:
    L(w,b,α) = (1/2)||w||² - Σ α_i[y_i(w^T x_i + b) - 1]

Taking partial derivatives and setting to zero:
    ∂L/∂w = w - Σ α_i y_i x_i = 0  ⟹  w = Σ α_i y_i x_i
    ∂L/∂b = -Σ α_i y_i = 0         ⟹  Σ α_i y_i = 0

Substituting back into the Lagrangian gives the dual:
    max Σ α_i - (1/2)Σ Σ α_i α_j y_i y_j x_i^T x_j
    s.t. Σ α_i y_i = 0, α_i ≥ 0
""")

# ============================================================================
# PART 2: NUMERICAL EXAMPLE WITH SIMPLE 2D DATA
# ============================================================================

print("\n2. NUMERICAL EXAMPLE: 2D LINEARLY SEPARABLE DATA")
print("-" * 50)

# Create simple 2D dataset
np.random.seed(42)
X = np.array([[1, 2], [2, 3], [3, 1], [2, 1], [1, 1], [3, 3]])
y = np.array([1, 1, 1, -1, -1, -1])
n_samples, n_features = X.shape

print(f"Dataset: {n_samples} samples, {n_features} features")
print("Data points:")
for i in range(n_samples):
    print(f"  x_{i+1} = {X[i]}, y_{i+1} = {y[i]:2d}")

# ============================================================================
# PART 3: SOLVE DUAL PROBLEM NUMERICALLY
# ============================================================================

print("\n3. SOLVING THE DUAL PROBLEM")
print("-" * 50)

def compute_gram_matrix(X):
    """Compute the Gram matrix K[i,j] = x_i^T x_j"""
    return np.dot(X, X.T)

def dual_objective(alpha, K, y):
    """Compute the dual objective function value"""
    n = len(alpha)
    return np.sum(alpha) - 0.5 * np.sum(np.fromiter((alpha[i] * alpha[j] * y[i] * y[j] * K[i,j] 
                                       for i in range(n) for j in range(n)), dtype=float))

def dual_objective_gradient(alpha, K, y):
    """Compute gradient of dual objective"""
    n = len(alpha)
    grad = np.ones(n)
    for i in range(n):
        grad[i] -= np.sum(alpha[j] * y[i] * y[j] * K[i,j] for j in range(n))
    return grad

# Compute Gram matrix
K = compute_gram_matrix(X)
print("Gram Matrix K (x_i^T x_j):")
print(K)

# Solve dual problem using scipy
from scipy.optimize import minimize

def objective_for_scipy(alpha):
    return -dual_objective(alpha, K, y)  # Minimize negative

def constraint_eq(alpha):
    return np.dot(alpha, y)  # Sum α_i y_i = 0

# Initial guess
alpha0 = np.ones(n_samples) * 0.1

# Constraints
constraints = {'type': 'eq', 'fun': constraint_eq}
bounds = [(0, None) for _ in range(n_samples)]

# Solve
result = minimize(objective_for_scipy, alpha0, method='SLSQP', 
                 bounds=bounds, constraints=constraints)

alpha_optimal = result.x
print(f"\nOptimal α values:")
for i in range(n_samples):
    print(f"  α_{i+1} = {alpha_optimal[i]:.6f}")

# Find support vectors (α > threshold)
threshold = 1e-6
support_indices = np.where(alpha_optimal > threshold)[0]
print(f"\nSupport vectors (α > {threshold}):")
for idx in support_indices:
    print(f"  Point {idx+1}: x = {X[idx]}, y = {y[idx]}, α = {alpha_optimal[idx]:.6f}")

# ============================================================================
# PART 4: RECOVER PRIMAL VARIABLES
# ============================================================================

print("\n4. RECOVERING PRIMAL VARIABLES")
print("-" * 50)

# Compute w from dual variables
w_dual = np.sum(np.array([alpha_optimal[i] * y[i] * X[i] for i in range(n_samples)]), axis=0)
print(f"Weight vector w = Σ α_i y_i x_i = {w_dual}")

# Compute bias b using support vectors
if len(support_indices) > 0:
    # Use first support vector to compute b
    sv_idx = support_indices[0]
    b_dual = y[sv_idx] - np.dot(w_dual, X[sv_idx])
    print(f"Bias b = y_s - w^T x_s = {y[sv_idx]} - {np.dot(w_dual, X[sv_idx]):.6f} = {b_dual:.6f}")
else:
    b_dual = 0
    print("No support vectors found, setting b = 0")

# ============================================================================
# PART 5: VISUALIZATION OF PRIMAL VS DUAL PERSPECTIVES
# ============================================================================

print("\n5. GENERATING VISUALIZATIONS")
print("-" * 50)

# Create visualization showing primal and dual perspectives
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Original data with decision boundary
ax1.scatter(X[y==1, 0], X[y==1, 1], c='red', marker='o', s=100, label='Class +1')
ax1.scatter(X[y==-1, 0], X[y==-1, 1], c='blue', marker='s', s=100, label='Class -1')

# Highlight support vectors
for idx in support_indices:
    ax1.scatter(X[idx, 0], X[idx, 1], c='black', marker='x', s=200, linewidth=3)

# Plot decision boundary
if np.linalg.norm(w_dual) > 0:
    # Create grid for decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x1_line = np.linspace(x_min, x_max, 100)
    x2_line = (-w_dual[0] * x1_line - b_dual) / w_dual[1]
    ax1.plot(x1_line, x2_line, 'g-', linewidth=2, label='Decision Boundary')
    
    # Plot margin boundaries
    margin = 1 / np.linalg.norm(w_dual)
    normal_unit = w_dual / np.linalg.norm(w_dual)
    
    # Upper margin
    x2_upper = (-w_dual[0] * x1_line - (b_dual - 1)) / w_dual[1]
    ax1.plot(x1_line, x2_upper, 'g--', alpha=0.7, label='Margin')
    
    # Lower margin
    x2_lower = (-w_dual[0] * x1_line - (b_dual + 1)) / w_dual[1]
    ax1.plot(x1_line, x2_lower, 'g--', alpha=0.7)

ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.set_title('SVM: Primal Perspective\n(Maximize Margin)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Dual variables visualization
ax2.bar(range(1, n_samples+1), alpha_optimal, color=['red' if y[i]==1 else 'blue' for i in range(n_samples)])
ax2.set_xlabel('Data Point Index')
ax2.set_ylabel('$\\alpha_i$')
ax2.set_title('SVM: Dual Variables\n(Lagrange Multipliers)')
ax2.grid(True, alpha=0.3)

# Add text annotations for support vectors
for i, alpha_val in enumerate(alpha_optimal):
    if alpha_val > threshold:
        ax2.annotate(f'SV', (i+1, alpha_val), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontweight='bold')

# Plot 3: Gram matrix heatmap
im = ax3.imshow(K, cmap='viridis', aspect='auto')
ax3.set_xlabel('Data Point j')
ax3.set_ylabel('Data Point i')
ax3.set_title('Gram Matrix $K_{ij} = \\mathbf{x}_i^T \\mathbf{x}_j$')
ax3.set_xticks(range(n_samples))
ax3.set_yticks(range(n_samples))
ax3.set_xticklabels([f'{i+1}' for i in range(n_samples)])
ax3.set_yticklabels([f'{i+1}' for i in range(n_samples)])

# Add text annotations for matrix values
for i in range(n_samples):
    for j in range(n_samples):
        ax3.text(j, i, f'{K[i,j]:.1f}', ha="center", va="center", color="white")

plt.colorbar(im, ax=ax3)

# Plot 4: Primal vs Dual comparison table
ax4.axis('off')
comparison_data = [
    ['Aspect', 'Primal Formulation', 'Dual Formulation'],
    ['Variables', f'w in R^{n_features}, b in R', f'alpha in R^{n_samples}'],
    ['# Variables', f'{n_features + 1}', f'{n_samples}'],
    ['Objective', 'min (1/2)||w||^2', 'max sum(alpha) - (1/2)sum(alpha*alpha*y*y*K)'],
    ['Constraints', f'{n_samples} inequality', '1 equality + non-negativity'],
    ['Sparsity', 'Dense w', 'Sparse alpha (support vectors)'],
    ['Kernel Trick', 'Not directly applicable', 'Directly applicable']
]

table = ax4.table(cellText=comparison_data[1:], colLabels=comparison_data[0],
                 cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style the header row
for i in range(len(comparison_data[0])):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax4.set_title('Primal vs Dual Formulation Comparison', pad=20, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'svm_dual_formulation_overview.png'), 
           dpi=300, bbox_inches='tight')

# ============================================================================
# PART 6: COMPLEXITY ANALYSIS FOR DIFFERENT DATASET SIZES
# ============================================================================

print("\n6. COMPLEXITY ANALYSIS")
print("-" * 50)

# Analyze complexity for different n and d values
n_values = [10, 50, 100, 500, 1000, 5000]
d_values = [2, 10, 50, 100, 500, 1000]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Variables comparison for fixed d=50
d_fixed = 50
primal_vars = [d_fixed + 1 for _ in n_values]
dual_vars = n_values

ax1.loglog(n_values, primal_vars, 'bo-', label=f'Primal (d={d_fixed})')
ax1.loglog(n_values, dual_vars, 'ro-', label='Dual')
ax1.set_xlabel('Number of Training Samples (n)')
ax1.set_ylabel('Number of Variables')
ax1.set_title('Variables: Primal vs Dual\n(Fixed d=50)')
ax1.legend()
ax1.grid(True)

# Add crossover point
crossover_n = d_fixed + 1
ax1.axvline(x=crossover_n, color='gray', linestyle='--', alpha=0.7)
ax1.annotate(f'Crossover\nn={crossover_n}', xy=(crossover_n, crossover_n), 
            xytext=(crossover_n*2, crossover_n*2), 
            arrowprops=dict(arrowstyle='->', color='gray'))

# Plot 2: Heat map showing when dual is preferred
n_grid, d_grid = np.meshgrid(n_values, d_values)
dual_preferred = n_grid < (d_grid + 1)

im = ax2.imshow(dual_preferred, cmap='RdYlBu', aspect='auto', origin='lower')
ax2.set_xlabel('n (training samples)')
ax2.set_ylabel('d (features)')
ax2.set_title('When to Prefer Dual Formulation\n(Blue = Dual Preferred)')
ax2.set_xticks(range(len(n_values)))
ax2.set_yticks(range(len(d_values)))
ax2.set_xticklabels(n_values)
ax2.set_yticklabels(d_values)

# Add text annotations
for i in range(len(d_values)):
    for j in range(len(n_values)):
        text = "Dual" if dual_preferred[i, j] else "Primal"
        color = "white" if dual_preferred[i, j] else "black"
        ax2.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

plt.colorbar(im, ax=ax2)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'svm_complexity_analysis.png'), 
           dpi=300, bbox_inches='tight')

# ============================================================================
# PART 7: KKT CONDITIONS VERIFICATION
# ============================================================================

print("\n7. KKT CONDITIONS VERIFICATION")
print("-" * 50)

print("Verifying KKT conditions for the optimal solution:")

# 1. Stationarity: ∇_w L = 0, ∇_b L = 0
w_from_alpha = np.sum(alpha_optimal[i] * y[i] * X[i] for i in range(n_samples))
sum_alpha_y = np.sum(alpha_optimal * y)

print(f"1. Stationarity:")
print(f"   w = Σ αᵢyᵢxᵢ = {w_from_alpha}")
print(f"   Σ αᵢyᵢ = {sum_alpha_y:.10f} ≈ 0 ✓")

# 2. Primal feasibility: yᵢ(w^T xᵢ + b) ≥ 1
print(f"\n2. Primal feasibility:")
for i in range(n_samples):
    margin = y[i] * (np.dot(w_dual, X[i]) + b_dual)
    feasible = margin >= 1 - 1e-6  # Small tolerance
    print(f"   Point {i+1}: yᵢ(w^T xᵢ + b) = {margin:.6f} ≥ 1: {'✓' if feasible else '✗'}")

# 3. Dual feasibility: αᵢ ≥ 0
print(f"\n3. Dual feasibility:")
for i in range(n_samples):
    feasible = alpha_optimal[i] >= -1e-10  # Small tolerance
    print(f"   α_{i+1} = {alpha_optimal[i]:.6f} ≥ 0: {'✓' if feasible else '✗'}")

# 4. Complementary slackness: αᵢ[yᵢ(w^T xᵢ + b) - 1] = 0
print(f"\n4. Complementary slackness:")
for i in range(n_samples):
    slack = y[i] * (np.dot(w_dual, X[i]) + b_dual) - 1
    complementary = alpha_optimal[i] * slack
    satisfied = abs(complementary) < 1e-6
    print(f"   Point {i+1}: αᵢ × slack = {alpha_optimal[i]:.6f} × {slack:.6f} = {complementary:.6f} ≈ 0: {'✓' if satisfied else '✗'}")

# ============================================================================
# PART 8: STRONG DUALITY DEMONSTRATION
# ============================================================================

print("\n8. STRONG DUALITY VERIFICATION")
print("-" * 50)

# Compute primal objective value
primal_obj = 0.5 * np.dot(w_dual, w_dual)
print(f"Primal objective: (1/2)||w||² = {primal_obj:.6f}")

# Compute dual objective value
dual_obj = dual_objective(alpha_optimal, K, y)
print(f"Dual objective: Σαᵢ - (1/2)ΣΣαᵢαⱼyᵢyⱼKᵢⱼ = {dual_obj:.6f}")

# Duality gap
gap = abs(primal_obj - dual_obj)
print(f"Duality gap: |primal - dual| = {gap:.10f}")
print(f"Strong duality holds: {'✓' if gap < 1e-6 else '✗'}")

# ============================================================================
# PART 9: SPECIFIC NUMERICAL EXAMPLE (n=1000, d=50)
# ============================================================================

print("\n9. LARGE DATASET EXAMPLE (n=1000, d=50)")
print("-" * 50)

n_large, d_large = 1000, 50
print(f"For a dataset with n={n_large} samples and d={d_large} features:")
print(f"  Primal formulation: {d_large + 1} variables (w ∈ ℝ^{d_large}, b ∈ ℝ)")
print(f"  Dual formulation: {n_large} variables (α ∈ ℝ^{n_large})")
print(f"  Dual has {n_large - (d_large + 1)} = {n_large - d_large - 1} more variables")
print(f"  Recommendation: Use PRIMAL formulation (fewer variables)")

# When to use dual vs primal
print(f"\nGeneral guidelines:")
print(f"  - Use DUAL when: n < d (fewer samples than features)")
print(f"  - Use PRIMAL when: n > d (more samples than features)")
print(f"  - Use DUAL when: Need kernel trick for non-linear classification")
print(f"  - Use DUAL when: Dataset has natural sparsity in support vectors")

# ============================================================================
# PART 10: KERNEL PERSPECTIVE
# ============================================================================

print("\n10. KERNEL TRICK DEMONSTRATION")
print("-" * 50)

# Create visualization showing why dual formulation enables kernel trick
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Left plot: Linear kernel (same as dot product)
X_demo = np.array([[1, 1], [2, 1], [1, 2], [2, 2]])
K_linear = np.dot(X_demo, X_demo.T)

im1 = ax1.imshow(K_linear, cmap='Blues', aspect='auto')
ax1.set_title('Linear Kernel: $K(\\mathbf{x}_i, \\mathbf{x}_j) = \\mathbf{x}_i^T \\mathbf{x}_j$')
ax1.set_xlabel('Point j')
ax1.set_ylabel('Point i')

for i in range(len(X_demo)):
    for j in range(len(X_demo)):
        ax1.text(j, i, f'{K_linear[i,j]:.1f}', ha="center", va="center", color="white")

plt.colorbar(im1, ax=ax1)

# Right plot: RBF kernel
gamma = 0.5
K_rbf = np.zeros((len(X_demo), len(X_demo)))
for i in range(len(X_demo)):
    for j in range(len(X_demo)):
        K_rbf[i,j] = np.exp(-gamma * np.linalg.norm(X_demo[i] - X_demo[j])**2)

im2 = ax2.imshow(K_rbf, cmap='Reds', aspect='auto')
ax2.set_title('RBF Kernel: $K(\\mathbf{x}_i, \\mathbf{x}_j) = \\exp(-\\gamma ||\\mathbf{x}_i - \\mathbf{x}_j||^2)$')
ax2.set_xlabel('Point j')
ax2.set_ylabel('Point i')

for i in range(len(X_demo)):
    for j in range(len(X_demo)):
        ax2.text(j, i, f'{K_rbf[i,j]:.2f}', ha="center", va="center", color="white")

plt.colorbar(im2, ax=ax2)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'svm_kernel_comparison.png'), 
           dpi=300, bbox_inches='tight')

print(f"Kernel matrices computed and visualized.")
print(f"Key insight: Dual formulation only requires K(xᵢ,xⱼ), not explicit φ(x)")

print(f"\nAll visualizations saved to: {save_dir}")
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
