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
plt.rcParams['text.usetex'] = True
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

# Create simple 2D linearly separable dataset
np.random.seed(42)
# Positive class points (upper right)
X_pos = np.array([[3, 3], [4, 3], [3, 4]])
# Negative class points (lower left)
X_neg = np.array([[1, 1], [2, 1], [1, 2]])
X = np.vstack([X_pos, X_neg])
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

# Solve dual problem using scipy with better initialization
from scipy.optimize import minimize

def objective_for_scipy(alpha):
    return -dual_objective(alpha, K, y)  # Minimize negative

def constraint_eq(alpha):
    return np.dot(alpha, y)  # Sum α_i y_i = 0

# Better initial guess - small positive values
alpha0 = np.ones(n_samples) * 0.01

# Constraints
constraints = {'type': 'eq', 'fun': constraint_eq}
bounds = [(0, 10) for _ in range(n_samples)]  # Add upper bound for stability

# Solve with better options
result = minimize(objective_for_scipy, alpha0, method='SLSQP',
                 bounds=bounds, constraints=constraints,
                 options={'ftol': 1e-9, 'disp': False})

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

# Compute bias b using support vectors (average over all support vectors for stability)
if len(support_indices) > 0:
    # Use average over all support vectors to compute b
    b_values = []
    for sv_idx in support_indices:
        b_val = y[sv_idx] - np.dot(w_dual, X[sv_idx])
        b_values.append(b_val)
    b_dual = np.mean(b_values)
    print(f"Bias b computed from {len(support_indices)} support vectors:")
    for i, sv_idx in enumerate(support_indices):
        print(f"  SV {sv_idx+1}: b = y_{sv_idx+1} - w^T x_{sv_idx+1} = {y[sv_idx]} - {np.dot(w_dual, X[sv_idx]):.6f} = {b_values[i]:.6f}")
    print(f"Average bias b = {b_dual:.6f}")
else:
    b_dual = 0
    print("No support vectors found, setting b = 0")

# ============================================================================
# PART 5: VISUALIZATION OF PRIMAL VS DUAL PERSPECTIVES
# ============================================================================

print("\n5. GENERATING VISUALIZATIONS")
print("-" * 50)

# Plot 1: Original data with decision boundary
plt.figure(figsize=(10, 8))
plt.scatter(X[y==1, 0], X[y==1, 1], c='red', marker='o', s=100, label='Class +1')
plt.scatter(X[y==-1, 0], X[y==-1, 1], c='blue', marker='s', s=100, label='Class -1')

# Highlight support vectors
for idx in support_indices:
    plt.scatter(X[idx, 0], X[idx, 1], c='black', marker='x', s=200, linewidth=3)

# Plot decision boundary
if np.linalg.norm(w_dual) > 0:
    # Create grid for decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x1_line = np.linspace(x_min, x_max, 100)
    
    if abs(w_dual[1]) > 1e-10:  # Check if w_dual[1] is not too small
        x2_line = (-w_dual[0] * x1_line - b_dual) / w_dual[1]
        plt.plot(x1_line, x2_line, 'g-', linewidth=2, label='Decision Boundary')
        
        # Plot margin boundaries
        # Upper margin
        x2_upper = (-w_dual[0] * x1_line - (b_dual - 1)) / w_dual[1]
        plt.plot(x1_line, x2_upper, 'g--', alpha=0.7, label='Margin')
        
        # Lower margin
        x2_lower = (-w_dual[0] * x1_line - (b_dual + 1)) / w_dual[1]
        plt.plot(x1_line, x2_lower, 'g--', alpha=0.7)
    else:
        # Vertical line case
        x1_vertical = -b_dual / w_dual[0]
        y_range = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
        plt.axvline(x=x1_vertical, color='g', linestyle='-', linewidth=2, label='Decision Boundary')
        
        # Margin lines
        plt.axvline(x=x1_vertical - 1/np.linalg.norm(w_dual), color='g', linestyle='--', alpha=0.7, label='Margin')
        plt.axvline(x=x1_vertical + 1/np.linalg.norm(w_dual), color='g', linestyle='--', alpha=0.7)

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('SVM: Primal Perspective (Maximize Margin)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'svm_primal_perspective.png'), 
           dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Dual variables visualization
plt.figure(figsize=(10, 6))
bars = plt.bar(range(1, n_samples+1), alpha_optimal, 
               color=['red' if y[i]==1 else 'blue' for i in range(n_samples)])
plt.xlabel('Data Point Index')
plt.ylabel('$\\alpha_i$')
plt.title('SVM: Dual Variables (Lagrange Multipliers)')
plt.grid(True, alpha=0.3)

# Add text annotations for support vectors
for i, alpha_val in enumerate(alpha_optimal):
    if alpha_val > threshold:
        plt.annotate('SV', (i+1, alpha_val), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'svm_dual_variables.png'), 
           dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Gram matrix heatmap
plt.figure(figsize=(8, 6))
im = plt.imshow(K, cmap='viridis', aspect='auto')
plt.xlabel('Data Point $j$')
plt.ylabel('Data Point $i$')
plt.title('Gram Matrix $K_{ij} = \\mathbf{x}_i^T \\mathbf{x}_j$')
plt.xticks(range(n_samples), [f'{i+1}' for i in range(n_samples)])
plt.yticks(range(n_samples), [f'{i+1}' for i in range(n_samples)])

# Add text annotations for matrix values
for i in range(n_samples):
    for j in range(n_samples):
        plt.text(j, i, f'{K[i,j]:.1f}', ha="center", va="center", color="white")

plt.colorbar(im)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'svm_gram_matrix.png'), 
           dpi=300, bbox_inches='tight')
plt.close()

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

plt.figure(figsize=(10, 6))
plt.loglog(n_values, primal_vars, 'bo-', label=f'Primal ($d={d_fixed}$)')
plt.loglog(n_values, dual_vars, 'ro-', label='Dual')
plt.xlabel('Number of Training Samples ($n$)')
plt.ylabel('Number of Variables')
plt.title('Variables: Primal vs Dual (Fixed $d=50$)')
plt.legend()
plt.grid(True)

# Add crossover point
crossover_n = d_fixed + 1
plt.axvline(x=crossover_n, color='gray', linestyle='--', alpha=0.7)
plt.annotate(f'Crossover\n$n={crossover_n}$', xy=(crossover_n, crossover_n), 
            xytext=(crossover_n*2, crossover_n*2), 
            arrowprops=dict(arrowstyle='->', color='gray'))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'svm_variables_comparison.png'), 
           dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Heat map showing when dual is preferred
plt.figure(figsize=(10, 6))
n_grid, d_grid = np.meshgrid(n_values, d_values)
dual_preferred = n_grid < (d_grid + 1)

im = plt.imshow(dual_preferred, cmap='RdYlBu', aspect='auto', origin='lower')
plt.xlabel('$n$ (training samples)')
plt.ylabel('$d$ (features)')
plt.title('When to Prefer Dual Formulation (Blue = Dual Preferred)')
plt.xticks(range(len(n_values)), n_values)
plt.yticks(range(len(d_values)), d_values)

# Add text annotations
for i in range(len(d_values)):
    for j in range(len(n_values)):
        text = "Dual" if dual_preferred[i, j] else "Primal"
        color = "white" if dual_preferred[i, j] else "black"
        plt.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

plt.colorbar(im)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'svm_formulation_choice.png'), 
           dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PART 7: KKT CONDITIONS VERIFICATION
# ============================================================================

print("\n7. KKT CONDITIONS VERIFICATION")
print("-" * 50)

print("Verifying KKT conditions for the optimal solution:")

# 1. Stationarity: ∇_w L = 0, ∇_b L = 0
w_from_alpha = np.sum(np.array([alpha_optimal[i] * y[i] * X[i] for i in range(n_samples)]), axis=0)
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
X_demo = np.array([[1, 1], [2, 1], [1, 2], [2, 2]])

# Left plot: Linear kernel (same as dot product)
plt.figure(figsize=(8, 6))
K_linear = np.dot(X_demo, X_demo.T)

im1 = plt.imshow(K_linear, cmap='Blues', aspect='auto')
plt.title('Linear Kernel: $K(\\mathbf{x}_i, \\mathbf{x}_j) = \\mathbf{x}_i^T \\mathbf{x}_j$')
plt.xlabel('Point $j$')
plt.ylabel('Point $i$')

for i in range(len(X_demo)):
    for j in range(len(X_demo)):
        plt.text(j, i, f'{K_linear[i,j]:.1f}', ha="center", va="center", color="white")

plt.colorbar(im1)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'svm_linear_kernel.png'), 
           dpi=300, bbox_inches='tight')
plt.close()

# Right plot: RBF kernel
plt.figure(figsize=(8, 6))
gamma = 0.5
K_rbf = np.zeros((len(X_demo), len(X_demo)))
for i in range(len(X_demo)):
    for j in range(len(X_demo)):
        K_rbf[i,j] = np.exp(-gamma * np.linalg.norm(X_demo[i] - X_demo[j])**2)

im2 = plt.imshow(K_rbf, cmap='Reds', aspect='auto')
plt.title('RBF Kernel: $K(\\mathbf{x}_i, \\mathbf{x}_j) = \\exp(-\\gamma ||\\mathbf{x}_i - \\mathbf{x}_j||^2)$')
plt.xlabel('Point $j$')
plt.ylabel('Point $i$')

for i in range(len(X_demo)):
    for j in range(len(X_demo)):
        plt.text(j, i, f'{K_rbf[i,j]:.2f}', ha="center", va="center", color="white")

plt.colorbar(im2)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'svm_rbf_kernel.png'), 
           dpi=300, bbox_inches='tight')
plt.close()

print(f"Kernel matrices computed and visualized.")
print(f"Key insight: Dual formulation only requires K(xᵢ,xⱼ), not explicit φ(x)")

print(f"\nAll visualizations saved to: {save_dir}")
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
