import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from cvxopt import matrix, solvers
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_2_Quiz_20")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("QUESTION 20: HARD VS SOFT MARGIN SVM COMPARISON")
print("=" * 80)

# Dataset with outlier
X_pos = np.array([[2, 2], [3, 3], [4, 1]])  # Class +1
X_neg = np.array([[0, 0], [1, 1], [2.5, 2.5]])  # Class -1 (with outlier)

# Combine data
X = np.vstack([X_pos, X_neg])
y = np.array([1, 1, 1, -1, -1, -1])

print("Dataset:")
print("Class +1:", X_pos)
print("Class -1:", X_neg)
print("Outlier point: (2.5, 2.5) - making data non-separable")

print("\n" + "="*50)
print("STEP 1: PROVING NO HARD MARGIN SOLUTION EXISTS")
print("="*50)

print("To prove no hard margin solution exists, we need to show that the data is not linearly separable.")
print("This means we cannot find w1, w2, b such that:")
print("  w1*x1 + w2*x2 + b ≥ 1  for all positive points")
print("  w1*x1 + w2*x2 + b ≤ -1 for all negative points")

print("\nLet's analyze the constraints:")
print("For positive points (y=1):")
for i, point in enumerate(X_pos):
    print(f"  w1*{point[0]} + w2*{point[1]} + b ≥ 1")
    print(f"  {point[0]}*w1 + {point[1]}*w2 + b ≥ 1")

print("\nFor negative points (y=-1):")
for i, point in enumerate(X_neg):
    print(f"  w1*{point[0]} + w2*{point[1]} + b ≤ -1")
    print(f"  {point[0]}*w1 + {point[1]}*w2 + b ≤ -1")

print("\nLet's check if the outlier point (2.5, 2.5) can be satisfied:")
print("The outlier is in class -1, so we need: 2.5*w1 + 2.5*w2 + b ≤ -1")

print("\nBut looking at the positive points:")
print("Point (2,2): 2*w1 + 2*w2 + b ≥ 1")
print("Point (3,3): 3*w1 + 3*w2 + b ≥ 1")
print("Point (4,1): 4*w1 + 1*w2 + b ≥ 1")

print("\nIf we try to satisfy the outlier constraint 2.5*w1 + 2.5*w2 + b ≤ -1,")
print("we would need a very negative b value. But then:")
print("For point (2,2): 2*w1 + 2*w2 + b ≥ 1")
print("For point (3,3): 3*w1 + 3*w2 + b ≥ 1")

print("\nThis creates a contradiction because:")
print("If 2.5*w1 + 2.5*w2 + b ≤ -1 (outlier constraint)")
print("And 2*w1 + 2*w2 + b ≥ 1 (positive point constraint)")
print("Then: 0.5*w1 + 0.5*w2 ≤ -2")
print("This would require w1 + w2 ≤ -4")

print("\nBut if w1 + w2 ≤ -4, then for point (3,3):")
print("3*w1 + 3*w2 + b = 3*(w1 + w2) + b ≤ 3*(-4) + b = -12 + b")
print("We need this to be ≥ 1, so b ≥ 13")

print("\nBut then for the outlier:")
print("2.5*w1 + 2.5*w2 + b = 2.5*(w1 + w2) + b ≤ 2.5*(-4) + 13 = -10 + 13 = 3")
print("We need this to be ≤ -1, which is impossible since 3 > -1")

print("\nTherefore, no hard margin solution exists!")
print("The data is not linearly separable due to the outlier point (2.5, 2.5).")

# Verification using solver (for confirmation only)
print("\n--- Verification using solver ---")
def solve_hard_margin_svm(X, y):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = np.dot(X[i], X[j])
    
    Q = np.outer(y, y) * K
    P = matrix(Q.astype(np.double))
    q = matrix(-np.ones(n_samples).astype(np.double))
    G = matrix(-np.eye(n_samples).astype(np.double))
    h = matrix(np.zeros(n_samples).astype(np.double))
    A = matrix(y.astype(np.double), (1, n_samples))
    b = matrix(0.0)
    
    try:
        sol = solvers.qp(P, q, G, h, A, b)
        if sol['status'] == 'optimal':
            alpha = np.array(sol['x']).flatten()
            return alpha, True
        else:
            return None, False
    except:
        return None, False

alpha_hard, success = solve_hard_margin_svm(X, y)
if not success or alpha_hard is None:
    print("✓ Solver confirms: Hard margin SVM FAILS - no solution exists!")
else:
    print("✗ Unexpected: Hard margin SVM succeeded")

print("\n" + "="*50)
print("STEP 2: SOLVING SOFT MARGIN SVM WITH C = 1")
print("="*50)

print("For soft margin SVM, we need to solve:")
print("minimize: (1/2)||w||² + C*Σξᵢ")
print("subject to: yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ for all i")
print("            ξᵢ ≥ 0 for all i")

print(f"\nWith C = 1, the objective becomes:")
print("minimize: (1/2)||w||² + Σξᵢ")

print("\nLet's solve this step by step using the dual formulation:")
print("The dual problem is:")
print("maximize: Σαᵢ - (1/2)Σᵢⱼ αᵢαⱼyᵢyⱼxᵢᵀxⱼ")
print("subject to: 0 ≤ αᵢ ≤ C for all i")
print("            Σαᵢyᵢ = 0")

print("\nLet's construct the kernel matrix Kᵢⱼ = xᵢᵀxⱼ:")
K = np.zeros((6, 6))
for i in range(6):
    for j in range(6):
        K[i, j] = np.dot(X[i], X[j])

print("Kernel matrix K:")
print(K)

print("\nThe dual objective becomes:")
print("maximize: Σαᵢ - (1/2)Σᵢⱼ αᵢαⱼyᵢyⱼKᵢⱼ")

print("\nWith constraints:")
print("0 ≤ αᵢ ≤ 1 for all i")
print("α₁ + α₂ + α₃ - α₄ - α₅ - α₆ = 0")

# Solve using solver for the actual solution
def solve_soft_margin_svm(X, y, C):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = np.dot(X[i], X[j])
    
    Q_extended = np.zeros((2*n_samples, 2*n_samples))
    Q_extended[:n_samples, :n_samples] = np.outer(y, y) * K
    Q_extended[n_samples:, n_samples:] = np.eye(n_samples) / C
    
    P = matrix(Q_extended.astype(np.double))
    q = matrix(np.concatenate([-np.ones(n_samples), np.zeros(n_samples)]).astype(np.double))
    
    G1 = -np.eye(2*n_samples)
    G2 = np.zeros((n_samples, 2*n_samples))
    G2[:, :n_samples] = np.eye(n_samples)
    G = matrix(np.vstack([G1, G2]).astype(np.double))
    
    h = matrix(np.concatenate([np.zeros(2*n_samples), C * np.ones(n_samples)]).astype(np.double))
    
    A = matrix(np.concatenate([y, np.zeros(n_samples)]).astype(np.double), (1, 2*n_samples))
    b = matrix(0.0)
    
    try:
        sol = solvers.qp(P, q, G, h, A, b)
        if sol['status'] == 'optimal':
            solution = np.array(sol['x']).flatten()
            alpha = solution[:n_samples]
            xi = solution[n_samples:]
            return alpha, xi, True
        else:
            return None, None, False
    except:
        return None, None, False

C = 1.0
print(f"\nSolving soft margin SVM with C = {C}...")
alpha_soft, xi_soft, success = solve_soft_margin_svm(X, y, C)

if success:
    print("✓ Soft margin SVM succeeded!")
    print(f"Alpha values: {alpha_soft}")
    print(f"Slack variables: {xi_soft}")
    
    # Calculate w and b from dual solution
    def get_svm_params(X, y, alpha):
        sv_indices = alpha > 1e-5
        sv_X = X[sv_indices]
        sv_y = y[sv_indices]
        sv_alpha = alpha[sv_indices]
        
        w = np.sum(sv_alpha[:, np.newaxis] * sv_y[:, np.newaxis] * sv_X, axis=0)
        b = np.mean(sv_y - np.dot(sv_X, w))
        
        return w, b, sv_indices
    
    w_soft, b_soft, sv_indices_soft = get_svm_params(X, y, alpha_soft)
    print(f"\nSoft margin solution:")
    print(f"w = {w_soft}")
    print(f"b = {b_soft:.3f}")
    print(f"Support vectors: {np.sum(sv_indices_soft)}")
    
    print(f"\nDecision boundary equation:")
    print(f"{w_soft[0]:.3f}*x1 + {w_soft[1]:.3f}*x2 + {b_soft:.3f} = 0")
    
    # Plot soft margin solution
    def plot_decision_boundary(X, y, w, b, title, filename, show_support_vectors=False, sv_indices=None):
        plt.figure(figsize=(10, 8))
        
        plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', marker='o', s=100, label='Class +1', edgecolors='black')
        plt.scatter(X[y == -1, 0], X[y == -1, 1], c='blue', marker='s', s=100, label='Class -1', edgecolors='black')
        
        outlier_idx = np.where((X == [2.5, 2.5]).all(axis=1))[0]
        if len(outlier_idx) > 0:
            plt.scatter(X[outlier_idx, 0], X[outlier_idx, 1], c='orange', marker='*', s=200, 
                       label='Outlier', edgecolors='black', linewidth=2)
        
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        
        Z = w[0] * xx + w[1] * yy + b
        Z = Z.reshape(xx.shape)
        
        plt.contour(xx, yy, Z, levels=[0], colors='green', linewidths=2, label='Decision Boundary')
        plt.contour(xx, yy, Z, levels=[-1, 1], colors='green', linestyles='--', alpha=0.5, label='Margins')
        
        plt.contourf(xx, yy, Z, levels=[-100, 0], colors=['lightblue'], alpha=0.3)
        plt.contourf(xx, yy, Z, levels=[0, 100], colors=['lightcoral'], alpha=0.3)
        
        if show_support_vectors and sv_indices is not None:
            sv_X = X[sv_indices]
            plt.scatter(sv_X[:, 0], sv_X[:, 1], c='yellow', marker='o', s=150, 
                       label='Support Vectors', edgecolors='black', linewidth=2)
        
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        eq_str = f'$w_1 x_1 + w_2 x_2 + b = 0$\n${w[0]:.3f} x_1 + {w[1]:.3f} x_2 + {b:.3f} = 0$'
        plt.annotate(eq_str, xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))
        
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    plot_decision_boundary(X, y, w_soft, b_soft, 
                          f'Soft Margin SVM (C = {C})', 
                          'soft_margin_svm.png', 
                          show_support_vectors=True, 
                          sv_indices=sv_indices_soft)
else:
    print("✗ Soft margin SVM failed")

print("\n" + "="*50)
print("STEP 3: CALCULATING SLACK VARIABLE FOR OUTLIER")
print("="*50)

if success:
    outlier_idx = np.where((X == [2.5, 2.5]).all(axis=1))[0][0]
    outlier_xi = xi_soft[outlier_idx]
    outlier_activation = np.dot(w_soft, X[outlier_idx]) + b_soft
    outlier_margin = y[outlier_idx] * outlier_activation
    
    print(f"Outlier point: {X[outlier_idx]}")
    print(f"True label: {y[outlier_idx]}")
    print(f"Activation: f(2.5, 2.5) = w1*2.5 + w2*2.5 + b")
    print(f"Activation: f(2.5, 2.5) = {w_soft[0]:.3f}*2.5 + {w_soft[1]:.3f}*2.5 + {b_soft:.3f}")
    print(f"Activation: f(2.5, 2.5) = {w_soft[0]*2.5:.3f} + {w_soft[1]*2.5:.3f} + {b_soft:.3f}")
    print(f"Activation: f(2.5, 2.5) = {outlier_activation:.3f}")
    
    print(f"\nMargin: y * f(x) = {y[outlier_idx]} * {outlier_activation:.3f} = {outlier_margin:.3f}")
    print(f"Slack variable: ξ = {outlier_xi:.3f}")
    
    print(f"\nInterpretation:")
    if outlier_xi > 1:
        print("✓ Outlier is MISCLASSIFIED (ξ > 1)")
    elif outlier_xi > 0:
        print("✓ Outlier is inside margin but correctly classified (0 < ξ ≤ 1)")
    else:
        print("✓ Outlier is correctly classified with margin ≥ 1 (ξ = 0)")
    
    print(f"\nThe slack variable ξ measures how much the margin constraint is violated.")
    print(f"For the outlier: ξ = max(0, 1 - y*f(x)) = max(0, 1 - {outlier_margin:.3f}) = {max(0, 1-outlier_margin):.3f}")

print("\n" + "="*50)
print("STEP 4: COMPARING WITH AND WITHOUT OUTLIER")
print("="*50)

# Dataset without outlier
X_clean = np.vstack([X_pos, X_neg[:-1]])
y_clean = np.array([1, 1, 1, -1, -1])

print("Dataset without outlier:")
print("Class +1:", X_pos)
print("Class -1:", X_neg[:-1])

print("\nNow the data is linearly separable. Let's solve the hard margin SVM:")
print("minimize: (1/2)||w||²")
print("subject to: yᵢ(wᵀxᵢ + b) ≥ 1 for all i")

print("\nThe dual problem becomes:")
print("maximize: Σαᵢ - (1/2)Σᵢⱼ αᵢαⱼyᵢyⱼxᵢᵀxⱼ")
print("subject to: αᵢ ≥ 0 for all i")
print("            Σαᵢyᵢ = 0")

# Solve hard margin SVM on clean data
alpha_clean, success_clean = solve_hard_margin_svm(X_clean, y_clean)

if success_clean:
    print("✓ Hard margin SVM succeeds on clean data!")
    
    w_clean, b_clean, sv_indices_clean = get_svm_params(X_clean, y_clean, alpha_clean)
    print(f"\nClean data solution:")
    print(f"w = {w_clean}")
    print(f"b = {b_clean:.3f}")
    print(f"Support vectors: {np.sum(sv_indices_clean)}")
    
    print(f"\nDecision boundary equation:")
    print(f"{w_clean[0]:.3f}*x1 + {w_clean[1]:.3f}*x2 + {b_clean:.3f} = 0")
    
    # Plot clean data solution
    plot_decision_boundary(X_clean, y_clean, w_clean, b_clean, 
                          'Hard Margin SVM (Clean Data)', 
                          'hard_margin_clean.png', 
                          show_support_vectors=True, 
                          sv_indices=sv_indices_clean)
    
    # Test how well clean solution works on data with outlier
    predictions_clean = np.sign(np.dot(X, w_clean) + b_clean)
    accuracy_clean = accuracy_score(y, predictions_clean)
    print(f"\nAccuracy of clean solution on full data: {accuracy_clean:.3f}")
    
    print(f"\nThe clean solution misclassifies the outlier point (2.5, 2.5).")
    print(f"This shows why the outlier makes the data non-separable.")
else:
    print("✗ Hard margin SVM still fails on clean data")

print("\n" + "="*50)
print("STEP 5: QUANTIFYING OUTLIER EFFECT")
print("="*50)

if success and success_clean:
    print("Comparison of solutions:")
    print(f"Clean data: w = {w_clean}, b = {b_clean:.3f}")
    print(f"With outlier: w = {w_soft}, b = {b_soft:.3f}")
    
    # Calculate differences
    w_diff = np.linalg.norm(w_soft - w_clean)
    b_diff = abs(b_soft - b_clean)
    
    print(f"\nQuantifying the effect:")
    print(f"Change in weight vector norm: ||w_soft - w_clean|| = {w_diff:.3f}")
    print(f"Change in bias: |b_soft - b_clean| = {b_diff:.3f}")
    
    # Calculate margin changes
    margin_clean = 2 / np.linalg.norm(w_clean)
    margin_soft = 2 / np.linalg.norm(w_soft)
    margin_change = margin_soft - margin_clean
    
    print(f"\nMargin comparison:")
    print(f"Margin with clean data: 2/||w_clean|| = 2/{np.linalg.norm(w_clean):.3f} = {margin_clean:.3f}")
    print(f"Margin with outlier: 2/||w_soft|| = 2/{np.linalg.norm(w_soft):.3f} = {margin_soft:.3f}")
    print(f"Margin change: {margin_change:.3f}")
    
    # Calculate total slack
    total_slack = np.sum(xi_soft)
    print(f"\nTotal slack variables: Σξᵢ = {total_slack:.3f}")
    
    print(f"\nInterpretation:")
    print(f"The outlier causes a significant change in the decision boundary.")
    print(f"The weight vector changes by {w_diff:.3f} in norm, and the bias changes by {b_diff:.3f}.")
    print(f"Interestingly, the soft margin solution achieves a wider margin ({margin_soft:.3f} vs {margin_clean:.3f}).")
    print(f"This is because the soft margin formulation allows for margin violations while optimizing the overall objective.")
    
    # Create comparison plot
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_clean[y_clean == 1, 0], X_clean[y_clean == 1, 1], c='red', marker='o', s=100, label='Class +1')
    plt.scatter(X_clean[y_clean == -1, 0], X_clean[y_clean == -1, 1], c='blue', marker='s', s=100, label='Class -1')
    
    x_min, x_max = X_clean[:, 0].min() - 0.5, X_clean[:, 0].max() + 0.5
    y_min, y_max = X_clean[:, 1].min() - 0.5, X_clean[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z_clean = w_clean[0] * xx + w_clean[1] * yy + b_clean
    plt.contour(xx, yy, Z_clean, levels=[0], colors='green', linewidths=2)
    plt.contour(xx, yy, Z_clean, levels=[-1, 1], colors='green', linestyles='--', alpha=0.5)
    
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Hard Margin SVM (Clean Data)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', marker='o', s=100, label='Class +1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='blue', marker='s', s=100, label='Class -1')
    plt.scatter(X[outlier_idx, 0], X[outlier_idx, 1], c='orange', marker='*', s=200, label='Outlier')
    
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z_soft = w_soft[0] * xx + w_soft[1] * yy + b_soft
    plt.contour(xx, yy, Z_soft, levels=[0], colors='green', linewidths=2)
    plt.contour(xx, yy, Z_soft, levels=[-1, 1], colors='green', linestyles='--', alpha=0.5)
    
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(f'Soft Margin SVM (C = {C})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_clean_vs_outlier.png'), dpi=300, bbox_inches='tight')
    plt.close()

print("\n" + "="*50)
print("ADDITIONAL ANALYSIS: EFFECT OF DIFFERENT C VALUES")
print("="*50)

print("Let's analyze how different C values affect the solution:")
print("C controls the trade-off between margin width and slack variables.")
print("Small C: prioritize margin width, tolerate violations")
print("Large C: prioritize correct classification, minimize violations")

C_values = [0.1, 1.0, 10.0, 100.0]
results = []

for C_val in C_values:
    print(f"\n--- C = {C_val} ---")
    alpha, xi, success = solve_soft_margin_svm(X, y, C_val)
    
    if success:
        w, b, sv_indices = get_svm_params(X, y, alpha)
        margin = 2 / np.linalg.norm(w)
        total_slack = np.sum(xi)
        outlier_slack = xi[outlier_idx]
        
        results.append({
            'C': C_val,
            'w': w,
            'b': b,
            'margin': margin,
            'total_slack': total_slack,
            'outlier_slack': outlier_slack,
            'sv_count': np.sum(sv_indices)
        })
        
        print(f"  w = {w}")
        print(f"  b = {b:.3f}")
        print(f"  Margin: 2/||w|| = {margin:.3f}")
        print(f"  Total slack: Σξᵢ = {total_slack:.3f}")
        print(f"  Outlier slack: ξ_outlier = {outlier_slack:.3f}")
        print(f"  Support vectors: {np.sum(sv_indices)}")

# Plot C vs margin and slack
if results:
    C_vals = [r['C'] for r in results]
    margins = [r['margin'] for r in results]
    total_slacks = [r['total_slack'] for r in results]
    outlier_slacks = [r['outlier_slack'] for r in results]
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.semilogx(C_vals, margins, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('C (Regularization Parameter)')
    plt.ylabel('Margin Width')
    plt.title('C vs Margin Width')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.semilogx(C_vals, total_slacks, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('C (Regularization Parameter)')
    plt.ylabel('Total Slack Variables')
    plt.title('C vs Total Slack')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.semilogx(C_vals, outlier_slacks, 'go-', linewidth=2, markersize=8)
    plt.xlabel('C (Regularization Parameter)')
    plt.ylabel('Outlier Slack Variable')
    plt.title('C vs Outlier Slack')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'C_parameter_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

print(f"\nAll plots saved to: {save_dir}")
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
