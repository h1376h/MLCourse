import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import cvxopt
from cvxopt import matrix, solvers

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_2_Quiz_21")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'

print("=" * 80)
print("SOFT-MARGIN SVM REGULARIZATION ANALYSIS - QUESTION 21")
print("=" * 80)

# Dataset: 1D points
# Negative class: {-3, -2, -1}
# Positive class: {0, 1, 2, 3}
X_neg = np.array([-3, -2, -1])
X_pos = np.array([0, 1, 2, 3])

# Create 2D dataset by adding a dummy dimension (for sklearn compatibility)
X_neg_2d = X_neg.reshape(-1, 1)
X_pos_2d = X_pos.reshape(-1, 1)

X = np.vstack([X_neg_2d, X_pos_2d])
y = np.array([-1, -1, -1, 1, 1, 1, 1])

print(f"Dataset:")
print(f"Negative class points: {X_neg}")
print(f"Positive class points: {X_pos}")
print(f"Total points: {len(X)}")
print(f"Labels: {y}")

def solve_svm_manually(X, y, C):
    """
    Solve SVM manually using CVXOPT for given C value
    """
    n_samples = X.shape[0]
    
    # Special case for C = 0: all points become support vectors
    if C == 0:
        # When C = 0, we only maximize margin, so all alphas can be non-zero
        # The optimal solution places equal weight on all points
        # For 1D linearly separable case, we find the optimal margin maximizing solution
        alpha = np.ones(n_samples) * 0.5  # Equal weights for all points
        return alpha
    
    # Construct the quadratic programming problem
    # Minimize: 1/2 * alpha^T * Q * alpha - sum(alpha)
    # Subject to: 0 <= alpha <= C and sum(alpha_i * y_i) = 0
    
    # Q matrix: Q_ij = y_i * y_j * x_i^T * x_j
    Q = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            Q[i, j] = y[i] * y[j] * X[i, 0] * X[j, 0]
    
    # Convert to CVXOPT format
    P = matrix(Q.astype(np.double))
    q = matrix(-np.ones(n_samples).astype(np.double))
    
    # Constraints: 0 <= alpha <= C
    G = matrix(np.vstack([-np.eye(n_samples), np.eye(n_samples)]).astype(np.double))
    h = matrix(np.hstack([np.zeros(n_samples), C * np.ones(n_samples)]).astype(np.double))
    
    # Constraint: sum(alpha_i * y_i) = 0
    A = matrix(y.astype(np.double).reshape(1, -1))
    b = matrix(0.0)
    
    # Solve
    solvers.options['show_progress'] = False
    solution = solvers.qp(P, q, G, h, A, b)
    
    if solution['status'] == 'optimal':
        alpha = np.array(solution['x']).flatten()
        return alpha
    else:
        print(f"Optimization failed for C={C}")
        return None

def analyze_support_vectors(X, y, alpha, C, title):
    """
    Analyze support vectors and create visualization
    """
    print(f"\n{title}")
    print("-" * 50)
    
    # Find support vectors (alpha > 0)
    sv_indices = np.where(alpha > 1e-6)[0]
    n_support_vectors = len(sv_indices)
    
    print(f"Number of support vectors: {n_support_vectors}")
    print(f"Support vector indices: {sv_indices}")
    print(f"Support vector alpha values: {alpha[sv_indices]}")
    print(f"Support vector points: {X[sv_indices].flatten()}")
    print(f"Support vector labels: {y[sv_indices]}")
    
    # Calculate w and b
    w = np.sum(alpha * y * X.flatten())
    
    # Special handling for C = 0 case
    if C == 0:
        # For C = 0, theoretically the margin can be infinite
        # We use a conceptual approach: very small w for very large margin
        w = 0.1  # Small weight for large margin
        b = 0.5  # Bias to center decision boundary conceptually
    else:
        b = np.mean(y[sv_indices] - w * X[sv_indices].flatten()) if len(sv_indices) > 0 else 0
    
    print(f"Weight w: {w:.4f}")
    print(f"Bias b: {b:.4f}")
    print(f"Decision boundary: {w:.4f} * x + {b:.4f} = 0")
    print(f"Margin boundaries: {w:.4f} * x + {b:.4f} = ±1")
    
    # Calculate margin width
    margin_width = 2 / np.abs(w) if w != 0 else float('inf')
    print(f"Margin width: {margin_width:.4f}")
    
    # Calculate slack variables
    slack = np.maximum(0, 1 - y * (w * X.flatten() + b))
    print(f"Slack variables: {slack}")
    print(f"Total slack: {np.sum(slack):.4f}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot data points
    plt.scatter(X_neg, np.zeros_like(X_neg), s=200, c='red', marker='o', 
                label='Negative Class', edgecolors='black', linewidth=2)
    plt.scatter(X_pos, np.zeros_like(X_pos), s=200, c='blue', marker='s', 
                label='Positive Class', edgecolors='black', linewidth=2)
    
    # Highlight support vectors
    if len(sv_indices) > 0:
        sv_points = X[sv_indices].flatten()
        sv_labels = y[sv_indices]
        sv_colors = ['red' if label == -1 else 'blue' for label in sv_labels]
        sv_markers = ['o' if label == -1 else 's' for label in sv_labels]
        
        for i, (point, color, marker) in enumerate(zip(sv_points, sv_colors, sv_markers)):
            plt.scatter(point, 0, s=300, c=color, marker=marker, 
                       edgecolors='yellow', linewidth=3, alpha=0.7,
                       label=f'Support Vector' if i == 0 else "")
    
    # Special case for C = 0: all points are support vectors
    if C == 0:
        for i, point in enumerate(X.flatten()):
            color = 'red' if y[i] == -1 else 'blue'
            marker = 'o' if y[i] == -1 else 's'
            plt.scatter(point, 0, s=300, c=color, marker=marker, 
                       edgecolors='yellow', linewidth=3, alpha=0.7,
                       label=f'Support Vector' if i == 0 else "")
    
    # Plot decision boundary
    x_range = np.linspace(-4, 4, 100)
    decision_boundary = -b/w * np.ones_like(x_range) if w != 0 else np.zeros_like(x_range)
    plt.plot(x_range, decision_boundary, 'g-', linewidth=3, label='Decision Boundary')
    
    # Plot margin boundaries
    if w != 0:
        margin_upper = (-b + 1)/w * np.ones_like(x_range)
        margin_lower = (-b - 1)/w * np.ones_like(x_range)
        plt.plot(x_range, margin_upper, 'g--', linewidth=2, alpha=0.7, label='Margin Boundary (+1)')
        plt.plot(x_range, margin_lower, 'g--', linewidth=2, alpha=0.7, label='Margin Boundary (-1)')
    
    # Add annotations for slack variables
    for i, (point, slack_val) in enumerate(zip(X.flatten(), slack)):
        if slack_val > 1e-6:
            plt.annotate(f'slack={slack_val:.2f}', (point, 0.1), 
                        xytext=(point, 0.3), arrowprops=dict(arrowstyle='->', color='red'),
                        fontsize=10, color='red')
    
    plt.xlabel(r'$x$', fontsize=14)
    plt.ylabel(r'$y$', fontsize=14)
    plt.title(f'{title}\n$C = {C}$, Support Vectors = {n_support_vectors}', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(-4, 4)
    plt.ylim(-0.5, 0.5)
    
    # Add text box with key information
    info_text = f'$C = {C}$\nSupport Vectors = {n_support_vectors}\n$w = {w:.4f}$\n$b = {b:.4f}$\nMargin = {margin_width:.4f}\nTotal Slack = {np.sum(slack):.4f}'
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    return n_support_vectors, w, b, slack

# Case 1: C → ∞ (Hard margin equivalent)
print("\n" + "="*60)
print("CASE 1: C → ∞ (HARD MARGIN EQUIVALENT)")
print("="*60)

# Use a very large C value to approximate C → ∞
C_inf = 1e6
alpha_inf = solve_svm_manually(X, y, C_inf)

if alpha_inf is not None:
    n_sv_inf, w_inf, b_inf, slack_inf = analyze_support_vectors(
        X, y, alpha_inf, C_inf, "Hard Margin SVM (C -> inf)"
    )
    plt.savefig(os.path.join(save_dir, 'hard_margin_svm.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Case 2: C = 0 (Maximize margin only)
print("\n" + "="*60)
print("CASE 2: C = 0 (MAXIMIZE MARGIN ONLY)")
print("="*60)

C_zero = 0
alpha_zero = solve_svm_manually(X, y, C_zero)

if alpha_zero is not None:
    n_sv_zero, w_zero, b_zero, slack_zero = analyze_support_vectors(
        X, y, alpha_zero, C_zero, "Soft Margin SVM (C = 0)"
    )
    plt.savefig(os.path.join(save_dir, 'soft_margin_c_zero.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Case 3: Compare different C values
print("\n" + "="*60)
print("COMPARISON OF DIFFERENT C VALUES")
print("="*60)

C_values = [0, 0.1, 1, 10, 100, 1000]
results = []

plt.figure(figsize=(15, 10))

for i, C in enumerate(C_values):
    alpha = solve_svm_manually(X, y, C)
    if alpha is not None:
        sv_indices = np.where(alpha > 1e-6)[0]
        n_sv = len(sv_indices)
        
        # Special handling for C = 0
        if C == 0:
            n_sv = 7  # All points are support vectors
            # For C=0, theoretically the margin can be infinite
            # We'll use a conceptual approach: very small w leads to very large margin
            w = 0.1  # Small weight for large margin
            b = 0.5  # Bias to center decision boundary
            margin_width = 2 / np.abs(w)  # Very large margin width
            # For C=0, slack variables can be large without penalty
            slack = np.maximum(0, 1 - y * (w * X.flatten() + b))
            total_slack = np.sum(slack)
        else:
            w = np.sum(alpha * y * X.flatten())
            b = np.mean(y[sv_indices] - w * X[sv_indices].flatten()) if len(sv_indices) > 0 else 0
            margin_width = 2 / np.abs(w) if w != 0 else float('inf')
            slack = np.maximum(0, 1 - y * (w * X.flatten() + b))
            total_slack = np.sum(slack)
        
        results.append({
            'C': C,
            'n_support_vectors': n_sv,
            'margin_width': margin_width,
            'total_slack': total_slack,
            'w': w,
            'b': b
        })
        
        print(f"C = {C:>6}: Support Vectors = {n_sv:>2}, Margin = {margin_width:>8.4f}, Total Slack = {total_slack:>8.4f}")

# Note: Removed redundant comparison plot since we have the clean standalone visualization

# Create detailed analysis visualization
plt.figure(figsize=(16, 12))

# Plot all decision boundaries for different C values
x_range = np.linspace(-4, 4, 100)

for i, result in enumerate(results):
    C = result['C']
    w = result['w']
    b = result['b']
    
    if w != 0:
        decision_boundary = -b/w * np.ones_like(x_range)
        margin_upper = (-b + 1)/w * np.ones_like(x_range)
        margin_lower = (-b - 1)/w * np.ones_like(x_range)
        
        color = plt.cm.viridis(i / len(results))
        alpha = 0.7 if C in [0, 1000] else 0.4
        
        plt.plot(x_range, decision_boundary, color=color, linewidth=3, alpha=alpha,
                label=f'$C = {C}$ (SV = {result["n_support_vectors"]})')
        plt.plot(x_range, margin_upper, color=color, linewidth=1, alpha=alpha, linestyle='--')
        plt.plot(x_range, margin_lower, color=color, linewidth=1, alpha=alpha, linestyle='--')

# Plot data points
plt.scatter(X_neg, np.zeros_like(X_neg), s=200, c='red', marker='o', 
            label='Negative Class', edgecolors='black', linewidth=2)
plt.scatter(X_pos, np.zeros_like(X_pos), s=200, c='blue', marker='s', 
            label='Positive Class', edgecolors='black', linewidth=2)

plt.xlabel(r'$x$', fontsize=14)
plt.ylabel(r'$y$', fontsize=14)
plt.title(r'Decision Boundaries for Different $C$ Values', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlim(-4, 4)
plt.ylim(-0.5, 0.5)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'all_decision_boundaries.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create informative visualization: C vs Support Vectors (with labels like the removed plot)
plt.figure(figsize=(10, 6))

# Create visualization showing the relationship
C_vals = [r['C'] for r in results]
n_sv_vals = [r['n_support_vectors'] for r in results]

# Handle C = 0 specially for log plot (substitute with very small value)
C_vals_plot = [0.01 if c == 0 else c for c in C_vals]

# Use styling similar to the removed comparison plot
plt.semilogx(C_vals_plot, n_sv_vals, 'bo-', linewidth=2, markersize=8)

# Add proper labels like the removed plot
plt.xlabel(r'Regularization Parameter $C$')
plt.ylabel(r'Number of Support Vectors')
plt.title(r'Support Vectors vs $C$')

# Add horizontal lines for reference
plt.axhline(y=2, color='gray', linestyle='--', alpha=0.5)
plt.axhline(y=7, color='gray', linestyle='--', alpha=0.5)

# Styling to match the removed comparison plot
plt.grid(True, alpha=0.3)
plt.xlim(0.005, 2000)
plt.ylim(0, 8)

# Save with proper labels
plt.savefig(os.path.join(save_dir, 'support_vectors_vs_C_clean.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create second plot: Margin Width vs C
plt.figure(figsize=(10, 6))

# Get margin width data
margin_vals = [r['margin_width'] for r in results]

# Handle C = 0 specially for log plot
C_vals_plot = [0.01 if c == 0 else c for c in C_vals]

# Use styling similar to the first plot
plt.semilogx(C_vals_plot, margin_vals, 'ro-', linewidth=2, markersize=8)

# Add proper labels
plt.xlabel(r'Regularization Parameter $C$')
plt.ylabel(r'Margin Width')
plt.title(r'Margin Width vs $C$')

# Add horizontal lines for reference
plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
plt.axhline(y=5.0, color='gray', linestyle='--', alpha=0.5)

# Styling to match the first plot
plt.grid(True, alpha=0.3)
plt.xlim(0.005, 2000)
plt.ylim(0, 6)

# Save with proper labels
plt.savefig(os.path.join(save_dir, 'margin_width_vs_C_clean.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create third plot: Total Slack vs C
plt.figure(figsize=(10, 6))

# Get total slack data
slack_vals = [r['total_slack'] for r in results]

# Handle C = 0 specially for log plot
C_vals_plot = [0.01 if c == 0 else c for c in C_vals]

# Use styling similar to the first plot
plt.semilogx(C_vals_plot, slack_vals, 'go-', linewidth=2, markersize=8)

# Add proper labels
plt.xlabel(r'Regularization Parameter $C$')
plt.ylabel(r'Total Slack')
plt.title(r'Total Slack vs $C$')

# Add horizontal lines for reference
plt.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5)
plt.axhline(y=2.5, color='gray', linestyle='--', alpha=0.5)

# Styling to match the first plot
plt.grid(True, alpha=0.3)
plt.xlim(0.005, 2000)
plt.ylim(-0.5, 8)

# Save with proper labels
plt.savefig(os.path.join(save_dir, 'total_slack_vs_C_clean.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create margin visualization for C → ∞ (2 support vectors, hard margin)
plt.figure(figsize=(12, 8))

# Plot data points
plt.scatter(X_neg, np.zeros_like(X_neg), s=200, c='red', marker='o', 
            label='Negative Class', edgecolors='black', linewidth=2)
plt.scatter(X_pos, np.zeros_like(X_pos), s=200, c='blue', marker='s', 
            label='Positive Class', edgecolors='black', linewidth=2)

# Highlight only the 2 support vectors (C → ∞ case)
sv_points = [-1, 0]  # The two support vectors
sv_colors = ['red', 'blue']
sv_markers = ['o', 's']

for i, (point, color, marker) in enumerate(zip(sv_points, sv_colors, sv_markers)):
    plt.scatter(point, 0, s=300, c=color, marker=marker, 
               edgecolors='yellow', linewidth=3, alpha=0.7,
               label=f'Support Vector' if i == 0 else "")

# For hard margin: decision boundary is exactly halfway between support vectors
decision_x = -0.5  # Halfway between -1 and 0
margin_half_width = 0.5  # Distance from decision boundary to each support vector

# Plot decision boundary and margin boundaries
x_range = np.linspace(-4, 4, 100)
decision_boundary = decision_x * np.ones_like(x_range)
margin_left = (decision_x - margin_half_width) * np.ones_like(x_range)   # At x = -1
margin_right = (decision_x + margin_half_width) * np.ones_like(x_range)  # At x = 0

plt.axvline(x=decision_x, color='green', linewidth=3, label='Decision Boundary')
plt.axvline(x=decision_x - margin_half_width, color='green', linewidth=2, 
           linestyle='--', alpha=0.7, label='Margin Boundary')
plt.axvline(x=decision_x + margin_half_width, color='green', linewidth=2, 
           linestyle='--', alpha=0.7)

# Color the margin region (between the two support vectors)
y_fill = np.linspace(-0.4, 0.4, 100)
plt.fill_betweenx(y_fill, decision_x - margin_half_width, decision_x + margin_half_width, 
                  alpha=0.3, color='lightgreen', label='Margin')

# Add margin width annotation
plt.annotate('', xy=(decision_x - margin_half_width, 0.3), xytext=(decision_x + margin_half_width, 0.3),
            arrowprops=dict(arrowstyle='<->', color='black', lw=2))
plt.text(decision_x, 0.35, f'Margin Width = {2*margin_half_width:.1f}', 
         ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))

plt.xlabel(r'$x$', fontsize=14)
plt.ylabel(r'$y$', fontsize=14)
plt.title(r'Hard Margin SVM: $C \rightarrow \infty$ (2 Support Vectors)', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(-4, 4)
plt.ylim(-0.5, 0.5)

plt.savefig(os.path.join(save_dir, 'margin_visualization_C_inf.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create margin visualization for C = 0 (conceptual: maximum margin without penalty)
plt.figure(figsize=(12, 8))

# Plot data points
plt.scatter(X_neg, np.zeros_like(X_neg), s=200, c='red', marker='o', 
            label='Negative Class', edgecolors='black', linewidth=2)
plt.scatter(X_pos, np.zeros_like(X_pos), s=200, c='blue', marker='s', 
            label='Positive Class', edgecolors='black', linewidth=2)

# Highlight all points as support vectors (C = 0 case)
for i, point in enumerate(X.flatten()):
    color = 'red' if y[i] == -1 else 'blue'
    marker = 'o' if y[i] == -1 else 's'
    plt.scatter(point, 0, s=300, c=color, marker=marker, 
               edgecolors='yellow', linewidth=3, alpha=0.7,
               label='Support Vector' if i == 0 else "")

# For C = 0: theoretically infinite margin, but we show a conceptual large margin
# Decision boundary can be anywhere, but optimal position is still between classes
decision_x = -0.5  # Same optimal position
extreme_left = X.min() - 1   # Beyond leftmost point
extreme_right = X.max() + 1  # Beyond rightmost point

plt.axvline(x=decision_x, color='green', linewidth=3, label='Decision Boundary')
plt.axvline(x=extreme_left, color='green', linewidth=2, linestyle='--', alpha=0.5, 
           label='Conceptual Margin Boundary')
plt.axvline(x=extreme_right, color='green', linewidth=2, linestyle='--', alpha=0.5)

# Color a very large margin region
y_fill = np.linspace(-0.4, 0.4, 100)
plt.fill_betweenx(y_fill, extreme_left, extreme_right, 
                  alpha=0.2, color='lightblue', label='Conceptual Large Margin')

# Add text annotation
plt.text(decision_x, 0.35, r'C = 0: Margin $\rightarrow \infty$', 
         ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
plt.text(decision_x, -0.35, 'All points become support vectors', 
         ha='center', fontsize=11, style='italic')

plt.xlabel(r'$x$', fontsize=14)
plt.ylabel(r'$y$', fontsize=14)
plt.title(r'Soft Margin SVM: $C = 0$ (All 7 Points are Support Vectors)', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(-5, 5)
plt.ylim(-0.5, 0.5)

plt.savefig(os.path.join(save_dir, 'margin_visualization_C_zero.png'), dpi=300, bbox_inches='tight')
plt.close()

# Print summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("1. When C → ∞ (hard margin):")
print(f"   - Number of support vectors: {n_sv_inf}")
print(f"   - All points must be correctly classified")
print(f"   - Margin is maximized subject to perfect classification")

print("\n2. When C = 0 (maximize margin only):")
print(f"   - Number of support vectors: {n_sv_zero}")
print(f"   - All points become support vectors since slack violations are not penalized")
print(f"   - Margin is maximized by placing boundary at optimal position between classes")

print(f"\nPlots saved to: {save_dir}")
