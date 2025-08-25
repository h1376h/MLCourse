import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_2_Quiz_24")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'

print("=== Soft Margin SVM Analysis ===\n")

# 1. Objective Function Components
print("1. Soft Margin Objective Function Components:")
print("Objective: min (1/2)||w||² + C∑ξᵢ")
print("   - (1/2)||w||²: Regularization term (maximizes margin)")
print("   - C∑ξᵢ: Penalty term for margin violations")
print("   - C: Regularization parameter (controls trade-off)")
print("   - ξᵢ: Slack variables (measure of margin violation)")

print("\nDetailed Analysis:")
print("Let's break down each component:")
print("1) Regularization term: (1/2)||w||²")
print("   - ||w||² = w₁² + w₂² + ... + wₙ² (Euclidean norm squared)")
print("   - This term encourages small weights, leading to larger margins")
print("   - The factor 1/2 is for mathematical convenience in derivatives")
print("   - Minimizing ||w||² maximizes the geometric margin γ = 1/||w||")

print("\n2) Penalty term: C∑ξᵢ")
print("   - ξᵢ ≥ 0 for all i (slack variables are non-negative)")
print("   - ξᵢ = 0 means point i is correctly classified with margin ≥ 1")
print("   - 0 < ξᵢ < 1 means point i is correctly classified but violates margin")
print("   - ξᵢ ≥ 1 means point i is misclassified")
print("   - C controls the trade-off: large C → small margin, few violations")
print("   - Small C → large margin, many violations allowed")

print("\n3) Complete objective function:")
print("   min_{w,b,ξ} (1/2)||w||² + C∑ᵢ₌₁ⁿ ξᵢ")
print("   subject to: yᵢ(w^T xᵢ + b) ≥ 1 - ξᵢ for all i")
print("   and: ξᵢ ≥ 0 for all i\n")

# 2. Hinge Loss Function
print("2. Hinge Loss Function:")
print("L(y, f(x)) = max(0, 1 - y*f(x))")
print("where f(x) = w^T x + b is the decision function")

print("\nDetailed Mathematical Analysis:")
print("1) Definition: L(y, f(x)) = max(0, 1 - y·f(x))")
print("   where y ∈ {-1, +1} and f(x) = w^T x + b")

print("\n2) Case Analysis:")
print("   Case 1: y·f(x) ≥ 1 (correctly classified with margin)")
print("   Then: 1 - y·f(x) ≤ 0")
print("   Therefore: L(y, f(x)) = max(0, 1 - y·f(x)) = 0")

print("\n   Case 2: y·f(x) < 1 (margin violation or misclassification)")
print("   Then: 1 - y·f(x) > 0")
print("   Therefore: L(y, f(x)) = max(0, 1 - y·f(x)) = 1 - y·f(x)")

print("\n3) Connection to Slack Variables:")
print("   ξᵢ = max(0, 1 - yᵢ(w^T xᵢ + b))")
print("   ξᵢ = max(0, 1 - yᵢ·f(xᵢ))")
print("   ξᵢ = L(yᵢ, f(xᵢ))")
print("   Therefore: ξᵢ = L(yᵢ, f(xᵢ))")

print("\n4) Properties:")
print("   - Non-negative: L(y, f(x)) ≥ 0 for all y, f(x)")
print("   - Convex: The function is convex in f(x)")
print("   - Non-differentiable at y·f(x) = 1")
print("   - Linear penalty: When y·f(x) < 1, loss increases linearly")

# Plot hinge loss
def hinge_loss(y_true, y_pred):
    return np.maximum(0, 1 - y_true * y_pred)

y_pred_range = np.linspace(-3, 3, 1000)
y_true = 1

loss_values = hinge_loss(y_true, y_pred_range)

plt.figure(figsize=(10, 6))
plt.plot(y_pred_range, loss_values, 'b-', linewidth=2, label='Hinge Loss')
plt.axvline(x=1, color='r', linestyle='--', alpha=0.7, label='Decision Boundary (y_pred = 1)')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.xlabel(r'$f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b$')
plt.ylabel(r'Hinge Loss $L(y, f(\mathbf{x}))$')
plt.title('Hinge Loss Function')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(-3, 3)
plt.ylim(0, 4)
plt.savefig(os.path.join(save_dir, 'hinge_loss.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Properties of Hinge Loss:")
print("- Non-negative: L(y, f(x)) ≥ 0")
print("- Zero when y*f(x) ≥ 1 (correctly classified with margin)")
print("- Linear penalty for margin violations")
print("- Non-differentiable at y*f(x) = 1\n")

# 3. Slack Variables and Geometric Interpretation
print("3. Slack Variables and Geometric Interpretation:")
print("ξᵢ = max(0, 1 - yᵢ(w^T xᵢ + b))")
print("ξᵢ = 0: Point is correctly classified with margin ≥ 1")
print("0 < ξᵢ < 1: Point is correctly classified but violates margin")
print("ξᵢ ≥ 1: Point is misclassified")

print("\nDetailed Mathematical Analysis:")
print("1) Definition: ξᵢ = max(0, 1 - yᵢ(w^T xᵢ + b))")
print("   where yᵢ ∈ {-1, +1} and f(xᵢ) = w^T xᵢ + b")

print("\n2) Geometric Interpretation:")
print("   Let's consider the distance from point xᵢ to the decision boundary:")
print("   Distance = |w^T xᵢ + b| / ||w||")
print("   Functional margin = yᵢ(w^T xᵢ + b)")
print("   Geometric margin = yᵢ(w^T xᵢ + b) / ||w||")

print("\n3) Case Analysis for ξᵢ:")
print("   Case 1: yᵢ(w^T xᵢ + b) ≥ 1")
print("   Then: 1 - yᵢ(w^T xᵢ + b) ≤ 0")
print("   Therefore: ξᵢ = max(0, 1 - yᵢ(w^T xᵢ + b)) = 0")
print("   Interpretation: Point is correctly classified with margin ≥ 1")

print("\n   Case 2: 0 < yᵢ(w^T xᵢ + b) < 1")
print("   Then: 0 < 1 - yᵢ(w^T xᵢ + b) < 1")
print("   Therefore: ξᵢ = 1 - yᵢ(w^T xᵢ + b)")
print("   Interpretation: Point is correctly classified but violates margin")

print("\n   Case 3: yᵢ(w^T xᵢ + b) ≤ 0")
print("   Then: 1 - yᵢ(w^T xᵢ + b) ≥ 1")
print("   Therefore: ξᵢ = 1 - yᵢ(w^T xᵢ + b) ≥ 1")
print("   Interpretation: Point is misclassified")

print("\n4) Relationship to Margin:")
print("   For correctly classified points: ξᵢ = max(0, 1 - margin)")
print("   ξᵢ measures how much the margin is violated")
print("   ξᵢ = 0 means perfect margin compliance")
print("   ξᵢ > 0 means margin violation")

# Create example with slack variables
np.random.seed(42)
X, y = make_blobs(n_samples=100, centers=2, cluster_std=1.5, random_state=42)
y = 2*y - 1  # Convert to {-1, 1}

# Fit SVM with different C values
C_values = [0.1, 1, 10]
svms = {}

for C in C_values:
    svm = SVC(kernel='linear', C=C, random_state=42)
    svm.fit(X, y)
    svms[C] = svm

# Plot slack variables
plt.figure(figsize=(15, 5))

for i, C in enumerate(C_values):
    svm = svms[C]
    w = svm.coef_[0]
    b = svm.intercept_[0]
    
    # Calculate slack variables
    decision_values = svm.decision_function(X)
    slack_vars = np.maximum(0, 1 - y * decision_values)
    
    plt.subplot(1, 3, i+1)
    
    # Plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    
    # Plot points with slack variable visualization
    for j, (x, y_true, slack) in enumerate(zip(X, y, slack_vars)):
        if slack == 0:
            color = 'green' if y_true == 1 else 'red'
            alpha = 0.7
        elif slack < 1:
            color = 'orange'
            alpha = 0.8
        else:
            color = 'purple'
            alpha = 1.0
        
        plt.scatter(x[0], x[1], c=color, alpha=alpha, s=50)
    
    plt.title(f'C = {C}\nSlack Variables')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.7, label=r'$\xi = 0$'),
                      Patch(facecolor='orange', alpha=0.8, label=r'$0 < \xi < 1$'),
                      Patch(facecolor='purple', alpha=1.0, label=r'$\xi \geq 1$')]
    plt.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'slack_variables.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Geometric interpretation:")
print("- ξᵢ = 0: Point is on correct side of margin")
print("- 0 < ξᵢ < 1: Point is on correct side but within margin")
print("- ξᵢ ≥ 1: Point is on wrong side (misclassified)")

print("\nNumerical Example:")
print("Let's calculate slack variables for a specific case:")
print("Suppose w = [2, -1]^T, b = -3, and we have point x = [1, 2]^T with y = 1")
print("Then: f(x) = w^T x + b = 2(1) + (-1)(2) + (-3) = 2 - 2 - 3 = -3")
print("Functional margin = y·f(x) = 1·(-3) = -3")
print("ξ = max(0, 1 - y·f(x)) = max(0, 1 - (-3)) = max(0, 4) = 4")
print("Since ξ = 4 ≥ 1, this point is misclassified\n")

# 4. Constraint Relationship
print("4. Constraint: yᵢ(w^T xᵢ + b) ≥ 1 - ξᵢ")
print("This constraint ensures:")
print("- If ξᵢ = 0: yᵢ(w^T xᵢ + b) ≥ 1 (hard margin constraint)")
print("- If ξᵢ > 0: yᵢ(w^T xᵢ + b) ≥ 1 - ξᵢ (allows margin violation)")
print("- ξᵢ ≥ 0: Slack variables are non-negative")

print("\nDetailed Mathematical Analysis:")
print("1) Original hard margin constraint: yᵢ(w^T xᵢ + b) ≥ 1")
print("   This requires all points to be correctly classified with margin ≥ 1")

print("\n2) Soft margin relaxation: yᵢ(w^T xᵢ + b) ≥ 1 - ξᵢ")
print("   This allows margin violations by introducing slack variables")

print("\n3) Constraint Analysis:")
print("   Case 1: ξᵢ = 0")
print("   Then: yᵢ(w^T xᵢ + b) ≥ 1 - 0 = 1")
print("   This is the original hard margin constraint")

print("\n   Case 2: ξᵢ > 0")
print("   Then: yᵢ(w^T xᵢ + b) ≥ 1 - ξᵢ")
print("   The right-hand side is reduced, allowing margin violations")

print("\n4) Relationship to Slack Variables:")
print("   From the constraint: yᵢ(w^T xᵢ + b) ≥ 1 - ξᵢ")
print("   Rearranging: ξᵢ ≥ 1 - yᵢ(w^T xᵢ + b)")
print("   But we also have: ξᵢ ≥ 0")
print("   Therefore: ξᵢ = max(0, 1 - yᵢ(w^T xᵢ + b))")

print("\n5) Complete Constraint Set:")
print("   yᵢ(w^T xᵢ + b) ≥ 1 - ξᵢ for all i")
print("   ξᵢ ≥ 0 for all i")
print("   These constraints ensure the optimization problem is well-defined\n")

# 5. KKT Conditions
print("5. KKT Condition: αᵢ + μᵢ = C")
print("Derivation:")
print("From KKT conditions:")
print("- ∂L/∂w = 0 → w = Σαᵢyᵢxᵢ")
print("- ∂L/∂b = 0 → Σαᵢyᵢ = 0")
print("- ∂L/∂ξᵢ = 0 → C - αᵢ - μᵢ = 0")
print("Therefore: αᵢ + μᵢ = C")
print("This means:")
print("- If αᵢ = 0: μᵢ = C (point is not a support vector)")
print("- If 0 < αᵢ < C: μᵢ = C - αᵢ (point is a support vector)")
print("- If αᵢ = C: μᵢ = 0 (point violates margin)")

print("\nDetailed Mathematical Derivation:")
print("1) Lagrangian Function:")
print("   L(w, b, ξ, α, μ) = (1/2)||w||² + C∑ξᵢ - ∑αᵢ[yᵢ(w^T xᵢ + b) - 1 + ξᵢ] - ∑μᵢξᵢ")

print("\n2) KKT Conditions (First-Order Optimality):")
print("   ∂L/∂w = 0:")
print("   w - ∑αᵢyᵢxᵢ = 0")
print("   Therefore: w = ∑αᵢyᵢxᵢ")

print("\n   ∂L/∂b = 0:")
print("   -∑αᵢyᵢ = 0")
print("   Therefore: ∑αᵢyᵢ = 0")

print("\n   ∂L/∂ξᵢ = 0:")
print("   C - αᵢ - μᵢ = 0")
print("   Therefore: αᵢ + μᵢ = C")

print("\n3) Complementary Slackness:")
print("   αᵢ[yᵢ(w^T xᵢ + b) - 1 + ξᵢ] = 0")
print("   μᵢξᵢ = 0")

print("\n4) Interpretation of αᵢ + μᵢ = C:")
print("   Since αᵢ ≥ 0 and μᵢ ≥ 0, and αᵢ + μᵢ = C:")
print("   - If αᵢ = 0: μᵢ = C (point not a support vector)")
print("   - If 0 < αᵢ < C: μᵢ = C - αᵢ (point is a support vector)")
print("   - If αᵢ = C: μᵢ = 0 (point violates margin)")

print("\n5) Dual Variables Relationship:")
print("   αᵢ: Lagrange multiplier for margin constraint")
print("   μᵢ: Lagrange multiplier for non-negativity constraint ξᵢ ≥ 0")
print("   Their sum equals C, the regularization parameter\n")

# 6. Functional vs Geometric Margin
print("6. Functional vs Geometric Margin:")
print("Functional margin: γ̂ = y(w^T x + b)")
print("Geometric margin: γ = γ̂ / ||w||")
print("In soft margin:")
print("- Functional margin can be < 1 due to slack variables")
print("- Geometric margin is still maximized subject to constraints")
print("- Slack variables allow functional margin to be violated")

print("\nDetailed Mathematical Analysis:")
print("1) Functional Margin Definition:")
print("   γ̂ = y(w^T x + b)")
print("   This is the signed distance from the decision boundary")

print("\n2) Geometric Margin Definition:")
print("   γ = γ̂ / ||w||")
print("   This is the actual distance from the decision boundary")

print("\n3) Relationship in Hard Margin SVM:")
print("   For hard margin: yᵢ(w^T xᵢ + b) ≥ 1 for all i")
print("   Therefore: γ̂ᵢ ≥ 1 for all i")
print("   The minimum functional margin is 1")

print("\n4) Relationship in Soft Margin SVM:")
print("   For soft margin: yᵢ(w^T xᵢ + b) ≥ 1 - ξᵢ for all i")
print("   Therefore: γ̂ᵢ ≥ 1 - ξᵢ for all i")
print("   The functional margin can be less than 1 due to slack variables")

print("\n5) Optimization Objective:")
print("   Hard margin: max γ = max (1/||w||)")
print("   Soft margin: max γ subject to constraints")
print("   Both aim to maximize the geometric margin")

print("\n6) Numerical Example:")
print("   Suppose w = [2, -1]^T, b = -3, x = [1, 2]^T, y = 1")
print("   Functional margin: γ̂ = 1·(2·1 + (-1)·2 + (-3)) = 1·(-3) = -3")
print("   Geometric margin: γ = -3 / √(2² + (-1)²) = -3 / √5 ≈ -1.34")
print("   Since γ̂ < 1, this point violates the margin\n")

# 7. Effect of C parameter
print("7. Effect of C parameter:")
print("C controls the trade-off between margin width and classification accuracy")

# Plot decision boundaries for different C values
plt.figure(figsize=(15, 5))

for i, C in enumerate(C_values):
    svm = svms[C]
    
    plt.subplot(1, 3, i+1)
    
    # Plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    
    # Plot points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', alpha=0.7, s=50)
    
    # Plot decision boundary and margins
    w = svm.coef_[0]
    b = svm.intercept_[0]
    
    # Decision boundary
    x_boundary = np.linspace(x_min, x_max, 100)
    y_boundary = (-w[0] * x_boundary - b) / w[1]
    plt.plot(x_boundary, y_boundary, 'k-', linewidth=2, label='Decision Boundary')
    
    # Margins
    margin = 1 / np.linalg.norm(w)
    y_margin1 = (-w[0] * x_boundary - b + 1) / w[1]
    y_margin2 = (-w[0] * x_boundary - b - 1) / w[1]
    plt.plot(x_boundary, y_margin1, 'k--', alpha=0.5, label='Margin')
    plt.plot(x_boundary, y_margin2, 'k--', alpha=0.5)
    
    plt.title(f'C = {C}\nMargin Width: {margin:.3f}')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'effect_of_C.png'), dpi=300, bbox_inches='tight')
plt.close()

print("C effect analysis:")
print("- C → 0: Large margin, many misclassifications")
print("- C → ∞: Small margin, few misclassifications")
print("- Optimal C balances margin width and accuracy")

print("\nDetailed Mathematical Analysis:")
print("1) Objective Function Trade-off:")
print("   min (1/2)||w||² + C∑ξᵢ")
print("   First term: encourages large margin (small ||w||)")
print("   Second term: penalizes margin violations")

print("\n2) Effect of C → 0:")
print("   The penalty term C∑ξᵢ becomes negligible")
print("   Optimization focuses on minimizing ||w||²")
print("   Results in large margin (small ||w||)")
print("   Many slack variables can be large (many violations)")

print("\n3) Effect of C → ∞:")
print("   The penalty term C∑ξᵢ dominates")
print("   Optimization focuses on minimizing ∑ξᵢ")
print("   Results in small margin (large ||w||)")
print("   Slack variables approach 0 (few violations)")

print("\n4) Optimal C Selection:")
print("   Cross-validation is typically used")
print("   Balance between margin width and classification accuracy")
print("   Depends on the specific dataset and noise level\n")

# 8. Simple Example with 3 Points
print("8. Simple Example with 3 Points:")

# Create 3 points example
X_simple = np.array([[1, 1], [2, 2], [3, 1]])
y_simple = np.array([1, 1, -1])

plt.figure(figsize=(12, 4))

for i, C in enumerate([0.1, 1, 10]):
    svm = SVC(kernel='linear', C=C, random_state=42)
    svm.fit(X_simple, y_simple)
    
    plt.subplot(1, 3, i+1)
    
    # Plot points
    colors = ['red' if y == -1 else 'blue' for y in y_simple]
    plt.scatter(X_simple[:, 0], X_simple[:, 1], c=colors, s=100, alpha=0.7)
    
    # Plot decision boundary
    w = svm.coef_[0]
    b = svm.intercept_[0]
    
    x_boundary = np.linspace(0, 4, 100)
    y_boundary = (-w[0] * x_boundary - b) / w[1]
    plt.plot(x_boundary, y_boundary, 'k-', linewidth=2)
    
    # Calculate and show slack variables
    decision_values = svm.decision_function(X_simple)
    slack_vars = np.maximum(0, 1 - y_simple * decision_values)
    
    for j, (x, slack) in enumerate(zip(X_simple, slack_vars)):
        plt.annotate(f'$\\xi={slack:.2f}$', (x[0], x[1]), 
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
    
    plt.title(f'C = {C}')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.xlim(0, 4)
    plt.ylim(0, 3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'simple_example.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Example analysis:")
print("Point (1,1): Class 1, ξ depends on C")
print("Point (2,2): Class 1, ξ depends on C")
print("Point (3,1): Class -1, ξ depends on C")
print("As C increases, slack variables decrease")

print("\nDetailed Mathematical Analysis:")
print("1) Problem Setup:")
print("   Points: (1,1), (2,2) from class 1, (3,1) from class -1")
print("   Decision boundary: w₁x₁ + w₂x₂ + b = 0")

print("\n2) Slack Variable Calculation:")
print("   For each point i: ξᵢ = max(0, 1 - yᵢ(w₁xᵢ₁ + w₂xᵢ₂ + b))")
print("   The values of w₁, w₂, b depend on the optimization with parameter C")

print("\n3) Effect of C on Slack Variables:")
print("   Small C: Allows larger slack variables")
print("   Large C: Forces smaller slack variables")
print("   The optimization balances margin width vs. classification accuracy")

print("\n4) Decision Boundary Evolution:")
print("   As C increases, the decision boundary becomes more sensitive")
print("   to individual points, potentially leading to overfitting")
print("   As C decreases, the decision boundary becomes smoother\n")

# 9. C → 0 and C → ∞
print("9. Behavior as C → 0 and C → ∞:")

# Test extreme C values
C_extreme = [0.001, 1000]
svms_extreme = {}

for C in C_extreme:
    svm = SVC(kernel='linear', C=C, random_state=42)
    svm.fit(X, y)
    svms_extreme[C] = svm

plt.figure(figsize=(12, 5))

for i, C in enumerate(C_extreme):
    svm = svms_extreme[C]
    
    plt.subplot(1, 2, i+1)
    
    # Plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', alpha=0.7, s=50)
    
    # Calculate slack variables
    decision_values = svm.decision_function(X)
    slack_vars = np.maximum(0, 1 - y * decision_values)
    total_slack = np.sum(slack_vars)
    
    plt.title(f'C = {C}\nTotal Slack: {total_slack:.2f}')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'extreme_C_values.png'), dpi=300, bbox_inches='tight')
plt.close()

print("C → 0:")
print("- Slack variables become large")
print("- Margin becomes very wide")
print("- Many misclassifications allowed")
print("- Model becomes underfit")
print("\nC → ∞:")
print("- Slack variables approach 0")
print("- Margin becomes very narrow")
print("- Few misclassifications allowed")
print("- Model becomes overfit")

print("\nDetailed Mathematical Analysis:")
print("1) Limiting Behavior Analysis:")
print("   As C → 0:")
print("   - The term C∑ξᵢ becomes negligible")
print("   - Objective becomes: min (1/2)||w||²")
print("   - This maximizes the margin (minimizes ||w||)")
print("   - Slack variables can be arbitrarily large")

print("\n   As C → ∞:")
print("   - The term C∑ξᵢ dominates")
print("   - Objective becomes: min C∑ξᵢ (equivalent to min ∑ξᵢ)")
print("   - This forces ξᵢ → 0 for all i")
print("   - Results in hard margin behavior")

print("\n2) Theoretical Limits:")
print("   C → 0: Approaches maximum margin classifier")
print("   C → ∞: Approaches hard margin SVM")
print("   Both limits may not have solutions for non-separable data")

print("\n3) Practical Implications:")
print("   Very small C: High bias, low variance (underfitting)")
print("   Very large C: Low bias, high variance (overfitting)")
print("   Optimal C: Balanced bias-variance trade-off\n")

# 10. Computational Complexity
print("10. Computational Complexity Comparison:")
print("Hard Margin SVM:")
print("- Objective: min (1/2)||w||²")
print("- Constraints: yᵢ(w^T xᵢ + b) ≥ 1")
print("- Complexity: O(n³) for interior point methods")
print("- May not have solution for non-separable data")
print("\nSoft Margin SVM:")
print("- Objective: min (1/2)||w||² + C∑ξᵢ")
print("- Constraints: yᵢ(w^T xᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0")
print("- Complexity: O(n³) for interior point methods")
print("- Always has solution")
print("- Additional variables (slack) increase problem size")

print("\nDetailed Mathematical Analysis:")
print("1) Problem Size Comparison:")
print("   Hard margin: n variables (w₁, w₂, ..., wₙ, b)")
print("   Soft margin: n + n variables (w₁, w₂, ..., wₙ, b, ξ₁, ξ₂, ..., ξₙ)")
print("   Soft margin has 2n variables vs n variables")

print("\n2) Constraint Analysis:")
print("   Hard margin: n constraints (yᵢ(w^T xᵢ + b) ≥ 1)")
print("   Soft margin: 2n constraints (yᵢ(w^T xᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0)")
print("   Soft margin has twice as many constraints")

print("\n3) Solution Existence:")
print("   Hard margin: May not have solution for non-separable data")
print("   Soft margin: Always has a solution (feasible region is non-empty)")
print("   This makes soft margin more practical")

print("\n4) Algorithmic Complexity:")
print("   Interior point methods: O(n³) for both")
print("   SMO (Sequential Minimal Optimization): O(n²) average case")
print("   The additional variables don't change asymptotic complexity")

print("\n5) Memory Requirements:")
print("   Hard margin: O(n) memory for variables")
print("   Soft margin: O(2n) memory for variables")
print("   Soft margin requires twice the memory")

# Plot complexity comparison
n_values = np.logspace(1, 3, 50)
complexity_hard = n_values**3
complexity_soft = n_values**3 * 1.2  # Slightly higher due to slack variables

plt.figure(figsize=(10, 6))
plt.loglog(n_values, complexity_hard, 'b-', linewidth=2, label='Hard Margin')
plt.loglog(n_values, complexity_soft, 'r-', linewidth=2, label='Soft Margin')
plt.xlabel('Number of samples (n)')
plt.ylabel('Computational Complexity')
plt.title('Computational Complexity Comparison')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(os.path.join(save_dir, 'complexity_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAll plots saved to: {save_dir}")
print("\n=== Analysis Complete ===")
