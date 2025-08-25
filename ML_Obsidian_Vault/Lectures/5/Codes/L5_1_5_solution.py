import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_1_Quiz_5")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("QUESTION 5: ANALYTICAL SOLUTION OF DUAL SVM PROBLEM")
print("=" * 80)

# Dataset
X = np.array([
    [0, 1],   # x1
    [1, 0],   # x2
    [-1, -1]  # x3
])

y = np.array([1, 1, -1])  # Labels

print("\nDataset:")
for i in range(len(X)):
    print(f"x_{i+1} = {X[i]}, y_{i+1} = {y[i]:+d}")

print("\n" + "="*50)
print("STEP 1: SET UP THE DUAL OPTIMIZATION PROBLEM")
print("="*50)

# Compute kernel matrix K_ij = y_i * y_j * x_i^T * x_j
K = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        K[i, j] = y[i] * y[j] * np.dot(X[i], X[j])

print("\nKernel Matrix K_ij = y_i * y_j * x_i^T * x_j:")
print("K =")
for i in range(3):
    row_str = "["
    for j in range(3):
        row_str += f"{K[i,j]:6.1f}"
        if j < 2:
            row_str += ", "
    row_str += "]"
    print(f"    {row_str}")

print("\nDetailed calculations:")
for i in range(3):
    for j in range(3):
        dot_product = np.dot(X[i], X[j])
        print(f"K_{i+1}{j+1} = y_{i+1} * y_{j+1} * x_{i+1}^T * x_{j+1} = {y[i]:+d} * {y[j]:+d} * {dot_product:4.1f} = {K[i,j]:6.1f}")

print("\nDual Problem:")
print("maximize: Σ α_i - (1/2) Σ_i Σ_j α_i α_j K_ij")
print("subject to: Σ α_i y_i = 0, α_i ≥ 0")

print("\n" + "="*50)
print("STEP 2: SOLVE THE DUAL PROBLEM ANALYTICALLY")
print("="*50)

# The dual problem is:
# maximize: α1 + α2 + α3 - (1/2)(α1²K11 + α2²K22 + α3²K33 + 2α1α2K12 + 2α1α3K13 + 2α2α3K23)
# subject to: α1 + α2 - α3 = 0 (since y1=1, y2=1, y3=-1)
#            α1, α2, α3 ≥ 0

print("Constraint: α1*y1 + α2*y2 + α3*y3 = α1*(1) + α2*(1) + α3*(-1) = α1 + α2 - α3 = 0")
print("Therefore: α3 = α1 + α2")

print("\nSubstituting α3 = α1 + α2 into the objective:")
print("L(α1, α2) = α1 + α2 + (α1 + α2) - (1/2)[α1²*1 + α2²*1 + (α1+α2)²*1 + 2α1α2*1 + 2α1(α1+α2)*(-2) + 2α2(α1+α2)*(-2)]")

# Expand the objective function
print("\nExpanding the quadratic terms:")
print("= 2α1 + 2α2 - (1/2)[α1² + α2² + α1² + 2α1α2 + α2² + 2α1α2 - 4α1² - 4α1α2 - 4α1α2 - 4α2²]")
print("= 2α1 + 2α2 - (1/2)[2α1² + 2α2² + 2α1α2 - 4α1² - 8α1α2 - 4α2²]")
print("= 2α1 + 2α2 - (1/2)[-2α1² - 2α2² - 6α1α2]")
print("= 2α1 + 2α2 + α1² + α2² + 3α1α2")

# Take derivatives to find critical points
print("\nTaking partial derivatives:")
print("∂L/∂α1 = 2 + 2α1 + 3α2 = 0")
print("∂L/∂α2 = 2 + 2α2 + 3α1 = 0")

print("\nSolving the system:")
print("2α1 + 3α2 = -2")
print("3α1 + 2α2 = -2")

# Solve the linear system
A_system = np.array([[2, 3], [3, 2]])
b_system = np.array([-2, -2])
alpha_solution = np.linalg.solve(A_system, b_system)

print(f"\nSolution: α1 = {alpha_solution[0]:.3f}, α2 = {alpha_solution[1]:.3f}")
print(f"Therefore: α3 = α1 + α2 = {alpha_solution[0]:.3f} + {alpha_solution[1]:.3f} = {alpha_solution[0] + alpha_solution[1]:.3f}")

# Check if solution satisfies constraints
alpha_opt = np.array([alpha_solution[0], alpha_solution[1], alpha_solution[0] + alpha_solution[1]])

print(f"\nChecking constraints:")
print(f"α1 = {alpha_opt[0]:.3f} ≥ 0? {alpha_opt[0] >= 0}")
print(f"α2 = {alpha_opt[1]:.3f} ≥ 0? {alpha_opt[1] >= 0}")
print(f"α3 = {alpha_opt[2]:.3f} ≥ 0? {alpha_opt[2] >= 0}")

constraint_sum = np.sum(alpha_opt * y)
print(f"Constraint check: Σ α_i y_i = {constraint_sum:.6f} ≈ 0? {abs(constraint_sum) < 1e-10}")

# Since we get negative alphas, we need to solve with proper constraints
print("\nSince the unconstrained solution gives negative α values, we need to solve with constraints.")
print("We must solve the constrained quadratic programming problem analytically.")

print("\n" + "="*60)
print("ANALYTICAL SOLUTION OF CONSTRAINED QUADRATIC PROGRAMMING")
print("="*60)

print("\nThe constrained problem is:")
print("maximize: L(α1, α2) = 2α1 + 2α2 + α1² + α2² + 3α1α2")
print("subject to: α1 ≥ 0, α2 ≥ 0, α3 = α1 + α2 ≥ 0")

print("\nSince α3 = α1 + α2, the constraint α3 ≥ 0 is automatically satisfied if α1, α2 ≥ 0.")
print("Therefore, we need to solve:")
print("maximize: L(α1, α2) = 2α1 + 2α2 + α1² + α2² + 3α1α2")
print("subject to: α1 ≥ 0, α2 ≥ 0")

print("\n" + "-"*50)
print("METHOD 1: KARUSH-KUHN-TUCKER (KKT) CONDITIONS")
print("-"*50)

print("\nThe KKT conditions for this problem are:")
print("1. Stationarity: ∇L = 0")
print("2. Primal feasibility: α1 ≥ 0, α2 ≥ 0")
print("3. Dual feasibility: λ1 ≥ 0, λ2 ≥ 0")
print("4. Complementary slackness: λ1*α1 = 0, λ2*α2 = 0")

print("\nThe Lagrangian is:")
print("L̃(α1, α2, λ1, λ2) = 2α1 + 2α2 + α1² + α2² + 3α1α2 - λ1*α1 - λ2*α2")

print("\nStationarity conditions:")
print("∂L̃/∂α1 = 2 + 2α1 + 3α2 - λ1 = 0")
print("∂L̃/∂α2 = 2 + 2α2 + 3α1 - λ2 = 0")

print("\nThis gives us:")
print("λ1 = 2 + 2α1 + 3α2")
print("λ2 = 2 + 2α2 + 3α1")

print("\nWe need to consider different cases based on which constraints are active.")

print("\n" + "-"*50)
print("CASE ANALYSIS")
print("-"*50)

print("\nCase 1: α1 > 0, α2 > 0 (both constraints inactive)")
print("Then λ1 = λ2 = 0 by complementary slackness.")
print("This gives us the unconstrained solution:")
print("2 + 2α1 + 3α2 = 0")
print("2 + 2α2 + 3α1 = 0")

# Solve the system analytically
print("\nSolving this system:")
print("2α1 + 3α2 = -2")
print("3α1 + 2α2 = -2")

print("\nUsing elimination method:")
print("Multiply first equation by 3: 6α1 + 9α2 = -6")
print("Multiply second equation by 2: 6α1 + 4α2 = -4")
print("Subtract: 5α2 = -2")
print("Therefore: α2 = -2/5 = -0.4")

print("\nSubstitute back:")
print("2α1 + 3(-0.4) = -2")
print("2α1 - 1.2 = -2")
print("2α1 = -0.8")
print("α1 = -0.4")

print(f"\nSolution: α1 = -0.4, α2 = -0.4")
print("But this violates the constraints α1 ≥ 0, α2 ≥ 0!")
print("Therefore, Case 1 is not valid.")

print("\nCase 2: α1 = 0, α2 > 0 (first constraint active)")
print("Then λ1 ≥ 0, λ2 = 0 by complementary slackness.")
print("This gives us:")
print("λ1 = 2 + 2(0) + 3α2 = 2 + 3α2 ≥ 0")
print("0 = 2 + 2α2 + 3(0) = 2 + 2α2")

print("\nFrom the second equation:")
print("2 + 2α2 = 0")
print("2α2 = -2")
print("α2 = -1")

print("But α2 = -1 violates α2 > 0!")
print("Therefore, Case 2 is not valid.")

print("\nCase 3: α1 > 0, α2 = 0 (second constraint active)")
print("Then λ1 = 0, λ2 ≥ 0 by complementary slackness.")
print("This gives us:")
print("0 = 2 + 2α1 + 3(0) = 2 + 2α1")
print("λ2 = 2 + 2(0) + 3α1 = 2 + 3α1 ≥ 0")

print("\nFrom the first equation:")
print("2 + 2α1 = 0")
print("2α1 = -2")
print("α1 = -1")

print("But α1 = -1 violates α1 > 0!")
print("Therefore, Case 3 is not valid.")

print("\nCase 4: α1 = 0, α2 = 0 (both constraints active)")
print("Then λ1 ≥ 0, λ2 ≥ 0 by complementary slackness.")
print("This gives us:")
print("λ1 = 2 + 2(0) + 3(0) = 2 ≥ 0 ✓")
print("λ2 = 2 + 2(0) + 3(0) = 2 ≥ 0 ✓")

print("\nThis case is feasible!")
print("α1 = 0, α2 = 0, α3 = 0 + 0 = 0")
print("Objective value: L(0,0) = 2(0) + 2(0) + 0² + 0² + 3(0)(0) = 0")

print("\n" + "-"*50)
print("METHOD 2: GEOMETRIC INTERPRETATION")
print("-"*50)

print("\nThe objective function L(α1, α2) = 2α1 + 2α2 + α1² + α2² + 3α1α2")
print("can be written in matrix form as:")
print("L(α) = c^T α + (1/2) α^T Q α")

print("\nwhere:")
print("c = [2, 2]^T")
print("Q = [[2, 3], [3, 2]]")

print("\nThe gradient is:")
print("∇L = c + Qα = [2, 2]^T + [[2, 3], [3, 2]] [α1, α2]^T")

print("\nAt the boundary α1 = 0:")
print("∇L = [2, 2]^T + [0, 3α2]^T + [0, 2α2]^T = [2, 2 + 5α2]^T")
print("For optimality, we need ∇L to point outward from the feasible region.")
print("This means the first component should be negative: 2 < 0 (false)")
print("Therefore, α1 = 0 is not optimal.")

print("\nAt the boundary α2 = 0:")
print("∇L = [2, 2]^T + [2α1, 0]^T + [3α1, 0]^T = [2 + 5α1, 2]^T")
print("For optimality, we need ∇L to point outward from the feasible region.")
print("This means the second component should be negative: 2 < 0 (false)")
print("Therefore, α2 = 0 is not optimal.")

print("\n" + "-"*50)
print("METHOD 3: ACTIVE SET METHOD")
print("-"*50)

print("\nLet's try to find the optimal solution by considering the gradient")
print("and moving in the direction that improves the objective while")
print("respecting the constraints.")

print("\nStarting from the origin (0,0):")
print("∇L(0,0) = [2, 2]^T")

print("\nThe gradient points in the positive direction, so we should")
print("move in the direction of the gradient until we hit a constraint.")

print("\nLet's move in the direction [1, 1] (normalized gradient):")
print("α1 = t, α2 = t, where t ≥ 0")

print("\nSubstituting into the objective:")
print("L(t) = 2t + 2t + t² + t² + 3t² = 4t + 5t²")

print("\nTaking derivative:")
print("dL/dt = 4 + 10t")

print("\nSetting to zero:")
print("4 + 10t = 0")
print("t = -0.4")

print("But t = -0.4 violates t ≥ 0!")
print("This suggests the optimal solution is at the boundary.")

print("\n" + "-"*50)
print("METHOD 4: COMPLETE ENUMERATION OF BOUNDARY POINTS")
print("-"*50)

print("\nSince the unconstrained solution is infeasible, the optimal")
print("solution must lie on the boundary. Let's check all boundary cases.")

print("\nBoundary case 1: α1 = 0, α2 varies")
print("L(0, α2) = 2α2 + α2²")
print("dL/dα2 = 2 + 2α2")
print("Setting to zero: 2 + 2α2 = 0 → α2 = -1")
print("But α2 = -1 violates α2 ≥ 0")

print("\nBoundary case 2: α2 = 0, α1 varies")
print("L(α1, 0) = 2α1 + α1²")
print("dL/dα1 = 2 + 2α1")
print("Setting to zero: 2 + 2α1 = 0 → α1 = -1")
print("But α1 = -1 violates α1 ≥ 0")

print("\nBoundary case 3: α1 = α2 (from constraint α3 = α1 + α2)")
print("Let α1 = α2 = t ≥ 0")
print("L(t, t) = 2t + 2t + t² + t² + 3t² = 4t + 5t²")
print("dL/dt = 4 + 10t")
print("Setting to zero: 4 + 10t = 0 → t = -0.4")
print("But t = -0.4 violates t ≥ 0")

print("\n" + "-"*50)
print("METHOD 5: QUADRATIC PROGRAMMING WITH INEQUALITY CONSTRAINTS")
print("-"*50)

print("\nLet's solve this systematically using the method of Lagrange multipliers")
print("with inequality constraints.")

print("\nThe problem is:")
print("maximize: L(α1, α2) = 2α1 + 2α2 + α1² + α2² + 3α1α2")
print("subject to: α1 ≥ 0, α2 ≥ 0")

print("\nThe KKT conditions are:")
print("1. ∇L - λ1∇g1 - λ2∇g2 = 0, where g1 = -α1 ≤ 0, g2 = -α2 ≤ 0")
print("2. g1 ≤ 0, g2 ≤ 0 (primal feasibility)")
print("3. λ1 ≥ 0, λ2 ≥ 0 (dual feasibility)")
print("4. λ1*g1 = 0, λ2*g2 = 0 (complementary slackness)")

print("\nThis gives us:")
print("∂L/∂α1 - λ1(-1) = 0 → 2 + 2α1 + 3α2 + λ1 = 0")
print("∂L/∂α2 - λ2(-1) = 0 → 2 + 2α2 + 3α1 + λ2 = 0")
print("-α1 ≤ 0, -α2 ≤ 0")
print("λ1 ≥ 0, λ2 ≥ 0")
print("λ1*(-α1) = 0, λ2*(-α2) = 0")

print("\nFrom complementary slackness:")
print("Either λ1 = 0 or α1 = 0")
print("Either λ2 = 0 or α2 = 0")

print("\nLet's consider the case where both constraints are active:")
print("α1 = 0, α2 = 0")
print("Then: 2 + λ1 = 0 → λ1 = -2")
print("      2 + λ2 = 0 → λ2 = -2")
print("But λ1 = -2, λ2 = -2 violate λ1 ≥ 0, λ2 ≥ 0")

print("\nLet's consider the case where one constraint is active:")
print("Case A: α1 = 0, α2 > 0")
print("Then: 2 + 3α2 + λ1 = 0")
print("      2 + 2α2 = 0")
print("From second equation: α2 = -1 (violates α2 > 0)")

print("Case B: α1 > 0, α2 = 0")
print("Then: 2 + 2α1 = 0")
print("      2 + 3α1 + λ2 = 0")
print("From first equation: α1 = -1 (violates α1 > 0)")

print("\nLet's consider the case where no constraints are active:")
print("α1 > 0, α2 > 0, λ1 = λ2 = 0")
print("Then: 2 + 2α1 + 3α2 = 0")
print("      2 + 2α2 + 3α1 = 0")
print("This gives us the unconstrained solution: α1 = α2 = -0.4")
print("But this violates α1 > 0, α2 > 0")

print("\n" + "-"*50)
print("METHOD 6: SIMPLIFIED APPROACH - FINDING THE CORRECT SOLUTION")
print("-"*50)

print("\nLet me reconsider the problem. The issue might be in our")
print("understanding of the constraints. Let's look at the original problem:")

print("\nOriginal dual problem:")
print("maximize: Σ α_i - (1/2) Σ_i Σ_j α_i α_j K_ij")
print("subject to: Σ α_i y_i = 0, α_i ≥ 0")

print("\nFor our specific case:")
print("maximize: α1 + α2 + α3 - (1/2)(α1² + α2² + α3² + 2α1α2 + 2α1α3 + 2α2α3)")
print("subject to: α1 + α2 - α3 = 0, α1 ≥ 0, α2 ≥ 0, α3 ≥ 0")

print("\nSubstituting α3 = α1 + α2:")
print("maximize: α1 + α2 + (α1 + α2) - (1/2)(α1² + α2² + (α1+α2)² + 2α1α2 + 2α1(α1+α2) + 2α2(α1+α2))")
print("subject to: α1 ≥ 0, α2 ≥ 0")

print("\nSimplifying the objective:")
print("= 2α1 + 2α2 - (1/2)(α1² + α2² + α1² + 2α1α2 + α2² + 2α1α2 + 2α1² + 2α1α2 + 2α1α2 + 2α2²)")
print("= 2α1 + 2α2 - (1/2)(2α1² + 2α2² + 6α1α2)")
print("= 2α1 + 2α2 - α1² - α2² - 3α1α2")

print("\nWait! I made an error in the earlier expansion.")
print("Let me recalculate the objective function correctly.")

print("\nThe kernel matrix K is:")
print("K = [[1, 0, 1], [0, 1, 1], [1, 1, 2]]")

print("\nThe objective is:")
print("α1 + α2 + α3 - (1/2)(α1²*1 + α2²*1 + α3²*2 + 2α1α2*0 + 2α1α3*1 + 2α2α3*1)")

print("\nSubstituting α3 = α1 + α2:")
print("= α1 + α2 + (α1 + α2) - (1/2)(α1² + α2² + 2(α1+α2)² + 2α1(α1+α2) + 2α2(α1+α2))")
print("= 2α1 + 2α2 - (1/2)(α1² + α2² + 2α1² + 4α1α2 + 2α2² + 2α1² + 2α1α2 + 2α1α2 + 2α2²)")
print("= 2α1 + 2α2 - (1/2)(5α1² + 5α2² + 8α1α2)")
print("= 2α1 + 2α2 - (5/2)α1² - (5/2)α2² - 4α1α2")

print("\nTaking partial derivatives:")
print("∂L/∂α1 = 2 - 5α1 - 4α2")
print("∂L/∂α2 = 2 - 5α2 - 4α1")

print("\nSetting to zero:")
print("2 - 5α1 - 4α2 = 0")
print("2 - 5α2 - 4α1 = 0")

print("\nThis gives us:")
print("5α1 + 4α2 = 2")
print("4α1 + 5α2 = 2")

print("\nSolving this system:")
print("Multiply first by 4: 20α1 + 16α2 = 8")
print("Multiply second by 5: 20α1 + 25α2 = 10")
print("Subtract: -9α2 = -2")
print("α2 = 2/9")

print("\nSubstitute back:")
print("5α1 + 4(2/9) = 2")
print("5α1 + 8/9 = 2")
print("5α1 = 2 - 8/9 = 10/9")
print("α1 = 2/9")

print(f"\nTherefore: α1 = 2/9, α2 = 2/9, α3 = 4/9")
print("This solution satisfies all constraints!")

# Calculate the optimal solution
alpha_optimal = np.array([2/9, 2/9, 4/9])

print(f"\nOptimal solution:")
for i in range(3):
    print(f"α_{i+1}* = {alpha_optimal[i]:.6f}")

print("\nVerifying constraints:")
print(f"α1 = {alpha_optimal[0]:.6f} ≥ 0? {alpha_optimal[0] >= 0} ✓")
print(f"α2 = {alpha_optimal[1]:.6f} ≥ 0? {alpha_optimal[1] >= 0} ✓")
print(f"α3 = {alpha_optimal[2]:.6f} ≥ 0? {alpha_optimal[2] >= 0} ✓")

constraint_sum = np.sum(alpha_optimal * y)
print(f"Constraint check: Σ α_i y_i = {constraint_sum:.6f} ≈ 0? {abs(constraint_sum) < 1e-10} ✓")

print(f"\nObjective value: L(2/9, 2/9) = {2*(2/9) + 2*(2/9) - (5/2)*(2/9)**2 - (5/2)*(2/9)**2 - 4*(2/9)*(2/9):.6f}")

print("\n" + "="*50)
print("STEP 3: CALCULATE THE OPTIMAL WEIGHT VECTOR")
print("="*50)

# Calculate w* = Σ α_i y_i x_i
w_optimal = np.zeros(2)
for i in range(3):
    w_optimal += alpha_optimal[i] * y[i] * X[i]

print("w* = Σ α_i* y_i x_i")
for i in range(3):
    contribution = alpha_optimal[i] * y[i] * X[i]
    print(f"   + α_{i+1}* * y_{i+1} * x_{i+1} = {alpha_optimal[i]:.6f} * {y[i]:+d} * {X[i]} = {contribution}")

print(f"\nw* = {w_optimal}")

print("\n" + "="*50)
print("STEP 4: FIND THE BIAS TERM b*")
print("="*50)

# Find support vectors (α_i > 0)
support_vectors = []
for i in range(3):
    if alpha_optimal[i] > 1e-6:  # Numerical tolerance
        support_vectors.append(i)
        print(f"Point {i+1} is a support vector (α_{i+1}* = {alpha_optimal[i]:.6f} > 0)")

# Calculate b using support vector conditions
b_values = []
for sv in support_vectors:
    # For support vectors: y_i(w^T x_i + b) = 1
    # Therefore: b = y_i - w^T x_i
    b_val = y[sv] - np.dot(w_optimal, X[sv])
    b_values.append(b_val)
    print(f"Using support vector {sv+1}: b = y_{sv+1} - w*^T x_{sv+1} = {y[sv]} - {np.dot(w_optimal, X[sv]):.6f} = {b_val:.6f}")

b_optimal = np.mean(b_values)
print(f"\nb* = {b_optimal:.6f}")

print("\n" + "="*50)
print("STEP 5: WRITE THE FINAL DECISION FUNCTION")
print("="*50)

print("Decision function: f(x) = sign(w*^T x + b*)")
print(f"f(x) = sign({w_optimal[0]:.6f} * x1 + {w_optimal[1]:.6f} * x2 + {b_optimal:.6f})")

# Verify the solution
print("\nVerification - checking all training points:")
for i in range(3):
    decision_value = np.dot(w_optimal, X[i]) + b_optimal
    prediction = np.sign(decision_value)
    margin = y[i] * decision_value
    print(f"Point {i+1}: f(x_{i+1}) = {decision_value:.6f}, prediction = {prediction:+.0f}, margin = {margin:.6f}")

print("\n" + "="*50)
print("STEP 6: VISUALIZATION")
print("="*50)

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Data points and decision boundary
x1_range = np.linspace(-2.5, 2.5, 100)
if abs(w_optimal[1]) > 1e-10:
    x2_boundary = -(w_optimal[0] * x1_range + b_optimal) / w_optimal[1]
    ax1.plot(x1_range, x2_boundary, 'k-', linewidth=2, label='Decision Boundary')

    # Plot margin boundaries
    margin_width = 1.0 / np.linalg.norm(w_optimal)
    x2_pos_margin = -(w_optimal[0] * x1_range + b_optimal - 1) / w_optimal[1]
    x2_neg_margin = -(w_optimal[0] * x1_range + b_optimal + 1) / w_optimal[1]
    ax1.plot(x1_range, x2_pos_margin, 'k--', alpha=0.7, label='Positive Margin')
    ax1.plot(x1_range, x2_neg_margin, 'k--', alpha=0.7, label='Negative Margin')

# Plot data points
colors = ['red', 'blue']
markers = ['o', 's']
for i in range(3):
    color_idx = 0 if y[i] == 1 else 1
    marker_size = 150 if i in support_vectors else 100
    edge_width = 3 if i in support_vectors else 1

    ax1.scatter(X[i, 0], X[i, 1], c=colors[color_idx], marker=markers[color_idx],
               s=marker_size, edgecolors='black', linewidth=edge_width,
               label=f'Class {y[i]:+d}' if i < 2 else None)

    # Add point labels
    ax1.annotate(f'x_{i+1}', (X[i, 0], X[i, 1]), xytext=(10, 10),
                textcoords='offset points', fontsize=12, fontweight='bold')

ax1.set_xlabel('$x_1$', fontsize=14)
ax1.set_ylabel('$x_2$', fontsize=14)
ax1.set_title('SVM Solution: Data Points and Decision Boundary', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.axis('equal')
ax1.set_xlim(-2.5, 2.5)
ax1.set_ylim(-2.5, 2.5)

# Plot 2: Dual variables
ax2.bar(range(1, 4), alpha_optimal, color=['lightblue', 'lightgreen', 'lightcoral'])
ax2.set_xlabel('Data Point Index', fontsize=14)
ax2.set_ylabel('$\\alpha_i^*$', fontsize=14)
ax2.set_title('Optimal Dual Variables', fontsize=14)
ax2.set_xticks(range(1, 4))
ax2.set_xticklabels([f'$\\alpha_{i}^*$' for i in range(1, 4)])
ax2.grid(True, alpha=0.3)

# Add value labels on bars
for i, v in enumerate(alpha_optimal):
    ax2.text(i+1, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'svm_analytical_solution.png'), dpi=300, bbox_inches='tight')

print(f"Visualization saved to: {os.path.join(save_dir, 'svm_analytical_solution.png')}")

print("\n" + "="*50)
print("STRATEGIC GAME INTERPRETATION")
print("="*50)

print("In the strategy game context:")
print("- Red Army units (Class +1) are at (0,1) and (1,0)")
print("- Blue Army unit (Class -1) is at (-1,-1)")
print(f"- Optimal defensive wall equation: {w_optimal[0]:.3f}x₁ + {w_optimal[1]:.3f}x₂ + {b_optimal:.3f} = 0")

margin_width = 2.0 / np.linalg.norm(w_optimal)
print(f"- Safety margin (distance between armies): {margin_width:.3f} units")

print(f"- Support vectors (critical positions): Points {[i+1 for i in support_vectors]}")
print("- For maximum advantage, place additional Red unit away from the decision boundary")
print("  but in the positive region (where w^T x + b > 0)")

print("\n" + "="*80)
print("SOLUTION COMPLETE")
print("="*80)