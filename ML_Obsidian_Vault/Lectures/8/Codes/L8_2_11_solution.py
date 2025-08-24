import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_2_Quiz_11")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("FEATURE SELECTION OPTIMIZATION - QUESTION 11 SOLUTION")
print("=" * 80)

# ============================================================================
# PART 1: Objective Function Components and Trade-offs
# ============================================================================
print("\n" + "="*60)
print("PART 1: OBJECTIVE FUNCTION COMPONENTS AND TRADE-OFFS")
print("="*60)

print("\nThe objective function for feature selection can be formulated as:")
print("f(n) = α × Accuracy(n) - β × Complexity(n) - γ × Cost(n)")
print("\nwhere:")
print("• n = number of features")
print("• α, β, γ = weighting coefficients")
print("• Accuracy(n) = model performance (0-1 scale)")
print("• Complexity(n) = interpretability and model complexity")
print("• Cost(n) = computational and storage costs")

print("\nTrade-offs:")
print("• More features → Higher accuracy but increased complexity and cost")
print("• Fewer features → Lower complexity and cost but potentially lower accuracy")
print("• The challenge is finding the optimal balance point")

# ============================================================================
# PART 2: Constraints and Budget Calculation
# ============================================================================
print("\n" + "="*60)
print("PART 2: CONSTRAINTS AND BUDGET CALCULATION")
print("="*60)

budget = 1000  # computational units
cost_per_feature = 50  # units per feature

max_features = budget // cost_per_feature
remaining_budget = budget % cost_per_feature

print(f"\nBudget constraint analysis:")
print(f"• Total budget: {budget} computational units")
print(f"• Cost per feature: {cost_per_feature} units")
print(f"• Maximum features possible: {max_features}")
print(f"• Remaining budget: {remaining_budget} units")

print(f"\nTherefore, the maximum number of features we can consider is: {max_features}")

# ============================================================================
# PART 3: Multi-objective Optimization
# ============================================================================
print("\n" + "="*60)
print("PART 3: MULTI-OBJECTIVE OPTIMIZATION")
print("="*60)

# Define the accuracy function
def accuracy_function(n):
    return 0.8 + 0.02*n - 0.001*n**2

# Define the objective function
def objective_function(n):
    return accuracy_function(n) - 0.1*n

# Find the optimal number of features
n_values = np.arange(0, 101)
accuracy_values = [accuracy_function(n) for n in n_values]
objective_values = [objective_function(n) for n in n_values]

# Find maximum using derivative
def derivative_objective(n):
    # d/dn [0.8 + 0.02n - 0.001n² - 0.1n]
    # = 0.02 - 0.002n - 0.1
    # = -0.08 - 0.002n
    return -0.08 - 0.002*n

# Set derivative to zero to find critical point
# -0.08 - 0.002n = 0
# -0.002n = 0.08
# n = -0.08 / 0.002 = -40
critical_point = -0.08 / 0.002

print(f"\nAnalytical solution using derivatives:")
print(f"• Objective function: f(n) = accuracy - 0.1n")
print(f"• Accuracy function: A(n) = 0.8 + 0.02n - 0.001n²")
print(f"• Therefore: f(n) = 0.8 + 0.02n - 0.001n² - 0.1n")
print(f"• Simplified: f(n) = 0.8 - 0.08n - 0.001n²")

print(f"\nDerivative calculation:")
print(f"• f'(n) = d/dn [0.8 - 0.08n - 0.001n²]")
print(f"• f'(n) = -0.08 - 0.002n")

print(f"\nSetting derivative to zero:")
print(f"• -0.08 - 0.002n = 0")
print(f"• -0.002n = 0.08")
print(f"• n = {critical_point}")

print(f"\nSince n = {critical_point} is negative and we need n ≥ 0,")
print(f"the optimal solution occurs at n = 0")

# Verify with numerical optimization
result = minimize_scalar(lambda x: -objective_function(x), bounds=(0, 100), method='bounded')
optimal_n = int(result.x)
max_objective = objective_function(optimal_n)

print(f"\nNumerical verification:")
print(f"• Optimal number of features: {optimal_n}")
print(f"• Maximum objective value: {max_objective:.4f}")
print(f"• Accuracy at optimal point: {accuracy_function(optimal_n):.4f}")

# Create visualization for Part 3
plt.figure(figsize=(12, 8))

# Plot accuracy function
plt.subplot(2, 2, 1)
plt.plot(n_values, accuracy_values, 'b-', linewidth=2, label='Accuracy(n)')
plt.axvline(x=optimal_n, color='r', linestyle='--', alpha=0.7, label=f'Optimal n = {optimal_n}')
plt.xlabel('Number of Features (n)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Number of Features')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot objective function
plt.subplot(2, 2, 2)
plt.plot(n_values, objective_values, 'g-', linewidth=2, label='f(n) = Accuracy - 0.1n')
plt.axvline(x=optimal_n, color='r', linestyle='--', alpha=0.7, label=f'Optimal n = {optimal_n}')
plt.axhline(y=max_objective, color='r', linestyle=':', alpha=0.7, label=f'Max f(n) = {max_objective:.4f}')
plt.xlabel('Number of Features (n)')
plt.ylabel('Objective Function f(n)')
plt.title('Objective Function vs. Number of Features')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot derivative
plt.subplot(2, 2, 3)
derivative_values = [derivative_objective(n) for n in n_values]
plt.plot(n_values, derivative_values, 'm-', linewidth=2, label="f'(n)")
plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
plt.axvline(x=critical_point, color='r', linestyle='--', alpha=0.7, label=f'Critical point n = {critical_point}')
plt.xlabel('Number of Features (n)')
plt.ylabel("Derivative f'(n)")
plt.title('Derivative of Objective Function')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot trade-off analysis
plt.subplot(2, 2, 4)
complexity_cost = 0.1 * n_values
plt.plot(n_values, accuracy_values, 'b-', linewidth=2, label='Accuracy')
plt.plot(n_values, complexity_cost, 'r-', linewidth=2, label='Complexity Cost (0.1n)')
plt.axvline(x=optimal_n, color='g', linestyle='--', alpha=0.7, label=f'Optimal n = {optimal_n}')
plt.xlabel('Number of Features (n)')
plt.ylabel('Value')
plt.title('Accuracy vs. Complexity Cost Trade-off')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'multi_objective_optimization.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# PART 4: Constrained Optimization
# ============================================================================
print("\n" + "="*60)
print("PART 4: CONSTRAINED OPTIMIZATION")
print("="*60)

# Define the new accuracy function
def accuracy_function_constrained(n):
    return 0.7 + 0.015*n - 0.0002*n**2

# Constraint: n ≤ 25
max_features_constraint = 25

# Find optimal within constraint
n_values_constrained = np.arange(0, max_features_constraint + 1)
accuracy_values_constrained = [accuracy_function_constrained(n) for n in n_values_constrained]

# Find maximum accuracy within constraint
optimal_n_constrained = np.argmax(accuracy_values_constrained)
max_accuracy_constrained = accuracy_values_constrained[optimal_n_constrained]

print(f"\nConstrained optimization problem:")
print(f"• Maximize: A(n) = 0.7 + 0.015n - 0.0002n²")
print(f"• Subject to: n ≤ {max_features_constraint}")

print(f"\nAnalytical solution:")
print(f"• A'(n) = d/dn [0.7 + 0.015n - 0.0002n²]")
print(f"• A'(n) = 0.015 - 0.0004n")

print(f"\nSetting derivative to zero:")
print(f"• 0.015 - 0.0004n = 0")
print(f"• 0.0004n = 0.015")
print(f"• n = {0.015/0.0004}")

unconstrained_optimal = 0.015 / 0.0004
print(f"\nUnconstrained optimal: n = {unconstrained_optimal}")

if unconstrained_optimal <= max_features_constraint:
    print(f"• Since {unconstrained_optimal} ≤ {max_features_constraint}, the constraint is not binding")
    print(f"• Optimal n = {unconstrained_optimal}")
else:
    print(f"• Since {unconstrained_optimal} > {max_features_constraint}, the constraint is binding")
    print(f"• Optimal n = {max_features_constraint}")

print(f"\nNumerical verification:")
print(f"• Optimal number of features: {optimal_n_constrained}")
print(f"• Maximum achievable accuracy: {max_accuracy_constrained:.4f}")

# Create visualization for Part 4
plt.figure(figsize=(12, 8))

# Plot accuracy function with constraint
plt.subplot(2, 2, 1)
plt.plot(n_values_constrained, accuracy_values_constrained, 'b-', linewidth=2, label='$A(n) = 0.7 + 0.015n - 0.0002n^2$')
plt.axvline(x=optimal_n_constrained, color='r', linestyle='--', alpha=0.7, label=f'Optimal n = {optimal_n_constrained}')
plt.axhline(y=max_accuracy_constrained, color='r', linestyle=':', alpha=0.7, label=f'Max A(n) = {max_accuracy_constrained:.4f}')
plt.xlabel('Number of Features (n)')
plt.ylabel('Accuracy A(n)')
plt.title('Accuracy vs. Number of Features (Constrained)')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot derivative
plt.subplot(2, 2, 2)
def derivative_accuracy_constrained(n):
    return 0.015 - 0.0004*n

derivative_values_constrained = [derivative_accuracy_constrained(n) for n in n_values_constrained]
plt.plot(n_values_constrained, derivative_values_constrained, 'm-', linewidth=2, label="A'(n)")
plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
plt.axvline(x=unconstrained_optimal, color='g', linestyle='--', alpha=0.7, label=f'Unconstrained optimal n = {unconstrained_optimal:.1f}')
plt.xlabel('Number of Features (n)')
plt.ylabel("Derivative A'(n)")
plt.title('Derivative of Accuracy Function')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot constraint visualization
plt.subplot(2, 2, 3)
plt.plot(n_values_constrained, accuracy_values_constrained, 'b-', linewidth=2, label='Accuracy')
plt.axvline(x=max_features_constraint, color='r', linestyle='-', linewidth=2, label=f'Constraint: n $\\leq$ {max_features_constraint}')
plt.fill_between(n_values_constrained, accuracy_values_constrained, alpha=0.3, color='lightblue')
plt.xlabel('Number of Features (n)')
plt.ylabel('Accuracy A(n)')
plt.title('Feasible Region (n $\\leq$ 25)')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot comparison of both problems
plt.subplot(2, 2, 4)
n_extended = np.arange(0, 101)
accuracy_extended = [accuracy_function_constrained(n) for n in n_extended]
plt.plot(n_extended, accuracy_extended, 'b-', linewidth=2, label='$A(n) = 0.7 + 0.015n - 0.0002n^2$')
plt.axvline(x=max_features_constraint, color='r', linestyle='-', linewidth=2, label=f'Constraint: n $\\leq$ {max_features_constraint}')
plt.axvline(x=unconstrained_optimal, color='g', linestyle='--', alpha=0.7, label=f'Unconstrained optimal n = {unconstrained_optimal:.1f}')
plt.axvline(x=optimal_n_constrained, color='orange', linestyle='--', alpha=0.7, label=f'Constrained optimal n = {optimal_n_constrained}')
plt.xlabel('Number of Features (n)')
plt.ylabel('Accuracy A(n)')
plt.title('Constrained vs. Unconstrained Optimization')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'constrained_optimization.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# SUMMARY AND CONCLUSIONS
# ============================================================================
print("\n" + "="*60)
print("SUMMARY AND CONCLUSIONS")
print("="*60)

print(f"\nPart 1: Objective Function")
print(f"• The objective function balances accuracy, complexity, and cost")
print(f"• Key trade-off: more features → higher accuracy but increased complexity")

print(f"\nPart 2: Budget Constraints")
print(f"• Maximum features possible with ${budget} budget: {max_features}")
print(f"• This represents a hard constraint on the optimization problem")

print(f"\nPart 3: Multi-objective Optimization")
print(f"• Optimal number of features: {optimal_n}")
print(f"• Maximum objective value: {max_objective:.4f}")
print(f"• The negative derivative at n=0 indicates diminishing returns")

print(f"\nPart 4: Constrained Optimization")
print(f"• Constraint: n <= {max_features_constraint}")
print(f"• Optimal number of features: {optimal_n_constrained}")
print(f"• Maximum achievable accuracy: {max_accuracy_constrained:.4f}")

print(f"\nKey Insights:")
print(f"• Feature selection is fundamentally a trade-off optimization problem")
print(f"• Budget constraints can significantly limit the solution space")
print(f"• The optimal solution often occurs at boundary points when constraints are binding")
print(f"• Derivative analysis provides analytical insights into optimal solutions")

print(f"\nPlots saved to: {save_dir}")
print("=" * 80)
