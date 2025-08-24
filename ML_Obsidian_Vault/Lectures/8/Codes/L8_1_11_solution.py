import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize_scalar
import pandas as pd

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_1_Quiz_11")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("QUESTION 11: FEATURE SELECTION OPTIMIZATION")
print("=" * 80)

# ============================================================================
# TASK 1: Objective Function for Feature Selection
# ============================================================================
print("\n" + "="*60)
print("TASK 1: Objective Function for Feature Selection")
print("="*60)

print("""
The objective function for feature selection typically involves multiple components:

1. Performance Metric (e.g., accuracy, F1-score, AUC)
2. Feature Count Penalty (to encourage sparsity)
3. Feature Cost Penalty (if applicable)
4. Regularization Terms (to prevent overfitting)

General Form:
Objective = Performance_Metric - λ₁ × Feature_Count - λ₂ × Feature_Cost - λ₃ × Regularization

Where λ₁, λ₂, λ₃ are hyperparameters that control the trade-offs.
""")

# ============================================================================
# TASK 2: Constraints in Feature Selection Optimization
# ============================================================================
print("\n" + "="*60)
print("TASK 2: Constraints in Feature Selection Optimization")
print("="*60)

print("""
Common constraints in feature selection optimization:

1. Feature Count Constraints:
   - Minimum features: F ≥ F_min
   - Maximum features: F ≤ F_max
   - Exact features: F = F_target

2. Performance Constraints:
   - Minimum accuracy: A ≥ A_min
   - Maximum error rate: E ≤ E_max

3. Budget Constraints:
   - Total feature cost ≤ Budget

4. Feature Dependencies:
   - If feature A is selected, feature B must also be selected
   - Mutual exclusivity: if feature A is selected, feature B cannot be selected

5. Domain Knowledge Constraints:
   - Certain features must be included
   - Certain features must be excluded
""")

# ============================================================================
# TASK 3: Balancing Multiple Objectives
# ============================================================================
print("\n" + "="*60)
print("TASK 3: Balancing Multiple Objectives")
print("="*60)

print("""
Strategies for balancing multiple objectives:

1. Weighted Sum Approach:
   - Combine objectives: w₁ × Accuracy + w₂ × (1/Feature_Count)
   - Requires setting appropriate weights

2. Pareto Frontier Approach:
   - Find all non-dominated solutions
   - Let decision maker choose based on preferences

3. Constraint-Based Approach:
   - Optimize one objective while constraining others
   - Example: Maximize accuracy subject to F ≤ 20

4. Multi-Objective Evolutionary Algorithms:
   - NSGA-II, MOEA/D
   - Find diverse set of solutions

5. Interactive Methods:
   - Start with one objective, gradually incorporate others
   - Adjust weights based on results
""")

# ============================================================================
# TASK 4: Mathematical Formulation
# ============================================================================
print("\n" + "="*60)
print("TASK 4: Mathematical Formulation")
print("="*60)

print("""
Mathematical formulation for maximizing accuracy while minimizing features:

Let A(F) be the accuracy function and F be the number of features.

1. Multi-Objective Formulation:
   Maximize: [A(F), -F]
   Subject to: F_min ≤ F ≤ F_max

2. Weighted Sum Formulation:
   Maximize: w₁ × A(F) - w₂ × F
   Subject to: F_min ≤ F ≤ F_max

3. Constraint-Based Formulation:
   Maximize: A(F)
   Subject to: F ≤ F_max
   Subject to: A(F) ≥ A_min

4. Penalty Method:
   Maximize: A(F) - λ × F
   Where λ controls the trade-off between accuracy and feature count
""")

# ============================================================================
# TASK 5: Comparison of Optimization Approaches
# ============================================================================
print("\n" + "="*60)
print("TASK 5: Comparison of Optimization Approaches")
print("="*60)

# Create comparison table
approaches_data = {
    'Approach': ['Greedy (Forward/Backward)', 'Genetic Algorithm', 'Exhaustive Search', 'Random Forest Importance', 'L1 Regularization'],
    'Pros': [
        'Fast, interpretable, good for large datasets',
        'Can find global optimum, handles non-linear relationships',
        'Guaranteed optimal solution',
        'Built-in feature importance, handles interactions',
        'Automatic feature selection, sparse solutions'
    ],
    'Cons': [
        'Local optimum, greedy decisions',
        'Computationally expensive, many parameters',
        'Computationally intractable for large feature sets',
        'Black box, may not find optimal subset',
        'May remove important correlated features'
    ],
    'Best For': [
        'Large datasets, quick solutions',
        'Complex feature interactions, global optimization',
        'Small feature sets, exact solutions',
        'Tree-based models, feature interactions',
        'Linear models, sparse solutions'
    ]
}

df = pd.DataFrame(approaches_data)
print("\nComparison of Feature Selection Optimization Approaches:")
print(df.to_string(index=False))

# ============================================================================
# TASK 6: Multi-Objective Optimization Problem
# ============================================================================
print("\n" + "="*60)
print("TASK 6: Multi-Objective Optimization Problem")
print("="*60)

# Given functions
def accuracy_function(F):
    """Accuracy as a function of feature count"""
    if F < 5:
        return 0.8  # Base accuracy for F < 5
    else:
        return 0.8 + 0.1 * np.log(F)

def objective_function(F):
    """Objective function to maximize: A - 0.05 * F"""
    return accuracy_function(F) - 0.05 * F

# Calculate objective values for given F values
F_values = [5, 10, 15]
print(f"\nGiven accuracy function: A = 0.8 + 0.1 × log(F) for F ≥ 5")
print(f"Objective function to maximize: A - 0.05 × F")
print(f"\nCalculating objective values:")

for F in F_values:
    A = accuracy_function(F)
    obj_val = objective_function(F)
    print(f"F = {F}: A = {A:.4f}, Objective = {obj_val:.4f}")

# Find optimal F using optimization
print(f"\nFinding optimal number of features...")

# Define the negative objective function (since we want to maximize)
def negative_objective(F):
    return -objective_function(F)

# Optimize over F ≥ 5
result = minimize_scalar(negative_objective, bounds=(5, 50), method='bounded')

optimal_F = result.x
optimal_objective = -result.fun
optimal_accuracy = accuracy_function(optimal_F)

print(f"Optimal number of features: F* = {optimal_F:.2f}")
print(f"Optimal accuracy: A* = {optimal_accuracy:.4f}")
print(f"Optimal objective value: {optimal_objective:.4f}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print(f"\nGenerating visualizations...")

# Create comprehensive plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Accuracy vs Feature Count
F_range = np.linspace(1, 20, 100)
A_range = [accuracy_function(F) for F in F_range]

ax1.plot(F_range, A_range, 'b-', linewidth=2, label='Accuracy Function')
ax1.scatter(F_values, [accuracy_function(F) for F in F_values], 
           color='red', s=100, zorder=5, label='Given Points')
ax1.scatter(optimal_F, optimal_accuracy, color='green', s=150, 
           marker='*', zorder=5, label=f'Optimal Point (F*={optimal_F:.2f})')
ax1.set_xlabel('Number of Features (F)')
ax1.set_ylabel('Accuracy (A)')
ax1.set_title('Accuracy vs Feature Count')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xlim(0, 20)

# Plot 2: Objective Function vs Feature Count
obj_range = [objective_function(F) for F in F_range]

ax2.plot(F_range, obj_range, 'r-', linewidth=2, label='Objective Function')
ax2.scatter(F_values, [objective_function(F) for F in F_values], 
           color='blue', s=100, zorder=5, label='Given Points')
ax2.scatter(optimal_F, optimal_objective, color='green', s=150, 
           marker='*', zorder=5, label=f'Optimal Point (F*={optimal_F:.2f})')
ax2.set_xlabel('Number of Features (F)')
ax2.set_ylabel('Objective Value (A - 0.05×F)')
ax2.set_title('Objective Function vs Feature Count')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_xlim(0, 20)

# Plot 3: Trade-off Analysis
# Show how different weights affect the optimal solution
lambda_values = [0.01, 0.05, 0.1, 0.2]
colors = ['blue', 'red', 'green', 'purple']

for i, lam in enumerate(lambda_values):
    def obj_with_lambda(F):
        return accuracy_function(F) - lam * F
    
    obj_lambda = [obj_with_lambda(F) for F in F_range]
    result_lambda = minimize_scalar(lambda F: -obj_with_lambda(F), bounds=(5, 50), method='bounded')
    
    ax3.plot(F_range, obj_lambda, color=colors[i], linewidth=2, 
             label=f'$\\lambda$ = {lam} (F* = {result_lambda.x:.2f})')

ax3.set_xlabel('Number of Features (F)')
ax3.set_ylabel('Objective Value')
ax3.set_title('Effect of Different Penalty Weights ($\\lambda$)')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.set_xlim(0, 20)

# Plot 4: 3D Surface Plot of Objective Function
F1, F2 = np.meshgrid(np.linspace(5, 20, 50), np.linspace(0.01, 0.2, 50))
Z = np.zeros_like(F1)

for i in range(F1.shape[0]):
    for j in range(F1.shape[1]):
        Z[i, j] = accuracy_function(F1[i, j]) - F2[i, j] * F1[i, j]

surf = ax4.contourf(F1, F2, Z, levels=20, cmap='viridis')
ax4.scatter(optimal_F, 0.05, color='red', s=200, 
           marker='*', zorder=5, label=f'Optimal (F*={optimal_F:.2f}, $\\lambda$=0.05)')
ax4.set_xlabel('Number of Features (F)')
ax4.set_ylabel('Penalty Weight ($\\lambda$)')
ax4.set_title('Objective Function Surface')
ax4.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_selection_optimization_analysis.png'), 
            dpi=300, bbox_inches='tight')

# Create detailed step-by-step solution visualization
plt.figure(figsize=(14, 10))

# Step-by-step solution
steps = [
    "Step 1: Define the accuracy function A(F) = 0.8 + 0.1 × log(F)",
    "Step 2: Define objective function: maximize A(F) - 0.05 × F",
    "Step 3: Calculate objective values at F = 5, 10, 15",
    "Step 4: Find optimal F by solving d/dF[A(F) - 0.05×F] = 0",
    "Step 5: Verify optimality and interpret results"
]

# Create step-by-step visualization
for i, step in enumerate(steps):
    plt.subplot(3, 2, i+1)
    
    if i == 0:  # Accuracy function
        plt.plot(F_range, A_range, 'b-', linewidth=2)
        plt.scatter(F_values, [accuracy_function(F) for F in F_values], 
                   color='red', s=100, zorder=5)
        plt.xlabel('F')
        plt.ylabel('A(F)')
        plt.title('Step 1: Accuracy Function')
        plt.grid(True, alpha=0.3)
        
    elif i == 1:  # Objective function
        plt.plot(F_range, obj_range, 'r-', linewidth=2)
        plt.scatter(F_values, [objective_function(F) for F in F_values], 
                   color='blue', s=100, zorder=5)
        plt.xlabel('F')
        plt.ylabel('Objective')
        plt.title('Step 2: Objective Function')
        plt.grid(True, alpha=0.3)
        
    elif i == 2:  # Numerical calculations
        plt.text(0.1, 0.8, f'F = 5: A = {accuracy_function(5):.4f}, Obj = {objective_function(5):.4f}', 
                transform=plt.gca().transAxes, fontsize=10)
        plt.text(0.1, 0.6, f'F = 10: A = {accuracy_function(10):.4f}, Obj = {objective_function(10):.4f}', 
                transform=plt.gca().transAxes, fontsize=10)
        plt.text(0.1, 0.4, f'F = 15: A = {accuracy_function(15):.4f}, Obj = {objective_function(15):.4f}', 
                transform=plt.gca().transAxes, fontsize=10)
        plt.title('Step 3: Numerical Calculations')
        plt.axis('off')
        
    elif i == 3:  # Mathematical solution
        # Show the derivative and solution
        plt.text(0.1, 0.8, 'd/dF[A(F) - 0.05×F] = 0', transform=plt.gca().transAxes, fontsize=12)
        plt.text(0.1, 0.6, 'd/dF[0.8 + 0.1×log(F) - 0.05×F] = 0', transform=plt.gca().transAxes, fontsize=10)
        plt.text(0.1, 0.4, '0.1/F - 0.05 = 0', transform=plt.gca().transAxes, fontsize=10)
        plt.text(0.1, 0.2, 'F = 0.1/0.05 = 2', transform=plt.gca().transAxes, fontsize=10)
        plt.text(0.1, 0.0, 'But F $\\geq$ 5, so F* = 5', transform=plt.gca().transAxes, fontsize=10)
        plt.title('Step 4: Mathematical Solution')
        plt.axis('off')
        
    elif i == 4:  # Results
        plt.text(0.1, 0.8, f'Optimal F* = {optimal_F:.2f}', transform=plt.gca().transAxes, fontsize=12)
        plt.text(0.1, 0.6, f'Optimal A* = {optimal_accuracy:.4f}', transform=plt.gca().transAxes, fontsize=12)
        plt.text(0.1, 0.4, f'Optimal Objective = {optimal_objective:.4f}', transform=plt.gca().transAxes, fontsize=12)
        plt.text(0.1, 0.2, 'This gives the best balance', transform=plt.gca().transAxes, fontsize=10)
        plt.text(0.1, 0.0, 'between accuracy and feature count', transform=plt.gca().transAxes, fontsize=10)
        plt.title('Step 5: Results and Interpretation')
        plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'step_by_step_solution.png'), dpi=300, bbox_inches='tight')

# Create optimization approaches comparison visualization
plt.figure(figsize=(15, 10))

# Simulate performance of different approaches
np.random.seed(42)
F_sizes = np.arange(5, 21)
n_approaches = 5

# Simulate different optimization approaches
approaches = ['Greedy', 'Genetic', 'Exhaustive', 'Random Forest', 'L1 Regularization']
colors = ['blue', 'red', 'green', 'orange', 'purple']

for i, approach in enumerate(approaches):
    if approach == 'Greedy':
        # Greedy: good performance, fast
        performance = 0.85 - 0.02 * np.random.randn(len(F_sizes))
    elif approach == 'Genetic':
        # Genetic: better performance, slower
        performance = 0.87 - 0.01 * np.random.randn(len(F_sizes))
    elif approach == 'Exhaustive':
        # Exhaustive: best performance, very slow
        performance = 0.89 - 0.005 * np.random.randn(len(F_sizes))
    elif approach == 'Random Forest':
        # Random Forest: moderate performance
        performance = 0.86 - 0.015 * np.random.randn(len(F_sizes))
    else:  # L1 Regularization
        # L1: good performance, automatic
        performance = 0.84 - 0.025 * np.random.randn(len(F_sizes))
    
    plt.subplot(2, 3, i+1)
    plt.plot(F_sizes, performance, color=colors[i], linewidth=2, label=approach)
    plt.scatter(F_sizes, performance, color=colors[i], s=50)
    plt.xlabel('Feature Count')
    plt.ylabel('Performance')
    plt.title(f'{approach} Approach')
    plt.grid(True, alpha=0.3)
    plt.ylim(0.8, 0.9)

# Add overall comparison
plt.subplot(2, 3, 6)
for i, approach in enumerate(approaches):
    if approach == 'Greedy':
        performance = 0.85 - 0.02 * np.random.randn(len(F_sizes))
    elif approach == 'Genetic':
        performance = 0.87 - 0.01 * np.random.randn(len(F_sizes))
    elif approach == 'Exhaustive':
        performance = 0.89 - 0.005 * np.random.randn(len(F_sizes))
    elif approach == 'Random Forest':
        performance = 0.86 - 0.015 * np.random.randn(len(F_sizes))
    else:
        performance = 0.84 - 0.025 * np.random.randn(len(F_sizes))
    
    plt.plot(F_sizes, performance, color=colors[i], linewidth=2, label=approach)

plt.xlabel('Feature Count')
plt.ylabel('Performance')
plt.title('Overall Comparison')
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(0.8, 0.9)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'optimization_approaches_comparison.png'), dpi=300, bbox_inches='tight')

print(f"\nAll visualizations saved to: {save_dir}")

# ============================================================================
# SUMMARY OF RESULTS
# ============================================================================
print("\n" + "="*80)
print("SUMMARY OF RESULTS")
print("="*80)

print(f"""
1. Objective Function: A(F) - λ×F where A(F) = 0.8 + 0.1×log(F) and λ = 0.05

2. Constraints: F ≥ 5 (minimum feature requirement)

3. Optimization Approaches:
   - Greedy: Fast but may find local optimum
   - Genetic: Better global search, more computationally expensive
   - Exhaustive: Guaranteed optimal but computationally intractable for large F
   - Random Forest: Built-in importance, handles interactions
   - L1 Regularization: Automatic selection, sparse solutions

4. Mathematical Formulation:
   - Multi-objective: Maximize [A(F), -F]
   - Weighted sum: Maximize A(F) - 0.05×F
   - Constraint-based: Maximize A(F) subject to F ≤ F_max

5. Optimal Solution:
   - Optimal F* = {optimal_F:.2f}
   - Optimal A* = {optimal_accuracy:.4f}
   - Optimal objective value = {optimal_objective:.4f}

6. Key Insights:
   - The log function provides diminishing returns for accuracy
   - The linear penalty encourages fewer features
   - The optimal solution balances these competing objectives
   - Different penalty weights (λ) lead to different optimal solutions
""")

print(f"\nDetailed analysis complete! Check the generated images in: {save_dir}")
