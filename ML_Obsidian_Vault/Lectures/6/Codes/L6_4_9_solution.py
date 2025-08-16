import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_4_Quiz_9")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Set matplotlib to non-interactive backend to avoid displaying plots
plt.ioff()

print("Question 9: MDL-Based Pruning")
print("=" * 80)
print("COMPREHENSIVE ANALYSIS OF MINIMUM DESCRIPTION LENGTH PRINCIPLE")
print("=" * 80)

# ============================================================================
# PEN AND PAPER MATHEMATICAL SOLUTIONS WITH SYMBOLIC CALCULATIONS
# ============================================================================

print("\n" + "="*80)
print("PEN AND PAPER MATHEMATICAL SOLUTIONS WITH SYMBOLIC CALCULATIONS")
print("="*80)

# Mathematical Foundation: MDL Principle
print("\n1. MATHEMATICAL FOUNDATION: MDL Principle")
print("-" * 70)

print("MDL Principle Definition:")
print("   The Minimum Description Length principle states that the best model")
print("   minimizes the total description length: $L(M) + L(D|M)$")
print("   where:")
print("   - $L(M)$ is the description length of the model")
print("   - $L(D|M)$ is the description length of the data given the model")

print("\nDescription Length Components:")
print("   For a decision tree:")
print("   - $L(M) = L_{\\text{structure}} + L_{\\text{parameters}}$")
print("   - $L_{\\text{structure}} = \\text{Number of nodes} \\times \\log_2(\\text{Number of features})$")
print("   - $L_{\\text{parameters}} = \\sum_{\\text{nodes}} \\log_2(\\text{Number of split values})$")

print("\nMDL Score Formula:")
print("   $\\text{MDL Score} = L(M) + L(D|M)$")
print("   where $L(D|M) = -\\log_2 P(D|M)$ (negative log-likelihood)")

print("\nOptimal Tree Selection:")
print("   $T^* = \\arg\\min_{T} \\{\\text{MDL Score}(T)\\}$")
print("   This balances model complexity with data fitting")

# ============================================================================
# TASK 1: EXPLAIN HOW MDL BALANCES MODEL COMPLEXITY AND ACCURACY
# ============================================================================

print("\n" + "="*80)
print("TASK 1: EXPLAIN HOW MDL BALANCES MODEL COMPLEXITY AND ACCURACY")
print("=" * 80)

print("\nPEN AND PAPER SOLUTION:")
print("-" * 50)

print("Step 1: Understand the trade-off")
print("   MDL balances two competing objectives:")
print("   - Model simplicity: Minimize $L(M)$")
print("   - Data fitting: Minimize $L(D|M)$")

print("\nStep 2: Mathematical formulation")
print("   Total cost = Model complexity + Data misfit")
print("   $C_{\\text{total}} = \\alpha \\cdot L(M) + \\beta \\cdot L(D|M)$")
print("   where $\\alpha$ and $\\beta$ are weighting factors")

print("\nStep 3: Optimal balance")
print("   The optimal model satisfies:")
print("   $\\frac{\\partial C_{\\text{total}}}{\\partial L(M)} = \\alpha - \\beta \\cdot \\frac{\\partial L(D|M)}{\\partial L(M)} = 0$")
print("   This gives: $\\alpha = \\beta \\cdot \\frac{\\partial L(D|M)}{\\partial L(M)}$")

print("\nStep 4: Interpretation")
print("   - Simple models: Low $L(M)$, potentially high $L(D|M)$")
print("   - Complex models: High $L(M)$, potentially low $L(D|M)$")
print("   - MDL finds the sweet spot where adding complexity")
print("     doesn't significantly improve data fitting")

# ============================================================================
# TASK 2: ESTIMATE DESCRIPTION LENGTH FOR TREE WITH 5 NODES
# ============================================================================

print("\n" + "="*80)
print("TASK 2: ESTIMATE DESCRIPTION LENGTH FOR TREE WITH 5 NODES")
print("=" * 80)

print("\nPEN AND PAPER SOLUTION:")
print("-" * 50)

print("Step 1: Define parameters")
print("   Given: Tree with 5 nodes")
print("   Assume: 10 features, binary splits (2 values per feature)")
print("   Assume: 1000 training samples")

print("\nStep 2: Calculate structure description length")
print("   $L_{\\text{structure}} = 5 \\times \\log_2(10) = 5 \\times 3.32 = 16.6$ bits")

print("\nStep 3: Calculate parameter description length")
print("   $L_{\\text{parameters}} = 5 \\times \\log_2(2) = 5 \\times 1 = 5$ bits")

print("\nStep 4: Calculate data description length")
print("   For binary classification with 5 leaf nodes:")
print("   $L(D|M) = -\\sum_{i=1}^{5} n_i \\log_2(p_i)$")
print("   where $n_i$ is samples in leaf $i$, $p_i$ is predicted probability")
print("   Assuming uniform distribution: $L(D|M) \\approx 1000 \\times \\log_2(5) = 2322$ bits")

print("\nStep 5: Total description length")
print("   $L_{\\text{total}} = 16.6 + 5 + 2322 = 2343.6$ bits")

# ============================================================================
# TASK 3: DESCRIBE HOW MDL PENALIZES OVERLY COMPLEX TREES
# ============================================================================

print("\n" + "="*80)
print("TASK 3: DESCRIBE HOW MDL PENALIZES OVERLY COMPLEX TREES")
print("=" * 80)

print("\nPEN AND PAPER SOLUTION:")
print("-" * 50)

print("Step 1: Complexity penalty mechanism")
print("   MDL penalizes complexity through $L(M)$:")
print("   - More nodes → Higher $L_{\\text{structure}}$")
print("   - More split values → Higher $L_{\\text{parameters}}$")

print("\nStep 2: Mathematical penalty")
print("   For a tree with $n$ nodes and $f$ features:")
print("   $L_{\\text{structure}} = n \\cdot \\log_2(f)$")
print("   $L_{\\text{parameters}} = n \\cdot \\log_2(\\text{avg split values})$")

print("\nStep 3: Penalty growth")
print("   As $n$ increases:")
print("   - $L(M)$ grows linearly: $O(n)$")
print("   - $L(D|M)$ may decrease but with diminishing returns")
print("   - Total cost eventually increases due to complexity penalty")

print("\nStep 4: Optimal complexity")
print("   The optimal tree size satisfies:")
print("   $\\frac{\\partial L(M)}{\\partial n} = -\\frac{\\partial L(D|M)}{\\partial n}$")
print("   This is the point where adding nodes doesn't improve the trade-off")

# ============================================================================
# TASK 4: LIST MAIN ADVANTAGES OF MDL-BASED PRUNING
# ============================================================================

print("\n" + "="*80)
print("TASK 4: LIST MAIN ADVANTAGES OF MDL-BASED PRUNING")
print("=" * 80)

print("\nPEN AND PAPER SOLUTION:")
print("-" * 50)

print("Step 1: Theoretical advantages")
print("   - Based on solid information theory principles")
print("   - Provides principled model selection")
print("   - Automatically balances complexity vs. accuracy")

print("\nStep 2: Practical advantages")
print("   - No need for cross-validation")
print("   - Computationally efficient")
print("   - Handles different data types uniformly")

print("\nStep 3: Statistical advantages")
print("   - Prevents overfitting through complexity penalty")
print("   - Provides confidence in model selection")
print("   - Robust to different data distributions")

print("\nStep 4: Implementation advantages")
print("   - Easy to implement and understand")
print("   - Scales well with data size")
print("   - Provides interpretable results")

# ============================================================================
# TASK 5: MDL SUGGESTION FOR SPLIT WITH 2 UNIQUE VALUES
# ============================================================================

print("\n" + "="*80)
print("TASK 5: MDL SUGGESTION FOR SPLIT WITH 2 UNIQUE VALUES")
print("=" * 80)

print("\nPEN AND PAPER SOLUTION:")
print("-" * 50)

print("Step 1: Analyze the split")
print("   Feature with only 2 unique values:")
print("   - Low information content: $\\log_2(2) = 1$ bit")
print("   - Simple parameter description: $L_{\\text{param}} = 1$ bit")

print("\nStep 2: MDL evaluation")
print("   For this split:")
print("   - $L_{\\text{structure}} = 1 \\times \\log_2(f)$")
print("   - $L_{\\text{parameters}} = 1$ bit")
print("   - Total model cost: $\\log_2(f) + 1$ bits")

print("\nStep 3: MDL recommendation")
print("   MDL suggests keeping this split if:")
print("   $L(D|M_{\\text{with split}}) + L(M_{\\text{with split}}) < L(D|M_{\\text{without split}}) + L(M_{\\text{without split}})$")

print("\nStep 4: Decision criteria")
print("   Keep the split if the reduction in $L(D|M)$ exceeds")
print("   the increase in $L(M)$: $\\Delta L(D|M) > \\Delta L(M)$")

# ============================================================================
# TASK 6: MDL FOR BANDWIDTH OPTIMIZATION
# ============================================================================

print("\n" + "="*80)
print("TASK 6: MDL FOR BANDWIDTH OPTIMIZATION")
print("=" * 80)

print("\nPEN AND PAPER SOLUTION:")
print("-" * 50)

print("Step 1: Bandwidth constraint formulation")
print("   For limited bandwidth, we want:")
print("   $L(M) \\leq B_{\\text{max}}$ (bandwidth constraint)")
print("   while minimizing $L(D|M)$")

print("\nStep 2: Constrained optimization")
print("   $\\min_{M} L(D|M)$")
print("   subject to: $L(M) \\leq B_{\\text{max}}$")

print("\nStep 3: Lagrangian formulation")
print("   $\\mathcal{L} = L(D|M) + \\lambda(L(M) - B_{\\text{max}})$")
print("   where $\\lambda$ is the Lagrange multiplier")

print("\nStep 4: Optimal solution")
print("   The optimal tree satisfies:")
print("   $\\frac{\\partial L(D|M)}{\\partial L(M)} = -\\lambda$")
print("   This gives the optimal complexity for the given bandwidth")

# ============================================================================
# TASK 7: CALCULATE DESCRIPTION LENGTH PENALTY
# ============================================================================

print("\n" + "="*80)
print("TASK 7: CALCULATE DESCRIPTION LENGTH PENALTY")
print("=" * 80)

print("\nPEN AND PAPER SOLUTION:")
print("-" * 50)

print("Step 1: Define the problem")
print("   Tree grows from 3 to 7 nodes")
print("   Assume: 10 features, binary splits")

print("\nStep 2: Calculate structure penalty")
print("   $L_{\\text{structure}}(3) = 3 \\times \\log_2(10) = 9.96$ bits")
print("   $L_{\\text{structure}}(7) = 7 \\times \\log_2(10) = 23.24$ bits")
print("   $\\Delta L_{\\text{structure}} = 23.24 - 9.96 = 13.28$ bits")

print("\nStep 3: Calculate parameter penalty")
print("   $L_{\\text{parameters}}(3) = 3 \\times \\log_2(2) = 3$ bits")
print("   $L_{\\text{parameters}}(7) = 7 \\times \\log_2(2) = 7$ bits")
print("   $\\Delta L_{\\text{parameters}} = 7 - 3 = 4$ bits")

print("\nStep 4: Total penalty")
print("   Total description length penalty:")
print("   $\\Delta L(M) = 13.28 + 4 = 17.28$ bits")

print("\nStep 5: Interpretation")
print("   Adding 4 nodes increases the model description")
print("   by 17.28 bits. This penalty must be justified by")
print("   a corresponding reduction in $L(D|M)$")

# ============================================================================
# TASK 8: BIAS-VARIANCE DECOMPOSITION AND MDL
# ============================================================================

print("\n" + "="*80)
print("TASK 8: BIAS-VARIANCE DECOMPOSITION AND MDL")
print("=" * 80)

print("\nPEN AND PAPER SOLUTION:")
print("-" * 50)

print("Step 1: Bias-variance decomposition")
print("   $E[\\text{Error}] = \\text{Bias}^2 + \\text{Variance} + \\text{Irreducible Error}$")
print("   where $\\text{Variance} = E[(\\hat{f}(x) - E[\\hat{f}(x)])^2]$")

print("\nStep 2: MDL effect on variance")
print("   MDL-based pruning reduces model complexity:")
print("   - Fewer parameters → Less sensitivity to data variations")
print("   - More stable predictions → Lower variance")

print("\nStep 3: Mathematical relationship")
print("   For a tree with $n$ nodes:")
print("   $\\text{Variance} \\propto \\frac{1}{n}$ (approximately)")
print("   Pruning from $n_1$ to $n_2 < n_1$ nodes:")
print("   $\\Delta \\text{Variance} = \\frac{1}{n_2} - \\frac{1}{n_1} > 0$")

print("\nStep 4: Trade-off analysis")
print("   MDL pruning increases bias but decreases variance:")
print("   - $\\Delta \\text{Bias}^2 > 0$ (bias increases)")
print("   - $\\Delta \\text{Variance} < 0$ (variance decreases)")
print("   - Optimal when $|\\Delta \\text{Bias}^2| < |\\Delta \\text{Variance}|$")

# ============================================================================
# VISUALIZATION AND PRACTICAL EXAMPLES
# ============================================================================

print("\n" + "="*80)
print("VISUALIZATION AND PRACTICAL EXAMPLES")
print("=" * 80)

# Generate synthetic data for demonstration
np.random.seed(42)
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                          n_redundant=3, n_clusters_per_class=1, random_state=42)

# Create trees of different complexities
depths = [3, 5, 7, 10, 15]
mdl_scores = []
accuracies = []
complexities = []

for depth in depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X, y)
    
    # Calculate accuracy
    y_pred = tree.predict(X)
    accuracy = accuracy_score(y, y_pred)
    accuracies.append(accuracy)
    
    # Calculate complexity (number of nodes)
    n_nodes = tree.tree_.node_count
    complexities.append(n_nodes)
    
    # Calculate MDL score (simplified)
    # L(M) = structure + parameters
    n_features = X.shape[1]
    structure_cost = n_nodes * np.log2(n_features)
    param_cost = n_nodes * np.log2(2)  # binary splits
    model_cost = structure_cost + param_cost
    
    # L(D|M) = negative log-likelihood (simplified)
    # For binary classification: -sum(n_i * log(p_i))
    # Simplified as: -log(accuracy) * n_samples
    data_cost = -np.log2(accuracy) * len(X) if accuracy > 0 else 1000
    
    mdl_score = model_cost + data_cost
    mdl_scores.append(mdl_score)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('MDL-Based Pruning Analysis', fontsize=16, fontweight='bold')

# Plot 1: MDL Score vs Tree Complexity
axes[0, 0].plot(complexities, mdl_scores, 'bo-', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('Number of Nodes')
axes[0, 0].set_ylabel('MDL Score')
axes[0, 0].set_title('MDL Score vs Tree Complexity')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].annotate('Optimal\nComplexity', 
                     xy=(complexities[np.argmin(mdl_scores)], min(mdl_scores)),
                     xytext=(complexities[np.argmin(mdl_scores)] + 2, min(mdl_scores) + 100),
                     arrowprops=dict(arrowstyle='->', color='red', lw=2),
                     fontsize=10, ha='center')

# Plot 2: Accuracy vs Complexity
axes[0, 1].plot(complexities, accuracies, 'ro-', linewidth=2, markersize=8)
axes[0, 1].set_xlabel('Number of Nodes')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Accuracy vs Tree Complexity')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Perfect Accuracy')
axes[0, 1].legend()

# Plot 3: Model Cost vs Data Cost
model_costs = [complexities[i] * (np.log2(10) + np.log2(2)) for i in range(len(complexities))]
data_costs = [mdl_scores[i] - model_costs[i] for i in range(len(complexities))]

axes[1, 0].plot(complexities, model_costs, 'go-', linewidth=2, markersize=8, label='Model Cost L(M)')
axes[1, 0].plot(complexities, data_costs, 'mo-', linewidth=2, markersize=8, label='Data Cost L(D|M)')
axes[1, 0].set_xlabel('Number of Nodes')
axes[1, 0].set_ylabel('Cost (bits)')
axes[1, 0].set_title('Model Cost vs Data Cost')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# Plot 4: Trade-off Analysis
axes[1, 1].scatter(model_costs, data_costs, c=complexities, cmap='viridis', s=100, alpha=0.7)
axes[1, 1].set_xlabel('Model Cost L(M)')
axes[1, 1].set_ylabel('Data Cost L(D|M)')
axes[1, 1].set_title('MDL Trade-off: Model vs Data Cost')
axes[1, 1].grid(True, alpha=0.3)

# Add colorbar
scatter = axes[1, 1].scatter(model_costs, data_costs, c=complexities, cmap='viridis', s=100, alpha=0.7)
cbar = plt.colorbar(scatter, ax=axes[1, 1])
cbar.set_label('Number of Nodes')

# Highlight optimal point
optimal_idx = np.argmin(mdl_scores)
axes[1, 1].scatter(model_costs[optimal_idx], data_costs[optimal_idx], 
                    c='red', s=200, marker='*', edgecolors='black', linewidth=2, 
                    label=f'Optimal ({complexities[optimal_idx]} nodes)')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'mdl_analysis.png'), dpi=300, bbox_inches='tight')

# Create detailed MDL calculation table
print("\n" + "="*80)
print("DETAILED MDL CALCULATION TABLE")
print("=" * 80)

print(f"{'Depth':<6} {'Nodes':<6} {'Structure':<12} {'Params':<8} {'Model Cost':<12} {'Data Cost':<12} {'Total MDL':<12}")
print("-" * 80)

for i, depth in enumerate(depths):
    structure_cost = complexities[i] * np.log2(10)
    param_cost = complexities[i] * np.log2(2)
    model_cost = structure_cost + param_cost
    data_cost = mdl_scores[i] - model_cost
    
    print(f"{depth:<6} {complexities[i]:<6} {structure_cost:<12.2f} {param_cost:<8.2f} "
          f"{model_cost:<12.2f} {data_cost:<12.2f} {mdl_scores[i]:<12.2f}")

# Find optimal tree
optimal_idx = np.argmin(mdl_scores)
print(f"\nOptimal tree: Depth {depths[optimal_idx]}, {complexities[optimal_idx]} nodes")
print(f"MDL Score: {mdl_scores[optimal_idx]:.2f} bits")
print(f"Accuracy: {accuracies[optimal_idx]:.4f}")

# Create bias-variance analysis visualization
fig, ax = plt.subplots(figsize=(12, 8))

# Simulate bias-variance trade-off
complexities_fine = np.linspace(1, 20, 100)
bias_squared = 0.1 + 0.05 / complexities_fine  # Bias decreases with complexity
variance = 0.02 * complexities_fine  # Variance increases with complexity
total_error = bias_squared + variance

ax.plot(complexities_fine, bias_squared, 'b-', linewidth=2, label='Bias²')
ax.plot(complexities_fine, variance, 'r-', linewidth=2, label='Variance')
ax.plot(complexities_fine, total_error, 'g-', linewidth=3, label='Total Error')

# Mark optimal complexity
optimal_complexity = complexities_fine[np.argmin(total_error)]
ax.axvline(x=optimal_complexity, color='purple', linestyle='--', alpha=0.7, 
           label=f'Optimal Complexity ({optimal_complexity:.1f})')

ax.set_xlabel('Tree Complexity (Number of Nodes)')
ax.set_ylabel('Error Components')
ax.set_title('Bias-Variance Trade-off in Decision Trees')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_ylim(0, 0.5)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'bias_variance_tradeoff.png'), dpi=300, bbox_inches='tight')

# Create MDL penalty visualization
fig, ax = plt.subplots(figsize=(10, 6))

# Show penalty for growing from 3 to 7 nodes
nodes_3 = 3
nodes_7 = 7
penalty_3 = nodes_3 * (np.log2(10) + np.log2(2))
penalty_7 = nodes_7 * (np.log2(10) + np.log2(2))
penalty_increase = penalty_7 - penalty_3

x_pos = [0, 1]
penalties = [penalty_3, penalty_7]
labels = [f'{nodes_3} nodes', f'{nodes_7} nodes']

bars = ax.bar(x_pos, penalties, color=['lightblue', 'lightcoral'], alpha=0.7, edgecolor='black')
ax.set_xlabel('Tree Configuration')
ax.set_ylabel('Description Length (bits)')
ax.set_title('MDL Penalty: Growing from 3 to 7 Nodes')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)

# Add value labels on bars
for i, (bar, penalty) in enumerate(zip(bars, penalties)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
            f'{penalty:.1f}', ha='center', va='bottom', fontweight='bold')

# Add penalty increase arrow
ax.annotate(f'Penalty Increase:\n{penalty_increase:.1f} bits', 
            xy=(0.5, (penalty_3 + penalty_7)/2), xytext=(0.5, (penalty_3 + penalty_7)/2 + 50),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2),
            ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'mdl_penalty_visualization.png'), dpi=300, bbox_inches='tight')

print(f"\nPlots saved to: {save_dir}")

# ============================================================================
# SUMMARY OF KEY INSIGHTS
# ============================================================================

print("\n" + "="*80)
print("SUMMARY OF KEY INSIGHTS")
print("=" * 80)

print("\n1. MDL Principle:")
print("   - Balances model complexity with data fitting")
print("   - Provides principled model selection")
print("   - Automatically prevents overfitting")

print("\n2. Description Length Components:")
print("   - Structure cost: $O(n \\log_2(f))$ where $n$ is nodes, $f$ is features")
print("   - Parameter cost: $O(n \\log_2(s))$ where $s$ is split values")
print("   - Data cost: Negative log-likelihood of data given model")

print("\n3. Optimal Complexity:")
print("   - Found by minimizing total MDL score")
print("   - Balances bias-variance trade-off")
print("   - Provides confidence in model selection")

print("\n4. Practical Applications:")
print("   - Bandwidth-constrained systems")
print("   - Interpretable models")
print("   - Robust model selection")

print("\n5. Mathematical Properties:")
print("   - Complexity penalty grows linearly with nodes")
print("   - Data fitting improvement has diminishing returns")
print("   - Optimal point where marginal benefits equal marginal costs")
