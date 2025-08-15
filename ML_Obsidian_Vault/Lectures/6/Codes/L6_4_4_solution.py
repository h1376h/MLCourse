import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_4_Quiz_4")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = False  # Disable LaTeX to avoid issues
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("=" * 80)
print("QUESTION 4: CART's Cost-Complexity Pruning Analysis")
print("=" * 80)

# ============================================================================
# PART 1: Cost-Complexity Function Definition
# ============================================================================
print("\n" + "="*60)
print("PART 1: Cost-Complexity Function Definition")
print("="*60)

print("The cost-complexity function is defined as:")
print("R_α(T) = R(T) + α|T|")
print("where:")
print("  R(T) = Total misclassification cost")
print("  |T| = Number of nodes in tree T")
print("  α = Complexity parameter (penalty for tree size)")

# ============================================================================
# PART 2: Cost Calculation for α = 0.1
# ============================================================================
print("\n" + "="*60)
print("PART 2: Cost Calculation for α = 0.1")
print("="*60)

alpha_1 = 0.1
nodes_1 = 7
error_1 = 0.3

# Calculate total misclassification cost
# Assuming equal class distribution and cost matrix
fp_cost = 10  # False positive cost
fn_cost = 100  # False negative cost
total_cost = error_1 * (fp_cost + fn_cost) / 2  # Average cost per misclassification

# Calculate cost-complexity
R_alpha_1 = total_cost + alpha_1 * nodes_1

print(f"Given:")
print(f"  α = {alpha_1}")
print(f"  |T| = {nodes_1} nodes")
print(f"  R(T) = {error_1} (misclassification rate)")
print(f"  False positive cost = ${fp_cost}")
print(f"  False negative cost = ${fn_cost}")
print()
print(f"Step-by-step calculation:")
print(f"  1. Total misclassification cost R(T) = {error_1} × (${fp_cost} + ${fn_cost})/2")
print(f"     R(T) = {error_1} × ${(fp_cost + fn_cost)/2} = ${total_cost:.1f}")
print(f"  2. Complexity penalty = α × |T| = {alpha_1} × {nodes_1} = {alpha_1 * nodes_1}")
print(f"  3. Cost-complexity R_α(T) = R(T) + α|T|")
print(f"     R_α(T) = ${total_cost:.1f} + {alpha_1 * nodes_1} = ${R_alpha_1:.1f}")

# ============================================================================
# PART 3: Tree Comparison for α = 0.05
# ============================================================================
print("\n" + "="*60)
print("PART 3: Tree Comparison for α = 0.05")
print("="*60)

alpha_2 = 0.05
tree1_nodes = 5
tree1_error = 0.35
tree2_nodes = 3
tree2_error = 0.40

# Calculate costs for both trees
R_alpha_tree1 = total_cost * (tree1_error / error_1) + alpha_2 * tree1_nodes
R_alpha_tree2 = total_cost * (tree2_error / error_1) + alpha_2 * tree2_nodes

print(f"Comparing two trees with α = {alpha_2}:")
print()
print(f"Tree 1: {tree1_nodes} nodes, error rate = {tree1_error}")
print(f"Tree 2: {tree2_nodes} nodes, error rate = {tree2_error}")
print()
print(f"Calculations:")
print(f"  Tree 1: R_α(T₁) = R(T₁) + α|T₁|")
print(f"           R_α(T₁) = {total_cost * (tree1_error / error_1):.1f} + {alpha_2} × {tree1_nodes}")
print(f"           R_α(T₁) = {total_cost * (tree1_error / error_1):.1f} + {alpha_2 * tree1_nodes}")
print(f"           R_α(T₁) = {R_alpha_tree1:.1f}")
print()
print(f"  Tree 2: R_α(T₂) = R(T₂) + α|T₂|")
print(f"           R_α(T₂) = {total_cost * (tree2_error / error_1):.1f} + {alpha_2} × {tree2_nodes}")
print(f"           R_α(T₂) = {total_cost * (tree2_error / error_1):.1f} + {alpha_2 * tree2_nodes}")
print(f"           R_α(T₂) = {R_alpha_tree2:.1f}")
print()
print(f"Decision: Tree {'1' if R_alpha_tree1 < R_alpha_tree2 else '2'} is preferred")
print(f"          (lower cost-complexity value)")

# ============================================================================
# PART 4: Relationship between α and Tree Complexity
# ============================================================================
print("\n" + "="*60)
print("PART 4: Relationship between α and Tree Complexity")
print("="*60)

print("The relationship between α and tree complexity:")
print("• α controls the trade-off between accuracy and model simplicity")
print("• Higher α values penalize complex trees more heavily")
print("• Lower α values allow more complex trees for better accuracy")
print("• Optimal α balances overfitting vs. underfitting")

# Create visualization for this relationship
alphas = np.linspace(0, 0.5, 100)
tree_sizes = np.array([3, 5, 7, 10, 15])
error_rates = np.array([0.40, 0.35, 0.30, 0.28, 0.25])

plt.figure(figsize=(12, 8))

# Plot cost-complexity curves for different tree sizes
for i, (size, error) in enumerate(zip(tree_sizes, error_rates)):
    costs = total_cost * (error / error_1) + alphas * size
    plt.plot(alphas, costs, label=f'Tree: {size} nodes, error={error}', linewidth=2)

plt.xlabel(r'Complexity Parameter α')
plt.ylabel(r'Cost-Complexity R_α(T)')
plt.title(r'Cost-Complexity vs. α for Different Tree Sizes')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'cost_complexity_vs_alpha.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# PART 5: Optimal α with Operational Costs
# ============================================================================
print("\n" + "="*60)
print("PART 5: Optimal α with Operational Costs")
print("="*60)

operational_cost_per_node = 5
print(f"Operational cost per node: ${operational_cost_per_node}")

# Calculate total cost including operational costs
def total_business_cost(alpha, nodes, error_rate, base_cost, op_cost_per_node):
    misclassification_cost = base_cost * (error_rate / error_1)
    complexity_penalty = alpha * nodes
    operational_cost = op_cost_per_node * nodes
    return misclassification_cost + complexity_penalty + operational_cost

# Test different α values
alpha_values = [0.01, 0.05, 0.1, 0.2, 0.3]
tree_configs = [
    (3, 0.40),
    (5, 0.35),
    (7, 0.30),
    (10, 0.28),
    (15, 0.25)
]

print(f"\nCalculating total business cost for different α values:")
print(f"{'α':<8} {'Nodes':<8} {'Error':<8} {'Misclass':<10} {'Complexity':<12} {'Operational':<12} {'Total':<10}")
print("-" * 70)

for alpha in alpha_values:
    best_cost = float('inf')
    best_config = None
    
    for nodes, error in tree_configs:
        total_cost_business = total_business_cost(alpha, nodes, error, total_cost, operational_cost_per_node)
        
        if total_cost_business < best_cost:
            best_cost = total_cost_business
            best_config = (nodes, error)
        
        if alpha in [0.01, 0.1, 0.3]:  # Show detailed breakdown for selected α values
            misclass_cost = total_cost * (error / error_1)
            complexity_penalty = alpha * nodes
            operational_cost = operational_cost_per_node * nodes
            
            print(f"{alpha:<8.2f} {nodes:<8} {error:<8.2f} ${misclass_cost:<9.1f} ${complexity_penalty:<11.1f} ${operational_cost:<11.1f} ${total_cost_business:<9.1f}")
    
    if alpha in [0.01, 0.1, 0.3]:
        print(f"Best for α={alpha}: {best_config[0]} nodes, error={best_config[1]:.2f}, total=${best_cost:.1f}")
        print("-" * 70)

# Find optimal α
optimal_alpha = None
min_total_cost = float('inf')
optimal_config = None

for alpha in np.linspace(0.01, 0.5, 50):
    for nodes, error in tree_configs:
        total_cost_business = total_business_cost(alpha, nodes, error, total_cost, operational_cost_per_node)
        if total_cost_business < min_total_cost:
            min_total_cost = total_cost_business
            optimal_alpha = alpha
            optimal_config = (nodes, error)

print(f"\nOptimal α = {optimal_alpha:.3f}")
print(f"Optimal tree: {optimal_config[0]} nodes, error rate = {optimal_config[1]:.2f}")
print(f"Minimum total cost = ${min_total_cost:.1f}")

# ============================================================================
# PART 6: Business Implications
# ============================================================================
print("\n" + "="*60)
print("PART 6: Business Implications")
print("="*60)

print("Business implications of different α values:")
print()
print("Low α (e.g., 0.01):")
print("  • More complex trees, higher accuracy")
print("  • Higher operational costs")
print("  • Risk of overfitting")
print("  • Better fraud detection but more expensive to maintain")
print()
print("High α (e.g., 0.3):")
print("  • Simpler trees, lower accuracy")
print("  • Lower operational costs")
print("  • Risk of underfitting")
print("  • Cheaper to maintain but may miss fraud cases")
print()
print("Optimal α balances:")
print("  • Fraud detection accuracy")
print("  • Operational costs")
print("  • Model interpretability")
print("  • Regulatory compliance")

# ============================================================================
# PART 7: Medical Diagnosis Cost Matrix
# ============================================================================
print("\n" + "="*60)
print("PART 7: Medical Diagnosis Cost Matrix")
print("="*60)

print("Designing cost matrix for medical diagnosis system:")
print("False negatives are 10x more expensive than false positives")
print()

# Example costs
fp_cost_medical = 1000  # False positive: unnecessary treatment
fn_cost_medical = 10000  # False negative: missed diagnosis

print(f"Cost matrix:")
print(f"  False Positive (unnecessary treatment): ${fp_cost_medical:,}")
print(f"  False Negative (missed diagnosis): ${fn_cost_medical:,}")
print(f"  Ratio: {fn_cost_medical/fp_cost_medical:.1f}x more expensive for false negatives")
print()
print("Justification:")
print("  • False negative: Patient goes untreated, condition worsens")
print("  • False positive: Patient receives unnecessary treatment, some side effects")
print("  • Medical and legal consequences of missed diagnosis are severe")

# Create cost matrix visualization
cost_matrix = np.array([[0, fp_cost_medical], [fn_cost_medical, 0]])
labels = ['Actual Negative', 'Actual Positive']

plt.figure(figsize=(10, 8))
sns.heatmap(cost_matrix, annot=True, fmt=',', cmap='Reds', 
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=labels, cbar_kws={'label': 'Cost ($)'})
plt.title('Medical Diagnosis Cost Matrix\n(False Negatives 10x More Expensive)')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'medical_cost_matrix.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# PART 8: Daily Fraud Detection Cost Analysis
# ============================================================================
print("\n" + "="*60)
print("PART 8: Daily Fraud Detection Cost Analysis")
print("="*60)

daily_transactions = 10000
fraud_rate = 0.01  # Assume 1% of transactions are fraudulent

print(f"Daily transaction volume: {daily_transactions:,}")
print(f"Assumed fraud rate: {fraud_rate:.1%}")
print(f"Expected fraudulent transactions per day: {daily_transactions * fraud_rate:.0f}")
print()

# Calculate daily costs for different α values
alpha_values_daily = [0.01, 0.05, 0.1, 0.2, 0.3]
daily_costs = []

print(f"{'α':<8} {'Nodes':<8} {'Error':<8} {'Daily Fraud':<12} {'Daily Cost':<12}")
print("-" * 55)

for alpha in alpha_values_daily:
    # Find best tree configuration for this α
    best_cost = float('inf')
    best_config = None
    
    for nodes, error in tree_configs:
        total_cost_business = total_business_cost(alpha, nodes, error, total_cost, operational_cost_per_node)
        if total_cost_business < best_cost:
            best_cost = total_cost_business
            best_config = (nodes, error)
    
    nodes, error = best_config
    
    # Calculate daily fraud detection cost
    daily_fraud_cost = daily_transactions * error * (fp_cost + fn_cost) / 2
    daily_operational_cost = nodes * operational_cost_per_node
    daily_total_cost = daily_fraud_cost + daily_operational_cost
    
    daily_costs.append(daily_total_cost)
    
    print(f"{alpha:<8.2f} {nodes:<8} {error:<8.2f} ${daily_fraud_cost:<11,.0f} ${daily_total_cost:<11,.0f}")

# Create daily cost comparison visualization
plt.figure(figsize=(12, 8))

# Bar chart of daily costs
bars = plt.bar(range(len(alpha_values_daily)), daily_costs, 
               color=['skyblue', 'lightgreen', 'gold', 'orange', 'red'])
plt.xlabel(r'Complexity Parameter α')
plt.ylabel('Daily Total Cost ($)')
plt.title('Daily Fraud Detection Cost vs. α')
plt.xticks(range(len(alpha_values_daily)), [f'{a:.2f}' for a in alpha_values_daily])

# Add value labels on bars
for i, (bar, cost) in enumerate(zip(bars, daily_costs)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(daily_costs)*0.01,
             f'${cost:,.0f}', ha='center', va='bottom', fontweight='bold')

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'daily_cost_comparison.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# PART 9: Comprehensive Analysis
# ============================================================================
print("\n" + "="*60)
print("PART 9: Comprehensive Analysis")
print("="*60)

print("Summary of findings:")
print()
print(f"1. Cost-complexity function: R_α(T) = R(T) + α|T|")
print(f"2. For α = 0.1: Tree with 7 nodes costs ${R_alpha_1:.1f}")
print(f"3. For α = 0.05: Tree with {tree1_nodes if R_alpha_tree1 < R_alpha_tree2 else tree2_nodes} nodes is preferred")
print(f"4. α controls complexity vs. accuracy trade-off")
print(f"5. Optimal α with operational costs: {optimal_alpha:.3f}")
print(f"6. Medical diagnosis: False negatives 10x more expensive")
print(f"7. Daily fraud detection costs range from ${min(daily_costs):,.0f} to ${max(daily_costs):,.0f}")

# Create final summary visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 1. Cost breakdown for optimal α
optimal_nodes, optimal_error = optimal_config
misclass_cost = total_cost * (optimal_error / error_1)
complexity_penalty = optimal_alpha * optimal_nodes
operational_cost = operational_cost_per_node * optimal_nodes

costs = [misclass_cost, complexity_penalty, operational_cost]
labels = ['Misclassification', 'Complexity\nPenalty', 'Operational']
colors = ['#ff9999', '#66b3ff', '#99ff99']

ax1.pie(costs, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax1.set_title(f'Cost Breakdown for Optimal α = {optimal_alpha:.3f}')

# 2. α vs. tree size relationship
ax2.plot(alpha_values_daily, [config[0] for config in [tree_configs[2], tree_configs[1], tree_configs[0], tree_configs[0], tree_configs[0]]], 
         'bo-', linewidth=2, markersize=8)
ax2.set_xlabel(r'α')
ax2.set_ylabel('Optimal Tree Size (nodes)')
ax2.set_title('α vs. Optimal Tree Size')
ax2.grid(True, alpha=0.3)

# 3. Daily cost trend
ax3.plot(alpha_values_daily, daily_costs, 'ro-', linewidth=2, markersize=8)
ax3.set_xlabel(r'α')
ax3.set_ylabel('Daily Cost ($)')
ax3.set_title('Daily Fraud Detection Cost vs. α')
ax3.grid(True, alpha=0.3)

# 4. Error rate vs. tree size
tree_sizes_plot = [config[0] for config in tree_configs]
error_rates_plot = [config[1] for config in tree_configs]
ax4.scatter(tree_sizes_plot, error_rates_plot, s=100, c='green', alpha=0.7)
ax4.set_xlabel('Tree Size (nodes)')
ax4.set_ylabel('Error Rate')
ax4.set_title('Error Rate vs. Tree Size')
ax4.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(tree_sizes_plot, error_rates_plot, 1)
p = np.poly1d(z)
ax4.plot(tree_sizes_plot, p(tree_sizes_plot), "r--", alpha=0.8)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'comprehensive_analysis.png'), dpi=300, bbox_inches='tight')

print(f"\nAll visualizations saved to: {save_dir}")
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
