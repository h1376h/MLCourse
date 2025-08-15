import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.patches as mpatches

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_4_Quiz_3")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'

print("=== Hospital Decision Tree Pruning Analysis ===\n")

# Given tree structure
tree_data = {
    'Root': {'samples': 200, 'train_error': 0.25, 'val_error': 0.28},
    'Left': {'samples': 120, 'train_error': 0.20, 'val_error': 0.25},
    'Right': {'samples': 80, 'train_error': 0.35, 'val_error': 0.40},
    'LL': {'samples': 60, 'train_error': 0.15, 'val_error': 0.20},
    'LR': {'samples': 60, 'train_error': 0.25, 'val_error': 0.30},
    'RL': {'samples': 40, 'train_error': 0.30, 'val_error': 0.35},
    'RR': {'samples': 40, 'train_error': 0.40, 'val_error': 0.45}
}

print("Given Decision Tree Structure:")
print("Root (200 samples, train_error=0.25, val_error=0.28)")
print("├── Left (120 samples, train_error=0.20, val_error=0.25)")
print("│   ├── LL (60 samples, train_error=0.15, val_error=0.20)")
print("│   └── LR (60 samples, train_error=0.25, val_error=0.30)")
print("└── Right (80 samples, train_error=0.35, val_error=0.40)")
print("    ├── RL (40 samples, train_error=0.30, val_error=0.35)")
print("    └── RR (40 samples, train_error=0.40, val_error=0.45)")
print()

# Step 1: Calculate validation error before and after pruning each subtree
print("=== Step 1: Validation Error Before and After Pruning Each Subtree ===\n")

def calculate_weighted_error(nodes, weights):
    """Calculate weighted error across multiple nodes"""
    total_error = 0
    total_weight = 0
    print(f"  Calculating weighted error for nodes: {nodes}")
    print(f"  Weights (sample sizes): {weights}")
    for i, (node, weight) in enumerate(zip(nodes, weights)):
        node_error = tree_data[node]['val_error']
        weighted_error = node_error * weight
        print(f"    {node}: {node_error:.3f} × {weight} = {weighted_error:.1f}")
        total_error += weighted_error
        total_weight += weight
    final_error = total_error / total_weight if total_weight > 0 else 0
    print(f"  Total weighted errors: {total_error:.1f}")
    print(f"  Total samples: {total_weight}")
    print(f"  Weighted average: {final_error:.3f}")
    return final_error

# Calculate errors for different pruning scenarios
pruning_scenarios = {}

# Scenario 1: Prune Left subtree (keep Root and Right)
left_samples = tree_data['Left']['samples']
right_samples = tree_data['Right']['samples']
root_samples = tree_data['Root']['samples']

print("1. Pruning Left Subtree:")
print("   Before pruning: Weighted average of Left and Right subtrees")
print(f"   Left: {tree_data['Left']['val_error']:.3f} × {left_samples} samples")
print(f"   Right: {tree_data['Right']['val_error']:.3f} × {right_samples} samples")

# Before pruning Left: weighted average of Left and Right
before_prune_left = calculate_weighted_error(['Left', 'Right'], [left_samples, right_samples])
# After pruning Left: just Right subtree error
after_prune_left = tree_data['Right']['val_error']

pruning_scenarios['Prune_Left'] = {
    'before': before_prune_left,
    'after': after_prune_left,
    'improvement': before_prune_left - after_prune_left
}

print(f"   After pruning: Right subtree error = {after_prune_left:.3f}")
print(f"   Improvement: {pruning_scenarios['Prune_Left']['improvement']:.3f}")
print()

# Scenario 2: Prune Right subtree (keep Root and Left)
print("2. Pruning Right Subtree:")
print("   Before pruning: Weighted average of Left and Right subtrees")
before_prune_right = calculate_weighted_error(['Left', 'Right'], [left_samples, right_samples])
after_prune_right = tree_data['Left']['val_error']

pruning_scenarios['Prune_Right'] = {
    'before': before_prune_left,  # Same as before_prune_left
    'after': after_prune_right,
    'improvement': before_prune_left - after_prune_right
}

print(f"   After pruning: Left subtree error = {after_prune_right:.3f}")
print(f"   Improvement: {pruning_scenarios['Prune_Right']['improvement']:.3f}")
print()

# Scenario 3: Prune LL and LR (keep Root, Left, Right)
print("3. Pruning LL and LR Subtrees:")
print("   Before pruning: Weighted average of LL and LR")
before_prune_ll_lr = calculate_weighted_error(['LL', 'LR'], [tree_data['LL']['samples'], tree_data['LR']['samples']])
after_prune_ll_lr = tree_data['Left']['val_error']

pruning_scenarios['Prune_LL_LR'] = {
    'before': before_prune_ll_lr,
    'after': after_prune_ll_lr,
    'improvement': before_prune_ll_lr - after_prune_ll_lr
}

print(f"   After pruning: Left subtree error = {after_prune_ll_lr:.3f}")
print(f"   Improvement: {pruning_scenarios['Prune_LL_LR']['improvement']:.3f}")
print()

# Scenario 4: Prune RL and RR (keep Root, Left, Right)
print("4. Pruning RL and RR Subtrees:")
print("   Before pruning: Weighted average of RL and RR")
before_prune_rl_rr = calculate_weighted_error(['RL', 'RR'], [tree_data['RL']['samples'], tree_data['RR']['samples']])
after_prune_rl_rr = tree_data['Right']['val_error']

pruning_scenarios['Prune_RL_RR'] = {
    'before': before_prune_rl_rr,
    'after': after_prune_rl_rr,
    'improvement': before_prune_rl_rr - after_prune_rl_rr
}

print(f"   After pruning: Right subtree error = {after_prune_rl_rr:.3f}")
print(f"   Improvement: {pruning_scenarios['Prune_RL_RR']['improvement']:.3f}")
print()

# Step 2: Determine which subtrees should be pruned using reduced error pruning
print("=== Step 2: Reduced Error Pruning Analysis ===\n")

# Sort pruning scenarios by improvement
sorted_pruning = sorted(pruning_scenarios.items(), key=lambda x: x[1]['improvement'], reverse=True)

print("Pruning Scenarios Ranked by Improvement:")
for i, (scenario, data) in enumerate(sorted_pruning, 1):
    print(f"{i}. {scenario}: Improvement = {data['improvement']:.3f}")
    print(f"   Before: {data['before']:.3f}, After: {data['after']:.3f}")

print(f"\nBest pruning strategy: {sorted_pruning[0][0]}")
print(f"Improvement: {sorted_pruning[0][1]['improvement']:.3f}")
print()

# Step 3: Draw the final tree structure after optimal pruning
print("=== Step 3: Final Tree Structure After Optimal Pruning ===\n")

# The best pruning is to prune the Left subtree (keep Root and Right)
final_tree_structure = {
    'Root': {'samples': 200, 'val_error': 0.28},
    'Right': {'samples': 80, 'val_error': 0.40}
}

print("Final Tree Structure After Optimal Pruning:")
print("Root (200 samples, val_error=0.28)")
print("└── Right (80 samples, val_error=0.40)")
print()

# Step 4: Calculate final validation error after pruning
print("=== Step 4: Final Validation Error After Pruning ===\n")

final_val_error = calculate_weighted_error(['Root', 'Right'], [200, 80])
print(f"Final validation error after pruning: {final_val_error:.3f}")
print(f"Original validation error: {tree_data['Root']['val_error']:.3f}")
print(f"Improvement: {tree_data['Root']['val_error'] - final_val_error:.3f}")
print()

# Step 5: Optimal pruning strategy for ≤3 nodes
print("=== Step 5: Optimal Pruning Strategy for ≤3 Nodes ===\n")

# We need to keep at most 3 nodes including root
# Options: Root only, Root+Left, Root+Right, Root+LL+LR, Root+RL+RR
pruning_3_nodes = {}

# Option 1: Root only
print("Option 1: Root_Only")
print("  Root error: {:.3f}".format(tree_data['Root']['val_error']))
pruning_3_nodes['Root_Only'] = tree_data['Root']['val_error']
print()

# Option 2: Root + Left
print("Option 2: Root_Left")
pruning_3_nodes['Root_Left'] = calculate_weighted_error(['Root', 'Left'], [200, 120])
print()

# Option 3: Root + Right
print("Option 3: Root_Right")
pruning_3_nodes['Root_Right'] = calculate_weighted_error(['Root', 'Right'], [200, 80])
print()

# Option 4: Root + LL + LR
print("Option 4: Root_LL_LR")
pruning_3_nodes['Root_LL_LR'] = calculate_weighted_error(['Root', 'LL', 'LR'], [200, 60, 60])
print()

# Option 5: Root + RL + RR
print("Option 5: Root_RL_RR")
pruning_3_nodes['Root_RL_RR'] = calculate_weighted_error(['Root', 'RL', 'RR'], [200, 40, 40])
print()

print("Pruning Strategies for ≤3 Nodes:")
for strategy, error in pruning_3_nodes.items():
    print(f"{strategy}: {error:.3f}")

best_3_node = min(pruning_3_nodes.items(), key=lambda x: x[1])
print(f"\nBest 3-node strategy: {best_3_node[0]} with error {best_3_node[1]:.3f}")
print()

# Step 6: Medical implications of aggressive pruning
print("=== Step 6: Medical Implications of Aggressive Pruning ===\n")

print("Medical Implications:")
print("1. False Negatives (Missing High-Risk Patients):")
print("   - Patients who need immediate attention may be missed")
print("   - Could lead to readmissions, complications, or worse outcomes")
print("   - Higher mortality risk for high-risk patients")
print()
print("2. False Positives (Unnecessary Interventions):")
print("   - Low-risk patients may receive unnecessary treatments")
print("   - Increased healthcare costs and patient anxiety")
print("   - Potential side effects from unnecessary interventions")
print()
print("3. Balance Considerations:")
print("   - In medical contexts, false negatives are often more costly than false positives")
print("   - Aggressive pruning may increase false negative rate")
print("   - Need to balance interpretability with accuracy")
print()

# Step 7: Cost analysis
print("=== Step 7: Cost Analysis ===\n")

# Cost parameters
false_negative_cost = 1000  # $1000
false_positive_cost = 100   # $100

# Calculate costs for different scenarios
def calculate_total_cost(val_error, samples, fn_cost, fp_cost):
    """Calculate total cost based on validation error and sample size"""
    print(f"  Calculating costs for validation error: {val_error:.3f}")
    print(f"  Total samples: {samples}")
    print(f"  False negative cost: ${fn_cost}")
    print(f"  False positive cost: ${fp_cost}")
    
    # Assume balanced classes for simplicity
    fn_count = val_error * samples / 2  # Half of errors are false negatives
    fp_count = val_error * samples / 2  # Half of errors are false positives
    
    print(f"  False negatives: {val_error:.3f} × {samples} × 0.5 = {fn_count:.1f} patients")
    print(f"  False positives: {val_error:.3f} × {samples} × 0.5 = {fp_count:.1f} patients")
    
    fn_cost_total = fn_count * fn_cost
    fp_cost_total = fp_count * fp_cost
    total_cost = fn_cost_total + fp_cost_total
    
    print(f"  False negative cost: {fn_count:.1f} × ${fn_cost} = ${fn_cost_total:.0f}")
    print(f"  False positive cost: {fp_count:.1f} × ${fp_cost} = ${fp_cost_total:.0f}")
    print(f"  Total cost: ${fn_cost_total:.0f} + ${fp_cost_total:.0f} = ${total_cost:.0f}")
    
    return total_cost, fn_count, fp_count

# Original tree cost
orig_cost, orig_fn, orig_fp = calculate_total_cost(tree_data['Root']['val_error'], 200, false_negative_cost, false_positive_cost)

# Pruned tree cost
pruned_cost, pruned_fn, pruned_fp = calculate_total_cost(final_val_error, 200, false_negative_cost, false_positive_cost)

print("Cost Analysis:")
print(f"Original Tree:")
print(f"  Validation Error: {tree_data['Root']['val_error']:.3f}")
print(f"  False Negatives: {orig_fn:.1f}")
print(f"  False Positives: {orig_fp:.1f}")
print(f"  Total Cost: ${orig_cost:.2f}")
print()
print(f"Pruned Tree:")
print(f"  Validation Error: {final_val_error:.3f}")
print(f"  False Negatives: {pruned_fn:.1f}")
print(f"  False Positives: {pruned_fp:.1f}")
print(f"  Total Cost: ${pruned_cost:.2f}")
print()
print(f"Cost Difference: ${orig_cost - pruned_cost:.2f}")
print(f"Cost Reduction: {((orig_cost - pruned_cost) / orig_cost * 100):.1f}%")
print()

# Step 8: Processing capacity and daily cost savings
print("=== Step 8: Processing Capacity and Daily Cost Savings ===\n")

# Processing parameters
pruned_capacity = 50  # patients per day
original_capacity = 30  # patients per day

# Calculate daily costs
print("Daily Cost Calculations:")
print(f"  Original tree cost per patient: ${orig_cost:.0f} ÷ 200 = ${orig_cost/200:.2f}")
print(f"  Pruned tree cost per patient: ${pruned_cost:.0f} ÷ 200 = ${pruned_cost/200:.2f}")
print()

daily_orig_cost = (orig_cost / 200) * original_capacity  # Cost per patient × daily capacity
daily_pruned_cost = (pruned_cost / 200) * pruned_capacity

print(f"  Original tree daily cost: ${orig_cost/200:.2f} × {original_capacity} = ${daily_orig_cost:.2f}")
print(f"  Pruned tree daily cost: ${pruned_cost/200:.2f} × {pruned_capacity} = ${daily_pruned_cost:.2f}")
print()

daily_savings = daily_orig_cost - daily_pruned_cost

print("Processing Capacity Analysis:")
print(f"Original Tree: {original_capacity} patients/day")
print(f"Pruned Tree: {pruned_capacity} patients/day")
print(f"Capacity Increase: {((pruned_capacity - original_capacity) / original_capacity * 100):.1f}%")
print()
print("Daily Cost Analysis:")
print(f"Original Tree Daily Cost: ${daily_orig_cost:.2f}")
print(f"Pruned Tree Daily Cost: ${daily_pruned_cost:.2f}")
print(f"Daily Cost Savings: ${daily_savings:.2f}")
print(f"Daily Savings Percentage: {(daily_savings / daily_orig_cost * 100):.1f}%")
print()

# Create visualizations
print("=== Creating Visualizations ===\n")

# Visualization 1: Original vs Pruned Tree Structure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Original tree
ax1.set_title('Original Decision Tree Structure', fontsize=14, fontweight='bold')
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')

# Draw original tree
# Root
root_box = FancyBboxPatch((4, 8), 2, 1, boxstyle="round,pad=0.1", 
                          facecolor='lightblue', edgecolor='black', linewidth=2)
ax1.add_patch(root_box)
ax1.text(5, 8.5, 'Root\n200 samples\nval_error=0.28', ha='center', va='center', fontsize=10)

# Left subtree
ax1.plot([5, 3], [8, 6], 'k-', linewidth=2)
left_box = FancyBboxPatch((1.5, 5), 3, 1, boxstyle="round,pad=0.1", 
                          facecolor='lightgreen', edgecolor='black', linewidth=2)
ax1.add_patch(left_box)
ax1.text(3, 5.5, 'Left\n120 samples\nval_error=0.25', ha='center', va='center', fontsize=10)

# LL and LR
ax1.plot([3, 2], [5, 3], 'k-', linewidth=2)
ax1.plot([3, 4], [5, 3], 'k-', linewidth=2)
ll_box = FancyBboxPatch((0.5, 2), 3, 1, boxstyle="round,pad=0.1", 
                        facecolor='lightyellow', edgecolor='black', linewidth=2)
lr_box = FancyBboxPatch((3.5, 2), 3, 1, boxstyle="round,pad=0.1", 
                        facecolor='lightyellow', edgecolor='black', linewidth=2)
ax1.add_patch(ll_box)
ax1.add_patch(lr_box)
ax1.text(2, 2.5, 'LL\n60 samples\nval_error=0.20', ha='center', va='center', fontsize=9)
ax1.text(5, 2.5, 'LR\n60 samples\nval_error=0.30', ha='center', va='center', fontsize=9)

# Right subtree
ax1.plot([5, 7], [8, 6], 'k-', linewidth=2)
right_box = FancyBboxPatch((6.5, 5), 3, 1, boxstyle="round,pad=0.1", 
                           facecolor='lightcoral', edgecolor='black', linewidth=2)
ax1.add_patch(right_box)
ax1.text(8, 5.5, 'Right\n80 samples\nval_error=0.40', ha='center', va='center', fontsize=10)

# RL and RR
ax1.plot([8, 7], [5, 3], 'k-', linewidth=2)
ax1.plot([8, 9], [5, 3], 'k-', linewidth=2)
rl_box = FancyBboxPatch((5.5, 2), 3, 1, boxstyle="round,pad=0.1", 
                        facecolor='lightyellow', edgecolor='black', linewidth=2)
rr_box = FancyBboxPatch((8.5, 2), 3, 1, boxstyle="round,pad=0.1", 
                        facecolor='lightyellow', edgecolor='black', linewidth=2)
ax1.add_patch(rl_box)
ax1.add_patch(rr_box)
ax1.text(7, 2.5, 'RL\n40 samples\nval_error=0.35', ha='center', va='center', fontsize=9)
ax1.text(9, 2.5, 'RR\n40 samples\nval_error=0.45', ha='center', va='center', fontsize=9)

# Pruned tree
ax2.set_title('Pruned Decision Tree Structure', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')

# Draw pruned tree
# Root
root_box2 = FancyBboxPatch((4, 8), 2, 1, boxstyle="round,pad=0.1", 
                           facecolor='lightblue', edgecolor='black', linewidth=2)
ax2.add_patch(root_box2)
ax2.text(5, 8.5, 'Root\n200 samples\nval_error=0.28', ha='center', va='center', fontsize=10)

# Right subtree only
ax2.plot([5, 7], [8, 6], 'k-', linewidth=2)
right_box2 = FancyBboxPatch((6.5, 5), 3, 1, boxstyle="round,pad=0.1", 
                            facecolor='lightcoral', edgecolor='black', linewidth=2)
ax2.add_patch(right_box2)
ax2.text(8, 5.5, 'Right\n80 samples\nval_error=0.40', ha='center', va='center', fontsize=10)

# Add pruning note
ax2.text(5, 1, 'Left subtree pruned\n(optimal strategy)', ha='center', va='center', 
         fontsize=12, fontweight='bold', color='red')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'tree_structure_comparison.png'), dpi=300, bbox_inches='tight')

# Visualization 2: Pruning Scenarios Comparison
fig, ax = plt.subplots(figsize=(12, 8))

scenarios = list(pruning_scenarios.keys())
before_errors = [data['before'] for data in pruning_scenarios.values()]
after_errors = [data['after'] for data in pruning_scenarios.values()]
improvements = [data['improvement'] for data in pruning_scenarios.values()]

x = np.arange(len(scenarios))
width = 0.35

bars1 = ax.bar(x - width/2, before_errors, width, label='Before Pruning', color='lightcoral', alpha=0.8)
bars2 = ax.bar(x + width/2, after_errors, width, label='After Pruning', color='lightgreen', alpha=0.8)

# Add improvement annotations
for i, (bar1, bar2, improvement) in enumerate(zip(bars1, bars2, improvements)):
    height = max(bar1.get_height(), bar2.get_height())
    ax.annotate(f'+{improvement:.3f}',
                xy=(bar2.get_x() + bar2.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontweight='bold', color='green')

ax.set_xlabel('Pruning Scenario')
ax.set_ylabel('Validation Error')
ax.set_title('Validation Error Before and After Pruning Each Subtree')
ax.set_xticks(x)
ax.set_xticklabels([s.replace('_', '\n') for s in scenarios], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'pruning_scenarios_comparison.png'), dpi=300, bbox_inches='tight')

# Visualization 3: Cost Analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Cost breakdown
costs = ['Original Tree', 'Pruned Tree']
fn_costs = [orig_fn * false_negative_cost, pruned_fn * false_negative_cost]
fp_costs = [orig_fp * false_positive_cost, pruned_fp * false_positive_cost]

x = np.arange(len(costs))
width = 0.35

bars1 = ax1.bar(x - width/2, fn_costs, width, label='False Negative Cost', color='red', alpha=0.7)
bars2 = ax1.bar(x + width/2, fp_costs, width, label='False Positive Cost', color='orange', alpha=0.7)

ax1.set_xlabel('Tree Configuration')
ax1.set_ylabel('Cost ($)', usetex=False)
ax1.set_title('Cost Breakdown by Error Type')
ax1.set_xticks(x)
ax1.set_xticklabels(costs)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add cost annotations
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    total_cost = bar1.get_height() + bar2.get_height()
    ax1.annotate(f'${total_cost:.0f}',
                 xy=(bar2.get_x() + bar2.get_width()/2, total_cost),
                 xytext=(0, 3), textcoords="offset points",
                 ha='center', va='bottom', fontweight='bold')

# Daily cost savings
daily_costs = [daily_orig_cost, daily_pruned_cost]
colors = ['lightcoral', 'lightgreen']

bars = ax2.bar(costs, daily_costs, color=colors, alpha=0.8)
ax2.set_xlabel('Tree Configuration')
ax2.set_ylabel('Daily Cost ($)', usetex=False)
ax2.set_title('Daily Processing Cost Comparison')
ax2.grid(True, alpha=0.3)

# Add savings annotation
ax2.annotate(f'Daily Savings:\n${daily_savings:.2f}',
             xy=(1, daily_pruned_cost),
             xytext=(0.5, daily_orig_cost * 0.8),
             arrowprops=dict(arrowstyle='->', lw=2, color='red'),
             fontsize=12, fontweight='bold', color='red',
             ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=2))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'cost_analysis.png'), dpi=300, bbox_inches='tight')

# Visualization 4: 3-Node Pruning Strategies
fig, ax = plt.subplots(figsize=(10, 6))

strategies = list(pruning_3_nodes.keys())
errors = list(pruning_3_nodes.values())

bars = ax.bar(strategies, errors, color='skyblue', alpha=0.8)
ax.set_xlabel('Pruning Strategy')
ax.set_ylabel('Validation Error')
ax.set_title('Validation Error for Different 3-Node Pruning Strategies')
ax.set_xticklabels([s.replace('_', '\n') for s in strategies], rotation=45, ha='right')
ax.grid(True, alpha=0.3)

# Highlight best strategy
best_idx = np.argmin(errors)
bars[best_idx].set_color('lightgreen')
bars[best_idx].set_edgecolor('green')
bars[best_idx].set_linewidth(2)

# Add best strategy annotation
ax.annotate(f'Best Strategy\n{best_3_node[0]}\nError: {best_3_node[1]:.3f}',
            xy=(best_idx, best_3_node[1]),
            xytext=(best_idx + 0.5, best_3_node[1] + 0.02),
            arrowprops=dict(arrowstyle='->', lw=2, color='green'),
            fontsize=10, fontweight='bold', color='green',
            ha='left', va='bottom',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", lw=2))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'three_node_pruning_strategies.png'), dpi=300, bbox_inches='tight')

print(f"All visualizations saved to: {save_dir}")
print("\n=== Analysis Complete ===")
