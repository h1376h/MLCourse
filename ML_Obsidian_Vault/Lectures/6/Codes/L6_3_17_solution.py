import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.patches import FancyBboxPatch, Rectangle
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_17")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['text.latex.preamble'] = r'\usepackage{amssymb}'

print("=" * 80)
print("QUESTION 17: PRUNING APPROACHES ACROSS ALGORITHMS")
print("=" * 80)

# Define the three algorithms and their pruning characteristics
algorithms = {
    'ID3': {
        'pruning': 'No built-in pruning',
        'reason': 'Designed as a basic algorithm without overfitting considerations',
        'characteristics': [
            'Grows trees to full depth',
            'No stopping criteria',
            'Prone to overfitting',
            'Requires manual post-pruning'
        ],
        'color': 'red'
    },
    'C4.5': {
        'pruning': 'Pessimistic Error Pruning (PEP)',
        'reason': 'Uses statistical confidence intervals to estimate true error rates',
        'characteristics': [
            'Uses confidence intervals',
            'Conservative pruning approach',
            'Reduces overfitting',
            'Built into the algorithm'
        ],
        'color': 'blue'
    },
    'CART': {
        'pruning': 'Cost-Complexity Pruning',
        'reason': 'Balances tree complexity with prediction accuracy using α parameter',
        'characteristics': [
            'Uses α parameter to control pruning',
            'Creates sequence of pruned trees',
            'Cross-validation for α selection',
            'Optimal balance of accuracy and complexity'
        ],
        'color': 'green'
    }
}

# 1. Does ID3 include built-in pruning capabilities?
print("\n1. Does ID3 include built-in pruning capabilities? Why or why not?")
print("-" * 70)
print("Answer: NO")
print(f"Reason: {algorithms['ID3']['reason']}")
print("\nCharacteristics:")
for char in algorithms['ID3']['characteristics']:
    print(f"  • {char}")

# 2. Describe C4.5's pessimistic error pruning
print("\n2. Describe C4.5's pessimistic error pruning in one sentence")
print("-" * 70)
print(f"Answer: {algorithms['C4.5']['pruning']}")
print(f"Description: Uses statistical confidence intervals to estimate true error rates")
print("\nHow it works:")
print("  • Calculates error rate with confidence intervals")
print("  • Prunes if upper bound of error rate doesn't increase significantly")
print("  • Conservative approach that prevents over-pruning")

# 3. Purpose of CART's cost-complexity pruning parameter α
print("\n3. What is the purpose of CART's cost-complexity pruning parameter α?")
print("-" * 70)
print("Answer: Balances tree complexity with prediction accuracy")
print("\nMathematical formulation:")
print("  Cost-Complexity = Error Rate + $\\alpha \\times |T|$")
print("  where $|T|$ is the number of terminal nodes")
print("\nEffects of $\\alpha$:")
print("  • $\\alpha = 0$: No pruning (full tree)")
print("  • $\\alpha \\to \\infty$: Heavy pruning (stump)")
print("  • Optimal $\\alpha$: Found through cross-validation")

# 4. Which algorithms would prune a subtree with training accuracy 90% but validation accuracy 75%
print("\n4. If a subtree has training accuracy 90% but validation accuracy 75%,")
print("   which algorithms would likely prune it?")
print("-" * 70)
print("Answer: C4.5 and CART would likely prune it")
print("\nReasoning:")
print("  • Training accuracy (90%) >> Validation accuracy (75%) = Overfitting")
print("  • ID3: No pruning capability")
print("  • C4.5: PEP would detect overfitting and prune")
print("  • CART: Cost-complexity pruning would favor simpler tree")

# Create separate visualizations for each aspect

# Plot 1: Pruning Capability Comparison
plt.figure(figsize=(10, 6))
algo_names = list(algorithms.keys())
pruning_scores = [0, 1, 1]  # ID3: 0, C4.5: 1, CART: 1
colors = [algorithms[algo]['color'] for algo in algo_names]

bars = plt.bar(algo_names, pruning_scores, color=colors, alpha=0.7, edgecolor='black')
plt.ylabel('Pruning Capability (0=No, 1=Yes)')
plt.ylim(0, 1.2)
plt.grid(True, alpha=0.3)
plt.title('Pruning Capability by Algorithm', fontsize=14, fontweight='bold')

# Add value labels on bars
for bar, score in zip(bars, pruning_scores):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{score}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'pruning_capability_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Pruning Method Comparison
plt.figure(figsize=(10, 6))
methods = ['No Pruning', 'Pessimistic Error', 'Cost-Complexity']
method_colors = ['red', 'blue', 'green']

plt.bar(methods, [1, 1, 1], color=method_colors, alpha=0.7, edgecolor='black')
plt.ylabel('Algorithm Count')
plt.grid(True, alpha=0.3)
plt.title('Pruning Methods', fontsize=14, fontweight='bold')

# Add algorithm labels
plt.text(0.5, 0.8, 'ID3', ha='center', va='center', transform=plt.gca().transAxes, 
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=2))
plt.text(0.5, 0.6, 'C4.5', ha='center', va='center', transform=plt.gca().transAxes,
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", lw=2))
plt.text(0.5, 0.4, 'CART', ha='center', va='center', transform=plt.gca().transAxes,
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", lw=2))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'pruning_methods.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Overfitting Detection Example
plt.figure(figsize=(10, 8))
plt.title('Overfitting Detection Example\n(Training: 90%, Validation: 75%)', fontsize=14, fontweight='bold')

# Create sample tree structure
tree_x = [0, -2, 2, -3, -1, 1, 3]
tree_y = [0, -1, -1, -2, -2, -2, -2]
tree_labels = ['Root', 'L', 'R', 'LL', 'LR', 'RL', 'RR']
tree_colors = ['lightblue', 'lightgreen', 'lightgreen', 'red', 'lightgreen', 'lightgreen', 'red']

# Plot tree nodes
for i, (x, y, label, color) in enumerate(zip(tree_x, tree_y, tree_labels, tree_colors)):
    plt.scatter(x, y, s=300, c=color, edgecolor='black', linewidth=2, zorder=3)
    plt.text(x, y, label, ha='center', va='center', fontweight='bold', fontsize=10)

# Plot tree edges
plt.plot([0, -2], [0, -1], 'k-', linewidth=2)
plt.plot([0, 2], [0, -1], 'k-', linewidth=2)
plt.plot([-2, -3], [-1, -2], 'k-', linewidth=2)
plt.plot([-2, -1], [-1, -2], 'k-', linewidth=2)
plt.plot([2, 1], [-1, -2], 'k-', linewidth=2)
plt.plot([2, 3], [-1, -2], 'k-', linewidth=2)

# Add pruning indicators
plt.annotate('Prune\n(Overfitting)', xy=(-3, -2), xytext=(-4, -1),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             bbox=dict(boxstyle="round,pad=0.3", fc="red", ec="darkred", alpha=0.7),
             color='white', fontweight='bold', ha='center')

plt.annotate('Prune\n(Overfitting)', xy=(3, -2), xytext=(4, -1),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             bbox=dict(boxstyle="round,pad=0.3", fc="red", ec="darkred", alpha=0.7),
             color='white', fontweight='bold', ha='center')

plt.xlim(-5, 5)
plt.ylim(-2.5, 0.5)
plt.gca().set_aspect('equal')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'overfitting_detection_example.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Cost-Complexity Trade-off
plt.figure(figsize=(10, 6))
plt.title('CART Cost-Complexity Trade-off', fontsize=14, fontweight='bold')

# Generate sample data for cost-complexity curve
alpha_values = np.linspace(0, 2, 100)
error_rates = 0.15 + 0.1 * np.exp(-alpha_values)  # Sample error rates
complexity_penalties = alpha_values * np.array([10, 8, 6, 5, 4, 3, 2.5, 2, 1.8, 1.5] + [1.2] * 90)
total_cost = error_rates + complexity_penalties

plt.plot(alpha_values, error_rates, 'b-', linewidth=2, label='Error Rate')
plt.plot(alpha_values, complexity_penalties, 'g-', linewidth=2, label='Complexity Penalty')
plt.plot(alpha_values, total_cost, 'r-', linewidth=3, label='Total Cost')

# Mark optimal α
optimal_alpha = alpha_values[np.argmin(total_cost)]
optimal_cost = np.min(total_cost)
plt.axvline(x=optimal_alpha, color='red', linestyle='--', alpha=0.7, label=f'Optimal $\\alpha$ = {optimal_alpha:.2f}')
plt.scatter(optimal_alpha, optimal_cost, color='red', s=100, zorder=5)

plt.xlabel('$\\alpha$ (Complexity Parameter)')
plt.ylabel('Cost')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'cost_complexity_tradeoff.png'), dpi=300, bbox_inches='tight')
plt.close()



# Create detailed comparison table
print("\n" + "=" * 80)
print("DETAILED PRUNING COMPARISON TABLE")
print("=" * 80)

comparison_data = {
    'Feature': [
        'Built-in Pruning',
        'Pruning Method',
        'Overfitting Protection',
        'Parameter Tuning',
        'Computational Cost',
        'Best For'
    ],
    'ID3': [
        'No',
        'Manual post-pruning only',
        'None (prone to overfitting)',
        'None required',
        'Lowest',
        'Simple datasets, educational purposes'
    ],
    'C4.5': [
        'Yes',
        'Pessimistic Error Pruning (PEP)',
        'Good (statistical approach)',
        'Confidence level',
        'Medium',
        'Interpretable rules, medical applications'
    ],
    'CART': [
        'Yes',
        'Cost-Complexity Pruning',
        'Excellent (cross-validation)',
        'α parameter',
        'Highest',
        'Production systems, mixed data types'
    ]
}

df = pd.DataFrame(comparison_data)
print(df.to_string(index=False))

# Mathematical details for C4.5 PEP
print("\n" + "=" * 80)
print("MATHEMATICAL DETAILS: C4.5 PESSIMISTIC ERROR PRUNING")
print("=" * 80)

print("The pessimistic error rate is calculated as:")
print("  e' = (e + 0.5) / n")
print("  where e = number of errors, n = number of samples")
print("\nThe standard error is:")
print("  SE = sqrt(e' × (1 - e') / n)")
print("\nThe upper bound of the error rate is:")
print("  UB = e' + z × SE")
print("  where z is the confidence level (typically 1.96 for 95% confidence)")

# Mathematical details for CART cost-complexity
print("\n" + "=" * 80)
print("MATHEMATICAL DETAILS: CART COST-COMPLEXITY PRUNING")
print("=" * 80)

print("The cost-complexity measure is:")
print("  $R_{\\alpha}(T) = R(T) + \\alpha \\times |T|$")
print("  where:")
print("    $R(T)$ = misclassification rate of tree T")
print("    $|T|$ = number of terminal nodes in T")
print("    $\\alpha$ = complexity parameter")
print("\nFor a subtree $T_t$:")
print("  $R_{\\alpha}(T_t) = R(T_t) + \\alpha \\times |T_t|$")
print("\nThe tree is pruned if:")
print("  $R_{\\alpha}(T_t) \\geq R_{\\alpha}(t)$")
print("  where $t$ is the root of subtree $T_t$")

print(f"\nPlots saved to: {save_dir}")
