import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.patches import FancyBboxPatch, Rectangle
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_20")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("=" * 80)
print("QUESTION 20: BIAS-VARIANCE TRADE-OFFS IN DECISION TREE ALGORITHMS")
print("=" * 80)

# Define the three algorithms and their bias-variance characteristics
algorithms = {
    'ID3': {
        'bias': 'High',
        'variance': 'Very High',
        'overfitting_tendency': 'Very High',
        'pruning_protection': 'None',
        'reason_high_bias': 'Simple splitting criteria, no optimization',
        'reason_high_variance': 'No pruning, grows to full depth',
        'characteristics': [
            'Grows trees to maximum depth',
            'No stopping criteria',
            'Simple entropy-based splitting',
            'No regularization'
        ],
        'color': 'red'
    },
    'C4.5': {
        'bias': 'Medium',
        'variance': 'Medium',
        'overfitting_tendency': 'Medium',
        'pruning_protection': 'Good',
        'reason_medium_bias': 'Gain ratio and statistical pruning',
        'reason_medium_variance': 'Pessimistic error pruning',
        'characteristics': [
            'Gain ratio reduces bias toward multi-valued features',
            'Pessimistic error pruning',
            'Statistical confidence intervals',
            'Moderate regularization'
        ],
        'color': 'blue'
    },
    'CART': {
        'bias': 'Low',
        'variance': 'Low',
        'overfitting_tendency': 'Low',
        'pruning_protection': 'Excellent',
        'reason_low_bias': 'Binary splitting optimization',
        'reason_low_variance': 'Cost-complexity pruning with cross-validation',
        'characteristics': [
            'Binary splitting strategy',
            'Cost-complexity pruning',
            'Cross-validation for α selection',
            'Strong regularization'
        ],
        'color': 'green'
    }
}

# 1. Which algorithm typically has the highest bias? Explain why
print("1. Which algorithm typically has the highest bias? Explain why")
print("-" * 80)
print("Answer: ID3 typically has the highest bias")
print(f"Bias Level: {algorithms['ID3']['bias']}")
print(f"Reason: {algorithms['ID3']['reason_high_bias']}")
print("\nDetailed explanation:")
print("  • ID3 uses simple entropy-based splitting without optimization")
print("  • No consideration of feature interactions or complex patterns")
print("  • Greedy approach may miss globally optimal splits")
print("  • No regularization to control model complexity")
print("  • Designed for simplicity rather than accuracy optimization")

# 2. Which algorithm is most prone to overfitting without pruning?
print("\n\n2. Which algorithm is most prone to overfitting without pruning?")
print("-" * 80)
print("Answer: ID3 is most prone to overfitting without pruning")
print(f"Overfitting Tendency: {algorithms['ID3']['overfitting_tendency']}")
print(f"Reason: {algorithms['ID3']['reason_high_variance']}")
print("\nDetailed explanation:")
print("  • ID3 grows trees to maximum depth without stopping criteria")
print("  • No built-in pruning mechanisms")
print("  • Creates overly complex trees that memorize training data")
print("  • High variance leads to poor generalization")
print("  • Requires manual post-pruning for practical use")

# 3. How does CART's binary splitting strategy affect the bias-variance trade-off?
print("\n\n3. How does CART's binary splitting strategy affect the bias-variance trade-off?")
print("-" * 80)
print("Answer: Binary splitting provides better bias-variance balance")
print(f"Bias Level: {algorithms['CART']['bias']}")
print(f"Variance Level: {algorithms['CART']['variance']}")
print("\nDetailed explanation:")
print("  • Binary splits create more balanced partitions")
print("  • Reduces overfitting by limiting tree depth")
print("  • More stable decision boundaries")
print("  • Better generalization to unseen data")
print("  • Cost-complexity pruning further optimizes the trade-off")

# 4. Which algorithm provides the best built-in protection against overfitting?
print("\n\n4. Which algorithm provides the best built-in protection against overfitting?")
print("-" * 80)
print("Answer: CART provides the best built-in protection against overfitting")
print(f"Protection Level: {algorithms['CART']['pruning_protection']}")
print(f"Reason: {algorithms['CART']['reason_low_variance']}")
print("\nDetailed explanation:")
print("  • Cost-complexity pruning with cross-validation")
print("  • Automatic α parameter selection")
print("  • Binary splitting reduces model complexity")
print("  • Surrogate splits handle missing values robustly")
print("  • Production-ready with minimal manual tuning")

# Create separate visualizations for each aspect

# Plot 1: Bias Comparison
plt.figure(figsize=(10, 6))
algo_names = list(algorithms.keys())
bias_levels = {'High': 3, 'Medium': 2, 'Low': 1}
bias_values = [bias_levels[algorithms[algo]['bias']] for algo in algo_names]
colors = [algorithms[algo]['color'] for algo in algo_names]

bars1 = plt.bar(algo_names, bias_values, color=colors, alpha=0.7, edgecolor='black')
plt.ylabel('Bias Level (Higher = More Bias)')
plt.ylim(0, 4)
plt.yticks([1, 2, 3], ['Low', 'Medium', 'High'])
plt.grid(True, alpha=0.3)
plt.title('Bias Levels by Algorithm', fontsize=14, fontweight='bold')

# Add bias labels on bars
for bar, bias_val, algo in zip(bars1, bias_values, algo_names):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             algorithms[algo]['bias'], ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'bias_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Variance Comparison
plt.figure(figsize=(10, 6))
variance_levels = {'Very High': 4, 'High': 3, 'Medium': 2, 'Low': 1}
variance_values = [variance_levels[algorithms[algo]['variance']] for algo in algo_names]

bars2 = plt.bar(algo_names, variance_values, color=colors, alpha=0.7, edgecolor='black')
plt.ylabel('Variance Level (Higher = More Variance)')
plt.ylim(0, 5)
plt.yticks([1, 2, 3, 4], ['Low', 'Medium', 'High', 'Very High'])
plt.grid(True, alpha=0.3)
plt.title('Variance Levels by Algorithm', fontsize=14, fontweight='bold')

# Add variance labels on bars
for bar, var_val, algo in zip(bars2, variance_values, algo_names):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             algorithms[algo]['variance'], ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'variance_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Overfitting Tendency
plt.figure(figsize=(10, 6))
overfitting_levels = {'Very High': 4, 'High': 3, 'Medium': 2, 'Low': 1}
overfitting_values = [overfitting_levels[algorithms[algo]['overfitting_tendency']] for algo in algo_names]

bars3 = plt.bar(algo_names, overfitting_values, color=colors, alpha=0.7, edgecolor='black')
plt.ylabel('Overfitting Tendency (Higher = More Overfitting)')
plt.ylim(0, 5)
plt.yticks([1, 2, 3, 4], ['Low', 'Medium', 'High', 'Very High'])
plt.grid(True, alpha=0.3)
plt.title('Overfitting Tendency', fontsize=14, fontweight='bold')

# Add overfitting labels on bars
for bar, overfit_val, algo in zip(bars3, overfitting_values, algo_names):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             algorithms[algo]['overfitting_tendency'], ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'overfitting_tendency.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Bias-Variance Trade-off Visualization
plt.figure(figsize=(10, 8))
plt.title('Bias-Variance Trade-off Space', fontsize=14, fontweight='bold')

# Create bias-variance space
bias_range = np.linspace(0, 1, 100)
variance_range = np.linspace(0, 1, 100)

# Create meshgrid for visualization
B, V = np.meshgrid(bias_range, variance_range)

# Calculate total error (bias² + variance)
total_error = B**2 + V

# Plot contours
contour = plt.contour(B, V, total_error, levels=[0.2, 0.4, 0.6, 0.8, 1.0], colors='gray', alpha=0.5)
plt.clabel(contour, inline=True, fontsize=10)

# Plot algorithm positions
algo_positions = {
    'ID3': (0.8, 0.9),      # High bias, very high variance
    'C4.5': (0.5, 0.6),     # Medium bias, medium variance
    'CART': (0.2, 0.3)      # Low bias, low variance
}

for algo_name, (bias_pos, var_pos) in algo_positions.items():
    color = algorithms[algo_name]['color']
    plt.scatter(bias_pos, var_pos, s=200, c=color, edgecolor='black', linewidth=2, 
                label=algo_name, zorder=5)
    plt.annotate(algo_name, (bias_pos, var_pos), xytext=(10, 10),
                 textcoords='offset points', fontweight='bold', fontsize=12)

plt.xlabel('Bias (Higher = More Bias)')
plt.ylabel('Variance (Higher = More Variance)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'bias_variance_tradeoff_space.png'), dpi=300, bbox_inches='tight')
plt.close()



# Create detailed comparison table
print("\n" + "=" * 80)
print("DETAILED BIAS-VARIANCE COMPARISON TABLE")
print("=" * 80)

comparison_data = {
    'Algorithm': [
        'ID3',
        'C4.5',
        'CART'
    ],
    'Bias Level': [
        'High',
        'Medium', 
        'Low'
    ],
    'Variance Level': [
        'Very High',
        'Medium',
        'Low'
    ],
    'Overfitting Tendency': [
        'Very High',
        'Medium',
        'Low'
    ],
    'Pruning Protection': [
        'None',
        'Good',
        'Excellent'
    ],
    'Best Use Case': [
        'Educational, simple datasets',
        'Interpretable models, medical',
        'Production systems, large datasets'
    ]
}

df_comparison = pd.DataFrame(comparison_data)
print(df_comparison.to_string(index=False))

# Mathematical analysis of bias-variance decomposition
print("\n" + "=" * 80)
print("MATHEMATICAL BIAS-VARIANCE DECOMPOSITION")
print("=" * 80)

print("The expected prediction error can be decomposed as:")
print("  $E[(y - \\hat{{f}}(x))^2] = \\text{{Bias}}^2(\\hat{{f}}(x)) + \\text{{Var}}(\\hat{{f}}(x)) + \\sigma^2$")
print("  where:")
print("    • $y$ = true value")
print("    • $\\hat{{f}}(x)$ = predicted value")
print("    • $\\text{{Bias}}^2 = (E[\\hat{{f}}(x)] - f(x))^2$")
print("    • $\\text{{Var}} = E[(\\hat{{f}}(x) - E[\\hat{{f}}(x)])^2]$")
print("    • $\\sigma^2$ = irreducible error")

print("\nFor decision trees:")
print("  • Bias: How well the model captures the true underlying function")
print("  • Variance: How much the model changes with different training data")
print("  • Trade-off: Reducing bias often increases variance and vice versa")

# Algorithm-specific analysis
print("\n" + "=" * 80)
print("ALGORITHM-SPECIFIC BIAS-VARIANCE ANALYSIS")
print("=" * 80)

print("1. ID3 Analysis:")
print("   • High Bias:")
print("     - Simple entropy-based splitting")
print("     - No optimization for complex patterns")
print("     - Greedy local optimization")
print("   • Very High Variance:")
print("     - No pruning or regularization")
print("     - Grows to maximum depth")
print("     - Memorizes training data")

print("\n2. C4.5 Analysis:")
print("   • Medium Bias:")
print("     - Gain ratio reduces bias toward multi-valued features")
print("     - Statistical pruning improves generalization")
print("     - Better feature selection")
print("   • Medium Variance:")
print("     - Pessimistic error pruning")
print("     - Confidence interval-based decisions")
print("     - Moderate regularization")

print("\n3. CART Analysis:")
print("   • Low Bias:")
print("     - Binary splitting optimization")
print("     - Cost-complexity pruning")
print("     - Cross-validation for parameter selection")
print("   • Low Variance:")
print("     - Strong regularization")
print("     - Stable decision boundaries")
print("     - Robust to training data variations")

# Practical implications
print("\n" + "=" * 80)
print("PRACTICAL IMPLICATIONS")
print("=" * 80)

print("1. Dataset Size Considerations:")
print("   • Small datasets: ID3 may overfit, CART provides better generalization")
print("   • Large datasets: All algorithms can work well, CART scales best")
print("   • Noisy data: CART's regularization is most beneficial")

print("\n2. Interpretability vs. Performance:")
print("   • ID3: Most interpretable but poorest performance")
print("   • C4.5: Good balance of interpretability and performance")
print("   • CART: Best performance but may be less interpretable")

print("\n3. Production Considerations:")
print("   • ID3: Educational and prototyping only")
print("   • C4.5: Research and interpretable applications")
print("   • CART: Production systems and real-time prediction")

# Overfitting prevention strategies
print("\n" + "=" * 80)
print("OVERFITTING PREVENTION STRATEGIES")
print("=" * 80)

print("1. Pruning Methods:")
print("   • ID3: Manual post-pruning required")
print("   • C4.5: Built-in pessimistic error pruning")
print("   • CART: Cost-complexity pruning with cross-validation")

print("\n2. Regularization Techniques:")
print("   • ID3: None")
print("   • C4.5: Statistical confidence intervals")
print("   • CART: α parameter controls tree complexity")

print("\n3. Validation Strategies:")
print("   • ID3: Manual cross-validation")
print("   • C4.5: Built-in statistical validation")
print("   • CART: Automatic cross-validation for α selection")

print(f"\nPlots saved to: {save_dir}")
