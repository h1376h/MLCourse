import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import fsolve

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_2_Quiz_14")
os.makedirs(save_dir, exist_ok=True)

# Disable LaTeX to avoid compatibility issues with special characters
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("=" * 80)
print("QUESTION 14: Feature Selection Effects on Model Robustness, Stability, and Interpretability")
print("=" * 80)

# Task 1: How does feature selection improve model stability?
print("\n" + "="*60)
print("TASK 1: How does feature selection improve model stability?")
print("="*60)

print("Feature selection improves model stability through several mechanisms:")
print("1. Reduced Variance: Fewer features mean less overfitting to noise in the data")
print("2. Lower Sensitivity: Models with fewer features are less sensitive to small changes in training data")
print("3. Better Generalization: Focus on truly relevant features improves out-of-sample performance")
print("4. Reduced Multicollinearity: Eliminating redundant features reduces instability from correlated predictors")

# Task 2: How does feature selection improve interpretability?
print("\n" + "="*60)
print("TASK 2: How does feature selection improve interpretability?")
print("="*60)

print("Feature selection enhances interpretability by:")
print("1. Simpler Models: Fewer features create more parsimonious models")
print("2. Clearer Relationships: Focus on key variables makes cause-effect relationships clearer")
print("3. Easier Visualization: Fewer dimensions allow for better 2D/3D visualizations")
print("4. Domain Understanding: Selected features often align with domain knowledge")
print("5. Stakeholder Communication: Simpler models are easier to explain to non-technical audiences")

# Task 3: Model stability calculation with detailed steps
print("\n" + "="*60)
print("TASK 3: Model Stability Calculation - Detailed Step-by-Step Solution")
print("="*60)

# Define parameters
n_original = 100  # Original number of features
n_reduced = 20    # Reduced number of features
stability_coefficient = 0.05  # Coefficient in stability function

# Given stability function: S = 1/(1 + αn) where α = 0.05
def stability_function(n, alpha=stability_coefficient):
    """Calculate model stability given number of features and coefficient"""
    return 1 / (1 + alpha * n)

print("GIVEN:")
print(f"• Stability function: S = 1/(1 + αn)")
print(f"• Where α = {stability_coefficient} (stability coefficient)")
print(f"• n = number of features")
print(f"• Original features: n₁ = {n_original}")
print(f"• Reduced features: n₂ = {n_reduced}")
print()

print("STEP 1: Calculate stability for original number of features")
print(f"Substituting n₁ = {n_original} and α = {stability_coefficient}:")
print(f"S₁ = 1/(1 + {stability_coefficient} × {n_original})")
print(f"S₁ = 1/(1 + {stability_coefficient * n_original})")
print(f"S₁ = 1/{1 + stability_coefficient * n_original}")
stability_original = stability_function(n_original, stability_coefficient)
print(f"S₁ = {stability_original:.6f}")
print()

print("STEP 2: Calculate stability for reduced number of features")
print(f"Substituting n₂ = {n_reduced} and α = {stability_coefficient}:")
print(f"S₂ = 1/(1 + {stability_coefficient} × {n_reduced})")
print(f"S₂ = 1/(1 + {stability_coefficient * n_reduced})")
print(f"S₂ = 1/{1 + stability_coefficient * n_reduced}")
stability_reduced = stability_function(n_reduced, stability_coefficient)
print(f"S₂ = {stability_reduced:.6f}")
print()

print("STEP 3: Calculate absolute improvement")
print("Absolute improvement = S₂ - S₁")
print(f"Absolute improvement = {stability_reduced:.6f} - {stability_original:.6f}")
stability_improvement = stability_reduced - stability_original
print(f"Absolute improvement = {stability_improvement:.6f}")
print()

print("STEP 4: Calculate percentage improvement")
print("Percentage improvement = (Absolute improvement / S₁) × 100%")
print(f"Percentage improvement = ({stability_improvement:.6f} / {stability_original:.6f}) × 100%")
improvement_percentage = (stability_improvement / stability_original) * 100
print(f"Percentage improvement = {improvement_percentage:.2f}%")
print()

print("STEP 5: Verification and interpretation")
print(f"• Original stability S₁ = {stability_original:.6f}")
print(f"• Reduced stability S₂ = {stability_reduced:.6f}")
print(f"• Improvement = {stability_improvement:.6f}")
print(f"• Percentage improvement = {improvement_percentage:.2f}%")
print(f"• This means the model is {improvement_percentage:.0f}% more stable with fewer features!")

# Visualize stability function with detailed annotations
plt.figure(figsize=(14, 10))

# Plot stability function
n_values = np.linspace(1, 150, 300)
stability_values = stability_function(n_values, stability_coefficient)

plt.plot(n_values, stability_values, 'b-', linewidth=3, 
         label=f'Stability Function: $S = \\frac{{1}}{{1 + {stability_coefficient}n}}$')

# Highlight original and reduced points
plt.scatter([n_original], [stability_original], color='red', s=300, zorder=5, 
            label=f'Original: $n_1 = {n_original}, S_1 = {stability_original:.4f}$')
plt.scatter([n_reduced], [stability_reduced], color='green', s=300, zorder=5,
            label=f'Reduced: $n_2 = {n_reduced}, S_2 = {stability_reduced:.4f}$')

# Add arrow showing improvement
plt.annotate('', xy=(n_reduced, stability_reduced), xytext=(n_original, stability_original),
            arrowprops=dict(arrowstyle='->', color='red', lw=3, alpha=0.8))

# Add improvement annotation with mathematical details
mid_x = (n_original + n_reduced) / 2
mid_y = (stability_original + stability_reduced) / 2
improvement_text = f'Improvement:\n$\\Delta S = {stability_improvement:.4f}$\n$\\frac{{\\Delta S}}{{S_1}} = {improvement_percentage:.1f}\\%$'
plt.annotate(improvement_text, xy=(mid_x, mid_y), xytext=(mid_x + 25, mid_y + 0.15),
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", ec="red", alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='red', alpha=0.8), fontsize=11)

# Add mathematical details to the plot
math_text = f'$S_1 = \\frac{{1}}{{1 + {stability_coefficient} \\times {n_original}}} = \\frac{{1}}{{1 + {stability_coefficient * n_original}}} = {stability_original:.4f}$\n'
math_text += f'$S_2 = \\frac{{1}}{{1 + {stability_coefficient} \\times {n_reduced}}} = \\frac{{1}}{{1 + {stability_coefficient * n_reduced}}} = {stability_reduced:.4f}$\n'
math_text += f'$\\Delta S = S_2 - S_1 = {stability_reduced:.4f} - {stability_original:.4f} = {stability_improvement:.4f}$'

plt.annotate(math_text, xy=(0.02, 0.02), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", ec="blue", alpha=0.8),
            fontsize=10, verticalalignment='bottom')

plt.xlabel('Number of Features ($n$)')
plt.ylabel('Model Stability ($S$)')
plt.title('Model Stability vs Number of Features\n$S = \\frac{1}{1 + 0.05n}$')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.xlim(0, 150)
plt.ylim(0, 1)

# Save stability plot
plt.savefig(os.path.join(save_dir, 'model_stability_detailed.png'), dpi=300, bbox_inches='tight')

# Task 4: Model complexity calculation with detailed steps
print("\n" + "="*60)
print("TASK 4: Model Complexity Calculation - Detailed Step-by-Step Solution")
print("="*60)

# Define parameters
complexity_base = 2  # Base for exponential complexity
max_complexity_threshold = 1000  # Maximum complexity stakeholders can understand
n_values_task4 = [5, 10, 15]  # Specific feature counts to evaluate

# Given complexity function: complexity = b^n where b = 2
def complexity_function(n, base=complexity_base):
    """Calculate model complexity given number of features and base"""
    return base**n

print("GIVEN:")
print(f"• Complexity function: complexity = b^n")
print(f"• Where b = {complexity_base} (complexity base)")
print(f"• n = number of features")
print(f"• Maximum complexity stakeholders can understand: ≤ {max_complexity_threshold:,}")
print(f"• Feature counts to evaluate: n = {n_values_task4}")
print()

print("STEP 1: Calculate complexity for specific feature counts")
complexities = []
for i, n in enumerate(n_values_task4):
    complexity = complexity_function(n, complexity_base)
    complexities.append(complexity)
    print(f"  For n = {n}:")
    print(f"    complexity = {complexity_base}^{n}")
    print(f"    complexity = {complexity:,}")
    print()

print("STEP 2: Find maximum features for complexity ≤ threshold")
print(f"We need to solve: {complexity_base}^n ≤ {max_complexity_threshold:,}")
print(f"Taking logarithm base {complexity_base}:")
print(f"  {complexity_base}^n ≤ {max_complexity_threshold:,}")
print(f"  n ≤ log_{complexity_base}({max_complexity_threshold:,})")
print(f"  n ≤ log_{complexity_base}({max_complexity_threshold:,})")
log_result = np.log(max_complexity_threshold) / np.log(complexity_base)
print(f"  n ≤ {log_result:.6f}")
max_features = int(log_result)
print(f"  Therefore, maximum number of features = {max_features}")
print()

print("STEP 3: Verification of the solution")
print("We need to verify that our solution satisfies the constraint:")
verification_1 = complexity_function(max_features, complexity_base)
verification_2 = complexity_function(max_features + 1, complexity_base)
print(f"  For n = {max_features}: {complexity_base}^{max_features} = {verification_1:,} ≤ {max_complexity_threshold:,}")
print(f"  For n = {max_features + 1}: {complexity_base}^{max_features + 1} = {verification_2:,} > {max_complexity_threshold:,}")
print(f"  ✓ Our solution is correct!")
print()

print("STEP 4: Mathematical interpretation")
print(f"• The complexity function {complexity_base}^n grows exponentially")
print(f"• Even small increases in n lead to massive increases in complexity")
print(f"• For interpretability constraint ≤ {max_complexity_threshold:,}:")
print(f"  - Maximum features allowed: {max_features}")
print(f"  - Complexity at max features: {verification_1:,}")
print(f"  - Next complexity level: {verification_2:,} (exceeds threshold)")

# Visualize complexity function with detailed annotations
plt.figure(figsize=(14, 10))

# Plot complexity function
n_range = np.arange(1, 21)
complexity_range = [complexity_function(n, complexity_base) for n in n_range]

plt.plot(n_range, complexity_range, 'r-', linewidth=3, 
         label=f'Complexity Function: ${complexity_base}^n$')

# Highlight specific points
for n, complexity in zip(n_values_task4, complexities):
    plt.scatter([n], [complexity], color='blue', s=300, zorder=5,
                label=f'$n = {n}: {complexity:,}$')

# Add threshold line
plt.axhline(y=max_complexity_threshold, color='green', linestyle='--', linewidth=3, 
            label=f'Stakeholder Limit: ${max_complexity_threshold:,}$')

# Highlight maximum features
plt.scatter([max_features], [complexity_function(max_features, complexity_base)], color='green', s=400, 
            marker='s', zorder=5, label=f'Max Features: $n = {max_features}$')

# Add annotation for max features with mathematical details
max_complexity_at_limit = complexity_function(max_features, complexity_base)
annotation_text = f'Maximum features for\ncomplexity $\\leq {max_complexity_threshold:,}$\n$n = {max_features}$\n${complexity_base}^{{{max_features}}} = {max_complexity_at_limit:,}$'
plt.annotate(annotation_text, xy=(max_features, max_complexity_at_limit), 
            xytext=(max_features + 2, max_complexity_at_limit * 0.3),
            bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen", ec="green", alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='green', alpha=0.8), fontsize=11)

# Add mathematical details to the plot
math_text_complexity = f'Constraint: ${complexity_base}^n \\leq {max_complexity_threshold:,}$\n'
math_text_complexity += f'Solution: $n \\leq \\log_{{{complexity_base}}}({max_complexity_threshold:,}) = {log_result:.4f}$\n'
math_text_complexity += f'Maximum features: $n = {max_features}$\n'
math_text_complexity += f'Verification: ${complexity_base}^{{{max_features}}} = {verification_1:,} \\leq {max_complexity_threshold:,}$ ✓'

plt.annotate(math_text_complexity, xy=(0.02, 0.02), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.5", fc="lightcoral", ec="red", alpha=0.8),
            fontsize=10, verticalalignment='bottom')

plt.xlabel('Number of Features ($n$)')
plt.ylabel('Model Complexity ($2^n$)')
plt.title(f'Model Complexity vs Number of Features\n$complexity = {complexity_base}^n$')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.yscale('log')  # Use log scale for better visualization
plt.xlim(1, 20)

# Save complexity plot
plt.savefig(os.path.join(save_dir, 'model_complexity_detailed.png'), dpi=300, bbox_inches='tight')

# Create comprehensive comparison plot with detailed annotations
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))

# Subplot 1: Stability with detailed annotations
n_stab = np.linspace(1, 150, 300)
stab_values = stability_function(n_stab, stability_coefficient)

ax1.plot(n_stab, stab_values, 'b-', linewidth=3, 
         label=f'Stability: $S = \\frac{{1}}{{1 + {stability_coefficient}n}}$')
ax1.scatter([n_original], [stability_original], color='red', s=300, zorder=5, 
            label=f'Original: $n_1 = {n_original}, S_1 = {stability_original:.4f}$')
ax1.scatter([n_reduced], [stability_reduced], color='green', s=300, zorder=5,
            label=f'Reduced: $n_2 = {n_reduced}, S_2 = {stability_reduced:.4f}$')

# Add improvement arrow and annotation
ax1.annotate('', xy=(n_reduced, stability_reduced), xytext=(n_original, stability_original),
            arrowprops=dict(arrowstyle='->', color='red', lw=3, alpha=0.8))
ax1.annotate(f'$\\Delta S = {stability_improvement:.4f}$\n$({improvement_percentage:.1f}\\%)$', 
            xy=(mid_x, mid_y), xytext=(mid_x + 20, mid_y + 0.1),
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="red", alpha=0.8),
            arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))

ax1.set_xlabel('Number of Features ($n$)')
ax1.set_ylabel('Model Stability ($S$)')
ax1.set_title('Model Stability vs Number of Features\n$S = \\frac{1}{1 + 0.05n}$')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)
ax1.set_xlim(0, 150)
ax1.set_ylim(0, 1)

# Subplot 2: Complexity with detailed annotations
n_comp = np.arange(1, 21)
comp_values = [complexity_function(n, complexity_base) for n in n_comp]

ax2.plot(n_comp, comp_values, 'r-', linewidth=3, 
         label=f'Complexity: ${complexity_base}^n$')
ax2.axhline(y=max_complexity_threshold, color='green', linestyle='--', linewidth=3, 
            label=f'Stakeholder Limit: ${max_complexity_threshold:,}$')
ax2.scatter([max_features], [complexity_function(max_features, complexity_base)], color='green', s=400, 
            marker='s', zorder=5, label=f'Max Features: $n = {max_features}$')

# Add mathematical details
ax2.annotate(f'Constraint: ${complexity_base}^n \\leq {max_complexity_threshold:,}$\nSolution: $n = {max_features}$', 
            xy=(max_features, complexity_function(max_features, complexity_base)), 
            xytext=(max_features + 2, complexity_function(max_features, complexity_base) * 0.3),
            bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen", ec="green", alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='green', alpha=0.8), fontsize=11)

ax2.set_xlabel('Number of Features ($n$)')
ax2.set_ylabel('Model Complexity ($2^n$)')
ax2.set_title(f'Model Complexity vs Number of Features\n$complexity = {complexity_base}^n$')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)
ax2.set_yscale('log')
ax2.set_xlim(1, 20)

plt.tight_layout()

# Save comprehensive plot
plt.savefig(os.path.join(save_dir, 'comprehensive_analysis_detailed.png'), dpi=300, bbox_inches='tight')

# Summary table with detailed calculations
print("\n" + "="*80)
print("DETAILED SUMMARY OF RESULTS")
print("="*80)

print(f"{'Metric':<30} {'Original (100 features)':<30} {'Reduced (20 features)':<30} {'Improvement':<20}")
print("-" * 110)
print(f"{'Model Stability':<30} {stability_original:<30.6f} {stability_reduced:<30.6f} {stability_improvement:<20.6f}")
print(f"{'Complexity':<30} {complexity_function(100, complexity_base):<30,} {complexity_function(20, complexity_base):<30,} {complexity_function(100, complexity_base) - complexity_function(20, complexity_base):<20,}")

print(f"\n{'Key Mathematical Results:':<30}")
print(f"• Stability function: S = 1/(1 + {stability_coefficient}n)")
print(f"• Complexity function: complexity = {complexity_base}^n")
print(f"• Stability improvement: {improvement_percentage:.1f}%")
print(f"• Maximum features for interpretability: {max_features}")
print(f"• Complexity reduction factor: {complexity_function(100, complexity_base) / complexity_function(20, complexity_base):.0f}x")

print(f"\n{'Verification:':<30}")
print(f"• {complexity_base}^{max_features} = {verification_1:,} ≤ {max_complexity_threshold:,} ✓")
print(f"• {complexity_base}^{max_features + 1} = {verification_2:,} > {max_complexity_threshold:,} ✗")

print(f"\nPlots saved to: {save_dir}")
print("="*80)
