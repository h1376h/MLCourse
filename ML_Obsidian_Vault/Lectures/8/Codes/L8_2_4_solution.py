import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
import pandas as pd

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_2_Quiz_4")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting (disabled due to Unicode issues)
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'

print("="*80)
print("PEARSON CORRELATION COEFFICIENT - STEP BY STEP SOLUTION")
print("="*80)

# 1. Formula for Pearson correlation
print("\n1. FORMULA FOR PEARSON CORRELATION")
print("-" * 40)
print("The Pearson correlation coefficient r is defined as:")
print("r = Σ[(xi - x̄)(yi - ȳ)] / √[Σ(xi - x̄)² × Σ(yi - ȳ)²]")
print("\nAlternatively:")
print("r = Cov(X,Y) / (σx × σy)")
print("\nWhere:")
print("- x̄, ȳ are the sample means")
print("- σx, σy are the sample standard deviations")
print("- Cov(X,Y) is the covariance between X and Y")

# 2. Range of values
print("\n2. RANGE OF PEARSON CORRELATION VALUES")
print("-" * 40)
print("Pearson correlation coefficient can take values in the range [-1, 1]:")
print("- r = 1:  Perfect positive linear relationship")
print("- r = 0:  No linear relationship")
print("- r = -1: Perfect negative linear relationship")
print("- |r| close to 1: Strong linear relationship")
print("- |r| close to 0: Weak linear relationship")

# 3. Step-by-step calculation for given data
print("\n3. STEP-BY-STEP CALCULATION")
print("-" * 40)

X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 5, 4, 6])

print(f"Given data:")
print(f"X = {X}")
print(f"Y = {Y}")
print(f"n = {len(X)} data points")

# Step 3a: Calculate means
mean_X = np.mean(X)
mean_Y = np.mean(Y)
print(f"\nStep 1: Calculate means")
print(f"x̄ = Σxi/n = ({'+'.join(map(str, X))})/5 = {X.sum()}/5 = {mean_X}")
print(f"ȳ = Σyi/n = ({'+'.join(map(str, Y))})/5 = {Y.sum()}/5 = {mean_Y}")

# Step 3b: Calculate deviations
deviations_X = X - mean_X
deviations_Y = Y - mean_Y
print(f"\nStep 2: Calculate deviations from means")
print(f"xi - x̄: {deviations_X}")
print(f"yi - ȳ: {deviations_Y}")

# Create detailed calculation table
print(f"\nDetailed calculation table:")
print(f"{'i':<3} {'xi':<3} {'yi':<3} {'xi-x̄':<8} {'yi-ȳ':<8} {'(xi-x̄)(yi-ȳ)':<15} {'(xi-x̄)²':<10} {'(yi-ȳ)²':<10}")
print("-" * 70)

products = deviations_X * deviations_Y
squared_dev_X = deviations_X ** 2
squared_dev_Y = deviations_Y ** 2

for i in range(len(X)):
    print(f"{i+1:<3} {X[i]:<3} {Y[i]:<3} {deviations_X[i]:<8.1f} {deviations_Y[i]:<8.1f} {products[i]:<15.1f} {squared_dev_X[i]:<10.1f} {squared_dev_Y[i]:<10.1f}")

# Step 3c: Calculate sums
sum_products = np.sum(products)
sum_squared_X = np.sum(squared_dev_X)
sum_squared_Y = np.sum(squared_dev_Y)

print(f"\nStep 3: Calculate sums")
print(f"Σ[(xi - x̄)(yi - ȳ)] = {sum_products}")
print(f"Σ(xi - x̄)² = {sum_squared_X}")
print(f"Σ(yi - ȳ)² = {sum_squared_Y}")

# Step 3d: Calculate correlation
denominator = np.sqrt(sum_squared_X * sum_squared_Y)
correlation_manual = sum_products / denominator

print(f"\nStep 4: Calculate correlation coefficient")
print(f"r = Σ[(xi - x̄)(yi - ȳ)] / √[Σ(xi - x̄)² × Σ(yi - ȳ)²]")
print(f"r = {sum_products} / √[{sum_squared_X} × {sum_squared_Y}]")
print(f"r = {sum_products} / √{sum_squared_X * sum_squared_Y}")
print(f"r = {sum_products} / {denominator:.4f}")
print(f"r = {correlation_manual:.4f}")

# Verify with built-in function
correlation_builtin, p_value = pearsonr(X, Y)
print(f"\nVerification with scipy.stats.pearsonr: r = {correlation_builtin:.4f}")

# 4. Linear relationship Y = aX + b
print("\n4. CORRELATION FOR LINEAR RELATIONSHIP Y = aX + b")
print("-" * 50)
print("For a perfect linear relationship Y = aX + b:")
print("- If a > 0: r = +1 (perfect positive correlation)")
print("- If a < 0: r = -1 (perfect negative correlation)")
print("- If a = 0: r = 0 (no correlation, Y is constant)")
print("\nThe correlation coefficient is independent of the intercept b")
print("and only depends on the sign of the slope a.")

# Example with different slopes
slopes = [2, -1.5, 0]
intercepts = [1, 3, 5]
X_linear = np.array([1, 2, 3, 4, 5])

print(f"\nExamples:")
for i, (a, b) in enumerate(zip(slopes, intercepts)):
    Y_linear = a * X_linear + b
    if a != 0:
        r_linear, _ = pearsonr(X_linear, Y_linear)
        print(f"Y = {a}X + {b}: r = {r_linear:.1f}")
    else:
        print(f"Y = {a}X + {b} = {b} (constant): r = undefined (or 0)")

# 5. Calculate correlation for X = [1,2,3,4,5] and Y = [1,4,9,16,25]
print("\n5. CORRELATION FOR QUADRATIC RELATIONSHIP")
print("-" * 45)

X2 = np.array([1, 2, 3, 4, 5])
Y2 = np.array([1, 4, 9, 16, 25])  # Y = X²

print(f"Given data:")
print(f"X = {X2}")
print(f"Y = {Y2}  (Note: Y = X²)")

# Calculate correlation for quadratic relationship
correlation_quad, _ = pearsonr(X2, Y2)
print(f"\nPearson correlation coefficient: r = {correlation_quad:.4f}")
print(f"\nNote: Even though there's a perfect mathematical relationship (Y = X²),")
print(f"the Pearson correlation is not ±1 because the relationship is not LINEAR.")
print(f"Pearson correlation only measures LINEAR relationships.")

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Pearson Correlation Coefficient Analysis', fontsize=16, fontweight='bold')

# Plot 1: Original data with correlation calculation
ax1 = axes[0, 0]
ax1.scatter(X, Y, s=100, color='blue', alpha=0.7, edgecolor='black')
ax1.plot(X, Y, 'b--', alpha=0.5, label='Data trend')
ax1.axhline(y=mean_Y, color='red', linestyle=':', alpha=0.7, label=f'$\\overline{{y}} = {mean_Y}$')
ax1.axvline(x=mean_X, color='red', linestyle=':', alpha=0.7, label=f'$\\overline{{x}} = {mean_X}$')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title(f'Original Data\n$r = {correlation_manual:.4f}$')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Annotate points
for i, (xi, yi) in enumerate(zip(X, Y)):
    ax1.annotate(f'({xi}, {yi})', (xi, yi), xytext=(5, 5), 
                textcoords='offset points', fontsize=9)

# Plot 2: Deviations visualization
ax2 = axes[0, 1]
ax2.scatter(deviations_X, deviations_Y, s=100, color='green', alpha=0.7, edgecolor='black')
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
ax2.set_xlabel('$x_i - \\bar{x}$')
ax2.set_ylabel('$y_i - \\bar{y}$')
ax2.set_title('Deviations from Means')
ax2.grid(True, alpha=0.3)

# Color code quadrants
for i, (dx, dy) in enumerate(zip(deviations_X, deviations_Y)):
    color = 'red' if dx * dy > 0 else 'blue'
    ax2.scatter(dx, dy, s=100, color=color, alpha=0.7, edgecolor='black')
    ax2.annotate(f'P{i+1}', (dx, dy), xytext=(5, 5), 
                textcoords='offset points', fontsize=9)

# Plot 3: Linear relationships with different slopes
ax3 = axes[0, 2]
X_demo = np.linspace(0, 5, 50)
colors = ['red', 'blue', 'green']
for i, (a, color) in enumerate(zip([2, -1.5, 0], colors)):
    Y_demo = a * X_demo + 1
    if a != 0:
        r_demo, _ = pearsonr(X_demo, Y_demo)
        ax3.plot(X_demo, Y_demo, color=color, linewidth=2, 
                label=f'$Y = {a}X + 1$, $r = {r_demo:.1f}$')
    else:
        ax3.plot(X_demo, Y_demo, color=color, linewidth=2, 
                label=f'$Y = {a}X + 1 = 1$, $r = 0$')

ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_title('Linear Relationships')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Plot 4: Correlation interpretation scale
ax4 = axes[1, 0]
corr_values = np.array([-1, -0.7, -0.3, 0, 0.3, 0.7, 1])
colors_scale = ['darkred', 'red', 'orange', 'gray', 'lightblue', 'blue', 'darkblue']
labels = ['Perfect\nNegative', 'Strong\nNegative', 'Weak\nNegative', 'No\nCorrelation', 
          'Weak\nPositive', 'Strong\nPositive', 'Perfect\nPositive']

for i, (val, color, label) in enumerate(zip(corr_values, colors_scale, labels)):
    ax4.bar(i, 1, color=color, alpha=0.7, edgecolor='black')
    ax4.text(i, 0.5, f'{val}', ha='center', va='center', fontweight='bold', color='white')
    ax4.text(i, -0.2, label, ha='center', va='top', fontsize=8, rotation=0)

ax4.set_xlim(-0.5, len(corr_values) - 0.5)
ax4.set_ylim(-0.5, 1.2)
ax4.set_title('Correlation Coefficient Interpretation Scale')
ax4.set_xticks([])
ax4.set_yticks([])

# Mark our calculated correlation
our_corr_pos = np.interp(correlation_manual, corr_values, range(len(corr_values)))
ax4.axvline(x=our_corr_pos, color='yellow', linewidth=3, 
           label=f'Our result: {correlation_manual:.4f}')
ax4.legend()

# Plot 5: Quadratic relationship
ax5 = axes[1, 1]
ax5.scatter(X2, Y2, s=100, color='purple', alpha=0.7, edgecolor='black')
ax5.plot(X2, Y2, 'purple', linewidth=2, alpha=0.7)
ax5.set_xlabel('X')
ax5.set_ylabel('Y')
ax5.set_title(f'Quadratic Relationship\n$Y = X^2$, $r = {correlation_quad:.4f}$')
ax5.grid(True, alpha=0.3)

# Annotate points
for i, (xi, yi) in enumerate(zip(X2, Y2)):
    ax5.annotate(f'({xi}, {yi})', (xi, yi), xytext=(5, 5), 
                textcoords='offset points', fontsize=9)

# Plot 6: Step-by-step calculation visualization
ax6 = axes[1, 2]
steps = ['Calculate\nMeans', 'Calculate\nDeviations', 'Multiply\nDeviations', 
         'Sum Products\n& Squares', 'Divide by\nSqrt Product']
values = [f'x̄={mean_X}, ȳ={mean_Y}', f'Range: {deviations_X.min():.1f} to {deviations_X.max():.1f}',
          f'Sum = {sum_products}', f'√({sum_squared_X}×{sum_squared_Y}) = {denominator:.2f}',
          f'r = {correlation_manual:.4f}']

y_pos = np.arange(len(steps))
bars = ax6.barh(y_pos, [1]*len(steps), color=['lightcoral', 'lightblue', 'lightgreen', 'lightyellow', 'lightpink'])

for i, (step, value) in enumerate(zip(steps, values)):
    ax6.text(0.5, i, f'{step}\n{value}', ha='center', va='center', fontsize=9, fontweight='bold')

ax6.set_yticks([])
ax6.set_xlim(0, 1)
ax6.set_title('Calculation Steps')
ax6.set_xticks([])

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'pearson_correlation_comprehensive.png'), dpi=300, bbox_inches='tight')

# Create a detailed step-by-step calculation figure
fig2, ax = plt.subplots(1, 1, figsize=(12, 8))

# Create a table showing the step-by-step calculation
table_data = []
for i in range(len(X)):
    table_data.append([
        f'{i+1}',
        f'{X[i]}',
        f'{Y[i]}',
        f'{deviations_X[i]:.1f}',
        f'{deviations_Y[i]:.1f}',
        f'{products[i]:.1f}',
        f'{squared_dev_X[i]:.1f}',
        f'{squared_dev_Y[i]:.1f}'
    ])

# Add sum row
table_data.append([
    'Sum',
    f'{X.sum()}',
    f'{Y.sum()}',
    '0.0',
    '0.0',
    f'{sum_products:.1f}',
    f'{sum_squared_X:.1f}',
    f'{sum_squared_Y:.1f}'
])

columns = ['i', '$x_i$', '$y_i$', '$x_i - \\overline{x}$', '$y_i - \\overline{y}$', 
           '$(x_i - \\overline{x})(y_i - \\overline{y})$', '$(x_i - \\overline{x})^2$', '$(y_i - \\overline{y})^2$']

# Create table
table = ax.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

# Style the table
for i in range(len(columns)):
    table[(0, i)].set_facecolor('#E6E6FA')
    table[(0, i)].set_text_props(weight='bold')

# Highlight the sum row
for i in range(len(columns)):
    table[(len(table_data), i)].set_facecolor('#FFE4E1')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('Step-by-Step Pearson Correlation Calculation\n' + 
            f'$\\overline{{x}} = {mean_X}, \\overline{{y}} = {mean_Y}$\n' +
            f'$r = \\frac{{{sum_products}}}{{\\sqrt{{{sum_squared_X} \\times {sum_squared_Y}}}}} = \\frac{{{sum_products}}}{{{denominator:.4f}}} = {correlation_manual:.4f}$',
            fontsize=14, fontweight='bold', pad=20)

plt.savefig(os.path.join(save_dir, 'pearson_correlation_calculation_table.png'), dpi=300, bbox_inches='tight')

# Create correlation examples figure
fig3, axes = plt.subplots(2, 3, figsize=(15, 10))
fig3.suptitle('Examples of Different Correlation Values', fontsize=16, fontweight='bold')

# Generate example datasets with different correlations
np.random.seed(42)
n_points = 50

correlations = [0.95, 0.7, 0.3, -0.3, -0.7, -0.95]
titles = ['Strong Positive\n(r ≈ 0.95)', 'Moderate Positive\n(r ≈ 0.7)', 'Weak Positive\n(r ≈ 0.3)',
          'Weak Negative\n(r ≈ -0.3)', 'Moderate Negative\n(r ≈ -0.7)', 'Strong Negative\n(r ≈ -0.95)']

for i, (target_corr, title) in enumerate(zip(correlations, titles)):
    ax = axes[i//3, i%3]
    
    # Generate data with specific correlation
    x = np.random.randn(n_points)
    y = target_corr * x + np.sqrt(1 - target_corr**2) * np.random.randn(n_points)
    
    actual_corr, _ = pearsonr(x, y)
    
    ax.scatter(x, y, alpha=0.6, s=30)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'{title}\nActual r = {actual_corr:.3f}')
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), "r--", alpha=0.8)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'correlation_examples.png'), dpi=300, bbox_inches='tight')

print(f"\n{'='*80}")
print("SUMMARY")
print("="*80)
print(f"1. Pearson correlation formula: r = Σ[(xi - x̄)(yi - ȳ)] / √[Σ(xi - x̄)² × Σ(yi - ȳ)²]")
print(f"2. Range: [-1, 1]")
print(f"3. For X = {X}, Y = {Y}: r = {correlation_manual:.4f}")
print(f"4. For Y = aX + b: r = +1 (if a > 0), r = -1 (if a < 0), r = 0 (if a = 0)")
print(f"5. For X = {X2}, Y = {Y2} (quadratic): r = {correlation_quad:.4f}")
print(f"\nAll plots saved to: {save_dir}")
