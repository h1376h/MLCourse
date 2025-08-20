import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_regression, chi2
from sklearn.preprocessing import KBinsDiscretizer
import seaborn as sns
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_1_Quiz_12")
os.makedirs(save_dir, exist_ok=True)

# Configuration parameters
CORRELATION_THRESHOLD = 0.7
N_SAMPLES = 200
RANDOM_SEED = 42
X_RANGE = (-5, 5)
BINS_FOR_MI = 10
POLYNOMIAL_DEGREES = [2, 3]

# Enable LaTeX style plotting for plot labels only
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'

print("=" * 80)
print("FEATURE-TARGET RELATIONSHIPS ANALYSIS")
print("=" * 80)

# ========================================================================
# PART 1: MEASURING FEATURE-TARGET RELATIONSHIPS FOR NUMERICAL DATA
# ========================================================================

print("\n1. MEASURING FEATURE-TARGET RELATIONSHIPS FOR NUMERICAL DATA")
print("-" * 60)

def generate_relationship_data(n_samples, random_seed):
    """Generate synthetic data with different relationship types"""
    np.random.seed(random_seed)
    
    # Linear relationship
    x_linear = np.random.normal(0, 1, n_samples)
    y_linear = 2 * x_linear + np.random.normal(0, 0.5, n_samples)
    
    # Non-linear relationship (quadratic)
    x_quadratic = np.random.uniform(-3, 3, n_samples)
    y_quadratic = x_quadratic**2 + np.random.normal(0, 1, n_samples)
    
    # Non-monotonic relationship (sine)
    x_sine = np.random.uniform(0, 4*np.pi, n_samples)
    y_sine = np.sin(x_sine) + np.random.normal(0, 0.2, n_samples)
    
    # No relationship
    x_random = np.random.normal(0, 1, n_samples)
    y_random = np.random.normal(0, 1, n_samples)
    
    return {
        'linear': (x_linear, y_linear),
        'quadratic': (x_quadratic, y_quadratic),
        'sine': (x_sine, y_sine),
        'random': (x_random, y_random)
    }

# Generate sample numerical data with different relationship types
relationship_data = generate_relationship_data(N_SAMPLES, RANDOM_SEED)

# Extract data from the generated relationships
x_linear, y_linear = relationship_data['linear']
x_quadratic, y_quadratic = relationship_data['quadratic']
x_sine, y_sine = relationship_data['sine']
x_random, y_random = relationship_data['random']

def calculate_pearson_correlation(x, y):
    """Calculate Pearson correlation coefficient manually and using scipy"""
    # Manual calculation
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Calculate deviations
    x_dev = x - x_mean
    y_dev = y - y_mean
    
    # Calculate products and squares
    xy_products = x_dev * y_dev
    x_squared = x_dev**2
    y_squared = y_dev**2
    
    # Calculate sums
    sum_xy = np.sum(xy_products)
    sum_x_squared = np.sum(x_squared)
    sum_y_squared = np.sum(y_squared)
    
    # Calculate correlation
    numerator = sum_xy
    denominator = np.sqrt(sum_x_squared * sum_y_squared)
    r_manual = numerator / denominator if denominator != 0 else 0
    
    # Using scipy for verification
    r_scipy, p_value = stats.pearsonr(x, y)
    
    return r_manual, r_scipy, p_value, {
        'x_mean': x_mean,
        'y_mean': y_mean,
        'x_dev': x_dev,
        'y_dev': y_dev,
        'xy_products': xy_products,
        'x_squared': x_squared,
        'y_squared': y_squared,
        'sum_xy': sum_xy,
        'sum_x_squared': sum_x_squared,
        'sum_y_squared': sum_y_squared,
        'numerator': numerator,
        'denominator': denominator
    }

def calculate_spearman_correlation(x, y):
    """Calculate Spearman rank correlation"""
    return stats.spearmanr(x, y)

def calculate_mutual_information(x, y, bins=10):
    """Calculate mutual information for continuous variables"""
    # Discretize variables for MI calculation
    discretizer_x = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
    discretizer_y = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
    
    x_discrete = discretizer_x.fit_transform(x.reshape(-1, 1)).ravel()
    y_discrete = discretizer_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    # Calculate MI using sklearn
    mi = mutual_info_regression(x.reshape(-1, 1), y_discrete)[0]
    
    return mi

# Create visualization showing different relationship types
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
relationships = [
    (x_linear, y_linear, "Linear Relationship"),
    (x_quadratic, y_quadratic, "Non-linear (Quadratic)"),
    (x_sine, y_sine, "Non-monotonic (Sine)"),
    (x_random, y_random, "No Relationship")
]

print("\nCorrelation analysis for different relationship types:")
print("-" * 50)

for i, (x, y, title) in enumerate(relationships):
    row, col = i // 2, i % 2
    ax = axes[row, col]
    
    # Calculate different correlation measures
    r_manual, r_scipy, p_value, calc_details = calculate_pearson_correlation(x, y)
    spearman_r, spearman_p = calculate_spearman_correlation(x, y)
    mi = calculate_mutual_information(x, y, BINS_FOR_MI)
    
    # Plot scatter plot with trend line
    ax.scatter(x, y, alpha=0.6, s=30)
    
    # Add trend line for visualization
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_sorted = np.sort(x)
    ax.plot(x_sorted, p(x_sorted), "r--", alpha=0.8, linewidth=2)
    
    ax.set_title(f'{title}')
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.grid(True, alpha=0.3)
    
    # Add correlation statistics to plot
    stats_text = r'Pearson $r$: ' + f'{r_scipy:.3f}\n' + r'Spearman $\rho$: ' + f'{spearman_r:.3f}\n' + r'Mutual Info: ' + f'{mi:.3f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            verticalalignment='top', fontsize=9)
    
    print(f"{title}:")
    print(f"  Pearson correlation: {r_scipy:.4f} (p-value: {p_value:.4f})")
    print(f"  Spearman correlation: {spearman_r:.4f} (p-value: {spearman_p:.4f})")
    print(f"  Mutual Information: {mi:.4f}")
    print()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'relationship_types_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()  # Close figure to free memory

# ========================================================================
# PART 2: DIFFICULT-TO-DETECT RELATIONSHIPS
# ========================================================================

print("\n2. RELATIONSHIPS HARD TO DETECT WITH SIMPLE CORRELATION")
print("-" * 60)

# Create examples of relationships that are hard to detect with Pearson correlation
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Anscombe's quartet-like examples
np.random.seed(42)

# 1. Perfect quadratic (zero linear correlation)
x1 = np.linspace(-2, 2, 100)
y1 = x1**2 + np.random.normal(0, 0.1, 100)

# 2. Circular relationship
theta = np.linspace(0, 2*np.pi, 100)
x2 = np.cos(theta) + np.random.normal(0, 0.1, 100)
y2 = np.sin(theta) + np.random.normal(0, 0.1, 100)

# 3. Categorical relationship
x3 = np.random.choice([1, 2, 3, 4, 5], 100)
y3 = np.where(x3 % 2 == 0, 3, 1) + np.random.normal(0, 0.2, 100)

# 4. Heteroscedastic relationship
x4 = np.random.uniform(0, 10, 100)
y4 = x4 + np.random.normal(0, x4/3, 100)

# 5. Outlier effect
x5 = np.random.normal(0, 1, 98)
y5 = np.random.normal(0, 1, 98)
x5 = np.append(x5, [5, 6])
y5 = np.append(y5, [5, 6])

# 6. Step function
x6 = np.random.uniform(-3, 3, 100)
y6 = np.where(x6 < 0, -1, 1) + np.random.normal(0, 0.2, 100)

difficult_relationships = [
    (x1, y1, "Perfect Quadratic\n(Zero Linear Correlation)"),
    (x2, y2, "Circular Relationship"),
    (x3, y3, "Categorical Relationship"),
    (x4, y4, "Heteroscedastic\n(Varying Variance)"),
    (x5, y5, "Outlier Effect"),
    (x6, y6, "Step Function")
]

print("Analysis of difficult-to-detect relationships:")
print("-" * 50)

for i, (x, y, title) in enumerate(difficult_relationships):
    row, col = i // 3, i % 3
    ax = axes[row, col]
    
    # Calculate correlations
    r_pearson, p_pearson = stats.pearsonr(x, y)
    r_spearman, p_spearman = stats.spearmanr(x, y)
    mi = calculate_mutual_information(x, y)
    
    # Plot
    ax.scatter(x, y, alpha=0.7, s=40)
    ax.set_title(title)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = r'Pearson $r$: ' + f'{r_pearson:.3f}\n' + r'Spearman $\rho$: ' + f'{r_spearman:.3f}\n' + r'MI: ' + f'{mi:.3f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
            verticalalignment='top', fontsize=9)
    
    print(f"{title.replace(chr(10), ' ')}:")
    print(f"  Pearson: {r_pearson:.4f}, Spearman: {r_spearman:.4f}, MI: {mi:.4f}")

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'difficult_relationships.png'), dpi=300, bbox_inches='tight')
plt.close()  # Close figure to free memory

# ========================================================================
# PART 3: HANDLING NON-LINEAR RELATIONSHIPS
# ========================================================================

print("\n\n3. HANDLING NON-LINEAR RELATIONSHIPS IN FEATURE SELECTION")
print("-" * 60)

# Generate non-linear data
np.random.seed(42)
x_nonlinear = np.random.uniform(-3, 3, 200)
y_nonlinear = x_nonlinear**3 - 3*x_nonlinear + np.random.normal(0, 1, 200)

# Methods to handle non-linear relationships
def polynomial_features(x, degree=2):
    """Create polynomial features"""
    features = np.column_stack([x**i for i in range(1, degree+1)])
    return features

def rank_transformation(x):
    """Transform to ranks"""
    return stats.rankdata(x)

def binning_transformation(x, n_bins=5):
    """Transform continuous variable to categorical bins"""
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    return discretizer.fit_transform(x.reshape(-1, 1)).ravel()

# Apply different transformations
x_poly = polynomial_features(x_nonlinear, degree=3)
x_rank = rank_transformation(x_nonlinear)
y_rank = rank_transformation(y_nonlinear)
x_binned = binning_transformation(x_nonlinear)

# Calculate correlations with different approaches
correlations = {
    'Original (Pearson)': stats.pearsonr(x_nonlinear, y_nonlinear)[0],
    'Rank correlation (Spearman)': stats.spearmanr(x_nonlinear, y_nonlinear)[0],
    'Polynomial features (degree 2)': stats.pearsonr(x_poly[:, 1], y_nonlinear)[0],  # x^2
    'Polynomial features (degree 3)': stats.pearsonr(x_poly[:, 2], y_nonlinear)[0],  # x^3
    'Mutual Information': calculate_mutual_information(x_nonlinear, y_nonlinear)
}

# Visualization of non-linear relationship handling
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Original relationship
axes[0, 0].scatter(x_nonlinear, y_nonlinear, alpha=0.6)
axes[0, 0].set_title('Original Non-linear Relationship')
axes[0, 0].set_xlabel(r'$x_1$')
axes[0, 0].set_ylabel(r'$x_2$')
axes[0, 0].grid(True, alpha=0.3)

# Polynomial feature (x^2)
axes[0, 1].scatter(x_poly[:, 1], y_nonlinear, alpha=0.6)
axes[0, 1].set_title(r'$x_1^2$ vs $x_2$')
axes[0, 1].set_xlabel(r'$x_1^2$')
axes[0, 1].set_ylabel(r'$x_2$')
axes[0, 1].grid(True, alpha=0.3)

# Polynomial feature (x^3)
axes[0, 2].scatter(x_poly[:, 2], y_nonlinear, alpha=0.6)
axes[0, 2].set_title(r'$x_1^3$ vs $x_2$')
axes[0, 2].set_xlabel(r'$x_1^3$')
axes[0, 2].set_ylabel(r'$x_2$')
axes[0, 2].grid(True, alpha=0.3)

# Rank transformation
axes[1, 0].scatter(x_rank, y_rank, alpha=0.6)
axes[1, 0].set_title('Rank Transformation')
axes[1, 0].set_xlabel(r'$\text{Rank}(x_1)$')
axes[1, 0].set_ylabel(r'$\text{Rank}(x_2)$')
axes[1, 0].grid(True, alpha=0.3)

# Binned transformation
axes[1, 1].boxplot([y_nonlinear[x_binned == i] for i in range(int(max(x_binned))+1)])
axes[1, 1].set_title('Binned X vs Y Distribution')
axes[1, 1].set_xlabel('X Bins')
axes[1, 1].set_ylabel(r'$x_2$')
axes[1, 1].grid(True, alpha=0.3)

# Correlation comparison
methods = list(correlations.keys())
values = list(correlations.values())
bars = axes[1, 2].bar(range(len(methods)), values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'])
axes[1, 2].set_title('Correlation Comparison')
axes[1, 2].set_ylabel('Correlation/MI Value')
axes[1, 2].set_xticks(range(len(methods)))
axes[1, 2].set_xticklabels(methods, rotation=45, ha='right')
axes[1, 2].grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, values):
    height = bar.get_height()
    axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'nonlinear_handling_methods.png'), dpi=300, bbox_inches='tight')
plt.close()  # Close figure to free memory

print("Correlation analysis with different approaches:")
for method, value in correlations.items():
    print(f"  {method}: {value:.4f}")

# ========================================================================
# PART 4: HIGH MUTUAL INFORMATION, LOW CORRELATION INTERPRETATION
# ========================================================================

print("\n\n4. HIGH MUTUAL INFORMATION, LOW CORRELATION INTERPRETATION")
print("-" * 60)

# Create example with high MI but low correlation
np.random.seed(42)
x_mi = np.random.uniform(-2, 2, 300)
y_mi = np.where(np.abs(x_mi) < 1, 1, -1) + np.random.normal(0, 0.1, 300)

# Calculate both measures
r_mi, p_mi = stats.pearsonr(x_mi, y_mi)
mi_value = calculate_mutual_information(x_mi, y_mi)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(x_mi, y_mi, alpha=0.6, c=y_mi, cmap='RdYlBu')
plt.title('High MI, Low Correlation Example')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.colorbar(label=r'$x_2$ value')
plt.grid(True, alpha=0.3)

# Add statistics text
stats_text = r'Pearson $r$: ' + f'{r_mi:.3f}\n' + r'Mutual Info: ' + f'{mi_value:.3f}'
plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
         verticalalignment='top')

# Information theory visualization
plt.subplot(1, 2, 2)
# Create joint distribution heatmap
from scipy.stats import gaussian_kde
kde = gaussian_kde([x_mi, y_mi])
xi, yi = np.mgrid[-3:3:50j, -2:2:50j]
zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
plt.contourf(xi, yi, zi.reshape(xi.shape), alpha=0.6)
plt.scatter(x_mi, y_mi, alpha=0.3, s=10)
plt.title('Joint Distribution Visualization')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.colorbar(label='Density')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'high_mi_low_correlation.png'), dpi=300, bbox_inches='tight')
plt.close()  # Close figure to free memory

print(f"High MI, Low Correlation Example:")
print(f"  Pearson correlation: {r_mi:.4f}")
print(f"  Mutual Information: {mi_value:.4f}")
print(f"  Interpretation: The feature contains information about the target")
print(f"  but in a non-linear way that Pearson correlation cannot detect.")

# ========================================================================
# PART 5: COMPARISON OF RELATIONSHIP MEASURES
# ========================================================================

print("\n\n5. COMPARISON OF DIFFERENT RELATIONSHIP MEASURES")
print("-" * 60)

# Create comprehensive comparison
np.random.seed(42)

# Generate different types of relationships for comparison
relationships_data = {
    'Strong Linear': (np.random.normal(0, 1, 100), lambda x: 3*x + np.random.normal(0, 0.5, 100)),
    'Weak Linear': (np.random.normal(0, 1, 100), lambda x: 0.5*x + np.random.normal(0, 2, 100)),
    'Quadratic': (np.random.uniform(-2, 2, 100), lambda x: x**2 + np.random.normal(0, 0.5, 100)),
    'Exponential': (np.random.uniform(0, 3, 100), lambda x: np.exp(x/2) + np.random.normal(0, 0.5, 100)),
    'Categorical': (np.random.choice([1, 2, 3, 4], 100), lambda x: np.where(x % 2 == 0, 5, 2) + np.random.normal(0, 0.3, 100)),
    'Sine Wave': (np.random.uniform(0, 4*np.pi, 100), lambda x: np.sin(x) + np.random.normal(0, 0.2, 100))
}

# Calculate chi-square for categorical data
def calculate_chi_square(x, y, bins=5):
    """Calculate chi-square statistic for continuous data"""
    # Discretize both variables
    x_binned = pd.cut(x, bins=bins, labels=False)
    y_binned = pd.cut(y, bins=bins, labels=False)
    
    # Create contingency table
    contingency_table = pd.crosstab(x_binned, y_binned)
    
    # Calculate chi-square
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    
    # Calculate Cramér's V for effect size
    n = contingency_table.sum().sum()
    cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
    
    return chi2_stat, p_value, cramers_v

# Create comparison table
comparison_results = {}

for name, (x_base, y_func) in relationships_data.items():
    y = y_func(x_base)
    
    # Pearson correlation
    r_pearson, p_pearson = stats.pearsonr(x_base, y)
    
    # Spearman correlation
    r_spearman, p_spearman = stats.spearmanr(x_base, y)
    
    # Mutual Information
    mi = calculate_mutual_information(x_base, y)
    
    # Chi-square (for discretized data)
    chi2_stat, chi2_p, cramers_v = calculate_chi_square(x_base, y)
    
    comparison_results[name] = {
        'Pearson r': r_pearson,
        'Spearman rho': r_spearman,
        'Mutual Info': mi,
        'Cramér V': cramers_v,
        'data': (x_base, y)
    }

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, (name, results) in enumerate(comparison_results.items()):
    ax = axes[i]
    x_data, y_data = results['data']
    
    ax.scatter(x_data, y_data, alpha=0.6, s=30)
    ax.set_title(f'{name}')
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.grid(True, alpha=0.3)
    
    # Add all statistics
    stats_text = (r"Pearson $r$: " + f"{results['Pearson r']:.3f}\n"
                  r"Spearman $\rho$: " + f"{results['Spearman rho']:.3f}\n"
                  r"Mutual Info: " + f"{results['Mutual Info']:.3f}\n"
                  r"Cramér V: " + f"{results['Cramér V']:.3f}")
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
            verticalalignment='top', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'measures_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()  # Close figure to free memory

# Create summary table
print("\nComparison of Relationship Measures:")
print("-" * 80)
print(f"{'Relationship':<15} {'Pearson r':<10} {'Spearman rho':<12} {'Mutual Info':<12} {'Cramér V':<10}")
print("-" * 80)

for name, results in comparison_results.items():
    print(f"{name:<15} {results['Pearson r']:<10.3f} {results['Spearman rho']:<12.3f} "
          f"{results['Mutual Info']:<12.3f} {results['Cramér V']:<10.3f}")

# ========================================================================
# PART 6: PEARSON CORRELATION CALCULATION EXAMPLE
# ========================================================================

print("\n\n6. PEARSON CORRELATION CALCULATION EXAMPLE")
print("-" * 60)

# Configuration parameters
CORRELATION_THRESHOLD = 0.7
N_SAMPLES = 200
RANDOM_SEED = 42
X_RANGE = (-5, 5)
BINS_FOR_MI = 10
POLYNOMIAL_DEGREES = [2, 3]

# Given data from the question
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 5, 4, 6])

print(f"Feature X: {X}")
print(f"Target Y:  {Y}")
print()

# Use the improved function for calculation
r_manual, r_scipy, p_value, calc_details = calculate_pearson_correlation(X, Y)

# Extract calculation details
X_mean = calc_details['x_mean']
Y_mean = calc_details['y_mean']
X_dev = calc_details['x_dev']
Y_dev = calc_details['y_dev']
XY_products = calc_details['xy_products']
X_squared = calc_details['x_squared']
Y_squared = calc_details['y_squared']
sum_XY = calc_details['sum_xy']
sum_X_squared = calc_details['sum_x_squared']
sum_Y_squared = calc_details['sum_y_squared']
numerator = calc_details['numerator']
denominator = calc_details['denominator']
r = r_manual

# Display step-by-step calculation
n = len(X)
print(f"Sample size (n): {n}")
print(f"Mean of X: {X_mean:.1f}")
print(f"Mean of Y: {Y_mean:.1f}")
print()

print("Deviations from mean:")
print(f"X - X̄: {X_dev}")
print(f"Y - Ȳ: {Y_dev}")
print()

print("Products and squares:")
print(f"(X - X̄)(Y - Ȳ): {XY_products}")
print(f"(X - X̄)²:       {X_squared}")
print(f"(Y - Ȳ)²:       {Y_squared}")
print()

print("Sums:")
print(f"Σ(X - X̄)(Y - Ȳ): {sum_XY:.1f}")
print(f"Σ(X - X̄)²:       {sum_X_squared:.1f}")
print(f"Σ(Y - Ȳ)²:       {sum_Y_squared:.1f}")
print()

print("Pearson correlation coefficient calculation:")
print(f"r = Σ(X - X̄)(Y - Ȳ) / √[Σ(X - X̄)² × Σ(Y - Ȳ)²]")
print(f"r = {sum_XY:.1f} / √[{sum_X_squared:.1f} × {sum_Y_squared:.1f}]")
print(f"r = {sum_XY:.1f} / √{sum_X_squared * sum_Y_squared:.1f}")
print(f"r = {sum_XY:.1f} / {denominator:.4f}")
print(f"r = {r:.4f}")
print()

print(f"Verification with scipy.stats.pearsonr: {r_scipy:.4f}")
print(f"P-value: {p_value:.4f}")
print()

# Feature selection decision
threshold = CORRELATION_THRESHOLD
abs_r = np.abs(r)

print("Feature Selection Decision:")
print(f"Correlation threshold: {threshold}")
print(f"Calculated correlation: {r:.4f}")
print(f"Absolute correlation: {abs_r:.4f}")
print()

if abs_r >= threshold:
    print(f"✓ Feature SELECTED (|r| = {abs_r:.4f} ≥ {threshold})")
else:
    print(f"✗ Feature REJECTED (|r| = {abs_r:.4f} < {threshold})")

if r >= threshold:
    print(f"Using signed correlation: Feature REJECTED (r = {r:.4f} < {threshold})")
else:
    print(f"Using signed correlation: Feature REJECTED (r = {r:.4f} < {threshold})")

# Visualization of the calculation
plt.figure(figsize=(15, 10))

# Main scatter plot
plt.subplot(2, 3, 1)
plt.scatter(X, Y, s=100, color='red', zorder=5)
for i in range(len(X)):
    plt.annotate(f'({X[i]}, {Y[i]})', (X[i], Y[i]), xytext=(5, 5), 
                textcoords='offset points', fontsize=10)

# Add mean lines
plt.axhline(y=Y_mean, color='blue', linestyle='--', alpha=0.7, label=f'Ȳ = {Y_mean}')
plt.axvline(x=X_mean, color='green', linestyle='--', alpha=0.7, label=f'X̄ = {X_mean}')

# Add trend line
z = np.polyfit(X, Y, 1)
p = np.poly1d(z)
plt.plot(X, p(X), "r--", alpha=0.8, linewidth=2, label=f'Trend line')

plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'Scatter Plot with Correlation $r = ' + f'{r:.4f}$')
plt.legend()
plt.grid(True, alpha=0.3)

# Deviations visualization
plt.subplot(2, 3, 2)
colors = ['red' if dev > 0 else 'blue' for dev in X_dev]
plt.bar(range(len(X)), X_dev, color=colors, alpha=0.7)
plt.xlabel('Data Point Index')
plt.ylabel('$X - \\bar{X}$')
plt.title('Deviations of $X$ from Mean')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 3)
colors = ['red' if dev > 0 else 'blue' for dev in Y_dev]
plt.bar(range(len(Y)), Y_dev, color=colors, alpha=0.7)
plt.xlabel('Data Point Index')
plt.ylabel('$Y - \\bar{Y}$')
plt.title('Deviations of $Y$ from Mean')
plt.grid(True, alpha=0.3)

# Products visualization
plt.subplot(2, 3, 4)
colors = ['green' if prod > 0 else 'orange' for prod in XY_products]
bars = plt.bar(range(len(XY_products)), XY_products, color=colors, alpha=0.7)
plt.xlabel('Data Point Index')
plt.ylabel('$(X - \\bar{X})(Y - \\bar{Y})$')
plt.title('Products of Deviations')
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, XY_products):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.1),
             f'{value:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)

# Calculation step summary
plt.subplot(2, 3, 5)
plt.axis('off')
calc_text = (r"Calculation Summary:\n\n"
             r"$n = " + f"{n}" + r"$\n"
             r"$\bar{X} = " + f"{X_mean:.1f}" + r"$, $\bar{Y} = " + f"{Y_mean:.1f}" + r"$\n\n"
             r"$\sum(X - \bar{X})(Y - \bar{Y}) = " + f"{sum_XY:.1f}" + r"$\n"
             r"$\sum(X - \bar{X})^2 = " + f"{sum_X_squared:.1f}" + r"$\n"
             r"$\sum(Y - \bar{Y})^2 = " + f"{sum_Y_squared:.1f}" + r"$\n\n"
             r"$r = " + f"{sum_XY:.1f}" + r" / \sqrt{" + f"{sum_X_squared:.1f}" + r" \times " + f"{sum_Y_squared:.1f}" + r"}$\n"
             r"$r = " + f"{sum_XY:.1f}" + r" / " + f"{denominator:.4f}" + r"$\n"
             r"$r = " + f"{r:.4f}" + r"$")

plt.text(0.1, 0.9, calc_text, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

# Decision summary
plt.subplot(2, 3, 6)
plt.axis('off')
decision_text = (r"Feature Selection Decision:\n\n"
                r"Correlation: $r = " + f"{r:.4f}" + r"$\n"
                r"Absolute correlation: $|r| = " + f"{abs_r:.4f}" + r"$\n"
                r"Threshold: " + f"{threshold}" + r"\n\n"
                r"Using absolute value:\n"
                r"$" + ('\\checkmark' if abs_r >= threshold else '\\times') + r"$ " + ('SELECTED' if abs_r >= threshold else 'REJECTED') + r"\n"
                r"$(" + f"{abs_r:.4f}" + r" " + ('\\geq' if abs_r >= threshold else '<') + r" " + f"{threshold}" + r")$\n\n"
                r"Using signed value:\n"
                r"$" + ('\\checkmark' if r >= threshold else '\\times') + r"$ " + ('SELECTED' if r >= threshold else 'REJECTED') + r"\n"
                r"$(" + f"{r:.4f}" + r" " + ('\\geq' if r >= threshold else '<') + r" " + f"{threshold}" + r")$")

color = "lightgreen" if abs_r >= threshold else "lightcoral"
plt.text(0.1, 0.9, decision_text, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'pearson_correlation_calculation.png'), dpi=300, bbox_inches='tight')
plt.close()  # Close figure to free memory

# ========================================================================
# SUMMARY AND RECOMMENDATIONS
# ========================================================================

print("\n\n" + "="*80)
print("SUMMARY AND RECOMMENDATIONS")
print("="*80)

print("\n1. For numerical data, use multiple measures:")
print("   - Pearson correlation for linear relationships")
print("   - Spearman correlation for monotonic relationships")  
print("   - Mutual information for any relationship type")

print("\n2. Relationships hard to detect with simple correlation:")
print("   - Non-linear (quadratic, exponential, sine)")
print("   - Non-monotonic relationships")
print("   - Categorical relationships")
print("   - Relationships with outliers")

print("\n3. Handling non-linear relationships:")
print("   - Create polynomial features")
print("   - Use rank-based methods (Spearman)")
print("   - Apply mutual information")
print("   - Use binning/discretization")

print("\n4. High MI + Low correlation suggests:")
print("   - Non-linear but informative relationship")
print("   - Feature contains predictive information")
print("   - Consider feature engineering")

print("\n5. Measure comparison:")
print("   - Pearson: Best for linear relationships")
print("   - Spearman: Good for monotonic relationships")
print("   - Mutual Information: Captures any dependency")
print("   - Chi-square/Cramér's V: Good for categorical data")

print(f"\nAll plots saved to: {save_dir}")
print("="*80)
