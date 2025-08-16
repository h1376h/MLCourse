import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_4_Quiz_8")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Set matplotlib to non-interactive backend to avoid displaying plots
plt.ioff()

print("Question 8: Learning Curves Analysis")
print("=" * 80)
print("COMPREHENSIVE ANALYSIS OF LEARNING CURVES FOR DECISION TREES")
print("=" * 80)

# ============================================================================
# PEN AND PAPER MATHEMATICAL SOLUTIONS WITH SYMBOLIC CALCULATIONS
# ============================================================================

print("\n" + "="*80)
print("PEN AND PAPER MATHEMATICAL SOLUTIONS WITH SYMBOLIC CALCULATIONS")
print("="*80)

# Mathematical Foundation: Learning Curves Theory
print("\n1. MATHEMATICAL FOUNDATION: Learning Curves Theory")
print("-" * 70)

print("Learning Curve Definition:")
print("   A learning curve plots the model performance metric (e.g., accuracy, error)")
print("   against the amount of training data used.")
print("   Mathematically: $f(n) = \\text{Performance}(\\text{Model trained on } n \\text{ samples})$")

print("\nBias-Variance Decomposition:")
print("   For a given model complexity, the expected generalization error can be decomposed as:")
print("   $E[\\text{Error}] = \\text{Bias}^2 + \\text{Variance} + \\text{Irreducible Error}$")

print("\nOverfitting Detection:")
print("   Let $\\text{Train}(n)$ and $\\text{Val}(n)$ be training and validation scores")
print("   for $n$ training samples. Overfitting occurs when:")
print("   $\\text{Gap}(n) = \\text{Train}(n) - \\text{Val}(n)$ becomes large")
print("   and $\\frac{d\\text{Val}(n)}{dn} < 0$ while $\\frac{d\\text{Train}(n)}{dn} > 0$")

print("\nOptimal Complexity Selection:")
print("   For hyperparameter $\\theta$ (e.g., max_depth), find:")
print("   $\\theta^* = \\arg\\max_{\\theta} \\text{Val}(\\theta)$")
print("   subject to: $\\text{Gap}(\\theta) \\leq \\epsilon$ (tolerance threshold)")

print("\nCost-Complexity Pruning Effect:")
print("   Pruning reduces model complexity from $\\theta_1$ to $\\theta_2 < \\theta_1$")
print("   Expected changes:")
print("   - $\\text{Train}(\\theta_2) \\leq \\text{Train}(\\theta_1)$ (training performance may decrease)")
print("   - $\\text{Val}(\\theta_2) \\geq \\text{Val}(\\theta_1)$ (validation performance should improve)")
print("   - $\\text{Gap}(\\theta_2) \\leq \\text{Gap}(\\theta_1)$ (generalization gap should decrease)")

# ============================================================================
# TASK 1: LARGE GAP BETWEEN TRAINING AND VALIDATION CURVES
# ============================================================================

print("\n" + "="*80)
print("TASK 1: LARGE GAP BETWEEN TRAINING AND VALIDATION CURVES")
print("=" * 80)

# Step-by-step mathematical solution
print("\nPEN AND PAPER SOLUTION:")
print("-" * 50)

print("Step 1: Define the gap function")
print("   Let $\\text{Gap}(n) = \\text{Train}(n) - \\text{Val}(n)$")
print("   where $n$ is the number of training samples")

print("\nStep 2: Identify overfitting mathematically")
print("   Overfitting occurs when:")
print("   - $\\text{Gap}(n) > \\epsilon$ (large gap threshold)")
print("   - $\\frac{d\\text{Val}(n)}{dn} < 0$ (validation performance decreases)")
print("   - $\\frac{d\\text{Train}(n)}{dn} > 0$ (training performance increases)")

print("\nStep 3: Calculate maximum gap")
print("   $\\text{MaxGap} = \\max_{n} \\{\\text{Gap}(n)\\}$")
print("   This identifies the worst case of overfitting")

print("\nStep 4: Interpret the gap")
print("   - Small gap ($< 0.05$): Good generalization")
print("   - Medium gap ($0.05-0.10$): Some overfitting")
print("   - Large gap ($> 0.10$): Severe overfitting")
print("   - Very large gap ($> 0.20$): Critical overfitting")

# Generate synthetic data for demonstration
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                          n_redundant=5, n_clusters_per_class=1, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a complex decision tree (likely to overfit)
complex_tree = DecisionTreeClassifier(max_depth=20, min_samples_split=2, random_state=42)

# Generate learning curves
train_sizes, train_scores, val_scores = learning_curve(
    complex_tree, X_train, y_train, 
    train_sizes=np.linspace(0.1, 1.0, 20),
    cv=5, scoring='accuracy', random_state=42
)

# Calculate means and standard deviations
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot learning curves showing large gap
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score', linewidth=2)
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.plot(train_sizes, val_mean, 'o-', color='red', label='Cross-Validation Score', linewidth=2)
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')

plt.xlabel('Training Examples')
plt.ylabel('Accuracy')
plt.title('Learning Curves: Large Gap Indicates Overfitting')
plt.legend()
plt.grid(True, alpha=0.3)

# Highlight the gap
max_gap_idx = np.argmax(train_mean - val_mean)
max_gap = train_mean[max_gap_idx] - val_mean[max_gap_idx]
plt.annotate(f'Gap = {max_gap:.3f}', 
             xy=(train_sizes[max_gap_idx], (train_mean[max_gap_idx] + val_mean[max_gap_idx])/2),
             xytext=(train_sizes[max_gap_idx] + 50, (train_mean[max_gap_idx] + val_mean[max_gap_idx])/2 + 0.1),
             arrowprops=dict(arrowstyle='->', lw=2, color='green'),
             fontsize=12, ha='center',
             bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", alpha=0.8))

print(f"Analysis of Large Gap:")
print(f"   - Maximum gap between training and validation: {max_gap:.3f}")
print(f"   - This indicates severe overfitting")
print(f"   - Training accuracy: {train_mean[-1]:.3f}")
print(f"   - Validation accuracy: {val_mean[-1]:.3f}")
print(f"   - Gap at full training set: {train_mean[-1] - val_mean[-1]:.3f}")

# ============================================================================
# TASK 2: IDENTIFYING OVERFITTING POINT
# ============================================================================

print("\n" + "="*80)
print("TASK 2: IDENTIFYING OVERFITTING POINT")
print("=" * 80)

# Step-by-step mathematical solution
print("\nPEN AND PAPER SOLUTION:")
print("-" * 50)

print("Step 1: Define overfitting mathematically")
print("   Overfitting begins at point $n^*$ where:")
print("   $\\frac{d\\text{Val}(n^*)}{dn} < 0$ and $\\frac{d\\text{Train}(n^*)}{dn} > 0$")

print("\nStep 2: Discrete approximation")
print("   For discrete data points, we approximate derivatives:")
print("   $\\frac{d\\text{Val}(n)}{dn} \\approx \\frac{\\text{Val}(n_i) - \\text{Val}(n_{i-1})}{n_i - n_{i-1}}$")
print("   $\\frac{d\\text{Train}(n)}{dn} \\approx \\frac{\\text{Train}(n_i) - \\text{Train}(n_{i-1})}{n_i - n_{i-1}}$")

print("\nStep 3: Overfitting detection algorithm")
print("   Find the smallest $i$ such that:")
print("   $\\text{Val}(n_i) < \\text{Val}(n_{i-1})$ AND $\\text{Train}(n_i) > \\text{Train}(n_{i-1})$")
print("   Then $n^* = n_i$ is the overfitting start point")

print("\nStep 4: Alternative criteria")
print("   We can also use:")
print("   - $\\text{Gap}(n_i) > \\text{Gap}(n_{i-1})$ (gap increases)")
print("   - $\\text{Val}(n_i) < \\text{Val}(n_{i-1})$ (validation decreases)")
print("   - $\\text{Train}(n_i) > \\text{Train}(n_{i-1})$ (training increases)")

print("\nStep 5: Practical considerations")
print("   - Use moving averages to smooth fluctuations")
print("   - Consider minimum change thresholds")
print("   - Validate with multiple random seeds")

# Find the point where overfitting begins
# Overfitting begins when validation score starts decreasing while training continues to increase
overfitting_start = None
for i in range(1, len(val_mean)):
    if val_mean[i] < val_mean[i-1] and train_mean[i] > train_mean[i-1]:
        overfitting_start = i
        break

if overfitting_start is None:
    overfitting_start = len(val_mean) - 1

plt.subplot(2, 2, 2)
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score', linewidth=2)
plt.plot(train_sizes, val_mean, 'o-', color='red', label='Cross-Validation Score', linewidth=2)

# Highlight overfitting point
plt.axvline(x=train_sizes[overfitting_start], color='green', linestyle='--', 
            label=f'Overfitting starts at {train_sizes[overfitting_start]:.0f} samples')
plt.scatter(train_sizes[overfitting_start], val_mean[overfitting_start], 
            color='green', s=100, zorder=5)

plt.xlabel('Training Examples')
plt.ylabel('Accuracy')
plt.title('Identifying Overfitting Point')
plt.legend()
plt.grid(True, alpha=0.3)

print(f"Overfitting Analysis:")
print(f"   - Overfitting begins at approximately {train_sizes[overfitting_start]:.0f} training samples")
print(f"   - At this point: Training accuracy = {train_mean[overfitting_start]:.3f}")
print(f"   - At this point: Validation accuracy = {val_mean[overfitting_start]:.3f}")
print(f"   - Training samples at overfitting start: {train_sizes[overfitting_start]:.0f}")

# ============================================================================
# TASK 3: EFFECTS OF COST-COMPLEXITY PRUNING
# ============================================================================

print("\n" + "="*80)
print("TASK 3: EFFECTS OF COST-COMPLEXITY PRUNING")
print("=" * 80)

# Step-by-step mathematical solution
print("\nPEN AND PAPER SOLUTION:")
print("-" * 50)

print("Step 1: Cost-complexity pruning formulation")
print("   The cost-complexity criterion is:")
print("   $C_\\alpha(T) = C(T) + \\alpha|T|$")
print("   where:")
print("   - $C(T)$ is the misclassification cost")
print("   - $|T|$ is the number of leaf nodes")
print("   - $\\alpha$ is the complexity parameter")

print("\nStep 2: Pruning effect on model complexity")
print("   Let $T_1$ be the original tree and $T_2$ be the pruned tree")
print("   Then: $|T_2| < |T_1|$ (fewer leaves)")
print("   And: $\\text{depth}(T_2) \\leq \\text{depth}(T_1)$ (shallower tree)")

print("\nStep 3: Expected changes in learning curves")
print("   For pruned tree $T_2$ vs original $T_1$:")
print("   - Training performance: $\\text{Train}_{T_2}(n) \\leq \\text{Train}_{T_1}(n)$")
print("   - Validation performance: $\\text{Val}_{T_2}(n) \\geq \\text{Val}_{T_1}(n)$")
print("   - Generalization gap: $\\text{Gap}_{T_2}(n) \\leq \\text{Gap}_{T_1}(n)$")

print("\nStep 4: Mathematical reasoning")
print("   Pruning reduces model capacity:")
print("   - Lower $\\text{Variance}$ (less overfitting)")
print("   - Higher $\\text{Bias}$ (more underfitting)")
print("   - Net effect depends on the bias-variance trade-off")

print("\nStep 5: Optimal pruning level")
print("   Find $\\alpha^*$ that minimizes:")
print("   $\\alpha^* = \\arg\\min_{\\alpha} \\{\\text{Val}(T_\\alpha)\\}$")
print("   subject to: $\\text{Gap}(T_\\alpha) \\leq \\epsilon$")

# Create different complexity trees
trees = {
    'Complex (max_depth=20)': DecisionTreeClassifier(max_depth=20, random_state=42),
    'Moderate (max_depth=10)': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Simple (max_depth=5)': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Very Simple (max_depth=3)': DecisionTreeClassifier(max_depth=3, random_state=42)
}

# Generate learning curves for different complexities
plt.subplot(2, 2, 3)
colors = ['blue', 'red', 'green', 'orange']
linestyles = ['-', '--', '-.', ':']

for i, (name, tree) in enumerate(trees.items()):
    train_sizes_comp, train_scores_comp, val_scores_comp = learning_curve(
        tree, X_train, y_train, 
        train_sizes=np.linspace(0.1, 1.0, 15),
        cv=5, scoring='accuracy', random_state=42
    )
    
    train_mean_comp = np.mean(train_scores_comp, axis=1)
    val_mean_comp = np.mean(val_scores_comp, axis=1)
    
    plt.plot(train_sizes_comp, train_mean_comp, color=colors[i], linestyle=linestyles[i], 
             label=f'{name} (Train)', linewidth=2)
    plt.plot(train_sizes_comp, val_mean_comp, color=colors[i], linestyle=linestyles[i], 
             label=f'{name} (Val)', linewidth=2, alpha=0.7)

plt.xlabel('Training Examples')
plt.ylabel('Accuracy')
plt.title('Learning Curves: Effect of Tree Complexity')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

print(f"Cost-Complexity Pruning Effects:")
print(f"   - Complex trees (depth=20): High training accuracy, low validation accuracy")
print(f"   - Moderate trees (depth=10): Balanced performance")
print(f"   - Simple trees (depth=5): Lower training accuracy, better generalization")
print(f"   - Very simple trees (depth=3): Underfitting, both scores are low")

# ============================================================================
# TASK 4: OPTIMAL TREE COMPLEXITY SELECTION
# ============================================================================

print("\n" + "="*80)
print("TASK 4: OPTIMAL TREE COMPLEXITY SELECTION")
print("=" * 80)

# Step-by-step mathematical solution
print("\nPEN AND PAPER SOLUTION:")
print("-" * 50)

print("Step 1: Formulate the optimization problem")
print("   For hyperparameter $\\theta$ (e.g., max_depth), solve:")
print("   $\\theta^* = \\arg\\max_{\\theta} \\{\\text{Val}(\\theta)\\}$")
print("   subject to constraints:")

print("\nStep 2: Define constraints")
print("   - Performance constraint: $\\text{Val}(\\theta) \\geq \\text{Val}_{\\min}$")
print("   - Gap constraint: $\\text{Gap}(\\theta) \\leq \\epsilon$")
print("   - Complexity constraint: $\\theta \\leq \\theta_{\\max}$")

print("\nStep 3: Multi-objective optimization")
print("   Alternative formulation using weighted sum:")
print("   $\\theta^* = \\arg\\max_{\\theta} \\{\\lambda \\cdot \\text{Val}(\\theta) - (1-\\lambda) \\cdot \\text{Gap}(\\theta)\\}$")
print("   where $\\lambda \\in [0,1]$ controls the trade-off")

print("\nStep 4: Validation curve analysis")
print("   Plot $\\text{Val}(\\theta)$ vs $\\theta$ and find:")
print("   - Global maximum: $\\theta_{\\text{max}} = \\arg\\max_{\\theta} \\{\\text{Val}(\\theta)\\}$")
print("   - Knee point: where $\\frac{d^2\\text{Val}(\\theta)}{d\\theta^2}$ changes sign")
print("   - Stability region: where $\\text{Val}(\\theta)$ is within $\\delta$ of maximum")

print("\nStep 5: Practical selection criteria")
print("   Choose $\\theta^*$ that:")
print("   - Maximizes validation performance")
print("   - Maintains reasonable generalization gap")
print("   - Provides stable performance across cross-validation folds")
print("   - Balances computational complexity and performance")

# Use validation curves to find optimal complexity
max_depths = range(1, 21)
train_scores, val_scores = validation_curve(
    DecisionTreeClassifier(random_state=42), X_train, y_train,
    param_name='max_depth', param_range=max_depths,
    cv=5, scoring='accuracy'
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.subplot(2, 2, 4)
plt.plot(max_depths, train_mean, 'o-', color='blue', label='Training Score', linewidth=2)
plt.fill_between(max_depths, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.plot(max_depths, val_mean, 'o-', color='red', label='Cross-Validation Score', linewidth=2)
plt.fill_between(max_depths, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')

# Find optimal depth
optimal_depth = max_depths[np.argmax(val_mean)]
plt.axvline(x=optimal_depth, color='green', linestyle='--', 
            label=f'Optimal depth = {optimal_depth}')
plt.scatter(optimal_depth, val_mean[np.argmax(val_mean)], 
            color='green', s=100, zorder=5)

plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Validation Curves: Optimal Tree Complexity')
plt.legend()
plt.grid(True, alpha=0.3)

print(f"Optimal Tree Complexity Selection:")
print(f"   - Optimal max_depth: {optimal_depth}")
print(f"   - Best validation accuracy: {np.max(val_mean):.3f}")
print(f"   - Training accuracy at optimal: {train_mean[np.argmax(val_mean)]:.3f}")
print(f"   - Gap at optimal: {train_mean[np.argmax(val_mean)] - np.max(val_mean):.3f}")

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'learning_curves_analysis.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# TASK 5: DECREASING ACCURACY PATTERNS
# ============================================================================

print("\n" + "="*80)
print("TASK 5: DECREASING ACCURACY PATTERNS")
print("=" * 80)

# Step-by-step mathematical solution
print("\nPEN AND PAPER SOLUTION:")
print("-" * 50)

print("Step 1: Define decreasing accuracy patterns")
print("   Both accuracies decrease when:")
print("   $\\frac{d\\text{Train}(n)}{dn} < 0$ AND $\\frac{d\\text{Val}(n)}{dn} < 0$")
print("   This is unusual and indicates underlying problems")

print("\nStep 2: Mathematical analysis of causes")
print("   Data quality issues:")
print("   - Noise level $\\eta$ increases: $y_{\\text{true}} = f(x) + \\eta$")
print("   - Data corruption: $P(\\text{corruption}) > \\text{threshold}$")
print("   - Concept drift: $f_t(x) \\neq f_{t+1}(x)$")

print("\nStep 3: Model complexity issues")
print("   Excessive complexity can cause:")
print("   - Numerical instability: $\\text{condition number} > 10^{10}$")
print("   - Gradient explosion: $\\|\\nabla L\\| > \\text{threshold}$")
print("   - Memory overflow: $\\text{memory usage} > \\text{available}$")

print("\nStep 4: Mathematical detection")
print("   Detect decreasing patterns by finding:")
print("   $n^* = \\min\\{n : \\text{Train}(n) < \\text{Train}(n-1) \\land \\text{Val}(n) < \\text{Val}(n-1)\\}$")
print("   If $n^*$ exists, investigate data and model issues")

print("\nStep 5: Diagnostic metrics")
print("   Calculate:")
print("   - Data quality score: $Q = 1 - \\frac{\\text{noise variance}}{\\text{signal variance}}$")
print("   - Model stability: $S = \\frac{1}{N}\\sum_{i=1}^{N} |\\text{Val}(n_i) - \\text{Val}(n_{i-1})|$")
print("   - Performance degradation: $D = \\frac{\\text{Val}(n_1) - \\text{Val}(n_N)}{\\text{Val}(n_1)}$")

# Create a scenario where both accuracies decrease
# This could happen with noisy data or when the model becomes too complex

# Generate noisy data
X_noisy, y_noisy = make_classification(n_samples=1000, n_features=20, n_informative=5, 
                                      n_redundant=10, n_clusters_per_class=1, 
                                      random_state=42, flip_y=0.3)  # 30% noise

# Add more noise to simulate data degradation
np.random.seed(42)
noise_mask = np.random.random(len(y_noisy)) < 0.4
y_noisy[noise_mask] = 1 - y_noisy[noise_mask]

X_train_noisy, X_test_noisy, y_train_noisy, y_test_noisy = train_test_split(
    X_noisy, y_noisy, test_size=0.3, random_state=42
)

# Create very complex tree
complex_tree_noisy = DecisionTreeClassifier(max_depth=25, min_samples_split=2, random_state=42)

# Generate learning curves for noisy data
train_sizes_noisy, train_scores_noisy, val_scores_noisy = learning_curve(
    complex_tree_noisy, X_train_noisy, y_train_noisy, 
    train_sizes=np.linspace(0.1, 1.0, 25),
    cv=5, scoring='accuracy', random_state=42
)

train_mean_noisy = np.mean(train_scores_noisy, axis=1)
val_mean_noisy = np.mean(val_scores_noisy, axis=1)

# Plot decreasing accuracy patterns
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.plot(train_sizes_noisy, train_mean_noisy, 'o-', color='blue', 
         label='Training Score', linewidth=2)
plt.plot(train_sizes_noisy, val_mean_noisy, 'o-', color='red', 
         label='Cross-Validation Score', linewidth=2)

# Find where both start decreasing
both_decreasing_start = None
for i in range(1, len(val_mean_noisy)):
    if val_mean_noisy[i] < val_mean_noisy[i-1] and train_mean_noisy[i] < train_mean_noisy[i-1]:
        both_decreasing_start = i
        break

if both_decreasing_start:
    plt.axvline(x=train_sizes_noisy[both_decreasing_start], color='green', linestyle='--', 
                label=f'Both decreasing at {train_sizes_noisy[both_decreasing_start]:.0f} samples')
    plt.scatter(train_sizes_noisy[both_decreasing_start], 
                (train_mean_noisy[both_decreasing_start] + val_mean_noisy[both_decreasing_start])/2, 
                color='green', s=100, zorder=5)

plt.xlabel('Training Examples')
plt.ylabel('Accuracy')
plt.title('Decreasing Accuracy Patterns')
plt.legend()
plt.grid(True, alpha=0.3)

print(f"Decreasing Accuracy Analysis:")
if both_decreasing_start:
    print(f"   - Both accuracies start decreasing at {train_sizes_noisy[both_decreasing_start]:.0f} samples")
    print(f"   - Indicates data quality issues or excessive complexity")
else:
    print(f"   - Both accuracies don't decrease together in this scenario")
    print(f"   - This is typical for well-behaved data")

# ============================================================================
# TASK 6: SEASONAL PATTERNS EFFECT
# ============================================================================

print("\n" + "="*80)
print("TASK 6: SEASONAL PATTERNS EFFECT")
print("=" * 80)

# Step-by-step mathematical solution
print("\nPEN AND PAPER SOLUTION:")
print("-" * 50)

print("Step 1: Model seasonal patterns mathematically")
print("   Seasonal data can be modeled as:")
print("   $y(t) = f(x(t)) + s(t) + \\epsilon(t)$")
print("   where:")
print("   - $f(x(t))$ is the underlying function")
print("   - $s(t)$ is the seasonal component: $s(t) = A\\sin(\\omega t + \\phi)$")
print("   - $\\epsilon(t)$ is random noise")

print("\nStep 2: Seasonal effect on learning curves")
print("   For seasonal data, validation performance varies:")
print("   $\\text{Val}(n) = \\text{Val}_{\\text{base}}(n) + \\Delta\\text{Val}_{\\text{seasonal}}(n)$")
print("   where $\\Delta\\text{Val}_{\\text{seasonal}}(n)$ oscillates with period $T$")

print("\nStep 3: Cross-validation considerations")
print("   Random splits can cause issues:")
print("   - Training set: $\\{t_1, t_2, ..., t_k\\}$")
print("   - Validation set: $\\{t_{k+1}, t_{k+2}, ..., t_n\\}$")
print("   - If seasonal patterns exist, random splits may not capture all seasons")

print("\nStep 4: Time-based validation strategy")
print("   Use time-based splits:")
print("   - Training: $\\{t_1, t_2, ..., t_{n-k}\\}$")
print("   - Validation: $\\{t_{n-k+1}, t_{n-k+2}, ..., t_n\\}$")
print("   - Ensures temporal consistency")

print("\nStep 5: Seasonal pattern detection")
print("   Detect seasonality using:")
print("   - Autocorrelation: $R(\\tau) = \\frac{1}{N}\\sum_{t=1}^{N-\\tau} y(t)y(t+\\tau)$")
print("   - Fourier analysis: $Y(f) = \\int_{-\\infty}^{\\infty} y(t)e^{-i2\\pi ft}dt$")
print("   - Seasonal decomposition: $y(t) = \\text{trend}(t) + \\text{seasonal}(t) + \\text{residual}(t)$")

# Simulate seasonal data
np.random.seed(42)
n_samples = 1000
time_steps = np.linspace(0, 4*np.pi, n_samples)  # 4 seasons

# Create seasonal features
seasonal_feature = np.sin(time_steps) + 0.3*np.random.randn(n_samples)
trend_feature = np.linspace(0, 1, n_samples) + 0.1*np.random.randn(n_samples)
noise_feature = 0.2*np.random.randn(n_samples)

# Combine features
X_seasonal = np.column_stack([seasonal_feature, trend_feature, noise_feature])

# Create seasonal target (more complex pattern)
y_seasonal = (np.sin(time_steps) > 0).astype(int)
y_seasonal = np.logical_xor(y_seasonal, (trend_feature > 0.5)).astype(int)

# Add some noise to target
np.random.seed(42)
noise_mask = np.random.random(len(y_seasonal)) < 0.1
y_seasonal[noise_mask] = 1 - y_seasonal[noise_mask]

# Split seasonal data
X_train_seasonal, X_test_seasonal, y_train_seasonal, y_test_seasonal = train_test_split(
    X_seasonal, y_seasonal, test_size=0.3, random_state=42
)

# Create tree for seasonal data
seasonal_tree = DecisionTreeClassifier(max_depth=15, random_state=42)

# Generate learning curves for seasonal data
train_sizes_seasonal, train_scores_seasonal, val_scores_seasonal = learning_curve(
    seasonal_tree, X_train_seasonal, y_train_seasonal, 
    train_sizes=np.linspace(0.1, 1.0, 20),
    cv=5, scoring='accuracy', random_state=42
)

train_mean_seasonal = np.mean(train_scores_seasonal, axis=1)
val_mean_seasonal = np.mean(val_scores_seasonal, axis=1)

plt.subplot(2, 3, 2)
plt.plot(train_sizes_seasonal, train_mean_seasonal, 'o-', color='blue', 
         label='Training Score', linewidth=2)
plt.plot(train_sizes_seasonal, val_mean_seasonal, 'o-', color='red', 
         label='Cross-Validation Score', linewidth=2)

plt.xlabel('Training Examples')
plt.ylabel('Accuracy')
plt.title('Learning Curves: Seasonal Data')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot seasonal pattern
plt.subplot(2, 3, 3)
plt.plot(time_steps[:200], seasonal_feature[:200], 'b-', label='Seasonal Feature', linewidth=2)
plt.plot(time_steps[:200], y_seasonal[:200], 'r.', label='Target', alpha=0.7, markersize=3)
plt.xlabel('Time Steps')
plt.ylabel('Feature Value / Target')
plt.title('Seasonal Pattern in Data')
plt.legend()
plt.grid(True, alpha=0.3)

print(f"Seasonal Patterns Effect:")
print(f"   - Seasonal data shows periodic patterns in learning curves")
print(f"   - Validation accuracy may fluctuate due to seasonal variations")
print(f"   - Need to ensure training and validation sets capture all seasons")
print(f"   - Time-based cross-validation may be more appropriate")

# ============================================================================
# TASK 7: SKETCH LEARNING CURVES SHOWING UNDERFITTING AND OVERFITTING
# ============================================================================

print("\n" + "="*80)
print("TASK 7: LEARNING CURVES SHOWING UNDERFITTING AND OVERFITTING")
print("=" * 80)

# Step-by-step mathematical solution
print("\nPEN AND PAPER SOLUTION:")
print("-" * 50)

print("Step 1: Mathematical characterization of learning curve patterns")
print("   Let $\\text{Train}(n)$ and $\\text{Val}(n)$ be the learning curves")
print("   where $n$ is the number of training samples")

print("\nStep 2: Underfitting pattern (High Bias)")
print("   Underfitting occurs when:")
print("   - $\\text{Train}(n) \\approx \\text{Val}(n)$ (small gap)")
print("   - $\\text{Train}(n) \\leq \\text{Performance}_{\\text{target}}$ (low performance)")
print("   - $\\frac{d\\text{Train}(n)}{dn} \\approx 0$ (plateau)")
print("   - $\\frac{d\\text{Val}(n)}{dn} \\approx 0$ (plateau)")

print("\nStep 3: Good fit pattern (Balanced)")
print("   Good fit occurs when:")
print("   - $\\text{Train}(n) \\approx \\text{Val}(n)$ (small gap)")
print("   - $\\text{Train}(n) \\geq \\text{Performance}_{\\text{target}}$ (high performance)")
print("   - $\\frac{d\\text{Train}(n)}{dn} > 0$ (improving)")
print("   - $\\frac{d\\text{Val}(n)}{dn} > 0$ (improving)")
print("   - $\\lim_{n \\to \\infty} \\text{Train}(n) = \\lim_{n \\to \\infty} \\text{Val}(n)$ (convergence)")

print("\nStep 4: Overfitting pattern (High Variance)")
print("   Overfitting occurs when:")
print("   - $\\text{Train}(n) > \\text{Val}(n)$ (large gap)")
print("   - $\\text{Train}(n)$ increases: $\\frac{d\\text{Train}(n)}{dn} > 0$")
print("   - $\\text{Val}(n)$ decreases after point $n^*$: $\\frac{d\\text{Val}(n)}{dn} < 0$ for $n > n^*$")
print("   - Gap increases: $\\frac{d}{dn}(\\text{Train}(n) - \\text{Val}(n)) > 0$")

print("\nStep 5: Mathematical detection criteria")
print("   For each pattern, calculate:")
print("   - Gap function: $G(n) = \\text{Train}(n) - \\text{Val}(n)$")
print("   - Gap derivative: $G'(n) = \\frac{dG(n)}{dn}$")
print("   - Performance derivative: $P'(n) = \\frac{d\\text{Val}(n)}{dn}$")
print("   - Convergence measure: $C(n) = |\\text{Train}(n) - \\text{Val}(n)|$")

print("\nStep 6: Pattern classification algorithm")
print("   Classify learning curves as:")
print("   - Underfitting: $G(n) < \\epsilon$ AND $\\text{Train}(n) < \\text{threshold}$")
print("   - Good fit: $G(n) < \\epsilon$ AND $\\text{Train}(n) \\geq \\text{threshold}$")
print("   - Overfitting: $G(n) > \\epsilon$ AND $G'(n) > 0$")
print("   where $\\epsilon$ and $\\text{threshold}$ are predefined values")

# Create different complexity scenarios
scenarios = {
    'Underfitting': DecisionTreeClassifier(max_depth=2, random_state=42),
    'Good Fit': DecisionTreeClassifier(max_depth=8, random_state=42),
    'Overfitting': DecisionTreeClassifier(max_depth=25, random_state=42)
}

plt.subplot(2, 3, 4)
colors_scenario = ['orange', 'green', 'red']
linestyles_scenario = ['-', '--', '-.']

for i, (name, tree) in enumerate(scenarios.items()):
    train_sizes_scenario, train_scores_scenario, val_scores_scenario = learning_curve(
        tree, X_train, y_train, 
        train_sizes=np.linspace(0.1, 1.0, 20),
        cv=5, scoring='accuracy', random_state=42
    )
    
    train_mean_scenario = np.mean(train_scores_scenario, axis=1)
    val_mean_scenario = np.mean(val_scores_scenario, axis=1)
    
    plt.plot(train_sizes_scenario, train_mean_scenario, color=colors_scenario[i], 
             linestyle=linestyles_scenario[i], label=f'{name} (Train)', linewidth=2)
    plt.plot(train_sizes_scenario, val_mean_scenario, color=colors_scenario[i], 
             linestyle=linestyles_scenario[i], label=f'{name} (Val)', linewidth=2, alpha=0.7)

plt.xlabel('Training Examples')
plt.ylabel('Accuracy')
plt.title('Learning Curves: Underfitting vs Overfitting')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# Create idealized learning curves
plt.subplot(2, 3, 5)
# Generate idealized curves
x_ideal = np.linspace(0.1, 1.0, 100)

# Underfitting: both curves are low and close
underfit_train = 0.6 + 0.1 * x_ideal + 0.05 * np.random.randn(100)
underfit_val = 0.58 + 0.08 * x_ideal + 0.05 * np.random.randn(100)

# Good fit: both curves increase and converge
goodfit_train = 0.5 + 0.4 * x_ideal + 0.02 * np.random.randn(100)
goodfit_val = 0.48 + 0.35 * x_ideal + 0.02 * np.random.randn(100)

# Overfitting: training increases, validation decreases after a point
overfit_train = 0.5 + 0.45 * x_ideal + 0.02 * np.random.randn(100)
overfit_val = 0.48 + 0.4 * x_ideal[:50] + 0.02 * np.random.randn(50)
overfit_val = np.concatenate([overfit_val, 0.88 - 0.3 * (x_ideal[50:] - x_ideal[49]) + 0.02 * np.random.randn(50)])

plt.plot(x_ideal, underfit_train, 'o-', color='orange', label='Underfitting (Train)', linewidth=2)
plt.plot(x_ideal, underfit_val, 'o-', color='orange', label='Underfitting (Val)', linewidth=2, alpha=0.7)
plt.plot(x_ideal, goodfit_train, 's-', color='green', label='Good Fit (Train)', linewidth=2)
plt.plot(x_ideal, goodfit_val, 's-', color='green', label='Good Fit (Val)', linewidth=2, alpha=0.7)
plt.plot(x_ideal, overfit_train, '^-', color='red', label='Overfitting (Train)', linewidth=2)
plt.plot(x_ideal, overfit_val, '^-', color='red', label='Overfitting (Val)', linewidth=2, alpha=0.7)

plt.xlabel('Training Examples (Normalized)')
plt.ylabel('Accuracy')
plt.title('Idealized Learning Curves')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# Create summary visualization
plt.subplot(2, 3, 6)
# Create a conceptual diagram
x_concept = np.linspace(0, 1, 100)
y_underfit = 0.6 + 0.1 * x_concept
y_goodfit = 0.5 + 0.4 * x_concept
y_overfit = 0.5 + 0.45 * x_concept
y_overfit_val = np.where(x_concept < 0.6, 0.5 + 0.35 * x_concept, 0.8 - 0.2 * (x_concept - 0.6))

plt.plot(x_concept, y_underfit, 'o-', color='orange', label='Underfitting', linewidth=3, markersize=8)
plt.plot(x_concept, y_goodfit, 's-', color='green', label='Good Fit', linewidth=3, markersize=8)
plt.plot(x_concept, y_overfit, '^-', color='red', label='Overfitting (Train)', linewidth=3, markersize=8)
plt.plot(x_concept, y_overfit_val, 'v-', color='red', label='Overfitting (Val)', linewidth=3, markersize=8, alpha=0.7)

plt.xlabel('Model Complexity')
plt.ylabel('Performance')
plt.title('Bias-Variance Trade-off')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'learning_curves_comprehensive.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# SUMMARY AND KEY INSIGHTS
# ============================================================================

print("\n" + "="*80)
print("SUMMARY AND KEY INSIGHTS")
print("=" * 80)

print(f"\nKey Findings:")
print(f"1. Large Gap Analysis:")
print(f"   - Gap of {max_gap:.3f} indicates severe overfitting")
print(f"   - Training accuracy: {train_mean[-1]:.3f}, Validation: {val_mean[-1]:.3f}")

print(f"\n2. Overfitting Point:")
print(f"   - Overfitting begins at {train_sizes[overfitting_start]:.0f} training samples")
print(f"   - This is when validation accuracy starts decreasing")

print(f"\n3. Cost-Complexity Pruning:")
print(f"   - Reduces overfitting by limiting tree depth")
print(f"   - Optimal depth found: {optimal_depth}")
print(f"   - Best validation accuracy: {np.max(val_mean):.3f}")

print(f"\n4. Optimal Complexity Selection:")
print(f"   - Use validation curves to find optimal parameters")
print(f"   - Balance between bias and variance")
print(f"   - Consider the gap between training and validation scores")

print(f"\n5. Decreasing Accuracy Patterns:")
if both_decreasing_start:
    print(f"   - Both accuracies decrease at {train_sizes_noisy[both_decreasing_start]:.0f} samples")
    print(f"   - Indicates data quality issues or excessive complexity")

print(f"\n6. Seasonal Patterns:")
print(f"   - Can cause fluctuations in learning curves")
print(f"   - Need proper cross-validation strategy")
print(f"   - Time-based splits may be necessary")

print(f"\n7. Underfitting vs Overfitting:")
print(f"   - Underfitting: Both curves are low and close")
print(f"   - Good fit: Both curves increase and converge")
print(f"   - Overfitting: Training increases, validation decreases")

print(f"\nPlots saved to: {save_dir}")
print(f"Files generated:")
print(f"  - learning_curves_analysis.png: Basic learning curves analysis")
print(f"  - learning_curves_comprehensive.png: Comprehensive analysis with all scenarios")
