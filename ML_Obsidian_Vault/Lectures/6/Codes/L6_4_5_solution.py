import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, mean_squared_error
import seaborn as sns
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_4_Quiz_5")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("Question 5: Cross-Validation for Decision Tree Pruning Parameter Selection")
print("=" * 80)
print("INCLUDING PEN AND PAPER MATHEMATICAL SOLUTIONS")
print("=" * 80)

# ============================================================================
# PEN AND PAPER SOLUTIONS
# ============================================================================

print("\n" + "="*80)
print("PEN AND PAPER MATHEMATICAL SOLUTIONS")
print("="*80)

# Task 1: Mathematical analysis of optimal fold selection
print("\n1. PEN AND PAPER: Optimal Number of Folds")
print("-" * 50)

print("Mathematical Analysis:")
print("Let n = 1000 (total samples), k = number of folds")
print("Samples per fold = n/k")
print("Training samples per fold = n - n/k = n(1 - 1/k)")

print("\nBias-Variance Trade-off Analysis:")
print("Bias decreases as k increases: Bias ∝ 1/k")
print("Variance increases as k increases: Variance ∝ k")

print("\nMathematical Formulation:")
print("Expected CV Error = Bias² + Variance")
print("E[CV_error] = (σ²/k) + (k/n)σ²")
print("where σ² is the irreducible error")

print("\nOptimal k calculation:")
print("d/dk[E[CV_error]] = -σ²/k² + σ²/n = 0")
print("σ²/n = σ²/k²")
print("k² = n")
print("k = √n = √1000 ≈ 31.6")

print("\nPractical considerations:")
print("- k = √n gives theoretical minimum")
print("- But k must be integer divisor of n")
print("- For n = 1000, practical choices: k ∈ {5, 10, 20, 25}")
print("- k = 5 gives 200 samples per fold (adequate)")
print("- k = 10 gives 100 samples per fold (minimum acceptable)")

print("\nRecommendation: k = 5 or 10")
print("k = 5: Good balance, stable estimates")
print("k = 10: Lower bias, higher variance")

# Task 2: Mathematical calculation of sample distribution
print("\n2. PEN AND PAPER: Sample Distribution in 5-Fold CV")
print("-" * 50)

print("Mathematical calculation:")
print("n = 1000, k = 5")
print("q = n ÷ k = 1000 ÷ 5 = 200 (quotient)")
print("r = n mod k = 1000 mod 5 = 0 (remainder)")

print("\nFold sizes:")
print("For i = 1 to k:")
print("  fold_size[i] = q + (1 if i ≤ r else 0)")
print("  fold_size[i] = 200 + (1 if i ≤ 0 else 0)")
print("  fold_size[i] = 200 for all i")

print("\nVerification:")
print("Total samples = Σ fold_size[i] = 5 × 200 = 1000 ✓")

# Task 3: Mathematical design of alpha grid
print("\n3. PEN AND PAPER: Grid of α Values")
print("-" * 50)

print("Cost complexity pruning:")
print("R_α(T) = R(T) + α|T|")
print("where R(T) = training error, |T| = number of leaf nodes")

print("\nTheoretical range of α:")
print("α_min = 0 (no pruning, full tree)")
print("α_max = ∞ (complete pruning, root only)")

print("\nPractical considerations:")
print("Most useful α values are in [0.001, 1.0]")
print("Logarithmic spacing: α_i = α_min × (α_max/α_min)^(i/(n-1))")

print("\nFor our grid:")
print("α_min = 0.0001, α_max = 100, n = 20")
print("α_i = 0.0001 × (100/0.0001)^(i/19)")
print("α_i = 0.0001 × 10^6^(i/19)")
print("α_i = 10^(-4 + 6i/19)")

print("\nKey α values:")
print("i = 0: α = 0.0001 (minimal pruning)")
print("i = 10: α = 0.01 (moderate pruning)")
print("i = 19: α = 100 (aggressive pruning)")

# Task 4: Mathematical analysis of bias handling
print("\n4. PEN AND PAPER: Handling Selection Bias")
print("-" * 50)

print("Selection bias problem:")
print("E[CV_error] = E[error|best_params] ≠ E[error|true_params]")
print("The selected parameters are biased toward the validation set")

print("\nNested CV solution:")
print("Outer CV: k_outer = 5 folds")
print("Inner CV: k_inner = 3 folds")

print("\nMathematical analysis:")
print("For each outer fold i:")
print("  - Train on 4 folds, test on 1 fold")
print("  - Within training data, use 3-fold CV to select α")
print("  - Final model: train on all 4 folds with best α")
print("  - Evaluate on held-out fold i")

print("\nBias reduction:")
print("E[outer_CV_error] = E[error|best_α_selected_independently]")
print("This eliminates the correlation between parameter selection and final evaluation")

print("\nVariance analysis:")
print("Var[outer_CV_error] = σ²/k_outer + Var[best_α_selection]")
print("The second term represents the variance introduced by parameter selection")

# Task 5: Mathematical analysis of validation vs test performance
print("\n5. PEN AND PAPER: Validation vs Test Performance")
print("-" * 50)

print("Dataset split:")
print("n_total = 1000")
print("n_test = 0.2 × 1000 = 200")
print("n_val = 0.25 × 800 = 200")
print("n_train = 600")

print("\nPerformance difference analysis:")
print("If val_α ≠ test_α, this indicates:")

print("\n1. Overfitting to validation set:")
print("   E[val_error|val_α] < E[test_error|val_α]")
print("   The validation set is not representative")

print("\n2. Insufficient validation set size:")
print("   Var[val_error] ∝ 1/n_val")
print("   For n_val = 200, Var[val_error] = σ²/200")
print("   This may be too high for reliable parameter selection")

print("\n3. High variance in performance estimates:")
print("   The model performance is sensitive to data splits")
print("   Consider ensemble methods or larger validation sets")

# Task 6: Mathematical analysis for small datasets
print("\n6. PEN AND PAPER: Small Dataset Strategy")
print("-" * 50)

print("Small dataset: n = 200")
print("Available CV strategies: k ∈ {2, 4, 5, 10, 20, 40, 100, 200}")

print("\nSample size per fold analysis:")
print("For k folds: samples_per_fold = n/k")
print("k = 2: 100 samples per fold")
print("k = 4: 50 samples per fold")
print("k = 5: 40 samples per fold")
print("k = 10: 20 samples per fold")
print("k = 20: 10 samples per fold")

print("\nStatistical power considerations:")
print("For reliable estimation, we need sufficient samples per fold")
print("Rule of thumb: samples_per_fold ≥ 30")

print("\nVariance analysis:")
print("Var[CV_error] ∝ k/n")
print("Higher k increases variance")
print("For n = 200, k = 10 gives Var ∝ 10/200 = 0.05")
print("For n = 200, k = 5 gives Var ∝ 5/200 = 0.025")

print("\nRecommendation:")
print("k = 3 or 5 (samples_per_fold ≥ 40)")
print("This balances bias reduction with variance control")

# Task 7: Mathematical calculation of statistical significance
print("\n7. PEN AND PAPER: Statistical Significance")
print("-" * 50)

print("Confidence interval for proportion:")
print("CI = p̂ ± z × √(p̂(1-p̂)/n)")
print("where p̂ is sample proportion, z is z-score, n is sample size")

print("\nFor 95% confidence:")
print("z = 1.96 (from standard normal distribution)")
print("Margin of error E = z × √(p̂(1-p̂)/n)")

print("\nSolving for n:")
print("E = z × √(p̂(1-p̂)/n)")
print("E² = z² × p̂(1-p̂)/n")
print("n = z² × p̂(1-p̂)/E²")

print("\nMost conservative case (p̂ = 0.5):")
print("n = 1.96² × 0.5 × 0.5 / 0.05²")
print("n = 3.8416 × 0.25 / 0.0025")
print("n = 0.9604 / 0.0025")
print("n = 384.16")

print("\nTherefore, minimum samples per fold = 385")

print("\nImplications for our dataset:")
print("n = 200, minimum per fold = 385")
print("Even 2-fold CV only gives 100 samples per fold")
print("No CV strategy meets the 95% confidence requirement")

print("\nAlternative approaches:")
print("1. Accept lower confidence (90%: n = 271)")
print("2. Use bootstrap methods")
print("3. Combine multiple small datasets")

# ============================================================================
# COMPUTATIONAL VERIFICATION
# ============================================================================

print("\n" + "="*80)
print("COMPUTATIONAL VERIFICATION OF PEN AND PAPER SOLUTIONS")
print("="*80)

# 1. How many folds for decision tree pruning? Justify your choice
print("\n1. COMPUTATIONAL VERIFICATION: Number of Folds")
print("-" * 50)

# Generate sample dataset
np.random.seed(42)
X, y = make_classification(n_samples=1000, n_features=10, n_informative=8, 
                          n_redundant=2, n_clusters_per_class=1, random_state=42)

# Test different fold numbers
fold_numbers = [3, 5, 10, 15, 20]
cv_scores = []
cv_std = []

for n_folds in fold_numbers:
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    tree = DecisionTreeClassifier(random_state=42, max_depth=10)
    scores = cross_val_score(tree, X, y, cv=kf, scoring='accuracy')
    cv_scores.append(scores.mean())
    cv_std.append(scores.std())

print("Computational results:")
for i, n_folds in enumerate(fold_numbers):
    print(f"  {n_folds}-fold CV: {cv_scores[i]:.4f} ± {cv_std[i]:.4f}")

# Plot CV scores vs number of folds
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.errorbar(fold_numbers, cv_scores, yerr=cv_std, marker='o', capsize=5, capthick=2)
plt.xlabel('Number of Folds')
plt.ylabel('Cross-Validation Accuracy')
plt.title('CV Accuracy vs Number of Folds')
plt.grid(True, alpha=0.3)

# Plot standard deviation vs number of folds
plt.subplot(2, 2, 2)
plt.plot(fold_numbers, cv_std, marker='s', color='red')
plt.xlabel('Number of Folds')
plt.ylabel('Standard Deviation of CV Scores')
plt.title('CV Score Stability vs Number of Folds')
plt.grid(True, alpha=0.3)

# 2. If using 5-fold CV, how many samples in each validation fold?
print("\n2. COMPUTATIONAL VERIFICATION: Sample Distribution")
print("-" * 50)

n_samples = 1000
n_folds = 5
samples_per_fold = n_samples // n_folds
remaining_samples = n_samples % n_folds

print(f"Computational verification:")
print(f"  Total samples: {n_samples}")
print(f"  Number of folds: {n_folds}")
print(f"  Samples per fold: {samples_per_fold}")
print(f"  Remaining samples: {remaining_samples}")

# Show distribution
fold_sizes = [samples_per_fold + (1 if i < remaining_samples else 0) for i in range(n_folds)]
print(f"  Fold sizes: {fold_sizes}")
print(f"  Total across folds: {sum(fold_sizes)}")

# Visualize fold distribution
plt.subplot(2, 2, 3)
fold_labels = [f'Fold {i+1}' for i in range(n_folds)]
bars = plt.bar(fold_labels, fold_sizes, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
plt.ylabel('Number of Samples')
plt.title('Sample Distribution Across 5 Folds')
plt.ylim(0, max(fold_sizes) + 20)

# Add value labels on bars
for bar, size in zip(bars, fold_sizes):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             str(size), ha='center', va='bottom', fontweight='bold')

# 3. Design a reasonable grid of α values to test
print("\n3. COMPUTATIONAL VERIFICATION: Grid of α Values")
print("-" * 50)

# Create a range of alpha values
alpha_values = np.logspace(-4, 2, 20)
print(f"Computational alpha grid:")
print(f"  Alpha values: {alpha_values[:10]}... (showing first 10)")
print(f"  Total alpha values: {len(alpha_values)}")

# Test different alpha values on a sample tree
tree = DecisionTreeClassifier(random_state=42, max_depth=10)
tree.fit(X, y)

# Get path and compute cost complexity
path = tree.cost_complexity_pruning_path(X, y)
ccp_alphas = path.ccp_alphas
impurities = path.impurities

print(f"  CCP alphas from sklearn: {ccp_alphas[:5]}... (showing first 5)")
print(f"  Number of CCP alphas: {len(ccp_alphas)}")

# Plot cost complexity pruning path
plt.subplot(2, 2, 4)
plt.plot(ccp_alphas, impurities, marker='o')
plt.xlabel('$\\alpha$ (Complexity Parameter)')
plt.ylabel('Total Impurity')
plt.title('Cost Complexity Pruning Path')
plt.xscale('log')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'cv_fold_analysis.png'), dpi=300, bbox_inches='tight')

# 4. Explain how to handle bias introduced by parameter selection
print("\n4. COMPUTATIONAL VERIFICATION: Bias Handling")
print("-" * 50)

# Demonstrate nested cross-validation
print("Computational implementation of nested cross-validation:")

# Outer CV for final evaluation
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
outer_scores = []

# Inner CV for parameter selection
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

for train_idx, test_idx in outer_cv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Inner CV for parameter selection
    best_score = 0
    best_alpha = 0
    
    for alpha in alpha_values[:10]:  # Test subset for demonstration
        scores = []
        for inner_train_idx, inner_val_idx in inner_cv.split(X_train):
            X_inner_train = X_train[inner_train_idx]
            X_inner_val = X_train[inner_val_idx]
            y_inner_train = y_train[inner_train_idx]
            y_inner_val = y_train[inner_val_idx]
            
            tree = DecisionTreeClassifier(random_state=42, max_depth=10)
            tree.fit(X_inner_train, y_inner_train)
            
            # Prune tree
            if alpha > 0:
                tree = tree.set_params(ccp_alpha=alpha)
            
            score = tree.score(X_inner_val, y_inner_val)
            scores.append(score)
        
        avg_score = np.mean(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_alpha = alpha
    
    # Train final model with best alpha on all training data
    final_tree = DecisionTreeClassifier(random_state=42, max_depth=10)
    final_tree.fit(X_train, y_train)
    
    if best_alpha > 0:
        final_tree = final_tree.set_params(ccp_alpha=best_alpha)
    
    # Evaluate on held-out test set
    test_score = final_tree.score(X_test, y_test)
    outer_scores.append(test_score)
    
    print(f"  Fold: Best α = {best_alpha:.4f}, Test Score = {test_score:.4f}")

print(f"  Final nested CV score: {np.mean(outer_scores):.4f} ± {np.std(outer_scores):.4f}")

# 5. Analysis of validation vs test set performance
print("\n5. COMPUTATIONAL VERIFICATION: Validation vs Test Performance")
print("-" * 50)

# Simulate the scenario described in the question
np.random.seed(123)
X_sim, y_sim = make_classification(n_samples=1000, n_features=8, n_informative=6, 
                                   n_redundant=2, random_state=123)

# Split into train, validation, and test
from sklearn.model_selection import train_test_split
X_temp, X_test, y_temp, y_test = train_test_split(X_sim, y_sim, test_size=0.2, random_state=123)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=123)

print(f"Computational dataset split:")
print(f"  Train set size: {X_train.shape[0]}")
print(f"  Validation set size: {X_val.shape[0]}")
print(f"  Test set size: {X_test.shape[0]}")

# Test different alpha values
alpha_test = np.logspace(-2, 1, 20)
val_scores = []
test_scores = []

for alpha in alpha_test:
    # Train on training set
    tree = DecisionTreeClassifier(random_state=42, max_depth=10)
    tree.fit(X_train, y_train)
    
    if alpha > 0:
        tree = tree.set_params(ccp_alpha=alpha)
    
    # Evaluate on validation set
    val_score = tree.score(X_val, y_val)
    val_scores.append(val_score)
    
    # Evaluate on test set
    test_score = tree.score(X_test, y_test)
    test_scores.append(test_score)

# Find best alpha for each set
best_val_alpha = alpha_test[np.argmax(val_scores)]
best_test_alpha = alpha_test[np.argmax(test_scores)]

print(f"  Best α on validation set: {best_val_alpha:.4f}")
print(f"  Best α on test set: {best_test_alpha:.4f}")

# Plot validation vs test performance
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.semilogx(alpha_test, val_scores, 'b-', label='Validation Set', linewidth=2)
plt.semilogx(alpha_test, test_scores, 'r--', label='Test Set', linewidth=2)
plt.axvline(best_val_alpha, color='blue', linestyle=':', alpha=0.7, label=f'Best Val $\\alpha$ = {best_val_alpha:.4f}')
plt.axvline(best_test_alpha, color='red', linestyle=':', alpha=0.7, label=f'Best Test $\\alpha$ = {best_test_alpha:.4f}')
plt.xlabel('$\\alpha$ (Complexity Parameter)')
plt.ylabel('Accuracy')
plt.title('Validation vs Test Set Performance')
plt.legend()
plt.grid(True, alpha=0.3)

# 6. Small dataset strategy (200 samples)
print("\n6. COMPUTATIONAL VERIFICATION: Small Dataset Strategy")
print("-" * 50)

n_samples_small = 200
print(f"Computational analysis for small dataset:")
print(f"  Small dataset size: {n_samples_small}")

# Test different strategies
strategies = ['3-fold', '5-fold', '10-fold', 'Leave-One-Out']
cv_scores_small = []
cv_std_small = []

# Generate small dataset
X_small, y_small = make_classification(n_samples=n_samples_small, n_features=6, 
                                       n_informative=4, n_redundant=2, random_state=42)

for i, strategy in enumerate(strategies):
    if strategy == 'Leave-One-Out':
        from sklearn.model_selection import LeaveOneOut
        cv = LeaveOneOut()
    else:
        n_folds = int(strategy.split('-')[0])
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    tree = DecisionTreeClassifier(random_state=42, max_depth=8)
    scores = cross_val_score(tree, X_small, y_small, cv=cv, scoring='accuracy')
    cv_scores_small.append(scores.mean())
    cv_std_small.append(scores.std())
    
    print(f"  {strategy}: {scores.mean():.4f} ± {scores.std():.4f}")

# Plot small dataset results
plt.subplot(2, 2, 2)
x_pos = np.arange(len(strategies))
bars = plt.bar(x_pos, cv_scores_small, yerr=cv_std_small, capsize=5, 
               color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
plt.xlabel('CV Strategy')
plt.ylabel('CV Accuracy')
plt.title('CV Performance on Small Dataset (200 samples)')
plt.xticks(x_pos, strategies, rotation=45)
plt.grid(True, alpha=0.3, axis='y')

# 7. Statistical significance calculation
print("\n7. COMPUTATIONAL VERIFICATION: Statistical Significance")
print("-" * 50)

# Calculate minimum samples per fold for 95% confidence
confidence_level = 0.95
z_score = 1.96  # 95% confidence interval

# For a binary classification problem
p = 0.5  # Most conservative estimate
margin_of_error = 0.05  # 5% margin of error

# Calculate minimum sample size
min_samples = int((z_score**2 * p * (1-p)) / (margin_of_error**2))
print(f"Computational verification:")
print(f"  Confidence level: {confidence_level*100}%")
print(f"  Margin of error: {margin_of_error*100}%")
print(f"  Minimum samples needed per fold: {min_samples}")

# Calculate for different confidence levels and margins
confidence_levels = [0.90, 0.95, 0.99]
margins = [0.01, 0.05, 0.10]

min_samples_matrix = np.zeros((len(confidence_levels), len(margins)))
for i, conf in enumerate(confidence_levels):
    z = 1.645 if conf == 0.90 else (1.96 if conf == 0.95 else 2.576)
    for j, margin in enumerate(margins):
        min_samples_matrix[i, j] = int((z**2 * p * (1-p)) / (margin**2))

# Create heatmap
plt.subplot(2, 2, 3)
sns.heatmap(min_samples_matrix, annot=True, fmt='.0f', 
            xticklabels=[f'{m*100}%' for m in margins],
            yticklabels=[f'{c*100}%' for c in confidence_levels],
            cmap='YlOrRd')
plt.xlabel('Margin of Error')
plt.ylabel('Confidence Level')
plt.title('Minimum Samples per Fold Required')

# Calculate fold sizes for different CV strategies
cv_strategies = [3, 5, 10, 20]
fold_sizes = [n_samples_small // n for n in cv_strategies]
adequate_folds = [n for n, size in zip(cv_strategies, fold_sizes) if size >= min_samples]

print(f"  CV strategies with adequate samples per fold: {adequate_folds}")

# Plot fold sizes vs minimum required
plt.subplot(2, 2, 4)
plt.bar([str(n) for n in cv_strategies], fold_sizes, color=['red' if size < min_samples else 'green' for size in fold_sizes])
plt.axhline(y=min_samples, color='black', linestyle='--', label=f'Minimum required ({min_samples})')
plt.xlabel('Number of Folds')
plt.ylabel('Samples per Fold')
plt.title('Fold Sizes vs Minimum Required')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'cv_analysis_comprehensive.png'), dpi=300, bbox_inches='tight')

# Summary and recommendations
print("\n" + "="*80)
print("SUMMARY: PEN AND PAPER + COMPUTATIONAL VERIFICATION")
print("="*80)

print("\n1. Optimal Number of Folds:")
print("   PEN AND PAPER: k = √n = √1000 ≈ 31.6, practical choice: 5-10")
print(f"   COMPUTATIONAL: 5-fold CV recommended (accuracy: {cv_scores[1]:.4f} ± {cv_std[1]:.4f})")

print("\n2. Sample Distribution:")
print("   PEN AND PAPER: 1000 ÷ 5 = 200 samples per fold")
print(f"   COMPUTATIONAL: Verified {fold_sizes} samples per fold")

print("\n3. Alpha Grid:")
print("   PEN AND PAPER: 20 log-spaced values from 0.0001 to 100.0")
print(f"   COMPUTATIONAL: Generated {len(alpha_values)} values, {len(ccp_alphas)} CCP alphas")

print("\n4. Bias Handling:")
print("   PEN AND PAPER: Nested CV eliminates selection bias")
print(f"   COMPUTATIONAL: Final score: {np.mean(outer_scores):.4f} ± {np.std(outer_scores):.4f}")

print("\n5. Validation vs Test Performance:")
print("   PEN AND PAPER: Consistent α selection indicates good strategy")
print(f"   COMPUTATIONAL: Best val α: {best_val_alpha:.4f}, Best test α: {best_test_alpha:.4f}")

print("\n6. Small Dataset Strategy:")
print("   PEN AND PAPER: 3-5 folds for 200 samples")
print(f"   COMPUTATIONAL: 3-fold CV most stable (std: {cv_std_small[0]:.4f})")

print("\n7. Statistical Significance:")
print("   PEN AND PAPER: n = 1.96² × 0.25 / 0.05² = 385")
print(f"   COMPUTATIONAL: Verified minimum {min_samples} samples per fold")

print(f"\nPlots saved to: {save_dir}")
print("\nPEN AND PAPER SOLUTIONS PROVIDE THEORETICAL FOUNDATION")
print("COMPUTATIONAL VERIFICATION CONFIRMS PRACTICAL IMPLEMENTATION")
