import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split, validation_curve, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_2_Quiz_4")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("=" * 80)
print("QUESTION 4: REGULARIZATION PARAMETER C ANALYSIS IN SOFT MARGIN SVM")
print("=" * 80)

# ============================================================================
# PART 1: QUALITATIVE BEHAVIOR ANALYSIS FOR DIFFERENT C VALUES
# ============================================================================

print("\n1. QUALITATIVE BEHAVIOR ANALYSIS FOR C = 0.1, 1, 10, 100")
print("-" * 60)

# Generate synthetic data for demonstration
np.random.seed(42)
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                          n_informative=2, n_clusters_per_class=1, 
                          class_sep=1.5, random_state=42)

# Add some noise to make it non-linearly separable
X += np.random.normal(0, 0.3, X.shape)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

C_values = [0.1, 1, 10, 100]
svm_models = {}

print("Training SVM models with different C values...")
for C in C_values:
    svm = SVC(kernel='linear', C=C, random_state=42)
    svm.fit(X_train, y_train)
    svm_models[C] = svm
    
    # Calculate metrics
    train_score = svm.score(X_train, y_train)
    test_score = svm.score(X_test, y_test)
    n_support_vectors = len(svm.support_vectors_)
    
    print(f"C = {C:>4}: Train Acc = {train_score:.3f}, Test Acc = {test_score:.3f}, "
          f"Support Vectors = {n_support_vectors}")

# Create visualization of decision boundaries for different C values
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for i, C in enumerate(C_values):
    svm = svm_models[C]
    
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    # Get predictions
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and data points
    axes[i].contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    axes[i].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu', 
                   edgecolors='black', s=50, alpha=0.7)
    
    # Highlight support vectors
    support_vectors = svm.support_vectors_
    axes[i].scatter(support_vectors[:, 0], support_vectors[:, 1], 
                   s=200, facecolors='none', edgecolors='red', linewidth=2, 
                   label=f'Support Vectors ({len(support_vectors)})')
    
    # Plot decision boundary
    w = svm.coef_[0]
    b = svm.intercept_[0]
    slope = -w[0] / w[1]
    intercept = -b / w[1]
    x_boundary = np.array([x_min, x_max])
    y_boundary = slope * x_boundary + intercept
    axes[i].plot(x_boundary, y_boundary, 'k-', linewidth=2, label='Decision Boundary')
    
    # Plot margin boundaries
    margin = 1 / np.sqrt(np.sum(w**2))
    y_margin_upper = y_boundary + margin / np.sqrt(1 + slope**2)
    y_margin_lower = y_boundary - margin / np.sqrt(1 + slope**2)
    axes[i].plot(x_boundary, y_margin_upper, 'k--', alpha=0.5, label='Margin')
    axes[i].plot(x_boundary, y_margin_lower, 'k--', alpha=0.5)
    
    axes[i].set_title(f'C = {C}\nTrain Acc: {svm.score(X_train, y_train):.3f}, '
                     f'Test Acc: {svm.score(X_test, y_test):.3f}\n'
                     f'Support Vectors: {len(support_vectors)}')
    axes[i].set_xlabel('$x_1$')
    axes[i].set_ylabel('$x_2$')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'svm_c_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PART 2: BIAS-VARIANCE TRADEOFF ANALYSIS
# ============================================================================

print("\n2. BIAS-VARIANCE TRADEOFF ANALYSIS")
print("-" * 60)

# Generate multiple datasets to analyze bias-variance tradeoff
n_datasets = 20
C_values_extended = np.logspace(-2, 3, 20)
bias_scores = []
variance_scores = []
total_scores = []

print("Analyzing bias-variance tradeoff across multiple datasets...")

for C in C_values_extended:
    predictions = []
    
    for _ in range(n_datasets):
        # Generate new dataset
        X_temp, y_temp = make_classification(n_samples=100, n_features=2, 
                                           n_redundant=0, n_informative=2, 
                                           n_clusters_per_class=1, class_sep=1.5, 
                                           random_state=np.random.randint(1000))
        X_temp += np.random.normal(0, 0.3, X_temp.shape)
        
        # Train SVM
        svm = SVC(kernel='linear', C=C, random_state=42)
        svm.fit(X_temp, y_temp)
        
        # Split data and make predictions on test set
        X_temp_train, X_temp_test, y_temp_train, y_temp_test = train_test_split(
            X_temp, y_temp, test_size=0.3, random_state=42)
        svm.fit(X_temp_train, y_temp_train)
        pred = svm.predict(X_temp_test)
        predictions.append(pred)
    
    # Calculate bias and variance
    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    
    # Bias: difference between mean prediction and true labels
    bias = np.mean((mean_pred - y_temp_test)**2)
    
    # Variance: variance of predictions across datasets
    variance = np.mean(np.var(predictions, axis=0))
    
    # Total error = bias + variance
    total_error = bias + variance
    
    bias_scores.append(bias)
    variance_scores.append(variance)
    total_scores.append(total_error)

# Plot bias-variance tradeoff
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.semilogx(C_values_extended, bias_scores, 'b-', linewidth=2, label='Bias')
plt.semilogx(C_values_extended, variance_scores, 'r-', linewidth=2, label='Variance')
plt.semilogx(C_values_extended, total_scores, 'g-', linewidth=2, label='Total Error')
plt.xlabel('Regularization Parameter C')
plt.ylabel('Error')
plt.title('Bias-Variance Tradeoff vs C')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.semilogx(C_values_extended, bias_scores, 'b-', linewidth=2)
plt.xlabel('Regularization Parameter C')
plt.ylabel('Bias')
plt.title('Bias vs C')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.semilogx(C_values_extended, variance_scores, 'r-', linewidth=2)
plt.xlabel('Regularization Parameter C')
plt.ylabel('Variance')
plt.title('Variance vs C')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.semilogx(C_values_extended, total_scores, 'g-', linewidth=2)
plt.xlabel('Regularization Parameter C')
plt.ylabel('Total Error')
plt.title('Total Error vs C')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'bias_variance_tradeoff.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PART 3: SUPPORT VECTOR ANALYSIS
# ============================================================================

print("\n3. SUPPORT VECTOR ANALYSIS")
print("-" * 60)

# Analyze number of support vectors vs C
C_range = np.logspace(-3, 3, 50)
n_support_vectors_list = []
margin_sizes = []

print("Analyzing number of support vectors vs C...")

for C in C_range:
    svm = SVC(kernel='linear', C=C, random_state=42)
    svm.fit(X_train, y_train)
    
    n_support_vectors_list.append(len(svm.support_vectors_))
    
    # Calculate margin size
    w = svm.coef_[0]
    margin_size = 2 / np.sqrt(np.sum(w**2))
    margin_sizes.append(margin_size)

# Plot support vectors vs C
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.semilogx(C_range, n_support_vectors_list, 'b-', linewidth=2)
ax1.set_xlabel('Regularization Parameter C')
ax1.set_ylabel('Number of Support Vectors')
ax1.set_title('Number of Support Vectors vs C')
ax1.grid(True, alpha=0.3)

# Add annotations for specific C values
for C in [0.1, 1, 10, 100]:
    idx = np.argmin(np.abs(C_range - C))
    ax1.annotate(f'C={C}', xy=(C_range[idx], n_support_vectors_list[idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

ax2.semilogx(C_range, margin_sizes, 'r-', linewidth=2)
ax2.set_xlabel('Regularization Parameter C')
ax2.set_ylabel('Margin Size')
ax2.set_title('Margin Size vs C')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'support_vectors_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PART 4: VALIDATION CURVE EXPERIMENT
# ============================================================================

print("\n4. VALIDATION CURVE EXPERIMENT")
print("-" * 60)

# Generate a more complex dataset for validation curve
X_complex, y_complex = make_circles(n_samples=200, noise=0.2, factor=0.5, random_state=42)
X_complex_train, X_complex_test, y_complex_train, y_complex_test = train_test_split(
    X_complex, y_complex, test_size=0.3, random_state=42)

# Calculate validation curves
C_range_validation = np.logspace(-3, 3, 20)
train_scores, val_scores = validation_curve(
    SVC(kernel='rbf', gamma='scale', random_state=42),
    X_complex_train, y_complex_train,
    param_name='C',
    param_range=C_range_validation,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Calculate mean and std
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Find optimal C
optimal_C_idx = np.argmax(val_mean)
optimal_C = C_range_validation[optimal_C_idx]

print(f"Optimal C value: {optimal_C:.4f}")
print(f"Best validation accuracy: {val_mean[optimal_C_idx]:.4f}")

# Plot validation curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.semilogx(C_range_validation, train_mean, 'b-', label='Training Score', linewidth=2)
plt.fill_between(C_range_validation, train_mean - train_std, train_mean + train_std, 
                alpha=0.3, color='blue')
plt.semilogx(C_range_validation, val_mean, 'r-', label='Validation Score', linewidth=2)
plt.fill_between(C_range_validation, val_mean - val_std, val_mean + val_std, 
                alpha=0.3, color='red')
plt.axvline(x=optimal_C, color='g', linestyle='--', label=f'Optimal C = {optimal_C:.4f}')
plt.xlabel('Regularization Parameter C')
plt.ylabel('Accuracy')
plt.title('Validation Curves for SVM')
plt.legend()
plt.grid(True, alpha=0.3)

# Test the optimal model
optimal_svm = SVC(kernel='rbf', C=optimal_C, gamma='scale', random_state=42)
optimal_svm.fit(X_complex_train, y_complex_train)
test_accuracy = optimal_svm.score(X_complex_test, y_complex_test)

print(f"Test accuracy with optimal C: {test_accuracy:.4f}")

# Visualize the optimal model
plt.subplot(1, 2, 2)
x_min, x_max = X_complex[:, 0].min() - 0.5, X_complex[:, 0].max() + 0.5
y_min, y_max = X_complex[:, 1].min() - 0.5, X_complex[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                    np.linspace(y_min, y_max, 100))

Z = optimal_svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
plt.scatter(X_complex_train[:, 0], X_complex_train[:, 1], c=y_complex_train, 
           cmap='RdYlBu', edgecolors='black', s=50, alpha=0.7)
plt.scatter(optimal_svm.support_vectors_[:, 0], optimal_svm.support_vectors_[:, 1], 
           s=200, facecolors='none', edgecolors='red', linewidth=2, 
           label=f'Support Vectors ({len(optimal_svm.support_vectors_)})')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title(f'Optimal SVM (C = {optimal_C:.4f})\nTest Accuracy: {test_accuracy:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'validation_curves.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PART 5: MATHEMATICAL PROOF - SOFT MARGIN TO HARD MARGIN
# ============================================================================

print("\n5. MATHEMATICAL PROOF: SOFT MARGIN → HARD MARGIN AS C → ∞")
print("-" * 60)

# Demonstrate the convergence numerically
C_large_values = [1, 10, 100, 1000, 10000]
soft_margin_errors = []
hard_margin_errors = []

print("Demonstrating convergence of soft margin to hard margin...")

# Generate linearly separable data for hard margin comparison
X_separable, y_separable = make_classification(n_samples=100, n_features=2, 
                                             n_redundant=0, n_informative=2, 
                                             n_clusters_per_class=1, class_sep=2.0, 
                                             random_state=42)

X_sep_train, X_sep_test, y_sep_train, y_sep_test = train_test_split(
    X_separable, y_separable, test_size=0.3, random_state=42)

# Train hard margin SVM (very large C)
hard_margin_svm = SVC(kernel='linear', C=1e6, random_state=42)
hard_margin_svm.fit(X_sep_train, y_sep_train)
hard_margin_error = 1 - hard_margin_svm.score(X_sep_test, y_sep_test)

for C in C_large_values:
    soft_margin_svm = SVC(kernel='linear', C=C, random_state=42)
    soft_margin_svm.fit(X_sep_train, y_sep_train)
    soft_margin_error = 1 - soft_margin_svm.score(X_sep_test, y_sep_test)
    soft_margin_errors.append(soft_margin_error)
    hard_margin_errors.append(hard_margin_error)

# Plot convergence
plt.figure(figsize=(10, 6))
plt.semilogx(C_large_values, soft_margin_errors, 'b-o', linewidth=2, 
            label='Soft Margin Error', markersize=8)
plt.axhline(y=hard_margin_error, color='r', linestyle='--', linewidth=2, 
           label=f'Hard Margin Error = {hard_margin_error:.4f}')
plt.xlabel('Regularization Parameter C')
plt.ylabel('Test Error')
plt.title('Convergence of Soft Margin to Hard Margin SVM')
plt.legend()
plt.grid(True, alpha=0.3)

# Add convergence analysis
convergence_diff = np.abs(np.array(soft_margin_errors) - hard_margin_error)
plt.figure(figsize=(10, 6))
plt.semilogx(C_large_values, convergence_diff, 'g-o', linewidth=2, markersize=8)
plt.xlabel('Regularization Parameter C')
plt.ylabel('|Soft Margin Error - Hard Margin Error|')
plt.title('Convergence Analysis: Difference from Hard Margin')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'convergence_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PART 6: COMPREHENSIVE SUMMARY
# ============================================================================

print("\n6. COMPREHENSIVE SUMMARY")
print("-" * 60)

# Create summary table
summary_data = []
for C in [0.1, 1, 10, 100]:
    svm = svm_models[C]
    train_acc = svm.score(X_train, y_train)
    test_acc = svm.score(X_test, y_test)
    n_sv = len(svm.support_vectors_)
    
    # Calculate margin size
    w = svm.coef_[0]
    margin_size = 2 / np.sqrt(np.sum(w**2))
    
    summary_data.append([C, train_acc, test_acc, n_sv, margin_size])

print("Summary Table:")
print("C\t\tTrain Acc\tTest Acc\tSupport Vectors\tMargin Size")
print("-" * 70)
for row in summary_data:
    print(f"{row[0]:.1f}\t\t{row[1]:.3f}\t\t{row[2]:.3f}\t\t{row[3]}\t\t{row[4]:.3f}")

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Decision boundaries comparison
for i, C in enumerate([0.1, 1, 10, 100]):
    svm = svm_models[C]
    w = svm.coef_[0]
    b = svm.intercept_[0]
    
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axes[0, 0].contour(xx, yy, Z, levels=[0], colors=['C'+str(i)], 
                      linewidths=2, label=f'C={C}')
    axes[0, 0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                      cmap='RdYlBu', alpha=0.6, s=30)

axes[0, 0].set_title('Decision Boundaries Comparison')
axes[0, 0].set_xlabel('$x_1$')
axes[0, 0].set_ylabel('$x_2$')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Accuracy vs C
C_plot = [row[0] for row in summary_data]
train_acc_plot = [row[1] for row in summary_data]
test_acc_plot = [row[2] for row in summary_data]

axes[0, 1].semilogx(C_plot, train_acc_plot, 'b-o', label='Training Accuracy', linewidth=2)
axes[0, 1].semilogx(C_plot, test_acc_plot, 'r-o', label='Test Accuracy', linewidth=2)
axes[0, 1].set_xlabel('Regularization Parameter C')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Accuracy vs C')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Support vectors vs C
n_sv_plot = [row[3] for row in summary_data]
axes[0, 2].semilogx(C_plot, n_sv_plot, 'g-o', linewidth=2)
axes[0, 2].set_xlabel('Regularization Parameter C')
axes[0, 2].set_ylabel('Number of Support Vectors')
axes[0, 2].set_title('Support Vectors vs C')
axes[0, 2].grid(True, alpha=0.3)

# 4. Margin size vs C
margin_plot = [row[4] for row in summary_data]
axes[1, 0].semilogx(C_plot, margin_plot, 'm-o', linewidth=2)
axes[1, 0].set_xlabel('Regularization Parameter C')
axes[1, 0].set_ylabel('Margin Size')
axes[1, 0].set_title('Margin Size vs C')
axes[1, 0].grid(True, alpha=0.3)

# 5. Bias-variance tradeoff
axes[1, 1].semilogx(C_values_extended, bias_scores, 'b-', linewidth=2, label='Bias')
axes[1, 1].semilogx(C_values_extended, variance_scores, 'r-', linewidth=2, label='Variance')
axes[1, 1].set_xlabel('Regularization Parameter C')
axes[1, 1].set_ylabel('Error')
axes[1, 1].set_title('Bias-Variance Tradeoff')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 6. Validation curve
axes[1, 2].semilogx(C_range_validation, train_mean, 'b-', label='Training Score', linewidth=2)
axes[1, 2].semilogx(C_range_validation, val_mean, 'r-', label='Validation Score', linewidth=2)
axes[1, 2].axvline(x=optimal_C, color='g', linestyle='--', label=f'Optimal C = {optimal_C:.2f}')
axes[1, 2].set_xlabel('Regularization Parameter C')
axes[1, 2].set_ylabel('Accuracy')
axes[1, 2].set_title('Validation Curve')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'comprehensive_summary.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAll plots saved to: {save_dir}")
print("\nAnalysis complete! The code demonstrates:")
print("1. Qualitative behavior for different C values")
print("2. Bias-variance tradeoff analysis")
print("3. Support vector count analysis")
print("4. Validation curve experiment for optimal C")
print("5. Mathematical proof of convergence to hard margin")
print("6. Comprehensive summary with all key insights")
