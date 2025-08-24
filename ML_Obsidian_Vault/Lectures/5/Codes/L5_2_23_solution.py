import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
from sklearn.datasets import make_blobs, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_2_Quiz_23")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("=" * 80)
print("QUESTION 23: THE ROLE OF HYPERPARAMETER C IN SOFT-MARGIN SVM")
print("=" * 80)

# 1. Generate synthetic datasets for demonstration
print("\n1. GENERATING SYNTHETIC DATASETS")
print("-" * 40)

# Dataset 1: Linearly separable with some noise
X1, y1 = make_blobs(n_samples=100, centers=2, cluster_std=1.0, random_state=42)
# Add some noise points to make it non-linearly separable
noise_points = np.array([[2, 2], [3, 1], [1, 3], [4, 0], [0, 4]])
noise_labels = np.array([1, 1, -1, -1, -1])  # Some mislabeled points
X1 = np.vstack([X1, noise_points])
y1 = np.hstack([y1, noise_labels])

# Dataset 2: Non-linearly separable (circles)
X2, y2 = make_circles(n_samples=100, noise=0.2, factor=0.5, random_state=42)

print(f"Dataset 1 shape: {X1.shape}, Classes: {np.unique(y1)}")
print(f"Dataset 2 shape: {X2.shape}, Classes: {np.unique(y2)}")

# 2. Define different C values to test
C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
print(f"\nC values to test: {C_values}")

# 3. Function to plot decision boundaries and margins
def plot_svm_decision_boundary(X, y, C, title, ax):
    """Plot SVM decision boundary and margins for given C value"""
    
    # Train SVM
    svm = SVC(kernel='linear', C=C, random_state=42)
    svm.fit(X, y)
    
    # Get support vectors
    support_vectors = svm.support_vectors_
    
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    # Get predictions for mesh grid
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and regions
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax.contour(xx, yy, Z, colors='black', linewidths=2, alpha=0.8)
    
    # Plot data points
    colors = ['red' if label == -1 else 'blue' for label in y]
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=50, alpha=0.7, edgecolors='black')
    
    # Highlight support vectors
    if len(support_vectors) > 0:
        ax.scatter(support_vectors[:, 0], support_vectors[:, 1], 
                  s=200, facecolors='none', edgecolors='yellow', 
                  linewidth=2, label='Support Vectors')
    
    # Calculate margin width
    w = svm.coef_[0]
    b = svm.intercept_[0]
    margin_width = 2 / np.linalg.norm(w)
    
    ax.set_title(f'{title}\nC={C}, Margin Width={margin_width:.3f}')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.grid(True, alpha=0.3)
    
    return svm, margin_width, len(support_vectors)

# 4. Analyze the effect of C on linearly separable data
print("\n2. ANALYZING EFFECT OF C ON LINEARLY SEPARABLE DATA")
print("-" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

results_linear = []

for i, C in enumerate(C_values):
    print(f"\nC = {C}:")
    
    # Train and plot SVM
    svm, margin_width, n_support_vectors = plot_svm_decision_boundary(
        X1, y1, C, f'Linear Dataset', axes[i])
    
    # Calculate accuracy
    y_pred = svm.predict(X1)
    accuracy = accuracy_score(y1, y_pred)
    
    # Count misclassifications
    misclassifications = np.sum(y1 != y_pred)
    
    print(f"  Margin width: {margin_width:.3f}")
    print(f"  Number of support vectors: {n_support_vectors}")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Misclassifications: {misclassifications}")
    
    results_linear.append({
        'C': C,
        'margin_width': margin_width,
        'n_support_vectors': n_support_vectors,
        'accuracy': accuracy,
        'misclassifications': misclassifications
    })

# Remove the last subplot if not needed
axes[-1].remove()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'linear_dataset_c_effect.png'), dpi=300, bbox_inches='tight')
plt.show()

# 5. Analyze the effect of C on non-linearly separable data
print("\n3. ANALYZING EFFECT OF C ON NON-LINEARLY SEPARABLE DATA")
print("-" * 65)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

results_nonlinear = []

for i, C in enumerate(C_values):
    print(f"\nC = {C}:")
    
    # Train and plot SVM
    svm, margin_width, n_support_vectors = plot_svm_decision_boundary(
        X2, y2, C, f'Non-linear Dataset', axes[i])
    
    # Calculate accuracy
    y_pred = svm.predict(X2)
    accuracy = accuracy_score(y2, y_pred)
    
    # Count misclassifications
    misclassifications = np.sum(y2 != y_pred)
    
    print(f"  Margin width: {margin_width:.3f}")
    print(f"  Number of support vectors: {n_support_vectors}")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Misclassifications: {misclassifications}")
    
    results_nonlinear.append({
        'C': C,
        'margin_width': margin_width,
        'n_support_vectors': n_support_vectors,
        'accuracy': accuracy,
        'misclassifications': misclassifications
    })

# Remove the last subplot if not needed
axes[-1].remove()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'nonlinear_dataset_c_effect.png'), dpi=300, bbox_inches='tight')
plt.show()

# 6. Create summary plots
print("\n4. CREATING SUMMARY PLOTS")
print("-" * 30)

# Convert results to arrays for plotting
C_array = np.array([r['C'] for r in results_linear])
margin_widths_linear = np.array([r['margin_width'] for r in results_linear])
margin_widths_nonlinear = np.array([r['margin_width'] for r in results_nonlinear])
support_vectors_linear = np.array([r['n_support_vectors'] for r in results_linear])
support_vectors_nonlinear = np.array([r['n_support_vectors'] for r in results_nonlinear])
misclassifications_linear = np.array([r['misclassifications'] for r in results_linear])
misclassifications_nonlinear = np.array([r['misclassifications'] for r in results_nonlinear])

# Plot 1: Margin width vs C
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

ax1.semilogx(C_array, margin_widths_linear, 'bo-', linewidth=2, markersize=8, label='Linear Dataset')
ax1.semilogx(C_array, margin_widths_nonlinear, 'ro-', linewidth=2, markersize=8, label='Non-linear Dataset')
ax1.set_xlabel('C (Regularization Parameter)')
ax1.set_ylabel('Margin Width')
ax1.set_title('Effect of C on Margin Width')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Number of support vectors vs C
ax2.semilogx(C_array, support_vectors_linear, 'bo-', linewidth=2, markersize=8, label='Linear Dataset')
ax2.semilogx(C_array, support_vectors_nonlinear, 'ro-', linewidth=2, markersize=8, label='Non-linear Dataset')
ax2.set_xlabel('C (Regularization Parameter)')
ax2.set_ylabel('Number of Support Vectors')
ax2.set_title('Effect of C on Number of Support Vectors')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: Misclassifications vs C
ax3.semilogx(C_array, misclassifications_linear, 'bo-', linewidth=2, markersize=8, label='Linear Dataset')
ax3.semilogx(C_array, misclassifications_nonlinear, 'ro-', linewidth=2, markersize=8, label='Non-linear Dataset')
ax3.set_xlabel('C (Regularization Parameter)')
ax3.set_ylabel('Number of Misclassifications')
ax3.set_title('Effect of C on Misclassifications')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Plot 4: Bias-Variance Trade-off visualization
# For small C: High bias, low variance (underfitting)
# For large C: Low bias, high variance (overfitting)
bias_linear = 1 / (1 + C_array)  # Simplified bias model
variance_linear = C_array / (1 + C_array)  # Simplified variance model

ax4.semilogx(C_array, bias_linear, 'g-', linewidth=2, label='Bias (Underfitting)')
ax4.semilogx(C_array, variance_linear, 'm-', linewidth=2, label='Variance (Overfitting)')
ax4.set_xlabel('C (Regularization Parameter)')
ax4.set_ylabel('Bias/Variance')
ax4.set_title('Bias-Variance Trade-off')
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'c_effect_summary.png'), dpi=300, bbox_inches='tight')
plt.show()

# 7. Mathematical analysis
print("\n5. MATHEMATICAL ANALYSIS")
print("-" * 25)

print("\nObjective Function:")
print("min (1/2)||w||² + C∑ᵢ ξᵢ")
print("\nWhere:")
print("- (1/2)||w||² is the margin maximization term")
print("- C∑ᵢ ξᵢ is the classification error term")
print("- C is the regularization parameter that balances these two objectives")

print(f"\nMathematical Analysis for Different C Values:")

for i, C in enumerate(C_values):
    print(f"\nC = {C}:")
    
    # Calculate the relative importance of each term
    # For demonstration, we'll use the margin width as a proxy for ||w||
    margin_term = 1 / (margin_widths_linear[i] ** 2) if margin_widths_linear[i] > 0 else float('inf')
    error_term = C * misclassifications_linear[i]
    
    print(f"  Margin term (1/2)||w||² ≈ {margin_term:.3f}")
    print(f"  Error term C∑ᵢ ξᵢ = {C} × {misclassifications_linear[i]} = {error_term}")
    print(f"  Relative importance of error term: {error_term/(margin_term + error_term)*100:.1f}%")

# 8. Practical recommendations
print("\n6. PRACTICAL RECOMMENDATIONS")
print("-" * 35)

print("\nGuidelines for choosing C:")
print("• Small C (0.01-0.1): Use when you want a wide margin and can tolerate some misclassifications")
print("• Medium C (1-10): Good starting point for most problems")
print("• Large C (100+): Use when you want to minimize misclassifications, even if it means a narrow margin")
print("• Very large C: Approaches hard-margin SVM behavior")

print("\nCross-validation approach:")
print("1. Try C values on a logarithmic scale: [0.01, 0.1, 1, 10, 100]")
print("2. Use k-fold cross-validation to find optimal C")
print("3. Consider the bias-variance trade-off for your specific problem")

# 9. Comparison with hard-margin SVM
print("\n7. COMPARISON WITH HARD-MARGIN SVM")
print("-" * 40)

print("\nHard-margin SVM (C → ∞):")
print("• Objective: min (1/2)||w||²")
print("• Constraint: yᵢ(wᵀxᵢ + b) ≥ 1 for all i")
print("• No misclassifications allowed")
print("• May not have a solution for non-linearly separable data")

print("\nSoft-margin SVM (finite C):")
print("• Objective: min (1/2)||w||² + C∑ᵢ ξᵢ")
print("• Constraint: yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ for all i")
print("• Allows some misclassifications (ξᵢ > 0)")
print("• Always has a solution")

print(f"\nResults Summary:")
print(f"Linear Dataset - Best C: {C_array[np.argmin(misclassifications_linear)]}")
print(f"Non-linear Dataset - Best C: {C_array[np.argmin(misclassifications_nonlinear)]}")

print(f"\nPlots saved to: {save_dir}")
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
