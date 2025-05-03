import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, hinge_loss, log_loss, zero_one_loss
from matplotlib.colors import ListedColormap

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_4_Quiz_17")
os.makedirs(save_dir, exist_ok=True)

print("Question 17: Robust Classification")
print("================================")

# Define Pocket algorithm for custom implementation
def pocket_algorithm(X, y, max_iterations=100, learning_rate=1.0):
    """
    Implementation of the Pocket algorithm for non-separable data.
    
    Parameters:
        X: Feature matrix (n_samples, n_features)
        y: Class labels (-1 or 1)
        max_iterations: Maximum number of iterations
        learning_rate: Learning rate
        
    Returns:
        best_w: Best weight vector (n_features)
        best_b: Best bias term
        best_accuracy: Best accuracy achieved
    """
    n_samples, n_features = X.shape
    
    # Initialize weights
    w = np.zeros(n_features)
    b = 0
    
    # Initialize best weights (pocket)
    best_w = w.copy()
    best_b = b
    
    # Calculate initial accuracy
    predictions = np.sign(np.dot(X, w) + b)
    best_accuracy = np.mean(predictions == y)
    
    # Run for a maximum number of iterations
    for _ in range(max_iterations):
        misclassified_indices = []
        
        # Find all misclassified samples
        for i in range(n_samples):
            y_pred = np.sign(np.dot(X[i], w) + b)
            if y[i] * y_pred <= 0:  # Misclassified
                misclassified_indices.append(i)
        
        # If no misclassifications, we're done
        if len(misclassified_indices) == 0:
            break
        
        # Pick a random misclassified sample
        i = np.random.choice(misclassified_indices)
        
        # Update weights
        w += learning_rate * y[i] * X[i]
        b += learning_rate * y[i]
        
        # Calculate new accuracy
        predictions = np.sign(np.dot(X, w) + b)
        accuracy = np.mean(predictions == y)
        
        # Update pocket if better
        if accuracy > best_accuracy:
            best_w = w.copy()
            best_b = b
            best_accuracy = accuracy
    
    return best_w, best_b, best_accuracy

# Task 1: Create a dataset with class overlap and outliers
print("\nTask 1: Generating Dataset with Class Overlap and Outliers")
print("------------------------------------------------------")

# Generate a dataset with significant class overlap
np.random.seed(42)

# Core data for two overlapping classes
n_samples = 200
n_features = 2

# Class 1: Gaussian centered at (2, 2)
mean1 = np.array([2, 2])
cov1 = np.array([[1.5, 0.6], [0.6, 1.5]])
X1 = np.random.multivariate_normal(mean1, cov1, n_samples)

# Class -1: Gaussian centered at (0, 0)
mean2 = np.array([0, 0])
cov2 = np.array([[1.5, -0.4], [-0.4, 1.5]])
X2 = np.random.multivariate_normal(mean2, cov2, n_samples)

# Combine data
X = np.vstack([X1, X2])
y = np.hstack([np.ones(n_samples), -np.ones(n_samples)])

# Add a few extreme outliers to the minority class
n_outliers = 5
outlier_points = np.array([
    [8, 0],
    [7, 1],
    [9, -1],
    [8, -2],
    [10, 0]
])
outlier_labels = np.ones(n_outliers)  # Assuming Class 1 as the minority class

# Add outliers to the dataset
X = np.vstack([X, outlier_points])
y = np.hstack([y, outlier_labels])

# Define color maps for better visualization
class1_color = '#4169E1'  # Royal blue
class2_color = '#FF6347'  # Tomato
outlier_color = '#32CD32'  # Lime green
region1_color = '#B3C6FF'  # Light blue
region2_color = '#FFB3B3'  # Light red
custom_cmap = ListedColormap([region2_color, region1_color])

# Visualize the dataset
plt.figure(figsize=(10, 8))
plt.scatter(X[y == 1, 0], X[y == 1, 1], c=class1_color, marker='o', alpha=0.7, label='Class 1', s=70)
plt.scatter(X[y == -1, 0], X[y == -1, 1], c=class2_color, marker='x', alpha=0.7, label='Class -1', s=70)
plt.scatter(outlier_points[:, 0], outlier_points[:, 1], c=outlier_color, marker='s', s=120, alpha=0.9, label='Outliers (Class 1)', edgecolors='k')

plt.xlabel('Feature 1', fontsize=14)
plt.ylabel('Feature 2', fontsize=14)
plt.title('Dataset with Class Overlap and Outliers', fontsize=16)
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "dataset_with_outliers.png"), dpi=300, bbox_inches='tight')

print("Dataset created with the following properties:")
print(f"- Total samples: {len(X)}")
print(f"- Class 1 samples: {np.sum(y == 1)}")
print(f"- Class -1 samples: {np.sum(y == -1)}")
print(f"- Number of extreme outliers: {n_outliers}")
print("- Significant class overlap in the feature space")
print("- Outliers are added to Class 1, making it a harder classification problem")

# Task 2: Compare 0-1 loss and hinge loss
print("\nTask 2: Comparing 0-1 Loss and Hinge Loss")
print("--------------------------------------")

# Train a model with 0-1 loss (Perceptron)
perceptron = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
perceptron.fit(X, y)
y_pred_perceptron = perceptron.predict(X)
accuracy_perceptron = accuracy_score(y, y_pred_perceptron)
zero_one_loss_value = zero_one_loss(y, y_pred_perceptron)

# Train a model with hinge loss (Linear SVM)
svm = LinearSVC(loss='hinge', C=1.0, dual="auto", max_iter=10000, random_state=42)
svm.fit(X, y)
y_pred_svm = svm.predict(X)
accuracy_svm = accuracy_score(y, y_pred_svm)
hinge_loss_value = np.mean(np.maximum(0, 1 - y * svm.decision_function(X)))

print(f"Perceptron (0-1 Loss):")
print(f"  Accuracy: {accuracy_perceptron:.4f}")
print(f"  0-1 Loss: {zero_one_loss_value:.4f}")

print(f"\nLinear SVM (Hinge Loss):")
print(f"  Accuracy: {accuracy_svm:.4f}")
print(f"  Hinge Loss: {hinge_loss_value:.4f}")

print("\nComparison:")
print("- The hinge loss is continuous and convex, penalizing not only misclassifications but also points")
print("  that are correctly classified but with small margins.")
print("- The 0-1 loss is discontinuous and only counts misclassifications, without considering margins.")
print("- For datasets with outliers, hinge loss is typically more robust because it focuses on")
print("  maximizing the margin rather than just minimizing misclassifications.")

# Set up meshgrid for contour plots
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Create a figure with side-by-side comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Plot Perceptron decision boundary (left subplot)
Z_perceptron = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
Z_perceptron = Z_perceptron.reshape(xx.shape)

axes[0].contourf(xx, yy, Z_perceptron, alpha=0.4, cmap=custom_cmap)
axes[0].contour(xx, yy, Z_perceptron, colors='k', linestyles=['-'], linewidths=2, levels=[-1, 0, 1])
axes[0].scatter(X[y == 1, 0], X[y == 1, 1], c=class1_color, marker='o', alpha=0.7, label='Class 1', s=70)
axes[0].scatter(X[y == -1, 0], X[y == -1, 1], c=class2_color, marker='x', alpha=0.7, label='Class -1', s=70)
axes[0].scatter(outlier_points[:, 0], outlier_points[:, 1], c=outlier_color, marker='s', s=120, alpha=0.9, label='Outliers', edgecolors='k')
axes[0].set_xlabel('Feature 1', fontsize=14)
axes[0].set_ylabel('Feature 2', fontsize=14)
axes[0].set_title('Perceptron (0-1 Loss)', fontsize=16)
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=12, loc='upper right')

# Add accuracy text below the plot
acc_text = f"Accuracy: {accuracy_perceptron:.4f}"
axes[0].text(0.5, -0.1, acc_text, ha='center', va='center', transform=axes[0].transAxes, fontsize=14, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

# Plot SVM decision boundary (right subplot)
Z_svm = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z_svm = Z_svm.reshape(xx.shape)

axes[1].contourf(xx, yy, Z_svm, alpha=0.4, cmap=custom_cmap)
axes[1].contour(xx, yy, Z_svm, colors='k', linestyles=['-'], linewidths=2, levels=[-1, 0, 1])
axes[1].scatter(X[y == 1, 0], X[y == 1, 1], c=class1_color, marker='o', alpha=0.7, label='Class 1', s=70)
axes[1].scatter(X[y == -1, 0], X[y == -1, 1], c=class2_color, marker='x', alpha=0.7, label='Class -1', s=70)
axes[1].scatter(outlier_points[:, 0], outlier_points[:, 1], c=outlier_color, marker='s', s=120, alpha=0.9, label='Outliers', edgecolors='k')
axes[1].set_xlabel('Feature 1', fontsize=14)
axes[1].set_ylabel('Feature 2', fontsize=14)
axes[1].set_title('Linear SVM (Hinge Loss)', fontsize=16)
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=12, loc='upper right')

# Add accuracy text below the plot
acc_text = f"Accuracy: {accuracy_svm:.4f}"
axes[1].text(0.5, -0.1, acc_text, ha='center', va='center', transform=axes[1].transAxes, fontsize=14, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "0-1_vs_hinge_loss.png"), dpi=300, bbox_inches='tight')

# Task 3: Handling outliers - Pocket vs Perceptron
print("\nTask 3: Handling Outliers - Pocket vs Perceptron")
print("---------------------------------------------")

# Implement standard Perceptron
perceptron_custom = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
perceptron_custom.fit(X, y)
y_pred_perceptron_custom = perceptron_custom.predict(X)
accuracy_perceptron_custom = accuracy_score(y, y_pred_perceptron_custom)

# Implement Pocket algorithm
pocket_w, pocket_b, pocket_accuracy = pocket_algorithm(X, y, max_iterations=1000)
y_pred_pocket = np.sign(np.dot(X, pocket_w) + pocket_b)
accuracy_pocket = np.mean(y_pred_pocket == y)

print(f"Standard Perceptron:")
print(f"  Accuracy: {accuracy_perceptron_custom:.4f}")

print(f"\nPocket Algorithm:")
print(f"  Accuracy: {accuracy_pocket:.4f}")

print("\nOutlier Handling Comparison:")
print("- The standard Perceptron is sensitive to outliers because it continues to update")
print("  weights as long as misclassifications exist, potentially overfitting to outliers.")
print("- The Pocket algorithm is more robust to outliers because it keeps the best-performing")
print("  weights overall, even if they don't perfectly classify all points including outliers.")
print("- This makes the Pocket algorithm better suited for real-world data with noise and outliers.")

# Create a figure with side-by-side comparison for Perceptron vs Pocket
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Plot Perceptron (left subplot)
Z_perceptron_custom = perceptron_custom.predict(np.c_[xx.ravel(), yy.ravel()])
Z_perceptron_custom = Z_perceptron_custom.reshape(xx.shape)

axes[0].contourf(xx, yy, Z_perceptron_custom, alpha=0.4, cmap=custom_cmap)
axes[0].contour(xx, yy, Z_perceptron_custom, colors='k', linestyles=['-'], linewidths=2, levels=[-1, 0, 1])
axes[0].scatter(X[y == 1, 0], X[y == 1, 1], c=class1_color, marker='o', alpha=0.7, label='Class 1', s=70)
axes[0].scatter(X[y == -1, 0], X[y == -1, 1], c=class2_color, marker='x', alpha=0.7, label='Class -1', s=70)
axes[0].scatter(outlier_points[:, 0], outlier_points[:, 1], c=outlier_color, marker='s', s=120, alpha=0.9, label='Outliers', edgecolors='k')
axes[0].set_xlabel('Feature 1', fontsize=14)
axes[0].set_ylabel('Feature 2', fontsize=14)
axes[0].set_title('Standard Perceptron', fontsize=16)
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=12, loc='upper right')

# Add accuracy text below the plot
acc_text = f"Accuracy: {accuracy_perceptron_custom:.4f}"
axes[0].text(0.5, -0.1, acc_text, ha='center', va='center', transform=axes[0].transAxes, fontsize=14, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

# Plot Pocket (right subplot)
Z_pocket = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], pocket_w) + pocket_b)
Z_pocket = Z_pocket.reshape(xx.shape)

axes[1].contourf(xx, yy, Z_pocket, alpha=0.4, cmap=custom_cmap)
axes[1].contour(xx, yy, Z_pocket, colors='k', linestyles=['-'], linewidths=2, levels=[-1, 0, 1])
axes[1].scatter(X[y == 1, 0], X[y == 1, 1], c=class1_color, marker='o', alpha=0.7, label='Class 1', s=70)
axes[1].scatter(X[y == -1, 0], X[y == -1, 1], c=class2_color, marker='x', alpha=0.7, label='Class -1', s=70)
axes[1].scatter(outlier_points[:, 0], outlier_points[:, 1], c=outlier_color, marker='s', s=120, alpha=0.9, label='Outliers', edgecolors='k')
axes[1].set_xlabel('Feature 1', fontsize=14)
axes[1].set_ylabel('Feature 2', fontsize=14)
axes[1].set_title('Pocket Algorithm', fontsize=16)
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=12, loc='upper right')

# Add accuracy text below the plot
acc_text = f"Accuracy: {accuracy_pocket:.4f}"
axes[1].text(0.5, -0.1, acc_text, ha='center', va='center', transform=axes[1].transAxes, fontsize=14, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "perceptron_vs_pocket.png"), dpi=300, bbox_inches='tight')

# Task 4: Bias-Variance Tradeoff
print("\nTask 4: Bias-Variance Tradeoff")
print("---------------------------")

print("Bias-Variance Tradeoff in the Context of this Problem:")
print("1. High-Variance Models:")
print("   - Models that perfectly fit all points including outliers (low bias, high variance)")
print("   - Example: Standard Perceptron trying to classify all points correctly")
print("   - These models may perform well on the training data but generalize poorly to new data")
print("   - They're sensitive to the specific noise/outliers in the training set")

print("\n2. High-Bias Models:")
print("   - Models that ignore some outliers to find a better overall pattern (high bias, low variance)")
print("   - Example: Linear SVM with regularization or Pocket algorithm")
print("   - These models may not fit the training data perfectly but generalize better to new data")
print("   - They're more robust to noise and outliers in the training set")

print("\n3. In this dataset:")
print("   - The standard Perceptron with 0-1 loss tends toward higher variance")
print("   - The SVM with hinge loss and the Pocket algorithm tend toward lower variance")
print("   - The tradeoff manifests in how the models handle the outliers in Class 1")

# Create a figure to illustrate the bias-variance tradeoff
plt.figure(figsize=(12, 8))

# Plot data points
plt.scatter(X[y == 1, 0], X[y == 1, 1], c=class1_color, marker='o', alpha=0.7, label='Class 1', s=70)
plt.scatter(X[y == -1, 0], X[y == -1, 1], c=class2_color, marker='x', alpha=0.7, label='Class -1', s=70)
plt.scatter(outlier_points[:, 0], outlier_points[:, 1], c=outlier_color, marker='s', s=120, alpha=0.9, label='Outliers', edgecolors='k')

# Plot decision boundaries
plt.contour(xx, yy, Z_perceptron_custom, colors='#e41a1c', linestyles=['-'], linewidths=2, levels=[0], label='Perceptron')
plt.contour(xx, yy, Z_svm, colors='#377eb8', linestyles=['--'], linewidths=2, levels=[0], label='SVM')
plt.contour(xx, yy, Z_pocket, colors='#4daf4a', linestyles=['-.'], linewidths=2, levels=[0], label='Pocket')

plt.xlabel('Feature 1', fontsize=14)
plt.ylabel('Feature 2', fontsize=14)
plt.title('Bias-Variance Tradeoff: Decision Boundaries Comparison', fontsize=16)
plt.grid(True, alpha=0.3)

# Add a custom legend for boundaries
from matplotlib.lines import Line2D
custom_lines = [
    Line2D([0], [0], color='#e41a1c', linestyle='-', linewidth=2),
    Line2D([0], [0], color='#377eb8', linestyle='--', linewidth=2),
    Line2D([0], [0], color='#4daf4a', linestyle='-.', linewidth=2)
]
first_legend = plt.legend(custom_lines, ['Perceptron (High Variance)', 'SVM (Balanced)', 'Pocket (Lower Variance)'], 
                         loc='lower left', fontsize=12, title='Decision Boundaries')
plt.gca().add_artist(first_legend)

# Add a second legend for the data points
plt.legend(loc='upper right', fontsize=12, title='Data Points')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "bias_variance_tradeoff.png"), dpi=300, bbox_inches='tight')

# Task 5: Linear Discriminant Analysis (LDA) comparison
print("\nTask 5: LDA Comparison")
print("------------------")

# Implement LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)
y_pred_lda = lda.predict(X)
accuracy_lda = accuracy_score(y, y_pred_lda)

print(f"LDA:")
print(f"  Accuracy: {accuracy_lda:.4f}")

print("\nLDA vs. Perceptron Decision Boundary:")
print("- LDA assumes Gaussian distributions with equal covariance matrices for each class")
print("- LDA focuses on finding the projection that maximizes class separation, considering")
print("  both the between-class and within-class variances")
print("- Perceptron simply tries to find any hyperplane that separates the classes")
print("- Due to these differences, LDA often handles class overlap better but can be")
print("  more affected by outliers if they significantly distort the class distributions")

# Create a figure with side-by-side comparison for Perceptron vs LDA
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Plot Perceptron (left subplot)
axes[0].contourf(xx, yy, Z_perceptron_custom, alpha=0.4, cmap=custom_cmap)
axes[0].contour(xx, yy, Z_perceptron_custom, colors='k', linestyles=['-'], linewidths=2, levels=[-1, 0, 1])
axes[0].scatter(X[y == 1, 0], X[y == 1, 1], c=class1_color, marker='o', alpha=0.7, label='Class 1', s=70)
axes[0].scatter(X[y == -1, 0], X[y == -1, 1], c=class2_color, marker='x', alpha=0.7, label='Class -1', s=70)
axes[0].scatter(outlier_points[:, 0], outlier_points[:, 1], c=outlier_color, marker='s', s=120, alpha=0.9, label='Outliers', edgecolors='k')
axes[0].set_xlabel('Feature 1', fontsize=14)
axes[0].set_ylabel('Feature 2', fontsize=14)
axes[0].set_title('Standard Perceptron', fontsize=16)
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=12, loc='upper right')

# Add accuracy text below the plot
acc_text = f"Accuracy: {accuracy_perceptron_custom:.4f}"
axes[0].text(0.5, -0.1, acc_text, ha='center', va='center', transform=axes[0].transAxes, fontsize=14, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

# Plot LDA (right subplot)
Z_lda = lda.predict(np.c_[xx.ravel(), yy.ravel()])
Z_lda = Z_lda.reshape(xx.shape)

axes[1].contourf(xx, yy, Z_lda, alpha=0.4, cmap=custom_cmap)
axes[1].contour(xx, yy, Z_lda, colors='k', linestyles=['-'], linewidths=2, levels=[-1, 0, 1])
axes[1].scatter(X[y == 1, 0], X[y == 1, 1], c=class1_color, marker='o', alpha=0.7, label='Class 1', s=70)
axes[1].scatter(X[y == -1, 0], X[y == -1, 1], c=class2_color, marker='x', alpha=0.7, label='Class -1', s=70)
axes[1].scatter(outlier_points[:, 0], outlier_points[:, 1], c=outlier_color, marker='s', s=120, alpha=0.9, label='Outliers', edgecolors='k')
axes[1].set_xlabel('Feature 1', fontsize=14)
axes[1].set_ylabel('Feature 2', fontsize=14)
axes[1].set_title('Linear Discriminant Analysis (LDA)', fontsize=16)
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=12, loc='upper right')

# Add accuracy text below the plot
acc_text = f"Accuracy: {accuracy_lda:.4f}"
axes[1].text(0.5, -0.1, acc_text, ha='center', va='center', transform=axes[1].transAxes, fontsize=14, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "perceptron_vs_lda.png"), dpi=300, bbox_inches='tight')

# Summary
print("\nSummary")
print("-------")
print("1. 0-1 loss vs. Hinge loss:")
print(f"   - 0-1 loss (Perceptron) accuracy: {accuracy_perceptron:.4f}")
print(f"   - Hinge loss (SVM) accuracy: {accuracy_svm:.4f}")
print("   - Hinge loss is more robust to outliers due to margin-awareness")

print("\n2. Perceptron vs. Pocket:")
print(f"   - Standard Perceptron accuracy: {accuracy_perceptron_custom:.4f}")
print(f"   - Pocket algorithm accuracy: {accuracy_pocket:.4f}")
print("   - Pocket is more robust to outliers by keeping the best weights overall")

print("\n3. Bias-Variance Tradeoff:")
print("   - Higher variance models (standard Perceptron) are more sensitive to outliers")
print("   - Lower variance models (SVM, Pocket) are more robust to outliers")

print("\n4. LDA vs. Perceptron:")
print(f"   - LDA accuracy: {accuracy_lda:.4f}")
print(f"   - Perceptron accuracy: {accuracy_perceptron_custom:.4f}")
print("   - LDA places the decision boundary differently due to its statistical approach")

print("\nConclusion:")
print("For datasets with significant class overlap and outliers, models that")
print("balance the bias-variance tradeoff (like SVM with hinge loss or the Pocket algorithm)")
print("tend to perform better than simple models focused only on classification error.") 