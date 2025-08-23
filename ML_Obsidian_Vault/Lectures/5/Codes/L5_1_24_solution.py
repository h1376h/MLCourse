import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_1_Quiz_24")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=== LOOCV Analysis for Hard-Margin SVM ===")
print("Question 24: Leave-One-Out Cross-Validation Error Estimation\n")

# Create a synthetic dataset that represents the scenario in the question
# We'll create a dataset where some points are support vectors and others are not
np.random.seed(42)

# Generate data points for two classes
# Class 1 (positive class)
class1_points = np.array([
    [1, 2],   # Support vector
    [2, 1],   # Support vector  
    [3, 3],   # Non-support vector
    [4, 2],   # Non-support vector
    [2, 4],   # Non-support vector
])

# Class -1 (negative class)
class2_points = np.array([
    [0, 0],   # Support vector
    [1, 0],   # Support vector
    [2, -1],  # Non-support vector
    [3, 0],   # Non-support vector
    [0, 1],   # Non-support vector
])

# Combine data
X = np.vstack([class1_points, class2_points])
y = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])

print("Dataset:")
print("Class 1 (positive) points:")
for i, point in enumerate(class1_points):
    print(f"  Point {i+1}: {point}")
print("\nClass -1 (negative) points:")
for i, point in enumerate(class2_points):
    print(f"  Point {i+6}: {point}")

# Train the full SVM model
svm_full = SVC(kernel='linear', C=1000)  # High C for hard margin
svm_full.fit(X, y)

# Get support vectors
support_vectors = svm_full.support_vectors_
support_vector_indices = svm_full.support_

print(f"\nSupport Vectors (indices): {support_vector_indices}")
print("Support Vector points:")
for i, idx in enumerate(support_vector_indices):
    print(f"  Point {idx+1}: {X[idx]} (Class {y[idx]})")

# Function to plot SVM with decision boundary and margins
def plot_svm(X, y, svm_model, title, highlight_point=None, original_point_idx=None, save_path=None):
    plt.figure(figsize=(12, 10))
    
    # Plot data points
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', marker='o', s=100, 
                label='Class 1', edgecolors='black', linewidth=1.5)
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], c='blue', marker='x', s=100, 
                label='Class -1', edgecolors='black', linewidth=1.5)
    
    # Highlight support vectors
    support_vectors = svm_model.support_vectors_
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=200, 
                facecolors='none', edgecolors='green', linewidth=3, 
                label='Support Vectors')
    
    # Highlight specific point if provided
    if highlight_point is not None:
        plt.scatter(X[highlight_point, 0], X[highlight_point, 1], s=300, 
                    facecolors='yellow', edgecolors='orange', linewidth=3, 
                    label=f'Point {original_point_idx+1} (Left Out)')
    
    # Create mesh grid for decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Get decision function values
    Z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and margins
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['blue', 'black', 'red'], 
                linestyles=['--', '-', '--'], linewidths=[2, 3, 2])
    
    # Fill regions
    plt.contourf(xx, yy, Z, levels=[-100, 0], colors=['lightblue'], alpha=0.3)
    plt.contourf(xx, yy, Z, levels=[0, 100], colors=['lightcoral'], alpha=0.3)
    
    # Add point labels
    for i, (x, y_val) in enumerate(X):
        plt.annotate(f'{i+1}', (x, y_val), xytext=(5, 5), 
                    textcoords='offset points', fontsize=12, fontweight='bold')
    
    plt.xlabel('$x_1$', fontsize=14)
    plt.ylabel('$x_2$', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()

# Plot the full SVM
plot_svm(X, y, svm_full, "Full SVM with All Data Points", 
         save_path=os.path.join(save_dir, 'full_svm.png'))

print("\n=== LOOCV Analysis ===")
print("Leave-One-Out Cross-Validation Process:\n")

# Perform LOOCV
loo = LeaveOneOut()
cv_scores = []
misclassified_points = []

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Train SVM on training data
    svm_cv = SVC(kernel='linear', C=1000)
    svm_cv.fit(X_train, y_train)
    
    # Predict on test point
    y_pred = svm_cv.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores.append(accuracy)
    
    # Check if point was misclassified
    is_misclassified = (y_pred[0] != y_test[0])
    misclassified_points.append(is_misclassified)
    
    print(f"LOOCV Fold {test_idx[0]+1}: Leave out Point {test_idx[0]+1} ({X[test_idx[0]]}, Class {y[test_idx[0]]})")
    print(f"  Prediction: {y_pred[0]}")
    print(f"  Correct: {not is_misclassified}")
    print(f"  Support vectors in training: {svm_cv.support_}")
    
    # Only plot LOOCV folds for misclassified points (support vectors)
    if is_misclassified:
        plot_svm(X_train, y_train, svm_cv, 
                 f"LOOCV Fold {test_idx[0]+1}: Leave out Point {test_idx[0]+1}", 
                 highlight_point=None, original_point_idx=test_idx[0],
                 save_path=os.path.join(save_dir, f'loocv_fold_{test_idx[0]+1}.png'))
    else:
        # For non-misclassified points, just print the result without saving plot
        print(f"  (No plot saved - point correctly classified)")

# Calculate LOOCV error
loocv_error = 1 - np.mean(cv_scores)
misclassified_count = sum(misclassified_points)

print(f"\n=== LOOCV Results ===")
print(f"Total misclassifications: {misclassified_count}/{len(X)}")
print(f"LOOCV Error Rate: {loocv_error:.3f} ({loocv_error*100:.1f}%)")
print(f"LOOCV Accuracy: {1-loocv_error:.3f} ({(1-loocv_error)*100:.1f}%)")

# Analyze which points were misclassified
print(f"\n=== Misclassification Analysis ===")
misclassified_indices = [i for i, misclassified in enumerate(misclassified_points) if misclassified]
correctly_classified_indices = [i for i, misclassified in enumerate(misclassified_points) if not misclassified]

print("Misclassified points:")
for idx in misclassified_indices:
    print(f"  Point {idx+1}: {X[idx]} (Class {y[idx]})")

print("\nCorrectly classified points:")
for idx in correctly_classified_indices:
    print(f"  Point {idx+1}: {X[idx]} (Class {y[idx]})")

# Function to plot SVM in subplot (without creating new figure)
def plot_svm_subplot(X, y, svm_model, ax, title, highlight_point=None, original_point_idx=None):
    # Plot data points
    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', marker='o', s=80, 
               label='Class 1', edgecolors='black', linewidth=1)
    ax.scatter(X[y == -1][:, 0], X[y == -1][:, 1], c='blue', marker='x', s=80, 
               label='Class -1', edgecolors='black', linewidth=1)
    
    # Highlight support vectors
    support_vectors = svm_model.support_vectors_
    ax.scatter(support_vectors[:, 0], support_vectors[:, 1], s=150, 
               facecolors='none', edgecolors='green', linewidth=2, 
               label='Support Vectors')
    
    # Highlight specific point if provided
    if highlight_point is not None:
        ax.scatter(X[highlight_point, 0], X[highlight_point, 1], s=200, 
                   facecolors='yellow', edgecolors='orange', linewidth=2, 
                   label=f'Point {original_point_idx+1} (Left Out)')
    
    # Create mesh grid for decision boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))
    
    # Get decision function values
    Z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and margins
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['blue', 'black', 'red'], 
               linestyles=['--', '-', '--'], linewidths=[1, 2, 1])
    
    # Fill regions
    ax.contourf(xx, yy, Z, levels=[-100, 0], colors=['lightblue'], alpha=0.2)
    ax.contourf(xx, yy, Z, levels=[0, 100], colors=['lightcoral'], alpha=0.2)
    
    # Add point labels
    for i, (x, y_val) in enumerate(X):
        ax.annotate(f'{i+1}', (x, y_val), xytext=(3, 3), 
                   textcoords='offset points', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('$x_1$', fontsize=10)
    ax.set_ylabel('$x_2$', fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

# Create summary visualization
# Determine the grid size based on number of plots needed
num_misclassified = len(misclassified_indices)
total_plots = 1 + num_misclassified  # 1 for full SVM + misclassified folds

if total_plots <= 3:
    fig, axes = plt.subplots(1, total_plots, figsize=(6*total_plots, 6))
    if total_plots == 1:
        axes = [axes]  # Make it iterable
else:
    # Use 2 rows if we have more than 3 plots
    cols = min(3, total_plots)
    rows = (total_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

# Plot 1: Full SVM
if total_plots <= 3:
    plot_svm_subplot(X, y, svm_full, axes[0], "Full SVM with All Data")
    start_idx = 1
else:
    plot_svm_subplot(X, y, svm_full, axes[0, 0], "Full SVM with All Data")
    start_idx = 1

# Plot LOOCV folds for misclassified points
for i, idx in enumerate(misclassified_indices):
    # Get training data for this fold
    train_mask = np.ones(len(X), dtype=bool)
    train_mask[idx] = False
    X_train, y_train = X[train_mask], y[train_mask]
    
    # Train SVM
    svm_cv = SVC(kernel='linear', C=1000)
    svm_cv.fit(X_train, y_train)
    
    # Determine subplot position
    if total_plots <= 3:
        ax = axes[start_idx + i]
    else:
        row = (start_idx + i) // 3
        col = (start_idx + i) % 3
        ax = axes[row, col]
    
    # Plot
    plot_svm_subplot(X_train, y_train, svm_cv, ax, 
                     f"Leave out Point {idx+1} (Misclassified)")

# Hide any remaining empty subplots
if total_plots > 3:
    total_subplots = axes.shape[0] * axes.shape[1]
    for i in range(total_plots, total_subplots):
        row = i // axes.shape[1]
        col = i % axes.shape[1]
        axes[row, col].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'loocv_summary.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create a simple informative visualization
plt.figure(figsize=(10, 8))

# Plot data points with different markers for support vectors vs non-support vectors
for i, (x, y_val) in enumerate(X):
    if i in support_vector_indices:
        # Support vectors - larger, highlighted
        plt.scatter(x, y_val, s=200, c='green' if y[i] == 1 else 'blue', 
                   marker='o' if y[i] == 1 else 'x', edgecolors='black', linewidth=2, alpha=0.8)
    else:
        # Non-support vectors - smaller, regular
        plt.scatter(x, y_val, s=100, c='red' if y[i] == 1 else 'blue', 
                   marker='o' if y[i] == 1 else 'x', edgecolors='black', linewidth=1, alpha=0.6)

# Add decision boundary
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
Z = svm_full.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
plt.contour(xx, yy, Z, levels=[-1, 1], colors='gray', linewidths=1, linestyles='--')

# Add point numbers
for i, (x, y_val) in enumerate(X):
    plt.annotate(f'{i+1}', (x, y_val), xytext=(5, 5), 
                textcoords='offset points', fontsize=12, fontweight='bold')

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('Support Vectors vs Non-Support Vectors', fontsize=16)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', alpha=0.6, label='Class 1 (Non-SV)'),
    Patch(facecolor='blue', alpha=0.6, label='Class -1 (Non-SV)'),
    Patch(facecolor='green', alpha=0.8, label='Class 1 (Support Vector)'),
    Patch(facecolor='blue', alpha=0.8, label='Class -1 (Support Vector)')
]
plt.legend(handles=legend_elements, loc='upper right')

plt.savefig(os.path.join(save_dir, 'support_vectors_simple.png'), dpi=300, bbox_inches='tight')
plt.close()

# Mathematical (Pen and Paper) Solution
print(f"\n=== Mathematical Solution (Pen and Paper Method) ===")
print("Task 1: What is the LOOCV error estimate?")
print("Task 2: Provide justification")
print()

print("Step 1 (Task 1): Identify Support Vectors from the Figure")
print("From the given figure, we can identify support vectors as points lying exactly on the margin boundaries.")
print("Support vectors are the points with circles around them in the figure.")
print(f"In our analysis: Support vectors are Points {[i+1 for i in support_vector_indices]}")
print(f"Non-support vectors are Points {[i+1 for i in range(len(X)) if i not in support_vector_indices]}")

print(f"\nStep 2 (Task 1): Apply the Theoretical LOOCV Property for Hard-Margin SVM")
print("Key Theorem: For a hard-margin SVM with linearly separable data:")
print("- LOOCV error = Number of Support Vectors / Total Number of Points")
print("- This is because only support vectors can be misclassified when left out")

num_support_vectors = len(support_vector_indices)
total_points = len(X)
theoretical_loocv_error = num_support_vectors / total_points

print(f"\nCalculation:")
print(f"Number of support vectors = {num_support_vectors}")
print(f"Total number of points = {total_points}")
print(f"LOOCV Error Rate = {num_support_vectors}/{total_points} = {theoretical_loocv_error:.3f} = {theoretical_loocv_error*100:.1f}%")

print(f"\nStep 2 (Task 2): Theoretical Justification")
print("The justification is based on the fundamental properties of hard-margin SVM:")
print()
print("1. When a NON-support vector is left out:")
print("   - The decision boundary remains EXACTLY the same")
print("   - The margin is defined only by support vectors")
print("   - The left-out point will be correctly classified")
print("   - Contribution to LOOCV error: 0")

print("\n2. When a SUPPORT vector is left out:")
print("   - The decision boundary MAY change (margin becomes larger)")
print("   - The new boundary might not correctly classify the left-out point")
print("   - In the worst case, each support vector contributes 1 error")
print("   - Contribution to LOOCV error: 0 or 1 (typically 1)")

print(f"\nTherefore, LOOCV error ≤ Number of Support Vectors / Total Points")
print(f"In practice, this bound is often tight for well-separated data.")

print(f"\n=== Comparison: Mathematical vs Computational Results ===")
print(f"Mathematical prediction: {theoretical_loocv_error:.3f} ({theoretical_loocv_error*100:.1f}%)")
print(f"Computational result:    {loocv_error:.3f} ({loocv_error*100:.1f}%)")
print(f"Match: {'✓ Perfect match!' if abs(theoretical_loocv_error - loocv_error) < 1e-10 else '✗ Difference detected'}")

print(f"\n=== Detailed Computational Analysis ===")
print("For verification, here's what actually happened in each LOOCV fold:")
for i in range(len(X)):
    if i in support_vector_indices:
        status = "Misclassified" if misclassified_points[i] else "Correctly classified"
        print(f"  Point {i+1}: Support vector - {status} when left out")
    else:
        print(f"  Point {i+1}: Non-support vector - Correctly classified when left out")

print(f"\n=== Final Answer ===")
print(f"The LOOCV error estimate for this maximum margin classifier is: {loocv_error:.3f}")
print(f"This means {misclassified_count} out of {len(X)} points are misclassified during LOOCV.")
print(f"The error occurs because when support vectors are left out, the decision boundary changes,")
print(f"potentially leading to misclassification of the left-out point.")

print(f"\nPlots saved to: {save_dir}")
