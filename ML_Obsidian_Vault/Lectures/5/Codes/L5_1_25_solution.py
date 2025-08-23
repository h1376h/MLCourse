import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_1_Quiz_25")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("1D HARD-MARGIN SVM ANALYSIS")
print("=" * 80)

# Dataset
X = np.array([1, 2, 3.5, 4, 5]).reshape(-1, 1)
y = np.array([-1, -1, -1, 1, 1])

print("\nDataset:")
print("X = ", X.flatten())
print("y = ", y)
print("\nData points:")
for i in range(len(X)):
    print(f"Point {i+1}: x = {X[i][0]}, y = {y[i]}")

# STEP 1: Visualize the data
print("\n" + "="*50)
print("STEP 1: DATA VISUALIZATION")
print("="*50)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)

# Plot data points
negative_points = X[y == -1]
positive_points = X[y == 1]

plt.scatter(negative_points, np.zeros_like(negative_points), 
           s=150, c='red', marker='o', facecolors='none', edgecolors='red', linewidth=2,
           label='Class -1 (Negative)')
plt.scatter(positive_points, np.zeros_like(positive_points), 
           s=150, c='blue', marker='o', facecolors='blue', edgecolors='blue', linewidth=2,
           label='Class +1 (Positive)')

# Add point labels
for i, x_val in enumerate(X.flatten()):
    plt.annotate(f'({x_val}, {y[i]})', (x_val, 0), 
                xytext=(0, 20), textcoords='offset points', 
                ha='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

plt.xlim(0, 6)
plt.ylim(-0.5, 0.5)
plt.xlabel('x')
plt.title('1D SVM Dataset')
plt.grid(True, alpha=0.3)
plt.legend()

# STEP 2: Analytical Solution for Hard-Margin SVM
print("\n" + "="*50)
print("STEP 2: ANALYTICAL SOLUTION")
print("="*50)

print("\nFor a 1D hard-margin SVM, we need to find the optimal hyperplane that maximizes the margin.")
print("The decision boundary is at x = -b/w")
print("The margin is 2/|w|, so we want to minimize |w| subject to constraints.")

# Find the optimal decision boundary analytically
# The optimal boundary should be equidistant from the closest points of each class
closest_negative = max(X[y == -1])  # rightmost negative point
closest_positive = min(X[y == 1])   # leftmost positive point

print(f"\nClosest negative point: x = {closest_negative[0]}")
print(f"Closest positive point: x = {closest_positive[0]}")

# Optimal decision boundary is the midpoint
optimal_boundary = (closest_negative[0] + closest_positive[0]) / 2
print(f"Optimal decision boundary: x = {optimal_boundary}")

# Calculate w and b
# Decision boundary: wx + b = 0, so x = -b/w
# We want x = optimal_boundary, so optimal_boundary = -b/w
# We can choose w = 1 for simplicity, then b = -optimal_boundary
w_analytical = 1.0
b_analytical = -optimal_boundary

print(f"\nAnalytical solution:")
print(f"w = {w_analytical}")
print(f"b = {b_analytical}")
print(f"Decision function: f(x) = sign({w_analytical}x + {b_analytical})")
print(f"Decision boundary: x = {optimal_boundary}")

# Calculate margin
margin_analytical = 2 / abs(w_analytical)
print(f"Margin = 2/|w| = {margin_analytical}")

# Distance from support vectors to boundary
distance_to_boundary = abs(closest_positive[0] - optimal_boundary)
print(f"Distance from each support vector to boundary: {distance_to_boundary}")

# STEP 3: sklearn SVM Solution
print("\n" + "="*50)
print("STEP 3: SKLEARN SVM SOLUTION")
print("="*50)

# Train hard-margin SVM (C=1e6 approximates hard-margin)
svm = SVC(kernel='linear', C=1e6)
svm.fit(X, y)

w_sklearn = svm.coef_[0][0]
b_sklearn = svm.intercept_[0]

print(f"sklearn SVM solution:")
print(f"w = {w_sklearn:.6f}")
print(f"b = {b_sklearn:.6f}")
print(f"Decision boundary: x = {-b_sklearn/w_sklearn:.6f}")

# Support vectors
support_vectors = svm.support_vectors_
support_vector_indices = svm.support_
print(f"\nSupport vectors:")
for i, sv_idx in enumerate(support_vector_indices):
    print(f"Support vector {i+1}: x = {X[sv_idx][0]}, y = {y[sv_idx]}")

# STEP 4: Detailed Visualization
print("\n" + "="*50)
print("STEP 4: DETAILED VISUALIZATION")
print("="*50)

plt.subplot(1, 2, 2)

# Create a dense grid for visualization
x_plot = np.linspace(0, 6, 1000)

# Plot decision boundary and margins
decision_boundary_x = -b_analytical / w_analytical
plt.axvline(x=decision_boundary_x, color='green', linestyle='-', linewidth=2, 
           label=f'Decision Boundary (x = {decision_boundary_x:.2f})')

# Plot margins
margin_left = decision_boundary_x - distance_to_boundary
margin_right = decision_boundary_x + distance_to_boundary
plt.axvline(x=margin_left, color='orange', linestyle='--', linewidth=1.5, 
           label=f'Left Margin (x = {margin_left:.2f})')
plt.axvline(x=margin_right, color='orange', linestyle='--', linewidth=1.5, 
           label=f'Right Margin (x = {margin_right:.2f})')

# Shade the margin region
plt.axvspan(margin_left, margin_right, alpha=0.2, color='yellow', 
           label=f'Margin Region (width = {margin_analytical:.2f})')

# Plot data points
plt.scatter(negative_points, np.zeros_like(negative_points), 
           s=150, c='red', marker='o', facecolors='none', edgecolors='red', linewidth=2,
           label='Class -1')
plt.scatter(positive_points, np.zeros_like(positive_points), 
           s=150, c='blue', marker='o', facecolors='blue', edgecolors='blue', linewidth=2,
           label='Class +1')

# Highlight support vectors
support_vector_x = [X[idx][0] for idx in support_vector_indices]
plt.scatter(support_vector_x, np.zeros_like(support_vector_x), 
           s=200, c='none', marker='s', edgecolors='black', linewidth=3,
           label='Support Vectors')

# Add point labels
for i, x_val in enumerate(X.flatten()):
    plt.annotate(f'x={x_val}', (x_val, 0), 
                xytext=(0, 20), textcoords='offset points', 
                ha='center', fontsize=9)

plt.xlim(0, 6)
plt.ylim(-0.5, 0.5)
plt.xlabel('x')
plt.title('1D Hard-Margin SVM Solution')
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, '1d_svm_solution.png'), dpi=300, bbox_inches='tight')

# STEP 5: Training Error Analysis
print("\n" + "="*50)
print("STEP 5: TRAINING ERROR ANALYSIS")
print("="*50)

# Predict on training set
y_pred_train = svm.predict(X)
train_accuracy = accuracy_score(y, y_pred_train)
train_error = (1 - train_accuracy) * 100

print(f"Training predictions:")
for i in range(len(X)):
    prediction = np.sign(w_analytical * X[i][0] + b_analytical)
    print(f"Point {i+1}: x = {X[i][0]}, true y = {y[i]}, predicted y = {int(prediction)}")

print(f"\nTraining accuracy: {train_accuracy:.4f}")
print(f"Training error: {train_error:.2f}%")

# STEP 6: Leave-One-Out Cross-Validation
print("\n" + "="*50)
print("STEP 6: LEAVE-ONE-OUT CROSS-VALIDATION")
print("="*50)

loo = LeaveOneOut()
loo_predictions = []
loo_details = []

print("LOOCV Details:")
for i, (train_idx, test_idx) in enumerate(loo.split(X)):
    # Split data
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Train SVM on training subset
    svm_loo = SVC(kernel='linear', C=1e6)
    svm_loo.fit(X_train, y_train)
    
    # Predict on test point
    y_pred = svm_loo.predict(X_test)
    loo_predictions.append(y_pred[0])
    
    # Store details
    w_loo = svm_loo.coef_[0][0]
    b_loo = svm_loo.intercept_[0]
    boundary_loo = -b_loo / w_loo
    
    print(f"\nFold {i+1}: Leave out point x = {X_test[0][0]}, y = {y_test[0]}")
    print(f"  Training set: {X_train.flatten()}")
    print(f"  Labels: {y_train}")
    print(f"  Learned w = {w_loo:.4f}, b = {b_loo:.4f}")
    print(f"  Decision boundary: x = {boundary_loo:.4f}")
    print(f"  Prediction for x = {X_test[0][0]}: {y_pred[0]}")
    print(f"  Correct: {'Yes' if y_pred[0] == y_test[0] else 'No'}")
    
    loo_details.append({
        'test_point': X_test[0][0],
        'true_label': y_test[0],
        'predicted_label': y_pred[0],
        'w': w_loo,
        'b': b_loo,
        'boundary': boundary_loo,
        'correct': y_pred[0] == y_test[0]
    })

# Calculate LOOCV error
loo_accuracy = accuracy_score(y, loo_predictions)
loo_error = (1 - loo_accuracy) * 100

print(f"\nLOOCV Results:")
print(f"True labels:      {y}")
print(f"LOOCV predictions: {loo_predictions}")
print(f"LOOCV accuracy: {loo_accuracy:.4f}")
print(f"LOOCV error: {loo_error:.2f}%")

# STEP 7: Visualization of LOOCV
print("\n" + "="*50)
print("STEP 7: LOOCV VISUALIZATION")
print("="*50)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, details in enumerate(loo_details):
    ax = axes[i]
    
    # Get training data for this fold
    train_idx = np.arange(len(X))[np.arange(len(X)) != i]
    X_train_fold = X[train_idx]
    y_train_fold = y[train_idx]
    
    # Plot training points
    neg_train = X_train_fold[y_train_fold == -1]
    pos_train = X_train_fold[y_train_fold == 1]
    
    ax.scatter(neg_train, np.zeros_like(neg_train), 
              s=100, c='red', marker='o', facecolors='none', edgecolors='red', linewidth=2,
              label='Train Class -1')
    ax.scatter(pos_train, np.zeros_like(pos_train), 
              s=100, c='blue', marker='o', facecolors='blue', edgecolors='blue', linewidth=2,
              label='Train Class +1')
    
    # Plot test point
    test_color = 'green' if details['correct'] else 'red'
    test_marker = 'o' if details['true_label'] == 1 else 's'
    ax.scatter(details['test_point'], 0, 
              s=150, c=test_color, marker=test_marker, 
              edgecolors='black', linewidth=2,
              label=f'Test Point ({"Correct" if details["correct"] else "Wrong"})')
    
    # Plot decision boundary
    ax.axvline(x=details['boundary'], color='green', linestyle='-', linewidth=2, 
              label=f'Boundary (x={details["boundary"]:.2f})')
    
    ax.set_xlim(0, 6)
    ax.set_ylim(-0.3, 0.3)
    ax.set_xlabel('x')
    ax.set_title(f'Fold {i+1}: Leave out x={details["test_point"]}')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

# Remove the 6th subplot (we only have 5 folds)
fig.delaxes(axes[5])

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'loocv_analysis.png'), dpi=300, bbox_inches='tight')

# STEP 8: Summary and Final Results
print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)

print(f"\n1. OPTIMAL SVM PARAMETERS:")
print(f"   w = {w_analytical}")
print(f"   b = {b_analytical}")
print(f"   Decision function: f(x) = sign({w_analytical}x + {b_analytical})")
print(f"   Decision boundary: x = {optimal_boundary}")
print(f"   Margin width: {margin_analytical}")

print(f"\n2. TRAINING SET ERROR:")
print(f"   Training error: {train_error:.1f}%")
print(f"   All training points are correctly classified (linearly separable data)")

print(f"\n3. LEAVE-ONE-OUT CROSS-VALIDATION ERROR:")
print(f"   LOOCV error: {loo_error:.1f}%")
print(f"   Number of misclassified points in LOOCV: {int(loo_error/20)}")

print(f"\n4. SUPPORT VECTORS:")
print(f"   Support vector 1: x = {closest_negative[0]} (class -1)")
print(f"   Support vector 2: x = {closest_positive[0]} (class +1)")
print(f"   Distance from each support vector to boundary: {distance_to_boundary}")

print(f"\nAnalysis complete! Images saved to: {save_dir}")
plt.show()
