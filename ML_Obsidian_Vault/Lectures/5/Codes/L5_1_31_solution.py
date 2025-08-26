import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_1_Quiz_31")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("=" * 80)
print("QUESTION 31: 2D Hard-Margin SVM with Three Support Vectors")
print("=" * 80)

# Define the dataset
X = np.array([
    [2, 2],  # Point 1 (support vector)
    [0, 2],  # Point 2
    [4, 2],  # Point 3 (support vector)
    [3, 3],  # Point 4 (support vector)
    [5, 3]   # Point 5
])

y = np.array([-1, -1, 1, 1, 1])  # Class labels

# Support vectors indices (0-indexed)
support_vectors_idx = [0, 2, 3]  # Points 1, 3, 4
support_vectors = X[support_vectors_idx]
support_vectors_labels = y[support_vectors_idx]

print("Dataset:")
print("Point ID | x1 | x2 | Class")
print("-" * 25)
for i, (point, label) in enumerate(zip(X, y)):
    sv_marker = " (SV)" if i in support_vectors_idx else ""
    print(f"   {i+1}    | {point[0]} | {point[1]} |  {label}{sv_marker}")

print(f"\nSupport Vectors: Points {[i+1 for i in support_vectors_idx]}")
print(f"Support Vector Coordinates: {support_vectors}")
print(f"Support Vector Labels: {support_vectors_labels}")

# Step 1: Visualize the dataset
print("\n" + "=" * 50)
print("STEP 1: DATASET VISUALIZATION")
print("=" * 50)

def plot_dataset(X, y, support_vectors_idx, title, filename):
    """Plot the dataset with support vectors highlighted"""
    plt.figure(figsize=(10, 8))
    
    # Plot all points
    for i, (point, label) in enumerate(zip(X, y)):
        if i in support_vectors_idx:
            # Support vectors
            plt.scatter(point[0], point[1], s=300, c='red' if label == 1 else 'blue',
                       marker='s', edgecolors='black', linewidth=3, 
                       label=f'Point {i+1} (SV)' if i == support_vectors_idx[0] else "")
        else:
            # Non-support vectors
            plt.scatter(point[0], point[1], s=150, c='red' if label == 1 else 'blue',
                       marker='o', alpha=0.7, 
                       label=f'Point {i+1}' if i == 0 or (i > 0 and i-1 not in support_vectors_idx) else "")
    
    plt.xlabel(r'$x_1$', fontsize=14)
    plt.ylabel(r'$x_2$', fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')
    plt.xlim(-1, 6)
    plt.ylim(1, 4)
    
    # Add point labels
    for i, point in enumerate(X):
        plt.annotate(f'{i+1}', (point[0], point[1]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()

plot_dataset(X, y, support_vectors_idx, 
            r'Dataset with Support Vectors Highlighted', 
            'step1_dataset.png')

# Step 2: Find optimal weight vector and bias using support vector constraints
print("\n" + "=" * 50)
print("STEP 2: FINDING OPTIMAL WEIGHT VECTOR AND BIAS")
print("=" * 50)

# For hard-margin SVM, support vectors satisfy: y_i(w^T x_i + b) = 1
# We have 3 support vectors, so we get 3 equations:
# y_1(w^T x_1 + b) = 1
# y_3(w^T x_3 + b) = 1  
# y_4(w^T x_4 + b) = 1

# Let's solve this system of equations
# Point 1 (2,2) with y=-1: -1 * (w1*2 + w2*2 + b) = 1
# Point 3 (4,2) with y=1:  1 * (w1*4 + w2*2 + b) = 1
# Point 4 (3,3) with y=1:  1 * (w1*3 + w2*3 + b) = 1

# This gives us:
# -2w1 - 2w2 - b = 1  (equation 1)
#  4w1 + 2w2 + b = 1  (equation 2)
#  3w1 + 3w2 + b = 1  (equation 3)

# Adding equations 1 and 2: 2w1 = 2, so w1 = 1
# Adding equations 1 and 3: w1 + w2 = 2, so w2 = 1
# Substituting into equation 2: 4*1 + 2*1 + b = 1, so b = -5

w_optimal = np.array([1, 1])
b_optimal = -5

print("Solving the system of equations for support vectors:")
print("For support vectors, y_i(w^T x_i + b) = 1")
print()
print("Point 1 (2,2) with y=-1: -1 * (w1*2 + w2*2 + b) = 1")
print("Point 3 (4,2) with y=1:  1 * (w1*4 + w2*2 + b) = 1")
print("Point 4 (3,3) with y=1:  1 * (w1*3 + w2*3 + b) = 1")
print()
print("This gives us:")
print("-2w1 - 2w2 - b = 1  (equation 1)")
print(" 4w1 + 2w2 + b = 1  (equation 2)")
print(" 3w1 + 3w2 + b = 1  (equation 3)")
print()
print("Solving:")
print("Adding equations 1 and 2: 2w1 = 2, so w1 = 1")
print("Adding equations 1 and 3: w1 + w2 = 2, so w2 = 1")
print("Substituting into equation 2: 4*1 + 2*1 + b = 1, so b = -5")
print()
print(f"Optimal weight vector: w = [{w_optimal[0]}, {w_optimal[1]}]^T")
print(f"Optimal bias: b = {b_optimal}")

# Verify the solution
print("\nVerifying the solution:")
for i, (point, label) in enumerate(zip(support_vectors, support_vectors_labels)):
    activation = np.dot(w_optimal, point) + b_optimal
    margin = label * activation
    print(f"Point {support_vectors_idx[i]+1} ({point}): y_i(w^T x_i + b) = {label} * {activation:.1f} = {margin:.1f}")

# Step 3: Visualize the decision boundary
print("\n" + "=" * 50)
print("STEP 3: DECISION BOUNDARY VISUALIZATION")
print("=" * 50)

def plot_decision_boundary(X, y, w, b, support_vectors_idx, title, filename):
    """Plot SVM decision boundary with support vectors highlighted"""
    plt.figure(figsize=(12, 10))
    
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Compute decision function
    Z = w[0] * xx + w[1] * yy + b
    Z = np.sign(Z)
    
    # Plot decision boundary and regions
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=3)
    
    # Plot margin lines
    margin_positive = w[0] * xx + w[1] * yy + b - 1
    margin_negative = w[0] * xx + w[1] * yy + b + 1
    plt.contour(xx, yy, margin_positive, levels=[0], colors='blue', linewidths=2, linestyles='--', alpha=0.8)
    plt.contour(xx, yy, margin_negative, levels=[0], colors='red', linewidths=2, linestyles='--', alpha=0.8)
    
    # Plot all points
    for i, (point, label) in enumerate(zip(X, y)):
        if i in support_vectors_idx:
            # Support vectors
            plt.scatter(point[0], point[1], s=300, c='red' if label == 1 else 'blue',
                       marker='s', edgecolors='black', linewidth=3, 
                       label=f'Point {i+1} (SV)' if i == support_vectors_idx[0] else "")
        else:
            # Non-support vectors
            plt.scatter(point[0], point[1], s=150, c='red' if label == 1 else 'blue',
                       marker='o', alpha=0.7, 
                       label=f'Point {i+1}' if i == 0 or (i > 0 and i-1 not in support_vectors_idx) else "")
    
    # Add weight vector arrow
    center_x, center_y = np.mean(X, axis=0)
    plt.arrow(center_x, center_y, w[0], w[1], head_width=0.3, head_length=0.3, 
              fc='green', ec='green', linewidth=3, label='Weight Vector')
    
    plt.xlabel(r'$x_1$', fontsize=14)
    plt.ylabel(r'$x_2$', fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')
    plt.xlim(-1, 6)
    plt.ylim(1, 4)
    
    # Add equation
    eq_text = r'Decision Boundary: $' + f'{w[0]:.1f}x_1 + {w[1]:.1f}x_2 + {b:.1f} = 0$'
    plt.text(0.05, 0.95, eq_text, transform=plt.gca().transAxes, fontsize=14,
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", lw=2))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()

plot_decision_boundary(X, y, w_optimal, b_optimal, support_vectors_idx,
                      r'Optimal Hard-Margin SVM with Support Vectors',
                      'step3_decision_boundary.png')

# Step 4: Calculate training set error
print("\n" + "=" * 50)
print("STEP 4: CALCULATING TRAINING SET ERROR")
print("=" * 50)

def predict_svm(x, w, b):
    """Predict class using SVM decision function"""
    activation = np.dot(w, x) + b
    return np.sign(activation)

# Predict all training points
predictions = []
activations = []
for point in X:
    activation = np.dot(w_optimal, point) + b_optimal
    pred = predict_svm(point, w_optimal, b_optimal)
    predictions.append(pred)
    activations.append(activation)

predictions = np.array(predictions)
activations = np.array(activations)

print("Predictions for all training points:")
print("Point | True Label | Activation | Prediction | Correct?")
print("-" * 55)
correct_count = 0
for i, (point, true_label, activation, pred) in enumerate(zip(X, y, activations, predictions)):
    correct = (true_label == pred)
    if correct:
        correct_count += 1
    print(f"  {i+1}   |     {true_label}     |   {activation:6.1f}   |     {pred}     |   {correct}")

training_accuracy = correct_count / len(y)
training_error = 1 - training_accuracy
print(f"\nTraining Accuracy: {training_accuracy:.3f} ({training_accuracy*100:.1f}%)")
print(f"Training Error: {training_error:.3f} ({training_error*100:.1f}%)")

# Step 5: Visualize activation values
print("\n" + "=" * 50)
print("STEP 5: ACTIVATION VALUES VISUALIZATION")
print("=" * 50)

def plot_activation_values(X, y, activations, support_vectors_idx, title, filename):
    """Plot activation values for all points"""
    plt.figure(figsize=(12, 8))
    
    # Create bar plot
    point_labels = [f'Point {i+1}' for i in range(len(X))]
    colors = ['red' if label == 1 else 'blue' for label in y]
    
    bars = plt.bar(point_labels, activations, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Highlight support vectors
    for i in support_vectors_idx:
        bars[i].set_alpha(1.0)
        bars[i].set_linewidth(2)
    
    # Add reference lines
    plt.axhline(y=0, color='black', linestyle='-', linewidth=2, label='Decision Boundary')
    plt.axhline(y=1, color='blue', linestyle='--', alpha=0.8, linewidth=2, label='Positive Margin')
    plt.axhline(y=-1, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Negative Margin')
    
    plt.xlabel('Training Points', fontsize=14)
    plt.ylabel(r'Activation: $\mathbf{w}^T\mathbf{x} + b$', fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()
    
    # Add value labels on bars
    for i, (bar, activation) in enumerate(zip(bars, activations)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{activation:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()

plot_activation_values(X, y, activations, support_vectors_idx,
                      r'Activation Values for All Training Points',
                      'step5_activation_values.png')

# Step 6: Leave-One-Out Cross-Validation
print("\n" + "=" * 50)
print("STEP 6: LEAVE-ONE-OUT CROSS-VALIDATION")
print("=" * 50)

def train_svm_without_point(X_train, y_train):
    """Train SVM without a specific point and return w, b"""
    # For hard-margin SVM, we need to find the optimal hyperplane
    # We'll use the dual formulation and solve for the support vectors
    
    # Create SVM model
    svm = SVC(kernel='linear', C=1000)  # High C for hard margin
    svm.fit(X_train, y_train)
    
    # Extract weight vector and bias
    w = svm.coef_[0]
    b = svm.intercept_[0]
    
    return w, b

# Perform LOOCV
loo = LeaveOneOut()
cv_predictions = []
cv_true_labels = []
cv_errors = []

print("LOOCV Results:")
print("Fold | Left Out Point | Test Prediction | True Label | Correct?")
print("-" * 60)

for fold, (train_idx, test_idx) in enumerate(loo.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Train SVM without the left-out point
    w_cv, b_cv = train_svm_without_point(X_train, y_train)
    
    # Predict the left-out point
    pred = predict_svm(X_test[0], w_cv, b_cv)
    
    correct = (pred == y_test[0])
    cv_predictions.append(pred)
    cv_true_labels.append(y_test[0])
    cv_errors.append(not correct)
    
    print(f" {fold+1}  |       {test_idx[0]+1}       |       {pred}       |     {y_test[0]}     |   {correct}")

cv_accuracy = accuracy_score(cv_true_labels, cv_predictions)
cv_error = 1 - cv_accuracy
print(f"\nLOOCV Accuracy: {cv_accuracy:.3f} ({cv_accuracy*100:.1f}%)")
print(f"LOOCV Error: {cv_error:.3f} ({cv_error*100:.1f}%)")

# Step 7: Visualize LOOCV results
print("\n" + "=" * 50)
print("STEP 7: LOOCV RESULTS VISUALIZATION")
print("=" * 50)

def plot_loocv_results(X, y, cv_errors, title, filename):
    """Plot LOOCV results showing which points were misclassified"""
    plt.figure(figsize=(10, 8))
    
    # Plot all points
    for i, (point, label) in enumerate(zip(X, y)):
        if cv_errors[i]:
            # Misclassified points in LOOCV
            plt.scatter(point[0], point[1], s=300, c='orange',
                       marker='x', linewidth=3, label='LOOCV Error' if i == 0 else "")
        else:
            # Correctly classified points in LOOCV
            plt.scatter(point[0], point[1], s=200, c='red' if label == 1 else 'blue',
                       marker='o', alpha=0.7, label='LOOCV Correct' if i == 0 else "")
    
    plt.xlabel(r'$x_1$', fontsize=14)
    plt.ylabel(r'$x_2$', fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')
    plt.xlim(-1, 6)
    plt.ylim(1, 4)
    
    # Add point labels
    for i, point in enumerate(X):
        plt.annotate(f'{i+1}', (point[0], point[1]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()

plot_loocv_results(X, y, cv_errors,
                   r'LOOCV Results: Points with Cross-Validation Errors',
                   'step7_loocv_results.png')

# Step 8: Final comprehensive visualization
print("\n" + "=" * 50)
print("STEP 8: COMPREHENSIVE ANALYSIS")
print("=" * 50)

def plot_comprehensive_analysis(X, y, w, b, support_vectors_idx, activations, cv_errors, title, filename):
    """Create a comprehensive analysis plot with multiple subplots"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(title, fontsize=18, fontweight='bold')
    
    # Subplot 1: Dataset with support vectors
    ax1 = axes[0, 0]
    for i, (point, label) in enumerate(zip(X, y)):
        if i in support_vectors_idx:
            ax1.scatter(point[0], point[1], s=200, c='red' if label == 1 else 'blue',
                       marker='s', edgecolors='black', linewidth=2)
        else:
            ax1.scatter(point[0], point[1], s=100, c='red' if label == 1 else 'blue',
                       marker='o', alpha=0.7)
    ax1.set_xlabel(r'$x_1$')
    ax1.set_ylabel(r'$x_2$')
    ax1.set_title('Dataset with Support Vectors')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1, 6)
    ax1.set_ylim(1, 4)
    
    # Subplot 2: Decision boundary
    ax2 = axes[0, 1]
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = w[0] * xx + w[1] * yy + b
    Z = np.sign(Z)
    ax2.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax2.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
    
    for i, (point, label) in enumerate(zip(X, y)):
        if i in support_vectors_idx:
            ax2.scatter(point[0], point[1], s=200, c='red' if label == 1 else 'blue',
                       marker='s', edgecolors='black', linewidth=2)
        else:
            ax2.scatter(point[0], point[1], s=100, c='red' if label == 1 else 'blue',
                       marker='o', alpha=0.7)
    ax2.set_xlabel(r'$x_1$')
    ax2.set_ylabel(r'$x_2$')
    ax2.set_title('Decision Boundary')
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Margin visualization
    ax3 = axes[0, 2]
    margin_positive = w[0] * xx + w[1] * yy + b - 1
    margin_negative = w[0] * xx + w[1] * yy + b + 1
    ax3.contour(xx, yy, margin_positive, levels=[0], colors='blue', linewidths=2, linestyles='--')
    ax3.contour(xx, yy, margin_negative, levels=[0], colors='red', linewidths=2, linestyles='--')
    ax3.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
    
    for i, (point, label) in enumerate(zip(X, y)):
        if i in support_vectors_idx:
            ax3.scatter(point[0], point[1], s=200, c='red' if label == 1 else 'blue',
                       marker='s', edgecolors='black', linewidth=2)
        else:
            ax3.scatter(point[0], point[1], s=100, c='red' if label == 1 else 'blue',
                       marker='o', alpha=0.7)
    ax3.set_xlabel(r'$x_1$')
    ax3.set_ylabel(r'$x_2$')
    ax3.set_title('Margin Lines')
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Weight vector
    ax4 = axes[1, 0]
    center_x, center_y = np.mean(X, axis=0)
    ax4.arrow(center_x, center_y, w[0], w[1], head_width=0.2, head_length=0.2, 
              fc='green', ec='green', linewidth=3)
    
    for i, (point, label) in enumerate(zip(X, y)):
        if i in support_vectors_idx:
            ax4.scatter(point[0], point[1], s=200, c='red' if label == 1 else 'blue',
                       marker='s', edgecolors='black', linewidth=2)
        else:
            ax4.scatter(point[0], point[1], s=100, c='red' if label == 1 else 'blue',
                       marker='o', alpha=0.7)
    ax4.set_xlabel(r'$x_1$')
    ax4.set_ylabel(r'$x_2$')
    ax4.set_title('Weight Vector')
    ax4.grid(True, alpha=0.3)
    
    # Subplot 5: Activation values
    ax5 = axes[1, 1]
    point_labels = [f'P{i+1}' for i in range(len(X))]
    colors = ['red' if label == 1 else 'blue' for label in y]
    bars = ax5.bar(point_labels, activations, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Highlight support vectors
    for i in support_vectors_idx:
        bars[i].set_alpha(1.0)
        bars[i].set_linewidth(2)
    
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax5.axhline(y=1, color='blue', linestyle='--', alpha=0.7)
    ax5.axhline(y=-1, color='red', linestyle='--', alpha=0.7)
    ax5.set_xlabel('Point ID')
    ax5.set_ylabel(r'Activation: $\mathbf{w}^T\mathbf{x} + b$')
    ax5.set_title('Activation Values')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Subplot 6: Summary statistics
    ax6 = axes[1, 2]
    ax6.axis('off')
    summary_text = f"""
SVM Solution Summary:

Weight Vector: $\\mathbf{{w}} = [{w[0]}, {w[1]}]^T$
Bias: $b = {b}$

Support Vectors: Points {[i+1 for i in support_vectors_idx]}

Training Error: {training_error*100:.1f}%
LOOCV Error: {cv_error*100:.1f}%

Decision Boundary:
${w[0]}x_1 + {w[1]}x_2 + {b} = 0$
"""
    ax6.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", lw=1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()

plot_comprehensive_analysis(X, y, w_optimal, b_optimal, support_vectors_idx, 
                           activations, cv_errors,
                           r'Comprehensive SVM Analysis',
                           'step8_comprehensive_analysis.png')

# Print final results
print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)
print(f"1. Optimal weight vector: w = [{w_optimal[0]}, {w_optimal[1]}]^T")
print(f"   Optimal bias: b = {b_optimal}")
print(f"   Decision boundary: {w_optimal[0]}x1 + {w_optimal[1]}x2 + {b_optimal} = 0")
print()
print(f"2. Training set error: {training_error*100:.1f}%")
print()
print(f"3. LOOCV error: {cv_error*100:.1f}%")
print()
print(f"Plots saved to: {save_dir}")
