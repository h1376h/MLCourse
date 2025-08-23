import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_1_Quiz_24")
os.makedirs(save_dir, exist_ok=True)

def create_dataset_scenario_a():
    """Create the first dataset scenario - matches the uploaded image pattern."""
    # Create linearly separable data that matches the expected visualization
    # Class 1 (x markers) - positioned in upper right region
    class_1 = np.array([
        [2.5, 2.0],  # Point 1
        [3.0, 1.5],  # Point 2 (support vector)
        [3.5, 2.5],  # Point 3
        [4.0, 2.0],  # Point 4
        [2.0, 2.5]   # Point 5
    ])

    # Class -1 (o markers) - positioned in lower left region
    class_neg1 = np.array([
        [0.5, 1.0],  # Point 6
        [1.0, 0.5],  # Point 7
        [1.5, 1.5],  # Point 8 (support vector)
        [0.0, 0.5],  # Point 9
        [2.0, 0.0]   # Point 10 (support vector)
    ])

    # Combine data
    X = np.vstack([class_1, class_neg1])
    y = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])

    return X, y

def create_dataset_scenario_b():
    """Create the second dataset scenario - configuration with many support vectors."""
    # Create a narrow margin scenario where many points become support vectors
    # Position classes very close to each other to maximize support vectors

    # Class 1 (x markers) - positioned close to the decision boundary
    class_1 = np.array([
        [2.2, 2.2],  # Point 1 (likely support vector)
        [2.0, 2.5],  # Point 2 (likely support vector)
        [2.5, 2.0],  # Point 3 (likely support vector)
        [2.8, 2.8],  # Point 4 (likely support vector)
        [3.5, 3.5]   # Point 5 (farther away)
    ])

    # Class -1 (o markers) - positioned close to create narrow margin
    class_neg1 = np.array([
        [1.8, 1.8],  # Point 6 (likely support vector)
        [1.5, 2.0],  # Point 7 (likely support vector)
        [2.0, 1.5],  # Point 8 (likely support vector)
        [1.2, 1.2],  # Point 9 (likely support vector)
        [0.5, 0.5]   # Point 10 (farther away)
    ])

    # Combine data
    X = np.vstack([class_1, class_neg1])
    y = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])

    return X, y

def plot_svm_decision_boundary(X, y, svm_model, ax, title="SVM Decision Boundary"):
    """Plot SVM decision boundary and margins."""
    # Create a mesh for plotting decision boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Get decision function values
    Z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary and margins
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['gray', 'black', 'gray'],
               linestyles=['--', '-', '--'], linewidths=[2, 3, 2])

    # Plot data points to match the uploaded image style
    class_1_mask = y == 1
    class_neg1_mask = y == -1

    # Class 1 with 'x' markers (red)
    ax.scatter(X[class_1_mask, 0], X[class_1_mask, 1], c='red', marker='x',
               s=150, linewidth=3, label="Class 1", zorder=5)
    # Class -1 with 'o' markers (blue, hollow circles)
    ax.scatter(X[class_neg1_mask, 0], X[class_neg1_mask, 1], marker='o',
               s=100, facecolors='none', edgecolors='blue', linewidth=2,
               label="Class -1", zorder=5)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

def create_scenario_a_visualization():
    """Create the first scenario visualization that matches the uploaded image style."""
    X, y = create_dataset_scenario_a()

    # Train SVM
    svm = SVC(kernel='linear', C=1000)  # Large C for hard margin
    svm.fit(X, y)

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_svm_decision_boundary(X, y, svm, ax, "Scenario A: SVM with Maximum Margin Decision Boundary")

    # Highlight support vectors with circles (like in the uploaded image)
    support_vectors = svm.support_vectors_
    ax.scatter(support_vectors[:, 0], support_vectors[:, 1],
               s=300, facecolors='none', edgecolors='green', linewidth=3,
               label='Support Vectors', zorder=6)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'scenario_a_svm_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Scenario A SVM visualization saved")
    print(f"Support vectors: {len(svm.support_)} out of {len(X)} points")
    return X, y, svm

def create_scenario_b_visualization():
    """Create the second scenario visualization with different configuration."""
    X, y = create_dataset_scenario_b()

    # Train SVM
    svm = SVC(kernel='linear', C=1000)  # Large C for hard margin
    svm.fit(X, y)

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_svm_decision_boundary(X, y, svm, ax, "Scenario B: SVM with Different Configuration")

    # Highlight support vectors with circles
    support_vectors = svm.support_vectors_
    ax.scatter(support_vectors[:, 0], support_vectors[:, 1],
               s=300, facecolors='none', edgecolors='green', linewidth=3,
               label='Support Vectors', zorder=6)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'scenario_b_svm_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Scenario B SVM visualization saved")
    print(f"Support vectors: {len(svm.support_)} out of {len(X)} points")
    return X, y, svm

def perform_loocv_analysis(X, y, scenario_name):
    """Perform LOOCV analysis and return the error rate."""
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import accuracy_score

    # Train full SVM to identify support vectors
    svm_full = SVC(kernel='linear', C=1000)
    svm_full.fit(X, y)
    support_vector_indices = svm_full.support_

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

    # Calculate LOOCV error
    loocv_error = 1 - np.mean(cv_scores)
    misclassified_count = sum(misclassified_points)

    print(f"\n{scenario_name} LOOCV Analysis:")
    print(f"Support vectors: {len(support_vector_indices)} out of {len(X)} points")
    print(f"Theoretical upper bound: {len(support_vector_indices)}/{len(X)} = {len(support_vector_indices)/len(X):.1%}")
    print(f"Actual LOOCV error: {misclassified_count}/{len(X)} = {loocv_error:.1%}")

    return loocv_error, len(support_vector_indices)

def analyze_both_scenarios():
    """Analyze both scenarios and return their data."""
    # Load datasets
    X_a, y_a = create_dataset_scenario_a()
    X_b, y_b = create_dataset_scenario_b()

    # Train SVMs
    svm_a = SVC(kernel='linear', C=1000)
    svm_a.fit(X_a, y_a)
    svm_b = SVC(kernel='linear', C=1000)
    svm_b.fit(X_b, y_b)

    # Perform LOOCV analysis for both scenarios
    loocv_a, sv_count_a = perform_loocv_analysis(X_a, y_a, "Scenario A")
    loocv_b, sv_count_b = perform_loocv_analysis(X_b, y_b, "Scenario B")

    return (X_a, y_a, svm_a, loocv_a, sv_count_a), (X_b, y_b, svm_b, loocv_b, sv_count_b)

def create_main_svm_visualization():
    """Create the main SVM visualization for backward compatibility."""
    # Use scenario A as the main visualization (matches uploaded image style)
    return create_scenario_a_visualization()

if __name__ == "__main__":
    print("=== SVM LOOCV Analysis: Two-Scenario Question ===")
    print("Question 24: Leave-One-Out Cross-Validation Error Estimation")
    print("=" * 60)

    # Create individual scenario visualizations
    print("\n1. Creating Scenario A visualization...")
    create_scenario_a_visualization()

    print("\n2. Creating Scenario B visualization...")
    create_scenario_b_visualization()

    print("\n3. Analyzing both scenarios...")
    scenario_a_data, scenario_b_data = analyze_both_scenarios()

    print(f"\n" + "=" * 60)
    print("QUESTION STATEMENT")
    print("=" * 60)

    print("\nConsider the two SVM scenarios shown in the figures below:")
    print("- Scenario A: scenario_a_svm_visualization.png")
    print("- Scenario B: scenario_b_svm_visualization.png")

    print(f"\nBoth scenarios show linearly separable datasets with maximum margin")
    print(f"decision boundaries. The support vectors are highlighted with green circles.")

    print(f"\n" + "-" * 40)
    print("TASK 1: Scenario A Analysis")
    print("-" * 40)
    print(f"For Scenario A (with {scenario_a_data[4]} support vectors out of 10 points):")
    print("1a. What is the leave-one-out cross-validation (LOOCV) error estimate?")
    print("1b. Provide a brief justification for your answer.")
    print("1c. Which specific points (if any) would be misclassified during LOOCV?")

    print(f"\n" + "-" * 40)
    print("TASK 2: Scenario B Analysis")
    print("-" * 40)
    print(f"For Scenario B (with {scenario_b_data[4]} support vectors out of 10 points):")
    print("2a. What is the leave-one-out cross-validation (LOOCV) error estimate?")
    print("2b. Provide a brief justification for your answer.")
    print("2c. Compare and contrast the results with Scenario A.")

    print(f"\n" + "-" * 40)
    print("TASK 3: Theoretical Analysis")
    print("-" * 40)
    print("3a. State the theoretical relationship between LOOCV error and support vectors.")
    print("3b. Explain why the actual LOOCV error can be less than the theoretical upper bound.")
    print("3c. Under what conditions would the theoretical bound be tight (exact)?")

    print(f"\n" + "=" * 60)
    print("SCENARIO SUMMARY")
    print("=" * 60)
    print(f"Scenario A: {scenario_a_data[4]} support vectors → Theoretical bound: {scenario_a_data[4]/10:.1%}, Actual: {scenario_a_data[3]:.1%}")
    print(f"Scenario B: {scenario_b_data[4]} support vectors → Theoretical bound: {scenario_b_data[4]/10:.1%}, Actual: {scenario_b_data[3]:.1%}")

    print(f"\nImages saved to: {save_dir}")
    print(f"Solution code: L5_1_24_solution.py")
    print(f"Explanation: ../Quiz/L5_1_24_explanation.md")
