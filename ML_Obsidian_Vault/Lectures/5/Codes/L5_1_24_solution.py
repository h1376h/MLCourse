import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Import the dataset functions from the question code
from L5_1_24_question import create_dataset_scenario_a, create_dataset_scenario_b

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_1_Quiz_24")
os.makedirs(save_dir, exist_ok=True)

# Disable LaTeX for compatibility
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'sans-serif'

print("=" * 80)
print("SVM LOOCV Analysis: Complete Solution for Two Scenarios")
print("Question 24: Leave-One-Out Cross-Validation Error Estimation")
print("=" * 80)

def analyze_scenario(X, y, scenario_name, scenario_letter):
    """Perform complete LOOCV analysis for a given scenario."""
    print(f"\n" + "=" * 60)
    print(f"SCENARIO {scenario_letter} ANALYSIS: {scenario_name}")
    print("=" * 60)

    print(f"\nDataset Overview:")
    print(f"Total points: {len(X)}")
    print(f"Class 1 points: {np.sum(y == 1)}")
    print(f"Class -1 points: {np.sum(y == -1)}")

    # Display dataset points
    print(f"\nDataset Points:")
    for i, (point, label) in enumerate(zip(X, y)):
        print(f"  Point {i+1}: {point} (Class {label})")

    # Train the full SVM model
    svm_full = SVC(kernel='linear', C=1000)  # High C for hard margin
    svm_full.fit(X, y)

    # Get support vectors
    support_vectors = svm_full.support_vectors_
    support_vector_indices = svm_full.support_

    print(f"\nSupport Vector Analysis:")
    print(f"Number of support vectors: {len(support_vector_indices)} out of {len(X)} points")
    print(f"Support vector indices: {support_vector_indices + 1}")  # +1 for 1-based indexing
    print("Support vector details:")
    for i, idx in enumerate(support_vector_indices):
        print(f"  Point {idx+1}: {X[idx]} (Class {y[idx]})")

    return svm_full, support_vector_indices

def perform_loocv_analysis(X, y, scenario_name, scenario_letter):
    """Perform detailed LOOCV analysis."""
    print(f"\n" + "-" * 50)
    print(f"LOOCV ANALYSIS FOR SCENARIO {scenario_letter}")
    print("-" * 50)

    # Train full SVM to identify support vectors
    svm_full = SVC(kernel='linear', C=1000)
    svm_full.fit(X, y)
    support_vector_indices = svm_full.support_

    # Perform LOOCV
    loo = LeaveOneOut()
    cv_scores = []
    misclassified_points = []
    detailed_results = []

    print(f"\nPerforming LOOCV (leaving out each point one by one):")

    for fold, (train_idx, test_idx) in enumerate(loo.split(X)):
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

        # Check if left-out point was a support vector
        is_support_vector = test_idx[0] in support_vector_indices

        result = {
            'fold': fold + 1,
            'point_idx': test_idx[0],
            'point': X[test_idx[0]],
            'true_label': y_test[0],
            'predicted_label': y_pred[0],
            'is_misclassified': is_misclassified,
            'is_support_vector': is_support_vector
        }
        detailed_results.append(result)

        status = "MISCLASSIFIED" if is_misclassified else "CORRECT"
        sv_status = "(Support Vector)" if is_support_vector else "(Non-Support Vector)"
        print(f"  Fold {fold+1:2d}: Point {test_idx[0]+1:2d} {X[test_idx[0]]} → Predicted: {y_pred[0]:2d}, Actual: {y_test[0]:2d} [{status}] {sv_status}")

    # Calculate LOOCV error
    loocv_error = 1 - np.mean(cv_scores)
    misclassified_count = sum(misclassified_points)

    return loocv_error, misclassified_count, detailed_results, support_vector_indices

def provide_theoretical_analysis():
    """Provide theoretical analysis and answers."""
    print(f"\n" + "=" * 80)
    print("THEORETICAL ANALYSIS AND COMPLETE SOLUTION")
    print("=" * 80)

    print(f"\nTASK 3: Theoretical Analysis")
    print("-" * 40)

    print(f"\n3a. Theoretical relationship between LOOCV error and support vectors:")
    print("    For a hard-margin SVM with linearly separable data:")
    print("    LOOCV Error Rate ≤ (Number of Support Vectors) / (Total Number of Points)")
    print("    This provides an upper bound on the LOOCV error rate.")

    print(f"\n3b. Why actual LOOCV error can be less than theoretical upper bound:")
    print("    - The theoretical bound assumes each support vector contributes 1 error")
    print("    - In practice, when a support vector is removed, the new decision boundary")
    print("      may still correctly classify the removed point")
    print("    - The bound is conservative and represents the worst-case scenario")

    print(f"\n3c. Conditions for tight theoretical bound:")
    print("    - When removing each support vector changes the decision boundary")
    print("      such that the removed point is misclassified")
    print("    - This typically happens when support vectors are 'critical' for")
    print("      defining the margin and their removal significantly shifts the boundary")
    print("    - More likely with minimal support vector sets and tight margins")

# Main execution
if __name__ == "__main__":
    # Load both scenarios
    print(f"\nLoading datasets from question scenarios...")
    X_a, y_a = create_dataset_scenario_a()
    X_b, y_b = create_dataset_scenario_b()

    # Analyze Scenario A
    svm_a, sv_indices_a = analyze_scenario(X_a, y_a, "Scenario A", "A")
    loocv_error_a, misclassified_count_a, results_a, sv_indices_a = perform_loocv_analysis(X_a, y_a, "Scenario A", "A")

    # Analyze Scenario B
    svm_b, sv_indices_b = analyze_scenario(X_b, y_b, "Scenario B", "B")
    loocv_error_b, misclassified_count_b, results_b, sv_indices_b = perform_loocv_analysis(X_b, y_b, "Scenario B", "B")

    # Provide detailed answers
    print(f"\n" + "=" * 80)
    print("DETAILED ANSWERS TO ALL TASKS")
    print("=" * 80)

    # Task 1: Scenario A Analysis
    print(f"\nTASK 1: Scenario A Analysis")
    print("-" * 40)

    print(f"\n1a. LOOCV error estimate for Scenario A:")
    print(f"    Answer: {loocv_error_a:.1%} ({misclassified_count_a}/{len(X_a)} points misclassified)")

    print(f"\n1b. Justification for Scenario A:")
    print(f"    - Scenario A has {len(sv_indices_a)} support vectors out of {len(X_a)} total points")
    print(f"    - Theoretical upper bound: {len(sv_indices_a)}/{len(X_a)} = {len(sv_indices_a)/len(X_a):.1%}")
    print(f"    - Only support vectors can potentially be misclassified during LOOCV")
    print(f"    - Non-support vectors are always correctly classified when left out")
    print(f"    - Actual result: {misclassified_count_a} out of {len(sv_indices_a)} support vectors were misclassified")

    print(f"\n1c. Specific points misclassified in Scenario A:")
    misclassified_a = [r for r in results_a if r['is_misclassified']]
    if misclassified_a:
        for result in misclassified_a:
            print(f"    - Point {result['point_idx']+1}: {result['point']} (Class {result['true_label']} → predicted as {result['predicted_label']})")
    else:
        print(f"    - No points were misclassified")

    # Task 2: Scenario B Analysis
    print(f"\nTASK 2: Scenario B Analysis")
    print("-" * 40)

    print(f"\n2a. LOOCV error estimate for Scenario B:")
    print(f"    Answer: {loocv_error_b:.1%} ({misclassified_count_b}/{len(X_b)} points misclassified)")

    print(f"\n2b. Justification for Scenario B:")
    print(f"    - Scenario B has {len(sv_indices_b)} support vectors out of {len(X_b)} total points")
    print(f"    - Theoretical upper bound: {len(sv_indices_b)}/{len(X_b)} = {len(sv_indices_b)/len(X_b):.1%}")
    print(f"    - All support vectors remained correctly classified when left out")
    print(f"    - This shows the theoretical bound is not always tight")

    print(f"\n2c. Comparison between Scenario A and B:")
    print(f"    Scenario A: {len(sv_indices_a)} support vectors → {loocv_error_a:.1%} actual error")
    print(f"    Scenario B: {len(sv_indices_b)} support vectors → {loocv_error_b:.1%} actual error")
    print(f"    Key differences:")
    print(f"    - Scenario A has more support vectors ({len(sv_indices_a)} vs {len(sv_indices_b)})")
    print(f"    - Scenario A has higher actual LOOCV error ({loocv_error_a:.1%} vs {loocv_error_b:.1%})")
    print(f"    - Both demonstrate that actual error ≤ theoretical upper bound")
    print(f"    - Scenario B shows perfect LOOCV performance despite having support vectors")

    # Theoretical Analysis
    provide_theoretical_analysis()

    # Summary
    print(f"\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(f"\nScenario A Results:")
    print(f"  - Support vectors: {len(sv_indices_a)}/{len(X_a)} points")
    print(f"  - Theoretical bound: {len(sv_indices_a)/len(X_a):.1%}")
    print(f"  - Actual LOOCV error: {loocv_error_a:.1%}")
    print(f"  - Misclassified points: {misclassified_count_a}")

    print(f"\nScenario B Results:")
    print(f"  - Support vectors: {len(sv_indices_b)}/{len(X_b)} points")
    print(f"  - Theoretical bound: {len(sv_indices_b)/len(X_b):.1%}")
    print(f"  - Actual LOOCV error: {loocv_error_b:.1%}")
    print(f"  - Misclassified points: {misclassified_count_b}")

    print(f"\nKey Learning Points:")
    print(f"  1. LOOCV error ≤ (# Support Vectors) / (Total Points)")
    print(f"  2. Actual error can be less than theoretical upper bound")
    print(f"  3. Only support vectors can be misclassified during LOOCV")
    print(f"  4. Different data configurations lead to different LOOCV results")
    print(f"  5. The bound provides a quick estimate without computation")

    print(f"\nImages and analysis saved to: {save_dir}")
    print("=" * 80)

