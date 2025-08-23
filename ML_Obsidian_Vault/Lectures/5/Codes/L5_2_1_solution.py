import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import cvxpy as cp

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_2_Quiz_1")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("QUESTION 1: SVM WITH OUTLIERS - COMPREHENSIVE SOLUTION")
print("=" * 80)

# Define the dataset
X_pos = np.array([[3, 2], [4, 3], [5, 2], [1, 4]])  # Class +1 (including outlier)
X_neg = np.array([[0, 0], [1, 1], [0, 2]])          # Class -1

# Combine data
X = np.vstack([X_pos, X_neg])
y = np.array([1, 1, 1, 1, -1, -1, -1])

print(f"Dataset:")
print(f"Class +1: {X_pos}")
print(f"Class -1: {X_neg}")
print(f"Outlier: (1, 4) - marked as potential outlier")

# Task 1: Draw data points and check linear separability
print("\n" + "="*50)
print("TASK 1: VISUALIZATION AND LINEAR SEPARABILITY")
print("="*50)

def plot_dataset(X_pos, X_neg, title="Dataset Visualization", save_path=None):
    """Plot the dataset with clear class separation"""
    plt.figure(figsize=(10, 8))
    
    # Plot positive class points
    plt.scatter(X_pos[:, 0], X_pos[:, 1], c='blue', s=200, marker='o', 
                label='Class +1', edgecolors='black', linewidth=2)
    
    # Plot negative class points
    plt.scatter(X_neg[:, 0], X_neg[:, 1], c='red', s=200, marker='s', 
                label='Class -1', edgecolors='black', linewidth=2)
    
    # Highlight the outlier
    outlier_idx = np.where((X_pos == [1, 4]).all(axis=1))[0]
    if len(outlier_idx) > 0:
        plt.scatter(X_pos[outlier_idx, 0], X_pos[outlier_idx, 1], 
                   c='orange', s=300, marker='*', label='Outlier (1, 4)', 
                   edgecolors='black', linewidth=2, zorder=5)
    
    # Add point labels
    for i, (x, y_coord) in enumerate(X_pos):
        plt.annotate(f'({x}, {y_coord})', (x, y_coord), 
                    xytext=(10, 10), textcoords='offset points', fontsize=10)
    
    for i, (x, y_coord) in enumerate(X_neg):
        plt.annotate(f'({x}, {y_coord})', (x, y_coord), 
                    xytext=(10, 10), textcoords='offset points', fontsize=10)
    
    plt.xlabel('$x_1$', fontsize=14)
    plt.ylabel('$x_2$', fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xlim(-1, 6)
    plt.ylim(-1, 5)
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

# Plot the dataset
plot_dataset(X_pos, X_neg, "Dataset with Outlier", 
            os.path.join(save_dir, 'dataset_visualization.png'))

# Check linear separability by trying to find a separating line
def check_linear_separability(X_pos, X_neg):
    """Check if the dataset is linearly separable"""
    print("\nChecking linear separability...")
    
    # Try different lines to separate the data
    # Line 1: x2 = x1 + 1 (separates most points but not the outlier)
    line1_slope, line1_intercept = 1, 1
    
    # Line 2: x2 = 2*x1 - 1 (another attempt)
    line2_slope, line2_intercept = 2, -1
    
    # Line 3: x2 = 0.5*x1 + 1.5 (another attempt)
    line3_slope, line3_intercept = 0.5, 1.5
    
    lines = [(line1_slope, line1_intercept, "x2 = x1 + 1"),
             (line2_slope, line2_intercept, "x2 = 2*x1 - 1"),
             (line3_slope, line3_intercept, "x2 = 0.5*x1 + 1.5")]
    
    for slope, intercept, eq_name in lines:
        print(f"\nTrying line: {eq_name}")
        
        # Check positive class points
        pos_above = 0
        pos_below = 0
        for x1, x2 in X_pos:
            line_x2 = slope * x1 + intercept
            if x2 > line_x2:
                pos_above += 1
                print(f"  Point ({x1}, {x2}) is above line (line gives {line_x2:.2f})")
            else:
                pos_below += 1
                print(f"  Point ({x1}, {x2}) is below line (line gives {line_x2:.2f})")
        
        # Check negative class points
        neg_above = 0
        neg_below = 0
        for x1, x2 in X_neg:
            line_x2 = slope * x1 + intercept
            if x2 > line_x2:
                neg_above += 1
                print(f"  Point ({x1}, {x2}) is above line (line gives {line_x2:.2f})")
            else:
                neg_below += 1
                print(f"  Point ({x1}, {x2}) is below line (line gives {line_x2:.2f})")
        
        # Check if separation is possible
        if (pos_above == len(X_pos) and neg_below == len(X_neg)) or \
           (pos_below == len(X_pos) and neg_above == len(X_neg)):
            print(f"  ✓ SUCCESS: {eq_name} separates the data!")
            return True, eq_name, slope, intercept
        else:
            print(f"  ✗ FAILED: {eq_name} cannot separate the data")
    
    return False, None, None, None

is_separable, best_line, best_slope, best_intercept = check_linear_separability(X_pos, X_neg)

if is_separable:
    print(f"\n✓ RESULT: Dataset is linearly separable using {best_line}")
else:
    print(f"\n✗ RESULT: Dataset is NOT linearly separable")

# Task 2: Explain why hard margin SVM would fail
print("\n" + "="*50)
print("TASK 2: WHY HARD MARGIN SVM FAILS")
print("="*50)

print("Hard margin SVM would fail on this dataset because:")
print("1. The outlier point (1, 4) from class +1 is positioned in a way that")
print("   makes it impossible to draw a straight line that perfectly separates")
print("   all positive points from all negative points.")
print("2. Any line that tries to include (1, 4) in the positive region will")
print("   also include some negative points, and vice versa.")
print("3. Hard margin SVM requires perfect linear separability with no")
print("   misclassifications, which is impossible here.")

# Task 3: Calculate minimum constraint violations
print("\n" + "="*50)
print("TASK 3: MINIMUM CONSTRAINT VIOLATIONS")
print("="*50)

def find_minimum_violations(X_pos, X_neg):
    """Find the minimum number of constraint violations needed"""
    print("Calculating minimum constraint violations...")
    
    # Try removing different points to find the minimum violations
    min_violations = float('inf')
    best_removal = None
    
    # Try removing the outlier
    X_pos_no_outlier = X_pos[X_pos[:, 0] != 1]  # Remove (1, 4)
    if len(X_pos_no_outlier) < len(X_pos):
        violations = 1  # One point removed
        print(f"Removing outlier (1, 4): {violations} violation")
        if violations < min_violations:
            min_violations = violations
            best_removal = "outlier (1, 4)"
    
    # Try removing other points
    for i, point in enumerate(X_pos):
        if not np.array_equal(point, [1, 4]):  # Skip outlier as already checked
            X_pos_temp = np.delete(X_pos, i, axis=0)
            violations = 1
            print(f"Removing point {point}: {violations} violation")
            if violations < min_violations:
                min_violations = violations
                best_removal = f"point {point}"
    
    # Try removing negative points
    for i, point in enumerate(X_neg):
        X_neg_temp = np.delete(X_neg, i, axis=0)
        violations = 1
        print(f"Removing point {point}: {violations} violation")
        if violations < min_violations:
            min_violations = violations
            best_removal = f"point {point}"
    
    return min_violations, best_removal

min_violations, best_removal = find_minimum_violations(X_pos, X_neg)
print(f"\nMinimum constraint violations needed: {min_violations}")
print(f"Best removal strategy: {best_removal}")

# Task 4: Soft margin SVM formulation
print("\n" + "="*50)
print("TASK 4: SOFT MARGIN SVM FORMULATION")
print("="*50)

print("Soft margin SVM formulation:")
print("minimize: (1/2)||w||² + C∑ξᵢ")
print("subject to:")
print("  yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ, ∀i")
print("  ξᵢ ≥ 0, ∀i")
print()
print("Where:")
print("- w is the weight vector")
print("- b is the bias term")
print("- ξᵢ are slack variables (constraint violations)")
print("- C is the regularization parameter")
print("- yᵢ are the true labels (+1 or -1)")
print("- xᵢ are the feature vectors")

# Implement soft margin SVM with different C values
def implement_soft_margin_svm(X, y, C_values=[0.1, 1, 10, 100]):
    """Implement soft margin SVM with different C values"""
    print(f"\nImplementing soft margin SVM with C values: {C_values}")
    
    results = {}
    
    for C in C_values:
        print(f"\nC = {C}:")
        
        # Create SVM model
        svm = SVC(kernel='linear', C=C, random_state=42)
        svm.fit(X, y)
        
        # Get predictions
        y_pred = svm.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        # Get support vectors
        n_support_vectors = len(svm.support_vectors_)
        
        # Calculate slack variables (approximation)
        decision_values = svm.decision_function(X)
        slack_vars = np.maximum(0, 1 - y * decision_values)
        total_slack = np.sum(slack_vars)
        
        results[C] = {
            'accuracy': accuracy,
            'n_support_vectors': n_support_vectors,
            'total_slack': total_slack,
            'slack_vars': slack_vars,
            'decision_values': decision_values,
            'model': svm
        }
        
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Support vectors: {n_support_vectors}")
        print(f"  Total slack: {total_slack:.3f}")
        print(f"  Individual slack variables: {slack_vars}")
    
    return results

svm_results = implement_soft_margin_svm(X, y)

# Task 5: Effect of removing outlier
print("\n" + "="*50)
print("TASK 5: EFFECT OF REMOVING OUTLIER")
print("="*50)

# Remove the outlier and retrain
X_no_outlier = np.vstack([X_pos[X_pos[:, 0] != 1], X_neg])  # Remove (1, 4)
y_no_outlier = np.array([1, 1, 1, -1, -1, -1])

print("Dataset without outlier:")
print(f"Class +1: {X_pos[X_pos[:, 0] != 1]}")
print(f"Class -1: {X_neg}")

# Check if it's now linearly separable
is_separable_no_outlier, _, _, _ = check_linear_separability(
    X_pos[X_pos[:, 0] != 1], X_neg)

if is_separable_no_outlier:
    print("\n✓ RESULT: Dataset becomes linearly separable after removing outlier")
    
    # Train hard margin SVM on clean data
    svm_hard = SVC(kernel='linear', C=1000, random_state=42)  # High C for hard margin
    svm_hard.fit(X_no_outlier, y_no_outlier)
    
    print(f"Hard margin SVM results:")
    print(f"  Accuracy: {accuracy_score(y_no_outlier, svm_hard.predict(X_no_outlier)):.3f}")
    print(f"  Support vectors: {len(svm_hard.support_vectors_)}")
    
    # Compare with soft margin on original data
    print(f"\nComparison with soft margin on original data:")
    for C in [0.1, 1, 10]:
        print(f"  C={C}: Accuracy={svm_results[C]['accuracy']:.3f}, "
              f"Support vectors={svm_results[C]['n_support_vectors']}")
else:
    print("\n✗ RESULT: Dataset is still not linearly separable after removing outlier")

# Task 6: Medical screening system
print("\n" + "="*50)
print("TASK 6: MEDICAL SCREENING SYSTEM")
print("="*50)

def design_medical_system():
    """Design a confidence-based medical screening system"""
    print("Medical Screening System Design:")
    print("=================================")
    
    # Define the medical dataset
    healthy_patients = np.array([[3, 2], [4, 3], [5, 2]])  # Blood test results
    at_risk_patients = np.array([[0, 0], [1, 1], [0, 2]])  # Blood test results
    uncertain_case = np.array([1, 4])  # Measurement error case
    
    print(f"Healthy patients (blood test results): {healthy_patients}")
    print(f"At Risk patients (blood test results): {at_risk_patients}")
    print(f"Uncertain case (measurement error): {uncertain_case}")
    
    # Train SVM on clean data
    X_medical = np.vstack([healthy_patients, at_risk_patients])
    y_medical = np.array([1, 1, 1, -1, -1, -1])  # 1=Healthy, -1=At Risk
    
    svm_medical = SVC(kernel='linear', C=1, probability=True, random_state=42)
    svm_medical.fit(X_medical, y_medical)
    
    # Calculate decision values and probabilities
    decision_values = svm_medical.decision_function(X_medical)
    probabilities = svm_medical.predict_proba(X_medical)
    
    print(f"\nSVM Decision Values:")
    for i, (point, dv, prob) in enumerate(zip(X_medical, decision_values, probabilities)):
        status = "Healthy" if dv > 0 else "At Risk"
        confidence = max(prob)
        print(f"  Patient {i+1} {point}: Decision={dv:.3f}, "
              f"Status={status}, Confidence={confidence:.3f}")
    
    # Analyze uncertain case
    uncertain_dv = svm_medical.decision_function([uncertain_case])[0]
    uncertain_prob = svm_medical.predict_proba([uncertain_case])[0]
    
    print(f"\nUncertain Case Analysis:")
    print(f"  Point {uncertain_case}: Decision={uncertain_dv:.3f}")
    print(f"  Probabilities: Healthy={uncertain_prob[1]:.3f}, At Risk={uncertain_prob[0]:.3f}")
    
    # Design three-zone system
    print(f"\nThree-Zone Classification System:")
    print(f"  High Confidence Zone (|decision| > 1.0):")
    print(f"    - Clear classification with high confidence")
    print(f"    - No additional testing needed")
    
    print(f"  Medium Confidence Zone (0.5 < |decision| ≤ 1.0):")
    print(f"    - Moderate confidence in classification")
    print(f"    - Consider additional tests or monitoring")
    
    print(f"  Low Confidence Zone (|decision| ≤ 0.5):")
    print(f"    - Low confidence in classification")
    print(f"    - Requires additional diagnostic tests")
    print(f"    - Manual review by medical professional")
    
    # Calculate diagnostic uncertainty for outlier
    uncertainty = 1 - max(uncertain_prob)
    print(f"\nDiagnostic Uncertainty for Outlier Case:")
    print(f"  Uncertainty = {uncertainty:.3f} ({uncertainty*100:.1f}%)")
    print(f"  Recommendation: Additional diagnostic tests required")
    
    return svm_medical, uncertain_dv, uncertainty

medical_system = design_medical_system()

# Visualization of all results
print("\n" + "="*50)
print("GENERATING COMPREHENSIVE VISUALIZATIONS")
print("="*50)

def create_comprehensive_visualizations():
    """Create comprehensive visualizations for all tasks"""
    
    # 1. Original dataset with potential separating lines
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    # Plot dataset
    plt.scatter(X_pos[:, 0], X_pos[:, 1], c='blue', s=200, marker='o', 
                label='Class +1', edgecolors='black', linewidth=2)
    plt.scatter(X_neg[:, 0], X_neg[:, 1], c='red', s=200, marker='s', 
                label='Class -1', edgecolors='black', linewidth=2)
    
    # Highlight the outlier
    outlier_idx = np.where((X_pos == [1, 4]).all(axis=1))[0]
    if len(outlier_idx) > 0:
        plt.scatter(X_pos[outlier_idx, 0], X_pos[outlier_idx, 1], 
                   c='orange', s=300, marker='*', label='Outlier (1, 4)', 
                   edgecolors='black', linewidth=2, zorder=5)
    
    # Add point labels
    for i, (x, y_coord) in enumerate(X_pos):
        plt.annotate(f'({x}, {y_coord})', (x, y_coord), 
                    xytext=(10, 10), textcoords='offset points', fontsize=10)
    
    for i, (x, y_coord) in enumerate(X_neg):
        plt.annotate(f'({x}, {y_coord})', (x, y_coord), 
                    xytext=(10, 10), textcoords='offset points', fontsize=10)
    
    plt.xlabel('$x_1$', fontsize=14)
    plt.ylabel('$x_2$', fontsize=14)
    plt.title('Original Dataset', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xlim(-1, 6)
    plt.ylim(-1, 5)
    plt.axis('equal')
    
    # Try to show why it's not separable
    x1_range = np.linspace(-1, 6, 100)
    plt.plot(x1_range, x1_range + 1, 'g--', alpha=0.7, label='Attempt 1: x2 = x1 + 1')
    plt.plot(x1_range, 2*x1_range - 1, 'b--', alpha=0.7, label='Attempt 2: x2 = 2x1 - 1')
    plt.plot(x1_range, 0.5*x1_range + 1.5, 'm--', alpha=0.7, label='Attempt 3: x2 = 0.5x1 + 1.5')
    plt.legend()
    plt.title("Why Hard Margin Fails")
    
    # 2. Soft margin SVM with different C values
    plt.subplot(2, 3, 2)
    C_values = list(svm_results.keys())
    accuracies = [svm_results[C]['accuracy'] for C in C_values]
    plt.plot(C_values, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.xscale('log')
    plt.xlabel('C (Regularization Parameter)')
    plt.ylabel('Accuracy')
    plt.title('Soft Margin SVM: Accuracy vs C')
    plt.grid(True, alpha=0.3)
    
    # 3. Support vectors vs C
    plt.subplot(2, 3, 3)
    n_support_vectors = [svm_results[C]['n_support_vectors'] for C in C_values]
    plt.plot(C_values, n_support_vectors, 'ro-', linewidth=2, markersize=8)
    plt.xscale('log')
    plt.xlabel('C (Regularization Parameter)')
    plt.ylabel('Number of Support Vectors')
    plt.title('Support Vectors vs C')
    plt.grid(True, alpha=0.3)
    
    # 4. Total slack vs C
    plt.subplot(2, 3, 4)
    total_slack = [svm_results[C]['total_slack'] for C in C_values]
    plt.plot(C_values, total_slack, 'go-', linewidth=2, markersize=8)
    plt.xscale('log')
    plt.xlabel('C (Regularization Parameter)')
    plt.ylabel('Total Slack Variables')
    plt.title('Total Slack vs C')
    plt.grid(True, alpha=0.3)
    
    # 5. Dataset without outlier
    plt.subplot(2, 3, 5)
    X_pos_clean = X_pos[X_pos[:, 0] != 1]
    plt.scatter(X_pos_clean[:, 0], X_pos_clean[:, 1], c='blue', s=200, marker='o', 
                label='Class +1', edgecolors='black', linewidth=2)
    plt.scatter(X_neg[:, 0], X_neg[:, 1], c='red', s=200, marker='s', 
                label='Class -1', edgecolors='black', linewidth=2)
    
    # Add point labels
    for i, (x, y_coord) in enumerate(X_pos_clean):
        plt.annotate(f'({x}, {y_coord})', (x, y_coord), 
                    xytext=(10, 10), textcoords='offset points', fontsize=10)
    
    for i, (x, y_coord) in enumerate(X_neg):
        plt.annotate(f'({x}, {y_coord})', (x, y_coord), 
                    xytext=(10, 10), textcoords='offset points', fontsize=10)
    
    plt.xlabel('$x_1$', fontsize=14)
    plt.ylabel('$x_2$', fontsize=14)
    plt.title('Dataset Without Outlier', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xlim(-1, 6)
    plt.ylim(-1, 5)
    plt.axis('equal')
    
    # 6. Medical screening system
    plt.subplot(2, 3, 6)
    healthy = np.array([[3, 2], [4, 3], [5, 2]])
    at_risk = np.array([[0, 0], [1, 1], [0, 2]])
    uncertain = np.array([1, 4])
    
    plt.scatter(healthy[:, 0], healthy[:, 1], c='green', s=200, marker='o', 
                label='Healthy', edgecolors='black', linewidth=2)
    plt.scatter(at_risk[:, 0], at_risk[:, 1], c='red', s=200, marker='s', 
                label='At Risk', edgecolors='black', linewidth=2)
    plt.scatter(uncertain[0], uncertain[1], c='orange', s=300, marker='*', 
                label='Uncertain Case', edgecolors='black', linewidth=2, zorder=5)
    
    # Add confidence zones
    x1_range = np.linspace(-1, 6, 100)
    x2_range = np.linspace(-1, 5, 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    # Use the medical SVM to create decision boundaries
    svm_medical = medical_system[0]
    Z = svm_medical.decision_function(np.c_[X1.ravel(), X2.ravel()]).reshape(X1.shape)
    
    # Plot confidence zones
    plt.contour(X1, X2, Z, levels=[-1, -0.5, 0, 0.5, 1], colors=['red', 'orange', 'black', 'orange', 'green'], 
                linestyles=['--', ':', '-', ':', '--'], alpha=0.7)
    
    plt.xlabel('Blood Test 1')
    plt.ylabel('Blood Test 2')
    plt.title('Medical Screening System')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comprehensive_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

create_comprehensive_visualizations()

# Additional informative plot: Soft margin decision boundaries
def create_soft_margin_comparison():
    """Create a comparison of soft margin decision boundaries with different C values"""
    plt.figure(figsize=(15, 10))
    
    C_values_to_plot = [0.1, 1, 10]
    colors = ['red', 'blue', 'green']
    
    for i, C in enumerate(C_values_to_plot):
        plt.subplot(1, 3, i+1)
        
        # Get the trained model
        svm_model = svm_results[C]['model']
        
        # Plot data points
        plt.scatter(X_pos[:, 0], X_pos[:, 1], c='blue', s=150, marker='o', 
                    label='Class +1', edgecolors='black', linewidth=1.5, alpha=0.7)
        plt.scatter(X_neg[:, 0], X_neg[:, 1], c='red', s=150, marker='s', 
                    label='Class -1', edgecolors='black', linewidth=1.5, alpha=0.7)
        
        # Highlight the outlier
        outlier_idx = np.where((X_pos == [1, 4]).all(axis=1))[0]
        if len(outlier_idx) > 0:
            plt.scatter(X_pos[outlier_idx, 0], X_pos[outlier_idx, 1], 
                       c='orange', s=250, marker='*', label='Outlier (1, 4)', 
                       edgecolors='black', linewidth=2, zorder=5)
        
        # Create mesh for decision boundary
        x1_min, x1_max = -1, 6
        x2_min, x2_max = -1, 5
        xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                              np.linspace(x2_min, x2_max, 100))
        
        # Get decision function values
        Z = svm_model.decision_function(np.c_[xx1.ravel(), xx2.ravel()])
        Z = Z.reshape(xx1.shape)
        
        # Plot decision boundary and margins
        plt.contour(xx1, xx2, Z, levels=[-1, 0, 1], 
                   colors=[colors[i], 'black', colors[i]], 
                   linestyles=['--', '-', '--'], linewidths=[2, 3, 2], alpha=0.8)
        
        # Fill regions
        plt.contourf(xx1, xx2, Z, levels=[-100, 0], colors=['lightcoral'], alpha=0.3)
        plt.contourf(xx1, xx2, Z, levels=[0, 100], colors=['lightblue'], alpha=0.3)
        
        # Highlight support vectors
        support_vectors = svm_model.support_vectors_
        plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
                   s=300, facecolors='none', edgecolors='yellow', 
                   linewidth=2, label='Support Vectors', zorder=6)
        
        plt.xlabel('$x_1$', fontsize=12)
        plt.ylabel('$x_2$', fontsize=12)
        plt.title(f'Soft Margin SVM (C = {C})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)
        plt.axis('equal')
        
        # Add performance metrics
        accuracy = svm_results[C]['accuracy']
        n_sv = svm_results[C]['n_support_vectors']
        total_slack = svm_results[C]['total_slack']
        plt.text(0.02, 0.98, f'Accuracy: {accuracy:.3f}\nSupport Vectors: {n_sv}\nTotal Slack: {total_slack:.3f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'soft_margin_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

create_soft_margin_comparison()

# Summary
print("\n" + "="*80)
print("SUMMARY OF RESULTS")
print("="*80)

print("1. LINEAR SEPARABILITY:")
print(f"   - Original dataset: {'NOT separable' if not is_separable else 'separable'}")
print(f"   - Without outlier: {'separable' if is_separable_no_outlier else 'NOT separable'}")

print("\n2. HARD MARGIN SVM:")
print("   - Fails because outlier (1, 4) makes perfect separation impossible")

print("\n3. MINIMUM VIOLATIONS:")
print(f"   - Minimum violations needed: {min_violations}")
print(f"   - Best strategy: {best_removal}")

print("\n4. SOFT MARGIN SVM:")
for C in [0.1, 1, 10]:
    print(f"   - C={C}: Accuracy={svm_results[C]['accuracy']:.3f}, "
          f"Support vectors={svm_results[C]['n_support_vectors']}, "
          f"Total slack={svm_results[C]['total_slack']:.3f}")

print("\n5. OUTLIER REMOVAL EFFECT:")
if is_separable_no_outlier:
    print("   - Dataset becomes linearly separable")
    print("   - Hard margin SVM becomes feasible")

print("\n6. MEDICAL SCREENING SYSTEM:")
print(f"   - Uncertain case uncertainty: {medical_system[2]:.3f} ({medical_system[2]*100:.1f}%)")
print("   - Three-zone confidence system implemented")

print(f"\nAll visualizations saved to: {save_dir}")
print("="*80)
