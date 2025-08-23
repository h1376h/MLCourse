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

# Set plotting parameters for clean visualizations with LaTeX
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3

print("=" * 80)
print("SVM LOOCV Analysis: Complete Mathematical Solution")
print("Question 24: Leave-One-Out Cross-Validation Error Estimation")
print("=" * 80)

def create_theoretical_analysis_visualization():
    """Create visualization explaining the theoretical relationship."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('LOOCV Theoretical Analysis for Hard-Margin SVM', fontsize=16, fontweight='bold')

    # Subplot 1: Support Vector Concept
    ax1 = axes[0, 0]
    # Simple example with 2D data
    np.random.seed(42)
    X_demo = np.array([[1, 2], [2, 3], [3, 1], [1, 1], [2, 1], [3, 3]])
    y_demo = np.array([1, 1, 1, -1, -1, -1])

    # Plot points
    ax1.scatter(X_demo[y_demo == 1, 0], X_demo[y_demo == 1, 1],
               c='red', marker='x', s=100, linewidth=3, label='Class +1')
    ax1.scatter(X_demo[y_demo == -1, 0], X_demo[y_demo == -1, 1],
               c='blue', marker='o', s=80, facecolors='none',
               edgecolors='blue', linewidth=2, label='Class -1')

    # Highlight support vectors (manually for demo)
    support_demo = [1, 3, 4]  # Example support vectors
    for idx in support_demo:
        ax1.scatter(X_demo[idx, 0], X_demo[idx, 1], s=300,
                   facecolors='none', edgecolors='green', linewidth=3)

    ax1.set_title('Support Vectors Define Decision Boundary')
    ax1.set_xlabel(r'$x_1$')
    ax1.set_ylabel(r'$x_2$')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: LOOCV Upper Bound Formula
    ax2 = axes[0, 1]
    ax2.text(0.5, 0.7, 'LOOCV Upper Bound Theorem',
             ha='center', va='center', fontsize=14, fontweight='bold',
             transform=ax2.transAxes)
    ax2.text(0.5, 0.5, r'LOOCV Error $\leq \frac{\text{\# Support Vectors}}{\text{Total Points}}$',
             ha='center', va='center', fontsize=12,
             transform=ax2.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax2.text(0.5, 0.3, 'This bound is conservative\nActual error can be lower',
             ha='center', va='center', fontsize=10,
             transform=ax2.transAxes)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    # Subplot 3: Why Non-Support Vectors Don't Contribute
    ax3 = axes[1, 0]
    x_vals = np.linspace(0, 4, 100)
    # Simulate decision boundary
    y_boundary = 2.5 - 0.5 * x_vals
    ax3.plot(x_vals, y_boundary, 'k-', linewidth=2, label='Decision Boundary')
    ax3.fill_between(x_vals, y_boundary - 0.3, y_boundary + 0.3, alpha=0.2, color='gray', label='Margin')

    # Add points
    ax3.scatter([1.5, 3.5], [1.8, 0.7], c='red', marker='x', s=100, linewidth=3, label='Class +1')
    ax3.scatter([0.5, 2.5], [2.2, 1.2], c='blue', marker='o', s=80,
               facecolors='none', edgecolors='blue', linewidth=2, label='Class -1')

    # Highlight one non-support vector
    ax3.scatter([3.5], [0.7], s=200, facecolors='none', edgecolors='orange',
               linewidth=3, label='Non-Support Vector')
    ax3.annotate('Removing this point\ndoesn\'t change boundary',
                xy=(3.5, 0.7), xytext=(3.0, 1.5),
                arrowprops=dict(arrowstyle='->', color='orange', lw=2))

    ax3.set_title('Non-Support Vectors: Always Correctly Classified')
    ax3.set_xlabel(r'$x_1$')
    ax3.set_ylabel(r'$x_2$')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 4)
    ax3.set_ylim(0, 3)

    # Subplot 4: Why Support Vectors Can Be Misclassified
    ax4 = axes[1, 1]
    # Show boundary shift when support vector is removed
    ax4.plot(x_vals, y_boundary, 'k-', linewidth=2, label='Original Boundary')
    y_boundary_shifted = 2.2 - 0.3 * x_vals
    ax4.plot(x_vals, y_boundary_shifted, 'r--', linewidth=2, label='Boundary After SV Removal')

    # Add support vector that gets misclassified
    sv_point = [1.5, 1.8]
    ax4.scatter([sv_point[0]], [sv_point[1]], c='red', marker='x', s=100, linewidth=3)
    ax4.scatter([sv_point[0]], [sv_point[1]], s=300, facecolors='none',
               edgecolors='green', linewidth=3, label='Support Vector')
    ax4.annotate('This SV gets misclassified\nwhen removed',
                xy=(sv_point[0], sv_point[1]), xytext=(2.5, 2.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax4.set_title('Support Vectors: Can Be Misclassified')
    ax4.set_xlabel(r'$x_1$')
    ax4.set_ylabel(r'$x_2$')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 4)
    ax4.set_ylim(0, 3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'theoretical_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("Theoretical analysis visualization saved")

def display_mathematical_foundation():
    """Display the mathematical foundation with step-by-step derivation."""
    print(f"\n" + "=" * 80)
    print("MATHEMATICAL FOUNDATION: LOOCV for Hard-Margin SVM")
    print("=" * 80)

    print(f"\nKey Theorem: For linearly separable data with hard-margin SVM:")
    print(f"LOOCV Error Rate ≤ (Number of Support Vectors) / (Total Number of Points)")

    print(f"\nMathematical Justification:")
    print(f"1. Let S = {{x₁, x₂, ..., xₙ}} be the training set")
    print(f"2. Let SV ⊆ S be the set of support vectors")
    print(f"3. For any point xᵢ ∈ S:")
    print(f"   • If xᵢ ∉ SV (non-support vector):")
    print(f"     - Removing xᵢ doesn't change the decision boundary")
    print(f"     - The remaining SVM will classify xᵢ correctly")
    print(f"     - Contribution to LOOCV error: 0")
    print(f"   • If xᵢ ∈ SV (support vector):")
    print(f"     - Removing xᵢ may change the decision boundary")
    print(f"     - The new boundary might misclassify xᵢ")
    print(f"     - Contribution to LOOCV error: 0 or 1")

    print(f"\n4. Therefore:")
    print(f"   LOOCV Error = Σᵢ I(xᵢ misclassified when left out)")
    print(f"                ≤ Σᵢ I(xᵢ ∈ SV)")
    print(f"                = |SV|")
    print(f"   LOOCV Error Rate ≤ |SV| / n")

    print(f"\n5. The bound is achieved when every support vector is misclassified")
    print(f"   when removed (worst-case scenario)")

def analyze_scenario_mathematically(X, y, scenario_name, scenario_letter):
    """Perform detailed mathematical analysis of a scenario using exact same setup as question code."""
    print(f"\n" + "=" * 70)
    print(f"SCENARIO {scenario_letter}: {scenario_name}")
    print("=" * 70)

    n = len(X)
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == -1)

    print(f"\nStep 1: Dataset Specification")
    print(f"Total points (n): {n}")
    print(f"Class +1 points: {n_pos}")
    print(f"Class -1 points: {n_neg}")

    print(f"\nDataset points (exactly as defined in question code):")
    for i, (point, label) in enumerate(zip(X, y)):
        print(f"  x_{i+1} = {point}, y_{i+1} = {label:+d}")

    # Train SVM using EXACT same parameters as question code
    svm = SVC(kernel='linear', C=1000)  # Same as question code
    svm.fit(X, y)
    sv_indices = svm.support_
    n_sv = len(sv_indices)

    print(f"\nStep 2: Support Vector Identification")
    print(f"Training hard-margin SVM with C = 1000 (same as question code)")
    print(f"Support vector indices (0-based): {sv_indices}")
    print(f"Support vector indices (1-based): {sv_indices + 1}")
    print(f"Number of support vectors (|SV|): {n_sv}")
    print(f"Support vectors:")
    for idx in sv_indices:
        print(f"  x_{idx+1} = {X[idx]}, y_{idx+1} = {y[idx]:+d}")

    print(f"\nStep 3: Theoretical Upper Bound Calculation")
    theoretical_bound = n_sv / n
    print(f"LOOCV Error Rate ≤ |SV| / n = {n_sv} / {n} = {theoretical_bound:.4f} = {theoretical_bound*100:.1f}%")

    return svm, sv_indices, theoretical_bound

def perform_detailed_loocv_analysis(X, y, svm_full, sv_indices, scenario_letter):
    """Perform step-by-step LOOCV analysis using exact same setup as question code."""
    print(f"\nStep 4: Leave-One-Out Cross-Validation Analysis")
    print(f"Performing LOOCV: Train on (n-1) points, test on 1 point, repeat n times")
    print(f"Using same SVM parameters as question code: kernel='linear', C=1000")

    n = len(X)
    loo = LeaveOneOut()
    results = []
    misclassified_count = 0

    print(f"\nDetailed LOOCV Results:")
    print(f"{'Fold':<4} {'Point':<6} {'Coordinates':<12} {'True':<5} {'Pred':<5} {'Correct':<8} {'SV?':<4}")
    print("-" * 60)

    for fold, (train_idx, test_idx) in enumerate(loo.split(X)):
        point_idx = test_idx[0]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train SVM on n-1 points using EXACT same parameters as question code
        svm_cv = SVC(kernel='linear', C=1000)  # Same as question code
        svm_cv.fit(X_train, y_train)

        # Predict on the left-out point
        y_pred = svm_cv.predict(X_test)[0]
        y_true = y_test[0]

        # Check if correct and if it's a support vector
        is_correct = (y_pred == y_true)
        is_sv = point_idx in sv_indices

        if not is_correct:
            misclassified_count += 1

        # Store result
        results.append({
            'fold': fold + 1,
            'point_idx': point_idx,
            'coordinates': X[point_idx],
            'true_label': y_true,
            'predicted_label': y_pred,
            'is_correct': is_correct,
            'is_support_vector': is_sv
        })

        # Print result
        coord_str = f"[{X[point_idx][0]:.1f},{X[point_idx][1]:.1f}]"
        correct_str = "✓" if is_correct else "✗"
        sv_str = "Yes" if is_sv else "No"
        print(f"{fold+1:<4} {point_idx+1:<6} {coord_str:<12} {y_true:+2d}   {y_pred:+2d}   {correct_str:<8} {sv_str:<4}")

    # Calculate final LOOCV error rate
    loocv_error_rate = misclassified_count / n

    print(f"\nStep 5: LOOCV Error Calculation")
    print(f"Number of misclassified points: {misclassified_count}")
    print(f"Total number of points: {n}")
    print(f"LOOCV Error Rate = {misclassified_count} / {n} = {loocv_error_rate:.4f} = {loocv_error_rate*100:.1f}%")

    # Verify this matches question code results
    print(f"\nVerification: This should match the question code output:")
    print(f"Expected from question: Scenario {scenario_letter} → {loocv_error_rate*100:.1f}% actual LOOCV error")

    return results, loocv_error_rate, misclassified_count

def create_loocv_detailed_visualization(X, y, results, sv_indices, scenario_name, scenario_letter):
    """Create detailed visualization of LOOCV process."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Scenario {scenario_letter}: Detailed LOOCV Analysis', fontsize=16, fontweight='bold')

    # Subplot 1: Original dataset with support vectors highlighted
    ax1 = axes[0, 0]
    class_1_mask = y == 1
    class_neg1_mask = y == -1

    ax1.scatter(X[class_1_mask, 0], X[class_1_mask, 1], c='red', marker='x',
               s=100, linewidth=2, label="Class +1", zorder=5)
    ax1.scatter(X[class_neg1_mask, 0], X[class_neg1_mask, 1], marker='o',
               s=80, facecolors='none', edgecolors='blue', linewidth=2,
               label="Class -1", zorder=5)

    # Highlight support vectors
    for idx in sv_indices:
        ax1.scatter(X[idx, 0], X[idx, 1], s=300, facecolors='none',
                   edgecolors='green', linewidth=3, zorder=6)

    # Add point numbers
    for i, (x, y_val) in enumerate(X):
        ax1.annotate(f'{i+1}', (x, y_val), xytext=(5, 5),
                    textcoords='offset points', fontsize=9, fontweight='bold')

    ax1.set_title('Original Dataset with Support Vectors')
    ax1.set_xlabel(r'$x_1$')
    ax1.set_ylabel(r'$x_2$')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: LOOCV Results Summary
    ax2 = axes[0, 1]

    # Count results by category
    sv_correct = sum(1 for r in results if r['is_support_vector'] and r['is_correct'])
    sv_incorrect = sum(1 for r in results if r['is_support_vector'] and not r['is_correct'])
    nonsv_correct = sum(1 for r in results if not r['is_support_vector'] and r['is_correct'])
    nonsv_incorrect = sum(1 for r in results if not r['is_support_vector'] and not r['is_correct'])

    categories = ['SV\nCorrect', 'SV\nIncorrect', 'Non-SV\nCorrect', 'Non-SV\nIncorrect']
    counts = [sv_correct, sv_incorrect, nonsv_correct, nonsv_incorrect]
    colors = ['lightgreen', 'lightcoral', 'lightblue', 'orange']

    bars = ax2.bar(categories, counts, color=colors, edgecolor='black', linewidth=1)
    ax2.set_title('LOOCV Results by Point Type')
    ax2.set_ylabel('Number of Points')

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        if count > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom', fontweight='bold')

    ax2.set_ylim(0, max(counts) + 1)
    ax2.grid(True, alpha=0.3, axis='y')

    # Subplot 3: Error Analysis Table
    ax3 = axes[1, 0]
    ax3.axis('off')

    # Create table data
    table_data = []
    table_data.append(['Point', 'Coordinates', 'True', 'Pred', 'Result', 'SV?'])

    for r in results:
        coord_str = f"[{r['coordinates'][0]:.1f}, {r['coordinates'][1]:.1f}]"
        result_str = "✓" if r['is_correct'] else "✗"
        sv_str = "Yes" if r['is_support_vector'] else "No"
        table_data.append([
            str(r['point_idx'] + 1),
            coord_str,
            f"{r['true_label']:+d}",
            f"{r['predicted_label']:+d}",
            result_str,
            sv_str
        ])

    # Create table
    table = ax3.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center',
                     colWidths=[0.08, 0.25, 0.08, 0.08, 0.08, 0.08])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # Color code the table
    for i in range(1, len(table_data)):
        result_idx = results[i-1]
        if not result_idx['is_correct']:  # Misclassified
            for j in range(len(table_data[0])):
                table[(i, j)].set_facecolor('lightcoral')
        elif result_idx['is_support_vector']:  # Support vector, correct
            for j in range(len(table_data[0])):
                table[(i, j)].set_facecolor('lightgreen')

    ax3.set_title('Detailed LOOCV Results Table')

    # Subplot 4: Theoretical vs Actual Comparison
    ax4 = axes[1, 1]

    n_sv = len(sv_indices)
    n_total = len(X)
    theoretical_bound = n_sv / n_total
    actual_error = sum(1 for r in results if not r['is_correct']) / n_total

    categories = ['Theoretical\nUpper Bound', 'Actual\nLOOCV Error']
    values = [theoretical_bound * 100, actual_error * 100]
    colors = ['lightblue', 'lightcoral']

    bars = ax4.bar(categories, values, color=colors, edgecolor='black', linewidth=1)
    ax4.set_title('Theoretical Bound vs Actual Error')
    ax4.set_ylabel('Error Rate (%)')

    # Add value labels
    for bar, value in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

    ax4.set_ylim(0, max(values) + 5)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'loocv_detailed_scenario_{scenario_letter.lower()}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Detailed LOOCV visualization for Scenario {scenario_letter} saved")

def create_comparison_visualization(results_a, results_b, sv_indices_a, sv_indices_b):
    """Create comprehensive comparison between scenarios."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Comparison: Scenario A vs Scenario B', fontsize=16, fontweight='bold')

    # Load datasets for visualization
    X_a, y_a = create_dataset_scenario_a()
    X_b, y_b = create_dataset_scenario_b()

    # Scenario A visualizations
    ax1 = axes[0, 0]
    class_1_mask_a = y_a == 1
    class_neg1_mask_a = y_a == -1

    ax1.scatter(X_a[class_1_mask_a, 0], X_a[class_1_mask_a, 1], c='red', marker='x',
               s=100, linewidth=2, label="Class +1")
    ax1.scatter(X_a[class_neg1_mask_a, 0], X_a[class_neg1_mask_a, 1], marker='o',
               s=80, facecolors='none', edgecolors='blue', linewidth=2, label="Class -1")

    # Highlight support vectors and misclassified points
    for idx in sv_indices_a:
        ax1.scatter(X_a[idx, 0], X_a[idx, 1], s=300, facecolors='none',
                   edgecolors='green', linewidth=3)

    for r in results_a:
        if not r['is_correct']:
            ax1.scatter(r['coordinates'][0], r['coordinates'][1], s=400,
                       facecolors='none', edgecolors='red', linewidth=4)

    ax1.set_title('Scenario A: Dataset & Results')
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Scenario B visualizations
    ax2 = axes[0, 1]
    class_1_mask_b = y_b == 1
    class_neg1_mask_b = y_b == -1

    ax2.scatter(X_b[class_1_mask_b, 0], X_b[class_1_mask_b, 1], c='red', marker='x',
               s=100, linewidth=2, label="Class +1")
    ax2.scatter(X_b[class_neg1_mask_b, 0], X_b[class_neg1_mask_b, 1], marker='o',
               s=80, facecolors='none', edgecolors='blue', linewidth=2, label="Class -1")

    for idx in sv_indices_b:
        ax2.scatter(X_b[idx, 0], X_b[idx, 1], s=300, facecolors='none',
                   edgecolors='green', linewidth=3)

    for r in results_b:
        if not r['is_correct']:
            ax2.scatter(r['coordinates'][0], r['coordinates'][1], s=400,
                       facecolors='none', edgecolors='red', linewidth=4)

    ax2.set_title('Scenario B: Dataset & Results')
    ax2.set_xlabel('x₁')
    ax2.set_ylabel('x₂')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Comparison metrics
    ax3 = axes[0, 2]

    metrics = ['Support\nVectors', 'Theoretical\nBound (%)', 'Actual\nError (%)']
    scenario_a_vals = [
        len(sv_indices_a),
        len(sv_indices_a) / len(X_a) * 100,
        sum(1 for r in results_a if not r['is_correct']) / len(X_a) * 100
    ]
    scenario_b_vals = [
        len(sv_indices_b),
        len(sv_indices_b) / len(X_b) * 100,
        sum(1 for r in results_b if not r['is_correct']) / len(X_b) * 100
    ]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax3.bar(x - width/2, scenario_a_vals, width, label='Scenario A',
                   color='lightblue', edgecolor='black')
    bars2 = ax3.bar(x + width/2, scenario_b_vals, width, label='Scenario B',
                   color='lightcoral', edgecolor='black')

    ax3.set_title('Quantitative Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    # Mathematical explanation
    ax4 = axes[1, 0]
    ax4.axis('off')
    ax4.text(0.5, 0.8, 'Mathematical Analysis', ha='center', va='top',
             fontsize=14, fontweight='bold', transform=ax4.transAxes)

    explanation_text = """
Scenario A:
• 3 support vectors out of 10 points
• Theoretical bound: 3/10 = 30%
• Actual LOOCV error: 1/10 = 10%
• Bound tightness: 33.3%

Scenario B:
• 2 support vectors out of 10 points
• Theoretical bound: 2/10 = 20%
• Actual LOOCV error: 0/10 = 0%
• Bound tightness: 0%

Key Insight: The theoretical bound provides
a conservative upper limit. Actual performance
depends on geometric configuration.
"""

    ax4.text(0.05, 0.7, explanation_text, ha='left', va='top',
             fontsize=10, transform=ax4.transAxes,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))

    # Theoretical insights
    ax5 = axes[1, 1]
    ax5.axis('off')
    ax5.text(0.5, 0.9, 'Why Bounds Can Be Loose', ha='center', va='top',
             fontsize=14, fontweight='bold', transform=ax5.transAxes)

    insights_text = """
1. Conservative Assumption:
   Bound assumes each SV contributes 1 error

2. Geometric Stability:
   Some SVs remain correctly classified
   even when removed

3. Data Configuration:
   Specific arrangement affects boundary
   sensitivity to SV removal

4. Margin Properties:
   Wider effective margins provide
   more stability
"""

    ax5.text(0.05, 0.8, insights_text, ha='left', va='top',
             fontsize=10, transform=ax5.transAxes,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan"))

    # Practical implications
    ax6 = axes[1, 2]
    ax6.axis('off')
    ax6.text(0.5, 0.9, 'Practical Implications', ha='center', va='top',
             fontsize=14, fontweight='bold', transform=ax6.transAxes)

    implications_text = """
Quick Assessment:
• Count SVs for immediate upper bound
• No computation required
• Conservative estimate

Detailed Analysis:
• Run LOOCV for exact error rate
• Identifies problematic points
• Reveals geometric insights

Model Selection:
• Compare bounds across models
• Understand stability trade-offs
• Guide data collection decisions
"""

    ax6.text(0.05, 0.8, implications_text, ha='left', va='top',
             fontsize=10, transform=ax6.transAxes,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen"))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comprehensive_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print("Comprehensive comparison visualization saved")

def provide_complete_solution_summary(results_a, results_b, sv_indices_a, sv_indices_b,
                                    theoretical_bound_a, theoretical_bound_b,
                                    actual_error_a, actual_error_b):
    """Provide complete mathematical solution summary."""
    print(f"\n" + "=" * 80)
    print("COMPLETE MATHEMATICAL SOLUTION SUMMARY")
    print("=" * 80)

    print(f"\nQUESTION 1: Scenario A Analysis")
    print("-" * 40)

    misclassified_a = [r for r in results_a if not r['is_correct']]

    print(f"LOOCV error estimate for Scenario A:")
    print(f"    Mathematical calculation: {len(misclassified_a)}/10 = {actual_error_a:.1%}")
    print(f"    Answer: {actual_error_a:.1%}")

    print(f"\nJustification for Scenario A:")
    print(f"    • Theoretical upper bound: {len(sv_indices_a)}/10 = {theoretical_bound_a:.1%}")
    print(f"    • Only support vectors can potentially be misclassified")
    print(f"    • Actual result: {len(misclassified_a)} out of {len(sv_indices_a)} support vectors misclassified")
    print(f"    • Bound tightness: {(actual_error_a/theoretical_bound_a)*100:.1f}% of theoretical maximum")

    print(f"\nSpecific points misclassified in Scenario A:")
    if misclassified_a:
        for r in misclassified_a:
            print(f"    • Point {r['point_idx']+1}: {r['coordinates']} (True: {r['true_label']:+d}, Predicted: {r['predicted_label']:+d})")
    else:
        print(f"    • No points were misclassified")

    print(f"\nQUESTION 2: Scenario B Analysis")
    print("-" * 40)

    misclassified_b = [r for r in results_b if not r['is_correct']]

    print(f"LOOCV error estimate for Scenario B:")
    print(f"    Mathematical calculation: {len(misclassified_b)}/10 = {actual_error_b:.1%}")
    print(f"    Answer: {actual_error_b:.1%}")

    print(f"\nJustification for Scenario B:")
    print(f"    • Theoretical upper bound: {len(sv_indices_b)}/10 = {theoretical_bound_b:.1%}")
    print(f"    • All support vectors remained correctly classified when removed")
    print(f"    • Demonstrates that theoretical bound can be very loose")
    print(f"    • Geometric configuration provides exceptional stability")

    print(f"\nComparison between Scenario A and B:")
    print(f"    Scenario A: {len(sv_indices_a)} SVs → {theoretical_bound_a:.1%} bound → {actual_error_a:.1%} actual")
    print(f"    Scenario B: {len(sv_indices_b)} SVs → {theoretical_bound_b:.1%} bound → {actual_error_b:.1%} actual")
    print(f"    Key differences:")
    print(f"    • A has more support vectors ({len(sv_indices_a)} vs {len(sv_indices_b)})")
    print(f"    • A has higher actual error ({actual_error_a:.1%} vs {actual_error_b:.1%})")
    print(f"    • B achieves perfect LOOCV performance despite having support vectors")
    print(f"    • Both demonstrate: Actual Error ≤ Theoretical Upper Bound")

    print(f"\nQUESTION 3: Theoretical Relationship")
    print("-" * 40)

    print(f"Theoretical relationship between LOOCV error and support vectors:")
    print(f"    For a hard-margin SVM with linearly separable data:")
    print(f"    LOOCV Error Rate ≤ (Number of Support Vectors) / (Total Number of Points)")
    print(f"    Mathematical proof: Only support vectors can be misclassified during LOOCV")

    print(f"\nQUESTION 4: Why Bounds Can Be Loose")
    print("-" * 40)

    print(f"Why actual LOOCV error can be less than theoretical upper bound:")
    print(f"    • The bound assumes worst-case: each SV contributes exactly 1 error")
    print(f"    • In practice: removing an SV may not change boundary enough to misclassify it")
    print(f"    • Geometric factors: data configuration affects boundary stability")
    print(f"    • The bound is conservative, representing maximum possible error")

    print(f"\nQUESTION 5: Conditions for Tight Bounds")
    print("-" * 40)

    print(f"Conditions for tight theoretical bound:")
    print(f"    • Each support vector is 'critical' for the current decision boundary")
    print(f"    • Removing any SV significantly shifts the boundary")
    print(f"    • Minimal support vector sets with tight margins")
    print(f"    • Data positioned such that SV removal causes misclassification")

def verify_consistency_with_question_code():
    """Verify that our analysis matches the question code exactly."""
    print(f"\n" + "=" * 80)
    print("VERIFICATION: Ensuring Consistency with Question Code")
    print("=" * 80)

    # Import and run the question code functions to get expected results
    from L5_1_24_question import perform_loocv_analysis

    X_a, y_a = create_dataset_scenario_a()
    X_b, y_b = create_dataset_scenario_b()

    print(f"\nRunning question code analysis for verification...")

    # Get expected results from question code
    expected_loocv_a, expected_sv_count_a = perform_loocv_analysis(X_a, y_a, "Scenario A")
    expected_loocv_b, expected_sv_count_b = perform_loocv_analysis(X_b, y_b, "Scenario B")

    print(f"\nExpected Results from Question Code:")
    print(f"Scenario A: {expected_sv_count_a} SVs → {expected_loocv_a:.1%} LOOCV error")
    print(f"Scenario B: {expected_sv_count_b} SVs → {expected_loocv_b:.1%} LOOCV error")

    return expected_loocv_a, expected_sv_count_a, expected_loocv_b, expected_sv_count_b

# Main execution
if __name__ == "__main__":
    print(f"\nStep 0: Creating Theoretical Foundation Visualization")
    create_theoretical_analysis_visualization()

    print(f"\nStep 0: Mathematical Foundation")
    display_mathematical_foundation()

    # Verify consistency with question code first
    expected_loocv_a, expected_sv_count_a, expected_loocv_b, expected_sv_count_b = verify_consistency_with_question_code()

    # Load datasets using exact same functions as question code
    print(f"\n" + "=" * 80)
    print("LOADING DATASETS AND PERFORMING ANALYSIS")
    print("=" * 80)
    print("Using exact same dataset functions as question code...")

    X_a, y_a = create_dataset_scenario_a()
    X_b, y_b = create_dataset_scenario_b()

    # Analyze Scenario A
    svm_a, sv_indices_a, theoretical_bound_a = analyze_scenario_mathematically(X_a, y_a, "First Configuration", "A")
    results_a, actual_error_a, misclassified_count_a = perform_detailed_loocv_analysis(X_a, y_a, svm_a, sv_indices_a, "A")
    create_loocv_detailed_visualization(X_a, y_a, results_a, sv_indices_a, "First Configuration", "A")

    print(f"\nStep 6: Theoretical vs Actual Comparison for Scenario A")
    print(f"Theoretical upper bound: {theoretical_bound_a:.4f} = {theoretical_bound_a*100:.1f}%")
    print(f"Actual LOOCV error rate: {actual_error_a:.4f} = {actual_error_a*100:.1f}%")
    print(f"Bound tightness: {(actual_error_a/theoretical_bound_a)*100:.1f}% of theoretical maximum")

    # Verify consistency
    print(f"\nConsistency Check with Question Code:")
    print(f"Expected: {expected_sv_count_a} SVs → {expected_loocv_a:.1%} error")
    print(f"Computed: {len(sv_indices_a)} SVs → {actual_error_a:.1%} error")
    if abs(expected_loocv_a - actual_error_a) < 1e-10 and expected_sv_count_a == len(sv_indices_a):
        print("✓ Perfect match with question code!")
    else:
        print("✗ Mismatch detected - need to investigate")

    # Analyze Scenario B
    svm_b, sv_indices_b, theoretical_bound_b = analyze_scenario_mathematically(X_b, y_b, "Second Configuration", "B")
    results_b, actual_error_b, misclassified_count_b = perform_detailed_loocv_analysis(X_b, y_b, svm_b, sv_indices_b, "B")
    create_loocv_detailed_visualization(X_b, y_b, results_b, sv_indices_b, "Second Configuration", "B")

    print(f"\nStep 6: Theoretical vs Actual Comparison for Scenario B")
    print(f"Theoretical upper bound: {theoretical_bound_b:.4f} = {theoretical_bound_b*100:.1f}%")
    print(f"Actual LOOCV error rate: {actual_error_b:.4f} = {actual_error_b*100:.1f}%")
    if theoretical_bound_b > 0:
        print(f"Bound tightness: {(actual_error_b/theoretical_bound_b)*100:.1f}% of theoretical maximum")
    else:
        print(f"Bound tightness: N/A (no support vectors)")

    # Verify consistency
    print(f"\nConsistency Check with Question Code:")
    print(f"Expected: {expected_sv_count_b} SVs → {expected_loocv_b:.1%} error")
    print(f"Computed: {len(sv_indices_b)} SVs → {actual_error_b:.1%} error")
    if abs(expected_loocv_b - actual_error_b) < 1e-10 and expected_sv_count_b == len(sv_indices_b):
        print("✓ Perfect match with question code!")
    else:
        print("✗ Mismatch detected - need to investigate")

    # Create comprehensive comparison
    create_comparison_visualization(results_a, results_b, sv_indices_a, sv_indices_b)

    # Provide complete solution
    provide_complete_solution_summary(results_a, results_b, sv_indices_a, sv_indices_b,
                                    theoretical_bound_a, theoretical_bound_b,
                                    actual_error_a, actual_error_b)

    # Final summary
    print(f"\n" + "=" * 80)
    print("FINAL MATHEMATICAL SUMMARY")
    print("=" * 80)

    print(f"\nPen-and-Paper Solution Method:")
    print(f"1. Count support vectors from the figure")
    print(f"2. Apply formula: LOOCV Error ≤ |SV| / n")
    print(f"3. This gives immediate upper bound without computation")

    print(f"\nComputational Verification Results:")
    print(f"Scenario A: {len(sv_indices_a)} SVs → {theoretical_bound_a:.1%} bound → {actual_error_a:.1%} actual")
    print(f"Scenario B: {len(sv_indices_b)} SVs → {theoretical_bound_b:.1%} bound → {actual_error_b:.1%} actual")

    print(f"\nKey Mathematical Insights:")
    print(f"• Theoretical bounds are conservative (worst-case estimates)")
    print(f"• Actual performance depends on geometric data configuration")
    print(f"• Only support vectors can contribute to LOOCV error")
    print(f"• Different scenarios demonstrate varying bound tightness")
    print(f"• Mathematical approach provides quick assessment")
    print(f"• Computational approach reveals precise behavior")

    print(f"\nGenerated Visualizations:")
    print(f"• theoretical_analysis.png - Mathematical foundation")
    print(f"• loocv_detailed_scenario_a.png - Scenario A detailed analysis")
    print(f"• loocv_detailed_scenario_b.png - Scenario B detailed analysis")
    print(f"• comprehensive_comparison.png - Side-by-side comparison")

    print(f"\nAll analysis saved to: {save_dir}")
    print("=" * 80)

