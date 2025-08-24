import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_2_Quiz_22")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'

def create_svm_visualization():
    """Create a comprehensive visualization of soft-margin SVM with slack variables"""
    
    # Create a dataset that's not perfectly linearly separable
    np.random.seed(42)
    X, y = make_blobs(n_samples=100, centers=2, cluster_std=1.5, random_state=42)
    
    # Convert labels from 0,1 to -1,1
    y = 2 * y - 1
    
    # Add some noise to make it non-separable
    noise_points = np.array([
        [2.5, 1.5],  # Point that will be inside margin (0 < ξ ≤ 1)
        [3.0, 0.5],  # Point that will be misclassified (ξ > 1)
        [-1.0, 2.0], # Another point inside margin
        [1.5, 3.0]   # Another misclassified point
    ])
    noise_labels = np.array([1, 1, -1, -1])
    
    X = np.vstack([X, noise_points])
    y = np.hstack([y, noise_labels])
    
    # Train soft-margin SVM
    svm = SVC(kernel='linear', C=1.0, random_state=42)
    svm.fit(X, y)
    
    # Get support vectors
    support_vectors = svm.support_vectors_
    
    # Calculate slack variables for all points
    slack_variables = []
    for i in range(len(X)):
        x = X[i]
        label = y[i]
        # Calculate distance from decision boundary
        decision_value = svm.decision_function([x])[0]
        
        # Calculate slack variable
        if label * decision_value >= 1:
            # Correctly classified with margin
            slack = 0
        else:
            # Violates margin or misclassified
            slack = 1 - label * decision_value
        
        slack_variables.append(slack)
    
    slack_variables = np.array(slack_variables)
    
    # Create the main visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Plot all points
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', s=50, alpha=0.6, label='Class +1')
    ax.scatter(X[y == -1, 0], X[y == -1, 1], c='blue', s=50, alpha=0.6, label='Class -1')
    
    # Highlight support vectors
    ax.scatter(support_vectors[:, 0], support_vectors[:, 1], 
               s=200, facecolors='none', edgecolors='black', linewidth=2, 
               label='Support Vectors')
    
    # Create grid for decision boundary and margins
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Calculate decision function values
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and margins
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['blue', 'black', 'red'], 
               linestyles=['--', '-', '--'], linewidths=[2, 3, 2])
    
    # Add labels for decision boundary and margins
    ax.text(0.5, 0.5, 'Decision Boundary\n(w·x + b = 0)', 
            transform=ax.transAxes, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
    
    # Highlight points based on their slack variable values
    for i in range(len(X)):
        x = X[i]
        slack = slack_variables[i]
        
        if slack == 0:
            # Correctly classified with margin
            ax.scatter(x[0], x[1], s=100, c='green', marker='o', 
                      edgecolors='black', linewidth=2, alpha=0.8)
            ax.annotate(f'$\\xi={slack:.2f}$', (x[0], x[1]), 
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, bbox=dict(boxstyle="round,pad=0.2", 
                                             fc="lightgreen", ec="black", alpha=0.8))
        elif 0 < slack <= 1:
            # Inside margin
            ax.scatter(x[0], x[1], s=100, c='orange', marker='s', 
                      edgecolors='black', linewidth=2, alpha=0.8)
            ax.annotate(f'$\\xi={slack:.2f}$', (x[0], x[1]), 
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, bbox=dict(boxstyle="round,pad=0.2", 
                                             fc="lightyellow", ec="black", alpha=0.8))
        else:
            # Misclassified
            ax.scatter(x[0], x[1], s=100, c='red', marker='^', 
                      edgecolors='black', linewidth=2, alpha=0.8)
            ax.annotate(f'$\\xi={slack:.2f}$', (x[0], x[1]), 
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, bbox=dict(boxstyle="round,pad=0.2", 
                                             fc="lightcoral", ec="black", alpha=0.8))
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Add title and labels
    ax.set_title('Soft-Margin SVM: Geometric Interpretation of Slack Variables', fontsize=16)
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add text box explaining the cases
    explanation_text = (
        'Slack Variable Cases:\n'
        '$\\bullet$ $\\xi = 0$: Correctly classified with margin\n'
        '$\\bullet$ $0 < \\xi \\leq 1$: Inside margin (violates margin)\n'
        '$\\bullet$ $\\xi > 1$: Misclassified'
    )
    ax.text(0.02, 0.98, explanation_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'slack_variables_geometric_interpretation.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return X, y, slack_variables, svm

def create_detailed_case_analysis():
    """Create detailed visualizations for each slack variable case"""
    
    # Create synthetic data for clear demonstration
    np.random.seed(42)
    
    # Case 1: ξ = 0 (correctly classified with margin)
    X1 = np.array([[2, 2], [3, 3], [-2, -2], [-3, -3]])
    y1 = np.array([1, 1, -1, -1])
    
    # Case 2: 0 < ξ ≤ 1 (inside margin)
    X2 = np.array([[1.5, 1.5], [-1.5, -1.5]])
    y2 = np.array([1, -1])
    
    # Case 3: ξ > 1 (misclassified)
    X3 = np.array([[0.5, 0.5], [-0.5, -0.5]])
    y3 = np.array([1, -1])
    
    # Combine all data
    X = np.vstack([X1, X2, X3])
    y = np.hstack([y1, y2, y3])
    
    # Train SVM
    svm = SVC(kernel='linear', C=1.0, random_state=42)
    svm.fit(X, y)
    
    # Calculate slack variables
    slack_variables = []
    for i in range(len(X)):
        x = X[i]
        label = y[i]
        decision_value = svm.decision_function([x])[0]
        if label * decision_value >= 1:
            slack = 0
        else:
            slack = 1 - label * decision_value
        slack_variables.append(slack)
    
    # Create subplots for each case
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    cases = [
        ("$\\xi = 0$: Correctly Classified with Margin", [0, 1, 2, 3], 'green', 'o'),
        ("$0 < \\xi \\leq 1$: Inside Margin", [4, 5], 'orange', 's'),
        ("$\\xi > 1$: Misclassified", [6, 7], 'red', '^')
    ]
    
    for idx, (title, indices, color, marker) in enumerate(cases):
        ax = axes[idx]
        
        # Plot all points
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', s=50, alpha=0.6, label='Class +1')
        ax.scatter(X[y == -1, 0], X[y == -1, 1], c='blue', s=50, alpha=0.6, label='Class -1')
        
        # Highlight case-specific points
        case_X = X[indices]
        case_slack = [slack_variables[i] for i in indices]
        
        ax.scatter(case_X[:, 0], case_X[:, 1], s=150, c=color, marker=marker,
                  edgecolors='black', linewidth=2, alpha=0.8, label=f'Case {idx+1}')
        
        # Add slack variable annotations
        for x, slack in zip(case_X, case_slack):
            ax.annotate(f'$\\xi={slack:.2f}$', (x[0], x[1]), 
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, bbox=dict(boxstyle="round,pad=0.2", 
                                             fc="white", ec="black", alpha=0.8))
        
        # Plot decision boundary and margins
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        
        Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['blue', 'black', 'red'], 
                   linestyles=['--', '-', '--'], linewidths=[2, 3, 2])
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'slack_variables_detailed_cases.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_mathematical_derivation():
    """Create visualization showing mathematical derivation of slack variables"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create a simple 1D example
    x = np.linspace(-3, 3, 100)
    decision_boundary = 0
    margin_upper = 1
    margin_lower = -1
    
    # Plot decision boundary and margins
    ax.axhline(y=decision_boundary, color='black', linestyle='-', linewidth=3, label='Decision Boundary')
    ax.axhline(y=margin_upper, color='red', linestyle='--', linewidth=2, label='Upper Margin')
    ax.axhline(y=margin_lower, color='blue', linestyle='--', linewidth=2, label='Lower Margin')
    
    # Add sample points
    points = {
        '$\\xi = 0$': [(2.5, 1), (-2.5, -1)],
        '$0 < \\xi \\leq 1$': [(1.5, 1), (-1.5, -1)],
        '$\\xi > 1$': [(0.5, 1), (-0.5, -1)]
    }
    
    colors = {'$\\xi = 0$': 'green', '$0 < \\xi \\leq 1$': 'orange', '$\\xi > 1$': 'red'}
    markers = {'$\\xi = 0$': 'o', '$0 < \\xi \\leq 1$': 's', '$\\xi > 1$': '^'}
    
    for case, point_list in points.items():
        for x_pos, y_true in point_list:
            # Calculate slack variable
            if y_true * x_pos >= 1:
                slack = 0
            else:
                slack = 1 - y_true * x_pos
            
            ax.scatter(x_pos, x_pos, s=100, c=colors[case], marker=markers[case],
                      edgecolors='black', linewidth=2, alpha=0.8, label=f'{case} ($\\xi={slack:.2f}$)')
            
            # Add annotation
            ax.annotate(f'$\\xi={slack:.2f}$', (x_pos, x_pos), 
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, bbox=dict(boxstyle="round,pad=0.2", 
                                             fc="white", ec="black", alpha=0.8))
    
    # Add mathematical formulas
    formulas = [
        r'$y_i(w \cdot x_i + b) \geq 1 - \xi_i$',
        r'$\xi_i \geq 0$',
        r'$\xi_i = 0$: Correctly classified with margin',
        r'$0 < \xi_i \leq 1$: Inside margin',
        r'$\xi_i > 1$: Misclassified'
    ]
    
    for i, formula in enumerate(formulas):
        ax.text(0.02, 0.95 - i*0.05, formula, transform=ax.transAxes, 
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", 
                                      fc="white", ec="black", alpha=0.9))
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('$x$')
    ax.set_ylabel(r'$f(x) = w \cdot x + b$')
    ax.set_title('Mathematical Interpretation of Slack Variables', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'slack_variables_mathematical_derivation.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_comprehensive_summary():
    """Create a comprehensive summary visualization addressing all tasks"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Create synthetic data for clear demonstration
    np.random.seed(42)
    
    # Generate data with clear separation and some overlap
    X_pos = np.random.multivariate_normal([2, 2], [[1, 0.5], [0.5, 1]], 20)
    X_neg = np.random.multivariate_normal([-2, -2], [[1, 0.5], [0.5, 1]], 20)
    
    # Add some overlapping points
    X_overlap = np.array([
        [0.5, 0.5],   # Will be inside margin
        [-0.5, -0.5], # Will be inside margin
        [1.5, -0.5],  # Will be misclassified
        [-1.5, 0.5]   # Will be misclassified
    ])
    
    X = np.vstack([X_pos, X_neg, X_overlap])
    y = np.hstack([np.ones(20), -np.ones(20), np.array([1, -1, 1, -1])])
    
    # Train SVM
    svm = SVC(kernel='linear', C=1.0, random_state=42)
    svm.fit(X, y)
    
    # Calculate slack variables
    slack_variables = []
    for i in range(len(X)):
        x = X[i]
        label = y[i]
        decision_value = svm.decision_function([x])[0]
        if label * decision_value >= 1:
            slack = 0
        else:
            slack = 1 - label * decision_value
        slack_variables.append(slack)
    
    slack_variables = np.array(slack_variables)
    
    # Plot 1: Overall visualization with all cases
    ax1.scatter(X[y == 1, 0], X[y == 1, 1], c='red', s=50, alpha=0.6, label='Class +1')
    ax1.scatter(X[y == -1, 0], X[y == -1, 1], c='blue', s=50, alpha=0.6, label='Class -1')
    
    # Highlight support vectors
    support_vectors = svm.support_vectors_
    ax1.scatter(support_vectors[:, 0], support_vectors[:, 1], 
               s=200, facecolors='none', edgecolors='black', linewidth=2, 
               label='Support Vectors')
    
    # Plot decision boundary and margins
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax1.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['blue', 'black', 'red'], 
               linestyles=['--', '-', '--'], linewidths=[2, 3, 2])
    
    # Highlight points by slack variable value
    for i in range(len(X)):
        x = X[i]
        slack = slack_variables[i]
        
        if slack == 0:
            ax1.scatter(x[0], x[1], s=100, c='green', marker='o', 
                      edgecolors='black', linewidth=2, alpha=0.8)
            ax1.annotate(f'$\\xi={slack:.2f}$', (x[0], x[1]), 
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=8, bbox=dict(boxstyle="round,pad=0.2", 
                                             fc="lightgreen", ec="black", alpha=0.8))
        elif 0 < slack <= 1:
            ax1.scatter(x[0], x[1], s=100, c='orange', marker='s', 
                      edgecolors='black', linewidth=2, alpha=0.8)
            ax1.annotate(f'$\\xi={slack:.2f}$', (x[0], x[1]), 
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=8, bbox=dict(boxstyle="round,pad=0.2", 
                                             fc="lightyellow", ec="black", alpha=0.8))
        else:
            ax1.scatter(x[0], x[1], s=100, c='red', marker='^', 
                      edgecolors='black', linewidth=2, alpha=0.8)
            ax1.annotate(f'$\\xi={slack:.2f}$', (x[0], x[1]), 
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=8, bbox=dict(boxstyle="round,pad=0.2", 
                                             fc="lightcoral", ec="black", alpha=0.8))
    
    ax1.set_title('Task 1: Purpose of Slack Variables\n(Handle Non-Separable Data)', fontsize=12)
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Case 1 - ξ = 0
    case1_indices = np.where(slack_variables == 0)[0]
    ax2.scatter(X[y == 1, 0], X[y == 1, 1], c='red', s=30, alpha=0.3)
    ax2.scatter(X[y == -1, 0], X[y == -1, 1], c='blue', s=30, alpha=0.3)
    ax2.scatter(X[case1_indices, 0], X[case1_indices, 1], s=150, c='green', marker='o',
               edgecolors='black', linewidth=2, alpha=0.8, label='$\\xi = 0$')
    
    ax2.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['blue', 'black', 'red'], 
               linestyles=['--', '-', '--'], linewidths=[2, 3, 2])
    
    for idx in case1_indices:
        x = X[idx]
        slack = slack_variables[idx]
        ax2.annotate(f'$\\xi={slack:.2f}$', (x[0], x[1]), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.2", 
                                         fc="lightgreen", ec="black", alpha=0.8))
    
    ax2.set_title('Task 2a: $\\xi_i = 0$\n(Correctly classified with margin)', fontsize=12)
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Case 2 - 0 < ξ ≤ 1
    case2_indices = np.where((slack_variables > 0) & (slack_variables <= 1))[0]
    ax3.scatter(X[y == 1, 0], X[y == 1, 1], c='red', s=30, alpha=0.3)
    ax3.scatter(X[y == -1, 0], X[y == -1, 1], c='blue', s=30, alpha=0.3)
    ax3.scatter(X[case2_indices, 0], X[case2_indices, 1], s=150, c='orange', marker='s',
               edgecolors='black', linewidth=2, alpha=0.8, label='$0 < \\xi \\leq 1$')
    
    ax3.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['blue', 'black', 'red'], 
               linestyles=['--', '-', '--'], linewidths=[2, 3, 2])
    
    for idx in case2_indices:
        x = X[idx]
        slack = slack_variables[idx]
        ax3.annotate(f'$\\xi={slack:.2f}$', (x[0], x[1]), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.2", 
                                         fc="lightyellow", ec="black", alpha=0.8))
    
    ax3.set_title('Task 2b: $0 < \\xi_i \\leq 1$\n(Inside margin, correctly classified)', fontsize=12)
    ax3.set_xlabel('$x_1$')
    ax3.set_ylabel('$x_2$')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Case 3 - ξ > 1
    case3_indices = np.where(slack_variables > 1)[0]
    ax4.scatter(X[y == 1, 0], X[y == 1, 1], c='red', s=30, alpha=0.3)
    ax4.scatter(X[y == -1, 0], X[y == -1, 1], c='blue', s=30, alpha=0.3)
    ax4.scatter(X[case3_indices, 0], X[case3_indices, 1], s=150, c='red', marker='^',
               edgecolors='black', linewidth=2, alpha=0.8, label='$\\xi > 1$')
    
    ax4.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['blue', 'black', 'red'], 
               linestyles=['--', '-', '--'], linewidths=[2, 3, 2])
    
    for idx in case3_indices:
        x = X[idx]
        slack = slack_variables[idx]
        ax4.annotate(f'$\\xi={slack:.2f}$', (x[0], x[1]), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.2", 
                                         fc="lightcoral", ec="black", alpha=0.8))
    
    ax4.set_title('Task 2c: $\\xi_i > 1$\n(Misclassified)', fontsize=12)
    ax4.set_xlabel('$x_1$')
    ax4.set_ylabel('$x_2$')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'slack_variables_comprehensive_summary.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()

def print_detailed_analysis():
    """Print detailed analysis of slack variables"""
    
    print("=" * 80)
    print("SOFT-MARGIN SVM: GEOMETRIC INTERPRETATION OF SLACK VARIABLES")
    print("=" * 80)
    
    print("\n1. PURPOSE OF SLACK VARIABLES:")
    print("-" * 50)
    print("Slack variables (ξ_i ≥ 0) are introduced in soft-margin SVM to handle")
    print("data that is not perfectly linearly separable. They allow the SVM to")
    print("violate the margin constraints while penalizing such violations in")
    print("the objective function.")
    print()
    print("Mathematical formulation:")
    print("minimize: (1/2)||w||² + C∑ξ_i")
    print("subject to: y_i(w·x_i + b) ≥ 1 - ξ_i")
    print("           ξ_i ≥ 0 for all i")
    
    print("\n2. GEOMETRIC INTERPRETATION:")
    print("-" * 50)
    
    print("\nCase 1: ξ_i = 0")
    print("• Point is correctly classified with margin")
    print("• y_i(w·x_i + b) ≥ 1")
    print("• Point is outside or exactly on the margin")
    print("• No penalty in the objective function")
    
    print("\nCase 2: 0 < ξ_i ≤ 1")
    print("• Point violates the margin but is still correctly classified")
    print("• 0 < y_i(w·x_i + b) < 1")
    print("• Point is inside the margin")
    print("• Penalty proportional to ξ_i in the objective function")
    
    print("\nCase 3: ξ_i > 1")
    print("• Point is misclassified")
    print("• y_i(w·x_i + b) ≤ 0")
    print("• Point is on the wrong side of the decision boundary")
    print("• Larger penalty in the objective function")
    
    print("\n3. RELATIONSHIP TO CLASSIFICATION:")
    print("-" * 50)
    print("• ξ_i = 0: Correctly classified with margin")
    print("• 0 < ξ_i ≤ 1: Correctly classified but inside margin")
    print("• ξ_i > 1: Misclassified")
    
    print("\n4. PRACTICAL IMPLICATIONS:")
    print("-" * 50)
    print("• C parameter controls the trade-off between margin size and")
    print("  classification errors")
    print("• Large C: Small margin, few misclassifications (close to hard margin)")
    print("• Small C: Large margin, more misclassifications allowed")
    print("• Slack variables enable SVM to handle noisy and non-separable data")

if __name__ == "__main__":
    print("Creating comprehensive visualization of soft-margin SVM slack variables...")
    
    # Create main visualization
    X, y, slack_variables, svm = create_svm_visualization()
    
    # Create detailed case analysis
    create_detailed_case_analysis()
    
    # Create mathematical derivation
    create_mathematical_derivation()
    
    # Create comprehensive summary addressing all tasks
    create_comprehensive_summary()
    
    # Print detailed analysis
    print_detailed_analysis()
    
    print(f"\nAll visualizations saved to: {save_dir}")
    print("Generated files:")
    print("1. slack_variables_geometric_interpretation.png")
    print("2. slack_variables_detailed_cases.png") 
    print("3. slack_variables_mathematical_derivation.png")
    print("4. slack_variables_comprehensive_summary.png")
