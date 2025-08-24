"""
Lecture 5.3 Quiz - Question 10: Kernel Selection Methodology
Develop a systematic kernel selection methodology.

Tasks:
1. List the factors to consider when choosing between linear, polynomial, and RBF kernels
2. Design a decision tree for kernel selection based on dataset characteristics
3. How would you use cross-validation to compare different kernel families?
4. What is kernel alignment and how can it guide kernel selection?
5. Create a practical algorithm for automated kernel selection
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, make_circles, make_moons
from sklearn.metrics import accuracy_score
import seaborn as sns
import pandas as pd
from itertools import product

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['text.usetex'] = False  # Disable LaTeX for compatibility
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

def create_output_directory():
    """Create directory for saving plots"""
    import os
    os.makedirs('../Images/L5_3_Quiz_10', exist_ok=True)

def generate_test_datasets():
    """Generate various synthetic datasets for kernel testing"""
    np.random.seed(42)

    datasets = {}

    # Linear separable data
    X_linear, y_linear = make_classification(
        n_samples=200, n_features=2, n_redundant=0, n_informative=2,
        n_clusters_per_class=1, random_state=42
    )
    datasets['linear'] = (X_linear, y_linear, "Linear Separable")

    # Non-linear but polynomial separable
    X_poly, y_poly = make_classification(
        n_samples=200, n_features=2, n_redundant=0, n_informative=2,
        n_clusters_per_class=2, random_state=42
    )
    datasets['polynomial'] = (X_poly, y_poly, "Polynomial Separable")

    # Circular data (RBF suitable)
    X_circles, y_circles = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)
    datasets['circles'] = (X_circles, y_circles, "Circular Pattern")

    # Moon-shaped data (RBF suitable)
    X_moons, y_moons = make_moons(n_samples=200, noise=0.1, random_state=42)
    datasets['moons'] = (X_moons, y_moons, "Moon Pattern")

    # High-dimensional sparse data
    X_sparse, y_sparse = make_classification(
        n_samples=200, n_features=20, n_informative=5, n_redundant=0,
        n_clusters_per_class=1, random_state=42
    )
    datasets['sparse'] = (X_sparse, y_sparse, "High-Dim Sparse")

    return datasets

def analyze_dataset_characteristics(X, y):
    """Analyze dataset characteristics to guide kernel selection"""
    characteristics = {}

    # Basic statistics
    characteristics['n_samples'] = X.shape[0]
    characteristics['n_features'] = X.shape[1]
    characteristics['n_classes'] = len(np.unique(y))

    # Dimensionality ratio
    characteristics['sample_to_feature_ratio'] = X.shape[0] / X.shape[1]

    # Data distribution
    characteristics['feature_variance'] = np.var(X, axis=0).mean()
    characteristics['feature_range'] = np.ptp(X, axis=0).mean()

    # Class balance
    class_counts = np.bincount(y)
    characteristics['class_balance'] = min(class_counts) / max(class_counts)

    # Linear separability estimate (using linear SVM)
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        linear_svm = SVC(kernel='linear', C=1.0)
        linear_score = cross_val_score(linear_svm, X_scaled, y, cv=3).mean()
        characteristics['linear_separability_score'] = linear_score
    except:
        characteristics['linear_separability_score'] = 0.0

    # Sparsity
    characteristics['sparsity'] = np.mean(X == 0)

    return characteristics

def kernel_alignment(K, y):
    """
    Compute kernel alignment score.
    Measures how well kernel matrix aligns with ideal kernel based on labels.
    """
    n = len(y)

    # Create ideal kernel matrix (1 if same class, 0 if different class)
    Y = np.outer(y, y)
    Y_ideal = (Y == Y.T).astype(float)

    # Center both kernels
    ones = np.ones((n, n)) / n
    K_centered = K - ones @ K - K @ ones + ones @ K @ ones
    Y_centered = Y_ideal - ones @ Y_ideal - Y_ideal @ ones + ones @ Y_ideal @ ones

    # Compute alignment
    numerator = np.trace(K_centered @ Y_centered)
    denominator = np.sqrt(np.trace(K_centered @ K_centered) * np.trace(Y_centered @ Y_centered))

    if denominator == 0:
        return 0

    return numerator / denominator

def evaluate_kernels_cv(X, y, kernels_params):
    """Evaluate different kernels using cross-validation"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for kernel_name, params in kernels_params.items():
        scores = []
        alignments = []

        for param_set in params:
            svm = SVC(**param_set)

            # Cross-validation score
            cv_scores = cross_val_score(svm, X_scaled, y, cv=cv, scoring='accuracy')
            mean_score = cv_scores.mean()

            # Kernel alignment
            svm.fit(X_scaled, y)
            if hasattr(svm, 'dual_coef_'):
                # Compute kernel matrix for alignment
                if kernel_name == 'linear':
                    K = X_scaled @ X_scaled.T
                elif kernel_name == 'rbf':
                    gamma = param_set.get('gamma', 'scale')
                    if gamma == 'scale':
                        gamma = 1.0 / (X_scaled.shape[1] * X_scaled.var())
                    K = np.exp(-gamma * np.sum((X_scaled[:, None] - X_scaled[None, :])**2, axis=2))
                elif kernel_name == 'poly':
                    degree = param_set.get('degree', 3)
                    coef0 = param_set.get('coef0', 0.0)
                    K = (X_scaled @ X_scaled.T + coef0) ** degree

                alignment = kernel_alignment(K, y)
                alignments.append(alignment)
            else:
                alignments.append(0)

            scores.append(mean_score)

        results[kernel_name] = {
            'scores': scores,
            'alignments': alignments,
            'best_score': max(scores),
            'best_alignment': max(alignments) if alignments else 0,
            'params': params
        }

    return results

def create_decision_tree_rules():
    """Create decision tree rules for kernel selection"""
    rules = [
        {
            'condition': lambda chars: chars['n_features'] > chars['n_samples'],
            'recommendation': 'linear',
            'reason': 'High-dimensional data with few samples - linear kernel prevents overfitting'
        },
        {
            'condition': lambda chars: chars['linear_separability_score'] > 0.9,
            'recommendation': 'linear',
            'reason': 'Data appears linearly separable - linear kernel is sufficient'
        },
        {
            'condition': lambda chars: chars['sparsity'] > 0.5,
            'recommendation': 'linear',
            'reason': 'Sparse data - linear kernel works well with sparse features'
        },
        {
            'condition': lambda chars: chars['n_features'] <= 10 and chars['linear_separability_score'] < 0.7,
            'recommendation': 'rbf',
            'reason': 'Low-dimensional non-linear data - RBF kernel can capture complex patterns'
        },
        {
            'condition': lambda chars: chars['sample_to_feature_ratio'] > 10 and chars['linear_separability_score'] < 0.8,
            'recommendation': 'polynomial',
            'reason': 'Sufficient samples for polynomial kernel - can capture polynomial relationships'
        },
        {
            'condition': lambda chars: True,  # Default case
            'recommendation': 'rbf',
            'reason': 'Default choice - RBF kernel is versatile for most problems'
        }
    ]
    return rules

def automated_kernel_selection(X, y):
    """Automated kernel selection algorithm"""
    print(f"\n=== Automated Kernel Selection ===")

    # Step 1: Analyze dataset characteristics
    characteristics = analyze_dataset_characteristics(X, y)
    print(f"Dataset characteristics:")
    for key, value in characteristics.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")

    # Step 2: Apply decision tree rules
    rules = create_decision_tree_rules()
    recommended_kernel = None
    reason = ""

    for rule in rules:
        if rule['condition'](characteristics):
            recommended_kernel = rule['recommendation']
            reason = rule['reason']
            break

    print(f"\nRule-based recommendation: {recommended_kernel}")
    print(f"Reason: {reason}")

    # Step 3: Empirical validation with cross-validation
    kernels_params = {
        'linear': [{'kernel': 'linear', 'C': C} for C in [0.1, 1.0, 10.0]],
        'rbf': [{'kernel': 'rbf', 'C': C, 'gamma': gamma}
                for C in [0.1, 1.0, 10.0] for gamma in [0.001, 0.01, 0.1, 1.0]],
        'poly': [{'kernel': 'poly', 'degree': d, 'C': C, 'coef0': coef0}
                 for d in [2, 3] for C in [0.1, 1.0, 10.0] for coef0 in [0.0, 1.0]]
    }

    print(f"\nEvaluating kernels with cross-validation...")
    results = evaluate_kernels_cv(X, y, kernels_params)

    # Step 4: Select best kernel based on CV score and alignment
    best_kernel = None
    best_score = 0
    best_alignment = 0

    print(f"\nCross-validation results:")
    for kernel_name, result in results.items():
        print(f"  {kernel_name}: Best CV Score = {result['best_score']:.3f}, "
              f"Best Alignment = {result['best_alignment']:.3f}")

        # Combined score (weighted average of CV score and alignment)
        combined_score = 0.7 * result['best_score'] + 0.3 * result['best_alignment']
        if combined_score > best_score:
            best_score = combined_score
            best_kernel = kernel_name
            best_alignment = result['best_alignment']

    print(f"\nFinal recommendation: {best_kernel}")
    print(f"Combined score: {best_score:.3f} (CV: {results[best_kernel]['best_score']:.3f}, "
          f"Alignment: {results[best_kernel]['best_alignment']:.3f})")

    return {
        'rule_based': recommended_kernel,
        'empirical_best': best_kernel,
        'characteristics': characteristics,
        'cv_results': results
    }

def visualize_kernel_selection_analysis():
    """Visualize kernel selection methodology and results"""
    create_output_directory()

    # Generate test datasets
    datasets = generate_test_datasets()

    # Analyze each dataset
    results_summary = []

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()

    for idx, (dataset_name, (X, y, title)) in enumerate(datasets.items()):
        # Plot dataset
        ax = axes[idx]
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
        ax.set_title(f'{title}\n({dataset_name})')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')

        # Run automated kernel selection
        selection_result = automated_kernel_selection(X, y)

        # Store results
        results_summary.append({
            'dataset': dataset_name,
            'title': title,
            'rule_based': selection_result['rule_based'],
            'empirical_best': selection_result['empirical_best'],
            'n_samples': selection_result['characteristics']['n_samples'],
            'n_features': selection_result['characteristics']['n_features'],
            'linear_score': selection_result['characteristics']['linear_separability_score']
        })

        # Add text annotation with recommendations
        ax.text(0.02, 0.98, f"Rule: {selection_result['rule_based']}\nCV: {selection_result['empirical_best']}",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Create decision tree visualization
    ax = axes[5]
    ax.axis('off')
    ax.set_title('Kernel Selection Decision Tree')

    # Decision tree text
    tree_text = """
    Decision Tree Rules:

    1. n_features > n_samples?
       → LINEAR (prevent overfitting)

    2. Linear separability > 0.9?
       → LINEAR (sufficient)

    3. Sparsity > 0.5?
       → LINEAR (sparse data)

    4. Low-dim & non-linear?
       → RBF (complex patterns)

    5. Many samples & polynomial?
       → POLYNOMIAL (interactions)

    6. Default → RBF (versatile)
    """
    ax.text(0.1, 0.9, tree_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')

    # Create summary table
    ax = axes[6]
    ax.axis('off')
    ax.set_title('Kernel Selection Summary')

    # Convert results to DataFrame for better visualization
    df = pd.DataFrame(results_summary)
    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            row['dataset'],
            f"{row['n_samples']}×{row['n_features']}",
            f"{row['linear_score']:.2f}",
            row['rule_based'],
            row['empirical_best']
        ])

    table = ax.table(cellText=table_data,
                     colLabels=['Dataset', 'Size', 'Linear Score', 'Rule-Based', 'CV-Best'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Kernel alignment visualization
    ax = axes[7]
    ax.set_title('Kernel Alignment Concept')

    # Create example kernel matrices
    n = 4
    y_example = np.array([0, 0, 1, 1])

    # Good alignment (block structure)
    K_good = np.array([[1.0, 0.8, 0.2, 0.1],
                       [0.8, 1.0, 0.1, 0.2],
                       [0.2, 0.1, 1.0, 0.9],
                       [0.1, 0.2, 0.9, 1.0]])

    im = ax.imshow(K_good, cmap='viridis')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(['C1', 'C1', 'C2', 'C2'])
    ax.set_yticklabels(['C1', 'C1', 'C2', 'C2'])

    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Cross-validation strategy
    ax = axes[8]
    ax.set_title('Cross-Validation Strategy')
    ax.axis('off')

    cv_text = """
    Cross-Validation Protocol:

    1. Stratified K-Fold (K=5)
    2. Grid search over parameters:
       • Linear: C ∈ [0.1, 1, 10]
       • RBF: C ∈ [0.1, 1, 10]
             γ ∈ [0.001, 0.01, 0.1, 1]
       • Poly: degree ∈ [2, 3]
              C ∈ [0.1, 1, 10]
              coef0 ∈ [0, 1]

    3. Combine CV score + alignment:
       Score = 0.7 × CV + 0.3 × Alignment
    """
    ax.text(0.1, 0.9, cv_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig('../Images/L5_3_Quiz_10/kernel_selection_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    return results_summary

def demonstrate_kernel_factors():
    """Demonstrate factors affecting kernel choice"""
    print("=== Factors for Kernel Selection ===")

    factors = {
        "Dataset Size": {
            "Small (n < 1000)": "Linear or RBF - avoid overfitting",
            "Medium (1000 < n < 10000)": "All kernels viable - use CV",
            "Large (n > 10000)": "Linear preferred - computational efficiency"
        },
        "Dimensionality": {
            "Low-dim (d < 10)": "RBF or Polynomial - capture non-linearity",
            "Medium-dim (10 < d < 100)": "RBF or Linear - balance complexity",
            "High-dim (d > 100)": "Linear - curse of dimensionality"
        },
        "Data Characteristics": {
            "Linear separable": "Linear kernel sufficient",
            "Non-linear patterns": "RBF or Polynomial kernel",
            "Sparse features": "Linear kernel effective",
            "Dense features": "RBF kernel versatile"
        },
        "Computational Resources": {
            "Limited memory": "Linear kernel - O(n) space",
            "Limited time": "Linear kernel - O(n²) training",
            "Abundant resources": "RBF with grid search"
        },
        "Interpretability": {
            "High importance": "Linear kernel - interpretable weights",
            "Medium importance": "Polynomial - feature interactions",
            "Low importance": "RBF - black box acceptable"
        }
    }

    for category, subcategories in factors.items():
        print(f"\n{category}:")
        for condition, recommendation in subcategories.items():
            print(f"  • {condition}: {recommendation}")

    return factors

def create_kernel_selection_flowchart():
    """Create detailed kernel selection methodology flowchart"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Dataset characteristics analysis
    ax = axes[0, 0]

    # Sample data characteristics
    characteristics = ['n_samples', 'n_features', 'sparsity', 'linear_sep', 'class_balance']
    dataset_types = ['Small Dense', 'Large Sparse', 'High-Dim', 'Non-linear']

    char_matrix = np.array([
        [500, 10, 0.1, 0.9, 0.8],      # Small Dense
        [10000, 1000, 0.8, 0.7, 0.6],  # Large Sparse
        [1000, 500, 0.3, 0.5, 0.9],    # High-Dim
        [2000, 5, 0.0, 0.3, 0.7]       # Non-linear
    ])

    # Normalize for visualization
    char_matrix_norm = char_matrix / char_matrix.max(axis=0)

    im = ax.imshow(char_matrix_norm, cmap='RdYlBu_r', aspect='auto')
    ax.set_xticks(range(len(characteristics)))
    ax.set_xticklabels(characteristics, rotation=45)
    ax.set_yticks(range(len(dataset_types)))
    ax.set_yticklabels(dataset_types)
    ax.set_title(r'Dataset Characteristics Matrix')

    # Add text annotations
    for i in range(len(dataset_types)):
        for j in range(len(characteristics)):
            if j == 0 or j == 1:  # Integer values
                text = f'{char_matrix[i, j]:.0f}'
            else:  # Float values
                text = f'{char_matrix[i, j]:.1f}'
            ax.text(j, i, text, ha='center', va='center',
                   color='white' if char_matrix_norm[i, j] > 0.5 else 'black', fontweight='bold')

    # 2. Kernel performance comparison
    ax = axes[0, 1]

    kernels = ['Linear', 'Polynomial', 'RBF']
    metrics = ['Accuracy', 'Training Time', 'Memory Usage', 'Interpretability']

    # Performance scores (higher is better, normalized to [0,1])
    performance_matrix = np.array([
        [0.7, 0.9, 0.9, 1.0],  # Linear
        [0.8, 0.6, 0.7, 0.6],  # Polynomial
        [0.9, 0.4, 0.3, 0.2]   # RBF
    ])

    im = ax.imshow(performance_matrix, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45)
    ax.set_yticks(range(len(kernels)))
    ax.set_yticklabels(kernels)
    ax.set_title(r'Kernel Performance Matrix')

    # Add text annotations
    for i in range(len(kernels)):
        for j in range(len(metrics)):
            text = f'{performance_matrix[i, j]:.1f}'
            ax.text(j, i, text, ha='center', va='center',
                   color='white' if performance_matrix[i, j] < 0.5 else 'black', fontweight='bold')

    # 3. Decision boundary complexity visualization
    ax = axes[1, 0]

    # Generate sample data points
    np.random.seed(42)
    X_sample = np.random.randn(100, 2)

    # Create different decision boundaries
    x_range = np.linspace(-3, 3, 100)

    # Linear boundary
    linear_boundary = 0.5 * x_range

    # Polynomial boundary (degree 2)
    poly_boundary = 0.3 * x_range**2 - 1

    # RBF-like boundary (circular)
    theta = np.linspace(0, 2*np.pi, 100)
    rbf_x = 1.5 * np.cos(theta)
    rbf_y = 1.5 * np.sin(theta)

    ax.plot(x_range, linear_boundary, 'b-', linewidth=3, label='Linear', alpha=0.8)
    ax.plot(x_range, poly_boundary, 'g-', linewidth=3, label='Polynomial', alpha=0.8)
    ax.plot(rbf_x, rbf_y, 'r-', linewidth=3, label='RBF', alpha=0.8)

    ax.scatter(X_sample[:50, 0], X_sample[:50, 1], c='lightblue', alpha=0.6, s=30)
    ax.scatter(X_sample[50:, 0], X_sample[50:, 1], c='lightcoral', alpha=0.6, s=30)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel(r'Feature 1')
    ax.set_ylabel(r'Feature 2')
    ax.set_title(r'Decision Boundary Complexity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Computational complexity comparison
    ax = axes[1, 1]

    sample_sizes = np.array([100, 500, 1000, 5000, 10000])

    # Training time complexity (relative)
    linear_time = sample_sizes**2 * 0.001
    poly_time = sample_sizes**2 * 0.002
    rbf_time = sample_sizes**2 * 0.003

    ax.loglog(sample_sizes, linear_time, 'b-o', label='Linear', linewidth=2)
    ax.loglog(sample_sizes, poly_time, 'g-s', label='Polynomial', linewidth=2)
    ax.loglog(sample_sizes, rbf_time, 'r-^', label='RBF', linewidth=2)

    ax.set_xlabel(r'Number of Samples')
    ax.set_ylabel(r'Training Time (relative)')
    ax.set_title(r'Computational Complexity Scaling')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../Images/L5_3_Quiz_10/kernel_selection_flowchart.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to run all demonstrations"""
    print("Kernel Selection Methodology - Comprehensive Analysis")
    print("=" * 60)

    # Demonstrate factors affecting kernel choice
    factors = demonstrate_kernel_factors()

    # Visualize kernel selection analysis
    results = visualize_kernel_selection_analysis()

    # Create detailed flowchart
    create_kernel_selection_flowchart()

    print("\n" + "=" * 60)
    print("SUMMARY OF KERNEL SELECTION METHODOLOGY")
    print("=" * 60)

    print("\n1. Key Factors for Kernel Selection:")
    print("   • Dataset size and dimensionality")
    print("   • Linear separability of the data")
    print("   • Computational constraints")
    print("   • Interpretability requirements")
    print("   • Domain-specific characteristics")

    print("\n2. Decision Tree Approach:")
    print("   • Rule-based initial recommendation")
    print("   • Based on dataset characteristics")
    print("   • Fast and interpretable")

    print("\n3. Cross-Validation Validation:")
    print("   • Empirical performance comparison")
    print("   • Grid search over hyperparameters")
    print("   • Stratified K-fold for reliability")

    print("\n4. Kernel Alignment:")
    print("   • Measures kernel-label compatibility")
    print("   • Theoretical foundation for selection")
    print("   • Combines with CV for robust choice")

    print("\n5. Automated Algorithm:")
    print("   • Combines rule-based and empirical approaches")
    print("   • Weighted scoring: 70% CV + 30% alignment")
    print("   • Provides both fast and thorough evaluation")

    print(f"\nAnalysis complete! Results saved to ../Images/L5_3_Quiz_10/")

if __name__ == "__main__":
    main()