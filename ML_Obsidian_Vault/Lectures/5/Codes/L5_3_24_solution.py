import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_3_Quiz_24")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("Question 24: Comparing Kernel Margins in SVM")
print("=" * 60)

# Set random seed for reproducibility
np.random.seed(42)

def calculate_margin(svm_model, X, y):
    """
    Calculate the geometric margin for an SVM model
    """
    # Get support vectors
    support_vectors = svm_model.support_vectors_
    support_vector_indices = svm_model.support_
    
    # Calculate decision function values for support vectors
    decision_values = svm_model.decision_function(support_vectors)
    
    # For linear kernel, we can calculate geometric margin
    if svm_model.kernel == 'linear':
        margins = np.abs(decision_values) / np.linalg.norm(svm_model.coef_[0])
    else:
        # For non-linear kernels, we use functional margin (decision function values)
        # This is a proxy for geometric margin in the feature space
        margins = np.abs(decision_values)
    
    return np.min(margins), np.mean(margins), np.max(margins)

def plot_decision_boundary_and_margins(X, y, svm_model, title, filename):
    """
    Plot decision boundary and margins for SVM model
    """
    plt.figure(figsize=(12, 10))
    
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Get decision function values
    Z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and margins
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['red', 'black', 'blue'], 
                linestyles=['--', '-', '--'], linewidths=[2, 3, 2])
    
    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, 
                edgecolors='black', s=50, alpha=0.8)
    
    # Highlight support vectors
    support_vectors = svm_model.support_vectors_
    if len(support_vectors) > 0:
        plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
                    s=200, facecolors='none', edgecolors='orange', 
                    linewidth=2, label=f'Support Vectors ({len(support_vectors)})')
    
    # Add margin annotations
    plt.fill_between([], [], [], color='red', alpha=0.2, label='Negative Margin')
    plt.fill_between([], [], [], color='blue', alpha=0.2, label='Positive Margin')
    
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Save plot
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to prevent it from opening

def compare_kernels_on_dataset(X, y, dataset_name):
    """
    Compare different kernels on the same dataset
    """
    print(f"\n{dataset_name}")
    print("-" * 40)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Define kernels to compare
    kernels = {
        'linear': {'kernel': 'linear', 'C': 1.0},
        'rbf': {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
        'poly2': {'kernel': 'poly', 'C': 1.0, 'degree': 2, 'coef0': 1},
        'poly3': {'kernel': 'poly', 'C': 1.0, 'degree': 3, 'coef0': 1}
    }
    
    results = {}
    
    for kernel_name, params in kernels.items():
        print(f"\n{kernel_name.upper()} Kernel:")
        
        # Train SVM
        svm = SVC(**params, random_state=42)
        svm.fit(X_train, y_train)
        
        # Calculate margins
        min_margin, mean_margin, max_margin = calculate_margin(svm, X_train, y_train)
        
        # Calculate accuracies
        train_acc = accuracy_score(y_train, svm.predict(X_train))
        test_acc = accuracy_score(y_test, svm.predict(X_test))
        
        # Store results
        results[kernel_name] = {
            'min_margin': min_margin,
            'mean_margin': mean_margin,
            'max_margin': max_margin,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'n_support_vectors': len(svm.support_vectors_),
            'model': svm
        }
        
        print(f"  Min Margin: {min_margin:.4f}")
        print(f"  Mean Margin: {mean_margin:.4f}")
        print(f"  Max Margin: {max_margin:.4f}")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Support Vectors: {len(svm.support_vectors_)}")
        
        # Plot decision boundary
        plot_decision_boundary_and_margins(
            X_train, y_train, svm,
            f'{kernel_name.upper()} Kernel - {dataset_name}',
            f'{dataset_name.lower().replace(" ", "_")}_{kernel_name}_boundary.png'
        )
    
    return results

def create_comparison_plots(results_list, dataset_names):
    """
    Create comparison plots for margin vs accuracy
    """
    # Prepare data for plotting
    kernel_names = list(results_list[0].keys())
    metrics = ['min_margin', 'mean_margin', 'max_margin', 'train_acc', 'test_acc']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        x_pos = np.arange(len(kernel_names))
        width = 0.35
        
        for j, (results, dataset_name) in enumerate(zip(results_list, dataset_names)):
            values = [results[kernel][metric] for kernel in kernel_names]
            ax.bar(x_pos + j*width, values, width, label=dataset_name, alpha=0.8)
        
        ax.set_xlabel('Kernel Type')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
        ax.set_xticks(x_pos + width/2)
        ax.set_xticklabels([k.upper() for k in kernel_names], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove the last subplot if not needed
    if len(metrics) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'kernel_comparison_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to prevent it from opening

def demonstrate_margin_scale_issues():
    """
    Demonstrate how margin values can be misleading due to different scales
    """
    print("\n" + "="*60)
    print("DEMONSTRATION: Why Raw Margin Values Can Be Misleading")
    print("="*60)
    
    # Create synthetic data
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                              n_informative=2, n_clusters_per_class=1, 
                              random_state=42)
    
    # Scale the data differently
    X_scaled_1 = X * 1.0  # Original scale
    X_scaled_2 = X * 10.0  # 10x larger scale
    X_scaled_3 = X * 0.1   # 10x smaller scale
    
    datasets = [
        (X_scaled_1, y, "Original Scale"),
        (X_scaled_2, y, "10x Larger Scale"),
        (X_scaled_3, y, "10x Smaller Scale")
    ]
    
    print("\nTraining linear SVM on same data with different scales:")
    print("-" * 60)
    
    for X_data, y_data, scale_name in datasets:
        print(f"\n{scale_name}:")
        
        # Train linear SVM
        svm = SVC(kernel='linear', C=1.0, random_state=42)
        svm.fit(X_data, y_data)
        
        # Calculate margins
        min_margin, mean_margin, max_margin = calculate_margin(svm, X_data, y_data)
        
        # Calculate accuracies
        train_acc = accuracy_score(y_data, svm.predict(X_data))
        
        print(f"  Min Margin: {min_margin:.6f}")
        print(f"  Mean Margin: {mean_margin:.6f}")
        print(f"  Max Margin: {max_margin:.6f}")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Data Scale Factor: {np.std(X_data):.2f}")
        
        # Plot
        plot_decision_boundary_and_margins(
            X_data, y_data, svm,
            f'Linear SVM - {scale_name}',
            f'scale_comparison_{scale_name.lower().replace(" ", "_")}.png'
        )

def demonstrate_kernel_specific_issues():
    """
    Demonstrate kernel-specific issues with margin interpretation
    """
    print("\n" + "="*60)
    print("DEMONSTRATION: Kernel-Specific Margin Issues")
    print("="*60)
    
    # Create non-linearly separable data (circles)
    X, y = make_circles(n_samples=200, noise=0.2, factor=0.5, random_state=42)
    
    kernels = {
        'linear': {'kernel': 'linear', 'C': 1.0},
        'rbf': {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
        'poly2': {'kernel': 'poly', 'C': 1.0, 'degree': 2, 'coef0': 1}
    }
    
    print("\nComparing kernels on non-linearly separable data:")
    print("-" * 60)
    
    for kernel_name, params in kernels.items():
        print(f"\n{kernel_name.upper()} Kernel:")
        
        # Train SVM
        svm = SVC(**params, random_state=42)
        svm.fit(X, y)
        
        # Calculate margins
        min_margin, mean_margin, max_margin = calculate_margin(svm, X, y)
        
        # Calculate accuracy
        train_acc = accuracy_score(y, svm.predict(X))
        
        print(f"  Min Margin: {min_margin:.6f}")
        print(f"  Mean Margin: {mean_margin:.6f}")
        print(f"  Max Margin: {max_margin:.6f}")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Support Vectors: {len(svm.support_vectors_)}")
        
        # Plot
        plot_decision_boundary_and_margins(
            X, y, svm,
            f'{kernel_name.upper()} Kernel - Circles Dataset',
            f'circles_{kernel_name}_boundary.png'
        )

# Main execution
if __name__ == "__main__":
    print("Question 24: Comparing Kernel Margins in SVM")
    print("Analyzing why raw margin values can be misleading...")
    
    # Generate different datasets
    print("\n1. LINEARLY SEPARABLE DATASET")
    X_linear, y_linear = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                                           n_informative=2, n_clusters_per_class=1, 
                                           random_state=42)
    
    print("\n2. NON-LINEARLY SEPARABLE DATASET (XOR-like)")
    X_nonlinear, y_nonlinear = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                                                 n_informative=2, n_clusters_per_class=2, 
                                                 random_state=42)
    
    # Compare kernels on both datasets
    results_linear = compare_kernels_on_dataset(X_linear, y_linear, "Linearly Separable")
    results_nonlinear = compare_kernels_on_dataset(X_nonlinear, y_nonlinear, "Non-linearly Separable")
    
    # Create comparison plots
    create_comparison_plots([results_linear, results_nonlinear], 
                           ["Linearly Separable", "Non-linearly Separable"])
    
    # Demonstrate scale issues
    demonstrate_margin_scale_issues()
    
    # Demonstrate kernel-specific issues
    demonstrate_kernel_specific_issues()
    
    print(f"\nAll plots saved to: {save_dir}")
    print("\nAnalysis complete!")
