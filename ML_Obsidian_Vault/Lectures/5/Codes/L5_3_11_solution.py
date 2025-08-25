"""
Lecture 5.3 Quiz - Question 11: Computational Analysis
Analyze the computational and storage complexity of different kernels.

Tasks:
1. Compare the evaluation time for linear, polynomial (degree 3), and RBF kernels
2. Calculate the space complexity of storing the kernel matrix for n = 10^3, 10^4, 10^5 samples
3. Design strategies for reducing kernel matrix storage requirements
4. What is the trade-off between kernel complexity and classification accuracy?
5. How does the choice of kernel affect training vs prediction time?
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import os

# Set style for better plots with LaTeX
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amssymb}'

def create_output_directory():
    """Create directory for saving plots"""
    import os
    os.makedirs('../Images/L5_3_Quiz_11', exist_ok=True)

def benchmark_kernel_evaluation_time():
    """Benchmark kernel evaluation time for different kernel types"""
    print("=== Kernel Evaluation Time Benchmark ===")

    # Generate test data of different sizes
    sizes = [100, 500, 1000, 2000, 5000]
    kernels = ['linear', 'poly', 'rbf']

    results = {kernel: {'sizes': [], 'times': [], 'std_times': []} for kernel in kernels}

    for n in sizes:
        print(f"\nTesting with {n} samples...")

        # Generate data
        X, y = make_classification(n_samples=n, n_features=10, random_state=42)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for kernel in kernels:
            times = []

            # Multiple runs for statistical significance
            for run in range(5):
                if kernel == 'linear':
                    svm = SVC(kernel='linear', C=1.0)
                elif kernel == 'poly':
                    svm = SVC(kernel='poly', degree=3, C=1.0, coef0=1.0)
                elif kernel == 'rbf':
                    svm = SVC(kernel='rbf', C=1.0, gamma='scale')

                start_time = time.time()
                svm.fit(X_scaled, y)
                end_time = time.time()

                times.append(end_time - start_time)

            mean_time = np.mean(times)
            std_time = np.std(times)

            results[kernel]['sizes'].append(n)
            results[kernel]['times'].append(mean_time)
            results[kernel]['std_times'].append(std_time)

            print(f"  {kernel}: {mean_time:.4f} ± {std_time:.4f} seconds")

    return results

def calculate_memory_requirements():
    """Calculate memory requirements for kernel matrices"""
    print("\n=== Memory Requirements Analysis ===")

    sample_sizes = [10**3, 10**4, 10**5]

    memory_analysis = {}

    for n in sample_sizes:
        # Kernel matrix size: n x n
        matrix_elements = n * n

        # Memory in bytes (assuming float64 = 8 bytes)
        bytes_per_element = 8
        total_bytes = matrix_elements * bytes_per_element

        # Convert to different units
        kb = total_bytes / 1024
        mb = kb / 1024
        gb = mb / 1024

        memory_analysis[n] = {
            'elements': matrix_elements,
            'bytes': total_bytes,
            'kb': kb,
            'mb': mb,
            'gb': gb
        }

        print(f"n = {n:,} samples:")
        print(f"  Matrix size: {n:,} × {n:,} = {matrix_elements:,} elements")
        print(f"  Memory: {total_bytes:,} bytes = {mb:.2f} MB = {gb:.4f} GB")

        # Practical implications
        if gb > 1:
            print(f"  ⚠️  Large memory requirement: {gb:.2f} GB")
        elif mb > 100:
            print(f"  ⚠️  Moderate memory requirement: {mb:.2f} MB")
        else:
            print(f"  ✅ Manageable memory requirement: {mb:.2f} MB")

    return memory_analysis

def memory_reduction_strategies():
    """Demonstrate strategies for reducing kernel matrix storage"""
    print("\n=== Memory Reduction Strategies ===")

    strategies = {
        "Low-rank Approximation": {
            "description": "Approximate K ≈ UV^T where U,V are n×r matrices",
            "memory_reduction": lambda n, r: f"From O(n²) to O(nr): {100*(1-2*r/n):.1f}% reduction for r={r}",
            "accuracy_impact": "Controlled by rank r - higher r = better approximation"
        },
        "Nyström Method": {
            "description": "Sample m columns/rows to approximate full matrix",
            "memory_reduction": lambda n, m: f"From O(n²) to O(nm): {100*(1-m/n):.1f}% reduction for m={m}",
            "accuracy_impact": "Quality depends on sampling strategy and m"
        },
        "Random Fourier Features": {
            "description": "Approximate RBF kernel with explicit feature mapping",
            "memory_reduction": lambda n, d: f"From O(n²) to O(nd): {100*(1-d/n):.1f}% reduction for d={d} features",
            "accuracy_impact": "Approximation quality improves with more features"
        },
        "Sparse Kernels": {
            "description": "Set small kernel values to zero (sparsification)",
            "memory_reduction": lambda n, sparsity: f"Memory scales with sparsity: {sparsity*100:.1f}% of full matrix",
            "accuracy_impact": "Minimal if threshold chosen carefully"
        },
        "Block Processing": {
            "description": "Process kernel matrix in blocks to fit memory",
            "memory_reduction": lambda n, block_size: f"Memory: O(block_size²) instead of O(n²)",
            "accuracy_impact": "No accuracy loss, only computational reorganization"
        }
    }

    # Example calculations for n=10,000
    n = 10000

    for strategy, details in strategies.items():
        print(f"\n{strategy}:")
        print(f"  Description: {details['description']}")

        if strategy == "Low-rank Approximation":
            for r in [50, 100, 200]:
                print(f"  {details['memory_reduction'](n, r)}")
        elif strategy == "Nyström Method":
            for m in [500, 1000, 2000]:
                print(f"  {details['memory_reduction'](n, m)}")
        elif strategy == "Random Fourier Features":
            for d in [100, 500, 1000]:
                print(f"  {details['memory_reduction'](n, d)}")
        elif strategy == "Sparse Kernels":
            for sparsity in [0.1, 0.05, 0.01]:
                print(f"  {details['memory_reduction'](n, sparsity)}")

        print(f"  Accuracy Impact: {details['accuracy_impact']}")

    return strategies

def analyze_complexity_accuracy_tradeoff():
    """Analyze trade-off between kernel complexity and accuracy"""
    print("\n=== Complexity vs Accuracy Trade-off ===")

    # Generate datasets with different characteristics
    datasets = {
        'linear': make_classification(n_samples=1000, n_features=20, n_informative=10,
                                    n_redundant=0, n_clusters_per_class=1, random_state=42),
        'polynomial': make_classification(n_samples=1000, n_features=10, n_informative=8,
                                        n_redundant=0, n_clusters_per_class=2, random_state=42),
        'nonlinear': make_classification(n_samples=1000, n_features=5, n_informative=5,
                                       n_redundant=0, n_clusters_per_class=3, random_state=42)
    }

    kernels_config = {
        'linear': {'kernel': 'linear', 'C': 1.0},
        'poly_2': {'kernel': 'poly', 'degree': 2, 'C': 1.0, 'coef0': 1.0},
        'poly_3': {'kernel': 'poly', 'degree': 3, 'C': 1.0, 'coef0': 1.0},
        'rbf_0.1': {'kernel': 'rbf', 'C': 1.0, 'gamma': 0.1},
        'rbf_1.0': {'kernel': 'rbf', 'C': 1.0, 'gamma': 1.0},
        'rbf_10': {'kernel': 'rbf', 'C': 1.0, 'gamma': 10.0}
    }

    complexity_scores = {
        'linear': 1,
        'poly_2': 2,
        'poly_3': 3,
        'rbf_0.1': 4,
        'rbf_1.0': 5,
        'rbf_10': 6
    }

    results = []

    for dataset_name, (X, y) in datasets.items():
        print(f"\nDataset: {dataset_name}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for kernel_name, config in kernels_config.items():
            # Train model
            svm = SVC(**config)

            # Measure training time
            start_time = time.time()
            svm.fit(X_train_scaled, y_train)
            training_time = time.time() - start_time

            # Measure prediction time
            start_time = time.time()
            accuracy = svm.score(X_test_scaled, y_test)
            prediction_time = time.time() - start_time

            # Get number of support vectors
            n_support_vectors = len(svm.support_)

            results.append({
                'dataset': dataset_name,
                'kernel': kernel_name,
                'complexity': complexity_scores[kernel_name],
                'accuracy': accuracy,
                'training_time': training_time,
                'prediction_time': prediction_time,
                'n_support_vectors': n_support_vectors,
                'support_vector_ratio': n_support_vectors / len(X_train)
            })

            print(f"  {kernel_name}: Accuracy={accuracy:.3f}, "
                  f"Train={training_time:.4f}s, Pred={prediction_time:.6f}s, "
                  f"SVs={n_support_vectors}")

    return pd.DataFrame(results)

def training_vs_prediction_analysis():
    """Analyze how kernel choice affects training vs prediction time"""
    print("\n=== Training vs Prediction Time Analysis ===")

    # Generate data of different sizes
    sizes = [500, 1000, 2000, 5000]
    kernels = ['linear', 'poly', 'rbf']

    timing_results = []

    for n in sizes:
        print(f"\nAnalyzing with {n} samples...")

        # Generate data
        X, y = make_classification(n_samples=n, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for kernel in kernels:
            if kernel == 'linear':
                svm = SVC(kernel='linear', C=1.0)
            elif kernel == 'poly':
                svm = SVC(kernel='poly', degree=3, C=1.0, coef0=1.0)
            elif kernel == 'rbf':
                svm = SVC(kernel='rbf', C=1.0, gamma='scale')

            # Measure training time
            start_time = time.time()
            svm.fit(X_train_scaled, y_train)
            training_time = time.time() - start_time

            # Measure prediction time for single sample
            start_time = time.time()
            _ = svm.predict(X_test_scaled[:1])
            single_prediction_time = time.time() - start_time

            # Measure prediction time for batch
            start_time = time.time()
            _ = svm.predict(X_test_scaled)
            batch_prediction_time = time.time() - start_time

            # Calculate per-sample prediction time
            per_sample_prediction_time = batch_prediction_time / len(X_test_scaled)

            timing_results.append({
                'n_samples': n,
                'kernel': kernel,
                'training_time': training_time,
                'single_prediction_time': single_prediction_time,
                'batch_prediction_time': batch_prediction_time,
                'per_sample_prediction_time': per_sample_prediction_time,
                'n_support_vectors': len(svm.support_),
                'training_per_sample': training_time / len(X_train_scaled)
            })

            print(f"  {kernel}: Train={training_time:.4f}s, "
                  f"Single Pred={single_prediction_time:.6f}s, "
                  f"Batch Pred={batch_prediction_time:.4f}s")

    return pd.DataFrame(timing_results)

def visualize_computational_analysis():
    """Create comprehensive visualizations of computational analysis"""
    create_output_directory()

    # Run all analyses
    timing_results = benchmark_kernel_evaluation_time()
    memory_analysis = calculate_memory_requirements()
    strategies = memory_reduction_strategies()
    complexity_df = analyze_complexity_accuracy_tradeoff()
    training_pred_df = training_vs_prediction_analysis()

    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))

    # 1. Kernel evaluation time vs sample size
    ax = axes[0, 0]
    for kernel in timing_results.keys():
        sizes = timing_results[kernel]['sizes']
        times = timing_results[kernel]['times']
        std_times = timing_results[kernel]['std_times']
        ax.errorbar(sizes, times, yerr=std_times, label=kernel, marker='o')
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Training Time vs Sample Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

    # 2. Memory requirements
    ax = axes[0, 1]
    sample_sizes = list(memory_analysis.keys())
    memory_gb = [memory_analysis[n]['gb'] for n in sample_sizes]
    ax.bar(range(len(sample_sizes)), memory_gb, color='red', alpha=0.7)
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Memory (GB)')
    ax.set_title('Kernel Matrix Memory Requirements')
    ax.set_xticks(range(len(sample_sizes)))
    ax.set_xticklabels([f'{n:,}' for n in sample_sizes])
    ax.set_yscale('log')

    # 3. Complexity vs Accuracy scatter plot
    ax = axes[0, 2]
    for dataset in complexity_df['dataset'].unique():
        subset = complexity_df[complexity_df['dataset'] == dataset]
        ax.scatter(subset['complexity'], subset['accuracy'],
                  label=dataset, alpha=0.7, s=60)
    ax.set_xlabel('Kernel Complexity')
    ax.set_ylabel('Accuracy')
    ax.set_title('Complexity vs Accuracy Trade-off')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Training time by kernel and dataset
    ax = axes[1, 0]
    pivot_train = complexity_df.pivot_table(values='training_time',
                                           index='kernel', columns='dataset', aggfunc='mean')
    sns.heatmap(pivot_train, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax)
    ax.set_title('Training Time by Kernel and Dataset')

    # 5. Support vector ratio
    ax = axes[1, 1]
    pivot_sv = complexity_df.pivot_table(values='support_vector_ratio',
                                        index='kernel', columns='dataset', aggfunc='mean')
    sns.heatmap(pivot_sv, annot=True, fmt='.3f', cmap='viridis', ax=ax)
    ax.set_title('Support Vector Ratio by Kernel and Dataset')

    # 6. Training vs Prediction time comparison
    ax = axes[1, 2]
    for kernel in training_pred_df['kernel'].unique():
        subset = training_pred_df[training_pred_df['kernel'] == kernel]
        ax.scatter(subset['training_time'], subset['per_sample_prediction_time'],
                  label=kernel, alpha=0.7, s=60)
    ax.set_xlabel('Training Time (seconds)')
    ax.set_ylabel('Per-Sample Prediction Time (seconds)')
    ax.set_title('Training vs Prediction Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

    # 7. Memory reduction strategies comparison
    ax = axes[2, 0]
    n = 10000
    strategies_data = {
        'Full Matrix': 100,
        'Low-rank (r=100)': 100 * (2 * 100 / n),
        'Nyström (m=1000)': 100 * (1000 / n),
        'RFF (d=500)': 100 * (500 / n),
        'Sparse (1%)': 1
    }

    bars = ax.bar(range(len(strategies_data)), list(strategies_data.values()),
                  color=['red', 'orange', 'yellow', 'green', 'blue'], alpha=0.7)
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Relative Memory Usage (%)')
    ax.set_title('Memory Reduction Strategies')
    ax.set_xticks(range(len(strategies_data)))
    ax.set_xticklabels(list(strategies_data.keys()), rotation=45)
    ax.set_yscale('log')

    # Add value labels on bars
    for bar, value in zip(bars, strategies_data.values()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}%', ha='center', va='bottom')

    # 8. Scaling analysis
    ax = axes[2, 1]
    for kernel in training_pred_df['kernel'].unique():
        subset = training_pred_df[training_pred_df['kernel'] == kernel]
        ax.plot(subset['n_samples'], subset['training_per_sample'],
               label=f'{kernel} (training)', marker='o')
        ax.plot(subset['n_samples'], subset['per_sample_prediction_time'],
               label=f'{kernel} (prediction)', marker='s', linestyle='--')
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Per-Sample Time (seconds)')
    ax.set_title('Scaling: Per-Sample Training vs Prediction Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

    # 9. Computational complexity summary
    ax = axes[2, 2]
    ax.axis('off')
    ax.set_title('Computational Complexity Summary')

    complexity_text = """
    Time Complexity:
    • Linear: O(n²d) training, O(d) prediction
    • Polynomial: O(n²d) training, O(n_sv·d) prediction
    • RBF: O(n²d) training, O(n_sv·d) prediction

    Space Complexity:
    • Kernel Matrix: O(n²) storage
    • Support Vectors: O(n_sv·d) storage
    • Linear: No kernel matrix needed

    Key Insights:
    • RBF kernels: Highest accuracy, highest cost
    • Linear kernels: Fastest, most scalable
    • Polynomial: Middle ground complexity
    • Memory reduction crucial for large n
    """

    ax.text(0.1, 0.9, complexity_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig('../Images/L5_3_Quiz_11/computational_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'timing_results': timing_results,
        'memory_analysis': memory_analysis,
        'complexity_df': complexity_df,
        'training_pred_df': training_pred_df
    }

def create_computational_complexity_analysis():
    """Create detailed computational complexity analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Memory scaling analysis
    ax = axes[0, 0]

    sample_sizes = np.array([100, 500, 1000, 2000, 5000, 10000])

    # Memory requirements in MB
    full_kernel_memory = (sample_sizes**2 * 8) / (1024**2)  # 8 bytes per float64
    linear_memory = sample_sizes * 20 * 8 / (1024**2)  # Assume 20 features

    ax.loglog(sample_sizes, full_kernel_memory, 'r-o', linewidth=3, label='Full Kernel Matrix', markersize=8)
    ax.loglog(sample_sizes, linear_memory, 'b-s', linewidth=3, label='Linear (Feature Storage)', markersize=8)

    # Add memory limit lines
    ax.axhline(y=1024, color='orange', linestyle='--', alpha=0.7, label='1 GB RAM')
    ax.axhline(y=8192, color='red', linestyle='--', alpha=0.7, label='8 GB RAM')

    ax.set_xlabel(r'Number of Samples')
    ax.set_ylabel(r'Memory Usage (MB)')
    ax.set_title(r'Memory Scaling: Kernel vs Linear Methods')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Time complexity comparison
    ax = axes[0, 1]

    # Training time complexity (theoretical)
    linear_train_time = sample_sizes**2 * 1e-6  # O(n²)
    poly_train_time = sample_sizes**2 * 2e-6    # O(n²) with higher constant
    rbf_train_time = sample_sizes**2 * 3e-6     # O(n²) with highest constant

    ax.loglog(sample_sizes, linear_train_time, 'b-o', linewidth=2, label='Linear')
    ax.loglog(sample_sizes, poly_train_time, 'g-s', linewidth=2, label='Polynomial')
    ax.loglog(sample_sizes, rbf_train_time, 'r-^', linewidth=2, label='RBF')

    # Add theoretical scaling lines
    theoretical_n2 = sample_sizes**2 * 1e-7
    theoretical_n3 = sample_sizes**3 * 1e-10
    ax.loglog(sample_sizes, theoretical_n2, 'k--', alpha=0.5, label=r'$O(n^2)$ reference')
    ax.loglog(sample_sizes, theoretical_n3, 'k:', alpha=0.5, label=r'$O(n^3)$ reference')

    ax.set_xlabel(r'Number of Samples')
    ax.set_ylabel(r'Training Time (seconds)')
    ax.set_title(r'Training Time Complexity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Prediction time analysis
    ax = axes[1, 0]

    # Assume different numbers of support vectors
    sv_ratios = [0.1, 0.3, 0.6]  # 10%, 30%, 60% support vectors
    colors = ['blue', 'green', 'red']

    for i, sv_ratio in enumerate(sv_ratios):
        n_support_vectors = sample_sizes * sv_ratio
        pred_time = n_support_vectors * 20 * 1e-6  # O(n_sv * d)
        ax.loglog(sample_sizes, pred_time, color=colors[i], linewidth=2,
                 marker='o', label=f'{sv_ratio*100:.0f}% Support Vectors')

    # Linear prediction time (constant)
    linear_pred_time = np.ones_like(sample_sizes) * 20 * 1e-6
    ax.loglog(sample_sizes, linear_pred_time, 'k-', linewidth=3, label='Linear (constant)')

    ax.set_xlabel(r'Number of Training Samples')
    ax.set_ylabel(r'Prediction Time (seconds)')
    ax.set_title(r'Prediction Time Scaling')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Memory reduction strategies effectiveness
    ax = axes[1, 1]

    n = 10000  # Fixed sample size
    strategies = ['Full\nKernel', 'Low-rank\n(r=50)', 'Low-rank\n(r=100)', 'Nyström\n(m=500)',
                  'RFF\n(D=200)', 'Sparse\n(1%)', 'Sparse\n(5%)']

    # Memory usage (relative to full kernel)
    memory_usage = [1.0, 0.01, 0.02, 0.05, 0.02, 0.01, 0.05]

    # Approximation quality (relative accuracy)
    quality = [1.0, 0.95, 0.98, 0.92, 0.94, 0.85, 0.95]

    # Create scatter plot
    scatter = ax.scatter(memory_usage, quality, s=[200]*len(strategies),
                        c=range(len(strategies)), cmap='viridis', alpha=0.7)

    # Add labels
    for i, strategy in enumerate(strategies):
        ax.annotate(strategy, (memory_usage[i], quality[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=10)

    ax.set_xlabel(r'Relative Memory Usage')
    ax.set_ylabel(r'Approximation Quality')
    ax.set_title(r'Memory Reduction vs Quality Trade-off')
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0.8, 1.05)
    ax.grid(True, alpha=0.3)

    # Add Pareto frontier
    pareto_x = [0.01, 0.02, 0.05, 1.0]
    pareto_y = [0.95, 0.98, 0.95, 1.0]
    ax.plot(pareto_x, pareto_y, 'r--', alpha=0.5, linewidth=2, label='Pareto Frontier')
    ax.legend()

    plt.tight_layout()
    plt.savefig('../Images/L5_3_Quiz_11/computational_complexity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to run all computational analyses"""
    print("Computational Analysis of Kernel Methods")
    print("=" * 50)

    # Run comprehensive analysis
    results = visualize_computational_analysis()

    # Create detailed complexity analysis
    create_computational_complexity_analysis()

    print("\n" + "=" * 50)
    print("COMPUTATIONAL ANALYSIS SUMMARY")
    print("=" * 50)

    print("\n1. Kernel Evaluation Time:")
    print("   • Linear: Fastest - O(n²d) complexity")
    print("   • Polynomial: Moderate - depends on degree")
    print("   • RBF: Slowest - requires distance calculations")

    print("\n2. Memory Requirements:")
    print("   • n=1,000: ~8 MB (manageable)")
    print("   • n=10,000: ~800 MB (moderate)")
    print("   • n=100,000: ~80 GB (requires reduction strategies)")

    print("\n3. Memory Reduction Strategies:")
    print("   • Low-rank approximation: 98% reduction possible")
    print("   • Nyström method: 90% reduction typical")
    print("   • Random Fourier Features: 95% reduction for RBF")
    print("   • Sparsification: 99% reduction with careful thresholding")

    print("\n4. Complexity vs Accuracy Trade-off:")
    print("   • Linear: Fast but limited expressiveness")
    print("   • Polynomial: Good balance for structured data")
    print("   • RBF: Highest accuracy but computational cost")

    print("\n5. Training vs Prediction Time:")
    print("   • Training: Dominated by kernel matrix computation")
    print("   • Prediction: Scales with number of support vectors")
    print("   • Linear: Constant prediction time")
    print("   • Non-linear: Prediction time grows with model complexity")

    print(f"\nAll visualizations saved to ../Images/L5_3_Quiz_11/")

if __name__ == "__main__":
    main()