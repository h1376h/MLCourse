import numpy as np
import matplotlib.pyplot as plt
import os
import time
from math import comb

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_3_Quiz_2")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amssymb}'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amssymb}'

print("=" * 60)
print("QUESTION 2: COMPUTATIONAL COMPLEXITY ANALYSIS")
print("=" * 60)

# Task 1: Derive the number of features in explicit polynomial mapping
print("\n" + "="*50)
print("TASK 1: POLYNOMIAL FEATURE MAPPING COMPLEXITY")
print("="*50)

def calculate_polynomial_features(n, d):
    """
    Calculate the number of features in polynomial mapping of degree d for n-dimensional input
    
    For polynomial kernel of degree d, we need all monomials of degree <= d
    This is equivalent to choosing d items from n+d-1 items with replacement
    Formula: C(n+d-1, d) = (n+d-1)! / (d! * (n-1)!)
    
    Alternative formula: sum from k=0 to d of C(n+k-1, k)
    """
    total_features = 0
    
    print(f"\nFor n={n} dimensions and degree d={d}:")
    print("We need all monomials x1^i1 * x2^i2 * ... * xn^in where i1+i2+...+in <= d")
    
    for degree in range(d + 1):
        # Number of monomials of exactly degree 'degree'
        if degree == 0:
            features_at_degree = 1  # constant term
        else:
            features_at_degree = comb(n + degree - 1, degree)
        
        total_features += features_at_degree
        print(f"  Degree {degree}: {features_at_degree} features")
    
    print(f"  Total features: {total_features}")
    
    # Alternative calculation using the direct formula
    direct_formula = comb(n + d, d)
    print(f"  Using direct formula C(n+d, d): {direct_formula}")
    
    return total_features

# Calculate for the given examples
print("\nCalculating for specific examples:")
features_10_3 = calculate_polynomial_features(10, 3)
features_100_2 = calculate_polynomial_features(100, 2)

# Task 2: Calculate for n=10, d=3 and n=100, d=2
print("\n" + "="*50)
print("TASK 2: SPECIFIC CALCULATIONS")
print("="*50)

print(f"\nCase 1: n=10, d=3")
print(f"Number of features: {features_10_3}")
print(f"Memory for feature vectors (float64): {features_10_3 * 8} bytes = {features_10_3 * 8 / 1024:.2f} KB")

print(f"\nCase 2: n=100, d=2")
print(f"Number of features: {features_100_2}")
print(f"Memory for feature vectors (float64): {features_100_2 * 8} bytes = {features_100_2 * 8 / 1024:.2f} KB")

# Task 3: Computational cost comparison
print("\n" + "="*50)
print("TASK 3: COMPUTATIONAL COST COMPARISON")
print("="*50)

def compare_computational_costs(n, d):
    """
    Compare computational costs of explicit mapping vs kernel trick
    """
    num_features = comb(n + d, d)
    
    print(f"\nFor n={n}, d={d}:")
    print(f"Explicit mapping dimension: {num_features}")
    
    # Cost of explicit mapping
    print("\nExplicit Feature Mapping:")
    print(f"1. Transform x to φ(x): O({num_features}) operations")
    print(f"2. Transform z to φ(z): O({num_features}) operations") 
    print(f"3. Compute φ(x)^T φ(z): O({num_features}) operations")
    print(f"Total: O({num_features}) = O(C(n+d, d))")
    
    # Cost of kernel trick
    print(f"\nKernel Trick K(x,z) = (x^T z + 1)^d:")
    print(f"1. Compute x^T z: O({n}) operations")
    print(f"2. Add 1: O(1) operations")
    print(f"3. Raise to power d: O(log d) operations")
    print(f"Total: O({n} + log d) ≈ O({n})")
    
    speedup = num_features / n
    print(f"\nSpeedup factor: {speedup:.2f}x")
    
    return num_features, n, speedup

# Compare for our examples
speedup_10_3 = compare_computational_costs(10, 3)
speedup_100_2 = compare_computational_costs(100, 2)

# Task 4: When does kernel trick provide significant savings?
print("\n" + "="*50)
print("TASK 4: WHEN KERNEL TRICK PROVIDES SAVINGS")
print("="*50)

def analyze_savings():
    """
    Analyze when kernel trick provides significant computational savings
    """
    print("\nKernel trick provides significant savings when:")
    print("C(n+d, d) >> n")
    print("\nThis happens when:")
    print("1. High degree d (exponential growth)")
    print("2. High dimensionality n (polynomial growth)")
    print("3. Both n and d are moderate to large")
    
    # Create visualization
    dimensions = [2, 5, 10, 20, 50, 100]
    degrees = [2, 3, 4, 5]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, d in enumerate(degrees):
        explicit_costs = []
        kernel_costs = []
        speedups = []
        
        for n in dimensions:
            explicit_cost = comb(n + d, d)
            kernel_cost = n
            speedup = explicit_cost / kernel_cost
            
            explicit_costs.append(explicit_cost)
            kernel_costs.append(kernel_cost)
            speedups.append(speedup)
        
        ax = axes[i]
        ax.semilogy(dimensions, explicit_costs, 'r-o', label='Explicit Mapping', linewidth=2)
        ax.semilogy(dimensions, kernel_costs, 'b-s', label='Kernel Trick', linewidth=2)
        ax.set_xlabel('Input Dimension (n)')
        ax.set_ylabel('Computational Cost')
        ax.set_title(f'Degree d={d}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'computational_complexity_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Speedup analysis
    plt.figure(figsize=(12, 8))
    for d in degrees:
        speedups = []
        for n in dimensions:
            explicit_cost = comb(n + d, d)
            kernel_cost = n
            speedup = explicit_cost / kernel_cost
            speedups.append(speedup)
        
        plt.semilogy(dimensions, speedups, '-o', label=f'd={d}', linewidth=2)
    
    plt.xlabel('Input Dimension (n)')
    plt.ylabel('Speedup Factor (Explicit/Kernel)')
    plt.title('Kernel Trick Speedup vs Input Dimension')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='10x speedup threshold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'kernel_speedup_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

analyze_savings()

# Task 5: Memory requirements analysis
print("\n" + "="*50)
print("TASK 5: MEMORY REQUIREMENTS ANALYSIS")
print("="*50)

def analyze_memory_requirements():
    """
    Analyze memory requirements for feature vectors vs kernel evaluations
    """
    print("\nMemory Requirements Analysis:")
    print("="*40)
    
    # Sample sizes
    sample_sizes = [100, 1000, 10000]
    
    print("\nFor different dataset sizes and polynomial kernels:")
    
    for m in sample_sizes:
        print(f"\nDataset size: {m} samples")
        print("-" * 30)
        
        for n, d in [(10, 3), (100, 2), (50, 4)]:
            num_features = comb(n + d, d)
            
            # Memory for explicit feature vectors
            feature_memory = m * num_features * 8  # 8 bytes per float64
            
            # Memory for kernel matrix (if stored)
            kernel_memory = m * m * 8  # 8 bytes per float64
            
            # Memory for original data
            original_memory = m * n * 8
            
            print(f"  n={n}, d={d} (features: {num_features})")
            print(f"    Original data: {original_memory/1024/1024:.2f} MB")
            print(f"    Explicit features: {feature_memory/1024/1024:.2f} MB")
            print(f"    Kernel matrix: {kernel_memory/1024/1024:.2f} MB")
            print(f"    Feature/Original ratio: {feature_memory/original_memory:.1f}x")
            print(f"    Kernel/Original ratio: {kernel_memory/original_memory:.1f}x")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Memory scaling with dataset size
    sizes = np.logspace(2, 5, 20)  # 100 to 100,000 samples
    
    for n, d in [(10, 3), (100, 2)]:
        num_features = comb(n + d, d)
        
        original_mem = sizes * n * 8 / 1024 / 1024  # MB
        feature_mem = sizes * num_features * 8 / 1024 / 1024  # MB
        kernel_mem = sizes * sizes * 8 / 1024 / 1024  # MB
        
        ax1.loglog(sizes, original_mem, '--', label=f'Original n={n}')
        ax1.loglog(sizes, feature_mem, '-', label=f'Features n={n},d={d}')
        ax1.loglog(sizes, kernel_mem, ':', label=f'Kernel n={n}')
    
    ax1.set_xlabel('Dataset Size (samples)')
    ax1.set_ylabel('Memory (MB)')
    ax1.set_title('Memory Scaling with Dataset Size')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Memory scaling with feature dimension
    dimensions = range(2, 21)
    d = 3
    m = 1000
    
    feature_dims = [comb(n + d, d) for n in dimensions]
    original_mem = [m * n * 8 / 1024 / 1024 for n in dimensions]
    feature_mem = [m * fd * 8 / 1024 / 1024 for fd in feature_dims]
    kernel_mem = [m * m * 8 / 1024 / 1024] * len(dimensions)
    
    ax2.semilogy(dimensions, original_mem, '--', label='Original Data')
    ax2.semilogy(dimensions, feature_mem, '-', label=f'Explicit Features (d={d})')
    ax2.semilogy(dimensions, kernel_mem, ':', label='Kernel Matrix')
    
    ax2.set_xlabel('Input Dimension (n)')
    ax2.set_ylabel('Memory (MB)')
    ax2.set_title(f'Memory Scaling with Dimension (m={m}, d={d})')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'memory_requirements_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

analyze_memory_requirements()

# Additional simple visualization for better understanding
print("\n" + "="*50)
print("CREATING SIMPLE VISUALIZATION")
print("="*50)

def create_simple_complexity_visualization():
    """
    Create a simple visualization showing the complexity difference
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Simple comparison chart
    methods = ['Kernel Trick', 'Explicit Mapping']
    n10_d3_costs = [10, 286]  # For n=10, d=3
    n100_d2_costs = [100, 5151]  # For n=100, d=2

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax1.bar(x - width/2, n10_d3_costs, width, label='n=10, d=3', color='skyblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, n100_d2_costs, width, label='n=100, d=2', color='lightcoral', alpha=0.8)

    ax1.set_ylabel('Computational Cost', fontsize=12, fontweight='bold')
    ax1.set_title('Kernel Trick vs Explicit Mapping', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=12)
    ax1.legend(fontsize=11)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    # Growth visualization
    degrees = [1, 2, 3, 4, 5]
    n = 10

    kernel_costs = [n] * len(degrees)  # Always O(n)
    explicit_costs = [comb(n + d, d) for d in degrees]

    ax2.plot(degrees, kernel_costs, 'o-', linewidth=3, markersize=8,
             color='green', label='Kernel Trick', alpha=0.8)
    ax2.plot(degrees, explicit_costs, 's-', linewidth=3, markersize=8,
             color='red', label='Explicit Mapping', alpha=0.8)

    ax2.set_xlabel('Polynomial Degree (d)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Features/Operations', fontsize=12, fontweight='bold')
    ax2.set_title('Growth with Polynomial Degree (n=10)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'complexity_simple_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()

create_simple_complexity_visualization()

print(f"\nPlots saved to: {save_dir}")
print("\n" + "="*60)
print("SOLUTION COMPLETE!")
print("="*60)
