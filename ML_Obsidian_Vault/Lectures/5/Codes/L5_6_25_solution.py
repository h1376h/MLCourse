import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_6_Quiz_25")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 60)
print("Question 25: Computational Complexity")
print("=" * 60)

# Task 1: Size of kernel matrix
print("\n1. For n training samples, what is the size of the kernel matrix?")
print("   Mathematical derivation:")
print("   Given: n training samples {x₁, x₂, ..., xₙ}")
print("   Kernel matrix K where K_ij = K(xᵢ, xⱼ)")
print("   ")
print("   Matrix dimensions:")
print("   - Rows: n (one for each sample xᵢ)")
print("   - Columns: n (one for each sample xⱼ)")
print("   - Total size: n × n")
print("   - Total elements: n²")
print("   ")
n_example = 1000
print(f"   Example calculation for n = {n_example}:")
print(f"   Matrix size: {n_example} × {n_example}")
print(f"   Total elements: {n_example}² = {n_example**2:,} elements")

# Task 2: Space complexity for storing kernel matrix
print("\n2. What is the space complexity for storing the full kernel matrix?")
print("   Mathematical analysis:")
print("   ")
print("   Memory per element: b bytes (typically b = 8 for float64)")
print("   Total elements: n²")
print("   Total memory: M = b × n² bytes")
print("   ")
print("   Space complexity: O(n²)")
print("   ")
print("   Unit conversions:")
print("   1 KB = 1,024 bytes = 2¹⁰ bytes")
print("   1 MB = 1,024² bytes = 2²⁰ bytes")
print("   1 GB = 1,024³ bytes = 2³⁰ bytes")
print("   ")
print(f"   Example for n = {n_example}, b = 8 bytes:")
print(f"   M = 8 × {n_example}² = 8 × {n_example**2:,} = {n_example**2 * 8:,} bytes")
print(f"   M = {n_example**2 * 8:,} ÷ 2²⁰ = {n_example**2 * 8 / (1024**2):.1f} MB")

# Calculate memory for different dataset sizes
sizes = [100, 1000, 5000, 10000, 50000, 100000]
print("\n   Memory requirements for different dataset sizes:")
print("   n\t\tMatrix Size\tMemory (MB)\tMemory (GB)")
print("   " + "-"*50)
for n in sizes:
    memory_bytes = n**2 * 8
    memory_mb = memory_bytes / (1024**2)
    memory_gb = memory_bytes / (1024**3)
    print(f"   {n:,}\t\t{n}×{n}\t\t{memory_mb:.1f}\t\t{memory_gb:.2f}")

# Task 3: Maximum number of support vectors
print("\n3. How many support vectors can an SVM have at most?")
print("   Maximum number of support vectors = n (all training samples)")
print("   This occurs when:")
print("   - Data is not linearly separable")
print("   - C parameter is very large")
print("   - All samples lie on or within the margin")
print("   In practice, good SVMs have much fewer support vectors")

# Task 4: Prediction time complexity
print("\n4. If you have 100 support vectors, what is the prediction time complexity for one new sample?")
print("   Mathematical derivation:")
print("   ")
print("   SVM decision function:")
print("   f(x) = Σᵢ∈SV αᵢyᵢK(x, xᵢ) + b")
print("   ")
print("   For each support vector xᵢ ∈ SV:")
print("   1. Compute kernel: K(x_new, xᵢ)     [1 operation]")
print("   2. Multiply: αᵢyᵢK(x_new, xᵢ)       [1 multiplication]")
print("   3. Add to sum                        [1 addition]")
print("   ")
n_sv = 100
print(f"   For n_sv = {n_sv} support vectors:")
print(f"   - Kernel evaluations: {n_sv}")
print(f"   - Multiplications: {n_sv}")
print(f"   - Additions: {n_sv-1} (sum) + 1 (bias) = {n_sv}")
print(f"   - Total operations: {3*n_sv} = O(n_sv)")
print(f"   ")
print(f"   Time complexity: O(n_sv) = O({n_sv})")

# Task 5: Training vs Prediction cost
print("\n5. Which is more expensive: training or prediction? Why?")
print("   TRAINING is more expensive:")
print("   - Training: O(n²) to O(n³) depending on algorithm")
print("   - Prediction: O(n_sv) where n_sv ≤ n")
print("   - Training involves solving optimization problem")
print("   - Prediction is just evaluating the decision function")

# Visualization 1: Kernel Matrix Size Growth
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: Matrix size visualization
n_values = np.array([100, 500, 1000, 2000, 5000])
matrix_sizes = n_values**2

ax1.loglog(n_values, matrix_sizes, 'bo-', linewidth=2, markersize=8, label='Matrix elements ($n^2$)')
ax1.loglog(n_values, n_values, 'r--', linewidth=2, label='Linear growth ($n$)')
ax1.set_xlabel('Number of training samples (n)')
ax1.set_ylabel('Number of matrix elements')
ax1.set_title('Kernel Matrix Size Growth', fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Add annotations
for i, (n, size) in enumerate(zip(n_values, matrix_sizes)):
    if i % 2 == 0:  # Annotate every other point
        ax1.annotate(f'n={n}\n{size:,} elements', 
                    (n, size), xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# Right plot: Memory requirements
memory_mb = matrix_sizes * 8 / (1024**2)
memory_gb = memory_mb / 1024

ax2.loglog(n_values, memory_mb, 'go-', linewidth=2, markersize=8, label='Memory (MB)')
ax2.set_xlabel('Number of training samples (n)')
ax2.set_ylabel('Memory requirement (MB)')
ax2.set_title('Memory Requirements (8 bytes per element)', fontweight='bold')
ax2.grid(True, alpha=0.3)

# Add GB labels on right axis
ax2_gb = ax2.twinx()
ax2_gb.loglog(n_values, memory_gb, 'go-', alpha=0)  # Invisible line for scaling
ax2_gb.set_ylabel('Memory requirement (GB)')

# Add memory threshold lines
ax2.axhline(y=1024, color='orange', linestyle='--', alpha=0.7, label='1 GB')
ax2.axhline(y=8*1024, color='red', linestyle='--', alpha=0.7, label='8 GB')
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'kernel_matrix_complexity.png'), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 2: Training vs Prediction Complexity
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

n_range = np.logspace(2, 5, 50)  # 100 to 100,000 samples
training_complexity_quadratic = n_range**2
training_complexity_cubic = n_range**3
prediction_complexity = n_range * 0.1  # Assuming 10% support vectors

ax.loglog(n_range, training_complexity_quadratic, 'r-', linewidth=3, 
          label='Training (SMO): $O(n^2)$')
ax.loglog(n_range, training_complexity_cubic, 'r--', linewidth=2, 
          label='Training (Standard QP): $O(n^3)$')
ax.loglog(n_range, prediction_complexity, 'b-', linewidth=3,
          label='Prediction: $O(n_{sv}) \\approx O(0.1n)$')

ax.set_xlabel('Number of training samples (n)')
ax.set_ylabel('Computational complexity (arbitrary units)')
ax.set_title('Training vs Prediction Complexity', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)

# Add annotations
ax.annotate('Training is much more expensive\nfor large datasets', 
            xy=(10000, 10**8), xytext=(20000, 10**6),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))

ax.annotate('Prediction scales linearly\nwith support vectors', 
            xy=(10000, 10**3), xytext=(30000, 10**4),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2),
            fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'training_vs_prediction_complexity.png'), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 3: Support Vector Analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: Support vector distribution
scenarios = ['Well-separated\ndata', 'Overlapping\ndata', 'Non-separable\ndata']
sv_percentages = [5, 25, 80]  # Typical percentages
colors = ['green', 'orange', 'red']

bars = ax1.bar(scenarios, sv_percentages, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Support vectors (\\% of training data)')
ax1.set_title('Support Vector Percentage by Data Type', fontweight='bold')
ax1.set_ylim(0, 100)

# Add percentage labels on bars
for bar, pct in zip(bars, sv_percentages):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{pct}%', ha='center', va='bottom', fontweight='bold')

# Right plot: Prediction time vs support vectors
n_sv_range = np.array([10, 50, 100, 500, 1000, 5000])
prediction_times = n_sv_range * 0.001  # Assume 1ms per 1000 support vectors

ax2.plot(n_sv_range, prediction_times, 'bo-', linewidth=2, markersize=8)
ax2.set_xlabel('Number of support vectors')
ax2.set_ylabel('Prediction time (ms)')
ax2.set_title('Prediction Time vs Support Vectors', fontweight='bold')
ax2.grid(True, alpha=0.3)

# Add annotations
for i, (sv, time) in enumerate(zip(n_sv_range, prediction_times)):
    if i % 2 == 0:
        ax2.annotate(f'{sv} SVs\n{time:.1f}ms', 
                    (sv, time), xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'support_vector_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAll visualizations saved to: {save_dir}")
print("\nSummary of Question 25:")
print("1. Kernel matrix size: n × n")
print("2. Space complexity: O(n²)")
print("3. Maximum support vectors: n (all samples)")
print("4. Prediction complexity: O(n_sv)")
print("5. Training is more expensive than prediction")
