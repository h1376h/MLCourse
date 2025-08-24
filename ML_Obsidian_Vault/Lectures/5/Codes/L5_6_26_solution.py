import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import matplotlib.patches as mpatches

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_6_Quiz_26")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 60)
print("Question 26: Computational Concept Check")
print("=" * 60)

# Task 1: True or False - Kernel matrix must always be stored in memory
print("\n1. True or False: The kernel matrix must always be stored in memory.")
print("   FALSE")
print("   Explanation:")
print("   - SMO algorithm computes kernel values on-demand")
print("   - Only a cache of recently used kernel values is stored")
print("   - Kernel values can be recomputed when needed")
print("   - This is what makes SMO memory-efficient for large datasets")

# Task 2: True or False - More support vectors always mean better accuracy
print("\n2. True or False: More support vectors always mean better accuracy.")
print("   FALSE")
print("   Explanation:")
print("   - More support vectors often indicate overfitting")
print("   - Good models are sparse (few support vectors)")
print("   - Too many support vectors suggest poor generalization")
print("   - Optimal models balance complexity and generalization")

# Task 3: Advantage of SMO over standard QP
print("\n3. What is the advantage of SMO over standard quadratic programming?")
print("   Key advantages:")
print("   - Memory efficiency: O(n) vs O(n²) space complexity")
print("   - No need for specialized QP solvers")
print("   - Analytical solution for 2-variable subproblems")
print("   - Scales to large datasets")
print("   - Avoids numerical issues of large matrix operations")

# Task 4: Why prefer primal formulation for linear SVMs
print("\n4. Why might you prefer the primal formulation for linear SVMs with many features?")
print("   Mathematical comparison:")
print("   ")
print("   Primal formulation variables: d (number of features)")
print("   Dual formulation variables: n (number of samples)")
print("   ")
print("   Case 1: d << n (few features, many samples)")
print("   → Primal is more efficient (d variables vs n variables)")
print("   ")
print("   Case 2: d >> n (many features, few samples)")
print("   → Dual is typically more efficient (n variables vs d variables)")
print("   ")
print("   However, for LINEAR kernels specifically:")
print("   - Primal works directly in feature space")
print("   - No kernel matrix computation needed")
print("   - Direct gradient-based optimization possible")
print("   - Can be more efficient even when d > n")

# Task 5: Computational challenges for large datasets
print("\n5. What computational challenges arise when training SVMs on very large datasets?")
print("   Major challenges:")
print("   - Memory: O(n²) kernel matrix storage")
print("   - Time: O(n²) to O(n³) training complexity")
print("   - I/O: Data doesn't fit in memory")
print("   - Convergence: More iterations needed")
print("   - Numerical stability: Ill-conditioned problems")

# Visualization 1: Memory Storage Strategies
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: Full kernel matrix storage
ax1.set_title('Traditional Approach: Full Kernel Matrix Storage', fontsize=12, fontweight='bold')
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 8)

# Draw full matrix
matrix_size = 4
start_x, start_y = 3, 2
cell_size = 0.8

for i in range(matrix_size):
    for j in range(matrix_size):
        rect = Rectangle((start_x + j*cell_size, start_y + i*cell_size), 
                        cell_size*0.9, cell_size*0.9, 
                        facecolor='lightcoral', edgecolor='darkred')
        ax1.add_patch(rect)
        ax1.text(start_x + j*cell_size + cell_size*0.45, 
                start_y + i*cell_size + cell_size*0.45, 
                f'K({i+1},{j+1})', ha='center', va='center', fontsize=8)

ax1.text(5, 1, r'Memory: $O(n^2)$' + '\n' + r'All kernel values stored',
         ha='center', va='center', fontsize=10,
         bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))

ax1.set_xticks([])
ax1.set_yticks([])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

# Right plot: SMO approach with caching
ax2.set_title('SMO Approach: On-demand + Caching', fontsize=12, fontweight='bold')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 8)

# Draw cache
cache_rect = FancyBboxPatch((2, 5), 6, 1.5, boxstyle="round,pad=0.1", 
                           facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
ax2.add_patch(cache_rect)
ax2.text(5, 5.75, r'Kernel Cache' + '\n' + r'(Recently used values)',
         ha='center', va='center', fontsize=10, fontweight='bold')

# Draw computation on demand
compute_rect = FancyBboxPatch((2, 2.5), 6, 1.5, boxstyle="round,pad=0.1", 
                             facecolor='lightblue', edgecolor='darkblue', linewidth=2)
ax2.add_patch(compute_rect)
ax2.text(5, 3.25, r'Compute on Demand' + '\n' + r'(When needed)',
         ha='center', va='center', fontsize=10, fontweight='bold')

ax2.text(5, 1, r'Memory: $O(n)$' + '\n' + r'Only cache + data stored',
         ha='center', va='center', fontsize=10,
         bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))

ax2.set_xticks([])
ax2.set_yticks([])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'memory_storage_strategies.png'), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 2: Support Vector Count vs Model Quality
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: Support vector percentage vs generalization
sv_percentages = np.array([5, 15, 30, 50, 70, 90])
generalization_scores = np.array([0.95, 0.92, 0.85, 0.75, 0.65, 0.55])

ax1.plot(sv_percentages, generalization_scores, 'ro-', linewidth=2, markersize=8)
ax1.set_xlabel('Support Vectors (\\% of training data)')
ax1.set_ylabel('Generalization Performance')
ax1.set_title('Support Vectors vs Generalization', fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0.5, 1.0)

# Add annotations
ax1.annotate(r'Optimal range' + '\n' + r'(sparse model)',
            xy=(10, 0.93), xytext=(25, 0.98),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))

ax1.annotate(r'Overfitting' + '\n' + r'(too complex)',
            xy=(80, 0.6), xytext=(60, 0.7),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'))

# Right plot: Model complexity illustration
scenarios = ['Well-separated', 'Moderate overlap', 'High overlap']
sv_counts = [8, 25, 45]
colors = ['green', 'orange', 'red']

bars = ax2.bar(scenarios, sv_counts, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Support Vectors (\\% of data)')
ax2.set_title('Typical Support Vector Counts', fontweight='bold')
ax2.set_ylim(0, 50)

# Add labels on bars
for bar, count in zip(bars, sv_counts):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{count}\\%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'support_vector_quality.png'), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 3: Computational Challenges for Large Datasets
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
ax.set_title('Computational Challenges for Large-Scale SVM Training', 
             fontsize=16, fontweight='bold')
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)

# Memory challenge
memory_box = FancyBboxPatch((1, 9), 10, 1.5, boxstyle="round,pad=0.1", 
                           facecolor='lightcoral', edgecolor='darkred', linewidth=2)
ax.add_patch(memory_box)
ax.text(6, 9.75, r'Memory Challenge: $O(n^2)$ kernel matrix storage' + '\n' + r'Solution: SMO with caching, kernel approximation',
        ha='center', va='center', fontsize=11, fontweight='bold')

# Time complexity challenge
time_box = FancyBboxPatch((1, 7), 10, 1.5, boxstyle="round,pad=0.1", 
                         facecolor='lightblue', edgecolor='darkblue', linewidth=2)
ax.add_patch(time_box)
ax.text(6, 7.75, r'Time Complexity: $O(n^2)$ to $O(n^3)$ training' + '\n' + r'Solution: SMO, parallel processing, approximation',
        ha='center', va='center', fontsize=11, fontweight='bold')

# I/O challenge
io_box = FancyBboxPatch((1, 5), 10, 1.5, boxstyle="round,pad=0.1", 
                       facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
ax.add_patch(io_box)
ax.text(6, 5.75, r'I/O Bottleneck: Data does not fit in memory' + '\n' + r'Solution: Streaming algorithms, data chunking',
        ha='center', va='center', fontsize=11, fontweight='bold')

# Convergence challenge
conv_box = FancyBboxPatch((1, 3), 10, 1.5, boxstyle="round,pad=0.1", 
                         facecolor='lightyellow', edgecolor='orange', linewidth=2)
ax.add_patch(conv_box)
ax.text(6, 3.75, r'Convergence: More iterations for large datasets' + '\n' + r'Solution: Better working set selection, warm starts',
        ha='center', va='center', fontsize=11, fontweight='bold')

# Numerical stability challenge
num_box = FancyBboxPatch((1, 1), 10, 1.5, boxstyle="round,pad=0.1", 
                        facecolor='lightpink', edgecolor='purple', linewidth=2)
ax.add_patch(num_box)
ax.text(6, 1.75, r'Numerical Stability: Ill-conditioned kernel matrices' + '\n' + r'Solution: Regularization, numerical precision control',
        ha='center', va='center', fontsize=11, fontweight='bold')

ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'computational_challenges.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAll visualizations saved to: {save_dir}")
print("\nSummary of Question 26:")
print("1. FALSE: Kernel matrix doesn't need full storage (SMO uses caching)")
print("2. FALSE: More support vectors often indicate overfitting")
print("3. SMO advantages: Memory efficiency, no QP solver needed")
print("4. Primal formulation: Direct optimization when d >> n")
print("5. Large dataset challenges: Memory, time, I/O, convergence, stability")
