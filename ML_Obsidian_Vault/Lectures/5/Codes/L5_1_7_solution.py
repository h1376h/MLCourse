import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
import cvxpy as cp

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_1_Quiz_7")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("QUESTION 7: SVM vs PERCEPTRON COMPARISON")
print("=" * 80)

# Dataset
X = np.array([
    [2, 1],   # x1
    [1, 2],   # x2
    [-1, -1]  # x3
])

y = np.array([1, 1, -1])  # Labels

print("\nDataset:")
for i in range(len(X)):
    print(f"x_{i+1} = {X[i]}, y_{i+1} = {y[i]:+d}")

print("\n" + "="*60)
print("STEP 1: FIND MAXIMUM MARGIN SEPARATING HYPERPLANE")
print("="*60)

print("Setting up the dual SVM optimization problem...")

# Compute kernel matrix K_ij = y_i * y_j * x_i^T * x_j
K = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        K[i, j] = y[i] * y[j] * np.dot(X[i], X[j])

print("\nKernel Matrix K_ij = y_i * y_j * x_i^T * x_j:")
print("K =")
for i in range(3):
    row_str = "["
    for j in range(3):
        row_str += f"{K[i,j]:6.1f}"
        if j < 2:
            row_str += ", "
    row_str += "]"
    print(f"    {row_str}")

# Solve using cvxpy
alpha = cp.Variable(3)
objective = cp.Maximize(cp.sum(alpha) - 0.5 * cp.quad_form(alpha, K))
constraints = [alpha >= 0, cp.sum(cp.multiply(alpha, y)) == 0]
prob = cp.Problem(objective, constraints)
prob.solve()

alpha_svm = alpha.value
print(f"\nOptimal dual variables (SVM):")
for i in range(3):
    print(f"α_{i+1}* = {alpha_svm[i]:.6f}")

# Calculate SVM weight vector and bias
w_svm = np.zeros(2)
for i in range(3):
    w_svm += alpha_svm[i] * y[i] * X[i]

# Find bias using support vectors
support_vectors = []
for i in range(3):
    if alpha_svm[i] > 1e-6:
        support_vectors.append(i)

b_svm = np.mean([y[sv] - np.dot(w_svm, X[sv]) for sv in support_vectors])

print(f"\nSVM Solution:")
print(f"w_SVM = {w_svm}")
print(f"b_SVM = {b_svm:.6f}")
print(f"Decision boundary: {w_svm[0]:.3f}x₁ + {w_svm[1]:.3f}x₂ + {b_svm:.3f} = 0")

# Calculate SVM margin
margin_svm = 2.0 / np.linalg.norm(w_svm)
print(f"SVM Margin width: {margin_svm:.6f}")

print("\n" + "="*60)
print("STEP 2: PERCEPTRON ALGORITHM SOLUTIONS")
print("="*60)

def perceptron_algorithm(X, y, max_iterations=1000, learning_rate=1.0):
    """
    Implement perceptron algorithm
    """
    n_samples, n_features = X.shape
    # Initialize weights and bias
    w = np.zeros(n_features)
    b = 0

    # Augment data with bias term
    X_augmented = np.column_stack([X, np.ones(n_samples)])
    w_augmented = np.zeros(n_features + 1)

    iterations = 0
    converged = False

    print(f"Running Perceptron Algorithm (learning rate = {learning_rate})...")

    while iterations < max_iterations and not converged:
        converged = True
        for i in range(n_samples):
            # Compute prediction
            prediction = np.sign(np.dot(w_augmented, X_augmented[i]))
            if prediction == 0:
                prediction = -1  # Handle zero case

            # Check if misclassified
            if prediction != y[i]:
                # Update weights
                w_augmented += learning_rate * y[i] * X_augmented[i]
                converged = False

        iterations += 1

    w = w_augmented[:-1]
    b = w_augmented[-1]

    return w, b, iterations

# Run perceptron with different initializations to show different solutions
perceptron_solutions = []
np.random.seed(42)

for trial in range(3):
    print(f"\nPerceptron Trial {trial + 1}:")

    # Shuffle data for different convergence paths
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    w_perc, b_perc, iterations = perceptron_algorithm(X_shuffled, y_shuffled)

    print(f"  Converged in {iterations} iterations")
    print(f"  w_Perceptron = {w_perc}")
    print(f"  b_Perceptron = {b_perc:.6f}")
    print(f"  Decision boundary: {w_perc[0]:.3f}x₁ + {w_perc[1]:.3f}x₂ + {b_perc:.3f} = 0")

    # Calculate margin for this perceptron solution
    margin_perc = 2.0 / np.linalg.norm(w_perc)
    print(f"  Perceptron Margin width: {margin_perc:.6f}")

    perceptron_solutions.append((w_perc, b_perc, margin_perc))

print("\n" + "="*60)
print("STEP 3: MARGIN COMPARISON")
print("="*60)

print(f"SVM Margin: {margin_svm:.6f}")
print(f"Perceptron Margins:")
for i, (w, b, margin) in enumerate(perceptron_solutions):
    print(f"  Trial {i+1}: {margin:.6f}")

print(f"\nMargin Ratios (Perceptron/SVM):")
for i, (w, b, margin) in enumerate(perceptron_solutions):
    ratio = margin / margin_svm
    print(f"  Trial {i+1}: {ratio:.3f} ({'smaller' if ratio < 1 else 'larger'} than SVM)")

print("\n" + "="*60)
print("STEP 4: GENERALIZATION ANALYSIS")
print("="*60)

print("Which method generalizes better?")
print("\n1. THEORETICAL PERSPECTIVE:")
print("   - SVM maximizes the margin, leading to better generalization bounds")
print("   - Larger margins provide more 'safety buffer' for new data points")
print("   - SVM solution is unique and optimal")
print("   - Perceptron finds 'any' separating hyperplane, not necessarily the best")

print("\n2. STATISTICAL LEARNING THEORY:")
print("   - Generalization error bound for SVM: O(√(R²/γ²n))")
print("   - Where R is data radius, γ is margin, n is sample size")
print("   - Larger margin γ leads to better bounds")

print("\n3. EMPIRICAL EVIDENCE:")
print("   - SVM consistently achieves maximum margin")
print("   - Perceptron margin depends on data ordering and initialization")
print("   - SVM is more robust to noise and outliers")

# Demonstrate with noise analysis
print("\n4. NOISE ROBUSTNESS ANALYSIS:")
noise_levels = [0.0, 0.1, 0.2, 0.3]
svm_performance = []
perceptron_performance = []

for noise_level in noise_levels:
    print(f"\n   Noise level: {noise_level}")

    # Add noise to data
    X_noisy = X + np.random.normal(0, noise_level, X.shape)

    # SVM performance (margin)
    # Recalculate SVM with noisy data
    K_noisy = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            K_noisy[i, j] = y[i] * y[j] * np.dot(X_noisy[i], X_noisy[j])

    alpha_noisy = cp.Variable(3)
    objective_noisy = cp.Maximize(cp.sum(alpha_noisy) - 0.5 * cp.quad_form(alpha_noisy, K_noisy))
    constraints_noisy = [alpha_noisy >= 0, cp.sum(cp.multiply(alpha_noisy, y)) == 0]
    prob_noisy = cp.Problem(objective_noisy, constraints_noisy)
    prob_noisy.solve()

    w_svm_noisy = np.zeros(2)
    for i in range(3):
        w_svm_noisy += alpha_noisy.value[i] * y[i] * X_noisy[i]

    margin_svm_noisy = 2.0 / np.linalg.norm(w_svm_noisy)
    svm_performance.append(margin_svm_noisy)

    # Perceptron performance (average margin)
    w_perc_noisy, b_perc_noisy, _ = perceptron_algorithm(X_noisy, y)
    margin_perc_noisy = 2.0 / np.linalg.norm(w_perc_noisy)
    perceptron_performance.append(margin_perc_noisy)

    print(f"     SVM margin: {margin_svm_noisy:.4f}")
    print(f"     Perceptron margin: {margin_perc_noisy:.4f}")

print("\n" + "="*60)
print("STEP 5: COMPUTATIONAL COMPLEXITY ANALYSIS")
print("="*60)

print("PERCEPTRON ALGORITHM:")
print("- Time Complexity: O(n·d·k) where k is number of iterations")
print("- Space Complexity: O(d) for storing weight vector")
print("- Iterations: Finite for linearly separable data")
print("- Convergence: Guaranteed for linearly separable data")
print("- Implementation: Very simple, online learning possible")

print("\nSVM ALGORITHM:")
print("- Time Complexity: O(n³) for general QP solvers, O(n²) for specialized SVM solvers")
print("- Space Complexity: O(n²) for storing kernel matrix")
print("- Optimization: Convex quadratic programming problem")
print("- Convergence: Global optimum guaranteed")
print("- Implementation: More complex, requires QP solver")

print("\nPRACTICAL CONSIDERATIONS:")
print("- Small datasets (n < 1000): SVM overhead acceptable, better generalization")
print("- Large datasets (n > 10000): Perceptron faster, SVM may be prohibitive")
print("- Online learning: Perceptron naturally online, SVM requires batch processing")
print("- Memory constraints: Perceptron more memory efficient")

n_samples = len(X)
d_features = X.shape[1]
print(f"\nFor our dataset (n={n_samples}, d={d_features}):")
print(f"- Perceptron: Very fast, O({n_samples}·{d_features}·k) ≈ O(6k)")
print(f"- SVM: Fast enough, O({n_samples}³) = O({n_samples**3}) = O(27)")
print("- SVM is preferred due to better generalization with manageable computational cost")

print("\n" + "="*60)
print("STEP 6: VISUALIZATION")
print("="*60)

# Create comprehensive visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Decision boundaries comparison
x1_range = np.linspace(-3, 4, 100)

# SVM boundary
if abs(w_svm[1]) > 1e-10:
    x2_svm = -(w_svm[0] * x1_range + b_svm) / w_svm[1]
    ax1.plot(x1_range, x2_svm, 'r-', linewidth=3, label='SVM Decision Boundary')

    # SVM margin boundaries
    x2_svm_pos = -(w_svm[0] * x1_range + b_svm - 1) / w_svm[1]
    x2_svm_neg = -(w_svm[0] * x1_range + b_svm + 1) / w_svm[1]
    ax1.plot(x1_range, x2_svm_pos, 'r--', alpha=0.7, label='SVM Margins')
    ax1.plot(x1_range, x2_svm_neg, 'r--', alpha=0.7)

# Perceptron boundaries
colors = ['blue', 'green', 'orange']
for i, (w_perc, b_perc, margin) in enumerate(perceptron_solutions):
    if abs(w_perc[1]) > 1e-10:
        x2_perc = -(w_perc[0] * x1_range + b_perc) / w_perc[1]
        ax1.plot(x1_range, x2_perc, color=colors[i], linewidth=2,
                linestyle=':', label=f'Perceptron {i+1}')

# Plot data points
for i in range(3):
    color = 'red' if y[i] == 1 else 'blue'
    marker = 'o' if y[i] == 1 else 's'
    ax1.scatter(X[i, 0], X[i, 1], c=color, marker=marker, s=150,
               edgecolors='black', linewidth=2)
    ax1.annotate(f'x_{i+1}', (X[i, 0], X[i, 1]), xytext=(10, 10),
                textcoords='offset points', fontsize=12, fontweight='bold')

ax1.set_xlabel('$x_1$', fontsize=14)
ax1.set_ylabel('$x_2$', fontsize=14)
ax1.set_title('SVM vs Perceptron Decision Boundaries', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xlim(-3, 4)
ax1.set_ylim(-3, 4)

# Plot 2: Margin comparison
methods = ['SVM'] + [f'Perceptron {i+1}' for i in range(len(perceptron_solutions))]
margins = [margin_svm] + [margin for _, _, margin in perceptron_solutions]

bars = ax2.bar(methods, margins, color=['red'] + colors[:len(perceptron_solutions)], alpha=0.7)
ax2.set_ylabel('Margin Width', fontsize=14)
ax2.set_title('Margin Width Comparison', fontsize=14)
ax2.grid(True, alpha=0.3)

# Add value labels on bars
for bar, margin in zip(bars, margins):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{margin:.3f}', ha='center', va='bottom', fontweight='bold')

# Highlight maximum margin
max_margin_idx = np.argmax(margins)
bars[max_margin_idx].set_edgecolor('black')
bars[max_margin_idx].set_linewidth(3)

# Plot 3: Noise robustness
ax3.plot(noise_levels, svm_performance, 'r-o', linewidth=2, markersize=8, label='SVM')
ax3.plot(noise_levels, perceptron_performance, 'b-s', linewidth=2, markersize=8, label='Perceptron')
ax3.set_xlabel('Noise Level', fontsize=14)
ax3.set_ylabel('Margin Width', fontsize=14)
ax3.set_title('Robustness to Noise', fontsize=14)
ax3.grid(True, alpha=0.3)
ax3.legend()

# Plot 4: Computational complexity comparison
dataset_sizes = [10, 50, 100, 500, 1000, 5000]
perceptron_complexity = [n * 2 * 10 for n in dataset_sizes]  # Assuming 10 iterations on average
svm_complexity = [n**2 for n in dataset_sizes]  # Simplified O(n²) for specialized solvers

ax4.loglog(dataset_sizes, perceptron_complexity, 'b-o', linewidth=2, markersize=6, label='Perceptron O(n·d·k)')
ax4.loglog(dataset_sizes, svm_complexity, 'r-s', linewidth=2, markersize=6, label='SVM O(n²)')
ax4.set_xlabel('Dataset Size (n)', fontsize=14)
ax4.set_ylabel('Computational Cost', fontsize=14)
ax4.set_title('Computational Complexity Comparison', fontsize=14)
ax4.grid(True, alpha=0.3)
ax4.legend()

# Add crossover point annotation
crossover_idx = np.where(np.array(svm_complexity) > np.array(perceptron_complexity))[0]
if len(crossover_idx) > 0:
    crossover_n = dataset_sizes[crossover_idx[0]]
    ax4.axvline(x=crossover_n, color='gray', linestyle='--', alpha=0.7)
    ax4.text(crossover_n, max(svm_complexity)/2, f'Crossover\nn≈{crossover_n}',
             ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'svm_vs_perceptron_comparison.png'), dpi=300, bbox_inches='tight')

print(f"Visualization saved to: {os.path.join(save_dir, 'svm_vs_perceptron_comparison.png')}")

print("\n" + "="*80)
print("SUMMARY OF COMPARISON")
print("="*80)

print("\n1. DECISION BOUNDARIES:")
print(f"   - SVM: Unique optimal boundary with maximum margin ({margin_svm:.3f})")
print("   - Perceptron: Multiple possible boundaries depending on initialization")
for i, (_, _, margin) in enumerate(perceptron_solutions):
    print(f"     * Trial {i+1}: margin = {margin:.3f}")

print("\n2. MARGIN ANALYSIS:")
print(f"   - SVM achieves maximum possible margin: {margin_svm:.3f}")
print("   - Perceptron margins are suboptimal:")
for i, (_, _, margin) in enumerate(perceptron_solutions):
    ratio = margin / margin_svm
    print(f"     * Trial {i+1}: {margin:.3f} ({ratio:.1%} of optimal)")

print("\n3. GENERALIZATION:")
print("   - SVM: Better generalization due to maximum margin principle")
print("   - Perceptron: Generalization depends on luck of initialization")
print("   - SVM provides theoretical guarantees, Perceptron does not")

print("\n4. COMPUTATIONAL COMPLEXITY:")
print("   - Small datasets: SVM preferred (better generalization, manageable cost)")
print("   - Large datasets: Perceptron may be preferred (faster training)")
print(f"   - Crossover point: approximately n ≈ {crossover_n} for this problem")

print("\n5. ROBUSTNESS:")
print("   - SVM: More robust to noise and outliers")
print("   - Perceptron: Sensitive to data ordering and noise")
print("   - SVM solution is deterministic, Perceptron is not")

print("\n6. PRACTICAL RECOMMENDATIONS:")
print("   - Use SVM when:")
print("     * Generalization performance is critical")
print("     * Dataset size is manageable (n < 10,000)")
print("     * Computational resources are available")
print("   - Use Perceptron when:")
print("     * Very large datasets (n > 100,000)")
print("     * Online learning is required")
print("     * Computational resources are limited")
print("     * Simple implementation is preferred")

print("\n" + "="*80)
print("SOLUTION COMPLETE")
print("="*80)