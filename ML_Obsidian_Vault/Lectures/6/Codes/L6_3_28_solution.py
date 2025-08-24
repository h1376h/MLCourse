import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_28")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Understanding the Problem
print_step_header(1, "Understanding the Problem")

print("Question 28: Entropy Edge Cases and Mathematical Properties")
print("We need to examine entropy calculation for various class distributions")
print("and understand the mathematical properties of entropy.")
print()
print("Tasks:")
print("1. Calculate entropy for different distributions")
print("2. Handle the empty node case mathematically")
print("3. Show that entropy is maximized for balanced distributions")
print("4. Derive maximum possible entropy for k classes")
print()

# Step 2: Entropy Function Definition
print_step_header(2, "Entropy Function Definition")

def entropy(class_counts):
    """
    Calculate entropy for given class distribution.
    
    Args:
        class_counts: Dictionary with class labels as keys and counts as values
    
    Returns:
        float: Entropy value in bits
    """
    total = sum(class_counts.values())
    
    if total == 0:
        return 0  # Handle empty node case
    
    entropy_val = 0
    for count in class_counts.values():
        if count > 0:
            p = count / total
            entropy_val -= p * np.log2(p)
    
    return entropy_val

print("Entropy Function:")
print("H(S) = -Σ(p_i * log2(p_i))")
print("where p_i is the probability of class i")
print()

# Step 3: Calculating Entropy for Different Distributions
print_step_header(3, "Calculating Entropy for Different Distributions")

print("1. Pure Node: [10, 0]")
pure_counts = {0: 10, 1: 0}
pure_entropy = entropy(pure_counts)
print(f"   Class distribution: {pure_counts}")
print(f"   Probabilities: p(0) = 10/10 = 1.0, p(1) = 0/10 = 0.0")
print(f"   Entropy calculation: H = -(1.0 * log2(1.0) + 0.0 * log2(0.0))")
print(f"   Note: 0 * log2(0) = 0 (by convention)")
print(f"   Entropy: {pure_entropy:.4f} bits")
print()

print("2. Balanced Node: [5, 5]")
balanced_counts = {0: 5, 1: 5}
balanced_entropy = entropy(balanced_counts)
print(f"   Class distribution: {balanced_counts}")
print(f"   Probabilities: p(0) = 5/10 = 0.5, p(1) = 5/10 = 0.5")
print(f"   Entropy calculation: H = -(0.5 * log2(0.5) + 0.5 * log2(0.5))")
print(f"   H = -(0.5 * (-1) + 0.5 * (-1)) = -(-0.5 - 0.5) = 1.0")
print(f"   Entropy: {balanced_entropy:.4f} bits")
print()

print("3. Skewed Node: [9, 1]")
skewed_counts = {0: 9, 1: 1}
skewed_entropy = entropy(skewed_counts)
print(f"   Class distribution: {skewed_counts}")
print(f"   Probabilities: p(0) = 9/10 = 0.9, p(1) = 1/10 = 0.1")
print(f"   Entropy calculation: H = -(0.9 * log2(0.9) + 0.1 * log2(0.1))")
print(f"   H = -(0.9 * (-0.152) + 0.1 * (-3.322))")
print(f"   H = -(-0.137 - 0.332) = 0.469")
print(f"   Entropy: {skewed_entropy:.4f} bits")
print()

print("4. Empty Node: [0, 0]")
empty_counts = {0: 0, 1: 0}
empty_entropy = entropy(empty_counts)
print(f"   Class distribution: {empty_counts}")
print(f"   Total samples: 0")
print(f"   Entropy: {empty_entropy:.4f} bits (by definition)")
print()

# Step 4: Handling the Empty Node Case
print_step_header(4, "Handling the Empty Node Case")

print("Mathematical Treatment of Empty Node:")
print()
print("The empty node case [0, 0] presents a mathematical challenge:")
print()
print("1. Problem:")
print("   - We have 0 samples")
print("   - Cannot calculate probabilities (division by zero)")
print("   - log2(0) is undefined")
print()
print("2. Mathematical Solutions:")
print("   a) Convention: Define 0 * log2(0) = 0")
print("      This is the standard approach in information theory")
print("      Justified by the limit: lim(x→0) x * log2(x) = 0")
print()
print("   b) Define entropy of empty set as 0")
print("      An empty set contains no information")
print("      No uncertainty about class assignment")
print()
print("3. Implementation:")
print("   - Check if total samples == 0")
print("   - Return 0 immediately")
print("   - Avoid division by zero errors")
print()

# Step 5: Proving Entropy Maximization for Balanced Distributions
print_step_header(5, "Proving Entropy Maximization for Balanced Distributions")

print("Theorem: For binary classification, entropy is maximized when p = 0.5")
print()
print("Proof:")
print("1. Entropy function: H(p) = -p * log2(p) - (1-p) * log2(1-p)")
print("   where p is the probability of class 0")
print()
print("2. Find critical points by taking derivative:")
print("   dH/dp = -log2(p) - p * (1/(p * ln(2))) + log2(1-p) + (1-p) * (1/((1-p) * ln(2)))")
print("   dH/dp = -log2(p) + log2(1-p)")
print()
print("3. Set derivative to zero:")
print("   -log2(p) + log2(1-p) = 0")
print("   log2(1-p) = log2(p)")
print("   1-p = p")
print("   p = 0.5")
print()
print("4. Verify it's a maximum:")
print("   Second derivative: d²H/dp² = -1/(p * ln(2)) - 1/((1-p) * ln(2))")
print("   At p = 0.5: d²H/dp² = -4/ln(2) < 0 (maximum)")
print()
print("5. Conclusion:")
print("   Entropy is maximized at p = 0.5 with value H(0.5) = 1 bit")
print()

# Step 6: Maximum Entropy for k Classes
print_step_header(6, "Maximum Entropy for k Classes")

print("Theorem: Maximum entropy for k classes is log2(k)")
print()
print("Proof:")
print("1. Entropy function: H = -Σ(p_i * log2(p_i)) for i = 1 to k")
print("2. Constraint: Σ(p_i) = 1 (probabilities sum to 1)")
print()
print("3. Using Lagrange multipliers:")
print("   L = -Σ(p_i * log2(p_i)) + λ(Σ(p_i) - 1)")
print()
print("4. Take partial derivatives:")
print("   ∂L/∂p_i = -log2(p_i) - 1/ln(2) + λ = 0")
print("   -log2(p_i) - 1/ln(2) + λ = 0")
print("   log2(p_i) = λ - 1/ln(2)")
print("   p_i = 2^(λ - 1/ln(2))")
print()
print("5. Since all p_i are equal (by symmetry):")
print("   p_i = 1/k for all i")
print()
print("6. Maximum entropy:")
print("   H_max = -Σ((1/k) * log2(1/k))")
print("   H_max = -k * (1/k) * log2(1/k)")
print("   H_max = -log2(1/k)")
print("   H_max = log2(k)")
print()
print("7. Verification for k = 2:")
print("   H_max = log2(2) = 1 bit ✓")
print("   H_max = log2(4) = 2 bits ✓")
print()

# Step 7: Visualizing Entropy Properties
print_step_header(7, "Visualizing Entropy Properties")

# Create visualizations
# Plot 1: Binary entropy function
plt.figure(figsize=(10, 8))
p_values = np.linspace(0.001, 0.999, 1000)
binary_entropy = -p_values * np.log2(p_values) - (1 - p_values) * np.log2(1 - p_values)

plt.plot(p_values, binary_entropy, 'b-', linewidth=2, label=r'H(p) = -p*log$_2$(p) - (1-p)*log$_2$(1-p)')
plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Maximum: H = 1 bit')
plt.axvline(x=0.5, color='g', linestyle='--', alpha=0.7, label='p = 0.5 (maximum)')
plt.scatter([0.5], [1], color='red', s=100, zorder=5, label='Maximum point')

plt.xlabel('Probability p (class 0)')
plt.ylabel('Entropy H(p) (bits)')
plt.title('Binary Entropy Function')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(0, 1)
plt.ylim(0, 1.1)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'binary_entropy_function.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Entropy for different distributions
plt.figure(figsize=(10, 8))
distributions = ['Pure [10,0]', 'Skewed [9,1]', 'Balanced [5,5]', 'Empty [0,0]']
entropy_values = [pure_entropy, skewed_entropy, balanced_entropy, empty_entropy]
colors = ['red', 'orange', 'green', 'gray']

bars = plt.bar(distributions, entropy_values, color=colors, alpha=0.7)
plt.xlabel('Class Distribution')
plt.ylabel('Entropy (bits)')
plt.title('Entropy for Different Distributions')
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, entropy_val in zip(bars, entropy_values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{entropy_val:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'entropy_distributions.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Maximum entropy for different numbers of classes
plt.figure(figsize=(10, 8))
k_values = np.arange(1, 11)
max_entropy = np.log2(k_values)

plt.plot(k_values, max_entropy, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Classes (k)')
plt.ylabel('Maximum Entropy (bits)')
plt.title('Maximum Entropy vs. Number of Classes')
plt.grid(True, alpha=0.3)

# Add value labels on points
for k, max_ent in zip(k_values, max_entropy):
    plt.annotate(f'{max_ent:.2f}', (k, max_ent), 
                 textcoords="offset points", xytext=(0,10), ha='center')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'max_entropy_vs_classes.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Entropy surface for 3 classes
plt.figure(figsize=(10, 8))
# Create a 3D surface plot for 3-class entropy
p1 = np.linspace(0.001, 0.998, 50)
p2 = np.linspace(0.001, 0.998, 50)
P1, P2 = np.meshgrid(p1, p2)

# Calculate entropy for each (p1, p2) pair
entropy_3d = np.zeros_like(P1)
for i in range(len(p1)):
    for j in range(len(p2)):
        if P1[i, j] + P2[i, j] < 1:  # Valid probability distribution
            p3 = 1 - P1[i, j] - P2[i, j]
            if p3 > 0:
                entropy_3d[i, j] = -P1[i, j] * np.log2(P1[i, j]) - P2[i, j] * np.log2(P2[i, j]) - p3 * np.log2(p3)

# Create contour plot
contour = plt.contourf(P1, P2, entropy_3d, levels=20, cmap='viridis')
plt.xlabel(r'$p_1$ (probability of class 1)')
plt.ylabel(r'$p_2$ (probability of class 2)')
plt.title('3-Class Entropy Surface\n(Entropy in bits)')

# Add colorbar
cbar = plt.colorbar(contour)
cbar.set_label('Entropy (bits)')

# Mark maximum point
plt.scatter([1/3], [1/3], color='red', s=100, marker='*', label=r'Maximum: $p_1=p_2=p_3=1/3$')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'entropy_3d_surface.png'), dpi=300, bbox_inches='tight')
plt.close()



# Step 8: Numerical Verification
print_step_header(8, "Numerical Verification")

print("Verifying our theoretical results with numerical calculations:")
print()

# Test different probability distributions
test_distributions = [
    ([0.5, 0.5], "Balanced binary"),
    ([0.25, 0.25, 0.25, 0.25], "Balanced 4-class"),
    ([0.1, 0.9], "Highly skewed binary"),
    ([0.8, 0.1, 0.1], "Skewed 3-class"),
    ([1.0, 0.0], "Pure class"),
    ([0.0, 0.0], "Empty set")
]

print("Distribution\t\t\tEntropy\t\tExpected Max")
print("-" * 60)

for dist, name in test_distributions:
    if sum(dist) == 0:
        entropy_val = 0
        expected_max = 0
    else:
        # Convert to class counts for our entropy function
        counts = {i: int(p * 100) for i, p in enumerate(dist) if p > 0}
        entropy_val = entropy(counts)
        expected_max = np.log2(len([p for p in dist if p > 0]))
    
    print(f"{name:<20}\t{entropy_val:.4f}\t\t{expected_max:.4f}")

print()

# Step 9: Key Insights and Applications
print_step_header(9, "Key Insights and Applications")

print("Key Mathematical Insights:")
print()
print("1. Entropy Properties:")
print("   - Entropy is always non-negative: H(S) ≥ 0")
print("   - Entropy is maximized for uniform distributions")
print("   - Entropy is minimized for pure distributions")
print("   - Entropy is concave function of probabilities")
print()
print("2. Edge Cases:")
print("   - Empty set: H(∅) = 0 (no uncertainty)")
print("   - Pure set: H({c}) = 0 (complete certainty)")
print("   - Single element: H({x}) = 0 (no randomness)")
print()
print("3. Practical Applications:")
print("   - Feature selection in decision trees")
print("   - Information theory and coding")
print("   - Machine learning evaluation metrics")
print("   - Data compression algorithms")
print()

print("Implementation Considerations:")
print()
print("1. Numerical Stability:")
print("   - Handle log(0) cases carefully")
print("   - Use small epsilon for zero probabilities")
print("   - Check for empty distributions")
print()
print("2. Performance:")
print("   - Cache entropy calculations when possible")
print("   - Use efficient probability calculations")
print("   - Vectorize operations for large datasets")
print()
print("3. Accuracy:")
print("   - Use appropriate precision for probabilities")
print("   - Validate probability distributions")
print("   - Handle floating-point errors")
print()

print("Educational Value:")
print()
print("1. Understanding Information Theory:")
print("   - Entropy measures uncertainty")
print("   - Higher entropy = more uncertainty")
print("   - Maximum entropy = maximum uncertainty")
print()
print("2. Decision Tree Learning:")
print("   - Information gain = reduction in entropy")
print("   - Best splits maximize information gain")
print("   - Entropy guides feature selection")
print()
print("3. Mathematical Foundations:")
print("   - Lagrange multipliers for optimization")
print("   - Calculus for finding extrema")
print("   - Probability theory fundamentals")
print()

print(f"\nPlots saved to: {save_dir}")
print("\nEntropy analysis complete!")


