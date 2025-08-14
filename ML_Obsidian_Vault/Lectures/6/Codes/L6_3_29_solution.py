import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_29")
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

print("Question 29: Entropy Mathematical Properties")
print("We need to examine entropy calculation for various class distributions")
print("and prove key mathematical properties of entropy.")
print()
print("Tasks:")
print("1. Calculate entropy for different class distributions")
print("2. Determine maximum possible entropy for binary classification")
print("3. Prove that entropy is maximized when classes are equally distributed")
print("4. Handle the log(0) case when calculating entropy")
print()

# Step 2: Entropy Function Implementation
print_step_header(2, "Entropy Function Implementation")

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

print("1. Pure Node: [8, 0]")
pure_counts = {0: 8, 1: 0}
pure_entropy = entropy(pure_counts)
print(f"   Class distribution: {pure_counts}")
print(f"   Probabilities: p(0) = 8/8 = 1.0, p(1) = 0/8 = 0.0")
print(f"   Entropy calculation: H = -(1.0 * log2(1.0) + 0.0 * log2(0.0))")
print(f"   Note: 0 * log2(0) = 0 (by convention)")
print(f"   Entropy: {pure_entropy:.4f} bits")
print()

print("2. Balanced Binary: [4, 4]")
balanced_counts = {0: 4, 1: 4}
balanced_entropy = entropy(balanced_counts)
print(f"   Class distribution: {balanced_counts}")
print(f"   Probabilities: p(0) = 4/8 = 0.5, p(1) = 4/8 = 0.5")
print(f"   Entropy calculation: H = -(0.5 * log2(0.5) + 0.5 * log2(0.5))")
print(f"   H = -(0.5 * (-1) + 0.5 * (-1)) = -(-0.5 - 0.5) = 1.0")
print(f"   Entropy: {balanced_entropy:.4f} bits")
print()

print("3. Highly Skewed: [7, 1]")
skewed_counts = {0: 7, 1: 1}
skewed_entropy = entropy(skewed_counts)
print(f"   Class distribution: {skewed_counts}")
print(f"   Probabilities: p(0) = 7/8 = 0.875, p(1) = 1/8 = 0.125")
print(f"   Entropy calculation: H = -(0.875 * log2(0.875) + 0.125 * log2(0.125))")
print(f"   H = -(0.875 * (-0.193) + 0.125 * (-3.000))")
print(f"   H = -(-0.169 - 0.375) = 0.544")
print(f"   Entropy: {skewed_entropy:.4f} bits")
print()

# Step 4: Maximum Entropy for Binary Classification
print_step_header(4, "Maximum Entropy for Binary Classification")

print("For binary classification with 2 classes:")
print()
print("1. Entropy Range:")
print("   - Minimum: 0 bits (when one class has probability 1)")
print("   - Maximum: 1 bit (when both classes have probability 0.5)")
print()
print("2. Mathematical Verification:")
print("   - Pure distribution [8, 0]: H = 0 bits ✓")
print("   - Balanced distribution [4, 4]: H = 1 bit ✓")
print("   - Skewed distribution [7, 1]: H = 0.544 bits ✓")
print()
print("3. Conclusion:")
print("   Maximum possible entropy for binary classification = 1 bit")
print("   This occurs when classes are equally distributed (p = 0.5)")
print()

# Step 5: Proving Entropy Maximization for Equal Distribution
print_step_header(5, "Proving Entropy Maximization for Equal Distribution")

print("Theorem: For binary classification, entropy H(p) is maximized when p = 0.5")
print()
print("Proof using calculus:")
print()
print("1. Entropy function:")
print("   H(p) = -p * log2(p) - (1-p) * log2(1-p)")
print("   where p is the probability of class 0")
print()
print("2. First derivative:")
print("   dH/dp = -log2(p) - p * (1/(p * ln(2))) + log2(1-p) + (1-p) * (1/((1-p) * ln(2)))")
print("   dH/dp = -log2(p) - 1/ln(2) + log2(1-p) + 1/ln(2)")
print("   dH/dp = -log2(p) + log2(1-p)")
print("   dH/dp = log2((1-p)/p)")
print()
print("3. Find critical points:")
print("   Set dH/dp = 0:")
print("   log2((1-p)/p) = 0")
print("   (1-p)/p = 2^0 = 1")
print("   1-p = p")
print("   1 = 2p")
print("   p = 0.5")
print()
print("4. Second derivative test:")
print("   d²H/dp² = d/dp[log2((1-p)/p)]")
print("   d²H/dp² = d/dp[ln((1-p)/p)/ln(2)]")
print("   d²H/dp² = (1/ln(2)) * d/dp[ln((1-p)/p)]")
print("   d²H/dp² = (1/ln(2)) * d/dp[ln(1-p) - ln(p)]")
print("   d²H/dp² = (1/ln(2)) * [-1/(1-p) - 1/p]")
print("   d²H/dp² = (1/ln(2)) * [-p - (1-p)] / [p(1-p)]")
print("   d²H/dp² = (1/ln(2)) * [-1] / [p(1-p)]")
print("   d²H/dp² = -1 / [ln(2) * p(1-p)]")
print()
print("5. At p = 0.5:")
print("   d²H/dp² = -1 / [ln(2) * 0.5 * 0.5]")
print("   d²H/dp² = -1 / [ln(2) * 0.25]")
print("   d²H/dp² = -4/ln(2) < 0")
print()
print("6. Conclusion:")
print("   Since d²H/dp² < 0 at p = 0.5, this is a maximum")
print("   Therefore, H(p) is maximized at p = 0.5")
print("   Maximum entropy: H(0.5) = 1 bit")
print()

# Step 6: Handling the log(0) Case
print_step_header(6, "Handling the log(0) Case")

print("The log(0) case occurs when calculating entropy for distributions with zero probabilities:")
print()
print("1. Problem Statement:")
print("   - When p_i = 0, we need to calculate log2(0)")
print("   - log2(0) is undefined (approaches -∞)")
print("   - But we multiply by p_i = 0, so we get 0 * (-∞)")
print()
print("2. Mathematical Solution:")
print("   - Use the limit: lim(x→0) x * log2(x) = 0")
print("   - This is a standard result in calculus")
print("   - Justified by L'Hôpital's rule")
print()
print("3. Implementation Approaches:")
print()
print("   a) Convention Method:")
print("      - Define 0 * log2(0) = 0")
print("      - Most common approach in information theory")
print("      - Used by standard libraries")
print()
print("   b) Epsilon Method:")
print("      - Replace 0 with small ε (e.g., 1e-10)")
print("      - Calculate log2(ε) for very small values")
print("      - More numerically stable")
print()
print("   c) Conditional Check:")
print("      - Only calculate log2(p) when p > 0")
print("      - Skip terms with zero probability")
print("      - Most robust approach")
print()
print("4. Code Implementation:")
print("   ```python")
print("   def entropy(class_counts):")
print("       total = sum(class_counts.values())")
print("       if total == 0:")
print("           return 0")
print("       entropy_val = 0")
print("       for count in class_counts.values():")
print("           if count > 0:  # Only process non-zero counts")
print("               p = count / total")
print("               entropy_val -= p * np.log2(p)")
print("       return entropy_val")
print("   ```")
print()

# Step 7: Numerical Verification and Examples
print_step_header(7, "Numerical Verification and Examples")

print("Let's verify our theoretical results with numerical examples:")
print()

# Test various probability distributions
test_cases = [
    ([1.0, 0.0], "Pure class [8, 0]"),
    ([0.5, 0.5], "Balanced [4, 4]"),
    ([0.875, 0.125], "Skewed [7, 1]"),
    ([0.9, 0.1], "Highly skewed [9, 1]"),
    ([0.75, 0.25], "Moderately skewed [6, 2]"),
    ([0.6, 0.4], "Slightly skewed [5, 3]")
]

print("Distribution\t\t\tProbability\t\tEntropy\t\tMax?")
print("-" * 80)

for probs, name in test_cases:
    # Convert to class counts for our entropy function
    counts = {0: int(probs[0] * 8), 1: int(probs[1] * 8)}
    entropy_val = entropy(counts)
    is_max = "Yes" if abs(entropy_val - 1.0) < 1e-6 else "No"
    
    print(f"{name:<20}\t[{probs[0]:.3f}, {probs[1]:.3f}]\t\t{entropy_val:.4f}\t\t{is_max}")

print()

# Step 8: Visualizing Entropy Properties
print_step_header(8, "Visualizing Entropy Properties")

# Calculate entropy values for the distributions we want to plot
entropy_1 = -0.3 * np.log2(0.3) - 0.7 * np.log2(0.7)
entropy_2 = -0.1 * np.log2(0.1) - 0.9 * np.log2(0.9)
max_binary_entropy = 1.0  # Maximum entropy for binary classification

# Create separate plots for better visualization
# Plot 1: Distribution 1
plt.figure(figsize=(8, 6))
plt.bar(['Class A', 'Class B'], [0.3, 0.7], color=['lightcoral', 'lightgreen'], alpha=0.7)
plt.title(r'Distribution 1: $[0.3, 0.7]$')
plt.ylabel('Probability')
plt.ylim(0, 1.1)
plt.text(0.5, 0.5, f'Entropy = {entropy_1:.4f}', 
         ha='center', va='center', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'distribution_1_entropy.png'), dpi=300, bbox_inches='tight')
# plt.show()  # Removed to prevent plots from opening

# Plot 2: Distribution 2
plt.figure(figsize=(8, 6))
plt.bar(['Class A', 'Class B'], [0.1, 0.9], color=['lightcoral', 'lightgreen'], alpha=0.7)
plt.title(r'Distribution 2: $[0.1, 0.9]$')
plt.ylabel('Probability')
plt.ylim(0, 1.1)
plt.text(0.5, 0.5, f'Entropy = {entropy_2:.4f}', 
         ha='center', va='center', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'distribution_2_entropy.png'), dpi=300, bbox_inches='tight')
# plt.show()  # Removed to prevent plots from opening

# Plot 3: Maximum Binary Entropy
plt.figure(figsize=(8, 6))
plt.bar(['Class A', 'Class B'], [0.5, 0.5], color=['lightcoral', 'lightgreen'], alpha=0.7)
plt.title(r'Maximum Binary Entropy: $[0.5, 0.5]$')
plt.ylabel('Probability')
plt.ylim(0, 1.1)
plt.text(0.5, 0.5, f'Entropy = {max_binary_entropy:.4f}', 
         ha='center', va='center', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'max_binary_entropy.png'), dpi=300, bbox_inches='tight')
# plt.show()  # Removed to prevent plots from opening

# Plot 4: Entropy Function
plt.figure(figsize=(10, 6))
p_values = np.linspace(0.01, 0.99, 100)
entropy_values = [-p * np.log2(p) - (1-p) * np.log2(1-p) for p in p_values]
plt.plot(p_values, entropy_values, 'b-', linewidth=2)
plt.title(r'Binary Entropy Function: $H(p) = -p\log_2(p) - (1-p)\log_2(1-p)$')
plt.xlabel('Probability p')
plt.ylabel('Entropy H(p)')
plt.grid(True, alpha=0.3)
plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.7, label=r'$p = 0.5$ (Maximum)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'binary_entropy_function.png'), dpi=300, bbox_inches='tight')
# plt.show()  # Removed to prevent plots from opening

# Step 9: Advanced Mathematical Insights
print_step_header(9, "Advanced Mathematical Insights")

print("Additional Mathematical Properties of Entropy:")
print()
print("1. Concavity:")
print("   - Entropy is a concave function of probabilities")
print("   - This means: H(λp + (1-λ)q) ≥ λH(p) + (1-λ)H(q)")
print("   - Concavity ensures unique maximum")
print()
print("2. Symmetry:")
print("   - H(p, 1-p) = H(1-p, p)")
print("   - Entropy is symmetric around p = 0.5")
print("   - This reflects the fact that class labels are arbitrary")
print()
print("3. Additivity:")
print("   - For independent random variables X and Y:")
print("   - H(X, Y) = H(X) + H(Y)")
print("   - This property is crucial for information theory")
print()
print("4. Bounds:")
print("   - 0 ≤ H(p) ≤ log2(k) for k classes")
print("   - Lower bound: achieved for pure distributions")
print("   - Upper bound: achieved for uniform distributions")
print()

print("Practical Implications:")
print()
print("1. Feature Selection:")
print("   - Features that create balanced splits are preferred")
print("   - Maximum information gain occurs at balanced splits")
print("   - This guides decision tree construction")
print()
print("2. Model Evaluation:")
print("   - Entropy-based metrics are well-behaved")
print("   - No numerical instabilities at boundaries")
print("   - Smooth optimization landscape")
print()
print("3. Algorithm Design:")
print("   - Entropy provides principled splitting criteria")
print("   - Mathematical properties ensure convergence")
print("   - Robust to data variations")
print()

print("Implementation Best Practices:")
print()
print("1. Numerical Stability:")
print("   - Always check for zero probabilities")
print("   - Use conditional calculations")
print("   - Handle edge cases explicitly")
print()
print("2. Performance:")
print("   - Cache entropy calculations when possible")
print("   - Use efficient probability computations")
print("   - Vectorize operations for large datasets")
print()
print("3. Accuracy:")
print("   - Use appropriate precision for probabilities")
print("   - Validate probability distributions")
print("   - Handle floating-point errors gracefully")
print()

print(f"\nPlots saved to: {save_dir}")
print("\nEntropy mathematical properties analysis complete!")
