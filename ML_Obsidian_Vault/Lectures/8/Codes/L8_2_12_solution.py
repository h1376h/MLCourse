import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import binom
from scipy.optimize import fsolve

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_2_Quiz_12")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'

print("Question 12: The Curse of Dimensionality and Feature Selection")
print("=" * 70)

# Given probability function
def prob_relevant(n):
    """Probability of a feature being relevant as a function of dimensionality"""
    return 0.1 * (0.95 ** n)

# Let's verify the function behavior
print("Verifying probability function:")
for n in [1, 5, 10, 20, 30, 40, 50]:
    p = prob_relevant(n)
    expected = n * p
    print(f"n = {n:2d}: P(relevant) = {p:.6f}, Expected relevant = {expected:.3f}")

print("\nGiven probability function: P(relevant) = 0.1 * 0.95^n")
print("where n is the number of features")

# Task 1: How does high dimensionality affect univariate methods?
print("\n1. How does high dimensionality affect univariate methods?")
print("-" * 60)

print("High dimensionality affects univariate methods in several ways:")
print("• Statistical power decreases: With more features, individual feature tests")
print("  become less reliable due to multiple testing problems")
print("• False positive rate increases: More features mean more chances for")
print("  spurious correlations to appear significant")
print("• Feature independence assumption becomes violated: In high dimensions,")
print("  features are often correlated, making univariate analysis misleading")
print("• Curse of dimensionality: The volume of the feature space grows")
print("  exponentially, making distance-based measures less meaningful")

# Task 2: What happens to feature relevance as dimensions increase?
print("\n2. What happens to feature relevance as dimensions increase?")
print("-" * 60)

print("As dimensions increase:")
print("• Individual feature relevance decreases exponentially")
print("• The probability of any single feature being relevant diminishes")
print("• More features become noise rather than signal")
print("• The ratio of relevant to irrelevant features decreases dramatically")

# Visualize the probability decay
n_values = np.arange(0, 100, 1)
prob_values = prob_relevant(n_values)

plt.figure(figsize=(12, 8))
plt.plot(n_values, prob_values, 'b-', linewidth=2, label='P(relevant) = 0.1 * 0.95^n')
plt.axhline(y=0.01, color='r', linestyle='--', alpha=0.7, label='P = 0.01 threshold')
plt.axhline(y=0.001, color='orange', linestyle='--', alpha=0.7, label='P = 0.001 threshold')

plt.xlabel('Number of Features (n)')
plt.ylabel('Probability of Feature Being Relevant')
plt.title('Feature Relevance Decay with Dimensionality')
plt.grid(True, alpha=0.3)
plt.legend()
plt.yscale('log')
plt.ylim(1e-4, 0.2)

# Add annotations
plt.annotate('Rapid decay in early dimensions', xy=(10, 0.06), xytext=(20, 0.1),
            arrowprops=dict(arrowstyle='->', lw=1.5), fontsize=10)
plt.annotate('Very low relevance\nin high dimensions', xy=(50, 0.001), xytext=(60, 0.01),
            arrowprops=dict(arrowstyle='->', lw=1.5), fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_relevance_decay.png'), dpi=300, bbox_inches='tight')

# Task 3: Calculate expected number of relevant features
print("\n3. Expected number of relevant features for different dimensionalities")
print("-" * 70)

datasets = [100, 1000, 10000]
print("Dataset sizes and expected relevant features:")
print(f"{'Features':<12} {'P(relevant)':<15} {'Expected Relevant':<20} {'% Relevant':<12}")
print("-" * 70)

expected_relevant = []
for n in datasets:
    p = prob_relevant(n)
    expected = n * p
    expected_relevant.append(expected)
    percent = (expected / n) * 100
    print(f"{n:<12} {p:<15.6f} {expected:<20.2f} {percent:<12.4f}%")

# Visualize expected relevant features vs total features
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.bar(range(len(datasets)), expected_relevant, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.7)
plt.xlabel('Dataset Size')
plt.ylabel('Expected Number of Relevant Features')
plt.title('Expected Relevant Features vs Dataset Size')
plt.xticks(range(len(datasets)), [f'{n:,}' for n in datasets])
plt.grid(True, alpha=0.3)

# Add value labels on bars
for i, v in enumerate(expected_relevant):
    plt.text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')

plt.subplot(2, 1, 2)
percentages = [(exp/n)*100 for exp, n in zip(expected_relevant, datasets)]
plt.bar(range(len(datasets)), percentages, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.7)
plt.xlabel('Dataset Size')
plt.ylabel('Percentage of Features that are Relevant (%)')
plt.title('Percentage of Relevant Features vs Dataset Size')
plt.xticks(range(len(datasets)), [f'{n:,}' for n in datasets])
plt.grid(True, alpha=0.3)

# Add value labels on bars
for i, v in enumerate(percentages):
    plt.text(i, v + 0.001, f'{v:.4f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'expected_relevant_features.png'), dpi=300, bbox_inches='tight')

# Task 4: Maximum dimensionality for at least 5 relevant features
print("\n4. Maximum dimensionality for at least 5 relevant features")
print("-" * 60)

# Find the maximum expected relevant features
max_expected = 0
n_max = 0
for n in range(1, 100):
    expected = n * prob_relevant(n)
    if expected > max_expected:
        max_expected = expected
        n_max = n

print(f"Analysis of the probability function:")
print(f"P(relevant) = 0.1 * 0.95^n")
print(f"Expected relevant features = n * P(relevant)")

print(f"\nMaximum expected relevant features occurs at n = {n_max}")
print(f"Maximum expected relevant features = {max_expected:.3f}")

print(f"\nThis means it's impossible to have 5+ relevant features")
print(f"with the given probability function!")

print(f"\nThe reason is that the exponential decay (0.95^n) is too rapid.")
print(f"Even at n = 1, we only get 0.095 expected relevant features.")
print(f"At n = 20, we get the maximum of {max_expected:.3f} expected relevant features.")

print(f"\nTo achieve 5+ relevant features, we would need:")
print(f"P(relevant) >= 5/n for some n")
print(f"0.1 * 0.95^n >= 5/n")
print(f"0.95^n >= 50/n")

# Check if there's any n where this inequality holds
found_solution = False
for n in range(1, 1000):
    if 0.95**n >= 50/n:
        print(f"\nThe inequality 0.95^n >= 50/n holds for n = {n}")
        found_solution = True
        break

if not found_solution:
    print(f"\nThe inequality 0.95^n >= 50/n never holds!")
    print(f"This confirms that 5+ relevant features is impossible.")

# Visualize the relationship
n_range = np.arange(1, 100, 1)
expected_range = n_range * prob_relevant(n_range)

plt.figure(figsize=(12, 8))
plt.plot(n_range, expected_range, 'b-', linewidth=2, label='Expected Relevant Features')
plt.axhline(y=5, color='r', linestyle='--', linewidth=2, label='Target: 5 relevant features')
plt.axvline(x=n_max, color='g', linestyle='--', alpha=0.7, 
           label=f'Maximum expected at n = {n_max}')

plt.xlabel('Number of Features (n)')
plt.ylabel('Expected Number of Relevant Features')
plt.title('Expected Relevant Features vs Dimensionality')
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(0, 1)

# Add annotation
plt.annotate(f'Maximum expected relevant features\n= {max_expected:.3f} at n = {n_max}', 
            xy=(n_max, max_expected), xytext=(n_max + 10, max_expected + 0.1),
            arrowprops=dict(arrowstyle='->', lw=1.5), fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

# Add text about impossibility
plt.text(0.5, 0.8, 'Impossible to achieve 5+ relevant features\nwith the given probability function!', 
         transform=plt.gca().transAxes, ha='center', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.5", fc="lightcoral", ec="red", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'maximum_dimensionality_threshold.png'), dpi=300, bbox_inches='tight')

# Additional analysis: Show the mathematical relationship
print("\nMathematical Analysis:")
print("-" * 30)
print("The expected number of relevant features is:")
print("E[relevant] = n * P(relevant)")
print("E[relevant] = n * 0.1 * 0.95^n")

print("\nTo find the maximum n where E[relevant] >= 5:")
print("n * 0.1 * 0.95^n >= 5")
print("0.1 * 0.95^n >= 5/n")
print("0.95^n >= 50/n")

print("\nThis is a transcendental equation that requires numerical solution.")
print("The solution gives us the maximum dimensionality where we can")
print("reasonably expect to have at least 5 relevant features.")

# Show the decay pattern more clearly
print("\nDetailed Analysis of the Decay Pattern:")
print("-" * 50)
print("Let's examine how the probability changes:")
for n in [1, 5, 10, 20, 50, 100]:
    p = prob_relevant(n)
    print(f"n = {n:3d}: P(relevant) = {p:.6f} = {p*100:.4f}%")

# Task 5: Modified scenario with different probability function
print("\n" + "="*70)
print("5. Modified Scenario: P(relevant) = 0.3 × 0.98^n")
print("="*70)

# Define the modified probability function
def prob_relevant_modified(n):
    """Modified probability function with slower decay"""
    return 0.3 * (0.98 ** n)

print("Step-by-step mathematical analysis:")
print("-" * 50)

print("\nGiven: P(relevant) = 0.3 × 0.98^n")
print("Expected relevant features: E[relevant] = n × P(relevant) = n × 0.3 × 0.98^n")

# Task 5a: Calculate expected number for specific values
print("\n5a. Expected number of relevant features for n = 50, 100, 200:")
print("-" * 60)

target_ns = [50, 100, 200]
print(f"{'n':<6} {'P(relevant)':<15} {'Expected Relevant':<20} {'% Relevant':<12}")
print("-" * 60)

modified_expected = []
for n in target_ns:
    p = prob_relevant_modified(n)
    expected = n * p
    modified_expected.append(expected)
    percent = (expected / n) * 100
    print(f"{n:<6} {p:<15.6f} {expected:<20.2f} {percent:<12.4f}%")

# Mathematical derivation for each case
print("\nDetailed calculations:")
for i, n in enumerate(target_ns):
    p = prob_relevant_modified(n)
    expected = n * p
    print(f"\nFor n = {n}:")
    print(f"  P(relevant) = 0.3 × 0.98^{n}")
    print(f"  P(relevant) = 0.3 × {0.98**n:.6f} = {p:.6f}")
    print(f"  E[relevant] = {n} × {p:.6f} = {expected:.2f}")

# Task 5b: Find maximum dimensionality for 5+ relevant features
print("\n5b. Maximum dimensionality for at least 5 relevant features:")
print("-" * 60)

print("Mathematical approach:")
print("We need: E[relevant] ≥ 5")
print("n × 0.3 × 0.98^n ≥ 5")
print("0.3 × 0.98^n ≥ 5/n")
print("0.98^n ≥ 5/(0.3×n)")
print("0.98^n ≥ 16.667/n")

# Find the solution using calculus approach
print("\nTo find the maximum, we differentiate E[relevant] = n × 0.3 × 0.98^n")
print("dE/dn = 0.3 × [0.98^n + n × 0.98^n × ln(0.98)]")
print("dE/dn = 0.3 × 0.98^n × [1 + n × ln(0.98)]")
print("Setting dE/dn = 0:")
print("1 + n × ln(0.98) = 0")
ln_098 = np.log(0.98)
n_critical = -1 / ln_098
print(f"n_critical = -1/ln(0.98) = -1/{ln_098:.6f} = {n_critical:.2f}")

# Find the actual maximum by searching
max_expected_modified = 0
n_max_modified = 0
for n in range(1, 200):
    expected = n * prob_relevant_modified(n)
    if expected > max_expected_modified:
        max_expected_modified = expected
        n_max_modified = n

print(f"\nNumerical verification:")
print(f"Maximum E[relevant] = {max_expected_modified:.2f} at n = {n_max_modified}")

# Find where E[relevant] >= 5
n_threshold = 0
for n in range(1, 500):
    if n * prob_relevant_modified(n) >= 5:
        n_threshold = n
        break

if n_threshold > 0:
    expected_at_threshold = n_threshold * prob_relevant_modified(n_threshold)
    print(f"\nMaximum dimensionality for 5+ relevant features: n = {n_threshold}")
    print(f"At n = {n_threshold}:")
    print(f"  P(relevant) = 0.3 × 0.98^{n_threshold} = {prob_relevant_modified(n_threshold):.6f}")
    print(f"  E[relevant] = {n_threshold} × {prob_relevant_modified(n_threshold):.6f} = {expected_at_threshold:.2f}")
    
    # Verify it drops below 5 after this point
    next_n = n_threshold + 1
    next_expected = next_n * prob_relevant_modified(next_n)
    if next_expected < 5:
        print(f"\nVerification - At n = {next_n}:")
        print(f"  E[relevant] = {next_expected:.2f} < 5 ✓")
else:
    print("\nNo solution found - 5+ relevant features not achievable")

# Task 5c: Comparison with original function
print("\n5c. Comparison with original function P(relevant) = 0.1 × 0.95^n:")
print("-" * 70)

print("Mathematical comparison:")
print("Original:  P₁(relevant) = 0.1 × 0.95^n")
print("Modified:  P₂(relevant) = 0.3 × 0.98^n")
print("\nKey differences:")
print("1. Initial probability: 0.3 vs 0.1 (3× higher)")
print("2. Decay rate: 0.98 vs 0.95 (slower decay)")

# Compare decay rates
print(f"\nDecay rate comparison:")
print(f"Original decay per step: {0.95:.2f} ({(1-0.95)*100:.0f}% reduction)")
print(f"Modified decay per step: {0.98:.2f} ({(1-0.98)*100:.0f}% reduction)")

# Show comparison table
print(f"\nComparison table:")
print(f"{'n':<6} {'Original P':<12} {'Original E':<12} {'Modified P':<12} {'Modified E':<12}")
print("-" * 70)
comparison_ns = [10, 20, 30, 50, 100]
for n in comparison_ns:
    p1 = prob_relevant(n)
    e1 = n * p1
    p2 = prob_relevant_modified(n)
    e2 = n * p2
    print(f"{n:<6} {p1:<12.6f} {e1:<12.2f} {p2:<12.6f} {e2:<12.2f}")

# Visualize the comparison
plt.figure(figsize=(15, 10))

# Plot 1: Probability comparison
plt.subplot(2, 2, 1)
n_range = np.arange(1, 150, 1)
prob_original = prob_relevant(n_range)
prob_modified = prob_relevant_modified(n_range)

plt.plot(n_range, prob_original, 'r-', linewidth=2, label='Original: 0.1 * 0.95^n')
plt.plot(n_range, prob_modified, 'b-', linewidth=2, label='Modified: 0.3 * 0.98^n')
plt.xlabel('Number of Features (n)')
plt.ylabel('P(relevant)')
plt.title('Probability Function Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Plot 2: Expected relevant features comparison
plt.subplot(2, 2, 2)
expected_original = n_range * prob_original
expected_modified = n_range * prob_modified

plt.plot(n_range, expected_original, 'r-', linewidth=2, label='Original function')
plt.plot(n_range, expected_modified, 'b-', linewidth=2, label='Modified function')
plt.axhline(y=5, color='g', linestyle='--', linewidth=2, label='Target: 5 relevant features')
plt.xlabel('Number of Features (n)')
plt.ylabel('Expected Relevant Features')
plt.title('Expected Relevant Features Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 20)

# Plot 3: Modified function analysis
plt.subplot(2, 2, 3)
n_range_detailed = np.arange(1, 100, 1)
expected_modified_detailed = n_range_detailed * prob_relevant_modified(n_range_detailed)

plt.plot(n_range_detailed, expected_modified_detailed, 'b-', linewidth=2, label='E[relevant] = n * 0.3 * 0.98^n')
plt.axhline(y=5, color='r', linestyle='--', linewidth=2, label='Target: 5 relevant features')
if n_threshold > 0:
    plt.axvline(x=n_threshold, color='g', linestyle='--', alpha=0.7, label=f'Solution: n = {n_threshold}')
plt.axvline(x=n_max_modified, color='orange', linestyle='--', alpha=0.7, label=f'Maximum at n = {n_max_modified}')

plt.xlabel('Number of Features (n)')
plt.ylabel('Expected Relevant Features')
plt.title('Modified Function: Detailed Analysis')
plt.legend()
plt.grid(True, alpha=0.3)

# Add annotation
if n_threshold > 0:
    plt.annotate(f'Solution: n = {n_threshold}\nE[relevant] = {expected_at_threshold:.1f}', 
                xy=(n_threshold, 5), xytext=(n_threshold + 10, 8),
                arrowprops=dict(arrowstyle='->', lw=1.5), fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="green", alpha=0.8))

# Plot 4: Ratio comparison
plt.subplot(2, 2, 4)
ratio = expected_modified / np.maximum(expected_original, 1e-10)  # Avoid division by zero
plt.plot(n_range, ratio, 'purple', linewidth=2, label='Modified/Original ratio')
plt.xlabel('Number of Features (n)')
plt.ylabel('Ratio (Modified/Original)')
plt.title('Improvement Factor of Modified Function')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'modified_function_analysis.png'), dpi=300, bbox_inches='tight')

print(f"\nSummary of Task 5:")
print(f"• Modified function allows {max_expected_modified:.2f} maximum expected relevant features")
print(f"• Maximum dimensionality for 5+ relevant features: {n_threshold if n_threshold > 0 else 'Not achievable'}")
print(f"• The slower decay rate (0.98 vs 0.95) and higher initial probability (0.3 vs 0.1)")
print(f"  make it possible to achieve the target of 5+ relevant features")

print(f"\nPlots saved to: {save_dir}")
print("\nOverall Summary:")
print(f"• Original function: Maximum {max_expected:.3f} relevant features at n = {n_max}")
print(f"• Modified function: Maximum {max_expected_modified:.2f} relevant features at n = {n_max_modified}")
print(f"• Target achievement: Original = Impossible, Modified = Possible at n ≤ {n_threshold if n_threshold > 0 else 'N/A'}")
print(f"• This demonstrates how parameter choices critically affect feature selection viability")
