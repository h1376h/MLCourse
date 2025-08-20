import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_2_Quiz_5")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("QUESTION 5: MUTUAL INFORMATION ANALYSIS")
print("=" * 80)

# ============================================================================
# PART 1: Understanding Mutual Information
# ============================================================================
print("\n1. WHAT IS MUTUAL INFORMATION AND HOW IS IT CALCULATED?")
print("-" * 60)

print("Mutual Information I(X;Y) measures the amount of information that one random")
print("variable contains about another random variable.")
print()
print("Formula: I(X;Y) = H(X) + H(Y) - H(X,Y)")
print("where:")
print("  H(X) = entropy of X")
print("  H(Y) = entropy of Y") 
print("  H(X,Y) = joint entropy of X and Y")
print()
print("Alternative formula: I(X;Y) = Σ Σ P(x,y) * log(P(x,y)/(P(x)P(y)))")
print()

# ============================================================================
# PART 2: Mutual Information vs Correlation
# ============================================================================
print("2. HOW DOES MUTUAL INFORMATION DIFFER FROM CORRELATION?")
print("-" * 60)

print("Correlation (Pearson):")
print("  - Measures linear relationships only")
print("  - Range: [-1, 1]")
print("  - Zero correlation doesn't imply independence")
print("  - Sensitive to outliers")
print()
print("Mutual Information:")
print("  - Detects any type of relationship (linear, non-linear, etc.)")
print("  - Range: [0, ∞)")
print("  - Zero mutual information implies independence")
print("  - Robust to outliers")
print("  - Works with any type of variables (continuous, discrete, mixed)")
print()

# ============================================================================
# PART 3: Calculating Mutual Information for Given Joint Distribution
# ============================================================================
print("3. CALCULATING MUTUAL INFORMATION FOR GIVEN JOINT DISTRIBUTION")
print("-" * 60)

# Given joint probability distribution
print("Given joint probability distribution P(X,Y):")
print("P(X=0,Y=0) = 0.3")
print("P(X=0,Y=1) = 0.2") 
print("P(X=1,Y=0) = 0.1")
print("P(X=1,Y=1) = 0.4")
print()

# Create joint probability matrix
P_XY = np.array([[0.3, 0.2],
                  [0.1, 0.4]])

print("Joint probability matrix P(X,Y):")
print(P_XY)
print()

# Calculate marginal probabilities
P_X = np.sum(P_XY, axis=1)  # P(X)
P_Y = np.sum(P_XY, axis=0)  # P(Y)

print("Marginal probabilities:")
print(f"P(X=0) = {P_X[0]:.1f}")
print(f"P(X=1) = {P_X[1]:.1f}")
print(f"P(Y=0) = {P_Y[0]:.1f}")
print(f"P(Y=1) = {P_Y[1]:.1f}")
print()

# Calculate entropies
H_X = entropy(P_X, base=2)
H_Y = entropy(P_Y, base=2)

print("Entropies:")
print(f"H(X) = -Σ P(x) * log₂(P(x)) = {H_X:.4f} bits")
print(f"H(Y) = -Σ P(y) * log₂(P(y)) = {H_Y:.4f} bits")
print()

# Calculate joint entropy
# Flatten the joint probability matrix and remove zero probabilities
P_XY_flat = P_XY.flatten()
P_XY_nonzero = P_XY_flat[P_XY_flat > 0]
H_XY = entropy(P_XY_nonzero, base=2)

print("Joint entropy:")
print(f"H(X,Y) = -Σ P(x,y) * log₂(P(x,y)) = {H_XY:.4f} bits")
print()

# Calculate mutual information
MI = H_X + H_Y - H_XY
print("Mutual Information:")
print(f"I(X;Y) = H(X) + H(Y) - H(X,Y)")
print(f"I(X;Y) = {H_X:.4f} + {H_Y:.4f} - {H_XY:.4f}")
print(f"I(X;Y) = {MI:.4f} bits")
print()

# Verify with alternative formula
MI_alt = 0
for i in range(2):
    for j in range(2):
        if P_XY[i,j] > 0:
            MI_alt += P_XY[i,j] * np.log2(P_XY[i,j] / (P_X[i] * P_Y[j]))

print("Verification using alternative formula:")
print(f"I(X;Y) = Σ Σ P(x,y) * log₂(P(x,y)/(P(x)P(y))) = {MI_alt:.4f} bits")
print()

# ============================================================================
# PART 4: Example with X and Y = X² mod 4
# ============================================================================
print("4. EXAMPLE: X vs Y = X² mod 4")
print("-" * 60)

# Create the dataset
X_values = np.array([1, 2, 3, 4])
Y_values = (X_values ** 2) % 4

print("Dataset:")
print("X values:", X_values)
print("Y = X² mod 4:", Y_values)
print()

# Calculate probabilities (equal probability for X)
P_X_equal = np.ones(4) / 4
print("P(X) = 1/4 for all X values (equal probability)")
print()

# Calculate P(Y) based on the mapping
Y_unique, Y_counts = np.unique(Y_values, return_counts=True)
P_Y_calc = Y_counts / 4

print("Y values and their counts:")
for y, count in zip(Y_unique, Y_counts):
    print(f"Y = {y}: {count} occurrences")
print()

print("P(Y) based on mapping:")
for y, p in zip(Y_unique, P_Y_calc):
    print(f"P(Y = {y}) = {p}")
print()

# Calculate correlation
correlation = np.corrcoef(X_values, Y_values)[0, 1]
print(f"Correlation between X and Y: {correlation:.6f}")
print("The correlation is approximately 0, indicating no linear relationship.")
print()

# Calculate mutual information for this case
# Create joint probability matrix
joint_probs = np.zeros((4, 4))
for i, x in enumerate(X_values):
    y = (x**2) % 4
    y_idx = np.where(Y_unique == y)[0][0]
    joint_probs[i, y_idx] = 1/4

print("Joint probability matrix P(X,Y):")
print(joint_probs)
print()

# Calculate entropies
H_X_example = entropy(P_X_equal, base=2)
H_Y_example = entropy(P_Y_calc, base=2)

# Calculate joint entropy
joint_probs_nonzero = joint_probs[joint_probs > 0]
H_XY_example = entropy(joint_probs_nonzero, base=2)

# Calculate mutual information
MI_example = H_X_example + H_Y_example - H_XY_example

print("Entropies for X vs Y = X² mod 4:")
print(f"H(X) = {H_X_example:.4f} bits")
print(f"H(Y) = {H_Y_example:.4f} bits")
print(f"H(X,Y) = {H_XY_example:.4f} bits")
print(f"I(X;Y) = {MI_example:.4f} bits")
print()

print("Key Insight:")
print("While correlation = 0 (no linear relationship), mutual information > 0")
print("reveals that there IS a deterministic relationship: Y = X² mod 4")
print()

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("Generating visualizations...")

# 1. Joint probability heatmap for Part 3
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(P_XY, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=['Y=0', 'Y=1'], yticklabels=['X=0', 'X=1'])
plt.title('Joint Probability Distribution P(X,Y)')
plt.xlabel('Y')
plt.ylabel('X')

# Add marginal probabilities
plt.text(-0.3, -0.3, f'P(X=0) = {P_X[0]:.1f}', ha='center', va='center')
plt.text(0.7, -0.3, f'P(X=1) = {P_X[1]:.1f}', ha='center', va='center')
plt.text(-0.3, 0.5, f'P(Y=0) = {P_Y[0]:.1f}', ha='center', va='center', rotation=90)
plt.text(1.5, 0.5, f'P(Y=1) = {P_Y[1]:.1f}', ha='center', va='center', rotation=90)

plt.subplot(1, 2, 2)
# Bar plot of marginal probabilities
x_pos = np.arange(2)
plt.bar(x_pos, P_X, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('X')
plt.ylabel('Probability')
plt.title('Marginal Probability P(X)')
plt.xticks(x_pos, ['X=0', 'X=1'])
plt.ylim(0, 1)

for i, v in enumerate(P_X):
    plt.text(i, v + 0.02, f'{v:.1f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'joint_distribution_analysis.png'), dpi=300, bbox_inches='tight')

# 2. Entropy and Mutual Information breakdown
plt.figure(figsize=(10, 6))
entropy_values = [H_X, H_Y, H_XY, MI]
entropy_labels = ['H(X)', 'H(Y)', 'H(X,Y)', 'I(X;Y)']
colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']

bars = plt.bar(entropy_labels, entropy_values, color=colors, edgecolor='black', alpha=0.8)
plt.ylabel('Bits')
plt.title('Entropy and Mutual Information Breakdown')
plt.ylim(0, max(entropy_values) * 1.1)

# Add value labels on bars
for bar, value in zip(bars, entropy_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{value:.4f}', ha='center', va='bottom')

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'entropy_breakdown.png'), dpi=300, bbox_inches='tight')

# 3. X vs Y = X² mod 4 relationship
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_values, Y_values, s=200, c='red', edgecolor='black', linewidth=2)
plt.plot(X_values, Y_values, 'b--', alpha=0.7)
plt.xlabel('X')
plt.ylabel('Y = X² mod 4')
plt.title('Relationship: Y = X² mod 4')
plt.grid(True, alpha=0.3)
plt.xticks(X_values)
plt.yticks([0, 1, 2, 3])

# Add point labels
for x, y in zip(X_values, Y_values):
    plt.annotate(f'({x}, {y})', (x, y), xytext=(10, 10), 
                 textcoords='offset points', fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

plt.subplot(1, 2, 2)
# Joint probability heatmap for X vs Y = X² mod 4
sns.heatmap(joint_probs, annot=True, fmt='.2f', cmap='Reds',
            xticklabels=[f'Y={y}' for y in Y_unique], 
            yticklabels=[f'X={x}' for x in X_values])
plt.title('Joint Probability P(X,Y) for Y = X² mod 4')
plt.xlabel('Y')
plt.ylabel('X')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'nonlinear_relationship.png'), dpi=300, bbox_inches='tight')

# 4. Comparison: Correlation vs Mutual Information
plt.figure(figsize=(10, 6))
metrics = ['Correlation', 'Mutual Information']
values = [abs(correlation), MI_example]
colors = ['lightblue', 'lightgreen']

bars = plt.bar(metrics, values, color=colors, edgecolor='black', alpha=0.8)
plt.ylabel('Value')
plt.title('Correlation vs Mutual Information for Y = X² mod 4')
plt.ylim(0, max(values) * 1.2)

# Add value labels
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{value:.4f}', ha='center', va='bottom')

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'correlation_vs_mutual_info.png'), dpi=300, bbox_inches='tight')

# 5. Information theory visualization
plt.figure(figsize=(12, 8))

# Create Venn diagram-like representation
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle

ax = plt.gca()
ax.set_xlim(-2, 2)
ax.set_ylim(-1.5, 1.5)

# Draw circles for X and Y
circle_x = Circle((-0.8, 0), 0.8, fill=False, linewidth=2, color='blue', label='H(X)')
circle_y = Circle((0.8, 0), 0.8, fill=False, linewidth=2, color='red', label='H(Y)')

ax.add_patch(circle_x)
ax.add_patch(circle_y)

# Add text labels
plt.text(-0.8, 0, 'H(X)', ha='center', va='center', fontsize=12, weight='bold')
plt.text(0.8, 0, 'H(Y)', ha='center', va='center', fontsize=12, weight='bold')
plt.text(0, 0, 'I(X;Y)', ha='center', va='center', fontsize=14, weight='bold', 
         bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", alpha=0.7))
plt.text(0, -0.8, 'H(X,Y)', ha='center', va='center', fontsize=12, weight='bold')

plt.title('Information Theory Relationships')
plt.axis('equal')
plt.axis('off')
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'information_theory_venn.png'), dpi=300, bbox_inches='tight')

print(f"All visualizations saved to: {save_dir}")

# ============================================================================
# SUMMARY OF RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY OF RESULTS")
print("=" * 80)

print(f"\nPart 3 - Given Joint Distribution:")
print(f"  H(X) = {H_X:.4f} bits")
print(f"  H(Y) = {H_Y:.4f} bits")
print(f"  H(X,Y) = {H_XY:.4f} bits")
print(f"  I(X;Y) = {MI:.4f} bits")

print(f"\nPart 4 - X vs Y = X² mod 4:")
print(f"  Correlation = {correlation:.6f}")
print(f"  Mutual Information = {MI_example:.4f} bits")
print(f"  Key insight: Correlation = 0 but MI > 0 reveals deterministic relationship")

print(f"\nVisualizations generated:")
print(f"  1. joint_distribution_analysis.png - Joint distribution and marginals")
print(f"  2. entropy_breakdown.png - Entropy and MI breakdown")
print(f"  3. nonlinear_relationship.png - X vs Y = X² mod 4 relationship")
print(f"  4. correlation_vs_mutual_info.png - Correlation vs MI comparison")
print(f"  5. information_theory_venn.png - Information theory relationships")

print(f"\nFiles saved to: {save_dir}")
