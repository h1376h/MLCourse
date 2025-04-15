import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

print("\n=== PMF, PDF, AND CDF EXAMPLES: STEP-BY-STEP SOLUTIONS ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Example 1: Discrete PMF and CDF (Number of Heads in Two Coin Flips)
print("Example 1: Discrete PMF and CDF (Number of Heads in Two Coin Flips)")
x = np.array([0, 1, 2])  # Possible number of heads
probs = np.array([0.25, 0.5, 0.25])  # Corresponding probabilities
print(f"Values (x): {x}")
print(f"Probabilities P(X=x): {probs}")

# Calculate CDF
print("\nStep 1: Calculate the CDF F(x) = P(X ≤ x)")
cdf = np.cumsum(probs)
print("CDF values:")
for i, val in enumerate(cdf):
    print(f"  F({x[i]}) = {val}")

# Create PMF plot
plt.figure(figsize=(10, 6))
plt.bar(x, probs, color='skyblue', alpha=0.7)
plt.grid(True, alpha=0.3)
plt.xlabel('Number of Heads (x)', fontsize=12)
plt.ylabel('Probability P(X=x)', fontsize=12)
plt.title('PMF of Number of Heads in Two Coin Flips', fontsize=14)
plt.xticks(x)
plt.ylim(0, 0.6)

# Add probability values on top of each bar
for i, prob in enumerate(probs):
    plt.text(x[i], prob + 0.02, f'{prob}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'coin_flip_pmf.png'), dpi=100, bbox_inches='tight')
plt.close()

# Create CDF plot
plt.figure(figsize=(10, 6))
plt.step(x, cdf, where='post', color='red', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('Number of Heads (x)', fontsize=12)
plt.ylabel('Cumulative Probability F(x)', fontsize=12)
plt.title('CDF of Number of Heads in Two Coin Flips', fontsize=14)
plt.xticks(x)
plt.ylim(0, 1.1)

# Add CDF values at each point
for i, val in enumerate(cdf):
    plt.text(x[i], val + 0.05, f'{val}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'coin_flip_cdf.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Continuous PDF and CDF (Uniform Distribution)
print("\n\nExample 2: Continuous PDF and CDF (Uniform Distribution)")
x = np.linspace(-0.5, 1.5, 1000)
pdf = np.where((x >= 0) & (x <= 1), 1, 0)
cdf = np.where(x < 0, 0, np.where(x > 1, 1, x))

print("For a uniform distribution on [0,1]:")
print("PDF: f(x) = 1 for 0 ≤ x ≤ 1, 0 otherwise")
print("CDF: F(x) = x for 0 ≤ x ≤ 1, 0 for x < 0, 1 for x > 1")

# Create PDF plot
plt.figure(figsize=(10, 6))
plt.plot(x, pdf, 'b-', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.title('PDF of Uniform Distribution on [0,1]', fontsize=14)
plt.ylim(-0.1, 1.5)

# Add area under curve
plt.fill_between(x, pdf, where=(x >= 0) & (x <= 1), color='skyblue', alpha=0.3)
plt.text(0.5, 0.5, 'Area = 1', ha='center', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'uniform_pdf.png'), dpi=100, bbox_inches='tight')
plt.close()

# Create CDF plot
plt.figure(figsize=(10, 6))
plt.plot(x, cdf, 'r-', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('x', fontsize=12)
plt.ylabel('F(x)', fontsize=12)
plt.title('CDF of Uniform Distribution on [0,1]', fontsize=14)
plt.ylim(-0.1, 1.5)

# Add example points
example_points = [0.2, 0.5, 0.8]
for point in example_points:
    plt.plot([point, point], [0, point], 'k--', alpha=0.5)
    plt.plot([-0.5, point], [point, point], 'k--', alpha=0.5)
    plt.text(point, -0.1, f'x={point}', ha='center', fontsize=10)
    plt.text(-0.4, point, f'F({point})={point}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'uniform_cdf.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Mixed Distribution
print("\n\nExample 3: Mixed Distribution")
x = np.linspace(-0.5, 1.5, 1000)
pmf_point = 0.5  # Probability mass at x=0
pdf_continuous = np.where((x > 0) & (x <= 1), 0.5, 0)  # PDF for continuous part
cdf = np.where(x < 0, 0, np.where(x > 1, 1, 0.5 + 0.5*x))

print("For a mixed distribution:")
print("PMF at x=0: p(0) = 0.5")
print("PDF for 0 < x ≤ 1: f(x) = 0.5")
print("CDF: F(x) = 0.5 + 0.5x for 0 ≤ x ≤ 1, 0 for x < 0, 1 for x > 1")

# Create PMF/PDF plot
plt.figure(figsize=(10, 6))
# Plot the point mass
plt.plot([0], [pmf_point], 'ro', markersize=10)
plt.plot([0, 0], [0, pmf_point], 'r-', linewidth=2)
# Plot the continuous part
plt.plot(x, pdf_continuous, 'b-', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('x', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.title('PMF/PDF of Mixed Distribution', fontsize=14)
plt.ylim(-0.1, 0.6)

# Add labels
plt.text(0, pmf_point + 0.02, 'p(0) = 0.5', ha='center', fontsize=10)
plt.text(0.5, 0.3, 'f(x) = 0.5 for 0 < x ≤ 1', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'mixed_pmf_pdf.png'), dpi=100, bbox_inches='tight')
plt.close()

# Create CDF plot
plt.figure(figsize=(10, 6))
plt.plot(x, cdf, 'g-', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('x', fontsize=12)
plt.ylabel('F(x)', fontsize=12)
plt.title('CDF of Mixed Distribution', fontsize=14)
plt.ylim(-0.1, 1.1)

# Add example points
example_points = [0, 0.5, 1]
for point in example_points:
    plt.plot([point, point], [0, cdf[np.abs(x - point).argmin()]], 'k--', alpha=0.5)
    plt.plot([-0.5, point], [cdf[np.abs(x - point).argmin()], cdf[np.abs(x - point).argmin()]], 'k--', alpha=0.5)
    plt.text(point, -0.05, f'x={point}', ha='center', fontsize=10)
    plt.text(-0.4, cdf[np.abs(x - point).argmin()], f'F({point})={cdf[np.abs(x - point).argmin()]:.2f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'mixed_cdf.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll example images created successfully.") 