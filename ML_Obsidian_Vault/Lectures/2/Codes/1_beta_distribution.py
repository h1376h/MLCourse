import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

print("\n=== BETA DISTRIBUTION EXAMPLES: STEP-BY-STEP SOLUTIONS ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Example 1: Basic Properties of Beta Distribution
print("Example 1: Basic Properties of Beta Distribution")
x = np.linspace(0, 1, 1000)
alpha, beta = 2, 5
pdf = stats.beta.pdf(x, alpha, beta)
cdf = stats.beta.cdf(x, alpha, beta)

print(f"Parameters:")
print(f"  Alpha (α) = {alpha}")
print(f"  Beta (β) = {beta}")

# Calculate key properties
mean = alpha / (alpha + beta)
mode = (alpha - 1) / (alpha + beta - 2) if alpha > 1 and beta > 1 else "NA"
variance = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))

print(f"\nCalculated Properties:")
print(f"  Mean: {mean:.4f}")
print(f"  Mode: {mode if isinstance(mode, str) else f'{mode:.4f}'}")
print(f"  Variance: {variance:.4f}")

# Plot PDF and CDF
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x, pdf, 'b-', linewidth=2)
plt.fill_between(x, pdf, alpha=0.2)
plt.title('Beta Distribution PDF')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(x, cdf, 'r-', linewidth=2)
plt.title('Beta Distribution CDF')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'beta_distribution_basic.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Different Parameter Combinations
print("\nExample 2: Different Parameter Combinations")
params = [
    (1, 1, "Uniform"),
    (0.5, 0.5, "Jeffrey's Prior"),
    (2, 2, "Symmetric"),
    (5, 2, "Right-skewed"),
    (2, 5, "Left-skewed")
]

plt.figure(figsize=(10, 6))
for alpha, beta, label in params:
    pdf = stats.beta.pdf(x, alpha, beta)
    plt.plot(x, pdf, linewidth=2, label=f'α={alpha}, β={beta} ({label})')

plt.title('Beta Distributions with Different Parameters')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'beta_distribution_parameters.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Bayesian Updating
print("\nExample 3: Bayesian Updating")
# Prior: Beta(2,2)
alpha_prior, beta_prior = 2, 2
# Data: 7 successes, 3 failures
successes, failures = 7, 3
# Posterior: Beta(α + successes, β + failures)
alpha_post = alpha_prior + successes
beta_post = beta_prior + failures

plt.figure(figsize=(10, 6))
pdf_prior = stats.beta.pdf(x, alpha_prior, beta_prior)
pdf_post = stats.beta.pdf(x, alpha_post, beta_post)

plt.plot(x, pdf_prior, 'b-', linewidth=2, label='Prior Beta(2,2)')
plt.plot(x, pdf_post, 'r-', linewidth=2, label=f'Posterior Beta({alpha_post},{beta_post})')
plt.title('Bayesian Updating with Beta Distribution')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'beta_bayesian_updating.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 4: Effect of Sample Size
print("\nExample 4: Effect of Sample Size")
sample_sizes = [10, 50, 200]
success_rate = 0.7  # True probability of success

plt.figure(figsize=(10, 6))
# Prior: Beta(1,1) (uniform)
alpha_prior, beta_prior = 1, 1
pdf_prior = stats.beta.pdf(x, alpha_prior, beta_prior)
plt.plot(x, pdf_prior, 'k--', linewidth=2, label='Prior Beta(1,1)')

for n in sample_sizes:
    successes = int(n * success_rate)
    failures = n - successes
    alpha_post = alpha_prior + successes
    beta_post = beta_prior + failures
    pdf_post = stats.beta.pdf(x, alpha_post, beta_post)
    plt.plot(x, pdf_post, linewidth=2, label=f'n={n}')

plt.axvline(x=success_rate, color='r', linestyle=':', label='True probability')
plt.title('Effect of Sample Size on Beta Distribution')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'beta_sample_size_effect.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 5: Credible Intervals
print("\nExample 5: Credible Intervals")
alpha, beta = 10, 5
intervals = [0.5, 0.8, 0.95]

plt.figure(figsize=(10, 6))
pdf = stats.beta.pdf(x, alpha, beta)
plt.plot(x, pdf, 'b-', linewidth=2)

for p in intervals:
    lower = stats.beta.ppf((1-p)/2, alpha, beta)
    upper = stats.beta.ppf((1+p)/2, alpha, beta)
    x_interval = np.linspace(lower, upper, 100)
    plt.fill_between(x_interval, stats.beta.pdf(x_interval, alpha, beta), 
                    alpha=0.3, label=f'{int(p*100)}% Credible Interval')
    print(f"{int(p*100)}% Credible Interval: [{lower:.4f}, {upper:.4f}]")

plt.title(f'Credible Intervals for Beta({alpha},{beta})')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'beta_credible_intervals.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 6: Effect of Sample Size on Beta Distribution
print("\nExample 6: Effect of Sample Size on Beta Distribution")
sample_sizes = [10, 50, 200]
success_rate = 0.7  # True probability of success

plt.figure(figsize=(10, 6))
# Prior: Beta(1,1) (uniform)
alpha_prior, beta_prior = 1, 1
pdf_prior = stats.beta.pdf(x, alpha_prior, beta_prior)
plt.plot(x, pdf_prior, 'k--', linewidth=2, label='Prior Beta(1,1)')

for n in sample_sizes:
    successes = int(n * success_rate)
    failures = n - successes
    alpha_post = alpha_prior + successes
    beta_post = beta_prior + failures
    pdf_post = stats.beta.pdf(x, alpha_post, beta_post)
    plt.plot(x, pdf_post, linewidth=2, label=f'n={n}')

plt.axvline(x=success_rate, color='r', linestyle=':', label='True probability')
plt.title('Effect of Sample Size on Beta Distribution')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'beta_sample_size_effect.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 7: Common Beta Distribution Shapes
print("\nExample 7: Common Beta Distribution Shapes")
shapes = [
    (0.5, 0.5, "Jeffrey's Prior"),
    (1, 1, "Uniform"),
    (2, 2, "Symmetric"),
    (5, 1, "Right-skewed"),
    (1, 5, "Left-skewed"),
    (2, 5, "Left-skewed moderate"),
    (5, 2, "Right-skewed moderate")
]

plt.figure(figsize=(12, 6))
for alpha, beta, label in shapes:
    pdf = stats.beta.pdf(x, alpha, beta)
    plt.plot(x, pdf, linewidth=2, label=f'Beta({alpha},{beta}) - {label}')

plt.title('Common Beta Distribution Shapes')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'beta_common_shapes.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 8: Parameter Space Visualization
print("\nExample 8: Parameter Space Visualization")
alphas = [0.5, 1, 2, 5]
betas = [0.5, 1, 2, 5]

plt.figure(figsize=(12, 12))
for i, alpha in enumerate(alphas):
    for j, beta in enumerate(betas):
        plt.subplot(4, 4, i*4 + j + 1)
        pdf = stats.beta.pdf(x, alpha, beta)
        plt.plot(x, pdf, 'b-', linewidth=2)
        plt.title(f'Beta({alpha},{beta})')
        plt.grid(True, alpha=0.3)
        if i < 3:
            plt.xticks([])
        if j > 0:
            plt.yticks([])

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'beta_parameter_space.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll beta distribution example images created successfully.") 