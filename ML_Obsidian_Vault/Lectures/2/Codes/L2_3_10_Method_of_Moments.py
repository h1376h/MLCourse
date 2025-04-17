import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as stats
from matplotlib.gridspec import GridSpec
from scipy.optimize import fsolve

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_3_10")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Understanding Method of Moments
print_step_header(1, "Understanding Method of Moments")

print("The Method of Moments (MoM) is one of the oldest techniques for parameter estimation.")
print("It works by equating sample moments to theoretical moments.")
print()
print("For a random variable X with parameter θ:")
print("1. Find the theoretical expectation E[X] in terms of θ")
print("2. Equate it to the sample mean: E[X] = 1/n * Σ(Xᵢ)")
print("3. Solve for θ")
print()
print("More generally, this can be extended to match higher moments if needed.")
print()

# Create a diagram to illustrate Method of Moments
plt.figure(figsize=(10, 5))
gs = GridSpec(1, 2, width_ratios=[1, 1])

# Left side: Population distribution
ax1 = plt.subplot(gs[0])
x = np.linspace(0, 5, 1000)
# Example distribution: a Beta distribution skewed right
y = stats.beta.pdf(x/5, 2, 5) 
ax1.plot(x, y, 'r-', linewidth=2)
ax1.set_title('Population Distribution', fontsize=12)
ax1.set_xlabel('x', fontsize=10)
ax1.set_ylabel('Probability Density', fontsize=10)
ax1.fill_between(x, 0, y, alpha=0.2, color='red')
ax1.text(2.5, 0.6, "E[X] = f(θ)", fontsize=12, ha='center')
ax1.grid(True)

# Right side: Sample from distribution
ax2 = plt.subplot(gs[1])
np.random.seed(42)
sample = np.random.beta(2, 5, 100) * 5
ax2.hist(sample, bins=15, density=True, alpha=0.7, color='blue')
ax2.set_title('Sample from Distribution', fontsize=12)
ax2.set_xlabel('x', fontsize=10)
ax2.set_ylabel('Frequency', fontsize=10)
ax2.axvline(x=np.mean(sample), color='black', linestyle='--', 
            label=f'Sample Mean = {np.mean(sample):.2f}')
ax2.text(2.5, 0.6, "x̄ = 1/n * Σ(xᵢ)", fontsize=12, ha='center')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
file_path = os.path.join(save_dir, "mom_concept.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 2: Specific Problem Statement
print_step_header(2, "Problem Statement")

print("Consider a random variable X with a distribution where:")
print("E[X] = θ/(1+θ)")
print()
print("Task: Derive the Method of Moments estimator for θ based on a random sample X₁, X₂, ..., Xₙ.")
print()

# Visualize the theoretical mean function: E[X] = θ/(1+θ)
theta_values = np.linspace(0.1, 10, 100)
expected_x = theta_values / (1 + theta_values)

plt.figure(figsize=(10, 6))
plt.plot(theta_values, expected_x, 'b-', linewidth=2, label='E[X] = θ/(1+θ)')
plt.title('Relationship between Parameter θ and Expected Value E[X]', fontsize=14)
plt.xlabel('θ (parameter)', fontsize=12)
plt.ylabel('E[X]', fontsize=12)
plt.grid(True)
plt.axhline(y=0.5, color='red', linestyle='--', label='Upper Limit of E[X] as θ→∞')
plt.annotate('As θ increases, E[X] approaches 1', xy=(9, 0.9), xytext=(7, 0.7),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
             fontsize=10)
plt.legend()
plt.tight_layout()

file_path = os.path.join(save_dir, "expected_value_function.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 3: Derivation of the Method of Moments Estimator
print_step_header(3, "Derivation of Method of Moments Estimator")

print("Step 1: We know the theoretical expectation: E[X] = θ/(1+θ)")
print("Step 2: We equate it to the sample mean:")
print("        θ/(1+θ) = x̄ = 1/n * Σ(Xᵢ)")
print("Step 3: Solve for θ:")
print("        θ/(1+θ) = x̄")
print("        θ = x̄(1+θ)")
print("        θ = x̄ + x̄θ")
print("        θ - x̄θ = x̄")
print("        θ(1-x̄) = x̄")
print("        θ = x̄/(1-x̄)")
print()
print("Therefore, the Method of Moments estimator for θ is:")
print("        θ̂ = x̄/(1-x̄)")
print()
print("Note: This formula only works when 0 < x̄ < 1, which is consistent with the")
print("constraint that E[X] = θ/(1+θ) is always between 0 and 1.")
print()

# Visualize the derivation
plt.figure(figsize=(10, 6))

# Plot the original curve E[X] = θ/(1+θ)
plt.plot(theta_values, expected_x, 'b-', linewidth=2, label='E[X] = θ/(1+θ)')

# Highlight the inversion (θ in terms of E[X])
ex_values = np.linspace(0.01, 0.99, 100)
theta_from_ex = ex_values / (1 - ex_values)
plt.plot(theta_from_ex, ex_values, 'r--', linewidth=2, label='θ = x̄/(1-x̄)')

plt.title('Derivation of Method of Moments Estimator', fontsize=14)
plt.xlabel('θ (parameter)', fontsize=12)
plt.ylabel('E[X] or Sample Mean (x̄)', fontsize=12)
plt.grid(True)

# Add arrows to indicate the inversion process
plt.annotate('', xy=(2, 2/3), xytext=(2, 0.1),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
plt.annotate('Given θ, find E[X]', xy=(2, 0.3), fontsize=10)

plt.annotate('', xy=(2, 2/3), xytext=(5, 2/3),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
plt.annotate('Given x̄, find θ̂', xy=(3, 0.7), fontsize=10)

plt.legend()
plt.tight_layout()

file_path = os.path.join(save_dir, "mom_derivation.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 4: Example with Simulated Data
print_step_header(4, "Example with Simulated Data")

# We'll need to simulate data from a distribution with E[X] = θ/(1+θ)
# A Beta distribution with parameters a = θ and b = 1 has mean θ/(θ+1)
true_theta = 2.0
print(f"True parameter value: θ = {true_theta}")
print(f"Theoretical mean: E[X] = {true_theta/(1+true_theta):.4f}")

np.random.seed(123)
n_samples = [10, 50, 500, 5000]
results = {}

plt.figure(figsize=(12, 8))
for i, n in enumerate(n_samples):
    # Generate samples from Beta(true_theta, 1)
    samples = np.random.beta(true_theta, 1, n)
    
    # Calculate sample mean
    sample_mean = np.mean(samples)
    
    # Calculate MoM estimator
    theta_mom = sample_mean / (1 - sample_mean)
    
    results[n] = {
        'sample_mean': sample_mean,
        'theta_mom': theta_mom
    }
    
    print(f"\nSample size n = {n}:")
    print(f"Sample mean: {sample_mean:.4f}")
    print(f"MoM estimate: θ̂ = {theta_mom:.4f}")
    print(f"Absolute error: |θ̂ - θ| = {abs(theta_mom - true_theta):.4f}")
    
    # Plot histogram and theoretical pdf
    plt.subplot(2, 2, i+1)
    plt.hist(samples, bins=20, density=True, alpha=0.6, color='blue', 
             label=f'Sample (n={n})')
    
    # Plot the true Beta distribution
    x = np.linspace(0, 1, 1000)
    y = stats.beta.pdf(x, true_theta, 1)
    plt.plot(x, y, 'r-', linewidth=2, label='True Distribution')
    
    # Add vertical lines for sample mean and theoretical mean
    plt.axvline(x=sample_mean, color='blue', linestyle='--', 
                label=f'Sample Mean = {sample_mean:.4f}')
    plt.axvline(x=true_theta/(1+true_theta), color='red', linestyle='--', 
                label=f'Theoretical Mean = {true_theta/(1+true_theta):.4f}')
    
    plt.title(f'n = {n}, θ̂ = {theta_mom:.4f}', fontsize=12)
    plt.xlabel('x', fontsize=10)
    plt.ylabel('Density', fontsize=10)
    plt.legend(fontsize=8)
    plt.grid(True)

plt.tight_layout()
file_path = os.path.join(save_dir, "mom_examples.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Plot the convergence of the estimator as sample size increases
sample_sizes = np.logspace(1, 4, 20).astype(int)
estimates = []
errors = []

for n in sample_sizes:
    samples = np.random.beta(true_theta, 1, n)
    sample_mean = np.mean(samples)
    theta_mom = sample_mean / (1 - sample_mean)
    estimates.append(theta_mom)
    errors.append(abs(theta_mom - true_theta))

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.semilogx(sample_sizes, estimates, 'bo-')
plt.axhline(y=true_theta, color='r', linestyle='--', label=f'True θ = {true_theta}')
plt.title('Convergence of MoM Estimator', fontsize=12)
plt.xlabel('Sample Size (log scale)', fontsize=10)
plt.ylabel('Estimated θ', fontsize=10)
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.loglog(sample_sizes, errors, 'go-')
plt.title('Estimation Error', fontsize=12)
plt.xlabel('Sample Size (log scale)', fontsize=10)
plt.ylabel('|θ̂ - θ| (log scale)', fontsize=10)
plt.grid(True)

plt.tight_layout()
file_path = os.path.join(save_dir, "mom_convergence.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 5: Distribution Analysis - What distribution could this be?
print_step_header(5, "Distribution Analysis")

print("Now let's analyze what distribution might have E[X] = θ/(1+θ).")
print()
print("One possibility is the Beta distribution with parameters (θ, 1):")
print("For Beta(a, b): E[X] = a/(a+b)")
print("If a = θ and b = 1, then E[X] = θ/(θ+1), which matches our formula.")
print()
print("Another possibility is a mixture of a point mass at 1 and a point mass at 0:")
print("Let X = 1 with probability p and X = 0 with probability (1-p)")
print("Then E[X] = p·1 + (1-p)·0 = p")
print("If p = θ/(1+θ), this also gives us the required expectation.")
print()

# Visualize these two possibilities
plt.figure(figsize=(12, 5))
gs = GridSpec(1, 2, width_ratios=[1, 1])

# Left panel: Beta distribution for different θ values
ax1 = plt.subplot(gs[0])
theta_values = [0.5, 1, 2, 5]
x = np.linspace(0, 1, 1000)

for theta in theta_values:
    pdf = stats.beta.pdf(x, theta, 1)
    ax1.plot(x, pdf, linewidth=2, label=f'θ = {theta}')
    
ax1.set_title('Beta(θ, 1) Distribution', fontsize=12)
ax1.set_xlabel('x', fontsize=10)
ax1.set_ylabel('Probability Density', fontsize=10)
ax1.legend()
ax1.grid(True)

# Right panel: Point mass mixture for different θ values
ax2 = plt.subplot(gs[1])
p_values = [theta/(1+theta) for theta in theta_values]

for i, (theta, p) in enumerate(zip(theta_values, p_values)):
    ax2.bar([0, 1], [1-p, p], width=0.1, alpha=0.6, label=f'θ = {theta}, p = {p:.2f}')
    
ax2.set_title('Mixture Distribution: p·δ(x-1) + (1-p)·δ(x-0)', fontsize=12)
ax2.set_xlabel('x', fontsize=10)
ax2.set_ylabel('Probability Mass', fontsize=10)
ax2.set_xticks([0, 1])
ax2.set_xlim(-0.5, 1.5)
ax2.legend()
ax2.grid(True)

plt.tight_layout()
file_path = os.path.join(save_dir, "possible_distributions.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 6: Properties of the MoM Estimator
print_step_header(6, "Properties of the MoM Estimator")

print("Let's analyze some properties of our Method of Moments estimator θ̂ = x̄/(1-x̄):")
print()
print("1. Consistency:")
print("   As sample size n → ∞, the sample mean x̄ → E[X] = θ/(1+θ) by law of large numbers")
print("   Therefore, θ̂ = x̄/(1-x̄) → θ/(1+θ) / (1-θ/(1+θ)) = θ/(1+θ) / (1/(1+θ)) = θ")
print("   This shows that θ̂ is a consistent estimator.")
print()
print("2. Constraint:")
print("   Our estimator θ̂ = x̄/(1-x̄) requires 0 < x̄ < 1")
print("   This aligns with the fact that for the distribution, 0 < E[X] < 1")
print()
print("3. Bias:")
print("   The estimator may be biased for small sample sizes, but the bias diminishes")
print("   as the sample size increases due to consistency.")
print()

# Visualize the transformation from sample mean to estimator
x_bar_values = np.linspace(0.01, 0.99, 100)
theta_est_values = x_bar_values / (1 - x_bar_values)

plt.figure(figsize=(10, 6))
plt.plot(x_bar_values, theta_est_values, 'b-', linewidth=2)
plt.title('Transformation from Sample Mean to MoM Estimator: θ̂ = x̄/(1-x̄)', fontsize=14)
plt.xlabel('Sample Mean (x̄)', fontsize=12)
plt.ylabel('MoM Estimator (θ̂)', fontsize=12)
plt.grid(True)

# Add annotations
plt.annotate('As x̄ → 1, θ̂ → ∞', xy=(0.9, 9), xytext=(0.7, 6),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
             fontsize=10)

plt.annotate('Estimator is undefined at x̄ = 1', xy=(0.99, 50), xytext=(0.8, 20),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
             fontsize=10)

# Highlight specific examples
example_means = [0.2, 0.5, 0.8]
for mean in example_means:
    est = mean / (1 - mean)
    plt.scatter(mean, est, color='red', s=50, zorder=3)
    plt.text(mean+0.02, est, f'x̄ = {mean}, θ̂ = {est:.2f}', fontsize=9)

plt.ylim(0, 20)
plt.tight_layout()
file_path = os.path.join(save_dir, "mom_transformation.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 7: Comparing MoM with Maximum Likelihood Estimation
print_step_header(7, "Comparing MoM with Maximum Likelihood Estimation")

print("For many distributions, both Method of Moments (MoM) and Maximum Likelihood")
print("Estimation (MLE) can be used to estimate parameters.")
print()
print("MoM advantages:")
print("- Often simpler to compute")
print("- Works when the likelihood function is complicated")
print("- Only requires knowledge of moments, not full distribution")
print()
print("MLE advantages:")
print("- Often more efficient (lower variance)")
print("- Asymptotically achieves the Cramér-Rao lower bound")
print("- Invariant to parameter transformations")
print()
print("For our problem with E[X] = θ/(1+θ):")
print("- If X ~ Beta(θ, 1), MoM and MLE give the same estimator: θ̂ = x̄/(1-x̄)")
print("- If X has a different distribution, the estimators might differ")
print()

# Visualize the comparison for the Beta(θ, 1) distribution
plt.figure(figsize=(10, 6))

# Function to compute log-likelihood for Beta(θ, 1)
def log_likelihood(theta, samples):
    if theta <= 0:
        return -np.inf
    return np.sum(np.log(stats.beta.pdf(samples, theta, 1)))

# Function to compute MoM estimator
def mom_estimator(samples):
    mean = np.mean(samples)
    if mean >= 1:
        return np.nan
    return mean / (1 - mean)

# Generate data and compute likelihood surface
np.random.seed(456)
sample_size = 50
true_theta = 2.0
samples = np.random.beta(true_theta, 1, sample_size)
sample_mean = np.mean(samples)
mom_est = mom_estimator(samples)

theta_range = np.linspace(0.1, 5, 100)
ll_values = [log_likelihood(t, samples) for t in theta_range]

# Normalize log-likelihood to make it easier to visualize
ll_values = np.array(ll_values)
ll_values = ll_values - np.max(ll_values)

# Plot the log-likelihood function
plt.plot(theta_range, ll_values, 'b-', linewidth=2, label='Log-Likelihood Function')
plt.axvline(x=mom_est, color='r', linestyle='--', 
            label=f'MoM Estimator: {mom_est:.4f}')

# Find MLE numerically
mle_est = theta_range[np.argmax(ll_values)]
plt.axvline(x=mle_est, color='g', linestyle=':', 
            label=f'MLE: {mle_est:.4f}')

plt.axvline(x=true_theta, color='k', linestyle='-', alpha=0.5,
            label=f'True θ: {true_theta}')

plt.title('Comparison of MoM and MLE for Beta(θ, 1) Distribution', fontsize=14)
plt.xlabel('θ', fontsize=12)
plt.ylabel('Log-Likelihood (normalized)', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
file_path = os.path.join(save_dir, "mom_vs_mle.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 8: Conclusion - Answer to the Question
print_step_header(8, "Conclusion - Answer to the Question")

print("The question asked: For a random variable X with a distribution where")
print("E[X] = θ/(1+θ), derive the Method of Moments estimator for θ.")
print()
print("Solution:")
print("1. Equate the theoretical mean to the sample mean:")
print("   θ/(1+θ) = x̄")
print("2. Solve for θ:")
print("   θ = x̄(1+θ)")
print("   θ = x̄ + x̄θ")
print("   θ - x̄θ = x̄")
print("   θ(1-x̄) = x̄")
print("   θ = x̄/(1-x̄)")
print()
print("The Method of Moments estimator is: θ̂ = x̄/(1-x̄)")
print("where x̄ is the sample mean of X₁, X₂, ..., Xₙ.")
print()
print("This estimator is valid when 0 < x̄ < 1, which aligns with the")
print("constraint that E[X] = θ/(1+θ) is always between 0 and 1 for θ > 0.")
print()

# Create a final summary visualization
plt.figure(figsize=(10, 6))

# Create a visual equation
plt.text(0.5, 0.7, r"$E[X] = \frac{\theta}{1+\theta}$", fontsize=24, ha='center')
plt.text(0.5, 0.5, r"$\Rightarrow \hat{\theta} = \frac{\bar{x}}{1-\bar{x}}$", fontsize=24, ha='center')

plt.text(0.5, 0.3, "Method of Moments Estimator", fontsize=18, ha='center')
plt.text(0.5, 0.2, "where x̄ is the sample mean of X₁, X₂, ..., Xₙ", fontsize=14, ha='center')

# Remove axes for a cleaner look
plt.axis('off')
plt.tight_layout()

file_path = os.path.join(save_dir, "mom_final_equation.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

print("\nQuestion 10 Solution Summary:")
print("For a random variable X with expectation E[X] = θ/(1+θ), the Method of Moments estimator is:")
print("θ̂ = x̄/(1-x̄)")
print("where x̄ is the sample mean. This is valid when 0 < x̄ < 1.") 