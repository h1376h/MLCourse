import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from scipy.stats import norm

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_3_9")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Understanding the Cramér-Rao Bound
print_step_header(1, "Understanding the Cramér-Rao Bound")

print("The Cramér-Rao inequality establishes the lower bound for the variance of any unbiased estimator.")
print("For an unbiased estimator θ̂ of parameter θ:")
print("   Var(θ̂) ≥ 1 / [n·I(θ)]")
print("where:")
print("   n is the sample size")
print("   I(θ) is the Fisher Information for a single observation")
print()
print("The Cramér-Rao Lower Bound (CRLB) is 1/[n·I(θ)]")
print()

# Step 2: Fisher Information Explanation
print_step_header(2, "Fisher Information")

print("Fisher Information I(θ) quantifies how much information a single observation")
print("carries about the parameter θ.")
print()
print("For a probability density function f(x|θ), Fisher Information is defined as:")
print("   I(θ) = E[(∂/∂θ log f(X|θ))²]")
print()
print("Or equivalently:")
print("   I(θ) = -E[∂²/∂θ² log f(X|θ)]")
print()
print("Higher Fisher Information means:")
print("   - The parameter has more influence on the distribution")
print("   - We can estimate θ more precisely (lower variance)")
print()

# Create a visual explanation of Fisher Information
plt.figure(figsize=(10, 6))

# Example distributions for different parameter values
theta_values = [1, 2, 3]
x = np.linspace(-5, 10, 1000)

# Use normal distribution with different means as an example
for theta in theta_values:
    y = norm.pdf(x, loc=theta, scale=1)
    plt.plot(x, y, label=f'θ = {theta}', linewidth=2)

plt.title('Effect of Parameter θ on Probability Distribution', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.legend()
plt.grid(True)

# Add annotation to explain Fisher Information
plt.annotate('Higher separation between curves\nindicates higher Fisher Information',
             xy=(3, 0.2), xytext=(5, 0.3),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
             fontsize=12, ha='center')

plt.tight_layout()
file_path = os.path.join(save_dir, "fisher_information_concept.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 3: Example - Fisher Information for Normal Distribution
print_step_header(3, "Example: Fisher Information for Normal Distribution")

print("For a normal distribution N(μ, σ²) with known variance σ²:")
print("   f(x|μ) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))")
print()
print("The score function (derivative of log-likelihood) is:")
print("   ∂/∂μ log f(x|μ) = (x-μ)/σ²")
print()
print("The Fisher Information for a single observation is:")
print("   I(μ) = E[((X-μ)/σ²)²] = 1/σ²")
print()
print("Therefore, the Cramér-Rao Lower Bound for estimating μ is:")
print("   CRLB = 1/(n·I(μ)) = σ²/n")
print()
print("For the sample mean X̄, we know that Var(X̄) = σ²/n")
print("This means the sample mean is an efficient estimator as it achieves the CRLB.")
print()

# Create a visual explanation of the normal distribution example
sigma = 1.0
n_values = [1, 5, 10, 20, 50]
theta = 0  # True parameter value

# This function computes the CRLB for normal distribution
def crlb_normal(sigma, n):
    return sigma**2 / n

plt.figure(figsize=(10, 6))

x_vals = np.linspace(-3, 3, 1000)
for n in n_values:
    # CRLB for estimator variance
    crlb = crlb_normal(sigma, n)
    
    # Plot the normal distribution of the estimator
    estimator_dist = norm.pdf(x_vals, loc=theta, scale=np.sqrt(crlb))
    plt.plot(x_vals, estimator_dist, label=f'n = {n}, CRLB = {crlb:.4f}', linewidth=2)

plt.title('Distribution of Sample Mean Estimator with Different Sample Sizes', fontsize=14)
plt.xlabel('Estimator Value', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.axvline(x=theta, color='black', linestyle='--', alpha=0.5, label='True θ = 0')
plt.legend()
plt.grid(True)
plt.tight_layout()

file_path = os.path.join(save_dir, "normal_crlb_example.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 4: Visualizing the CRLB for various distributions
print_step_header(4, "Visualizing CRLB for Various Distributions")

# Create a set of Fisher Information values for different distributions
distributions = {
    "Normal(μ, σ²) - estimating μ": "1/σ²",
    "Normal(μ, σ²) - estimating σ²": "2/σ⁴",
    "Bernoulli(p)": "1/[p(1-p)]",
    "Poisson(λ)": "1/λ",
    "Exponential(λ)": "1/λ²",
    "Uniform(0, θ)": "1/θ²"
}

print("Fisher Information for common distributions:")
for dist, formula in distributions.items():
    print(f"   {dist}: I(θ) = {formula}")
print()

# Visualize Fisher Information for Bernoulli distribution as an example
p_vals = np.linspace(0.01, 0.99, 100)
bernoulli_fisher = 1 / (p_vals * (1 - p_vals))

plt.figure(figsize=(10, 6))
plt.plot(p_vals, bernoulli_fisher, 'r-', linewidth=2)
plt.title('Fisher Information for Bernoulli Distribution', fontsize=14)
plt.xlabel('p (success probability)', fontsize=12)
plt.ylabel('Fisher Information I(p)', fontsize=12)
plt.grid(True)
plt.tight_layout()

file_path = os.path.join(save_dir, "bernoulli_fisher_information.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Plot CRLB for Bernoulli with different sample sizes
plt.figure(figsize=(10, 6))
n_values = [10, 50, 100, 500]

for n in n_values:
    crlb = 1 / (n * bernoulli_fisher)
    plt.plot(p_vals, crlb, label=f'n = {n}', linewidth=2)

plt.title('Cramér-Rao Lower Bound for Bernoulli Parameter p', fontsize=14)
plt.xlabel('p (success probability)', fontsize=12)
plt.ylabel('CRLB for Variance of p̂', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()

file_path = os.path.join(save_dir, "bernoulli_crlb.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 5: Efficiency of Estimators
print_step_header(5, "Efficiency of Estimators")

print("An estimator is called efficient if its variance equals the CRLB.")
print("That is, if Var(θ̂) = 1/[n·I(θ)], then θ̂ is efficient.")
print()
print("For some distributions and parameters, efficient estimators exist:")
print("   - Sample mean for normal mean (with known variance)")
print("   - Sample proportion for Bernoulli parameter")
print("   - Sample mean for Poisson parameter")
print()
print("For other distributions or parameters, efficient estimators may not exist.")
print()

# Visualization comparing an efficient estimator with an inefficient one
plt.figure(figsize=(12, 6))
gs = GridSpec(1, 2, width_ratios=[1, 1])

# Parameters
n = 20
sigma = 1
true_theta = 0
x = np.linspace(-1, 1, 1000)

# Efficient estimator (sample mean for normal mean)
efficient_var = sigma**2 / n
efficient_density = norm.pdf(x, loc=true_theta, scale=np.sqrt(efficient_var))

# Inefficient estimator (e.g., "Sample median" for normal mean)
inefficient_var = efficient_var * 1.5  # 50% more variance than CRLB
inefficient_density = norm.pdf(x, loc=true_theta, scale=np.sqrt(inefficient_var))

# Left subplot: Efficient estimator
ax1 = plt.subplot(gs[0])
ax1.plot(x, efficient_density, 'g-', linewidth=2, label='Efficient Estimator\n(Sample Mean)')
ax1.axvline(x=true_theta, color='red', linestyle='--', label='True θ')
ax1.set_title('Efficient Estimator (Achieves CRLB)', fontsize=12)
ax1.set_xlabel('θ̂', fontsize=10)
ax1.set_ylabel('Probability Density', fontsize=10)
ax1.legend()
ax1.grid(True)

# Right subplot: Inefficient estimator
ax2 = plt.subplot(gs[1])
ax2.plot(x, inefficient_density, 'b-', linewidth=2, label='Inefficient Estimator\n(e.g., Sample Median)')
ax2.fill_between(x, 0, inefficient_density, alpha=0.2, color='blue')
ax2.plot(x, efficient_density, 'g--', linewidth=2, alpha=0.5, label='CRLB Reference')
ax2.axvline(x=true_theta, color='red', linestyle='--', label='True θ')
ax2.set_title('Inefficient Estimator (Exceeds CRLB)', fontsize=12)
ax2.set_xlabel('θ̂', fontsize=10)
ax2.set_ylabel('Probability Density', fontsize=10)
ax2.legend()
ax2.grid(True)

plt.tight_layout()
file_path = os.path.join(save_dir, "efficiency_comparison.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 6: Cramér-Rao Bound and Sample Size
print_step_header(6, "Cramér-Rao Bound and Sample Size")

print("The CRLB decreases proportionally to 1/n as sample size n increases.")
print("This means that with more data, the lower bound on variance decreases.")
print()
print("For an unbiased estimator θ̂ with variance achieving CRLB:")
print("   Var(θ̂) = 1/[n·I(θ)] ∝ 1/n")
print()
print("The standard error (standard deviation of θ̂) decreases proportionally to 1/√n:")
print("   SE(θ̂) = √Var(θ̂) ∝ 1/√n")
print()
print("This is why we often need to quadruple the sample size to halve the standard error.")
print()

# Create visualization for decreasing CRLB with increasing sample size
n_range = np.arange(1, 200)
crlb_values = 1 / n_range  # Assuming I(θ) = 1 for simplicity

plt.figure(figsize=(10, 6))
plt.plot(n_range, crlb_values, 'b-', linewidth=2)
plt.title('Cramér-Rao Lower Bound vs. Sample Size', fontsize=14)
plt.xlabel('Sample Size (n)', fontsize=12)
plt.ylabel('CRLB for Var(θ̂)', fontsize=12)
plt.xlim(0, 200)
plt.ylim(0, 1)
plt.grid(True)

# Add annotations for specific sample sizes
sample_sizes = [10, 40, 90, 160]
for n in sample_sizes:
    crlb = 1/n
    plt.scatter(n, crlb, color='red', s=50, zorder=3)
    plt.annotate(f'n={n}, CRLB={crlb:.4f}', 
                 xy=(n, crlb), xytext=(n+10, crlb+0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1),
                 fontsize=9)

plt.tight_layout()
file_path = os.path.join(save_dir, "crlb_vs_sample_size.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 7: Conclusions and Formula
print_step_header(7, "Cramér-Rao Inequality - Conclusion")

print("The Cramér-Rao inequality is:")
print("   Var(θ̂) ≥ 1/[n·I(θ)]")
print()
print("Where:")
print("   θ̂ is an unbiased estimator of θ")
print("   n is the sample size")
print("   I(θ) is the Fisher Information for a single observation")
print()
print("Key insights:")
print("1. The CRLB sets a theoretical minimum variance for any unbiased estimator")
print("2. Estimators that achieve this bound are called efficient")
print("3. The bound decreases as sample size increases (∝ 1/n)")
print("4. Fisher Information quantifies how much information a distribution carries about θ")
print()

# Create a final summary visualization
plt.figure(figsize=(10, 6))

# Create a visual equation
plt.text(0.5, 0.6, r"$\mathbf{Var(\hat{\theta}) \geq \frac{1}{n \cdot I(\theta)}}$", 
         fontsize=24, ha='center')

plt.text(0.5, 0.4, "The Cramér-Rao Inequality", fontsize=18, ha='center')
plt.text(0.5, 0.3, "Establishes the lower bound for the variance of any unbiased estimator", 
         fontsize=14, ha='center')

# Remove axes for a cleaner look
plt.axis('off')
plt.tight_layout()

file_path = os.path.join(save_dir, "cramer_rao_equation.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

print("\nQuestion 9 Solution Summary:")
print("The Cramér-Rao inequality establishes that the variance of any unbiased estimator θ̂")
print("is bounded below by the inverse of the product of sample size n and Fisher Information I(θ):")
print("Var(θ̂) ≥ 1/[n·I(θ)]") 