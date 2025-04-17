import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_3_6")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Set up the problem
print_step_header(1, "Understanding the Question")

print("QUESTION 6: For a Bernoulli distribution with parameter p, write down the Fisher Information I(p).")
print("\nLet's understand what this question is asking:")
print("- We have a Bernoulli distribution with parameter p")
print("- The PDF of a Bernoulli distribution is: f(x|p) = p^x * (1-p)^(1-x) for x ∈ {0, 1}")
print("- We want to find the Fisher Information I(p)")

# Step 2: Define the Fisher Information
print_step_header(2, "Understanding Fisher Information")

print("The Fisher Information quantifies the amount of information that an observable random variable X carries about an unknown parameter θ of the distribution.")
print("\nFor a single observation, the Fisher Information is defined as:")
print("I(θ) = E[(∂/∂θ log f(X|θ))²]")
print("     = E[(score function)²]")
print("\nAlternatively, it can be expressed as:")
print("I(θ) = -E[∂²/∂θ² log f(X|θ)]")
print("     = -E[second derivative of log-likelihood]")
print("\nThe Fisher Information has several important properties:")
print("1. It is always non-negative")
print("2. It is additive for independent observations")
print("3. It is related to the Cramér-Rao lower bound: Var(θ̂) ≥ 1/I(θ)")

# Step 3: Derive the Fisher Information for Bernoulli
print_step_header(3, "Deriving the Fisher Information for Bernoulli Distribution")

print("For a Bernoulli distribution with parameter p, the PMF is:")
print("f(x|p) = p^x * (1-p)^(1-x) for x ∈ {0, 1}")
print("\nStep 3.1: Find the log-likelihood")
print("log f(x|p) = x log(p) + (1-x) log(1-p)")
print("\nStep 3.2: Find the score function (first derivative of log-likelihood)")
print("∂/∂p log f(x|p) = x/p - (1-x)/(1-p)")
print("\nStep 3.3: Square the score function")
print("(∂/∂p log f(x|p))² = (x/p - (1-x)/(1-p))²")
print("\nStep 3.4: Find the expected value of the squared score function")
print("E[(∂/∂p log f(x|p))²] = E[(x/p - (1-x)/(1-p))²]")
print("                      = p * (1/p - 0/(1-p))² + (1-p) * (0/p - 1/(1-p))²")
print("                      = p * (1/p)² + (1-p) * (1/(1-p))²")
print("                      = 1/p + 1/(1-p)")
print("                      = (1-p + p)/(p(1-p))")
print("                      = 1/(p(1-p))")
print("\nAlternatively, we can compute the second derivative:")
print("∂²/∂p² log f(x|p) = -x/p² - (1-x)/(1-p)²")
print("\nThe expected value is:")
print("E[∂²/∂p² log f(x|p)] = -p/p² - (1-p)/(1-p)² = -1/p - 1/(1-p) = -(1-p + p)/(p(1-p)) = -1/(p(1-p))")
print("\nTherefore, the Fisher Information is:")
print("I(p) = -E[∂²/∂p² log f(x|p)] = 1/(p(1-p))")

# Step 4: Visual Demonstration
print_step_header(4, "Visual Demonstration")

# Simulate Bernoulli samples with different parameters and visualize the information
p_values = np.linspace(0.01, 0.99, 1000)
fisher_info_values = 1 / (p_values * (1 - p_values))

# Fig 1: Basic Fisher Information Plot
plt.figure(figsize=(10, 6))
plt.plot(p_values, fisher_info_values, 'b-', linewidth=2)
plt.title('Fisher Information for Bernoulli Distribution: I(p) = 1/(p(1-p))', fontsize=14)
plt.xlabel('Parameter p', fontsize=12)
plt.ylabel('Fisher Information I(p)', fontsize=12)
plt.grid(True)

# Add annotation for the minimum
min_idx = np.argmin(fisher_info_values)
min_p = p_values[min_idx]
min_info = fisher_info_values[min_idx]
plt.scatter([min_p], [min_info], color='red', s=100, zorder=5)
plt.annotate(f'Minimum at p = 0.5, I(0.5) = 4', 
             xy=(min_p, min_info), xytext=(0.3, 6),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
             fontsize=12)

# Add annotations for interpretation
plt.text(0.7, 35, 'High Information\n(p close to 0 or 1)', 
         fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.8))
plt.text(0.5, 3, 'Low Information\n(p ≈ 0.5)', 
         fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.8))

# Limit the y-axis to make the plot more readable
plt.ylim(0, 50)
plt.tight_layout()
file_path = os.path.join(save_dir, "fisher_information_curve.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure 1 saved to: {file_path}")

# Fig 2: Cramér-Rao Lower Bound Plot
plt.figure(figsize=(10, 6))

# Define the Cramér-Rao lower bound for the variance
def cramer_rao_bound(p, n):
    return p * (1 - p) / n

# Plot for different sample sizes
sample_sizes = [10, 50, 100, 200]
for n in sample_sizes:
    cr_bounds = cramer_rao_bound(p_values, n)
    plt.plot(p_values, cr_bounds, label=f'n = {n}')

plt.title('Cramér-Rao Lower Bound for Var(p̂) by Sample Size', fontsize=14)
plt.xlabel('Parameter p', fontsize=12)
plt.ylabel('Lower Bound for Var(p̂)', fontsize=12)
plt.grid(True)
plt.legend()
plt.ylim(0, 0.025)
plt.tight_layout()
file_path = os.path.join(save_dir, "cramer_rao_bound.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure 2 saved to: {file_path}")

# Fig 3: MLE Variance vs Cramér-Rao Bound
plt.figure(figsize=(10, 6))

# Generate Bernoulli samples and compute variance of sample mean
np.random.seed(42)
n_simulations = 10000
n_samples = 50
p_test_values = [0.1, 0.3, 0.5, 0.7, 0.9]
variances = []
cr_bounds_for_test = []

for p_test in p_test_values:
    # Generate samples
    samples = np.random.binomial(1, p_test, (n_simulations, n_samples))
    
    # Calculate sample means (which is the MLE for Bernoulli)
    sample_means = np.mean(samples, axis=1)
    
    # Calculate variance of the MLEs
    var_mle = np.var(sample_means)
    variances.append(var_mle)
    
    # Calculate the Cramér-Rao bound
    cr_bound = p_test * (1 - p_test) / n_samples
    cr_bounds_for_test.append(cr_bound)

# Plot the empirical variances and theoretical bounds
plt.bar(np.arange(len(p_test_values)) - 0.2, variances, width=0.4, label='Empirical Var(p̂)', alpha=0.7)
plt.bar(np.arange(len(p_test_values)) + 0.2, cr_bounds_for_test, width=0.4, label='C-R Bound', alpha=0.7)
plt.xticks(np.arange(len(p_test_values)), [f'p = {p}' for p in p_test_values])
plt.title(f'Variance of MLE vs. Cramér-Rao Bound (n = {n_samples})', fontsize=14)
plt.ylabel('Variance', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
file_path = os.path.join(save_dir, "mle_variance_vs_bound.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure 3 saved to: {file_path}")

# Fig 4: New Visualization - Estimation Accuracy with Sample Size
plt.figure(figsize=(10, 6))

# Function to simulate estimation accuracy for different sample sizes
def simulate_estimation_accuracy(true_p, sample_sizes, n_simulations=5000):
    std_errors = []
    for n in sample_sizes:
        # Generate Bernoulli samples for each sample size
        samples = np.random.binomial(1, true_p, (n_simulations, n))
        
        # Calculate sample means (MLEs)
        p_hats = np.mean(samples, axis=1)
        
        # Calculate standard errors
        std_error = np.std(p_hats)
        std_errors.append(std_error)
    return std_errors

# Select different p values
test_ps = [0.1, 0.3, 0.5, 0.7, 0.9]
sample_sizes_to_test = np.array([10, 20, 50, 100, 200, 500])

# Plot for each p value
for p in test_ps:
    std_errors = simulate_estimation_accuracy(p, sample_sizes_to_test)
    plt.plot(sample_sizes_to_test, std_errors, marker='o', label=f'p = {p}')
    
    # Plot the theoretical line (square root of Cramér-Rao bound)
    theoretical = np.sqrt(p * (1 - p) / sample_sizes_to_test)
    plt.plot(sample_sizes_to_test, theoretical, 'k--', alpha=0.3)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Sample Size (n)', fontsize=12)
plt.ylabel('Standard Error of p̂', fontsize=12)
plt.title('Estimation Accuracy vs. Sample Size for Different p Values', fontsize=14)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()
plt.tight_layout()
file_path = os.path.join(save_dir, "estimation_accuracy.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure 4 saved to: {file_path}")

# Step 5: Implications and Applications
print_step_header(5, "Implications and Applications")

print("The Fisher Information for a Bernoulli distribution has several important implications:")
print("\n1. PARAMETER ESTIMATION EFFICIENCY:")
print("   - The Cramér-Rao lower bound shows that the variance of any unbiased estimator of p must be at least p(1-p)/n")
print("   - The MLE (sample mean) achieves this bound, making it an efficient estimator")
print("\n2. SAMPLE SIZE REQUIREMENTS:")
print("   - When p is close to 0.5, the Fisher Information is at its minimum (I(0.5) = 4)")
print("   - This means more samples are needed to achieve the same precision in estimating p when p ≈ 0.5")
print("   - When p is close to 0 or 1, fewer samples are needed for the same precision")
print("\n3. EXPERIMENT DESIGN:")
print("   - If we're designing an experiment to estimate p, and have prior knowledge that p might be close to 0.5, we should plan for larger sample sizes")
print("   - Conversely, if p is expected to be close to 0 or 1, smaller samples may suffice")

# Step 6: Conclusion
print_step_header(6, "Conclusion")

print("ANSWER: For a Bernoulli distribution with parameter p, the Fisher Information is I(p) = 1/(p(1-p)).")
print("\nThe Fisher Information measures how much information a sample from a Bernoulli distribution carries about the parameter p.")
print("When p is close to 0.5, each sample provides less information, requiring more samples for accurate estimation.")
print("When p is close to 0 or 1, each sample provides more information about p.") 