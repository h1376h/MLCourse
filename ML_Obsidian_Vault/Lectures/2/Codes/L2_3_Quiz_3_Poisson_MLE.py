import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from scipy.stats import poisson
import math

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_3_3")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Set up the problem
print_step_header(1, "Problem Setup")

print("Given:")
print("- We have a random sample X₁, X₂, ..., Xₙ from a Poisson distribution with parameter λ")
print("- The PMF of a Poisson distribution is: P(X = k) = (e^(-λ) * λ^k) / k!")
print("- We need to derive the MLE for λ, check if it's unbiased, calculate its variance, and find the CRLB")

# Visualize the Poisson distribution for different values of lambda
lambdas = [0.5, 1, 3, 5, 10]
x = np.arange(0, 20)  # Range of values for x

plt.figure(figsize=(12, 6))
for lambda_val in lambdas:
    pmf = poisson.pmf(x, lambda_val)
    plt.plot(x, pmf, 'o-', linewidth=2, label=f'λ = {lambda_val}')
    
plt.title('Poisson Distribution PMF for Different Values of λ', fontsize=14)
plt.xlabel('k (Number of events)', fontsize=12)
plt.ylabel('Probability P(X = k)', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "poisson_pmf.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 2: Derive the Likelihood Function and MLE
print_step_header(2, "Deriving the Maximum Likelihood Estimator")

print("The PMF of a Poisson random variable X is:")
print("P(X = k) = (e^(-λ) * λ^k) / k!")
print("\nThe likelihood function for a sample X₁, X₂, ..., Xₙ is:")
print("L(λ) = ∏ᵢ₌₁ⁿ (e^(-λ) * λ^Xᵢ) / Xᵢ!")
print("     = e^(-nλ) * λ^(∑Xᵢ) * ∏ᵢ₌₁ⁿ (1/Xᵢ!)")
print("\nThe log-likelihood function is:")
print("ℓ(λ) = -nλ + (∑Xᵢ)log(λ) - ∑log(Xᵢ!)")
print("\nTo find the MLE, we take the derivative of ℓ(λ) with respect to λ and set it to zero:")
print("dℓ(λ)/dλ = -n + (∑Xᵢ)/λ = 0")
print("\nSolving for λ:")
print("(∑Xᵢ)/λ = n")
print("λ = (∑Xᵢ)/n = X̄")
print("\nTherefore, the MLE for λ is the sample mean X̄ = (∑Xᵢ)/n")

# Simulate data to verify the MLE
np.random.seed(42)
lambda_true = 3.5  # True parameter value
n_samples = 100    # Sample size
samples = np.random.poisson(lambda_true, n_samples)

# Calculate MLE
lambda_mle = np.mean(samples)

# Create a range of lambda values to plot the likelihood function
lambda_values = np.linspace(2.5, 4.5, 1000)

# Define the log-likelihood function
def log_likelihood(lambda_val, data):
    n = len(data)
    sum_x = np.sum(data)
    return -n * lambda_val + sum_x * np.log(lambda_val) - np.sum([np.log(math.factorial(x)) for x in data])

# Calculate log-likelihood values
log_likelihood_values = [log_likelihood(lam, samples) for lam in lambda_values]

# Plot the log-likelihood function
plt.figure(figsize=(10, 6))
plt.plot(lambda_values, log_likelihood_values, 'g-', linewidth=2)
plt.axvline(x=lambda_mle, color='r', linestyle='--', 
            label=f'MLE λ̂ = {lambda_mle:.4f}')
plt.axvline(x=lambda_true, color='b', linestyle='--', 
            label=f'True λ = {lambda_true:.4f}')
plt.title('Log-Likelihood Function for Poisson Parameter λ', fontsize=14)
plt.xlabel('λ', fontsize=12)
plt.ylabel('ℓ(λ)', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "log_likelihood.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Visually confirm MLE with a histogram
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=range(0, max(samples) + 2), alpha=0.7, density=True, label='Sample Data')

# Overlay the true Poisson PMF
x_range = np.arange(0, max(samples) + 1)
plt.plot(x_range, poisson.pmf(x_range, lambda_true), 'bo-', label=f'True PDF (λ = {lambda_true})')
plt.plot(x_range, poisson.pmf(x_range, lambda_mle), 'ro-', label=f'MLE PDF (λ = {lambda_mle:.4f})')

plt.axvline(x=lambda_mle, color='r', linestyle='--', alpha=0.5, 
            label=f'Sample Mean = {lambda_mle:.4f}')
plt.axvline(x=lambda_true, color='b', linestyle='--', alpha=0.5, 
            label=f'True Mean = {lambda_true}')

plt.title('Histogram of Poisson Sample with True and MLE PDFs', fontsize=14)
plt.xlabel('k (Number of events)', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "histogram_with_pmfs.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 3: Check if the MLE is Unbiased
print_step_header(3, "Checking if the MLE is Unbiased")

print("An estimator is unbiased if its expected value equals the true parameter value.")
print("\nFor the Poisson MLE λ̂ = X̄ = (∑Xᵢ)/n:")
print("E[λ̂] = E[X̄] = E[(∑Xᵢ)/n]")
print("     = (1/n) * E[∑Xᵢ]")
print("     = (1/n) * ∑E[Xᵢ]")
print("     = (1/n) * n * λ")
print("     = λ")
print("\nSince E[λ̂] = λ, the MLE is an unbiased estimator for the Poisson parameter λ.")

# Verify with simulation
num_simulations = 1000
sample_sizes = [10, 30, 100, 300, 1000]
results = []

for n in sample_sizes:
    mle_estimates = []
    for _ in range(num_simulations):
        sample = np.random.poisson(lambda_true, n)
        mle_estimates.append(np.mean(sample))
    
    bias = np.mean(mle_estimates) - lambda_true
    results.append({
        'n': n, 
        'mean_estimate': np.mean(mle_estimates), 
        'bias': bias,
        'variance': np.var(mle_estimates, ddof=0),  # Population variance
        'std_dev': np.std(mle_estimates, ddof=0)
    })

# Plot the empirical distribution of MLE for different sample sizes
plt.figure(figsize=(12, 8))
for i, n in enumerate(sample_sizes):
    plt.subplot(2, 3, i+1)
    sample = np.random.poisson(lambda_true, n * num_simulations).reshape(num_simulations, n)
    mle_estimates = np.mean(sample, axis=1)
    
    plt.hist(mle_estimates, bins=30, alpha=0.7, density=True)
    plt.axvline(x=lambda_true, color='r', linestyle='--', 
                label=f'True λ = {lambda_true}')
    plt.axvline(x=np.mean(mle_estimates), color='g', linestyle='--', 
                label=f'Mean λ̂ = {np.mean(mle_estimates):.4f}')
    
    plt.title(f'Distribution of MLE for n = {n}', fontsize=10)
    plt.xlabel('λ̂', fontsize=8)
    plt.ylabel('Density', fontsize=8)
    plt.grid(True)
    plt.legend(fontsize=8)

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "mle_distributions.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Print the bias results
print("\nEmpirical results from simulations:")
print(f"{'Sample Size':^12} | {'Mean Estimate':^15} | {'Bias':^10} | {'Variance':^10}")
print("-" * 55)
for result in results:
    print(f"{result['n']:^12} | {result['mean_estimate']:^15.6f} | {result['bias']:^10.6f} | {result['variance']:^10.6f}")

# Step 4: Calculate the Variance of the MLE
print_step_header(4, "Calculating the Variance of the MLE")

print("For the Poisson MLE λ̂ = X̄:")
print("Var(λ̂) = Var(X̄) = Var((∑Xᵢ)/n)")
print("       = (1/n²) * Var(∑Xᵢ)")
print("       = (1/n²) * ∑Var(Xᵢ)")
print("       = (1/n²) * n * λ")
print("       = λ/n")
print("\nTherefore, the variance of the MLE is Var(λ̂) = λ/n.")

# Plot the theoretical variance vs. empirical variance
plt.figure(figsize=(10, 6))

# Extract empirical variances from results
n_values = [result['n'] for result in results]
empirical_variances = [result['variance'] for result in results]

# Theoretical variances
theoretical_variances = [lambda_true / n for n in n_values]

# Plot both
plt.plot(n_values, empirical_variances, 'bo-', linewidth=2, label='Empirical Variance')
plt.plot(n_values, theoretical_variances, 'r--', linewidth=2, label='Theoretical Variance: λ/n')

plt.title('Variance of the MLE for Different Sample Sizes', fontsize=14)
plt.xlabel('Sample Size (n)', fontsize=12)
plt.ylabel('Variance of λ̂', fontsize=12)
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "variance_comparison.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 5: Find the Cramér-Rao Lower Bound
print_step_header(5, "Finding the Cramér-Rao Lower Bound")

print("The Cramér-Rao Lower Bound (CRLB) establishes the minimum variance for any unbiased estimator.")
print("It is given by the inverse of the Fisher Information:")
print("CRLB = 1 / I(λ)")
print("\nFor a Poisson distribution, the Fisher Information is:")
print("I(λ) = E[-(d²/dλ²)log f(X|λ)]")
print("\nFrom our log-likelihood function:")
print("d²ℓ(λ)/dλ² = -(∑Xᵢ)/λ²")
print("\nFor a single observation, this becomes:")
print("d²log f(x|λ)/dλ² = -x/λ²")
print("\nTaking the expectation:")
print("I(λ) = E[X/λ²] = E[X]/λ² = λ/λ² = 1/λ")
print("\nFor n independent observations, the total Fisher Information is:")
print("I_n(λ) = n * I(λ) = n/λ")
print("\nTherefore, the CRLB is:")
print("CRLB = 1/I_n(λ) = λ/n")
print("\nSince Var(λ̂) = λ/n = CRLB, the MLE achieves the Cramér-Rao lower bound.")
print("This means the MLE is an efficient estimator, having the minimum possible variance among all unbiased estimators.")

# Visualize the efficiency of the MLE
plt.figure(figsize=(10, 6))

# Plot the CRLB (which is the same as the theoretical variance in this case)
plt.plot(n_values, theoretical_variances, 'r-', linewidth=2, label='CRLB = λ/n')
plt.plot(n_values, empirical_variances, 'bo', markersize=8, label='Empirical Variance')

plt.title('Efficiency of the MLE for Poisson Parameter', fontsize=14)
plt.xlabel('Sample Size (n)', fontsize=12)
plt.ylabel('Variance', fontsize=12)
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "mle_efficiency.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 6: Visual Summary and Interpretation
print_step_header(6, "Visual Summary and Interpretation")

# Create a comprehensive figure to illustrate the findings
plt.figure(figsize=(15, 10))
gs = GridSpec(2, 2)

# Plot 1: Poisson PMF
ax1 = plt.subplot(gs[0, 0])
for lambda_val in [1, 3, 5]:
    pmf = poisson.pmf(x, lambda_val)
    ax1.plot(x, pmf, 'o-', linewidth=2, label=f'λ = {lambda_val}')
ax1.set_title('Poisson Distribution for Different λ Values', fontsize=12)
ax1.set_xlabel('k', fontsize=10)
ax1.set_ylabel('P(X = k)', fontsize=10)
ax1.grid(True)
ax1.legend()

# Plot 2: MLE Distribution converging to normal as n increases
ax2 = plt.subplot(gs[0, 1])
for i, n in enumerate([10, 100, 1000]):
    # Generate samples and compute MLEs
    sample = np.random.poisson(lambda_true, n * 500).reshape(500, n)
    mle_estimates = np.mean(sample, axis=1)
    
    # Plot histogram
    ax2.hist(mle_estimates, bins=20, alpha=0.3, density=True, 
             label=f'n = {n}')

# Add vertical line for true lambda
ax2.axvline(x=lambda_true, color='r', linestyle='--', 
           label=f'True λ = {lambda_true}')
ax2.set_title('Distribution of MLE λ̂ for Different Sample Sizes', fontsize=12)
ax2.set_xlabel('λ̂', fontsize=10)
ax2.set_ylabel('Density', fontsize=10)
ax2.grid(True)
ax2.legend()

# Plot 3: Bias of the MLE
ax3 = plt.subplot(gs[1, 0])
biases = [result['bias'] for result in results]
ax3.plot(n_values, biases, 'go-', linewidth=2)
ax3.axhline(y=0, color='r', linestyle='--', label='Unbiased (Bias = 0)')
ax3.set_title('Bias of the MLE for Different Sample Sizes', fontsize=12)
ax3.set_xlabel('Sample Size (n)', fontsize=10)
ax3.set_ylabel('Bias', fontsize=10)
ax3.set_xscale('log')
ax3.grid(True)
ax3.legend()

# Plot 4: Variance and CRLB
ax4 = plt.subplot(gs[1, 1])
ax4.plot(n_values, theoretical_variances, 'r-', linewidth=2, label='CRLB = λ/n')
ax4.plot(n_values, empirical_variances, 'bo', markersize=8, label='Empirical Variance')
ax4.set_title('Variance of the MLE and CRLB', fontsize=12)
ax4.set_xlabel('Sample Size (n)', fontsize=10)
ax4.set_ylabel('Variance', fontsize=10)
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.grid(True)
ax4.legend()

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "summary_results.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Final summary
print("\nSummary of results:")
print("1. Maximum Likelihood Estimator (MLE) for Poisson parameter λ: λ̂ = X̄ (sample mean)")
print("2. The MLE is unbiased: E[λ̂] = λ")
print("3. Variance of the MLE: Var(λ̂) = λ/n")
print("4. Cramér-Rao Lower Bound: CRLB = λ/n")
print("5. The MLE achieves the CRLB, making it an efficient estimator")

print("\nConclusion:")
print("The sample mean X̄ is the MLE for the Poisson parameter λ. This estimator is:")
print("- Unbiased: It correctly estimates λ on average")
print("- Efficient: It achieves the minimum possible variance as given by the CRLB")
print("- Consistent: Its variance approaches zero as the sample size increases (λ/n → 0 as n → ∞)") 