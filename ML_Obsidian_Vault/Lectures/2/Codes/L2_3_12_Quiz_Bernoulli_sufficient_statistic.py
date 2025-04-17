import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from scipy.stats import binom

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_3_12")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Set up the problem
print_step_header(1, "Understanding the Problem")

print("Given:")
print("- We have a random sample X₁, X₂, ..., Xₙ from a Bernoulli distribution with parameter p")
print("- Each Xᵢ is either 0 or 1 with P(Xᵢ = 1) = p")
print("- The random variables are independent")
print()
print("Question: What is the sufficient statistic for estimating the parameter p?")
print()
print("Options:")
print("A) The sample median")
print("B) The sample mean")
print("C) The sample variance")
print("D) The sample size")
print()

# Step 2: Review the Bernoulli distribution
print_step_header(2, "Understanding Bernoulli Distribution")

print("The Bernoulli distribution is a discrete probability distribution for a random variable")
print("that takes the value 1 with probability p and the value 0 with probability 1-p.")
print()
print("Properties of Bernoulli(p):")
print("- PMF: P(X = x) = p^x * (1-p)^(1-x) for x ∈ {0, 1}")
print("- Mean: E[X] = p")
print("- Variance: Var(X) = p(1-p)")
print()

# Visualize Bernoulli PMF for different p values
p_values = [0.2, 0.5, 0.8]
x = np.array([0, 1])

plt.figure(figsize=(10, 6))
for p in p_values:
    pmf = [1-p, p]  # P(X=0) = 1-p, P(X=1) = p
    plt.bar(x + (p-0.5)*0.1, pmf, width=0.1, alpha=0.7, label=f'p = {p}')
    
plt.title('Bernoulli Probability Mass Function (PMF)', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.xticks([0, 1])
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "bernoulli_pmf.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 3: Define what a sufficient statistic is
print_step_header(3, "Definition of Sufficient Statistic")

print("A statistic T(X) is sufficient for a parameter θ if the conditional distribution")
print("of the sample X given T(X) does not depend on θ.")
print()
print("Intuitively, a sufficient statistic contains all the information in the data")
print("that is relevant for estimating the parameter.")
print()
print("We can use the factorization theorem to identify sufficient statistics:")
print("A statistic T(X) is sufficient for θ if and only if the likelihood function can be factorized as:")
print("L(θ; x) = g(T(x), θ) · h(x)")
print("where g depends on x only through T(x), and h does not depend on θ.")
print()

# Step 4: Derive the likelihood function for Bernoulli samples
print_step_header(4, "Deriving the Likelihood Function")

print("For a sample X₁, X₂, ..., Xₙ from Bernoulli(p), the likelihood function is:")
print("L(p; x₁, x₂, ..., xₙ) = ∏ᵢ₌₁ⁿ p^xᵢ * (1-p)^(1-xᵢ)")
print()
print("We can rewrite this as:")
print("L(p; x₁, x₂, ..., xₙ) = p^(∑xᵢ) * (1-p)^(n-∑xᵢ)")
print()
print("where ∑xᵢ is the sum of all observations, which equals the number of 1's in the sample.")
print()
print("Using the factorization theorem, we can identify that T(X) = ∑xᵢ is a sufficient statistic")
print("because the likelihood depends on the data only through this sum.")
print()
print("The sample mean X̄ = (1/n)∑xᵢ is a one-to-one function of ∑xᵢ, so it is also a sufficient statistic.")
print("Since the sample size n is known, knowing X̄ is equivalent to knowing ∑xᵢ.")
print()

# Step 5: Visual demonstration with simulations
print_step_header(5, "Demonstration with Simulations")

# Simulate samples from Bernoulli distributions with different p values
np.random.seed(42)
p_true = 0.7
sample_sizes = [10, 50, 200, 1000]
num_simulations = 1000

plt.figure(figsize=(12, 8))

for i, n in enumerate(sample_sizes):
    sample_means = []
    sample_sums = []
    sample_medians = []
    sample_variances = []
    
    for _ in range(num_simulations):
        # Generate one sample of size n
        sample = np.random.binomial(1, p_true, n)
        
        # Calculate different statistics
        sample_means.append(np.mean(sample))
        sample_sums.append(np.sum(sample))
        sample_medians.append(np.median(sample))
        sample_variances.append(np.var(sample))
    
    # Plot histograms of the sample means
    plt.subplot(2, 2, i+1)
    plt.hist(sample_means, bins=20, alpha=0.7, color='blue',
             label=f'Mean: {np.mean(sample_means):.3f}\nStd: {np.std(sample_means):.3f}')
    plt.axvline(p_true, color='r', linestyle='--', label=f'True p: {p_true}')
    plt.title(f'Distribution of Sample Mean (n={n})', fontsize=12)
    plt.xlabel('Sample Mean', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.legend()
    plt.grid(True)
    
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "sample_mean_distribution.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Compare different statistics
plt.figure(figsize=(12, 10))

n = 50  # Use a moderate sample size for comparison
sample_means = []
sample_medians = []
sample_variances = []
    
for _ in range(num_simulations):
    # Generate one sample of size n
    sample = np.random.binomial(1, p_true, n)
    
    # Calculate different statistics
    sample_means.append(np.mean(sample))
    sample_medians.append(np.median(sample))
    sample_variances.append(np.var(sample))

# Plot the distributions of different statistics
plt.subplot(3, 1, 1)
plt.hist(sample_means, bins=20, alpha=0.7, color='blue',
         label=f'Mean: {np.mean(sample_means):.3f}\nStd: {np.std(sample_means):.3f}')
plt.axvline(p_true, color='r', linestyle='--', label=f'True p: {p_true}')
plt.title(f'Distribution of Sample Mean (n={n})', fontsize=12)
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.hist(sample_medians, bins=np.linspace(0, 1, 21), alpha=0.7, color='green',
         label=f'Mean: {np.mean(sample_medians):.3f}\nStd: {np.std(sample_medians):.3f}')
plt.axvline(p_true, color='r', linestyle='--', label=f'True p: {p_true}')
plt.title(f'Distribution of Sample Median (n={n})', fontsize=12)
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
expected_variance = p_true * (1 - p_true)
plt.hist(sample_variances, bins=20, alpha=0.7, color='purple',
         label=f'Mean: {np.mean(sample_variances):.3f}\nStd: {np.std(sample_variances):.3f}')
plt.axvline(expected_variance, color='r', linestyle='--', label=f'Expected Var: {expected_variance:.3f}')
plt.title(f'Distribution of Sample Variance (n={n})', fontsize=12)
plt.legend()
plt.grid(True)

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "statistics_comparison.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 6: Visualize the relationship between statistic and likelihood
print_step_header(6, "Relationship Between Sufficient Statistic and Likelihood")

# Create a comparison of likelihoods for different sample realizations but with the same sufficient statistic
n = 10  # Number of observations
k = 7   # Number of successes (sum of observations)

# Generate 3 different samples with the same sum
samples = [
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],  # Sample 1 with sum = 7
    [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],  # Sample 2 with sum = 7
    [0, 1, 1, 0, 1, 1, 1, 1, 1, 0]   # Sample 3 with sum = 7
]

# Calculate likelihood function for each sample
p_values = np.linspace(0.01, 0.99, 100)
likelihoods = []

for sample in samples:
    likelihood = []
    for p in p_values:
        l = p**sum(sample) * (1-p)**(n-sum(sample))
        likelihood.append(l)
    likelihoods.append(likelihood)

# Plot the likelihoods
plt.figure(figsize=(10, 6))
linestyles = ['-', '--', '-.']
colors = ['blue', 'green', 'purple']

for i, (sample, likelihood) in enumerate(zip(samples, likelihoods)):
    plt.plot(p_values, likelihood, linestyle=linestyles[i], color=colors[i], linewidth=2,
             label=f'Sample {i+1}: {sample}')

plt.title('Likelihood Functions for Different Samples with Same Sum (k=7)', fontsize=14)
plt.xlabel('Parameter p', fontsize=12)
plt.ylabel('Likelihood', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "likelihood_comparison.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 7: Visualize the maximum likelihood estimation
print_step_header(7, "Maximum Likelihood Estimation")

# Sample size and number of successes
n_values = [10, 50, 100]
k_values = [7, 35, 70]  # Same proportion of successes (0.7)

plt.figure(figsize=(12, 4))

for i, (n, k) in enumerate(zip(n_values, k_values)):
    # Calculate likelihood function
    p_values = np.linspace(0.01, 0.99, 100)
    likelihood = [p**k * (1-p)**(n-k) for p in p_values]
    
    # Calculate MLE
    mle = k / n
    
    # Plot
    plt.subplot(1, 3, i+1)
    plt.plot(p_values, likelihood, 'b-', linewidth=2)
    plt.axvline(mle, color='r', linestyle='--', label=f'MLE: {mle:.2f}')
    plt.title(f'Likelihood for n={n}, k={k}', fontsize=12)
    plt.xlabel('Parameter p', fontsize=10)
    plt.ylabel('Likelihood', fontsize=10)
    plt.grid(True)
    plt.legend()

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "mle_estimation.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 8: Conclusion
print_step_header(8, "Conclusion and Answer")

print("Based on our analysis:")
print("1. The likelihood function for a Bernoulli sample depends on the data only through ∑xᵢ (the sum of observations).")
print("2. This makes ∑xᵢ a sufficient statistic for p.")
print("3. The sample mean X̄ = (1/n)∑xᵢ is equivalent to ∑xᵢ when n is known.")
print("4. Neither the sample median nor sample variance are sufficient statistics.")
print("5. The sample size by itself contains no information about the parameter p.")
print()
print("Therefore, the correct answer is:")
print("B) The sample mean")
print()
print("The sample mean X̄ is a sufficient statistic for the parameter p of a Bernoulli distribution.")
print("This is because the sample mean contains all the information available in the sample for estimating p.")
print("In fact, the MLE of p is precisely the sample mean: p̂ = X̄ = (1/n)∑xᵢ.") 