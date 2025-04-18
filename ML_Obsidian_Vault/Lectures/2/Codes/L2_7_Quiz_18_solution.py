import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_7_Quiz_18")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Understanding the Problem
print_step_header(1, "Understanding the Problem")

print("Given:")
print("- We have N IID samples following the model: x_i = A + n_i")
print("- The sample distribution is normal: f(x|A) = (1/√(2πσ²))e^(-(x_i-A)²/(2σ²))")
print("- The prior for A is also normal: f(A) = (1/√(2πσ_A²))e^(-(A-μ_A)²/(2σ_A²))")
print("- Parameters σ², σ_A², and μ_A are constant and known")
print("\nTask:")
print("1. Determine the MAP estimator for parameter A")
print("2. Analyze what happens when σ_A² is extremely large")

# Step 2: Deriving the MAP Estimator
print_step_header(2, "Deriving the MAP Estimator")

print("The posterior distribution p(A|x) is proportional to the likelihood p(x|A) times the prior p(A):")
print("p(A|x) ∝ p(x|A) × p(A)")

print("\nLikelihood for N IID samples:")
print("p(x|A) = ∏_{i=1}^N (1/√(2πσ²))e^(-(x_i-A)²/(2σ²))")
print("log p(x|A) = -N/2 log(2πσ²) - 1/(2σ²) ∑_{i=1}^N (x_i-A)²")

print("\nPrior distribution:")
print("p(A) = (1/√(2πσ_A²))e^(-(A-μ_A)²/(2σ_A²))")
print("log p(A) = -1/2 log(2πσ_A²) - (A-μ_A)²/(2σ_A²)")

print("\nLog posterior (ignoring constants):")
print("log p(A|x) ∝ log p(x|A) + log p(A)")
print("log p(A|x) ∝ -1/(2σ²) ∑_{i=1}^N (x_i-A)² - (A-μ_A)²/(2σ_A²)")

print("\nExpanding the sum in the likelihood term:")
print("∑_{i=1}^N (x_i-A)² = ∑_{i=1}^N x_i² - 2A∑_{i=1}^N x_i + NA²")
print("∑_{i=1}^N (x_i-A)² = ∑_{i=1}^N x_i² - 2A∑_{i=1}^N x_i + NA²")

print("\nPutting this back into the log posterior:")
print("log p(A|x) ∝ -1/(2σ²)[∑_{i=1}^N x_i² - 2A∑_{i=1}^N x_i + NA²] - (A-μ_A)²/(2σ_A²)")
print("log p(A|x) ∝ -1/(2σ²)[∑_{i=1}^N x_i² - 2A∑_{i=1}^N x_i + NA²] - (A²-2Aμ_A+μ_A²)/(2σ_A²)")

print("\nRearranging to collect terms with A:")
print("log p(A|x) ∝ -1/(2σ²)[∑_{i=1}^N x_i² - 2A∑_{i=1}^N x_i + NA²] - (A²-2Aμ_A+μ_A²)/(2σ_A²)")
print("log p(A|x) ∝ -[NA²-2A∑_{i=1}^N x_i]/(2σ²) - [A²-2Aμ_A]/(2σ_A²) + constant terms")
print("log p(A|x) ∝ -[NA²/(2σ²) - (∑_{i=1}^N x_i)A/σ² + A²/(2σ_A²) - Aμ_A/σ_A²] + constant terms")
print("log p(A|x) ∝ -[A²(N/(2σ²) + 1/(2σ_A²)) - A((∑_{i=1}^N x_i)/σ² + μ_A/σ_A²)] + constant terms")

print("\nTo find the MAP estimate, we take the derivative with respect to A and set it to zero:")
print("d/dA(log p(A|x)) = -[2A(N/(2σ²) + 1/(2σ_A²)) - ((∑_{i=1}^N x_i)/σ² + μ_A/σ_A²)]")
print("Setting this equal to zero:")
print("2A(N/(2σ²) + 1/(2σ_A²)) = (∑_{i=1}^N x_i)/σ² + μ_A/σ_A²")

print("\nSolving for A:")
print("A = [(∑_{i=1}^N x_i)/σ² + μ_A/σ_A²] / [2(N/(2σ²) + 1/(2σ_A²))]")
print("A = [(∑_{i=1}^N x_i)/σ² + μ_A/σ_A²] / [N/σ² + 1/σ_A²]")

print("\nThis can be rewritten as a weighted average:")
print("A_MAP = (N/σ² × x̄ + 1/σ_A² × μ_A) / (N/σ² + 1/σ_A²)")
print("Where x̄ = (∑_{i=1}^N x_i)/N is the sample mean.")

print("\nAlternative form:")
print("A_MAP = (σ_A² × N × x̄ + σ² × μ_A) / (σ_A² × N + σ²)")

# Step 3: Analyzing the MAP Estimator
print_step_header(3, "Analyzing the MAP Estimator")

print("The MAP estimator for A can be written as:")
print("A_MAP = (σ_A² × N × x̄ + σ² × μ_A) / (σ_A² × N + σ²)")

print("\nThis is a weighted average of the sample mean (x̄) and the prior mean (μ_A):")
print("A_MAP = w × x̄ + (1-w) × μ_A   where w = (σ_A² × N) / (σ_A² × N + σ²)")

print("\nThe weight 'w' determines how much we trust the data versus the prior:")
print("- When σ_A² is small (strong prior), w is closer to 0, and A_MAP is closer to μ_A")
print("- When σ_A² is large (weak prior), w is closer to 1, and A_MAP is closer to x̄")
print("- When N is large (lots of data), w is closer to 1, and A_MAP is closer to x̄")

print("\nWhen σ_A² → ∞ (extremely large), the prior becomes non-informative:")
print("lim_{σ_A² → ∞} A_MAP = lim_{σ_A² → ∞} (σ_A² × N × x̄ + σ² × μ_A) / (σ_A² × N + σ²)")
print("lim_{σ_A² → ∞} A_MAP = lim_{σ_A² → ∞} (σ_A² × N × x̄) / (σ_A² × N) = x̄")

print("\nTherefore, when σ_A² is extremely large, the MAP estimator approaches the sample mean (x̄),")
print("which is exactly the Maximum Likelihood Estimator (MLE) for this problem.")

# Step 4: Visualization
print_step_header(4, "Visualization of the MAP Estimator")

# Function to calculate the MAP estimate
def calculate_MAP(x_bar, mu_A, sigma_sq, sigma_A_sq, N):
    numerator = sigma_A_sq * N * x_bar + sigma_sq * mu_A
    denominator = sigma_A_sq * N + sigma_sq
    return numerator / denominator

# Setup some example values
N_samples = 10  # Number of samples
sigma_sq = 1.0  # Data variance
mu_A_true = 5.0  # True parameter value for generating data
x_bar = mu_A_true + np.random.normal(0, np.sqrt(sigma_sq/N_samples))  # Sample mean with some noise

print(f"Example scenario:")
print(f"- True parameter A = {mu_A_true}")
print(f"- Number of samples N = {N_samples}")
print(f"- Data variance σ² = {sigma_sq}")
print(f"- Sample mean x̄ = {x_bar:.4f}")

# Different prior means to visualize
mu_A_values = [2.0, 5.0, 8.0]
prior_colors = ['r', 'g', 'b']

plt.figure(figsize=(12, 8))

# Varying prior variance to see effect on MAP estimate
sigma_A_sq_values = np.logspace(-1, 2, 100)  # From 0.1 to 100
map_estimates = {}

for i, mu_A in enumerate(mu_A_values):
    map_estimates[mu_A] = [calculate_MAP(x_bar, mu_A, sigma_sq, s, N_samples) for s in sigma_A_sq_values]
    plt.plot(sigma_A_sq_values, map_estimates[mu_A], 
             color=prior_colors[i], linestyle='-', linewidth=2,
             label=f'Prior μ_A = {mu_A}')
    
    # Mark the special points
    plt.scatter([sigma_A_sq_values[0]], [map_estimates[mu_A][0]], 
                color=prior_colors[i], s=100, marker='o')
    plt.scatter([sigma_A_sq_values[-1]], [map_estimates[mu_A][-1]], 
                color=prior_colors[i], s=100, marker='s')

# Add a horizontal line for the sample mean (MLE)
plt.axhline(y=x_bar, color='k', linestyle='--', linewidth=2, 
            label=f'Sample mean (MLE) = {x_bar:.4f}')

plt.xscale('log')
plt.xlabel('Prior Variance (σ_A²)', fontsize=12)
plt.ylabel('MAP Estimate for A', fontsize=12)
plt.title('Effect of Prior Variance on MAP Estimate', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "prior_variance_effect.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Visualize the posterior distributions for different prior variances
plt.figure(figsize=(12, 8))

# Function to calculate unnormalized log posterior
def log_posterior(A, x_bar, mu_A, sigma_sq, sigma_A_sq, N):
    log_likelihood = -N/(2*sigma_sq) * (x_bar - A)**2 * N  # Using N*(x_bar - A)² to represent Σ(x_i - A)²
    log_prior = -1/(2*sigma_A_sq) * (A - mu_A)**2
    return log_likelihood + log_prior

# Function to normalize the posterior for plotting
def normalized_posterior(A_values, x_bar, mu_A, sigma_sq, sigma_A_sq, N):
    log_post = [log_posterior(A, x_bar, mu_A, sigma_sq, sigma_A_sq, N) for A in A_values]
    post = np.exp(log_post - np.max(log_post))  # Stabilize computation
    return post / np.trapz(post, A_values)  # Normalize

# Create range of A values to plot
A_values = np.linspace(1, 9, 1000)

# Select a prior mean for visualization
mu_A = 2.0  # Using a prior mean that's different from the true value

# Different prior variances
sigma_A_sq_examples = [0.1, 1.0, 10.0, 100.0]
line_styles = ['-', '--', '-.', ':']

for i, sigma_A_sq in enumerate(sigma_A_sq_examples):
    # Calculate MAP for this scenario
    map_value = calculate_MAP(x_bar, mu_A, sigma_sq, sigma_A_sq, N_samples)
    
    # Calculate and plot the posterior
    posterior = normalized_posterior(A_values, x_bar, mu_A, sigma_sq, sigma_A_sq, N_samples)
    plt.plot(A_values, posterior, 
             linestyle=line_styles[i], linewidth=2, 
             label=f'σ_A² = {sigma_A_sq:.1f}, MAP = {map_value:.4f}')
    
    # Mark the MAP estimate
    idx = np.argmax(posterior)
    plt.scatter([A_values[idx]], [posterior[idx]], s=100, marker='o')

# Add the likelihood (which is proportional to the posterior with a flat prior)
likelihood = normalized_posterior(A_values, x_bar, 0, sigma_sq, 1e6, N_samples)  # Using a very large prior variance
plt.plot(A_values, likelihood, 'k--', linewidth=2, label=f'Likelihood (MLE = {x_bar:.4f})')

plt.axvline(x=mu_A, color='r', linestyle='--', linewidth=1, label=f'Prior Mean = {mu_A}')
plt.axvline(x=x_bar, color='g', linestyle='--', linewidth=1, label=f'Sample Mean = {x_bar:.4f}')

plt.xlabel('Parameter A', fontsize=12)
plt.ylabel('Posterior Density', fontsize=12)
plt.title('Posterior Distributions for Different Prior Variances', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "posterior_distributions.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Visualize the weight of the sample mean in the MAP estimator
plt.figure(figsize=(10, 6))

# Function to calculate the weight of the sample mean
def sample_mean_weight(sigma_sq, sigma_A_sq, N):
    return (sigma_A_sq * N) / (sigma_A_sq * N + sigma_sq)

# Different sample sizes
N_values = [1, 5, 10, 20, 50]
line_styles = ['-', '--', '-.', ':', '-']

for i, N in enumerate(N_values):
    weights = [sample_mean_weight(sigma_sq, s, N) for s in sigma_A_sq_values]
    plt.plot(sigma_A_sq_values, weights, 
             linestyle=line_styles[i], linewidth=2, 
             label=f'N = {N}')

plt.xscale('log')
plt.xlabel('Prior Variance (σ_A²)', fontsize=12)
plt.ylabel('Weight of Sample Mean in MAP Estimate', fontsize=12)
plt.title('How Sample Size and Prior Variance Affect Weight of Data', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "sample_mean_weight.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 5: Summary and Conclusions
print_step_header(5, "Summary and Conclusions")

print("Key findings:")
print("1. The MAP estimator for A is: A_MAP = (σ_A² × N × x̄ + σ² × μ_A) / (σ_A² × N + σ²)")
print("2. This is a weighted average between the sample mean (x̄) and the prior mean (μ_A)")
print("3. When σ_A² → ∞, the MAP estimator approaches the sample mean (x̄), which is the MLE")
print("4. The influence of the prior depends on both σ_A² and the sample size N")
print("5. With large sample sizes, the data dominates the estimate regardless of the prior")

print("\nPractical implications:")
print("- The MAP estimator provides a systematic way to combine prior knowledge with observed data")
print("- It automatically balances between prior beliefs and evidence from data")
print("- As more data is collected, the influence of the prior naturally diminishes")
print("- When we have little confidence in our prior (σ_A² is large), the estimator relies more on the data")
print("- The MAP framework naturally encompasses the MLE as a special case with a non-informative prior") 