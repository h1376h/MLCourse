import numpy as np
import matplotlib.pyplot as plt
import os
import time
from scipy.stats import norm, multivariate_normal

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_7_Quiz_13")
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
print("- A deep neural network with 1 million parameters")
print("\nTask:")
print("1. Identify and explain two major computational challenges in applying full Bayesian inference")
print("2. Compare computational requirements of MAP estimation vs. full Bayesian inference")
print("3. Suggest a practical approximation method to make Bayesian inference more tractable")

# Step 2: Computational Challenges in Full Bayesian Inference
print_step_header(2, "Computational Challenges in Full Bayesian Inference")

print("Challenge 1: High-Dimensional Integration")
print("-" * 40)
print("In full Bayesian inference, we need to compute integrals over the entire parameter space")
print("to obtain the posterior distribution and make predictions. For a neural network with")
print("1 million parameters, this means computing integrals in a 1-million dimensional space,")
print("which is computationally intractable using traditional numerical integration methods.")
print("\nThe curse of dimensionality makes the volume of the parameter space grow exponentially")
print("with the number of dimensions, making exhaustive exploration impossible.")
print("\nFor example, even if we only considered 2 possible values for each parameter,")
print("we would need to evaluate 2^(1,000,000) parameter combinations, which is astronomical.")

print("\nChallenge 2: Posterior Landscape Complexity")
print("-" * 40)
print("Neural networks generally have complex, multi-modal posterior distributions with")
print("complicated dependencies between parameters. This makes it difficult to:")
print("- Efficiently sample from the posterior distribution")
print("- Design appropriate proposal distributions for MCMC methods")
print("- Ensure convergence to the true posterior")
print("\nNeural networks can have many local optima, saddle points, and flat regions in the")
print("parameter space, making the posterior landscape difficult to navigate.")
print("\nAdditionally, parameters in neural networks are often highly correlated, which can")
print("cause slow mixing in standard MCMC samplers like Metropolis-Hastings or Gibbs sampling.")

# Step 3: Visualizing the Challenges
print_step_header(3, "Visualizing the Challenges")

# Plot 1: Curse of dimensionality visualization
dimensions = np.arange(1, 11)
relative_volume = np.array([0.1**d for d in dimensions])

plt.figure(figsize=(10, 6))
plt.semilogy(dimensions, relative_volume, 'o-', linewidth=2)
plt.xlabel('Number of Dimensions', fontsize=12)
plt.ylabel('Relative Volume of Unit Hypercube Covered\nby 10% in Each Dimension', fontsize=12)
plt.title('The Curse of Dimensionality', fontsize=14)
plt.grid(True)
plt.xticks(dimensions)
plt.tight_layout()

# Annotate the extreme drop
plt.annotate('At 10 dimensions,\nonly 10^-10 of the space\nis covered',
             xy=(10, 1e-10), xytext=(7, 1e-5),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
             fontsize=10)

plt.text(5.5, 1e-2, 'Imagine 1,000,000 dimensions!', 
         fontsize=12, bbox=dict(facecolor='yellow', alpha=0.2))

file_path = os.path.join(save_dir, "curse_of_dimensionality.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Plot 2: Posterior complexity visualization (2D example)
def complex_posterior(x, y):
    # Create a complex posterior landscape with multiple modes
    z1 = multivariate_normal.pdf(np.dstack((x, y)), mean=[1, 1], cov=[[0.3, 0.1], [0.1, 0.3]])
    z2 = multivariate_normal.pdf(np.dstack((x, y)), mean=[-1, -1], cov=[[0.2, -0.1], [-0.1, 0.2]])
    z3 = multivariate_normal.pdf(np.dstack((x, y)), mean=[1, -1], cov=[[0.2, 0.05], [0.05, 0.2]])
    z4 = multivariate_normal.pdf(np.dstack((x, y)), mean=[-1, 1], cov=[[0.3, -0.05], [-0.05, 0.3]])
    return 0.4*z1 + 0.3*z2 + 0.15*z3 + 0.15*z4

# Generate data for the plot
x_range = np.linspace(-3, 3, 100)
y_range = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = complex_posterior(X, Y)

plt.figure(figsize=(10, 8))

# Plot contour
plt.contourf(X, Y, Z, 20, cmap='viridis')
plt.colorbar(label='Posterior Density')
plt.contour(X, Y, Z, 10, colors='white', alpha=0.5, linestyles='--')

# Mark the modes
plt.scatter([1, -1, 1, -1], [1, -1, -1, 1], color='red', s=100, marker='*')

# Add labels
plt.xlabel('Parameter θ₁', fontsize=12)
plt.ylabel('Parameter θ₂', fontsize=12)
plt.title('Complex Posterior Distribution with Multiple Modes', fontsize=14)
plt.grid(True)
plt.tight_layout()

file_path = os.path.join(save_dir, "complex_posterior.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 4: Comparing MAP vs. Full Bayesian Inference
print_step_header(4, "Comparing MAP vs. Full Bayesian Inference")

print("MAP Estimation:")
print("-" * 40)
print("Maximum A Posteriori (MAP) estimation finds a single point estimate that maximizes")
print("the posterior distribution: θ_MAP = argmax_θ p(θ|D)")
print("\nComputational characteristics:")
print("1. Optimization problem that can be solved using gradient-based methods")
print("2. Similar computational complexity to standard neural network training")
print("3. Only requires computing gradients of log posterior w.r.t. parameters")
print("4. Can leverage highly optimized deep learning libraries and hardware (GPUs, TPUs)")
print("5. Memory requirements scale linearly with the number of parameters")

print("\nFull Bayesian Inference:")
print("-" * 40)
print("Full Bayesian inference computes the entire posterior distribution p(θ|D) and")
print("integrates over it to make predictions: p(y|x,D) = ∫ p(y|x,θ) p(θ|D) dθ")
print("\nComputational characteristics:")
print("1. Requires sampling from a high-dimensional posterior or approximating it")
print("2. MCMC methods need many iterations to converge and provide independent samples")
print("3. Need to store multiple samples of parameter vectors (high memory requirements)")
print("4. Integration over posterior adds significant computation for predictions")
print("5. Many evaluations of the likelihood function required")
print("6. Computational and memory requirements scale poorly with model size")

# Create a comparison table visualization
comparison_metrics = [
    "Training Time", 
    "Memory Usage",
    "Computational Complexity",
    "Hardware Acceleration",
    "Uncertainty Quantification"
]

map_scores = [5, 5, 5, 5, 1]  # Relative scores for MAP (1=worst, 5=best)
bayes_scores = [1, 2, 1, 3, 5]  # Relative scores for Bayesian

plt.figure(figsize=(12, 8))
x = np.arange(len(comparison_metrics))
width = 0.35

plt.bar(x - width/2, map_scores, width, label='MAP Estimation', color='skyblue')
plt.bar(x + width/2, bayes_scores, width, label='Full Bayesian Inference', color='coral')

plt.xlabel('Evaluation Metric', fontsize=12)
plt.ylabel('Score (Higher is Better)', fontsize=12)
plt.title('MAP Estimation vs. Full Bayesian Inference: Comparison', fontsize=14)
plt.xticks(x, comparison_metrics, rotation=15)
plt.yticks(range(0, 6))
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()

# Remove explanations from plot and print them instead
explanations = [
    "MAP is much faster",
    "MAP requires less memory",
    "MAP has lower computational complexity",
    "MAP can use GPUs/TPUs more efficiently",
    "Bayesian provides full uncertainty"
]

print("\nExplanations for comparison metrics:")
for i, metric in enumerate(comparison_metrics):
    print(f"- {metric}: {explanations[i]}")

file_path = os.path.join(save_dir, "map_vs_bayes_comparison.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 5: Theoretical Performance Comparison
print_step_header(5, "Theoretical Performance Comparison")

print("Computational complexity comparison for a neural network with n parameters:")

# Create a table of computational complexities
methods = [
    "MAP Estimation", 
    "MCMC (Metropolis-Hastings)",
    "Hamiltonian Monte Carlo",
    "Variational Inference",
    "Laplace Approximation",
    "Monte Carlo Dropout",
    "Deep Ensembles (5 models)"
]

complexity = [
    "O(n)",          # MAP
    "O(n²)",         # MH
    "O(n log n)",    # HMC
    "O(n)",          # VI
    "O(n²)",         # Laplace
    "O(n)",          # MC Dropout
    "O(5n)"          # Deep Ensembles
]

memory = [
    "O(n)",          # MAP
    "O(S·n)",        # MH (S = number of samples)
    "O(S·n)",        # HMC
    "O(2n)",         # VI (means + variances)
    "O(n²)",         # Laplace (Hessian)
    "O(n)",          # MC Dropout
    "O(5n)"          # Deep Ensembles
]

accuracy = [
    "Point estimate",               # MAP
    "Full posterior (asymptotic)",  # MH
    "Full posterior (efficient)",   # HMC
    "Approximate posterior",        # VI
    "Local approximation",          # Laplace
    "Approximate uncertainty",      # MC Dropout
    "Ensemble uncertainty"          # Deep Ensembles
]

# Create a visualization of method scalability
plt.figure(figsize=(12, 8))

# Model sizes (parameters)
sizes = np.logspace(2, 6, 100)  # 100 to 1 million parameters

# Theoretical computation time curves (simplified models)
map_time = sizes  # O(n)
mh_time = sizes**2  # O(n²)
hmc_time = sizes * np.log(sizes)  # O(n log n)
vi_time = sizes  # O(n)
laplace_time = sizes**2  # O(n²)
mc_dropout_time = sizes  # O(n)
ensemble_time = 5 * sizes  # O(5n)

# Normalize curves for visualization
max_val = np.max([
    map_time[-1], 
    mh_time[-1]/10000,  # Scaled for visibility
    hmc_time[-1], 
    vi_time[-1]*2, 
    laplace_time[-1]/1000,  # Scaled for visibility
    mc_dropout_time[-1]*1.2,
    ensemble_time[-1]
])

plt.loglog(sizes, map_time/max_val, 'b-', label='MAP Estimation', linewidth=2)
plt.loglog(sizes, mh_time/max_val/10000, 'r-', label='MCMC (Metropolis-Hastings)', linewidth=2)
plt.loglog(sizes, hmc_time/max_val, 'g-', label='Hamiltonian Monte Carlo', linewidth=2)
plt.loglog(sizes, vi_time/max_val*2, 'm-', label='Variational Inference', linewidth=2)
plt.loglog(sizes, laplace_time/max_val/1000, 'c-', label='Laplace Approximation', linewidth=2)
plt.loglog(sizes, mc_dropout_time/max_val*1.2, 'y-', label='Monte Carlo Dropout', linewidth=2)
plt.loglog(sizes, ensemble_time/max_val, 'k-', label='Deep Ensembles (5 models)', linewidth=2)

plt.axvline(x=1e6, color='gray', linestyle='--', alpha=0.7)
plt.text(1.1e6, 0.1, '1 million parameters', rotation=90, alpha=0.7)

plt.xlabel('Number of Parameters (log scale)', fontsize=12)
plt.ylabel('Relative Computation Time (log scale)', fontsize=12)
plt.title('Theoretical Computational Scaling of Bayesian Methods', fontsize=14)
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()

file_path = os.path.join(save_dir, "theoretical_scaling.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 6: Practical Approximations for Bayesian Inference
print_step_header(6, "Practical Approximations for Bayesian Inference")

print("Here are some practical approximation methods to make Bayesian inference more tractable")
print("for large neural networks:")

print("\n1. Variational Inference")
print("-" * 40)
print("Variational Inference (VI) approximates the posterior p(θ|D) with a simpler distribution q(θ)")
print("by minimizing the KL divergence between them. This transforms the integration problem into")
print("an optimization problem, which is more tractable for large models.")
print("\nAdvantages:")
print("- Scales better to large networks than MCMC")
print("- Can leverage gradient-based optimization and GPU acceleration")
print("- Provides uncertainty estimates")
print("\nLimitations:")
print("- Quality of approximation depends on the chosen variational family")
print("- Typically underestimates uncertainty")
print("- May not capture multi-modality")

print("\n2. Monte Carlo Dropout")
print("-" * 40)
print("MC Dropout interprets dropout in neural networks as a variational approximation to")
print("a Gaussian process. By keeping dropout active during inference, each forward pass")
print("samples from an approximate posterior.")
print("\nAdvantages:")
print("- Extremely simple to implement in existing networks")
print("- Minimal computational overhead compared to standard training")
print("- Provides reasonable uncertainty estimates")
print("\nLimitations:")
print("- Limited flexibility in the approximate posterior")
print("- Uncertainty quality depends on dropout rate and network architecture")
print("- Not suitable for all types of neural networks")

print("\n3. Laplace Approximation")
print("-" * 40)
print("Laplace approximation fits a Gaussian approximation to the posterior by using the")
print("Hessian at the MAP estimate. This provides a local approximation of uncertainty.")
print("\nAdvantages:")
print("- Builds directly on MAP estimate")
print("- Relatively straightforward to implement")
print("- Can provide good local uncertainty estimates")
print("\nLimitations:")
print("- Requires computing the Hessian, which can be expensive")
print("- Only provides a local approximation near the MAP")
print("- Cannot capture multi-modality")

print("\n4. Deep Ensembles")
print("-" * 40)
print("Trains multiple neural networks with different random initializations and combines")
print("their predictions. While not strictly Bayesian, it provides a practical approach to")
print("uncertainty quantification.")
print("\nAdvantages:")
print("- Simple to implement")
print("- Can capture some multi-modality")
print("- Often competitive with or superior to formal Bayesian methods")
print("\nLimitations:")
print("- Computationally expensive (requires training multiple networks)")
print("- Storage requirements scale linearly with the number of ensemble members")
print("- Not a formal Bayesian approach (though can be interpreted as such)")

print("\n5. Stochastic Gradient MCMC")
print("-" * 40)
print("Methods like Stochastic Gradient Langevin Dynamics (SGLD) combine stochastic gradient")
print("descent with Langevin dynamics to sample from the posterior using mini-batches of data.")
print("\nAdvantages:")
print("- More scalable than traditional MCMC")
print("- Can leverage mini-batch processing")
print("- Theoretically converges to the true posterior")
print("\nLimitations:")
print("- Slower than variational methods")
print("- Samples may be correlated")
print("- Requires careful tuning of hyperparameters")

# Visualization of the different approximation methods
# We'll show how they approximate a complex 1D posterior
def true_posterior(x):
    """A complex multimodal posterior"""
    return 0.4 * norm.pdf(x, 0, 1) + 0.3 * norm.pdf(x, 4, 0.8) + 0.3 * norm.pdf(x, -4, 0.8)

def map_estimate(x, posterior):
    """MAP estimate (just a point)"""
    idx = np.argmax(posterior)
    return x[idx]

def laplace_approx(x, posterior, map_idx):
    """Laplace approximation (local Gaussian)"""
    # Find the curvature at the MAP (approximate second derivative)
    if map_idx > 0 and map_idx < len(posterior) - 1:
        h = x[1] - x[0]
        d2 = (posterior[map_idx+1] - 2*posterior[map_idx] + posterior[map_idx-1]) / (h*h)
        var = -1 / d2 if d2 < 0 else 1.0  # Ensure positive variance
        return norm.pdf(x, x[map_idx], np.sqrt(var))
    else:
        return norm.pdf(x, x[map_idx], 1.0)

def vi_approx(x, true_posterior):
    """Simple variational approximation (unimodal Gaussian)"""
    # For simplicity, find mean and variance of the true posterior
    mean = np.sum(x * true_posterior) / np.sum(true_posterior)
    var = np.sum((x - mean)**2 * true_posterior) / np.sum(true_posterior)
    return norm.pdf(x, mean, np.sqrt(var))

def mc_dropout_approx(x, true_posterior, n_samples=5):
    """MC Dropout approximation (mixture of similar Gaussians)"""
    # Simulate by sampling points near the main mode and fitting Gaussians
    main_mode = x[np.argmax(true_posterior)]
    np.random.seed(42)  # For reproducibility
    samples = main_mode + np.random.normal(0, 1, n_samples)
    mixture = np.zeros_like(x)
    for s in samples:
        mixture += norm.pdf(x, s, 0.8)
    return mixture / n_samples

def ensemble_approx(x, true_posterior, n_models=3):
    """Deep ensemble approximation (mixture of more varied predictions)"""
    # Simulate by finding multiple local modes
    peaks = [-4, 0, 4]  # Locations of the modes in our true posterior
    np.random.seed(42)  # For reproducibility
    mixture = np.zeros_like(x)
    for peak in peaks[:n_models]:
        weight = true_posterior[np.abs(x - peak).argmin()]
        variance = np.random.uniform(0.5, 2.0)  # Random variance for each ensemble member
        mixture += norm.pdf(x, peak, np.sqrt(variance))
    return mixture / n_models

# Generate data for the plot
x_range = np.linspace(-8, 8, 1000)
true_post = true_posterior(x_range)
map_idx = np.argmax(true_post)

# Create plot
plt.figure(figsize=(12, 8))

# Plot the true posterior
plt.plot(x_range, true_post, 'k-', linewidth=2.5, label='True Posterior')

# Plot the approximations
plt.plot(x_range, laplace_approx(x_range, true_post, map_idx), 'g--', linewidth=2, 
         label='Laplace Approximation')
plt.plot(x_range, vi_approx(x_range, true_post), 'r--', linewidth=2, 
         label='Variational Inference')
plt.plot(x_range, mc_dropout_approx(x_range, true_post), 'b--', linewidth=2, 
         label='MC Dropout')
plt.plot(x_range, ensemble_approx(x_range, true_post), 'm--', linewidth=2, 
         label='Deep Ensemble')

# Mark the MAP
plt.axvline(x=x_range[map_idx], color='c', linestyle='-', linewidth=2, label='MAP Estimate')

plt.xlabel('Parameter θ', fontsize=12)
plt.ylabel('Posterior Density', fontsize=12)
plt.title('Comparison of Bayesian Approximation Methods', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()

file_path = os.path.join(save_dir, "approximation_methods.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Summary
print_step_header(7, "Summary and Recommendations")

print("Summary:")
print("-" * 40)
print("1. Full Bayesian inference for deep neural networks with 1 million parameters faces")
print("   major computational challenges: high-dimensional integration and posterior complexity.")
print("\n2. MAP estimation is significantly more computationally efficient than full Bayesian")
print("   inference, but doesn't capture parameter uncertainty.")
print("\n3. Practical approximation methods like variational inference, MC dropout, Laplace")
print("   approximation, deep ensembles, and SG-MCMC offer different trade-offs between")
print("   computational efficiency and quality of uncertainty estimates.")

print("\nRecommendations:")
print("-" * 40)
print("For a neural network with 1 million parameters:")
print("- Start with MC Dropout if you already use dropout in your network")
print("- Use Deep Ensembles if you can afford the computational cost and need reliable uncertainty")
print("- Consider variational inference if you need a more formal Bayesian approach with scalability")
print("- Use Laplace approximation if you need a quick uncertainty estimate around your MAP solution")
print("- Consider SG-MCMC methods if you need more thorough posterior exploration at scale")

print("\nThe best approach depends on your specific requirements regarding:")
print("- Computational budget")
print("- Importance of uncertainty quantification")
print("- Need for capturing multi-modality")
print("- Available implementation tools")
print("- Domain-specific requirements") 