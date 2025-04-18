import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import multivariate_normal

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_7_Quiz_17")
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
print("- Neural network with weights w and dataset D")
print("- Log-likelihood is log p(D|w)")
print("- Need to approximate the posterior p(w|D) using Laplace approximation")
print("\nTask:")
print("1. Explain the key idea behind Laplace approximation of the posterior")
print("2. Write the Laplace approximation formula for p(w|D)")
print("3. Explain how to use this approximation for predictive distribution p(y_new|x_new, D)")
print("4. Compare the computational efficiency of this approach with MCMC sampling")

# Step 2: Key Idea Behind Laplace Approximation
print_step_header(2, "Key Idea Behind Laplace Approximation")

print("The key idea behind Laplace approximation is to approximate the posterior")
print("distribution p(w|D) with a multivariate Gaussian distribution centered at the MAP")
print("estimate of the parameters (w_MAP). This is done by using a second-order Taylor")
print("expansion of the log-posterior around w_MAP.")
print("\nLaplace approximation is based on the observation that as the amount of data")
print("increases, the posterior distribution often becomes more concentrated and")
print("approximately Gaussian in shape.")
print("\nThe steps involved are:")
print("1. Find the Maximum A Posteriori (MAP) estimate w_MAP by maximizing the log-posterior")
print("2. Compute the Hessian (H) of the negative log-posterior at w_MAP")
print("3. Approximate the posterior as a multivariate Gaussian with mean w_MAP and")
print("   covariance matrix given by the inverse of the Hessian")

# Create a visualization of the Laplace approximation concept
# Generate a 1D example
w_range = np.linspace(-5, 5, 1000)
# True posterior (non-Gaussian for illustration)
posterior = np.exp(-0.5 * (w_range**4 - 4*w_range**2))
posterior /= np.sum(posterior) * (w_range[1] - w_range[0])  # Normalize

# Find MAP
w_map_idx = np.argmax(posterior)
w_map = w_range[w_map_idx]

# Fit Gaussian at the MAP (Laplace approximation)
# For this example, manually compute the second derivative
h = 0.001
second_deriv = ((np.log(posterior[w_map_idx+1]) - 2*np.log(posterior[w_map_idx]) + 
                np.log(posterior[w_map_idx-1])) / h**2)
sigma = np.sqrt(1.0 / -second_deriv)

# Generate Laplace approximation
laplace_approx = multivariate_normal.pdf(w_range, mean=w_map, cov=sigma**2)

plt.figure(figsize=(10, 6))
plt.plot(w_range, posterior, 'b-', label='True Posterior', linewidth=2)
plt.plot(w_range, laplace_approx, 'r--', label='Laplace Approximation', linewidth=2)
plt.axvline(x=w_map, color='g', linestyle='-', 
            label=f'MAP Estimate (w_MAP = {w_map:.2f})', linewidth=1.5)

plt.xlabel('Parameter (w)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title('Laplace Approximation of a Posterior Distribution', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "laplace_approximation_concept.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 3: Laplace Approximation Formula
print_step_header(3, "Laplace Approximation Formula")

print("The Laplace approximation of the posterior p(w|D) is a multivariate Gaussian:")
print("\np(w|D) ≈ N(w | w_MAP, H⁻¹)")
print("\nwhere:")
print("- w_MAP is the MAP estimate of the parameters")
print("- H is the Hessian matrix of the negative log-posterior at w_MAP")
print("- H⁻¹ is the inverse of the Hessian, which serves as the covariance matrix")
print("\nMore explicitly, the approximation is:")
print("\np(w|D) ≈ (2π)^(-d/2) |H|^(1/2) exp(-(1/2)(w - w_MAP)ᵀ H (w - w_MAP))")
print("\nwhere d is the dimensionality of w (number of parameters).")
print("\nThe Hessian H is given by:")
print("\nH = -∇²log p(w|D)|_{w=w_MAP}")
print("\nOr, using Bayes' rule and assuming the prior is also Gaussian:")
print("\nH = -∇²log p(D|w)|_{w=w_MAP} - ∇²log p(w)|_{w=w_MAP}")

# Step 4: Using Laplace Approximation for Prediction
print_step_header(4, "Using Laplace Approximation for Prediction")

print("To compute the predictive distribution p(y_new|x_new, D) using the Laplace")
print("approximation, we need to marginalize over the parameter posterior:")
print("\np(y_new|x_new, D) = ∫ p(y_new|x_new, w) p(w|D) dw")
print("\nWith the Laplace approximation, p(w|D) ≈ N(w | w_MAP, H⁻¹), we have:")
print("\np(y_new|x_new, D) ≈ ∫ p(y_new|x_new, w) N(w | w_MAP, H⁻¹) dw")
print("\nThis integral typically doesn't have a closed-form solution for neural networks,")
print("so it's usually approximated using Monte Carlo sampling:")
print("\n1. Generate samples from the approximate posterior: w⁽ⁱ⁾ ~ N(w_MAP, H⁻¹)")
print("2. For each sample, compute the prediction: p(y_new|x_new, w⁽ⁱ⁾)")
print("3. Average these predictions to get: p(y_new|x_new, D) ≈ (1/M)∑ᵢp(y_new|x_new, w⁽ⁱ⁾)")
print("\nIn practice, for a neural network classification problem, this would give us")
print("a probability distribution over classes that accounts for parameter uncertainty.")

# Create a visualization of the predictive distribution
# Simulate a simple 1D regression example
np.random.seed(42)
# Generate some data
n_samples = 20
X = np.linspace(-4, 4, n_samples)
y = 0.5 * X**2 + np.random.normal(0, 1, n_samples)

# Define a simple model: y = w₁ + w₂x + w₃x²
def model(x, w):
    return w[0] + w[1]*x + w[2]*x**2

# For this example, let's assume we've already found w_MAP and the Hessian
# In practice, these would be computed using optimization
w_map = np.array([0.1, 0.0, 0.5])  # Example MAP estimate

# Instead of a full Hessian, we'll use a simplified diagonal approximation
# In practice, this would be the full Hessian matrix
H_inv = np.diag([0.2, 0.1, 0.05])  # Example covariance matrix

# Generate new x values for prediction
x_new = np.linspace(-5, 5, 100)

# Generate samples from the approximate posterior
n_samples = 100
w_samples = np.random.multivariate_normal(w_map, H_inv, n_samples)

# Compute predictions for each sample
y_samples = np.zeros((n_samples, len(x_new)))
for i, w in enumerate(w_samples):
    y_samples[i] = model(x_new, w)

# Compute mean and 95% CI of the predictions
y_mean = np.mean(y_samples, axis=0)
y_std = np.std(y_samples, axis=0)
y_upper = y_mean + 1.96 * y_std
y_lower = y_mean - 1.96 * y_std

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Observed Data')
plt.plot(x_new, model(x_new, w_map), 'r-', 
         label='MAP Prediction (without uncertainty)', linewidth=2)

# Plot the uncertainty
plt.fill_between(x_new, y_lower, y_upper, color='gray', alpha=0.3,
                label='95% CI from Laplace Approximation')

# Plot a subset of samples
for i in range(0, 20, 4):
    plt.plot(x_new, y_samples[i], 'g-', alpha=0.1)
plt.plot([], [], 'g-', alpha=0.5, label='Samples from Posterior')  # For legend

plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Predictive Distribution using Laplace Approximation', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "laplace_predictive_distribution.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 5: Computational Efficiency Comparison with MCMC
print_step_header(5, "Computational Efficiency Comparison with MCMC")

print("Comparing the computational efficiency of Laplace approximation with MCMC sampling:")

print("\nLaplace Approximation:")
print("1. Requires finding the MAP estimate (optimization problem)")
print("2. Requires computing the Hessian matrix at the MAP point")
print("3. Once the approximation is constructed, sampling is very efficient")
print("4. For large neural networks, computing and storing the Hessian can be challenging")
print("   (O(d²) memory where d is the number of parameters)")
print("5. Various approximations of the Hessian can be used to improve efficiency")
print("   (diagonal approximation, low-rank approximation, etc.)")

print("\nMCMC Sampling:")
print("1. Requires many iterations to explore the posterior")
print("2. Each iteration typically involves evaluating the model on the entire dataset")
print("3. Can struggle with high-dimensional posteriors (slow mixing)")
print("4. No need to compute or store the Hessian")
print("5. Can capture complex, multi-modal posteriors more accurately")
print("6. Implementations like Hamiltonian Monte Carlo (HMC) improve efficiency")
print("   but still typically much slower than Laplace for high-dimensional problems")

print("\nComparison Summary:")
print("- Laplace approximation is typically much faster for high-dimensional problems")
print("- MCMC provides a more accurate representation of the posterior, especially")
print("  when it's complex or multi-modal")
print("- Laplace works well when the posterior is approximately Gaussian")
print("- For neural networks with millions of parameters, Laplace often uses")
print("  additional approximations for tractability")
print("- MCMC is generally more computationally intensive but makes fewer assumptions")
print("  about the shape of the posterior")

# Create a visual comparison of computational characteristics
methods = ['Laplace Approximation', 'MCMC Sampling']
categories = ['Setup Time', 'Sampling Speed', 'Memory Requirements', 'Accuracy for Complex Posteriors', 'Scalability to High Dimensions']

# These are subjective ratings from 1-10 for illustration purposes
laplace_scores = [6, 9, 3, 5, 7]
mcmc_scores = [4, 3, 8, 9, 3]

# Create a grouped bar chart
x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 8))
ax.bar(x - width/2, laplace_scores, width, label='Laplace Approximation', color='royalblue')
ax.bar(x + width/2, mcmc_scores, width, label='MCMC Sampling', color='firebrick')

ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=15, ha='right')
ax.set_yticks(np.arange(0, 11, 2))
ax.set_ylabel('Score (Higher is Better)', fontsize=12)
ax.set_title('Computational Characteristics Comparison', fontsize=14)
ax.legend()

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "computational_comparison.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 6: Practical Implementation Example (Neural Network)
print_step_header(6, "Practical Implementation Example (Neural Network)")

print("To illustrate how Laplace approximation would be implemented for a neural network,")
print("let's consider a simple example with a small neural network.")
print("\nThe steps for implementation would look like:")
print("1. Define a neural network model")
print("2. Find the MAP estimate by training the network with regularization")
print("3. Compute the Hessian at the MAP point")
print("4. Use the Laplace approximation for prediction")

# Define a simple neural network architecture (pseudocode)
print("\nNeural Network Architecture (pseudocode):")
print("class SimpleNN:")
print("  def __init__(self, input_dim=1, hidden_dim=10, output_dim=1):")
print("    self.W1 = matrix(input_dim, hidden_dim)  # First layer weights")
print("    self.b1 = vector(hidden_dim)             # First layer bias")
print("    self.W2 = matrix(hidden_dim, output_dim) # Second layer weights")
print("    self.b2 = vector(output_dim)             # Second layer bias")
print("    ")
print("  def forward(self, x):")
print("    h = tanh(x @ W1 + b1)  # Hidden layer with tanh activation")
print("    y = h @ W2 + b2        # Output layer (linear)")
print("    return y")

print("\nFinding the MAP estimate by training the model...")
print("1. Define a loss function: negative log-posterior = negative log-likelihood + negative log-prior")
print("2. For regression: negative log-likelihood = mean squared error")
print("3. Common negative log-prior: L2 regularization (weight decay)")
print("4. Optimize using gradient descent or similar algorithm")
print("Model training completed. MAP estimate found.")

print("\nComputing Hessian at the MAP estimate...")
print("1. The Hessian is the matrix of second derivatives of the loss function")
print("2. Can be computed exactly for small networks or approximated for larger ones")
print("3. For neural networks, common approximations include:")
print("   - Diagonal approximation (ignore parameter correlations)")
print("   - Block-diagonal approximation (by layers)")
print("   - Kronecker-factored approximation")
print("   - Low-rank approximation")
print("Hessian computation or approximation completed.")

print("\nGenerating predictions with uncertainty using the Laplace approximation...")
print("1. Sample parameter vectors from the approximate posterior: N(w_MAP, H⁻¹)")
print("2. For each sample, perform a forward pass through the network")
print("3. Compute statistics (mean, variance) of the predictions")
print("Predictive distribution computed and visualized in the previous figure.")

# Create a flowchart-style visualization of the process
fig, ax = plt.subplots(figsize=(12, 10))
ax.axis('off')

# Nodes
nodes = [
    (0.5, 0.9, 'Neural Network\nwith Parameters w'),
    (0.5, 0.75, 'Find MAP Estimate\nw_MAP'),
    (0.5, 0.6, 'Compute Hessian H\nat w_MAP'),
    (0.5, 0.45, 'Laplace Approximation\np(w|D) ≈ N(w_MAP, H⁻¹)'),
    (0.25, 0.3, 'Sample Parameters\nw⁽ⁱ⁾ ~ N(w_MAP, H⁻¹)'),
    (0.75, 0.3, 'Compute Predictions\np(y_new|x_new, w⁽ⁱ⁾)'),
    (0.5, 0.15, 'Average Predictions\np(y_new|x_new, D) ≈ (1/M)∑ᵢp(y_new|x_new, w⁽ⁱ⁾)')
]

# Arrows
arrows = [
    (0.5, 0.87, 0.5, 0.78, 'Training with\nRegularization'),
    (0.5, 0.72, 0.5, 0.63, 'Second Derivatives\nof Loss Function'),
    (0.5, 0.57, 0.5, 0.48, 'Gaussian\nApproximation'),
    (0.45, 0.42, 0.3, 0.33, 'Monte Carlo\nSampling'),
    (0.55, 0.42, 0.7, 0.33, 'Forward Pass'),
    (0.3, 0.27, 0.45, 0.18, ''),
    (0.7, 0.27, 0.55, 0.18, '')
]

# Draw nodes
for x, y, text in nodes:
    ax.text(x, y, text, ha='center', va='center', bbox=dict(
        facecolor='lightblue', edgecolor='blue', alpha=0.7, boxstyle='round,pad=0.5'
    ), fontsize=10)

# Draw arrows
for x1, y1, x2, y2, text in arrows:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='black'))
    # Add arrow text
    if text:
        ax.text((x1+x2)/2, (y1+y2)/2, text, ha='center', va='center',
               bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7),
               fontsize=8)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('Laplace Approximation Process for Neural Networks', fontsize=14)

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "laplace_process_flowchart.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

print("\nConclusion:")
print("1. Laplace approximation provides a computationally efficient way to approximate")
print("   the posterior distribution of neural network parameters.")
print("2. It enables uncertainty quantification in predictions with relatively low")
print("   computational overhead compared to MCMC methods.")
print("3. The accuracy of the approximation depends on how well the posterior can be")
print("   approximated by a Gaussian distribution.")
print("4. For modern deep learning models with millions of parameters, further")
print("   approximations of the Hessian are typically needed for practicality.") 