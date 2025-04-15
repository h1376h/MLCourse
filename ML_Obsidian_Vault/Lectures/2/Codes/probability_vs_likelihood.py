import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Example 1: Side-by-side comparison of probability vs likelihood
plt.figure(figsize=(14, 6))

# Left subplot for probability (fixed parameter, variable data)
plt.subplot(1, 2, 1)
x = np.arange(0, 11)  # Possible outcomes (0 to 10 heads)
theta = 0.6  # Fixed parameter (probability of heads)
probs = [stats.binom.pmf(k, 10, theta) for k in x]

plt.bar(x, probs, color='skyblue', alpha=0.7)
plt.axvline(x=6, color='r', linestyle='--', alpha=0.7)
plt.grid(True, alpha=0.3)
plt.xlabel('Number of Heads (X)', fontsize=12)
plt.ylabel('Probability P(X|θ=0.6)', fontsize=12)
plt.title('Probability: Fixed θ=0.6, Variable X', fontsize=14)
plt.xticks(x)
plt.ylim(0, 0.3)

# Annotation for probability
plt.annotate('For θ=0.6:\nP(X=6|θ=0.6) ≈ 0.25', 
            xy=(6, stats.binom.pmf(6, 10, 0.6)),
            xytext=(7, 0.25),
            fontsize=10,
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

# Right subplot for likelihood (fixed data, variable parameter)
plt.subplot(1, 2, 2)
theta_values = np.linspace(0, 1, 100)
observed_heads = 6  # Fixed observation
likelihoods = [stats.binom.pmf(observed_heads, 10, t) for t in theta_values]

plt.plot(theta_values, likelihoods, 'b-', linewidth=2)
plt.axvline(x=0.6, color='r', linestyle='--', alpha=0.7)
plt.fill_between(theta_values, likelihoods, alpha=0.2)
plt.grid(True, alpha=0.3)
plt.xlabel('Parameter θ (Probability of Heads)', fontsize=12)
plt.ylabel('Likelihood L(θ|X=6)', fontsize=12)
plt.title('Likelihood: Fixed X=6, Variable θ', fontsize=14)
plt.ylim(0, 0.3)

# Annotation for likelihood
plt.annotate('For X=6:\nL(θ=0.6|X=6) ≈ 0.25\nMaximum at θ=0.6', 
            xy=(0.6, stats.binom.pmf(6, 10, 0.6)),
            xytext=(0.3, 0.25),
            fontsize=10,
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'probability_vs_likelihood.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: 3D visualization showing how both perspectives relate
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Create a grid of values
X, Y = np.meshgrid(np.arange(0, 11), np.linspace(0.1, 0.9, 9))
Z = np.zeros_like(X, dtype=float)

# Calculate binomial PMF for each point
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        k = X[i, j]  # Number of successes
        p = Y[i, j]  # Probability parameter
        Z[i, j] = stats.binom.pmf(k, 10, p)

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8, 
                       linewidth=0, antialiased=True)

# Add color bar
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
cbar.set_label('Probability/Likelihood Value')

# Highlight probability slice (fixed θ=0.6)
theta_fixed = 0.6
idx = np.abs(Y[:, 0] - theta_fixed).argmin()
ax.plot(np.arange(0, 11), [theta_fixed] * 11, Z[idx, :], 
        color='r', linewidth=3, label='Probability: P(X|θ=0.6)')

# Highlight likelihood slice (fixed X=6)
x_fixed = 6
ax.plot([x_fixed] * 9, Y[:, 0], Z[:, x_fixed], 
        color='blue', linewidth=3, label='Likelihood: L(θ|X=6)')

ax.set_xlabel('Number of Heads (X)', fontsize=12)
ax.set_ylabel('Parameter θ', fontsize=12)
ax.set_zlabel('Probability/Likelihood', fontsize=12)
ax.set_title('3D View: Probability vs Likelihood for Binomial Distribution', fontsize=14)
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'probability_vs_likelihood_3d.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Likelihood ratio and decision boundaries
plt.figure(figsize=(10, 6))

# Define two competing models/distributions
x = np.linspace(-5, 15, 1000)
mu1, sigma1 = 3, 2  # Model 1 parameters
mu2, sigma2 = 7, 2  # Model 2 parameters

# Compute probability densities
pdf1 = stats.norm.pdf(x, mu1, sigma1)
pdf2 = stats.norm.pdf(x, mu2, sigma2)

# Compute likelihood ratio
likelihood_ratio = pdf1 / pdf2

# Find decision boundary where likelihood ratio = 1
decision_point = x[np.argmin(np.abs(likelihood_ratio - 1))]

# Plot probability densities
plt.plot(x, pdf1, 'b-', linewidth=2, label=f'Model 1: N({mu1}, {sigma1}²)')
plt.plot(x, pdf2, 'r-', linewidth=2, label=f'Model 2: N({mu2}, {sigma2}²)')

# Fill areas to show decision regions
region1 = x <= decision_point
region2 = x >= decision_point
plt.fill_between(x[region1], pdf1[region1], alpha=0.2, color='blue', 
                label='Choose Model 1')
plt.fill_between(x[region2], pdf2[region2], alpha=0.2, color='red', 
                label='Choose Model 2')

# Mark decision boundary
plt.axvline(x=decision_point, color='k', linestyle='--', alpha=0.7, 
           label=f'Decision Boundary: x={decision_point:.2f}')

plt.grid(True, alpha=0.3)
plt.xlabel('Observation (x)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title('Likelihood Ratio for Decision Making', fontsize=14)
plt.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'likelihood_ratio_decision.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 4: Bayesian perspective
plt.figure(figsize=(10, 6))

# Define parameter space
theta_values = np.linspace(0, 1, 100)

# Example data: 8 heads in a 10-coin trial
x = 8
n = 10

# Likelihood function
likelihood = [stats.binom.pmf(x, n, theta) for theta in theta_values]

# Prior options
uniform_prior = np.ones_like(theta_values)  # Uniform prior
beta_prior = [stats.beta.pdf(theta, 2, 2) for theta in theta_values]  # Beta(2,2) prior

# Compute posteriors (ignoring normalization constant)
uniform_posterior = np.array(likelihood) * uniform_prior
beta_posterior = np.array(likelihood) * np.array(beta_prior)

# Normalize posteriors
uniform_posterior /= np.trapz(uniform_posterior, theta_values)
beta_posterior /= np.trapz(beta_posterior, theta_values)

# Plot
plt.plot(theta_values, likelihood / np.max(likelihood), 'b--', linewidth=2, 
        label='Likelihood L(θ|X=8)')
plt.plot(theta_values, beta_prior / np.max(beta_prior), 'g--', linewidth=2, 
        label='Prior P(θ) ~ Beta(2,2)')
plt.plot(theta_values, beta_posterior, 'r-', linewidth=2, 
        label='Posterior P(θ|X=8)')

# Mark maximum likelihood and MAP estimates
mle = theta_values[np.argmax(likelihood)]
map_estimate = theta_values[np.argmax(beta_posterior)]

plt.axvline(x=mle, color='b', linestyle=':', alpha=0.7, 
           label=f'MLE: θ={mle:.2f}')
plt.axvline(x=map_estimate, color='r', linestyle=':', alpha=0.7, 
           label=f'MAP: θ={map_estimate:.2f}')

plt.grid(True, alpha=0.3)
plt.xlabel('Parameter θ', fontsize=12)
plt.ylabel('Relative Density (scaled)', fontsize=12)
plt.title('Likelihood, Prior and Posterior', fontsize=14)
plt.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'bayesian_perspective.png'), dpi=100, bbox_inches='tight')
plt.close()

print("Probability vs Likelihood concept images created successfully.") 