import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, norm
import os
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_7_Quiz_25")
os.makedirs(save_dir, exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath,amssymb}',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step a: Generate true posterior distributions for Bernoulli parameter with different priors
print_step_header(1, "Generating Posteriors for Bernoulli with Different Priors")

# Define scenario
# We have a coin with unknown bias parameter θ (probability of heads)
# We observe 60 heads in 100 flips

# Data
heads = 60
n = 100
ML_estimate = heads / n  # MLE = 0.6

# Define the Beta priors
x = np.linspace(0.001, 0.999, 1000)

# Prior 1: Beta(1,1) - Uniform prior
alpha1, beta1 = 1, 1
prior1 = beta.pdf(x, alpha1, beta1)
posterior1 = beta.pdf(x, alpha1 + heads, beta1 + n - heads)
MAP1 = (alpha1 + heads - 1) / (alpha1 + beta1 + n - 2)  # Mode of Beta

# Prior 2: Beta(3,7) - Biased toward tails
alpha2, beta2 = 3, 7
prior2 = beta.pdf(x, alpha2, beta2)
posterior2 = beta.pdf(x, alpha2 + heads, beta2 + n - heads)
MAP2 = (alpha2 + heads - 1) / (alpha2 + beta2 + n - 2)

# Prior 3: Beta(7,3) - Biased toward heads
alpha3, beta3 = 7, 3
prior3 = beta.pdf(x, alpha3, beta3)
posterior3 = beta.pdf(x, alpha3 + heads, beta3 + n - heads)
MAP3 = (alpha3 + heads - 1) / (alpha3 + beta3 + n - 2)

# Prior 4: Beta(50,50) - Strong belief in fair coin
alpha4, beta4 = 50, 50
prior4 = beta.pdf(x, alpha4, beta4)
posterior4 = beta.pdf(x, alpha4 + heads, beta4 + n - heads)
MAP4 = (alpha4 + heads - 1) / (alpha4 + beta4 + n - 2)

# Likelihood (not a proper PDF, but scaled for visualization)
# Bernoulli likelihood for θ given the data
def likelihood(theta):
    return theta**heads * (1-theta)**(n-heads)

# Normalize likelihood for visualization
likelihood_values = likelihood(x)
likelihood_scaled = likelihood_values / np.max(likelihood_values) * np.max(posterior1)

print(f"MLE: {ML_estimate:.4f}")
print(f"MAP with Uniform Prior: {MAP1:.4f}")
print(f"MAP with Beta(3,7) Prior: {MAP2:.4f}")
print(f"MAP with Beta(7,3) Prior: {MAP3:.4f}")
print(f"MAP with Beta(50,50) Prior: {MAP4:.4f}")

# Plot Likelihood
plt.figure(figsize=(10, 6))
plt.plot(x, likelihood_scaled, 'k-', linewidth=2)
plt.axvline(x=ML_estimate, color='red', linestyle='--', linewidth=2)
plt.xlabel(r'$\theta$ (Probability of Heads)')
plt.ylabel('Scaled Likelihood')
plt.title('Likelihood Function for Coin Flip Data (60 Heads in 100 Flips)')
plt.xlim(0, 1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'likelihood.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot each prior
plt.figure(figsize=(10, 6))
plt.plot(x, prior1, 'b-', linewidth=2, label=r'Uniform Prior: Beta(1,1)')
plt.xlabel(r'$\theta$ (Probability of Heads)')
plt.ylabel('Probability Density')
plt.title('Uniform Prior: Beta(1,1)')
plt.xlim(0, 1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'prior_uniform.png'), dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(x, prior2, 'g-', linewidth=2, label=r'Biased toward Tails: Beta(3,7)')
plt.xlabel(r'$\theta$ (Probability of Heads)')
plt.ylabel('Probability Density')
plt.title('Prior Biased toward Tails: Beta(3,7)')
plt.xlim(0, 1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'prior_tails.png'), dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(x, prior3, 'r-', linewidth=2, label=r'Biased toward Heads: Beta(7,3)')
plt.xlabel(r'$\theta$ (Probability of Heads)')
plt.ylabel('Probability Density')
plt.title('Prior Biased toward Heads: Beta(7,3)')
plt.xlim(0, 1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'prior_heads.png'), dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(x, prior4, 'm-', linewidth=2, label=r'Strong Belief in Fair Coin: Beta(50,50)')
plt.xlabel(r'$\theta$ (Probability of Heads)')
plt.ylabel('Probability Density')
plt.title('Prior with Strong Belief in Fair Coin: Beta(50,50)')
plt.xlim(0, 1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'prior_fair.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot each posterior
plt.figure(figsize=(10, 6))
plt.plot(x, posterior1, 'b-', linewidth=2)
plt.axvline(x=MAP1, color='blue', linestyle='--', linewidth=2, 
            label=f'MAP = {MAP1:.3f}')
plt.axvline(x=ML_estimate, color='red', linestyle='--', linewidth=2, 
            label=f'MLE = {ML_estimate:.3f}')
plt.xlabel(r'$\theta$ (Probability of Heads)')
plt.ylabel('Posterior Probability Density')
plt.title('Posterior with Uniform Prior: Beta(1+60, 1+40)')
plt.legend()
plt.xlim(0, 1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'posterior_uniform.png'), dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(x, posterior2, 'g-', linewidth=2)
plt.axvline(x=MAP2, color='green', linestyle='--', linewidth=2, 
            label=f'MAP = {MAP2:.3f}')
plt.axvline(x=ML_estimate, color='red', linestyle='--', linewidth=2, 
            label=f'MLE = {ML_estimate:.3f}')
plt.xlabel(r'$\theta$ (Probability of Heads)')
plt.ylabel('Posterior Probability Density')
plt.title('Posterior with Tails-Biased Prior: Beta(3+60, 7+40)')
plt.legend()
plt.xlim(0, 1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'posterior_tails.png'), dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(x, posterior3, 'r-', linewidth=2)
plt.axvline(x=MAP3, color='red', linestyle='--', linewidth=2, 
            label=f'MAP = {MAP3:.3f}')
plt.axvline(x=ML_estimate, color='red', linestyle=':', linewidth=2, 
            label=f'MLE = {ML_estimate:.3f}')
plt.xlabel(r'$\theta$ (Probability of Heads)')
plt.ylabel('Posterior Probability Density')
plt.title('Posterior with Heads-Biased Prior: Beta(7+60, 3+40)')
plt.legend()
plt.xlim(0, 1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'posterior_heads.png'), dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(x, posterior4, 'm-', linewidth=2)
plt.axvline(x=MAP4, color='magenta', linestyle='--', linewidth=2, 
            label=f'MAP = {MAP4:.3f}')
plt.axvline(x=ML_estimate, color='red', linestyle='--', linewidth=2, 
            label=f'MLE = {ML_estimate:.3f}')
plt.xlabel(r'$\theta$ (Probability of Heads)')
plt.ylabel('Posterior Probability Density')
plt.title('Posterior with Fair Coin Prior: Beta(50+60, 50+40)')
plt.legend()
plt.xlim(0, 1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'posterior_fair.png'), dpi=300, bbox_inches='tight')
plt.close()

# Step b: Create comparison plot showing all posteriors together with MLE and MAP values
plt.figure(figsize=(12, 7))
plt.plot(x, posterior1, 'b-', linewidth=2, label='Posterior with Uniform Prior')
plt.plot(x, posterior2, 'g-', linewidth=2, label='Posterior with Tails-Biased Prior')
plt.plot(x, posterior3, 'r-', linewidth=2, label='Posterior with Heads-Biased Prior')
plt.plot(x, posterior4, 'm-', linewidth=2, label='Posterior with Fair Coin Prior')
plt.axvline(x=ML_estimate, color='k', linestyle='-', linewidth=2, 
            label=f'MLE = {ML_estimate:.3f}')
plt.axvline(x=MAP1, color='b', linestyle='--', linewidth=1.5)
plt.axvline(x=MAP2, color='g', linestyle='--', linewidth=1.5)
plt.axvline(x=MAP3, color='r', linestyle='--', linewidth=1.5)
plt.axvline(x=MAP4, color='m', linestyle='--', linewidth=1.5)

plt.xlabel(r'$\theta$ (Probability of Heads)')
plt.ylabel('Posterior Probability Density')
plt.title('Comparison of All Posterior Distributions')
plt.legend(loc='best')
plt.xlim(0.3, 0.8)  # Zoomed in to highlight differences
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'posterior_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# Step c: Create a progression showing how posteriors change with increasing data
print_step_header(2, "Showing Posterior Evolution with Increasing Data")

# We'll show how the Beta(3,7) prior evolves as we observe more data
# Maintaining the same 60% heads ratio
data_proportions = [(3, 2), (6, 4), (15, 10), (30, 20), (60, 40), (150, 100), (300, 200)]

plt.figure(figsize=(12, 8))
plt.plot(x, prior2, 'k--', linewidth=2, label='Prior: Beta(3,7)')

colors = plt.cm.viridis(np.linspace(0, 1, len(data_proportions)))

for i, (h, t) in enumerate(data_proportions):
    total = h + t
    post = beta.pdf(x, alpha2 + h, beta2 + t)
    map_est = (alpha2 + h - 1) / (alpha2 + beta2 + total - 2)
    mle = h / total
    
    plt.plot(x, post, color=colors[i], linewidth=2, 
             label=f'{h} heads, {t} tails (MAP={map_est:.3f}, MLE={mle:.3f})')
    plt.axvline(x=map_est, color=colors[i], linestyle='--', linewidth=1)

plt.axvline(x=ML_estimate, color='red', linestyle='-', linewidth=2, 
            label=f'Asymptotic MLE = {ML_estimate:.3f}')

plt.xlabel(r'$\theta$ (Probability of Heads)')
plt.ylabel('Probability Density')
plt.title('Evolution of Posterior with Increasing Data (From Prior Beta(3,7))')
plt.legend(loc='upper left', fontsize=10)
plt.xlim(0, 1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'posterior_evolution.png'), dpi=300, bbox_inches='tight')
plt.close()

# Step d: Create visualization showing how choice of loss function affects Bayesian estimation
print_step_header(3, "Visualizing Different Bayesian Estimators with Loss Functions")

# Using posterior1 (uniform prior + data) to demonstrate different estimators
post_mean = beta.mean(alpha1 + heads, beta1 + n - heads)  # Mean of Beta
post_median = beta.median(alpha1 + heads, beta1 + n - heads)  # Median of Beta
# MAP1 already calculated above

plt.figure(figsize=(10, 6))
plt.plot(x, posterior1, 'b-', linewidth=2, label='Posterior Distribution')
plt.axvline(x=MAP1, color='red', linestyle='--', linewidth=2, 
            label=f'MAP (0-1 Loss): $\\theta$ = {MAP1:.3f}')
plt.axvline(x=post_mean, color='green', linestyle='--', linewidth=2, 
            label=f'Mean (Squared Loss): $\\theta$ = {post_mean:.3f}')
plt.axvline(x=post_median, color='purple', linestyle='--', linewidth=2, 
            label=f'Median (Absolute Loss): $\\theta$ = {post_median:.3f}')

plt.xlabel(r'$\theta$ (Probability of Heads)')
plt.ylabel('Posterior Probability Density')
plt.title('Different Bayesian Estimators Based on Loss Functions')
plt.legend(loc='best')
plt.xlim(0.45, 0.75)  # Zoomed in to highlight differences
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'bayesian_estimators.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"All visualizations saved in '{save_dir}'")

print("\nSummary of MAP Estimates:")
print(f"MLE: {ML_estimate:.4f}")
print(f"MAP with Uniform Prior: {MAP1:.4f}")
print(f"MAP with Beta(3,7) Prior: {MAP2:.4f}")
print(f"MAP with Beta(7,3) Prior: {MAP3:.4f}")
print(f"MAP with Beta(50,50) Prior: {MAP4:.4f}")

print("\nBayesian Estimators:")
print(f"MAP (0-1 Loss): {MAP1:.4f}")
print(f"Mean (Squared Loss): {post_mean:.4f}")
print(f"Median (Absolute Loss): {post_median:.4f}") 