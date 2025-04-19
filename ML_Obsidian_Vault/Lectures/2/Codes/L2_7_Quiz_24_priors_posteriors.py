import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as stats
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_7_Quiz_24")
os.makedirs(save_dir, exist_ok=True)

# Set LaTeX style for plots
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

# Define parameter space
theta = np.linspace(-3, 13, 1000)

# Define different priors (to demonstrate how prior strength affects MAP)
def prior_1(theta):
    """Weak prior centered at theta=1"""
    return stats.norm.pdf(theta, loc=1, scale=3.0)

def prior_2(theta):
    """Medium prior centered at theta=1"""
    return stats.norm.pdf(theta, loc=1, scale=1.5)

def prior_3(theta):
    """Strong prior centered at theta=1"""
    return stats.norm.pdf(theta, loc=1, scale=0.8)

# Define likelihood based on 5 observed data points
def likelihood(theta):
    """Likelihood assuming 5 observed data points with sample mean of 7"""
    # This simulates having 5 data points from N(theta, 2²)
    sample_mean = 7
    n = 5
    sample_var = 4  # sigma² = 4, so sigma = 2
    return stats.norm.pdf(sample_mean, loc=theta, scale=np.sqrt(sample_var/n))

# Define unnormalized posteriors
def posterior_1(theta):
    return likelihood(theta) * prior_1(theta)

def posterior_2(theta):
    return likelihood(theta) * prior_2(theta)

def posterior_3(theta):
    return likelihood(theta) * prior_3(theta)

# Create the plots
def save_plot(func, title, filename, xlim=(-3, 13), ylim=None, vertical_lines=None):
    """Create and save a plot for the given function"""
    plt.figure(figsize=(8, 6))
    y_values = func(theta)
    
    # Normalize for better visualization
    y_values = y_values / np.max(y_values)
    
    plt.plot(theta, y_values)
    
    if vertical_lines:
        for x_value, label, color in vertical_lines:
            plt.axvline(x=x_value, color=color, linestyle='--', 
                        label=f'{label}: $\\theta = {x_value:.2f}$')
    
    plt.xlabel(r'$\theta$')
    plt.ylabel('Normalized Density')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.xlim(xlim)
    
    if ylim:
        plt.ylim(ylim)
        
    if vertical_lines:
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# MLE (maximum of likelihood)
mle_value = 7.0  # This is the sample mean

# Find MAP estimates
map_values = []
for posterior_func in [posterior_1, posterior_2, posterior_3]:
    map_index = np.argmax(posterior_func(theta))
    map_values.append(theta[map_index])

# Plot the likelihood (represents MLE)
save_plot(
    likelihood,
    r'Likelihood Function $p(D|\theta)$ - 5 Data Points with $\bar{x}=7$',
    'graph1_likelihood.png',
    vertical_lines=[(mle_value, 'MLE', 'red')]
)

# Plot the priors
save_plot(
    prior_1,
    r'Prior 1: Weak Prior $p(\theta)$ - $\mathcal{N}(1, 3^2)$',
    'graph2_weak_prior.png'
)

save_plot(
    prior_2,
    r'Prior 2: Medium Prior $p(\theta)$ - $\mathcal{N}(1, 1.5^2)$',
    'graph3_medium_prior.png'
)

save_plot(
    prior_3,
    r'Prior 3: Strong Prior $p(\theta)$ - $\mathcal{N}(1, 0.8^2)$',
    'graph4_strong_prior.png'
)

# Plot the posteriors with MAP estimates
save_plot(
    posterior_1,
    r'Posterior 1: $p(\theta|D)$ with Weak Prior',
    'graph5_posterior_weak.png',
    vertical_lines=[(map_values[0], 'MAP', 'blue'), (mle_value, 'MLE', 'red')]
)

save_plot(
    posterior_2,
    r'Posterior 2: $p(\theta|D)$ with Medium Prior',
    'graph6_posterior_medium.png',
    vertical_lines=[(map_values[1], 'MAP', 'blue'), (mle_value, 'MLE', 'red')]
)

save_plot(
    posterior_3,
    r'Posterior 3: $p(\theta|D)$ with Strong Prior',
    'graph7_posterior_strong.png',
    vertical_lines=[(map_values[2], 'MAP', 'blue'), (mle_value, 'MLE', 'red')]
)

# Create a comparison plot of all posteriors
plt.figure(figsize=(10, 6))
plt.plot(theta, posterior_1(theta)/np.max(posterior_1(theta)), label='Posterior with Weak Prior')
plt.plot(theta, posterior_2(theta)/np.max(posterior_2(theta)), label='Posterior with Medium Prior')
plt.plot(theta, posterior_3(theta)/np.max(posterior_3(theta)), label='Posterior with Strong Prior')

# Add vertical lines for MAP estimates
plt.axvline(x=map_values[0], color='blue', linestyle='--', 
            label=f'MAP (Weak): $\\theta = {map_values[0]:.2f}$')
plt.axvline(x=map_values[1], color='green', linestyle='--', 
            label=f'MAP (Medium): $\\theta = {map_values[1]:.2f}$')
plt.axvline(x=map_values[2], color='purple', linestyle='--', 
            label=f'MAP (Strong): $\\theta = {map_values[2]:.2f}$')
plt.axvline(x=mle_value, color='red', linestyle='--', 
            label=f'MLE: $\\theta = {mle_value:.2f}$')

plt.xlabel(r'$\theta$')
plt.ylabel('Normalized Density')
plt.title('Comparison of Posteriors with Different Prior Strengths')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-3, 13)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'graph8_posterior_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"All visualizations saved in '{save_dir}'")
print(f"MLE value: {mle_value}")
print(f"MAP values:")
print(f"  - Weak prior: {map_values[0]:.4f}")
print(f"  - Medium prior: {map_values[1]:.4f}")
print(f"  - Strong prior: {map_values[2]:.4f}") 