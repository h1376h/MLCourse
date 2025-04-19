import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, norm
import os
import matplotlib as mpl

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_5_Quiz_14")
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

# Define a prior distribution (Beta distribution for parameter theta)
def plot_prior():
    """Plot the prior distribution (Beta)"""
    x = np.linspace(0, 1, 1000)
    
    # Three different priors for comparison
    prior1 = beta(2, 5)  # Skewed toward lower values
    prior2 = beta(5, 2)  # Skewed toward higher values
    prior3 = beta(3, 3)  # Symmetric around 0.5
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, prior1.pdf(x), 'r-', label=r'Prior 1: Beta(2, 5)')
    plt.plot(x, prior2.pdf(x), 'g-', label=r'Prior 2: Beta(5, 2)')
    plt.plot(x, prior3.pdf(x), 'b-', label=r'Prior 3: Beta(3, 3)')
    
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'Density')
    plt.title(r'Prior Distributions')
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.savefig(os.path.join(save_dir, "prior_distributions.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Define likelihood function (Binomial for coin flips)
def plot_likelihood():
    """Plot likelihood function for different observed data"""
    x = np.linspace(0, 1, 1000)
    
    # Calculate likelihood for different observations (heads, total)
    def likelihood(theta, h, n):
        # p(D|θ) ∝ θ^h * (1-θ)^(n-h)
        return theta**h * (1-theta)**(n-h)
    
    # Three different datasets
    h1, n1 = 3, 10  # 3 heads out of 10 flips
    h2, n2 = 6, 10  # 6 heads out of 10 flips
    h3, n3 = 9, 10  # 9 heads out of 10 flips
    
    l1 = likelihood(x, h1, n1)
    l2 = likelihood(x, h2, n2)
    l3 = likelihood(x, h3, n3)
    
    # Normalize for better visualization
    l1 = l1 / np.max(l1) * 3
    l2 = l2 / np.max(l2) * 3
    l3 = l3 / np.max(l3) * 3
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, l1, 'r--', label=r'Likelihood: 3 heads out of 10')
    plt.plot(x, l2, 'g--', label=r'Likelihood: 6 heads out of 10')
    plt.plot(x, l3, 'b--', label=r'Likelihood: 9 heads out of 10')
    
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'Likelihood (normalized)')
    plt.title(r'Likelihood Functions for Different Observations')
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.savefig(os.path.join(save_dir, "likelihood_functions.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Calculate and plot posterior distributions
def plot_posterior():
    """Plot posterior distributions for different prior and likelihood combinations"""
    x = np.linspace(0, 1, 1000)
    
    # Prior parameters
    prior_params = [(2, 5), (5, 2), (3, 3)]
    colors = ['r', 'g', 'b']
    
    # Observation data (6 heads out of 10)
    h, n = 6, 10
    
    plt.figure(figsize=(10, 6))
    
    for i, (a, b) in enumerate(prior_params):
        # Calculate posterior parameters
        a_post = a + h
        b_post = b + (n - h)
        
        # Create posterior distribution
        posterior = beta(a_post, b_post)
        
        # Plot
        plt.plot(x, posterior.pdf(x), f'{colors[i]}-', 
                 label=f'Posterior with Beta({a}, {b}) prior')
        
        # Mark the posterior mean
        mean = a_post / (a_post + b_post)
        plt.axvline(x=mean, color=colors[i], linestyle='--', alpha=0.5)
        
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'Posterior Density')
    plt.title(r'Posterior Distributions with Different Priors (Data: 6 heads out of 10)')
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.savefig(os.path.join(save_dir, "posterior_distributions.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Plot the credible intervals
def plot_credible_intervals():
    """Plot 90% credible intervals for different posteriors"""
    x = np.linspace(0, 1, 1000)
    
    # Prior parameters
    prior_params = [(2, 5), (5, 2), (3, 3)]
    labels = ['Prior 1: Beta(2, 5)', 'Prior 2: Beta(5, 2)', 'Prior 3: Beta(3, 3)']
    colors = ['r', 'g', 'b']
    
    # Observation data (6 heads out of 10)
    h, n = 6, 10
    
    plt.figure(figsize=(10, 6))
    
    for i, (a, b) in enumerate(prior_params):
        # Calculate posterior parameters
        a_post = a + h
        b_post = b + (n - h)
        
        # Create posterior distribution
        posterior = beta(a_post, b_post)
        
        # Calculate 90% credible interval
        lower = posterior.ppf(0.05)
        upper = posterior.ppf(0.95)
        
        # Plot PDF
        plt.plot(x, posterior.pdf(x), f'{colors[i]}-', label=f'Posterior with {labels[i]}')
        
        # Shade the credible interval
        idx = (x >= lower) & (x <= upper)
        plt.fill_between(x[idx], 0, posterior.pdf(x)[idx], color=colors[i], alpha=0.3)
        
        # Add vertical lines for interval bounds
        plt.axvline(x=lower, color=colors[i], linestyle='--', alpha=0.7)
        plt.axvline(x=upper, color=colors[i], linestyle='--', alpha=0.7)
        
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'Posterior Density')
    plt.title(r'90\% Credible Intervals for Different Posteriors')
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.savefig(os.path.join(save_dir, "credible_intervals.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Create a figure showing the Bayesian updating process
def plot_bayesian_updating():
    """Show the process of Bayesian updating with more data"""
    x = np.linspace(0, 1, 1000)
    
    # Start with Beta(2, 2) prior
    a_prior, b_prior = 2, 2
    
    # Data sequence: increasing evidence of biased coin
    data_sequence = [(1, 1), (2, 3), (5, 6), (8, 9), (12, 12)]  # (heads, total flips)
    colors = ['k', 'm', 'c', 'g', 'r']  # Using basic matplotlib color codes
    
    plt.figure(figsize=(10, 6))
    
    # Plot the prior
    prior = beta(a_prior, b_prior)
    plt.plot(x, prior.pdf(x), f'{colors[0]}-', 
             label=f'Prior: Beta({a_prior}, {b_prior})')
    
    # Plot posteriors after each data point
    for i, (h, n) in enumerate(data_sequence[1:], 1):
        # Calculate cumulative heads and flips
        cum_h = data_sequence[i][0]
        cum_n = data_sequence[i][1]
        
        # Calculate posterior parameters
        a_post = a_prior + cum_h
        b_post = b_prior + (cum_n - cum_h)
        
        # Create posterior distribution
        posterior = beta(a_post, b_post)
        
        # Plot posterior
        plt.plot(x, posterior.pdf(x), f'{colors[i]}-', 
                 label=f'After {cum_n} flips ({cum_h} heads)')
    
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'Density')
    plt.title(r'Bayesian Updating: Evolution of Belief with More Data')
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.savefig(os.path.join(save_dir, "bayesian_updating.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Generate all plots
plot_prior()
plot_likelihood()
plot_posterior() 
plot_credible_intervals()
plot_bayesian_updating()

print(f"All visualizations saved in '{save_dir}'") 