import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import bernoulli
from scipy.optimize import minimize_scalar

def bernoulli_pmf(x, p):
    """Compute Bernoulli PMF for given x and p."""
    return p**x * (1-p)**(1-x)

def log_likelihood(p, data):
    """Compute log-likelihood for Bernoulli distribution."""
    n = len(data)
    sum_x = np.sum(data)
    return sum_x * np.log(p) + (n - sum_x) * np.log(1-p)

def fisher_information(p, n):
    """Compute Fisher information for Bernoulli distribution."""
    return n / (p * (1-p))

def asymptotic_variance(p, n):
    """Compute asymptotic variance of MLE."""
    return p * (1-p) / n

def plot_bernoulli_pmfs():
    """Plot PMFs for different values of p."""
    p_values = [0.3, 0.5, 0.7]
    x = np.array([0, 1])
    
    plt.figure(figsize=(10, 6))
    for p in p_values:
        pmf = bernoulli_pmf(x, p)
        plt.bar(x, pmf, alpha=0.6, label=f'p = {p}')
    
    plt.xlabel('x')
    plt.ylabel('Probability')
    plt.title('Bernoulli PMFs for Different Values of p')
    plt.xticks(x)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../Images/L2_4_Quiz_3/bernoulli_pmfs.png')
    plt.close()

def plot_likelihood_surface():
    """Plot likelihood surface for different values of p."""
    # Generate sample data
    np.random.seed(42)
    true_p = 0.6
    n = 100
    data = bernoulli.rvs(true_p, size=n)
    
    # Compute log-likelihood for different p values
    p_values = np.linspace(0.01, 0.99, 100)
    log_likelihoods = [log_likelihood(p, data) for p in p_values]
    
    # Find MLE
    mle = np.mean(data)
    
    plt.figure(figsize=(10, 6))
    plt.plot(p_values, log_likelihoods, 'b-', label='Log-likelihood')
    plt.axvline(x=mle, color='r', linestyle='--', label=f'MLE = {mle:.3f}')
    plt.axvline(x=true_p, color='g', linestyle='--', label=f'True p = {true_p}')
    
    plt.xlabel('p')
    plt.ylabel('Log-likelihood')
    plt.title('Log-likelihood Function for Bernoulli Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../Images/L2_4_Quiz_3/likelihood_surface.png')
    plt.close()

def plot_fisher_information():
    """Plot Fisher information as a function of p."""
    n = 100
    p_values = np.linspace(0.01, 0.99, 100)
    fisher_info = [fisher_information(p, n) for p in p_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(p_values, fisher_info, 'b-')
    plt.xlabel('p')
    plt.ylabel('Fisher Information')
    plt.title('Fisher Information for Bernoulli Distribution')
    plt.grid(True, alpha=0.3)
    plt.savefig('../Images/L2_4_Quiz_3/fisher_information.png')
    plt.close()

def plot_asymptotic_variance():
    """Plot asymptotic variance as a function of p."""
    n = 100
    p_values = np.linspace(0.01, 0.99, 100)
    var_values = [asymptotic_variance(p, n) for p in p_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(p_values, var_values, 'b-')
    plt.xlabel('p')
    plt.ylabel('Asymptotic Variance')
    plt.title('Asymptotic Variance of MLE for Bernoulli Distribution')
    plt.grid(True, alpha=0.3)
    plt.savefig('../Images/L2_4_Quiz_3/asymptotic_variance.png')
    plt.close()

def main():
    # Create directory for images if it doesn't exist
    os.makedirs('../Images/L2_4_Quiz_3', exist_ok=True)
    
    # Generate all plots
    plot_bernoulli_pmfs()
    plot_likelihood_surface()
    plot_fisher_information()
    plot_asymptotic_variance()
    
    print("All visualizations have been generated and saved.")

if __name__ == "__main__":
    main() 