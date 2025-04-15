import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

def normal_pdf(x, mu, sigma=2):
    """Probability density function for normal distribution with known variance"""
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-(x-mu)**2/(2*sigma**2))

def log_likelihood(mu, data, sigma=2):
    """Log-likelihood function for normal distribution"""
    n = len(data)
    return -n/2 * np.log(2*np.pi*sigma**2) - 1/(2*sigma**2) * np.sum((data - mu)**2)

def plot_normal_pdfs(mu_values, sigma=2, save_path=None):
    """Plot normal PDFs for different mu values"""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(-10, 10, 1000)
    
    for mu in mu_values:
        y = normal_pdf(x, mu, sigma)
        ax.plot(x, y, label=f'μ = {mu}')
    
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density f(x|μ)')
    ax.set_title('Normal Distribution PDFs for Different μ Values (σ² = 4)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_likelihood_surface(data, sigma=2, save_path=None):
    """Plot the likelihood function surface"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate MLE estimate (sample mean)
    mle_mu = np.mean(data)
    
    # Create range of possible mu values
    mu_range = np.linspace(mle_mu - 3, mle_mu + 3, 1000)
    log_likelihoods = [log_likelihood(mu, data, sigma) for mu in mu_range]
    
    ax.plot(mu_range, log_likelihoods, 'b-', linewidth=2)
    ax.axvline(x=mle_mu, color='r', linestyle='--', 
               label=f'MLE μ = {mle_mu:.4f}')
    
    ax.set_xlabel('μ')
    ax.set_ylabel('Log-Likelihood ℓ(μ)')
    ax.set_title('Log-Likelihood Function for Normal Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return mle_mu

def plot_mle_fit(data, sigma=2, save_path=None):
    """Plot the fitted MLE distribution against the data"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate MLE estimate
    mle_mu = np.mean(data)
    
    # Generate x values for plotting
    x = np.linspace(min(data)-3, max(data)+3, 1000)
    y_mle = normal_pdf(x, mle_mu, sigma)
    
    # Plot histogram of the data
    ax.hist(data, bins=min(15, len(data)), density=True, alpha=0.5, color='blue', 
             label='Observed Data')
    
    # Plot the fitted PDF based on MLE
    ax.plot(x, y_mle, 'r-', linewidth=2, 
            label=f'MLE Fit (μ = {mle_mu:.4f})')
    
    # Mark the observed data points
    ax.plot(data, np.zeros_like(data), 'bo', markersize=8, alpha=0.6)
    
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density')
    ax.set_title('Maximum Likelihood Estimation for Normal Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return mle_mu

def plot_unbiasedness_demonstration(save_path=None):
    """Demonstrate that the MLE estimator is unbiased"""
    np.random.seed(42)
    true_mu = 5
    sigma = 2
    n_samples = 1000
    sample_sizes = [10, 50, 100, 500]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for n in sample_sizes:
        mle_estimates = []
        for _ in range(n_samples):
            data = np.random.normal(true_mu, sigma, n)
            mle_estimates.append(np.mean(data))
        
        ax.hist(mle_estimates, bins=30, alpha=0.5, 
                label=f'n = {n}, Mean = {np.mean(mle_estimates):.4f}')
    
    ax.axvline(x=true_mu, color='r', linestyle='--', 
               label=f'True μ = {true_mu}')
    
    ax.set_xlabel('MLE Estimate of μ')
    ax.set_ylabel('Frequency')
    ax.set_title('Demonstration of MLE Unbiasedness for Different Sample Sizes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def main():
    """Generate all visualizations for Question 2 of the L2.4 quiz"""
    # Create synthetic data
    np.random.seed(42)
    true_mu = 5
    sigma = 2
    n = 100
    data = np.random.normal(true_mu, sigma, n)
    
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_4_Quiz_2")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 2 of the L2.4 MLE quiz...")
    
    # 1. Plot PDFs for different mu values
    plot_normal_pdfs([3, 5, 7], save_path=os.path.join(save_dir, "normal_pdfs.png"))
    print("1. PDF visualization created")
    
    # 2. Plot likelihood surface
    mle_mu = plot_likelihood_surface(data, save_path=os.path.join(save_dir, "likelihood_surface.png"))
    print(f"2. Likelihood surface visualization created, MLE μ = {mle_mu:.4f}")
    
    # 3. Plot MLE fit to data
    plot_mle_fit(data, save_path=os.path.join(save_dir, "mle_fit.png"))
    print("3. MLE fit visualization created")
    
    # 4. Demonstrate unbiasedness
    plot_unbiasedness_demonstration(save_path=os.path.join(save_dir, "unbiasedness.png"))
    print("4. Unbiasedness demonstration created")
    
    print(f"\nAll visualizations have been saved to: {save_dir}")
    print(f"For the synthetic data used, the MLE estimate is μ = {mle_mu:.4f}")

if __name__ == "__main__":
    main() 