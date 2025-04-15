import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm
from scipy.special import factorial
import os

def poisson_pmf(x, lambda_param):
    """Probability mass function for Poisson distribution"""
    return np.exp(-lambda_param) * (lambda_param ** x) / factorial(x)

def log_likelihood(lambda_param, data):
    """Log-likelihood function for Poisson distribution"""
    n = len(data)
    sum_x = np.sum(data)
    # Using Stirling's approximation for factorial to simplify
    return sum_x * np.log(lambda_param) - n * lambda_param

def plot_poisson_pmfs(lambda_values, save_path=None):
    """Plot Poisson PMFs for different lambda values"""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(0, 15)
    
    for lambda_val in lambda_values:
        y = [poisson_pmf(k, lambda_val) for k in x]
        ax.plot(x, y, 'o-', markersize=6, label=f'λ = {lambda_val}')
    
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Mass f(x|λ)')
    ax.set_title('Poisson Distribution PMFs for Different λ Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_likelihood_surface(data, save_path=None):
    """Plot the likelihood function surface"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate MLE estimate (sample mean)
    mle_lambda = np.mean(data)
    
    # Create range of possible lambda values
    lambda_range = np.linspace(max(0.1, mle_lambda*0.5), mle_lambda*1.5, 1000)
    log_likelihoods = [log_likelihood(lambda_val, data) for lambda_val in lambda_range]
    
    ax.plot(lambda_range, log_likelihoods, 'b-', linewidth=2)
    ax.axvline(x=mle_lambda, color='r', linestyle='--', 
               label=f'MLE λ = {mle_lambda:.4f}')
    
    ax.set_xlabel('λ')
    ax.set_ylabel('Log-Likelihood ℓ(λ)')
    ax.set_title('Log-Likelihood Function for Poisson Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return mle_lambda

def plot_mle_fit(data, save_path=None):
    """Plot the fitted MLE distribution against the data"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate MLE estimate (sample mean)
    mle_lambda = np.mean(data)
    
    # Generate x values for plotting
    x_max = max(data) + 5
    x = np.arange(0, x_max)
    y_mle = [poisson_pmf(k, mle_lambda) for k in x]
    
    # Calculate histogram counts and normalize to create PMF
    unique, counts = np.unique(data, return_counts=True)
    observed_pmf = counts / len(data)
    
    # Plot observed PMF
    ax.stem(unique, observed_pmf, linefmt='b-', markerfmt='bo', 
            basefmt=' ', label='Observed PMF')
    
    # Plot the fitted PMF based on MLE
    ax.stem(x, y_mle, linefmt='r-', markerfmt='rx', basefmt=' ', 
            label=f'MLE Fit (λ = {mle_lambda:.4f})')
    
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Mass')
    ax.set_title('Maximum Likelihood Estimation for Poisson Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return mle_lambda

def plot_efficiency_demonstration(save_path=None):
    """Demonstrate that the MLE estimator is efficient"""
    np.random.seed(42)
    true_lambda = 5
    n = 100  # Sample size
    n_samples = 1000  # Number of simulations
    
    # Generate different estimators:
    # 1. MLE: sample mean
    # 2. Alternative 1: median
    # 3. Alternative 2: geometric mean
    
    mle_variances = []
    alt1_variances = []
    alt2_variances = []
    
    sample_sizes = [10, 30, 50, 100, 200, 500]
    
    for size in sample_sizes:
        mle_estimates = []
        alt1_estimates = []
        alt2_estimates = []
        
        for _ in range(n_samples):
            data = np.random.poisson(true_lambda, size)
            
            # MLE estimator (sample mean)
            mle_estimates.append(np.mean(data))
            
            # Alternative 1: median adjusted to be approximately unbiased
            # For Poisson, median is approximately λ - 1/3
            alt1_estimates.append(np.median(data) + 1/3)
            
            # Alternative 2: geometric mean adjusted for Poisson
            # This is not a typical estimator but shown for comparison
            # Add 1 to avoid zero values, then subtract
            alt2_estimates.append(np.exp(np.mean(np.log(data + 1))) - 1)
        
        mle_variances.append(np.var(mle_estimates))
        alt1_variances.append(np.var(alt1_estimates))
        alt2_variances.append(np.var(alt2_estimates))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sample_sizes, mle_variances, 'b-o', linewidth=2, 
            label='MLE (Sample Mean)')
    ax.plot(sample_sizes, alt1_variances, 'r-s', linewidth=2, 
            label='Alternative 1 (Adjusted Median)')
    ax.plot(sample_sizes, alt2_variances, 'g-^', linewidth=2, 
            label='Alternative 2 (Adjusted Geometric Mean)')
    
    # Plot theoretical variance (1/n * λ)
    theoretical_variance = [true_lambda/size for size in sample_sizes]
    ax.plot(sample_sizes, theoretical_variance, 'k--', linewidth=2, 
            label='Theoretical Variance (λ/n)')
    
    ax.set_xlabel('Sample Size (n)')
    ax.set_ylabel('Variance of Estimator')
    ax.set_title('Efficiency Comparison of Different Estimators')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_asymptotic_distribution(save_path=None):
    """Plot the asymptotic distribution of the MLE estimator"""
    np.random.seed(42)
    true_lambda = 5
    n_samples = 10000
    
    # Compare different sample sizes
    sample_sizes = [10, 50, 200]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for size in sample_sizes:
        mle_estimates = []
        
        for _ in range(n_samples):
            data = np.random.poisson(true_lambda, size)
            mle_estimates.append(np.mean(data))
        
        # Standardize the estimates to have mean 0 and variance 1
        standardized = (np.array(mle_estimates) - true_lambda) / np.sqrt(true_lambda/size)
        
        # Plot histogram of standardized estimates
        ax.hist(standardized, bins=50, alpha=0.5, density=True,
                label=f'n = {size}')
    
    # Plot the standard normal density for comparison
    x = np.linspace(-4, 4, 1000)
    ax.plot(x, norm.pdf(x), 'r-', linewidth=2, 
            label='Standard Normal')
    
    ax.set_xlabel('Standardized MLE Estimate')
    ax.set_ylabel('Density')
    ax.set_title('Asymptotic Distribution of MLE Estimator')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def main():
    """Generate all visualizations for Question 5 of the L2.4 quiz"""
    # Create synthetic data
    np.random.seed(42)
    true_lambda = 5
    n = 100
    data = np.random.poisson(true_lambda, n)
    
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_4_Quiz_5")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 5 of the L2.4 MLE quiz...")
    
    # 1. Plot PMFs for different lambda values
    plot_poisson_pmfs([1, 3, 5, 10], save_path=os.path.join(save_dir, "poisson_pmfs.png"))
    print("1. PMF visualization created")
    
    # 2. Plot likelihood surface
    mle_lambda = plot_likelihood_surface(data, save_path=os.path.join(save_dir, "likelihood_surface.png"))
    print(f"2. Likelihood surface visualization created, MLE λ = {mle_lambda:.4f}")
    
    # 3. Plot MLE fit to data
    plot_mle_fit(data, save_path=os.path.join(save_dir, "mle_fit.png"))
    print("3. MLE fit visualization created")
    
    # 4. Demonstrate efficiency
    plot_efficiency_demonstration(save_path=os.path.join(save_dir, "efficiency.png"))
    print("4. Efficiency demonstration created")
    
    # 5. Plot asymptotic distribution
    plot_asymptotic_distribution(save_path=os.path.join(save_dir, "asymptotic_distribution.png"))
    print("5. Asymptotic distribution visualization created")
    
    print(f"\nAll visualizations have been saved to: {save_dir}")
    print(f"For the synthetic data used, the MLE estimate is λ = {mle_lambda:.4f}")

if __name__ == "__main__":
    main() 