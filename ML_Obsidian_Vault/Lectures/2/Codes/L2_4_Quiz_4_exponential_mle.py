import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon
import os

def exponential_pdf(x, lambda_param):
    """Probability density function for exponential distribution"""
    return lambda_param * np.exp(-lambda_param * x)

def log_likelihood(lambda_param, data):
    """Log-likelihood function for exponential distribution"""
    n = len(data)
    return n * np.log(lambda_param) - lambda_param * np.sum(data)

def plot_exponential_pdfs(lambda_values, save_path=None):
    """Plot exponential PDFs for different lambda values"""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(0, 5, 1000)
    
    for lambda_val in lambda_values:
        y = exponential_pdf(x, lambda_val)
        ax.plot(x, y, label=f'λ = {lambda_val}')
    
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density f(x|λ)')
    ax.set_title('Exponential Distribution PDFs for Different λ Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_likelihood_surface(data, save_path=None):
    """Plot the likelihood function surface"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate MLE estimate (reciprocal of sample mean)
    mle_lambda = 1 / np.mean(data)
    
    # Create range of possible lambda values
    lambda_range = np.linspace(0.1, 2*mle_lambda, 1000)
    log_likelihoods = [log_likelihood(lambda_val, data) for lambda_val in lambda_range]
    
    ax.plot(lambda_range, log_likelihoods, 'b-', linewidth=2)
    ax.axvline(x=mle_lambda, color='r', linestyle='--', 
               label=f'MLE λ = {mle_lambda:.4f}')
    
    ax.set_xlabel('λ')
    ax.set_ylabel('Log-Likelihood ℓ(λ)')
    ax.set_title('Log-Likelihood Function for Exponential Distribution')
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
    
    # Calculate MLE estimate
    mle_lambda = 1 / np.mean(data)
    
    # Generate x values for plotting
    x = np.linspace(0, max(data)*1.5, 1000)
    y_mle = exponential_pdf(x, mle_lambda)
    
    # Plot histogram of the data
    ax.hist(data, bins=min(15, len(data)), density=True, alpha=0.5, color='blue', 
             label='Observed Data')
    
    # Plot the fitted PDF based on MLE
    ax.plot(x, y_mle, 'r-', linewidth=2, 
            label=f'MLE Fit (λ = {mle_lambda:.4f})')
    
    # Mark the observed data points
    ax.plot(data, np.zeros_like(data), 'bo', markersize=8, alpha=0.6)
    
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density')
    ax.set_title('Maximum Likelihood Estimation for Exponential Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return mle_lambda

def plot_consistency_demonstration(save_path=None):
    """Demonstrate that the MLE estimator is consistent"""
    np.random.seed(42)
    true_lambda = 2
    n_samples = 1000
    sample_sizes = [10, 50, 100, 500]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for n in sample_sizes:
        mle_estimates = []
        for _ in range(n_samples):
            data = np.random.exponential(1/true_lambda, n)
            mle_estimates.append(1/np.mean(data))
        
        ax.hist(mle_estimates, bins=30, alpha=0.5, 
                label=f'n = {n}, Mean = {np.mean(mle_estimates):.4f}')
    
    ax.axvline(x=true_lambda, color='r', linestyle='--', 
               label=f'True λ = {true_lambda}')
    
    ax.set_xlabel('MLE Estimate of λ')
    ax.set_ylabel('Frequency')
    ax.set_title('Demonstration of MLE Consistency for Different Sample Sizes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_mse_demonstration(save_path=None):
    """Plot MSE of MLE estimator for different sample sizes"""
    np.random.seed(42)
    true_lambda = 2
    n_samples = 1000
    sample_sizes = np.arange(10, 501, 10)
    
    mse_values = []
    for n in sample_sizes:
        mle_estimates = []
        for _ in range(n_samples):
            data = np.random.exponential(1/true_lambda, n)
            mle_estimates.append(1/np.mean(data))
        mse = np.mean((np.array(mle_estimates) - true_lambda)**2)
        mse_values.append(mse)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sample_sizes, mse_values, 'b-', linewidth=2)
    
    ax.set_xlabel('Sample Size (n)')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('MSE of MLE Estimator vs Sample Size')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def main():
    """Generate all visualizations for Question 4 of the L2.4 quiz"""
    # Create synthetic data
    np.random.seed(42)
    true_lambda = 2
    n = 100
    data = np.random.exponential(1/true_lambda, n)
    
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_4_Quiz_4")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 4 of the L2.4 MLE quiz...")
    
    # 1. Plot PDFs for different lambda values
    plot_exponential_pdfs([0.5, 1.0, 2.0], save_path=os.path.join(save_dir, "exponential_pdfs.png"))
    print("1. PDF visualization created")
    
    # 2. Plot likelihood surface
    mle_lambda = plot_likelihood_surface(data, save_path=os.path.join(save_dir, "likelihood_surface.png"))
    print(f"2. Likelihood surface visualization created, MLE λ = {mle_lambda:.4f}")
    
    # 3. Plot MLE fit to data
    plot_mle_fit(data, save_path=os.path.join(save_dir, "mle_fit.png"))
    print("3. MLE fit visualization created")
    
    # 4. Demonstrate consistency
    plot_consistency_demonstration(save_path=os.path.join(save_dir, "consistency.png"))
    print("4. Consistency demonstration created")
    
    # 5. Plot MSE vs sample size
    plot_mse_demonstration(save_path=os.path.join(save_dir, "mse.png"))
    print("5. MSE visualization created")
    
    print(f"\nAll visualizations have been saved to: {save_dir}")
    print(f"For the synthetic data used, the MLE estimate is λ = {mle_lambda:.4f}")

if __name__ == "__main__":
    main() 