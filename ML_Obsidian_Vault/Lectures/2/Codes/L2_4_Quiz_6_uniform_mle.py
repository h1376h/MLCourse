import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
import os

def uniform_pdf(x, theta):
    """Probability density function for uniform distribution on [0, theta]"""
    return np.where((x >= 0) & (x <= theta), 1/theta, 0)

def log_likelihood(theta, data):
    """Log-likelihood function for uniform distribution on [0, theta]"""
    if theta < np.max(data):
        return -np.inf  # Impossible case
    return -len(data) * np.log(theta)

def plot_uniform_pdfs(theta_values, save_path=None):
    """Plot uniform PDFs for different theta values"""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(0, max(theta_values) * 1.2, 1000)
    
    for theta in theta_values:
        y = uniform_pdf(x, theta)
        ax.plot(x, y, linewidth=2, label=f'θ = {theta}')
    
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density f(x|θ)')
    ax.set_title('Uniform Distribution PDFs for Different θ Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_likelihood_surface(data, save_path=None):
    """Plot the likelihood function surface"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate MLE (maximum of the data)
    mle_theta = np.max(data)
    
    # Create range of possible theta values
    theta_range = np.linspace(mle_theta * 0.8, mle_theta * 1.5, 1000)
    log_likelihoods = [log_likelihood(theta, data) for theta in theta_range]
    
    ax.plot(theta_range, log_likelihoods, 'b-', linewidth=2)
    ax.axvline(x=mle_theta, color='r', linestyle='--', 
               label=f'MLE θ = {mle_theta:.4f}')
    
    ax.set_xlabel('θ')
    ax.set_ylabel('Log-Likelihood ℓ(θ)')
    ax.set_title('Log-Likelihood Function for Uniform Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return mle_theta

def plot_mle_fit(data, save_path=None):
    """Plot the fitted MLE distribution against the data"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate MLE (maximum of the data)
    mle_theta = np.max(data)
    
    # Generate x values for plotting
    x = np.linspace(0, mle_theta * 1.5, 1000)
    y_mle = uniform_pdf(x, mle_theta)
    
    # Plot histogram of the data
    ax.hist(data, bins=min(20, len(data)//5), density=True, alpha=0.5, 
            color='blue', label='Observed Data')
    
    # Plot the fitted PDF based on MLE
    ax.plot(x, y_mle, 'r-', linewidth=2, 
            label=f'MLE Fit (θ = {mle_theta:.4f})')
    
    # Mark the observed data points
    ax.plot(data, np.zeros_like(data), 'bo', markersize=5, alpha=0.6)
    
    # Mark the MLE estimate
    ax.plot(mle_theta, 0, 'rx', markersize=10, label='MLE θ (max observation)')
    
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density')
    ax.set_title('Maximum Likelihood Estimation for Uniform Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return mle_theta

def plot_bias_demonstration(save_path=None):
    """Demonstrate that the MLE estimator is biased"""
    np.random.seed(42)
    true_theta = 10
    n_samples = 10000
    
    sample_sizes = [5, 10, 30, 100, 300]
    bias_values = []
    mle_means = []
    unbiased_means = []
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for n in sample_sizes:
        mle_estimates = []
        unbiased_estimates = []
        
        for _ in range(n_samples):
            data = np.random.uniform(0, true_theta, n)
            mle_theta = np.max(data)
            mle_estimates.append(mle_theta)
            # Unbiased estimator: (n+1)/n * max(X_i)
            unbiased_estimates.append((n+1)/n * mle_theta)
        
        # Calculate bias of MLE
        mle_mean = np.mean(mle_estimates)
        bias = mle_mean - true_theta
        mle_means.append(mle_mean)
        bias_values.append(bias)
        unbiased_means.append(np.mean(unbiased_estimates))
        
        # Plot distribution of estimates
        ax.hist(mle_estimates, bins=50, alpha=0.3, density=True,
                label=f'MLE (n={n}, Mean={mle_mean:.4f})')
    
    ax.axvline(x=true_theta, color='r', linestyle='--', 
               label=f'True θ = {true_theta}')
    
    ax.set_xlabel('Estimated θ')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of MLE Estimates for Different Sample Sizes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return sample_sizes, mle_means, unbiased_means, bias_values

def plot_bias_curve(sample_sizes, mle_means, unbiased_means, true_theta, save_path=None):
    """Plot the bias curve for different sample sizes"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot MLE means
    ax.plot(sample_sizes, mle_means, 'b-o', linewidth=2, 
            label='MLE Estimates')
    
    # Plot unbiased estimator means
    ax.plot(sample_sizes, unbiased_means, 'g-^', linewidth=2, 
            label='Unbiased Estimates')
    
    # Plot true theta
    ax.axhline(y=true_theta, color='r', linestyle='--', 
               label=f'True θ = {true_theta}')
    
    # Plot theoretical expected value of MLE
    theoretical_mean = [true_theta * n/(n+1) for n in sample_sizes]
    ax.plot(sample_sizes, theoretical_mean, 'k:', linewidth=2, 
            label='Theoretical E[MLE]')
    
    ax.set_xlabel('Sample Size (n)')
    ax.set_ylabel('Average Estimated Value')
    ax.set_title('Bias in MLE and Unbiased Estimator for Uniform Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def main():
    """Generate all visualizations for Question 6 of the L2.4 quiz"""
    # Create synthetic data
    np.random.seed(42)
    true_theta = 10
    n = 50
    data = np.random.uniform(0, true_theta, n)
    
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_4_Quiz_6")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 6 of the L2.4 MLE quiz...")
    
    # 1. Plot PDFs for different theta values
    plot_uniform_pdfs([5, 10, 15], save_path=os.path.join(save_dir, "uniform_pdfs.png"))
    print("1. PDF visualization created")
    
    # 2. Plot likelihood surface
    mle_theta = plot_likelihood_surface(data, save_path=os.path.join(save_dir, "likelihood_surface.png"))
    print(f"2. Likelihood surface visualization created, MLE θ = {mle_theta:.4f}")
    
    # 3. Plot MLE fit to data
    plot_mle_fit(data, save_path=os.path.join(save_dir, "mle_fit.png"))
    print("3. MLE fit visualization created")
    
    # 4. Demonstrate bias
    sample_sizes, mle_means, unbiased_means, bias_values = plot_bias_demonstration(
        save_path=os.path.join(save_dir, "bias_demonstration.png"))
    print("4. Bias demonstration created")
    
    # 5. Plot bias curve
    plot_bias_curve(sample_sizes, mle_means, unbiased_means, true_theta,
                   save_path=os.path.join(save_dir, "bias_curve.png"))
    print("5. Bias curve visualization created")
    
    # Print summary
    print(f"\nSummary of findings:")
    print(f"True θ = {true_theta}")
    print(f"MLE for sample of size {n}: θ = {mle_theta:.4f}")
    print(f"Expected bias for sample of size {n}: {true_theta/(n+1):.4f}")
    print(f"Unbiased estimator: (n+1)/n * max(X_i) = {(n+1)/n * mle_theta:.4f}")
    
    print(f"\nAll visualizations have been saved to: {save_dir}")

if __name__ == "__main__":
    main() 