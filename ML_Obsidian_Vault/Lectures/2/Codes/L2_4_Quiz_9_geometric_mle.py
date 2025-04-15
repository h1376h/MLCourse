import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import geom
from scipy.optimize import minimize_scalar

def geometric_pmf(x, p):
    """Probability mass function for geometric distribution"""
    return p * (1 - p) ** (x - 1)

def log_likelihood(p, data):
    """Log-likelihood function for geometric distribution"""
    n = len(data)
    return n * np.log(p) + np.sum(np.log(1 - p) * (data - 1))

def plot_geometric_pmfs(p_values, save_path=None):
    """Plot geometric PMFs for different p values"""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(1, 16)
    
    for p_val in p_values:
        y = geometric_pmf(x, p_val)
        ax.bar(x + 0.1*p_values.index(p_val), y, width=0.1, alpha=0.7, 
               label=f'p = {p_val}')
    
    ax.set_xlabel('x (Number of trials until first success)')
    ax.set_ylabel('Probability P(X=x)')
    ax.set_title('Geometric Distribution PMFs for Different p Values')
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
    mle_p = 1 / np.mean(data)
    
    # Create range of possible p values
    p_range = np.linspace(0.01, 0.99, 1000)
    log_likelihoods = [log_likelihood(p_val, data) for p_val in p_range]
    
    ax.plot(p_range, log_likelihoods, 'b-', linewidth=2)
    ax.axvline(x=mle_p, color='r', linestyle='--', 
               label=f'MLE p = {mle_p:.4f}')
    
    ax.set_xlabel('p (Probability of success)')
    ax.set_ylabel('Log-Likelihood ℓ(p)')
    ax.set_title('Log-Likelihood Function for Geometric Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return mle_p

def plot_mle_fit(data, save_path=None):
    """Plot the fitted MLE distribution against the data"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate MLE estimate
    mle_p = 1 / np.mean(data)
    
    # Generate x values for plotting
    x = np.arange(1, max(data) + 5)
    y_mle = geometric_pmf(x, mle_p)
    
    # Plot histogram of the data
    counts, bins = np.histogram(data, bins=range(1, max(data) + 2), density=True)
    ax.bar(bins[:-1], counts, alpha=0.5, color='blue', label='Observed Data')
    
    # Plot the fitted PMF based on MLE
    ax.bar(x, y_mle, alpha=0.5, color='red', 
           label=f'MLE Fit (p = {mle_p:.4f})')
    
    ax.set_xlabel('Number of trials until first success')
    ax.set_ylabel('Probability')
    ax.set_title('Maximum Likelihood Estimation for Geometric Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return mle_p

def plot_expected_value_variance(p_range, save_path=None):
    """Plot expected value and variance for different p values"""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Calculate expected value and variance for different p values
    expected_values = 1 / p_range
    variances = (1 - p_range) / (p_range ** 2)
    
    ax1.plot(p_range, expected_values, 'b-', linewidth=2, label='Expected Value')
    ax1.set_xlabel('p (Probability of success)')
    ax1.set_ylabel('Expected Number of Trials E[X]', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax2 = ax1.twinx()
    ax2.plot(p_range, variances, 'r-', linewidth=2, label='Variance')
    ax2.set_ylabel('Variance Var[X]', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title('Expected Value and Variance of Geometric Distribution')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def main():
    """Generate all visualizations for Question 9 of the L2.4 quiz"""
    # Data from the question
    data = np.array([3, 1, 4, 2, 5, 2, 1, 3])
    
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_4_Quiz_9")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 9 of the L2.4 MLE quiz...")
    
    # 1. Plot PMFs for different p values
    plot_geometric_pmfs([0.2, 0.4, 0.6], save_path=os.path.join(save_dir, "geometric_pmfs.png"))
    print("1. PMF visualization created")
    
    # 2. Plot likelihood surface
    mle_p = plot_likelihood_surface(data, save_path=os.path.join(save_dir, "likelihood_surface.png"))
    print(f"2. Likelihood surface visualization created, MLE p = {mle_p:.4f}")
    
    # 3. Plot MLE fit to data
    plot_mle_fit(data, save_path=os.path.join(save_dir, "mle_fit.png"))
    print("3. MLE fit visualization created")
    
    # 4. Plot expected value and variance
    p_range = np.linspace(0.1, 0.9, 100)
    plot_expected_value_variance(p_range, save_path=os.path.join(save_dir, "expected_variance.png"))
    print("4. Expected value and variance visualization created")
    
    # Calculate results for Question 9
    print("\nQuestion 9 Results:")
    print(f"MLE for p: {mle_p:.4f}")
    expected_trials = 1 / mle_p
    variance = (1 - mle_p) / (mle_p ** 2)
    print(f"Expected number of trials until first success: {expected_trials:.4f}")
    print(f"Variance of the number of trials: {variance:.4f}")
    
    print(f"\nAll visualizations have been saved to: {save_dir}")

if __name__ == "__main__":
    main() 