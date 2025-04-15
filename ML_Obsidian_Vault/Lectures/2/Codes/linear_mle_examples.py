import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import os

def linear_pdf(x, theta):
    """Probability density function for f(x) = 1/2(1 + θx), -1 ≤ x ≤ 1"""
    if not np.all((-1 <= x) & (x <= 1)):
        return np.zeros_like(x)
    return 0.5 * (1 + theta * x)

def log_likelihood(theta, data):
    """Log-likelihood function for linear distribution"""
    # Check if data is within bounds
    if not np.all((-1 <= data) & (data <= 1)):
        return float('-inf')
    
    # Check if theta is in valid range (must be between -1 and 1)
    if not (-1 <= theta <= 1):
        return float('-inf')
    
    # Calculate log-likelihood
    return np.sum(np.log(0.5 * (1 + theta * data)))

def plot_linear_likelihood(data, title, save_path=None):
    """Plot the likelihood function for linear data and highlight the MLE."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a range of possible theta values to plot
    possible_thetas = np.linspace(-1, 1, 1000)
    
    # Calculate the log-likelihood for each possible theta
    log_likelihoods = []
    for theta in possible_thetas:
        ll = log_likelihood(theta, data)
        log_likelihoods.append(ll)
    
    # Find MLE using numerical optimization
    def neg_log_likelihood(theta):
        return -log_likelihood(theta, data)
    
    result = minimize_scalar(neg_log_likelihood, bounds=(-1, 1), method='bounded')
    mle_theta = result.x
    
    # Normalize the log-likelihood for better visualization
    log_likelihoods = np.array(log_likelihoods)
    log_likelihoods = log_likelihoods - np.min(log_likelihoods)
    log_likelihoods = log_likelihoods / np.max(log_likelihoods)
    
    # Plot the log-likelihood function
    ax.plot(possible_thetas, log_likelihoods, 'b-', linewidth=2)
    ax.axvline(x=mle_theta, color='r', linestyle='--', 
              label=f'MLE θ = {mle_theta:.2f}')
    
    ax.set_title(f"{title} - Log-Likelihood Function")
    ax.set_xlabel('θ')
    ax.set_ylabel('Normalized Log-Likelihood')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return mle_theta

def plot_linear_distribution(data, title, mle_theta, save_path=None):
    """Plot the data and the estimated linear distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate x values for plotting
    x = np.linspace(-1, 1, 1000)
    y = linear_pdf(x, mle_theta)
    
    # Plot histogram of the data
    ax.hist(data, bins=10, density=True, alpha=0.5, color='blue', 
             label='Observed Data')
    
    # Plot the estimated PDF
    ax.plot(x, y, 'r-', linewidth=2, 
            label=f'Estimated Linear Distribution (θ = {mle_theta:.2f})')
    
    # Mark the observed data points
    ax.plot(data, np.zeros_like(data), 'bo', markersize=8, alpha=0.6)
    
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'MLE for Linear Distribution - {title}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 1)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def analyze_linear_data(name, data, context, save_dir=None):
    """Analyze linear data with detailed steps using MLE."""
    print(f"\n{'='*50}")
    print(f"{name} Example")
    print(f"{'='*50}")
    print(f"Context: {context}")
    
    # Step 1: Data analysis
    print("\nStep 1: Data Analysis")
    print(f"- Data: {data}")
    print(f"- Number of observations: {len(data)}")
    
    # Step 2: Calculate MLE numerically
    print("\nStep 2: Maximum Likelihood Estimation")
    print("- For our linear distribution, MLE of θ needs to be calculated numerically")
    
    def neg_log_likelihood(theta):
        return -log_likelihood(theta, data)
    
    result = minimize_scalar(neg_log_likelihood, bounds=(-1, 1), method='bounded')
    mle_theta = result.x
    print(f"- MLE theta (θ) = {mle_theta:.4f}")
    
    # Create save paths if directory is provided
    likelihood_save_path = None
    distribution_save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base_filename = f"linear_mle_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
        likelihood_save_path = os.path.join(save_dir, base_filename + "_likelihood.png")
        distribution_save_path = os.path.join(save_dir, base_filename + ".png")
    
    # Step 3: Visualize likelihood function
    print("\nStep 3: Likelihood Visualization")
    plot_linear_likelihood(data, name, likelihood_save_path)
    
    # Step 4: Visualize distribution
    print("\nStep 4: Distribution Visualization")
    plot_linear_distribution(data, name, mle_theta, distribution_save_path)
    
    # Step 5: Confidence Interval
    print("\nStep 5: Confidence Interval for Theta")
    # We'll use bootstrap for confidence interval
    n_bootstrap = 1000
    bootstrap_thetas = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        # Calculate MLE for bootstrap sample
        result = minimize_scalar(lambda theta: -log_likelihood(theta, bootstrap_sample), 
                               bounds=(-1, 1), method='bounded')
        bootstrap_thetas.append(result.x)
    
    # Calculate confidence interval from bootstrap samples
    ci_lower = np.percentile(bootstrap_thetas, 2.5)
    ci_upper = np.percentile(bootstrap_thetas, 97.5)
    print(f"- 95% Bootstrap Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Step 6: Interpretation
    print("\nStep 6: Interpretation")
    print(f"- Based on the observed data alone, the most likely parameter θ is {mle_theta:.4f}")
    print(f"- This linear distribution with θ = {mle_theta:.4f} best explains the observed data")
    
    return {"theta": mle_theta, "ci": [ci_lower, ci_upper], 
            "likelihood_path": likelihood_save_path, 
            "distribution_path": distribution_save_path}

def generate_linear_examples(save_dir=None):
    results = {}
    
    print("""
    Let's analyze different scenarios using Maximum Likelihood Estimation for linear distributions!
    Each example will show how we can estimate the parameter θ using only the observed data.
    """)

    # Example 1: Linear Distribution
    linear_data = np.array([-0.8, 0.2, 0.5, -0.3, 0.7, -0.1])  # example data
    linear_context = """
    A researcher is studying a linear distribution with PDF f(x) = 1/2(1 + θx), -1 ≤ x ≤ 1.
    - You have 6 observations from this distribution
    - Using only the observed data (no prior assumptions)
    """
    linear_results = analyze_linear_data("Linear Distribution", linear_data, linear_context, save_dir)
    results["Linear Distribution"] = linear_results
    
    return results

if __name__ == "__main__":
    # Use relative path for save directory
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level and then to Images
    save_dir = os.path.join(os.path.dirname(script_dir), "Images")
    generate_linear_examples(save_dir) 