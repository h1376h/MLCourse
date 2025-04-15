import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm
from scipy.special import factorial
import os

def poisson_pmf(k, lambda_param):
    """Probability mass function for Poisson distribution"""
    return np.exp(-lambda_param) * (lambda_param ** k) / factorial(k)

def log_likelihood(lambda_param, data):
    """Log-likelihood function for Poisson distribution"""
    n = len(data)
    return np.sum(data) * np.log(lambda_param) - n * lambda_param - np.sum([np.log(factorial(x)) for x in data])

def plot_poisson_pmfs(lambda_values, save_path=None):
    """Plot Poisson PMFs for different lambda values"""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(0, 25)
    
    for lambda_val in lambda_values:
        y = [poisson_pmf(k, lambda_val) for k in x]
        ax.plot(x, y, 'o-', label=f'λ = {lambda_val}')
    
    ax.set_xlabel('k (Number of Walk-in Customers)')
    ax.set_ylabel('Probability Mass P(X=k)')
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
    lambda_range = np.linspace(max(0.1, mle_lambda-5), mle_lambda+5, 1000)
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
    
    # Calculate MLE estimate
    mle_lambda = np.mean(data)
    
    # Set up x values for plotting
    x_range = np.arange(max(0, min(data)-2), max(data)+5)
    
    # Calculate PMF values using MLE estimate
    pmf_values = [poisson_pmf(k, mle_lambda) for k in x_range]
    
    # Count frequencies in observed data
    unique, counts = np.unique(data, return_counts=True)
    observed_freq = dict(zip(unique, counts/len(data)))
    
    # Plot the observed frequencies
    ax.bar(unique, [observed_freq.get(k, 0) for k in unique], 
           alpha=0.6, color='blue', label='Observed Frequencies')
    
    # Plot the fitted PMF based on MLE
    ax.plot(x_range, pmf_values, 'ro-', linewidth=2, 
            label=f'MLE Fit (λ = {mle_lambda:.4f})')
    
    ax.set_xlabel('k (Number of Walk-in Customers)')
    ax.set_ylabel('Probability')
    ax.set_title('Maximum Likelihood Estimation for Poisson Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return mle_lambda

def plot_confidence_interval(data, save_path=None):
    """Plot the confidence interval for lambda"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate MLE estimate
    mle_lambda = np.mean(data)
    n = len(data)
    
    # Calculate 95% confidence interval
    # For Poisson, Var(X) = λ, so SE(λ_MLE) = sqrt(λ/n)
    se = np.sqrt(mle_lambda / n)
    z_critical = norm.ppf(0.975)  # 95% CI (two-tailed)
    ci_lower = mle_lambda - z_critical * se
    ci_upper = mle_lambda + z_critical * se
    
    # Create range of possible lambda values
    lambda_range = np.linspace(max(0.1, ci_lower-1), ci_upper+1, 1000)
    log_likelihoods = [log_likelihood(lambda_val, data) for lambda_val in lambda_range]
    
    # Normalize log-likelihoods for better visualization
    log_likelihoods = np.array(log_likelihoods)
    log_likelihoods = log_likelihoods - np.max(log_likelihoods)
    
    ax.plot(lambda_range, log_likelihoods, 'b-', linewidth=2)
    ax.axvline(x=mle_lambda, color='r', linestyle='--', 
               label=f'MLE λ = {mle_lambda:.4f}')
    
    # Add confidence interval
    ax.axvline(x=ci_lower, color='g', linestyle=':', 
               label=f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
    ax.axvline(x=ci_upper, color='g', linestyle=':')
    
    # Shade the confidence interval region
    ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='green')
    
    ax.set_xlabel('λ')
    ax.set_ylabel('Normalized Log-Likelihood')
    ax.set_title('95% Confidence Interval for Poisson Rate Parameter λ')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return mle_lambda, ci_lower, ci_upper

def plot_probability_demonstration(data, target_k=11, save_path=None):
    """Plot the probability of observing exactly k customers"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate MLE estimate
    mle_lambda = np.mean(data)
    
    # Set up x values for plotting
    x_range = np.arange(max(0, min(data)-2), max(data)+10)
    
    # Calculate PMF values using MLE estimate
    pmf_values = [poisson_pmf(k, mle_lambda) for k in x_range]
    target_prob = poisson_pmf(target_k, mle_lambda)
    
    # Plot the PMF
    ax.bar(x_range, pmf_values, color='lightblue', alpha=0.7)
    
    # Highlight the target k value
    ax.bar([target_k], [target_prob], color='red', 
           label=f'P(X={target_k}) = {target_prob:.4f}')
    
    ax.set_xlabel('k (Number of Walk-in Customers)')
    ax.set_ylabel('Probability P(X=k)')
    ax.set_title(f'Probability of Observing Exactly {target_k} Walk-in Customers')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return target_prob

def main():
    """Generate all visualizations for Question 20 of the L2.4 quiz"""
    # Original data from the problem
    data = np.array([6, 8, 5, 7, 9, 12, 10])
    
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_4_Quiz_20")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 20 of the L2.4 MLE quiz...")
    
    # 1. Plot PMFs for different lambda values
    plot_poisson_pmfs([5, 8, 11], save_path=os.path.join(save_dir, "poisson_pmfs.png"))
    print("1. PMF visualization created")
    
    # 2. Plot likelihood surface
    mle_lambda = plot_likelihood_surface(data, save_path=os.path.join(save_dir, "likelihood_surface.png"))
    print(f"2. Likelihood surface visualization created, MLE λ = {mle_lambda:.4f}")
    
    # 3. Plot MLE fit to data
    plot_mle_fit(data, save_path=os.path.join(save_dir, "mle_fit.png"))
    print("3. MLE fit visualization created")
    
    # 4. Plot confidence interval
    mle_lambda, ci_lower, ci_upper = plot_confidence_interval(data, save_path=os.path.join(save_dir, "confidence_interval.png"))
    print(f"4. Confidence interval visualization created: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # 5. Plot probability of exactly 11 customers
    target_k = 11
    prob_k = plot_probability_demonstration(data, target_k, save_path=os.path.join(save_dir, "probability_k.png"))
    print(f"5. Probability visualization created: P(X={target_k}) = {prob_k:.4f}")
    
    # Print summary of results
    print("\nSummary of Results:")
    print(f"Maximum Likelihood Estimate (MLE) for λ: {mle_lambda:.4f}")
    print(f"95% Confidence Interval for λ: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"Probability of exactly {target_k} walk-in customers: {prob_k:.4f}")
    
    print(f"\nAll visualizations have been saved to: {save_dir}")

if __name__ == "__main__":
    main() 