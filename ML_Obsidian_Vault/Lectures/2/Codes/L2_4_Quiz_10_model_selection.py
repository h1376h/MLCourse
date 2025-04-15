import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, nbinom
from scipy.optimize import minimize
import os
import math

def poisson_pmf(k, lambda_param):
    """Probability mass function for Poisson distribution"""
    return np.exp(-lambda_param) * lambda_param**k / math.factorial(k)

def negative_binomial_pmf(k, r, p):
    """Probability mass function for Negative Binomial distribution with fixed r"""
    from scipy.special import comb
    return comb(k+r-1, k) * p**r * (1-p)**k

def poisson_log_likelihood(lambda_param, data):
    """Log-likelihood function for Poisson distribution"""
    return sum([np.log(poisson_pmf(k, lambda_param)) for k in data])

def negative_binomial_log_likelihood(p, r, data):
    """Log-likelihood function for Negative Binomial distribution with fixed r"""
    return sum([np.log(negative_binomial_pmf(k, r, p)) for k in data])

def neg_poisson_log_likelihood(lambda_param, data):
    """Negative log-likelihood for Poisson (for optimization)"""
    return -poisson_log_likelihood(lambda_param, data)

def neg_negative_binomial_log_likelihood(p, r, data):
    """Negative log-likelihood for Negative Binomial (for optimization)"""
    return -negative_binomial_log_likelihood(p, r, data)

def plot_data_histogram(data, save_path=None):
    """Plot histogram of the observed data"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bins = np.arange(min(data)-0.5, max(data)+1.5, 1)
    ax.hist(data, bins=bins, alpha=0.7, color='blue', 
            label='Observed Data', edgecolor='black')
    
    ax.set_xlabel('Number of Defects')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Defect Counts in Fabric Samples')
    ax.set_xticks(range(min(data), max(data)+1))
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_model_fits(data, lambda_mle, p_mle, r, save_path=None):
    """Plot the fitted models against the data"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate the empirical PMF
    values, counts = np.unique(data, return_counts=True)
    frequencies = counts / len(data)
    
    # Plot the empirical PMF as a bar chart
    ax.bar(values, frequencies, alpha=0.5, color='blue', 
           label='Observed Data', edgecolor='black', width=0.4)
    
    # Plot the theoretical PMFs for both models
    x_range = np.arange(0, max(data) + 3)
    
    # Poisson PMF
    poisson_probs = [poisson_pmf(k, lambda_mle) for k in x_range]
    ax.plot(x_range, poisson_probs, 'ro-', markersize=8, alpha=0.7,
            label=f'Poisson Model (λ = {lambda_mle:.4f})')
    
    # Negative Binomial PMF with fixed r=2
    nb_probs = [negative_binomial_pmf(k, r, p_mle) for k in x_range]
    ax.plot(x_range, nb_probs, 'go-', markersize=8, alpha=0.7,
            label=f'Negative Binomial Model (r = {r}, p = {p_mle:.4f})')
    
    ax.set_xlabel('Number of Defects')
    ax.set_ylabel('Probability')
    ax.set_title('Model Comparison: Poisson vs Negative Binomial')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_likelihood_ratio_test(data, lambda_mle, p_mle, r, save_path=None):
    """Visualize the likelihood ratio test"""
    # Calculate log-likelihoods
    poisson_ll = poisson_log_likelihood(lambda_mle, data)
    nb_ll = negative_binomial_log_likelihood(p_mle, r, data)
    
    # Calculate AIC values
    poisson_aic = 2*1 - 2*poisson_ll  # k=1 parameter
    nb_aic = 2*1 - 2*nb_ll            # k=1 parameter (r is fixed)
    
    # Create bar chart for log-likelihoods
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    models = ['Poisson', 'Negative Binomial']
    log_likelihoods = [poisson_ll, nb_ll]
    
    bars = ax1.bar(models, log_likelihoods, color=['red', 'green'], alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom' if height > 0 else 'top')
    
    ax1.set_ylabel('Log-Likelihood')
    ax1.set_title('Log-Likelihood Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Create bar chart for AIC values
    aic_values = [poisson_aic, nb_aic]
    
    bars = ax2.bar(models, aic_values, color=['red', 'green'], alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    ax2.set_ylabel('AIC Value')
    ax2.set_title('AIC Comparison (Lower is Better)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def main():
    """Generate all visualizations for Question 10 of the L2.4 quiz"""
    # Data from the question
    data = np.array([0, 1, 2, 3, 1, 0, 2, 4, 1, 2])
    r = 2  # Fixed parameter for Negative Binomial
    
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_4_Quiz_10")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 10 of the L2.4 MLE quiz...")
    
    # 1. Plot the data histogram
    plot_data_histogram(data, save_path=os.path.join(save_dir, "data_histogram.png"))
    print("1. Data histogram visualization created")
    
    # 2. Find MLE for Poisson model
    result_poisson = minimize(neg_poisson_log_likelihood, x0=1.0, args=(data,), 
                             bounds=[(0.001, None)])
    lambda_mle = result_poisson.x[0]
    poisson_ll = -result_poisson.fun
    print(f"2. MLE for Poisson λ = {lambda_mle:.4f}")
    
    # 3. Find MLE for Negative Binomial model with fixed r=2
    result_nb = minimize(lambda p: neg_negative_binomial_log_likelihood(p[0], r, data), 
                        x0=[0.5], bounds=[(0.001, 0.999)])
    p_mle = result_nb.x[0]
    nb_ll = -result_nb.fun
    print(f"3. MLE for Negative Binomial p (with r={r}) = {p_mle:.4f}")
    
    # 4. Plot model fits
    plot_model_fits(data, lambda_mle, p_mle, r, 
                   save_path=os.path.join(save_dir, "model_fits.png"))
    print("4. Model fits visualization created")
    
    # 5. Likelihood ratio test visualization
    plot_likelihood_ratio_test(data, lambda_mle, p_mle, r, 
                              save_path=os.path.join(save_dir, "likelihood_ratio_test.png"))
    print("5. Likelihood ratio test visualization created")
    
    # Calculate likelihood ratio
    likelihood_ratio = 2 * (nb_ll - poisson_ll)
    
    # Calculate AIC values
    poisson_aic = 2*1 - 2*poisson_ll  # k=1 parameter
    nb_aic = 2*1 - 2*nb_ll            # k=1 parameter (r is fixed)
    
    # Print results
    print("\nQuestion 10 Results:")
    print(f"Poisson Model (Model A) - λ_MLE = {lambda_mle:.4f}")
    print(f"Log-likelihood: {poisson_ll:.4f}")
    print(f"AIC: {poisson_aic:.4f}")
    
    print(f"\nNegative Binomial Model (Model B) - p_MLE = {p_mle:.4f} (with r={r})")
    print(f"Log-likelihood: {nb_ll:.4f}")
    print(f"AIC: {nb_aic:.4f}")
    
    print(f"\nLikelihood Ratio Test Statistic: {likelihood_ratio:.4f}")
    
    if nb_ll > poisson_ll:
        print("The Negative Binomial model fits the data better based on log-likelihood.")
    else:
        print("The Poisson model fits the data better based on log-likelihood.")
    
    if nb_aic < poisson_aic:
        print("The Negative Binomial model is preferred based on AIC.")
    else:
        print("The Poisson model is preferred based on AIC.")
    
    print(f"\nAll visualizations have been saved to: {save_dir}")

if __name__ == "__main__":
    main() 