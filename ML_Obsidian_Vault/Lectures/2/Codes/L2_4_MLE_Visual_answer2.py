import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson, nbinom, beta
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.optimize import minimize

def load_mystery_data():
    """Generate the same mystery data as in the question script"""
    np.random.seed(42)
    return np.random.negative_binomial(5, 0.6, 150)

def fit_distributions(data):
    """
    Fit various distributions to the data using MLE and calculate log-likelihoods
    """
    results = {}
    
    # Fit Poisson distribution
    # MLE for Poisson lambda is the sample mean
    lambda_mle = np.mean(data)
    poisson_log_likelihood = np.sum(poisson.logpmf(data, lambda_mle))
    results['poisson'] = {
        'params': {'lambda': lambda_mle},
        'log_likelihood': poisson_log_likelihood
    }
    
    # Fit negative binomial distribution
    # Using method of moments to initialize
    mean = np.mean(data)
    var = np.var(data)
    
    # Method of moments estimators for negative binomial
    # mean = n*(1-p)/p, var = n*(1-p)/p^2
    # Solve for p: var/mean = (1-p)/p + 1 => p = mean / var
    if var > mean:  # Ensure overdispersion compared to Poisson
        p_init = mean / var
        p_init = min(max(0.01, p_init), 0.99)  # Ensure p is in (0,1)
        n_init = mean * p_init / (1 - p_init)
        n_init = max(1, n_init)  # Ensure n ≥ 1
    else:
        # Default values if method of moments gives invalid results
        p_init = 0.5
        n_init = mean
    
    # Fine-tune with numerical optimization
    def neg_log_likelihood(params):
        n = max(1, int(params[0]))  # Ensure n is integer ≥ 1
        p = min(max(0.001, params[1]), 0.999)  # Ensure p is in (0,1)
        return -np.sum([nbinom.logpmf(x, n, p) for x in data])
    
    # Use method of moments estimates as starting point
    result = minimize(neg_log_likelihood, [n_init, p_init], 
                      bounds=[(1, 100), (0.001, 0.999)])
    
    n_mle = int(max(1, result.x[0]))  # Round to nearest integer and ensure ≥ 1
    p_mle = result.x[1]
    nbinom_log_likelihood = -result.fun
    
    results['nbinom'] = {
        'params': {'n': n_mle, 'p': p_mle},
        'log_likelihood': nbinom_log_likelihood
    }
    
    # Fit normal distribution
    mu_mle = np.mean(data)
    sigma_mle = np.std(data, ddof=0)  # MLE uses n, not n-1
    normal_log_likelihood = np.sum(norm.logpdf(data, mu_mle, sigma_mle))
    results['normal'] = {
        'params': {'mu': mu_mle, 'sigma': sigma_mle},
        'log_likelihood': normal_log_likelihood
    }
    
    # Fit beta distribution (data must be in [0,1])
    # Normalize data to [0,1] for beta fitting
    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    
    # Use method of moments to estimate beta parameters
    mean = np.mean(normalized_data)
    var = np.var(normalized_data)
    
    if 0 < mean < 1 and 0 < var < mean * (1 - mean):
        # Method of moments estimators for beta
        temp = mean * (1 - mean) / var - 1
        alpha_mle = mean * temp
        beta_mle = (1 - mean) * temp
    else:
        # Default values if method of moments fails
        alpha_mle = 2.0
        beta_mle = 2.0
    
    beta_log_likelihood = np.sum(beta.logpdf(normalized_data, alpha_mle, beta_mle))
    
    # Check if log-likelihood is valid
    if np.isnan(beta_log_likelihood):
        # Handle NaN by using a very negative value
        beta_log_likelihood = -10000.0
    
    results['beta'] = {
        'params': {'alpha': alpha_mle, 'beta': beta_mle},
        'log_likelihood': beta_log_likelihood,
        'normalized': True,
        'data_min': np.min(data),
        'data_max': np.max(data)
    }
    
    return results

def compare_distributions_visually(data, fitting_results, save_dir=None):
    """Generate a plot comparing the fits of different distributions"""
    plt.figure(figsize=(10, 6))
    
    # Create integer bins for count data
    bins = np.arange(np.min(data) - 0.5, np.max(data) + 1.5, 1)
    
    # Plot histogram of the original data
    plt.hist(data, bins=bins, density=True, alpha=0.5, color='lightgray', 
             edgecolor='black', label='Data')
    
    # Create a range of x values for plotting the PMFs
    x = np.arange(max(0, np.min(data) - 2), np.max(data) + 3)
    
    # Plot fitted distributions with consistent colors
    colors = {'poisson': 'blue', 'nbinom': 'red', 'normal': 'green', 'beta': 'orange'}
    
    for dist_name, result in fitting_results.items():
        if dist_name == 'poisson':
            lambda_val = result['params']['lambda']
            pmf = np.array([poisson.pmf(k, lambda_val) for k in x])
            plt.plot(x, pmf, 'b-o', markersize=4, label=f'Poisson(λ={lambda_val:.2f})', linewidth=2)
            
        elif dist_name == 'nbinom':
            n = result['params']['n']
            p = result['params']['p']
            pmf = np.array([nbinom.pmf(k, n, p) for k in x])
            plt.plot(x, pmf, 'r-o', markersize=4, label=f'NegBinom(n={n}, p={p:.2f})', linewidth=2)
            
        elif dist_name == 'normal':
            mu = result['params']['mu']
            sigma = result['params']['sigma']
            # For normal, we need to be careful about discrete data approximation
            x_continuous = np.linspace(np.min(data) - 2, np.max(data) + 2, 1000)
            pdf = norm.pdf(x_continuous, mu, sigma)
            plt.plot(x_continuous, pdf, 'g-', label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})', linewidth=2)
            
        elif dist_name == 'beta':
            # For beta, skip plotting if the fit was poor (indicated by very negative log-likelihood)
            if result['log_likelihood'] > -1000:
                alpha = result['params']['alpha']
                beta_param = result['params']['beta']
                
                # Map x from original scale to [0,1] for beta PDF
                x_continuous = np.linspace(0, 1, 1000)
                pdf = beta.pdf(x_continuous, alpha, beta_param)
                
                # Transform back to original scale
                x_orig = x_continuous * (result['data_max'] - result['data_min']) + result['data_min']
                pdf_orig = pdf / (result['data_max'] - result['data_min'])
                
                plt.plot(x_orig, pdf_orig, 'y-', label=f'Beta(α={alpha:.2f}, β={beta_param:.2f})', linewidth=2)
    
    # Add a note about overdispersion
    mean = np.mean(data)
    var = np.var(data)
    plt.text(0.6, 0.9, f'Variance/Mean = {var/mean:.2f}\n(>1 indicates overdispersion)',
            transform=plt.gca().transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add title and labels
    plt.title('Comparison of Fitted Distributions', fontsize=14)
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Probability Mass/Density', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=10)
    
    # Save the figure if a directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'ex2_distribution_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.close()

def plot_likelihood_contours(data, save_dir=None):
    """
    Generate contour plots of log-likelihood surface for negative binomial distribution
    to better visualize MLE estimation
    """
    # Estimate reasonable parameter ranges around the MLE
    mean = np.mean(data)
    var = np.var(data)
    
    # Method of moments estimators for negative binomial
    if var > mean:
        p_est = mean / var
        p_est = min(max(0.01, p_est), 0.99)
        n_est = mean * p_est / (1 - p_est)
        n_est = max(1, n_est)
    else:
        p_est = 0.5
        n_est = mean
    
    # Define parameter grid ranges
    n_range = np.linspace(max(1, n_est * 0.5), n_est * 1.5, 100)
    p_range = np.linspace(max(0.1, p_est * 0.5), min(0.9, p_est * 1.5), 100)
    
    n_grid, p_grid = np.meshgrid(n_range, p_range)
    log_likelihood = np.zeros_like(n_grid)
    
    # Calculate log-likelihood for each parameter combination
    for i in range(len(p_range)):
        for j in range(len(n_range)):
            n_int = max(1, int(n_grid[i, j]))
            p_val = min(max(0.001, p_grid[i, j]), 0.999)
            
            logpmf_vals = np.array([nbinom.logpmf(x, n_int, p_val) for x in data])
            log_likelihood[i, j] = np.sum(logpmf_vals)
    
    # Normalize log-likelihood
    log_likelihood = log_likelihood - np.max(log_likelihood)
    
    # Create the contour plot
    plt.figure(figsize=(9, 6))
    
    # Plot filled contours
    contour_levels = np.linspace(-20, 0, 20)
    contour = plt.contourf(n_grid, p_grid, log_likelihood, 
                         levels=contour_levels, cmap='viridis')
    
    # Add contour lines
    line_levels = np.linspace(-15, 0, 6)
    lines = plt.contour(n_grid, p_grid, log_likelihood, 
                      levels=line_levels, colors='white', alpha=0.6, linewidths=0.8)
    plt.clabel(lines, inline=True, fontsize=8, fmt='%.1f')
    
    # Find and mark the MLE
    max_idx = np.unravel_index(np.argmax(log_likelihood), log_likelihood.shape)
    mle_n = n_grid[max_idx]
    mle_p = p_grid[max_idx]
    plt.plot(mle_n, mle_p, 'ro', markersize=10, 
           label=f'MLE: n={int(mle_n)}, p={mle_p:.2f}')
    
    # Mark the true parameter values
    plt.plot(5.0, 0.6, 'w*', markersize=12, 
           label='True: n=5, p=0.60')
    
    # Add explanatory text
    plt.text(n_range[0] * 1.1, p_range[-1] * 0.9, 
             'Higher values (yellow)\nindicate parameter combinations\nwith higher likelihood', 
             bbox=dict(facecolor='white', alpha=0.7), fontsize=10)
    
    # Add title and labels
    plt.title('Negative Binomial Log-Likelihood Contours', fontsize=14)
    plt.xlabel('n (Number of Failures)', fontsize=12)
    plt.ylabel('p (Success Probability)', fontsize=12)
    
    # Add colorbar and legend
    cbar = plt.colorbar(contour)
    cbar.set_label('Log-Likelihood', fontsize=12)
    plt.legend(loc='lower right')
    
    plt.grid(alpha=0.3)
    
    # Save the figure if a directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'ex2_nbinom_likelihood_contours.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Likelihood contours saved to {save_path}")
    
    plt.close()

def compare_log_likelihoods(fitting_results, save_dir=None):
    """Create a bar chart comparing the log-likelihoods of different distributions"""
    plt.figure(figsize=(10, 6))
    
    # Extract distribution names and log-likelihoods
    distributions = list(fitting_results.keys())
    log_likelihoods = [result['log_likelihood'] for result in fitting_results.values()]
    
    # Sort by log-likelihood for better visualization
    sorted_indices = np.argsort(log_likelihoods)
    distributions = [distributions[i] for i in sorted_indices]
    log_likelihoods = [log_likelihoods[i] for i in sorted_indices]
    
    # Create the bar chart with nice colors
    bar_colors = ['lightblue', 'lightgreen', 'salmon', 'gold']
    if len(distributions) == 4:
        colors = [bar_colors[i] for i in range(4)]
    else:
        colors = ['skyblue' for _ in distributions]
    
    bars = plt.bar(distributions, log_likelihoods, color=colors, edgecolor='navy')
    
    # Add data labels - fix: handle potential NaN or infinity values
    offset = min(0.01 * abs(min([ll for ll in log_likelihoods if np.isfinite(ll)])), 1.0)
    for bar, ll in zip(bars, log_likelihoods):
        if np.isfinite(ll) and np.isfinite(bar.get_height()):
            plt.text(bar.get_x() + bar.get_width()/2, 
                     bar.get_height() + offset, 
                     f'{ll:.1f}', 
                     ha='center', va='bottom', fontsize=10)
        else:
            # Handle non-finite values (inf, -inf, nan)
            plt.text(bar.get_x() + bar.get_width()/2, 
                     0,  # Place text at the base
                     f'N/A', 
                     ha='center', va='bottom', fontsize=10)
    
    # Add title and labels with simpler design
    plt.title('Comparison of Maximum Log-Likelihoods', fontsize=14)
    plt.xlabel('Distribution', fontsize=12)
    plt.ylabel('Log-Likelihood', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Save the figure if a directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'ex2_log_likelihood_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Log-likelihood comparison saved to {save_path}")
    
    plt.close()

def plot_overdispersion_comparison(data, fitting_results, save_dir=None):
    """
    Create a plot showing how negative binomial captures overdispersion better than Poisson
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate observed variance and mean
    observed_mean = np.mean(data)
    observed_var = np.var(data)
    
    # Get parameters from fitted distributions
    poisson_lambda = fitting_results['poisson']['params']['lambda']
    nbinom_n = fitting_results['nbinom']['params']['n']
    nbinom_p = fitting_results['nbinom']['params']['p']
    
    # Calculate theoretical variance for each distribution
    poisson_var = poisson_lambda  # For Poisson, variance equals mean
    nbinom_var = nbinom_n * (1 - nbinom_p) / (nbinom_p ** 2)
    
    # Create bars for comparison
    labels = ['Observed', 'Poisson', 'Negative Binomial']
    means = [observed_mean, poisson_lambda, observed_mean]  # All should have same mean
    variances = [observed_var, poisson_var, nbinom_var]
    
    # Create grouped bar chart
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_bars = ax.bar(x - width/2, means, width, label='Mean', color='royalblue')
    var_bars = ax.bar(x + width/2, variances, width, label='Variance', color='forestgreen')
    
    # Add variance-to-mean ratio as text
    for i, (m, v) in enumerate(zip(means, variances)):
        ratio = v / m if m > 0 else 0
        ax.text(i, v + 0.1, f'Var/Mean: {ratio:.2f}', ha='center', fontsize=10)
    
    # Add explanatory annotations
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(0, observed_var * 1.2, "Observed data has\nvariance > mean\n(overdispersed)", 
           fontsize=10, ha='center', bbox=props)
    
    ax.text(1, poisson_var * 1.2, "Poisson forces\nvariance = mean\n(can't model overdispersion)", 
           fontsize=10, ha='center', bbox=props)
    
    ax.text(2, nbinom_var * 1.2, "Negative Binomial allows\nvariance > mean\n(captures overdispersion)", 
           fontsize=10, ha='center', bbox=props)
    
    # Customize plot
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Mean-Variance Comparison: Overdispersion', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Save the figure if a directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'ex2_overdispersion_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Overdispersion comparison saved to {save_path}")
    
    plt.close()

def create_simplified_explanation(data, fitting_results, save_dir=None):
    """
    Create a simple visual explaining the key concepts of overdispersion
    and why negative binomial is the best fit for the count data
    """
    # Create 2x1 subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    
    # 1. In the first subplot, show the data histogram with the key distributions
    # Create integer bins for count data
    bins = np.arange(np.min(data) - 0.5, np.max(data) + 1.5, 1)
    
    # Plot histogram
    axs[0].hist(data, bins=bins, density=True, alpha=0.5, color='lightgray', 
               edgecolor='black', label='Data')
    
    # Get parameters from results
    poisson_lambda = fitting_results['poisson']['params']['lambda']
    nbinom_n = fitting_results['nbinom']['params']['n']
    nbinom_p = fitting_results['nbinom']['params']['p']
    
    # Create x range for plotting
    x = np.arange(max(0, np.min(data) - 2), np.max(data) + 3)
    
    # Plot Poisson PMF
    poisson_pmf = np.array([poisson.pmf(k, poisson_lambda) for k in x])
    axs[0].plot(x, poisson_pmf, 'b-o', markersize=4, linewidth=2,
               label=f'Poisson(λ={poisson_lambda:.2f})')
    
    # Plot Negative Binomial PMF
    nbinom_pmf = np.array([nbinom.pmf(k, nbinom_n, nbinom_p) for k in x])
    axs[0].plot(x, nbinom_pmf, 'r-o', markersize=4, linewidth=2,
               label=f'NegBinom(n={nbinom_n}, p={nbinom_p:.2f})')
    
    # Annotate the key differences
    axs[0].annotate('Poisson underestimates\nthe spread of data',
                   xy=(max(x) * 0.8, max(poisson_pmf) * 0.7),
                   xytext=(max(x) * 0.6, max(poisson_pmf) * 0.9),
                   arrowprops=dict(facecolor='blue', shrink=0.05, width=1.5),
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                   fontsize=10)
    
    axs[0].annotate('Negative Binomial captures\nthe wider spread (overdispersion)',
                   xy=(min(x) + 1, nbinom_pmf[1] * 1.2),
                   xytext=(min(x) + 3, nbinom_pmf[1] * 2),
                   arrowprops=dict(facecolor='red', shrink=0.05, width=1.5),
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                   fontsize=10)
    
    # Add explanatory text about overdispersion
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    axs[0].text(0.02, 0.97, 
               f"Overdispersion in Count Data:\n" +
               f"• Observed Mean: {np.mean(data):.2f}\n" +
               f"• Observed Variance: {np.var(data):.2f}\n" +
               f"• Variance/Mean Ratio: {np.var(data)/np.mean(data):.2f} (>1 indicates overdispersion)",
               transform=axs[0].transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
    
    axs[0].set_title('Distribution Fit Comparison', fontsize=14)
    axs[0].set_xlabel('Count Value', fontsize=12)
    axs[0].set_ylabel('Probability Mass', fontsize=12)
    axs[0].grid(alpha=0.3)
    axs[0].legend(fontsize=10)
    
    # 2. In the second subplot, show the theoretical variance-mean relationship
    x_vals = np.linspace(0.5, 10, 100)
    
    # Plot variance = mean line (Poisson constraint)
    axs[1].plot(x_vals, x_vals, 'b-', linewidth=2, label='Poisson (Var = Mean)')
    
    # Plot NB variance with the fitted n parameter
    n_fitted = nbinom_n
    axs[1].plot(x_vals, x_vals + x_vals**2/n_fitted, 'r-', linewidth=2, 
               label=f'Neg. Binomial (n={n_fitted})')
    
    # Plot observed value
    axs[1].plot(np.mean(data), np.var(data), 'ko', markersize=10, 
               label='Observed Data')
    
    # Draw a horizontal line from mean to the NB variance curve
    mean_val = np.mean(data)
    var_val = np.var(data)
    nb_var = mean_val + mean_val**2/n_fitted
    
    axs[1].plot([mean_val, mean_val], [mean_val, var_val], 'k--', alpha=0.7)
    
    # Add annotations explaining the plot
    axs[1].annotate('Poisson constraint:\nvariance = mean',
                   xy=(7, 7),
                   xytext=(8, 5),
                   arrowprops=dict(facecolor='blue', shrink=0.05),
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                   fontsize=10)
    
    axs[1].annotate('Negative Binomial allows\nvariance > mean',
                   xy=(7, 7 + 7**2/n_fitted),
                   xytext=(8, 10),
                   arrowprops=dict(facecolor='red', shrink=0.05),
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                   fontsize=10)
    
    axs[1].annotate('Observed data shows\noverdispersion',
                   xy=(mean_val, var_val),
                   xytext=(mean_val - 1, var_val + 1),
                   arrowprops=dict(facecolor='black', shrink=0.05),
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                   fontsize=10)
    
    axs[1].set_title('Variance-Mean Relationship', fontsize=14)
    axs[1].set_xlabel('Mean', fontsize=12)
    axs[1].set_ylabel('Variance', fontsize=12)
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save the figure if a directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'ex2_simplified_explanation.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Simplified explanation saved to {save_path}")
    
    plt.close()

def generate_answer_materials():
    """Generate answer materials for the MLE visual question"""
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    answer_dir = os.path.join(images_dir, "MLE_Visual_Answer")
    os.makedirs(answer_dir, exist_ok=True)
    
    print("Generating MLE Visual Answer Example 2 materials...")
    
    # Load the mystery data
    mystery_data = load_mystery_data()
    
    # Fit distributions to the data
    fitting_results = fit_distributions(mystery_data)
    
    # Compare distributions visually
    compare_distributions_visually(mystery_data, fitting_results, save_dir=answer_dir)
    
    # Generate contour plot for likelihood surface
    plot_likelihood_contours(mystery_data, save_dir=answer_dir)
    
    # Compare log-likelihoods
    compare_log_likelihoods(fitting_results, save_dir=answer_dir)
    
    # Create overdispersion comparison plot
    plot_overdispersion_comparison(mystery_data, fitting_results, save_dir=answer_dir)
    
    # Create simplified explanation figure
    create_simplified_explanation(mystery_data, fitting_results, save_dir=answer_dir)
    
    # Print the results
    print("\nFitting Results:")
    for dist_name, result in fitting_results.items():
        print(f"\n{dist_name.capitalize()} Distribution:")
        print(f"Parameters: {result['params']}")
        print(f"Log-Likelihood: {result['log_likelihood']:.2f}")
    
    print(f"\nAnswer materials saved to: {answer_dir}")
    
    return fitting_results, answer_dir

if __name__ == "__main__":
    # Generate the answer materials
    fitting_results, answer_dir = generate_answer_materials() 