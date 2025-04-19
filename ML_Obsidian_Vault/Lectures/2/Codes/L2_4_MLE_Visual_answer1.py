import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, gamma, beta
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.special import logsumexp
from scipy.optimize import minimize

def load_mystery_data():
    """Generate the same mystery data as in the question script"""
    np.random.seed(42)
    return np.random.gamma(2.0, 1.5, 200)

def fit_distributions(data):
    """
    Fit various distributions to the data using MLE and calculate log-likelihoods
    """
    results = {}
    
    # Fit normal distribution
    mu_mle = np.mean(data)
    sigma_mle = np.std(data, ddof=0)  # MLE uses n, not n-1
    normal_log_likelihood = np.sum(norm.logpdf(data, mu_mle, sigma_mle))
    results['normal'] = {
        'params': {'mu': mu_mle, 'sigma': sigma_mle},
        'log_likelihood': normal_log_likelihood
    }
    
    # Fit exponential distribution
    # MLE for exponential is 1/mean for the rate parameter
    rate_mle = 1 / np.mean(data)
    scale_mle = 1 / rate_mle
    exp_log_likelihood = np.sum(expon.logpdf(data, scale=scale_mle))
    results['exponential'] = {
        'params': {'rate': rate_mle, 'scale': scale_mle},
        'log_likelihood': exp_log_likelihood
    }
    
    # Fit gamma distribution (using method of moments to initialize)
    mean = np.mean(data)
    var = np.var(data)
    shape_init = (mean ** 2) / var
    scale_init = var / mean
    
    # Fine-tune with numerical optimization
    def neg_log_likelihood(params):
        shape, scale = params
        return -np.sum(gamma.logpdf(data, a=shape, scale=scale))
    
    result = minimize(neg_log_likelihood, [shape_init, scale_init], 
                      bounds=[(0.1, 10), (0.1, 10)])
    
    shape_mle, scale_mle = result.x
    gamma_log_likelihood = -result.fun
    
    results['gamma'] = {
        'params': {'shape': shape_mle, 'scale': scale_mle},
        'log_likelihood': gamma_log_likelihood
    }
    
    # Fit beta distribution (data must be in [0,1])
    # Normalize data to [0,1] for beta fitting
    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    
    # Use method of moments to estimate beta parameters
    mean = np.mean(normalized_data)
    var = np.var(normalized_data)
    
    # Method of moments estimators for beta
    if 0 < mean < 1 and 0 < var < mean * (1 - mean):
        temp = mean * (1 - mean) / var - 1
        alpha_init = mean * temp
        beta_init = (1 - mean) * temp
    else:
        # Default values if method of moments fails
        alpha_init = 2.0
        beta_init = 5.0
    
    # Fine-tune with numerical optimization
    def beta_neg_log_likelihood(params):
        alpha, beta_param = params
        if alpha <= 0 or beta_param <= 0:
            return 1e10  # Large penalty for invalid parameters
        return -np.sum(beta.logpdf(normalized_data, alpha, beta_param))
    
    result = minimize(beta_neg_log_likelihood, [alpha_init, beta_init], 
                      bounds=[(0.1, 50), (0.1, 50)])
    
    alpha_mle, beta_mle = result.x
    beta_log_likelihood = -result.fun
    
    # Check if log-likelihood is valid
    if np.isnan(beta_log_likelihood):
        # Handle NaN by using a simpler approach
        beta_log_likelihood = -1000.0  # Set to a very negative value to indicate poor fit
    
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
    
    # Plot histogram of the original data
    counts, bins, _ = plt.hist(data, bins=30, density=True, alpha=0.5, color='lightgray', 
                               edgecolor='black', label='Data')
    
    # Add KDE for better visualization of data shape
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data)
    x_kde = np.linspace(min(data) * 0.8, max(data) * 1.2, 1000)
    plt.plot(x_kde, kde(x_kde), 'k--', linewidth=1.5, label='KDE Estimate', alpha=0.7)
    
    # Create a range of x values for plotting the PDFs
    x = np.linspace(min(data) * 0.8, max(data) * 1.2, 1000)
    
    # Plot fitted distributions
    colors = {'normal': 'blue', 'exponential': 'green', 'gamma': 'red', 'beta': 'orange'}
    linestyles = {'normal': '-', 'exponential': '-', 'gamma': '-', 'beta': '-'}
    
    for dist_name, result in fitting_results.items():
        if dist_name == 'normal':
            mu = result['params']['mu']
            sigma = result['params']['sigma']
            pdf = norm.pdf(x, mu, sigma)
            plt.plot(x, pdf, color=colors[dist_name], linestyle=linestyles[dist_name],
                     label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})', linewidth=2)
            
        elif dist_name == 'exponential':
            scale = result['params']['scale']
            pdf = expon.pdf(x, scale=scale)
            plt.plot(x, pdf, color=colors[dist_name], linestyle=linestyles[dist_name],
                     label=f'Exponential(λ={1/scale:.2f})', linewidth=2)
            
        elif dist_name == 'gamma':
            shape = result['params']['shape']
            scale = result['params']['scale']
            pdf = gamma.pdf(x, a=shape, scale=scale)
            plt.plot(x, pdf, color=colors[dist_name], linestyle=linestyles[dist_name],
                     label=f'Gamma(k={shape:.2f}, θ={scale:.2f})', linewidth=2)
            
        elif dist_name == 'beta':
            # For beta, we need to transform back if data was normalized
            if result.get('normalized', False):
                alpha = result['params']['alpha']
                beta_param = result['params']['beta']
                
                # Create proper range for beta distribution visualization
                x_beta = np.linspace(0, 1, 1000)
                # Get PDF on [0,1] scale
                pdf_beta = beta.pdf(x_beta, alpha, beta_param)
                
                # Transform to original scale for plotting
                x_orig = x_beta * (result['data_max'] - result['data_min']) + result['data_min']
                # Scale PDF according to change of variables formula
                pdf_orig = pdf_beta / (result['data_max'] - result['data_min'])
                
                plt.plot(x_orig, pdf_orig, color=colors[dist_name], linestyle=linestyles[dist_name],
                         label=f'Beta(α={alpha:.2f}, β={beta_param:.2f})', linewidth=2)
    
    # Add title and labels with simplified design
    plt.title('Comparison of Fitted Distributions', fontsize=14)
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=10)
    
    # Save the figure if a directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'ex1_distribution_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.close()

def plot_likelihood_contours(data, save_dir=None):
    """
    Generate contour plots of log-likelihood surfaces for gamma distribution
    to better visualize MLE estimation
    """
    # Calculate log-likelihood on a grid for gamma distribution
    shape_range = np.linspace(1.0, 3.0, 100)  # Range around the true value of 2.0
    scale_range = np.linspace(0.8, 2.2, 100)  # Range around the true value of 1.5
    
    shape_grid, scale_grid = np.meshgrid(shape_range, scale_range)
    log_likelihood = np.zeros_like(shape_grid)
    
    for i in range(len(scale_range)):
        for j in range(len(shape_range)):
            log_likelihood[i, j] = np.sum(gamma.logpdf(data, a=shape_grid[i, j], scale=scale_grid[i, j]))
    
    # Normalize log-likelihood
    log_likelihood = log_likelihood - np.max(log_likelihood)
    
    # Create the contour plot
    plt.figure(figsize=(9, 6))
    
    # Plot filled contours
    contour_levels = np.linspace(-20, 0, 20)
    contour = plt.contourf(shape_grid, scale_grid, log_likelihood, 
                         levels=contour_levels, cmap='viridis')
    
    # Add contour lines
    line_levels = np.linspace(-15, 0, 6)
    lines = plt.contour(shape_grid, scale_grid, log_likelihood, 
                      levels=line_levels, colors='white', alpha=0.6, linewidths=0.8)
    plt.clabel(lines, inline=True, fontsize=8, fmt='%.1f')
    
    # Find and mark the MLE
    max_idx = np.unravel_index(np.argmax(log_likelihood), log_likelihood.shape)
    mle_shape = shape_grid[max_idx]
    mle_scale = scale_grid[max_idx]
    plt.plot(mle_shape, mle_scale, 'ro', markersize=10, 
           label=f'MLE: k={mle_shape:.2f}, θ={mle_scale:.2f}')
    
    # Mark the true parameter values
    plt.plot(2.0, 1.5, 'w*', markersize=12, 
           label='True: k=2.00, θ=1.50')
    
    # Add explanatory text
    plt.text(1.2, 2.0, 'Higher values (yellow)\nindicate parameter combinations\nwith higher likelihood', 
             bbox=dict(facecolor='white', alpha=0.7), fontsize=10)
    
    # Add title and labels
    plt.title('Gamma Distribution Log-Likelihood Contours', fontsize=14)
    plt.xlabel('Shape Parameter (k)', fontsize=12)
    plt.ylabel('Scale Parameter (θ)', fontsize=12)
    
    # Add colorbar and legend
    cbar = plt.colorbar(contour)
    cbar.set_label('Log-Likelihood', fontsize=12)
    plt.legend(loc='lower right')
    
    plt.grid(alpha=0.3)
    
    # Save the figure if a directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'ex1_gamma_likelihood_contours.png')
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
    
    # Add data labels
    for bar, ll in zip(bars, log_likelihoods):
        plt.text(bar.get_x() + bar.get_width()/2, 
                 bar.get_height() + (abs(min(log_likelihoods)) * 0.01), 
                 f'{ll:.1f}', 
                 ha='center', va='bottom', fontsize=10)
    
    # Add title and labels with simpler design
    plt.title('Comparison of Maximum Log-Likelihoods', fontsize=14)
    plt.xlabel('Distribution', fontsize=12)
    plt.ylabel('Log-Likelihood', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add explanatory text
    plt.text(0, max(log_likelihoods) * 0.9, 
             'Higher values indicate better fit\nGamma has the highest log-likelihood',
             bbox=dict(facecolor='white', alpha=0.7), fontsize=10)
    
    # Save the figure if a directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'ex1_log_likelihood_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Log-likelihood comparison saved to {save_path}")
    
    plt.close()

def create_simplified_explanation(data, fitting_results, save_dir=None):
    """
    Create a simple visual explaining why gamma is the best fit
    for the mystery data
    """
    plt.figure(figsize=(10, 8))
    
    # Create 2x2 subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 9))
    
    # 1. In the first subplot, show the data histogram with the gamma fit
    axs[0].hist(data, bins=30, density=True, alpha=0.5, color='lightgray', 
               edgecolor='black', label='Data Histogram')
    
    # Get gamma parameters from results
    shape = fitting_results['gamma']['params']['shape']
    scale = fitting_results['gamma']['params']['scale']
    
    # Plot the gamma PDF
    x = np.linspace(min(data) * 0.8, max(data) * 1.2, 1000)
    gamma_pdf = gamma.pdf(x, a=shape, scale=scale)
    axs[0].plot(x, gamma_pdf, 'r-', linewidth=2, 
               label=f'Gamma(k={shape:.2f}, θ={scale:.2f})')
    
    # Add visual indicators of key gamma distribution properties
    mode = (shape - 1) * scale if shape > 1 else 0
    mean = shape * scale
    std_dev = np.sqrt(shape * scale**2)
    
    # Mark the mode, mean on the plot
    axs[0].axvline(mode, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                 label=f'Mode: {mode:.2f}')
    axs[0].axvline(mean, color='blue', linestyle='--', linewidth=1.5, alpha=0.7,
                 label=f'Mean: {mean:.2f}')
    
    # Add annotation about key properties of gamma distribution
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    axs[0].text(0.02, 0.97, 
               "Key Properties of Gamma Distribution:\n" +
               f"• Right-skewed (Mode < Mean)\n" +
               f"• Supports positive values only\n" +
               f"• Flexible shape parameter (k={shape:.2f})\n" +
               f"• Scale parameter controls spread (θ={scale:.2f})",
               transform=axs[0].transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
    
    axs[0].set_title('Gamma Distribution Fit to Mystery Data', fontsize=14)
    axs[0].set_xlabel('Value', fontsize=12)
    axs[0].set_ylabel('Density', fontsize=12)
    axs[0].grid(alpha=0.3)
    axs[0].legend(fontsize=10, loc='upper right')
    
    # 2. In the second subplot, show a bar chart of log-likelihoods with explanations
    distributions = list(fitting_results.keys())
    log_likelihoods = [result['log_likelihood'] for result in fitting_results.values()]
    
    # Sort by log-likelihood
    sorted_indices = np.argsort(log_likelihoods)
    distributions = [distributions[i] for i in sorted_indices]
    log_likelihoods = [log_likelihoods[i] for i in sorted_indices]
    
    # Create the bar chart
    bar_colors = ['lightgray', 'lightgreen', 'lightblue', 'salmon']
    bars = axs[1].bar(distributions, log_likelihoods, color=bar_colors)
    
    # Add data labels
    for bar, ll in zip(bars, log_likelihoods):
        axs[1].text(bar.get_x() + bar.get_width()/2, 
                   bar.get_height() + abs(min(log_likelihoods)) * 0.01, 
                   f'{ll:.1f}', 
                   ha='center', va='bottom', fontsize=10)
    
    # Add explanations about why each distribution fits or doesn't fit
    explanations = {
        'beta': "Beta is defined on [0,1]; requires normalization",
        'exponential': "Exponential is too restrictive (only one parameter)",
        'normal': "Normal allows negative values & is symmetric",
        'gamma': "Gamma is right-skewed & positive values only"
    }
    
    # Add the explanations as annotations
    for i, dist in enumerate(distributions):
        y_pos = log_likelihoods[i] / 2  # Position text in middle of bar
        axs[1].text(i, y_pos, explanations[dist], ha='center', va='center', 
                   fontsize=9, rotation=90, color='black')
    
    axs[1].set_title('Log-Likelihood Comparison with Explanations', fontsize=14)
    axs[1].set_xlabel('Distribution', fontsize=12)
    axs[1].set_ylabel('Log-Likelihood', fontsize=12)
    axs[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure if a directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'ex1_simplified_explanation.png')
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
    
    print("Generating MLE Visual Answer Example 1 materials...")
    
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
    
    # Create simplified explanation
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