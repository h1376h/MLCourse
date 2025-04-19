import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, gamma, beta
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def generate_mystery_data(distribution_type='normal', sample_size=100, params=None, random_seed=42):
    """
    Generate data from a mystery distribution
    
    Parameters:
    -----------
    distribution_type : str
        The type of distribution to generate data from ('normal', 'exponential', 'gamma', 'beta')
    sample_size : int
        Number of data points to generate
    params : dict
        Parameters for the chosen distribution
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    data : array-like
        Generated data samples
    true_params : dict
        The true parameters used to generate the data
    """
    np.random.seed(random_seed)
    
    if params is None:
        params = {}
    
    if distribution_type == 'normal':
        mu = params.get('mu', 5.0)
        sigma = params.get('sigma', 1.5)
        data = np.random.normal(mu, sigma, sample_size)
        true_params = {'mu': mu, 'sigma': sigma}
        
    elif distribution_type == 'exponential':
        scale = params.get('scale', 2.0)  # scale = 1/rate
        data = np.random.exponential(scale, sample_size)
        true_params = {'scale': scale, 'rate': 1/scale}
        
    elif distribution_type == 'gamma':
        shape = params.get('shape', 2.0)
        scale = params.get('scale', 1.5)
        data = np.random.gamma(shape, scale, sample_size)
        true_params = {'shape': shape, 'scale': scale}
        
    elif distribution_type == 'beta':
        alpha = params.get('alpha', 2.0)
        beta_param = params.get('beta', 5.0)
        data = np.random.beta(alpha, beta_param, sample_size)
        true_params = {'alpha': alpha, 'beta': beta_param}
    
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")
    
    return data, true_params

def generate_likelihood_surface(data, distribution_type='normal', param_ranges=None):
    """
    Generate a likelihood surface for a given dataset and distribution
    
    Parameters:
    -----------
    data : array-like
        The data to use for likelihood calculation
    distribution_type : str
        Distribution type ('normal', 'exponential', 'gamma', 'beta')
    param_ranges : dict
        Ranges for the parameters to evaluate the likelihood
    
    Returns:
    --------
    param_grid1, param_grid2 : 2D arrays
        Meshgrid of parameter values
    log_likelihood : 2D array
        Log-likelihood values for the parameter combinations
    """
    if param_ranges is None:
        param_ranges = {}
    
    # For normal distribution, we'll plot likelihood for mu and sigma
    if distribution_type == 'normal':
        # Determine reasonable ranges if not provided
        if 'mu_range' not in param_ranges:
            mu_min, mu_max = np.mean(data) - 3*np.std(data), np.mean(data) + 3*np.std(data)
            param_ranges['mu_range'] = (mu_min, mu_max)
        
        if 'sigma_range' not in param_ranges:
            sigma_min, sigma_max = np.std(data) * 0.3, np.std(data) * 2.0
            param_ranges['sigma_range'] = (sigma_min, sigma_max)
        
        mu_range = np.linspace(*param_ranges['mu_range'], 50)
        sigma_range = np.linspace(*param_ranges['sigma_range'], 50)
        
        mu_grid, sigma_grid = np.meshgrid(mu_range, sigma_range)
        log_likelihood = np.zeros_like(mu_grid)
        
        for i in range(len(sigma_range)):
            for j in range(len(mu_range)):
                log_likelihood[i, j] = np.sum(norm.logpdf(data, mu_grid[i, j], sigma_grid[i, j]))
        
        return mu_grid, sigma_grid, log_likelihood
    
    # For exponential distribution, we'll plot likelihood for rate parameter
    elif distribution_type == 'exponential':
        # For exponential, we parametrize by rate (lambda) = 1/scale
        if 'rate_range' not in param_ranges:
            scale_est = np.mean(data)  # MLE for scale is the mean
            rate_est = 1/scale_est
            rate_min, rate_max = rate_est * 0.3, rate_est * 2.0
            param_ranges['rate_range'] = (rate_min, rate_max)
        
        rate_range = np.linspace(*param_ranges['rate_range'], 50)
        scale_range = 1/rate_range
        
        # Create a dummy y-axis (since we only have one parameter)
        dummy_range = np.linspace(0, 1, 50)
        rate_grid, dummy_grid = np.meshgrid(rate_range, dummy_range)
        scale_grid = 1/rate_grid
        
        log_likelihood = np.zeros_like(rate_grid)
        
        for i in range(len(dummy_range)):
            for j in range(len(rate_range)):
                # expon.logpdf takes scale parameter
                log_likelihood[i, j] = np.sum(expon.logpdf(data, scale=scale_grid[i, j]))
        
        return rate_grid, dummy_grid, log_likelihood
    
    # For gamma distribution, we'll plot likelihood for shape and scale
    elif distribution_type == 'gamma':
        if 'shape_range' not in param_ranges:
            # Simple moment-based estimates for initial ranges
            mean = np.mean(data)
            var = np.var(data)
            shape_est = (mean ** 2) / var
            shape_min, shape_max = shape_est * 0.3, shape_est * 2.0
            param_ranges['shape_range'] = (shape_min, shape_max)
        
        if 'scale_range' not in param_ranges:
            mean = np.mean(data)
            var = np.var(data)
            shape_est = (mean ** 2) / var
            scale_est = var / mean
            scale_min, scale_max = scale_est * 0.3, scale_est * 2.0
            param_ranges['scale_range'] = (scale_min, scale_max)
        
        shape_range = np.linspace(*param_ranges['shape_range'], 50)
        scale_range = np.linspace(*param_ranges['scale_range'], 50)
        
        shape_grid, scale_grid = np.meshgrid(shape_range, scale_range)
        log_likelihood = np.zeros_like(shape_grid)
        
        for i in range(len(scale_range)):
            for j in range(len(shape_range)):
                log_likelihood[i, j] = np.sum(gamma.logpdf(data, a=shape_grid[i, j], scale=scale_grid[i, j]))
        
        return shape_grid, scale_grid, log_likelihood
    
    # For beta distribution, we'll plot likelihood for alpha and beta
    elif distribution_type == 'beta':
        # For beta distribution, data must be normalized to [0,1] range
        normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
        
        if 'alpha_range' not in param_ranges:
            # Simple method of moments estimates for normalized data
            mean = np.mean(normalized_data)
            var = np.var(normalized_data)
            
            # Method of moments estimates
            if 0 < mean < 1 and 0 < var < mean * (1 - mean):
                temp = max(0.01, mean * (1 - mean) / var - 1)  # Ensure positive value
                alpha_est = mean * temp
                beta_est = (1 - mean) * temp
            else:
                # Default values if method of moments fails
                alpha_est = 2.0
                beta_est = 5.0
            
            # Ensure reasonable range limits
            alpha_min, alpha_max = max(0.5, alpha_est * 0.3), max(5.0, alpha_est * 3.0)
            param_ranges['alpha_range'] = (alpha_min, alpha_max)
        
        if 'beta_range' not in param_ranges:
            # Use the same method of moments estimates from above
            mean = np.mean(normalized_data)
            var = np.var(normalized_data)
            
            # Method of moments estimates
            if 0 < mean < 1 and 0 < var < mean * (1 - mean):
                temp = max(0.01, mean * (1 - mean) / var - 1)  # Ensure positive value
                alpha_est = mean * temp
                beta_est = (1 - mean) * temp
            else:
                # Default values if method of moments fails
                alpha_est = 2.0
                beta_est = 5.0
            
            # Ensure reasonable range limits
            beta_min, beta_max = max(0.5, beta_est * 0.3), max(5.0, beta_est * 3.0)
            param_ranges['beta_range'] = (beta_min, beta_max)
        
        alpha_range = np.linspace(*param_ranges['alpha_range'], 50)
        beta_range = np.linspace(*param_ranges['beta_range'], 50)
        
        alpha_grid, beta_grid = np.meshgrid(alpha_range, beta_range)
        log_likelihood = np.zeros_like(alpha_grid)
        
        # Ensure data is strictly between 0 and 1 for beta distribution
        epsilon = 1e-10
        normalized_data = np.clip(normalized_data, epsilon, 1-epsilon)
        
        for i in range(len(beta_range)):
            for j in range(len(alpha_range)):
                # Use normalized data for beta distribution
                try:
                    log_likelihood[i, j] = np.sum(beta.logpdf(normalized_data, alpha_grid[i, j], beta_grid[i, j]))
                except:
                    log_likelihood[i, j] = -np.inf  # Handle any numerical issues
        
        return alpha_grid, beta_grid, log_likelihood
    
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")

def plot_mystery_likelihood_surface(param_grid1, param_grid2, log_likelihood, plot_title, param1_name, param2_name, save_path=None):
    """
    Plot the likelihood surface as a 3D plot
    """
    # Normalize log-likelihood for better visualization
    log_likelihood = log_likelihood - np.max(log_likelihood)
    
    # Handle extreme values in log-likelihood (common with beta distribution)
    min_ll = np.max([np.min(log_likelihood), -50])  # Clip very negative values
    log_likelihood = np.maximum(log_likelihood, min_ll)
    
    # Create the 3D surface plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(
        param_grid1, param_grid2, log_likelihood,
        cmap=cm.viridis,
        linewidth=0,
        antialiased=True,
        alpha=0.7
    )
    
    # Find MLE (maximum of log-likelihood)
    max_idx = np.unravel_index(np.argmax(log_likelihood), log_likelihood.shape)
    mle_param1 = param_grid1[max_idx]
    mle_param2 = param_grid2[max_idx]
    
    # Add a point at the MLE
    ax.scatter([mle_param1], [mle_param2], [log_likelihood[max_idx]], 
              color='red', s=100, label=f'Maximum at ({mle_param1:.2f}, {mle_param2:.2f})')
    
    # Add labels and title
    ax.set_xlabel(param1_name, fontsize=12)
    ax.set_ylabel(param2_name, fontsize=12)
    ax.set_zlabel('Log-Likelihood', fontsize=12)
    ax.set_title(plot_title, fontsize=14)
    
    # Set z-axis limits for better visualization
    z_range = log_likelihood[max_idx] - min_ll
    ax.set_zlim(min_ll, log_likelihood[max_idx] + 0.05 * z_range)
    
    # Add a colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Add legend
    ax.legend()
    
    # Adjust viewing angle for better visualization
    ax.view_init(elev=35, azim=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"3D plot saved to {save_path}")
    
    plt.close()

def plot_mystery_likelihood_contour(param_grid1, param_grid2, log_likelihood, plot_title, param1_name, param2_name, save_path=None):
    """
    Plot the likelihood surface as a 2D contour plot for easier interpretation
    """
    # Normalize log-likelihood for better visualization
    log_likelihood = log_likelihood - np.max(log_likelihood)
    
    # Handle extreme values in log-likelihood
    min_ll = np.max([np.min(log_likelihood), -50])  # Clip very negative values
    log_likelihood = np.maximum(log_likelihood, min_ll)
    
    # Create the 2D contour plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the contours
    contour_levels = np.linspace(min_ll, 0, 20)
    contour = ax.contourf(param_grid1, param_grid2, log_likelihood, 
                        levels=contour_levels, cmap='viridis')
    
    # Add contour lines
    line_levels = np.linspace(min_ll, 0, 10)
    ax.contour(param_grid1, param_grid2, log_likelihood, 
              levels=line_levels, colors='white', alpha=0.3, linewidths=0.5)
    
    # Find MLE (maximum of log-likelihood)
    max_idx = np.unravel_index(np.argmax(log_likelihood), log_likelihood.shape)
    mle_param1 = param_grid1[max_idx]
    mle_param2 = param_grid2[max_idx]
    
    # Add a point at the MLE
    ax.plot(mle_param1, mle_param2, 'ro', markersize=10, 
           label=f'Maximum: ({mle_param1:.2f}, {mle_param2:.2f})')
    
    # Add labels and title
    ax.set_xlabel(param1_name, fontsize=12)
    ax.set_ylabel(param2_name, fontsize=12)
    ax.set_title(plot_title, fontsize=14)
    
    # Add a colorbar
    cbar = fig.colorbar(contour)
    cbar.set_label('Log-Likelihood', fontsize=12)
    
    # Add legend
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Contour plot saved to {save_path}")
    
    plt.close()

def plot_mystery_data_histogram(data, save_path=None):
    """
    Plot a histogram of the mystery data with density fit curves for better visualization
    """
    plt.figure(figsize=(10, 6))
    
    # Plot histogram 
    counts, bins, _ = plt.hist(data, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Data')
    
    # Add KDE (Kernel Density Estimation) curve to better visualize the data shape
    x = np.linspace(min(data) * 0.8, max(data) * 1.2, 1000)
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data)
    plt.plot(x, kde(x), 'r-', linewidth=2, label='KDE Estimate')
    
    # Add vertical lines for key statistics
    plt.axvline(np.mean(data), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(data):.2f}')
    plt.axvline(np.median(data), color='green', linestyle='-.', linewidth=2, label=f'Median: {np.median(data):.2f}')
    
    # Add text box with key statistics
    textstr = '\n'.join((
        f'Mean: {np.mean(data):.2f}',
        f'Median: {np.median(data):.2f}',
        f'Std Dev: {np.std(data):.2f}',
        f'Min: {np.min(data):.2f}',
        f'Max: {np.max(data):.2f}',
        f'Skewness: {float(np.mean(((data - np.mean(data)) / np.std(data)) ** 3)):.2f}'
    ))
    
    # Place the text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.title('Mystery Distribution Data', fontsize=14)
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Histogram saved to {save_path}")
    
    plt.close()

def plot_fitted_distributions(data, save_path=None):
    """
    Plot the data histogram with the four candidate distributions fitted to it
    """
    plt.figure(figsize=(10, 6))
    
    # Plot histogram
    counts, bins, _ = plt.hist(data, bins=30, density=True, alpha=0.5, color='lightgray', 
                              edgecolor='black', label='Data')
    
    # Fit and plot candidate distributions
    x = np.linspace(min(data) * 0.8, max(data) * 1.2, 1000)
    
    # Normal distribution
    mu, sigma = norm.fit(data)
    plt.plot(x, norm.pdf(x, mu, sigma), 'b-', linewidth=2, label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')
    
    # Exponential distribution (fit to data shifted to start at 0 if needed)
    if np.min(data) < 0:
        shifted_data = data - np.min(data)
    else:
        shifted_data = data
    scale = np.mean(shifted_data)
    plt.plot(x, expon.pdf(x - np.min(data), scale=scale), 'g-', linewidth=2, 
             label=f'Exponential(λ={1/scale:.2f})')
    
    # Gamma distribution
    # Method of moments estimator for gamma parameters
    mean = np.mean(data)
    var = np.var(data)
    shape = mean**2 / var
    scale = var / mean
    plt.plot(x, gamma.pdf(x, a=shape, scale=scale), 'r-', linewidth=2, 
             label=f'Gamma(k={shape:.2f}, θ={scale:.2f})')
    
    # Beta distribution (need to normalize data to [0,1])
    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    
    # Ensure data is strictly in (0,1) range to avoid beta fitting errors
    epsilon = 1e-10
    normalized_data = np.clip(normalized_data, epsilon, 1-epsilon)
    
    try:
        a, b = beta.fit(normalized_data, floc=0, fscale=1)
        # Transform the fitted beta PDF back to original scale for plotting
        beta_x = np.linspace(epsilon, 1-epsilon, 1000)
        beta_y = beta.pdf(beta_x, a, b)
        # Scale the PDF back to original data range
        orig_x = beta_x * (np.max(data) - np.min(data)) + np.min(data)
        orig_y = beta_y / (np.max(data) - np.min(data))
        plt.plot(orig_x, orig_y, 'y-', linewidth=2, label=f'Beta(α={a:.2f}, β={b:.2f})')
    except Exception as e:
        print(f"Warning: Could not fit beta distribution: {e}")
    
    plt.title('Fitted Distributions Comparison', fontsize=14)
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Fitted distributions plot saved to {save_path}")
    
    plt.close()

def generate_mle_visual_question():
    """Generate the visual MLE question with its materials"""
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    question_dir = os.path.join(images_dir, "MLE_Visual_Question")
    os.makedirs(question_dir, exist_ok=True)
    
    print("Generating MLE Visual Question Example 1 materials...")
    
    # 1. Generate data from a mystery distribution (gamma in this case, but don't reveal it)
    # Using gamma distribution with shape=2, scale=1.5
    mystery_data, true_params = generate_mystery_data(
        distribution_type='gamma',
        sample_size=200,
        params={'shape': 2.0, 'scale': 1.5},
        random_seed=42
    )
    
    # 2. Plot histogram of the mystery data
    histogram_path = os.path.join(question_dir, "ex1_mystery_data_histogram.png")
    plot_mystery_data_histogram(mystery_data, save_path=histogram_path)
    
    # 3. Plot fitted distributions for reference (new)
    fitted_path = os.path.join(question_dir, "ex1_fitted_distributions.png")
    plot_fitted_distributions(mystery_data, save_path=fitted_path)
    
    # 4. Generate likelihood surfaces and contour plots for different distributions
    
    # Normal distribution 
    normal_param1, normal_param2, normal_ll = generate_likelihood_surface(
        mystery_data, 
        distribution_type='normal'
    )
    # 3D surface
    normal_surface_path = os.path.join(question_dir, "ex1_normal_likelihood_surface.png")
    plot_mystery_likelihood_surface(
        normal_param1, normal_param2, normal_ll,
        "Normal Distribution Log-Likelihood Surface",
        "μ (Mean)", "σ (Standard Deviation)",
        save_path=normal_surface_path
    )
    # 2D contour
    normal_contour_path = os.path.join(question_dir, "ex1_normal_likelihood_contour.png")
    plot_mystery_likelihood_contour(
        normal_param1, normal_param2, normal_ll,
        "Normal Distribution Log-Likelihood Contours",
        "μ (Mean)", "σ (Standard Deviation)",
        save_path=normal_contour_path
    )
    
    # Exponential distribution
    exp_param1, exp_param2, exp_ll = generate_likelihood_surface(
        mystery_data, 
        distribution_type='exponential'
    )
    # 3D surface
    exp_surface_path = os.path.join(question_dir, "ex1_exponential_likelihood_surface.png")
    plot_mystery_likelihood_surface(
        exp_param1, exp_param2, exp_ll,
        "Exponential Distribution Log-Likelihood Surface",
        "λ (Rate)", "Dummy Parameter (Ignore)",
        save_path=exp_surface_path
    )
    # For exponential, no need for 2D contour as it's a 1D parameter
    
    # Gamma distribution (the correct one)
    gamma_param1, gamma_param2, gamma_ll = generate_likelihood_surface(
        mystery_data, 
        distribution_type='gamma'
    )
    # 3D surface
    gamma_surface_path = os.path.join(question_dir, "ex1_gamma_likelihood_surface.png")
    plot_mystery_likelihood_surface(
        gamma_param1, gamma_param2, gamma_ll,
        "Gamma Distribution Log-Likelihood Surface",
        "k (Shape)", "θ (Scale)",
        save_path=gamma_surface_path
    )
    # 2D contour
    gamma_contour_path = os.path.join(question_dir, "ex1_gamma_likelihood_contour.png")
    plot_mystery_likelihood_contour(
        gamma_param1, gamma_param2, gamma_ll,
        "Gamma Distribution Log-Likelihood Contours",
        "k (Shape)", "θ (Scale)",
        save_path=gamma_contour_path
    )
    
    # Beta distribution
    beta_param1, beta_param2, beta_ll = generate_likelihood_surface(
        mystery_data, 
        distribution_type='beta'
    )
    # 3D surface
    beta_surface_path = os.path.join(question_dir, "ex1_beta_likelihood_surface.png")
    plot_mystery_likelihood_surface(
        beta_param1, beta_param2, beta_ll,
        "Beta Distribution Log-Likelihood Surface",
        "α (Alpha)", "β (Beta)",
        save_path=beta_surface_path
    )
    # 2D contour
    beta_contour_path = os.path.join(question_dir, "ex1_beta_likelihood_contour.png")
    plot_mystery_likelihood_contour(
        beta_param1, beta_param2, beta_ll,
        "Beta Distribution Log-Likelihood Contours",
        "α (Alpha)", "β (Beta)",
        save_path=beta_contour_path
    )
    
    print(f"\nAll question materials generated in: {question_dir}")
    print(f"True distribution: Gamma with parameters {true_params}")
    
    # Find the MLE for the correct distribution (gamma)
    max_idx = np.unravel_index(np.argmax(gamma_ll), gamma_ll.shape)
    mle_shape = gamma_param1[max_idx]
    mle_scale = gamma_param2[max_idx]
    print(f"MLE estimates: shape = {mle_shape:.2f}, scale = {mle_scale:.2f}")
    
    return question_dir, true_params

if __name__ == "__main__":
    question_dir, true_params = generate_mle_visual_question() 