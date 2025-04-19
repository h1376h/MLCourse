import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson, nbinom, beta
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def generate_mystery_data(distribution_type='poisson', sample_size=150, params=None, random_seed=42):
    """
    Generate data from a mystery distribution
    
    Parameters:
    -----------
    distribution_type : str
        The type of distribution to generate data from ('poisson', 'nbinom', 'norm', 'beta')
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
    
    if distribution_type == 'poisson':
        mu = params.get('mu', 4.0)
        data = np.random.poisson(mu, sample_size)
        true_params = {'mu': mu}
        
    elif distribution_type == 'nbinom':
        n = params.get('n', 5)
        p = params.get('p', 0.6)
        data = np.random.negative_binomial(n, p, sample_size)
        true_params = {'n': n, 'p': p}
        
    elif distribution_type == 'norm':
        mu = params.get('mu', 5.0)
        sigma = params.get('sigma', 1.5)
        data = np.random.normal(mu, sigma, sample_size)
        true_params = {'mu': mu, 'sigma': sigma}
        
    elif distribution_type == 'beta':
        alpha = params.get('alpha', 2.0)
        beta_param = params.get('beta', 5.0)
        data = np.random.beta(alpha, beta_param, sample_size)
        true_params = {'alpha': alpha, 'beta': beta_param}
    
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")
    
    return data, true_params

def generate_likelihood_surface(data, distribution_type='poisson', param_ranges=None):
    """
    Generate a likelihood surface for a given dataset and distribution
    
    Parameters:
    -----------
    data : array-like
        The data to use for likelihood calculation
    distribution_type : str
        Distribution type ('poisson', 'nbinom', 'norm', 'beta')
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
    
    # For Poisson distribution, we'll plot likelihood for mu
    if distribution_type == 'poisson':
        # Determine reasonable ranges if not provided
        if 'mu_range' not in param_ranges:
            mu_min, mu_max = max(0.5, np.mean(data) * 0.5), np.mean(data) * 1.5
            param_ranges['mu_range'] = (mu_min, mu_max)
        
        mu_range = np.linspace(*param_ranges['mu_range'], 50)
        
        # Create a dummy y-axis (since we only have one parameter)
        dummy_range = np.linspace(0, 1, 50)
        mu_grid, dummy_grid = np.meshgrid(mu_range, dummy_range)
        
        log_likelihood = np.zeros_like(mu_grid)
        
        for i in range(len(dummy_range)):
            for j in range(len(mu_range)):
                log_likelihood[i, j] = np.sum(poisson.logpmf(data, mu_grid[i, j]))
        
        return mu_grid, dummy_grid, log_likelihood
    
    # For negative binomial distribution, we'll plot likelihood for n and p
    elif distribution_type == 'nbinom':
        # Determine reasonable ranges if not provided
        if 'n_range' not in param_ranges:
            # For n (number of successes), we'll use a range around the mean / (1 - p)
            # where p is estimated from mean and variance
            mean = np.mean(data)
            var = np.var(data)
            
            # Estimate p and n using method of moments
            # For negative binomial: mean = n*(1-p)/p, variance = n*(1-p)/p^2
            if var > mean:
                # Solve for p: var/mean = (1-p)/p -> p = mean/var
                p_est = mean / var
                # Clamp to valid range
                p_est = min(max(0.05, p_est), 0.95)
                n_est = mean * p_est / (1 - p_est)
                n_est = max(1, n_est)
            else:
                # Default values if method of moments fails
                p_est = 0.5
                n_est = mean
            
            n_min, n_max = max(1, n_est * 0.5), max(10, n_est * 2.0)
            param_ranges['n_range'] = (n_min, n_max)
        
        if 'p_range' not in param_ranges:
            # For p (probability), we'll use a reasonable range
            mean = np.mean(data)
            var = np.var(data)
            if var > mean:
                p_est = mean / var
                p_est = min(max(0.05, p_est), 0.95)
            else:
                p_est = 0.5
            
            p_min, p_max = max(0.1, p_est * 0.5), min(0.9, p_est * 1.5)
            param_ranges['p_range'] = (p_min, p_max)
        
        n_range = np.linspace(*param_ranges['n_range'], 50)
        p_range = np.linspace(*param_ranges['p_range'], 50)
        
        n_grid, p_grid = np.meshgrid(n_range, p_range)
        log_likelihood = np.zeros_like(n_grid)
        
        for i in range(len(p_range)):
            for j in range(len(n_range)):
                # Round n to integer and ensure it's at least 1
                n_int = max(1, int(n_grid[i, j]))
                # Ensure p is between 0 and 1
                p_val = min(max(0.001, p_grid[i, j]), 0.999)
                
                # Calculate log likelihood using negative binomial PMF
                logpmf_vals = np.array([nbinom.logpmf(x, n_int, p_val) for x in data])
                log_likelihood[i, j] = np.sum(logpmf_vals)
        
        return n_grid, p_grid, log_likelihood
    
    # For normal distribution, we'll plot likelihood for mu and sigma
    elif distribution_type == 'norm':
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
    
    # For beta distribution, we'll plot likelihood for alpha and beta
    elif distribution_type == 'beta':
        # For beta distribution, data must be normalized to [0,1] range
        normalized_data = data
        if np.min(data) < 0 or np.max(data) > 1:
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
    Plot a histogram of the mystery data with better visualization
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate bin width for integer data
    if np.all(np.equal(np.mod(data, 1), 0)):  # Check if all data are integers
        bins = np.arange(np.min(data) - 0.5, np.max(data) + 1.5, 1)
        plt.hist(data, bins=bins, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Data')
    else:
        plt.hist(data, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Data')
    
    # Add vertical lines for key statistics
    plt.axvline(np.mean(data), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(data):.2f}')
    plt.axvline(np.median(data), color='green', linestyle='-.', linewidth=2, label=f'Median: {np.median(data):.2f}')
    
    # Add text box with key statistics
    textstr = '\n'.join((
        f'Mean: {np.mean(data):.2f}',
        f'Median: {np.median(data):.2f}',
        f'Std Dev: {np.std(data):.2f}',
        f'Variance: {np.var(data):.2f}',
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
    
    # Calculate bin width for integer data
    if np.all(np.equal(np.mod(data, 1), 0)):  # Check if all data are integers
        bins = np.arange(np.min(data) - 0.5, np.max(data) + 1.5, 1)
        plt.hist(data, bins=bins, density=True, alpha=0.5, color='lightgray', 
                edgecolor='black', label='Data')
    else:
        plt.hist(data, bins=30, density=True, alpha=0.5, color='lightgray', 
                edgecolor='black', label='Data')
    
    # Create x values for plotting
    x = np.arange(max(0, np.min(data) - 2), np.max(data) + 3)
    
    # Poisson distribution
    lambda_param = np.mean(data)  # MLE for Poisson
    pmf_poisson = np.array([poisson.pmf(k, lambda_param) for k in x])
    plt.plot(x, pmf_poisson, 'b-o', linewidth=2, markersize=4, 
             label=f'Poisson(λ={lambda_param:.2f})')
    
    # Negative binomial
    # Estimate parameters using method of moments
    mean = np.mean(data)
    var = np.var(data)
    # For negative binomial: mean = n*(1-p)/p, variance = n*(1-p)/p^2
    if var > mean:
        p_est = mean / var
        p_est = min(max(0.01, p_est), 0.99)
        n_est = max(1, int(mean * p_est / (1 - p_est)))
    else:
        p_est = 0.5
        n_est = max(1, int(mean))
    
    pmf_nbinom = np.array([nbinom.pmf(k, n_est, p_est) for k in x])
    plt.plot(x, pmf_nbinom, 'r-o', linewidth=2, markersize=4, 
             label=f'NegBinom(n={n_est}, p={p_est:.2f})')
    
    # Normal distribution
    mu, sigma = norm.fit(data)
    x_continuous = np.linspace(np.min(data) - 2, np.max(data) + 2, 1000)
    pdf_normal = norm.pdf(x_continuous, mu, sigma)
    plt.plot(x_continuous, pdf_normal, 'g-', linewidth=2, 
             label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')
    
    # Beta distribution (need to normalize data to [0,1])
    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    try:
        a, b = beta.fit(normalized_data, floc=0, fscale=1)
        # Transform the fitted beta PDF back to original scale for plotting
        beta_x = np.linspace(0, 1, 1000)
        beta_y = beta.pdf(beta_x, a, b)
        # Scale the PDF back to original data range
        orig_x = beta_x * (np.max(data) - np.min(data)) + np.min(data)
        orig_y = beta_y / (np.max(data) - np.min(data))
        plt.plot(orig_x, orig_y, 'y-', linewidth=2, 
                label=f'Beta(α={a:.2f}, β={b:.2f})')
    except:
        print("Beta fit failed, skipping plot")
    
    # Add variance/mean comparison text
    plt.text(0.5, 0.9, f'Variance/Mean ratio: {var/mean:.2f}\n(>1 indicates overdispersion)',
            transform=plt.gca().transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.title('Fitted Distributions Comparison', fontsize=14)
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Probability Mass/Density', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Fitted distributions plot saved to {save_path}")
    
    plt.close()

def plot_variance_vs_mean(save_path=None):
    """Create a simple visual showing the variance-mean relationship for Poisson vs NegBinom"""
    plt.figure(figsize=(8, 6))
    
    # Plot the variance = mean line (Poisson)
    x = np.linspace(0, 10, 100)
    plt.plot(x, x, 'b-', linewidth=2, label='Poisson (Var = Mean)')
    
    # Plot some example Negative Binomial variance curves
    for n, color, label in [(1, 'r', 'n=1'), (5, 'g', 'n=5'), (10, 'purple', 'n=10')]:
        # For NegBin: Var = Mean + Mean²/n
        plt.plot(x, x + x**2/n, color=color, linestyle='-', linewidth=2, 
                label=f'NegBinom {label} (Var > Mean)')
    
    plt.xlabel('Mean', fontsize=12)
    plt.ylabel('Variance', fontsize=12)
    plt.title('Variance-Mean Relationship', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add explanation text
    plt.text(1, 8, 'Negative Binomial can model overdispersed count data\nwhere variance exceeds the mean', 
             fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Variance vs. mean plot saved to {save_path}")
    
    plt.close()

def generate_mle_visual_question():
    """Generate the visual MLE question with its materials"""
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    question_dir = os.path.join(images_dir, "MLE_Visual_Question")
    os.makedirs(question_dir, exist_ok=True)
    
    print("Generating MLE Visual Question Example 2 materials...")
    
    # 1. Generate data from a mystery distribution (negative binomial in this case, but don't reveal it)
    mystery_data, true_params = generate_mystery_data(
        distribution_type='nbinom',
        sample_size=150,
        params={'n': 5, 'p': 0.6},
        random_seed=42
    )
    
    # 2. Plot histogram of the mystery data
    histogram_path = os.path.join(question_dir, "ex2_mystery_data_histogram.png")
    plot_mystery_data_histogram(mystery_data, save_path=histogram_path)
    
    # 3. Plot fitted distributions for reference (new)
    fitted_path = os.path.join(question_dir, "ex2_fitted_distributions.png")
    plot_fitted_distributions(mystery_data, save_path=fitted_path)
    
    # 4. Create a conceptual plot about variance vs mean (new)
    variance_mean_path = os.path.join(question_dir, "ex2_variance_mean_relationship.png")
    plot_variance_vs_mean(save_path=variance_mean_path)
    
    # 5. Generate likelihood surfaces and contour plots for different distributions
    
    # Poisson distribution
    poisson_param1, poisson_param2, poisson_ll = generate_likelihood_surface(
        mystery_data, 
        distribution_type='poisson'
    )
    # 3D surface
    poisson_surface_path = os.path.join(question_dir, "ex2_poisson_likelihood_surface.png")
    plot_mystery_likelihood_surface(
        poisson_param1, poisson_param2, poisson_ll,
        "Poisson Distribution Log-Likelihood Surface",
        "μ (Rate)", "Dummy Parameter (Ignore)",
        save_path=poisson_surface_path
    )
    # For Poisson, no need for 2D contour as it's a 1D parameter
    
    # Negative binomial distribution (the correct one)
    nbinom_param1, nbinom_param2, nbinom_ll = generate_likelihood_surface(
        mystery_data, 
        distribution_type='nbinom'
    )
    # 3D surface
    nbinom_surface_path = os.path.join(question_dir, "ex2_nbinom_likelihood_surface.png")
    plot_mystery_likelihood_surface(
        nbinom_param1, nbinom_param2, nbinom_ll,
        "Negative Binomial Distribution Log-Likelihood Surface",
        "n (Number of Successes)", "p (Success Probability)",
        save_path=nbinom_surface_path
    )
    # 2D contour
    nbinom_contour_path = os.path.join(question_dir, "ex2_nbinom_likelihood_contour.png")
    plot_mystery_likelihood_contour(
        nbinom_param1, nbinom_param2, nbinom_ll,
        "Negative Binomial Distribution Log-Likelihood Contours",
        "n (Number of Successes)", "p (Success Probability)",
        save_path=nbinom_contour_path
    )
    
    # Normal distribution
    normal_param1, normal_param2, normal_ll = generate_likelihood_surface(
        mystery_data, 
        distribution_type='norm'
    )
    # 3D surface
    normal_surface_path = os.path.join(question_dir, "ex2_normal_likelihood_surface.png")
    plot_mystery_likelihood_surface(
        normal_param1, normal_param2, normal_ll,
        "Normal Distribution Log-Likelihood Surface",
        "μ (Mean)", "σ (Standard Deviation)",
        save_path=normal_surface_path
    )
    # 2D contour
    normal_contour_path = os.path.join(question_dir, "ex2_normal_likelihood_contour.png")
    plot_mystery_likelihood_contour(
        normal_param1, normal_param2, normal_ll,
        "Normal Distribution Log-Likelihood Contours",
        "μ (Mean)", "σ (Standard Deviation)",
        save_path=normal_contour_path
    )
    
    # Beta distribution
    beta_param1, beta_param2, beta_ll = generate_likelihood_surface(
        mystery_data, 
        distribution_type='beta'
    )
    # 3D surface
    beta_surface_path = os.path.join(question_dir, "ex2_beta_likelihood_surface.png")
    plot_mystery_likelihood_surface(
        beta_param1, beta_param2, beta_ll,
        "Beta Distribution Log-Likelihood Surface",
        "α (Alpha)", "β (Beta)",
        save_path=beta_surface_path
    )
    # 2D contour
    beta_contour_path = os.path.join(question_dir, "ex2_beta_likelihood_contour.png")
    plot_mystery_likelihood_contour(
        beta_param1, beta_param2, beta_ll,
        "Beta Distribution Log-Likelihood Contours",
        "α (Alpha)", "β (Beta)",
        save_path=beta_contour_path
    )
    
    print(f"\nAll question materials generated in: {question_dir}")
    print(f"True distribution: Negative Binomial with parameters {true_params}")
    
    # Find the MLE for the correct distribution (negative binomial)
    max_idx = np.unravel_index(np.argmax(nbinom_ll), nbinom_ll.shape)
    mle_n = nbinom_param1[max_idx]
    mle_p = nbinom_param2[max_idx]
    print(f"MLE estimates: n = {mle_n:.2f}, p = {mle_p:.2f}")
    
    return question_dir, true_params

if __name__ == "__main__":
    question_dir, true_params = generate_mle_visual_question() 