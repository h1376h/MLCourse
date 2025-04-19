import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def generate_normal_samples(true_mean=5.0, true_std=2.0, sample_sizes=[10, 30, 100, 1000], 
                           n_simulations=1000, random_seed=42):
    """
    Generate samples from a normal distribution with different sample sizes
    and calculate MLEs for each sample to demonstrate consistency and efficiency.
    """
    np.random.seed(random_seed)
    
    # Dictionary to store results for each sample size
    results = {}
    
    for n in sample_sizes:
        # Initialize arrays to store estimates
        mean_estimates = np.zeros(n_simulations)
        var_estimates = np.zeros(n_simulations)
        
        # Generate samples and compute MLEs
        for i in range(n_simulations):
            # Generate a sample of size n
            sample = np.random.normal(true_mean, true_std, n)
            
            # Calculate MLEs
            mean_mle = np.mean(sample)  # MLE for mean
            var_mle = np.sum((sample - mean_mle) ** 2) / n  # MLE for variance (biased)
            
            # Store estimates
            mean_estimates[i] = mean_mle
            var_estimates[i] = var_mle
        
        # Store results for this sample size
        results[n] = {
            'mean_estimates': mean_estimates,
            'var_estimates': var_estimates,
            'mean_mle_mean': np.mean(mean_estimates),
            'mean_mle_std': np.std(mean_estimates),
            'var_mle_mean': np.mean(var_estimates),
            'var_mle_std': np.std(var_estimates)
        }
    
    return results

def plot_mle_consistency(estimation_results, true_mean=5.0, true_var=4.0, save_path=None):
    """
    Create a visual showing how MLE estimates converge to true parameter values
    as sample size increases.
    """
    sample_sizes = sorted(list(estimation_results.keys()))
    
    # Extract means and standard deviations of the estimates
    mean_means = [estimation_results[n]['mean_mle_mean'] for n in sample_sizes]
    mean_stds = [estimation_results[n]['mean_mle_std'] for n in sample_sizes]
    var_means = [estimation_results[n]['var_mle_mean'] for n in sample_sizes]
    var_stds = [estimation_results[n]['var_mle_std'] for n in sample_sizes]
    
    # Create the figure
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    
    # Mean estimator plot
    axs[0].errorbar(sample_sizes, mean_means, yerr=mean_stds, fmt='o-', 
                   color='blue', ecolor='lightblue', elinewidth=3, capsize=5,
                   linewidth=2, markersize=8)
    axs[0].axhline(y=true_mean, color='red', linestyle='--', linewidth=2, label=f'True Mean: {true_mean}')
    
    # Customize mean plot
    axs[0].set_xscale('log')
    axs[0].set_title('MLE for Mean (μ)', fontsize=14)
    axs[0].set_xlabel('Sample Size (log scale)', fontsize=12)
    axs[0].set_ylabel('Estimated Value', fontsize=12)
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(fontsize=12)
    
    # Variance estimator plot
    axs[1].errorbar(sample_sizes, var_means, yerr=var_stds, fmt='o-',
                   color='green', ecolor='lightgreen', elinewidth=3, capsize=5,
                   linewidth=2, markersize=8)
    axs[1].axhline(y=true_var, color='red', linestyle='--', linewidth=2, label=f'True Variance: {true_var}')
    
    # Customize variance plot
    axs[1].set_xscale('log')
    axs[1].set_title('MLE for Variance (σ²)', fontsize=14)
    axs[1].set_xlabel('Sample Size (log scale)', fontsize=12)
    axs[1].set_ylabel('Estimated Value', fontsize=12)
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(fontsize=12)
    
    # Overall figure title
    plt.suptitle('MLE Consistency: Convergence to True Parameters with Increasing Sample Size', 
                fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"MLE consistency plot saved to {save_path}")
    
    plt.close()

def create_sampling_distribution_visualization(estimation_results, save_path=None):
    """
    Visualize the sampling distributions of the MLE estimators for different sample sizes.
    This demonstrates how the distributions become more concentrated around the true values.
    """
    sample_sizes = sorted(list(estimation_results.keys()))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    axs = axs.flatten()
    
    # Plot sampling distributions for mean
    for i, n in enumerate(sample_sizes):
        mean_estimates = estimation_results[n]['mean_estimates']
        kde_x = np.linspace(min(mean_estimates), max(mean_estimates), 1000)
        kde = lambda x, h: np.mean(norm.pdf((x - mean_estimates[:, np.newaxis]) / h, 0, 1), axis=0) / h
        kde_y = kde(kde_x, 0.1)
        
        axs[i].plot(kde_x, kde_y, color=colors[i], linewidth=2, 
                  label=f'Sample Size n={n}')
        axs[i].axvline(x=5.0, color='red', linestyle='--', linewidth=2, 
                     label='True Mean: 5.0')
        axs[i].set_title(f'Sampling Distribution of Mean Estimator (n={n})', fontsize=12)
        axs[i].set_xlabel('Estimated Mean', fontsize=10)
        axs[i].set_ylabel('Density', fontsize=10)
        axs[i].grid(True, alpha=0.3)
        axs[i].legend()
        
        # Add standard error annotation
        se = np.std(mean_estimates)
        axs[i].annotate(f'Standard Error: {se:.4f}', xy=(0.05, 0.9), xycoords='axes fraction',
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Overall figure title
    plt.suptitle('Sampling Distribution of Mean Estimator for Different Sample Sizes', 
                fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sampling distribution plot saved to {save_path}")
    
    plt.close()

def generate_likelihood_surface(mean_values, sigma_values, data, save_path=None):
    """
    Generate 3D visualization of the log-likelihood surface for a normal distribution
    with different sample sizes to show how it becomes more peaked with larger samples.
    """
    # Create meshgrid
    mean_grid, sigma_grid = np.meshgrid(mean_values, sigma_values)
    
    # Calculate log-likelihood for each parameter combination
    log_likelihood = np.zeros_like(mean_grid)
    for i in range(mean_grid.shape[0]):
        for j in range(mean_grid.shape[1]):
            mu = mean_grid[i, j]
            sigma = sigma_grid[i, j]
            log_likelihood[i, j] = np.sum(norm.logpdf(data, mu, sigma))
    
    # Normalize log-likelihood to have a maximum of 0
    log_likelihood = log_likelihood - np.max(log_likelihood)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(mean_grid, sigma_grid, log_likelihood, cmap=cm.viridis,
                          linewidth=0, antialiased=True, alpha=0.8)
    
    # Find and mark the MLE
    max_idx = np.unravel_index(np.argmax(log_likelihood), log_likelihood.shape)
    mle_mean = mean_grid[max_idx]
    mle_sigma = sigma_grid[max_idx]
    
    # Plot the MLE point
    ax.scatter([mle_mean], [mle_sigma], [np.max(log_likelihood)], 
              color='red', s=100, label=f'MLE: μ={mle_mean:.2f}, σ={mle_sigma:.2f}')
    
    # Customize the plot
    ax.set_xlabel('Mean (μ)', fontsize=12)
    ax.set_ylabel('Standard Deviation (σ)', fontsize=12)
    ax.set_zlabel('Log-Likelihood', fontsize=12)
    ax.set_title(f'Log-Likelihood Surface (n={len(data)})', fontsize=14)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.1, label='Log-Likelihood')
    
    # Add a text annotation about the sharpness
    ax.text2D(0.02, 0.98, f"Sample Size: {len(data)}\nSharpness increases with sample size",
             transform=ax.transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Adjust view angle
    ax.view_init(elev=25, azim=150)
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Log-likelihood surface plot saved to {save_path}")
    
    plt.close()

def generate_mle_visual_question():
    """Generate all the visual materials for the MLE properties question"""
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images", "MLE_Visual_Question")
    os.makedirs(images_dir, exist_ok=True)
    
    print("Generating MLE Visual Question materials (Normal Distribution)...")
    
    # 1. Generate data for MLE consistency demonstration
    true_mean = 5.0
    true_std = 2.0  # true_var = 4.0
    sample_sizes = [10, 30, 100, 1000]
    
    # Generate samples and compute MLEs
    estimation_results = generate_normal_samples(
        true_mean=true_mean, 
        true_std=true_std, 
        sample_sizes=sample_sizes,
        n_simulations=1000
    )
    
    # 2. Create consistency visualization
    consistency_plot_path = os.path.join(images_dir, "mle_consistency.png")
    plot_mle_consistency(
        estimation_results, 
        true_mean=true_mean, 
        true_var=true_std**2,
        save_path=consistency_plot_path
    )
    
    # 3. Generate likelihood surfaces for different sample sizes
    np.random.seed(42)
    
    # Small sample (n=10)
    small_sample = np.random.normal(true_mean, true_std, 10)
    mean_range = np.linspace(3.0, 7.0, 50)
    sigma_range = np.linspace(1.0, 3.0, 50)
    small_sample_plot_path = os.path.join(images_dir, "normal_loglik_n10.png")
    generate_likelihood_surface(mean_range, sigma_range, small_sample, small_sample_plot_path)
    
    # Large sample (n=100)
    large_sample = np.random.normal(true_mean, true_std, 100)
    mean_range = np.linspace(4.0, 6.0, 50)  # Narrower range as we expect more concentration
    sigma_range = np.linspace(1.5, 2.5, 50)
    large_sample_plot_path = os.path.join(images_dir, "normal_loglik_n100.png")
    generate_likelihood_surface(mean_range, sigma_range, large_sample, large_sample_plot_path)
    
    print("MLE Visual Question materials (Normal Distribution) generated successfully!")
    return estimation_results

if __name__ == "__main__":
    # Generate all the visual materials
    estimation_results = generate_mle_visual_question() 