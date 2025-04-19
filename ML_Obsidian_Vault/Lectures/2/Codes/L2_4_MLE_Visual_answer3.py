import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
from matplotlib import cm

def load_normal_samples(true_mean=5.0, true_std=2.0, sample_sizes=[10, 30, 100, 1000], 
                      n_simulations=1000, random_seed=42):
    """
    Generate the same normal samples as in the question script for consistency,
    and calculate MLEs for each sample to demonstrate statistical properties.
    """
    np.random.seed(random_seed)
    
    # Dictionary to store results for each sample size
    results = {}
    
    for n in sample_sizes:
        # Initialize arrays to store estimates
        mean_estimates = np.zeros(n_simulations)
        var_estimates = np.zeros(n_simulations)
        var_unbiased_estimates = np.zeros(n_simulations)
        
        # Generate samples and compute MLEs
        for i in range(n_simulations):
            # Generate a sample of size n
            sample = np.random.normal(true_mean, true_std, n)
            
            # Calculate MLEs
            mean_mle = np.mean(sample)  # MLE for mean
            var_mle = np.sum((sample - mean_mle) ** 2) / n  # MLE for variance (biased)
            var_unbiased = np.sum((sample - mean_mle) ** 2) / (n - 1)  # Unbiased variance estimator
            
            # Store estimates
            mean_estimates[i] = mean_mle
            var_estimates[i] = var_mle
            var_unbiased_estimates[i] = var_unbiased
        
        # Store results for this sample size
        results[n] = {
            'mean_estimates': mean_estimates,
            'var_estimates': var_estimates,
            'var_unbiased_estimates': var_unbiased_estimates,
            'mean_mle_mean': np.mean(mean_estimates),
            'mean_mle_std': np.std(mean_estimates),
            'var_mle_mean': np.mean(var_estimates),
            'var_mle_std': np.std(var_estimates),
            'var_unbiased_mean': np.mean(var_unbiased_estimates),
            'var_unbiased_std': np.std(var_unbiased_estimates)
        }
    
    return results

def plot_sampling_distribution(estimation_results, true_mean=5.0, true_var=4.0, save_path=None):
    """
    Create a comprehensive visualization of the sampling distributions of MLEs
    for different sample sizes, illustrating key statistical properties.
    """
    sample_sizes = sorted(list(estimation_results.keys()))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    
    # Create figure with 2 rows for mean and variance estimators
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot for mean estimator
    for i, n in enumerate(sample_sizes):
        means = estimation_results[n]['mean_estimates']
        
        # Calculate kernel density estimation
        x = np.linspace(min(means) - 0.5, max(means) + 0.5, 1000)
        bandwidth = 0.2 / np.sqrt(n)  # Bandwidth decreases with sample size
        kde = np.zeros_like(x)
        for m in means:
            kde += norm.pdf(x, m, bandwidth)
        kde /= len(means)
        
        # Plot the density
        axs[0].plot(x, kde, color=colors[i], linewidth=2, label=f'n = {n}')
        
        # Add vertical line for mean and annotation
        std_error = np.std(means)
        axs[0].axvline(x=true_mean, color='black', linestyle='--', linewidth=1.5)
        
        # Add annotation about standard error
        axs[0].annotate(f'SE(n={n}) = {std_error:.4f}', 
                      xy=(true_mean + 0.5, 0.7 * max(kde) * (4-i)/4),
                      xytext=(true_mean + 0.5, 0.7 * max(kde) * (4-i)/4),
                      color=colors[i], fontweight='bold')
    
    # Customize mean plot
    axs[0].set_title('Sampling Distribution of Mean Estimator (MLE)', fontsize=14)
    axs[0].set_xlabel('Estimated Mean', fontsize=12)
    axs[0].set_ylabel('Density', fontsize=12)
    axs[0].axvline(x=true_mean, color='red', linestyle='-', linewidth=2, 
                 label=f'True Mean: {true_mean}')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(fontsize=10)
    
    # Add theoretical note
    axs[0].text(0.02, 0.98, 
               "Key Properties:\n" +
               "• Unbiased: E[μ̂] = μ\n" +
               "• Efficient: Achieves CRLB\n" +
               "• SE(μ̂) = σ/√n (decreases with √n)",
               transform=axs[0].transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Plot for variance estimator 
    for i, n in enumerate(sample_sizes):
        variances = estimation_results[n]['var_estimates']
        unbiased_vars = estimation_results[n]['var_unbiased_estimates']
        
        # Calculate kde for MLE variance
        x = np.linspace(min(variances) - 0.5, max(variances) + 0.5, 1000)
        bandwidth = 0.3 / np.sqrt(n)
        kde = np.zeros_like(x)
        for v in variances:
            kde += norm.pdf(x, v, bandwidth)
        kde /= len(variances)
        
        # Plot the density
        axs[1].plot(x, kde, color=colors[i], linewidth=2, linestyle='-', 
                  label=f'MLE (n = {n})')
        
        # Only show unbiased estimator for the largest sample size to avoid clutter
        if n == max(sample_sizes):
            # Calculate kde for unbiased variance
            x_unbiased = np.linspace(min(unbiased_vars) - 0.5, max(unbiased_vars) + 0.5, 1000)
            kde_unbiased = np.zeros_like(x_unbiased)
            for v in unbiased_vars:
                kde_unbiased += norm.pdf(x_unbiased, v, bandwidth)
            kde_unbiased /= len(unbiased_vars)
            
            # Plot the unbiased estimator
            axs[1].plot(x_unbiased, kde_unbiased, color='purple', linewidth=2, linestyle='--',
                      label=f'Unbiased (n = {n})')
        
        # Add annotation about bias for larger sample sizes
        if n >= 100:
            mean_var = np.mean(variances)
            bias = mean_var - true_var
            axs[1].annotate(f'Bias ≈ {bias:.4f}', 
                         xy=(mean_var, 0.7 * max(kde)),
                         xytext=(mean_var + 0.2, 0.7 * max(kde)),
                         color=colors[i], fontweight='bold',
                         arrowprops=dict(arrowstyle="->", color=colors[i]))
    
    # Customize variance plot
    axs[1].set_title('Sampling Distribution of Variance Estimator', fontsize=14)
    axs[1].set_xlabel('Estimated Variance', fontsize=12)
    axs[1].set_ylabel('Density', fontsize=12)
    axs[1].axvline(x=true_var, color='red', linestyle='-', linewidth=2, 
                 label=f'True Variance: {true_var}')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(fontsize=10)
    
    # Add theoretical note for variance
    axs[1].text(0.02, 0.98, 
               "Key Properties:\n" +
               "• Biased: E[σ̂²] = (n-1)σ²/n\n" +
               "• Bias decreases as n increases\n" +
               "• The unbiased estimator is s² = n/(n-1) · σ̂²",
               transform=axs[1].transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Overall figure title
    plt.suptitle('Sampling Distributions of Normal MLEs', fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sampling distribution plot saved to {save_path}")
    
    plt.close()

def plot_mle_properties(estimation_results, true_mean=5.0, true_var=4.0, save_path=None):
    """
    Create a visualization of key MLE properties (consistency, efficiency, bias)
    for normal distribution estimators.
    """
    sample_sizes = sorted(list(estimation_results.keys()))
    
    # Extract statistics
    mean_means = [estimation_results[n]['mean_mle_mean'] for n in sample_sizes]
    mean_stds = [estimation_results[n]['mean_mle_std'] for n in sample_sizes]
    mean_mse = [(m - true_mean)**2 + s**2 for m, s in zip(mean_means, mean_stds)]
    
    var_means = [estimation_results[n]['var_mle_mean'] for n in sample_sizes]
    var_stds = [estimation_results[n]['var_mle_std'] for n in sample_sizes]
    var_mse = [(v - true_var)**2 + s**2 for v, s in zip(var_means, var_stds)]
    
    var_unbiased_means = [estimation_results[n]['var_unbiased_mean'] for n in sample_sizes]
    
    # Calculate true standard deviation from variance
    true_std = np.sqrt(true_var)
    
    # Create figure with 2 rows
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Mean estimator - Bias & Variance
    axs[0, 0].errorbar(sample_sizes, mean_means, yerr=mean_stds, fmt='o-', 
                     color='blue', ecolor='lightblue', elinewidth=3, capsize=5,
                     linewidth=2, markersize=8, label='Mean Estimate with SE')
    axs[0, 0].axhline(y=true_mean, color='red', linestyle='--', linewidth=2, 
                    label=f'True Mean: {true_mean}')
    
    # Add theoretical standard error
    theory_se = [true_std / np.sqrt(n) for n in sample_sizes]
    axs[0, 0].plot(sample_sizes, [true_mean + 2*se for se in theory_se], 'k--', alpha=0.3)
    axs[0, 0].plot(sample_sizes, [true_mean - 2*se for se in theory_se], 'k--', alpha=0.3)
    axs[0, 0].fill_between(sample_sizes, 
                         [true_mean - 2*se for se in theory_se],
                         [true_mean + 2*se for se in theory_se],
                         alpha=0.1, color='blue', label='95% CI (Theory)')
    
    # Customize plot
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_title('Mean Estimator: Consistency & Efficiency', fontsize=14)
    axs[0, 0].set_xlabel('Sample Size (log scale)', fontsize=12)
    axs[0, 0].set_ylabel('Mean Estimate', fontsize=12)
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend(fontsize=10)
    
    # Plot 2: Variance estimator - Bias & Variance
    axs[0, 1].errorbar(sample_sizes, var_means, yerr=var_stds, fmt='o-',
                     color='green', ecolor='lightgreen', elinewidth=3, capsize=5,
                     linewidth=2, markersize=8, label='MLE Variance')
    axs[0, 1].plot(sample_sizes, var_unbiased_means, 's-', color='purple',
                 linewidth=2, markersize=8, label='Unbiased Variance')
    axs[0, 1].axhline(y=true_var, color='red', linestyle='--', linewidth=2, 
                    label=f'True Variance: {true_var}')
    
    # Add theoretical bias line
    theory_biased_var = [true_var * (n-1)/n for n in sample_sizes]
    axs[0, 1].plot(sample_sizes, theory_biased_var, 'g--', alpha=0.5, 
                 label='Theoretical E[MLE]')
    
    # Customize plot
    axs[0, 1].set_xscale('log')
    axs[0, 1].set_title('Variance Estimator: Bias & Consistency', fontsize=14)
    axs[0, 1].set_xlabel('Sample Size (log scale)', fontsize=12)
    axs[0, 1].set_ylabel('Variance Estimate', fontsize=12)
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend(fontsize=10)
    
    # Plot 3: Standard Errors scaling with n
    axs[1, 0].plot(sample_sizes, mean_stds, 'o-', color='blue', linewidth=2, 
                 label='Observed SE (Mean)')
    axs[1, 0].plot(sample_sizes, theory_se, '--', color='blue', linewidth=2, 
                 label='Theoretical SE = σ/√n')
    
    # Add 1/sqrt(n) reference line
    scale_factor = mean_stds[0] * np.sqrt(sample_sizes[0])
    reference = [scale_factor / np.sqrt(n) for n in sample_sizes]
    axs[1, 0].plot(sample_sizes, reference, ':', color='black', linewidth=2,
                 label='1/√n Reference')
    
    # Customize plot
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_title('Efficiency: Standard Error Scaling', fontsize=14)
    axs[1, 0].set_xlabel('Sample Size (log scale)', fontsize=12)
    axs[1, 0].set_ylabel('Standard Error (log scale)', fontsize=12)
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend(fontsize=10)
    
    # Add annotation about efficiency
    axs[1, 0].text(0.05, 0.2, "Standard Error decreases as 1/√n\n" +
                  "This is the theoretical optimal rate\n" +
                  "confirming the mean estimator is efficient",
                  transform=axs[1, 0].transAxes, fontsize=10,
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Plot 4: MSE decomposition (Bias^2 + Variance)
    ax4 = axs[1, 1]
    width = 0.35
    x = np.arange(len(sample_sizes))
    
    # Calculate bias squared for both estimators
    mean_bias_squared = [(m - true_mean)**2 for m in mean_means]
    var_bias_squared = [(v - true_var)**2 for v in var_means]
    
    # Calculate variance component of MSE
    mean_var_component = [s**2 for s in mean_stds]
    var_var_component = [s**2 for s in var_stds]
    
    # Plot stacked bars for mean estimator
    ax4.bar(x - width/2, mean_bias_squared, width, label='Mean Bias²', color='lightblue')
    ax4.bar(x - width/2, mean_var_component, width, bottom=mean_bias_squared, 
           label='Mean Variance', color='blue')
    
    # Plot stacked bars for variance estimator
    ax4.bar(x + width/2, var_bias_squared, width, label='Var Bias²', color='lightgreen')
    ax4.bar(x + width/2, var_var_component, width, bottom=var_bias_squared, 
           label='Var Variance', color='green')
    
    # Customize plot
    ax4.set_yscale('log')
    ax4.set_title('MSE Decomposition: Bias² + Variance', fontsize=14)
    ax4.set_xlabel('Sample Size', fontsize=12)
    ax4.set_ylabel('MSE Components (log scale)', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(sample_sizes)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    
    # Overall figure title
    plt.suptitle('Normal Distribution MLE Properties', fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"MLE properties plot saved to {save_path}")
    
    plt.close()

def create_asymptotic_normality_visualization(estimation_results, save_path=None):
    """
    Visualize the asymptotic normality of the MLE estimators by comparing
    the sampling distribution to the theoretical normal approximation.
    """
    # Focus on the variance estimator as it's not exactly normal for small samples
    sample_sizes = sorted(list(estimation_results.keys()))
    
    # Create a 2x2 grid of plots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    
    for i, n in enumerate(sample_sizes):
        var_estimates = estimation_results[n]['var_estimates']
        
        # Calculate statistics
        var_mean = np.mean(var_estimates)
        var_std = np.std(var_estimates)
        
        # Create histogram
        axs[i].hist(var_estimates, bins=30, density=True, alpha=0.5, color='lightgreen',
                  edgecolor='darkgreen', label=f'Empirical (n={n})')
        
        # Add theoretical normal approximation
        x = np.linspace(min(var_estimates), max(var_estimates), 1000)
        y = norm.pdf(x, var_mean, var_std)
        axs[i].plot(x, y, 'r-', linewidth=2, label='Normal Approximation')
        
        # Add vertical line for true variance
        axs[i].axvline(x=4.0, color='blue', linestyle='--', linewidth=2, 
                     label='True Variance')
        
        # Calculate KL divergence as a measure of closeness to normal
        hist, bin_edges = np.histogram(var_estimates, bins=30, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        norm_pdf = norm.pdf(bin_centers, var_mean, var_std)
        
        # Avoid log(0) in KL calculation
        hist_valid = hist[hist > 0]
        norm_pdf_valid = norm_pdf[hist > 0]
        
        # Simple approximation of KL divergence
        kl_div = np.sum(hist_valid * np.log(hist_valid / norm_pdf_valid)) * (bin_centers[1] - bin_centers[0])
        
        # Customize plot
        axs[i].set_title(f'Variance Estimator Distribution (n={n})', fontsize=12)
        axs[i].set_xlabel('Estimated Variance', fontsize=10)
        axs[i].set_ylabel('Density', fontsize=10)
        axs[i].grid(True, alpha=0.3)
        axs[i].legend(fontsize=9)
        
        # Add annotation about normality
        axs[i].text(0.05, 0.9, f"KL Divergence from Normal: {kl_div:.4f}\n" +
                  f"{'Closer to normal' if kl_div < 0.1 else 'Deviation from normal'}",
                  transform=axs[i].transAxes, fontsize=9,
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Overall figure title
    plt.suptitle('Asymptotic Normality of Variance Estimator', fontsize=16, y=0.98)
    
    # Add explanation in the center
    fig.text(0.5, 0.01, 
            "As sample size increases, the distribution of the MLE approaches a normal distribution\n" +
            "This property is known as asymptotic normality and is a key feature of MLEs",
            ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Asymptotic normality plot saved to {save_path}")
    
    plt.close()

def generate_answer_materials():
    """Generate answer materials for the MLE visual question on normal distribution"""
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    answer_dir = os.path.join(images_dir, "MLE_Visual_Answer")
    os.makedirs(answer_dir, exist_ok=True)
    
    print("Generating MLE Visual Answer Example 3 materials...")
    
    # Load the data (same parameters as the question)
    true_mean = 5.0
    true_std = 2.0  # true_var = 4.0
    sample_sizes = [10, 30, 100, 1000]
    
    # Generate samples and compute MLEs
    estimation_results = load_normal_samples(
        true_mean=true_mean, 
        true_std=true_std, 
        sample_sizes=sample_sizes,
        n_simulations=1000
    )
    
    # Generate the sampling distribution visualization
    sampling_dist_path = os.path.join(answer_dir, "ex3_sampling_distribution.png")
    plot_sampling_distribution(
        estimation_results, 
        true_mean=true_mean, 
        true_var=true_std**2,
        save_path=sampling_dist_path
    )
    
    # Generate the MLE properties visualization
    properties_path = os.path.join(answer_dir, "ex3_mle_properties.png")
    plot_mle_properties(
        estimation_results, 
        true_mean=true_mean, 
        true_var=true_std**2,
        save_path=properties_path
    )
    
    # Generate the asymptotic normality visualization
    asymp_norm_path = os.path.join(answer_dir, "ex3_asymptotic_normality.png")
    create_asymptotic_normality_visualization(
        estimation_results,
        save_path=asymp_norm_path
    )
    
    print("MLE Visual Answer Example 3 materials generated successfully!")
    
    return estimation_results, answer_dir

if __name__ == "__main__":
    # Generate all the answer materials
    estimation_results, answer_dir = generate_answer_materials() 