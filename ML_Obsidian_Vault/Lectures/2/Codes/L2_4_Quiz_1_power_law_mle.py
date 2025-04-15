import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import os
from scipy import stats

def power_law_pdf(x, theta, upper_bound=3):
    """Probability density function for the custom power law from Question 1"""
    if theta <= 0:
        return np.zeros_like(x)
    
    # Create a mask for valid x values (0 <= x < upper_bound)
    mask = (x >= 0) & (x < upper_bound)
    
    # Initialize result array with zeros
    result = np.zeros_like(x, dtype=float)
    
    # Calculate PDF for valid x values
    valid_x = x[mask]
    result[mask] = (theta * valid_x**(theta-1)) / (upper_bound**theta)
    
    return result

def log_likelihood(theta, data, upper_bound=3):
    """Log-likelihood function for power law distribution in Question 1"""
    # Check if theta is in valid range (must be positive)
    if theta <= 0:
        return float('-inf')
    
    # Convert data to numpy array if it's not already
    data = np.array(data)
    
    # Check if all data is within bounds
    if np.any(data < 0) or np.any(data >= upper_bound):
        return float('-inf')
    
    n = len(data)
    sum_log_x = np.sum(np.log(data))
    
    # Log-likelihood formula
    ll = n * np.log(theta) + (theta - 1) * sum_log_x - n * theta * np.log(upper_bound)
    
    return ll

def plot_pdf_for_different_theta(upper_bound=3, save_path=None):
    """Plot the PDF for different values of theta to understand its effect"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate x values for plotting
    x = np.linspace(0.01, upper_bound, 1000)
    
    # Plot PDF for various theta values
    theta_values = [0.5, 1.0, 1.5, 2.0, 3.0]
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    for theta, color in zip(theta_values, colors):
        y = power_law_pdf(x, theta, upper_bound)
        ax.plot(x, y, color=color, linewidth=2, label=f'θ = {theta}')
    
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density f(x|θ)')
    ax.set_title('Power Law Density Functions for Different θ Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, upper_bound)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_likelihood_surface(data, upper_bound=3, save_path=None):
    """Plot the likelihood function surface for the power law data"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate MLE estimate analytically
    n = len(data)
    sum_log_x = np.sum(np.log(data))
    mle_theta = n / (n * np.log(upper_bound) - sum_log_x)
    
    # Create a range of possible theta values to plot
    possible_thetas = np.linspace(max(0.1, mle_theta - 1.5), mle_theta + 1.5, 1000)
    
    # Calculate the log-likelihood for each possible theta
    log_likelihoods = []
    for theta in possible_thetas:
        ll = log_likelihood(theta, data, upper_bound)
        log_likelihoods.append(ll)
    
    # Normalize the log-likelihood for better visualization
    log_likelihoods = np.array(log_likelihoods)
    
    # Plot the log-likelihood function
    ax.plot(possible_thetas, log_likelihoods, 'b-', linewidth=2)
    ax.axvline(x=mle_theta, color='r', linestyle='--', 
              label=f'MLE θ = {mle_theta:.4f}')
    
    ax.set_title(f"Log-Likelihood Function for Power Law Distribution")
    ax.set_xlabel('θ (Shape Parameter)')
    ax.set_ylabel('Log-Likelihood ℓ(θ)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return mle_theta

def plot_mle_fit(data, upper_bound=3, save_path=None):
    """Plot the fitted MLE distribution against the data"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate MLE estimate
    n = len(data)
    sum_log_x = np.sum(np.log(data))
    mle_theta = n / (n * np.log(upper_bound) - sum_log_x)
    
    # Generate x values for plotting
    x = np.linspace(0.01, upper_bound, 1000)
    y_mle = power_law_pdf(x, mle_theta, upper_bound)
    
    # Plot histogram of the data
    ax.hist(data, bins=min(15, len(data)), density=True, alpha=0.5, color='blue', 
             label='Observed Data')
    
    # Plot the fitted PDF based on MLE
    ax.plot(x, y_mle, 'r-', linewidth=2, 
            label=f'MLE Fit (θ = {mle_theta:.4f})')
    
    # Mark the observed data points
    ax.plot(data, np.zeros_like(data), 'bo', markersize=8, alpha=0.6)
    
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density')
    ax.set_title('Maximum Likelihood Estimation for Power Law Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, upper_bound)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return mle_theta

def plot_mle_step_by_step(data, upper_bound=3, save_path=None):
    """Visualize the MLE calculation process step by step"""
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    # Calculate MLE estimate
    n = len(data)
    sum_log_x = np.sum(np.log(data))
    mle_theta = n / (n * np.log(upper_bound) - sum_log_x)
    
    # Plot 1: The PDF definition
    ax1 = axes[0, 0]
    x_range = np.linspace(0.01, upper_bound, 1000)
    theta_examples = [0.8, 1.0, 1.5, 2.5]
    colors = ['red', 'blue', 'green', 'purple']
    
    for theta, color in zip(theta_examples, colors):
        y = power_law_pdf(x_range, theta, upper_bound)
        ax1.plot(x_range, y, color=color, linewidth=2, label=f'θ = {theta}')
    
    ax1.set_title('Probability Density Function (PDF)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x|θ)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: The likelihood function
    ax2 = axes[0, 1]
    theta_range = np.linspace(max(0.1, mle_theta - 1.5), mle_theta + 1.5, 1000)
    likelihoods = []
    
    for theta in theta_range:
        likelihood = np.prod([power_law_pdf(x, theta, upper_bound) for x in data])
        likelihoods.append(likelihood)
    
    ax2.plot(theta_range, likelihoods, 'b-', linewidth=2)
    ax2.axvline(x=mle_theta, color='r', linestyle='--', label=f'MLE θ = {mle_theta:.4f}')
    ax2.set_title('Likelihood Function L(θ)')
    ax2.set_xlabel('θ (Shape Parameter)')
    ax2.set_ylabel('Likelihood')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: The log-likelihood function
    ax3 = axes[1, 0]
    log_likelihoods = []
    
    for theta in theta_range:
        ll = log_likelihood(theta, data, upper_bound)
        log_likelihoods.append(ll)
    
    ax3.plot(theta_range, log_likelihoods, 'g-', linewidth=2)
    ax3.axvline(x=mle_theta, color='r', linestyle='--', label=f'MLE θ = {mle_theta:.4f}')
    ax3.set_title('Log-Likelihood Function ℓ(θ)')
    ax3.set_xlabel('θ (Shape Parameter)')
    ax3.set_ylabel('Log-Likelihood')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: The derivative of log-likelihood
    ax4 = axes[1, 1]
    derivatives = []
    
    for theta in theta_range:
        # Analytical derivative of log-likelihood
        derivative = n/theta + sum_log_x - n*np.log(upper_bound)
        derivatives.append(derivative)
    
    ax4.plot(theta_range, derivatives, 'm-', linewidth=2)
    ax4.axvline(x=mle_theta, color='r', linestyle='--', label=f'MLE θ = {mle_theta:.4f}')
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax4.set_title('Derivative of Log-Likelihood Function')
    ax4.set_xlabel('θ (Shape Parameter)')
    ax4.set_ylabel('d/dθ [ℓ(θ)]')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.suptitle('Maximum Likelihood Estimation Process', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_simplified_method(data, upper_bound=3, save_path=None):
    """Visualize the simplified calculation method mentioned in the notes"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate MLE using the full method
    n = len(data)
    sum_log_x = np.sum(np.log(data))
    mle_theta = n / (n * np.log(upper_bound) - sum_log_x)
    
    # Calculate MLE using the simplified method
    log_ratios = [np.log(upper_bound / x) for x in data]
    mean_log_ratio = np.mean(log_ratios)
    simplified_mle = 1 / mean_log_ratio
    
    # Display the calculation steps
    x_positions = np.arange(len(data))
    width = 0.35
    
    # Plot the log(upper_bound/x) values for each data point
    bars = ax.bar(x_positions, log_ratios, width, label='log(3/x_i)')
    
    # Add a horizontal line for the mean
    ax.axhline(y=mean_log_ratio, color='r', linestyle='-', 
               label=f'Mean = {mean_log_ratio:.4f}')
    
    # Add a text annotation for the final MLE
    ax.text(0.5, 0.9, f'MLE θ = 1 / Mean = {simplified_mle:.4f}', 
            horizontalalignment='center', verticalalignment='center', 
            transform=ax.transAxes, fontsize=12, 
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    
    ax.set_xlabel('Data Point Index')
    ax.set_ylabel('log(3/x_i)')
    ax.set_title('Simplified MLE Calculation for Power Law Distribution')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'x_{i+1}' for i in range(len(data))])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    for i, v in enumerate(log_ratios):
        ax.text(i, v + 0.05, f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return simplified_mle

def plot_3d_likelihood_surface(save_path=None):
    """Create a 3D visualization of the likelihood surface for different data and theta values"""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define the theta range
    theta_range = np.linspace(0.5, 3.0, 50)
    
    # Define different data sets
    upper_bound = 3
    data_sets = [
        np.array([0.5, 1.0, 1.5, 2.0, 2.5]),  # Uniform spread
        np.array([0.1, 0.2, 0.3, 0.4, 0.5]),  # Low values
        np.array([2.5, 2.6, 2.7, 2.8, 2.9]),  # High values
        np.array([0.1, 0.3, 1.5, 2.7, 2.9]),  # Mixed values
    ]
    
    # Colors for different data sets
    colors = ['blue', 'red', 'green', 'purple']
    
    # Plot likelihood curves for each data set
    for i, data in enumerate(data_sets):
        # Calculate MLE for this data set
        n = len(data)
        sum_log_x = np.sum(np.log(data))
        mle_theta = n / (n * np.log(upper_bound) - sum_log_x)
        
        # Calculate log-likelihoods for the theta range
        log_likes = []
        for theta in theta_range:
            ll = log_likelihood(theta, data, upper_bound)
            log_likes.append(ll)
        
        # Normalize log-likelihoods for better visualization
        log_likes = np.array(log_likes)
        log_likes = log_likes - np.min(log_likes)
        log_likes = log_likes / np.max(log_likes)
        
        # Plot the likelihood curve for this data set
        z_offset = i * 1.5  # Offset in z direction for each curve
        ax.plot(theta_range, log_likes, zs=z_offset, zdir='y', color=colors[i], 
                label=f'Data set {i+1}, MLE={mle_theta:.2f}')
        
        # Mark the MLE point
        ax.scatter(mle_theta, 1.0, z_offset, color=colors[i], s=100, marker='o')
    
    ax.set_xlabel('θ (Shape Parameter)')
    ax.set_ylabel('Data Set')
    ax.set_zlabel('Normalized Log-Likelihood')
    ax.set_yticks([0, 1.5, 3.0, 4.5])
    ax.set_yticklabels(['Data 1', 'Data 2', 'Data 3', 'Data 4'])
    ax.set_title('Log-Likelihood Surfaces for Different Data Sets')
    ax.legend()
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def main():
    """Generate all visualizations for Question 1 of the L2.4 quiz"""
    # Create synthetic data similar to what might be in the problem
    np.random.seed(42)  # For reproducibility
    synthetic_data = np.array([0.5, 1.2, 2.1, 1.8, 0.7, 2.5, 1.0, 1.5])
    
    # Create a directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level and then to Images
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    # Create a subfolder for this quiz
    save_dir = os.path.join(images_dir, "L2_4_Quiz_1")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 1 of the L2.4 MLE quiz...")
    
    # 1. Plot PDFs for different theta values
    plot_pdf_for_different_theta(save_path=os.path.join(save_dir, "power_law_pdfs.png"))
    print("1. PDF visualization created")
    
    # 2. Plot likelihood surface
    mle_theta = plot_likelihood_surface(synthetic_data, save_path=os.path.join(save_dir, "likelihood_surface.png"))
    print(f"2. Likelihood surface visualization created, MLE θ = {mle_theta:.4f}")
    
    # 3. Plot MLE fit to data
    plot_mle_fit(synthetic_data, save_path=os.path.join(save_dir, "mle_fit.png"))
    print("3. MLE fit visualization created")
    
    # 4. Plot step-by-step MLE process
    plot_mle_step_by_step(synthetic_data, save_path=os.path.join(save_dir, "mle_process.png"))
    print("4. Step-by-step MLE process visualization created")
    
    # 5. Plot the simplified calculation method
    simplified_mle = plot_simplified_method(synthetic_data, save_path=os.path.join(save_dir, "simplified_method.png"))
    print(f"5. Simplified method visualization created, MLE θ = {simplified_mle:.4f}")
    
    # 6. 3D visualization of likelihood surface
    plot_3d_likelihood_surface(save_path=os.path.join(save_dir, "3d_likelihood_surface.png"))
    print("6. 3D likelihood surface visualization created")
    
    # 7. Plot MLE sampling distribution
    mean_mle, std_mle, bias, mse = plot_mle_sampling_distribution(true_theta=1.5, sample_size=8, 
                                                                num_samples=5000, 
                                                                save_path=os.path.join(save_dir, "sampling_distribution.png"))
    print(f"7. Sampling distribution visualization created (Mean MLE: {mean_mle:.4f}, Std Dev: {std_mle:.4f})")
    
    # 8. Plot asymptotic properties
    asymptotic_results = plot_asymptotic_properties(true_theta=1.5,
                                                  sample_sizes=[5, 10, 20, 50, 100, 500, 1000],
                                                  num_samples=1000,
                                                  save_path=os.path.join(save_dir, "asymptotic_properties.png"))
    print("8. Asymptotic properties visualization created")
    
    # 9. Plot confidence intervals
    mle, std_err, ci_lower, ci_upper = plot_confidence_intervals(synthetic_data, confidence=0.95,
                                                              save_path=os.path.join(save_dir, "confidence_intervals.png"))
    print(f"9. Confidence interval visualization created (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
    
    # 10. Plot interactive MLE calculation
    datasets = plot_interactive_mle_calculation(save_path=os.path.join(save_dir, "interactive_mle.png"))
    print("10. Interactive MLE calculation visualization created")
    
    # Create a summary table of all results
    print("\n=== Summary of Results ===")
    print(f"MLE estimate for the synthetic data: θ = {mle_theta:.4f}")
    print(f"Standard error: {std_err:.4f}")
    print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    print(f"\nAll visualizations have been saved to: {save_dir}")
    print("Detailed visualizations help demonstrate the MLE calculation process step-by-step")
    print("and provide insights into the properties of the MLE estimator.")

def plot_mle_sampling_distribution(true_theta=1.5, sample_size=10, num_samples=1000, upper_bound=3, save_path=None):
    """
    Demonstrates the sampling distribution of the MLE estimator by repeatedly sampling
    from the power law distribution with a known true theta and calculating the MLE
    """
    np.random.seed(42)  # For reproducibility
    
    # Function to generate samples from power law distribution
    def generate_power_law_sample(theta, size, upper_bound=3):
        # Use inverse transform sampling
        u = np.random.uniform(0, 1, size=size)
        return upper_bound * (u ** (1/theta))
    
    # Generate multiple samples and compute MLE for each
    mle_estimates = []
    for _ in range(num_samples):
        # Generate a sample from the true distribution
        sample = generate_power_law_sample(true_theta, sample_size, upper_bound)
        
        # Calculate MLE for this sample
        n = len(sample)
        sum_log_x = np.sum(np.log(sample))
        mle = n / (n * np.log(upper_bound) - sum_log_x)
        
        mle_estimates.append(mle)
    
    # Calculate statistics
    mean_mle = np.mean(mle_estimates)
    std_mle = np.std(mle_estimates)
    bias = mean_mle - true_theta
    mse = np.mean((np.array(mle_estimates) - true_theta)**2)
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot histogram of MLEs
    bins = min(50, int(num_samples/20))
    ax.hist(mle_estimates, bins=bins, density=True, alpha=0.7, color='skyblue',
            edgecolor='black', linewidth=0.5)
    
    # Overlay normal approximation based on observed mean and std
    x = np.linspace(min(mle_estimates), max(mle_estimates), 1000)
    y = stats.norm.pdf(x, loc=mean_mle, scale=std_mle)
    ax.plot(x, y, 'r-', linewidth=2, label='Normal Approximation')
    
    # Mark the true value and mean MLE
    ax.axvline(x=true_theta, color='green', linestyle='--', linewidth=2, 
               label=f'True θ = {true_theta:.4f}')
    ax.axvline(x=mean_mle, color='blue', linestyle='-', linewidth=2, 
               label=f'Mean MLE = {mean_mle:.4f}')
    
    # Add text annotations for statistics
    stats_text = (
        f'Number of Samples: {num_samples}\n'
        f'Sample Size: {sample_size}\n'
        f'Mean MLE: {mean_mle:.4f}\n'
        f'Std Dev of MLE: {std_mle:.4f}\n'
        f'Bias: {bias:.4f}\n'
        f'MSE: {mse:.4f}'
    )
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    ax.set_xlabel('MLE Estimates (θ̂)')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Sampling Distribution of MLE for Power Law (True θ = {true_theta}, n = {sample_size})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sampling distribution figure saved to {save_path}")
    
    plt.close()
    
    return mean_mle, std_mle, bias, mse

def plot_asymptotic_properties(true_theta=1.5, sample_sizes=[10, 50, 100, 500, 1000], 
                             num_samples=500, upper_bound=3, save_path=None):
    """
    Demonstrates the asymptotic properties of the MLE estimator by showing how
    bias, variance, and MSE change with increasing sample size
    """
    np.random.seed(42)  # For reproducibility
    
    # Function to generate samples from power law distribution
    def generate_power_law_sample(theta, size, upper_bound=3):
        # Use inverse transform sampling
        u = np.random.uniform(0, 1, size=size)
        return upper_bound * (u ** (1/theta))
    
    # Generate statistics for each sample size
    results = []
    for n in sample_sizes:
        mle_estimates = []
        for _ in range(num_samples):
            sample = generate_power_law_sample(true_theta, n, upper_bound)
            sum_log_x = np.sum(np.log(sample))
            mle = n / (n * np.log(upper_bound) - sum_log_x)
            mle_estimates.append(mle)
        
        mean_mle = np.mean(mle_estimates)
        var_mle = np.var(mle_estimates)
        bias = mean_mle - true_theta
        mse = np.mean((np.array(mle_estimates) - true_theta)**2)
        
        results.append({
            'sample_size': n,
            'mean_mle': mean_mle,
            'variance': var_mle,
            'bias': bias,
            'mse': mse
        })
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data for plotting
    sizes = [r['sample_size'] for r in results]
    means = [r['mean_mle'] for r in results]
    variances = [r['variance'] for r in results]
    biases = [r['bias'] for r in results]
    mses = [r['mse'] for r in results]
    
    # Plot 1: Mean MLE estimates
    axs[0, 0].plot(sizes, means, 'bo-', linewidth=2)
    axs[0, 0].axhline(y=true_theta, color='r', linestyle='--', 
                     label=f'True θ = {true_theta}')
    axs[0, 0].set_title('Mean of MLE Estimates')
    axs[0, 0].set_xlabel('Sample Size (n)')
    axs[0, 0].set_ylabel('Mean θ̂')
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()
    
    # Plot 2: Variance of MLEs
    axs[0, 1].plot(sizes, variances, 'go-', linewidth=2)
    
    # Overlay theoretical asymptotic variance
    theoretical_var = [(true_theta**2) / n for n in sizes]
    axs[0, 1].plot(sizes, theoretical_var, 'r--', 
                  label='Theoretical Asymptotic Variance')
    
    axs[0, 1].set_title('Variance of MLE Estimates')
    axs[0, 1].set_xlabel('Sample Size (n)')
    axs[0, 1].set_ylabel('Var(θ̂)')
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend()
    
    # Plot 3: Bias of MLEs
    axs[1, 0].plot(sizes, biases, 'mo-', linewidth=2)
    axs[1, 0].axhline(y=0, color='r', linestyle='--', label='Zero Bias')
    axs[1, 0].set_title('Bias of MLE Estimates')
    axs[1, 0].set_xlabel('Sample Size (n)')
    axs[1, 0].set_ylabel('Bias(θ̂)')
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend()
    
    # Plot 4: MSE of MLEs
    axs[1, 1].plot(sizes, mses, 'co-', linewidth=2)
    axs[1, 1].plot(sizes, theoretical_var, 'r--', 
                  label='Theoretical Asymptotic Variance')
    axs[1, 1].set_title('Mean Squared Error of MLE Estimates')
    axs[1, 1].set_xlabel('Sample Size (n)')
    axs[1, 1].set_ylabel('MSE(θ̂)')
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend()
    
    # Use log scale for x-axis to better see the asymptotic behavior
    for ax in axs.flat:
        ax.set_xscale('log')
    
    plt.suptitle(f'Asymptotic Properties of MLE for Power Law (True θ = {true_theta})', 
                fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Asymptotic properties figure saved to {save_path}")
    
    plt.close()
    
    return results

def plot_confidence_intervals(data, upper_bound=3, confidence=0.95, save_path=None):
    """
    Visualizes the construction of confidence intervals for the MLE estimator
    """
    # Calculate MLE
    n = len(data)
    sum_log_x = np.sum(np.log(data))
    mle_theta = n / (n * np.log(upper_bound) - sum_log_x)
    
    # Calculate asymptotic variance of MLE
    asymp_var = mle_theta**2 / n
    asymp_std = np.sqrt(asymp_var)
    
    # Z critical value for the given confidence level
    z_critical = stats.norm.ppf((1 + confidence) / 2)
    
    # Calculate confidence interval
    ci_lower = mle_theta - z_critical * asymp_std
    ci_upper = mle_theta + z_critical * asymp_std
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate theta values for plotting
    theta_range = np.linspace(max(0.1, mle_theta - 3*asymp_std), 
                              mle_theta + 3*asymp_std, 1000)
    
    # Calculate the log-likelihood for each theta
    log_likelihoods = []
    for theta in theta_range:
        ll = log_likelihood(theta, data, upper_bound)
        log_likelihoods.append(ll)
    
    # Normalize log-likelihoods for better visualization
    ll_array = np.array(log_likelihoods)
    ll_normalized = (ll_array - np.min(ll_array)) / (np.max(ll_array) - np.min(ll_array))
    
    # Plot the normalized log-likelihood function
    ax.plot(theta_range, ll_normalized, 'b-', linewidth=2)
    
    # Shade the confidence interval region
    ci_region = (theta_range >= ci_lower) & (theta_range <= ci_upper)
    ax.fill_between(theta_range, 0, ll_normalized, where=ci_region, 
                   color='green', alpha=0.3, label=f'{confidence*100:.0f}% CI')
    
    # Mark the MLE point
    ax.axvline(x=mle_theta, color='r', linestyle='-', 
              label=f'MLE θ̂ = {mle_theta:.4f}')
    
    # Mark CI boundaries
    ax.axvline(x=ci_lower, color='g', linestyle='--')
    ax.axvline(x=ci_upper, color='g', linestyle='--')
    
    # Add text annotations
    ci_text = (
        f'MLE Estimate: θ̂ = {mle_theta:.4f}\n'
        f'Standard Error: SE(θ̂) = {asymp_std:.4f}\n'
        f'{confidence*100:.0f}% Confidence Interval:\n'
        f'[{ci_lower:.4f}, {ci_upper:.4f}]'
    )
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.03, 0.97, ci_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left', bbox=props)
    
    ax.set_xlabel('θ (Shape Parameter)')
    ax.set_ylabel('Normalized Log-Likelihood')
    ax.set_title(f'Maximum Likelihood Estimate with {confidence*100:.0f}% Confidence Interval')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confidence interval figure saved to {save_path}")
    
    plt.close()
    
    return mle_theta, asymp_std, ci_lower, ci_upper

def plot_interactive_mle_calculation(save_path=None):
    """
    Creates an interactive-style visualization showing how the MLE calculation
    changes with different data points
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Define multiple datasets to demonstrate
    datasets = [
        {"name": "Uniformly Spread", "data": np.array([0.5, 1.0, 1.5, 2.0, 2.5])},
        {"name": "Small Values", "data": np.array([0.1, 0.3, 0.2, 0.4, 0.5])},
        {"name": "Large Values", "data": np.array([2.0, 2.2, 2.4, 2.6, 2.8])},
        {"name": "Mixed Values", "data": np.array([0.2, 0.8, 1.5, 2.2, 2.8])}
    ]
    
    upper_bound = 3
    
    # Calculate MLE for each dataset
    for dataset in datasets:
        n = len(dataset["data"])
        sum_log_x = np.sum(np.log(dataset["data"]))
        dataset["mle"] = n / (n * np.log(upper_bound) - sum_log_x)
        dataset["log_ratios"] = [np.log(upper_bound / x) for x in dataset["data"]]
        dataset["mean_log_ratio"] = np.mean(dataset["log_ratios"])
    
    # Plot 1: Datasets visualization
    ax1 = axs[0, 0]
    bar_width = 0.2
    x_positions = np.arange(5)  # Assuming all datasets have 5 points
    
    for i, dataset in enumerate(datasets):
        offset = (i - 1.5) * bar_width
        bars = ax1.bar(x_positions + offset, dataset["data"], width=bar_width, 
                      label=dataset["name"])
    
    ax1.set_xlabel('Data Point Index')
    ax1.set_ylabel('Value')
    ax1.set_title('Different Datasets for MLE Calculation')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([f'x_{i+1}' for i in range(5)])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log ratios for each dataset
    ax2 = axs[0, 1]
    
    for i, dataset in enumerate(datasets):
        offset = (i - 1.5) * bar_width
        bars = ax2.bar(x_positions + offset, dataset["log_ratios"], width=bar_width,
                      label=f'{dataset["name"]} (Mean: {dataset["mean_log_ratio"]:.3f})')
    
    ax2.set_xlabel('Data Point Index')
    ax2.set_ylabel('log(3/x_i)')
    ax2.set_title('Log Ratios Used in MLE Calculation')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([f'x_{i+1}' for i in range(5)])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: MLE calculation results
    ax3 = axs[1, 0]
    dataset_names = [d["name"] for d in datasets]
    mle_values = [d["mle"] for d in datasets]
    
    bars = ax3.bar(dataset_names, mle_values, color='skyblue')
    
    # Add value labels on top of bars
    for bar, value in zip(bars, mle_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{value:.3f}', ha='center', va='bottom')
    
    ax3.set_xlabel('Dataset')
    ax3.set_ylabel('MLE Estimate (θ̂)')
    ax3.set_title('MLE Estimates for Different Datasets')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: PDF curves with MLE estimates
    ax4 = axs[1, 1]
    x_range = np.linspace(0.01, upper_bound, 1000)
    
    for i, dataset in enumerate(datasets):
        y_values = power_law_pdf(x_range, dataset["mle"], upper_bound)
        ax4.plot(x_range, y_values, linewidth=2, 
                label=f'{dataset["name"]} (θ̂ = {dataset["mle"]:.3f})')
    
    ax4.set_xlabel('x')
    ax4.set_ylabel('Probability Density f(x|θ̂)')
    ax4.set_title('Fitted Power Law PDFs with MLE Estimates')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Interactive Exploration of MLE Calculation', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Interactive MLE calculation figure saved to {save_path}")
    
    plt.close()
    
    return datasets

if __name__ == "__main__":
    main() 