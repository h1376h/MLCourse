import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
from matplotlib.patches import Patch

def calculate_uniform_statistics(a=1, b=5):
    """
    Calculate statistics for a uniform distribution U(a, b)
    
    Args:
        a: Lower bound
        b: Upper bound
    
    Returns:
        Dictionary with mean, variance, and standard deviation
    """
    # Mean of uniform distribution
    mean = (a + b) / 2
    
    # Variance of uniform distribution
    variance = (b - a)**2 / 12
    
    # Standard deviation
    std_dev = np.sqrt(variance)
    
    return {
        'mean': mean,
        'variance': variance,
        'std_dev': std_dev
    }

def calculate_sample_mean_statistics(pop_mean, pop_variance, n=36):
    """
    Calculate statistics for the sampling distribution of the mean
    
    Args:
        pop_mean: Population mean
        pop_variance: Population variance
        n: Sample size
    
    Returns:
        Dictionary with mean, variance, and standard deviation of the sampling distribution
    """
    # Mean of the sampling distribution
    sample_mean = pop_mean
    
    # Variance of the sampling distribution
    sample_variance = pop_variance / n
    
    # Standard deviation of the sampling distribution (standard error)
    standard_error = np.sqrt(sample_variance)
    
    return {
        'mean': sample_mean,
        'variance': sample_variance,
        'std_dev': standard_error
    }

def calculate_probability_between(mean, std_dev, lower, upper):
    """
    Calculate probability of a normal random variable being between two values
    
    Args:
        mean: Mean of the normal distribution
        std_dev: Standard deviation of the normal distribution
        lower: Lower bound
        upper: Upper bound
    
    Returns:
        Probability that the random variable is between lower and upper
    """
    # Standardize the lower and upper bounds
    z_lower = (lower - mean) / std_dev
    z_upper = (upper - mean) / std_dev
    
    # Calculate the probability using the CDF of the standard normal distribution
    prob = stats.norm.cdf(z_upper) - stats.norm.cdf(z_lower)
    
    return prob

def calculate_required_sample_size(pop_variance, margin_of_error, confidence_level=0.95):
    """
    Calculate the required sample size for a given margin of error and confidence level
    
    Args:
        pop_variance: Population variance
        margin_of_error: Desired margin of error
        confidence_level: Confidence level (default: 0.95)
    
    Returns:
        Required sample size (rounded up)
    """
    # Critical value for the given confidence level
    # For a 95% confidence level, the critical value is 1.96
    alpha = 1 - confidence_level
    z_critical = stats.norm.ppf(1 - alpha/2)
    
    # Calculate the required sample size
    n = (z_critical**2 * pop_variance) / (margin_of_error**2)
    
    # Round up to the nearest integer
    return int(np.ceil(n))

def visualize_uniform_distribution(a=1, b=5, save_path=None):
    """
    Create a visualization of the uniform distribution
    
    Args:
        a: Lower bound
        b: Upper bound
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 6))
    
    # Create x values for the uniform distribution
    x = np.linspace(0, 6, 1000)
    
    # Calculate PDF values
    pdf = np.zeros_like(x)
    pdf[(x >= a) & (x <= b)] = 1/(b-a)
    
    # Plot the PDF
    plt.plot(x, pdf, 'b-', lw=2, label=f'Uniform PDF U({a}, {b})')
    plt.fill_between(x, pdf, alpha=0.3)
    
    # Calculate statistics
    stats_dict = calculate_uniform_statistics(a, b)
    mean = stats_dict['mean']
    std_dev = stats_dict['std_dev']
    
    # Add lines for mean and ±1 standard deviation
    plt.axvline(mean, color='r', linestyle='--', lw=2, label=f'Mean = {mean}')
    plt.axvline(mean - std_dev, color='g', linestyle=':', lw=2, label=f'Mean ± σ = {mean-std_dev:.2f}, {mean+std_dev:.2f}')
    plt.axvline(mean + std_dev, color='g', linestyle=':', lw=2)
    
    # Add labels and legend
    plt.xlabel('Package Weight (kg)', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.title('Uniform Distribution of Package Weights', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Annotate the variance
    plt.text(0.7, 0.15, f'Variance = {stats_dict["variance"]:.4f}', transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Uniform distribution visualization saved to {save_path}")
    
    plt.close()

def visualize_sampling_distribution(pop_mean, pop_std, n=36, save_path=None):
    """
    Create a visualization of the sampling distribution of the mean
    
    Args:
        pop_mean: Population mean
        pop_std: Population standard deviation
        n: Sample size
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate the standard error
    se = pop_std / np.sqrt(n)
    
    # Create x values for the normal distribution
    x = np.linspace(pop_mean - 4*se, pop_mean + 4*se, 1000)
    
    # Calculate the PDF values
    pdf = stats.norm.pdf(x, pop_mean, se)
    
    # Plot the PDF
    plt.plot(x, pdf, 'b-', lw=2, label=f'Normal PDF N({pop_mean:.2f}, {se:.4f}²)')
    plt.fill_between(x, pdf, alpha=0.3)
    
    # Add lines for mean and ±1 standard error
    plt.axvline(pop_mean, color='r', linestyle='--', lw=2, label=f'Mean = {pop_mean}')
    plt.axvline(pop_mean - se, color='g', linestyle=':', lw=2, label=f'Mean ± SE = {pop_mean-se:.4f}, {pop_mean+se:.4f}')
    plt.axvline(pop_mean + se, color='g', linestyle=':', lw=2)
    
    # Add labels and legend
    plt.xlabel('Sample Mean Weight (kg)', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.title(f'Sampling Distribution of the Mean (n={n})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add annotation for the standard error
    plt.text(0.7, 0.85, f'Standard Error = {se:.4f}', transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8))
    plt.text(0.7, 0.78, f'Variance = {se**2:.6f}', transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sampling distribution visualization saved to {save_path}")
    
    plt.close()

def visualize_probability_interval(pop_mean, pop_std, n=36, lower=2.8, upper=3.2, save_path=None):
    """
    Create a visualization of the probability of the sample mean being in a specific interval
    
    Args:
        pop_mean: Population mean
        pop_std: Population standard deviation
        n: Sample size
        lower: Lower bound of the interval
        upper: Upper bound of the interval
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate the standard error
    se = pop_std / np.sqrt(n)
    
    # Create x values for the normal distribution
    x = np.linspace(pop_mean - 4*se, pop_mean + 4*se, 1000)
    
    # Calculate the PDF values
    pdf = stats.norm.pdf(x, pop_mean, se)
    
    # Plot the PDF
    plt.plot(x, pdf, 'b-', lw=2, label=f'Normal PDF N({pop_mean:.2f}, {se:.4f}²)')
    
    # Highlight the region of interest
    mask = (x >= lower) & (x <= upper)
    plt.fill_between(x, pdf, where=mask, alpha=0.5, color='green')
    
    # Add vertical lines for the bounds
    plt.axvline(lower, color='g', linestyle='-', lw=2, label=f'Bounds: [{lower}, {upper}]')
    plt.axvline(upper, color='g', linestyle='-', lw=2)
    
    # Calculate the probability
    prob = calculate_probability_between(pop_mean, se, lower, upper)
    
    # Add labels and legend
    plt.xlabel('Sample Mean Weight (kg)', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.title(f'Probability of Sample Mean in [{lower}, {upper}]', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add annotation for the probability
    plt.text(0.65, 0.85, f'Probability = {prob:.4f}', transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Add standardized values
    z_lower = (lower - pop_mean) / se
    z_upper = (upper - pop_mean) / se
    plt.text(0.65, 0.78, f'z-values: [{z_lower:.2f}, {z_upper:.2f}]', transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Probability interval visualization saved to {save_path}")
    
    plt.close()

def visualize_sample_size_calculation(pop_variance, margin_of_error=0.2, confidence_level=0.95, save_path=None):
    """
    Create a visualization of the required sample size for different margins of error
    
    Args:
        pop_variance: Population variance
        margin_of_error: Target margin of error
        confidence_level: Confidence level
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 6))
    
    # Create a range of margins of error
    margins = np.linspace(0.05, 0.5, 100)
    
    # Calculate the required sample size for each margin of error
    sample_sizes = [calculate_required_sample_size(pop_variance, m, confidence_level) for m in margins]
    
    # Plot the relationship
    plt.plot(margins, sample_sizes, 'b-', lw=2)
    
    # Highlight the margin of error of interest
    required_n = calculate_required_sample_size(pop_variance, margin_of_error, confidence_level)
    plt.scatter(margin_of_error, required_n, color='r', s=100, zorder=3)
    
    # Add text annotation
    plt.annotate(f'({margin_of_error}, {required_n})', 
                xy=(margin_of_error, required_n),
                xytext=(margin_of_error+0.05, required_n+20),
                fontsize=12,
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
    
    # Add labels and title
    plt.xlabel('Margin of Error (kg)', fontsize=12)
    plt.ylabel('Required Sample Size', fontsize=12)
    plt.title(f'Required Sample Size for {confidence_level*100:.0f}% Confidence Level', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add annotation
    alpha = 1 - confidence_level
    z_critical = stats.norm.ppf(1 - alpha/2)
    plt.text(0.65, 0.85, f'z-critical = {z_critical:.4f}', transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8))
    plt.text(0.65, 0.78, f'Population Variance = {pop_variance:.4f}', transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Formula annotation
    formula = r'$n = \frac{z^2 \sigma^2}{E^2}$'
    plt.text(0.65, 0.71, f'Formula: {formula}', transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample size calculation visualization saved to {save_path}")
    
    plt.close()

def visualize_clt_simulation(a=1, b=5, n=36, num_samples=10000, save_path=None):
    """
    Create a visualization of the Central Limit Theorem through simulation
    
    Args:
        a: Lower bound of uniform distribution
        b: Upper bound of uniform distribution
        n: Sample size
        num_samples: Number of sample means to generate
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(12, 8))
    
    # Generate random samples from the uniform distribution
    np.random.seed(42)  # For reproducibility
    
    # Generate samples and calculate sample means
    sample_means = np.zeros(num_samples)
    for i in range(num_samples):
        sample = np.random.uniform(a, b, n)
        sample_means[i] = np.mean(sample)
    
    # Calculate theoretical values
    pop_stats = calculate_uniform_statistics(a, b)
    sample_stats = calculate_sample_mean_statistics(pop_stats['mean'], pop_stats['variance'], n)
    
    # Plot histogram of sample means
    plt.hist(sample_means, bins=50, density=True, alpha=0.6, label=f'Histogram of {num_samples} Sample Means')
    
    # Plot the theoretical normal distribution
    x = np.linspace(min(sample_means), max(sample_means), 1000)
    pdf = stats.norm.pdf(x, sample_stats['mean'], sample_stats['std_dev'])
    plt.plot(x, pdf, 'r-', lw=2, label=f'Normal PDF N({sample_stats["mean"]:.2f}, {sample_stats["std_dev"]:.4f}²)')
    
    # Add vertical line for theoretical mean
    plt.axvline(pop_stats['mean'], color='g', linestyle='--', lw=2, label=f'Theoretical Mean = {pop_stats["mean"]}')
    
    # Calculate empirical statistics
    empirical_mean = np.mean(sample_means)
    empirical_std = np.std(sample_means)
    
    # Add annotation
    plt.text(0.02, 0.95, f'Theoretical Mean = {pop_stats["mean"]:.4f}', transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8))
    plt.text(0.02, 0.89, f'Empirical Mean = {empirical_mean:.4f}', transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8))
    plt.text(0.02, 0.83, f'Theoretical Std Dev = {sample_stats["std_dev"]:.4f}', transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8))
    plt.text(0.02, 0.77, f'Empirical Std Dev = {empirical_std:.4f}', transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Add labels and legend
    plt.xlabel('Sample Mean Weight (kg)', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.title(f'Central Limit Theorem Demonstration (n={n}, {num_samples} simulations)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"CLT simulation visualization saved to {save_path}")
    
    plt.close()

def main():
    """Generate all visualizations for Question 13 of the L2.1 quiz"""
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_1_Quiz_13")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 13 of the L2.1 Probability quiz: Central Limit Theorem Application...")
    
    # Problem parameters
    a = 1  # Lower bound of uniform distribution
    b = 5  # Upper bound of uniform distribution
    n = 36  # Sample size
    
    # Calculate the statistics for the uniform distribution
    pop_stats = calculate_uniform_statistics(a, b)
    
    # Task 1: Expected value and variance of a single package weight
    print("\nTask 1: Expected value and variance of a single package weight")
    print(f"Expected value: {pop_stats['mean']} kg")
    print(f"Variance: {pop_stats['variance']} kg²")
    
    # Task 2: Approximate distribution of the sample mean
    print("\nTask 2: Approximate distribution of the sample mean")
    sample_stats = calculate_sample_mean_statistics(pop_stats['mean'], pop_stats['variance'], n)
    print(f"The sample mean follows approximately N({sample_stats['mean']}, {sample_stats['variance']:.6f})")
    print(f"Standard error: {sample_stats['std_dev']:.4f} kg")
    
    # Task 3: Probability that the sample mean is between 2.8kg and 3.2kg
    print("\nTask 3: Probability that the sample mean weight is between 2.8kg and 3.2kg")
    lower_bound = 2.8
    upper_bound = 3.2
    prob = calculate_probability_between(sample_stats['mean'], sample_stats['std_dev'], lower_bound, upper_bound)
    print(f"Probability: {prob:.4f}")
    
    # Task 4: Required sample size for 95% confidence interval with margin of error 0.2kg
    print("\nTask 4: Required sample size for 95% confidence within 0.2kg of true mean")
    margin_of_error = 0.2
    confidence_level = 0.95
    required_n = calculate_required_sample_size(pop_stats['variance'], margin_of_error, confidence_level)
    print(f"Required sample size: {required_n}")
    
    # Generate visualizations
    visualize_uniform_distribution(a, b, save_path=os.path.join(save_dir, "uniform_distribution.png"))
    visualize_sampling_distribution(pop_stats['mean'], pop_stats['std_dev'], n, 
                                  save_path=os.path.join(save_dir, "sampling_distribution.png"))
    visualize_probability_interval(pop_stats['mean'], pop_stats['std_dev'], n, lower_bound, upper_bound,
                                 save_path=os.path.join(save_dir, "probability_interval.png"))
    visualize_sample_size_calculation(pop_stats['variance'], margin_of_error, confidence_level,
                                    save_path=os.path.join(save_dir, "sample_size_calculation.png"))
    visualize_clt_simulation(a, b, n, num_samples=10000, 
                           save_path=os.path.join(save_dir, "clt_simulation.png"))
    
    print(f"\nAll visualizations have been saved to: {save_dir}")

if __name__ == "__main__":
    main() 