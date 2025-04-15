import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as stats
from matplotlib.patches import Patch

def calculate_poisson_probabilities(rate_per_minute=3):
    """
    Calculate various probabilities for the Poisson distribution scenario.
    
    Args:
        rate_per_minute: Average rate of requests per minute
    
    Returns:
        Dictionary with calculated probabilities
    """
    # Calculate lambda (mean) for different time intervals
    lambda_1min = rate_per_minute
    lambda_2min = rate_per_minute * 2
    lambda_10min = rate_per_minute * 10
    
    # Task 1: Probability of exactly 5 requests in a 2-minute interval
    p_exactly_5_in_2min = stats.poisson.pmf(5, lambda_2min)
    
    # Task 2: Probability of no requests in a 1-minute interval
    p_no_requests_in_1min = stats.poisson.pmf(0, lambda_1min)
    
    # Task 3: Expected number of requests in a 10-minute interval
    expected_requests_10min = lambda_10min
    
    # Task 4: Probability of system overload (more than 8 requests in 2 minutes)
    p_overload_in_2min = 1 - stats.poisson.cdf(8, lambda_2min)
    
    # Additional useful calculations for visualizations
    # PMF values for 0 to 20 requests in a 2-minute interval
    k_values_2min = np.arange(0, 21)
    pmf_values_2min = stats.poisson.pmf(k_values_2min, lambda_2min)
    
    # CDF values for 0 to 20 requests in a 2-minute interval
    cdf_values_2min = stats.poisson.cdf(k_values_2min, lambda_2min)
    
    return {
        'lambda_1min': lambda_1min,
        'lambda_2min': lambda_2min,
        'lambda_10min': lambda_10min,
        'p_exactly_5_in_2min': p_exactly_5_in_2min,
        'p_no_requests_in_1min': p_no_requests_in_1min,
        'expected_requests_10min': expected_requests_10min,
        'p_overload_in_2min': p_overload_in_2min,
        'k_values_2min': k_values_2min,
        'pmf_values_2min': pmf_values_2min,
        'cdf_values_2min': cdf_values_2min
    }

def create_pmf_visualization(probs, save_path=None):
    """
    Create a visualization of the Poisson PMF for a 2-minute interval.
    
    Args:
        probs: Dictionary with calculated probabilities
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Extract data
    k_values = probs['k_values_2min']
    pmf_values = probs['pmf_values_2min']
    lambda_2min = probs['lambda_2min']
    
    # Create bar plot
    bars = plt.bar(k_values, pmf_values, color='#1f77b4', alpha=0.7)
    
    # Highlight specific bars
    # Exactly 5 requests
    bars[5].set_color('#d62728')
    
    # Add grid and labels
    plt.grid(True, alpha=0.3)
    plt.xlabel('Number of Requests in 2 Minutes', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title(f'Poisson PMF with λ = {lambda_2min} (2-minute interval)', fontsize=14)
    
    # Annotate the probability of exactly 5 requests
    plt.annotate(f'P(X=5) = {probs["p_exactly_5_in_2min"]:.4f}', 
                xy=(5, probs['pmf_values_2min'][5]), 
                xytext=(5, probs['pmf_values_2min'][5] + 0.02),
                ha='center', fontsize=10,
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PMF visualization saved to {save_path}")
    
    plt.close()

def create_system_overload_visualization(probs, save_path=None):
    """
    Create a visualization showing the system overload probability.
    
    Args:
        probs: Dictionary with calculated probabilities
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Extract data
    k_values = probs['k_values_2min']
    pmf_values = probs['pmf_values_2min']
    lambda_2min = probs['lambda_2min']
    
    # Create bar plot with color coding
    colors = ['#1f77b4' if k <= 8 else '#d62728' for k in k_values]
    bars = plt.bar(k_values, pmf_values, color=colors, alpha=0.7)
    
    # Add vertical line at threshold
    plt.axvline(x=8.5, color='red', linestyle='--', alpha=0.7, label='Overload Threshold')
    
    # Add grid and labels
    plt.grid(True, alpha=0.3)
    plt.xlabel('Number of Requests in 2 Minutes', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title(f'System Overload Analysis with λ = {lambda_2min} (2-minute interval)', fontsize=14)
    
    # Create custom legend
    legend_elements = [
        Patch(facecolor='#1f77b4', alpha=0.7, label='System Handles (≤ 8)'),
        Patch(facecolor='#d62728', alpha=0.7, label='System Overloaded (> 8)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Annotate the overload probability
    plt.annotate(f'P(X > 8) = {probs["p_overload_in_2min"]:.4f}', 
                xy=(12, 0.05), 
                xytext=(12, 0.08),
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.7),
                arrowprops=dict(facecolor='red', shrink=0.05, width=1.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"System overload visualization saved to {save_path}")
    
    plt.close()

def create_cdf_visualization(probs, save_path=None):
    """
    Create a visualization of the Poisson CDF for a 2-minute interval.
    
    Args:
        probs: Dictionary with calculated probabilities
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Extract data
    k_values = probs['k_values_2min']
    cdf_values = probs['cdf_values_2min']
    lambda_2min = probs['lambda_2min']
    
    # Plot CDF as step function
    plt.step(k_values, cdf_values, where='post', color='#1f77b4', linewidth=2)
    plt.scatter(k_values, cdf_values, color='#1f77b4', s=50)
    
    # Highlight the CDF value at x=8
    plt.plot(8, cdf_values[8], 'ro', markersize=10)
    
    # Add a horizontal line at the CDF value
    plt.axhline(y=cdf_values[8], color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=8, color='red', linestyle='--', alpha=0.5)
    
    # Add grid and labels
    plt.grid(True, alpha=0.3)
    plt.xlabel('Number of Requests in 2 Minutes', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.title(f'Poisson CDF with λ = {lambda_2min} (2-minute interval)', fontsize=14)
    
    # Annotate probability of at most 8 requests
    plt.annotate(f'P(X ≤ 8) = {cdf_values[8]:.4f}', 
                xy=(8, cdf_values[8]), 
                xytext=(10, cdf_values[8] - 0.1),
                fontsize=12,
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    # Annotate probability of more than 8 requests (overload)
    plt.annotate(f'P(X > 8) = {1-cdf_values[8]:.4f}', 
                xy=(8, cdf_values[8]), 
                xytext=(10, cdf_values[8] + 0.1),
                fontsize=12,
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"CDF visualization saved to {save_path}")
    
    plt.close()

def create_time_scaling_visualization(probs, save_path=None):
    """
    Create a visualization showing how Poisson probabilities scale with time.
    
    Args:
        probs: Dictionary with calculated probabilities
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Time intervals to show in minutes
    time_intervals = [1, 2, 5, 10]
    
    # Range of possible number of requests to show
    max_k = 30
    k_values = np.arange(0, max_k + 1)
    
    # Plot PMF for each time interval
    for interval in time_intervals:
        lambda_t = probs['lambda_1min'] * interval
        pmf_values = stats.poisson.pmf(k_values, lambda_t)
        
        plt.plot(k_values, pmf_values, linewidth=2, 
                 marker='o', markersize=4, alpha=0.7,
                 label=f'{interval} min (λ = {lambda_t})')
    
    # Mark expected values
    for interval in time_intervals:
        lambda_t = probs['lambda_1min'] * interval
        plt.axvline(x=lambda_t, color='gray', linestyle='--', alpha=0.5)
        plt.scatter([lambda_t], [stats.poisson.pmf(int(lambda_t), lambda_t)], 
                   s=100, color='red', zorder=5)
        plt.annotate(f'E[X] = {lambda_t}', 
                    xy=(lambda_t, stats.poisson.pmf(int(lambda_t), lambda_t)),
                    xytext=(lambda_t, stats.poisson.pmf(int(lambda_t), lambda_t) + 0.01),
                    ha='center', fontsize=10)
    
    # Add grid and labels
    plt.grid(True, alpha=0.3)
    plt.xlabel('Number of Requests', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title('Poisson Distributions for Different Time Intervals', fontsize=14)
    plt.legend(title='Time Interval', loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Time scaling visualization saved to {save_path}")
    
    plt.close()

def main():
    """Generate all visualizations for Question 16 of the L2.1 quiz"""
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_1_Quiz_16")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 16 of the L2.1 Probability quiz: Poisson Distribution Application...")
    
    # Calculate probabilities
    probs = calculate_poisson_probabilities(rate_per_minute=3)
    
    # Print results
    print("\nProbabilities for Poisson Distribution:")
    print(f"Rate parameter (λ) per minute: {probs['lambda_1min']}")
    print(f"Rate parameter (λ) for 2 minutes: {probs['lambda_2min']}")
    print(f"Rate parameter (λ) for 10 minutes: {probs['lambda_10min']}")
    print(f"Task 1: P(X=5) in 2 minutes: {probs['p_exactly_5_in_2min']:.6f}")
    print(f"Task 2: P(X=0) in 1 minute: {probs['p_no_requests_in_1min']:.6f}")
    print(f"Task 3: Expected requests in 10 minutes: {probs['expected_requests_10min']}")
    print(f"Task 4: P(X>8) in 2 minutes (system overload): {probs['p_overload_in_2min']:.6f}")
    
    # Generate visualizations
    create_pmf_visualization(probs, save_path=os.path.join(save_dir, "poisson_pmf.png"))
    print("1. PMF visualization created")
    
    create_system_overload_visualization(probs, save_path=os.path.join(save_dir, "system_overload.png"))
    print("2. System overload visualization created")
    
    create_cdf_visualization(probs, save_path=os.path.join(save_dir, "poisson_cdf.png"))
    print("3. CDF visualization created")
    
    create_time_scaling_visualization(probs, save_path=os.path.join(save_dir, "time_scaling.png"))
    print("4. Time scaling visualization created")
    
    print(f"\nAll visualizations have been saved to: {save_dir}")

if __name__ == "__main__":
    main() 