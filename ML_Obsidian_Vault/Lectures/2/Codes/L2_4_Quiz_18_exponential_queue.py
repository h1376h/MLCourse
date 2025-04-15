import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon
import os

def plot_exponential_pdf(lambda_values, x_max=10, save_path=None):
    """Plot the PDF of exponential distribution for different lambda values"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.linspace(0, x_max, 1000)
    
    for lambda_val in lambda_values:
        pdf = lambda_val * np.exp(-lambda_val * x)
        ax.plot(x, pdf, label=f'λ = {lambda_val:.2f}')
    
    ax.set_xlabel('Waiting Time (minutes)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Exponential Distribution PDF for Different λ Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def compute_mle_for_exponential(data, save_path=None):
    """Compute MLE for exponential distribution and visualize"""
    # Calculate MLE
    lambda_hat = 1 / np.mean(data)
    mean_hat = 1 / lambda_hat
    std_hat = mean_hat  # For exponential, std = mean
    
    # Calculate log-likelihood function
    lambda_range = np.linspace(max(0.1, lambda_hat*0.5), lambda_hat*1.5, 1000)
    log_likelihood = np.array([np.sum(np.log(lambda_val) - lambda_val * data) for lambda_val in lambda_range])
    
    # Create visualization of the likelihood function
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(lambda_range, log_likelihood, 'b-', linewidth=2)
    ax.axvline(x=lambda_hat, color='r', linestyle='--', 
               label=f'MLE λ = {lambda_hat:.4f}')
    
    ax.set_xlabel('Rate Parameter (λ)')
    ax.set_ylabel('Log-Likelihood ℓ(λ)')
    ax.set_title('Log-Likelihood Function for Exponential Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add text with key statistics
    text_str = '\n'.join((
        f'Sample Size: {len(data)}',
        f'Sample Mean: {np.mean(data):.4f}',
        f'MLE λ: {lambda_hat:.4f}',
        f'Estimated Mean: {mean_hat:.4f}',
        f'Estimated Std Dev: {std_hat:.4f}'))
    
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return lambda_hat, mean_hat, std_hat

def plot_data_fit(data, lambda_hat, save_path=None):
    """Plot the data and fitted exponential distribution"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate x values for plotting the PDF
    x_max = max(data) * 1.5
    x = np.linspace(0, x_max, 1000)
    pdf = lambda_hat * np.exp(-lambda_hat * x)
    
    # Plot the PDF
    ax.plot(x, pdf, 'r-', linewidth=2, 
            label=f'Fitted Exponential(λ={lambda_hat:.4f})')
    
    # Plot histogram of the data
    ax.hist(data, bins='auto', density=True, alpha=0.5, color='blue',
            label=f'Waiting Time Data (n={len(data)})')
    
    # Mark the data points
    ax.plot(data, np.zeros_like(data), 'bo', markersize=8, alpha=0.6,
            label='Observed Waiting Times')
    
    ax.set_xlabel('Waiting Time (minutes)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Coffee Shop Waiting Time - Exponential Distribution Fit')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_cdf_with_percentiles(data, lambda_hat, save_path=None):
    """Plot CDF with percentiles"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate x values for plotting the CDF
    x_max = max(data) * 1.5
    x = np.linspace(0, x_max, 1000)
    cdf = 1 - np.exp(-lambda_hat * x)
    
    # Plot the CDF
    ax.plot(x, cdf, 'r-', linewidth=2, 
            label=f'Exponential CDF (λ={lambda_hat:.4f})')
    
    # Add percentile lines
    percentiles = [0.25, 0.5, 0.75, 0.9]
    colors = ['green', 'blue', 'purple', 'orange']
    
    for p, color in zip(percentiles, colors):
        x_p = -np.log(1-p) / lambda_hat
        ax.axhline(y=p, color=color, linestyle='--', alpha=0.5)
        ax.axvline(x=x_p, color=color, linestyle='--', alpha=0.5)
        ax.plot([0, x_p], [p, p], color, linestyle='--')
        ax.plot([x_p, x_p], [0, p], color, linestyle='--')
        ax.text(x_p + 0.1, p - 0.05, f'{int(p*100)}th percentile: {x_p:.2f} min', 
                color=color, fontweight='bold')
    
    # Plot the empirical CDF
    sorted_data = np.sort(data)
    empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax.step(sorted_data, empirical_cdf, 'b-', alpha=0.7, 
            label='Empirical CDF')
    
    ax.set_xlabel('Waiting Time (minutes)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Waiting Time Cumulative Distribution Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def main():
    """Generate all visualizations for Question 18 of the L2.4 quiz"""
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_4_Quiz_18")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 18 of the L2.4 MLE quiz...")
    
    # Waiting time data from the problem (in minutes)
    waiting_times = np.array([2.5, 1.8, 3.2, 2.0, 2.7, 1.5])
    
    # 1. Plot exponential PDF for different lambda values
    plot_exponential_pdf([0.2, 0.4, 0.45, 0.6], x_max=10, 
                        save_path=os.path.join(save_dir, "exponential_pdf.png"))
    print("1. Exponential PDF visualization created")
    
    # 2. Compute MLE and visualize likelihood function
    lambda_hat, mean_hat, std_hat = compute_mle_for_exponential(
        waiting_times, save_path=os.path.join(save_dir, "likelihood_function.png"))
    print("2. Likelihood function visualization created")
    print(f"   MLE for rate parameter λ: {lambda_hat:.4f}")
    print(f"   Estimated mean waiting time: {mean_hat:.4f} minutes")
    print(f"   Estimated standard deviation: {std_hat:.4f} minutes")
    
    # 3. Plot data fit
    plot_data_fit(waiting_times, lambda_hat, 
                 save_path=os.path.join(save_dir, "data_fit.png"))
    print("3. Data fit visualization created")
    
    # 4. Plot CDF with percentiles
    plot_cdf_with_percentiles(waiting_times, lambda_hat, 
                             save_path=os.path.join(save_dir, "cdf_percentiles.png"))
    print("4. CDF with percentiles visualization created")
    
    # 5. Print key statistics for reference
    print("\nKey Statistics:")
    print(f"   Sample size: {len(waiting_times)}")
    print(f"   Sample mean: {np.mean(waiting_times):.4f} minutes")
    print(f"   Sample standard deviation: {np.std(waiting_times, ddof=1):.4f} minutes")
    
    print(f"\nAll visualizations have been saved to: {save_dir}")

if __name__ == "__main__":
    main() 