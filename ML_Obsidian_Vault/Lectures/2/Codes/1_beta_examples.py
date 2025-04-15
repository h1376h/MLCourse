import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import os

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images/Beta_Distribution relative to the parent directory
images_dir = os.path.join(parent_dir, "Images", "Beta_Distribution")

# Create images directory if it doesn't exist
os.makedirs(images_dir, exist_ok=True)

def plot_beta_distribution(alpha, beta_param, title, filename, show_cdf=False):
    x = np.linspace(0, 1, 1000)
    y = beta.pdf(x, alpha, beta_param)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.grid(True)
    
    # Add mean and mode lines
    mean = alpha / (alpha + beta_param)
    plt.axvline(x=mean, color='r', linestyle='--', label=f'Mean = {mean:.3f}')
    
    if alpha > 1 and beta_param > 1:
        mode = (alpha - 1) / (alpha + beta_param - 2)
        plt.axvline(x=mode, color='g', linestyle=':', label=f'Mode = {mode:.3f}')
    
    if show_cdf:
        y_cdf = beta.cdf(x, alpha, beta_param)
        plt.twinx()
        plt.plot(x, y_cdf, 'r--', linewidth=2, label='CDF')
        plt.ylabel('Cumulative Probability')
    
    plt.legend()
    plt.savefig(os.path.join(images_dir, f'{filename}.png'))
    plt.close()

def compare_beta_distributions():
    x = np.linspace(0, 1, 1000)
    distributions = [
        (1, 1, 'Beta(1,1) - Uniform', 'b-'),
        (2, 5, 'Beta(2,5) - Skewed Left', 'g-'),
        (5, 2, 'Beta(5,2) - Skewed Right', 'r-')
    ]
    
    # Plot individual distributions
    for a, b, label, style in distributions:
        plot_beta_distribution(a, b, label, f'beta_{a}_{b}')
    
    # Plot all together
    plt.figure(figsize=(12, 6))
    for a, b, label, style in distributions:
        y = beta.pdf(x, a, b)
        plt.plot(x, y, style, linewidth=2, label=label)
        
        # Add mean and mode annotations
        mean = a / (a + b)
        plt.axvline(x=mean, color=style[0], linestyle='--', alpha=0.3)
        if a > 1 and b > 1:
            mode = (a - 1) / (a + b - 2)
            plt.axvline(x=mode, color=style[0], linestyle=':', alpha=0.3)
    
    plt.title('Comparison of Different Beta Distributions')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(images_dir, 'beta_comparison.png'))
    plt.close()

def main():
    print("\n=== Beta Distribution Examples ===\n")
    
    # Example 1: Basic Beta Distribution
    print("Example 1: Basic Beta Distribution (α=2, β=5)")
    alpha, beta_param = 2, 5
    mean = alpha / (alpha + beta_param)
    mode = (alpha - 1) / (alpha + beta_param - 2)
    variance = (alpha * beta_param) / ((alpha + beta_param)**2 * (alpha + beta_param + 1))
    p_less_than_03 = beta.cdf(0.3, alpha, beta_param)
    
    print(f"Mean: {mean:.3f}")
    print(f"Mode: {mode:.3f}")
    print(f"Variance: {variance:.3f}")
    print(f"P(X < 0.3): {p_less_than_03:.3f}")
    
    # Plot PDF
    plot_beta_distribution(alpha, beta_param, 
                         f'Beta Distribution (α={alpha}, β={beta_param})',
                         'beta_basic')
    
    # Plot PDF with CDF
    plot_beta_distribution(alpha, beta_param, 
                         f'Beta Distribution with CDF (α={alpha}, β={beta_param})',
                         'beta_basic_with_cdf',
                         show_cdf=True)
    
    # Example 2: Comparing Distributions
    print("\nExample 2: Comparing Different Beta Distributions")
    compare_beta_distributions()
    
    print("\nAll examples completed. Check the Images/Beta_Distribution directory for visualizations.")

if __name__ == "__main__":
    main() 