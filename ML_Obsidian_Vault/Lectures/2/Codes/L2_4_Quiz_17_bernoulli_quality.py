import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, norm, binom
import os

def plot_bernoulli_pmf(p_values, save_path=None):
    """Plot the PMF of Bernoulli distribution for different p values"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.array([0, 1])
    width = 0.2
    
    for i, p in enumerate(p_values):
        pmf = np.array([1-p, p])
        ax.bar(x + (i-len(p_values)/2+0.5)*width, pmf, width=width, alpha=0.7, 
               label=f'p = {p}')
    
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Mass')
    ax.set_title('Bernoulli Distribution PMF for Different p Values')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['0 (Non-defective)', '1 (Defective)'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def compute_mle_and_ci(n, k, alpha=0.05, save_path=None):
    """
    Compute MLE and confidence interval for a Bernoulli distribution
    
    Parameters:
    n (int): Number of trials
    k (int): Number of successes
    alpha (float): Significance level
    save_path (str): Path to save the visualization
    """
    # Calculate MLE
    p_hat = k / n
    
    # Calculate standard error
    se = np.sqrt(p_hat * (1 - p_hat) / n)
    
    # Calculate confidence interval
    z = norm.ppf(1 - alpha/2)
    ci_lower = max(0, p_hat - z * se)
    ci_upper = min(1, p_hat + z * se)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a range of p values for the binomial PMF
    p_range = np.linspace(max(0, p_hat-3*se), min(1, p_hat+3*se), 1000)
    
    # Create distribution of p_hat
    var_p_hat = p_range * (1 - p_range) / n
    std_p_hat = np.sqrt(var_p_hat)
    
    # Plot the normal approximation to the distribution of p_hat
    for p in p_range[::100]:
        if abs(p - p_hat) < 2.5*se:  # Only plot a few curves for clarity
            y = norm.pdf(p_range, p, np.sqrt(p * (1 - p) / n))
            ax.plot(p_range, y, 'k-', alpha=0.1, linewidth=1)
    
    # Plot the MLE and its distribution
    y_mle = norm.pdf(p_range, p_hat, se)
    ax.plot(p_range, y_mle, 'b-', linewidth=2, 
            label=f'Distribution of p̂ ~ N({p_hat:.3f}, {se**2:.6f})')
    
    # Fill the confidence interval
    ci_x = np.linspace(ci_lower, ci_upper, 100)
    ci_y = norm.pdf(ci_x, p_hat, se)
    ax.fill_between(ci_x, ci_y, alpha=0.3, color='green',
                   label=f'{(1-alpha)*100:.0f}% Confidence Interval')
    
    # Mark the MLE and CI bounds
    ax.axvline(x=p_hat, color='red', linestyle='-', 
               label=f'MLE p̂ = {p_hat:.3f}')
    ax.axvline(x=ci_lower, color='green', linestyle='--',
               label=f'CI: ({ci_lower:.3f}, {ci_upper:.3f})')
    ax.axvline(x=ci_upper, color='green', linestyle='--')
    
    # Add text with key statistics
    text_str = '\n'.join((
        f'Sample Size (n): {n}',
        f'Defective Items (k): {k}',
        f'MLE Estimate (p̂): {p_hat:.3f}',
        f'Standard Error: {se:.3f}',
        f'95% CI: ({ci_lower:.3f}, {ci_upper:.3f})'))
    
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Probability of Defective Item (p)')
    ax.set_ylabel('Probability Density')
    ax.set_title('MLE and Confidence Interval for Bernoulli Parameter')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return p_hat, se, (ci_lower, ci_upper)

def plot_binomial_pmf(n, p, save_path=None):
    """Plot the PMF of the binomial distribution"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a range of k values
    k_range = np.arange(0, n+1)
    
    # Calculate PMF
    pmf = binom.pmf(k_range, n, p)
    
    # Plot PMF
    ax.bar(k_range, pmf, width=0.4, alpha=0.7, color='blue')
    
    # Highlight the observed value
    observed_k = np.round(p * n).astype(int)
    ax.bar(observed_k, pmf[observed_k], width=0.4, alpha=0.7, color='red',
           label=f'Expected defective count: {observed_k}')
    
    ax.set_xlabel('Number of Defective Items (k)')
    ax.set_ylabel('Probability Mass')
    ax.set_title(f'Binomial Distribution PMF (n={n}, p={p:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def main():
    """Generate all visualizations for Question 17 of the L2.4 quiz"""
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_4_Quiz_17")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 17 of the L2.4 MLE quiz...")
    
    # Problem parameters
    n = 20  # sample size
    k = 3   # number of defective items
    
    # 1. Plot Bernoulli PMF
    plot_bernoulli_pmf([0.1, 0.15, 0.2, 0.3], 
                      save_path=os.path.join(save_dir, "bernoulli_pmf.png"))
    print("1. Bernoulli PMF visualization created")
    
    # 2. Compute MLE and confidence interval
    p_hat, se, ci = compute_mle_and_ci(n, k, 
                                     save_path=os.path.join(save_dir, "mle_ci.png"))
    print("2. MLE and confidence interval visualization created")
    print(f"   Maximum Likelihood Estimate (p̂): {p_hat:.3f}")
    print(f"   Standard Error: {se:.3f}")
    print(f"   95% Confidence Interval: ({ci[0]:.3f}, {ci[1]:.3f})")
    
    # 3. Plot binomial PMF
    plot_binomial_pmf(n, p_hat, 
                     save_path=os.path.join(save_dir, "binomial_pmf.png"))
    print("3. Binomial PMF visualization created")
    
    # 4. Formulas for future reference
    print("\nFormulas for Reference:")
    print("   MLE: p̂ = k/n = number of defective items / sample size")
    print("   Standard Error: SE(p̂) = sqrt(p̂(1-p̂)/n)")
    print("   95% Confidence Interval: p̂ ± 1.96 × SE(p̂)")
    
    print(f"\nAll visualizations have been saved to: {save_dir}")

if __name__ == "__main__":
    main() 