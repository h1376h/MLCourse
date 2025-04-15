import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom
import os

def plot_bernoulli_pmf(p_values, save_path=None):
    """Plot the PMF of Bernoulli distribution for different p values"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.array([0, 1])
    
    for p in p_values:
        pmf = np.array([1-p, p])
        ax.bar(x + 0.1*(p_values.index(p)), pmf, width=0.1, alpha=0.7, 
               label=f'p = {p}')
    
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Mass')
    ax.set_title('Bernoulli Distribution PMF for Different p Values')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['0 (Failure)', '1 (Success)'])
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_fisher_information(save_path=None):
    """Plot the Fisher information for Bernoulli distribution"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    p_range = np.linspace(0.01, 0.99, 1000)
    
    # Fisher information for a single Bernoulli trial I(p) = 1/(p(1-p))
    fisher_info = 1 / (p_range * (1 - p_range))
    
    ax.plot(p_range, fisher_info, 'b-', linewidth=2)
    
    # Highlight some specific values
    highlight_points = [0.1, 0.3, 0.5, 0.7, 0.9]
    for p in highlight_points:
        fi = 1 / (p * (1 - p))
        ax.plot(p, fi, 'ro')
        ax.annotate(f'I({p}) = {fi:.2f}', 
                   xy=(p, fi),
                   xytext=(p, fi + 1),
                   ha='center')
    
    # Highlight p=0.3 which is used in the example
    p_example = 0.3
    fi_example = 1 / (p_example * (1 - p_example))
    ax.axvline(x=p_example, linestyle='--', color='red', alpha=0.5)
    ax.axhline(y=fi_example, linestyle='--', color='red', alpha=0.5)
    
    ax.set_xlabel('Parameter p')
    ax.set_ylabel('Fisher Information I(p)')
    ax.set_title('Fisher Information for Bernoulli Distribution')
    ax.grid(True, alpha=0.3)
    
    # Set a reasonable y-axis limit to focus on most of the curve
    ax.set_ylim(0, 50)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_sampling_distribution(n=100, p=0.3, n_samples=10000, save_path=None):
    """Plot the sampling distribution of the MLE for Bernoulli and its theoretical limit"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Simulate n_samples of Bernoulli samples and compute MLE
    np.random.seed(42)
    mle_estimates = []
    
    for _ in range(n_samples):
        sample = np.random.binomial(1, p, n)
        mle_estimates.append(np.mean(sample))
    
    # Compute the histogram of MLE estimates
    hist, bins = np.histogram(mle_estimates, bins=30, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Plot the histogram
    ax.bar(bin_centers, hist, width=bins[1] - bins[0], alpha=0.5, color='blue',
           label='Simulated MLE Distribution')
    
    # Plot the theoretical asymptotic normal distribution
    var_p_hat = p * (1 - p) / n  # Variance of p_hat
    x = np.linspace(p - 4*np.sqrt(var_p_hat), p + 4*np.sqrt(var_p_hat), 1000)
    normal_pdf = norm.pdf(x, p, np.sqrt(var_p_hat))
    
    ax.plot(x, normal_pdf, 'r-', linewidth=2, 
            label='Theoretical Normal Distribution')
    
    # Add vertical line for true parameter
    ax.axvline(x=p, color='k', linestyle='--', label=f'True p = {p}')
    
    # Highlight the standard error
    ax.annotate(f'SE = √(p(1-p)/n) = {np.sqrt(var_p_hat):.4f}', 
               xy=(p, np.max(normal_pdf)),
               xytext=(p, np.max(normal_pdf) * 1.1),
               ha='center',
               bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
    
    # Calculate empirical mean and variance
    empirical_mean = np.mean(mle_estimates)
    empirical_var = np.var(mle_estimates)
    
    # Add text with key statistics
    ax.text(0.05, 0.95, 
            f'Theoretical Mean: {p}\n'
            f'Empirical Mean: {empirical_mean:.4f}\n'
            f'Theoretical Variance: {var_p_hat:.6f}\n'
            f'Empirical Variance: {empirical_var:.6f}\n'
            f'Cramér-Rao Lower Bound: {var_p_hat:.6f}',
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_xlabel('MLE Estimate (p̂)')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Sampling Distribution of MLE for Bernoulli(p={p}) with n={n}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return empirical_mean, empirical_var, var_p_hat

def plot_confidence_interval(p=0.3, n=100, save_path=None):
    """Plot the 95% confidence interval for the parameter p"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate standard error and confidence interval
    se = np.sqrt(p * (1 - p) / n)
    ci_lower = max(0, p - 1.96 * se)
    ci_upper = min(1, p + 1.96 * se)
    
    # Create a range for plotting
    x = np.linspace(max(0, p - 4*se), min(1, p + 4*se), 1000)
    normal_pdf = norm.pdf(x, p, se)
    
    # Plot the PDF
    ax.plot(x, normal_pdf, 'b-', linewidth=2, label='Sampling Distribution of p̂')
    
    # Fill the area within the confidence interval
    x_ci = np.linspace(ci_lower, ci_upper, 1000)
    ax.fill_between(x_ci, norm.pdf(x_ci, p, se), alpha=0.3, color='green',
                   label='95% Confidence Interval')
    
    # Add vertical line for true parameter and bounds
    ax.axvline(x=p, color='r', linestyle='-', label=f'True p = {p}')
    ax.axvline(x=ci_lower, color='g', linestyle='--', label=f'Lower Bound = {ci_lower:.4f}')
    ax.axvline(x=ci_upper, color='g', linestyle='--', label=f'Upper Bound = {ci_upper:.4f}')
    
    # Add annotation for CI
    ax.annotate(f'95% CI: ({ci_lower:.4f}, {ci_upper:.4f})',
               xy=(p, np.max(normal_pdf)/2),
               xytext=(p, np.max(normal_pdf)*0.6),
               ha='center',
               bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
    
    ax.set_xlabel('Parameter p')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'95% Confidence Interval for Bernoulli Parameter (n={n}, p={p})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return ci_lower, ci_upper

def main():
    """Generate all visualizations for Question 15 of the L2.4 quiz"""
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_4_Quiz_15")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 15 of the L2.4 MLE quiz...")
    
    # Parameters for the example
    p = 0.3
    n = 100
    
    # 1. Plot Bernoulli PMF for different p values
    plot_bernoulli_pmf([0.2, 0.3, 0.5, 0.7, 0.8], 
                       save_path=os.path.join(save_dir, "bernoulli_pmf.png"))
    print("1. Bernoulli PMF visualization created")
    
    # 2. Plot Fisher Information function
    plot_fisher_information(save_path=os.path.join(save_dir, "fisher_information.png"))
    print("2. Fisher Information visualization created")
    
    # 3. Calculate Fisher Information for p=0.3
    fisher_info_single = 1 / (p * (1 - p))
    fisher_info_sample = n * fisher_info_single
    cramer_rao_bound = 1 / fisher_info_sample
    
    print(f"3. For p = {p}:")
    print(f"   a. Fisher Information (single observation): I(p) = {fisher_info_single:.4f}")
    print(f"   b. Fisher Information (sample of n={n}): I_n(p) = {fisher_info_sample:.4f}")
    print(f"   c. Cramér-Rao Lower Bound: CRLB = {cramer_rao_bound:.6f}")
    
    # 4. Simulate sampling distribution of MLE
    empirical_mean, empirical_var, theoretical_var = plot_sampling_distribution(
        n=n, p=p, n_samples=10000, 
        save_path=os.path.join(save_dir, "sampling_distribution.png")
    )
    print("4. Sampling distribution visualization created")
    print(f"   a. Empirical mean of MLE: {empirical_mean:.4f} (theoretical: {p})")
    print(f"   b. Empirical variance of MLE: {empirical_var:.6f}")
    print(f"   c. Theoretical variance (CRLB): {theoretical_var:.6f}")
    
    # 5. Calculate and plot 95% confidence interval
    ci_lower, ci_upper = plot_confidence_interval(
        p=p, n=n, 
        save_path=os.path.join(save_dir, "confidence_interval.png")
    )
    print("5. Confidence interval visualization created")
    print(f"   a. 95% Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})")
    
    print(f"\nAll visualizations have been saved to: {save_dir}")

if __name__ == "__main__":
    main() 