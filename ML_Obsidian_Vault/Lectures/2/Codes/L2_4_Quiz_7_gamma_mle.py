import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm
from scipy.special import gamma as gamma_func
import os

def gamma_pdf(x, alpha, beta):
    """Probability density function for gamma distribution"""
    return (beta**alpha * x**(alpha-1) * np.exp(-beta*x)) / gamma_func(alpha)

def log_likelihood(beta, data, alpha=2):
    """Log-likelihood function for gamma distribution with known alpha=2"""
    n = len(data)
    return n * alpha * np.log(beta) - beta * np.sum(data) + (alpha - 1) * np.sum(np.log(data))

def plot_gamma_pdfs(beta_values, alpha=2, save_path=None):
    """Plot gamma PDFs for different beta values with fixed alpha=2"""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(0, 10, 1000)
    
    for beta in beta_values:
        y = gamma_pdf(x, alpha, beta)
        ax.plot(x, y, linewidth=2, label=f'β = {beta}')
    
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density f(x|α,β)')
    ax.set_title('Gamma Distribution PDFs (α = 2) for Different β Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_likelihood_surface(data, alpha=2, save_path=None):
    """Plot the likelihood function surface"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate MLE for beta (alpha/mean)
    mle_beta = alpha / np.mean(data)
    
    # Create range of possible beta values
    beta_range = np.linspace(max(0.1, mle_beta*0.5), mle_beta*1.5, 1000)
    log_likelihoods = [log_likelihood(beta, data, alpha) for beta in beta_range]
    
    ax.plot(beta_range, log_likelihoods, 'b-', linewidth=2)
    ax.axvline(x=mle_beta, color='r', linestyle='--', 
               label=f'MLE β = {mle_beta:.4f}')
    
    ax.set_xlabel('β')
    ax.set_ylabel('Log-Likelihood ℓ(β)')
    ax.set_title('Log-Likelihood Function for Gamma Distribution (α = 2)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return mle_beta

def plot_mle_fit(data, alpha=2, save_path=None):
    """Plot the fitted MLE distribution against the data"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate MLE for beta
    mle_beta = alpha / np.mean(data)
    
    # Generate x values for plotting
    x = np.linspace(0, max(data) * 1.5, 1000)
    y_mle = gamma_pdf(x, alpha, mle_beta)
    
    # Plot histogram of the data
    ax.hist(data, bins=min(30, len(data)//5), density=True, alpha=0.5, 
            color='blue', label='Observed Data')
    
    # Plot the fitted PDF based on MLE
    ax.plot(x, y_mle, 'r-', linewidth=2, 
            label=f'MLE Fit (α = {alpha}, β = {mle_beta:.4f})')
    
    # Mark the observed data points
    ax.plot(data, np.zeros_like(data), 'bo', markersize=5, alpha=0.6)
    
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density')
    ax.set_title('Maximum Likelihood Estimation for Gamma Distribution (α = 2)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return mle_beta

def calculate_fisher_information(beta, alpha=2, n=1):
    """Calculate the Fisher information for gamma distribution with known alpha"""
    # For gamma distribution with known α, the Fisher information is:
    # I(β) = α/β²
    return n * alpha / (beta**2)

def plot_fisher_information(beta_range, alpha=2, save_path=None):
    """Plot the Fisher information for different beta values"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate Fisher information for each beta
    fisher_info = [calculate_fisher_information(beta, alpha) for beta in beta_range]
    
    ax.plot(beta_range, fisher_info, 'b-', linewidth=2)
    
    ax.set_xlabel('β')
    ax.set_ylabel('Fisher Information I(β)')
    ax.set_title('Fisher Information for Gamma Distribution (α = 2)')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_confidence_interval_demonstration(true_beta, alpha=2, sample_sizes=[20, 50, 100, 500], n_samples=10000, save_path=None):
    """Demonstrate confidence interval construction for different sample sizes"""
    np.random.seed(42)
    
    fig, axes = plt.subplots(len(sample_sizes), 1, figsize=(12, 4*len(sample_sizes)), sharex=True)
    
    for i, n in enumerate(sample_sizes):
        # Generate samples and calculate MLEs
        mle_estimates = []
        for _ in range(n_samples):
            data = np.random.gamma(alpha, 1/true_beta, n)  # shape, scale=1/rate
            mle_beta = alpha / np.mean(data)
            mle_estimates.append(mle_beta)
        
        # Calculate confidence intervals
        confidence_intervals = []
        for mle in mle_estimates:
            # Fisher information for sample size n
            fisher_info = calculate_fisher_information(mle, alpha, n)
            # Standard error
            se = np.sqrt(1 / fisher_info)
            # 95% confidence interval (normal approximation)
            lower = mle - 1.96 * se
            upper = mle + 1.96 * se
            confidence_intervals.append((lower, upper))
        
        # Calculate coverage
        coverage = sum(1 for l, u in confidence_intervals if l <= true_beta <= u) / n_samples
        
        # Plot the first 100 intervals for visualization
        y_pos = np.arange(100)
        for j, (lower, upper) in enumerate(confidence_intervals[:100]):
            color = 'g' if lower <= true_beta <= upper else 'r'
            axes[i].plot([lower, upper], [j, j], color=color, linewidth=1)
            axes[i].plot(mle_estimates[j], j, 'bo', markersize=3)
        
        axes[i].axvline(x=true_beta, color='k', linestyle='--', label='True β')
        axes[i].set_title(f'Sample Size n = {n}, Coverage = {coverage:.4f}')
        axes[i].set_ylabel('Sample Index')
        axes[i].set_ylim(-1, 100)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('β')
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def main():
    """Generate all visualizations for Question 7 of the L2.4 quiz"""
    # Create synthetic data
    np.random.seed(42)
    alpha = 2
    true_beta = 0.5
    n = 100
    data = np.random.gamma(alpha, 1/true_beta, n)  # shape, scale=1/rate
    
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_4_Quiz_7")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 7 of the L2.4 MLE quiz...")
    
    # 1. Plot PDFs for different beta values
    plot_gamma_pdfs([0.2, 0.5, 1.0, 2.0], alpha=2, 
                   save_path=os.path.join(save_dir, "gamma_pdfs.png"))
    print("1. PDF visualization created")
    
    # 2. Plot likelihood surface
    mle_beta = plot_likelihood_surface(data, alpha=2, 
                                      save_path=os.path.join(save_dir, "likelihood_surface.png"))
    print(f"2. Likelihood surface visualization created, MLE β = {mle_beta:.4f}")
    
    # 3. Plot MLE fit to data
    plot_mle_fit(data, alpha=2, 
                save_path=os.path.join(save_dir, "mle_fit.png"))
    print("3. MLE fit visualization created")
    
    # 4. Plot Fisher information
    beta_range = np.linspace(0.1, 2.0, 1000)
    plot_fisher_information(beta_range, alpha=2, 
                           save_path=os.path.join(save_dir, "fisher_information.png"))
    print("4. Fisher information visualization created")
    
    # 5. Demonstrate confidence interval construction
    plot_confidence_interval_demonstration(true_beta, alpha=2, 
                                          save_path=os.path.join(save_dir, "confidence_intervals.png"))
    print("5. Confidence interval demonstration created")
    
    # Calculate and display the confidence interval for current data
    fisher_info = calculate_fisher_information(mle_beta, alpha, n)
    se = np.sqrt(1 / fisher_info)
    ci_lower = mle_beta - 1.96 * se
    ci_upper = mle_beta + 1.96 * se
    
    print(f"\nSummary of findings:")
    print(f"True parameters: α = {alpha}, β = {true_beta}")
    print(f"MLE for β (sample size {n}): β = {mle_beta:.4f}")
    print(f"Fisher Information I(β): {fisher_info:.4f}")
    print(f"Standard Error: {se:.4f}")
    print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    print(f"\nAll visualizations have been saved to: {save_dir}")

if __name__ == "__main__":
    main() 