import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, norm
import os

def plot_exponential_pdf(lambda_values, save_path=None):
    """Plot the PDF of exponential distribution for different lambda values"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.linspace(0, 10, 1000)
    
    for lambda_val in lambda_values:
        pdf = lambda_val * np.exp(-lambda_val * x)
        ax.plot(x, pdf, label=f'λ = {lambda_val}')
    
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density')
    ax.set_title('Exponential Distribution PDF for Different λ Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def demonstrate_consistency(true_lambda=0.5, n_values=[10, 50, 100, 500, 1000], n_simulations=1000, save_path=None):
    """Demonstrate the consistency of MLE for exponential distribution"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    np.random.seed(42)
    
    mle_means = []
    mle_variances = []
    
    # For each sample size
    for n in n_values:
        mle_estimates = []
        
        # Run multiple simulations
        for _ in range(n_simulations):
            # Generate a sample from exponential distribution
            sample = np.random.exponential(scale=1/true_lambda, size=n)
            
            # Calculate MLE estimate (reciprocal of sample mean)
            mle_lambda = 1 / np.mean(sample)
            mle_estimates.append(mle_lambda)
        
        # Calculate mean and variance of MLE estimates
        mle_mean = np.mean(mle_estimates)
        mle_var = np.var(mle_estimates)
        
        mle_means.append(mle_mean)
        mle_variances.append(mle_var)
        
        # Plot histogram of MLE estimates for each sample size
        ax.hist(mle_estimates, bins=30, density=True, alpha=0.5, 
                label=f'n = {n}', histtype='step', linewidth=2)
    
    # Add vertical line for true lambda
    ax.axvline(x=true_lambda, color='r', linestyle='--', label=f'True λ = {true_lambda}')
    
    ax.set_xlabel('MLE Estimate of λ')
    ax.set_ylabel('Density')
    ax.set_title('Consistency of MLE for Exponential Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    # Create a second plot for mean and variance vs sample size
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(n_values, mle_means, 'bo-', linewidth=2)
    ax1.axhline(y=true_lambda, color='r', linestyle='--', label=f'True λ = {true_lambda}')
    ax1.set_ylabel('Mean of MLE Estimates')
    ax1.set_title('Consistency of MLE for Exponential Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(n_values, mle_variances, 'go-', linewidth=2)
    # Theoretical variance: true_lambda^2/n
    theoretical_var = [true_lambda**2/n for n in n_values]
    ax2.plot(n_values, theoretical_var, 'r--', label='Theoretical Variance')
    ax2.set_xlabel('Sample Size (n)')
    ax2.set_ylabel('Variance of MLE Estimates')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    if save_path:
        save_path_stats = save_path.replace('.png', '_stats.png')
        plt.savefig(save_path_stats, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path_stats}")
    
    plt.close()
    
    return mle_means, mle_variances

def demonstrate_asymptotic_normality(true_lambda=0.5, n=50, n_simulations=10000, save_path=None):
    """Demonstrate the asymptotic normality of MLE for exponential distribution"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    np.random.seed(42)
    
    # Generate MLE estimates
    mle_estimates = []
    for _ in range(n_simulations):
        sample = np.random.exponential(scale=1/true_lambda, size=n)
        mle_lambda = 1 / np.mean(sample)
        mle_estimates.append(mle_lambda)
    
    # Calculate mean and standard deviation of MLE estimates
    mean_mle = np.mean(mle_estimates)
    std_mle = np.std(mle_estimates)
    
    # Compute standardized MLE estimates: (λ_hat - λ) / (λ/√n)
    standardized_mle = (np.array(mle_estimates) - true_lambda) / (true_lambda / np.sqrt(n))
    
    # Plot histogram of standardized MLE estimates
    hist, bins = np.histogram(standardized_mle, bins=30, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    ax.bar(bin_centers, hist, width=bins[1] - bins[0], alpha=0.5, color='blue',
           label='Standardized MLE Distribution')
    
    # Plot standard normal distribution
    x = np.linspace(-4, 4, 1000)
    ax.plot(x, norm.pdf(x), 'r-', linewidth=2, label='Standard Normal Distribution')
    
    ax.set_xlabel('Standardized MLE Estimate: (λ_hat - λ) / (λ/√n)')
    ax.set_ylabel('Density')
    ax.set_title(f'Asymptotic Normality of MLE for Exponential Distribution (n={n})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add QQ plot as an inset
    axins = ax.inset_axes([0.6, 0.1, 0.35, 0.35])
    from scipy.stats import probplot
    probplot(standardized_mle, dist="norm", plot=axins)
    axins.set_title('Normal Q-Q Plot')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return mean_mle, std_mle

def demonstrate_asymptotic_efficiency(true_lambda=0.5, n_values=[10, 50, 100, 500, 1000], n_simulations=1000, save_path=None):
    """Demonstrate the asymptotic efficiency of MLE for exponential distribution"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    np.random.seed(42)
    
    mse_values = []
    theoretical_values = []
    
    # For each sample size
    for n in n_values:
        mle_estimates = []
        moment_estimates = []  # Method of moments estimator for comparison
        
        # Run multiple simulations
        for _ in range(n_simulations):
            # Generate a sample from exponential distribution
            sample = np.random.exponential(scale=1/true_lambda, size=n)
            
            # Calculate MLE estimate (reciprocal of sample mean)
            mle_lambda = 1 / np.mean(sample)
            mle_estimates.append(mle_lambda)
            
            # Calculate a biased estimator for comparison
            # We'll use a slightly biased version of the MLE
            biased_lambda = 1 / (np.mean(sample) * 1.05)  # Introduce small bias
            moment_estimates.append(biased_lambda)
        
        # Calculate MSE of MLE estimates
        mse_mle = np.mean((np.array(mle_estimates) - true_lambda)**2)
        mse_moment = np.mean((np.array(moment_estimates) - true_lambda)**2)
        
        mse_values.append((mse_mle, mse_moment))
        
        # Calculate theoretical MSE (equals variance for unbiased estimator)
        theoretical_mse = true_lambda**2 / n
        theoretical_values.append(theoretical_mse)
    
    # Plot MSE vs sample size
    ax.plot(n_values, [mse[0] for mse in mse_values], 'bo-', linewidth=2, 
            label='MLE MSE')
    ax.plot(n_values, [mse[1] for mse in mse_values], 'go-', linewidth=2, 
            label='Biased Estimator MSE')
    ax.plot(n_values, theoretical_values, 'r--', linewidth=2, 
            label='Theoretical Minimum MSE (CRLB)')
    
    ax.set_xlabel('Sample Size (n)')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('Asymptotic Efficiency of MLE for Exponential Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set logarithmic scales for better visualization
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return mse_values, theoretical_values

def calculate_within_percentage(true_lambda=0.5, percentage=0.1, n=50, n_simulations=100000, save_path=None):
    """Calculate probability that MLE falls within given percentage of true value"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    np.random.seed(42)
    
    # Generate MLE estimates
    mle_estimates = []
    for _ in range(n_simulations):
        sample = np.random.exponential(scale=1/true_lambda, size=n)
        mle_lambda = 1 / np.mean(sample)
        mle_estimates.append(mle_lambda)
    
    # Convert list to numpy array
    mle_estimates = np.array(mle_estimates)
    
    # Calculate lower and upper bounds
    lower_bound = true_lambda * (1 - percentage)
    upper_bound = true_lambda * (1 + percentage)
    
    # Count estimates within bounds
    within_bounds = np.sum((mle_estimates >= lower_bound) & (mle_estimates <= upper_bound))
    probability = within_bounds / n_simulations
    
    # Plot histogram of MLE estimates
    hist, bins = np.histogram(mle_estimates, bins=50, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    ax.bar(bin_centers, hist, width=bins[1] - bins[0], alpha=0.5, color='blue')
    
    # Highlight the region within bounds
    within_region = [(b >= lower_bound) and (b <= upper_bound) for b in bin_centers]
    ax.bar(bin_centers[within_region], hist[within_region], width=bins[1] - bins[0], 
           color='green', alpha=0.5, label=f'Within {percentage*100}% of true λ')
    
    # Add vertical lines for bounds
    ax.axvline(x=true_lambda, color='r', linestyle='-', linewidth=2, label=f'True λ = {true_lambda}')
    ax.axvline(x=lower_bound, color='k', linestyle='--', linewidth=1, 
               label=f'Bounds: {lower_bound:.3f}, {upper_bound:.3f}')
    ax.axvline(x=upper_bound, color='k', linestyle='--', linewidth=1)
    
    # Add text with probability
    ax.text(0.05, 0.95, f'P(|λ_hat - λ| < {percentage*100}%λ) = {probability:.4f}',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_xlabel('MLE Estimate of λ')
    ax.set_ylabel('Density')
    ax.set_title(f'Probability that MLE falls within {percentage*100}% of true λ (n={n})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    # Theoretical calculation using asymptotic normality
    # For large n, λ_hat ~ N(λ, λ²/n)
    # So (λ_hat - λ)/(λ/√n) ~ N(0, 1)
    # P(|λ_hat - λ| < p*λ) = P(-p*λ < λ_hat - λ < p*λ)
    # = P(-p*√n < (λ_hat - λ)/(λ/√n) < p*√n)
    # = Φ(p*√n) - Φ(-p*√n) = 2*Φ(p*√n) - 1
    
    theoretical_prob = 2*norm.cdf(percentage*np.sqrt(n)) - 1
    
    return probability, theoretical_prob

def main():
    """Generate all visualizations for Question 16 of the L2.4 quiz"""
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_4_Quiz_16")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 16 of the L2.4 MLE quiz...")
    
    # Parameters for the example
    true_lambda = 0.5
    n = 50
    
    # 1. Plot exponential PDF for different lambda values
    plot_exponential_pdf([0.2, 0.5, 1.0, 2.0], 
                        save_path=os.path.join(save_dir, "exponential_pdf.png"))
    print("1. Exponential PDF visualization created")
    
    # 2. Demonstrate consistency
    mle_means, mle_variances = demonstrate_consistency(
        true_lambda=true_lambda, 
        save_path=os.path.join(save_dir, "consistency.png")
    )
    print("2. Consistency demonstration created")
    print("   Mean of MLE estimates for increasing sample sizes:")
    for i, n_val in enumerate([10, 50, 100, 500, 1000]):
        print(f"   n = {n_val}: {mle_means[i]:.4f} (true λ = {true_lambda})")
    print("   Variance of MLE estimates for increasing sample sizes:")
    for i, n_val in enumerate([10, 50, 100, 500, 1000]):
        theo_var = true_lambda**2/n_val
        print(f"   n = {n_val}: {mle_variances[i]:.6f} (theoretical: {theo_var:.6f})")
    
    # 3. Demonstrate asymptotic normality
    mean_mle, std_mle = demonstrate_asymptotic_normality(
        true_lambda=true_lambda, n=n, 
        save_path=os.path.join(save_dir, "asymptotic_normality.png")
    )
    theo_std = true_lambda / np.sqrt(n)
    print("3. Asymptotic normality demonstration created")
    print(f"   Mean of MLE estimates: {mean_mle:.4f} (true λ = {true_lambda})")
    print(f"   Std dev of MLE estimates: {std_mle:.4f} (theoretical: {theo_std:.4f})")
    
    # 4. Demonstrate asymptotic efficiency
    mse_values, theoretical_values = demonstrate_asymptotic_efficiency(
        true_lambda=true_lambda, 
        save_path=os.path.join(save_dir, "asymptotic_efficiency.png")
    )
    print("4. Asymptotic efficiency demonstration created")
    print("   MSE of MLE and theoretical minimum (CRLB) for increasing sample sizes:")
    for i, n_val in enumerate([10, 50, 100, 500, 1000]):
        print(f"   n = {n_val}: MSE = {mse_values[i][0]:.6f}, CRLB = {theoretical_values[i]:.6f}")
    
    # 5. Calculate probability of MLE falling within 10% of true value
    prob, theo_prob = calculate_within_percentage(
        true_lambda=true_lambda, percentage=0.1, n=n, 
        save_path=os.path.join(save_dir, "within_percentage.png")
    )
    print("5. Within percentage probability calculation created")
    print(f"   P(|λ_hat - λ| < 10%λ) for n={n}:")
    print(f"   Empirical: {prob:.4f}")
    print(f"   Theoretical (based on asymptotic normality): {theo_prob:.4f}")
    
    print(f"\nAll visualizations have been saved to: {save_dir}")

if __name__ == "__main__":
    main() 