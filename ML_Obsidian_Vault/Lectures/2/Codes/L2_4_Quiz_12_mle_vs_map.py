import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import os

def bernoulli_likelihood(p, k, n):
    """
    Likelihood function for binomial distribution
    p: probability of success
    k: number of successes
    n: number of trials
    """
    return p**k * (1-p)**(n-k)

def log_likelihood(p, k, n):
    """Log-likelihood function for binomial distribution"""
    return k * np.log(p) + (n - k) * np.log(1 - p)

def beta_prior(p, alpha, beta_param):
    """Beta prior distribution"""
    return beta.pdf(p, alpha, beta_param)

def log_beta_prior(p, alpha, beta_param):
    """Log of beta prior distribution"""
    return (alpha - 1) * np.log(p) + (beta_param - 1) * np.log(1 - p) - np.log(beta.pdf(0.5, alpha, beta_param) / beta.pdf(0.5, 1, 1))

def posterior(p, k, n, alpha, beta_param):
    """Posterior distribution (proportional)"""
    return bernoulli_likelihood(p, k, n) * beta_prior(p, alpha, beta_param)

def log_posterior(p, k, n, alpha, beta_param):
    """Log-posterior distribution (proportional)"""
    return log_likelihood(p, k, n) + log_beta_prior(p, alpha, beta_param)

def plot_likelihood_and_prior(k, n, alpha, beta_param, save_path=None):
    """Plot likelihood function and prior distribution"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create p values for plotting
    p_values = np.linspace(0.001, 0.999, 1000)
    
    # Calculate likelihood values
    likelihood_values = [bernoulli_likelihood(p, k, n) for p in p_values]
    likelihood_values = likelihood_values / np.max(likelihood_values)  # Normalize to 1
    
    # Calculate prior values
    prior_values = [beta_prior(p, alpha, beta_param) for p in p_values]
    prior_values = prior_values / np.max(prior_values)  # Normalize to 1
    
    # Calculate posterior values
    posterior_values = [posterior(p, k, n, alpha, beta_param) for p in p_values]
    posterior_values = posterior_values / np.max(posterior_values)  # Normalize to 1
    
    # MLE for p
    mle_p = k / n
    
    # MAP for p
    map_p = (k + alpha - 1) / (n + alpha + beta_param - 2)
    
    # Plot the distributions
    ax.plot(p_values, likelihood_values, 'b-', linewidth=2, 
           label=f'Likelihood (MLE = {mle_p:.4f})')
    ax.plot(p_values, prior_values, 'g-', linewidth=2, 
           label=f'Prior: Beta({alpha}, {beta_param})')
    ax.plot(p_values, posterior_values, 'r-', linewidth=2, 
           label=f'Posterior (MAP = {map_p:.4f})')
    
    # Mark the MLE and MAP
    ax.axvline(x=mle_p, color='blue', linestyle='--')
    ax.axvline(x=map_p, color='red', linestyle='--')
    
    ax.set_xlabel('Probability (p)')
    ax.set_ylabel('Normalized Density')
    ax.set_title(f'Comparison of Likelihood, Prior, and Posterior ({k} successes out of {n} trials)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_mle_vs_map_comparison(save_path=None):
    """Compare MLE and MAP estimates for different sample sizes"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # True p value
    true_p = 0.2
    
    # Prior parameters
    alpha = 2
    beta_param = 8
    
    # Sample sizes
    sample_sizes = np.logspace(0, 3, 100).astype(int)
    sample_sizes = np.unique(sample_sizes)
    
    # Initialize arrays to store MLE and MAP estimates
    mle_estimates = []
    map_estimates = []
    
    # For each sample size, calculate MLE and MAP
    for n in sample_sizes:
        # Generate synthetic data with true p
        np.random.seed(42)  # For reproducibility
        data = np.random.binomial(1, true_p, n)
        k = np.sum(data)
        
        # Calculate MLE and MAP
        mle = k / n
        map_est = (k + alpha - 1) / (n + alpha + beta_param - 2)
        
        mle_estimates.append(mle)
        map_estimates.append(map_est)
    
    # Plot the estimates
    ax.plot(sample_sizes, mle_estimates, 'b-', linewidth=2, label='MLE')
    ax.plot(sample_sizes, map_estimates, 'r-', linewidth=2, label='MAP')
    ax.axhline(y=true_p, color='g', linestyle='--', label=f'True p = {true_p}')
    
    ax.set_xscale('log')
    ax.set_xlabel('Sample Size (log scale)')
    ax.set_ylabel('Probability Estimate')
    ax.set_title('Comparison of MLE and MAP Estimates vs. Sample Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_credible_interval(k, n, alpha, beta_param, save_path=None):
    """Plot posterior distribution with credible interval"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create p values for plotting
    p_values = np.linspace(0.001, 0.999, 1000)
    
    # Calculate posterior distribution
    posterior_alpha = k + alpha
    posterior_beta = n - k + beta_param
    
    # Plot the posterior distribution
    posterior_pdf = [beta.pdf(p, posterior_alpha, posterior_beta) for p in p_values]
    ax.plot(p_values, posterior_pdf, 'r-', linewidth=2, label='Posterior Distribution')
    
    # Find MAP
    map_p = (posterior_alpha - 1) / (posterior_alpha + posterior_beta - 2)
    ax.axvline(x=map_p, color='r', linestyle='--', label=f'MAP = {map_p:.4f}')
    
    # Find 95% credible interval
    lower = beta.ppf(0.025, posterior_alpha, posterior_beta)
    upper = beta.ppf(0.975, posterior_alpha, posterior_beta)
    
    # Shade the credible interval
    interval_x = np.linspace(lower, upper, 1000)
    interval_y = [beta.pdf(p, posterior_alpha, posterior_beta) for p in interval_x]
    ax.fill_between(interval_x, interval_y, alpha=0.3, color='red')
    
    # Add vertical lines for the interval
    ax.axvline(x=lower, color='r', linestyle=':', alpha=0.7)
    ax.axvline(x=upper, color='r', linestyle=':', alpha=0.7)
    
    # Add text for the credible interval
    ax.text(0.5, 0.9, f'95% Credible Interval: [{lower:.4f}, {upper:.4f}]',
            transform=ax.transAxes, ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.7))
    
    ax.set_xlabel('Probability (p)')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Posterior Distribution: Beta({posterior_alpha}, {posterior_beta})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_prior_influence(save_path=None):
    """Demonstrate how different priors influence the MAP estimate"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Data
    k = 4  # successes
    n = 20  # trials
    
    # MLE
    mle_p = k / n
    
    # Range of prior strengths (alpha + beta)
    prior_strengths = np.logspace(0, 2, 100)
    
    # Different prior means
    prior_means = [0.1, 0.2, 0.3, 0.5]
    colors = ['b', 'g', 'r', 'c']
    
    # Plot MLE
    ax.axhline(y=mle_p, color='k', linestyle='--', label=f'MLE = {mle_p:.4f}')
    
    # For each prior mean, calculate MAP for different prior strengths
    for i, prior_mean in enumerate(prior_means):
        map_estimates = []
        
        for strength in prior_strengths:
            # Calculate alpha and beta for the given mean and strength
            alpha = prior_mean * strength
            beta_param = (1 - prior_mean) * strength
            
            # Calculate MAP
            map_p = (k + alpha - 1) / (n + alpha + beta_param - 2)
            map_estimates.append(map_p)
        
        ax.plot(prior_strengths, map_estimates, f'{colors[i]}-', linewidth=2, 
               label=f'Prior Mean = {prior_mean}')
    
    ax.set_xscale('log')
    ax.set_xlabel('Prior Strength (α + β) - log scale')
    ax.set_ylabel('MAP Estimate')
    ax.set_title('Influence of Prior Strength and Mean on MAP Estimate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def main():
    """Generate all visualizations for Question 12 of the L2.4 quiz"""
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_4_Quiz_12")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 12 of the L2.4 MLE quiz...")
    
    # Problem data
    k1 = 4  # 4 conversions
    n1 = 20  # out of 20 visitors
    alpha = 2  # Beta prior parameters
    beta_param = 8
    
    # Calculate MLE and MAP for original data
    mle_p1 = k1 / n1
    map_p1 = (k1 + alpha - 1) / (n1 + alpha + beta_param - 2)
    
    print(f"Original data: {k1} conversions out of {n1} visitors")
    print(f"MLE for p: {mle_p1:.4f}")
    print(f"MAP for p (with Beta({alpha}, {beta_param}) prior): {map_p1:.4f}")
    
    # Plot likelihood, prior, and posterior for original data
    plot_likelihood_and_prior(k1, n1, alpha, beta_param, 
                             save_path=os.path.join(save_dir, "likelihood_prior_posterior_20.png"))
    print("1. Likelihood, prior, and posterior visualization created for n=20")
    
    # Increased sample size
    k2 = 40  # 40 conversions
    n2 = 200  # out of 200 visitors
    
    # Calculate MLE and MAP for increased data
    mle_p2 = k2 / n2
    map_p2 = (k2 + alpha - 1) / (n2 + alpha + beta_param - 2)
    
    print(f"\nLarger data: {k2} conversions out of {n2} visitors")
    print(f"MLE for p: {mle_p2:.4f}")
    print(f"MAP for p (with Beta({alpha}, {beta_param}) prior): {map_p2:.4f}")
    
    # Plot likelihood, prior, and posterior for larger data
    plot_likelihood_and_prior(k2, n2, alpha, beta_param, 
                             save_path=os.path.join(save_dir, "likelihood_prior_posterior_200.png"))
    print("2. Likelihood, prior, and posterior visualization created for n=200")
    
    # Plot MLE vs MAP comparison for different sample sizes
    plot_mle_vs_map_comparison(save_path=os.path.join(save_dir, "mle_vs_map_sample_size.png"))
    print("3. MLE vs MAP comparison visualization created")
    
    # Plot credible interval for original data
    plot_credible_interval(k1, n1, alpha, beta_param, 
                          save_path=os.path.join(save_dir, "credible_interval.png"))
    print("4. Credible interval visualization created")
    
    # Plot influence of prior on MAP estimate
    plot_prior_influence(save_path=os.path.join(save_dir, "prior_influence.png"))
    print("5. Prior influence visualization created")
    
    # Print summary of results for Question 12
    print("\nQuestion 12 Results:")
    print("===================")
    
    print(f"Part 1: MLE for conversion rate = {mle_p1:.4f}")
    
    print(f"\nPart 2: MAP estimate with Beta({alpha}, {beta_param}) prior = {map_p1:.4f}")
    prior_mean = alpha / (alpha + beta_param)
    print(f"Prior mean = {prior_mean:.4f}")
    
    print(f"\nPart 3: Comparison of MLE and MAP")
    print(f"MLE (n=20): {mle_p1:.4f}")
    print(f"MAP (n=20): {map_p1:.4f}")
    print(f"Difference: {abs(mle_p1 - map_p1):.4f}")
    print("MAP is pulled toward the prior mean compared to MLE")
    
    print(f"\nPart 4: Larger sample (n=200)")
    print(f"MLE (n=200): {mle_p2:.4f}")
    print(f"MAP (n=200): {map_p2:.4f}")
    print(f"Difference: {abs(mle_p2 - map_p2):.4f}")
    
    print(f"\nPart 5: As sample size increases, MAP approaches MLE")
    print("MLE is consistent (converges to true value)")
    print("MAP is also consistent if prior has non-zero density at true value")
    print("When sample size is large, the likelihood dominates the prior")
    
    print(f"\nAll visualizations have been saved to: {save_dir}")

if __name__ == "__main__":
    main() 