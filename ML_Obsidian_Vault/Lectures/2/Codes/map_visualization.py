import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, beta
import os

def save_visualization(fig, filename, save_dir):
    """Save the visualization to the specified directory."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {filepath}")
    plt.close(fig)

def normal_map_estimate(mu0, sigma0_sq, new_data, sigma_sq):
    """Calculate the MAP estimate for a normal distribution with known variance."""
    N = len(new_data)
    ratio = sigma0_sq / sigma_sq
    
    numerator = mu0 + ratio * sum(new_data)
    denominator = 1 + ratio * N
    
    return numerator / denominator

def visualize_variance_ratio():
    """Generate visualization for example 6: variance ratio effects."""
    # Setup 
    mu0 = 60  # prior mean
    data = [70, 75, 68, 73]  # observed data
    sample_mean = np.mean(data)
    
    # Three scenarios
    scenarios = [
        {"name": "Original", "prior_var": 16, "data_var": 4},
        {"name": "Higher Prior Uncertainty", "prior_var": 64, "data_var": 4},
        {"name": "Lower Measurement Error", "prior_var": 16, "data_var": 1}
    ]
    
    # Calculate MAP estimates
    for s in scenarios:
        s["ratio"] = s["prior_var"] / s["data_var"]
        s["map"] = normal_map_estimate(mu0, s["prior_var"], data, s["data_var"])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot prior and data
    x = np.linspace(50, 80, 1000)
    ax.plot([mu0, mu0], [0, 0.3], 'b--', label=f'Prior Mean ({mu0})')
    ax.plot([sample_mean, sample_mean], [0, 0.3], 'g--', label=f'Sample Mean ({sample_mean:.1f})')
    
    # Plot MAP estimates
    for i, s in enumerate(scenarios):
        ax.axvline(x=s["map"], color=f'C{i}', linestyle='-', linewidth=2, 
                 label=f'{s["name"]}: r={s["ratio"]}, MAP={s["map"]:.2f}')
    
    # Add text explanation
    ax.text(0.05, 0.95, 
           "Higher r values give more weight to the data\nand move the MAP estimate closer to the sample mean.", 
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Format plot
    ax.set_title("Effect of Variance Ratio on MAP Estimates")
    ax.set_xlabel("Efficacy Score")
    ax.set_ylabel("(Visual guide only - not to scale)")
    ax.set_ylim(0, 0.3)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    return fig

def visualize_mle_convergence():
    """Generate visualization for example 7: convergence to MLE with increasing N."""
    # Setup
    mu0 = 80  # prior mean
    sigma0_sq = 25  # prior variance
    data_var = 100  # data variance
    sample_mean = 97.4  # observed mean in all samples
    
    # Sample sizes to visualize
    sample_sizes = [3, 10, 30, 100, 1000]
    
    # Calculate MAP estimates for each sample size
    map_estimates = []
    ratio = sigma0_sq / data_var
    
    for N in sample_sizes:
        numerator = mu0 + ratio * N * sample_mean
        denominator = 1 + ratio * N
        map_est = numerator / denominator
        map_estimates.append(map_est)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot MLE (sample mean) as horizontal line
    ax.axhline(y=sample_mean, color='g', linestyle='--', label=f'MLE (Sample Mean = {sample_mean})')
    
    # Plot prior mean as horizontal line
    ax.axhline(y=mu0, color='b', linestyle='--', label=f'Prior Mean = {mu0}')
    
    # Plot MAP estimates vs sample sizes
    ax.plot(sample_sizes, map_estimates, 'ro-', label='MAP Estimates')
    
    # Add text annotations for specific points
    for i, (N, map_est) in enumerate(zip(sample_sizes, map_estimates)):
        ax.annotate(f'{map_est:.2f}', 
                  xy=(N, map_est), 
                  xytext=(10, (-1)**i * 10),
                  textcoords='offset points',
                  fontsize=9,
                  bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    
    # Set log scale for x-axis to better show the range of sample sizes
    ax.set_xscale('log')
    ax.set_xticks(sample_sizes)
    ax.set_xticklabels([str(n) for n in sample_sizes])
    
    # Format plot
    ax.set_title("Convergence of MAP Estimate to MLE as Sample Size Increases")
    ax.set_xlabel("Sample Size (N)")
    ax.set_ylabel("Estimate Value")
    ax.legend(loc='center right')
    ax.grid(True, alpha=0.3)
    
    return fig

def visualize_conflicting_data():
    """Generate visualization for example 8: prior and data in conflict."""
    # Setup 
    mu0 = 75  # prior mean
    data = [50, 55, 45, 52, 48]  # observed data
    sample_mean = np.mean(data)
    data_var = 16  # data variance
    
    # Three prior strength scenarios
    scenarios = [
        {"name": "Strong Prior", "prior_var": 4},
        {"name": "Standard Prior", "prior_var": 9},
        {"name": "Weak Prior", "prior_var": 36}
    ]
    
    # Calculate MAP estimates
    for s in scenarios:
        s["ratio"] = s["prior_var"] / data_var
        s["map"] = normal_map_estimate(mu0, s["prior_var"], data, data_var)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot distributions
    x = np.linspace(30, 90, 1000)
    
    # Plot the prior distribution
    prior_std = np.sqrt(scenarios[1]["prior_var"])
    prior_pdf = norm.pdf(x, mu0, prior_std)
    ax.plot(x, prior_pdf, 'b-', label=f'Prior (μ₀={mu0}, σ₀²={scenarios[1]["prior_var"]})')
    
    # Plot the likelihood (data)
    likelihood_std = np.sqrt(data_var / len(data))
    likelihood_pdf = norm.pdf(x, sample_mean, likelihood_std)
    ax.plot(x, likelihood_pdf, 'g-', label=f'Likelihood (mean={sample_mean}, σ²={data_var})')
    
    # Add vertical lines for the MAP estimates
    colors = ['r', 'purple', 'orange']
    for i, s in enumerate(scenarios):
        ax.axvline(x=s["map"], color=colors[i], linestyle='--', 
                 label=f'{s["name"]}: r={s["ratio"]:.2f}, MAP={s["map"]:.1f}')
    
    # Add vertical lines for the prior and sample mean
    ax.axvline(x=mu0, color='b', linestyle=':')
    ax.axvline(x=sample_mean, color='g', linestyle=':')
    
    # Format plot
    ax.set_title("MAP Estimates with Conflicting Prior and Data")
    ax.set_xlabel("Test Score")
    ax.set_ylabel("Probability Density")
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    
    return fig

def visualize_medical_diagnosis():
    """Generate visualization for example 9: medical diagnosis with binomial data."""
    # Setup 
    prior_mean = 0.85  # prior mean (true positive rate)
    prior_var = 0.0036  # prior variance
    successes = 22  # observed successes
    trials = 30  # total trials
    observed_rate = successes / trials  # observed true positive rate
    
    # For Beta distribution with given mean and variance
    # α/(α+β) = mean
    # αβ/((α+β)²(α+β+1)) = variance
    alpha_0 = prior_mean * (prior_mean * (1 - prior_mean) / prior_var - 1)
    beta_0 = (1 - prior_mean) * (prior_mean * (1 - prior_mean) / prior_var - 1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the Beta distributions
    x = np.linspace(0.5, 1.0, 1000)
    
    # Prior (Beta distribution)
    prior_pdf = beta.pdf(x, alpha_0, beta_0)
    ax.plot(x, prior_pdf, 'b-', label=f'Prior: Beta({alpha_0:.1f}, {beta_0:.1f})')
    
    # Likelihood (Binomial likelihood)
    likelihood = np.array([stats_binom_pmf(successes, trials, p) for p in x])
    likelihood = likelihood / np.max(likelihood) * np.max(prior_pdf)  # Scale for visualization
    ax.plot(x, likelihood, 'g-', label=f'Scaled Likelihood: {successes}/{trials} = {observed_rate:.2f}')
    
    # Posterior (Beta distribution)
    posterior_pdf = beta.pdf(x, alpha_0 + successes, beta_0 + trials - successes)
    ax.plot(x, posterior_pdf, 'r-', label=f'Posterior: Beta({alpha_0 + successes:.1f}, {beta_0 + trials - successes:.1f})')
    
    # MAP estimate from Beta posterior
    map_beta = (alpha_0 + successes - 1) / (alpha_0 + beta_0 + trials - 2)
    ax.axvline(x=map_beta, color='r', linestyle='--', 
             label=f'MAP (Beta): {map_beta:.3f}')
    
    # Add vertical lines for the prior and observed rates
    ax.axvline(x=prior_mean, color='b', linestyle=':', label=f'Prior Mean: {prior_mean}')
    ax.axvline(x=observed_rate, color='g', linestyle=':', label=f'Observed Rate: {observed_rate}')
    
    # Format plot
    ax.set_title("MAP Estimation for Diagnostic Test True Positive Rate")
    ax.set_xlabel("True Positive Rate (θ)")
    ax.set_ylabel("Probability Density")
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    
    return fig

def stats_binom_pmf(k, n, p):
    """Calculate binomial PMF (without scipy dependency)."""
    from math import comb
    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

if __name__ == "__main__":
    # Define save directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(os.path.dirname(script_dir), "Images")
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate and save all visualizations
    fig1 = visualize_variance_ratio()
    save_visualization(fig1, "variance_ratio_map.png", save_dir)
    
    fig2 = visualize_mle_convergence()
    save_visualization(fig2, "convergence_map.png", save_dir)
    
    fig3 = visualize_conflicting_data()
    save_visualization(fig3, "conflict_map.png", save_dir)
    
    fig4 = visualize_medical_diagnosis()
    save_visualization(fig4, "medical_map.png", save_dir)
    
    print("All visualizations created successfully!") 