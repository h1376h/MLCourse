import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import matplotlib as mpl

# Set global matplotlib style for prettier plots
plt.style.use('seaborn-v0_8-pastel')
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.titlesize'] = 16
mpl.rcParams['figure.figsize'] = (10, 6)

def chebyshev_bound(k):
    """
    Calculate the Chebyshev upper bound on the probability that a random variable
    deviates from its mean by more than k standard deviations
    
    Args:
        k: Number of standard deviations
    
    Returns:
        Upper bound on the probability
    """
    return 1 / (k**2)

def markov_bound(mean, threshold):
    """
    Calculate the Markov upper bound on the probability that a non-negative
    random variable exceeds a threshold
    
    Args:
        mean: Mean of the random variable
        threshold: Threshold value
    
    Returns:
        Upper bound on the probability
    """
    if threshold <= 0:
        return 1.0
    return mean / threshold

def normal_probability_beyond_k_std(k):
    """
    Calculate the exact probability that a normal random variable
    deviates from its mean by more than k standard deviations
    
    Args:
        k: Number of standard deviations
    
    Returns:
        Probability
    """
    # P(|X - μ| > k*σ) = P(Z < -k) + P(Z > k) = 2 * P(Z > k)
    return 2 * (1 - stats.norm.cdf(k))

def visualize_chebyshev_comparison(mean=50, std_dev=10, save_path=None):
    """
    Create a simplified visualization comparing Chebyshev's bound with the actual 
    probability for a normal distribution
    """
    plt.figure(figsize=(9, 5))
    
    # Range of k values (number of standard deviations)
    k_values = np.linspace(0.5, 5, 100)
    
    # Calculate Chebyshev bounds
    chebyshev_bounds = [chebyshev_bound(k) for k in k_values]
    
    # Calculate exact probabilities for the normal distribution
    normal_probs = [normal_probability_beyond_k_std(k) for k in k_values]
    
    # Plot the bounds and exact probabilities
    plt.plot(k_values, chebyshev_bounds, color='#FF5733', lw=2.5, label="Chebyshev's Bound")
    plt.plot(k_values, normal_probs, color='#33A1FF', lw=2.5, label="Normal Distribution")
    
    # Highlight specific value from the problem: k = 2 (20 units = 2*std_dev)
    k_specific = 2
    chebyshev_specific = chebyshev_bound(k_specific)
    normal_specific = normal_probability_beyond_k_std(k_specific)
    
    plt.scatter([k_specific], [chebyshev_specific], color='#FF5733', s=100, zorder=3, edgecolor='white', linewidth=1.5)
    plt.scatter([k_specific], [normal_specific], color='#33A1FF', s=100, zorder=3, edgecolor='white', linewidth=1.5)
    
    # Add labels and title
    plt.xlabel('Number of Standard Deviations (k)')
    plt.ylabel('Probability P(|X - μ| > k·σ)')
    plt.title("Probability Bounds Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Log scale makes it easier to compare these probabilities
    plt.yscale('log')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Print detailed information
        print("\nChebyshev comparison details:")
        print(f"For k = {k_specific} (deviation of {k_specific*std_dev} units from the mean):")
        print(f"Chebyshev bound: {chebyshev_specific:.4f}")
        print(f"Actual probability (normal distribution): {normal_specific:.4f}")
        print(f"The Chebyshev bound is {chebyshev_specific/normal_specific:.1f} times larger than the exact probability")
        print(f"Visualization saved to {save_path}")
    
    plt.close()

def visualize_markov_bound(mean=50, save_path=None):
    """
    Create a simplified visualization of Markov's inequality
    """
    plt.figure(figsize=(9, 5))
    
    # Range of threshold values
    thresholds = np.linspace(mean/2, mean*3, 100)
    
    # Calculate Markov bounds
    markov_bounds = [markov_bound(mean, t) for t in thresholds]
    
    # Plot the bounds
    plt.plot(thresholds, markov_bounds, color='#2ECC71', lw=2.5, label="Markov's Bound")
    
    # Highlight specific value from the problem: threshold = 100
    threshold_specific = 100
    markov_specific = markov_bound(mean, threshold_specific)
    
    plt.scatter([threshold_specific], [markov_specific], color='#2ECC71', s=100, zorder=3, edgecolor='white', linewidth=1.5)
    
    # Add vertical line for the mean
    plt.axvline(mean, color='#7D3C98', linestyle='--', alpha=0.7, lw=1.5, label='Mean')
    
    # Add labels and title
    plt.xlabel('Threshold a')
    plt.ylabel('Probability Bound P(X ≥ a)')
    plt.title("Markov's Inequality")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.ylim(0, 1.1)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Print detailed information
        print("\nMarkov bound details:")
        print(f"For a non-negative random variable with E[X] = {mean}:")
        print(f"P(X ≥ {threshold_specific}) ≤ {mean}/{threshold_specific} = {markov_specific:.2f}")
        print(f"Visualization saved to {save_path}")
    
    plt.close()

def visualize_normal_distribution_tails(mean=50, std_dev=10, k=2, save_path=None):
    """
    Create a simplified visualization of the normal distribution with tails beyond k standard deviations highlighted
    """
    plt.figure(figsize=(9, 5))
    
    # Create x values for the normal distribution
    x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)
    
    # Calculate the PDF values
    pdf = stats.norm.pdf(x, mean, std_dev)
    
    # Plot the PDF
    plt.plot(x, pdf, color='#3498DB', lw=2.5, label='Normal Distribution')
    
    # Calculate the thresholds
    lower_threshold = mean - k*std_dev
    upper_threshold = mean + k*std_dev
    
    # Highlight the tails
    mask_lower = x <= lower_threshold
    mask_upper = x >= upper_threshold
    
    plt.fill_between(x, pdf, where=mask_lower, alpha=0.7, color='#E74C3C')
    plt.fill_between(x, pdf, where=mask_upper, alpha=0.7, color='#E74C3C')
    
    # Add vertical lines for the thresholds
    plt.axvline(lower_threshold, color='#E74C3C', linestyle='--', lw=1.5)
    plt.axvline(upper_threshold, color='#E74C3C', linestyle='--', lw=1.5, 
               label='μ ± 2σ')
    
    # Add vertical line for the mean
    plt.axvline(mean, color='#7D3C98', linestyle='-', lw=1.5, label='Mean')
    
    # Add labels and title
    plt.xlabel('Feature Value')
    plt.ylabel('Probability Density')
    plt.title('Normal Distribution with Outlier Regions')
    plt.grid(False)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Print detailed information
        prob = normal_probability_beyond_k_std(k)
        chebyshev = chebyshev_bound(k)
        print("\nNormal distribution tails details:")
        print(f"Normal distribution with mean={mean}, std_dev={std_dev}")
        print(f"Highlighted regions: Outside μ ± {k}σ = {mean} ± {k*std_dev} = [{lower_threshold}, {upper_threshold}]")
        print(f"P(|X - μ| > {k}σ) = {prob:.6f}")
        print(f"Chebyshev bound: {chebyshev:.4f}")
        print(f"Visualization saved to {save_path}")
    
    plt.close()

def visualize_outlier_distributions(mean=50, std_dev=10, save_path=None):
    """
    Create a simplified visualization comparing probability distributions for outlier detection
    """
    plt.figure(figsize=(9, 5))
    
    # Create x values
    x = np.linspace(0, mean + 5*std_dev, 1000)
    
    # Calculate PDFs for different distributions
    normal_pdf = stats.norm.pdf(x, mean, std_dev)
    
    # Create a heavy-tailed distribution (t-distribution with 3 degrees of freedom, scaled)
    t_dist_scaled = stats.t.pdf((x - mean)/(std_dev/np.sqrt(3/5)), 3) / (std_dev/np.sqrt(3/5))
    
    # Create a light-tailed distribution (uniform distribution with same mean and variance)
    uniform_width = np.sqrt(12) * std_dev
    uniform_left = mean - uniform_width/2
    uniform_right = mean + uniform_width/2
    uniform_pdf = np.zeros_like(x)
    uniform_pdf[(x >= uniform_left) & (x <= uniform_right)] = 1/uniform_width
    
    # Plot the PDFs
    plt.plot(x, normal_pdf, color='#3498DB', lw=2.5, label='Normal')
    plt.plot(x, t_dist_scaled, color='#E74C3C', lw=2.5, label='Heavy-tailed')
    plt.plot(x, uniform_pdf, color='#2ECC71', lw=2.5, label='Light-tailed')
    
    # Add vertical line for the mean
    plt.axvline(mean, color='#7D3C98', linestyle='--', lw=1.5, label='Mean')
    
    # Add vertical lines for outlier thresholds
    k = 2
    upper_threshold = mean + k*std_dev
    plt.axvline(upper_threshold, color='#F39C12', linestyle=':', lw=2, 
              label='Outlier Threshold')
    
    # Add labels and title
    plt.xlabel('Feature Value')
    plt.ylabel('Probability Density')
    plt.title('Different Distributions with Same Mean and Variance')
    plt.grid(False)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Print detailed information
        normal_prob = 1 - stats.norm.cdf(upper_threshold, mean, std_dev)
        t_prob = 1 - stats.t.cdf((upper_threshold - mean)/(std_dev/np.sqrt(3/5)), 3)
        cheb_bound = chebyshev_bound(k)/2  # Divide by 2 for one-sided bound
        
        print("\nOutlier distributions details:")
        print(f"Comparing distributions with the same mean ({mean}) and variance ({std_dev}²)")
        print(f"P(X > μ + {k}σ = {upper_threshold}) for different distributions:")
        print(f"  Normal distribution: {normal_prob:.6f}")
        print(f"  Heavy-tailed distribution: {t_prob:.6f}")
        print(f"  Uniform distribution: 0 (bounded)")
        print(f"  One-sided Chebyshev bound: {cheb_bound:.4f}")
        print(f"Visualization saved to {save_path}")
    
    plt.close()

def visualize_distributions_3d(mean=50, std_dev=10, save_path=None):
    """
    Create a 3D visualization showing how the normal distribution compares to bounds
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create a figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create grid for x (feature values) and k (number of std devs)
    x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 100)
    k_values = np.linspace(0.5, 4, 20)
    X, K = np.meshgrid(x, k_values)
    
    # Calculate z values for the normal PDF
    Z = np.zeros_like(X)
    for i, k in enumerate(k_values):
        Z[i, :] = stats.norm.pdf(x, mean, std_dev)
    
    # Create the 3D surface plot
    surf = ax.plot_surface(X, K, Z, cmap='Blues', alpha=0.8, edgecolor='none')
    
    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Probability Density')
    
    # Add planes for specific standard deviations
    for k_val in [1, 2, 3]:
        # Calculate the bounds
        lower_bound = mean - k_val * std_dev
        upper_bound = mean + k_val * std_dev
        
        # Plot vertical planes at the bounds
        xx = np.ones(10) * lower_bound
        kk = np.linspace(0.5, 4, 10)
        zz = np.linspace(0, stats.norm.pdf(mean, mean, std_dev)*1.2, 10)
        KK, ZZ = np.meshgrid(kk, zz)
        XX = np.ones_like(KK) * lower_bound
        ax.plot_surface(XX, KK, ZZ, color='red', alpha=0.3)
        
        xx = np.ones(10) * upper_bound
        XX = np.ones_like(KK) * upper_bound
        ax.plot_surface(XX, KK, ZZ, color='red', alpha=0.3)
    
    # Add labels
    ax.set_xlabel('Feature Value')
    ax.set_ylabel('Number of Standard Deviations (k)')
    ax.set_zlabel('Probability Density')
    ax.set_title('Normal Distribution with Standard Deviation Bounds')
    
    # Add a text for the mean
    ax.text(mean, 4, 0, f'μ = {mean}', color='purple', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n3D visualization saved to {save_path}")
    
    plt.close()

def visualize_bound_comparison(mean=50, std_dev=10, save_path=None):
    """
    Create a visualization comparing both Chebyshev and Markov bounds
    """
    plt.figure()
    
    # Set up x-axis values
    x_values = np.linspace(mean - 4*std_dev, mean + 6*std_dev, 1000)
    
    # Calculate normal PDF
    pdf = stats.norm.pdf(x_values, mean, std_dev)
    
    # Plot normal distribution
    plt.plot(x_values, pdf, color='#3498DB', lw=2.5, label='Normal Distribution')
    
    # Calculate and plot CDF
    cdf = stats.norm.cdf(x_values, mean, std_dev)
    plt.plot(x_values, cdf/5, color='#9B59B6', lw=2, linestyle=':', label='CDF (scaled)')
    
    # Highlight areas for bounds
    # For Chebyshev: |X - μ| > 2σ
    lower_cheb = mean - 2*std_dev
    upper_cheb = mean + 2*std_dev
    cheb_bound_val = chebyshev_bound(2)
    
    # For Markov: X > 100
    markov_threshold = 100
    markov_bound_val = markov_bound(mean, markov_threshold)
    
    # Add vertical lines
    plt.axvline(lower_cheb, color='#E74C3C', linestyle='--', lw=1.5, label='Chebyshev Bounds')
    plt.axvline(upper_cheb, color='#E74C3C', linestyle='--', lw=1.5)
    plt.axvline(markov_threshold, color='#2ECC71', linestyle='--', lw=1.5, label='Markov Threshold')
    
    # Add horizontal line for bounds
    x_range = [x_values[0], x_values[-1]]
    plt.plot(x_range, [cheb_bound_val/5, cheb_bound_val/5], 'r-', lw=1.5, alpha=0.7)
    plt.plot([markov_threshold, x_values[-1]], [markov_bound_val/5, markov_bound_val/5], 'g-', lw=1.5, alpha=0.7)
    
    # Fill areas representing probability bounds
    plt.fill_between(x_values, pdf, where=(x_values <= lower_cheb) | (x_values >= upper_cheb), 
                   color='#E74C3C', alpha=0.3)
    plt.fill_between(x_values, pdf, where=(x_values >= markov_threshold), 
                   color='#2ECC71', alpha=0.3)
    
    # Add labels and title
    plt.xlabel('Feature Value')
    plt.ylabel('Probability Density / Scaled Bounds')
    plt.title('Chebyshev and Markov Bounds Comparison')
    plt.grid(False)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        print("\nBound comparison details:")
        print(f"Chebyshev bound for 2σ deviation: {cheb_bound_val:.4f}")
        print(f"Markov bound for X ≥ 100: {markov_bound_val:.4f}")
        print(f"Visualization saved to {save_path}")
    
    plt.close()

def main():
    """Generate all visualizations for Question 15 of the L2.1 quiz"""
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_1_Quiz_15")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 15 of the L2.1 Probability quiz: Probability Inequalities in ML...")
    
    # Problem parameters
    mean = 50  # Mean of the feature values
    std_dev = 10  # Standard deviation of the feature values
    
    # Task 1: Using Chebyshev's inequality for deviation > 20 units
    print("\nTask 1: Using Chebyshev's inequality for deviation > 20 units")
    k = 20 / std_dev  # Number of standard deviations
    chebyshev_probability = chebyshev_bound(k)
    print(f"Chebyshev's inequality gives P(|X - {mean}| > 20) ≤ {chebyshev_probability:.4f}")
    
    # Task 2: Using Markov's inequality for P(X > 100)
    print("\nTask 2: Using Markov's inequality for P(X > 100)")
    threshold = 100
    markov_probability = markov_bound(mean, threshold)
    print(f"Markov's inequality gives P(X ≥ {threshold}) ≤ {markov_probability:.4f}")
    
    # Task 4: Exact probability if the distribution is normal
    print("\nTask 4: Exact probability if the distribution is normal")
    exact_probability = normal_probability_beyond_k_std(k)
    print(f"For a normal distribution, P(|X - {mean}| > 20) = {exact_probability:.6f}")
    print(f"Comparison: Chebyshev bound is {chebyshev_probability/exact_probability:.1f} times larger than the exact probability")
    
    # Generate visualizations
    visualize_chebyshev_comparison(mean, std_dev, 
                                 save_path=os.path.join(save_dir, "chebyshev_comparison.png"))
    
    visualize_markov_bound(mean,
                         save_path=os.path.join(save_dir, "markov_bound.png"))
    
    visualize_normal_distribution_tails(mean, std_dev, k,
                                      save_path=os.path.join(save_dir, "normal_tails.png"))
    
    visualize_outlier_distributions(mean, std_dev,
                                  save_path=os.path.join(save_dir, "outlier_distributions.png"))
    
    # Generate new visualizations
    visualize_distributions_3d(mean, std_dev,
                             save_path=os.path.join(save_dir, "distributions_3d.png"))
    
    visualize_bound_comparison(mean, std_dev,
                              save_path=os.path.join(save_dir, "bound_comparison.png"))
    
    print(f"\nAll visualizations have been saved to: {save_dir}")

if __name__ == "__main__":
    main() 