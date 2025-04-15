import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
import math

def calculate_binomial_probabilities(n, p, k_values=None):
    """
    Calculate binomial probabilities for specified values of k.
    
    Parameters:
    n -- Number of trials
    p -- Success probability for each trial
    k_values -- List of k values for which to calculate probabilities
                If None, calculates for all k from 0 to n
    
    Returns:
    k_values -- List of k values
    probabilities -- List of corresponding probabilities
    """
    if k_values is None:
        k_values = list(range(n + 1))
    
    probabilities = []
    for k in k_values:
        # P(X = k) = (n choose k) * p^k * (1-p)^(n-k)
        probability = stats.binom.pmf(k, n, p)
        probabilities.append(probability)
    
    return k_values, probabilities

def calculate_cumulative_probability(n, p, k, lower_tail=True):
    """
    Calculate cumulative binomial probability.
    
    Parameters:
    n -- Number of trials
    p -- Success probability for each trial
    k -- Threshold value
    lower_tail -- If True, calculates P(X ≤ k), otherwise P(X > k)
    
    Returns:
    probability -- Cumulative probability
    """
    if lower_tail:
        # P(X ≤ k)
        probability = stats.binom.cdf(k, n, p)
    else:
        # P(X > k) = 1 - P(X ≤ k)
        probability = 1 - stats.binom.cdf(k, n, p)
    
    return probability

def calculate_expected_value(n, p):
    """
    Calculate the expected value (mean) of a binomial distribution.
    
    Parameters:
    n -- Number of trials
    p -- Success probability for each trial
    
    Returns:
    expected_value -- E[X] = n*p
    """
    return n * p

def calculate_variance(n, p):
    """
    Calculate the variance of a binomial distribution.
    
    Parameters:
    n -- Number of trials
    p -- Success probability for each trial
    
    Returns:
    variance -- Var(X) = n*p*(1-p)
    """
    return n * p * (1 - p)

def plot_binomial_pmf(n, p, save_path=None):
    """
    Plot the probability mass function (PMF) of a binomial distribution.
    
    Parameters:
    n -- Number of trials
    p -- Success probability for each trial
    save_path -- Path to save the figure
    """
    k_values, probabilities = calculate_binomial_probabilities(n, p)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot the PMF
    ax.bar(k_values, probabilities, alpha=0.7, color='skyblue', 
          label=f'Binomial({n}, {p}) PMF')
    
    # Highlight specific values
    specific_k = [3]
    for k in specific_k:
        ax.bar([k], [stats.binom.pmf(k, n, p)], alpha=0.7, color='red',
              label=f'P(X = {k}) = {stats.binom.pmf(k, n, p):.6f}')
    
    # Highlight P(X ≥ 1)
    at_least_one_prob = 1 - stats.binom.pmf(0, n, p)
    at_least_one_k = list(range(1, n+1))
    ax.bar(at_least_one_k, [stats.binom.pmf(k, n, p) for k in at_least_one_k], 
          alpha=0.4, color='green', label=f'P(X ≥ 1) = {at_least_one_prob:.6f}')
    
    # Highlight P(1 ≤ X ≤ 3)
    between_prob = stats.binom.cdf(3, n, p) - stats.binom.cdf(0, n, p)
    between_k = list(range(1, 4))
    ax.bar(between_k, [stats.binom.pmf(k, n, p) for k in between_k], 
          alpha=0.6, color='purple', label=f'P(1 ≤ X ≤ 3) = {between_prob:.6f}')
    
    # Add expected value line
    expected_value = calculate_expected_value(n, p)
    ax.axvline(x=expected_value, color='red', linestyle='--', 
              label=f'E[X] = {expected_value}')
    
    # Add title and labels
    ax.set_title(f'Binomial Distribution PMF: n={n}, p={p}')
    ax.set_xlabel('Number of Outliers (k)')
    ax.set_ylabel('Probability: P(X = k)')
    ax.set_xticks(list(range(n+1)))
    
    # Add legend
    ax.legend()
    
    # Add the formula
    formula = r'$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, formula, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Binomial PMF plot saved to {save_path}")
    
    plt.close()

def plot_binomial_cdf(n, p, save_path=None):
    """
    Plot the cumulative distribution function (CDF) of a binomial distribution.
    
    Parameters:
    n -- Number of trials
    p -- Success probability for each trial
    save_path -- Path to save the figure
    """
    k_values = list(range(n + 1))
    cdf_values = [stats.binom.cdf(k, n, p) for k in k_values]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot the CDF
    ax.step(k_values, cdf_values, where='post', alpha=0.7, color='blue', 
           label=f'Binomial({n}, {p}) CDF')
    ax.plot(k_values, cdf_values, 'o', alpha=0.5, color='blue')
    
    # Highlight specific values
    specific_k = [0, 3]
    for k in specific_k:
        cdf_value = stats.binom.cdf(k, n, p)
        ax.plot([k], [cdf_value], 'ro', markersize=8)
        ax.text(k + 0.1, cdf_value, f'P(X ≤ {k}) = {cdf_value:.6f}', 
               verticalalignment='bottom')
    
    # Highlight P(1 ≤ X ≤ 3)
    between_prob = stats.binom.cdf(3, n, p) - stats.binom.cdf(0, n, p)
    ax.fill_between([0, 3], [stats.binom.cdf(0, n, p), stats.binom.cdf(0, n, p)], 
                   [stats.binom.cdf(0, n, p), stats.binom.cdf(3, n, p)], 
                   alpha=0.3, color='purple', 
                   label=f'P(1 ≤ X ≤ 3) = {between_prob:.6f}')
    
    # Add title and labels
    ax.set_title(f'Binomial Distribution CDF: n={n}, p={p}')
    ax.set_xlabel('Number of Outliers (k)')
    ax.set_ylabel('Cumulative Probability: P(X ≤ k)')
    ax.set_xticks(list(range(n+1)))
    
    # Add legend
    ax.legend()
    
    # Add the formula
    formula = r'$P(X \leq k) = \sum_{i=0}^{k} \binom{n}{i} p^i (1-p)^{n-i}$'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.05, formula, transform=ax.transAxes, fontsize=12,
           verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Binomial CDF plot saved to {save_path}")
    
    plt.close()

def plot_sampling_simulation(n, p, num_simulations=1000, save_path=None):
    """
    Plot a simulation of sampling outliers from a dataset.
    
    Parameters:
    n -- Number of trials (sample size)
    p -- Success probability for each trial (probability of outlier)
    num_simulations -- Number of simulation runs
    save_path -- Path to save the figure
    """
    # Run simulations of sampling n points with outlier probability p
    simulation_results = np.random.binomial(n, p, size=num_simulations)
    
    # Count occurrences of each result
    counts = np.bincount(simulation_results, minlength=n+1)
    frequencies = counts / num_simulations
    
    # Calculate theoretical probabilities
    k_values = list(range(n + 1))
    theoretical_probs = [stats.binom.pmf(k, n, p) for k in k_values]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot simulated results
    ax1.bar(k_values, frequencies, alpha=0.7, color='skyblue',
           label=f'Simulated ({num_simulations} runs)')
    ax1.plot(k_values, theoretical_probs, 'ro-', markersize=5, alpha=0.7,
            label='Theoretical Probability')
    
    # Add title and labels
    ax1.set_title(f'Simulation of Outlier Sampling\nn={n}, p={p}, {num_simulations} simulations')
    ax1.set_xlabel('Number of Outliers in Sample')
    ax1.set_ylabel('Relative Frequency')
    ax1.set_xticks(list(range(n+1)))
    ax1.legend()
    
    # Create a visual representation of outlier detection
    # Generate one sample dataset with outliers
    np.random.seed(42)  # For reproducibility
    is_outlier = np.random.binomial(1, p, size=n)
    
    # Create scatter plot
    x = np.random.normal(0, 1, size=n)  # x coordinates
    y = np.random.normal(0, 1, size=n)  # y coordinates
    
    # Make outliers more extreme
    outlier_offset = 3
    x[is_outlier == 1] += np.sign(x[is_outlier == 1]) * outlier_offset
    y[is_outlier == 1] += np.sign(y[is_outlier == 1]) * outlier_offset
    
    # Plot regular data points
    ax2.scatter(x[is_outlier == 0], y[is_outlier == 0], color='blue', alpha=0.6, 
               label='Regular Data')
    
    # Plot outliers
    ax2.scatter(x[is_outlier == 1], y[is_outlier == 1], color='red', alpha=0.8,
               label='Outliers')
    
    # Add a circle to represent the boundary
    circle = plt.Circle((0, 0), outlier_offset, fill=False, color='green', 
                       linestyle='--', linewidth=2, label='Outlier Boundary')
    ax2.add_artist(circle)
    
    # Add title and labels
    num_outliers = np.sum(is_outlier)
    ax2.set_title(f'Sample Dataset with {num_outliers} Outliers\nProbability of outlier: {p}')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.set_aspect('equal')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sampling simulation plot saved to {save_path}")
    
    plt.close()

def main():
    """Solve outlier probability problem and create visualizations"""
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_1_Quiz_10")
    os.makedirs(save_dir, exist_ok=True)
    
    # Problem parameters
    n = 20  # sample size
    p = 0.1  # probability of outlier
    
    print("\n=== Question 10: Outlier Probability ===")
    print(f"A machine learning engineer is working with a dataset where each data point has a {p} probability")
    print(f"of being an outlier. The engineer decides to randomly sample {n} data points for a preliminary analysis.")
    
    # Task 1: Probability of exactly 3 outliers
    prob_exactly_3 = stats.binom.pmf(3, n, p)
    
    print("\n1. What is the probability that the sample contains exactly 3 outliers?")
    print(f"   P(X = 3) = ({n} choose 3) * {p}^3 * (1-{p})^{n-3}")
    print(f"   P(X = 3) = {math.comb(n, 3)} * {p}^3 * {1-p}^{n-3}")
    print(f"   P(X = 3) = {math.comb(n, 3)} * {p**3} * {(1-p)**(n-3)}")
    print(f"   P(X = 3) = {prob_exactly_3:.10f}")
    
    # Task 2: Probability of at least 1 outlier
    prob_at_least_1 = 1 - stats.binom.pmf(0, n, p)
    
    print("\n2. What is the probability that the sample contains at least 1 outlier?")
    print(f"   P(X ≥ 1) = 1 - P(X = 0)")
    print(f"   P(X ≥ 1) = 1 - ({n} choose 0) * {p}^0 * (1-{p})^{n}")
    print(f"   P(X ≥ 1) = 1 - {(1-p)**n}")
    print(f"   P(X ≥ 1) = {prob_at_least_1:.10f}")
    
    # Task 3: Expected number of outliers
    expected_value = calculate_expected_value(n, p)
    variance = calculate_variance(n, p)
    std_deviation = np.sqrt(variance)
    
    print("\n3. What is the expected number of outliers in the sample?")
    print(f"   E[X] = n * p = {n} * {p} = {expected_value}")
    print(f"   Var(X) = n * p * (1-p) = {n} * {p} * {1-p} = {variance}")
    print(f"   Std(X) = sqrt(Var(X)) = {std_deviation:.10f}")
    
    # Task 4: Probability of between 1 and 3 outliers, inclusive
    prob_between_1_and_3 = stats.binom.cdf(3, n, p) - stats.binom.cdf(0, n, p)
    
    print("\n4. Calculate the probability that the sample contains between 1 and 3 outliers, inclusive:")
    print(f"   P(1 ≤ X ≤ 3) = P(X ≤ 3) - P(X ≤ 0)")
    print(f"   P(1 ≤ X ≤ 3) = {stats.binom.cdf(3, n, p):.10f} - {stats.binom.cdf(0, n, p):.10f}")
    print(f"   P(1 ≤ X ≤ 3) = {prob_between_1_and_3:.10f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    plot_binomial_pmf(n, p, save_path=os.path.join(save_dir, "binomial_pmf.png"))
    plot_binomial_cdf(n, p, save_path=os.path.join(save_dir, "binomial_cdf.png"))
    plot_sampling_simulation(n, p, save_path=os.path.join(save_dir, "sampling_simulation.png"))
    
    print(f"\nAll calculations and visualizations for Question 10 have been completed.")
    print(f"Visualization files have been saved to: {save_dir}")
    
    # Summary of results
    print("\n=== Summary of Results ===")
    print(f"1. P(X = 3) = {prob_exactly_3:.10f}")
    print(f"2. P(X ≥ 1) = {prob_at_least_1:.10f}")
    print(f"3. E[X] = {expected_value}")
    print(f"4. P(1 ≤ X ≤ 3) = {prob_between_1_and_3:.10f}")

if __name__ == "__main__":
    main() 