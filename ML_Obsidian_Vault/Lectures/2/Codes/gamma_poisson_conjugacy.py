import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

def gamma_poisson_example():
    """
    Demonstrates Gamma-Poisson conjugate prior relationship.
    Gamma prior with Poisson likelihood results in Gamma posterior.
    Useful for modeling count data such as website visits or arrival times.
    """
    # Prior parameters
    alpha_prior = 3
    beta_prior = 0.5  # Using rate parameterization

    # Observed data: counts from a Poisson process
    counts = [5, 8, 10, 4, 7, 12, 9]
    sum_counts = sum(counts)
    n_observations = len(counts)

    # Posterior parameters
    alpha_posterior = alpha_prior + sum_counts
    beta_posterior = beta_prior + n_observations

    # Plot the prior and posterior distributions
    x = np.linspace(0, 30, 1000)
    prior = stats.gamma.pdf(x, alpha_prior, scale=1/beta_prior)
    posterior = stats.gamma.pdf(x, alpha_posterior, scale=1/beta_posterior)

    plt.figure(figsize=(10, 6))
    plt.plot(x, prior, 'r-', lw=2, label=f'Prior: Gamma({alpha_prior}, {beta_prior})')
    plt.plot(x, posterior, 'b-', lw=2, label=f'Posterior: Gamma({alpha_posterior}, {beta_posterior})')
    plt.axvline(x=sum_counts/n_observations, color='g', linestyle='--', label='MLE')
    
    plt.fill_between(x, 0, prior, color='red', alpha=0.2)
    plt.fill_between(x, 0, posterior, color='blue', alpha=0.2)

    plt.xlabel('λ (rate parameter)')
    plt.ylabel('Probability Density')
    plt.title('Gamma-Poisson Conjugacy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure using consistent path construction
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(os.path.dirname(script_dir), "Images")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "gamma_poisson_conjugacy.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    plt.close()

    # Calculate MAP and mean estimates
    map_estimate = (alpha_posterior - 1) / beta_posterior
    mean_estimate = alpha_posterior / beta_posterior
    mle_estimate = sum_counts / n_observations

    print(f"MAP estimate: {map_estimate:.4f}")
    print(f"Posterior mean: {mean_estimate:.4f}")
    print(f"MLE estimate: {mle_estimate:.4f}")

if __name__ == "__main__":
    gamma_poisson_example() 