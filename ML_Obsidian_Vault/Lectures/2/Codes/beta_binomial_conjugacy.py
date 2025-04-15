import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

def beta_binomial_example():
    """
    Demonstrates Beta-Binomial conjugate prior relationship.
    Beta prior with Binomial likelihood results in Beta posterior.
    """
    # Prior parameters
    alpha_prior = 2
    beta_prior = 2

    # Observed data: 7 successes out of 10 trials
    successes = 7
    trials = 10

    # Posterior parameters
    alpha_posterior = alpha_prior + successes
    beta_posterior = beta_prior + (trials - successes)

    # Plot the prior and posterior distributions
    x = np.linspace(0, 1, 1000)
    prior = stats.beta.pdf(x, alpha_prior, beta_prior)
    posterior = stats.beta.pdf(x, alpha_posterior, beta_posterior)
    likelihood = stats.binom.pmf(successes, trials, x) * 4  # Scaled for visibility

    plt.figure(figsize=(10, 6))
    plt.plot(x, prior, 'r-', lw=2, label=f'Prior: Beta({alpha_prior}, {beta_prior})')
    plt.plot(x, posterior, 'b-', lw=2, label=f'Posterior: Beta({alpha_posterior}, {beta_posterior})')
    plt.plot(x, likelihood, 'g--', lw=2, label=f'Scaled Likelihood: Bin({successes}|{trials}, θ)')
    plt.fill_between(x, 0, posterior, color='blue', alpha=0.2)
    plt.fill_between(x, 0, prior, color='red', alpha=0.2)

    plt.xlabel('θ (probability of success)')
    plt.ylabel('Probability Density')
    plt.title('Beta-Binomial Conjugacy: Prior → Posterior')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure with correct path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(os.path.dirname(script_dir), "Images")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'beta_binomial_conjugacy.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Calculate MAP and mean estimates
    map_estimate = (alpha_posterior - 1) / (alpha_posterior + beta_posterior - 2)
    mean_estimate = alpha_posterior / (alpha_posterior + beta_posterior)

    print(f"MAP estimate: {map_estimate:.4f}")
    print(f"Posterior mean: {mean_estimate:.4f}")
    print(f"True proportion: {successes/trials:.4f}")

if __name__ == "__main__":
    beta_binomial_example() 