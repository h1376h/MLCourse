import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

def normal_normal_example():
    """
    Demonstrates Normal-Normal conjugate prior relationship with known variance.
    Normal prior with Normal likelihood (known variance) results in Normal posterior.
    """
    # Known data variance
    data_variance = 4.0

    # Prior parameters
    prior_mean = 0.0
    prior_variance = 2.0

    # Generate some observed data
    np.random.seed(42)  # For reproducibility
    true_mean = 5.0
    data = np.random.normal(true_mean, np.sqrt(data_variance), size=10)
    data_mean = np.mean(data)
    n_samples = len(data)

    # Posterior parameters
    posterior_variance = 1.0 / (1.0/prior_variance + n_samples/data_variance)
    posterior_mean = posterior_variance * (prior_mean/prior_variance + n_samples*data_mean/data_variance)

    # Plot the prior, likelihood, and posterior distributions
    x = np.linspace(-5, 10, 1000)
    prior = stats.norm.pdf(x, prior_mean, np.sqrt(prior_variance))
    likelihood = stats.norm.pdf(x, data_mean, np.sqrt(data_variance/n_samples))
    posterior = stats.norm.pdf(x, posterior_mean, np.sqrt(posterior_variance))

    plt.figure(figsize=(10, 6))
    plt.plot(x, prior, 'r-', lw=2, label=f'Prior: N({prior_mean}, {prior_variance})')
    plt.plot(x, likelihood, 'g-', lw=2, label=f'Likelihood: N({data_mean:.2f}, {data_variance/n_samples:.2f})')
    plt.plot(x, posterior, 'b-', lw=2, label=f'Posterior: N({posterior_mean:.2f}, {posterior_variance:.2f})')
    
    plt.fill_between(x, 0, prior, color='red', alpha=0.2)
    plt.fill_between(x, 0, likelihood, color='green', alpha=0.2)
    plt.fill_between(x, 0, posterior, color='blue', alpha=0.2)

    plt.xlabel('μ (mean)')
    plt.ylabel('Probability Density')
    plt.title('Normal-Normal Conjugacy (Known Variance)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure using consistent path construction
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(os.path.dirname(script_dir), "Images")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "normal_normal_conjugacy.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    plt.show()

    print(f"Prior mean: {prior_mean}")
    print(f"Likelihood mean (MLE): {data_mean:.4f}")
    print(f"Posterior mean: {posterior_mean:.4f}")
    print(f"True mean: {true_mean}")

if __name__ == "__main__":
    normal_normal_example() 