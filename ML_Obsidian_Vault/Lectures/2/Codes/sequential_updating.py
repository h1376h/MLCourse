import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

def sequential_updating_example():
    """
    Demonstrates sequential Bayesian updating with Beta-Binomial conjugacy.
    Shows how beliefs evolve as new data arrives incrementally.
    """
    # Initial prior parameters for Beta distribution
    alpha = 2
    beta = 2

    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Simulated coin flips (1 = heads, 0 = tails)
    true_probability = 0.7
    flips = np.random.binomial(1, true_probability, size=50)

    # Initialize storage for evolution of beliefs
    alphas = [alpha]
    betas = [beta]

    # Sequentially update beliefs after each coin flip
    for flip in flips:
        if flip == 1:  # Heads
            alpha += 1
        else:  # Tails
            beta += 1
        alphas.append(alpha)
        betas.append(beta)

    # Calculate the mean (expected probability) at each step
    means = [a/(a+b) for a, b in zip(alphas, betas)]

    # Plot how our belief about the coin's bias evolves
    plt.figure(figsize=(10, 6))
    plt.plot(means, '-o', alpha=0.7)
    plt.axhline(y=true_probability, color='r', linestyle='--', label='True probability')
    plt.xlabel('Number of flips observed')
    plt.ylabel('Estimated probability of heads')
    plt.title('Sequential Bayesian Updating with Beta-Binomial Conjugacy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure with consistent path handling
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(os.path.dirname(script_dir), "Images")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'sequential_updating.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Final estimate after {len(flips)} flips: {means[-1]:.4f}")
    print(f"True probability: {true_probability:.4f}")
    
    # Plot the evolution of the Beta distribution
    plt.figure(figsize=(12, 8))
    x = np.linspace(0, 1, 1000)
    
    # Plot selected iterations to show evolution
    iterations = [0, 5, 10, 25, 50]
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    
    for i, iter_num in enumerate(iterations):
        if iter_num < len(alphas):
            a, b = alphas[iter_num], betas[iter_num]
            y = stats.beta.pdf(x, a, b)
            plt.plot(x, y, color=colors[i], lw=2, 
                    label=f'After {iter_num} flips: Beta({a}, {b})')
    
    plt.axvline(x=true_probability, color='k', linestyle='--', label='True probability')
    plt.xlabel('θ (probability of heads)')
    plt.ylabel('Probability Density')
    plt.title('Evolution of Beta Distribution with Sequential Updating')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the second figure with consistent path handling
    plt.savefig(os.path.join(save_dir, 'sequential_updating_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    sequential_updating_example() 