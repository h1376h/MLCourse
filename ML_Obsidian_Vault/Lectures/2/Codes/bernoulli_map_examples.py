import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import os

def plot_beta_distributions(prior_params, data, title, save_path=None):
    """Plot the prior, likelihood, posterior distributions, and likelihood function."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    theta = np.linspace(0, 1, 1000)
    
    # Prior Beta distribution
    prior = beta.pdf(theta, prior_params[0], prior_params[1])
    ax1.plot(theta, prior, 'b-', label='Prior', alpha=0.6)
    
    # Posterior Beta distribution
    successes = sum(data)
    trials = len(data)
    posterior_a = prior_params[0] + successes
    posterior_b = prior_params[1] + trials - successes
    posterior = beta.pdf(theta, posterior_a, posterior_b)
    ax1.plot(theta, posterior, 'r-', label='Posterior', alpha=0.6)
    
    # MAP estimate
    map_estimate = (posterior_a - 1) / (posterior_a + posterior_b - 2)
    ax1.axvline(x=map_estimate, color='k', linestyle='--', 
                label=f'MAP Estimate = {map_estimate:.3f}')
    
    ax1.set_title(f"{title} - Distributions")
    ax1.set_xlabel('θ (Probability of Success)')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Likelihood function plot
    # For Bernoulli data, likelihood = θ^s * (1-θ)^(n-s)
    likelihood = theta**successes * (1-theta)**(trials-successes)
    # Remove normalization to show actual likelihood values
    ax2.plot(theta, likelihood, 'g-', label=f'Likelihood (s={successes}, n={trials})')
    ax2.set_title('Likelihood Function')
    ax2.set_xlabel('θ')
    ax2.set_ylabel('Likelihood')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()  # Close the figure to avoid displaying in notebooks
    
    return map_estimate, posterior_a, posterior_b

def analyze_bernoulli_data(name, prior_params, data, context, save_dir=None):
    """Analyze Bernoulli data with detailed steps."""
    print(f"\n{'='*50}")
    print(f"{name} Example")
    print(f"{'='*50}")
    print(f"Context: {context}")
    
    # Step 1: Prior information
    print("\nStep 1: Prior Information")
    print(f"- Using Beta({prior_params[0]}, {prior_params[1]}) as prior")
    print(f"- This represents our initial belief about the probability")
    
    # Step 2: Data analysis
    print("\nStep 2: Data Analysis")
    successes = sum(data)
    trials = len(data)
    print(f"- Data: {data}")
    print(f"- Number of successes: {successes}")
    print(f"- Number of trials: {trials}")
    print(f"- Sample success rate: {successes/trials:.3f}")
    
    # Step 3: Calculate posterior
    print("\nStep 3: Posterior Calculation")
    print("- Posterior follows Beta(α + successes, β + failures)")
    posterior_a = prior_params[0] + successes
    posterior_b = prior_params[1] + trials - successes
    print(f"- Posterior parameters: Beta({posterior_a}, {posterior_b})")
    
    # Create save path if directory is provided
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"bernoulli_map_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        save_path = os.path.join(save_dir, filename)
    
    # Step 4: Calculate MAP and plot
    print("\nStep 4: MAP Estimation")
    map_estimate, _, _ = plot_beta_distributions(prior_params, data, name, save_path)
    print(f"- MAP estimate = {map_estimate:.3f}")
    
    # Step 5: Interpretation
    print("\nStep 5: Interpretation")
    print(f"- Based on our analysis, the most likely probability of success is {map_estimate:.1%}")
    
    return map_estimate, save_path

def generate_bernoulli_examples(save_dir=None):
    results = {}
    
    print("""
    Let's analyze different scenarios using Bernoulli trials!
    Each example will show how we can use Bayesian analysis to estimate probabilities.
    """)

    # Example 1: Coin Flips
    coin_data = [1, 1, 1, 0, 1]  # 1 = heads, 0 = tails
    coin_context = """
    You found an old coin and want to check if it's fair.
    - You flip it 5 times
    - 1 represents heads, 0 represents tails
    - Prior assumption: The coin is probably fair (Beta(2,2) represents this belief)
    """
    map_result, path = analyze_bernoulli_data("Coin Flip", (2, 2), coin_data, coin_context, save_dir)
    results["Coin Flip"] = {"map": map_result, "path": path}

    # Example 2: Basketball Shots
    basketball_data = [1, 1, 0, 1, 1, 1, 0]  # 1 = made shot, 0 = missed
    basketball_context = """
    You're practicing basketball free throws.
    - You take 7 shots
    - 1 represents made shots, 0 represents misses
    - Prior assumption: Average teen success rate is 40% (Beta(2,3) represents this belief)
    """
    map_result, path = analyze_bernoulli_data("Basketball Shots", (2, 3), basketball_data, basketball_context, save_dir)
    results["Basketball Shots"] = {"map": map_result, "path": path}

    # Example 3: Video Game Wins (Beginner)
    game_data = [1, 0, 0, 1, 1, 1]  # 1 = win, 0 = loss
    game_context = """
    You're tracking your wins in a video game.
    - You play 6 matches
    - 1 represents wins, 0 represents losses
    - Prior assumption: You're a beginner (Beta(1,2) represents this belief)
    """
    map_result, path = analyze_bernoulli_data("Video Game Matches (Beginner)", (1, 2), game_data, game_context, save_dir)
    results["Video Game Matches (Beginner)"] = {"map": map_result, "path": path}

    # Example 4: Pro Player Video Game Wins
    pro_game_data = [1, 1, 0, 1, 1, 1]  # 1 = win, 0 = loss
    pro_game_context = """
    You're tracking a pro player's wins in the same video game.
    - They play 6 matches
    - 1 represents wins, 0 represents losses
    - Prior assumption: They're highly skilled (Beta(4,1) represents this belief of ~80% win rate)
    """
    map_result, path = analyze_bernoulli_data("Pro Player Video Game Matches", (4, 1), pro_game_data, pro_game_context, save_dir)
    results["Pro Player Video Game Matches"] = {"map": map_result, "path": path}
    
    return results

if __name__ == "__main__":
    # Use relative path for save directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(os.path.dirname(script_dir), "Images")
    generate_bernoulli_examples(save_dir) 