import numpy as np
import matplotlib.pyplot as plt
import os

def plot_likelihood_function(data, title, save_path=None):
    """Plot the likelihood function for Bernoulli data and highlight the MLE."""
    fig, ax = plt.subplots(figsize=(10, 6))
    theta = np.linspace(0, 1, 1000)
    
    # Calculate the likelihood function
    successes = sum(data)
    trials = len(data)
    likelihood = theta**successes * (1-theta)**(trials-successes)
    
    # Find the MLE
    mle_estimate = successes / trials
    
    # Plot the likelihood function
    ax.plot(theta, likelihood, 'g-', linewidth=2)
    ax.axvline(x=mle_estimate, color='r', linestyle='--', 
              label=f'MLE Estimate = {mle_estimate:.3f}')
    
    ax.set_title(f"{title} - Likelihood Function")
    ax.set_xlabel('θ (Probability of Success)')
    ax.set_ylabel('Likelihood')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()  # Close the figure to avoid displaying in notebooks
    
    return mle_estimate

def analyze_bernoulli_data(name, data, context, save_dir=None):
    """Analyze Bernoulli data with detailed steps using MLE."""
    print(f"\n{'='*50}")
    print(f"{name} Example")
    print(f"{'='*50}")
    print(f"Context: {context}")
    
    # Step 1: Data analysis
    print("\nStep 1: Data Analysis")
    successes = sum(data)
    trials = len(data)
    print(f"- Data: {data}")
    print(f"- Number of successes: {successes}")
    print(f"- Number of trials: {trials}")
    
    # Step 2: Calculate MLE
    print("\nStep 2: Maximum Likelihood Estimation")
    print("- MLE is simply the proportion of successes in the data")
    mle_estimate = successes / trials
    print(f"- MLE estimate = {mle_estimate:.3f}")
    
    # Create save path if directory is provided
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"bernoulli_mle_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        save_path = os.path.join(save_dir, filename)
    
    # Step 3: Visualize and confirm
    print("\nStep 3: Visualization")
    plot_likelihood_function(data, name, save_path)
    
    # Step 4: Confidence Interval (using normal approximation)
    print("\nStep 4: Confidence Interval")
    if trials >= 30:  # Enough samples for normal approximation
        # 95% confidence interval using normal approximation
        z = 1.96  # 95% confidence level
        std_error = np.sqrt((mle_estimate * (1 - mle_estimate)) / trials)
        ci_lower = max(0, mle_estimate - z * std_error)
        ci_upper = min(1, mle_estimate + z * std_error)
        print(f"- 95% Confidence Interval: [{ci_lower:.3f}, {ci_upper:.3f}]")
    else:
        print("- Sample size too small for reliable confidence interval")
        
    # Step 5: Interpretation
    print("\nStep 5: Interpretation")
    print(f"- Based on our data alone, the most likely probability of success is {mle_estimate:.1%}")
    print(f"- Unlike MAP estimation, MLE does not incorporate prior beliefs")
    
    return mle_estimate, save_path

def generate_bernoulli_examples(save_dir=None):
    results = {}
    
    print("""
    Let's analyze different scenarios using Maximum Likelihood Estimation for Bernoulli trials!
    Each example will show how we can estimate probabilities using only the observed data.
    """)

    # Example 1: Coin Flips
    coin_data = [1, 1, 1, 0, 1]  # 1 = heads, 0 = tails
    coin_context = """
    You found an old coin and want to check if it's fair.
    - You flip it 5 times
    - 1 represents heads, 0 represents tails
    - Using only the observed data (no prior assumptions)
    """
    mle_result, path = analyze_bernoulli_data("Coin Flip", coin_data, coin_context, save_dir)
    results["Coin Flip"] = {"mle": mle_result, "path": path}

    # Example 2: Basketball Shots
    basketball_data = [1, 1, 0, 1, 1, 1, 0]  # 1 = made shot, 0 = missed
    basketball_context = """
    You're practicing basketball free throws.
    - You take 7 shots
    - 1 represents made shots, 0 represents misses
    - Using only the observed data (no prior assumptions)
    """
    mle_result, path = analyze_bernoulli_data("Basketball Shots", basketball_data, basketball_context, save_dir)
    results["Basketball Shots"] = {"mle": mle_result, "path": path}

    # Example 3: Video Game Wins
    game_data = [1, 0, 0, 1, 1, 1]  # 1 = win, 0 = loss
    game_context = """
    You're tracking your wins in a video game.
    - You play 6 matches
    - 1 represents wins, 0 represents losses
    - Using only the observed data (no prior assumptions)
    """
    mle_result, path = analyze_bernoulli_data("Video Game Matches", game_data, game_context, save_dir)
    results["Video Game Matches"] = {"mle": mle_result, "path": path}

    # Example 4: Email Click Rates
    email_data = [1, 0, 0, 0, 1, 0, 0, 0, 0, 1]  # 1 = clicked, 0 = not clicked
    email_context = """
    You're analyzing email marketing campaign results.
    - You sent 10 emails
    - 1 represents when a recipient clicked a link, 0 represents no click
    - Using only the observed data (no prior assumptions)
    """
    mle_result, path = analyze_bernoulli_data("Email Click Rates", email_data, email_context, save_dir)
    results["Email Click Rates"] = {"mle": mle_result, "path": path}
    
    return results

if __name__ == "__main__":
    # Use relative path for save directory
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level and then to Images
    save_dir = os.path.join(os.path.dirname(script_dir), "Images")
    generate_bernoulli_examples(save_dir) 