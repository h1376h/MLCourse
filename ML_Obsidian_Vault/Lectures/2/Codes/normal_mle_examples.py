import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

def plot_normal_likelihood(data, title, save_path=None):
    """Plot the likelihood function for normal data and highlight the MLE."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate MLE estimates
    mle_mean = np.mean(data)
    mle_std = np.std(data, ddof=0)  # Maximum likelihood estimate uses ddof=0
    
    # Create a range of possible means to plot
    possible_means = np.linspace(mle_mean - 3*mle_std, mle_mean + 3*mle_std, 1000)
    
    # Calculate the log-likelihood for each possible mean
    # For normal distributions, the likelihood function is maximized at the sample mean
    # So we plot the log-likelihood function, assuming variance is the MLE estimate
    log_likelihood = []
    for mu in possible_means:
        ll = sum(norm.logpdf(data, mu, mle_std))
        log_likelihood.append(ll)
    
    # Normalize the log-likelihood for better visualization
    log_likelihood = np.array(log_likelihood)
    log_likelihood = log_likelihood - np.min(log_likelihood)
    log_likelihood = log_likelihood / np.max(log_likelihood)
    
    # Plot the log-likelihood function
    ax.plot(possible_means, log_likelihood, 'b-', linewidth=2)
    ax.axvline(x=mle_mean, color='r', linestyle='--', 
              label=f'MLE Mean = {mle_mean:.2f}')
    
    ax.set_title(f"{title} - Log-Likelihood Function")
    ax.set_xlabel('μ (Mean Parameter)')
    ax.set_ylabel('Normalized Log-Likelihood')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()  # Close the figure to avoid displaying in notebooks
    
    return mle_mean, mle_std

def analyze_normal_data(name, data, context, save_dir=None):
    """Analyze normal data with detailed steps using MLE."""
    print(f"\n{'='*50}")
    print(f"{name} Example")
    print(f"{'='*50}")
    print(f"Context: {context}")
    
    # Step 1: Data analysis
    print("\nStep 1: Data Analysis")
    print(f"- Data: {data}")
    print(f"- Number of observations (n): {len(data)}")
    
    # Step 2: Calculate MLE for mean with detailed steps
    print("\nStep 2: Maximum Likelihood Estimation for Mean")
    print("- For a normal distribution, the MLE for the mean is the sample mean:")
    print("  μ̂_MLE = (1/n) * ∑(x_i)")
    
    # Calculate sum of observations
    data_sum = sum(data)
    n = len(data)
    
    print(f"- Sum of all observations: {' + '.join([str(x) for x in data])} = {data_sum:.4f}")
    
    # Calculate the MLE for mean
    mle_mean = data_sum / n
    print(f"- μ̂_MLE = {data_sum:.4f} / {n} = {mle_mean:.4f}")
    
    # Step 3: Calculate MLE for variance with detailed steps
    print("\nStep 3: Maximum Likelihood Estimation for Variance")
    print("- For a normal distribution, the MLE for the variance is:")
    print("  σ²_MLE = (1/n) * ∑(x_i - μ̂_MLE)²")
    
    # Calculate squared deviations from mean
    deviations = [x - mle_mean for x in data]
    squared_devs = [dev**2 for dev in deviations]
    
    # Print each deviation and squared deviation
    print("- Calculating deviations from mean:")
    for i, (x, dev, sq_dev) in enumerate(zip(data, deviations, squared_devs)):
        print(f"  ({x} - {mle_mean:.4f})² = ({dev:.4f})² = {sq_dev:.4f}")
    
    # Sum of squared deviations
    sum_squared_devs = sum(squared_devs)
    print(f"- Sum of squared deviations: {' + '.join([f'{sq:.4f}' for sq in squared_devs])} = {sum_squared_devs:.4f}")
    
    # Calculate the MLE for variance
    mle_var = sum_squared_devs / n
    print(f"- σ²_MLE = {sum_squared_devs:.4f} / {n} = {mle_var:.4f}")
    
    # Calculate the MLE for standard deviation
    mle_std = np.sqrt(mle_var)
    print(f"- σ_MLE = √{mle_var:.4f} = {mle_std:.4f}")
    
    # Create save path if directory is provided
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"normal_mle_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        save_path = os.path.join(save_dir, filename)
    
    # Step 4: Visualize and confirm
    print("\nStep 4: Visualization")
    mle_mean, mle_std = plot_normal_likelihood(data, name, save_path)
    
    # Step 5: Confidence Interval
    print("\nStep 5: Confidence Interval for Mean")
    sem = mle_std / np.sqrt(n)  # Standard error of the mean
    # Use t-distribution for small samples, but approximating with normal distribution here
    z = 1.96  # 95% confidence level
    ci_lower = mle_mean - z * sem
    ci_upper = mle_mean + z * sem
    
    print(f"- Standard Error of Mean (SEM) = σ_MLE / √n = {mle_std:.4f} / √{n} = {sem:.4f}")
    print(f"- 95% Confidence Interval = μ̂_MLE ± 1.96 × SEM = {mle_mean:.4f} ± 1.96 × {sem:.4f}")
    print(f"- 95% Confidence Interval = [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Step 6: Interpretation
    print("\nStep 6: Interpretation")
    print(f"- The MLE for the mean is {mle_mean:.4f}")
    print(f"- The MLE for the standard deviation is {mle_std:.4f}")
    print(f"- We are 95% confident that the true mean is between {ci_lower:.4f} and {ci_upper:.4f}")
    print(f"- Approximately 68% of observations should fall within one standard deviation of the mean: [{mle_mean-mle_std:.4f}, {mle_mean+mle_std:.4f}]")
    print(f"- Approximately 95% of observations should fall within two standard deviations of the mean: [{mle_mean-2*mle_std:.4f}, {mle_mean+2*mle_std:.4f}]")
    
    return {"mean": mle_mean, "std": mle_std, "path": save_path}

def generate_normal_examples(save_dir=None):
    results = {}
    
    print("""
    Let's analyze different scenarios using Maximum Likelihood Estimation for normal distributions!
    Each example will show how we can estimate parameters using only the observed data.
    """)

    # Example 1: Basketball Shot Distance
    basketball_data = [13.8, 14.2, 15.1, 13.5, 15.8, 14.9, 15.5]  # feet
    basketball_context = """
    A basketball player is practicing shots from different distances.
    - You measure the distance (in feet) of each shot attempt
    - Using only the observed data (no prior assumptions)
    """
    basketball_results = analyze_normal_data("Basketball Shot Distance", basketball_data, basketball_context, save_dir)
    results["Basketball Shot Distance"] = basketball_results

    # Example 2: Video Game Score
    game_data = [850, 920, 880, 950, 910, 890, 930, 900]  # scores
    game_context = """
    You're tracking your scores in a video game.
    - You recorded 8 recent scores
    - Using only the observed data (no prior assumptions)
    """
    game_results = analyze_normal_data("Video Game Score", game_data, game_context, save_dir)
    results["Video Game Score"] = game_results

    # Example 3: Test Scores
    test_data = [85, 92, 78, 88, 95, 82, 90, 84, 88]  # percentages
    test_context = """
    A teacher is analyzing student test scores.
    - Records show 9 recent test scores (as percentages)
    - Using only the observed data (no prior assumptions)
    """
    test_results = analyze_normal_data("Test Scores", test_data, test_context, save_dir)
    results["Test Scores"] = test_results

    # Example 4: Daily Steps
    steps_data = [8200, 7500, 10300, 9100, 7800, 8500, 9400, 8200, 9100, 8700]  # steps
    steps_context = """
    You're tracking your daily step count to monitor physical activity.
    - You recorded your steps for 10 days
    - Using only the observed data (no prior assumptions)
    """
    steps_results = analyze_normal_data("Daily Steps", steps_data, steps_context, save_dir)
    results["Daily Steps"] = steps_results
    
    return results

if __name__ == "__main__":
    # Use relative path for save directory
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level and then to Images
    save_dir = os.path.join(os.path.dirname(script_dir), "Images")
    generate_normal_examples(save_dir) 