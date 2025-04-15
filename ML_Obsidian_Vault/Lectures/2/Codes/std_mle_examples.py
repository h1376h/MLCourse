import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

def plot_std_likelihood(data, title, save_path=None):
    """Plot the likelihood function for the standard deviation and highlight the MLE."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate MLE estimates
    mle_mean = np.mean(data)
    mle_std = np.std(data, ddof=0)  # MLE uses n in denominator
    
    # Create a range of possible std values to plot
    possible_stds = np.linspace(mle_std * 0.5, mle_std * 2, 1000)
    
    # Calculate the log-likelihood for each possible std
    log_likelihood = []
    for std in possible_stds:
        # Log-likelihood for normal with fixed mean and varying std
        ll = sum(norm.logpdf(data, mle_mean, std))
        log_likelihood.append(ll)
    
    # Normalize the log-likelihood for better visualization
    log_likelihood = np.array(log_likelihood)
    log_likelihood = log_likelihood - np.min(log_likelihood)
    log_likelihood = log_likelihood / np.max(log_likelihood)
    
    # Plot the log-likelihood function
    ax.plot(possible_stds, log_likelihood, 'b-', linewidth=2)
    ax.axvline(x=mle_std, color='r', linestyle='--', 
              label=f'MLE σ = {mle_std:.4f}')
    
    ax.set_title(f"{title} - Log-Likelihood for Standard Deviation")
    ax.set_xlabel('σ (Standard Deviation Parameter)')
    ax.set_ylabel('Normalized Log-Likelihood')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return mle_std

def analyze_std_mle(name, data, context, save_dir=None):
    """Analyze the standard deviation of data using MLE."""
    print(f"\n{'='*50}")
    print(f"{name} Example")
    print(f"{'='*50}")
    print(f"Context: {context}")
    
    # Step 1: Data analysis
    print("\nStep 1: Data Analysis")
    print(f"- Data: {data}")
    print(f"- Number of observations: {len(data)}")
    
    # Step 2: Calculate MLE for mean
    print("\nStep 2: Calculate Sample Mean")
    mle_mean = np.mean(data)
    print(f"- Sample mean (μ) = {mle_mean:.4f}")
    
    # Step 3: Calculate MLE for std
    print("\nStep 3: Maximum Likelihood Estimation for Standard Deviation")
    # Calculate squared deviations from mean
    squared_devs = [(x - mle_mean)**2 for x in data]
    print(f"- Squared deviations: {[f'{dev:.8f}' for dev in squared_devs]}")
    
    # Sum squared deviations
    sum_sq_devs = sum(squared_devs)
    print(f"- Sum of squared deviations: {sum_sq_devs:.8f}")
    
    # MLE variance and std
    n = len(data)
    mle_var = sum_sq_devs / n
    mle_std = np.sqrt(mle_var)
    print(f"- MLE variance (σ²) = {mle_var:.8f}")
    print(f"- MLE standard deviation (σ) = {mle_std:.4f}")
    
    # Create save path if directory is provided
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"std_mle_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        save_path = os.path.join(save_dir, filename)
    
    # Step 4: Visualize and confirm
    print("\nStep 4: Visualization")
    plot_std_likelihood(data, name, save_path)
    
    # Step 5: Compare with biased vs unbiased estimator
    print("\nStep 5: Biased vs. Unbiased Estimator")
    unbiased_var = sum_sq_devs / (n - 1)
    unbiased_std = np.sqrt(unbiased_var)
    print(f"- Biased MLE (σ) = {mle_std:.4f}")
    print(f"- Unbiased estimator (s) = {unbiased_std:.4f}")
    print(f"- Difference: {((unbiased_std / mle_std) - 1) * 100:.2f}%")
    
    # Step 6: Interpretation
    print("\nStep 6: Interpretation")
    print(f"- The MLE standard deviation of {mle_std:.4f} estimates the true variability in the process")
    print(f"- Approximately 95% of values are expected to fall within ±{2*mle_std:.4f} of the mean")
    print(f"- The MLE is a biased estimator that underestimates the true standard deviation")
    print(f"- For an unbiased estimate, use the sample standard deviation: {unbiased_std:.4f}")
    
    return {"std": mle_std, "mean": mle_mean, "path": save_path}

def generate_std_examples(save_dir=None):
    results = {}
    
    print("""
    Let's analyze different scenarios using Maximum Likelihood Estimation for standard deviation!
    Each example will show how we can estimate variability using only the observed data.
    """)

    # Example 1: Ball Bearing Diameter
    bearing_data = [10.02, 9.98, 10.05, 9.97, 10.01, 10.03, 9.99, 10.04, 10.00, 9.96]
    bearing_context = """
    A quality control engineer is analyzing the diameter of manufactured ball bearings.
    - You measured 10 ball bearings (in mm)
    - Using only the observed data to estimate the manufacturing variability
    """
    bearing_results = analyze_std_mle("Ball Bearing Diameter", bearing_data, bearing_context, save_dir)
    results["Ball Bearing Diameter"] = bearing_results

    # Example 2: Battery Life
    battery_data = [4.8, 5.2, 4.9, 5.1, 4.7, 5.0, 4.9, 5.3, 4.8, 5.2, 5.0, 4.8]  # hours
    battery_context = """
    An electronics manufacturer is testing the life of rechargeable batteries.
    - You measured the runtime of 12 batteries (in hours)
    - Using only the observed data to estimate the variability in battery life
    """
    battery_results = analyze_std_mle("Battery Life", battery_data, battery_context, save_dir)
    results["Battery Life"] = battery_results

    # Example 3: Reaction Time
    reaction_data = [0.32, 0.29, 0.35, 0.30, 0.28, 0.33, 0.31, 0.34, 0.30, 0.32]  # seconds
    reaction_context = """
    A researcher is measuring reaction times in a psychology experiment.
    - You recorded reaction times from 10 participants (in seconds)
    - Using only the observed data to estimate the variability in human response
    """
    reaction_results = analyze_std_mle("Reaction Time", reaction_data, reaction_context, save_dir)
    results["Reaction Time"] = reaction_results
    
    # Example 4: Temperature Readings
    temp_data = [21.2, 20.8, 21.5, 20.9, 21.3, 21.1, 20.7, 21.0, 21.2, 20.9, 21.4, 21.1]  # Celsius
    temp_context = """
    A climate scientist is analyzing daily temperature readings.
    - You recorded temperatures for 12 days (in Celsius)
    - Using only the observed data to estimate temperature variability
    """
    temp_results = analyze_std_mle("Temperature Readings", temp_data, temp_context, save_dir)
    results["Temperature Readings"] = temp_results
    
    return results

if __name__ == "__main__":
    # Use relative path for save directory
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level and then to Images
    save_dir = os.path.join(os.path.dirname(script_dir), "Images")
    generate_std_examples(save_dir) 