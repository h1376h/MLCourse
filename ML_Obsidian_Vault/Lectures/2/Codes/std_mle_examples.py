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
    
    # Step 1: Data analysis with detailed information
    print("\nStep 1: Data Analysis")
    print(f"- Data: {data}")
    print(f"- Number of observations (n): {len(data)}")
    print(f"- Range: [{min(data)}, {max(data)}]")
    print(f"- Data type: {type(data[0]).__name__}")
    
    # Step 2: Calculate MLE for mean with detailed steps
    print("\nStep 2: Calculate Sample Mean (MLE for Mean)")
    print("- For a normal distribution, the MLE for the mean is the sample mean:")
    print("  μ̂_MLE = (1/n) * ∑(x_i)")
    
    # Calculate sum of observations
    data_sum = sum(data)
    n = len(data)
    
    print(f"- Sum of all observations: {' + '.join([str(x) for x in data])} = {data_sum:.4f}")
    
    # Calculate the MLE for mean
    mle_mean = data_sum / n
    print(f"- μ̂_MLE = {data_sum:.4f} / {n} = {mle_mean:.4f}")
    
    # Step 3: Calculate MLE for variance and standard deviation with detailed steps
    print("\nStep 3: Maximum Likelihood Estimation for Standard Deviation")
    print("- For a normal distribution, the MLE for the variance is:")
    print("  σ²_MLE = (1/n) * ∑(x_i - μ̂_MLE)²")
    print("- And the MLE for standard deviation is:")
    print("  σ_MLE = √(σ²_MLE)")
    
    # Calculate each deviation from mean
    print("\n- Step 3.1: Calculate deviations from the mean (x_i - μ̂_MLE)")
    deviations = []
    for i, x in enumerate(data):
        dev = x - mle_mean
        deviations.append(dev)
        print(f"  x_{i+1} - μ̂_MLE = {x} - {mle_mean:.4f} = {dev:.4f}")
    
    # Calculate squared deviations
    print("\n- Step 3.2: Square each deviation (x_i - μ̂_MLE)²")
    squared_devs = []
    for i, dev in enumerate(deviations):
        sq_dev = dev**2
        squared_devs.append(sq_dev)
        print(f"  (x_{i+1} - μ̂_MLE)² = ({deviations[i]:.4f})² = {sq_dev:.8f}")
    
    # Sum squared deviations
    sum_sq_devs = sum(squared_devs)
    print("\n- Step 3.3: Sum all squared deviations ∑(x_i - μ̂_MLE)²")
    print(f"  ∑(x_i - μ̂_MLE)² = {' + '.join([f'{dev:.8f}' for dev in squared_devs])}")
    print(f"  ∑(x_i - μ̂_MLE)² = {sum_sq_devs:.8f}")
    
    # Calculate variance (MLE)
    print("\n- Step 3.4: Calculate the MLE for variance σ²_MLE = (1/n) * ∑(x_i - μ̂_MLE)²")
    mle_var = sum_sq_devs / n
    print(f"  σ²_MLE = {sum_sq_devs:.8f} / {n} = {mle_var:.8f}")
    
    # Calculate standard deviation (MLE)
    mle_std = np.sqrt(mle_var)
    print("\n- Step 3.5: Calculate the MLE for standard deviation σ_MLE = √(σ²_MLE)")
    print(f"  σ_MLE = √{mle_var:.8f} = {mle_std:.4f}")
    
    # Create save path if directory is provided
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"std_mle_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        save_path = os.path.join(save_dir, filename)
    
    # Step 4: Visualize and confirm
    print("\nStep 4: Visualization of Likelihood Function")
    print("- Plotting the likelihood function for different values of σ")
    print("- The maximum of this function corresponds to the MLE")
    plot_std_likelihood(data, name, save_path)
    
    # Step 5: Compare with biased vs unbiased estimator
    print("\nStep 5: Compare MLE (Biased) vs. Unbiased Estimator")
    print("- The MLE for variance uses n in the denominator and is a biased estimator")
    print("- The unbiased estimator for variance uses (n-1) in the denominator")
    
    unbiased_var = sum_sq_devs / (n - 1)
    unbiased_std = np.sqrt(unbiased_var)
    
    print(f"- Biased MLE for variance (σ²): {mle_var:.8f}")
    print(f"- Unbiased estimator for variance (s²): {unbiased_var:.8f}")
    print(f"- Biased MLE for standard deviation (σ): {mle_std:.4f}")
    print(f"- Unbiased estimator for standard deviation (s): {unbiased_std:.4f}")
    
    percent_diff = ((unbiased_std / mle_std) - 1) * 100
    print(f"- Percent difference: {percent_diff:.2f}%")
    print(f"- This bias is more significant for small sample sizes")
    
    # Step 6: Interpretation
    print("\nStep 6: Interpretation of Results")
    print(f"- The MLE for the standard deviation is {mle_std:.4f}")
    print(f"- This estimates the true variability in the {name.lower()} measurements")
    print(f"- Approximately 68.3% of values are expected to fall within range: [{mle_mean-mle_std:.4f}, {mle_mean+mle_std:.4f}]")
    print(f"- Approximately 95.5% of values are expected to fall within range: [{mle_mean-2*mle_std:.4f}, {mle_mean+2*mle_std:.4f}]")
    print(f"- Approximately 99.7% of values are expected to fall within range: [{mle_mean-3*mle_std:.4f}, {mle_mean+3*mle_std:.4f}]")
    print(f"- For a more accurate estimate of population standard deviation in small samples, consider using the unbiased estimator: {unbiased_std:.4f}")
    
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