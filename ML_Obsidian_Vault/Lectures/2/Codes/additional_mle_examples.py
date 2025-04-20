import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
from scipy.optimize import minimize

def plot_mle_result(data, title, mean=None, std=None, is_censored=False, censoring_point=None, 
                   is_grouped=False, bins=None, bin_counts=None, save_path=None):
    """Plot the data and fitted normal distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if is_censored:
        # For censored data, we need to handle differently
        uncensored_data = [x for x in data if x < censoring_point]
        n_censored = sum(1 for x in data if x >= censoring_point)
        
        # Plot histogram of uncensored data
        counts, bins, _ = ax.hist(uncensored_data, bins=10, density=True, alpha=0.6,
                                 label=f'Uncensored Data (n={len(uncensored_data)})')
        
        # Add vertical line for censoring point
        ax.axvline(x=censoring_point, color='red', linestyle='--', 
                  label=f'Censoring Point ({censoring_point}mm)')
        
        # Add text annotation for censored data
        ax.text(censoring_point + 1, max(counts) * 0.8, 
                f'Censored\nData\n(n={n_censored})', 
                color='red', fontsize=10)
        
    elif is_grouped:
        # For grouped data, plot the histogram with provided bins and counts
        bin_edges = [b[0] for b in bins] + [bins[-1][1]]  # Get bin edges
        bin_centers = [(b[0] + b[1])/2 for b in bins]  # Get bin centers
        
        # Create histogram
        ax.bar(bin_centers, bin_counts, width=[(b[1]-b[0]) for b in bins], alpha=0.6, 
              edgecolor='black', align='center', label='Observed Frequencies')
        
    else:
        # For regular data, plot histogram
        ax.hist(data, bins=10, density=True, alpha=0.6, label='Data')
    
    # If we have mean and std, plot the fitted normal distribution
    if mean is not None and std is not None:
        # Create x values for plotting the distribution
        if is_grouped:
            # For grouped data, use bin edges for x range
            x = np.linspace(min(bin_edges), max(bin_edges), 1000)
        else:
            x = np.linspace(min(data) - 3*std, max(data) + 3*std, 1000)
        
        # Plot the fitted normal PDF
        ax.plot(x, norm.pdf(x, mean, std), 'r-', lw=2, 
               label=f'Fitted Normal: μ={mean:.2f}, σ={std:.2f}')
    
    ax.set_title(title)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density/Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return fig

def censored_neg_log_likelihood(params, data, censoring_point):
    """Negative log-likelihood function for censored normal data."""
    mu, sigma = params
    
    # Handle numerical issues
    if sigma <= 0:
        return 1e10  # Return a very large number if sigma is not positive
    
    # Initialize log-likelihood
    ll = 0
    
    # Add contribution from uncensored observations (PDF)
    for x in data:
        if x < censoring_point:
            ll += norm.logpdf(x, mu, sigma)
        else:
            # Add contribution from censored observations (1-CDF)
            ll += np.log(1 - norm.cdf(censoring_point, mu, sigma))
    
    return -ll  # Return negative log-likelihood for minimization

def grouped_neg_log_likelihood(params, bins, counts):
    """Negative log-likelihood function for grouped normal data."""
    mu, sigma = params
    
    # Handle numerical issues
    if sigma <= 0:
        return 1e10
    
    # Initialize log-likelihood
    ll = 0
    
    # For each bin, calculate probability of falling in that bin
    for (bin_min, bin_max), count in zip(bins, counts):
        # Probability of being in this bin under normal(mu, sigma)
        bin_prob = norm.cdf(bin_max, mu, sigma) - norm.cdf(bin_min, mu, sigma)
        
        # Add to log-likelihood, handling potential numerical issues
        if bin_prob > 0:
            ll += count * np.log(bin_prob)
        else:
            # If probability is too small, add a large negative number
            ll += count * -1e10
    
    return -ll  # Return negative log-likelihood for minimization

def analyze_rainfall_censored(save_dir=None):
    """Example 9: Rainfall Measurements (Censored Data)"""
    print("\n" + "="*70)
    print("Example 9: Rainfall Measurements (Censored Data)")
    print("="*70)
    
    # Problem statement
    print("\nProblem Statement:")
    print("A meteorologist is analyzing rainfall data, but their measurement instrument")
    print("can only record values up to 25mm; any rainfall above this is simply")
    print("recorded as \"25+mm\" (right-censored data).")
    
    # Data
    # Create the full data (we'll convert to censored data for display)
    data_uncensored = [12.3, 8.7, 27.5, 15.2, 10.8, 28.1, 18.4, 7.2, 14.9, 20.1, 31.2, 11.5, 16.8, 9.3, 26.4]
    
    # Convert to censored data for analysis
    censoring_point = 25.0
    data_censored = [min(x, censoring_point) for x in data_uncensored]
    
    # Display the data
    print("\nStep 1: Gather the Data")
    print(f"- Observed rainfall measurements (mm): {data_censored}")
    print(f"- Number of observations: {len(data_censored)}")
    censored_count = sum(1 for x in data_censored if x >= censoring_point)
    print(f"- Number of censored observations (25+mm): {censored_count}")
    print(f"- Number of uncensored observations: {len(data_censored) - censored_count}")
    print(f"- Censoring point: {censoring_point}mm")
    
    # Step 2: Naive approach (ignoring censoring)
    print("\nStep 2: Naive Approach (Incorrectly Ignoring Censoring)")
    naive_mean = np.mean(data_censored)
    naive_std = np.std(data_censored, ddof=0)  # MLE estimate uses ddof=0
    
    print(f"- Naive MLE for mean (treating censored values as exact): {naive_mean:.2f}mm")
    print(f"- Naive MLE for standard deviation: {naive_std:.2f}mm")
    print("- This approach is biased because it treats censored values as exact measurements")
    
    # Step 3: Proper maximum likelihood for censored data
    print("\nStep 3: Maximum Likelihood Estimation with Censored Data")
    print("- For censored normal data, we need to account for both:")
    print("  * Exact values for uncensored observations (using PDF)")
    print("  * Only knowing values exceed the censoring point (using 1-CDF)")
    
    # Initial guesses based on uncensored data only
    uncensored_only = [x for x in data_censored if x < censoring_point]
    initial_mean = np.mean(uncensored_only)
    initial_std = np.std(uncensored_only, ddof=0)
    
    print(f"- Initial guess using only uncensored data: μ = {initial_mean:.2f}, σ = {initial_std:.2f}")
    
    # Use numerical optimization to find MLE for censored data
    result = minimize(
        censored_neg_log_likelihood,
        [initial_mean, initial_std],
        args=(data_censored, censoring_point),
        method='Nelder-Mead'
    )
    
    # Extract the MLE estimates
    mle_mean, mle_std = result.x
    
    print(f"- MLE for mean with censored data: {mle_mean:.2f}mm")
    print(f"- MLE for standard deviation with censored data: {mle_std:.2f}mm")
    
    # Calculate the difference between naive and proper methods
    mean_diff_percent = ((mle_mean / naive_mean) - 1) * 100
    std_diff_percent = ((mle_std / naive_std) - 1) * 100
    
    print("\nStep 4: Compare Naive vs. Proper MLE")
    print(f"- Mean estimate increased by {mean_diff_percent:.1f}% when properly accounting for censoring")
    print(f"- Standard deviation estimate increased by {std_diff_percent:.1f}% when properly accounting for censoring")
    
    # Step 5: Interpretation
    print("\nStep 5: Interpretation")
    print(f"- The MLE for the mean rainfall is {mle_mean:.2f}mm")
    print(f"- The MLE for the standard deviation is {mle_std:.2f}mm")
    
    # Calculate the probability of exceeding the censoring point
    exceed_prob = 1 - norm.cdf(censoring_point, mle_mean, mle_std)
    print(f"- According to our model, {exceed_prob*100:.1f}% of rainfall days exceed the 25mm measurement threshold")
    
    # Plot the results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "normal_mle_rainfall_censored.png")
    else:
        save_path = None
    
    plot_mle_result(data_censored, "Rainfall Measurements (Censored Data) MLE",
                   mean=mle_mean, std=mle_std, 
                   is_censored=True, censoring_point=censoring_point,
                   save_path=save_path)
    
    return {"mean": mle_mean, "std": mle_std, "path": save_path}

def analyze_plant_height_grouped(save_dir=None):
    """Example 10: Plant Height Study (Grouped Data)"""
    print("\n" + "="*70)
    print("Example 10: Plant Height Study (Grouped Data)")
    print("="*70)
    
    # Problem statement
    print("\nProblem Statement:")
    print("A botanist is studying the heights of a particular plant species. Due to time")
    print("constraints, instead of measuring each plant individually, the botanist grouped")
    print("them into height ranges and counted how many fell into each range.")
    
    # Data
    # Define the bins (height ranges) and counts
    bins = [
        (10.0, 12.0),
        (12.1, 14.0),
        (14.1, 16.0),
        (16.1, 18.0),
        (18.1, 20.0),
        (20.1, 22.0)
    ]
    
    bin_counts = [6, 12, 25, 32, 18, 7]
    total_count = sum(bin_counts)
    
    # Display the data
    print("\nStep 1: Gather the Data")
    print("Height ranges and frequencies:")
    for (low, high), count in zip(bins, bin_counts):
        print(f"- {low} - {high} cm: {count} plants")
    print(f"- Total sample size: {total_count} plants")
    
    # Step 2: Calculate approximate mean and variance directly from grouped data
    print("\nStep 2: Calculate Approximate Statistics from Grouped Data")
    
    # Calculate midpoints of bins
    midpoints = [(low + high)/2 for low, high in bins]
    
    # Calculate approximate mean using midpoints
    approx_mean = sum(mid * count for mid, count in zip(midpoints, bin_counts)) / total_count
    
    # Calculate approximate variance using midpoints
    approx_var = sum(count * (mid - approx_mean)**2 for mid, count in zip(midpoints, bin_counts)) / total_count
    approx_std = np.sqrt(approx_var)
    
    print(f"- Approximate mean using midpoints: {approx_mean:.2f}cm")
    print(f"- Approximate standard deviation using midpoints: {approx_std:.2f}cm")
    print("- Note: This is just a rough approximation assuming uniform distribution within each bin")
    
    # Step 3: Maximum likelihood estimation for grouped data
    print("\nStep 3: Maximum Likelihood Estimation with Grouped Data")
    print("- For grouped normal data, we use the probabilities of values falling into each bin")
    print("- The likelihood function becomes a product of terms: [F(upper_bound) - F(lower_bound)]^count")
    print("- Where F is the cumulative distribution function (CDF) of the normal distribution")
    
    # Use numerical optimization to find MLE for grouped data
    result = minimize(
        grouped_neg_log_likelihood,
        [approx_mean, approx_std],  # Initial guess
        args=(bins, bin_counts),
        method='Nelder-Mead'
    )
    
    # Extract the MLE estimates
    mle_mean, mle_std = result.x
    
    print(f"- MLE for mean with grouped data: {mle_mean:.2f}cm")
    print(f"- MLE for standard deviation with grouped data: {mle_std:.2f}cm")
    
    # Step 4: Calculate expected frequencies
    print("\nStep 4: Compare Observed vs. Expected Frequencies")
    
    expected_counts = []
    for bin_range in bins:
        low, high = bin_range
        # Probability of falling in this bin under the MLE normal distribution
        prob = norm.cdf(high, mle_mean, mle_std) - norm.cdf(low, mle_mean, mle_std)
        # Expected count
        expected = prob * total_count
        expected_counts.append(expected)
    
    # Display the comparison
    print("| Height Range (cm) | Observed | Expected |")
    print("|-------------------|----------|----------|")
    for i, ((low, high), obs, exp) in enumerate(zip(bins, bin_counts, expected_counts)):
        print(f"| {low} - {high} | {obs} | {exp:.1f} |")
    
    # Calculate goodness of fit
    chi_square = sum((obs - exp)**2 / exp for obs, exp in zip(bin_counts, expected_counts))
    print(f"\n- Chi-square statistic: {chi_square:.2f}")
    print(f"- Degrees of freedom: {len(bins) - 2 - 1}")  # bins - 2 parameters - 1
    
    # Step 5: Interpretation
    print("\nStep 5: Interpretation")
    print(f"- The MLE for the mean plant height is {mle_mean:.2f}cm")
    print(f"- The MLE for the standard deviation is {mle_std:.2f}cm")
    print(f"- About 68% of plants have heights between {mle_mean - mle_std:.1f}cm and {mle_mean + mle_std:.1f}cm (μ ± σ)")
    print(f"- The distribution appears to be reasonably normal based on the comparison between observed and expected frequencies")
    
    # Plot the results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "normal_mle_plant_height_grouped.png")
    else:
        save_path = None
    
    plot_mle_result(None, "Plant Height Study (Grouped Data) MLE",
                   mean=mle_mean, std=mle_std, 
                   is_grouped=True, bins=bins, bin_counts=bin_counts,
                   save_path=save_path)
    
    return {"mean": mle_mean, "std": mle_std, "path": save_path}

def analyze_student_heights(save_dir=None):
    """Example 11: Height of Students (Simple Sample Mean)"""
    print("\n" + "="*70)
    print("Example 11: Height of Students (Simple Sample Mean)")
    print("="*70)
    
    # Problem statement
    print("\nProblem Statement:")
    print("A researcher is collecting data on the heights of students in a class.")
    print("They measured the heights of 5 randomly selected students (in cm):")
    print("165, 172, 168, 175, and 170.")
    
    # Data
    heights = [165, 172, 168, 175, 170]  # in cm
    
    # Step 1: Gather the data
    print("\nStep 1: Gather the Data")
    print(f"- Heights: {heights} cm")
    print(f"- Number of observations (n): {len(heights)}")
    
    # Step 2: Calculate MLE for mean
    print("\nStep 2: Calculate MLE for Mean")
    print("- For normally distributed data, the MLE for the mean is the sample mean:")
    print("  μ̂_MLE = (1/n) * ∑(x_i)")
    
    # Calculate sum of observations
    heights_sum = sum(heights)
    n = len(heights)
    
    print(f"- Sum of all observations: {' + '.join([str(x) for x in heights])} = {heights_sum} cm")
    
    # Calculate the MLE for mean
    mle_mean = heights_sum / n
    print(f"- μ̂_MLE = {heights_sum} / {n} = {mle_mean} cm")
    
    # Step 3: Interpret the result
    print("\nStep 3: Interpret the Result")
    print(f"- The MLE for the mean height is {mle_mean} cm")
    print("- This represents our best estimate of the true average height")
    print("  of all students in the class based on our sample")
    
    # Optional: Calculate the MLE for variance and standard deviation
    deviations = [x - mle_mean for x in heights]
    squared_devs = [dev**2 for dev in deviations]
    mle_var = sum(squared_devs) / n
    mle_std = np.sqrt(mle_var)
    
    print("\nAdditional Analysis: MLE for Variance and Standard Deviation")
    print(f"- MLE for variance: {mle_var:.2f} cm²")
    print(f"- MLE for standard deviation: {mle_std:.2f} cm")
    
    # Plot the results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "normal_mle_student_heights.png")
    else:
        save_path = None
    
    plot_mle_result(heights, "Student Heights MLE Example",
                   mean=mle_mean, std=mle_std, save_path=save_path)
    
    return {"mean": mle_mean, "std": mle_std, "path": save_path}

def analyze_weight_measurements(save_dir=None):
    """Example 12: Weight Measurements (Known Mean, Unknown Variance)"""
    print("\n" + "="*70)
    print("Example 12: Weight Measurements (Known Mean, Unknown Variance)")
    print("="*70)
    
    # Problem statement
    print("\nProblem Statement:")
    print("A nutritionist is studying weight fluctuations in a clinical trial.")
    print("Based on extensive prior research, they know the true mean weight")
    print("of the population is 68 kg. They collect 6 weight measurements (in kg):")
    print("65, 70, 67, 71, 66, and 69.")
    
    # Data
    weights = [65, 70, 67, 71, 66, 69]  # in kg
    known_mean = 68  # in kg
    
    # Step 1: Gather the data
    print("\nStep 1: Gather the Data")
    print(f"- Weights: {weights} kg")
    print(f"- Number of observations (n): {len(weights)}")
    print(f"- Known mean (μ): {known_mean} kg")
    
    # Step 2: Calculate MLE for variance with known mean
    print("\nStep 2: Calculate MLE for Variance with Known Mean")
    print("- When the mean is known, the MLE for variance is:")
    print("  σ²_MLE = (1/n) * ∑(x_i - μ)²")
    
    # Calculate squared deviations from known mean
    deviations = [x - known_mean for x in weights]
    squared_devs = [dev**2 for dev in deviations]
    
    # Print each deviation and squared deviation
    print("- Calculating squared deviations from known mean:")
    for i, (x, dev, sq_dev) in enumerate(zip(weights, deviations, squared_devs)):
        print(f"  ({x} - {known_mean})² = ({dev})² = {sq_dev}")
    
    # Sum of squared deviations
    sum_sq_devs = sum(squared_devs)
    print(f"- Sum of squared deviations: {' + '.join([str(sq) for sq in squared_devs])} = {sum_sq_devs}")
    
    # Calculate the MLE for variance
    n = len(weights)
    mle_var = sum_sq_devs / n
    print(f"- σ²_MLE = {sum_sq_devs} / {n} = {mle_var:.2f} kg²")
    
    # Calculate the MLE for standard deviation
    mle_std = np.sqrt(mle_var)
    print(f"- σ_MLE = √{mle_var:.2f} = {mle_std:.2f} kg")
    
    # Step 3: Interpret the result
    print("\nStep 3: Interpret the Result")
    print(f"- The MLE for the variance is {mle_var:.2f} kg²")
    print(f"- The MLE for the standard deviation is {mle_std:.2f} kg")
    print("- This represents the estimated variability in weights when the true mean is known")
    
    # Plot the results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "normal_mle_weight_measurements.png")
    else:
        save_path = None
    
    plot_mle_result(weights, "Weight Measurements MLE Example (Known Mean)",
                   mean=known_mean, std=mle_std, save_path=save_path)
    
    return {"mean": known_mean, "std": mle_std, "path": save_path}

def run_all_examples(save_dir=None):
    """Run all examples and return their results."""
    results = {}
    
    results["Rainfall Measurements"] = analyze_rainfall_censored(save_dir)
    results["Plant Height Study"] = analyze_plant_height_grouped(save_dir)
    results["Student Heights"] = analyze_student_heights(save_dir)
    results["Weight Measurements"] = analyze_weight_measurements(save_dir)
    
    return results

if __name__ == "__main__":
    # Use relative path for save directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level and then to Images
    save_dir = os.path.join(os.path.dirname(script_dir), "Images")
    run_all_examples(save_dir) 