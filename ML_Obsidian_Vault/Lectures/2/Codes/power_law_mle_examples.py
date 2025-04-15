import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import os
from scipy import stats

def power_law_pdf(x, theta, upper_bound=3):
    """Probability density function for our custom power law"""
    if theta <= 0:
        return np.zeros_like(x)
    
    # Create a mask for valid x values (0 <= x < upper_bound)
    mask = (x >= 0) & (x < upper_bound)
    
    # Initialize result array with zeros
    result = np.zeros_like(x, dtype=float)
    
    # Calculate PDF for valid x values
    valid_x = x[mask]
    result[mask] = (theta * valid_x**(theta-1)) / (upper_bound**theta)
    
    return result

def power_law_cdf(x, theta, upper_bound=3):
    """Cumulative distribution function for our custom power law"""
    if theta <= 0:
        return 0 if np.isscalar(x) else np.zeros_like(x)
    
    # Handle scalar input
    if np.isscalar(x):
        if x < 0:
            return 0
        elif x >= upper_bound:
            return 1
        else:
            return (x**theta) / (upper_bound**theta)
    
    # Handle array input
    # Create a mask for valid x values
    mask = (x >= 0)
    
    # Initialize result array with zeros
    result = np.zeros_like(x, dtype=float)
    
    # Values above upper_bound have CDF = 1
    result[x >= upper_bound] = 1.0
    
    # Calculate CDF for valid x values below upper_bound
    valid_indices = mask & (x < upper_bound)
    valid_x = x[valid_indices]
    result[valid_indices] = (valid_x**theta) / (upper_bound**theta)
    
    return result

def power_law_quantile(p, theta, upper_bound=3):
    """Quantile function (inverse CDF) for our custom power law"""
    return upper_bound * (p**(1/theta))

def power_law_sample(n, theta, upper_bound=3, random_state=None):
    """Generate n random samples from power law distribution"""
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate uniform random numbers
    u = np.random.uniform(0, 1, n)
    
    # Transform to power law using inverse CDF
    samples = power_law_quantile(u, theta, upper_bound)
    
    return samples

def log_likelihood(theta, data, upper_bound=3):
    """Log-likelihood function for power law distribution"""
    # Check if theta is in valid range (must be positive)
    if theta <= 0:
        return float('-inf')
    
    # Convert data to numpy array if it's not already
    data = np.array(data)
    
    # Check if all data is within bounds
    if np.any(data < 0) or np.any(data >= upper_bound):
        return float('-inf')
    
    n = len(data)
    sum_log_x = np.sum(np.log(data))
    
    # Log-likelihood formula
    ll = n * np.log(theta) + (theta - 1) * sum_log_x - n * theta * np.log(upper_bound)
    
    return ll

def plot_power_law_likelihood(data, title, save_path=None, upper_bound=3):
    """Plot the likelihood function for power law data and highlight the MLE."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate MLE estimate analytically
    n = len(data)
    sum_log_x = np.sum(np.log(data))
    mle_theta = n / (n * np.log(upper_bound) - sum_log_x)
    
    # Create a range of possible theta values to plot
    possible_thetas = np.linspace(max(0.1, mle_theta - 1.5), mle_theta + 1.5, 1000)
    
    # Calculate the log-likelihood for each possible theta
    log_likelihoods = []
    for theta in possible_thetas:
        ll = log_likelihood(theta, data, upper_bound)
        log_likelihoods.append(ll)
    
    # Normalize the log-likelihood for better visualization
    log_likelihoods = np.array(log_likelihoods)
    log_likelihoods = log_likelihoods - np.min(log_likelihoods)
    log_likelihoods = log_likelihoods / np.max(log_likelihoods)
    
    # Plot the log-likelihood function
    ax.plot(possible_thetas, log_likelihoods, 'b-', linewidth=2)
    ax.axvline(x=mle_theta, color='r', linestyle='--', 
              label=f'MLE θ = {mle_theta:.2f}')
    
    ax.set_title(f"{title} - Log-Likelihood Function")
    ax.set_xlabel('θ (Shape Parameter)')
    ax.set_ylabel('Normalized Log-Likelihood')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return mle_theta

def plot_power_law_distribution(data, title, mle_theta, save_path=None, upper_bound=3, theta_ref=1.5):
    """Plot the data and the estimated power law distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate x values for plotting
    x = np.linspace(0.01, upper_bound, 1000)
    y = power_law_pdf(x, mle_theta, upper_bound)
    
    # Plot histogram of the data
    ax.hist(data, bins=10, density=True, alpha=0.5, color='blue', 
             label='Observed Data')
    
    # Plot the estimated PDF
    ax.plot(x, y, 'r-', linewidth=2, 
            label=f'Estimated Power Law (θ = {mle_theta:.2f})')
    
    # Add theoretical PDF with a reference theta for comparison
    y_theory = power_law_pdf(x, theta_ref, upper_bound)
    ax.plot(x, y_theory, 'g--', linewidth=2, 
            label=f'Reference Power Law (θ = {theta_ref:.2f})')
    
    # Mark the observed data points
    ax.plot(data, np.zeros_like(data), 'bo', markersize=8, alpha=0.6)
    
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'MLE for Power Law Distribution - {title}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, upper_bound)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_hypothesis_test(data, title, null_theta, save_path=None, upper_bound=3):
    """Plot visualization for hypothesis testing problem."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate MLE estimate
    n = len(data)
    sum_log_x = np.sum(np.log(data))
    mle_theta = n / (n * np.log(upper_bound) - sum_log_x)
    
    # Generate x values for plotting
    x = np.linspace(0.01, upper_bound, 1000)
    
    # Plot histogram of the data
    ax.hist(data, bins=10, density=True, alpha=0.5, color='blue', 
             label='Observed Data')
    
    # Plot the null hypothesis PDF
    y_null = power_law_pdf(x, null_theta, upper_bound)
    ax.plot(x, y_null, 'r-', linewidth=2, 
            label=f'Null Hypothesis (θ = {null_theta:.2f})')
    
    # Plot the MLE PDF
    y_mle = power_law_pdf(x, mle_theta, upper_bound)
    ax.plot(x, y_mle, 'g--', linewidth=2, 
            label=f'MLE Estimate (θ = {mle_theta:.2f})')
    
    # Mark the observed data points
    ax.plot(data, np.zeros_like(data), 'bo', markersize=8, alpha=0.6)
    
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Hypothesis Testing - {title}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, upper_bound)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return mle_theta

def plot_prediction_problem(data, title, mle_theta, prediction_value, save_path=None, upper_bound=3):
    """Plot visualization for prediction problem."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate x values for plotting
    x = np.linspace(0.01, upper_bound, 1000)
    y = power_law_pdf(x, mle_theta, upper_bound)
    
    # Calculate probability P(X > prediction_value)
    prob_exceed = 1 - power_law_cdf(prediction_value, mle_theta, upper_bound)
    
    # Plot histogram of the data
    ax.hist(data, bins=10, density=True, alpha=0.5, color='blue', 
             label='Observed Data')
    
    # Plot the estimated PDF
    ax.plot(x, y, 'r-', linewidth=2, 
            label=f'Estimated Power Law (θ = {mle_theta:.2f})')
    
    # Shade the area representing P(X > prediction_value)
    x_shade = np.linspace(prediction_value, upper_bound, 100)
    y_shade = power_law_pdf(x_shade, mle_theta, upper_bound)
    ax.fill_between(x_shade, y_shade, alpha=0.3, color='orange',
                   label=f'P(X > {prediction_value}) = {prob_exceed:.4f}')
    
    # Draw vertical line at prediction_value
    ax.axvline(x=prediction_value, color='k', linestyle='--')
    
    # Mark the observed data points
    ax.plot(data, np.zeros_like(data), 'bo', markersize=8, alpha=0.6)
    
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Prediction Problem - {title}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, upper_bound)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return prob_exceed

def analyze_basic_mle(name, data, context, save_dir=None, upper_bound=3):
    """Analyze basic MLE estimation problem for power law data."""
    print(f"\n{'='*50}")
    print(f"Type 1: BASIC MLE ESTIMATION - {name}")
    print(f"{'='*50}")
    print(f"Context: {context}")
    
    # Step 1: Data analysis
    print("\nStep 1: Data Analysis")
    print(f"- Data: {data}")
    print(f"- Number of observations: {len(data)}")
    
    # Step 2: Calculate MLE analytically
    print("\nStep 2: Maximum Likelihood Estimation")
    print("- For our power law distribution, MLE of θ needs to be calculated from the formula")
    n = len(data)
    sum_log_x = np.sum(np.log(data))
    mle_theta = n / (n * np.log(upper_bound) - sum_log_x)
    print(f"- Sum of log(x_i): {sum_log_x:.4f}")
    print(f"- n * ln({upper_bound}): {n * np.log(upper_bound):.4f}")
    print(f"- n * ln({upper_bound}) - sum_log_x: {n * np.log(upper_bound) - sum_log_x:.4f}")
    print(f"- MLE theta (θ) = {n} / {n * np.log(upper_bound) - sum_log_x:.4f} = {mle_theta:.4f}")
    
    # Step 3: Verify with numerical optimization
    def neg_log_likelihood(theta, data=data, upper_bound=upper_bound):
        return -log_likelihood(theta, data, upper_bound)
    
    result = minimize_scalar(neg_log_likelihood, method='brent', bracket=(0.5, 2.5))
    theta_numerical = result.x
    print(f"- Numerical optimization result: θ = {theta_numerical:.4f}")
    print(f"- Difference between analytical and numerical: {abs(mle_theta - theta_numerical):.6f}")
    
    # Create save paths if directory is provided
    likelihood_save_path = None
    distribution_save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base_filename = f"basic_mle_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
        likelihood_save_path = os.path.join(save_dir, base_filename + "_likelihood.png")
        distribution_save_path = os.path.join(save_dir, base_filename + ".png")
    
    # Step 4: Visualize likelihood function
    print("\nStep 4: Likelihood Visualization")
    plot_power_law_likelihood(data, name, likelihood_save_path, upper_bound)
    
    # Step 5: Visualize distribution
    print("\nStep 5: Distribution Visualization")
    plot_power_law_distribution(data, name, mle_theta, distribution_save_path, upper_bound)
    
    # Step 6: Confidence Interval
    print("\nStep 6: Confidence Interval for Theta")
    # We'll use bootstrap for confidence interval
    n_bootstrap = 1000
    bootstrap_thetas = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        # Calculate MLE for bootstrap sample
        sum_log_bootstrap = np.sum(np.log(bootstrap_sample))
        bootstrap_theta = n / (n * np.log(upper_bound) - sum_log_bootstrap)
        bootstrap_thetas.append(bootstrap_theta)
    
    # Calculate confidence interval from bootstrap samples
    ci_lower = np.percentile(bootstrap_thetas, 2.5)
    ci_upper = np.percentile(bootstrap_thetas, 97.5)
    print(f"- 95% Bootstrap Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Step 7: Interpretation
    print("\nStep 7: Interpretation")
    print(f"- Based on the observed data alone, the most likely shape parameter θ is {mle_theta:.4f}")
    print(f"- This power law distribution with θ = {mle_theta:.4f} best explains the observed data")
    
    return {"theta": mle_theta, "ci": [ci_lower, ci_upper], "path": distribution_save_path}

def analyze_hypothesis_test(name, data, null_theta, context, save_dir=None, upper_bound=3):
    """Analyze hypothesis testing problem for power law data."""
    print(f"\n{'='*50}")
    print(f"Type 2: HYPOTHESIS TESTING - {name}")
    print(f"{'='*50}")
    print(f"Context: {context}")
    
    # Step 1: Data analysis
    print("\nStep 1: Data Analysis")
    print(f"- Data: {data}")
    print(f"- Number of observations: {len(data)}")
    print(f"- Null hypothesis: θ = {null_theta}")
    
    # Step 2: Calculate likelihood ratio test statistic
    print("\nStep 2: Likelihood Ratio Test")
    # Calculate MLE
    n = len(data)
    sum_log_x = np.sum(np.log(data))
    mle_theta = n / (n * np.log(upper_bound) - sum_log_x)
    
    # Calculate log-likelihood under null hypothesis
    ll_null = log_likelihood(null_theta, data, upper_bound)
    
    # Calculate log-likelihood under alternative (MLE)
    ll_mle = log_likelihood(mle_theta, data, upper_bound)
    
    # Calculate likelihood ratio
    lr = 2 * (ll_mle - ll_null)
    print(f"- Log-likelihood under null (θ = {null_theta}): {ll_null:.4f}")
    print(f"- Log-likelihood under MLE (θ = {mle_theta:.4f}): {ll_mle:.4f}")
    print(f"- Likelihood ratio test statistic: 2(L_MLE - L_null) = {lr:.4f}")
    
    # Step 3: Calculate p-value using chi-square
    # Under H0, LR approximately follows chi-squared with 1 df
    p_value = 1 - stats.chi2.cdf(lr, df=1)
    print(f"- P-value: {p_value:.4f}")
    
    # Step 4: Make decision
    alpha = 0.05
    print(f"- Using significance level α = {alpha}")
    if p_value < alpha:
        decision = "Reject null hypothesis"
        print(f"- Decision: {decision} (p-value < α)")
        print(f"- There is significant evidence against θ = {null_theta}")
    else:
        decision = "Fail to reject null hypothesis"
        print(f"- Decision: {decision} (p-value >= α)")
        print(f"- There is not enough evidence against θ = {null_theta}")
    
    # Create save path if directory is provided
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base_filename = f"hypothesis_test_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
        save_path = os.path.join(save_dir, base_filename + ".png")
    
    # Step 5: Visualize
    print("\nStep 5: Visualization")
    plot_hypothesis_test(data, name, null_theta, save_path, upper_bound)
    
    return {
        "mle_theta": mle_theta, 
        "lr_statistic": lr, 
        "p_value": p_value, 
        "decision": decision,
        "path": save_path
    }

def analyze_prediction_problem(name, data, prediction_value, context, save_dir=None, upper_bound=3):
    """Analyze prediction problem for power law data."""
    print(f"\n{'='*50}")
    print(f"Type 3: PREDICTION PROBLEM - {name}")
    print(f"{'='*50}")
    print(f"Context: {context}")
    
    # Step 1: Data analysis
    print("\nStep 1: Data Analysis")
    print(f"- Data: {data}")
    print(f"- Number of observations: {len(data)}")
    print(f"- Prediction threshold: {prediction_value}")
    
    # Step 2: Calculate MLE
    print("\nStep 2: Maximum Likelihood Estimation")
    n = len(data)
    sum_log_x = np.sum(np.log(data))
    mle_theta = n / (n * np.log(upper_bound) - sum_log_x)
    print(f"- MLE theta (θ) = {mle_theta:.4f}")
    
    # Step 3: Calculate prediction
    print("\nStep 3: Calculate Prediction")
    # Calculate probability P(X > prediction_value)
    prob_exceed = 1 - power_law_cdf(prediction_value, mle_theta, upper_bound)
    print(f"- Probability P(X > {prediction_value}) = {prob_exceed:.4f}")
    
    # Additional calculations for other types of predictions
    median = power_law_quantile(0.5, mle_theta, upper_bound)
    mean = (mle_theta * upper_bound**mle_theta) / (mle_theta + 1) if mle_theta > 1 else "undefined (θ ≤ 1)"
    
    print(f"- Median value: {median:.4f}")
    print(f"- Mean value: {mean}")
    
    # Generate future samples
    print("\nStep 4: Simulate Future Observations")
    future_samples = power_law_sample(5, mle_theta, upper_bound, random_state=42)
    print(f"- 5 simulated future observations: {future_samples}")
    
    # Calculate percentile of prediction_value
    percentile = power_law_cdf(prediction_value, mle_theta, upper_bound) * 100
    print(f"- The value {prediction_value} is at the {percentile:.2f}th percentile")
    
    # Create save path if directory is provided
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base_filename = f"prediction_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
        save_path = os.path.join(save_dir, base_filename + ".png")
    
    # Step 5: Visualization
    print("\nStep 5: Visualization")
    plot_prediction_problem(data, name, mle_theta, prediction_value, save_path, upper_bound)
    
    return {
        "theta": mle_theta,
        "prob_exceed": prob_exceed,
        "median": median,
        "future_samples": future_samples,
        "path": save_path
    }

def generate_power_law_examples(save_dir=None):
    results = {}
    
    print("""
    Let's analyze three different types of problems for power law distributions!
    Each example illustrates a different approach to working with power laws.
    """)

    # Example 1: Basic MLE Estimation - Social Media Followers
    followers_data = [0.3, 0.5, 0.8, 0.2, 0.6, 0.9]  # followers in millions
    followers_context = """
    A data scientist is studying the distribution of social media influencer follower counts.
    - You analyze follower counts (in millions) for 6 influencers in a specific category
    - The distribution follows a power law with PDF f(x|θ) = (θx^(θ-1))/(1^θ) for 0≤x<1
    - Using only the observed data, estimate the shape parameter θ
    - This is a direct application of Maximum Likelihood Estimation
    """
    followers_results = analyze_basic_mle("Social Media Followers", followers_data, followers_context, save_dir, upper_bound=1)
    results["Social Media Followers"] = followers_results

    # Example 2: Hypothesis Testing - Earthquake Magnitudes
    earthquake_data = [2.1, 3.4, 2.8, 3.0, 3.7, 2.5, 3.2]  # magnitudes on Richter scale
    earthquake_context = """
    A seismologist is testing whether earthquake magnitudes follow a specific power law.
    - You analyze the magnitudes (on Richter scale) of 7 recent earthquakes
    - The distribution follows a power law with PDF f(x|θ) = (θx^(θ-1))/(4^θ) for 0≤x<4
    - Test the null hypothesis H₀: θ = 2.0 against the alternative H₁: θ ≠ 2.0
    - This is a statistical hypothesis test using likelihood ratio
    """
    earthquake_results = analyze_hypothesis_test("Earthquake Magnitudes", earthquake_data, null_theta=2.0, 
                                               context=earthquake_context, save_dir=save_dir, upper_bound=4)
    results["Earthquake Magnitudes"] = earthquake_results
    
    # Example 3: Prediction Problem - Internet Traffic
    traffic_data = [0.5, 1.2, 1.8, 0.3, 0.7, 1.5, 0.9, 1.1]  # traffic volume in GB
    traffic_context = """
    A network engineer wants to predict the probability of extreme traffic volumes.
    - You have measured traffic volume (in GB) during 8 time intervals
    - The distribution follows a power law with PDF f(x|θ) = (θx^(θ-1))/(2^θ) for 0≤x<2
    - First estimate the power law parameter θ using MLE
    - Then calculate the probability that future traffic will exceed 1.5 GB
    - This is a practical application of power laws for prediction
    """
    traffic_results = analyze_prediction_problem("Internet Traffic", traffic_data, prediction_value=1.5, 
                                               context=traffic_context, save_dir=save_dir, upper_bound=2)
    results["Internet Traffic"] = traffic_results
    
    return results

if __name__ == "__main__":
    # Use relative path for save directory
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level and then to Images
    save_dir = os.path.join(os.path.dirname(script_dir), "Images")
    generate_power_law_examples(save_dir) 