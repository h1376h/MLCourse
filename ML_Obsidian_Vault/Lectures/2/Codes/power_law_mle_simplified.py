import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy import stats

def power_law_cdf(x, theta, upper_bound):
    """
    Calculate CDF for power law distribution.
    F(x|θ) = (x^θ)/(upper_bound^θ) for 0 ≤ x < upper_bound
    F(x|θ) = 1 for x ≥ upper_bound
    F(x|θ) = 0 for x < 0
    """
    if np.isscalar(x):
        if x < 0:
            return 0
        elif x >= upper_bound:
            return 1
        else:
            return (x**theta) / (upper_bound**theta)
    else:
        # Handle array input
        result = np.zeros_like(x, dtype=float)
        result[x >= upper_bound] = 1.0
        valid_indices = (x >= 0) & (x < upper_bound)
        result[valid_indices] = (x[valid_indices]**theta) / (upper_bound**theta)
        return result

def power_law_quantile(p, theta, upper_bound):
    """Calculate quantile (inverse CDF) for power law distribution."""
    return upper_bound * (p**(1/theta))

def power_law_mean(theta, upper_bound):
    """Calculate mean of power law distribution."""
    if theta <= 1:
        return "undefined (θ ≤ 1)"
    return (theta * upper_bound**theta) / (theta + 1)

def detailed_power_law_mle(data, upper_bound):
    """Calculate MLE for power law distribution using the detailed method."""
    n = len(data)
    sum_log_x = np.sum(np.log(data))
    
    # MLE formula derived from setting derivative of log-likelihood to zero
    theta_mle = n / (n * np.log(upper_bound) - sum_log_x)
    
    return theta_mle

def simple_power_law_mle(data, upper_bound):
    """Calculate MLE for power law distribution using a simpler method."""
    # Calculate mean of log(upper_bound/x)
    mean_log_ratio = np.mean(np.log(upper_bound/np.array(data)))
    
    # Simple formula: θ = 1/mean_log_ratio
    theta_mle = 1/mean_log_ratio
    
    return theta_mle

def numerical_power_law_mle(data, upper_bound):
    """Calculate MLE using numerical optimization."""
    def neg_log_likelihood(theta):
        if theta <= 0:
            return float('inf')
        return -np.sum(np.log(theta) + (theta-1)*np.log(data) - theta*np.log(upper_bound))
    
    result = minimize_scalar(neg_log_likelihood, bounds=(0.1, 10), method='bounded')
    return result.x

def power_law_hypothesis_test(data, null_theta, upper_bound):
    """Perform a likelihood ratio test for power law distribution."""
    # Calculate MLE
    mle_theta = detailed_power_law_mle(data, upper_bound)
    
    # Calculate log-likelihood under null hypothesis
    n = len(data)
    sum_log_x = np.sum(np.log(data))
    ll_null = n * np.log(null_theta) + (null_theta - 1) * sum_log_x - n * null_theta * np.log(upper_bound)
    
    # Calculate log-likelihood under alternative (MLE)
    ll_mle = n * np.log(mle_theta) + (mle_theta - 1) * sum_log_x - n * mle_theta * np.log(upper_bound)
    
    # Calculate likelihood ratio test statistic
    lr = 2 * (ll_mle - ll_null)
    
    # Calculate p-value using chi-square (1 df for 1 parameter)
    p_value = 1 - stats.chi2.cdf(lr, df=1)
    
    # Make decision
    alpha = 0.05
    if p_value < alpha:
        decision = "Reject null hypothesis"
    else:
        decision = "Fail to reject null hypothesis"
    
    return {
        "mle_theta": mle_theta,
        "lr_statistic": lr,
        "p_value": p_value,
        "decision": decision
    }

def power_law_prediction(data, threshold, upper_bound):
    """Calculate prediction probabilities based on fitted power law model."""
    # Estimate θ using MLE
    mle_theta = detailed_power_law_mle(data, upper_bound)
    
    # Calculate probability P(X > threshold)
    prob_exceed = 1 - power_law_cdf(threshold, mle_theta, upper_bound)
    
    # Calculate median
    median = power_law_quantile(0.5, mle_theta, upper_bound)
    
    # Calculate mean (if defined)
    mean = power_law_mean(mle_theta, upper_bound)
    
    # Calculate percentile of threshold
    percentile = power_law_cdf(threshold, mle_theta, upper_bound) * 100
    
    return {
        "theta": mle_theta,
        "prob_exceed": prob_exceed,
        "median": median,
        "mean": mean,
        "percentile": percentile
    }

def analyze_basic_mle_example(name, data, upper_bound):
    """Analyze a basic MLE estimation problem for power law data."""
    print(f"\n{'='*60}")
    print(f"TYPE 1: BASIC MLE ESTIMATION - {name}")
    print(f"{'='*60}")
    print(f"Data: {data}")
    print(f"Number of observations: {len(data)}")
    print(f"Upper bound: {upper_bound}")
    
    # Method 1: Detailed MLE
    theta_detailed = detailed_power_law_mle(data, upper_bound)
    print(f"\nMethod 1 - Detailed MLE:")
    print(f"θ = {theta_detailed:.4f}")
    print("Formula: θ = n / (n*ln(upper_bound) - sum(ln(x_i)))")
    
    # Method 2: Simple MLE
    theta_simple = simple_power_law_mle(data, upper_bound)
    print(f"\nMethod 2 - Simple MLE (Quick Formula for Exams!):")
    print(f"θ = {theta_simple:.4f}")
    print("Formula: θ = 1/mean(ln(upper_bound/x_i))")
    
    # Method 3: Numerical MLE (verification)
    theta_numerical = numerical_power_law_mle(data, upper_bound)
    print(f"\nMethod 3 - Numerical MLE (Verification):")
    print(f"θ = {theta_numerical:.4f}")
    
    # Calculate bootstrap confidence interval
    n_bootstrap = 1000
    bootstrap_thetas = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_theta = detailed_power_law_mle(bootstrap_sample, upper_bound)
        bootstrap_thetas.append(bootstrap_theta)
    
    ci_lower = np.percentile(bootstrap_thetas, 2.5)
    ci_upper = np.percentile(bootstrap_thetas, 97.5)
    print(f"\nConfidence Interval:")
    print(f"95% Bootstrap CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Compare results
    print("\nComparison of Methods:")
    print(f"{'Method':<20} {'θ':<10} {'Difference from Numerical':<25}")
    print("-"*55)
    print(f"{'Detailed':<20} {theta_detailed:<10.4f} {abs(theta_detailed-theta_numerical):<25.6f}")
    print(f"{'Simple':<20} {theta_simple:<10.4f} {abs(theta_simple-theta_numerical):<25.6f}")
    print(f"{'Numerical':<20} {theta_numerical:<10.4f} {'0':>25}")
    
    return theta_detailed

def analyze_hypothesis_testing_example(name, data, null_theta, upper_bound):
    """Analyze a hypothesis testing problem for power law data."""
    print(f"\n{'='*60}")
    print(f"TYPE 2: HYPOTHESIS TESTING - {name}")
    print(f"{'='*60}")
    print(f"Data: {data}")
    print(f"Number of observations: {len(data)}")
    print(f"Upper bound: {upper_bound}")
    print(f"Null hypothesis: H₀: θ = {null_theta}")
    print(f"Alternative hypothesis: H₁: θ ≠ {null_theta}")
    
    # Calculate MLE
    mle_theta = detailed_power_law_mle(data, upper_bound)
    print(f"\nMLE estimate of θ: {mle_theta:.4f}")
    
    # Perform hypothesis test
    test_result = power_law_hypothesis_test(data, null_theta, upper_bound)
    
    print(f"\nLikelihood Ratio Test:")
    print(f"LR statistic: {test_result['lr_statistic']:.4f}")
    print(f"P-value: {test_result['p_value']:.4f}")
    print(f"\nDecision (at α = 0.05): {test_result['decision']}")
    
    # Simple explanation
    print("\nSimplified Method for Hypothesis Testing:")
    print("1. Calculate θ̂ using MLE")
    print("2. Calculate log-likelihood under H₀: L₀ = n*ln(θ₀) + (θ₀-1)*sum(ln(x_i)) - n*θ₀*ln(upper_bound)")
    print("3. Calculate log-likelihood under H₁: L₁ = n*ln(θ̂) + (θ̂-1)*sum(ln(x_i)) - n*θ̂*ln(upper_bound)")
    print("4. Calculate LR = 2*(L₁ - L₀)")
    print("5. Calculate p-value from chi-squared distribution with 1 df")
    print("6. Reject H₀ if p-value < α")
    
    return test_result

def analyze_prediction_example(name, data, threshold, upper_bound):
    """Analyze a prediction problem for power law data."""
    print(f"\n{'='*60}")
    print(f"TYPE 3: PREDICTION PROBLEM - {name}")
    print(f"{'='*60}")
    print(f"Data: {data}")
    print(f"Number of observations: {len(data)}")
    print(f"Upper bound: {upper_bound}")
    print(f"Prediction threshold: {threshold}")
    
    # Make predictions
    pred_result = power_law_prediction(data, threshold, upper_bound)
    
    print(f"\nMLE estimate of θ: {pred_result['theta']:.4f}")
    print(f"\nPrediction Results:")
    print(f"P(X > {threshold}) = {pred_result['prob_exceed']:.4f} or {pred_result['prob_exceed']*100:.2f}%")
    print(f"Median value: {pred_result['median']:.4f}")
    print(f"Mean value: {pred_result['mean'] if isinstance(pred_result['mean'], str) else f'{pred_result['mean']:.4f}'}")
    print(f"The value {threshold} is at the {pred_result['percentile']:.2f}th percentile")
    
    # Simulate future observations
    np.random.seed(42)  # for reproducibility
    future_samples = []
    for _ in range(5):
        u = np.random.uniform(0, 1)
        sample = power_law_quantile(u, pred_result['theta'], upper_bound)
        future_samples.append(sample)
    
    print(f"\nSimulated future observations:")
    print(future_samples)
    
    # Simple explanation
    print("\nSimplified Method for Prediction Problems:")
    print("1. Calculate θ̂ using MLE (θ̂ = n / (n*ln(upper_bound) - sum(ln(x_i))))")
    print("2. Calculate P(X > threshold) = 1 - (threshold^θ̂)/(upper_bound^θ̂)")
    print("3. For median: x_median = upper_bound * (0.5)^(1/θ̂)")
    print("4. For mean (if θ̂ > 1): E[X] = (θ̂ * upper_bound^θ̂) / (θ̂ + 1)")
    
    return pred_result

def main():
    print("\n===== POWER LAW DISTRIBUTION: THREE TYPES OF PROBLEMS =====")
    print("This script demonstrates simplified methods for solving different types")
    print("of problems involving power law distributions.")
    
    # Example 1: Basic MLE Estimation - Social Media Followers
    followers_data = [0.3, 0.5, 0.8, 0.2, 0.6, 0.9]  # followers in millions
    print("\nSOCIAL MEDIA FOLLOWERS EXAMPLE")
    print("A data scientist is studying the distribution of social media influencer follower counts.")
    analyze_basic_mle_example("Social Media Followers", followers_data, upper_bound=1)
    
    # Example 2: Hypothesis Testing - Earthquake Magnitudes
    earthquake_data = [2.1, 3.4, 2.8, 3.0, 3.7, 2.5, 3.2]  # magnitudes on Richter scale
    print("\nEARTHQUAKE MAGNITUDES EXAMPLE")
    print("A seismologist is testing whether earthquake magnitudes follow a specific power law.")
    analyze_hypothesis_testing_example("Earthquake Magnitudes", earthquake_data, null_theta=2.0, upper_bound=4)
    
    # Example 3: Prediction Problem - Internet Traffic
    traffic_data = [0.5, 1.2, 1.8, 0.3, 0.7, 1.5, 0.9, 1.1]  # traffic volume in GB
    print("\nINTERNET TRAFFIC EXAMPLE")
    print("A network engineer wants to predict the probability of extreme traffic volumes.")
    analyze_prediction_example("Internet Traffic", traffic_data, threshold=1.5, upper_bound=2)

if __name__ == "__main__":
    main()