import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import os

def plot_poisson_likelihood(data, title, save_path=None):
    """Plot the likelihood function for Poisson data and highlight the MLE."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate MLE estimate
    mle_lambda = np.mean(data)
    
    # Create a range of possible lambda values to plot
    max_value = max(data)
    possible_lambdas = np.linspace(max(0.1, mle_lambda - 3), mle_lambda + 3, 1000)
    
    # Calculate the log-likelihood for each possible lambda
    log_likelihood = []
    for lam in possible_lambdas:
        ll = sum(poisson.logpmf(data, lam))
        log_likelihood.append(ll)
    
    # Normalize the log-likelihood for better visualization
    log_likelihood = np.array(log_likelihood)
    log_likelihood = log_likelihood - np.min(log_likelihood)
    log_likelihood = log_likelihood / np.max(log_likelihood)
    
    # Plot the log-likelihood function
    ax.plot(possible_lambdas, log_likelihood, 'b-', linewidth=2)
    ax.axvline(x=mle_lambda, color='r', linestyle='--', 
              label=f'MLE λ = {mle_lambda:.2f}')
    
    ax.set_title(f"{title} - Log-Likelihood Function")
    ax.set_xlabel('λ (Rate Parameter)')
    ax.set_ylabel('Normalized Log-Likelihood')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return mle_lambda

def analyze_poisson_data(name, data, context, save_dir=None):
    """Analyze Poisson data with detailed steps using MLE."""
    print(f"\n{'='*50}")
    print(f"{name} Example")
    print(f"{'='*50}")
    print(f"Context: {context}")
    
    # Step 1: Data analysis
    print("\nStep 1: Data Analysis")
    print(f"- Data: {data}")
    print(f"- Number of observations: {len(data)}")
    
    # Step 2: Calculate MLE
    print("\nStep 2: Maximum Likelihood Estimation")
    print("- For Poisson distribution, MLE of lambda is the sample mean")
    mle_lambda = np.mean(data)
    print(f"- MLE lambda (λ) = {mle_lambda:.2f}")
    
    # Create save path if directory is provided
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"poisson_mle_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        save_path = os.path.join(save_dir, filename)
    
    # Step 3: Visualize and confirm
    print("\nStep 3: Visualization")
    plot_poisson_likelihood(data, name, save_path)
    
    # Step 4: Confidence Interval
    print("\nStep 4: Confidence Interval for Lambda")
    n = len(data)
    # For Poisson, variance equals the mean parameter
    sem = np.sqrt(mle_lambda / n)  # Standard error of lambda
    z = 1.96  # 95% confidence level
    ci_lower = max(0, mle_lambda - z * sem)
    ci_upper = mle_lambda + z * sem
    print(f"- 95% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}]")
    
    # Step 5: Interpretation
    print("\nStep 5: Interpretation")
    print(f"- Based on the observed data alone, the most likely rate is {mle_lambda:.2f}")
    print(f"- This represents the average number of occurrences per unit time/space")
    
    return {"lambda": mle_lambda, "path": save_path}

def generate_poisson_examples(save_dir=None):
    results = {}
    
    print("""
    Let's analyze different scenarios using Maximum Likelihood Estimation for Poisson distributions!
    Each example will show how we can estimate the rate parameter using only the observed data.
    """)

    # Example 1: Customer Service Calls
    call_data = [3, 5, 2, 4, 3, 6, 4, 3]  # calls per hour
    call_context = """
    A data scientist is analyzing the number of customer service calls received per hour.
    - You collected data for 8 hours
    - Using only the observed data (no prior assumptions)
    """
    call_results = analyze_poisson_data("Customer Service Calls", call_data, call_context, save_dir)
    results["Customer Service Calls"] = call_results

    # Example 2: Website Errors
    error_data = [0, 2, 1, 0, 3, 1, 2, 0, 1]  # errors per day
    error_context = """
    A web developer is tracking daily website errors.
    - You recorded errors for 9 days
    - Using only the observed data (no prior assumptions)
    """
    error_results = analyze_poisson_data("Website Errors", error_data, error_context, save_dir)
    results["Website Errors"] = error_results

    # Example 3: Traffic Accidents
    accident_data = [2, 0, 1, 3, 1, 0, 2, 1]  # accidents per week
    accident_context = """
    A traffic analyst is studying weekly accident rates at an intersection.
    - You collected data for 8 weeks
    - Using only the observed data (no prior assumptions)
    """
    accident_results = analyze_poisson_data("Traffic Accidents", accident_data, accident_context, save_dir)
    results["Traffic Accidents"] = accident_results
    
    # Example 4: Email Arrivals
    email_data = [12, 15, 9, 13, 11, 14, 10, 16]  # emails per hour
    email_context = """
    An office worker is analyzing their hourly email volume.
    - You recorded incoming emails for 8 hours
    - Using only the observed data (no prior assumptions)
    """
    email_results = analyze_poisson_data("Email Arrivals", email_data, email_context, save_dir)
    results["Email Arrivals"] = email_results
    
    return results

if __name__ == "__main__":
    # Use relative path for save directory
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level and then to Images
    save_dir = os.path.join(os.path.dirname(script_dir), "Images")
    generate_poisson_examples(save_dir) 