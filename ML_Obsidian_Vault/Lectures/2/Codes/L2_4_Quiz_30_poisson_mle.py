import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from math import factorial
import sympy as sp

def poisson_pmf(y, theta):
    """Probability mass function for the Poisson distribution."""
    if y < 0 or not isinstance(y, (int, np.integer)):
        return 0
    return (theta**y * np.exp(-theta)) / factorial(y)

def log_likelihood(theta, data):
    """Log-likelihood function for Poisson distribution."""
    if theta <= 0:
        return float('-inf')
    
    n = len(data)
    sum_y = np.sum(data)
    
    # Log-likelihood formula for Poisson
    ll = sum_y * np.log(theta) - n * theta - np.sum([np.log(factorial(int(y))) for y in data])
    
    return ll

def plot_poisson_pmf(theta_values, ylim=15, save_path=None):
    """Plot the Poisson PMF for different theta values."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Range of y values to plot
    y_values = np.arange(0, ylim)
    
    # Plot PMF for various theta values
    for theta in theta_values:
        pmf_values = [poisson_pmf(y, theta) for y in y_values]
        ax.plot(y_values, pmf_values, marker='o', linestyle='-', label=f'θ = {theta}')
    
    ax.set_xlabel('y (Count)')
    ax.set_ylabel('Probability Mass P(y|θ)')
    ax.set_title('Poisson Probability Mass Function for Different θ Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_likelihood_surface(data, save_path=None):
    """Plot the likelihood function surface for the Poisson data."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate MLE estimate analytically
    mle_theta = np.mean(data)
    
    # Create a range of possible theta values to plot
    possible_thetas = np.linspace(max(0.1, mle_theta - 1.5), mle_theta + 1.5, 1000)
    
    # Calculate the log-likelihood for each possible theta
    log_likelihoods = [log_likelihood(theta, data) for theta in possible_thetas]
    
    # Plot the log-likelihood function
    ax.plot(possible_thetas, log_likelihoods, 'b-', linewidth=2)
    ax.axvline(x=mle_theta, color='r', linestyle='--', 
              label=f'MLE θ = {mle_theta:.4f}')
    
    ax.set_title("Log-Likelihood Function for Poisson Distribution")
    ax.set_xlabel('θ (Rate Parameter)')
    ax.set_ylabel('Log-Likelihood ℓ(θ)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return mle_theta

def plot_mle_step_by_step(data, save_path=None):
    """Visualize the MLE calculation process step by step."""
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    # Calculate MLE estimate
    mle_theta = np.mean(data)
    
    # Plot 1: The PMF definition
    ax1 = axes[0, 0]
    y_range = np.arange(0, 15)
    theta_examples = [1.0, 2.0, 3.0, 5.0]
    colors = ['red', 'blue', 'green', 'purple']
    
    for theta, color in zip(theta_examples, colors):
        pmf_values = [poisson_pmf(y, theta) for y in y_range]
        ax1.plot(y_range, pmf_values, color=color, marker='o', linestyle='-', linewidth=2, 
                 label=f'θ = {theta}')
    
    ax1.set_title('Poisson Probability Mass Function (PMF)')
    ax1.set_xlabel('y (Count)')
    ax1.set_ylabel('P(y|θ)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: The likelihood function
    ax2 = axes[0, 1]
    theta_range = np.linspace(max(0.1, mle_theta - 1.5), mle_theta + 1.5, 100)
    
    # Calculate likelihood (not log-likelihood)
    likelihoods = []
    for theta in theta_range:
        likelihood = np.prod([poisson_pmf(y, theta) for y in data])
        likelihoods.append(likelihood)
    
    ax2.plot(theta_range, likelihoods, 'b-', linewidth=2)
    ax2.axvline(x=mle_theta, color='r', linestyle='--', label=f'MLE θ = {mle_theta:.4f}')
    ax2.set_title('Likelihood Function L(θ)')
    ax2.set_xlabel('θ (Rate Parameter)')
    ax2.set_ylabel('Likelihood')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: The log-likelihood function
    ax3 = axes[1, 0]
    log_likelihoods = [log_likelihood(theta, data) for theta in theta_range]
    
    ax3.plot(theta_range, log_likelihoods, 'g-', linewidth=2)
    ax3.axvline(x=mle_theta, color='r', linestyle='--', label=f'MLE θ = {mle_theta:.4f}')
    ax3.set_title('Log-Likelihood Function ℓ(θ)')
    ax3.set_xlabel('θ (Rate Parameter)')
    ax3.set_ylabel('Log-Likelihood')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: The derivative of log-likelihood
    ax4 = axes[1, 1]
    
    # Analytical derivative of log-likelihood for Poisson
    # d/dθ[ℓ(θ)] = sum(y_i)/θ - n
    n = len(data)
    sum_y = np.sum(data)
    derivatives = [(sum_y/theta - n) for theta in theta_range]
    
    ax4.plot(theta_range, derivatives, 'm-', linewidth=2)
    ax4.axvline(x=mle_theta, color='r', linestyle='--', label=f'MLE θ = {mle_theta:.4f}')
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax4.set_title('Derivative of Log-Likelihood Function')
    ax4.set_xlabel('θ (Rate Parameter)')
    ax4.set_ylabel('d/dθ [ℓ(θ)]')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.suptitle('Maximum Likelihood Estimation Process for Poisson', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_data_visualization(data, save_path=None):
    """Visualize the observed data and the MLE fitted Poisson."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Calculate MLE
    mle_theta = np.mean(data)
    
    # Create histogram of the data
    max_y = max(data) + 2
    bins = np.arange(-0.5, max_y + 0.5, 1)  # Centered bins for integers
    ax.hist(data, bins=bins, density=True, alpha=0.6, color='skyblue', 
            edgecolor='black', label='Observed Data')
    
    # Overlay the fitted Poisson PMF
    y_range = np.arange(0, max_y)
    pmf_values = [poisson_pmf(y, mle_theta) for y in y_range]
    
    ax.plot(y_range, pmf_values, 'ro-', markersize=8, linewidth=2, 
            label=f'MLE Fit (θ = {mle_theta:.2f})')
    
    # Create a text box with the dataset information
    dataset_text = "Dataset:\n"
    for i, y in enumerate(data, 1):
        dataset_text += f"Day {i}: {y}\n"
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, dataset_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Add a text box with MLE calculation
    mle_text = (
        f"MLE Calculation:\n"
        f"θ̂ = (1/n)∑y_i\n"
        f"θ̂ = ({sum(data)}/{len(data)})\n"
        f"θ̂ = {mle_theta:.2f}"
    )
    
    ax.text(0.75, 0.95, mle_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    ax.set_xlabel('y (Error Count)')
    ax.set_ylabel('Probability')
    ax.set_title('Observed Error Counts with Fitted Poisson Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_probability_visualization(theta, save_path=None):
    """Visualize the probability calculation P(Y ≥ 5)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Range of y values to plot
    max_y = max(15, int(theta * 3))
    y_range = np.arange(0, max_y)
    
    # Calculate PMF values
    pmf_values = [poisson_pmf(y, theta) for y in y_range]
    
    # Calculate P(Y ≥ 5)
    prob_y_geq_5 = 1 - sum([poisson_pmf(y, theta) for y in range(5)])
    # Alternative calculation using survival function
    prob_y_geq_5_scipy = stats.poisson.sf(4, theta)
    
    # Plot the PMF
    ax.bar(y_range, pmf_values, color='skyblue', alpha=0.7, 
           edgecolor='black', label=f'Poisson PMF (θ = {theta:.2f})')
    
    # Highlight the region Y ≥ 5
    ax.bar(y_range[5:], pmf_values[5:], color='red', alpha=0.5,
           edgecolor='black', label=f'P(Y ≥ 5) = {prob_y_geq_5:.4f}')
    
    # Add text annotation for the probability
    text = (
        f"P(Y ≥ 5 | θ = {theta:.2f})\n"
        f"= 1 - P(Y < 5 | θ = {theta:.2f})\n"
        f"= 1 - [P(0) + P(1) + P(2) + P(3) + P(4)]\n"
        f"= {prob_y_geq_5:.6f}"
    )
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.65, 0.95, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    ax.set_xlabel('y (Error Count)')
    ax.set_ylabel('Probability Mass P(y|θ)')
    ax.set_title('Probability of Observing 5 or More Errors')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return prob_y_geq_5, prob_y_geq_5_scipy

def calculate_detailed_pmf_values(theta):
    """Calculate the detailed PMF values for y=0 to y=4 and their sum."""
    pmfs = {}
    for y in range(5):
        pmf_value = poisson_pmf(y, theta)
        pmfs[y] = pmf_value
        
    sum_pmf = sum(pmfs.values())
    prob_y_geq_5 = 1 - sum_pmf
    
    return pmfs, sum_pmf, prob_y_geq_5

def plot_ci_illustration(data, confidence=0.95, save_path=None):
    """Illustrate the confidence interval for the Poisson parameter."""
    # Calculate MLE
    theta_mle = np.mean(data)
    n = len(data)
    
    # Calculate confidence interval using normal approximation
    z_critical = stats.norm.ppf((1 + confidence) / 2)
    standard_error = np.sqrt(theta_mle / n)
    ci_lower = theta_mle - z_critical * standard_error
    ci_upper = theta_mle + z_critical * standard_error
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate theta values for plotting
    theta_range = np.linspace(max(0.01, theta_mle - 3*standard_error), 
                               theta_mle + 3*standard_error, 1000)
    
    # Calculate the log-likelihood for each theta
    log_likelihoods = [log_likelihood(theta, data) for theta in theta_range]
    
    # Normalize log-likelihoods for better visualization
    ll_array = np.array(log_likelihoods)
    ll_normalized = (ll_array - np.min(ll_array)) / (np.max(ll_array) - np.min(ll_array))
    
    # Plot the normalized log-likelihood function
    ax.plot(theta_range, ll_normalized, 'b-', linewidth=2)
    
    # Shade the confidence interval region
    ci_region = (theta_range >= ci_lower) & (theta_range <= ci_upper)
    ax.fill_between(theta_range, 0, ll_normalized, where=ci_region, 
                    color='green', alpha=0.3, label=f'{confidence*100:.0f}% CI')
    
    # Mark the MLE point
    ax.axvline(x=theta_mle, color='r', linestyle='-', 
               label=f'MLE θ̂ = {theta_mle:.4f}')
    
    # Mark CI boundaries
    ax.axvline(x=ci_lower, color='g', linestyle='--')
    ax.axvline(x=ci_upper, color='g', linestyle='--')
    
    # Add text annotations
    ci_text = (
        f'MLE Estimate: θ̂ = {theta_mle:.4f}\n'
        f'Standard Error: SE(θ̂) = {standard_error:.4f}\n'
        f'{confidence*100:.0f}% Confidence Interval:\n'
        f'[{ci_lower:.4f}, {ci_upper:.4f}]'
    )
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.03, 0.97, ci_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left', bbox=props)
    
    ax.set_xlabel('θ (Rate Parameter)')
    ax.set_ylabel('Normalized Log-Likelihood')
    ax.set_title(f'Maximum Likelihood Estimate with {confidence*100:.0f}% Confidence Interval')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return theta_mle, standard_error, ci_lower, ci_upper

def sympy_mle_derivation():
    """Perform symbolic derivation of the MLE for Poisson distribution using SymPy."""
    # Define symbolic variables
    theta = sp.Symbol('theta', positive=True)  # Rate parameter
    y = sp.Symbol('y', integer=True)  # Individual observation
    n = sp.Symbol('n', integer=True, positive=True)  # Sample size
    sum_y = sp.Symbol('sum_y', real=True)  # Sum of observations
    
    # Poisson PMF for a single observation
    poisson_pmf = (theta**y * sp.exp(-theta)) / sp.factorial(y)
    
    # Likelihood function for n independent observations
    # L = Product(poisson_pmf_i) from i=1 to n
    # Simplified version knowing that product of exponentials = exponential of sum
    likelihood = (theta**sum_y * sp.exp(-n*theta)) / sp.Symbol('prod_factorial')
    
    # Log-likelihood
    log_likelihood = sum_y * sp.log(theta) - n*theta - sp.log(sp.Symbol('prod_factorial'))
    
    # Derivative of log-likelihood with respect to theta
    d_log_likelihood = sp.diff(log_likelihood, theta)
    
    # Solve for critical points (where derivative = 0)
    mle_equation = sp.Eq(d_log_likelihood, 0)
    mle_solution = sp.solve(mle_equation, theta)[0]
    
    # Second derivative to confirm it's a maximum
    d2_log_likelihood = sp.diff(d_log_likelihood, theta)
    
    # Evaluating the second derivative at the critical point
    d2_evaluated = d2_log_likelihood.subs(theta, mle_solution)
    
    derivation_result = {
        'poisson_pmf': poisson_pmf,
        'likelihood': likelihood,
        'log_likelihood': log_likelihood,
        'd_log_likelihood': d_log_likelihood,
        'mle_equation': mle_equation,
        'mle_solution': mle_solution,
        'd2_log_likelihood': d2_log_likelihood,
        'd2_evaluated': d2_evaluated
    }
    
    return derivation_result

def detailed_numeric_mle_calculation(data):
    """Perform a detailed numerical calculation of the MLE for Poisson data."""
    n = len(data)
    sum_y = sum(data)
    
    # MLE formula: theta_hat = sum(y_i) / n
    theta_hat = sum_y / n
    
    # Calculate the likelihood and log-likelihood at the MLE
    likelihood_at_mle = np.prod([poisson_pmf(y, theta_hat) for y in data])
    log_likelihood_at_mle = log_likelihood(theta_hat, data)
    
    # Calculate derivative at the MLE (should be approximately zero)
    derivative_at_mle = sum_y/theta_hat - n
    
    # Double-check using log-likelihood values around MLE
    epsilon = 0.001
    log_ll_below = log_likelihood(theta_hat - epsilon, data)
    log_ll_at = log_likelihood(theta_hat, data)
    log_ll_above = log_likelihood(theta_hat + epsilon, data)
    
    # If log_ll_at is greater than both log_ll_below and log_ll_above, it's a maximum
    is_maximum = (log_ll_at > log_ll_below) and (log_ll_at > log_ll_above)
    
    result = {
        'data': data,
        'n': n,
        'sum_y': sum_y,
        'theta_hat': theta_hat,
        'likelihood_at_mle': likelihood_at_mle,
        'log_likelihood_at_mle': log_likelihood_at_mle,
        'derivative_at_mle': derivative_at_mle,
        'is_maximum': is_maximum,
        'log_ll_comparison': {
            f'ℓ(θ̂-{epsilon})': log_ll_below,
            'ℓ(θ̂)': log_ll_at,
            f'ℓ(θ̂+{epsilon})': log_ll_above
        }
    }
    
    return result

def detailed_probability_calculation(theta):
    """Calculate the probability P(Y ≥ 5) with detailed steps."""
    # Calculate individual probabilities
    detailed_probs = {}
    for y in range(5):
        prob = poisson_pmf(y, theta)
        detailed_probs[f'P(Y={y})'] = {
            'formula': f'({theta}^{y} * e^(-{theta})) / {y}!',
            'value': prob
        }
    
    # Calculate P(Y < 5)
    p_y_lt_5 = sum(detailed_probs[f'P(Y={y})']['value'] for y in range(5))
    
    # Calculate P(Y ≥ 5)
    p_y_geq_5 = 1 - p_y_lt_5
    
    # Alternative calculation using scipy
    p_y_geq_5_scipy = stats.poisson.sf(4, theta)
    
    result = {
        'theta': theta,
        'detailed_probs': detailed_probs,
        'P(Y<5)': p_y_lt_5,
        'P(Y≥5)': p_y_geq_5,
        'P(Y≥5)_scipy': p_y_geq_5_scipy
    }
    
    return result

def plot_mle_derivation(save_path=None):
    """Visualize the derivation of the MLE for Poisson."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')  # Hide axes
    
    derivation_text = """
    # Derivation of MLE for Poisson Distribution

    ## Step 1: Write down the likelihood function
    
    For data $y_1, y_2, ..., y_n$ from Poisson($\\theta$), the likelihood is:
    
    $L(\\theta) = \\prod_{i=1}^{n} \\frac{\\theta^{y_i} e^{-\\theta}}{y_i!}$
    
    $L(\\theta) = \\frac{\\theta^{\\sum y_i} e^{-n\\theta}}{\\prod y_i!}$
    
    ## Step 2: Take the logarithm to get log-likelihood
    
    $\\ell(\\theta) = \\ln L(\\theta) = \\sum y_i \\ln(\\theta) - n\\theta - \\sum \\ln(y_i!)$
    
    ## Step 3: Find the derivative and set to zero
    
    $\\frac{d\\ell}{d\\theta} = \\frac{\\sum y_i}{\\theta} - n = 0$
    
    ## Step 4: Solve for $\\theta$
    
    $\\frac{\\sum y_i}{\\theta} = n$
    
    $\\hat{\\theta}_{MLE} = \\frac{\\sum y_i}{n} = \\bar{y}$
    
    ## Step 5: Verify it's a maximum
    
    $\\frac{d^2\\ell}{d\\theta^2} = -\\frac{\\sum y_i}{\\theta^2} < 0$
    
    Since the second derivative is negative, we confirm $\\hat{\\theta}_{MLE} = \\bar{y}$ is a maximum.
    
    ## Summary
    
    The MLE for the Poisson parameter $\\theta$ is the sample mean $\\bar{y}$.
    """
    
    ax.text(0.5, 0.5, derivation_text, transform=ax.transAxes, fontsize=14,
            verticalalignment='center', horizontalalignment='center',
            bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=1'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def main():
    """Generate all visualizations for Question 30 of the L2.4 quiz."""
    # The dataset from the problem
    quiz_data = np.array([3, 1, 2, 0, 4, 2, 3, 1, 2, 2])
    
    # Create a directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level and then to Images
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    # Create a subfolder for this quiz
    save_dir = os.path.join(images_dir, "L2_4_Quiz_30")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 30 of the L2.4 MLE quiz...")
    print("-" * 60)
    
    # 1. Symbolic MLE derivation
    try:
        print("1. Symbolic derivation of the MLE for Poisson:")
        symbolic_results = sympy_mle_derivation()
        
        print(f"Poisson PMF (single observation): {symbolic_results['poisson_pmf']}")
        print(f"Likelihood function: {symbolic_results['likelihood']}")
        print(f"Log-likelihood function: {symbolic_results['log_likelihood']}")
        print(f"Derivative of log-likelihood: {symbolic_results['d_log_likelihood']}")
        print(f"MLE equation (derivative = 0): {symbolic_results['mle_equation']}")
        print(f"MLE solution: θ̂ = {symbolic_results['mle_solution']}")
        print(f"Second derivative: {symbolic_results['d2_log_likelihood']}")
        print(f"Second derivative at MLE: {symbolic_results['d2_evaluated']}")
        print("-" * 60)
    except ImportError:
        print("SymPy not available, skipping symbolic derivation.")
        print("-" * 60)
    
    # 2. Detailed numerical calculation of MLE
    print("2. Detailed numerical calculation of MLE:")
    numeric_results = detailed_numeric_mle_calculation(quiz_data)
    
    print(f"Data: {numeric_results['data']}")
    print(f"Number of observations (n): {numeric_results['n']}")
    print(f"Sum of observations (Σy_i): {numeric_results['sum_y']}")
    print(f"MLE estimate (θ̂ = Σy_i/n): {numeric_results['sum_y']} / {numeric_results['n']} = {numeric_results['theta_hat']:.6f}")
    print(f"Likelihood at MLE: {numeric_results['likelihood_at_mle']:.6e}")
    print(f"Log-likelihood at MLE: {numeric_results['log_likelihood_at_mle']:.6f}")
    print(f"Derivative at MLE (should be ≈ 0): {numeric_results['derivative_at_mle']:.8f}")
    print(f"Is a maximum: {numeric_results['is_maximum']}")
    print("Log-likelihood comparison:")
    for label, value in numeric_results['log_ll_comparison'].items():
        print(f"  {label}: {value:.6f}")
    print("-" * 60)
    
    # 3. Detailed probability calculation
    mle_theta = numeric_results['theta_hat']
    print(f"3. Detailed calculation of P(Y ≥ 5) with θ = {mle_theta}:")
    prob_results = detailed_probability_calculation(mle_theta)
    
    print("Individual probabilities:")
    for label, prob_info in prob_results['detailed_probs'].items():
        print(f"  {label} = {prob_info['formula']} = {prob_info['value']:.6f}")
    print(f"P(Y < 5) = {prob_results['P(Y<5)']:.6f}")
    print(f"P(Y ≥ 5) = 1 - P(Y < 5) = 1 - {prob_results['P(Y<5)']:.6f} = {prob_results['P(Y≥5)']:.6f}")
    print(f"P(Y ≥ 5) using scipy: {prob_results['P(Y≥5)_scipy']:.6f}")
    print("-" * 60)
    
    # 4. Generate visualizations
    print("4. Generating visualizations...")
    
    # 4.1 Plot PMFs for different theta values
    plot_poisson_pmf([1, 2, 3, 5], save_path=os.path.join(save_dir, "poisson_pmfs.png"))
    print("4.1 Poisson PMF visualization created")
    
    # 4.2 Plot the derivation of MLE
    plot_mle_derivation(save_path=os.path.join(save_dir, "mle_derivation.png"))
    print("4.2 MLE derivation visualization created")
    
    # 4.3 Plot likelihood surface
    mle_theta = plot_likelihood_surface(quiz_data, save_path=os.path.join(save_dir, "likelihood_surface.png"))
    print(f"4.3 Likelihood surface visualization created, MLE θ = {mle_theta:.4f}")
    
    # 4.4 Plot step-by-step MLE process
    plot_mle_step_by_step(quiz_data, save_path=os.path.join(save_dir, "mle_process.png"))
    print("4.4 Step-by-step MLE process visualization created")
    
    # 4.5 Plot data visualization
    plot_data_visualization(quiz_data, save_path=os.path.join(save_dir, "data_visualization.png"))
    print("4.5 Data visualization created")
    
    # 4.6 Plot probability calculation for P(Y ≥ 5)
    prob, prob_scipy = plot_probability_visualization(mle_theta, save_path=os.path.join(save_dir, "probability_visualization.png"))
    print(f"4.6 Probability visualization created, P(Y ≥ 5) = {prob:.6f}")
    
    # 4.7 Plot confidence interval
    mle, std_err, ci_lower, ci_upper = plot_ci_illustration(quiz_data, save_path=os.path.join(save_dir, "confidence_interval.png"))
    print(f"4.7 Confidence interval visualization created (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
    print("-" * 60)
    
    # Create a summary of results
    print("\n=== Summary of Results ===")
    print(f"Dataset: {quiz_data}")
    print(f"MLE estimate for θ: {mle_theta:.4f}")
    print(f"P(Y ≥ 5 | θ = {mle_theta:.4f}) = {prob:.6f}")
    print(f"95% Confidence Interval for θ: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    print(f"\nAll visualizations have been saved to: {save_dir}")

if __name__ == "__main__":
    main() 