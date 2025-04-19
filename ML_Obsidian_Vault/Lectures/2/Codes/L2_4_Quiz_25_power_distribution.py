import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import os
import scipy.stats as stats
from matplotlib.ticker import MaxNLocator

def create_directory(img_dir):
    """Create directory for saving images if it doesn't exist."""
    os.makedirs(img_dir, exist_ok=True)
    print(f"Images will be saved to: {img_dir}")

def pdf_function(x, theta):
    """
    PDF of the power distribution: f(x) = 3x²/θ³ for 0 ≤ x ≤ θ
    """
    if np.isscalar(x):
        if 0 <= x <= theta:
            return 3 * (x**2) / (theta**3)
        else:
            return 0
    else:
        result = np.zeros_like(x, dtype=float)
        mask = (0 <= x) & (x <= theta)
        result[mask] = 3 * (x[mask]**2) / (theta**3)
        return result

def plot_pdf(theta_values, img_dir):
    """
    Plot the PDF for different values of theta.
    """
    plt.figure(figsize=(10, 6))
    
    x_values = np.linspace(0, max(theta_values) + 0.5, 1000)
    
    for theta in theta_values:
        y_values = [pdf_function(x, theta) for x in x_values]
        plt.plot(x_values, y_values, label=f'θ = {theta}')
        
        # Mark the boundary at x = theta
        plt.axvline(x=theta, linestyle='--', color='gray', alpha=0.5)
    
    plt.xlabel('x')
    plt.ylabel('f(x|θ)')
    plt.title('PDF: f(x|θ) = 3x²/θ³ for 0 ≤ x ≤ θ')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(img_dir, 'pdf_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("PDF plot created.")
    print("The PDF shows how the distribution changes with different values of θ.")
    print("Note that for larger θ values, the distribution becomes more spread out with a lower peak.")
    print("The distribution is always bounded in the range [0, θ] for any given θ value.")

def generate_random_samples(theta, n):
    """
    Generate n random samples from the power distribution with parameter theta.
    
    For the CDF: F(x) = ∫(0 to x) 3t²/θ³ dt = (x³/θ³) for 0 ≤ x ≤ θ
    
    To generate samples, we use inverse transform sampling:
    1. Generate u ~ Uniform(0,1)
    2. Return x = F^(-1)(u) = θ × u^(1/3)
    """
    u = np.random.uniform(0, 1, n)
    return theta * np.power(u, 1/3)

def log_likelihood_function(theta, samples):
    """
    Calculate the log-likelihood for the power distribution.
    
    log L(θ) = log[∏(i=1 to n) 3x_i²/θ³]
    = ∑(i=1 to n) log(3x_i²/θ³)
    = n*log(3) + 2*∑(i=1 to n) log(x_i) - 3n*log(θ)
    """
    n = len(samples)
    valid_samples = samples[samples <= theta]  # Only consider samples within the bound
    
    if len(valid_samples) < n:
        return -np.inf  # If any sample is outside [0,θ], likelihood is 0
    
    log_likelihood = n * np.log(3) + 2 * np.sum(np.log(samples)) - 3 * n * np.log(theta)
    return log_likelihood

def negative_log_likelihood(theta, samples):
    """Negative log-likelihood for minimization."""
    return -log_likelihood_function(theta, samples)

def likelihood_function(theta, samples):
    """Calculate the likelihood for the given theta value."""
    return np.exp(log_likelihood_function(theta, samples))

def find_mle_analytical(samples):
    """
    Find the MLE analytically. For this PDF, the MLE is max(x_1, x_2, ..., x_n).
    """
    return np.max(samples)

def plot_likelihood(samples, img_dir):
    """
    Plot the likelihood and log-likelihood functions for different theta values.
    """
    mle_theta = find_mle_analytical(samples)
    
    # Define range for theta values
    min_theta = max(0.9 * mle_theta, 0.01)
    max_theta = 1.5 * mle_theta
    theta_values = np.linspace(min_theta, max_theta, 1000)
    
    # Calculate likelihood and log-likelihood values
    likelihood_values = [likelihood_function(theta, samples) for theta in theta_values]
    log_likelihood_values = [log_likelihood_function(theta, samples) for theta in theta_values]
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Likelihood plot
    ax1.plot(theta_values, likelihood_values, 'b-', linewidth=2)
    ax1.axvline(x=mle_theta, color='red', linestyle='--', 
               label=f'MLE θ̂ = {mle_theta:.4f}')
    ax1.set_xlabel('θ')
    ax1.set_ylabel('Likelihood L(θ)')
    ax1.set_title('Likelihood Function')
    
    # Calculate the likelihood at MLE for annotation
    likelihood_at_mle = likelihood_function(mle_theta, samples)
    ax1.plot([mle_theta], [likelihood_at_mle], 'ro', markersize=8)
    ax1.annotate(f"Max at θ = {mle_theta:.4f}", 
                xy=(mle_theta, likelihood_at_mle),
                xytext=(mle_theta + 0.05, likelihood_at_mle),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Log-likelihood plot
    ax2.plot(theta_values, log_likelihood_values, 'g-', linewidth=2)
    ax2.axvline(x=mle_theta, color='red', linestyle='--', 
               label=f'MLE θ̂ = {mle_theta:.4f}')
    ax2.set_xlabel('θ')
    ax2.set_ylabel('Log-Likelihood ℓ(θ)')
    ax2.set_title('Log-Likelihood Function')
    
    # Calculate the log-likelihood at MLE for annotation
    log_likelihood_at_mle = log_likelihood_function(mle_theta, samples)
    ax2.plot([mle_theta], [log_likelihood_at_mle], 'ro', markersize=8)
    ax2.annotate(f"Max at θ = {mle_theta:.4f}", 
                xy=(mle_theta, log_likelihood_at_mle),
                xytext=(mle_theta + 0.05, log_likelihood_at_mle),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'likelihood_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nLikelihood and log-likelihood plots created.")
    print(f"MLE for θ (analytical): {mle_theta:.6f}")
    print(f"Likelihood at MLE: {likelihood_at_mle:.6e}")
    print(f"Log-likelihood at MLE: {log_likelihood_at_mle:.6f}")
    print("\nThe likelihood function attains its maximum at the largest observed value in the sample.")
    print("This is because the likelihood increases as θ approaches the maximum sample value from above,")
    print("but becomes zero if θ is less than any observed value (as samples must be in the range [0,θ]).")

def print_step_by_step_derivation():
    """
    Print a step-by-step derivation of the MLE for this distribution.
    """
    print("\n=== Step-by-Step Derivation of the MLE ===")
    print("Step 1: Write out the likelihood function")
    print("For n i.i.d. samples from f(x|θ) = 3x²/θ³ for 0 ≤ x ≤ θ:")
    print("L(θ) = ∏(i=1 to n) f(x_i|θ) = ∏(i=1 to n) 3x_i²/θ³ = 3^n/θ^(3n) × ∏(i=1 to n) x_i²")
    print("Note: This formula is valid only when 0 ≤ x_i ≤ θ for all i. Otherwise, L(θ) = 0.\n")
    
    print("Step 2: Take the logarithm to get the log-likelihood function")
    print("ℓ(θ) = ln L(θ) = n×ln(3) + 2×∑(i=1 to n)ln(x_i) - 3n×ln(θ)\n")
    
    print("Step 3: Find critical points by taking the derivative")
    print("dℓ/dθ = -3n/θ")
    print("This is always negative for θ > 0, so there is no critical point.\n")
    
    print("Step 4: Consider the domain constraints")
    print("The likelihood function is only valid when θ ≥ max(x_1, x_2, ..., x_n)")
    print("Since dℓ/dθ < 0, the log-likelihood is strictly decreasing in θ.")
    print("Therefore, the maximum occurs at the smallest valid value of θ, which is max(x_1, x_2, ..., x_n).\n")
    
    print("Step 5: Write the MLE formula")
    print("θ̂_MLE = max(x_1, x_2, ..., x_n)")
    
    print("\nThe derivation shows why the MLE for this distribution is the maximum observed value in the sample.")

def plot_sample_distribution(samples, theta_true, img_dir):
    """
    Plot the histogram of samples compared with the theoretical PDF.
    """
    plt.figure(figsize=(10, 6))
    
    # Histogram of samples
    plt.hist(samples, bins=20, density=True, alpha=0.7, 
             label='Sample Histogram', color='skyblue', edgecolor='black')
    
    # Theoretical PDF curve
    x_values = np.linspace(0, theta_true, 1000)
    y_values = [pdf_function(x, theta_true) for x in x_values]
    plt.plot(x_values, y_values, 'r-', linewidth=2, 
             label=f'Theoretical PDF (θ = {theta_true})')
    
    # Mark the MLE
    mle_theta = find_mle_analytical(samples)
    plt.axvline(x=mle_theta, color='green', linestyle='--', 
               label=f'MLE θ̂ = {mle_theta:.4f}')
    
    # Mark the true theta
    plt.axvline(x=theta_true, color='blue', linestyle=':', 
               label=f'True θ = {theta_true}')
    
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title('Sample Distribution and Theoretical PDF')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(img_dir, 'sample_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nSample distribution plot created.")
    print(f"True θ value: {theta_true}")
    print(f"MLE estimate θ̂: {mle_theta:.6f}")
    print(f"Relative error: {(mle_theta - theta_true) / theta_true * 100:.2f}%")

def analyze_mle_bias(true_theta, sample_sizes, num_simulations, img_dir):
    """
    Analyze the bias and consistency of the MLE estimator for different sample sizes.
    """
    mle_estimates = np.zeros((len(sample_sizes), num_simulations))
    
    for i, n in enumerate(sample_sizes):
        for j in range(num_simulations):
            samples = generate_random_samples(true_theta, n)
            mle_estimates[i, j] = find_mle_analytical(samples)
    
    # Calculate mean estimates and MSEs
    mean_estimates = np.mean(mle_estimates, axis=1)
    biases = mean_estimates - true_theta
    mses = np.mean((mle_estimates - true_theta)**2, axis=1)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Bias plot
    ax1.plot(sample_sizes, biases, 'bo-', markersize=6)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Sample Size (n)')
    ax1.set_ylabel('Bias: E[θ̂] - θ')
    ax1.set_title('Bias of MLE Estimator vs. Sample Size')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # MSE plot
    ax2.plot(sample_sizes, mses, 'go-', markersize=6)
    ax2.set_xlabel('Sample Size (n)')
    ax2.set_ylabel('Mean Squared Error')
    ax2.set_title('MSE of MLE Estimator vs. Sample Size')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'mle_bias_mse.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Theoretical analysis
    # For this distribution, E[θ̂] = (n/(n+3)) * θ
    theoretical_expectations = true_theta * np.array(sample_sizes) / (np.array(sample_sizes) + 3)
    theoretical_biases = theoretical_expectations - true_theta
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sample_sizes, biases, 'bo-', markersize=6, label='Empirical Bias')
    ax.plot(sample_sizes, theoretical_biases, 'r--', linewidth=2, label='Theoretical Bias = -3θ/(n+3)')
    ax.axhline(y=0, color='k', linestyle=':', alpha=0.7)
    ax.set_xlabel('Sample Size (n)')
    ax.set_ylabel('Bias: E[θ̂] - θ')
    ax.set_title('Empirical vs. Theoretical Bias of MLE Estimator')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(os.path.join(img_dir, 'theoretical_vs_empirical_bias.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nBias and MSE analysis created.")
    print("\nTheoretical properties of this MLE:")
    print(f"For the true θ = {true_theta}, the theoretical formula for the bias is: Bias = -3θ/(n+3)")
    
    for i, n in enumerate(sample_sizes):
        theoretical_bias = -3 * true_theta / (n + 3)
        print(f"n = {n}: Empirical Bias = {biases[i]:.6f}, Theoretical Bias = {theoretical_bias:.6f}")
    
    print("\nAs the sample size increases, the bias approaches zero, showing that the estimator is consistent.")
    print("However, for small sample sizes, this MLE is biased downward.")

def main():
    """Main function to run the analysis."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Parameters
    true_theta = 5.0
    sample_size = 50
    
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_4_Quiz_25")
    create_directory(save_dir)
    
    print("=== Maximum Likelihood Estimation for Power Distribution ===")
    print(f"Distribution: f(x|θ) = 3x²/θ³ for 0 ≤ x ≤ θ")
    print(f"True parameter value: θ = {true_theta}")
    print(f"Sample size: n = {sample_size}")
    
    # Plot the PDF for different values of theta
    plot_pdf([3.0, 4.0, 5.0, 6.0], save_dir)
    
    # Generate random samples
    samples = generate_random_samples(true_theta, sample_size)
    
    # Print basic statistics about the samples
    print("\n=== Sample Statistics ===")
    print(f"Sample size: {len(samples)}")
    print(f"Sample mean: {np.mean(samples):.6f}")
    print(f"Sample variance: {np.var(samples):.6f}")
    print(f"Sample minimum: {np.min(samples):.6f}")
    print(f"Sample maximum: {np.max(samples):.6f}")
    
    # For the PDF f(x) = 3x²/θ³ for 0 ≤ x ≤ θ, the theoretical mean is E[X] = 3θ/4
    theoretical_mean = 3 * true_theta / 4
    print(f"Theoretical mean: E[X] = 3θ/4 = {theoretical_mean:.6f}")
    
    # Theoretical variance: Var[X] = 3θ²/80
    theoretical_var = 3 * true_theta**2 / 80
    print(f"Theoretical variance: Var[X] = 3θ²/80 = {theoretical_var:.6f}")
    
    # Plot likelihood and find MLE
    plot_likelihood(samples, save_dir)
    
    # Print the step-by-step derivation instead of creating an image
    print_step_by_step_derivation()
    
    # Plot sample distribution
    plot_sample_distribution(samples, true_theta, save_dir)
    
    # Analyze bias and consistency
    sample_sizes = [10, 20, 50, 100, 200, 500, 1000]
    analyze_mle_bias(true_theta, sample_sizes, 1000, save_dir)
    
    # Summary
    mle_theta = find_mle_analytical(samples)
    print("\n=== Summary ===")
    print(f"True parameter: θ = {true_theta}")
    print(f"MLE estimate: θ̂ = {mle_theta:.6f}")
    print(f"Bias (Monte Carlo): {mle_theta - true_theta:.6f}")
    
    # Unbiased estimator: θ̂_unbiased = (n+3)/n * θ̂_MLE
    unbiased_estimator = (sample_size + 3) / sample_size * mle_theta
    print(f"Unbiased estimator: θ̂_unbiased = (n+3)/n × θ̂_MLE = {unbiased_estimator:.6f}")
    
    print("\nImages and analysis have been saved to:")
    print(save_dir)
    print("\nExecution completed successfully.")

if __name__ == "__main__":
    main() 