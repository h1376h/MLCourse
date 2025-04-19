import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon
import os
from matplotlib import cm

def exponential_pdf(x, rate):
    """
    Probability density function of exponential distribution with rate parameter lambda
    """
    return rate * np.exp(-rate * x)

def log_likelihood(rate, data):
    """
    Log-likelihood function for exponential distribution
    """
    n = len(data)
    return n * np.log(rate) - rate * np.sum(data)

def generate_data(n, rate, random_seed=42):
    """
    Generate random data from exponential distribution
    """
    np.random.seed(random_seed)
    return np.random.exponential(scale=1/rate, size=n)

def plot_exponential_pdfs(rate_values, save_path=None):
    """
    Plot PDF of exponential distribution for different rate parameters
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.linspace(0, 5, 1000)
    
    for rate in rate_values:
        y = exponential_pdf(x, rate)
        ax.plot(x, y, linewidth=2, label=f'λ = {rate}')
    
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density')
    ax.set_title('Exponential Distribution PDFs for Different λ Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_likelihood_surface(data, save_path=None):
    """
    Plot log-likelihood function surface for given data
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate true MLE (reciprocal of sample mean)
    mle_lambda = 1 / np.mean(data)
    
    # Create range of possible lambda values
    lambda_range = np.linspace(max(0.1, mle_lambda*0.2), mle_lambda*1.8, 1000)
    log_likelihoods = [log_likelihood(lambda_val, data) for lambda_val in lambda_range]
    
    ax.plot(lambda_range, log_likelihoods, 'b-', linewidth=2)
    ax.axvline(x=mle_lambda, color='r', linestyle='--', 
               label=f'MLE λ = {mle_lambda:.4f}')
    
    # Mark the maximum point
    max_log_likelihood = log_likelihood(mle_lambda, data)
    ax.plot([mle_lambda], [max_log_likelihood], 'ro', markersize=8)
    
    ax.set_xlabel('λ')
    ax.set_ylabel('Log-Likelihood ℓ(λ)')
    ax.set_title('Log-Likelihood Function for Exponential Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return mle_lambda

def plot_mle_fit(data, save_path=None):
    """
    Plot data histogram with MLE-fitted PDF
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate MLE
    mle_lambda = 1 / np.mean(data)
    
    # Plot histogram of the data
    ax.hist(data, bins='auto', density=True, alpha=0.6, color='blue', 
             label='Data Histogram')
    
    # Plot the fitted PDF
    x = np.linspace(0, max(data)*1.5, 1000)
    y = exponential_pdf(x, mle_lambda)
    ax.plot(x, y, 'r-', linewidth=2, 
            label=f'MLE Fit (λ = {mle_lambda:.4f})')
    
    # Plot the true PDF if available
    true_lambda = 2.0  # Example value, replace with actual if known
    y_true = exponential_pdf(x, true_lambda)
    ax.plot(x, y_true, 'g--', linewidth=2, 
            label=f'True Distribution (λ = {true_lambda:.4f})')
    
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density')
    ax.set_title('MLE Fit to Exponential Distribution Data')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return mle_lambda

def plot_sample_size_effect(true_lambda=2.0, sample_sizes=(10, 50, 100, 500), save_path=None):
    """
    Demonstrate how MLE converges with increasing sample size
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    np.random.seed(42)  # For reproducibility
    
    for n in sample_sizes:
        # Generate data
        data = generate_data(n, true_lambda)
        
        # Calculate MLE
        mle_lambda = 1 / np.mean(data)
        
        # Calculate log-likelihood function
        lambda_range = np.linspace(max(0.1, mle_lambda*0.5), mle_lambda*1.5, 1000)
        log_likelihoods = [log_likelihood(lambda_val, data) for lambda_val in lambda_range]
        
        # Normalize for better comparison
        log_likelihoods = np.array(log_likelihoods)
        log_likelihoods = log_likelihoods - np.max(log_likelihoods)
        
        ax.plot(lambda_range, log_likelihoods, linewidth=2, 
                label=f'n = {n}, MLE = {mle_lambda:.4f}')
    
    # Highlight true value
    ax.axvline(x=true_lambda, color='k', linestyle='--', 
               label=f'True λ = {true_lambda:.4f}')
    
    ax.set_xlabel('λ')
    ax.set_ylabel('Normalized Log-Likelihood')
    ax.set_title('Effect of Sample Size on MLE Precision')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def main():
    """Main function to execute all visualizations"""
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_4_Quiz_24")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 24 of the L2.4 MLE quiz...")
    
    # Generate synthetic data
    true_lambda = 2.0
    n = 100
    data = generate_data(n, true_lambda)
    
    print(f"\nGenerated data summary:")
    print(f"- Sample size: {n}")
    print(f"- True λ: {true_lambda}")
    print(f"- Sample mean: {np.mean(data):.4f}")
    print(f"- Theoretical mean (1/λ): {1/true_lambda:.4f}")
    
    # 1. Plot PDFs for different lambda values
    plot_exponential_pdfs([0.5, 1.0, 2.0, 3.0], 
                         save_path=os.path.join(save_dir, "exponential_pdfs.png"))
    print("\n1. PDF visualization created")
    
    # 2. Plot likelihood surface
    mle_lambda = plot_likelihood_surface(data, 
                                       save_path=os.path.join(save_dir, "likelihood_surface.png"))
    print(f"2. Likelihood surface visualization created")
    print(f"   MLE estimate: λ = {mle_lambda:.4f}")
    
    # 3. Plot MLE fit to data
    plot_mle_fit(data, save_path=os.path.join(save_dir, "mle_fit.png"))
    print("3. MLE fit visualization created")
    
    # 4. Demonstrate effect of sample size
    plot_sample_size_effect(true_lambda=true_lambda, 
                           save_path=os.path.join(save_dir, "sample_size_effect.png"))
    print("4. Sample size effect visualization created")
    
    # Print steps to find MLE algebraically
    print("\nAlgebraic Steps to Find MLE for Exponential Distribution:")
    print("Step 1: Express the likelihood function")
    print(f"   L(λ) = λ^n × exp(-λ × Σ(x_i))")
    print("Step 2: Take the logarithm")
    print(f"   ℓ(λ) = n×log(λ) - λ×Σ(x_i)")
    print("Step 3: Take the derivative and set to zero")
    print(f"   dℓ/dλ = n/λ - Σ(x_i) = 0")
    print("Step 4: Solve for λ")
    print(f"   λ = n/Σ(x_i) = 1/x̄")
    
    print(f"\nNumerical Solution:")
    print(f"For our data with n = {n}:")
    print(f"- Sample mean: {np.mean(data):.4f}")
    print(f"- MLE estimate: λ = 1/x̄ = 1/{np.mean(data):.4f} = {mle_lambda:.4f}")
    print(f"- True λ: {true_lambda}")
    print(f"- Relative error: {abs(mle_lambda - true_lambda)/true_lambda*100:.2f}%")
    
    # Print formal answer to the question
    print("\nFormal Answer to Question 24:")
    print("Given a sample D = {x₁, ..., xₙ} from an exponential distribution")
    print("with parameter λ > 0, the ML estimate of λ is:")
    print("   λ̂ = n / Σ(x_i) = 1 / x̄")
    print("where x̄ is the sample mean.")

if __name__ == "__main__":
    main() 