import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, norm, binom
import os

def calculate_mle(samples):
    """Calculate the Maximum Likelihood Estimator for a Bernoulli distribution."""
    return np.mean(samples)

def compute_likelihood(p, samples):
    """Compute the likelihood function for a Bernoulli distribution."""
    n_success = np.sum(samples)
    n_failure = len(samples) - n_success
    return p**n_success * (1-p)**(n_failure)

def compute_log_likelihood(p, samples):
    """Compute the log-likelihood function for a Bernoulli distribution."""
    n_success = np.sum(samples)
    n_failure = len(samples) - n_success
    return n_success * np.log(p) + n_failure * np.log(1-p)

def plot_bernoulli_pmf(p_values, save_path=None):
    """Plot the PMF of Bernoulli distribution for different p values."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.array([0, 1])
    width = 0.2
    
    for i, p in enumerate(p_values):
        pmf = np.array([1-p, p])
        ax.bar(x + (i-len(p_values)/2+0.5)*width, pmf, width=width, alpha=0.7, 
               label=f'p = {p}')
    
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Mass')
    ax.set_title('Bernoulli Distribution PMF for Different p Values')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['0', '1'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    # Print explanations instead of including in the image
    print("\n=== Bernoulli PMF Explanation ===")
    print("The Bernoulli probability mass function (PMF) is:")
    print("P(X=x|p) = p^x (1-p)^(1-x), for x ∈ {0,1}")
    print("\nFor different values of p, the probabilities are:")
    for p in p_values:
        print(f"p = {p}:")
        print(f"  P(X=0) = {1-p:.4f}")
        print(f"  P(X=1) = {p:.4f}")

def plot_likelihood_function(samples, save_path=None):
    """Plot the likelihood function for the given sample data."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Create a range of p values for plotting
    p_range = np.linspace(0.001, 0.999, 1000)
    
    # Compute likelihood and log-likelihood for each p value
    likelihoods = [compute_likelihood(p, samples) for p in p_range]
    log_likelihoods = [compute_log_likelihood(p, samples) for p in p_range]
    
    # Find the MLE
    p_mle = calculate_mle(samples)
    
    # Plot likelihood function
    ax1.plot(p_range, likelihoods, 'b-', linewidth=2)
    ax1.axvline(x=p_mle, color='red', linestyle='--', 
               label=f'MLE θ̂ = {p_mle:.2f}')
    ax1.set_xlabel('θ')
    ax1.set_ylabel('Likelihood L(θ)')
    ax1.set_title('Likelihood Function for Bernoulli Parameter θ')
    
    # Calculate the likelihood at MLE for annotation
    likelihood_at_mle = compute_likelihood(p_mle, samples)
    ax1.plot([p_mle], [likelihood_at_mle], 'ro', markersize=8)
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot log-likelihood function
    ax2.plot(p_range, log_likelihoods, 'g-', linewidth=2)
    ax2.axvline(x=p_mle, color='red', linestyle='--', 
               label=f'MLE θ̂ = {p_mle:.2f}')
    ax2.set_xlabel('θ')
    ax2.set_ylabel('Log-Likelihood ℓ(θ)')
    ax2.set_title('Log-Likelihood Function for Bernoulli Parameter θ')
    
    # Calculate the log-likelihood at MLE for annotation
    log_likelihood_at_mle = compute_log_likelihood(p_mle, samples)
    ax2.plot([p_mle], [log_likelihood_at_mle], 'ro', markersize=8)
    
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    # Print explanations instead of including in the image
    n = len(samples)
    k = np.sum(samples)
    
    print("\n=== Likelihood Function Explanation ===")
    print(f"For our sample data with {k} successes out of {n} trials:")
    print("The likelihood function is:")
    print(f"L(θ) = θ^{k} × (1-θ)^{n-k} = θ^{k} × (1-θ)^{n-k}")
    print(f"At θ = {p_mle:.4f} (MLE), the likelihood is L({p_mle:.4f}) = {likelihood_at_mle:.8f}")
    
    print("\nThe log-likelihood function is:")
    print(f"ℓ(θ) = {k}×log(θ) + {n-k}×log(1-θ)")
    print(f"At θ = {p_mle:.4f} (MLE), the log-likelihood is ℓ({p_mle:.4f}) = {log_likelihood_at_mle:.6f}")
    
    # Print likelihood and log-likelihood values for a few selected points
    test_points = [0.1, 0.2, 0.3, p_mle, 0.5, 0.6, 0.7]
    print("\nLikelihood and log-likelihood values at selected points:")
    for p in test_points:
        l_value = compute_likelihood(p, samples)
        ll_value = compute_log_likelihood(p, samples)
        print(f"θ = {p:.2f}: L(θ) = {l_value:.8f}, ℓ(θ) = {ll_value:.6f}")

def visualize_sample_data(samples, save_path=None):
    """Visualize the sample data with annotations."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the samples
    x = np.arange(1, len(samples)+1)
    ax.stem(x, samples, linefmt='b-', markerfmt='bo', basefmt='r-',
           label='Sample Data')
    
    # Add text annotations
    for i, sample in enumerate(samples):
        ax.annotate(f'{sample}', 
                   xy=(i+1, sample), 
                   xytext=(i+1, sample + 0.1),
                   ha='center')
    
    # Calculate and display MLE
    mle = calculate_mle(samples)
    ax.axhline(y=mle, color='green', linestyle='--', 
              label=f'MLE θ̂ = {mle:.2f}')
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Value (0 or 1)')
    ax.set_title('Bernoulli Sample Data Visualization')
    ax.set_xticks(x)
    ax.set_yticks([0, 1])
    ax.set_ylim(-0.1, 1.2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    # Print explanations instead of including in the image
    n = len(samples)
    k = np.sum(samples)
    
    print("\n=== Sample Data Explanation ===")
    print(f"Sample data: {samples}")
    print(f"Sample size (n): {n}")
    print(f"Number of successes (k): {k}")
    print(f"Number of failures: {n-k}")
    print(f"MLE Estimate (θ̂): {mle:.4f}")
    print("\nThe Maximum Likelihood Estimator (MLE) for a Bernoulli distribution is:")
    print("θ̂ = k/n = number of successes / sample size")
    print(f"θ̂ = {k}/{n} = {mle:.4f}")

def print_step_by_step_derivation(samples):
    """Print step-by-step derivation of the MLE for a Bernoulli distribution."""
    n = len(samples)
    k = np.sum(samples)
    p_mle = k/n
    
    print("\n=== Step-by-Step Derivation of MLE ===")
    print("Starting with the Bernoulli likelihood function:")
    print(f"L(θ) = ∏(i=1 to {n}) θ^x_i (1-θ)^(1-x_i) = θ^{k} (1-θ)^{n-k}")
    
    print("\nTaking the logarithm to get the log-likelihood:")
    print(f"ℓ(θ) = log(L(θ)) = {k} log(θ) + {n-k} log(1-θ)")
    
    print("\nTaking the derivative with respect to θ and setting it to zero:")
    print(f"dℓ/dθ = {k}/θ - {n-k}/(1-θ) = 0")
    
    print("\nSolving for θ:")
    print(f"{k}/θ = {n-k}/(1-θ)")
    print(f"(1-θ){k} = θ({n-k})")
    print(f"{k} - {k}θ = {n}θ - {k}θ")
    print(f"{k} = {n}θ")
    print(f"θ = {k}/{n} = {p_mle:.4f}")
    
    print("\nVerifying this is a maximum (not a minimum):")
    print("The second derivative is:")
    print(f"d²ℓ/dθ² = -{k}/θ² - {n-k}/(1-θ)²")
    print(f"At θ = {p_mle:.4f}, d²ℓ/dθ² = -{k}/{p_mle**2:.4f} - {n-k}/{(1-p_mle)**2:.4f} < 0")
    print("Since the second derivative is negative, this is indeed a maximum.")

def main():
    """Main function to generate all visualizations for Question 22."""
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_4_Quiz_22")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 22 of the L2.4 MLE quiz...")
    
    # Problem parameters: samples from the Bernoulli distribution
    samples = np.array([0, 0, 1, 1, 0])
    
    # Calculate MLE
    mle = calculate_mle(samples)
    print(f"\n=== MLE Calculation ===")
    print(f"Sample data: {samples}")
    print(f"MLE Estimate (θ̂): {mle:.4f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Visualize the sample data
    visualize_sample_data(samples, 
                         save_path=os.path.join(save_dir, "sample_data.png"))
    print("1. Sample data visualization created")
    
    # 2. Plot Bernoulli PMF for different p values
    plot_bernoulli_pmf([0.2, 0.4, mle, 0.6, 0.8], 
                      save_path=os.path.join(save_dir, "bernoulli_pmf.png"))
    print("2. Bernoulli PMF visualization created")
    
    # 3. Plot likelihood and log-likelihood functions
    plot_likelihood_function(samples, 
                           save_path=os.path.join(save_dir, "likelihood_functions.png"))
    print("3. Likelihood functions visualization created")
    
    # 4. Print step-by-step derivation
    print_step_by_step_derivation(samples)
    
    print(f"\n=== Summary ===")
    print(f"All visualizations have been saved to: {save_dir}")
    print("The Maximum Likelihood Estimator for the Bernoulli parameter θ is:")
    print(f"θ̂ = {mle:.4f}")
    
    # Calculate standard error for the MLE
    se = np.sqrt(mle * (1-mle) / len(samples))
    print(f"\nStandard Error of the MLE: {se:.4f}")
    
    # Calculate approximate 95% confidence interval
    ci_lower = max(0, mle - 1.96 * se)
    ci_upper = min(1, mle + 1.96 * se)
    print(f"Approximate 95% Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})")
    
    # Calculate probability mass function at different values
    print(f"\nProbability Mass Function at different values with θ = {mle:.4f}:")
    print(f"P(X=0|θ={mle:.4f}) = {1-mle:.4f}")
    print(f"P(X=1|θ={mle:.4f}) = {mle:.4f}")

if __name__ == "__main__":
    main() 