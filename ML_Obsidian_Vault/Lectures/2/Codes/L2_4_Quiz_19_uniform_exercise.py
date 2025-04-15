import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
import os

def plot_uniform_pdf(thetas, save_path=None):
    """Plot the PDF of uniform distribution for different theta values"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.linspace(0, 30, 1000)
    
    for theta in thetas:
        pdf = np.where((x >= 0) & (x <= theta), 1/theta, 0)
        ax.plot(x, pdf, label=f'θ = {theta}')
    
    ax.set_xlabel('Exercise Duration (minutes)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Uniform Distribution PDF for Different θ Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def compute_mle_for_uniform(data, save_path=None):
    """Compute MLE for uniform distribution and visualize"""
    # Calculate MLE (maximum observation)
    theta_hat = np.max(data)
    
    # Calculate log-likelihood function
    theta_range = np.linspace(theta_hat*0.8, theta_hat*1.2, 1000)
    log_likelihood = np.array([-len(data) * np.log(t) if t >= theta_hat else -np.inf for t in theta_range])
    
    # Create visualization of the likelihood function
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(theta_range, log_likelihood, 'b-', linewidth=2)
    ax.axvline(x=theta_hat, color='r', linestyle='--', 
               label=f'MLE θ = {theta_hat:.2f}')
    
    # Mark the data points
    for x in data:
        ax.axvline(x=x, color='g', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('Parameter θ')
    ax.set_ylabel('Log-Likelihood ℓ(θ)')
    ax.set_title('Log-Likelihood Function for Uniform Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add text with key statistics
    mean_hat = theta_hat / 2  # Mean of uniform [0, theta]
    var_hat = (theta_hat ** 2) / 12  # Variance of uniform [0, theta]
    
    text_str = '\n'.join((
        f'Sample Size: {len(data)}',
        f'MLE (θ̂): {theta_hat:.2f}',
        f'Max Observation: {theta_hat:.2f}',
        f'Estimated Mean (θ̂/2): {mean_hat:.2f}',
        f'Estimated Variance (θ̂²/12): {var_hat:.2f}'))
    
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return theta_hat, mean_hat, var_hat

def compute_unbiased_estimator(data, save_path=None):
    """Compute unbiased estimator for uniform distribution and compare with MLE"""
    n = len(data)
    
    # Calculate MLE and unbiased estimator
    theta_hat_mle = np.max(data)
    theta_hat_unbiased = theta_hat_mle * (n + 1) / n
    
    # Create visualization comparing the estimators
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot true PDF with some simulated value
    x = np.linspace(0, theta_hat_unbiased*1.2, 1000)
    
    # Plot the MLE and unbiased PDFs
    pdf_mle = np.where((x >= 0) & (x <= theta_hat_mle), 1/theta_hat_mle, 0)
    pdf_unbiased = np.where((x >= 0) & (x <= theta_hat_unbiased), 1/theta_hat_unbiased, 0)
    
    ax.plot(x, pdf_mle, 'r-', linewidth=2, 
            label=f'MLE PDF: Uniform(0, {theta_hat_mle:.2f})')
    ax.plot(x, pdf_unbiased, 'g-', linewidth=2, 
            label=f'Unbiased PDF: Uniform(0, {theta_hat_unbiased:.2f})')
    
    # Plot histogram of the data
    ax.hist(data, bins='auto', density=True, alpha=0.3, color='blue',
            label=f'Exercise Duration Data (n={n})')
    
    # Mark the data points
    ax.plot(data, np.zeros_like(data), 'bo', markersize=8, alpha=0.6,
            label='Observed Exercise Durations')
    
    # Mark the estimators
    ax.axvline(x=theta_hat_mle, color='r', linestyle='--')
    ax.axvline(x=theta_hat_unbiased, color='g', linestyle='--')
    
    ax.set_xlabel('Exercise Duration (minutes)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Uniform Distribution: MLE vs. Unbiased Estimator')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add text with comparison
    bias = (n / (n + 1) - 1) * theta_hat_unbiased
    
    text_str = '\n'.join((
        f'MLE (θ̂): {theta_hat_mle:.2f}',
        f'Unbiased Estimator: {theta_hat_unbiased:.2f}',
        f'Formula: θ̂_unbiased = θ̂_MLE × (n+1)/n',
        f'Bias of MLE: {bias:.4f}',
        f'Difference: {theta_hat_unbiased - theta_hat_mle:.4f}'))
    
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return theta_hat_mle, theta_hat_unbiased

def plot_bias_vs_sample_size(true_theta=25, max_n=50, n_simulations=1000, save_path=None):
    """Plot bias of MLE vs sample size through simulation"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_values = np.arange(2, max_n+1)
    biases_mle = []
    biases_unbiased = []
    mses_mle = []
    mses_unbiased = []
    
    np.random.seed(42)
    
    for n in n_values:
        mle_estimates = []
        unbiased_estimates = []
        
        for _ in range(n_simulations):
            # Generate sample from uniform(0, true_theta)
            sample = np.random.uniform(0, true_theta, n)
            
            # Calculate MLE and unbiased estimator
            theta_hat_mle = np.max(sample)
            theta_hat_unbiased = theta_hat_mle * (n + 1) / n
            
            mle_estimates.append(theta_hat_mle)
            unbiased_estimates.append(theta_hat_unbiased)
        
        # Calculate bias and MSE
        bias_mle = np.mean(mle_estimates) - true_theta
        bias_unbiased = np.mean(unbiased_estimates) - true_theta
        
        mse_mle = np.mean((np.array(mle_estimates) - true_theta)**2)
        mse_unbiased = np.mean((np.array(unbiased_estimates) - true_theta)**2)
        
        biases_mle.append(bias_mle)
        biases_unbiased.append(bias_unbiased)
        mses_mle.append(mse_mle)
        mses_unbiased.append(mse_unbiased)
    
    # Plot bias vs sample size
    ax.plot(n_values, biases_mle, 'r-', linewidth=2, label='MLE Bias')
    ax.plot(n_values, biases_unbiased, 'g-', linewidth=2, label='Unbiased Estimator Bias')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Sample Size (n)')
    ax.set_ylabel('Bias (E[θ̂] - θ)')
    ax.set_title(f'Bias of Estimators vs Sample Size (True θ = {true_theta})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    # Create a second plot for MSE
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(n_values, mses_mle, 'r-', linewidth=2, label='MLE MSE')
    ax.plot(n_values, mses_unbiased, 'g-', linewidth=2, label='Unbiased Estimator MSE')
    
    ax.set_xlabel('Sample Size (n)')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title(f'MSE of Estimators vs Sample Size (True θ = {true_theta})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        mse_save_path = save_path.replace('.png', '_mse.png')
        plt.savefig(mse_save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {mse_save_path}")
    
    plt.close()
    
    # Calculate theoretical bias and MSE
    theoretical_bias = -true_theta / (n_values + 1)
    theoretical_mse_mle = true_theta**2 * n_values / ((n_values + 1)**2 * (n_values + 2))
    
    return biases_mle, biases_unbiased, theoretical_bias

def main():
    """Generate all visualizations for Question 19 of the L2.4 quiz"""
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_4_Quiz_19")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 19 of the L2.4 MLE quiz...")
    
    # Exercise duration data from the problem (in minutes)
    durations = np.array([15, 22, 18, 25, 20, 17, 23, 19])
    
    # 1. Plot uniform PDF for different theta values
    plot_uniform_pdf([15, 20, 25, 30], 
                    save_path=os.path.join(save_dir, "uniform_pdf.png"))
    print("1. Uniform PDF visualization created")
    
    # 2. Compute MLE and visualize likelihood function
    theta_hat, mean_hat, var_hat = compute_mle_for_uniform(
        durations, save_path=os.path.join(save_dir, "likelihood_function.png"))
    print("2. Likelihood function visualization created")
    print(f"   MLE for maximum exercise duration (θ): {theta_hat:.2f} minutes")
    print(f"   Estimated mean (θ/2): {mean_hat:.2f} minutes")
    print(f"   Estimated variance (θ²/12): {var_hat:.2f}")
    
    # 3. Compute unbiased estimator and compare with MLE
    theta_hat_mle, theta_hat_unbiased = compute_unbiased_estimator(
        durations, save_path=os.path.join(save_dir, "unbiased_estimator.png"))
    print("3. Unbiased estimator visualization created")
    print(f"   MLE (θ̂): {theta_hat_mle:.2f} minutes")
    print(f"   Unbiased Estimator: {theta_hat_unbiased:.2f} minutes")
    print(f"   Formula: θ̂_unbiased = θ̂_MLE × (n+1)/n = {theta_hat_mle:.2f} × ({len(durations)}+1)/{len(durations)}")
    
    # 4. Plot bias vs sample size
    biases_mle, biases_unbiased, theoretical_bias = plot_bias_vs_sample_size(
        true_theta=25, max_n=50, n_simulations=1000, 
        save_path=os.path.join(save_dir, "bias_vs_sample_size.png"))
    print("4. Bias vs sample size visualization created")
    
    # Calculate theoretical bias for n=8
    n = len(durations)
    theoretical_bias_n8 = -25 / (n + 1)
    print(f"   Theoretical bias for n={n}: {theoretical_bias_n8:.4f}")
    print(f"   Expected MLE value for n={n}: {25 + theoretical_bias_n8:.4f}")
    
    # 5. Print formulas for reference
    print("\nFormulas for Reference:")
    print(f"   MLE: θ̂_MLE = max(x_1, x_2, ..., x_n)")
    print(f"   Unbiased Estimator: θ̂_unbiased = θ̂_MLE × (n+1)/n")
    print(f"   Bias of MLE: E[θ̂_MLE] - θ = -θ/(n+1)")
    print(f"   Theoretical MSE of MLE: θ² × n / ((n+1)² × (n+2))")
    
    print(f"\nAll visualizations have been saved to: {save_dir}")

if __name__ == "__main__":
    main() 