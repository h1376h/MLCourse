import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import os

def plot_beta_pdfs(alphas, betas, save_path=None):
    """Plot beta PDFs for different parameter values"""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(0, 1, 1000)
    
    for alpha, beta_val in zip(alphas, betas):
        y = beta.pdf(x, alpha, beta_val)
        ax.plot(x, y, label=f'α = {alpha:.2f}, β = {beta_val:.2f}')
    
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density f(x|α,β)')
    ax.set_title('Beta Distribution PDFs for Different Parameter Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def method_of_moments(data):
    """
    Estimate Beta distribution parameters using method of moments
    Formulas:
    α = μ*(μ*(1-μ)/σ² - 1)
    β = (1-μ)*(μ*(1-μ)/σ² - 1)
    """
    mean = np.mean(data)
    var = np.var(data, ddof=1)  # sample variance
    
    # Calculate the common factor in both formulas
    common_factor = mean * (1 - mean) / var - 1
    
    # Calculate alpha and beta
    alpha = mean * common_factor
    beta_val = (1 - mean) * common_factor
    
    return alpha, beta_val

def plot_data_and_fit(data, save_path=None):
    """Plot the data and fitted Beta distribution"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate method of moments estimates
    alpha, beta_val = method_of_moments(data)
    
    # Generate x values for plotting
    x = np.linspace(0, 1, 1000)
    y_fit = beta.pdf(x, alpha, beta_val)
    
    # Plot histogram of the data
    ax.hist(data, bins=10, density=True, alpha=0.5, color='blue', 
            label='CTR Data')
    
    # Plot the fitted PDF
    ax.plot(x, y_fit, 'r-', linewidth=2, 
            label=f'Fitted Beta(α={alpha:.2f}, β={beta_val:.2f})')
    
    # Mark the observed data points
    ax.plot(data, np.zeros_like(data), 'bo', markersize=8, alpha=0.6)
    
    ax.set_xlabel('Click-Through Rate')
    ax.set_ylabel('Probability Density')
    ax.set_title('Method of Moments Estimation for Beta Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return alpha, beta_val

def plot_beta_cdf(alpha, beta_val, threshold=0.15, save_path=None):
    """Plot the CDF of the Beta distribution with the probability of exceeding threshold"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate x values for plotting
    x = np.linspace(0, 1, 1000)
    cdf_values = beta.cdf(x, alpha, beta_val)
    
    # Plot the CDF
    ax.plot(x, cdf_values, 'b-', linewidth=2, label='CDF')
    
    # Highlight the area above the threshold
    idx = np.where(x >= threshold)[0][0]
    ax.fill_between(x[idx:], 0, cdf_values[idx:], alpha=0.3, color='red')
    
    # Calculate the probability of exceeding threshold
    exceed_prob = 1 - beta.cdf(threshold, alpha, beta_val)
    
    # Add a vertical line at the threshold
    ax.axvline(x=threshold, color='r', linestyle='--', 
               label=f'Threshold = {threshold}')
    
    ax.set_xlabel('Click-Through Rate')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title(f'Beta Distribution CDF\nP(CTR > {threshold}) = {exceed_prob:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add an annotation for the probability
    ax.annotate(f'P(CTR > {threshold}) = {exceed_prob:.4f}',
                xy=(threshold, beta.cdf(threshold, alpha, beta_val)),
                xytext=(threshold+0.1, beta.cdf(threshold, alpha, beta_val)-0.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return exceed_prob

def main():
    """Generate all visualizations for Question 13 of the L2.4 quiz"""
    # CTR data from the question
    data = np.array([0.12, 0.15, 0.11, 0.14, 0.13, 0.16, 0.12, 0.14, 0.13, 0.15])
    
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_4_Quiz_13")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 13 of the L2.4 MLE quiz...")
    
    # 1. Plot PDFs for different parameter values
    plot_beta_pdfs([1, 2, 5, 10], [1, 2, 2, 5], 
                   save_path=os.path.join(save_dir, "beta_pdfs.png"))
    print("1. Beta PDFs visualization created")
    
    # 2. Plot data and fitted distribution
    alpha, beta_val = plot_data_and_fit(data, 
                                        save_path=os.path.join(save_dir, "data_fit.png"))
    print(f"2. Data fit visualization created, estimated parameters: α = {alpha:.4f}, β = {beta_val:.4f}")
    
    # 3. Calculate and display statistics
    mean = alpha / (alpha + beta_val)
    var = (alpha * beta_val) / ((alpha + beta_val)**2 * (alpha + beta_val + 1))
    print(f"3. Estimated mean CTR: {mean:.4f}")
    print(f"4. Estimated variance of CTR: {var:.6f}")
    
    # 4. Plot CDF and calculate probability of exceeding threshold
    exceed_prob = plot_beta_cdf(alpha, beta_val, threshold=0.15, 
                               save_path=os.path.join(save_dir, "beta_cdf.png"))
    print(f"5. CDF visualization created, P(CTR > 0.15) = {exceed_prob:.4f}")
    
    print(f"\nAll visualizations have been saved to: {save_dir}")
    print("\nSummary of results:")
    print(f"CTR Data: {data}")
    print(f"Sample Mean: {np.mean(data):.4f}")
    print(f"Sample Variance: {np.var(data, ddof=1):.6f}")
    print(f"Estimated Beta Parameters: α = {alpha:.4f}, β = {beta_val:.4f}")
    print(f"Estimated Mean from Beta: {mean:.4f}")
    print(f"Estimated Variance from Beta: {var:.6f}")
    print(f"Probability that CTR exceeds 0.15: {exceed_prob:.4f}")

if __name__ == "__main__":
    main() 