import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon
import os
from matplotlib import cm
from scipy.optimize import minimize

def generate_mystery_data(sample_size=50, rate_param=0.5, random_seed=42):
    """Generate exponential distributed data with given rate parameter"""
    np.random.seed(random_seed)
    # Generate from exponential distribution with scale = 1/rate
    return np.random.exponential(scale=1/rate_param, size=sample_size)

def plot_exponential_histogram(data, save_path=None):
    """Plot a histogram of the exponential data"""
    plt.figure(figsize=(10, 6))
    
    # Plot histogram
    counts, bins, _ = plt.hist(data, bins=20, density=True, alpha=0.6, color='skyblue', 
                            edgecolor='black', label='Observed Data')
    
    # Plot the PDF of the exponential distribution with true parameter
    x = np.linspace(0, max(data) * 1.2, 1000)
    rate_est = 1 / np.mean(data)  # MLE estimate
    pdf = expon.pdf(x, scale=1/rate_est)
    plt.plot(x, pdf, 'r-', linewidth=2, label=f'Exponential PDF (λ≈{rate_est:.3f})')
    
    # Add title and labels
    plt.title("Exponential Distribution Waiting Times", fontsize=14)
    plt.xlabel("Waiting Time (x)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Add annotation about sufficient statistics
    mean_value = np.mean(data)
    plt.text(0.02, 0.95, f"Sample Mean: {mean_value:.2f}\n"
                        f"Sample Size: {len(data)}", 
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Histogram saved to {save_path}")
    
    plt.close()

def visualize_likelihood_dependency(data, save_path=None):
    """
    Visualize how the likelihood function depends on the data through
    the sufficient statistic (sample mean)
    """
    plt.figure(figsize=(12, 8))
    
    # Create a 2x2 subplot grid
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    
    # Original data and its log-likelihood curve
    original_mean = np.mean(data)
    lambda_range = np.linspace(0.1, 1.0, 100)
    
    # Calculate log-likelihood for original data
    original_log_lik = [np.sum(np.log(lam) - lam * data) for lam in lambda_range]
    original_log_lik = np.array(original_log_lik)
    
    # Plot original data histogram in first subplot
    axs[0, 0].hist(data, bins=15, alpha=0.6, color='skyblue', density=True,
                 edgecolor='black', label='Original Data')
    axs[0, 0].set_title(f"Original Data (mean={original_mean:.3f})", fontsize=12)
    axs[0, 0].set_xlabel("Value", fontsize=10)
    axs[0, 0].set_ylabel("Density", fontsize=10)
    axs[0, 0].grid(alpha=0.3)
    
    # Plot original log-likelihood curve in second subplot
    axs[0, 1].plot(lambda_range, original_log_lik, 'b-', linewidth=2, label='Original Data')
    max_idx = np.argmax(original_log_lik)
    max_lambda = lambda_range[max_idx]
    axs[0, 1].axvline(max_lambda, color='blue', linestyle='--', alpha=0.7,
                    label=f'MLE λ={max_lambda:.3f}')
    axs[0, 1].set_title("Log-Likelihood Function", fontsize=12)
    axs[0, 1].set_xlabel("Rate Parameter (λ)", fontsize=10)
    axs[0, 1].set_ylabel("Log-Likelihood", fontsize=10)
    axs[0, 1].grid(alpha=0.3)
    axs[0, 1].legend(fontsize=9)
    
    # Generate alternative datasets with same mean but different distributions
    def generate_alternative_data(original_data, distribution_type):
        """Generate alternative data with same mean but different distribution"""
        n = len(original_data)
        mean = np.mean(original_data)
        
        if distribution_type == 'uniform':
            # Uniform distribution with same mean
            a = mean * 0.5  # Lower bound
            b = mean * 1.5  # Upper bound
            # Adjust so that mean = (a+b)/2 = original_mean
            center = (a + b) / 2
            shift = original_mean - center
            a += shift
            b += shift
            return np.random.uniform(a, b, n)
        
        elif distribution_type == 'bimodal':
            # Bimodal distribution with same mean
            mode1 = mean * 0.5
            mode2 = mean * 1.5
            # Adjust weights to ensure same mean
            p = (mean - mode2) / (mode1 - mode2)
            # Sample from the two modes
            samples = np.random.choice([0, 1], size=n, p=[p, 1-p])
            # Add some noise around each mode
            noise = np.random.normal(0, mean * 0.1, n)
            return samples * mode1 + (1 - samples) * mode2 + noise
    
    # Generate alternative datasets
    uniform_data = generate_alternative_data(data, 'uniform')
    bimodal_data = generate_alternative_data(data, 'bimodal')
    
    # Verify all datasets have approximately the same mean
    uniform_mean = np.mean(uniform_data)
    bimodal_mean = np.mean(bimodal_data)
    
    # Fine-tune to get very close to same mean (optional)
    uniform_data = uniform_data * (original_mean / uniform_mean)
    bimodal_data = bimodal_data * (original_mean / bimodal_mean)
    
    # Update means after adjustment
    uniform_mean = np.mean(uniform_data)
    bimodal_mean = np.mean(bimodal_data)
    
    # Calculate log-likelihood for alternative datasets
    uniform_log_lik = [np.sum(np.log(lam) - lam * uniform_data) for lam in lambda_range]
    bimodal_log_lik = [np.sum(np.log(lam) - lam * bimodal_data) for lam in lambda_range]
    
    # Plot alternative datasets in bottom row
    axs[1, 0].hist(uniform_data, bins=15, alpha=0.6, color='green', density=True,
                 edgecolor='black', label='Uniform-like Data')
    axs[1, 0].set_title(f"Uniform-like Data (mean={uniform_mean:.3f})", fontsize=12)
    axs[1, 0].set_xlabel("Value", fontsize=10)
    axs[1, 0].set_ylabel("Density", fontsize=10)
    axs[1, 0].grid(alpha=0.3)
    
    axs[1, 1].hist(bimodal_data, bins=15, alpha=0.6, color='red', density=True,
                 edgecolor='black', label='Bimodal-like Data')
    axs[1, 1].set_title(f"Bimodal-like Data (mean={bimodal_mean:.3f})", fontsize=12)
    axs[1, 1].set_xlabel("Value", fontsize=10)
    axs[1, 1].set_ylabel("Density", fontsize=10)
    axs[1, 1].grid(alpha=0.3)
    
    # Add log-likelihood curves for all datasets to one plot
    axs[0, 1].plot(lambda_range, np.array(uniform_log_lik), 'g-', linewidth=2, label='Uniform-like Data')
    axs[0, 1].plot(lambda_range, np.array(bimodal_log_lik), 'r-', linewidth=2, label='Bimodal-like Data')
    
    # Overlay the maxima for all three
    max_idx_uniform = np.argmax(uniform_log_lik)
    max_lambda_uniform = lambda_range[max_idx_uniform]
    axs[0, 1].axvline(max_lambda_uniform, color='green', linestyle='--', alpha=0.7,
                    label=f'MLE λ={max_lambda_uniform:.3f}')
    
    max_idx_bimodal = np.argmax(bimodal_log_lik)
    max_lambda_bimodal = lambda_range[max_idx_bimodal]
    axs[0, 1].axvline(max_lambda_bimodal, color='red', linestyle='--', alpha=0.7,
                    label=f'MLE λ={max_lambda_bimodal:.3f}')
    
    # Add annotation about sufficient statistics
    axs[0, 1].text(0.1, 0.02, 
                 "Notice: Despite having different distributions,\n"
                 "datasets with the same mean produce nearly identical\n"
                 "likelihood functions and MLE estimates for λ",
                 transform=axs[0, 1].transAxes, fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Update legend
    axs[0, 1].legend(fontsize=8, loc='upper right')
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Likelihood dependency visualization saved to {save_path}")
    
    plt.close()

def generate_mle_visual_question():
    """Generate all materials for the MLE visual question"""
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    question_dir = os.path.join(images_dir, "MLE_Visual_Question")
    os.makedirs(question_dir, exist_ok=True)
    
    print("Generating MLE Visual Question Example 5 materials...")
    
    # Generate exponential data
    np.random.seed(42)  # Ensure reproducibility
    data = generate_mystery_data(sample_size=50, rate_param=0.5)
    
    # Generate histogram
    hist_path = os.path.join(question_dir, "ex5_exponential_histogram.png")
    plot_exponential_histogram(data, save_path=hist_path)
    
    # Generate likelihood dependency visualization
    dep_path = os.path.join(question_dir, "ex5_likelihood_dependency.png")
    visualize_likelihood_dependency(data, save_path=dep_path)
    
    print(f"Question materials saved to: {question_dir}")

if __name__ == "__main__":
    # Generate the question materials
    generate_mle_visual_question() 