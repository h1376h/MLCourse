import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon
import os
from matplotlib import cm
from scipy.optimize import minimize

def load_mystery_data():
    """Generate the same mystery data as in the question script"""
    np.random.seed(42)
    return np.random.exponential(scale=1/0.5, size=50)

def step1_demonstrate_data_and_mle(save_dir=None):
    """
    Step 1: Show the original data and explain the MLE formula for exponential distribution
    """
    # Load data
    data = load_mystery_data()
    mean = np.mean(data)
    rate_mle = 1 / mean
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram of original data
    counts, bins, _ = plt.hist(data, bins=20, density=True, alpha=0.6, color='skyblue', 
                            edgecolor='black', label='Observed Data')
    
    # Plot the PDF with MLE parameter
    x = np.linspace(0, max(data) * 1.2, 1000)
    pdf = expon.pdf(x, scale=1/rate_mle)
    plt.plot(x, pdf, 'r-', linewidth=2, label=f'Fitted Exponential PDF')
    
    # Add title and labels - clean and simple
    plt.title("Step 1: Data Exploration and MLE Estimation", fontsize=14)
    plt.xlabel("Waiting Time (x)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Add basic statistics - minimal text on the plot
    plt.text(0.02, 0.95, f"Sample Mean: {mean:.2f}\nMLE Rate: {rate_mle:.3f}", 
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'ex5_step1_data_and_mle.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Step 1 visualization saved to {save_path}")
    
    plt.close()
    
    return mean, rate_mle

def step2_demonstrate_loglikelihood(data, save_dir=None):
    """
    Step 2: Show the log-likelihood function and the sufficiency of sample mean
    """
    mean = np.mean(data)
    
    # Create clean, simple plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define range for rate parameter
    lambda_range = np.linspace(0.1, 1.0, 1000)
    
    # Calculate and plot the log-likelihood
    log_lik = [np.sum(np.log(lam) - lam * data) for lam in lambda_range]
    ax.plot(lambda_range, log_lik, 'b-', linewidth=2, label='Log-Likelihood')
    
    # Mark the MLE
    max_idx = np.argmax(log_lik)
    max_lambda = lambda_range[max_idx]
    ax.axvline(max_lambda, color='red', linestyle='--', linewidth=2, label=f'MLE λ={max_lambda:.3f}')
    
    # Simple vertical line at 1/mean (theoretical MLE)
    theoretical_mle = 1/mean
    ax.axvline(theoretical_mle, color='green', linestyle=':', linewidth=2, label=f'Theoretical 1/mean={theoretical_mle:.3f}')
    
    # Add title and labels - clean and simple
    ax.set_title("Step 2: Log-Likelihood Function", fontsize=14)
    ax.set_xlabel("Rate Parameter (λ)", fontsize=12)
    ax.set_ylabel("Log-Likelihood", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Save if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'ex5_step2_log_likelihood.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Step 2 visualization saved to {save_path}")
    
    plt.close()

def step3_demonstrate_sufficient_statistics(save_dir=None):
    """
    Step 3: Create datasets with same mean but different distributions
    to show how sample mean is a sufficient statistic
    """
    # Load the original data
    data = load_mystery_data()
    original_mean = np.mean(data)
    n = len(data)
    
    # Create different distributions with the same mean
    np.random.seed(43)  # Different seed for reproducibility
    
    # Create uniform data with the same mean
    uniform_data = np.random.uniform(0, original_mean*2, n)
    # Adjust to get the exact same mean
    uniform_data = uniform_data * (original_mean / np.mean(uniform_data))
    
    # Create bimodal data with the same mean
    bimodal_data = np.zeros(n)
    half = n // 2
    bimodal_data[:half] = original_mean * 0.5  # First peak
    bimodal_data[half:] = original_mean * 1.5  # Second peak
    # Add some noise
    bimodal_data += np.random.normal(0, original_mean * 0.1, n)
    # Adjust to get the exact same mean
    bimodal_data = bimodal_data * (original_mean / np.mean(bimodal_data))
    
    # Create gamma data with the same mean
    gamma_shape = 2.0
    gamma_scale = original_mean / gamma_shape
    gamma_data = np.random.gamma(gamma_shape, gamma_scale, n)
    # Adjust to get the exact same mean
    gamma_data = gamma_data * (original_mean / np.mean(gamma_data))
    
    # Organize datasets
    datasets = {
        'Original': data,
        'Uniform': uniform_data,
        'Bimodal': bimodal_data,
        'Gamma': gamma_data
    }
    
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    
    # Plot histograms
    colors = ['blue', 'green', 'red', 'purple']
    
    for i, (name, dataset) in enumerate(datasets.items()):
        # Calculate actual mean
        actual_mean = np.mean(dataset)
        
        # Plot histogram
        axs[i].hist(dataset, bins=15, alpha=0.7, color=colors[i], density=True,
                   edgecolor='black')
        
        axs[i].set_title(f"{name} Distribution (mean={actual_mean:.3f})", fontsize=12)
        axs[i].set_xlabel("Value", fontsize=10)
        axs[i].set_ylabel("Density", fontsize=10)
        axs[i].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'ex5_step3_different_distributions.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Step 3 visualization saved to {save_path}")
    
    plt.close()
    
    return datasets

def step4_compare_likelihoods(datasets, save_dir=None):
    """
    Step 4: Compare likelihood functions for different datasets with same mean
    """
    # Create clean, focused plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define rate parameter range
    lambda_range = np.linspace(0.1, 1.0, 100)
    
    # Plot log-likelihood for each dataset
    colors = ['blue', 'green', 'red', 'purple']
    
    for i, (name, dataset) in enumerate(datasets.items()):
        # Calculate log-likelihood
        log_lik = [np.sum(np.log(lam) - lam * dataset) for lam in lambda_range]
        
        # Plot
        ax.plot(lambda_range, log_lik, color=colors[i], linewidth=2, label=f'{name}')
        
        # Mark MLE
        max_idx = np.argmax(log_lik)
        max_lambda = lambda_range[max_idx]
        ax.axvline(max_lambda, color=colors[i], linestyle='--', alpha=0.5)
        
        # Add text for MLE value
        ax.text(max_lambda, np.max(log_lik) - 5*i, f'λ={max_lambda:.3f}', 
               color=colors[i], fontsize=9, ha='center',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Clean, focused labels
    ax.set_title("Step 4: Log-Likelihood Functions Comparison", fontsize=14)
    ax.set_xlabel("Rate Parameter (λ)", fontsize=12)
    ax.set_ylabel("Log-Likelihood", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Save if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'ex5_step4_likelihood_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Step 4 visualization saved to {save_path}")
    
    plt.close()

def step5_analytical_mle(datasets, save_dir=None):
    """
    Step 5: Demonstrate analytical MLE vs computed MLE
    """
    # Create clean, focused plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate mean and MLE for each dataset
    names = []
    means = []
    mle_estimates = []
    
    # Define rate parameter range
    lambda_range = np.linspace(0.1, 1.0, 1000)
    
    for name, dataset in datasets.items():
        # Calculate mean
        mean = np.mean(dataset)
        means.append(mean)
        
        # Calculate MLE from likelihood function
        log_lik = [np.sum(np.log(lam) - lam * dataset) for lam in lambda_range]
        max_idx = np.argmax(log_lik)
        mle = lambda_range[max_idx]
        mle_estimates.append(mle)
        
        names.append(name)
    
    # Plot as bar charts - simple visualization
    x = np.arange(len(names))
    width = 0.35
    
    # Plot means
    analytical_mle = [1/m for m in means]
    
    # Plot comparison
    ax.bar(x - width/2, analytical_mle, width, label='Analytical MLE (1/mean)', color='steelblue')
    ax.bar(x + width/2, mle_estimates, width, label='Numerical MLE', color='lightcoral')
    
    # Add labels
    ax.set_title("Step 5: Analytical vs. Numerical MLE", fontsize=14)
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Rate Parameter (λ)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add data labels
    for i in range(len(names)):
        ax.text(i - width/2, analytical_mle[i], f'{analytical_mle[i]:.3f}', 
               ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, mle_estimates[i], f'{mle_estimates[i]:.3f}', 
               ha='center', va='bottom', fontsize=9)
    
    # Save if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'ex5_step5_analytical_vs_numerical.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Step 5 visualization saved to {save_path}")
    
    plt.close()

def generate_answer_materials():
    """Generate all materials for the MLE visual answer with step-by-step approach"""
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    answer_dir = os.path.join(images_dir, "MLE_Visual_Answer")
    os.makedirs(answer_dir, exist_ok=True)
    
    print("Generating MLE Visual Answer Example 5 materials - Step by Step Solution...")
    
    # Load data
    data = load_mystery_data()
    
    # Step 1: Data and MLE formula
    mean, rate_mle = step1_demonstrate_data_and_mle(save_dir=answer_dir)
    
    # Step 2: Log-likelihood function
    step2_demonstrate_loglikelihood(data, save_dir=answer_dir)
    
    # Step 3: Create different distributions with same mean
    datasets = step3_demonstrate_sufficient_statistics(save_dir=answer_dir)
    
    # Step 4: Compare likelihood functions
    step4_compare_likelihoods(datasets, save_dir=answer_dir)
    
    # Step 5: Analytical vs numerical MLE
    step5_analytical_mle(datasets, save_dir=answer_dir)
    
    print(f"Step-by-step answer materials saved to: {answer_dir}")

if __name__ == "__main__":
    # Generate the answer materials
    generate_answer_materials() 