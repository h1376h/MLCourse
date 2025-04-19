import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import bernoulli
from matplotlib import cm

def generate_bernoulli_data(p=0.7, sample_sizes=[10, 20, 50, 100], random_seed=42):
    """
    Generate Bernoulli data for different sample sizes with the same true parameter p.
    This is for visualizing how the likelihood function changes with sample size.
    """
    np.random.seed(random_seed)
    
    # Dictionary to store results for each sample size
    data_dict = {}
    
    for n in sample_sizes:
        # Generate Bernoulli sample of size n
        data = np.random.binomial(1, p, n)
        
        # Calculate sufficient statistic (sum of successes)
        sum_x = np.sum(data)
        
        # Store data and statistics
        data_dict[n] = {
            'data': data,
            'n': n,
            'sum_x': sum_x,
            'p_mle': sum_x / n
        }
    
    return data_dict

def visualize_bernoulli_data(data_dict, save_path=None):
    """
    Visualize the Bernoulli data for different sample sizes
    """
    sample_sizes = sorted(list(data_dict.keys()))
    
    # Calculate the number of rows needed
    n_rows = (len(sample_sizes) + 1) // 2
    
    fig, axs = plt.subplots(n_rows, 2, figsize=(10, 3 * n_rows))
    if n_rows == 1:
        axs = np.array([axs])  # Make sure axs is a 2D array
    
    # Flatten for easy indexing
    axs = axs.flatten()
    
    for i, n in enumerate(sample_sizes):
        data = data_dict[n]['data']
        p_mle = data_dict[n]['p_mle']
        
        # Count successes and failures
        successes = np.sum(data)
        failures = n - successes
        
        # Create bar chart
        axs[i].bar([0, 1], [failures, successes], color=['lightcoral', 'lightgreen'])
        
        # Customize the plot
        axs[i].set_title(f'Bernoulli Sample (n={n}, p_MLE={p_mle:.3f})', fontsize=12)
        axs[i].set_xticks([0, 1])
        axs[i].set_xticklabels(['Failure (0)', 'Success (1)'])
        axs[i].set_ylabel('Count')
        
        # Add text annotations
        axs[i].text(0, failures/2, f"{failures}", ha='center', va='center', fontweight='bold')
        axs[i].text(1, successes/2, f"{successes}", ha='center', va='center', fontweight='bold')
        
        # Add horizontal line for the expected count
        true_p = 0.7  # The parameter used to generate data
        expected_successes = n * true_p
        expected_failures = n * (1 - true_p)
        axs[i].axhline(y=expected_successes, color='green', linestyle='--', 
                      label=f'Expected successes: {expected_successes:.1f}')
        axs[i].axhline(y=expected_failures, color='red', linestyle='--', 
                      label=f'Expected failures: {expected_failures:.1f}')
        
        axs[i].legend(fontsize=9)
    
    # Hide any unused subplots
    for j in range(i+1, len(axs)):
        axs[j].axis('off')
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Bernoulli data visualization saved to {save_path}")
    
    plt.close()

def plot_likelihood_functions(data_dict, save_path=None):
    """
    Plot likelihood and log-likelihood functions for Bernoulli MLE
    with different sample sizes and observed proportions
    """
    sample_sizes = sorted(list(data_dict.keys()))
    colors = ['blue', 'green', 'red', 'purple']
    
    # Create figure with 2 rows: likelihood and log-likelihood
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    
    # Fine grid for parameter values
    p_values = np.linspace(0.01, 0.99, 500)
    
    # Plot likelihood functions
    for i, n in enumerate(sample_sizes):
        sum_x = data_dict[n]['sum_x']
        p_mle = data_dict[n]['p_mle']
        
        # Calculate likelihood: L(p) = p^sum_x * (1-p)^(n-sum_x)
        likelihood = p_values**sum_x * (1-p_values)**(n-sum_x)
        
        # Scale the likelihood to have maximum of 1 for better visualization
        likelihood = likelihood / np.max(likelihood)
        
        # Plot the likelihood function
        axs[0].plot(p_values, likelihood, color=colors[i], linewidth=2, 
                  label=f'n={n}, sum_x={sum_x}, p_MLE={p_mle:.2f}')
        
        # Mark the MLE
        axs[0].plot(p_mle, 1.0, 'o', color=colors[i], markersize=8)
        
        # Add vertical line at the MLE
        axs[0].axvline(x=p_mle, color=colors[i], linestyle='--', alpha=0.5)
    
    # Add vertical line at the true parameter value
    true_p = 0.7
    axs[0].axvline(x=true_p, color='black', linestyle='-', linewidth=2, 
                 label='True p=0.7')
    
    # Customize the likelihood plot
    axs[0].set_title('Likelihood Function L(p) for Bernoulli MLE', fontsize=14)
    axs[0].set_xlabel('p (probability parameter)', fontsize=12)
    axs[0].set_ylabel('Normalized Likelihood', fontsize=12)
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(fontsize=10)
    
    # Add annotation about likelihood interpretation
    axs[0].text(0.05, 0.5, 
               "Likelihood Function:\n" +
               "L(p) = p^(sum_x) * (1-p)^(n-sum_x)\n\n" +
               "• Peak gives the MLE: p̂ = sum_x/n\n" +
               "• Curve gets narrower with larger n\n" +
               "• Sharpness indicates precision",
               transform=axs[0].transAxes, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Plot log-likelihood functions
    for i, n in enumerate(sample_sizes):
        sum_x = data_dict[n]['sum_x']
        p_mle = data_dict[n]['p_mle']
        
        # Calculate log-likelihood: log[L(p)] = sum_x*log(p) + (n-sum_x)*log(1-p)
        log_likelihood = sum_x * np.log(p_values) + (n-sum_x) * np.log(1-p_values)
        
        # Normalize to have maximum of 0 for better visualization
        log_likelihood = log_likelihood - np.max(log_likelihood)
        
        # Plot the log-likelihood function
        axs[1].plot(p_values, log_likelihood, color=colors[i], linewidth=2, 
                  label=f'n={n}, sum_x={sum_x}, p_MLE={p_mle:.2f}')
        
        # Mark the MLE
        axs[1].plot(p_mle, 0.0, 'o', color=colors[i], markersize=8)
        
        # Add vertical line at the MLE
        axs[1].axvline(x=p_mle, color=colors[i], linestyle='--', alpha=0.5)
    
    # Add vertical line at the true parameter value
    axs[1].axvline(x=true_p, color='black', linestyle='-', linewidth=2, 
                 label='True p=0.7')
    
    # Customize the log-likelihood plot
    axs[1].set_title('Log-Likelihood Function log[L(p)] for Bernoulli MLE', fontsize=14)
    axs[1].set_xlabel('p (probability parameter)', fontsize=12)
    axs[1].set_ylabel('Normalized Log-Likelihood', fontsize=12)
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(fontsize=10)
    
    # Add annotation about log-likelihood advantages
    axs[1].text(0.05, 0.5, 
               "Log-Likelihood Function:\n" +
               "log[L(p)] = sum_x*log(p) + (n-sum_x)*log(1-p)\n\n" +
               "• Maximum gives same MLE as likelihood\n" +
               "• Computational advantages for optimization\n" +
               "• Steepness related to Fisher Information",
               transform=axs[1].transAxes, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Likelihood functions plot saved to {save_path}")
    
    plt.close()

def generate_sampling_distribution(true_p=0.7, sample_sizes=[10, 20, 50, 100], 
                                 n_simulations=10000, random_seed=42):
    """
    Generate the sampling distribution of the MLE for a Bernoulli distribution
    with different sample sizes to visualize its key properties.
    """
    np.random.seed(random_seed)
    
    # Dictionary to store results
    results = {}
    
    for n in sample_sizes:
        # Initialize array to store MLE estimates
        p_mle_estimates = np.zeros(n_simulations)
        
        # Generate simulations and compute MLEs
        for i in range(n_simulations):
            # Generate a sample of size n
            sample = np.random.binomial(1, true_p, n)
            
            # Calculate MLE (sample proportion)
            p_mle = np.sum(sample) / n
            
            # Store the estimate
            p_mle_estimates[i] = p_mle
        
        # Store results for this sample size
        results[n] = {
            'p_mle_estimates': p_mle_estimates,
            'mean': np.mean(p_mle_estimates),
            'std': np.std(p_mle_estimates),
            'theoretical_std': np.sqrt(true_p * (1-true_p) / n)  # Theoretical SE
        }
    
    return results

def generate_mle_visual_question():
    """Generate all the visual materials for the Bernoulli MLE question"""
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images", "MLE_Visual_Question")
    os.makedirs(images_dir, exist_ok=True)
    
    print("Generating MLE Visual Question materials (Bernoulli Distribution)...")
    
    # True parameter used for data generation
    true_p = 0.7
    sample_sizes = [10, 20, 50, 100]
    
    # 1. Generate Bernoulli data for different sample sizes
    data_dict = generate_bernoulli_data(
        p=true_p, 
        sample_sizes=sample_sizes,
        random_seed=42
    )
    
    # 2. Visualize the Bernoulli data
    data_viz_path = os.path.join(images_dir, "ex4_bernoulli_data.png")
    visualize_bernoulli_data(data_dict, save_path=data_viz_path)
    
    # 3. Plot likelihood functions
    likelihood_plot_path = os.path.join(images_dir, "ex4_bernoulli_likelihood.png")
    plot_likelihood_functions(data_dict, save_path=likelihood_plot_path)
    
    # 4. Generate and visualize sampling distribution
    sampling_results = generate_sampling_distribution(
        true_p=true_p,
        sample_sizes=sample_sizes,
        n_simulations=10000
    )
    
    print("MLE Visual Question materials (Bernoulli Distribution) generated successfully!")
    
    return data_dict, sampling_results

if __name__ == "__main__":
    # Generate all the visual materials
    data_dict, sampling_results = generate_mle_visual_question() 