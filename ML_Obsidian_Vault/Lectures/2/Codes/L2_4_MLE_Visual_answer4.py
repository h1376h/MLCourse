import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import bernoulli, norm, beta

def load_bernoulli_data():
    """Load the same Bernoulli data used in the question for consistency"""
    np.random.seed(42)  # Same seed as question
    true_p = 0.7
    sample_sizes = [10, 20, 50, 100]
    
    # Dictionary to store results for each sample size
    data_dict = {}
    
    for n in sample_sizes:
        # Generate Bernoulli sample of size n
        data = np.random.binomial(1, true_p, n)
        
        # Calculate sufficient statistic (sum of successes)
        sum_x = np.sum(data)
        
        # Store data and statistics
        data_dict[n] = {
            'data': data,
            'n': n,
            'sum_x': sum_x,
            'p_mle': sum_x / n
        }
    
    return data_dict, true_p

def generate_sampling_distribution(true_p=0.7, sample_sizes=[10, 20, 50, 100], 
                                 n_simulations=10000, random_seed=42):
    """
    Generate the sampling distribution of the MLE for a Bernoulli distribution
    with different sample sizes to visualize its key properties.
    Same as in the question script but used in answer as well.
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
            'var': np.var(p_mle_estimates),
            'std': np.std(p_mle_estimates),
            'theoretical_var': true_p * (1-true_p) / n,  # Theoretical variance
            'theoretical_std': np.sqrt(true_p * (1-true_p) / n)  # Theoretical SE
        }
    
    return results

def plot_mle_sampling_distribution(sampling_results, true_p, save_path=None):
    """
    Create visualization of the sampling distribution of the Bernoulli MLE
    for different sample sizes, showing convergence to normal distribution.
    """
    sample_sizes = sorted(list(sampling_results.keys()))
    colors = ['blue', 'green', 'red', 'purple']
    
    # Create 2x2 grid for the four sample sizes
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    
    for i, n in enumerate(sample_sizes):
        # Get the MLE estimates for this sample size
        p_mle_estimates = sampling_results[n]['p_mle_estimates']
        mean = sampling_results[n]['mean']
        std = sampling_results[n]['std']
        theoretical_std = sampling_results[n]['theoretical_std']
        
        # Create histogram of the MLE estimates
        axs[i].hist(p_mle_estimates, bins=30, density=True, alpha=0.6, color=colors[i],
                  edgecolor='black', label=f'Empirical (n={n})')
        
        # Add normal approximation (using theoretical parameters)
        x = np.linspace(max(0, mean - 4*std), min(1, mean + 4*std), 1000)
        y_theory = norm.pdf(x, true_p, theoretical_std)
        axs[i].plot(x, y_theory, 'r--', linewidth=2, label='Theoretical Normal')
        
        # Add beta distribution (conjugate posterior with uniform prior)
        sum_x = int(np.round(true_p * n))  # Expected number of successes
        alpha = sum_x + 1  # Prior Beta(1,1) + data
        beta_param = n - sum_x + 1
        y_beta = beta.pdf(x, alpha, beta_param)
        axs[i].plot(x, y_beta, 'g-.', linewidth=2, label='Beta Distribution')
        
        # Mark the true parameter and mean of estimates
        axs[i].axvline(x=true_p, color='black', linestyle='-', linewidth=2, 
                     label=f'True p={true_p}')
        axs[i].axvline(x=mean, color=colors[i], linestyle='--', linewidth=2, 
                     label=f'Mean estimate={mean:.4f}')
        
        # Add annotation about standard error
        axs[i].text(0.05, 0.9, 
                  f"Theoretical SE: {theoretical_std:.4f}\n" +
                  f"Observed SE: {std:.4f}\n" +
                  f"Shapiro-Wilk p-value: {shapiro_p_value(p_mle_estimates):.4f}",
                  transform=axs[i].transAxes, fontsize=9,
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Customize the plot
        axs[i].set_title(f'Sampling Distribution of p̂ (n={n})', fontsize=12)
        axs[i].set_xlabel('p̂ (MLE Estimate)', fontsize=10)
        axs[i].set_ylabel('Density', fontsize=10)
        axs[i].grid(True, alpha=0.3)
        axs[i].legend(fontsize=8, loc='upper right')
    
    # Overall figure title
    plt.suptitle('Sampling Distribution of Bernoulli MLE with Different Sample Sizes', 
                fontsize=16, y=0.98)
    
    # Add explanation about asymptotic properties
    fig.text(0.5, 0.01, 
            "As sample size increases, the sampling distribution approaches a normal distribution\n" +
            f"with mean p={true_p} and variance p(1-p)/n = {true_p*(1-true_p):.2f}/n",
            ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sampling distribution plot saved to {save_path}")
    
    plt.close()

def shapiro_p_value(data):
    """Calculate Shapiro-Wilk test p-value to test for normality"""
    from scipy.stats import shapiro
    try:
        _, p_value = shapiro(data)
        return p_value
    except:
        return np.nan

def plot_mle_properties(sampling_results, true_p, save_path=None):
    """
    Create visualization demonstrating key MLE properties:
    1. Unbiasedness of the estimator
    2. Consistency (convergence to true value)
    3. Efficiency (variance decreases at optimal rate)
    """
    sample_sizes = sorted(list(sampling_results.keys()))
    
    # Extract relevant statistics
    means = [sampling_results[n]['mean'] for n in sample_sizes]
    stds = [sampling_results[n]['std'] for n in sample_sizes]
    theoretical_stds = [sampling_results[n]['theoretical_std'] for n in sample_sizes]
    
    # Calculate MSE and its components
    bias = [m - true_p for m in means]
    bias_squared = [b**2 for b in bias]
    variance = [s**2 for s in stds]
    mse = [b2 + v for b2, v in zip(bias_squared, variance)]
    
    # Create figure with 2 rows, 2 columns
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Bias - should be close to zero
    axs[0, 0].plot(sample_sizes, bias, 'bo-', linewidth=2, markersize=8)
    axs[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2, label='Unbiased (Bias=0)')
    
    # Add error bars showing standard error of the mean estimate
    se_of_mean = [s/np.sqrt(len(sampling_results[n]['p_mle_estimates'])) for n, s in zip(sample_sizes, stds)]
    axs[0, 0].errorbar(sample_sizes, bias, yerr=se_of_mean, fmt='none', ecolor='lightblue', 
                     capsize=5, elinewidth=2, label='95% CI for Bias')
    
    # Customize the bias plot
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_title('Bias of Bernoulli MLE', fontsize=12)
    axs[0, 0].set_xlabel('Sample Size (log scale)', fontsize=10)
    axs[0, 0].set_ylabel('Bias (p̂ - p)', fontsize=10)
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend(fontsize=9)
    
    # Add annotation about unbiasedness
    axs[0, 0].text(0.05, 0.05, 
                 "The MLE for Bernoulli is unbiased:\nE[p̂] = p\n" +
                 "All observed bias values are within\nsampling error of zero",
                 transform=axs[0, 0].transAxes, fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Plot 2: Variance and theoretical variance
    axs[0, 1].plot(sample_sizes, variance, 'go-', linewidth=2, markersize=8, 
                 label='Observed Variance')
    axs[0, 1].plot(sample_sizes, [true_p*(1-true_p)/n for n in sample_sizes], 'r--', 
                 linewidth=2, label=f'Theoretical: p(1-p)/n')
    
    # Customize the variance plot
    axs[0, 1].set_xscale('log')
    axs[0, 1].set_yscale('log')
    axs[0, 1].set_title('Variance of Bernoulli MLE', fontsize=12)
    axs[0, 1].set_xlabel('Sample Size (log scale)', fontsize=10)
    axs[0, 1].set_ylabel('Variance (log scale)', fontsize=10)
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend(fontsize=9)
    
    # Add annotation about efficiency
    axs[0, 1].text(0.05, 0.05, 
                 "The MLE for Bernoulli is efficient:\n" +
                 f"Var(p̂) = p(1-p)/n = {true_p*(1-true_p):.2f}/n\n" +
                 "Variance decreases at the optimal rate of 1/n",
                 transform=axs[0, 1].transAxes, fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Plot 3: MSE decomposition (Bias^2 + Variance)
    width = 0.35
    x = np.arange(len(sample_sizes))
    
    axs[1, 0].bar(x, bias_squared, width, label='Bias²', color='lightblue')
    axs[1, 0].bar(x, variance, width, bottom=bias_squared, label='Variance', color='lightgreen')
    
    # Customize the MSE plot
    axs[1, 0].set_title('MSE Decomposition: Bias² + Variance', fontsize=12)
    axs[1, 0].set_xlabel('Sample Size', fontsize=10)
    axs[1, 0].set_ylabel('MSE Components', fontsize=10)
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(sample_sizes)
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend(fontsize=9)
    
    # Add annotation about MSE
    axs[1, 0].text(0.05, 0.9, 
                 "MSE = Bias² + Variance\n" +
                 "For unbiased estimators like p̂,\nBias² ≈ 0 and MSE ≈ Variance",
                 transform=axs[1, 0].transAxes, fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Plot 4: Consistency demonstration (distribution narrowing)
    for i, n in enumerate(sample_sizes):
        # Get the MLE estimates for this sample size
        p_mle_estimates = sampling_results[n]['p_mle_estimates']
        
        # Create kernel density estimate
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(p_mle_estimates)
        x = np.linspace(max(0, true_p - 0.3), min(1, true_p + 0.3), 1000)
        y = kde(x)
        
        # Scale for better visualization
        y = y / np.max(y)
        
        # Plot the density
        axs[1, 1].plot(x, y, linewidth=2, label=f'n = {n}')
    
    # Add vertical line at true parameter value
    axs[1, 1].axvline(x=true_p, color='black', linestyle='--', linewidth=2, 
                    label=f'True p = {true_p}')
    
    # Customize the consistency plot
    axs[1, 1].set_title('Consistency: Distribution Narrows Around True Value', fontsize=12)
    axs[1, 1].set_xlabel('p̂ (MLE Estimate)', fontsize=10)
    axs[1, 1].set_ylabel('Scaled Density', fontsize=10)
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend(fontsize=9)
    
    # Add annotation about consistency
    axs[1, 1].text(0.05, 0.05, 
                 "The MLE for Bernoulli is consistent:\n" +
                 "As n increases, the distribution\n" +
                 f"concentrates around the true value p = {true_p}",
                 transform=axs[1, 1].transAxes, fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Overall figure title
    plt.suptitle('Bernoulli MLE Properties: Unbiasedness, Efficiency & Consistency', 
                fontsize=16, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"MLE properties plot saved to {save_path}")
    
    plt.close()

def plot_sufficient_statistic_demonstration(save_path=None):
    """
    Create a visualization demonstrating that the sample sum (or sample proportion)
    is a sufficient statistic for the Bernoulli parameter.
    """
    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot 1: Different datasets with the same sufficient statistic
    np.random.seed(42)
    
    # Sample size for demonstration
    n = 20
    
    # Target sum value (sufficient statistic)
    target_sum = 14  # This gives p_mle = 14/20 = 0.7
    
    # Generate 3 different datasets with the same sufficient statistic
    datasets = []
    for i in range(3):
        if i == 0:
            # First dataset: just use a random sample with the target sum
            while True:
                data = np.random.binomial(1, 0.7, n)
                if np.sum(data) == target_sum:
                    break
            datasets.append(data)
        else:
            # Other datasets: modify the first dataset while preserving the sum
            new_data = datasets[0].copy()
            # Swap some 0s and 1s to get a different dataset with same sum
            for j in range(5):  # Make 5 swaps
                idx_zero = np.random.choice(np.where(new_data == 0)[0])
                idx_one = np.random.choice(np.where(new_data == 1)[0])
                # Swap values
                new_data[idx_zero] = 1
                new_data[idx_one] = 0
            datasets.append(new_data)
    
    # Plot the datasets
    colors = ['blue', 'green', 'red']
    p_vals = np.linspace(0.01, 0.99, 500)
    
    for i, data in enumerate(datasets):
        # Calculate likelihood function
        sum_x = np.sum(data)
        likelihood = p_vals**sum_x * (1-p_vals)**(n-sum_x)
        likelihood = likelihood / np.max(likelihood)
        
        # Plot likelihood
        axs[0].plot(p_vals, likelihood, color=colors[i], linewidth=2, 
                  label=f'Dataset {i+1}')
        
        # Mark the MLE
        p_mle = sum_x / n
        axs[0].plot(p_mle, 1.0, 'o', color=colors[i], markersize=8)
    
    # Customize the plot
    axs[0].set_title('Likelihood Functions for Different Datasets\nwith Same Sufficient Statistic', 
                   fontsize=12)
    axs[0].set_xlabel('p (probability parameter)', fontsize=10)
    axs[0].set_ylabel('Normalized Likelihood', fontsize=10)
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(fontsize=9)
    
    # Add annotation about sufficiency
    axs[0].text(0.05, 0.5, 
               "For Bernoulli(p), the sample sum Σx_i\n" +
               "(or equivalently, the sample proportion)\n" +
               "is a sufficient statistic for p.\n\n" +
               "Different datasets with the same sum\n" +
               "yield identical likelihood functions.",
               transform=axs[0].transAxes, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Plot 2: Fisher-Neyman factorization theorem demonstration
    # Create a small dataset for demonstration
    n_small = 5
    data_small = np.array([1, 0, 1, 1, 0])
    sum_x_small = np.sum(data_small)
    
    # Create grid of parameter values
    p_grid = np.linspace(0.1, 0.9, 9)
    
    # Initialize likelihood values
    full_likelihood = []
    factorized_g = []  # g(T(x), p) part
    factorized_h = 1.0  # h(x) part (constant for Bernoulli)
    
    # Calculate full and factorized likelihood for each p value
    for p in p_grid:
        # Full likelihood
        L_full = p**sum_x_small * (1-p)**(n_small-sum_x_small)
        full_likelihood.append(L_full)
        
        # Factorized likelihood
        g = p**sum_x_small * (1-p)**(n_small-sum_x_small)  # Depends on data only through sum_x
        factorized_g.append(g)
    
    # Create step plots to visualize the equality
    axs[1].step(p_grid, full_likelihood, 'b-', linewidth=2, where='mid', 
              label='Full likelihood L(p; x)')
    axs[1].step(p_grid, factorized_g, 'r--', linewidth=2, where='mid', 
              label='Factorized g(T(x), p)')
    
    # Customize the plot
    axs[1].set_title('Fisher-Neyman Factorization Theorem\nL(p; x) = h(x) · g(T(x), p)', 
                   fontsize=12)
    axs[1].set_xlabel('p (probability parameter)', fontsize=10)
    axs[1].set_ylabel('Likelihood Value', fontsize=10)
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(fontsize=9)
    
    # Add annotation for the factorization theorem
    data_str = ", ".join([str(int(x)) for x in data_small])
    axs[1].text(0.05, 0.5, 
               f"Data: [{data_str}], T(x) = Σx_i = {sum_x_small}\n\n" +
               "Likelihood factorization:\n" +
               "L(p; x) = h(x) · g(T(x), p)\n\n" +
               f"= 1 · p^{sum_x_small} · (1-p)^{n_small-sum_x_small}\n\n" +
               "Since h(x) = 1, the likelihood\ndepends on data only through\nthe sufficient statistic T(x).",
               transform=axs[1].transAxes, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Overall figure title
    plt.suptitle('Sufficient Statistics for Bernoulli Distribution', fontsize=16, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sufficient statistic demonstration plot saved to {save_path}")
    
    plt.close()

def generate_answer_materials():
    """Generate answer materials for the MLE visual question on Bernoulli distribution"""
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    answer_dir = os.path.join(images_dir, "MLE_Visual_Answer")
    os.makedirs(answer_dir, exist_ok=True)
    
    print("Generating MLE Visual Answer Example 4 materials...")
    
    # Load the Bernoulli data used in the question
    data_dict, true_p = load_bernoulli_data()
    
    # Generate the sampling distribution
    sample_sizes = sorted(list(data_dict.keys()))
    sampling_results = generate_sampling_distribution(
        true_p=true_p,
        sample_sizes=sample_sizes,
        n_simulations=10000
    )
    
    # Generate the sampling distribution visualization
    sampling_dist_path = os.path.join(answer_dir, "ex4_sampling_distribution.png")
    plot_mle_sampling_distribution(sampling_results, true_p, save_path=sampling_dist_path)
    
    # Generate the MLE properties visualization
    properties_path = os.path.join(answer_dir, "ex4_mle_properties.png")
    plot_mle_properties(sampling_results, true_p, save_path=properties_path)
    
    # Generate the sufficient statistic demonstration
    sufficiency_path = os.path.join(answer_dir, "ex4_sufficient_statistic.png")
    plot_sufficient_statistic_demonstration(save_path=sufficiency_path)
    
    print("MLE Visual Answer Example 4 materials generated successfully!")
    
    return data_dict, sampling_results, answer_dir

if __name__ == "__main__":
    # Generate all the answer materials
    data_dict, sampling_results, answer_dir = generate_answer_materials() 