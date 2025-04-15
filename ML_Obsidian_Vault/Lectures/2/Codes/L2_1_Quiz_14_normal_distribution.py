import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

def calculate_normal_probability(mean, std_dev, lower, upper):
    """
    Calculate the probability that a normal random variable falls between two values
    
    Args:
        mean: Mean of the normal distribution
        std_dev: Standard deviation of the normal distribution
        lower: Lower bound
        upper: Upper bound
    
    Returns:
        Probability that the random variable is between lower and upper
    """
    # Standardize the lower and upper bounds
    z_lower = (lower - mean) / std_dev
    z_upper = (upper - mean) / std_dev
    
    # Calculate the probability using the CDF of the standard normal distribution
    prob = stats.norm.cdf(z_upper) - stats.norm.cdf(z_lower)
    
    return prob

def calculate_symmetric_interval(mean, std_dev, coverage_probability):
    """
    Calculate a symmetric interval around the mean that contains a specified probability mass
    
    Args:
        mean: Mean of the normal distribution
        std_dev: Standard deviation of the normal distribution
        coverage_probability: Desired probability mass contained in the interval
    
    Returns:
        Tuple containing the lower and upper bounds of the interval
    """
    # Calculate the z-score corresponding to the desired coverage probability
    # For a symmetric interval, we want half of (1-p) in each tail
    alpha = 1 - coverage_probability
    z_score = stats.norm.ppf(1 - alpha/2)
    
    # Calculate the lower and upper bounds
    lower = mean - z_score * std_dev
    upper = mean + z_score * std_dev
    
    return lower, upper

def calculate_tall_threshold(mean, std_dev, tall_percentile=0.95):
    """
    Calculate the threshold for being considered tall
    
    Args:
        mean: Mean of the normal distribution
        std_dev: Standard deviation of the normal distribution
        tall_percentile: Percentile above which someone is considered tall
    
    Returns:
        Height threshold
    """
    # Calculate the z-score corresponding to the desired percentile
    z_score = stats.norm.ppf(tall_percentile)
    
    # Calculate the height threshold
    threshold = mean + z_score * std_dev
    
    return threshold

def calculate_sample_mean_probability(pop_mean, pop_std_dev, sample_size, threshold):
    """
    Calculate the probability that a sample mean exceeds a threshold
    
    Args:
        pop_mean: Population mean
        pop_std_dev: Population standard deviation
        sample_size: Sample size
        threshold: Threshold value
    
    Returns:
        Probability that the sample mean exceeds the threshold
    """
    # Calculate the standard error of the sample mean
    standard_error = pop_std_dev / np.sqrt(sample_size)
    
    # Calculate the z-score
    z_score = (threshold - pop_mean) / standard_error
    
    # Calculate the probability
    probability = 1 - stats.norm.cdf(z_score)
    
    return probability

def visualize_normal(mean, std_dev, vis_type='basic', **kwargs):
    """
    Create a simplified visualization of the normal distribution
    
    Args:
        mean: Mean of the normal distribution
        std_dev: Standard deviation of the normal distribution
        vis_type: Type of visualization ('basic', 'interval', 'threshold', 'std_devs', 'sample_mean', or 'compare')
        **kwargs: Additional parameters based on visualization type
            - For 'interval': lower, upper bounds
            - For 'threshold': threshold value
            - For 'std_devs': number of std_devs to show (default 3)
            - For 'sample_mean': sample_size, threshold
            - For 'compare': sample_sizes as list, common threshold
        save_path: Path to save the visualization (optional)
    """
    plt.figure(figsize=(8, 5))
    
    # Create x values for the normal distribution
    if vis_type == 'sample_mean':
        standard_error = std_dev / np.sqrt(kwargs.get('sample_size', 100))
        x = np.linspace(mean - 4*standard_error, mean + 4*standard_error, 1000)
    elif vis_type == 'compare':
        # Use the most extreme values for x-range
        sample_sizes = kwargs.get('sample_sizes', [25, 100, 400])
        min_se = std_dev / np.sqrt(max(sample_sizes))
        max_se = std_dev / np.sqrt(min(sample_sizes))
        x = np.linspace(mean - 4*max_se, mean + 4*max_se, 1000)
    else:
        x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)
    
    # Calculate the PDF values
    if vis_type == 'sample_mean':
        standard_error = std_dev / np.sqrt(kwargs.get('sample_size', 100))
        pdf = stats.norm.pdf(x, mean, standard_error)
    elif vis_type == 'compare':
        # Don't plot PDF yet, will do it in the loop
        pass
    else:
        pdf = stats.norm.pdf(x, mean, std_dev)
    
    # Plot the PDF with a clean design
    if vis_type != 'compare':
        plt.plot(x, pdf, color='#3498db', lw=2)
    
    # Add vertical line for the mean
    if vis_type != 'compare':
        plt.axvline(mean, color='#e74c3c', linestyle='--', lw=1.5, label=f'Mean = {mean}')
    
    # Handle specific visualization types
    title = 'Normal Distribution'
    
    if vis_type == 'interval':
        lower, upper = kwargs.get('lower', mean-std_dev), kwargs.get('upper', mean+std_dev)
        mask = (x >= lower) & (x <= upper)
        plt.fill_between(x, pdf, where=mask, alpha=0.4, color='#2ecc71')
        plt.axvline(lower, color='#27ae60', linestyle='-', lw=1.5)
        plt.axvline(upper, color='#27ae60', linestyle='-', lw=1.5, 
                   label=f'Bounds: [{lower:.1f}, {upper:.1f}]')
        
        # Add z-scores
        z_lower = (lower - mean) / std_dev
        z_upper = (upper - mean) / std_dev
        
        title = 'Height Range for Normal Distribution'
        
    elif vis_type == 'threshold':
        threshold = kwargs.get('threshold', mean+std_dev)
        mask = x >= threshold
        plt.fill_between(x, pdf, where=mask, alpha=0.4, color='#9b59b6')
        plt.axvline(threshold, color='#8e44ad', linestyle='-', lw=1.5, 
                   label=f'Threshold = {threshold:.1f}')
        
        title = 'Threshold for Normal Distribution'
        
    elif vis_type == 'sample_mean':
        threshold = kwargs.get('threshold', mean)
        sample_size = kwargs.get('sample_size', 100)
        standard_error = std_dev / np.sqrt(sample_size)
        
        mask = x >= threshold
        plt.fill_between(x, pdf, where=mask, alpha=0.4, color='#f39c12')
        plt.axvline(threshold, color='#d35400', linestyle='-', lw=1.5, 
                   label=f'Threshold = {threshold:.1f}')
        
        title = f'Sample Mean Distribution (n={sample_size})'
        
    elif vis_type == 'std_devs':
        num_std = kwargs.get('num_std', 3)
        # Further improved color scheme - more professional, accessible colors
        colors = ['#4e79a7', '#59a14f', '#f28e2b', '#76b7b2', '#edc948', '#af7aa1']
        
        # Plot standard deviation regions
        for i in range(1, num_std + 1):
            lower = mean - i * std_dev
            upper = mean + i * std_dev
            mask = (x >= lower) & (x <= upper)
            
            # Use different alpha for each region
            alpha = 0.6 - (i-1) * 0.15
            if alpha < 0.2:
                alpha = 0.2
                
            color_idx = (i-1) % len(colors)
            plt.fill_between(x, pdf, where=mask, alpha=alpha, color=colors[color_idx])
            
            # Add vertical lines
            plt.axvline(lower, color=colors[color_idx], linestyle='--', lw=1.0)
            plt.axvline(upper, color=colors[color_idx], linestyle='--', lw=1.0)
            
            # Add text for probability
            prob = calculate_normal_probability(mean, std_dev, lower, upper)
            # Print the probability information instead of annotating on the plot
            print(f"±{i}σ: {prob*100:.1f}% of data falls within {i} standard deviation(s)")
        
        title = 'Standard Deviation Regions'
    
    elif vis_type == 'compare':
        sample_sizes = kwargs.get('sample_sizes', [25, 100, 400])
        threshold = kwargs.get('threshold', mean)
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        
        # Add vertical line for the mean
        plt.axvline(mean, color='#8e44ad', linestyle='--', lw=1.5, label=f'Mean = {mean}')
        
        # Add vertical line for threshold
        plt.axvline(threshold, color='black', linestyle='-', lw=1.5, 
                   label=f'Threshold = {threshold:.1f}')
        
        # Plot distribution for each sample size
        for i, n in enumerate(sample_sizes):
            standard_error = std_dev / np.sqrt(n)
            pdf_n = stats.norm.pdf(x, mean, standard_error)
            
            plt.plot(x, pdf_n, color=colors[i % len(colors)], lw=2, 
                    label=f'n = {n}, SE = {standard_error:.2f}')
            
            # Calculate and show probability
            prob = calculate_sample_mean_probability(mean, std_dev, n, threshold)
            
            # Print the probability information instead of annotating on the plot
            print(f"P(X̄ > {threshold} | n={n}) = {prob:.3f}")
            
        title = 'Effect of Sample Size on Sampling Distribution'
    
    # Add labels and legend
    plt.xlabel('Value', fontsize=11)
    plt.ylabel('Probability Density', fontsize=11)
    plt.title(title, fontsize=12)
    plt.grid(True, alpha=0.2)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save if a path is provided
    save_path = kwargs.get('save_path', None)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.close()

def export_probabilities_to_markdown(mean, std_dev, sample_sizes, threshold, output_file):
    """
    Export probability data for sample means exceeding a threshold to a markdown file
    
    Args:
        mean: Population mean
        std_dev: Population standard deviation
        sample_sizes: List of sample sizes
        threshold: Threshold value
        output_file: Path to the output markdown file
    """
    with open(output_file, 'w') as f:
        f.write("# Sampling Distribution Probabilities\n\n")
        f.write(f"Population mean: {mean}\n")
        f.write(f"Population standard deviation: {std_dev}\n")
        f.write(f"Threshold: {threshold}\n\n")
        
        f.write("| Sample Size (n) | Standard Error | P(X̄ > {}) |\n".format(threshold))
        f.write("|----------------|----------------|------------|\n")
        
        for n in sample_sizes:
            standard_error = std_dev / np.sqrt(n)
            prob = calculate_sample_mean_probability(mean, std_dev, n, threshold)
            f.write(f"| {n} | {standard_error:.2f} | {prob:.3f} |\n")
        
        print(f"Probability data exported to {output_file}")

def export_std_dev_regions_to_markdown(mean, std_dev, num_std, output_file):
    """
    Export standard deviation region probabilities to a markdown file
    
    Args:
        mean: Population mean
        std_dev: Population standard deviation
        num_std: Number of standard deviations to include
        output_file: Path to the output markdown file
    """
    with open(output_file, 'w') as f:
        f.write("# Standard Deviation Region Probabilities\n\n")
        f.write(f"Population mean: {mean}\n")
        f.write(f"Population standard deviation: {std_dev}\n\n")
        
        f.write("| Region | Range | Probability |\n")
        f.write("|--------|-------|------------|\n")
        
        for i in range(1, num_std + 1):
            lower = mean - i * std_dev
            upper = mean + i * std_dev
            prob = calculate_normal_probability(mean, std_dev, lower, upper)
            f.write(f"| ±{i}σ | {lower:.2f} to {upper:.2f} | {prob*100:.1f}% |\n")
        
        print(f"Standard deviation region data exported to {output_file}")

def main():
    """Generate all visualizations for Question 14 of the L2.1 quiz"""
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_1_Quiz_14")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 14 of the L2.1 Probability quiz: Normal Distribution Application...")
    
    # Problem parameters
    mean = 165  # Mean height in cm
    std_dev = 6  # Standard deviation in cm
    
    # Task 1: Percentage of women with heights between 160cm and 170cm
    print("\nTask 1: Percentage of women with heights between 160cm and 170cm")
    lower_bound = 160
    upper_bound = 170
    probability = calculate_normal_probability(mean, std_dev, lower_bound, upper_bound)
    print(f"Percentage: {probability*100:.2f}%")
    
    # Task 2: Design dress for heights that fits at least 90% of women
    print("\nTask 2: Design dress for heights that fits at least 90% of women")
    coverage_probability = 0.90
    x_min, x_max = calculate_symmetric_interval(mean, std_dev, coverage_probability)
    print(f"The dress should be designed for heights from {x_min:.2f}cm to {x_max:.2f}cm")
    
    # Task 3: Probability that average height of 100 women exceeds 166cm
    print("\nTask 3: Probability that average height of 100 women exceeds 166cm")
    sample_size = 100
    threshold = 166
    prob_exceed = calculate_sample_mean_probability(mean, std_dev, sample_size, threshold)
    print(f"Probability: {prob_exceed:.4f}")
    
    # Task 4: Minimum height to be considered "tall" (top 5%)
    print("\nTask 4: Minimum height to be considered 'tall' (top 5%)")
    tall_percentile = 0.95
    tall_threshold = calculate_tall_threshold(mean, std_dev, tall_percentile)
    print(f"Minimum height to be considered 'tall': {tall_threshold:.2f}cm")
    
    # Generate visualizations with the simplified function
    print("\nGenerating visualizations...")
    
    # Keep the most important visualizations
    
    # 1. Basic normal distribution - essential foundation
    visualize_normal(mean, std_dev, vis_type='basic',
                    save_path=os.path.join(save_dir, "basic_distribution.png"))
    
    # 2. Standard deviation regions - important for understanding the empirical rule
    print("\nStandard Deviation Regions:")
    num_std = 3
    visualize_normal(mean, std_dev, vis_type='std_devs', num_std=num_std,
                    save_path=os.path.join(save_dir, "standard_deviation_regions.png"))
    
    # Export standard deviation region data to markdown
    export_std_dev_regions_to_markdown(mean, std_dev, num_std, 
                                    os.path.join(save_dir, "standard_deviation_regions.md"))
    
    # 3. Height range from task 1 - directly relevant to a quiz question
    visualize_normal(mean, std_dev, vis_type='interval', 
                    lower=lower_bound, upper=upper_bound,
                    save_path=os.path.join(save_dir, "height_distribution.png"))
    
    # 4. Dress design range from task 2 - directly relevant to a quiz question
    visualize_normal(mean, std_dev, vis_type='interval', 
                    lower=x_min, upper=x_max,
                    save_path=os.path.join(save_dir, "dress_design_range.png"))
    
    # 5. Sample mean distribution from task 3 - directly relevant to a quiz question
    visualize_normal(mean, std_dev, vis_type='sample_mean', 
                    sample_size=sample_size, threshold=threshold,
                    save_path=os.path.join(save_dir, "sample_mean_distribution.png"))
    
    # 6. Sample size comparison - important for understanding the CLT effect
    print("\nProbabilities for different sample sizes exceeding threshold:")
    sample_sizes = [25, 100, 400]
    visualize_normal(mean, std_dev, vis_type='compare', 
                   sample_sizes=sample_sizes, threshold=166,
                   save_path=os.path.join(save_dir, "sample_size_comparison.png"))
    
    # Export probability data to markdown
    export_probabilities_to_markdown(mean, std_dev, sample_sizes, 166, 
                                    os.path.join(save_dir, "sampling_distribution_probabilities.md"))
    
    # 7. Tall threshold from task 4 - directly relevant to a quiz question
    visualize_normal(mean, std_dev, vis_type='threshold', 
                    threshold=tall_threshold,
                    save_path=os.path.join(save_dir, "tall_threshold.png"))
    
    print(f"\nAll visualizations have been saved to: {save_dir}")

if __name__ == "__main__":
    main() 