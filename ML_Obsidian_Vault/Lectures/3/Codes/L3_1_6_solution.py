import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
import scipy.stats as stats

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_1_Quiz_6")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Define the given variance of the error term
sigma_squared = 25

# Step 1: Calculate the standard error of the error term
def calculate_standard_error(sigma_squared):
    """Calculate the standard error of the error term."""
    sigma = np.sqrt(sigma_squared)
    
    print("Step 1: Calculate the standard error of the error term")
    print(f"The standard error (σ) is the square root of the variance (σ²):")
    print(f"σ = √σ² = √{sigma_squared} = {sigma}")
    print()
    
    return sigma

sigma = calculate_standard_error(sigma_squared)

# Step 2: Demonstrate how variance affects coefficient estimators
def demonstrate_variance_effect():
    """Demonstrate how the variance of error terms affects coefficient estimator variance."""
    print("Step 2: Demonstrate how error variance affects coefficient estimator variance")
    print(f"In linear regression, the variance of coefficient estimators is directly proportional to σ².")
    print(f"For the slope coefficient (β₁), the formula is:")
    print(f"Var(β̂₁) = σ²/Σ(xᵢ - x̄)²")
    print(f"For the intercept coefficient (β₀), the formula is:")
    print(f"Var(β̂₀) = σ²[1/n + x̄²/Σ(xᵢ - x̄)²]")
    print()
    print(f"Let's demonstrate this with a simulated example:")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate some example data
    n = 50  # sample size
    x = np.linspace(0, 10, n)
    
    # Define true parameters
    beta0_true = 3
    beta1_true = 2
    
    # Variances to compare
    variance_levels = [5, 25, 100]
    results = []
    
    print(f"Sample size: {n}")
    print(f"True β₀: {beta0_true}")
    print(f"True β₁: {beta1_true}")
    print()
    
    for var in variance_levels:
        sigma = np.sqrt(var)
        betas0 = []
        betas1 = []
        
        # Run multiple simulations to observe coefficient variance
        n_simulations = 1000
        
        for _ in range(n_simulations):
            # Generate errors with the specified variance
            epsilon = np.random.normal(0, sigma, n)
            
            # Generate y values
            y = beta0_true + beta1_true * x + epsilon
            
            # Calculate regression coefficients
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            
            # Calculate the slope (β₁)
            numerator = np.sum((x - x_mean) * (y - y_mean))
            denominator = np.sum((x - x_mean) ** 2)
            beta1 = numerator / denominator
            
            # Calculate the intercept (β₀)
            beta0 = y_mean - beta1 * x_mean
            
            betas0.append(beta0)
            betas1.append(beta1)
        
        # Calculate empirical variance of estimators
        var_beta0 = np.var(betas0)
        var_beta1 = np.var(betas1)
        
        # Calculate theoretical variance
        denom = np.sum((x - x_mean) ** 2)
        theo_var_beta1 = var / denom
        theo_var_beta0 = var * (1/n + x_mean**2/denom)
        
        results.append({
            'error_var': var,
            'beta0_var': var_beta0,
            'beta1_var': var_beta1,
            'theo_beta0_var': theo_var_beta0,
            'theo_beta1_var': theo_var_beta1,
            'beta0_samples': betas0,
            'beta1_samples': betas1
        })
        
        print(f"For error variance σ² = {var}:")
        print(f"  Empirical Var(β̂₀): {var_beta0:.6f}")
        print(f"  Theoretical Var(β̂₀): {theo_var_beta0:.6f}")
        print(f"  Empirical Var(β̂₁): {var_beta1:.6f}")
        print(f"  Theoretical Var(β̂₁): {theo_var_beta1:.6f}")
        print()
    
    return x, results

x, variance_results = demonstrate_variance_effect()

# Step 3: Explain why we want estimators with minimum variance
def explain_minimum_variance():
    """Explain the importance of estimators with minimum variance."""
    print("Step 3: Why we want estimators with minimum variance")
    print("We want estimators with minimum variance for several important reasons:")
    print("1. Precision: Lower variance means more precise estimates that are closer to the true parameter values.")
    print("2. Reliability: Less variability across samples means more consistent results.")
    print("3. Narrower confidence intervals: Lower variance leads to narrower confidence intervals.")
    print("4. Statistical efficiency: Among unbiased estimators, those with minimum variance make optimal use of data.")
    print("5. Better predictions: More precise coefficient estimates lead to more accurate predictions.")
    print()
    
    # Demonstrating the effect of sample size on variance of estimators
    np.random.seed(42)
    
    # True parameters
    beta0_true = 3
    beta1_true = 2
    error_var = 25  # Fixed variance
    
    # Different sample sizes
    sample_sizes = [10, 30, 100, 300]
    
    # Print header
    print("Effect of Sample Size on Estimator Variance:")
    print("-------------------------------------------")
    print("Sample Size | Var(β̂₀) | Var(β̂₁)")
    print("-------------------------------------------")
    
    for n in sample_sizes:
        # Generate predictor variable
        x = np.linspace(0, 10, n)
        x_mean = np.mean(x)
        
        # Calculate theoretical variances based on formula
        denom = np.sum((x - x_mean) ** 2)
        theo_var_beta1 = error_var / denom
        theo_var_beta0 = error_var * (1/n + x_mean**2/denom)
        
        print(f"{n:^11} | {theo_var_beta0:^7.4f} | {theo_var_beta1:^7.4f}")
    
    print()
    print("As sample size increases, the variance of estimators decreases, leading to more precise estimates.")
    print("This demonstrates that efficient estimators allow us to achieve the same precision with fewer data points.")
    
    return sample_sizes

sample_sizes = explain_minimum_variance()

# Create visualizations
def create_visualizations(sigma_squared, sigma, x, variance_results, sample_sizes, save_dir=None):
    """Create visualizations to help understand the statistical properties of estimators."""
    saved_files = []
    
    # Plot 1: Standard Error Visualization
    plt.figure(figsize=(10, 6))
    
    # Create a normal distribution with variance sigma_squared
    x_range = np.linspace(-15, 15, 1000)
    y_normal = stats.norm.pdf(x_range, 0, sigma)
    
    plt.plot(x_range, y_normal, 'b-', linewidth=2, label='Error Distribution')
    
    # Fill the standard error range
    plt.fill_between(x_range, 0, y_normal, 
                    where=(x_range >= -sigma) & (x_range <= sigma),
                    color='blue', alpha=0.3, 
                    label=f'Standard Error (σ = {sigma})')
    
    # Add vertical lines for standard error
    plt.axvline(x=-sigma, color='red', linestyle='--')
    plt.axvline(x=sigma, color='red', linestyle='--')
    
    # Add text annotations
    plt.annotate(f'σ = {sigma}', xy=(sigma, 0.01), xytext=(sigma+1, 0.01),
                arrowprops=dict(arrowstyle='->'), fontsize=12)
    plt.annotate(f'-σ = -{sigma}', xy=(-sigma, 0.01), xytext=(-sigma-1, 0.01),
                arrowprops=dict(arrowstyle='->'), fontsize=12)
    
    plt.title('Standard Error of the Error Term', fontsize=14)
    plt.xlabel('Error Value', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot1_standard_error.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 2: Coefficient Distributions with Different Error Variances
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = ['green', 'blue', 'red']
    
    for i, result in enumerate(variance_results):
        var = result['error_var']
        color = colors[i]
        
        # Plot β₀ distribution
        sns_beta0 = axes[0].hist(result['beta0_samples'], bins=30, alpha=0.3, 
                              color=color, density=True, 
                              label=f'σ² = {var}')
        
        # Plot β₁ distribution
        sns_beta1 = axes[1].hist(result['beta1_samples'], bins=30, alpha=0.3, 
                              color=color, density=True,
                              label=f'σ² = {var}')
        
        # Add vertical lines for true values
        axes[0].axvline(x=3, color='black', linestyle='--', alpha=0.7, label='True β₀' if i == 0 else "")
        axes[1].axvline(x=2, color='black', linestyle='--', alpha=0.7, label='True β₁' if i == 0 else "")
    
    axes[0].set_title('Distribution of Intercept Estimator (β̂₀)', fontsize=12)
    axes[0].set_xlabel('β̂₀ Value', fontsize=10)
    axes[0].set_ylabel('Density', fontsize=10)
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].set_title('Distribution of Slope Estimator (β̂₁)', fontsize=12)
    axes[1].set_xlabel('β̂₁ Value', fontsize=10)
    axes[1].set_ylabel('Density', fontsize=10)
    axes[1].legend()
    axes[1].grid(True)
    
    plt.suptitle('Effect of Error Variance on Coefficient Estimator Distributions', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot2_coefficient_distributions.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 3: Relationship between Error Variance and Coefficient Variance
    plt.figure(figsize=(10, 6))
    
    error_vars = [r['error_var'] for r in variance_results]
    beta0_vars = [r['beta0_var'] for r in variance_results]
    beta1_vars = [r['beta1_var'] for r in variance_results]
    
    theo_beta0_vars = [r['theo_beta0_var'] for r in variance_results]
    theo_beta1_vars = [r['theo_beta1_var'] for r in variance_results]
    
    plt.plot(error_vars, beta0_vars, 'bo-', linewidth=2, markersize=8, label='Empirical Var(β̂₀)')
    plt.plot(error_vars, theo_beta0_vars, 'b--', linewidth=1, label='Theoretical Var(β̂₀)')
    
    plt.plot(error_vars, beta1_vars, 'ro-', linewidth=2, markersize=8, label='Empirical Var(β̂₁)')
    plt.plot(error_vars, theo_beta1_vars, 'r--', linewidth=1, label='Theoretical Var(β̂₁)')
    
    plt.title('Relationship Between Error Variance and Coefficient Variance', fontsize=14)
    plt.xlabel('Error Variance (σ²)', fontsize=12)
    plt.ylabel('Coefficient Variance', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=10)
    
    # Add text to highlight the direct proportionality
    plt.text(0.5, 0.9, 'Var(β̂) ∝ σ²\nCoefficient variance is directly\nproportional to error variance', 
            transform=plt.gca().transAxes, fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot3_variance_relationship.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 4: Effect of Sample Size on Estimator Precision
    plt.figure(figsize=(10, 6))
    
    # Generate data for different sample sizes
    np.random.seed(42)
    beta0_true = 3
    beta1_true = 2
    error_std = np.sqrt(sigma_squared)
    
    n_simulations = 100
    sample_sizes_extended = np.logspace(1, 3, 20).astype(int)  # from 10 to 1000
    
    beta0_vars = []
    beta1_vars = []
    theoretical_beta0_vars = []
    theoretical_beta1_vars = []
    
    for n in sample_sizes_extended:
        beta0_samples = []
        beta1_samples = []
        
        # Theoretical variance calculation
        x = np.linspace(0, 10, n)
        x_mean = np.mean(x)
        denom = np.sum((x - x_mean) ** 2)
        
        theo_var_beta1 = sigma_squared / denom
        theo_var_beta0 = sigma_squared * (1/n + x_mean**2/denom)
        
        theoretical_beta0_vars.append(theo_var_beta0)
        theoretical_beta1_vars.append(theo_var_beta1)
        
        # Empirical variance calculation
        for _ in range(n_simulations):
            # Generate data
            epsilon = np.random.normal(0, error_std, n)
            y = beta0_true + beta1_true * x + epsilon
            
            # Calculate regression coefficients
            numerator = np.sum((x - x_mean) * (y - np.mean(y)))
            beta1 = numerator / denom
            beta0 = np.mean(y) - beta1 * x_mean
            
            beta0_samples.append(beta0)
            beta1_samples.append(beta1)
        
        beta0_vars.append(np.var(beta0_samples))
        beta1_vars.append(np.var(beta1_samples))
    
    plt.loglog(sample_sizes_extended, beta0_vars, 'bo-', linewidth=2, markersize=6, label='Empirical Var(β̂₀)')
    plt.loglog(sample_sizes_extended, theoretical_beta0_vars, 'b--', linewidth=1, label='Theoretical Var(β̂₀)')
    
    plt.loglog(sample_sizes_extended, beta1_vars, 'ro-', linewidth=2, markersize=6, label='Empirical Var(β̂₁)')
    plt.loglog(sample_sizes_extended, theoretical_beta1_vars, 'r--', linewidth=1, label='Theoretical Var(β̂₁)')
    
    plt.title('Effect of Sample Size on Coefficient Estimator Variance', fontsize=14)
    plt.xlabel('Sample Size (log scale)', fontsize=12)
    plt.ylabel('Coefficient Variance (log scale)', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=10)
    
    # Add text to highlight the relationship
    plt.text(0.05, 0.2, 'Var(β̂) ∝ 1/n\nCoefficient variance is inversely\nproportional to sample size', 
            transform=plt.gca().transAxes, fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot4_sample_size_effect.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 5: Confidence interval visualization
    plt.figure(figsize=(10, 6))
    
    # Generate a sample dataset
    np.random.seed(456)
    n = 50
    x = np.linspace(0, 10, n)
    
    variances = [5, 25, 100]
    colors = ['green', 'blue', 'red']
    
    for i, variance in enumerate(variances):
        error_std = np.sqrt(variance)
        epsilon = np.random.normal(0, error_std, n)
        y = 3 + 2 * x + epsilon
        
        # Fit regression model
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        # Calculate the slope (β₁)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        beta1 = numerator / denominator
        
        # Calculate the intercept (β₀)
        beta0 = y_mean - beta1 * x_mean
        
        # Calculate standard errors
        y_pred = beta0 + beta1 * x
        residuals = y - y_pred
        residual_std = np.sqrt(np.sum(residuals**2) / (n - 2))
        
        se_beta1 = residual_std / np.sqrt(denominator)
        se_beta0 = residual_std * np.sqrt(1/n + x_mean**2/denominator)
        
        # Calculate confidence intervals (95%)
        t_critical = stats.t.ppf(0.975, n - 2)
        
        beta1_lower = beta1 - t_critical * se_beta1
        beta1_upper = beta1 + t_critical * se_beta1
        
        # Generate confidence bands
        x_range = np.linspace(0, 10, 100)
        y_pred_line = beta0 + beta1 * x_range
        
        # For each point in x_range, calculate prediction standard error
        y_pred_se = np.zeros_like(x_range)
        for j, x_j in enumerate(x_range):
            y_pred_se[j] = residual_std * np.sqrt(1/n + (x_j - x_mean)**2/denominator)
        
        # Calculate confidence bands
        lower_band = y_pred_line - t_critical * y_pred_se
        upper_band = y_pred_line + t_critical * y_pred_se
        
        # Plot regression line and confidence bands
        plt.plot(x_range, y_pred_line, color=colors[i], linewidth=2, 
                label=f'σ² = {variance} (β̂₁ = {beta1:.2f})')
        
        plt.fill_between(x_range, lower_band, upper_band, color=colors[i], alpha=0.15)
        
        # Add annotation for the confidence interval of β₁
        text_y_pos = 5 + i * 2
        plt.text(6, text_y_pos, f'95% CI for β₁: [{beta1_lower:.2f}, {beta1_upper:.2f}]',
                fontsize=10, color=colors[i],
                bbox=dict(facecolor='white', alpha=0.7))
    
    plt.scatter(x, y, color='black', s=30, alpha=0.5, label='Data points')
    
    plt.title('Regression Lines with Confidence Bands for Different Error Variances', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot5_confidence_intervals.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    return saved_files

# Fix missing import
try:
    import seaborn as sns
    plt.style.use('seaborn-v0_8-whitegrid')
except ImportError:
    print("Seaborn not available, using default style")

# Create visualizations
saved_files = create_visualizations(sigma_squared, sigma, x, variance_results, sample_sizes, save_dir)

print(f"Visualizations saved to: {', '.join(saved_files)}")
print("\nQuestion 6 Solution Summary:")
print(f"1. Standard error of the error term: σ = {sigma}")
print(f"2. The variance of coefficient estimators is directly proportional to the error variance σ².")
print(f"3. We want estimators with minimum variance for greater precision, reliability, efficiency, and better predictions.") 