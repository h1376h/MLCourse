import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
import os

def linear_model(x, beta0, beta1):
    """Linear regression model: y = beta0 + beta1 * x"""
    return beta0 + beta1 * x

def log_likelihood(params, x, y):
    """Log-likelihood function for linear regression with normal errors"""
    beta0, beta1, sigma = params
    y_pred = linear_model(x, beta0, beta1)
    n = len(y)
    # Log-likelihood for normal distribution
    log_like = -n/2 * np.log(2 * np.pi) - n * np.log(sigma) - np.sum((y - y_pred)**2) / (2 * sigma**2)
    return log_like

def neg_log_likelihood(params, x, y):
    """Negative log-likelihood for minimization"""
    return -log_likelihood(params, x, y)

def plot_data_and_regression_line(x, y, beta0, beta1, save_path=None):
    """Plot the data points and regression line"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data points
    ax.scatter(x, y, color='blue', s=80, alpha=0.7, label='Observed Data')
    
    # Plot regression line
    x_line = np.linspace(min(x) - 0.5, max(x) + 0.5, 100)
    y_line = linear_model(x_line, beta0, beta1)
    ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'Regression Line: y = {beta0:.3f} + {beta1:.3f}x')
    
    # Add equation on the plot
    equation_text = f'ŷ = {beta0:.3f} + {beta1:.3f}x'
    ax.text(0.5, 0.9, equation_text, transform=ax.transAxes, 
            fontsize=14, ha='center', bbox=dict(facecolor='white', alpha=0.7))
    
    ax.set_xlabel('Advertising Expenditure (thousands $)')
    ax.set_ylabel('Sales (thousands units)')
    ax.set_title('Linear Regression: Sales vs. Advertising')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add text with R-squared
    y_mean = np.mean(y)
    ss_total = np.sum((y - y_mean)**2)
    y_pred = linear_model(x, beta0, beta1)
    ss_residual = np.sum((y - y_pred)**2)
    r_squared = 1 - (ss_residual / ss_total)
    
    ax.text(0.05, 0.1, f'R² = {r_squared:.4f}', transform=ax.transAxes, 
            fontsize=12, ha='left', bbox=dict(facecolor='white', alpha=0.7))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_residuals(x, y, beta0, beta1, save_path=None):
    """Plot residuals from the regression model"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate predicted values
    y_pred = linear_model(x, beta0, beta1)
    
    # Calculate residuals
    residuals = y - y_pred
    
    # Plot residuals
    ax.scatter(x, residuals, color='green', s=80, alpha=0.7)
    ax.axhline(y=0, color='r', linestyle='-')
    
    ax.set_xlabel('Advertising Expenditure (thousands $)')
    ax.set_ylabel('Residuals')
    ax.set_title('Residual Plot')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_log_likelihood_surface(x, y, beta0_range, beta1_range, true_beta0, true_beta1, save_path=None):
    """Plot log-likelihood surface for beta0 and beta1 (with fixed sigma)"""
    # Estimate sigma using true betas
    y_pred = linear_model(x, true_beta0, true_beta1)
    sigma_est = np.sqrt(np.sum((y - y_pred)**2) / len(y))
    
    # Create grid
    beta0_vals = np.linspace(beta0_range[0], beta0_range[1], 50)
    beta1_vals = np.linspace(beta1_range[0], beta1_range[1], 50)
    B0, B1 = np.meshgrid(beta0_vals, beta1_vals)
    
    # Calculate log-likelihood at each point
    log_like_vals = np.zeros_like(B0)
    for i in range(len(beta0_vals)):
        for j in range(len(beta1_vals)):
            log_like_vals[j, i] = log_likelihood([beta0_vals[i], beta1_vals[j], sigma_est], x, y)
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Surface plot
    surf = ax1.plot_surface(B0, B1, log_like_vals, cmap='viridis', alpha=0.8, 
                          rstride=1, cstride=1, linewidth=0, antialiased=True)
    
    # Mark the maximum
    max_idx = np.unravel_index(np.argmax(log_like_vals), log_like_vals.shape)
    ax1.scatter([true_beta0], [true_beta1], 
               [log_likelihood([true_beta0, true_beta1, sigma_est], x, y)], 
               color='red', s=100, marker='*', label='MLE')
    
    ax1.set_xlabel('β₀')
    ax1.set_ylabel('β₁')
    ax1.set_zlabel('Log-Likelihood')
    ax1.set_title('Log-Likelihood Surface')
    ax1.legend()
    
    # Add contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(B0, B1, log_like_vals, 20, cmap='viridis')
    ax2.clabel(contour, inline=1, fontsize=10)
    ax2.plot(true_beta0, true_beta1, 'r*', markersize=15, label='MLE')
    
    ax2.set_xlabel('β₀')
    ax2.set_ylabel('β₁')
    ax2.set_title('Log-Likelihood Contour Plot')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def analytical_mle(x, y):
    """Compute MLE estimators for beta0, beta1 and sigma using analytical formulas"""
    n = len(x)
    
    # Calculate the means
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Calculate beta1 (slope)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean)**2)
    beta1 = numerator / denominator
    
    # Calculate beta0 (intercept)
    beta0 = y_mean - beta1 * x_mean
    
    # Calculate sigma
    y_pred = beta0 + beta1 * x
    sigma = np.sqrt(np.sum((y - y_pred)**2) / n)
    
    return beta0, beta1, sigma

def plot_normal_distribution(x, y, beta0, beta1, sigma, save_path=None):
    """Plot the normal distribution of the error terms"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate residuals
    y_pred = linear_model(x, beta0, beta1)
    residuals = y - y_pred
    
    # Plot histogram of residuals
    ax.hist(residuals, bins=10, density=True, alpha=0.6, color='green', 
            label='Residual Distribution')
    
    # Plot fitted normal distribution
    x_range = np.linspace(min(residuals) - 0.5, max(residuals) + 0.5, 1000)
    y_range = norm.pdf(x_range, 0, sigma)
    ax.plot(x_range, y_range, 'r-', linewidth=2, 
            label=f'Normal Distribution (σ = {sigma:.3f})')
    
    ax.set_xlabel('Residual Value')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Residuals')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def numerical_mle(x, y, initial_params=None):
    """Compute MLE estimators for beta0, beta1 and sigma using numerical optimization"""
    if initial_params is None:
        initial_params = [0.0, 1.0, 1.0]  # Initial guess for beta0, beta1, sigma
    
    # Bounds for parameters (sigma must be positive)
    bounds = [(None, None), (None, None), (1e-10, None)]
    
    # Minimize negative log-likelihood
    result = minimize(neg_log_likelihood, initial_params, args=(x, y), bounds=bounds)
    
    # Extract results
    beta0_mle, beta1_mle, sigma_mle = result.x
    
    return beta0_mle, beta1_mle, sigma_mle, result.fun

def main():
    """Generate all visualizations for Question 11 of the L2.4 quiz"""
    # Data from the question
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Advertising expenditure
    y = np.array([2.1, 3.5, 4.5, 5.8, 6.3])  # Sales
    
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_4_Quiz_11")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 11 of the L2.4 MLE quiz...")
    
    # 1. Compute analytical MLE estimators
    beta0_mle, beta1_mle, sigma_mle = analytical_mle(x, y)
    print(f"Analytical MLE estimates:")
    print(f"β₀ = {beta0_mle:.4f}")
    print(f"β₁ = {beta1_mle:.4f}")
    print(f"σ² = {sigma_mle**2:.4f}")
    
    # 2. Plot data and regression line
    plot_data_and_regression_line(x, y, beta0_mle, beta1_mle, 
                                 save_path=os.path.join(save_dir, "regression_line.png"))
    print("1. Regression line visualization created")
    
    # 3. Plot residuals
    plot_residuals(x, y, beta0_mle, beta1_mle, 
                  save_path=os.path.join(save_dir, "residuals.png"))
    print("2. Residuals visualization created")
    
    # 4. Plot log-likelihood surface
    plot_log_likelihood_surface(x, y, (0.5, 1.5), (1.0, 1.5), beta0_mle, beta1_mle, 
                               save_path=os.path.join(save_dir, "log_likelihood_surface.png"))
    print("3. Log-likelihood surface visualization created")
    
    # 5. Plot normal distribution of residuals
    plot_normal_distribution(x, y, beta0_mle, beta1_mle, sigma_mle, 
                            save_path=os.path.join(save_dir, "residual_distribution.png"))
    print("4. Residual distribution visualization created")
    
    # 6. Verify with numerical optimization
    num_beta0, num_beta1, num_sigma, neg_ll = numerical_mle(x, y)
    print(f"\nNumerical MLE estimates (verification):")
    print(f"β₀ = {num_beta0:.4f}")
    print(f"β₁ = {num_beta1:.4f}")
    print(f"σ² = {num_sigma**2:.4f}")
    
    # Calculate sum of squared residuals
    y_pred = linear_model(x, beta0_mle, beta1_mle)
    ssr = np.sum((y - y_pred)**2)
    
    # Print detailed results for Question 11
    print("\nQuestion 11 Results:")
    print("Linear Regression: Sales vs. Advertising")
    print(f"Data points: {len(x)}")
    print(f"Regression equation: ŷ = {beta0_mle:.4f} + {beta1_mle:.4f}x")
    print(f"Sum of Squared Residuals: {ssr:.4f}")
    print(f"MLE for σ²: {sigma_mle**2:.4f}")
    
    # Calculate predictions for each data point
    print("\nFitted values:")
    for i in range(len(x)):
        pred = linear_model(x[i], beta0_mle, beta1_mle)
        residual = y[i] - pred
        print(f"x = {x[i]:.1f}, y = {y[i]:.1f}, ŷ = {pred:.4f}, residual = {residual:.4f}")
    
    print(f"\nAll visualizations have been saved to: {save_dir}")

if __name__ == "__main__":
    main() 