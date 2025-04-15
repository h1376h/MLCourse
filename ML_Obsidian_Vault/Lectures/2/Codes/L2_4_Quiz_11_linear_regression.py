import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

def plot_data_and_regression(x, y, beta0, beta1, save_path=None):
    """Plot the data points and fitted regression line"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data points
    ax.scatter(x, y, color='blue', marker='o', s=60, label='Observed Data')
    
    # Create x values for plotting the regression line
    x_line = np.linspace(min(x)-0.5, max(x)+0.5, 100)
    y_line = beta0 + beta1 * x_line
    
    # Plot regression line
    ax.plot(x_line, y_line, 'r-', linewidth=2, 
            label=f'MLE Fit: y = {beta0:.4f} + {beta1:.4f}x')
    
    # Add residual lines
    for i in range(len(x)):
        y_pred = beta0 + beta1 * x[i]
        ax.plot([x[i], x[i]], [y[i], y_pred], 'g--', alpha=0.5)
    
    ax.set_xlabel('Advertising Expenditure (thousands $)')
    ax.set_ylabel('Sales (thousands units)')
    ax.set_title('MLE Regression Line and Observed Data')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_log_likelihood_surface(x, y, sigma_squared, save_path=None):
    """Plot the log-likelihood surface for beta0 and beta1"""
    # Calculate MLEs for beta0 and beta1
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    beta1_mle = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
    beta0_mle = y_mean - beta1_mle * x_mean
    
    # Create grid of beta0 and beta1 values
    beta0_range = np.linspace(beta0_mle - 1, beta0_mle + 1, 100)
    beta1_range = np.linspace(beta1_mle - 0.5, beta1_mle + 0.5, 100)
    beta0_grid, beta1_grid = np.meshgrid(beta0_range, beta1_range)
    
    # Calculate log-likelihood for each combination
    log_likelihood = np.zeros_like(beta0_grid)
    for i in range(len(beta0_range)):
        for j in range(len(beta1_range)):
            b0 = beta0_range[i]
            b1 = beta1_range[j]
            residuals = y - (b0 + b1 * x)
            log_likelihood[j, i] = -n/2 * np.log(2 * np.pi * sigma_squared) - np.sum(residuals**2) / (2 * sigma_squared)
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # 3D surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(beta0_grid, beta1_grid, log_likelihood, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('β₀')
    ax1.set_ylabel('β₁')
    ax1.set_zlabel('Log-Likelihood')
    ax1.set_title('Log-Likelihood Surface')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    
    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(beta0_grid, beta1_grid, log_likelihood, 50, cmap='viridis')
    ax2.set_xlabel('β₀')
    ax2.set_ylabel('β₁')
    ax2.set_title('Log-Likelihood Contour Plot')
    ax2.plot(beta0_mle, beta1_mle, 'r*', markersize=10, 
             label=f'MLE: (β₀, β₁) = ({beta0_mle:.4f}, {beta1_mle:.4f})')
    ax2.legend()
    fig.colorbar(contour, ax=ax2, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return beta0_mle, beta1_mle

def plot_residuals(x, y, beta0, beta1, save_path=None):
    """Plot the residuals analysis"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Calculate residuals
    y_pred = beta0 + beta1 * x
    residuals = y - y_pred
    
    # Residuals vs Fitted
    ax1.scatter(y_pred, residuals, color='blue')
    ax1.axhline(y=0, color='r', linestyle='-')
    ax1.set_xlabel('Fitted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Fitted')
    ax1.grid(True, alpha=0.3)
    
    # QQ plot
    stats.probplot(residuals, plot=ax2)
    ax2.set_title('Normal Q-Q Plot')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def main():
    """Generate all visualizations for Question 11 of the L2.4 quiz"""
    # Data from the question
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.1, 3.5, 4.5, 5.8, 6.3])
    
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_4_Quiz_11")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 11 of the L2.4 MLE quiz (Linear Regression)...")
    
    # Calculate MLE for beta0 and beta1
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    beta1_mle = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
    beta0_mle = y_mean - beta1_mle * x_mean
    
    # Calculate MLE for sigma^2
    residuals = y - (beta0_mle + beta1_mle * x)
    sigma_squared_mle = np.sum(residuals**2) / n
    
    # Plot data and regression line
    plot_data_and_regression(x, y, beta0_mle, beta1_mle, save_path=os.path.join(save_dir, "regression_line.png"))
    print("1. Regression line visualization created")
    
    # Plot log-likelihood surface
    beta0_mle_plot, beta1_mle_plot = plot_log_likelihood_surface(
        x, y, sigma_squared_mle, save_path=os.path.join(save_dir, "log_likelihood_surface.png"))
    print("2. Log-likelihood surface visualization created")
    
    # Plot residuals analysis
    plot_residuals(x, y, beta0_mle, beta1_mle, save_path=os.path.join(save_dir, "residuals_analysis.png"))
    print("3. Residuals analysis visualization created")
    
    # Print results
    print("\nResults:")
    print(f"MLE for β₀: {beta0_mle:.4f}")
    print(f"MLE for β₁: {beta1_mle:.4f}")
    print(f"MLE for σ²: {sigma_squared_mle:.4f}")
    
    # Verify the results with sklearn
    try:
        from sklearn.linear_model import LinearRegression
        x_reshaped = x.reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(x_reshaped, y)
        print("\nVerification with sklearn:")
        print(f"β₀ (intercept): {lr.intercept_:.4f}")
        print(f"β₁ (slope): {lr.coef_[0]:.4f}")
    except:
        pass
    
    print(f"\nAll visualizations have been saved to: {save_dir}")

if __name__ == "__main__":
    main() 