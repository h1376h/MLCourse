import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.stats as stats
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_3_Quiz_19")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Given dataset
X = np.array([1, 2, 3])
y = np.array([2, 4, 5])
n = len(X)

# Given parameters
sigma_squared = 1  # Noise variance
tau0_squared = 10  # Prior variance for w0
tau1_squared = 2   # Prior variance for w1

# Helper function to print mathematical expressions
def print_math(text):
    print(text)
    print()

# Step 1: Posterior distribution for the parameters
def derive_posterior():
    print("Step 1: Derive the posterior distribution for the parameters w0 and w1")
    
    print_math("Using Bayes' rule, the posterior distribution is proportional to the likelihood times the prior:")
    print_math("P(w0, w1|X, y) ∝ P(y|X, w0, w1) × P(w0, w1)")
    
    print_math("The likelihood P(y|X, w0, w1) under the noise model ε ~ N(0, σ²) is:")
    print_math("P(y|X, w0, w1) = ∏_{i=1}^n N(y_i|w0 + w1x_i, σ²)")
    print_math("= (2πσ²)^(-n/2) × exp[-1/(2σ²) × ∑_{i=1}^n (y_i - w0 - w1x_i)²]")
    
    print_math("The prior for the parameters is:")
    print_math("P(w0, w1) = N(w0|0, τ0²) × N(w1|0, τ1²)")
    print_math("= (2πτ0²)^(-1/2) × exp[-w0²/(2τ0²)] × (2πτ1²)^(-1/2) × exp[-w1²/(2τ1²)]")
    
    print_math("The posterior is proportional to the product of likelihood and prior:")
    print_math("P(w0, w1|X, y) ∝ exp[-1/(2σ²) × ∑_{i=1}^n (y_i - w0 - w1x_i)²] × exp[-w0²/(2τ0²)] × exp[-w1²/(2τ1²)]")
    
    print_math("This is a bivariate Gaussian distribution with mean vector μ and covariance matrix Σ that can be derived by completing the squares in the exponent.")
    
    return

# Step 2: Log of the posterior distribution
def log_posterior():
    print("\nStep 2: Derive the logarithm of the posterior distribution")
    
    print_math("Taking the logarithm of the posterior (ignoring constant terms):")
    print_math("log P(w0, w1|X, y) ∝ -1/(2σ²) × ∑_{i=1}^n (y_i - w0 - w1x_i)² - w0²/(2τ0²) - w1²/(2τ1²)")
    
    print_math("Rearranging to highlight the ridge regression form:")
    print_math("log P(w0, w1|X, y) ∝ -1/(2σ²) [∑_{i=1}^n (y_i - w0 - w1x_i)² + (σ²/τ0²)w0² + (σ²/τ1²)w1²]")
    
    # For our specific dataset, let's compute the log posterior explicitly
    def log_post(w0, w1):
        likelihood_term = np.sum((y - w0 - w1*X)**2) / (2*sigma_squared)
        prior_w0_term = (w0**2) / (2*tau0_squared)
        prior_w1_term = (w1**2) / (2*tau1_squared)
        return -(likelihood_term + prior_w0_term + prior_w1_term)  # Negative because we want to maximize
    
    return log_post

# Step 3: MAP estimation and ridge regression equivalence
def show_ridge_equivalence():
    print("\nStep 3: Show that MAP estimation with Gaussian priors is equivalent to ridge regression")
    
    print_math("The negative log posterior (ignoring constant terms) is:")
    print_math("- log P(w0, w1|X, y) ∝ 1/(2σ²) [∑_{i=1}^n (y_i - w0 - w1x_i)² + (σ²/τ0²)w0² + (σ²/τ1²)w1²]")
    
    print_math("Since we want to maximize the posterior (or minimize the negative log posterior), this is equivalent to minimizing:")
    print_math("L(w0, w1) = ∑_{i=1}^n (y_i - w0 - w1x_i)² + λ0 w0² + λ1 w1²")
    
    print_math("where λ0 = σ²/τ0² and λ1 = σ²/τ1² are the regularization parameters.")
    
    lambda0 = sigma_squared / tau0_squared
    lambda1 = sigma_squared / tau1_squared
    
    print_math(f"For our given parameters (σ² = {sigma_squared}, τ0² = {tau0_squared}, τ1² = {tau1_squared}):")
    print_math(f"λ0 = σ²/τ0² = {sigma_squared}/{tau0_squared} = {lambda0}")
    print_math(f"λ1 = σ²/τ1² = {sigma_squared}/{tau1_squared} = {lambda1}")
    
    print_math("This is a form of ridge regression with different regularization parameters for each coefficient.")
    print_math("If τ0² = τ1² = τ², then we would have the standard ridge regression with a single regularization parameter λ = σ²/τ².")
    
    return lambda0, lambda1

# Step 4: Calculate the MAP estimates
def calculate_map_estimates(lambda0, lambda1):
    print("\nStep 4: Calculate the MAP estimates for w0 and w1")
    
    print_math("To find the MAP estimates, we need to minimize the negative log posterior:")
    print_math("L(w0, w1) = ∑_{i=1}^n (y_i - w0 - w1x_i)² + λ0 w0² + λ1 w1²")
    
    print_math("Taking partial derivatives and setting them to zero:")
    print_math("∂L/∂w0 = -2∑_{i=1}^n (y_i - w0 - w1x_i) + 2λ0 w0 = 0")
    print_math("∂L/∂w1 = -2∑_{i=1}^n (y_i - w0 - w1x_i)x_i + 2λ1 w1 = 0")
    
    print_math("These equations can be rewritten in matrix form as:")
    print_math("(X^T X + Λ) w = X^T y")
    
    print_math("where X is the design matrix (with a column of ones for the intercept), Λ is the diagonal matrix of regularization parameters, and w = [w0, w1]^T.")
    
    # Create the design matrix
    X_design = np.column_stack((np.ones(n), X))
    
    # Create the regularization matrix
    Lambda = np.diag([lambda0, lambda1])
    
    # Calculate the MAP estimates
    w_map = np.linalg.solve(X_design.T @ X_design + Lambda, X_design.T @ y)
    
    w0_map, w1_map = w_map
    
    print_math(f"For our dataset and parameters, the MAP estimates are:")
    print_math(f"w0 = {w0_map:.4f}")
    print_math(f"w1 = {w1_map:.4f}")
    
    # Calculate the OLS estimates for comparison
    w_ols = np.linalg.solve(X_design.T @ X_design, X_design.T @ y)
    w0_ols, w1_ols = w_ols
    
    print_math(f"For comparison, the OLS estimates (without regularization) are:")
    print_math(f"w0 = {w0_ols:.4f}")
    print_math(f"w1 = {w1_ols:.4f}")
    
    return w0_map, w1_map, w0_ols, w1_ols

# Create visualizations
def create_visualizations(log_post, w0_map, w1_map, w0_ols, w1_ols):
    print("\nCreating visualizations...")
    saved_files = []
    
    # Plot 1: Data and regression lines
    plt.figure(figsize=(10, 6))
    
    # Plot the data points
    plt.scatter(X, y, color='blue', s=60, label='Data points')
    
    # Plot the MAP regression line
    x_line = np.linspace(0, 4, 100)
    y_map = w0_map + w1_map * x_line
    plt.plot(x_line, y_map, 'r-', linewidth=2, label=f'MAP: y = {w0_map:.4f} + {w1_map:.4f}x')
    
    # Plot the OLS regression line
    y_ols = w0_ols + w1_ols * x_line
    plt.plot(x_line, y_ols, 'g--', linewidth=2, label=f'OLS: y = {w0_ols:.4f} + {w1_ols:.4f}x')
    
    plt.title('Data with MAP and OLS Regression Lines', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    file_path = os.path.join(save_dir, "plot1_regression_lines.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    saved_files.append(file_path)
    plt.close()
    
    # Plot 2: Log posterior surface
    w0_range = np.linspace(-2, 4, 100)
    w1_range = np.linspace(0, 3, 100)
    W0, W1 = np.meshgrid(w0_range, w1_range)
    
    # Calculate the log posterior for each combination of w0 and w1
    Z = np.zeros_like(W0)
    for i in range(len(w0_range)):
        for j in range(len(w1_range)):
            Z[j, i] = log_post(w0_range[i], w1_range[j])
    
    # Create the 3D surface plot
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1])
    
    # 3D surface
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    surf = ax1.plot_surface(W0, W1, Z, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.8)
    
    # Mark the MAP estimate on the surface
    ax1.scatter([w0_map], [w1_map], [log_post(w0_map, w1_map)], color='red', s=50, label='MAP estimate')
    
    # Mark the OLS estimate on the surface
    ax1.scatter([w0_ols], [w1_ols], [log_post(w0_ols, w1_ols)], color='green', s=50, label='OLS estimate')
    
    ax1.set_xlabel('w0', fontsize=12)
    ax1.set_ylabel('w1', fontsize=12)
    ax1.set_zlabel('Log Posterior', fontsize=12)
    ax1.set_title('Log Posterior Surface', fontsize=14)
    
    # Contour plot
    ax2 = fig.add_subplot(gs[0, 1])
    contour = ax2.contourf(W0, W1, Z, cmap=cm.viridis, levels=20)
    ax2.scatter([w0_map], [w1_map], color='red', s=50, label='MAP estimate')
    ax2.scatter([w0_ols], [w1_ols], color='green', s=50, label='OLS estimate')
    ax2.set_xlabel('w0', fontsize=10)
    ax2.set_ylabel('w1', fontsize=10)
    ax2.set_title('Contour Plot', fontsize=12)
    
    # Create the colorbar for the contour plot
    fig.colorbar(contour, ax=ax2, shrink=0.8, label='Log Posterior')
    
    # w0 marginal
    ax3 = fig.add_subplot(gs[1, 0])
    # For each w0, find the w1 that maximizes the log posterior
    w0_marginal = [np.max(Z[:, i]) for i in range(len(w0_range))]
    ax3.plot(w0_range, w0_marginal, 'b-')
    ax3.axvline(x=w0_map, color='red', linestyle='--', label='MAP w0')
    ax3.axvline(x=w0_ols, color='green', linestyle='--', label='OLS w0')
    ax3.set_xlabel('w0', fontsize=10)
    ax3.set_ylabel('Max Log Posterior', fontsize=10)
    ax3.set_title('w0 Marginal', fontsize=12)
    ax3.legend(fontsize=8)
    
    # w1 marginal
    ax4 = fig.add_subplot(gs[1, 1])
    # For each w1, find the w0 that maximizes the log posterior
    w1_marginal = [np.max(Z[j, :]) for j in range(len(w1_range))]
    ax4.plot(w1_range, w1_marginal, 'b-')
    ax4.axvline(x=w1_map, color='red', linestyle='--', label='MAP w1')
    ax4.axvline(x=w1_ols, color='green', linestyle='--', label='OLS w1')
    ax4.set_xlabel('w1', fontsize=10)
    ax4.set_ylabel('Max Log Posterior', fontsize=10)
    ax4.set_title('w1 Marginal', fontsize=12)
    ax4.legend(fontsize=8)
    
    plt.tight_layout()
    
    file_path = os.path.join(save_dir, "plot2_log_posterior.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    saved_files.append(file_path)
    plt.close()
    
    # Plot 3: Ridge path for different regularization values
    lambdas = np.logspace(-3, 1, 100)
    w0_values = []
    w1_values = []
    
    # Calculate MAP estimates for different regularization parameters (assuming λ0 = λ1 = λ)
    for lam in lambdas:
        Lambda = np.diag([lam, lam])
        X_design = np.column_stack((np.ones(n), X))
        w_map = np.linalg.solve(X_design.T @ X_design + Lambda, X_design.T @ y)
        w0_values.append(w_map[0])
        w1_values.append(w_map[1])
    
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, w0_values, 'b-', linewidth=2, label='w0')
    plt.plot(lambdas, w1_values, 'r-', linewidth=2, label='w1')
    
    # Mark the actual MAP estimates (with separate λ0 and λ1)
    plt.axhline(y=w0_map, color='blue', linestyle='--')
    plt.axhline(y=w1_map, color='red', linestyle='--')
    plt.text(0.1, w0_map+0.1, f'w0 MAP = {w0_map:.4f}', fontsize=10)
    plt.text(0.1, w1_map+0.1, f'w1 MAP = {w1_map:.4f}', fontsize=10)
    
    # Mark the OLS estimates (no regularization)
    plt.scatter(0, w0_ols, color='blue', s=50, label='w0 OLS')
    plt.scatter(0, w1_ols, color='red', s=50, label='w1 OLS')
    
    plt.xscale('log')
    plt.grid(True)
    plt.xlabel('Regularization Parameter λ (log scale)', fontsize=12)
    plt.ylabel('Parameter Value', fontsize=12)
    plt.title('Ridge Path: How Parameters Vary with Regularization', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    file_path = os.path.join(save_dir, "plot3_ridge_path.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    saved_files.append(file_path)
    plt.close()
    
    # Plot 4: Posterior distributions
    def calculate_posterior_mean_cov():
        # For a Bayesian linear regression with Gaussian prior and likelihood, the posterior is Gaussian
        X_design = np.column_stack((np.ones(n), X))
        Sigma_0_inv = np.diag([1/tau0_squared, 1/tau1_squared])  # Prior precision
        Sigma_n_inv = X_design.T @ X_design / sigma_squared  # Data precision
        
        # Posterior precision is the sum of prior and data precision
        Sigma_post_inv = Sigma_0_inv + Sigma_n_inv
        Sigma_post = np.linalg.inv(Sigma_post_inv)
        
        # Posterior mean
        mu_0 = np.array([0, 0])  # Prior mean
        mu_n = np.linalg.solve(X_design.T @ X_design, X_design.T @ y)  # OLS estimate
        mu_post = Sigma_post @ (Sigma_0_inv @ mu_0 + Sigma_n_inv @ mu_n)
        
        return mu_post, Sigma_post
    
    mu_post, Sigma_post = calculate_posterior_mean_cov()
    
    # Create a grid of points
    w0_grid = np.linspace(-1, 4, 100)
    w1_grid = np.linspace(0, 3, 100)
    W0, W1 = np.meshgrid(w0_grid, w1_grid)
    pos = np.dstack((W0, W1))
    
    # Calculate the prior PDF
    rv_prior = stats.multivariate_normal([0, 0], np.diag([tau0_squared, tau1_squared]))
    prior_pdf = rv_prior.pdf(pos)
    
    # Calculate the posterior PDF
    rv_post = stats.multivariate_normal(mu_post, Sigma_post)
    post_pdf = rv_post.pdf(pos)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prior contour plot
    ctf1 = axes[0].contourf(W0, W1, prior_pdf, levels=20, cmap=cm.viridis)
    axes[0].set_xlabel('w0', fontsize=12)
    axes[0].set_ylabel('w1', fontsize=12)
    axes[0].set_title('Prior Distribution: w0 ~ N(0, τ0²), w1 ~ N(0, τ1²)', fontsize=14)
    plt.colorbar(ctf1, ax=axes[0], label='Probability Density')
    
    # Mark the origin (prior mean)
    axes[0].scatter([0], [0], color='red', s=50, label='Prior Mean')
    
    # Draw 2-sigma ellipse for the prior
    theta = np.linspace(0, 2*np.pi, 100)
    ellipse_x = 2 * np.sqrt(tau0_squared) * np.cos(theta)
    ellipse_y = 2 * np.sqrt(tau1_squared) * np.sin(theta)
    axes[0].plot(ellipse_x, ellipse_y, 'r--', alpha=0.5, label='2σ region')
    
    axes[0].legend(fontsize=10)
    
    # Posterior contour plot
    ctf2 = axes[1].contourf(W0, W1, post_pdf, levels=20, cmap=cm.viridis)
    axes[1].set_xlabel('w0', fontsize=12)
    axes[1].set_ylabel('w1', fontsize=12)
    axes[1].set_title('Posterior Distribution: P(w0, w1|X, y)', fontsize=14)
    plt.colorbar(ctf2, ax=axes[1], label='Probability Density')
    
    # Mark the MAP estimate
    axes[1].scatter([w0_map], [w1_map], color='red', s=50, label='MAP Estimate')
    
    # Mark the OLS estimate
    axes[1].scatter([w0_ols], [w1_ols], color='green', s=50, label='OLS Estimate')
    
    # Draw data points in w-space
    axes[1].plot(w0_range, -(w0_range * 1 - 2) / 1, 'k-', alpha=0.2, linewidth=1)
    axes[1].plot(w0_range, -(w0_range * 2 - 4) / 2, 'k-', alpha=0.2, linewidth=1)
    axes[1].plot(w0_range, -(w0_range * 3 - 5) / 3, 'k-', alpha=0.2, linewidth=1)
    
    # Draw 2-sigma ellipse for the posterior
    eigvals, eigvecs = np.linalg.eigh(Sigma_post)
    theta = np.linspace(0, 2*np.pi, 100)
    ellipse_x = mu_post[0] + 2 * np.sqrt(eigvals[0]) * np.cos(theta) * eigvecs[0, 0] + 2 * np.sqrt(eigvals[1]) * np.sin(theta) * eigvecs[0, 1]
    ellipse_y = mu_post[1] + 2 * np.sqrt(eigvals[0]) * np.cos(theta) * eigvecs[1, 0] + 2 * np.sqrt(eigvals[1]) * np.sin(theta) * eigvecs[1, 1]
    axes[1].plot(ellipse_x, ellipse_y, 'r--', alpha=0.5, label='2σ region')
    
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    
    file_path = os.path.join(save_dir, "plot4_posterior_distributions.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    saved_files.append(file_path)
    plt.close()
    
    print(f"Visualizations saved to: {', '.join(saved_files)}")
    
    return saved_files

# Main execution
def main():
    # Step 1: Derive the posterior distribution
    derive_posterior()
    
    # Step 2: Derive the log posterior
    log_post = log_posterior()
    
    # Step 3: Show equivalence to ridge regression
    lambda0, lambda1 = show_ridge_equivalence()
    
    # Step 4: Calculate MAP estimates
    w0_map, w1_map, w0_ols, w1_ols = calculate_map_estimates(lambda0, lambda1)
    
    # Create visualizations
    saved_files = create_visualizations(log_post, w0_map, w1_map, w0_ols, w1_ols)
    
    # Summary
    print("\nQuestion 19 Solution Summary:")
    print(f"1. We derived the posterior distribution for parameters w0 and w1.")
    print(f"2. We computed the logarithm of the posterior distribution.")
    print(f"3. We showed that MAP estimation with Gaussian priors is equivalent to ridge regression with λ0 = {lambda0} and λ1 = {lambda1}.")
    print(f"4. The MAP estimates are w0 = {w0_map:.4f} and w1 = {w1_map:.4f}.")
    print(f"   (For comparison, the OLS estimates are w0 = {w0_ols:.4f} and w1 = {w1_ols:.4f}.)")

if __name__ == "__main__":
    main() 