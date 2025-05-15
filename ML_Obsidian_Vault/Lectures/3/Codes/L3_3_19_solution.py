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

# Set a nicer style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Set default font sizes for better readability
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})

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

# Fix the posterior distribution plot (Figure 4)
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
    mu_post = Sigma_post @ (Sigma_n_inv @ mu_n)  # Fixed posterior mean calculation
    
    return mu_post, Sigma_post

# Create visualizations
def create_visualizations(log_post, w0_map, w1_map, w0_ols, w1_ols, lambda0, lambda1):
    print("\nCreating visualizations...")
    saved_files = []
    
    # Plot 1: Data and regression lines
    plt.figure(figsize=(12, 8))
    
    # Create a grid of points for the regression lines
    x_line = np.linspace(0, 4, 100)
    y_map = w0_map + w1_map * x_line
    y_ols = w0_ols + w1_ols * x_line
    
    # Plot the data points
    plt.scatter(X, y, color='darkblue', s=120, label='Data points', zorder=3)
    
    # Plot the MAP regression line
    plt.plot(x_line, y_map, 'r-', linewidth=3, label=f'MAP: y = {w0_map:.4f} + {w1_map:.4f}x')
    
    # Plot the OLS regression line
    plt.plot(x_line, y_ols, 'g--', linewidth=3, label=f'OLS: y = {w0_ols:.4f} + {w1_ols:.4f}x')
    
    # Add a shaded region to show the effect of the priors
    plt.fill_between(x_line, y_map, y_ols, color='gray', alpha=0.2, label='Regularization effect')
    
    plt.title('Data with MAP and OLS Regression Lines')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    
    print("\nRegularization Effect Explanation:")
    print("- The MAP estimate is pulled toward zero due to the prior")
    print(f"- MAP estimates (w0={w0_map:.4f}, w1={w1_map:.4f}) differ from OLS estimates (w0={w0_ols:.4f}, w1={w1_ols:.4f})")
    print("- The gray shaded area shows the effect of regularization")
    
    plt.tight_layout()
    
    file_path = os.path.join(save_dir, "plot1_regression_lines.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    saved_files.append(file_path)
    plt.close()
    
    # Plot 2: Log posterior surface
    w0_range = np.linspace(-1, 3, 100)
    w1_range = np.linspace(0.5, 2.5, 100)
    W0, W1 = np.meshgrid(w0_range, w1_range)
    
    # Calculate the log posterior for each combination of w0 and w1
    Z = np.zeros_like(W0)
    for i in range(len(w0_range)):
        for j in range(len(w1_range)):
            Z[j, i] = log_post(w0_range[i], w1_range[j])
    
    # Create the figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1])
    
    # 3D surface plot
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    surf = ax1.plot_surface(W0, W1, Z, cmap=cm.viridis, linewidth=0, 
                           antialiased=True, alpha=0.8)
    
    # Mark the MAP estimate on the surface
    ax1.scatter([w0_map], [w1_map], [log_post(w0_map, w1_map)], 
               color='red', s=100, label='MAP estimate')
    
    # Mark the OLS estimate on the surface
    ax1.scatter([w0_ols], [w1_ols], [log_post(w0_ols, w1_ols)], 
               color='green', s=100, label='OLS estimate')
    
    ax1.set_xlabel('w0')
    ax1.set_ylabel('w1')
    ax1.set_zlabel('Log Posterior')
    ax1.set_title('Log Posterior Surface')
    ax1.legend()
    ax1.view_init(elev=30, azim=135)  # Better viewing angle
    
    # Contour plot
    ax2 = fig.add_subplot(gs[0, 1])
    contour = ax2.contourf(W0, W1, Z, cmap=cm.viridis, levels=20)
    ax2.scatter([w0_map], [w1_map], color='red', s=100, label='MAP')
    ax2.scatter([w0_ols], [w1_ols], color='green', s=100, label='OLS')
    ax2.set_xlabel('w0')
    ax2.set_ylabel('w1')
    ax2.set_title('Contour Plot')
    ax2.legend()
    
    # Create the colorbar for the contour plot
    cbar = fig.colorbar(contour, ax=ax2, shrink=0.8)
    cbar.set_label('Log Posterior')
    
    # w0 marginal
    ax3 = fig.add_subplot(gs[1, 0])
    # For each w0, find the w1 that maximizes the log posterior
    w0_marginal = [np.max(Z[:, i]) for i in range(len(w0_range))]
    ax3.plot(w0_range, w0_marginal, 'b-', linewidth=2)
    ax3.axvline(x=w0_map, color='red', linestyle='--', linewidth=2, label='MAP w0')
    ax3.axvline(x=w0_ols, color='green', linestyle='--', linewidth=2, label='OLS w0')
    ax3.set_xlabel('w0')
    ax3.set_ylabel('Max Log Posterior')
    ax3.set_title('w0 Marginal Distribution')
    ax3.legend()
    
    # w1 marginal
    ax4 = fig.add_subplot(gs[1, 1])
    # For each w1, find the w0 that maximizes the log posterior
    w1_marginal = [np.max(Z[j, :]) for j in range(len(w1_range))]
    ax4.plot(w1_range, w1_marginal, 'b-', linewidth=2)
    ax4.axvline(x=w1_map, color='red', linestyle='--', linewidth=2, label='MAP w1')
    ax4.axvline(x=w1_ols, color='green', linestyle='--', linewidth=2, label='OLS w1')
    ax4.set_xlabel('w1')
    ax4.set_ylabel('Max Log Posterior')
    ax4.set_title('w1 Marginal Distribution')
    ax4.legend()
    
    print("\nLog Posterior Analysis:")
    print(f"- The MAP estimates (w0={w0_map:.4f}, w1={w1_map:.4f}) correspond to the peak of the posterior distribution")
    print(f"- OLS estimates (w0={w0_ols:.4f}, w1={w1_ols:.4f}) maximize the likelihood but not the posterior")
    print("- The contour plot shows the joint distribution of w0 and w1")
    print("- The marginal plots show the maximum log posterior for each parameter value")
    
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
    
    plt.figure(figsize=(12, 8))
    
    # Plot the parameter values against regularization strength
    plt.plot(lambdas, w0_values, 'b-', linewidth=3, label='w0')
    plt.plot(lambdas, w1_values, 'r-', linewidth=3, label='w1')
    
    # Mark the actual MAP estimates (with separate λ0 and λ1)
    plt.axhline(y=w0_map, color='blue', linestyle='--', linewidth=2)
    plt.axhline(y=w1_map, color='red', linestyle='--', linewidth=2)
    
    # Mark the OLS estimates (no regularization)
    plt.scatter(lambdas[0], w0_ols, color='blue', s=150, marker='*', label='w0 OLS')
    plt.scatter(lambdas[0], w1_ols, color='red', s=150, marker='*', label='w1 OLS')
    
    # Add vertical lines for our specific regularization parameters
    plt.axvline(x=lambda0, color='blue', linestyle=':', linewidth=2, label=f'λ0 = {lambda0}')
    plt.axvline(x=lambda1, color='red', linestyle=':', linewidth=2, label=f'λ1 = {lambda1}')
    
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Regularization Parameter λ (log scale)')
    plt.ylabel('Parameter Value')
    plt.title('Ridge Path: How Parameters Vary with Regularization')
    plt.legend(loc='best')
    
    print("\nRidge Path Analysis:")
    print("- As regularization increases, parameter estimates shrink toward zero (the prior mean)")
    print(f"- MAP estimates with our regularization parameters (λ0={lambda0}, λ1={lambda1}): w0={w0_map:.4f}, w1={w1_map:.4f}")
    print(f"- OLS estimates (no regularization): w0={w0_ols:.4f}, w1={w1_ols:.4f}")
    print("- The plot shows how different regularization strengths affect the parameter estimates")
    
    plt.tight_layout()
    
    file_path = os.path.join(save_dir, "plot3_ridge_path.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    saved_files.append(file_path)
    plt.close()
    
    # Plot 4: Prior and Posterior Distributions
    mu_post, Sigma_post = calculate_posterior_mean_cov()
    
    # Create a grid of points
    w0_grid = np.linspace(-2, 4, 100)
    w1_grid = np.linspace(0, 3, 100)
    W0, W1 = np.meshgrid(w0_grid, w1_grid)
    pos = np.dstack((W0, W1))
    
    # Calculate the prior PDF
    rv_prior = stats.multivariate_normal([0, 0], np.diag([tau0_squared, tau1_squared]))
    prior_pdf = rv_prior.pdf(pos)
    
    # Calculate the posterior PDF
    rv_post = stats.multivariate_normal(mu_post, Sigma_post)
    post_pdf = rv_post.pdf(pos)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Prior contour plot
    ctf1 = axes[0].contourf(W0, W1, prior_pdf, levels=20, cmap=cm.viridis)
    axes[0].set_xlabel('w0')
    axes[0].set_ylabel('w1')
    axes[0].set_title('Prior Distribution: w0 ~ N(0, τ0²), w1 ~ N(0, τ1²)')
    cbar1 = plt.colorbar(ctf1, ax=axes[0])
    cbar1.set_label('Probability Density')
    
    # Mark the origin (prior mean)
    axes[0].scatter([0], [0], color='red', s=150, label='Prior Mean')
    
    # Draw 2-sigma ellipse for the prior
    theta = np.linspace(0, 2*np.pi, 100)
    ellipse_x = 2 * np.sqrt(tau0_squared) * np.cos(theta)
    ellipse_y = 2 * np.sqrt(tau1_squared) * np.sin(theta)
    axes[0].plot(ellipse_x, ellipse_y, 'r--', linewidth=2, alpha=0.7, label='2σ region')
    
    axes[0].legend(loc='upper right')
    axes[0].set_xlim(w0_grid.min(), w0_grid.max())
    axes[0].set_ylim(w1_grid.min(), w1_grid.max())
    
    # Posterior contour plot
    ctf2 = axes[1].contourf(W0, W1, post_pdf, levels=20, cmap=cm.viridis)
    axes[1].set_xlabel('w0')
    axes[1].set_ylabel('w1')
    axes[1].set_title('Posterior Distribution: P(w0, w1|X, y)')
    cbar2 = plt.colorbar(ctf2, ax=axes[1])
    cbar2.set_label('Probability Density')
    
    # Mark the MAP estimate (mode of posterior)
    axes[1].scatter([w0_map], [w1_map], color='red', s=150, label='MAP Estimate')
    
    # Mark the OLS estimate
    axes[1].scatter([w0_ols], [w1_ols], color='green', s=150, label='OLS Estimate')
    
    # Better visualization of data constraints
    for i, (xi, yi) in enumerate(zip(X, y)):
        # Each data point constrains w0 and w1 along a line
        w0_line = np.linspace(-1, 3, 100)
        w1_line = (yi - w0_line) / xi  # Rearranged from yi = w0 + w1*xi
        valid_idx = (w1_line >= w1_grid.min()) & (w1_line <= w1_grid.max())
        if np.any(valid_idx):
            axes[1].plot(w0_line[valid_idx], w1_line[valid_idx], 'k--', alpha=0.3, linewidth=2)
    
    # Draw 2-sigma ellipse for the posterior
    eigvals, eigvecs = np.linalg.eigh(Sigma_post)
    theta = np.linspace(0, 2*np.pi, 100)
    ellipse_x = mu_post[0] + 2 * np.sqrt(eigvals[0]) * np.cos(theta) * eigvecs[0, 0] + 2 * np.sqrt(eigvals[1]) * np.sin(theta) * eigvecs[0, 1]
    ellipse_y = mu_post[1] + 2 * np.sqrt(eigvals[0]) * np.cos(theta) * eigvecs[1, 0] + 2 * np.sqrt(eigvals[1]) * np.sin(theta) * eigvecs[1, 1]
    axes[1].plot(ellipse_x, ellipse_y, 'r--', linewidth=2, alpha=0.7, label='2σ region')
    
    axes[1].legend(loc='upper right')
    axes[1].set_xlim(w0_grid.min(), w0_grid.max())
    axes[1].set_ylim(w1_grid.min(), w1_grid.max())
    
    print("\nPrior and Posterior Analysis:")
    print(f"- Prior distribution: w0 ~ N(0, {tau0_squared}), w1 ~ N(0, {tau1_squared})")
    print("- The prior is centered at (0,0) with the 2σ region shown by the red dashed ellipse")
    print("- Posterior distribution combines the prior with data likelihood")
    print("- MAP estimate is the mode of the posterior distribution")
    print("- Data constraints are shown as dashed lines in the posterior plot")
    print("- The posterior is more concentrated than the prior due to the data information")

    legend = axes[0].legend()

    # Change the color of all legend text
    for text in legend.get_texts():
        text.set_color('white')

    legend = axes[1].legend()

    # Change the color of all legend text
    for text in legend.get_texts():
        text.set_color('white')
    
    plt.tight_layout()
    
    file_path = os.path.join(save_dir, "plot4_posterior_distributions.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    saved_files.append(file_path)
    plt.close()
    
    # Split Plot 5 into two separate plots
    # Plot 5a: Effect of Different Priors on MAP Regression Lines
    
    # Define different prior variance combinations to demonstrate their effect
    prior_combinations = [
        (100, 100, "Weak Priors (τ0²=100, τ1²=100)"),
        (10, 10, "Medium Priors (τ0²=10, τ1²=10)"),
        (10, 2, "Our Case (τ0²=10, τ1²=2)"),
        (2, 2, "Strong Priors (τ0²=2, τ1²=2)"),
        (1, 1, "Very Strong Priors (τ0²=1, τ1²=1)")
    ]
    
    # Create a design matrix
    X_design = np.column_stack((np.ones(n), X))
    
    # Calculate estimates for each prior combination
    estimates = []
    for tau0_sq, tau1_sq, label in prior_combinations:
        lambda0_temp = sigma_squared / tau0_sq
        lambda1_temp = sigma_squared / tau1_sq
        Lambda = np.diag([lambda0_temp, lambda1_temp])
        
        # Calculate MAP estimates
        w_map = np.linalg.solve(X_design.T @ X_design + Lambda, X_design.T @ y)
        estimates.append((w_map[0], w_map[1], label))
    
    # Calculate OLS estimate for comparison
    w_ols = np.linalg.solve(X_design.T @ X_design, X_design.T @ y)
    
    # Set up a color map
    colors = plt.cm.viridis(np.linspace(0, 1, len(prior_combinations)))
    
    # Plot 5a: Regression Lines for Different Priors
    plt.figure(figsize=(12, 8))
    
    # Plot the data points
    plt.scatter(X, y, color='darkblue', s=150, label='Data points', zorder=3)
    
    # Plot the OLS line as a reference
    x_line = np.linspace(0, 4, 100)
    y_ols = w_ols[0] + w_ols[1] * x_line
    plt.plot(x_line, y_ols, 'k--', linewidth=3, label=f'OLS: y = {w_ols[0]:.4f} + {w_ols[1]:.4f}x')
    
    # Plot the regression lines for different priors
    for i, (w0, w1, label) in enumerate(estimates):
        y_pred = w0 + w1 * x_line
        plt.plot(x_line, y_pred, color=colors[i], linewidth=3, 
                label=f'{label}: y = {w0:.4f} + {w1:.4f}x')
    
    plt.title('Effect of Different Priors on MAP Regression Lines')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    
    print("\nEffect of Different Priors on Regression Lines:")
    print("- Stronger priors (smaller variances) pull the estimates more strongly toward the prior mean (0,0)")
    print("- Weak priors result in estimates close to OLS")
    print("- Very strong priors force the line to be nearly flat (close to y = 0)")
    print("- Our case (τ0²=10, τ1²=2) shows moderate regularization")
    
    plt.tight_layout()
    
    file_path = os.path.join(save_dir, "plot5a_prior_effect_lines.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    saved_files.append(file_path)
    plt.close()
    
    # Plot 5b: Prior Distribution Shapes
    plt.figure(figsize=(10, 8))
    
    # Create 2-sigma ellipses for each prior
    theta = np.linspace(0, 2*np.pi, 100)
    
    for i, (tau0_sq, tau1_sq, label) in enumerate(prior_combinations):
        ellipse_x = 2 * np.sqrt(tau0_sq) * np.cos(theta)
        ellipse_y = 2 * np.sqrt(tau1_sq) * np.sin(theta)
        plt.plot(ellipse_x, ellipse_y, color=colors[i], linewidth=3, label=label)
    
    plt.scatter([0], [0], color='black', s=100, label='Prior Mean')
    plt.xlabel('w0')
    plt.ylabel('w1')
    plt.title('Prior Distributions (2σ Regions)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.axis('equal')  # Equal scaling for better visualization of ellipses
    
    print("\nPrior Distribution Shapes:")
    print("- Each ellipse shows the 2σ region of a different prior distribution")
    print("- Larger ellipses correspond to weaker priors (larger variances)")
    print("- Smaller ellipses correspond to stronger priors (smaller variances)")
    print("- All priors are centered at (0,0)")
    print("- The ellipse shape shows the relative strength of regularization for each parameter")
    
    plt.tight_layout()
    
    file_path = os.path.join(save_dir, "plot5b_prior_shapes.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    saved_files.append(file_path)
    plt.close()
    
    # Plot 6: Simple Likelihood, Prior, and Posterior Distributions for w0
    plt.figure(figsize=(12, 8))
    
    # Generate values for visualization
    w0_values = np.linspace(-2, 4, 1000)
    
    # Calculate likelihood for w0 (assuming optimal w1 for each w0)
    likelihood = np.zeros_like(w0_values)
    for i, w0 in enumerate(w0_values):
        # For each w0, find the best w1 to maximize likelihood
        w1_values = np.linspace(0, 3, 100)
        best_nll = float('inf')  # Best negative log likelihood
        for w1 in w1_values:
            nll = np.sum((y - w0 - w1*X)**2) / (2*sigma_squared)
            if nll < best_nll:
                best_nll = nll
        likelihood[i] = np.exp(-best_nll)
    
    # Normalize for better visualization
    likelihood = likelihood / np.max(likelihood)
    
    # Calculate prior for w0
    prior = np.exp(-(w0_values**2) / (2*tau0_squared)) / np.sqrt(2*np.pi*tau0_squared)
    prior = prior / np.max(prior)  # Normalize
    
    # Calculate posterior (proportional to likelihood * prior)
    posterior = likelihood * prior
    posterior = posterior / np.max(posterior)  # Normalize
    
    # Plot the distributions
    plt.plot(w0_values, likelihood, 'b-', linewidth=3, label='Likelihood')
    plt.plot(w0_values, prior, 'g-', linewidth=3, label='Prior')
    plt.plot(w0_values, posterior, 'r-', linewidth=3, label='Posterior')
    
    # Add vertical lines for key values
    plt.axvline(x=0, color='green', linestyle='--', linewidth=2, label='Prior Mean')
    plt.axvline(x=w0_ols, color='blue', linestyle='--', linewidth=2, label='MLE (OLS)')
    plt.axvline(x=w0_map, color='red', linestyle='--', linewidth=2, label='MAP')
    
    plt.title('Likelihood, Prior and Posterior for Parameter w0')
    plt.xlabel('w0')
    plt.ylabel('Normalized Density')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    print("\nLikelihood, Prior, and Posterior for w0:")
    print("- The likelihood (blue) peaks at the MLE (OLS estimate)")
    print("- The prior (green) is centered at 0 with variance τ0²")
    print("- The posterior (red) is proportional to the product of likelihood and prior")
    print("- The MAP estimate lies between the MLE and the prior mean due to the regularization effect")
    print(f"- Prior Mean = 0, MLE = {w0_ols:.4f}, MAP = {w0_map:.4f}")
    
    plt.tight_layout()
    
    file_path = os.path.join(save_dir, "plot6_marginal_distributions.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    saved_files.append(file_path)
    plt.close()
    
    print(f"\nVisualizations saved to: {save_dir}")
    
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
    
    # Create visualizations - Pass lambda0 and lambda1 to the function
    saved_files = create_visualizations(log_post, w0_map, w1_map, w0_ols, w1_ols, lambda0, lambda1)
    
    # Summary
    print("\nQuestion 19 Solution Summary:")
    print(f"1. We derived the posterior distribution for parameters w0 and w1.")
    print(f"2. We computed the logarithm of the posterior distribution.")
    print(f"3. We showed that MAP estimation with Gaussian priors is equivalent to ridge regression with λ0 = {lambda0} and λ1 = {lambda1}.")
    print(f"4. The MAP estimates are w0 = {w0_map:.4f} and w1 = {w1_map:.4f}.")
    print(f"   (For comparison, the OLS estimates are w0 = {w0_ols:.4f} and w1 = {w1_ols:.4f}.)")

if __name__ == "__main__":
    main() 