import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.special import gamma, digamma
import os
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_3_Quiz_21")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
saved_files = []

def plot_inverse_gamma(alpha, beta, save_path=None):
    """Plot the inverse-gamma distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.linspace(0.01, 10, 1000)
    y = stats.invgamma.pdf(x, alpha, scale=beta)
    
    ax.plot(x, y, 'b-', linewidth=2)
    ax.fill_between(x, 0, y, alpha=0.2, color='blue')
    
    # Calculate mean and mode
    mean = beta / (alpha - 1) if alpha > 1 else np.nan
    mode = beta / (alpha + 1)
    
    # Add vertical lines for mean and mode
    if not np.isnan(mean):
        ax.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')
    ax.axvline(mode, color='green', linestyle='--', label=f'Mode: {mode:.2f}')
    
    ax.set_title(f'Inverse-Gamma Distribution (α={alpha}, β={beta})', fontsize=14)
    ax.set_xlabel('σ²', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        saved_files.append(save_path)
    
    return fig, ax

def generate_linear_data(w_true, num_samples, x_range, sigma_true):
    """Generate data from a linear model with Gaussian noise."""
    # For simplicity, just use 1D data
    X = np.random.uniform(x_range[0], x_range[1], num_samples).reshape(-1, 1)
    
    # Add a column of ones for the intercept
    X_with_intercept = np.column_stack([np.ones(num_samples), X])
    
    # Generate the true outputs (without noise)
    y_true = X_with_intercept @ w_true
    
    # Add Gaussian noise
    noise = np.random.normal(0, np.sqrt(sigma_true), num_samples)
    y = y_true + noise
    
    return X, y, y_true

def plot_joint_prior(save_path=None):
    """Plot the joint prior distribution for w and sigma^2."""
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3], figure=fig)
    
    # Create a grid for w and sigma^2
    w_range = np.linspace(-3, 3, 100)
    sigma2_range = np.linspace(0.1, 5, 100)
    w_mesh, sigma2_mesh = np.meshgrid(w_range, sigma2_range)
    
    # Prior for w (normal distribution)
    w_prior_mean = 0
    w_prior_var = 1
    w_prior = stats.norm.pdf(w_mesh, w_prior_mean, np.sqrt(w_prior_var))
    
    # Prior for sigma^2 (inverse-gamma)
    alpha_prior = 2
    beta_prior = 4
    sigma2_prior = stats.invgamma.pdf(sigma2_mesh, alpha_prior, scale=beta_prior)
    
    # Joint prior (assuming independence)
    joint_prior = w_prior * sigma2_prior
    
    # Plot the joint prior as a color mesh
    ax_joint = fig.add_subplot(gs[1, 0])
    cf = ax_joint.contourf(w_mesh, sigma2_mesh, joint_prior, cmap='viridis', levels=20)
    plt.colorbar(cf, ax=ax_joint, label='Joint Probability Density')
    ax_joint.set_xlabel('w (slope coefficient)')
    ax_joint.set_ylabel('σ² (noise variance)')
    ax_joint.set_title('Joint Prior Distribution p(w, σ²)')
    
    # Plot the marginal priors
    ax_w = fig.add_subplot(gs[0, 0])
    ax_w.plot(w_range, stats.norm.pdf(w_range, w_prior_mean, np.sqrt(w_prior_var)))
    ax_w.fill_between(w_range, 0, stats.norm.pdf(w_range, w_prior_mean, np.sqrt(w_prior_var)), alpha=0.2)
    ax_w.set_title('Prior for w ~ N(0, 1)')
    ax_w.set_xlim(w_range[0], w_range[-1])
    ax_w.set_ylim(bottom=0)
    ax_w.set_ylabel('p(w)')
    
    ax_sigma2 = fig.add_subplot(gs[1, 1])
    ax_sigma2.plot(stats.invgamma.pdf(sigma2_range, alpha_prior, scale=beta_prior), sigma2_range)
    ax_sigma2.fill_betweenx(sigma2_range, 0, stats.invgamma.pdf(sigma2_range, alpha_prior, scale=beta_prior), alpha=0.2)
    ax_sigma2.set_title('Prior for σ² ~ InvGamma(2, 4)')
    ax_sigma2.set_ylim(sigma2_range[0], sigma2_range[-1])
    ax_sigma2.set_xlim(left=0)
    ax_sigma2.set_xlabel('p(σ²)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        saved_files.append(save_path)
    
    return fig

def plot_joint_posterior(X, y, w_prior_mean, w_prior_var, alpha_prior, beta_prior, save_path=None):
    """Plot the joint posterior distribution for w and sigma^2."""
    # For simplicity, we'll work with a simple linear regression with intercept
    n = len(y)
    X_with_intercept = np.column_stack([np.ones(n), X])
    
    # Compute the maximum likelihood estimate for w
    w_mle = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
    
    # Compute the sum of squared residuals
    y_pred = X_with_intercept @ w_mle
    residuals = y - y_pred
    ssr = np.sum(residuals**2)
    
    # Compute the posterior parameters
    XTX = X_with_intercept.T @ X_with_intercept
    XTy = X_with_intercept.T @ y
    
    # For simplicity, we'll just focus on the slope coefficient
    # Extract the relevant parts for the slope (ignoring intercept)
    slope_idx = 1  # Index for the slope coefficient
    
    # Prior precision for the slope
    prior_precision = 1 / w_prior_var
    
    # Posterior precision for the slope
    slope_posterior_precision = prior_precision + XTX[slope_idx, slope_idx]
    slope_posterior_var = 1 / slope_posterior_precision
    
    # Posterior mean for the slope
    slope_posterior_mean = slope_posterior_var * (prior_precision * w_prior_mean + XTX[slope_idx, :] @ w_mle)
    
    # Posterior parameters for sigma^2
    alpha_posterior = alpha_prior + n/2
    beta_posterior = beta_prior + ssr/2
    
    # Create a figure
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3], figure=fig)
    
    # Create a grid for w (slope) and sigma^2
    w_range = np.linspace(w_mle[slope_idx] - 3, w_mle[slope_idx] + 3, 100)
    sigma2_range = np.linspace(0.1, beta_posterior/(alpha_posterior-1)*3, 100)
    w_mesh, sigma2_mesh = np.meshgrid(w_range, sigma2_range)
    
    # Compute the posterior distributions
    # For slope
    w_posterior = stats.norm.pdf(w_mesh, slope_posterior_mean, np.sqrt(slope_posterior_var))
    
    # For sigma^2
    sigma2_posterior = stats.invgamma.pdf(sigma2_mesh, alpha_posterior, scale=beta_posterior)
    
    # Joint posterior (we'll assume conditional independence for visualization)
    joint_posterior = w_posterior * sigma2_posterior
    
    # Plot the joint posterior as a color mesh
    ax_joint = fig.add_subplot(gs[1, 0])
    cf = ax_joint.contourf(w_mesh, sigma2_mesh, joint_posterior, cmap='viridis', levels=20)
    plt.colorbar(cf, ax=ax_joint, label='Joint Probability Density')
    ax_joint.set_xlabel('w (slope coefficient)')
    ax_joint.set_ylabel('σ² (noise variance)')
    ax_joint.set_title('Joint Posterior Distribution p(w, σ²|X,y)')
    
    # Plot the marginal posteriors
    ax_w = fig.add_subplot(gs[0, 0])
    ax_w.plot(w_range, stats.norm.pdf(w_range, slope_posterior_mean, np.sqrt(slope_posterior_var)))
    ax_w.fill_between(w_range, 0, stats.norm.pdf(w_range, slope_posterior_mean, np.sqrt(slope_posterior_var)), alpha=0.2)
    ax_w.axvline(w_mle[slope_idx], color='red', linestyle='--', label=f'MLE: {w_mle[slope_idx]:.2f}')
    ax_w.axvline(slope_posterior_mean, color='green', linestyle='--', label=f'Posterior Mean: {slope_posterior_mean:.2f}')
    ax_w.set_title('Posterior for w (slope)')
    ax_w.set_xlim(w_range[0], w_range[-1])
    ax_w.set_ylim(bottom=0)
    ax_w.set_ylabel('p(w|X,y)')
    ax_w.legend()
    
    ax_sigma2 = fig.add_subplot(gs[1, 1])
    ax_sigma2.plot(stats.invgamma.pdf(sigma2_range, alpha_posterior, scale=beta_posterior), sigma2_range)
    ax_sigma2.fill_betweenx(sigma2_range, 0, stats.invgamma.pdf(sigma2_range, alpha_posterior, scale=beta_posterior), alpha=0.2)
    
    # MLE for noise variance
    sigma2_mle = ssr / n
    sigma2_posterior_mean = beta_posterior / (alpha_posterior - 1) if alpha_posterior > 1 else float('inf')
    
    ax_sigma2.axhline(sigma2_mle, color='red', linestyle='--', label=f'MLE: {sigma2_mle:.2f}')
    ax_sigma2.axhline(sigma2_posterior_mean, color='green', linestyle='--', label=f'Posterior Mean: {sigma2_posterior_mean:.2f}')
    ax_sigma2.set_title('Posterior for σ²')
    ax_sigma2.set_ylim(sigma2_range[0], sigma2_range[-1])
    ax_sigma2.set_xlim(left=0)
    ax_sigma2.set_xlabel('p(σ²|X,y)')
    ax_sigma2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        saved_files.append(save_path)
    
    return fig, (slope_posterior_mean, slope_posterior_var, alpha_posterior, beta_posterior)

def plot_marginal_likelihood(X, y, save_path=None):
    """Visualize the concept of marginal likelihood and model comparison."""
    # For simplicity, consider different polynomial degrees as different models
    max_degree = 5
    n_samples = len(y)
    X = X.flatten()  # Ensure X is 1D
    
    # Compute the log marginal likelihood for each model
    log_marginal_likelihoods = []
    
    # Set common priors
    alpha_prior = 2
    beta_prior = 4
    w_prior_var = 1.0
    
    # For each polynomial degree
    degrees = range(1, max_degree + 1)
    bic_scores = []
    aic_scores = []
    
    for degree in degrees:
        # Create polynomial features
        X_poly = np.column_stack([X ** i for i in range(1, degree + 1)])
        
        # Add a column of ones for the intercept
        X_poly = np.column_stack([np.ones(n_samples), X_poly])
        
        # Compute maximum likelihood estimates
        w_mle = np.linalg.lstsq(X_poly, y, rcond=None)[0]
        
        # Compute residuals and RSS
        y_pred = X_poly @ w_mle
        residuals = y - y_pred
        rss = np.sum(residuals ** 2)
        
        # Compute BIC and AIC (approximations to negative log marginal likelihood)
        k = degree + 1  # Number of parameters (including intercept)
        bic = n_samples * np.log(rss / n_samples) + k * np.log(n_samples)
        aic = n_samples * np.log(rss / n_samples) + 2 * k
        
        bic_scores.append(bic)
        aic_scores.append(aic)
        
        # Compute a simple approximation to log marginal likelihood
        # (this is a rough approximation for visualization purposes)
        # Real computation would require integration over w and sigma^2
        log_ml = -0.5 * bic
        log_marginal_likelihoods.append(log_ml)
    
    # Create the figure with increased height to accommodate annotations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
    
    # Plot the log marginal likelihood
    ax1.plot(degrees, log_marginal_likelihoods, 'bo-', linewidth=2, markersize=8)
    ax1.set_title('Approximate Log Marginal Likelihood vs. Model Complexity', fontsize=14)
    ax1.set_xlabel('Polynomial Degree', fontsize=12)
    ax1.set_ylabel('Log Marginal Likelihood (approximation)', fontsize=12)
    
    # Highlight the best model
    best_idx = np.argmax(log_marginal_likelihoods)
    ax1.scatter([degrees[best_idx]], [log_marginal_likelihoods[best_idx]], 
               color='red', s=100, zorder=10, label=f'Best Model (Degree {degrees[best_idx]})')
    
    ax1.axhline(y=log_marginal_likelihoods[best_idx], color='red', linestyle='--', alpha=0.3)
    ax1.legend()
    
    # Plot information criteria
    ax2.plot(degrees, bic_scores, 'ro-', linewidth=2, markersize=8, label='BIC')
    ax2.plot(degrees, aic_scores, 'go-', linewidth=2, markersize=8, label='AIC')
    ax2.set_title('Information Criteria vs. Model Complexity', fontsize=14)
    ax2.set_xlabel('Polynomial Degree', fontsize=12)
    ax2.set_ylabel('Score (lower is better)', fontsize=12)
    
    # Highlight the best models
    best_bic_idx = np.argmin(bic_scores)
    best_aic_idx = np.argmin(aic_scores)
    
    ax2.scatter([degrees[best_bic_idx]], [bic_scores[best_bic_idx]], 
               color='darkred', s=100, zorder=10, label=f'Best BIC (Degree {degrees[best_bic_idx]})')
    ax2.scatter([degrees[best_aic_idx]], [aic_scores[best_aic_idx]], 
               color='darkgreen', s=100, zorder=10, label=f'Best AIC (Degree {degrees[best_aic_idx]})')
    
    ax2.legend()
    
    plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        saved_files.append(save_path)
    
    return fig

def plot_posterior_predictive(X, y, slope_posterior_mean, slope_posterior_var, alpha_posterior, beta_posterior, save_path=None):
    """Plot the posterior predictive distribution."""
    X = X.flatten()  # Ensure X is 1D
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the training data
    ax.scatter(X, y, color='blue', alpha=0.6, label='Training Data')
    
    # Generate test points for prediction
    X_test = np.linspace(np.min(X) - 1, np.max(X) + 1, 100)
    X_test_with_intercept = np.column_stack([np.ones(len(X_test)), X_test])
    
    # Compute the MLE fit for comparison
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    w_mle = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
    
    # Compute the posterior mean for the intercept
    # For simplicity, we'll use the MLE for intercept
    intercept = w_mle[0]
    
    # Compute the posterior predictive mean
    y_pred_mean = intercept + slope_posterior_mean * X_test
    
    # The posterior predictive variance includes uncertainty in w and noise variance
    # Compute the posterior mean for sigma^2
    sigma2_posterior_mean = beta_posterior / (alpha_posterior - 1) if alpha_posterior > 1 else float('inf')
    
    # Each prediction has uncertainty from:
    # 1. The parameter uncertainty (slope_posterior_var)
    # 2. The noise in the data (sigma2_posterior_mean)
    pred_var = sigma2_posterior_mean + (X_test**2) * slope_posterior_var
    pred_std = np.sqrt(pred_var)
    
    # Plot the posterior mean prediction
    ax.plot(X_test, y_pred_mean, 'r-', linewidth=2, label='Posterior Mean')
    
    # Plot the MLE fit for comparison
    y_mle = w_mle[0] + w_mle[1] * X_test
    ax.plot(X_test, y_mle, 'g--', linewidth=1.5, label='MLE Fit')
    
    # Plot credible intervals
    ax.fill_between(X_test, y_pred_mean - 2*pred_std, y_pred_mean + 2*pred_std, 
                   color='red', alpha=0.2, label='95% Credible Interval')
    
    ax.set_title('Posterior Predictive Distribution', fontsize=14)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        saved_files.append(save_path)
    
    return fig

def calculate_posterior_variance(n, ssr, alpha, beta):
    """Calculate the posterior mean of sigma^2 given data and Inverse-Gamma prior."""
    alpha_posterior = alpha + n/2
    beta_posterior = beta + ssr/2
    
    # Posterior mean exists if alpha_posterior > 1
    if alpha_posterior > 1:
        posterior_mean = beta_posterior / (alpha_posterior - 1)
        return alpha_posterior, beta_posterior, posterior_mean
    else:
        return alpha_posterior, beta_posterior, float('inf')

def plot_detailed_calculation(n, ssr, alpha_prior, beta_prior, save_path=None):
    """Create a detailed step-by-step visualization of the posterior calculation."""
    # Create a figure with a grid
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1.5])
    
    # Calculate posterior parameters
    alpha_posterior = alpha_prior + n/2
    beta_posterior = beta_prior + ssr/2
    
    # Posterior mean exists if alpha_posterior > 1
    posterior_mean = beta_posterior / (alpha_posterior - 1) if alpha_posterior > 1 else float('inf')
    
    # MLE estimate
    mle_sigma2 = ssr / n
    
    # Plot the prior and posterior distributions
    ax1 = fig.add_subplot(gs[0, :])
    x_range = np.linspace(0.01, 15, 1000)
    prior_pdf = stats.invgamma.pdf(x_range, alpha_prior, scale=beta_prior)
    posterior_pdf = stats.invgamma.pdf(x_range, alpha_posterior, scale=beta_posterior)
    
    ax1.plot(x_range, prior_pdf, 'b-', linewidth=2, label=f'Prior: Inv-Gamma({alpha_prior}, {beta_prior})')
    ax1.plot(x_range, posterior_pdf, 'r-', linewidth=2, label=f'Posterior: Inv-Gamma({alpha_posterior:.1f}, {beta_posterior:.1f})')
    
    # Fill under the curves
    ax1.fill_between(x_range, 0, prior_pdf, alpha=0.2, color='blue')
    ax1.fill_between(x_range, 0, posterior_pdf, alpha=0.2, color='red')
    
    # Add vertical lines for the means
    prior_mean = beta_prior / (alpha_prior - 1) if alpha_prior > 1 else np.nan
    if not np.isnan(prior_mean):
        ax1.axvline(prior_mean, color='blue', linestyle='--', label=f'Prior Mean: {prior_mean:.2f}')
    
    ax1.axvline(posterior_mean, color='red', linestyle='--', label=f'Posterior Mean: {posterior_mean:.2f}')
    ax1.axvline(mle_sigma2, color='green', linestyle='--', label=f'MLE: {mle_sigma2:.2f}')
    
    ax1.set_title(f'Posterior Distribution of σ² with n={n}, SSR={ssr}', fontsize=14)
    ax1.set_xlabel('σ²', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True)
    
    # Step by step calculations in equation form
    ax2 = fig.add_subplot(gs[1, :])
    ax2.axis('off')  # No axes for text
    
    calculation_text = [
        r"$\mathbf{Step\ 1:\ Prior\ Distribution}$",
        r"$\sigma^2 \sim \text{Inverse-Gamma}(\alpha_0, \beta_0)$ with $\alpha_0=" + f"{alpha_prior}$, $\\beta_0={beta_prior}$",
        r"$p(\sigma^2) = \frac{\beta_0^{\alpha_0}}{\Gamma(\alpha_0)}(\sigma^2)^{-\alpha_0-1}e^{-\beta_0/\sigma^2}$",
        r"$\mathbf{Step\ 2:\ Data\ Information}$",
        r"$n = " + f"{n}$ observations, Sum of Squared Residuals $SSR = {ssr}$",
        r"$\mathbf{Step\ 3:\ Posterior\ Parameters}$",
        r"$\alpha_n = \alpha_0 + \frac{n}{2} = " + f"{alpha_prior} + \\frac{{{n}}}{{2}} = {alpha_posterior}$",
        r"$\beta_n = \beta_0 + \frac{SSR}{2} = " + f"{beta_prior} + \\frac{{{ssr}}}{{2}} = {beta_posterior}$",
        r"$\mathbf{Step\ 4:\ Posterior\ Mean\ Calculation}$",
        r"$E[\sigma^2|X,y] = \frac{\beta_n}{\alpha_n - 1} = \frac{" + f"{beta_posterior}}}{{{alpha_posterior} - 1}} = \\frac{{{beta_posterior}}}{{{alpha_posterior-1}}} = {posterior_mean:.4f}$",
        r"$\mathbf{Step\ 5:\ Comparison\ with\ MLE}$",
        r"$\hat{\sigma}^2_{MLE} = \frac{SSR}{n} = \frac{" + f"{ssr}}}{{{n}}} = {mle_sigma2:.4f}$",
        r"$\mathbf{Bayesian\ vs.\ Frequentist:}$ Posterior mean = " + f"{posterior_mean:.4f}, MLE = {mle_sigma2:.4f}"
    ]
    
    step_text = "\n".join(calculation_text)
    ax2.text(0.01, 0.99, step_text, fontsize=12, va='top', ha='left', 
            bbox=dict(facecolor='wheat', alpha=0.2, boxstyle='round,pad=0.5'))
    
    # Add a comparison of the likelihood, prior, and posterior
    ax3 = fig.add_subplot(gs[2, :])
    
    # Create data points for visualization
    sigma2_values = np.linspace(0.5, 10, 100)
    
    # Prior probability
    prior_probs = stats.invgamma.pdf(sigma2_values, alpha_prior, scale=beta_prior)
    prior_probs = prior_probs / np.max(prior_probs) * 0.8  # Scale for visualization
    
    # Likelihood (using chi-squared distribution for sum of squared Gaussian errors)
    likelihood = stats.chi2.pdf(ssr / sigma2_values, df=n) * np.power(sigma2_values, -n/2)
    likelihood = likelihood / np.max(likelihood) * 0.8  # Scale for visualization
    
    # Posterior probability
    posterior_probs = stats.invgamma.pdf(sigma2_values, alpha_posterior, scale=beta_posterior)
    posterior_probs = posterior_probs / np.max(posterior_probs) * 0.8  # Scale for visualization
    
    # Plot
    ax3.plot(sigma2_values, prior_probs, 'b-', linewidth=2, label='Prior')
    ax3.plot(sigma2_values, likelihood, 'g-', linewidth=2, label='Likelihood')
    ax3.plot(sigma2_values, posterior_probs, 'r-', linewidth=2, label='Posterior')
    
    ax3.axvline(prior_mean, color='blue', linestyle='--', alpha=0.6)
    ax3.axvline(mle_sigma2, color='green', linestyle='--', alpha=0.6)
    ax3.axvline(posterior_mean, color='red', linestyle='--', alpha=0.6)
    
    ax3.annotate('Prior Mean', xy=(prior_mean, 0.1), xytext=(prior_mean-1.5, 0.3),
                arrowprops=dict(arrowstyle="->"), color='blue')
    ax3.annotate('MLE', xy=(mle_sigma2, 0.1), xytext=(mle_sigma2+0.5, 0.2),
                arrowprops=dict(arrowstyle="->"), color='green')
    ax3.annotate('Posterior Mean', xy=(posterior_mean, 0.1), xytext=(posterior_mean+0.5, 0.4),
                arrowprops=dict(arrowstyle="->"), color='red')
    
    ax3.set_title('Bayesian Inference: Prior → Likelihood → Posterior', fontsize=14)
    ax3.set_xlabel('σ²', fontsize=12)
    ax3.set_ylabel('Scaled Probability Density', fontsize=12)
    ax3.legend()
    ax3.grid(True)
    
    plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.1, right=0.9, top=0.95, bottom=0.05)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        saved_files.append(save_path)
    
    return fig

def plot_bayesian_vs_frequentist(X, y, w_prior_mean, w_prior_var, alpha_prior, beta_prior, save_path=None):
    """Create a visualization comparing Bayesian and Frequentist approaches to linear regression."""
    # Ensure X is 1D
    X = X.flatten()
    
    # Prepare the figure
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1])
    
    # 1. Original data with different sample sizes (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Plot all available data points
    ax1.scatter(X, y, color='blue', alpha=0.6, label='All Data Points')
    
    # Fit a line using all data (frequentist approach)
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    w_mle_all = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
    
    # Generate prediction line
    x_line = np.linspace(min(X) - 1, max(X) + 1, 100)
    y_line_mle_all = w_mle_all[0] + w_mle_all[1] * x_line
    
    # Plot the MLE fit
    ax1.plot(x_line, y_line_mle_all, 'r-', linewidth=2, label='MLE Fit (All Data)')
    
    # Select random subsets of data
    np.random.seed(42)  # For reproducibility
    small_sample_idx = np.random.choice(len(X), size=5, replace=False)
    X_small = X[small_sample_idx]
    y_small = y[small_sample_idx]
    
    # Highlight the small sample
    ax1.scatter(X_small, y_small, color='green', s=100, alpha=0.8, label='Small Sample (n=5)')
    
    # Fit a line using small sample (frequentist approach)
    X_small_with_intercept = np.column_stack([np.ones(len(X_small)), X_small])
    w_mle_small = np.linalg.lstsq(X_small_with_intercept, y_small, rcond=None)[0]
    
    # Generate prediction line for small sample
    y_line_mle_small = w_mle_small[0] + w_mle_small[1] * x_line
    
    # Plot the MLE fit for small sample
    ax1.plot(x_line, y_line_mle_small, 'g--', linewidth=2, label='MLE Fit (Small Sample)')
    
    ax1.set_title('Data and Frequentist (MLE) Fits', fontsize=14)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.legend()
    ax1.grid(True)
    
    # 2. Difference in variance estimation (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Calculate residuals and variance estimates for both samples
    residuals_all = y - (w_mle_all[0] + w_mle_all[1] * X)
    residuals_small = y_small - (w_mle_small[0] + w_mle_small[1] * X_small)
    
    sigma2_mle_all = np.sum(residuals_all**2) / len(residuals_all)
    sigma2_mle_small = np.sum(residuals_small**2) / len(residuals_small)
    
    # Bayesian estimate for small sample
    ssr_small = np.sum(residuals_small**2)
    alpha_posterior = alpha_prior + len(X_small)/2
    beta_posterior = beta_prior + ssr_small/2
    sigma2_bayes_small = beta_posterior / (alpha_posterior - 1) if alpha_posterior > 1 else float('inf')
    
    # Create bar chart
    methods = ['MLE (All Data)', 'MLE (Small Sample)', 'Bayesian (Small Sample)']
    estimates = [sigma2_mle_all, sigma2_mle_small, sigma2_bayes_small]
    colors = ['darkred', 'darkgreen', 'darkblue']
    
    bars = ax2.bar(methods, estimates, color=colors, alpha=0.7)
    
    # Add true sigma2 line if available
    if 'sigma2_true' in globals():
        ax2.axhline(y=sigma2_true, color='black', linestyle='--', label=f'True σ² = {sigma2_true}')
    
    ax2.set_title('Variance Estimates Comparison', fontsize=14)
    ax2.set_ylabel('Estimated σ²', fontsize=12)
    ax2.grid(axis='y')
    
    # Add text annotations on bars
    for bar, estimate in zip(bars, estimates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{estimate:.2f}', ha='center', va='bottom')
    
    if 'sigma2_true' in globals():
        ax2.legend()
    
    # 3. Bayesian vs Frequentist prediction with uncertainty (bottom)
    ax3 = fig.add_subplot(gs[1, :])
    
    # Compute the Bayesian posterior for slope and intercept
    X_small_with_intercept = np.column_stack([np.ones(len(X_small)), X_small])
    
    # For simplicity, we'll compute the Bayesian posterior for slope only
    # (using the MLE for intercept, which is a simplification)
    XTX = X_small_with_intercept.T @ X_small_with_intercept
    XTy = X_small_with_intercept.T @ y_small
    
    # Extract slope information (index 1)
    slope_idx = 1
    prior_precision = 1 / w_prior_var
    slope_posterior_precision = prior_precision + XTX[slope_idx, slope_idx]
    slope_posterior_var = 1 / slope_posterior_precision
    slope_posterior_mean = slope_posterior_var * (prior_precision * w_prior_mean + XTX[slope_idx, :] @ w_mle_small)
    
    # Use intercept from MLE (simplification)
    intercept = w_mle_small[0]
    
    # Compute the Bayesian prediction with uncertainty
    y_pred_bayes_mean = intercept + slope_posterior_mean * x_line
    
    # Calculate prediction uncertainty
    # (combination of parameter uncertainty and noise)
    pred_var = sigma2_bayes_small + (x_line**2) * slope_posterior_var
    pred_std = np.sqrt(pred_var)
    
    # Plot the data and fits
    ax3.scatter(X, y, color='blue', alpha=0.3, label='All Data')
    ax3.scatter(X_small, y_small, color='green', s=80, alpha=0.7, label='Small Sample (n=5)')
    
    # Plot the MLE fits
    ax3.plot(x_line, y_line_mle_all, 'r-', linewidth=1.5, label='MLE Fit (All Data)')
    ax3.plot(x_line, y_line_mle_small, 'g-', linewidth=1.5, label='MLE Fit (Small Sample)')
    
    # Plot the Bayesian fit with uncertainty
    ax3.plot(x_line, y_pred_bayes_mean, 'b-', linewidth=2, label='Bayesian Mean (Small Sample)')
    ax3.fill_between(x_line, y_pred_bayes_mean - 2*pred_std, y_pred_bayes_mean + 2*pred_std,
                    color='blue', alpha=0.15, label='Bayesian 95% Credible Interval')
    
    # Compute frequentist confidence interval (simplified)
    # Note: This is a simplified approach to CI calculation
    t_critical = stats.t.ppf(0.975, len(X_small) - 2)
    ci_width = t_critical * np.sqrt(sigma2_mle_small) * np.sqrt(1 + (x_line - np.mean(X_small))**2 / np.var(X_small) / len(X_small))
    
    # Plot the frequentist confidence interval (dashed)
    ax3.fill_between(x_line, y_line_mle_small - ci_width, y_line_mle_small + ci_width,
                   color='green', alpha=0.1, linestyle='--', label='Frequentist 95% CI')
    
    ax3.set_title('Bayesian vs. Frequentist Approach with Uncertainty', fontsize=14)
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('y', fontsize=12)
    ax3.legend(loc='upper left')
    ax3.grid(True)
    
    # Add explanatory annotations
    ax3.annotate(
        "Bayesian approach:\n- Incorporates prior beliefs\n- Provides full uncertainty estimates\n- More robust with small samples",
        xy=(min(X) + 0.5, max(y) - 1),
        xytext=(min(X) + 0.5, max(y) - 1),
        bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", alpha=0.8)
    )
    
    ax3.annotate(
        "Frequentist approach:\n- Maximum likelihood estimation\n- Point estimates with CIs\n- Requires more data for stability",
        xy=(max(X) - 2, min(y) + 1),
        xytext=(max(X) - 2, min(y) + 1),
        bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen", alpha=0.8)
    )
    
    plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.1, right=0.9, top=0.95, bottom=0.05)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        saved_files.append(save_path)
    
    return fig

# Generate some synthetic data
np.random.seed(42)
n_samples = 20
w_true = np.array([1.0, 2.5])  # True weights: intercept and slope
sigma2_true = 4.0  # True noise variance

# Generate features and targets
X, y, y_true = generate_linear_data(w_true, n_samples, [-3, 3], sigma2_true)

# Prior parameters
w_prior_mean = 0.0  # Prior mean for the slope
w_prior_var = 1.0   # Prior variance for the slope
alpha_prior = 2.0
beta_prior = 4.0

# Save all plots
# 1. Plot the inverse-gamma prior
fig_prior, _ = plot_inverse_gamma(alpha_prior, beta_prior, os.path.join(save_dir, "inverse_gamma_prior.png"))

# 2. Plot the joint prior
fig_joint_prior = plot_joint_prior(os.path.join(save_dir, "joint_prior.png"))

# 3. Plot the joint posterior
fig_posterior, posterior_params = plot_joint_posterior(
    X, y, w_prior_mean, w_prior_var, alpha_prior, beta_prior,
    os.path.join(save_dir, "joint_posterior.png")
)

# 4. Plot the marginal likelihood concept
fig_ml = plot_marginal_likelihood(X, y, os.path.join(save_dir, "marginal_likelihood.png"))

# 5. Plot the posterior predictive distribution
fig_pred = plot_posterior_predictive(
    X, y, *posterior_params,
    os.path.join(save_dir, "posterior_predictive.png")
)

# Task 4: Calculate posterior mean for sigma^2 with specific data
n_specific = 5
ssr_specific = 12
alpha_specific = 2
beta_specific = 4

alpha_posterior, beta_posterior, posterior_mean = calculate_posterior_variance(
    n_specific, ssr_specific, alpha_specific, beta_specific
)

# Create detailed calculation visualization
fig_detailed = plot_detailed_calculation(
    n_specific, ssr_specific, alpha_specific, beta_specific,
    os.path.join(save_dir, "detailed_calculation.png")
)

# Plot the specific case for Task 4 (original simpler plot still included)
fig_task4, _ = plt.subplots(figsize=(10, 6))

# Plot the prior
x_range = np.linspace(0.01, 15, 1000)
prior_pdf = stats.invgamma.pdf(x_range, alpha_specific, scale=beta_specific)
posterior_pdf = stats.invgamma.pdf(x_range, alpha_posterior, scale=beta_posterior)

plt.plot(x_range, prior_pdf, 'b-', linewidth=2, label=f'Prior: Inv-Gamma({alpha_specific}, {beta_specific})')
plt.plot(x_range, posterior_pdf, 'r-', linewidth=2, label=f'Posterior: Inv-Gamma({alpha_posterior:.1f}, {beta_posterior:.1f})')

# Fill under the curves
plt.fill_between(x_range, 0, prior_pdf, alpha=0.2, color='blue')
plt.fill_between(x_range, 0, posterior_pdf, alpha=0.2, color='red')

# Add vertical lines for the means
prior_mean = beta_specific / (alpha_specific - 1) if alpha_specific > 1 else np.nan
if not np.isnan(prior_mean):
    plt.axvline(prior_mean, color='blue', linestyle='--', label=f'Prior Mean: {prior_mean:.2f}')

plt.axvline(posterior_mean, color='red', linestyle='--', label=f'Posterior Mean: {posterior_mean:.2f}')

# MLE estimate would be ssr/n
mle_sigma2 = ssr_specific / n_specific
plt.axvline(mle_sigma2, color='green', linestyle='--', label=f'MLE: {mle_sigma2:.2f}')

plt.title(f'Posterior Distribution of σ² with n={n_specific}, SSR={ssr_specific}', fontsize=14)
plt.xlabel('σ²', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(save_dir, "task4_posterior_variance.png"), dpi=300, bbox_inches='tight')
saved_files.append(os.path.join(save_dir, "task4_posterior_variance.png"))
plt.close()

# Print results of the specific calculation for Task 4
print("\nTask 4: Calculate posterior mean for σ² given specific data")
print(f"Prior: Inverse-Gamma(α={alpha_specific}, β={beta_specific})")
print(f"Data: n={n_specific}, SSR={ssr_specific}")
print(f"Posterior: Inverse-Gamma(α'={alpha_posterior}, β'={beta_posterior})")
print(f"Posterior mean for σ²: {posterior_mean:.4f}")

# Print summary of results
print("\nKey Insights:")
print("1. The conjugate prior for σ² in linear regression is the Inverse-Gamma distribution")
print("2. The joint posterior distribution factorizes into a Normal distribution for w and an Inverse-Gamma for σ²")
print("3. The marginal likelihood integrates out all parameters and is used for model comparison")
print("4. Posterior mean calculation combines prior information with observed data")

print("\nFiles saved:")
for file in saved_files:
    print(f" - {file}")

# 6. Add a Bayesian vs Frequentist comparison visualization
fig_comparison = plot_bayesian_vs_frequentist(
    X, y, w_prior_mean, w_prior_var, alpha_prior, beta_prior,
    os.path.join(save_dir, "bayesian_vs_frequentist.png")
) 