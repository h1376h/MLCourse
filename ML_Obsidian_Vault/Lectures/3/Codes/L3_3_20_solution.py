import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import scipy.stats as stats

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_3_Quiz_20")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Helper function to create confidence ellipses
def confidence_ellipse(ax, mean, cov, n_std=1.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    mean : array-like, shape (2, )
        Mean vector of the distribution.
    cov : array-like, shape (2, 2)
        Covariance matrix of the distribution.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    facecolor : str
        Facecolor of the ellipse. Default is 'none'.
    **kwargs
        Additional arguments passed to ax.add_patch
    """
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Scale by standard deviations
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean[0], mean[1])

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# Task 1: Identify the conjugate prior distribution
def identify_conjugate_prior():
    """Identify the conjugate prior distribution for linear regression parameters."""
    print("Step 1: Identify the conjugate prior distribution for the parameter vector w = [w0, w1]^T")
    print("In Bayesian linear regression with Gaussian noise, the conjugate prior for the parameter vector w")
    print("is a multivariate Gaussian distribution:")
    print("w ~ N(μ_0, Σ_0)")
    print("where:")
    print("  - μ_0 is the prior mean vector")
    print("  - Σ_0 is the prior covariance matrix")
    print()
    print("With a Gaussian likelihood and a Gaussian prior, the posterior distribution will also be Gaussian")
    print("due to conjugacy. This makes the computations analytically tractable.")
    print()
    
    return "Multivariate Gaussian distribution"

# Task 2: Derive the posterior distribution
def derive_posterior_distribution(sigma_squared=1.0):
    """
    Derive the posterior distribution after observing the given data points.
    
    Parameters:
    -----------
    sigma_squared : float
        The known variance of the noise
    
    Returns:
    --------
    posterior_mean : numpy.ndarray
        The posterior mean vector
    posterior_cov : numpy.ndarray
        The posterior covariance matrix
    """
    print("Step 2: Derive the posterior distribution after observing the data points")
    print("Given:")
    print("  - Prior: w ~ N(μ_0, Σ_0)")
    print("  - μ_0 = [0, 0]^T")
    print("  - Σ_0 = [[2, 0], [0, 3]]")
    print("  - Data points: (x^(1), y^(1)) = (1, 3) and (x^(2), y^(2)) = (2, 5)")
    print("  - Noise variance: σ² =", sigma_squared)
    print()
    
    # Prior distribution parameters
    prior_mean = np.array([0, 0])
    prior_cov = np.array([[2, 0], [0, 3]])
    
    # Data points
    X = np.array([[1, 1], [1, 2]])  # Design matrix with bias term (first column of 1s)
    y = np.array([3, 5])
    
    print("The design matrix X (including the bias term):")
    print(X)
    print()
    
    # Compute posterior parameters
    # For Bayesian linear regression, the posterior is proportional to the product of the prior and the likelihood
    # The posterior mean and covariance are given by:
    # Σ_n = (Σ_0^(-1) + (1/σ²) * X^T * X)^(-1)
    # μ_n = Σ_n * (Σ_0^(-1) * μ_0 + (1/σ²) * X^T * y)
    
    # Compute the posterior covariance matrix
    prior_cov_inv = np.linalg.inv(prior_cov)
    X_transpose_X = X.T @ X
    posterior_precision = prior_cov_inv + (1/sigma_squared) * X_transpose_X
    posterior_cov = np.linalg.inv(posterior_precision)
    
    # Compute the posterior mean vector
    X_transpose_y = X.T @ y
    posterior_mean = posterior_cov @ (prior_cov_inv @ prior_mean + (1/sigma_squared) * X_transpose_y)
    
    print("Posterior derivation:")
    print("1. The posterior precision matrix (inverse of covariance):")
    print("   Σ_n^(-1) = Σ_0^(-1) + (1/σ²) * X^T * X")
    print(f"   Σ_n^(-1) = {prior_cov_inv} + (1/{sigma_squared}) * {X_transpose_X}")
    print(f"   Σ_n^(-1) = {posterior_precision}")
    print()
    
    print("2. The posterior covariance matrix:")
    print("   Σ_n = (Σ_n^(-1))^(-1)")
    print(f"   Σ_n = {posterior_cov}")
    print()
    
    print("3. The posterior mean vector:")
    print("   μ_n = Σ_n * (Σ_0^(-1) * μ_0 + (1/σ²) * X^T * y)")
    print(f"   μ_n = {posterior_cov} * ({prior_cov_inv @ prior_mean} + (1/{sigma_squared}) * {X_transpose_y})")
    print(f"   μ_n = {posterior_mean}")
    print()
    
    print("Therefore, the posterior distribution is:")
    print(f"w | D ~ N({posterior_mean}, {posterior_cov})")
    print()
    
    return posterior_mean, posterior_cov, prior_mean, prior_cov

# Task 3: Calculate the posterior predictive distribution
def calculate_posterior_predictive(posterior_mean, posterior_cov, x_new, sigma_squared=1.0):
    """
    Calculate the posterior predictive distribution for a new input.
    
    Parameters:
    -----------
    posterior_mean : numpy.ndarray
        The posterior mean vector
    posterior_cov : numpy.ndarray
        The posterior covariance matrix
    x_new : float
        The new input value
    sigma_squared : float
        The known variance of the noise
    
    Returns:
    --------
    pred_mean : float
        The mean of the posterior predictive distribution
    pred_var : float
        The variance of the posterior predictive distribution
    """
    print("Step 3: Calculate the posterior predictive distribution for a new input")
    print(f"New input: x_new = {x_new}")
    print()
    
    # Design vector for the new input (including the bias term)
    x_new_vec = np.array([1, x_new])
    
    # Mean of the posterior predictive distribution
    pred_mean = x_new_vec @ posterior_mean
    
    # Variance of the posterior predictive distribution
    # It has two components: 
    # 1. Uncertainty from the posterior parameter distribution
    # 2. Inherent noise variance
    pred_var_from_params = x_new_vec @ posterior_cov @ x_new_vec
    pred_var = pred_var_from_params + sigma_squared
    
    print("The posterior predictive distribution for a new input is:")
    print("p(y_new | x_new, D) = N(μ_pred, σ²_pred)")
    print()
    
    print("1. Mean of the posterior predictive distribution:")
    print("   μ_pred = x_new^T * μ_n")
    print(f"   μ_pred = {x_new_vec} ⋅ {posterior_mean}")
    print(f"   μ_pred = {pred_mean}")
    print()
    
    print("2. Variance of the posterior predictive distribution:")
    print("   σ²_pred = x_new^T * Σ_n * x_new + σ²")
    print(f"   σ²_pred = {x_new_vec} ⋅ {posterior_cov} ⋅ {x_new_vec} + {sigma_squared}")
    print(f"   σ²_pred = {pred_var_from_params} + {sigma_squared}")
    print(f"   σ²_pred = {pred_var}")
    print()
    
    print("Therefore, the posterior predictive distribution is:")
    print(f"y_new | x_new, D ~ N({pred_mean}, {pred_var})")
    print()
    
    return pred_mean, pred_var, pred_var_from_params

# Task 4: Explain the effect of posterior uncertainty on prediction
def explain_uncertainty_effect(pred_var, pred_var_from_params, sigma_squared):
    """
    Explain how posterior uncertainty affects prediction uncertainty.
    
    Parameters:
    -----------
    pred_var : float
        The total variance of the posterior predictive distribution
    pred_var_from_params : float
        The variance component from parameter uncertainty
    sigma_squared : float
        The known variance of the noise
    """
    print("Step 4: Explain how posterior uncertainty affects prediction uncertainty")
    print()
    
    print("In Bayesian linear regression, the total prediction uncertainty has two components:")
    print(f"1. Uncertainty from parameter estimates (epistemic uncertainty): {pred_var_from_params:.4f}")
    print(f"2. Inherent noise variance (aleatoric uncertainty): {sigma_squared:.4f}")
    print(f"Total prediction variance: {pred_var:.4f}")
    print()
    
    uncertainty_percentage = (pred_var_from_params / pred_var) * 100
    print(f"Parameter uncertainty accounts for {uncertainty_percentage:.2f}% of the total prediction variance.")
    print()
    
    print("Comparison with Maximum Likelihood Estimation (MLE):")
    print("- MLE treats parameters as fixed point estimates")
    print("- MLE prediction variance only includes the inherent noise variance (σ²)")
    print("- MLE underestimates the total prediction uncertainty by ignoring parameter uncertainty")
    print(f"- In this case, MLE would underestimate the prediction variance by {uncertainty_percentage:.2f}%")
    print()
    
    print("Key differences between Bayesian and MLE approaches:")
    print("1. Bayesian approach provides more honest uncertainty estimates by accounting for parameter uncertainty")
    print("2. Bayesian prediction intervals are typically wider than MLE confidence intervals")
    print("3. The difference is more pronounced when:")
    print("   - Sample size is small")
    print("   - We predict far from the observed data")
    print("   - The prior uncertainty is large")
    print()
    
    return uncertainty_percentage

# Create visualizations to illustrate the concepts
def create_visualizations(posterior_mean, posterior_cov, prior_mean, prior_cov, 
                         pred_mean, pred_var, pred_var_from_params, sigma_squared, save_dir=None):
    """
    Create visualizations to illustrate Bayesian linear regression concepts.
    
    Parameters:
    -----------
    posterior_mean : numpy.ndarray
        The posterior mean vector
    posterior_cov : numpy.ndarray
        The posterior covariance matrix
    prior_mean : numpy.ndarray
        The prior mean vector
    prior_cov : numpy.ndarray
        The prior covariance matrix
    pred_mean : float
        The mean of the posterior predictive distribution
    pred_var : float
        The variance of the posterior predictive distribution
    pred_var_from_params : float
        The variance component from parameter uncertainty
    sigma_squared : float
        The known variance of the noise
    save_dir : str
        Directory to save the figures
    
    Returns:
    --------
    saved_files : list
        List of saved file paths
    """
    saved_files = []
    
    # Data points
    X_data = np.array([1, 2])
    y_data = np.array([3, 5])
    
    # Create Figure 1: Prior and Posterior Parameter Distributions
    plt.figure(figsize=(10, 8))
    
    # Set up a 2D grid for visualization
    w0_range = np.linspace(-2, 6, 100)
    w1_range = np.linspace(-2, 6, 100)
    W0, W1 = np.meshgrid(w0_range, w1_range)
    
    # Create the subplot
    ax = plt.subplot(111)
    
    # Plot the data points
    plt.scatter(X_data, y_data, color='black', s=100, zorder=5, label='Observed data')
    
    # Plot confidence ellipses for prior distribution
    confidence_ellipse(ax, prior_mean, prior_cov, n_std=1, edgecolor='blue', linestyle='--', 
                      label='Prior (68% confidence)')
    confidence_ellipse(ax, prior_mean, prior_cov, n_std=2, edgecolor='blue', linestyle=':')
    
    # Plot confidence ellipses for posterior distribution
    confidence_ellipse(ax, posterior_mean, posterior_cov, n_std=1, edgecolor='red', 
                      label='Posterior (68% confidence)')
    confidence_ellipse(ax, posterior_mean, posterior_cov, n_std=2, edgecolor='red', linestyle=':')
    
    # Plot the prior mean and posterior mean
    plt.scatter(prior_mean[0], prior_mean[1], color='blue', s=100, marker='o', label='Prior mean')
    plt.scatter(posterior_mean[0], posterior_mean[1], color='red', s=100, marker='x', label='Posterior mean')
    
    # Generate and plot regression lines for samples from the posterior
    np.random.seed(42)
    n_samples = 15
    w_samples = np.random.multivariate_normal(posterior_mean, posterior_cov, n_samples)
    
    x_plot = np.linspace(0, 3, 100)
    for i, w in enumerate(w_samples):
        plt.plot(x_plot, w[0] + w[1] * x_plot, 'r-', alpha=0.1)
    
    # Generate and plot regression lines for samples from the prior
    np.random.seed(43)
    w_prior_samples = np.random.multivariate_normal(prior_mean, prior_cov, n_samples)
    
    for i, w in enumerate(w_prior_samples):
        plt.plot(x_plot, w[0] + w[1] * x_plot, 'b-', alpha=0.1)
    
    # Plot the mean regression line from the posterior
    plt.plot(x_plot, posterior_mean[0] + posterior_mean[1] * x_plot, 'r-', linewidth=2, 
             label='Posterior mean')
    
    # Add annotations
    plt.text(4.5, 5.0, f"Prior: w ~ N({prior_mean}, {np.diag(prior_cov)})", 
             fontsize=10, color='blue', bbox=dict(facecolor='white', alpha=0.5))
    plt.text(4.5, 4.0, f"Posterior: w ~ N([{posterior_mean[0]:.2f}, {posterior_mean[1]:.2f}], ...)", 
             fontsize=10, color='red', bbox=dict(facecolor='white', alpha=0.5))
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Bayesian Linear Regression: Prior and Posterior Distributions', fontsize=14)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True)
    plt.xlim(0, 3)
    plt.ylim(0, 8)
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot1_parameter_distributions.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Create Figure 2: Posterior Predictive Distribution
    plt.figure(figsize=(10, 6))
    
    # Plot the data points
    plt.scatter(X_data, y_data, color='black', s=100, zorder=5, label='Observed data')
    
    # Plot the mean regression line from the posterior
    plt.plot(x_plot, posterior_mean[0] + posterior_mean[1] * x_plot, 'r-', linewidth=2, 
             label='Posterior mean')
    
    # Calculate predictive distribution for each point
    pred_means = posterior_mean[0] + posterior_mean[1] * x_plot
    
    # Plot samples from the posterior predictive
    np.random.seed(44)
    n_pred_samples = 50
    
    for i in range(n_pred_samples):
        # Sample a parameter vector from the posterior
        w_sample = np.random.multivariate_normal(posterior_mean, posterior_cov)
        
        # Calculate the predictive mean for this sample
        y_sample = w_sample[0] + w_sample[1] * x_plot
        
        # Add noise to represent the aleatoric uncertainty
        y_sample_with_noise = y_sample + np.random.normal(0, np.sqrt(sigma_squared), len(x_plot))
        
        # Plot the sample with low alpha
        plt.plot(x_plot, y_sample, 'r-', alpha=0.05)
        plt.plot(x_plot, y_sample_with_noise, 'g-', alpha=0.025)
    
    # Calculate and plot predictive confidence intervals
    pred_std_from_params = np.sqrt(np.array([np.array([1, x]) @ posterior_cov @ np.array([1, x]) for x in x_plot]))
    pred_std_total = np.sqrt(pred_std_from_params**2 + sigma_squared)
    
    # Plot 68% and 95% confidence intervals for parameter uncertainty only
    plt.fill_between(x_plot, 
                     posterior_mean[0] + posterior_mean[1] * x_plot - pred_std_from_params,
                     posterior_mean[0] + posterior_mean[1] * x_plot + pred_std_from_params,
                     color='red', alpha=0.2, label='Parameter uncertainty (68%)')
    
    plt.fill_between(x_plot, 
                     posterior_mean[0] + posterior_mean[1] * x_plot - 2*pred_std_from_params,
                     posterior_mean[0] + posterior_mean[1] * x_plot + 2*pred_std_from_params,
                     color='red', alpha=0.1)
    
    # Plot 68% and 95% confidence intervals including noise
    plt.fill_between(x_plot, 
                     posterior_mean[0] + posterior_mean[1] * x_plot - pred_std_total,
                     posterior_mean[0] + posterior_mean[1] * x_plot + pred_std_total,
                     color='green', alpha=0.2, label='Total uncertainty (68%)')
    
    plt.fill_between(x_plot, 
                     posterior_mean[0] + posterior_mean[1] * x_plot - 2*pred_std_total,
                     posterior_mean[0] + posterior_mean[1] * x_plot + 2*pred_std_total,
                     color='green', alpha=0.1)
    
    # Mark the new input point
    x_new = 1.5
    y_pred = posterior_mean[0] + posterior_mean[1] * x_new
    pred_std_from_params_new = np.sqrt(np.array([1, x_new]) @ posterior_cov @ np.array([1, x_new]))
    pred_std_total_new = np.sqrt(pred_std_from_params_new**2 + sigma_squared)
    
    plt.scatter(x_new, y_pred, color='blue', s=100, zorder=6, label=f'Prediction at x={x_new}')
    
    # Add error bars for the new prediction
    plt.errorbar(x_new, y_pred, yerr=pred_std_from_params_new, 
                 fmt='o', color='blue', ecolor='red', capsize=5, linewidth=2, label='Parameter uncertainty')
    plt.errorbar(x_new, y_pred, yerr=pred_std_total_new, 
                 fmt='o', color='blue', ecolor='green', capsize=5, linewidth=2, label='Total uncertainty')
    
    # Add text annotations
    plt.text(2.2, 0.5, f"Prediction at x={x_new}:\nMean = {y_pred:.2f}\nParameter uncertainty = {pred_std_from_params_new:.2f}\nTotal uncertainty = {pred_std_total_new:.2f}", 
             fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Bayesian Linear Regression: Posterior Predictive Distribution', fontsize=14)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True)
    plt.xlim(0, 3)
    plt.ylim(0, 8)
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot2_predictive_distribution.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Create Figure 3: Comparison of Bayesian and MLE Prediction Intervals
    plt.figure(figsize=(10, 6))
    
    # Calculate MLE parameters
    X_design = np.vstack([np.ones_like(X_data), X_data]).T
    XTX_inv = np.linalg.inv(X_design.T @ X_design)
    beta_mle = XTX_inv @ X_design.T @ y_data
    
    # Plot the data points
    plt.scatter(X_data, y_data, color='black', s=100, zorder=5, label='Observed data')
    
    # Plot the Bayesian mean regression line
    plt.plot(x_plot, posterior_mean[0] + posterior_mean[1] * x_plot, 'r-', linewidth=2, 
             label='Bayesian posterior mean')
    
    # Plot the MLE regression line
    plt.plot(x_plot, beta_mle[0] + beta_mle[1] * x_plot, 'b-', linewidth=2, 
             label='MLE regression line')
    
    # Calculate standard errors for MLE prediction
    pred_vars_mle = np.array([sigma_squared * (1 + np.array([1, x]) @ XTX_inv @ np.array([1, x])) for x in x_plot])
    pred_std_mle = np.sqrt(pred_vars_mle)
    
    # Calculate standard errors for Bayesian prediction
    pred_std_total = np.sqrt(pred_std_from_params**2 + sigma_squared)
    
    # Plot 95% confidence intervals for MLE
    plt.fill_between(x_plot, 
                     beta_mle[0] + beta_mle[1] * x_plot - 1.96*np.sqrt(pred_vars_mle),
                     beta_mle[0] + beta_mle[1] * x_plot + 1.96*np.sqrt(pred_vars_mle),
                     color='blue', alpha=0.2, label='MLE 95% confidence interval')
    
    # Plot 95% confidence intervals for Bayesian
    plt.fill_between(x_plot, 
                     posterior_mean[0] + posterior_mean[1] * x_plot - 1.96*pred_std_total,
                     posterior_mean[0] + posterior_mean[1] * x_plot + 1.96*pred_std_total,
                     color='red', alpha=0.2, label='Bayesian 95% credible interval')
    
    # Mark regions where we have data vs. where we don't
    plt.axvspan(0, 1, alpha=0.1, color='gray', label='No observed data')
    plt.axvspan(2, 3, alpha=0.1, color='gray')
    plt.axvspan(1, 2, alpha=0.1, color='yellow', label='Observed data range')
    
    # Add text annotations
    plt.text(1.5, 7.0, "Bayesian intervals account for\nparameter uncertainty +\ninherent noise variance", 
             fontsize=10, color='red', bbox=dict(facecolor='white', alpha=0.7))
    plt.text(1.5, 6.0, "MLE intervals only account for\ninherent noise variance\nwith fixed parameters", 
             fontsize=10, color='blue', bbox=dict(facecolor='white', alpha=0.7))
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Comparison of Bayesian and MLE Prediction Intervals', fontsize=14)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True)
    plt.xlim(0, 3)
    plt.ylim(0, 8)
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot3_bayesian_vs_mle.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Create Figure 4: Decomposition of Prediction Uncertainty
    uncertainty_percentage = pred_var_from_params / pred_var * 100
    noise_percentage = sigma_squared / pred_var * 100
    
    plt.figure(figsize=(10, 6))
    
    labels = ['Parameter Uncertainty', 'Inherent Noise']
    sizes = [uncertainty_percentage, noise_percentage]
    explode = (0.1, 0)
    colors = ['firebrick', 'royalblue']
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.axis('equal')
    plt.title('Decomposition of Prediction Uncertainty at x_new = 1.5', fontsize=14)
    
    # Add text annotations
    plt.text(-2.0, -1.4, f"Total prediction variance: {pred_var:.4f}\nParameter uncertainty: {pred_var_from_params:.4f}\nInherent noise variance: {sigma_squared:.4f}", 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot4_uncertainty_decomposition.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    return saved_files

# Main execution script
if __name__ == "__main__":
    # Set the known noise variance
    sigma_squared = 1.0
    
    # Task 1: Identify the conjugate prior
    conjugate_prior = identify_conjugate_prior()
    
    # Task 2: Derive the posterior distribution
    posterior_mean, posterior_cov, prior_mean, prior_cov = derive_posterior_distribution(sigma_squared)
    
    # Task 3: Calculate the posterior predictive distribution
    x_new = 1.5
    pred_mean, pred_var, pred_var_from_params = calculate_posterior_predictive(
        posterior_mean, posterior_cov, x_new, sigma_squared)
    
    # Task 4: Explain how posterior uncertainty affects prediction uncertainty
    uncertainty_percentage = explain_uncertainty_effect(pred_var, pred_var_from_params, sigma_squared)
    
    # Create visualizations
    saved_files = create_visualizations(
        posterior_mean, posterior_cov, prior_mean, prior_cov, 
        pred_mean, pred_var, pred_var_from_params, sigma_squared, save_dir)
    
    print(f"\nVisualizations saved to: {', '.join(saved_files)}")
    print("\nQuestion 20 Solution Summary:")
    print(f"1. The conjugate prior distribution for linear regression parameters is: {conjugate_prior}")
    print(f"2. The posterior distribution after observing the data is: N({posterior_mean}, {posterior_cov})")
    print(f"3. The posterior predictive distribution for x_new = {x_new} is: N({pred_mean:.4f}, {pred_var:.4f})")
    print(f"4. Parameter uncertainty accounts for {uncertainty_percentage:.2f}% of the total prediction variance") 