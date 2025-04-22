import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import multivariate_normal
from matplotlib import cm
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def create_directories():
    """Create necessary directories for saving plots"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    answer_dir = os.path.join(images_dir, "Contour_Plot_Visual_Answer")
    
    os.makedirs(answer_dir, exist_ok=True)
    
    return answer_dir

def step_by_step_correlation_visual(save_dir):
    """Generate a step-by-step visualization of how correlation affects bivariate normal distribution"""
    # Create a grid of points
    x, y = np.mgrid[-3:3:.01, -3:3:.01]
    pos = np.dstack((x, y))
    
    # Choose correlation values for demonstration
    correlations = [0, 0.3, 0.6, 0.9]
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot step 1: Standard Bivariate Normal (rho = 0)
    cov_matrix = [[1, 0], [0, 1]]
    rv = multivariate_normal(mean=[0, 0], cov=cov_matrix)
    pdf = rv.pdf(pos)
    
    plt.subplot(2, 2, 1)
    contour = plt.contour(x, y, pdf, levels=10, cmap='viridis')
    plt.contourf(x, y, pdf, levels=20, cmap='viridis', alpha=0.3)
    plt.title(f'Step 1: No Correlation (ρ = 0)\nCircular Contours', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.grid(alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Draw the 95% confidence ellipse
    confidence_ellipse(0, 0, 1, 1, 0, plt.gca(), n_std=2, facecolor='pink', alpha=0.3)
    
    # Plot step 2: Low Correlation (rho = 0.3)
    cov_matrix = [[1, 0.3], [0.3, 1]]
    rv = multivariate_normal(mean=[0, 0], cov=cov_matrix)
    pdf = rv.pdf(pos)
    
    plt.subplot(2, 2, 2)
    contour = plt.contour(x, y, pdf, levels=10, cmap='viridis')
    plt.contourf(x, y, pdf, levels=20, cmap='viridis', alpha=0.3)
    plt.title(f'Step 2: Low Correlation (ρ = 0.3)\nSlightly Elliptical', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.grid(alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Draw the 95% confidence ellipse
    confidence_ellipse(0, 0, 1, 1, 0.3, plt.gca(), n_std=2, facecolor='pink', alpha=0.3)
    
    # Plot step 3: Medium Correlation (rho = 0.6)
    cov_matrix = [[1, 0.6], [0.6, 1]]
    rv = multivariate_normal(mean=[0, 0], cov=cov_matrix)
    pdf = rv.pdf(pos)
    
    plt.subplot(2, 2, 3)
    contour = plt.contour(x, y, pdf, levels=10, cmap='viridis')
    plt.contourf(x, y, pdf, levels=20, cmap='viridis', alpha=0.3)
    plt.title(f'Step 3: Medium Correlation (ρ = 0.6)\nMore Elongated', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.grid(alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Draw the 95% confidence ellipse
    confidence_ellipse(0, 0, 1, 1, 0.6, plt.gca(), n_std=2, facecolor='pink', alpha=0.3)
    
    # Plot step 4: High Correlation (rho = 0.9)
    cov_matrix = [[1, 0.9], [0.9, 1]]
    rv = multivariate_normal(mean=[0, 0], cov=cov_matrix)
    pdf = rv.pdf(pos)
    
    plt.subplot(2, 2, 4)
    contour = plt.contour(x, y, pdf, levels=10, cmap='viridis')
    plt.contourf(x, y, pdf, levels=20, cmap='viridis', alpha=0.3)
    plt.title(f'Step 4: High Correlation (ρ = 0.9)\nHighly Elliptical', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.grid(alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Draw the 95% confidence ellipse
    confidence_ellipse(0, 0, 1, 1, 0.9, plt.gca(), n_std=2, facecolor='pink', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'step_by_step_correlation.png'), dpi=300)
    plt.close()

def correlation_comparison(save_dir):
    """Generate contour plots showing the effect of increasing correlation"""
    # Create a grid of points
    x, y = np.mgrid[-3:3:.01, -3:3:.01]
    pos = np.dstack((x, y))
    
    # Create bivariate normals with different correlations
    correlations = [0, 0.3, 0.6, 0.9]
    pdfs = []
    
    for corr in correlations:
        cov_matrix = [[1, corr], [corr, 1]]
        rv = multivariate_normal(mean=[0, 0], cov=cov_matrix)
        pdfs.append(rv.pdf(pos))
    
    # Create figure with multiple plots
    plt.figure(figsize=(15, 12))
    
    for i, (corr, pdf) in enumerate(zip(correlations, pdfs)):
        plt.subplot(2, 2, i+1)
        plt.contour(x, y, pdf, levels=10, cmap='viridis')
        plt.title(f'Correlation ρ = {corr}', fontsize=14)
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12)
        plt.grid(alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        
        # Draw a confidence ellipse
        confidence_ellipse(0, 0, 1, 1, corr, plt.gca(), n_std=2, facecolor='pink', alpha=0.3, label='95% region')
        
        if i == 0:
            plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'correlation_comparison.png'), dpi=300)
    plt.close()

def confidence_ellipse(mu_x, mu_y, sigma_x, sigma_y, corr, ax, n_std=3.0, **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`
    
    Parameters
    ----------
    mu_x, mu_y : float
        Mean of x and y
    sigma_x, sigma_y : float
        Standard deviation of x and y
    corr : float
        Correlation coefficient
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`
    
    Returns
    -------
    matplotlib.patches.Ellipse
    """
    cov = [[sigma_x**2, corr * sigma_x * sigma_y], 
           [corr * sigma_x * sigma_y, sigma_y**2]]
    
    pearson = cov[0][1]/np.sqrt(cov[0][0] * cov[1][1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                     **kwargs)
    
    # Calculating the standard deviation of x from the matrix and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0][0]) * n_std
    mean_x = mu_x
    
    # Calculating the standard deviation of y from the matrix and multiplying
    # with the given number of standard deviations.
    scale_y = np.sqrt(cov[1][1]) * n_std
    mean_y = mu_y
    
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def step_by_step_mahalanobis(save_dir):
    """Generate a step-by-step visualization of Mahalanobis distance vs probability contours"""
    # Create a grid of points
    x, y = np.mgrid[-3:3:.01, -3:3:.01]
    pos = np.dstack((x, y))
    
    # Create a correlated bivariate normal
    cov_matrix = [[1, 0.7], [0.7, 2]]
    inv_cov = np.linalg.inv(cov_matrix)
    mu = np.array([0, 0])
    
    # Calculate Mahalanobis distance for each point
    x_flat = pos[:,:,0].flatten()
    y_flat = pos[:,:,1].flatten()
    xy = np.vstack([x_flat, y_flat]).T
    md = np.array([np.sqrt((p-mu).T @ inv_cov @ (p-mu)) for p in xy])
    md = md.reshape(x.shape)
    
    # Create multivariate normal for PDF
    rv = multivariate_normal(mean=[0, 0], cov=cov_matrix)
    pdf = rv.pdf(pos)
    
    # ----- Step 1: Original probability density -----
    plt.figure(figsize=(10, 8))
    contour1 = plt.contour(x, y, pdf, levels=10, cmap='viridis')
    plt.contourf(x, y, pdf, levels=20, cmap='viridis', alpha=0.3)
    plt.clabel(contour1, inline=1, fontsize=10)
    plt.title('Step 1: Probability Density Contours\nElliptical due to correlation', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.grid(alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mahalanobis_step1.png'), dpi=300)
    plt.close()
    
    # ----- Step 2: Show the covariance matrix and its inverse -----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display the covariance matrix
    ax1.matshow(cov_matrix, cmap='Blues')
    ax1.set_title('Step 2a: Covariance Matrix\nEncodes correlation and variances', fontsize=12)
    for (i, j), val in np.ndenumerate(cov_matrix):
        ax1.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=14)
    
    # Display the inverse covariance matrix
    ax2.matshow(inv_cov, cmap='Reds')
    ax2.set_title('Step 2b: Inverse Covariance Matrix\nUsed in Mahalanobis distance', fontsize=12)
    for (i, j), val in np.ndenumerate(inv_cov):
        ax2.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mahalanobis_step2.png'), dpi=300)
    plt.close()
    
    # ----- Step 3: Show Mahalanobis distance contours -----
    plt.figure(figsize=(10, 8))
    contour2 = plt.contour(x, y, md, levels=[1, 2, 3], colors=['red', 'green', 'blue'])
    plt.clabel(contour2, inline=1, fontsize=10, fmt='%.0f')
    plt.title('Step 3: Mahalanobis Distance Contours\nCircular in standardized space', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.grid(alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Add annotations
    prob_annotations = {
        1: ('39.4%', (1.0, 0.5)),
        2: ('86.5%', (2.0, 1.0)),
        3: ('98.9%', (3.0, 1.5))
    }
    
    for d, (prob, pos) in prob_annotations.items():
        plt.annotate(f'Distance {d} → P(inside) = {prob}', xy=pos, xytext=(pos[0]+0.5, pos[1]+0.5),
                  arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mahalanobis_step3.png'), dpi=300)
    plt.close()
    
    # ----- Step 4: Compare probability and Mahalanobis contours -----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot probability density contours
    contour1 = ax1.contour(x, y, pdf, levels=10, cmap='viridis')
    ax1.contourf(x, y, pdf, levels=20, cmap='viridis', alpha=0.3)
    ax1.clabel(contour1, inline=1, fontsize=10)
    ax1.set_title('Step 4a: Probability Density\nElliptical contours', fontsize=14)
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot Mahalanobis distance contours
    contour2 = ax2.contour(x, y, md, levels=[1, 2, 3], colors=['red', 'green', 'blue'])
    ax2.clabel(contour2, inline=1, fontsize=10, fmt='%.0f')
    ax2.set_title('Step 4b: Mahalanobis Distance\nCircular contours', fontsize=14)
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Add probability annotations
    for d, (prob, pos) in prob_annotations.items():
        ax2.annotate(f'P(inside) = {prob}', xy=pos, xytext=(pos[0]+0.5, pos[1]+0.5),
                  arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mahalanobis_step4.png'), dpi=300)
    plt.close()

def mahalanobis_distance_contours(save_dir):
    """Generate visualization of Mahalanobis distance contours"""
    # Create a grid of points
    x, y = np.mgrid[-3:3:.01, -3:3:.01]
    pos = np.dstack((x, y))
    
    # Create a correlated bivariate normal
    cov_matrix = [[1, 0.7], [0.7, 2]]
    inv_cov = np.linalg.inv(cov_matrix)
    
    # Calculate Mahalanobis distance for each point
    mu = np.array([0, 0])
    
    # Vectorized Mahalanobis distance calculation
    x_flat = pos[:,:,0].flatten()
    y_flat = pos[:,:,1].flatten()
    xy = np.vstack([x_flat, y_flat]).T
    
    # For each point, calculate (x-μ)^T Σ^-1 (x-μ)
    md = np.array([np.sqrt((p-mu).T @ inv_cov @ (p-mu)) for p in xy])
    md = md.reshape(x.shape)
    
    # Create multivariate normal for PDF
    rv = multivariate_normal(mean=[0, 0], cov=cov_matrix)
    pdf = rv.pdf(pos)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot 1: Contours of probability density
    contour1 = ax1.contour(x, y, pdf, levels=10, cmap='viridis')
    ax1.contourf(x, y, pdf, levels=20, cmap='viridis', alpha=0.3)
    ax1.clabel(contour1, inline=1, fontsize=10)
    ax1.set_title('Probability Density Contours', fontsize=14)
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot 2: Contours of Mahalanobis distance
    contour2 = ax2.contour(x, y, md, levels=[1, 2, 3], colors=['red', 'green', 'blue'])
    ax2.clabel(contour2, inline=1, fontsize=10, fmt='%.0f')
    ax2.set_title('Mahalanobis Distance Contours', fontsize=14)
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Add probability annotations at fixed positions
    prob_annotations = {
        1: ('39.4%', (1.0, 0.5)),
        2: ('86.5%', (2.0, 1.0)),
        3: ('98.9%', (3.0, 1.5))
    }
    
    for d, (prob, pos) in prob_annotations.items():
        ax2.annotate(f'P(inside) = {prob}', xy=pos, xytext=(pos[0]+0.5, pos[1]+0.5),
                  arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mahalanobis_distance.png'), dpi=300)
    plt.close()

def step_by_step_conditional(save_dir):
    """Generate a step-by-step guide to understanding conditional distributions"""
    # Create a grid of points
    x, y = np.mgrid[-3:3:.01, -3:3:.01]
    pos = np.dstack((x, y))
    
    # Create a correlated bivariate normal
    rho = 0.8
    cov_matrix = [[1, rho], [rho, 1]]
    rv = multivariate_normal(mean=[0, 0], cov=cov_matrix)
    pdf = rv.pdf(pos)
    
    # Step 1: Visualize the joint bivariate normal distribution
    plt.figure(figsize=(10, 8))
    plt.contour(x, y, pdf, levels=10, colors='black', alpha=0.5)
    plt.contourf(x, y, pdf, levels=20, cmap='viridis', alpha=0.5)
    plt.title(f'Step 1: Joint Distribution\nBivariate Normal with ρ = {rho}', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.grid(alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Add annotation about the joint PDF
    plt.annotate(
        "Joint PDF f(x,y) describes\nthe probability density\nat each point (x,y)",
        xy=(1, 1), xytext=(1.5, 1.5),
        arrowprops=dict(facecolor='black', shrink=0.05),
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'conditional_step1.png'), dpi=300)
    plt.close()
    
    # Step 2: Illustrate the concept of conditioning on X = 0
    plt.figure(figsize=(10, 8))
    plt.contour(x, y, pdf, levels=10, colors='black', alpha=0.3)
    plt.contourf(x, y, pdf, levels=20, cmap='viridis', alpha=0.3)
    
    # Draw a vertical line at X = 0
    x_val = 0
    plt.axvline(x=x_val, color='red', linestyle='--', linewidth=2)
    
    plt.title(f'Step 2: Conditioning on X = {x_val}\nVertical slice through joint distribution', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.grid(alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
    
    # Add annotation about conditioning
    plt.annotate(
        "Conditioning on X = 0\nmeans restricting to this\nvertical slice",
        xy=(0, 1.5), xytext=(1, 2),
        arrowprops=dict(facecolor='red', shrink=0.05),
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'conditional_step2.png'), dpi=300)
    plt.close()
    
    # Step 3: Show the resulting conditional distribution
    plt.figure(figsize=(10, 8))
    plt.contour(x, y, pdf, levels=10, colors='black', alpha=0.3)
    plt.contourf(x, y, pdf, levels=20, cmap='viridis', alpha=0.3)
    
    # Calculate conditional distribution for X = 0
    x_val = 0
    # Conditional mean: mu_Y|X = mu_Y + rho*(sigma_Y/sigma_X)*(x - mu_X)
    # For standardized variables: mu_Y|X = rho*x
    cond_mean = rho * x_val  
    
    # Conditional variance: sigma²_Y|X = sigma²_Y * (1 - rho²)
    cond_var = 1 * (1 - rho**2)
    cond_std = np.sqrt(cond_var)
    
    # Generate y values for this conditional distribution
    y_vals = np.linspace(-3, 3, 1000)
    cond_pdf = (1 / (cond_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((y_vals - cond_mean) / cond_std)**2)
    
    # Normalize for plotting alongside the contours
    cond_pdf_normalized = cond_pdf / np.max(cond_pdf) * 1.2
    
    # Plot the conditional distribution as a line at fixed X = 0
    plt.plot(x_val + cond_pdf_normalized, y_vals, color='red', linewidth=3, 
            label=f'P(Y|X={x_val})')
    
    # Mark the mean of the conditional distribution
    plt.plot(x_val, cond_mean, 'ro', markersize=8)
    
    # Vertical line at X = 0
    plt.axvline(x=x_val, color='red', linestyle='--', alpha=0.7)
    
    plt.title(f'Step 3: Conditional Distribution P(Y|X={x_val})\nA normal distribution with reduced variance', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.grid(alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Add annotation about the conditional mean and variance
    plt.annotate(
        f"Conditional Mean:\nE[Y|X={x_val}] = {cond_mean:.2f}\n\nConditional Variance:\nVar(Y|X) = {cond_var:.2f}",
        xy=(x_val, cond_mean), xytext=(1, 0),
        arrowprops=dict(facecolor='red', shrink=0.05),
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )
    
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'conditional_step3.png'), dpi=300)
    plt.close()
    
    # Step 4: Demonstrate how conditional distribution changes with X value
    plt.figure(figsize=(10, 8))
    plt.contour(x, y, pdf, levels=10, colors='black', alpha=0.3)
    plt.contourf(x, y, pdf, levels=20, cmap='viridis', alpha=0.3)
    
    # Calculate conditional distributions for X = -1, 0, and 1
    x_values = [-1, 0, 1]
    colors = ['blue', 'green', 'red']
    
    for i, x_val in enumerate(x_values):
        # Calculate conditional mean and variance
        cond_mean = rho * x_val
        cond_var = 1 * (1 - rho**2)
        cond_std = np.sqrt(cond_var)
        
        # Generate y values for this conditional distribution
        y_vals = np.linspace(-3, 3, 1000)
        cond_pdf = (1 / (cond_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((y_vals - cond_mean) / cond_std)**2)
        
        # Normalize for plotting
        cond_pdf_normalized = cond_pdf / np.max(cond_pdf) * 0.8
        
        # Plot the conditional distribution
        plt.plot(x_val + cond_pdf_normalized, y_vals, color=colors[i], linewidth=2, 
                label=f'P(Y|X={x_val})')
        
        # Mark the conditional mean
        plt.plot(x_val, cond_mean, 'o', color=colors[i], markersize=6)
        
        # Vertical line at X = x_val
        plt.axvline(x=x_val, color=colors[i], linestyle='--', alpha=0.5)
    
    plt.title(f'Step 4: Changing the Conditioning Value\nMean shifts, variance remains constant', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.grid(alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Add annotation about the regression line
    plt.plot([-1.5, 1.5], [-1.5*rho, 1.5*rho], 'k--', alpha=0.7)
    plt.annotate(
        f"Regression Line: E[Y|X=x] = {rho}x\nConditional means lie on this line",
        xy=(1, rho), xytext=(1.5, 1.5),
        arrowprops=dict(facecolor='black', shrink=0.05),
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )
    
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'conditional_step4.png'), dpi=300)
    plt.close()

def interactive_conditional_demo(save_dir):
    """Generate frames for a simulation of conditional distributions"""
    # Create a grid of points
    x, y = np.mgrid[-3:3:.01, -3:3:.01]
    pos = np.dstack((x, y))
    
    # Create a correlated bivariate normal
    cov_matrix = [[1, 0.8], [0.8, 1]]
    rv = multivariate_normal(mean=[0, 0], cov=cov_matrix)
    pdf = rv.pdf(pos)
    
    # Generate a sequence of x values
    x_values = np.linspace(-2, 2, 9)
    
    # Create figure for each x value
    for i, x_val in enumerate(x_values):
        plt.figure(figsize=(10, 8))
        
        # Plot contours of the joint distribution
        plt.contour(x, y, pdf, levels=10, colors='black', alpha=0.3)
        plt.contourf(x, y, pdf, levels=20, cmap='viridis', alpha=0.3)
        
        # Calculate conditional mean and std
        cond_mean = 0.8 * x_val  # mu_Y|X = rho * x_val for standardized vars
        cond_var = 1 * (1 - 0.8**2)  # sigma²_Y|X = 1 * (1 - 0.8²)
        cond_std = np.sqrt(cond_var)
        
        # Generate y values for this conditional distribution
        y_vals = np.linspace(-3, 3, 1000)
        cond_pdf = (1 / (cond_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((y_vals - cond_mean) / cond_std)**2)
        
        # Normalize for plotting alongside the contours
        cond_pdf_normalized = cond_pdf / np.max(cond_pdf) * 1.2
        
        # Plot the conditional distribution as a line at fixed x
        plt.plot(x_val + cond_pdf_normalized, y_vals, color='red', linewidth=3, 
                label=f'P(Y|X={x_val:.1f})')
        
        # Mark the mean of this conditional distribution
        plt.plot(x_val, cond_mean, 'ro', markersize=8)
        
        # Draw a vertical line at this x value
        plt.axvline(x=x_val, color='red', linestyle='--', alpha=0.7)
        
        # Add title and labels
        plt.title(f'Conditional Distribution P(Y|X={x_val:.1f})\nBivariate Normal with ρ = 0.8', fontsize=14)
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12)
        plt.grid(alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        
        # Add annotations
        plt.annotate(f'Conditional Mean\nE[Y|X={x_val:.1f}] = {cond_mean:.2f}',
                   xy=(x_val, cond_mean),
                   xytext=(x_val+0.5, cond_mean+0.5),
                   arrowprops=dict(facecolor='black', shrink=0.05),
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.legend(loc='upper right')
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'conditional_demo_frame_{i+1}.png'), dpi=300)
        plt.close()

def step_by_step_geometric(save_dir):
    """Generate a step-by-step explanation of the geometric interpretation of contour plots"""
    # Create a grid of points
    x, y = np.mgrid[-3:3:.1, -3:3:.1]
    pos = np.dstack((x, y))
    
    # Create a standard bivariate normal for simplicity
    rv = multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]])
    pdf = rv.pdf(pos)
    
    # Step 1: Show the 3D probability density function
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the 3D surface
    surf = ax.plot_surface(x, y, pdf, cmap='viridis', alpha=0.8,
                         linewidth=0, antialiased=True)
    
    ax.set_title('Step 1: 3D Probability Density Surface\nHeight = probability density at each point (x,y)', fontsize=14)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Probability Density', fontsize=12)
    
    # Add annotation
    ax.text(-2, 2, 0.15, "Height represents\nprobability density", color='white', fontsize=12)
    
    # Set optimal viewing angle
    ax.view_init(elev=30, azim=135)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'geometric_step1.png'), dpi=300)
    plt.close()
    
    # Step 2: Show a horizontal slice through the PDF
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the 3D surface
    surf = ax.plot_surface(x, y, pdf, cmap='viridis', alpha=0.7,
                         linewidth=0, antialiased=True)
    
    # Add a horizontal plane at a specific height value
    level = 0.1
    x_plane = np.linspace(-3, 3, 2)
    y_plane = np.linspace(-3, 3, 2)
    X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
    Z_plane = np.ones_like(X_plane) * level
    
    ax.plot_surface(X_plane, Y_plane, Z_plane, color='red', alpha=0.3)
    
    ax.set_title('Step 2: Horizontal Slice at Density = 0.1\nIntersection forms a circle for standard normal', fontsize=14)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Probability Density', fontsize=12)
    
    # Add annotation
    ax.text(-2, 2, 0.15, "Horizontal plane\nintersects the surface", color='white', fontsize=12)
    
    # Set optimal viewing angle
    ax.view_init(elev=30, azim=135)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'geometric_step2.png'), dpi=300)
    plt.close()
    
    # Step 3: Show the intersection curve
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the 3D surface with transparency
    surf = ax.plot_surface(x, y, pdf, cmap='viridis', alpha=0.4,
                         linewidth=0, antialiased=True)
    
    # Add a horizontal plane
    ax.plot_surface(X_plane, Y_plane, Z_plane, color='red', alpha=0.2)
    
    # Plot the intersection curve (for standard normal, it's a circle)
    theta = np.linspace(0, 2*np.pi, 100)
    # For standard normal: level = (1/(2π)) * exp(-r²/2), solving for r:
    r = np.sqrt(-2 * np.log(level * 2*np.pi))
    x_level = r * np.cos(theta)
    y_level = r * np.sin(theta)
    z_level = np.ones_like(x_level) * level
    
    ax.plot(x_level, y_level, z_level, 'r-', linewidth=4, label='Intersection curve')
    
    ax.set_title('Step 3: Intersection Curve\nAll points with equal probability density', fontsize=14)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Probability Density', fontsize=12)
    
    # Add annotation
    ax.text(-2, 2, 0.15, "Intersection forms\na 3D curve of\nconstant density", color='white', fontsize=12)
    
    # Set optimal viewing angle
    ax.view_init(elev=30, azim=135)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'geometric_step3.png'), dpi=300)
    plt.close()
    
    # Step 4: Show the projection of the curve to the xy-plane (the contour)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the 3D surface with high transparency
    surf = ax.plot_surface(x, y, pdf, cmap='viridis', alpha=0.3,
                         linewidth=0, antialiased=True)
    
    # Plot the intersection curve
    ax.plot(x_level, y_level, z_level, 'r-', linewidth=3)
    
    # Plot the projection to z=0 (the contour)
    ax.plot(x_level, y_level, np.zeros_like(x_level), 'r-', linewidth=4)
    
    # Add vertical lines connecting the curve to its projection
    for i in range(0, len(x_level), 10):  # Plot every 10th point to avoid cluttering
        xval, yval = x_level[i], y_level[i]
        ax.plot([xval, xval], [yval, yval], [0, level], 'r--', alpha=0.3)
    
    ax.set_title('Step 4: Projecting to 2D\nThis creates the contour line', fontsize=14)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Probability Density', fontsize=12)
    
    # Add annotation
    ax.text(-2, 2, 0.05, "Projection of curve\nto xy-plane is the\ncontour line", color='black', fontsize=12)
    
    # Set optimal viewing angle
    ax.view_init(elev=30, azim=135)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'geometric_step4.png'), dpi=300)
    plt.close()
    
    # Step 5: Show multiple contours (the complete contour plot)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the 3D surface with high transparency
    surf = ax.plot_surface(x, y, pdf, cmap='viridis', alpha=0.3,
                         linewidth=0, antialiased=True)
    
    # Define multiple levels
    levels = [0.05, 0.1, 0.15]
    colors = ['red', 'green', 'blue']
    
    # Add horizontal planes and projected contours for each level
    for level, color in zip(levels, colors):
        # Add a horizontal plane at this height
        Z_plane = np.ones_like(X_plane) * level
        ax.plot_surface(X_plane, Y_plane, Z_plane, color=color, alpha=0.2)
        
        # Get the x,y points for this level
        r = np.sqrt(-2 * np.log(level * np.sqrt(2*np.pi)))
        x_level = r * np.cos(theta)
        y_level = r * np.sin(theta)
        z_level = np.ones_like(x_level) * level
        
        # Plot the intersection curve at z=level
        ax.plot(x_level, y_level, z_level, color=color, linewidth=2)
        
        # Plot the contour at z=0
        ax.plot(x_level, y_level, np.zeros_like(x_level), color=color, linewidth=3)
    
    ax.set_title('Step 5: Multiple Contour Lines\nA complete contour plot', fontsize=14)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Probability Density', fontsize=12)
    
    # Add legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=c, lw=2) for c in colors]
    ax.legend(custom_lines, [f'Density = {level}' for level in levels], loc='upper right')
    
    # Set optimal viewing angle
    ax.view_init(elev=30, azim=240)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'geometric_step5.png'), dpi=300)
    plt.close()

def geometric_interpretation(save_dir):
    """Generate visualization showing geometric interpretation of contour plots"""
    # Create a grid of points
    x, y = np.mgrid[-3:3:.1, -3:3:.1]
    pos = np.dstack((x, y))
    
    # Create a standard bivariate normal
    rv = multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]])
    pdf = rv.pdf(pos)
    
    # Create 3D figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the 3D surface
    surf = ax.plot_surface(x, y, pdf, cmap='viridis', alpha=0.7,
                         linewidth=0, antialiased=True)
    
    # Select a few specific height values for contours
    levels = [0.05, 0.1, 0.15]
    colors = ['red', 'green', 'blue']
    
    # Add horizontal planes and projected contours
    for level, color in zip(levels, colors):
        # Add a horizontal plane at this height
        x_plane = np.linspace(-3, 3, 2)
        y_plane = np.linspace(-3, 3, 2)
        X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
        Z_plane = np.ones_like(X_plane) * level
        
        ax.plot_surface(X_plane, Y_plane, Z_plane, color=color, alpha=0.2)
        
        # Get the x,y points for this level using a simple approximation
        # Create a dense circle of points and filter those close to the density level
        theta = np.linspace(0, 2*np.pi, 100)
        # For standard normal, we know the level curves are circles with radius = sqrt(-2*log(level))
        r = np.sqrt(-2 * np.log(level * np.sqrt(2*np.pi)))  # Radius for this level
        x_level = r * np.cos(theta)
        y_level = r * np.sin(theta)
        
        # Plot the contour at z=0
        ax.plot(x_level, y_level, np.zeros_like(x_level), color=color, linewidth=3)
        
        # Plot vertical projection lines from contour to the surface
        for i in range(0, len(x_level), 10):  # Plot every 10th point to avoid cluttering
            xval, yval = x_level[i], y_level[i]
            z_vals = np.linspace(0, level, 20)
            ax.plot([xval]*20, [yval]*20, z_vals, color=color, linestyle='--', alpha=0.4)
    
    ax.set_title('Geometric Interpretation of Contour Plots', fontsize=14)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Probability Density', fontsize=12)
    
    # Add legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=c, lw=2) for c in colors]
    ax.legend(custom_lines, [f'Level {level}' for level in levels], loc='upper right')
    
    # Set the optimal viewing angle
    ax.view_init(elev=30, azim=240)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'geometric_interpretation.png'), dpi=300)
    plt.close()

def application_examples(save_dir):
    """Generate visualizations showing practical applications of contour plots"""
    # Create a figure with multiple plots
    plt.figure(figsize=(15, 12))
    
    # Example 1: Bayesian Posterior for two parameters
    plt.subplot(221)
    
    # Generate a grid
    theta1, theta2 = np.mgrid[0:10:100j, 0:10:100j]
    pos = np.dstack((theta1, theta2))
    
    # Create a multivariate normal for the posterior
    posterior = multivariate_normal(mean=[5, 5], cov=[[1, 0.7], [0.7, 1]])
    posterior_pdf = posterior.pdf(pos)
    
    # Plot contours
    plt.contour(theta1, theta2, posterior_pdf, levels=10, colors='black', alpha=0.5)
    plt.contourf(theta1, theta2, posterior_pdf, levels=20, cmap='Blues', alpha=0.7)
    
    # Mark the MAP (maximum a posteriori) estimate
    plt.plot(5, 5, 'ro', markersize=10, label='MAP Estimate')
    
    # Add 95% credible region
    confidence_ellipse(5, 5, 1, 1, 0.7, plt.gca(), n_std=2, edgecolor='red', linewidth=2, 
                     label='95% Credible Region', fill=False)
    
    plt.title('Bayesian Posterior Distribution', fontsize=14)
    plt.xlabel('Parameter θ₁', fontsize=12)
    plt.ylabel('Parameter θ₂', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Example 2: Clustering visualization
    plt.subplot(222)
    
    # Generate three clusters
    np.random.seed(42)
    n_points = 300
    
    # Generate the cluster centers and data
    centers = [(2, 2), (5, 7), (8, 3)]
    colors = ['red', 'green', 'blue']
    
    # Create the dataset
    X = np.zeros((n_points, 2))
    true_labels = np.zeros(n_points, dtype=int)
    
    for i in range(n_points):
        cluster = i % 3
        X[i] = np.random.multivariate_normal(centers[cluster], [[1, 0.5], [0.5, 1]])
        true_labels[i] = cluster
    
    # Plot points
    for i, color in enumerate(colors):
        cluster_points = X[true_labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=color, s=30, alpha=0.6,
                  label=f'Cluster {i+1}')
    
    # Plot contours for each cluster
    for i, (center, color) in enumerate(zip(centers, colors)):
        # Estimate covariance matrix from the points
        cluster_points = X[true_labels == i]
        cov = np.cov(cluster_points.T)
        
        # Create a grid for this cluster
        grid_x, grid_y = np.mgrid[0:10:100j, 0:10:100j]
        grid_pos = np.dstack((grid_x, grid_y))
        
        # Create the multivariate normal
        cluster_rv = multivariate_normal(mean=center, cov=cov)
        cluster_pdf = cluster_rv.pdf(grid_pos)
        
        # Plot contours
        plt.contour(grid_x, grid_y, cluster_pdf, levels=3, colors=color, alpha=0.8)
    
    plt.title('Clustering Analysis', fontsize=14)
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Example 3: Optimization landscape
    plt.subplot(223)
    
    # Create a grid
    x, y = np.mgrid[-4:4:100j, -4:4:100j]
    
    # Create a complex optimization landscape (e.g., a loss function)
    # Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²
    z = (1 - x)**2 + 100 * (y - x**2)**2
    
    # Take log to better visualize
    z_log = np.log(z + 1)
    
    # Contour plot of the optimization landscape
    plt.contour(x, y, z_log, levels=20, colors='black', alpha=0.5)
    plt.contourf(x, y, z_log, levels=50, cmap='viridis')
    
    # Mark the global minimum
    plt.plot(1, 1, 'r*', markersize=15, label='Global Minimum')
    
    # Add an optimization path
    # Simulated gradient descent path
    np.random.seed(42)
    path_x = [-3]
    path_y = [3]
    
    for _ in range(20):
        # Add some noise to make the path interesting
        step_x = 0.3 * (1 - path_x[-1]) + 0.05 * np.random.randn()
        step_y = 0.3 * (path_x[-1]**2 - path_y[-1]) + 0.05 * np.random.randn()
        
        path_x.append(path_x[-1] + step_x)
        path_y.append(path_y[-1] + step_y)
    
    plt.plot(path_x, path_y, 'r-o', linewidth=2, markersize=5, label='Optimization Path')
    
    plt.title('Optimization Landscape', fontsize=14)
    plt.xlabel('Parameter x', fontsize=12)
    plt.ylabel('Parameter y', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Example 4: Error ellipses in measurements
    plt.subplot(224)
    
    # Create some measurement data with errors
    np.random.seed(42)
    n_measurements = 100
    
    # True value
    true_x, true_y = 5, 5
    
    # Generate measurements with correlated errors
    cov_matrix = [[0.5, 0.3], [0.3, 0.8]]
    measurements = np.random.multivariate_normal([true_x, true_y], cov_matrix, n_measurements)
    
    # Plot the measurements
    plt.scatter(measurements[:, 0], measurements[:, 1], s=30, alpha=0.6, label='Measurements')
    
    # Plot the true value
    plt.plot(true_x, true_y, 'r*', markersize=15, label='True Value')
    
    # Calculate mean and covariance of measurements
    mean_x = np.mean(measurements[:, 0])
    mean_y = np.mean(measurements[:, 1])
    
    # Plot error ellipses
    for n_std in [1, 2, 3]:
        confidence_ellipse(mean_x, mean_y, np.sqrt(cov_matrix[0][0]), np.sqrt(cov_matrix[1][1]), 
                         cov_matrix[0][1]/np.sqrt(cov_matrix[0][0]*cov_matrix[1][1]),
                         plt.gca(), n_std=n_std, edgecolor='red', alpha=0.3, facecolor='pink' if n_std==1 else None)
    
    plt.title('Error Ellipses in Measurements', fontsize=14)
    plt.xlabel('Measurement X', fontsize=12)
    plt.ylabel('Measurement Y', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'applications.png'), dpi=300)
    plt.close()

def generate_answer_images():
    """Generate all images for the contour plot visual answers"""
    # Create directories
    save_dir = create_directories()
    
    # Generate standard visualizations
    correlation_comparison(save_dir)
    mahalanobis_distance_contours(save_dir)
    interactive_conditional_demo(save_dir)
    geometric_interpretation(save_dir)
    application_examples(save_dir)
    
    # Generate step-by-step solutions
    step_by_step_correlation_visual(save_dir)
    step_by_step_mahalanobis(save_dir)
    step_by_step_conditional(save_dir)
    step_by_step_geometric(save_dir)
    
    print(f"Generated all answer images in {save_dir}")
    return save_dir

if __name__ == "__main__":
    generate_answer_images() 