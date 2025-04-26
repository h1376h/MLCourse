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
    print("  • Creating grid of points for visualization...")
    # Create a grid of points
    x, y = np.mgrid[-3:3:.01, -3:3:.01]
    pos = np.dstack((x, y))
    
    # Choose correlation values for demonstration
    correlations = [0, 0.3, 0.6, 0.9]
    print(f"  • Setting up correlation values for comparison: {correlations}")
    
    # Create figure
    print("  • Setting up 2x2 grid of contour plots...")
    plt.figure(figsize=(12, 10))
    
    # Plot step 1: Standard Bivariate Normal (rho = 0)
    print("  • Step 1: Creating standard bivariate normal (ρ = 0) with circular contours...")
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
    print("  • Step 2: Creating bivariate normal with low correlation (ρ = 0.3) showing slight elliptical shape...")
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
    print("  • Step 3: Creating bivariate normal with medium correlation (ρ = 0.6) showing more elongated ellipse...")
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
    print("  • Step 4: Creating bivariate normal with high correlation (ρ = 0.9) showing highly elliptical shape...")
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
    print("  • Saving step-by-step correlation visualization...")
    plt.savefig(os.path.join(save_dir, 'step_by_step_correlation.png'), dpi=300)
    plt.close()
    print("  • Step-by-step correlation visualization complete")

def correlation_comparison(save_dir):
    """Generate contour plots showing the effect of increasing correlation"""
    print("  • Creating grid of points for correlation comparison...")
    # Create a grid of points
    x, y = np.mgrid[-3:3:.01, -3:3:.01]
    pos = np.dstack((x, y))
    
    # Create bivariate normals with different correlations
    correlations = [0, 0.3, 0.6, 0.9]
    pdfs = []
    
    print("  • Computing probability density functions for four correlation values...")
    for corr in correlations:
        cov_matrix = [[1, corr], [corr, 1]]
        rv = multivariate_normal(mean=[0, 0], cov=cov_matrix)
        pdfs.append(rv.pdf(pos))
    
    # Create figure with multiple plots
    print("  • Setting up 2x2 grid for correlation comparison...")
    plt.figure(figsize=(15, 12))
    
    for i, (corr, pdf) in enumerate(zip(correlations, pdfs)):
        print(f"  • Creating contour plot for correlation ρ = {corr}...")
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
    print("  • Saving correlation comparison visualization...")
    plt.savefig(os.path.join(save_dir, 'correlation_comparison.png'), dpi=300)
    plt.close()
    print("  • Correlation comparison visualization complete")

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
    print("  • Creating grid for Mahalanobis distance visualization...")
    # Create a grid of points
    x, y = np.mgrid[-3:3:.01, -3:3:.01]
    pos = np.dstack((x, y))
    
    # Create a correlated bivariate normal
    print("  • Setting up correlated bivariate normal with covariance matrix...")
    cov_matrix = [[1, 0.7], [0.7, 2]]
    inv_cov = np.linalg.inv(cov_matrix)
    mu = np.array([0, 0])
    
    # Calculate Mahalanobis distance for each point
    print("  • Calculating Mahalanobis distances...")
    x_flat = pos[:,:,0].flatten()
    y_flat = pos[:,:,1].flatten()
    xy = np.vstack([x_flat, y_flat]).T
    md = np.array([np.sqrt((p-mu).T @ inv_cov @ (p-mu)) for p in xy])
    md = md.reshape(x.shape)
    
    # Create multivariate normal for PDF
    print("  • Computing probability density function...")
    rv = multivariate_normal(mean=[0, 0], cov=cov_matrix)
    pdf = rv.pdf(pos)
    
    # ----- Step 1: Original probability density -----
    print("  • Step 1: Visualizing original probability density contours...")
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
    print("  • Step 2: Visualizing covariance matrix and its inverse...")
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
    print("  • Step 3: Creating Mahalanobis distance contours...")
    plt.figure(figsize=(10, 8))
    
    # Create circular Mahalanobis distance contours for proper visualization
    circle_grid_x, circle_grid_y = np.mgrid[-3:3:.01, -3:3:.01]
    circle_dist = np.sqrt(circle_grid_x**2 + circle_grid_y**2)
    
    # Plot the circular contours
    contour2 = plt.contour(circle_grid_x, circle_grid_y, circle_dist, levels=[1, 2, 3], colors=['red', 'green', 'blue'])
    plt.clabel(contour2, inline=1, fontsize=10, fmt='%.0f')
    plt.title('Step 3: Mahalanobis Distance Contours\nCircular in standardized space', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.grid(alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Add annotations
    print("  • Adding probability region annotations...")
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
    print("  • Step 4: Comparing probability density and Mahalanobis distance contours...")
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
    
    # Create circular Mahalanobis contours for the right panel
    circle_grid_x, circle_grid_y = np.mgrid[-3:3:.01, -3:3:.01]
    circle_dist = np.sqrt(circle_grid_x**2 + circle_grid_y**2)
    
    # Plot circular Mahalanobis distance contours
    contour2 = ax2.contour(circle_grid_x, circle_grid_y, circle_dist, levels=[1, 2, 3], colors=['red', 'green', 'blue'])
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
    print("  • Step-by-step Mahalanobis distance visualization complete")

def mahalanobis_distance_contours(save_dir):
    """Generate visualization of Mahalanobis distance contours"""
    print("  • Setting up for Mahalanobis distance contours visualization...")
    # Create a grid of points
    x, y = np.mgrid[-3:3:.01, -3:3:.01]
    pos = np.dstack((x, y))
    
    # Create a correlated bivariate normal
    print("  • Creating correlated bivariate normal distribution with correlation ρ = 0.7...")
    cov_matrix = [[1, 0.7], [0.7, 2]]
    inv_cov = np.linalg.inv(cov_matrix)
    
    # Calculate Mahalanobis distance for each point
    print("  • Calculating Mahalanobis distances for the grid points...")
    mu = np.array([0, 0])
    
    # Vectorized Mahalanobis distance calculation
    x_flat = pos[:,:,0].flatten()
    y_flat = pos[:,:,1].flatten()
    xy = np.vstack([x_flat, y_flat]).T
    
    # For each point, calculate (x-μ)^T Σ^-1 (x-μ)
    md = np.array([np.sqrt((p-mu).T @ inv_cov @ (p-mu)) for p in xy])
    md = md.reshape(x.shape)
    
    # Create multivariate normal for PDF
    print("  • Computing probability density function...")
    rv = multivariate_normal(mean=[0, 0], cov=cov_matrix)
    pdf = rv.pdf(pos)
    
    # Create figure with two subplots
    print("  • Creating comparative visualization with density contours and Mahalanobis contours...")
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
    
    # Plot 2: For the right panel, we'll create true circular Mahalanobis contours
    print("  • Calculating eigenvalues and eigenvectors for coordinate transformation...")
    # Calculate eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Create a transform matrix using eigenvectors and eigenvalues
    transform_matrix = eigenvectors @ np.diag(np.sqrt(eigenvalues))

    # Create a new grid in the transformed space for perfectly circular contours
    u, v = np.mgrid[-3:3:.01, -3:3:.01]
    transformed_pos = np.zeros_like(pos)
    
    # Create circular Mahalanobis contours
    print("  • Creating Mahalanobis contours in standardized space...")
    circle_grid_x, circle_grid_y = np.mgrid[-3:3:.01, -3:3:.01]
    circle_pos = np.dstack((circle_grid_x, circle_grid_y))
    
    # Simple Euclidean distance in the transformed space = Mahalanobis distance in original space
    circle_dist = np.sqrt(circle_grid_x**2 + circle_grid_y**2)
    
    # Plot circular Mahalanobis distance contours
    contour2 = ax2.contour(circle_grid_x, circle_grid_y, circle_dist, levels=[1, 2, 3], colors=['red', 'green', 'blue'])
    ax2.clabel(contour2, inline=1, fontsize=10, fmt='%.0f')
    ax2.set_title('Mahalanobis Distance Contours', fontsize=14)
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Add probability annotations at fixed positions
    print("  • Adding probability region annotations...")
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
    print("  • Saving Mahalanobis distance comparison visualization...")
    plt.savefig(os.path.join(save_dir, 'mahalanobis_distance.png'), dpi=300)
    plt.close()
    print("  • Mahalanobis distance comparison visualization complete")

def step_by_step_conditional(save_dir):
    """Generate a step-by-step guide to understanding conditional distributions"""
    print("  • Setting up for conditional distribution visualization...")
    # Create a grid of points
    x, y = np.mgrid[-3:3:.01, -3:3:.01]
    pos = np.dstack((x, y))
    
    # Create a correlated bivariate normal
    rho = 0.8
    print(f"  • Creating bivariate normal with correlation ρ = {rho}...")
    cov_matrix = [[1, rho], [rho, 1]]
    rv = multivariate_normal(mean=[0, 0], cov=cov_matrix)
    pdf = rv.pdf(pos)
    
    # Step 1: Visualize the joint bivariate normal distribution
    print("  • Step 1: Visualizing the joint bivariate normal distribution...")
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
    print("  • Step 2: Illustrating the concept of conditioning on X = 0...")
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
    print("  • Step 3: Showing the resulting conditional distribution...")
    plt.figure(figsize=(10, 8))
    plt.contour(x, y, pdf, levels=10, colors='black', alpha=0.3)
    plt.contourf(x, y, pdf, levels=20, cmap='viridis', alpha=0.3)
    
    # Calculate conditional distribution for X = 0
    x_val = 0
    print(f"  • Calculating conditional distribution for X = {x_val}...")
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
    print("  • Step 4: Demonstrating how conditional distribution changes with different X values...")
    plt.figure(figsize=(10, 8))
    plt.contour(x, y, pdf, levels=10, colors='black', alpha=0.3)
    plt.contourf(x, y, pdf, levels=20, cmap='viridis', alpha=0.3)
    
    # Calculate conditional distributions for X = -1, 0, and 1
    x_values = [-1, 0, 1]
    colors = ['blue', 'green', 'red']
    
    for i, x_val in enumerate(x_values):
        print(f"  • Calculating conditional distribution for X = {x_val}...")
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
    print("  • Step-by-step conditional distribution visualization complete")

def interactive_conditional_demo(save_dir):
    """Generate frames for a simulation of conditional distributions"""
    print("  • Setting up grid for conditional distribution animation frames...")
    # Create a grid of points
    x, y = np.mgrid[-3:3:.01, -3:3:.01]
    pos = np.dstack((x, y))
    
    # Create a correlated bivariate normal
    print("  • Creating bivariate normal with correlation ρ = 0.8...")
    cov_matrix = [[1, 0.8], [0.8, 1]]
    rv = multivariate_normal(mean=[0, 0], cov=cov_matrix)
    pdf = rv.pdf(pos)
    
    # Generate a sequence of x values
    x_values = np.linspace(-2, 2, 9)
    print(f"  • Generating frames for X values from {x_values[0]} to {x_values[-1]}...")
    
    # Create figure for each x value
    for i, x_val in enumerate(x_values):
        print(f"  • Creating frame {i+1}/9 for X = {x_val:.1f}...")
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
    
    print("  • All conditional distribution animation frames generated successfully")

def step_by_step_geometric(save_dir):
    """Generate a step-by-step explanation of the geometric interpretation of contour plots"""
    print("  • Setting up for geometric interpretation visualization...")
    # Create a grid of points
    x, y = np.mgrid[-3:3:.1, -3:3:.1]
    pos = np.dstack((x, y))
    
    # Create a standard bivariate normal for simplicity
    print("  • Creating standard bivariate normal distribution (simpler for visualization)...")
    rv = multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]])
    pdf = rv.pdf(pos)
    
    # Step 1: Show the 3D probability density function
    print("  • Step 1: Creating 3D probability density surface...")
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
    print("  • Step 2: Adding a horizontal slice through the PDF at a constant height...")
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
    print("  • Step 3: Highlighting the intersection curve between the horizontal plane and the surface...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the 3D surface with transparency
    surf = ax.plot_surface(x, y, pdf, cmap='viridis', alpha=0.4,
                         linewidth=0, antialiased=True)
    
    # Add a horizontal plane
    ax.plot_surface(X_plane, Y_plane, Z_plane, color='red', alpha=0.2)
    
    # Plot the intersection curve (for standard normal, it's a circle)
    print("  • Calculating intersection curve (a circle for standard normal)...")
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
    print("  • Step 4: Projecting the intersection curve to the xy-plane to create the contour line...")
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
    print("  • Adding vertical projection lines to show relationship between curve and contour...")
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
    print("  • Step 5: Adding multiple horizontal planes to create multiple contour lines...")
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
        print(f"  • Adding horizontal plane and contour for density level = {level}...")
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
    print("  • Adding legend for different density levels...")
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=c, lw=2) for c in colors]
    ax.legend(custom_lines, [f'Density = {level}' for level in levels], loc='upper right')
    
    # Set optimal viewing angle
    ax.view_init(elev=30, azim=240)
    
    plt.tight_layout()
    print("  • Saving comprehensive geometric interpretation visualization...")
    plt.savefig(os.path.join(save_dir, 'geometric_step5.png'), dpi=300)
    plt.close()
    print("  • Step-by-step geometric interpretation visualization complete")

def geometric_interpretation(save_dir):
    """Generate visualization showing geometric interpretation of contour plots"""
    print("  • Creating comprehensive geometric interpretation visualization...")
    # Create a grid of points
    x, y = np.mgrid[-3:3:.1, -3:3:.1]
    pos = np.dstack((x, y))
    
    # Create a standard bivariate normal
    print("  • Creating standard bivariate normal distribution...")
    rv = multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]])
    pdf = rv.pdf(pos)
    
    # Create 3D figure
    print("  • Setting up 3D visualization with horizontal planes and contours...")
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
        print(f"  • Adding plane and contour for density level = {level}...")
        # Add a horizontal plane at this height
        x_plane = np.linspace(-3, 3, 2)
        y_plane = np.linspace(-3, 3, 2)
        X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
        Z_plane = np.ones_like(X_plane) * level
        
        ax.plot_surface(X_plane, Y_plane, Z_plane, color=color, alpha=0.2)
        
        # Get the x,y points for this level using a simple approximation
        theta = np.linspace(0, 2*np.pi, 100)
        # For standard normal, we know the level curves are circles with radius = sqrt(-2*log(level))
        r = np.sqrt(-2 * np.log(level * np.sqrt(2*np.pi)))  # Radius for this level
        x_level = r * np.cos(theta)
        y_level = r * np.sin(theta)
        
        # Plot the contour at z=0
        ax.plot(x_level, y_level, np.zeros_like(x_level), color=color, linewidth=3)
        
        # Plot vertical projection lines from contour to the surface
        print("  • Adding vertical projection lines for this level...")
        for i in range(0, len(x_level), 10):  # Plot every 10th point to avoid cluttering
            xval, yval = x_level[i], y_level[i]
            z_vals = np.linspace(0, level, 20)
            ax.plot([xval]*20, [yval]*20, z_vals, color=color, linestyle='--', alpha=0.4)
    
    ax.set_title('Geometric Interpretation of Contour Plots', fontsize=14)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Probability Density', fontsize=12)
    
    # Add legend
    print("  • Adding legend for different density levels...")
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=c, lw=2) for c in colors]
    ax.legend(custom_lines, [f'Level {level}' for level in levels], loc='upper right')
    
    # Set the optimal viewing angle
    ax.view_init(elev=30, azim=240)
    
    plt.tight_layout()
    print("  • Saving comprehensive geometric interpretation visualization...")
    plt.savefig(os.path.join(save_dir, 'geometric_interpretation.png'), dpi=300)
    plt.close()
    print("  • Geometric interpretation visualization complete")

def distribution_identification_answer(save_dir):
    """Generate the answer to the distribution identification challenge"""
    print("  • Creating visualization for the distribution identification challenge answer...")
    
    # Create a grid of points
    a, b = np.mgrid[-3:3:.01, -3:3:.01]
    
    # Create the mystery function (from question)
    mystery = np.sin(a**2 + b**2) * np.exp(-(a**2 + b**2)/8)
    # Normalize to ensure it's positive
    mystery = mystery - np.min(mystery)
    mystery = mystery / np.max(mystery)
    
    # Create another function with very similar contours
    # This is a mathematical function that is not a probability distribution
    alternate = np.cos((a**2 + b**2)/2) * np.exp(-(a**2 + b**2)/8)
    alternate = alternate - np.min(alternate)
    alternate = alternate / np.max(alternate)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot the mystery contours
    contour1 = ax1.contour(a, b, mystery, levels=10, cmap='viridis')
    ax1.contourf(a, b, mystery, levels=20, cmap='viridis', alpha=0.3)
    ax1.set_title('Mystery Contour Pattern', fontsize=14)
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)
    ax1.grid(alpha=0.3)
    
    # Plot the alternative function contours
    contour2 = ax2.contour(a, b, alternate, levels=10, cmap='plasma')
    ax2.contourf(a, b, alternate, levels=20, cmap='plasma', alpha=0.3)
    ax2.set_title('Similar Pattern: Not a Distribution', fontsize=14)
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.grid(alpha=0.3)
    
    # Add annotations explaining the key insights
    ax1.annotate(
        "Initially resembles a radial distribution\nwith oscillating probability",
        xy=(0, 0), xytext=(1, 2),
        arrowprops=dict(facecolor='black', shrink=0.05),
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )
    
    ax2.annotate(
        "Function: cos((x²+y²)/2)·e^(-(x²+y²)/8)\nSimilar contours, but not integrable to 1",
        xy=(0, 0), xytext=(1, 2),
        arrowprops=dict(facecolor='black', shrink=0.05),
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )
    
    # Show 3D visualizations of both functions
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 7), subplot_kw={'projection': '3d'})
    
    # Create a coarser grid for 3D visualization
    x_3d, y_3d = np.mgrid[-3:3:.1, -3:3:.1]
    
    # Calculate the functions on this grid
    mystery_3d = np.sin(x_3d**2 + y_3d**2) * np.exp(-(x_3d**2 + y_3d**2)/8)
    mystery_3d = mystery_3d - np.min(mystery_3d)
    mystery_3d = mystery_3d / np.max(mystery_3d)
    
    alternate_3d = np.cos((x_3d**2 + y_3d**2)/2) * np.exp(-(x_3d**2 + y_3d**2)/8)
    alternate_3d = alternate_3d - np.min(alternate_3d)
    alternate_3d = alternate_3d / np.max(alternate_3d)
    
    # Plot 3D surfaces
    surf1 = ax3.plot_surface(x_3d, y_3d, mystery_3d, cmap='viridis', alpha=0.8,
                           linewidth=0, antialiased=True)
    ax3.set_title('3D Surface of Mystery Function', fontsize=14)
    ax3.set_xlabel('X', fontsize=12)
    ax3.set_ylabel('Y', fontsize=12)
    ax3.set_zlabel('Value', fontsize=12)
    
    surf2 = ax4.plot_surface(x_3d, y_3d, alternate_3d, cmap='plasma', alpha=0.8,
                           linewidth=0, antialiased=True)
    ax4.set_title('3D Surface of Alternative Function', fontsize=14)
    ax4.set_xlabel('X', fontsize=12)
    ax4.set_ylabel('Y', fontsize=12)
    ax4.set_zlabel('Value', fontsize=12)
    
    # Adjust viewing angles
    ax3.view_init(elev=30, azim=45)
    ax4.view_init(elev=30, azim=45)
    
    print("\n  • MYSTERY FUNCTION VS ALTERNATIVE FUNCTION\n")
    print("    Mystery Function:")
    print("    f(x,y) = sin(x² + y²) · e^(-(x² + y²)/8)")
    print("\n    Alternative Function:")
    print("    g(x,y) = cos((x² + y²)/2) · e^(-(x² + y²)/8)")
    print("\n    Key Insights:")
    print("    • Both functions have circular, oscillating contours")
    print("    • Both have values that go negative and need normalization")
    print("    • Neither integrates to 1 across the plane")
    print("    • They are not valid probability density functions")
    print("    • The contours alone cannot determine if a function is a valid PDF")
    print("    • This demonstrates how similar contour patterns can arise from")
    print("      different functions, not all of which are probability distributions\n")
    
    plt.tight_layout()
    print("  • Saving distribution identification answer visualizations...")
    fig.savefig(os.path.join(save_dir, 'distribution_identification_answer_contours.png'), dpi=300)
    fig2.savefig(os.path.join(save_dir, 'distribution_identification_answer_3d.png'), dpi=300)
    plt.close('all')
    print("  • Distribution identification answer visualizations complete")

def generate_answer_images():
    """Generate all images for the contour plot visual answers"""
    # Create directories
    save_dir = create_directories()
    
    print("\n" + "="*80)
    print(" CONTOUR PLOT VISUAL EXAMPLES: STEP-BY-STEP SOLUTIONS ")
    print("="*80 + "\n")
    
    # Question 1
    print("\n" + "-"*50)
    print("QUESTION 1: How does changing the correlation coefficient and variance affect the shape of contour plots in a bivariate normal distribution?")
    print("-"*50)
    print("Step 1: Generating visualization showing how correlation transforms contours from circles to ellipses")
    step_by_step_correlation_visual(save_dir)
    print("Step 2: Creating comparison of different correlation coefficients (0, 0.3, 0.6, 0.9)")
    correlation_comparison(save_dir)
    print("✓ Generated correlation visualizations: step_by_step_correlation.png and correlation_comparison.png")
    
    # Question 2
    print("\n" + "-"*50)
    print("QUESTION 2: How do contour plots represent complex non-Gaussian probability distributions, and what insights can be gained from these visualizations?")
    print("-"*50)
    print("✓ This is addressed through visualization of different distributions in the question file.")
    print("  Complex distributions like multimodal, ring-shaped, and other non-Gaussian forms are visualized with contour plots.")
    
    # Question 3
    print("\n" + "-"*50)
    print("QUESTION 3: What is the relationship between contour lines and probability regions in a bivariate normal distribution, and how does this extend our understanding of confidence intervals?")
    print("-"*50)
    print("Step 1: Creating step-by-step Mahalanobis distance visualization")
    step_by_step_mahalanobis(save_dir)
    print("Step 2: Generating comparison between probability density contours and Mahalanobis distance contours")
    mahalanobis_distance_contours(save_dir)
    print("✓ Generated Mahalanobis distance visualizations: mahalanobis_step1-4.png and mahalanobis_distance.png")
    
    # Question 4
    print("\n" + "-"*50)
    print("QUESTION 4: How do conditional distributions change as we vary the value of one variable in a bivariate normal distribution, and what does this tell us about the relationship between variables?")
    print("-"*50)
    print("Step 1: Creating step-by-step explanation of conditional distributions")
    step_by_step_conditional(save_dir)
    print("Step 2: Generating animation frames showing how conditional distributions shift as we vary X")
    interactive_conditional_demo(save_dir)
    print("✓ Generated conditional distribution visualizations: conditional_step1-4.png and conditional_demo_frame_1-9.png")
    
    # Question 5 
    print("\n" + "-"*50)
    print("QUESTION 5: What is the relationship between joint and marginal distributions, and what information is preserved or lost when examining only marginal distributions?")
    print("-"*50)
    print("✓ This is addressed through visualization of marginal distributions in the question file.")
    print("  Marginal distributions lose information about correlation structure between variables when projected from joint distribution.")
    
    # Question 6
    print("\n" + "-"*50)
    print("QUESTION 6: What is the geometric relationship between 3D probability density surfaces and their 2D contour plot representations, and what are the advantages of each visualization?")
    print("-"*50)
    print("Step 1: Creating step-by-step explanation of how contour plots are formed from 3D surfaces")
    step_by_step_geometric(save_dir)
    print("Step 2: Generating comprehensive geometric interpretation visualization")
    geometric_interpretation(save_dir)
    print("✓ Generated geometric interpretation visualizations: geometric_step1-5.png and geometric_interpretation.png")
    
    # Question 7
    print("\n" + "-"*50)
    print("QUESTION 7: Identify which probability distribution the mystery contour most closely resembles, and explain your reasoning based on the shape and characteristics of the contour.")
    print("-"*50)
    print("Step 1: Creating answer visualizations for the distribution identification challenge")
    distribution_identification_answer(save_dir)
    print("✓ Generated distribution identification answer visualizations: distribution_identification_answer_contours.png, distribution_identification_answer_3d.png")
    
    print("\n" + "="*80)
    print(f"All visualizations have been generated and saved to:")
    print(f"{save_dir}")
    print("="*80 + "\n")
    
    print("SUMMARY OF GENERATED IMAGES:")
    print("1. Correlation effects: step_by_step_correlation.png, correlation_comparison.png")
    print("2. Mahalanobis distance: mahalanobis_step1.png through mahalanobis_step4.png, mahalanobis_distance.png")
    print("3. Conditional distributions: conditional_step1.png through conditional_step4.png")
    print("4. Conditional distribution animation: conditional_demo_frame_1-9.png")
    print("5. Geometric interpretation: geometric_step1-5.png, geometric_interpretation.png")
    print("6. Distribution identification: distribution_identification_answer_contours.png, distribution_identification_answer_3d.png")
    
    return save_dir

if __name__ == "__main__":
    generate_answer_images() 