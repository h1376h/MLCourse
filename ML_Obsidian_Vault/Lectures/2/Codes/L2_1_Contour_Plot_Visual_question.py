import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import multivariate_normal
from matplotlib import cm

def create_directories():
    """Create necessary directories for saving plots"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    question_dir = os.path.join(images_dir, "Contour_Plot_Visual_Question")
    
    os.makedirs(question_dir, exist_ok=True)
    
    return question_dir

def bivariate_normal_contour(save_dir):
    """Generate contour plots for bivariate normal distributions with different covariance matrices
    
    QUESTION: How does changing the correlation coefficient and variance affect the shape of 
    contour plots in a bivariate normal distribution?
    """
    # Create a grid of points
    x, y = np.mgrid[-3:3:.01, -3:3:.01]
    pos = np.dstack((x, y))
    
    # 1. Standard bivariate normal
    rv1 = multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]])
    
    # 2. Correlated bivariate normal (positive correlation)
    rv2 = multivariate_normal(mean=[0, 0], cov=[[1, 0.7], [0.7, 1]])
    
    # 3. Correlated bivariate normal (negative correlation)
    rv3 = multivariate_normal(mean=[0, 0], cov=[[1, -0.7], [-0.7, 1]])
    
    # 4. Bivariate normal with different variances
    rv4 = multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 2]])
    
    # Calculate PDFs
    pdf1 = rv1.pdf(pos)
    pdf2 = rv2.pdf(pos)
    pdf3 = rv3.pdf(pos)
    pdf4 = rv4.pdf(pos)
    
    # Create figure with multiple plots
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Standard bivariate normal
    plt.subplot(221)
    plt.contour(x, y, pdf1, levels=10, cmap='viridis')
    plt.title('Standard Bivariate Normal\n$\\rho = 0$', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.grid(alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot 2: Positive correlation
    plt.subplot(222)
    plt.contour(x, y, pdf2, levels=10, cmap='viridis')
    plt.title('Positive Correlation\n$\\rho = 0.7$', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.grid(alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot 3: Negative correlation
    plt.subplot(223)
    plt.contour(x, y, pdf3, levels=10, cmap='viridis')
    plt.title('Negative Correlation\n$\\rho = -0.7$', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.grid(alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot 4: Different variances
    plt.subplot(224)
    plt.contour(x, y, pdf4, levels=10, cmap='viridis')
    plt.title('Different Variances\n$\\sigma_X^2 = 1, \\sigma_Y^2 = 2$', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.grid(alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'bivariate_normal_contours.png'), dpi=300)
    plt.close()
    
    print("QUESTION 1: How does changing the correlation coefficient and variance affect the shape of contour plots in a bivariate normal distribution?")

def different_distributions_contour(save_dir):
    """Generate contour plots for various bivariate distributions
    
    QUESTION: How do contour plots represent complex non-Gaussian probability distributions,
    and what insights can be gained from these visualizations?
    """
    # Create a grid of points
    x, y = np.mgrid[-3:3:.01, -3:3:.01]
    pos = np.dstack((x, y))
    
    # 1. Create a mixture of two bivariate normals
    rv1 = multivariate_normal(mean=[-1, -1], cov=[[0.5, 0], [0, 0.5]])
    rv2 = multivariate_normal(mean=[1, 1], cov=[[0.5, 0], [0, 0.5]])
    pdf_mixture = 0.5 * rv1.pdf(pos) + 0.5 * rv2.pdf(pos)
    
    # 2. Create a distribution with a ring-like shape
    # Using a formula for a ring shape
    r_squared = x**2 + y**2
    ring_pdf = np.exp(-0.5 * (r_squared - 2)**2 / 0.2)
    
    # 3. Create a distribution with multiple modes
    rv3 = multivariate_normal(mean=[-1.5, 1.5], cov=[[0.4, 0], [0, 0.4]])
    rv4 = multivariate_normal(mean=[1.5, 1.5], cov=[[0.4, 0], [0, 0.4]])
    rv5 = multivariate_normal(mean=[0, -1.5], cov=[[0.4, 0], [0, 0.4]])
    pdf_multimodal = (1/3) * rv3.pdf(pos) + (1/3) * rv4.pdf(pos) + (1/3) * rv5.pdf(pos)
    
    # 4. Create a distribution with a valley (negative exponential of abs)
    valley_pdf = np.exp(-2 * (np.abs(x) + np.abs(y)))
    
    # Create figure with multiple plots
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Mixture of two normals
    plt.subplot(221)
    plt.contour(x, y, pdf_mixture, levels=10, cmap='viridis')
    plt.title('Mixture of Two Normals', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.grid(alpha=0.3)
    
    # Plot 2: Ring-shaped distribution
    plt.subplot(222)
    plt.contour(x, y, ring_pdf, levels=10, cmap='viridis')
    plt.title('Ring-Shaped Distribution', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.grid(alpha=0.3)
    
    # Plot 3: Multimodal distribution
    plt.subplot(223)
    plt.contour(x, y, pdf_multimodal, levels=10, cmap='viridis')
    plt.title('Multimodal Distribution', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.grid(alpha=0.3)
    
    # Plot 4: Valley-shaped distribution
    plt.subplot(224)
    plt.contour(x, y, valley_pdf, levels=10, cmap='viridis')
    plt.title('Valley-Shaped Distribution', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'different_distributions_contours.png'), dpi=300)
    plt.close()
    
    print("QUESTION 2: How do contour plots represent complex non-Gaussian probability distributions, and what insights can be gained from these visualizations?")

def bivariate_normal_probability_regions(save_dir):
    """Generate contour plots showing probability regions for bivariate normal
    
    QUESTION: What is the relationship between contour lines and probability regions in a 
    bivariate normal distribution, and how does this extend our understanding of confidence intervals?
    """
    # Create a grid of points
    x, y = np.mgrid[-3:3:.01, -3:3:.01]
    pos = np.dstack((x, y))
    
    # Create a correlated bivariate normal
    cov_matrix = [[1, 0.5], [0.5, 1]]
    rv = multivariate_normal(mean=[0, 0], cov=cov_matrix)
    pdf = rv.pdf(pos)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Compute contour levels for different probability regions
    # For bivariate normal, these correspond to specific Mahalanobis distances
    # 1 sigma = 39.4% probability region
    # 2 sigma = 86.5% probability region
    # 3 sigma = 98.9% probability region
    
    # Calculate the peak PDF value (at the mean)
    peak_pdf = rv.pdf(np.array([0, 0]))
    
    # Get contour levels corresponding to 1, 2, and 3 sigma regions
    # For bivariate normal, these are exp(-0.5 * d^2) where d is Mahalanobis distance
    level_1sigma = peak_pdf * np.exp(-0.5 * 1**2)
    level_2sigma = peak_pdf * np.exp(-0.5 * 2**2)
    level_3sigma = peak_pdf * np.exp(-0.5 * 3**2)
    
    # Create contour plot with these levels
    contour = plt.contour(x, y, pdf, levels=[level_3sigma, level_2sigma, level_1sigma], 
                       colors=['blue', 'green', 'red'])
    
    # Add filled contours for better visualization
    plt.contourf(x, y, pdf, levels=[level_3sigma, level_2sigma, level_1sigma, peak_pdf],
                colors=['lightblue', 'lightgreen', 'pink'], alpha=0.3)
    
    # Label the contours with probability
    plt.clabel(contour, inline=1, fontsize=10, fmt={level_1sigma: '39.4%', 
                                                 level_2sigma: '86.5%', 
                                                 level_3sigma: '98.9%'})
    
    # Add title and labels
    plt.title('Probability Regions for Bivariate Normal\nwith Correlation = 0.5', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.grid(alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Add legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='red', lw=2),
                   Line2D([0], [0], color='green', lw=2),
                   Line2D([0], [0], color='blue', lw=2)]
    plt.legend(custom_lines, ['1σ (39.4%)', '2σ (86.5%)', '3σ (98.9%)'], 
              loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'bivariate_normal_probability_regions.png'), dpi=300)
    plt.close()
    
    print("QUESTION 3: What is the relationship between contour lines and probability regions in a bivariate normal distribution, and how does this extend our understanding of confidence intervals?")

def conditional_distributions_visual(save_dir):
    """Generate visualization of conditional distributions from a joint distribution
    
    QUESTION: How do conditional distributions change as we vary the value of one variable in a
    bivariate normal distribution, and what does this tell us about the relationship between variables?
    """
    # Create a grid of points
    x, y = np.mgrid[-3:3:.01, -3:3:.01]
    pos = np.dstack((x, y))
    
    # Create a correlated bivariate normal
    cov_matrix = [[1, 0.7], [0.7, 1]]
    rv = multivariate_normal(mean=[0, 0], cov=cov_matrix)
    pdf = rv.pdf(pos)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot contours of the joint distribution
    plt.contour(x, y, pdf, levels=10, colors='black', alpha=0.5)
    plt.contourf(x, y, pdf, levels=20, cmap='viridis', alpha=0.5)
    
    # Calculate conditional distributions for specific values of x
    x_values = [-1.5, 0, 1.5]
    colors = ['red', 'blue', 'green']
    
    # For each x value, plot the conditional distribution P(Y|X=x)
    for i, x_val in enumerate(x_values):
        # Formula for conditional mean: mu_Y|X = mu_Y + rho*(sigma_Y/sigma_X)*(x - mu_X)
        # For our case: mu_X = mu_Y = 0, sigma_X = sigma_Y = 1, so this simplifies to:
        cond_mean = 0.7 * x_val  # mu_Y|X = rho * x_val
        
        # Formula for conditional variance: sigma²_Y|X = sigma²_Y * (1 - rho²)
        cond_var = 1 * (1 - 0.7**2)  # sigma²_Y|X = 1 * (1 - 0.7²)
        cond_std = np.sqrt(cond_var)
        
        # Generate y values for this conditional distribution
        y_vals = np.linspace(-3, 3, 1000)
        cond_pdf = (1 / (cond_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((y_vals - cond_mean) / cond_std)**2)
        
        # Normalize for plotting alongside the contours
        cond_pdf_normalized = cond_pdf / np.max(cond_pdf) * 0.8
        
        # Plot the conditional distribution as a line at fixed x
        plt.plot(x_val + cond_pdf_normalized, y_vals, color=colors[i], linewidth=2, 
                label=f'P(Y|X={x_val})')
        
        # Mark the mean of this conditional distribution
        plt.plot(x_val, cond_mean, 'o', color=colors[i], markersize=6)
        
        # Draw a vertical line at this x value
        plt.axvline(x=x_val, color=colors[i], linestyle='--', alpha=0.5)
    
    # Add title and labels
    plt.title('Conditional Distributions in a Bivariate Normal\nwith Correlation = 0.7', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.grid(alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'conditional_distributions.png'), dpi=300)
    plt.close()
    
    print("QUESTION 4: How do conditional distributions change as we vary the value of one variable in a bivariate normal distribution, and what does this tell us about the relationship between variables?")

def marginal_distributions_visual(save_dir):
    """Generate visualization of marginal distributions from a joint distribution
    
    QUESTION: What is the relationship between joint and marginal distributions, and what
    information is preserved or lost when examining only marginal distributions?
    """
    # Create a grid of points
    x, y = np.mgrid[-3:3:.01, -3:3:.01]
    pos = np.dstack((x, y))
    
    # Create a correlated bivariate normal
    cov_matrix = [[1, 0.7], [0.7, 1]]
    rv = multivariate_normal(mean=[0, 0], cov=cov_matrix)
    pdf = rv.pdf(pos)
    
    # Create figure with gridspec for custom layout
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3],
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)
    
    # Create the three subplots
    ax_joint = fig.add_subplot(gs[1, 0])  # Joint distribution
    ax_marg_x = fig.add_subplot(gs[0, 0], sharex=ax_joint)  # Marginal for X
    ax_marg_y = fig.add_subplot(gs[1, 1], sharey=ax_joint)  # Marginal for Y
    
    # Plot joint distribution
    contour = ax_joint.contourf(x, y, pdf, levels=20, cmap='viridis')
    ax_joint.contour(x, y, pdf, levels=10, colors='black', alpha=0.5)
    ax_joint.set_xlabel('X', fontsize=12)
    ax_joint.set_ylabel('Y', fontsize=12)
    ax_joint.grid(alpha=0.3)
    
    # Calculate marginal distributions
    x_vals = np.linspace(-3, 3, 1000)
    marginal_x = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x_vals**2)  # Standard normal for X
    marginal_y = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x_vals**2)  # Standard normal for Y
    
    # Plot marginal for X
    ax_marg_x.plot(x_vals, marginal_x, 'r-', linewidth=2)
    ax_marg_x.fill_between(x_vals, marginal_x, alpha=0.3, color='red')
    ax_marg_x.set_ylabel('P(X)', fontsize=12)
    ax_marg_x.grid(alpha=0.3)
    ax_marg_x.set_yticks([0, 0.2, 0.4])  # Simplified y-ticks
    
    # Plot marginal for Y
    ax_marg_y.plot(marginal_y, x_vals, 'b-', linewidth=2)
    ax_marg_y.fill_betweenx(x_vals, marginal_y, alpha=0.3, color='blue')
    ax_marg_y.set_xlabel('P(Y)', fontsize=12)
    ax_marg_y.grid(alpha=0.3)
    ax_marg_y.set_xticks([0, 0.2, 0.4])  # Simplified x-ticks
    
    # Turn off tick labels on shared axes
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    
    # Add title to overall figure
    fig.suptitle('Joint and Marginal Distributions\nBivariate Normal with Correlation = 0.7', 
                fontsize=14)
    
    plt.savefig(os.path.join(save_dir, 'marginal_distributions.png'), dpi=300)
    plt.close()
    
    print("QUESTION 5: What is the relationship between joint and marginal distributions, and what information is preserved or lost when examining only marginal distributions?")

def contour_3d_comparison(save_dir):
    """Generate comparison of contour plots and 3D surface plots
    
    QUESTION: What is the geometric relationship between 3D probability density surfaces and 
    their 2D contour plot representations, and what are the advantages of each visualization?
    """
    # Create a grid of points
    x, y = np.mgrid[-3:3:.1, -3:3:.1]
    pos = np.dstack((x, y))
    
    # Create a correlated bivariate normal
    cov_matrix = [[1, 0.5], [0.5, 1]]
    rv = multivariate_normal(mean=[0, 0], cov=cov_matrix)
    pdf = rv.pdf(pos)
    
    # Create figure with two subplots - contour plot and 3D surface
    fig = plt.figure(figsize=(15, 7))
    
    # Contour plot
    ax1 = fig.add_subplot(121)
    contour = ax1.contour(x, y, pdf, levels=10, cmap='viridis')
    ax1.contourf(x, y, pdf, levels=20, cmap='viridis', alpha=0.5)
    ax1.clabel(contour, inline=1, fontsize=10)
    ax1.set_title('Contour Plot', fontsize=14)
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)
    ax1.grid(alpha=0.3)
    
    # 3D surface plot
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(x, y, pdf, cmap='viridis', alpha=0.8,
                          linewidth=0, antialiased=True)
    ax2.set_title('3D Surface Plot', fontsize=14)
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.set_zlabel('Probability Density', fontsize=12)
    
    # Adjust viewing angle for better visualization
    ax2.view_init(elev=30, azim=135)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'contour_3d_comparison.png'), dpi=300)
    plt.close()
    
    print("QUESTION 6: What is the geometric relationship between 3D probability density surfaces and their 2D contour plot representations, and what are the advantages of each visualization?")

def generate_question_images():
    """Generate all images for the contour plot visual examples"""
    # Create directories
    save_dir = create_directories()
    
    # Generate all the visualizations
    bivariate_normal_contour(save_dir)
    different_distributions_contour(save_dir)
    bivariate_normal_probability_regions(save_dir)
    conditional_distributions_visual(save_dir)
    marginal_distributions_visual(save_dir)
    contour_3d_comparison(save_dir)
    
    print(f"Generated all question images in {save_dir}")
    print("\nThe following questions need to be solved in the answer file:")
    print("1. How does changing the correlation coefficient and variance affect the shape of contour plots in a bivariate normal distribution?")
    print("2. How do contour plots represent complex non-Gaussian probability distributions, and what insights can be gained from these visualizations?")
    print("3. What is the relationship between contour lines and probability regions in a bivariate normal distribution, and how does this extend our understanding of confidence intervals?")
    print("4. How do conditional distributions change as we vary the value of one variable in a bivariate normal distribution, and what does this tell us about the relationship between variables?")
    print("5. What is the relationship between joint and marginal distributions, and what information is preserved or lost when examining only marginal distributions?")
    print("6. What is the geometric relationship between 3D probability density surfaces and their 2D contour plot representations, and what are the advantages of each visualization?")
    
    return save_dir

if __name__ == "__main__":
    generate_question_images() 