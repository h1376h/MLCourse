import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D

def ensure_directory_exists(directory):
    """Ensure the specified directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def covariance_matrix_contours():
    """Visualize multivariate Gaussians with different covariance matrices"""
    # Create a figure with 2x2 subplots
    fig = plt.figure(figsize=(15, 15))
    
    # Create a grid of points
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    def multivariate_gaussian(pos, mu, Sigma):
        """Return the multivariate Gaussian distribution on array pos."""
        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2*np.pi)**n * Sigma_det)
        
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) for each point
        fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
        
        return np.exp(-fac / 2) / N
    
    # Case 1: Diagonal covariance with equal variances (scaled identity matrix)
    mu1 = np.array([0., 0.])
    Sigma1 = np.array([[1.0, 0.0], [0.0, 1.0]])  # Identity matrix
    
    ax1 = fig.add_subplot(221)
    Z1 = multivariate_gaussian(pos, mu1, Sigma1)
    
    # Plot contours
    contour_levels = np.linspace(0.01, 0.1, 5)
    cp1 = ax1.contour(X, Y, Z1, levels=contour_levels, colors='black')
    ax1.clabel(cp1, inline=True, fontsize=10)
    
    # Add an ellipse to represent the covariance
    lambda_, v = np.linalg.eig(Sigma1)
    lambda_ = np.sqrt(lambda_)
    
    for j in range(1, 4):
        ell = Ellipse(xy=(0, 0),
                     width=lambda_[0]*j*2, height=lambda_[1]*j*2,
                     angle=np.rad2deg(np.arctan2(v[1, 0], v[0, 0])),
                     edgecolor='red', facecolor='none', linestyle='--')
        ax1.add_patch(ell)
        if j == 2:
            ax1.text(0, lambda_[1]*j, '2σ', color='red', ha='center', va='bottom')
            ax1.text(lambda_[0]*j, 0, '2σ', color='red', ha='left', va='center')
    
    ax1.set_title('Case 1: Diagonal Covariance (σ₁² = σ₂² = 1)\nCircular Contours')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True)
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)
    
    # Case 2: Diagonal covariance with different variances
    mu2 = np.array([0., 0.])
    Sigma2 = np.array([[3.0, 0.0], [0.0, 0.5]])  # Diagonal matrix with different values
    
    ax2 = fig.add_subplot(222)
    Z2 = multivariate_gaussian(pos, mu2, Sigma2)
    
    # Plot contours
    cp2 = ax2.contour(X, Y, Z2, levels=contour_levels, colors='black')
    ax2.clabel(cp2, inline=True, fontsize=10)
    
    # Add an ellipse to represent the covariance
    lambda_, v = np.linalg.eig(Sigma2)
    lambda_ = np.sqrt(lambda_)
    
    for j in range(1, 4):
        ell = Ellipse(xy=(0, 0),
                     width=lambda_[0]*j*2, height=lambda_[1]*j*2,
                     angle=np.rad2deg(np.arctan2(v[1, 0], v[0, 0])),
                     edgecolor='red', facecolor='none', linestyle='--')
        ax2.add_patch(ell)
        if j == 2:
            ax2.text(0, lambda_[1]*j, '2σ₂', color='red', ha='center', va='bottom')
            ax2.text(lambda_[0]*j, 0, '2σ₁', color='red', ha='left', va='center')
    
    ax2.set_title('Case 2: Diagonal Covariance (σ₁² = 3, σ₂² = 0.5)\nAxis-Aligned Elliptical Contours')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.grid(True)
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-5, 5)
    
    # Case 3: Non-diagonal covariance with positive correlation
    mu3 = np.array([0., 0.])
    Sigma3 = np.array([[2.0, 1.5], [1.5, 2.0]])  # Non-diagonal with positive correlation
    
    ax3 = fig.add_subplot(223)
    Z3 = multivariate_gaussian(pos, mu3, Sigma3)
    
    # Plot contours
    cp3 = ax3.contour(X, Y, Z3, levels=contour_levels, colors='black')
    ax3.clabel(cp3, inline=True, fontsize=10)
    
    # Add an ellipse to represent the covariance
    lambda_, v = np.linalg.eig(Sigma3)
    lambda_ = np.sqrt(lambda_)
    
    for j in range(1, 4):
        ell = Ellipse(xy=(0, 0),
                     width=lambda_[0]*j*2, height=lambda_[1]*j*2,
                     angle=np.rad2deg(np.arctan2(v[1, 0], v[0, 0])),
                     edgecolor='red', facecolor='none', linestyle='--')
        ax3.add_patch(ell)
    
    # Add correlation explanation
    ax3.plot([-5, 5], [-5, 5], 'r--', alpha=0.5)
    corr = Sigma3[0, 1] / np.sqrt(Sigma3[0, 0] * Sigma3[1, 1])
    ax3.text(-4, -3, f'Correlation: ρ = {corr:.2f}', color='red',
             bbox=dict(facecolor='white', alpha=0.7))
    
    ax3.set_title('Case 3: Non-Diagonal Covariance (Positive Correlation)\nRotated Elliptical Contours')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.grid(True)
    ax3.set_xlim(-5, 5)
    ax3.set_ylim(-5, 5)
    
    # Case 4: Non-diagonal covariance with negative correlation
    mu4 = np.array([0., 0.])
    Sigma4 = np.array([[2.0, -1.5], [-1.5, 2.0]])  # Non-diagonal with negative correlation
    
    ax4 = fig.add_subplot(224)
    Z4 = multivariate_gaussian(pos, mu4, Sigma4)
    
    # Plot contours
    cp4 = ax4.contour(X, Y, Z4, levels=contour_levels, colors='black')
    ax4.clabel(cp4, inline=True, fontsize=10)
    
    # Add an ellipse to represent the covariance
    lambda_, v = np.linalg.eig(Sigma4)
    lambda_ = np.sqrt(lambda_)
    
    for j in range(1, 4):
        ell = Ellipse(xy=(0, 0),
                     width=lambda_[0]*j*2, height=lambda_[1]*j*2,
                     angle=np.rad2deg(np.arctan2(v[1, 0], v[0, 0])),
                     edgecolor='red', facecolor='none', linestyle='--')
        ax4.add_patch(ell)
    
    # Add correlation explanation
    ax4.plot([-5, 5], [5, -5], 'r--', alpha=0.5)
    corr = Sigma4[0, 1] / np.sqrt(Sigma4[0, 0] * Sigma4[1, 1])
    ax4.text(-4, 3, f'Correlation: ρ = {corr:.2f}', color='red',
             bbox=dict(facecolor='white', alpha=0.7))
    
    ax4.set_title('Case 4: Non-Diagonal Covariance (Negative Correlation)\nRotated Elliptical Contours')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.grid(True)
    ax4.set_xlim(-5, 5)
    ax4.set_ylim(-5, 5)
    
    plt.tight_layout()
    return fig

def basic_2d_example():
    """Simple example showing 1D and 2D normal distributions"""
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: 1D Normal Distributions with different variances
    ax1 = fig.add_subplot(131)
    x = np.linspace(-5, 5, 1000)
    
    # Standard normal distribution
    y1 = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * x**2)
    # Normal with variance 0.5
    y2 = (1/np.sqrt(2*np.pi*0.5)) * np.exp(-0.5 * x**2/0.5)
    # Normal with variance 2
    y3 = (1/np.sqrt(2*np.pi*2)) * np.exp(-0.5 * x**2/2)
    
    ax1.plot(x, y1, 'b-', label='σ² = 1')
    ax1.plot(x, y2, 'r-', label='σ² = 0.5')
    ax1.plot(x, y3, 'g-', label='σ² = 2')
    
    # Add vertical lines at ±σ, ±2σ, ±3σ for standard normal
    for i in range(1, 4):
        ax1.axvline(i, color='b', linestyle='--', alpha=0.3)
        ax1.axvline(-i, color='b', linestyle='--', alpha=0.3)
        if i == 1:
            ax1.text(i, 0.05, f'{i}σ', ha='left', va='bottom', color='b')
            
    ax1.set_title('1D Normal Distributions with Different Variances')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Probability Density')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: 2D Independent Normal Distribution (Diagonal Covariance)
    ax2 = fig.add_subplot(132)
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate PDF values for a 2D independent normal (diagonal covariance)
    Z = (1/(2*np.pi)) * np.exp(-0.5*(X**2 + Y**2))
    
    # Plot the contours
    contour_levels = np.linspace(0.01, 0.15, 5)
    cp = ax2.contour(X, Y, Z, levels=contour_levels, colors='black')
    ax2.clabel(cp, inline=True, fontsize=10)
    
    # Add 1σ, 2σ and 3σ circles
    for i in range(1, 4):
        circle = plt.Circle((0, 0), i, fill=False, edgecolor='red', linestyle='--')
        ax2.add_patch(circle)
        if i == 2:
            ax2.text(0, i, f'{i}σ', ha='center', va='bottom', color='red')
    
    ax2.set_title('2D Standard Normal Distribution\n(Independent Variables)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.grid(True)
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    
    # Plot 3: 2D Normal with different variances but still independent
    ax3 = fig.add_subplot(133)
    
    # Calculate PDF for 2D normal with different variances
    Z = (1/(2*np.pi*np.sqrt(2*0.5))) * np.exp(-0.5*(X**2/2 + Y**2/0.5))
    
    # Plot the contours
    cp = ax3.contour(X, Y, Z, levels=contour_levels, colors='black')
    ax3.clabel(cp, inline=True, fontsize=10)
    
    # Add ellipses to represent the covariance
    for i in range(1, 4):
        ellipse = Ellipse(xy=(0, 0), width=i*2*np.sqrt(2), height=i*2*np.sqrt(0.5), 
                         fill=False, edgecolor='red', linestyle='--')
        ax3.add_patch(ellipse)
        if i == 2:
            ax3.text(0, np.sqrt(0.5)*i, f'{i}σ₂', ha='center', va='bottom', color='red')
            ax3.text(np.sqrt(2)*i, 0, f'{i}σ₁', ha='left', va='center', color='red')
    
    ax3.set_title('2D Normal with Different Variances\n(Independent Variables)')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.grid(True)
    ax3.set_xlim(-3, 3)
    ax3.set_ylim(-3, 3)
    
    plt.tight_layout()
    return fig

def gaussian_3d_visualization():
    """Create 3D visualization of Gaussian probability density functions"""
    fig = plt.figure(figsize=(18, 6))
    
    # Create a grid of points
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    def multivariate_gaussian(pos, mu, Sigma):
        """Return the multivariate Gaussian distribution on array pos."""
        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2*np.pi)**n * Sigma_det)
        
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) for each point
        fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
        
        return np.exp(-fac / 2) / N
    
    # Case 1: Standard Normal Distribution (Identity Covariance)
    ax1 = fig.add_subplot(131, projection='3d')
    mu1 = np.array([0., 0.])
    Sigma1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    Z1 = multivariate_gaussian(pos, mu1, Sigma1)
    
    # Plot the surface
    surf1 = ax1.plot_surface(X, Y, Z1, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.7)
    
    # Plot the contours on the bottom of the graph
    ax1.contour(X, Y, Z1, zdir='z', offset=0, cmap=cm.viridis)
    
    ax1.set_title('Standard Bivariate Normal\n(Identity Covariance)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Probability Density')
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.view_init(30, 45)
    
    # Case 2: Diagonal Covariance with Different Variances
    ax2 = fig.add_subplot(132, projection='3d')
    mu2 = np.array([0., 0.])
    Sigma2 = np.array([[2.0, 0.0], [0.0, 0.5]])
    Z2 = multivariate_gaussian(pos, mu2, Sigma2)
    
    # Plot the surface
    surf2 = ax2.plot_surface(X, Y, Z2, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.7)
    
    # Plot the contours on the bottom of the graph
    ax2.contour(X, Y, Z2, zdir='z', offset=0, cmap=cm.viridis)
    
    ax2.set_title('Bivariate Normal with Different Variances\n(Diagonal Covariance)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Probability Density')
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.view_init(30, 45)
    
    # Case 3: Non-diagonal Covariance with Correlation
    ax3 = fig.add_subplot(133, projection='3d')
    mu3 = np.array([0., 0.])
    Sigma3 = np.array([[1.0, 0.8], [0.8, 1.0]])
    Z3 = multivariate_gaussian(pos, mu3, Sigma3)
    
    # Plot the surface
    surf3 = ax3.plot_surface(X, Y, Z3, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.7)
    
    # Plot the contours on the bottom of the graph
    ax3.contour(X, Y, Z3, zdir='z', offset=0, cmap=cm.viridis)
    
    # Calculate correlation coefficient
    corr = Sigma3[0, 1] / np.sqrt(Sigma3[0, 0] * Sigma3[1, 1])
    
    ax3.set_title(f'Bivariate Normal with Correlation\n(ρ = {corr:.2f})')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Probability Density')
    ax3.set_xlim(-3, 3)
    ax3.set_ylim(-3, 3)
    ax3.view_init(30, 45)
    
    plt.tight_layout()
    return fig

def covariance_eigenvalue_visualization():
    """Visualize the relationship between covariance matrices, eigenvalues, and eigenvectors"""
    fig = plt.figure(figsize=(15, 15))
    
    # Create a grid of points
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    def multivariate_gaussian(pos, mu, Sigma):
        """Return the multivariate Gaussian distribution on array pos."""
        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2*np.pi)**n * Sigma_det)
        
        fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
        return np.exp(-fac / 2) / N
    
    # Define covariance matrices with increasing correlation
    mu = np.array([0., 0.])
    correlations = [0, 0.3, 0.6, 0.9]
    
    for i, corr in enumerate(correlations):
        # Create covariance matrix
        Sigma = np.array([
            [1.0, corr],
            [corr, 1.0]
        ])
        
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(Sigma)
        
        # Calculate PDF
        Z = multivariate_gaussian(pos, mu, Sigma)
        
        # Plot
        ax = fig.add_subplot(2, 2, i+1)
        
        # Plot contours
        contour_levels = np.linspace(0.01, 0.2, 5)
        cp = ax.contour(X, Y, Z, levels=contour_levels, colors='black')
        ax.clabel(cp, inline=True, fontsize=10)
        
        # Plot ellipses
        for j in range(1, 3):
            ell = Ellipse(xy=(0, 0),
                         width=np.sqrt(eigenvalues[0])*j*2, 
                         height=np.sqrt(eigenvalues[1])*j*2,
                         angle=np.rad2deg(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])),
                         edgecolor='red', facecolor='none', linestyle='--')
            ax.add_patch(ell)
        
        # Plot eigenvectors
        for j in range(2):
            vec = eigenvectors[:, j] * np.sqrt(eigenvalues[j]) * 2
            ax.arrow(0, 0, vec[0], vec[1], head_width=0.2, head_length=0.3, 
                     fc='blue', ec='blue', label=f'Eigenvector {j+1}')
            ax.text(vec[0]*1.1, vec[1]*1.1, f'λ{j+1}={eigenvalues[j]:.2f}', 
                    color='blue', ha='center', va='center')
        
        ax.set_title(f'Correlation: ρ = {corr:.1f}\nEigenvalues: λ₁={eigenvalues[0]:.2f}, λ₂={eigenvalues[1]:.2f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
    
    plt.tight_layout()
    return fig

def generate_covariance_contour_plots():
    """Generate and save covariance matrix contour plots"""
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images", "Contour_Plots")
    ensure_directory_exists(images_dir)
    
    # Generate and save examples
    examples = [
        {"function": covariance_matrix_contours, "filename": "covariance_matrix_contours.png"},
        {"function": basic_2d_example, "filename": "basic_2d_normal_examples.png"},
        {"function": gaussian_3d_visualization, "filename": "gaussian_3d_visualization.png"},
        {"function": covariance_eigenvalue_visualization, "filename": "covariance_eigenvalue_visualization.png"}
    ]
    
    for example in examples:
        try:
            fig = example["function"]()
            save_path = os.path.join(images_dir, example["filename"])
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            print(f"Generated {example['filename']}")
            print(f"Saved to: {save_path}")
        except Exception as e:
            print(f"Error generating {example['filename']}: {e}")
    
    return "Covariance matrix contour plots generated successfully!"

if __name__ == "__main__":
    generate_covariance_contour_plots() 