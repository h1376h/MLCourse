import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button

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

def simple_covariance_example_real_world():
    """Simple real-world example of covariance using height and weight data."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Simulated height (cm) and weight (kg) data with positive correlation
    np.random.seed(42)  # For reproducibility
    heights = 170 + np.random.normal(0, 7, 100)  # Mean 170cm, std 7cm
    weights = heights * 0.5 + np.random.normal(0, 5, 100)  # Positively correlated with heights
    
    # Calculate covariance matrix
    data = np.vstack([heights, weights]).T
    cov_matrix = np.cov(data, rowvar=False)
    
    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Plot the data points
    ax.scatter(heights, weights, alpha=0.7, label='Height-Weight Data')
    
    # Calculate mean
    mean_height, mean_weight = np.mean(heights), np.mean(weights)
    
    # Draw the covariance ellipse (2σ)
    for j in [1, 2]:
        ell = Ellipse(xy=(mean_height, mean_weight),
                     width=2*j*np.sqrt(eigenvalues[0]), 
                     height=2*j*np.sqrt(eigenvalues[1]),
                     angle=np.rad2deg(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])),
                     edgecolor='red', facecolor='none', linestyle='--')
        ax.add_patch(ell)
        if j == 2:
            ax.text(mean_height, mean_weight + j*np.sqrt(eigenvalues[1]), 
                    f'2σ confidence region', color='red', ha='center', va='bottom')
    
    # Plot the eigenvectors (principal components)
    for i in range(2):
        vec = eigenvectors[:, i] * np.sqrt(eigenvalues[i])
        ax.arrow(mean_height, mean_weight, vec[0], vec[1], 
                 head_width=1, head_length=1.5, fc='blue', ec='blue')
        ax.text(mean_height + vec[0]*1.1, mean_weight + vec[1]*1.1, 
                f'PC{i+1}', color='blue', ha='center', va='center')
    
    # Add labels and title
    ax.set_xlabel('Height (cm)')
    ax.set_ylabel('Weight (kg)')
    ax.set_title('Height vs Weight: A Natural Example of Positive Covariance')
    ax.grid(True)
    ax.axis('equal')
    
    # Add text explaining the covariance
    corr = cov_matrix[0, 1] / np.sqrt(cov_matrix[0, 0] * cov_matrix[1, 1])
    textstr = f'Covariance Matrix:\n[[{cov_matrix[0,0]:.1f}, {cov_matrix[0,1]:.1f}],\n [{cov_matrix[1,0]:.1f}, {cov_matrix[1,1]:.1f}]]\n\nCorrelation: {corr:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    return fig

def toy_data_covariance_change():
    """Visualize how a dataset's covariance changes with rotation."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Create a simple 2D dataset
    np.random.seed(42)
    n_points = 300
    x = np.random.normal(0, 1, n_points)
    y = np.random.normal(0, 1, n_points)
    data_original = np.vstack([x, y]).T
    
    # Rotation matrices for different angles
    angles = [0, 30, 60]
    titles = ['Original Data', '30° Rotation', '60° Rotation']
    
    for i, (angle, title) in enumerate(zip(angles, titles)):
        # Convert angle to radians
        theta = np.radians(angle)
        
        # Create rotation matrix
        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        # Rotate the data
        data_rotated = np.dot(data_original, rot_matrix)
        
        # Calculate covariance matrix
        cov_matrix = np.cov(data_rotated, rowvar=False)
        
        # Plot the data
        axs[i].scatter(data_rotated[:, 0], data_rotated[:, 1], alpha=0.5, s=10)
        
        # Get eigenvalues and eigenvectors for ellipse
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Draw 2σ ellipse
        ell = Ellipse(xy=(0, 0),
                     width=4*np.sqrt(eigenvalues[0]), 
                     height=4*np.sqrt(eigenvalues[1]),
                     angle=np.rad2deg(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])),
                     edgecolor='red', facecolor='none', linestyle='--')
        axs[i].add_patch(ell)
        
        # Add covariance info
        corr = cov_matrix[0, 1] / np.sqrt(cov_matrix[0, 0] * cov_matrix[1, 1])
        axs[i].text(0.05, 0.95, f'Cov(x,y) = {cov_matrix[0,1]:.2f}\nCorr = {corr:.2f}', 
                   transform=axs[i].transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        axs[i].set_title(title)
        axs[i].set_xlabel('x')
        axs[i].set_ylabel('y')
        axs[i].set_xlim(-4, 4)
        axs[i].set_ylim(-4, 4)
        axs[i].grid(True)
        axs[i].set_aspect('equal')
    
    plt.tight_layout()
    return fig

def simple_mahalanobis_distance():
    """Visualize Mahalanobis distance vs Euclidean distance for correlated data."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create correlated data
    np.random.seed(42)
    cov_matrix = np.array([[2.0, 1.5], [1.5, 2.0]])  # Positive correlation
    mean = np.array([0, 0])
    
    # Generate multivariate normal data
    data = np.random.multivariate_normal(mean, cov_matrix, 300)
    
    # Calculate the inverse of the covariance matrix
    cov_inv = np.linalg.inv(cov_matrix)
    
    # Test points for distance calculation
    test_points = np.array([
        [2, 0],    # Point along x-axis
        [0, 2],    # Point along y-axis
        [2, 2],    # Point in first quadrant
        [-1.5, 1.5]  # Point in second quadrant
    ])
    
    # Compute Mahalanobis distances
    mahalanobis_distances = []
    for point in test_points:
        diff = point - mean
        mahalanobis_distance = np.sqrt(diff.dot(cov_inv).dot(diff))
        mahalanobis_distances.append(mahalanobis_distance)
    
    # Plot the data points
    ax.scatter(data[:, 0], data[:, 1], alpha=0.5, s=10, label='Data Points')
    
    # Plot test points
    ax.scatter(test_points[:, 0], test_points[:, 1], color='red', s=100, marker='*', label='Test Points')
    
    # Get eigenvalues and eigenvectors for contour ellipses
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Draw multiple contour ellipses representing equal Mahalanobis distances
    for m_dist in [1, 2, 3]:
        ell = Ellipse(xy=(0, 0),
                     width=2*m_dist*np.sqrt(eigenvalues[0]), 
                     height=2*m_dist*np.sqrt(eigenvalues[1]),
                     angle=np.rad2deg(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])),
                     edgecolor='purple', facecolor='none', linestyle='-', alpha=0.7)
        ax.add_patch(ell)
        ax.text(0, m_dist*np.sqrt(eigenvalues[1]), f'M-dist = {m_dist}', 
               color='purple', ha='center', va='bottom')
    
    # Add text for the test points
    for i, (point, dist) in enumerate(zip(test_points, mahalanobis_distances)):
        ax.text(point[0], point[1] + 0.3, f'P{i+1}: M-dist = {dist:.2f}', ha='center')
    
    # Draw Euclidean distance circles for comparison
    for e_dist in [1, 2, 3]:
        circle = plt.Circle((0, 0), e_dist, fill=False, edgecolor='green', linestyle='--')
        ax.add_patch(circle)
        ax.text(e_dist, 0, f'E-dist = {e_dist}', color='green', ha='left', va='center')
    
    ax.set_title('Mahalanobis Distance vs Euclidean Distance\nfor Correlated Data')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.legend()
    
    # Add covariance matrix info
    corr = cov_matrix[0, 1] / np.sqrt(cov_matrix[0, 0] * cov_matrix[1, 1])
    textstr = f'Covariance Matrix:\n[[{cov_matrix[0,0]:.1f}, {cov_matrix[0,1]:.1f}],\n [{cov_matrix[1,0]:.1f}, {cov_matrix[1,1]:.1f}]]\n\nCorrelation: {corr:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    return fig

def emoji_covariance_example():
    """Create a fun example using emoji-like shapes to show covariance concepts."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Create a smiley face for the positive correlation
    theta = np.linspace(0, 2*np.pi, 100)
    # Face circle
    face_x = 3 * np.cos(theta)
    face_y = 3 * np.sin(theta)
    
    # Eyes (ellipses showing covariance)
    eye_left_x = -1.2 + 0.5 * np.cos(theta)
    eye_left_y = 1 + 0.5 * np.sin(theta)
    
    eye_right_x = 1.2 + 0.5 * np.cos(theta)
    eye_right_y = 1 + 0.5 * np.sin(theta)
    
    # Smiling mouth (showing positive correlation)
    mouth_theta = np.linspace(0, np.pi, 50)
    mouth_x = 2 * np.cos(mouth_theta)
    mouth_y = -1 + 1.2 * np.sin(mouth_theta)
    
    # Plot the happy face on the left subplot
    ax1.plot(face_x, face_y, 'k-', linewidth=2)
    ax1.plot(eye_left_x, eye_left_y, 'k-', linewidth=2)
    ax1.plot(eye_right_x, eye_right_y, 'k-', linewidth=2)
    ax1.plot(mouth_x, mouth_y, 'k-', linewidth=2)
    
    # Add positive correlation contour
    x, y = np.mgrid[-3:3:.01, -3:3:.01]
    pos = np.dstack((x, y))
    cov_pos = np.array([[1.0, 0.8], [0.8, 1.0]])
    mean = np.array([0, 0])
    
    rv = np.random.multivariate_normal(mean, cov_pos, 100)
    ax1.scatter(rv[:, 0], rv[:, 1], color='blue', alpha=0.3, s=10)
    
    # Add positive covariance ellipse
    eigenvalues, eigenvectors = np.linalg.eig(cov_pos)
    ell_pos = Ellipse(xy=(0, 0),
                    width=4*np.sqrt(eigenvalues[0]), 
                    height=4*np.sqrt(eigenvalues[1]),
                    angle=np.rad2deg(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])),
                    edgecolor='blue', facecolor='none', linestyle='--')
    ax1.add_patch(ell_pos)
    
    ax1.set_title('Positive Correlation: Happy Data! 😊\nPoints tend to increase together')
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.set_aspect('equal')
    ax1.grid(True)
    
    # Create a sad face for the negative correlation on the right subplot
    # Face circle (reuse from above)
    
    # Eyes (reuse from above)
    
    # Sad mouth (showing negative correlation)
    sad_mouth_theta = np.linspace(np.pi, 2*np.pi, 50)
    sad_mouth_x = 2 * np.cos(sad_mouth_theta)
    sad_mouth_y = -1 + 1.2 * np.sin(sad_mouth_theta)
    
    # Plot the sad face
    ax2.plot(face_x, face_y, 'k-', linewidth=2)
    ax2.plot(eye_left_x, eye_left_y, 'k-', linewidth=2)
    ax2.plot(eye_right_x, eye_right_y, 'k-', linewidth=2)
    ax2.plot(sad_mouth_x, sad_mouth_y, 'k-', linewidth=2)
    
    # Add negative correlation contour
    cov_neg = np.array([[1.0, -0.8], [-0.8, 1.0]])
    
    rv_neg = np.random.multivariate_normal(mean, cov_neg, 100)
    ax2.scatter(rv_neg[:, 0], rv_neg[:, 1], color='red', alpha=0.3, s=10)
    
    # Add negative covariance ellipse
    eigenvalues, eigenvectors = np.linalg.eig(cov_neg)
    ell_neg = Ellipse(xy=(0, 0),
                    width=4*np.sqrt(eigenvalues[0]), 
                    height=4*np.sqrt(eigenvalues[1]),
                    angle=np.rad2deg(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])),
                    edgecolor='red', facecolor='none', linestyle='--')
    ax2.add_patch(ell_neg)
    
    ax2.set_title('Negative Correlation: Sad Data! 😢\nAs one variable increases, the other decreases')
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    ax2.set_aspect('equal')
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

def sketch_contour_problem():
    """Create an interactive visualization for sketching contours of bivariate normal distributions."""
    # Create figure with a grid layout
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1])
    
    # Main plot area for contours
    ax_contour = fig.add_subplot(gs[0, 0])
    # Mathematical formula area
    ax_formula = fig.add_subplot(gs[0, 1])
    # Sliders area
    ax_sigma1 = fig.add_subplot(gs[1, 0])
    ax_sigma2 = fig.add_subplot(gs[1, 1])
    
    # Turn off axis for formula display
    ax_formula.axis('off')
    
    # Setup the initial plot data
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    # Initial covariance matrix parameters
    sigma1_init = 1.0
    sigma2_init = 1.0
    
    def multivariate_gaussian(pos, mu, Sigma):
        """Return the multivariate Gaussian distribution on array pos."""
        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2*np.pi)**n * Sigma_det)
        
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) for each point
        fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
        
        return np.exp(-fac / 2) / N
    
    # Create initial covariance matrix and mean
    mu = np.array([0., 0.])
    Sigma = np.array([[sigma1_init, 0], [0, sigma2_init]])
    
    # Calculate initial PDF
    Z = multivariate_gaussian(pos, mu, Sigma)
    
    # Create contour plot
    contour_levels = np.linspace(0.01, 0.15, 5)
    contour = ax_contour.contour(X, Y, Z, levels=contour_levels, colors='black')
    ax_contour.clabel(contour, inline=True, fontsize=10)
    
    # Add an ellipse to represent the covariance
    lambda_, v = np.linalg.eig(Sigma)
    lambda_ = np.sqrt(lambda_)
    
    # Create ellipses for 1σ, 2σ, and 3σ
    ellipses = []
    for j in range(1, 4):
        ell = Ellipse(xy=(0, 0),
                     width=lambda_[0]*j*2, height=lambda_[1]*j*2,
                     angle=np.rad2deg(np.arctan2(v[1, 0], v[0, 0])),
                     edgecolor='red', facecolor='none', linestyle='--')
        ax_contour.add_patch(ell)
        ellipses.append(ell)
    
    # Add title and labels
    ax_contour.set_title('Contour Lines for Bivariate Normal Distribution\nwith Diagonal Covariance Matrix')
    ax_contour.set_xlabel('x')
    ax_contour.set_ylabel('y')
    ax_contour.grid(True)
    ax_contour.set_xlim(-3, 3)
    ax_contour.set_ylim(-3, 3)
    ax_contour.set_aspect('equal')
    
    # Display the mathematical formula in a simplified format
    formula_text = ("Bivariate Normal Distribution\n\n" +
                   "f(x,y) = (1/2π√|Σ|) exp(-1/2 (x,y)ᵀ Σ⁻¹ (x,y))\n\n" +
                   "Covariance Matrix Σ:\n" +
                   f"[[{sigma1_init:.1f}, 0]\n [0, {sigma2_init:.1f}]]\n\n" +
                   "Mean μ = (0, 0)")
    ax_formula.text(0.5, 0.5, formula_text, ha='center', va='center', fontsize=12)
    
    # Create sliders
    slider_sigma1 = Slider(ax_sigma1, 'σ₁²', 0.1, 3.0, valinit=sigma1_init)
    slider_sigma2 = Slider(ax_sigma2, 'σ₂²', 0.1, 3.0, valinit=sigma2_init)
    
    # Update function for sliders
    def update(val):
        # Get current slider values
        sigma1 = slider_sigma1.val
        sigma2 = slider_sigma2.val
        
        # Update covariance matrix
        Sigma = np.array([[sigma1, 0], [0, sigma2]])
        
        # Recalculate PDF
        Z = multivariate_gaussian(pos, mu, Sigma)
        
        # Clear previous contours
        for c in ax_contour.collections:
            c.remove()
        
        # Redraw contours
        contour = ax_contour.contour(X, Y, Z, levels=contour_levels, colors='black')
        ax_contour.clabel(contour, inline=True, fontsize=10)
        
        # Update ellipses
        lambda_, v = np.linalg.eig(Sigma)
        lambda_ = np.sqrt(lambda_)
        
        # Remove old ellipses
        for ell in ellipses:
            ell.remove()
        
        # Create new ellipses
        ellipses.clear()
        for j in range(1, 4):
            ell = Ellipse(xy=(0, 0),
                         width=lambda_[0]*j*2, height=lambda_[1]*j*2,
                         angle=np.rad2deg(np.arctan2(v[1, 0], v[0, 0])),
                         edgecolor='red', facecolor='none', linestyle='--')
            ax_contour.add_patch(ell)
            ellipses.append(ell)
        
        # Update formula text
        new_formula_text = ("Bivariate Normal Distribution\n\n" +
                           "f(x,y) = (1/2π√|Σ|) exp(-1/2 (x,y)ᵀ Σ⁻¹ (x,y))\n\n" +
                           "Covariance Matrix Σ:\n" +
                           f"[[{sigma1:.1f}, 0]\n [0, {sigma2:.1f}]]\n\n" +
                           "Mean μ = (0, 0)")
        ax_formula.clear()
        ax_formula.axis('off')
        ax_formula.text(0.5, 0.5, new_formula_text, ha='center', va='center', fontsize=12)
        
        # Redraw
        fig.canvas.draw_idle()
    
    # Connect the sliders to the update function
    slider_sigma1.on_changed(update)
    slider_sigma2.on_changed(update)
    
    plt.tight_layout()
    return fig

def explain_sketch_contour_problem():
    """Print detailed explanations for the interactive sketch contour problem."""
    print(f"\n{'='*80}")
    print(f"Example: Sketch Contour Lines for Bivariate Normal Distribution")
    print(f"{'='*80}")
    
    print("\nProblem Statement:")
    print("Sketch the contour lines for the probability density function of a bivariate normal distribution")
    print("with mean μ = (0,0) and covariance matrix Σ = [[σ₁², 0], [0, σ₂²]].")
    
    print("\nStep-by-Step Solution:")
    
    print("\nStep 1: Understand the mathematical formula")
    print("The PDF of a bivariate normal distribution is given by:")
    print("f(x,y) = (1/2π√|Σ|) * exp(-1/2 * (x,y)ᵀ Σ⁻¹ (x,y))")
    print("where Σ is the covariance matrix and |Σ| is its determinant.")
    
    print("\nStep 2: Analyze the covariance matrix")
    print("For Σ = [[σ₁², 0], [0, σ₂²]]:")
    print("- This is a diagonal matrix with variances σ₁² and σ₂² along the diagonal")
    print("- Zero covariance means the variables are uncorrelated")
    print("- The determinant |Σ| = σ₁² * σ₂²")
    print("- The inverse Σ⁻¹ = [[1/σ₁², 0], [0, 1/σ₂²]]")
    
    print("\nStep 3: Identify the equation for contour lines")
    print("Contour lines connect points with equal probability density")
    print("For a specific contour value c, the points satisfy:")
    print("(x,y)ᵀ Σ⁻¹ (x,y) = -2ln(c*2π√|Σ|) = constant")
    print("Which simplifies to: (x²/σ₁² + y²/σ₂²) = constant")
    
    print("\nStep 4: Recognize that contours form ellipses")
    print("The equation (x²/σ₁² + y²/σ₂²) = constant describes an ellipse:")
    print("- Centered at the origin (0,0)")
    print("- Semi-axes aligned with the coordinate axes")
    print("- Semi-axis lengths proportional to √σ₁² and √σ₂²")
    
    print("\nStep 5: Sketch the contours")
    print("Draw concentric ellipses centered at the origin:")
    print("- If σ₁² = σ₂²: The ellipses become circles (equal spread in all directions)")
    print("- If σ₁² > σ₂²: The ellipses are stretched along the x-axis")
    print("- If σ₁² < σ₂²: The ellipses are stretched along the y-axis")
    
    print("\nConclusion:")
    print("The contour lines are concentric ellipses centered at the mean (0,0).")
    print("The shape of these ellipses directly reflects the covariance structure:")
    print("- The axes of the ellipses align with the coordinate axes when the covariance matrix is diagonal")
    print("- The relative sizes of the semi-axes are determined by the square roots of the variances")
    
    print(f"\n{'='*80}")
    
    return "Sketch contour problem explanation generated successfully!"

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
        {"function": covariance_eigenvalue_visualization, "filename": "covariance_eigenvalue_visualization.png"},
        # New simple examples
        {"function": simple_covariance_example_real_world, "filename": "simple_covariance_real_world.png"},
        {"function": toy_data_covariance_change, "filename": "toy_data_covariance_change.png"},
        {"function": simple_mahalanobis_distance, "filename": "simple_mahalanobis_distance.png"},
        {"function": emoji_covariance_example, "filename": "emoji_covariance_example.png"},
        {"function": sketch_contour_problem, "filename": "sketch_contour_problem.png"}
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