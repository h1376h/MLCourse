import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from matplotlib.patches import Ellipse

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

def generate_covariance_contour_plots():
    """Generate and save covariance matrix contour plots"""
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images", "Contour_Plots")
    ensure_directory_exists(images_dir)
    
    # Generate and save example
    try:
        fig = covariance_matrix_contours()
        save_path = os.path.join(images_dir, "covariance_matrix_contours.png")
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Generated covariance matrix visualization")
        print(f"Saved to: {save_path}")
    except Exception as e:
        print(f"Error generating covariance matrix plots: {e}")
    
    return "Covariance matrix contour plots generated successfully!"

if __name__ == "__main__":
    generate_covariance_contour_plots() 