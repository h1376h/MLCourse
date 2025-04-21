import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Ellipse

def ensure_directory_exists(directory):
    """Ensure the specified directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) for each point
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    
    return np.exp(-fac / 2) / N

def covariance_matrix_contours():
    """Visualize multivariate Gaussians with different covariance matrices"""
    # Print detailed step-by-step solution
    print("\n" + "="*80)
    print("Example: Multivariate Gaussians with Different Covariance Matrices")
    print("="*80)
    
    print(f"Function: f(x,y) = (1/√(2π|Σ|)) * exp(-1/2 * [(x,y)ᵀ Σ⁻¹ (x,y)])")
    
    print("\nStep-by-Step Solution:")
    
    print("\nStep 1: Understand the multivariate Gaussian PDF")
    print("The probability density function of a bivariate Gaussian with mean μ = (0,0) and")
    print("covariance matrix Σ defines a surface whose contours we want to analyze.")
    print("For a bivariate Gaussian:")
    print("f(x,y) = (1/√(2π|Σ|)) * exp(-1/2 * [(x,y)ᵀ Σ⁻¹ (x,y)])")
    print("where Σ is the covariance matrix and |Σ| is its determinant.")
    
    print("\nStep 2: Create a grid of points for visualization")
    print("We'll create a 100x100 grid spanning from -5 to 5 in both dimensions")
    print("This gives us a total of 10,000 points at which to evaluate the PDF")
    print("The goal is to visualize the shape of the probability density function")
    print("by creating contour plots that connect points of equal density")
    
    # Create a figure with 2x2 subplots
    fig = plt.figure(figsize=(15, 15))
    
    # Create a grid of points
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    print("\nStep 3: Analyze the quadratic form in the exponent")
    print("The key term that determines the shape of the contours is the quadratic form")
    print("(x,y)ᵀ Σ⁻¹ (x,y), which creates elliptical level curves.")
    print("The exponent term -1/2 * [(x,y)ᵀ Σ⁻¹ (x,y)] determines the contour shapes.")
    print("For constant density c, the contours satisfy:")
    print("(x,y)ᵀ Σ⁻¹ (x,y) = -2log(c·√(2π|Σ|)) = constant")
    
    print("\nStep 4: Analyze Case 1 - Diagonal Covariance with Equal Variances")
    print("Covariance Matrix Σ = [[1.0, 0.0], [0.0, 1.0]] (Identity Matrix)")
    print("Properties:")
    print("- Equal variances (σ₁² = σ₂² = 1)")
    print("- Zero correlation (ρ = 0)")
    print("- Determinant |Σ| = 1")
    print("- Eigenvalues: λ₁ = λ₂ = 1")
    print("- The resulting contours form perfect circles")
    print("- The equation for these contours is x² + y² = constant")
    print("- This is the standard bivariate normal distribution")
    print("- The pdf simplifies to: f(x,y) = (1/2π) * exp(-(x² + y²)/2)")
    
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
    
    ax1.set_title('Case 1: Circular Contours\nIdentity Covariance Matrix')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True)
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)
    
    print("\nStep 5: Analyze Case 2 - Diagonal Covariance with Different Variances")
    print("Covariance Matrix Σ = [[3.0, 0.0], [0.0, 0.5]]")
    print("Properties:")
    print("- Different variances (σ₁² = 3, σ₂² = 0.5)")
    print("- Zero correlation (ρ = 0)")
    print("- Determinant |Σ| = 1.5")
    print("- Eigenvalues: λ₁ = 3, λ₂ = 0.5 (same as variances since matrix is diagonal)")
    print("- The resulting contours form axis-aligned ellipses")
    print("- The equation for these contours is x²/3 + y²/0.5 = constant")
    print("- The ellipses are stretched along the x-axis and compressed along the y-axis")
    print("- The pdf is: f(x,y) = (1/2π√1.5) * exp(-1/2 * (x²/3 + y²/0.5))")
    print("- The semi-axes of the ellipses are in the ratio √3 : √0.5 ≈ 1.73 : 0.71")
    
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
    
    ax2.set_title('Case 2: Axis-Aligned Elliptical Contours\nDiagonal Covariance Matrix')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.grid(True)
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-5, 5)
    
    print("\nStep 6: Analyze Case 3 - Non-Diagonal Covariance with Positive Correlation")
    print("Covariance Matrix Σ = [[2.0, 1.5], [1.5, 2.0]]")
    print("Properties:")
    corr3 = 1.5 / np.sqrt(2.0 * 2.0)
    print(f"- Equal variances (σ₁² = σ₂² = 2)")
    print(f"- Positive correlation (ρ = {corr3:.2f})")
    print("- Determinant |Σ| = 1.75")
    
    lambda_3, v3 = np.linalg.eig(np.array([[2.0, 1.5], [1.5, 2.0]]))
    print(f"- Eigenvalues: λ₁ = {lambda_3[0]:.2f}, λ₂ = {lambda_3[1]:.2f}")
    print(f"- Eigenvectors: v₁ = [{v3[0,0]:.2f}, {v3[1,0]:.2f}], v₂ = [{v3[0,1]:.2f}, {v3[1,1]:.2f}]")
    print("- The resulting contours form rotated ellipses")
    print("- The ellipses are tilted along the y = x direction (positive correlation)")
    print("- The principal axes align with the eigenvectors of the covariance matrix")
    print("- The semi-axes lengths are proportional to √3.5 and √0.5")
    print("- The quadratic form in the exponent is:")
    print("  (x,y)ᵀ Σ⁻¹ (x,y) = [x y] [[a b], [b c]] [x, y]ᵀ = a·x² + 2b·xy + c·y²")
    print("  where Σ⁻¹ = [[a b], [b c]] is the inverse of the covariance matrix")
    
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
    
    ax3.set_title('Case 3: Rotated Elliptical Contours\nPositive Correlation (ρ = 0.75)')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.grid(True)
    ax3.set_xlim(-5, 5)
    ax3.set_ylim(-5, 5)
    
    print("\nStep 7: Analyze Case 4 - Non-Diagonal Covariance with Negative Correlation")
    print("Covariance Matrix Σ = [[2.0, -1.5], [-1.5, 2.0]]")
    print("Properties:")
    corr4 = -1.5 / np.sqrt(2.0 * 2.0)
    print(f"- Equal variances (σ₁² = σ₂² = 2)")
    print(f"- Negative correlation (ρ = {corr4:.2f})")
    print("- Determinant |Σ| = 1.75")
    
    lambda_4, v4 = np.linalg.eig(np.array([[2.0, -1.5], [-1.5, 2.0]]))
    print(f"- Eigenvalues: λ₁ = {lambda_4[0]:.2f}, λ₂ = {lambda_4[1]:.2f}")
    print(f"- Eigenvectors: v₁ = [{v4[0,0]:.2f}, {v4[1,0]:.2f}], v₂ = [{v4[0,1]:.2f}, {v4[1,1]:.2f}]")
    print("- The resulting contours form rotated ellipses")
    print("- The ellipses are tilted along the y = -x direction (negative correlation)")
    print("- The principal axes align with the eigenvectors of the covariance matrix")
    print("- The semi-axes lengths are proportional to √3.5 and √0.5")
    print("- The negative correlation means that as one variable increases,")
    print("  the other tends to decrease, creating the rotation in the opposite direction")
    
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
    
    ax4.set_title('Case 4: Rotated Elliptical Contours\nNegative Correlation (ρ = -0.75)')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.grid(True)
    ax4.set_xlim(-5, 5)
    ax4.set_ylim(-5, 5)
    
    print("\nStep 8: Compare and Analyze All Cases")
    print("Key Insights:")
    print("1. Diagonal covariance matrices produce axis-aligned ellipses or circles:")
    print("   - Equal variances (Case 1): Perfect circles")
    print("   - Different variances (Case 2): Axis-aligned ellipses")
    print("2. Non-diagonal covariance matrices produce rotated ellipses:")
    print("   - Positive correlation (Case 3): Ellipses tilted along y = x")
    print("   - Negative correlation (Case 4): Ellipses tilted along y = -x")
    print("3. The shape and orientation of the ellipses directly reflect the covariance structure:")
    print("   - The principal axes of the ellipses align with the eigenvectors of the covariance matrix")
    print("   - The length of each principal axis is proportional to the square root of the corresponding eigenvalue")
    print("4. The density contours connect points of equal probability density")
    print("5. Mathematical relationship between correlation and geometry:")
    print("   - As correlation increases in magnitude, ellipses become more elongated")
    print("   - The angle of the principal axis is tan⁻¹(ρσ₂/σ₁) for positive correlation")
    print("   - The eccentricity of the ellipses increases with stronger correlation")
    
    print("\nPractical Applications:")
    print("- Visualizing multivariate probability distributions")
    print("- Understanding correlation structure in data")
    print("- Analyzing principal components and directions of maximum variance")
    print("- Designing confidence regions for statistical inference")
    print("- Implementing anomaly detection based on Mahalanobis distance")
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("\n\n" + "*"*80)
    print("EXAMPLE 1: COVARIANCE MATRIX CONTOURS")
    print("*"*80)
    
    # Run the example with detailed step-by-step printing
    fig = covariance_matrix_contours()
    
    # Save the figure if needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images", "Contour_Plots")
    ensure_directory_exists(images_dir)
    
    try:
        save_path = os.path.join(images_dir, "covariance_matrix_contours.png")
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\nFigure saved to: {save_path}")
    except Exception as e:
        print(f"\nError saving figure: {e}")
    