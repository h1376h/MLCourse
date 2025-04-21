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
    # Print detailed step-by-step solution
    print("\n" + "="*80)
    print("Covariance Matrix Contours: Step-by-Step Solution")
    print("="*80)
    
    print("\nStep 1: Define the multivariate Gaussian probability density function")
    print("We'll use the formula:")
    print("f(x,y) = (1/2π√|Σ|) * exp(-1/2 * (x-μ)ᵀΣ⁻¹(x-μ))")
    print("where:")
    print("- (x,y) is the position")
    print("- μ is the mean vector")
    print("- Σ is the covariance matrix")
    print("- |Σ| is the determinant of the covariance matrix")
    print("- Σ⁻¹ is the inverse of the covariance matrix")
    
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
    
    def multivariate_gaussian(pos, mu, Sigma):
        """Return the multivariate Gaussian distribution on array pos."""
        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2*np.pi)**n * Sigma_det)
        
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) for each point
        fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
        
        return np.exp(-fac / 2) / N
    
    print("\nStep 3: Analyze Case 1 - Diagonal Covariance with Equal Variances")
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
    
    print("\nStep 4: Analyze Case 2 - Diagonal Covariance with Different Variances")
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
    
    print("\nStep 5: Analyze Case 3 - Non-Diagonal Covariance with Positive Correlation")
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
    
    print("\nStep 6: Analyze Case 4 - Non-Diagonal Covariance with Negative Correlation")
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
    
    print("\nStep 7: Compare and Analyze All Cases")
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
    
    plt.tight_layout()
    return fig

def basic_2d_example():
    """Simple example showing 1D and 2D normal distributions"""
    # Print detailed step-by-step solution
    print("\n" + "="*80)
    print("Basic 2D Normal Distributions: Step-by-Step Solution")
    print("="*80)
    
    print("\nStep 1: Understanding 1D Normal Distributions with Different Variances")
    print("The probability density function of a 1D normal distribution is:")
    print("f(x) = (1/√(2πσ²)) * exp(-x²/(2σ²))")
    print("where σ² is the variance parameter.")
    print("\nWe'll visualize three cases:")
    print("1. Standard normal (σ² = 1): f(x) = (1/√(2π)) * exp(-x²/2)")
    print("2. Narrow normal (σ² = 0.5): f(x) = (1/√(π)) * exp(-x²/1)")
    print("   - This has a taller peak (larger maximum value)")
    print("   - It decreases more rapidly as x moves away from the mean")
    print("3. Wide normal (σ² = 2): f(x) = (1/√(4π)) * exp(-x²/4)")
    print("   - This has a shorter peak (smaller maximum value)")
    print("   - It decreases more slowly as x moves away from the mean")
    print("\nThe key insight: total area under each curve = 1 (probability axiom)")
    print("So curves with higher peaks must be narrower, and those with lower peaks must be wider")
    
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
    
    print("\nStep 2: Extending to 2D - The Standard Bivariate Normal Distribution")
    print("The PDF of a 2D standard normal distribution (with identity covariance matrix) is:")
    print("f(x,y) = (1/2π) * exp(-(x² + y²)/2)")
    print("\nKey properties:")
    print("- Equal variance in both dimensions (σ₁² = σ₂² = 1)")
    print("- Zero correlation between x and y (ρ = 0)")
    print("- Contours form perfect circles centered at the origin")
    print("- The equation for the contours is x² + y² = constant")
    print("- The contour value c corresponds to the constant: -2ln(2πc)")
    print("- 1σ, 2σ, and 3σ circles have radii of 1, 2, and 3 respectively")
    print("- The 1σ circle contains approximately 39% of the probability mass")
    print("- The 2σ circle contains approximately 86% of the probability mass")
    print("- The 3σ circle contains approximately 99% of the probability mass")
    
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
    
    print("\nStep 3: 2D Normal with Different Variances (Diagonal Covariance Matrix)")
    print("Now we'll examine a bivariate normal where the variances are different:")
    print("f(x,y) = (1/(2π√|Σ|)) * exp(-1/2 * ((x²/σ₁²) + (y²/σ₂²)))")
    print("where σ₁² = 2 and σ₂² = 0.5")
    print("\nKey properties:")
    print("- Covariance matrix Σ = [[2, 0], [0, 0.5]]")
    print("- Determinant |Σ| = 2 * 0.5 = 1")
    print("- Different variances in x and y directions")
    print("- Still zero correlation between variables (ρ = 0)")
    print("- Contours form axis-aligned ellipses")
    print("- The equation for the contours is (x²/2 + y²/0.5) = constant")
    print("- The semi-axes of the ellipses are in the ratio √2 : √0.5 ≈ 1.41 : 0.71")
    print("- The ellipses are stretched along the x-axis and compressed along the y-axis")
    print("- This reflects greater variance in the x direction than in the y direction")
    
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
    
    print("\nStep 4: Comparing the Three Cases")
    print("Key insights from these visualizations:")
    print("1. 1D normal distributions: As variance increases, the peak height decreases")
    print("   and the spread increases, but the total area remains constant (= 1)")
    print("2. 2D standard normal (equal variances): Circular contours indicating")
    print("   equal spread in all directions. This is the simplest case.")
    print("3. 2D normal with different variances: Elliptical contours indicating")
    print("   different spread in different directions. The direction of greater")
    print("   variance corresponds to the longer axis of the ellipse.")
    print("\nThe mathematical relationship: The shape of the contours directly reflects")
    print("the structure of the covariance matrix. In these examples, the variables are")
    print("uncorrelated, so the ellipses are aligned with the coordinate axes.")
    
    plt.tight_layout()
    return fig

def gaussian_3d_visualization():
    """Create 3D visualization of Gaussian probability density functions"""
    # Print detailed step-by-step solution
    print("\n" + "="*80)
    print("3D Visualization of Gaussian PDFs: Step-by-Step Solution")
    print("="*80)
    
    print("\nStep 1: Setting Up the Visualization Framework")
    print("To visualize bivariate normal distributions in 3D, we need to:")
    print("- Create a 2D grid of (x,y) points where we'll evaluate the PDF")
    print("- Calculate the PDF value at each point, giving us a 3D surface z = f(x,y)")
    print("- Plot this surface in 3D space, with contours projected on the xy-plane")
    print("\nThis gives us a comprehensive view of both the probability density surface")
    print("and its contour lines, helping us understand the distribution's shape")
    
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
    
    print("\nStep 2: Case 1 - Standard Bivariate Normal (Identity Covariance)")
    print("For a standard bivariate normal distribution:")
    print("- Mean vector: μ = [0, 0] (centered at the origin)")
    print("- Covariance matrix: Σ = [[1, 0], [0, 1]] (identity matrix)")
    print("- PDF: f(x,y) = (1/2π) * exp(-(x² + y²)/2)")
    print("\nKey properties of the 3D surface:")
    print("- The peak occurs at (0,0) with a value of 1/(2π) ≈ 0.159")
    print("- The surface has perfect radial symmetry around the z-axis")
    print("- The contours projected onto the xy-plane form perfect circles")
    print("- The surface falls off equally in all directions from the peak")
    print("- The volume under the entire surface equals 1 (probability axiom)")
    
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
    
    print("\nStep 3: Case 2 - Bivariate Normal with Different Variances")
    print("For a bivariate normal with different variances:")
    print("- Mean vector: μ = [0, 0] (still centered at the origin)")
    print("- Covariance matrix: Σ = [[2.0, 0], [0, 0.5]] (diagonal but unequal)")
    print("- PDF: f(x,y) = (1/(2π√|Σ|)) * exp(-1/2 * (x²/2 + y²/0.5))")
    print("- Determinant |Σ| = 2.0 * 0.5 = 1.0")
    print("\nKey properties of the 3D surface:")
    print("- The peak still occurs at (0,0) with the same height as Case 1")
    print("- The surface is stretched along the x-axis and compressed along the y-axis")
    print("- The contours projected onto the xy-plane form axis-aligned ellipses")
    print("- The surface falls off more slowly in the x-direction (larger variance)")
    print("- The surface falls off more quickly in the y-direction (smaller variance)")
    print("- The volume under the surface still equals 1")
    
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
    
    print("\nStep 4: Case 3 - Bivariate Normal with Correlation")
    print("For a bivariate normal with correlation:")
    print("- Mean vector: μ = [0, 0]")
    print("- Covariance matrix: Σ = [[1.0, 0.8], [0.8, 1.0]] (non-diagonal)")
    corr = 0.8 / np.sqrt(1.0 * 1.0)
    print(f"- Correlation coefficient: ρ = {corr:.2f} (strong positive correlation)")
    print("- PDF: f(x,y) = (1/(2π√|Σ|)) * exp(-1/2 * (x,y)ᵀ Σ⁻¹ (x,y))")
    print("- Determinant |Σ| = 1.0² - 0.8² = 0.36")
    print("\nKey properties of the 3D surface:")
    print("- The peak still occurs at (0,0), but its height is different due to the determinant")
    print("- The surface is tilted, with its principal axes rotated from the coordinate axes")
    print("- The contours projected onto the xy-plane form rotated ellipses")
    print("- The primary direction of spread is along the y = x line (reflecting positive correlation)")
    print("- The surface shows that x and y tend to increase or decrease together")
    print("- The correlation creates a 'ridge' along the y = x direction")
    print("- The volume under the surface still equals 1")
    
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
    
    print("\nStep 5: Comparing All Three 3D Visualizations")
    print("Key insights from comparing these 3D surfaces:")
    print("1. The covariance matrix directly determines the shape and orientation of the PDF surface")
    print("2. Identity covariance (Case 1): Symmetric bell shape with circular contours")
    print("3. Diagonal covariance with different variances (Case 2): Stretched bell shape")
    print("   with axis-aligned elliptical contours")
    print("4. Non-diagonal covariance with correlation (Case 3): Tilted bell shape")
    print("   with rotated elliptical contours")
    print("\nMathematical relationships:")
    print("- The exponent term in the PDF formula: -1/2 * (x,y)ᵀ Σ⁻¹ (x,y) creates the shape")
    print("- The determinant term in the denominator: √|Σ| adjusts the height of the peak")
    print("- Together they ensure that the volume under the surface equals 1")
    
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
    print("\n\n" + "*"*80)
    print("RUNNING COVARIANCE MATRIX CONTOUR EXAMPLES WITH DETAILED STEP-BY-STEP SOLUTIONS")
    print("*"*80)
    
    # Run examples with detailed step-by-step printing
    print("\nRunning Example 1: Covariance Matrix Contours")
    fig1 = covariance_matrix_contours()
    
    print("\nRunning Example 2: Basic 2D Normal Distributions")
    fig2 = basic_2d_example()
    
    print("\nRunning Example 3: 3D Visualization of Gaussian PDFs")
    fig3 = gaussian_3d_visualization()
    
    print("\nRunning Example 4: Eigenvalue and Eigenvector Visualization")
    fig4 = covariance_eigenvalue_visualization()
    
    print("\nRunning Example 5: Real-World Height-Weight Covariance")
    fig5 = simple_covariance_example_real_world()
    
    print("\nRunning Example 6: Rotation and Covariance Change")
    fig6 = toy_data_covariance_change()
    
    print("\nRunning Example 7: Mahalanobis Distance vs Euclidean Distance")
    fig7 = simple_mahalanobis_distance()
    
    print("\nRunning Example 8: Emoji Covariance Example")
    fig8 = emoji_covariance_example()
    
    print("\nRunning Example 9: Sketch Contour Problem")
    fig9 = sketch_contour_problem()
    
    # Generate plots (optional)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images", "Contour_Plots")
    ensure_directory_exists(images_dir)
    
    # Save figures
    examples = [
        {"fig": fig1, "filename": "covariance_matrix_contours.png"},
        {"fig": fig2, "filename": "basic_2d_normal_examples.png"},
        {"fig": fig3, "filename": "gaussian_3d_visualization.png"},
        {"fig": fig4, "filename": "covariance_eigenvalue_visualization.png"},
        {"fig": fig5, "filename": "simple_covariance_real_world.png"},
        {"fig": fig6, "filename": "toy_data_covariance_change.png"},
        {"fig": fig7, "filename": "simple_mahalanobis_distance.png"},
        {"fig": fig8, "filename": "emoji_covariance_example.png"},
        {"fig": fig9, "filename": "sketch_contour_problem.png"}
    ]
    
    for example in examples:
        try:
            save_path = os.path.join(images_dir, example["filename"])
            example["fig"].savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Generated and saved {example['filename']}")
        except Exception as e:
            print(f"Error generating {example['filename']}: {e}")
    
    print("\nAll examples completed successfully!") 