import numpy as np
import matplotlib.pyplot as plt
import os

def ensure_directory_exists(directory):
    """Ensure the specified directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def explain_covariance_contours():
    """Print detailed explanations for the covariance matrix contours example."""
    print(f"\n{'='*80}")
    print(f"Example: Multivariate Gaussians with Different Covariance Matrices")
    print(f"{'='*80}")
    
    print(f"Function: f(x,y) = (1/√(2π|Σ|)) * exp(-1/2 * [(x,y)ᵀ Σ⁻¹ (x,y)])")
    
    print("\nStep-by-Step Solution:")
    
    steps = [
        {
            "title": "Understand the multivariate Gaussian PDF",
            "description": "The probability density function of a bivariate Gaussian with mean μ = (0,0) and covariance matrix Σ defines a surface whose contours we want to analyze.",
            "math": "For a bivariate Gaussian:\nf(x,y) = (1/√(2π|Σ|)) * exp(-1/2 * [(x,y)ᵀ Σ⁻¹ (x,y)])\nwhere Σ is the covariance matrix and |Σ| is its determinant."
        },
        {
            "title": "Analyze the quadratic form in the exponent",
            "description": "The key term that determines the shape of the contours is the quadratic form (x,y)ᵀ Σ⁻¹ (x,y), which creates elliptical level curves.",
            "math": "The exponent term -1/2 * [(x,y)ᵀ Σ⁻¹ (x,y)] determines the contour shapes.\nFor constant density c, the contours satisfy:\n(x,y)ᵀ Σ⁻¹ (x,y) = -2log(c·√(2π|Σ|)) = constant"
        },
        {
            "title": "Consider different types of covariance matrices",
            "description": "Different covariance matrices lead to different shaped contours, which can be analyzed based on the eigenvalues and eigenvectors of Σ.",
            "math": "Case 1: Diagonal covariance Σ = [[σ₁², 0], [0, σ₂²]]\n  When σ₁² = σ₂² (scaled identity): Circular contours\n  When σ₁² ≠ σ₂² (different variances): Axis-aligned ellipses\n\nCase 2: Non-diagonal covariance Σ = [[σ₁², ρσ₁σ₂], [ρσ₁σ₂, σ₂²]]\n  When ρ ≠ 0: Rotated ellipses\n  When ρ > 0: Ellipses tilted along y = x direction\n  When ρ < 0: Ellipses tilted along y = -x direction"
        },
        {
            "title": "Draw the contours for each case",
            "description": "For each type of covariance matrix, we can sketch the resulting contours based on our analysis.",
            "math": "For diagonal covariance: The principal axes of the elliptical contours align with coordinate axes.\n  - The lengths of the semi-axes are proportional to √σ₁² and √σ₂².\n\nFor non-diagonal covariance: The principal axes of the elliptical contours align with the eigenvectors of Σ.\n  - The lengths of the semi-axes are proportional to the square roots of the eigenvalues.\n  - The correlation coefficient ρ determines the rotation angle of the ellipses."
        }
    ]
    
    for i, step in enumerate(steps, 1):
        print(f"\nStep {i}: {step['title']}")
        print(f"{step['description']}")
        for line in step['math'].split('\n'):
            print(f"  {line}")
    
    print("\nContour Values:")
    print("c = varies based on probability density")
    
    print("\nKey Insights:")
    print("- The contour lines connect all points (x,y) where f(x,y) equals the contour value c.")
    print("- Each contour shape provides insight into the function's behavior in that region.")
    print("- This function produces elliptical contours, which indicate multivariate Gaussian distributions.")
    print("- The shape and orientation of the ellipses directly reflect the covariance structure.")
    
    print("\nCovariance Matrix Effects on Contour Shapes:")
    print("1. Diagonal covariance matrices:")
    print("   - Equal variances (σ₁² = σ₂²): Circular contours")
    print("   - Different variances (σ₁² ≠ σ₂²): Axis-aligned elliptical contours")
    print("   - Major/minor axes proportional to the square roots of the variances")
    
    print("\n2. Non-diagonal covariance matrices:")
    print("   - Produce rotated elliptical contours not aligned with coordinate axes")
    print("   - Positive correlation (ρ > 0): Ellipses tilted along y = x direction")
    print("   - Negative correlation (ρ < 0): Ellipses tilted along y = -x direction")
    print("   - The principal axes align with the eigenvectors of the covariance matrix")
    print("   - The lengths of these axes are proportional to the square roots of the eigenvalues")
    
    print(f"\n{'='*80}")
    
    print("\nPractical Applications:")
    print("- Visualizing multivariate probability distributions")
    print("- Understanding correlation structure in data")
    print("- Analyzing principal components and directions of maximum variance")
    print("- Designing confidence regions for statistical inference")
    print("- Implementing anomaly detection based on Mahalanobis distance")
    
    return "Covariance matrix contour explanations generated successfully!"

def explain_basic_2d_example():
    """Print detailed explanations for the basic 2D normal distribution examples."""
    print(f"\n{'='*80}")
    print(f"Example: Basic 1D and 2D Normal Distributions")
    print(f"{'='*80}")
    
    print("\nExample 1: 1D Normal Distributions with Different Variances")
    print("\nMathematical Formula:")
    print("f(x) = (1/√(2πσ²)) * exp(-x²/(2σ²))")
    
    print("\nKey Points:")
    print("- Standard normal distribution (σ² = 1): Balanced trade-off between peak height and spread")
    print("- Smaller variance (σ² = 0.5): Taller peak, narrower spread (more concentrated)")
    print("- Larger variance (σ² = 2): Lower peak, wider spread (more dispersed)")
    print("- The total area under each curve is always 1 (probability axioms)")
    print("- Standard deviation (σ) quantifies the typical distance from the mean")
    
    print("\nExample 2: 2D Standard Normal Distribution (Independent Variables)")
    print("\nMathematical Formula:")
    print("f(x,y) = (1/(2π)) * exp(-(x² + y²)/2)")
    
    print("\nKey Points:")
    print("- Identity covariance matrix: Σ = [[1, 0], [0, 1]]")
    print("- Variables x and y are independent (zero correlation)")
    print("- Equal variances in both dimensions lead to circular contours")
    print("- Distance from origin determines probability density (radial symmetry)")
    print("- 1σ, 2σ, and 3σ circles contain approximately 39%, 86%, and 99% of the probability mass")
    
    print("\nExample 3: 2D Normal with Different Variances (Independent Variables)")
    print("\nMathematical Formula:")
    print("f(x,y) = (1/(2π√(σ₁²σ₂²))) * exp(-1/2 * (x²/σ₁² + y²/σ₂²))")
    
    print("\nKey Points:")
    print("- Diagonal covariance matrix: Σ = [[σ₁², 0], [0, σ₂²]]")
    print("- Variables x and y are still independent (zero correlation)")
    print("- Different variances lead to axis-aligned elliptical contours")
    print("- The semi-axes of the ellipses are proportional to σ₁ and σ₂")
    print("- The shape of the ellipse directly visualizes the relative scaling between variables")
    
    print(f"\n{'='*80}")

def explain_3d_visualization():
    """Print detailed explanations for the 3D Gaussian visualization examples."""
    print(f"\n{'='*80}")
    print(f"Example: 3D Visualization of Multivariate Gaussians")
    print(f"{'='*80}")
    
    print("\nExample 1: Standard Bivariate Normal (Identity Covariance)")
    print("\nMathematical Formula:")
    print("f(x,y) = (1/(2π)) * exp(-(x² + y²)/2)")
    
    print("\nKey Points:")
    print("- Perfect bell-shaped surface with radial symmetry")
    print("- Peak at the mean (0,0) with probability density of 1/(2π) ≈ 0.159")
    print("- Circular contours when projected onto the x-y plane")
    print("- Symmetric decay in all directions from the peak")
    print("- The volume under the surface is exactly 1 (probability axioms)")
    
    print("\nExample 2: Bivariate Normal with Different Variances (Diagonal Covariance)")
    print("\nMathematical Formula:")
    print("f(x,y) = (1/(2π√(σ₁²σ₂²))) * exp(-1/2 * (x²/σ₁² + y²/σ₂²))")
    
    print("\nKey Points:")
    print("- Bell-shaped surface stretched along one axis and compressed along the other")
    print("- Elliptical contours when projected onto the x-y plane")
    print("- Peak height is reduced compared to standard case due to normalization")
    print("- The spread reflects the different uncertainties in each dimension")
    print("- The principal axes of the ellipses align with the coordinate axes")
    
    print("\nExample 3: Bivariate Normal with Correlation (Non-Diagonal Covariance)")
    print("\nMathematical Formula:")
    print("f(x,y) = (1/(2π√|Σ|)) * exp(-1/2 * [(x,y)ᵀ Σ⁻¹ (x,y)])")
    print("where Σ = [[σ₁², ρσ₁σ₂], [ρσ₁σ₂, σ₂²]]")
    
    print("\nKey Points:")
    print("- Bell-shaped surface tilted according to the correlation structure")
    print("- Rotated elliptical contours when projected onto the x-y plane")
    print("- The tilt direction indicates the type of correlation (positive/negative)")
    print("- The strength of correlation affects the eccentricity of the ellipses")
    print("- This visualization shows how correlated variables cluster along a specific direction")
    
    print(f"\n{'='*80}")

def explain_eigenvalue_visualization():
    """Print detailed explanations for the eigenvalue and eigenvector visualization."""
    print(f"\n{'='*80}")
    print(f"Example: Eigenvalues, Eigenvectors, and Covariance Effects")
    print(f"{'='*80}")
    
    print("\nMathematical Background:")
    print("- Covariance matrix Σ can be decomposed as Σ = VΛV^T")
    print("- V contains eigenvectors (principal directions)")
    print("- Λ is a diagonal matrix of eigenvalues (variance along principal directions)")
    
    print("\nExample 1: No Correlation (ρ = 0)")
    print("\nKey Points:")
    print("- Covariance matrix: Σ = [[1, 0], [0, 1]]")
    print("- Eigenvalues: λ₁ = λ₂ = 1")
    print("- Eigenvectors align with the coordinate axes")
    print("- Circular contours indicate equal variance in all directions")
    print("- No preferred direction of variability in the data")
    
    print("\nExample 2: Weak Correlation (ρ = 0.3)")
    print("\nKey Points:")
    print("- Covariance matrix: Σ = [[1, 0.3], [0.3, 1]]")
    print("- Eigenvalues: λ₁ ≈ 1.3, λ₂ ≈ 0.7")
    print("- Eigenvectors begin to rotate from the coordinate axes")
    print("- Slightly elliptical contours with mild rotation")
    print("- Beginning of a preferred direction of variability")
    
    print("\nExample 3: Moderate Correlation (ρ = 0.6)")
    print("\nKey Points:")
    print("- Covariance matrix: Σ = [[1, 0.6], [0.6, 1]]")
    print("- Eigenvalues: λ₁ ≈ 1.6, λ₂ ≈ 0.4")
    print("- Eigenvectors rotate further from the coordinate axes")
    print("- More eccentric elliptical contours with significant rotation")
    print("- Clear preferred direction of variability emerges")
    
    print("\nExample 4: Strong Correlation (ρ = 0.9)")
    print("\nKey Points:")
    print("- Covariance matrix: Σ = [[1, 0.9], [0.9, 1]]")
    print("- Eigenvalues: λ₁ ≈ 1.9, λ₂ ≈ 0.1")
    print("- Eigenvectors nearly align with the y = x and y = -x directions")
    print("- Highly eccentric elliptical contours with strong rotation")
    print("- Dominant direction of variability along the first eigenvector")
    print("- Very little variability along the second eigenvector")
    
    print("\nKey Insights:")
    print("- As correlation increases, eigenvalues become more disparate")
    print("- The largest eigenvalue increases, the smallest decreases")
    print("- The orientation of eigenvectors approaches y = x (for positive correlation)")
    print("- The ellipses become increasingly elongated (higher eccentricity)")
    print("- This illustrates why PCA works: it identifies the directions of maximum variance")
    
    print(f"\n{'='*80}")

def generate_covariance_explanations():
    """Generate detailed explanations for all covariance examples."""
    # Print explanations for all examples
    explain_covariance_contours()
    explain_basic_2d_example()
    explain_3d_visualization()
    explain_eigenvalue_visualization()
    
    return "Covariance matrix explanations generated successfully!"

if __name__ == "__main__":
    generate_covariance_explanations() 