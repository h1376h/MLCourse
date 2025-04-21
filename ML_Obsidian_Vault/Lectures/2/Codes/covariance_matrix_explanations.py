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

if __name__ == "__main__":
    explain_covariance_contours() 