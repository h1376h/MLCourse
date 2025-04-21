import numpy as np
import matplotlib.pyplot as plt
import os

def ensure_directory_exists(directory):
    """Ensure the specified directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def explain_sketch_contour_problem():
    """Print detailed explanations for the sketch contour problem example."""
    print(f"\n{'='*80}")
    print(f"Example: Sketch Contour Lines for Bivariate Normal Distribution")
    print(f"{'='*80}")
    
    print("\nProblem Statement:")
    print("Sketch the contour lines for the probability density function of a bivariate normal distribution")
    print("with mean μ = (0,0) and covariance matrix Σ = [[1, 0], [0, 1]].")
    
    print("\nStep-by-Step Solution:")
    
    print("\nStep 1: Understand the mathematical formula")
    print("The PDF of a bivariate normal distribution is given by:")
    print("f(x,y) = (1/2π√|Σ|) * exp(-1/2 * (x,y)ᵀ Σ⁻¹ (x,y))")
    print("where Σ is the covariance matrix and |Σ| is its determinant.")
    
    print("\nStep 2: Analyze the covariance matrix")
    print("For Σ = [[1, 0], [0, 1]]:")
    print("- This is the identity matrix (variances = 1, covariance = 0)")
    print("- The variables are uncorrelated and have equal variances")
    print("- The determinant |Σ| = 1")
    print("- The inverse Σ⁻¹ = Σ (identity matrix is its own inverse)")
    
    print("\nStep 3: Identify the equation for contour lines")
    print("Contour lines connect points with equal probability density")
    print("For a specific contour value c, the points satisfy:")
    print("(x,y)ᵀ Σ⁻¹ (x,y) = -2ln(c*2π) = constant")
    print("Which simplifies to: x² + y² = constant")
    
    print("\nStep 4: Recognize that contours form circles")
    print("The equation x² + y² = constant describes a circle:")
    print("- Centered at the origin (0,0)")
    print("- With radius r = √constant")
    
    print("\nStep 5: Sketch the contours")
    print("Draw concentric circles centered at the origin with various radii.")
    print("These circles represent different probability density levels:")
    print("- Inner circles (smaller radii): higher probability density")
    print("- Outer circles (larger radii): lower probability density")
    print("- The 1σ circle has radius 1")
    print("- The 2σ circle has radius 2")
    print("- The 3σ circle has radius 3")
    
    print("\nConclusion:")
    print("For a standard bivariate normal distribution (Σ = I), the contour lines form")
    print("concentric circles centered at the mean (0,0), reflecting equal spread in all directions.")
    print("This is the simplest case of a multivariate normal distribution, where the")
    print("variables are uncorrelated and have equal variances.")
    
    print(f"\n{'='*80}")
    
    return "Sketch contour problem explanation generated successfully!"

def generate_covariance_explanations():
    """Generate detailed explanations for the sketch contour problem."""
    # Print explanations for the sketch contour problem
    explain_sketch_contour_problem()
    
    return "Sketch contour problem explanation generated successfully!"

if __name__ == "__main__":
    generate_covariance_explanations() 