import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
import os

def create_output_dir():
    """Create output directory for images"""
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the Lectures/2 directory
    parent_dir = os.path.dirname(current_dir)
    # Use Images/Multivariate_Density_Examples relative to the parent directory
    output_dir = os.path.join(parent_dir, "Images", "Multivariate_Density_Examples")

    # Make sure images directory exists
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def print_formula_box(formula_text):
    """Print a formula or explanation in a visually distinct box"""
    width = max(len(line) for line in formula_text.split('\n')) + 4
    print("\n" + "=" * width)
    for line in formula_text.split('\n'):
        print("| " + line.ljust(width - 4) + " |")
    print("=" * width + "\n")

def example1_bivariate_normal():
    """
    Example 1: Bivariate Normal Density Function
    Mean vector: μ = (2, 3)
    Covariance matrix: Σ = [[4, 2], [2, 5]]
    """
    print("\n======== EXAMPLE 1: BIVARIATE NORMAL DENSITY FUNCTION ========")
    output_dir = create_output_dir()
    
    # Print the general formula for multivariate normal distribution
    general_formula = """Multivariate Normal Density Function Formula:

f(x) = (1/((2π)^(p/2) |Σ|^(1/2))) * exp(-1/2 * (x-μ)^T Σ^(-1) (x-μ))

Where:
- f(x) is the probability density at point x
- p is the dimension (number of variables)
- μ is the mean vector
- Σ is the covariance matrix
- |Σ| is the determinant of the covariance matrix
- Σ^(-1) is the inverse of the covariance matrix
- (x-μ)^T is the transpose of (x-μ)"""
    print_formula_box(general_formula)
    
    # Parameters from the example
    mean = np.array([2, 3])
    cov = np.array([[4, 2], [2, 5]])
    
    # Detailed explanation of the problem
    problem_statement = f"""Problem Statement:
Let X = (X₁, X₂) follow a bivariate normal distribution with:
- Mean vector: μ = ({mean[0]}, {mean[1]})
- Covariance matrix: Σ = [
    [{cov[0,0]}, {cov[0,1]}],
    [{cov[1,0]}, {cov[1,1]}]
]

We need to:
a) Find the probability density function (PDF) of X
b) Calculate P(X₁ ≤ 3, X₂ ≤ 4)
c) Find the conditional distribution of X₁ given X₂ = 4"""
    print_formula_box(problem_statement)
    
    print("PART A: Finding the Probability Density Function (PDF)")
    print("====================================================")
    
    print("\nDetailed, step-by-step solution:")
    
    # Calculate the determinant
    det_cov = np.linalg.det(cov)
    print("\nStep 1: Calculate the determinant of the covariance matrix")
    print("----------------------------------------------------------")
    print(f"det(Σ) = det([[{cov[0,0]}, {cov[0,1]}], [{cov[1,0]}, {cov[1,1]}]])")
    print(f"      = ({cov[0,0]} × {cov[1,1]}) - ({cov[0,1]} × {cov[1,0]})")
    print(f"      = {cov[0,0] * cov[1,1]} - {cov[0,1] * cov[1,0]}")
    print(f"      = {det_cov}")
    
    # Calculate the inverse
    inv_cov = np.linalg.inv(cov)
    
    print("\nStep 2: Calculate the inverse of the covariance matrix")
    print("------------------------------------------------------")
    print("For a 2×2 matrix, the inverse formula is:")
    print("        1     |  d  -b |")
    print("A^(-1) = --- × | -c   a |")
    print("       det(A)  |        |")
    
    print(f"\nApplying this to our covariance matrix:")
    print(f"Σ^(-1) = (1/{det_cov}) × [[{cov[1,1]}, -{cov[0,1]}], [−{cov[1,0]}, {cov[0,0]}]]")
    print(f"       = (1/{det_cov}) × [[{cov[1,1]}, -{cov[0,1]}], [−{cov[1,0]}, {cov[0,0]}]]")
    print(f"       = [[{inv_cov[0,0]:.6f}, {inv_cov[0,1]:.6f}], [{inv_cov[1,0]:.6f}, {inv_cov[1,1]:.6f}]]")
    
    # Step 3: Substitute into the PDF formula
    print("\nStep 3: Substitute into the multivariate normal PDF formula")
    print("----------------------------------------------------------")
    pdf_formula = f"""For bivariate normal (p=2):
f(x₁,x₂) = (1/(2π√|Σ|)) * exp(-1/2 * [(x₁-μ₁), (x₂-μ₂)] × Σ^(-1) × [(x₁-μ₁), (x₂-μ₂)]^T)

Substituting our values:
f(x₁,x₂) = (1/(2π√{det_cov})) * exp(-1/2 * [(x₁-{mean[0]}), (x₂-{mean[1]})] × 
           [[{inv_cov[0,0]:.4f}, {inv_cov[0,1]:.4f}], [{inv_cov[1,0]:.4f}, {inv_cov[1,1]:.4f}]] × 
           [(x₁-{mean[0]}), (x₂-{mean[1]})]^T)

f(x₁,x₂) = (1/(2π×{np.sqrt(det_cov):.4f})) * 
           exp(-1/2 * [{inv_cov[0,0]:.4f}(x₁-{mean[0]})² + 
                       {2*inv_cov[0,1]:.4f}(x₁-{mean[0]})(x₂-{mean[1]}) + 
                       {inv_cov[1,1]:.4f}(x₂-{mean[1]})²])

f(x₁,x₂) = {1/(2*np.pi*np.sqrt(det_cov)):.6f} * 
           exp(-1/2 * [{inv_cov[0,0]:.4f}(x₁-{mean[0]})² + 
                       {2*inv_cov[0,1]:.4f}(x₁-{mean[0]})(x₂-{mean[1]}) + 
                       {inv_cov[1,1]:.4f}(x₂-{mean[1]})²])"""
    print_formula_box(pdf_formula)
    
    # Create grid for visualization
    x = np.linspace(-2, 6, 100)
    y = np.linspace(-2, 8, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    # Calculate PDF using scipy
    rv = multivariate_normal(mean, cov)
    Z = rv.pdf(pos)
    
    # Create contour plot visualization of the PDF
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, Z, 20, cmap='viridis')
    plt.colorbar(contour, label='Probability Density')
    plt.plot(mean[0], mean[1], 'ro', markersize=8)
    plt.text(mean[0]+0.2, mean[1]+0.2, f'μ=({mean[0]}, {mean[1]})', fontsize=12)
    plt.title('Bivariate Normal PDF: Contour Plot')
    plt.xlabel('$X_1$')
    plt.ylabel('$X_2$')
    plt.grid(alpha=0.3)
    plt.savefig(f"{output_dir}/example1_bivariate_normal_contour.png", dpi=300, bbox_inches='tight')
    print(f"\nContour plot saved to {output_dir}/example1_bivariate_normal_contour.png")
    plt.close()
    
    # Create 3D surface plot separately
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, linewidth=0)
    ax.contour(X, Y, Z, zdir='z', offset=0, cmap='viridis', alpha=0.5)
    plt.colorbar(surf, ax=ax, shrink=0.7, label='Probability Density')
    ax.set_title('Bivariate Normal PDF: 3D View')
    ax.set_xlabel('$X_1$')
    ax.set_ylabel('$X_2$')
    ax.set_zlabel('Density')
    plt.savefig(f"{output_dir}/example1_bivariate_normal_3d.png", dpi=300, bbox_inches='tight')
    print(f"3D surface plot saved to {output_dir}/example1_bivariate_normal_3d.png")
    plt.close()
    
    # Part B: Calculate the probability P(X_1 ≤ 3, X_2 ≤ 4)
    print("\n\nPART B: Calculate the probability P(X₁ ≤ 3, X₂ ≤ 4)")
    print("====================================================")
    
    # Calculate standardized bounds
    x1_bound = 3
    x2_bound = 4
    
    z1 = (x1_bound - mean[0]) / np.sqrt(cov[0, 0])
    z2 = (x2_bound - mean[1]) / np.sqrt(cov[1, 1])
    
    rho = cov[0, 1] / (np.sqrt(cov[0, 0]) * np.sqrt(cov[1, 1]))
    
    print("\nStep 1: Define the region and standardize the bounds")
    print("--------------------------------------------------")
    print(f"We need to find: P(X₁ ≤ {x1_bound}, X₂ ≤ {x2_bound})")
    print("\nTo use standard bivariate normal properties, we first standardize the variables:")
    
    standardization = f"""For X₁ ≤ {x1_bound}:
Z₁ = (X₁ - μ₁)/σ₁ = ({x1_bound} - {mean[0]})/{np.sqrt(cov[0, 0])} = {z1:.6f}

For X₂ ≤ {x2_bound}:
Z₂ = (X₂ - μ₂)/σ₂ = ({x2_bound} - {mean[1]})/{np.sqrt(cov[1, 1])} = {z2:.6f}"""
    print_formula_box(standardization)
    
    print("\nStep 2: Account for the correlation between variables")
    print("---------------------------------------------------")
    print("The correlation coefficient is needed for the bivariate normal CDF:")
    
    correlation = f"""ρ = σ₁₂/(σ₁σ₂) 
  = {cov[0, 1]}/({np.sqrt(cov[0, 0])} × {np.sqrt(cov[1, 1])})
  = {cov[0, 1]}/({np.sqrt(cov[0, 0] * cov[1, 1])})
  = {rho:.6f}"""
    print_formula_box(correlation)
    
    # Calculate the probability using scipy's multivariate normal CDF
    prob = rv.cdf([x1_bound, x2_bound])
    
    print("\nStep 3: Calculate the probability using the bivariate normal CDF")
    print("--------------------------------------------------------------")
    print("For a bivariate normal distribution, this probability requires evaluating the bivariate normal CDF:")
    print(f"P(X₁ ≤ {x1_bound}, X₂ ≤ {x2_bound}) = Φ₂({z1:.6f}, {z2:.6f}; {rho:.6f})")
    print("\nWhere Φ₂(a, b; ρ) is the CDF of the standard bivariate normal distribution")
    print("up to points a and b with correlation ρ.")
    print("\nUsing numerical computation, we find:")
    print(f"P(X₁ ≤ {x1_bound}, X₂ ≤ {x2_bound}) ≈ {prob:.6f} or {prob*100:.4f}%")
    
    detailed_cdf_explanation = f"""Calculation Process for Bivariate Normal CDF:

The exact computation involves a double integral:
P(X₁ ≤ {x1_bound}, X₂ ≤ {x2_bound}) = ∫_(-infinity)^{x2_bound} ∫_(-infinity)^{x1_bound} f(x₁, x₂) dx₁ dx₂

Where f(x₁, x₂) is the bivariate normal PDF we derived in Part A.

This integral has no closed-form solution for correlated variables (ρ ≠ 0).
For numerical calculation, transformation to standard bivariate normal and
specialized algorithms are used.

Result: P(X₁ ≤ {x1_bound}, X₂ ≤ {x2_bound}) ≈ {prob:.6f} or {prob*100:.4f}%"""
    print_formula_box(detailed_cdf_explanation)
    
    # Create visualization for the probability region (separate plot)
    plt.figure(figsize=(8, 7))
    
    # Create contour plot
    contour = plt.contour(X, Y, Z, 6, colors='gray', alpha=0.6)
    plt.clabel(contour, inline=True, fontsize=8)
    
    # Shade the region where X1 ≤ 3 and X2 ≤ 4
    x_region = np.linspace(-2, x1_bound, 100)
    y_region = np.linspace(-2, x2_bound, 100)
    X_region, Y_region = np.meshgrid(x_region, y_region)
    
    # Plot the region of interest
    plt.fill_between(x_region, -2, x2_bound, alpha=0.3, color='blue')
    plt.fill_betweenx(y_region, -2, x1_bound, alpha=0.3, color='blue')
    
    # Mark important points
    plt.plot(mean[0], mean[1], 'ro', markersize=8, label='Mean (2,3)')
    plt.plot(x1_bound, x2_bound, 'go', markersize=8, label='Point (3,4)')
    
    # Add arrows showing the limits
    plt.axvline(x=x1_bound, color='red', linestyle='--', alpha=0.7)
    plt.axhline(y=x2_bound, color='red', linestyle='--', alpha=0.7)
    
    # Add text annotation for the probability value
    plt.text(0, 6, f"P(X₁ ≤ 3, X₂ ≤ 4) ≈ {prob:.4f}", fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Set labels and title
    plt.title('Probability Region for P(X₁ ≤ 3, X₂ ≤ 4)')
    plt.xlabel('$X_1$')
    plt.ylabel('$X_2$')
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Save the figure
    plt.savefig(f"{output_dir}/example1_probability_region.png", dpi=300, bbox_inches='tight')
    print(f"\nProbability region plot saved to {output_dir}/example1_probability_region.png")
    plt.close()
    
    # Part C: Conditional distribution calculation
    print("\n\nPART C: Find the conditional distribution of X₁ given X₂ = 4")
    print("========================================================")
    
    print("\nDetailed, step-by-step solution:")
    
    # Print the formula for conditional distribution
    conditional_theory = """For a bivariate normal distribution, the conditional distribution of X₁ given X₂ = x₂ is also normal.

The parameters of this conditional distribution are:
1. Conditional Mean: μ₁|₂ = μ₁ + (σ₁₂/σ₂²)(x₂ - μ₂)
2. Conditional Variance: σ²₁|₂ = σ₁² - (σ₁₂²/σ₂²)

Where:
- μ₁ is the mean of X₁
- μ₂ is the mean of X₂
- σ₁² is the variance of X₁
- σ₂² is the variance of X₂
- σ₁₂ is the covariance between X₁ and X₂
- x₂ is the conditioning value of X₂"""
    print_formula_box(conditional_theory)
    
    print("\nStep 1: Calculate the conditional mean")
    print("-------------------------------------")
    
    sigma12 = cov[0, 1]
    sigma2_sq = cov[1, 1]
    x2_value = 4
    
    print(f"Using the formula: μ₁|₂ = μ₁ + (σ₁₂/σ₂²)(x₂ - μ₂)")
    print(f"We substitute the values:")
    
    cond_mean_steps = f"""μ₁|₂ = {mean[0]} + ({sigma12}/{sigma2_sq})({x2_value} - {mean[1]})
      = {mean[0]} + ({sigma12}/{sigma2_sq})({x2_value-mean[1]})
      = {mean[0]} + {sigma12/sigma2_sq:.6f} × {x2_value-mean[1]}
      = {mean[0]} + {(sigma12/sigma2_sq)*(x2_value-mean[1]):.6f}
      = {mean[0] + (sigma12/sigma2_sq)*(x2_value-mean[1]):.6f}"""
    print_formula_box(cond_mean_steps)
    
    cond_mean = mean[0] + (sigma12/sigma2_sq) * (x2_value - mean[1])
    
    print("\nStep 2: Calculate the conditional variance")
    print("----------------------------------------")
    print(f"Using the formula: σ²₁|₂ = σ₁² - (σ₁₂²/σ₂²)")
    print(f"We substitute the values:")
    
    cond_var = cov[0, 0] - (sigma12**2)/sigma2_sq
    
    cond_var_steps = f"""σ²₁|₂ = {cov[0, 0]} - ({sigma12}²/{sigma2_sq})
      = {cov[0, 0]} - ({sigma12**2}/{sigma2_sq})
      = {cov[0, 0]} - {(sigma12**2)/sigma2_sq:.6f}
      = {cov[0, 0] - (sigma12**2)/sigma2_sq:.6f}"""
    print_formula_box(cond_var_steps)
    
    print("\nStep 3: Write the conditional distribution")
    print("-----------------------------------------")
    
    conditional_result = f"""The conditional distribution is:
X₁|(X₂={x2_value}) ~ N({cond_mean:.6f}, {cond_var:.6f})

This means that when we know X₂ = {x2_value}, the random variable X₁ follows
a normal distribution with:
- Mean: {cond_mean:.6f}
- Variance: {cond_var:.6f}

Note that:
1. The mean has shifted from the original {mean[0]} to {cond_mean:.6f}
2. The variance has decreased from the original {cov[0,0]} to {cond_var:.6f}

The shift in mean reflects the correlation between X₁ and X₂, while 
the reduction in variance shows how knowledge about X₂ reduces uncertainty about X₁."""
    print_formula_box(conditional_result)
    
    # Visualize the conditional distribution (separate plot)
    plt.figure(figsize=(10, 6))
    
    # Plot the marginal distribution of X1
    x1_range = np.linspace(-2, 6, 1000)
    marginal_x1 = multivariate_normal(mean[0], cov[0, 0]).pdf(x1_range)
    plt.plot(x1_range, marginal_x1, 'b--', linewidth=2, 
             label=f'Marginal: X₁ ~ N({mean[0]}, {cov[0, 0]})')
    
    # Plot the conditional distribution X1|X2=4
    cond_x1 = multivariate_normal(cond_mean, cond_var).pdf(x1_range)
    plt.plot(x1_range, cond_x1, 'r-', linewidth=2, 
             label=f'Conditional: X₁|(X₂=4) ~ N({cond_mean:.2f}, {cond_var:.2f})')
    
    # Add annotations and labels
    plt.title('Conditional vs. Marginal Distribution of $X_1$')
    plt.xlabel('$X_1$')
    plt.ylabel('Density')
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Save this figure too
    plt.savefig(f"{output_dir}/example1_conditional_distribution.png", dpi=300, bbox_inches='tight')
    print(f"\nConditional distribution plot saved to {output_dir}/example1_conditional_distribution.png")
    plt.close()

def example2_trivariate_normal():
    """
    Example 2: Trivariate Normal Distribution with Marginal and Conditional
    Mean vector: μ = (1, 2, 3)
    Covariance matrix: Σ = [[3, 1, -1], [1, 2, 0], [-1, 0, 4]]
    """
    print("\n======== EXAMPLE 2: TRIVARIATE NORMAL DISTRIBUTION ========")
    output_dir = create_output_dir()
    
    # Print the problem statement
    problem_statement = """Problem Statement:
Consider a trivariate normal distribution with random variables (X, Y, Z) where:
- Mean vector: μ = (1, 2, 3)
- Covariance matrix: Σ = [
    [3, 1, -1],
    [1, 2, 0],
    [-1, 0, 4]
]

We need to:
a) Find the marginal density function f(x, y)
b) Find the conditional density function f(z | x=2, y=1)"""
    print_formula_box(problem_statement)
    
    # Parameters from the example
    mean = np.array([1, 2, 3])
    cov = np.array([[3, 1, -1], [1, 2, 0], [-1, 0, 4]])
    
    # Part a: Marginal distribution of (X, Y)
    print("\nPART A: Finding the marginal density function f(x, y)")
    print("==================================================")
    
    # Print theory about marginal distributions in MVN
    marginal_theory = """Marginal Distributions in Multivariate Normal:

For multivariate normal distributions, any marginal distribution is also
multivariate normal.

To find the marginal distribution of a subset of variables:
1. Extract the corresponding elements of the mean vector
2. Extract the corresponding submatrix of the covariance matrix

The resulting density function will follow the multivariate normal formula
with these extracted parameters."""
    print_formula_box(marginal_theory)
    
    print("\nDetailed, step-by-step solution:")
    
    # Extract the mean vector and covariance matrix for X and Y
    mean_xy = mean[0:2]
    cov_xy = cov[0:2, 0:2]
    
    print("\nStep 1: Extract the corresponding elements of the mean vector and covariance matrix")
    print("------------------------------------------------------------------------------")
    
    extraction_explanation = f"""From the original mean vector μ = [{mean[0]}, {mean[1]}, {mean[2]}], we extract:
μ_xy = [{mean_xy[0]}, {mean_xy[1]}]

From the original covariance matrix:
Σ = [
    [{cov[0,0]}, {cov[0,1]}, {cov[0,2]}],
    [{cov[1,0]}, {cov[1,1]}, {cov[1,2]}],
    [{cov[2,0]}, {cov[2,1]}, {cov[2,2]}]
]

We extract the submatrix corresponding to X and Y:
Σ_xy = [
    [{cov_xy[0,0]}, {cov_xy[0,1]}],
    [{cov_xy[1,0]}, {cov_xy[1,1]}]
]"""
    print_formula_box(extraction_explanation)
    
    # Calculate the determinant of the covariance matrix
    det_cov_xy = np.linalg.det(cov_xy)
    
    print("\nStep 2: Calculate the determinant of the covariance submatrix")
    print("---------------------------------------------------------")
    print(f"det(Σ_xy) = det([[{cov_xy[0,0]}, {cov_xy[0,1]}], [{cov_xy[1,0]}, {cov_xy[1,1]}]])")
    print(f"         = ({cov_xy[0,0]} × {cov_xy[1,1]}) - ({cov_xy[0,1]} × {cov_xy[1,0]})")
    print(f"         = {cov_xy[0,0] * cov_xy[1,1]} - {cov_xy[0,1] * cov_xy[1,0]}")
    print(f"         = {det_cov_xy}")
    
    # Calculate the inverse of the covariance matrix
    inv_cov_xy = np.linalg.inv(cov_xy)
    
    print("\nStep 3: Calculate the inverse of the covariance submatrix")
    print("-------------------------------------------------------")
    print("For a 2×2 matrix, the inverse formula is:")
    print("        1     |  d  -b |")
    print("A^(-1) = --- × | -c   a |")
    print("       det(A)  |        |")
    
    print(f"\nApplying this to our covariance submatrix:")
    print(f"Σ_xy^(-1) = (1/{det_cov_xy}) × [[{cov_xy[1,1]}, -{cov_xy[0,1]}], [−{cov_xy[1,0]}, {cov_xy[0,0]}]]")
    print(f"          = [[{inv_cov_xy[0,0]:.6f}, {inv_cov_xy[0,1]:.6f}], [{inv_cov_xy[1,0]:.6f}, {inv_cov_xy[1,1]:.6f}]]")
    
    # Step 4: Write the marginal PDF formula
    print("\nStep 4: Write the marginal PDF formula")
    print("------------------------------------")
    
    marginal_pdf = f"""Using the multivariate normal formula with our extracted parameters:

f(x, y) = (1/(2π√|Σ_xy|)) * exp(-1/2 * [(x-μ_x), (y-μ_y)] × Σ_xy^(-1) × [(x-μ_x), (y-μ_y)]^T)

f(x, y) = (1/(2π√{det_cov_xy})) * exp(-1/2 * [(x-{mean_xy[0]}), (y-{mean_xy[1]})] × 
         [[{inv_cov_xy[0,0]:.4f}, {inv_cov_xy[0,1]:.4f}], [{inv_cov_xy[1,0]:.4f}, {inv_cov_xy[1,1]:.4f}]] × 
         [(x-{mean_xy[0]}), (y-{mean_xy[1]})]^T)

f(x, y) = (1/(2π×{np.sqrt(det_cov_xy):.4f})) * 
         exp(-1/2 * [{inv_cov_xy[0,0]:.4f}(x-{mean_xy[0]})² + 
                     {2*inv_cov_xy[0,1]:.4f}(x-{mean_xy[0]})(y-{mean_xy[1]}) + 
                     {inv_cov_xy[1,1]:.4f}(y-{mean_xy[1]})²])

f(x, y) = {1/(2*np.pi*np.sqrt(det_cov_xy)):.6f} * 
         exp(-1/2 * [{inv_cov_xy[0,0]:.4f}(x-{mean_xy[0]})² + 
                     {2*inv_cov_xy[0,1]:.4f}(x-{mean_xy[0]})(y-{mean_xy[1]}) + 
                     {inv_cov_xy[1,1]:.4f}(y-{mean_xy[1]})²])"""
    print_formula_box(marginal_pdf)
    
    print("\nTherefore, the marginal distribution of (X, Y) is:")
    print(f"(X, Y) ~ N(μ_xy, Σ_xy)")
    print(f"(X, Y) ~ N([{mean_xy[0]}, {mean_xy[1]}], [[{cov_xy[0,0]}, {cov_xy[0,1]}], [{cov_xy[1,0]}, {cov_xy[1,1]}]])")
    
    # Visualize the marginal distribution separately
    # Create a grid for visualization
    x = np.linspace(-3, 5, 100)
    y = np.linspace(-2, 6, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    # Calculate the marginal PDF
    rv_xy = multivariate_normal(mean_xy, cov_xy)
    Z_xy = rv_xy.pdf(pos)
    
    # Create the contour visualization
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, Z_xy, 20, cmap='plasma')
    plt.colorbar(contour, label='Probability Density')
    plt.plot(mean_xy[0], mean_xy[1], 'ro', markersize=8)
    plt.text(mean_xy[0]+0.2, mean_xy[1]+0.2, f'μ=({mean_xy[0]}, {mean_xy[1]})', fontsize=12)
    plt.title('Marginal Distribution of (X, Y): Contour Plot')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(alpha=0.3)
    plt.savefig(f"{output_dir}/example2_marginal_xy_contour.png", dpi=300, bbox_inches='tight')
    print(f"\nContour plot saved to {output_dir}/example2_marginal_xy_contour.png")
    plt.close()
    
    # Create 3D surface plot separately
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, Y, Z_xy, cmap='plasma', alpha=0.8, linewidth=0)
    ax.contour(X, Y, Z_xy, zdir='z', offset=0, cmap='plasma', alpha=0.5)
    plt.colorbar(surf, ax=ax, shrink=0.7, label='Probability Density')
    ax.set_title('Marginal Distribution of (X, Y): 3D View')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Density')
    plt.savefig(f"{output_dir}/example2_marginal_xy_3d.png", dpi=300, bbox_inches='tight')
    print(f"3D surface plot saved to {output_dir}/example2_marginal_xy_3d.png")
    plt.close()
    
    # Part b: Conditional distribution of Z given X=2, Y=1
    print("\n\nPART B: Finding the conditional density function f(z | x=2, y=1)")
    print("============================================================")
    
    # Print theory about conditional distributions in MVN
    conditional_theory = """Conditional Distributions in Multivariate Normal:

For multivariate normal distributions, any conditional distribution is also
normal. To find the conditional distribution, we partition the variables:

Let X = [X₁, X₂] be multivariate normal, where:
- X₁ are the variables of interest (what we're conditioning on)
- X₂ are the conditioning variables (what we know values for)

The conditional distribution has parameters:
1. μ₁|₂ = μ₁ + Σ₁₂ Σ₂₂⁻¹ (x₂ - μ₂)
2. Σ₁|₂ = Σ₁₁ - Σ₁₂ Σ₂₂⁻¹ Σ₂₁

Where:
- μ₁, μ₂ are the mean vectors of X₁ and X₂
- Σ₁₁ is the covariance matrix of X₁
- Σ₂₂ is the covariance matrix of X₂
- Σ₁₂ is the cross-covariance matrix between X₁ and X₂
- Σ₂₁ is the transpose of Σ₁₂"""
    print_formula_box(conditional_theory)
    
    print("\nDetailed, step-by-step solution:")
    
    # For conditional distributions, partition the variables
    print("\nStep 1: Partition the variables and identify components")
    print("----------------------------------------------------")
    
    # Partition explanation
    partition_explanation = """For our problem, we partition as follows:
- X₁ = Z (the variable we want the distribution for)
- X₂ = (X, Y) (the variables we are conditioning on)

We need to identify all the components required for the conditional formula:
- μ₁: mean of Z
- μ₂: mean vector of (X, Y)
- Σ₁₁: variance of Z
- Σ₂₂: covariance matrix of (X, Y)
- Σ₁₂: covariance vector between Z and (X, Y)
- Σ₂₁: transpose of Σ₁₂"""
    print_formula_box(partition_explanation)
    
    # Define all components
    mu1 = mean[2]  # mean of Z
    mu2 = mean[:2]  # mean of (X, Y)
    sigma11 = cov[2, 2]  # variance of Z
    sigma12 = cov[2, :2]  # covariance between Z and (X, Y)
    sigma21 = cov[:2, 2]  # covariance between (X, Y) and Z
    sigma22 = cov[:2, :2]  # covariance matrix of (X, Y)
    
    components_values = f"""From our original parameters, we identify:

μ₁ = {mu1} (mean of Z)
μ₂ = [{mu2[0]}, {mu2[1]}] (mean vector of (X, Y))

Σ₁₁ = {sigma11} (variance of Z)

Σ₁₂ = [{sigma12[0]}, {sigma12[1]}] (covariance vector between Z and (X, Y))

Σ₂₁ = [
    {sigma21[0]},
    {sigma21[1]}
] (covariance vector between (X, Y) and Z)

Σ₂₂ = [
    [{sigma22[0,0]}, {sigma22[0,1]}],
    [{sigma22[1,0]}, {sigma22[1,1]}]
] (covariance matrix of (X, Y))"""
    print_formula_box(components_values)
    
    # Conditioning values
    x_value = 2
    y_value = 1
    x2_value = np.array([x_value, y_value])
    
    print(f"\nWe are conditioning on X = {x_value} and Y = {y_value}")
    print(f"So x₂ = [{x_value}, {y_value}]")
    
    print("\nStep 2: Calculate the conditional mean")
    print("------------------------------------")
    print("Using the formula: μ₁|₂ = μ₁ + Σ₁₂ Σ₂₂⁻¹ (x₂ - μ₂)")
    
    # Calculate inverse of sigma22
    inv_sigma22 = np.linalg.inv(sigma22)
    diff = x2_value - mu2
    
    inv_sigma22_calculation = f"""First, we need to calculate Σ₂₂⁻¹:
Σ₂₂⁻¹ = [
    [{sigma22[0,0]}, {sigma22[0,1]}],
    [{sigma22[1,0]}, {sigma22[1,1]}]
]⁻¹

Using the formula for 2×2 matrix inverse:
Σ₂₂⁻¹ = (1/{np.linalg.det(sigma22)}) × [
    [{sigma22[1,1]}, -{sigma22[0,1]}],
    [-{sigma22[1,0]}, {sigma22[0,0]}]
]

This gives us:
Σ₂₂⁻¹ = [
    [{inv_sigma22[0,0]:.6f}, {inv_sigma22[0,1]:.6f}],
    [{inv_sigma22[1,0]:.6f}, {inv_sigma22[1,1]:.6f}]
]

Next, we calculate (x₂ - μ₂):
(x₂ - μ₂) = [{x2_value[0]}, {x2_value[1]}] - [{mu2[0]}, {mu2[1]}] = [{diff[0]}, {diff[1]}]"""
    print_formula_box(inv_sigma22_calculation)
    
    # Calculate the matrix multiplication part
    term = np.dot(sigma12, np.dot(inv_sigma22, diff))
    cond_mean = mu1 + term
    
    matrix_multiplication = f"""Now we calculate Σ₁₂ Σ₂₂⁻¹ (x₂ - μ₂):

Σ₁₂ Σ₂₂⁻¹ (x₂ - μ₂) = [{sigma12[0]}, {sigma12[1]}] × [
    [{inv_sigma22[0,0]:.6f}, {inv_sigma22[0,1]:.6f}],
    [{inv_sigma22[1,0]:.6f}, {inv_sigma22[1,1]:.6f}]
] × [
    {diff[0]},
    {diff[1]}
]

= [{sigma12[0]}, {sigma12[1]}] × [
    {inv_sigma22[0,0]*diff[0] + inv_sigma22[0,1]*diff[1]:.6f},
    {inv_sigma22[1,0]*diff[0] + inv_sigma22[1,1]*diff[1]:.6f}
]

= {sigma12[0]}*{inv_sigma22[0,0]*diff[0] + inv_sigma22[0,1]*diff[1]:.6f} + 
  {sigma12[1]}*{inv_sigma22[1,0]*diff[0] + inv_sigma22[1,1]*diff[1]:.6f}

= {term:.6f}"""
    print_formula_box(matrix_multiplication)
    
    cond_mean_final = f"""Finally, we can calculate the conditional mean:

μ₁|₂ = μ₁ + Σ₁₂ Σ₂₂⁻¹ (x₂ - μ₂)
     = {mu1} + {term:.6f}
     = {cond_mean:.6f}"""
    print_formula_box(cond_mean_final)
    
    # Calculate conditional variance
    print("\nStep 3: Calculate the conditional variance")
    print("----------------------------------------")
    print("Using the formula: σ²₁|₂ = Σ₁₁ - Σ₁₂ Σ₂₂⁻¹ Σ₂₁")
    
    # Calculate the term Σ₁₂ Σ₂₂⁻¹ Σ₂₁
    term2 = np.dot(sigma12, np.dot(inv_sigma22, sigma21))
    cond_var = sigma11 - term2
    
    variance_calculation = f"""Σ₁₂ Σ₂₂⁻¹ Σ₂₁ = [{sigma12[0]}, {sigma12[1]}] × [
    [{inv_sigma22[0,0]:.6f}, {inv_sigma22[0,1]:.6f}],
    [{inv_sigma22[1,0]:.6f}, {inv_sigma22[1,1]:.6f}]
] × [
    {sigma21[0]},
    {sigma21[1]}
]

This requires two matrix multiplications:

First, let's calculate Σ₂₂⁻¹ Σ₂₁:
Σ₂₂⁻¹ Σ₂₁ = [
    [{inv_sigma22[0,0]:.6f}, {inv_sigma22[0,1]:.6f}],
    [{inv_sigma22[1,0]:.6f}, {inv_sigma22[1,1]:.6f}]
] × [
    {sigma21[0]},
    {sigma21[1]}
]

= [
    {inv_sigma22[0,0]*sigma21[0] + inv_sigma22[0,1]*sigma21[1]:.6f},
    {inv_sigma22[1,0]*sigma21[0] + inv_sigma22[1,1]*sigma21[1]:.6f}
]

Then, we multiply Σ₁₂ by this result:
Σ₁₂ × (Σ₂₂⁻¹ Σ₂₁) = [{sigma12[0]}, {sigma12[1]}] × [
    {inv_sigma22[0,0]*sigma21[0] + inv_sigma22[0,1]*sigma21[1]:.6f},
    {inv_sigma22[1,0]*sigma21[0] + inv_sigma22[1,1]*sigma21[1]:.6f}
]

= {sigma12[0]}*{inv_sigma22[0,0]*sigma21[0] + inv_sigma22[0,1]*sigma21[1]:.6f} + 
  {sigma12[1]}*{inv_sigma22[1,0]*sigma21[0] + inv_sigma22[1,1]*sigma21[1]:.6f}

= {term2:.6f}"""
    print_formula_box(variance_calculation)
    
    cond_var_final = f"""Now we can calculate the conditional variance:

σ²₁|₂ = Σ₁₁ - Σ₁₂ Σ₂₂⁻¹ Σ₂₁
      = {sigma11} - {term2:.6f}
      = {cond_var:.6f}"""
    print_formula_box(cond_var_final)
    
    print("\nStep 4: Write the conditional distribution")
    print("-----------------------------------------")
    
    conditional_distribution = f"""The conditional distribution is:
Z|(X={x_value},Y={y_value}) ~ N({cond_mean:.6f}, {cond_var:.6f})

This means that when we know X = {x_value} and Y = {y_value}, the random variable Z 
follows a normal distribution with:
- Mean: {cond_mean:.6f}
- Variance: {cond_var:.6f}

Notes:
1. The mean has shifted from the original {mu1} to {cond_mean:.6f}
2. The variance has decreased from the original {sigma11} to {cond_var:.6f}
3. The negative covariance between Z and X (-1) explains why the 
   conditional mean decreases when X=2 (which is above X's mean of 1)"""
    print_formula_box(conditional_distribution)
    
    # Visualize the conditional distribution
    plt.figure(figsize=(10, 6))
    
    # Plot the marginal distribution of Z
    z_range = np.linspace(-2, 8, 1000)
    marginal_z = multivariate_normal(mu1, sigma11).pdf(z_range)
    plt.plot(z_range, marginal_z, 'b--', linewidth=2, 
             label=f'Marginal: Z ~ N({mu1}, {sigma11})')
    
    # Plot the conditional distribution Z|X=2,Y=1
    cond_z = multivariate_normal(cond_mean, cond_var).pdf(z_range)
    plt.plot(z_range, cond_z, 'r-', linewidth=2, 
             label=f'Conditional: Z|(X=2,Y=1) ~ N({cond_mean:.2f}, {cond_var:.2f})')
    
    # Add annotations and labels
    plt.title('Conditional vs. Marginal Distribution of Z')
    plt.xlabel('Z')
    plt.ylabel('Density')
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Add formula annotation
    formula = f"Z|(X=2,Y=1) ~ N({cond_mean:.2f}, {cond_var:.2f})"
    plt.annotate(formula, xy=(0.5, 0.9), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                 ha='center', fontsize=12)
    
    # Save this figure
    plt.savefig(f"{output_dir}/example2_conditional_z_given_xy.png", dpi=300, bbox_inches='tight')
    print(f"\nConditional distribution plot saved to {output_dir}/example2_conditional_z_given_xy.png")
    plt.close()

def main():
    """Main function to run all examples"""
    print("MULTIVARIATE DENSITY FUNCTION EXAMPLES - DETAILED SOLUTIONS")
    print("=========================================================")
    
    # Run Example 1
    example1_bivariate_normal()
    
    # Run Example 2
    example2_trivariate_normal()
    
    print("\nAll examples completed successfully!")

if __name__ == "__main__":
    main() 