import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images/Conditional_Distribution relative to the parent directory
images_dir = os.path.join(parent_dir, "Images", "Conditional_Distribution")

# Create images directory if it doesn't exist
os.makedirs(images_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

def print_section_header(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")

def plot_bivariate_normal(mu, cov, title, filename):
    """Plot simplified bivariate normal distribution"""
    # Create grid of points
    x = np.linspace(-5, 10, 100)
    y = np.linspace(-5, 15, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    # Calculate PDF values
    rv = stats.multivariate_normal(mu, cov)
    Z = rv.pdf(pos)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot contours
    contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    ax.contour(X, Y, Z, levels=10, colors='black', alpha=0.3)
    
    # Add correlation ellipse
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    # Calculate angles and lengths for ellipse
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * np.sqrt(eigenvalues)
    ellipse = Ellipse(mu, width=width, height=height, angle=angle, 
                    edgecolor='red', facecolor='none', linewidth=2)
    ax.add_patch(ellipse)
    
    # Add mean point
    ax.scatter(mu[0], mu[1], c='red', s=100, label='Mean')
    
    # Basic plot settings
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('X₁', fontsize=10)
    ax.set_ylabel('X₂', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {filename}")


def example1_bivariate_normal_conditional():
    """Example 1: Conditional Distributions in Bivariate Normal"""
    print_section_header("EXAMPLE 1: CONDITIONAL DISTRIBUTIONS IN BIVARIATE NORMAL")
    
    print("Problem Statement:")
    print("Consider a bivariate normal distribution where X = (X₁, X₂) has:")
    print("- Mean vector μ = (3, 5)")
    print("- Covariance matrix Σ = [[9, 6], [6, 16]]")
    print("\nWe want to:")
    print("a) Find the conditional distribution of X₁ given X₂ = 7")
    print("b) Determine the best prediction for X₁ given X₂ = 7")
    print("c) Calculate the reduction in variance when predicting X₁ after observing X₂")
    
    # Define parameters
    mu = np.array([3, 5])
    cov = np.array([[9, 6], [6, 16]])
    x2_value = 7
    
    # Step 0: Calculate correlation coefficient
    print("\nStep 0: Calculate correlation coefficient")
    rho = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    print(f"ρ = σ₁₂/√(σ₁₁σ₂₂) = {cov[0, 1]}/√({cov[0, 0]} × {cov[1, 1]})")
    print(f"  = {cov[0, 1]}/√{cov[0, 0] * cov[1, 1]}")
    print(f"  = {cov[0, 1]}/{np.sqrt(cov[0, 0] * cov[1, 1]):.4f}")
    print(f"  = {rho:.4f}")
    
    # Step 1: Identify parameters
    print("\nStep 1: Identify parameters from the bivariate normal distribution")
    mu1, mu2 = mu
    sigma11, sigma12 = cov[0, 0], cov[0, 1]
    sigma21, sigma22 = cov[1, 0], cov[1, 1]
    
    print(f"μ₁ = {mu1} (mean of X₁)")
    print(f"μ₂ = {mu2} (mean of X₂)")
    print(f"σ₁₁ = {sigma11} (variance of X₁)")
    print(f"σ₁₂ = σ₂₁ = {sigma12} (covariance between X₁ and X₂)")
    print(f"σ₂₂ = {sigma22} (variance of X₂)")
    
    # Step 2: Calculate conditional mean
    print("\nStep 2: Calculate the conditional mean")
    # Calculate regression coefficient
    beta = sigma12 / sigma22
    print(f"Regression coefficient: β = σ₁₂/σ₂₂ = {sigma12}/{sigma22} = {beta:.4f}")
    
    # Calculate the deviation from X₂'s mean
    deviation = x2_value - mu2
    print(f"Deviation from X₂'s mean: (x₂ - μ₂) = {x2_value} - {mu2} = {deviation}")
    
    # Calculate the adjustment
    adjustment = beta * deviation
    print(f"Mean adjustment: β × (x₂ - μ₂) = {beta:.4f} × {deviation} = {adjustment:.4f}")
    
    # Calculate the conditional mean
    mu1_given_2 = mu1 + adjustment
    print(f"Conditional mean: μ₁|₂ = μ₁ + β(x₂ - μ₂) = {mu1} + {adjustment:.4f} = {mu1_given_2:.4f}")
    
    # Step 3: Calculate conditional variance
    print("\nStep 3: Calculate the conditional variance")
    # Calculate squared covariance
    sigma12_squared = sigma12**2
    print(f"Squared covariance: σ₁₂² = {sigma12}² = {sigma12_squared}")
    
    # Calculate variance reduction
    variance_reduction = sigma12_squared / sigma22
    print(f"Variance reduction: σ₁₂²/σ₂₂ = {sigma12_squared}/{sigma22} = {variance_reduction:.4f}")
    
    # Calculate conditional variance
    sigma1_given_2_squared = sigma11 - variance_reduction
    print(f"Conditional variance: σ₁|₂² = σ₁₁ - σ₁₂²/σ₂₂ = {sigma11} - {variance_reduction:.4f} = {sigma1_given_2_squared:.4f}")
    
    # Calculate conditional standard deviation
    sigma1_given_2 = np.sqrt(sigma1_given_2_squared)
    print(f"Conditional standard deviation: σ₁|₂ = √σ₁|₂² = √{sigma1_given_2_squared:.4f} = {sigma1_given_2:.4f}")
    
    # Summarize the conditional distribution
    print("\nSolution to part a:")
    print(f"The conditional distribution is X₁|(X₂={x2_value}) ~ N({mu1_given_2:.4f}, {sigma1_given_2_squared:.4f})")
    
    # Best prediction (answer to part b)
    print("\nSolution to part b:")
    print(f"The best prediction for X₁ given X₂ = {x2_value} is the conditional mean: {mu1_given_2:.4f}")
    
    # Variance reduction (answer to part c)
    print("\nSolution to part c:")
    absolute_reduction = sigma11 - sigma1_given_2_squared
    percentage_reduction = (absolute_reduction / sigma11) * 100
    print(f"Absolute reduction in variance: {sigma11} - {sigma1_given_2_squared:.4f} = {absolute_reduction:.4f}")
    print(f"Percentage reduction in variance: ({absolute_reduction:.4f}/{sigma11}) × 100% = {percentage_reduction:.2f}%")
    
    # Verify relation with correlation coefficient
    rho_squared = rho**2
    print(f"\nVerifying relationship with correlation coefficient:")
    print(f"ρ² = {rho:.4f}² = {rho_squared:.4f} = {rho_squared * 100:.2f}%")
    print(f"This equals the percentage reduction in variance: {percentage_reduction:.2f}%")
    
    # Derive general regression equation
    print("\nGeneral regression equation for predicting X₁ from any value of X₂:")
    intercept = mu1 - beta * mu2
    print(f"E[X₁|X₂=x₂] = μ₁ + β(x₂ - μ₂)")
    print(f"E[X₁|X₂=x₂] = {mu1} + {beta:.4f}(x₂ - {mu2})")
    print(f"E[X₁|X₂=x₂] = {mu1} + {beta:.4f}x₂ - {beta:.4f}×{mu2}")
    print(f"E[X₁|X₂=x₂] = {intercept:.4f} + {beta:.4f}x₂")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 1. Plot the joint distribution
    plot_bivariate_normal(mu, cov, 
                         'Bivariate Normal Distribution',
                         'example1_joint_distribution.png')
    
    # 2. Visualize conditional distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Range for X₁
    x1_range = np.linspace(-5, 10, 1000)
    
    # Calculate marginal distribution of X₁
    x1_marginal = stats.norm.pdf(x1_range, mu1, np.sqrt(sigma11))
    
    # Calculate conditional distribution of X₁ given X₂=7
    x1_conditional = stats.norm.pdf(x1_range, mu1_given_2, sigma1_given_2)
    
    # Plot both distributions
    ax.plot(x1_range, x1_marginal, 'b-', linewidth=2, label='Marginal Distribution')
    ax.plot(x1_range, x1_conditional, 'r-', linewidth=2, label='Conditional Distribution')
    
    # Add mean lines
    ax.axvline(x=mu1, color='blue', linestyle='--', alpha=0.7, label='Marginal Mean')
    ax.axvline(x=mu1_given_2, color='red', linestyle='--', alpha=0.7, label='Conditional Mean')
    
    # Basic plot settings
    ax.set_title('Marginal vs Conditional Distribution', fontsize=12)
    ax.set_xlabel('X₁', fontsize=10)
    ax.set_ylabel('Probability Density', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'example1_conditional_vs_marginal.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure saved to example1_conditional_vs_marginal.png")
    
    # 3. Create 3D visualization showing the conditioning
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create grid
    x1 = np.linspace(-5, 10, 50)
    x2 = np.linspace(-5, 15, 50)
    X1, X2 = np.meshgrid(x1, x2)
    pos = np.dstack((X1, X2))
    
    # Calculate PDF
    rv = stats.multivariate_normal(mu, cov)
    Z = rv.pdf(pos)
    
    # Plot surface
    surf = ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)
    
    # Add conditioning plane at X₂=7
    x1_plane = np.linspace(-5, 10, 20)
    x2_plane = np.ones(20) * x2_value
    z_plane = np.zeros(20)
    
    # Get PDF values along the conditioning line
    for i in range(len(x1_plane)):
        z_plane[i] = rv.pdf(np.array([x1_plane[i], x2_plane[i]]))
    
    # Plot conditioning plane
    ax.plot(x1_plane, x2_plane, z_plane, 'r-', linewidth=2, label='Conditioning Plane')
    
    # Basic plot settings
    ax.set_xlabel('X₁', fontsize=10)
    ax.set_ylabel('X₂', fontsize=10)
    ax.set_zlabel('Density', fontsize=10)
    ax.set_title('3D Visualization of Conditioning', fontsize=12)
    ax.legend(fontsize=10)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'example1_3d_conditioning.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure saved to example1_3d_conditioning.png")
    
    return mu, cov, mu1_given_2, sigma1_given_2_squared

def example2_trivariate_normal_conditional():
    """Example 2: Conditional Distributions and Inference in Trivariate Normal"""
    # Define parameters
    mu = np.array([5, 7, 10])
    cov = np.array([[4, 2, 1], [2, 9, 3], [1, 3, 5]])
    x23_value = np.array([8, 11])
    
    # Calculate correlation matrix
    sigma_sqrt = np.sqrt(np.diag(cov))
    corr = cov / (sigma_sqrt[:, None] * sigma_sqrt[None, :])
    
    # Partition parameters for X₁|(X₂=8,X₃=11)
    mu1 = mu[0]
    mu2 = mu[1:]
    Sigma11 = cov[0, 0]
    Sigma12 = cov[0, 1:]
    Sigma21 = cov[1:, 0]
    Sigma22 = cov[1:, 1:]
    
    # Calculate conditional mean and variance for X₁|(X₂=8,X₃=11)
    Sigma22_inv = np.linalg.inv(Sigma22)
    x2_minus_mu2 = x23_value - mu2
    mu1_given_23 = mu1 + Sigma12 @ Sigma22_inv @ x2_minus_mu2
    sigma1_given_23 = Sigma11 - Sigma12 @ Sigma22_inv @ Sigma21
    
    # Calculate adjugate matrix
    adj_Sigma22 = np.array([
        [Sigma22[1, 1], -Sigma22[0, 1]],
        [-Sigma22[1, 0], Sigma22[0, 0]]
    ])
    print(f"Adjugate matrix of Σ₂₂: \n{adj_Sigma22}")
    
    # Calculate inverse
    Sigma22_inv = adj_Sigma22 / det_Sigma22
    print(f"Inverse of Σ₂₂: Σ₂₂⁻¹ = \n{Sigma22_inv}")
    
    # Verify inverse calculation using numpy
    Sigma22_inv_np = np.linalg.inv(Sigma22)
    print(f"\nVerification using numpy: \n{Sigma22_inv_np}")
    
    # Step 3: Calculate (x₂ - μ₂)
    print("\nStep 3: Calculate (x₂ - μ₂)")
    x2_minus_mu2 = x23_value - mu2
    print(f"x₂ - μ₂ = {x23_value} - {mu2} = {x2_minus_mu2}")
    
    # Step 4: Calculate conditional mean
    print("\nStep 4: Calculate conditional mean")
    print("Formula: μ₁|₂ = μ₁ + Σ₁₂Σ₂₂⁻¹(x₂ - μ₂)")
    
    # Calculate Σ₂₂⁻¹(x₂ - μ₂)
    sigma22_inv_x2_minus_mu2 = Sigma22_inv @ x2_minus_mu2
    print(f"Σ₂₂⁻¹(x₂ - μ₂) = \n{Sigma22_inv} @ {x2_minus_mu2} = {sigma22_inv_x2_minus_mu2}")
    
    # Calculate Σ₁₂Σ₂₂⁻¹(x₂ - μ₂)
    sigma12_sigma22_inv_x2_minus_mu2 = Sigma12 @ sigma22_inv_x2_minus_mu2
    print(f"Σ₁₂Σ₂₂⁻¹(x₂ - μ₂) = {Sigma12} @ {sigma22_inv_x2_minus_mu2} = {sigma12_sigma22_inv_x2_minus_mu2:.4f}")
    
    # Calculate conditional mean
    mu1_given_23 = mu1 + sigma12_sigma22_inv_x2_minus_mu2
    print(f"μ₁|₂₃ = {mu1} + {sigma12_sigma22_inv_x2_minus_mu2:.4f} = {mu1_given_23:.4f}")
    
    # Step 5: Calculate conditional variance
    print("\nStep 5: Calculate conditional variance")
    print("Formula: Σ₁|₂ = Σ₁₁ - Σ₁₂Σ₂₂⁻¹Σ₂₁")
    
    # Calculate Σ₂₂⁻¹Σ₂₁
    sigma22_inv_sigma21 = Sigma22_inv @ Sigma21
    print(f"Σ₂₂⁻¹Σ₂₁ = \n{Sigma22_inv} @ {Sigma21} = {sigma22_inv_sigma21}")
    
    # Calculate Σ₁₂Σ₂₂⁻¹Σ₂₁
    sigma12_sigma22_inv_sigma21 = Sigma12 @ sigma22_inv_sigma21
    print(f"Σ₁₂Σ₂₂⁻¹Σ₂₁ = {Sigma12} @ {sigma22_inv_sigma21} = {sigma12_sigma22_inv_sigma21:.4f}")
    
    # Calculate conditional variance
    sigma1_given_23 = Sigma11 - sigma12_sigma22_inv_sigma21
    print(f"Σ₁|₂₃ = {Sigma11} - {sigma12_sigma22_inv_sigma21:.4f} = {sigma1_given_23:.4f}")
    
    # Calculate conditional standard deviation
    sigma1_given_23_std = np.sqrt(sigma1_given_23)
    print(f"σ₁|₂₃ = √{sigma1_given_23:.4f} = {sigma1_given_23_std:.4f}")
    
    # Summarize conditional distribution given X₂ and X₃
    print("\nSolution to part a:")
    print(f"X₁|(X₂={x23_value[0]}, X₃={x23_value[1]}) ~ N({mu1_given_23:.4f}, {sigma1_given_23:.4f})")
    
    # Part b: Conditional distribution given only X₂=8
    print("\nPart b: Find the best prediction for X₁ given only X₂ = 8")
    
    # Extract parameters for X₁ given X₂
    mu1_b = mu[0]
    mu2_b = mu[1]
    sigma11_b = cov[0, 0]
    sigma12_b = cov[0, 1]
    sigma21_b = cov[1, 0]
    sigma22_b = cov[1, 1]
    
    print("\nExtract relevant parameters:")
    print(f"μ₁ = {mu1_b}")
    print(f"μ₂ = {mu2_b}")
    print(f"σ₁₁ = {sigma11_b}")
    print(f"σ₁₂ = σ₂₁ = {sigma12_b}")
    print(f"σ₂₂ = {sigma22_b}")
    
    # Calculate regression coefficient
    beta_b = sigma12_b / sigma22_b
    print(f"\nRegression coefficient: β = σ₁₂/σ₂₂ = {sigma12_b}/{sigma22_b} = {beta_b:.4f}")
    
    # Calculate conditional mean
    x2_value_b = x23_value[0]
    mu1_given_2 = mu1_b + beta_b * (x2_value_b - mu2_b)
    print(f"μ₁|₂ = μ₁ + β(x₂ - μ₂) = {mu1_b} + {beta_b:.4f}({x2_value_b} - {mu2_b})")
    print(f"μ₁|₂ = {mu1_b} + {beta_b:.4f} × {x2_value_b - mu2_b}")
    print(f"μ₁|₂ = {mu1_b} + {beta_b * (x2_value_b - mu2_b):.4f}")
    print(f"μ₁|₂ = {mu1_given_2:.4f}")
    
    # Calculate conditional variance
    sigma1_given_2 = sigma11_b - (sigma12_b**2 / sigma22_b)
    print(f"\nσ₁|₂² = σ₁₁ - σ₁₂²/σ₂₂ = {sigma11_b} - {sigma12_b}²/{sigma22_b}")
    print(f"σ₁|₂² = {sigma11_b} - {sigma12_b**2}/{sigma22_b}")
    print(f"σ₁|₂² = {sigma11_b} - {sigma12_b**2 / sigma22_b:.4f}")
    print(f"σ₁|₂² = {sigma1_given_2:.4f}")
    
    # Calculate conditional standard deviation
    sigma1_given_2_std = np.sqrt(sigma1_given_2)
    print(f"σ₁|₂ = √{sigma1_given_2:.4f} = {sigma1_given_2_std:.4f}")
    
    # Summarize conditional distribution given only X₂
    print("\nSolution to part b:")
    print(f"The best prediction for X₁ given only X₂ = {x2_value_b} is: {mu1_given_2:.4f}")
    print(f"The conditional distribution is: X₁|(X₂={x2_value_b}) ~ N({mu1_given_2:.4f}, {sigma1_given_2:.4f})")
    
    # Part c: Calculate variance reduction
    print("\nPart c: Calculate the reduction in variance")
    
    # Calculate absolute reductions
    abs_reduction_step1 = Sigma11 - sigma1_given_2
    abs_reduction_step2 = sigma1_given_2 - sigma1_given_23
    abs_reduction_total = Sigma11 - sigma1_given_23
    
    # Calculate percentage reductions
    pct_reduction_step1 = (abs_reduction_step1 / Sigma11) * 100
    pct_reduction_step2 = (abs_reduction_step2 / sigma1_given_2) * 100
    pct_reduction_total = (abs_reduction_total / Sigma11) * 100
    
    print("\nStep 1: Reduction from unconditional to conditioning on X₂")
    print(f"Absolute reduction: {Sigma11} - {sigma1_given_2:.4f} = {abs_reduction_step1:.4f}")
    print(f"Percentage reduction: {abs_reduction_step1:.4f}/{Sigma11} × 100% = {pct_reduction_step1:.2f}%")
    
    print("\nStep 2: Additional reduction when also conditioning on X₃")
    print(f"Absolute reduction: {sigma1_given_2:.4f} - {sigma1_given_23:.4f} = {abs_reduction_step2:.4f}")
    print(f"Percentage reduction: {abs_reduction_step2:.4f}/{sigma1_given_2:.4f} × 100% = {pct_reduction_step2:.2f}%")
    
    print("\nTotal reduction from unconditional to conditioning on both X₂ and X₃")
    print(f"Absolute reduction: {Sigma11} - {sigma1_given_23:.4f} = {abs_reduction_total:.4f}")
    print(f"Percentage reduction: {abs_reduction_total:.4f}/{Sigma11} × 100% = {pct_reduction_total:.2f}%")
    
    # Calculate squared multiple correlation
    r_squared = abs_reduction_total / Sigma11
    print(f"\nSquared multiple correlation coefficient: R² = {r_squared:.4f}")
    print(f"This means that knowing both X₂ and X₃ explains {r_squared * 100:.2f}% of the variability in X₁")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 1. Visualize marginal vs conditional distributions
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Range for X₁
    x1_range = np.linspace(0, 10, 1000)
    
    # Calculate distributions
    x1_marginal = stats.norm.pdf(x1_range, mu1, np.sqrt(Sigma11))
    x1_cond_given_2 = stats.norm.pdf(x1_range, mu1_given_2, np.sqrt(sigma1_given_2))
    x1_cond_given_23 = stats.norm.pdf(x1_range, mu1_given_23, np.sqrt(sigma1_given_23))
    
    # Plot distributions
    ax.plot(x1_range, x1_marginal, 'b-', linewidth=2, label=f'Marginal: X₁ ~ N({mu1}, {Sigma11})')
    ax.plot(x1_range, x1_cond_given_2, 'g-', linewidth=2, 
           label=f'Conditional on X₂: X₁|(X₂={x2_value_b}) ~ N({mu1_given_2:.4f}, {sigma1_given_2:.4f})')
    ax.plot(x1_range, x1_cond_given_23, 'r-', linewidth=2, 
           label=f'Conditional on X₂,X₃: X₁|(X₂={x23_value[0]},X₃={x23_value[1]}) ~ N({mu1_given_23:.4f}, {sigma1_given_23:.4f})')
    
    # Add mean lines
    ax.axvline(x=mu1, color='blue', linestyle='--', alpha=0.5)
    ax.axvline(x=mu1_given_2, color='green', linestyle='--', alpha=0.5)
    ax.axvline(x=mu1_given_23, color='red', linestyle='--', alpha=0.5)
    
    # Add annotations
    ax.annotate(f'Variance reduction with X₂: {pct_reduction_step1:.2f}%', 
               xy=(0.05, 0.95), xycoords='axes fraction',
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
               fontsize=12)
    ax.annotate(f'Additional reduction with X₃: {pct_reduction_step2:.2f}%', 
               xy=(0.05, 0.89), xycoords='axes fraction',
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
               fontsize=12)
    ax.annotate(f'Total variance reduction: {pct_reduction_total:.2f}%', 
               xy=(0.05, 0.83), xycoords='axes fraction',
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
               fontsize=12)
    
    # Add labels and title
    ax.set_title('Marginal vs Conditional Distributions of X₁', fontsize=16)
    ax.set_xlabel('X₁', fontsize=14)
    ax.set_ylabel('Probability Density', fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=12)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'example2_conditional_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure saved to example2_conditional_distributions.png")
    
    # 2. Create bar chart comparing variances
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data for bar chart
    labels = ['Unconditional\nVar(X₁)', 'Conditional\nVar(X₁|X₂)', 'Conditional\nVar(X₁|X₂,X₃)']
    variances = [Sigma11, sigma1_given_2, sigma1_given_23]
    colors = ['blue', 'green', 'red']
    
    # Create bar chart
    bars = ax.bar(labels, variances, color=colors, alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Add variance reduction arrows and text
    ax.annotate('', xy=(0.5, Sigma11*0.6), xytext=(0, Sigma11*0.6),
                arrowprops=dict(arrowstyle='<->', color='black'))
    ax.text(0.25, Sigma11*0.65, f'{abs_reduction_step1:.2f} ({pct_reduction_step1:.1f}%)', 
           ha='center', fontsize=10)
    
    ax.annotate('', xy=(1.5, sigma1_given_2*0.6), xytext=(1, sigma1_given_2*0.6),
                arrowprops=dict(arrowstyle='<->', color='black'))
    ax.text(1.25, sigma1_given_2*0.65, f'{abs_reduction_step2:.2f} ({pct_reduction_step2:.1f}%)', 
           ha='center', fontsize=10)
    
    ax.annotate('', xy=(2, Sigma11*0.2), xytext=(0, Sigma11*0.2),
                arrowprops=dict(arrowstyle='<->', color='black', linestyle='--'))
    ax.text(1, Sigma11*0.25, f'Total reduction: {abs_reduction_total:.2f} ({pct_reduction_total:.1f}%)', 
           ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    # Add labels and title
    ax.set_title('Comparison of Variances', fontsize=16)
    ax.set_ylabel('Variance', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'example2_variance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure saved to example2_variance_comparison.png")
    
    return mu, cov, mu1_given_23, sigma1_given_23

def example3_prediction_conditional_inference():
    """Example 3: Prediction and Conditional Inference"""
    print_section_header("EXAMPLE 3: PREDICTION AND CONDITIONAL INFERENCE")
    
    print("Problem Statement:")
    print("A professor wants to predict a student's final exam score based on midterm and homework scores.")
    print("From historical data, scores follow a trivariate normal distribution with parameters:")
    print("- Mean vector: μ = (82, 78, 85) (Final, Midterm, Homework)")
    print("- Covariance matrix: Σ = [[100, 60, 40], [60, 64, 30], [40, 30, 25]]")
    print("\nIf a student scores 85 on the midterm and 90 on homework:")
    print("- What is the predicted final exam score?")
    print("- Provide a 95% prediction interval for the student's final exam score.")
    
    # Define parameters
    mu = np.array([82, 78, 85])  # Final, Midterm, Homework
    cov = np.array([
        [100, 60, 40],
        [60, 64, 30],
        [40, 30, 25]
    ])
    observed = np.array([85, 90])  # Midterm, Homework
    
    # Step 0: Analyze correlation structure
    print("\nStep 0: Analyze correlation structure")
    
    # Calculate correlations
    sigma_sqrt = np.sqrt(np.diag(cov))
    
    rho_fm = cov[0, 1] / (sigma_sqrt[0] * sigma_sqrt[1])
    rho_fh = cov[0, 2] / (sigma_sqrt[0] * sigma_sqrt[2])
    rho_mh = cov[1, 2] / (sigma_sqrt[1] * sigma_sqrt[2])
    
    print(f"Correlation between Final and Midterm: ρ_FM = {cov[0,1]}/√({cov[0,0]}×{cov[1,1]})")
    print(f"  = {cov[0,1]}/√{cov[0,0]*cov[1,1]}")
    print(f"  = {cov[0,1]}/{np.sqrt(cov[0,0]*cov[1,1]):.4f} = {rho_fm:.4f}")
    
    print(f"\nCorrelation between Final and Homework: ρ_FH = {cov[0,2]}/√({cov[0,0]}×{cov[2,2]})")
    print(f"  = {cov[0,2]}/√{cov[0,0]*cov[2,2]}")
    print(f"  = {cov[0,2]}/{np.sqrt(cov[0,0]*cov[2,2]):.4f} = {rho_fh:.4f}")
    
    print(f"\nCorrelation between Midterm and Homework: ρ_MH = {cov[1,2]}/√({cov[1,1]}×{cov[2,2]})")
    print(f"  = {cov[1,2]}/√{cov[1,1]*cov[2,2]}")
    print(f"  = {cov[1,2]}/{np.sqrt(cov[1,1]*cov[2,2]):.4f} = {rho_mh:.4f}")
    
    print("\nCorrelation analysis:")
    print(f"- Final exam has a strong positive correlation of {rho_fm:.2f} with midterm scores")
    print(f"- Final exam has an even stronger correlation of {rho_fh:.2f} with homework scores")
    print(f"- Midterm has a strong correlation of {rho_mh:.2f} with homework scores")
    print("These strong positive correlations suggest both scores are good predictors of final exam performance.")
    
    # Step 1: Partition the variables
    print("\nStep 1: Partition the variables for conditional distribution")
    
    # Partition mean vector
    mu1 = mu[0]  # Final
    mu2 = mu[1:]  # Midterm and Homework
    
    # Partition covariance matrix
    sigma11 = cov[0, 0]  # Variance of Final
    sigma12 = cov[0, 1:]  # Covariance between Final and (Midterm, Homework)
    sigma21 = cov[1:, 0]  # Transpose of sigma12
    sigma22 = cov[1:, 1:]  # Covariance matrix of Midterm and Homework
    
    print(f"μ₁ = {mu1} (mean of Final)")
    print(f"μ₂ = {mu2} (mean of Midterm and Homework)")
    print(f"σ₁₁ = {sigma11} (variance of Final)")
    print(f"σ₁₂ = {sigma12} (covariance between Final and (Midterm, Homework))")
    print(f"σ₂₁ = {sigma21} (transpose of σ₁₂)")
    print(f"σ₂₂ = \n{sigma22} (covariance matrix of Midterm and Homework)")
    
    # Step 2: Calculate sigma22_inv
    print("\nStep 2: Calculate σ₂₂⁻¹")
    
    # Calculate determinant
    det_sigma22 = np.linalg.det(sigma22)
    print(f"Determinant of σ₂₂: |σ₂₂| = {det_sigma22}")
    
    # Calculate adjugate matrix
    adj_sigma22 = np.array([
        [sigma22[1, 1], -sigma22[0, 1]],
        [-sigma22[1, 0], sigma22[0, 0]]
    ])
    print(f"Adjugate matrix of σ₂₂: \n{adj_sigma22}")
    
    # Calculate inverse
    sigma22_inv = adj_sigma22 / det_sigma22
    print(f"Inverse of σ₂₂: σ₂₂⁻¹ = \n{sigma22_inv}")
    
    # Verify with numpy
    print(f"Verification with numpy: \n{np.linalg.inv(sigma22)}")
    
    # Step 3: Calculate deviation from mean
    print("\nStep 3: Calculate deviation from mean")
    x2_minus_mu2 = observed - mu2
    print(f"x₂ - μ₂ = {observed} - {mu2} = {x2_minus_mu2}")
    
    # Step 4: Calculate conditional mean (predicted final score)
    print("\nStep 4: Calculate conditional mean (predicted final score)")
    
    # Calculate sigma22_inv * (x2 - mu2)
    temp = sigma22_inv @ x2_minus_mu2
    print(f"σ₂₂⁻¹(x₂ - μ₂) = \n{sigma22_inv} @ {x2_minus_mu2} = {temp}")
    
    # Calculate sigma12 * sigma22_inv * (x2 - mu2)
    adjustment = sigma12 @ temp
    print(f"σ₁₂σ₂₂⁻¹(x₂ - μ₂) = {sigma12} @ {temp} = {adjustment:.4f}")
    
    # Calculate conditional mean
    mu1_given_2 = mu1 + adjustment
    print(f"μ₁|₂ = μ₁ + σ₁₂σ₂₂⁻¹(x₂ - μ₂) = {mu1} + {adjustment:.4f} = {mu1_given_2:.4f}")
    
    # Step 5: Calculate conditional variance
    print("\nStep 5: Calculate conditional variance")
    
    # Calculate sigma12 * sigma22_inv * sigma21
    temp2 = sigma22_inv @ sigma21
    print(f"σ₂₂⁻¹σ₂₁ = \n{sigma22_inv} @ {sigma21} = {temp2}")
    
    variance_reduction = sigma12 @ temp2
    print(f"σ₁₂σ₂₂⁻¹σ₂₁ = {sigma12} @ {temp2} = {variance_reduction:.4f}")
    
    # Calculate conditional variance
    sigma1_given_2 = sigma11 - variance_reduction
    print(f"σ₁|₂² = σ₁₁ - σ₁₂σ₂₂⁻¹σ₂₁ = {sigma11} - {variance_reduction:.4f} = {sigma1_given_2:.4f}")
    
    # Calculate conditional standard deviation
    sigma1_given_2_std = np.sqrt(sigma1_given_2)
    print(f"σ₁|₂ = √{sigma1_given_2:.4f} = {sigma1_given_2_std:.4f}")
    
    # Step 6: Calculate 95% prediction interval
    print("\nStep 6: Calculate 95% prediction interval")
    
    # For 95% confidence, use z = 1.96
    z = 1.96
    interval_lower = mu1_given_2 - z * sigma1_given_2_std
    interval_upper = mu1_given_2 + z * sigma1_given_2_std
    
    print(f"95% prediction interval: μ₁|₂ ± 1.96σ₁|₂")
    print(f"  = {mu1_given_2:.4f} ± 1.96 × {sigma1_given_2_std:.4f}")
    print(f"  = {mu1_given_2:.4f} ± {z * sigma1_given_2_std:.4f}")
    print(f"  = [{interval_lower:.4f}, {interval_upper:.4f}]")
    
    # Step 7: Analysis of variance explained
    print("\nStep 7: Analysis of variance explained")
    
    # Calculate variance reduction
    absolute_reduction = sigma11 - sigma1_given_2
    percentage_reduction = (absolute_reduction / sigma11) * 100
    
    print(f"Variance reduction: {sigma11} - {sigma1_given_2:.4f} = {absolute_reduction:.4f}")
    print(f"Proportion of variance explained (R²): {absolute_reduction:.4f}/{sigma11} = {absolute_reduction/sigma11:.4f}")
    print(f"Percentage of variance explained: {percentage_reduction:.2f}%")
    
    # Calculate multiple correlation coefficient
    R = np.sqrt(absolute_reduction / sigma11)
    print(f"Multiple correlation coefficient: R = √{absolute_reduction/sigma11:.4f} = {R:.4f}")
    
    # Step 8: Express as regression equation
    print("\nStep 8: Express as a regression equation")
    
    # Calculate regression coefficients
    beta = sigma12 @ sigma22_inv
    beta1, beta2 = beta
    
    # Calculate intercept
    beta0 = mu1 - beta @ mu2
    
    print(f"Regression coefficients: β = σ₁₂σ₂₂⁻¹ = {sigma12} @ \n{sigma22_inv} = [{beta1:.4f}, {beta2:.4f}]")
    print(f"Intercept: β₀ = μ₁ - β·μ₂ = {mu1} - [{beta1:.4f}, {beta2:.4f}]·{mu2} = {beta0:.4f}")
    print("\nRegression equation:")
    print(f"Final = {beta0:.4f} + {beta1:.4f} × Midterm + {beta2:.4f} × Homework")
    
    # Verification
    verification = beta0 + beta1 * observed[0] + beta2 * observed[1]
    print(f"\nVerification: {beta0:.4f} + {beta1:.4f} × {observed[0]} + {beta2:.4f} × {observed[1]} = {verification:.4f}")
    print(f"This matches our earlier calculation of the conditional mean: {mu1_given_2:.4f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 1. Visualize conditional vs marginal distribution
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Range for final exam score
    x_range = np.linspace(40, 120, 1000)
    
    # Calculate distributions
    x_marginal = stats.norm.pdf(x_range, mu1, np.sqrt(sigma11))
    x_conditional = stats.norm.pdf(x_range, mu1_given_2, np.sqrt(sigma1_given_2))
    
    # Plot distributions
    ax.plot(x_range, x_marginal, 'b-', linewidth=2, 
           label=f'Marginal: Final ~ N({mu1}, {sigma11})')
    ax.plot(x_range, x_conditional, 'r-', linewidth=2, 
           label=f'Conditional: Final|(Midterm={observed[0]}, Homework={observed[1]}) ~ N({mu1_given_2:.1f}, {sigma1_given_2:.1f})')
    
    # Add mean lines
    ax.axvline(x=mu1, color='blue', linestyle='--', alpha=0.7, label=f'Marginal mean: {mu1}')
    ax.axvline(x=mu1_given_2, color='red', linestyle='--', alpha=0.7, label=f'Predicted score: {mu1_given_2:.1f}')
    
    # Shade prediction interval
    ax.axvspan(interval_lower, interval_upper, alpha=0.2, color='red', 
              label=f'95% prediction interval: [{interval_lower:.1f}, {interval_upper:.1f}]')
    
    # Add annotations
    ax.annotate(f'Variance reduction: {percentage_reduction:.1f}%', 
               xy=(0.05, 0.95), xycoords='axes fraction',
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
               fontsize=12)
    
    ax.annotate(f'Multiple correlation (R): {R:.4f}', 
               xy=(0.05, 0.89), xycoords='axes fraction',
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
               fontsize=12)
    
    # Add labels and title
    ax.set_title('Final Exam Score Prediction', fontsize=16)
    ax.set_xlabel('Final Exam Score', fontsize=14)
    ax.set_ylabel('Probability Density', fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'example3_score_prediction.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure saved to example3_score_prediction.png")
    
    # 2. Create 3D visualization of relationship
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create grid for midterm and homework scores
    midterm_range = np.linspace(60, 100, 20)
    homework_range = np.linspace(70, 100, 20)
    midterm_grid, homework_grid = np.meshgrid(midterm_range, homework_range)
    
    # Calculate predicted final scores for each combination
    final_predictions = np.zeros_like(midterm_grid)
    
    for i in range(midterm_grid.shape[0]):
        for j in range(midterm_grid.shape[1]):
            # Current midterm and homework scores
            current_scores = np.array([midterm_grid[i, j], homework_grid[i, j]])
            # Calculate deviation from mean
            current_deviation = current_scores - mu2
            # Calculate predicted final score
            final_predictions[i, j] = mu1 + sigma12 @ sigma22_inv @ current_deviation
    
    # Plot surface
    surf = ax.plot_surface(midterm_grid, homework_grid, final_predictions, 
                          cmap='viridis', alpha=0.8, rstride=1, cstride=1)
    
    # Add point for the specific student
    ax.scatter([observed[0]], [observed[1]], [mu1_given_2], 
              c='red', s=200, marker='o', label='Student scores')
    
    # Add vertical line from surface to point
    ax.plot([observed[0], observed[0]], [observed[1], observed[1]], 
           [min(final_predictions.flatten()), mu1_given_2], 'r-', linewidth=2)
    
    # Add labels and annotations
    ax.set_xlabel('Midterm Score', fontsize=14)
    ax.set_ylabel('Homework Score', fontsize=14)
    ax.set_zlabel('Predicted Final Score', fontsize=14)
    ax.set_title('Prediction Surface for Final Exam Scores', fontsize=16)
    
    # Add text annotation for student prediction
    ax.text(observed[0], observed[1], mu1_given_2+2, 
           f'Predicted: {mu1_given_2:.1f}', color='black', fontsize=12,
           bbox=dict(facecolor='white', alpha=0.7))
    
    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10)
    cbar.set_label('Predicted Final Score', fontsize=12)
    
    # Add regression equation as text
    equation_text = f'Final = {beta0:.1f} + {beta1:.2f} × Midterm + {beta2:.2f} × Homework'
    ax.text2D(0.05, 0.95, equation_text, transform=ax.transAxes, fontsize=14,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'example3_prediction_surface.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure saved to example3_prediction_surface.png")
    
    return mu, cov, mu1_given_2, sigma1_given_2, interval_lower, interval_upper


def main():
    """Main function to run all examples"""
    print("\n" + "="*80)
    print(" CONDITIONAL DISTRIBUTIONS EXAMPLES ".center(80, "="))
    print("="*80 + "\n")
    
    print("This script provides detailed step-by-step solutions for conditional distribution examples.")
    print("Each example includes calculations, explanations, and visualizations.\n")
    
    # Run Example 1
    example1_result = example1_bivariate_normal_conditional()
    
    # Run Example 2
    example2_result = example2_trivariate_normal_conditional()
    
    # Run Example 3
    example3_result = example3_prediction_conditional_inference()
    
    print("\n" + "="*80)
    print(" SUMMARY OF RESULTS ".center(80, "="))
    print("="*80 + "\n")
    
    print("Example 1: Bivariate Normal Conditional Distribution")
    print(f"- Conditional distribution: X₁|(X₂=7) ~ N({example1_result[2]:.4f}, {example1_result[3]:.4f})")
    
    print("\nExample 2: Trivariate Normal Conditional Distribution")
    print(f"- Conditional distribution: X₁|(X₂=8, X₃=11) ~ N({example2_result[2]:.4f}, {example2_result[3]:.4f})")
    
    print("\nExample 3: Final Exam Score Prediction")
    print(f"- Predicted score: {example3_result[2]:.4f}")
    print(f"- 95% prediction interval: [{example3_result[4]:.4f}, {example3_result[5]:.4f}]")
    
    print("\nAll examples completed. Check the Images/Conditional_Distribution directory for visualizations.")


if __name__ == "__main__":
    main()