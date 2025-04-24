import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
import matplotlib.cm as cm

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
    """Plot enhanced bivariate normal distribution with correlation ellipses and confidence regions"""
    # Create grid of points
    x = np.linspace(-5, 10, 100)
    y = np.linspace(-5, 15, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    # Calculate PDF values
    rv = stats.multivariate_normal(mu, cov)
    Z = rv.pdf(pos)
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(15, 6))
    
    # 3D surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)
    
    # Add contours at the bottom of 3D plot
    offset = Z.min()
    ax1.contour(X, Y, Z, zdir='z', offset=offset, levels=10, cmap='viridis', alpha=0.5)
    
    # Add conditioning plane at x2 = 7
    x2_value = 7
    x1_plane = np.linspace(-5, 10, 100)
    z_plane = np.linspace(offset, Z.max(), 100)
    X1_plane, Z_plane = np.meshgrid(x1_plane, z_plane)
    Y_plane = np.full_like(X1_plane, x2_value)
    ax1.plot_surface(X1_plane, Y_plane, Z_plane, alpha=0.3, color='red')
    
    # Add mean point and vertical line to peak
    ax1.scatter([mu[0]], [mu[1]], [Z.max()], color='red', s=100, marker='*')
    ax1.plot([mu[0], mu[0]], [mu[1], mu[1]], [offset, Z.max()], 'r--', alpha=0.5)
    
    # Add correlation structure visualization
    if cov[0,1] != 0:
        # Add regression line at the bottom
        slope = cov[0,1] / cov[1,1]
        intercept = mu[0] - slope * mu[1]
        x_line = np.array([mu[1] - 2*np.sqrt(cov[1,1]), mu[1] + 2*np.sqrt(cov[1,1])])
        y_line = slope * x_line + intercept
        ax1.plot(y_line, x_line, [offset, offset], 'r-', linewidth=2, alpha=0.7)
    
    # Set labels and title
    ax1.set_xlabel('X₁')
    ax1.set_ylabel('X₂')
    ax1.set_zlabel('Probability Density')
    ax1.set_title('3D PDF Surface with Conditioning Plane')
    ax1.view_init(elev=30, azim=45)
    
    # 2D contour plot with confidence regions
    ax2 = fig.add_subplot(122)
    
    # Plot contours
    ax2.contour(X, Y, Z, levels=15, cmap='viridis')
    
    # Add conditioning line
    ax2.axhline(y=x2_value, color='red', linestyle='--', alpha=0.7, label='Conditioning on X₂=7')
    
    # Add mean point
    ax2.scatter(mu[0], mu[1], color='red', s=100, marker='*', label='Mean')
    
    # Add regression line if there's correlation
    if cov[0,1] != 0:
        slope = cov[0,1] / cov[1,1]
        intercept = mu[0] - slope * mu[1]
        x_line = np.array([mu[1] - 2*np.sqrt(cov[1,1]), mu[1] + 2*np.sqrt(cov[1,1])])
        y_line = slope * x_line + intercept
        ax2.plot(y_line, x_line, 'r-', linewidth=2, alpha=0.7, label='Regression Line')
    
    # Set labels and title
    ax2.set_xlabel('X₁')
    ax2.set_ylabel('X₂')
    ax2.set_title('Contour Plot with Conditioning Line')
    ax2.grid(True, alpha=0.15, linestyle='--')
    ax2.set_aspect('equal')
    ax2.legend()
    
    # Add colorbar
    plt.colorbar(surf, ax=ax1, label='Probability Density')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {filename}")

def plot_conditional_steps(mu, cov, x2_value, mu1_given_2, sigma1_given_2, filename_prefix):
    """Create step-by-step visualizations explaining conditional distributions"""
    
    # Step 1: Joint Distribution with Conditioning Line
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create grid of points
    x = np.linspace(-5, 10, 100)
    y = np.linspace(-5, 15, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    # Calculate PDF values
    rv = stats.multivariate_normal(mu, cov)
    Z = rv.pdf(pos)
    
    # Plot contours
    contour = ax.contour(X, Y, Z, levels=15, cmap='viridis')
    plt.colorbar(contour, label='Joint Probability Density')
    
    # Add conditioning line
    ax.axhline(y=x2_value, color='red', linestyle='--', linewidth=2, 
               label=f'Conditioning on X₂={x2_value}')
    
    # Add mean point
    ax.scatter(mu[0], mu[1], color='red', s=100, marker='*', label='Mean (μ₁,μ₂)')
    
    # Add regression line
    slope = cov[0,1] / cov[1,1]
    intercept = mu[0] - slope * mu[1]
    x_line = np.array([mu[1] - 2*np.sqrt(cov[1,1]), mu[1] + 2*np.sqrt(cov[1,1])])
    y_line = slope * x_line + intercept
    ax.plot(y_line, x_line, 'g-', linewidth=2, alpha=0.7, label='E[X₁|X₂]')
    
    # Set labels and title
    ax.set_xlabel('X₁', fontsize=14)
    ax.set_ylabel('X₂', fontsize=14)
    ax.set_title('Step 1: Joint Distribution and Conditioning', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, f'{filename_prefix}_step1.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Step 2: Slice at X₂=x2_value
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate conditional distribution
    x1_range = np.linspace(-5, 10, 1000)
    conditional_pdf = stats.norm.pdf(x1_range, mu1_given_2, np.sqrt(sigma1_given_2))
    
    # Plot the slice
    ax.plot(x1_range, conditional_pdf, 'r-', linewidth=2.5, 
            label=f'P(X₁|X₂={x2_value})')
    
    # Add mean line
    ax.axvline(mu1_given_2, color='red', linestyle='--', linewidth=2,
               label=f'Conditional Mean (μ₁|₂={mu1_given_2:.2f})')
    
    # Add confidence interval
    ci_lower = mu1_given_2 - 1.96 * np.sqrt(sigma1_given_2)
    ci_upper = mu1_given_2 + 1.96 * np.sqrt(sigma1_given_2)
    ax.fill_between(x1_range, conditional_pdf, where=(x1_range >= ci_lower) & (x1_range <= ci_upper),
                   alpha=0.3, color='red', label='95% Interval')
    
    # Set labels and title
    ax.set_xlabel('X₁', fontsize=14)
    ax.set_ylabel('Conditional Probability Density', fontsize=14)
    ax.set_title('Step 2: Conditional Distribution at X₂=7', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, f'{filename_prefix}_step2.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Step 3: Variance Reduction Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate marginal distribution
    marginal_pdf = stats.norm.pdf(x1_range, mu[0], np.sqrt(cov[0,0]))
    
    # Plot both distributions
    ax.plot(x1_range, marginal_pdf, 'b-', linewidth=2.5, alpha=0.8,
            label=f'Marginal: X₁~N({mu[0]}, {cov[0,0]})')
    ax.plot(x1_range, conditional_pdf, 'r-', linewidth=2.5, alpha=0.8,
            label=f'Conditional: X₁|X₂={x2_value}~N({mu1_given_2:.2f}, {sigma1_given_2:.2f})')
    
    # Fill areas to show variance reduction
    ax.fill_between(x1_range, marginal_pdf, alpha=0.2, color='blue')
    ax.fill_between(x1_range, conditional_pdf, alpha=0.2, color='red')
    
    # Add means
    ax.axvline(mu[0], color='blue', linestyle='--', alpha=0.8,
               label=f'Marginal Mean (μ₁={mu[0]})')
    ax.axvline(mu1_given_2, color='red', linestyle='--', alpha=0.8,
               label=f'Conditional Mean (μ₁|₂={mu1_given_2:.2f})')
    
    # Add variance reduction annotation
    variance_reduction = (cov[0,0] - sigma1_given_2) / cov[0,0] * 100
    ax.text(0.02, 0.98, f'Variance Reduction: {variance_reduction:.1f}%',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set labels and title
    ax.set_xlabel('X₁', fontsize=14)
    ax.set_ylabel('Probability Density', fontsize=14)
    ax.set_title('Step 3: Variance Reduction through Conditioning', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, f'{filename_prefix}_step3.png'), dpi=300, bbox_inches='tight')
    plt.close()

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
    
    # Create step-by-step visualizations
    print("\nCreating step-by-step visualizations...")
    plot_conditional_steps(mu, cov, x2_value, mu1_given_2, sigma1_given_2, 'example1')
    
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
    
    # Calculate determinant
    det_Sigma22 = Sigma22[0, 0] * Sigma22[1, 1] - Sigma22[0, 1] * Sigma22[1, 0]
    print(f"Determinant of Σ₂₂: |Σ₂₂| = {det_Sigma22}")
    
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
    
    # Create step-by-step visualizations
    print("\nCreating step-by-step visualizations...")
    plot_trivariate_steps(mu, cov, x23_value[0], x23_value[1], 
                         mu1_given_2, sigma1_given_2,
                         mu1_given_23, sigma1_given_23, 'example2')
    
    return mu, cov, mu1_given_23, sigma1_given_23

def plot_trivariate_steps(mu, cov, x2, x3, mu1_given_2, sigma1_given_2, mu1_given_23, sigma1_given_23, filename_prefix):
    """Create step-by-step visualizations for trivariate case"""
    
    # Step 1: Variance Reduction Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate distributions
    x1_range = np.linspace(mu[0]-4*np.sqrt(cov[0,0]), mu[0]+4*np.sqrt(cov[0,0]), 1000)
    marginal = stats.norm.pdf(x1_range, mu[0], np.sqrt(cov[0,0]))
    cond_x2 = stats.norm.pdf(x1_range, mu1_given_2, np.sqrt(sigma1_given_2))
    cond_x2x3 = stats.norm.pdf(x1_range, mu1_given_23, np.sqrt(sigma1_given_23))
    
    # Plot distributions
    ax.plot(x1_range, marginal, 'b-', linewidth=2.5, alpha=0.8,
            label=f'Marginal: X₁')
    ax.plot(x1_range, cond_x2, 'g-', linewidth=2.5, alpha=0.8,
            label=f'Given X₂={x2}')
    ax.plot(x1_range, cond_x2x3, 'r-', linewidth=2.5, alpha=0.8,
            label=f'Given X₂={x2}, X₃={x3}')
    
    # Fill areas
    ax.fill_between(x1_range, marginal, alpha=0.2, color='blue')
    ax.fill_between(x1_range, cond_x2, alpha=0.2, color='green')
    ax.fill_between(x1_range, cond_x2x3, alpha=0.2, color='red')
    
    # Add means
    ax.axvline(mu[0], color='blue', linestyle='--', alpha=0.8)
    ax.axvline(mu1_given_2, color='green', linestyle='--', alpha=0.8)
    ax.axvline(mu1_given_23, color='red', linestyle='--', alpha=0.8)
    
    # Add variance reduction annotations
    var_red_x2 = (cov[0,0] - sigma1_given_2) / cov[0,0] * 100
    var_red_x2x3 = (cov[0,0] - sigma1_given_23) / cov[0,0] * 100
    additional_red = (sigma1_given_2 - sigma1_given_23) / sigma1_given_2 * 100
    
    ax.text(0.02, 0.98, f'Variance Reduction:\nWith X₂: {var_red_x2:.1f}%\nWith X₂,X₃: {var_red_x2x3:.1f}%\nAdditional from X₃: {additional_red:.1f}%',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set labels and title
    ax.set_xlabel('X₁', fontsize=14)
    ax.set_ylabel('Probability Density', fontsize=14)
    ax.set_title('Progressive Variance Reduction through Conditioning', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, f'{filename_prefix}_variance_reduction.png'), dpi=300, bbox_inches='tight')
    plt.close()

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
    print(f"  = {cov[0,1]}/{np.sqrt(cov[0,0]*cov[1,1]):.8f} = {rho_fm:.8f}")
    
    print(f"\nCorrelation between Final and Homework: ρ_FH = {cov[0,2]}/√({cov[0,0]}×{cov[2,2]})")
    print(f"  = {cov[0,2]}/√{cov[0,0]*cov[2,2]}")
    print(f"  = {cov[0,2]}/{np.sqrt(cov[0,0]*cov[2,2]):.8f} = {rho_fh:.8f}")
    
    print(f"\nCorrelation between Midterm and Homework: ρ_MH = {cov[1,2]}/√({cov[1,1]}×{cov[2,2]})")
    print(f"  = {cov[1,2]}/√{cov[1,1]*cov[2,2]}")
    print(f"  = {cov[1,2]}/{np.sqrt(cov[1,1]*cov[2,2]):.8f} = {rho_mh:.8f}")
    
    print("\nCorrelation analysis:")
    print(f"- Final exam has a strong positive correlation of {rho_fm:.8f} with midterm scores")
    print(f"- Final exam has an even stronger correlation of {rho_fh:.8f} with homework scores")
    print(f"- Midterm has a strong correlation of {rho_mh:.8f} with homework scores")
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
    print(f"Determinant of σ₂₂: |σ₂₂| = {det_sigma22:.16f}")
    
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
    adjustment = np.dot(sigma12, temp)  # Using np.dot for more precise calculation
    print(f"σ₁₂σ₂₂⁻¹(x₂ - μ₂) = {sigma12} @ {temp} = {adjustment:.8f}")
    
    # Calculate conditional mean
    mu1_given_2 = mu1 + adjustment
    print(f"μ₁|₂ = μ₁ + σ₁₂σ₂₂⁻¹(x₂ - μ₂) = {mu1} + {adjustment:.8f} = {mu1_given_2:.8f}")
    
    # Step 5: Calculate conditional variance
    print("\nStep 5: Calculate conditional variance")
    
    # Calculate sigma22_inv * sigma21
    temp2 = sigma22_inv @ sigma21
    print(f"σ₂₂⁻¹σ₂₁ = \n{sigma22_inv} @ {sigma21} = {temp2}")
    
    # Calculate sigma12 * sigma22_inv * sigma21
    variance_reduction = np.dot(sigma12, temp2)  # Using np.dot for more precise calculation
    print(f"σ₁₂σ₂₂⁻¹σ₂₁ = {sigma12} @ {temp2} = {variance_reduction:.8f}")
    
    # Calculate conditional variance
    sigma1_given_2 = sigma11 - variance_reduction
    print(f"σ₁|₂² = σ₁₁ - σ₁₂σ₂₂⁻¹σ₂₁ = {sigma11} - {variance_reduction:.8f} = {sigma1_given_2:.8f}")
    
    # Calculate conditional standard deviation
    sigma1_given_2_std = np.sqrt(sigma1_given_2)
    print(f"σ₁|₂ = √{sigma1_given_2:.8f} = {sigma1_given_2_std:.8f}")
    
    # Step 6: Calculate 95% prediction interval
    print("\nStep 6: Calculate 95% prediction interval")
    
    # For 95% confidence, use z = 1.96
    z = 1.96
    interval_lower = mu1_given_2 - z * sigma1_given_2_std
    interval_upper = mu1_given_2 + z * sigma1_given_2_std
    
    print(f"95% prediction interval: μ₁|₂ ± 1.96σ₁|₂")
    print(f"  = {mu1_given_2:.8f} ± 1.96 × {sigma1_given_2_std:.8f}")
    print(f"  = {mu1_given_2:.8f} ± {z * sigma1_given_2_std:.8f}")
    print(f"  = [{interval_lower:.8f}, {interval_upper:.8f}]")
    
    # Step 7: Analysis of variance explained
    print("\nStep 7: Analysis of variance explained")
    
    # Calculate variance reduction
    absolute_reduction = sigma11 - sigma1_given_2
    percentage_reduction = (absolute_reduction / sigma11) * 100
    
    print(f"Variance reduction: {sigma11} - {sigma1_given_2:.8f} = {absolute_reduction:.8f}")
    print(f"Proportion of variance explained (R²): {absolute_reduction:.8f}/{sigma11} = {absolute_reduction/sigma11:.8f}")
    print(f"Percentage of variance explained: {percentage_reduction:.8f}%")
    
    # Calculate multiple correlation coefficient
    R = np.sqrt(absolute_reduction / sigma11)
    print(f"Multiple correlation coefficient: R = √{absolute_reduction/sigma11:.8f} = {R:.8f}")
    
    # Step 8: Express as regression equation
    print("\nStep 8: Express as a regression equation")
    
    # Calculate regression coefficients
    beta = sigma12 @ sigma22_inv
    beta1, beta2 = beta
    
    # Calculate intercept
    beta0 = mu1 - beta @ mu2
    
    print(f"Regression coefficients: β = σ₁₂σ₂₂⁻¹ = {sigma12} @ \n{sigma22_inv} = [{beta1:.8f}, {beta2:.8f}]")
    print(f"Intercept: β₀ = μ₁ - β·μ₂ = {mu1} - [{beta1:.8f}, {beta2:.8f}]·{mu2} = {beta0:.8f}")
    print("\nRegression equation:")
    print(f"Final = {beta0:.8f} + {beta1:.8f} × Midterm + {beta2:.8f} × Homework")
    
    # Verification
    verification = beta0 + beta1 * observed[0] + beta2 * observed[1]
    print(f"\nVerification: {beta0:.8f} + {beta1:.8f} × {observed[0]} + {beta2:.8f} × {observed[1]} = {verification:.8f}")
    print(f"This matches our earlier calculation of the conditional mean: {mu1_given_2:.8f}")
    
    # Create step-by-step visualizations
    print("\nCreating step-by-step visualizations...")
    plot_exam_prediction_steps(mu, cov, observed[0], observed[1],
                             mu1_given_2, sigma1_given_2,
                             [interval_lower, interval_upper], 'example3')
    
    return mu, cov, mu1_given_2, sigma1_given_2, interval_lower, interval_upper

def plot_exam_prediction_steps(mu, cov, midterm, homework, mu_pred, sigma_pred, interval, filename_prefix):
    """Create step-by-step visualizations for exam score prediction"""
    
    # Step 1: Correlation Structure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create correlation matrix visualization
    sigma_sqrt = np.sqrt(np.diag(cov))
    corr = cov / (sigma_sqrt[:, None] * sigma_sqrt[None, :])
    
    im = ax.imshow(corr, cmap='RdYlBu', vmin=-1, vmax=1)
    plt.colorbar(im, label='Correlation')
    
    # Add correlation values
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center',
                   color='black' if abs(corr[i,j]) < 0.7 else 'white')
    
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(['Final', 'Midterm', 'Homework'])
    ax.set_yticklabels(['Final', 'Midterm', 'Homework'])
    ax.set_title('Step 1: Score Correlation Structure', fontsize=16)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, f'{filename_prefix}_step1.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Step 2: Prediction Distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate distribution
    x_range = np.linspace(mu_pred-4*np.sqrt(sigma_pred), mu_pred+4*np.sqrt(sigma_pred), 1000)
    pred_pdf = stats.norm.pdf(x_range, mu_pred, np.sqrt(sigma_pred))
    
    # Plot distribution
    ax.plot(x_range, pred_pdf, 'b-', linewidth=2.5)
    
    # Add prediction interval
    ax.fill_between(x_range, pred_pdf, 
                   where=(x_range >= interval[0]) & (x_range <= interval[1]),
                   alpha=0.3, color='blue', 
                   label='95% Prediction Interval')
    
    # Add mean line
    ax.axvline(mu_pred, color='red', linestyle='--', linewidth=2,
               label=f'Predicted Score: {mu_pred:.1f}')
    
    # Add annotations
    ax.text(0.02, 0.98, 
            f'Student Scores:\nMidterm: {midterm}\nHomework: {homework}\n\nPrediction Interval:\n[{interval[0]:.1f}, {interval[1]:.1f}]',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set labels and title
    ax.set_xlabel('Final Exam Score', fontsize=14)
    ax.set_ylabel('Probability Density', fontsize=14)
    ax.set_title('Step 2: Final Exam Score Prediction', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, f'{filename_prefix}_step2.png'), dpi=300, bbox_inches='tight')
    plt.close()

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