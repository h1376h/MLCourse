import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import os
import seaborn as sns

def example3_probabilities_linear_transformations():
    """
    Example 3: Calculating Probabilities of Linear Transformations
    
    Problem Statement:
    Given a multivariate normal random vector X = [X₁, X₂, X₃, X₄]ᵀ ~ N₄(μ, Σ) with:
    
    μ = [15, 30, 7, 10]ᵀ and Σ = [
        [3, -4, 0, 2],
        [-4, 1, 2, 1],
        [0, 2, 9, 9],
        [2, 1, 9, 1]
    ]
    
    Find the following probabilities:
    a) P(X₁ - 5X₄ < 16)
    b) P(3X₂ - 4X₃ > 35)
    c) P(7X₁ + 3X₂ + 2X₃ < 56)
    """
    print("\n" + "="*80)
    print("Example 3: Calculating Probabilities of Linear Transformations")
    print("="*80)
    
    # Define given parameters
    mu = np.array([15, 30, 7, 10])
    Sigma = np.array([
        [3, -4, 0, 2],
        [-4, 1, 2, 1],
        [0, 2, 9, 9],
        [2, 1, 9, 1]
    ])
    
    print("\nGiven:")
    print(f"Mean vector μ = {mu}")
    print(f"Covariance matrix Σ = \n{Sigma}")
    
    # Verify that Sigma is symmetric (important property of covariance matrices)
    is_symmetric = np.allclose(Sigma, Sigma.T)
    print(f"\nVerification: Is covariance matrix symmetric? {is_symmetric}")
    
    # Key concept explanation
    print("\n" + "-"*60)
    print("Key Concept: Linear Transformations of Multivariate Normal")
    print("-"*60)
    print("For a linear transformation Y = a^T X of a multivariate normal X ~ N(μ, Σ),")
    print("the resulting random variable Y follows a univariate normal distribution with:")
    print("  Mean: μ_Y = a^T μ")
    print("  Variance: σ²_Y = a^T Σ a")
    print("\nThis allows us to convert questions about linear combinations of multivariate normal")
    print("variables into questions about univariate normal distributions, which we can solve")
    print("using the standard normal CDF (cumulative distribution function) Φ.")
    
    # Create images directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    images_dir = os.path.join(parent_dir, "Images", "Linear_Transformations")
    os.makedirs(images_dir, exist_ok=True)
    
    # Track saved image paths
    saved_images = []
    
    # (a) P(X₁ - 5X₄ < 16)
    print("\n" + "-"*60)
    print("(a) Finding P(X₁ - 5X₄ < 16):")
    print("-"*60)
    
    # Define the coefficient vector for Y₁ = X₁ - 5X₄
    a1 = np.array([1, 0, 0, -5])
    print(f"The linear combination Y₁ = X₁ - 5X₄ corresponds to coefficient vector:")
    print(f"a₁ = {a1}")
    
    # Calculate mean of Y₁ - detailed calculation
    print("\nStep 1: Calculate the mean of Y₁ using μ_Y₁ = a₁ᵀμ")
    print(f"μ_Y₁ = a₁ᵀμ = {a1[0]}×{mu[0]} + {a1[1]}×{mu[1]} + {a1[2]}×{mu[2]} + {a1[3]}×{mu[3]}")
    print(f"μ_Y₁ = {a1[0]*mu[0]} + {a1[1]*mu[1]} + {a1[2]*mu[2]} + {a1[3]*mu[3]}")
    mu_Y1 = np.dot(a1, mu)
    print(f"μ_Y₁ = {mu_Y1}")
    
    # Calculate variance of Y₁ - detailed calculation
    print("\nStep 2: Calculate the variance of Y₁ using σ²_Y₁ = a₁ᵀΣa₁")
    
    # First, calculate Σa₁
    print("\nStep 2.1: First calculate Σa₁:")
    Sigma_a1 = np.dot(Sigma, a1)
    print(f"Σa₁ = Σ·a₁ = \n{Sigma} · {a1} = {Sigma_a1}")
    
    # Then, calculate a₁ᵀ(Σa₁)
    print("\nStep 2.2: Then calculate a₁ᵀ(Σa₁):")
    print(f"a₁ᵀ(Σa₁) = {a1[0]}×{Sigma_a1[0]} + {a1[1]}×{Sigma_a1[1]} + {a1[2]}×{Sigma_a1[2]} + {a1[3]}×{Sigma_a1[3]}")
    
    # Calculate detailed terms
    terms = [a1[i] * Sigma_a1[i] for i in range(4)]
    print(f"a₁ᵀ(Σa₁) = {terms[0]} + {terms[1]} + {terms[2]} + {terms[3]}")
    
    sigma2_Y1 = np.dot(a1, Sigma_a1)
    print(f"σ²_Y₁ = {sigma2_Y1}")
    
    # Calculate standard deviation
    sigma_Y1 = np.sqrt(sigma2_Y1)
    print(f"σ_Y₁ = √{sigma2_Y1} = {sigma_Y1}")
    
    # Calculate the standardized value - detailed calculation
    print("\nStep 3: Standardize the threshold by calculating z₁ = (16 - μ_Y₁)/σ_Y₁")
    z1 = (16 - mu_Y1) / sigma_Y1
    print(f"z₁ = (16 - {mu_Y1})/{sigma_Y1} = {16 - mu_Y1}/{sigma_Y1} = {z1}")
    
    # Calculate the probability using standard normal CDF
    print("\nStep 4: Find the probability using the standard normal CDF")
    prob_a = norm.cdf(z1)
    print(f"P(Y₁ < 16) = P(Z < {z1}) = Φ({z1}) = {prob_a}")
    
    # Create visualization for part (a)
    plt.figure(figsize=(10, 6))
    x = np.linspace(mu_Y1 - 4*sigma_Y1, mu_Y1 + 4*sigma_Y1, 1000)
    y = norm.pdf(x, mu_Y1, sigma_Y1)
    
    plt.plot(x, y, 'b-', linewidth=2, label=f'Normal({mu_Y1:.1f}, {sigma_Y1:.1f}²)')
    
    # Shade area representing the probability
    x_fill = np.linspace(mu_Y1 - 4*sigma_Y1, 16, 1000)
    y_fill = norm.pdf(x_fill, mu_Y1, sigma_Y1)
    plt.fill_between(x_fill, y_fill, color='skyblue', alpha=0.4, label=f'P(Y1 < 16) = {prob_a:.4f}')
    
    # Add vertical line at the threshold
    plt.axvline(x=16, color='r', linestyle='--', label='Threshold = 16')
    
    plt.title(f'Probability Distribution for Y1 = X1 - 5X4')
    plt.xlabel('Y1')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    save_path_a = os.path.join(images_dir, "example3_probability_a.png")
    plt.savefig(save_path_a, bbox_inches='tight', dpi=300)
    print(f"\nVisualization for part (a) saved to: {save_path_a}")
    plt.close()
    saved_images.append(save_path_a)
    
    # (b) P(3X₂ - 4X₃ > 35)
    print("\n" + "-"*60)
    print("(b) Finding P(3X₂ - 4X₃ > 35):")
    print("-"*60)
    
    # Define the coefficient vector for Y₂ = 3X₂ - 4X₃
    a2 = np.array([0, 3, -4, 0])
    print(f"The linear combination Y₂ = 3X₂ - 4X₃ corresponds to coefficient vector:")
    print(f"a₂ = {a2}")
    
    # Calculate mean of Y₂ - detailed calculation
    print("\nStep 1: Calculate the mean of Y₂ using μ_Y₂ = a₂ᵀμ")
    print(f"μ_Y₂ = a₂ᵀμ = {a2[0]}×{mu[0]} + {a2[1]}×{mu[1]} + {a2[2]}×{mu[2]} + {a2[3]}×{mu[3]}")
    print(f"μ_Y₂ = {a2[0]*mu[0]} + {a2[1]*mu[1]} + {a2[2]*mu[2]} + {a2[3]*mu[3]}")
    mu_Y2 = np.dot(a2, mu)
    print(f"μ_Y₂ = {mu_Y2}")
    
    # Calculate variance of Y₂ - detailed calculation
    print("\nStep 2: Calculate the variance of Y₂ using σ²_Y₂ = a₂ᵀΣa₂")
    
    # First, calculate Σa₂
    print("\nStep 2.1: First calculate Σa₂:")
    Sigma_a2 = np.dot(Sigma, a2)
    print(f"Σa₂ = Σ·a₂ = {Sigma_a2}")
    
    # Then, calculate a₂ᵀ(Σa₂)
    print("\nStep 2.2: Then calculate a₂ᵀ(Σa₂):")
    print(f"a₂ᵀ(Σa₂) = {a2[0]}×{Sigma_a2[0]} + {a2[1]}×{Sigma_a2[1]} + {a2[2]}×{Sigma_a2[2]} + {a2[3]}×{Sigma_a2[3]}")
    
    # Calculate detailed terms
    terms = [a2[i] * Sigma_a2[i] for i in range(4)]
    print(f"a₂ᵀ(Σa₂) = {terms[0]} + {terms[1]} + {terms[2]} + {terms[3]}")
    
    sigma2_Y2 = np.dot(a2, Sigma_a2)
    print(f"σ²_Y₂ = {sigma2_Y2}")
    
    # Calculate standard deviation
    sigma_Y2 = np.sqrt(sigma2_Y2)
    print(f"σ_Y₂ = √{sigma2_Y2} = {sigma_Y2}")
    
    # Calculate the standardized value for Y₂ > 35 - detailed calculation
    print("\nStep 3: Standardize the threshold by calculating z₂ = (35 - μ_Y₂)/σ_Y₂")
    z2 = (35 - mu_Y2) / sigma_Y2
    print(f"z₂ = (35 - {mu_Y2})/{sigma_Y2} = {35 - mu_Y2}/{sigma_Y2} = {z2}")
    
    # Calculate the probability
    print("\nStep 4: Find the probability using the standard normal CDF")
    print(f"Note: Since we want P(Y₂ > 35) = P(Z > {z2}), we need to use 1 - Φ({z2})")
    prob_b = 1 - norm.cdf(z2)
    print(f"P(Y₂ > 35) = P(Z > {z2}) = 1 - Φ({z2}) = 1 - {norm.cdf(z2):.6f} = {prob_b}")
    
    # Create visualization for part (b)
    plt.figure(figsize=(10, 6))
    x = np.linspace(mu_Y2 - 4*sigma_Y2, mu_Y2 + 4*sigma_Y2, 1000)
    y = norm.pdf(x, mu_Y2, sigma_Y2)
    
    plt.plot(x, y, 'g-', linewidth=2, label=f'Normal({mu_Y2:.1f}, {sigma_Y2:.1f}²)')
    
    # Shade area representing the probability
    x_fill = np.linspace(35, mu_Y2 + 4*sigma_Y2, 1000)
    y_fill = norm.pdf(x_fill, mu_Y2, sigma_Y2)
    plt.fill_between(x_fill, y_fill, color='lightgreen', alpha=0.4, label=f'P(Y2 > 35) = {prob_b:.4f}')
    
    # Add vertical line at the threshold
    plt.axvline(x=35, color='r', linestyle='--', label='Threshold = 35')
    
    plt.title(f'Probability Distribution for Y2 = 3X2 - 4X3')
    plt.xlabel('Y2')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    save_path_b = os.path.join(images_dir, "example3_probability_b.png")
    plt.savefig(save_path_b, bbox_inches='tight', dpi=300)
    print(f"\nVisualization for part (b) saved to: {save_path_b}")
    plt.close()
    saved_images.append(save_path_b)
    
    # (c) P(7X₁ + 3X₂ + 2X₃ < 56)
    print("\n" + "-"*60)
    print("(c) Finding P(7X₁ + 3X₂ + 2X₃ < 56):")
    print("-"*60)
    
    # Define the coefficient vector for Y₃ = 7X₁ + 3X₂ + 2X₃
    a3 = np.array([7, 3, 2, 0])
    print(f"The linear combination Y₃ = 7X₁ + 3X₂ + 2X₃ corresponds to coefficient vector:")
    print(f"a₃ = {a3}")
    
    # Calculate mean of Y₃ - detailed calculation
    print("\nStep 1: Calculate the mean of Y₃ using μ_Y₃ = a₃ᵀμ")
    print(f"μ_Y₃ = a₃ᵀμ = {a3[0]}×{mu[0]} + {a3[1]}×{mu[1]} + {a3[2]}×{mu[2]} + {a3[3]}×{mu[3]}")
    print(f"μ_Y₃ = {a3[0]*mu[0]} + {a3[1]*mu[1]} + {a3[2]*mu[2]} + {a3[3]*mu[3]}")
    mu_Y3 = np.dot(a3, mu)
    print(f"μ_Y₃ = {mu_Y3}")
    
    # Calculate variance of Y₃ - detailed calculation
    print("\nStep 2: Calculate the variance of Y₃ using σ²_Y₃ = a₃ᵀΣa₃")
    
    # First, calculate Σa₃
    print("\nStep 2.1: First calculate Σa₃:")
    Sigma_a3 = np.dot(Sigma, a3)
    print(f"Σa₃ = Σ·a₃ = {Sigma_a3}")
    
    # Then, calculate a₃ᵀ(Σa₃)
    print("\nStep 2.2: Then calculate a₃ᵀ(Σa₃):")
    print(f"a₃ᵀ(Σa₃) = {a3[0]}×{Sigma_a3[0]} + {a3[1]}×{Sigma_a3[1]} + {a3[2]}×{Sigma_a3[2]} + {a3[3]}×{Sigma_a3[3]}")
    
    # Calculate detailed terms
    terms = [a3[i] * Sigma_a3[i] for i in range(4)]
    print(f"a₃ᵀ(Σa₃) = {terms[0]} + {terms[1]} + {terms[2]} + {terms[3]}")
    
    sigma2_Y3 = np.dot(a3, Sigma_a3)
    print(f"σ²_Y₃ = {sigma2_Y3}")
    
    # Calculate standard deviation
    sigma_Y3 = np.sqrt(sigma2_Y3)
    print(f"σ_Y₃ = √{sigma2_Y3} = {sigma_Y3}")
    
    # Calculate the standardized value - detailed calculation
    print("\nStep 3: Standardize the threshold by calculating z₃ = (56 - μ_Y₃)/σ_Y₃")
    z3 = (56 - mu_Y3) / sigma_Y3
    print(f"z₃ = (56 - {mu_Y3})/{sigma_Y3} = {56 - mu_Y3}/{sigma_Y3} = {z3}")
    
    # Calculate the probability
    print("\nStep 4: Find the probability using the standard normal CDF")
    prob_c = norm.cdf(z3)
    print(f"P(Y₃ < 56) = P(Z < {z3}) = Φ({z3}) = {prob_c}")
    
    # Create visualization for part (c)
    plt.figure(figsize=(10, 6))
    x = np.linspace(mu_Y3 - 4*sigma_Y3, mu_Y3 + 4*sigma_Y3, 1000)
    y = norm.pdf(x, mu_Y3, sigma_Y3)
    
    plt.plot(x, y, 'purple', linewidth=2, label=f'Normal({mu_Y3:.1f}, {sigma_Y3:.1f}²)')
    
    # Shade area representing the probability
    x_fill = np.linspace(mu_Y3 - 4*sigma_Y3, 56, 1000)
    y_fill = norm.pdf(x_fill, mu_Y3, sigma_Y3)
    plt.fill_between(x_fill, y_fill, color='lavender', alpha=0.4, label=f'P(Y3 < 56) = {prob_c:.4f}')
    
    # Add vertical line at the threshold
    plt.axvline(x=56, color='r', linestyle='--', label='Threshold = 56')
    
    plt.title(f'Probability Distribution for Y3 = 7X1 + 3X2 + 2X3')
    plt.xlabel('Y3')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    save_path_c = os.path.join(images_dir, "example3_probability_c.png")
    plt.savefig(save_path_c, bbox_inches='tight', dpi=300)
    print(f"\nVisualization for part (c) saved to: {save_path_c}")
    plt.close()
    saved_images.append(save_path_c)
    
    # Create a combined visualization of all three probability calculations
    plt.figure(figsize=(10, 6))
    
    # Add all three distributions to one plot
    x1 = np.linspace(mu_Y1 - 4*sigma_Y1, mu_Y1 + 4*sigma_Y1, 1000)
    y1 = norm.pdf(x1, mu_Y1, sigma_Y1)
    plt.plot(x1, y1, 'b-', linewidth=2, label=f'Y1 ~ N({mu_Y1:.1f}, {sigma_Y1:.1f}²)')
    
    x2 = np.linspace(mu_Y2 - 4*sigma_Y2, mu_Y2 + 4*sigma_Y2, 1000)
    y2 = norm.pdf(x2, mu_Y2, sigma_Y2)
    plt.plot(x2, y2, 'g-', linewidth=2, label=f'Y2 ~ N({mu_Y2:.1f}, {sigma_Y2:.1f}²)')
    
    x3 = np.linspace(mu_Y3 - 4*sigma_Y3, mu_Y3 + 4*sigma_Y3, 1000)
    y3 = norm.pdf(x3, mu_Y3, sigma_Y3)
    plt.plot(x3, y3, 'purple', linewidth=2, label=f'Y3 ~ N({mu_Y3:.1f}, {sigma_Y3:.1f}²)')
    
    # Add threshold lines
    plt.axvline(x=16, color='blue', linestyle='--', alpha=0.5)
    plt.axvline(x=35, color='green', linestyle='--', alpha=0.5)
    plt.axvline(x=56, color='purple', linestyle='--', alpha=0.5)
    
    plt.title('Probability Distributions of Linear Combinations')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the combined figure
    save_path_combined = os.path.join(images_dir, "example3_combined_distributions.png")
    plt.savefig(save_path_combined, bbox_inches='tight', dpi=300)
    print(f"\nCombined visualization saved to: {save_path_combined}")
    plt.close()
    saved_images.append(save_path_combined)
    
    # Summary
    print("\n" + "="*60)
    print("Summary of Example 3:")
    print("="*60)
    print(f"(a) P(X1 - 5X4 < 16) = {prob_a:.6f} ≈ {1 if prob_a > 0.9999 else prob_a:.4f}")
    print(f"(b) P(3X2 - 4X3 > 35) = {prob_b:.6f} ≈ {prob_b:.4f}")
    print(f"(c) P(7X1 + 3X2 + 2X3 < 56) = {prob_c:.6f} ≈ {0 if prob_c < 0.0001 else prob_c:.4f}")
    print("\nThe key insight is that linear combinations of multivariate normal variables")
    print("are also normally distributed, allowing us to convert to standard normal probabilities.")
    
    return saved_images

def plot_normal_distribution(ax, mu, sigma, threshold, title, inequality="<", idx=1):
    """Helper function to plot normal distributions with shaded areas"""
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    y = norm.pdf(x, mu, sigma)
    
    # Plot PDF
    ax.plot(x, y, linewidth=2, label=f'Y{idx} ~ N({mu:.1f}, {sigma:.1f}²)')
    
    # Shade area under the curve
    if inequality == "<":
        x_fill = np.linspace(mu - 4*sigma, threshold, 1000)
        color = 'skyblue' if idx == 1 else ('lavender' if idx == 3 else 'lightgreen')
        label = f'P(Y{idx} < {threshold})'
    else:
        x_fill = np.linspace(threshold, mu + 4*sigma, 1000)
        color = 'lightgreen'
        label = f'P(Y{idx} > {threshold})'
    
    y_fill = norm.pdf(x_fill, mu, sigma)
    ax.fill_between(x_fill, y_fill, color=color, alpha=0.4, label=label)
    
    # Add vertical line at the threshold
    ax.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold = {threshold}')
    
    ax.set_title(title)
    ax.set_xlabel(f'Y{idx}')
    ax.set_ylabel('Probability Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

if __name__ == "__main__":
    example3_probabilities_linear_transformations() 