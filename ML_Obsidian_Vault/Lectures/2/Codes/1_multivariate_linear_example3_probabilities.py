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
    
    # Calculate mean of Y₁
    mu_Y1 = np.dot(a1, mu)
    print(f"\nMean of Y1 = {mu_Y1}")
    
    # Calculate variance of Y₁
    Sigma_a1 = np.dot(Sigma, a1)
    sigma2_Y1 = np.dot(a1, Sigma_a1)
    sigma_Y1 = np.sqrt(sigma2_Y1)
    print(f"Variance of Y1 = {sigma2_Y1}")
    print(f"Standard deviation of Y1 = {sigma_Y1}")
    
    # Calculate the standardized value
    z1 = (16 - mu_Y1) / sigma_Y1
    print(f"\nStandardized value z1 = {z1}")
    
    # Calculate the probability using standard normal CDF
    prob_a = norm.cdf(z1)
    print(f"P(Y1 < 16) = P(Z < {z1}) = {prob_a}")
    
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
    
    # Calculate mean and variance
    mu_Y2 = np.dot(a2, mu)
    Sigma_a2 = np.dot(Sigma, a2)
    sigma2_Y2 = np.dot(a2, Sigma_a2)
    sigma_Y2 = np.sqrt(sigma2_Y2)
    
    print(f"\nMean of Y2 = {mu_Y2}")
    print(f"Variance of Y2 = {sigma2_Y2}")
    print(f"Standard deviation of Y2 = {sigma_Y2}")
    
    # Calculate the standardized value for Y₂ > 35
    z2 = (35 - mu_Y2) / sigma_Y2
    print(f"\nStandardized value z2 = {z2}")
    
    # Calculate the probability
    prob_b = 1 - norm.cdf(z2)
    print(f"P(Y2 > 35) = P(Z > {z2}) = 1 - Φ({z2}) = {prob_b}")
    
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
    
    # Calculate mean and variance
    mu_Y3 = np.dot(a3, mu)
    Sigma_a3 = np.dot(Sigma, a3)
    sigma2_Y3 = np.dot(a3, Sigma_a3)
    sigma_Y3 = np.sqrt(sigma2_Y3)
    
    print(f"\nMean of Y3 = {mu_Y3}")
    print(f"Variance of Y3 = {sigma2_Y3}")
    print(f"Standard deviation of Y3 = {sigma_Y3}")
    
    # Calculate the standardized value
    z3 = (56 - mu_Y3) / sigma_Y3
    print(f"\nStandardized value z3 = {z3}")
    
    # Calculate the probability
    prob_c = norm.cdf(z3)
    print(f"P(Y3 < 56) = P(Z < {z3}) = Φ({z3}) = {prob_c}")
    
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