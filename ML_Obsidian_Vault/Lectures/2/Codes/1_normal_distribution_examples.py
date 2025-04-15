import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images/Normal_Distribution relative to the parent directory
images_dir = os.path.join(parent_dir, "Images", "Normal_Distribution")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

def plot_normal_distribution(mu, sigma, title, xlabel, ylabel, filename):
    """Helper function to plot normal distribution with annotations"""
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    y = stats.norm.pdf(x, mu, sigma)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.fill_between(x, y, alpha=0.2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    
    # Add mean and standard deviation lines
    plt.axvline(x=mu, color='r', linestyle='--', label=f'Mean (μ) = {mu}')
    plt.axvline(x=mu + sigma, color='g', linestyle=':', label=f'μ + σ = {mu + sigma:.2f}')
    plt.axvline(x=mu - sigma, color='g', linestyle=':', label=f'μ - σ = {mu - sigma:.2f}')
    plt.legend()
    
    plt.savefig(os.path.join(images_dir, f'{filename}.png'))
    plt.close()

def example1_coin_toss():
    """Example 1: Coin Toss Analysis"""
    print("\n=== Example 1: Coin Toss Analysis ===")
    n = 100  # number of tosses
    p = 0.5  # probability of heads
    
    # Calculate mean and standard deviation
    mu = n * p
    sigma = np.sqrt(n * p * (1 - p))
    
    print(f"Number of tosses (n): {n}")
    print(f"Probability of heads (p): {p}")
    print(f"Mean number of heads (μ): {mu}")
    print(f"Standard deviation (σ): {sigma:.2f}")
    
    # Plot the distribution
    plot_normal_distribution(mu, sigma, 
                           "Number of Heads in 100 Coin Tosses",
                           "Number of Heads",
                           "Probability Density",
                           "coin_toss")

def example2_dice_roll():
    """Example 2: Dice Roll Analysis"""
    print("\n=== Example 2: Dice Roll Analysis ===")
    n = 60  # number of rolls
    p = 1/6  # probability of rolling a 6
    
    # Calculate mean and standard deviation
    mu = n * p
    sigma = np.sqrt(n * p * (1 - p))
    
    print(f"Number of rolls (n): {n}")
    print(f"Probability of rolling a 6 (p): {p:.4f}")
    print(f"Expected number of 6's (μ): {mu}")
    print(f"Standard deviation (σ): {sigma:.2f}")
    
    # Plot the distribution
    plot_normal_distribution(mu, sigma, 
                           "Number of 6's in 60 Dice Rolls",
                           "Number of 6's",
                           "Probability Density",
                           "dice_roll")

def example3_height_distribution():
    """Example 3: Height Distribution Analysis"""
    print("\n=== Example 3: Height Distribution Analysis ===")
    mu = 175  # mean height in cm
    sigma = 7  # standard deviation in cm
    
    # a) Probability of being taller than 185 cm
    z_score = (185 - mu) / sigma
    prob_taller = 1 - stats.norm.cdf(z_score)
    print(f"\na) Probability of being taller than 185 cm:")
    print(f"Z-score = (185 - {mu}) / {sigma} = {z_score:.2f}")
    print(f"P(X > 185) = P(Z > {z_score:.2f}) = {prob_taller:.4f} or {prob_taller*100:.2f}%")
    
    # b) Probability between 170 cm and 180 cm
    z_lower = (170 - mu) / sigma
    z_upper = (180 - mu) / sigma
    prob_between = stats.norm.cdf(z_upper) - stats.norm.cdf(z_lower)
    print(f"\nb) Probability between 170 cm and 180 cm:")
    print(f"Z_lower = (170 - {mu}) / {sigma} = {z_lower:.2f}")
    print(f"Z_upper = (180 - {mu}) / {sigma} = {z_upper:.2f}")
    print(f"P(170 < X < 180) = P({z_lower:.2f} < Z < {z_upper:.2f}) = {prob_between:.4f} or {prob_between*100:.2f}%")
    
    # c) 90th percentile
    z_90 = stats.norm.ppf(0.90)
    height_90 = mu + z_90 * sigma
    print(f"\nc) 90th percentile height:")
    print(f"Z_90 = Φ⁻¹(0.90) = {z_90:.2f}")
    print(f"Height_90 = {mu} + {z_90:.2f} × {sigma} = {height_90:.2f} cm")
    
    # Plot the distribution
    plot_normal_distribution(mu, sigma, 
                           "Adult Male Height Distribution",
                           "Height (cm)",
                           "Probability Density",
                           "height_distribution")

def example4_quality_control():
    """Example 4: Quality Control in Manufacturing"""
    print("\n=== Example 4: Quality Control in Manufacturing ===")
    mu = 10  # mean diameter in mm
    sigma = 0.05  # standard deviation in mm
    lower_spec = 9.9
    upper_spec = 10.1
    
    # a) Percentage within specifications
    z_lower = (lower_spec - mu) / sigma
    z_upper = (upper_spec - mu) / sigma
    prob_within = stats.norm.cdf(z_upper) - stats.norm.cdf(z_lower)
    print(f"\na) Percentage within specifications:")
    print(f"Z_lower = (9.9 - {mu}) / {sigma} = {z_lower:.2f}")
    print(f"Z_upper = (10.1 - {mu}) / {sigma} = {z_upper:.2f}")
    print(f"P(9.9 ≤ X ≤ 10.1) = P({z_lower:.2f} ≤ Z ≤ {z_upper:.2f}) = {prob_within:.4f} or {prob_within*100:.2f}%")
    
    # b) Expected number of non-conforming rods in 100
    prob_outside = 1 - prob_within
    expected_non_conforming = 100 * prob_outside
    print(f"\nb) Expected non-conforming rods in 100:")
    print(f"Probability outside specs = 1 - {prob_within:.4f} = {prob_outside:.4f}")
    print(f"Expected non-conforming = 100 × {prob_outside:.4f} = {expected_non_conforming:.2f} rods")
    
    # c) Maximum allowable standard deviation for 99% within specs
    z_99 = stats.norm.ppf(0.995)  # 0.5% in each tail
    max_sigma = (upper_spec - mu) / z_99
    print(f"\nc) Maximum allowable standard deviation:")
    print(f"Z_99 = Φ⁻¹(0.995) = {z_99:.2f}")
    print(f"Maximum σ = (10.1 - 10) / {z_99:.2f} = {max_sigma:.4f} mm")
    
    # Plot the distribution with specification limits
    plt.figure(figsize=(10, 6))
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    y = stats.norm.pdf(x, mu, sigma)
    plt.plot(x, y, 'b-', linewidth=2)
    plt.fill_between(x, y, alpha=0.2)
    plt.axvline(x=lower_spec, color='r', linestyle='--', label='Lower Spec (9.9 mm)')
    plt.axvline(x=upper_spec, color='r', linestyle='--', label='Upper Spec (10.1 mm)')
    plt.title("Rod Diameter Distribution with Specification Limits")
    plt.xlabel("Diameter (mm)")
    plt.ylabel("Probability Density")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(images_dir, 'quality_control.png'))
    plt.close()

def example5_central_limit_theorem():
    """Example 5: Central Limit Theorem in Machine Learning"""
    print("\n=== Example 5: Central Limit Theorem in Machine Learning ===")
    mu = 50  # population mean
    sigma = 15  # population standard deviation
    n = 25  # sample size
    
    # a) Distribution of sample mean
    sigma_xbar = sigma / np.sqrt(n)
    print(f"\na) Distribution of sample mean:")
    print(f"μ_X̄ = μ = {mu}")
    print(f"σ_X̄ = σ/√n = {sigma}/√{n} = {sigma_xbar:.2f}")
    print(f"Therefore, X̄ ~ N({mu}, {sigma_xbar:.2f}²)")
    
    # b) Probability between 45 and 55
    z_lower = (45 - mu) / sigma_xbar
    z_upper = (55 - mu) / sigma_xbar
    prob_between = stats.norm.cdf(z_upper) - stats.norm.cdf(z_lower)
    print(f"\nb) Probability between 45 and 55:")
    print(f"Z_lower = (45 - {mu}) / {sigma_xbar:.2f} = {z_lower:.2f}")
    print(f"Z_upper = (55 - {mu}) / {sigma_xbar:.2f} = {z_upper:.2f}")
    print(f"P(45 ≤ X̄ ≤ 55) = P({z_lower:.2f} ≤ Z ≤ {z_upper:.2f}) = {prob_between:.4f} or {prob_between*100:.2f}%")
    
    # c) Required sample size for 95% probability within ±2 units
    z_95 = stats.norm.ppf(0.975)  # 2.5% in each tail
    required_n = (z_95 * sigma / 2)**2
    print(f"\nc) Required sample size for 95% probability within ±2 units:")
    print(f"Z_95 = Φ⁻¹(0.975) = {z_95:.2f}")
    print(f"Required n = ({z_95:.2f} × {sigma} / 2)² = {required_n:.2f}")
    print(f"Rounding up to n = {np.ceil(required_n):.0f}")
    
    # Plot the sampling distribution
    plt.figure(figsize=(10, 6))
    x = np.linspace(mu - 4*sigma_xbar, mu + 4*sigma_xbar, 1000)
    y = stats.norm.pdf(x, mu, sigma_xbar)
    plt.plot(x, y, 'b-', linewidth=2)
    plt.fill_between(x, y, alpha=0.2)
    plt.axvline(x=mu - 2, color='r', linestyle='--', label='μ - 2')
    plt.axvline(x=mu + 2, color='r', linestyle='--', label='μ + 2')
    plt.title("Sampling Distribution of the Mean (n=25)")
    plt.xlabel("Sample Mean")
    plt.ylabel("Probability Density")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(images_dir, 'central_limit_theorem.png'))
    plt.close()

if __name__ == "__main__":
    print("Normal Distribution Examples")
    print("===========================")
    
    example1_coin_toss()
    example2_dice_roll()
    example3_height_distribution()
    example4_quality_control()
    example5_central_limit_theorem()
    
    print(f"\nAll examples completed. Check the {images_dir} directory for visualizations.") 