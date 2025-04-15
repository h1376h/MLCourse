import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
import os

def calculate_binomial_probabilities():
    """Calculate all probabilities for the binomial problem"""
    n = 10  # number of trials
    p = 0.6  # probability of success (brown eyes)
    
    # 1. Probability of exactly 7 people with brown eyes
    p_exactly_7 = binom.pmf(7, n, p)
    
    # 2. Probability of at least 8 people with brown eyes
    p_at_least_8 = 1 - binom.cdf(7, n, p)
    
    # 3. Expected number of people with brown eyes
    expected_value = n * p
    
    # 4. Standard deviation
    std_dev = np.sqrt(n * p * (1 - p))
    
    return {
        'p_exactly_7': p_exactly_7,
        'p_at_least_8': p_at_least_8,
        'expected_value': expected_value,
        'std_dev': std_dev
    }

def plot_probability_mass_function(save_path=None):
    """Plot the probability mass function for the binomial distribution"""
    n = 10
    p = 0.6
    x = np.arange(0, n + 1)
    y = binom.pmf(x, n, p)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(x, y, color='skyblue', edgecolor='black')
    
    # Highlight the bars for exactly 7 and at least 8
    for i, bar in enumerate(bars):
        if i == 7:
            bar.set_color('lightblue')
        elif i >= 8:
            bar.set_color('lightgreen')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if height > 0.01:  # Only label significant probabilities
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom')
    
    ax.set_xlabel('Number of People with Brown Eyes')
    ax.set_ylabel('Probability')
    ax.set_title('Binomial Probability Mass Function (n=10, p=0.6)')
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', edgecolor='black', label='Exactly 7'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightgreen', edgecolor='black', label='At least 8'),
        plt.Rectangle((0, 0), 1, 1, facecolor='skyblue', edgecolor='black', label='Other')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_cumulative_distribution(save_path=None):
    """Plot the cumulative distribution function"""
    n = 10
    p = 0.6
    x = np.arange(0, n + 1)
    y = binom.cdf(x, n, p)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.step(x, y, 'r-', where='post', linewidth=2)
    
    # Add markers at key points
    ax.plot(x, y, 'ro', markersize=8)
    
    # Add horizontal line at P(X ≤ 7)
    ax.axhline(y=binom.cdf(7, n, p), color='blue', linestyle='--', alpha=0.5)
    ax.text(0, binom.cdf(7, n, p), f'P(X ≤ 7) = {binom.cdf(7, n, p):.3f}',
            ha='left', va='bottom')
    
    ax.set_xlabel('Number of People with Brown Eyes')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution Function (n=10, p=0.6)')
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_normal_approximation(save_path=None):
    """Plot the normal approximation to the binomial distribution"""
    n = 10
    p = 0.6
    mu = n * p
    sigma = np.sqrt(n * p * (1 - p))
    
    x = np.linspace(0, 10, 1000)
    y = binom.pmf(np.round(x), n, p)
    
    # Normal approximation
    x_norm = np.linspace(0, 10, 1000)
    y_norm = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x_norm - mu)/sigma)**2)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, y, width=0.1, color='skyblue', edgecolor='black', alpha=0.5, label='Binomial PMF')
    ax.plot(x_norm, y_norm, 'r-', linewidth=2, label='Normal Approximation')
    
    # Add vertical lines for mean and standard deviations
    ax.axvline(x=mu, color='green', linestyle='--', label=f'Mean = {mu:.1f}')
    ax.axvline(x=mu - sigma, color='orange', linestyle=':', label=f'Mean ± σ')
    ax.axvline(x=mu + sigma, color='orange', linestyle=':')
    
    ax.set_xlabel('Number of People with Brown Eyes')
    ax.set_ylabel('Probability Density')
    ax.set_title('Binomial Distribution and Normal Approximation')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def main():
    """Generate all visualizations for Question 4"""
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_1_Quiz_4")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 4 of the L2.1 Probability quiz...")
    
    # Calculate probabilities
    probs = calculate_binomial_probabilities()
    print("\nCalculated Values:")
    print(f"P(Exactly 7) = {probs['p_exactly_7']:.4f}")
    print(f"P(At least 8) = {probs['p_at_least_8']:.4f}")
    print(f"Expected Value = {probs['expected_value']:.4f}")
    print(f"Standard Deviation = {probs['std_dev']:.4f}")
    
    # Generate visualizations
    plot_probability_mass_function(save_path=os.path.join(save_dir, "pmf.png"))
    print("1. Probability mass function visualization created")
    
    plot_cumulative_distribution(save_path=os.path.join(save_dir, "cdf.png"))
    print("2. Cumulative distribution function visualization created")
    
    plot_normal_approximation(save_path=os.path.join(save_dir, "normal_approx.png"))
    print("3. Normal approximation visualization created")
    
    print(f"\nAll visualizations have been saved to: {save_dir}")

if __name__ == "__main__":
    main() 