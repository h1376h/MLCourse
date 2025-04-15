import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import os

def power_law_pdf(x, c):
    """Probability density function for the given distribution"""
    return c * x**2 if 0 <= x <= 2 else 0

def find_constant_c():
    """Find the value of c that makes the PDF valid"""
    # The integral of the PDF from 0 to 2 must equal 1
    def integrand(x):
        return x**2
    
    integral, _ = quad(integrand, 0, 2)
    c = 1 / integral
    return c

def calculate_probability(a, b, c):
    """Calculate P(a ≤ X ≤ b)"""
    def integrand(x):
        return power_law_pdf(x, c)
    
    probability, _ = quad(integrand, a, b)
    return probability

def calculate_expected_value(c):
    """Calculate E[X]"""
    def integrand(x):
        return x * power_law_pdf(x, c)
    
    expected_value, _ = quad(integrand, 0, 2)
    return expected_value

def calculate_variance(c):
    """Calculate Var(X) = E[X²] - (E[X])²"""
    # Calculate E[X²]
    def integrand_x2(x):
        return x**2 * power_law_pdf(x, c)
    
    ex2, _ = quad(integrand_x2, 0, 2)
    
    # Calculate E[X]
    ex = calculate_expected_value(c)
    
    # Calculate variance
    variance = ex2 - ex**2
    return variance

def plot_pdf(c, save_path=None):
    """Plot the probability density function"""
    x = np.linspace(-0.5, 2.5, 1000)
    y = [power_law_pdf(xi, c) for xi in x]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, 'b-', linewidth=2)
    
    # Shade the area for P(0.5 ≤ X ≤ 1.5)
    x_shade = np.linspace(0.5, 1.5, 100)
    y_shade = [power_law_pdf(xi, c) for xi in x_shade]
    ax.fill_between(x_shade, y_shade, alpha=0.3, color='blue')
    
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Probability Density Function')
    ax.grid(True, alpha=0.3)
    
    # Add text for the shaded area
    ax.text(1, 0.2, 'P(0.5 ≤ X ≤ 1.5)', ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_cdf(c, save_path=None):
    """Plot the cumulative distribution function"""
    x = np.linspace(-0.5, 2.5, 1000)
    y = np.zeros_like(x)
    
    for i, xi in enumerate(x):
        if xi < 0:
            y[i] = 0
        elif xi > 2:
            y[i] = 1
        else:
            y[i], _ = quad(lambda t: power_law_pdf(t, c), 0, xi)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, 'r-', linewidth=2)
    
    ax.set_xlabel('x')
    ax.set_ylabel('F(x)')
    ax.set_title('Cumulative Distribution Function')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def main():
    """Generate all visualizations for Question 3"""
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_1_Quiz_3")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 3 of the L2.1 Probability quiz...")
    
    # Find the constant c
    c = find_constant_c()
    print(f"\nConstant c = {c:.4f}")
    
    # Calculate P(0.5 ≤ X ≤ 1.5)
    p_range = calculate_probability(0.5, 1.5, c)
    print(f"P(0.5 ≤ X ≤ 1.5) = {p_range:.4f}")
    
    # Calculate expected value and variance
    expected_value = calculate_expected_value(c)
    variance = calculate_variance(c)
    print(f"E[X] = {expected_value:.4f}")
    print(f"Var(X) = {variance:.4f}")
    
    # Generate visualizations
    plot_pdf(c, save_path=os.path.join(save_dir, "pdf.png"))
    print("1. PDF visualization created")
    
    plot_cdf(c, save_path=os.path.join(save_dir, "cdf.png"))
    print("2. CDF visualization created")
    
    print(f"\nAll visualizations have been saved to: {save_dir}")

if __name__ == "__main__":
    main() 