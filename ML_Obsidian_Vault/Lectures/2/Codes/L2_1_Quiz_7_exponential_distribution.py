import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
import math

def verify_pdf_integrates_to_one(lambda_val):
    """
    Verify that the exponential PDF integrates to 1.
    
    This is done by calculating the integral of λe^(-λx) from 0 to infinity,
    which has a closed-form solution of 1.
    """
    # Theoretical result: integral of λe^(-λx) from 0 to infinity = 1
    theoretical_result = 1
    
    # We can also verify numerically for finite upper bound
    upper_bound = 10  # Large enough for practical purposes with λ=2
    x = np.linspace(0, upper_bound, 1000)
    pdf_values = lambda_val * np.exp(-lambda_val * x)
    numerical_result = np.trapz(pdf_values, x)
    
    return {
        'theoretical_result': theoretical_result,
        'numerical_result': numerical_result,
        'is_valid': abs(numerical_result - theoretical_result) < 1e-3
    }

def calculate_probability_greater_than(lambda_val, x_val):
    """
    Calculate P(X > x) for exponential distribution.
    
    For exponential distribution, P(X > x) = e^(-λx)
    """
    return np.exp(-lambda_val * x_val)

def calculate_probability_between(lambda_val, lower, upper):
    """
    Calculate P(lower < X < upper) for exponential distribution.
    
    For exponential distribution:
    P(lower < X < upper) = e^(-λ*lower) - e^(-λ*upper)
    """
    return np.exp(-lambda_val * lower) - np.exp(-lambda_val * upper)

def calculate_expected_value(lambda_val):
    """
    Calculate E[X] for exponential distribution.
    
    For exponential distribution, E[X] = 1/λ
    """
    return 1 / lambda_val

def calculate_variance(lambda_val):
    """
    Calculate Var(X) for exponential distribution.
    
    For exponential distribution, Var(X) = 1/λ²
    """
    return 1 / (lambda_val ** 2)

def calculate_median(lambda_val):
    """
    Calculate median of exponential distribution.
    
    For exponential distribution, median = ln(2)/λ
    """
    return np.log(2) / lambda_val

def plot_exponential_pdf(lambda_val, save_path=None):
    """
    Plot the probability density function (PDF) for the exponential distribution.
    """
    x = np.linspace(0, 3, 1000)  # x from 0 to 3
    pdf = lambda_val * np.exp(-lambda_val * x)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, pdf, 'b-', lw=2, label=f'PDF: λe^(-λx), λ={lambda_val}')
    
    # Fill areas for probabilities
    # P(X > 1)
    x_fill = np.linspace(1, 3, 100)
    pdf_fill = lambda_val * np.exp(-lambda_val * x_fill)
    ax.fill_between(x_fill, pdf_fill, alpha=0.3, color='green', label='P(X > 1)')
    
    # P(0.5 < X < 1.5)
    x_fill = np.linspace(0.5, 1.5, 100)
    pdf_fill = lambda_val * np.exp(-lambda_val * x_fill)
    ax.fill_between(x_fill, pdf_fill, alpha=0.3, color='red', label='P(0.5 < X < 1.5)')
    
    # Add vertical line for median
    median = calculate_median(lambda_val)
    ax.axvline(x=median, color='purple', linestyle='--', 
              label=f'Median = ln(2)/λ ≈ {median:.4f}')
    
    # Add vertical line for mean
    mean = calculate_expected_value(lambda_val)
    ax.axvline(x=mean, color='orange', linestyle='--', 
              label=f'Mean = 1/λ = {mean:.4f}')
    
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title(f'Exponential Distribution PDF with λ={lambda_val}', fontsize=14)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PDF plot saved to {save_path}")
    
    plt.close()

def plot_exponential_cdf(lambda_val, save_path=None):
    """
    Plot the cumulative distribution function (CDF) for the exponential distribution.
    """
    x = np.linspace(0, 3, 1000)  # x from 0 to 3
    cdf = 1 - np.exp(-lambda_val * x)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, cdf, 'r-', lw=2, label=f'CDF: 1-e^(-λx), λ={lambda_val}')
    
    # Add reference points
    # P(X ≤ 1)
    p_leq_1 = 1 - np.exp(-lambda_val * 1)
    ax.plot(1, p_leq_1, 'go', markersize=8)
    ax.annotate(f'P(X ≤ 1) = {p_leq_1:.4f}', 
                xy=(1, p_leq_1), xytext=(1.1, p_leq_1-0.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    # P(X ≤ 0.5)
    p_leq_05 = 1 - np.exp(-lambda_val * 0.5)
    ax.plot(0.5, p_leq_05, 'bo', markersize=8)
    ax.annotate(f'P(X ≤ 0.5) = {p_leq_05:.4f}', 
                xy=(0.5, p_leq_05), xytext=(0.6, p_leq_05-0.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    # P(X ≤ 1.5)
    p_leq_15 = 1 - np.exp(-lambda_val * 1.5)
    ax.plot(1.5, p_leq_15, 'mo', markersize=8)
    ax.annotate(f'P(X ≤ 1.5) = {p_leq_15:.4f}', 
                xy=(1.5, p_leq_15), xytext=(1.6, p_leq_15-0.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    # Add median
    median = calculate_median(lambda_val)
    ax.plot(median, 0.5, 'ko', markersize=8)
    ax.annotate(f'Median = {median:.4f}', 
                xy=(median, 0.5), xytext=(median+0.1, 0.4),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('F(x)', fontsize=12)
    ax.set_title(f'Exponential Distribution CDF with λ={lambda_val}', fontsize=14)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"CDF plot saved to {save_path}")
    
    plt.close()

def plot_moments_and_quantiles(lambda_val, save_path=None):
    """
    Create a visual representation of the exponential distribution's key properties
    including the mean, variance, and quantiles.
    """
    # Generate data points from exponential distribution
    np.random.seed(42)  # For reproducibility
    data = np.random.exponential(scale=1/lambda_val, size=1000)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot histogram with theoretical PDF
    x = np.linspace(0, 3, 1000)
    pdf = lambda_val * np.exp(-lambda_val * x)
    
    # Calculate key statistics
    mean = calculate_expected_value(lambda_val)
    variance = calculate_variance(lambda_val)
    std_dev = np.sqrt(variance)
    median = calculate_median(lambda_val)
    
    # Histogram plot
    axes[0].hist(data, bins=30, density=True, alpha=0.6, color='skyblue', 
                label=f'Simulated data (n=1000)')
    axes[0].plot(x, pdf, 'r-', lw=2, label=f'PDF: λe^(-λx), λ={lambda_val}')
    
    # Add vertical lines for mean and median
    axes[0].axvline(x=mean, color='green', linestyle='--', 
                   label=f'Mean = 1/λ = {mean:.4f}')
    axes[0].axvline(x=median, color='purple', linestyle='--', 
                   label=f'Median = ln(2)/λ ≈ {median:.4f}')
    
    # Add area for standard deviation
    axes[0].axvspan(mean-std_dev, mean+std_dev, alpha=0.2, color='yellow',
                   label=f'± 1 Std Dev = {std_dev:.4f}')
    
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('Probability Density', fontsize=12)
    axes[0].set_title(f'Exponential Distribution (λ={lambda_val}) with Key Statistics', 
                     fontsize=14)
    axes[0].legend(fontsize=10)
    
    # Box plot
    axes[1].boxplot(data, vert=False, widths=0.7)
    
    # Add annotation for key percentiles
    percentiles = [25, 50, 75]
    for p in percentiles:
        q = -np.log(1 - p/100) / lambda_val
        axes[1].axvline(x=q, color='red', linestyle='--')
        axes[1].text(q, 1.1, f'{p}th percentile\n= {q:.4f}', 
                    ha='center', va='center', fontsize=10)
    
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_title('Box Plot with Key Percentiles', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Moments and quantiles plot saved to {save_path}")
    
    plt.close()

def main():
    """Solve and visualize exponential distribution problem"""
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_1_Quiz_7")
    os.makedirs(save_dir, exist_ok=True)
    
    # Problem parameters
    lambda_val = 2  # Given in the problem statement
    
    print("\n=== Question 7: Exponential Distribution Analysis ===")
    print(f"Parameter: λ = {lambda_val}")
    
    # Task 1: Verify PDF integrates to 1
    pdf_verification = verify_pdf_integrates_to_one(lambda_val)
    print("\n1. Verifying that the PDF integrates to 1:")
    print(f"   Theoretical result: {pdf_verification['theoretical_result']}")
    print(f"   Numerical approximation: {pdf_verification['numerical_result']:.10f}")
    print(f"   Is valid PDF? {pdf_verification['is_valid']}")
    
    # Task 2: Calculate P(X > 1)
    p_greater_than_1 = calculate_probability_greater_than(lambda_val, 1)
    print("\n2. P(X > 1):")
    print(f"   P(X > 1) = e^(-λ·1) = e^(-{lambda_val}) ≈ {p_greater_than_1:.10f}")
    
    # Task 3: Calculate P(0.5 < X < 1.5)
    p_between = calculate_probability_between(lambda_val, 0.5, 1.5)
    print("\n3. P(0.5 < X < 1.5):")
    print(f"   P(0.5 < X < 1.5) = e^(-λ·0.5) - e^(-λ·1.5)")
    print(f"                     = e^(-{lambda_val}·0.5) - e^(-{lambda_val}·1.5)")
    print(f"                     = e^(-1) - e^(-3)")
    print(f"                     ≈ {p_between:.10f}")
    
    # Task 4: Find E[X] and Var(X)
    mean = calculate_expected_value(lambda_val)
    variance = calculate_variance(lambda_val)
    print("\n4. Expected Value and Variance:")
    print(f"   E[X] = 1/λ = 1/{lambda_val} = {mean}")
    print(f"   Var(X) = 1/λ² = 1/{lambda_val}² = {variance}")
    
    # Task 5: Calculate the median
    median = calculate_median(lambda_val)
    print("\n5. Median:")
    print(f"   Median = ln(2)/λ = ln(2)/{lambda_val} ≈ {median:.10f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    plot_exponential_pdf(lambda_val, save_path=os.path.join(save_dir, "exponential_pdf.png"))
    plot_exponential_cdf(lambda_val, save_path=os.path.join(save_dir, "exponential_cdf.png"))
    plot_moments_and_quantiles(lambda_val, save_path=os.path.join(save_dir, "exponential_moments.png"))
    
    print(f"\nAll calculations and visualizations for Question 7 have been completed.")
    print(f"Visualization files have been saved to: {save_dir}")
    
    # Summary of results
    print("\n=== Summary of Results ===")
    print(f"1. Is a valid PDF? Yes (integrates to {pdf_verification['numerical_result']:.10f})")
    print(f"2. P(X > 1) = {p_greater_than_1:.10f}")
    print(f"3. P(0.5 < X < 1.5) = {p_between:.10f}")
    print(f"4. E[X] = {mean}, Var(X) = {variance}")
    print(f"5. Median = {median:.10f}")

if __name__ == "__main__":
    main() 