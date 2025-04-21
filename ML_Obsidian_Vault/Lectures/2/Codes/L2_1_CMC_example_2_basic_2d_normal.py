import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Ellipse

def ensure_directory_exists(directory):
    """Ensure the specified directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def basic_2d_example():
    """Simple example showing 1D and 2D normal distributions"""
    # Print detailed step-by-step solution
    print("\n" + "="*80)
    print("Example: Basic 1D and 2D Normal Distributions")
    print("="*80)
    
    print("\nStep 1: Understanding 1D Normal Distributions with Different Variances")
    print("The probability density function of a 1D normal distribution is:")
    print("f(x) = (1/√(2πσ²)) * exp(-x²/(2σ²))")
    print("where σ² is the variance parameter.")
    print("\nWe'll visualize three cases:")
    print("1. Standard normal (σ² = 1): f(x) = (1/√(2π)) * exp(-x²/2)")
    print("2. Narrow normal (σ² = 0.5): f(x) = (1/√(π)) * exp(-x²/1)")
    print("   - This has a taller peak (larger maximum value)")
    print("   - It decreases more rapidly as x moves away from the mean")
    print("3. Wide normal (σ² = 2): f(x) = (1/√(4π)) * exp(-x²/4)")
    print("   - This has a shorter peak (smaller maximum value)")
    print("   - It decreases more slowly as x moves away from the mean")
    print("\nThe key insight: total area under each curve = 1 (probability axiom)")
    print("So curves with higher peaks must be narrower, and those with lower peaks must be wider")
    
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: 1D Normal Distributions with different variances
    ax1 = fig.add_subplot(131)
    x = np.linspace(-5, 5, 1000)
    
    # Standard normal distribution
    y1 = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * x**2)
    # Normal with variance 0.5
    y2 = (1/np.sqrt(2*np.pi*0.5)) * np.exp(-0.5 * x**2/0.5)
    # Normal with variance 2
    y3 = (1/np.sqrt(2*np.pi*2)) * np.exp(-0.5 * x**2/2)
    
    ax1.plot(x, y1, 'b-', label='σ² = 1')
    ax1.plot(x, y2, 'r-', label='σ² = 0.5')
    ax1.plot(x, y3, 'g-', label='σ² = 2')
    
    # Add vertical lines at ±σ, ±2σ, ±3σ for standard normal
    for i in range(1, 4):
        ax1.axvline(i, color='b', linestyle='--', alpha=0.3)
        ax1.axvline(-i, color='b', linestyle='--', alpha=0.3)
        if i == 1:
            ax1.text(i, 0.05, f'{i}σ', ha='left', va='bottom', color='b')
            
    ax1.set_title('1D Normal Distributions with Different Variances')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Probability Density')
    ax1.legend()
    ax1.grid(True)
    
    print("\nStep 2: Extending to 2D - The Standard Bivariate Normal Distribution")
    print("The PDF of a 2D standard normal distribution (with identity covariance matrix) is:")
    print("f(x,y) = (1/2π) * exp(-(x² + y²)/2)")
    print("\nKey properties:")
    print("- Equal variance in both dimensions (σ₁² = σ₂² = 1)")
    print("- Zero correlation between x and y (ρ = 0)")
    print("- Contours form perfect circles centered at the origin")
    print("- The equation for the contours is x² + y² = constant")
    print("- The contour value c corresponds to the constant: -2ln(2πc)")
    print("- 1σ, 2σ, and 3σ circles have radii of 1, 2, and 3 respectively")
    print("- The 1σ circle contains approximately 39% of the probability mass")
    print("- The 2σ circle contains approximately 86% of the probability mass")
    print("- The 3σ circle contains approximately 99% of the probability mass")
    
    # Plot 2: 2D Independent Normal Distribution (Diagonal Covariance)
    ax2 = fig.add_subplot(132)
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate PDF values for a 2D independent normal (diagonal covariance)
    Z = (1/(2*np.pi)) * np.exp(-0.5*(X**2 + Y**2))
    
    # Plot the contours
    contour_levels = np.linspace(0.01, 0.15, 5)
    cp = ax2.contour(X, Y, Z, levels=contour_levels, colors='black')
    ax2.clabel(cp, inline=True, fontsize=10)
    
    # Add 1σ, 2σ and 3σ circles
    for i in range(1, 4):
        circle = plt.Circle((0, 0), i, fill=False, edgecolor='red', linestyle='--')
        ax2.add_patch(circle)
        if i == 2:
            ax2.text(0, i, f'{i}σ', ha='center', va='bottom', color='red')
    
    ax2.set_title('2D Standard Normal Distribution\n(Independent Variables)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.grid(True)
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    
    print("\nStep 3: 2D Normal with Different Variances (Diagonal Covariance Matrix)")
    print("Now we'll examine a bivariate normal where the variances are different:")
    print("f(x,y) = (1/(2π√|Σ|)) * exp(-1/2 * ((x²/σ₁²) + (y²/σ₂²)))")
    print("where σ₁² = 2 and σ₂² = 0.5")
    print("\nKey properties:")
    print("- Covariance matrix Σ = [[2, 0], [0, 0.5]]")
    print("- Determinant |Σ| = 2 * 0.5 = 1")
    print("- Different variances in x and y directions")
    print("- Still zero correlation between variables (ρ = 0)")
    print("- Contours form axis-aligned ellipses")
    print("- The equation for the contours is (x²/2 + y²/0.5) = constant")
    print("- The semi-axes of the ellipses are in the ratio √2 : √0.5 ≈ 1.41 : 0.71")
    print("- The ellipses are stretched along the x-axis and compressed along the y-axis")
    print("- This reflects greater variance in the x direction than in the y direction")
    
    # Plot 3: 2D Normal with different variances but still independent
    ax3 = fig.add_subplot(133)
    
    # Calculate PDF for 2D normal with different variances
    Z = (1/(2*np.pi*np.sqrt(2*0.5))) * np.exp(-0.5*(X**2/2 + Y**2/0.5))
    
    # Plot the contours
    cp = ax3.contour(X, Y, Z, levels=contour_levels, colors='black')
    ax3.clabel(cp, inline=True, fontsize=10)
    
    # Add ellipses to represent the covariance
    for i in range(1, 4):
        ellipse = Ellipse(xy=(0, 0), width=i*2*np.sqrt(2), height=i*2*np.sqrt(0.5), 
                         fill=False, edgecolor='red', linestyle='--')
        ax3.add_patch(ellipse)
        if i == 2:
            ax3.text(0, np.sqrt(0.5)*i, f'{i}σ₂', ha='center', va='bottom', color='red')
            ax3.text(np.sqrt(2)*i, 0, f'{i}σ₁', ha='left', va='center', color='red')
    
    ax3.set_title('2D Normal with Different Variances\n(Independent Variables)')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.grid(True)
    ax3.set_xlim(-3, 3)
    ax3.set_ylim(-3, 3)
    
    print("\nStep 4: Comparing the Three Cases")
    print("Key insights from these visualizations:")
    print("1. 1D normal distributions: As variance increases, the peak height decreases")
    print("   and the spread increases, but the total area remains constant (= 1)")
    print("2. 2D standard normal (equal variances): Circular contours indicating")
    print("   equal spread in all directions. This is the simplest case.")
    print("3. 2D normal with different variances: Elliptical contours indicating")
    print("   different spread in different directions. The direction of greater")
    print("   variance corresponds to the longer axis of the ellipse.")
    print("\nThe mathematical relationship: The shape of the contours directly reflects")
    print("the structure of the covariance matrix. In these examples, the variables are")
    print("uncorrelated, so the ellipses are aligned with the coordinate axes.")
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("\n\n" + "*"*80)
    print("EXAMPLE 2: BASIC 2D NORMAL DISTRIBUTIONS")
    print("*"*80)
    
    # Run the example with detailed step-by-step printing
    fig = basic_2d_example()
    
    # Save the figure if needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images", "Contour_Plots")
    ensure_directory_exists(images_dir)
    
    try:
        save_path = os.path.join(images_dir, "basic_2d_normal_examples.png")
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\nFigure saved to: {save_path}")
    except Exception as e:
        print(f"\nError saving figure: {e}")
    