import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Ellipse, Rectangle
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

def ensure_directory_exists(directory):
    """Ensure the specified directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_1d_with_regions(ax, x, sigma_sq=1.0):
    """Plot 1D normal distribution with shaded sigma regions."""
    # Calculate PDF
    y = (1/np.sqrt(2*np.pi*sigma_sq)) * np.exp(-0.5 * x**2/sigma_sq)
    
    # Plot the PDF
    ax.plot(x, y, 'k-', linewidth=2)
    
    # Calculate sigma values
    sigma = np.sqrt(sigma_sq)
    
    # Define sigma regions
    regions = [
        (-sigma, sigma, 0.6827, 'red'),
        (-2*sigma, 2*sigma, 0.9545, 'green'),
        (-3*sigma, 3*sigma, 0.9973, 'blue')
    ]
    
    # Add shaded regions
    for start, end, prob, color in regions:
        mask = (x >= start) & (x <= end)
        ax.fill_between(x[mask], 0, y[mask], alpha=0.3, color=color)
        
        # Calculate the center position for the text
        idx = np.abs(x - (start + end)/2).argmin()
        text_y = y[idx] / 2
        
        # Add text for probability
        ax.text((start + end)/2, text_y, f"{prob:.1%}", 
                ha='center', va='center', color=color, fontweight='bold')
    
    # Add sigma markers
    sigmas = [1, 2, 3]
    for i in sigmas:
        ax.axvline(i*sigma, color='k', linestyle='--', alpha=0.5)
        ax.axvline(-i*sigma, color='k', linestyle='--', alpha=0.5)
        ax.text(i*sigma, 0, f"{i}σ", ha='left', va='bottom')
        ax.text(-i*sigma, 0, f"-{i}σ", ha='right', va='bottom')
    
    # Add title and labels
    ax.set_title(f'1D Normal Distribution (σ² = {sigma_sq})')
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density')
    ax.grid(True, alpha=0.3)

def plot_2d_with_regions(ax, x, y, sigma_x_sq=1.0, sigma_y_sq=1.0):
    """Plot 2D normal distribution with shaded sigma regions."""
    # Create meshgrid
    X, Y = np.meshgrid(x, y)
    
    # Calculate PDF
    Z = (1/(2*np.pi*np.sqrt(sigma_x_sq*sigma_y_sq))) * np.exp(-0.5*(X**2/sigma_x_sq + Y**2/sigma_y_sq))
    
    # Plot contours
    contour_levels = np.linspace(0.01, 0.15, 5)
    cp = ax.contour(X, Y, Z, levels=contour_levels, colors='black')
    ax.clabel(cp, inline=True, fontsize=8)
    
    # Calculate sigma values
    sigma_x = np.sqrt(sigma_x_sq)
    sigma_y = np.sqrt(sigma_y_sq)
    
    # Define sigma ellipses with probabilities
    ellipses = [
        (1, 0.3935, 'red'),     # 1-sigma ellipse: 39.35% of data
        (2, 0.8647, 'green'),   # 2-sigma ellipse: 86.47% of data
        (3, 0.9889, 'blue')     # 3-sigma ellipse: 98.89% of data
    ]
    
    # Add ellipses
    for scale, prob, color in ellipses:
        ellipse = Ellipse(xy=(0, 0), 
                          width=2*scale*sigma_x, 
                          height=2*scale*sigma_y, 
                          fill=True, 
                          facecolor=color, 
                          alpha=0.2, 
                          edgecolor='k', 
                          linestyle='-')
        ax.add_patch(ellipse)
        
        # Add text for probability
        ax.text(0, 0, f"{prob:.1%}", 
                ha='center', va='center', color='black', fontweight='bold')
    
    # Add title and labels
    ax.set_title(f'2D Normal Distribution (σ_x² = {sigma_x_sq}, σ_y² = {sigma_y_sq})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3*max(sigma_x, 1), 3*max(sigma_x, 1))
    ax.set_ylim(-3*max(sigma_y, 1), 3*max(sigma_y, 1))
    ax.set_aspect('equal')

def plot_3d_gaussian(sigma_x_sq, sigma_y_sq, ax, title):
    """Plot 3D Gaussian surface."""
    # Create grid
    x = np.linspace(-3*np.sqrt(max(sigma_x_sq, 1)), 3*np.sqrt(max(sigma_x_sq, 1)), 50)
    y = np.linspace(-3*np.sqrt(max(sigma_y_sq, 1)), 3*np.sqrt(max(sigma_y_sq, 1)), 50)
    X, Y = np.meshgrid(x, y)
    
    # Calculate PDF
    Z = (1/(2*np.pi*np.sqrt(sigma_x_sq*sigma_y_sq))) * np.exp(-0.5*(X**2/sigma_x_sq + Y**2/sigma_y_sq))
    
    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.7, linewidth=0, antialiased=True)
    
    # Add contours on xy-plane
    offset = 0
    contours = ax.contour(X, Y, Z, zdir='z', offset=offset, cmap=cm.coolwarm, levels=5, alpha=0.8)
    
    # Add sigma ellipses on xy-plane
    for i in range(1, 4):
        theta = np.linspace(0, 2*np.pi, 100)
        x_ellipse = i * np.sqrt(sigma_x_sq) * np.cos(theta)
        y_ellipse = i * np.sqrt(sigma_y_sq) * np.sin(theta)
        z_ellipse = np.zeros_like(theta) + offset
        ax.plot(x_ellipse, y_ellipse, z_ellipse, 'r-', linewidth=2, alpha=0.7)
    
    # Label axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Probability Density')
    ax.set_title(title)
    
    # Adjust view angle
    ax.view_init(elev=30, azim=45)
    
    return surf

def plot_1d_to_2d_transition(fig, x, y, variances):
    """Create a series of plots showing transition from 1D to 2D normal distributions."""
    n_plots = len(variances)
    grid = gridspec.GridSpec(2, n_plots, figure=fig)
    
    for i, (sigma_x_sq, sigma_y_sq) in enumerate(variances):
        # 1D Plot
        ax1d = fig.add_subplot(grid[0, i])
        plot_1d_with_regions(ax1d, x, sigma_x_sq)
        
        # 2D Plot
        ax2d = fig.add_subplot(grid[1, i])
        plot_2d_with_regions(ax2d, x, y, sigma_x_sq, sigma_y_sq)
        
        # Adjust titles for sequence
        ax1d.set_title(f'Step {i+1}: 1D Normal (σ² = {sigma_x_sq})')
        ax2d.set_title(f'Step {i+1}: 2D Normal (σ_x² = {sigma_x_sq}, σ_y² = {sigma_y_sq})')

def create_variance_effect_visualization(sigma_x_values, sigma_y_values):
    """Create a grid visualization showing the effect of different variance combinations."""
    # Create the figure
    fig, axes = plt.subplots(len(sigma_y_values), len(sigma_x_values), figsize=(15, 15))
    
    # Generate the grid of points
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # Loop through all variance combinations
    for i, sigma_y_sq in enumerate(sigma_y_values):
        for j, sigma_x_sq in enumerate(sigma_x_values):
            ax = axes[i, j]
            
            # Calculate PDF
            Z = (1/(2*np.pi*np.sqrt(sigma_x_sq*sigma_y_sq))) * np.exp(-0.5*(X**2/sigma_x_sq + Y**2/sigma_y_sq))
            
            # Plot contours
            contour_levels = np.linspace(0.01, 0.15, 5)
            cp = ax.contour(X, Y, Z, levels=contour_levels, colors='black')
            
            # Add 2-sigma ellipse
            ellipse = Ellipse(xy=(0, 0), 
                             width=2*2*np.sqrt(sigma_x_sq), 
                             height=2*2*np.sqrt(sigma_y_sq), 
                             fill=False, 
                             edgecolor='red', 
                             linestyle='--')
            ax.add_patch(ellipse)
            
            # Set title and grid
            ax.set_title(f'σ_x² = {sigma_x_sq}, σ_y² = {sigma_y_sq}')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            
            # Only show labels on outer axes
            if i == len(sigma_y_values) - 1:
                ax.set_xlabel('x')
            else:
                ax.set_xticklabels([])
                
            if j == 0:
                ax.set_ylabel('y')
            else:
                ax.set_yticklabels([])
    
    # Add a figure title
    fig.suptitle('Effect of Variances on 2D Normal Distribution Contours', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    
    return fig

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
    
    # Create figure for basic example
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
    
    # Save the basic figure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images", "Contour_Plots")
    ensure_directory_exists(images_dir)
    
    try:
        save_path = os.path.join(images_dir, "ex2_basic_2d_normal_examples.png")
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\nBasic examples figure saved to: {save_path}")
    except Exception as e:
        print(f"\nError saving figure: {e}")
    
    # Create visualization showing probability mass within different sigma regions
    print("\nCreating visualization of probability mass within sigma regions...")
    fig_regions = plt.figure(figsize=(15, 10))
    
    # 1D with regions
    ax_1d = fig_regions.add_subplot(221)
    plot_1d_with_regions(ax_1d, x)
    
    # 2D with regions
    ax_2d = fig_regions.add_subplot(222)
    plot_2d_with_regions(ax_2d, x, y)
    
    # 3D visualization for standard normal
    ax_3d_1 = fig_regions.add_subplot(223, projection='3d')
    plot_3d_gaussian(1.0, 1.0, ax_3d_1, '3D Standard Normal Distribution')
    
    # 3D visualization for different variances
    ax_3d_2 = fig_regions.add_subplot(224, projection='3d')
    plot_3d_gaussian(2.0, 0.5, ax_3d_2, '3D Normal with Different Variances')
    
    # Save probability mass visualization
    try:
        save_path = os.path.join(images_dir, "ex2_normal_probability_mass_visualization.png")
        fig_regions.tight_layout()
        fig_regions.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\nProbability mass visualization saved to: {save_path}")
    except Exception as e:
        print(f"\nError saving figure: {e}")
    
    # Create 1D to 2D transition visualization
    print("\nCreating 1D to 2D transition visualization...")
    fig_transition = plt.figure(figsize=(15, 8))
    
    # Define the transition sequence
    transition_variances = [
        (1.0, 1.0),   # Standard normal in both dimensions
        (0.5, 1.0),   # Narrow in x, standard in y
        (2.0, 0.5)    # Wide in x, narrow in y
    ]
    
    # Create the transition plots
    plot_1d_to_2d_transition(fig_transition, x, y, transition_variances)
    
    # Save transition visualization
    try:
        save_path = os.path.join(images_dir, "ex2_normal_1d_to_2d_transition.png")
        fig_transition.tight_layout()
        fig_transition.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\n1D to 2D transition visualization saved to: {save_path}")
    except Exception as e:
        print(f"\nError saving figure: {e}")
    
    # Create variance effect grid visualization
    print("\nCreating variance effect grid visualization...")
    sigma_x_values = [0.5, 1.0, 2.0, 3.0]
    sigma_y_values = [0.5, 1.0, 2.0, 3.0]
    
    fig_variance_grid = create_variance_effect_visualization(sigma_x_values, sigma_y_values)
    
    # Save variance grid visualization
    try:
        save_path = os.path.join(images_dir, "ex2_normal_variance_effect_grid.png")
        fig_variance_grid.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\nVariance effect grid visualization saved to: {save_path}")
    except Exception as e:
        print(f"\nError saving figure: {e}")
    
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
    
    return fig

if __name__ == "__main__":
    print("\n\n" + "*"*80)
    print("EXAMPLE 2: BASIC 2D NORMAL DISTRIBUTIONS")
    print("*"*80)
    
    # Run the example with detailed step-by-step printing
    fig = basic_2d_example()
    