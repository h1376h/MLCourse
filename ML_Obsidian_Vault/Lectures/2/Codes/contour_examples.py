import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def ensure_directory_exists(directory):
    """Ensure the specified directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def example1_quadratic_function():
    """Example 1: Simple Quadratic Function f(x,y) = x^2 + y^2"""
    # Create a grid of x, y points
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate function values
    Z = X**2 + Y**2
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=(15, 10))
    
    # 2D contour plot
    ax1 = fig.add_subplot(121)
    contour_levels = [1, 4, 9, 16, 25]
    contour = ax1.contour(X, Y, Z, levels=contour_levels, colors='black')
    ax1.clabel(contour, inline=True, fontsize=10)
    ax1.set_title('Contour Plot: f(x,y) = x² + y²')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True)
    
    # Add circles to highlight specific contours
    for level in [1, 4, 9]:
        radius = np.sqrt(level)
        circle = plt.Circle((0, 0), radius, fill=False, linestyle='--', 
                            color='red', alpha=0.7)
        ax1.add_patch(circle)
        ax1.text(radius/np.sqrt(2), radius/np.sqrt(2), f'r = {radius}', 
                color='red', fontsize=9)
    
    # 3D surface plot
    ax2 = fig.add_subplot(122, projection='3d')
    surface = ax2.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8)
    
    # Add contour lines on the 3D plot
    contour_z_vals = [1, 4, 9]
    for z_val in contour_z_vals:
        # Plot circles at heights z_val
        theta = np.linspace(0, 2*np.pi, 100)
        radius = np.sqrt(z_val)
        x_circle = radius * np.cos(theta)
        y_circle = radius * np.sin(theta)
        z_circle = np.ones_like(theta) * z_val
        ax2.plot(x_circle, y_circle, z_circle, 'r-', linewidth=2)
    
    ax2.set_title('3D Surface: f(x,y) = x² + y²')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('f(x,y)')
    
    plt.tight_layout()
    return fig

def example2_linear_function():
    """Example 2: Linear Function f(x,y) = 2x + 3y"""
    # Create a grid of x, y points
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate function values
    Z = 2*X + 3*Y
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=(15, 10))
    
    # 2D contour plot
    ax1 = fig.add_subplot(121)
    contour_levels = [-10, -5, 0, 5, 10, 15]
    contour = ax1.contour(X, Y, Z, levels=contour_levels, colors='black')
    ax1.clabel(contour, inline=True, fontsize=10)
    ax1.set_title('Contour Plot: f(x,y) = 2x + 3y')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True)
    
    # Highlight specific points on contour lines
    for c in contour_levels:
        # Points on the y-axis where x=0
        y_val = c/3
        if -5 <= y_val <= 5:
            ax1.plot(0, y_val, 'ro', markersize=5)
            ax1.text(0.2, y_val, f'(0, {y_val:.1f})', fontsize=9)
        
        # Points on the x-axis where y=0
        x_val = c/2
        if -5 <= x_val <= 5:
            ax1.plot(x_val, 0, 'bo', markersize=5)
            ax1.text(x_val, 0.2, f'({x_val:.1f}, 0)', fontsize=9)
    
    # 3D surface plot
    ax2 = fig.add_subplot(122, projection='3d')
    surface = ax2.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8)
    
    # Add contour lines on the 3D plot
    for z_val in contour_levels:
        # For each z value, plot a line on the surface
        x_line = np.linspace(-5, 5, 100)
        y_line = (z_val - 2*x_line) / 3
        # Filter out points outside our domain
        mask = (y_line >= -5) & (y_line <= 5)
        ax2.plot(x_line[mask], y_line[mask], z_val * np.ones_like(x_line[mask]), 
                 'r-', linewidth=2)
    
    ax2.set_title('3D Surface: f(x,y) = 2x + 3y')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('f(x,y)')
    
    plt.tight_layout()
    return fig

def example3_manhattan_distance():
    """Example 3: Manhattan Distance f(x,y) = |x| + |y|"""
    # Create a grid of x, y points
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate function values
    Z = np.abs(X) + np.abs(Y)
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=(15, 10))
    
    # 2D contour plot
    ax1 = fig.add_subplot(121)
    contour_levels = [1, 2, 3, 4, 5]
    contour = ax1.contour(X, Y, Z, levels=contour_levels, colors='black')
    ax1.clabel(contour, inline=True, fontsize=10)
    ax1.set_title('Contour Plot: f(x,y) = |x| + |y| (Manhattan Distance)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True)
    
    # Add diamond annotations
    for c in [1, 3, 5]:
        # Add diamond shapes
        diamond_x = [c, 0, -c, 0, c]
        diamond_y = [0, c, 0, -c, 0]
        ax1.plot(diamond_x, diamond_y, 'r--', alpha=0.7)
        ax1.text(c/np.sqrt(2), c/np.sqrt(2), f'c={c}', color='red', fontsize=9)
    
    # 3D surface plot
    ax2 = fig.add_subplot(122, projection='3d')
    surface = ax2.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8)
    
    # Add contour lines on the 3D plot
    for z_val in [1, 2, 3]:
        # Plot diamonds at heights z_val
        t = np.linspace(0, 2*np.pi, 100)
        x_diamond = []
        y_diamond = []
        
        # Create diamond shape parametrically
        for angle in np.linspace(0, 2*np.pi, 5):
            if angle % (np.pi/2) < 0.01:  # At multiples of 90 degrees
                r = z_val / (np.abs(np.cos(angle)) + np.abs(np.sin(angle)))
                x_diamond.append(r * np.cos(angle))
                y_diamond.append(r * np.sin(angle))
        
        z_diamond = np.ones_like(x_diamond) * z_val
        ax2.plot(x_diamond, y_diamond, z_diamond, 'r-', linewidth=2)
    
    ax2.set_title('3D Surface: f(x,y) = |x| + |y|')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('f(x,y)')
    
    plt.tight_layout()
    return fig

def example4_product_function():
    """Example 4: Product Function f(x,y) = xy"""
    # Create a grid of x, y points
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate function values
    Z = X * Y
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=(15, 10))
    
    # 2D contour plot
    ax1 = fig.add_subplot(121)
    contour_levels = [-4, -2, -1, 0, 1, 2, 4]
    contour = ax1.contour(X, Y, Z, levels=contour_levels, colors='black')
    ax1.clabel(contour, inline=True, fontsize=10)
    ax1.set_title('Contour Plot: f(x,y) = xy')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True)
    
    # Add annotations for hyperbolas
    for c in [-2, 2]:
        # Plot points on the hyperbola xy = c
        x_vals = np.linspace(0.5, 4, 10)
        y_vals = c / x_vals
        ax1.plot(x_vals, y_vals, 'ro', markersize=3, alpha=0.5)
        
        # Add annotation
        ax1.text(3, c/3, f'xy = {c}', color='red', fontsize=9, 
                 bbox=dict(facecolor='white', alpha=0.7))
    
    # Highlight the axes
    ax1.plot([-5, 5], [0, 0], 'b--', linewidth=1)
    ax1.plot([0, 0], [-5, 5], 'b--', linewidth=1)
    ax1.text(4, 0.3, 'x-axis (y=0)', color='blue', fontsize=9)
    ax1.text(0.3, 4, 'y-axis (x=0)', color='blue', fontsize=9)
    
    # 3D surface plot
    ax2 = fig.add_subplot(122, projection='3d')
    surface = ax2.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8)
    
    # Add contour lines on the 3D plot
    for z_val in [-2, 0, 2]:
        if z_val == 0:
            # For z=0, plot the axes
            ax2.plot([-5, 5], [0, 0], [0, 0], 'r-', linewidth=2)
            ax2.plot([0, 0], [-5, 5], [0, 0], 'r-', linewidth=2)
        else:
            # For non-zero z, plot hyperbolas
            x_curve = np.linspace(0.5, 5, 50)
            y_curve = z_val / x_curve
            mask = (y_curve >= -5) & (y_curve <= 5)
            ax2.plot(x_curve[mask], y_curve[mask], 
                     z_val * np.ones_like(x_curve[mask]), 'r-', linewidth=2)
            
            # Plot the other branch of the hyperbola
            x_curve = np.linspace(-5, -0.5, 50)
            y_curve = z_val / x_curve
            mask = (y_curve >= -5) & (y_curve <= 5)
            ax2.plot(x_curve[mask], y_curve[mask], 
                     z_val * np.ones_like(x_curve[mask]), 'r-', linewidth=2)
    
    ax2.set_title('3D Surface: f(x,y) = xy')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('f(x,y)')
    
    plt.tight_layout()
    return fig

def example5_circle_plus_line():
    """Example 5: Circle Plus Line f(x,y) = x^2 + y^2 + y"""
    # Create a grid of x, y points
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate function values
    Z = X**2 + Y**2 + Y
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=(15, 10))
    
    # 2D contour plot
    ax1 = fig.add_subplot(121)
    contour_levels = [0, 1, 4, 9, 16]
    contour = ax1.contour(X, Y, Z, levels=contour_levels, colors='black')
    ax1.clabel(contour, inline=True, fontsize=10)
    ax1.set_title('Contour Plot: f(x,y) = x² + y² + y')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True)
    
    # Add annotation about the shifted center
    ax1.plot(0, -0.5, 'ro', markersize=5)
    ax1.text(0.2, -0.5, 'Center (0, -0.5)', color='red', fontsize=9)
    
    # Add a circle to show the minimum
    circle = plt.Circle((0, -0.5), 0.5, fill=False, linestyle='--', 
                        color='red', alpha=0.7)
    ax1.add_patch(circle)
    
    # 3D surface plot
    ax2 = fig.add_subplot(122, projection='3d')
    surface = ax2.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8)
    
    # Mark the minimum point
    ax2.plot([0], [-0.5], [-0.25], 'ro', markersize=5)
    
    # Add contour lines on the 3D plot
    for z_val in contour_levels:
        # For each z value, create a parametric circle representation
        theta = np.linspace(0, 2*np.pi, 100)
        r_squared = z_val - Y  # From the equation x^2 + y^2 + y = z
        
        # For a specific y value where the circle exists
        for y_val in [-2, -1, -0.5, 0, 1]:
            radius_squared = z_val - y_val
            if radius_squared < 0:
                continue
            
            radius = np.sqrt(radius_squared)
            x_circle = radius * np.cos(theta)
            y_circle = np.ones_like(theta) * y_val
            z_circle = np.ones_like(theta) * z_val
            
            ax2.plot(x_circle, y_circle, z_circle, 'r-', linewidth=1, alpha=0.6)
    
    ax2.set_title('3D Surface: f(x,y) = x² + y² + y')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('f(x,y)')
    
    plt.tight_layout()
    return fig

def example6_saddle_function():
    """Example 6: Saddle Function f(x,y) = x^2 - y^2"""
    # Create a grid of x, y points
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate function values
    Z = X**2 - Y**2
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=(15, 10))
    
    # 2D contour plot
    ax1 = fig.add_subplot(121)
    contour_levels = [-4, -2, -1, 0, 1, 2, 4]
    contour = ax1.contour(X, Y, Z, levels=contour_levels, colors='black')
    ax1.clabel(contour, inline=True, fontsize=10)
    ax1.set_title('Contour Plot: f(x,y) = x² - y² (Saddle Function)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True)
    
    # Add annotations for the saddle point and hyperbolas
    ax1.plot(0, 0, 'ro', markersize=5)
    ax1.text(0.2, 0.2, 'Saddle point (0,0)', fontsize=9, color='red')
    
    # Add lines for the zero contour
    ax1.plot([-3, 3], [3, -3], 'r--', alpha=0.7)
    ax1.plot([-3, 3], [-3, 3], 'r--', alpha=0.7)
    ax1.text(2, 2, 'x = y (z = 0)', color='red', fontsize=9, rotation=45)
    ax1.text(2, -2, 'x = -y (z = 0)', color='red', fontsize=9, rotation=-45)
    
    # 3D surface plot
    ax2 = fig.add_subplot(122, projection='3d')
    surface = ax2.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8)
    
    # Mark the saddle point
    ax2.plot([0], [0], [0], 'ro', markersize=5)
    
    # Add contour lines on the 3D plot
    for z_val in contour_levels:
        # For non-zero z values, plot the hyperbola
        if z_val != 0:
            # Create hyperbola parametrically
            t = np.linspace(-2, 2, 100)
            if z_val > 0:
                # For positive z, hyperbola along x-axis
                x_curve = np.sqrt(z_val) * np.cosh(t)
                y_curve = np.sqrt(z_val) * np.sinh(t) / np.sqrt(z_val)
            else:
                # For negative z, hyperbola along y-axis
                y_curve = np.sqrt(-z_val) * np.cosh(t)
                x_curve = np.sqrt(-z_val) * np.sinh(t) / np.sqrt(-z_val)
            
            # Filter points within our domain
            mask = (x_curve >= -3) & (x_curve <= 3) & (y_curve >= -3) & (y_curve <= 3)
            if any(mask):
                ax2.plot(x_curve[mask], y_curve[mask], 
                        z_val * np.ones_like(x_curve[mask]), 'r-', linewidth=2)
            
            # Plot the negative branch of the hyperbola
            if z_val > 0:
                x_curve = -np.sqrt(z_val) * np.cosh(t)
            else:
                y_curve = -np.sqrt(-z_val) * np.cosh(t)
                
            # Filter points within our domain
            mask = (x_curve >= -3) & (x_curve <= 3) & (y_curve >= -3) & (y_curve <= 3)
            if any(mask):
                ax2.plot(x_curve[mask], y_curve[mask], 
                        z_val * np.ones_like(x_curve[mask]), 'r-', linewidth=2)
        else:
            # For z=0, plot the asymptotes
            ax2.plot([-3, 3], [3, -3], [0, 0], 'r-', linewidth=2)
            ax2.plot([-3, 3], [-3, 3], [0, 0], 'r-', linewidth=2)
    
    ax2.set_title('3D Surface: f(x,y) = x² - y²')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('f(x,y)')
    
    plt.tight_layout()
    return fig

def example7_local_extrema():
    """Example 7: Function with Local Extrema f(x,y) = x^2 + y^2 - 4x - 6y + 13"""
    # Create a grid of x, y points
    x = np.linspace(-2, 6, 100)
    y = np.linspace(-2, 8, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate function values
    Z = X**2 + Y**2 - 4*X - 6*Y + 13
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=(15, 10))
    
    # 2D contour plot
    ax1 = fig.add_subplot(121)
    contour_levels = [0, 1, 4, 9, 16, 25]
    contour = ax1.contour(X, Y, Z, levels=contour_levels, colors='black')
    ax1.clabel(contour, inline=True, fontsize=10)
    ax1.set_title('Contour Plot: f(x,y) = x² + y² - 4x - 6y + 13')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True)
    
    # Add annotation about the minimum point
    ax1.plot(2, 3, 'ro', markersize=5)
    ax1.text(2.2, 3.2, 'Minimum (2, 3)', color='red', fontsize=9)
    
    # Add a circle to show a level curve
    circle = plt.Circle((2, 3), 1, fill=False, linestyle='--', 
                        color='red', alpha=0.7)
    ax1.add_patch(circle)
    
    # 3D surface plot
    ax2 = fig.add_subplot(122, projection='3d')
    surface = ax2.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8)
    
    # Mark the minimum point
    min_z = (2**2 + 3**2 - 4*2 - 6*3 + 13)  # Calculate the minimum value
    ax2.plot([2], [3], [min_z], 'ro', markersize=5)
    
    # Add contour lines on the 3D plot
    for z_val in contour_levels:
        # For each z value, create a parametric circle representation
        theta = np.linspace(0, 2*np.pi, 100)
        # From completing the square, we have (x-2)^2 + (y-3)^2 = z - min_z
        radius_squared = z_val - min_z
        if radius_squared < 0:
            continue
        
        radius = np.sqrt(radius_squared)
        x_circle = 2 + radius * np.cos(theta)
        y_circle = 3 + radius * np.sin(theta)
        z_circle = np.ones_like(theta) * z_val
        
        ax2.plot(x_circle, y_circle, z_circle, 'r-', linewidth=2)
    
    ax2.set_title('3D Surface: f(x,y) = x² + y² - 4x - 6y + 13')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('f(x,y)')
    
    plt.tight_layout()
    return fig

def generate_contour_examples():
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images", "Contour_Plots")
    ensure_directory_exists(images_dir)
    
    # Generate and save examples
    examples = [
        (example1_quadratic_function, "example1_quadratic"),
        (example2_linear_function, "example2_linear"),
        (example3_manhattan_distance, "example3_manhattan"),
        (example4_product_function, "example4_product"),
        (example5_circle_plus_line, "example5_circle_plus_line"),
        (example6_saddle_function, "example6_saddle"),
        (example7_local_extrema, "example7_local_extrema")
    ]
    
    results = {}
    for example_func, name in examples:
        try:
            fig = example_func()
            save_path = os.path.join(images_dir, f"{name}.png")
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            print(f"Generated {name} visualization")
            results[name] = save_path
        except Exception as e:
            print(f"Error generating {name}: {e}")
    
    print(f"\nAll visualizations saved to: {images_dir}")
    return results

if __name__ == "__main__":
    generate_contour_examples() 