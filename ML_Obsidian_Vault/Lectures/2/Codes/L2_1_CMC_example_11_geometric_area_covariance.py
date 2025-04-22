import numpy as np
import matplotlib.pyplot as plt
import os

def ensure_directory_exists(directory):
    """Ensure the specified directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_geometric_area_visualization():
    """Create a visualization showing correlation as geometric area."""
    print("\n" + "="*80)
    print("Geometric Area Interpretation of Covariance")
    print("="*80)
    
    print("\nProblem Statement:")
    print("How can we visualize covariance as a geometric area to provide an intuitive understanding?")
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Generate data for different correlations
    np.random.seed(42)
    n_points = 200
    
    # Positive correlation (ρ ≈ 0.8)
    cov_pos = np.array([[1.0, 0.8], [0.8, 1.0]])
    data_pos = np.random.multivariate_normal([0, 0], cov_pos, n_points)
    
    # No correlation (ρ ≈ 0)
    cov_zero = np.array([[1.0, 0.0], [0.0, 1.0]])
    data_zero = np.random.multivariate_normal([0, 0], cov_zero, n_points)
    
    # Negative correlation (ρ ≈ -0.8)
    cov_neg = np.array([[1.0, -0.8], [-0.8, 1.0]])
    data_neg = np.random.multivariate_normal([0, 0], cov_neg, n_points)
    
    # Step 1: Calculate means
    print("\nStep 1: Calculate Means and Center the Data")
    
    # For demonstrations, we'll select a subset of points to keep the visualization clean
    selected_indices = np.random.choice(range(n_points), 20, replace=False)
    
    # Function to calculate and display centered data
    def process_data(data, ax, color, title):
        mean_x = np.mean(data[:, 0])
        mean_y = np.mean(data[:, 1])
        
        print(f"\n{title}:")
        print(f"Mean X = {mean_x:.4f}")
        print(f"Mean Y = {mean_y:.4f}")
        
        # Center the data (subtract means)
        centered_data = data.copy()
        centered_data[:, 0] = data[:, 0] - mean_x
        centered_data[:, 1] = data[:, 1] - mean_y
        
        # Plot all data points in light color
        ax.scatter(data[:, 0], data[:, 1], color=color, alpha=0.2, s=20)
        
        # Plot selected points in darker color
        selected_data = data[selected_indices]
        selected_centered = centered_data[selected_indices]
        
        ax.scatter(selected_data[:, 0], selected_data[:, 1], color=color, s=50)
        
        # Plot mean point
        ax.scatter(mean_x, mean_y, color='black', s=100, marker='X', label='Mean')
        
        # Draw lines to mean for selected points to visualize deviations
        for i in range(len(selected_data)):
            ax.plot([selected_data[i, 0], mean_x], [selected_data[i, 1], mean_y], 
                    color=color, alpha=0.5, linestyle='--')
        
        print(f"Centered a subset of data points by subtracting the mean")
        
        # Return the centered data for further calculations
        return selected_centered, selected_data
    
    # Process and plot data for each correlation type
    centered_pos, selected_pos = process_data(data_pos, ax1, 'blue', "Positive Correlation")
    centered_zero, selected_zero = process_data(data_zero, ax2, 'purple', "Zero Correlation")
    centered_neg, selected_neg = process_data(data_neg, ax3, 'red', "Negative Correlation")
    
    # Step 2: Calculate and visualize the areas representing covariance
    print("\nStep 2: Calculate Covariance as Area")
    
    # Function to calculate and visualize covariance as area
    def visualize_covariance_area(centered_data, selected_data, mean_x, mean_y, ax, color, title):
        # Calculate covariance
        cov_xy = np.mean(centered_data[:, 0] * centered_data[:, 1])
        
        # Sample calculation for a few points
        n_samples = min(5, len(centered_data))
        
        print(f"\n{title} - Covariance Calculation:")
        print(f"Cov(X,Y) = (1/n) * Σ[(x_i - μ_x) * (y_i - μ_y)]")
        print("For the first few points:")
        
        sum_product = 0
        for i in range(n_samples):
            dx = centered_data[i, 0]
            dy = centered_data[i, 1]
            product = dx * dy
            sum_product += product
            
            print(f"Point {i+1}: (x={selected_data[i, 0]:.2f}, y={selected_data[i, 1]:.2f})")
            print(f"  Deviation x: {dx:.4f}, Deviation y: {dy:.4f}")
            print(f"  Product: {dx:.4f} × {dy:.4f} = {product:.4f}")
            
            # Draw rectangle representing the area (dx * dy)
            if abs(product) > 0.1:  # Only draw rectangles for non-tiny areas
                # Determine rectangle corners
                if dx > 0 and dy > 0:  # Positive area (1st quadrant)
                    rect = plt.Rectangle((0, 0), dx, dy, alpha=0.2, color=color)
                elif dx < 0 and dy < 0:  # Positive area (3rd quadrant)
                    rect = plt.Rectangle((dx, dy), -dx, -dy, alpha=0.2, color=color)
                elif dx > 0 and dy < 0:  # Negative area (4th quadrant)
                    rect = plt.Rectangle((0, dy), dx, -dy, alpha=0.2, color='gray')
                else:  # Negative area (2nd quadrant)
                    rect = plt.Rectangle((dx, 0), -dx, dy, alpha=0.2, color='gray')
                
                ax.add_patch(rect)
                
                # Add text annotation for the area
                area_text = f"{product:.2f}"
                ax.text(dx/2 if dx > 0 else dx/2, dy/2 if dy > 0 else dy/2, 
                        area_text, ha='center', va='center', fontsize=8)
        
        print(f"Sum of products = {sum_product:.4f}")
        print(f"Covariance = (1/{n_samples}) × {sum_product:.4f} = {sum_product/n_samples:.4f}")
        
        # Calculate correlation
        var_x = np.var(centered_data[:, 0], ddof=0)
        var_y = np.var(centered_data[:, 1], ddof=0)
        corr_xy = cov_xy / np.sqrt(var_x * var_y)
        
        print(f"Variance X = {var_x:.4f}")
        print(f"Variance Y = {var_y:.4f}")
        print(f"Correlation = {cov_xy:.4f} / (√{var_x:.4f} × √{var_y:.4f}) = {corr_xy:.4f}")
        
        return cov_xy, corr_xy
    
    # Visualize covariance as area for each correlation type
    cov_pos, corr_pos = visualize_covariance_area(centered_pos, selected_pos, 
                                                 np.mean(data_pos[:, 0]), np.mean(data_pos[:, 1]), 
                                                 ax1, 'blue', "Positive Correlation")
    
    cov_zero, corr_zero = visualize_covariance_area(centered_zero, selected_zero, 
                                                   np.mean(data_zero[:, 0]), np.mean(data_zero[:, 1]), 
                                                   ax2, 'purple', "Zero Correlation")
    
    cov_neg, corr_neg = visualize_covariance_area(centered_neg, selected_neg, 
                                                 np.mean(data_neg[:, 0]), np.mean(data_neg[:, 1]), 
                                                 ax3, 'red', "Negative Correlation")
    
    # Set titles and labels
    ax1.set_title(f'Positive Correlation\nCov(X,Y) = {cov_pos:.2f}, ρ = {corr_pos:.2f}\nMostly Positive Areas', fontsize=12)
    ax2.set_title(f'Zero Correlation\nCov(X,Y) = {cov_zero:.2f}, ρ = {corr_zero:.2f}\nPositive & Negative Areas Balance', fontsize=12)
    ax3.set_title(f'Negative Correlation\nCov(X,Y) = {cov_neg:.2f}, ρ = {corr_neg:.2f}\nMostly Negative Areas', fontsize=12)
    
    # Set equal aspect ratio for all subplots
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.legend()
    
    # Add a main title
    fig.suptitle('Covariance as Geometric Area', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    print("\nStep 3: Interpreting the Geometric Area Visualization")
    print("- In positive correlation: Most rectangles contribute positive area")
    print("  Points tend to be in 1st and 3rd quadrants (x,y both positive or both negative)")
    print("- In zero correlation: Positive and negative areas roughly cancel out")
    print("  Points are evenly distributed across all quadrants")
    print("- In negative correlation: Most rectangles contribute negative area")
    print("  Points tend to be in 2nd and 4th quadrants (x,y have opposite signs)")
    
    # Save the figure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images", "Contour_Plots")
    ensure_directory_exists(images_dir)
    
    try:
        save_path = os.path.join(images_dir, "ex11_correlation_geometric.png")
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\nFigure saved to: {save_path}")
    except Exception as e:
        print(f"\nError saving figure: {e}")
    
    return fig

if __name__ == "__main__":
    print("\n\n" + "*"*80)
    print("EXAMPLE: GEOMETRIC AREA INTERPRETATION OF COVARIANCE")
    print("*"*80)
    
    # Create the geometric area visualization
    fig = create_geometric_area_visualization()