import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_correlation(x, y):
    """
    Calculate the Pearson correlation coefficient between two variables with detailed steps.
    
    Parameters:
    x, y -- Arrays of values for the two variables
    
    Returns:
    correlation -- Pearson correlation coefficient
    detailed_steps -- Dictionary containing detailed calculation steps
    """
    # Step 1: Calculate means
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # Step 2: Calculate deviations from means
    dev_x = x - mean_x
    dev_y = y - mean_y
    
    # Step 3: Calculate product of deviations
    product_deviations = dev_x * dev_y
    
    # Step 4: Calculate sum of products
    sum_products = np.sum(product_deviations)
    
    # Step 5: Calculate covariance
    n = len(x)
    covariance = sum_products / (n - 1)
    
    # Step 6: Calculate squared deviations for standard deviations
    squared_dev_x = dev_x ** 2
    squared_dev_y = dev_y ** 2
    
    # Step 7: Calculate sum of squared deviations
    sum_squared_dev_x = np.sum(squared_dev_x)
    sum_squared_dev_y = np.sum(squared_dev_y)
    
    # Step 8: Calculate standard deviations
    std_x = np.sqrt(sum_squared_dev_x / (n - 1))
    std_y = np.sqrt(sum_squared_dev_y / (n - 1))
    
    # Step 9: Calculate correlation
    correlation = covariance / (std_x * std_y)
    
    # Store detailed steps
    detailed_steps = {
        'means': {'x': mean_x, 'y': mean_y},
        'deviations': {'x': dev_x, 'y': dev_y},
        'product_deviations': product_deviations,
        'sum_products': sum_products,
        'covariance': covariance,
        'squared_deviations': {'x': squared_dev_x, 'y': squared_dev_y},
        'sum_squared_deviations': {'x': sum_squared_dev_x, 'y': sum_squared_dev_y},
        'standard_deviations': {'x': std_x, 'y': std_y},
        'correlation': correlation
    }
    
    return correlation, detailed_steps

def plot_correlation(x, y, title, xlabel, ylabel, save_path=None):
    """
    Create a scatter plot with correlation information.
    
    Parameters:
    x, y -- Arrays of values to plot
    title -- Plot title
    xlabel, ylabel -- Axis labels
    save_path -- Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot
    plt.scatter(x, y, color='blue', alpha=0.6)
    
    # Add regression line
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b, color='red', linestyle='--', label=f'Regression line: y = {m:.2f}x + {b:.2f}')
    
    # Add mean lines
    plt.axvline(x=np.mean(x), color='green', linestyle=':', label=f'Mean {xlabel}: {np.mean(x):.2f}')
    plt.axhline(y=np.mean(y), color='purple', linestyle=':', label=f'Mean {ylabel}: {np.mean(y):.2f}')
    
    # Calculate correlation
    correlation, _ = calculate_correlation(x, y)
    
    # Add correlation information
    plt.title(f'{title}\nCorrelation: {correlation:.3f}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.close()

def print_detailed_steps(x, y, x_name, y_name):
    """
    Print detailed calculation steps for correlation analysis.
    
    Parameters:
    x, y -- Arrays of values
    x_name, y_name -- Names of the variables
    """
    correlation, steps = calculate_correlation(x, y)
    
    print(f"\nDetailed Steps for {x_name} and {y_name} Correlation:")
    print("=" * 50)
    
    print("\nStep 1: Calculate Means")
    print(f"Mean of {x_name}: {steps['means']['x']:.2f}")
    print(f"Mean of {y_name}: {steps['means']['y']:.2f}")
    
    print("\nStep 2: Calculate Deviations from Means")
    print(f"Deviations of {x_name}: {steps['deviations']['x']}")
    print(f"Deviations of {y_name}: {steps['deviations']['y']}")
    
    print("\nStep 3: Calculate Product of Deviations")
    print(f"Product of deviations: {steps['product_deviations']}")
    
    print("\nStep 4: Calculate Sum of Products")
    print(f"Sum of products: {steps['sum_products']:.2f}")
    
    print("\nStep 5: Calculate Covariance")
    print(f"Covariance: {steps['covariance']:.2f}")
    
    print("\nStep 6: Calculate Squared Deviations")
    print(f"Squared deviations of {x_name}: {steps['squared_deviations']['x']}")
    print(f"Squared deviations of {y_name}: {steps['squared_deviations']['y']}")
    
    print("\nStep 7: Calculate Sum of Squared Deviations")
    print(f"Sum of squared deviations of {x_name}: {steps['sum_squared_deviations']['x']:.2f}")
    print(f"Sum of squared deviations of {y_name}: {steps['sum_squared_deviations']['y']:.2f}")
    
    print("\nStep 8: Calculate Standard Deviations")
    print(f"Standard deviation of {x_name}: {steps['standard_deviations']['x']:.2f}")
    print(f"Standard deviation of {y_name}: {steps['standard_deviations']['y']:.2f}")
    
    print("\nStep 9: Calculate Correlation")
    print(f"Correlation coefficient: {steps['correlation']:.3f}")
    print("=" * 50)

def main():
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    images_dir = os.path.join(parent_dir, "Images", "L2_1_Correlation")
    os.makedirs(images_dir, exist_ok=True)
    
    print("\n=== CORRELATION EXAMPLES: DETAILED STEP-BY-STEP SOLUTIONS ===\n")
    
    # Example 1: Stock Market Returns
    print("\nExample 1: Stock Market Returns")
    tc_returns = np.array([1.2, -0.5, 0.8, -1.1, 1.6])
    fsi_returns = np.array([0.8, -0.3, 0.2, -0.9, 1.2])
    print_detailed_steps(tc_returns, fsi_returns, "TC Returns", "FSI Returns")
    plot_correlation(tc_returns, fsi_returns, 
                    "Stock Returns Correlation", 
                    "Tech Corp Returns (%)", 
                    "Financial Services Inc Returns (%)",
                    os.path.join(images_dir, "stock_correlation.png"))
    
    # Example 2: Housing Data Analysis
    print("\nExample 2: Housing Data Analysis")
    sizes = np.array([1500, 2200, 1800, 3000, 2500])
    prices = np.array([250, 340, 275, 455, 390])
    print_detailed_steps(sizes, prices, "House Size", "Price")
    plot_correlation(sizes, prices, 
                    "Housing Data Correlation", 
                    "House Size (sq ft)", 
                    "Price ($k)",
                    os.path.join(images_dir, "housing_correlation.png"))
    
    # Example 3: Temperature and Ice Cream Sales
    print("\nExample 3: Temperature and Ice Cream Sales")
    temperatures = np.array([18, 23, 25, 21, 20, 26, 24])
    sales = np.array([56, 74, 82, 65, 68, 79, 75])
    print_detailed_steps(temperatures, sales, "Temperature", "Sales")
    plot_correlation(temperatures, sales, 
                    "Temperature and Ice Cream Sales Correlation", 
                    "Temperature (°C)", 
                    "Ice Cream Sales (units)",
                    os.path.join(images_dir, "temperature_sales_correlation.png"))
    
    # Example 4: Student Performance Analysis
    print("\nExample 4: Student Performance Analysis")
    study_hours = np.array([5, 3, 7, 2, 6])
    prev_scores = np.array([85, 70, 90, 65, 95])
    sleep_hours = np.array([7, 5, 8, 6, 7])
    final_scores = np.array([88, 72, 95, 70, 93])
    
    print("\nStudy Hours vs Final Scores:")
    print_detailed_steps(study_hours, final_scores, "Study Hours", "Final Scores")
    plot_correlation(study_hours, final_scores,
                    "Study Hours vs Final Score",
                    "Study Hours",
                    "Final Score",
                    os.path.join(images_dir, "study_hours_correlation.png"))
    
    print("\nPrevious Scores vs Final Scores:")
    print_detailed_steps(prev_scores, final_scores, "Previous Scores", "Final Scores")
    plot_correlation(prev_scores, final_scores,
                    "Previous Scores vs Final Score",
                    "Previous Test Score",
                    "Final Score",
                    os.path.join(images_dir, "prev_scores_correlation.png"))
    
    print("\nSleep Hours vs Final Scores:")
    print_detailed_steps(sleep_hours, final_scores, "Sleep Hours", "Final Scores")
    plot_correlation(sleep_hours, final_scores,
                    "Sleep Hours vs Final Score",
                    "Sleep Hours",
                    "Final Score",
                    os.path.join(images_dir, "sleep_hours_correlation.png"))
    
    # Example 5: Spurious Correlation
    print("\nExample 5: Spurious Correlation Analysis")
    food_sales = np.array([120, 150, 180, 210, 250])
    cs_degrees = np.array([5000, 5500, 6200, 7500, 8000])
    print_detailed_steps(food_sales, cs_degrees, "Food Sales", "CS Degrees")
    plot_correlation(food_sales, cs_degrees,
                    "Organic Food Sales vs CS Degrees",
                    "Organic Food Sales ($M)",
                    "CS Degrees Awarded",
                    os.path.join(images_dir, "spurious_correlation.png"))
    
    print("\nAll calculations and visualizations have been completed.")
    print(f"Visualization files have been saved to: {images_dir}")

if __name__ == "__main__":
    main() 