import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images/Linear_Regression_Detailed relative to the parent directory
images_dir = os.path.join(parent_dir, "Images", "Linear_Regression_Detailed")

# Create images directory if it doesn't exist
os.makedirs(images_dir, exist_ok=True)

def save_figure(plt, filename):
    """Save figure to the images directory"""
    plt.savefig(os.path.join(images_dir, f'{filename}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_data_with_regression(x, y, title, filename, prediction_line=True, annotate_points=True):
    """
    Plot data points and optionally the regression line
    
    Parameters:
    -----------
    x, y : array-like
        The data points
    title : str
        Plot title
    filename : str
        Name for saving the figure
    prediction_line : bool
        Whether to show the regression line
    annotate_points : bool
        Whether to annotate data points with coordinates
    """
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.scatter(x, y, color='blue', s=50, label='Data points')
    
    if annotate_points:
        # Annotate each point with its coordinates
        for i, (xi, yi) in enumerate(zip(x, y)):
            plt.annotate(f'({xi}, {yi})', 
                         xy=(xi, yi),
                         xytext=(5, 5),
                         textcoords='offset points')
    
    if prediction_line:
        # Calculate regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Generate points for the line
        x_line = np.linspace(min(x) - 0.5, max(x) + 0.5, 100)
        y_line = intercept + slope * x_line
        
        # Plot the regression line
        plt.plot(x_line, y_line, color='red', lw=2,
                 label=f'Regression line')
    
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    save_figure(plt, filename)

def plot_computation_steps(x, y, title, filename):
    """
    Create a visualization of the computation steps in linear regression
    """
    # Calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Calculate the terms needed for the slope
    x_dev = x - x_mean
    y_dev = y - y_mean
    
    # Calculate products and squares
    products = x_dev * y_dev
    squared_x_dev = x_dev ** 2
    
    # Calculate slope and intercept
    slope = np.sum(products) / np.sum(squared_x_dev)
    intercept = y_mean - slope * x_mean
    
    # Create a figure with a single plot
    plt.figure(figsize=(10, 6))
    
    # Scatter plot with regression line
    plt.scatter(x, y, color='blue', s=50)
    
    # Plot means
    plt.axvline(x=x_mean, color='darkgreen', linestyle='--', alpha=0.7, label=f'x̄ = {x_mean:.2f}')
    plt.axhline(y=y_mean, color='purple', linestyle='--', alpha=0.7, label=f'ȳ = {y_mean:.2f}')
    
    # Plot regression line
    x_line = np.linspace(min(x) - 0.5, max(x) + 0.5, 100)
    y_line = intercept + slope * x_line
    plt.plot(x_line, y_line, color='red', lw=2, label=f'Regression line')
    
    # Add deviations for a sample point
    idx = len(x) // 2  # Middle point for demonstration
    plt.annotate('', xy=(x[idx], y_mean), xytext=(x[idx], y[idx]),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=1.5))
    plt.annotate('', xy=(x_mean, y[idx]), xytext=(x[idx], y[idx]),
                arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=1.5))
    
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    save_figure(plt, filename)

def car_price_estimation():
    """
    Example 1: Estimating Car Prices Based on Age
    
    This example demonstrates how to:
    1. Calculate means, slope, and intercept from data points
    2. Visualize the calculations step by step
    3. Make predictions using the model
    """
    print("\n=== Example 1: Estimating Car Prices Based on Age ===\n")
    
    # Data: Car age (years) and price ($1000s)
    car_age = np.array([1, 3, 5, 7, 9, 10])
    car_price = np.array([32, 27, 22, 20, 16, 15])
    
    # Step 1: Calculate the means
    x_mean = np.mean(car_age)
    y_mean = np.mean(car_price)
    print(f"Step 1: Calculate means")
    print(f"    x̄ (mean car age) = {x_mean:.2f} years")
    print(f"    ȳ (mean car price) = {y_mean:.2f} thousand dollars")
    
    # Step 2: Calculate the slope coefficient (β₁)
    numerator = sum((car_age - x_mean) * (car_price - y_mean))
    denominator = sum((car_age - x_mean) ** 2)
    slope = numerator / denominator
    print(f"\nStep 2: Calculate the slope coefficient (β₁)")
    print(f"    β₁ = Σ(x_i - x̄)(y_i - ȳ) / Σ(x_i - x̄)²")
    print(f"    β₁ = {numerator:.2f} / {denominator:.2f} = {slope:.3f}")
    
    # Step 3: Calculate the intercept (β₀)
    intercept = y_mean - slope * x_mean
    print(f"\nStep 3: Calculate the intercept (β₀)")
    print(f"    β₀ = ȳ - β₁x̄")
    print(f"    β₀ = {y_mean:.2f} - ({slope:.3f} × {x_mean:.2f}) = {intercept:.3f}")
    
    # Step 4: Write the regression equation
    print(f"\nStep 4: Write the regression equation")
    print(f"    Price = {intercept:.3f} - {abs(slope):.3f} × Age")
    
    # Detailed computation steps table
    print("\nDetailed computation table:")
    table_data = []
    for i, (x, y) in enumerate(zip(car_age, car_price)):
        x_dev = x - x_mean
        y_dev = y - y_mean
        row = [
            x, y, 
            f"{x_dev:.2f}", f"{y_dev:.2f}", 
            f"{x_dev * y_dev:.2f}", f"{x_dev**2:.2f}"
        ]
        table_data.append(row)
        print(f"    {i+1}: x_i={x:2d}, y_i={y:2d}, x_i-x̄={x_dev:6.2f}, y_i-ȳ={y_dev:6.2f}, "
              f"(x_i-x̄)(y_i-ȳ)={x_dev*y_dev:7.2f}, (x_i-x̄)²={x_dev**2:7.2f}")
    
    # Calculate sums for verification
    sum_x_dev_y_dev = sum((car_age - x_mean) * (car_price - y_mean))
    sum_x_dev_squared = sum((car_age - x_mean) ** 2)
    print(f"    Σ(x_i-x̄)(y_i-ȳ) = {sum_x_dev_y_dev:.2f}")
    print(f"    Σ(x_i-x̄)² = {sum_x_dev_squared:.2f}")
    
    # Plot the data with regression line
    plot_data_with_regression(
        car_age, car_price,
        "Car Price vs. Age with Regression Line",
        "car_price_vs_age_regression"
    )
    
    # Plot computation steps
    plot_computation_steps(
        car_age, car_price,
        "Step-by-Step Linear Regression for Car Prices",
        "car_price_computation_steps"
    )
    
    # Make predictions
    test_age = 4  # Age of car to predict price for
    predicted_price = intercept + slope * test_age
    print(f"\nStep 5: Make a prediction for a {test_age}-year-old car")
    print(f"    Predicted Price = {intercept:.3f} + ({slope:.3f} × {test_age}) = ${predicted_price:.2f}k")
    
    # Calculate coefficient of determination (R²)
    correlation = np.corrcoef(car_age, car_price)[0, 1]
    r_squared = correlation ** 2
    print(f"\nStep 6: Calculate the coefficient of determination (R²)")
    print(f"    Correlation coefficient (r) = {correlation:.3f}")
    print(f"    R² = {r_squared:.3f}")
    
    # Generate model quality assessment plot - Just create the plot without annotations
    y_pred = intercept + slope * car_age
    residuals = car_price - y_pred
    
    plt.figure(figsize=(10, 8))
    
    # Original data with regression line
    plt.subplot(2, 1, 1)
    plt.scatter(car_age, car_price, color='blue', label='Actual data')
    plt.plot(car_age, y_pred, color='red', label='Regression line')
    for i, (x, y, p) in enumerate(zip(car_age, car_price, y_pred)):
        plt.vlines(x=x, ymin=p, ymax=y, colors='green', linestyles='dashed', alpha=0.7)
    plt.title('Car Price vs. Age with Prediction Errors')
    plt.xlabel('Car Age (years)')
    plt.ylabel('Car Price ($1000s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Residual plot
    plt.subplot(2, 1, 2)
    plt.scatter(car_age, residuals, color='purple')
    plt.axhline(y=0, color='red', linestyle='-', alpha=0.7)
    plt.title('Residuals (Prediction Errors)')
    plt.xlabel('Car Age (years)')
    plt.ylabel('Residual ($1000s)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(plt, "car_price_model_assessment")
    
    # Print the residual statistics instead of plotting them
    print(f"\nModel Assessment - Residual Statistics:")
    print(f"    Mean of residuals = {np.mean(residuals):.3f}")
    print(f"    Standard deviation = {np.std(residuals):.3f}")
    print(f"    Sum of squared residuals (SSE) = {np.sum(residuals**2):.3f}")
    
    return car_age, car_price, slope, intercept, r_squared

def salary_prediction():
    """
    Example 2: Salary Prediction Based on Experience
    
    This example demonstrates:
    1. Computing linear regression coefficients step by step
    2. Interpreting the coefficients in a real-world context
    3. Visualizing the relationship and making predictions
    """
    print("\n=== Example 2: Salary Prediction Based on Experience ===\n")
    
    # Data: Experience (years) and Salary ($1000s)
    experience = np.array([1, 3, 5, 7, 10])
    salary = np.array([45, 60, 75, 83, 100])
    
    # Step 1: Calculate the means
    x_mean = np.mean(experience)
    y_mean = np.mean(salary)
    print(f"Step 1: Calculate means")
    print(f"    x̄ (mean experience) = {x_mean:.2f} years")
    print(f"    ȳ (mean salary) = {y_mean:.2f} thousand dollars")
    
    # Step 2: Calculate the slope coefficient (β₁)
    numerator = sum((experience - x_mean) * (salary - y_mean))
    denominator = sum((experience - x_mean) ** 2)
    slope = numerator / denominator
    print(f"\nStep 2: Calculate the slope coefficient (β₁)")
    print(f"    β₁ = Σ(x_i - x̄)(y_i - ȳ) / Σ(x_i - x̄)²")
    print(f"    β₁ = {numerator:.2f} / {denominator:.2f} = {slope:.3f}")
    
    # Step 3: Calculate the intercept (β₀)
    intercept = y_mean - slope * x_mean
    print(f"\nStep 3: Calculate the intercept (β₀)")
    print(f"    β₀ = ȳ - β₁x̄")
    print(f"    β₀ = {y_mean:.2f} - ({slope:.3f} × {x_mean:.2f}) = {intercept:.3f}")
    
    # Step 4: Write the regression equation
    print(f"\nStep 4: Write the regression equation")
    print(f"    Salary = {intercept:.3f} + {slope:.3f} × Experience")
    
    # Detailed computation steps table
    print("\nDetailed computation table:")
    for i, (x, y) in enumerate(zip(experience, salary)):
        x_dev = x - x_mean
        y_dev = y - y_mean
        print(f"    {i+1}: x_i={x:2d}, y_i={y:3d}, x_i-x̄={x_dev:6.2f}, y_i-ȳ={y_dev:6.2f}, "
              f"(x_i-x̄)(y_i-ȳ)={x_dev*y_dev:7.2f}, (x_i-x̄)²={x_dev**2:7.2f}")
    
    # Calculate sums for verification
    sum_x_dev_y_dev = sum((experience - x_mean) * (salary - y_mean))
    sum_x_dev_squared = sum((experience - x_mean) ** 2)
    print(f"    Σ(x_i-x̄)(y_i-ȳ) = {sum_x_dev_y_dev:.2f}")
    print(f"    Σ(x_i-x̄)² = {sum_x_dev_squared:.2f}")
    
    # Plot the data with regression line
    plot_data_with_regression(
        experience, salary,
        "Salary vs. Experience with Regression Line",
        "salary_vs_experience_regression"
    )
    
    # Plot computation steps
    plot_computation_steps(
        experience, salary,
        "Step-by-Step Linear Regression for Salary Prediction",
        "salary_computation_steps"
    )
    
    # Make predictions
    test_experience = 6  # Years of experience to predict salary for
    predicted_salary = intercept + slope * test_experience
    print(f"\nStep 5: Make a prediction for someone with {test_experience} years of experience")
    print(f"    Predicted Salary = {intercept:.3f} + ({slope:.3f} × {test_experience}) = ${predicted_salary:.2f}k")
    
    # Calculate coefficient of determination (R²)
    correlation = np.corrcoef(experience, salary)[0, 1]
    r_squared = correlation ** 2
    print(f"\nStep 6: Calculate the coefficient of determination (R²)")
    print(f"    Correlation coefficient (r) = {correlation:.3f}")
    print(f"    R² = {r_squared:.3f}")
    
    # Print the interpretation instead of adding to the plot
    print(f"\nModel Interpretation:")
    print(f"    • Base salary (β₀): ${intercept:.2f}k (theoretical salary at 0 years)")
    print(f"    • Salary increase (β₁): ${slope:.2f}k per year of experience")
    print(f"    • Model accuracy (R²): {r_squared:.3f} ({r_squared*100:.1f}% of salary variation explained by experience)")
    
    # Create simpler plot for model visualization
    plot_data_with_regression(
        experience, salary,
        "Interpreting the Salary Prediction Model",
        "salary_model_interpretation"
    )
    
    # Create prediction visualization for a range of experience levels
    x_pred = np.linspace(0, 12, 100)
    y_pred = intercept + slope * x_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(experience, salary, color='blue', s=50, label='Training data')
    plt.plot(x_pred, y_pred, color='red', label='Regression line')
    
    # Highlight prediction for 6 years
    plt.scatter([test_experience], [predicted_salary], color='green', s=80, 
                label=f'Prediction for {test_experience} years')
    plt.vlines(x=test_experience, ymin=0, ymax=predicted_salary, 
               colors='green', linestyles='dashed')
    plt.hlines(y=predicted_salary, xmin=0, xmax=test_experience, 
               colors='green', linestyles='dashed')
    
    plt.title("Salary Prediction Based on Years of Experience")
    plt.xlabel("Experience (years)")
    plt.ylabel("Salary ($1000s)")
    plt.xlim(0, 12)
    plt.ylim(30, 120)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    save_figure(plt, "salary_prediction_visualization")
    
    # Print the prediction information
    print(f"\nPrediction Details:")
    print(f"    Prediction formula: Salary = {intercept:.2f} + {slope:.2f} × Experience")
    print(f"    For {test_experience} years: ${predicted_salary:.2f}k")
    
    return experience, salary, slope, intercept, r_squared

def main():
    """Run all examples and generate images"""
    print("# Linear Regression Detailed Examples - Step-by-Step Solutions")
    print("-----------------------------------------------------------")
    
    # Example 1: Car Price Estimation
    car_age, car_price, car_slope, car_intercept, car_r_squared = car_price_estimation()
    
    # Example 2: Salary Prediction
    exp, salary, salary_slope, salary_intercept, salary_r_squared = salary_prediction()
    
    # Summary of all examples
    print("\n## Summary of Examples")
    
    print("\n### Car Price Estimation")
    print(f"* **Regression equation**: Price = {car_intercept:.3f} + ({car_slope:.3f} × Age)")
    print(f"* **Interpretation**: For each additional year of age, a car's price decreases by ${abs(car_slope):.3f}k")
    print(f"* **Model accuracy**: R² = {car_r_squared:.3f} ({car_r_squared*100:.1f}% of price variation explained by age)")
    
    print("\n### Salary Prediction")
    print(f"* **Regression equation**: Salary = {salary_intercept:.3f} + ({salary_slope:.3f} × Experience)")
    print(f"* **Interpretation**: For each additional year of experience, salary increases by ${salary_slope:.3f}k")
    print(f"* **Model accuracy**: R² = {salary_r_squared:.3f} ({salary_r_squared*100:.1f}% of salary variation explained by experience)")
    
    print("\n### Images")
    print(f"All examples completed. Images saved to:")
    print(f"* `{images_dir}`")
    
    # Print markdown notice for generated images
    print("\n### Generated Visualizations")
    print("\n#### Car Price Example")
    print(f"![Car Price vs Age](../Images/Linear_Regression_Detailed/car_price_vs_age_regression.png)")
    print(f"![Car Price Computation](../Images/Linear_Regression_Detailed/car_price_computation_steps.png)")
    print(f"![Car Price Model Assessment](../Images/Linear_Regression_Detailed/car_price_model_assessment.png)")
    
    print("\n#### Salary Prediction Example")
    print(f"![Salary vs Experience](../Images/Linear_Regression_Detailed/salary_vs_experience_regression.png)")
    print(f"![Salary Computation](../Images/Linear_Regression_Detailed/salary_computation_steps.png)")
    print(f"![Salary Model](../Images/Linear_Regression_Detailed/salary_model_interpretation.png)")
    print(f"![Salary Prediction](../Images/Linear_Regression_Detailed/salary_prediction_visualization.png)")

if __name__ == "__main__":
    main() 