import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_2_Quiz_9")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Define the data from Question 9
time_on_site = np.array([2, 5, 8, 12, 15])
purchase_amount = np.array([15, 35, 40, 60, 75])

# Step 1: Compute the least squares estimates
def calculate_least_squares(x, y):
    """Calculate the least squares estimates for slope and intercept."""
    n = len(x)
    
    # Calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    print(f"Step 1.1: Calculate means")
    print(f"Mean of time on site (x̄): {x_mean} minutes")
    print(f"Mean of purchase amount (ȳ): ${y_mean}")
    print()
    
    # Calculate numerator (covariance)
    numerator = sum([(x[i] - x_mean) * (y[i] - y_mean) for i in range(n)])
    
    # Calculate denominator (sum of squared deviations)
    denominator = sum([(x[i] - x_mean) ** 2 for i in range(n)])
    
    # Calculate slope
    beta_1 = numerator / denominator
    
    # Calculate intercept
    beta_0 = y_mean - beta_1 * x_mean
    
    print(f"Step 1.2: Calculate covariance and variance")
    print(f"Covariance: Sum of (x_i - x̄)(y_i - ȳ) = {numerator}")
    print(f"Variance of x: Sum of (x_i - x̄)² = {denominator}")
    print()
    
    print(f"Step 1.3: Calculate slope (β₁)")
    print(f"β₁ = Covariance / Variance of x = {numerator} / {denominator} = {beta_1}")
    print()
    
    print(f"Step 1.4: Calculate intercept (β₀)")
    print(f"β₀ = ȳ - β₁·x̄ = {y_mean} - {beta_1} · {x_mean} = {beta_0}")
    print()
    
    return beta_0, beta_1

# Execute Step 1
beta_0, beta_1 = calculate_least_squares(time_on_site, purchase_amount)

# Print the regression equation
print(f"Linear Regression Equation: Purchase Amount = {beta_0:.2f} + {beta_1:.2f} · Time on Site")
print()

# Step 2: Calculate the predicted purchase amount for a user who spends 10 minutes on the site
new_time = 10
predicted_amount = beta_0 + beta_1 * new_time

print("Step 2: Calculate the predicted purchase amount for a user who spends 10 minutes on the site")
print(f"Predicted purchase amount = {beta_0:.2f} + {beta_1:.2f} × 10 = ${predicted_amount:.2f}")
print()

# Step 3 & 4: Calculate squared errors and Mean Squared Error (MSE)
def calculate_errors(x, y, beta_0, beta_1):
    """Calculate predicted values, errors, squared errors, and MSE."""
    # Calculate predicted values
    y_pred = beta_0 + beta_1 * x
    
    # Calculate errors (residuals)
    errors = y - y_pred
    
    # Calculate squared errors
    squared_errors = errors ** 2
    
    # Calculate MSE
    mse = np.mean(squared_errors)
    
    print("Step 3: Calculate the squared error for each data point")
    print("Time (x) | Purchase ($) | Predicted Purchase ($) | Error | Squared Error")
    print("----------------------------------------------------------------------")
    
    for i in range(len(x)):
        print(f"{x[i]:^8} | ${y[i]:^12} | ${y_pred[i]:^22.2f} | ${errors[i]:^5.2f} | ${squared_errors[i]:^12.2f}")
    
    print(f"\nStep 4: Calculate the Mean Squared Error (MSE)")
    print(f"MSE = Sum of Squared Errors / n = {sum(squared_errors):.2f} / {len(x)} = {mse:.2f}")
    
    return y_pred, errors, squared_errors, mse

# Execute Step 3 & 4
y_pred, errors, squared_errors, mse = calculate_errors(time_on_site, purchase_amount, beta_0, beta_1)

# Create visualizations
def create_visualizations(x, y, y_pred, errors, beta_0, beta_1, new_x=None, new_y=None, save_dir=None):
    """Create visualizations to help understand the regression analysis."""
    saved_files = []
    
    # Plot 1: Scatter plot with regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', s=100, label='Observed Data')
    
    # Generate points for the regression line
    x_line = np.linspace(0, max(x) + 2, 100)
    y_line = beta_0 + beta_1 * x_line
    
    plt.plot(x_line, y_line, color='red', linestyle='-', linewidth=2, 
             label=f'Regression Line: y = {beta_0:.2f} + {beta_1:.2f}x')
    
    # If we have a new prediction point, add it
    if new_x is not None and new_y is not None:
        plt.scatter(new_x, new_y, color='green', s=100, marker='*', 
                    label=f'Prediction: ${new_y:.2f}')
        plt.plot([new_x, new_x], [0, new_y], 'g--', alpha=0.5)
    
    # Add data point labels
    for i in range(len(x)):
        plt.annotate(f"({x[i]}, ${y[i]})", 
                    (x[i], y[i]), 
                    xytext=(5, 10), 
                    textcoords='offset points')
    
    plt.xlabel('Time on Site (minutes)', fontsize=12)
    plt.ylabel('Purchase Amount ($)', fontsize=12)
    plt.title('Linear Regression: Purchase Amount vs Time on Site', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot1_regression_line.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 2: Residuals plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, errors, color='purple', s=100)
    plt.axhline(y=0, color='red', linestyle='-', alpha=0.7)
    
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [0, errors[i]], 'purple', linestyle='--', alpha=0.5)
        plt.annotate(f"Error: ${errors[i]:.2f}", 
                    (x[i], errors[i]), 
                    xytext=(5, 10 if errors[i] >= 0 else -25), 
                    textcoords='offset points')
    
    plt.xlabel('Time on Site (minutes)', fontsize=12)
    plt.ylabel('Residuals ($)', fontsize=12)
    plt.title('Residuals Plot', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot2_residuals.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 3: Visual representation of the squared errors
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', s=100, label='Observed Data')
    plt.plot(x_line, y_line, color='red', linestyle='-', linewidth=2, label='Regression Line')
    
    # Draw lines for the residuals
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [y[i], y_pred[i]], 'k--', alpha=0.7)
        
        # Create square to represent squared error
        width = 0.5  # Width of the squared error illustration
        rect_x = x[i] - width/2
        
        if errors[i] >= 0:
            rect_y = y_pred[i]
            rect_height = errors[i]
        else:
            rect_y = y[i]
            rect_height = -errors[i]
            
        # Draw the square with area proportional to squared error
        rect_width = np.sqrt(abs(errors[i])) if width * np.sqrt(abs(errors[i])) < 5 else 2
        plt.gca().add_patch(plt.Rectangle((rect_x, rect_y), rect_width, rect_height, 
                                       color='orange', alpha=0.4, 
                                       label='Squared Error' if i == 0 else None))
        
        # Add squared error value as text
        plt.annotate(f"SE: ${squared_errors[i]:.2f}", 
                     (x[i] + rect_width/2, (y_pred[i] + y[i])/2), 
                     xytext=(10, 0), 
                     textcoords='offset points')
    
    plt.xlabel('Time on Site (minutes)', fontsize=12)
    plt.ylabel('Purchase Amount ($)', fontsize=12)
    plt.title('Visualization of Squared Errors', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot3_squared_errors.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 4: Actual vs. Predicted Values
    plt.figure(figsize=(8, 8))
    plt.scatter(y, y_pred, color='green', s=100)
    
    # Add reference line (perfect predictions)
    min_val = min(min(y), min(y_pred)) - 5
    max_val = max(max(y), max(y_pred)) + 5
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    
    # Add data point labels
    for i in range(len(x)):
        plt.annotate(f"{x[i]} min", (y[i], y_pred[i]), 
                    xytext=(7, 3), textcoords='offset points')
    
    plt.xlabel('Actual Purchase Amount ($)', fontsize=12)
    plt.ylabel('Predicted Purchase Amount ($)', fontsize=12)
    plt.title('Actual vs. Predicted Purchase Amounts', fontsize=14)
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot4_actual_vs_predicted.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 5: Bar chart of squared errors
    plt.figure(figsize=(10, 6))
    plt.bar(x, squared_errors, color='orange', alpha=0.7, width=0.8)
    
    # Add squared error values as text
    for i in range(len(x)):
        plt.annotate(f"${squared_errors[i]:.2f}", 
                     (x[i], squared_errors[i]), 
                     xytext=(0, 5), 
                     textcoords='offset points',
                     ha='center')
    
    plt.xlabel('Time on Site (minutes)', fontsize=12)
    plt.ylabel('Squared Error ($²)', fontsize=12)
    plt.title('Squared Errors for Each Data Point', fontsize=14)
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot5_squared_errors_bar.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    return saved_files

# Execute visualizations
saved_files = create_visualizations(time_on_site, purchase_amount, y_pred, errors, 
                                   beta_0, beta_1, new_time, predicted_amount, save_dir)

print(f"\nVisualizations saved to: {save_dir}")

# Print summary
print("\nQuestion 9 Solution Summary:")
print(f"1. Least squares estimates: β₀ = {beta_0:.2f}, β₁ = {beta_1:.2f}")
print(f"2. Regression equation: Purchase Amount = {beta_0:.2f} + {beta_1:.2f} × Time on Site")
print(f"3. Predicted purchase amount for 10 minutes on site: ${predicted_amount:.2f}")
print(f"4. Mean Squared Error: {mse:.2f}") 