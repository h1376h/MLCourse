import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_2_Quiz_1")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Define the data from Question 1
house_sizes = np.array([1000, 1500, 2000, 2500, 3000])
house_prices = np.array([150, 200, 250, 300, 350])

# Step 1: Calculate the least squares estimates for slope and intercept
def calculate_least_squares(x, y):
    """Calculate the least squares estimates for slope and intercept."""
    n = len(x)
    
    # Calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    print(f"Step 1.1: Calculate means")
    print(f"Mean of house sizes (x̄): {x_mean} sq ft")
    print(f"Mean of house prices (ȳ): ${y_mean}k")
    print()
    
    # Calculate sum of squares and cross-products
    numerator = 0
    denominator = 0
    
    for i in range(n):
        numerator += (x[i] - x_mean) * (y[i] - y_mean)
        denominator += (x[i] - x_mean) ** 2
    
    # Calculate slope
    beta_1 = numerator / denominator
    
    # Calculate intercept
    beta_0 = y_mean - beta_1 * x_mean
    
    print(f"Step 1.2: Calculate slope (β₁)")
    print(f"Numerator: Sum of (x_i - x̄)(y_i - ȳ) = {numerator}")
    print(f"Denominator: Sum of (x_i - x̄)² = {denominator}")
    print(f"β₁ = Numerator / Denominator = {beta_1}")
    print()
    
    print(f"Step 1.3: Calculate intercept (β₀)")
    print(f"β₀ = ȳ - β₁·x̄ = {y_mean} - {beta_1} · {x_mean} = {beta_0}")
    print()
    
    return beta_0, beta_1

# Execute Step 1
beta_0, beta_1 = calculate_least_squares(house_sizes, house_prices)

# Print the regression equation
print(f"Linear Regression Equation: Price = {beta_0:.2f} + {beta_1:.4f} · Size")
print()

# Step 2: Interpret the slope coefficient
print("Step 2: Interpret the slope coefficient")
print(f"The slope coefficient (β₁ = {beta_1:.4f}) represents the change in house price (in $1000s)")
print(f"for each additional square foot of house size.")
print(f"In other words, for every additional square foot, the house price increases by ${beta_1*1000:.2f}.")
print()

# Step 3: Calculate prediction for a house with 1800 square feet
new_size = 1800
predicted_price = beta_0 + beta_1 * new_size

print("Step 3: Calculate prediction for a house with 1800 square feet")
print(f"Predicted price = {beta_0:.2f} + {beta_1:.4f} · 1800 = ${predicted_price:.2f}k")
print()

# Step 4: Calculate residuals and residual sum of squares (RSS)
def calculate_residuals(x, y, beta_0, beta_1):
    """Calculate residuals and residual sum of squares."""
    # Calculate predicted values
    y_pred = beta_0 + beta_1 * x
    
    # Calculate residuals
    residuals = y - y_pred
    
    # Calculate RSS
    rss = np.sum(residuals ** 2)
    
    print("Step 4: Calculate residuals and residual sum of squares (RSS)")
    print("House Size (x) | Price (y) | Predicted Price (ŷ) | Residual (y - ŷ)")
    print("----------------------------------------------------------------")
    
    for i in range(len(x)):
        print(f"{x[i]:^13} | ${y[i]:^8}k | ${y_pred[i]:^17.2f}k | ${residuals[i]:^14.2f}k")
    
    print(f"\nResidual Sum of Squares (RSS) = {rss:.4f}")
    
    return y_pred, residuals, rss

# Execute Step 4
y_pred, residuals, rss = calculate_residuals(house_sizes, house_prices, beta_0, beta_1)

# Create visualizations
def create_visualizations(x, y, y_pred, residuals, beta_0, beta_1, new_x=None, new_y=None, save_dir=None):
    """Create visualizations to help understand the regression analysis."""
    saved_files = []
    
    # Plot 1: Scatter plot with regression line
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', s=100, label='Observed Data')
    
    # Generate points for the regression line
    x_line = np.linspace(min(x) - 200, max(x) + 200, 100)
    y_line = beta_0 + beta_1 * x_line
    
    plt.plot(x_line, y_line, color='red', linestyle='-', linewidth=2, label=f'Regression Line: y = {beta_0:.2f} + {beta_1:.4f}x')
    
    # If we have a new prediction point, add it
    if new_x is not None and new_y is not None:
        plt.scatter(new_x, new_y, color='green', s=100, label=f'Prediction: ${new_y:.2f}k')
        plt.plot([new_x, new_x], [0, new_y], 'g--', alpha=0.5)
    
    plt.xlabel('House Size (sq ft)', fontsize=12)
    plt.ylabel('House Price ($1000s)', fontsize=12)
    plt.title('Linear Regression: House Price vs Size', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot1_regression_line.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 2: Residuals plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, residuals, color='purple', s=100)
    plt.axhline(y=0, color='red', linestyle='-', alpha=0.7)
    
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [0, residuals[i]], 'purple', linestyle='--', alpha=0.5)
    
    plt.xlabel('House Size (sq ft)', fontsize=12)
    plt.ylabel('Residuals ($1000s)', fontsize=12)
    plt.title('Residuals Plot', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot2_residuals.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 3: Visual representation of the least squares principle
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', s=100, label='Observed Data')
    plt.plot(x_line, y_line, color='red', linestyle='-', linewidth=2, label='Regression Line')
    
    # Draw lines for the squared residuals
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [y[i], y_pred[i]], 'k--', alpha=0.7)
        
        # Create rectangle to represent squared error
        width = 100  # Width of the squared error illustration
        rect_x = x[i] - width/2
        
        if residuals[i] >= 0:
            rect_y = y_pred[i]
            rect_height = residuals[i]
        else:
            rect_y = y[i]
            rect_height = -residuals[i]
            
        plt.gca().add_patch(plt.Rectangle((rect_x, rect_y), width, rect_height, 
                                    color='orange', alpha=0.4))
    
    plt.xlabel('House Size (sq ft)', fontsize=12)
    plt.ylabel('House Price ($1000s)', fontsize=12)
    plt.title('Visualization of Squared Residuals', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot3_squared_residuals.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 4: Actual vs. Predicted Values
    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred, color='green', s=100)
    
    # Add reference line (perfect predictions)
    min_val = min(min(y), min(y_pred))
    max_val = max(max(y), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    
    # Add data point labels
    for i in range(len(x)):
        plt.annotate(f"{x[i]} sq ft", (y[i], y_pred[i]), 
                    xytext=(7, 3), textcoords='offset points')
    
    plt.xlabel('Actual Price ($1000s)', fontsize=12)
    plt.ylabel('Predicted Price ($1000s)', fontsize=12)
    plt.title('Actual vs. Predicted Prices', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot4_actual_vs_predicted.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 5: Histogram of Residuals
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=5, color='purple', alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='-', alpha=0.7)
    plt.xlabel('Residuals ($1000s)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Histogram of Residuals', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot5_residuals_histogram.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    return saved_files

# Execute visualizations
saved_files = create_visualizations(house_sizes, house_prices, y_pred, residuals, beta_0, beta_1, new_size, predicted_price, save_dir)

print(f"\nVisualizations saved to: {', '.join(saved_files)}")
print("\nQuestion 1 Solution Summary:")
print(f"1. Least squares estimates: β₀ = {beta_0:.2f}, β₁ = {beta_1:.4f}")
print(f"2. Slope interpretation: For each additional square foot, house price increases by ${beta_1*1000:.2f}")
print(f"3. Predicted price for a 1800 sq ft house: ${predicted_price:.2f}k")
print(f"4. Residual Sum of Squares (RSS): {rss:.4f}") 