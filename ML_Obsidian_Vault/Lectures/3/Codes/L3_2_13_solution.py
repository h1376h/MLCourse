import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_2_Quiz_13")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Define the data from Question 13
car_models = ['A', 'B', 'C', 'D', 'E']
weights = np.array([2.5, 3.0, 3.5, 4.0, 4.5])  # in 1000 lbs
mpg = np.array([30, 25, 23, 20, 18])  # miles per gallon

print("Question 13: Car Weight and Fuel Efficiency")
print("=" * 50)
print("\nData:")
print("Car Model | Weight (x) in 1000 lbs | Fuel Efficiency (y) in MPG")
print("-" * 60)
for i in range(len(car_models)):
    print(f"{car_models[i]:^9} | {weights[i]:^20} | {mpg[i]:^25}")
print("=" * 50)

# Step 1: Calculate means, sample covariance, and variance
def calculate_statistics(x, y):
    """Calculate means, sample covariance, and variance needed for regression."""
    n = len(x)
    
    # Calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    print(f"\nStep 1: Calculate means, sample covariance, and variance")
    print(f"1.1: Calculate means")
    print(f"Mean of weights (x̄): {x_mean:.2f} thousand lbs")
    print(f"Mean of fuel efficiency (ȳ): {y_mean:.2f} MPG")
    
    # Calculate covariance
    cov_xy = 0
    var_x = 0
    
    for i in range(n):
        cov_xy += (x[i] - x_mean) * (y[i] - y_mean)
        var_x += (x[i] - x_mean) ** 2
    
    # Sample covariance
    cov_xy = cov_xy / n
    
    # Sample variance
    var_x = var_x / n
    
    print(f"\n1.2: Calculate sample covariance and variance")
    print(f"Sample covariance (Cov(x,y)): {cov_xy:.4f}")
    print(f"Sample variance of weights (Var(x)): {var_x:.4f}")
    
    return x_mean, y_mean, cov_xy, var_x

# Execute Step 1
x_mean, y_mean, cov_xy, var_x = calculate_statistics(weights, mpg)
print("=" * 50)

# Step 2: Determine least squares estimates for slope and intercept
def calculate_regression_coefficients(x_mean, y_mean, cov_xy, var_x):
    """Calculate the regression coefficients using the statistics."""
    # Calculate slope
    beta_1 = cov_xy / var_x
    
    # Calculate intercept
    beta_0 = y_mean - beta_1 * x_mean
    
    print("\nStep 2: Determine least squares estimates for slope and intercept")
    print(f"2.1: Calculate slope (β₁)")
    print(f"β₁ = Cov(x,y) / Var(x) = {cov_xy:.4f} / {var_x:.4f} = {beta_1:.4f}")
    
    print(f"\n2.2: Calculate intercept (β₀)")
    print(f"β₀ = ȳ - β₁·x̄ = {y_mean:.2f} - ({beta_1:.4f} × {x_mean:.2f}) = {beta_0:.4f}")
    
    return beta_0, beta_1

# Execute Step 2
beta_0, beta_1 = calculate_regression_coefficients(x_mean, y_mean, cov_xy, var_x)

# Print the regression equation
print(f"\nLinear Regression Equation: MPG = {beta_0:.4f} + {beta_1:.4f} × Weight")
print("=" * 50)

# Step 3: Interpret the meaning of the slope coefficient
print("\nStep 3: Interpret the meaning of the slope coefficient")
print(f"The slope coefficient (β₁ = {beta_1:.4f}) represents the change in fuel efficiency (MPG)")
print(f"for each additional thousand pounds of car weight.")
print(f"Since the slope is negative ({beta_1:.4f}), this means that for every additional")
print(f"thousand pounds, the fuel efficiency decreases by {abs(beta_1):.2f} MPG.")
print("=" * 50)

# Step 4: Calculate coefficient of determination (R²)
def calculate_r_squared(x, y, beta_0, beta_1, y_mean):
    """Calculate the coefficient of determination (R²)."""
    # Calculate predicted values
    y_pred = beta_0 + beta_1 * x
    
    # Calculate residuals
    residuals = y - y_pred
    
    # Calculate sum of squared residuals (SSR)
    ssr = np.sum(residuals ** 2)
    
    # Calculate total sum of squares (SST)
    sst = np.sum((y - y_mean) ** 2)
    
    # Calculate R²
    r_squared = 1 - (ssr / sst)
    
    print("\nStep 4: Calculate the coefficient of determination (R²)")
    print("4.1: Calculate predicted values and residuals")
    print("Car Model | Weight (x) | MPG (y) | Predicted MPG (ŷ) | Residual (y - ŷ)")
    print("-" * 70)
    
    for i in range(len(x)):
        print(f"{car_models[i]:^9} | {x[i]:^10.1f} | {y[i]:^7} | {y_pred[i]:^16.2f} | {residuals[i]:^16.2f}")
    
    print(f"\n4.2: Calculate Sum of Squared Residuals (SSR)")
    print(f"SSR = Σ(y_i - ŷ_i)² = {ssr:.4f}")
    
    print(f"\n4.3: Calculate Total Sum of Squares (SST)")
    print(f"SST = Σ(y_i - ȳ)² = {sst:.4f}")
    
    print(f"\n4.4: Calculate R²")
    print(f"R² = 1 - (SSR / SST) = 1 - ({ssr:.4f} / {sst:.4f}) = {r_squared:.4f}")
    
    return y_pred, residuals, r_squared

# Execute Step 4
y_pred, residuals, r_squared = calculate_r_squared(weights, mpg, beta_0, beta_1, y_mean)

# Explain R² in context
print(f"\nInterpretation of R²:")
print(f"The coefficient of determination (R² = {r_squared:.4f} or {r_squared*100:.1f}%) indicates that")
print(f"approximately {r_squared*100:.1f}% of the variation in fuel efficiency (MPG)")
print(f"can be explained by the variation in car weight.")
print(f"This suggests a very strong negative linear relationship between weight and fuel efficiency.")
print("=" * 50)

# Create visualizations
def create_visualizations(x, y, beta_0, beta_1, y_pred, residuals, r_squared, save_dir=None):
    """Create visualizations to help understand the regression analysis."""
    saved_files = []
    
    # Figure 1: Scatter plot with regression line
    plt.figure(figsize=(10, 8))
    
    # Plot the data points
    plt.scatter(x, y, color='blue', s=100, label='Observed Data')
    
    # Label each point with the car model
    for i, model in enumerate(car_models):
        plt.annotate(model, (x[i], y[i]), xytext=(5, 5), textcoords='offset points')
    
    # Generate points for the regression line
    x_line = np.linspace(2.0, 5.0, 100)
    y_line = beta_0 + beta_1 * x_line
    
    # Plot regression line
    plt.plot(x_line, y_line, color='red', linestyle='-', linewidth=2, 
             label=f'Regression Line: MPG = {beta_0:.2f} + ({beta_1:.2f}) × Weight')
    
    # Customize the plot
    plt.xlabel('Car Weight (1000 lbs)', fontsize=12)
    plt.ylabel('Fuel Efficiency (MPG)', fontsize=12)
    plt.title('Regression Analysis: Fuel Efficiency vs. Car Weight', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Add R² to the plot
    plt.text(0.05, 0.05, f'R² = {r_squared:.4f}', transform=plt.gca().transAxes, 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "fig1_regression_line.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Figure 2: Residuals plot
    plt.figure(figsize=(10, 6))
    
    # Plot residuals
    plt.scatter(x, residuals, color='purple', s=80, label='Residuals')
    plt.axhline(y=0, color='red', linestyle='-', alpha=0.7)
    
    # Label each point with the car model
    for i, model in enumerate(car_models):
        plt.annotate(model, (x[i], residuals[i]), xytext=(5, 5), textcoords='offset points')
    
    # Add vertical lines to show the magnitude of each residual
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [0, residuals[i]], 'purple', linestyle='--', alpha=0.5)
    
    # Customize the plot
    plt.xlabel('Car Weight (1000 lbs)', fontsize=12)
    plt.ylabel('Residual (MPG)', fontsize=12)
    plt.title('Residuals Plot', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "fig2_residuals_plot.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Figure 3: Actual vs. Predicted Values
    plt.figure(figsize=(8, 8))
    
    # Plot actual vs. predicted values
    plt.scatter(y, y_pred, color='green', s=100)
    
    # Label each point with the car model
    for i, model in enumerate(car_models):
        plt.annotate(model, (y[i], y_pred[i]), xytext=(5, 5), textcoords='offset points')
    
    # Add a diagonal line (perfect predictions)
    min_val = min(min(y), min(y_pred))
    max_val = max(max(y), max(y_pred))
    buffer = (max_val - min_val) * 0.1  # 10% buffer
    plt.plot([min_val - buffer, max_val + buffer], 
             [min_val - buffer, max_val + buffer], 
             'r--', alpha=0.7)
    
    # Customize the plot
    plt.xlabel('Actual MPG', fontsize=12)
    plt.ylabel('Predicted MPG', fontsize=12)
    plt.title('Actual vs. Predicted Fuel Efficiency', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add R² to the plot
    plt.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=plt.gca().transAxes, 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.axis('equal')  # Equal scaling
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "fig3_actual_vs_predicted.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Figure 4: Visual Explanation of R²
    plt.figure(figsize=(12, 8))
    
    gs = GridSpec(1, 2, width_ratios=[1, 1])
    
    # Plot 1: Total Variation (SST)
    ax1 = plt.subplot(gs[0])
    ax1.scatter(x, y, color='blue', s=80)
    
    # Horizontal line at mean
    ax1.axhline(y=y_mean, color='green', linestyle='-', linewidth=2, 
               label=f'Mean MPG (ȳ = {y_mean:.2f})')
    
    # Vertical lines showing deviations from mean
    for i in range(len(x)):
        ax1.plot([x[i], x[i]], [y_mean, y[i]], 'green', linestyle='--', alpha=0.7)
    
    ax1.set_xlabel('Car Weight (1000 lbs)', fontsize=12)
    ax1.set_ylabel('Fuel Efficiency (MPG)', fontsize=12)
    ax1.set_title('Total Variation (SST)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot 2: Unexplained Variation (SSR)
    ax2 = plt.subplot(gs[1])
    ax2.scatter(x, y, color='blue', s=80)
    
    # Regression line
    ax2.plot(x_line, y_line, color='red', linestyle='-', linewidth=2, 
            label='Regression Line')
    
    # Vertical lines showing residuals
    for i in range(len(x)):
        ax2.plot([x[i], x[i]], [y_pred[i], y[i]], 'red', linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Car Weight (1000 lbs)', fontsize=12)
    ax2.set_ylabel('Fuel Efficiency (MPG)', fontsize=12)
    ax2.set_title('Unexplained Variation (SSR)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.suptitle(f'Understanding R² = {r_squared:.4f}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    
    if save_dir:
        file_path = os.path.join(save_dir, "fig4_r_squared_explanation.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Figure 5: Exploring the relationship with different models
    plt.figure(figsize=(10, 8))
    
    # Plot the data points
    plt.scatter(x, y, color='blue', s=100, label='Observed Data')
    
    # Linear model
    x_line = np.linspace(2.0, 5.0, 100)
    y_line_linear = beta_0 + beta_1 * x_line
    plt.plot(x_line, y_line_linear, color='red', linestyle='-', linewidth=2, 
             label=f'Linear Model (R² = {r_squared:.4f})')
    
    # Try a quadratic model (just for comparison)
    X_poly = np.column_stack((np.ones(len(x)), x, x**2))
    beta_poly = np.linalg.lstsq(X_poly, y, rcond=None)[0]
    y_poly = beta_poly[0] + beta_poly[1] * x_line + beta_poly[2] * x_line**2
    
    # Calculate R² for the quadratic model
    y_pred_poly = beta_poly[0] + beta_poly[1] * x + beta_poly[2] * x**2
    ssr_poly = np.sum((y - y_pred_poly) ** 2)
    r_squared_poly = 1 - (ssr_poly / np.sum((y - y_mean) ** 2))
    
    plt.plot(x_line, y_poly, color='green', linestyle='--', linewidth=2, 
             label=f'Quadratic Model (R² = {r_squared_poly:.4f})')
    
    # Try an exponential model (just for comparison)
    # We'll fit log(y) = a + b*x and then transform back
    log_y = np.log(y)
    beta_exp = np.polyfit(x, log_y, 1)
    y_exp = np.exp(beta_exp[1]) * np.exp(beta_exp[0] * x_line)
    
    # Calculate R² for the exponential model
    y_pred_exp = np.exp(beta_exp[1]) * np.exp(beta_exp[0] * x)
    ssr_exp = np.sum((y - y_pred_exp) ** 2)
    r_squared_exp = 1 - (ssr_exp / np.sum((y - y_mean) ** 2))
    
    plt.plot(x_line, y_exp, color='purple', linestyle='-.', linewidth=2, 
             label=f'Exponential Model (R² = {r_squared_exp:.4f})')
    
    # Customize the plot
    plt.xlabel('Car Weight (1000 lbs)', fontsize=12)
    plt.ylabel('Fuel Efficiency (MPG)', fontsize=12)
    plt.title('Comparing Different Regression Models', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "fig5_model_comparison.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    return saved_files

# Execute visualizations
saved_files = create_visualizations(weights, mpg, beta_0, beta_1, y_pred, residuals, r_squared, save_dir)

# Final summary
print("\nFinal Results:")
print(f"Regression Equation: MPG = {beta_0:.4f} + ({beta_1:.4f}) × Weight")
print(f"Slope Interpretation: For each additional 1000 lbs, MPG decreases by {abs(beta_1):.2f}")
print(f"Coefficient of Determination (R²): {r_squared:.4f}")
print(f"Percentage of Variance Explained: {r_squared*100:.1f}%")
print(f"\nVisualizations saved to: {save_dir}")

# Additional insights
print("\nAdditional Insights:")
print("1. The strong negative relationship confirms that heavier cars tend to be less fuel-efficient.")
print("2. The high R² value suggests that weight is a very important factor in determining fuel efficiency.")
print("3. The model could be used to estimate the potential fuel savings from reducing car weight.")
print("4. The linear model fits the data well, though other functional forms might also be worth exploring.") 