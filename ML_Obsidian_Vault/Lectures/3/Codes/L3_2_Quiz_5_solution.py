import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_2_Quiz_5")
os.makedirs(save_dir, exist_ok=True)

print("# Question 5: Ice Cream Sales Prediction Based on Temperature")

# Define the data from the problem
temperatures = np.array([15, 18, 20, 25, 30, 35])
sales = np.array([4, 5, 6, 8, 11, 15])

print("\n## Data")
print("Temperature (°C) | Ice Cream Sales ($100s)")
print("-----------------|----------------------")
for i in range(len(temperatures)):
    print(f"{temperatures[i]:^16} | {sales[i]:^22}")

print("\n## Part 1: Creating a Simple Linear Regression Model")

# Calculate means
x_mean = np.mean(temperatures)
y_mean = np.mean(sales)

# Calculate slope
numerator = np.sum((temperatures - x_mean) * (sales - y_mean))
denominator = np.sum((temperatures - x_mean) ** 2)
beta_1 = numerator / denominator

# Calculate intercept
beta_0 = y_mean - beta_1 * x_mean

# Calculate predictions
y_pred = beta_0 + beta_1 * temperatures

# Calculate residuals
residuals = sales - y_pred

# Calculate metrics
ss_total = np.sum((sales - y_mean) ** 2)
ss_residual = np.sum(residuals ** 2)
r_squared = 1 - (ss_residual / ss_total)
std_error = np.sqrt(ss_residual / (len(temperatures) - 2))

print(f"Linear Regression Equation: Sales = {beta_0:.4f} + {beta_1:.4f} × Temperature")
print(f"R-squared: {r_squared:.4f}")
print(f"Standard Error: {std_error:.4f}")

print("\nPredicted Sales vs Actual Sales:")
print("Temperature (°C) | Actual Sales | Predicted Sales | Residual")
print("-----------------|--------------|-----------------|----------")
for i in range(len(temperatures)):
    print(f"{temperatures[i]:^16} | {sales[i]:^12} | {y_pred[i]:^15.2f} | {residuals[i]:^10.2f}")

print("\n## Part 2: Visualizing Sales Predictions and Residuals")

# Plot 1: Scatter plot with regression line
plt.figure(figsize=(10, 6))
plt.scatter(temperatures, sales, color='blue', s=100, label='Observed Data')

# Plot the regression line
temp_line = np.linspace(min(temperatures) - 2, max(temperatures) + 2, 100)
sales_line = beta_0 + beta_1 * temp_line

plt.plot(temp_line, sales_line, 'r-', linewidth=2, 
         label=f'Linear Model: Sales = {beta_0:.2f} + {beta_1:.2f} × Temp')

plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('Ice Cream Sales ($100s)', fontsize=12)
plt.title('Ice Cream Sales vs Temperature', fontsize=14)
plt.grid(True)
plt.legend()

file_path = os.path.join(save_dir, "plot1_linear_regression.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Residuals vs Temperature
plt.figure(figsize=(10, 6))
plt.scatter(temperatures, residuals, color='purple', s=100)
plt.axhline(y=0, color='red', linestyle='-')

for i in range(len(temperatures)):
    plt.plot([temperatures[i], temperatures[i]], [0, residuals[i]], 
            'purple', linestyle='--', alpha=0.5)

plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('Residuals vs Temperature', fontsize=14)
plt.grid(True)

file_path = os.path.join(save_dir, "plot2_residuals_vs_temp.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Actual vs Predicted Sales
plt.figure(figsize=(10, 6))

# Line of perfect prediction
max_val = max(max(sales), max(y_pred))
min_val = min(min(sales), min(y_pred))
plt.plot([min_val-1, max_val+1], [min_val-1, max_val+1], 'k--', alpha=0.5, 
         label='Perfect Prediction')

plt.scatter(y_pred, sales, color='green', s=100)

# Add labels for each point
for i in range(len(temperatures)):
    plt.annotate(f"{temperatures[i]}°C", 
                (y_pred[i], sales[i]), 
                xytext=(7, 0), 
                textcoords='offset points')

plt.xlabel('Predicted Sales ($100s)', fontsize=12)
plt.ylabel('Actual Sales ($100s)', fontsize=12)
plt.title('Actual vs Predicted Sales', fontsize=14)
plt.grid(True)
plt.legend()

file_path = os.path.join(save_dir, "plot3_actual_vs_predicted.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Pattern of Errors
plt.figure(figsize=(10, 6))

# Create an array with evenly spaced temperatures
temps_dense = np.linspace(min(temperatures), max(temperatures), 100)

# Create point sizes proportional to the magnitude of residuals
sizes = np.abs(residuals) * 100 + 50

# Color points based on whether they're under or over-predicted
colors = ['red' if r < 0 else 'green' for r in residuals]

plt.scatter(temperatures, sales, c=colors, s=sizes, alpha=0.7, 
           label='Actual Data (green=under-predicted, red=over-predicted)')
plt.plot(temps_dense, beta_0 + beta_1 * temps_dense, 'b-', linewidth=2, 
         label='Linear Model')

plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('Ice Cream Sales ($100s)', fontsize=12)
plt.title('Pattern of Prediction Errors', fontsize=14)
plt.grid(True)
plt.legend()

file_path = os.path.join(save_dir, "plot4_error_pattern.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()

# Plot 5: Transformed Models Comparison
plt.figure(figsize=(10, 6))

# Original data and linear model
plt.scatter(temperatures, sales, color='blue', s=80, label='Observed Data')
plt.plot(temp_line, sales_line, 'r-', linewidth=2, 
         label=f'Linear Model: Sales = {beta_0:.2f} + {beta_1:.2f} × Temp')

# Fit a quadratic model using polynomial features
X = np.column_stack((temperatures**2, temperatures, np.ones_like(temperatures)))
coeffs = np.linalg.lstsq(X, sales, rcond=None)[0]
a, b, c = coeffs

# Generate quadratic predictions
quad_pred = a * temp_line**2 + b * temp_line + c

plt.plot(temp_line, quad_pred, 'g--', linewidth=2, 
         label=f'Quadratic Model: Sales = {a:.4f}×Temp² + {b:.4f}×Temp + {c:.4f}')

# Fit exponential model by transforming
log_sales = np.log(sales)
beta_0_exp = np.sum(log_sales) / len(log_sales) - (beta_1 / beta_0) * np.sum(temperatures) / len(temperatures)
beta_1_exp = np.sum((temperatures - np.mean(temperatures)) * (log_sales - np.mean(log_sales))) / np.sum((temperatures - np.mean(temperatures))**2)
exp_pred = np.exp(beta_0_exp + beta_1_exp * temp_line)

plt.plot(temp_line, exp_pred, 'c:', linewidth=2, 
         label=f'Exponential Model: Sales = exp({beta_0_exp:.4f} + {beta_1_exp:.4f}×Temp)')

plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('Ice Cream Sales ($100s)', fontsize=12)
plt.title('Comparison of Different Models', fontsize=14)
plt.grid(True)
plt.legend()

file_path = os.path.join(save_dir, "plot5_model_comparison.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()

# Plot 6: Linear model with error zone
plt.figure(figsize=(10, 6))
plt.scatter(temperatures, sales, color='blue', s=80, label='Observed Data')

# Add error "comfort zone" around linear predictions
plt.plot(temp_line, sales_line, 'r-', linewidth=2, label='Linear Model')
error_margin = 2 * std_error  # 2 standard errors
plt.fill_between(temp_line, 
                sales_line - error_margin,
                sales_line + error_margin,
                color='red', alpha=0.2, label=f'Error Margin (±{error_margin:.2f})')

plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('Ice Cream Sales ($100s)', fontsize=12)
plt.title('Linear Model with Error Zone', fontsize=14)
plt.grid(True)
plt.legend()

file_path = os.path.join(save_dir, "plot6_error_zone.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()

saved_files = [
    "plot1_linear_regression.png",
    "plot2_residuals_vs_temp.png",
    "plot3_actual_vs_predicted.png",
    "plot4_error_pattern.png",
    "plot5_model_comparison.png",
    "plot6_error_zone.png"
]

print("\n## Part 3: Analysis of Error Patterns")
print("\nPattern observed in the residuals:")
print("- The linear model tends to over-predict sales at moderate temperatures (around 20-25°C)")
print("- The linear model under-predicts sales at higher temperatures (30-35°C)")
print("- This systematic pattern suggests a nonlinear relationship rather than random variation")
print("- The residuals show a curved pattern, increasing as temperature increases")

print("\n## Part 4: Linear Regression Assumption Violation and Suggested Transformation")
print("\nViolated Assumption:")
print("- Linearity: The relationship between temperature and ice cream sales appears to be nonlinear")
print("- The sales increase more dramatically at higher temperatures than a linear model predicts")
print("- This violates the fundamental assumption that the relationship is linear")

print("\nSuggested Transformations:")
print("1. Quadratic transformation: Include a squared term for temperature")
print("   - Sales = a·Temperature² + b·Temperature + c")
print("   - This can model the accelerating relationship at higher temperatures")
print("\n2. Exponential transformation: Model the exponential growth in sales")
print("   - Sales = exp(a + b·Temperature)")
print("   - This fits situations where growth rate increases with the predictor")
print("\n3. Log-transform the predictor: Use the logarithm of temperature")
print("   - Sales = a + b·log(Temperature)")
print("   - This can model relationships where the effect diminishes at higher values")

print("\n## Generated Visualizations")
for i, file_path in enumerate(saved_files, 1):
    print(f"{i}. {file_path}")

print("\nCode execution complete.") 