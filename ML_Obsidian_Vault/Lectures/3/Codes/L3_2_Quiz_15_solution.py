import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_2_Quiz_15")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Define the data from Question 15
ages = np.array([43, 21, 25, 42, 57, 59])
glucose_levels = np.array([99, 65, 79, 75, 87, 81])

# Step 1: Calculate the linear regression equation
def calculate_least_squares(x, y):
    """Calculate the least squares estimates for slope and intercept."""
    n = len(x)
    
    # Calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    print(f"Step 1.1: Calculate means")
    print(f"Mean of ages (x̄): {x_mean:.2f} years")
    print(f"Mean of glucose levels (ȳ): {y_mean:.2f} units")
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
    print(f"Numerator: Sum of (x_i - x̄)(y_i - ȳ) = {numerator:.2f}")
    print(f"Denominator: Sum of (x_i - x̄)² = {denominator:.2f}")
    print(f"β₁ = Numerator / Denominator = {beta_1:.4f}")
    print()
    
    print(f"Step 1.3: Calculate intercept (β₀)")
    print(f"β₀ = ȳ - β₁·x̄ = {y_mean:.2f} - {beta_1:.4f} · {x_mean:.2f} = {beta_0:.4f}")
    print()
    
    return beta_0, beta_1

# Execute Step 1
beta_0, beta_1 = calculate_least_squares(ages, glucose_levels)

# Print the regression equation
print(f"Linear Regression Equation: Glucose Level = {beta_0:.4f} + {beta_1:.4f} · Age")
print()

# Step 2: Calculate the correlation coefficient
def calculate_correlation(x, y):
    """Calculate the correlation coefficient between x and y."""
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Calculate numerator (covariance)
    numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
    
    # Calculate denominators (standard deviations)
    x_std = np.sqrt(sum((x[i] - x_mean) ** 2 for i in range(n)))
    y_std = np.sqrt(sum((y[i] - y_mean) ** 2 for i in range(n)))
    
    # Calculate correlation
    r = numerator / (x_std * y_std)
    
    print(f"Step 2: Calculate correlation coefficient (r)")
    print(f"Numerator (Covariance): {numerator:.2f}")
    print(f"Denominator (std_x * std_y): {x_std:.2f} * {y_std:.2f} = {x_std * y_std:.2f}")
    print(f"Correlation coefficient (r) = {r:.4f}")
    print()
    
    return r

# Execute Step 2
r = calculate_correlation(ages, glucose_levels)

# Step 3: Predict glucose level for a 55-year-old subject
def predict_glucose(age, beta_0, beta_1):
    """Predict glucose level for a given age."""
    predicted_glucose = beta_0 + beta_1 * age
    
    print(f"Step 3: Predict glucose level for a 55-year-old subject")
    print(f"Predicted glucose level = {beta_0:.4f} + {beta_1:.4f} · 55 = {predicted_glucose:.2f}")
    print()
    
    return predicted_glucose

# Execute Step 3
prediction_age = 55
predicted_glucose = predict_glucose(prediction_age, beta_0, beta_1)

# Step 4: Calculate coefficient of determination (R²)
def calculate_r_squared(x, y, beta_0, beta_1, r):
    """Calculate R² and interpret the result."""
    n = len(x)
    y_mean = np.mean(y)
    
    # Calculate predicted values
    y_pred = beta_0 + beta_1 * x
    
    # Calculate total sum of squares (TSS)
    tss = sum((y[i] - y_mean) ** 2 for i in range(n))
    
    # Calculate residual sum of squares (RSS)
    rss = sum((y[i] - y_pred[i]) ** 2 for i in range(n))
    
    # Calculate explained sum of squares (ESS)
    ess = tss - rss
    
    # Calculate R²
    r_squared = ess / tss
    # Alternative: r_squared = r**2
    
    print(f"Step 4: Calculate coefficient of determination (R²)")
    print(f"Total sum of squares (TSS): {tss:.2f}")
    print(f"Residual sum of squares (RSS): {rss:.2f}")
    print(f"Explained sum of squares (ESS): {ess:.2f}")
    print(f"R² = ESS / TSS = {r_squared:.4f}")
    print(f"R² = r² = {r**2:.4f} (should match above)")
    print(f"Interpretation: {r_squared*100:.2f}% of the variation in glucose levels can be explained by age.")
    print()
    
    return r_squared, y_pred

# Execute Step 4
r_squared, y_pred = calculate_r_squared(ages, glucose_levels, beta_0, beta_1, r)

# Create visualizations
def create_visualizations(x, y, y_pred, beta_0, beta_1, pred_x, pred_y, r, r_squared, save_dir=None):
    """Create visualizations to help understand the regression analysis."""
    saved_files = []
    
    # Plot 1: Scatter plot with regression line
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of original data
    plt.scatter(x, y, color='blue', s=100, label='Observed Data')
    
    # Generate points for the regression line
    x_line = np.linspace(min(x) - 5, max(x) + 5, 100)
    y_line = beta_0 + beta_1 * x_line
    
    # Plot regression line
    plt.plot(x_line, y_line, color='red', linestyle='-', linewidth=2, 
             label=f'Regression Line')
    
    # Add prediction point
    plt.scatter(pred_x, pred_y, color='green', s=100, 
                label=f'Prediction (55, {pred_y:.2f})')
    plt.plot([pred_x, pred_x], [0, pred_y], 'g--', alpha=0.5)
    
    # Add mean point
    plt.scatter(np.mean(x), np.mean(y), color='purple', s=100, 
                label=f'Mean ({np.mean(x):.2f}, {np.mean(y):.2f})')
    
    # Add title, labels, and legend
    plt.title('Glucose Level vs Age', fontsize=14)
    plt.xlabel('Age (years)', fontsize=12)
    plt.ylabel('Glucose Level', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot1_regression_line.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 2: Residuals plot
    plt.figure(figsize=(10, 6))
    
    # Calculate residuals
    residuals = y - y_pred
    
    # Scatter plot of residuals
    plt.scatter(x, residuals, color='purple', s=100)
    plt.axhline(y=0, color='red', linestyle='-', alpha=0.7)
    
    # Add vertical lines to represent residuals
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [0, residuals[i]], 'purple', linestyle='--', alpha=0.5)
    
    # Add title, labels
    plt.title('Residuals Plot', fontsize=14)
    plt.xlabel('Age (years)', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot2_residuals.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 3: Actual vs. Predicted Values
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of actual vs predicted
    plt.scatter(y, y_pred, color='green', s=100)
    
    # Add reference line (perfect predictions)
    min_val = min(min(y), min(y_pred))
    max_val = max(max(y), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, 
             label='Perfect Prediction Line')
    
    # Add title, labels, and legend
    plt.title('Actual vs. Predicted Glucose Levels', fontsize=14)
    plt.xlabel('Actual Glucose Level', fontsize=12)
    plt.ylabel('Predicted Glucose Level', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot3_actual_vs_predicted.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 4: Explained vs Unexplained Variance (Pie Chart)
    plt.figure(figsize=(8, 8))
    
    # Create pie chart
    labels = [f'Explained Variance ({r_squared*100:.1f}%)', 
              f'Unexplained Variance ({(1-r_squared)*100:.1f}%)']
    sizes = [r_squared, 1-r_squared]
    colors = ['#5cb85c', '#d9534f']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, shadow=True, explode=(0.1, 0))
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    # Add title
    plt.title('Coefficient of Determination (R²)', fontsize=14)
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot4_r_squared_pie.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    return saved_files

# Execute visualizations
saved_files = create_visualizations(
    ages, glucose_levels, y_pred, beta_0, beta_1, 
    prediction_age, predicted_glucose, r, r_squared, save_dir
)

print(f"\nVisualizations saved to: {', '.join(saved_files)}")
print("\nQuestion 15 Solution Summary:")
print(f"1. Linear regression equation: Glucose Level = {beta_0:.4f} + {beta_1:.4f} * Age")
print(f"2. Correlation coefficient: r = {r:.4f}")
print(f"3. Predicted glucose level for a 55-year-old: {predicted_glucose:.2f}")
print(f"4. Coefficient of determination (R²): {r_squared:.4f}")
print(f"   {r_squared*100:.2f}% of the variation in glucose levels can be explained by age.") 