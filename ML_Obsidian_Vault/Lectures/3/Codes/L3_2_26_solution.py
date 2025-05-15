import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
import scipy.stats as stats

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(script_dir):
    os.makedirs(script_dir, exist_ok=True)
    
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_2_Quiz_26")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Given values
sum_x_minus_xbar_squared = 50
sum_y_minus_ybar_squared = 200
sum_xy_minus_xbar_ybar = 80

print("Question 26: Linear Regression Analysis")
print("\nGiven:")
print(f"Σ(xi - x̄)² = {sum_x_minus_xbar_squared}")
print(f"Σ(yi - ȳ)² = {sum_y_minus_ybar_squared}")
print(f"Σ(xi - x̄)(yi - ȳ) = {sum_xy_minus_xbar_ybar}")
print("\n" + "="*50)

# Step 1: Calculate the slope coefficient w1
def calculate_slope():
    """Calculate the slope coefficient w1 using the formula."""
    w1 = sum_xy_minus_xbar_ybar / sum_x_minus_xbar_squared
    
    print("\nStep 1: Calculate the slope coefficient w1")
    print(f"w1 = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²")
    print(f"w1 = {sum_xy_minus_xbar_ybar} / {sum_x_minus_xbar_squared}")
    print(f"w1 = {w1}")
    
    return w1

w1 = calculate_slope()

# Step 2: Calculate the correlation coefficient r
def calculate_correlation():
    """Calculate the correlation coefficient r using the formula."""
    denominator = np.sqrt(sum_x_minus_xbar_squared * sum_y_minus_ybar_squared)
    r = sum_xy_minus_xbar_ybar / denominator
    
    print("\nStep 2: Calculate the correlation coefficient r")
    print(f"r = Σ(xi - x̄)(yi - ȳ) / √[Σ(xi - x̄)² × Σ(yi - ȳ)²]")
    print(f"r = {sum_xy_minus_xbar_ybar} / √({sum_x_minus_xbar_squared} × {sum_y_minus_ybar_squared})")
    print(f"r = {sum_xy_minus_xbar_ybar} / √{sum_x_minus_xbar_squared * sum_y_minus_ybar_squared}")
    print(f"r = {sum_xy_minus_xbar_ybar} / {denominator}")
    print(f"r = {r}")
    
    return r

r = calculate_correlation()

# Step 3: Calculate the coefficient of determination R²
def calculate_r_squared():
    """Calculate the coefficient of determination R² and show its relationship with r."""
    r_squared = r**2
    
    print("\nStep 3: Calculate the coefficient of determination R²")
    print(f"R² = r²")
    print(f"R² = ({r})²")
    print(f"R² = {r_squared}")
    
    # Alternative calculation to verify the relationship
    print("\nAlternative calculation:")
    print(f"In simple linear regression, R² is the square of the correlation coefficient r.")
    print(f"This gives us R² = r² = ({r})² = {r_squared}")
    
    return r_squared

r_squared = calculate_r_squared()

# Step 4: Explain the relationship between r and R²
def explain_relationship():
    """Explain the relationship between r and R² in simple linear regression."""
    print("\nStep 4: Relationship between r and R²")
    print("In simple linear regression (with only one predictor variable):")
    print("1. R² is exactly equal to the square of the correlation coefficient r, i.e., R² = r².")
    print("2. This means that the proportion of variance explained by the regression model (R²)")
    print("   is equal to the squared correlation between x and y.")
    print("3. If r is positive, the slope coefficient w1 is positive; if r is negative, w1 is negative.")
    print(f"4. In this case, r = {r} and R² = {r_squared}, which means that {r_squared*100:.2f}% of the")
    print("   variability in y is explained by the linear relationship with x.")
    print(f"5. The positive value of r = {r} indicates a positive linear relationship,")
    print("   which is consistent with the positive slope coefficient w1 = {:.2f}.".format(w1))

explain_relationship()

# Create visualizations
def create_visualizations():
    """Create visualizations to help understand the concepts."""
    saved_files = []
    
    # Plot 1: Visualize the relationship between w1, r, and R²
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a dummy dataset that matches our statistics
    # We'll create x and y such that their statistics match our given values
    np.random.seed(42)
    n = 50  # Sample size
    
    # We know: 
    # Σ(xi - x̄)² = 50
    # This means var(x) = 50/(n-1) because var(x) = Σ(xi - x̄)²/(n-1)
    var_x = sum_x_minus_xbar_squared / (n-1)
    std_x = np.sqrt(var_x)
    
    # Similarly for y:
    var_y = sum_y_minus_ybar_squared / (n-1)
    std_y = np.sqrt(var_y)
    
    # We want the correlation to be r
    # Create uncorrelated standard normal variables
    x_std = np.random.normal(0, 1, n)
    z = np.random.normal(0, 1, n)
    
    # Create correlated y using the formula y = r*x + sqrt(1-r²)*z
    y_std = r * x_std + np.sqrt(1 - r**2) * z
    
    # Scale to match the desired standard deviations and add means
    mean_x = 5  # Arbitrary mean
    mean_y = 10  # Arbitrary mean
    
    x = mean_x + std_x * x_std
    y = mean_y + std_y * y_std
    
    # Calculate actual statistics from our data to verify
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    sum_x_diff_sq_actual = np.sum((x - x_mean)**2)
    sum_y_diff_sq_actual = np.sum((y - y_mean)**2)
    sum_xy_diff_actual = np.sum((x - x_mean) * (y - y_mean))
    corr_actual = np.corrcoef(x, y)[0, 1]
    
    print("\nVerification of our synthetic data:")
    print(f"Actual Σ(xi - x̄)² ≈ {sum_x_diff_sq_actual:.2f} (vs. given 50)")
    print(f"Actual Σ(yi - ȳ)² ≈ {sum_y_diff_sq_actual:.2f} (vs. given 200)")
    print(f"Actual Σ(xi - x̄)(yi - ȳ) ≈ {sum_xy_diff_actual:.2f} (vs. given 80)")
    print(f"Actual correlation coefficient r ≈ {corr_actual:.4f} (vs. calculated {r:.4f})")
    
    # Adjust data to match the given values exactly (scaling)
    scale_factor_x = np.sqrt(sum_x_minus_xbar_squared / sum_x_diff_sq_actual)
    scale_factor_y = np.sqrt(sum_y_minus_ybar_squared / sum_y_diff_sq_actual)
    
    x = x_mean + (x - x_mean) * scale_factor_x
    y = y_mean + (y - y_mean) * scale_factor_y
    
    # Recalculate to verify
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    sum_x_diff_sq_adjusted = np.sum((x - x_mean)**2)
    sum_y_diff_sq_adjusted = np.sum((y - y_mean)**2)
    sum_xy_diff_adjusted = np.sum((x - x_mean) * (y - y_mean))
    
    print("\nAfter adjustment:")
    print(f"Adjusted Σ(xi - x̄)² = {sum_x_diff_sq_adjusted:.4f}")
    print(f"Adjusted Σ(yi - ȳ)² = {sum_y_diff_sq_adjusted:.4f}")
    print(f"Adjusted Σ(xi - x̄)(yi - ȳ) = {sum_xy_diff_adjusted:.4f}")
    
    # Final adjustment to match the covariance exactly
    cov_scale = sum_xy_minus_xbar_ybar / sum_xy_diff_adjusted
    y = y_mean + (y - y_mean) * cov_scale
    
    # Calculate regression line using our computed slope
    x_line = np.linspace(min(x), max(x), 100)
    
    # For slope w1, we need to calculate intercept w0
    # w0 = ȳ - w1 * x̄
    y_mean_final = np.mean(y)
    x_mean_final = np.mean(x)
    w0 = y_mean_final - w1 * x_mean_final
    
    y_line = w0 + w1 * x_line
    
    # Plot the data and regression line
    plt.scatter(x, y, color='blue', alpha=0.6, label='Data Points')
    plt.plot(x_line, y_line, color='red', linewidth=2, label=f'Regression Line: y = {w0:.2f} + {w1:.2f}x')
    
    # Calculate the R² to show the proportion of variance explained
    y_pred = w0 + w1 * x
    SS_total = np.sum((y - y_mean_final)**2)
    SS_explained = np.sum((y_pred - y_mean_final)**2)
    SS_residual = np.sum((y - y_pred)**2)
    
    R_sq_calculated = SS_explained / SS_total
    
    # Add text annotations for key metrics
    plt.text(0.05, 0.95, f"Slope (w1) = {w1:.4f}", transform=plt.gca().transAxes, fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8))
    plt.text(0.05, 0.90, f"Correlation (r) = {r:.4f}", transform=plt.gca().transAxes, fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8))
    plt.text(0.05, 0.85, f"R² = {r_squared:.4f}", transform=plt.gca().transAxes, fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title('Linear Regression: Relationship Between Data and Model', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    file_path = os.path.join(save_dir, "plot1_regression_relationship.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    saved_files.append(file_path)
    plt.close()
    
    # Plot 2: Visualize the concept of R² as proportion of variance explained
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot the data again
    axes[0].scatter(x, y, color='blue', alpha=0.6)
    axes[0].plot(x_line, y_line, color='red', linewidth=2, 
              label=f'Regression Line')
    
    # Add horizontal line at y mean
    axes[0].axhline(y=y_mean_final, color='green', linestyle='--', 
                 label='Mean of y')
    
    # First plot: Total variation and explained variation
    for i, xi in enumerate(x):
        # Plot the vertical line from point to mean (total deviation)
        axes[0].plot([xi, xi], [y_mean_final, y[i]], color='green', alpha=0.3)
        
        # Plot vertical line from regression line to mean (explained deviation)
        y_pred_i = w0 + w1 * xi
        axes[0].plot([xi, xi], [y_mean_final, y_pred_i], color='red', alpha=0.3)
    
    axes[0].set_title('Total Variation vs. Explained Variation', fontsize=12)
    axes[0].set_xlabel('x', fontsize=10)
    axes[0].set_ylabel('y', fontsize=10)
    axes[0].legend(fontsize=10)
    axes[0].grid(True)
    
    # Second plot: Residual visualization
    axes[1].scatter(x, y, color='blue', alpha=0.6)
    axes[1].plot(x_line, y_line, color='red', linewidth=2, 
              label=f'Regression Line')
    
    # Plot residuals
    for i, xi in enumerate(x):
        y_pred_i = w0 + w1 * xi
        # Plot the vertical line from point to regression line (residual)
        axes[1].plot([xi, xi], [y_pred_i, y[i]], color='purple', alpha=0.5)
    
    # Add annotation for R²
    axes[1].text(0.05, 0.95, f"R² = {r_squared:.4f}", transform=axes[1].transAxes, fontsize=12,
               bbox=dict(facecolor='white', alpha=0.8))
    axes[1].text(0.05, 0.89, f"SS_total = {SS_total:.2f}", transform=axes[1].transAxes, fontsize=12,
               bbox=dict(facecolor='white', alpha=0.8))
    axes[1].text(0.05, 0.83, f"SS_explained = {SS_explained:.2f}", transform=axes[1].transAxes, fontsize=12,
               bbox=dict(facecolor='white', alpha=0.8))
    axes[1].text(0.05, 0.77, f"SS_residual = {SS_residual:.2f}", transform=axes[1].transAxes, fontsize=12,
               bbox=dict(facecolor='white', alpha=0.8))
    
    axes[1].set_title('Residuals: Unexplained Variation', fontsize=12)
    axes[1].set_xlabel('x', fontsize=10)
    axes[1].set_ylabel('y', fontsize=10)
    axes[1].legend(fontsize=10)
    axes[1].grid(True)
    
    plt.suptitle('Understanding R² as Proportion of Variance Explained', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    file_path = os.path.join(save_dir, "plot2_r_squared_visualization.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    saved_files.append(file_path)
    plt.close()
    
    # Plot 3: Visual explanation of the relationship between r and R²
    correlation_values = np.linspace(-1, 1, 100)
    r_squared_values = correlation_values**2
    
    plt.figure(figsize=(10, 6))
    plt.plot(correlation_values, r_squared_values, linewidth=3, color='blue')
    
    # Highlight our specific r and R² values
    plt.scatter([r], [r_squared], color='red', s=100, zorder=5, 
               label=f'Our case: r={r:.4f}, R²={r_squared:.4f}')
    
    # Add vertical and horizontal lines to highlight the relationship
    plt.axvline(x=r, color='red', linestyle='--', alpha=0.5)
    plt.axhline(y=r_squared, color='red', linestyle='--', alpha=0.5)
    
    # Add some additional points to show the relationship
    highlight_rs = [-0.8, -0.6, -0.3, 0.3, 0.6, 0.8]
    for highlight_r in highlight_rs:
        highlight_r2 = highlight_r**2
        plt.scatter([highlight_r], [highlight_r2], color='gray', s=50, alpha=0.7)
        plt.text(highlight_r+0.05, highlight_r2, f'r={highlight_r}, R²={highlight_r2:.2f}', 
                fontsize=8, alpha=0.7)
    
    plt.title('Relationship Between Correlation Coefficient (r) and Coefficient of Determination (R²)', fontsize=14)
    plt.xlabel('Correlation Coefficient (r)', fontsize=12)
    plt.ylabel('Coefficient of Determination (R²)', fontsize=12)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True)
    plt.legend(fontsize=12)
    
    # Add text explanation
    plt.text(0.02, 0.75, 'In simple linear regression:\nR² = r²\n\nThis means R² is always non-negative,\neven when r is negative.', 
            transform=plt.gca().transAxes, fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    file_path = os.path.join(save_dir, "plot3_r_and_r_squared_relationship.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    saved_files.append(file_path)
    plt.close()
    
    return saved_files

saved_files = create_visualizations()

print("\nVisualizations saved to:")
for file in saved_files:
    print(f"- {file}")

print("\nSummary of Results:")
print(f"1. Slope coefficient (w1) = {w1}")
print(f"2. Correlation coefficient (r) = {r}")
print(f"3. Coefficient of determination (R²) = {r_squared}")
print("4. Relationship: In simple linear regression, R² = r²") 