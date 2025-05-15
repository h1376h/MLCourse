import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_2_Quiz_6")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Define the data from Question 6
sleep_hours = np.array([5, 6, 7, 8, 9, 10])
cognitive_scores = np.array([65, 70, 80, 85, 88, 90])
std_sleep = 1.8  # Given standard deviation of sleep hours
std_scores = 9.6  # Given standard deviation of cognitive scores

# Step 1: Calculate correlation coefficient
def calculate_correlation(x, y):
    """Calculate the correlation coefficient between x and y."""
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Calculate covariance numerator sum
    cov_sum = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
    
    # Calculate variance denominators
    x_var_sum = sum((x[i] - x_mean) ** 2 for i in range(n))
    y_var_sum = sum((y[i] - y_mean) ** 2 for i in range(n))
    
    # Calculate correlation coefficient
    r = cov_sum / (np.sqrt(x_var_sum) * np.sqrt(y_var_sum))
    
    print(f"Step 1: Calculate the correlation coefficient (r)")
    print(f"r = Cov(x,y) / (σx * σy)")
    print(f"  = {cov_sum:.2f} / ({np.sqrt(x_var_sum):.2f} * {np.sqrt(y_var_sum):.2f})")
    print(f"r = {r:.4f}")
    print()
    
    return r

correlation = calculate_correlation(sleep_hours, cognitive_scores)

# Step 2: Calculate coefficient of determination
def calculate_r_squared(r):
    """Calculate the coefficient of determination (R²) from correlation coefficient."""
    r_squared = r ** 2
    
    print(f"Step 2: Calculate the coefficient of determination (R²)")
    print(f"R² = r² = ({correlation:.4f})² = {r_squared:.4f}")
    print(f"This means that {r_squared:.2%} of the variation in cognitive test scores")
    print(f"can be explained by the variation in hours of sleep.")
    print()
    
    return r_squared

r_squared = calculate_r_squared(correlation)

# Step 3: Calculate slope using correlation coefficient and standard deviations
def calculate_slope(r, std_y, std_x):
    """Calculate the slope using correlation coefficient and standard deviations."""
    beta_1 = r * (std_y / std_x)
    
    print(f"Step 3: Calculate the slope (β₁) using correlation and standard deviations")
    print(f"β₁ = r * (σy / σx)")
    print(f"   = {r:.4f} * ({std_scores:.1f} / {std_sleep:.1f})")
    print(f"   = {beta_1:.4f}")
    print()
    
    return beta_1

beta_1 = calculate_slope(correlation, std_scores, std_sleep)

# Calculate intercept (not directly asked, but needed for visualizations)
def calculate_intercept(x, y, beta_1):
    """Calculate the intercept of the regression line."""
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    beta_0 = y_mean - beta_1 * x_mean
    
    print(f"Additional calculation: Finding the intercept (β₀)")
    print(f"β₀ = ȳ - β₁·x̄")
    print(f"   = {y_mean:.2f} - {beta_1:.4f} * {x_mean:.2f}")
    print(f"   = {beta_0:.4f}")
    print()
    
    return beta_0

beta_0 = calculate_intercept(sleep_hours, cognitive_scores, beta_1)

# Step 4: Explain the proportion of variance
def explain_variance_proportion(r_squared):
    """Explain what proportion of variance is explained."""
    print(f"Step 4: Explain the proportion of variance in cognitive scores explained by sleep")
    print(f"The proportion of variance in cognitive test scores that can be explained")
    print(f"by hours of sleep is {r_squared:.4f} or {r_squared:.2%}.")
    print(f"This means that {r_squared:.2%} of the variability in cognitive test performance")
    print(f"among the participants can be attributed to differences in sleep duration.")
    print(f"The remaining {1-r_squared:.2%} is due to other factors not captured in this model.")
    print()

explain_variance_proportion(r_squared)

# Create visualizations to help understand the results
def create_visualizations(x, y, beta_0, beta_1, r, r_squared, save_dir=None):
    """Create visualizations for the regression analysis."""
    saved_files = []
    
    # Plot 1: Data points with regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', s=100, label='Observed Data')
    
    # Generate points for the regression line
    x_line = np.linspace(min(x) - 0.5, max(x) + 0.5, 100)
    y_line = beta_0 + beta_1 * x_line
    
    plt.plot(x_line, y_line, color='red', linestyle='-', linewidth=2, 
            label=f'Regression Line\ny = {beta_0:.2f} + {beta_1:.2f}x')
    
    plt.xlabel('Hours of Sleep', fontsize=12)
    plt.ylabel('Cognitive Test Score', fontsize=12)
    plt.title('Relationship Between Sleep and Cognitive Performance', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    
    # Add correlation information to the plot
    plt.annotate(f"Correlation (r) = {r:.4f}\nR² = {r_squared:.4f}", 
               xy=(0.05, 0.95), xycoords='axes fraction',
               fontsize=12, ha='left', va='top',
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot1_regression_line.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 2: Correlation visualization
    plt.figure(figsize=(10, 6))
    
    # Calculate z-scores for better visualization
    z_x = (x - np.mean(x)) / np.std(x)
    z_y = (y - np.mean(y)) / np.std(y)
    
    plt.scatter(z_x, z_y, color='purple', s=100)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    
    # Draw line with slope = r through the origin
    z_line = np.linspace(min(z_x) - 0.5, max(z_x) + 0.5, 100)
    plt.plot(z_line, r * z_line, color='red', linestyle='-', linewidth=2,
            label=f'Slope = r = {r:.4f}')
    
    plt.xlabel('Standardized Hours of Sleep (z-score)', fontsize=12)
    plt.ylabel('Standardized Cognitive Score (z-score)', fontsize=12)
    plt.title('Correlation Visualization with Standardized Variables', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    
    # Draw quadrant labels
    plt.annotate('Above average sleep\nAbove average score', 
               xy=(0.85, 0.85), xycoords='axes fraction', 
               fontsize=10, ha='center', bbox=dict(boxstyle="round", fc="lightgreen", alpha=0.3))
    
    plt.annotate('Below average sleep\nBelow average score', 
               xy=(0.15, 0.15), xycoords='axes fraction', 
               fontsize=10, ha='center', bbox=dict(boxstyle="round", fc="lightgreen", alpha=0.3))
    
    plt.annotate('Above average sleep\nBelow average score', 
               xy=(0.85, 0.15), xycoords='axes fraction', 
               fontsize=10, ha='center', bbox=dict(boxstyle="round", fc="lightcoral", alpha=0.3))
    
    plt.annotate('Below average sleep\nAbove average score', 
               xy=(0.15, 0.85), xycoords='axes fraction', 
               fontsize=10, ha='center', bbox=dict(boxstyle="round", fc="lightcoral", alpha=0.3))
    
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot2_correlation_visualization.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 3: R-squared visualization
    plt.figure(figsize=(10, 6))
    
    # Calculate predicted values
    y_pred = beta_0 + beta_1 * x
    
    # Calculate mean predicted value
    y_mean = np.mean(y)
    
    # Set up the plot
    plt.scatter(x, y, color='blue', s=100, label='Actual Scores')
    plt.hlines(y=y, xmin=x, xmax=x, color='green', linestyle='-', linewidth=2, label='Total Variance')
    
    # Draw horizontal lines to visualize total variance
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [y_mean, y[i]], 'green', alpha=0.5, linewidth=2)
    
    # Draw regression line
    plt.plot(x_line, y_line, color='red', linestyle='-', linewidth=2, label='Regression Line')
    
    # Visualize explained and unexplained variance
    for i in range(len(x)):
        # Unexplained variance (distance from actual to predicted)
        plt.plot([x[i], x[i]], [y[i], y_pred[i]], 'red', alpha=0.7, linewidth=2, linestyle='--')
        
        # Explained variance (distance from mean to predicted)
        plt.plot([x[i], x[i]], [y_mean, y_pred[i]], 'blue', alpha=0.7, linewidth=2, linestyle='--')
    
    plt.axhline(y=y_mean, color='black', linestyle='--', alpha=0.7, label=f'Mean Score = {y_mean:.2f}')
    
    plt.xlabel('Hours of Sleep', fontsize=12)
    plt.ylabel('Cognitive Test Score', fontsize=12)
    plt.title(f'R² Visualization: {r_squared:.2%} of Variance Explained', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=10, loc='lower right')
    
    # Add R-squared explanation to the plot
    plt.annotate(f"R² = {r_squared:.4f}\nExplained variance: {r_squared:.2%}\nUnexplained variance: {1-r_squared:.2%}", 
               xy=(0.05, 0.95), xycoords='axes fraction',
               fontsize=12, ha='left', va='top',
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot3_r_squared_visualization.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 4: Using standard deviations to find slope
    plt.figure(figsize=(10, 6))
    
    # Set up the plot with standardized data
    plt.scatter(z_x, z_y, color='blue', s=100, label='Standardized Data')
    
    # Draw slope determination
    plt.plot(z_line, z_line, color='gray', linestyle='--', linewidth=1, label='Slope = 1 (if r = 1)')
    plt.plot(z_line, r * z_line, color='red', linestyle='-', linewidth=2, label=f'Slope = r = {r:.4f}')
    
    # Visualize the relationship between slope, correlation, and standard deviations
    plt.title('Relationship Between Correlation and Regression Slope', fontsize=14)
    plt.xlabel('Standardized Hours of Sleep (z-score)', fontsize=12)
    plt.ylabel('Standardized Cognitive Score (z-score)', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=12)
    
    # Add formula to the plot
    formula_text = f"β₁ = r × (σy/σx) = {r:.4f} × ({std_scores:.1f}/{std_sleep:.1f}) = {beta_1:.2f}"
    plt.annotate(formula_text, xy=(0.5, 0.05), xycoords='axes fraction',
                fontsize=14, ha='center', va='bottom',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot4_slope_determination.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    return saved_files

# Create visualizations
saved_files = create_visualizations(sleep_hours, cognitive_scores, beta_0, beta_1, correlation, r_squared, save_dir)

print(f"Visualizations saved to: {save_dir}")
print("\nQuestion 6 Solution Summary:")
print(f"1. Correlation coefficient (r): {correlation:.4f}")
print(f"2. Coefficient of determination (R²): {r_squared:.4f}")
print(f"3. Regression slope (β₁): {beta_1:.4f}")
print(f"4. Proportion of variance explained: {r_squared:.2%}")
print(f"5. Regression equation: Score = {beta_0:.2f} + {beta_1:.2f} × Hours") 