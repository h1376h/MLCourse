import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_2_Quiz_10")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Define the data from Question 10
light_exposure = np.array([4, 6, 8, 10, 12])
plant_growth = np.array([2.1, 3.4, 4.7, 5.9, 7.2])

print("Question 10: Light Exposure and Plant Growth")
print("===========================================")
print("A plant biologist is studying the effect of light exposure (hours per day)")
print("on plant growth (in cm). The following data was collected over a week:")
print()
print("| Light Exposure (x) in hours | Plant Growth (y) in cm |")
print("|----------------------------|------------------------|")
for i in range(len(light_exposure)):
    print(f"| {light_exposure[i]:<26} | {plant_growth[i]:<22} |")
print()

# Step 1: Calculate average light exposure and average plant growth
def calculate_means(x, y):
    """Calculate the mean of x and y."""
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    print(f"Step 1: Calculate the means")
    print(f"Mean of light exposure (x̄): {x_mean:.2f} hours")
    print(f"Mean of plant growth (ȳ): {y_mean:.2f} cm")
    print()
    
    return x_mean, y_mean

x_mean, y_mean = calculate_means(light_exposure, plant_growth)

# Step 2: Calculate the covariance between light exposure and plant growth
def calculate_covariance(x, y, x_mean, y_mean):
    """Calculate the covariance between x and y."""
    n = len(x)
    cov_sum = 0
    
    print(f"Step 2: Calculate the covariance between light exposure and plant growth")
    print(f"Cov(x,y) = (1/n) * Σ[(x_i - x̄)(y_i - ȳ)]")
    print(f"Cov(x,y) = (1/{n}) * [", end="")
    
    for i in range(n):
        term = (x[i] - x_mean) * (y[i] - y_mean)
        cov_sum += term
        
        if i < n - 1:
            print(f"({x[i]} - {x_mean:.2f})({y[i]} - {y_mean:.2f}) + ", end="")
        else:
            print(f"({x[i]} - {x_mean:.2f})({y[i]} - {y_mean:.2f})]", end="")
    
    covariance = cov_sum / n
    print(f"\nCov(x,y) = {covariance:.4f}")
    print()
    
    return covariance

covariance = calculate_covariance(light_exposure, plant_growth, x_mean, y_mean)

# Step 3: Calculate the variance of light exposure
def calculate_variance(x, x_mean):
    """Calculate the variance of x."""
    n = len(x)
    var_sum = 0
    
    print(f"Step 3: Calculate the variance of light exposure")
    print(f"Var(x) = (1/n) * Σ[(x_i - x̄)²]")
    print(f"Var(x) = (1/{n}) * [", end="")
    
    for i in range(n):
        term = (x[i] - x_mean) ** 2
        var_sum += term
        
        if i < n - 1:
            print(f"({x[i]} - {x_mean:.2f})² + ", end="")
        else:
            print(f"({x[i]} - {x_mean:.2f})²]", end="")
    
    variance = var_sum / n
    print(f"\nVar(x) = {variance:.4f}")
    print()
    
    return variance

variance = calculate_variance(light_exposure, x_mean)

# Step 4: Calculate the slope and intercept
def calculate_slope_intercept(covariance, variance, x_mean, y_mean):
    """Calculate the slope and intercept using covariance and variance."""
    beta_1 = covariance / variance
    beta_0 = y_mean - beta_1 * x_mean
    
    print(f"Step 4: Calculate the slope (β₁) and intercept (β₀)")
    print(f"β₁ = Cov(x,y) / Var(x) = {covariance:.4f} / {variance:.4f} = {beta_1:.4f}")
    print(f"β₀ = ȳ - β₁·x̄ = {y_mean:.2f} - {beta_1:.4f} · {x_mean:.2f} = {beta_0:.4f}")
    print()
    
    return beta_0, beta_1

beta_0, beta_1 = calculate_slope_intercept(covariance, variance, x_mean, y_mean)

# Print the regression equation
print(f"Growth Prediction Equation: Growth = {beta_0:.4f} + {beta_1:.4f} · Light Exposure")
print()

# Step 5: Calculate expected growth for 9 hours of light
def predict_growth(beta_0, beta_1, hours):
    """Predict plant growth for a given amount of light exposure."""
    growth = beta_0 + beta_1 * hours
    return growth

nine_hour_growth = predict_growth(beta_0, beta_1, 9)
print(f"Step 5: Calculate expected growth for 9 hours of light exposure")
print(f"Growth = {beta_0:.4f} + {beta_1:.4f} · 9 = {nine_hour_growth:.4f} cm")
print()

# Step 6: Calculate additional growth for 2 more hours of light
two_hour_effect = beta_1 * 2
print(f"Step 6: Calculate additional growth for 2 more hours of light")
print(f"Additional growth = β₁ · 2 = {beta_1:.4f} · 2 = {two_hour_effect:.4f} cm")
print()

# Step 7: Calculate predicted values and residuals
def calculate_predictions_residuals(x, y, beta_0, beta_1):
    """Calculate predictions and residuals."""
    # Calculate predicted values
    y_pred = beta_0 + beta_1 * x
    
    # Calculate residuals
    residuals = y - y_pred
    
    # Calculate SSR (sum of squared residuals)
    ssr = np.sum(residuals ** 2)
    
    # Calculate SST (total sum of squares)
    sst = np.sum((y - np.mean(y)) ** 2)
    
    # Calculate R²
    r_squared = 1 - (ssr / sst)
    
    print("Step 7: Calculate predicted values and residuals")
    print("Light Hours (x) | Growth (y) | Predicted Growth (ŷ) | Residual (y - ŷ)")
    print("----------------------------------------------------------------")
    
    for i in range(len(x)):
        print(f"{x[i]:^13} | {y[i]:^9.2f} | {y_pred[i]:^19.4f} | {residuals[i]:^16.4f}")
    
    print()
    print(f"Sum of Squared Residuals (SSR) = {ssr:.4f}")
    print(f"Total Sum of Squares (SST) = {sst:.4f}")
    print(f"R² = 1 - (SSR/SST) = 1 - ({ssr:.4f}/{sst:.4f}) = {r_squared:.4f}")
    print()
    
    return y_pred, residuals, ssr, sst, r_squared

y_pred, residuals, ssr, sst, r_squared = calculate_predictions_residuals(light_exposure, plant_growth, beta_0, beta_1)

# Step 8: Interpret R² value
print("Step 8: Interpret the R² value")
print(f"R² = {r_squared:.4f} means that approximately {r_squared*100:.2f}% of the variation in plant growth")
print(f"can be explained by the variation in light exposure. This indicates a")
print(f"{'very strong' if r_squared > 0.9 else 'strong' if r_squared > 0.7 else 'moderate' if r_squared > 0.5 else 'weak'}")
print(f"linear relationship between light exposure and plant growth.")
print()

# Create visualizations
def create_visualizations(x, y, x_mean, y_mean, y_pred, residuals, beta_0, beta_1, r_squared, save_dir=None):
    """Create visualizations to help understand the regression analysis."""
    saved_files = []
    
    # Plot 1: Data points with means
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='green', s=100, label='Observed Data')
    plt.axhline(y=y_mean, color='orange', linestyle='--', label=f'Mean Growth: {y_mean:.2f} cm')
    plt.axvline(x=x_mean, color='blue', linestyle='--', label=f'Mean Light: {x_mean:.2f} hours')
    
    # Mark the mean point
    plt.scatter([x_mean], [y_mean], color='red', s=150, marker='X', label='Mean Point')
    
    plt.xlabel('Light Exposure (hours/day)', fontsize=12)
    plt.ylabel('Plant Growth (cm)', fontsize=12)
    plt.title('Plant Growth Data with Mean Values', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot1_data_with_means.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 2: Scatter plot with regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='green', s=100, label='Observed Data')
    
    # Generate points for the regression line
    x_line = np.linspace(min(x) - 1, max(x) + 1, 100)
    y_line = beta_0 + beta_1 * x_line
    
    plt.plot(x_line, y_line, color='red', linestyle='-', linewidth=2, 
            label=f'Regression Line\nGrowth = {beta_0:.4f} + {beta_1:.4f} × Light')
    
    # Highlight prediction for 9 hours
    nine_hour_growth = beta_0 + beta_1 * 9
    plt.scatter([9], [nine_hour_growth], color='purple', s=150, 
               label=f'Prediction for 9 hours: {nine_hour_growth:.2f} cm')
    plt.plot([9, 9], [0, nine_hour_growth], 'k--', alpha=0.5)
    plt.plot([0, 9], [nine_hour_growth, nine_hour_growth], 'k--', alpha=0.5)
    
    plt.xlabel('Light Exposure (hours/day)', fontsize=12)
    plt.ylabel('Plant Growth (cm)', fontsize=12)
    plt.title('Linear Regression: Plant Growth vs Light Exposure', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Add R² value to the plot
    plt.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=plt.gca().transAxes, 
            fontsize=12, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot2_regression_line.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 3: Residuals plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, residuals, color='purple', s=100)
    plt.axhline(y=0, color='red', linestyle='-', alpha=0.7)
    
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [0, residuals[i]], 'purple', linestyle='--', alpha=0.5)
    
    plt.xlabel('Light Exposure (hours/day)', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.title('Residuals Plot', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot3_residuals.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 4: Visual demonstration of two-hour effect
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='green', s=100, label='Observed Data')
    plt.plot(x_line, y_line, color='red', linestyle='-', linewidth=2, 
            label=f'Regression Line\nGrowth = {beta_0:.4f} + {beta_1:.4f} × Light')
    
    # Example point
    example_hour = 7
    example_growth = beta_0 + beta_1 * example_hour
    example_growth_plus_2 = beta_0 + beta_1 * (example_hour + 2)
    
    # Plot example point
    plt.scatter([example_hour], [example_growth], color='blue', s=120, 
               label=f'{example_hour} hours: {example_growth:.2f} cm')
    plt.scatter([example_hour + 2], [example_growth_plus_2], color='purple', s=120,
               label=f'{example_hour + 2} hours: {example_growth_plus_2:.2f} cm')
    
    # Connect points
    plt.plot([example_hour, example_hour + 2], [example_growth, example_growth_plus_2], 'k--', linewidth=2)
    
    # Show the difference
    plt.annotate(f'+2 hours = +{beta_1*2:.2f} cm growth', 
               xy=((2*example_hour + 2)/2, (example_growth + example_growth_plus_2)/2),
               xytext=(10, 20),
               textcoords='offset points',
               arrowprops=dict(arrowstyle='->'),
               fontsize=12,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.2))
    
    plt.xlabel('Light Exposure (hours/day)', fontsize=12)
    plt.ylabel('Plant Growth (cm)', fontsize=12)
    plt.title('Effect of 2 Additional Hours of Light Exposure', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot4_two_hour_effect.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 5: R² visualization
    plt.figure(figsize=(12, 8))
    
    # Create a grid for subplots
    gs = GridSpec(2, 2, height_ratios=[3, 1])
    
    # Subplot 1: Model and data
    ax1 = plt.subplot(gs[0, :])
    ax1.scatter(x, y, color='green', s=100, label='Observed Data')
    ax1.plot(x_line, y_line, color='red', linestyle='-', linewidth=2, label='Regression Line')
    ax1.axhline(y=y_mean, color='orange', linestyle='--', label=f'Mean Growth: {y_mean:.2f} cm')
    
    # Draw lines to show deviations
    for i in range(len(x)):
        # Total deviation
        ax1.plot([x[i], x[i]], [y_mean, y[i]], 'k-', alpha=0.3)
        # Explained deviation
        ax1.plot([x[i], x[i]], [y_mean, y_pred[i]], 'g-', alpha=0.5)
        # Unexplained deviation
        ax1.plot([x[i], x[i]], [y_pred[i], y[i]], 'r-', alpha=0.5)
    
    ax1.set_xlabel('Light Exposure (hours/day)', fontsize=12)
    ax1.set_ylabel('Plant Growth (cm)', fontsize=12)
    ax1.set_title('$R^2$ Visualization: Explained vs. Unexplained Variance', fontsize=14)
    ax1.grid(True)
    ax1.legend()
    
    # Subplot 2: Stacked bar chart of variance components
    ax2 = plt.subplot(gs[1, :])
    explained = sst - ssr  # Explained variance
    unexplained = ssr      # Unexplained variance
    
    ax2.bar(['Variance Components'], [explained], label=f'Explained ({r_squared*100:.1f}%)', color='green', alpha=0.6)
    ax2.bar(['Variance Components'], [unexplained], bottom=[explained], label=f'Unexplained ({(1-r_squared)*100:.1f}%)', color='red', alpha=0.6)
    
    # Add text for percentages
    ax2.text(0, explained/2, f'{r_squared*100:.1f}%', ha='center', va='center', fontsize=12, fontweight='bold')
    ax2.text(0, explained + unexplained/2, f'{(1-r_squared)*100:.1f}%', ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax2.set_ylabel('Variance (Sum of Squares)', fontsize=12)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot5_r_squared_visualization.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    return saved_files

# Create visualizations
saved_files = create_visualizations(light_exposure, plant_growth, x_mean, y_mean, y_pred, residuals, beta_0, beta_1, r_squared, save_dir)

print(f"Visualizations saved to: {save_dir}")
print("\nQuestion 10 Solution Summary:")
print(f"1. Light exposure and plant growth regression equation: Growth = {beta_0:.4f} + {beta_1:.4f} × Hours")
print(f"2. Expected growth with 9 hours of light: {nine_hour_growth:.4f} cm")
print(f"3. Additional growth from 2 more hours of light: {two_hour_effect:.4f} cm")
print(f"4. R² value: {r_squared:.4f} ({r_squared*100:.1f}% of growth variation is explained by light exposure)") 