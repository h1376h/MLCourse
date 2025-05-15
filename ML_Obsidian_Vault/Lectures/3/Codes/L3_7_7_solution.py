import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_2_Quiz_7")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Define the data from Question 7
temperature = np.array([20, 25, 30, 35, 40])
water_consumed = np.array([35, 45, 60, 80, 95])

# Step 1: Create a scatter plot to determine if a linear relationship is appropriate
def create_scatter_plot(x, y, save_dir=None):
    """Create a scatter plot of the data to visualize the relationship."""
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', s=100)
    
    plt.xlabel('Temperature (°C)', fontsize=12)
    plt.ylabel('Water Consumed (liters)', fontsize=12)
    plt.title('Relationship Between Temperature and Water Consumption', fontsize=14)
    plt.grid(True)
    
    # Add a note about linearity
    plt.annotate('The points appear to follow a linear pattern,\nsuggesting a linear model is appropriate.',
                xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=12, ha='left', va='top',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot1_scatter_plot.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved to: {file_path}")
    
    # Let's check visually if the relationship looks linear
    print("Step 1: Examine if a linear relationship is appropriate")
    print("Based on the scatter plot, the relationship between temperature and water")
    print("consumption appears to be approximately linear. The points generally follow")
    print("a straight-line pattern, suggesting that a linear regression model would be appropriate.")
    print()
    
    plt.close()

create_scatter_plot(temperature, water_consumed, save_dir)

# Step 2: Calculate the means of x and y
def calculate_means(x, y):
    """Calculate the means of x and y."""
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    print(f"Step 2: Calculate the means of temperature and water consumption")
    print(f"Mean temperature (x̄): {x_mean} °C")
    print(f"Mean water consumption (ȳ): {y_mean} liters")
    print()
    
    return x_mean, y_mean

x_mean, y_mean = calculate_means(temperature, water_consumed)

# Step 3: Find the linear equation that best fits the data using least squares
def calculate_regression_coefficients(x, y, x_mean, y_mean):
    """Calculate the slope and intercept using the least squares method."""
    # Calculate numerator (covariance)
    numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
    
    # Calculate denominator (variance of x)
    denominator = sum((x[i] - x_mean) ** 2 for i in range(len(x)))
    
    # Calculate slope
    beta_1 = numerator / denominator
    
    # Calculate intercept
    beta_0 = y_mean - beta_1 * x_mean
    
    print(f"Step 3: Calculate the regression coefficients using least squares")
    print(f"First, we calculate the slope (β₁):")
    print(f"β₁ = Σ[(x_i - x̄)(y_i - ȳ)] / Σ[(x_i - x̄)²]")
    
    # Print the calculation details
    print("Calculation details:")
    print(f"Numerator calculation:")
    numerator_terms = [(x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x))]
    for i in range(len(x)):
        print(f"  ({x[i]} - {x_mean}) * ({y[i]} - {y_mean}) = {numerator_terms[i]:.2f}")
    print(f"  Sum = {numerator:.2f}")
    
    print(f"Denominator calculation:")
    denominator_terms = [(x[i] - x_mean) ** 2 for i in range(len(x))]
    for i in range(len(x)):
        print(f"  ({x[i]} - {x_mean})² = {denominator_terms[i]:.2f}")
    print(f"  Sum = {denominator:.2f}")
    
    print(f"β₁ = {numerator:.2f} / {denominator:.2f} = {beta_1:.2f}")
    
    print(f"Next, we calculate the intercept (β₀):")
    print(f"β₀ = ȳ - β₁ * x̄")
    print(f"β₀ = {y_mean:.2f} - {beta_1:.2f} * {x_mean:.2f} = {beta_0:.2f}")
    
    print(f"Therefore, the regression equation is:")
    print(f"Water Consumed = {beta_0:.2f} + {beta_1:.2f} × Temperature")
    print()
    
    return beta_0, beta_1

beta_0, beta_1 = calculate_regression_coefficients(temperature, water_consumed, x_mean, y_mean)

# Step 4: Predict water consumption for a temperature of 28°C
def predict_water_consumption(beta_0, beta_1, temp):
    """Predict water consumption for a given temperature."""
    prediction = beta_0 + beta_1 * temp
    
    print(f"Step 4: Predict water consumption for a temperature of {temp}°C")
    print(f"Using our regression equation: Water = {beta_0:.2f} + {beta_1:.2f} × Temperature")
    print(f"Water = {beta_0:.2f} + {beta_1:.2f} × {temp} = {prediction:.2f} liters")
    print(f"The restaurant should prepare approximately {prediction:.0f} liters of water for a {temp}°C day.")
    print()
    
    return prediction

prediction_28 = predict_water_consumption(beta_0, beta_1, 28)

# Step 5: Calculate the residual for the day when temperature was 30°C
def calculate_residual(x_value, y_value, beta_0, beta_1):
    """Calculate the residual for a specific data point."""
    predicted = beta_0 + beta_1 * x_value
    residual = y_value - predicted
    
    print(f"Step 5: Calculate the residual for the day when temperature was {x_value}°C")
    print(f"Actual water consumption: {y_value} liters")
    print(f"Predicted water consumption: {beta_0:.2f} + {beta_1:.2f} × {x_value} = {predicted:.2f} liters")
    print(f"Residual = Actual - Predicted = {y_value} - {predicted:.2f} = {residual:.2f} liters")
    print(f"This means that on the {x_value}°C day, the actual water consumption was")
    print(f"{abs(residual):.2f} liters {'more' if residual > 0 else 'less'} than what our model predicted.")
    print()
    
    return residual

# Find the index of temperature 30°C
temp_30_idx = np.where(temperature == 30)[0][0]
residual_30 = calculate_residual(temperature[temp_30_idx], water_consumed[temp_30_idx], beta_0, beta_1)

# Create visualizations for better understanding
def create_visualizations(x, y, beta_0, beta_1, prediction_temp, save_dir=None):
    """Create visualizations to explain the regression analysis."""
    saved_files = []
    
    # Plot 2: Scatter plot with regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', s=100, label='Observed Data')
    
    # Generate points for the regression line
    x_line = np.linspace(min(x) - 2, max(x) + 2, 100)
    y_line = beta_0 + beta_1 * x_line
    
    plt.plot(x_line, y_line, color='red', linestyle='-', linewidth=2, 
            label=f'Regression Line\nWater = {beta_0:.2f} + {beta_1:.2f} × Temp')
    
    # Highlight the prediction for 28°C
    predicted_water = beta_0 + beta_1 * prediction_temp
    plt.scatter([prediction_temp], [predicted_water], color='green', s=150, 
               marker='*', label=f'Prediction for {prediction_temp}°C')
    
    plt.plot([prediction_temp, prediction_temp], [0, predicted_water], 'g--', alpha=0.7)
    plt.plot([0, prediction_temp], [predicted_water, predicted_water], 'g--', alpha=0.7)
    
    plt.annotate(f'{predicted_water:.1f} liters', 
               xy=(prediction_temp, predicted_water), 
               xytext=(prediction_temp-5, predicted_water-10),
               fontsize=12, color='green',
               arrowprops=dict(arrowstyle="->", color='green'))
    
    plt.xlabel('Temperature (°C)', fontsize=12)
    plt.ylabel('Water Consumed (liters)', fontsize=12)
    plt.title('Linear Regression: Water Consumption vs. Temperature', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=10, loc='upper left')
    
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot2_regression_line.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 3: Residuals visualization
    plt.figure(figsize=(10, 6))
    
    # Calculate predicted values
    y_pred = beta_0 + beta_1 * x
    
    # Calculate residuals
    residuals = y - y_pred
    
    # Scatter plot with original data and predictions
    plt.scatter(x, y, color='blue', s=100, label='Actual Consumption')
    plt.scatter(x, y_pred, color='red', s=100, label='Predicted Consumption')
    
    # Draw residual lines
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [y_pred[i], y[i]], 'k--', alpha=0.7, linewidth=2)
        # Add residual labels
        if x[i] == 30:  # Highlight the 30°C residual
            plt.annotate(f'Residual at 30°C:\n{residuals[i]:.2f} liters',
                        xy=(x[i], (y_pred[i] + y[i])/2),
                        xytext=(x[i]+2, (y_pred[i] + y[i])/2),
                        fontsize=12,
                        arrowprops=dict(arrowstyle="->"))
        else:
            plt.annotate(f'{residuals[i]:.2f}',
                        xy=(x[i], (y_pred[i] + y[i])/2),
                        xytext=(x[i]+0.5, (y_pred[i] + y[i])/2),
                        fontsize=10)
    
    plt.plot(x_line, y_line, 'r-', alpha=0.7, linewidth=2)
    
    plt.xlabel('Temperature (°C)', fontsize=12)
    plt.ylabel('Water Consumed (liters)', fontsize=12)
    plt.title('Residuals: Differences Between Actual and Predicted Consumption', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=10, loc='upper left')
    
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot3_residuals.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 4: Step-by-step regression illustration
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot 4a: Deviation from means
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(x, y, color='blue', s=80)
    ax1.axhline(y=y_mean, color='green', linestyle='--', label=f'Mean Water: {y_mean:.1f}L')
    ax1.axvline(x=x_mean, color='red', linestyle='--', label=f'Mean Temp: {x_mean:.1f}°C')
    
    # Draw lines from points to means
    for i in range(len(x)):
        ax1.plot([x[i], x[i]], [y_mean, y[i]], 'k:', alpha=0.5)
        ax1.plot([x_mean, x[i]], [y_mean, y_mean], 'k:', alpha=0.5)
    
    ax1.set_xlabel('Temperature (°C)', fontsize=10)
    ax1.set_ylabel('Water Consumed (liters)', fontsize=10)
    ax1.set_title('Step 1: Deviations from Means', fontsize=12)
    ax1.grid(True)
    ax1.legend(fontsize=8)
    
    # Plot 4b: Finding the best-fit line
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(x, y, color='blue', s=80)
    
    # Draw several potential regression lines
    slopes = [beta_1-1, beta_1-0.5, beta_1, beta_1+0.5, beta_1+1]
    colors = ['gray', 'gray', 'red', 'gray', 'gray']
    alphas = [0.4, 0.6, 1.0, 0.6, 0.4]
    linewidths = [1, 1.5, 2, 1.5, 1]
    
    for i, slope in enumerate(slopes):
        intercept = y_mean - slope * x_mean
        ax2.plot(x_line, intercept + slope * x_line, 
                color=colors[i], alpha=alphas[i], linewidth=linewidths[i], 
                label=f'Slope = {slope:.2f}')
    
    ax2.set_xlabel('Temperature (°C)', fontsize=10)
    ax2.set_ylabel('Water Consumed (liters)', fontsize=10)
    ax2.set_title('Step 2: Finding the Best-Fit Line', fontsize=12)
    ax2.grid(True)
    ax2.legend(fontsize=8, loc='upper left')
    
    # Plot 4c: The final regression equation
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(x, y, color='blue', s=80, label='Observed Data')
    ax3.plot(x_line, y_line, color='red', linewidth=2, 
            label=f'Regression Line')
    
    # Add the equation to the plot
    equation = f"Water = {beta_0:.2f} + {beta_1:.2f} × Temp"
    ax3.text(0.05, 0.95, equation, transform=ax3.transAxes, 
            fontsize=12, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax3.set_xlabel('Temperature (°C)', fontsize=10)
    ax3.set_ylabel('Water Consumed (liters)', fontsize=10)
    ax3.set_title('Step 3: Final Regression Equation', fontsize=12)
    ax3.grid(True)
    ax3.legend(fontsize=8, loc='upper left')
    
    # Plot 4d: Making predictions
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(x, y, color='blue', s=80, label='Observed Data')
    ax4.plot(x_line, y_line, color='red', linewidth=2, 
            label=f'Regression Line')
    
    # Highlight the prediction point
    ax4.scatter([prediction_temp], [predicted_water], color='green', s=120, 
               marker='*', label=f'Prediction for {prediction_temp}°C')
    
    # Add visual guides for the prediction
    ax4.plot([prediction_temp, prediction_temp], [0, predicted_water], 'g--', alpha=0.7)
    ax4.plot([0, prediction_temp], [predicted_water, predicted_water], 'g--', alpha=0.7)
    
    ax4.annotate(f'For {prediction_temp}°C:\nPredicted consumption\n= {predicted_water:.1f} liters',
                xy=(prediction_temp, predicted_water),
                xytext=(prediction_temp+2, predicted_water-15),
                fontsize=10,
                arrowprops=dict(arrowstyle="->", color='green'))
    
    ax4.set_xlabel('Temperature (°C)', fontsize=10)
    ax4.set_ylabel('Water Consumed (liters)', fontsize=10)
    ax4.set_title('Step 4: Making Predictions', fontsize=12)
    ax4.grid(True)
    ax4.legend(fontsize=8, loc='upper left')
    
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot4_step_by_step.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    return saved_files

# Create visualizations
saved_files = create_visualizations(temperature, water_consumed, beta_0, beta_1, 28, save_dir)

print("Visualizations saved to:", save_dir)
print("\nQuestion 7 Solution Summary:")
print(f"1. Mean temperature: {x_mean} °C")
print(f"2. Mean water consumption: {y_mean} liters")
print(f"3. Regression equation: Water = {beta_0:.2f} + {beta_1:.2f} × Temperature")
print(f"4. Predicted water consumption for 28°C: {prediction_28:.2f} liters")
print(f"5. Residual for 30°C day: {residual_30:.2f} liters") 