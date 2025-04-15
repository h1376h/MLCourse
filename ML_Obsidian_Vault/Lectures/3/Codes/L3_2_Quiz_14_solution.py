import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_2_Quiz_14")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Define the data from Question 14
patients = [1, 2, 3, 4, 5, 6]
vitamin_c_intake = np.array([50, 100, 150, 200, 250, 300])  # in mg
cold_duration = np.array([7, 6, 5, 4, 3, 3])  # in days

print("Question 14: Vitamin C and Cold Duration")
print("=" * 50)
print("\nData:")
print("Patient | Vitamin C Intake (x) in mg | Cold Duration (y) in days")
print("-" * 65)
for i in range(len(patients)):
    print(f"{patients[i]:^7} | {vitamin_c_intake[i]:^24} | {cold_duration[i]:^25}")
print("=" * 50)

# Step 1: Calculate the least squares estimates for slope and intercept
def calculate_least_squares(x, y):
    """Calculate the least squares estimates for slope and intercept."""
    n = len(x)
    
    # Calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    print(f"\nStep 1: Calculate least squares estimates for slope and intercept")
    print(f"1.1: Calculate means")
    print(f"Mean of vitamin C intake (x̄): {x_mean:.2f} mg")
    print(f"Mean of cold duration (ȳ): {y_mean:.2f} days")
    
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
    
    print(f"\n1.2: Calculate slope (β₁)")
    print(f"Numerator: Σ(x_i - x̄)(y_i - ȳ) = {numerator}")
    print(f"Denominator: Σ(x_i - x̄)² = {denominator}")
    print(f"β₁ = Numerator / Denominator = {beta_1}")
    
    print(f"\n1.3: Calculate intercept (β₀)")
    print(f"β₀ = ȳ - β₁·x̄ = {y_mean} - ({beta_1} × {x_mean}) = {beta_0}")
    
    return beta_0, beta_1

# Execute Step 1
beta_0, beta_1 = calculate_least_squares(vitamin_c_intake, cold_duration)

# Step 2: Write the equation of the linear regression line
print("\nStep 2: Write the equation of the linear regression line")
print(f"Cold Duration = {beta_0:.4f} + ({beta_1:.6f} × Vitamin C Intake)")
print(f"Cold Duration = {beta_0:.4f} - {abs(beta_1):.6f} × Vitamin C Intake")
print("=" * 50)

# Step 3: Define the "effectiveness" of vitamin C
def calculate_effectiveness(beta_1):
    """Calculate the effectiveness of vitamin C per 100mg."""
    effectiveness_per_100mg = beta_1 * 100
    print("\nStep 3: Calculate the effectiveness of vitamin C")
    print(f"Effectiveness per 100mg = β₁ × 100 = {beta_1} × 100 = {effectiveness_per_100mg}")
    print(f"Since β₁ is negative ({beta_1}), this means that for every additional 100mg of vitamin C,")
    print(f"the cold duration decreases by {abs(effectiveness_per_100mg):.2f} days.")
    return effectiveness_per_100mg

# Execute Step 3
effectiveness = calculate_effectiveness(beta_1)
print("=" * 50)

# Step 4: Predict cold duration for 175mg vitamin C intake
def predict_duration(beta_0, beta_1, vitamin_c):
    """Predict cold duration for a given vitamin C intake."""
    predicted_duration = beta_0 + beta_1 * vitamin_c
    print("\nStep 4: Predict cold duration for 175mg vitamin C intake")
    print(f"Predicted duration = β₀ + β₁ × Vitamin C")
    print(f"Predicted duration = {beta_0:.4f} + ({beta_1} × 175)")
    print(f"Predicted duration = {predicted_duration:.2f} days")
    return predicted_duration

# Execute Step 4
new_vitamin_c = 175
predicted_duration = predict_duration(beta_0, beta_1, new_vitamin_c)
print("=" * 50)

# Calculate additional statistics for visualizations and further analysis
def calculate_statistics(x, y, beta_0, beta_1):
    """Calculate additional statistics for the regression model."""
    # Calculate predicted values
    y_pred = beta_0 + beta_1 * x
    
    # Calculate residuals
    residuals = y - y_pred
    
    # Calculate residual sum of squares (RSS)
    rss = np.sum(residuals ** 2)
    
    # Calculate total sum of squares (TSS)
    y_mean = np.mean(y)
    tss = np.sum((y - y_mean) ** 2)
    
    # Calculate R-squared
    r_squared = 1 - (rss / tss)
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(x, y)[0, 1]
    
    print("\nAdditional Statistics:")
    print(f"R-squared (R²): {r_squared:.4f}")
    print(f"Correlation Coefficient (r): {correlation:.4f}")
    print(f"Residual Sum of Squares (RSS): {rss:.4f}")
    print(f"Residuals: {[round(res, 4) for res in residuals]}")
    
    return y_pred, residuals, r_squared, correlation

# Execute additional statistics
y_pred, residuals, r_squared, correlation = calculate_statistics(vitamin_c_intake, cold_duration, beta_0, beta_1)

# Create visualizations
def create_visualizations(x, y, beta_0, beta_1, y_pred, residuals, r_squared, new_x, new_y, save_dir=None):
    """Create visualizations to help understand the regression analysis."""
    saved_files = []
    
    # Figure 1: Scatter plot with regression line
    plt.figure(figsize=(10, 8))
    
    # Plot the data points
    plt.scatter(x, y, color='blue', s=100, label='Observed Data')
    
    # Label each point with the patient number
    for i, patient in enumerate(patients):
        plt.annotate(str(patient), (x[i], y[i]), xytext=(5, 5), textcoords='offset points')
    
    # Generate points for the regression line
    x_line = np.linspace(0, 350, 100)
    y_line = beta_0 + beta_1 * x_line
    
    # Plot regression line
    plt.plot(x_line, y_line, color='red', linestyle='-', linewidth=2, 
             label=f'Regression Line: y = {beta_0:.2f} + ({beta_1:.6f}) × x')
    
    # Plot the prediction for new data point
    plt.scatter(new_x, new_y, color='green', s=150, zorder=5, 
               label=f'Prediction: {new_y:.2f} days for {new_x}mg')
    plt.plot([new_x, new_x], [0, new_y], 'g--', alpha=0.5)
    
    # Customize the plot
    plt.xlabel('Vitamin C Intake (mg)', fontsize=12)
    plt.ylabel('Cold Duration (days)', fontsize=12)
    plt.title('Linear Regression: Cold Duration vs. Vitamin C Intake', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=10)
    
    # Add R² to the plot
    plt.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=plt.gca().transAxes, 
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
    plt.scatter(x, residuals, color='purple', s=100, label='Residuals')
    plt.axhline(y=0, color='red', linestyle='-', alpha=0.7)
    
    # Add vertical lines to show the magnitude of each residual
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [0, residuals[i]], 'purple', linestyle='--', alpha=0.5)
    
    # Customize the plot
    plt.xlabel('Vitamin C Intake (mg)', fontsize=12)
    plt.ylabel('Residual (days)', fontsize=12)
    plt.title('Residuals Plot', fontsize=14)
    plt.grid(True)
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
    
    # Label each point with the patient number
    for i, patient in enumerate(patients):
        plt.annotate(str(patient), (y[i], y_pred[i]), xytext=(5, 5), textcoords='offset points')
    
    # Add a diagonal line (perfect predictions)
    min_val = min(min(y), min(y_pred))
    max_val = max(max(y), max(y_pred))
    plt.plot([min_val - 0.5, max_val + 0.5], 
             [min_val - 0.5, max_val + 0.5], 
             'r--', alpha=0.7)
    
    # Customize the plot
    plt.xlabel('Actual Cold Duration (days)', fontsize=12)
    plt.ylabel('Predicted Cold Duration (days)', fontsize=12)
    plt.title('Actual vs. Predicted Values', fontsize=14)
    plt.grid(True)
    plt.axis('equal')  # Equal scaling
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "fig3_actual_vs_predicted.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Figure 4: Effectiveness visualization
    plt.figure(figsize=(12, 8))
    
    # Create a gradient of vitamin C intakes
    vitamin_c_levels = np.arange(0, 501, 100)
    durations = beta_0 + beta_1 * vitamin_c_levels
    
    # Bar chart
    bars = plt.bar(vitamin_c_levels, durations, width=50, alpha=0.7, color='skyblue')
    
    # Annotate with duration values
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{durations[i]:.1f} days', ha='center', va='bottom')
    
    # Add arrows to show the effectiveness (reduction of 100mg)
    for i in range(len(vitamin_c_levels)-1):
        x1 = vitamin_c_levels[i] + 25
        x2 = vitamin_c_levels[i+1] + 25
        y1 = durations[i] - 0.2
        y2 = durations[i+1] - 0.2
        plt.annotate(
            f'{abs(effectiveness):.2f} days reduction',
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle='->'),
            ha='center', va='center'
        )
    
    # Customize the plot
    plt.xlabel('Vitamin C Intake (mg)', fontsize=12)
    plt.ylabel('Predicted Cold Duration (days)', fontsize=12)
    plt.title('Effectiveness of Vitamin C in Reducing Cold Duration', fontsize=14)
    plt.grid(True, axis='y', alpha=0.3)
    plt.ylim(0, max(durations) + 1)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "fig4_effectiveness_visualization.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Figure 5: Prediction confidence
    plt.figure(figsize=(10, 8))
    
    # Create a more detailed range for plotting
    x_detailed = np.linspace(0, 350, 100)
    y_detailed = beta_0 + beta_1 * x_detailed
    
    # Plot the regression line
    plt.plot(x_detailed, y_detailed, 'r-', linewidth=2, label='Regression Line')
    
    # Plot the data points
    plt.scatter(x, y, color='blue', s=100, label='Observed Data')
    
    # Plot the prediction for new data point
    plt.scatter(new_x, new_y, color='green', s=150, zorder=5, 
               label=f'Prediction: {new_y:.2f} days for {new_x}mg')
    
    # Add confidence bands (simple approximation, not exact)
    # This is a simplified visualization, not a rigorous statistical calculation
    plt.fill_between(x_detailed, 
                   y_detailed - 0.5, 
                   y_detailed + 0.5, 
                   color='red', alpha=0.1, 
                   label='Approximate Confidence Band')
    
    # Add an annotation pointing to the prediction
    plt.annotate(f"Predicted duration: {new_y:.2f} days", 
                xy=(new_x, new_y), 
                xytext=(new_x+30, new_y+1),
                arrowprops=dict(arrowstyle='->'))
    
    # Customize the plot
    plt.xlabel('Vitamin C Intake (mg)', fontsize=12)
    plt.ylabel('Cold Duration (days)', fontsize=12)
    plt.title('Prediction of Cold Duration for 175mg Vitamin C', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "fig5_prediction_visualization.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    return saved_files

# Execute visualizations
saved_files = create_visualizations(
    vitamin_c_intake, 
    cold_duration, 
    beta_0, 
    beta_1, 
    y_pred, 
    residuals, 
    r_squared, 
    new_vitamin_c, 
    predicted_duration, 
    save_dir
)

# Final summary
print("\nFinal Results:")
print(f"1. Linear Regression Equation: Cold Duration = {beta_0:.4f} - {abs(beta_1):.6f} × Vitamin C Intake")
print(f"2. Effectiveness of Vitamin C: {abs(effectiveness):.2f} days reduction per additional 100mg")
print(f"3. Predicted Cold Duration for 175mg: {predicted_duration:.2f} days")
print(f"\nVisualizations saved to: {save_dir}")

# Clinical Interpretation
print("\nClinical Interpretation:")
print("1. The negative slope coefficient suggests that higher vitamin C intake is associated with shorter cold duration.")
print(f"2. The R² value of {r_squared:.4f} indicates that approximately {r_squared*100:.1f}% of the variation")
print("   in cold duration can be explained by vitamin C intake.")
print(f"3. For every 100mg increase in daily vitamin C, we expect approximately {abs(effectiveness):.2f} days")
print("   reduction in cold duration.")
print("4. The near-perfect linear relationship may suggest the need for further controlled studies")
print("   to validate these findings in larger and more diverse populations.")
print("5. The model predicts that cold symptoms would persist for very low vitamin C intakes")
print("   and might even suggest negative durations for very high intakes, which is not realistic.")
print("   This highlights the limitations of extending the linear model beyond the observed data range.") 