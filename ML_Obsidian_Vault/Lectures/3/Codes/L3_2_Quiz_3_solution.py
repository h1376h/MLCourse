import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_2_Quiz_3")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Define the model equation
beta_0 = 3  # Intercept
beta_1 = 2  # Slope
model_equation = f"ŷ = {beta_0} + {beta_1}x"

print("Question 3: Basic Prediction and Residual Calculation")
print("====================================================")
print(f"Given a simple linear regression model with the equation: {model_equation}")
print("\nTask:")
print("1. Calculate the predicted value ŷ when x = 4")
print("2. If the actual observed value when x = 4 is y = 12, what is the residual?")
print("====================================================\n")

# Step 1: Calculate the predicted value
x_value = 4
y_pred = beta_0 + beta_1 * x_value

print("Step 1: Calculate the predicted value when x = 4")
print(f"ŷ = {beta_0} + {beta_1} × {x_value}")
print(f"ŷ = {beta_0} + {beta_1 * x_value}")
print(f"ŷ = {y_pred}")
print(f"Therefore, the predicted value when x = 4 is {y_pred}.\n")

# Step 2: Calculate the residual
y_actual = 12
residual = y_actual - y_pred

print("Step 2: Calculate the residual")
print(f"The actual observed value when x = 4 is y = {y_actual}")
print(f"Residual = Actual value - Predicted value")
print(f"Residual = {y_actual} - {y_pred}")
print(f"Residual = {residual}")
print(f"Therefore, the residual is {residual}.\n")

# Create visualizations
def create_visualizations(x_value, y_pred, y_actual, beta_0, beta_1, save_dir=None):
    """Create visualizations to help understand the prediction and residual."""
    saved_files = []
    
    # Generate points for the regression line
    x_line = np.linspace(0, 6, 100)
    y_line = beta_0 + beta_1 * x_line
    
    # Plot 1: Model and Prediction Visualization
    plt.figure(figsize=(10, 6))
    
    # Plot the regression line
    plt.plot(x_line, y_line, color='blue', linewidth=2, label=f'Regression Line: ŷ = {beta_0} + {beta_1}x')
    
    # Plot the actual and predicted points
    plt.scatter(x_value, y_pred, color='red', s=100, label=f'Predicted: ({x_value}, {y_pred})')
    plt.scatter(x_value, y_actual, color='green', s=100, label=f'Actual: ({x_value}, {y_actual})')
    
    # Draw a vertical line from x-axis to the predicted point
    plt.vlines(x=x_value, ymin=0, ymax=y_pred, colors='red', linestyles='--', alpha=0.7)
    
    # Draw a vertical line from the predicted to the actual point (representing the residual)
    plt.vlines(x=x_value, ymin=y_pred, ymax=y_actual, colors='purple', linestyles='--', alpha=0.7, linewidth=2)
    
    # Add a text annotation for the residual
    plt.annotate(f'Residual = {residual}', 
                xy=(x_value, (y_pred + y_actual)/2),
                xytext=(x_value + 0.5, (y_pred + y_actual)/2),
                arrowprops=dict(arrowstyle='->'),
                fontsize=12)
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Linear Regression Model: Prediction and Residual', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "linear_regression_prediction_residual.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 2: Residual Visualization
    plt.figure(figsize=(8, 4))
    
    plt.scatter(x_value, residual, color='purple', s=100)
    plt.axhline(y=0, color='red', linestyle='-', alpha=0.7)
    plt.vlines(x=x_value, ymin=0, ymax=residual, colors='purple', linestyles='--', alpha=0.7)
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('Residual (y - ŷ)', fontsize=12)
    plt.title('Residual Plot', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "residual_plot.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 3: General explanatory diagram of a residual
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate example data with multiple points
    x_example = np.array([1, 2, 3, 4, 5])
    y_example = np.array([5, 7, 7, 12, 15])  # Actual values
    y_example_pred = beta_0 + beta_1 * x_example  # Predicted values
    residuals_example = y_example - y_example_pred
    
    # Plot the regression line
    x_line_ex = np.linspace(0, 6, 100)
    y_line_ex = beta_0 + beta_1 * x_line_ex
    plt.plot(x_line_ex, y_line_ex, color='blue', linewidth=2, label=f'Regression Line: ŷ = {beta_0} + {beta_1}x')
    
    # Plot the data points and their predictions
    plt.scatter(x_example, y_example, color='green', s=80, label='Actual Values')
    plt.scatter(x_example, y_example_pred, color='red', s=80, label='Predicted Values')
    
    # Draw vertical lines representing the residuals
    for i in range(len(x_example)):
        plt.vlines(x=x_example[i], ymin=y_example_pred[i], ymax=y_example[i], 
                colors='purple', linestyles='--', alpha=0.7)
        
        # Annotate the residual if it's significant enough
        if abs(residuals_example[i]) > 0.5:
            plt.annotate(f'e={residuals_example[i]:.1f}', 
                        xy=(x_example[i], (y_example_pred[i] + y_example[i])/2),
                        xytext=(x_example[i] + 0.15, (y_example_pred[i] + y_example[i])/2),
                        fontsize=10)
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Understanding Residuals in Linear Regression', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "residuals_explanation.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    return saved_files

# Execute visualizations
saved_files = create_visualizations(x_value, y_pred, y_actual, beta_0, beta_1, save_dir)

print(f"Visualizations saved to: {', '.join(saved_files)}")
print("\nSummary:")
print(f"1. The predicted value ŷ when x = 4 is {y_pred}.")
print(f"2. Given that the actual observed value is y = {y_actual}, the residual is {residual}.")
print("\nConclusion:")
print("The residual represents how much the actual observation differs from what our model predicts.")
print("A positive residual (as in this case) means the actual value is higher than predicted.")
print("A negative residual would mean the actual value is lower than predicted.")
print("In a good model, residuals should be random and have a mean close to zero.") 