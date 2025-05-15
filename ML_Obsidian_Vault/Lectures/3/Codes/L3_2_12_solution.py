import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_2_Quiz_12")
os.makedirs(save_dir, exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = [10, 6]

print("Question 12: Simple Linear Regression Calculation\n")

# Define the problem
print("Problem Statement:")
print("Given the sample data points (1, 2), (2, 4), and (3, 6):")
print()

# Define the data
x = np.array([1, 2, 3])
y = np.array([2, 4, 6])

print("Data Points:")
for i in range(len(x)):
    print(f"({x[i]}, {y[i]})")
print()

# Task 1: Calculate the means
x_mean = np.mean(x)
y_mean = np.mean(y)

print("Task 1: Calculate the means x̄ and ȳ")
print(f"x̄ = ({x[0]} + {x[1]} + {x[2]}) / 3 = {x_mean}")
print(f"ȳ = ({y[0]} + {y[1]} + {y[2]}) / 3 = {y_mean}")
print()

# Task 2: Calculate the slope
print("Task 2: Find the slope of the simple linear regression model by hand")
print("Using the formula: β₁ = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²")
print()

# Calculate the numerator (covariance)
numerator = 0
for i in range(len(x)):
    term = (x[i] - x_mean) * (y[i] - y_mean)
    print(f"({x[i]} - {x_mean})({y[i]} - {y_mean}) = ({x[i] - x_mean:.1f})({y[i] - y_mean:.1f}) = {term:.2f}")
    numerator += term

print(f"Numerator: Σ(xᵢ - x̄)(yᵢ - ȳ) = {numerator:.2f}")
print()

# Calculate the denominator (sum of squared deviations)
denominator = 0
for i in range(len(x)):
    term = (x[i] - x_mean) ** 2
    print(f"({x[i]} - {x_mean})² = ({x[i] - x_mean:.1f})² = {term:.2f}")
    denominator += term

print(f"Denominator: Σ(xᵢ - x̄)² = {denominator:.2f}")
print()

# Calculate the slope
beta_1 = numerator / denominator
print(f"β₁ = {numerator:.2f} / {denominator:.2f} = {beta_1:.2f}")
print()

# Task 3: Calculate the intercept
print("Task 3: Find the intercept of the model")
print("Using the formula: β₀ = ȳ - β₁x̄")
beta_0 = y_mean - beta_1 * x_mean
print(f"β₀ = {y_mean} - {beta_1:.2f} × {x_mean} = {beta_0:.2f}")
print()

# Task 4: Write the regression equation
print("Task 4: Write down the resulting equation for predicting y from x")
print(f"ŷ = {beta_0:.2f} + {beta_1:.2f}x")
print()

# Create visualizations
def create_visualizations(x, y, beta_0, beta_1, save_dir=None):
    """Create visualizations to illustrate the simple linear regression."""
    saved_files = []
    
    # Plot 1: Data points and regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', s=100, label='Data Points')
    
    # Generate points for the regression line
    x_line = np.linspace(0, 4, 100)
    y_line = beta_0 + beta_1 * x_line
    
    plt.plot(x_line, y_line, color='red', linestyle='-', linewidth=2, 
             label=f'Regression Line: y = {beta_0:.2f} + {beta_1:.2f}x')
    
    # Add annotations for each point
    for i in range(len(x)):
        plt.annotate(f"({x[i]}, {y[i]})", 
                     (x[i], y[i]), 
                     xytext=(x[i]+0.1, y[i]+0.2),
                     fontsize=12)
    
    # Add point for the mean
    plt.scatter(x_mean, y_mean, color='green', s=150, marker='x', 
                label=f'Mean Point (x̄, ȳ) = ({x_mean}, {y_mean})')
    
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.title('Simple Linear Regression', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "regression_line.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 2: Illustrating the least squares calculation
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', s=100, label='Data Points')
    plt.plot(x_line, y_line, color='red', linestyle='-', linewidth=2, 
             label=f'Regression Line: y = {beta_0:.2f} + {beta_1:.2f}x')
    
    # Add lines to show the mean point
    plt.axhline(y=y_mean, color='green', linestyle='--', alpha=0.5, label=f'ȳ = {y_mean}')
    plt.axvline(x=x_mean, color='green', linestyle='--', alpha=0.5, label=f'x̄ = {x_mean}')
    
    # Highlight the mean point
    plt.scatter(x_mean, y_mean, color='green', s=150, marker='x', 
                label=f'Mean Point (x̄, ȳ) = ({x_mean}, {y_mean})')
    
    # Show deviations from mean for each point (for numerator calculation)
    for i in range(len(x)):
        # Horizontal line from (x_i, y_mean) to (x_i, y_i)
        plt.plot([x[i], x[i]], [y_mean, y[i]], 'purple', linestyle='--', alpha=0.5)
        plt.annotate(f"(y_i - ȳ) = {y[i] - y_mean:.1f}", 
                     (x[i], (y[i] + y_mean)/2), 
                     xytext=(x[i]+0.1, (y[i] + y_mean)/2),
                     fontsize=10, color='purple')
        
        # Vertical line from (x_mean, y_mean) to (x_i, y_mean)
        plt.plot([x_mean, x[i]], [y_mean, y_mean], 'orange', linestyle='--', alpha=0.5)
        plt.annotate(f"(x_i - x̄) = {x[i] - x_mean:.1f}", 
                     ((x[i] + x_mean)/2, y_mean), 
                     xytext=((x[i] + x_mean)/2, y_mean-0.5),
                     fontsize=10, color='orange')
    
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.title('Understanding the Slope Calculation', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "slope_calculation.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 3: Verifying the regression line passes through (x̄, ȳ)
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', s=100, label='Data Points')
    plt.plot(x_line, y_line, color='red', linestyle='-', linewidth=2, 
             label=f'Regression Line: y = {beta_0:.2f} + {beta_1:.2f}x')
    
    # Highlight the mean point
    plt.scatter(x_mean, y_mean, color='green', s=150, marker='x', 
                label=f'Mean Point (x̄, ȳ) = ({x_mean}, {y_mean})')
    
    # Show the verification calculation
    verification_text = [
        f"Verification that regression line passes through (x̄, ȳ):",
        f"ŷ = β₀ + β₁x̄",
        f"ŷ = {beta_0:.2f} + {beta_1:.2f} × {x_mean}",
        f"ŷ = {beta_0:.2f} + {beta_1*x_mean:.2f}",
        f"ŷ = {beta_0 + beta_1*x_mean:.2f} = ȳ"
    ]
    
    plt.annotate('\n'.join(verification_text), 
                 xy=(0.05, 0.05), 
                 xycoords='figure fraction',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8),
                 fontsize=12)
    
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.title('Regression Line Passes Through the Mean Point', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "verification.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 4: Prediction demonstration
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', s=100, label='Data Points')
    
    # Extend the line for better visualization
    x_line = np.linspace(0, 5, 100)
    y_line = beta_0 + beta_1 * x_line
    
    plt.plot(x_line, y_line, color='red', linestyle='-', linewidth=2, 
             label=f'Regression Line: y = {beta_0:.2f} + {beta_1:.2f}x')
    
    # Show prediction for x = 4
    x_pred = 4
    y_pred = beta_0 + beta_1 * x_pred
    
    plt.scatter(x_pred, y_pred, color='purple', s=100, 
                label=f'Prediction: x={x_pred}, ŷ={y_pred:.2f}')
    
    plt.plot([x_pred, x_pred], [0, y_pred], 'k--', alpha=0.5)
    plt.plot([0, x_pred], [y_pred, y_pred], 'k--', alpha=0.5)
    
    prediction_text = [
        f"Prediction for x = 4:",
        f"ŷ = {beta_0:.2f} + {beta_1:.2f} × 4",
        f"ŷ = {beta_0:.2f} + {beta_1*4:.2f}",
        f"ŷ = {beta_0 + beta_1*4:.2f}"
    ]
    
    plt.annotate('\n'.join(prediction_text), 
                 xy=(0.05, 0.05), 
                 xycoords='figure fraction',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8),
                 fontsize=12)
    
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.title('Using the Regression Model for Prediction', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "prediction.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    return saved_files

# Execute visualizations
saved_files = create_visualizations(x, y, beta_0, beta_1, save_dir)

print(f"Visualizations saved to: {save_dir}")
print("\nQuestion 12 Solution Summary:")
print(f"1. Mean values: x̄ = {x_mean}, ȳ = {y_mean}")
print(f"2. Slope (β₁) = {beta_1:.2f}")
print(f"3. Intercept (β₀) = {beta_0:.2f}")
print(f"4. Regression equation: ŷ = {beta_0:.2f} + {beta_1:.2f}x")
print("\nThe solution demonstrates that:")
print("- The slope of 2.0 means that for each unit increase in x, y increases by 2 units")
print("- The intercept of 0.0 means that when x = 0, the predicted value of y is 0")
print("- This equation perfectly fits the given data points")
print("- The regression line passes through the mean point (x̄, ȳ) = (2, 4)") 