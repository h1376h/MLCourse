import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_2_Quiz_4")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Define the problem parameters from Question 4
x_mean = 5
y_mean = 10
beta_1 = 2

# Step 1: Calculate the intercept term beta_0
def calculate_intercept(y_mean, x_mean, beta_1):
    """Calculate the intercept term using the formula beta_0 = y_mean - beta_1 * x_mean."""
    beta_0 = y_mean - beta_1 * x_mean
    
    print(f"Step 1: Calculate the intercept term (β₀)")
    print(f"Given information:")
    print(f"  - Mean of x (x̄): {x_mean}")
    print(f"  - Mean of y (ȳ): {y_mean}")
    print(f"  - Slope coefficient (β₁): {beta_1}")
    print(f"\nUsing the formula: β₀ = ȳ - β₁·x̄")
    print(f"β₀ = {y_mean} - {beta_1} · {x_mean}")
    print(f"β₀ = {y_mean} - {beta_1 * x_mean}")
    print(f"β₀ = {beta_0}")
    
    return beta_0

# Execute Step 1
beta_0 = calculate_intercept(y_mean, x_mean, beta_1)

# Step 2: Write down the complete regression equation
def write_regression_equation(beta_0, beta_1):
    """Write down the complete regression equation."""
    print(f"\nStep 2: Write down the complete regression equation")
    print(f"ŷ = β₀ + β₁·x")
    print(f"ŷ = {beta_0} + {beta_1}·x")
    
    return f"ŷ = {beta_0} + {beta_1}·x"

# Execute Step 2
regression_equation = write_regression_equation(beta_0, beta_1)

# Create visualizations
def create_visualizations(x_mean, y_mean, beta_0, beta_1, save_dir=None):
    """Create visualizations to help understand the regression line and its relation to the means."""
    saved_files = []
    
    # Generate some sample data around the means for visualization purposes
    # The actual data points are not given, so we'll create some that are consistent
    # with the mean values and a reasonable fit to the regression line
    np.random.seed(42)  # For reproducibility
    x_values = np.linspace(x_mean - 3, x_mean + 3, 10)
    # Generate y values that are scattered around the regression line
    y_pred = beta_0 + beta_1 * x_values
    y_values = y_pred + np.random.normal(0, 1, size=len(x_values))
    
    # Ensure the means of our sample match the given means
    x_values = x_values - np.mean(x_values) + x_mean
    y_values = y_values - np.mean(y_values) + y_mean
    
    # Plot 1: Scatter plot with regression line and means
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values, color='blue', s=80, alpha=0.7, label='Sample Data Points')
    
    # Generate points for the regression line over a wider range
    x_line = np.linspace(min(x_values) - 1, max(x_values) + 1, 100)
    y_line = beta_0 + beta_1 * x_line
    
    plt.plot(x_line, y_line, color='red', linestyle='-', linewidth=2, 
             label=f'Regression Line: ŷ = {beta_0} + {beta_1}·x')
    
    # Plot the mean point
    plt.scatter([x_mean], [y_mean], color='green', s=120, marker='*', 
                label=f'Mean Point (x̄={x_mean}, ȳ={y_mean})')
    
    # Add the lines to show the mean point location
    plt.axvline(x=x_mean, color='green', linestyle='--', alpha=0.5)
    plt.axhline(y=y_mean, color='green', linestyle='--', alpha=0.5)
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Regression Line with Mean Point Highlighted', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "regression_line_with_means.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 2: Visual explanation of the intercept calculation
    plt.figure(figsize=(12, 6))
    
    # Create a 1x2 grid of subplots
    gs = GridSpec(1, 2, width_ratios=[1, 1.5])
    
    # First subplot: Equation
    ax1 = plt.subplot(gs[0])
    ax1.axis('off')  # Turn off axes
    
    # Create text explanation
    text = (
        "Calculating the Intercept\n"
        "------------------------\n\n"
        f"Given:\n"
        f"  • Mean of x (x̄) = {x_mean}\n"
        f"  • Mean of y (ȳ) = {y_mean}\n"
        f"  • Slope (β₁) = {beta_1}\n\n"
        f"Formula:\n"
        f"  β₀ = ȳ - β₁·x̄\n\n"
        f"Substitution:\n"
        f"  β₀ = {y_mean} - {beta_1}·{x_mean}\n"
        f"  β₀ = {y_mean} - {beta_1 * x_mean}\n"
        f"  β₀ = {beta_0}\n\n"
        f"Regression Equation:\n"
        f"  ŷ = {beta_0} + {beta_1}·x"
    )
    
    ax1.text(0.05, 0.95, text, transform=ax1.transAxes, 
             fontsize=12, verticalalignment='top', 
             family='monospace', bbox=dict(facecolor='lightgray', alpha=0.5))
    
    # Second subplot: Geometric interpretation
    ax2 = plt.subplot(gs[1])
    
    ax2.scatter(x_values, y_values, color='blue', s=60, alpha=0.5, label='Sample Data')
    ax2.plot(x_line, y_line, color='red', linewidth=2, label=f'ŷ = {beta_0} + {beta_1}·x')
    
    # Plot the mean point
    ax2.scatter([x_mean], [y_mean], color='green', s=120, marker='*', 
                label=f'Mean Point (x̄={x_mean}, ȳ={y_mean})')
    
    # Show the y-intercept point
    ax2.scatter([0], [beta_0], color='purple', s=100, marker='o', 
                label=f'Y-intercept (0, {beta_0})')
    
    # Add the lines to show the mean point and its relation to the slope
    ax2.axvline(x=x_mean, color='green', linestyle='--', alpha=0.5)
    ax2.axhline(y=y_mean, color='green', linestyle='--', alpha=0.5)
    
    # Add arrow to show the effect of the slope from mean point to intercept
    ax2.annotate('', xy=(0, beta_0), xytext=(x_mean, y_mean),
                 arrowprops=dict(facecolor='orange', shrink=0.05, width=1.5, headwidth=8))
    
    ax2.text(x_mean/2, (y_mean + beta_0)/2, f"-{beta_1}·x̄ = -{beta_1*x_mean}",
             color='orange', fontsize=12, ha='center', va='bottom')
    
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_title('Geometric Interpretation of Intercept Calculation', fontsize=14)
    ax2.grid(True)
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "intercept_calculation_explained.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 3: Predictions for different x values
    plt.figure(figsize=(10, 6))
    
    # Sample x values for predictions
    x_pred = np.array([2, 4, 6, 8])
    y_pred = beta_0 + beta_1 * x_pred
    
    plt.scatter(x_values, y_values, color='blue', s=60, alpha=0.5, label='Sample Data')
    plt.plot(x_line, y_line, color='red', linewidth=2, label=f'ŷ = {beta_0} + {beta_1}·x')
    
    # Plot prediction points
    plt.scatter(x_pred, y_pred, color='orange', s=80, marker='D', label='Predicted Values')
    
    # Add text labels for predictions
    for i, (x, y) in enumerate(zip(x_pred, y_pred)):
        plt.annotate(f'({x}, {y:.1f})', xy=(x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Predictions Using the Regression Equation', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "predictions_demonstration.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    return saved_files

# Execute visualizations
saved_files = create_visualizations(x_mean, y_mean, beta_0, beta_1, save_dir)

print(f"\nVisualizations saved to: {', '.join(saved_files)}")
print("\nQuestion 4 Solution Summary:")
print(f"1. Intercept (β₀): {beta_0}")
print(f"2. Complete regression equation: {regression_equation}")
print("\nNote: The regression line passes through the point (x̄, ȳ) = ({x_mean}, {y_mean}).")
print("This is a fundamental property of least squares regression.")

if __name__ == "__main__":
    print("Executing solution for Question 4: Finding the Intercept from Means")
    # All calculations and visualizations have already been executed above 