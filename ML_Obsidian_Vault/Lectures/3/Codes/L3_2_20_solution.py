import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

# Create directory to save figures if it doesn't exist
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_2_Quiz_20")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Define the data points
x = np.array([1, 2, 3])
y = np.array([3, 5, 8])

print("## Question 20: Calculating Linear Regression Coefficients by Hand")
print("\n### Step 1: Calculate the least squares estimates for β₀ and β₁")
print("\nGiven data points:")
for i in range(len(x)):
    print(f"  (x₍{i+1}₎, y₍{i+1}₎) = ({x[i]}, {y[i]})")

# Calculate the means
x_mean = np.mean(x)
y_mean = np.mean(y)

print(f"\nStep 1.1: Calculate the means")
print(f"  x̄ = ({' + '.join(map(str, x))}) / {len(x)} = {x_mean}")
print(f"  ȳ = ({' + '.join(map(str, y))}) / {len(y)} = {y_mean}")

# Calculate (xi - x_mean) and (yi - y_mean)
x_diff = x - x_mean
y_diff = y - y_mean

print("\nStep 1.2: Calculate (xᵢ - x̄) and (yᵢ - ȳ) for each data point")
for i in range(len(x)):
    print(f"  x₍{i+1}₎ - x̄ = {x[i]} - {x_mean} = {x_diff[i]}")
    print(f"  y₍{i+1}₎ - ȳ = {y[i]} - {y_mean} = {y_diff[i]}")

# Calculate (xi - x_mean)(yi - y_mean) and (xi - x_mean)²
xy_diff_prod = x_diff * y_diff
x_diff_sq = x_diff**2

print("\nStep 1.3: Calculate (xᵢ - x̄)(yᵢ - ȳ) and (xᵢ - x̄)² for each data point")
for i in range(len(x)):
    print(f"  (x₍{i+1}₎ - x̄)(y₍{i+1}₎ - ȳ) = {x_diff[i]} × {y_diff[i]} = {xy_diff_prod[i]}")
    print(f"  (x₍{i+1}₎ - x̄)² = {x_diff[i]}² = {x_diff_sq[i]}")

# Calculate sums
xy_diff_sum = np.sum(xy_diff_prod)
x_diff_sq_sum = np.sum(x_diff_sq)

print("\nStep 1.4: Calculate the sums")
print(f"  Σ(xᵢ - x̄)(yᵢ - ȳ) = {' + '.join(map(str, xy_diff_prod))} = {xy_diff_sum}")
print(f"  Σ(xᵢ - x̄)² = {' + '.join(map(str, x_diff_sq))} = {x_diff_sq_sum}")

# Calculate β₁
beta1 = xy_diff_sum / x_diff_sq_sum

print("\nStep 1.5: Calculate β₁ using the formula: β₁ = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²")
print(f"  β₁ = {xy_diff_sum} / {x_diff_sq_sum} = {beta1}")

# Calculate β₀
beta0 = y_mean - beta1 * x_mean

print("\nStep 1.6: Calculate β₀ using the formula: β₀ = ȳ - β₁ × x̄")
print(f"  β₀ = {y_mean} - {beta1} × {x_mean} = {beta0}")

print("\nThe least squares regression line is:")
print(f"y = {beta0} + {beta1}x")

# Create the regression line
x_line = np.linspace(0.5, 3.5, 100)
y_line = beta0 + beta1 * x_line

# Task 2: Calculate the predicted value for x = 2.5
x_pred = 2.5
y_pred = beta0 + beta1 * x_pred

print("\n### Step 2: Find the predicted value when x = 2.5")
print(f"For x = 2.5, the predicted value is:")
print(f"y = {beta0} + {beta1} × 2.5 = {y_pred}")

# Task 3: Calculate the residuals and RSS
y_fitted = beta0 + beta1 * x
residuals = y - y_fitted
rss = np.sum(residuals**2)

print("\n### Step 3: Calculate the residual sum of squares (RSS)")
print("\nStep 3.1: Calculate fitted values for each x")
for i in range(len(x)):
    print(f"  ŷ₍{i+1}₎ = {beta0} + {beta1} × {x[i]} = {y_fitted[i]}")

print("\nStep 3.2: Calculate residuals (actual - fitted) for each point")
for i in range(len(x)):
    print(f"  e₍{i+1}₎ = y₍{i+1}₎ - ŷ₍{i+1}₎ = {y[i]} - {y_fitted[i]} = {residuals[i]}")

print("\nStep 3.3: Square each residual")
for i in range(len(x)):
    print(f"  e₍{i+1}₎² = {residuals[i]}² = {residuals[i]**2}")

print("\nStep 3.4: Calculate the residual sum of squares (RSS)")
print(f"  RSS = Σe₍ᵢ₎² = {' + '.join(map(str, residuals**2))} = {rss}")

# Create visualizations
def create_visualizations(x, y, beta0, beta1, x_line, y_line, x_pred, y_pred, residuals, save_dir=None):
    saved_files = []
    
    # Figure 1: Data points with regression line
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.scatter(x, y, color='blue', s=100, label='Data points')
    
    # Plot regression line
    plt.plot(x_line, y_line, color='red', linewidth=2, label=f'y = {beta0:.2f} + {beta1:.2f}x')
    
    # Plot predicted point
    plt.scatter([x_pred], [y_pred], color='green', s=100, label=f'Predicted at x=2.5')
    
    # Add vertical line from x-axis to predicted point
    plt.vlines(x=x_pred, ymin=0, ymax=y_pred, linestyle='--', color='green', alpha=0.7)
    
    # Add horizontal line from y-axis to predicted point
    plt.axhline(y=y_pred, xmin=0, xmax=x_pred/4, linestyle='--', color='green', alpha=0.7)
    
    # Annotate the data points
    for i in range(len(x)):
        plt.annotate(f"({x[i]}, {y[i]})", 
                    xy=(x[i], y[i]), 
                    xytext=(x[i]-0.15, y[i]+0.3),
                    fontsize=12)
    
    # Annotate the predicted point
    plt.annotate(f"({x_pred}, {y_pred:.2f})", 
                xy=(x_pred, y_pred), 
                xytext=(x_pred+0.1, y_pred-0.5),
                fontsize=12,
                arrowprops=dict(arrowstyle="->", color='green'))
    
    plt.title('Linear Regression: Data Points and Regression Line', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.xlim(0, 4)
    plt.ylim(0, 10)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot1_regression_line.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Figure 2: Visualizing residuals
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.scatter(x, y, color='blue', s=100, label='Data points')
    
    # Plot regression line
    plt.plot(x_line, y_line, color='red', linewidth=2, label=f'y = {beta0:.2f} + {beta1:.2f}x')
    
    # Plot fitted points
    y_fitted = beta0 + beta1 * x
    plt.scatter(x, y_fitted, color='green', s=80, marker='x', label='Fitted values')
    
    # Plot residuals as vertical lines
    for i in range(len(x)):
        plt.vlines(x=x[i], ymin=y_fitted[i], ymax=y[i], linestyle='--', color='purple', alpha=0.7)
    
    # Annotate the residuals
    for i in range(len(x)):
        mid_y = (y[i] + y_fitted[i]) / 2
        plt.annotate(f"e₍{i+1}₎ = {residuals[i]:.2f}", 
                    xy=(x[i], mid_y), 
                    xytext=(x[i]+0.1, mid_y),
                    fontsize=10,
                    color='purple')
    
    plt.title('Visualization of Residuals', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.xlim(0, 4)
    plt.ylim(0, 10)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot2_residuals.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Figure 3: Step-by-step calculation visualization
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot 1: Data points with means
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(x, y, color='blue', s=80, label='Data points')
    ax1.axhline(y=y_mean, color='green', linestyle='--', alpha=0.7, label=f'ȳ = {y_mean}')
    ax1.axvline(x=x_mean, color='red', linestyle='--', alpha=0.7, label=f'x̄ = {x_mean}')
    ax1.scatter([x_mean], [y_mean], color='purple', s=100, label='(x̄, ȳ)')
    
    ax1.set_title('Data Points with Mean Lines', fontsize=12)
    ax1.set_xlabel('x', fontsize=10)
    ax1.set_ylabel('y', fontsize=10)
    ax1.grid(True)
    ax1.legend(fontsize=8)
    
    # Plot 2: Formula and calculations
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    
    formula_text = (
        "Least Squares Formulas:\n\n"
        f"β₁ = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²\n"
        f"   = {xy_diff_sum} / {x_diff_sq_sum} = {beta1}\n\n"
        f"β₀ = ȳ - β₁ × x̄\n"
        f"   = {y_mean} - {beta1} × {x_mean} = {beta0}\n\n"
        f"Regression Equation:\n"
        f"y = {beta0} + {beta1}x"
    )
    
    ax2.text(0.05, 0.5, formula_text, fontsize=12, verticalalignment='center')
    
    # Plot 3: Residuals calculation
    ax3 = fig.add_subplot(gs[1, 0])
    
    residual_text = "Residuals Calculation:\n\n"
    for i in range(len(x)):
        residual_text += f"e₍{i+1}₎ = y₍{i+1}₎ - ŷ₍{i+1}₎ = {y[i]} - {y_fitted[i]:.2f} = {residuals[i]:.2f}\n"
    
    residual_text += f"\nRSS = Σe₍ᵢ₎² = {' + '.join([f'{r**2:.4f}' for r in residuals])} = {rss:.4f}"
    
    ax3.text(0.05, 0.5, residual_text, fontsize=10, verticalalignment='center')
    ax3.axis('off')
    
    # Plot 4: Prediction at x = 2.5
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(x, y, color='blue', s=60, alpha=0.5, label='Data points')
    ax4.plot(x_line, y_line, color='red', linewidth=2, label=f'y = {beta0:.2f} + {beta1:.2f}x')
    ax4.scatter([x_pred], [y_pred], color='green', s=80, label=f'Prediction')
    
    ax4.annotate(f"x = 2.5\nŷ = {y_pred:.2f}", 
                xy=(x_pred, y_pred), 
                xytext=(x_pred+0.3, y_pred-1),
                fontsize=10,
                arrowprops=dict(arrowstyle="->", color='green'))
    
    ax4.set_title('Prediction at x = 2.5', fontsize=12)
    ax4.set_xlabel('x', fontsize=10)
    ax4.set_ylabel('y', fontsize=10)
    ax4.grid(True)
    ax4.legend(fontsize=8)
    
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot3_step_by_step.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    return saved_files

# Create visualizations
saved_files = create_visualizations(x, y, beta0, beta1, x_line, y_line, x_pred, y_pred, residuals, save_dir)

# Print summary of results
print("\n## Summary of Results:")
print(f"1. Least squares estimates: β₀ = {beta0}, β₁ = {beta1}")
print(f"2. Regression equation: y = {beta0} + {beta1}x")
print(f"3. Predicted value at x = 2.5: y = {y_pred}")
print(f"4. Residual sum of squares (RSS): {rss}")
print(f"\nVisualizations saved to: {save_dir}")
print(f"Files created: {', '.join([os.path.basename(f) for f in saved_files])}") 