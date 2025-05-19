import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_2_Quiz_37")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Part 1: Verify which statements are true for least squares solutions
print_step_header(1, "Verifying Properties of Least Squares Solutions")

print("For a least squares solution w* = [w0*, w1*], we need to verify which statements must be true.")
print("Let's derive the expressions by taking the derivative of the loss function J(w):")

print("\nLoss function: J(w) = (1/n) * sum((yi - w0 - w1*xi)^2)")
print("\nDerivative with respect to w0:")
print("∂J/∂w0 = -(2/n) * sum(yi - w0 - w1*xi)")
print("\nDerivative with respect to w1:")
print("∂J/∂w1 = -(2/n) * sum((yi - w0 - w1*xi) * xi)")

print("\nSetting derivatives to zero for w0* and w1*:")
print("sum(yi - w0* - w1*xi) = 0        ... (1)")
print("sum((yi - w0* - w1*xi) * xi) = 0  ... (2)")

print("\nNow let's analyze each given statement:")

# Define symbolic variables
w0, w1, y, x, y_bar, x_bar = sp.symbols('w0 w1 y x y_bar x_bar')
residual = y - w0 - w1*x

# Statement 1: (1/n)∑(yi - w0* - w1*xi)yi = 0
print("\nStatement 1: (1/n)∑(yi - w0* - w1*xi)yi = 0")
expression1 = residual * y
print(f"Expanding: {expression1}")
expanded1 = sp.expand(expression1)
print(f"Expanded: {expanded1}")
print("For proper verification, let's rewrite this:")
print("∑(yi - w0* - w1*xi)yi = ∑yi²- w0*∑yi - w1*∑xiyi")
print("To determine if this is always true, we need to check if it follows from our optimality conditions.")
print("Unfortunately, we cannot directly derive this from our optimality conditions (1) and (2).")
print("So we cannot prove that this statement is always true for any dataset.")

# Statement 2: (1/n)∑(yi - w0* - w1*xi)(yi - y_bar) = 0
print("\nStatement 2: (1/n)∑(yi - w0* - w1*xi)(yi - y_bar) = 0")
expression2 = residual * (y - y_bar)
print(f"Expanding: {expression2}")
expanded2 = sp.expand(expression2)
print(f"Expanded: {expanded2}")
print("After rearranging:")
print("∑(yi - w0* - w1*xi)(yi - y_bar) = ∑yi(yi - y_bar) - w0*∑(yi - y_bar) - w1*∑xi(yi - y_bar)")
print("From optimality condition (1), we know that ∑(yi - w0* - w1*xi) = 0")
print("This means w0*∑1 + w1*∑xi = ∑yi, or w0*n + w1*∑xi = ∑yi")
print("Therefore, w0* = (∑yi - w1*∑xi)/n = y_bar - w1*x_bar")
print("However, we still can't directly derive statement 2 from our optimality conditions.")
print("So we cannot conclusively prove that statement 2 is always true.")

# Statement 3: (1/n)∑(yi - w0* - w1*xi)(xi - x_bar) = 0
print("\nStatement 3: (1/n)∑(yi - w0* - w1*xi)(xi - x_bar) = 0")
expression3 = residual * (x - x_bar)
print(f"Expanding: {expression3}")
expanded3 = sp.expand(expression3)
print(f"Expanded: {expanded3}")
print("Let's rearrange to see if this matches our conditions:")
print("∑(yi - w0* - w1*xi)(xi - x_bar) = ∑(yi - w0* - w1*xi)xi - x_bar∑(yi - w0* - w1*xi)")
print("From our optimality conditions:")
print("(1) ∑(yi - w0* - w1*xi) = 0")
print("(2) ∑(yi - w0* - w1*xi)xi = 0")
print("Substituting these into our expression:")
print("∑(yi - w0* - w1*xi)(xi - x_bar) = 0 - x_bar*0 = 0")
print("Therefore, statement 3 is TRUE. It directly follows from our optimality conditions.")

# Statement 4: (1/n)∑(yi - w0* - w1*xi)(w0* + w1*xi) = 0
print("\nStatement 4: (1/n)∑(yi - w0* - w1*xi)(w0* + w1*xi) = 0")
expression4 = residual * (w0 + w1*x)
print(f"Expanding: {expression4}")
expanded4 = sp.expand(expression4)
print(f"Expanded: {expanded4}")
print("Let's rewrite this more carefully:")
print("∑(yi - w0* - w1*xi)(w0* + w1*xi) = ")
print("w0*∑(yi - w0* - w1*xi) + w1*∑(yi - w0* - w1*xi)xi")
print("From our optimality conditions (1) and (2):")
print("∑(yi - w0* - w1*xi) = 0")
print("∑(yi - w0* - w1*xi)xi = 0")
print("Substituting these values:")
print("w0*∑(yi - w0* - w1*xi) + w1*∑(yi - w0* - w1*xi)xi = w0*·0 + w1*·0 = 0")
print("Therefore, statement 4 is actually TRUE. It follows directly from our optimality conditions.")

print("\nSummary of Part 1:")
print("Statement 1: FALSE (cannot be proven to be always true)")
print("Statement 2: FALSE (cannot be proven to be always true)")
print("Statement 3: TRUE (follows directly from the least squares optimality conditions)")
print("Statement 4: TRUE (follows directly from the least squares optimality conditions)")

# Create a table visualization for statement verification
print_step_header(1.1, "Creating Summary Table for Statements")

plt.figure(figsize=(10, 6))
statements = [
    "$\\frac{1}{n}\\sum_{i=1}^n (y_i - w_0^* - w_1^* x_i) y_i = 0$",
    "$\\frac{1}{n}\\sum_{i=1}^n (y_i - w_0^* - w_1^* x_i)(y_i - \\bar{y}) = 0$",
    "$\\frac{1}{n}\\sum_{i=1}^n (y_i - w_0^* - w_1^* x_i)(x_i - \\bar{x}) = 0$",
    "$\\frac{1}{n}\\sum_{i=1}^n (y_i - w_0^* - w_1^* x_i)(w_0^* + w_1^* x_i) = 0$"
]
results = ["FALSE", "FALSE", "TRUE", "TRUE"]
colors = ["red", "red", "green", "green"]

# Create a table plot
table_data = [
    [statements[i], results[i]] for i in range(len(statements))
]
table = plt.table(cellText=table_data, 
                 colLabels=["Statement", "True?"],
                 cellLoc='center',
                 loc='center',
                 cellColours=[[None, colors[i]] for i in range(len(statements))],
                 colWidths=[0.8, 0.2])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)
plt.axis('off')
plt.title('Summary of Least Squares Properties', fontsize=14)

# Save the figure
file_path = os.path.join(save_dir, "statement_verification_table.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nTable visualization saved to: {file_path}")

# Part 2: Calculate least squares for the given dataset
print_step_header(2, "Least Squares Solution for the Dataset")

# Given dataset
x_data = np.array([1, 2, 3])
y_data = np.array([2, 3, 5])

# Calculate means
x_mean = np.mean(x_data)
y_mean = np.mean(y_data)

print(f"Dataset: (1,2), (2,3), (3,5)")
print(f"Mean of x: {x_mean}")
print(f"Mean of y: {y_mean}")

# Calculate the least squares parameters
numerator = np.sum((x_data - x_mean) * (y_data - y_mean))
denominator = np.sum((x_data - x_mean) ** 2)
w1_star = numerator / denominator
w0_star = y_mean - w1_star * x_mean

print(f"\nCalculating w1*:")
print(f"w1* = sum((xi - x_mean) * (yi - y_mean)) / sum((xi - x_mean)^2)")
print(f"w1* = {numerator} / {denominator} = {w1_star}")

print(f"\nCalculating w0*:")
print(f"w0* = y_mean - w1* * x_mean")
print(f"w0* = {y_mean} - {w1_star} * {x_mean} = {w0_star}")

print(f"\nLeast squares line: y = {w0_star} + {w1_star}x")

# Create visualization
plt.figure(figsize=(10, 7))
plt.scatter(x_data, y_data, color='blue', s=100, label='Data points')

# Plot the least squares line
x_line = np.linspace(0, 4, 100)
y_line = w0_star + w1_star * x_line
plt.plot(x_line, y_line, 'r-', label=f'Least squares: y = {w0_star:.2f} + {w1_star:.2f}x')

# Add grid for better visual estimation
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, 4)
plt.ylim(0, 6)

# Highlight the mean point
plt.scatter([x_mean], [y_mean], color='green', s=150, marker='X', label='Mean point')

# Add annotations
for i in range(len(x_data)):
    plt.annotate(f'({x_data[i]}, {y_data[i]})', 
                 (x_data[i], y_data[i]), 
                 xytext=(10, 10),
                 textcoords='offset points')

plt.title('Least Squares Fitting', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend()

# Save the figure
file_path = os.path.join(save_dir, "least_squares_fit.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nFigure saved to: {file_path}")

# Create 3D loss function visualization
print_step_header(2.1, "Loss Function Visualization")

# Create a meshgrid for w0 and w1 values
w0_range = np.linspace(-1, 2, 50)
w1_range = np.linspace(0, 3, 50)
W0, W1 = np.meshgrid(w0_range, w1_range)

# Calculate the loss function value for each (w0, w1) pair
J = np.zeros(W0.shape)
n = len(x_data)

for i in range(len(w0_range)):
    for j in range(len(w1_range)):
        w0_val = w0_range[i]
        w1_val = w1_range[j]
        predictions = w0_val + w1_val * x_data
        errors = y_data - predictions
        J[j, i] = np.mean(errors**2)  # MSE

# Plot the loss function as a 3D surface
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(W0, W1, J, cmap=cm.coolwarm, linewidth=0, antialiased=True, alpha=0.8)

# Mark the minimum point
min_j_idx = np.unravel_index(np.argmin(J), J.shape)
min_w0 = W0[min_j_idx]
min_w1 = W1[min_j_idx]
min_j = J[min_j_idx]

ax.scatter([w0_star], [w1_star], [np.min(J)], color='black', s=100, marker='*')
ax.text(w0_star, w1_star, np.min(J), f'Minimum: ({w0_star:.2f}, {w1_star:.2f})', 
        fontsize=12, verticalalignment='bottom')

ax.set_xlabel('w0', fontsize=12)
ax.set_ylabel('w1', fontsize=12)
ax.set_zlabel('J(w0, w1)', fontsize=12)
ax.set_title('Loss Function Surface', fontsize=14)

# Add a color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Save the figure
file_path = os.path.join(save_dir, "loss_function_3d.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nFigure saved to: {file_path}")

# Create contour plot of residual sum of squares
print_step_header(2.2, "Contour Plot of Residual Sum of Squares")

plt.figure(figsize=(10, 8))
contour = plt.contourf(W0, W1, J, 50, cmap=cm.coolwarm, alpha=0.8)
plt.colorbar(contour, label='Mean Squared Error')

# Mark the minimum point
plt.scatter([w0_star], [w1_star], color='black', s=150, marker='*', label=f'Minimum: ({w0_star:.2f}, {w1_star:.2f})')

# Add contour lines
contour_lines = plt.contour(W0, W1, J, 10, colors='black', alpha=0.4)
plt.clabel(contour_lines, inline=True, fontsize=10, fmt='%.2f')

# Annotate the axes
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

plt.title('Contour Plot of Mean Squared Error', fontsize=14)
plt.xlabel('w0', fontsize=12)
plt.ylabel('w1', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.3)

# Save the figure
file_path = os.path.join(save_dir, "mse_contour.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nFigure saved to: {file_path}")

# Part 3: Calculate residuals and verify properties
print_step_header(3, "Residual Properties Verification")

# Given parameters
w0_given = 1
w1_given = 2

# Given data points
x_points = np.array([0, 1, 2, 3])
y_points = np.array([0, 2, 5, 9])

# Calculate predicted values
y_pred = w0_given + w1_given * x_points

# Calculate residuals
residuals = y_points - y_pred

print("For the least squares solution w0* = 1, w1* = 2:")
print("\nCalculating residuals for each point:")
for i in range(len(x_points)):
    print(f"Point ({x_points[i]}, {y_points[i]}):")
    print(f"  Predicted value: {w0_given} + {w1_given} * {x_points[i]} = {y_pred[i]}")
    print(f"  Residual: {y_points[i]} - {y_pred[i]} = {residuals[i]}")

# Verify properties
sum_residuals = np.sum(residuals)
sum_residuals_x = np.sum(residuals * x_points)

print("\nVerifying properties:")
print(f"Sum of residuals: {sum_residuals}")
print(f"Sum of residuals * x: {sum_residuals_x}")

# Calculate the actual least squares solution for this dataset
x_points_mean = np.mean(x_points)
y_points_mean = np.mean(y_points)
w1_actual = np.sum((x_points - x_points_mean) * (y_points - y_points_mean)) / np.sum((x_points - x_points_mean) ** 2)
w0_actual = y_points_mean - w1_actual * x_points_mean

print("\nActual least squares solution for this dataset:")
print(f"w1* = {w1_actual}")
print(f"w0* = {w0_actual}")
print(f"Actual best fit line: y = {w0_actual} + {w1_actual}x")

# Calculate residuals for the actual solution
y_pred_actual = w0_actual + w1_actual * x_points
residuals_actual = y_points - y_pred_actual
sum_residuals_actual = np.sum(residuals_actual)
sum_residuals_x_actual = np.sum(residuals_actual * x_points)

print("\nResiduals with the actual least squares solution:")
for i in range(len(x_points)):
    print(f"Point ({x_points[i]}, {y_points[i]}):")
    print(f"  Predicted value: {w0_actual} + {w1_actual} * {x_points[i]} = {y_pred_actual[i]}")
    print(f"  Residual: {y_points[i]} - {y_pred_actual[i]} = {residuals_actual[i]}")

print("\nVerifying properties for actual solution:")
print(f"Sum of residuals: {sum_residuals_actual}")
print(f"Sum of residuals * x: {sum_residuals_x_actual}")

# Create residual plot
plt.figure(figsize=(10, 7))

# Plot data points and regression line
plt.scatter(x_points, y_points, color='blue', s=100, label='Data points')
x_line = np.linspace(-0.5, 3.5, 100)
y_line_given = w0_given + w1_given * x_line
y_line_actual = w0_actual + w1_actual * x_line
plt.plot(x_line, y_line_given, 'r-', label=f'Given model: y = {w0_given} + {w1_given}x')
plt.plot(x_line, y_line_actual, 'g--', label=f'Actual LS: y = {w0_actual:.2f} + {w1_actual:.2f}x')

# Plot residuals for the given model
for i in range(len(x_points)):
    plt.vlines(x=x_points[i], ymin=y_pred[i], ymax=y_points[i], 
               colors='red', linestyles='dashed', label='Given model residuals' if i == 0 else "")
    plt.annotate(f'{residuals[i]:.1f}', 
                 (x_points[i], (y_points[i] + y_pred[i])/2), 
                 xytext=(10, 0),
                 textcoords='offset points',
                 color='red')

plt.grid(True, linestyle='--', alpha=0.7)
plt.title('Residuals Comparison', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend()

# Save the figure
file_path = os.path.join(save_dir, "residuals_plot.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nFigure saved to: {file_path}")

# Create a visual representation of residual properties
plt.figure(figsize=(12, 8))
gs = GridSpec(2, 2)

# Plot showing sum of residuals = 0 for given model
ax1 = plt.subplot(gs[0, 0])
bars = ax1.bar(range(len(residuals)), residuals, color='red', alpha=0.7)
ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax1.set_title('Given Model Residuals', fontsize=12)
ax1.set_xlabel('Data Points', fontsize=10)
ax1.set_ylabel('Residual Value', fontsize=10)
ax1.set_xticks(range(len(residuals)))
ax1.set_xticklabels([f'({x},{y})' for x, y in zip(x_points, y_points)])

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., 
             height + 0.1 if height > 0 else height - 0.3,
             f'{residuals[i]:.1f}',
             ha='center', va='bottom')

# Text annotation for sum of residuals
sum_text = f'Sum of Residuals = {sum_residuals:.1f}'
ax1.text(0.5, -0.2, sum_text, ha='center', transform=ax1.transAxes, fontsize=12)

# Plot showing sum of residuals * x = 0 for given model
ax2 = plt.subplot(gs[0, 1])
weighted_residuals = residuals * x_points
bars = ax2.bar(range(len(weighted_residuals)), weighted_residuals, color='red', alpha=0.7)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax2.set_title('Given Model Residuals × x', fontsize=12)
ax2.set_xlabel('Data Points', fontsize=10)
ax2.set_ylabel('Residual × x Value', fontsize=10)
ax2.set_xticks(range(len(weighted_residuals)))
ax2.set_xticklabels([f'({x},{y})' for x, y in zip(x_points, y_points)])

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., 
             height + 0.1 if height > 0 else height - 0.3,
             f'{weighted_residuals[i]:.1f}',
             ha='center', va='bottom')

# Text annotation for sum of weighted residuals
weighted_sum_text = f'Sum of Residuals × x = {sum_residuals_x:.1f}'
ax2.text(0.5, -0.2, weighted_sum_text, ha='center', transform=ax2.transAxes, fontsize=12)

# Plot showing sum of residuals = 0 for actual model
ax3 = plt.subplot(gs[1, 0])
bars = ax3.bar(range(len(residuals_actual)), residuals_actual, color='green', alpha=0.7)
ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax3.set_title('Actual LS Model Residuals', fontsize=12)
ax3.set_xlabel('Data Points', fontsize=10)
ax3.set_ylabel('Residual Value', fontsize=10)
ax3.set_xticks(range(len(residuals_actual)))
ax3.set_xticklabels([f'({x},{y})' for x, y in zip(x_points, y_points)])

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., 
             height + 0.1 if height > 0 else height - 0.3,
             f'{residuals_actual[i]:.2f}',
             ha='center', va='bottom')

# Text annotation for sum of residuals
sum_actual_text = f'Sum of Residuals = {sum_residuals_actual:.2f}'
ax3.text(0.5, -0.2, sum_actual_text, ha='center', transform=ax3.transAxes, fontsize=12)

# Plot showing sum of residuals * x = 0 for actual model
ax4 = plt.subplot(gs[1, 1])
weighted_residuals_actual = residuals_actual * x_points
bars = ax4.bar(range(len(weighted_residuals_actual)), weighted_residuals_actual, color='green', alpha=0.7)
ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax4.set_title('Actual LS Model Residuals × x', fontsize=12)
ax4.set_xlabel('Data Points', fontsize=10)
ax4.set_ylabel('Residual × x Value', fontsize=10)
ax4.set_xticks(range(len(weighted_residuals_actual)))
ax4.set_xticklabels([f'({x},{y})' for x, y in zip(x_points, y_points)])

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., 
             height + 0.1 if height > 0 else height - 0.3,
             f'{weighted_residuals_actual[i]:.2f}',
             ha='center', va='bottom')

# Text annotation for sum of weighted residuals
weighted_sum_actual_text = f'Sum of Residuals × x = {sum_residuals_x_actual:.2f}'
ax4.text(0.5, -0.2, weighted_sum_actual_text, ha='center', transform=ax4.transAxes, fontsize=12)

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "residual_properties.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nFigure saved to: {file_path}")

# Final conclusion
print_step_header(4, "Conclusion")

print("Summary of findings:")
print("\n1. Properties of Least Squares Solutions:")
print("   - Statement 1: FALSE (not proven to be always true)")
print("   - Statement 2: FALSE (not proven to be always true)")
print("   - Statement 3: TRUE (follows directly from the least squares optimality conditions)")
print("   - Statement 4: TRUE (follows directly from the least squares optimality conditions)")

print("\n2. Least Squares Solution for the Dataset (1,2), (2,3), (3,5):")
print(f"   - w1* = {w1_star}")
print(f"   - w0* = {w0_star}")
print(f"   - Best fit line: y = {w0_star} + {w1_star}x")

print("\n3. Residual Properties:")
print("   - For the given model (w0 = 1, w1 = 2):")
print(f"     - Sum of residuals = {sum_residuals} (zero, as expected)")
print(f"     - Sum of residuals × x = {sum_residuals_x} (not zero)")
print("   - For the actual least squares solution:")
print(f"     - w0* = {w0_actual}, w1* = {w1_actual}")
print(f"     - Sum of residuals = {sum_residuals_actual:.2f} (approximately zero)")
print(f"     - Sum of residuals × x = {sum_residuals_x_actual:.2f} (approximately zero)")
print("   - This confirms that the model w0* = 1, w1* = 2 is not the least squares solution")
print("     for the given dataset, since it doesn't satisfy all optimality conditions.")

print("\nAll outputs and visualizations have been saved to the Images directory.") 