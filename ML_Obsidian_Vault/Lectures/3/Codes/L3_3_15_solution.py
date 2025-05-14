import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle
import pandas as pd

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_3_Quiz_15")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

print("Question 15: Error Decomposition in Regression Models")
print("="*80)

# Given data - Original problem specification
x = np.array([1, 2, 3])
true_function = lambda x: x**2  # f(x) = x^2
optimal_model = lambda x: -1 + 3*x  # Best possible linear approximation
estimated_model = lambda x: -0.5 + 2.5*x  # Estimated model

# Calculate true values and model predictions
y_true = np.zeros(len(x))
y_opt = np.zeros(len(x))
y_est = np.zeros(len(x))

for i, xi in enumerate(x):
    y_true[i] = true_function(xi)
    y_opt[i] = optimal_model(xi)
    y_est[i] = estimated_model(xi)

print("Step 1: Calculate the predictions from both models for each x value")
print("-"*80)
print(f"x values: {x}")
print(f"True function: f(x) = x²")
print(f"Optimal model: ŷ_opt = -1 + 3x")
print(f"Estimated model: ŷ_est = -0.5 + 2.5x")
print()

print("True y values (y = x²):")
for i, xi in enumerate(x):
    print(f"x = {xi}: y = {xi}² = {y_true[i]}")
print()

print("Predictions from the optimal linear model (ŷ_opt = -1 + 3x):")
for i, xi in enumerate(x):
    print(f"x = {xi}: ŷ_opt = -1 + 3*{xi} = -1 + {3*xi} = {y_opt[i]}")
print()

print("Predictions from the estimated model (ŷ_est = -0.5 + 2.5x):")
for i, xi in enumerate(x):
    print(f"x = {xi}: ŷ_est = -0.5 + 2.5*{xi} = -0.5 + {2.5*xi} = {y_est[i]}")
print()

print("Step 2: Compute the structural error for each point")
print("-"*80)
structural_error = np.zeros(len(x))
for i, xi in enumerate(x):
    # Calculate structural error for each point
    error = y_true[i] - y_opt[i]
    squared_error = error**2
    structural_error[i] = squared_error
    
    print(f"x = {xi}:")
    print(f"  True value: f({xi}) = {y_true[i]}")
    print(f"  Optimal model: ŷ_opt({xi}) = {y_opt[i]}")
    print(f"  Error: f({xi}) - ŷ_opt({xi}) = {y_true[i]} - {y_opt[i]} = {error}")
    print(f"  Squared error: [{error}]² = {squared_error:.2f}")

avg_structural_error = sum(structural_error) / len(structural_error)
print(f"\nAverage structural error: ({' + '.join([f'{err:.2f}' for err in structural_error])}) / {len(structural_error)} = {avg_structural_error:.2f}")
print()

print("Step 3: Compute the estimation error for each point")
print("-"*80)
estimation_error = np.zeros(len(x))
for i, xi in enumerate(x):
    # Calculate estimation error for each point
    error = y_opt[i] - y_est[i]
    squared_error = error**2
    estimation_error[i] = squared_error
    
    print(f"x = {xi}:")
    print(f"  Optimal model: ŷ_opt({xi}) = {y_opt[i]}")
    print(f"  Estimated model: ŷ_est({xi}) = {y_est[i]}")
    print(f"  Error: ŷ_opt({xi}) - ŷ_est({xi}) = {y_opt[i]} - {y_est[i]} = {error}")
    print(f"  Squared error: [{error}]² = {squared_error:.2f}")

avg_estimation_error = sum(estimation_error) / len(estimation_error)
print(f"\nAverage estimation error: ({' + '.join([f'{err:.2f}' for err in estimation_error])}) / {len(estimation_error)} = {avg_estimation_error:.2f}")
print()

print("Step 4: Calculate the cross-product term for each point")
print("-"*80)
cross_term = np.zeros(len(x))
for i, xi in enumerate(x):
    # Calculate the components for the cross term
    diff1 = y_true[i] - y_opt[i]  # (y - y_opt)
    diff2 = y_opt[i] - y_est[i]   # (y_opt - y_est)
    term = 2 * diff1 * diff2      # 2(y - y_opt)(y_opt - y_est)
    cross_term[i] = term
    
    print(f"x = {xi}:")
    print(f"  (y - y_opt): {y_true[i]} - {y_opt[i]} = {diff1}")
    print(f"  (y_opt - y_est): {y_opt[i]} - {y_est[i]} = {diff2}")
    print(f"  Cross term: 2 × ({diff1}) × ({diff2}) = 2 × {diff1 * diff2} = {term:.2f}")

avg_cross_term = sum(cross_term) / len(cross_term)
print(f"\nAverage cross-product term: ({' + '.join([f'{term:.2f}' for term in cross_term])}) / {len(cross_term)} = {avg_cross_term:.2f}")
print()

print("Step 5: Calculate the total squared error and verify decomposition")
print("-"*80)
total_error = np.zeros(len(x))
for i, xi in enumerate(x):
    # Calculate total error directly
    error = y_true[i] - y_est[i]
    squared_error = error**2
    total_error[i] = squared_error
    
    print(f"x = {xi}:")
    print(f"  True value: f({xi}) = {y_true[i]}")
    print(f"  Estimated model: ŷ_est({xi}) = {y_est[i]}")
    print(f"  Error: f({xi}) - ŷ_est({xi}) = {y_true[i]} - {y_est[i]} = {error}")
    print(f"  Squared error: [{error}]² = {squared_error:.2f}")

avg_total_error = sum(total_error) / len(total_error)
print(f"\nAverage total error: ({' + '.join([f'{err:.2f}' for err in total_error])}) / {len(total_error)} = {avg_total_error:.2f}")
print()

# Verify complete decomposition with cross-term
print("Step 6: Verification of complete error decomposition")
print("-"*80)
for i, xi in enumerate(x):
    sum_with_cross = structural_error[i] + estimation_error[i] + cross_term[i]
    
    print(f"x = {xi}:")
    print(f"  Structural error: {structural_error[i]:.2f}")
    print(f"  Estimation error: {estimation_error[i]:.2f}")
    print(f"  Cross term: {cross_term[i]:.2f}")
    print(f"  Sum of components: {structural_error[i]:.2f} + {estimation_error[i]:.2f} + {cross_term[i]:.2f} = {sum_with_cross:.2f}")
    print(f"  Total error: {total_error[i]:.2f}")
    
    if np.isclose(sum_with_cross, total_error[i]):
        diff = sum_with_cross - total_error[i]
        print(f"  ✓ Complete error decomposition verified (difference: {diff:.8f})")
    else:
        diff = sum_with_cross - total_error[i]
        print(f"  ✗ Complete error decomposition failed (difference: {diff:.8f})")
print()

print("Step 7: Examine orthogonality condition for error decomposition")
print("-"*80)
print("For orthogonal models, the cross-product term would be zero.")
print("If we have orthogonality, the error decomposition simplifies to:")
print("(y - y_est)² = (y - y_opt)² + (y_opt - y_est)²")
print()
print("Let's examine if the optimal and estimated models are orthogonal:")

# Calculate dot product manually
dot_product = 0
for i in range(len(x)):
    term = (y_true[i] - y_opt[i]) * (y_opt[i] - y_est[i])
    dot_product += term
    print(f"  Term {i+1}: ({y_true[i]} - {y_opt[i]}) × ({y_opt[i]} - {y_est[i]}) = {term:.4f}")

print(f"Dot product: {' + '.join([f'({y_true[i]} - {y_opt[i]})({y_opt[i]} - {y_est[i]})' for i in range(len(x))])} = {dot_product:.4f}")
print("Since this is not zero, the models are not orthogonal.")
print("Therefore, the simple decomposition without the cross-term does not hold.")
print()

print("Step 8: Calculate the percentage of error due to each component")
print("-"*80)
# Calculate percentages of each error component
percent_structural = (avg_structural_error / avg_total_error) * 100
percent_estimation = (avg_estimation_error / avg_total_error) * 100
percent_cross = (avg_cross_term / avg_total_error) * 100

print(f"Average structural error: {avg_structural_error:.2f}")
print(f"Average estimation error: {avg_estimation_error:.2f}")
print(f"Average cross-term: {avg_cross_term:.2f}")
print(f"Average total error: {avg_total_error:.2f}")
print()
print(f"Percentage of structural error: ({avg_structural_error:.2f} / {avg_total_error:.2f}) × 100% = {percent_structural:.2f}%")
print(f"Percentage of estimation error: ({avg_estimation_error:.2f} / {avg_total_error:.2f}) × 100% = {percent_estimation:.2f}%")
print(f"Percentage of cross-term: ({avg_cross_term:.2f} / {avg_total_error:.2f}) × 100% = {percent_cross:.2f}%")
print(f"Sum of percentages: {percent_structural:.2f}% + {percent_estimation:.2f}% + {percent_cross:.2f}% = {percent_structural + percent_estimation + percent_cross:.2f}%")
print()

# Create summary table
data = {
    'x': x,
    'True y=x²': y_true,
    'Optimal ŷ=-1+3x': y_opt,
    'Estimated ŷ=-0.5+2.5x': y_est,
    'Structural Error': structural_error,
    'Estimation Error': estimation_error,
    'Cross Term': cross_term,
    'Total Error': total_error
}
df = pd.DataFrame(data)
print("Summary of calculations:")
print(df.to_string(index=False))
print()
print(f"Average Structural Error: {avg_structural_error:.2f} ({percent_structural:.2f}% of total)")
print(f"Average Estimation Error: {avg_estimation_error:.2f} ({percent_estimation:.2f}% of total)")
print(f"Average Cross-Term: {avg_cross_term:.2f} ({percent_cross:.2f}% of total)")
print(f"Average Total Error: {avg_total_error:.2f}")

# Create visualizations
def create_visualizations(x, y_true, y_opt, y_est, structural_error, estimation_error, 
                         cross_term, total_error, percent_structural, percent_estimation, 
                         percent_cross, save_dir=None):
    saved_files = []
    
    # Plot 1: Compare the three functions
    plt.figure(figsize=(10, 6))
    
    # Generate more points for smoother curves
    x_smooth = np.linspace(0.5, 3.5, 100)
    y_true_smooth = np.array([true_function(xi) for xi in x_smooth])
    y_opt_smooth = np.array([optimal_model(xi) for xi in x_smooth])
    y_est_smooth = np.array([estimated_model(xi) for xi in x_smooth])
    
    # Plot the curves (fix the escape sequences)
    plt.plot(x_smooth, y_true_smooth, 'b-', linewidth=2, label='True: $y = x^2$')
    plt.plot(x_smooth, y_opt_smooth, 'g-', linewidth=2, label='Optimal: $\\hat{y} = -1 + 3x$')
    plt.plot(x_smooth, y_est_smooth, 'r-', linewidth=2, label='Estimated: $\\hat{y} = -0.5 + 2.5x$')
    
    # Add data points
    plt.scatter(x, y_true, color='blue', s=60, zorder=5)
    
    # Add grid
    plt.grid(True)
    plt.title('Comparison of True Function and Models', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot1_function_comparison.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 2: Visualize errors at each point
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # For each data point
    for i, ax in enumerate(axes):
        xi = x[i]
        y_true_i = y_true[i]
        y_opt_i = y_opt[i]
        y_est_i = y_est[i]
        
        # Plot vertical lines showing the errors (fix the escape sequences)
        ax.scatter([xi], [y_true_i], color='blue', s=80, zorder=5, label='True $y$')
        ax.scatter([xi], [y_opt_i], color='green', s=80, zorder=5, label='Optimal $\\hat{y}_{opt}$')
        ax.scatter([xi], [y_est_i], color='red', s=80, zorder=5, label='Estimated $\\hat{y}_{est}$')
        
        # Draw vertical lines for each error
        ax.plot([xi, xi], [y_true_i, y_opt_i], 'g--', linewidth=2, 
                label='Structural Error')
        ax.plot([xi, xi], [y_opt_i, y_est_i], 'r--', linewidth=2, 
                label='Estimation Error')
        ax.plot([xi, xi], [y_true_i, y_est_i], 'k--', linewidth=1, 
                label='Total Error')
        
        # Add annotations for error values
        ax.annotate(f"SE: {structural_error[i]:.2f}", 
                   xy=(xi+0.05, (y_true_i + y_opt_i)/2), 
                   fontsize=10, color='green',
                   bbox=dict(facecolor='white', alpha=0.8))
        
        ax.annotate(f"EE: {estimation_error[i]:.2f}", 
                   xy=(xi+0.05, (y_opt_i + y_est_i)/2), 
                   fontsize=10, color='red',
                   bbox=dict(facecolor='white', alpha=0.8))
        
        ax.annotate(f"TE: {total_error[i]:.2f}", 
                   xy=(xi+0.1, (y_true_i + y_est_i)/2), 
                   fontsize=10, color='black',
                   bbox=dict(facecolor='white', alpha=0.8))
        
        # Set title and labels
        ax.set_title(f'Errors at x = {xi}', fontsize=12)
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('y', fontsize=10)
        
        # Set consistent y-axis limits
        ax.set_ylim(0, 10)
        ax.set_xlim(xi-0.5, xi+0.5)
        
        if i == 0:
            ax.legend(loc='upper left', fontsize=8)
        
        ax.grid(True)
    
    plt.suptitle('Error Decomposition at Each Data Point', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot2_error_decomposition.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 3: Bar chart of errors
    plt.figure(figsize=(12, 6))
    
    # Set up data for grouped bar chart
    bar_width = 0.2
    index = np.arange(len(x))
    
    plt.bar(index - 1.5*bar_width, structural_error, bar_width, 
            label='Structural Error', color='green', alpha=0.7)
    plt.bar(index - 0.5*bar_width, estimation_error, bar_width, 
            label='Estimation Error', color='red', alpha=0.7)
    plt.bar(index + 0.5*bar_width, cross_term, bar_width, 
            label='Cross Term', color='purple', alpha=0.7)
    plt.bar(index + 1.5*bar_width, total_error, bar_width, 
            label='Total Error', color='black', alpha=0.7)
    
    # Add value labels on top of bars
    for i, v in enumerate(structural_error):
        plt.text(i - 1.5*bar_width, v + 0.1, f'{v:.2f}', ha='center', fontsize=9)
    for i, v in enumerate(estimation_error):
        plt.text(i - 0.5*bar_width, v + 0.1, f'{v:.2f}', ha='center', fontsize=9)
    for i, v in enumerate(cross_term):
        plt.text(i + 0.5*bar_width, v + 0.1, f'{v:.2f}', ha='center', fontsize=9)
    for i, v in enumerate(total_error):
        plt.text(i + 1.5*bar_width, v + 0.1, f'{v:.2f}', ha='center', fontsize=9)
    
    plt.xlabel('Data Point (x value)', fontsize=12)
    plt.ylabel('Squared Error', fontsize=12)
    plt.title('Comparison of Error Components at Each Data Point', fontsize=14)
    plt.xticks(index, [f'x={val}' for val in x])
    plt.legend(fontsize=10)
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot3_error_comparison.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 4: Pie chart of error percentages
    plt.figure(figsize=(8, 8))
    
    labels = [f'Structural Error\n({percent_structural:.1f}%)', 
              f'Estimation Error\n({percent_estimation:.1f}%)',
              f'Cross Term\n({percent_cross:.1f}%)']
    sizes = [percent_structural, percent_estimation, percent_cross]
    colors = ['green', 'red', 'purple']
    explode = (0.1, 0, 0)  # explode the 1st slice
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 12})
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Proportion of Total Error by Component', fontsize=14)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot4_error_proportion.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 5: Error decomposition formula
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    plt.axis('off')
    
    plt.text(0.5, 0.8, "Complete Error Decomposition Formula:", 
             fontsize=16, ha='center', weight='bold')
    
    plt.text(0.5, 0.6, r"$(y - \hat{y}_{est})^2 = (y - \hat{y}_{opt})^2 + (\hat{y}_{opt} - \hat{y}_{est})^2 + 2(y - \hat{y}_{opt})(\hat{y}_{opt} - \hat{y}_{est})$", 
             fontsize=14, ha='center')
    
    plt.text(0.5, 0.4, r"Total Error = Structural Error + Estimation Error + Cross Term", 
             fontsize=14, ha='center')
    
    # Add a note about orthogonality
    plt.text(0.5, 0.2, r"When models are orthogonal: $(y - \hat{y}_{opt}) \perp (\hat{y}_{opt} - \hat{y}_{est})$", 
             fontsize=12, ha='center', style='italic')
    plt.text(0.5, 0.1, r"the cross term becomes zero, simplifying the decomposition.", 
             fontsize=12, ha='center', style='italic')
    
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot5_error_formula.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    return saved_files

# Create visualizations
saved_files = create_visualizations(
    x, y_true, y_opt, y_est, 
    structural_error, estimation_error, cross_term, total_error,
    percent_structural, percent_estimation, percent_cross, save_dir
)

print(f"\nVisualizations saved to: {save_dir}")
for i, file in enumerate(saved_files):
    print(f"{i+1}. {os.path.basename(file)}") 