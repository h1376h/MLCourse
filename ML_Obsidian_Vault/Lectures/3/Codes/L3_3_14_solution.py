import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_3_Quiz_14")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Problem parameters
true_function = lambda x: x**2  # f(x) = x²
optimal_w0 = 2  # w₀*
optimal_w1 = 2  # w₁*
estimated_w0 = 1.5  # ŵ₀
estimated_w1 = 2.5  # ŵ₁
evaluation_point = 3  # x = 3

# Define the optimal and estimated linear models
optimal_model = lambda x: optimal_w0 + optimal_w1 * x
estimated_model = lambda x: estimated_w0 + estimated_w1 * x

# Step 1: Calculate the true value y = f(3)
true_value = true_function(evaluation_point)
print(f"Step 1: Calculate the true value y = f(3)")
print(f"y = f(3) = 3² = {true_value}")
print()

# Step 2: Calculate the prediction from the optimal linear model
optimal_prediction = optimal_model(evaluation_point)
print(f"Step 2: Calculate the prediction from the optimal linear model")
print(f"y_opt = w₀* + w₁* × 3 = {optimal_w0} + {optimal_w1} × 3 = {optimal_prediction}")
print()

# Step 3: Calculate the prediction from the estimated model
estimated_prediction = estimated_model(evaluation_point)
print(f"Step 3: Calculate the prediction from the estimated model")
print(f"ŷ = ŵ₀ + ŵ₁ × 3 = {estimated_w0} + {estimated_w1} × 3 = {estimated_prediction}")
print()

# Step 4: Compute the structural error
structural_error = (true_value - optimal_prediction)**2
print(f"Step 4: Compute the structural error")
print(f"Structural error = (y - y_opt)² = ({true_value} - {optimal_prediction})² = {structural_error}")
print()

# Step 5: Compute the approximation error
approximation_error = (optimal_prediction - estimated_prediction)**2
print(f"Step 5: Compute the approximation error")
print(f"Approximation error = (y_opt - ŷ)² = ({optimal_prediction} - {estimated_prediction})² = {approximation_error}")
print()

# Step 6: Verify that the total squared error equals the sum of structural and approximation errors
total_error = (true_value - estimated_prediction)**2
sum_of_errors = structural_error + approximation_error
print(f"Step 6: Verify that the total squared error equals the sum of structural and approximation errors")
print(f"Total squared error = (y - ŷ)² = ({true_value} - {estimated_prediction})² = {total_error}")
print(f"Sum of errors = Structural error + Approximation error = {structural_error} + {approximation_error} = {sum_of_errors}")
print(f"Are they equal? {np.isclose(total_error, sum_of_errors)}")
print()

# Create a visualization of the functions and errors
def create_visualizations():
    # Range of x values
    x = np.linspace(-1, 5, 1000)
    
    # Calculate function values
    y_true = true_function(x)
    y_optimal = optimal_model(x)
    y_estimated = estimated_model(x)
    
    # Plot 1: All three functions
    plt.figure(figsize=(12, 8))
    plt.plot(x, y_true, 'b-', linewidth=2.5, label='True function $f(x) = x^2$')
    plt.plot(x, y_optimal, 'g-', linewidth=2.5, label='Optimal linear model $y_{opt} = 2 + 2x$')
    plt.plot(x, y_estimated, 'r-', linewidth=2.5, label='Estimated linear model $\hat{y} = 1.5 + 2.5x$')
    
    # Add a marker at the evaluation point
    plt.scatter([evaluation_point], [true_value], color='blue', s=80, zorder=5)
    plt.scatter([evaluation_point], [optimal_prediction], color='green', s=80, zorder=5)
    plt.scatter([evaluation_point], [estimated_prediction], color='red', s=80, zorder=5)
    
    # Add vertical lines to highlight errors
    plt.vlines(x=evaluation_point, ymin=min(true_value, optimal_prediction, estimated_prediction) - 0.5, 
              ymax=max(true_value, optimal_prediction, estimated_prediction) + 0.5, 
              linestyles='dashed', colors='gray')
    
    # Add text annotations for the values
    plt.text(evaluation_point + 0.1, true_value, f'True: {true_value}', fontsize=12, verticalalignment='center')
    plt.text(evaluation_point + 0.1, optimal_prediction, f'Optimal: {optimal_prediction}', fontsize=12, verticalalignment='center')
    plt.text(evaluation_point + 0.1, estimated_prediction, f'Estimated: {estimated_prediction}', fontsize=12, verticalalignment='center')
    
    plt.title('Question 14: Function Comparison at x = 3', fontsize=16)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    
    # Add a text box with the error calculations
    text_box = (
        f"At x = 3:\n"
        f"True value (y): {true_value}\n"
        f"Optimal model prediction (y_opt): {optimal_prediction}\n"
        f"Estimated model prediction (ŷ): {estimated_prediction}\n\n"
        f"Structural error: {structural_error:.2f}\n"
        f"Approximation error: {approximation_error:.2f}\n"
        f"Total error: {total_error:.2f}"
    )
    plt.figtext(0.15, 0.15, text_box, fontsize=12, 
               bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the figure
    file_path1 = os.path.join(save_dir, "function_comparison.png")
    plt.savefig(file_path1, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Visualizing the errors
    plt.figure(figsize=(12, 8))
    
    # Focus on the area around the evaluation point
    x_focus = np.linspace(evaluation_point - 1, evaluation_point + 1, 1000)
    y_true_focus = true_function(x_focus)
    y_optimal_focus = optimal_model(x_focus)
    y_estimated_focus = estimated_model(x_focus)
    
    plt.plot(x_focus, y_true_focus, 'b-', linewidth=2.5, label='True function $f(x) = x^2$')
    plt.plot(x_focus, y_optimal_focus, 'g-', linewidth=2.5, label='Optimal linear model $y_{opt} = 2 + 2x$')
    plt.plot(x_focus, y_estimated_focus, 'r-', linewidth=2.5, label='Estimated linear model $\hat{y} = 1.5 + 2.5x$')
    
    # Add a marker at the evaluation point
    plt.scatter([evaluation_point], [true_value], color='blue', s=80, zorder=5)
    plt.scatter([evaluation_point], [optimal_prediction], color='green', s=80, zorder=5)
    plt.scatter([evaluation_point], [estimated_prediction], color='red', s=80, zorder=5)
    
    # Highlight structural error
    plt.fill_between([evaluation_point-0.1, evaluation_point+0.1], 
                    [true_value, true_value], 
                    [optimal_prediction, optimal_prediction], 
                    color='green', alpha=0.3, label='Structural Error')
    
    # Highlight approximation error
    plt.fill_between([evaluation_point-0.1, evaluation_point+0.1], 
                    [optimal_prediction, optimal_prediction], 
                    [estimated_prediction, estimated_prediction], 
                    color='red', alpha=0.3, label='Approximation Error')
    
    # Add arrows and labels to show the errors
    plt.annotate('', xy=(evaluation_point, optimal_prediction), xytext=(evaluation_point, true_value),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    
    plt.annotate('', xy=(evaluation_point, estimated_prediction), xytext=(evaluation_point, optimal_prediction),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    
    # Add text annotations for the errors
    mid_structural = (true_value + optimal_prediction) / 2
    mid_approximation = (optimal_prediction + estimated_prediction) / 2
    
    plt.text(evaluation_point + 0.15, mid_structural, 
            f'Structural Error: {structural_error:.2f}', 
            fontsize=12, color='green')
    
    plt.text(evaluation_point + 0.15, mid_approximation, 
            f'Approximation Error: {approximation_error:.2f}', 
            fontsize=12, color='red')
    
    plt.title('Question 14: Visualization of Errors at x = 3', fontsize=16)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12, loc='upper left')
    
    # Add a mathematical expression showing error decomposition
    math_text = r"$(y - \hat{y})^2 = (y - y_{opt})^2 + (y_{opt} - \hat{y})^2$"
    plt.figtext(0.5, 0.02, math_text, fontsize=14, ha='center')
    
    plt.tight_layout()
    
    # Save the figure
    file_path2 = os.path.join(save_dir, "error_visualization.png")
    plt.savefig(file_path2, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Visualizing error decomposition
    plt.figure(figsize=(12, 8))
    
    # Create a bar chart of the errors
    error_types = ['Structural Error', 'Approximation Error', 'Total Error']
    error_values = [structural_error, approximation_error, total_error]
    colors = ['green', 'red', 'blue']
    
    plt.bar(error_types, error_values, color=colors, alpha=0.7)
    
    # Add value labels on top of each bar
    for i, v in enumerate(error_values):
        plt.text(i, v + 0.1, f'{v:.2f}', ha='center', fontsize=12)
        
    # Add text to verify sum of errors
    plt.text(1.5, max(error_values) * 0.6, 
            f'Structural Error + Approximation Error = {structural_error:.2f} + {approximation_error:.2f} = {sum_of_errors:.2f}\n'
            f'Total Error = {total_error:.2f}', 
            fontsize=12, ha='center',
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title('Question 14: Error Decomposition at x = 3', fontsize=16)
    plt.ylabel('Squared Error', fontsize=14)
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    
    # Save the figure
    file_path3 = os.path.join(save_dir, "error_decomposition.png")
    plt.savefig(file_path3, dpi=300, bbox_inches='tight')
    plt.close()
    
    return [file_path1, file_path2, file_path3]

# Create the visualizations
saved_files = create_visualizations()

print(f"Visualizations saved to: {', '.join(saved_files)}")
print("\nQuestion 14 Solution Summary:")
print(f"1. True value at x = 3: y = {true_value}")
print(f"2. Optimal linear model prediction: y_opt = {optimal_prediction}")
print(f"3. Estimated model prediction: ŷ = {estimated_prediction}")
print(f"4. Structural error: {structural_error}")
print(f"5. Approximation error: {approximation_error}")
print(f"6. Total squared error: {total_error}")
print(f"   Sum of structural and approximation errors: {sum_of_errors}")
print(f"   Verification: {np.isclose(total_error, sum_of_errors)}") 