import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle

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
print("=" * 80)
print("STEP-BY-STEP DETAILED SOLUTION")
print("=" * 80)
print("\nStep 1: Calculate the true value y = f(3)")
print("-" * 50)
print("Given information:")
print(f"* True function: f(x) = x²")
print(f"* Evaluation point: x = {evaluation_point}")
print("\nCalculation:")
print(f"y = f({evaluation_point})")
print(f"y = ({evaluation_point})²")
print(f"y = {evaluation_point} × {evaluation_point}")
true_value = true_function(evaluation_point)
print(f"y = {true_value}")
print("\nAnswer: The true value at x = 3 is y = 9.")
print("-" * 50)

# Step 2: Calculate the prediction from the optimal linear model
print("\nStep 2: Calculate the prediction from the optimal linear model")
print("-" * 50)
print("Given information:")
print(f"* Optimal linear model: y_opt = w₀* + w₁* × x")
print(f"* w₀* = {optimal_w0}")
print(f"* w₁* = {optimal_w1}")
print(f"* Evaluation point: x = {evaluation_point}")
print("\nCalculation:")
print(f"y_opt = w₀* + w₁* × {evaluation_point}")
print(f"y_opt = {optimal_w0} + {optimal_w1} × {evaluation_point}")
print(f"y_opt = {optimal_w0} + {optimal_w1 * evaluation_point}")
optimal_prediction = optimal_model(evaluation_point)
print(f"y_opt = {optimal_prediction}")
print("\nAnswer: The optimal linear model prediction at x = 3 is y_opt = 8.")
print("-" * 50)

# Step 3: Calculate the prediction from the estimated model
print("\nStep 3: Calculate the prediction from the estimated model")
print("-" * 50)
print("Given information:")
print(f"* Estimated linear model: ŷ = ŵ₀ + ŵ₁ × x")
print(f"* ŵ₀ = {estimated_w0}")
print(f"* ŵ₁ = {estimated_w1}")
print(f"* Evaluation point: x = {evaluation_point}")
print("\nCalculation:")
print(f"ŷ = ŵ₀ + ŵ₁ × {evaluation_point}")
print(f"ŷ = {estimated_w0} + {estimated_w1} × {evaluation_point}")
print(f"ŷ = {estimated_w0} + {estimated_w1 * evaluation_point}")
estimated_prediction = estimated_model(evaluation_point)
print(f"ŷ = {estimated_prediction}")
print("\nAnswer: The estimated model prediction at x = 3 is ŷ = 9.0.")
print("-" * 50)

# Step 4: Compute the structural error
print("\nStep 4: Compute the structural error")
print("-" * 50)
print("Definition: Structural error is the squared difference between the true value and")
print("the prediction from the optimal model for the model class.")
print("\nCalculation:")
print(f"Structural error = (y - y_opt)²")
print(f"Structural error = ({true_value} - {optimal_prediction})²")
print(f"Structural error = ({true_value - optimal_prediction})²")
structural_error = (true_value - optimal_prediction)**2
print(f"Structural error = {structural_error}")
print("\nAnswer: The structural error at x = 3 is 1.")
print("This represents the inherent limitation of using a linear model to approximate")
print("a quadratic function, even with optimal parameters.")
print("-" * 50)

# Step 5: Compute the approximation error
print("\nStep 5: Compute the approximation error")
print("-" * 50)
print("Definition: Approximation error is the squared difference between the prediction")
print("from the optimal model and the prediction from the estimated model.")
print("\nCalculation:")
print(f"Approximation error = (y_opt - ŷ)²")
print(f"Approximation error = ({optimal_prediction} - {estimated_prediction})²")
print(f"Approximation error = ({optimal_prediction - estimated_prediction})²")
approximation_error = (optimal_prediction - estimated_prediction)**2
print(f"Approximation error = {approximation_error}")
print("\nAnswer: The approximation error at x = 3 is 1.")
print("This error arises because our estimated parameters (ŵ₀ = 1.5, ŵ₁ = 2.5) differ from")
print("the optimal parameters (w₀* = 2, w₁* = 2), likely due to limited training data.")
print("-" * 50)

# Step 6: Verify that the total squared error equals the sum of structural and approximation errors
print("\nStep 6: Verify that the total squared error equals the sum of errors")
print("-" * 50)
print("Definition: Total squared error is the squared difference between the true value")
print("and the prediction from the estimated model.")
print("\nCalculation of total squared error:")
print(f"Total squared error = (y - ŷ)²")
print(f"Total squared error = ({true_value} - {estimated_prediction})²")
print(f"Total squared error = ({true_value - estimated_prediction})²")
total_error = (true_value - estimated_prediction)**2
print(f"Total squared error = {total_error}")
print("\nCalculation of sum of component errors:")
print(f"Sum of errors = Structural error + Approximation error")
print(f"Sum of errors = {structural_error} + {approximation_error}")
sum_of_errors = structural_error + approximation_error
print(f"Sum of errors = {sum_of_errors}")
print("\nVerification:")
print(f"Total squared error = {total_error}")
print(f"Sum of errors = {sum_of_errors}")
equal = np.isclose(total_error, sum_of_errors)
print(f"Are they equal? {equal}")
print("\nAnswer: The total squared error (0) does NOT equal the sum of structural")
print("and approximation errors (2). This is because we have a special case where")
print("the errors have opposite signs and happen to cancel out exactly at x = 3.")
print("-" * 50)

# Create a visualization of the functions and errors
def create_visualizations():
    saved_files = []
    
    # Plot 1: All three functions
    plt.figure(figsize=(10, 6))
    
    # Range of x values
    x = np.linspace(-1, 5, 1000)
    
    # Calculate function values
    y_true = true_function(x)
    y_optimal = optimal_model(x)
    y_estimated = estimated_model(x)
    
    plt.plot(x, y_true, 'b-', linewidth=2.5, label=f'True function $f(x) = x^2$')
    plt.plot(x, y_optimal, 'g-', linewidth=2.5, label=f'Optimal linear model $y_{{opt}} = {optimal_w0} + {optimal_w1}x$')
    plt.plot(x, y_estimated, 'r-', linewidth=2.5, label=f'Estimated linear model $\\hat{{y}} = {estimated_w0} + {estimated_w1}x$')
    
    # Add a marker at the evaluation point
    plt.scatter([evaluation_point], [true_value], color='blue', s=80, zorder=5)
    plt.scatter([evaluation_point], [optimal_prediction], color='green', s=80, zorder=5)
    plt.scatter([evaluation_point], [estimated_prediction], color='red', s=80, zorder=5)
    
    # Add vertical lines to highlight errors
    plt.vlines(x=evaluation_point, ymin=min(optimal_prediction, estimated_prediction) - 0.5, 
              ymax=max(true_value, estimated_prediction) + 0.5, 
              linestyles='dashed', colors='gray')
    
    # Add text annotations for the values (with better positioning to avoid overlap)
    plt.text(evaluation_point + 0.1, true_value, f'True: {true_value}', 
             fontsize=11, verticalalignment='bottom', horizontalalignment='left')
    plt.text(evaluation_point + 0.1, optimal_prediction - 0.3, f'Optimal: {optimal_prediction}', 
             fontsize=11, verticalalignment='top', horizontalalignment='left')
    plt.text(evaluation_point + 0.1, estimated_prediction, f'Estimated: {estimated_prediction}', 
             fontsize=11, verticalalignment='center', horizontalalignment='left')
    
    plt.title('Question 14: Function Comparison at x = 3', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=11, loc='upper left')
    
    # Add a text box with the error calculations
    text_box = (
        f"At x = {evaluation_point}:\n"
        f"True value (y): {true_value}\n"
        f"Optimal model (y_opt): {optimal_prediction}\n"
        f"Estimated model (ŷ): {estimated_prediction}\n\n"
        f"Structural error: {structural_error:.2f}\n"
        f"Approximation error: {approximation_error:.2f}\n"
        f"Total error: {total_error:.2f}"
    )
    plt.figtext(0.15, 0.15, text_box, fontsize=10, 
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    
    # Save the figure
    file_path1 = os.path.join(save_dir, "function_comparison.png")
    plt.savefig(file_path1, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files.append(file_path1)
    
    # Plot 2: Visualizing the errors
    plt.figure(figsize=(10, 6))
    
    # Focus on the area around the evaluation point
    x_focus = np.linspace(evaluation_point - 1, evaluation_point + 1, 1000)
    y_true_focus = true_function(x_focus)
    y_optimal_focus = optimal_model(x_focus)
    y_estimated_focus = estimated_model(x_focus)
    
    plt.plot(x_focus, y_true_focus, 'b-', linewidth=2.5, label=f'True function $f(x) = x^2$')
    plt.plot(x_focus, y_optimal_focus, 'g-', linewidth=2.5, label=f'Optimal linear model $y_{{opt}} = {optimal_w0} + {optimal_w1}x$')
    plt.plot(x_focus, y_estimated_focus, 'r-', linewidth=2.5, label=f'Estimated linear model $\\hat{{y}} = {estimated_w0} + {estimated_w1}x$')
    
    # Add a marker at the evaluation point
    plt.scatter([evaluation_point], [true_value], color='blue', s=60, zorder=5)
    plt.scatter([evaluation_point], [optimal_prediction], color='green', s=60, zorder=5)
    plt.scatter([evaluation_point], [estimated_prediction], color='red', s=60, zorder=5)
    
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
    
    # Add arrows and labels to show the errors (with improved positioning)
    plt.annotate('', xy=(evaluation_point - 0.05, optimal_prediction), 
                xytext=(evaluation_point - 0.05, true_value),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    
    plt.annotate('', xy=(evaluation_point + 0.05, estimated_prediction), 
                xytext=(evaluation_point + 0.05, optimal_prediction),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    
    # Add text annotations for the errors (with improved positioning)
    mid_structural = (true_value + optimal_prediction) / 2
    mid_approximation = (optimal_prediction + estimated_prediction) / 2
    
    plt.text(evaluation_point - 0.2, mid_structural, 
            f'Structural Error: {structural_error:.2f}', 
            fontsize=10, color='green', horizontalalignment='right')
    
    plt.text(evaluation_point + 0.2, mid_approximation, 
            f'Approximation Error: {approximation_error:.2f}', 
            fontsize=10, color='red', horizontalalignment='left')
    
    plt.title('Question 14: Visualization of Errors at x = 3', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=10, loc='upper left')
    
    # Add a mathematical expression showing error decomposition formula
    math_text = r"$(y - \hat{y})^2 = (y - y_{opt})^2 + (y_{opt} - \hat{y})^2$"
    plt.figtext(0.5, 0.02, math_text, fontsize=12, ha='center')
    
    plt.tight_layout()
    
    # Save the figure
    file_path2 = os.path.join(save_dir, "error_visualization.png")
    plt.savefig(file_path2, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files.append(file_path2)
    
    # Plot 3: Visualizing error decomposition
    plt.figure(figsize=(10, 6))
    
    # Create a bar chart of the errors
    error_types = ['Structural Error', 'Approximation Error', 'Total Error']
    error_values = [structural_error, approximation_error, total_error]
    colors = ['green', 'red', 'blue']
    
    bars = plt.bar(error_types, error_values, color=colors, alpha=0.7)
    
    # Add value labels on top of each bar
    for i, v in enumerate(error_values):
        plt.text(i, v + 0.1, f'{v:.2f}', ha='center', fontsize=12)
        
    # Add text to verify sum of errors (with improved positioning)
    plt.text(1, max(error_values) * 0.7, 
            f'Structural Error + Approximation Error = {structural_error:.2f} + {approximation_error:.2f} = {sum_of_errors:.2f}\n'
            f'Total Error = {total_error:.2f}', 
            fontsize=11, ha='center',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.title('Question 14: Error Decomposition at x = 3', fontsize=14)
    plt.ylabel('Squared Error', fontsize=12)
    plt.ylim(0, max(error_values) * 1.3)  # Set appropriate y-axis limit
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    
    # Save the figure
    file_path3 = os.path.join(save_dir, "error_decomposition.png")
    plt.savefig(file_path3, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files.append(file_path3)
    
    # NEW PLOT 4: Error behavior across different x values
    plt.figure(figsize=(12, 8))
    
    # Define x range for evaluation
    x_range = np.linspace(0, 6, 100)
    
    # Calculate errors at each point
    structural_errors = np.array([(true_function(x) - optimal_model(x))**2 for x in x_range])
    approximation_errors = np.array([(optimal_model(x) - estimated_model(x))**2 for x in x_range])
    total_errors = np.array([(true_function(x) - estimated_model(x))**2 for x in x_range])
    sum_of_component_errors = structural_errors + approximation_errors
    
    # Plot the errors
    plt.plot(x_range, structural_errors, 'g-', linewidth=2, label='Structural Error: $(y - y_{opt})^2$')
    plt.plot(x_range, approximation_errors, 'r-', linewidth=2, label='Approximation Error: $(y_{opt} - \hat{y})^2$')
    plt.plot(x_range, total_errors, 'b-', linewidth=2, label='Total Error: $(y - \hat{y})^2$')
    plt.plot(x_range, sum_of_component_errors, 'k--', linewidth=1.5, label='Sum of Component Errors')
    
    # Add vertical line at x=3
    plt.axvline(x=evaluation_point, color='gray', linestyle=':', linewidth=1.5)
    plt.text(evaluation_point+0.1, max(total_errors)/2, f'x = {evaluation_point}', 
             rotation=90, verticalalignment='center')
    
    # Add markers at x=3
    plt.scatter([evaluation_point], [structural_error], color='green', s=80, zorder=5)
    plt.scatter([evaluation_point], [approximation_error], color='red', s=80, zorder=5)
    plt.scatter([evaluation_point], [total_error], color='blue', s=80, zorder=5)
    plt.scatter([evaluation_point], [sum_of_errors], color='black', s=80, zorder=5)
    
    # Highlight the special point where errors don't add up
    plt.annotate('Special case:\nTotal error ≠ Sum of component errors', 
                xy=(evaluation_point, total_error), 
                xytext=(evaluation_point+0.5, 5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    # Add text box explaining the special case
    explanation_text = (
        "At x = 3, we have:\n"
        f"• Structural Error: {structural_error:.2f}\n"
        f"• Approximation Error: {approximation_error:.2f}\n"
        f"• Total Error: {total_error:.2f}\n"
        f"• Sum of Component Errors: {sum_of_errors:.2f}\n\n"
        "The errors don't add up at this point because the\n"
        "estimated model coincidentally gives the correct\n"
        "prediction, despite being a linear approximation\n"
        "of a quadratic function."
    )
    
    plt.text(0.5, 15, explanation_text, fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.title('Error Behavior Across Different x Values', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('Squared Error', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=10, loc='upper right')
    
    plt.tight_layout()
    
    # Save the figure
    file_path4 = os.path.join(save_dir, "error_behavior.png")
    plt.savefig(file_path4, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files.append(file_path4)
    
    return saved_files

# Create the visualizations
saved_files = create_visualizations()

print(f"\nVisualizations saved to: {', '.join(saved_files)}")
print("\nQuestion 14 Solution Summary:")
print(f"1. True value at x = 3: y = {true_value}")
print(f"2. Optimal linear model prediction: y_opt = {optimal_prediction}")
print(f"3. Estimated model prediction: ŷ = {estimated_prediction}")
print(f"4. Structural error: {structural_error}")
print(f"5. Approximation error: {approximation_error}")
print(f"6. Total squared error: {total_error}")
print(f"   Sum of structural and approximation errors: {sum_of_errors}")
print(f"   Verification: {np.isclose(total_error, sum_of_errors)}") 