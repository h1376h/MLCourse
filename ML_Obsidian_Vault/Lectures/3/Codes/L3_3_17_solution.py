import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures if it doesn't exist
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_3_Quiz_17")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Define the models
def true_model(x):
    """The true relationship: f(x) = 0.5x^2"""
    return 0.5 * x**2

def optimal_linear_model(x):
    """The optimal linear approximation: y_hat = 2 + 3x"""
    return 2 + 3 * x

def estimated_linear_model(x):
    """The estimated linear model with limited data: y_hat = 1 + 3.5x"""
    return 1 + 3.5 * x

def calculate_errors(x):
    """Calculate all error components for a given x value"""
    true_value = true_model(x)
    optimal_prediction = optimal_linear_model(x)
    estimated_prediction = estimated_linear_model(x)
    
    # Calculate error components
    structural_error = true_value - optimal_prediction
    approximation_error = optimal_prediction - estimated_prediction
    total_error = true_value - estimated_prediction
    
    # Calculate squared errors
    structural_error_squared = structural_error**2
    approximation_error_squared = approximation_error**2
    total_error_squared = total_error**2
    
    return {
        'x': x,
        'true_value': true_value,
        'optimal_prediction': optimal_prediction,
        'estimated_prediction': estimated_prediction,
        'structural_error': structural_error,
        'approximation_error': approximation_error,
        'total_error': total_error,
        'structural_error_squared': structural_error_squared,
        'approximation_error_squared': approximation_error_squared,
        'total_error_squared': total_error_squared
    }

# Task 1: Calculate errors for a user with 5 connections
print("Task 1: Errors for a user with 5 connections")
result_5 = calculate_errors(5)

print(f"True expected engagement: {result_5['true_value']} hours")
print(f"Prediction from optimal linear model: {result_5['optimal_prediction']} hours")
print(f"Prediction from estimated model: {result_5['estimated_prediction']} hours")
print(f"Structural error: {result_5['structural_error']} hours")
print(f"Approximation error: {result_5['approximation_error']} hours")
print(f"Total error: {result_5['total_error']} hours")
print(f"Structural error squared: {result_5['structural_error_squared']} hours²")
print(f"Approximation error squared: {result_5['approximation_error_squared']} hours²")
print(f"Total error squared: {result_5['total_error_squared']} hours²")

# Verify error decomposition
error_sum = result_5['structural_error_squared'] + result_5['approximation_error_squared']
decomposition_term = 2 * result_5['structural_error'] * result_5['approximation_error']

print(f"\nVerification of error decomposition:")
print(f"Sum of squared errors: {error_sum} hours²")
print(f"Total squared error: {result_5['total_error_squared']} hours²")
print(f"Decomposition cross-term: {decomposition_term} hours²")
print(f"Sum of squared errors + cross-term: {error_sum + decomposition_term} hours²")

if abs(result_5['total_error_squared'] - (error_sum + decomposition_term)) < 1e-10:
    print("Verified: Total squared error equals sum of squared errors plus the cross-term")
    print("Note: For error decomposition to be additive, the cross-term should be zero,")
    print("which happens when errors are uncorrelated.")

# Task 2: Calculate errors for users with 4, 6, and 8 connections
print("\nTask 2: Error components for users with 4, 6, and 8 connections")
connections = [4, 6, 8]
results = {x: calculate_errors(x) for x in connections}

for x in connections:
    result = results[x]
    total_squared = result['total_error_squared']
    structural_squared = result['structural_error_squared']
    approximation_squared = result['approximation_error_squared']
    
    # Calculate percentages
    structural_percentage = (structural_squared / total_squared) * 100
    approximation_percentage = (approximation_squared / total_squared) * 100
    
    # Determine which error component contributes more
    dominant_error = "Structural" if structural_squared > approximation_squared else "Approximation"
    difference_percentage = abs(structural_percentage - approximation_percentage)
    
    print(f"\nFor a user with {x} connections:")
    print(f"Structural error squared: {structural_squared:.2f} hours² ({structural_percentage:.2f}%)")
    print(f"Approximation error squared: {approximation_squared:.2f} hours² ({approximation_percentage:.2f}%)")
    print(f"Total squared error: {total_squared:.2f} hours²")
    print(f"Dominant error component: {dominant_error} by {difference_percentage:.2f}%")

# Task 3: Determine if collecting more data or using a non-linear model would be better
print("\nTask 3: Strategy to reduce total error below 5 hours²")

# Generate data for a range of x values
x_range = np.linspace(0, 10, 100)
structural_errors = np.array([calculate_errors(x)['structural_error_squared'] for x in x_range])
approximation_errors = np.array([calculate_errors(x)['approximation_error_squared'] for x in x_range])

# Collecting more data can only reduce approximation error, not structural error
# The maximum possible reduction is to eliminate the approximation error completely
reduced_errors = structural_errors  # If approximation error becomes 0

# Find range where even with perfect estimation, error would still be above 5
above_threshold_with_perfect_estimation = x_range[reduced_errors > 5]

if len(above_threshold_with_perfect_estimation) > 0:
    min_x_above_threshold = min(above_threshold_with_perfect_estimation)
    print(f"Even with perfect estimation (no approximation error), for x ≥ {min_x_above_threshold:.2f},")
    print(f"the structural error alone would exceed 5 hours². This means for these values,")
    print(f"collecting more data is insufficient - a non-linear model is required.")
    
    # Calculate the x value where structural error equals 5
    # Solving 0.5x^2 - (2 + 3x) = ±√5
    # This is a quadratic equation: 0.5x^2 - 3x - 2 = ±√5
    # We need to solve for both the positive and negative cases
    
    # For the positive case:
    # 0.5x^2 - 3x - 2 - √5 = 0
    # For the negative case:
    # 0.5x^2 - 3x - 2 + √5 = 0
    
    sqrt_5 = np.sqrt(5)
    
    # Using the quadratic formula: x = (-b ± √(b^2 - 4ac)) / 2a
    a = 0.5
    b = -3
    
    # Positive case
    c_pos = -2 - sqrt_5
    discriminant_pos = b**2 - 4*a*c_pos
    if discriminant_pos >= 0:
        x1_pos = (-b + np.sqrt(discriminant_pos)) / (2*a)
        x2_pos = (-b - np.sqrt(discriminant_pos)) / (2*a)
        x_pos_roots = [x for x in [x1_pos, x2_pos] if x >= 0]
    else:
        x_pos_roots = []
    
    # Negative case
    c_neg = -2 + sqrt_5
    discriminant_neg = b**2 - 4*a*c_neg
    if discriminant_neg >= 0:
        x1_neg = (-b + np.sqrt(discriminant_neg)) / (2*a)
        x2_neg = (-b - np.sqrt(discriminant_neg)) / (2*a)
        x_neg_roots = [x for x in [x1_neg, x2_neg] if x >= 0]
    else:
        x_neg_roots = []
    
    # Combine and sort the roots
    all_roots = sorted(x_pos_roots + x_neg_roots)
    
    print(f"By solving the equation where structural error squared equals 5,")
    print(f"we find that for x values outside the range [{all_roots[0]:.2f}, {all_roots[1]:.2f}],")
    print(f"the structural error squared exceeds 5 hours².")
    
    # Calculate when total error exceeds 5 with current models
    total_errors = np.array([calculate_errors(x)['total_error_squared'] for x in x_range])
    above_threshold_current = x_range[total_errors > 5]
    
    if len(above_threshold_current) > 0:
        print(f"\nWith the current estimated model, total squared error exceeds 5 hours²")
        print(f"for connections outside the range [{min(above_threshold_current):.2f}, {max(above_threshold_current):.2f}].")
        
        # Calculate the range where collecting more data could help
        could_help = []
        for x in np.linspace(0, 10, 100):
            result = calculate_errors(x)
            # If total error > 5 but structural error < 5, collecting more data could help
            if result['total_error_squared'] > 5 and result['structural_error_squared'] < 5:
                could_help.append(x)
        
        if len(could_help) > 0:
            print(f"For connections in approximately [{min(could_help):.2f}, {max(could_help):.2f}],")
            print(f"collecting more data could potentially reduce error below 5 hours².")
        else:
            print(f"There are no ranges where collecting more data alone could reduce error below 5 hours².")
else:
    print("The structural error never exceeds 5 hours² in the examined range.")
    print("This means collecting more data could theoretically reduce total error below 5 hours².")

# Create visualizations

# Plot 1: Model comparison
plt.figure(figsize=(10, 6))
x_plot = np.linspace(0, 10, 100)
plt.plot(x_plot, true_model(x_plot), 'r-', linewidth=2, label='True Model: f(x) = 0.5x²')
plt.plot(x_plot, optimal_linear_model(x_plot), 'g--', linewidth=2, label='Optimal Linear Model: y = 2 + 3x')
plt.plot(x_plot, estimated_linear_model(x_plot), 'b:', linewidth=2, label='Estimated Linear Model: y = 1 + 3.5x')

plt.title('Comparison of Models', fontsize=14)
plt.xlabel('Number of Connections', fontsize=12)
plt.ylabel('Engagement Hours', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()

# Mark the special points: x = 4, 5, 6, 8
special_points = [4, 5, 6, 8]
for x in special_points:
    y_true = true_model(x)
    y_optimal = optimal_linear_model(x)
    y_estimated = estimated_linear_model(x)
    
    plt.plot([x, x], [y_estimated, y_true], 'k-', alpha=0.5)
    plt.scatter([x], [y_true], c='red', s=50, zorder=5)
    plt.scatter([x], [y_optimal], c='green', s=50, zorder=5)
    plt.scatter([x], [y_estimated], c='blue', s=50, zorder=5)

plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')

# Plot 2: Error components
plt.figure(figsize=(10, 6))
x_plot = np.linspace(0, 10, 100)

structural_errors = [calculate_errors(x)['structural_error_squared'] for x in x_plot]
approximation_errors = [calculate_errors(x)['approximation_error_squared'] for x in x_plot]
total_errors = [calculate_errors(x)['total_error_squared'] for x in x_plot]

plt.plot(x_plot, structural_errors, 'r-', linewidth=2, label='Structural Error²')
plt.plot(x_plot, approximation_errors, 'b-', linewidth=2, label='Approximation Error²')
plt.plot(x_plot, total_errors, 'k--', linewidth=2, label='Total Error²')

# Add a horizontal line at y = 5
plt.axhline(y=5, color='green', linestyle='--', alpha=0.7, label='Error Threshold = 5')

plt.title('Error Components by Number of Connections', fontsize=14)
plt.xlabel('Number of Connections', fontsize=12)
plt.ylabel('Squared Error (hours²)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()

# Mark the special points
for x in special_points:
    result = calculate_errors(x)
    plt.scatter([x], [result['structural_error_squared']], c='red', s=50, zorder=5)
    plt.scatter([x], [result['approximation_error_squared']], c='blue', s=50, zorder=5)
    plt.scatter([x], [result['total_error_squared']], c='black', s=50, zorder=5)
    
plt.savefig(os.path.join(save_dir, 'error_components.png'), dpi=300, bbox_inches='tight')

# Plot 3: Error decomposition for x = 5
plt.figure(figsize=(10, 6))

# Create a bar chart for the error components
errors_5 = [result_5['structural_error_squared'], result_5['approximation_error_squared'], 
           result_5['total_error_squared'], error_sum, result_5['total_error_squared']]
labels = ['Structural Error²', 'Approximation Error²', 'Total Error²', 
         'Sum of Squared Errors', 'Total Error² (Verification)']
colors = ['red', 'blue', 'black', 'purple', 'gray']

plt.bar(labels, errors_5, color=colors, alpha=0.7)
plt.axhline(y=5, color='green', linestyle='--', alpha=0.7, label='Error Threshold = 5')

plt.title('Error Decomposition for a User with 5 Connections', fontsize=14)
plt.ylabel('Squared Error (hours²)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, axis='y')
plt.tight_layout()

# Add value labels on the bars
for i, v in enumerate(errors_5):
    plt.text(i, v + 0.1, f'{v:.2f}', ha='center', fontsize=10)

plt.savefig(os.path.join(save_dir, 'error_decomposition_x5.png'), dpi=300, bbox_inches='tight')

# Plot 4: Structural vs Approximation error percentage for specified points
plt.figure(figsize=(10, 6))

special_points = [4, 5, 6, 8]
structural_percentages = []
approximation_percentages = []

for x in special_points:
    result = calculate_errors(x)
    total_squared = result['total_error_squared']
    structural_squared = result['structural_error_squared']
    approximation_squared = result['approximation_error_squared']
    
    structural_percentage = (structural_squared / total_squared) * 100
    approximation_percentage = (approximation_squared / total_squared) * 100
    
    structural_percentages.append(structural_percentage)
    approximation_percentages.append(approximation_percentage)

x_pos = np.arange(len(special_points))
width = 0.35

plt.bar(x_pos - width/2, structural_percentages, width, label='Structural Error', color='red', alpha=0.7)
plt.bar(x_pos + width/2, approximation_percentages, width, label='Approximation Error', color='blue', alpha=0.7)

plt.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='50% Threshold')

plt.title('Error Component Percentage by Number of Connections', fontsize=14)
plt.xlabel('Number of Connections', fontsize=12)
plt.ylabel('Percentage of Total Error (%)', fontsize=12)
plt.xticks(x_pos, special_points)
plt.legend(fontsize=10)
plt.grid(True, axis='y')
plt.tight_layout()

# Add value labels on the bars
for i, v in enumerate(structural_percentages):
    plt.text(i - width/2, v + 1, f'{v:.1f}%', ha='center', fontsize=9)
for i, v in enumerate(approximation_percentages):
    plt.text(i + width/2, v + 1, f'{v:.1f}%', ha='center', fontsize=9)

plt.savefig(os.path.join(save_dir, 'error_percentage.png'), dpi=300, bbox_inches='tight')

print(f"\nVisualizations saved to: {save_dir}") 