import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_2_Quiz_11")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Define the data from Question 11
students = ['A', 'B', 'C', 'D', 'E', 'F']
study_hours = np.array([2, 3, 5, 7, 8, 10])
exam_scores = np.array([50, 60, 70, 80, 90, 95])

# Professor Andrew's formula: Score = 40 + 5.5 × (Study Hours)
print("Question 11: Study Hours and Exam Scores Prediction")
print("====================================================")
print("Dataset:")
print("Students:", students)
print("Study Hours:", study_hours)
print("Exam Scores:", exam_scores)
print("\nProfessor Andrew's formula: Score = 40 + 5.5 × (Study Hours)")
print()

# Step 1: Calculate predicted scores using Professor Andrew's formula
def calculate_predicted_scores(hours, formula_intercept=40, formula_slope=5.5):
    """Calculate predicted exam scores using the given formula."""
    return formula_intercept + formula_slope * hours

# Calculate predictions
predicted_scores = calculate_predicted_scores(study_hours)

print("Step 1: Calculate the predicted score for each student")
print("Student | Study Hours | Actual Score | Predicted Score")
print("--------------------------------------------------")
for i in range(len(students)):
    print(f"{students[i]:^8} | {study_hours[i]:^11} | {exam_scores[i]:^12} | {predicted_scores[i]:^15.1f}")
print()

# Step 2: Calculate residuals
def calculate_residuals(actual, predicted):
    """Calculate residuals (the difference between actual and predicted values)."""
    return actual - predicted

# Calculate residuals
residuals = calculate_residuals(exam_scores, predicted_scores)

print("Step 2: Calculate the residual for each student")
print("Student | Actual Score | Predicted Score | Residual")
print("--------------------------------------------------")
for i in range(len(students)):
    print(f"{students[i]:^8} | {exam_scores[i]:^12} | {predicted_scores[i]:^15.1f} | {residuals[i]:^8.1f}")
print()

# Step 3: Calculate Residual Sum of Squares (RSS)
rss = np.sum(residuals ** 2)

print("Step 3: Calculate the Residual Sum of Squares (RSS)")
print("RSS = Sum of squared residuals")
print("RSS = " + " + ".join([f"({r:.1f})²" for r in residuals]))
print(f"RSS = {rss:.1f}")
print()

# Step 4: Predict score for a new student who studies for 6 hours
new_hours = 6
new_predicted_score = calculate_predicted_scores(new_hours)

print(f"Step 4: Predict the exam score for a new student who studies for {new_hours} hours")
print(f"Predicted Score = 40 + 5.5 × {new_hours} = {new_predicted_score:.1f}")
print()

# Compare with alternative linear regression model
def calculate_least_squares(x, y):
    """Calculate the least squares estimates for slope and intercept."""
    n = len(x)
    
    # Calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Calculate slope
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    slope = numerator / denominator
    
    # Calculate intercept
    intercept = y_mean - slope * x_mean
    
    return intercept, slope

# Calculate least squares estimates
ls_intercept, ls_slope = calculate_least_squares(study_hours, exam_scores)
ls_predicted_scores = ls_intercept + ls_slope * study_hours
ls_residuals = exam_scores - ls_predicted_scores
ls_rss = np.sum(ls_residuals ** 2)
ls_new_predicted = ls_intercept + ls_slope * new_hours

print("For comparison, here's the least squares regression model:")
print(f"Score = {ls_intercept:.2f} + {ls_slope:.2f} × (Study Hours)")
print(f"RSS = {ls_rss:.2f}")
print(f"Predicted score for {new_hours} hours: {ls_new_predicted:.2f}")
print()

# Create visualizations
def create_visualizations(x, y, formula_predicted, ls_predicted, residuals, formula_intercept, formula_slope, 
                         ls_intercept, ls_slope, new_x, new_y, save_dir=None):
    """Create visualizations to help understand the regression analysis."""
    saved_files = []
    
    # Plot 1: Scatter plot with Professor Andrew's line and LS line
    plt.figure(figsize=(12, 7))
    
    # Plot the data points
    plt.scatter(x, y, color='blue', s=100, label='Student Data')
    
    # Generate points for the lines
    x_line = np.linspace(min(x) - 1, max(x) + 1, 100)
    formula_y_line = formula_intercept + formula_slope * x_line
    ls_y_line = ls_intercept + ls_slope * x_line
    
    # Plot Professor Andrew's formula
    plt.plot(x_line, formula_y_line, color='red', linestyle='-', linewidth=2,
             label=f'Professor Andrew: y = {formula_intercept} + {formula_slope}x')
    
    # Plot least squares line
    plt.plot(x_line, ls_y_line, color='green', linestyle='--', linewidth=2,
             label=f'Least Squares: y = {ls_intercept:.2f} + {ls_slope:.2f}x')
    
    # Add student labels
    for i in range(len(x)):
        plt.annotate(f"Student {students[i]}", 
                    (x[i], y[i]), 
                    xytext=(5, 10), 
                    textcoords='offset points')
    
    # Mark the new prediction point
    plt.scatter([new_x], [new_y], color='purple', s=120, marker='*',
                label=f'New Student (6 hrs): {new_y:.1f}')
    
    plt.xlabel('Study Hours', fontsize=12)
    plt.ylabel('Exam Score', fontsize=12)
    plt.title('Exam Scores vs. Study Hours with Prediction Models', fontsize=14)
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot1_regression_lines.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 2: Residuals plot for Professor Andrew's formula
    plt.figure(figsize=(12, 7))
    
    # Plot the reference line at y=0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Plot the residuals as a scatter plot
    plt.scatter(x, residuals, color='red', s=100, label='Residuals (Andrew)')
    
    # Connect residuals to zero line
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [0, residuals[i]], 'r--', alpha=0.5)
        plt.annotate(f"{residuals[i]:.1f}", 
                    (x[i], residuals[i]), 
                    xytext=(5, 0 if residuals[i] >= 0 else -15), 
                    textcoords='offset points')
    
    # Add student labels
    for i in range(len(x)):
        plt.annotate(f"Student {students[i]}", 
                    (x[i], 0), 
                    xytext=(5, -10 if residuals[i] >= 0 else 10), 
                    textcoords='offset points',
                    ha='left', va='center')
    
    plt.xlabel('Study Hours', fontsize=12)
    plt.ylabel('Residual (Actual - Predicted)', fontsize=12)
    plt.title('Residuals for Professor Andrew\'s Formula', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot2_residuals.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 3: Actual vs. Predicted values for Professor Andrew's formula
    plt.figure(figsize=(10, 8))
    
    # Plot the perfect prediction line
    min_val = min(min(y), min(formula_predicted)) - 5
    max_val = max(max(y), max(formula_predicted)) + 5
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
    
    # Plot the predicted vs. actual
    plt.scatter(formula_predicted, y, color='blue', s=100, label='Andrew\'s Predictions')
    
    # Add student labels
    for i in range(len(x)):
        plt.annotate(f"Student {students[i]}", 
                    (formula_predicted[i], y[i]), 
                    xytext=(5, 5), 
                    textcoords='offset points')
    
    plt.xlabel('Predicted Score', fontsize=12)
    plt.ylabel('Actual Score', fontsize=12)
    plt.title('Actual vs. Predicted Exam Scores (Andrew\'s Formula)', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot3_actual_vs_predicted.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 4: Squared residuals visualization
    plt.figure(figsize=(12, 7))
    
    # Calculate squared residuals
    squared_residuals = residuals ** 2
    
    # Create bar chart
    plt.bar(students, squared_residuals, color='orange', alpha=0.7, width=0.6)
    
    # Add values on top of bars
    for i in range(len(students)):
        plt.text(i, squared_residuals[i] + 1, f"{squared_residuals[i]:.1f}", 
                ha='center', va='bottom', fontsize=10)
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    plt.xlabel('Student', fontsize=12)
    plt.ylabel('Squared Residual', fontsize=12)
    plt.title(f'Squared Residuals for Each Student (RSS = {rss:.1f})', fontsize=14)
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot4_squared_residuals.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 5: Comparison of predictions and residuals between the two models
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[2, 1])
    
    # Top left: Predictions
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x_line, formula_y_line, color='red', linestyle='-', linewidth=2,
            label=f'Andrew: y = {formula_intercept} + {formula_slope}x')
    ax1.plot(x_line, ls_y_line, color='green', linestyle='--', linewidth=2,
            label=f'LS: y = {ls_intercept:.2f} + {ls_slope:.2f}x')
    ax1.scatter(x, y, color='blue', s=80, label='Student Data')
    ax1.scatter([new_x], [new_y], color='purple', s=100, marker='*',
               label=f'New (Andrew): {new_y:.1f}')
    ax1.scatter([new_x], [ls_new_predicted], color='orange', s=100, marker='*',
               label=f'New (LS): {ls_new_predicted:.1f}')
    ax1.set_xlabel('Study Hours', fontsize=12)
    ax1.set_ylabel('Exam Score', fontsize=12)
    ax1.set_title('Prediction Models Comparison', fontsize=14)
    ax1.grid(True)
    ax1.legend(loc='lower right')
    
    # Top right: Actual vs. Predicted
    ax2 = fig.add_subplot(gs[0, 1])
    min_val = min(min(y), min(formula_predicted), min(ls_predicted)) - 5
    max_val = max(max(y), max(formula_predicted), max(ls_predicted)) + 5
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
    ax2.scatter(formula_predicted, y, color='red', s=80, label='Andrew\'s')
    ax2.scatter(ls_predicted, y, color='green', s=80, marker='s', label='LS')
    ax2.set_xlabel('Predicted Score', fontsize=12)
    ax2.set_ylabel('Actual Score', fontsize=12)
    ax2.set_title('Actual vs. Predicted Scores', fontsize=14)
    ax2.grid(True)
    ax2.legend()
    
    # Bottom left: Residuals comparison
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.scatter(x, residuals, color='red', s=80, label=f'Andrew\'s (RSS = {rss:.1f})')
    ax3.scatter(x, ls_residuals, color='green', s=80, marker='s', label=f'LS (RSS = {ls_rss:.1f})')
    ax3.set_xlabel('Study Hours', fontsize=12)
    ax3.set_ylabel('Residual', fontsize=12)
    ax3.set_title('Residuals Comparison', fontsize=14)
    ax3.grid(True)
    ax3.legend()
    
    # Bottom right: RSS comparison
    ax4 = fig.add_subplot(gs[1, 1])
    models = ['Professor Andrew', 'Least Squares']
    rss_values = [rss, ls_rss]
    ax4.bar(models, rss_values, color=['red', 'green'], alpha=0.7, width=0.6)
    ax4.set_ylabel('Residual Sum of Squares (RSS)', fontsize=12)
    ax4.set_title('RSS Comparison', fontsize=14)
    ax4.grid(True, axis='y')
    
    # Add values on top of bars
    for i in range(len(models)):
        ax4.text(i, rss_values[i] + 5, f"{rss_values[i]:.1f}", 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot5_model_comparison.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    return saved_files

# Execute the visualizations
saved_files = create_visualizations(
    study_hours, exam_scores, predicted_scores, ls_predicted_scores, 
    residuals, 40, 5.5, ls_intercept, ls_slope, 
    new_hours, new_predicted_score, save_dir
)

# Print summary
print("\nQuestion 11 Solution Summary:")
print(f"1. Using Professor Andrew's formula (Score = 40 + 5.5 × Hours):")
print(f"   - Predicted scores: {', '.join([f'{s:.1f}' for s in predicted_scores])}")
print(f"   - Residuals: {', '.join([f'{r:.1f}' for r in residuals])}")
print(f"   - Residual Sum of Squares (RSS): {rss:.1f}")
print(f"   - Predicted score for 6 hours of study: {new_predicted_score:.1f}")
print(f"\n2. For comparison, the least squares model (Score = {ls_intercept:.2f} + {ls_slope:.2f} × Hours):")
print(f"   - Residual Sum of Squares (RSS): {ls_rss:.2f}")
print(f"   - Predicted score for 6 hours of study: {ls_new_predicted:.2f}")
print("\nPlots have been saved to:", save_dir) 