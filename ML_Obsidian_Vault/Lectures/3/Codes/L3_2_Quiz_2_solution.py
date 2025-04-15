import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_2_Quiz_2")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Define the data from Question 2
study_hours = np.array([2, 3, 5, 7, 8])
exam_scores = np.array([65, 70, 85, 90, 95])

# Step 1: Calculate average study time and average exam score
def calculate_means(x, y):
    """Calculate the mean of x and y."""
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    print(f"Step 1: Calculate average study time (x̄) and average exam score (ȳ)")
    print(f"Mean of study hours (x̄): {x_mean} hours")
    print(f"Mean of exam scores (ȳ): {y_mean} points")
    print()
    
    return x_mean, y_mean

x_mean, y_mean = calculate_means(study_hours, exam_scores)

# Step 2: Calculate the covariance between study hours and exam scores
def calculate_covariance(x, y, x_mean, y_mean):
    """Calculate the covariance between x and y."""
    n = len(x)
    cov_sum = 0
    
    print(f"Step 2: Calculate the covariance between study hours and exam scores")
    print(f"Cov(x,y) = (1/n) * Σ[(x_i - x̄)(y_i - ȳ)]")
    print(f"Cov(x,y) = (1/{n}) * [", end="")
    
    for i in range(n):
        term = (x[i] - x_mean) * (y[i] - y_mean)
        cov_sum += term
        
        if i < n - 1:
            print(f"({x[i]} - {x_mean})({y[i]} - {y_mean}) + ", end="")
        else:
            print(f"({x[i]} - {x_mean})({y[i]} - {y_mean})]", end="")
    
    covariance = cov_sum / n
    print(f"\nCov(x,y) = {covariance}")
    print()
    
    return covariance

covariance = calculate_covariance(study_hours, exam_scores, x_mean, y_mean)

# Step 3: Calculate the variance of study hours
def calculate_variance(x, x_mean):
    """Calculate the variance of x."""
    n = len(x)
    var_sum = 0
    
    print(f"Step 3: Calculate the variance of study hours")
    print(f"Var(x) = (1/n) * Σ[(x_i - x̄)²]")
    print(f"Var(x) = (1/{n}) * [", end="")
    
    for i in range(n):
        term = (x[i] - x_mean) ** 2
        var_sum += term
        
        if i < n - 1:
            print(f"({x[i]} - {x_mean})² + ", end="")
        else:
            print(f"({x[i]} - {x_mean})²]", end="")
    
    variance = var_sum / n
    print(f"\nVar(x) = {variance}")
    print()
    
    return variance

variance = calculate_variance(study_hours, x_mean)

# Step 4: Calculate the slope and intercept
def calculate_slope_intercept(covariance, variance, x_mean, y_mean):
    """Calculate the slope and intercept using covariance and variance."""
    beta_1 = covariance / variance
    beta_0 = y_mean - beta_1 * x_mean
    
    print(f"Step 4: Calculate the slope (β₁) and intercept (β₀)")
    print(f"β₁ = Cov(x,y) / Var(x) = {covariance} / {variance} = {beta_1}")
    print(f"β₀ = ȳ - β₁·x̄ = {y_mean} - {beta_1} · {x_mean} = {beta_0}")
    print()
    
    return beta_0, beta_1

beta_0, beta_1 = calculate_slope_intercept(covariance, variance, x_mean, y_mean)

# Print the regression equation
print(f"Study Hour Predictor Equation: Score = {beta_0:.2f} + {beta_1:.2f} · Hours")
print()

# Calculate predicted scores and residuals
def calculate_predictions_residuals(x, y, beta_0, beta_1):
    """Calculate predictions and residuals."""
    # Calculate predicted values
    y_pred = beta_0 + beta_1 * x
    
    # Calculate residuals
    residuals = y - y_pred
    
    # Calculate RSS
    rss = np.sum(residuals ** 2)
    
    print("Predictions and Residuals:")
    print("Study Hours (x) | Exam Score (y) | Predicted Score (ŷ) | Residual (y - ŷ)")
    print("----------------------------------------------------------------")
    
    for i in range(len(x)):
        print(f"{x[i]:^13} | {y[i]:^13} | {y_pred[i]:^17.2f} | {residuals[i]:^14.2f}")
    
    print(f"\nResidual Sum of Squares (RSS) = {rss:.4f}")
    print()
    
    return y_pred, residuals, rss

y_pred, residuals, rss = calculate_predictions_residuals(study_hours, exam_scores, beta_0, beta_1)

# Create visualizations
def create_visualizations(x, y, x_mean, y_mean, y_pred, residuals, beta_0, beta_1, save_dir=None):
    """Create visualizations to help understand the regression analysis."""
    saved_files = []
    
    # Plot 1: Data points with means
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', s=100, label='Observed Data')
    plt.axhline(y=y_mean, color='green', linestyle='--', label=f'Mean Score: {y_mean:.2f}')
    plt.axvline(x=x_mean, color='red', linestyle='--', label=f'Mean Hours: {x_mean:.2f}')
    
    # Mark the mean point
    plt.scatter([x_mean], [y_mean], color='purple', s=150, marker='X', label='Mean Point')
    
    plt.xlabel('Study Hours', fontsize=12)
    plt.ylabel('Exam Score', fontsize=12)
    plt.title('Data with Mean Values', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot1_data_with_means.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 2: Covariance visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', s=100)
    plt.axhline(y=y_mean, color='green', linestyle='--')
    plt.axvline(x=x_mean, color='red', linestyle='--')
    
    # Create four quadrants
    for i in range(len(x)):
        # Draw lines to the point
        plt.plot([x_mean, x[i]], [y_mean, y_mean], 'k--', alpha=0.3)
        plt.plot([x[i], x[i]], [y_mean, y[i]], 'k--', alpha=0.3)
        
        # Color based on quadrant
        rect_color = 'green' if (x[i] - x_mean) * (y[i] - y_mean) > 0 else 'red'
        rect_alpha = 0.3
        
        # Create rectangle representing the product (x_i - x̄)(y_i - ȳ)
        width = abs(x[i] - x_mean)
        height = abs(y[i] - y_mean)
        
        rect_x = min(x_mean, x[i])
        rect_y = min(y_mean, y[i])
        
        plt.gca().add_patch(plt.Rectangle((rect_x, rect_y), width, height, 
                                    color=rect_color, alpha=rect_alpha))
        
        # Add the product value as text
        product = (x[i] - x_mean) * (y[i] - y_mean)
        plt.annotate(f"{product:.1f}", 
                   xy=(x[i], y[i]), 
                   xytext=(5, 0), 
                   textcoords='offset points',
                   fontsize=10)
    
    plt.xlabel('Study Hours', fontsize=12)
    plt.ylabel('Exam Score', fontsize=12)
    plt.title('Covariance Visualization', fontsize=14)
    plt.annotate('Positive\nContribution', xy=(6, 75), fontsize=10, 
                ha='center', color='green', fontweight='bold')
    plt.annotate('Negative\nContribution', xy=(3, 90), fontsize=10, 
                ha='center', color='red', fontweight='bold')
    plt.grid(True)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot2_covariance_visualization.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 3: Scatter plot with regression line
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', s=100, label='Observed Data')
    
    # Generate points for the regression line
    x_line = np.linspace(min(x) - 1, max(x) + 1, 100)
    y_line = beta_0 + beta_1 * x_line
    
    plt.plot(x_line, y_line, color='red', linestyle='-', linewidth=2, 
            label=f'Regression Line\ny = {beta_0:.2f} + {beta_1:.2f}x')
    
    plt.xlabel('Study Hours', fontsize=12)
    plt.ylabel('Exam Score', fontsize=12)
    plt.title('Linear Regression: Exam Score vs Study Hours', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot3_regression_line.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 4: Residuals plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, residuals, color='purple', s=100)
    plt.axhline(y=0, color='red', linestyle='-', alpha=0.7)
    
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [0, residuals[i]], 'purple', linestyle='--', alpha=0.5)
    
    plt.xlabel('Study Hours', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.title('Residuals Plot', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot4_residuals.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    # Plot 5: Study hour predictor (interactive visualization)
    plt.figure(figsize=(10, 6))
    
    # Create a more detailed line for the study hour predictor
    study_hours_range = np.linspace(0, 12, 100)
    predicted_scores = beta_0 + beta_1 * study_hours_range
    
    plt.plot(study_hours_range, predicted_scores, 'r-', linewidth=3, 
            label='Study Hour Predictor')
    plt.scatter(x, y, color='blue', s=100, label='Training Data')
    
    # Highlight specific predictions for new students
    new_hours = np.array([4, 6, 9])
    new_scores = beta_0 + beta_1 * new_hours
    
    plt.scatter(new_hours, new_scores, color='green', s=120, 
               label='Predictions for New Students')
    
    # Add annotations for new predictions
    for i, (hour, score) in enumerate(zip(new_hours, new_scores)):
        plt.annotate(f"{hour} hours → {score:.1f} points", 
                   xy=(hour, score), 
                   xytext=(10, (-1)**i * 15), 
                   textcoords='offset points',
                   fontsize=12,
                   arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    
    plt.xlabel('Study Hours', fontsize=14)
    plt.ylabel('Predicted Exam Score', fontsize=14)
    plt.title('Study Hour Predictor for Future Students', fontsize=16)
    plt.grid(True)
    plt.legend(loc='lower right', fontsize=12)
    
    # Add formula to the plot
    formula_text = f"Score = {beta_0:.2f} + {beta_1:.2f} × Hours"
    plt.text(0.05, 0.95, formula_text, transform=plt.gca().transAxes, 
            fontsize=14, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot5_study_hour_predictor.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    plt.close()
    
    return saved_files

# Create visualizations
saved_files = create_visualizations(study_hours, exam_scores, x_mean, y_mean, y_pred, residuals, beta_0, beta_1, save_dir)

print(f"Visualizations saved to: {', '.join(saved_files)}")
print("\nQuestion 2 Solution Summary:")
print(f"1. Mean study time: x̄ = {x_mean} hours")
print(f"2. Mean exam score: ȳ = {y_mean} points")
print(f"3. Covariance between study hours and exam scores: Cov(x,y) = {covariance:.2f}")
print(f"4. Variance of study hours: Var(x) = {variance:.2f}")
print(f"5. Regression coefficients: β₀ = {beta_0:.2f}, β₁ = {beta_1:.2f}")
print(f"6. Study Hour Predictor: Score = {beta_0:.2f} + {beta_1:.2f} × Hours") 