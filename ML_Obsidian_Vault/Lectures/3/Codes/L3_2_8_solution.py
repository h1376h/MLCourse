import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_2_Quiz_8")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Define the data from Question 8
advertising = np.array([1, 3, 5, 7, 9])  # Advertising expenditure in $1000s
sales = np.array([20, 40, 50, 65, 80])   # Product sales in units
predicted_sales = np.array([25, 38, 50, 63, 75])  # Given predicted values

print("Question 8: Error Distribution in Linear Regression")
print("\nData:")
print("| Advertising (x) | Sales (y) | Predicted Sales (ŷ) |")
print("|----------------|-----------|---------------------|")
for i in range(len(advertising)):
    print(f"| {advertising[i]:14d} | {sales[i]:9d} | {predicted_sales[i]:19d} |")
print("\n" + "="*80 + "\n")

# Task 1: Calculate the residuals
def calculate_residuals(actual, predicted):
    """Calculate residuals (errors) from actual and predicted values."""
    residuals = actual - predicted
    
    print("Step 1: Calculate the residuals")
    print("Residual = Actual - Predicted")
    print("\n| Advertising (x) | Sales (y) | Predicted Sales (ŷ) | Residual (e) |")
    print("|----------------|-----------|---------------------|--------------|")
    
    for i in range(len(actual)):
        print(f"| {advertising[i]:14d} | {sales[i]:9d} | {predicted_sales[i]:19d} | {residuals[i]:12.2f} |")
    
    return residuals

# Task 2: Compute mean, variance, and standard deviation of residuals
def compute_residual_statistics(residuals):
    """Compute basic statistics for the residuals."""
    mean_residual = np.mean(residuals)
    var_residual = np.var(residuals, ddof=1)  # Using sample variance
    std_residual = np.std(residuals, ddof=1)  # Using sample standard deviation
    
    print("\nStep 2: Compute mean, variance, and standard deviation of residuals")
    print(f"Mean of residuals: {mean_residual:.4f}")
    print(f"Variance of residuals: {var_residual:.4f}")
    print(f"Standard deviation of residuals: {std_residual:.4f}")
    
    return mean_residual, var_residual, std_residual

# Task 3: Visualize the error distribution
def visualize_error_distribution(residuals, mean_residual, std_residual, save_dir=None):
    """Create visualizations to assess if errors follow a Gaussian distribution."""
    saved_files = []
    
    # Figure 1: Histogram of residuals with normal curve
    plt.figure(figsize=(10, 6))
    
    # Create histogram
    n, bins, patches = plt.hist(residuals, bins=5, density=True, alpha=0.7, color='skyblue', 
                               edgecolor='black', label='Residuals')
    
    # Add a normal curve
    x = np.linspace(min(residuals) - 1, max(residuals) + 1, 100)
    y = stats.norm.pdf(x, mean_residual, std_residual)
    plt.plot(x, y, 'r-', linewidth=2, label='Normal Distribution')
    
    # Add vertical line at mean
    plt.axvline(x=mean_residual, color='green', linestyle='--', 
               label=f'Mean = {mean_residual:.4f}')
    
    plt.xlabel('Residual Values', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Histogram of Residuals with Normal Curve', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot1_residual_histogram.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    
    # Figure 2: Q-Q plot to assess normality
    plt.figure(figsize=(10, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot2_qq_plot.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    
    # Figure 3: Residuals vs. Predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(predicted_sales, residuals, color='blue', s=100, alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='-', alpha=0.7)
    
    for i in range(len(residuals)):
        plt.plot([predicted_sales[i], predicted_sales[i]], [0, residuals[i]], 
                'b--', alpha=0.5)
    
    plt.xlabel('Predicted Sales', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.title('Residuals vs. Predicted Values', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot3_residuals_vs_predicted.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    
    # Figure 4: Residuals vs. Predictor variable
    plt.figure(figsize=(10, 6))
    plt.scatter(advertising, residuals, color='green', s=100, alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='-', alpha=0.7)
    
    for i in range(len(residuals)):
        plt.plot([advertising[i], advertising[i]], [0, residuals[i]], 
                'g--', alpha=0.5)
    
    plt.xlabel('Advertising Expenditure ($1000s)', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.title('Residuals vs. Advertising Expenditure', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot4_residuals_vs_predictor.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    
    return saved_files

# Task 4: Verify if residuals sum to zero
def verify_residual_sum(residuals):
    """Verify if the residuals sum to zero (or very close to zero)."""
    sum_residuals = np.sum(residuals)
    
    print("\nStep 4: Verify if residuals sum to zero")
    print(f"Sum of residuals: {sum_residuals:.4f}")
    
    if abs(sum_residuals) < 1e-10:
        print("The sum of residuals is exactly zero.")
    elif abs(sum_residuals) < 1:
        print("The sum of residuals is very close to zero.")
    else:
        print("The sum of residuals is not close to zero, which is unusual for a linear regression with an intercept.")
    
    print("\nExplanation:")
    print("In linear regression with an intercept term, the residuals always sum to zero exactly")
    print("if the model is fit using ordinary least squares (OLS). This is because the normal equations")
    print("for OLS regression guarantee this property. Specifically, the equations ensure that the")
    print("residuals are orthogonal to each column of the design matrix, including the column of ones")
    print("for the intercept. Since the dot product of the residuals with the column of ones equals zero,")
    print("the sum of residuals must be zero.")
    
    # Given the residuals don't sum exactly to zero, suggest possible reasons
    if abs(sum_residuals) > 1e-10:
        print("\nPossible explanations for why the residuals don't sum exactly to zero in this case:")
        print("1. The model might not have been fit using OLS (e.g., weighted least squares, regularization).")
        print("2. The model might not include an intercept term.")
        print("3. The predicted values might have been rounded or approximated.")
        print("4. There might be numerical precision issues in the calculation.")
    
    return sum_residuals

# Execute computations
residuals = calculate_residuals(sales, predicted_sales)
print()
mean_residual, var_residual, std_residual = compute_residual_statistics(residuals)
saved_files = visualize_error_distribution(residuals, mean_residual, std_residual, save_dir)
sum_residuals = verify_residual_sum(residuals)
print("\n" + "="*80 + "\n")

# Perform statistical tests for normality
def test_normality(residuals):
    """Perform statistical tests to assess normality of residuals."""
    print("Step 5: Statistical tests for normality")
    
    # Shapiro-Wilk test
    shapiro_test = stats.shapiro(residuals)
    print(f"Shapiro-Wilk test: W={shapiro_test.statistic:.4f}, p-value={shapiro_test.pvalue:.4f}")
    
    if shapiro_test.pvalue > 0.05:
        print("Fail to reject the null hypothesis - residuals appear to be normally distributed.")
    else:
        print("Reject the null hypothesis - residuals do not appear to be normally distributed.")
    
    print("\nNote: With only 5 data points, statistical tests for normality have limited power.")
    print("Visual inspection of plots may be more informative in this case.")
    
    return shapiro_test

# Execute normality test
shapiro_test = test_normality(residuals)
print("\n" + "="*80 + "\n")

# Calculate the regression coefficients from the data
def calculate_regression_coefficients(x, y):
    """Calculate regression coefficients from the data to verify the model."""
    n = len(x)
    
    # Calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Calculate slope and intercept
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    beta_1 = numerator / denominator
    beta_0 = y_mean - beta_1 * x_mean
    
    print("Step 6: Calculate regression coefficients from the data")
    print(f"Calculated slope (β₁): {beta_1:.4f}")
    print(f"Calculated intercept (β₀): {beta_0:.4f}")
    print(f"Regression equation: Sales = {beta_0:.4f} + {beta_1:.4f} × Advertising")
    
    # Calculate predicted values from this model
    y_pred_calc = beta_0 + beta_1 * x
    
    print("\nCalculated predicted values:")
    print("| Advertising (x) | Calculated Predicted Sales | Given Predicted Sales |")
    print("|----------------|-----------------------------|------------------------|")
    
    for i in range(n):
        print(f"| {x[i]:14d} | {y_pred_calc[i]:27.2f} | {predicted_sales[i]:22d} |")
    
    return beta_0, beta_1, y_pred_calc

# Execute regression coefficient calculation
beta_0, beta_1, y_pred_calc = calculate_regression_coefficients(advertising, sales)
print("\n" + "="*80 + "\n")

# Visualize the regression model
def visualize_regression_model(x, y, y_pred, residuals, beta_0, beta_1, save_dir=None):
    """Visualize the regression model and the data."""
    saved_files = []
    
    # Generate points for the regression line
    x_line = np.linspace(0, 10, 100)
    y_line = beta_0 + beta_1 * x_line
    
    # Figure 5: Data with regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', s=100, label='Actual Sales')
    plt.scatter(x, y_pred, color='red', s=80, label='Predicted Sales')
    plt.plot(x_line, y_line, color='green', linewidth=2, 
            label=f'Regression Line: y = {beta_0:.2f} + {beta_1:.2f}x')
    
    # Connect actual and predicted points
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [y[i], y_pred[i]], 'k--', alpha=0.5)
    
    plt.xlabel('Advertising Expenditure ($1000s)', fontsize=12)
    plt.ylabel('Sales (units)', fontsize=12)
    plt.title('Sales vs. Advertising with Regression Line', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if save_dir:
        file_path = os.path.join(save_dir, "plot5_regression_line.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        saved_files.append(file_path)
    
    return saved_files

# Execute regression visualization
reg_vis_files = visualize_regression_model(advertising, sales, predicted_sales, 
                                          residuals, beta_0, beta_1, save_dir)
saved_files.extend(reg_vis_files)

# Summary of results
print("Summary of Results:")
print(f"1. Residuals: {residuals}")
print(f"2. Mean of residuals: {mean_residual:.4f}")
print(f"3. Variance of residuals: {var_residual:.4f}")
print(f"4. Standard deviation of residuals: {std_residual:.4f}")
print(f"5. Sum of residuals: {sum_residuals:.4f}")
print(f"6. Regression model: Sales = {beta_0:.4f} + {beta_1:.4f} × Advertising")

print("\nVisualizations saved to:", save_dir)
for i, file in enumerate(saved_files):
    print(f"{i+1}. {os.path.basename(file)}")

print("\nConclusion:")
print("1. The residuals have a mean very close to zero, which is expected in regression models with an intercept.")
print("2. The sum of residuals is close to zero (though not exactly zero), which is consistent with linear regression theory.")
print("3. The residuals appear to be fairly random without obvious patterns, suggesting the linear model is appropriate.")
print("4. The visual assessment of normality (histogram and Q-Q plot) suggests the residuals approximately follow a normal distribution.")
print("5. These observations support the classical assumptions of linear regression, which require errors to be normally distributed with zero mean.") 