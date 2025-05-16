import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_6_Quiz_14")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('ggplot')
sns.set_style("whitegrid")
np.random.seed(42)  # For reproducibility

def statement7_mae_vs_mse():
    """
    Statement 7: Mean Absolute Error (MAE) is less sensitive to outliers than Mean Squared Error (MSE).
    """
    print("\n==== Statement 7: MAE vs. MSE for Outlier Sensitivity ====")
    
    # Print detailed explanation of MAE and MSE
    print("\nMAE vs. MSE Explained:")
    print("- Mean Absolute Error (MAE): Average of absolute differences between predicted and actual values")
    print("  Formula: MAE = (1/n) * Σ|y_i - ŷ_i|")
    print("- Mean Squared Error (MSE): Average of squared differences between predicted and actual values")
    print("  Formula: MSE = (1/n) * Σ(y_i - ŷ_i)²")
    print("\nKey Differences in Outlier Handling:")
    print("- MAE treats all errors linearly (proportional to their magnitude)")
    print("- MSE penalizes larger errors more heavily (proportional to square of magnitude)")
    print("- An error of 10 contributes 10x more to MAE than error of 1")
    print("- An error of 10 contributes 100x more to MSE than error of 1")
    
    # Generate data with outliers
    np.random.seed(42)
    n_samples = 100
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    
    # Generate target without outliers first
    y_true = 2 * X.squeeze() + np.random.normal(0, 3, n_samples)
    
    # Create a copy with outliers
    y_with_outliers = y_true.copy()
    
    # Add outliers at specific positions
    outlier_indices = [10, 30, 50, 70, 90]
    outlier_values = [25, -20, 30, -25, 28]
    
    for idx, val in zip(outlier_indices, outlier_values):
        y_with_outliers[idx] = val
    
    # Train a model on data with outliers
    model = LinearRegression()
    model.fit(X, y_with_outliers)
    y_pred = model.predict(X)
    
    # Calculate MAE and MSE for data without outliers for baseline
    mae_no_outliers = mean_absolute_error(y_true, y_pred)
    mse_no_outliers = mean_squared_error(y_true, y_pred)
    
    # Calculate MAE and MSE for all data (with outliers)
    mae = mean_absolute_error(y_with_outliers, y_pred)
    mse = mean_squared_error(y_with_outliers, y_pred)
    
    # Calculate MAE and MSE for just the outliers
    mae_outliers = mean_absolute_error(y_with_outliers[outlier_indices], y_pred[outlier_indices])
    mse_outliers = mean_squared_error(y_with_outliers[outlier_indices], y_pred[outlier_indices])
    
    # Calculate error ratios to show sensitivity
    mae_ratio = mae / mae_no_outliers
    mse_ratio = mse / mse_no_outliers
    
    # Print detailed metrics
    print("\nDetailed Error Metrics:")
    print(f"MAE (no outliers): {mae_no_outliers:.2f}")
    print(f"MSE (no outliers): {mse_no_outliers:.2f}")
    print(f"MAE (with outliers): {mae:.2f}")
    print(f"MSE (with outliers): {mse:.2f}")
    print(f"MAE increase ratio due to outliers: {mae_ratio:.2f}x")
    print(f"MSE increase ratio due to outliers: {mse_ratio:.2f}x")
    print(f"MAE for outliers only: {mae_outliers:.2f}")
    print(f"MSE for outliers only: {mse_outliers:.2f}")
    
    print("\nInterpretation:")
    print(f"- MSE increased by {mse_ratio:.2f}x when outliers were added")
    print(f"- MAE increased by {mae_ratio:.2f}x when outliers were added")
    print(f"- MSE increase was {(mse_ratio/mae_ratio):.2f}x greater than MAE increase")
    print("- This confirms that MSE is significantly more sensitive to outliers than MAE")
    
    # Plot 1: Data visualization with model and outliers
    plt.figure(figsize=(10, 6))
    
    # Plot regular data points
    regular_indices = np.ones(n_samples, dtype=bool)
    regular_indices[outlier_indices] = False
    
    plt.scatter(X[regular_indices], y_with_outliers[regular_indices], 
               color='blue', alpha=0.6, label='Regular data points')
    
    # Plot outliers
    plt.scatter(X[outlier_indices], y_with_outliers[outlier_indices], 
               color='red', s=100, label='Outliers', edgecolor='black')
    
    # Plot model prediction
    plt.plot(X, y_pred, color='green', linewidth=2, label='Model prediction')
    
    # Draw error lines for outliers
    for i, idx in enumerate(outlier_indices):
        plt.plot([X[idx], X[idx]], [y_with_outliers[idx], y_pred[idx]], 
                 'k--', alpha=0.7)
    
    plt.title('Linear Regression with Outliers', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.legend()
    plt.grid(True)
    
    # Add formula annotations with LaTeX
    plt.figtext(0.15, 0.02, r'$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$', fontsize=12)
    plt.figtext(0.55, 0.02, r'$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$', fontsize=12)
    
    plt.savefig(os.path.join(save_dir, 'statement7_mae_vs_mse.png'), dpi=300, bbox_inches='tight')
    
    # Plot 2: Error Growth Comparison
    errors = np.linspace(0, 20, 100)  # Range of possible errors
    abs_errors = np.abs(errors)  # MAE (linear growth)
    squared_errors = errors**2  # MSE (quadratic growth)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(errors, abs_errors, 'b-', linewidth=3, label='Absolute Error (MAE)')
    plt.plot(errors, squared_errors, 'r-', linewidth=3, label='Squared Error (MSE)')
    
    # Mark specific error points
    error_points = [1, 5, 10, 15]
    for e in error_points:
        plt.scatter(e, e, color='blue', s=80, zorder=10, edgecolor='black')
        plt.scatter(e, e**2, color='red', s=80, zorder=10, edgecolor='black')
        
        # Connect the points with vertical lines
        plt.plot([e, e], [e, e**2], 'k--', alpha=0.7)
        
        # Add labels
        if e > 1:  # Skip the smallest error to avoid clutter
            plt.annotate(f'Error = {e}\nMAE = {e}\nMSE = {e**2}', 
                        xy=(e, e), 
                        xytext=(e+1, e), 
                        fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.title('Error Growth: MAE vs MSE', fontsize=14)
    plt.xlabel('Error Magnitude |y - ŷ|', fontsize=12)
    plt.ylabel('Contribution to Cost Function', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True)
    
    # Add explanatory annotation
    plt.annotate('MSE grows quadratically with error magnitude\nmaking it more sensitive to outliers', 
                xy=(10, 100),
                xytext=(5, 250),
                arrowprops=dict(arrowstyle='->'),
                fontsize=12,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.2))
    
    plt.savefig(os.path.join(save_dir, 'statement7_error_growth.png'), dpi=300, bbox_inches='tight')
    
    # Plot 3: Bar chart comparing the increase ratios with scaled visualization
    plt.figure(figsize=(10, 6))
    
    # Calculate errors for all points
    errors = y_with_outliers - y_pred
    
    # Create bar width for grouped bars
    bar_width = 0.35
    index = np.arange(2)
    
    # Calculate MAE and MSE contributions for regular points and outliers separately
    regular_mae = np.mean(np.abs(errors[regular_indices]))
    regular_mse = np.mean(errors[regular_indices]**2)
    
    outlier_mae = np.mean(np.abs(errors[outlier_indices]))
    outlier_mse = np.mean(errors[outlier_indices]**2)
    
    # Normalize to show relative contribution
    total_mae = mae
    total_mse = mse
    
    # Calculate percentage contributions
    regular_mae_pct = (regular_mae * len(regular_indices[regular_indices])) / (total_mae * n_samples) * 100
    outlier_mae_pct = (outlier_mae * len(outlier_indices)) / (total_mae * n_samples) * 100
    
    regular_mse_pct = (regular_mse * len(regular_indices[regular_indices])) / (total_mse * n_samples) * 100
    outlier_mse_pct = (outlier_mse * len(outlier_indices)) / (total_mse * n_samples) * 100
    
    # Create bars for regular points and outliers
    plt.bar(index, [regular_mae_pct, regular_mse_pct], bar_width, 
            color='blue', alpha=0.7, label='Regular points')
    plt.bar(index + bar_width, [outlier_mae_pct, outlier_mse_pct], bar_width,
            color='red', alpha=0.7, label='Outliers')
    
    # Add actual percentage on top of bars
    for i, v in enumerate([regular_mae_pct, regular_mse_pct]):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=10)
    
    for i, v in enumerate([outlier_mae_pct, outlier_mse_pct]):
        plt.text(i + bar_width, v + 1, f'{v:.1f}%', ha='center', fontsize=10)
    
    plt.title('Contribution to Error Metrics: Regular Points vs. Outliers', fontsize=14)
    plt.xticks(index + bar_width/2, ['MAE', 'MSE'])
    plt.ylabel('Percentage of Total Error (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.grid(axis='y')
    plt.legend()
    
    # Add explanatory annotation
    if outlier_mse_pct > outlier_mae_pct + 20:
        plt.annotate('Outliers contribute much more\nto MSE than to MAE', 
                    xy=(1 + bar_width/2, outlier_mse_pct),
                    xytext=(1.1, outlier_mse_pct - 20),
                    arrowprops=dict(arrowstyle='->'),
                    fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.savefig(os.path.join(save_dir, 'statement7_error_contribution.png'), dpi=300, bbox_inches='tight')
    
    result = {
        'statement': "Mean Absolute Error (MAE) is less sensitive to outliers than Mean Squared Error (MSE).",
        'is_true': True,
        'explanation': "This statement is TRUE. Mean Absolute Error (MAE) is less sensitive to outliers than Mean Squared Error (MSE) because MAE uses the absolute value of errors (|y - ŷ|), while MSE squares the errors ((y - ŷ)²). When errors are squared, larger errors (like those from outliers) are penalized much more heavily than smaller ones. As demonstrated in our analysis, the MSE increased by a factor of {:.2f}x when outliers were added, compared to only {:.2f}x for MAE. This makes MSE much more influenced by outliers, which can be either desirable or undesirable depending on the context.".format(mse_ratio, mae_ratio),
        'image_path': ['statement7_mae_vs_mse.png', 'statement7_error_growth.png', 'statement7_error_contribution.png']
    }
    
    return result

if __name__ == "__main__":
    result = statement7_mae_vs_mse()
    print(f"\nStatement: {result['statement']}")
    print(f"True or False: {'True' if result['is_true'] else 'False'}")
    print(f"Explanation: {result['explanation']}")
    print(f"Images saved: {', '.join(result['image_path'])}") 