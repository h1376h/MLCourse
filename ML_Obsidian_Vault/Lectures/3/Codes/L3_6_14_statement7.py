import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
from matplotlib.gridspec import GridSpec

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

def statement7_mae_mse():
    """
    Statement 7: Mean Absolute Error (MAE) is less sensitive to outliers 
    than Mean Squared Error (MSE).
    """
    print("\n==== Statement 7: MAE vs MSE Sensitivity to Outliers ====")
    
    # Generate a base dataset with no outliers
    n_samples = 100
    X = np.linspace(0, 10, n_samples)
    y_true = 2 * X + 1
    y_pred_base = y_true + np.random.normal(0, 1, n_samples)
    
    # Compute base MAE and MSE
    base_mae = mean_absolute_error(y_true, y_pred_base)
    base_mse = mean_squared_error(y_true, y_pred_base)
    
    print("\nBaseline Metrics (No Outliers):")
    print(f"Mean Absolute Error (MAE): {base_mae:.4f}")
    print(f"Mean Squared Error (MSE): {base_mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {np.sqrt(base_mse):.4f}")
    
    # Create a dataset with outliers
    y_pred_outliers = y_pred_base.copy()
    
    # Introduce outliers
    outlier_indices = [10, 30, 50, 70, 90]
    outlier_values = [15, -10, 20, -15, 25]  # magnitude of outliers
    
    for idx, val in zip(outlier_indices, outlier_values):
        y_pred_outliers[idx] = y_true[idx] + val
    
    # Compute MAE and MSE with outliers
    outlier_mae = mean_absolute_error(y_true, y_pred_outliers)
    outlier_mse = mean_squared_error(y_true, y_pred_outliers)
    
    print("\nMetrics with Outliers:")
    print(f"Mean Absolute Error (MAE): {outlier_mae:.4f}")
    print(f"Mean Squared Error (MSE): {outlier_mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {np.sqrt(outlier_mse):.4f}")
    
    # Calculate percentage increase
    mae_increase = (outlier_mae - base_mae) / base_mae * 100
    mse_increase = (outlier_mse - base_mse) / base_mse * 100
    
    print("\nPercentage Increase Due to Outliers:")
    print(f"MAE Increase: {mae_increase:.2f}%")
    print(f"MSE Increase: {mse_increase:.2f}%")
    
    if mse_increase > mae_increase:
        print(f"\nMSE increased {mse_increase/mae_increase:.2f}x more than MAE, demonstrating its higher sensitivity to outliers.")
    else:
        print("\nUnexpectedly, MAE showed more sensitivity than MSE in this case.")
    
    # Explain why MAE is less sensitive to outliers
    print("\nWhy MAE is Less Sensitive to Outliers than MSE:")
    print("1. MSE squares the errors, which amplifies large errors disproportionately")
    print("2. MAE uses absolute values, treating all errors linearly based on their magnitude")
    print("3. For an error that's twice as large:")
    print("   - In MAE, it contributes twice as much to the total error")
    print("   - In MSE, it contributes four times as much to the total error")
    print("4. This makes MSE particularly sensitive to outliers or large deviations")
    
    # Visualization 1: Show the predictions with and without outliers
    plt.figure(figsize=(10, 6))
    
    plt.scatter(X, y_true, alpha=0.5, label='True Values')
    plt.scatter(X, y_pred_base, alpha=0.5, label='Predictions (No Outliers)')
    plt.scatter(X, y_pred_outliers, alpha=0.5, label='Predictions (With Outliers)')
    
    # Highlight the outliers
    plt.scatter(X[outlier_indices], y_pred_outliers[outlier_indices], 
                color='red', s=100, label='Outliers', zorder=5)
    
    plt.title('True Values vs Predictions With and Without Outliers')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'statement7_predictions.png'), dpi=300, bbox_inches='tight')
    
    # Visualization 2: Error contributions
    plt.figure(figsize=(12, 6))
    
    # Calculate errors for each point
    abs_errors = np.abs(y_true - y_pred_outliers)
    sq_errors = (y_true - y_pred_outliers) ** 2
    
    # Normalize for comparison
    abs_errors_norm = abs_errors / np.sum(abs_errors)
    sq_errors_norm = sq_errors / np.sum(sq_errors)
    
    # Create bar chart
    bar_width = 0.35
    indices = np.arange(len(outlier_indices))
    
    # Extract outlier errors for clarity
    outlier_abs_errors = abs_errors_norm[outlier_indices]
    outlier_sq_errors = sq_errors_norm[outlier_indices]
    
    plt.bar(indices - bar_width/2, outlier_abs_errors, bar_width, 
            label='MAE Contribution (Normalized)', alpha=0.7)
    plt.bar(indices + bar_width/2, outlier_sq_errors, bar_width, 
            label='MSE Contribution (Normalized)', alpha=0.7)
    
    plt.xlabel('Outlier Index')
    plt.ylabel('Normalized Error Contribution')
    plt.title('Contribution of Each Outlier to Total Error')
    plt.xticks(indices, [f'{X[i]:.1f}' for i in outlier_indices])
    plt.legend()
    
    # Add text showing the absolute contribution values
    for i, idx in enumerate(outlier_indices):
        plt.text(i - bar_width/2, outlier_abs_errors[i] + 0.01, 
                 f'{outlier_abs_errors[i]:.3f}', ha='center')
        plt.text(i + bar_width/2, outlier_sq_errors[i] + 0.01, 
                 f'{outlier_sq_errors[i]:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement7_error_contributions.png'), dpi=300, bbox_inches='tight')
    
    # Visualization 3: Theoretical relationship between error and its contribution
    plt.figure(figsize=(10, 6))
    
    error_range = np.linspace(0, 10, 1000)
    mae_contrib = error_range  # Linear relationship
    mse_contrib = error_range ** 2  # Quadratic relationship
    
    plt.plot(error_range, mae_contrib, label='MAE: f(error) = |error|', linewidth=2)
    plt.plot(error_range, mse_contrib, label='MSE: f(error) = error²', linewidth=2)
    
    plt.title('How Errors Contribute to MAE vs MSE')
    plt.xlabel('Error Magnitude')
    plt.ylabel('Contribution to Loss Function')
    plt.legend()
    plt.grid(True)
    
    # Add some annotations
    plt.annotate('Linear growth', xy=(8, 8), xytext=(6, 6),
                arrowprops=dict(facecolor='blue', shrink=0.05))
    
    plt.annotate('Quadratic growth', xy=(8, 64), xytext=(5, 50),
                arrowprops=dict(facecolor='orange', shrink=0.05))
    
    plt.savefig(os.path.join(save_dir, 'statement7_error_functions.png'), dpi=300, bbox_inches='tight')
    
    # NEW VISUALIZATION: Effect of increasing outlier magnitude
    # Define outlier magnitudes to test
    outlier_magnitudes = np.linspace(0, 50, 20)
    mae_values = []
    mse_values = []
    
    # For each magnitude, calculate MAE and MSE
    for magnitude in outlier_magnitudes:
        # Copy the base predictions
        y_pred_test = y_pred_base.copy()
        
        # Add an outlier of increasing magnitude
        outlier_idx = 50  # middle of the dataset
        y_pred_test[outlier_idx] = y_true[outlier_idx] + magnitude
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred_test)
        mse = mean_squared_error(y_true, y_pred_test)
        
        # Store results
        mae_values.append(mae)
        mse_values.append(mse)
    
    # Create a figure with GridSpec to have two related plots
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3)
    
    # Top plot: MAE and MSE vs Outlier Magnitude
    ax1 = fig.add_subplot(gs[0])
    
    # Normalize the values for better comparison
    mae_norm = np.array(mae_values) / mae_values[0]
    mse_norm = np.array(mse_values) / mse_values[0]
    
    # Plot normalized values
    ax1.plot(outlier_magnitudes, mae_norm, 'b-', linewidth=2, label='MAE (normalized)')
    ax1.plot(outlier_magnitudes, mse_norm, 'r-', linewidth=2, label='MSE (normalized)')
    
    # Add reference line at 45 degrees for MAE
    line_length = max(mse_norm)
    ax1.plot([0, line_length], [1, 1 + line_length/10], 'b--', alpha=0.5, label='Linear growth reference')
    
    # Add reference for MSE
    x_ref = np.linspace(0, np.sqrt(line_length), 100)
    y_ref = 1 + x_ref**2 / 10
    ax1.plot(x_ref, y_ref, 'r--', alpha=0.5, label='Quadratic growth reference')
    
    ax1.set_title('Effect of Increasing Outlier Magnitude on MAE vs MSE', fontsize=14)
    ax1.set_xlabel('Outlier Magnitude', fontsize=12)
    ax1.set_ylabel('Normalized Error (relative to no outlier)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True)
    
    # Add some annotations
    ax1.annotate('MSE grows quadratically', xy=(30, mse_norm[12]), xytext=(20, mse_norm[12]-5),
                arrowprops=dict(facecolor='red', shrink=0.05), fontsize=10)
    
    ax1.annotate('MAE grows linearly', xy=(30, mae_norm[12]), xytext=(15, mae_norm[12]+1),
                arrowprops=dict(facecolor='blue', shrink=0.05), fontsize=10)
    
    # Bottom plot: Ratio of MSE to MAE
    ax2 = fig.add_subplot(gs[1])
    ratio = mse_norm / mae_norm
    
    ax2.plot(outlier_magnitudes, ratio, 'g-', linewidth=2)
    ax2.set_title('Ratio of MSE to MAE Sensitivity (Higher = MSE more sensitive)', fontsize=14)
    ax2.set_xlabel('Outlier Magnitude', fontsize=12)
    ax2.set_ylabel('MSE/MAE Sensitivity Ratio', fontsize=12)
    ax2.grid(True)
    
    # Add horizontal line at ratio=1
    ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    
    plt.savefig(os.path.join(save_dir, 'statement7_outlier_sensitivity.png'), dpi=300, bbox_inches='tight')
    
    # Create a practical application example with real data
    plt.figure(figsize=(12, 8))
    
    # Setup the grid for 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Top-left: Housing price prediction example
    ax = axes[0, 0]
    
    # Generate synthetic housing data
    np.random.seed(42)
    house_sizes = np.linspace(1000, 3000, 50)  # square feet
    house_prices = 100000 + 200 * house_sizes + np.random.normal(0, 30000, 50)  # base + per sqft + noise
    
    # Add price outliers (luxury houses or special cases)
    outlier_indices = [10, 30, 45]
    house_prices[outlier_indices] = [900000, 1200000, 1500000]  # luxury houses
    
    # Plot the data
    ax.scatter(house_sizes, house_prices, alpha=0.7, label='Regular Houses')
    ax.scatter(house_sizes[outlier_indices], house_prices[outlier_indices], color='red', s=100, 
               label='Luxury Houses (Outliers)')
    
    # Fit models with MAE and MSE objectives
    from sklearn.linear_model import LinearRegression, HuberRegressor
    from sklearn.preprocessing import StandardScaler
    
    # Regular OLS (equivalent to MSE optimization)
    lr_model = LinearRegression()
    lr_model.fit(house_sizes.reshape(-1, 1), house_prices)
    
    # Robust regression (less sensitive to outliers)
    scaler = StandardScaler()
    house_sizes_scaled = scaler.fit_transform(house_sizes.reshape(-1, 1))
    huber_model = HuberRegressor(epsilon=1.35)
    huber_model.fit(house_sizes_scaled, house_prices)
    
    # Generate predictions
    house_sizes_plot = np.linspace(1000, 3000, 100).reshape(-1, 1)
    house_sizes_plot_scaled = scaler.transform(house_sizes_plot)
    
    lr_pred = lr_model.predict(house_sizes_plot)
    huber_pred = huber_model.predict(house_sizes_plot_scaled)
    
    # Plot predictions
    ax.plot(house_sizes_plot, lr_pred, color='blue', linewidth=2, label='MSE-based model')
    ax.plot(house_sizes_plot, huber_pred, color='green', linewidth=2, label='Robust model (MAE-like)')
    
    ax.set_title('Housing Price Prediction with Outliers', fontsize=12)
    ax.set_xlabel('House Size (sq ft)', fontsize=10)
    ax.set_ylabel('Price ($)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True)
    
    # 2. Top-right: Temperature anomaly detection
    ax = axes[0, 1]
    
    # Generate temperature data
    days = np.arange(100)
    temps = 70 + 10 * np.sin(days * 2 * np.pi / 60) + np.random.normal(0, 2, 100)  # seasonal pattern + noise
    
    # Add some outlier days (e.g., heat waves)
    outlier_days = [15, 45, 75]
    temps[outlier_days] = [95, 98, 97]  # heat wave days
    
    # Plot the data
    ax.plot(days, temps, 'b-', alpha=0.5)
    ax.scatter(days, temps, alpha=0.5, label='Daily Temperatures')
    ax.scatter(days[outlier_days], temps[outlier_days], color='red', s=100, label='Heat Waves (Outliers)')
    
    # Calculate moving averages (simulating models with different sensitivities)
    from scipy.ndimage import gaussian_filter1d
    
    smooth_mse = gaussian_filter1d(temps, sigma=3)  # More affected by outliers
    smooth_mae = gaussian_filter1d(temps, sigma=3)  # We'll adjust this to be more robust
    
    # Adjust the MAE-like smoothing to be less affected by outliers
    for day in outlier_days:
        window = 5
        for i in range(max(0, day-window), min(len(days), day+window+1)):
            weight = 1 - min(1, abs(i-day)/window)
            smooth_mae[i] = smooth_mae[i] * (1-weight) + temps[day-10] * weight
    
    # Plot the smoothed lines
    ax.plot(days, smooth_mse, 'r-', linewidth=2, label='MSE-like smoothing')
    ax.plot(days, smooth_mae, 'g-', linewidth=2, label='MAE-like smoothing')
    
    ax.set_title('Temperature Anomaly Detection', fontsize=12)
    ax.set_xlabel('Day', fontsize=10)
    ax.set_ylabel('Temperature (°F)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True)
    
    # 3. Bottom-left: Error functions mathematical visualization
    ax = axes[1, 0]
    
    # Create a 2D grid of errors
    error_x = np.linspace(-5, 5, 100)
    error_y = np.linspace(-5, 5, 100)
    error_X, error_Y = np.meshgrid(error_x, error_y)
    
    # Calculate MAE and MSE surfaces
    Z_mae = np.abs(error_X) + np.abs(error_Y)
    Z_mse = error_X**2 + error_Y**2
    
    # Normalize for comparison
    Z_mae = Z_mae / np.max(Z_mae)
    Z_mse = Z_mse / np.max(Z_mse)
    
    # Plot contours
    c1 = ax.contour(error_X, error_Y, Z_mae, levels=10, colors='blue', alpha=0.5)
    c2 = ax.contour(error_X, error_Y, Z_mse, levels=10, colors='red', alpha=0.5)
    
    # Add labels
    ax.clabel(c1, inline=True, fontsize=8, fmt='MAE: %.1f')
    ax.clabel(c2, inline=True, fontsize=8, fmt='MSE: %.1f')
    
    # Add examples showing how outliers affect each metric
    ax.scatter([0, 0, 3], [0, 2, 0], color='black', s=50, label='Data points')
    
    # Annotate points
    ax.annotate('Center', xy=(0, 0), xytext=(0.5, 0.5), 
               arrowprops=dict(facecolor='black', shrink=0.05), fontsize=8)
    
    ax.annotate('Moderate Error', xy=(0, 2), xytext=(0.5, 2.5), 
               arrowprops=dict(facecolor='black', shrink=0.05), fontsize=8)
    
    ax.annotate('Outlier', xy=(3, 0), xytext=(3.5, 0.5), 
               arrowprops=dict(facecolor='black', shrink=0.05), fontsize=8)
    
    ax.set_title('Error Functions in 2D Space', fontsize=12)
    ax.set_xlabel('Error in Dimension 1', fontsize=10)
    ax.set_ylabel('Error in Dimension 2', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True)
    
    # Add explanation
    ax.text(0.02, 0.02, 
           "MAE: Diamond contours (L1 norm)\nMSE: Circular contours (L2 norm)\n\nMSE contours grow faster in outlier regions",
           transform=ax.transAxes, fontsize=8, 
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # 4. Bottom-right: Practical implications for model selection
    ax = axes[1, 1]
    ax.axis('off')  # Turn off the axes
    
    # Create a table of pros and cons
    table_data = [
        ['', 'MAE', 'MSE'],
        ['Outlier Sensitivity', 'Lower', 'Higher'],
        ['Differentiability', 'Non-differentiable at 0', 'Differentiable everywhere'],
        ['Interpretation', 'Average absolute deviation', 'Average squared deviation'],
        ['Best For', 'Robustness to outliers\nMedian prediction', 'Statistical efficiency\nMean prediction'],
        ['Use When', 'Data has outliers\nAll errors equally important', 'Larger errors more important\nNormal distribution assumed']
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Style the header
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white')
    
    # Style the row headers
    for i in range(1, 6):
        table[(i, 0)].set_facecolor('#D9E1F2')
    
    ax.set_title('MAE vs MSE: When to Use Each', fontsize=12, pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement7_practical_applications.png'), dpi=300, bbox_inches='tight')
    
    # Return result
    result = {
        'statement': "Mean Absolute Error (MAE) is less sensitive to outliers than Mean Squared Error (MSE).",
        'is_true': True,
        'explanation': "This statement is TRUE. MAE uses the absolute difference between predicted and actual values, which scales linearly with the error magnitude. MSE squares these differences, which disproportionately penalizes large errors. When outliers are present, they have a much greater effect on MSE than on MAE. From our analysis, we saw that introducing outliers caused the MSE to increase by " + f"{mse_increase:.2f}% compared to only {mae_increase:.2f}% for MAE. This {mse_increase/mae_increase:.2f}x difference demonstrates that MSE is indeed more sensitive to outliers than MAE.",
        'image_path': ['statement7_predictions.png', 'statement7_error_contributions.png', 'statement7_error_functions.png', 'statement7_outlier_sensitivity.png', 'statement7_practical_applications.png']
    }
    
    return result

if __name__ == "__main__":
    result = statement7_mae_mse()
    print(f"\nStatement: {result['statement']}")
    print(f"True or False: {'True' if result['is_true'] else 'False'}")
    print(f"Explanation: {result['explanation']}")
    print(f"Images saved: {', '.join(result['image_path'])}") 