import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
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

def statement4_r2_negative():
    """
    Statement 4: A negative R-squared value indicates that the model is worse than a 
    horizontal line at predicting the target variable.
    """
    print("\n==== Statement 4: Negative R-squared and Horizontal Line ====")
    
    # Print explanation of R-squared
    print("\nR-squared (coefficient of determination) explanation:")
    print("- R² measures how well a model explains the variance in the target variable")
    print("- Formula: R² = 1 - (SSres / SStot)")
    print("  where SSres = sum((y - ŷ)²) and SStot = sum((y - mean(y))²)")
    print("- R² = 1: Perfect predictions (all variance explained)")
    print("- R² = 0: Model performs the same as predicting the mean value")
    print("- R² < 0: Model performs worse than predicting the mean value")
    
    # Generate data
    np.random.seed(42)
    n_samples = 100
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    y = 2 * X.squeeze() + np.random.normal(0, 5, n_samples)
    
    # Calculate mean of y (used for baseline model)
    y_mean = np.mean(y)
    
    # Train a model with negative R-squared (by fitting to inverse y)
    model_negative = LinearRegression()
    model_negative.fit(X, -y)  # Negate y to force negative R-squared
    y_pred_negative = model_negative.predict(X)
    r2_negative = r2_score(y, y_pred_negative)
    
    # Also train a normal model for comparison
    model_positive = LinearRegression()
    model_positive.fit(X, y)
    y_pred_positive = model_positive.predict(X)
    r2_positive = r2_score(y, y_pred_positive)
    
    print(f"\nModel performance metrics:")
    print(f"Negative model R-squared: {r2_negative:.2f}")
    print(f"Positive model R-squared: {r2_positive:.2f}")
    
    # Calculate Sum of Squared Errors for both models and baseline
    sse_negative = np.sum((y - y_pred_negative)**2)
    sse_positive = np.sum((y - y_pred_positive)**2)
    sse_baseline = np.sum((y - y_mean)**2)  # Sum of squared deviations from mean
    
    print(f"\nSum of Squared Errors (SSE):")
    print(f"SSE for negative R² model: {sse_negative:.2f}")
    print(f"SSE for baseline (mean) model: {sse_baseline:.2f}")
    print(f"SSE for positive R² model: {sse_positive:.2f}")
    
    print("\nRelative performance:")
    print(f"- Negative R² model: {sse_negative/sse_baseline:.2f}x worse than baseline")
    print(f"- Positive R² model: {sse_baseline/sse_positive:.2f}x better than baseline")
    
    # Print interpretation of negative R²
    print("\nInterpretation of negative R-squared:")
    print("- When R² is negative, the model's predictions are worse than simply predicting the mean")
    print("- The mean prediction is a horizontal line at y =", y_mean)
    print("- The negative R² model has a higher SSE than the baseline model")
    print("- This means the model is actively worse than a horizontal line at predicting y")
    
    # Plot 1: Model comparison with baseline
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='black', alpha=0.5, label='Data points')
    plt.plot(X, y_pred_negative, color='red', linewidth=2, label=f'Model with R² = {r2_negative:.2f}')
    plt.axhline(y=y_mean, color='blue', linestyle='--', linewidth=2, label=f'Mean line (R² = 0)')
    plt.plot(X, y_pred_positive, color='green', linewidth=2, label=f'Good model with R² = {r2_positive:.2f}')
    
    plt.title('Comparing Models with Different R-squared Values', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.legend()
    plt.grid(True)
    
    # Add LaTeX formula annotation
    plt.annotate(r'$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$',
                xy=(0.05, 0.05), xycoords='axes fraction', fontsize=14,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.savefig(os.path.join(save_dir, 'statement4_r2_negative.png'), dpi=300, bbox_inches='tight')
    
    # Plot 2: Bar chart of SSE
    plt.figure(figsize=(10, 6))
    
    models = ['Negative R² Model', 'Baseline (Mean)', 'Positive R² Model']
    sse_values = [sse_negative, sse_baseline, sse_positive]
    colors = ['red', 'blue', 'green']
    r2_values = [r2_negative, 0, r2_positive]
    
    bars = plt.bar(models, sse_values, color=colors, alpha=0.7)
    
    # Add R² values as text on bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
                 f'R² = {r2_values[i]:.2f}', 
                 ha='center', va='bottom', fontweight='bold')
    
    plt.axhline(y=sse_baseline, color='blue', linestyle='--', 
                label='Baseline SSE (Mean prediction)')
    
    plt.title('Sum of Squared Errors by Model Type', fontsize=14)
    plt.ylabel('Sum of Squared Errors (SSE)', fontsize=12)
    plt.grid(axis='y')
    plt.legend()
    
    plt.savefig(os.path.join(save_dir, 'statement4_sse_comparison.png'), dpi=300, bbox_inches='tight')
    
    # Plot 3: New visualization - R² as a function of error ratio
    plt.figure(figsize=(10, 6))
    
    # Generate data for the relationship between R² and the ratio of model error to baseline error
    error_ratios = np.linspace(0, 2, 100)  # From 0 to 2x baseline error
    r2_values = 1 - error_ratios  # R² = 1 - (SSres / SStot)
    
    plt.plot(error_ratios, r2_values, 'b-', linewidth=2)
    
    # Mark key points
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=1, color='black', linestyle='--', alpha=0.5)
    
    # Add current models to the plot
    model_error_ratios = [sse_negative/sse_baseline, sse_baseline/sse_baseline, sse_positive/sse_baseline]
    model_r2s = [r2_negative, 0, r2_positive]
    
    for ratio, r2, color, name in zip(model_error_ratios, model_r2s, colors, models):
        plt.scatter([ratio], [r2], color=color, s=100, zorder=10, 
                   label=f'{name}: R² = {r2:.2f}')
    
    # Add annotations for key regions
    plt.annotate('Better than mean\nR² > 0', xy=(0.5, 0.5), 
                xytext=(0.5, 0.5), ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.annotate('Worse than mean\nR² < 0', xy=(1.5, -0.5), 
                xytext=(1.5, -0.5), ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    
    plt.xlabel('Ratio of Model Error to Baseline Error (SSres / SStot)', fontsize=12)
    plt.ylabel('R-squared Value', fontsize=12)
    plt.title('R-squared as a Function of Error Ratio', fontsize=14)
    plt.xlim(0, 2)
    plt.ylim(-1.1, 1.1)
    plt.grid(True)
    plt.legend()
    
    # Add LaTeX formula annotation
    plt.annotate(r'$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$', 
                xy=(0.05, 0.92), xycoords='axes fraction', fontsize=14,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.savefig(os.path.join(save_dir, 'statement4_r2_function.png'), dpi=300, bbox_inches='tight')
    
    result = {
        'statement': "A negative R-squared value indicates that the model is worse than a horizontal line at predicting the target variable.",
        'is_true': True,
        'explanation': "This statement is TRUE. R-squared measures the proportion of variance in the dependent variable that is predictable from the independent variables. It is calculated as 1 - (Sum of Squared Residuals / Total Sum of Squares). When R-squared is negative, it means the model performs worse than simply predicting the mean value (horizontal line) for all observations. In other words, the sum of squared errors for the model is greater than the sum of squared errors for the baseline model, which just predicts the mean for every input.",
        'image_path': ['statement4_r2_negative.png', 'statement4_sse_comparison.png', 'statement4_r2_function.png']
    }
    
    return result

if __name__ == "__main__":
    result = statement4_r2_negative()
    print(f"\nStatement: {result['statement']}")
    print(f"True or False: {'True' if result['is_true'] else 'False'}")
    print(f"Explanation: {result['explanation']}")
    print(f"Images saved: {', '.join(result['image_path'])}") 