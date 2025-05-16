import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
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

def statement5_residual_analysis():
    """
    Statement 5: Residual analysis is essential to ensure that the assumptions of a 
    linear regression model are met.
    """
    print("\n==== Statement 5: Residual Analysis for Linear Regression ====")
    
    # Print explanation of linear regression assumptions and residual analysis
    print("\nLinear Regression Assumptions:")
    print("1. Linearity: The relationship between predictors and response is linear")
    print("2. Independence: The residuals are independent (not correlated with each other)")
    print("3. Homoscedasticity: The residuals have constant variance across all predictor values")
    print("4. Normality: The residuals are normally distributed")
    print("5. No multicollinearity: Predictors are not highly correlated (for multiple regression)")
    
    print("\nImportance of Residual Analysis:")
    print("- Residuals = Actual values - Predicted values")
    print("- Residual analysis helps verify if model assumptions are met")
    print("- Violations of assumptions can lead to:")
    print("  • Biased coefficient estimates")
    print("  • Incorrect standard errors")
    print("  • Invalid hypothesis tests and confidence intervals")
    print("  • Poor predictive performance")
    
    # Generate two datasets: one that satisfies assumptions, one that doesn't
    np.random.seed(42)
    n_samples = 100
    
    # Dataset 1: Linear relationship with homoscedastic errors
    X1 = np.linspace(0, 10, n_samples).reshape(-1, 1)
    y1 = 2 * X1.squeeze() + np.random.normal(0, 3, n_samples)
    
    # Dataset 2: Non-linear relationship with heteroscedastic errors
    X2 = np.linspace(0, 10, n_samples).reshape(-1, 1)
    y2 = 2 * X2.squeeze() + 0.5 * X2.squeeze()**2 + np.random.normal(0, 1 + X2.squeeze(), n_samples)
    
    # Fit linear models to both datasets
    model1 = LinearRegression()
    model1.fit(X1, y1)
    y1_pred = model1.predict(X1)
    residuals1 = y1 - y1_pred
    
    model2 = LinearRegression()
    model2.fit(X2, y2)
    y2_pred = model2.predict(X2)
    residuals2 = y2 - y2_pred
    
    # Print analysis of model performance
    print("\nModel Performance Metrics:")
    print("Good Model (Linear Data):")
    print(f"  Coefficient: {model1.coef_[0]:.2f}")
    print(f"  Intercept: {model1.intercept_:.2f}")
    print(f"  Mean of residuals: {np.mean(residuals1):.4f} (should be close to 0)")
    print(f"  Standard deviation of residuals: {np.std(residuals1):.4f}")
    
    print("\nProblematic Model (Non-linear Data):")
    print(f"  Coefficient: {model2.coef_[0]:.2f}")
    print(f"  Intercept: {model2.intercept_:.2f}")
    print(f"  Mean of residuals: {np.mean(residuals2):.4f} (should be close to 0)")
    print(f"  Standard deviation of residuals: {np.std(residuals2):.4f}")
    
    # Perform residual analysis and print results
    print("\nResidual Analysis:")
    
    # Test for normality using Shapiro-Wilk test
    _, p_value1 = stats.shapiro(residuals1)
    _, p_value2 = stats.shapiro(residuals2)
    
    print("Normality Test (Shapiro-Wilk):")
    print(f"  Good model p-value: {p_value1:.4f} (p>0.05 suggests normality)")
    print(f"  Problematic model p-value: {p_value2:.4f} (p<0.05 suggests non-normality)")
    
    # Test for homoscedasticity using Breusch-Pagan test (simplified)
    # We'll use a correlation test between residuals and predicted values as a simple check
    corr1 = np.corrcoef(np.abs(residuals1), y1_pred)[0, 1]
    corr2 = np.corrcoef(np.abs(residuals2), y2_pred)[0, 1]
    
    print("\nHomoscedasticity Check (Correlation between |residuals| and predicted values):")
    print(f"  Good model correlation: {corr1:.4f} (close to 0 suggests homoscedasticity)")
    print(f"  Problematic model correlation: {corr2:.4f} (away from 0 suggests heteroscedasticity)")
    
    # Plot 1: Simplified residual analysis comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Top row: Model fits
    # Good model
    ax = axes[0, 0]
    ax.scatter(X1, y1, color='blue', alpha=0.5, label='Data')
    ax.plot(X1, y1_pred, color='red', linewidth=2, label='Linear fit')
    ax.set_title('Good Model: Linear Data', fontsize=12)
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.legend()
    
    # Bad model
    ax = axes[0, 1]
    ax.scatter(X2, y2, color='blue', alpha=0.5, label='Data')
    ax.plot(X2, y2_pred, color='red', linewidth=2, label='Linear fit')
    # Add the true quadratic function for comparison
    x_plot = np.linspace(0, 10, 100)
    y_true = 2 * x_plot + 0.5 * x_plot**2
    ax.plot(x_plot, y_true, 'g--', label='True relationship')
    ax.set_title('Bad Model: Non-linear Data', fontsize=12)
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.legend()
    
    # Bottom row: Residual plots
    # Good model residuals
    ax = axes[1, 0]
    ax.scatter(y1_pred, residuals1, color='green', alpha=0.7)
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_title('Good Model: Residuals vs Predicted', fontsize=12)
    ax.set_xlabel('Predicted Values', fontsize=10)
    ax.set_ylabel('Residuals', fontsize=10)
    
    # Bad model residuals
    ax = axes[1, 1]
    ax.scatter(y2_pred, residuals2, color='green', alpha=0.7)
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_title('Bad Model: Residuals vs Predicted', fontsize=12)
    ax.set_xlabel('Predicted Values', fontsize=10)
    ax.set_ylabel('Residuals', fontsize=10)
    
    plt.savefig(os.path.join(save_dir, 'statement5_residual_comparison.png'), dpi=300, bbox_inches='tight')
    
    # Print interpretation of the residual plots
    print("\nInterpretation of Residual Plots:")
    print("Good Model:")
    print("- Residuals randomly scattered around zero")
    print("- No obvious pattern in residuals")
    print("- Relatively constant variance across predicted values")
    print("- Assumptions of linearity and homoscedasticity likely met")
    
    print("\nProblematic Model:")
    print("- Clear pattern in residuals (U-shaped)")
    print("- Indicates that a linear model is inappropriate (non-linearity)")
    print("- Variance increases with predicted values (heteroscedasticity)")
    print("- Linear regression assumptions are violated")
    
    # Plot 2: Q-Q plots for normality assessment
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Q-Q plot for good model
    stats.probplot(residuals1, dist="norm", plot=axes[0])
    axes[0].set_title('Good Model: Q-Q Plot', fontsize=12)
    
    # Q-Q plot for bad model
    stats.probplot(residuals2, dist="norm", plot=axes[1])
    axes[1].set_title('Problematic Model: Q-Q Plot', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement5_qq_plots.png'), dpi=300, bbox_inches='tight')
    
    # Print interpretation of Q-Q plots
    print("\nInterpretation of Q-Q Plots:")
    print("Good Model:")
    print("- Points follow the diagonal line closely")
    print("- Suggests residuals are approximately normally distributed")
    print("- Normality assumption is reasonably satisfied")
    
    print("\nProblematic Model:")
    print("- Substantial deviations from the diagonal line")
    print("- Indicates non-normal distribution of residuals")
    print("- Normality assumption is violated")
    
    # Plot 3: New visualization - Consequences of assumption violations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Generate some data for visualization
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    
    # 1. Linearity Violation
    ax = axes[0, 0]
    # True quadratic relationship
    y_true = 1 + 2*x + 0.5*x**2
    # Add noise
    y_obs = y_true + np.random.normal(0, 3, size=len(x))
    # Fit linear model
    X_lin = x.reshape(-1, 1)
    model_lin = LinearRegression().fit(X_lin, y_obs)
    y_pred = model_lin.predict(X_lin)
    
    # Plot
    ax.scatter(x, y_obs, alpha=0.5, label='Data')
    ax.plot(x, y_true, 'g-', label='True Relationship')
    ax.plot(x, y_pred, 'r--', label='Linear Model')
    ax.set_title('Linearity Violation', fontsize=12)
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.legend(fontsize=8)
    
    # Add annotation with consequences
    ax.text(0.05, 0.95, "Consequences:\n- Biased estimates\n- Poor predictions\n- Incorrect inference",
           transform=ax.transAxes, fontsize=9, va='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Heteroscedasticity
    ax = axes[0, 1]
    # Increasing variance with x
    noise = np.random.normal(0, 0.5 + 0.5*x, size=len(x))
    y_obs = 1 + 2*x + noise
    # Fit linear model
    model_lin = LinearRegression().fit(X_lin, y_obs)
    y_pred = model_lin.predict(X_lin)
    
    # Plot
    ax.scatter(x, y_obs, alpha=0.5, label='Data')
    ax.plot(x, 1 + 2*x, 'g-', label='True Relationship')
    ax.plot(x, y_pred, 'r--', label='Linear Model')
    ax.set_title('Heteroscedasticity', fontsize=12)
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.legend(fontsize=8)
    
    # Add annotation with consequences
    ax.text(0.05, 0.95, "Consequences:\n- Inefficient estimates\n- Invalid confidence intervals\n- Poor hypothesis tests",
           transform=ax.transAxes, fontsize=9, va='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Non-normality
    ax = axes[1, 0]
    # Skewed residuals
    noise = np.random.exponential(3, size=len(x)) - 3  # Mean-centered exponential
    y_obs = 1 + 2*x + noise
    # Fit linear model
    model_lin = LinearRegression().fit(X_lin, y_obs)
    y_pred = model_lin.predict(X_lin)
    residuals = y_obs - y_pred
    
    # Plot residuals histogram
    ax.hist(residuals, bins=20, alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--')
    ax.set_title('Non-normal Residuals', fontsize=12)
    ax.set_xlabel('Residual Value', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    
    # Add annotation with consequences
    ax.text(0.05, 0.95, "Consequences:\n- Invalid confidence intervals\n- Unreliable hypothesis tests\n- Poor prediction intervals",
           transform=ax.transAxes, fontsize=9, va='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Autocorrelation (temporal data)
    ax = axes[1, 1]
    # Generate autocorrelated residuals
    n = len(x)
    e = np.random.normal(0, 1, n)
    residuals = np.zeros(n)
    residuals[0] = e[0]
    for i in range(1, n):
        residuals[i] = 0.8 * residuals[i-1] + e[i]  # AR(1) process
    
    y_obs = 1 + 2*x + residuals
    # Fit linear model
    model_lin = LinearRegression().fit(X_lin, y_obs)
    y_pred = model_lin.predict(X_lin)
    residuals = y_obs - y_pred
    
    # Plot residuals vs. index (time)
    ax.plot(range(n), residuals, 'b-')
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_title('Autocorrelated Residuals', fontsize=12)
    ax.set_xlabel('Index/Time', fontsize=10)
    ax.set_ylabel('Residual Value', fontsize=10)
    
    # Add annotation with consequences
    ax.text(0.05, 0.95, "Consequences:\n- Underestimated standard errors\n- Overconfident inference\n- Invalid hypothesis tests",
           transform=ax.transAxes, fontsize=9, va='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement5_assumption_violations.png'), dpi=300, bbox_inches='tight')
    
    # NEW VISUALIZATION: Comprehensive residual analysis diagnostic framework
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    # Generate a dataset with good properties
    np.random.seed(42)
    x_good = np.linspace(0, 10, 100)
    y_good = 3 + 2*x_good + np.random.normal(0, 3, 100)
    X_good = x_good.reshape(-1, 1)
    
    # Fit a linear model
    model_good = LinearRegression().fit(X_good, y_good)
    y_pred_good = model_good.predict(X_good)
    residuals_good = y_good - y_pred_good
    
    # 1. Original Data and Model Fit (top-left)
    ax = axes[0, 0]
    ax.scatter(x_good, y_good, alpha=0.6, label='Data')
    ax.plot(x_good, y_pred_good, 'r-', linewidth=2, label='Model Fit')
    ax.set_title('1. Data and Model Fit', fontsize=12)
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.legend(fontsize=9)
    
    # Annotation
    ax.text(0.05, 0.95, "Check: Does the model fit appear\nreasonable for the data pattern?",
           transform=ax.transAxes, fontsize=9, va='top',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # 2. Residuals vs. Fitted (top-right) - Check Linearity and Homoscedasticity
    ax = axes[0, 1]
    ax.scatter(y_pred_good, residuals_good, alpha=0.6)
    ax.axhline(y=0, color='r', linestyle='--')
    # Add a smooth line
    sorted_idx = np.argsort(y_pred_good)
    sorted_pred = y_pred_good[sorted_idx]
    sorted_resid = residuals_good[sorted_idx]
    smooth_resid = np.convolve(sorted_resid, np.ones(10)/10, mode='valid')
    smooth_pred = sorted_pred[5:-4]  # Adjust for kernel size
    ax.plot(smooth_pred, smooth_resid, 'g-', linewidth=2)
    
    ax.set_title('2. Residuals vs. Fitted Values', fontsize=12)
    ax.set_xlabel('Fitted Values', fontsize=10)
    ax.set_ylabel('Residuals', fontsize=10)
    
    # Annotations
    ax.text(0.05, 0.95, "Checks:\n- Linearity: Trend line should be flat\n- Homoscedasticity: Equal scatter at all X",
           transform=ax.transAxes, fontsize=9, va='top',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # 3. Scale-Location (middle-left) - Check Homoscedasticity more clearly
    ax = axes[1, 0]
    # Standardized residuals
    std_residuals = residuals_good / np.std(residuals_good)
    ax.scatter(y_pred_good, np.sqrt(np.abs(std_residuals)), alpha=0.6)
    
    # Add a smooth line
    sorted_idx = np.argsort(y_pred_good)
    sorted_pred = y_pred_good[sorted_idx]
    sorted_sqrt_resid = np.sqrt(np.abs(std_residuals[sorted_idx]))
    smooth_sqrt_resid = np.convolve(sorted_sqrt_resid, np.ones(10)/10, mode='valid')
    smooth_pred = sorted_pred[5:-4]  # Adjust for kernel size
    ax.plot(smooth_pred, smooth_sqrt_resid, 'g-', linewidth=2)
    
    ax.set_title('3. Scale-Location Plot', fontsize=12)
    ax.set_xlabel('Fitted Values', fontsize=10)
    ax.set_ylabel('√|Standardized Residuals|', fontsize=10)
    
    # Annotations
    ax.text(0.05, 0.95, "Check: Homoscedasticity\n- Line should be flat\n- Points equally spread",
           transform=ax.transAxes, fontsize=9, va='top',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # 4. Normal Q-Q Plot (middle-right) - Check Normality
    ax = axes[1, 1]
    # Create a Q-Q plot
    stats.probplot(residuals_good, dist="norm", plot=ax)
    ax.set_title('4. Normal Q-Q Plot', fontsize=12)
    
    # Annotation
    ax.text(0.05, 0.95, "Check: Normality\n- Points should follow the line\n- Deviations at tails are common",
           transform=ax.transAxes, fontsize=9, va='top',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # 5. Residuals vs. X (bottom-left) - Check Independent Errors and Linearity
    ax = axes[2, 0]
    ax.scatter(x_good, residuals_good, alpha=0.6)
    ax.axhline(y=0, color='r', linestyle='--')
    
    # Add a smooth line
    sorted_idx = np.argsort(x_good)
    sorted_x = x_good[sorted_idx]
    sorted_resid = residuals_good[sorted_idx]
    smooth_resid = np.convolve(sorted_resid, np.ones(10)/10, mode='valid')
    smooth_x = sorted_x[5:-4]  # Adjust for kernel size
    ax.plot(smooth_x, smooth_resid, 'g-', linewidth=2)
    
    ax.set_title('5. Residuals vs. X', fontsize=12)
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Residuals', fontsize=10)
    
    # Annotation
    ax.text(0.05, 0.95, "Checks:\n- Linearity: No pattern against X\n- Independence: No systematic pattern",
           transform=ax.transAxes, fontsize=9, va='top',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # 6. Residual Histogram (bottom-right) - Check Normality
    ax = axes[2, 1]
    ax.hist(residuals_good, bins=20, alpha=0.7, color='skyblue')
    ax.axvline(x=0, color='r', linestyle='--')
    
    # Add a normal curve for reference
    from scipy.stats import norm
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, np.mean(residuals_good), np.std(residuals_good))
    ax.plot(x, p * len(residuals_good) * (xmax-xmin)/20, 'k-', linewidth=2)
    
    ax.set_title('6. Residual Histogram', fontsize=12)
    ax.set_xlabel('Residual Value', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    
    # Annotation
    ax.text(0.05, 0.95, "Check: Normality\n- Bell-shaped distribution\n- Centered at zero",
           transform=ax.transAxes, fontsize=9, va='top',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.suptitle('Comprehensive Residual Analysis Framework', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_dir, 'statement5_comprehensive_residuals.png'), dpi=300, bbox_inches='tight')
    
    # Print the overall importance of residual analysis
    print("\nWhy Residual Analysis is Essential:")
    print("1. It verifies that model assumptions are met, ensuring reliable inference")
    print("2. It helps identify misspecification in the model (e.g., missing predictors, wrong functional form)")
    print("3. It guides model improvement and transformation decisions")
    print("4. It helps detect outliers and influential observations")
    print("5. It ensures that p-values, confidence intervals, and predictions are trustworthy")
    print("\nWithout proper residual analysis, we might make incorrect conclusions or predictions!")
    
    result = {
        'statement': "Residual analysis is essential to ensure that the assumptions of a linear regression model are met.",
        'is_true': True,
        'explanation': "This statement is TRUE. Residual analysis is a crucial step in linear regression to ensure that the model's assumptions are met. By examining the residuals (the differences between the actual and predicted values), we can assess whether key assumptions like linearity, independence, homoscedasticity (constant variance), and normality are satisfied. The visualizations demonstrate how residual plots can reveal patterns that indicate assumption violations. When assumptions are met, residuals should be randomly scattered around zero with no discernible pattern, have constant variance, and follow a normal distribution. Violations of these assumptions can lead to unreliable inference, predictions, and confidence intervals.",
        'image_path': ['statement5_residual_comparison.png', 'statement5_qq_plots.png', 'statement5_assumption_violations.png', 'statement5_comprehensive_residuals.png']
    }
    
    return result

if __name__ == "__main__":
    result = statement5_residual_analysis()
    print(f"\nStatement: {result['statement']}")
    print(f"True or False: {'True' if result['is_true'] else 'False'}")
    print(f"Explanation: {result['explanation']}")
    print(f"Images saved: {', '.join(result['image_path'])}") 