import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
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

def statement3_high_bias():
    """
    Statement 3: A model with high bias will typically show a large gap 
    between training and test error.
    """
    print("\n==== Statement 3: High Bias and Error Gap ====")
    
    # Print explanation of bias-variance tradeoff
    print("\nBias-Variance Tradeoff Explanation:")
    print("High bias (underfitting):")
    print("- Model is too simple to capture the underlying pattern")
    print("- High training error (unable to fit training data well)")
    print("- High test error (for the same reason)")
    print("- Small gap between training and test error (both are high)")
    
    print("\nHigh variance (overfitting):")
    print("- Model is too complex and captures noise in the training data")
    print("- Low training error (fits training data extremely well)")
    print("- High test error (doesn't generalize well to new data)")
    print("- Large gap between training and test error")
    
    # Generate complex, nonlinear data
    np.random.seed(42)
    n_samples = 100
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    
    # True function with high nonlinearity
    true_func = lambda x: 3 + 2*x + 0.5*x**2 - 0.1*x**3 + 0.01*x**4
    y = true_func(X.squeeze()) + np.random.normal(0, 5, n_samples)
    
    # Create train-test split
    train_size = 70
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train models with different complexities
    degrees = [1, 2, 5, 15]  # Low, medium, high complexity
    model_complexities = ["Very Low (High Bias)", "Low", "Medium", "High (Low Bias)"]
    train_errors = []
    test_errors = []
    error_gaps = []
    models = []
    
    for degree in degrees:
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression())
        ])
        model.fit(X_train, y_train)
        models.append(model)
        
        # Predict and calculate errors
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        gap = test_mse - train_mse
        gap_ratio = test_mse / train_mse if train_mse > 0 else float('inf')
        
        train_errors.append(train_mse)
        test_errors.append(test_mse)
        error_gaps.append(gap)
        
        print(f"\nPolynomial Degree {degree} ({model_complexities[degrees.index(degree)]}):")
        print(f"  Training MSE: {train_mse:.2f}")
        print(f"  Test MSE: {test_mse:.2f}")
        print(f"  Gap (Test - Train): {gap:.2f}")
        print(f"  Ratio (Test/Train): {gap_ratio:.2f}x")
    
    # Print summary of results and explicitly state the misconception
    print("\nSummary of Results:")
    print("1. High Bias Model (Degree 1):")
    print(f"   - Gap: {error_gaps[0]:.2f}")
    print(f"   - The gap is {'large' if error_gaps[0] > 100 else 'relatively small'} compared to the error magnitudes")
    
    print("\n2. High Variance Model (Degree 15):")
    print(f"   - Gap: {error_gaps[3]:.2f}")
    print(f"   - The gap is {'extremely large' if error_gaps[3] > 10000 else 'large'} compared to the training error")
    
    print("\nMisconception Explanation:")
    print("The statement suggests that high-bias models have large error gaps, but this is incorrect.")
    print("High-bias models typically have small gaps relative to error magnitude, because:")
    print("- They underfit both training and test data similarly")
    print("- They don't memorize the training data, so train and test performance are similarly poor")
    print("- The bias dominates both errors, leading to a smaller gap")
    
    # Plot 1: Bar chart of train/test errors (simplified)
    plt.figure(figsize=(10, 6))
    
    # Set up bar positions
    bar_width = 0.35
    index = np.arange(len(degrees))
    
    # Check if degree 15 errors are extremely large and need log scale
    use_log_scale = max(train_errors) > 1000 or max(test_errors) > 1000
    
    # If errors are huge, use log scale
    if use_log_scale:
        plt.bar(index - bar_width/2, train_errors, bar_width, label='Training Error', color='blue', alpha=0.7, log=True)
        plt.bar(index + bar_width/2, test_errors, bar_width, label='Test Error', color='red', alpha=0.7, log=True)
        plt.ylabel('Mean Squared Error (log scale)', fontsize=12)
    else:
        plt.bar(index - bar_width/2, train_errors, bar_width, label='Training Error', color='blue', alpha=0.7)
        plt.bar(index + bar_width/2, test_errors, bar_width, label='Test Error', color='red', alpha=0.7)
        plt.ylabel('Mean Squared Error', fontsize=12)
    
    # Add labels with gaps
    for i, (train_err, test_err) in enumerate(zip(train_errors, test_errors)):
        gap = test_err - train_err
        plt.text(i, max(train_err, test_err) * 1.1, f'Gap: {gap:.1f}', 
                ha='center', fontweight='bold')
    
    plt.xlabel('Model Complexity (Polynomial Degree)', fontsize=12)
    plt.title('Training and Test Error by Model Complexity', fontsize=14)
    plt.xticks(index, [f'Degree {d}\n{c}' for d, c in zip(degrees, model_complexities)])
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement3_train_test_errors.png'), dpi=300, bbox_inches='tight')
    
    # Plot 2: Visual explanation of bias and variance
    plt.figure(figsize=(10, 6))
    
    # Plot the data points
    x_plot = np.linspace(0, 10, 1000).reshape(-1, 1)
    plt.scatter(X_train, y_train, color='gray', alpha=0.3, label='Training data')
    plt.plot(x_plot, true_func(x_plot.squeeze()), color='black', linewidth=2, label='True function')
    
    # Plot the high bias and high variance models
    high_bias_model = models[0]  # Degree 1
    high_variance_model = models[3]  # Degree 15
    
    y_high_bias = high_bias_model.predict(x_plot)
    y_high_variance = high_variance_model.predict(x_plot)
    
    plt.plot(x_plot, y_high_bias, color='blue', linewidth=2, label='High Bias Model (Degree 1)')
    plt.plot(x_plot, y_high_variance, color='red', linewidth=2, label='High Variance Model (Degree 15)')
    
    # Add annotations
    plt.annotate('High Bias: Underfits\nSmall gap between\ntrain and test error', 
                xy=(7, y_high_bias.flatten()[700]), 
                xytext=(7, y_high_bias.flatten()[700] - 15),
                arrowprops=dict(arrowstyle='->'), fontsize=10, ha='center')
    
    plt.annotate('High Variance: Overfits\nLarge gap between\ntrain and test error', 
                xy=(3, y_high_variance.flatten()[300]), 
                xytext=(3, y_high_variance.flatten()[300] + 15),
                arrowprops=dict(arrowstyle='->'), fontsize=10, ha='center')
    
    plt.xlabel('X', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('High Bias vs. High Variance Models', fontsize=14)
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(save_dir, 'statement3_bias_variance_models.png'), dpi=300, bbox_inches='tight')
    
    # Plot 3: New visualization - Train-Test gap ratio
    plt.figure(figsize=(10, 6))
    
    # Calculate the gap ratios
    gap_ratios = [test_err/train_err if train_err > 0 else float('inf') for train_err, test_err in zip(train_errors, test_errors)]
    
    # Create a bar chart of the gap ratios
    bars = plt.bar(index, gap_ratios, color=['blue', 'green', 'orange', 'red'])
    
    # Add values on top of bars
    for i, ratio in enumerate(gap_ratios):
        if ratio > 1000:  # Handle extremely large ratios
            plt.text(i, min(ratio, 100), f'{ratio:.1e}x', ha='center', va='bottom', fontweight='bold')
        else:
            plt.text(i, ratio, f'{ratio:.1f}x', ha='center', va='bottom', fontweight='bold')
    
    # Add a horizontal line at y=1 for reference (where test error = training error)
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Test Error = Training Error')
    
    plt.xlabel('Model Complexity (Polynomial Degree)', fontsize=12)
    plt.ylabel('Test/Train Error Ratio', fontsize=12)
    plt.title('Test/Train Error Ratio by Model Complexity', fontsize=14)
    plt.xticks(index, [f'Degree {d}\n{c}' for d, c in zip(degrees, model_complexities)])
    
    # Use log scale if any ratio is extremely large
    if max(gap_ratios) > 1000:
        plt.yscale('log')
        plt.ylabel('Test/Train Error Ratio (log scale)', fontsize=12)
    
    plt.grid(True)
    plt.legend()
    
    # Add LaTeX formula annotation
    plt.annotate(r'$\text{Gap Ratio} = \frac{\text{Test Error}}{\text{Training Error}}$', 
                xy=(0.05, 0.92), xycoords='axes fraction', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement3_gap_ratio.png'), dpi=300, bbox_inches='tight')
    
    # NEW VISUALIZATION: Comprehensive bias-variance trade-off visualization
    plt.figure(figsize=(12, 10))
    
    # Generate a new dataset with more models to show the full trade-off
    more_degrees = np.array([1, 2, 3, 4, 5, 7, 10, 15, 20])
    more_train_errors = []
    more_test_errors = []
    
    # Generate true function values (no noise) for bias calculation
    y_true = true_func(X.squeeze())
    y_true_train = y_true[:train_size]
    y_true_test = y_true[train_size:]
    
    # Train models and calculate errors
    for degree in more_degrees:
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression())
        ])
        model.fit(X_train, y_train)
        
        # Predict and calculate errors
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        more_train_errors.append(train_mse)
        more_test_errors.append(test_mse)
    
    # Create a figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Training and test errors vs model complexity (top left)
    ax = axes[0, 0]
    ax.plot(more_degrees, more_train_errors, 'o-', color='blue', label='Training error')
    ax.plot(more_degrees, more_test_errors, 'o-', color='red', label='Test error')
    
    # Highlight the gap
    for i, degree in enumerate(more_degrees):
        if i % 2 == 0:  # Skip some to avoid clutter
            ax.plot([degree, degree], 
                   [more_train_errors[i], more_test_errors[i]], 
                   'k--', alpha=0.3)
            
    ax.set_xlabel('Model Complexity (Polynomial Degree)')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('Error vs. Model Complexity')
    ax.legend()
    ax.grid(True)
    
    # Add annotations for bias and variance
    ax.annotate('High Bias\nSmall Gap', xy=(1.5, np.mean([more_train_errors[0], more_test_errors[0]])),
                xytext=(2, more_test_errors[0] * 1.2),
                arrowprops=dict(arrowstyle='->'), fontsize=9)
    
    ax.annotate('High Variance\nLarge Gap', xy=(15, np.mean([more_train_errors[-2], more_test_errors[-2]])),
                xytext=(10, more_test_errors[-2] * 0.5),
                arrowprops=dict(arrowstyle='->'), fontsize=9)
    
    # Plot 2: Error gap vs model complexity (top right)
    ax = axes[0, 1]
    error_gaps = [test - train for train, test in zip(more_train_errors, more_test_errors)]
    ax.plot(more_degrees, error_gaps, 'o-', color='purple')
    
    # Add reference line at 0 (no gap)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    ax.set_xlabel('Model Complexity (Polynomial Degree)')
    ax.set_ylabel('Error Gap (Test - Train)')
    ax.set_title('Error Gap vs. Model Complexity')
    
    # Add annotations
    ax.annotate('High Bias Region\nSmall Gap', xy=(2, error_gaps[1]),
                xytext=(2, error_gaps[1] + 0.2 * max(error_gaps)),
                arrowprops=dict(arrowstyle='->'), fontsize=9)
    
    ax.annotate('High Variance Region\nLarge Gap', xy=(15, error_gaps[-2]),
                xytext=(12, error_gaps[-2] * 0.8),
                arrowprops=dict(arrowstyle='->'), fontsize=9)
    
    ax.grid(True)
    
    # Plot 3: Model fits (bottom left)
    ax = axes[1, 0]
    
    # Plot the data and true function
    ax.scatter(X_train, y_train, color='gray', alpha=0.3, label='Training data')
    ax.plot(x_plot, true_func(x_plot.squeeze()), color='black', linewidth=2, label='True function')
    
    # Plot high bias model
    high_bias_model = Pipeline([
        ('poly', PolynomialFeatures(degree=1)),
        ('linear', LinearRegression())
    ]).fit(X_train, y_train)
    y_high_bias = high_bias_model.predict(x_plot)
    ax.plot(x_plot, y_high_bias, color='blue', linewidth=2, label='High Bias (Degree 1)')
    
    # Plot high variance model
    high_variance_model = Pipeline([
        ('poly', PolynomialFeatures(degree=15)),
        ('linear', LinearRegression())
    ]).fit(X_train, y_train)
    y_high_variance = high_variance_model.predict(x_plot)
    ax.plot(x_plot, y_high_variance, color='red', linewidth=2, label='High Variance (Degree 15)')
    
    # Plot optimal model
    optimal_model = Pipeline([
        ('poly', PolynomialFeatures(degree=4)),  # Assume degree 4 is optimal
        ('linear', LinearRegression())
    ]).fit(X_train, y_train)
    y_optimal = optimal_model.predict(x_plot)
    ax.plot(x_plot, y_optimal, color='green', linewidth=2, label='Optimal (Degree 4)')
    
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title('Model Fits: Bias vs. Variance')
    ax.legend(fontsize=8)
    ax.grid(True)
    
    # Plot 4: Bias-Variance conceptual diagram (bottom right)
    ax = axes[1, 1]
    
    # Generate data for conceptual bias-variance tradeoff curve
    x_concept = np.linspace(0, 10, 100)
    bias_curve = 50 * np.exp(-0.5 * x_concept)
    variance_curve = 3 * np.exp(0.3 * x_concept)
    total_error = bias_curve + variance_curve
    
    # Plot the curves
    ax.plot(x_concept, bias_curve, 'b-', label='Bias²')
    ax.plot(x_concept, variance_curve, 'r-', label='Variance')
    ax.plot(x_concept, total_error, 'k-', linewidth=2, label='Total Error')
    
    # Add optimal point
    optimal_idx = np.argmin(total_error)
    ax.scatter([x_concept[optimal_idx]], [total_error[optimal_idx]], color='green', s=100, 
              zorder=5, label='Optimal Complexity')
    
    # Add regions
    ax.fill_between([0, x_concept[optimal_idx]], 0, 100, color='blue', alpha=0.1, label='High Bias Region')
    ax.fill_between([x_concept[optimal_idx], 10], 0, 100, color='red', alpha=0.1, label='High Variance Region')
    
    # Add text
    ax.text(1, 60, 'High Bias Region\n- Simple Models\n- Small Gap\n- Underfitting',
           fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
    
    ax.text(7, 60, 'High Variance Region\n- Complex Models\n- Large Gap\n- Overfitting',
           fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Model Complexity')
    ax.set_ylabel('Error')
    ax.set_title('Bias-Variance Tradeoff')
    ax.legend(loc='upper center', fontsize=8)
    ax.grid(True)
    
    # Add title for the whole figure
    plt.suptitle('The Bias-Variance Tradeoff and Error Gap', fontsize=16, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_dir, 'statement3_bias_variance_tradeoff.png'), dpi=300, bbox_inches='tight')
    
    result = {
        'statement': "A model with high bias will typically show a large gap between training and test error.",
        'is_true': False,
        'explanation': "A model with high bias (underfitting) typically shows a SMALL gap between training and test error, not a large one. High-bias models perform poorly on both training and test data. In contrast, high-variance models (overfitting) show a LARGE gap between training and test error, with low training error but high test error. The visualization shows that the simplest model (Degree 1) has high bias but a small train-test error gap, while the most complex model (Degree 15) has low bias but high variance, resulting in a much larger error gap.",
        'image_path': ['statement3_train_test_errors.png', 'statement3_bias_variance_models.png', 'statement3_gap_ratio.png', 'statement3_bias_variance_tradeoff.png']
    }
    
    return result

if __name__ == "__main__":
    result = statement3_high_bias()
    print(f"\nStatement: {result['statement']}")
    print(f"True or False: {'True' if result['is_true'] else 'False'}")
    print(f"Explanation: {result['explanation']}")
    print(f"Images saved: {', '.join(result['image_path'])}") 