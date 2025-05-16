import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression
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

def statement8_learning_curves():
    """
    Statement 8: In learning curves, if the validation error continues to decrease as more 
    training samples are added, adding more data is likely to improve model performance.
    """
    print("\n==== Statement 8: Learning Curves and Model Improvement ====")
    
    # Print detailed explanation of learning curves
    print("\nLearning Curves Explained:")
    print("- Learning curves show model error as a function of training set size")
    print("- They help diagnose bias, variance, and the effect of dataset size")
    print("- Typically show both training and validation error rates")
    print("- The gap between training and validation curves indicates variance")
    print("- The absolute error level indicates bias")
    
    print("\nWhen More Data Helps:")
    print("1. When validation error is STILL DECREASING as training set size increases")
    print("2. When validation error hasn't plateaued")
    print("3. When the gap between training and validation error isn't too large")
    
    print("\nWhen More Data May Not Help:")
    print("1. When validation error has PLATEAUED (flat curve)")
    print("2. When both training and validation errors are high (high bias/underfitting)")
    print("3. When there's a large gap that doesn't decrease with more data (high variance/overfitting)")
    
    # Generate different dataset scenarios
    np.random.seed(42)
    
    # Scenario 1: Model benefits from more data (validation error still decreasing)
    X1, y1 = make_regression(n_samples=1000, n_features=20, n_informative=10, noise=30, random_state=42)
    
    # Scenario 2: Model has plateaued (validation error has flattened)
    X2, y2 = make_regression(n_samples=1000, n_features=5, n_informative=3, noise=10, random_state=42)
    
    # Scenario 3: Complex dataset with insufficient model capacity (high bias)
    X3 = np.linspace(0, 10, 1000).reshape(-1, 1)
    y3 = 3 * np.sin(X3.squeeze()) + 2 * np.cos(2 * X3.squeeze()) + np.random.normal(0, 0.5, 1000)
    
    datasets = [
        ('Scenario 1: Model benefits from more data', X1, y1, Ridge(), 'blue'),
        ('Scenario 2: Model has plateaued', X2, y2, Ridge(), 'green'),
        ('Scenario 3: High bias (underfitting)', X3, y3, Ridge(), 'red')
    ]
    
    # Store results for each scenario
    scenario_results = []
    
    for name, X, y, model, color in datasets:
        # Calculate learning curves
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        
        # Convert to positive MSE
        train_scores_mean = -np.mean(train_scores, axis=1)
        val_scores_mean = -np.mean(val_scores, axis=1)
        
        # Calculate slope of validation curve at the end
        end_idx = len(val_scores_mean) - 1
        start_idx = max(0, end_idx - 3)  # Look at the last few points
        
        # Calculate slope using linear regression on the last few points
        x_points = train_sizes[start_idx:end_idx+1]
        y_points = val_scores_mean[start_idx:end_idx+1]
        slope, _ = np.polyfit(x_points, y_points, 1)
        
        # Calculate error ratio and gap
        error_ratio = val_scores_mean[-1] / train_scores_mean[-1] if train_scores_mean[-1] > 0 else float('inf')
        error_gap = val_scores_mean[-1] - train_scores_mean[-1]
        
        # Determine trend
        if slope < -0.01 * val_scores_mean[start_idx]:  # Decreasing significantly
            trend = "Decreasing"
            more_data_help = "Likely to help"
        elif abs(slope) < 0.01 * val_scores_mean[start_idx]:  # Flat (close to zero slope)
            trend = "Plateaued"
            more_data_help = "Unlikely to help"
        else:  # Slope is positive (unusual)
            trend = "Increasing"
            more_data_help = "Will not help (investigate issues)"
        
        # Store results
        scenario_results.append({
            'name': name,
            'train_sizes': train_sizes,
            'train_scores_mean': train_scores_mean,
            'val_scores_mean': val_scores_mean,
            'color': color,
            'slope': slope,
            'trend': trend,
            'more_data_help': more_data_help,
            'error_gap': error_gap,
            'error_ratio': error_ratio
        })
    
    # Print analysis for each scenario
    print("\nScenario Analysis:")
    for result in scenario_results:
        print(f"\n{result['name']}:")
        print(f"  - Final training error: {result['train_scores_mean'][-1]:.2f}")
        print(f"  - Final validation error: {result['val_scores_mean'][-1]:.2f}")
        print(f"  - Error gap: {result['error_gap']:.2f} (ratio: {result['error_ratio']:.2f}x)")
        print(f"  - Validation curve trend: {result['trend']}")
        print(f"  - Will more data help? {result['more_data_help']}")
    
    # Plot 1: Learning curves for all scenarios
    plt.figure(figsize=(12, 8))
    
    # Set up colors for consistent labeling
    colors = {'train': 'darkblue', 'val1': 'blue', 'val2': 'green', 'val3': 'red'}
    
    # Plot training error curves
    for i, result in enumerate(scenario_results):
        # Only label the first training curve to avoid legend duplication
        if i == 0:
            plt.plot(result['train_sizes'], result['train_scores_mean'], 'o-', color=colors['train'], 
                    alpha=0.7, label='Training error')
        else:
            plt.plot(result['train_sizes'], result['train_scores_mean'], 'o-', color=colors['train'], 
                    alpha=0.7)
    
    # Plot validation error curves
    for i, result in enumerate(scenario_results):
        label = f"Validation error: {result['trend']}"
        plt.plot(result['train_sizes'], result['val_scores_mean'], 'o-', color=result['color'], 
                label=label)
        
        # Add projected trend
        x_extend = np.linspace(result['train_sizes'][-1], result['train_sizes'][-1] * 1.3, 10)
        y_extend = result['slope'] * (x_extend - result['train_sizes'][-1]) + result['val_scores_mean'][-1]
        plt.plot(x_extend, y_extend, linestyle='--', color=result['color'], alpha=0.7)
    
    # Add vertical line at last training size
    plt.axvline(x=scenario_results[0]['train_sizes'][-1], color='gray', linestyle='--', 
               label='Current data size')
    
    plt.title('Learning Curves: Effect of Training Set Size', fontsize=14)
    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True)
    
    # Add LaTeX annotation explaining the concept
    plt.figtext(0.5, 0.01, r'$\text{If } \frac{d\text{ValidationError}}{d\text{TrainingSize}} < 0 \text{ at current size} \Rightarrow \text{More data will help}$', 
                ha='center', fontsize=12)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(save_dir, 'statement8_learning_curves.png'), dpi=300, bbox_inches='tight')
    
    # Plot 2: Pattern recognition guide for learning curves
    plt.figure(figsize=(12, 10))
    
    # Create example learning curve patterns
    n_points = 10
    train_sizes = np.linspace(100, 1000, n_points)
    
    # Pattern 1: High bias (underfitting) - both errors high, small gap
    train_error1 = 70 - 10 * np.log(train_sizes/100)
    val_error1 = 80 - 8 * np.log(train_sizes/100)
    
    # Pattern 2: High variance (overfitting) - low training, high validation
    train_error2 = 10 * np.ones_like(train_sizes)
    val_error2 = 60 - 5 * np.log(train_sizes/100)
    
    # Pattern 3: Needs more data - both decreasing, validation not plateaued
    train_error3 = 40 * np.exp(-0.001 * train_sizes)
    val_error3 = 80 * np.exp(-0.0008 * train_sizes)
    
    # Pattern 4: Optimal capacity - both errors plateaued with small gap
    train_error4 = 20 * np.ones_like(train_sizes)
    val_error4 = np.concatenate([30 - 5 * np.log(train_sizes[:6]/100), 25 * np.ones(n_points-6)])
    
    # Create 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Titles, descriptions and colors
    patterns = [
        "Pattern 1: High Bias (Underfitting)",
        "Pattern 2: High Variance (Overfitting)",
        "Pattern 3: Needs More Data",
        "Pattern 4: Optimal Model Capacity"
    ]
    
    descriptions = [
        "Both errors high with small gap\nAction: Increase model complexity",
        "Low training error, high validation error\nAction: Add regularization or reduce complexity",
        "Both errors decreasing, validation not plateaued\nAction: Collect more training data",
        "Both errors plateaued with small gap\nAction: Try different model architecture"
    ]
    
    colors = ['red', 'blue', 'green', 'purple']
    train_errors = [train_error1, train_error2, train_error3, train_error4]
    val_errors = [val_error1, val_error2, val_error3, val_error4]
    
    # Plot each pattern
    for i in range(4):
        ax = axes[i]
        
        # Plot curves
        ax.plot(train_sizes, train_errors[i], 'o-', color='darkblue', label='Training error')
        ax.plot(train_sizes, val_errors[i], 'o-', color=colors[i], label='Validation error')
        
        # Add current data size line
        ax.axvline(x=train_sizes[-1], color='gray', linestyle='--')
        
        # Add future projection region
        extended_sizes = np.linspace(train_sizes[-1], train_sizes[-1] * 1.3, 5)
        ax.fill_between([train_sizes[-1], extended_sizes[-1]], 0, 100, 
                       color='lightgray', alpha=0.2)
        
        # Annotation for projection region
        ax.text(extended_sizes[-1] - 50, 90, 'Future\ndata', ha='right', fontsize=8, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Will more data help?
        if i == 2:  # Needs more data pattern
            will_help = "YES - More data will help"
            color = 'green'
        else:
            will_help = "NO - Other actions needed first"
            color = 'red'
        
        # Add annotation
        ax.text(0.5, 0.05, will_help, transform=ax.transAxes, ha='center',
               color=color, fontweight='bold', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add pattern description
        ax.text(0.05, 0.95, descriptions[i], transform=ax.transAxes, ha='left', va='top',
               fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(patterns[i], fontsize=12, color=colors[i])
        ax.set_xlabel('Training Set Size', fontsize=10)
        ax.set_ylabel('Error', fontsize=10)
        ax.grid(True)
        ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    # Add main title
    plt.suptitle('Learning Curve Patterns and When More Data Helps', fontsize=16, y=1.02)
    
    # Add LaTeX formula at the bottom explaining how to diagnose bias vs variance
    plt.figtext(0.5, 0.01, 
                r'$\text{High Bias: High } E_{train} \approx E_{val}$ vs '
                r'$\text{High Variance: Low } E_{train} \ll E_{val}$ vs '
                r'$\text{Need More Data: Decreasing } \frac{dE_{val}}{d\text{size}} < 0$', 
                ha='center', fontsize=12)
    
    plt.savefig(os.path.join(save_dir, 'statement8_learning_curve_patterns.png'), dpi=300, bbox_inches='tight')
    
    # Print key conclusions about when more data helps
    print("\nKey Conclusions:")
    print("1. More data helps when validation error is still decreasing")
    print("2. More data helps when the model isn't yet overfitting or underfitting")
    print("3. Learning curves help diagnose why a model isn't performing well")
    print("4. High bias (underfitting) and high variance (overfitting) should be addressed before collecting more data")
    print("5. The slope of the validation curve at the current training size is the key indicator")
    
    result = {
        'statement': "In learning curves, if the validation error continues to decrease as more training samples are added, adding more data is likely to improve model performance.",
        'is_true': True,
        'explanation': "This statement is TRUE. When the validation error continues to decrease as the training set size increases, it indicates that the model is still learning from additional data and has not yet reached its full capacity. In this scenario, collecting and adding more training data is likely to further improve model performance. The analysis of our learning curve scenarios confirms this: when the validation curve shows a negative slope at the current training size, more data will help; when it has plateaued, other actions like addressing bias or variance should be prioritized first.",
        'image_path': ['statement8_learning_curves.png', 'statement8_learning_curve_patterns.png']
    }
    
    return result

if __name__ == "__main__":
    result = statement8_learning_curves()
    print(f"\nStatement: {result['statement']}")
    print(f"True or False: {'True' if result['is_true'] else 'False'}")
    print(f"Explanation: {result['explanation']}")
    print(f"Images saved: {', '.join(result['image_path'])}") 