import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.model_selection import train_test_split

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
    Statement 8: In learning curves, if the validation error continues to decrease 
    as more training samples are added, adding more data is likely to improve 
    model performance.
    """
    print("\n==== Statement 8: Learning Curves and Model Improvement with More Data ====")
    
    # Generate two datasets with different characteristics to demonstrate different scenarios
    
    # Dataset 1: Linear pattern with moderate noise - model will converge with sufficient data
    np.random.seed(42)
    n_samples = 1000
    X1 = np.linspace(0, 10, n_samples).reshape(-1, 1)
    y1 = 2 * X1.squeeze() + 1 + np.random.normal(0, 1, n_samples)
    
    # Create a train/test split for evaluation
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
    
    # Dataset 2: Nonlinear pattern with high noise - model will benefit from more data
    X2 = np.linspace(0, 10, n_samples).reshape(-1, 1)
    y2 = 0.5 * X2.squeeze()**2 + 3 * X2.squeeze() + np.random.normal(0, 5, n_samples)
    
    # Create a train/test split for evaluation
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
    
    # For Dataset 1, calculate learning curves with LinearRegression
    train_sizes1, train_scores1, val_scores1 = learning_curve(
        LinearRegression(), X1_train, y1_train, 
        train_sizes=np.linspace(0.1, 1.0, 10), 
        cv=5, scoring='neg_mean_squared_error')
    
    # For Dataset 2, calculate learning curves with RandomForestRegressor
    train_sizes2, train_scores2, val_scores2 = learning_curve(
        RandomForestRegressor(n_estimators=100, random_state=42), X2_train, y2_train, 
        train_sizes=np.linspace(0.1, 1.0, 10), 
        cv=5, scoring='neg_mean_squared_error')
    
    # Prepare the scores for plotting (convert to positive MSE)
    train_mse1 = -np.mean(train_scores1, axis=1)
    val_mse1 = -np.mean(val_scores1, axis=1)
    train_mse2 = -np.mean(train_scores2, axis=1)
    val_mse2 = -np.mean(val_scores2, axis=1)
    
    # Print learning curve information
    print("\nLearning Curve Analysis for Dataset 1 (Simple Linear Pattern):")
    print("Training sizes:", train_sizes1)
    print("Initial validation MSE (few samples):", val_mse1[0])
    print("Final validation MSE (maximum samples):", val_mse1[-1])
    print("Improvement with more data:", (val_mse1[0] - val_mse1[-1]) / val_mse1[0] * 100, "%")
    
    # Check if the validation error is still decreasing or has plateaued
    improvement_rate1 = (val_mse1[-2] - val_mse1[-1]) / val_mse1[-2] * 100
    print(f"Recent improvement rate: {improvement_rate1:.4f}%")
    if improvement_rate1 < 1:
        print("The learning curve has plateaued (< 1% improvement).")
        print("Adding more data is unlikely to significantly improve model performance.")
    else:
        print("The learning curve is still showing improvements.")
        print("Adding more data may continue to improve model performance.")
    
    print("\nLearning Curve Analysis for Dataset 2 (Complex Nonlinear Pattern):")
    print("Training sizes:", train_sizes2)
    print("Initial validation MSE (few samples):", val_mse2[0])
    print("Final validation MSE (maximum samples):", val_mse2[-1])
    print("Improvement with more data:", (val_mse2[0] - val_mse2[-1]) / val_mse2[0] * 100, "%")
    
    # Check if the validation error is still decreasing or has plateaued
    improvement_rate2 = (val_mse2[-2] - val_mse2[-1]) / val_mse2[-2] * 100
    print(f"Recent improvement rate: {improvement_rate2:.4f}%")
    if improvement_rate2 < 1:
        print("The learning curve has plateaued (< 1% improvement).")
        print("Adding more data is unlikely to significantly improve model performance.")
    else:
        print("The learning curve is still showing improvements.")
        print("Adding more data may continue to improve model performance.")
    
    # Plot 1: Learning curves for Dataset 1
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes1, train_mse1, 'o-', color='blue', label='Training error')
    plt.plot(train_sizes1, val_mse1, 'o-', color='red', label='Validation error')
    
    # Add trend line to see where validation error is going
    from scipy.optimize import curve_fit
    
    def exponential_decay(x, a, b, c):
        return a * np.exp(-b * x) + c
    
    try:
        # Fit an exponential decay curve to validation error
        popt, _ = curve_fit(exponential_decay, train_sizes1, val_mse1, p0=[1, 0.01, 1])
        
        # Project future performance with more data
        future_sizes = np.linspace(train_sizes1[-1], train_sizes1[-1] * 2, 10)
        future_val_mse = exponential_decay(future_sizes, *popt)
        
        # Plot the projection
        plt.plot(future_sizes, future_val_mse, '--', color='red', alpha=0.5, label='Projected validation error')
    except:
        print("Could not fit exponential decay to validation error for Dataset 1")
    
    plt.title('Learning Curves: Simple Linear Dataset', fontsize=14)
    plt.xlabel('Number of training samples', fontsize=12)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # Annotate convergence point
    plt.annotate('Convergence: Low Error Gap', 
                xy=(train_sizes1[-1], (train_mse1[-1] + val_mse1[-1])/2),
                xytext=(train_sizes1[-1]*0.6, val_mse1[0]*0.7),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=12,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Add interpretive text
    plt.figtext(0.5, 0.01, 
               "Analysis: The validation error has plateaued and is close to the training error.\nAdding more data is unlikely to significantly improve model performance.",
               ha="center", fontsize=12, 
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(save_dir, 'statement8_learning_curve_convergent.png'), dpi=300, bbox_inches='tight')
    
    # Plot 2: Learning curves for Dataset 2
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes2, train_mse2, 'o-', color='blue', label='Training error')
    plt.plot(train_sizes2, val_mse2, 'o-', color='red', label='Validation error')
    
    try:
        # Fit an exponential decay curve to validation error
        popt, _ = curve_fit(exponential_decay, train_sizes2, val_mse2)
        
        # Project future performance with more data
        future_sizes = np.linspace(train_sizes2[-1], train_sizes2[-1] * 2, 10)
        future_val_mse = exponential_decay(future_sizes, *popt)
        
        # Plot the projection
        plt.plot(future_sizes, future_val_mse, '--', color='red', alpha=0.5, label='Projected validation error')
    except:
        print("Could not fit exponential decay to validation error for Dataset 2")
    
    plt.title('Learning Curves: Complex Nonlinear Dataset', fontsize=14)
    plt.xlabel('Number of training samples', fontsize=12)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # Annotate divergence point
    plt.annotate('Still Improving: Validation Error Decreasing', 
                xy=(train_sizes2[-1], val_mse2[-1]),
                xytext=(train_sizes2[-1]*0.6, val_mse2[-1]*1.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=12,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Add interpretive text
    plt.figtext(0.5, 0.01, 
               "Analysis: The validation error continues to decrease as training samples increase.\nAdding more data is likely to improve model performance.",
               ha="center", fontsize=12, 
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(save_dir, 'statement8_learning_curve_improving.png'), dpi=300, bbox_inches='tight')
    
    # Add comparison plot showing different learning curve scenarios
    plt.figure(figsize=(12, 8))
    
    # Create a grid of 4 scenarios
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Add titles and labels
    axes[0, 0].set_title('Scenario 1: High Bias (Underfitting)', fontsize=12)
    axes[0, 1].set_title('Scenario 2: Optimal Model', fontsize=12)
    axes[1, 0].set_title('Scenario 3: High Variance (Overfitting)', fontsize=12)
    axes[1, 1].set_title('Scenario 4: Still Learning', fontsize=12)
    
    for ax in axes.flatten():
        ax.set_xlabel('Training Set Size', fontsize=10)
        ax.set_ylabel('Error', fontsize=10)
        ax.grid(True)
    
    # Scenario 1: High Bias (Underfitting)
    train_sizes = np.linspace(0.1, 1.0, 100)
    train_error = np.ones_like(train_sizes) * 0.6 + np.random.normal(0, 0.03, 100)
    val_error = np.ones_like(train_sizes) * 0.7 + np.random.normal(0, 0.05, 100)
    
    axes[0, 0].plot(train_sizes, train_error, color='blue', label='Training Error')
    axes[0, 0].plot(train_sizes, val_error, color='red', label='Validation Error')
    axes[0, 0].legend(fontsize=9)
    
    # Add annotation
    axes[0, 0].text(0.5, 0.7, "• Both errors high\n• Small gap between errors\n• Both plateau early\n\n➔ More data won't help\n➔ Need a more complex model", 
            transform=axes[0, 0].transAxes, fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Add more data projection
    axes[0, 0].plot(np.linspace(1.0, 1.5, 20), 
                   np.ones(20) * train_error[-1] + np.random.normal(0, 0.01, 20), 
                   '--', color='blue', alpha=0.5)
    axes[0, 0].plot(np.linspace(1.0, 1.5, 20), 
                   np.ones(20) * val_error[-1] + np.random.normal(0, 0.01, 20), 
                   '--', color='red', alpha=0.5)
    
    # Scenario 2: Optimal Model
    train_sizes = np.linspace(0.1, 1.0, 100)
    train_error = 0.5 * np.exp(-2*train_sizes) + 0.1 + np.random.normal(0, 0.01, 100)
    val_error = 0.7 * np.exp(-1.8*train_sizes) + 0.15 + np.random.normal(0, 0.02, 100)
    
    axes[0, 1].plot(train_sizes, train_error, color='blue', label='Training Error')
    axes[0, 1].plot(train_sizes, val_error, color='red', label='Validation Error')
    axes[0, 1].legend(fontsize=9)
    
    # Add annotation
    axes[0, 1].text(0.5, 0.7, "• Both errors low\n• Small gap between errors\n• Both have converged\n\n➔ More data won't help significantly\n➔ Model is well-tuned", 
            transform=axes[0, 1].transAxes, fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Add more data projection
    axes[0, 1].plot(np.linspace(1.0, 1.5, 20), 
                   np.ones(20) * train_error[-1] + np.random.normal(0, 0.01, 20), 
                   '--', color='blue', alpha=0.5)
    axes[0, 1].plot(np.linspace(1.0, 1.5, 20), 
                   np.ones(20) * val_error[-1] + np.random.normal(0, 0.01, 20), 
                   '--', color='red', alpha=0.5)
    
    # Scenario 3: High Variance (Overfitting)
    train_sizes = np.linspace(0.1, 1.0, 100)
    train_error = 0.3 * np.exp(-3*train_sizes) + 0.05 + np.random.normal(0, 0.01, 100)
    val_error = 0.5 * np.exp(-0.5*train_sizes) + 0.3 + np.random.normal(0, 0.03, 100)
    
    axes[1, 0].plot(train_sizes, train_error, color='blue', label='Training Error')
    axes[1, 0].plot(train_sizes, val_error, color='red', label='Validation Error')
    axes[1, 0].legend(fontsize=9)
    
    # Add annotation
    axes[1, 0].text(0.5, 0.7, "• Low training error\n• High validation error\n• Large gap between errors\n\n➔ More data may help somewhat\n➔ Consider regularization", 
            transform=axes[1, 0].transAxes, fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Add more data projection
    axes[1, 0].plot(np.linspace(1.0, 1.5, 20), 
                   np.ones(20) * train_error[-1] + np.random.normal(0, 0.01, 20), 
                   '--', color='blue', alpha=0.5)
    
    # Validation error might improve a bit with more data
    improvement = np.linspace(val_error[-1], val_error[-1]*0.85, 20) + np.random.normal(0, 0.01, 20)
    axes[1, 0].plot(np.linspace(1.0, 1.5, 20), improvement, '--', color='red', alpha=0.5)
    
    # Scenario 4: Still Learning
    train_sizes = np.linspace(0.1, 1.0, 100)
    train_error = 0.6 * np.exp(-2*train_sizes) + 0.1 + np.random.normal(0, 0.01, 100)
    # Validation error still going down
    val_error = 0.9 * np.exp(-1*train_sizes) + 0.2 + np.random.normal(0, 0.03, 100)
    
    axes[1, 1].plot(train_sizes, train_error, color='blue', label='Training Error')
    axes[1, 1].plot(train_sizes, val_error, color='red', label='Validation Error')
    axes[1, 1].legend(fontsize=9)
    
    # Add annotation
    axes[1, 1].text(0.5, 0.7, "• Training error stabilized\n• Validation error still decreasing\n• Gap still narrowing\n\n➔ More data will help\n➔ Model still learning", 
            transform=axes[1, 1].transAxes, fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Add more data projection
    axes[1, 1].plot(np.linspace(1.0, 1.5, 20), 
                   np.ones(20) * train_error[-1] + np.random.normal(0, 0.01, 20), 
                   '--', color='blue', alpha=0.5)
    
    # Continuing improvement with more data
    improvement = np.linspace(val_error[-1], val_error[-1]*0.6, 20) + np.random.normal(0, 0.02, 20)
    axes[1, 1].plot(np.linspace(1.0, 1.5, 20), improvement, '--', color='red', alpha=0.5)
    
    # Add overall title
    plt.suptitle('Learning Curve Scenarios & When More Data Helps', fontsize=16, y=0.98)
    
    # Add conclusions
    plt.figtext(0.5, 0.01, 
               "Key Takeaway: When validation error is still decreasing with available training data, adding more data will likely improve performance.",
               ha="center", fontsize=12, 
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_dir, 'statement8_learning_curve_scenarios.png'), dpi=300, bbox_inches='tight')
    
    # Create an additional visualization showing learning curves with multiple models
    # This demonstrates when more data helps different types of models
    
    # Generate a more complex dataset where complex models need more data
    np.random.seed(42)
    n_samples = 1000
    X_multi = np.linspace(0, 10, n_samples).reshape(-1, 1)
    y_multi = 0.1 * X_multi.squeeze()**3 - 0.5 * X_multi.squeeze()**2 + 2 * X_multi.squeeze() + np.random.normal(0, 10, n_samples)
    
    # Create train/val/test splits
    X_train, X_temp, y_train, y_temp = train_test_split(X_multi, y_multi, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Create models of increasing complexity
    models = {
        'Linear Regression': LinearRegression(),
        'Polynomial (degree=2)': Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('linear', LinearRegression())
        ]),
        'Polynomial (degree=3)': Pipeline([
            ('poly', PolynomialFeatures(degree=3)),
            ('linear', LinearRegression())
        ]),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    # Function to simulate learning with different amounts of data
    def evaluate_with_increasing_data(models, X_train, y_train, X_val, y_val, fractions):
        results = {model_name: {'train_mse': [], 'val_mse': []} for model_name in models}
        training_sizes = []
        
        for fraction in fractions:
            n_samples = int(len(X_train) * fraction)
            training_sizes.append(n_samples)
            
            # Use a subset of the training data
            X_subset = X_train[:n_samples]
            y_subset = y_train[:n_samples]
            
            for model_name, model in models.items():
                # Train model on subset
                model.fit(X_subset, y_subset)
                
                # Evaluate on training subset
                y_train_pred = model.predict(X_subset)
                train_mse = np.mean((y_subset - y_train_pred) ** 2)
                
                # Evaluate on validation set
                y_val_pred = model.predict(X_val)
                val_mse = np.mean((y_val - y_val_pred) ** 2)
                
                # Store results
                results[model_name]['train_mse'].append(train_mse)
                results[model_name]['val_mse'].append(val_mse)
        
        return results, training_sizes
    
    # Evaluate models with increasing data
    fractions = np.linspace(0.1, 1.0, 10)
    model_results, training_sizes = evaluate_with_increasing_data(
        models, X_train, y_train, X_val, y_val, fractions
    )
    
    # Plot results for each model
    plt.figure(figsize=(15, 10))
    
    # Create a 2x2 grid for the models
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Define colors for consistency
    colors = {
        'train': 'blue',
        'val': 'red'
    }
    
    for i, (model_name, results) in enumerate(model_results.items()):
        ax = axes[i]
        
        # Plot training and validation curves
        ax.plot(training_sizes, results['train_mse'], 'o-', color=colors['train'], label='Training MSE')
        ax.plot(training_sizes, results['val_mse'], 'o-', color=colors['val'], label='Validation MSE')
        
        # Project future performance if validation error is still decreasing
        if results['val_mse'][-1] < results['val_mse'][-2]:
            # Simple linear projection for illustration
            slope = (results['val_mse'][-1] - results['val_mse'][-2]) / (training_sizes[-1] - training_sizes[-2])
            future_sizes = np.linspace(training_sizes[-1], training_sizes[-1] * 1.5, 5)
            future_val_mse = results['val_mse'][-1] + slope * (future_sizes - training_sizes[-1])
            ax.plot(future_sizes, future_val_mse, '--', color=colors['val'], alpha=0.5, label='Projected val MSE')
        
        ax.set_title(f'Learning Curve: {model_name}', fontsize=12)
        ax.set_xlabel('Number of Training Samples', fontsize=10)
        ax.set_ylabel('Mean Squared Error', fontsize=10)
        ax.grid(True)
        ax.legend(fontsize=9)
        
        # Add analysis text
        val_change = (results['val_mse'][-2] - results['val_mse'][-1]) / results['val_mse'][-2] * 100
        
        if val_change > 5:
            conclusion = "Still improving significantly.\nMore data will likely help."
            color = 'lightgreen'
        elif val_change > 1:
            conclusion = "Still improving slightly.\nMore data may help."
            color = 'lightyellow'
        else:
            conclusion = "Plateaued.\nMore data unlikely to help."
            color = 'lightcoral'
        
        ax.text(0.05, 0.95, f"Recent improvement: {val_change:.1f}%\n{conclusion}", 
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.5))
    
    plt.suptitle('Impact of More Data on Different Model Types', fontsize=16, y=0.98)
    
    # Add key takeaways at the bottom
    plt.figtext(0.5, 0.01, 
               "Key Takeaways:\n- Simple models (linear) may converge with less data\n"
               "- Complex models (polynomial, tree-based) often benefit more from additional data\n"
               "- When validation error is still decreasing, more data will likely improve performance",
               ha="center", fontsize=12, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(save_dir, 'statement8_model_comparison.png'), dpi=300, bbox_inches='tight')
    
    # Return the results
    result = {
        'statement': "In learning curves, if the validation error continues to decrease as more training samples are added, adding more data is likely to improve model performance.",
        'is_true': True,
        'explanation': "This statement is TRUE. When a learning curve shows validation error decreasing as training set size increases, it indicates the model is still learning from the data and has not yet plateaued. This is a clear signal that providing more training examples would likely lead to better model performance. In contrast, when validation error flattens out, additional data is unlikely to help significantly. As demonstrated in our scenarios, different model types (simple vs. complex) and different problem types (low noise vs. high noise) can show distinct learning curve patterns, but the principle remains: a decreasing validation error curve suggests more data will improve performance, while a flat validation error curve suggests the model has extracted all possible information from the available features.",
        'image_path': ['statement8_learning_curve_convergent.png', 'statement8_learning_curve_improving.png', 'statement8_learning_curve_scenarios.png', 'statement8_model_comparison.png']
    }
    
    return result

if __name__ == "__main__":
    result = statement8_learning_curves()
    print(f"\nStatement: {result['statement']}")
    print(f"True or False: {'True' if result['is_true'] else 'False'}")
    print(f"Explanation: {result['explanation']}")
    print(f"Images saved: {', '.join(result['image_path'])}") 