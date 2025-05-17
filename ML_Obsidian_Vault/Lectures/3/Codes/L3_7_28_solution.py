import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from matplotlib.patches import Patch
import scipy.linalg

def create_directories():
    """Create necessary directories for saving plots"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    question_dir = os.path.join(images_dir, "L3_7_Quiz_28")
    
    os.makedirs(question_dir, exist_ok=True)
    
    return question_dir

def true_function(x):
    """The true underlying cubic function"""
    return 0.1*x**3 - 0.5*x**2 + 1.5*x - 2

def fit_models(x_train, y_train, lambda_values, degrees=[1, 3, 10]):
    """Fit polynomial models with different degrees and regularization strengths"""
    models = []
    
    for deg in degrees:
        poly = PolynomialFeatures(degree=deg)
        X_poly = poly.fit_transform(x_train.reshape(-1, 1))
        
        # Fit Ridge regression with different lambda values
        model_set = []
        for lam in lambda_values:
            model = Ridge(alpha=lam)
            model.fit(X_poly, y_train)
            model_set.append((model, poly))
            
        models.append(model_set)
    
    return models

def predict(models, x_test, lambda_idx=0):
    """Generate predictions for given models and lambda index"""
    predictions = []
    
    for model_set in models:
        model, poly = model_set[lambda_idx]
        X_test_poly = poly.transform(x_test.reshape(-1, 1))
        y_pred = model.predict(X_test_poly)
        predictions.append(y_pred)
    
    return predictions

def calculate_errors(models, x_train, y_train, x_test, y_test, lambda_values):
    """Calculate training and test errors for all model combinations"""
    degrees = [1, 3, 10]
    train_errors = np.zeros((len(degrees), len(lambda_values)))
    test_errors = np.zeros((len(degrees), len(lambda_values)))
    
    for i, deg in enumerate(degrees):
        for j, _ in enumerate(lambda_values):
            model, poly = models[i][j]
            
            # Training error
            X_train_poly = poly.transform(x_train.reshape(-1, 1))
            y_train_pred = model.predict(X_train_poly)
            train_errors[i, j] = mean_squared_error(y_train, y_train_pred)
            
            # Test error
            X_test_poly = poly.transform(x_test.reshape(-1, 1))
            y_test_pred = model.predict(X_test_poly)
            test_errors[i, j] = mean_squared_error(y_test, y_test_pred)
    
    return train_errors, test_errors

def analyze_coefficients(models, lambda_values):
    """Analyze how coefficients change with regularization"""
    degrees = [1, 3, 10]
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, deg in enumerate(degrees):
        coeffs = []
        for j, _ in enumerate(lambda_values):
            model, _ = models[i][j]
            coeffs.append(model.coef_)
        
        coeffs = np.array(coeffs)
        
        # Plot coefficient magnitudes
        axs[i].set_title(f"Degree {deg} Polynomial", fontsize=14)
        axs[i].set_xlabel("Regularization Strength (λ)", fontsize=12)
        axs[i].set_ylabel("Coefficient Magnitude", fontsize=12)
        axs[i].set_xscale('log')
        
        # Plot each coefficient separately
        for k in range(1, min(coeffs.shape[1], 11)):  # Skip intercept and limit to 10 coefficients
            axs[i].plot(lambda_values, np.abs(coeffs[:, k]), 
                     label=f"β{k}" if k < 6 else None)  # Only show legend for first 5 coeffs
        
        if i == 2:  # For the degree 10 plot
            axs[i].legend(loc='upper right', fontsize=10)
        
        axs[i].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def plot_error_metrics(train_errors, test_errors, lambda_values, save_dir):
    """Plot training and test errors for all model and lambda combinations"""
    # Plot errors versus regularization strength
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    model_names = ['Linear Model', 'Cubic Model', 'Degree 10 Polynomial']
    colors = ['blue', 'green', 'red']
    markers = ['o', 's', '^']
    
    # Training errors
    ax = axs[0]
    for i, (name, color, marker) in enumerate(zip(model_names, colors, markers)):
        ax.loglog(lambda_values, train_errors[i], marker=marker, 
                 color=color, label=name, linewidth=2, markersize=8)
    
    ax.set_title('Training Error vs. Regularization', fontsize=14)
    ax.set_xlabel('Regularization Strength (λ)', fontsize=12)
    ax.set_ylabel('Mean Squared Error (Training)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Test errors
    ax = axs[1]
    for i, (name, color, marker) in enumerate(zip(model_names, colors, markers)):
        ax.loglog(lambda_values, test_errors[i], marker=marker, 
                color=color, label=name, linewidth=2, markersize=8)
    
    ax.set_title('Test Error vs. Regularization', fontsize=14)
    ax.set_xlabel('Regularization Strength (λ)', fontsize=12)
    ax.set_ylabel('Mean Squared Error (Test)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_metrics.png'), dpi=300)
    plt.close()

def plot_learning_curves(lambda_values, save_dir):
    """Plot model predictions for different training set sizes to show learning curves"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create true data for testing
    x_test = np.linspace(-4, 4, 100)
    y_test = true_function(x_test)
    
    # Training set sizes to consider
    training_sizes = [5, 10, 20, 50, 100]
    
    # Only analyze cubic model (index 1) and degree 10 model (index 2)
    model_indices = [1, 2]
    model_names = ['Cubic Model', 'Degree 10 Polynomial']
    colors = ['green', 'red']
    
    # Use two lambda values: low (0.0001) and moderate (0.1)
    lambda_indices = [0, 1]  # 0.0001 and 0.1
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()
    
    for plot_idx, (model_idx, lambda_idx) in enumerate([(m, l) for m in model_indices for l in lambda_indices]):
        # Get subplot
        ax = axs[plot_idx]
        degree = 3 if model_idx == 1 else 10
        lam = lambda_values[lambda_idx]
        
        # Plot true function
        ax.plot(x_test, y_test, 'k-', label='True Function', linewidth=2)
        
        # Plot predictions for different training set sizes
        for size in training_sizes:
            # Generate training data
            x_train = np.random.uniform(-4, 4, size)
            y_train = true_function(x_train) + 0.5 * np.random.randn(len(x_train))
            
            # Fit model
            poly = PolynomialFeatures(degree=degree)
            X_train_poly = poly.fit_transform(x_train.reshape(-1, 1))
            X_test_poly = poly.transform(x_test.reshape(-1, 1))
            
            model = Ridge(alpha=lam)
            model.fit(X_train_poly, y_train)
            
            # Generate predictions
            y_pred = model.predict(X_test_poly)
            
            # Plot predictions with different alpha based on training size
            alpha = 0.3 + 0.5 * (size / max(training_sizes))
            ax.plot(x_test, y_pred, color=colors[model_idx-1], alpha=alpha, 
                   label=f'n={size}' if size == training_sizes[0] or size == training_sizes[-1] else None)
            
            # Plot training points for largest and smallest datasets
            if size == training_sizes[0] or size == training_sizes[-1]:
                ax.scatter(x_train, y_train, color=colors[model_idx-1], alpha=alpha, s=20, marker='o')
        
        # Set labels and title
        ax.set_title(f'{model_names[model_idx-1]}, λ={lam}', fontsize=14)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_ylim(-8, 8)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add a custom legend for the first subplot
        if plot_idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            # Only keep the first few and last handles/labels (true function + smallest and largest datasets)
            keep_indices = [0, 1, -1]  # true function + n=5 + n=100
            handles = [handles[i] for i in keep_indices]
            labels = [labels[i] for i in keep_indices]
            ax.legend(handles, labels, loc='upper left', fontsize=10)
    
    # Add a title explaining the visualization
    fig.suptitle('Learning Curves: Model Predictions vs. Training Set Size', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(save_dir, 'learning_curves.png'), dpi=300)
    plt.close()

def plot_effective_degrees_of_freedom(lambda_values, save_dir):
    """Visualize the effective degrees of freedom for each model as regularization strength increases"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate training data
    x_train = np.random.uniform(-4, 4, 50)
    y_train = true_function(x_train) + 0.5 * np.random.randn(len(x_train))
    
    # Degrees to analyze
    degrees = [1, 3, 10]
    model_names = ['Linear Model', 'Cubic Model', 'Degree 10 Polynomial']
    colors = ['blue', 'green', 'red']
    markers = ['o', 's', '^']
    
    # More fine-grained lambda values for smoother curve
    lambda_range = np.logspace(-4, 3, 100)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # For each model complexity
    for i, degree in enumerate(degrees):
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(x_train.reshape(-1, 1))
        
        # Calculate effective degrees of freedom for each lambda
        effective_df = []
        
        for lam in lambda_range:
            # Calculate the hat matrix diagonal for ridge regression
            # For ridge regression, the effective degrees of freedom is:
            # df = trace(X(X^TX + λI)^(-1)X^T)
            X = X_poly
            n, p = X.shape
            XTX = X.T @ X
            
            # Use SVD to calculate effective df more stably
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            df = np.sum(s**2 / (s**2 + lam))
            
            effective_df.append(df)
        
        # Plot effective df versus lambda
        ax.semilogx(lambda_range, effective_df, color=colors[i], marker=markers[i],
                   label=f"{model_names[i]} (max df: {poly.n_output_features_})",
                   alpha=0.7, markersize=4, markevery=10)
    
    # Add reference lines for original lambda values
    for lam in lambda_values:
        ax.axvline(x=lam, color='grey', linestyle='--', alpha=0.5)
        ax.text(lam, 1, f'λ={lam}', rotation=90, alpha=0.7, ha='right')
    
    # Set labels and title
    ax.set_title('Effective Degrees of Freedom vs. Regularization Strength', fontsize=14)
    ax.set_xlabel('Regularization Strength (λ)', fontsize=12)
    ax.set_ylabel('Effective Degrees of Freedom', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Set y-axis limits
    ax.set_ylim(0, max(degrees) + 5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'effective_degrees_of_freedom.png'), dpi=300)
    plt.close()

def bias_variance_regularization_analysis():
    """Analyze the impact of regularization on bias and variance for different model complexities"""
    save_dir = create_directories()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create true data and training/test sets
    x_true = np.linspace(-4, 4, 100)
    y_true = true_function(x_true)
    
    # Generate training data with noise
    x_train = np.random.uniform(-4, 4, 20)
    y_train = true_function(x_train) + 0.5 * np.random.randn(len(x_train))
    
    # Generate test data (noiseless to measure true error)
    x_test = np.linspace(-4, 4, 50)
    y_test = true_function(x_test)
    
    # Define lambda values for regularization
    lambda_values = [0.0001, 0.1, 10, 1000]
    
    # Fit models
    all_models = fit_models(x_train, y_train, lambda_values)
    
    # 1. First recreate the original visualization
    recreate_original_visualization(x_train, y_train, x_true, y_true, all_models, lambda_values, save_dir)
    
    # 2. Calculate training and test errors
    train_errors, test_errors = calculate_errors(all_models, x_train, y_train, x_test, y_test, lambda_values)
    
    # 3. Plot error metrics
    plot_error_metrics(train_errors, test_errors, lambda_values, save_dir)
    
    # 4. Analyze coefficient changes
    coef_fig = analyze_coefficients(all_models, lambda_values)
    coef_fig.savefig(os.path.join(save_dir, 'coefficient_analysis.png'), dpi=300)
    
    # 5. Understand bias-variance decomposition
    bias_variance_visualization(x_train, y_train, x_test, y_test, lambda_values, save_dir)
    
    # 6. Generate multiple datasets to show variance
    show_variance_across_datasets(lambda_values, save_dir)
    
    # 7. Learning curves visualization
    plot_learning_curves(lambda_values, save_dir)
    
    # 8. New visualization: Effective degrees of freedom
    plot_effective_degrees_of_freedom(lambda_values, save_dir)
    
    print(f"Analysis complete. All plots saved to {save_dir}")
    
    # Return the error metrics for report generation
    return {
        'lambda_values': lambda_values,
        'train_errors': train_errors,
        'test_errors': test_errors
    }

def recreate_original_visualization(x_train, y_train, x_true, y_true, all_models, lambda_values, save_dir):
    """Recreate the original visualization from the question"""
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    colors = ['blue', 'green', 'red']
    model_names = ['Linear Model', 'Cubic Model', 'Degree 10 Polynomial']
    
    lambda_titles = [r'Very Low Regularization ($\lambda=0.0001$)', 
                   r'Optimal Regularization ($\lambda=0.1$)',
                   r'Strong Regularization ($\lambda=10$)', 
                   r'Extreme Regularization ($\lambda=1000$)']
    
    for i, (ax, lambda_idx) in enumerate(zip(axs.flatten(), range(4))):
        # Plot the true function
        ax.plot(x_true, y_true, 'k-', label='True Function', linewidth=2)
        
        # Plot the training data
        ax.scatter(x_train, y_train, color='black', alpha=0.6, label='Training Data')
        
        # Plot predictions
        predictions = predict(all_models, x_true, lambda_idx)
        
        for j, (y_pred, color, name) in enumerate(zip(predictions, colors, model_names)):
            ax.plot(x_true, y_pred, color=color, label=name, linewidth=2, alpha=0.7)
        
        ax.set_title(lambda_titles[i], fontsize=14)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Only add legend to the first plot
        if i == 0:
            ax.legend(loc='upper left', fontsize=10)
        
        # Set same y-axis limits for all plots
        ax.set_ylim(-10, 10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'bias_variance_regularization.png'), dpi=300)
    plt.close()

def bias_variance_visualization(x_train, y_train, x_test, y_test, lambda_values, save_dir):
    """Create visualization showing the decomposition of error into bias and variance components"""
    # For simplicity, focus on test points
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colors for different regularization levels
    lambda_colors = ['#d73027', '#fc8d59', '#91bfdb', '#4575b4']
    models = ['Linear', 'Cubic', 'Degree 10']
    
    # General positioning
    width = 0.2
    x = np.array([1, 2, 3])  # Position for each model type
    
    # For each model type, compute errors for each lambda
    test_errors = []
    
    for model_idx, degree in enumerate([1, 3, 10]):
        model_errors = []
        
        for lam_idx, lam in enumerate(lambda_values):
            # Fit model
            poly = PolynomialFeatures(degree=degree)
            X_train_poly = poly.fit_transform(x_train.reshape(-1, 1))
            X_test_poly = poly.transform(x_test.reshape(-1, 1))
            
            model = Ridge(alpha=lam)
            model.fit(X_train_poly, y_train)
            
            # Calculate test error
            y_pred = model.predict(X_test_poly)
            error = mean_squared_error(y_test, y_pred)
            model_errors.append(error)
        
        test_errors.append(model_errors)
    
    # Convert to numpy array
    test_errors = np.array(test_errors)
    
    # Plot grouped bars
    for i, lam_idx in enumerate(range(len(lambda_values))):
        offset = (i - 1.5) * width
        ax.bar(x + offset, 
               [test_errors[0, lam_idx], test_errors[1, lam_idx], test_errors[2, lam_idx]], 
               width=width, color=lambda_colors[i], 
               label=f'λ = {lambda_values[lam_idx]}')
    
    # Set labels
    ax.set_title('Test Error for Different Models and Regularization Strengths', fontsize=14)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Linear Model', 'Cubic Model', 'Degree 10 Polynomial'], fontsize=12)
    ax.legend(title='Regularization Strength', fontsize=10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Highlight the best model for each lambda
    best_model_indices = np.argmin(test_errors, axis=0)
    for lam_idx, model_idx in enumerate(best_model_indices):
        position = x[model_idx] + (lam_idx - 1.5) * width
        ax.plot(position, test_errors[model_idx, lam_idx], 'k*', markersize=15)
    
    # Add text explanation
    plt.figtext(0.5, 0.01, '* indicates best model for each regularization strength', 
               ha='center', fontsize=12)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(save_dir, 'bias_variance_decomposition.png'), dpi=300)
    plt.close()

def show_variance_across_datasets(lambda_values, save_dir):
    """Show how variance changes with regularization by fitting to multiple datasets"""
    np.random.seed(42)
    
    # Create multiple datasets with noise
    n_datasets = 20
    x_data = np.linspace(-4, 4, 100)
    
    # Visualize only for cubic and degree 10 models
    degrees = [3, 10]
    degrees_names = ['Cubic Model', 'Degree 10 Polynomial']
    
    fig, axs = plt.subplots(len(degrees), len(lambda_values), figsize=(16, 8))
    
    for deg_idx, degree in enumerate(degrees):
        for lam_idx, lam in enumerate(lambda_values):
            ax = axs[deg_idx, lam_idx]
            
            # Plot true function
            y_true = true_function(x_data)
            ax.plot(x_data, y_true, 'k-', label='True', linewidth=2)
            
            # Generate multiple datasets and fit models
            for i in range(n_datasets):
                # Generate noisy training data
                x_train = np.random.uniform(-4, 4, 20)
                y_train = true_function(x_train) + 0.5 * np.random.randn(len(x_train))
                
                # Fit model
                poly = PolynomialFeatures(degree=degree)
                X_train_poly = poly.fit_transform(x_train.reshape(-1, 1))
                X_data_poly = poly.transform(x_data.reshape(-1, 1))
                
                model = Ridge(alpha=lam)
                model.fit(X_train_poly, y_train)
                
                # Predict and plot
                y_pred = model.predict(X_data_poly)
                ax.plot(x_data, y_pred, 'r-', alpha=0.1)
            
            # Set labels
            if deg_idx == 0:
                ax.set_title(f'λ = {lam}', fontsize=12)
            if lam_idx == 0:
                ax.set_ylabel(degrees_names[deg_idx], fontsize=12)
            
            ax.set_ylim(-10, 10)
            ax.grid(True, linestyle='--', alpha=0.5)
    
    # Add custom legend
    legend_elements = [
        Patch(facecolor='k', label='True Function'),
        Patch(facecolor='r', alpha=0.3, label='Model Predictions')
    ]
    fig.legend(handles=legend_elements, loc='upper right', ncol=2, fontsize=12)
    
    # Add title
    fig.suptitle('Model Variance: Multiple Fits with Different Training Datasets', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    plt.savefig(os.path.join(save_dir, 'model_variance.png'), dpi=300)
    plt.close()

def generate_report(results):
    """Generate a report with key findings from the analysis"""
    lambda_values = results['lambda_values']
    train_errors = results['train_errors']
    test_errors = results['test_errors']
    
    # Find best model for each lambda
    best_models = np.argmin(test_errors, axis=0)
    model_names = ['Linear Model', 'Cubic Model', 'Degree 10 Polynomial']
    
    report = "# Question 28: Analysis Results\n\n"
    
    # Question 1: Best model for each regularization level
    report += "## 1. Best Model for Each Regularization Level\n\n"
    report += "| Regularization (λ) | Best Model | Test Error |\n"
    report += "|:------------------:|:----------:|:----------:|\n"
    
    for i, lam in enumerate(lambda_values):
        best_model_idx = best_models[i]
        report += f"| {lam} | {model_names[best_model_idx]} | {test_errors[best_model_idx, i]:.6f} |\n"
    
    # Question 2: Bias-variance tradeoff for degree 10
    report += "\n## 2. Bias-Variance Tradeoff for Degree 10 Polynomial\n\n"
    report += "| Regularization (λ) | Training Error | Test Error | Difference (Approx. Variance) |\n"
    report += "|:------------------:|:--------------:|:----------:|:-----------------------------:|\n"
    
    for i, lam in enumerate(lambda_values):
        train_err = train_errors[2, i]  # Degree 10 is index 2
        test_err = test_errors[2, i]
        diff = test_err - train_err
        report += f"| {lam} | {train_err:.6f} | {test_err:.6f} | {diff:.6f} |\n"
    
    # Question 3: Why linear model changes less
    report += "\n## 3. Effect of Regularization on Different Models\n\n"
    report += "| Model | λ=0.0001 Test Error | λ=1000 Test Error | Relative Change |\n"
    report += "|:-----:|:-------------------:|:------------------:|:---------------:|\n"
    
    for i, model in enumerate(model_names):
        low_reg = test_errors[i, 0]
        high_reg = test_errors[i, 3]
        rel_change = (high_reg - low_reg) / low_reg * 100
        report += f"| {model} | {low_reg:.6f} | {high_reg:.6f} | {rel_change:.2f}% |\n"
    
    # Question 4: Optimal regularization for cubic model
    report += "\n## 4. Optimal Regularization for Cubic Model\n\n"
    report += "| Regularization (λ) | Test Error |\n"
    report += "|:------------------:|:----------:|\n"
    
    for i, lam in enumerate(lambda_values):
        report += f"| {lam} | {test_errors[1, i]:.6f} |\n"
    
    best_lambda_idx = np.argmin(test_errors[1, :])
    report += f"\nOptimal λ for cubic model: **{lambda_values[best_lambda_idx]}**\n"
    
    # Question 5: Poor generalization of degree 10 with low regularization
    report += "\n## 5. Degree 10 Polynomial with Low Regularization\n\n"
    report += f"Training Error: {train_errors[2, 0]:.6f}\n"
    report += f"Test Error: {test_errors[2, 0]:.6f}\n"
    report += f"Ratio (Test/Train): {test_errors[2, 0]/train_errors[2, 0]:.2f}x higher\n"
    
    # Question 6: Best overall combination
    report += "\n## 6. Best Overall Combination\n\n"
    
    flat_idx = np.argmin(test_errors.flatten())
    model_idx = flat_idx // len(lambda_values)
    lambda_idx = flat_idx % len(lambda_values)
    
    report += f"Best overall model: **{model_names[model_idx]}** with λ = **{lambda_values[lambda_idx]}**\n"
    report += f"Test Error: {test_errors[model_idx, lambda_idx]:.6f}\n"
    
    return report

if __name__ == "__main__":
    results = bias_variance_regularization_analysis()
    report = generate_report(results)
    
    # Save the report
    save_dir = create_directories()
    with open(os.path.join(save_dir, 'analysis_report.md'), 'w') as f:
        f.write(report)
    
    print(f"Report saved to {save_dir}/analysis_report.md") 