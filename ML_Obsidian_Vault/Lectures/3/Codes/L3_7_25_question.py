import numpy as np
import matplotlib.pyplot as plt
import os

def create_directories():
    """Create necessary directories for saving plots"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    question_dir = os.path.join(images_dir, "L3_7_Quiz_25")
    
    os.makedirs(question_dir, exist_ok=True)
    
    return question_dir

def error_vs_lambda_plot(save_dir):
    """Generate a plot showing training and validation error vs regularization parameter
    
    QUESTION: How do training and validation errors behave as regularization strength increases,
    and what does this tell us about the bias-variance tradeoff?
    """
    # Generate log spaced lambda values
    lambda_values = np.logspace(-5, 5, 100)
    log_lambda = np.log10(lambda_values)
    
    # Simulate training and validation errors
    # For small lambda: training error low, validation error high (overfitting)
    # For large lambda: both errors high (underfitting)
    # For optimal lambda: validation error minimized
    
    # Example functions to model this behavior
    training_error = 0.1 + 0.4 / (1 + np.exp(-1.5 * (log_lambda + 1)))
    validation_error = 0.3 + 0.6 * np.exp(-0.3 * (log_lambda + 1)**2)
    
    plt.figure(figsize=(10, 6))
    
    # Plot the errors
    plt.plot(log_lambda, training_error, 'b-', linewidth=2, label='Training Error')
    plt.plot(log_lambda, validation_error, 'r-', linewidth=2, label='Validation Error')
    
    # Add vertical lines dividing regions
    plt.axvline(x=-2.5, color='green', linestyle='--', alpha=0.7)
    plt.axvline(x=1.5, color='green', linestyle='--', alpha=0.7)
    
    # Add region labels
    plt.text(-4, 0.8, 'Overfitting Region', fontsize=12, color='darkred')
    plt.text(-1, 0.8, 'Optimal Fitting', fontsize=12, color='darkgreen')
    plt.text(2.5, 0.8, 'Underfitting Region', fontsize=12, color='darkblue')
    
    # Add minimum validation error indicator
    min_val_idx = np.argmin(validation_error)
    plt.plot(log_lambda[min_val_idx], validation_error[min_val_idx], 'ro', markersize=8)
    plt.annotate('Minimum Validation Error', 
                xy=(log_lambda[min_val_idx], validation_error[min_val_idx]),
                xytext=(log_lambda[min_val_idx]+1, validation_error[min_val_idx]+0.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    # Add grid, labels and title
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel(r'log($\lambda$)', fontsize=14)
    plt.ylabel('Error', fontsize=14)
    plt.title('Training and Validation Error vs Regularization Strength', fontsize=16)
    plt.legend(loc='upper center')
    
    # Set x and y limits
    plt.xlim(-5, 5)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_vs_lambda.png'), dpi=300)
    plt.close()
    
    print("QUESTION: How do training and validation errors behave as regularization strength increases, and what does this tell us about the bias-variance tradeoff?")

def empty_error_vs_lambda_plot(save_dir):
    """Generate an empty plot for students to sketch on
    
    QUESTION: Sketch the general shape of the training and validation error curves 
    as functions of the regularization parameter λ.
    """
    # Generate x-axis for log lambda
    log_lambda = np.linspace(-5, 5, 100)
    
    plt.figure(figsize=(10, 6))
    
    # Set up the plot
    plt.xlabel(r'log($\lambda$)', fontsize=14)
    plt.ylabel('Error', fontsize=14)
    plt.title('Training and Validation Error vs Regularization Strength', fontsize=16)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set x and y limits
    plt.xlim(-5, 5)
    plt.ylim(0, 1)
    
    # Add a legend box for the future curves
    plt.legend(['Training Error', 'Validation Error'], loc='upper center')
    
    # Add annotations for regions (to be filled by students)
    plt.annotate('', xy=(-4, 0.1), xytext=(-4, 0.05), 
                arrowprops=dict(arrowstyle='->'))
    plt.annotate('', xy=(0, 0.1), xytext=(0, 0.05), 
                arrowprops=dict(arrowstyle='->'))
    plt.annotate('', xy=(4, 0.1), xytext=(4, 0.05), 
                arrowprops=dict(arrowstyle='->'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'empty_error_vs_lambda.png'), dpi=300)
    plt.close()
    
    print("QUESTION: Sketch the general shape of the training and validation error curves as functions of the regularization parameter λ.")

def regularization_paths_plot(save_dir):
    """Generate plots showing regularization paths for Ridge and Lasso
    
    QUESTION: How do coefficient paths differ between Ridge and Lasso regularization 
    as the regularization parameter changes?
    """
    # Generate log spaced lambda values
    lambda_values = np.logspace(-2, 2, 20)
    log_lambda = np.log10(lambda_values)
    
    # Set up features: 3 relevant, 7 irrelevant
    n_features = 10
    np.random.seed(42)  # For reproducibility
    
    # True weights: only first 3 are non-zero
    true_weights = np.zeros(n_features)
    true_weights[0:3] = [2.5, -1.8, 3.0]
    
    # Simulate Ridge coefficients
    ridge_coefficients = np.zeros((len(lambda_values), n_features))
    for i, lam in enumerate(lambda_values):
        # Ridge shrinks all coefficients proportionally
        shrinkage_factor = 1 / (1 + lam)
        ridge_coefficients[i] = true_weights * shrinkage_factor
    
    # Simulate Lasso coefficients
    lasso_coefficients = np.zeros((len(lambda_values), n_features))
    for i, lam in enumerate(lambda_values):
        # For Lasso, small coefficients get set to zero
        for j in range(n_features):
            if abs(true_weights[j]) > lam / 2:
                lasso_coefficients[i, j] = true_weights[j] - np.sign(true_weights[j]) * lam / 2
            else:
                lasso_coefficients[i, j] = 0
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # 1. Ridge Regularization Path
    plt.subplot(211)
    for i in range(n_features):
        if i < 3:  # Relevant features
            plt.plot(log_lambda, ridge_coefficients[:, i], linewidth=2, 
                    label=f'Feature {i+1} (Relevant)')
        else:  # Irrelevant features
            plt.plot(log_lambda, ridge_coefficients[:, i], 'k--', alpha=0.3, linewidth=1)
    
    plt.title('Ridge Regression (L2): Coefficient Paths', fontsize=14)
    plt.xlabel(r'log($\lambda$)', fontsize=12)
    plt.ylabel('Coefficient Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    # 2. Lasso Regularization Path
    plt.subplot(212)
    for i in range(n_features):
        if i < 3:  # Relevant features
            plt.plot(log_lambda, lasso_coefficients[:, i], linewidth=2, 
                    label=f'Feature {i+1} (Relevant)')
        else:  # Irrelevant features
            plt.plot(log_lambda, lasso_coefficients[:, i], 'k--', alpha=0.3, linewidth=1)
    
    plt.title('Lasso Regression (L1): Coefficient Paths', fontsize=14)
    plt.xlabel(r'log($\lambda$)', fontsize=12)
    plt.ylabel('Coefficient Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'regularization_paths.png'), dpi=300)
    plt.close()
    
    print("QUESTION: How do coefficient paths differ between Ridge and Lasso regularization as the regularization parameter changes?")

def bias_variance_regularization_plot(save_dir):
    """Generate plots showing how regularization affects models of different complexity
    
    QUESTION: How does regularization strength impact the bias-variance tradeoff for 
    models of different complexity?
    """
    # Generate data from a cubic function
    np.random.seed(42)
    
    def true_function(x):
        return 0.1*x**3 - 0.5*x**2 + 1.5*x - 2
    
    x_true = np.linspace(-4, 4, 100)
    y_true = true_function(x_true)
    
    # Generate some noisy training data
    n_samples = 20
    x_train = np.random.uniform(-4, 4, n_samples)
    y_train = true_function(x_train) + 0.5 * np.random.randn(n_samples)
    
    # Create grid for different lambda values and model complexities
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Lambda values for regularization
    lambda_values = [0.0001, 0.1, 10, 1000]
    lambda_names = [r'Very Low ($\lambda=0.0001$)', r'Optimal ($\lambda=0.1$)', 
                   r'High ($\lambda=10$)', r'Very High ($\lambda=1000$)']
    
    # Model complexities
    degrees = [1, 3, 10]  # Linear, Cubic, Degree 10
    colors = ['blue', 'green', 'red']
    model_names = ['Linear Model', 'Cubic Model', 'Degree 10 Polynomial']
    
    # Function to simulate fit with regularization
    def fit_model(degree, lam, x, y):
        # Create polynomial features
        X_poly = np.column_stack([x**i for i in range(degree+1)])
        
        # Ridge regression closed form solution
        I = np.eye(degree+1)
        I[0, 0] = 0  # Don't regularize intercept
        w = np.linalg.inv(X_poly.T @ X_poly + lam * I) @ X_poly.T @ y
        
        # Predict
        X_true_poly = np.column_stack([x_true**i for i in range(degree+1)])
        y_pred = X_true_poly @ w
        
        return y_pred
    
    # Generate plots
    for i, (ax, lam, name) in enumerate(zip(axs.flatten(), lambda_values, lambda_names)):
        # Plot true function
        ax.plot(x_true, y_true, 'k-', label='True Function', linewidth=2)
        
        # Plot training data
        ax.scatter(x_train, y_train, color='black', alpha=0.6, label='Training Data')
        
        # Plot models with different complexity
        for j, (degree, color, model_name) in enumerate(zip(degrees, colors, model_names)):
            y_pred = fit_model(degree, lam, x_train, y_train)
            ax.plot(x_true, y_pred, color=color, label=model_name, linewidth=2, alpha=0.7)
        
        ax.set_title(f'Regularization: {name}', fontsize=14)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Only add legend to the first plot
        if i == 0:
            ax.legend(loc='upper left', fontsize=10)
        
        # Set same y limits
        ax.set_ylim(-10, 10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'bias_variance_regularization.png'), dpi=300)
    plt.close()
    
    print("QUESTION: How does regularization strength impact the bias-variance tradeoff for models of different complexity?")

def generate_question_images():
    """Generate all images for the regularization visualization questions"""
    save_dir = create_directories()
    
    error_vs_lambda_plot(save_dir)
    empty_error_vs_lambda_plot(save_dir)
    regularization_paths_plot(save_dir)
    bias_variance_regularization_plot(save_dir)
    
    print(f"All images saved to {save_dir}")

if __name__ == "__main__":
    generate_question_images() 