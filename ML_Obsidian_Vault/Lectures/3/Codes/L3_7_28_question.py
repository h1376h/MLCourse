import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

def create_directories():
    """Create necessary directories for saving plots"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    question_dir = os.path.join(images_dir, "L3_7_Quiz_28")
    
    os.makedirs(question_dir, exist_ok=True)
    
    return question_dir

def bias_variance_regularization_visualization():
    """Generate visualizations showing the impact of regularization on models of different complexity"""
    save_dir = create_directories()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic data from a cubic function
    def true_function(x):
        return 0.1*x**3 - 0.5*x**2 + 1.5*x - 2
    
    # Create data points
    x_true = np.linspace(-4, 4, 100)
    y_true = true_function(x_true)
    
    # Different models
    def fit_models(x_train, y_train, lambda_values):
        # Create polynomial features
        degrees = [1, 3, 10]  # Linear, Cubic, Degree 10
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
    
    # Function to predict
    def predict(models, x_test, lambda_idx=0):
        predictions = []
        
        for model_set in models:
            model, poly = model_set[lambda_idx]
            X_test_poly = poly.transform(x_test.reshape(-1, 1))
            y_pred = model.predict(X_test_poly)
            predictions.append(y_pred)
        
        return predictions
    
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Generate data and create training set
    x_train = np.random.uniform(-4, 4, 20)
    y_train = true_function(x_train) + 0.5 * np.random.randn(len(x_train))  # Add noise
    
    # Lambda values for regularization
    lambda_values = [0.0001, 0.1, 10, 1000]
    
    # Fit models
    all_models = fit_models(x_train, y_train, lambda_values)
    
    # Colors for different model complexities
    colors = ['blue', 'green', 'red']
    model_names = ['Linear Model', 'Cubic Model', 'Degree 10 Polynomial']
    
    # Plot for four different lambda values
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
    
    # Generate annotated version
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    for i, (ax, lambda_idx) in enumerate(zip(axs.flatten(), range(4))):
        # Plot the true function
        ax.plot(x_true, y_true, 'k-', label='True Function', linewidth=2)
        
        # Plot the training data
        ax.scatter(x_train, y_train, color='black', alpha=0.6, label='Training Data')
        
        # Plot predictions
        predictions = predict(all_models, x_true, lambda_idx)
        
        for j, (y_pred, color, name) in enumerate(zip(predictions, colors, model_names)):
            ax.plot(x_true, y_pred, color=color, label=name, linewidth=2, alpha=0.7)
        
        # Add subplot-specific annotations
        if i == 0:  # Very Low Regularization
            ax.annotate('High Variance\nOverfits to noise', 
                       xy=(2, y_pred[60]), xytext=(2.5, 8),
                       arrowprops=dict(facecolor='red', shrink=0.05),
                       fontsize=10, color='red')
        elif i == 1:  # Optimal Regularization
            ax.annotate('Good Balance\nBetween Bias & Variance', 
                       xy=(1, y_pred[50]), xytext=(1.5, 6),
                       arrowprops=dict(facecolor='green', shrink=0.05),
                       fontsize=10, color='green')
        elif i == 2:  # Strong Regularization
            ax.annotate('Increased Bias\nReducing Complexity', 
                       xy=(2, y_pred[60]), xytext=(2.5, 4),
                       arrowprops=dict(facecolor='blue', shrink=0.05),
                       fontsize=10, color='blue')
        elif i == 3:  # Extreme Regularization
            ax.annotate('Very High Bias\nAll Models Similar', 
                       xy=(1, y_pred[50]), xytext=(1.5, 2),
                       arrowprops=dict(facecolor='black', shrink=0.05),
                       fontsize=10)
        
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
    plt.savefig(os.path.join(save_dir, 'bias_variance_regularization_annotated.png'), dpi=300)
    plt.close()
    
    print(f"Bias-variance regularization plots saved to {save_dir}")

if __name__ == "__main__":
    bias_variance_regularization_visualization() 