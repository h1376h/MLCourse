import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_6_Quiz_19")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Function to generate data with underlying pattern
def generate_data(n_samples=1000, noise_level=0.5, seed=42):
    np.random.seed(seed)
    X = np.random.uniform(-1, 1, n_samples).reshape(-1, 1)
    # True underlying pattern: y = sin(2πx) + ε
    y_true = np.sin(2 * np.pi * X.flatten())
    y = y_true + np.random.normal(0, noise_level, n_samples)
    return X, y, y_true

# Function to create models with different complexity
def create_models():
    # Simple model: linear (high bias)
    simple_model = Pipeline([
        ('poly', PolynomialFeatures(degree=1)),
        ('linear', LinearRegression())
    ])
    
    # Complex model: high-degree polynomial (high variance)
    complex_model = Pipeline([
        ('poly', PolynomialFeatures(degree=15)),
        ('linear', LinearRegression())
    ])
    
    return simple_model, complex_model

# Function to evaluate models with different training sizes
def evaluate_learning_curves(max_samples=800, step=50, test_size=200):
    # Generate large dataset
    X_full, y_full, y_true_full = generate_data(n_samples=max_samples+test_size)
    
    # Split into test set (fixed) and potential training set
    X_test, y_test = X_full[-test_size:], y_full[-test_size:]
    X_full_train, y_full_train = X_full[:-test_size], y_full[:-test_size]
    
    # Create models
    simple_model, complex_model = create_models()
    
    # Train sizes to evaluate
    train_sizes = np.arange(30, max_samples+1, step)
    
    # Initialize results arrays
    simple_train_errors = []
    simple_test_errors = []
    complex_train_errors = []
    complex_test_errors = []
    
    # Evaluate models with increasing training data
    for n in train_sizes:
        # Use first n samples for training
        X_train, y_train = X_full_train[:n], y_full_train[:n]
        
        # Train and evaluate simple model
        simple_model.fit(X_train, y_train)
        simple_train_pred = simple_model.predict(X_train)
        simple_test_pred = simple_model.predict(X_test)
        simple_train_errors.append(mean_squared_error(y_train, simple_train_pred))
        simple_test_errors.append(mean_squared_error(y_test, simple_test_pred))
        
        # Train and evaluate complex model
        complex_model.fit(X_train, y_train)
        complex_train_pred = complex_model.predict(X_train)
        complex_test_pred = complex_model.predict(X_test)
        complex_train_errors.append(mean_squared_error(y_train, complex_train_pred))
        complex_test_errors.append(mean_squared_error(y_test, complex_test_pred))
    
    return train_sizes, simple_train_errors, simple_test_errors, complex_train_errors, complex_test_errors

# Function to visualize the model fits
def visualize_model_fits(n_samples=100):
    # Generate data
    X, y, y_true = generate_data(n_samples=n_samples)
    
    # Sort for visualization
    sort_idx = np.argsort(X.flatten())
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    y_true_sorted = y_true[sort_idx]
    
    # Train models
    simple_model, complex_model = create_models()
    simple_model.fit(X, y)
    complex_model.fit(X, y)
    
    # Generate predictions
    X_plot = np.linspace(-1, 1, 1000).reshape(-1, 1)
    y_simple_pred = simple_model.predict(X_plot)
    y_complex_pred = complex_model.predict(X_plot)
    y_true_plot = np.sin(2 * np.pi * X_plot.flatten())
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, alpha=0.6, label='Training data')
    ax.plot(X_plot, y_true_plot, 'g-', linewidth=2, label='True function')
    ax.plot(X_plot, y_simple_pred, 'r-', linewidth=2, label='Simple model (linear)')
    ax.plot(X_plot, y_complex_pred, 'b-', linewidth=2, label='Complex model (degree 15)')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Model Fits: Simple vs. Complex', fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "model_fits.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Function to generate learning curve plots
def plot_learning_curves(train_sizes, simple_train_errors, simple_test_errors, 
                         complex_train_errors, complex_test_errors):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Simple model (high bias)
    axes[0].plot(train_sizes, simple_train_errors, 'r-', linewidth=2, marker='o', 
                label='Training error')
    axes[0].plot(train_sizes, simple_test_errors, 'b-', linewidth=2, marker='s', 
                label='Test error')
    axes[0].set_xlabel('Number of training samples', fontsize=12)
    axes[0].set_ylabel('Mean Squared Error', fontsize=12)
    axes[0].set_title('Learning Curves: Simple Model (High Bias)', fontsize=14)
    axes[0].axhline(y=0.5, color='g', linestyle='--', label='Irreducible error')
    axes[0].legend()
    axes[0].grid(True)
    
    # Complex model (high variance)
    axes[1].plot(train_sizes, complex_train_errors, 'r-', linewidth=2, marker='o', 
                label='Training error')
    axes[1].plot(train_sizes, complex_test_errors, 'b-', linewidth=2, marker='s', 
                label='Test error')
    axes[1].set_xlabel('Number of training samples', fontsize=12)
    axes[1].set_ylabel('Mean Squared Error', fontsize=12)
    axes[1].set_title('Learning Curves: Complex Model (High Variance)', fontsize=14)
    axes[1].axhline(y=0.5, color='g', linestyle='--', label='Irreducible error')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "learning_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Function to create a combined visualization showing the convergence
def plot_convergence(train_sizes, simple_train_errors, simple_test_errors, 
                     complex_train_errors, complex_test_errors):
    # Calculate hypothetical irreducible error level (noise variance)
    irreducible_error = 0.5**2
    
    # Create extended x-axis to show convergence
    extended_sizes = np.concatenate([train_sizes, 
                                    np.arange(train_sizes[-1]+50, 3000, 100)])
    
    # Create extended curves: asymptotic behavior for visualization
    def asymptotic(x, final_val, rate=0.005):
        return final_val + (x[0] - final_val) * np.exp(-rate * (x - x[0]))
    
    # Simple model - converges quickly to a higher error (bias limited)
    simple_train_ext = asymptotic(extended_sizes, irreducible_error + 0.3, 0.001)
    simple_test_ext = asymptotic(extended_sizes, irreducible_error + 0.3, 0.001)
    simple_train_ext[:len(train_sizes)] = simple_train_errors
    simple_test_ext[:len(train_sizes)] = simple_test_errors
    
    # Complex model - converges slowly to irreducible error
    complex_train_ext = asymptotic(extended_sizes, irreducible_error, 0.0005)
    complex_test_ext = asymptotic(extended_sizes, irreducible_error, 0.0002)
    complex_train_ext[:len(train_sizes)] = complex_train_errors
    complex_test_ext[:len(train_sizes)] = complex_test_errors
    
    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Split the x-axis into actual data and extrapolation
    split_idx = len(train_sizes)
    
    # Simple model (high bias)
    axes[0].plot(extended_sizes[:split_idx], simple_train_ext[:split_idx], 'r-', 
                linewidth=2, marker='o', label='Training error')
    axes[0].plot(extended_sizes[:split_idx], simple_test_ext[:split_idx], 'b-', 
                linewidth=2, marker='s', label='Test error')
    axes[0].plot(extended_sizes[split_idx-1:], simple_train_ext[split_idx-1:], 'r--', 
                linewidth=2, label='Projected training error')
    axes[0].plot(extended_sizes[split_idx-1:], simple_test_ext[split_idx-1:], 'b--', 
                linewidth=2, label='Projected test error')
    axes[0].axhline(y=irreducible_error, color='g', linestyle='--', 
                   label='Irreducible error')
    axes[0].axhline(y=irreducible_error+0.3, color='m', linestyle=':', 
                   label='Convergence point')
    
    axes[0].set_xlabel('Number of training samples', fontsize=12)
    axes[0].set_ylabel('Mean Squared Error', fontsize=12)
    axes[0].set_title('Convergence: Simple Model (High Bias)', fontsize=14)
    axes[0].legend()
    axes[0].grid(True)
    
    # Complex model (high variance)
    axes[1].plot(extended_sizes[:split_idx], complex_train_ext[:split_idx], 'r-', 
                linewidth=2, marker='o', label='Training error')
    axes[1].plot(extended_sizes[:split_idx], complex_test_ext[:split_idx], 'b-', 
                linewidth=2, marker='s', label='Test error')
    axes[1].plot(extended_sizes[split_idx-1:], complex_train_ext[split_idx-1:], 'r--', 
                linewidth=2, label='Projected training error')
    axes[1].plot(extended_sizes[split_idx-1:], complex_test_ext[split_idx-1:], 'b--', 
                linewidth=2, label='Projected test error')
    axes[1].axhline(y=irreducible_error, color='g', linestyle='--', 
                   label='Irreducible error')
    
    axes[1].set_xlabel('Number of training samples', fontsize=12)
    axes[1].set_ylabel('Mean Squared Error', fontsize=12)
    axes[1].set_title('Convergence: Complex Model (High Variance)', fontsize=14)
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "learning_curves_convergence.png"), dpi=300, 
               bbox_inches='tight')
    plt.close()

# Function to visualize the bias-variance tradeoff
def plot_bias_variance_tradeoff():
    # Generate x values for the plot
    model_complexity = np.linspace(0, 10, 1000)
    
    # Generate hypothetical error curves
    bias = 5 * np.exp(-0.5 * model_complexity)
    variance = 0.1 * np.exp(0.4 * model_complexity)
    irreducible_error = 0.25 * np.ones_like(model_complexity)
    total_error = bias + variance + irreducible_error
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(model_complexity, bias, 'b-', linewidth=2, label='Bias²')
    plt.plot(model_complexity, variance, 'r-', linewidth=2, label='Variance')
    plt.plot(model_complexity, irreducible_error, 'g-', linewidth=2, 
            label='Irreducible Error')
    plt.plot(model_complexity, total_error, 'k-', linewidth=3, label='Total Error')
    
    # Mark the points for simple and complex models
    simple_idx = 100  # Low complexity index
    complex_idx = 800  # High complexity index
    
    plt.scatter([model_complexity[simple_idx]], [total_error[simple_idx]], 
               s=100, c='purple', marker='o', label='Simple Model')
    plt.scatter([model_complexity[complex_idx]], [total_error[complex_idx]], 
               s=100, c='orange', marker='s', label='Complex Model')
    
    plt.xlabel('Model Complexity', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.title('Bias-Variance Tradeoff', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bias_variance_tradeoff.png"), dpi=300, 
               bbox_inches='tight')
    plt.close()
    
# Function to show the key insights textually
def print_insights():
    print("\nKey Insights on Training and Test Error Behavior")
    print("="*50)
    
    print("\n1. Simple Model (High Bias):")
    print("-"*30)
    print("• Training error: Starts high and stays high regardless of sample size")
    print("• Test error: Similarly high, converges quickly toward training error")
    print("• Convergence point: Both converge to a value higher than irreducible error")
    print("• Problem: Underfitting - model is too simple to capture the true pattern")
    print("• Adding more data doesn't help much after a certain point")
    
    print("\n2. Complex Model (High Variance):")
    print("-"*30)
    print("• Training error: Starts very low (model can memorize small datasets)")
    print("• Test error: Starts very high, decreases with more data")
    print("• Gap: Large gap between training and test error that narrows with more data")
    print("• Convergence point: Both eventually converge to irreducible error (much slower)")
    print("• Problem: Overfitting with small datasets, improves with more data")
    print("• Benefits significantly from additional training data")
    
    print("\n3. Why Training Error Often Increases with More Data:")
    print("-"*30)
    print("• With few samples, complex models can memorize the training data")
    print("• As more examples are added, perfect memorization becomes harder")
    print("• The optimization objective is averaged over more points")
    print("• For simpler models, training error may remain relatively stable")
    
    print("\n4. Why Test Error Decreases with More Data:")
    print("-"*30)
    print("• More training data leads to better parameter estimation")
    print("• Reduces variance in predictions on unseen data")
    print("• Model captures true patterns rather than noise")
    print("• For complex models, prevents overfitting to training noise")
    
    print("\n5. Which Model Requires More Training Data:")
    print("-"*30)
    print("• Complex model (high variance) requires substantially more training data")
    print("• Simple model converges quickly but to a suboptimal error level")
    print("• Complex model converges slowly but can achieve lower error eventually")
    print("• With limited data, simpler models may outperform complex ones")
    print("• With abundant data, complex models can outperform simple ones")
    
    print("\n6. Convergence Values:")
    print("-"*30)
    print("• Irreducible error: The minimal achievable error due to noise in the data")
    print("• Simple model: Converges to irreducible error + bias term")
    print("• Complex model: Can potentially converge to just the irreducible error")
    print("• Theoretically optimal model: Matches the complexity of the true function")
    
    print("\nVisualizations saved to:", save_dir)

# Run all functions
if __name__ == "__main__":
    # Visualize model fits
    visualize_model_fits(n_samples=100)
    
    # Evaluate learning curves
    train_sizes, simple_train_errors, simple_test_errors, complex_train_errors, complex_test_errors = evaluate_learning_curves()
    
    # Plot learning curves
    plot_learning_curves(train_sizes, simple_train_errors, simple_test_errors, complex_train_errors, complex_test_errors)
    
    # Plot extended convergence
    plot_convergence(train_sizes, simple_train_errors, simple_test_errors, complex_train_errors, complex_test_errors)
    
    # Plot bias-variance tradeoff
    plot_bias_variance_tradeoff()
    
    # Print insights
    print_insights() 