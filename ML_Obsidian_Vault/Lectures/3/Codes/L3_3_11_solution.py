import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
images_dir = os.path.join(parent_dir, "Images")
save_dir = os.path.join(images_dir, "L3_3_Quiz_11")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Set random seed for reproducibility
np.random.seed(42)

def generate_data(n_samples, n_features, noise_level, true_w):
    """
    Generate synthetic data with known true parameters.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of features
    noise_level : float
        Standard deviation of the noise
    true_w : numpy array
        True parameter vector (including intercept as the first element)
        
    Returns:
    --------
    X : numpy array
        Feature matrix of shape (n_samples, n_features)
    y : numpy array
        Target vector of shape (n_samples,)
    """
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Generate target with noise
    y_true = X @ true_w[1:] + true_w[0]
    noise = np.random.normal(0, noise_level, n_samples)
    y = y_true + noise
    
    return X, y

def step1_structural_error():
    """
    Explain and demonstrate the structural error in linear regression.
    """
    print("\nStep 1: Structural Error in Linear Regression")
    print("----------------------------------------------")
    print("The structural error is the inherent error due to noise in the data.")
    print("It represents the irreducible error that remains even with the optimal parameters.")
    print()
    print("Mathematical expression for structural error:")
    print("E_x,y[(y - w*^T x)^2]")
    print("where:")
    print("- E_x,y denotes expectation over the joint distribution of x and y")
    print("- w* is the optimal parameter vector with infinite training data")
    print("- y is the true target value")
    print("- x is the feature vector")
    print()
    
    # Demonstrate structural error with a simple example
    n_samples = 500
    n_features = 1
    noise_level = 2.0
    true_w = np.array([3.0, 2.0])  # Intercept and slope
    
    # Generate data
    X, y = generate_data(n_samples, n_features, noise_level, true_w)
    
    # Calculate predictions using true parameters
    y_pred_true = X @ true_w[1:] + true_w[0]
    
    # Calculate structural error
    structural_error = mean_squared_error(y, y_pred_true)
    
    print(f"Demonstration with synthetic data:")
    print(f"True parameters: w* = {true_w}")
    print(f"Noise level: σ = {noise_level}")
    print(f"Structural error (estimated): {structural_error:.4f}")
    print(f"Theoretical structural error (σ²): {noise_level**2:.4f}")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(X.ravel(), y.ravel(), alpha=0.5, label='Data points')
    plt.plot(X.ravel(), y_pred_true, 'r-', label='True model: y = 3 + 2x')
    
    # Add prediction bands to show structural error
    x_sort_idx = np.argsort(X[:, 0])
    X_sorted = X[x_sort_idx]
    y_pred_sorted = y_pred_true[x_sort_idx]
    
    plt.fill_between(X_sorted.ravel(), 
                     y_pred_sorted - 2*noise_level,
                     y_pred_sorted + 2*noise_level,
                     color='red', alpha=0.1, label='±2σ prediction band')
    
    plt.title('Structural Error Visualization', fontsize=14)
    plt.xlabel('Feature (x)', fontsize=12)
    plt.ylabel('Target (y)', fontsize=12)
    plt.legend()
    plt.grid(True)
    
    # Add text box explaining structural error
    textstr = f'Structural Error = {structural_error:.4f}\nTheoretical = {noise_level**2:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save the figure
    fig_path = os.path.join(save_dir, 'step1_structural_error.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return X, y, true_w, noise_level, fig_path

def step2_approximation_error(X, y, true_w):
    """
    Explain and demonstrate the approximation error in linear regression.
    """
    print("\nStep 2: Approximation Error in Linear Regression")
    print("-------------------------------------------------")
    print("The approximation error is the error due to using finite samples to estimate parameters.")
    print("It represents the difference between predictions made with the optimal parameters")
    print("and predictions made with the estimated parameters.")
    print()
    print("Mathematical expression for approximation error:")
    print("E_x[(w*^T x - ŵ^T x)^2]")
    print("where:")
    print("- E_x denotes expectation over the distribution of x")
    print("- w* is the optimal parameter vector with infinite training data")
    print("- ŵ is the parameter vector estimated from a finite training set")
    print("- x is the feature vector")
    print()
    
    # Demonstrate approximation error with different sample sizes
    n_features = X.shape[1]
    sample_sizes = [10, 20, 50, 100, 200, 500]
    approximation_errors = []
    estimated_w_values = []
    
    # For visualization
    X_vis = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_true_vis = X_vis @ true_w[1:] + true_w[0]
    
    plt.figure(figsize=(12, 8))
    
    for i, n in enumerate(sample_sizes):
        # Take a subsample
        indices = np.random.choice(len(X), size=n, replace=False)
        X_sample = X[indices]
        y_sample = y[indices]
        
        # Train linear regression model
        model = LinearRegression()
        model.fit(X_sample, y_sample)
        estimated_w = np.array([model.intercept_, *model.coef_])
        estimated_w_values.append(estimated_w)
        
        # Calculate predictions using estimated parameters
        y_pred_estimated = X @ model.coef_ + model.intercept_
        
        # Calculate predictions using true parameters
        y_pred_true = X @ true_w[1:] + true_w[0]
        
        # Calculate approximation error
        approx_error = np.mean((y_pred_true - y_pred_estimated)**2)
        approximation_errors.append(approx_error)
        
        # Plot results
        plt.subplot(2, 3, i+1)
        plt.scatter(X_sample.ravel(), y_sample.ravel(), alpha=0.5, s=30, label='Training data')
        plt.plot(X_vis.ravel(), y_true_vis, 'r-', label='True model')
        plt.plot(X_vis.ravel(), X_vis @ model.coef_ + model.intercept_, 'b--', 
                 label=f'Estimated (n={n})')
        
        plt.title(f'Sample Size = {n}\nApprox. Error = {approx_error:.4f}')
        if i == 0 or i == 3:
            plt.ylabel('Target (y)')
        if i >= 3:
            plt.xlabel('Feature (x)')
        plt.grid(True)
        if i == 0:
            plt.legend(loc='upper left')
    
    plt.tight_layout()
    
    # Save the figure
    fig_path1 = os.path.join(save_dir, 'step2_approximation_error_samples.png')
    plt.savefig(fig_path1, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot approximation error vs sample size
    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, approximation_errors, 'bo-', linewidth=2, markersize=10)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Sample Size (log scale)', fontsize=12)
    plt.ylabel('Approximation Error (log scale)', fontsize=12)
    plt.title('Approximation Error vs Sample Size', fontsize=14)
    plt.grid(True)
    
    # Add text explaining the relationship
    plt.text(0.5, 0.2, 'Approximation Error $\\propto$ 1/n\nDecreases as sample size increases', 
            transform=plt.gca().transAxes, fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    # Save the figure
    fig_path2 = os.path.join(save_dir, 'step2_approximation_error_trend.png')
    plt.savefig(fig_path2, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot parameter convergence
    estimated_w_values = np.array(estimated_w_values)
    
    plt.figure(figsize=(10, 6))
    plt.axhline(y=true_w[0], color='r', linestyle='-', label='True intercept')
    plt.axhline(y=true_w[1], color='b', linestyle='-', label='True slope')
    
    plt.plot(sample_sizes, estimated_w_values[:, 0], 'ro--', label='Estimated intercept')
    plt.plot(sample_sizes, estimated_w_values[:, 1], 'bo--', label='Estimated slope')
    
    plt.xscale('log')
    plt.xlabel('Sample Size (log scale)', fontsize=12)
    plt.ylabel('Parameter Value', fontsize=12)
    plt.title('Parameter Convergence with Increasing Sample Size', fontsize=14)
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # Save the figure
    fig_path3 = os.path.join(save_dir, 'step2_parameter_convergence.png')
    plt.savefig(fig_path3, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Demonstration with different sample sizes:")
    for n, err, w_est in zip(sample_sizes, approximation_errors, estimated_w_values):
        print(f"Sample size: {n}, Approximation error: {err:.6f}, Estimated w: {w_est}")
    
    return approximation_errors, [fig_path1, fig_path2, fig_path3]

def step3_error_decomposition(X, y, true_w, noise_level):
    """
    Prove that the expected error can be decomposed into structural and approximation errors.
    """
    print("\nStep 3: Error Decomposition Proof")
    print("----------------------------------")
    print("We want to prove: E_x,y[(y - ŵ^T x)^2] = E_x,y[(y - w*^T x)^2] + E_x[(w*^T x - ŵ^T x)^2]")
    print()
    print("Proof Steps:")
    print("1. Let's denote μ(x) = w*^T x as the true regression function")
    print("2. Let's denote μ̂(x) = ŵ^T x as the estimated regression function")
    print("3. The total expected error is: E_x,y[(y - μ̂(x))^2]")
    print()
    print("4. We can rewrite this as: E_x,y[(y - μ(x) + μ(x) - μ̂(x))^2]")
    print("5. Expanding the square: E_x,y[(y - μ(x))^2 + 2(y - μ(x))(μ(x) - μ̂(x)) + (μ(x) - μ̂(x))^2]")
    print()
    print("6. Taking the expectation term by term:")
    print("   a) E_x,y[(y - μ(x))^2] is the structural error")
    print("   b) E_x,y[2(y - μ(x))(μ(x) - μ̂(x))]")
    print("   c) E_x,y[(μ(x) - μ̂(x))^2]")
    print()
    print("7. For term (b), we can use the law of iterated expectations:")
    print("   E_x,y[2(y - μ(x))(μ(x) - μ̂(x))] = E_x[E_y|x[2(y - μ(x))](μ(x) - μ̂(x))]")
    print()
    print("8. Since μ(x) = E[y|x], we have E_y|x[y - μ(x)] = 0")
    print("9. Therefore, term (b) = 0")
    print()
    print("10. Term (c) is E_x,y[(μ(x) - μ̂(x))^2] = E_x[(μ(x) - μ̂(x))^2], which is the approximation error")
    print()
    print("11. Thus, we have proven: E_x,y[(y - ŵ^T x)^2] = E_x,y[(y - w*^T x)^2] + E_x[(w*^T x - ŵ^T x)^2]")
    print()
    
    # Demonstration with different sample sizes
    n_simulations = 100
    sample_sizes = [10, 20, 50, 100, 200, 500]
    
    # Arrays to store results
    total_errors = np.zeros((len(sample_sizes), n_simulations))
    structural_errors = np.zeros((len(sample_sizes), n_simulations))
    approximation_errors = np.zeros((len(sample_sizes), n_simulations))
    
    # Test data for evaluation (larger sample for better estimation)
    n_test = 1000
    X_test, y_test = generate_data(n_test, X.shape[1], noise_level, true_w)
    
    # True predictions on test data
    y_pred_true = X_test @ true_w[1:] + true_w[0]
    
    # Expected structural error
    expected_structural_error = noise_level**2
    
    for i, n in enumerate(sample_sizes):
        for j in range(n_simulations):
            # Generate training data
            X_train, y_train = generate_data(n, X.shape[1], noise_level, true_w)
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Calculate predictions using estimated parameters
            y_pred_est = X_test @ model.coef_ + model.intercept_
            
            # Calculate errors
            total_err = mean_squared_error(y_test, y_pred_est)
            struct_err = mean_squared_error(y_test, y_pred_true)
            approx_err = mean_squared_error(y_pred_true, y_pred_est)
            
            total_errors[i, j] = total_err
            structural_errors[i, j] = struct_err
            approximation_errors[i, j] = approx_err
    
    # Calculate means and standard errors
    mean_total = np.mean(total_errors, axis=1)
    mean_structural = np.mean(structural_errors, axis=1)
    mean_approximation = np.mean(approximation_errors, axis=1)
    
    se_total = np.std(total_errors, axis=1) / np.sqrt(n_simulations)
    se_structural = np.std(structural_errors, axis=1) / np.sqrt(n_simulations)
    se_approximation = np.std(approximation_errors, axis=1) / np.sqrt(n_simulations)
    
    # Calculate sum of errors for comparison
    sum_errors = mean_structural + mean_approximation
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Error decomposition vs. sample size
    plt.subplot(2, 1, 1)
    plt.errorbar(sample_sizes, mean_total, yerr=se_total, fmt='ko-', linewidth=2, markersize=8, 
                 label='Total Error')
    plt.errorbar(sample_sizes, sum_errors, yerr=np.sqrt(se_structural**2 + se_approximation**2), 
                 fmt='mo--', linewidth=2, markersize=8, label='Sum of Errors')
    plt.errorbar(sample_sizes, mean_structural, yerr=se_structural, fmt='ro-', linewidth=2, 
                 markersize=8, label='Structural Error')
    plt.errorbar(sample_sizes, mean_approximation, yerr=se_approximation, fmt='bo-', linewidth=2, 
                 markersize=8, label='Approximation Error')
    
    plt.axhline(y=expected_structural_error, color='r', linestyle='--', 
                label=f'Theoretical Structural Error (σ² = {expected_structural_error})')
    
    plt.xscale('log')
    plt.xlabel('Sample Size (log scale)', fontsize=12)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.title('Error Decomposition in Linear Regression', fontsize=14)
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Proportion of errors
    plt.subplot(2, 1, 2)
    
    proportions_structural = mean_structural / mean_total * 100
    proportions_approximation = mean_approximation / mean_total * 100
    
    plt.bar(range(len(sample_sizes)), proportions_structural, color='red', alpha=0.7,
            label='Structural Error')
    plt.bar(range(len(sample_sizes)), proportions_approximation, bottom=proportions_structural,
            color='blue', alpha=0.7, label='Approximation Error')
    
    plt.xticks(range(len(sample_sizes)), [str(n) for n in sample_sizes])
    plt.xlabel('Sample Size', fontsize=12)
    plt.ylabel('Percentage of Total Error (%)', fontsize=12)
    plt.title('Relative Contribution of Error Components', fontsize=14)
    plt.grid(True, axis='y')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the figure
    fig_path = os.path.join(save_dir, 'step3_error_decomposition.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Display numeric results
    print("Numeric demonstration of error decomposition:")
    print("Sample Size | Total Error | Structural Error | Approximation Error | Sum | Theoretical Structural")
    print("----------------------------------------------------------------------------")
    for n, tot, stru, appr, sum_err in zip(sample_sizes, mean_total, mean_structural, 
                                           mean_approximation, sum_errors):
        print(f"{n:11d} | {tot:11.4f} | {stru:16.4f} | {appr:19.4f} | {sum_err:3.4f} | {expected_structural_error:20.4f}")
    
    return fig_path

def step4_practical_significance():
    """
    Explain the practical significance of error decomposition for model selection.
    """
    print("\nStep 4: Practical Significance of Error Decomposition")
    print("-----------------------------------------------------")
    print("The error decomposition has significant practical implications for model selection:")
    print()
    print("1. Bias-Variance Tradeoff:")
    print("   - Structural error represents the irreducible error (noise) in the data")
    print("   - Approximation error can be further decomposed into bias^2 and variance")
    print("   - Complex models reduce bias but may increase variance")
    print("   - Simple models may have higher bias but lower variance")
    print()
    print("2. Model Selection Guidelines:")
    print("   - When structural error dominates: focus on collecting better quality data")
    print("   - When approximation error dominates: consider more complex models or more data")
    print("   - For small datasets: prefer simpler models to avoid high variance")
    print("   - For large datasets: can afford more complex models as approximation error decreases")
    print()
    
    # Demonstrate with different model complexities
    np.random.seed(42)
    
    # Generate data from a cubic function with noise
    n_samples = 50
    X = np.sort(np.random.uniform(-3, 3, n_samples)).reshape(-1, 1)
    # Define a vectorized version of the function
    true_function = lambda x: 1 + 2*x.ravel() - 0.5*np.power(x.ravel(), 2) + 0.1*np.power(x.ravel(), 3)
    # Apply the function to X
    y_true = true_function(X)
    noise = np.random.normal(0, 1, n_samples)
    y = y_true + noise
    
    # Print shapes for debugging
    print(f"DEBUG: X shape: {X.shape}, y shape: {y.shape}")
    print(f"DEBUG: X.ravel() shape: {X.ravel().shape}, y.ravel() shape: {y.ravel().shape}")
    
    # Test data (dense grid)
    X_test = np.linspace(-3.5, 3.5, 200).reshape(-1, 1)
    y_test_true = true_function(X_test)
    
    # Different polynomial degrees representing model complexity
    degrees = [1, 2, 3, 5, 10]
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    plt.figure(figsize=(14, 10))
    
    # Plot the data and true function
    plt.subplot(2, 1, 1)
    # Make sure X and y have the same shape for scatter plot
    plt.scatter(X[:, 0], y.ravel(), color='black', s=30, alpha=0.6, label='Training data')
    plt.plot(X_test[:, 0], y_test_true, 'k-', label='True function', linewidth=2)
    
    mse_train = []
    mse_test = []
    bias_squared = []
    variance = []
    
    # Fit different polynomial degrees
    for i, degree in enumerate(degrees):
        # Create polynomial features
        X_poly_train = np.column_stack([X**p for p in range(1, degree+1)])
        X_poly_test = np.column_stack([X_test**p for p in range(1, degree+1)])
        
        # Fit model
        model = LinearRegression()
        model.fit(X_poly_train, y)
        
        # Make predictions
        y_pred_train = model.predict(X_poly_train)
        y_pred_test = model.predict(X_poly_test)
        
        # Calculate errors
        mse_train.append(mean_squared_error(y, y_pred_train))
        mse_test.append(mean_squared_error(y_test_true, y_pred_test))
        
        # Calculate bias^2 (using test data)
        bias_squared.append(np.mean((y_test_true - y_pred_test)**2))
        
        # Calculate variance through multiple random samples
        preds = []
        for _ in range(50):
            # Generate new random sample
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            
            # Create polynomial features
            X_poly_sample = np.column_stack([X_sample**p for p in range(1, degree+1)])
            
            # Fit model
            model_sample = LinearRegression()
            model_sample.fit(X_poly_sample, y_sample)
            
            # Predict on test data
            preds.append(model_sample.predict(X_poly_test))
        
        # Variance is average squared deviation from mean prediction
        preds = np.array(preds)
        variance.append(np.mean(np.var(preds, axis=0)))
        
        # Plot fitted function
        plt.plot(X_test[:, 0], y_pred_test, color=colors[i], linestyle='-', linewidth=2,
                label=f'Degree {degree}')
    
    plt.title('Polynomial Regression with Different Complexities', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True)
    
    # Plot error decomposition
    plt.subplot(2, 1, 2)
    
    width = 0.35
    x_pos = np.arange(len(degrees))
    
    # Plot bias squared and variance
    plt.bar(x_pos - width/2, bias_squared, width, color='blue', alpha=0.7, label='Bias²')
    plt.bar(x_pos + width/2, variance, width, color='red', alpha=0.7, label='Variance')
    
    # Plot total error (MSE on test)
    plt.plot(x_pos, mse_test, 'ko-', linewidth=2, markersize=8, label='Test Error')
    
    plt.xticks(x_pos, [str(d) for d in degrees])
    plt.xlabel('Polynomial Degree (Model Complexity)', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.title('Bias-Variance Tradeoff with Increasing Model Complexity', fontsize=14)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the figure
    fig_path1 = os.path.join(save_dir, 'step4_model_complexity.png')
    plt.savefig(fig_path1, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a figure showing training vs test error with increasing sample size
    np.random.seed(42)
    
    # Generate larger dataset
    n_total = 1000
    X_total = np.sort(np.random.uniform(-3, 3, n_total)).reshape(-1, 1)
    y_total_true = true_function(X_total)
    noise_total = np.random.normal(0, 1, n_total)
    y_total = y_total_true + noise_total
    
    # Sample sizes to test
    sample_sizes = [10, 20, 50, 100, 200, 500]
    
    # Test set (for final evaluation)
    X_eval = np.linspace(-3.5, 3.5, 200).reshape(-1, 1)
    y_eval_true = true_function(X_eval)
    
    # Fixed high complexity model (overfitting prone)
    degree_high = 10
    
    # Fixed low complexity model (potentially underfitting)
    degree_low = 2
    
    # Arrays to store results
    train_errors_high = []
    test_errors_high = []
    train_errors_low = []
    test_errors_low = []
    
    for n in sample_sizes:
        # Sample data
        indices = np.random.choice(n_total, size=n, replace=False)
        X_sample = X_total[indices]
        y_sample = y_total[indices]
        
        # High complexity model
        X_high_train = np.column_stack([X_sample**p for p in range(1, degree_high+1)])
        X_high_test = np.column_stack([X_eval**p for p in range(1, degree_high+1)])
        
        model_high = LinearRegression()
        model_high.fit(X_high_train, y_sample)
        
        y_pred_train_high = model_high.predict(X_high_train)
        y_pred_test_high = model_high.predict(X_high_test)
        
        train_errors_high.append(mean_squared_error(y_sample, y_pred_train_high))
        test_errors_high.append(mean_squared_error(y_eval_true, y_pred_test_high))
        
        # Low complexity model
        X_low_train = np.column_stack([X_sample**p for p in range(1, degree_low+1)])
        X_low_test = np.column_stack([X_eval**p for p in range(1, degree_low+1)])
        
        model_low = LinearRegression()
        model_low.fit(X_low_train, y_sample)
        
        y_pred_train_low = model_low.predict(X_low_train)
        y_pred_test_low = model_low.predict(X_low_test)
        
        train_errors_low.append(mean_squared_error(y_sample, y_pred_train_low))
        test_errors_low.append(mean_squared_error(y_eval_true, y_pred_test_low))
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(sample_sizes, train_errors_low, 'g--', linewidth=2, label='Train Error (Low)')
    plt.plot(sample_sizes, test_errors_low, 'g-', linewidth=2, label='Test Error (Low)')
    plt.plot(sample_sizes, train_errors_high, 'r--', linewidth=2, label='Train Error (High)')
    plt.plot(sample_sizes, test_errors_high, 'r-', linewidth=2, label='Test Error (High)')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'Error vs Sample Size\nLow (d={degree_low}) vs High (d={degree_high}) Complexity', fontsize=14)
    plt.xlabel('Sample Size (log scale)', fontsize=12)
    plt.ylabel('Mean Squared Error (log scale)', fontsize=12)
    plt.grid(True)
    plt.legend()
    
    # Plot overfitting gap
    plt.subplot(1, 2, 2)
    gap_high = np.array(test_errors_high) - np.array(train_errors_high)
    gap_low = np.array(test_errors_low) - np.array(train_errors_low)
    
    plt.plot(sample_sizes, gap_high, 'r-', linewidth=2, label=f'High Complexity (d={degree_high})')
    plt.plot(sample_sizes, gap_low, 'g-', linewidth=2, label=f'Low Complexity (d={degree_low})')
    
    plt.xscale('log')
    plt.title('Overfitting Gap (Test Error - Train Error)', fontsize=14)
    plt.xlabel('Sample Size (log scale)', fontsize=12)
    plt.ylabel('Error Gap', fontsize=12)
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # Save the figure
    fig_path2 = os.path.join(save_dir, 'step4_sample_size_complexity.png')
    plt.savefig(fig_path2, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary of findings
    print("Results from model complexity experiment:")
    print("Degree | Train Error | Test Error | Bias² | Variance")
    print("------------------------------------------------")
    for i, degree in enumerate(degrees):
        print(f"{degree:6d} | {mse_train[i]:11.4f} | {mse_test[i]:10.4f} | {bias_squared[i]:4.4f} | {variance[i]:8.4f}")
    
    print()
    print("3. Key insights for practical model selection:")
    print("   - For limited data, the test error of complex models is much higher than training error")
    print("   - As data increases, high complexity models improve more drastically")
    print("   - Structural error sets a lower bound that cannot be overcome by increasing model complexity")
    print("   - Select models with the right complexity for your dataset size to minimize total error")
    print("   - Use cross-validation to estimate the total expected error")
    
    return [fig_path1, fig_path2]

# Run all the steps
if __name__ == "__main__":
    print("Question 11: Error Decomposition in Linear Regression")
    print("====================================================")
    
    # Step 1: Structural Error
    X, y, true_w, noise_level, structural_fig = step1_structural_error()
    
    # Step 2: Approximation Error
    approximation_errors, approximation_figs = step2_approximation_error(X, y, true_w)
    
    # Step 3: Error Decomposition Proof
    decomposition_fig = step3_error_decomposition(X, y, true_w, noise_level)
    
    # Step 4: Practical Significance
    practical_figs = step4_practical_significance()
    
    # Summarize all figures
    all_figures = [structural_fig] + approximation_figs + [decomposition_fig] + practical_figs
    
    print("\nSummary of Figures Created:")
    for i, fig in enumerate(all_figures):
        print(f"{i+1}. {os.path.basename(fig)}")
    
    print("\nQuestion 11 Solution Complete") 