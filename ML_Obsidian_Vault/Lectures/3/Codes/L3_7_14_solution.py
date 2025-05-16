import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_7_Quiz_14")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('ggplot')
sns.set_style("whitegrid")
np.random.seed(42)  # For reproducibility

def statement1_regularization_bias():
    """
    Statement 1: Increasing the regularization parameter always increases the bias.
    """
    print("\n==== Statement 1: Increasing regularization parameter and bias ====")
    
    # Generate synthetic data with a complex pattern
    n_samples = 100
    X = np.linspace(0, 1, n_samples).reshape(-1, 1)
    true_function = lambda x: np.sin(2 * np.pi * x) + 0.3 * np.cos(4 * np.pi * x)
    y_true = true_function(X.ravel())
    y = y_true + np.random.normal(0, 0.1, n_samples)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create models with different regularization strengths
    alphas = [0, 0.001, 0.01, 0.1, 1, 10, 100]
    degrees = [5, 10]  # Polynomial degrees to show the effect
    
    fig, axes = plt.subplots(len(degrees), 1, figsize=(10, 10))
    
    for i, degree in enumerate(degrees):
        ax = axes[i]
        
        # Calculate bias for different regularization strengths
        bias_values = []
        variance_values = []
        train_errors = []
        test_errors = []
        
        for alpha in alphas:
            # Set up polynomial regression with Ridge regularization
            model = Pipeline([
                ('poly', PolynomialFeatures(degree=degree)),
                ('ridge', Ridge(alpha=alpha))
            ])
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            train_errors.append(train_mse)
            test_errors.append(test_mse)
            
            # Calculate bias and variance components
            y_pred_true = true_function(X_test.ravel())
            bias = np.mean((y_pred_test - y_pred_true) ** 2)
            bias_values.append(bias)
            
            # We're approximating variance here - in practice would need multiple datasets
            variance = test_mse - bias
            variance_values.append(variance)
        
        # Plot bias, variance, and errors
        ax.plot(alphas, bias_values, 'r-', label='Bias²')
        ax.plot(alphas, variance_values, 'b-', label='Variance')
        ax.plot(alphas, train_errors, 'g--', label='Training Error')
        ax.plot(alphas, test_errors, 'k-.', label='Test Error')
        
        ax.set_xscale('log')
        ax.set_xlabel('Regularization Parameter (alpha)')
        ax.set_ylabel('Error')
        ax.set_title(f'Bias-Variance Tradeoff with Degree {degree} Polynomial')
        ax.legend()
        
        # Add text annotations
        if alpha == alphas[-1]:
            ax.annotate('High Bias\nLow Variance', xy=(alphas[-2], bias_values[-2]), 
                     xytext=(alphas[-3], bias_values[-3]), 
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
        
        if alpha == alphas[0]:
            ax.annotate('Low Bias\nHigh Variance', xy=(alphas[1], bias_values[1]), 
                     xytext=(alphas[2], bias_values[2]), 
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement1_regularization_bias.png'), dpi=300, bbox_inches='tight')
    
    # Visualize model fits with different regularization
    X_plot = np.linspace(0, 1, 1000).reshape(-1, 1)
    selected_alphas = [0, 0.01, 1, 100]
    
    fig, axes = plt.subplots(len(selected_alphas), 1, figsize=(10, 10))
    
    for i, alpha in enumerate(selected_alphas):
        ax = axes[i]
        
        # Create and fit the model
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=10)),
            ('ridge', Ridge(alpha=alpha))
        ])
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_plot)
        
        # Plot data and model
        ax.scatter(X_train, y_train, color='blue', alpha=0.5, label='Training data')
        ax.plot(X_plot, true_function(X_plot.ravel()), 'g-', label='True function')
        ax.plot(X_plot, y_pred, 'r-', label=f'Model (alpha={alpha})')
        
        # Calculate bias component
        y_pred_train = model.predict(X_train)
        bias = np.mean((y_pred_train - true_function(X_train.ravel())) ** 2)
        mse = mean_squared_error(y_train, y_pred_train)
        
        ax.set_title(f'Alpha={alpha}, Bias²={bias:.4f}, Training MSE={mse:.4f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement1_model_fits.png'), dpi=300, bbox_inches='tight')
    
    print("\nFindings for Statement 1:")
    print("- As the regularization parameter increases, the model becomes simpler (less flexible)")
    print("- This typically leads to increased bias, as the model is less able to capture complex patterns")
    print("- At the same time, variance decreases as the model becomes more stable")
    print("- For very complex models (high-degree polynomials), this tradeoff is more pronounced")
    print("- In some cases with moderate regularization, both bias and variance can decrease")
    
    result = {
        'statement': "Increasing the regularization parameter always increases the bias.",
        'is_true': False,
        'explanation': "While increasing regularization generally increases bias as the model becomes simpler, this is not always true. In some scenarios, moderate regularization can reduce overfitting without significantly increasing bias. The relationship depends on the complexity of the true function, the model, and the data distribution.",
        'image_path': ['statement1_regularization_bias.png', 'statement1_model_fits.png']
    }
    
    return result

def statement2_unregularized_training_error():
    """
    Statement 2: An unregularized model will always have lower training error than a regularized 
    version of the same model.
    """
    print("\n==== Statement 2: Unregularized vs Regularized Training Error ====")
    
    # Generate synthetic data
    n_samples = 100
    n_features = 20
    X = np.random.normal(0, 1, (n_samples, n_features))
    true_weights = np.zeros(n_features)
    true_weights[:5] = [1, 0.8, 0.6, 0.4, 0.2]  # Only first 5 features matter
    y_true = X.dot(true_weights)
    y = y_true + np.random.normal(0, 0.1, n_samples)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Compare unregularized and regularized models
    alphas = [0, 0.001, 0.01, 0.1, 1, 10, 100]
    
    train_errors_ridge = []
    test_errors_ridge = []
    train_errors_lasso = []
    test_errors_lasso = []
    
    for alpha in alphas:
        # Ridge regression
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)
        y_pred_train_ridge = ridge.predict(X_train)
        y_pred_test_ridge = ridge.predict(X_test)
        train_errors_ridge.append(mean_squared_error(y_train, y_pred_train_ridge))
        test_errors_ridge.append(mean_squared_error(y_test, y_pred_test_ridge))
        
        # Lasso regression
        lasso = Lasso(alpha=alpha)
        lasso.fit(X_train, y_train)
        y_pred_train_lasso = lasso.predict(X_train)
        y_pred_test_lasso = lasso.predict(X_test)
        train_errors_lasso.append(mean_squared_error(y_train, y_pred_train_lasso))
        test_errors_lasso.append(mean_squared_error(y_test, y_pred_test_lasso))
    
    # Plot training and test errors
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Ridge plot
    ax1.plot(alphas, train_errors_ridge, 'b-o', label='Training Error')
    ax1.plot(alphas, test_errors_ridge, 'r-o', label='Test Error')
    ax1.set_xscale('log')
    ax1.set_xlabel('Regularization Parameter (alpha)')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_title('Ridge Regression: Error vs Regularization')
    ax1.legend()
    ax1.grid(True)
    
    # Lasso plot
    ax2.plot(alphas, train_errors_lasso, 'b-o', label='Training Error')
    ax2.plot(alphas, test_errors_lasso, 'r-o', label='Test Error')
    ax2.set_xscale('log')
    ax2.set_xlabel('Regularization Parameter (alpha)')
    ax2.set_ylabel('Mean Squared Error')
    ax2.set_title('Lasso Regression: Error vs Regularization')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement2_training_error.png'), dpi=300, bbox_inches='tight')
    
    # Create a table of results
    df = pd.DataFrame({
        'Alpha': alphas,
        'Ridge Train MSE': train_errors_ridge,
        'Ridge Test MSE': test_errors_ridge,
        'Lasso Train MSE': train_errors_lasso,
        'Lasso Test MSE': test_errors_lasso
    })
    
    print("\nTraining and Test Errors for Different Regularization Strengths:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    
    print("\nFindings for Statement 2:")
    print("- The unregularized model (alpha=0) has the lowest training error for both Ridge and Lasso")
    print("- As regularization increases, training error generally increases")
    print("- The test error often decreases first (due to reduced overfitting) before increasing again")
    print("- This demonstrates the regularization parameter's role in the bias-variance tradeoff")
    
    result = {
        'statement': "An unregularized model will always have lower training error than a regularized version of the same model.",
        'is_true': True,
        'explanation': "An unregularized model will always have lower or equal training error compared to its regularized counterpart. This is because regularization adds constraints to the optimization objective, forcing the model to balance fitting the data with keeping the coefficients small. The unregularized model focuses solely on minimizing training error without these constraints.",
        'image_path': ['statement2_training_error.png']
    }
    
    return result

def statement3_coefficient_magnitudes():
    """
    Statement 3: If two models have the same training error, the one with smaller coefficient 
    magnitudes will likely generalize better.
    """
    print("\n==== Statement 3: Coefficient Magnitudes and Generalization ====")
    
    # Generate synthetic data with noise
    n_samples = 100
    n_features = 10
    np.random.seed(42)
    X = np.random.normal(0, 1, (n_samples, n_features))
    
    # True model has small coefficients
    true_weights = np.array([0.5, -0.3, 0.2, -0.1, 0.05, 0, 0, 0, 0, 0])
    y_true = X.dot(true_weights)
    y = y_true + np.random.normal(0, 0.1, n_samples)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
    # Create two models with similar training error but different coefficient magnitudes
    # Model 1: Ridge with moderate regularization
    ridge = Ridge(alpha=0.1)
    ridge.fit(X_train, y_train)
    
    # Model 2: Create a model with larger coefficients but similar training error
    # We'll use LinearRegression but manipulate the data slightly
    # Add random noise to X_train for the second model to create a different solution
    X_train_noisy = X_train + np.random.normal(0, 0.01, X_train.shape)
    lr = LinearRegression()
    lr.fit(X_train_noisy, y_train)
    
    # Calculate training and test errors
    y_train_pred_ridge = ridge.predict(X_train)
    y_test_pred_ridge = ridge.predict(X_test)
    train_error_ridge = mean_squared_error(y_train, y_train_pred_ridge)
    test_error_ridge = mean_squared_error(y_test, y_test_pred_ridge)
    
    y_train_pred_lr = lr.predict(X_train)
    y_test_pred_lr = lr.predict(X_test)
    train_error_lr = mean_squared_error(y_train, y_train_pred_lr)
    test_error_lr = mean_squared_error(y_test, y_test_pred_lr)
    
    # Calculate coefficient magnitudes (L2 norm)
    ridge_coef_norm = np.linalg.norm(ridge.coef_)
    lr_coef_norm = np.linalg.norm(lr.coef_)
    
    # Print results
    print("\nComparison of Models with Similar Training Error:")
    print(f"Ridge Model (smaller coefficients):")
    print(f"  Coefficient L2 Norm: {ridge_coef_norm:.6f}")
    print(f"  Training MSE: {train_error_ridge:.6f}")
    print(f"  Test MSE: {test_error_ridge:.6f}")
    print(f"\nLinear Model (larger coefficients):")
    print(f"  Coefficient L2 Norm: {lr_coef_norm:.6f}")
    print(f"  Training MSE: {train_error_lr:.6f}")
    print(f"  Test MSE: {test_error_lr:.6f}")
    
    # Create a visualization of coefficient magnitudes and test error
    # Generate a range of Ridge models with different regularization strengths
    alphas = np.logspace(-3, 3, 20)
    coef_norms = []
    test_errors = []
    train_errors = []
    
    for alpha in alphas:
        ridge_model = Ridge(alpha=alpha)
        ridge_model.fit(X_train, y_train)
        
        # Calculate coefficient norm
        coef_norm = np.linalg.norm(ridge_model.coef_)
        coef_norms.append(coef_norm)
        
        # Calculate errors
        y_train_pred = ridge_model.predict(X_train)
        y_test_pred = ridge_model.predict(X_test)
        train_errors.append(mean_squared_error(y_train, y_train_pred))
        test_errors.append(mean_squared_error(y_test, y_test_pred))
    
    # Plot test error vs coefficient norm
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.plot(coef_norms, test_errors, 'ro-')
    ax1.set_xlabel('Coefficient L2 Norm')
    ax1.set_ylabel('Test MSE')
    ax1.set_title('Test Error vs Coefficient Magnitude')
    ax1.grid(True)
    
    # Add explanation text
    min_idx = np.argmin(test_errors)
    ax1.annotate('Optimal balance', xy=(coef_norms[min_idx], test_errors[min_idx]), 
                xytext=(coef_norms[min_idx]*1.2, test_errors[min_idx]*1.2),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    # Second plot: coefficient values for different models
    ax2.bar(np.arange(n_features), true_weights, alpha=0.3, label='True Coefficients')
    ax2.bar(np.arange(n_features) - 0.2, ridge.coef_, width=0.2, label='Ridge (small coefs)')
    ax2.bar(np.arange(n_features) + 0.2, lr.coef_, width=0.2, label='Linear (large coefs)')
    ax2.set_xlabel('Feature Index')
    ax2.set_ylabel('Coefficient Value')
    ax2.set_title('Model Coefficients Comparison')
    ax2.legend()
    ax2.set_xticks(np.arange(n_features))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement3_coefficient_magnitudes.png'), dpi=300, bbox_inches='tight')
    
    # Plot training vs test error for a range of coefficient norms
    plt.figure(figsize=(10, 6))
    plt.plot(coef_norms, train_errors, 'b-o', label='Training Error')
    plt.plot(coef_norms, test_errors, 'r-o', label='Test Error')
    plt.xlabel('Coefficient L2 Norm')
    plt.ylabel('Mean Squared Error')
    plt.title('Training and Test Error vs Coefficient Magnitude')
    plt.legend()
    plt.grid(True)
    
    # Highlight the region of good generalization
    idx_best = np.argmin(test_errors)
    plt.axvline(x=coef_norms[idx_best], color='green', linestyle='--', alpha=0.7)
    plt.annotate('Best Generalization', xy=(coef_norms[idx_best], test_errors[idx_best]),
                xytext=(coef_norms[idx_best]*1.5, test_errors[idx_best]*1.2),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement3_error_vs_norm.png'), dpi=300, bbox_inches='tight')
    
    print("\nFindings for Statement 3:")
    print("- Models with smaller coefficient magnitudes tend to generalize better")
    print("- This aligns with Occam's razor principle: simpler models (smaller coefficients) generalize better")
    print("- Regularization explicitly controls coefficient magnitudes to improve generalization")
    print("- There's an optimal level of coefficient magnitude that minimizes test error")
    
    result = {
        'statement': "If two models have the same training error, the one with smaller coefficient magnitudes will likely generalize better.",
        'is_true': True,
        'explanation': "This statement is generally true and aligns with Occam's razor. When two models fit the training data equally well, the simpler one (with smaller coefficient magnitudes) typically generalizes better to unseen data. This is because smaller coefficients indicate a smoother, less complex model that is less likely to capture noise in the training data.",
        'image_path': ['statement3_coefficient_magnitudes.png', 'statement3_error_vs_norm.png']
    }
    
    return result

def statement4_lasso_vs_ridge_sparsity():
    """
    Statement 4: Lasso regression typically produces more sparse models than Ridge regression 
    with the same λ value.
    """
    print("\n==== Statement 4: Lasso vs Ridge Sparsity ====")
    
    # Generate synthetic data where only a few features are relevant
    n_samples = 100
    n_features = 20
    np.random.seed(42)
    X = np.random.normal(0, 1, (n_samples, n_features))
    
    # True model has only a few non-zero coefficients (sparse)
    true_weights = np.zeros(n_features)
    true_weights[:3] = [1, 0.7, 0.5]  # Only first three features are relevant
    y_true = X.dot(true_weights)
    y = y_true + np.random.normal(0, 0.1, n_samples)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Compare Lasso and Ridge with same regularization strength
    alphas = [0.001, 0.01, 0.1, 1, 10, 100]
    
    fig, axes = plt.subplots(len(alphas), 2, figsize=(14, 4*len(alphas)))
    
    lasso_nonzero_counts = []
    ridge_nonzero_counts = []
    lasso_test_errors = []
    ridge_test_errors = []
    
    for i, alpha in enumerate(alphas):
        # Ridge regression
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)
        y_pred_test_ridge = ridge.predict(X_test)
        ridge_test_error = mean_squared_error(y_test, y_pred_test_ridge)
        ridge_test_errors.append(ridge_test_error)
        
        # Count "effective" non-zero coefficients in Ridge (coefficients larger than a small threshold)
        ridge_nonzero = np.sum(np.abs(ridge.coef_) > 0.01)
        ridge_nonzero_counts.append(ridge_nonzero)
        
        # Lasso regression
        lasso = Lasso(alpha=alpha)
        lasso.fit(X_train, y_train)
        y_pred_test_lasso = lasso.predict(X_test)
        lasso_test_error = mean_squared_error(y_test, y_pred_test_lasso)
        lasso_test_errors.append(lasso_test_error)
        
        # Count non-zero coefficients in Lasso
        lasso_nonzero = np.sum(np.abs(lasso.coef_) > 0.01)
        lasso_nonzero_counts.append(lasso_nonzero)
        
        # Plot coefficients
        ax_ridge = axes[i, 0]
        ax_lasso = axes[i, 1]
        
        ax_ridge.bar(range(n_features), ridge.coef_)
        ax_ridge.set_title(f'Ridge (α={alpha}): {ridge_nonzero} non-zero coefs, MSE={ridge_test_error:.4f}')
        ax_ridge.set_xlabel('Feature Index')
        ax_ridge.set_ylabel('Coefficient Value')
        ax_ridge.grid(True, alpha=0.3)
        
        ax_lasso.bar(range(n_features), lasso.coef_)
        ax_lasso.set_title(f'Lasso (α={alpha}): {lasso_nonzero} non-zero coefs, MSE={lasso_test_error:.4f}')
        ax_lasso.set_xlabel('Feature Index')
        ax_lasso.set_ylabel('Coefficient Value')
        ax_lasso.grid(True, alpha=0.3)
        
        # Highlight true non-zero coefficients
        for ax in [ax_ridge, ax_lasso]:
            for j in range(len(true_weights)):
                if true_weights[j] != 0:
                    ax.axvspan(j-0.4, j+0.4, alpha=0.2, color='green')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement4_lasso_vs_ridge_coefs.png'), dpi=300, bbox_inches='tight')
    
    # Plot number of non-zero coefficients vs alpha
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, ridge_nonzero_counts, 'bo-', label='Ridge')
    plt.plot(alphas, lasso_nonzero_counts, 'ro-', label='Lasso')
    plt.xscale('log')
    plt.xlabel('Regularization Parameter (alpha)')
    plt.ylabel('Number of Non-zero Coefficients')
    plt.title('Model Sparsity: Lasso vs Ridge')
    plt.legend()
    plt.grid(True)
    
    for i, alpha in enumerate(alphas):
        plt.annotate(f'{lasso_nonzero_counts[i]}', xy=(alpha, lasso_nonzero_counts[i]),
                    xytext=(0, 10), textcoords='offset points', ha='center')
        plt.annotate(f'{ridge_nonzero_counts[i]}', xy=(alpha, ridge_nonzero_counts[i]),
                    xytext=(0, 10), textcoords='offset points', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement4_sparsity_comparison.png'), dpi=300, bbox_inches='tight')
    
    # Create a table of results
    df = pd.DataFrame({
        'Alpha': alphas,
        'Ridge Non-zero Coefs': ridge_nonzero_counts,
        'Ridge Test MSE': ridge_test_errors,
        'Lasso Non-zero Coefs': lasso_nonzero_counts,
        'Lasso Test MSE': lasso_test_errors
    })
    
    print("\nComparison of Ridge and Lasso Models:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    
    # Plot test error and sparsity together
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot number of non-zero coefficients
    ax1.set_xlabel('Regularization Parameter (alpha)')
    ax1.set_ylabel('Number of Non-zero Coefficients', color='b')
    ax1.plot(alphas, ridge_nonzero_counts, 'b--o', label='Ridge Nonzero Coefs')
    ax1.plot(alphas, lasso_nonzero_counts, 'b-o', label='Lasso Nonzero Coefs')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_xscale('log')
    
    # Create second y-axis for test error
    ax2 = ax1.twinx()
    ax2.set_ylabel('Test MSE', color='r')
    ax2.plot(alphas, ridge_test_errors, 'r--o', label='Ridge Test MSE')
    ax2.plot(alphas, lasso_test_errors, 'r-o', label='Lasso Test MSE')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
    
    plt.title('Sparsity and Test Error vs Regularization Strength')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement4_sparsity_and_error.png'), dpi=300, bbox_inches='tight')
    
    print("\nFindings for Statement 4:")
    print("- Lasso consistently produces more sparse models (fewer non-zero coefficients) than Ridge")
    print("- This is due to Lasso's L1 penalty which can shrink coefficients exactly to zero")
    print("- Ridge's L2 penalty only makes coefficients very small but rarely exactly zero")
    print("- Lasso is more effective at feature selection by eliminating irrelevant features")
    print("- For high alpha values, Lasso may eliminate even some relevant features")
    
    result = {
        'statement': "Lasso regression typically produces more sparse models than Ridge regression with the same λ value.",
        'is_true': True,
        'explanation': "Lasso regression uses an L1 penalty which tends to shrink some coefficients exactly to zero, especially for irrelevant features. In contrast, Ridge regression uses an L2 penalty which shrinks all coefficients toward zero but rarely makes them exactly zero. This fundamental difference in regularization approach makes Lasso effective for feature selection and producing sparse models.",
        'image_path': ['statement4_lasso_vs_ridge_coefs.png', 'statement4_sparsity_comparison.png', 'statement4_sparsity_and_error.png']
    }
    
    return result

def run_all_statements():
    results = []
    results.append(statement1_regularization_bias())
    results.append(statement2_unregularized_training_error())
    results.append(statement3_coefficient_magnitudes())
    results.append(statement4_lasso_vs_ridge_sparsity())
    
    # Summarize all results
    print("\n==== Summary of Results ====")
    for i, result in enumerate(results):
        print(f"Statement {i+1}: {result['statement']}")
        print(f"True or False: {'True' if result['is_true'] else 'False'}")
        print(f"Explanation: {result['explanation']}")
        print(f"Images: {', '.join(result['image_path'])}")
        print()
    
    return results

if __name__ == "__main__":
    results = run_all_statements() 