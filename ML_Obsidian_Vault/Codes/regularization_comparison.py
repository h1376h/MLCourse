import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

def generate_regularization_comparison():
    """
    Generate visualizations comparing Ridge and Lasso regularization.
    """
    # Create synthetic data with noise
    np.random.seed(42)
    X = np.sort(np.random.uniform(0, 1, 30))[:, np.newaxis]
    y = np.sin(2 * np.pi * X.ravel()) + 0.3 * np.random.randn(30)
    
    # Create test data for prediction
    X_test = np.linspace(0, 1, 1000)[:, np.newaxis]
    
    # Create figure for coefficient paths
    plt.figure(figsize=(12, 10))
    
    # ========== First subplot: Ridge vs Lasso coefficient paths ==========
    plt.subplot(2, 1, 1)
    
    # Create polynomial features
    degree = 14
    X_poly = PolynomialFeatures(degree).fit_transform(X)
    X_test_poly = PolynomialFeatures(degree).fit_transform(X_test)
    
    # Standardize features
    scaler = StandardScaler()
    X_poly_scaled = scaler.fit_transform(X_poly)
    X_test_poly_scaled = scaler.transform(X_test_poly)
    
    # Compute coefficient paths for Ridge
    alphas_ridge = np.logspace(0, 3, 20)
    coefs_ridge = []
    
    for alpha in alphas_ridge:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_poly_scaled, y)
        coefs_ridge.append(ridge.coef_)
    
    # Compute coefficient paths for Lasso
    alphas_lasso = np.logspace(-3, 0, 20)
    coefs_lasso = []
    
    for alpha in alphas_lasso:
        lasso = Lasso(alpha=alpha, max_iter=10000, tol=1e-2)
        lasso.fit(X_poly_scaled, y)
        coefs_lasso.append(lasso.coef_)
    
    # Plot Ridge coefficient paths
    ax1 = plt.gca()
    lines = ax1.plot(alphas_ridge, coefs_ridge, marker='o', markersize=3)
    plt.xscale('log')
    plt.xlabel('Regularization parameter (α)', fontsize=12)
    plt.ylabel('Coefficient value', fontsize=12)
    plt.title('Ridge Regression: Coefficient Paths', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add a legend in a box outside the main plot
    plt.legend(lines, [f'coef_{i}' for i in range(degree+1)], 
               loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8)
    
    # ========== Second subplot: Ridge vs Lasso predictions ==========
    plt.subplot(2, 1, 2)
    
    # Train models with specific alphas
    models = [
        ('Linear Regression', LinearRegression()),
        ('Ridge (α=1)', Ridge(alpha=1.0)),
        ('Ridge (α=10)', Ridge(alpha=10.0)),
        ('Lasso (α=0.01)', Lasso(alpha=0.01, max_iter=10000)),
        ('Lasso (α=0.1)', Lasso(alpha=0.1, max_iter=10000))
    ]
    
    colors = ['black', 'blue', 'cyan', 'red', 'orange']
    linestyles = ['-', '-', '--', '-', '--']
    
    # Plot the true function
    X_plot = np.linspace(0, 1, 1000)
    plt.plot(X_plot, np.sin(2 * np.pi * X_plot), 'g-', label='True function', linewidth=2)
    
    # Plot the training data
    plt.scatter(X, y, color='green', s=30, label='Training data')
    
    # Train and plot each model
    for i, (name, model) in enumerate(models):
        # Use polynomial features for all models
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree)),
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X_test)
        
        plt.plot(X_test, y_pred, color=colors[i], linestyle=linestyles[i], 
                label=name, linewidth=2, alpha=0.7)
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Comparison of Regularization Methods', fontsize=14)
    plt.ylim(-1.5, 1.5)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Add explanatory text
    plt.figtext(0.1, 0.02, 
                "Ridge: All coefficients are shrunk by the same factor (L2 penalty).\n" +
                "Lasso: Some coefficients are shrunk to exactly zero (L1 penalty).",
                fontsize=12)
    
    plt.tight_layout()
    plt.savefig('plots/regularization_comparison.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    generate_regularization_comparison()
    print("Regularization comparison visualization generated successfully.") 