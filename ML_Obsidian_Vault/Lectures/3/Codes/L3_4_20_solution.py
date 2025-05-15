import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import scipy.stats as stats

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_4_Quiz_20")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def linear_basis(x):
    """Linear basis function: φ(x) = x"""
    return x

def polynomial_basis(x, degree=3):
    """Polynomial basis functions of specified degree."""
    return [x**i for i in range(1, degree+1)]

def gaussian_rbf(x, centers, width):
    """Gaussian radial basis functions with specified centers and width."""
    return [np.exp(-(x - c)**2 / (2 * width**2)) for c in centers]

def step_1_explain_glm():
    """Explain how GLMs extend basic linear regression."""
    print("\nStep 1: How Generalized Linear Models Extend Basic Linear Regression")
    print("-" * 80)
    print("In basic linear regression, we model the target variable as a linear combination of inputs:")
    print("    f(x) = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ")
    print("\nGeneralized Linear Models (GLMs) extend this by applying transformations to the inputs:")
    print("    f(x) = w₀ + w₁φ₁(x) + w₂φ₂(x) + ... + wₘφₘ(x)")
    print("\nWhere φᵢ(x) are basis functions that transform the inputs.")
    print("\nDespite these transformations, GLMs preserve linear optimization techniques because:")
    print("1. The model remains linear in the parameters (w₀, w₁, ..., wₘ)")
    print("2. The same linear algebra methods can be used for parameter estimation")
    print("3. The objective function (like sum of squared errors) maintains the same form")
    print("4. The normal equations for finding the optimal weights remain valid")
    print("\nThis means we can use the same efficient optimization methods as linear regression")
    print("while gaining the ability to model non-linear relationships in the original input space.")
    print("-" * 80)

def step_2_define_basis_functions():
    """Define and explain different basis functions."""
    print("\nStep 2: Defining Different Basis Functions")
    print("-" * 80)
    
    # Define input range for visualization
    x = np.linspace(-5, 5, 1000)
    
    # a) Linear regression basis
    print("a) Linear regression basis functions:")
    print("   φ(x) = x")
    print("\nThis is the standard linear model where the input is used directly.")
    
    # b) Polynomial regression of degree 3
    print("\nb) Polynomial regression of degree 3 basis functions:")
    print("   φ₁(x) = x")
    print("   φ₂(x) = x²")
    print("   φ₃(x) = x³")
    print("\nHere, we transform the input into powers of x up to degree 3.")
    
    # c) Gaussian RBF
    centers = [1, 2, 3]
    width = 0.5
    print("\nc) Gaussian radial basis functions with centers at c₁=1, c₂=2, c₃=3 and width σ=0.5:")
    for i, c in enumerate(centers, 1):
        print(f"   φ{i}(x) = exp(-(x-{c})²/(2×{width}²))")
    print("\nThese functions create 'bumps' centered at specific points with controlled width.")
    print("-" * 80)
    
    # Return the x values for visualization
    return x

def step_3_visualize_basis_functions(x):
    """Create visualizations of the different basis functions."""
    print("\nStep 3: Visualizing Basis Functions")
    print("-" * 80)
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig)
    
    # a) Linear basis function
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, linear_basis(x), 'b-', linewidth=2)
    ax1.set_title('Linear Basis Function', fontsize=14)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel(r'$\phi(x) = x$', fontsize=12)
    ax1.grid(True)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # b) Polynomial basis functions
    ax2 = fig.add_subplot(gs[0, 1])
    for i, p in enumerate(polynomial_basis(x), 1):
        ax2.plot(x, p, linewidth=2, label=r'$\phi_{%d}(x) = x^{%d}$' % (i, i))
    ax2.set_title('Polynomial Basis Functions (Degree 3)', fontsize=14)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel(r'$\phi_i(x)$', fontsize=12)
    ax2.legend()
    ax2.grid(True)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # c) Gaussian RBF
    ax3 = fig.add_subplot(gs[1, 0])
    centers = [1, 2, 3]
    width = 0.5
    rbfs = gaussian_rbf(x, centers, width)
    colors = ['r', 'g', 'b']
    
    for i, (rbf, center, color) in enumerate(zip(rbfs, centers, colors), 1):
        ax3.plot(x, rbf, color=color, linewidth=2, 
                label=r'$\phi_{%d}(x) = \exp(-(x-%d)^2/(2\times%.1f^2))$' % (i, center, width))
        ax3.axvline(x=center, color=color, linestyle='--', alpha=0.5)
    
    ax3.set_title('Gaussian Radial Basis Functions', fontsize=14)
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel(r'$\phi_i(x)$', fontsize=12)
    ax3.legend()
    ax3.grid(True)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Combined RBF functions with weighting
    ax4 = fig.add_subplot(gs[1, 1])
    weights = [0.5, 1.0, -0.7]
    weighted_sum = np.zeros_like(x)
    
    for i, (rbf, weight, center, color) in enumerate(zip(rbfs, weights, centers, colors), 1):
        weighted_component = weight * rbf
        ax4.plot(x, weighted_component, color=color, linestyle='--', alpha=0.5,
                label=r'$%.1f \times \phi_{%d}(x; \mathrm{center}=%d)$' % (weight, i, center))
        weighted_sum += weighted_component
    
    ax4.plot(x, weighted_sum, 'k-', linewidth=2, label='Combined function')
    ax4.set_title('Linear Combination of RBFs', fontsize=14)
    ax4.set_xlabel('x', fontsize=12)
    ax4.set_ylabel('f(x)', fontsize=12)
    ax4.legend()
    ax4.grid(True)
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Generate data with non-linear pattern for model comparison
    np.random.seed(42)
    x_data = np.random.uniform(-3, 3, 50)
    x_data.sort()
    
    # True function: a non-linear function with noise
    def true_function(x):
        return np.sin(2*x) + 0.5*x**2 - 0.1*x**3
    
    y_data = true_function(x_data) + np.random.normal(0, 0.3, size=x_data.shape)
    
    # Plot the data and true function
    ax5 = fig.add_subplot(gs[2, :])
    ax5.scatter(x_data, y_data, color='black', s=30, alpha=0.6, label='Data points')
    
    # Original true function
    x_true = np.linspace(-3, 3, 1000)
    y_true = true_function(x_true)
    ax5.plot(x_true, y_true, 'k--', linewidth=2, label='True function')
    
    # Linear model
    linear_model = make_pipeline(LinearRegression())
    linear_model.fit(x_data.reshape(-1, 1), y_data)
    y_linear = linear_model.predict(x_true.reshape(-1, 1))
    ax5.plot(x_true, y_linear, 'b-', linewidth=2, label='Linear model')
    
    # Polynomial model
    poly_model = make_pipeline(PolynomialFeatures(3), LinearRegression())
    poly_model.fit(x_data.reshape(-1, 1), y_data)
    y_poly = poly_model.predict(x_true.reshape(-1, 1))
    ax5.plot(x_true, y_poly, 'r-', linewidth=2, label='Polynomial model (degree 3)')
    
    # RBF model (manually implemented)
    def rbf_transform(x, centers, width):
        X_rbf = np.column_stack([np.exp(-(x.reshape(-1, 1) - c)**2 / (2 * width**2)) 
                                for c in centers])
        return np.column_stack([np.ones(len(x)), X_rbf])
    
    centers = np.linspace(-2.5, 2.5, 6)
    width = 0.5
    X_rbf = rbf_transform(x_data, centers, width)
    
    # Fit using least squares
    rbf_weights = np.linalg.lstsq(X_rbf, y_data, rcond=None)[0]
    
    # Predict
    X_rbf_pred = rbf_transform(x_true, centers, width)
    y_rbf = X_rbf_pred @ rbf_weights
    
    ax5.plot(x_true, y_rbf, 'g-', linewidth=2, label='RBF model')
    
    # Calculate and show MSE
    mse_linear = mean_squared_error(true_function(x_true), y_linear)
    mse_poly = mean_squared_error(true_function(x_true), y_poly)
    mse_rbf = mean_squared_error(true_function(x_true), y_rbf)
    
    ax5.text(0.02, 0.05, f'Linear MSE: {mse_linear:.4f}\nPoly MSE: {mse_poly:.4f}\nRBF MSE: {mse_rbf:.4f}',
            transform=ax5.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    ax5.set_title('Model Comparison on Non-linear Data', fontsize=14)
    ax5.set_xlabel('x', fontsize=12)
    ax5.set_ylabel('y', fontsize=12)
    ax5.legend()
    ax5.grid(True)
    
    plt.tight_layout()
    
    # Save the figure
    fig_path = os.path.join(save_dir, "basis_functions_visualization.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {fig_path}")
    print("\nKey observations from the visualizations:")
    print("1. The linear basis function is a straight line, limited to modeling linear relationships.")
    print("2. Polynomial basis functions can capture increasingly complex curves as degree increases.")
    print("3. Gaussian RBFs create localized 'bumps' that can be combined to model complex patterns.")
    print("4. When comparing models on non-linear data:")
    print(f"   - Linear model (MSE: {mse_linear:.4f}) fails to capture the non-linear pattern")
    print(f"   - Polynomial model (MSE: {mse_poly:.4f}) fits the overall trend but struggles with local variations")
    print(f"   - RBF model (MSE: {mse_rbf:.4f}) captures both global and local patterns effectively")
    print("-" * 80)
    
    return fig_path, mse_linear, mse_poly, mse_rbf

def step_4_overfitting_demonstration():
    """Demonstrate overfitting with high-degree polynomials vs. RBFs."""
    print("\nStep 4: Demonstrating Overfitting with Different Basis Functions")
    print("-" * 80)
    
    # Generate sparse data with noise
    np.random.seed(123)
    x_data = np.random.uniform(-3, 3, 15)  # Fewer points
    x_data.sort()
    
    # True function: a simple non-linear function with noise
    def true_function(x):
        return np.sin(x) + 0.1*x**2
    
    y_data = true_function(x_data) + np.random.normal(0, 0.2, size=x_data.shape)
    
    # Evaluation points
    x_true = np.linspace(-4, 4, 1000)  # Extend beyond training range
    y_true = true_function(x_true)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Polynomial overfitting
    ax1 = axes[0]
    ax1.scatter(x_data, y_data, color='black', s=50, alpha=0.6, label='Training data')
    ax1.plot(x_true, y_true, 'k--', linewidth=2, label='True function')
    
    # Polynomial models with increasing degrees
    degrees = [1, 3, 9]
    colors = ['blue', 'green', 'red']
    mse_train_poly = []
    mse_test_poly = []
    
    for degree, color in zip(degrees, colors):
        poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        poly_model.fit(x_data.reshape(-1, 1), y_data)
        
        y_poly = poly_model.predict(x_true.reshape(-1, 1))
        y_poly_train = poly_model.predict(x_data.reshape(-1, 1))
        
        # Calculate MSE
        mse_train = mean_squared_error(y_data, y_poly_train)
        mse_test = mean_squared_error(true_function(x_true), y_poly)
        mse_train_poly.append(mse_train)
        mse_test_poly.append(mse_test)
        
        ax1.plot(x_true, y_poly, color=color, linewidth=2, 
                label=f'Poly degree {degree} (MSE: {mse_test:.4f})')
    
    ax1.set_title('Polynomial Models with Increasing Degrees', fontsize=14)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim(-3, 3)
    
    # Plot 2: RBF comparison
    ax2 = axes[1]
    ax2.scatter(x_data, y_data, color='black', s=50, alpha=0.6, label='Training data')
    ax2.plot(x_true, y_true, 'k--', linewidth=2, label='True function')
    
    # RBF models with different numbers of centers
    center_counts = [3, 6, 15]
    colors = ['blue', 'green', 'red']
    width = 0.7
    mse_train_rbf = []
    mse_test_rbf = []
    
    for n_centers, color in zip(center_counts, colors):
        # Create evenly spaced centers
        centers = np.linspace(-3, 3, n_centers)
        
        # Transform the input data
        def rbf_transform(x, centers, width):
            X_rbf = np.column_stack([np.exp(-(x.reshape(-1, 1) - c)**2 / (2 * width**2)) 
                                    for c in centers])
            return np.column_stack([np.ones(len(x)), X_rbf])
        
        X_rbf_train = rbf_transform(x_data, centers, width)
        
        # Fit using least squares
        rbf_weights = np.linalg.lstsq(X_rbf_train, y_data, rcond=None)[0]
        
        # Predict
        X_rbf_pred = rbf_transform(x_true, centers, width)
        y_rbf = X_rbf_pred @ rbf_weights
        
        y_rbf_train = X_rbf_train @ rbf_weights
        
        # Calculate MSE
        mse_train = mean_squared_error(y_data, y_rbf_train)
        mse_test = mean_squared_error(true_function(x_true), y_rbf)
        mse_train_rbf.append(mse_train)
        mse_test_rbf.append(mse_test)
        
        ax2.plot(x_true, y_rbf, color=color, linewidth=2, 
                label=f'RBF ({n_centers} centers) (MSE: {mse_test:.4f})')
    
    ax2.set_title('RBF Models with Increasing Centers', fontsize=14)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim(-3, 3)
    
    plt.tight_layout()
    
    # Save the figure
    fig_path = os.path.join(save_dir, "overfitting_demonstration.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a summary figure for MSE comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot training and test MSE for polynomial models
    ax.plot([1, 3, 9], mse_train_poly, 'b-o', linewidth=2, label='Polynomial - Training MSE')
    ax.plot([1, 3, 9], mse_test_poly, 'b--o', linewidth=2, label='Polynomial - Test MSE')
    
    # Plot training and test MSE for RBF models
    ax.plot([3, 6, 15], mse_train_rbf, 'r-o', linewidth=2, label='RBF - Training MSE')
    ax.plot([3, 6, 15], mse_test_rbf, 'r--o', linewidth=2, label='RBF - Test MSE')
    
    ax.set_title('Training vs. Test MSE Comparison', fontsize=14)
    ax.set_xlabel('Model Complexity (Polynomial Degree / Number of RBF Centers)', fontsize=12)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)
    
    # Save the figure
    mse_fig_path = os.path.join(save_dir, "mse_comparison.png")
    plt.savefig(mse_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Overfitting demonstration saved to: {fig_path}")
    print(f"MSE comparison saved to: {mse_fig_path}")
    print("\nKey observations about overfitting:")
    print("1. For polynomial models:")
    print("   - Low degree (1): Underfits the data (high bias)")
    print("   - Medium degree (3): Balances bias and variance")
    print("   - High degree (9): Overfits severely, especially outside the training range")
    print("2. For RBF models:")
    print("   - Few centers (3): Provides a smoother fit that generalizes better")
    print("   - Medium centers (6): Captures more local variations while maintaining generalization")
    print("   - Many centers (15): Can overfit but tends to be more stable than high-degree polynomials")
    print("3. Training vs. Test MSE comparison:")
    print("   - As complexity increases, training error always decreases")
    print("   - Test error forms a U-shape: decreases initially, then increases due to overfitting")
    print("   - RBF models tend to show more graceful degradation with increasing complexity")
    print("-" * 80)
    
    return fig_path, mse_fig_path

def step_5_compare_basis_functions():
    """Compare advantages and disadvantages of different basis functions."""
    print("\nStep 5: Comparing Different Basis Functions")
    print("-" * 80)
    
    print("Polynomial Basis Functions")
    print("-------------------------")
    print("Advantages:")
    print("1. Simple to implement and interpret")
    print("2. Good for smooth, global trends")
    print("3. Efficient computation for low degrees")
    print("4. Well-studied mathematical properties")
    print("5. Can exactly represent many common functions")
    print("\nDisadvantages:")
    print("1. Prone to severe overfitting with high degrees")
    print("2. Poor extrapolation beyond training data")
    print("3. Ill-conditioned for high degrees (numerical instability)")
    print("4. Global nature means local changes affect the entire curve")
    print("5. Not suitable for periodic or highly localized patterns")
    
    print("\nRadial Basis Functions")
    print("---------------------")
    print("Advantages:")
    print("1. Excellent for local patterns and irregularities")
    print("2. More stable outside training range (bounded activation)")
    print("3. Universal approximation capability for any continuous function")
    print("4. Each basis function affects only a local region")
    print("5. Robust to outliers with proper center and width selection")
    print("\nDisadvantages:")
    print("1. Requires choosing centers and widths (hyperparameter selection)")
    print("2. Can be computationally expensive with many centers")
    print("3. Less interpretable than polynomials")
    print("4. May require more parameters for global trends")
    print("5. Determining optimal center placement can be challenging")
    print("-" * 80)

def step_6_recommendations():
    """Provide recommendations for basis functions with non-linear data."""
    print("\nStep 6: Recommendations for Highly Non-linear Data")
    print("-" * 80)
    
    print("For datasets with highly non-linear patterns, I recommend the following approach:")
    print("\n1. Primary Recommendation: Radial Basis Functions")
    print("   - RBFs excel at capturing complex, localized patterns")
    print("   - They provide more stable extrapolation behavior")
    print("   - They're less prone to catastrophic overfitting")
    print("   - Strategies for implementation:")
    print("     a) Use k-means clustering to determine centers")
    print("     b) Adjust width parameter based on data density")
    print("     c) Consider regularization to prevent overfitting")
    
    print("\n2. Alternative Approach: Combination of Basis Functions")
    print("   - Combine polynomial terms (for global trends) with RBFs (for local patterns)")
    print("   - This hybrid approach can capture both smooth global behavior and local irregularities")
    print("   - Example: f(x) = w₀ + w₁x + w₂x² + w₃φ₁(x) + w₄φ₂(x) + ...")
    
    print("\n3. Considerations for Implementation:")
    print("   - Start simple and gradually increase complexity")
    print("   - Use cross-validation to select optimal hyperparameters")
    print("   - Monitor training vs. validation error to detect overfitting")
    print("   - Consider regularization techniques (L1/L2) to control model complexity")
    print("   - For very complex patterns, consider modern approaches like: ")
    print("     a) Gaussian Processes (extension of RBF approach)")
    print("     b) Spline-based methods")
    print("     c) Neural networks with appropriate activation functions")
    print("-" * 80)

def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("SOLUTION TO QUESTION 20: GENERALIZED LINEAR MODELS WITH BASIS FUNCTIONS")
    print("=" * 80)
    
    # Step 1: Explain GLMs
    step_1_explain_glm()
    
    # Step 2: Define basis functions
    x = step_2_define_basis_functions()
    
    # Step 3: Visualize basis functions
    basis_fig_path, mse_linear, mse_poly, mse_rbf = step_3_visualize_basis_functions(x)
    
    # Step 4: Demonstrate overfitting
    overfit_fig_path, mse_fig_path = step_4_overfitting_demonstration()
    
    # Step 5: Compare basis functions
    step_5_compare_basis_functions()
    
    # Step 6: Provide recommendations
    step_6_recommendations()
    
    # Print summary of saved files
    print("\nSUMMARY OF RESULTS")
    print("=" * 80)
    print(f"1. Basis functions visualization: {basis_fig_path}")
    print(f"2. Overfitting demonstration: {overfit_fig_path}")
    print(f"3. MSE comparison chart: {mse_fig_path}")
    print(f"4. Model performance comparison:")
    print(f"   - Linear model MSE: {mse_linear:.4f}")
    print(f"   - Polynomial model MSE: {mse_poly:.4f}")
    print(f"   - RBF model MSE: {mse_rbf:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main() 