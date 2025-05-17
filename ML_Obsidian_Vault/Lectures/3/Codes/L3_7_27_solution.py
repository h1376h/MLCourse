import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib.gridspec import GridSpec
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def create_directories():
    """Create necessary directories for saving plots"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    solution_dir = os.path.join(images_dir, "L3_7_Quiz_27")
    
    os.makedirs(solution_dir, exist_ok=True)
    
    return solution_dir

def generate_synthetic_data(n_samples=100, n_features=10, noise=0.5):
    """Generate synthetic data with only 3 relevant features"""
    np.random.seed(42)
    
    # Create feature matrix
    X = np.random.randn(n_samples, n_features)
    
    # True coefficients: only first 3 are non-zero
    true_coef = np.zeros(n_features)
    true_coef[0:3] = [2.5, -1.8, 3.0]
    
    # Generate target with noise
    y = np.dot(X, true_coef) + noise * np.random.randn(n_samples)
    
    return X, y, true_coef

def plot_regularization_paths(save_dir):
    """Generate actual regularization path plots using scikit-learn models"""
    # Generate data
    X, y, true_coef = generate_synthetic_data(n_samples=100, n_features=10)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Regularization parameter values
    alphas = np.logspace(-3, 2, 50)
    
    # Store coefficients and errors
    ridge_coefs = []
    lasso_coefs = []
    ridge_train_errors = []
    ridge_test_errors = []
    lasso_train_errors = []
    lasso_test_errors = []
    
    # Calculate regularization paths
    for alpha in alphas:
        # Ridge
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_scaled, y_train)
        ridge_coefs.append(ridge.coef_.copy())
        
        # Errors for Ridge
        y_train_pred_ridge = ridge.predict(X_train_scaled)
        y_test_pred_ridge = ridge.predict(X_test_scaled)
        ridge_train_errors.append(mean_squared_error(y_train, y_train_pred_ridge))
        ridge_test_errors.append(mean_squared_error(y_test, y_test_pred_ridge))
        
        # Lasso
        lasso = Lasso(alpha=alpha, max_iter=10000, tol=0.0001)
        lasso.fit(X_train_scaled, y_train)
        lasso_coefs.append(lasso.coef_.copy())
        
        # Errors for Lasso
        y_train_pred_lasso = lasso.predict(X_train_scaled)
        y_test_pred_lasso = lasso.predict(X_test_scaled)
        lasso_train_errors.append(mean_squared_error(y_train, y_train_pred_lasso))
        lasso_test_errors.append(mean_squared_error(y_test, y_test_pred_lasso))
    
    # Convert to numpy arrays
    ridge_coefs = np.array(ridge_coefs)
    lasso_coefs = np.array(lasso_coefs)
    
    # Calculate the number of non-zero coefficients for Lasso
    lasso_n_nonzero = np.sum(np.abs(lasso_coefs) > 1e-10, axis=1)
    
    # Create figure for regularization paths
    plt.figure(figsize=(12, 14))
    gs = GridSpec(3, 1, height_ratios=[1, 1, 0.8])
    
    # Plot Ridge coefficients
    ax1 = plt.subplot(gs[0])
    for i in range(X.shape[1]):
        if i < 3:  # Relevant features
            ax1.semilogx(alphas, ridge_coefs[:, i], linewidth=2, 
                        label=f'Feature {i+1} (Relevant)', alpha=0.8)
        else:  # Irrelevant features
            ax1.semilogx(alphas, ridge_coefs[:, i], 'k--', alpha=0.3, linewidth=1)
    
    ax1.set_title('Ridge Regression (L2): Coefficient Paths', fontsize=14)
    ax1.set_xlabel(r'Regularization parameter ($\lambda$)', fontsize=12)
    ax1.set_ylabel('Coefficient Value', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper right')
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Plot Lasso coefficients
    ax2 = plt.subplot(gs[1])
    for i in range(X.shape[1]):
        if i < 3:  # Relevant features
            ax2.semilogx(alphas, lasso_coefs[:, i], linewidth=2, 
                        label=f'Feature {i+1} (Relevant)', alpha=0.8)
        else:  # Irrelevant features
            ax2.semilogx(alphas, lasso_coefs[:, i], 'k--', alpha=0.3, linewidth=1)
    
    ax2.set_title('Lasso Regression (L1): Coefficient Paths', fontsize=14)
    ax2.set_xlabel(r'Regularization parameter ($\lambda$)', fontsize=12)
    ax2.set_ylabel('Coefficient Value', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper right')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Plot number of non-zero coefficients for Lasso
    ax3 = plt.subplot(gs[2])
    ax3.semilogx(alphas, lasso_n_nonzero, 'r-', linewidth=2, marker='o')
    ax3.set_title('Number of Non-Zero Coefficients (Lasso)', fontsize=14)
    ax3.set_xlabel(r'Regularization parameter ($\lambda$)', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'actual_regularization_paths.png'), dpi=300)
    plt.close()
    
    # Return key values for analysis
    return {
        'ridge_coefs': ridge_coefs,
        'lasso_coefs': lasso_coefs,
        'alphas': alphas,
        'ridge_train_errors': ridge_train_errors,
        'ridge_test_errors': ridge_test_errors,
        'lasso_train_errors': lasso_train_errors,
        'lasso_test_errors': lasso_test_errors,
        'lasso_n_nonzero': lasso_n_nonzero
    }

def plot_error_curves(results, save_dir):
    """Plot train and test error curves to find optimal lambda values"""
    alphas = results['alphas']
    ridge_train_errors = results['ridge_train_errors']
    ridge_test_errors = results['ridge_test_errors']
    lasso_train_errors = results['lasso_train_errors']
    lasso_test_errors = results['lasso_test_errors']
    
    # Find optimal lambdas (minimum test error)
    ridge_optimal_idx = np.argmin(ridge_test_errors)
    lasso_optimal_idx = np.argmin(lasso_test_errors)
    
    ridge_optimal_alpha = alphas[ridge_optimal_idx]
    lasso_optimal_alpha = alphas[lasso_optimal_idx]
    
    # Plot error curves
    plt.figure(figsize=(14, 7))
    
    plt.subplot(1, 2, 1)
    plt.semilogx(alphas, ridge_train_errors, 'b-', linewidth=2, alpha=0.7, label='Training Error')
    plt.semilogx(alphas, ridge_test_errors, 'r-', linewidth=2, label='Test Error')
    plt.axvline(x=ridge_optimal_alpha, color='k', linestyle='--', 
                label=f'Optimal λ = {ridge_optimal_alpha:.4f}')
    plt.title('Ridge Regression: Error vs. Regularization', fontsize=14)
    plt.xlabel(r'Regularization parameter ($\lambda$)', fontsize=12)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.semilogx(alphas, lasso_train_errors, 'b-', linewidth=2, alpha=0.7, label='Training Error')
    plt.semilogx(alphas, lasso_test_errors, 'r-', linewidth=2, label='Test Error')
    plt.axvline(x=lasso_optimal_alpha, color='k', linestyle='--', 
                label=f'Optimal λ = {lasso_optimal_alpha:.4f}')
    plt.title('Lasso Regression: Error vs. Regularization', fontsize=14)
    plt.xlabel(r'Regularization parameter ($\lambda$)', fontsize=12)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_curves.png'), dpi=300)
    plt.close()
    
    return ridge_optimal_alpha, lasso_optimal_alpha

def plot_regularization_geometry(save_dir):
    """Plot the geometric interpretation of L1 vs L2 regularization"""
    plt.figure(figsize=(10, 10))
    
    # Create a meshgrid
    x = np.linspace(-2, 2, 1000)
    y = np.linspace(-2, 2, 1000)
    X, Y = np.meshgrid(x, y)
    
    # L1 and L2 norms
    L1_norm = np.abs(X) + np.abs(Y)
    L2_norm = np.sqrt(X**2 + Y**2)
    
    # Create a loss function (sum of squared errors)
    # Simplified as a set of concentric circles centered at (1.5, 1)
    center_x, center_y = 1.5, 1.0
    loss = (X - center_x)**2 + (Y - center_y)**2
    
    # Plot
    plt.contour(X, Y, L1_norm, levels=[1], colors='r', linewidths=2, label='L1 constraint')
    plt.contour(X, Y, L2_norm, levels=[1], colors='b', linewidths=2, label='L2 constraint')
    plt.contour(X, Y, loss, levels=np.linspace(0.1, 3, 10), colors='g', alpha=0.5, 
                linestyles='--', linewidths=1)
    
    # Mark the unconstrained optimum (center of loss circles)
    plt.plot(center_x, center_y, 'go', markersize=10, label='Unconstrained optimum')
    
    # Approximate constrained optima
    # For L1 (where contour meets the diamond)
    l1_opt_x, l1_opt_y = 1.0, 0.0  # Simplified for clarity
    # For L2 (where contour meets the circle)
    l2_opt_x, l2_opt_y = 1.3, 0.7  # Approximated
    
    plt.plot(l1_opt_x, l1_opt_y, 'ro', markersize=8, label='L1 optimum (Lasso)')
    plt.plot(l2_opt_x, l2_opt_y, 'bo', markersize=8, label='L2 optimum (Ridge)')
    
    # Annotation lines
    plt.annotate('', xy=(l1_opt_x, l1_opt_y), xytext=(center_x, center_y),
                arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
    plt.annotate('', xy=(l2_opt_x, l2_opt_y), xytext=(center_x, center_y),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
    
    # Axis lines
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.xlim(-1.5, 2.5)
    plt.ylim(-1.5, 2.5)
    plt.title('Geometric Comparison: L1 vs L2 Regularization', fontsize=14)
    plt.xlabel(r'$\beta_1$', fontsize=12)
    plt.ylabel(r'$\beta_2$', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'regularization_geometry.png'), dpi=300)
    plt.close()

def feature_selection_analysis(results, save_dir):
    """Analyze and visualize feature selection in Ridge vs Lasso"""
    alphas = results['alphas']
    lasso_coefs = results['lasso_coefs']
    ridge_coefs = results['ridge_coefs']
    
    # Find where features are eliminated in Lasso
    # Consider a coefficient zero if it's below this threshold
    threshold = 1e-3
    
    # For each feature, find the alpha where it first becomes zero
    feature_elimination = []
    for i in range(lasso_coefs.shape[1]):
        # Find first index where coefficient becomes smaller than threshold
        zero_indices = np.where(np.abs(lasso_coefs[:, i]) < threshold)[0]
        if len(zero_indices) > 0:
            elim_idx = zero_indices[0]
            elim_alpha = alphas[elim_idx]
        else:
            elim_alpha = float('inf')  # Feature never eliminated
        
        feature_elimination.append((i, elim_alpha))
    
    # Sort features by when they are eliminated
    feature_elimination.sort(key=lambda x: x[1])
    
    # Prepare data for the elimination order plot
    eliminated_features = [f"Feature {f+1}" for f, _ in feature_elimination if f >= 3]  # Irrelevant features
    elimination_alphas = [a for f, a in feature_elimination if f >= 3]
    
    # Find alpha where the first relevant feature is eliminated (if any)
    relevant_eliminated = [a for f, a in feature_elimination if f < 3 and a < float('inf')]
    first_relevant_alpha = min(relevant_eliminated) if relevant_eliminated else None
    
    # Plot elimination order
    plt.figure(figsize=(12, 6))
    bars = plt.barh(eliminated_features, np.log10(elimination_alphas), color='skyblue')
    
    # Add vertical line for first relevant feature elimination
    if first_relevant_alpha:
        plt.axvline(x=np.log10(first_relevant_alpha), color='r', linestyle='--', 
                    label=f'First relevant feature elimination (λ={first_relevant_alpha:.4f})')
        plt.legend()
    
    plt.title('Order of Feature Elimination in Lasso Regression', fontsize=14)
    plt.xlabel(r'log10(λ) when coefficient becomes zero', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_elimination.png'), dpi=300)
    plt.close()
    
    # Coefficient shrinkage comparison
    # Normalize coefficients to show shrinkage proportion
    ridge_normalized = ridge_coefs / ridge_coefs[0]
    
    plt.figure(figsize=(10, 6))
    
    for i in range(3):  # Show only the relevant features for clarity
        plt.semilogx(alphas, ridge_normalized[:, i], linewidth=2, 
                   label=f'Feature {i+1}', alpha=0.8)
    
    plt.title('Ridge Regression: Proportional Shrinkage of Coefficients', fontsize=14)
    plt.xlabel(r'Regularization parameter ($\lambda$)', fontsize=12)
    plt.ylabel('Coefficient Value (normalized)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'proportional_shrinkage.png'), dpi=300)
    plt.close()
    
    return feature_elimination

def run_analysis():
    """Run the full analysis and generate all visualizations"""
    save_dir = create_directories()
    
    # Generate data and analyze regularization paths
    results = plot_regularization_paths(save_dir)
    
    # Find optimal lambda values
    ridge_optimal_alpha, lasso_optimal_alpha = plot_error_curves(results, save_dir)
    
    # Plot geometric interpretation
    plot_regularization_geometry(save_dir)
    
    # Analyze feature selection
    feature_elimination = feature_selection_analysis(results, save_dir)
    
    # Print summary of results
    print("\n===== REGULARIZATION ANALYSIS RESULTS =====")
    print("\nOptimal Lambda Values:")
    print(f"Ridge Regression: λ = {ridge_optimal_alpha:.4f} (log10(λ) = {np.log10(ridge_optimal_alpha):.2f})")
    print(f"Lasso Regression: λ = {lasso_optimal_alpha:.4f} (log10(λ) = {np.log10(lasso_optimal_alpha):.2f})")
    
    print("\nFeature Elimination Order in Lasso:")
    for i, (feature, alpha) in enumerate(feature_elimination):
        if alpha < float('inf'):
            relevant = "RELEVANT" if feature < 3 else "irrelevant"
            print(f"{i+1}. Feature {feature+1} ({relevant}) - eliminated at λ = {alpha:.4f} (log10(λ) = {np.log10(alpha):.2f})")
        else:
            print(f"{i+1}. Feature {feature+1} - never eliminated")
    
    print("\nKey Insights:")
    print("1. Ridge regression shrinks all coefficients proportionally, but never to exactly zero")
    print("2. Lasso regression can set coefficients to exactly zero, performing feature selection")
    print("3. Irrelevant features are eliminated first in Lasso regression as λ increases")
    print("4. The geometric difference: L1 (Lasso) constraint has corners that coincide with axes, L2 (Ridge) doesn't")
    print("5. For feature selection tasks, Lasso is preferred over Ridge")
    
    print("\nImages saved to:", save_dir)

if __name__ == "__main__":
    run_analysis() 