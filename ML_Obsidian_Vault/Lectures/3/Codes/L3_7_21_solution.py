import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_7_Quiz_21")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
# Enable LaTeX rendering for all text
plt.rcParams.update({
    "text.usetex": False,  # Don't use true LaTeX rendering
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif"  # Use matplotlib's mathtext rendering
})

print("Regularization Methods Comparison: Ridge, Lasso, and Elastic Net")
print("===============================================================")
print("This script demonstrates the differences between three regularization methods:")
print("1. Ridge Regression: Uses L2 penalty (sum of squared coefficients)")
print("2. Lasso Regression: Uses L1 penalty (sum of absolute coefficients)")
print("3. Elastic Net: Combines both L1 and L2 penalties")
print("\nMathematically, these methods minimize the following objective functions:")
print("Ridge: min ||y - Xw||^2 + λ||w||^2")
print("Lasso: min ||y - Xw||^2 + λ||w||^1")
print("Elastic Net: min ||y - Xw||^2 + λ1||w||^1 + λ2||w||^2")
print("\nKey differences:")
print("- Ridge shrinks coefficients but rarely sets them to exactly zero")
print("- Lasso can set coefficients to exactly zero (feature selection)")
print("- Elastic Net balances these properties")

# Part 1: Generate synthetic data with irrelevant features and correlation
print("\n1. Generating Synthetic Data")
print("---------------------------")

# Set random seed for reproducibility
np.random.seed(42)

def generate_correlated_features(n_samples, n_features, n_informative, correlation=0.8):
    """Generate dataset with correlated features."""
    print(f"Step 1.1: Generating {n_informative} informative features")
    # Generate the informative features
    X_informative, y = make_regression(
        n_samples=n_samples, 
        n_features=n_informative, 
        n_informative=n_informative, 
        noise=20, 
        random_state=42
    )
    
    print(f"Step 1.2: Creating {n_features - n_informative} correlated/irrelevant features")
    # Generate correlated features (versions of the informative features)
    X_correlated = np.zeros((n_samples, n_features))
    X_correlated[:, :n_informative] = X_informative
    
    # Create correlated features by adding noise to existing ones
    for i in range(n_informative, n_features):
        # Select a random feature to correlate with
        source_feature = np.random.randint(0, n_informative)
        # Create correlation by mixing the source feature with random noise
        # Formula: new_feature = correlation * source_feature + (1-correlation) * noise
        X_correlated[:, i] = (correlation * X_correlated[:, source_feature] + 
                            (1 - correlation) * np.random.normal(0, 1, n_samples))
    
    print(f"Step 1.3: Setting up true coefficients (only for informative features)")
    # Add a constant term (intercept) to the true model
    true_coefficients = np.zeros(n_features)
    # Only the first n_informative features will have non-zero coefficients
    true_coefficients[:n_informative] = np.random.uniform(1, 3, n_informative)
    
    # Make some random coefficients negative
    neg_idx = np.random.choice(range(n_informative), n_informative // 2, replace=False)
    true_coefficients[neg_idx] *= -1
    
    print(f"Step 1.4: Generating response variable using only informative features")
    # Calculate y using only significant features
    y = np.dot(X_correlated[:, :n_informative], true_coefficients[:n_informative])
    # Add random noise to make the problem more realistic
    noise = np.random.normal(0, 5, n_samples)
    y += noise
    
    # Calculate signal-to-noise ratio
    signal_variance = np.var(np.dot(X_correlated[:, :n_informative], true_coefficients[:n_informative]))
    noise_variance = np.var(noise)
    snr = signal_variance / noise_variance
    print(f"Step 1.5: Signal-to-noise ratio: {snr:.4f}")
    
    return X_correlated, y, true_coefficients

# Generate data with 1000 samples, 100 features (20 informative, 80 correlated/irrelevant)
n_samples = 1000
n_features = 100
n_informative = 20

X, y, true_coef = generate_correlated_features(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=n_informative,
    correlation=0.8
)

print(f"Generated dataset with {n_samples} samples and {n_features} features")
print(f"Of these, {n_informative} features are truly informative")
print(f"The other {n_features - n_informative} features are either correlated or noise")

# Split the dataset
print("\nStep 1.6: Splitting data into training (70%) and test (30%) sets")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Standardize features
print("\nStep 1.7: Standardizing features (mean=0, std=1)")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Standardization is crucial for regularization methods as they are sensitive to feature scales")

# Part 2: Implement and compare regularization methods
print("\n2. Comparing Regularization Methods")
print("----------------------------------")
print("Step 2.1: Ridge Regression Theory")
print("Ridge regression adds an L2 penalty to the ordinary least squares objective:")
print("  min ||y - Xw||^2 + λ||w||^2")
print("This has a closed-form solution:")
print("  w_ridge = (X^T X + λI)^(-1) X^T y")
print("Ridge shrinks all coefficients toward zero but rarely makes them exactly zero")
print("\nStep 2.2: Lasso Regression Theory")
print("Lasso regression adds an L1 penalty to the ordinary least squares objective:")
print("  min ||y - Xw||^2 + λ||w||^1")
print("There is no closed-form solution for Lasso; it requires iterative optimization")
print("Lasso can shrink coefficients exactly to zero, performing feature selection")
print("\nStep 2.3: Elastic Net Theory")
print("Elastic Net combines L1 and L2 penalties:")
print("  min ||y - Xw||^2 + λ1||w||^1 + λ2||w||^2")
print("It balances the properties of Ridge and Lasso")
print("Like Lasso, it requires iterative optimization methods")

# Function to train models with different regularization parameters
def train_models(X_train, y_train, X_test, y_test, alphas):
    print("\nStep 2.4: Training models with different regularization strengths")
    results = {'ridge': [], 'lasso': [], 'elastic': []}
    
    for alpha in alphas:
        print(f"\nRegularization strength α = {alpha}:")
        
        # Ridge Regression
        print("  Training Ridge model...")
        start_time = time.time()
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)
        ridge_time = time.time() - start_time
        
        # Compute performance metrics
        ridge_pred = ridge.predict(X_test)
        ridge_mse = mean_squared_error(y_test, ridge_pred)
        ridge_r2 = r2_score(y_test, ridge_pred)
        ridge_nonzero = np.sum(np.abs(ridge.coef_) > 1e-6)
        print(f"  Ridge MSE: {ridge_mse:.4f}, R²: {ridge_r2:.4f}, Non-zero: {ridge_nonzero}/{len(ridge.coef_)}, Time: {ridge_time:.4f}s")
        
        # Lasso Regression
        print("  Training Lasso model...")
        start_time = time.time()
        lasso = Lasso(alpha=alpha, max_iter=10000, tol=1e-4)
        lasso.fit(X_train, y_train)
        lasso_time = time.time() - start_time
        
        # Compute performance metrics
        lasso_pred = lasso.predict(X_test)
        lasso_mse = mean_squared_error(y_test, lasso_pred)
        lasso_r2 = r2_score(y_test, lasso_pred)
        lasso_nonzero = np.sum(np.abs(lasso.coef_) > 1e-6)
        print(f"  Lasso MSE: {lasso_mse:.4f}, R²: {lasso_r2:.4f}, Non-zero: {lasso_nonzero}/{len(lasso.coef_)}, Time: {lasso_time:.4f}s")
        
        # Elastic Net Regression
        print("  Training Elastic Net model...")
        start_time = time.time()
        elastic = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000, tol=1e-4)
        elastic.fit(X_train, y_train)
        elastic_time = time.time() - start_time
        
        # Compute performance metrics
        elastic_pred = elastic.predict(X_test)
        elastic_mse = mean_squared_error(y_test, elastic_pred)
        elastic_r2 = r2_score(y_test, elastic_pred)
        elastic_nonzero = np.sum(np.abs(elastic.coef_) > 1e-6)
        print(f"  Elastic Net MSE: {elastic_mse:.4f}, R²: {elastic_r2:.4f}, Non-zero: {elastic_nonzero}/{len(elastic.coef_)}, Time: {elastic_time:.4f}s")
        
        # Store results for later analysis
        results['ridge'].append({
            'alpha': alpha,
            'coef': ridge.coef_,
            'mse': ridge_mse,
            'r2': ridge_r2,
            'nonzero': ridge_nonzero,
            'time': ridge_time
        })
        
        results['lasso'].append({
            'alpha': alpha,
            'coef': lasso.coef_,
            'mse': lasso_mse,
            'r2': lasso_r2,
            'nonzero': lasso_nonzero,
            'time': lasso_time
        })
        
        results['elastic'].append({
            'alpha': alpha,
            'coef': elastic.coef_,
            'mse': elastic_mse,
            'r2': elastic_r2,
            'nonzero': elastic_nonzero,
            'time': elastic_time
        })
    
    return results

# Try different regularization strengths
print("\nStep 2.5: Testing different regularization strengths (α)")
print("Lower α values mean less regularization (closer to ordinary least squares)")
print("Higher α values mean more regularization (more coefficient shrinkage)")
alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
results = train_models(X_train_scaled, y_train, X_test_scaled, y_test, alphas)

# Print comparative results for α=1.0 (middle value)
print("\nStep 2.6: Detailed comparison at α=1.0 (moderate regularization):")
alpha_idx = alphas.index(1.0)

print(f"Ridge Regression:")
print(f"  MSE: {results['ridge'][alpha_idx]['mse']:.4f}")
print(f"  R²: {results['ridge'][alpha_idx]['r2']:.4f}")
print(f"  Non-zero coefficients: {results['ridge'][alpha_idx]['nonzero']} out of {n_features}")
print(f"  Training time: {results['ridge'][alpha_idx]['time']:.4f} seconds")

print(f"\nLasso Regression:")
print(f"  MSE: {results['lasso'][alpha_idx]['mse']:.4f}")
print(f"  R²: {results['lasso'][alpha_idx]['r2']:.4f}")
print(f"  Non-zero coefficients: {results['lasso'][alpha_idx]['nonzero']} out of {n_features}")
print(f"  Training time: {results['lasso'][alpha_idx]['time']:.4f} seconds")

print(f"\nElastic Net (L1_ratio=0.5):")
print(f"  MSE: {results['elastic'][alpha_idx]['mse']:.4f}")
print(f"  R²: {results['elastic'][alpha_idx]['r2']:.4f}")
print(f"  Non-zero coefficients: {results['elastic'][alpha_idx]['nonzero']} out of {n_features}")
print(f"  Training time: {results['elastic'][alpha_idx]['time']:.4f} seconds")

print("\nStep 2.7: Analysis of coefficient magnitudes at α=1.0")
# Calculate some statistics about coefficients
ridge_coef = results['ridge'][alpha_idx]['coef']
lasso_coef = results['lasso'][alpha_idx]['coef']
elastic_coef = results['elastic'][alpha_idx]['coef']

print(f"Ridge: Mean absolute coefficient value: {np.mean(np.abs(ridge_coef)):.6f}")
print(f"Lasso: Mean absolute coefficient value: {np.mean(np.abs(lasso_coef)):.6f}")
print(f"Elastic Net: Mean absolute coefficient value: {np.mean(np.abs(elastic_coef)):.6f}")

print(f"Ridge: Max coefficient value: {np.max(np.abs(ridge_coef)):.6f}")
print(f"Lasso: Max coefficient value: {np.max(np.abs(lasso_coef)):.6f}")
print(f"Elastic Net: Max coefficient value: {np.max(np.abs(elastic_coef)):.6f}")

# Part 3: Visualizations
print("\n3. Creating Visualizations")
print("------------------------")
print("Step 3.1: Visualizing coefficient values for each method")

# Visualization 1: Coefficient values for each method
plt.figure(figsize=(15, 8))
alpha_idx = alphas.index(1.0)  # Use alpha=1.0 for visualization

# Plotting original coefficients
plt.subplot(2, 2, 1)
plt.stem(range(n_features), true_coef, markerfmt='go', linefmt='g-', basefmt='r-')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('True Coefficients')
plt.grid(True)
print("  - Top-left: True coefficients (only first 20 features are non-zero)")

# Plotting Ridge coefficients
plt.subplot(2, 2, 2)
plt.stem(range(n_features), results['ridge'][alpha_idx]['coef'], markerfmt='bo', linefmt='b-', basefmt='r-')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Ridge Coefficients ($\\alpha=1.0$)')
plt.grid(True)
print("  - Top-right: Ridge coefficients (most features have small non-zero values)")

# Plotting Lasso coefficients
plt.subplot(2, 2, 3)
plt.stem(range(n_features), results['lasso'][alpha_idx]['coef'], markerfmt='ro', linefmt='r-', basefmt='r-')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Lasso Coefficients ($\\alpha=1.0$)')
plt.grid(True)
print("  - Bottom-left: Lasso coefficients (many features are exactly zero)")

# Plotting Elastic Net coefficients
plt.subplot(2, 2, 4)
plt.stem(range(n_features), results['elastic'][alpha_idx]['coef'], markerfmt='mo', linefmt='m-', basefmt='r-')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Elastic Net Coefficients ($\\alpha=1.0$, L1\\_ratio=0.5)')
plt.grid(True)
print("  - Bottom-right: Elastic Net coefficients (combines properties of Ridge and Lasso)")

plt.tight_layout()
print("  Saving visualization to coefficient_comparison.png")
plt.savefig(os.path.join(save_dir, "coefficient_comparison.png"), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 2: Number of non-zero coefficients vs regularization strength
print("\nStep 3.2: Visualizing feature selection behavior across regularization strengths")
plt.figure(figsize=(10, 6))

ridge_nonzero = [result['nonzero'] for result in results['ridge']]
lasso_nonzero = [result['nonzero'] for result in results['lasso']]
elastic_nonzero = [result['nonzero'] for result in results['elastic']]

plt.semilogx(alphas, ridge_nonzero, 'bo-', label='Ridge')
plt.semilogx(alphas, lasso_nonzero, 'ro-', label='Lasso')
plt.semilogx(alphas, elastic_nonzero, 'mo-', label='Elastic Net')
plt.axhline(y=n_informative, color='g', linestyle='--', label='True Informative Features')

print("  This plot shows how each method selects features as regularization strength increases:")
print("  - Ridge (blue): Maintains most features regardless of regularization strength")
print("  - Lasso (red): Dramatically reduces features as regularization increases")
print("  - Elastic Net (purple): Behavior between Ridge and Lasso")
print("  - Green dashed line: Number of truly informative features (20)")

plt.xlabel('Regularization Strength ($\\alpha$)')
plt.ylabel('Number of Non-Zero Coefficients')
plt.title('Feature Selection by Regularization Method')
plt.legend()
plt.grid(True)
print("  Saving visualization to feature_selection.png")
plt.savefig(os.path.join(save_dir, "feature_selection.png"), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 3: Model performance vs regularization strength
print("\nStep 3.3: Visualizing model performance across regularization strengths")
plt.figure(figsize=(12, 10))

# MSE subplot
plt.subplot(2, 1, 1)
ridge_mse = [result['mse'] for result in results['ridge']]
lasso_mse = [result['mse'] for result in results['lasso']]
elastic_mse = [result['mse'] for result in results['elastic']]

plt.semilogx(alphas, ridge_mse, 'bo-', label='Ridge')
plt.semilogx(alphas, lasso_mse, 'ro-', label='Lasso')
plt.semilogx(alphas, elastic_mse, 'mo-', label='Elastic Net')

plt.xlabel('Regularization Strength ($\\alpha$)')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs Regularization Strength')
plt.legend()
plt.grid(True)
print("  Top plot shows Mean Squared Error vs regularization strength:")
print("  - Lower values indicate better prediction accuracy")
print("  - Note how error increases with too much regularization")

# R² subplot
plt.subplot(2, 1, 2)
ridge_r2 = [result['r2'] for result in results['ridge']]
lasso_r2 = [result['r2'] for result in results['lasso']]
elastic_r2 = [result['r2'] for result in results['elastic']]

plt.semilogx(alphas, ridge_r2, 'bo-', label='Ridge')
plt.semilogx(alphas, lasso_r2, 'ro-', label='Lasso')
plt.semilogx(alphas, elastic_r2, 'mo-', label='Elastic Net')

plt.xlabel('Regularization Strength ($\\alpha$)')
plt.ylabel('$R^2$ Score')
plt.title('$R^2$ vs Regularization Strength')
plt.legend()
plt.grid(True)
print("  Bottom plot shows R² score vs regularization strength:")
print("  - Higher values indicate better fit (1.0 is perfect)")
print("  - Shows the same trend as MSE but in reverse")

plt.tight_layout()
print("  Saving visualization to performance_comparison.png")
plt.savefig(os.path.join(save_dir, "performance_comparison.png"), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 4: Training time comparison
print("\nStep 3.4: Comparing computational efficiency")
plt.figure(figsize=(10, 6))

ridge_time = [result['time'] for result in results['ridge']]
lasso_time = [result['time'] for result in results['lasso']]
elastic_time = [result['time'] for result in results['elastic']]

plt.semilogx(alphas, ridge_time, 'bo-', label='Ridge')
plt.semilogx(alphas, lasso_time, 'ro-', label='Lasso')
plt.semilogx(alphas, elastic_time, 'mo-', label='Elastic Net')

plt.xlabel('Regularization Strength ($\\alpha$)')
plt.ylabel('Training Time (seconds)')
plt.title('Computational Efficiency Comparison')
plt.legend()
plt.grid(True)
print("  This plot shows training time for each method:")
print("  - Ridge is fastest due to closed-form solution")
print("  - Lasso and Elastic Net require iterative optimization")
print("  - Higher regularization can sometimes converge faster for iterative methods")

print("  Saving visualization to computational_efficiency.png")
plt.savefig(os.path.join(save_dir, "computational_efficiency.png"), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 5: L1 and L2 constraint regions
print("\nStep 3.5: Visualizing geometric interpretation of L1 and L2 constraints")
plt.figure(figsize=(10, 8))

# Create a grid of points
x = np.linspace(-1.5, 1.5, 1000)
y = np.linspace(-1.5, 1.5, 1000)
X, Y = np.meshgrid(x, y)
Z_l1 = np.abs(X) + np.abs(Y)  # L1 norm
Z_l2 = np.sqrt(X**2 + Y**2)   # L2 norm

print("  Creating geometric visualization of constraint regions:")
print("  - L1 norm: |w₁| + |w₂| ≤ t (diamond shape)")
print("  - L2 norm: √(w₁² + w₂²) ≤ t (circular shape)")
print("  - These constraints represent the regularization terms")

# Plot L1 constraint region
plt.subplot(1, 2, 1)
plt.contour(X, Y, Z_l1, levels=[1], colors='red', linewidths=2)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.grid(True)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.gca().set_aspect('equal')
plt.title('L1 Norm Constraint Region (Lasso)')
plt.xlabel('$w_1$')
plt.ylabel('$w_2$')

# Plot L2 constraint region
plt.subplot(1, 2, 2)
plt.contour(X, Y, Z_l2, levels=[1], colors='blue', linewidths=2)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.grid(True)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.gca().set_aspect('equal')
plt.title('L2 Norm Constraint Region (Ridge)')
plt.xlabel('$w_1$')
plt.ylabel('$w_2$')

plt.tight_layout()
print("  Saving visualization to constraint_regions.png")
plt.savefig(os.path.join(save_dir, "constraint_regions.png"), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 6: Geometric explanation of L1 sparsity
print("\nStep 3.6: Demonstrating why L1 regularization promotes sparsity")
plt.figure(figsize=(12, 10))

# Create contour lines for the loss function (ellipse)
theta = np.linspace(0, 2*np.pi, 1000)
a, b = 1.0, 0.5  # Ellipse parameters
x_ellipse = a * np.cos(theta)
y_ellipse = b * np.sin(theta)

# Add small offset to move the center of the ellipse
x_ellipse += 0.5
y_ellipse += 0.2

# Create L1 and L2 norm boundaries
t = np.linspace(-2, 2, 1000)
l1_x = t
l1_y = 1 - np.abs(t)
l1_y[l1_y < 0] = np.nan  # Clip values below 0

l2_x = np.cos(theta)
l2_y = np.sin(theta)

print("  Creating geometric explanation of sparsity:")
print("  - The ellipses represent contours of the loss function")
print("  - The boundaries represent the regularization constraints")
print("  - The optimal solution is where the loss contour first touches the constraint region")
print("  - For L1 (Lasso), this often happens at a corner where some coefficients = 0")
print("  - For L2 (Ridge), this usually happens away from axes, giving non-zero coefficients")

# Plot L1 norm and loss function
plt.subplot(1, 2, 1)
plt.plot(x_ellipse, y_ellipse, 'g-', label='Loss contours')
plt.plot(l1_x, l1_y, 'r-', label='L1 constraint')
plt.plot(l1_x, -l1_y, 'r-')
plt.plot(-l1_x, l1_y, 'r-')
plt.plot(-l1_x, -l1_y, 'r-')

# Mark the intersection point that occurs at a corner (sparsity)
intersect_x = 1.0  # Example point where L1 norm hits the corner
intersect_y = 0.0
plt.scatter([intersect_x], [intersect_y], color='red', s=100, zorder=5)

plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.grid(True)
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.gca().set_aspect('equal')
plt.title('L1 Regularization (Lasso): Promotes Sparsity')
plt.xlabel('$w_1$')
plt.ylabel('$w_2$')
plt.legend()

# Annotate the sparse solution
plt.annotate('Sparse solution\n($w_2 = 0$)', xy=(intersect_x, intersect_y), 
             xytext=(intersect_x+0.3, intersect_y+0.5),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

# Plot L2 norm and loss function
plt.subplot(1, 2, 2)
plt.plot(x_ellipse, y_ellipse, 'g-', label='Loss contours')
plt.plot(l2_x, l2_y, 'b-', label='L2 constraint')

# Mark the intersection point that is not at the corner
intersect_x = 0.8  # Example intersection point
intersect_y = 0.4
plt.scatter([intersect_x], [intersect_y], color='blue', s=100, zorder=5)

plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.grid(True)
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.gca().set_aspect('equal')
plt.title('L2 Regularization (Ridge): Non-sparse Solution')
plt.xlabel('$w_1$')
plt.ylabel('$w_2$')
plt.legend()

# Annotate the non-sparse solution
plt.annotate('Non-sparse solution\n(both $w_1, w_2 \\neq 0$)', xy=(intersect_x, intersect_y), 
             xytext=(intersect_x+0.3, intersect_y+0.5),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

plt.tight_layout()
print("  Saving visualization to sparsity_explanation.png")
plt.savefig(os.path.join(save_dir, "sparsity_explanation.png"), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 7: Effect on highly correlated features
print("\nStep 3.7: Demonstrating handling of correlated features")
# Create a correlation matrix
corr_matrix = np.corrcoef(X.T)

# Select a subset of features for visualization
n_selected = 30  # First 30 features
corr_subset = corr_matrix[:n_selected, :n_selected]

plt.figure(figsize=(15, 10))

# Plot correlation matrix
plt.subplot(2, 2, 1)
plt.imshow(corr_subset, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Correlation')
plt.title('Feature Correlation Matrix')
plt.xlabel('Feature Index')
plt.ylabel('Feature Index')
print("  Top-left: Correlation matrix for the first 30 features")
print("  - Red indicates positive correlation")
print("  - Blue indicates negative correlation")
print("  - Look for blocks of similar color indicating correlated feature groups")

# Plot coefficients for selected features for Ridge
plt.subplot(2, 2, 2)
selected_coefs_ridge = results['ridge'][alpha_idx]['coef'][:n_selected]
plt.stem(range(n_selected), selected_coefs_ridge, markerfmt='bo', linefmt='b-', basefmt='r-')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.title('Ridge Coefficients for Correlated Features')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
print("  Top-right: Ridge coefficients for these features")
print("  - Notice how Ridge tends to distribute weight among correlated features")

# Plot coefficients for selected features for Lasso
plt.subplot(2, 2, 3)
selected_coefs_lasso = results['lasso'][alpha_idx]['coef'][:n_selected]
plt.stem(range(n_selected), selected_coefs_lasso, markerfmt='ro', linefmt='r-', basefmt='r-')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.title('Lasso Coefficients for Correlated Features')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
print("  Bottom-left: Lasso coefficients for these features")
print("  - Notice how Lasso tends to select one feature from each correlated group")
print("  - This creates a sparse model but may be unstable across different samples")

# Plot coefficients for selected features for Elastic Net
plt.subplot(2, 2, 4)
selected_coefs_elastic = results['elastic'][alpha_idx]['coef'][:n_selected]
plt.stem(range(n_selected), selected_coefs_elastic, markerfmt='mo', linefmt='m-', basefmt='r-')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.title('Elastic Net Coefficients for Correlated Features')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
print("  Bottom-right: Elastic Net coefficients for these features")
print("  - Elastic Net balances between Ridge and Lasso")
print("  - It may select multiple correlated features but with reduced weights")

plt.tight_layout()
print("  Saving visualization to correlated_features.png")
plt.savefig(os.path.join(save_dir, "correlated_features.png"), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 8: Mathematical explanation of coefficient paths
print("\nStep 3.8: Visualizing coefficient paths as regularization strength changes")
plt.figure(figsize=(15, 10))

# Select a subset of informative features for visualization
n_paths = 10  # Number of coefficient paths to show
selected_features = np.arange(n_paths)
    
# Plot Ridge coefficient paths
plt.subplot(1, 3, 1)
for i in selected_features:
    coef_path = [results['ridge'][j]['coef'][i] for j in range(len(alphas))]
    plt.semilogx(alphas, coef_path, 'o-', label=f'Feature {i}')

plt.xlabel('Regularization Strength ($\\alpha$)')
plt.ylabel('Coefficient Value')
plt.title('Ridge Coefficient Paths')
plt.grid(True)
plt.legend(loc='upper right', fontsize='small')
print("  Left: Ridge coefficient paths")
print("  - Shows how each coefficient value changes with regularization strength")
print("  - Ridge coefficients shrink smoothly but rarely reach exactly zero")

# Plot Lasso coefficient paths
plt.subplot(1, 3, 2)
for i in selected_features:
    coef_path = [results['lasso'][j]['coef'][i] for j in range(len(alphas))]
    plt.semilogx(alphas, coef_path, 'o-', label=f'Feature {i}')

plt.xlabel('Regularization Strength ($\\alpha$)')
plt.ylabel('Coefficient Value')
plt.title('Lasso Coefficient Paths')
plt.grid(True)
plt.legend(loc='upper right', fontsize='small')
print("  Middle: Lasso coefficient paths")
print("  - Lasso coefficients reach zero at different regularization strengths")
print("  - This creates a natural feature selection mechanism")

# Plot Elastic Net coefficient paths
plt.subplot(1, 3, 3)
for i in selected_features:
    coef_path = [results['elastic'][j]['coef'][i] for j in range(len(alphas))]
    plt.semilogx(alphas, coef_path, 'o-', label=f'Feature {i}')

plt.xlabel('Regularization Strength ($\\alpha$)')
plt.ylabel('Coefficient Value')
plt.title('Elastic Net Coefficient Paths')
plt.grid(True)
plt.legend(loc='upper right', fontsize='small')
print("  Right: Elastic Net coefficient paths")
print("  - Behavior between Ridge and Lasso")
print("  - Some coefficients reach zero, but the paths are more stable than Lasso")

plt.tight_layout()
print("  Saving visualization to coefficient_paths.png")
plt.savefig(os.path.join(save_dir, "coefficient_paths.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAll visualizations saved to: {save_dir}")
print("\nSummary of Findings:")
print("-------------------")
print("1. Ridge Regression: Retains most features but shrinks coefficients uniformly")
print("2. Lasso Regression: Selects a sparse subset of features by setting many to zero")
print("3. Elastic Net: Combines the benefits of Ridge and Lasso")
print("4. For correlated features, Lasso tends to select one from each correlated group")
print("5. Ridge tends to distribute weight among correlated features")
print("6. Elastic Net offers a compromise for handling correlation")
print("7. L1 regularization promotes sparsity due to the geometric properties of its constraint region")
print("8. Lasso and Elastic Net are generally more interpretable when many features are irrelevant")
print("9. Ridge is computationally more efficient than Lasso or Elastic Net") 