import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_7_Quiz_24")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
# Allow LaTeX rendering
plt.rcParams['text.usetex'] = False  # Keep this False initially to avoid extra dependencies
plt.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern font for math

# Generate synthetic data with 8 features, only 3 relevant
np.random.seed(42)
n_samples = 200
n_features = 8
n_informative = 3

# Create the true coefficients (only 3 non-zero)
true_coef = np.zeros(n_features)
true_coef[:n_informative] = np.array([5.0, -3.5, 2.0])  # Only first 3 features are relevant
print("True coefficients:", true_coef)

# Generate feature matrix
X = np.random.randn(n_samples, n_features)

# Add correlation between features 4 and 5
X[:, 4] = 0.7 * X[:, 3] + 0.3 * np.random.randn(n_samples)

# Generate target with noise
y = np.dot(X, true_coef) + np.random.normal(0, 1, size=n_samples)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Fit models with different regularization methods
def fit_models():
    ols = LinearRegression()
    ridge = Ridge(alpha=1.0)
    lasso = Lasso(alpha=0.1)
    elasticnet = ElasticNet(alpha=0.1, l1_ratio=0.5)
    
    ols.fit(X_train_scaled, y_train)
    ridge.fit(X_train_scaled, y_train)
    lasso.fit(X_train_scaled, y_train)
    elasticnet.fit(X_train_scaled, y_train)
    
    return {
        'No Regularization': ols,
        'Ridge': ridge,
        'Lasso': lasso,
        'ElasticNet': elasticnet
    }

models = fit_models()

# Visualization 1: Compare coefficients of different regularization methods
plt.figure(figsize=(14, 7))
feature_labels = [f'X{i+1}' for i in range(n_features)]
x_pos = np.arange(len(feature_labels))
width = 0.2

# Create a bar plot for each model
for i, (name, model) in enumerate(models.items()):
    plt.bar(x_pos + (i - 1.5) * width, model.coef_, width, label=name, alpha=0.7)

plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.xticks(x_pos, feature_labels)
plt.title('Coefficient Values for Different Regularization Methods', fontsize=14)
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Coefficient Value', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "regularization_comparison.png"), dpi=300, bbox_inches='tight')
plt.close()

# 2. Create a table summarizing coefficient properties
def count_near_zero(coef, threshold=0.01):
    return np.sum(np.abs(coef) < threshold)

def count_exact_zero(coef, threshold=1e-10):
    return np.sum(np.abs(coef) < threshold)

results = {}
for name, model in models.items():
    coef = model.coef_
    results[name] = {
        'Most Approach Zero': count_near_zero(coef) > n_features / 2,
        'Some Exactly Zero': count_exact_zero(coef) > 0,
        'Non-zeros Shrink Proportionally': name in ['Ridge', 'No Regularization']
    }

results_df = pd.DataFrame(results).T
print("\nRegularization Methods Comparison:")
print(results_df)

# Print table in markdown format for direct inclusion in the markdown file
print("\nMarkdown Table:")
print("| Regularization Method | Most coefficients approach zero? | Some coefficients exactly zero? | All non-zero coefficients shrink proportionally? |")
print("| --------------------- | -------------------------------- | ------------------------------- | ------------------------------------------------ |")
for method, properties in results.items():
    row = f"| {method} | {'Yes' if properties['Most Approach Zero'] else 'No'} | {'Yes' if properties['Some Exactly Zero'] else 'No'} | {'Yes' if properties['Non-zeros Shrink Proportionally'] else 'No'} |"
    print(row)

# 3. Comparing Ridge and Lasso on a zero coefficient
alphas = np.logspace(-3, 3, 20)
ridge_coefs = []
lasso_coefs = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    lasso = Lasso(alpha=alpha)
    
    ridge.fit(X_train_scaled, y_train)
    lasso.fit(X_train_scaled, y_train)
    
    ridge_coefs.append(ridge.coef_[3])  # 4th coefficient (w3) which is truly zero
    lasso_coefs.append(lasso.coef_[3])  # 4th coefficient (w3) which is truly zero

plt.figure(figsize=(12, 7))
plt.semilogx(alphas, ridge_coefs, 'bo-', label='Ridge')
plt.semilogx(alphas, lasso_coefs, 'ro-', label='Lasso')
plt.axhline(y=0, color='k', linestyle='--', label='True Value (w3=0)')
plt.title('Ridge vs Lasso: Estimate of Zero Coefficient (w3) vs Regularization Strength', fontsize=14)
plt.xlabel('Regularization Parameter (Alpha)', fontsize=12)
plt.ylabel('Coefficient Value', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "zero_coefficient_comparison.png"), dpi=300, bbox_inches='tight')
plt.close()

# 4. Examine behavior with correlated features
alphas = np.logspace(-3, 3, 20)
ridge_coefs_corr = []
lasso_coefs_corr = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    lasso = Lasso(alpha=alpha)
    
    ridge.fit(X_train_scaled, y_train)
    lasso.fit(X_train_scaled, y_train)
    
    # Extract coefficients for the correlated features (3 and 4)
    ridge_coefs_corr.append([ridge.coef_[3], ridge.coef_[4]])
    lasso_coefs_corr.append([lasso.coef_[3], lasso.coef_[4]])

ridge_coefs_corr = np.array(ridge_coefs_corr)
lasso_coefs_corr = np.array(lasso_coefs_corr)

# Plot the coefficients of correlated features
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.semilogx(alphas, ridge_coefs_corr[:, 0], 'b-', marker='o', label='Feature X4')
plt.semilogx(alphas, ridge_coefs_corr[:, 1], 'b--', marker='s', label='Feature X5 (correlated)')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.title('Ridge Regression: Correlated Features', fontsize=14)
plt.xlabel('Regularization Parameter (Alpha)', fontsize=12)
plt.ylabel('Coefficient Value', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)

plt.subplot(1, 2, 2)
plt.semilogx(alphas, lasso_coefs_corr[:, 0], 'r-', marker='o', label='Feature X4')
plt.semilogx(alphas, lasso_coefs_corr[:, 1], 'r--', marker='s', label='Feature X5 (correlated)')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.title('Lasso Regression: Correlated Features', fontsize=14)
plt.xlabel('Regularization Parameter (Alpha)', fontsize=12)
plt.ylabel('Coefficient Value', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "correlated_features.png"), dpi=300, bbox_inches='tight')
plt.close()

# 5. Calculate correlation matrix to visualize feature correlation
correlation = np.corrcoef(X.T)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", 
            xticklabels=feature_labels, yticklabels=feature_labels)
plt.title('Feature Correlation Matrix', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "correlation_matrix.png"), dpi=300, bbox_inches='tight')
plt.close()

# 6. Geometric visualization of L1 vs L2 regularization constraints
plt.figure(figsize=(10, 10))

# Create data points for visualization
theta = np.linspace(0, 2*np.pi, 1000)
x_circle = np.cos(theta)
y_circle = np.sin(theta)

# Generate points for L1 norm (diamond shape)
t = np.linspace(0, 2*np.pi, 1000)
x_diamond = np.zeros_like(t)
y_diamond = np.zeros_like(t)

# Each quadrant of the diamond
mask1 = (t >= 0) & (t < np.pi/2)
mask2 = (t >= np.pi/2) & (t < np.pi)
mask3 = (t >= np.pi) & (t < 3*np.pi/2)
mask4 = (t >= 3*np.pi/2) & (t <= 2*np.pi)

x_diamond[mask1] = 1 - 2*t[mask1]/np.pi
y_diamond[mask1] = 2*t[mask1]/np.pi

x_diamond[mask2] = -2*(t[mask2]-np.pi/2)/np.pi
y_diamond[mask2] = 1 - 2*(t[mask2]-np.pi/2)/np.pi

x_diamond[mask3] = -1 + 2*(t[mask3]-np.pi)/np.pi
y_diamond[mask3] = -2*(t[mask3]-np.pi)/np.pi

x_diamond[mask4] = 2*(t[mask4]-3*np.pi/2)/np.pi
y_diamond[mask4] = -1 + 2*(t[mask4]-3*np.pi/2)/np.pi

# Plot L2 norm (circle) and L1 norm (diamond)
plt.plot(x_circle, y_circle, 'b-', linewidth=2, label='L2 Norm = 1 (Ridge)')
plt.plot(x_diamond, y_diamond, 'r-', linewidth=2, label='L1 Norm = 1 (Lasso)')

# Add contours of a typical loss function (elliptical contours)
# Hypothetical loss function contours
for r in [0.4, 0.8, 1.2, 1.6, 2.0]:
    # Make it slightly elongated to represent correlation
    plt.plot(r*1.2*x_circle, r*0.8*y_circle, 'g--', alpha=0.5)

# Add optimal points where contours touch the constraint regions
plt.plot([0, 1, 0, -1, 0], [1, 0, -1, 0, 1], 'ro', markersize=8)  # L1 vertices (often optimal)
plt.plot([0.707, 0, -0.707, 0], [0.707, 1, -0.707, -1], 'bo', markersize=8, alpha=0.7)  # Some L2 points

# Add decoration
plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.2)
plt.grid(True, alpha=0.3)
plt.title('Geometric Interpretation: L1 vs L2 Regularization Constraints', fontsize=14)
# Fix the glyph issue by using LaTeX notation
plt.xlabel(r'$\beta_1$', fontsize=14)
plt.ylabel(r'$\beta_2$', fontsize=14)
plt.legend(fontsize=12)
plt.axis('equal')
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "regularization_geometry.png"), dpi=300, bbox_inches='tight')
plt.close()

# 7. NEW: Regularization paths comparison
alphas_path = np.logspace(-3, 2, 50)
ridge_all_coefs = []
lasso_all_coefs = []

for alpha in alphas_path:
    ridge = Ridge(alpha=alpha)
    lasso = Lasso(alpha=alpha)
    
    ridge.fit(X_train_scaled, y_train)
    lasso.fit(X_train_scaled, y_train)
    
    ridge_all_coefs.append(ridge.coef_.copy())
    lasso_all_coefs.append(lasso.coef_.copy())

ridge_all_coefs = np.array(ridge_all_coefs)
lasso_all_coefs = np.array(lasso_all_coefs)

# Plot regularization paths
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
for i in range(n_features):
    plt.semilogx(alphas_path, ridge_all_coefs[:, i], '-', linewidth=2, alpha=0.7, 
                 label=f'X{i+1}' if i < 3 else None)
    # Highlight true non-zero coefficients with thicker lines
    if i < 3:
        plt.semilogx(alphas_path, ridge_all_coefs[:, i], '-', linewidth=3)

plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.title('Ridge Regularization Path', fontsize=14)
plt.xlabel('Regularization Parameter (Alpha)', fontsize=12)
plt.ylabel('Coefficient Value', fontsize=12)
plt.grid(True)
# Only include legend for the first 3 (relevant) features
plt.legend(fontsize=10, loc='upper right')

plt.subplot(1, 2, 2)
for i in range(n_features):
    plt.semilogx(alphas_path, lasso_all_coefs[:, i], '-', linewidth=2, alpha=0.7,
                 label=f'X{i+1}' if i < 3 else None)
    # Highlight true non-zero coefficients with thicker lines
    if i < 3:
        plt.semilogx(alphas_path, lasso_all_coefs[:, i], '-', linewidth=3)

plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.title('Lasso Regularization Path', fontsize=14)
plt.xlabel('Regularization Parameter (Alpha)', fontsize=12)
plt.ylabel('Coefficient Value', fontsize=12)
plt.grid(True)
# Only include legend for the first 3 (relevant) features
plt.legend(fontsize=10, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "regularization_paths.png"), dpi=300, bbox_inches='tight')
plt.close()

# Print summary
print("\nImages saved to:", save_dir)
print("\nSummary of findings:")
print("1. Coefficients comparison:")
for name, model in models.items():
    zero_count = count_exact_zero(model.coef_)
    near_zero = count_near_zero(model.coef_)
    print(f"  - {name}: {zero_count} exact zeros, {near_zero} near-zero coefficients")

print("\n2. Why Lasso is useful for this scenario:")
print("  - The true model has only 3 relevant features out of 8")
print("  - Lasso can identify these relevant features through feature selection")
print("  - Our Lasso model identified:", n_features - count_near_zero(models['Lasso'].coef_), "features as important")

print("\n3. For the zero coefficient (w3):")
print("  - Ridge gradually shrinks it toward zero as alpha increases")
print("  - Lasso sets it exactly to zero beyond a certain threshold of alpha")

print("\n4. With correlated features (X4 and X5):")
print("  - Ridge tends to shrink both coefficients similarly")
print("  - Lasso tends to select one feature and set the other to zero")
print("  - When features are correlated, Ridge is often preferred for prediction")
print("  - When feature selection is the goal, Lasso provides a clearer selection") 