import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
import os
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images", "L3_7_Quiz_26")
os.makedirs(images_dir, exist_ok=True)

# Set the style for plotting
plt.style.use('seaborn-v0_8-whitegrid')

# Original coefficient vector from the problem
w_original = np.array([0.8, 2.1, 0.05, 3.7, -1.2, 0.02, 4.5, -2.3, 0.09, 0.01, 1.4, -0.8])
feature_names = [f"X{i+1}" for i in range(len(w_original))]

# Calculate L1 and L2 norms
l1_norm = np.sum(np.abs(w_original))
l2_norm = np.sqrt(np.sum(w_original**2))

print(f"Original coefficient vector: {w_original}")
print(f"L1 norm: {l1_norm:.4f}")
print(f"L2 norm: {l2_norm:.4f}")

# Generate synthetic data for demonstration
np.random.seed(42)
n_samples = 200
X = np.random.randn(n_samples, len(w_original))
y = X @ w_original + np.random.randn(n_samples) * 0.5

# Function to apply regularization with varying strengths
def apply_regularization():
    alphas = np.logspace(-3, 2, 20)  # Range of regularization strengths
    
    ridge_coefs = []
    lasso_coefs = []
    
    for alpha in alphas:
        # Ridge regression
        ridge = Ridge(alpha=alpha, fit_intercept=False)
        ridge.fit(X, y)
        ridge_coefs.append(ridge.coef_.copy())
        
        # Lasso regression
        lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000, tol=1e-4)
        lasso.fit(X, y)
        lasso_coefs.append(lasso.coef_.copy())
    
    return alphas, np.array(ridge_coefs), np.array(lasso_coefs)

alphas, ridge_coefs, lasso_coefs = apply_regularization()

# Visualization 1: Regularization paths for Ridge and Lasso
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Ridge regularization path
for i in range(len(w_original)):
    ax1.semilogx(alphas, ridge_coefs[:, i], '-', label=feature_names[i], alpha=0.7)
ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax1.set_xlabel('Regularization Parameter (Alpha)', fontsize=12)
ax1.set_ylabel('Coefficient Value', fontsize=12)
ax1.set_title('Ridge (L2) Regularization Path', fontsize=14)
ax1.grid(True)

# Lasso regularization path
for i in range(len(w_original)):
    ax2.semilogx(alphas, lasso_coefs[:, i], '-', label=feature_names[i], alpha=0.7)
ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax2.set_xlabel('Regularization Parameter (Alpha)', fontsize=12)
ax2.set_ylabel('Coefficient Value', fontsize=12)
ax2.set_title('Lasso (L1) Regularization Path', fontsize=14)
ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, "regularization_paths.png"), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 2: Coefficient magnitude analysis
# Categorize coefficients by magnitude
small_indices = np.where(np.abs(w_original) < 0.1)[0]
medium_indices = np.where((np.abs(w_original) >= 0.1) & (np.abs(w_original) < 2.0))[0]
large_indices = np.where(np.abs(w_original) >= 2.0)[0]

# Create a categorization table
magnitude_categories = pd.DataFrame({
    'Feature': [f"X{i+1}" for i in range(len(w_original))],
    'Coefficient': w_original,
    'Absolute Value': np.abs(w_original),
    'Category': ['Small (<0.1)' if i in small_indices else 
                'Medium (0.1-2.0)' if i in medium_indices else 
                'Large (>=2.0)' for i in range(len(w_original))]
})
magnitude_categories = magnitude_categories.sort_values('Absolute Value', ascending=False)

# Print the categorized indices and categories
print("\nCoefficients Categorized by Magnitude:")
print(magnitude_categories)

# Print summary of coefficient categories
print("\nSummary of Coefficient Categories:")
print(f"Large coefficients (>= 2.0): {len(large_indices)} coefficients")
print(f"Medium coefficients (0.1-2.0): {len(medium_indices)} coefficients")
print(f"Small coefficients (< 0.1): {len(small_indices)} coefficients")

# Calculate average shrinkage ratio for each magnitude group
def calculate_shrinkage(coefs, indices, alpha_idx):
    if len(indices) == 0:
        return np.nan
    original_values = np.abs(w_original[indices])
    new_values = np.abs(coefs[alpha_idx, indices])
    # Avoid division by zero
    shrinkage_ratios = np.where(original_values > 0, new_values / original_values, 0)
    return np.mean(shrinkage_ratios)

# Select a few representative alpha values for shrinkage comparison
alpha_indices = [5, 10, 15]  # Mild, medium, strong regularization
alpha_values = [alphas[i] for i in alpha_indices]

# Create tables of shrinkage ratios
ridge_shrinkage = np.zeros((3, 3))  # 3 magnitude groups, 3 alpha values
lasso_shrinkage = np.zeros((3, 3))
lasso_zero_counts = np.zeros((3, 3), dtype=int)

for i, alpha_idx in enumerate(alpha_indices):
    ridge_shrinkage[0, i] = calculate_shrinkage(ridge_coefs, large_indices, alpha_idx)
    ridge_shrinkage[1, i] = calculate_shrinkage(ridge_coefs, medium_indices, alpha_idx)
    ridge_shrinkage[2, i] = calculate_shrinkage(ridge_coefs, small_indices, alpha_idx)
    
    lasso_shrinkage[0, i] = calculate_shrinkage(lasso_coefs, large_indices, alpha_idx)
    lasso_shrinkage[1, i] = calculate_shrinkage(lasso_coefs, medium_indices, alpha_idx)
    lasso_shrinkage[2, i] = calculate_shrinkage(lasso_coefs, small_indices, alpha_idx)
    
    # Count zero coefficients for Lasso
    lasso_zero_counts[0, i] = np.sum(np.abs(lasso_coefs[alpha_idx, large_indices]) < 1e-6)
    lasso_zero_counts[1, i] = np.sum(np.abs(lasso_coefs[alpha_idx, medium_indices]) < 1e-6)
    lasso_zero_counts[2, i] = np.sum(np.abs(lasso_coefs[alpha_idx, small_indices]) < 1e-6)

# Create DataFrame for shrinkage analysis
magnitude_labels = ["Large", "Medium", "Small"]
ridge_shrinkage_df = pd.DataFrame(ridge_shrinkage, 
                                 index=magnitude_labels, 
                                 columns=[f"Alpha = {alpha:.3f}" for alpha in alpha_values])

lasso_shrinkage_df = pd.DataFrame(lasso_shrinkage, 
                                 index=magnitude_labels, 
                                 columns=[f"Alpha = {alpha:.3f}" for alpha in alpha_values])

lasso_zeros_df = pd.DataFrame({
    f"Alpha = {alpha_values[0]:.3f}": [f"{lasso_zero_counts[0, 0]}/{len(large_indices)}", 
                                    f"{lasso_zero_counts[1, 0]}/{len(medium_indices)}", 
                                    f"{lasso_zero_counts[2, 0]}/{len(small_indices)}"],
    f"Alpha = {alpha_values[1]:.3f}": [f"{lasso_zero_counts[0, 1]}/{len(large_indices)}", 
                                    f"{lasso_zero_counts[1, 1]}/{len(medium_indices)}", 
                                    f"{lasso_zero_counts[2, 1]}/{len(small_indices)}"],
    f"Alpha = {alpha_values[2]:.3f}": [f"{lasso_zero_counts[0, 2]}/{len(large_indices)}", 
                                    f"{lasso_zero_counts[1, 2]}/{len(medium_indices)}", 
                                    f"{lasso_zero_counts[2, 2]}/{len(small_indices)}"]
}, index=magnitude_labels)

# Print shrinkage analysis
print("\nShrinkage Ratio Analysis (coefficient / original):")
print("\nRidge Regression:")
print(ridge_shrinkage_df)

print("\nLasso Regression:")
print(lasso_shrinkage_df)

print("\nLasso Zero Coefficients Count:")
print(lasso_zeros_df)

# Calculate theoretical shrinkage factors for orthogonal features
def ridge_theoretical_shrinkage(alpha):
    """
    Calculate the theoretical shrinkage factor for ridge regression 
    with orthogonal features: beta_ridge = beta_ols / (1 + alpha)
    """
    return 1 / (1 + alpha)

def lasso_theoretical_shrinkage(original_coef, alpha):
    """
    Calculate the theoretical shrinkage for lasso regression with orthogonal features
    using the soft thresholding operator:
    beta_lasso = sign(beta_ols) * max(|beta_ols| - alpha/2, 0)
    """
    abs_coef = np.abs(original_coef)
    threshold = alpha/2
    if abs_coef <= threshold:
        return 0
    else:
        return (abs_coef - threshold) / abs_coef

# Calculate theoretical shrinkage for different coefficient magnitudes
theoretical_alphas = alpha_values
theoretical_examples = {
    "Large (4.5)": 4.5,
    "Medium (1.4)": 1.4,
    "Small (0.05)": 0.05
}

# Create theoretical shrinkage table
theoretical_data = []
for alpha in theoretical_alphas:
    ridge_factor = ridge_theoretical_shrinkage(alpha)
    row = {"Alpha": f"{alpha:.3f}", "Ridge Factor": f"{ridge_factor:.3f}"}
    
    for label, coef in theoretical_examples.items():
        lasso_factor = lasso_theoretical_shrinkage(coef, alpha)
        row[f"Ridge ({label})"] = f"{coef * ridge_factor:.3f}"
        row[f"Lasso ({label})"] = f"{coef * lasso_factor:.3f}"
    
    theoretical_data.append(row)

theoretical_df = pd.DataFrame(theoretical_data)

print("\nTheoretical Shrinkage Analysis (Orthogonal Features):")
print(theoretical_df)

# Visualization 3: Heat map of coefficient shrinkage at different alpha levels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

# Sort the coefficients by magnitude for better visualization
sorted_indices = np.argsort(np.abs(w_original))[::-1]
sorted_values = w_original[sorted_indices]
sorted_names = [feature_names[i] for i in sorted_indices]

# Select a subset of alpha values for visualization
alpha_subset = list(range(0, len(alphas), 2))
alpha_labels = [f"{alpha:.3f}" for alpha in alphas[alpha_subset]]

# Ridge coefficients heatmap
ridge_data = ridge_coefs[alpha_subset][:, sorted_indices].T
cmap_ridge = LinearSegmentedColormap.from_list("ridge_cmap", ["#4575B4", "white", "#D73027"])
sns.heatmap(ridge_data / np.abs(sorted_values).reshape(-1, 1), 
            cmap=cmap_ridge, center=0, vmin=-1.2, vmax=1.2,
            ax=ax1, xticklabels=alpha_labels, yticklabels=sorted_names)
ax1.set_title("Ridge Coefficient Shrinkage (relative to original)", fontsize=14)
ax1.set_xlabel("Regularization Parameter (Alpha)", fontsize=12)
ax1.set_ylabel("Feature", fontsize=12)

# Lasso coefficients heatmap
lasso_data = lasso_coefs[alpha_subset][:, sorted_indices].T
cmap_lasso = LinearSegmentedColormap.from_list("lasso_cmap", ["#4575B4", "white", "#D73027"])
sns.heatmap(lasso_data / np.abs(sorted_values).reshape(-1, 1), 
            cmap=cmap_lasso, center=0, vmin=-1.2, vmax=1.2,
            ax=ax2, xticklabels=alpha_labels, yticklabels=sorted_names)
ax2.set_title("Lasso Coefficient Shrinkage (relative to original)", fontsize=14)
ax2.set_xlabel("Regularization Parameter (Alpha)", fontsize=12)
ax2.set_ylabel("Feature", fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, "coefficient_shrinkage_heatmap.png"), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 4: Regularization geometry for 2D case
def plot_constraint_regions():
    plt.figure(figsize=(10, 10))
    
    # Create data
    x = np.linspace(-2, 2, 1000)
    y = np.linspace(-2, 2, 1000)
    X, Y = np.meshgrid(x, y)
    
    # Create contours for L1 and L2 norms
    L1_norm = np.abs(X) + np.abs(Y)
    L2_norm = np.sqrt(X**2 + Y**2)
    
    # Plot constraint regions
    plt.contour(X, Y, L1_norm, levels=[1], colors='r', linewidths=2)
    plt.contour(X, Y, L2_norm, levels=[1], colors='b', linewidths=2)
    
    # Loss function contours (elliptical to show the effect of different solutions)
    contour_levels = np.arange(0.2, 2, 0.2)
    
    # Elliptical contours (elongated in one direction to simulate correlated features)
    Z = 2*X**2 + Y**2/2  # Elongated along Y-axis
    plt.contour(X, Y, Z, levels=contour_levels**2, colors='green', alpha=0.5, linestyles='--')
    
    # Add optimal points - where contours first touch the constraint regions
    # For our elliptical case, this happens at different points for L1 and L2
    l1_solution = np.array([0, 1])  # Example solution for L1
    l2_solution = np.array([0.2, 0.9])  # Example solution for L2
    
    # Highlight vertices of L1 norm
    plt.plot([0, 1, 0, -1, 0], [1, 0, -1, 0, 1], 'ro', markersize=8)
    
    # Add labels
    plt.annotate('L1 Solution\n(Sparse)', xy=(0, 1), xytext=(0.2, 1.2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=12, ha='center')
    
    plt.annotate('L2 Solution\n(Non-sparse)', xy=(0.2, 0.9), xytext=(0.6, 1.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=12, ha='center')
    
    # Make it look like a proper coordinate system
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.title('L1 vs L2 Regularization: Geometric Interpretation', fontsize=14)
    plt.xlabel(r'$\beta_1$', fontsize=12)
    plt.ylabel(r'$\beta_2$', fontsize=12)
    plt.legend(['L1 Norm = 1 (Lasso)', 'L2 Norm = 1 (Ridge)', 'Loss Function Contours'], 
               fontsize=12, loc='lower right')
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "regularization_geometry.png"), dpi=300, bbox_inches='tight')
    plt.close()

plot_constraint_regions()

# Visualization 5: Comparative bar chart of original vs regularized coefficients
def plot_coefficient_comparison():
    # Choose a specific alpha for demonstration
    alpha_idx = 10  # Medium regularization strength
    alpha_value = alphas[alpha_idx]
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(w_original))
    bar_width = 0.25
    
    # Sort indices by absolute coefficient value for better visualization
    sorted_idx = np.argsort(np.abs(w_original))[::-1]
    
    # Plot the bars
    bars1 = ax.bar(x - bar_width, w_original[sorted_idx], bar_width, label='Original', color='skyblue')
    bars2 = ax.bar(x, ridge_coefs[alpha_idx, sorted_idx], bar_width, label=f'Ridge (α={alpha_value:.3f})', color='orangered')
    bars3 = ax.bar(x + bar_width, lasso_coefs[alpha_idx, sorted_idx], bar_width, label=f'Lasso (α={alpha_value:.3f})', color='green')
    
    # Add some text for labels, title and axes ticks
    ax.set_xlabel('Features (sorted by original coefficient magnitude)', fontsize=12)
    ax.set_ylabel('Coefficient Value', fontsize=12)
    ax.set_title('Comparison of Original vs. Regularized Coefficients', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([feature_names[i] for i in sorted_idx], rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    
    # Add value annotations
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height < 0:
                va = 'top'
                y_pos = height - 0.2
            else:
                va = 'bottom'
                y_pos = height + 0.1
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{height:.2f}', ha='center', va=va, rotation=90, fontsize=9)
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "coefficient_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

plot_coefficient_comparison()

# Create the final comparison table for part 2 of the question
print("\nComparison Table for Different Coefficient Magnitudes:")
comparison_table = pd.DataFrame({
    'Original magnitude': ['Very large (e.g., 4.5)', 'Medium (e.g., 1.4)', 'Very small (e.g., 0.02)'],
    'Effect with Ridge (L2)': ['Shrunk proportionally', 'Shrunk proportionally', 'Shrunk proportionally, never zero'],
    'Effect with Lasso (L1)': ['Shrunk less than smaller coeffs, rarely zeroed', 
                              'May be zeroed with sufficient regularization', 
                              'Quickly shrunk to exactly zero']
})

print(comparison_table)

print(f"\nVisualizations saved to: {images_dir}") 