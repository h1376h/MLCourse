import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_4_Quiz_16")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Function to save figures
def save_figure(fig, filename):
    file_path = os.path.join(save_dir, filename)
    fig.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {file_path}")
    return file_path

saved_figures = []

print("# Question 16: Key Concepts in Linear Regression\n")

# 1. Multicollinearity demonstration
print("## 1. Addressing Multicollinearity\n")

# Generate data with multicollinearity
np.random.seed(42)
n_samples = 100
x1 = np.random.normal(0, 1, n_samples)
x2 = 0.9 * x1 + 0.1 * np.random.normal(0, 1, n_samples)  # x2 is highly correlated with x1
x3 = np.random.normal(0, 1, n_samples)  # Independent variable
y = 2 * x1 + 0.5 * x3 + np.random.normal(0, 1, n_samples)

# Create a DataFrame for easy analysis
data = pd.DataFrame({
    'x1': x1,
    'x2': x2,
    'x3': x3,
    'y': y
})

# Display correlation matrix
correlation_matrix = data.corr()
print("Correlation Matrix:")
print(correlation_matrix)
print()

# Visualize correlation matrix
fig1, ax1 = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax1)
plt.title('Correlation Matrix: Demonstrating Multicollinearity between x1 and x2')
saved_figures.append(save_figure(fig1, "1_multicollinearity_correlation.png"))

# Different approaches to address multicollinearity
print("Different approaches to address multicollinearity:\n")

# 1. Original model with multicollinearity
X_multi = data[['x1', 'x2', 'x3']]
model_multi = LinearRegression().fit(X_multi, y)
print("1. Model with multicollinearity:")
print(f"   Coefficients: {model_multi.coef_}")
print(f"   R² score: {model_multi.score(X_multi, y):.4f}")

# 2. Remove one of the correlated variables
X_remove = data[['x1', 'x3']]  # Removed x2
model_remove = LinearRegression().fit(X_remove, y)
print("\n2. Model after removing one correlated variable (x2):")
print(f"   Coefficients: {model_remove.coef_}")
print(f"   R² score: {model_remove.score(X_remove, y):.4f}")

# 3. Combine correlated variables
data['x1_x2_mean'] = (data['x1'] + data['x2']) / 2
X_combine = data[['x1_x2_mean', 'x3']]
model_combine = LinearRegression().fit(X_combine, y)
print("\n3. Model after combining correlated variables (x1 and x2):")
print(f"   Coefficients: {model_combine.coef_}")
print(f"   R² score: {model_combine.score(X_combine, y):.4f}")

# 4. Use regularization (Ridge regression)
X_reg = data[['x1', 'x2', 'x3']]
model_reg = Ridge(alpha=1.0).fit(X_reg, y)
print("\n4. Model with regularization (Ridge):")
print(f"   Coefficients: {model_reg.coef_}")
print(f"   R² score: {model_reg.score(X_reg, y):.4f}")

# 5. Square all features (NOT a valid way to address multicollinearity)
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_squared = poly.fit_transform(data[['x1', 'x2', 'x3']])
model_squared = LinearRegression().fit(X_squared, y)
print("\n5. Model with squared features (INVALID approach):")
print(f"   R² score: {model_squared.score(X_squared, y):.4f}")
print("   Note: Squaring features typically worsens multicollinearity!")

print("\nResult: Option D (Square all the input features) is NOT a valid way to address multicollinearity.")

# 2. Dummy variables demonstration
print("\n## 2. Dummy Variables for Categorical Predictors\n")

# Create a synthetic dataset with a categorical variable with 4 levels
np.random.seed(42)
n_samples = 200
categories = ['A', 'B', 'C', 'D']
cat_var = np.random.choice(categories, size=n_samples)
x_numeric = np.random.normal(0, 1, n_samples)

# Different means for different categories
effects = {'A': 2, 'B': 5, 'C': 8, 'D': 11}
y = x_numeric + np.array([effects[cat] for cat in cat_var]) + np.random.normal(0, 1, n_samples)

# Create DataFrame
cat_data = pd.DataFrame({
    'category': cat_var,
    'x': x_numeric,
    'y': y
})

# Create dummy variables
cat_dummies = pd.get_dummies(cat_data['category'], drop_first=True)  # Drop first to avoid multicollinearity
print("Dummy variables created (with drop_first=True):")
print(cat_dummies.head())
print()

# Number of dummy variables
num_dummies = cat_dummies.shape[1]
print(f"Number of dummy variables created: {num_dummies}")
print(f"Number of levels in the categorical variable: {len(categories)}")
print("Typically number of dummy variables = Number of levels - 1")

# Visualize the data by category
fig2, ax2 = plt.subplots(figsize=(10, 6))
for cat in categories:
    subset = cat_data[cat_data['category'] == cat]
    ax2.scatter(subset['x'], subset['y'], label=f'Category {cat}', alpha=0.7)

ax2.set_xlabel('X Variable')
ax2.set_ylabel('Y Variable')
ax2.set_title('Data by Category: Each Level Needs a Separate Dummy Variable')
ax2.legend()
saved_figures.append(save_figure(fig2, "2_dummy_variables.png"))

print("\nResult: For a categorical predictor with 4 levels, we typically use 3 dummy variables (option C).")

# 3. Interaction terms demonstration
print("\n## 3. Interaction Terms in Regression\n")

# Generate data with interaction effects
np.random.seed(42)
n_samples = 100
x1 = np.random.uniform(-2, 2, n_samples)
x2 = np.random.uniform(-2, 2, n_samples)
# Create interaction effect: the effect of x1 depends on the value of x2
y = 2 * x1 + 3 * x2 + 4 * x1 * x2 + np.random.normal(0, 1, n_samples)

# Create a mesh for plotting
x1_mesh, x2_mesh = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
y_mesh = 2 * x1_mesh + 3 * x2_mesh + 4 * x1_mesh * x2_mesh

# Create DataFrame
interaction_data = pd.DataFrame({
    'x1': x1,
    'x2': x2,
    'y': y
})

# Fit models with and without interaction
X = interaction_data[['x1', 'x2']]
X_with_interaction = interaction_data[['x1', 'x2']]
X_with_interaction['x1_x2'] = interaction_data['x1'] * interaction_data['x2']

model_no_interaction = LinearRegression().fit(X, y)
model_with_interaction = LinearRegression().fit(X_with_interaction, y)

print("Model without interaction term:")
print(f"Coefficients: {model_no_interaction.coef_}")
print(f"R² score: {model_no_interaction.score(X, y):.4f}")

print("\nModel with interaction term (x1 × x2):")
print(f"Coefficients: {model_with_interaction.coef_}")
print(f"R² score: {model_with_interaction.score(X_with_interaction, y):.4f}")

# Visualizing the interaction effect
fig3 = plt.figure(figsize=(15, 10))
gs = fig3.add_gridspec(2, 2)

# 3D surface plot
ax3a = fig3.add_subplot(gs[0, :], projection='3d')
surf = ax3a.plot_surface(x1_mesh, x2_mesh, y_mesh, cmap='viridis', alpha=0.8)
ax3a.scatter(x1, x2, y, color='red', alpha=0.5)
ax3a.set_xlabel('x1')
ax3a.set_ylabel('x2')
ax3a.set_zlabel('y')
ax3a.set_title('3D Surface with Interaction Effect (Nonplanar Surface)')

# Effect of x1 for different values of x2
ax3b = fig3.add_subplot(gs[1, 0])
for x2_val in [-1.5, 0, 1.5]:
    x1_line = np.linspace(-2, 2, 100)
    y_line = 2 * x1_line + 3 * x2_val + 4 * x1_line * x2_val
    ax3b.plot(x1_line, y_line, label=f'x2 = {x2_val}')

ax3b.set_xlabel('x1')
ax3b.set_ylabel('y')
ax3b.set_title('Effect of x1 Changes Based on Value of x2')
ax3b.legend()

# Contour plot
ax3c = fig3.add_subplot(gs[1, 1])
contour = ax3c.contourf(x1_mesh, x2_mesh, y_mesh, cmap='viridis', levels=20)
ax3c.set_xlabel('x1')
ax3c.set_ylabel('x2')
ax3c.set_title('Contour Plot: Non-parallel Lines Show Interaction')
plt.colorbar(contour, ax=ax3c)

plt.tight_layout()
saved_figures.append(save_figure(fig3, "3_interaction_terms.png"))

print("\nResult: The interaction term x1 × x2 captures how the effect of x1 changes based on the value of x2 (option B).")

# 4 & 6. Polynomial regression demonstration
print("\n## 4 & 6. Polynomial Regression and Effects of Increasing Degree\n")

# Generate non-linear data
np.random.seed(42)
n_samples = 50
X_orig = np.sort(np.random.uniform(0, 1, n_samples))
y_true = np.sin(2 * np.pi * X_orig)
y = y_true + np.random.normal(0, 0.1, n_samples)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_orig.reshape(-1, 1), y, test_size=0.3, random_state=42)

# Fit polynomial models of different degrees
max_degree = 15
train_errors = []
test_errors = []
models = []

for degree in range(1, max_degree + 1):
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    
    # Fit model
    model = LinearRegression()
    model.fit(X_poly_train, y_train)
    models.append(model)
    
    # Calculate errors
    train_pred = model.predict(X_poly_train)
    test_pred = model.predict(X_poly_test)
    
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    
    train_errors.append(train_mse)
    test_errors.append(test_mse)
    
    print(f"Degree {degree}:")
    print(f"  Train MSE: {train_mse:.6f}")
    print(f"  Test MSE: {test_mse:.6f}")

# Find best degree based on test error
best_degree = np.argmin(test_errors) + 1
print(f"\nBest polynomial degree based on test error: {best_degree}")

# Plot the models and errors
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(15, 6))

# Plot training and test error vs. degree
ax4a.plot(range(1, max_degree + 1), train_errors, 'o-', label='Training Error')
ax4a.plot(range(1, max_degree + 1), test_errors, 'o-', label='Test Error')
ax4a.set_xlabel('Polynomial Degree')
ax4a.set_ylabel('Mean Squared Error')
ax4a.set_title('Training and Test Error vs. Polynomial Degree')
ax4a.legend()
ax4a.set_yscale('log')
ax4a.axvline(x=best_degree, color='red', linestyle='--', alpha=0.3)
ax4a.grid(True)

# Plot selected models
X_line = np.linspace(0, 1, 100).reshape(-1, 1)
y_line_true = np.sin(2 * np.pi * X_line.ravel())

degrees_to_show = [1, 3, best_degree, max_degree]
for degree in degrees_to_show:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_line_poly = poly.fit_transform(X_line)
    y_line_pred = models[degree-1].predict(X_line_poly)
    ax4b.plot(X_line, y_line_pred, label=f'Degree {degree}')

ax4b.scatter(X_train, y_train, color='blue', alpha=0.5, label='Training data')
ax4b.scatter(X_test, y_test, color='green', alpha=0.5, label='Test data')
ax4b.plot(X_line, y_line_true, 'r--', label='True function')
ax4b.set_xlabel('X')
ax4b.set_ylabel('y')
ax4b.set_title('Polynomial Regression Models of Different Degrees')
ax4b.legend()

plt.tight_layout()
saved_figures.append(save_figure(fig4, "4_polynomial_regression.png"))

print("\nResult for Question 4: Polynomial regression can capture nonlinear relationships in the data (option C).")
print("Result for Question 6: As the degree of polynomial regression increases, training error always decreases (option A).")

# 5. Radial Basis Functions
print("\n## 5. Radial Basis Functions in Regression\n")

def rbf_kernel(x, center, width=1.0):
    """Radial basis function (Gaussian kernel)"""
    return np.exp(-((x - center) ** 2) / (2 * width ** 2))

# Generate some non-linear data
np.random.seed(42)
n_samples = 50
X_rbf = np.sort(np.random.uniform(-5, 5, n_samples)).reshape(-1, 1)
y_rbf = np.sin(X_rbf.ravel()) + 0.1 * np.random.normal(0, 1, n_samples)

# Create RBF features
n_centers = 10
centers = np.linspace(-5, 5, n_centers)
width = 0.5

X_rbf_features = np.zeros((n_samples, n_centers))
for i, center in enumerate(centers):
    X_rbf_features[:, i] = rbf_kernel(X_rbf.ravel(), center, width)

# Train linear model on RBF features
rbf_model = LinearRegression().fit(X_rbf_features, y_rbf)

# Create a smooth line for prediction
X_line = np.linspace(-6, 6, 500).reshape(-1, 1)
X_line_rbf = np.zeros((500, n_centers))
for i, center in enumerate(centers):
    X_line_rbf[:, i] = rbf_kernel(X_line.ravel(), center, width)

y_line_rbf = rbf_model.predict(X_line_rbf)

# Visualize RBF regression
fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(10, 12))

# Plot the basis functions
for i, center in enumerate(centers):
    y_basis = rbf_kernel(X_line.ravel(), center, width)
    weight = rbf_model.coef_[i]
    ax5a.plot(X_line, y_basis, label=f'RBF {i+1}' if i < 3 or i > n_centers-3 else "")
    ax5a.axvline(x=center, color='gray', linestyle='--', alpha=0.3)

ax5a.set_xlabel('X')
ax5a.set_ylabel('RBF Value')
ax5a.set_title('Radial Basis Functions: Each is Centered at a Training Point')
ax5a.legend()

# Plot the data and RBF fit
ax5b.scatter(X_rbf, y_rbf, color='blue', label='Data points')
ax5b.plot(X_line, y_line_rbf, color='red', label='RBF Regression')
ax5b.plot(X_line, np.sin(X_line.ravel()), 'g--', label='True function')

for center in centers:
    ax5b.axvline(x=center, color='gray', linestyle='--', alpha=0.2)

ax5b.set_xlabel('X')
ax5b.set_ylabel('y')
ax5b.set_title('RBF Regression: Capturing Similarities Based on Distance')
ax5b.legend()

plt.tight_layout()
saved_figures.append(save_figure(fig5, "5_radial_basis_functions.png"))

print("Radial Basis Functions (RBFs) are used to capture similarities between data points based on their distance.")
print("Each RBF is centered at a specific point (often a training example or key location).")
print("Points closer to the center of an RBF have higher feature values for that RBF.")
print("This allows the model to capture local patterns and nonlinear relationships.")

print("\nResult: The primary purpose of using radial basis functions is to capture similarities between data points based on their distance (option C).")

# 7. Normal Equations demonstration
print("\n## 7. Normal Equations in Linear Regression\n")

# Generate well-conditioned data
np.random.seed(42)
n_samples = 100
n_features = 3
X_well = np.random.normal(0, 1, (n_samples, n_features))
true_coef = np.array([1.5, -0.5, 2.0])
y_well = X_well @ true_coef + np.random.normal(0, 0.5, n_samples)

# Calculate the normal equation solution
XT_X = X_well.T @ X_well
XT_y = X_well.T @ y_well
w_normal = np.linalg.inv(XT_X) @ XT_y

# Compare with sklearn
model_well = LinearRegression().fit(X_well, y_well)

print("Normal equations solution:")
print(f"Coefficients: {w_normal}")
print("\nScikit-learn solution:")
print(f"Coefficients: {model_well.coef_}")

# Generate poorly-conditioned data (multicollinearity)
X_poor = np.zeros((n_samples, n_features))
X_poor[:, 0] = np.random.normal(0, 1, n_samples)
X_poor[:, 1] = X_poor[:, 0] + 1e-10 * np.random.normal(0, 1, n_samples)  # Nearly identical to first column
X_poor[:, 2] = np.random.normal(0, 1, n_samples)
y_poor = X_poor @ true_coef + np.random.normal(0, 0.5, n_samples)

# Calculate condition number
cond_well = np.linalg.cond(X_well.T @ X_well)
cond_poor = np.linalg.cond(X_poor.T @ X_poor)

print(f"\nCondition number for well-conditioned data: {cond_well:.2f}")
print(f"Condition number for poorly-conditioned data: {cond_poor:.2e}")

# Visualize eigenvalues
XT_X_well = X_well.T @ X_well
XT_X_poor = X_poor.T @ X_poor

evals_well = np.linalg.eigvals(XT_X_well)
evals_poor = np.linalg.eigvals(XT_X_poor)

fig7, ax7 = plt.subplots(1, 2, figsize=(12, 6))

ax7[0].bar(range(n_features), np.sort(evals_well)[::-1], color='blue')
ax7[0].set_title(f'Eigenvalues of X^T X (Well-conditioned)\nCondition Number: {cond_well:.2f}')
ax7[0].set_xlabel('Eigenvalue Index')
ax7[0].set_ylabel('Eigenvalue')
ax7[0].grid(True)

ax7[1].bar(range(n_features), np.sort(evals_poor)[::-1], color='red')
ax7[1].set_title(f'Eigenvalues of X^T X (Poorly-conditioned)\nCondition Number: {cond_poor:.2e}')
ax7[1].set_xlabel('Eigenvalue Index')
ax7[1].set_ylabel('Eigenvalue')
ax7[1].grid(True)

plt.tight_layout()
saved_figures.append(save_figure(fig7, "7_normal_equations.png"))

print("\nResult: The normal equations solution provides the unique global minimum of the cost function only when X^T X is invertible (option A).")

# 8. Curse of dimensionality
print("\n## 8. Curse of Dimensionality\n")

# Function to generate data with varying dimensions and sample sizes
def generate_high_dim_data(n_samples, n_dimensions, relevant_dims=1):
    X = np.random.normal(0, 1, (n_samples, n_dimensions))
    # Only the first 'relevant_dims' features affect the target
    true_coef = np.zeros(n_dimensions)
    true_coef[:relevant_dims] = 1.0
    y = X @ true_coef + np.random.normal(0, 0.5, n_samples)
    return X, y

# Function to evaluate model performance with varying dimensions
def evaluate_curse_of_dimensionality(dims_list, samples_list):
    results = np.zeros((len(dims_list), len(samples_list)))
    
    for i, dims in enumerate(dims_list):
        for j, samples in enumerate(samples_list):
            # Generate data
            X, y = generate_high_dim_data(samples, dims)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Standardize features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Evaluate
            test_score = model.score(X_test, y_test)
            results[i, j] = test_score
    
    return results

# Evaluate with different dimensions and sample sizes
dims_list = [5, 10, 20, 50, 100]
samples_list = [50, 100, 500, 1000, 5000]

print("Evaluating how model performance changes with dimensions and sample size...")
results = evaluate_curse_of_dimensionality(dims_list, samples_list)

# Visualize the curse of dimensionality
fig8, (ax8a, ax8b) = plt.subplots(1, 2, figsize=(15, 6))

# Plot test scores as a function of dimensions for different sample sizes
for j, samples in enumerate(samples_list):
    ax8a.plot(dims_list, results[:, j], 'o-', label=f'n={samples}')

ax8a.set_xscale('log')
ax8a.set_xlabel('Number of Dimensions (log scale)')
ax8a.set_ylabel('Test R² Score')
ax8a.set_title('Effect of Dimensionality on Model Performance\nfor Different Sample Sizes')
ax8a.legend()
ax8a.grid(True)

# Plot required sample size for given performance
sample_size_needed = []
target_r2 = 0.7
for i, dims in enumerate(dims_list):
    for j, samples in enumerate(samples_list):
        if results[i, j] >= target_r2:
            sample_size_needed.append((dims, samples))
            break
    else:
        # If no sample size achieved target, use the maximum with a marker
        sample_size_needed.append((dims, samples_list[-1] * 2))  # Estimate beyond our tests

dims_plot, samples_plot = zip(*sample_size_needed)
ax8b.plot(dims_plot, samples_plot, 'ro-', linewidth=2)
ax8b.set_xscale('log')
ax8b.set_yscale('log')
ax8b.set_xlabel('Number of Dimensions (log scale)')
ax8b.set_ylabel('Required Sample Size (log scale)')
ax8b.set_title(f'Sample Size Needed to Achieve R² ≥ {target_r2}\nGrows Exponentially with Dimensions')
ax8b.grid(True)

plt.tight_layout()
saved_figures.append(save_figure(fig8, "8_curse_of_dimensionality.png"))

print("\nKey findings about the curse of dimensionality:")
print("1. As dimensions increase, model performance decreases for the same sample size")
print("2. To maintain the same level of performance, sample size must grow exponentially")
print("3. High-dimensional spaces become increasingly sparse, making learning more difficult")
print("4. Distance metrics become less meaningful in high dimensions")

print("\nResult: As the number of features increases, the amount of data needed to generalize accurately grows exponentially (option B).")

print("\n# Summary of Results for Question 16:\n")
print("1. NOT a valid way to address multicollinearity: D. Square all the input features")
print("2. Number of dummy variables for a categorical predictor with 4 levels: C. 3")
print("3. What interaction term x1 × x2 captures: B. How the effect of x1 changes based on the value of x2")
print("4. Key advantage of polynomial regression: C. Can capture nonlinear relationships in the data")
print("5. Primary purpose of radial basis functions: C. To capture similarities between data points based on their distance")
print("6. As polynomial degree increases: A. Training error always decreases")
print("7. True statement about normal equations: A. It provides the unique global minimum of the cost function only when X^T X is invertible")
print("8. True statement about curse of dimensionality: B. As the number of features increases, the amount of data needed to generalize accurately grows exponentially")

print("\nFigures saved to:", save_dir)
for i, fig_path in enumerate(saved_figures):
    print(f"{i+1}. {os.path.basename(fig_path)}") 