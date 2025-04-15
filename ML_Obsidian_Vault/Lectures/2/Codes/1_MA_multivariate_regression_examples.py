import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

print("\n=== MULTIVARIATE REGRESSION EXAMPLES: STEP-BY-STEP SOLUTIONS ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images", "Multivariate_Regression")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Example 1: House Price Prediction
print("Example 1: House Price Prediction")

# Dataset with 3 variables: area (X₁), bedrooms (X₂), and price (Y)
data = np.array([
    [1500, 3, 250000],
    [2000, 4, 300000],
    [1200, 2, 200000],
    [1800, 3, 275000],
    [2200, 4, 350000],
    [1600, 3, 260000],
    [1400, 2, 215000],
    [2300, 5, 380000],
    [1900, 3, 290000],
    [2100, 4, 330000]
])

print("Dataset with 3 variables: house area in sq.ft. (X₁), number of bedrooms (X₂), and price in $ (Y)")
print("\n| House | Area (X₁) | Bedrooms (X₂) | Price (Y) |")
print("|-------|-----------|--------------|-----------|")
for i, row in enumerate(data):
    print(f"| {i+1:<5} | {row[0]:<9} | {row[1]:<12} | {row[2]:<9} |")

# Step 1: Prepare data for regression
print("\nStep 1: Prepare data for regression")
X = data[:, :2]  # Features: area, bedrooms
y = data[:, 2]   # Target: price

n = X.shape[0]  # Number of observations
p = X.shape[1]  # Number of predictors

print(f"\nFeatures (X):")
print("| House | Area (X₁) | Bedrooms (X₂) |")
print("|-------|-----------|--------------|")
for i, row in enumerate(X):
    print(f"| {i+1:<5} | {row[0]:<9} | {row[1]:<12} |")

print(f"\nTarget (y):")
print("| House | Price (Y) |")
print("|-------|-----------|")
for i, val in enumerate(y):
    print(f"| {i+1:<5} | {val:<9} |")

# Step 2: Calculate the regression coefficient using the normal equation
print("\nStep 2: Calculate the regression coefficient using the normal equation")
print("\nThe multivariate linear regression model is: Y = β₀ + β₁X₁ + β₂X₂ + ε")
print("We can write this as: Y = Xβ + ε, where X includes a column of ones for the intercept")

# Add a column of ones to X for the intercept
X_with_intercept = np.c_[np.ones(n), X]

print("\nDesign matrix (X with intercept column):")
print("| House | Intercept | Area (X₁) | Bedrooms (X₂) |")
print("|-------|-----------|-----------|--------------|")
for i, row in enumerate(X_with_intercept):
    print(f"| {i+1:<5} | {row[0]:<9} | {row[1]:<9} | {row[2]:<12} |")

# Calculate β using the normal equation: β = (X^T X)^(-1) X^T y
X_transpose = X_with_intercept.T
X_transpose_X = X_transpose.dot(X_with_intercept)
X_transpose_X_inv = np.linalg.inv(X_transpose_X)
X_transpose_y = X_transpose.dot(y)
beta = X_transpose_X_inv.dot(X_transpose_y)

print("\nCalculating β using the normal equation: β = (X^T X)^(-1) X^T y")
print("\nStep 2.1: Calculate X^T X")
print("\nX^T X = ")
for row in X_transpose_X:
    print("[", end=" ")
    for val in row:
        print(f"{val:<12.2f}", end=" ")
    print("]")

print("\nStep 2.2: Calculate (X^T X)^(-1)")
print("\n(X^T X)^(-1) = ")
for row in X_transpose_X_inv:
    print("[", end=" ")
    for val in row:
        print(f"{val:<12.6f}", end=" ")
    print("]")

print("\nStep 2.3: Calculate X^T y")
print("\nX^T y = ")
for val in X_transpose_y:
    print(f"[{val:<12.2f}]")

print("\nStep 2.4: Calculate β = (X^T X)^(-1) X^T y")
print("\nβ = ")
for i, val in enumerate(beta):
    if i == 0:
        print(f"β₀ (Intercept) = {val:.6f}")
    else:
        print(f"β{i} = {val:.6f}")

# Step 3: Calculate fitted values and residuals
print("\nStep 3: Calculate fitted values and residuals")
y_pred = X_with_intercept.dot(beta)
residuals = y - y_pred

print("\n| House | Actual (Y) | Predicted (Ŷ) | Residual (Y - Ŷ) |")
print("|-------|------------|---------------|-----------------|")
for i in range(n):
    print(f"| {i+1:<5} | {y[i]:<10} | {y_pred[i]:<13.2f} | {residuals[i]:<15.2f} |")

# Step 4: Evaluate the model
print("\nStep 4: Evaluate the model")

# Calculate the Total Sum of Squares (TSS)
y_mean = np.mean(y)
TSS = np.sum((y - y_mean) ** 2)

# Calculate the Regression Sum of Squares (RSS)
RSS = np.sum((y_pred - y_mean) ** 2)

# Calculate the Error Sum of Squares (ESS)
ESS = np.sum(residuals ** 2)

# Calculate R-squared
r_squared = RSS / TSS

# Calculate adjusted R-squared
adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

# Calculate the standard error of the estimate (SEE)
SEE = np.sqrt(ESS / (n - p - 1))

print(f"\nTotal Sum of Squares (TSS) = {TSS:.2f}")
print(f"Regression Sum of Squares (RSS) = {RSS:.2f}")
print(f"Error Sum of Squares (ESS) = {ESS:.2f}")
print(f"Coefficient of determination (R²) = {r_squared:.4f}")
print(f"Adjusted R² = {adjusted_r_squared:.4f}")
print(f"Standard Error of the Estimate = {SEE:.2f}")

# Step 5: Interpret the model
print("\nStep 5: Interpret the model")
print(f"\nThe regression equation is: Price = {beta[0]:.2f} + {beta[1]:.2f} × Area + {beta[2]:.2f} × Bedrooms")
print(f"\nInterpretation:")
print(f"- β₀ = {beta[0]:.2f}: The expected price of a house with zero area and zero bedrooms (not meaningful in this context)")
print(f"- β₁ = {beta[1]:.2f}: For each additional square foot of area, the house price increases by ${beta[1]:.2f} on average, holding the number of bedrooms constant")
print(f"- β₂ = {beta[2]:.2f}: For each additional bedroom, the house price increases by ${beta[2]:.2f} on average, holding the area constant")
print(f"- R² = {r_squared:.4f}: Approximately {r_squared*100:.1f}% of the variation in house prices can be explained by the area and number of bedrooms")

# Visualization: 3D scatter plot with regression plane
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the actual data points
ax.scatter(X[:, 0], X[:, 1], y, c='blue', s=100, alpha=0.7, label='Actual Data')

# Create a meshgrid for the regression plane
x1_range = np.linspace(min(X[:, 0]) - 100, max(X[:, 0]) + 100, 20)
x2_range = np.linspace(min(X[:, 1]) - 0.5, max(X[:, 1]) + 0.5, 20)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
X_grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]
X_grid_with_intercept = np.c_[np.ones(X_grid.shape[0]), X_grid]
y_grid = X_grid_with_intercept.dot(beta).reshape(x1_grid.shape)

# Plot the regression plane
surf = ax.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.5, cmap='viridis', linewidth=0)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Predicted Price ($)')

# Add vertical lines from points to the plane
for i in range(n):
    ax.plot([X[i, 0], X[i, 0]], [X[i, 1], X[i, 1]], [y[i], y_pred[i]], 'r-', alpha=0.5)

ax.set_xlabel('House Area (sq.ft.)')
ax.set_ylabel('Number of Bedrooms')
ax.set_zlabel('House Price ($)')
ax.set_title('3D Visualization of Multivariate Regression: House Price Prediction')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'house_price_regression_3d.png'), dpi=100, bbox_inches='tight')
plt.close()

# Visualization: Predicted vs Actual Values
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, c='blue', alpha=0.7, s=100)
plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', linewidth=2)  # Perfect prediction line

for i in range(n):
    plt.annotate(f'House {i+1}', (y[i], y_pred[i]), fontsize=9, 
                 xytext=(5, 5), textcoords='offset points')

plt.xlabel('Actual House Price ($)')
plt.ylabel('Predicted House Price ($)')
plt.title('Actual vs Predicted House Prices')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'house_price_actual_vs_predicted.png'), dpi=100, bbox_inches='tight')
plt.close()

# Visualization: Residual Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, c='green', alpha=0.7, s=100)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)

for i in range(n):
    plt.annotate(f'House {i+1}', (y_pred[i], residuals[i]), fontsize=9, 
                 xytext=(5, 5), textcoords='offset points')

plt.xlabel('Predicted House Price ($)')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'house_price_residual_plot.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Student Performance Prediction
print("\n\nExample 2: Student Performance Prediction")

# Dataset with 4 variables: study hours (X₁), attendance (X₂), previous GPA (X₃), and final score (Y)
student_data = np.array([
    [20, 85, 3.2, 78],
    [15, 90, 3.5, 82],
    [25, 95, 3.8, 90],
    [10, 70, 2.9, 65],
    [18, 80, 3.0, 75],
    [22, 92, 3.7, 88],
    [12, 75, 3.1, 70],
    [30, 98, 4.0, 95],
    [17, 82, 3.4, 80],
    [23, 88, 3.6, 85],
    [14, 78, 3.3, 72],
    [19, 86, 3.5, 81]
])

print("Dataset with 4 variables: weekly study hours (X₁), attendance percentage (X₂), previous GPA (X₃), and final score (Y)")
print("\n| Student | Study Hours (X₁) | Attendance % (X₂) | Prev GPA (X₃) | Final Score (Y) |")
print("|---------|-----------------|------------------|--------------|----------------|")
for i, row in enumerate(student_data):
    print(f"| {i+1:<7} | {row[0]:<15} | {row[1]:<16} | {row[2]:<12} | {row[3]:<14} |")

# Step 1: Prepare data for regression
print("\nStep 1: Prepare data for regression")
X_student = student_data[:, :3]  # Features: study hours, attendance, previous GPA
y_student = student_data[:, 3]   # Target: final score

n_student = X_student.shape[0]  # Number of observations
p_student = X_student.shape[1]  # Number of predictors

# Step 2: Split data into training and testing sets
print("\nStep 2: Split data into training and testing sets (75% train, 25% test)")
X_train, X_test, y_train, y_test = train_test_split(X_student, y_student, test_size=0.25, random_state=42)

print(f"\nTraining set: {X_train.shape[0]} students")
print(f"Testing set: {X_test.shape[0]} students")

# Step 3: Standardize features (optional but recommended for multiple regression)
print("\nStep 3: Standardize features")
X_train_mean = np.mean(X_train, axis=0)
X_train_std = np.std(X_train, axis=0)

X_train_scaled = (X_train - X_train_mean) / X_train_std
X_test_scaled = (X_test - X_train_mean) / X_train_std

print("\nFeature means (Training set):")
print(f"Study Hours mean: {X_train_mean[0]:.2f}")
print(f"Attendance mean: {X_train_mean[1]:.2f}%")
print(f"Previous GPA mean: {X_train_mean[2]:.2f}")

print("\nFeature standard deviations (Training set):")
print(f"Study Hours std: {X_train_std[0]:.2f}")
print(f"Attendance std: {X_train_std[1]:.2f}%")
print(f"Previous GPA std: {X_train_std[2]:.2f}")

# Step 4: Fit the multiple linear regression model
print("\nStep 4: Fit the multiple linear regression model")
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Extract coefficients
beta_0 = model.intercept_
beta_1, beta_2, beta_3 = model.coef_

print(f"\nRegression coefficients:")
print(f"β₀ (Intercept): {beta_0:.4f}")
print(f"β₁ (Study Hours): {beta_1:.4f}")
print(f"β₂ (Attendance): {beta_2:.4f}")
print(f"β₃ (Previous GPA): {beta_3:.4f}")

# Step 5: Model Evaluation on Training Data
print("\nStep 5: Model Evaluation on Training Data")
y_train_pred = model.predict(X_train_scaled)

train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_train_pred)

print(f"Training Mean Squared Error (MSE): {train_mse:.4f}")
print(f"Training Root Mean Squared Error (RMSE): {train_rmse:.4f}")
print(f"Training R²: {train_r2:.4f}")

# Step 6: Model Evaluation on Testing Data
print("\nStep 6: Model Evaluation on Testing Data")
y_test_pred = model.predict(X_test_scaled)

test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Testing Mean Squared Error (MSE): {test_mse:.4f}")
print(f"Testing Root Mean Squared Error (RMSE): {test_rmse:.4f}")
print(f"Testing R²: {test_r2:.4f}")

# Calculate the standardized coefficients
std_coef = model.coef_ * X_train_std

print("\nStep 7: Interpret Standardized Coefficients")
print(f"\nStandardized coefficients:")
print(f"β₁* (Study Hours): {std_coef[0]:.4f}")
print(f"β₂* (Attendance): {std_coef[1]:.4f}")
print(f"β₃* (Previous GPA): {std_coef[2]:.4f}")

print("\nInterpretation:")
features = ["Study Hours", "Attendance", "Previous GPA"]
std_coef_abs = np.abs(std_coef)
importance_order = np.argsort(std_coef_abs)[::-1]  # Sort in descending order

print(f"Feature importance ranking (based on absolute standardized coefficients):")
for rank, idx in enumerate(importance_order):
    print(f"{rank+1}. {features[idx]}: {std_coef_abs[idx]:.4f}")

most_important_idx = importance_order[0]
print(f"\nThe most important predictor is {features[most_important_idx]}.")
print(f"When {features[most_important_idx]} increases by one standard deviation ({X_train_std[most_important_idx]:.2f}), ", end="")
print(f"the final score is expected to change by {std_coef[most_important_idx]:.2f} points, holding other variables constant.")

# Step 8: Equation for Predicting Student Final Score
print("\nStep 8: Equation for Predicting Student Final Score")

# Unstandardized coefficients (convert back from standardized model)
unstd_intercept = beta_0 - np.sum(model.coef_ * X_train_mean / X_train_std)
unstd_coef = model.coef_ / X_train_std

print(f"\nOriginal scale equation:")
print(f"Final Score = {unstd_intercept:.4f} + {unstd_coef[0]:.4f} × Study Hours + {unstd_coef[1]:.4f} × Attendance + {unstd_coef[2]:.4f} × Previous GPA")

# Visualization: Feature Importance
plt.figure(figsize=(10, 6))
plt.bar(features, std_coef_abs, color=['blue', 'green', 'red'])
plt.xlabel('Features')
plt.ylabel('Absolute Standardized Coefficient')
plt.title('Feature Importance in Predicting Student Performance')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'student_performance_feature_importance.png'), dpi=100, bbox_inches='tight')
plt.close()

# Visualization: Actual vs Predicted Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_train_pred, label='Training Data', alpha=0.7, color='blue')
plt.scatter(y_test, y_test_pred, label='Testing Data', alpha=0.7, color='red')
plt.plot([min(y_student), max(y_student)], [min(y_student), max(y_student)], 'k--', label='Perfect Prediction')

for i, (actual, pred) in enumerate(zip(y_test, y_test_pred)):
    plt.annotate(f'Test {i+1}', (actual, pred), fontsize=9, 
                 xytext=(5, 5), textcoords='offset points')

plt.xlabel('Actual Final Score')
plt.ylabel('Predicted Final Score')
plt.title('Actual vs Predicted Student Performance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'student_performance_actual_vs_predicted.png'), dpi=100, bbox_inches='tight')
plt.close()

# Visualization: Residual Plots for Each Feature
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
feature_names = ['Study Hours', 'Attendance', 'Previous GPA']

residuals_train = y_train - y_train_pred
residuals_test = y_test - y_test_pred

for i in range(3):
    axes[i].scatter(X_train[:, i], residuals_train, label='Training', alpha=0.7, color='blue')
    axes[i].scatter(X_test[:, i], residuals_test, label='Testing', alpha=0.7, color='red')
    axes[i].axhline(y=0, color='k', linestyle='--')
    axes[i].set_xlabel(feature_names[i])
    axes[i].set_ylabel('Residuals')
    axes[i].set_title(f'Residuals vs {feature_names[i]}')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'student_performance_residual_plots.png'), dpi=100, bbox_inches='tight')
plt.close()

# Visualization: Partial Regression Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

X_train_with_intercept = np.c_[np.ones(X_train.shape[0]), X_train_scaled]

for i in range(3):
    # Create X matrices excluding the i-th variable
    X_train_exclude_i = np.delete(X_train_with_intercept, i+1, axis=1)
    
    # Fit model without the i-th variable to predict Y
    model_exclude_i_for_y = LinearRegression(fit_intercept=False)
    model_exclude_i_for_y.fit(X_train_exclude_i, y_train)
    y_train_pred_exclude_i = model_exclude_i_for_y.predict(X_train_exclude_i)
    
    # Fit model without the i-th variable to predict the i-th variable
    model_exclude_i_for_x = LinearRegression(fit_intercept=False)
    model_exclude_i_for_x.fit(X_train_exclude_i, X_train_scaled[:, i])
    x_train_i_pred = model_exclude_i_for_x.predict(X_train_exclude_i)
    
    # Calculate residuals for both models
    y_train_resid = y_train - y_train_pred_exclude_i
    x_train_i_resid = X_train_scaled[:, i] - x_train_i_pred
    
    # Plot the partial regression plot
    axes[i].scatter(x_train_i_resid, y_train_resid, alpha=0.7, color='purple')
    
    # Add regression line
    slope = model.coef_[i]
    axes[i].plot(x_train_i_resid, slope * x_train_i_resid, 'r-', linewidth=2)
    
    axes[i].set_xlabel(f'{feature_names[i]} (residualized)')
    axes[i].set_ylabel('Final Score (residualized)')
    axes[i].set_title(f'Partial Regression Plot for {feature_names[i]}')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'student_performance_partial_regression_plots.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll multivariate regression example images created successfully.") 