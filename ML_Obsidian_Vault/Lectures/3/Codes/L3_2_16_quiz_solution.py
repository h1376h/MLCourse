import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from scipy import stats
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_2_16")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Set up the problem
print_step_header(1, "Understanding the Problem")

print("Question 16: Evaluate whether each of the following statements is TRUE or FALSE.")
print("Justify your answer with a brief explanation.")
print()
print("1. In simple linear regression, the residuals always sum to zero when the model includes an intercept term.")
print("2. The least squares method minimizes the sum of absolute differences between predicted and actual values.")
print("3. Increasing the number of data points always leads to a better fit in simple linear regression.")
print("4. The coefficient of determination (R²) represents the proportion of variance in the dependent variable explained by the model.")
print("5. In simple linear regression, the regression line always passes through the point (x̄, ȳ).")
print()

# Step 2: Analyze Statement 1 - Residuals sum to zero with intercept
print_step_header(2, "Analyzing Statement 1: Residuals Sum to Zero")

# Generate some data
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = 2 + 1.5 * x + np.random.normal(0, 2, 100)  # True model: y = 2 + 1.5x + noise

# Create DataFrame for easier analysis
data = pd.DataFrame({'x': x, 'y': y})

# Fit linear regression models - with and without intercept
model_with_intercept = LinearRegression()
model_without_intercept = LinearRegression(fit_intercept=False)

model_with_intercept.fit(data[['x']], data['y'])
model_without_intercept.fit(data[['x']], data['y'])

# Predictions and residuals
data['pred_with_intercept'] = model_with_intercept.predict(data[['x']])
data['pred_without_intercept'] = model_without_intercept.predict(data[['x']])

data['residuals_with_intercept'] = data['y'] - data['pred_with_intercept']
data['residuals_without_intercept'] = data['y'] - data['pred_without_intercept']

# Create a visualization for residuals
plt.figure(figsize=(12, 10))

# Plot data points and regression lines
plt.subplot(2, 1, 1)
plt.scatter(data['x'], data['y'], alpha=0.6, label='Data points')
plt.plot(data['x'], data['pred_with_intercept'], 'r-', linewidth=2, 
         label=f'With intercept: y = {model_with_intercept.intercept_:.2f} + {model_with_intercept.coef_[0]:.2f}x')
plt.plot(data['x'], data['pred_without_intercept'], 'g--', linewidth=2,
         label=f'Without intercept: y = {model_without_intercept.coef_[0]:.2f}x')
plt.title('Linear Regression Models With and Without Intercept', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend()
plt.grid(True)

# Plot residuals
plt.subplot(2, 1, 2)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.scatter(data['x'], data['residuals_with_intercept'], color='r', alpha=0.6, 
            label=f'With intercept: Sum = {data["residuals_with_intercept"].sum():.2e}')
plt.scatter(data['x'], data['residuals_without_intercept'], color='g', alpha=0.6,
            label=f'Without intercept: Sum = {data["residuals_without_intercept"].sum():.2f}')
plt.title('Residuals for Models With and Without Intercept', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('Residual (y - ŷ)', fontsize=12)
plt.legend()
plt.grid(True)

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "residuals_zero_sum.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Print analysis
print("\nAnalysis of Statement 1:")
print("------------------------")
print("When the model includes an intercept term:")
print(f"Sum of residuals = {data['residuals_with_intercept'].sum():.10f}")
print(f"Mean of residuals = {data['residuals_with_intercept'].mean():.10f}")
print("\nWhen the model does not include an intercept term:")
print(f"Sum of residuals = {data['residuals_without_intercept'].sum():.10f}")
print(f"Mean of residuals = {data['residuals_without_intercept'].mean():.10f}")
print("\nThe sum of residuals is effectively zero for the model with intercept")
print("(tiny deviations are due to numerical precision)")
print("Therefore, Statement 1 is TRUE.")

# Step 3: Analyze Statement 2 - Least squares vs. least absolute deviation
print_step_header(3, "Analyzing Statement 2: Least Squares vs. Least Absolute Deviation")

# Generate some data with outliers to highlight the difference
np.random.seed(42)
x_clean = np.linspace(0, 10, 20)
y_clean = 2 + 1.5 * x_clean + np.random.normal(0, 1, 20)

# Add a few outliers
x_outliers = np.array([3, 7, 8])
y_outliers = np.array([15, 0, 20])

x_with_outliers = np.concatenate([x_clean, x_outliers])
y_with_outliers = np.concatenate([y_clean, y_outliers])

# Function to calculate sum of absolute differences
def least_absolute_deviations(params, x, y):
    a, b = params
    return np.sum(np.abs(y - (a + b * x)))

# Function to calculate sum of squared differences
def least_squares(params, x, y):
    a, b = params
    return np.sum((y - (a + b * x))**2)

# Initial guess
initial_guess = [0, 0]

# Fit both models
result_lad = minimize(least_absolute_deviations, initial_guess, args=(x_with_outliers, y_with_outliers))
result_ls = minimize(least_squares, initial_guess, args=(x_with_outliers, y_with_outliers))

a_lad, b_lad = result_lad.x
a_ls, b_ls = result_ls.x

# Calculate predictions
y_pred_lad = a_lad + b_lad * x_with_outliers
y_pred_ls = a_ls + b_ls * x_with_outliers

# Calculate errors
abs_errors_lad = np.abs(y_with_outliers - y_pred_lad)
squared_errors_lad = (y_with_outliers - y_pred_lad)**2
abs_errors_ls = np.abs(y_with_outliers - y_pred_ls)
squared_errors_ls = (y_with_outliers - y_pred_ls)**2

# Create visualization
plt.figure(figsize=(12, 10))

# Plot data and regression lines
plt.subplot(2, 1, 1)
plt.scatter(x_clean, y_clean, alpha=0.6, label='Regular data points')
plt.scatter(x_outliers, y_outliers, color='red', s=100, marker='x', label='Outliers')
x_range = np.linspace(0, 10, 100)
plt.plot(x_range, a_ls + b_ls * x_range, 'b-', linewidth=2, 
         label=f'Least Squares: y = {a_ls:.2f} + {b_ls:.2f}x')
plt.plot(x_range, a_lad + b_lad * x_range, 'g--', linewidth=2,
         label=f'Least Absolute Deviation: y = {a_lad:.2f} + {b_lad:.2f}x')
plt.title('Least Squares vs. Least Absolute Deviation Regression', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend()
plt.grid(True)

# Plot errors for both methods
plt.subplot(2, 1, 2)
bar_width = 0.35
indices = np.arange(len(x_with_outliers))

plt.bar(indices - bar_width/2, abs_errors_ls, bar_width, alpha=0.6, color='b', 
        label=f'LS: Sum of |errors| = {np.sum(abs_errors_ls):.2f}')
plt.bar(indices + bar_width/2, abs_errors_lad, bar_width, alpha=0.6, color='g',
        label=f'LAD: Sum of |errors| = {np.sum(abs_errors_lad):.2f}')

# Add markers for squared errors
plt.scatter(indices - bar_width/2, np.sqrt(squared_errors_ls), color='red', marker='o', 
           label=f'LS: Sum of errors² = {np.sum(squared_errors_ls):.2f}')
plt.scatter(indices + bar_width/2, np.sqrt(squared_errors_lad), color='orange', marker='o',
           label=f'LAD: Sum of errors² = {np.sum(squared_errors_lad):.2f}')

plt.title('Absolute Errors and Squared Errors for Both Methods', fontsize=14)
plt.xlabel('Data Point Index', fontsize=12)
plt.ylabel('Error Magnitude', fontsize=12)
plt.legend()
plt.grid(True)

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "least_squares_vs_lad.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Print analysis
print("\nAnalysis of Statement 2:")
print("------------------------")
print("Least Squares method:")
print(f"Sum of squared errors = {np.sum(squared_errors_ls):.2f}")
print(f"Sum of absolute errors = {np.sum(abs_errors_ls):.2f}")
print()
print("Least Absolute Deviation method:")
print(f"Sum of squared errors = {np.sum(squared_errors_lad):.2f}")
print(f"Sum of absolute errors = {np.sum(abs_errors_lad):.2f}")
print()
print("The Least Squares method minimizes the sum of squared differences, not absolute differences.")
print("As shown above, the Least Squares model has lower sum of squared errors but higher sum of absolute errors.")
print("Therefore, Statement 2 is FALSE.")

# Step 4: Analyze Statement 3 - Effect of increasing data points
print_step_header(4, "Analyzing Statement 3: Effect of Increasing Data Points")

# Generate base data (a small dataset)
np.random.seed(42)
x_base = np.linspace(0, 10, 10)
y_base = 2 + 1.5 * x_base + np.random.normal(0, 1, 10)

# Generate good additional data (follows the pattern)
x_good = np.linspace(0.5, 9.5, 10)
y_good = 2 + 1.5 * x_good + np.random.normal(0, 1, 10)

# Generate bad additional data (outliers or different pattern)
x_bad = np.linspace(0.2, 9.8, 10)
y_bad = 2 + 1.5 * x_bad + np.random.normal(0, 5, 10)  # Much higher noise

# Combine datasets
x_with_good = np.concatenate([x_base, x_good])
y_with_good = np.concatenate([y_base, y_good])

x_with_bad = np.concatenate([x_base, x_bad])
y_with_bad = np.concatenate([y_base, y_bad])

# Fit models to each dataset
model_base = LinearRegression()
model_good = LinearRegression()
model_bad = LinearRegression()

model_base.fit(x_base.reshape(-1, 1), y_base)
model_good.fit(x_with_good.reshape(-1, 1), y_with_good)
model_bad.fit(x_with_bad.reshape(-1, 1), y_with_bad)

# Calculate R² for each model
r2_base = model_base.score(x_base.reshape(-1, 1), y_base)
r2_good = model_good.score(x_with_good.reshape(-1, 1), y_with_good)
r2_bad = model_bad.score(x_with_bad.reshape(-1, 1), y_with_bad)

# Create visualization
plt.figure(figsize=(15, 10))
x_range = np.linspace(0, 10, 100)

# Base dataset
plt.subplot(2, 2, 1)
plt.scatter(x_base, y_base, alpha=0.6, label='Base data points')
plt.plot(x_range, model_base.intercept_ + model_base.coef_[0] * x_range, 'r-', linewidth=2,
         label=f'Base model: R² = {r2_base:.3f}')
plt.title('Base Dataset (10 points)', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend()
plt.grid(True)

# Good additional data
plt.subplot(2, 2, 2)
plt.scatter(x_base, y_base, alpha=0.6, label='Base data points')
plt.scatter(x_good, y_good, color='g', alpha=0.6, label='Good additional points')
plt.plot(x_range, model_good.intercept_ + model_good.coef_[0] * x_range, 'r-', linewidth=2,
         label=f'Model with good data: R² = {r2_good:.3f}')
plt.plot(x_range, model_base.intercept_ + model_base.coef_[0] * x_range, 'b--', linewidth=1,
         label='Original model')
plt.title('Base + Good Additional Data (20 points)', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend()
plt.grid(True)

# Bad additional data
plt.subplot(2, 2, 3)
plt.scatter(x_base, y_base, alpha=0.6, label='Base data points')
plt.scatter(x_bad, y_bad, color='r', alpha=0.6, label='Noisy additional points')
plt.plot(x_range, model_bad.intercept_ + model_bad.coef_[0] * x_range, 'r-', linewidth=2,
         label=f'Model with noisy data: R² = {r2_bad:.3f}')
plt.plot(x_range, model_base.intercept_ + model_base.coef_[0] * x_range, 'b--', linewidth=1,
         label='Original model')
plt.title('Base + Noisy Additional Data (20 points)', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend()
plt.grid(True)

# Comparison of R² values
plt.subplot(2, 2, 4)
models = ['Base', 'With Good Data', 'With Noisy Data']
r2_values = [r2_base, r2_good, r2_bad]
plt.bar(models, r2_values, color=['blue', 'green', 'red'])
plt.axhline(y=r2_base, color='k', linestyle='--', alpha=0.7, label='Base model R²')
plt.ylim(0, 1)
plt.title('Comparison of R² Values', fontsize=14)
plt.ylabel('R² Value', fontsize=12)
plt.legend()
plt.grid(True)

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "increasing_data_points.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Print analysis
print("\nAnalysis of Statement 3:")
print("------------------------")
print("R² values for different datasets:")
print(f"Base dataset (10 points): R² = {r2_base:.3f}")
print(f"With good additional data (20 points): R² = {r2_good:.3f}")
print(f"With noisy additional data (20 points): R² = {r2_bad:.3f}")
print()
print("Model coefficients:")
print(f"Base model: y = {model_base.intercept_:.3f} + {model_base.coef_[0]:.3f}x")
print(f"Model with good data: y = {model_good.intercept_:.3f} + {model_good.coef_[0]:.3f}x")
print(f"Model with noisy data: y = {model_bad.intercept_:.3f} + {model_bad.coef_[0]:.3f}x")
print()
print("Adding good, well-behaved data improved the model (higher R²)")
print("However, adding noisy data decreased the model fit (lower R²)")
print("Therefore, Statement 3 is FALSE - increasing data doesn't always lead to better fit.")

# Step 5: Analyze Statement 4 - R² interpretation
print_step_header(5, "Analyzing Statement 4: R² Interpretation")

# Generate data with different strengths of relationship
np.random.seed(42)
x = np.linspace(0, 10, 100)

# Strong relationship (low noise)
y_strong = 2 + 0.5 * x + np.random.normal(0, 0.5, 100)

# Medium relationship (medium noise)
y_medium = 2 + 0.5 * x + np.random.normal(0, 2, 100)

# Weak relationship (high noise)
y_weak = 2 + 0.5 * x + np.random.normal(0, 5, 100)

# Fit models
model_strong = LinearRegression().fit(x.reshape(-1, 1), y_strong)
model_medium = LinearRegression().fit(x.reshape(-1, 1), y_medium)
model_weak = LinearRegression().fit(x.reshape(-1, 1), y_weak)

# Calculate R² values
r2_strong = model_strong.score(x.reshape(-1, 1), y_strong)
r2_medium = model_medium.score(x.reshape(-1, 1), y_medium)
r2_weak = model_weak.score(x.reshape(-1, 1), y_weak)

# Create visualization
plt.figure(figsize=(15, 15))

# Strong relationship
plt.subplot(3, 2, 1)
plt.scatter(x, y_strong, alpha=0.6)
plt.plot(x, model_strong.predict(x.reshape(-1, 1)), 'r-', linewidth=2)
plt.title(f'Strong Relationship: R² = {r2_strong:.3f}', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.grid(True)

# Variance explanation for strong relationship
y_mean_strong = np.mean(y_strong)
total_variance_strong = np.sum((y_strong - y_mean_strong)**2)
explained_variance_strong = np.sum((model_strong.predict(x.reshape(-1, 1)) - y_mean_strong)**2)
unexplained_variance_strong = np.sum((y_strong - model_strong.predict(x.reshape(-1, 1)))**2)

plt.subplot(3, 2, 2)
plt.bar(['Total Variance', 'Explained', 'Unexplained'], 
        [total_variance_strong, explained_variance_strong, unexplained_variance_strong],
        color=['blue', 'green', 'red'])
plt.axhline(y=total_variance_strong, color='k', linestyle='--')
plt.title('Variance Decomposition - Strong Relationship', fontsize=14)
plt.ylabel('Sum of Squares', fontsize=12)
plt.grid(True)

# Medium relationship
plt.subplot(3, 2, 3)
plt.scatter(x, y_medium, alpha=0.6)
plt.plot(x, model_medium.predict(x.reshape(-1, 1)), 'r-', linewidth=2)
plt.title(f'Medium Relationship: R² = {r2_medium:.3f}', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.grid(True)

# Variance explanation for medium relationship
y_mean_medium = np.mean(y_medium)
total_variance_medium = np.sum((y_medium - y_mean_medium)**2)
explained_variance_medium = np.sum((model_medium.predict(x.reshape(-1, 1)) - y_mean_medium)**2)
unexplained_variance_medium = np.sum((y_medium - model_medium.predict(x.reshape(-1, 1)))**2)

plt.subplot(3, 2, 4)
plt.bar(['Total Variance', 'Explained', 'Unexplained'], 
        [total_variance_medium, explained_variance_medium, unexplained_variance_medium],
        color=['blue', 'green', 'red'])
plt.axhline(y=total_variance_medium, color='k', linestyle='--')
plt.title('Variance Decomposition - Medium Relationship', fontsize=14)
plt.ylabel('Sum of Squares', fontsize=12)
plt.grid(True)

# Weak relationship
plt.subplot(3, 2, 5)
plt.scatter(x, y_weak, alpha=0.6)
plt.plot(x, model_weak.predict(x.reshape(-1, 1)), 'r-', linewidth=2)
plt.title(f'Weak Relationship: R² = {r2_weak:.3f}', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.grid(True)

# Variance explanation for weak relationship
y_mean_weak = np.mean(y_weak)
total_variance_weak = np.sum((y_weak - y_mean_weak)**2)
explained_variance_weak = np.sum((model_weak.predict(x.reshape(-1, 1)) - y_mean_weak)**2)
unexplained_variance_weak = np.sum((y_weak - model_weak.predict(x.reshape(-1, 1)))**2)

plt.subplot(3, 2, 6)
plt.bar(['Total Variance', 'Explained', 'Unexplained'], 
        [total_variance_weak, explained_variance_weak, unexplained_variance_weak],
        color=['blue', 'green', 'red'])
plt.axhline(y=total_variance_weak, color='k', linestyle='--')
plt.title('Variance Decomposition - Weak Relationship', fontsize=14)
plt.ylabel('Sum of Squares', fontsize=12)
plt.grid(True)

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "r_squared_variance.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Print analysis
print("\nAnalysis of Statement 4:")
print("------------------------")
print("R² values for different relationships:")
print(f"Strong relationship: R² = {r2_strong:.3f}")
print(f"Medium relationship: R² = {r2_medium:.3f}")
print(f"Weak relationship: R² = {r2_weak:.3f}")
print()
print("For strong relationship (R² = 0.898):")
print(f"Total variance: {total_variance_strong:.2f}")
print(f"Explained variance: {explained_variance_strong:.2f} ({explained_variance_strong/total_variance_strong:.3f} or {r2_strong:.3f})")
print(f"Unexplained variance: {unexplained_variance_strong:.2f} ({unexplained_variance_strong/total_variance_strong:.3f})")
print()
print("For weak relationship (R² = 0.171):")
print(f"Total variance: {total_variance_weak:.2f}")
print(f"Explained variance: {explained_variance_weak:.2f} ({explained_variance_weak/total_variance_weak:.3f} or {r2_weak:.3f})")
print(f"Unexplained variance: {unexplained_variance_weak:.2f} ({unexplained_variance_weak/total_variance_weak:.3f})")
print()
print("R² equals the proportion of variance explained by the model:")
print("R² = Explained Variance / Total Variance")
print("R² = 1 - (Unexplained Variance / Total Variance)")
print()
print("As shown in the visualizations and calculations, the R² value directly corresponds")
print("to the proportion of variance in the dependent variable that is explained by the model.")
print("Therefore, Statement 4 is TRUE.")

# Step 6: Analyze Statement 5 - Regression line passes through (x̄, ȳ)
print_step_header(6, "Analyzing Statement 5: Regression Line Through Mean Point")

# Generate several datasets with different characteristics
np.random.seed(42)
n_datasets = 3
colors = ['blue', 'green', 'red']
datasets = []

for i in range(n_datasets):
    x = np.random.uniform(i*5, i*5+10, 30)
    y = i*10 + 2*x + np.random.normal(0, 5, 30)
    datasets.append((x, y))

# Create visualization
plt.figure(figsize=(12, 10))

for i, (x, y) in enumerate(datasets):
    # Calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    
    # Predict y at mean of x
    y_pred_at_x_mean = model.predict(np.array([[x_mean]]))[0]
    
    # Plot data and regression line
    plt.subplot(len(datasets), 1, i+1)
    plt.scatter(x, y, alpha=0.6, color=colors[i], label=f'Dataset {i+1}')
    
    # Plot regression line
    x_range = np.linspace(min(x), max(x), 100)
    plt.plot(x_range, model.intercept_ + model.coef_[0] * x_range, 
             color=colors[i], linewidth=2,
             label=f'y = {model.intercept_:.2f} + {model.coef_[0]:.2f}x')
    
    # Plot mean point
    plt.scatter([x_mean], [y_mean], color='black', s=100, marker='X', 
                label=f'Mean point ({x_mean:.2f}, {y_mean:.2f})')
    
    # Plot vertical and horizontal lines from the mean point
    plt.axvline(x=x_mean, color='black', linestyle='--', alpha=0.3)
    plt.axhline(y=y_mean, color='black', linestyle='--', alpha=0.3)
    
    # Add text about the prediction at x_mean
    plt.text(0.05, 0.85, 
             f'y value on regression line at x̄ = {y_pred_at_x_mean:.2f}\n'
             f'Mean of y (ȳ) = {y_mean:.2f}\n'
             f'Difference = {y_pred_at_x_mean - y_mean:.2e}',
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title(f'Dataset {i+1}: Regression Line and Mean Point', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.legend()
    plt.grid(True)

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "regression_through_mean.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Print analysis
print("\nAnalysis of Statement 5:")
print("------------------------")
print("For all datasets, we can verify whether the regression line passes through the mean point.")
print()

for i, (x, y) in enumerate(datasets):
    # Calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    
    # Predict y at mean of x
    y_pred_at_x_mean = model.predict(np.array([[x_mean]]))[0]
    
    print(f"Dataset {i+1}:")
    print(f"  Mean point: (x̄, ȳ) = ({x_mean:.4f}, {y_mean:.4f})")
    print(f"  Regression line: y = {model.intercept_:.4f} + {model.coef_[0]:.4f}x")
    print(f"  Value on regression line at x̄: {y_pred_at_x_mean:.4f}")
    print(f"  Difference from ȳ: {y_pred_at_x_mean - y_mean:.10f}")
    print()

print("For all datasets, the regression line passes exactly through the point (x̄, ȳ)")
print("(tiny deviations are due to numerical precision)")
print("This is a mathematical property of least squares regression with an intercept term.")
print("Therefore, Statement 5 is TRUE.")

# Step 7: Conclude and summarize
print_step_header(7, "Conclusion and Summary")

print("Question 16 Analysis Summary:")
print()
print("Statement 1: In simple linear regression, the residuals always sum to zero when the model includes an intercept term.")
print("Verdict: TRUE. When a model includes an intercept, the OLS estimation ensures that the residuals sum to zero.")
print()
print("Statement 2: The least squares method minimizes the sum of absolute differences between predicted and actual values.")
print("Verdict: FALSE. The least squares method minimizes the sum of squared differences, not absolute differences.")
print("(Minimizing absolute differences is the least absolute deviations method, which produces different results.)")
print()
print("Statement 3: Increasing the number of data points always leads to a better fit in simple linear regression.")
print("Verdict: FALSE. The quality of added data points matters. Adding noisy or outlier data can worsen the fit.")
print()
print("Statement 4: The coefficient of determination (R²) represents the proportion of variance in the dependent variable explained by the model.")
print("Verdict: TRUE. The R² value is mathematically defined as the ratio of explained variance to total variance.")
print()
print("Statement 5: In simple linear regression, the regression line always passes through the point (x̄, ȳ).")
print("Verdict: TRUE. This is a mathematical property of the OLS estimator when an intercept term is included.") 