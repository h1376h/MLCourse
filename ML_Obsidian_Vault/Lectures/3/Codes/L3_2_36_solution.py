import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_2_Quiz_36")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Define the data
subjects = np.array([1, 2, 3, 4, 5, 6])
age = np.array([43, 21, 25, 42, 57, 59])
glucose = np.array([99, 65, 79, 75, 87, 81])
xy = np.array([4257, 1365, 1975, 3150, 4959, 4779])
x_squared = np.array([1849, 441, 625, 1764, 3249, 3481])
y_squared = np.array([9801, 4225, 6241, 5625, 7569, 6561])

# Calculate sums (to verify the given sums in the table)
sum_x = np.sum(age)
sum_y = np.sum(glucose)
sum_xy = np.sum(xy)
sum_x2 = np.sum(x_squared)
sum_y2 = np.sum(y_squared)

# Number of data points
n = len(age)

print("Step 1: Verify the sums from the table")
print(f"Sum of x: {sum_x}")
print(f"Sum of y: {sum_y}")
print(f"Sum of xy: {sum_xy}")
print(f"Sum of x²: {sum_x2}")
print(f"Sum of y²: {sum_y2}")
print()

# Calculate means
mean_x = sum_x / n
mean_y = sum_y / n

print("Step 2: Calculate the means")
print(f"Mean of x (mean age): {mean_x}")
print(f"Mean of y (mean glucose level): {mean_y}")
print()

# Calculate the slope (β₁)
# Formula: β₁ = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
numerator = n * sum_xy - sum_x * sum_y
denominator = n * sum_x2 - sum_x**2
beta1 = numerator / denominator

print("Step 3: Calculate the slope (β₁)")
print(f"β₁ = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)")
print(f"β₁ = ({n}*{sum_xy} - {sum_x}*{sum_y}) / ({n}*{sum_x2} - {sum_x}²)")
print(f"β₁ = ({n*sum_xy} - {sum_x*sum_y}) / ({n*sum_x2} - {sum_x**2})")
print(f"β₁ = {numerator} / {denominator}")
print(f"β₁ = {beta1}")
print()

# Calculate the intercept (β₀)
# Formula: β₀ = ȳ - β₁*x̄
beta0 = mean_y - beta1 * mean_x

print("Step 4: Calculate the intercept (β₀)")
print(f"β₀ = ȳ - β₁*x̄")
print(f"β₀ = {mean_y} - {beta1}*{mean_x}")
print(f"β₀ = {mean_y} - {beta1*mean_x}")
print(f"β₀ = {beta0}")
print()

# The regression equation
print("Step 5: Write the regression equation")
print(f"y = β₀ + β₁*x")
print(f"y = {beta0:.4f} + {beta1:.4f}*x")
print()

# Calculate the predicted values
y_pred = beta0 + beta1 * age

# Calculate R²
# First, calculate SST (Total Sum of Squares)
sst = sum_y2 - (sum_y**2) / n

# Calculate SSR (Regression Sum of Squares)
ssr = np.sum((y_pred - mean_y)**2)

# Calculate SSE (Error Sum of Squares)
sse = np.sum((glucose - y_pred)**2)

# Calculate R²
r_squared = ssr / sst

# Alternative calculation of R² 
# R² = 1 - SSE/SST
r_squared_alt = 1 - sse / sst

print("Step 6: Calculate R²")
print(f"SST (Total Sum of Squares) = Σy² - (Σy)²/n = {sum_y2} - {sum_y}²/{n} = {sst}")
print(f"SSR (Regression Sum of Squares) = Σ(ŷ - ȳ)² = {ssr}")
print(f"SSE (Error Sum of Squares) = Σ(y - ŷ)² = {sse}")
print(f"R² = SSR / SST = {ssr} / {sst} = {r_squared}")
print(f"Alternatively, R² = 1 - SSE/SST = 1 - {sse}/{sst} = {r_squared_alt}")
print()

# Create a correlation coefficient from the original data to cross-check
correlation_coef = np.corrcoef(age, glucose)[0, 1]
r_squared_from_corr = correlation_coef**2

print(f"Correlation coefficient: {correlation_coef}")
print(f"R² calculated from correlation: {r_squared_from_corr}")
print()

# Make predictions for age 35 and 78
age_35_pred = beta0 + beta1 * 35
age_78_pred = beta0 + beta1 * 78

print("Step 7: Make predictions")
print(f"Prediction for age 35: y = {beta0:.4f} + {beta1:.4f}*35 = {age_35_pred:.2f}")
print(f"Prediction for age 78: y = {beta0:.4f} + {beta1:.4f}*78 = {age_78_pred:.2f}")
print()

# Assess if the predictions are appropriate
min_age = np.min(age)
max_age = np.max(age)
min_glucose = np.min(glucose)
max_glucose = np.max(glucose)

print("Step 8: Assess if the predictions are appropriate")
print(f"Age range in the data: {min_age} to {max_age}")
print(f"Glucose level range in the data: {min_glucose} to {max_glucose}")
print()

print("Assessment of prediction for age 35:")
if min_age <= 35 <= max_age:
    print("Age 35 is within the range of the data, so the prediction is appropriate.")
else:
    print("Age 35 is outside the range of the data, so extrapolation is needed, which may reduce prediction accuracy.")

print()

print("Assessment of prediction for age 78:")
if min_age <= 78 <= max_age:
    print("Age 78 is within the range of the data, so the prediction is appropriate.")
else:
    print("Age 78 is outside the range of the data, so extrapolation is needed, which may reduce prediction accuracy.")
    print(f"The closest observed age in the data is {max_age}, which is {78 - max_age} years less than 78.")
    print("Extrapolating far beyond the observed data can lead to unreliable predictions.")

print()

# Calculate standard error of the estimate
df = n - 2  # Degrees of freedom
se = np.sqrt(sse / df)

print("Step 9: Calculate standard error and confidence intervals")
print(f"Standard error of the estimate: {se:.4f}")

# Calculate confidence intervals for the predictions
alpha = 0.05  # 95% confidence
t_critical = stats.t.ppf(1 - alpha/2, df)
print(f"t-critical value (95% confidence, {df} df): {t_critical:.4f}")

# Create a range of ages for plotting
x_range = np.linspace(0, 90, 100)

# Calculate predictions for the range
y_range_pred = beta0 + beta1 * x_range

# Create visualizations

# Plot 1: Scatter plot with regression line
plt.figure(figsize=(10, 6))
plt.scatter(age, glucose, color='blue', s=60, alpha=0.7, label='Observed data')
plt.plot(x_range, y_range_pred, color='red', linewidth=2, label=f'Regression line: y = {beta0:.2f} + {beta1:.2f}x')

# Add prediction points
plt.scatter([35, 78], [age_35_pred, age_78_pred], color='green', s=100, marker='X', 
           label='Predictions (age 35 and 78)')

# Add labels and annotations
for i, (x, y) in enumerate(zip(age, glucose)):
    plt.annotate(f'Subject {i+1}', (x, y), xytext=(5, 5), textcoords='offset points')

plt.axvspan(min_age-1, max_age+1, alpha=0.2, color='gray', label='Data range')

# Add vertical lines for prediction ages
plt.axvline(x=35, color='green', linestyle='--', alpha=0.5)
plt.axvline(x=78, color='green', linestyle='--', alpha=0.5)

plt.title('Linear Regression: Age vs Glucose Level', fontsize=14)
plt.xlabel('Age (years)', fontsize=12)
plt.ylabel('Glucose Level', fontsize=12)
plt.grid(True)
plt.legend(fontsize=10)
plt.tight_layout()

# Save the plot
plot1_path = os.path.join(save_dir, "plot1_regression_line.png")
plt.savefig(plot1_path, dpi=300, bbox_inches='tight')

print(f"Plot saved to: {plot1_path}")

# Plot 2: Regression with confidence intervals
plt.figure(figsize=(10, 6))

# Calculate confidence bands
# For each x, calculate the standard error of the prediction
se_pred = np.zeros_like(x_range)
for i, x in enumerate(x_range):
    se_pred[i] = se * np.sqrt(1 + 1/n + (x - mean_x)**2 / np.sum((age - mean_x)**2))

# Calculate confidence bands
lower_band = y_range_pred - t_critical * se_pred
upper_band = y_range_pred + t_critical * se_pred

# Plot data and regression line
plt.scatter(age, glucose, color='blue', s=60, alpha=0.7, label='Observed data')
plt.plot(x_range, y_range_pred, color='red', linewidth=2, label=f'Regression line: y = {beta0:.2f} + {beta1:.2f}x')

# Plot confidence bands
plt.fill_between(x_range, lower_band, upper_band, color='red', alpha=0.15, 
                label='95% Confidence interval')

# Highlight the data range and extrapolation regions
plt.axvspan(0, min_age, alpha=0.1, color='yellow', label='Extrapolation region')
plt.axvspan(max_age, 90, alpha=0.1, color='yellow')
plt.axvspan(min_age, max_age, alpha=0.1, color='green', label='Interpolation region')

# Add predictions
plt.scatter([35, 78], [age_35_pred, age_78_pred], color='green', s=100, marker='X', 
           label='Predictions (age 35 and 78)')

# Add vertical lines for prediction ages
plt.axvline(x=35, color='green', linestyle='--', alpha=0.5)
plt.axvline(x=78, color='green', linestyle='--', alpha=0.5)

# Add annotations for the confidence intervals at prediction points
se_35 = se * np.sqrt(1 + 1/n + (35 - mean_x)**2 / np.sum((age - mean_x)**2))
se_78 = se * np.sqrt(1 + 1/n + (78 - mean_x)**2 / np.sum((age - mean_x)**2))

ci_35_lower = age_35_pred - t_critical * se_35
ci_35_upper = age_35_pred + t_critical * se_35
ci_78_lower = age_78_pred - t_critical * se_78
ci_78_upper = age_78_pred + t_critical * se_78

plt.annotate(f'Age 35: {age_35_pred:.2f} [{ci_35_lower:.2f}, {ci_35_upper:.2f}]', 
             xy=(35, age_35_pred), xytext=(35+2, age_35_pred+5),
             arrowprops=dict(arrowstyle='->'), fontsize=10)

plt.annotate(f'Age 78: {age_78_pred:.2f} [{ci_78_lower:.2f}, {ci_78_upper:.2f}]', 
             xy=(78, age_78_pred), xytext=(78-25, age_78_pred+5),
             arrowprops=dict(arrowstyle='->'), fontsize=10)

plt.title('Linear Regression with Confidence Intervals', fontsize=14)
plt.xlabel('Age (years)', fontsize=12)
plt.ylabel('Glucose Level', fontsize=12)
plt.grid(True)
plt.legend(fontsize=10, loc='upper left')
plt.tight_layout()

# Save the plot
plot2_path = os.path.join(save_dir, "plot2_confidence_intervals.png")
plt.savefig(plot2_path, dpi=300, bbox_inches='tight')

print(f"Plot saved to: {plot2_path}")

# Plot 3: Residual plot
residuals = glucose - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(age, residuals, color='blue', s=60, alpha=0.7)
plt.axhline(y=0, color='red', linestyle='-', linewidth=2)

# Add reference lines for standard error
plt.axhline(y=se, color='gray', linestyle='--', alpha=0.7, label=f'±1 SE ({se:.2f})')
plt.axhline(y=-se, color='gray', linestyle='--', alpha=0.7)

# Add labels
for i, (x, r) in enumerate(zip(age, residuals)):
    plt.annotate(f'Subject {i+1}', (x, r), xytext=(5, 5), textcoords='offset points')

plt.title('Residual Plot', fontsize=14)
plt.xlabel('Age (years)', fontsize=12)
plt.ylabel('Residuals (Observed - Predicted)', fontsize=12)
plt.grid(True)
plt.legend(fontsize=10)
plt.tight_layout()

# Save the plot
plot3_path = os.path.join(save_dir, "plot3_residuals.png")
plt.savefig(plot3_path, dpi=300, bbox_inches='tight')

print(f"Plot saved to: {plot3_path}")

# Print a summary of the findings
print("\nSUMMARY:")
print("=========")
print(f"Linear Regression Equation: y = {beta0:.4f} + {beta1:.4f}x")
print(f"Where y is the glucose level and x is the age")
print(f"R² = {r_squared:.4f}")
print(f"Prediction for age 35: {age_35_pred:.2f}")
print(f"Prediction for age 78: {age_78_pred:.2f}")
print("Age 35 is within the data range, so this prediction is appropriate for interpolation.")
print("Age 78 is outside the data range, requiring extrapolation, which may be less reliable.") 