import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L7_3_Quiz_15")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("Question 15: Random Forest Performance Analysis")
print("=" * 50)

# Given data
trees = np.array([25, 50, 75])
accuracies = np.array([85, 87, 89])

print(f"Given data:")
print(f"Number of trees: {trees}")
print(f"Accuracies: {accuracies}%")

# Step 1: Linear trend analysis and prediction for 100 trees
print("\n" + "="*50)
print("STEP 1: Linear Trend Analysis and Prediction for 100 Trees")
print("="*50)

# Calculate linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(trees, accuracies)

print(f"Linear regression parameters:")
print(f"Slope (m) = {slope:.4f}")
print(f"Intercept (b) = {intercept:.4f}")
print(f"R-squared = {r_value**2:.4f}")
print(f"P-value = {p_value:.6f}")

# Equation of the line: accuracy = m * trees + b
print(f"\nLinear equation: accuracy = {slope:.4f} × trees + {intercept:.4f}")

# Predict accuracy for 100 trees
predicted_100 = slope * 100 + intercept
print(f"\nPredicted accuracy for 100 trees:")
print(f"accuracy = {slope:.4f} × 100 + {intercept:.4f} = {predicted_100:.2f}%")

# Step 2: Calculate trees needed for 92% accuracy
print("\n" + "="*50)
print("STEP 2: Trees Needed for 92% Accuracy")
print("="*50)

# Solve for trees: 92 = m * trees + b
# trees = (92 - b) / m
trees_for_92 = (92 - intercept) / slope
print(f"To achieve 92% accuracy:")
print(f"92 = {slope:.4f} × trees + {intercept:.4f}")
print(f"trees = (92 - {intercept:.4f}) / {slope:.4f}")
print(f"trees = {trees_for_92:.2f}")

# Round up to nearest whole number since we can't have partial trees
trees_for_92_rounded = np.ceil(trees_for_92)
print(f"Rounded up: {trees_for_92_rounded:.0f} trees needed")

# Step 3: Accuracy improvement per additional tree
print("\n" + "="*50)
print("STEP 3: Accuracy Improvement per Additional Tree")
print("="*50)

print(f"Accuracy improvement per additional tree = slope = {slope:.4f}%")
print(f"This means each additional tree improves accuracy by approximately {slope:.4f}%")

# Step 4: Time calculation for 92% accuracy
print("\n" + "="*50)
print("STEP 4: Time Calculation for 92% Accuracy")
print("="*50)

time_per_tree = 3  # minutes
total_time_minutes = trees_for_92_rounded * time_per_tree
total_time_hours = total_time_minutes / 60

print(f"Time per tree: {time_per_tree} minutes")
print(f"Total trees needed: {trees_for_92_rounded:.0f}")
print(f"Total time: {total_time_minutes:.0f} minutes = {total_time_hours:.2f} hours")

# Step 5: Correlation analysis and statistical significance
print("\n" + "="*50)
print("STEP 5: Correlation Analysis and Statistical Significance")
print("="*50)

# Calculate correlation coefficient
correlation = np.corrcoef(trees, accuracies)[0, 1]
print(f"Correlation coefficient (r) = {correlation:.4f}")

# Calculate t-statistic and p-value for correlation
n = len(trees)
t_stat = correlation * np.sqrt((n-2) / (1 - correlation**2))
p_value_corr = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))

print(f"T-statistic = {t_stat:.4f}")
print(f"P-value for correlation = {p_value_corr:.6f}")

# Determine statistical significance
alpha = 0.05
is_significant = p_value_corr < alpha
print(f"Statistical significance (α = {alpha}): {'Significant' if is_significant else 'Not significant'}")

# Create comprehensive visualization
plt.figure(figsize=(15, 10))

# Subplot 1: Linear regression and predictions
plt.subplot(2, 2, 1)
x_range = np.linspace(0, 120, 100)
y_range = slope * x_range + intercept

plt.plot(x_range, y_range, 'b-', label=f'Linear fit: y = {slope:.4f}x + {intercept:.4f}')
plt.scatter(trees, accuracies, color='red', s=100, label='Given data points')
plt.scatter(100, predicted_100, color='green', s=100, marker='s', label=f'Predicted (100 trees): {predicted_100:.2f}%')
plt.scatter(trees_for_92, 92, color='purple', s=100, marker='^', label=f'Target (92%): {trees_for_92:.1f} trees')

plt.xlabel('Number of Trees')
plt.ylabel('Accuracy (%)')
plt.title('Linear Regression Analysis')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(0, 120)
plt.ylim(80, 95)

# Subplot 2: Accuracy improvement per tree
plt.subplot(2, 2, 2)
trees_extended = np.arange(1, 101)
accuracies_extended = slope * trees_extended + intercept
improvements = np.diff(accuracies_extended)

plt.plot(trees_extended[1:], improvements, 'g-', linewidth=2)
plt.axhline(y=slope, color='r', linestyle='--', label=f'Constant improvement: {slope:.4f}%')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy Improvement (%)')
plt.title('Accuracy Improvement per Additional Tree')
plt.grid(True, alpha=0.3)
plt.legend()

# Subplot 3: Time vs Accuracy trade-off
plt.subplot(2, 2, 3)
time_minutes = trees_extended * time_per_tree
plt.plot(time_minutes, accuracies_extended, 'orange', linewidth=2)
plt.scatter(trees * time_per_tree, accuracies, color='red', s=100, label='Given data points')
plt.scatter(trees_for_92_rounded * time_per_tree, 92, color='purple', s=100, marker='^', label=f'Target: {trees_for_92_rounded:.0f} trees')

plt.xlabel('Training Time (minutes)')
plt.ylabel('Accuracy (%)')
plt.title('Time vs Accuracy Trade-off')
plt.grid(True, alpha=0.3)
plt.legend()

# Subplot 4: Residuals analysis
plt.subplot(2, 2, 4)
predicted_given = slope * trees + intercept
residuals = accuracies - predicted_given

plt.scatter(trees, residuals, color='red', s=100)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
plt.xlabel('Number of Trees')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residuals Analysis')
plt.grid(True, alpha=0.3)

# Add residual values as annotations
for i, (x, y) in enumerate(zip(trees, residuals)):
    plt.annotate(f'{y:.2f}', (x, y), xytext=(5, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'random_forest_performance_analysis.png'), dpi=300, bbox_inches='tight')

# Create detailed results table
print("\n" + "="*50)
print("SUMMARY OF RESULTS")
print("="*50)

results_data = {
    'Metric': [
        'Linear equation',
        'Predicted accuracy (100 trees)',
        'Trees needed for 92% accuracy',
        'Accuracy improvement per tree',
        'Time for 92% accuracy',
        'Correlation coefficient',
        'Statistical significance'
    ],
    'Value': [
        f'y = {slope:.4f}x + {intercept:.4f}',
        f'{predicted_100:.2f}%',
        f'{trees_for_92_rounded:.0f} trees',
        f'{slope:.4f}%',
        f'{total_time_hours:.2f} hours',
        f'{correlation:.4f}',
        'Significant' if is_significant else 'Not significant'
    ]
}

results_df = pd.DataFrame(results_data)
print(results_df.to_string(index=False))

# Save results to text file
with open(os.path.join(save_dir, 'results_summary.txt'), 'w') as f:
    f.write("Random Forest Performance Analysis - Question 15\n")
    f.write("=" * 50 + "\n\n")
    f.write("Given Data:\n")
    f.write(f"Trees: {trees}\n")
    f.write(f"Accuracies: {accuracies}%\n\n")
    f.write("Results:\n")
    f.write(f"Linear equation: y = {slope:.4f}x + {intercept:.4f}\n")
    f.write(f"Predicted accuracy for 100 trees: {predicted_100:.2f}%\n")
    f.write(f"Trees needed for 92% accuracy: {trees_for_92_rounded:.0f}\n")
    f.write(f"Accuracy improvement per tree: {slope:.4f}%\n")
    f.write(f"Time for 92% accuracy: {total_time_hours:.2f} hours\n")
    f.write(f"Correlation coefficient: {correlation:.4f}\n")
    f.write(f"Statistical significance: {'Significant' if is_significant else 'Not significant'}\n")

print(f"\nResults saved to: {save_dir}")
print("Analysis complete!")
