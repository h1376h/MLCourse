import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_1_Quiz_20")
os.makedirs(save_dir, exist_ok=True)

# Set a clean style for plots
plt.style.use('seaborn-v0_8-whitegrid')

print("# Statement 2: The variance of a random variable can be negative.")

# Image 1: Histogram with squared deviations visualization
np.random.seed(42)
data = np.random.normal(5, 2, 1000)
mean_val = np.mean(data)
var_val = np.var(data)

fig1, ax1 = plt.subplots(figsize=(8, 5))

# Plot histogram
ax1.hist(data, bins=30, alpha=0.7, color='skyblue', density=True)

# Plot normal distribution curve
x = np.linspace(min(data), max(data), 1000)
ax1.plot(x, stats.norm.pdf(x, mean_val, np.sqrt(var_val)), 'k-', linewidth=2)

# Highlight some deviations from mean
points = [1, 3, 5, 7, 9]
for point in points:
    # Draw lines showing deviation
    ax1.plot([point, point], [0, stats.norm.pdf(point, mean_val, np.sqrt(var_val))], 'g--', alpha=0.7)
    ax1.plot([point, mean_val], [0.02, 0.02], 'g--', alpha=0.7)
    
    # Label squared deviation
    sq_deviation = (point - mean_val)**2
    ax1.text(point, 0.03, f'(x-μ)²={sq_deviation:.2f}', ha='center', rotation=90, fontsize=10)

# Mark the mean
ax1.axvline(x=mean_val, color='red', linestyle='--', linewidth=2)

ax1.set_xlabel('Value (x)', fontsize=12)
ax1.set_ylabel('Probability Density', fontsize=12)
ax1.set_title('Visualization of Squared Deviations', fontsize=14)

# Save the figure
variance_img_path1 = os.path.join(save_dir, "statement2_squared_deviations.png")
plt.savefig(variance_img_path1, dpi=300, bbox_inches='tight')
plt.close()

# Image 2: Squares are always non-negative visualization
fig2, ax2 = plt.subplots(figsize=(8, 6))

# Generate x values from -5 to 5
x_values = np.linspace(-5, 5, 1000)
# Calculate squared values
squared_values = x_values**2

# Plot the function
ax2.plot(x_values, squared_values, 'b-', linewidth=2)

# Fill area to show non-negative property
ax2.fill_between(x_values, 0, squared_values, alpha=0.3, color='blue')

# Add reference lines
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.7)
ax2.axvline(x=0, color='black', linestyle='-', alpha=0.7)

# Mark some example points
example_points = [-4, -2, 0, 2, 4]
for point in example_points:
    ax2.plot([point], [point**2], 'ro', markersize=6)
    ax2.text(point, point**2 + 0.5, f'({point})² = {point**2}', 
             ha='center', fontsize=10)

ax2.set_xlabel('Value (x)', fontsize=12)
ax2.set_ylabel('Squared Value (x²)', fontsize=12)
ax2.set_title('Squared Values are Always Non-Negative', fontsize=14)

# Save the figure
variance_img_path2 = os.path.join(save_dir, "statement2_squared_function.png")
plt.savefig(variance_img_path2, dpi=300, bbox_inches='tight')
plt.close()

# Image 3: Variance formula and interpretation
fig3, ax3 = plt.subplots(figsize=(10, 6))

# Generate data with different variances
np.random.seed(42)
data1 = np.random.normal(0, 1, 1000)  # Low variance
data2 = np.random.normal(0, 2, 1000)  # Medium variance
data3 = np.random.normal(0, 3, 1000)  # High variance

# Calculate means and variances
mean1, var1 = np.mean(data1), np.var(data1)
mean2, var2 = np.mean(data2), np.var(data2)
mean3, var3 = np.mean(data3), np.var(data3)

# Plot histograms with KDE
x = np.linspace(-10, 10, 1000)
ax3.plot(x, stats.norm.pdf(x, mean1, np.sqrt(var1)), 'b-', linewidth=2, 
        label=f'Low Spread: σ² = {var1:.2f}')
ax3.plot(x, stats.norm.pdf(x, mean2, np.sqrt(var2)), 'g-', linewidth=2, 
        label=f'Medium Spread: σ² = {var2:.2f}')
ax3.plot(x, stats.norm.pdf(x, mean3, np.sqrt(var3)), 'r-', linewidth=2, 
        label=f'High Spread: σ² = {var3:.2f}')

# Add vertical line at mean
ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)

ax3.set_xlabel('Value', fontsize=12)
ax3.set_ylabel('Probability Density', fontsize=12)
ax3.set_title('Distributions with Different Variances', fontsize=14)
ax3.legend(loc='upper right')

# Save the figure
variance_img_path3 = os.path.join(save_dir, "statement2_variance_comparison.png")
plt.savefig(variance_img_path3, dpi=300, bbox_inches='tight')
plt.close()

# Mathematical explanation of variance - Print to terminal
print("#### Mathematical Analysis of Variance")
print("The variance of a random variable X is defined as:")
print("Var(X) = E[(X - μ)²] - The expected value of squared deviations from the mean")
print("")
print(f"For our example data with mean μ = {mean_val:.4f}:")
for point in points:
    sq_dev = (point - mean_val)**2
    print(f"  When x = {point}: (x - μ)² = ({point} - {mean_val:.4f})² = {sq_dev:.4f}")
print("")
print(f"The sample variance is: {var_val:.4f}")
print("")
print("#### Key Properties of Variance:")
print("1. Variance is the average of squared deviations from the mean")
print("2. Since we square the deviations (X - μ), each term (X - μ)² is non-negative")
print("3. The expected value of non-negative terms must also be non-negative")
print("4. Variance equals zero only when all values are identical (no variation)")
print("5. For any valid probability distribution, variance must be ≥ 0")
print("")
print("#### Examples of Variances:")
print(f"- Distribution with low spread: σ² = {var1:.4f}")
print(f"- Distribution with medium spread: σ² = {var2:.4f}")
print(f"- Distribution with high spread: σ² = {var3:.4f}")
print(f"- Constant random variable (all values are identical): σ² = 0")
print("")
print("#### Visual Verification:")
print(f"1. Visualization of squared deviations: {variance_img_path1}")
print(f"2. Squared values are always non-negative: {variance_img_path2}")
print(f"3. Comparison of distributions with different variances: {variance_img_path3}")
print("")
print("#### Conclusion:")
print("Since variance is defined as the expected value of squared terms, and squared terms")
print("are always non-negative, variance cannot be negative by definition.")
print("")
print("Therefore, Statement 2 is FALSE.") 