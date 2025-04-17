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

print("# Statement 4: If the correlation coefficient between random variables X and Y is 0, then X and Y must be independent.")

# Image 1: Scatter plot of dependent but uncorrelated variables
np.random.seed(42)
n = 1000

# Generate x as standard normal
x = np.random.normal(0, 1, n)

# Generate y = x² (dependent on x but uncorrelated)
y = x**2 - 1  # Subtracting 1 to make E[Y] close to 0

# Calculate correlation
correlation = np.corrcoef(x, y)[0, 1]

# Create scatter plot
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.scatter(x, y, alpha=0.6, color='blue', s=20)
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

ax1.set_xlabel('X', fontsize=12)
ax1.set_ylabel('Y = X² - 1', fontsize=12)
ax1.set_title('Dependent but Uncorrelated Variables', fontsize=14)
ax1.grid(True, alpha=0.3)

# Save the figure
correlation_img_path1 = os.path.join(save_dir, "statement4_dependent_uncorrelated.png")
plt.savefig(correlation_img_path1, dpi=300, bbox_inches='tight')
plt.close()

# Image 2: Truly independent variables for comparison
np.random.seed(123)
x2 = np.random.normal(0, 1, n)
y2 = np.random.normal(0, 1, n)
corr2 = np.corrcoef(x2, y2)[0, 1]

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.scatter(x2, y2, alpha=0.6, color='green', s=20)
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

ax2.set_xlabel('X', fontsize=12)
ax2.set_ylabel('Y (independent of X)', fontsize=12)
ax2.set_title('Truly Independent Variables', fontsize=14)
ax2.grid(True, alpha=0.3)

# Save the figure
correlation_img_path2 = os.path.join(save_dir, "statement4_truly_independent.png")
plt.savefig(correlation_img_path2, dpi=300, bbox_inches='tight')
plt.close()

# Image 3: Visualization of correlation vs independence
# Generate several examples
np.random.seed(42)
n = 200

# Example 1: Linear relationship (correlated, dependent)
x1 = np.random.normal(0, 1, n)
y1 = 0.8 * x1 + 0.2 * np.random.normal(0, 1, n)
corr1 = np.corrcoef(x1, y1)[0, 1]

# Example 2: Quadratic relationship (uncorrelated, dependent)
x2 = np.random.normal(0, 1, n)
y2 = x2**2 - 1 + 0.2 * np.random.normal(0, 1, n)
corr2 = np.corrcoef(x2, y2)[0, 1]

# Example 3: Sinusoidal relationship (uncorrelated, dependent)
x3 = np.random.uniform(-np.pi, np.pi, n)
y3 = np.sin(x3) + 0.2 * np.random.normal(0, 1, n)
corr3 = np.corrcoef(x3, y3)[0, 1]

# Example 4: Independent variables (uncorrelated, independent)
x4 = np.random.normal(0, 1, n)
y4 = np.random.normal(0, 1, n)
corr4 = np.corrcoef(x4, y4)[0, 1]

# Create visualization with four subplots
fig3, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot example 1
axes[0, 0].scatter(x1, y1, alpha=0.6, color='blue', s=20)
axes[0, 0].set_title(f'Linear Relationship\nCorrelation: {corr1:.4f}\nDependent: Yes', fontsize=12)
axes[0, 0].grid(True, alpha=0.3)

# Plot example 2
axes[0, 1].scatter(x2, y2, alpha=0.6, color='red', s=20)
axes[0, 1].set_title(f'Quadratic Relationship\nCorrelation: {corr2:.4f}\nDependent: Yes', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)

# Plot example 3
axes[1, 0].scatter(x3, y3, alpha=0.6, color='green', s=20)
axes[1, 0].set_title(f'Sinusoidal Relationship\nCorrelation: {corr3:.4f}\nDependent: Yes', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)

# Plot example 4
axes[1, 1].scatter(x4, y4, alpha=0.6, color='purple', s=20)
axes[1, 1].set_title(f'No Relationship\nCorrelation: {corr4:.4f}\nDependent: No', fontsize=12)
axes[1, 1].grid(True, alpha=0.3)

# Add overall title
fig3.suptitle('Correlation vs. Independence: Four Scenarios', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save the figure
correlation_img_path3 = os.path.join(save_dir, "statement4_correlation_vs_independence.png")
plt.savefig(correlation_img_path3, dpi=300, bbox_inches='tight')
plt.close()

# Explain correlation vs independence - Print to terminal
print("#### Mathematical Analysis of Correlation and Independence")
print("The correlation coefficient between random variables X and Y is defined as:")
print("ρ(X,Y) = Cov(X,Y) / (σ_X σ_Y)")
print("where Cov(X,Y) is the covariance and σ_X, σ_Y are standard deviations.")
print("")
print("#### Key Relationship Between Correlation and Independence:")
print("- If X and Y are independent, then Cov(X,Y) = 0 and thus ρ(X,Y) = 0")
print("- However, the converse is not true: ρ(X,Y) = 0 does not imply independence")
print("- Zero correlation only indicates no linear relationship between X and Y")
print("- Non-linear dependencies can exist even when correlation is zero")
print("")
print("#### Numerical Examples:")
print(f"1. Y = X² - 1 (Quadratic relationship):")
print(f"   Correlation coefficient: {correlation:.6f}")
print(f"   This is nearly zero, yet Y is completely determined by X")
print("")
print(f"2. Truly independent variables:")
print(f"   Correlation coefficient: {corr2:.6f}")
print(f"   This is also nearly zero, and X and Y are genuinely independent")
print("")
print("#### Additional Examples:")
print(f"3. Linear relationship: ρ(X,Y) = {corr1:.4f} (Correlated and dependent)")
print(f"4. Quadratic relationship: ρ(X,Y) = {corr2:.4f} (Uncorrelated but dependent)")
print(f"5. Sinusoidal relationship: ρ(X,Y) = {corr3:.4f} (Uncorrelated but dependent)")
print(f"6. Independent variables: ρ(X,Y) = {corr4:.4f} (Uncorrelated and independent)")
print("")
print("#### Counterexample Analysis:")
print("For Y = X² - 1:")
print("1. Knowing X completely determines Y, so they are clearly dependent")
print("2. Yet their correlation coefficient is approximately zero")
print("3. This is because correlation only measures linear relationships")
print("4. The positive correlation in the right half cancels with the negative correlation in the left half")
print("")
print("#### Visual Verification:")
print(f"1. Dependent but uncorrelated variables: {correlation_img_path1}")
print(f"2. Truly independent variables: {correlation_img_path2}")
print(f"3. Four scenarios comparing correlation and independence: {correlation_img_path3}")
print("")
print("#### Conclusion:")
print("Zero correlation between X and Y indicates no linear relationship, but")
print("non-linear dependencies can still exist. Independence is a stronger")
print("condition that ensures no relationship of any form between the variables.")
print("")
print("Therefore, Statement 4 is FALSE.") 