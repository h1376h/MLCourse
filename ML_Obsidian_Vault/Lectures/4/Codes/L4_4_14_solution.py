import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_4_Quiz_14")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

print("Question 14: Comparing Classifiers")
print("==================================")

# Step 1: Define the decision boundary equation for Model A (Perceptron)
print("\nStep 1: Decision Boundary Equation for Model A (Perceptron)")
print("----------------------------------------------------------")

# Model A parameters
w_perceptron = np.array([2, -1])
b_perceptron = 0.5

print(f"Given information:")
print(f"Perceptron weights: w = [{w_perceptron[0]}, {w_perceptron[1]}]^T")
print(f"Perceptron bias: b = {b_perceptron}")

# Write the decision boundary equation
print("\nDecision boundary equation in the form w₁x₁ + w₂x₂ + b = 0:")
print(f"{w_perceptron[0]}x₁ + {w_perceptron[1]}x₂ + {b_perceptron} = 0")
print(f"2x₁ - 1x₂ + 0.5 = 0")

# Rearrange to standard form
print("\nRearranging to solve for x₂:")
print(f"x₂ = {w_perceptron[0]}x₁/{-w_perceptron[1]} + {b_perceptron}/{-w_perceptron[1]}")
print(f"x₂ = 2x₁ + 0.5")

# Step 2: Detailed step-by-step calculations for markdown (printed to console)
print("\nStep-by-step Calculation for Decision Boundary:")
print("===============================================")
print("Given:")
print("- Weights: w = [2, -1]^T")
print("- Bias: b = 0.5")
print("\nDecision boundary equation:")
print("w_1 x_1 + w_2 x_2 + b = 0")
print("2x_1 + (-1)x_2 + 0.5 = 0")
print("2x_1 - x_2 + 0.5 = 0")
print("\nSolving for x_2:")
print("x_2 = 2x_1 + 0.5")
print("This is a line with slope 2 and y-intercept 0.5")

# Step 3: Classify the test point (1, 2) using Model A
print("\nStep 3: Classify Test Point (1, 2) using Model A (Perceptron)")
print("-----------------------------------------------------------")

# Test point
test_point = np.array([1, 2])
print(f"Test point: ({test_point[0]}, {test_point[1]})")

# Calculate the decision function value
decision_value = np.dot(w_perceptron, test_point) + b_perceptron
print(f"Decision function value: w^T·x + b = {w_perceptron[0]}×{test_point[0]} + {w_perceptron[1]}×{test_point[1]} + {b_perceptron} = {decision_value}")

# Determine the predicted class
predicted_class = 1 if decision_value > 0 else -1
print(f"Since the decision value is {decision_value}, which is {'positive' if decision_value > 0 else 'negative'},")
print(f"Model A predicts that the point belongs to class {predicted_class}.")

# Step 4: Detailed step-by-step calculations for classifying the test point (printed to console)
print("\nStep-by-step Calculation for Classifying Test Point (1, 2):")
print("==========================================================")
print("Given:")
print("- Test point: x = [1, 2]^T")
print("- Weights: w = [2, -1]^T")
print("- Bias: b = 0.5")
print("\nCalculate decision function value:")
print("f(x) = w^T · x + b")
print("f(x) = w_1 x_1 + w_2 x_2 + b")
print("f(x) = 2 × 1 + (-1) × 2 + 0.5")
print("f(x) = 2 - 2 + 0.5 = 0.5")
print("\nClassification rule:")
print("If f(x) > 0, predict class +1")
print("If f(x) < 0, predict class -1")
print("\nSince f(x) = 0.5 > 0, the point (1, 2) is classified as class +1")

# Step 5: Theoretical comparison of LDA and Perceptron (printed to console)
print("\nTheoretical Comparison of LDA vs Perceptron:")
print("===========================================")
print("LDA Theory:")
print("- Assumes classes follow Gaussian distributions: P(x|C_k) = N(x|μ_k, Σ)")
print("- With equal covariance matrices Σ and equal priors P(C_1) = P(C_2)")
print("- The decision boundary is given by:")
print("  (μ_1 - μ_2)^T Σ^(-1) x - 1/2(μ_1 - μ_2)^T Σ^(-1) (μ_1 + μ_2) = 0")
print("- The weight vector is w = Σ^(-1)(μ_1 - μ_2)")
print("- This is the Bayes-optimal boundary when assumptions hold")
print("\nPerceptron Theory:")
print("- Iteratively updates weights: w_(t+1) = w_t + η y_i x_i for misclassified points")
print("- Stops when all training points are correctly classified")
print("- Makes no assumptions about underlying distributions")
print("- Weight vector direction depends on the training algorithm, initialization, and misclassified points")
print("\nKey Differences:")
print("1. LDA: Based on probabilistic modeling and Bayes' rule")
print("2. Perceptron: Based on iterative error correction")
print("3. LDA: Optimal when data is Gaussian with equal covariance")
print("4. Perceptron: Can find any linear separator, not necessarily optimal")

# Create a prettier visualization of the decision boundary
plt.figure(figsize=(9, 7))

# Define a range for the plot
x1_range = np.linspace(-1.5, 3.5, 500)
x2_range = np.linspace(-1.5, 4.5, 500)
xx, yy = np.meshgrid(x1_range, x2_range)
grid = np.c_[xx.ravel(), yy.ravel()]

# Calculate decision function values
Z_perceptron = np.dot(grid, w_perceptron) + b_perceptron
Z_perceptron = Z_perceptron.reshape(xx.shape)

# Create a custom colormap for a more elegant visualization
colors = ["#9bc1bc", "#f4f1bb"]
custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=256)

# Plot the decision regions with improved coloring
plt.contourf(xx, yy, Z_perceptron, levels=[-100, 0, 100], cmap=custom_cmap, alpha=0.9)

# Plot the decision boundary as a simple line with a cleaner look
plt.contour(xx, yy, Z_perceptron, levels=[0], colors='#ed6a5a', linewidths=3)

# Add some sample points for each class
np.random.seed(42)
pos_points = np.array([[0.5, 3.0], [0.8, 2.5], [1.2, 3.2], [1.5, 3.6]])
neg_points = np.array([[2.0, 1.0], [2.5, 0.8], [3.0, 1.5], [2.2, 1.2]])

# Add some random variation
pos_points += np.random.normal(0, 0.1, pos_points.shape)
neg_points += np.random.normal(0, 0.1, neg_points.shape)

# Plot the points
plt.scatter(pos_points[:, 0], pos_points[:, 1], color='#5d576b', s=80, marker='o', edgecolor='white', linewidth=1.5, alpha=0.8)
plt.scatter(neg_points[:, 0], neg_points[:, 1], color='#e07a5f', s=80, marker='s', edgecolor='white', linewidth=1.5, alpha=0.8)

# Plot the test point with a more noticeable style
plt.scatter(test_point[0], test_point[1], color='#FF1493', s=180, marker='*', edgecolor='white', linewidth=1.5, zorder=10)

# Add a subtle effect to show the test point's position
circle = plt.Circle((test_point[0], test_point[1]), 0.2, color='#FF1493', fill=False, linestyle='--', linewidth=2, alpha=0.6)
plt.gca().add_patch(circle)

# Draw the coordinate axes with a cleaner style
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)

# Clean up the plot with a more elegant style
plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('Perceptron Decision Boundary', fontsize=18, pad=15)
plt.grid(True, alpha=0.2)

# Add a slight margin around the plot
plt.xlim(-1.2, 3.2)
plt.ylim(-0.8, 4.2)

# Save the improved plot
plt.savefig(os.path.join(save_dir, "simple_decision_boundary.png"), dpi=300, bbox_inches='tight')

# Create a new visualization: decision function values across the feature space
plt.figure(figsize=(10, 8))

# Calculate decision function values
x1_fine = np.linspace(-2, 4, 100)
x2_fine = np.linspace(-2, 5, 100)
xx_fine, yy_fine = np.meshgrid(x1_fine, x2_fine)
grid_fine = np.c_[xx_fine.ravel(), yy_fine.ravel()]
Z_values = np.dot(grid_fine, w_perceptron) + b_perceptron
Z_values = Z_values.reshape(xx_fine.shape)

# Create a gradient visualization of the decision function
plt.figure(figsize=(9, 7))

# Create a more informative colormap for the decision values
gradient_cmap = plt.cm.RdBu_r

# Plot the decision function values
plt.contourf(xx_fine, yy_fine, Z_values, levels=np.linspace(-3, 3, 50), cmap=gradient_cmap, alpha=0.9)

# Add contour lines to show specific decision function values
contour_levels = [-2, -1, -0.5, 0, 0.5, 1, 2]
contours = plt.contour(xx_fine, yy_fine, Z_values, levels=contour_levels, colors='black', linewidths=1, alpha=0.7)
plt.clabel(contours, inline=True, fontsize=10, fmt='%.1f')

# Add a strong line for the decision boundary
plt.contour(xx_fine, yy_fine, Z_values, levels=[0], colors='#ed6a5a', linewidths=3)

# Plot the test point with the same style as before
plt.scatter(test_point[0], test_point[1], color='#FF1493', s=180, marker='*', edgecolor='white', linewidth=1.5, zorder=10)

# Draw the coordinate axes
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)

# Clean up the plot
plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('Decision Function Values in Feature Space', fontsize=18, pad=15)
plt.grid(True, alpha=0.2)

# Add a colorbar to show the mapping between colors and decision function values
cbar = plt.colorbar()
cbar.set_label('Decision Function Value ($w^T x + b$)', fontsize=12)

# Add a slight margin around the plot
plt.xlim(-1.2, 3.2)
plt.ylim(-0.8, 4.2)

# Save the new visualization
plt.savefig(os.path.join(save_dir, "decision_function_values.png"), dpi=300, bbox_inches='tight')

# Create a simple LDA vs Perceptron comparison
plt.figure(figsize=(10, 6))

# Generate minimal synthetic data for clear visualization
np.random.seed(42)
n_samples = 25  # Fewer samples for clearer visualization

# Class means
mean1 = np.array([2, 2])
mean2 = np.array([0, 0])
cov = np.array([[1, 0], [0, 1]])  # Simplified covariance (diagonal)

# Generate data
X1 = np.random.multivariate_normal(mean1, cov, n_samples)
X2 = np.random.multivariate_normal(mean2, cov, n_samples)

# Theoretical decision boundaries
# For equal covariance and equal priors, the LDA boundary passes through the midpoint of means
mid_point = (mean1 + mean2) / 2
direction = mean1 - mean2  # This is parallel to w for LDA

# Create a theoretical LDA boundary
x1_line = np.linspace(-1, 3, 100)
# LDA: perpendicular to the line connecting means, passing through midpoint
# Slope is -direction[0]/direction[1]
slope_lda = -direction[0]/direction[1] if direction[1] != 0 else np.inf
x2_line_lda = mid_point[1] + slope_lda * (x1_line - mid_point[0])

# Create a theoretical Perceptron boundary (slightly different angle)
# This is just for visualization
slope_perceptron = -1.1 * direction[0]/direction[1] if direction[1] != 0 else np.inf
x2_line_perceptron = mid_point[1] + slope_perceptron * (x1_line - mid_point[0])

# Plot the data points
plt.scatter(X1[:, 0], X1[:, 1], color='blue', s=30, alpha=0.7, label='Class +1')
plt.scatter(X2[:, 0], X2[:, 1], color='red', s=30, alpha=0.7, label='Class -1')

# Plot class means
plt.scatter(mean1[0], mean1[1], color='blue', s=100, marker='X')
plt.annotate('$\\mu_1$', (mean1[0]+0.1, mean1[1]+0.1), fontsize=14)

plt.scatter(mean2[0], mean2[1], color='red', s=100, marker='X')
plt.annotate('$\\mu_2$', (mean2[0]+0.1, mean2[1]+0.1), fontsize=14)

# Plot the theoretical decision boundaries
plt.plot(x1_line, x2_line_lda, 'g-', linewidth=2, label='LDA Boundary')
plt.plot(x1_line, x2_line_perceptron, 'purple', linestyle='--', linewidth=2, label='Perceptron Boundary')

# Connect the means to show direction
plt.plot([mean1[0], mean2[0]], [mean1[1], mean2[1]], 'k:', alpha=0.5)
plt.annotate('Direction between means', ((mean1[0]+mean2[0])/2 + 0.2, (mean1[1]+mean2[1])/2 + 0.2), fontsize=10)

# Clean up the plot
plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('LDA vs Perceptron: Conceptual Comparison', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.xlim(-1, 3)
plt.ylim(-1, 3)

# Save the simplified comparison plot
plt.savefig(os.path.join(save_dir, "simple_lda_vs_perceptron.png"), dpi=300, bbox_inches='tight')

# Create a simple illustration of the two approaches
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left subplot: Perceptron approach
ax1.set_title('Perceptron Approach', fontsize=14)
ax1.scatter(X1[:, 0], X1[:, 1], color='blue', s=30, alpha=0.7)
ax1.scatter(X2[:, 0], X2[:, 1], color='red', s=30, alpha=0.7)
ax1.plot(x1_line, x2_line_perceptron, 'purple', linestyle='-', linewidth=2)

# Add simple iteration visualization for Perceptron
iteration_start = np.array([1, 0.5])
iterations = [
    np.array([1.2, 0.8]),
    np.array([1.5, 1.2]),
    np.array([1.0, 1.5])
]

ax1.scatter(iteration_start[0], iteration_start[1], color='purple', s=50)
ax1.annotate('Start', (iteration_start[0]+0.1, iteration_start[1]), fontsize=10)

for i, update in enumerate(iterations):
    ax1.annotate('', xy=update, xytext=iteration_start,
                arrowprops=dict(arrowstyle='->', color='purple', lw=1.5, alpha=0.6))
    iteration_start = update
    if i == len(iterations)-1:
        ax1.annotate('End', (update[0]+0.1, update[1]), fontsize=10)

ax1.annotate('Iterative Updates\nNo Distributional Assumptions', (0.2, 2.5), fontsize=11,
           bbox=dict(facecolor='white', alpha=0.7))

# Right subplot: LDA approach
ax2.set_title('LDA Approach', fontsize=14)
ax2.scatter(X1[:, 0], X1[:, 1], color='blue', s=30, alpha=0.7)
ax2.scatter(X2[:, 0], X2[:, 1], color='red', s=30, alpha=0.7)
ax2.plot(x1_line, x2_line_lda, 'green', linestyle='-', linewidth=2)

# Add ellipses to represent distributions (simplified)
from matplotlib.patches import Ellipse
ellipse1 = Ellipse(xy=mean1, width=2.5, height=2.5, angle=0, 
                  facecolor='blue', alpha=0.2)
ellipse2 = Ellipse(xy=mean2, width=2.5, height=2.5, angle=0,
                  facecolor='red', alpha=0.2)
ax2.add_patch(ellipse1)
ax2.add_patch(ellipse2)

# Plot means
ax2.scatter(mean1[0], mean1[1], color='blue', s=100, marker='X')
ax2.scatter(mean2[0], mean2[1], color='red', s=100, marker='X')

# Connect the means
ax2.plot([mean1[0], mean2[0]], [mean1[1], mean2[1]], 'k:', alpha=0.5)
ax2.annotate('$\\mu_1 - \\mu_2$', ((mean1[0]+mean2[0])/2 + 0.2, (mean1[1]+mean2[1])/2 + 0.2), fontsize=12)

# Add LDA explanation
ax2.annotate('Based on Gaussian Distributions\nand Bayes\' Rule', (0.2, 2.5), fontsize=11,
           bbox=dict(facecolor='white', alpha=0.7))

# Set limits for both subplots
for ax in [ax1, ax2]:
    ax.set_xlim(-1, 3)
    ax.set_ylim(-1, 3)
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "simple_classifier_approaches.png"), dpi=300, bbox_inches='tight')

print("\nConclusion:")
print("-----------")
print("1. The decision boundary equation for Model A (Perceptron) is: 2x₁ - x₂ + 0.5 = 0")
print("2. LDA is theoretically more appropriate when data follows Gaussian distributions with equal covariance")
print("   because it is derived using Bayes' rule under these specific assumptions.")
print("3. For the test point (1, 2), Model A predicts class +1 because the decision value is positive.")
print("4. The key difference between LDA and Perceptron is their approach to finding the decision boundary:")
print("   - LDA uses probabilistic modeling based on data distributions and Bayes' rule")
print("   - Perceptron iteratively seeks any boundary that separates the training data correctly")
print("\nVisualizations have been created to illustrate these concepts.") 