import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
import os
from scipy import stats
from scipy.stats import multivariate_normal

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_4_Quiz_9")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

print("Question 9: Linear Discriminant Analysis (LDA) Assumptions")
print("=======================================================")

# Step 1: Explain the key assumptions of LDA
print("\nStep 1: Key assumptions of LDA")
print("----------------------------")

print("Linear Discriminant Analysis (LDA) makes several important assumptions:")
print("1. The classes have multivariate Gaussian (normal) distributions")
print("2. The classes share the same covariance matrix (homoscedasticity)")
print("3. The features are not perfectly correlated (no multi-collinearity)")
print("4. The sample size is larger than the number of features")
print("\nWhen these assumptions are met, LDA provides the optimal decision boundary")
print("in the sense of minimizing the Bayes error rate.")

# Step 2: Visualize the LDA assumptions
print("\nStep 2: Visualizing LDA assumptions")
print("--------------------------------")

# Generate data to illustrate LDA assumptions
np.random.seed(42)

# Equal covariance matrices (LDA assumption)
mean1 = np.array([2, 3])  # Class 1 mean
mean2 = np.array([4, 1])  # Class 2 mean
cov = np.array([[2, 0], [0, 1]])  # Shared covariance matrix

# Generate data for class 1
n_samples1 = 200
X1 = np.random.multivariate_normal(mean1, cov, n_samples1)

# Generate data for class 2
n_samples2 = 200
X2 = np.random.multivariate_normal(mean2, cov, n_samples2)

# Plot data and equal-probability contours
plt.figure(figsize=(12, 8))
plt.scatter(X1[:, 0], X1[:, 1], c='blue', alpha=0.5, label='Class 1')
plt.scatter(X2[:, 0], X2[:, 1], c='red', alpha=0.5, label='Class 2')

# Plot class means
plt.scatter(mean1[0], mean1[1], c='blue', s=200, marker='*', edgecolor='k', label='Class 1 Mean')
plt.scatter(mean2[0], mean2[1], c='red', s=200, marker='*', edgecolor='k', label='Class 2 Mean')

# Function to create an ellipse representing equal-probability contours
def plot_ellipse(mean, cov, color, alpha=0.3, std_multiplier=2):
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * std_multiplier * np.sqrt(eigenvalues)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, 
                     facecolor=color, alpha=alpha, edgecolor='black')
    return ellipse

# Add equal-probability contours (2-sigma)
ax = plt.gca()
ax.add_patch(plot_ellipse(mean1, cov, 'blue'))
ax.add_patch(plot_ellipse(mean2, cov, 'red'))

# Add axis labels
plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('LDA Assumption: Equal Covariance Matrices', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Save the plot
plt.savefig(os.path.join(save_dir, "lda_assumptions_equal_covariance.png"), dpi=300, bbox_inches='tight')

# Step 3: Calculate the LDA projection vector w
print("\nStep 3: Calculating the LDA projection vector")
print("-----------------------------------------")

# Given data from the problem
mu1 = np.array([1, 2])  # Class 1 mean
mu2 = np.array([3, 0])  # Class 2 mean
Sigma = np.array([[2, 0], [0, 1]])  # Shared covariance matrix

# Calculate the inverse of the covariance matrix
Sigma_inv = np.linalg.inv(Sigma)

# Calculate the LDA projection vector w = Σ^(-1) * (μ1 - μ2)
w = Sigma_inv.dot(mu1 - mu2)

# Normalize w to unit length
w_norm = w / np.linalg.norm(w)

print(f"Given information:")
print(f"  Class 1 mean (μ1): [{mu1[0]}, {mu1[1]}]^T")
print(f"  Class 2 mean (μ2): [{mu2[0]}, {mu2[1]}]^T")
print(f"  Shared covariance matrix (Σ): [[{Sigma[0, 0]}, {Sigma[0, 1]}], [{Sigma[1, 0]}, {Sigma[1, 1]}]]")
print("\nStep 1: Calculate the inverse of the covariance matrix:")
print(f"  Σ^(-1) = [[{Sigma_inv[0, 0]:.4f}, {Sigma_inv[0, 1]:.4f}], [{Sigma_inv[1, 0]:.4f}, {Sigma_inv[1, 1]:.4f}]]")
print("\nStep 2: Calculate the difference between class means:")
print(f"  μ1 - μ2 = [{mu1[0] - mu2[0]}, {mu1[1] - mu2[1]}]^T")
print("\nStep 3: Calculate the projection vector w = Σ^(-1) * (μ1 - μ2):")
print(f"  w = [{w[0]:.4f}, {w[1]:.4f}]^T")
print(f"  |w| = {np.linalg.norm(w):.4f}")
print(f"  w_normalized = [{w_norm[0]:.4f}, {w_norm[1]:.4f}]^T")

# Step 4: Visualize the LDA projection
print("\nStep 4: Visualizing the LDA projection")
print("----------------------------------")

# Create a figure to visualize the LDA projection
plt.figure(figsize=(12, 8))

# Generate data points for the visualization
n_points = 100
X1 = np.random.multivariate_normal(mu1, Sigma, n_points)
X2 = np.random.multivariate_normal(mu2, Sigma, n_points)

# Plot the data points
plt.scatter(X1[:, 0], X1[:, 1], c='blue', alpha=0.5, label='Class 1')
plt.scatter(X2[:, 0], X2[:, 1], c='red', alpha=0.5, label='Class 2')

# Plot the class means
plt.scatter(mu1[0], mu1[1], c='blue', s=200, marker='*', edgecolor='k', label='Class 1 Mean')
plt.scatter(mu2[0], mu2[1], c='red', s=200, marker='*', edgecolor='k', label='Class 2 Mean')

# Add equal-probability contours
ax = plt.gca()
ax.add_patch(plot_ellipse(mu1, Sigma, 'blue'))
ax.add_patch(plot_ellipse(mu2, Sigma, 'red'))

# Calculate the midpoint between the class means
midpoint = (mu1 + mu2) / 2

# Calculate the direction perpendicular to w (for the decision boundary)
perp_w = np.array([-w[1], w[0]])
perp_w = perp_w / np.linalg.norm(perp_w)

# Draw the projection vector w starting from the midpoint
scale = 3  # Scale for vector visualization
plt.arrow(midpoint[0], midpoint[1], scale * w[0], scale * w[1], 
          head_width=0.2, head_length=0.3, fc='green', ec='green', width=0.05,
          length_includes_head=True, label='Projection Direction')

# Draw the decision boundary (perpendicular to w, passing through midpoint)
boundary_x = np.array([midpoint[0] - 5 * perp_w[0], midpoint[0] + 5 * perp_w[0]])
boundary_y = np.array([midpoint[1] - 5 * perp_w[1], midpoint[1] + 5 * perp_w[1]])
plt.plot(boundary_x, boundary_y, 'k--', linewidth=2, label='Decision Boundary')

# Draw lines from class means to the projection direction
for i, (mean, color) in enumerate(zip([mu1, mu2], ['blue', 'red'])):
    # Project the mean onto the w direction
    proj = (np.dot(mean - midpoint, w) / np.dot(w, w)) * w + midpoint
    plt.plot([mean[0], proj[0]], [mean[1], proj[1]], color=color, linestyle='--', alpha=0.7)
    plt.annotate(f'Projection of μ{i+1}', xy=(proj[0], proj[1]), 
                 xytext=(proj[0] + 0.2, proj[1] - 0.2), fontsize=10,
                 arrowprops=dict(facecolor=color, shrink=0.05))

# Add annotations
plt.annotate('w = Σ^(-1)(μ1-μ2)', xy=(midpoint[0] + scale * w[0]/2, midpoint[1] + scale * w[1]/2), 
             xytext=(midpoint[0] + 1, midpoint[1] - 1), fontsize=12,
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('LDA Projection and Decision Boundary', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.axis('equal')

# Save the figure
plt.savefig(os.path.join(save_dir, "lda_projection.png"), dpi=300, bbox_inches='tight')

# Step 5: Calculate the threshold for classification
print("\nStep 5: Calculating the threshold for classification")
print("------------------------------------------------")

# Project the class means onto the direction w
proj_mu1 = np.dot(w, mu1)
proj_mu2 = np.dot(w, mu2)

# Calculate the threshold (midpoint of projected means)
threshold = (proj_mu1 + proj_mu2) / 2

# For equal prior probabilities P(C1) = P(C2), the threshold is the midpoint
print("For the case of equal prior probabilities P(C1) = P(C2):")
print(f"  Projection of μ1 onto w: {proj_mu1:.4f}")
print(f"  Projection of μ2 onto w: {proj_mu2:.4f}")
print(f"  Threshold value: {threshold:.4f}")
print("\nThe decision rule is:")
print(f"  If w^T x > {threshold:.4f}, classify as Class 1")
print(f"  If w^T x < {threshold:.4f}, classify as Class 2")

# Step 6: Classify new data points
print("\nStep 6: Classifying new data points")
print("-------------------------------")

# Given new data points to classify
x1 = np.array([2, 1])
x2 = np.array([0, 3])

# Project the new points onto w
proj_x1 = np.dot(w, x1)
proj_x2 = np.dot(w, x2)

# Classify based on the projection and threshold
class_x1 = 1 if proj_x1 > threshold else 2
class_x2 = 1 if proj_x2 > threshold else 2

print(f"New data point x1 = [{x1[0]}, {x1[1]}]^T:")
print(f"  Projection onto w: {proj_x1:.4f}")
print(f"  Threshold: {threshold:.4f}")
print(f"  Classification: Class {class_x1}")

print(f"\nNew data point x2 = [{x2[0]}, {x2[1]}]^T:")
print(f"  Projection onto w: {proj_x2:.4f}")
print(f"  Threshold: {threshold:.4f}")
print(f"  Classification: Class {class_x2}")

# Step 7: Visualize the classification of new points
print("\nStep 7: Visualizing the classification of new points")
print("------------------------------------------------")

plt.figure(figsize=(12, 8))

# Plot the distribution of classes
plt.scatter(X1[:, 0], X1[:, 1], c='blue', alpha=0.3, label='Class 1')
plt.scatter(X2[:, 0], X2[:, 1], c='red', alpha=0.3, label='Class 2')

# Plot the class means
plt.scatter(mu1[0], mu1[1], c='blue', s=200, marker='*', edgecolor='k', label='Class 1 Mean')
plt.scatter(mu2[0], mu2[1], c='red', s=200, marker='*', edgecolor='k', label='Class 2 Mean')

# Plot the new points to classify
plt.scatter(x1[0], x1[1], c='green' if class_x1 == 1 else 'purple', s=150, marker='o', edgecolor='k', label=f'New Point x1 (Class {class_x1})')
plt.scatter(x2[0], x2[1], c='green' if class_x2 == 1 else 'purple', s=150, marker='s', edgecolor='k', label=f'New Point x2 (Class {class_x2})')

# Draw the decision boundary
plt.plot(boundary_x, boundary_y, 'k--', linewidth=2, label='Decision Boundary')

# Draw projection lines for the new points
for i, (point, name, marker) in enumerate(zip([x1, x2], ['x1', 'x2'], ['o', 's'])):
    # Project the point onto the w direction
    proj_val = np.dot(w, point)
    proj_point = (proj_val / np.dot(w, w)) * w
    
    # Find closest point on decision boundary
    boundary_point = midpoint + (np.dot(point - midpoint, perp_w) * perp_w)
    
    # Draw projection line to the boundary
    plt.plot([point[0], boundary_point[0]], [point[1], boundary_point[1]], 
             'k:', linewidth=1, alpha=0.7)
    
    # Annotate the distance to boundary
    dist = np.linalg.norm(point - boundary_point)
    sign = 1 if (proj_val > threshold) else -1
    plt.annotate(f'Distance = {sign * dist:.2f}', 
                 xy=((point[0] + boundary_point[0])/2, (point[1] + boundary_point[1])/2), 
                 xytext=((point[0] + boundary_point[0])/2 + 0.2, (point[1] + boundary_point[1])/2 + 0.2), 
                 fontsize=10, arrowprops=dict(facecolor='black', shrink=0.05, width=1))

# Add equal-probability contours
ax = plt.gca()
ax.add_patch(plot_ellipse(mu1, Sigma, 'blue', alpha=0.1))
ax.add_patch(plot_ellipse(mu2, Sigma, 'red', alpha=0.1))

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('Classification of New Points with LDA', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10, loc='upper right')
plt.axis('equal')

# Save the figure
plt.savefig(os.path.join(save_dir, "lda_classification.png"), dpi=300, bbox_inches='tight')

# Step 8: Visualize posterior probabilities with LDA
print("\nStep 8: Visualizing posterior probabilities")
print("----------------------------------------")

# Define a grid of points for visualization
x1_range = np.linspace(-2, 6, 100)
x2_range = np.linspace(-2, 5, 100)
X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)
grid_points = np.vstack([X1_grid.ravel(), X2_grid.ravel()]).T

# Calculate posterior for class 1 using Bayes' rule with equal priors
def lda_posterior(x, mu1, mu2, Sigma, prior1=0.5, prior2=0.5):
    # Multivariate Gaussian PDFs for each class
    pdf1 = multivariate_normal.pdf(x, mean=mu1, cov=Sigma)
    pdf2 = multivariate_normal.pdf(x, mean=mu2, cov=Sigma)
    
    # Apply Bayes' rule
    posterior1 = (pdf1 * prior1) / (pdf1 * prior1 + pdf2 * prior2)
    return posterior1

# Calculate posterior probabilities for grid points
posteriors = np.array([lda_posterior(x, mu1, mu2, Sigma) for x in grid_points])
posterior_grid = posteriors.reshape(X1_grid.shape)

# Create a visualization of the posterior probabilities
plt.figure(figsize=(12, 8))

# Plot the posterior probability contour
contour = plt.contourf(X1_grid, X2_grid, posterior_grid, levels=20, cmap='RdBu', alpha=0.7)
plt.colorbar(contour, label='P(Class 1 | x)')

# Add contour line for decision boundary (P(Class 1 | x) = 0.5)
plt.contour(X1_grid, X2_grid, posterior_grid, levels=[0.5], colors='k', linewidths=2, linestyles='--')

# Plot the class means
plt.scatter(mu1[0], mu1[1], c='blue', s=200, marker='*', edgecolor='k', label='Class 1 Mean')
plt.scatter(mu2[0], mu2[1], c='red', s=200, marker='*', edgecolor='k', label='Class 2 Mean')

# Plot the new points to classify
plt.scatter(x1[0], x1[1], c='green', s=150, marker='o', edgecolor='k', label=f'New Point x1 (Class {class_x1})')
plt.scatter(x2[0], x2[1], c='purple', s=150, marker='s', edgecolor='k', label=f'New Point x2 (Class {class_x2})')

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('LDA Posterior Probability P(Class 1 | x)', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

# Save the figure
plt.savefig(os.path.join(save_dir, "lda_posterior.png"), dpi=300, bbox_inches='tight')

# Step 9: Compare LDA with Perceptron
print("\nStep 9: Comparing LDA with Perceptron")
print("----------------------------------")

# Create a figure to compare LDA with Perceptron
plt.figure(figsize=(10, 8))

# Plot the distribution of classes (just a few points for clarity)
np.random.seed(42)
X1_sample = np.random.multivariate_normal(mu1, Sigma, 50)
X2_sample = np.random.multivariate_normal(mu2, Sigma, 50)

plt.scatter(X1_sample[:, 0], X1_sample[:, 1], c='blue', alpha=0.5, label='Class 1')
plt.scatter(X2_sample[:, 0], X2_sample[:, 1], c='red', alpha=0.5, label='Class 2')

# Plot the class means
plt.scatter(mu1[0], mu1[1], c='blue', s=200, marker='*', edgecolor='k', label='Class 1 Mean')
plt.scatter(mu2[0], mu2[1], c='red', s=200, marker='*', edgecolor='k', label='Class 2 Mean')

# Draw the LDA decision boundary
plt.plot(boundary_x, boundary_y, 'g--', linewidth=2, label='LDA Boundary')

# Simulate a Perceptron decision boundary (trained on the same data)
# This is simplified and just for illustration
perc_vector = mu1 - mu2  # Simplified perceptron direction
perc_midpoint = (mu1 + mu2) / 2
perc_perp = np.array([-perc_vector[1], perc_vector[0]])
perc_perp = perc_perp / np.linalg.norm(perc_perp)

perc_boundary_x = np.array([perc_midpoint[0] - 5 * perc_perp[0], perc_midpoint[0] + 5 * perc_perp[0]])
perc_boundary_y = np.array([perc_midpoint[1] - 5 * perc_perp[1], perc_midpoint[1] + 5 * perc_perp[1]])
plt.plot(perc_boundary_x, perc_boundary_y, 'r-.', linewidth=2, label='Perceptron Boundary (Simplified)')

# Add a legend explaining the difference
plt.text(0.05, 0.95, 'LDA: Optimizes statistical separability\nPerceptron: Finds any separating hyperplane', 
         transform=plt.gca().transAxes, fontsize=12, 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('Comparison: LDA vs Perceptron Decision Boundaries', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.axis('equal')

# Save the figure
plt.savefig(os.path.join(save_dir, "lda_vs_perceptron.png"), dpi=300, bbox_inches='tight')

# Step 10: Summary of findings
print("\nStep 10: Summary of findings")
print("-------------------------")

print("1. Key assumptions of LDA:")
print("   - Classes follow multivariate Gaussian distributions")
print("   - Classes share the same covariance matrix")
print("   - The covariance matrix is invertible (no perfect multicollinearity)")
print("\n2. LDA projection direction:")
print(f"   w = Σ^(-1)(μ1 - μ2) = [{w[0]:.4f}, {w[1]:.4f}]^T")
print("\n3. Classification threshold (equal priors):")
print(f"   threshold = {threshold:.4f}")
print("\n4. Classification of new points:")
print(f"   x1 = [{x1[0]}, {x1[1]}]^T is classified as Class {class_x1}")
print(f"   x2 = [{x2[0]}, {x2[1]}]^T is classified as Class {class_x2}")
print("\n5. Difference from Perceptron:")
print("   - LDA takes a probabilistic approach based on class distributions")
print("   - LDA finds the optimal boundary in terms of statistical separability")
print("   - Perceptron simply tries to find any hyperplane that separates the classes")
print("   - LDA is more robust when assumptions are met, but more constrained by assumptions") 