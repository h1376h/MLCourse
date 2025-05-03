import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import os
from scipy.stats import multivariate_normal
import numpy.linalg as LA
import seaborn as sns

# Create directory for images
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_4_Quiz_21")
os.makedirs(save_dir, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')

print("Question 21: LDA Prediction with Different Covariance Matrices")
print("====================================================")

# Step 1: Define the dataset
print("\nStep 1: Load and visualize the dataset")
print("----------------------------------")

# Given data
data = np.array([
    # Format: A, B, Class
    [3.5, 4.0, 1],
    [2.0, 4.0, 1],
    [2.0, 6.0, 1],
    [1.5, 7.0, 1],
    [7.0, 6.5, 1],
    [2.1, 2.5, 0],
    [8.0, 4.0, 0],
    [9.1, 4.5, 0]
])

# Extract features and class labels
X = data[:, :2]
y = data[:, 2].astype(int)

# Split data by class
class0_data = X[y == 0]
class1_data = X[y == 1]

# Given statistics from the problem
mu1 = np.array([3.2, 5.5])  # Mean for Class 1
mu0 = np.array([6.4, 3.7])  # Mean for Class 0

Sigma1 = np.array([
    [5.08, 0.5],
    [0.5, 2.0]
])  # Covariance for Class 1

Sigma0 = np.array([
    [14.7, 3.9],
    [3.9, 1.08]
])  # Covariance for Class 0

# Calculate the pooled covariance matrix for standard LDA
n1 = len(class1_data)
n0 = len(class0_data)
n_total = n1 + n0
Sigma_pooled = ((n1 * Sigma1) + (n0 * Sigma0)) / n_total

# Print dataset info
print(f"Number of samples in Class 0: {n0}")
print(f"Number of samples in Class 1: {n1}")

print("\nClass 0 (negative) data points:")
for i, point in enumerate(class0_data):
    print(f"  Point {i+1}: A={point[0]}, B={point[1]}")

print("\nClass 1 (positive) data points:")
for i, point in enumerate(class1_data):
    print(f"  Point {i+1}: A={point[0]}, B={point[1]}")

print("\nClass means provided in the problem:")
print(f"  Class 0 (negative) mean: μ₀ = {mu0}")
print(f"  Class 1 (positive) mean: μ₁ = {mu1}")

print("\nCovariance matrices provided in the problem:")
print("  Class 0 (negative) covariance:")
print(f"  Σ₀ = {Sigma0[0]}")
print(f"      {Sigma0[1]}")
print("  Class 1 (positive) covariance:")
print(f"  Σ₁ = {Sigma1[0]}")
print(f"      {Sigma1[1]}")

print("\nCalculated pooled covariance for standard LDA:")
print(f"  Σₚₒₒₗₑd = {Sigma_pooled[0]}")
print(f"           {Sigma_pooled[1]}")

# Function to plot confidence ellipses
def plot_confidence_ellipse(ax, mean, cov, n_std=2.0, facecolor='none', **kwargs):
    """
    Plot an ellipse that represents a confidence region based on the covariance matrix.
    
    Parameters:
    -----------
    ax : matplotlib.axes
        The axes to plot on
    mean : array, shape (2, )
        The center of the ellipse
    cov : array, shape (2, 2)
        The covariance matrix
    n_std : float
        The number of standard deviations for the ellipse
    facecolor : str
        The color of the ellipse
    **kwargs : dict
        Additional arguments to pass to the ellipse patch
    """
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor,
                      **kwargs)

    # Calculating the standard deviation of x from
    # the square root of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transform = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean[0], mean[1])

    ellipse.set_transform(transform + ax.transData)
    return ax.add_patch(ellipse)

# Plot the dataset and the given means and covariances
fig, ax = plt.subplots(figsize=(10, 8))

# Plot data points
ax.scatter(class0_data[:, 0], class0_data[:, 1], color='red', s=100, marker='x', label='Class 0')
ax.scatter(class1_data[:, 0], class1_data[:, 1], color='blue', s=100, marker='o', label='Class 1')

# Plot means
ax.scatter(mu0[0], mu0[1], color='darkred', s=150, marker='*', label='Class 0 Mean')
ax.scatter(mu1[0], mu1[1], color='darkblue', s=150, marker='*', label='Class 1 Mean')

# Plot confidence ellipses (2 standard deviations)
plot_confidence_ellipse(ax, mu0, Sigma0, n_std=2.0, edgecolor='red', linewidth=2, linestyle='--', label='Class 0 Covariance')
plot_confidence_ellipse(ax, mu1, Sigma1, n_std=2.0, edgecolor='blue', linewidth=2, linestyle='--', label='Class 1 Covariance')

# Add labels and title
ax.set_xlabel('Feature A', fontsize=14)
ax.set_ylabel('Feature B', fontsize=14)
ax.set_title('Dataset with Class Means and Covariance Ellipses', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# Add point labels
for i, point in enumerate(class0_data):
    ax.annotate(f'({point[0]}, {point[1]})', (point[0], point[1]), 
               xytext=(5, 5), textcoords='offset points', fontsize=10)
for i, point in enumerate(class1_data):
    ax.annotate(f'({point[0]}, {point[1]})', (point[0], point[1]), 
               xytext=(5, 5), textcoords='offset points', fontsize=10)

# Save the figure
plt.savefig(os.path.join(save_dir, "dataset_visualization.png"), dpi=300, bbox_inches='tight')
plt.close()

# Step 2: Calculate discriminant functions and decision boundary
print("\nStep 2: Calculate quadratic discriminant functions")
print("----------------------------------------------")

# Prior probabilities (assuming equal priors)
pi1 = 0.5  # Prior for Class 1
pi0 = 0.5  # Prior for Class 0

# For QDA, the decision functions are quadratic
def qda_discriminant(x, mu, Sigma, pi):
    """
    Quadratic discriminant function: -0.5(x-μ)ᵀΣ⁻¹(x-μ) - 0.5ln|Σ| + ln(π)
    """
    diff = x - mu
    inv_Sigma = LA.inv(Sigma)
    det_Sigma = LA.det(Sigma)
    
    # Calculate each component
    quad_term = -0.5 * diff.T @ inv_Sigma @ diff
    log_det_term = -0.5 * np.log(det_Sigma)
    log_prior_term = np.log(pi)
    
    return quad_term + log_det_term + log_prior_term

# For LDA with pooled covariance, the decision function simplifies
def lda_discriminant(x, mu, Sigma_pooled, pi):
    """
    Linear discriminant function: -0.5(x-μ)ᵀΣ⁻¹(x-μ) + ln(π)
    Since -0.5ln|Σ| is the same for all classes with pooled covariance, it cancels out
    """
    diff = x - mu
    inv_Sigma = LA.inv(Sigma_pooled)
    
    # Calculate each component
    quad_term = -0.5 * diff.T @ inv_Sigma @ diff
    log_prior_term = np.log(pi)
    
    return quad_term + log_prior_term

# Calculate posterior probabilities for a new point using the discriminant functions
def posterior_probabilities(x_new, mu0, mu1, Sigma0, Sigma1, pi0, pi1, use_pooled=False):
    """
    Calculate posterior probabilities P(C_k|x) using discriminant functions
    """
    if use_pooled:
        Sigma = Sigma_pooled
        g0 = lda_discriminant(x_new, mu0, Sigma, pi0)
        g1 = lda_discriminant(x_new, mu1, Sigma, pi1)
    else:
        g0 = qda_discriminant(x_new, mu0, Sigma0, pi0)
        g1 = qda_discriminant(x_new, mu1, Sigma1, pi1)
    
    # Convert discriminant values to probabilities using softmax
    max_g = max(g0, g1)  # For numerical stability
    exp_g0 = np.exp(g0 - max_g)
    exp_g1 = np.exp(g1 - max_g)
    sum_exp = exp_g0 + exp_g1
    
    # Posterior probabilities
    p0 = exp_g0 / sum_exp
    p1 = exp_g1 / sum_exp
    
    return p0, p1

# The new data point to classify
x_new = np.array([4, 5])

# Calculate discriminant values for the new point
g0_qda = qda_discriminant(x_new, mu0, Sigma0, pi0)
g1_qda = qda_discriminant(x_new, mu1, Sigma1, pi1)

g0_lda = lda_discriminant(x_new, mu0, Sigma_pooled, pi0)
g1_lda = lda_discriminant(x_new, mu1, Sigma_pooled, pi1)

# Calculate posterior probabilities
p0_qda, p1_qda = posterior_probabilities(x_new, mu0, mu1, Sigma0, Sigma1, pi0, pi1)
p0_lda, p1_lda = posterior_probabilities(x_new, mu0, mu1, Sigma0, Sigma1, pi0, pi1, use_pooled=True)

# Predicted class based on discriminant values
predicted_class_qda = 1 if g1_qda > g0_qda else 0
predicted_class_lda = 1 if g1_lda > g0_lda else 0

print(f"New point to classify: x_new = ({x_new[0]}, {x_new[1]})")
print("\nQDA Discriminant Values:")
print(f"  g₀(x_new) = {g0_qda:.4f}")
print(f"  g₁(x_new) = {g1_qda:.4f}")

print("\nQDA Posterior Probabilities:")
print(f"  P(C₀|x_new) = {p0_qda:.4f}")
print(f"  P(C₁|x_new) = {p1_qda:.4f}")

print(f"\nQDA Predicted Class: Class {predicted_class_qda}")

print("\nLDA (pooled covariance) Discriminant Values:")
print(f"  g₀(x_new) = {g0_lda:.4f}")
print(f"  g₁(x_new) = {g1_lda:.4f}")

print("\nLDA Posterior Probabilities:")
print(f"  P(C₀|x_new) = {p0_lda:.4f}")
print(f"  P(C₁|x_new) = {p1_lda:.4f}")

print(f"\nLDA Predicted Class: Class {predicted_class_lda}")

# Step 3: Visualize the decision boundaries
print("\nStep 3: Visualize decision boundaries")
print("----------------------------------")

# Create a grid for visualization
x_min, x_max = 0, 10
y_min, y_max = 0, 8
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Calculate QDA scores for the grid
qda_scores = np.zeros(len(grid_points))
for i, point in enumerate(grid_points):
    g0 = qda_discriminant(point, mu0, Sigma0, pi0)
    g1 = qda_discriminant(point, mu1, Sigma1, pi1)
    qda_scores[i] = g1 - g0  # Positive for Class 1, negative for Class 0

qda_scores = qda_scores.reshape(xx.shape)

# Calculate LDA scores for the grid
lda_scores = np.zeros(len(grid_points))
for i, point in enumerate(grid_points):
    g0 = lda_discriminant(point, mu0, Sigma_pooled, pi0)
    g1 = lda_discriminant(point, mu1, Sigma_pooled, pi1)
    lda_scores[i] = g1 - g0  # Positive for Class 1, negative for Class 0

lda_scores = lda_scores.reshape(xx.shape)

# Create figure for QDA
fig, ax = plt.subplots(figsize=(10, 8))

# Plot data points
ax.scatter(class0_data[:, 0], class0_data[:, 1], color='red', s=100, marker='x', label='Class 0')
ax.scatter(class1_data[:, 0], class1_data[:, 1], color='blue', s=100, marker='o', label='Class 1')

# Plot means
ax.scatter(mu0[0], mu0[1], color='darkred', s=150, marker='*', label='Class 0 Mean')
ax.scatter(mu1[0], mu1[1], color='darkblue', s=150, marker='*', label='Class 1 Mean')

# Plot the new point
ax.scatter(x_new[0], x_new[1], color='green', s=150, marker='D', label=f'New Point ({x_new[0]}, {x_new[1]})')

# Plot QDA decision boundary
ax.contour(xx, yy, qda_scores, levels=[0], colors='black', linewidths=2, linestyles='-')
ax.contourf(xx, yy, qda_scores, levels=[-100, 0, 100], alpha=0.3, colors=['lightsalmon', 'lightblue'])

# Add confidence ellipses
plot_confidence_ellipse(ax, mu0, Sigma0, n_std=2.0, edgecolor='red', linewidth=2, linestyle='--')
plot_confidence_ellipse(ax, mu1, Sigma1, n_std=2.0, edgecolor='blue', linewidth=2, linestyle='--')

# Add labels and title
ax.set_xlabel('Feature A', fontsize=14)
ax.set_ylabel('Feature B', fontsize=14)
ax.set_title('QDA Decision Boundary with Different Covariance Matrices', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# Save the QDA figure
plt.savefig(os.path.join(save_dir, "qda_decision_boundary.png"), dpi=300, bbox_inches='tight')
plt.close()

# Create figure for LDA (pooled covariance)
fig, ax = plt.subplots(figsize=(10, 8))

# Plot data points
ax.scatter(class0_data[:, 0], class0_data[:, 1], color='red', s=100, marker='x', label='Class 0')
ax.scatter(class1_data[:, 0], class1_data[:, 1], color='blue', s=100, marker='o', label='Class 1')

# Plot means
ax.scatter(mu0[0], mu0[1], color='darkred', s=150, marker='*', label='Class 0 Mean')
ax.scatter(mu1[0], mu1[1], color='darkblue', s=150, marker='*', label='Class 1 Mean')

# Plot the new point
ax.scatter(x_new[0], x_new[1], color='green', s=150, marker='D', label=f'New Point ({x_new[0]}, {x_new[1]})')

# Plot LDA decision boundary
ax.contour(xx, yy, lda_scores, levels=[0], colors='black', linewidths=2, linestyles='-')
ax.contourf(xx, yy, lda_scores, levels=[-100, 0, 100], alpha=0.3, colors=['lightsalmon', 'lightblue'])

# Add confidence ellipse for pooled covariance
plot_confidence_ellipse(ax, mu0, Sigma_pooled, n_std=2.0, edgecolor='red', linewidth=2, linestyle='--')
plot_confidence_ellipse(ax, mu1, Sigma_pooled, n_std=2.0, edgecolor='blue', linewidth=2, linestyle='--')

# Add labels and title
ax.set_xlabel('Feature A', fontsize=14)
ax.set_ylabel('Feature B', fontsize=14)
ax.set_title('LDA Decision Boundary with Pooled Covariance Matrix', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# Save the LDA figure
plt.savefig(os.path.join(save_dir, "lda_decision_boundary.png"), dpi=300, bbox_inches='tight')
plt.close()

# Create a comparison figure
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Plot QDA on the first subplot
ax = axes[0]
ax.scatter(class0_data[:, 0], class0_data[:, 1], color='red', s=100, marker='x', label='Class 0')
ax.scatter(class1_data[:, 0], class1_data[:, 1], color='blue', s=100, marker='o', label='Class 1')
ax.scatter(mu0[0], mu0[1], color='darkred', s=150, marker='*')
ax.scatter(mu1[0], mu1[1], color='darkblue', s=150, marker='*')
ax.scatter(x_new[0], x_new[1], color='green', s=150, marker='D', label=f'New Point')
ax.contour(xx, yy, qda_scores, levels=[0], colors='black', linewidths=2, linestyles='-')
ax.contourf(xx, yy, qda_scores, levels=[-100, 0, 100], alpha=0.3, colors=['lightsalmon', 'lightblue'])
plot_confidence_ellipse(ax, mu0, Sigma0, n_std=2.0, edgecolor='red', linewidth=2, linestyle='--')
plot_confidence_ellipse(ax, mu1, Sigma1, n_std=2.0, edgecolor='blue', linewidth=2, linestyle='--')
ax.set_xlabel('Feature A', fontsize=14)
ax.set_ylabel('Feature B', fontsize=14)
ax.set_title('QDA: Different Covariance Matrices', fontsize=16)
ax.grid(True, alpha=0.3)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.legend(fontsize=12)

# Plot LDA on the second subplot
ax = axes[1]
ax.scatter(class0_data[:, 0], class0_data[:, 1], color='red', s=100, marker='x', label='Class 0')
ax.scatter(class1_data[:, 0], class1_data[:, 1], color='blue', s=100, marker='o', label='Class 1')
ax.scatter(mu0[0], mu0[1], color='darkred', s=150, marker='*')
ax.scatter(mu1[0], mu1[1], color='darkblue', s=150, marker='*')
ax.scatter(x_new[0], x_new[1], color='green', s=150, marker='D', label=f'New Point')
ax.contour(xx, yy, lda_scores, levels=[0], colors='black', linewidths=2, linestyles='-')
ax.contourf(xx, yy, lda_scores, levels=[-100, 0, 100], alpha=0.3, colors=['lightsalmon', 'lightblue'])
plot_confidence_ellipse(ax, mu0, Sigma_pooled, n_std=2.0, edgecolor='red', linewidth=2, linestyle='--')
plot_confidence_ellipse(ax, mu1, Sigma_pooled, n_std=2.0, edgecolor='blue', linewidth=2, linestyle='--')
ax.set_xlabel('Feature A', fontsize=14)
ax.set_ylabel('Feature B', fontsize=14)
ax.set_title('LDA: Pooled Covariance Matrix', fontsize=16)
ax.grid(True, alpha=0.3)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.legend(fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "comparison_decision_boundaries.png"), dpi=300, bbox_inches='tight')
plt.close()

# Step 4: Summary and Conclusions
print("\nStep 4: Summary and Conclusions")
print("---------------------------")

print("\nTask 1: Effect of Different Covariance Matrices on LDA Assumptions")
print("LDA assumes that all classes share the same covariance matrix, but in this case,")
print("we have different covariance matrices for each class. This means:")
print("1. Standard LDA's assumption of shared covariance is violated")
print("2. The correct approach is Quadratic Discriminant Analysis (QDA)")
print("3. The decision boundary becomes quadratic (curved) rather than linear")
print("4. The pooled covariance matrix approximates both class-specific matrices")

print("\nTask 2: Posterior Probabilities for New Point (4, 5)")
print(f"Using QDA (different covariance matrices):")
print(f"  P(C₀|x_new) = {p0_qda:.4f}")
print(f"  P(C₁|x_new) = {p1_qda:.4f}")

print("\nTask 3: Predicted Class for New Point (4, 5)")
print(f"Based on QDA: The new point belongs to Class {predicted_class_qda}")
print(f"Based on LDA: The new point belongs to Class {predicted_class_lda}")

print("\nTask 4: Difference in Decision Boundary with Pooled Covariance")
print("LDA with pooled covariance creates a linear decision boundary, while")
print("QDA with separate covariance matrices creates a quadratic (curved) boundary.")
print("The LDA boundary is a straight line, whereas the QDA boundary is a curve that")
print("accounts for the different shapes and orientations of the class distributions.")
print("This leads to different classification regions and potentially different predictions.")
print("For this specific dataset, the difference results in:")
print(f"- QDA (different covariances) classifies the new point as Class {predicted_class_qda}")
print(f"- LDA (pooled covariance) classifies the new point as Class {predicted_class_lda}")

print("\nImages saved to:", save_dir) 