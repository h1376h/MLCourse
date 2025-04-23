import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from pathlib import Path
import scipy.stats as stats
from scipy.stats import multivariate_normal
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

# Create the directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_1_Quiz_34")
os.makedirs(save_dir, exist_ok=True)

# Helper functions
def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n\n{'='*80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'='*80}")

def print_substep(substep_title):
    """Print a formatted substep header."""
    print(f"\n{'-'*60}")
    print(f"{substep_title}")
    print(f"{'-'*60}")

def save_figure(fig, filename):
    """Save figure to both the PNG file and show it."""
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {filepath}")
    plt.close(fig)

# Define the data
species_A_data = np.array([[3, 1], [4, 2], [3, 2]])
species_B_data = np.array([[7, 4], [8, 3], [9, 4]])
new_flower = np.array([5, 4])

# Step 1: Calculate mean vectors
print_step_header(1, "Calculate the mean vector for each species")

# Calculate means
mean_A = np.mean(species_A_data, axis=0)
mean_B = np.mean(species_B_data, axis=0)

print(f"Mean vector for Species A: {mean_A}")
print(f"Mean vector for Species B: {mean_B}")

# Step 1.2: Calculate covariance matrices
print_substep("Calculate covariance matrices")

# Calculate covariance matrices (using the sample covariance with n-1 denominator)
cov_A = np.cov(species_A_data, rowvar=False)
cov_B = np.cov(species_B_data, rowvar=False)

print(f"Covariance matrix for Species A:\n{cov_A}")
print(f"Covariance matrix for Species B:\n{cov_B}")

# Helper function to plot confidence ellipses
def plot_confidence_ellipse(ax, mean, cov, color, alpha=0.3, n_std=2.0, label=None):
    """
    Plot an ellipse representing the covariance matrix cov centered at mean.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse onto
    mean : array-like, shape (2, )
        The center of the ellipse
    cov : array-like, shape (2, 2)
        The covariance matrix
    color : string
        The color of the ellipse
    alpha : float, optional (default=0.3)
        The transparency of the ellipse
    n_std : float, optional (default=2.)
        The number of standard deviations to determine the ellipse's size
    label : string, optional (default=None)
        The label for the ellipse
    """
    # Calculate eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    
    # Get the index of the largest eigenvalue
    largest_eigval_indx = np.argmax(eigenvals)
    largest_eigval = eigenvals[largest_eigval_indx]
    largest_eigvec = eigenvecs[:, largest_eigval_indx]
    
    # Get the smallest eigenvalue and eigenvector
    smallest_eigval_indx = np.argmin(eigenvals)
    smallest_eigval = eigenvals[smallest_eigval_indx]
    smallest_eigvec = eigenvecs[:, smallest_eigval_indx]
    
    # Calculate angle of rotation
    angle = np.arctan2(largest_eigvec[1], largest_eigvec[0])
    angle = np.degrees(angle)
    
    # Calculate width and height of the ellipse
    width, height = 2 * n_std * np.sqrt(eigenvals)
    
    # Create the ellipse
    ellipse = Ellipse(mean, width=width, height=height, angle=angle, 
                     facecolor=color, alpha=alpha, label=label)
    
    return ax.add_patch(ellipse)

# Visualize the data, means, and covariance ellipses
fig, ax = plt.subplots(figsize=(10, 6))

# Plot data points
ax.scatter(species_A_data[:, 0], species_A_data[:, 1], color='blue', label='Species A', s=100)
ax.scatter(species_B_data[:, 0], species_B_data[:, 1], color='red', label='Species B', s=100)
ax.scatter(new_flower[0], new_flower[1], color='green', marker='x', s=150, linewidth=3, label='New Flower')

# Plot means
ax.scatter(mean_A[0], mean_A[1], color='blue', marker='*', s=300, edgecolor='black', label='Mean A')
ax.scatter(mean_B[0], mean_B[1], color='red', marker='*', s=300, edgecolor='black', label='Mean B')

# Add labels and lines connecting points to means
for i, point in enumerate(species_A_data):
    ax.annotate(f'A{i+1}', (point[0], point[1]), xytext=(point[0]+0.1, point[1]+0.1))
    ax.plot([point[0], mean_A[0]], [point[1], mean_A[1]], 'b--', alpha=0.3)

for i, point in enumerate(species_B_data):
    ax.annotate(f'B{i+1}', (point[0], point[1]), xytext=(point[0]+0.1, point[1]+0.1))
    ax.plot([point[0], mean_B[0]], [point[1], mean_B[1]], 'r--', alpha=0.3)

ax.annotate('New', (new_flower[0], new_flower[1]), xytext=(new_flower[0]+0.1, new_flower[1]+0.1))

# Plot covariance ellipses (2 standard deviations)
plot_confidence_ellipse(ax, mean_A, cov_A, 'blue', alpha=0.1, n_std=2.0, label='Cov A (2σ)')
plot_confidence_ellipse(ax, mean_B, cov_B, 'red', alpha=0.1, n_std=2.0, label='Cov B (2σ)')

# Set labels and title
ax.set_xlabel('Petal Length (cm)')
ax.set_ylabel('Petal Width (cm)')
ax.set_title('Flower Petal Measurements, Mean Vectors, and Covariance Ellipses')
ax.grid(True, alpha=0.3)
ax.legend()

# Adjust axis limits
ax.set_xlim(2, 10)
ax.set_ylim(0, 5)

# Add axes at origin
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)

save_figure(fig, "step1_data_means_covariance.png")

# Step 2: Classify using Euclidean distance with equal priors
print_step_header(2, "Classify using Euclidean distance with equal priors")

# Calculate distances
dist_to_A = np.linalg.norm(new_flower - mean_A)
dist_to_B = np.linalg.norm(new_flower - mean_B)

print(f"Euclidean distance to Species A mean: {dist_to_A:.4f}")
print(f"Euclidean distance to Species B mean: {dist_to_B:.4f}")

# Determine classification
classification_equal_priors = "Species A" if dist_to_A < dist_to_B else "Species B"
print(f"Classification with equal priors: {classification_equal_priors}")

# Visualize the classification with distances
fig, ax = plt.subplots(figsize=(10, 6))

# Create grid for decision regions
xx_grid, yy_grid = np.meshgrid(np.linspace(2, 10, 100), np.linspace(0, 5, 100))
points = np.c_[xx_grid.ravel(), yy_grid.ravel()]

# Calculate distances for each point in the grid
distances_to_A = np.linalg.norm(points - mean_A, axis=1).reshape(xx_grid.shape)
distances_to_B = np.linalg.norm(points - mean_B, axis=1).reshape(xx_grid.shape)

# Decision regions (blue for Species A, red for Species B)
decision_regions = np.zeros_like(xx_grid)
decision_regions[distances_to_A <= distances_to_B] = 1  # Species A regions

# Fill areas with colors (Species A: blue, Species B: red)
ax.contourf(xx_grid, yy_grid, decision_regions, alpha=0.1, levels=[-0.5, 0.5, 1.5], colors=['red', 'blue'])

# Plot data and means
ax.scatter(species_A_data[:, 0], species_A_data[:, 1], color='blue', label='Species A', s=100)
ax.scatter(species_B_data[:, 0], species_B_data[:, 1], color='red', label='Species B', s=100)
ax.scatter(mean_A[0], mean_A[1], color='blue', marker='*', s=300, edgecolor='black', label='Mean A')
ax.scatter(mean_B[0], mean_B[1], color='red', marker='*', s=300, edgecolor='black', label='Mean B')
ax.scatter(new_flower[0], new_flower[1], color='green', marker='x', s=150, linewidth=3, label='New Flower')

# Draw lines from new point to means with distances
ax.plot([new_flower[0], mean_A[0]], [new_flower[1], mean_A[1]], 'b-', linewidth=2)
ax.plot([new_flower[0], mean_B[0]], [new_flower[1], mean_B[1]], 'r-', linewidth=2)

# Annotate distances
midpoint_A = (new_flower + mean_A) / 2
midpoint_B = (new_flower + mean_B) / 2

ax.annotate(f"d = {dist_to_A:.2f}", (midpoint_A[0], midpoint_A[1]), 
            xytext=(midpoint_A[0]-0.5, midpoint_A[1]+0.3), color='blue',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8))

ax.annotate(f"d = {dist_to_B:.2f}", (midpoint_B[0], midpoint_B[1]), 
            xytext=(midpoint_B[0]-0.5, midpoint_B[1]+0.3), color='red',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))

# Draw the decision boundary
mid_point = (mean_A + mean_B) / 2
# Direction perpendicular to the line connecting means
direction = np.array([-(mean_B[1] - mean_A[1]), mean_B[0] - mean_A[0]])
direction = direction / np.linalg.norm(direction)
# Create points for the boundary line
boundary_points = np.vstack([
    mid_point + 10 * direction,
    mid_point - 10 * direction
])
ax.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', linewidth=2)

# Add text for equal priors
ax.text(3, 4.5, "P(A) = P(B) = 0.5", 
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

# Set labels and title
ax.set_xlabel('Petal Length (cm)')
ax.set_ylabel('Petal Width (cm)')
ax.set_title('Classification with Equal Priors Using Euclidean Distance')
ax.grid(True, alpha=0.3)
ax.legend()

# Adjust axis limits
ax.set_xlim(2, 10)
ax.set_ylim(0, 5)

save_figure(fig, "step2_classification_equal_priors.png")

# Step 3: Classify with multivariate Gaussian PDF
print_step_header(3, "Classify using Multivariate Gaussian PDF with equal priors")

# Function to calculate multivariate Gaussian PDF
def multivariate_gaussian_pdf(x, mean, cov):
    """Calculate the multivariate Gaussian probability density function at x."""
    d = len(x)
    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)
    
    # Calculate the Mahalanobis distance
    diff = x - mean
    mahalanobis_sq = diff.T @ inv_cov @ diff
    
    # Calculate the PDF
    pdf = (1 / (np.sqrt((2 * np.pi) ** d * det_cov))) * np.exp(-0.5 * mahalanobis_sq)
    return pdf

# Calculate PDFs for both species
pdf_A = multivariate_gaussian_pdf(new_flower, mean_A, cov_A)
pdf_B = multivariate_gaussian_pdf(new_flower, mean_B, cov_B)

print(f"PDF value for Species A: {pdf_A:.10f}")
print(f"PDF value for Species B: {pdf_B:.10f}")
print(f"Likelihood ratio (A:B): {pdf_A/pdf_B:.10f}")

# Classify using PDF with equal priors
classification_pdf_equal = "Species A" if pdf_A > pdf_B else "Species B"
print(f"Classification with equal priors using PDF: {classification_pdf_equal}")

# Generate a grid of points for PDF visualization
x_min, x_max = 2, 10
y_min, y_max = 0, 5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
positions = np.vstack([xx.ravel(), yy.ravel()]).T

# Calculate PDFs over the grid
Z_A = np.zeros(positions.shape[0])
Z_B = np.zeros(positions.shape[0])

for i, pos in enumerate(positions):
    Z_A[i] = multivariate_gaussian_pdf(pos, mean_A, cov_A)
    Z_B[i] = multivariate_gaussian_pdf(pos, mean_B, cov_B)

# Reshape for plotting
Z_A = Z_A.reshape(xx.shape)
Z_B = Z_B.reshape(xx.shape)

# Create the decision boundary based on the PDFs
Z_decision = Z_A - Z_B
decision_boundary = np.zeros_like(Z_decision)

# Visualize the PDFs
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the contours of the PDFs
levels = np.linspace(0, max(np.max(Z_A), np.max(Z_B)), 10)
contour_A = ax.contour(xx, yy, Z_A, levels=levels, colors='blue', alpha=0.5)
contour_B = ax.contour(xx, yy, Z_B, levels=levels, colors='red', alpha=0.5)

# Plot the decision boundary
ax.contour(xx, yy, Z_decision, levels=[0], colors='black', linewidths=2, linestyles='solid')

# Create decision regions (1 for Species A, 0 for Species B)
decision_regions_gaussian = (Z_A > Z_B).astype(int)

# Add filled contours for better visualization - blue for Species A, red for Species B
ax.contourf(xx, yy, decision_regions_gaussian, alpha=0.1, levels=[-0.5, 0.5, 1.5], colors=['red', 'blue'])

# Plot data points
ax.scatter(species_A_data[:, 0], species_A_data[:, 1], color='blue', label='Species A', s=100)
ax.scatter(species_B_data[:, 0], species_B_data[:, 1], color='red', label='Species B', s=100)
ax.scatter(mean_A[0], mean_A[1], color='blue', marker='*', s=300, edgecolor='black')
ax.scatter(mean_B[0], mean_B[1], color='red', marker='*', s=300, edgecolor='black')
ax.scatter(new_flower[0], new_flower[1], color='green', marker='x', s=150, linewidth=3, label='New Flower')

# Set labels and title
ax.set_xlabel('Petal Length (cm)')
ax.set_ylabel('Petal Width (cm)')
ax.set_title('Classification with Multivariate Gaussian PDFs')
ax.grid(True, alpha=0.3)
ax.legend()

# Adjust axis limits
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

save_figure(fig, "step3_classification_pdf.png")

# Step 4: Classify with unequal prior probabilities
print_step_header(4, "Classify with unequal priors (Species B is three times more common)")

# Define priors
prior_A = 0.25  # Species B is three times more common
prior_B = 0.75

# Calculate posterior probabilities using Bayes' theorem with Euclidean distance
# P(Species|x) ∝ P(x|Species) * P(Species)
# Since P(x|Species) is based on distance, we use e^(-distance^2) as a proxy for Euclidean distance method
likelihood_A_dist = np.exp(-dist_to_A**2)
likelihood_B_dist = np.exp(-dist_to_B**2)

posterior_A_dist = likelihood_A_dist * prior_A
posterior_B_dist = likelihood_B_dist * prior_B

# Normalize posteriors for Euclidean distance method
total_dist = posterior_A_dist + posterior_B_dist
posterior_A_dist_normalized = posterior_A_dist / total_dist
posterior_B_dist_normalized = posterior_B_dist / total_dist

# Calculate posterior probabilities using Bayes' theorem with PDF
posterior_A_pdf = pdf_A * prior_A
posterior_B_pdf = pdf_B * prior_B

# Normalize posteriors for PDF method
total_pdf = posterior_A_pdf + posterior_B_pdf
posterior_A_pdf_normalized = posterior_A_pdf / total_pdf
posterior_B_pdf_normalized = posterior_B_pdf / total_pdf

print("Using Euclidean distance proxy:")
print(f"Prior for Species A: {prior_A}")
print(f"Prior for Species B: {prior_B}")
print(f"Likelihood proxy for Species A: {likelihood_A_dist:.6f}")
print(f"Likelihood proxy for Species B: {likelihood_B_dist:.6f}")
print(f"Normalized posterior for Species A: {posterior_A_dist_normalized:.6f}")
print(f"Normalized posterior for Species B: {posterior_B_dist_normalized:.6f}")

print("\nUsing Multivariate Gaussian PDF:")
print(f"Likelihood (PDF) for Species A: {pdf_A:.10f}")
print(f"Likelihood (PDF) for Species B: {pdf_B:.10f}")
print(f"Normalized posterior for Species A: {posterior_A_pdf_normalized:.6f}")
print(f"Normalized posterior for Species B: {posterior_B_pdf_normalized:.6f}")

# Determine classification for both methods
classification_dist_unequal = "Species A" if posterior_A_dist > posterior_B_dist else "Species B"
classification_pdf_unequal = "Species A" if posterior_A_pdf > posterior_B_pdf else "Species B"

print(f"Classification with unequal priors using distance: {classification_dist_unequal}")
print(f"Classification with unequal priors using PDF: {classification_pdf_unequal}")

# Create a composite visualization showing the impact of priors on PDF decision boundaries
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Equal priors plot (left)
ax1.scatter(species_A_data[:, 0], species_A_data[:, 1], color='blue', label='Species A', s=100)
ax1.scatter(species_B_data[:, 0], species_B_data[:, 1], color='red', label='Species B', s=100)
ax1.scatter(mean_A[0], mean_A[1], color='blue', marker='*', s=300, edgecolor='black')
ax1.scatter(mean_B[0], mean_B[1], color='red', marker='*', s=300, edgecolor='black')
ax1.scatter(new_flower[0], new_flower[1], color='green', marker='x', s=150, linewidth=3, label='New Flower')

# Create grid for decision regions (left plot - equal priors)
xx_grid, yy_grid = np.meshgrid(np.linspace(2, 10, 100), np.linspace(0, 5, 100))
points = np.c_[xx_grid.ravel(), yy_grid.ravel()]

# Calculate distances for each point in the grid
distances_to_A = np.linalg.norm(points - mean_A, axis=1).reshape(xx_grid.shape)
distances_to_B = np.linalg.norm(points - mean_B, axis=1).reshape(xx_grid.shape)

# Decision regions (blue for Species A, red for Species B)
decision_regions = np.zeros_like(xx_grid)
decision_regions[distances_to_A <= distances_to_B] = 1  # Species A regions

# Draw decision boundary for equal priors
mid_point = (mean_A + mean_B) / 2
direction = np.array([-(mean_B[1] - mean_A[1]), mean_B[0] - mean_A[0]])
direction = direction / np.linalg.norm(direction)
boundary_points = np.vstack([
    mid_point + 10 * direction,
    mid_point - 10 * direction
])
ax1.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', linewidth=2)

# Fill areas with colors (Species A: blue, Species B: red)
ax1.contourf(xx_grid, yy_grid, decision_regions, alpha=0.1, levels=[-0.5, 0.5, 1.5], colors=['red', 'blue'])

ax1.text(3, 4.5, "P(A) = P(B) = 0.5", 
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
ax1.set_xlabel('Petal Length (cm)')
ax1.set_ylabel('Petal Width (cm)')
ax1.set_title('Classification with Equal Priors Using Euclidean Distance')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(y_min, y_max)

# Unequal priors plot (right)
ax2.scatter(species_A_data[:, 0], species_A_data[:, 1], color='blue', label='Species A', s=100)
ax2.scatter(species_B_data[:, 0], species_B_data[:, 1], color='red', label='Species B', s=100)
ax2.scatter(mean_A[0], mean_A[1], color='blue', marker='*', s=300, edgecolor='black')
ax2.scatter(mean_B[0], mean_B[1], color='red', marker='*', s=300, edgecolor='black')
ax2.scatter(new_flower[0], new_flower[1], color='green', marker='x', s=150, linewidth=3, label='New Flower')

# Calculate shifted boundary for unequal priors
ln_prior_ratio = np.log(prior_B / prior_A)
boundary_shift = ln_prior_ratio / np.linalg.norm(mean_B - mean_A)**2 * (mean_B - mean_A)
shifted_mid_point = mid_point + boundary_shift

# Draw shifted decision boundary
shifted_boundary_points = np.vstack([
    shifted_mid_point + 10 * direction,
    shifted_mid_point - 10 * direction
])
ax2.plot(shifted_boundary_points[:, 0], shifted_boundary_points[:, 1], 'k-', linewidth=2)

# Create decision regions for unequal priors with Euclidean distance
decision_regions_unequal = np.zeros_like(xx_grid)

# Function to determine which side of the shifted boundary a point falls on
def is_species_A(point, midpoint, normal_vector):
    direction_vector = point - midpoint
    return np.dot(direction_vector, normal_vector) > 0

# Normal vector to the boundary (pointing toward Species A region)
normal = np.array([direction[1], -direction[0]])  # Rotate 90 degrees
if np.dot(normal, mean_A - shifted_mid_point) < 0:
    normal = -normal  # Ensure it points toward Species A

# Determine regions
for i in range(xx_grid.shape[0]):
    for j in range(xx_grid.shape[1]):
        point = np.array([xx_grid[i, j], yy_grid[i, j]])
        decision_regions_unequal[i, j] = 1 if is_species_A(point, shifted_mid_point, normal) else 0

# Fill areas with colors (Species A: blue, Species B: red)
ax2.contourf(xx_grid, yy_grid, decision_regions_unequal, alpha=0.1, levels=[-0.5, 0.5, 1.5], colors=['red', 'blue'])

ax2.text(3, 4.5, "P(A) = 0.25, P(B) = 0.75", 
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
ax2.set_xlabel('Petal Length (cm)')
ax2.set_ylabel('Petal Width (cm)')
ax2.set_title('Classification with Unequal Priors Using Euclidean Distance')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_xlim(x_min, x_max)
ax2.set_ylim(y_min, y_max)

plt.tight_layout()
save_figure(fig, "step4_euclidean_equal_vs_unequal_priors.png")

# Now create a separate visualization for Gaussian PDF with equal and unequal priors
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Equal priors plot (left)
ax1.scatter(species_A_data[:, 0], species_A_data[:, 1], color='blue', label='Species A', s=100)
ax1.scatter(species_B_data[:, 0], species_B_data[:, 1], color='red', label='Species B', s=100)
ax1.scatter(mean_A[0], mean_A[1], color='blue', marker='*', s=300, edgecolor='black')
ax1.scatter(mean_B[0], mean_B[1], color='red', marker='*', s=300, edgecolor='black')
ax1.scatter(new_flower[0], new_flower[1], color='green', marker='x', s=150, linewidth=3, label='New Flower')

# Plot the contours of the PDFs
ax1.contour(xx, yy, Z_A, levels=6, colors='blue', alpha=0.5)
ax1.contour(xx, yy, Z_B, levels=6, colors='red', alpha=0.5)
ax1.contour(xx, yy, Z_decision, levels=[0], colors='black', linewidths=2)

# Create decision regions (1 for Species A, 0 for Species B)
decision_regions_gaussian = (Z_A > Z_B).astype(int)
ax1.contourf(xx, yy, decision_regions_gaussian, alpha=0.1, levels=[-0.5, 0.5, 1.5], colors=['red', 'blue'])

ax1.text(3, 4.5, "P(A) = P(B) = 0.5", 
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
ax1.set_xlabel('Petal Length (cm)')
ax1.set_ylabel('Petal Width (cm)')
ax1.set_title('Classification with Equal Priors Using Gaussian PDF')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(y_min, y_max)

# Unequal priors plot (right)
ax2.scatter(species_A_data[:, 0], species_A_data[:, 1], color='blue', label='Species A', s=100)
ax2.scatter(species_B_data[:, 0], species_B_data[:, 1], color='red', label='Species B', s=100)
ax2.scatter(mean_A[0], mean_A[1], color='blue', marker='*', s=300, edgecolor='black')
ax2.scatter(mean_B[0], mean_B[1], color='red', marker='*', s=300, edgecolor='black')
ax2.scatter(new_flower[0], new_flower[1], color='green', marker='x', s=150, linewidth=3, label='New Flower')

# Plot the contours of the PDFs
ax2.contour(xx, yy, Z_A, levels=6, colors='blue', alpha=0.5)
ax2.contour(xx, yy, Z_B, levels=6, colors='red', alpha=0.5)

# Apply the unequal priors to create a new decision boundary with Gaussian PDFs
Z_decision_unequal = Z_A * prior_A - Z_B * prior_B
ax2.contour(xx, yy, Z_decision_unequal, levels=[0], colors='black', linewidths=2)

# Shade the decision regions for the unequal priors case
decision_regions_gaussian_unequal = (Z_A * prior_A > Z_B * prior_B).astype(int)
ax2.contourf(xx, yy, decision_regions_gaussian_unequal, alpha=0.1, levels=[-0.5, 0.5, 1.5], colors=['red', 'blue'])

ax2.text(3, 4.5, "P(A) = 0.25, P(B) = 0.75", 
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
ax2.set_xlabel('Petal Length (cm)')
ax2.set_ylabel('Petal Width (cm)')
ax2.set_title('Classification with Unequal Priors Using Gaussian PDF')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_xlim(x_min, x_max)
ax2.set_ylim(y_min, y_max)

plt.tight_layout()
save_figure(fig, "step4_gaussian_equal_vs_unequal_priors.png")

# Create 3D visualization of PDFs
fig = plt.figure(figsize=(15, 10))

# 3D plot for Species A
ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(xx, yy, Z_A, cmap='Blues', alpha=0.8)
ax1.scatter(species_A_data[:, 0], species_A_data[:, 1], np.zeros_like(species_A_data[:, 0]), 
           color='blue', s=50, label='Species A')
ax1.scatter(new_flower[0], new_flower[1], 0, color='green', marker='x', s=100, label='New Flower')
ax1.set_xlabel('Petal Length (cm)')
ax1.set_ylabel('Petal Width (cm)')
ax1.set_zlabel('Probability Density')
ax1.set_title('Species A PDF')

# 3D plot for Species B
ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(xx, yy, Z_B, cmap='Reds', alpha=0.8)
ax2.scatter(species_B_data[:, 0], species_B_data[:, 1], np.zeros_like(species_B_data[:, 0]), 
           color='red', s=50, label='Species B')
ax2.scatter(new_flower[0], new_flower[1], 0, color='green', marker='x', s=100, label='New Flower')
ax2.set_xlabel('Petal Length (cm)')
ax2.set_ylabel('Petal Width (cm)')
ax2.set_zlabel('Probability Density')
ax2.set_title('Species B PDF')

plt.tight_layout()
save_figure(fig, "step5_3d_pdfs.png")

# Summary
print("\nSummary:")
print(f"1. Mean vector for Species A: [{mean_A[0]}, {mean_A[1]}]")
print(f"2. Mean vector for Species B: [{mean_B[0]}, {mean_B[1]}]")
print(f"3. Covariance matrix for Species A:\n{cov_A}")
print(f"4. Covariance matrix for Species B:\n{cov_B}")
print(f"5. Distance from new flower to Species A mean: {dist_to_A:.4f}")
print(f"6. Distance from new flower to Species B mean: {dist_to_B:.4f}")
print(f"7. Multivariate Gaussian PDF for Species A: {pdf_A:.10f}")
print(f"8. Multivariate Gaussian PDF for Species B: {pdf_B:.10f}")
print(f"9. Classification with equal priors (distance): {classification_equal_priors}")
print(f"10. Classification with equal priors (PDF): {classification_pdf_equal}")
print(f"11. Classification with unequal priors (distance): {classification_dist_unequal}")
print(f"12. Classification with unequal priors (PDF): {classification_pdf_unequal}") 