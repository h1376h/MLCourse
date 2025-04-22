import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_1_Quiz_31")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

def print_substep(substep_title):
    """Print a formatted substep header."""
    print(f"\n{'-' * 50}")
    print(f"{substep_title}")
    print(f"{'-' * 50}")

def save_figure(fig, filename):
    """Save figure to the specified directory."""
    file_path = os.path.join(save_dir, filename)
    fig.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {file_path}")
    plt.close(fig)

# Define the data
class0_data = np.array([[1, 2], [2, 3], [3, 3]])
class1_data = np.array([[5, 2], [6, 3], [6, 4]])
new_point = np.array([4, 3])

# ==============================
# STEP 1: Calculate mean vectors and covariance matrices
# ==============================
print_step_header(1, "Calculating Mean Vectors and Covariance Matrices")

print_substep("Data Exploration")
print(f"Class 0 data points:")
for i, point in enumerate(class0_data):
    print(f"  Point {i+1}: {point}")

print(f"\nClass 1 data points:")
for i, point in enumerate(class1_data):
    print(f"  Point {i+1}: {point}")

print(f"\nPoint to classify: {new_point}")

# Calculate mean vectors
print_substep("Calculating Mean Vectors")
print("For Class 0:")
print(f"Mean vector = (1/3)([1,2] + [2,3] + [3,3])")

mean_class0 = np.mean(class0_data, axis=0)
print(f"Mean vector for Class 0: {mean_class0}")

print("\nFor Class 1:")
print(f"Mean vector = (1/3)([5,2] + [6,3] + [6,4])")

mean_class1 = np.mean(class1_data, axis=0)
print(f"Mean vector for Class 1: {mean_class1}")

# Calculate covariance matrices
print_substep("Calculating Covariance Matrices")
print("For Class 0:")
print("Covariance Matrix = (1/n)∑(x_i - μ)(x_i - μ)ᵀ")

diff0 = class0_data - mean_class0
print("\nDifferences from mean:")
for i, d in enumerate(diff0):
    print(f"  Point {i+1} - mean = {d}")

outer_products0 = [np.outer(d, d) for d in diff0]
print("\nOuter products (point-mean)·(point-mean)ᵀ:")
for i, op in enumerate(outer_products0):
    print(f"  Point {i+1}:\n{op}")

cov_class0 = np.cov(class0_data.T)
print(f"\nCovariance matrix for Class 0 (from np.cov):\n{cov_class0}")

print("\nFor Class 1:")
print("Covariance Matrix = (1/n)∑(x_i - μ)(x_i - μ)ᵀ")

diff1 = class1_data - mean_class1
print("\nDifferences from mean:")
for i, d in enumerate(diff1):
    print(f"  Point {i+1} - mean = {d}")

outer_products1 = [np.outer(d, d) for d in diff1]
print("\nOuter products (point-mean)·(point-mean)ᵀ:")
for i, op in enumerate(outer_products1):
    print(f"  Point {i+1}:\n{op}")

cov_class1 = np.cov(class1_data.T)
print(f"\nCovariance matrix for Class 1 (from np.cov):\n{cov_class1}")

# Function to create a scatter plot with ellipses
def plot_data_with_ellipses(ax, class0_data, class1_data, mean_class0, mean_class1, 
                           cov_class0, cov_class1, new_point=None, title=None):
    
    # Plot data points
    ax.scatter(class0_data[:, 0], class0_data[:, 1], color='blue', label='Class 0', s=80)
    ax.scatter(class1_data[:, 0], class1_data[:, 1], color='red', label='Class 1', s=80)
    
    # Plot mean vectors
    ax.scatter(mean_class0[0], mean_class0[1], color='blue', marker='X', s=150, 
              edgecolor='black', linewidth=1.5, label='Class 0 Mean')
    ax.scatter(mean_class1[0], mean_class1[1], color='red', marker='X', s=150, 
              edgecolor='black', linewidth=1.5, label='Class 1 Mean')
    
    # Function to create confidence ellipses for multivariate Gaussian
    def plot_ellipse(ax, mean, cov, color, alpha=0.3):
        # Find eigenvalues and eigenvectors
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        
        # Width and height are "full" widths, not radii
        width, height = 2 * np.sqrt(5.991 * eigenvals)  # 95% confidence ellipse
        
        # Angle in degrees
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        
        # Create ellipse
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, 
                         facecolor=color, alpha=alpha, edgecolor=color, linewidth=2)
        ax.add_patch(ellipse)
    
    # Add ellipses for both classes
    plot_ellipse(ax, mean_class0, cov_class0, 'blue')
    plot_ellipse(ax, mean_class1, cov_class1, 'red')
    
    # Plot new point if provided
    if new_point is not None:
        ax.scatter(new_point[0], new_point[1], color='green', marker='*', s=200, 
                  edgecolor='black', linewidth=1.5, label='New Point')
    
    # Add labels and legend
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    ax.legend()
    
    # Set limits with some padding
    x_min = min(np.min(class0_data[:, 0]), np.min(class1_data[:, 0])) - 1
    x_max = max(np.max(class0_data[:, 0]), np.max(class1_data[:, 0])) + 1
    y_min = min(np.min(class0_data[:, 1]), np.min(class1_data[:, 1])) - 1
    y_max = max(np.max(class0_data[:, 1]), np.max(class1_data[:, 1])) + 1
    
    if new_point is not None:
        x_min = min(x_min, new_point[0] - 1)
        x_max = max(x_max, new_point[0] + 1)
        y_min = min(y_min, new_point[1] - 1)
        y_max = max(y_max, new_point[1] + 1)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    return ax

# Create visualization of data and distribution parameters
print_substep("Creating Visualizations")

# Create a detailed data visualization
fig1 = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1])

# Main scatter plot
ax_main = fig1.add_subplot(gs[0, 0])
plot_data_with_ellipses(ax_main, class0_data, class1_data, mean_class0, mean_class1, 
                       cov_class0, cov_class1, new_point=new_point,
                       title='Data Points with Mean Vectors and Covariance Ellipses')

# Feature 1 distribution
ax_feat1 = fig1.add_subplot(gs[1, 0], sharex=ax_main)
sns.kdeplot(class0_data[:, 0], color='blue', fill=True, label='Class 0 - Feature 1', ax=ax_feat1, alpha=0.5)
sns.kdeplot(class1_data[:, 0], color='red', fill=True, label='Class 1 - Feature 1', ax=ax_feat1, alpha=0.5)
ax_feat1.axvline(x=new_point[0], color='green', linestyle='--', label='New Point - Feature 1')
ax_feat1.legend()
ax_feat1.set_ylabel('Density')
ax_feat1.set_xlabel('Feature 1')

# Feature 2 distribution
ax_feat2 = fig1.add_subplot(gs[0, 1], sharey=ax_main)
sns.kdeplot(y=class0_data[:, 1], color='blue', fill=True, label='Class 0 - Feature 2', ax=ax_feat2, alpha=0.5)
sns.kdeplot(y=class1_data[:, 1], color='red', fill=True, label='Class 1 - Feature 2', ax=ax_feat2, alpha=0.5)
ax_feat2.axhline(y=new_point[1], color='green', linestyle='--', label='New Point - Feature 2')
ax_feat2.legend()
ax_feat2.set_xlabel('Density')
ax_feat2.set_ylabel('Feature 2')

# Remove the corner plot
fig1.delaxes(fig1.add_subplot(gs[1, 1]))

plt.tight_layout()
save_figure(fig1, "step1_data_visualization.png")

# Create a simple version of the scatter plot
fig1b, ax1b = plt.subplots(figsize=(10, 8))
plot_data_with_ellipses(ax1b, class0_data, class1_data, mean_class0, mean_class1, 
                       cov_class0, cov_class1, new_point=new_point,
                       title='Data Points with Mean Vectors and Covariance Ellipses')
save_figure(fig1b, "step1b_data_scatter.png")

# Plot the covariance matrices as heatmaps
fig1c, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cov_class0, annot=True, fmt=".3f", cmap="Blues", square=True, ax=ax1)
ax1.set_title("Covariance Matrix - Class 0")
ax1.set_xticklabels(["Feature 1", "Feature 2"])
ax1.set_yticklabels(["Feature 1", "Feature 2"])

sns.heatmap(cov_class1, annot=True, fmt=".3f", cmap="Reds", square=True, ax=ax2)
ax2.set_title("Covariance Matrix - Class 1")
ax2.set_xticklabels(["Feature 1", "Feature 2"])
ax2.set_yticklabels(["Feature 1", "Feature 2"])

plt.tight_layout()
save_figure(fig1c, "step1c_covariance_matrices.png")

# ==============================
# STEP 2: Multivariate Gaussian PDF
# ==============================
print_step_header(2, "Multivariate Gaussian Probability Density Function")

# Define the multivariate Gaussian PDF function
def multivariate_gaussian_pdf(x, mean, cov):
    """Calculate the PDF value for a multivariate Gaussian distribution."""
    n = len(mean)
    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)
    diff = x - mean
    
    # Multivariate Gaussian formula
    exponent = -0.5 * diff.T @ inv_cov @ diff
    normalizer = 1 / ((2 * np.pi) ** (n/2) * np.sqrt(det_cov))
    pdf_value = normalizer * np.exp(exponent)
    
    print(f"  - n (dimensions): {n}")
    print(f"  - |Σ| (determinant): {det_cov:.6f}")
    print(f"  - x - μ (difference): {diff}")
    print(f"  - (x - μ)ᵀ Σ⁻¹ (x - μ) (quadratic form): {diff.T @ inv_cov @ diff:.6f}")
    print(f"  - exp(-0.5 * quadratic form): {np.exp(exponent):.6f}")
    print(f"  - Normalizer: {normalizer:.6f}")
    print(f"  - Final PDF value: {normalizer * np.exp(exponent):.8f}")
    
    return pdf_value

print_substep("The Multivariate Gaussian Formula")
print("Multivariate Gaussian PDF formula:")
print("f(x|μ,Σ) = (1/((2π)^(d/2) |Σ|^(1/2))) * exp(-0.5 * (x-μ)^T Σ^(-1) (x-μ))")
print("\nwhere:")
print("x is the feature vector")
print("μ is the mean vector")
print("Σ is the covariance matrix")
print("d is the dimension (2 in our case)")
print("|Σ| is the determinant of the covariance matrix")
print("Σ^(-1) is the inverse of the covariance matrix")

# Calculate determinants for normalization
print_substep("Calculating Determinants and Inverse Matrices")
det_class0 = np.linalg.det(cov_class0)
det_class1 = np.linalg.det(cov_class1)

print(f"Determinant for Class 0 covariance: |Σ₀| = {det_class0:.6f}")
print(f"Determinant for Class 1 covariance: |Σ₁| = {det_class1:.6f}")

# Calculate inverse matrices
inv_cov_class0 = np.linalg.inv(cov_class0)
inv_cov_class1 = np.linalg.inv(cov_class1)

print(f"\nInverse covariance matrix for Class 0: Σ₀⁻¹ =\n{inv_cov_class0}")
print(f"\nInverse covariance matrix for Class 1: Σ₁⁻¹ =\n{inv_cov_class1}")

# Print out the specific PDF expressions for both classes
print_substep("Deriving PDF Expressions")
norm_const_class0 = 1/((2*np.pi)**(1) * np.sqrt(det_class0))
norm_const_class1 = 1/((2*np.pi)**(1) * np.sqrt(det_class1))

print(f"PDF for Class 0:")
print(f"f(x|class 0) = {norm_const_class0:.6f} * exp(-0.5 * (x-μ₀)ᵀ Σ₀⁻¹ (x-μ₀))")
print("\nwhere:")
print(f"μ₀ = {mean_class0}")
print(f"Σ₀⁻¹ = \n{inv_cov_class0}")

print(f"\nPDF for Class 1:")
print(f"f(x|class 1) = {norm_const_class1:.6f} * exp(-0.5 * (x-μ₁)ᵀ Σ₁⁻¹ (x-μ₁))")
print("\nwhere:")
print(f"μ₁ = {mean_class1}")
print(f"Σ₁⁻¹ = \n{inv_cov_class1}")

# Calculate PDF values for the new point
print_substep("Evaluating PDFs for the New Point")
print(f"New point to classify: {new_point}")

print("\nCalculating P(x|class 0):")
pdf_new_class0 = multivariate_gaussian_pdf(new_point, mean_class0, cov_class0)
print(f"\nP(x|class 0) = {pdf_new_class0:.8f}")

print("\nCalculating P(x|class 1):")
pdf_new_class1 = multivariate_gaussian_pdf(new_point, mean_class1, cov_class1)
print(f"\nP(x|class 1) = {pdf_new_class1:.8f}")

print(f"\nLog likelihood ratio: ln[P(x|class 1)/P(x|class 0)] = {np.log(pdf_new_class1/pdf_new_class0):.6f}")

# Visualize the PDFs
print_substep("Visualizing the PDFs")
# Create a grid of points
x_min, x_max = 0, 7
y_min, y_max = 1, 5
x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
pos = np.dstack((x_grid, y_grid))

# Calculate PDFs for each class
pdf_class0 = multivariate_normal.pdf(pos, mean=mean_class0, cov=cov_class0)
pdf_class1 = multivariate_normal.pdf(pos, mean=mean_class1, cov=cov_class1)

# Plot the PDFs - Separate visualizations
fig2a, ax1 = plt.subplots(figsize=(8, 6))
contour0 = ax1.contourf(x_grid, y_grid, pdf_class0, levels=20, cmap='Blues')
ax1.scatter(class0_data[:, 0], class0_data[:, 1], color='blue', s=80, edgecolor='black')
ax1.scatter(mean_class0[0], mean_class0[1], color='blue', marker='X', s=150, edgecolor='black')
ax1.scatter(new_point[0], new_point[1], color='green', marker='*', s=200, edgecolor='black')
ax1.set_title('PDF for Class 0', fontsize=14)
ax1.set_xlabel('Feature 1', fontsize=12)
ax1.set_ylabel('Feature 2', fontsize=12)
plt.colorbar(contour0, ax=ax1, label='Probability Density')
save_figure(fig2a, "step2a_pdf_class0.png")

fig2b, ax2 = plt.subplots(figsize=(8, 6))
contour1 = ax2.contourf(x_grid, y_grid, pdf_class1, levels=20, cmap='Reds')
ax2.scatter(class1_data[:, 0], class1_data[:, 1], color='red', s=80, edgecolor='black')
ax2.scatter(mean_class1[0], mean_class1[1], color='red', marker='X', s=150, edgecolor='black')
ax2.scatter(new_point[0], new_point[1], color='green', marker='*', s=200, edgecolor='black')
ax2.set_title('PDF for Class 1', fontsize=14)
ax2.set_xlabel('Feature 1', fontsize=12)
ax2.set_ylabel('Feature 2', fontsize=12)
plt.colorbar(contour1, ax=ax2, label='Probability Density')
save_figure(fig2b, "step2b_pdf_class1.png")

# Combined view
fig2c, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Class 0 PDF
c0 = ax1.contourf(x_grid, y_grid, pdf_class0, levels=20, cmap='Blues')
ax1.scatter(class0_data[:, 0], class0_data[:, 1], color='blue', s=80, edgecolor='black')
ax1.scatter(mean_class0[0], mean_class0[1], color='blue', marker='X', s=150, edgecolor='black')
ax1.scatter(new_point[0], new_point[1], color='green', marker='*', s=200, edgecolor='black')
ax1.set_title('PDF for Class 0', fontsize=14)
ax1.set_xlabel('Feature 1', fontsize=12)
ax1.set_ylabel('Feature 2', fontsize=12)
plt.colorbar(c0, ax=ax1, label='Probability Density')

# Class 1 PDF
c1 = ax2.contourf(x_grid, y_grid, pdf_class1, levels=20, cmap='Reds')
ax2.scatter(class1_data[:, 0], class1_data[:, 1], color='red', s=80, edgecolor='black')
ax2.scatter(mean_class1[0], mean_class1[1], color='red', marker='X', s=150, edgecolor='black')
ax2.scatter(new_point[0], new_point[1], color='green', marker='*', s=200, edgecolor='black')
ax2.set_title('PDF for Class 1', fontsize=14)
ax2.set_xlabel('Feature 1', fontsize=12)
ax2.set_ylabel('Feature 2', fontsize=12)
plt.colorbar(c1, ax=ax2, label='Probability Density')

plt.tight_layout()
save_figure(fig2c, "step2c_gaussian_pdfs.png")

# Visualize 3D PDFs
fig2d = plt.figure(figsize=(16, 7))

# 3D plot for Class 0
ax1 = fig2d.add_subplot(1, 2, 1, projection='3d')
surf1 = ax1.plot_surface(x_grid, y_grid, pdf_class0, cmap='Blues', alpha=0.8)
ax1.set_title('PDF for Class 0', fontsize=14)
ax1.set_xlabel('Feature 1', fontsize=12)
ax1.set_ylabel('Feature 2', fontsize=12)
ax1.set_zlabel('Probability Density', fontsize=12)
ax1.view_init(elev=30, azim=45)  # Set viewing angle

# Scatter points on the 3D surface
zs_0 = np.zeros(len(class0_data))
ax1.scatter(class0_data[:, 0], class0_data[:, 1], zs_0, color='blue', s=50, edgecolor='black')
ax1.scatter(mean_class0[0], mean_class0[1], 0, color='blue', marker='X', s=100, edgecolor='black')
ax1.scatter(new_point[0], new_point[1], 0, color='green', marker='*', s=150, edgecolor='black')

# 3D plot for Class 1
ax2 = fig2d.add_subplot(1, 2, 2, projection='3d')
surf2 = ax2.plot_surface(x_grid, y_grid, pdf_class1, cmap='Reds', alpha=0.8)
ax2.set_title('PDF for Class 1', fontsize=14)
ax2.set_xlabel('Feature 1', fontsize=12)
ax2.set_ylabel('Feature 2', fontsize=12)
ax2.set_zlabel('Probability Density', fontsize=12)
ax2.view_init(elev=30, azim=45)  # Set viewing angle

# Scatter points on the 3D surface
zs_1 = np.zeros(len(class1_data))
ax2.scatter(class1_data[:, 0], class1_data[:, 1], zs_1, color='red', s=50, edgecolor='black')
ax2.scatter(mean_class1[0], mean_class1[1], 0, color='red', marker='X', s=100, edgecolor='black')
ax2.scatter(new_point[0], new_point[1], 0, color='green', marker='*', s=150, edgecolor='black')

plt.tight_layout()
save_figure(fig2d, "step2d_3d_gaussian_pdfs.png")

# ==============================
# STEP 3: Bayes' Theorem Classification
# ==============================
print_step_header(3, "Bayes' Theorem Classification with Equal Priors")

print_substep("Introduction to Bayes' Theorem")
print("Bayes' theorem states:")
print("P(class|x) = [P(x|class) × P(class)] / P(x)")
print("\nwhere:")
print("- P(class|x) is the posterior probability")
print("- P(x|class) is the class-conditional density (likelihood)")
print("- P(class) is the prior probability")
print("- P(x) is the evidence (total probability)")

# Define prior probabilities
prior_class0 = 0.5
prior_class1 = 0.5

print_substep("Setting Prior Probabilities")
print(f"Prior probability for Class 0: P(class 0) = {prior_class0}")
print(f"Prior probability for Class 1: P(class 1) = {prior_class1}")

print_substep("Calculating Class-Conditional Densities (Likelihoods)")
print(f"New point to classify: {new_point}")
print(f"\nClass-conditional density for Class 0: P(x|class 0) = {pdf_new_class0:.8f}")
print(f"Class-conditional density for Class 1: P(x|class 1) = {pdf_new_class1:.8f}")
print(f"\nLikelihood ratio: P(x|class 0) / P(x|class 1) = {pdf_new_class0 / pdf_new_class1:.8f}")

# Calculate the evidence (total probability)
print_substep("Calculating the Evidence (Total Probability)")
print("P(x) = P(x|class 0)×P(class 0) + P(x|class 1)×P(class 1)")
print(f"P(x) = {pdf_new_class0:.8f}×{prior_class0} + {pdf_new_class1:.8f}×{prior_class1}")

evidence = pdf_new_class0 * prior_class0 + pdf_new_class1 * prior_class1
print(f"P(x) = {pdf_new_class0 * prior_class0:.8f} + {pdf_new_class1 * prior_class1:.8f} = {evidence:.8f}")

# Apply Bayes' theorem
print_substep("Applying Bayes' Theorem")
print("For Class 0:")
print(f"P(class 0|x) = [P(x|class 0)×P(class 0)] / P(x)")
print(f"P(class 0|x) = [{pdf_new_class0:.8f}×{prior_class0}] / {evidence:.8f}")

posterior_class0 = (pdf_new_class0 * prior_class0) / evidence
print(f"P(class 0|x) = {pdf_new_class0 * prior_class0:.8f} / {evidence:.8f} = {posterior_class0:.8f}")

print("\nFor Class 1:")
print(f"P(class 1|x) = [P(x|class 1)×P(class 1)] / P(x)")
print(f"P(class 1|x) = [{pdf_new_class1:.8f}×{prior_class1}] / {evidence:.8f}")

posterior_class1 = (pdf_new_class1 * prior_class1) / evidence
print(f"P(class 1|x) = {pdf_new_class1 * prior_class1:.8f} / {evidence:.8f} = {posterior_class1:.8f}")

# Verify that posteriors sum to 1
print(f"\nSum of posterior probabilities: {posterior_class0 + posterior_class1:.8f}")

# Make classification decision
print_substep("Making the Classification Decision")
print(f"P(class 0|x) = {posterior_class0:.8f}")
print(f"P(class 1|x) = {posterior_class1:.8f}")

if posterior_class0 > posterior_class1:
    decision = "Class 0"
    print(f"\nSince P(class 0|x) > P(class 1|x), classify as {decision}")
else:
    decision = "Class 1"
    print(f"\nSince P(class 1|x) > P(class 0|x), classify as {decision}")

# Calculate log likelihood ratio for visualization
print_substep("Visualizing the Decision Boundary")
print("The decision boundary is determined by the log likelihood ratio:")
print("ln[P(x|class 1)/P(x|class 0)]")
print("\nFor equal priors, the decision boundary is where:")
print("ln[P(x|class 1)/P(x|class 0)] = 0")

log_likelihood_ratio = np.log(pdf_class1 / pdf_class0)

# Create visualization of the decision boundary
fig3a, ax3a = plt.subplots(figsize=(10, 8))

# Plot the data points
ax3a.scatter(class0_data[:, 0], class0_data[:, 1], color='blue', marker='o', s=100, 
          edgecolor='black', linewidth=1.5, label='Class 0')
ax3a.scatter(class1_data[:, 0], class1_data[:, 1], color='red', marker='s', s=100, 
          edgecolor='black', linewidth=1.5, label='Class 1')

# Plot mean vectors
ax3a.scatter(mean_class0[0], mean_class0[1], color='blue', marker='X', s=200, 
          edgecolor='black', linewidth=2, label='Class 0 Mean')
ax3a.scatter(mean_class1[0], mean_class1[1], color='red', marker='X', s=200, 
          edgecolor='black', linewidth=2, label='Class 1 Mean')

# Plot new point
ax3a.scatter(new_point[0], new_point[1], color='green', marker='*', s=300, 
          edgecolor='black', linewidth=2, label=f'New Point ({decision})')

# Plot the decision boundary
ax3a.contour(x_grid, y_grid, log_likelihood_ratio, levels=[0], colors='k', linewidths=2)

# Add labels and legend
ax3a.set_xlabel('Feature 1', fontsize=14)
ax3a.set_ylabel('Feature 2', fontsize=14)
ax3a.set_title('Decision Boundary with Equal Priors', fontsize=16)
ax3a.legend(fontsize=12)

ax3a.set_xlim(x_min, x_max)
ax3a.set_ylim(y_min, y_max)
save_figure(fig3a, "step3a_decision_boundary.png")

# Create full visualization with log likelihood ratio
fig3b, ax3b = plt.subplots(figsize=(12, 10))

# Plot the log likelihood ratio and decision boundary
contour = ax3b.contourf(x_grid, y_grid, log_likelihood_ratio, cmap='coolwarm', alpha=0.5,
                      levels=np.linspace(-5, 5, 21))
plt.colorbar(contour, ax=ax3b, label='Log Likelihood Ratio (ln[P(x|class 1) / P(x|class 0)])')

# Add the decision boundary
ax3b.contour(x_grid, y_grid, log_likelihood_ratio, levels=[0], colors='k', linewidths=2)

# Plot data points with different markers and colors
ax3b.scatter(class0_data[:, 0], class0_data[:, 1], color='blue', marker='o', s=100, 
           edgecolor='black', linewidth=1.5, label='Class 0')
ax3b.scatter(class1_data[:, 0], class1_data[:, 1], color='red', marker='s', s=100, 
           edgecolor='black', linewidth=1.5, label='Class 1')

# Plot mean vectors
ax3b.scatter(mean_class0[0], mean_class0[1], color='blue', marker='X', s=200, 
           edgecolor='black', linewidth=2, label='Class 0 Mean')
ax3b.scatter(mean_class1[0], mean_class1[1], color='red', marker='X', s=200, 
           edgecolor='black', linewidth=2, label='Class 1 Mean')

# Plot new point
ax3b.scatter(new_point[0], new_point[1], color='green', marker='*', s=300, 
           edgecolor='black', linewidth=2, label=f'New Point ({decision})')

# Add labels and legend
ax3b.set_xlabel('Feature 1', fontsize=14)
ax3b.set_ylabel('Feature 2', fontsize=14)
ax3b.set_title('Bayesian Classification with Equal Priors', fontsize=16)
ax3b.legend(fontsize=12)

ax3b.set_xlim(x_min, x_max)
ax3b.set_ylim(y_min, y_max)

save_figure(fig3b, "step3b_equal_priors_classification.png")

# ==============================
# STEP 4: Classification with Different Priors
# ==============================
print_step_header(4, "Classification with Different Priors")

print_substep("Changing the Prior Probabilities")
# Define new prior probabilities
new_prior_class0 = 0.8
new_prior_class1 = 0.2

print(f"New prior probability for Class 0: P(class 0) = {new_prior_class0}")
print(f"New prior probability for Class 1: P(class 1) = {new_prior_class1}")
print(f"Prior ratio: P(class 0) / P(class 1) = {new_prior_class0 / new_prior_class1:.2f}")

# Recalculate with new priors
print_substep("Recalculating Evidence and Posteriors")
print("P(x) = P(x|class 0)×P(class 0) + P(x|class 1)×P(class 1)")
print(f"P(x) = {pdf_new_class0:.8f}×{new_prior_class0} + {pdf_new_class1:.8f}×{new_prior_class1}")

new_evidence = pdf_new_class0 * new_prior_class0 + pdf_new_class1 * new_prior_class1
print(f"P(x) = {pdf_new_class0 * new_prior_class0:.8f} + {pdf_new_class1 * new_prior_class1:.8f} = {new_evidence:.8f}")

print("\nFor Class 0:")
print(f"P(class 0|x) = [P(x|class 0)×P(class 0)] / P(x)")
print(f"P(class 0|x) = [{pdf_new_class0:.8f}×{new_prior_class0}] / {new_evidence:.8f}")

new_posterior_class0 = (pdf_new_class0 * new_prior_class0) / new_evidence
print(f"P(class 0|x) = {pdf_new_class0 * new_prior_class0:.8f} / {new_evidence:.8f} = {new_posterior_class0:.8f}")

print("\nFor Class 1:")
print(f"P(class 1|x) = [P(x|class 1)×P(class 1)] / P(x)")
print(f"P(class 1|x) = [{pdf_new_class1:.8f}×{new_prior_class1}] / {new_evidence:.8f}")

new_posterior_class1 = (pdf_new_class1 * new_prior_class1) / new_evidence
print(f"P(class 1|x) = {pdf_new_class1 * new_prior_class1:.8f} / {new_evidence:.8f} = {new_posterior_class1:.8f}")

# Verify that new posteriors sum to 1
print(f"\nSum of new posterior probabilities: {new_posterior_class0 + new_posterior_class1:.8f}")

# Make new classification decision
print_substep("Making the New Classification Decision")
print(f"P(class 0|x) = {new_posterior_class0:.8f}")
print(f"P(class 1|x) = {new_posterior_class1:.8f}")

if new_posterior_class0 > new_posterior_class1:
    new_decision = "Class 0"
    print(f"\nSince P(class 0|x) > P(class 1|x), classify as {new_decision}")
else:
    new_decision = "Class 1"
    print(f"\nSince P(class 1|x) > P(class 0|x), classify as {new_decision}")

# Compare with previous decision
print_substep("Comparing Decisions")
print(f"Decision with equal priors: {decision}")
print(f"Decision with unequal priors: {new_decision}")

if decision == new_decision:
    print("\nThe classification decision remains the same.")
    print("However, the posterior probability for Class 0 has increased from " +
         f"{posterior_class0:.6f} to {new_posterior_class0:.6f}.")
else:
    print(f"\nThe classification decision changed from {decision} to {new_decision}.")
    print("This shows how prior probabilities can affect the classification decision.")

# Compute the adjusted decision boundary based on the new priors
print_substep("Adjusting the Decision Boundary")
print("For equal priors, the decision boundary is where:")
print("ln[P(x|class 1)/P(x|class 0)] = 0")

print("\nWith unequal priors, the decision boundary shifts to where:")
print("ln[P(x|class 1)/P(x|class 0)] = ln[P(class 0)/P(class 1)]")

log_prior_ratio = np.log(new_prior_class0 / new_prior_class1)
print(f"\nLog prior ratio: ln(P(class 0)/P(class 1)) = ln({new_prior_class0}/{new_prior_class1}) = {log_prior_ratio:.6f}")
print(f"The decision boundary shifts to where ln[P(x|class 1)/P(x|class 0)] = {log_prior_ratio:.6f}")

# Create visualization of the new decision boundary
print_substep("Visualizing the Adjusted Decision Boundary")

# Simple visualization with just the boundaries
fig4a, ax4a = plt.subplots(figsize=(10, 8))

# Plot data points
ax4a.scatter(class0_data[:, 0], class0_data[:, 1], color='blue', marker='o', s=100, 
          edgecolor='black', linewidth=1.5, label='Class 0')
ax4a.scatter(class1_data[:, 0], class1_data[:, 1], color='red', marker='s', s=100, 
          edgecolor='black', linewidth=1.5, label='Class 1')

# Plot mean vectors
ax4a.scatter(mean_class0[0], mean_class0[1], color='blue', marker='X', s=200, 
          edgecolor='black', linewidth=2, label='Class 0 Mean')
ax4a.scatter(mean_class1[0], mean_class1[1], color='red', marker='X', s=200, 
          edgecolor='black', linewidth=2, label='Class 1 Mean')

# Plot new point
ax4a.scatter(new_point[0], new_point[1], color='green', marker='*', s=300, 
          edgecolor='black', linewidth=2, label=f'New Point ({new_decision})')

# Add both decision boundaries
ax4a.contour(x_grid, y_grid, log_likelihood_ratio, levels=[0], colors='k', linewidths=2, 
          linestyles='solid', label='Equal Priors Boundary')
ax4a.contour(x_grid, y_grid, log_likelihood_ratio, levels=[log_prior_ratio], colors='k', 
          linewidths=2, linestyles='dashed', label='Unequal Priors Boundary')

# Add legend items for boundaries
ax4a.plot([], [], color='k', linewidth=2, linestyle='solid', label='Equal Priors Boundary')
ax4a.plot([], [], color='k', linewidth=2, linestyle='dashed', label='Unequal Priors Boundary')

# Add labels and legend
ax4a.set_xlabel('Feature 1', fontsize=14)
ax4a.set_ylabel('Feature 2', fontsize=14)
ax4a.set_title('Decision Boundaries with Different Priors', fontsize=16)
ax4a.legend(fontsize=12)

ax4a.set_xlim(x_min, x_max)
ax4a.set_ylim(y_min, y_max)
save_figure(fig4a, "step4a_boundaries_comparison.png")

# Full visualization with log likelihood ratio
fig4b, ax4b = plt.subplots(figsize=(12, 10))

# Plot the log likelihood ratio and decision boundary
contour = ax4b.contourf(x_grid, y_grid, log_likelihood_ratio, cmap='coolwarm', alpha=0.5,
                      levels=np.linspace(-5, 5, 21))
plt.colorbar(contour, ax=ax4b, label='Log Likelihood Ratio (ln[P(x|class 1)/P(x|class 0)])')

# Add both decision boundaries
ax4b.contour(x_grid, y_grid, log_likelihood_ratio, levels=[0], colors='k', linewidths=2, 
          linestyles='solid', label='Equal Priors Boundary')
ax4b.contour(x_grid, y_grid, log_likelihood_ratio, levels=[log_prior_ratio], colors='k', 
          linewidths=2, linestyles='dashed', label='Unequal Priors Boundary')

# Add legend items for boundaries
ax4b.plot([], [], color='k', linewidth=2, linestyle='solid', label='Equal Priors Boundary')
ax4b.plot([], [], color='k', linewidth=2, linestyle='dashed', label='Unequal Priors Boundary')

# Plot data points with different markers and colors
ax4b.scatter(class0_data[:, 0], class0_data[:, 1], color='blue', marker='o', s=100, 
           edgecolor='black', linewidth=1.5, label='Class 0')
ax4b.scatter(class1_data[:, 0], class1_data[:, 1], color='red', marker='s', s=100, 
           edgecolor='black', linewidth=1.5, label='Class 1')

# Plot mean vectors
ax4b.scatter(mean_class0[0], mean_class0[1], color='blue', marker='X', s=200, 
           edgecolor='black', linewidth=2, label='Class 0 Mean')
ax4b.scatter(mean_class1[0], mean_class1[1], color='red', marker='X', s=200, 
           edgecolor='black', linewidth=2, label='Class 1 Mean')

# Plot new point
ax4b.scatter(new_point[0], new_point[1], color='green', marker='*', s=300, 
           edgecolor='black', linewidth=2, label=f'New Point ({new_decision})')

# Add labels and legend
ax4b.set_xlabel('Feature 1', fontsize=14)
ax4b.set_ylabel('Feature 2', fontsize=14)
ax4b.set_title('Bayesian Classification with Different Priors', fontsize=16)
ax4b.legend(fontsize=12)

ax4b.set_xlim(x_min, x_max)
ax4b.set_ylim(y_min, y_max)

save_figure(fig4b, "step4b_unequal_priors_classification.png")

# ==============================
# STEP 5: Compare both classification approaches
# ==============================
print_step_header(5, "Comparing Equal vs. Unequal Priors")

print_substep("Summary of Classification Results")
print(f"Equal priors classification result: {decision}")
print(f"   - P(class 0) = {prior_class0}, P(class 1) = {prior_class1}")
print(f"   - P(class 0|x) = {posterior_class0:.6f}")
print(f"   - P(class 1|x) = {posterior_class1:.6f}")

print(f"\nUnequal priors classification result: {new_decision}")
print(f"   - P(class 0) = {new_prior_class0}, P(class 1) = {new_prior_class1}")
print(f"   - P(class 0|x) = {new_posterior_class0:.6f}")
print(f"   - P(class 1|x) = {new_posterior_class1:.6f}")

print_substep("Effect of Changing Priors")
if decision != new_decision:
    print(f"   - The classification decision changed from {decision} to {new_decision}.")
    print(f"   - This demonstrates the significant impact of prior probabilities on the decision.")
else:
    print(f"   - The classification decision remains {decision}.")
    print(f"   - However, the posterior probabilities have changed:")
    print(f"     * P(class 0|x) increased from {posterior_class0:.6f} to {new_posterior_class0:.6f}")
    print(f"     * P(class 1|x) decreased from {posterior_class1:.6f} to {new_posterior_class1:.6f}")

print_substep("Key Insights")
print("1. When prior probabilities are equal, the decision boundary is determined solely by the likelihood ratio.")
print("   - Points are classified based only on their relative likelihoods under each class model.")
print("\n2. When prior probabilities favor one class, the decision boundary shifts to make it easier to classify points as the favored class.")
print(f"   - In this case, increasing P(class 0) to {new_prior_class0} shifted the boundary toward class 1.")
print("   - This means we need stronger evidence (higher likelihood) to classify a point as class 1.")
print("\n3. The mathematical relationship between the decision boundaries:")
print("   - Equal priors: Classify as Class 1 if ln[P(x|class 1)/P(x|class 0)] > 0")
print(f"   - Unequal priors: Classify as Class 1 if ln[P(x|class 1)/P(x|class 0)] > {log_prior_ratio:.4f}")

# Create a side-by-side comparison plot
print_substep("Creating Visual Comparisons")
fig5a, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Function to set up a comparison plot
def setup_comparison_plot(ax, class0_data, class1_data, mean_class0, mean_class1, 
                          new_point, log_likelihood_ratio, boundary_level, title, decision):
    ax.contourf(x_grid, y_grid, log_likelihood_ratio, cmap='coolwarm', alpha=0.5,
               levels=np.linspace(-5, 5, 21))
    
    # Add decision boundary
    ax.contour(x_grid, y_grid, log_likelihood_ratio, levels=[boundary_level], 
              colors='k', linewidths=2)
    
    # Plot data points
    ax.scatter(class0_data[:, 0], class0_data[:, 1], color='blue', marker='o', s=80, 
              edgecolor='black', linewidth=1.5, label='Class 0')
    ax.scatter(class1_data[:, 0], class1_data[:, 1], color='red', marker='s', s=80, 
              edgecolor='black', linewidth=1.5, label='Class 1')
    
    # Plot means
    ax.scatter(mean_class0[0], mean_class0[1], color='blue', marker='X', s=150, 
              edgecolor='black', linewidth=1.5)
    ax.scatter(mean_class1[0], mean_class1[1], color='red', marker='X', s=150, 
              edgecolor='black', linewidth=1.5)
    
    # Plot new point
    ax.scatter(new_point[0], new_point[1], color='green', marker='*', s=200, 
              edgecolor='black', linewidth=1.5, label=f'New Point ({decision})')
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend(fontsize=10)

# Plot equal priors case
setup_comparison_plot(ax1, class0_data, class1_data, mean_class0, mean_class1, 
                     new_point, log_likelihood_ratio, 0, 
                     f"Equal Priors (P(C0)={prior_class0}, P(C1)={prior_class1})", decision)

# Plot unequal priors case
setup_comparison_plot(ax2, class0_data, class1_data, mean_class0, mean_class1, 
                     new_point, log_likelihood_ratio, log_prior_ratio, 
                     f"Unequal Priors (P(C0)={new_prior_class0}, P(C1)={new_prior_class1})", new_decision)

plt.tight_layout()
save_figure(fig5a, "step5a_comparison.png")

# Create a visualization of regions assigned to each class
fig5b, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Calculate decision regions
decision_regions_equal = log_likelihood_ratio <= 0
decision_regions_unequal = log_likelihood_ratio <= log_prior_ratio

# Equal priors
ax1.contourf(x_grid, y_grid, decision_regions_equal, levels=[0, 0.5, 1], 
           colors=['blue', 'red'], alpha=0.3)
ax1.contour(x_grid, y_grid, log_likelihood_ratio, levels=[0], colors='k', linewidths=2)
ax1.scatter(class0_data[:, 0], class0_data[:, 1], color='blue', marker='o', s=80, 
          edgecolor='black', linewidth=1.5, label='Class 0')
ax1.scatter(class1_data[:, 0], class1_data[:, 1], color='red', marker='s', s=80, 
          edgecolor='black', linewidth=1.5, label='Class 1')
ax1.scatter(new_point[0], new_point[1], color='green', marker='*', s=150, 
          edgecolor='black', linewidth=1.5, label=f'New Point ({decision})')
ax1.set_title(f"Equal Priors: Decision Regions", fontsize=14)
ax1.set_xlabel('Feature 1', fontsize=12)
ax1.set_ylabel('Feature 2', fontsize=12)
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(y_min, y_max)
ax1.legend(fontsize=10)

# Unequal priors
ax2.contourf(x_grid, y_grid, decision_regions_unequal, levels=[0, 0.5, 1], 
           colors=['blue', 'red'], alpha=0.3)
ax2.contour(x_grid, y_grid, log_likelihood_ratio, levels=[log_prior_ratio], colors='k', linewidths=2)
ax2.scatter(class0_data[:, 0], class0_data[:, 1], color='blue', marker='o', s=80, 
          edgecolor='black', linewidth=1.5, label='Class 0')
ax2.scatter(class1_data[:, 0], class1_data[:, 1], color='red', marker='s', s=80, 
          edgecolor='black', linewidth=1.5, label='Class 1')
ax2.scatter(new_point[0], new_point[1], color='green', marker='*', s=150, 
          edgecolor='black', linewidth=1.5, label=f'New Point ({new_decision})')
ax2.set_title(f"Unequal Priors: Decision Regions", fontsize=14)
ax2.set_xlabel('Feature 1', fontsize=12)
ax2.set_ylabel('Feature 2', fontsize=12)
ax2.set_xlim(x_min, x_max)
ax2.set_ylim(y_min, y_max)
ax2.legend(fontsize=10)

plt.tight_layout()
save_figure(fig5b, "step5b_decision_regions.png")

# ==============================
# STEP 6: SUMMARY
# ==============================
print_step_header(6, "SUMMARY")

print("We analyzed a 2-class problem with 2D multivariate Gaussian distributions.")
print("\nProblem statement:")
print("- Given data from two classes, we need to model them using multivariate Gaussian distributions")
print("- Then use Bayes' theorem to classify a new point under different prior probability assumptions")

print(f"\nKey results:")

print(f"\n1. Mean vectors:")
print(f"   - Class 0: {mean_class0}")
print(f"   - Class 1: {mean_class1}")

print(f"\n2. Covariance matrices:")
print(f"   - Class 0:\n{cov_class0}")
print(f"   - Class 1:\n{cov_class1}")

print(f"\n3. PDF Expressions:")
print(f"   - P(x|class 0) = {norm_const_class0:.6f} * exp(-0.5 * (x-μ₀)ᵀ Σ₀⁻¹ (x-μ₀))")
print(f"   - P(x|class 1) = {norm_const_class1:.6f} * exp(-0.5 * (x-μ₁)ᵀ Σ₁⁻¹ (x-μ₁))")

print(f"\n4. Classification of new point {new_point}:")
print(f"   - With equal priors: {decision}")
print(f"     P(class 0|x) = {posterior_class0:.6f}, P(class 1|x) = {posterior_class1:.6f}")
print(f"   - With priors P(C0)={new_prior_class0}, P(C1)={new_prior_class1}: {new_decision}")
print(f"     P(class 0|x) = {new_posterior_class0:.6f}, P(class 1|x) = {new_posterior_class1:.6f}")

print("\n5. Theoretical insights:")
print("   - Bayes' theorem provides a principled way to update beliefs based on new evidence")
print("   - Prior probabilities encode our initial belief about class membership")
print("   - Likelihoods encode how well the data fits each class model")
print("   - Posterior probabilities combine prior beliefs with evidence from the data")
print("   - The decision boundary shifts based on the prior probability ratio")

print("\nThis demonstrates how Bayesian classification combines the information from:")
print("1. The data distributions (likelihood)")
print("2. Prior knowledge/assumptions (priors)")
print("to make optimal classification decisions.") 