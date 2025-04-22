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
save_dir = os.path.join(images_dir, "L2_7_Quiz_27")
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
class0_data = np.array([[1, 3], [2, 4], [3, 5]])
class1_data = np.array([[6, 3], [7, 4], [8, 5]])
new_point = np.array([5, 4])

# Define prior probabilities
prior_class0 = 0.7
prior_class1 = 0.3

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
print(f"Mean vector = (1/3)([1,3] + [2,4] + [3,5])")

mean_class0 = np.mean(class0_data, axis=0)
print(f"Mean vector for Class 0: {mean_class0}")

print("\nFor Class 1:")
print(f"Mean vector = (1/3)([6,3] + [7,4] + [8,5])")

mean_class1 = np.mean(class1_data, axis=0)
print(f"Mean vector for Class 1: {mean_class1}")

# Calculate covariance matrices
print_substep("Calculating Covariance Matrices")
print("For Class 0:")
print("Covariance Matrix = (1/(n-1))∑(x_i - μ)(x_i - μ)ᵀ")

# Calculate deviations for Class 0
diff0 = class0_data - mean_class0
print("\nStep 1: Calculate deviations from mean for Class 0:")
for i, d in enumerate(diff0):
    print(f"  Point {i+1} - mean = {d}")

# Calculate outer products for Class 0
outer_products0 = [np.outer(d, d) for d in diff0]
print("\nStep 2: Calculate outer products (x_i - μ)(x_i - μ)ᵀ for Class 0:")
for i, op in enumerate(outer_products0):
    print(f"  Point {i+1}:\n{op}")

# Sum outer products and divide by (n-1) for Class 0
sum_outer_products0 = sum(outer_products0)
print("\nStep 3: Sum all outer products for Class 0:")
print(sum_outer_products0)

print("\nStep 4: Divide by (n-1) = 2 for Class 0:")
cov_class0 = sum_outer_products0 / (len(class0_data) - 1)
print(f"Final covariance matrix for Class 0:\n{cov_class0}")

print("\nFor Class 1:")
print("Covariance Matrix = (1/(n-1))∑(x_i - μ)(x_i - μ)ᵀ")

# Calculate deviations for Class 1
diff1 = class1_data - mean_class1
print("\nStep 1: Calculate deviations from mean for Class 1:")
for i, d in enumerate(diff1):
    print(f"  Point {i+1} - mean = {d}")

# Calculate outer products for Class 1
outer_products1 = [np.outer(d, d) for d in diff1]
print("\nStep 2: Calculate outer products (x_i - μ)(x_i - μ)ᵀ for Class 1:")
for i, op in enumerate(outer_products1):
    print(f"  Point {i+1}:\n{op}")

# Sum outer products and divide by (n-1) for Class 1
sum_outer_products1 = sum(outer_products1)
print("\nStep 3: Sum all outer products for Class 1:")
print(sum_outer_products1)

print("\nStep 4: Divide by (n-1) = 2 for Class 1:")
cov_class1 = sum_outer_products1 / (len(class1_data) - 1)
print(f"Final covariance matrix for Class 1:\n{cov_class1}")

# Add small regularization to avoid singular matrices
print("\nStep 5: Adding small regularization to avoid singular matrices")
epsilon = 1e-6
cov_class0 = cov_class0 + epsilon * np.eye(2)
cov_class1 = cov_class1 + epsilon * np.eye(2)
print(f"Regularized covariance matrix for Class 0:\n{cov_class0}")
print(f"Regularized covariance matrix for Class 1:\n{cov_class1}")

# Verify results using numpy's cov function
print("\nVerification using numpy's cov function:")
print(f"Class 0 covariance matrix (np.cov):\n{np.cov(class0_data.T)}")
print(f"Class 1 covariance matrix (np.cov):\n{np.cov(class1_data.T)}")

# Calculate correlation coefficients
print("\nCorrelation coefficients:")
corr_class0 = cov_class0[0,1] / np.sqrt(cov_class0[0,0] * cov_class0[1,1])
corr_class1 = cov_class1[0,1] / np.sqrt(cov_class1[0,0] * cov_class1[1,1])
print(f"Class 0 correlation: {corr_class0:.4f}")
print(f"Class 1 correlation: {corr_class1:.4f}")

# Calculate determinants and inverse matrices
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
# STEP 2: Multivariate Gaussian PDF and MAP Classification
# ==============================
print_step_header(2, "Multivariate Gaussian PDF and MAP Classification")

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
    print(f"  - exp(-0.5 * quadratic form): {np.exp(exponent):.8f}")
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

print(f"\nLikelihood ratio: P(x|class 0)/P(x|class 1) = {pdf_new_class0/pdf_new_class1:.8f}")
print(f"Log likelihood ratio: ln[P(x|class 0)/P(x|class 1)] = {np.log(pdf_new_class0/pdf_new_class1):.6f}")

# Visualize the PDFs
print_substep("Visualizing the PDFs")
# Create a grid of points
x_min, x_max = 0, 9
y_min, y_max = 2, 6
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

# Add visualization for Class 1 PDF
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

# ==============================
# STEP 3: MAP Classification
# ==============================
print_step_header(3, "MAP Classification")

print_substep("Introduction to MAP Estimation")
print("MAP (Maximum A Posteriori) uses Bayes' theorem to classify points:")
print("P(class|x) = [P(x|class) × P(class)] / P(x)")
print("\nwhere:")
print("- P(class|x) is the posterior probability")
print("- P(x|class) is the class-conditional density (likelihood)")
print("- P(class) is the prior probability")
print("- P(x) is the evidence (total probability)")
print("\nMAP chooses the class with the highest posterior probability.")

# Apply MAP classification with the given priors
print_substep("Applying MAP Classification with Original Priors")
print(f"Prior probabilities:")
print(f"P(class 0) = {prior_class0}")
print(f"P(class 1) = {prior_class1}")
print(f"Prior ratio: P(class 0) / P(class 1) = {prior_class0 / prior_class1:.2f}")

# Calculate posteriors
print_substep("Calculating Posterior Probabilities")
print("Using Bayes' theorem:")
print("P(class|x) ∝ P(x|class) × P(class)")
print("\nFor Class 0:")
posterior_unnorm_class0 = pdf_new_class0 * prior_class0
print(f"P(class 0|x) ∝ {pdf_new_class0:.8f} × {prior_class0} = {posterior_unnorm_class0:.8f}")

print("\nFor Class 1:")
posterior_unnorm_class1 = pdf_new_class1 * prior_class1
print(f"P(class 1|x) ∝ {pdf_new_class1:.8f} × {prior_class1} = {posterior_unnorm_class1:.8f}")

# Calculate evidence (normalization constant)
evidence = posterior_unnorm_class0 + posterior_unnorm_class1
print(f"\nEvidence (normalizing constant): {evidence:.8f}")

# Calculate normalized posteriors
posterior_class0 = posterior_unnorm_class0 / evidence
posterior_class1 = posterior_unnorm_class1 / evidence

print(f"\nNormalized posterior probabilities:")
print(f"P(class 0|x) = {posterior_class0:.8f}")
print(f"P(class 1|x) = {posterior_class1:.8f}")

# Make classification decision
print_substep("Making the Classification Decision")
if posterior_class0 > posterior_class1:
    decision = "Class 0"
    print(f"Since P(class 0|x) > P(class 1|x), classify as {decision}")
else:
    decision = "Class 1"
    print(f"Since P(class 1|x) > P(class 0|x), classify as {decision}")

# Map estimation with alternative priors
print_substep("MAP Classification with Alternative Priors")
alt_prior_class0 = 0.3
alt_prior_class1 = 0.7

print(f"Alternative prior probabilities:")
print(f"P(class 0) = {alt_prior_class0}")
print(f"P(class 1) = {alt_prior_class1}")
print(f"Prior ratio: P(class 0) / P(class 1) = {alt_prior_class0 / alt_prior_class1:.2f}")

# Calculate posteriors with alternative priors
alt_posterior_unnorm_class0 = pdf_new_class0 * alt_prior_class0
alt_posterior_unnorm_class1 = pdf_new_class1 * alt_prior_class1

alt_evidence = alt_posterior_unnorm_class0 + alt_posterior_unnorm_class1
alt_posterior_class0 = alt_posterior_unnorm_class0 / alt_evidence
alt_posterior_class1 = alt_posterior_unnorm_class1 / alt_evidence

print(f"\nAlternative posterior probabilities:")
print(f"P(class 0|x) = {alt_posterior_class0:.8f}")
print(f"P(class 1|x) = {alt_posterior_class1:.8f}")

# Make classification decision with alternative priors
if alt_posterior_class0 > alt_posterior_class1:
    alt_decision = "Class 0"
    print(f"\nSince P(class 0|x) > P(class 1|x), classify as {alt_decision}")
else:
    alt_decision = "Class 1"
    print(f"\nSince P(class 1|x) > P(class 0|x), classify as {alt_decision}")

# Compare decisions
print_substep("Comparing Decisions")
print(f"Decision with original priors: {decision}")
print(f"Decision with alternative priors: {alt_decision}")

if decision != alt_decision:
    print("\nThe classification decision changed because the prior probabilities were adjusted.")
    print("This demonstrates how prior knowledge can influence the classification outcome.")
else:
    print("\nThe classification decision remained the same despite the change in prior probabilities.")
    print("The likelihood ratio was strong enough to maintain the same classification.")

# ==============================
# STEP 4: Decision Boundaries and Precomputation Strategy
# ==============================
print_step_header(4, "Decision Boundaries and Precomputation Strategy")

print_substep("Visualizing Decision Boundaries")
print("The decision boundary is determined by where the posterior probabilities are equal:")
print("P(class 0|x) = P(class 1|x)")
print("\nUsing Bayes' theorem and simplifying, this occurs where:")
print("ln[P(x|class 0)/P(x|class 1)] = ln[P(class 1)/P(class 0)]")

# Calculate log-likelihood ratio for visualization
log_likelihood_ratio = np.log(pdf_class0 / pdf_class1)

# Calculate log-prior ratios
log_prior_ratio_orig = np.log(prior_class1 / prior_class0)
log_prior_ratio_alt = np.log(alt_prior_class1 / alt_prior_class0)

print(f"\nLog prior ratio (original): ln(P(class 1)/P(class 0)) = ln({prior_class1}/{prior_class0}) = {log_prior_ratio_orig:.6f}")
print(f"Log prior ratio (alternative): ln(P(class 1)/P(class 0)) = ln({alt_prior_class1}/{alt_prior_class0}) = {log_prior_ratio_alt:.6f}")

# Create visualization of decision boundaries
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
           edgecolor='black', linewidth=2, label=f'New Point')

# Add decision boundaries
ax4a.contour(x_grid, y_grid, log_likelihood_ratio, levels=[log_prior_ratio_orig], colors='blue', 
           linewidths=2, linestyles='solid')
ax4a.contour(x_grid, y_grid, log_likelihood_ratio, levels=[log_prior_ratio_alt], colors='red', 
           linewidths=2, linestyles='dashed')

# Add legend items for boundaries
ax4a.plot([], [], color='blue', linewidth=2, linestyle='solid', 
        label=f'Decision Boundary (P(C0)={prior_class0}, P(C1)={prior_class1})')
ax4a.plot([], [], color='red', linewidth=2, linestyle='dashed', 
        label=f'Decision Boundary (P(C0)={alt_prior_class0}, P(C1)={alt_prior_class1})')

# Add labels and legend
ax4a.set_xlabel('Feature 1', fontsize=14)
ax4a.set_ylabel('Feature 2', fontsize=14)
ax4a.set_title('Decision Boundaries with Different Priors', fontsize=16)
ax4a.legend(fontsize=10)

ax4a.set_xlim(x_min, x_max)
ax4a.set_ylim(y_min, y_max)
save_figure(fig4a, "step4a_decision_boundaries.png")

# Create a visualization of regions assigned to each class
fig4b, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Calculate decision regions
decision_regions_orig = log_likelihood_ratio >= log_prior_ratio_orig
decision_regions_alt = log_likelihood_ratio >= log_prior_ratio_alt

# Original priors
ax1.contourf(x_grid, y_grid, decision_regions_orig, levels=[0, 0.5, 1], 
           colors=['blue', 'red'], alpha=0.3)
ax1.contour(x_grid, y_grid, log_likelihood_ratio, levels=[log_prior_ratio_orig], 
          colors='k', linewidths=2)
ax1.scatter(class0_data[:, 0], class0_data[:, 1], color='blue', marker='o', s=80, 
          edgecolor='black', linewidth=1.5, label='Class 0')
ax1.scatter(class1_data[:, 0], class1_data[:, 1], color='red', marker='s', s=80, 
          edgecolor='black', linewidth=1.5, label='Class 1')
ax1.scatter(new_point[0], new_point[1], color='green', marker='*', s=150, 
          edgecolor='black', linewidth=1.5, label=f'New Point ({decision})')
ax1.set_title(f"Original Priors: P(C0)={prior_class0}, P(C1)={prior_class1}", fontsize=14)
ax1.set_xlabel('Feature 1', fontsize=12)
ax1.set_ylabel('Feature 2', fontsize=12)
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(y_min, y_max)
ax1.legend(fontsize=10)

# Alternative priors
ax2.contourf(x_grid, y_grid, decision_regions_alt, levels=[0, 0.5, 1], 
           colors=['blue', 'red'], alpha=0.3)
ax2.contour(x_grid, y_grid, log_likelihood_ratio, levels=[log_prior_ratio_alt], 
          colors='k', linewidths=2)
ax2.scatter(class0_data[:, 0], class0_data[:, 1], color='blue', marker='o', s=80, 
          edgecolor='black', linewidth=1.5, label='Class 0')
ax2.scatter(class1_data[:, 0], class1_data[:, 1], color='red', marker='s', s=80, 
          edgecolor='black', linewidth=1.5, label='Class 1')
ax2.scatter(new_point[0], new_point[1], color='green', marker='*', s=150, 
          edgecolor='black', linewidth=1.5, label=f'New Point ({alt_decision})')
ax2.set_title(f"Alternative Priors: P(C0)={alt_prior_class0}, P(C1)={alt_prior_class1}", fontsize=14)
ax2.set_xlabel('Feature 1', fontsize=12)
ax2.set_ylabel('Feature 2', fontsize=12)
ax2.set_xlim(x_min, x_max)
ax2.set_ylim(y_min, y_max)
ax2.legend(fontsize=10)

plt.tight_layout()
save_figure(fig4b, "step4b_decision_regions.png")

print_substep("Precomputation Strategy for Efficient Classification")
print("For a real-time system classifying thousands of points per second, we can:")
print("\n1. Precompute the matrix inverses Σ₀⁻¹ and Σ₁⁻¹")
print("2. Precompute the determinants |Σ₀| and |Σ₁|")
print("3. Precompute the normalizing constants for the PDFs")
print("4. Derive a simplified discriminant function for classification")

print("\nThe discriminant function for MAP classification simplifies to:")
print("Classify as Class 0 if:")
print("(x-μ₀)ᵀΣ₀⁻¹(x-μ₀) - (x-μ₁)ᵀΣ₁⁻¹(x-μ₁) < ln|Σ₁|/|Σ₀| + 2ln[P(C0)/P(C1)]")

print("\nThis can be further simplified to a quadratic discriminant function:")
print("g(x) = xᵀAx + bᵀx + c, where:")
print("- A = 0.5(Σ₁⁻¹ - Σ₀⁻¹)")
print("- b = Σ₀⁻¹μ₀ - Σ₁⁻¹μ₁")
print("- c includes all constant terms")

print("\nBy precomputing A, b, and c, we can rapidly evaluate g(x) for each new point")
print("and classify based on whether g(x) > 0 or g(x) < 0.")

# ==============================
# STEP 5: SUMMARY
# ==============================
print_step_header(5, "SUMMARY")

print("We analyzed a 2-class problem with 2D multivariate Gaussian distributions using MAP estimation.")
print("\nProblem statement:")
print("- Given data from two classes, model them using multivariate Gaussian distributions")
print("- Use MAP estimation to classify a new point under different prior probability assumptions")
print("- Design a precomputation strategy for efficient real-time classification")

print(f"\nKey results:")

print(f"\n1. Mean vectors:")
print(f"   - Class 0: {mean_class0}")
print(f"   - Class 1: {mean_class1}")

print(f"\n2. Covariance matrices:")
print(f"   - Class 0:\n{cov_class0}")
print(f"   - Class 1:\n{cov_class1}")

print(f"\n3. Classification of new point {new_point}:")
print(f"   - With priors P(C0)={prior_class0}, P(C1)={prior_class1}: {decision}")
print(f"     P(class 0|x) = {posterior_class0:.6f}, P(class 1|x) = {posterior_class1:.6f}")
print(f"   - With priors P(C0)={alt_prior_class0}, P(C1)={alt_prior_class1}: {alt_decision}")
print(f"     P(class 0|x) = {alt_posterior_class0:.6f}, P(class 1|x) = {alt_posterior_class1:.6f}")

print("\n4. Precomputation strategy for efficient classification:")
print("   - Precompute matrix inverses, determinants, and constants")
print("   - Derive quadratic discriminant function: g(x) = xᵀAx + bᵀx + c")
print("   - Classification decision reduces to evaluating sign of g(x)")

print("\n5. Effect of changing priors:")
print("   - Decision boundary shifts based on the prior ratio P(C0)/P(C1)")
print("   - Improves classification accuracy when prior matches true class distribution")
print("   - For our new point, the classification ", end="")
if decision != alt_decision:
    print(f"changed from {decision} to {alt_decision}")
else:
    print(f"remained {decision}")

print("\nThis demonstrates how MAP estimation combines the data distributions (likelihood)")
print("with prior knowledge (priors) to make optimal classification decisions.") 