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
save_dir = os.path.join(images_dir, "L2_5_Quiz_16")
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

# Define the problem parameters
class_means = [np.array([-1, -1]), np.array([2, 2])]
covariance_matrix = np.array([[2, 0.5], [0.5, 3]])
new_point = np.array([1.0, 0.2])
prior_equal = [0.5, 0.5]  # Equal priors for first part
prior_unequal = [0.8, 0.2]  # Unequal priors for additional analysis

# ==============================
# STEP 1: Understand the Problem
# ==============================
print_step_header(1, "Understanding the Problem")

print("Problem Statement:")
print("In a two-class, two-dimensional classification task, the feature vectors are generated")
print("by two normal distributions sharing the same covariance matrix")
print(f"Σ = \n{covariance_matrix}")
print("\nand the mean vectors are:")
print(f"μ₁ = {class_means[0]}")
print(f"μ₂ = {class_means[1]}")
print("\nTask: Classify the vector")
print(f"x = {new_point}")
print("according to the Bayesian classifier.")

print("\nIn Bayesian classification, we assign the new point to the class with")
print("the highest posterior probability P(class|x).")
print("To calculate this, we use Bayes' theorem:")
print("P(class|x) = [P(x|class) × P(class)] / P(x)")

# ==============================
# STEP 2: Visualize the Problem
# ==============================
print_step_header(2, "Visualizing the Problem")

def plot_data_with_ellipses(ax, means, cov, new_point=None, title=None):
    """Plot the distribution means and covariance ellipses."""
    colors = ['blue', 'red']
    labels = ['Class 1', 'Class 2']
    
    # Plot mean vectors
    for i, mean in enumerate(means):
        ax.scatter(mean[0], mean[1], color=colors[i], marker='X', s=150, 
                  edgecolor='black', linewidth=1.5, label=f'{labels[i]} Mean')
    
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
    for i, mean in enumerate(means):
        plot_ellipse(ax, mean, cov, colors[i])
    
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
    
    # Set reasonable axis limits
    ax.set_xlim(-4, 5)
    ax.set_ylim(-4, 5)
    
    return ax

# Create visualization of data and distribution parameters
print_substep("Creating Data Visualization")

fig1, ax1 = plt.subplots(figsize=(10, 8))
plot_data_with_ellipses(ax1, class_means, covariance_matrix, new_point=new_point,
                       title='Problem Visualization: Class Distributions and Point to Classify')
save_figure(fig1, "step2_problem_visualization.png")

# ==============================
# STEP 3: Calculate the PDF for Each Class
# ==============================
print_step_header(3, "Calculating the PDFs (Likelihoods)")

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

# Calculate determinant and inverse of covariance matrix
det_cov = np.linalg.det(covariance_matrix)
inv_cov = np.linalg.inv(covariance_matrix)

print_substep("Covariance Matrix Properties")
print(f"Covariance matrix:\n{covariance_matrix}")
print(f"Determinant: |Σ| = {det_cov:.6f}")
print(f"Inverse matrix: Σ⁻¹ = \n{inv_cov}")

# Calculate PDFs for the new point
print_substep("Calculating Likelihoods for the New Point")
print(f"New point to classify: {new_point}")

print("\nCalculating P(x|class 1):")
pdf_class1 = multivariate_gaussian_pdf(new_point, class_means[0], covariance_matrix)
print(f"\nP(x|class 1) = {pdf_class1:.8f}")

print("\nCalculating P(x|class 2):")
pdf_class2 = multivariate_gaussian_pdf(new_point, class_means[1], covariance_matrix)
print(f"\nP(x|class 2) = {pdf_class2:.8f}")

print(f"\nLikelihood ratio: P(x|class 2)/P(x|class 1) = {pdf_class2/pdf_class1:.6f}")

# ==============================
# STEP 4: Bayesian Classification with Equal Priors
# ==============================
print_step_header(4, "Bayesian Classification with Equal Priors")

print_substep("Setting Equal Prior Probabilities")
print(f"Prior probability for Class 1: P(class 1) = {prior_equal[0]}")
print(f"Prior probability for Class 2: P(class 2) = {prior_equal[1]}")

# Calculate evidence (total probability)
evidence_equal = pdf_class1 * prior_equal[0] + pdf_class2 * prior_equal[1]
print_substep("Calculating the Evidence")
print(f"P(x) = P(x|class 1)×P(class 1) + P(x|class 2)×P(class 2)")
print(f"P(x) = {pdf_class1:.8f}×{prior_equal[0]} + {pdf_class2:.8f}×{prior_equal[1]}")
print(f"P(x) = {pdf_class1 * prior_equal[0]:.8f} + {pdf_class2 * prior_equal[1]:.8f} = {evidence_equal:.8f}")

# Calculate posterior probabilities
posterior_class1_equal = (pdf_class1 * prior_equal[0]) / evidence_equal
posterior_class2_equal = (pdf_class2 * prior_equal[1]) / evidence_equal

print_substep("Calculating Posterior Probabilities")
print("For Class 1:")
print(f"P(class 1|x) = [P(x|class 1)×P(class 1)] / P(x)")
print(f"P(class 1|x) = [{pdf_class1:.8f}×{prior_equal[0]}] / {evidence_equal:.8f}")
print(f"P(class 1|x) = {pdf_class1 * prior_equal[0]:.8f} / {evidence_equal:.8f} = {posterior_class1_equal:.8f}")

print("\nFor Class 2:")
print(f"P(class 2|x) = [P(x|class 2)×P(class 2)] / P(x)")
print(f"P(class 2|x) = [{pdf_class2:.8f}×{prior_equal[1]}] / {evidence_equal:.8f}")
print(f"P(class 2|x) = {pdf_class2 * prior_equal[1]:.8f} / {evidence_equal:.8f} = {posterior_class2_equal:.8f}")

# Make classification decision
print_substep("Making the Classification Decision")
print(f"P(class 1|x) = {posterior_class1_equal:.8f}")
print(f"P(class 2|x) = {posterior_class2_equal:.8f}")

if posterior_class1_equal > posterior_class2_equal:
    equal_decision = "Class 1"
    print(f"\nSince P(class 1|x) > P(class 2|x), classify as {equal_decision}")
else:
    equal_decision = "Class 2"
    print(f"\nSince P(class 2|x) > P(class 1|x), classify as {equal_decision}")

# ==============================
# STEP 5: Visualize the Decision Boundary
# ==============================
print_step_header(5, "Visualizing the Decision Boundary")

# Create a grid of points
x_min, x_max = -4, 5
y_min, y_max = -4, 5
x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
pos = np.dstack((x_grid, y_grid))

# Calculate PDFs for each class over the grid
pdf_grid_class1 = multivariate_normal.pdf(pos, mean=class_means[0], cov=covariance_matrix)
pdf_grid_class2 = multivariate_normal.pdf(pos, mean=class_means[1], cov=covariance_matrix)

# Calculate log-likelihood ratio
log_likelihood_ratio = np.log(pdf_grid_class2 / pdf_grid_class1)

# For equal priors, the decision boundary is where log-likelihood ratio = 0
print("For equal priors P(class 1) = P(class 2) = 0.5,")
print("the decision boundary is where: ln[P(x|class 2)/P(x|class 1)] = 0")

# Visualize the decision boundary
fig5, ax5 = plt.subplots(figsize=(10, 8))

# Plot data
ax5.scatter(class_means[0][0], class_means[0][1], color='blue', marker='X', s=150, 
          edgecolor='black', linewidth=1.5, label='Class 1 Mean')
ax5.scatter(class_means[1][0], class_means[1][1], color='red', marker='X', s=150, 
          edgecolor='black', linewidth=1.5, label='Class 2 Mean')
ax5.scatter(new_point[0], new_point[1], color='green', marker='*', s=200, 
          edgecolor='black', linewidth=1.5, label=f'New Point ({equal_decision})')

# Plot decision boundary
ax5.contour(x_grid, y_grid, log_likelihood_ratio, levels=[0], colors='k', linewidths=2)

# Calculate and plot decision regions
decision_regions = log_likelihood_ratio <= 0
ax5.contourf(x_grid, y_grid, decision_regions, levels=[0, 0.5, 1], 
           colors=['blue', 'red'], alpha=0.2)

ax5.set_xlabel('Feature 1', fontsize=12)
ax5.set_ylabel('Feature 2', fontsize=12)
ax5.set_title('Decision Boundary and Regions with Equal Priors', fontsize=14)
ax5.legend()
ax5.set_xlim(x_min, x_max)
ax5.set_ylim(y_min, y_max)

save_figure(fig5, "step5_decision_boundary.png")

# ==============================
# STEP 6: Additional Analysis with Unequal Priors
# ==============================
print_step_header(6, "Additional Analysis with Unequal Priors")

print_substep("Setting Unequal Prior Probabilities")
print(f"Prior probability for Class 1: P(class 1) = {prior_unequal[0]}")
print(f"Prior probability for Class 2: P(class 2) = {prior_unequal[1]}")
print(f"Prior ratio: P(class 1) / P(class 2) = {prior_unequal[0] / prior_unequal[1]:.2f}")

# Calculate evidence with unequal priors
evidence_unequal = pdf_class1 * prior_unequal[0] + pdf_class2 * prior_unequal[1]
print_substep("Recalculating Evidence and Posteriors")
print(f"P(x) = {pdf_class1:.8f}×{prior_unequal[0]} + {pdf_class2:.8f}×{prior_unequal[1]}")
print(f"P(x) = {pdf_class1 * prior_unequal[0]:.8f} + {pdf_class2 * prior_unequal[1]:.8f} = {evidence_unequal:.8f}")

# Calculate posterior probabilities with unequal priors
posterior_class1_unequal = (pdf_class1 * prior_unequal[0]) / evidence_unequal
posterior_class2_unequal = (pdf_class2 * prior_unequal[1]) / evidence_unequal

print("\nFor Class 1:")
print(f"P(class 1|x) = {posterior_class1_unequal:.8f}")

print("\nFor Class 2:")
print(f"P(class 2|x) = {posterior_class2_unequal:.8f}")

# Make classification decision with unequal priors
if posterior_class1_unequal > posterior_class2_unequal:
    unequal_decision = "Class 1"
    print(f"\nWith unequal priors, classify as {unequal_decision}")
else:
    unequal_decision = "Class 2"
    print(f"\nWith unequal priors, classify as {unequal_decision}")

# Adjusted decision boundary with unequal priors
log_prior_ratio = np.log(prior_unequal[1] / prior_unequal[0])
print(f"\nLog prior ratio: ln(P(class 2)/P(class 1)) = ln({prior_unequal[1]}/{prior_unequal[0]}) = {log_prior_ratio:.6f}")
print(f"The decision boundary shifts to where ln[P(x|class 2)/P(x|class 1)] = {log_prior_ratio:.6f}")

# Visualize both decision boundaries
fig6, ax6 = plt.subplots(figsize=(10, 8))

# Plot data
ax6.scatter(class_means[0][0], class_means[0][1], color='blue', marker='X', s=150, 
          edgecolor='black', linewidth=1.5, label='Class 1 Mean')
ax6.scatter(class_means[1][0], class_means[1][1], color='red', marker='X', s=150, 
          edgecolor='black', linewidth=1.5, label='Class 2 Mean')
ax6.scatter(new_point[0], new_point[1], color='green', marker='*', s=200, 
          edgecolor='black', linewidth=1.5, label=f'New Point')

# Plot both decision boundaries
ax6.contour(x_grid, y_grid, log_likelihood_ratio, levels=[0], colors='k', linewidths=2, 
          linestyles='solid')
ax6.contour(x_grid, y_grid, log_likelihood_ratio, levels=[log_prior_ratio], colors='k', 
          linewidths=2, linestyles='dashed')

# Add legend items for boundaries
ax6.plot([], [], color='k', linewidth=2, linestyle='solid', 
       label='Equal Priors Boundary P(C1)=P(C2)=0.5')
ax6.plot([], [], color='k', linewidth=2, linestyle='dashed', 
       label=f'Unequal Priors Boundary P(C1)={prior_unequal[0]}, P(C2)={prior_unequal[1]}')

# Calculate and plot decision regions for unequal priors
decision_regions_unequal = log_likelihood_ratio <= log_prior_ratio
ax6.contourf(x_grid, y_grid, decision_regions_unequal, levels=[0, 0.5, 1], 
           colors=['blue', 'red'], alpha=0.2)

ax6.set_xlabel('Feature 1', fontsize=12)
ax6.set_ylabel('Feature 2', fontsize=12)
ax6.set_title('Decision Boundaries Comparison: Equal vs. Unequal Priors', fontsize=14)
ax6.legend(fontsize=10)
ax6.set_xlim(x_min, x_max)
ax6.set_ylim(y_min, y_max)

save_figure(fig6, "step6_boundaries_comparison.png")

# ==============================
# STEP 7: Visualize PDFs and Decision Process
# ==============================
print_step_header(7, "Visualizing PDFs and Decision Process")

# Create 3D visualization of the PDFs
fig7, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), subplot_kw={'projection': '3d'})

# 3D plot for Class 1
surf1 = ax1.plot_surface(x_grid, y_grid, pdf_grid_class1, cmap='Blues', alpha=0.8)
ax1.set_title('PDF for Class 1', fontsize=14)
ax1.set_xlabel('Feature 1', fontsize=12)
ax1.set_ylabel('Feature 2', fontsize=12)
ax1.set_zlabel('Probability Density', fontsize=12)
ax1.view_init(elev=30, azim=45)

# 3D plot for Class 2
surf2 = ax2.plot_surface(x_grid, y_grid, pdf_grid_class2, cmap='Reds', alpha=0.8)
ax2.set_title('PDF for Class 2', fontsize=14)
ax2.set_xlabel('Feature 1', fontsize=12)
ax2.set_ylabel('Feature 2', fontsize=12)
ax2.set_zlabel('Probability Density', fontsize=12)
ax2.view_init(elev=30, azim=45)

# Mark the new point on both surfaces
ax1.scatter(new_point[0], new_point[1], multivariate_normal.pdf(new_point, mean=class_means[0], cov=covariance_matrix), 
          color='green', marker='*', s=150, edgecolor='black')
ax2.scatter(new_point[0], new_point[1], multivariate_normal.pdf(new_point, mean=class_means[1], cov=covariance_matrix), 
          color='green', marker='*', s=150, edgecolor='black')

plt.tight_layout()
save_figure(fig7, "step7_3d_pdfs.png")

# ==============================
# STEP 8: Summarize the Solution
# ==============================
print_step_header(8, "Summary of the Solution")

print("Problem:")
print("- Two-class, two-dimensional classification problem")
print("- Both classes follow multivariate Gaussian distributions with same covariance")
print("- Need to classify a new point using Bayesian classification")

print("\nKey components of the Bayesian classification:")
print("1. Prior probabilities: P(class 1) and P(class 2)")
print("2. Likelihoods: P(x|class 1) and P(x|class 2)")
print("3. Evidence: P(x) = P(x|class 1)×P(class 1) + P(x|class 2)×P(class 2)")
print("4. Posterior probabilities: P(class|x) = P(x|class)×P(class)/P(x)")

print("\nWith equal priors:")
print(f"- P(class 1|x) = {posterior_class1_equal:.8f}")
print(f"- P(class 2|x) = {posterior_class2_equal:.8f}")
print(f"- Classification: {equal_decision}")

print("\nWith unequal priors (P(C1)=0.8, P(C2)=0.2):")
print(f"- P(class 1|x) = {posterior_class1_unequal:.8f}")
print(f"- P(class 2|x) = {posterior_class2_unequal:.8f}")
print(f"- Classification: {unequal_decision}")

print("\nInsights:")
print("- The decision boundary is a line in 2D space when the covariance matrices are equal")
print("- The boundary shifts based on the prior probabilities")
print("- A higher prior for one class makes classification in favor of that class more likely")
print("- Bayesian classification provides a principled way to combine prior knowledge with observed data") 