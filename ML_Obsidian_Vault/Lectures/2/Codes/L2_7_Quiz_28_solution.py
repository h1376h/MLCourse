import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_7_Quiz_28")
os.makedirs(save_dir, exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')

# Helper functions
def save_figure(fig, filename):
    """Save figure to the specified directory."""
    file_path = os.path.join(save_dir, filename)
    fig.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {file_path}")
    plt.close(fig)

# Define the data
class0_data = np.array([[1, 2], [2, 4], [3, 5]])
class1_data = np.array([[6, 3], [7, 2], [8, 4]])
new_point = np.array([5, 3])

print("="*80)
print("STEP 1: Calculating Mean Vectors and Covariance Matrices")
print("="*80)

# Calculate mean vectors
mean_class0 = np.mean(class0_data, axis=0)
mean_class1 = np.mean(class1_data, axis=0)

print(f"Mean vector for Class 0: {mean_class0}")
print(f"Mean vector for Class 1: {mean_class1}")

# Calculate covariance matrices with n-1 denominator for unbiased estimation
cov_class0 = np.cov(class0_data.T)
cov_class1 = np.cov(class1_data.T)

print(f"\nCovariance matrix for Class 0:\n{cov_class0}")
print(f"\nCovariance matrix for Class 1:\n{cov_class1}")

# Verify non-singularity by computing determinants
det_class0 = np.linalg.det(cov_class0)
det_class1 = np.linalg.det(cov_class1)

print(f"\nDeterminant for Class 0 covariance: {det_class0:.6f}")
print(f"\nDeterminant for Class 1 covariance: {det_class1:.6f}")

if det_class0 > 0 and det_class1 > 0:
    print("Both covariance matrices are non-singular (positive determinants).")
else:
    print("Warning: At least one covariance matrix is singular.")

# Calculate inverse matrices
inv_cov_class0 = np.linalg.inv(cov_class0)
inv_cov_class1 = np.linalg.inv(cov_class1)

print(f"\nInverse covariance matrix for Class 0:\n{inv_cov_class0}")
print(f"\nInverse covariance matrix for Class 1:\n{inv_cov_class1}")

# Visualize data with mean vectors and covariance ellipses
fig1, ax1 = plt.subplots(figsize=(10, 8))

# Plot data points
ax1.scatter(class0_data[:, 0], class0_data[:, 1], color='blue', 
           s=100, label='Class 0', edgecolor='black')
ax1.scatter(class1_data[:, 0], class1_data[:, 1], color='red', 
           s=100, label='Class 1', edgecolor='black')
ax1.scatter(new_point[0], new_point[1], color='green', marker='*', 
           s=200, label='New Point', edgecolor='black')

# Plot mean vectors
ax1.scatter(mean_class0[0], mean_class0[1], color='blue', marker='X', 
           s=150, label='Class 0 Mean', edgecolor='black')
ax1.scatter(mean_class1[0], mean_class1[1], color='red', marker='X', 
           s=150, label='Class 1 Mean', edgecolor='black')

# Function to create confidence ellipses
def plot_ellipse(ax, mean, cov, color, alpha=0.3):
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    width, height = 2 * np.sqrt(5.991 * eigenvals)  # 95% confidence
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, 
                     facecolor=color, alpha=alpha, edgecolor=color, linewidth=2)
    ax.add_patch(ellipse)

# Add ellipses for both classes
plot_ellipse(ax1, mean_class0, cov_class0, 'blue')
plot_ellipse(ax1, mean_class1, cov_class1, 'red')

ax1.set_xlabel('Feature 1', fontsize=12)
ax1.set_ylabel('Feature 2', fontsize=12)
ax1.set_title('Data Visualization with Mean Vectors and Covariance Ellipses', fontsize=14)
ax1.legend()

save_figure(fig1, "step1_data_visualization.png")

print("\n" + "="*80)
print("STEP 2: Multivariate Gaussian PDFs and Likelihood Calculation")
print("="*80)

# Define the multivariate Gaussian PDF function
def multivariate_gaussian_pdf(x, mean, cov):
    """Calculate the PDF value for a multivariate Gaussian distribution."""
    n = len(mean)
    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)
    diff = x - mean
    
    exponent = -0.5 * diff.T @ inv_cov @ diff
    normalizer = 1 / ((2 * np.pi) ** (n/2) * np.sqrt(det_cov))
    pdf_value = normalizer * np.exp(exponent)
    
    print(f"  - Difference vector (x - μ): {diff}")
    print(f"  - Quadratic form (x - μ)ᵀ Σ⁻¹ (x - μ): {diff.T @ inv_cov @ diff:.6f}")
    print(f"  - Final PDF value: {pdf_value:.8f}")
    
    return pdf_value

# Calculate PDF values for the new point
print("Calculating PDF for new point in Class 0:")
pdf_new_class0 = multivariate_gaussian_pdf(new_point, mean_class0, cov_class0)

print("\nCalculating PDF for new point in Class 1:")
pdf_new_class1 = multivariate_gaussian_pdf(new_point, mean_class1, cov_class1)

print(f"\nLikelihood ratio P(x|class 0) / P(x|class 1): {pdf_new_class0/pdf_new_class1:.6f}")

# Create a grid for PDF visualization
x_min, x_max = 0, 9
y_min, y_max = 1, 6
x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 100), 
                            np.linspace(y_min, y_max, 100))
pos = np.dstack((x_grid, y_grid))

# Calculate PDFs for each class
pdf_class0 = multivariate_normal.pdf(pos, mean=mean_class0, cov=cov_class0)
pdf_class1 = multivariate_normal.pdf(pos, mean=mean_class1, cov=cov_class1)

# Plot PDFs for both classes
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Class 0 PDF
contour0 = ax1.contourf(x_grid, y_grid, pdf_class0, levels=20, cmap='Blues')
ax1.scatter(class0_data[:, 0], class0_data[:, 1], color='blue', s=80, edgecolor='black')
ax1.scatter(mean_class0[0], mean_class0[1], color='blue', marker='X', s=150, edgecolor='black')
ax1.scatter(new_point[0], new_point[1], color='green', marker='*', s=200, edgecolor='black')
ax1.set_title('PDF for Class 0', fontsize=14)
ax1.set_xlabel('Feature 1', fontsize=12)
ax1.set_ylabel('Feature 2', fontsize=12)
plt.colorbar(contour0, ax=ax1, label='Probability Density')

# Class 1 PDF
contour1 = ax2.contourf(x_grid, y_grid, pdf_class1, levels=20, cmap='Reds')
ax2.scatter(class1_data[:, 0], class1_data[:, 1], color='red', s=80, edgecolor='black')
ax2.scatter(mean_class1[0], mean_class1[1], color='red', marker='X', s=150, edgecolor='black')
ax2.scatter(new_point[0], new_point[1], color='green', marker='*', s=200, edgecolor='black')
ax2.set_title('PDF for Class 1', fontsize=14)
ax2.set_xlabel('Feature 1', fontsize=12)
ax2.set_ylabel('Feature 2', fontsize=12)
plt.colorbar(contour1, ax=ax2, label='Probability Density')

plt.tight_layout()
save_figure(fig2, "step2_pdfs.png")

print("\n" + "="*80)
print("STEP 3: MAP Classification with Prior Probabilities")
print("="*80)

# Define prior probabilities
prior_class0 = 0.6
prior_class1 = 0.4

print(f"Prior probability for Class 0: P(class 0) = {prior_class0}")
print(f"Prior probability for Class 1: P(class 1) = {prior_class1}")

# Calculate the evidence (total probability)
evidence = pdf_new_class0 * prior_class0 + pdf_new_class1 * prior_class1
print(f"Evidence P(x): {evidence:.8f}")

# Apply Bayes' theorem
posterior_class0 = (pdf_new_class0 * prior_class0) / evidence
posterior_class1 = (pdf_new_class1 * prior_class1) / evidence

print(f"Posterior P(class 0|x): {posterior_class0:.8f}")
print(f"Posterior P(class 1|x): {posterior_class1:.8f}")

# Make classification decision
if posterior_class0 > posterior_class1:
    decision = "Class 0"
    print(f"\nSince P(class 0|x) > P(class 1|x), classify as {decision}")
else:
    decision = "Class 1"
    print(f"\nSince P(class 1|x) > P(class 0|x), classify as {decision}")

# Calculate log likelihood ratio for visualization
log_likelihood_ratio = np.log(pdf_class1 / pdf_class0)
log_prior_ratio = np.log(prior_class0 / prior_class1)

# Plot decision boundary
fig3, ax3 = plt.subplots(figsize=(10, 8))

# Plot the data points
ax3.scatter(class0_data[:, 0], class0_data[:, 1], color='blue', s=100, 
          edgecolor='black', label='Class 0')
ax3.scatter(class1_data[:, 0], class1_data[:, 1], color='red', s=100, 
          edgecolor='black', label='Class 1')
ax3.scatter(new_point[0], new_point[1], color='green', marker='*', s=200, 
          edgecolor='black', label=f'New Point ({decision})')

# Plot the decision boundary where log likelihood ratio = log prior ratio
ax3.contour(x_grid, y_grid, log_likelihood_ratio, 
           levels=[log_prior_ratio], colors='k', linewidths=2)

# Plot regions
decision_regions = log_likelihood_ratio <= log_prior_ratio
ax3.contourf(x_grid, y_grid, decision_regions, levels=[0, 0.5, 1], 
           colors=['blue', 'red'], alpha=0.3)

ax3.set_xlabel('Feature 1', fontsize=14)
ax3.set_ylabel('Feature 2', fontsize=14)
ax3.set_title('Decision Boundary with Priors (0.6, 0.4)', fontsize=16)
ax3.legend(fontsize=12)

ax3.set_xlim(x_min, x_max)
ax3.set_ylim(y_min, y_max)
save_figure(fig3, "step3_decision_boundary.png")

print("\n" + "="*80)
print("STEP 4: Changed Prior Probabilities")
print("="*80)

# Define new prior probabilities
new_prior_class0 = 0.4
new_prior_class1 = 0.6

print(f"New prior probability for Class 0: P(class 0) = {new_prior_class0}")
print(f"New prior probability for Class 1: P(class 1) = {new_prior_class1}")

# Recalculate with new priors
new_evidence = pdf_new_class0 * new_prior_class0 + pdf_new_class1 * new_prior_class1
new_posterior_class0 = (pdf_new_class0 * new_prior_class0) / new_evidence
new_posterior_class1 = (pdf_new_class1 * new_prior_class1) / new_evidence

print(f"New posterior P(class 0|x): {new_posterior_class0:.8f}")
print(f"New posterior P(class 1|x): {new_posterior_class1:.8f}")

# Make new classification decision
if new_posterior_class0 > new_posterior_class1:
    new_decision = "Class 0"
    print(f"\nSince P(class 0|x) > P(class 1|x), classify as {new_decision}")
else:
    new_decision = "Class 1"
    print(f"\nSince P(class 1|x) > P(class 0|x), classify as {new_decision}")

# Compare with previous decision
print(f"\nDecision with priors (0.6, 0.4): {decision}")
print(f"Decision with priors (0.4, 0.6): {new_decision}")

# Calculate the new log prior ratio
new_log_prior_ratio = np.log(new_prior_class0 / new_prior_class1)

# Visualize both decision boundaries
fig4, ax4 = plt.subplots(figsize=(10, 8))

# Plot data
ax4.scatter(class0_data[:, 0], class0_data[:, 1], color='blue', s=100, 
          edgecolor='black', label='Class 0')
ax4.scatter(class1_data[:, 0], class1_data[:, 1], color='red', s=100, 
          edgecolor='black', label='Class 1')
ax4.scatter(new_point[0], new_point[1], color='green', marker='*', s=200, 
          edgecolor='black', label=f'New Point')

# Add both decision boundaries
ax4.contour(x_grid, y_grid, log_likelihood_ratio, levels=[log_prior_ratio], 
          colors='blue', linewidths=2, linestyles='solid')
ax4.contour(x_grid, y_grid, log_likelihood_ratio, levels=[new_log_prior_ratio], 
          colors='red', linewidths=2, linestyles='dashed')

# Add legend items for boundaries
ax4.plot([], [], color='blue', linewidth=2, linestyle='solid', 
        label='Boundary (0.6, 0.4)')
ax4.plot([], [], color='red', linewidth=2, linestyle='dashed', 
        label='Boundary (0.4, 0.6)')

# Add labels and legend
ax4.set_xlabel('Feature 1', fontsize=14)
ax4.set_ylabel('Feature 2', fontsize=14)
ax4.set_title('Decision Boundaries with Different Priors', fontsize=16)
ax4.legend(fontsize=12)

ax4.set_xlim(x_min, x_max)
ax4.set_ylim(y_min, y_max)
save_figure(fig4, "step4_boundaries_comparison.png")

print("\n" + "="*80)
print("STEP 5: Quadratic Discriminant Function")
print("="*80)

# Derive the quadratic discriminant function
print("The quadratic discriminant function is:")
print("g(x) = ln[P(x|class 1)] + ln[P(class 1)] - ln[P(x|class 0)] - ln[P(class 0)]")
print("    = -0.5(x-μ₁)ᵀΣ₁⁻¹(x-μ₁) + ln[(2π)^(-d/2)|Σ₁|^(-1/2)] + ln[P(class 1)]")
print("      +0.5(x-μ₀)ᵀΣ₀⁻¹(x-μ₀) - ln[(2π)^(-d/2)|Σ₀|^(-1/2)] - ln[P(class 0)]")
print("    = -0.5(x-μ₁)ᵀΣ₁⁻¹(x-μ₁) + 0.5(x-μ₀)ᵀΣ₀⁻¹(x-μ₀) - 0.5ln|Σ₁| + 0.5ln|Σ₀| + ln[P(class 1)/P(class 0)]")

# Extract the quadratic, linear, and constant terms
print("\nThe quadratic discriminant function can be written as:")
print("g(x) = x^T Q x + L^T x + c")

print("\nwhere:")
print("Q = -0.5Σ₁⁻¹ + 0.5Σ₀⁻¹")
print("L = Σ₁⁻¹μ₁ - Σ₀⁻¹μ₀")
print("c = -0.5μ₁ᵀΣ₁⁻¹μ₁ + 0.5μ₀ᵀΣ₀⁻¹μ₀ - 0.5ln|Σ₁| + 0.5ln|Σ₀| + ln[P(class 1)/P(class 0)]")

# Calculate the quadratic discriminant function for visualization
Q = -0.5 * inv_cov_class1 + 0.5 * inv_cov_class0
L = inv_cov_class1 @ mean_class1 - inv_cov_class0 @ mean_class0
c1 = -0.5 * mean_class1.T @ inv_cov_class1 @ mean_class1
c2 = 0.5 * mean_class0.T @ inv_cov_class0 @ mean_class0
c3 = -0.5 * np.log(det_class1) + 0.5 * np.log(det_class0)
c4 = np.log(prior_class1 / prior_class0)
c = c1 + c2 + c3 + c4

print(f"\nFor our problem:")
print(f"Q = \n{Q}")
print(f"L = {L}")
print(f"c = {c}")

# Compute discriminant for each point in grid
def discriminant(x, Q, L, c):
    return x.T @ Q @ x + L.T @ x + c

discriminant_values = np.zeros_like(x_grid)
for i in range(x_grid.shape[0]):
    for j in range(x_grid.shape[1]):
        point = np.array([x_grid[i, j], y_grid[i, j]])
        discriminant_values[i, j] = discriminant(point, Q, L, c)

# Plot the quadratic discriminant function
fig5, ax5 = plt.subplots(figsize=(10, 8))

# Plot data points
ax5.scatter(class0_data[:, 0], class0_data[:, 1], color='blue', s=100, 
          edgecolor='black', label='Class 0')
ax5.scatter(class1_data[:, 0], class1_data[:, 1], color='red', s=100, 
          edgecolor='black', label='Class 1')
ax5.scatter(new_point[0], new_point[1], color='green', marker='*', s=200, 
          edgecolor='black', label=f'New Point')

# Plot the discriminant function contours
contour = ax5.contourf(x_grid, y_grid, discriminant_values, levels=20, 
                     cmap='coolwarm', alpha=0.5)
ax5.contour(x_grid, y_grid, discriminant_values, levels=[0], colors='k', 
          linewidths=3, linestyles='solid')

# Colorbar and formatting
plt.colorbar(contour, ax=ax5, label='Discriminant Function Value')
ax5.set_xlabel('Feature 1', fontsize=14)
ax5.set_ylabel('Feature 2', fontsize=14)
ax5.set_title('Quadratic Discriminant Function', fontsize=16)
ax5.legend(fontsize=12)
ax5.set_xlim(x_min, x_max)
ax5.set_ylim(y_min, y_max)

save_figure(fig5, "step5_quadratic_discriminant.png")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("We solved a classification problem with two classes, each represented by")
print("three 2D data points. The key results are:")

print(f"\n1. Mean vectors:")
print(f"   - Class 0: {mean_class0}")
print(f"   - Class 1: {mean_class1}")

print(f"\n2. Covariance matrices:")
print(f"   - Class 0:\n{cov_class0}")
print(f"   - Class 1:\n{cov_class1}")
print(f"   Both covariance matrices are non-singular with determinants:")
print(f"   - |Σ₀| = {det_class0:.6f}")
print(f"   - |Σ₁| = {det_class1:.6f}")

print(f"\n3. Classification of new point {new_point}:")
print(f"   - With priors (0.6, 0.4): {decision}")
print(f"     P(class 0|x) = {posterior_class0:.6f}, P(class 1|x) = {posterior_class1:.6f}")
print(f"   - With priors (0.4, 0.6): {new_decision}")
print(f"     P(class 0|x) = {new_posterior_class0:.6f}, P(class 1|x) = {new_posterior_class1:.6f}")

print("\n4. Non-singular covariance matrices enable the use of the quadratic discriminant function,")
print("   resulting in a curved decision boundary between the classes.") 